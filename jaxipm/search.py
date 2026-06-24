"""file containing the REGULAR iteration type, covers BT, SOC, SFR, TS, WD"""
import jax
import jax.numpy as jnp
import numpy as np
from pathlib import Path
from jaxipm.structures import Iterate, WatchdogState, OptimizationState
from jaxipm.utils.eqx_utils import filter_tree_at_select, filter_select, filter_while_loop
from jaxipm.quantities import calc_values_pre_mu, calc_values_post_mu
from jaxipm.barrier import calc_updated_mu
from jaxipm.initialization import (
    post_initialize_line_search,
    initialize_resto_args, initialize_iterate_resto,
    initialize_inertia_correction_state,
    initialize_adaptive_mu_filter_state,
    initialize_line_search_filter_state,
    initialize_iterate_flags_state,
    initialize_line_search_state,
    initialize_watchdog,
    compute_scaling,
    push_to_interior
)
from equinox.internal import ω
import equinox as eqx
import jaxipm.utils.sparse_utils as spu
import jax.experimental.sparse as jsparse

# Debug logging for backtracking line search
_LS_DEBUG_DIR = Path("tmp/batch_solve_basic_nmpc/ls_debug")
_LS_DEBUG_FILE = _LS_DEBUG_DIR / "alpha_values.csv"
_LS_DEBUG_INITIALIZED = False
_LS_DEBUG_CALL_COUNT = 0
_LS_BATCH_SIZE = 1  # Must match batch size in solver

def _init_ls_debug():
    """Initialize the debug output directory and file."""
    global _LS_DEBUG_INITIALIZED, _LS_DEBUG_CALL_COUNT
    # Reset if file was deleted or on first call
    if not _LS_DEBUG_INITIALIZED or not _LS_DEBUG_FILE.exists():
        _LS_DEBUG_DIR.mkdir(parents=True, exist_ok=True)
        with open(_LS_DEBUG_FILE, 'w') as f:
            f.write("outer_iter,n_steps,alpha_pr,alpha_min,accept,in_resto\n")
        _LS_DEBUG_INITIALIZED = True
        _LS_DEBUG_CALL_COUNT = 0

def _log_ls_alpha_callback(outer_iter, n_steps, alpha_pr, alpha_min, accept, in_resto):
    """Callback to log alpha values from the backtracking line search.
    Only logs first batch element using counter."""
    global _LS_DEBUG_CALL_COUNT
    _init_ls_debug()

    # Only log every BATCH_SIZE calls (first element of each batch)
    _LS_DEBUG_CALL_COUNT += 1
    if (_LS_DEBUG_CALL_COUNT - 1) % _LS_BATCH_SIZE != 0:
        return

    # Convert to numpy scalars
    outer_iter = int(np.asarray(outer_iter).flatten()[0])
    n_steps = int(np.asarray(n_steps).flatten()[0])
    alpha_pr = float(np.asarray(alpha_pr).flatten()[0])
    alpha_min = float(np.asarray(alpha_min).flatten()[0])
    accept = int(np.asarray(accept).flatten()[0])
    in_resto = int(np.asarray(in_resto).flatten()[0])

    # Append to CSV - one row per unique line search step
    with open(_LS_DEBUG_FILE, 'a') as f:
        f.write(f"{outer_iter},{n_steps},{alpha_pr:.15e},{alpha_min:.15e},{accept},{in_resto}\n")

def log_ls_alpha(outer_iter, n_steps, alpha_pr, alpha_min, accept, in_resto):
    """Log alpha values using debug.callback (works in jitted/vmapped code)."""
    jax.debug.callback(
        _log_ls_alpha_callback,
        outer_iter, n_steps, alpha_pr, alpha_min, accept, in_resto
    )

# Restoration convergence check - determines if we should exit restoration and return to regular
# Based on IPOPT's IpRestoConvCheck.cpp and IpRestoFilterConvCheck.cpp
def check_resto_convergence(
    it, reg_c, reg_d, reg_f, saved_ls, saved_mu, saved_orig_inf_pr, cp, first_resto_iter,
    resto_converged, resto_tol
):
    """
    Check if we should exit restoration phase and return to regular optimization.

    Args:
        it: Current iterate (restoration iterate, x[:cp.nx] is original x)
        reg_c: Original problem's equality constraints at current x
        reg_d: Original problem's inequality constraints at current x
        reg_f: Original problem's objective at current x
        saved_ls: Saved line search state from resto entry (contains original filter)
        saved_mu: Saved barrier parameter from resto entry
        saved_orig_inf_pr: Original problem's inf-norm primal infeasibility at resto entry
        cp: Common problem structure
        first_resto_iter: Boolean, True if this is the first restoration iteration
        resto_converged: Boolean, True if restoration problem itself has converged
        resto_tol: Restoration problem's convergence tolerance (can be tightened)

    Returns:
        Tuple of:
        - should_exit_resto: bool, True if restoration converged normally
        - is_locally_infeasible: bool, True if problem is locally infeasible
        - is_feasible_filter_rejects: bool, True if feasible but filter rejects
        - should_tighten_tol: bool, True if should tighten resto tolerance
        - is_acceptable_convergence: bool, True for square problem acceptable convergence
    """
    # Extract parameters
    kappa_resto = cp.p.get("required_infeasibility_reduction", 0.9) if hasattr(cp.p, 'get') else cp.p["required_infeasibility_reduction"] if "required_infeasibility_reduction" in cp.p._dict else 0.9
    orig_tol = cp.p["tol"]
    resto_tol = resto_tol.squeeze()  # Use tracked resto_tol instead of p["tol"]
    constr_viol_tol = cp.p["constr_viol_tol"]

    # Compute original problem's primal infeasibility at trial point (inf-norm)
    # This is IPOPT's orig_trial_inf_pr = trial_primal_infeasibility(NORM_MAX)
    reg_dms = reg_d - it.s[:cp.nyd]
    orig_trial_inf_pr = jnp.maximum(
        jnp.linalg.norm(reg_c, ord=jnp.inf),
        jnp.linalg.norm(reg_dms, ord=jnp.inf)
    )

    # Original problem's infeasibility at resto entry (inf-norm)
    # This is IPOPT's orig_curr_inf_pr = curr_primal_infeasibility(NORM_MAX)
    orig_curr_inf_pr = saved_orig_inf_pr.squeeze()

    # Compute threshold for required infeasibility reduction
    # IPOPT: orig_inf_pr_max = Max(kappa_resto * orig_curr_inf_pr, Min(tol, constr_viol_tol))
    orig_inf_pr_max = jnp.maximum(
        kappa_resto * orig_curr_inf_pr,
        jnp.minimum(orig_tol, constr_viol_tol)
    )
    # Special case: if kappa_resto == 0, orig_inf_pr_max = 0
    orig_inf_pr_max = jnp.where(kappa_resto == 0.0, 0.0, orig_inf_pr_max)

    # Check 1: First resto iteration - always continue (take at least one step)
    # Check 2: Square problem - if original problem has n_vars == n_eq_constraints and feasible, converge
    is_square_problem = (cp.nx == cp.nyc)
    square_problem_converged = is_square_problem & (orig_trial_inf_pr <= jnp.minimum(orig_tol, constr_viol_tol))

    # Check 3: Infeasibility reduction - must satisfy orig_trial_inf_pr <= orig_inf_pr_max
    infeas_check_passes = orig_trial_inf_pr <= orig_inf_pr_max

    # Compute original problem's theta and barrier at trial point for filter checks
    orig_trial_theta = cp.nstqf.calc_theta(reg_c, reg_dms)

    # Compute original barrier - need slacks at current point
    orig_slacks = cp.nstqf.calc_slacks(it)
    orig_trial_barr = cp.nstqf.calc_barrier_obj(reg_f, saved_mu, orig_slacks)

    # Check 4: IsAcceptableToCurrentFilter - trial point must be acceptable to original filter
    filter_acceptable = cp.stqf.is_acceptable_to_current_filter(
        orig_trial_barr, orig_trial_theta, saved_ls.filter.F
    )

    # Check 5: IsAcceptableToCurrentIterate - with called_from_restoration=True (third arg)
    # This checks if trial is acceptable w.r.t. the saved reference point
    iterate_acceptable = cp.stqf.is_acceptable_to_current_iterate(
        orig_trial_barr, orig_trial_theta,
        saved_ls.filter.ref_barr, saved_ls.filter.ref_theta,
        jnp.array([[1]])  # called_from_restoration=True skips obj_max_inc check
    )

    # Filter/iterate checks must all pass for normal convergence path
    filter_checks_pass = infeas_check_passes & filter_acceptable & iterate_acceptable

    # Exit restoration if: (not first iter) AND (square problem converged OR filter checks pass)
    should_exit_resto = (~first_resto_iter) & (square_problem_converged | filter_checks_pass)

    # =========================================================================
    # Local infeasibility handling (IPOPT lines 203-242)
    # If we didn't exit but resto problem itself converged, handle edge cases
    # =========================================================================

    # Only check local infeasibility if we're continuing (didn't exit) AND resto converged
    check_local_infeas = (~should_exit_resto) & (~first_resto_iter) & resto_converged

    # Case 1: Nearly feasible, can tighten tolerance
    # IPOPT: orig_trial_inf_pr <= 1e2 * resto_tol && resto_tol > 1e-1 * orig_tol
    nearly_feasible = orig_trial_inf_pr <= 1e2 * resto_tol
    can_tighten = resto_tol > 1e-1 * orig_tol
    should_tighten_tol = check_local_infeas & nearly_feasible & can_tighten

    # Case 2: Square problem, feasible w.r.t. constr_viol_tol (but not tol)
    # IPOPT: IsSquareProblem() && orig_trial_inf_pr <= constr_viol_tol
    is_acceptable_convergence = check_local_infeas & (~should_tighten_tol) & is_square_problem & (orig_trial_inf_pr <= constr_viol_tol)

    # Case 3: Feasible but filter rejects
    # IPOPT: orig_trial_inf_pr <= 1e2 * resto_tol (but didn't pass earlier checks)
    is_feasible_filter_rejects = check_local_infeas & (~should_tighten_tol) & (~is_acceptable_convergence) & nearly_feasible

    # Case 4: Locally infeasible
    # IPOPT: else (resto converged but not feasible enough)
    is_locally_infeasible = check_local_infeas & (~should_tighten_tol) & (~is_acceptable_convergence) & (~is_feasible_filter_rejects)

    return (
        jnp.atleast_2d(should_exit_resto),
        jnp.atleast_2d(is_locally_infeasible),
        jnp.atleast_2d(is_feasible_filter_rejects),
        jnp.atleast_2d(should_tighten_tol),
        jnp.atleast_2d(is_acceptable_convergence)
    )


# while loop condition
def line_search_cond(state, cp):
    ls_stop_cond = state.ls.accept  # | (state.fl.expect_infeasible_problem & (state.ls.count_successive_shortened_steps >= 5)))
    ls_stop_cond = ls_stop_cond | (state.ls.alpha_pr <= state.ls.alpha_min)
    ls_continue_cond = 1 - ls_stop_cond
    return (ls_continue_cond == 1).squeeze()

def line_search_step(state, cp):

    if cp.p["DEBUG_MODE"]:
        jax.debug.print("performing ls step: {x}", x=True)

    resto = state.fl.in_restoration.squeeze()
    nxr = cp.nx + cp.nyc * 2 + cp.nyd * 2

    # Check skip_first_trial flag (set on watchdog timeout)
    # If set, reduce initial alpha by alpha_red_factor (IPOPT behavior)
    skip_first = state.fl.skip_first_trial.squeeze()
    alpha_pr = jnp.where(
        skip_first,
        state.ls.alpha_pr * cp.p["alpha_red_factor"],
        state.ls.alpha_pr
    )

    # Debug logging: capture alpha values for the full batch
    # log_ls_alpha(
    #     state.iter_count,
    #     state.ls.n_steps,
    #     alpha_pr,
    #     state.ls.alpha_min,
    #     state.ls.accept,
    #     state.fl.in_restoration
    # )

    it_trial_LS = Iterate(
        state.it.x + alpha_pr * state.cqpo.step.x,
        state.it.s + alpha_pr * state.cqpo.step.s,
        state.it.y_c,
        state.it.y_d,
        state.it.z_L,
        state.it.z_U,
        state.it.v_L,
        state.it.v_U,
    )

    slacks_reg_trial_LS = cp.nstqf.calc_slacks(it_trial_LS)
    slacks_reg_trial_LS = (jnp.vstack([slacks_reg_trial_LS[0], jnp.zeros([cp.nyc*2+cp.nyd*2,1])]), *slacks_reg_trial_LS[1:])

    x_ref, dr_x = state.args[0][1:3]

    f_reg_trial_LS = jnp.atleast_2d(cp.nstqf.calc_f(it_trial_LS.x[:cp.nx], *state.args[0]))
    c_reg_trial_LS = cp.nstqf.calc_c(it_trial_LS.x[:cp.nx], *state.args[1])
    d_reg_trial_LS = cp.nstqf.calc_d(it_trial_LS.x[:cp.nx], *state.args[2])
    f_resto_trial_LS = jnp.atleast_2d(cp.nstqfr.calc_f(it_trial_LS.x.flatten(), state.mu, x_ref, dr_x))
    c_resto_trial_LS = cp.nstqfr.calc_c(c_reg_trial_LS.flatten(), it_trial_LS.x.flatten())[:, None]
    d_resto_trial_LS = cp.nstqfr.calc_d(d_reg_trial_LS.flatten(), it_trial_LS.x.flatten())[:, None]

    f_trial_LS, c_trial_LS, d_trial_LS, slacks_trial_LS = filter_select(
        resto, (
            f_resto_trial_LS, c_resto_trial_LS, d_resto_trial_LS, cp.nstqfr.calc_slacks(it_trial_LS)
        ), (
            f_reg_trial_LS, c_reg_trial_LS, d_reg_trial_LS, slacks_reg_trial_LS
        )
    )

    # f_trial_LS, c_trial_LS, d_trial_LS, slacks_trial_LS = filter_select(
    #     resto, (
    #         cp.nstqfr.calc_f(it_trial_LS.x[:nxr], *state.args[0]),
    #         cp.nstqfr.calc_c(it_trial_LS.x[:nxr], *state.args[1]),
    #         cp.nstqfr.calc_d(it_trial_LS.x[:nxr], *state.args[2]),
    #         cp.nstqfr.calc_slacks(it_trial_LS)
    #     ), (
    #         cp.nstqf.calc_f(it_trial_LS.x[:cp.nx], *state.args[0]),
    #         cp.nstqf.calc_c(it_trial_LS.x[:cp.nx], *state.args[1]),
    #         cp.nstqf.calc_d(it_trial_LS.x[:cp.nx], *state.args[2]),
    #         slacks_reg_trial_LS
    #     )
    # )
    dms_trial_LS = d_trial_LS - it_trial_LS.s

    # calculations required for this iteration:
    barr_trial_LS, theta_trial_LS = filter_select(
        resto, (
            cp.nstqfr.calc_barrier_obj(f_trial_LS, state.mu, slacks_trial_LS),
            cp.nstqfr.calc_theta(c_trial_LS, dms_trial_LS)
        ), (
            cp.nstqf.calc_barrier_obj(f_trial_LS, state.mu, slacks_trial_LS),
            cp.nstqf.calc_theta(c_trial_LS, dms_trial_LS)
        )
    )

    trivial_cond = cp.p["accept_every_trial_step"] | (
        cp.p["accept_after_max_steps"]
    ) & (state.iter_count >= cp.p["accept_after_max_steps"] - 1)
    true_branch = lambda: (  # noqa: E731
        jnp.array([[1]]),
        state.ls.filter.F,
        jnp.atleast_2d(state.ls.filter.count_successive_filter_rejections),
        state.ls.filter.n_filter_resets,
        state.ls.filter.last_rejection_due_to_filter,
    )
    false_branch = lambda: cp.stqf.check_acceptability_of_trial_point(  # noqa: E731
        barr_trial_LS,
        theta_trial_LS,
        state.ls.filter.ref_barr, # watchdog aware
        state.ls.filter.ref_theta, # watchdog aware
        state.ls.filter.ref_gBD, # watchdog aware
        state.ls.filter.theta_min,
        state.ls.filter.theta_max,
        alpha_pr,
        state.ls.filter.last_rejection_due_to_filter,
        state.ls.filter.count_successive_filter_rejections,
        state.ls.filter.n_filter_resets,
        state.ls.filter.F,
        state.fl.in_restoration,
    )
    (
        accept,
        F,
        count_successive_filter_rejections,
        n_filter_resets,
        last_rejection_due_to_filter,
    ) = jax.lax.cond(trivial_cond.squeeze(), true_branch, false_branch)

    # if first step and not accept -> SOC
    # only reduce alpha if not accept, not first backtrack step, and still above alpha_min
    reduce_alpha = (1 - accept) & (state.ls.n_steps > 0) & (alpha_pr > state.ls.alpha_min)
    alpha_pr *= (1 - reduce_alpha) + cp.p["alpha_red_factor"] * reduce_alpha  

    state = eqx.tree_at(
        lambda t: (
            t.ls.filter.F,
            t.ls.filter.count_successive_filter_rejections,
            t.ls.filter.n_filter_resets,
            t.ls.filter.last_rejection_due_to_filter,
            t.ls.accept,
            t.ls.it_trial,
            t.ls.alpha_pr,
            t.ls.trial_theta,
            t.ls.n_steps,
            t.ls.trial_step,
            t.fl.skip_first_trial,  # Reset skip_first_trial flag after use
        ),
        state,
        (
            F,
            count_successive_filter_rejections,
            n_filter_resets,
            last_rejection_due_to_filter,
            accept,
            it_trial_LS,
            alpha_pr,
            theta_trial_LS,
            state.ls.n_steps + (1 - accept),
            state.cqpo.step,
            jnp.array([[0]]),  # Reset flag
        ),
    )

    return state


# while loop condition
def soc_cond(state, cp):
    # ls_k, it_k, F_k, alpha_pr_j, alpha_min = carry_k

    soc_continue_cond = (
        (state.ls.count_soc < cp.p["max_soc"])
        & (1 - state.ls.accept)
        & (
            (state.ls.count_soc == 0)
            | (
                state.ls.trial_theta
                <= cp.p["kappa_soc"] * state.ls.theta_soc_old
            )
        )
    )

    return (soc_continue_cond == 1).squeeze()


def soc_step(state, cp):
    """a single SOC step - utilized by eip and no wd branches here
    NOTE: the SOC is special as it is the only place we can have
    a dynamic number of linear solves occur - therefore it is
    very important we think of how we asynchronously do this later."""

    resto = state.fl.in_restoration.squeeze()
    rnx = cp.nx + cp.nyc * 2 + cp.nyd * 2

    # if resto is True: 
    #     nstqf = cp.nstqfr
    #     nx = cp.nx + cp.nyc * 2 + cp.nyd * 2  # adjusted for resto

    # elif resto is False: 
    #     nstqf = cp.nstqf
    #     nx = cp.nx

    # else: raise ValueError("resto flag must be True or False")

    if cp.p["DEBUG_MODE"]:
        jax.debug.print("performing soc step: {x}", x=True)

    # save old values we need to keep around
    state = eqx.tree_at(
        lambda t: t.ls.theta_soc_old, state, state.ls.trial_theta
    )

    # pre-calculations
    alpha_pr_soc = state.ls.alpha_pr  # unpack for clarity
    
    # trial_c, trial_dms = filter_select(
    #     resto,
    #     (
    #         cp.nstqfr.calc_c(state.ls.it_trial.x, *state.args[1]),
    #         cp.nstqfr.calc_d(state.ls.it_trial.x, *state.args[2]) - state.ls.it_trial.s
    #     ), (
    #         cp.nstqf.calc_c(state.ls.it_trial.x, *state.args[1]),
    #         cp.nstqf.calc_d(state.ls.it_trial.x, *state.args[2]) - state.ls.it_trial.s
    #     )
    # )

    reg_c = cp.nstqf.calc_c(state.ls.it_trial.x[:cp.nx], *state.args[1])
    reg_d = cp.nstqf.calc_d(state.ls.it_trial.x[:cp.nx], *state.args[2])
    reg_dms = reg_d - state.ls.it_trial.s
    resto_c = cp.nstqfr.calc_c(reg_c, state.ls.it_trial.x.flatten())[:,None]
    resto_d = cp.nstqfr.calc_d(reg_d, state.ls.it_trial.x.flatten())[:,None]
    resto_dms = resto_d - state.ls.it_trial.s
    trial_c, trial_dms = filter_select(
        resto, (
            resto_c, resto_dms
        ), (
            reg_c, reg_dms 
        )
    )
    
    c_soc = alpha_pr_soc * state.ls.c_soc + trial_c
    dms_soc = alpha_pr_soc * state.ls.dms_soc + trial_dms

    rrhs_soc_full = cp.stqf.calc_vector_to_iterate(
        cp.nstqfr.kkt.calc_resto_red_pd_RHS(
            state.it,
            state.mu,
            state.cqpr.slacks,
            state.cqpr.grad_lag_x,
            state.cqpr.grad_lag_s,
            c_soc,   # SOC-modified
            dms_soc, # SOC-modified
            state.cqpr.Sigma_nc_inv,
            state.cqpr.Sigma_pc_inv,
            state.cqpr.Sigma_nd_inv,
            state.cqpr.Sigma_pd_inv,
        )
    )
    if cp.p["DEBUG_MODE"]: jax.debug.print("CRITICAL WARNING: resto SOC rhs incorrect almost certainly")

    rrhs_red_soc = jnp.vstack([rrhs_soc_full.x[:cp.nx], rrhs_soc_full.s, rrhs_soc_full.y_c, rrhs_soc_full.y_d])
    rhs_aug_soc = jnp.vstack([state.cqpo.rhs.x[:cp.nx], state.cqpo.rhs.s, -c_soc, -dms_soc])

    # unified cuDSS call with iterative refinement!
    rhs = jnp.where(resto, rrhs_red_soc, rhs_aug_soc)
    step = cp.linear_solve(rhs.flatten(), state.ic.perturbed_data)[0][:, None]

    # resto / reg
    rstep_aug_soc = cp.nstqfr.kkt.calc_transform_red_to_aug(step, rrhs_soc_full, state.cqpr.Sigma_nc_inv, state.cqpr.Sigma_pc_inv, state.cqpr.Sigma_nd_inv, state.cqpr.Sigma_pd_inv)
    step_aug_soc = jnp.vstack([step[:cp.nx], jnp.zeros([cp.nyc*2+cp.nyd*2,1]), step[cp.nx:]])
    step_aug_soc = jnp.where(resto, rstep_aug_soc, step_aug_soc)

    rstep_soc = cp.stqf.calc_vector_to_iterate(
        cp.nstqfr.calc_transform_aug_to_full(
            state.it, step_aug_soc, state.cqpo.rhs, state.cqpr.slacks, state.fl
        )
    )
    step_soc = cp.stqf.calc_vector_to_iterate(
        cp.nstqf.calc_transform_aug_to_full(
            state.it, step_aug_soc, state.cqpo.rhs, state.cqpr.slacks, state.fl
        )
    )
    step_soc = filter_select(resto, [rstep_soc], [step_soc])[0]

    tmp = cp.nstqf.calc_slack_derivatives(step_soc)
    reg_slack_derivatives = (jnp.vstack([tmp[0], jnp.zeros([cp.nyc*2+cp.nyd*2,1])]), *tmp[1:])

    slack_derivatives_soc = filter_select(
        resto,
        cp.nstqfr.calc_slack_derivatives(step_soc),
        reg_slack_derivatives,
    )
    alpha_pr_soc = jnp.atleast_2d(jnp.where(resto,
        cp.nstqfr.calc_alpha_pr(
            state.tau, state.cqpr.slacks, slack_derivatives_soc
        ),
        cp.nstqf.calc_alpha_pr(
            state.tau, state.cqpr.slacks, slack_derivatives_soc
        )
    ))

    it_trial = Iterate(
        state.it.x + alpha_pr_soc * step_soc.x,
        state.it.s + alpha_pr_soc * step_soc.s,
        state.it.y_c,
        state.it.y_d,
        state.it.z_L,
        state.it.z_U,
        state.it.v_L,
        state.it.v_U,
    )

    # more necessary calculations
    # trial_slacks_k = nstqf.calc_slacks(it_trial_k)
    # f_k = nstqf.calc_f(it_trial_k.x[:nx], *state.args[0])
    # trial_barr_k = nstqf.calc_barrier_obj(f_k, state.mu, trial_slacks_k)
    # c_k = nstqf.calc_c(it_trial_k.x, *state.args[1])
    # dms_k = nstqf.calc_d(it_trial_k.x, *state.args[2]) - it_trial_k.s
    # trial_theta_k = nstqf.calc_theta(c_k, dms_k)

    x_ref, dr_x = state.args[0][1:3]

    f_reg_trial_LS = jnp.atleast_2d(cp.nstqf.calc_f(it_trial.x[:cp.nx], *state.args[0]))
    c_reg_trial_LS = cp.nstqf.calc_c(it_trial.x[:cp.nx], *state.args[1])
    d_reg_trial_LS = cp.nstqf.calc_d(it_trial.x[:cp.nx], *state.args[2])
    f_resto_trial_LS = jnp.atleast_2d(cp.nstqfr.calc_f(it_trial.x.flatten(), state.mu, x_ref, dr_x))
    c_resto_trial_LS = cp.nstqfr.calc_c(c_reg_trial_LS.flatten(), it_trial.x.flatten())[:, None]
    d_resto_trial_LS = cp.nstqfr.calc_d(d_reg_trial_LS.flatten(), it_trial.x.flatten())[:, None]
    
    resto_trial_slacks = cp.nstqfr.calc_slacks(it_trial)
    tmp = cp.nstqf.calc_slacks(it_trial)
    reg_trial_slacks = (jnp.vstack([tmp[0], jnp.zeros([cp.nyc*2+cp.nyd*2,1])]), *tmp[1:])

    f, c, d, trial_slacks = filter_select(
        resto, (
            f_resto_trial_LS, c_resto_trial_LS, d_resto_trial_LS,
            resto_trial_slacks
        ), (
            f_reg_trial_LS, c_reg_trial_LS, d_reg_trial_LS,
            reg_trial_slacks
        )
    )
    dms = d - it_trial.s

    # trial_slacks, f = filter_select(
    #     resto,
    #     (
    #         cp.nstqfr.calc_slacks(it_trial),
    #         cp.nstqfr.calc_f(it_trial.x[:rnx], *state.args[0]),
    #     ), (
    #         cp.nstqf.calc_slacks(it_trial),
    #         cp.nstqf.calc_f(it_trial.x[:cp.nx], *state.args[0]),
    #     )
    # )

    trial_barr = jnp.where(resto, 
        cp.nstqfr.calc_barrier_obj(f, state.mu, trial_slacks), 
        cp.nstqf.calc_barrier_obj(f, state.mu, trial_slacks)
    )

    # trial_barr, c, dms = filter_select(
    #     resto,
    #     (
    #         cp.nstqfr.calc_barrier_obj(f, state.mu, trial_slacks),
    #         cp.nstqfr.calc_c(it_trial.x, *state.args[1]),
    #         cp.nstqfr.calc_d(it_trial.x, *state.args[2]) - it_trial.s
    #     ), (
    #         cp.nstqf.calc_barrier_obj(f, state.mu, trial_slacks),
    #         cp.nstqf.calc_c(it_trial.x, *state.args[1]),
    #         cp.nstqf.calc_d(it_trial.x, *state.args[2]) - it_trial.s
    #     )
    # )

    trial_theta = jnp.where(resto, 
        cp.nstqfr.calc_theta(c, dms),
        cp.nstqf.calc_theta(c, dms)
    )

    (
        accept,
        F,
        count_successive_filter_rejections,
        n_filter_resets,
        last_rejection_due_to_filter,
    ) = cp.stqf.check_acceptability_of_trial_point(
        trial_barr,
        trial_theta,
        state.cqpo.barr,
        state.cqpr.theta,
        state.cqpo.gBD,
        state.ls.filter.theta_min,
        state.ls.filter.theta_max,
        state.cqpo.alpha_pr,  # Use original rejected alpha for Armijo test (IPOPT uses fixed alpha_primal_test)
        state.ls.filter.last_rejection_due_to_filter,
        state.ls.filter.count_successive_filter_rejections,
        state.ls.filter.n_filter_resets,
        state.ls.filter.F,
        state.fl.in_restoration,
    )

    state = eqx.tree_at(
        lambda t: (
            t.ls.filter.F,
            t.ls.filter.count_successive_filter_rejections,
            t.ls.filter.n_filter_resets,
            t.ls.filter.last_rejection_due_to_filter,
            t.ls.accept,
            t.ls.it_trial,
            t.ls.alpha_pr,
            t.ls.trial_theta,
            t.ls.count_soc,
            t.ls.c_soc,
            t.ls.dms_soc,
            t.ls.trial_step,
        ),
        state,
        (
            F,
            count_successive_filter_rejections,
            n_filter_resets,
            last_rejection_due_to_filter,
            accept,
            it_trial,
            alpha_pr_soc,
            trial_theta,
            state.ls.count_soc
            + (1 - accept),  # only add to count if we didnt accept
            c_soc,
            dms_soc,
            step_soc,
        ),
    )

    # jax.debug.print("SOC: alpha_pr_soc: {x}", x=alpha_pr_soc)
    # jax.debug.print("SOC: trial_theta: {x}", x=trial_theta)
    # jax.debug.print("SOC: accept: {x}", x=accept)
    # jax.debug.print("SOC: count_soc: {x}", x=state.ls.count_soc)
    # jax.debug.print("SOC: min slack: {x}", x=jnp.min(jnp.vstack(trial_slacks), initial=jnp.inf))
    # jax.debug.print("SOC: trial barr: {x}", x=trial_barr)

    # # After line 435:                                                                                                                                    
    # jax.debug.print("SOC c_soc: {x}, dms_soc: {y}", x=jnp.isnan(c_soc).any(), y=jnp.isnan(dms_soc).any())                                                
                                                                                                                                                        
    # # After line 459:                                                                                                                                    
    # jax.debug.print("SOC step has nan: {x}", x=jnp.isnan(step).any())                                                                                    
                                                                                                                                                        
    # # After line 476:                                                                                                                                    
    # jax.debug.print("SOC step_soc.x has nan: {x}", x=jnp.isnan(step_soc.x).any())           

    return state


def start_watchdog(mu, cqpr, cqpo, it, fl, wd):
    # NOTE: IPOPT's StartWatchDog does NOT reset shortened_iter - only StopWatchDog does.
    # Preserve the existing shortened_iter value.
    return WatchdogState(
        shortened_iter=wd.shortened_iter,  # Preserve existing value (IPOPT behavior)
        trial_iter=jnp.array([[0]]),
        alpha_pr_test=cqpo.alpha_pr,
        it=it,
        delta=cqpo.step,
        last_mu=mu,
        theta=cqpr.theta,
        barr=cqpo.barr,
        gBD=cqpo.gBD
    ), eqx.tree_at(lambda t: t.in_watchdog, fl, jnp.array([[1]]))

def stop_watchdog(state):# fl, F, wd):

    state = eqx.tree_at(
        lambda t: (
            t.fl.in_watchdog,
            t.ls.filter.ref_theta,
            t.ls.filter.ref_barr,
            t.ls.filter.ref_gBD,
            t.cqpo.step,
            t.it,
            t.wd.shortened_iter
        ),
        state,
        (
            jnp.array([[0]]), # no longer in watchdog
            state.wd.theta, # transfer watchdog values 
            state.wd.barr, # transfer watchdog values
            state.wd.gBD, # transfer watchdog values
            state.wd.delta, # set the step used to the watchdog one
            state.wd.it, # restore checkpoint
            jnp.array([[0]]) # reset shortened iter
        )
    )
    return state

# IpRestoRestoPhase.cpp
def resto_resto_iter(resto_opt_state, cp):
    # this is repeated in the initialization.py for something else
    def solve_quadratic(a, b):
        ret = a
        ret *= a
        ret += b
        ret = jnp.sqrt(ret)
        ret += a
        return ret

    # x value remains unchanged
    x = resto_opt_state.it.x
    s = resto_opt_state.it.s
    mu = resto_opt_state.mu
    rho = cp.p["resto_penalty_parameter"]

    # compute the initial values for n and p for eq constraints
    nc = x[cp.nx : cp.nx + cp.nyc]
    pc = x[cp.nx + cp.nyc : cp.nx + cp.nyc * 2]
    cvec = cp.calc_c(
        x[: cp.nx], *resto_opt_state.function_args[1]
    )  # orig.c with scaling calculated at resto x[:cp.nx] - which we KNOW from resto_opt_state.function_args[1][0] for c and [2][0] for d!
    a = jnp.full(nc.shape, mu / (2.0 * rho))
    a -= 0.5 * cvec
    b = cvec * mu / (2.0 * rho)
    nc = solve_quadratic(a, b)
    pc = cvec + nc

    # initial values for the n and pc vars for the ineq constraints
    nd = x[cp.nx + cp.nyc * 2 : cp.nx + cp.nyc * 2 + cp.nyd]
    pd = x[cp.nx + cp.nyc * 2 + cp.nyd : cp.nx + cp.nyc * 2 + cp.nyd * 2]
    dvec = cp.calc_d(x[: cp.nx], *resto_opt_state.function_args[2])
    dvec -= s
    a = jnp.full(nd.shape, mu / (2.0 * rho))
    a -= 0.5 * dvec
    b = dvec * mu / (2.0 * rho)
    nd = solve_quadratic(a, b)
    pd = dvec + nd

    # set trial point, only x is changed
    resto_opt_state = eqx.tree_at(
        lambda t: t.it.x, resto_opt_state, jnp.vstack([x[: cp.nx], nc, pc, nd, pd])
    )

    return resto_opt_state

def execute_search(state, cp):
    """
    This function performs the regular iteration type, which can consist of one of
    the five following branches:

    Single step calculation branches
    - Soft Feasibility Restoration (SFR)
    - Tiny Step (TS)
    - WatchDog (WD)

    While loops
    - BackTrack (BT)
    - Second Order Correction (SOC)

    Before BT/SOC while loops (which occur sequentially and are selected from),
    we must determine which loop to perform. In order to do so we must perform one
    step from both of them.
    """

    # =========================================================================
    # ITERATION START: resets and fallback handling
    # =========================================================================

    original_state = state

    # Reset watchdog if mu changed (different barrier problem)
    state = filter_tree_at_select(
        (state.mu != state.wd.last_mu).squeeze(),
        lambda t: (t.fl.in_watchdog, t.wd.shortened_iter, t.wd.last_mu),
        state,
        (jnp.array([[0]]), jnp.array([[0]]), state.mu),
    )

    # Fallback handling: if fallback_activated, zero step and set goto_resto
    # Also stop watchdog if we're in it
    goto_resto = state.fl.fallback_activated
    state = filter_tree_at_select(
        state.fl.fallback_activated.squeeze(),
        lambda t: (t.fl.fallback_activated, t.cqpo.step),
        state,
        (jnp.array([[0]]), (state.cqpo.step ** ω * 0.0).ω),  # clear flag, zero step
    )
    # Stop watchdog if in_watchdog and goto_resto
    state = filter_tree_at_select(
        (state.fl.in_watchdog & goto_resto).squeeze(),
        lambda t: t,
        state,
        stop_watchdog(state),
    )

    # easy to switch later...
    # nstqf = cp.nstqf
    # nx = cp.nx

    resto = state.fl.in_restoration.squeeze()
    rnx = cp.nx + cp.nyc * 2 + cp.nyd * 2

    # alpha_pr_max = nstqf.calc_alpha_pr(state.tau, state.cqpr.slacks, state.cqpo.slack_derivatives)
    # alpha_du_max = nstqf.calc_alpha_du(state.it, state.cqpo.step, state.tau)

    alpha_pr_max, alpha_du_max = filter_select(
        resto, (
            cp.nstqfr.calc_alpha_pr(state.tau, state.cqpr.slacks, state.cqpo.slack_derivatives),
            cp.nstqfr.calc_alpha_du(state.it, state.cqpo.step, state.tau),
        ), (
            cp.nstqf.calc_alpha_pr(state.tau, state.cqpr.slacks, state.cqpo.slack_derivatives),
            cp.nstqf.calc_alpha_du(state.it, state.cqpo.step, state.tau),
        )
    )

    # SFR quantities
    alpha_SFR = jnp.minimum(alpha_pr_max, alpha_du_max)
    it_trial_SFR = Iterate( # applies min F2B step to whole iterate
        state.it.x + alpha_SFR * state.cqpo.step.x,
        state.it.s + alpha_SFR * state.cqpo.step.s,
        state.it.y_c + alpha_SFR * state.cqpo.step.y_c,
        state.it.y_d + alpha_SFR * state.cqpo.step.y_d,
        state.it.z_L + alpha_SFR * state.cqpo.step.z_L,
        state.it.z_U + alpha_SFR * state.cqpo.step.z_U,
        state.it.v_L + alpha_SFR * state.cqpo.step.v_L,
        state.it.v_U + alpha_SFR * state.cqpo.step.v_U,
    )

    # SFR calculations
    x_ref, dr_x = state.args[0][1:3]

    f_reg_trial_SFR = jnp.atleast_2d(cp.nstqf.calc_f(it_trial_SFR.x[:cp.nx], *state.args[0]))
    c_reg_trial_SFR = cp.nstqf.calc_c(it_trial_SFR.x[:cp.nx], *state.args[1])
    d_reg_trial_SFR = cp.nstqf.calc_d(it_trial_SFR.x[:cp.nx], *state.args[2])
    f_resto_trial_SFR = jnp.atleast_2d(cp.nstqfr.calc_f(it_trial_SFR.x.flatten(), state.mu, x_ref, dr_x))
    c_resto_trial_SFR = cp.nstqfr.calc_c(c_reg_trial_SFR.flatten(), it_trial_SFR.x.flatten())[:, None]
    d_resto_trial_SFR = cp.nstqfr.calc_d(d_reg_trial_SFR.flatten(), it_trial_SFR.x.flatten())[:, None]

    f_trial_SFR, c_trial_SFR, d_trial_SFR = filter_select(
        resto, (
            f_resto_trial_SFR, c_resto_trial_SFR, d_resto_trial_SFR
        ), (
            f_reg_trial_SFR, c_reg_trial_SFR, d_reg_trial_SFR
        )
    )

    tmp = cp.nstqf.calc_slacks(it_trial_SFR)
    reg_slacks = (jnp.vstack([tmp[0], jnp.zeros([cp.nyc*2+cp.nyd*2,1])]), *tmp[1:])
    theta_trial_SFR, slacks_trial_SFR = filter_select(
        resto, (
            cp.nstqfr.calc_theta(c_trial_SFR, d_trial_SFR - it_trial_SFR.s),
            cp.nstqfr.calc_slacks(it_trial_SFR)
        ), (
            cp.nstqf.calc_theta(c_trial_SFR, d_trial_SFR - it_trial_SFR.s),
            reg_slacks
        )
    )

    pad_rows = cp.nyc * 2 + cp.nyd * 2 
    reg_jac_f_unpadded = cp.nstqf.calc_jac_f(state.it.x[:cp.nx], *state.args[0]).todense()
    reg_jac_c_unpadded = cp.nstqf.calc_jac_c(state.it.x[:cp.nx], *state.args[1])
    reg_jac_d_unpadded = cp.nstqf.calc_jac_d(state.it.x[:cp.nx], *state.args[2])
    reg_jac_f = jnp.vstack([reg_jac_f_unpadded, jnp.zeros([cp.nyc*2+cp.nyd*2,1])])
    # reg_jac_f = spu.vstack([reg_jac_f_unpadded, spu.ones_with_nse([pad_rows, 1], cp.nse_rJf - cp.nse_Jf)]).todense() # pad the regular jacobians
    reg_jac_c = spu.vstack([reg_jac_c_unpadded, spu.ones_with_nse([pad_rows, cp.nyc], cp.nse_rJc - cp.nse_Jc)]) # pad the regular jacobians
    reg_jac_d = spu.vstack([reg_jac_d_unpadded, spu.ones_with_nse([pad_rows, cp.nyd], cp.nse_rJd - cp.nse_Jd)]) # pad the regular jacobians

    resto_jac_f = cp.nstqfr.calc_jac_f(it_trial_SFR.x.flatten(), state.mu, x_ref, dr_x)
    resto_jac_c = cp.nstqfr.calc_jac_c(reg_jac_c_unpadded)
    resto_jac_d = cp.nstqfr.calc_jac_d(reg_jac_d_unpadded)

    # we evaluate the error at the new point - which requires recomputing the jacobians at that point
    barr_trial_SFR, jac_f_trial_SFR, jac_c_trial_SFR, jac_d_trial_SFR = filter_select(
        resto, (
            cp.nstqfr.calc_barrier_obj(f_trial_SFR, state.mu, slacks_trial_SFR), 
            resto_jac_f, 
            resto_jac_c, 
            resto_jac_d
        ), (
            cp.nstqf.calc_barrier_obj(f_trial_SFR, state.mu, slacks_trial_SFR),
            reg_jac_f,
            reg_jac_c,
            reg_jac_d
        )
    )

    jacobians_trial_SFR = (jac_f_trial_SFR, jac_c_trial_SFR, jac_d_trial_SFR)
    reg_grad_lag_x = jnp.vstack([
        cp.nstqf.calc_grad_lag_x(it_trial_SFR, jacobians_trial_SFR),
        jnp.zeros([cp.nyc*2+cp.nyd*2,1])
    ])
    grad_lag_x_trial_SFR = jnp.where(resto,
        cp.nstqfr.calc_grad_lag_x(it_trial_SFR, jacobians_trial_SFR),
        reg_grad_lag_x
    )
    grad_lag_s_trial_SFR = cp.stqf.calc_grad_lag_s(it_trial_SFR)

    # SFR logic
    (
        satisfies_original_criterion,
        F,
        count_successive_filter_rejections,
        n_filter_resets,
        last_rejection_due_to_filter,
    ) = cp.stqf.check_acceptability_of_trial_point(
        barr_trial_SFR,
        theta_trial_SFR,
        state.cqpo.barr,
        state.cqpr.theta,
        state.cqpo.gBD,
        state.ls.filter.theta_min,
        state.ls.filter.theta_max,
        0.0,
        state.ls.filter.last_rejection_due_to_filter,
        state.ls.filter.count_successive_filter_rejections,
        state.ls.filter.n_filter_resets,
        state.ls.filter.F,
        state.fl.in_restoration,
    )

    def calc_pd_sys_error(it, mu, grad_lag_x, grad_lag_s, c, d, slacks, fl):
        inf_du, inf_compl = filter_select(
            fl.in_restoration.squeeze(),
            (
                cp.nstqfr.calc_dual_inf(grad_lag_x, grad_lag_s),
                cp.nstqfr.calc_complementarity(it, mu, slacks)
            ), (
                cp.nstqf.calc_dual_inf(grad_lag_x, grad_lag_s),
                cp.nstqf.calc_complementarity(it, mu, slacks)
            )
        )
        inf_pr = cp.stqf.calc_barrier_constr_viol(c, d - it.s)
        return inf_du + inf_pr + inf_compl

    trial_pderror_SFR = calc_pd_sys_error(it_trial_SFR, state.mu, grad_lag_x_trial_SFR,
        grad_lag_s_trial_SFR, c_trial_SFR, d_trial_SFR, slacks_trial_SFR, state.fl)
    curr_pderror_SFR = calc_pd_sys_error(state.it, state.mu, state.cqpr.grad_lag_x,
        state.cqpr.grad_lag_s, state.cqpr.c, state.cqpr.d, state.cqpr.slacks, state.fl)

    accept_SFR = satisfies_original_criterion | (
        trial_pderror_SFR <= cp.p["soft_resto_pderror_reduction_factor"] * curr_pderror_SFR
    )

    is_entry_SFR = state.fl.soft_resto_entry_requested
    is_cont_SFR = state.fl.in_soft_resto_phase
    enter_soft_resto = is_entry_SFR & accept_SFR & (1 - satisfies_original_criterion)
    entry_fail_SFR = is_entry_SFR & (1 - accept_SFR)
    exit_soft_resto = is_cont_SFR & accept_SFR & satisfies_original_criterion
    continuation_fail_SFR = is_cont_SFR & (1 - accept_SFR)

    new_in_soft_resto_SFR = jnp.where(enter_soft_resto, jnp.array([[1]]),
        jnp.where(exit_soft_resto | continuation_fail_SFR, jnp.array([[0]]), state.fl.in_soft_resto_phase))
    new_fallback_SFR = jnp.where(entry_fail_SFR | continuation_fail_SFR, jnp.array([[1]]), state.fl.fallback_activated)
    new_entry_requested_SFR = jnp.array([[0]])
    # Counter: reset on exit, keep unchanged on failure, increment only on successful continuation
    # (matches soft_feas_resto.py which does NOT increment on entry_fail or continuation_fail)
    new_counter_SFR = jnp.where(
        exit_soft_resto, jnp.array([[0]]),
        jnp.where(entry_fail_SFR | continuation_fail_SFR,
            state.ls.soft_resto_phase_counter,  # keep unchanged on failure
            state.ls.soft_resto_phase_counter + 1)  # increment on success
    )

    # SFR filter values (renamed to avoid confusion with BT filter values)
    F_SFR = F
    count_successive_filter_rejections_SFR = count_successive_filter_rejections
    n_filter_resets_SFR = n_filter_resets
    last_rejection_due_to_filter_SFR = last_rejection_due_to_filter

    
    # TS quantities (matches regular_LS_finalize: y uses alpha_pr, z/v use alpha_du_max)
    it_trial_TS = Iterate(
        state.it.x + state.cqpo.alpha_pr * state.cqpo.step.x,
        state.it.s + state.cqpo.alpha_pr * state.cqpo.step.s,
        state.it.y_c + state.cqpo.step.y_c * state.cqpo.alpha_pr,  # uses alpha_pr per finalize
        state.it.y_d + state.cqpo.step.y_d * state.cqpo.alpha_pr,  # uses alpha_pr per finalize
        state.it.z_L + state.cqpo.step.z_L * alpha_du_max,
        state.it.z_U + state.cqpo.step.z_U * alpha_du_max,
        state.it.v_L + state.cqpo.step.v_L * alpha_du_max,
        state.it.v_U + state.cqpo.step.v_U * alpha_du_max,
    )
    delta_y_norm = jnp.maximum(
        jnp.linalg.norm(state.cqpo.step.y_c, ord=jnp.inf),
        jnp.linalg.norm(state.cqpo.step.y_d, ord=jnp.inf),
    )
    tiny_step_last_iter = jnp.atleast_2d(
        (delta_y_norm < cp.p["tiny_step_y_tol"]).astype(int)
    )

    # TS Logic - if n_steps == 0 then reset wd.shortened_iter
    reset_wd_shortened_TS = state.ls.n_steps == 0
    shortened_iter_TS = jnp.where(reset_wd_shortened_TS,
        jnp.zeros_like(state.wd.shortened_iter), state.wd.shortened_iter + 1)

    # WD pre-LS setup: init ref values based on watchdog state
    ref_theta, ref_barr, ref_gBD = filter_select(
        state.fl.in_watchdog.squeeze(),
        (state.wd.theta, state.wd.barr, state.wd.gBD),
        (state.cqpr.theta, state.cqpo.barr, state.cqpo.gBD)
    )
    state = eqx.tree_at(
        lambda t: (t.ls.filter.ref_theta, t.ls.filter.ref_barr, t.ls.filter.ref_gBD),
        state, (ref_theta, ref_barr, ref_gBD)
    )

    # First LS step
    state = line_search_step(state, cp)

    # WD outcome calculations (after first LS step)
    wd_accepted = state.ls.accept
    wd_at_max = state.wd.trial_iter >= cp.p["watchdog_trial_iter_max"]
    wd_timeout = (1 - wd_accepted) & wd_at_max
    wd_forced = (1 - wd_accepted) & (1 - wd_at_max)

    # SOC/BT branching: determine if we should try SOC first
    # SOC triggered when: first step not accepted AND theta increased AND not in watchdog
    theta_increased = state.cqpr.theta <= state.ls.trial_theta
    try_soc = (1 - state.ls.accept) & theta_increased & (1 - state.fl.in_watchdog)

    # Initialize SOC state (c_soc, dms_soc for accumulation)
    # IPOPT uses curr_c() and curr_d_minus_s() - the current iterate, not trial
    reg_c = cp.nstqf.calc_c(state.it.x[:cp.nx], *state.args[1])
    reg_d = cp.nstqf.calc_d(state.it.x[:cp.nx], *state.args[2])
    reg_dms = reg_d - state.it.s
    resto_c = cp.nstqfr.calc_c(reg_c, state.it.x.flatten())[:,None]
    resto_d = cp.nstqfr.calc_d(reg_d, state.it.x.flatten())[:,None]
    resto_dms = resto_d - state.it.s
    c_soc_init, dms_soc_init = filter_select(
        resto, (
            resto_c, resto_dms
        ), (
            reg_c, reg_dms 
        )
    )

    state = eqx.tree_at(
        lambda t: (t.ls.c_soc, t.ls.dms_soc, t.ls.count_soc, t.ls.theta_soc_old),
        state,
        # theta_soc_old = 0.0 mirrors IPOPT's function-local init in
        # TrySecondOrderCorrection; without this re-init a stale value from a
        # previous iteration's SOC leaks into this iteration's post-LS state.
        (c_soc_init, dms_soc_init, jnp.array([[0]]), jnp.atleast_2d(0.0)),
    )

    # SOC while loop - only runs if try_soc, else passes through unchanged
    # If theta decreased or in watchdog, SOC is skipped (state unchanged)
    state_after_soc = filter_while_loop(soc_cond, soc_step, state, cp)
    state_after_soc = filter_tree_at_select(
        try_soc.squeeze(), lambda t: t, state, state_after_soc, state
    )

    # TODO: placeholder for return-to-backtrack condition from SOC
    # soc_return_to_bt = ...
    soc_accepted = state_after_soc.ls.accept
    soc_alpha_pr = state_after_soc.ls.alpha_pr  # Save SOC alpha before overwrite for BT

    # Initialize alpha_min, alpha_pr, and n_steps BEFORE BT while loop
    alpha_min = cp.stqf.calc_alpha_min(state.cqpo.gBD, state.cqpr.theta, state.ls.filter.theta_min)
    state_after_soc = eqx.tree_at(
        lambda t: (t.ls.alpha_min, t.ls.alpha_pr, t.ls.n_steps),
        state_after_soc,
        (alpha_min, state.cqpo.alpha_pr, jnp.array([[0]]))  # Reset alpha_pr and n_steps for fresh BT
    )

    # BT while loop - runs if SOC didn't accept (or was skipped)
    state_after_bt = filter_while_loop(line_search_cond, line_search_step, state_after_soc, cp)

    # Select SOC result if accepted, else BT result
    state_soc_bt = filter_tree_at_select(
        soc_accepted.squeeze(),
        lambda t: t,
        state_after_bt,
        state_after_soc,
        state_after_bt
    )
    # Restore SOC alpha_pr if SOC was accepted (was overwritten for BT loop init)
    state_soc_bt = eqx.tree_at(
        lambda t: t.ls.alpha_pr,
        state_soc_bt,
        jnp.where(soc_accepted, soc_alpha_pr, state_soc_bt.ls.alpha_pr)
    )

    # BT failure detection (alpha_min already computed above)
    bt_failed = (1 - state_soc_bt.ls.accept) & (state_soc_bt.ls.alpha_pr < alpha_min)

    # BT failure handling: try soft resto IMMEDIATELY (same iteration, like IPOPT)
    # IPOPT flow: BT fails -> try soft resto step -> if fails, hard resto
    # We already computed accept_SFR above, so use it here when BT fails
    can_try_soft_resto = (
        bt_failed
        & (1 - state.fl.in_soft_resto_phase)
        & (1 - goto_resto)
        & (cp.p["soft_resto_pderror_reduction_factor"] > 0)
    )

    # When BT fails and we can try soft resto:
    # - If accept_SFR: enter soft resto (or take step if satisfies_original_criterion)
    # - If not accept_SFR: go directly to hard resto (like IPOPT)
    soft_resto_accepted_on_bt_fail = can_try_soft_resto & accept_SFR
    soft_resto_rejected_on_bt_fail = can_try_soft_resto & (1 - accept_SFR)

    # Must go to full resto if:
    # 1. BT failed and can't try soft resto, OR
    # 2. BT failed, tried soft resto, but it was rejected
    must_go_full_resto = bt_failed & ((1 - can_try_soft_resto) | soft_resto_rejected_on_bt_fail)

    # Update flags based on immediate soft resto attempt
    entry_requested_bt = state.fl.soft_resto_entry_requested  # Don't defer, handle immediately
    fallback_bt = jnp.where(must_go_full_resto, jnp.array([[1]]), state.fl.fallback_activated)

    # Check if soft resto counter exceeded (triggers fallback to full resto)
    soft_resto_counter_exceeded = state.fl.in_soft_resto_phase & (
        state.ls.soft_resto_phase_counter >= cp.p["max_soft_resto_iters"]
    )

    # Final branch selection conditions
    # is_sfr: either from previous iteration's flag OR from BT failure in this iteration
    # exclude goto_resto cases (must be part of regular_search like classify.py)
    is_sfr_from_prev = (1 - goto_resto) & (state.fl.in_soft_resto_phase | state.fl.soft_resto_entry_requested) & (1 - soft_resto_counter_exceeded)
    # Also count as SFR if BT failed and soft resto was accepted in same iteration
    is_sfr = is_sfr_from_prev | soft_resto_accepted_on_bt_fail
    # is_ts: no theta constraint (matches classify.py which doesn't check theta here)
    is_ts = (cp.p["tiny_step_tol"] > 0) & (1 - goto_resto) & (
        (jnp.linalg.norm(state.cqpo.step.x / (jnp.abs(state.it.x) + 1.0), ord=jnp.inf) <= cp.p["tiny_step_tol"]) &
        (jnp.linalg.norm(state.cqpo.step.s / (jnp.abs(state.it.s) + 1.0), ord=jnp.inf) <= cp.p["tiny_step_tol"])
    )

    # Stop watchdog if in_watchdog and tiny_step (goto_resto already handled at start)
    state = filter_tree_at_select(
        (state.fl.in_watchdog & is_ts).squeeze(),
        lambda t: t,
        state,
        stop_watchdog(state),
    )

    # Start watchdog if conditions met
    activate_wd_cond = (
        (cp.p["watchdog_shortened_iter_trigger"] > 0)
        & (1 - state.fl.in_watchdog)
        & (1 - goto_resto)
        & (1 - is_ts)
        & (1 - state.fl.in_soft_resto_phase)
        & (state.wd.shortened_iter >= cp.p["watchdog_shortened_iter_trigger"])
    )
    wd_new, fl_new = start_watchdog(state.mu, state.cqpr, state.cqpo, state.it, state.fl, state.wd)
    state = filter_tree_at_select(
        activate_wd_cond.squeeze(),
        lambda t: (t.wd, t.fl),
        state,
        (wd_new, fl_new),
    )

    is_wd = state.fl.in_watchdog

    # =========================================================================
    # BRANCH SELECTION (Priority: SFR > TS > WD > SOC/BT)
    # =========================================================================

    # SOC/BT path updates (already in state_soc_bt, just need to finalize iterate)
    alpha_du_final = jnp.where(
        resto, 
        cp.nstqfr.calc_alpha_du(state.it, state_soc_bt.ls.trial_step, state.tau),
        cp.nstqf.calc_alpha_du(state.it, state_soc_bt.ls.trial_step, state.tau)
    )
    it_final_soc_bt = Iterate(
        state_soc_bt.ls.it_trial.x,
        state_soc_bt.ls.it_trial.s,
        state.it.y_c + state_soc_bt.ls.trial_step.y_c * state_soc_bt.ls.alpha_pr,
        state.it.y_d + state_soc_bt.ls.trial_step.y_d * state_soc_bt.ls.alpha_pr,
        state.it.z_L + state_soc_bt.ls.trial_step.z_L * alpha_du_final,
        state.it.z_U + state_soc_bt.ls.trial_step.z_U * alpha_du_final,
        state.it.v_L + state_soc_bt.ls.trial_step.v_L * alpha_du_final,
        state.it.v_U + state_soc_bt.ls.trial_step.v_U * alpha_du_final,
    )

    # SOC/BT shortened_iter update (same logic as regular_LS_finalize)
    # If n_steps == 0: reset to 0, else increment by 1
    shortened_iter_soc_bt = jnp.where(
        state_soc_bt.ls.n_steps == 0,
        jnp.zeros_like(state.wd.shortened_iter),
        state.wd.shortened_iter + 1
    )

    # WD path - three outcomes: accepted, timeout, forced
    # Timeout must restore full checkpoint: ref_theta/barr/gBD from wd state, step from wd.delta
    (
        it_final_wd, in_wd_final, trial_iter_final, shortened_iter_wd, skip_first_wd, accept_wd,
        ref_theta_wd, ref_barr_wd, ref_gBD_wd, step_wd
    ) = filter_select(
        wd_accepted.squeeze(),
        # accepted: exit WD, use normal values
        (it_final_soc_bt, jnp.array([[0]]), jnp.array([[0]]), jnp.array([[0]]), jnp.array([[0]]), jnp.array([[1]]),
         state.ls.filter.ref_theta, state.ls.filter.ref_barr, state.ls.filter.ref_gBD, state_soc_bt.ls.trial_step),
        filter_select(
            wd_timeout.squeeze(),
            # timeout: restore checkpoint (ref values from wd, step from wd.delta)
            (state.wd.it, jnp.array([[0]]), state.wd.trial_iter, jnp.array([[0]]), jnp.array([[1]]), jnp.array([[0]]),
             state.wd.theta, state.wd.barr, state.wd.gBD, state.wd.delta),
            # forced accept: stay in WD, increment trial_iter
            (it_final_soc_bt, state.fl.in_watchdog, state.wd.trial_iter + 1, state.wd.shortened_iter, jnp.array([[0]]), jnp.array([[1]]),
             state.ls.filter.ref_theta, state.ls.filter.ref_barr, state.ls.filter.ref_gBD, state_soc_bt.ls.trial_step)
        )
    )

    # TS path updates
    it_final_ts = it_trial_TS

    # SFR path updates
    it_final_sfr = filter_select(accept_SFR.squeeze(), [it_trial_SFR], [state.it])[0]

    # Final selection: SFR > TS > WD > SOC/BT > FULL_RESTO
    # WD timeout needs to propagate ref values and step restoration
    # Also propagate ls.it_trial, ls.alpha_pr, ls.n_steps for each path
    # IMPORTANT: When must_go_full_resto=1, keep original iterate (not rejected trial)
    (
        it_final, accept_final, in_wd_out, trial_iter_out, shortened_iter_out, skip_first_out,
        ref_theta_out, ref_barr_out, ref_gBD_out, step_out,
        it_trial_out, alpha_pr_out, n_steps_out
    ) = filter_select(
        is_sfr.squeeze(),
        (it_final_sfr, accept_SFR, state.fl.in_watchdog, state.wd.trial_iter, state.wd.shortened_iter, jnp.array([[0]]),
         state.ls.filter.ref_theta, state.ls.filter.ref_barr, state.ls.filter.ref_gBD, state.cqpo.step,
         it_trial_SFR, jnp.atleast_2d(alpha_SFR), state.ls.n_steps),
        filter_select(
            is_ts.squeeze(),
            # TS: it_trial=it_trial_TS, alpha_pr=cqpo.alpha_pr, n_steps=0 (no LS)
            (it_final_ts, jnp.array([[1]]), state.fl.in_watchdog, state.wd.trial_iter, shortened_iter_TS, jnp.array([[0]]),
             state.ls.filter.ref_theta, state.ls.filter.ref_barr, state.ls.filter.ref_gBD, state.cqpo.step,
             it_trial_TS, jnp.atleast_2d(state.cqpo.alpha_pr), jnp.array([[0]])),
            filter_select(
                is_wd.squeeze(),
                (it_final_wd, accept_wd, in_wd_final, trial_iter_final, shortened_iter_wd, skip_first_wd,
                 ref_theta_wd, ref_barr_wd, ref_gBD_wd, step_wd,
                 state_soc_bt.ls.it_trial, state_soc_bt.ls.alpha_pr, state_soc_bt.ls.n_steps),
                # When must_go_full_resto: keep original iterate, don't accept rejected trial
                filter_select(
                    must_go_full_resto.squeeze(),
                    # FULL RESTO: keep original iterate (state.it), mark as not accepted
                    (state.it, jnp.array([[0]]), state.fl.in_watchdog, state.wd.trial_iter, state.wd.shortened_iter, jnp.array([[0]]),
                     state.ls.filter.ref_theta, state.ls.filter.ref_barr, state.ls.filter.ref_gBD, state.cqpo.step,
                     state.it, state_soc_bt.ls.alpha_pr, state_soc_bt.ls.n_steps),
                    # Normal SOC/BT: use trial iterate if accepted
                    (it_final_soc_bt, state_soc_bt.ls.accept, state.fl.in_watchdog, state.wd.trial_iter, shortened_iter_soc_bt, jnp.array([[0]]),
                     state.ls.filter.ref_theta, state.ls.filter.ref_barr, state.ls.filter.ref_gBD, state_soc_bt.ls.trial_step,
                     state_soc_bt.ls.it_trial, state_soc_bt.ls.alpha_pr, state_soc_bt.ls.n_steps)
                )
            )
        )
    )

    # SFR-specific flag updates
    # There are now THREE cases for SFR:
    # 1. is_sfr_from_prev: soft resto from previous iteration's flag (use new_*_SFR values)
    # 2. soft_resto_accepted_on_bt_fail: BT failed, soft resto accepted in same iteration
    # 3. non-SFR: normal BT path or soft resto counter exceeded

    # Case 2: BT failed and soft resto was accepted in same iteration
    # If satisfies_original_criterion: don't enter soft resto phase (in_soft_resto = 0)
    # If not: enter soft resto phase (in_soft_resto = 1)
    bt_fail_sfr_in_soft_resto = jnp.where(
        satisfies_original_criterion, jnp.array([[0]]), jnp.array([[1]])
    )
    bt_fail_sfr_fallback = jnp.array([[0]])  # No fallback needed, soft resto was accepted
    bt_fail_sfr_entry_requested = jnp.array([[0]])  # Clear the flag
    bt_fail_sfr_counter = jnp.array([[1]])  # First iteration in soft resto

    # Case 3: non-SFR path
    non_sfr_in_soft_resto = jnp.where(soft_resto_counter_exceeded, jnp.array([[0]]), state.fl.in_soft_resto_phase)
    non_sfr_fallback = jnp.where(soft_resto_counter_exceeded, jnp.array([[1]]), fallback_bt)
    non_sfr_counter = jnp.where(soft_resto_counter_exceeded, jnp.array([[0]]), state.ls.soft_resto_phase_counter)

    # Priority selection: is_sfr_from_prev > soft_resto_accepted_on_bt_fail > non-SFR
    in_soft_resto_out, fallback_out, entry_requested_out, counter_out = filter_select(
        is_sfr_from_prev.squeeze(),
        (new_in_soft_resto_SFR, new_fallback_SFR, new_entry_requested_SFR, new_counter_SFR),
        filter_select(
            soft_resto_accepted_on_bt_fail.squeeze(),
            (bt_fail_sfr_in_soft_resto, bt_fail_sfr_fallback, bt_fail_sfr_entry_requested, bt_fail_sfr_counter),
            (non_sfr_in_soft_resto, non_sfr_fallback, entry_requested_bt, non_sfr_counter)
        )
    )

    # TS-specific flag updates
    tiny_step_last_out = jnp.where(is_ts, tiny_step_last_iter, state.fl.tiny_step_last_iter)

    # Compute needs_resto_init: fallback requested AND not already in restoration
    needs_resto_init_out = fallback_out & (1 - state.fl.in_restoration)

    # Filter updates - SFR uses SFR values, TS uses unchanged values, WD/SOC/BT uses loop result
    F_out, csf_out, nfr_out, lrd_out = filter_select(
        is_sfr.squeeze(),
        (F_SFR, count_successive_filter_rejections_SFR, n_filter_resets_SFR, last_rejection_due_to_filter_SFR),
        filter_select(
            is_ts.squeeze(),
            # TS does not modify filter (matches tiny_step.py - filter not augmented)
            (state.ls.filter.F, state.ls.filter.count_successive_filter_rejections,
             state.ls.filter.n_filter_resets, state.ls.filter.last_rejection_due_to_filter),
            (state_soc_bt.ls.filter.F, state_soc_bt.ls.filter.count_successive_filter_rejections,
             state_soc_bt.ls.filter.n_filter_resets, state_soc_bt.ls.filter.last_rejection_due_to_filter)
        )
    )

    # Apply all updates to state
    state = eqx.tree_at(
        lambda t: (
            t.it,
            t.ls.accept,
            t.ls.it_trial,
            t.ls.alpha_pr,
            t.ls.n_steps,
            t.ls.filter.F,
            t.ls.filter.count_successive_filter_rejections,
            t.ls.filter.n_filter_resets,
            t.ls.filter.last_rejection_due_to_filter,
            t.ls.filter.ref_theta,  # WD timeout restores these
            t.ls.filter.ref_barr,
            t.ls.filter.ref_gBD,
            t.cqpo.step,  # WD timeout restores step from wd.delta
            # SOC bookkeeping: the merge below rebuilds ls from the PRE-LS state,
            # which silently dropped the loop-internal SOC fields (count of
            # rejected attempts, the accumulated SOC rhs, theta_soc_old, and the
            # actually-taken direction in trial_step). They are dead state for the
            # algorithm (re-initialized at every LS start), but carrying them makes
            # the post-LS state comparable against IPOPT's per-attempt soc.txt dump.
            t.ls.trial_step,
            t.ls.count_soc,
            t.ls.theta_soc_old,
            t.ls.c_soc,
            t.ls.dms_soc,
            t.fl.in_watchdog,
            t.fl.skip_first_trial,
            t.fl.in_soft_resto_phase,
            t.fl.fallback_activated,
            t.fl.soft_resto_entry_requested,
            t.fl.tiny_step_last_iter,
            t.fl.needs_resto_init,
            t.wd.trial_iter,
            t.wd.shortened_iter,
            t.ls.soft_resto_phase_counter,
        ),
        state,
        (
            it_final,
            accept_final,
            it_trial_out,
            alpha_pr_out,
            n_steps_out,
            F_out,
            csf_out,
            nfr_out,
            lrd_out,
            ref_theta_out,
            ref_barr_out,
            ref_gBD_out,
            step_out,
            step_out,  # trial_step == cqpo.step == the direction actually taken
            state_soc_bt.ls.count_soc,
            state_soc_bt.ls.theta_soc_old,
            state_soc_bt.ls.c_soc,
            state_soc_bt.ls.dms_soc,
            in_wd_out,
            skip_first_out,
            in_soft_resto_out,
            fallback_out,
            entry_requested_out,
            tiny_step_last_out,
            needs_resto_init_out,
            trial_iter_out,
            shortened_iter_out,
            counter_out,
        )
    )

    # bypass execute search if we are initializing regular.
    state = filter_select(
        original_state.fl.needs_regular_init.squeeze(),
        (original_state,),
        (state,)
    )[0]

    return state

def post_process(original_state, result, cp):
    """
    Post-process the iteration result and prepare for next iteration.
    Uses filter_select to handle resto/non-resto without if/else branching.
    """
    in_resto = result.fl.in_restoration.squeeze()
    init_regular = (result.fl.needs_regular_init.squeeze() > 0)
    init_resto = result.fl.needs_resto_init.squeeze() & ~init_regular


    # Step 0: Potentially startup resto - therefore save state
    # Only save when entering restoration (init_resto), otherwise preserve existing saved state
    saved_fl = jax.lax.cond(init_resto, lambda: result.fl, lambda: result.saved_fl)
    saved_wd = jax.lax.cond(init_resto, lambda: result.wd, lambda: result.saved_wd)
    saved_ls = jax.lax.cond(init_resto, lambda: result.ls, lambda: result.saved_ls)
    saved_ic = jax.lax.cond(init_resto, lambda: result.ic, lambda: result.saved_ic)
    saved_adfs = jnp.where(init_resto, result.adfs, result.saved_adfs)
    saved_mu = jnp.where(init_resto, result.mu, result.saved_mu)
    saved_tau = jnp.where(init_resto, result.tau, result.saved_tau)
    saved_mu_max = jnp.where(init_resto, result.mu_max, result.saved_mu_max)
    saved_init_dual_inf = jnp.where(init_resto, result.init_dual_inf, result.saved_init_dual_inf)
    saved_init_primal_inf = jnp.where(init_resto, result.init_primal_inf, result.saved_init_primal_inf)

    # Save bound multipliers and slacks when entering restoration (needed for bound mult step on exit)
    saved_z_L = jnp.where(init_resto, result.it.z_L[:cp.nxL], result.saved_z_L)
    saved_z_U = jnp.where(init_resto, result.it.z_U, result.saved_z_U)
    saved_v_L = jnp.where(init_resto, result.it.v_L, result.saved_v_L)
    saved_v_U = jnp.where(init_resto, result.it.v_U, result.saved_v_U)
    curr_slacks = cp.nstqf.calc_slacks(result.it)
    saved_slacks = jax.lax.cond(init_resto, lambda: curr_slacks, lambda: result.saved_slacks)

    rargs, rmu = initialize_resto_args(result, cp)
    rit = initialize_iterate_resto(result.it, cp, rmu, result.cqpr)
    ric = initialize_inertia_correction_state(cp)
    rmu = jnp.atleast_2d(rmu)

    # --- Init regular: scaling + push_to_interior + mults ---
    # Use pre-execute_search x for init_regular. When hot-restarting, execute_search
    # runs with a stale search direction (from the previous converged solve) which moves
    # x from the intended restart x0 to a garbage point. Using original_state.it.x avoids
    # building the KKT system at this garbage point, which causes excessive IC iterations
    # that penalize the entire vmapped batch.
    ir_x0 = result.it.x[:cp.nx]
    f_args_ir, c_args_ir, d_args_ir = result.args
    # Reset scaling to 1.0 before computing new scaling (compute_scaling expects unscaled jacobians,
    # matching cold-start which uses fcd_default_scale=[1.0, 1.0, 1.0])
    f_args_unscaled = (*f_args_ir[:3], jnp.ones_like(f_args_ir[3]), *f_args_ir[4:])
    c_args_unscaled = (jnp.ones_like(c_args_ir[0]), *c_args_ir[1:])
    d_args_unscaled = (jnp.ones_like(d_args_ir[0]), *d_args_ir[1:])
    ir_jac_f = cp.nstqf.calc_jac_f(ir_x0, *f_args_unscaled)
    ir_jac_c = cp.nstqf.calc_jac_c(ir_x0, *c_args_unscaled)
    ir_jac_d = cp.nstqf.calc_jac_d(ir_x0, *d_args_unscaled)
    ir_df, ir_dc, ir_dd = compute_scaling(ir_jac_f, ir_jac_c, ir_jac_d, cp)
    ir_args = (
        (*f_args_ir[:3], ir_df, *f_args_ir[4:]),
        (ir_dc, *c_args_ir[1:]),
        (ir_dd, *d_args_ir[1:]),
    )

    # Slacks + push to interior
    ir_x = ir_x0
    _, _, ir_d_args = ir_args
    ir_s = jnp.max(jnp.hstack([
        cp.nstqf.calc_d(ir_x, *ir_d_args),
        cp.p["Ktol"] * jnp.ones([cp.nyd, 1])
    ]), axis=1, keepdims=True)
    ir_x, ir_s = push_to_interior(ir_x, ir_s, cp)

    # Mults
    ir_z_L = jnp.vstack([jnp.ones([cp.nxL, 1]), jnp.zeros([cp.nyc*2+cp.nyd*2, 1])])
    ir_z_U = jnp.ones([cp.nxU, 1])
    ir_v_L = jnp.ones([cp.ndL, 1])
    ir_v_U = jnp.ones([cp.ndU, 1])
    ir_x_padded = jnp.vstack([ir_x, jnp.zeros([cp.nyc*2+cp.nyd*2, 1])])
    ir_it = Iterate(ir_x_padded, ir_s, jnp.zeros([cp.nyc,1]), jnp.zeros([cp.nyd,1]),
                    ir_z_L, ir_z_U, ir_v_L, ir_v_U)
    ir_ic = initialize_inertia_correction_state(cp)
    ir_mu = cp.nstqf.calc_avrg_compl(ir_it, cp.nstqf.calc_slacks(ir_it))  # mu = avrg_compl

    # only if we are instantiating a new resto do we change the it, ic, mu, args...
    it, ic, mu, args = filter_select(
        init_regular,
        (ir_it, ir_ic, ir_mu, ir_args),
        filter_select(
            init_resto, (rit, ric, rmu, rargs),
            (result.it, result.ic, result.mu, result.args)
        )
    )
    # it, ic, mu, args = filter_select(init_resto, (rit, ric, rmu, rargs), (result.it, result.ic, result.mu, result.args))
    f_args, c_args, d_args = args
    x_ref, dr_x = f_args[1:3]

    # perform expensive sparse function evaluations
    rnx = cp.nx + cp.nyc * 2 + cp.nyd * 2

    # padded regular jacobians
    pad_rows = cp.nyc * 2 + cp.nyd * 2 
    reg_jac_f_unpadded = cp.nstqf.calc_jac_f(it.x[:cp.nx], *f_args).todense()
    reg_jac_c_unpadded = cp.nstqf.calc_jac_c(it.x[:cp.nx], *c_args)
    reg_jac_d_unpadded = cp.nstqf.calc_jac_d(it.x[:cp.nx], *d_args)
    reg_jac_f = jnp.vstack([reg_jac_f_unpadded, jnp.zeros([cp.nyc*2+cp.nyd*2,1])])
    # reg_jac_f = spu.vstack([reg_jac_f_unpadded, spu.ones_with_nse([pad_rows, 1], cp.nse_rJf - cp.nse_Jf)]) # pad the regular jacobians
    reg_jac_c = spu.vstack([reg_jac_c_unpadded, spu.ones_with_nse([pad_rows, cp.nyc], cp.nse_rJc - cp.nse_Jc)]) # pad the regular jacobians
    reg_jac_d = spu.vstack([reg_jac_d_unpadded, spu.ones_with_nse([pad_rows, cp.nyd], cp.nse_rJd - cp.nse_Jd)]) # pad the regular jacobians
    reg_f = jnp.atleast_2d(cp.nstqf.calc_f(it.x[:cp.nx], *f_args))
    reg_c = cp.nstqf.calc_c(it.x[:cp.nx], *c_args)
    reg_d = cp.nstqf.calc_d(it.x[:cp.nx], *d_args)

    # Compute original problem's inf-norm primal infeasibility for resto convergence check
    # This is IPOPT's orig_curr_inf_pr = curr_primal_infeasibility(NORM_MAX)
    reg_dms = reg_d - it.s[:cp.nyd]
    orig_inf_pr_at_entry = jnp.maximum(
        jnp.linalg.norm(reg_c, ord=jnp.inf),
        jnp.linalg.norm(reg_dms, ord=jnp.inf)
    )
    # Only store when entering restoration, otherwise keep existing value
    saved_orig_inf_pr = jnp.where(init_resto, orig_inf_pr_at_entry, result.saved_orig_inf_pr)

    # =========================================================================
    # EARLY restoration-exit decision
    # IPOPT decides leave-restoration at the START of the next iteration
    # (RestoConvergenceCheck::CheckConvergence on the freshly accepted point) and
    # then runs the ENTIRE regular iteration-start machinery on that point:
    # PerformRestoration's bound-multiplier step + least-squares y mults, then the
    # adaptive mu update (fixed->free switch-back + oracle + linesearch Reset) and
    # the regular quantity evaluation (IpRestoMinC_1Nrm.cpp:342-432,
    # IpAdaptiveMuUpdate.cpp:300-440). So the exit must be known BEFORE the
    # quantity/oracle pipeline below, and an exiting pass must run in REGULAR mode
    # over the cleaned iterate with the restored saved_* state. should_exit_resto
    # needs only the regular constraint values + the frozen saved_* (NOT cqpr), so
    # it is computed here; the local-infeasibility/tighten-tol outputs need the
    # resto convergence flag (cqpr-dependent) and come from the late call below.
    resto_tol = jnp.where(init_resto, jnp.atleast_2d(cp.p["tol"]), result.resto_tol)
    should_exit_resto, _, _, _, _ = check_resto_convergence(
        it, reg_c, reg_d, reg_f.squeeze(), saved_ls, saved_mu, saved_orig_inf_pr, cp,
        init_resto, jnp.array(False), resto_tol
    )
    should_exit_resto = should_exit_resto & (in_resto | init_resto) & ~init_regular
    exiting = should_exit_resto.squeeze()
    # Effective phase for THIS pass's quantities: an exiting pass is a regular pass.
    resto_mode = ((in_resto > 0) & ~exiting) | (init_resto > 0)

    # An exiting pass consumes the RESTORED regular bookkeeping: IPOPT's resumed
    # regular iteration starts from the pre-restoration mu/tau/filter/IC state (the
    # orig ip_data was frozen while the resto NLP ran). Restore BEFORE the oracle /
    # quantity pipeline so calc_updated_mu performs the fixed->free switch-back +
    # oracle over the restored state, instead of clobbering the oracle's output
    # with saved_* afterwards. (tau/mu_max/init_infs/fl/adfs/ls are restored at
    # their respective selection points further down, for the same reason.)
    mu = jnp.where(exiting, saved_mu, mu)
    ic = jax.lax.cond(exiting, lambda: saved_ic, lambda: ic)

    reg_hess_f_unpadded = cp.nstqf.calc_hess_f(it.x[:cp.nx], *f_args)
    reg_hess_c_unpadded = cp.nstqf.calc_hess_c(it.x[:cp.nx], *c_args)
    reg_hess_d_unpadded = cp.nstqf.calc_hess_d(it.x[:cp.nx], *d_args)
    reg_hess_f = jsparse.BCOO(
        (reg_hess_f_unpadded.data, reg_hess_f_unpadded.indices),
        shape=(rnx, rnx)
    )
    # conform to unified hess_f sparsity pattern
    reg_hess_f = spu.sum_duplicates_to_pattern(cp.hess_f_coo_indices, reg_hess_f, cp.hess_f_nnz)
    reg_hess_c = jsparse.BCOO(
        (reg_hess_c_unpadded.data, reg_hess_c_unpadded.indices),
        shape=(cp.nyc, rnx, rnx)
    )
    reg_hess_d = jsparse.BCOO(
        (reg_hess_d_unpadded.data, reg_hess_d_unpadded.indices),
        shape=(cp.nyd, rnx, rnx)
    )

    resto_f = jnp.atleast_2d(cp.nstqfr.calc_f(it.x.flatten(), mu, x_ref, dr_x))
    resto_c = cp.nstqfr.calc_c(reg_c, it.x.flatten())[:,None]
    resto_d = cp.nstqfr.calc_d(reg_d, it.x.flatten())[:,None]
    resto_jac_f = cp.nstqfr.calc_jac_f(it.x.flatten(), mu, x_ref, dr_x)
    resto_jac_c = cp.nstqfr.calc_jac_c(reg_jac_c_unpadded)
    resto_jac_d = cp.nstqfr.calc_jac_d(reg_jac_d_unpadded)
    resto_hess_f = cp.nstqfr.calc_hess_f(mu, dr_x)
    # conform to unified hess_f sparsity pattern
    resto_hess_f = spu.sum_duplicates_to_pattern(cp.hess_f_coo_indices, resto_hess_f, cp.hess_f_nnz)
    resto_hess_c = cp.nstqfr.calc_hess_c(reg_hess_c_unpadded)
    resto_hess_d = cp.nstqfr.calc_hess_d(reg_hess_d_unpadded)

    f, c, d, jac_f, jac_c, jac_d, hess_f, hess_c, hess_d = filter_select(
        resto_mode, ( # on_true
            resto_f,
            resto_c,
            resto_d,
            resto_jac_f,
            resto_jac_c,
            resto_jac_d,
            resto_hess_f,
            resto_hess_c,
            resto_hess_d,
        ), ( # on_false
            reg_f,
            reg_c,
            reg_d,
            reg_jac_f,
            reg_jac_c,
            reg_jac_d,
            reg_hess_f,
            reg_hess_c,
            reg_hess_d,
        )
    )
    dms = d - it.s

    # common operations
    fun_outs = (f, c, d)
    jacobians = (jac_f, jac_c, jac_d)
    hessians = (hess_f, hess_c, hess_d)

    # Step 1: Increment iteration counter
    # result = eqx.tree_at(lambda t: t.iter_count, result, result.iter_count + 1)
    iter_count = jnp.where(init_regular, jnp.array([[0]]), result.iter_count + 1) # result.iter_count + 1

    # Step 2: Correct bound multipliers (kappa_sigma correction)
    nxr = cp.nx + cp.nyc*2 + cp.nyd*2
    slacks_reg = cp.nstqf.calc_slacks(it)
    slacks_reg = (jnp.vstack([slacks_reg[0], jnp.zeros([nxr-cp.nx, 1])]), *slacks_reg[1:])
    slacks_resto = cp.nstqfr.calc_slacks(it)
    sxL, sxU, sdL, sdU = filter_select(resto_mode, slacks_resto, slacks_reg)

    kappa_sigma = cp.p["kappa_sigma"]
    # mu = result.mu

    def correct_bound_multiplier(z, slack):
        z_min = mu / (kappa_sigma * slack)
        z_max = kappa_sigma * mu / slack
        return jnp.clip(z, z_min, z_max)

    # Regular: only correct up to nxL, pad rest with ones
    z_L_reg = jnp.vstack([correct_bound_multiplier(it.z_L[:cp.nxL], sxL[:cp.nxL]), jnp.zeros([cp.nyc*2+cp.nyd*2, 1])])
    # Resto: correct all
    z_L_resto = correct_bound_multiplier(it.z_L, sxL)
    new_z_L = jnp.where(resto_mode, z_L_resto, z_L_reg)
    new_z_U = correct_bound_multiplier(it.z_U, sxU)
    new_v_L = correct_bound_multiplier(it.v_L, sdL)
    new_v_U = correct_bound_multiplier(it.v_U, sdU)

    it = eqx.tree_at(
        lambda t: (t.z_L, t.z_U, t.v_L, t.v_U), it,
        (jnp.where(init_regular, it.z_L, new_z_L),
        jnp.where(init_regular, it.z_U, new_z_U),
        jnp.where(init_regular, it.v_L, new_v_L),
        jnp.where(init_regular, it.v_U, new_v_U)) # (new_z_L, new_z_U, new_v_L, new_v_U)
    )

    # =========================================================================
    # Restoration-exit iterate surgery — BEFORE quantity evaluation, so the
    # regular cqpr/oracle below see the post-exit iterate exactly as IPOPT's
    # resumed regular iteration does (IpRestoMinC_1Nrm.cpp::PerformRestoration):
    #   1. drop the resto slack blocks (zero the x / z_L padding),
    #   2. bound-multiplier step: treat the whole restoration progress as one
    #      primal-dual Newton step (lines 374-399), with frac-to-the-bound at the
    #      SAVED tau and the SAVED entry mu/slacks/mults,
    #   3. reset all bound mults to 1.0 if any exceeds bound_mult_reset_threshold
    #      (lines 401-419),
    #   4. least-squares y_c/y_d (line 421) — done in the shared LSQ block below.
    # =========================================================================
    it = eqx.tree_at(
        lambda t: (t.x, t.z_L), it,
        (jnp.where(exiting, it.x.at[cp.nx:].set(0.0), it.x),
         jnp.where(exiting, it.z_L.at[cp.nxL:].set(0.0), it.z_L)),
    )

    ex_trial_sxL, ex_trial_sxU, ex_trial_sdL, ex_trial_sdU = cp.nstqf.calc_slacks(it)
    ex_saved_sxL, ex_saved_sxU, ex_saved_sdL, ex_saved_sdU = saved_slacks

    def compute_bound_mult_step(curr_z, curr_slack, trial_slack, mu_val):
        # delta_z = (mu - trial_slack * curr_z) / curr_slack
        return (mu_val - trial_slack * curr_z) / curr_slack

    ex_dz_L = compute_bound_mult_step(saved_z_L, ex_saved_sxL[:cp.nxL], ex_trial_sxL[:cp.nxL], saved_mu)
    ex_dz_U = compute_bound_mult_step(saved_z_U, ex_saved_sxU, ex_trial_sxU, saved_mu)
    ex_dv_L = compute_bound_mult_step(saved_v_L, ex_saved_sdL, ex_trial_sdL, saved_mu)
    ex_dv_U = compute_bound_mult_step(saved_v_U, ex_saved_sdU, ex_trial_sdU, saved_mu)

    def frac_to_bound(z, dz, tau_val):
        ratios = -tau_val * z / dz
        masked = jnp.where(dz < 0, ratios, jnp.inf)
        return jnp.minimum(1.0, jnp.min(masked))

    ex_alpha_z_L = frac_to_bound(saved_z_L, ex_dz_L, saved_tau) if cp.nxL > 0 else 1.0
    ex_alpha_z_U = frac_to_bound(saved_z_U, ex_dz_U, saved_tau) if cp.nxU > 0 else 1.0
    ex_alpha_v_L = frac_to_bound(saved_v_L, ex_dv_L, saved_tau) if cp.ndL > 0 else 1.0
    ex_alpha_v_U = frac_to_bound(saved_v_U, ex_dv_U, saved_tau) if cp.ndU > 0 else 1.0
    ex_alpha_dual = jnp.minimum(jnp.minimum(ex_alpha_z_L, ex_alpha_z_U),
                                jnp.minimum(ex_alpha_v_L, ex_alpha_v_U))

    ex_z_L = saved_z_L + ex_alpha_dual * ex_dz_L
    ex_z_U = saved_z_U + ex_alpha_dual * ex_dz_U
    ex_v_L = saved_v_L + ex_alpha_dual * ex_dv_L
    ex_v_U = saved_v_U + ex_alpha_dual * ex_dv_U
    ex_z_L_full = jnp.vstack([ex_z_L, jnp.zeros([cp.nyc*2 + cp.nyd*2, 1])])

    # Reset all bound multipliers to 1.0 if the stepped ones got too large
    ex_bound_mult_max = jnp.maximum(
        jnp.maximum(
            jnp.max(jnp.abs(ex_z_L)) if cp.nxL > 0 else 0.0,
            jnp.max(jnp.abs(ex_z_U)) if cp.nxU > 0 else 0.0
        ),
        jnp.maximum(
            jnp.max(jnp.abs(ex_v_L)) if cp.ndL > 0 else 0.0,
            jnp.max(jnp.abs(ex_v_U)) if cp.ndU > 0 else 0.0
        )
    )
    ex_reset = ex_bound_mult_max > cp.p["bound_mult_reset_threshold"]
    ex_z_L_full = jnp.where(ex_reset, jnp.ones_like(ex_z_L_full).at[cp.nxL:].set(0.0), ex_z_L_full)
    ex_z_U = jnp.where(ex_reset, jnp.ones_like(ex_z_U), ex_z_U)
    ex_v_L = jnp.where(ex_reset, jnp.ones_like(ex_v_L), ex_v_L)
    ex_v_U = jnp.where(ex_reset, jnp.ones_like(ex_v_U), ex_v_U)

    it = eqx.tree_at(
        lambda t: (t.z_L, t.z_U, t.v_L, t.v_U), it,
        (jnp.where(exiting, ex_z_L_full, it.z_L),
         jnp.where(exiting, ex_z_U, it.z_U),
         jnp.where(exiting, ex_v_L, it.v_L),
         jnp.where(exiting, ex_v_U, it.v_U)),
    )

    # Step 3: Update line search filter for h-type acceptances unconditionally - will conditionally wipe later if init_resto
    # IPOPT augments the filter (h-type) iff the accepted step is NOT (f-type switching AND Armijo):
    # FilterLSAcceptor::UpdateForNextIteration -> augment iff !IsFtype(alpha) || !ArmijoHolds(alpha).
    # NOTE: IsFtype here is the bare switching test (ref_gBD<0 & alpha*|ref_gBD|^s_phi > delta*ref_theta^s_theta);
    # it does NOT carry the (ref_theta<=theta_min) gate that acceptance uses. ArmijoHolds compares the accepted
    # trial barrier objective (evaluated at the accepted iterate result.it with the pre-update mu) against ref_barr.
    # The previous `theta_decreased` proxy wrongly augmented for f-type Armijo steps that also reduced theta
    # (e.g. iter 12/13), leaving stale filter entries IPOPT never adds.
    trial_barr_acc, = filter_select(
        resto_mode,
        (cp.nstqfr.calc_barrier_obj(resto_f, mu, slacks_resto),),
        (cp.nstqf.calc_barrier_obj(reg_f, mu, slacks_reg),),
    )
    alpha_acc = result.ls.alpha_pr
    acc_is_ftype = cp.stqf.is_ftype(result.ls.filter.ref_theta, result.ls.filter.ref_gBD, alpha_acc)
    acc_armijo = cp.stqf.armijo_holds(trial_barr_acc, result.ls.filter.ref_barr, result.ls.filter.ref_gBD, alpha_acc)
    is_h_type = (result.ls.accept & ~(acc_is_ftype & acc_armijo) & ~init_regular).squeeze()
    new_ls_F = cp.stqf.augment_ls_filter(
        result.ls.filter.F, result.ls.filter.ref_barr,
        result.ls.filter.ref_theta, result.iter_count
    )
    ls = filter_tree_at_select(is_h_type, lambda t: t.filter.F, result.ls, new_ls_F, result.ls.filter.F)

    # Step 4: unconditionally reset iteration-specific state
    ls = eqx.tree_at(
        lambda t: (t.n_steps, t.count_soc),
        ls, (jnp.array([[0]]), jnp.array([[0]]))
    )

    # For init_regular AND restoration exit: compute LS mults and apply to iterate
    # BEFORE quantities/KKT step. init_regular matches cold-start (LS solve before
    # the first KKT iteration), the exit matches IPOPT's least_square_mults call at
    # the end of PerformRestoration (IpRestoMinC_1Nrm.cpp:421). run AFTER the
    # bound-multiplier step above, whose z/v feed the LSQ RHS. The magnitude cutoff
    # differs by path: constr_mult_init_max (init) vs constr_mult_reset_threshold
    # (exit, IPOPT default 0 -> the LSQ y are computed but always reset to ZERO).
    ls_init_lhs = cp.stqf.calc_ls_mults_LHS(jacobians)
    ls_init_csr = jsparse.BCSR.from_bcoo(ls_init_lhs)
    ls_init_rhs = cp.stqf.calc_ls_mults_RHS(
        jacobians[0], it.z_L[:cp.nxL], it.z_U, it.v_L, it.v_U
    )
    ls_step = cp.ls_refactorize_and_solve(ls_init_rhs.flatten(), ls_init_csr.data)[0][:, None]
    y_c_ls = ls_step[cp.nx + cp.nyd : cp.nx + cp.nyd + cp.nyc]
    y_d_ls = ls_step[cp.nx + cp.nyd + cp.nyc : cp.nx + cp.nyd + cp.nyc + cp.nyd]
    yinitnrm = jnp.maximum(
        jnp.where(cp.nyc > 0, jnp.max(jnp.abs(y_c_ls), initial=0.0), 0.0),
        jnp.where(cp.nyd > 0, jnp.max(jnp.abs(y_d_ls), initial=0.0), 0.0)
    )
    _cm_reset_thresh = cp.p.get("constr_mult_reset_threshold", 0.0) if hasattr(cp.p, 'get') else 0.0
    ls_mult_cutoff = jnp.where(exiting, _cm_reset_thresh, cp.p['constr_mult_init_max'])
    y_c_ls = jnp.where(yinitnrm > ls_mult_cutoff, jnp.zeros_like(y_c_ls), y_c_ls)
    y_d_ls = jnp.where(yinitnrm > ls_mult_cutoff, jnp.zeros_like(y_d_ls), y_d_ls)
    use_ls_mults = init_regular | exiting
    it = eqx.tree_at(lambda t: (t.y_c, t.y_d), it,
        (jnp.where(use_ls_mults, y_c_ls, it.y_c),
         jnp.where(use_ls_mults, y_d_ls, it.y_d)))

    # setup restoration phase potential startup

    # RESTO INITIALIZE QUANTITIES - but allowing for no repeated calculations
    # theta = cp.nstqfr.calc_theta(c, dms)
    # grad_lag_x = cp.nstqfr.calc_grad_lag_x(it, jacobians)
    grad_lag_s = cp.stqf.calc_grad_lag_s(it)
    # slacks = cp.nstqfr.calc_slacks(it)

    pad = jnp.zeros([cp.nyc * 2 + cp.nyd * 2, 1])  # how much to pad by
    reg_grad_lag_x = jnp.vstack([cp.nstqf.calc_grad_lag_x(it, jacobians), pad])
    reg_slacks = cp.nstqf.calc_slacks(it)
    reg_slacks = (jnp.vstack([reg_slacks[0], jnp.zeros_like(pad)]), *reg_slacks[1:])
    rslacks = cp.nstqfr.calc_slacks(it)

    grad_lag_x, slacks, theta, avrg_compl = filter_select(
        resto_mode, (
            cp.nstqfr.calc_grad_lag_x(it, jacobians),
            rslacks,
            cp.nstqfr.calc_theta(c, dms),
            cp.nstqfr.calc_avrg_compl(it, rslacks)
        ), (
            reg_grad_lag_x,
            reg_slacks,
            cp.nstqf.calc_theta(c, dms),
            cp.nstqf.calc_avrg_compl(it, reg_slacks)
        )
    )
    quantities = grad_lag_x, slacks, theta, avrg_compl, grad_lag_s
    
    r_init_initial_average_complementarity = avrg_compl
    r_init_mu = r_init_initial_average_complementarity

    # Clip to at least 1.0 to avoid division by zero when initial point is feasible
    # (matches IPOPT's Max(1.0, ...) pattern in AdaptiveMuUpdate::lower_mu_safeguard)
    r_init_dual_inf = jnp.maximum(1.0, cp.nstqfr.calc_dual_inf(grad_lag_x, grad_lag_s, ord=1))
    r_init_primal_inf = jnp.maximum(1.0, cp.nstqfr.calc_primal_inf_L1(c, dms))

    # theta_max_fact = 1e8 (NOT the regular 1e4): IPOPT defaults the RESTORATION
    # phase's theta_max_fact to 1e8 (IpRestoMinC_1Nrm.cpp:86-91, "resto.theta_max_fact"),
    # giving a deliberately permissive constraint-violation ceiling during restoration.
    # The regular phase keeps 1e4. theta_min_fact is unchanged (1e-4) in both.
    r_init_theta_max = jnp.atleast_2d(
        1e8 * jnp.maximum(1, theta)
    )  # we cannot ONLY calculate this here as if tiny step is instantiated immediately

    # then we cannot instantiate theta min/max until we have escaped tiny step or converged
    r_init_theta_min = jnp.atleast_2d(1e-4 * jnp.maximum(1, theta))

    r_init_mu_max = cp.p["mu_max_fact"] * r_init_initial_average_complementarity
    r_init_mu_max = jnp.clip(r_init_mu_max, min=None, max=cp.p["mu_max_upper"])
    r_init_F = jnp.vstack([jnp.hstack([r_init_theta_max, cp.p["varphi_max"]])] * cp.p["filter_size"])
    r_init_tau = jnp.maximum(cp.p["tau_min"], 1 - r_init_mu)
    r_init_F_iter = jnp.arange(-cp.p["filter_size"], 0)[:, None]
    r_init_F = jnp.hstack([r_init_F_iter, r_init_F])
    r_init_fl = initialize_iterate_flags_state()
    r_init_fl = eqx.tree_at(lambda t: t.in_restoration, r_init_fl, jnp.array([[1]]))

    ir_init_dual_inf = jnp.maximum(1.0, cp.nstqf.calc_dual_inf(grad_lag_x, grad_lag_s, ord=1))
    ir_init_primal_inf = jnp.maximum(1.0, cp.nstqf.calc_primal_inf_L1(c, dms))
    ir_mu_max = jnp.clip(cp.p["mu_max_fact"] * avrg_compl, max=cp.p["mu_max_upper"])
    ir_tau = jnp.maximum(cp.p["tau_min"], 1 - mu)
    ir_fl = initialize_iterate_flags_state()  # all flags zeroed, needs_regular_init=0
    ir_fl = eqx.tree_at(lambda t: t.needs_regular_init, ir_fl, jnp.array([[1]]))

    # tau, mu_max, init_dual_inf, init_primal_inf, fl = filter_select(
    #     init_regular,
    #     (ir_tau, ir_mu_max, ir_init_dual_inf, ir_init_primal_inf, ir_fl),
    #     filter_select(init_resto, (r_init_*, ...), (result.*, ...))
    # )

    tau, mu_max, init_dual_inf, init_primal_inf, fl = filter_select(
        init_regular,
        (ir_tau, ir_mu_max, ir_init_dual_inf, ir_init_primal_inf, ir_fl),
        filter_select(
            init_resto, (
                r_init_tau,
                r_init_mu_max,
                r_init_dual_inf,
                r_init_primal_inf,
                r_init_fl
            ), (
                result.tau,
                result.mu_max,
                result.init_dual_inf,
                result.init_primal_inf,
                result.fl
            )
        )
    )

    # Restoration exit: restore the frozen regular bookkeeping BEFORE the oracle.
    # saved_fl carries the entry pass's needs_resto_init=1; the flag-clear a few
    # lines below wipes it, so the restored state cannot re-trigger resto init.
    tau = jnp.where(exiting, saved_tau, tau)
    mu_max = jnp.where(exiting, saved_mu_max, mu_max)
    init_dual_inf = jnp.where(exiting, saved_init_dual_inf, init_dual_inf)
    init_primal_inf = jnp.where(exiting, saved_init_primal_inf, init_primal_inf)
    fl = jax.lax.cond(exiting, lambda: saved_fl, lambda: fl)

    # Step 5: Recalculate pre-mu quantities (it, ic, cp, fl, fun_outs, jacobians, hessians, iter_count)
    cqpr, ic = calc_values_pre_mu(it, ic, cp, fl, fun_outs, jacobians, hessians, quantities, iter_count)
    # clear the flag now we have the LS direction
    fl = eqx.tree_at(
        lambda t: (t.needs_resto_init, t.needs_regular_init),
        fl, (jnp.array([[0]]), jnp.maximum(0, fl.needs_regular_init - 1))
    )
    # Override init_dual_inf/init_primal_inf for init_regular with values at the reset iterate
    init_dual_inf = jnp.where(init_regular, ir_init_dual_inf, init_dual_inf)
    init_primal_inf = jnp.where(init_regular, ir_init_primal_inf, init_primal_inf)

    # result = eqx.tree_at(lambda t: (t.cqpr, t.ic), result, (cqpr, ic))

    r_init_adfs = initialize_adaptive_mu_filter_state(cp)
    r_init_lsfs = initialize_line_search_filter_state(r_init_theta_min, r_init_theta_max, r_init_F, cqpr)
    r_init_ls = initialize_line_search_state(it, cqpr, r_init_lsfs, cp)

    ir_theta_max = jnp.atleast_2d(1e4 * jnp.maximum(1, cqpr.theta))
    ir_theta_min = jnp.atleast_2d(1e-4 * jnp.maximum(1, cqpr.theta))
    ir_F = jnp.hstack([jnp.arange(-cp.p["filter_size"], 0)[:, None],
                        jnp.vstack([jnp.hstack([ir_theta_max, cp.p["varphi_max"]])] *
    cp.p["filter_size"])])
    ir_adfs = initialize_adaptive_mu_filter_state(cp)
    ir_lsfs = initialize_line_search_filter_state(ir_theta_min, ir_theta_max, ir_F,
    cqpr)
    ir_ls = initialize_line_search_state(it, cqpr, ir_lsfs, cp)

    adfs, ls = filter_select(
        init_regular, (ir_adfs, ir_ls),
        filter_select(init_resto, (r_init_adfs, r_init_ls), (result.adfs, ls))
    )
    # Restoration exit: the oracle must see the RESTORED original filter state
    # (saved adfs + saved W-B filter incl. theta_max/min). If the mu update then
    # switches back to free mode, its own linesearch-Reset clears the filter
    # contents, exactly as IPOPT does at the first resumed regular iteration.
    adfs = jnp.where(exiting, saved_adfs, adfs)
    ls = filter_select(exiting, (saved_ls,), (ls,))[0]

    # adfs, ls = filter_select(
    #     init_resto, (
    #         r_init_adfs, r_init_ls
    #     ), (
    #         result.adfs, ls # already modified ls
    #     )
    # )

    # Step 6: Update barrier parameter (mu)
    df_ = args[0][3]
    mu, tau, ls, adfs, terminate_from_mu, new_free_mu_mode = calc_updated_mu(
        it, cqpr, mu, tau,
        mu_max, init_dual_inf, init_primal_inf, df_,
        fl, adfs, ls, cp, iter_count
    )

    # ls/adfs must be fresh for init_regular (clean filter state for new problem)
    ls = filter_select(init_regular, (ir_ls,), (ls,))[0]
    adfs = jnp.where(init_regular, ir_adfs, adfs)
    terminate_from_mu = jnp.where(init_regular, jnp.array([[0]]), terminate_from_mu)
    new_free_mu_mode = jnp.where(init_regular, jnp.array([[0]]), new_free_mu_mode)

    # Update fl.free_mu_mode with the new value from adaptive mu control flow
    fl = eqx.tree_at(lambda t: t.free_mu_mode, fl, new_free_mu_mode)
    # result = eqx.tree_at(lambda t: (t.mu, t.tau, t.ls, t.adfs), result, (mu, tau, ls, adfs))

    # =========================================================================
    # Restoration convergence check (secondary outputs only) — should_exit_resto
    # itself was decided EARLY (before the quantity/oracle pipeline above), since
    # an exiting pass must run as a regular pass. This late call only supplies the
    # cqpr-dependent local-infeasibility / tighten-tol outcomes; its should_exit
    # output is identical to the early one (the converged flag does not enter it)
    # and is discarded.
    # =========================================================================
    # first_resto_iter is true when we just initialized restoration (haven't taken a step yet)
    first_resto_iter = init_resto

    # Compute resto convergence check (uses cqpr values, available before cqpo)
    # Use current mu for convergence check (consistent with IPOPT checking at iteration start)
    converged_resto_early, _ = cp.nstqfr.calc_check_converged(
        it, mu, cqpr.grad_lag_x, cqpr.grad_lag_s,
        cqpr.slacks, cqpr.c, cqpr.d, args
    )

    (
        _,
        is_locally_infeasible,
        is_feasible_filter_rejects,
        should_tighten_tol,
        is_acceptable_convergence
    ) = check_resto_convergence(
        it, reg_c, reg_d, reg_f.squeeze(), saved_ls, saved_mu, saved_orig_inf_pr, cp,
        first_resto_iter, converged_resto_early, resto_tol
    )

    # Only apply resto exit logic if we're actually in restoration
    is_locally_infeasible = is_locally_infeasible & (in_resto | init_resto) & ~init_regular
    is_feasible_filter_rejects = is_feasible_filter_rejects & (in_resto | init_resto) & ~init_regular
    is_acceptable_convergence = is_acceptable_convergence & (in_resto | init_resto) & ~init_regular
    should_tighten_tol = should_tighten_tol & (in_resto | init_resto) & ~init_regular

    # Tighten resto_tol when should_tighten_tol is true (multiply by 0.01)
    # IPOPT: IpData().Set_tol(1e-2 * IpData().tol())
    resto_tol = jnp.where(should_tighten_tol, resto_tol * 1e-2, resto_tol)

    # Re-evaluate the restoration objective with the POST-oracle mu so cqpr.f/jac_f
    # (and downstream cqpo.barr/gBD, ls.filter.ref_barr/ref_gBD) reflect THIS iteration's
    # barrier parameter, matching IPOPT (its mu oracle runs BEFORE it evaluates the resto
    # objective; IpRestoIpoptNLP.cpp Eta uses the current mu). The regular objective is
    # mu-independent, so this only matters in restoration. calc_updated_mu above already
    # consumed the pre-oracle cqpr for its affine predictor (which matches IPOPT's predictor,
    # also computed with the pre-update mu), so we patch strictly AFTER it and BEFORE cqpo.
    # The pre-oracle versions are computed at ~1490/1493; here we only swap in the updated mu.
    # Gate to CONTINUING restoration: the regular phase and the restoration-entry pass
    # (init_resto, which has its own mu init and already matches) are left untouched. Only
    # cqpr.f/jac_f are patched -> cqpo's step RHS (grad_lag_x/c) and LHS (hess_f) are
    # unchanged, so the validated it.x trajectory does not move.
    _use_pm_mu = (in_resto > 0) & ~exiting & (init_resto == 0) & (~init_regular)
    resto_f_pm = jnp.atleast_2d(cp.nstqfr.calc_f(it.x.flatten(), mu, x_ref, dr_x))
    resto_jac_f_pm = cp.nstqfr.calc_jac_f(it.x.flatten(), mu, x_ref, dr_x)
    cqpr = eqx.tree_at(
        lambda t: (t.f, t.jac_f),
        cqpr,
        (
            jnp.where(_use_pm_mu, resto_f_pm, cqpr.f),
            jnp.where(_use_pm_mu, resto_jac_f_pm, cqpr.jac_f),
        ),
    )

    # Step 7: Recalculate post-mu quantities (LS handled by dedicated solver in calc_values_pre_mu)
    cqpo = calc_values_post_mu(it, mu, tau, cqpr, ic, cp, fl)
    # result = eqx.tree_at(lambda t: t.cqpo, result, cqpo)

    # Step 8: Update line search state for next iteration
    ls = post_initialize_line_search(ls, cqpo, cqpr, cp)
    # result = eqx.tree_at(lambda t: t.ls, result, )

    # Step 9: Check convergence (for termination decision)
    converged_reg, _ = cp.nstqf.calc_check_converged(
        it, mu, cqpr.grad_lag_x, cqpr.grad_lag_s,
        cqpr.slacks, cqpr.c, cqpr.d, args
    )
    # Reuse converged_resto_early for consistency
    converged_resto = converged_resto_early
    converged = jnp.where(in_resto | init_resto, converged_resto, converged_reg)
    # Suppress convergence when exiting restoration — element returns to regular algorithm,
    # not declaring overall convergence. Restoration convergence != original problem convergence.
    converged = converged & ~should_exit_resto.squeeze()
    max_iter_exceeded = iter_count >= cp.p["max_iter"]

    # =========================================================================
    # When exiting restoration, the regular bookkeeping (mu/tau/ls/adfs/fl/ic/
    # mu_max/init_infs) was already restored BEFORE the oracle, and the iterate
    # surgery (pad zeroing, bound-multiplier step, LSQ y mults) ran BEFORE the
    # quantity evaluation — see the exit blocks above. Only the watchdog state,
    # which nothing upstream consumes, is restored here.
    # =========================================================================
    wd = jax.lax.cond(exiting, lambda: saved_wd, lambda: result.wd)
    ir_wd = initialize_watchdog(mu, it, cqpr, cqpo)
    wd = jax.lax.cond(init_regular, lambda: ir_wd, lambda: wd)

    # Set in_restoration = 0 when exiting (the restored saved_fl already carries 0;
    # kept as an explicit invariant)
    fl = eqx.tree_at(lambda t: t.in_restoration, fl,
                     jnp.where(exiting, jnp.array([[0]]), fl.in_restoration))

    # Termination codes:
    # 0 = continue
    # 1 = converged
    # 2 = max_iter_exceeded
    # 3 = locally_infeasible (restoration converged but problem is infeasible)
    # 4 = feasible_but_filter_rejects (restoration found feasible point unacceptable to filter)
    # 5 = acceptable_convergence (for square problems)
    terminate = jax.lax.select(
        (terminate_from_mu.squeeze() > 0),
        terminate_from_mu,
        jax.lax.select(is_locally_infeasible.squeeze(), jnp.array([[3]]),
            jax.lax.select(is_feasible_filter_rejects.squeeze(), jnp.array([[4]]),
                jax.lax.select(is_acceptable_convergence.squeeze(), jnp.array([[5]]),
                    jax.lax.select(converged.squeeze(), jnp.array([[1]]),
                        jax.lax.select(max_iter_exceeded.squeeze(), jnp.array([[2]]), jnp.array([[0]])))))))

    # suppress termination if initializing regular
    terminate = jnp.where(init_regular, jnp.array([[0]]), terminate)

    if cp.p["hot_restarting"]:
        should_restart = (terminate > 0).squeeze()
        fl = eqx.tree_at(lambda t: t.needs_regular_init, fl,
            jnp.where(should_restart, jnp.array([[1]]), fl.needs_regular_init))
    else:
        pass # should_restart = result.iter_count == 0  # only restart on first iteration (after initialization)


    # JOHN YOU ARE HERE - we need to do a couple things now that we have proven the 
    # algorithm is correctly instantiating with the new warm starting code. 
    # 1. we need to ensure that on termination, and we start the flag, that we skip the
    # next execute search, and then run the initialization in the next post_process, and then at
    # the end of that next post_process we reset the flag for needs_regular_init back to 0
    # 2. we need to make the termination criterion, which adds the data to the 
    # throughput solver, only add it if we converged successfully.

    # wd is already set above (either restored from saved_wd or kept as result.wd)

    processed_result = OptimizationState(
        it=it, cqpr=cqpr, cqpo=cqpo, fl=fl, wd=wd, ls=ls, ic=ic, adfs=adfs,
        saved_fl=saved_fl, saved_wd=saved_wd, saved_ls=saved_ls, saved_ic=saved_ic,
        saved_adfs=saved_adfs, saved_mu=saved_mu, saved_tau=saved_tau, saved_mu_max=saved_mu_max,
        saved_init_dual_inf=saved_init_dual_inf, saved_init_primal_inf=saved_init_primal_inf,
        saved_orig_inf_pr=saved_orig_inf_pr,
        saved_z_L=saved_z_L, saved_z_U=saved_z_U, saved_v_L=saved_v_L, saved_v_U=saved_v_U,
        saved_slacks=saved_slacks,
        resto_tol=resto_tol,
        mu=mu, tau=tau, mu_max=mu_max, init_dual_inf=init_dual_inf, init_primal_inf=init_primal_inf,
        iter_count=iter_count, args=args
    )

    return processed_result, terminate

if __name__ == "__main__":
    

    # do the setup for validation ----------------------------------------------
    from jaxipm.utils.validation_boilerplate import *
    # jax.config.update("jax_log_compiles", False)

    # if os.path.exists("ipopt_logs"):
    #     shutil.rmtree("ipopt_logs")

    # result = minimize_ipopt(
    #     fun=obj,
    #     x0=x0,
    #     jac=obj_grad,
    #     hess=obj_hess,
    #     constraints=constraints,
    #     bounds=bounds,
    #     options={
    #         "disp": 5,
    #         "maxiter": 10,
    #         # "mu_strategy": "monotone", # forces filter entries to happen pretty much
    #         # "start_with_resto": "yes",  # lets us test feas resto initialization as well
    #     },
    # )

    # common problem across entire batch
    cp = initialize_common_problem(
        f, c, d, x_L, x_U, d_L, d_U, x0, p, [f_args, c_args, d_args]
    )

    # state = initialize_problem_regular(cp, x0, args=[f_args, c_args, d_args])

    # def validate_directions(i, cq, it, resto=False):
    #     if resto is False: # trim off resto parts as needed
    #         it = eqx.tree_at(lambda t: t.x, it, it.x[: cp.nx, :])  
    #         it = eqx.tree_at(lambda t: t.z_L, it, it.z_L[: cp.nxL, :])  
    #         cq = eqx.tree_at(lambda t: t.step.x, cq, cq.step.x[: cp.nx, :])  
    #         cq = eqx.tree_at(lambda t: t.step.z_L, cq, cq.step.z_L[: cp.nxL, :]) 
    #     components = ["x", "s", "y_c", "y_d", "z_L", "z_U", "v_L", "v_U"]
    #     ref_step = Iterate(
    #         *[load_vector(f"{save_dir}/delta_{i}_{c}.txt") for c in components]
    #     )
    #     # ref_step = eqx.tree_at(lambda t: (t.x, t.s), ref_step, (cq.step.x.flatten(), cq.step.s.flatten()))
    #     diff_step = (jax.tree.map(jnp.ravel, cq.step) ** ω - ref_step**ω).ω
    #     diff_step_norm = jnp.linalg.norm(jnp.hstack(jax.tree.leaves(diff_step)))
    #     print(f"diff step norm: {diff_step_norm}")
    #     # load up the current iterates
    #     ref_it = Iterate(*[load_vector(f"{save_dir}/iterate_{i}_{c}.txt") for c in components])

    #     diff_it = (jax.tree.map(jnp.ravel, it) ** ω - ref_it**ω).ω
    #     diff_it_norm = jnp.linalg.norm(jnp.hstack(jax.tree.leaves(diff_it)))
    #     print(f"diff it norm: {diff_it_norm}")

    # # verify directions for the regular initialized iterate (E2E test)
    # validate_directions(i=0, cq=state.cqpo, it=state.it, resto=False)

    # execute_search(state, cp)

    from jaxipm.tests.load_ipopt_state_regular import load_state, align_reference_state, analyze_diff

    num_iters = len(glob.glob(f'{save_dir}/iteration_type_*.txt'))
    jax.config.update("jax_log_compiles", False)

    for iter_num in range(2, num_iters):  # Start from 2 to ensure both iter-1 and iter have complete data

        state = load_state(iter_num-1, cp)

        state_next = execute_search(state, cp)
        state_next, terminate = post_process(state, state_next, cp)

        ref_state_next = load_state(iter_num, cp)
        ref_state_next = align_reference_state(state_next, ref_state_next, resto=False, soc=False, cp=cp)

        test = analyze_diff(state_next, ref_state_next)# , path_filter="it")

        

        pass

    pass

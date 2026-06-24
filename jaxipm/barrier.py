"""
The adaptive mu update algorithm.

TODO outline algorithm in full for free and fixed modes and control flow between
them both.
"""

import jax
import jax.numpy as jnp
from equinox.internal import ω
from jaxipm.utils.eqx_utils import filter_select_n, filter_select
from jaxipm.structures import IterateFlags


def golden_section_search(
    sigma_up_in, q_up_in, sigma_lo_in, q_lo_in,
    sigma_tol, qf_tol, max_steps,
    calculate_quality_function,
):

    gfac = (3. - jnp.sqrt(5.)) / 2.

    # Initial interior points
    sigma_mid1 = sigma_lo_in + gfac * (sigma_up_in - sigma_lo_in)
    sigma_mid2 = sigma_lo_in + (1. - gfac) * (sigma_up_in - sigma_lo_in)
    qmid1 = jnp.atleast_2d(calculate_quality_function(sigma_mid1))
    qmid2 = jnp.atleast_2d(calculate_quality_function(sigma_mid2))

    init_carry = (sigma_lo_in, sigma_up_in, sigma_mid1, sigma_mid2,
                  q_lo_in, q_up_in, qmid1, qmid2, jnp.array(0))

    def cond_fn(carry):
        sigma_lo, sigma_up, _, _, q_lo, q_up, qmid1, qmid2, nsections = carry
        sigma_cond = (sigma_up - sigma_lo) >= sigma_tol * sigma_up
        q_vals = jnp.array([q_lo, q_up, qmid1, qmid2])
        q_vals_clean = jnp.where(q_vals < 0, jnp.inf, q_vals)
        q_min = jnp.min(q_vals_clean)
        q_max = jnp.max(jnp.where(q_vals < 0, -jnp.inf, q_vals))
        qf_cond = (1. - q_min / q_max) >= qf_tol
        return (sigma_cond & qf_cond & (nsections < max_steps)).squeeze()

    def body_fn(carry):
        sigma_lo, sigma_up, sigma_mid1, sigma_mid2, q_lo, q_up, qmid1, qmid2, nsections = carry

        def eliminate_left(_):
            # Keep [sigma_mid1, sigma_up]
            new_lo, new_q_lo = sigma_mid1, qmid1
            new_mid1, new_qmid1 = sigma_mid2, qmid2
            new_mid2 = new_lo + (1. - gfac) * (sigma_up - new_lo)
            new_qmid2 = jnp.atleast_2d(calculate_quality_function(new_mid2))
            return (new_lo, sigma_up, new_mid1, new_mid2, new_q_lo, q_up, new_qmid1, new_qmid2)

        def eliminate_right(_):
            # Keep [sigma_lo, sigma_mid2]
            new_up, new_q_up = sigma_mid2, qmid2
            new_mid2, new_qmid2 = sigma_mid1, qmid1
            new_mid1 = sigma_lo + gfac * (new_up - sigma_lo)
            new_qmid1 = jnp.atleast_2d(calculate_quality_function(new_mid1))
            return (sigma_lo, new_up, new_mid1, new_mid2, q_lo, new_q_up, new_qmid1, new_qmid2)

        result = jax.lax.cond((qmid1 > qmid2).squeeze(), eliminate_left, eliminate_right, None)
        return (*result, nsections + 1)

    final = jax.lax.while_loop(cond_fn, body_fn, init_carry)
    sigma_lo, sigma_up, sigma_mid1, sigma_mid2, q_lo, q_up, qmid1, qmid2, _ = final

    # Final selection
    q_vals = jnp.array([q_lo, q_up, qmid1, qmid2])
    sigma_vals = jnp.array([sigma_lo, sigma_up, sigma_mid1, sigma_mid2])
    q_vals_clean = jnp.where(q_vals < 0, jnp.inf, q_vals)
    q_min, q_max = jnp.min(q_vals_clean), jnp.max(jnp.where(q_vals < 0, -jnp.inf, q_vals))

    sigma_range_large = (sigma_up - sigma_lo) >= sigma_tol * sigma_up
    qf_converged = (1. - q_min / q_max) < qf_tol

    def qf_tolerance_case(_):
        # QF converged - select minimum QF point
        return sigma_vals[jnp.argmin(q_vals_clean)]

    def sigma_tolerance_case(_):
        # Select best interior, then check boundaries
        sigma_int, q_int = jax.lax.cond(
            (qmid1 < qmid2).squeeze(),
            lambda _: (sigma_mid1, qmid1),
            lambda _: (sigma_mid2, qmid2),
            None
        )

        # Check upper boundary if untouched
        q_up_eval = jax.lax.cond(
            (q_up_in < 0).squeeze(),
            lambda _: jnp.atleast_2d(calculate_quality_function(sigma_up_in)),
            lambda _: q_up_in,
            None
        )
        sigma_curr, q_curr = jax.lax.cond(
            ((sigma_up == sigma_up_in) & (q_up_eval < q_int)).squeeze(),
            lambda _: (sigma_up_in, q_up_eval),
            lambda _: (sigma_int, q_int),
            None
        )

        # Check lower boundary if untouched
        q_lo_eval = jax.lax.cond(
            (q_lo_in < 0).squeeze(),
            lambda _: jnp.atleast_2d(calculate_quality_function(sigma_lo_in)),
            lambda _: q_lo_in,
            None
        )
        return jax.lax.cond(
            ((sigma_lo == sigma_lo_in) & (q_lo_eval < q_curr)).squeeze(),
            lambda _: sigma_lo_in,
            lambda _: sigma_curr,
            None
        )

    sigma = jax.lax.cond(
        (sigma_range_large & qf_converged).squeeze(),
        qf_tolerance_case,
        sigma_tolerance_case,
        None
    )

    return sigma

# calculates duplicate values as this is only done once at the start
def calc_init_mu(it, nstqf):
    slacks = nstqf.calc_slacks(it)
    return nstqf.calc_avrg_compl(it, slacks)

def calc_quality_function_mu(cp, it, tau, cqpr, mu_max, init_dual_inf, init_primal_inf, fl, mu_min):

    assert isinstance(fl, IterateFlags)

    rnx = cp.nx + cp.nyc * 2 + cp.nyd * 2
    rnxL = cp.nxL + cp.nyc * 2 + cp.nyd * 2

    # When no complementarity components, use safe defaults that skip the mu adaptation
    def calc_quality_function(sigma):
        """we want to minimize this quality function"""
        tmp_step = cqpr.step_aff_full + sigma * cqpr.step_cen_full  # although we dont use c and d terms...
        tmp_step = cp.stqf.calc_vector_to_iterate(tmp_step)

        tmp = cp.nstqf.calc_slack_derivatives(tmp_step)
        reg_tmp_slack_derivatives = (
            jnp.vstack([tmp[0], jnp.zeros([cp.nyc*2+cp.nyd*2,1])]), *tmp[1:]
        )

        tmp_slack_derivatives = filter_select(
            fl.in_restoration.squeeze(), 
            [cp.nstqfr.calc_slack_derivatives(tmp_step)],
            [reg_tmp_slack_derivatives]
        )[0]

        alpha_pr, alpha_du = filter_select(
            fl.in_restoration.squeeze(), (
                cp.nstqfr.calc_alpha_pr(tau, cqpr.slacks, tmp_slack_derivatives),
                cp.nstqfr.calc_alpha_du(it, tmp_step, tau),
            ), (
                cp.nstqf.calc_alpha_pr(tau, cqpr.slacks, tmp_slack_derivatives),
                cp.nstqf.calc_alpha_du(it, tmp_step, tau),
            )
        )

        # compute new terms for tmp step on the FLY
        slacks = cqpr.slacks # (cqpr.slacks[0][:nxL], *cqpr.slacks[1:]) # truncate the z_L slack if not in resto
        tmp_slack = (slacks**ω + alpha_pr * tmp_slack_derivatives**ω).ω

        tmp_mults = (
            (it.z_L, it.z_U, it.v_L, it.v_U) ** ω
            + alpha_du * (tmp_step.z_L, tmp_step.z_U, tmp_step.v_L, tmp_step.v_U) ** ω
        ).ω
        tmp_compl = (tmp_slack**ω * tmp_mults**ω).ω

        # infeasibilities are calculated with 2-norm here - different to inf norm in other spots...
        dual_inf = (1.0 - alpha_du) ** 2 * (cqpr.grad_lag_x_nrm2**2 + cqpr.grad_lag_s_nrm2**2)
        primal_inf = (1.0 - alpha_pr) ** 2 * (cqpr.c_nrm2**2 + cqpr.d_minus_s_nrm2**2)

        # ensure we don't use padded ones in slack to miscount compl_inf
        reg_compl_inf_0 = jnp.sum(tmp_compl[0][:cp.nxL] ** 2)
        resto_compl_inf_0 = jnp.sum(tmp_compl[0] ** 2)
        compl_res = jnp.sum(tmp_compl[1] ** 2) \
            + jnp.sum(tmp_compl[2] ** 2) \
            + jnp.sum(tmp_compl[3] ** 2)
        compl_inf_reg = reg_compl_inf_0 + compl_res
        compl_inf_resto = resto_compl_inf_0 + compl_res
        compl_inf = jnp.where(fl.in_restoration.squeeze(), compl_inf_resto, compl_inf_reg)

        # ensure we dont divide by zero...
        nx = jnp.where(fl.in_restoration.squeeze(), rnx, cp.nx)
        dual_inf /= nx + cp.nyd
        if cp.nyc + cp.nyd > 0:
            primal_inf /= cp.nyc + cp.nyd
        # In resto mode, ncomps includes nc, pc, nd, pd lower bounds (rnxL vs nxL)
        ncomps_reg = cp.nxL + cp.nxU + cp.ndL + cp.ndU
        ncomps_resto = rnxL + cp.nxU + cp.ndL + cp.ndU
        ncomps = jnp.where(fl.in_restoration.squeeze(), ncomps_resto, ncomps_reg)
        if cp.nxL + cp.nxU + cp.ndL + cp.ndU > 0:
            compl_inf /= ncomps

        return dual_inf + primal_inf + compl_inf

    sigma_lo = jnp.maximum(cp.p["sigma_min"], cp.p["mu_min"] / cqpr.avrg_compl)
    sigma_up = jnp.minimum(cp.p["sigma_max"], mu_max / cqpr.avrg_compl)
    mu_max_over_compl = mu_max / cqpr.avrg_compl  # IPOPT uses this separately for downward search

    # Slope estimation at sigma=1 to determine search direction
    qf_1 = jnp.atleast_2d(calc_quality_function(jnp.array(1.0)))
    sigma_1minus = jnp.atleast_2d(1.0 - jnp.maximum(1e-4, cp.p["quality_function_section_sigma_tol"]))
    qf_1minus = jnp.atleast_2d(calc_quality_function(sigma_1minus))

    # Determine search direction and bounds (matches IPOPT exactly)
    def search_upward(_):
        # QF decreasing for sigma > 1: search [1, min(sigma_max, mu_max/compl)]
        lo = jnp.array([[1.0]])
        up = sigma_up
        q_lo = qf_1
        q_up = jnp.array([[-1.0]])  # sentinel: not computed
        return lo, up, q_lo, q_up

    def search_downward(_):
        # QF increasing for sigma > 1: search [sigma_lo, min(max(sigma_lo, sigma_1minus), mu_max/compl)]
        lo = sigma_lo
        up = jnp.minimum(jnp.maximum(sigma_lo, sigma_1minus), mu_max_over_compl)
        q_lo = jnp.array([[-1.0]])  # sentinel: not computed
        q_up = qf_1minus
        return lo, up, q_lo, q_up

    search_lo, search_up, q_lo, q_up = jax.lax.cond(
        (qf_1minus > qf_1).squeeze(), search_upward, search_downward, None
    )

    # Skip search if bounds are invalid
    sigma = jax.lax.cond(
        (search_lo >= search_up).squeeze(),
        lambda _: jax.lax.cond((qf_1minus > qf_1).squeeze(), lambda _: search_up, lambda _: search_lo, None),
        lambda _: golden_section_search(
            search_up, q_up, search_lo, q_lo,
            cp.p["quality_function_section_sigma_tol"],
            cp.p["quality_function_section_qf_tol"],
            cp.p["quality_function_max_section_steps"],
            calc_quality_function,
        ),
        None
    )

    # return updated mu
    new_mu = sigma * cqpr.avrg_compl

    # lower safeguard
    lower_mu_safeguard_reg = cp.nstqf.calc_lower_mu_safeguard(cqpr.grad_lag_x, cqpr.grad_lag_s, cqpr.c, cqpr.dms, init_dual_inf, init_primal_inf)
    lower_mu_safeguard_resto = cp.nstqfr.calc_lower_mu_safeguard(cqpr.grad_lag_x, cqpr.grad_lag_s, cqpr.c, cqpr.dms, init_dual_inf, init_primal_inf)
    lower_mu_safeguard = jnp.where(fl.in_restoration.squeeze(), lower_mu_safeguard_resto, lower_mu_safeguard_reg)
    low = jnp.maximum(mu_min, lower_mu_safeguard)
    high = mu_max

    # if not in restoration and ncomps == 0 then we just use mu min always
    ncomps = cp.nxL + it.z_U.size + it.v_L.size + it.v_U.size
    new_mu = jnp.where(
        (ncomps == 0) & (1 - fl.in_restoration.squeeze()),
        jnp.atleast_2d(mu_min),
        jnp.clip(new_mu, low, high)
    )
    
    return new_mu

def calc_new_monotone_mu(cp, cqpr, mu_max, init_dual_inf, init_primal_inf, fl, mu_min):
    # if resto is True: nstqf = cp.nstqfr
    # elif resto is False: nstqf = cp.nstqf
    # else: raise ValueError("resto flag must be True or False")
    max_ref = 1e20 # hard coded
    new_mu = cp.p["adaptive_mu_monotone_init_factor"] * cqpr.avrg_compl
    # lower_mu_safeguard = nstqf.calc_lower_mu_safeguard(
    #     cqpr.grad_lag_x, cqpr.grad_lag_s, cqpr.c, cqpr.dms, init_dual_inf, init_primal_inf)

    lower_mu_safeguard_reg = cp.nstqf.calc_lower_mu_safeguard(cqpr.grad_lag_x, cqpr.grad_lag_s, cqpr.c, cqpr.dms, init_dual_inf, init_primal_inf)
    lower_mu_safeguard_resto = cp.nstqfr.calc_lower_mu_safeguard(cqpr.grad_lag_x, cqpr.grad_lag_s, cqpr.c, cqpr.dms, init_dual_inf, init_primal_inf)
    lower_mu_safeguard = jnp.where(fl.in_restoration.squeeze(), lower_mu_safeguard_resto, lower_mu_safeguard_reg)

    low = jnp.maximum(mu_min, lower_mu_safeguard)
    high = jnp.minimum(mu_max, max_ref * 0.1)
    return jnp.clip(new_mu, low, high)

def calc_updated_monotone_mu(df, mu, cp):
    compl_inf_tol = cp.p["compl_inf_tol"] * df
    mu_monotone = jnp.minimum(cp.p["mu_linear_decrease_factor"] * mu, mu ** cp.p["mu_superlinear_decrease_power"])
    return jnp.maximum(mu_monotone, jnp.minimum(compl_inf_tol, cp.p["tol"]) / (cp.p["barrier_tol_factor"] + 1.))

def calc_updated_mu_control_flow(mu_quality, tau_quality, mu_monotone, tau_monotone, new_mu_monotone, new_tau_monotone, cqpr, mu, tau, fl, adfs, ls, cp, iter_count):

    # control flow -------------------------------------------------------------
    def check_sufficient_progress(f, nlp_error, nlp_constr_viol, adfs):
        margin = cp.p["filter_margin_fact"] * jnp.minimum(cp.p["filter_max_margin"], nlp_error)
        return cp.stqf.is_acceptable_to_current_filter(f + margin, nlp_constr_viol + margin, adfs)

    # WARNING - we are using the BARRIER CONSTR VIOL here
    sufficient_progress = check_sufficient_progress(cqpr.f, cqpr.nlp_error, cqpr.barrier_constr_viol, adfs)

    fixed_to_free = (1-fl.free_mu_mode) & (sufficient_progress & (1-fl.tiny_step_flag))

    staying_fixed = (1 - fl.free_mu_mode) & (1 - fixed_to_free)
    sub_problem_solved = (cqpr.nlp_error <= cp.p["barrier_tol_factor"] * mu) | fl.tiny_step_flag

    fixed_tiny_step_break = staying_fixed & sub_problem_solved & fl.tiny_step_flag & (mu_monotone == mu)
    fixed_update_mu_tau = staying_fixed & sub_problem_solved & (1 - fixed_tiny_step_break)
    fixed_same_mu_tau = staying_fixed & (1 - sub_problem_solved)

    # IPOPT's CheckSkippedLineSearch() equivalent: if line search failed but didn't go to restoration,
    # force insufficient progress to trigger switch from free to fixed mu mode.
    # This only applies in free mode (IPOPT lines 402-405 in IpAdaptiveMuUpdate.cpp)
    skipped_line_search = fl.fallback_activated | fl.soft_resto_entry_requested
    stay_free = fl.free_mu_mode & sufficient_progress & (1 - fl.tiny_step_flag) & (1 - skipped_line_search)

    switching_free_to_fixed = fl.free_mu_mode & (1 - stay_free)
    free_to_fixed_break = switching_free_to_fixed & fl.tiny_step_flag & (new_mu_monotone == mu)
    free_to_fixed = switching_free_to_fixed & (1 - free_to_fixed_break)

    branches = jnp.vstack([
        fixed_to_free,
        fixed_same_mu_tau,
        fixed_update_mu_tau,
        fixed_tiny_step_break,
        stay_free,
        free_to_fixed,
        free_to_fixed_break,
    ])

    # feel free to assert branches.sum() == 1 if you doubt the control flow

    # ls and adfs can be reset/augmented respectively here
    _terminate, _no_terminate = jnp.array([[3]]), jnp.array([[0]])  # 3 = locally_infeasible (tiny step break)
    _free, _fixed = jnp.array([[1]]), jnp.array([[0]])  # free_mu_mode values
    # Filter stores (theta=constraint_violation, phi=objective) per is_acceptable_to_current_filter
    aug_adfs = cp.stqf.augment_raw_filter(adfs, cqpr.primal_inf, cqpr.f, iter_count)
    ls_reset = cp.stqf.ls_reset(ls)

    mu, tau, ls, adfs, terminate, new_free_mu_mode = filter_select_n(
        jnp.argmax(branches.astype(jnp.int32), axis=0).squeeze(),
        *[
            # fixed to free
            (mu_quality, tau_quality, ls_reset, aug_adfs, _no_terminate, _free),
            # fixed same mu tau
            (mu, tau, ls, adfs, _no_terminate, _fixed),
            # fixed update mu tau
            (mu_monotone, tau_monotone, ls_reset, adfs, _no_terminate, _fixed),
            # fixed tiny step break
            (mu, tau, ls, adfs, _terminate, _fixed),
            # stay free
            (mu_quality, tau_quality, ls_reset, aug_adfs, _no_terminate, _free),
            # free to fixed
            (new_mu_monotone, new_tau_monotone, ls_reset, adfs, _no_terminate, _fixed),
            # free to fixed break
            (mu, tau, ls, adfs, _terminate, _fixed),
        ]
    )

    return mu, tau, ls, adfs, terminate, new_free_mu_mode

def calc_updated_mu(it, cqpr, mu, tau, mu_max, init_dual_inf, init_primal_inf, df, fl, adfs, ls, cp, iter_count):

    # In restoration phase, mu_min is multiplied by 100 (more conservative)
    mu_min = jnp.where(
        fl.in_restoration.squeeze(),
        cp.p["mu_min"] * 100.0,
        cp.p["mu_min"]
    )

    # adaptive -----------------------------------------------------------------
    tau_quality = jnp.maximum(cp.p["tau_min"], 1 - cqpr.nlp_error)
    tau_quality = jnp.atleast_2d(tau_quality)
    mu_quality = calc_quality_function_mu(cp, it, tau_quality, cqpr, mu_max, init_dual_inf, init_primal_inf, fl, mu_min)

    # monotone -----------------------------------------------------------------
    mu_monotone = calc_updated_monotone_mu(df, mu, cp)
    tau_monotone = jnp.maximum(cp.p["tau_min"], 1.-mu_monotone)

    # if we are generating a mu for a new monotone update sequence
    new_mu_monotone = calc_new_monotone_mu(cp, cqpr, mu_max, init_dual_inf, init_primal_inf, fl, mu_min)
    new_tau_monotone = jnp.maximum(cp.p["tau_min"], 1.-new_mu_monotone)

    # control flow -------------------------------------------------------------
    mu, tau, ls, adfs, terminate, new_free_mu_mode = calc_updated_mu_control_flow(mu_quality, tau_quality, mu_monotone, tau_monotone, new_mu_monotone, new_tau_monotone, cqpr, mu, tau, fl, adfs, ls, cp, iter_count)
    return mu, tau, ls, adfs, terminate, new_free_mu_mode

if __name__ == "__main__":

    import json
    
    with open("jaxipm/params.json") as f:
        p = json.load(f)

    # do the setup for validation ----------------------------------------------
    from jaxipm.utils.validation_boilerplate import *
    from jaxipm.initialization import initialize_common_problem

    result = minimize_ipopt(
        fun=obj,
        x0=x0,
        jac=obj_grad,
        hess=obj_hess,
        constraints=constraints,
        bounds=bounds,
        options={
            "disp": 5,
            "maxiter": num_iters,
            "disp": 5,
            "start_with_resto": "yes",  # lets us test feas resto initialization as well
        },
    )

    # common problem across entire batch
    cp = initialize_common_problem(
        f, c, d, x_L, x_U, d_L, d_U, x0, p, [f_args, c_args, d_args]
    )

    # 

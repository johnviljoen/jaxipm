from typing import Callable, Tuple, Dict

import equinox as eqx
from jaxtyping import Array

# --------------------------------------------------------- #
# ---------------------- Functions ------------------------ #
# --------------------------------------------------------- #

# KKT functions specific to regular phase
class KKTRegularFunctions(eqx.Module):
    calc_aug_pd_RHS:  Callable
    calc_aug_pd_RHS_aff: Callable
    calc_aug_pd_RHS_cen: Callable
    calc_aug_pd_LHS_given_deltas: Callable # temporary

# KKT functions specific to restoration phase
class KKTRestoFunctions(eqx.Module):
    calc_resto_red_pd_RHS: Callable
    calc_resto_red_pd_RHS_aff: Callable
    calc_resto_red_pd_RHS_cen: Callable
    calc_resto_red_pd_LHS_given_deltas: Callable # temporary
    calc_transform_red_to_aug: Callable
    # Least squares multiplier calculation (for resto exit - when we synchronize hot restart)
    # calc_ls_mults_RHS: Callable
    # calc_ls_mults_LHS: Callable

# KKT functions for condensed formulation (SPD, Cholesky-factorable)
class KKTCondensedFunctions(eqx.Module):
    build_condensed_data: Callable   # scatter W, Sigma_x, D*J'DJ into CSR data
    condense_rhs: Callable           # 4-block RHS -> condensed 1-block RHS
    recover_step: Callable           # dx -> (dx, ds_all, dy_all)

# Quantity functions that cannot share traces between regular and restoration
class NonSharedTraceQuantityFunctions(eqx.Module):
    calc_f: Callable
    calc_jac_f: Callable
    calc_hess_f: Callable
    calc_c: Callable
    calc_jac_c: Callable 
    calc_hess_c: Callable
    calc_d: Callable
    calc_jac_d: Callable 
    calc_hess_d: Callable
    calc_theta: Callable
    calc_slacks: Callable
    calc_slack_derivatives: Callable
    calc_grad_barrier_obj_x: Callable
    calc_grad_lag_x: Callable
    calc_complementarity: Callable
    calc_complementarity_unscaled: Callable
    calc_lower_mu_safeguard: Callable
    calc_nlp_error: Callable
    calc_check_converged: Callable
    calc_current_is_acceptable: Callable
    calc_avrg_compl: Callable
    calc_barrier_obj: Callable
    calc_grad_barr_T_delta: Callable
    calc_x_frac_to_bound: Callable
    calc_alpha_pr: Callable
    calc_alpha_du: Callable
    kkt: KKTRegularFunctions | KKTRestoFunctions
    # trace share compatible - but non-trace share functions depend on them
    calc_dual_inf: Callable
    calc_dual_inf_unscaled: Callable
    calc_nlp_constr_viol: Callable
    calc_nlp_constr_viol_unscaled: Callable
    calc_primal_inf_L1: Callable
    calc_s_frac_to_bound: Callable
    calc_transform_aug_to_full: Callable # trace share - NOT ANYMORE WITH SLACKS CHANGES

class SharedTraceQuantityFunctions(eqx.Module):
    calc_grad_barrier_obj_s: Callable # trace share
    calc_alpha_min: Callable # trace share
    calc_grad_lag_s: Callable # trace share
    calc_vector_to_iterate: Callable # trace share
    calc_iterate_to_vector: Callable # trace share
    is_ftype: Callable # trace share
    armijo_holds: Callable # trace share
    is_acceptable_to_current_iterate: Callable # trace share
    is_acceptable_to_current_filter: Callable # trace share
    augment_raw_filter: Callable # trace share
    augment_ls_filter: Callable # trace share
    ls_reset: Callable # trace share
    check_acceptability_of_trial_point: Callable # trace share
    calc_barrier_constr_viol: Callable
    calc_ls_mults_RHS: Callable
    calc_ls_mults_LHS: Callable
    # calc_ls_mults_LHS_dedicated: Callable


# --------------------------------------------------------- #
# ---------------------- Parameters ----------------------- #
# --------------------------------------------------------- #

# common data across all problems in batch
class CommonProblem(eqx.Module):
    # sparse linear solver for the KKT systems
    refactorize: Callable
    refactorize_and_linear_solve: Callable
    linear_solve: Callable
    # dedicated LS multiplier solver (separate cuDSS handle, own sparsity pattern)
    ls_refactorize_and_solve: Callable
    ls_linear_solve: Callable  # standalone solve (no refactorize) for IR accuracy
    ls_coo_indices: Array  # COO indices for LS-specific sparsity pattern
    ls_nnz_triu: int  # number of nonzeros in upper triangular part of LS LHS
    # All arrays below are problem structure metadata - static across batch
    coo_indices: Array # regular csr data for LHS system
    W_full_coo_indices: Array # conforming bcoos to original sparsity for W
    dxs_diag_indices: Array # csr.data indices for x, s diagonal entries in KKT LHS
    dcd_diag_indices: Array # csr.data indices for c, d diagonal entries in KKT LHS
    dc_diag_indices: Array  # csr.data indices for c diagonal (DcR in resto) - dcd_diag_indices[:nyc]
    dd_diag_indices: Array  # csr.data indices for d diagonal (DdR in resto) - dcd_diag_indices[nyc:]
    nnz_triu: int  # number of nonzeros in upper triangular part of KKT LHS
    nse_Jf: int
    nse_Jc: int
    nse_Jd: int
    nse_rJf: int
    nse_rJc: int
    nse_rJd: int
    W_nnz_triu: int  # nnz in triu of W
    W_nnz: int
    hess_f_coo_indices: Array  # unified hess_f sparsity pattern (union of regular + resto diagonal)
    hess_f_nnz: int  # number of nonzeros in unified hess_f pattern
    nx: int
    nyc: int
    nyd: int
    nxL: int
    nxU: int
    ndL: int
    ndU: int
    x_L: Array 
    x_U: Array 
    d_L: Array 
    d_U: Array 
    ind_x_L: Array 
    ind_x_U: Array 
    ind_x_LU: Array 
    ind_d_L: Array 
    ind_d_U: Array 
    ind_d_LU: Array 
    dampind_x_L: Array 
    dampind_x_U: Array 
    dampind_d_L: Array 
    dampind_d_U: Array 
    np_L: Array 
    np_U: Array 
    ind_np_L: Array 
    ind_np_U: Array 
    dampind_np_L: Array
    dampind_np_U: Array
    p: Dict  # static params dict wrapper
    # functions
    stqf: SharedTraceQuantityFunctions
    nstqf: NonSharedTraceQuantityFunctions
    nstqfr: NonSharedTraceQuantityFunctions
    calc_next_problem: Callable
    # # condensed KKT fields (populated only when p["kkt_system"] == "condensed")
    # condensed_jptr: Array   # (N_schur, 4) scatter indices for J_all'*D*J_all: [csr_idx, constraint_idx, Jd_k, Jd_l]
    # condensed_hptr: Array   # (N_hess, 2) scatter indices for W entries: [csr_idx, W_data_idx]
    # condensed_dptr: Array   # (nx, 2) scatter indices for diagonal: [csr_idx, variable_idx]
    # condensed_nnz: int      # total nnz in condensed upper triangle CSR
    # condensed_nx: int       # size of condensed system (= original nx)
    # condensed_sort_order: Array  # permutation: concat(Jc.data, Jd.data)[sort_order] → CSC-sorted J_all data
    # # condensed KKT closures (populated only when p["kkt_system"] == "condensed")
    # condensed_fns: KKTCondensedFunctions

# --------------------------------------------------------- #
# ------------ State and Calculated Quantities ------------ #
# --------------------------------------------------------- #


# State
# an iterate for an individual optimization in the batch
class Iterate(eqx.Module):
    # the actual iterate
    x: Array  # primal optimization variables
    # nc: Array
    # pc: Array
    # nd: Array
    # pd: Array
    s: Array  # slack variables
    y_c: Array  # equality constraint lagrangian multipliers
    y_d: Array  # inequality constraint lagrangian multipliers
    z_L: Array  # x lower bound multipliers
    # ncL: Array
    # pcL: Array
    # ndL: Array
    # pdL: Array
    z_U: Array  # x upper bound multipliers
    v_L: Array  # s lower bound multipliers
    v_U: Array  # s upper bound multipliers

# Intermediate Values (not state, but saved to avoid recomputation)
# Except for the y_c, y_d init values at the bottom
# to prevent duplicate calculations we save a "state" at the start of every iteration
# in which we calculate values used in many places throughout most iterates.
class CalculatedQuantitiesPreMu(eqx.Module):
    f: Array
    c: Array
    d: Array
    jac_f: Array
    dms: Array
    y_nrminf: Array
    grad_lag_x: Array
    grad_lag_s: Array
    slacks: Array
    avrg_compl: Array
    theta: Array
    grad_lag_x_nrm2: Array
    grad_lag_s_nrm2: Array
    c_nrm2: Array
    d_minus_s_nrm2: Array
    nlp_error: Array
    nlp_constr_viol: Array
    barrier_constr_viol: Array
    primal_inf: Array
    step_aff_full: Array
    step_cen_full: Array
    Sigma_nc_inv: Array
    Sigma_pc_inv: Array
    Sigma_nd_inv: Array
    Sigma_pd_inv: Array
    # we run initialization here for problems that are being restarted
    y_c_init: Array
    y_d_init: Array

# intermediate values
class CalculatedQuantitiesPostMu(eqx.Module):
    rhs_aug: Array
    step_aug: Array
    rhs: Iterate
    step: Iterate
    alpha_pr: Array
    barr: Array
    gBD: Array
    slack_derivatives: Array

# State
# the state for the iterate for an individual optimization in the batch
class IterateFlags(eqx.Module):
    in_watchdog: Array
    in_soft_resto_phase: Array
    in_restoration: Array
    theta_max_instantiated: Array  # tiny step -> regular step, only instantiate t_max in first non-tiny step
    fallback_activated: Array  # if resto is available it literally just means goto resto
    tiny_step_last_iter: Array  # to check if we hit tiny step twice in a row
    skip_first_trial: Array  # skip first trial point (start with reduced alpha) - set on watchdog timeout
    soft_resto_entry_requested: Array  # backtrack failed, try soft resto before full resto
    # signals
    free_mu_mode: Array # are we in monotone or adaptive right now?
    tiny_step_flag: Array  # tells us if we have hit tiny_step twice
    needs_resto_init: Array  # fallback_activated & not in_restoration -> init resto in post_process
    needs_regular_init: Array # flag (0 or 1) to trigger hot restart initialization
    should_exit_resto: Array  # True when restoration converged and we should return to regular

# Mostly intermediate values - some state.
# the state for the watchdog for an individual optimization in the batch
class WatchdogState(eqx.Module):
    # things in the backtrack LS
    shortened_iter: Array  # counter for shortened iterations
    trial_iter: Array  # counter for watchdog iterations
    alpha_pr_test: Array  # step size for armijo test in watch dog
    it: Iterate  # watchdog reference iterate
    delta: Iterate  # watchdog search direction at reference point
    last_mu: Array  # barr parameter value during last line search
    # things in the filter LS acceptor
    theta: Array
    barr: Array
    gBD: Array

# So we 
# the state for the filter for an individual optimization in the batch
class LineSearchFilterState(eqx.Module):
    theta_min: Array
    theta_max: Array
    last_rejection_due_to_filter: Array
    count_successive_filter_rejections: Array
    n_filter_resets: Array
    F: Array
    ref_theta: Array  # we swap the ref values in the filter for these upon stopwatchdog
    ref_barr: Array  # we swap the ref values in the filter for these upon stopwatchdog
    ref_gBD: Array  # we swap the ref values in the filter for these upon stopwatchdog

# the state for the line search for an individual optimization in the batch
class LineSearchState(eqx.Module):
    acceptable_point: Iterate
    n_steps: Array
    # regular backtrack LS carry state
    accept: Array
    trial_step: Iterate
    it_trial: Iterate
    alpha_pr: Array
    alpha_min: Array
    n_filter_resets: Array
    trial_theta: Array
    last_obj_val: Array
    # SOC specific state
    count_soc: Array
    theta_soc_old: Array
    c_soc: Array
    dms_soc: Array
    # soft feas resto
    soft_resto_phase_counter: Array
    satisfies_original_criterion: Array
    # hard feas resto
    count_restorations: Array
    required_infeasibility_reduction: Array  # 0. if square, 1e-3 otherwise
    filter: LineSearchFilterState

# the state for the inertia correction for an individual optimization in the batch.
# Currently this is underutilized, as our dcd never moves from zero as cuDSS doesnt
# give us reliable zero eigenvalue detection.
class InertiaCorrectionState(eqx.Module):
    dxs: Array  # = 0 (init)
    dcd: Array  # = 0 (init)
    dxs_old: Array  # = 0 (init)
    dcd_old: Array  # = 0 (init)
    jac_degen: Array  # {0 (init): not yet determined, 1: not degen, 2: degen}
    hess_degen: Array  # {0 (init): not yet determined, 1: not degen, 2: degen}
    test_status: Array  # {0 (init): no_test, 1: dcd==0 & dxs==0, 2: dcd>0 & dxs==0, 3: dcd==0 & dxs>0, 4: dcd>0 & dxs>0}
    degen_iters: Array  # = 0 (init)
    inertia: Array  # = jnp.array([0,0]) (init) # I do not need this to be kept, but I like it for debugging
    perturbed_data: Array  # = jnp.zeros_like(op.offsets_size - 1)

# the state for an individual optimization in the batch
class OptimizationState(eqx.Module):
    """
    Mode Switching Semantics:
    - Single active set: fl, wd, ls, ic, adfs used by BOTH regular and restoration modes
    - Save slots: saved_fl, saved_wd, saved_ls, saved_ic, saved_adfs store regular state during restoration

    On enter restoration (initialize_problem_resto):
    - Current (regular) state saved to saved_* slots
    - Fresh restoration state initialized in active slots (fl, wd, ls, ic, adfs)

    On exit restoration (exit_restoration_mode):
    - Active slots restored from saved_* slots
    - Iterate x[:cp.nx] extracted from restoration iterate
    """
    # Active structures (used by both regular and restoration modes)
    it: Iterate
    cqpr: CalculatedQuantitiesPreMu
    cqpo: CalculatedQuantitiesPostMu
    fl: IterateFlags
    wd: WatchdogState
    ls: LineSearchState
    ic: InertiaCorrectionState
    adfs: Array  # adaptive mu filter state (different to line search filter)
    mu: Array
    tau: Array
    mu_max: Array
    init_dual_inf: Array
    init_primal_inf: Array

    # Save slots for regular state (populated when entering restoration)
    saved_fl: IterateFlags
    saved_wd: WatchdogState
    saved_ls: LineSearchState
    saved_ic: InertiaCorrectionState
    saved_adfs: Array
    saved_mu: Array
    saved_tau: Array
    saved_mu_max: Array
    saved_init_dual_inf: Array
    saved_init_primal_inf: Array

    # Original problem's inf-norm primal infeasibility at restoration entry
    # Used for restoration convergence check (IPOPT's orig_curr_inf_pr)
    saved_orig_inf_pr: Array

    # Saved bound multipliers at restoration entry (used for bound mult step computation on exit)
    saved_z_L: Array
    saved_z_U: Array
    saved_v_L: Array
    saved_v_U: Array

    # Saved slacks at restoration entry (sxL, sxU, sdL, sdU)
    saved_slacks: Tuple[Array, Array, Array, Array]

    # Restoration problem's convergence tolerance (can be tightened during resto)
    # Initialized to p["tol"], tightened by 0.01x when nearly feasible
    resto_tol: Array

    # Iteration tracking
    iter_count: Array

    # Problem function arguments with scaling - consistent across regular/resto
    args: Tuple[
        Tuple[Array, ...],  # first array defines scaling for f
        Tuple[Array, ...],  # first array defines scaling for c
        Tuple[Array, ...],  # first array defines scaling for d
    ]


if __name__ == "__main__":
    # initialize an opt state with all Nones
    from inspect import signature

    def nones(cls):
        return cls(*[None] * len(signature(cls).parameters))

    opt_state = OptimizationState(
        # Active structures
        it=nones(Iterate),
        cqpr=nones(CalculatedQuantitiesPreMu),
        cqpo=nones(CalculatedQuantitiesPostMu),
        fl=nones(IterateFlags),
        wd=nones(WatchdogState),
        ls=nones(LineSearchState),
        ic=nones(InertiaCorrectionState),
        adfs=None,
        mu=None,
        tau=None,
        mu_max=None,
        init_dual_inf=None,
        init_primal_inf=None,
        # Save slots
        saved_fl=nones(IterateFlags),
        saved_wd=nones(WatchdogState),
        saved_ls=nones(LineSearchState),
        saved_ic=nones(InertiaCorrectionState),
        saved_adfs=None,
        saved_mu=None,
        saved_tau=None,
        saved_mu_max=None,
        saved_init_dual_inf=None,
        saved_init_primal_inf=None,
        saved_orig_inf_pr=None,
        saved_z_L=None,
        saved_z_U=None,
        saved_v_L=None,
        saved_v_U=None,
        saved_slacks=(None, None, None, None),
        # Restoration tolerance / iteration tracking
        resto_tol=None,
        iter_count=None,
        # Problem function arguments with scaling
        args=((None,), (None,), (None,)),
    )

    pass
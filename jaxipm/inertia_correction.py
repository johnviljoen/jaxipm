"""CuDSS has some rather... inaccurate inertias. We cannot reliably detect singular
KKT matrices, even less so when running in batch (zeros get placed in + or - counts
randomly afaik). Therefore we make the ASSUMPTION that we will always have non-singular 
matrices. This is equivalent to saying we expect constraint qualifications to always
be satisfied for a given iteration, as well as the lagrangian hessian to have no
zero eigenvalues. This also simplifies the inertia correction algorithm substantially,
we just assume that it is "wrong_inertia", and correct accordingly following the
IPOPT algorithm. We do not have internal CuDSS information such as what IPOPT uses
for Pardisos inertia correction."""

import jax
import jax.numpy as jnp
import equinox as eqx

def _compute_DcR_DdR_diag(Sigma_nc, Sigma_pc, Sigma_nd, Sigma_pd, dxs):
    """Compute DcR and DdR diagonal values for resto reduced system.

    DcR[i] = -1/(Sigma_nc[i] + dxs) - 1/(Sigma_pc[i] + dxs)
    DdR[i] = -1/(Sigma_nd[i] + dxs) - 1/(Sigma_pd[i] + dxs)
    """
    eps = 1e-20
    DcR_diag = -1.0 / jnp.maximum(Sigma_nc + dxs, eps) - 1.0 / jnp.maximum(Sigma_pc + dxs, eps)
    DdR_diag = -1.0 / jnp.maximum(Sigma_nd + dxs, eps) - 1.0 / jnp.maximum(Sigma_pd + dxs, eps)
    return DcR_diag.flatten(), DdR_diag.flatten()


def _compute_rhs_cR_dR_negated(Sigma_nc, Sigma_pc, Sigma_nd, Sigma_pd, dxs, rhs_intermediates):
    """Compute negated rhs_cR and rhs_dR for resto reduced system RHS update.

    In the reduced system, the RHS for c and d blocks depend on sigma_tilde_inv = 1/(Sigma + dxs):
        rhs_cR = rhs_c - sigma_nc_inv * mod_rhs_nc + sigma_pc_inv * mod_rhs_pc
        rhs_dR = rhs_d - sigma_nd_inv * mod_rhs_nd + sigma_pd_inv * mod_rhs_pd

    The full RHS is stored as -rhs_cR and -rhs_dR (negated), so we return the negated values:
        rhs_cR_negated = -rhs_c + sigma_nc_inv * mod_rhs_nc - sigma_pc_inv * mod_rhs_pc
        rhs_dR_negated = -rhs_d + sigma_nd_inv * mod_rhs_nd - sigma_pd_inv * mod_rhs_pd
    """
    mod_rhs_nc, mod_rhs_pc, mod_rhs_nd, mod_rhs_pd, rhs_c, rhs_d = rhs_intermediates
    eps = 1e-20
    sigma_nc_inv = 1.0 / jnp.maximum(Sigma_nc + dxs, eps)
    sigma_pc_inv = 1.0 / jnp.maximum(Sigma_pc + dxs, eps)
    sigma_nd_inv = 1.0 / jnp.maximum(Sigma_nd + dxs, eps)
    sigma_pd_inv = 1.0 / jnp.maximum(Sigma_pd + dxs, eps)

    rhs_cR_negated = -rhs_c + sigma_nc_inv * mod_rhs_nc - sigma_pc_inv * mod_rhs_pc
    rhs_dR_negated = -rhs_d + sigma_nd_inv * mod_rhs_nd - sigma_pd_inv * mod_rhs_pd
    return rhs_cR_negated.flatten(), rhs_dR_negated.flatten()


def solve_with_inertia_correction(csr_lhs, rhs, ic, cp, fl, resto, Sigma_nc, Sigma_pc, Sigma_nd, Sigma_pd, rhs_intermediates):
    """Solve the KKT system with inertia correction.

    For restoration mode, when dxs perturbation changes, BOTH the LHS and RHS must be updated:
    - LHS: DcR/DdR diagonal blocks contain sigma_tilde_inv = 1/(Sigma + dxs) terms
    - RHS: rhs_cR and rhs_dR also depend on sigma_tilde_inv

    Args:
        resto: Whether in restoration mode (Array for vmap compatibility)
        Sigma_nc, Sigma_pc, Sigma_nd, Sigma_pd: Raw Sigma values for resto DcR/DdR updates
            (pass zeros for regular mode)
        rhs_intermediates: Tuple of (mod_rhs_nc, mod_rhs_pc, mod_rhs_nd, mod_rhs_pd, rhs_c, rhs_d)
            for recomputing rhs_cR and rhs_dR when dxs changes (pass zeros for regular mode)
    """
    at_degen_limit = ic.degen_iters + 1 >= cp.p["degen_iters_max"]
    hess_degen_pre = jnp.select(
        [ic.hess_degen > 0, ic.test_status == 1],
        [ic.hess_degen, jnp.array(1)],
        default=ic.hess_degen)
    degen_iters = jax.lax.cond((ic.test_status == 3) & (ic.hess_degen == 0) & ~at_degen_limit,
                                lambda: ic.degen_iters + 1, lambda: ic.degen_iters)
    # Start at decayed value only if hess_degen is ALREADY 2 (from previous call)
    # Don't set hess_degen=2 here - we need to try dxs=0 first
    dxs_init = jax.lax.cond(hess_degen_pre == 2,
        lambda: jnp.maximum(cp.p["min_hessian_perturbation"], ic.dxs * cp.p["perturb_dec_fact"]),
        lambda: jnp.array(0.0))
    # jax.debug.callback(print_ic_start, _get_batch_idx(), dxs_init, hess_degen_pre)
    perturbed_data = csr_lhs.data.at[cp.dxs_diag_indices].add(dxs_init)
    # For resto mode, also update DcR/DdR diagonal with sigma_tilde_inv = 1/(Sigma + dxs_init)
    DcR_diag_init, DdR_diag_init = _compute_DcR_DdR_diag(Sigma_nc, Sigma_pc, Sigma_nd, Sigma_pd, dxs_init)
    perturbed_data = jax.lax.cond(
        resto,
        lambda d: d.at[cp.dc_diag_indices].set(DcR_diag_init).at[cp.dd_diag_indices].set(DdR_diag_init),
        lambda d: d,
        perturbed_data
    )
    # For resto mode, also update RHS (rhs_cR and rhs_dR depend on sigma_tilde_inv)
    # RHS structure: [x(nx), s(nyd), y_c(nyc), y_d(nyd)]
    rhs_cR_start = cp.nx + cp.nyd
    rhs_dR_start = cp.nx + cp.nyd + cp.nyc
    rhs_cR_negated_init, rhs_dR_negated_init = _compute_rhs_cR_dR_negated(
        Sigma_nc, Sigma_pc, Sigma_nd, Sigma_pd, dxs_init, rhs_intermediates)
    perturbed_rhs = jax.lax.cond(
        resto,
        lambda r: r.at[rhs_cR_start:rhs_cR_start+cp.nyc].set(rhs_cR_negated_init).at[rhs_dR_start:rhs_dR_start+cp.nyd].set(rhs_dR_negated_init),
        lambda r: r,
        rhs.flatten()
    )
    # Only update dxs_old if previous iteration used perturbation (ic.dxs > 0)
    dxs_old = jax.lax.cond(ic.dxs > 0, lambda: ic.dxs, lambda: ic.dxs_old)
    ic = eqx.tree_at(lambda t: (t.perturbed_data, t.dxs_old, t.dxs, t.hess_degen, t.degen_iters),
                        ic, (perturbed_data, dxs_old, dxs_init, hess_degen_pre, degen_iters))

    def cond(carry):
        dxs, dxs_old, data, rhs_vec, sol, inertia, first = carry
        return ((inertia[0] != cp.nx+cp.nyd) | (inertia[1] != cp.nyc+cp.nyd)) & (dxs <= cp.p["max_hessian_perturbation"])
    def body(carry):
        dxs, dxs_old, data, rhs_vec, sol, inertia, first = carry
        dxs_new = jax.lax.cond(
            first,
            lambda: dxs,
            lambda: jax.lax.cond(
                dxs == 0,
                lambda: jax.lax.cond(dxs_old == 0, lambda: cp.p["first_hessian_perturbation"],
                                        lambda: jnp.maximum(cp.p["min_hessian_perturbation"], dxs_old * cp.p["perturb_dec_fact"])),
                lambda: dxs * jax.lax.cond((dxs_old == 0) | (1e5 * dxs_old < dxs),
                                            lambda: cp.p["perturb_inc_fact_first"], lambda: cp.p["perturb_inc_fact"]))
        )
        data = data.at[cp.dxs_diag_indices].add(dxs_new - dxs)
        # For resto mode, update DcR/DdR diagonal when dxs changes
        DcR_diag_new, DdR_diag_new = _compute_DcR_DdR_diag(Sigma_nc, Sigma_pc, Sigma_nd, Sigma_pd, dxs_new)
        data = jax.lax.cond(
            resto,
            lambda d: d.at[cp.dc_diag_indices].set(DcR_diag_new).at[cp.dd_diag_indices].set(DdR_diag_new),
            lambda d: d,
            data
        )
        # For resto mode, update RHS (rhs_cR and rhs_dR depend on sigma_tilde_inv)
        rhs_cR_negated_new, rhs_dR_negated_new = _compute_rhs_cR_dR_negated(
            Sigma_nc, Sigma_pc, Sigma_nd, Sigma_pd, dxs_new, rhs_intermediates)
        rhs_vec = jax.lax.cond(
            resto,
            lambda r: r.at[rhs_cR_start:rhs_cR_start+cp.nyc].set(rhs_cR_negated_new).at[rhs_dR_start:rhs_dR_start+cp.nyd].set(rhs_dR_negated_new),
            lambda r: r,
            rhs_vec
        )
        _, inertia = cp.refactorize(rhs_vec, data)
        return dxs_new, dxs_old, data, rhs_vec, sol, inertia, False

    init = (ic.dxs, ic.dxs_old, ic.perturbed_data, perturbed_rhs, jnp.zeros([cp.nx+cp.nyd*2+cp.nyc]), jnp.int32([0, 0]), True)
    dxs, _, perturbed_data, rhs_vec, sol, _, _ = jax.lax.while_loop(cond, body, init)
    # if cp.p["DEBUG_MODE"]:
    #     jax.debug.print("IC solution: {sol}, has_nan: {has_nan}", sol=sol, has_nan=jnp.any(jnp.isnan(sol)))
    sol, _ = cp.refactorize_and_linear_solve(rhs_vec, perturbed_data)
    step = sol[:, None]
    test_status = jax.lax.cond(dxs > 0, lambda: jnp.array(3), lambda: jnp.array(1))
    # Now finalize hess_degen: set to 2 (DEGENERATE) only if dxs>0 and at degen limit
    # This matches IPOPT's timing where hess_degenerate_ is set inside finalize_test(),
    # which is called inside PerturbForWrongInertia(), AFTER dxs=0 has been tried and failed.
    hess_degen_final = jax.lax.cond(
        (dxs > 0) & at_degen_limit & (hess_degen_pre == 0),
        lambda: jnp.array(2),
        lambda: hess_degen_pre)
    ic = eqx.tree_at(lambda t: (t.dxs, t.perturbed_data, t.test_status, t.hess_degen),
                     ic, (dxs, perturbed_data, test_status, hess_degen_final))
    return step, ic

# currently redundant until a batched cholesky with per element graceful failures
# for inertia correction is integrated.
def solve_with_inertia_correction_condensed(
    condensed_data, condensed_rhs, ic, cp, resto,
    build_condensed_data_fn, condense_rhs_fn,
    W_data, Sigma_x, Sigma_s, Jd_data,
    mod_rhs_x, mod_rhs_s, rhs_d_all,
    Sigma_nc, Sigma_pc, Sigma_nd, Sigma_pd,
):
    """Solve the condensed KKT system with inertia correction.

    Key difference from augmented IC: changing dxs affects Sigma_s_pert which changes
    the diag_buffer D, which changes ALL Schur complement entries J'DJ. So each IC
    iteration requires a full rebuild of the condensed data array.

    Expected inertia: (nx, 0) — all positive eigenvalues (SPD).

    Args:
        condensed_data: initial condensed CSR data (dxs=0)
        condensed_rhs: condensed RHS (nx,)
        build_condensed_data_fn: closure to rebuild CSR data
        condense_rhs_fn: closure to re-condense RHS
        W_data: Hessian CSR data values
        Sigma_x: raw Sigma_x diagonal (nx,)
        Sigma_s: raw Sigma_s diagonal (m,) — NO perturbation
        Jd_data: J_all Jacobian data values
        mod_rhs_x: modified RHS x block (nx, 1) for re-condensing
        mod_rhs_s: modified RHS s block (m, 1) raw values
        rhs_d_all: constraint RHS (m, 1)
        Sigma_nc, Sigma_pc, Sigma_nd, Sigma_pd: for resto DdR_all recomputation
    """
    nx = cp.condensed_nx

    # finalize_test: same logic as augmented IC
    at_degen_limit = ic.degen_iters + 1 >= cp.p["degen_iters_max"]
    hess_degen_pre = jnp.select(
        [ic.hess_degen > 0, ic.test_status == 1],
        [ic.hess_degen, jnp.array(1)],
        default=ic.hess_degen)
    degen_iters = jax.lax.cond((ic.test_status == 3) & (ic.hess_degen == 0) & ~at_degen_limit,
                                lambda: ic.degen_iters + 1, lambda: ic.degen_iters)
    dxs_init = jax.lax.cond(hess_degen_pre == 2,
        lambda: jnp.maximum(cp.p["min_hessian_perturbation"], ic.dxs * cp.p["perturb_dec_fact"]),
        lambda: jnp.array(0.0))

    # Build initial condensed data with dxs_init
    Sigma_s_pert = Sigma_s + dxs_init

    # For resto: compute DdR_all (all constraints as inequality)
    def _compute_DdR_all(Sigma_nc, Sigma_pc, Sigma_nd, Sigma_pd, dxs):
        eps = 1e-20
        DcR = -1.0 / jnp.maximum(Sigma_nc + dxs, eps) - 1.0 / jnp.maximum(Sigma_pc + dxs, eps)
        DdR = -1.0 / jnp.maximum(Sigma_nd + dxs, eps) - 1.0 / jnp.maximum(Sigma_pd + dxs, eps)
        return jnp.concatenate([DcR.flatten(), DdR.flatten()])

    DdR_all_init = _compute_DdR_all(Sigma_nc, Sigma_pc, Sigma_nd, Sigma_pd, dxs_init)
    # For regular: delta_d = 0 -> diag_buffer = Sigma_s_pert / 1 = Sigma_s_pert
    # For resto: delta_d = DdR_all -> diag_buffer = Sigma_s_pert / (1 - DdR_all * Sigma_s_pert)
    # Note: In MadNLP, du_diag stores the "delta_d" equivalent, and
    # diag_buffer = Sigma_s / (1 - du_diag * Sigma_s)
    delta_d_init = jnp.where(resto, DdR_all_init, jnp.zeros_like(DdR_all_init))
    diag_buffer_init = Sigma_s_pert / (1.0 - delta_d_init * Sigma_s_pert)

    initial_data = build_condensed_data_fn(W_data, Sigma_x, Sigma_s_pert, Jd_data, dxs_init, diag_buffer_init)

    # Re-condense RHS with dxs_init (needed for resto where RHS depends on dxs)
    initial_rhs, _ = condense_rhs_fn(mod_rhs_x, mod_rhs_s, rhs_d_all, Sigma_s_pert, diag_buffer_init, Jd_data)

    perturbed_rhs = jnp.where(dxs_init > 0, initial_rhs, condensed_rhs)
    perturbed_data = jnp.where(dxs_init > 0, initial_data, condensed_data)

    dxs_old = jax.lax.cond(ic.dxs > 0, lambda: ic.dxs, lambda: ic.dxs_old)
    ic = eqx.tree_at(lambda t: (t.perturbed_data, t.dxs_old, t.dxs, t.hess_degen, t.degen_iters),
                        ic, (perturbed_data, dxs_old, dxs_init, hess_degen_pre, degen_iters))

    def cond(carry):
        dxs, dxs_old, data, rhs_vec, sol, inertia, first = carry
        # SPD: ALL eigenvalues must be positive
        return (inertia[0] != nx) & (dxs <= cp.p["max_hessian_perturbation"])

    def body(carry):
        dxs, dxs_old, data, rhs_vec, sol, inertia, first = carry
        # Same dxs escalation logic as augmented IC
        dxs_new = jax.lax.cond(
            first,
            lambda: dxs,
            lambda: jax.lax.cond(
                dxs == 0,
                lambda: jax.lax.cond(dxs_old == 0, lambda: cp.p["first_hessian_perturbation"],
                                        lambda: jnp.maximum(cp.p["min_hessian_perturbation"], dxs_old * cp.p["perturb_dec_fact"])),
                lambda: dxs * jax.lax.cond((dxs_old == 0) | (1e5 * dxs_old < dxs),
                                            lambda: cp.p["perturb_inc_fact_first"], lambda: cp.p["perturb_inc_fact"]))
        )

        # FULL REBUILD of condensed data (key difference from augmented IC)
        Sigma_s_pert_new = Sigma_s + dxs_new
        DdR_all_new = _compute_DdR_all(Sigma_nc, Sigma_pc, Sigma_nd, Sigma_pd, dxs_new)
        delta_d_new = jnp.where(resto, DdR_all_new, jnp.zeros_like(DdR_all_new))
        diag_buffer_new = Sigma_s_pert_new / (1.0 - delta_d_new * Sigma_s_pert_new)

        data_new = build_condensed_data_fn(W_data, Sigma_x, Sigma_s_pert_new, Jd_data, dxs_new, diag_buffer_new)

        # Re-condense RHS (for resto where RHS depends on dxs)
        rhs_new, _ = condense_rhs_fn(mod_rhs_x, mod_rhs_s, rhs_d_all, Sigma_s_pert_new, diag_buffer_new, Jd_data)

        _, inertia = cp.refactorize(rhs_new, data_new)
        return dxs_new, dxs_old, data_new, rhs_new, sol, inertia, False

    init = (ic.dxs, ic.dxs_old, ic.perturbed_data, perturbed_rhs, jnp.zeros(nx), jnp.int32([0, 0]), True)
    dxs, _, perturbed_data, rhs_vec, sol, _, _ = jax.lax.while_loop(cond, body, init)

    sol, _ = cp.refactorize_and_linear_solve(rhs_vec, perturbed_data)
    step = sol[:, None]
    test_status = jax.lax.cond(dxs > 0, lambda: jnp.array(3), lambda: jnp.array(1))
    hess_degen_final = jax.lax.cond(
        (dxs > 0) & at_degen_limit & (hess_degen_pre == 0),
        lambda: jnp.array(2),
        lambda: hess_degen_pre)
    ic = eqx.tree_at(lambda t: (t.dxs, t.perturbed_data, t.test_status, t.hess_degen),
                     ic, (dxs, perturbed_data, test_status, hess_degen_final))
    return step, ic

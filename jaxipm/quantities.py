"""This time we move back to non-duplicated code, and remove the quantities
functions from the state - instead creating a function that forms the resto and
regular quantities at the start of the opt and then adds these to cp."""

import equinox as eqx
import jax
import jax.experimental.sparse as jsparse
import jax.numpy as jnp
import numpy as np
from pathlib import Path
from equinox.internal import ω

from jaxipm.utils.eqx_utils import filter_select
import jaxipm.utils.sparse_utils as spu
# from jaxipm.barrier import calc_quality_function_mu
from jaxipm.inertia_correction import solve_with_inertia_correction, solve_with_inertia_correction_condensed
from jaxipm.structures import CalculatedQuantitiesPreMu, CalculatedQuantitiesPostMu, Iterate, NonSharedTraceQuantityFunctions, SharedTraceQuantityFunctions, KKTRegularFunctions, KKTRestoFunctions, KKTCondensedFunctions

save_dir = "ipopt_logs"

# Debug logging for acceptance checking
_ACCEPT_DEBUG_DIR = Path("tmp/batch_solve_basic_nmpc/accept_debug")
_ACCEPT_DEBUG_FILE = _ACCEPT_DEBUG_DIR / "acceptance_values.csv"
_ACCEPT_DEBUG_INITIALIZED = False
_ACCEPT_DEBUG_CALL_COUNT = 0
_ACCEPT_BATCH_SIZE = 1  # Must match batch size in solver

def _init_accept_debug():
    """Initialize the debug output directory and file."""
    global _ACCEPT_DEBUG_INITIALIZED, _ACCEPT_DEBUG_CALL_COUNT
    if not _ACCEPT_DEBUG_INITIALIZED or not _ACCEPT_DEBUG_FILE.exists():
        _ACCEPT_DEBUG_DIR.mkdir(parents=True, exist_ok=True)
        with open(_ACCEPT_DEBUG_FILE, 'w') as f:
            f.write("trial_theta,ref_theta,trial_barr,ref_barr,alpha_pr_test,in_resto,accept_sanity,accept_primary,accept_historical,final_accept\n")
        _ACCEPT_DEBUG_INITIALIZED = True
        _ACCEPT_DEBUG_CALL_COUNT = 0

def _log_accept_callback(trial_theta, ref_theta, trial_barr, ref_barr, alpha_pr_test, in_resto, accept_sanity, accept_primary, accept_historical, final_accept):
    """Callback to log acceptance check values. Only logs first batch element using counter."""
    global _ACCEPT_DEBUG_CALL_COUNT
    _init_accept_debug()

    # Only log every BATCH_SIZE calls (first element of each batch)
    _ACCEPT_DEBUG_CALL_COUNT += 1
    if (_ACCEPT_DEBUG_CALL_COUNT - 1) % _ACCEPT_BATCH_SIZE != 0:
        return

    # Convert to numpy scalars
    trial_theta = float(np.asarray(trial_theta).flatten()[0])
    ref_theta = float(np.asarray(ref_theta).flatten()[0])
    trial_barr = float(np.asarray(trial_barr).flatten()[0])
    ref_barr = float(np.asarray(ref_barr).flatten()[0])
    alpha_pr_test = float(np.asarray(alpha_pr_test).flatten()[0])
    in_resto = int(np.asarray(in_resto).flatten()[0])
    accept_sanity = int(np.asarray(accept_sanity).flatten()[0])
    accept_primary = int(np.asarray(accept_primary).flatten()[0])
    accept_historical = int(np.asarray(accept_historical).flatten()[0])
    final_accept = int(np.asarray(final_accept).flatten()[0])

    with open(_ACCEPT_DEBUG_FILE, 'a') as f:
        f.write(f"{trial_theta:.15e},{ref_theta:.15e},{trial_barr:.15e},{ref_barr:.15e},{alpha_pr_test:.15e},{in_resto},{accept_sanity},{accept_primary},{accept_historical},{final_accept}\n")

def log_accept_check(trial_theta, ref_theta, trial_barr, ref_barr, alpha_pr_test, in_resto, accept_sanity, accept_primary, accept_historical, final_accept):
    """Log acceptance values using debug.callback (works in jitted/vmapped code)."""
    jax.debug.callback(
        _log_accept_callback,
        trial_theta, ref_theta, trial_barr, ref_barr, alpha_pr_test, in_resto, accept_sanity, accept_primary, accept_historical, final_accept
    )

def generate_non_shared_trace_functions(funcs, general_dims, phase_dims, kkt, p, resto):

    """
    IPOPT contains two problems, the regular and the feasibility restoration.
    They have different shapes, but the functions remain the same. Therefore
    this function factory produces the functions for the regular and resto
    without duplicating code by closing over different shapes conditionally 
    based on the "resto" boolean
    
    The only true difference in the functions being called is during the KKTs, 
    where we use the resto boolean to determine what the functions are.
    """

    calc_f, calc_jac_f, calc_hess_f, calc_c, calc_jac_c, calc_hess_c, calc_d, calc_jac_d, calc_hess_d = funcs
    nx, nxL, x_L, x_U, ind_x_L, ind_x_U, dampind_x_L, dampind_x_U = phase_dims
    _nx, _nxL, nyc, nyd, nxU, ndL, ndU, d_L, d_U, ind_d_L, ind_d_U, dampind_d_L, dampind_d_U, ind_np_L, ind_np_U, _ind_x_L, _ind_x_U = general_dims
    coo_indices, W_full_coo_indices, dxs_diag_indices, dcd_diag_indices, nnz_triu, W_nnz_triu, W_nnz = kkt

    # close over jax array creations to avoid excess tracing
    ind_x_Lr = jnp.hstack([_ind_x_L, ind_np_L+_nx])
    ind_x_Ur = jnp.hstack([_ind_x_U, ind_np_U+_nx])

    if resto is not True:

        def calc_aug_pd_RHS(it, mu, slacks, grad_lag_x, grad_lag_s, c, dms):
            sxL, sxU, sdL, sdU = slacks
            rhs_x = grad_lag_x[:nx] + p["kappa_d"] * mu * (dampind_x_L - dampind_x_U)[:, None]
            rhs_s = (grad_lag_s + p["kappa_d"] * mu * (dampind_d_L - dampind_d_U)[:, None])
            rhs_z_L = sxL[:nxL] * it.z_L[:nxL] - mu
            rhs_z_U = sxU * it.z_U - mu
            rhs_v_L = sdL * it.v_L - mu
            rhs_v_U = sdU * it.v_U - mu
            mod_rhs_x = (
                rhs_x
                + spu.expand_vector(ind_x_L, (nx, 1), rhs_z_L / sxL[:nxL])
                - spu.expand_vector(ind_x_U, (nx, 1), rhs_z_U / sxU)
            )
            mod_rhs_s = (
                rhs_s
                + spu.expand_vector(ind_d_L, (nyd, 1), rhs_v_L / sdL)
                - spu.expand_vector(ind_d_U, (nyd, 1), rhs_v_U / sdU)
            )
            return -jnp.vstack([mod_rhs_x, mod_rhs_s, c, dms, rhs_z_L, rhs_z_U, rhs_v_L, rhs_v_U])
        
        def calc_aug_pd_RHS_aff(it, slacks, grad_lag_x, grad_lag_s, c, dms):
            sxL, sxU, sdL, sdU = slacks
            rhs_x = grad_lag_x[:nx]
            rhs_s = grad_lag_s
            rhs_c = c
            rhs_d = dms
            rhs_z_L = (sxL[:nxL] * it.z_L[:nxL])  # we use the NON relaxed complementary RHS terms here - mu
            rhs_z_U = (sxU * it.z_U)  # we use the NON relaxed complementary RHS terms here - mu
            rhs_v_L = (sdL * it.v_L)  # we use the NON relaxed complementary RHS terms here - mu
            rhs_v_U = (sdU * it.v_U)  # we use the NON relaxed complementary RHS terms here - mu
            mod_rhs_x = (
                rhs_x
                + spu.expand_vector(ind_x_L, (nx, 1), rhs_z_L / sxL[:nxL])
                - spu.expand_vector(ind_x_U, (nx, 1), rhs_z_U / sxU)
            )
            mod_rhs_s = (
                rhs_s
                + spu.expand_vector(ind_d_L, (nyd, 1), rhs_v_L / sdL)
                - spu.expand_vector(ind_d_U, (nyd, 1), rhs_v_U / sdU)
            )
            return -jnp.vstack([mod_rhs_x, mod_rhs_s, rhs_c, rhs_d, rhs_z_L, rhs_z_U, rhs_v_L, rhs_v_U])
            
        def calc_aug_pd_RHS_cen(it, slacks, avrg_compl):
            sxL, sxU, sdL, sdU = slacks
            rhs_x = (-avrg_compl * p["kappa_d"] * (dampind_x_L - dampind_x_U)).T
            rhs_s = (-avrg_compl * p["kappa_d"] * (dampind_d_L - dampind_d_U)).T
            mod_rhs_x = (
                rhs_x
                + spu.expand_vector(ind_x_L, (nx, 1), jnp.full([nxL, 1], avrg_compl) / sxL[:nxL])
                - spu.expand_vector(ind_x_U, (nx, 1), jnp.full([nxU, 1], avrg_compl) / sxU)
            )
            mod_rhs_s = (
                rhs_s
                + spu.expand_vector(ind_d_L, (nyd, 1), jnp.full([ndL, 1], avrg_compl) / sdL)
                - spu.expand_vector(ind_d_U, (nyd, 1), jnp.full([ndU, 1], avrg_compl) / sdU)
            )
            rhs_c = jnp.zeros_like(it.y_c)
            rhs_d = jnp.zeros_like(it.y_d)
            rhs_z_L = jnp.full([nxL, 1], avrg_compl)
            rhs_z_U = jnp.full([nxU, 1], avrg_compl)
            rhs_v_L = jnp.full([ndL, 1], avrg_compl)
            rhs_v_U = jnp.full([ndU, 1], avrg_compl)
            # NOTE the lack of negative sign
            return jnp.vstack(
                [mod_rhs_x, mod_rhs_s, rhs_c, rhs_d, rhs_z_L, rhs_z_U, rhs_v_L, rhs_v_U]
            )
        
        def calc_aug_pd_LHS_given_deltas(it, jacobians, hessians, perts):
            _, Jc, Jd = jacobians # Jc = calc_jac_c(it.x[:nx], *c_args) # Jd = calc_jac_d(it.x[:nx], *d_args)
            hess_f, hess_c, hess_d = hessians # calc_hess_f(it.x[:nx], *f_args) # calc_hess_c(it.x[:nx], *c_args) # calc_hess_d(it.x[:nx], *d_args)
            pert_x, pert_s, pert_c, pert_d = perts

            hess_f = hess_f[:_nx, :_nx]
            hess_c = hess_c[:, :_nx, :_nx]
            hess_d = hess_d[:, :_nx, :_nx]
            Jc = Jc[:_nx]
            Jd = Jd[:_nx]

            # einsums are not legal for BCOO's :(
            if nyc > 0 and nyd > 0:
                W = (
                    hess_f
                    + jsparse.bcoo_reduce_sum(hess_c * it.y_c[:, None], axes=[0])
                    + jsparse.bcoo_reduce_sum(hess_d * it.y_d[:, None], axes=[0])
                )
            elif nyc > 0:
                W = hess_f + jsparse.bcoo_reduce_sum(
                    hess_c * it.y_c[:, None], axes=[0]
                )
            elif nyd > 0:
                W = hess_f + jsparse.bcoo_reduce_sum(
                    hess_d * it.y_d[:, None], axes=[0]
                )
            else:
                W = hess_f

            W = spu.sum_duplicates_to_pattern(W_full_coo_indices, W, W_nnz)
            W = spu.conform_bcoo_to_new_sparsity(W_full_coo_indices, W.sort_indices())
            W = spu.triu(W, nse=W_nnz_triu)  # get only upper triangular section back

            if p["DEBUG_MODE"]: print("TODO: use pre-calculated (safe) slacks and expand them here instead")
            Sigma_x = spu.expand_vector(ind_x_L, (nx, 1), it.z_L[:nxL]) / (it.x[:nx] - x_L) + spu.expand_vector(ind_x_U, (nx, 1), it.z_U) / (x_U - it.x[:nx])
            Sigma_s = spu.expand_vector(ind_d_L, (nyd, 1), it.v_L) / (it.s - d_L) + spu.expand_vector(ind_d_U, (nyd, 1), it.v_U) / (d_U - it.s)
            Sigma_x = Sigma_x.flatten()
            Sigma_s = Sigma_s.flatten()
            
            LHS_upper_triangular = spu.vstack([
                spu.hstack([W + spu.diagflat(Sigma_x) + pert_x * spu.eye(nx), spu.zeros([nx, nyd]),                          Jc[:nx],                             Jd[:nx]                          ]),
                spu.hstack([spu.zeros([nyd, nx]),                          spu.diagflat(Sigma_s) + pert_s * spu.eye(nyd),   spu.zeros([nyd, nyc]),    -spu.eye(nyd)            ]),
                spu.hstack([spu.zeros([nyc, nx]),                          spu.zeros([nyc, nyd]),                        -pert_c * spu.eye(nyc),      spu.zeros([nyc, nyd]) ]),
                spu.hstack([spu.zeros([nyd, nx]),                          spu.zeros([nyd, nyd]),                        spu.zeros([nyd, nyc]),    -pert_d * spu.eye(nyd)   ]),
            ])

            LHS_upper_triangular = spu.sum_duplicates_to_pattern(coo_indices, LHS_upper_triangular, nse=nnz_triu)
            LHS_upper_triangular = LHS_upper_triangular.sort_indices()
            # conform to the expected sparsity pattern symbolically derived
            LHS_upper_triangular = spu.conform_bcoo_to_new_sparsity(coo_indices, LHS_upper_triangular)
            return LHS_upper_triangular

        kkt = KKTRegularFunctions(
            calc_aug_pd_RHS=calc_aug_pd_RHS,
            calc_aug_pd_RHS_aff=calc_aug_pd_RHS_aff,
            calc_aug_pd_RHS_cen=calc_aug_pd_RHS_cen,
            calc_aug_pd_LHS_given_deltas=calc_aug_pd_LHS_given_deltas,
        )

    else:

        def calc_resto_red_pd_RHS(it, mu, slacks, grad_lag_x, grad_lag_s, c, dms, Sigma_nc_inv, Sigma_pc_inv, Sigma_nd_inv, Sigma_pd_inv):
            # sometimes want this branch to not be called when a non-resto state is passed to the switch in dispatch
            sxL, sxU, sdL, sdU = slacks
            rhs_x = grad_lag_x + p["kappa_d"] * mu * (dampind_x_L - dampind_x_U)[:, None]
            rhs_s = (grad_lag_s + p["kappa_d"] * mu * (dampind_d_L - dampind_d_U)[:, None])
            rhs_c = c
            rhs_d = dms
            rhs_z_L = sxL * it.z_L - mu
            rhs_z_U = sxU * it.z_U - mu
            rhs_v_L = sdL * it.v_L - mu
            rhs_v_U = sdU * it.v_U - mu
            mod_rhs_x = (
                rhs_x
                + spu.expand_vector(ind_x_L, (nx, 1), rhs_z_L / sxL)
                - spu.expand_vector(ind_x_U, (nx, 1), rhs_z_U / sxU)
            )
            mod_rhs_s = (
                rhs_s
                + spu.expand_vector(ind_d_L, (nyd, 1), rhs_v_L / sdL)
                - spu.expand_vector(ind_d_U, (nyd, 1), rhs_v_U / sdU)
            )
            # reduced system changes rhs_c and rhs_d
            # mod_rhs_x_only = mod_rhs_x[: _nx]
            mod_rhs_nc = mod_rhs_x[_nx : _nx + nyc]
            mod_rhs_pc = mod_rhs_x[_nx + nyc : _nx + nyc * 2]
            mod_rhs_nd = mod_rhs_x[_nx + nyc * 2 : _nx + nyc * 2 + nyd]
            mod_rhs_pd = mod_rhs_x[_nx + nyc * 2 + nyd :]
            rhs_cR = rhs_c - Sigma_nc_inv * mod_rhs_nc + Sigma_pc_inv * mod_rhs_pc
            rhs_dR = rhs_d - Sigma_nd_inv * mod_rhs_nd + Sigma_pd_inv * mod_rhs_pd
            return -jnp.vstack(
                [mod_rhs_x, mod_rhs_s, rhs_cR, rhs_dR, rhs_z_L, rhs_z_U, rhs_v_L, rhs_v_U]
            )

        def calc_resto_red_pd_RHS_aff(it, slacks, grad_lag_x, grad_lag_s, c, dms, Sigma_nc_inv, Sigma_pc_inv, Sigma_nd_inv, Sigma_pd_inv):
            sxL, sxU, sdL, sdU = slacks
            rhs_x = grad_lag_x
            rhs_s = grad_lag_s
            rhs_c = c
            rhs_d = dms
            rhs_z_L = (sxL[:nxL] * it.z_L[:nxL])  # we use the NON relaxed complementary RHS terms here - mu
            rhs_z_U = (sxU * it.z_U)  # we use the NON relaxed complementary RHS terms here - mu
            rhs_v_L = (sdL * it.v_L)  # we use the NON relaxed complementary RHS terms here - mu
            rhs_v_U = (sdU * it.v_U)  # we use the NON relaxed complementary RHS terms here - mu
            mod_rhs_x = (
                rhs_x
                + spu.expand_vector(ind_x_L, (nx, 1), rhs_z_L / sxL[:nxL])
                - spu.expand_vector(ind_x_U, (nx, 1), rhs_z_U / sxU)
            )
            mod_rhs_s = (
                rhs_s
                + spu.expand_vector(ind_d_L, (nyd, 1), rhs_v_L / sdL)
                - spu.expand_vector(ind_d_U, (nyd, 1), rhs_v_U / sdU)
            )
            # reduced system changes rhs_c and rhs_d
            mod_rhs_x_only = mod_rhs_x[: _nx]
            mod_rhs_nc = mod_rhs_x[_nx : _nx + nyc]
            mod_rhs_pc = mod_rhs_x[_nx + nyc : _nx + nyc * 2]
            mod_rhs_nd = mod_rhs_x[_nx + nyc * 2 : _nx + nyc * 2 + nyd]
            mod_rhs_pd = mod_rhs_x[_nx + nyc * 2 + nyd :]
            rhs_cR = rhs_c - Sigma_nc_inv * mod_rhs_nc + Sigma_pc_inv * mod_rhs_pc
            rhs_dR = rhs_d - Sigma_nd_inv * mod_rhs_nd + Sigma_pd_inv * mod_rhs_pd
            return -jnp.vstack([mod_rhs_x, mod_rhs_s, rhs_cR, rhs_dR, rhs_z_L, rhs_z_U, rhs_v_L, rhs_v_U])
    
        def calc_resto_red_pd_RHS_cen(it, slacks, avrg_compl, Sigma_nc_inv, Sigma_pc_inv, Sigma_nd_inv, Sigma_pd_inv):
            sxL, sxU, sdL, sdU = slacks
            rhs_x = (-avrg_compl * p["kappa_d"] * (dampind_x_L - dampind_x_U)).T
            rhs_s = (-avrg_compl * p["kappa_d"] * (dampind_d_L - dampind_d_U)).T
            mod_rhs_x = (
                rhs_x
                + spu.expand_vector(ind_x_L, (nx, 1), jnp.full([nxL, 1], avrg_compl) / sxL[:nxL])
                - spu.expand_vector(ind_x_U, (nx, 1), jnp.full([nxU, 1], avrg_compl) / sxU)
            )
            mod_rhs_s = (
                rhs_s
                + spu.expand_vector(ind_d_L, (nyd, 1), jnp.full([ndL, 1], avrg_compl) / sdL)
                - spu.expand_vector(ind_d_U, (nyd, 1), jnp.full([ndU, 1], avrg_compl) / sdU)
            )
            rhs_c = jnp.zeros_like(it.y_c)
            rhs_d = jnp.zeros_like(it.y_d)
            rhs_z_L = jnp.full([nxL, 1], avrg_compl)
            rhs_z_U = jnp.full([nxU, 1], avrg_compl)
            rhs_v_L = jnp.full([ndL, 1], avrg_compl)
            rhs_v_U = jnp.full([ndU, 1], avrg_compl)
            # NOTE the lack of negative sign
            # reduced system changes rhs_c and rhs_d
            mod_rhs_x_only = mod_rhs_x[: _nx]
            mod_rhs_nc = mod_rhs_x[_nx : _nx + nyc]
            mod_rhs_pc = mod_rhs_x[_nx + nyc : _nx + nyc * 2]
            mod_rhs_nd = mod_rhs_x[_nx + nyc * 2 : _nx + nyc * 2 + nyd]
            mod_rhs_pd = mod_rhs_x[_nx + nyc * 2 + nyd :]
            rhs_cR = rhs_c - Sigma_nc_inv * mod_rhs_nc + Sigma_pc_inv * mod_rhs_pc
            rhs_dR = rhs_d - Sigma_nd_inv * mod_rhs_nd + Sigma_pd_inv * mod_rhs_pd
            return jnp.vstack([mod_rhs_x, mod_rhs_s, rhs_cR, rhs_dR, rhs_z_L, rhs_z_U, rhs_v_L, rhs_v_U])

        def calc_resto_red_pd_LHS_given_deltas(it, jacobians, hessians, perts, Sigma_x_only, Sigma_nc_inv, Sigma_pc_inv, Sigma_nd_inv, Sigma_pd_inv):
            _, jac_c, jac_d = jacobians # Jc = calc_jac_c(it.x[:nx], *c_args)[:_nx] # Jd = calc_jac_d(it.x[:nx], *d_args)[:_nx]
            hess_f, hess_c, hess_d = hessians # calc_hess_f_x_only(it.x[:nx], *f_args) # calc_hess_c_x_only(it.x[:nx], *c_args) # calc_hess_d_x_only(it.x[:nx], *d_args)
            pert_x, pert_s, pert_c, pert_d = perts
            hess_f_x_only = hess_f[:_nx, :_nx]
            hess_c_x_only = hess_c[:, :_nx, :_nx]
            hess_d_x_only = hess_d[:, :_nx, :_nx]
            Jc = jac_c[:_nx]
            Jd = jac_d[:_nx]

            # einsums are not legal for BCOO's :(
            if nyc > 0 and nyd > 0:
                W = (
                    hess_f_x_only
                    + jsparse.bcoo_reduce_sum(hess_c_x_only * it.y_c[:, None], axes=[0])
                    + jsparse.bcoo_reduce_sum(hess_d_x_only * it.y_d[:, None], axes=[0])
                )
            elif nyc > 0:
                W = hess_f_x_only + jsparse.bcoo_reduce_sum(
                    hess_c_x_only * it.y_c[:, None], axes=[0]
                )
            elif nyd > 0:
                W = hess_f_x_only + jsparse.bcoo_reduce_sum(
                    hess_d_x_only * it.y_d[:, None], axes=[0]
                )
            else:
                W = hess_f_x_only

            W = spu.sum_duplicates_to_pattern(W_full_coo_indices, W, W_nnz)
            W = spu.conform_bcoo_to_new_sparsity(W_full_coo_indices, W.sort_indices())
            W = spu.triu(W, nse=W_nnz_triu)  # get only upper triangular section back

            if p["DEBUG_MODE"]: print("TODO: use pre-calculated (safe) slacks and expand them here instead")
            Sigma_s = spu.expand_vector(ind_d_L, (nyd, 1), it.v_L) / (it.s - d_L) + spu.expand_vector(ind_d_U, (nyd, 1), it.v_U) / (d_U - it.s)
            Sigma_x = Sigma_x_only.flatten()
            Sigma_s = Sigma_s.flatten()

            DcR = spu.diagflat((-Sigma_nc_inv - Sigma_pc_inv + pert_c).flatten())
            DdR = spu.diagflat((-Sigma_nd_inv - Sigma_pd_inv + pert_d).flatten())
            DxR = W + spu.diagflat(Sigma_x) + pert_x * spu.eye(_nx)

            LHS_upper_triangular = spu.vstack([
                spu.hstack([DxR,                        spu.zeros([_nx, nyd]),                         Jc[:_nx],                             Jd[:_nx]                          ]),
                spu.hstack([spu.zeros([nyd, _nx]), spu.diagflat(Sigma_s) + pert_s * spu.eye(nyd),   spu.zeros([nyd, nyc]),    -spu.eye(nyd)            ]),
                spu.hstack([spu.zeros([nyc, _nx]), spu.zeros([nyc, nyd]),                        DcR,                            spu.zeros([nyc, nyd]) ]),
                spu.hstack([spu.zeros([nyd, _nx]), spu.zeros([nyd, nyd]),                        spu.zeros([nyd, nyc]),    DdR                         ]),
            ])
            
            LHS_upper_triangular = spu.sum_duplicates_to_pattern(coo_indices, LHS_upper_triangular, nse=nnz_triu)
            LHS_upper_triangular = LHS_upper_triangular.sort_indices()
            # print(f"Sigma_x_only.shape: {Sigma_x_only.shape}")
            # print(f"LHS pre-sum: shape={LHS_upper_triangular.shape}, max_idx=({LHS_upper_triangular.indices[:,0].max()}, {LHS_upper_triangular.indices[:,1].max()})")
            # conform to the expected sparsity pattern symbolically derived
            LHS_upper_triangular = spu.conform_bcoo_to_new_sparsity(coo_indices, LHS_upper_triangular)
            return LHS_upper_triangular

        def calc_transform_red_to_aug(red, rhs, Sigma_nc_inv, Sigma_pc_inv, Sigma_nd_inv, Sigma_pd_inv):
            """red: {x, s, y_c, y_d}, rhs: {x, [nc, pc, nd, pd], s, c, d, z_L, [ncL, pcL, ndL, pdL], z_U, v_L, v_U}"""
            cnt = 0
            d_x_only    = red[cnt:cnt+_nx];   cnt += _nx
            d_s         = red[cnt:cnt+nyd];  cnt += nyd
            d_y_c       = red[cnt:cnt+nyc];  cnt += nyc
            d_y_d       = red[cnt:cnt+nyd]
            cnt = _nx
            rhs_n_c = rhs.x[cnt:cnt+nyc];    cnt += nyc
            rhs_p_c = rhs.x[cnt:cnt+nyc];    cnt += nyc
            rhs_n_d = rhs.x[cnt:cnt+nyd];    cnt += nyd
            rhs_p_d = rhs.x[cnt:]
            d_n_c = (rhs_n_c - d_y_c) * Sigma_nc_inv
            d_p_c = (rhs_p_c + d_y_c) * Sigma_pc_inv
            d_n_d = (rhs_n_d - d_y_d) * Sigma_nd_inv
            d_p_d = (rhs_p_d + d_y_d) * Sigma_pd_inv
            d_x = jnp.vstack([d_x_only, d_n_c, d_p_c, d_n_d, d_p_d])
            return jnp.vstack([d_x, d_s, d_y_c, d_y_d])


        kkt = KKTRestoFunctions(
            calc_resto_red_pd_RHS=calc_resto_red_pd_RHS,
            calc_resto_red_pd_RHS_aff=calc_resto_red_pd_RHS_aff,
            calc_resto_red_pd_RHS_cen=calc_resto_red_pd_RHS_cen,
            calc_resto_red_pd_LHS_given_deltas=calc_resto_red_pd_LHS_given_deltas,
            calc_transform_red_to_aug=calc_transform_red_to_aug,
            # calc_ls_mults_RHS=calc_ls_mults_RHS,
            # calc_ls_mults_LHS=calc_ls_mults_LHS,
        )

    # common functions
    def calc_theta(c, dms):
        # calc_c(it.x[:nx], *c_args) # calc_d(it.x[:nx], *d_args) - it.s
        return jnp.linalg.norm(
            jnp.vstack([c, dms]),
            ord=p["constraint_violation_norm_type"],
            keepdims=True,
        )
    
    def calc_slacks(it):
        return (
            it.x[:nx][ind_x_L] - x_L[ind_x_L],
            x_U[ind_x_U] - it.x[:nx][ind_x_U],
            it.s[ind_d_L] - d_L[ind_d_L],
            d_U[ind_d_U] - it.s[ind_d_U],
        )
    
    def calc_slack_derivatives(it):
        return (
            it.x[:nx][ind_x_L],
            -it.x[:nx][ind_x_U],
            it.s[ind_d_L],
            -it.s[ind_d_U],
        )
    
    def calc_grad_barrier_obj_x(jac_f, mu, slacks):
        # calc_jac_f(it.x[:nx], *f_args) # ASSUMES WE RECIEVE DENSE VECTOR JAC_F
        sxL, sxU, _, _ = slacks
        if p["DEBUG_MODE"]: print("WARNING: safe slack division not used!!! This could diverge us from IPOPT")
        # Barrier gradient: ∇φ = ∇f - μ/s_L + μ/s_U + damping
        return (
            jac_f[:nx]
            - spu.expand_vector(ind_x_L, [nx, 1], mu / sxL[:nxL])
            + spu.expand_vector(ind_x_U, [nx, 1], mu / sxU)
            + p["kappa_d"] * mu * (dampind_x_L - dampind_x_U)[:, None]
        )
    
    def calc_grad_lag_x(it, jacobians):
        jac_f, jac_c, jac_d = jacobians
        # calc_jac_c(it.x[:nx], *c_args) # calc_jac_d(it.x[:nx], *d_args)
        return (
            jac_f[:nx] # calc_jac_f(it.x[:nx], *f_args).todense()
            + jac_c[:nx] @ it.y_c
            + jac_d[:nx] @ it.y_d
            - spu.expand_vector(ind_x_L, (nx, 1), it.z_L[:nxL])
            + spu.expand_vector(ind_x_U, (nx, 1), it.z_U)
        )
    
    def calc_complementarity(it, mu, slacks):
        sxL, sxU, sdL, sdU = slacks
        compl_x_L = sxL[:nxL] * it.z_L[:nxL] - mu
        compl_x_U = sxU * it.z_U - mu
        compl_s_L = sdL * it.v_L - mu
        compl_s_U = sdU * it.v_U - mu
        return jnp.linalg.norm(
            jnp.vstack([compl_x_L, compl_x_U, compl_s_L, compl_s_U]), ord=jnp.inf, keepdims=True
        )
    
    def calc_complementarity_unscaled(it, mu, slacks, df):
        return calc_complementarity(it, mu, slacks) / df

    # trace share compatible - but non-trace share functions depend on it
    def calc_dual_inf(grad_lag_x, grad_lag_s, ord=jnp.inf):
        return jnp.linalg.norm(jnp.vstack([grad_lag_x, grad_lag_s]), ord=ord, keepdims=True)

    # trace share compatible - but non-trace share functions depend on it
    def calc_dual_inf_unscaled(grad_lag_x, grad_lag_s, df, dd):
        return calc_dual_inf(grad_lag_x / df, grad_lag_s * dd[:, None] / df)

    # trace share compatible - but non-trace share functions depend on it
    def calc_primal_inf_L1(c, dms):
        return jnp.linalg.norm(c, ord=1, keepdims=True) + jnp.linalg.norm(dms, ord=1, keepdims=True)

    
    def calc_lower_mu_safeguard(grad_lag_x, grad_lag_s, c, dms, init_dual_inf, init_primal_inf):
        dual_inf = calc_dual_inf(grad_lag_x, grad_lag_s, ord=1) / (nx + nyd)
        primal_inf = calc_primal_inf_L1(c, dms) / (1 if (nyc + nyd) == 0 else (nyc + nyd))
        
        lower_mu_safeguard = jnp.maximum(p["adaptive_mu_safeguard_factor"] * (dual_inf / init_dual_inf),
                                         p["adaptive_mu_safeguard_factor"] * (primal_inf / init_primal_inf))
        
        return lower_mu_safeguard

    def calc_sc(it):
        n = nxL + nxU + ndL + ndU
        if n == 0:
            sc = 1.0
        else:
            sc = (
                jnp.linalg.norm(it.z_L[:nxL], ord=1)
                + jnp.linalg.norm(it.z_U, ord=1)
                + jnp.linalg.norm(it.v_L, ord=1)
                + jnp.linalg.norm(it.v_U, ord=1)
            ) / n
            sc = jnp.maximum(p["s_max"], sc) / p["s_max"]
        return sc
    
    def calc_sd(it):
        n = nyc + nyd + nxL + nxU + ndL + ndU
        if n == 0:
            sd = 1.0
        else:
            sd = (
                jnp.linalg.norm(it.y_c, ord=1)
                + jnp.linalg.norm(it.y_d, ord=1)
                + jnp.linalg.norm(it.z_L[:nxL], ord=1)
                + jnp.linalg.norm(it.z_U, ord=1)
                + jnp.linalg.norm(it.v_L, ord=1)
                + jnp.linalg.norm(it.v_U, ord=1)
            ) / n
            sd = jnp.maximum(p["s_max"], sd) / p["s_max"]
        return sd

    # trace share compatible - but non-trace share functions depend on it
    def calc_nlp_constr_viol(c, d, ord=jnp.inf):
        """Original NLP constraint violation (for convergence)"""
        d_viol_L = d_L[ind_d_L] - d[ind_d_L]  # Use actual d_L bounds, not just -d
        d_viol_L = jnp.maximum(d_viol_L, 0)

        d_viol_U = d[ind_d_U] - d_U[ind_d_U]  # Use actual d_U bounds
        d_viol_U = jnp.maximum(d_viol_U, 0)
        return jnp.linalg.norm(jnp.vstack([c, d_viol_L, d_viol_U]), ord=ord, keepdims=True)

    # trace share compatible - but non-trace share functions depend on it
    def calc_nlp_constr_viol_unscaled(c, d, dc, dd, ord=jnp.inf):
        return calc_nlp_constr_viol(c / dc[:, None], d / dd[:, None], ord=ord)
    

    def calc_nlp_error(it, grad_lag_x, grad_lag_s, slacks, mu, c, d):
        sc, sd = calc_sc(it), calc_sd(it)
        dual_inf = calc_dual_inf(grad_lag_x, grad_lag_s)
        complementarity = calc_complementarity(it, mu, slacks)
        nlp_constr_viol = calc_nlp_constr_viol(c, d)
        return jnp.max(
            jnp.hstack([dual_inf / sd, complementarity / sc, nlp_constr_viol])
        )
    
    def calc_check_converged(
        it, mu, grad_lag_x, grad_lag_s, slacks, c, d, args
    ):
        # args[0] = (mu, x_ref, dr_x, df, ...), so df is at index 3
        df, dc, dd = args[0][3], args[1][0], args[2][0]

        sc, sd = calc_sc(it), calc_sd(it)

        dual_inf = calc_dual_inf(grad_lag_x, grad_lag_s)
        unscaled_dual_inf = calc_dual_inf_unscaled(grad_lag_x, grad_lag_s, df, dd)

        complementarity = calc_complementarity(it, mu, slacks)
        unscaled_complementarity = calc_complementarity_unscaled(it, mu, slacks, df)

        nlp_constr_viol = calc_nlp_constr_viol(c, d)
        unscaled_nlp_constr_viol = calc_nlp_constr_viol_unscaled(c, d, dc, dd)

        overall_error = jnp.max(
            jnp.hstack([dual_inf / sd, complementarity / sc, nlp_constr_viol])
        )

        converged = (
            (overall_error <= p["tol"])
            & (unscaled_dual_inf <= p["dual_inf_tol"])
            & (unscaled_nlp_constr_viol <= p["constr_viol_tol"])
            & (unscaled_complementarity <= p["compl_inf_tol"])
        )

        if p["DEBUG_MODE"]: print("WARNING: we are not handling current_is_acceptable control flow (this is current is acceptable in the opterrorconvcheck not the LS)")
        if p["DEBUG_MODE"]: print("WARNING: we are not handling DIVERGING, MAXITER, CPUTIME, WALTIME control flow")

        return converged, (overall_error, dual_inf, nlp_constr_viol, complementarity)

    def calc_current_is_acceptable(it, mu, grad_lag_x, grad_lag_s, slacks, c, d, curr_obj_val, last_obj_val, args):
        _, (overall_error, dual_inf, constr_viol, compl_inf) = calc_check_converged(
            it, mu, grad_lag_x, grad_lag_s, slacks, c, d, args
        )
        print("WARNING: we are assuming we have already calculated objval for this iter")

        return (
            (overall_error <= p["acceptable_tol"])
            & (dual_inf <= p["acceptable_dual_inf_tol"])
            & (constr_viol <= p["acceptable_constr_viol_tol"])
            & (compl_inf <= p["acceptable_compl_inf_tol"])
            & (
                jnp.abs(curr_obj_val - last_obj_val)
                / jnp.maximum(1.0, jnp.abs(curr_obj_val))
                <= p["acceptable_obj_change_tol"]
            )
        )

    def calc_avrg_compl(it, slacks):
        sxL, sxU, sdL, sdU = slacks
        ncomps = nxL + it.z_U.size + it.v_L.size + it.v_U.size
        if ncomps == 0:
            # in this case avoid division by zero, want result to be zero so set inf
            ncomps = jnp.inf
        return (
            it.z_L[:nxL].T @ sxL[:nxL]
            + it.z_U.T @ sxU
            + it.v_L.T @ sdL
            + it.v_U.T @ sdU
        ) / ncomps

    def calc_barrier_obj(f, mu, slacks):
        # calc_f(it.x[:nx], *f_args)
        sxL, sxU, sdL, sdU = slacks
        sxL_mask = dampind_x_L[ind_x_L].astype(jnp.bool)
        sxU_mask = dampind_x_U[ind_x_U].astype(jnp.bool)
        sdL_mask = dampind_d_L[ind_d_L].astype(jnp.bool)
        sdU_mask = dampind_d_U[ind_d_U].astype(jnp.bool)
        return (
            f
            - mu
            * (
                jnp.sum(jnp.log(sxL[:nxL]))
                + jnp.sum(jnp.log(sxU))
                + jnp.sum(jnp.log(sdL))
                + jnp.sum(jnp.log(sdU))
            )
            + p["kappa_d"]
            * mu
            * (
                jnp.linalg.norm(sxL[:nxL] * sxL_mask, ord=1)
                + jnp.linalg.norm(sxU * sxU_mask, ord=1)
                + jnp.linalg.norm(sdL * sdL_mask, ord=1)
                + jnp.linalg.norm(sdU * sdU_mask, ord=1)
            )
        )
    
    def calc_grad_barr_T_delta(d, grad_barrier_obj_x, grad_barrier_obj_s):
        return grad_barrier_obj_x[:nx].T @ d.x[:nx] + grad_barrier_obj_s.T @ d.s

    def calc_x_frac_to_bound(tau, slack_L, delta_slack_L, slack_U, delta_slack_U):
        alpha = 1.0
        if nxL > 0:
            ratios_L = -tau * slack_L / delta_slack_L[:nxL]
            masked_ratios_L = jnp.where(delta_slack_L[:nxL] < 0, ratios_L, jnp.inf)
            alpha = jnp.minimum(alpha, jnp.min(masked_ratios_L))
        if nxU > 0:
            ratios_U = -tau * slack_U / delta_slack_U
            masked_ratios_U = jnp.where(delta_slack_U < 0, ratios_U, jnp.inf)
            alpha = jnp.minimum(alpha, jnp.min(masked_ratios_U))
        return alpha

    # trace share compatible - but non-trace share functions depend on it
    def calc_s_frac_to_bound(tau, slack_L, delta_slack_L, slack_U, delta_slack_U):
        alpha = 1.0
        if ndL > 0:
            ratios_L = -tau * slack_L / delta_slack_L
            masked_ratios_L = jnp.where(delta_slack_L < 0, ratios_L, jnp.inf)
            alpha = jnp.minimum(alpha, jnp.min(masked_ratios_L))
        if ndU > 0:
            ratios_U = -tau * slack_U / delta_slack_U
            masked_ratios_U = jnp.where(delta_slack_U < 0, ratios_U, jnp.inf)
            alpha = jnp.minimum(alpha, jnp.min(masked_ratios_U))
        return alpha

    def calc_alpha_pr(tau, slacks, slack_derivatives):
        sxL, sxU, sdL, sdU = slacks
        dsxL, dsxU, dsdL, dsdU = slack_derivatives
        alpha_pr_x = calc_x_frac_to_bound(tau, sxL[:nxL], dsxL[:nxL], sxU, dsxU)
        alpha_pr_s = calc_s_frac_to_bound(tau, sdL, dsdL, sdU, dsdU)
        return jnp.minimum(alpha_pr_x, alpha_pr_s)

    def calc_alpha_du(it, d, tau):
        alpha_du_z = calc_x_frac_to_bound(tau, it.z_L[:nxL], d.z_L, it.z_U, d.z_U)
        alpha_du_v = calc_s_frac_to_bound(tau, it.v_L, d.v_L, it.v_U, d.v_U)
        return jnp.minimum(alpha_du_z, alpha_du_v)

    # trace share compatible
    def calc_transform_aug_to_full(it, aug, full_rhs, slacks, fl):
        """
        aug: {x, [nc, pc, nd, pd], s, y_c, y_d}, 
        full_rhs: {x, [nc, pc, nd, pd], s, y_c, y_d, z_L, [ncL, pcL, ndL, pdL], z_U, v_L, v_U}
        it.x: {x, [nc, pc, nd, pd]}
        it.z_L: {z_L, [ncL, pcL, ndL, pdL]}
        """
        sxL, sxU, sdL, sdU = slacks
        # ensures no division by zero when passing regular instead of resto
        sxL_reg = sxL.at[_nxL:].set(jnp.full([nyc*2+nyd*2,1], jnp.inf))
        sxL = jnp.where(fl.in_restoration.squeeze(), sxL, sxL_reg)
        cnt = 0
        d_x = aug[cnt:cnt+_nx+nyc*2+nyd*2];  cnt += _nx+nyc*2+nyd*2
        d_s = aug[cnt:cnt+nyd];  cnt += nyd
        d_y_c = aug[cnt:cnt+nyc];  cnt += nyc
        d_y_d = aug[cnt:cnt+nyd];  cnt += nyd
        d_z_L = (full_rhs.z_L - it.z_L * d_x[ind_x_Lr]) / sxL
        d_z_U = (full_rhs.z_U + it.z_U * d_x[ind_x_Ur]) / sxU
        d_v_L = (full_rhs.v_L - it.v_L * d_s[ind_d_L]) / sdL
        d_v_U = (full_rhs.v_U + it.v_U * d_s[ind_d_U]) / sdU
        out = jnp.vstack([d_x, d_s, d_y_c, d_y_d, d_z_L, d_z_U, d_v_L, d_v_U])
        # if jnp.isnan(out).any():
        #     pass
        return out

    nstqf = NonSharedTraceQuantityFunctions(
        calc_f=calc_f,
        calc_jac_f=calc_jac_f,
        calc_hess_f=calc_hess_f,
        calc_c=calc_c,
        calc_jac_c=calc_jac_c, 
        calc_hess_c=calc_hess_c,
        calc_d=calc_d,
        calc_jac_d=calc_jac_d, 
        calc_hess_d=calc_hess_d,
        calc_theta=calc_theta,
        calc_slacks=calc_slacks,
        calc_slack_derivatives=calc_slack_derivatives,
        calc_grad_barrier_obj_x=calc_grad_barrier_obj_x,
        calc_grad_lag_x=calc_grad_lag_x,
        calc_complementarity=calc_complementarity,
        calc_complementarity_unscaled=calc_complementarity_unscaled,
        calc_lower_mu_safeguard=calc_lower_mu_safeguard,
        calc_nlp_error=calc_nlp_error,
        calc_check_converged=calc_check_converged,
        calc_current_is_acceptable=calc_current_is_acceptable,
        calc_avrg_compl=calc_avrg_compl,
        calc_barrier_obj=calc_barrier_obj,
        calc_grad_barr_T_delta=calc_grad_barr_T_delta,
        calc_x_frac_to_bound=calc_x_frac_to_bound,
        calc_alpha_pr=calc_alpha_pr,
        calc_alpha_du=calc_alpha_du,
        kkt=kkt,
        # trace share compatible - but non-trace share functions depend on them
        calc_dual_inf=calc_dual_inf,
        calc_dual_inf_unscaled=calc_dual_inf_unscaled,
        calc_nlp_constr_viol=calc_nlp_constr_viol,
        calc_nlp_constr_viol_unscaled=calc_nlp_constr_viol_unscaled,
        calc_primal_inf_L1=calc_primal_inf_L1,
        calc_s_frac_to_bound=calc_s_frac_to_bound,
        calc_transform_aug_to_full=calc_transform_aug_to_full
    )

    return nstqf

def generate_shared_trace_functions(general_dims, kkt, p, ls_coo_indices=None, ls_nnz_triu=None):

    # nx, nxL, x_L, x_U, ind_x_L, ind_x_U, dampind_x_L, dampind_x_U = phase_dims
    nx, nyc, nyd, nxL, nxU, ndL, ndU, ind_x_L, ind_x_U, ind_d_L, ind_d_U, ind_d_L, dampind_d_L, dampind_d_U, ind_np_L, ind_np_U = general_dims
    nxLr = nxL + nyc * 2 + nyd * 2  # reduced size of z_L in resto
    coo_indices, W_full_coo_indices, dxs_diag_indices, dcd_diag_indices, nnz_triu, W_nnz_triu, W_nnz = kkt


    # trace share compatible
    def calc_grad_barrier_obj_s(mu, slacks):
        _, _, sdL, sdU = slacks
        if p["DEBUG_MODE"]: print("WARNING: safe slack division not used!!! This could diverge us from IPOPT")
        return (
            +spu.expand_vector(ind_d_L, [nyd, 1], -mu / sdL)
            + spu.expand_vector(ind_d_U, [nyd, 1], mu / sdU)
            + p["kappa_d"] * mu * (dampind_d_L - dampind_d_U)[:, None]
        )
    
    # trace share compatible
    def calc_alpha_min(grad_barr_T_delta, theta, theta_min):
        alpha_min = p["gamma_theta"]
        val_if_gBD_neg = p["gamma_phi"] * theta / (-grad_barr_T_delta)
        val_if_theta_small = (p["delta"] * theta ** p["s_theta"] / ((-grad_barr_T_delta) ** p["s_phi"]))
        alpha_min = jnp.full_like(grad_barr_T_delta, p["gamma_theta"])  # full_like ensures correct shaping
        alpha_min = jnp.where(grad_barr_T_delta < 0, jnp.minimum(p["gamma_theta"], val_if_gBD_neg), alpha_min)
        is_theta_small_cond = (grad_barr_T_delta < 0) & (theta <= theta_min)
        alpha_min = jnp.where(is_theta_small_cond, jnp.minimum(alpha_min, val_if_theta_small), alpha_min)
        return p["alpha_min_frac"] * alpha_min

    # trace share compatible
    def calc_grad_lag_s(it):
        return (
            -it.y_d
            - spu.expand_vector(ind_d_L, (nyd, 1), it.v_L)
            + spu.expand_vector(ind_d_U, (nyd, 1), it.v_U)
        )

    # trace share compatible
    def calc_barrier_constr_viol(c, dms, ord=p["constraint_violation_norm_type"]):
        """Barrier subproblem constraint violation (for line search/filter)"""
        return jnp.linalg.norm(jnp.vstack([c, dms]), ord=ord, keepdims=True)

    # trace share compatible
    def calc_vector_to_iterate(v):
        """padded vector input: 
        {x, [nc, pc, nd, pd], s, y_c, y_d, z_L, [ncL, pcL, ndL, pdL], z_U, v_L, v_U}
        where the bracketed sections [] are only for resto, but we load in the whole thing
        every time to maintain consistent structures throughout"""
        cnt = 0
        x =   v[cnt:cnt+nx+nyc*2+nyd*2];   cnt += nx + nyc*2 + nyd*2 # skip nc, pc, nd, pd
        s =   v[cnt:cnt+nyd];                    cnt += nyd
        y_c = v[cnt:cnt+nyc];                    cnt += nyc
        y_d = v[cnt:cnt+nyd];                    cnt += nyd
        z_L = v[cnt:cnt+nxL+nyc*2+nyd*2];  cnt += nxL + nyc*2 + nyd*2 # skip nc, pc, nd, pd
        z_U = v[cnt:cnt+nxU];                    cnt += nxU
        v_L = v[cnt:cnt+ndL];                    cnt += ndL
        v_U = v[cnt:cnt+ndU]
        return Iterate(x, s, y_c, y_d, z_L, z_U, v_L, v_U)

    # trace share compatible
    def calc_iterate_to_vector(it):
        return jnp.vstack([it.x, it.s, it.y_c, it.y_d, it.z_L, it.z_U, it.v_L, it.v_U])

    # trace share compatible
    def compare_le(lhs, rhs, basval):
        return lhs - rhs <= 10.0 * p["eps"] * jnp.abs(basval)

    # trace share compatible
    def is_ftype(ref_theta, ref_gBD, alpha_pr):
        swap_gBD_cond = (ref_theta == 0.0) & (ref_gBD > 0) & (ref_gBD < 100.0 * p["eps"])
        ref_gBD = jnp.where(swap_gBD_cond, -p["eps"], ref_gBD)
        # original criterion: (ref_gBD < 0) & (alpha_pr * (-ref_gBD) ** p["s_phi"] > p["delta"] * ref_theta ** p["s_theta"])
        # the problem is -ref_gBD ** s_phi can be nan if ref_gBD > 0, therefore the and cond with jnp.abs is what I use
        return (ref_gBD < 0) & (alpha_pr * jnp.abs(ref_gBD) ** p["s_phi"] > p["delta"] * ref_theta ** p["s_theta"])

    # trace share compatible
    def armijo_holds(trial_barr, ref_barr, ref_gBD, alpha_pr_test):
        return compare_le(trial_barr - ref_barr, p["eta_phi"] * alpha_pr_test * ref_gBD, ref_barr)

    # trace share compatible
    def is_acceptable_to_current_iterate(trial_barr, trial_theta, ref_barr, ref_theta, in_restoration):
        basval = jnp.where(jnp.abs(ref_barr) > 10.0, jnp.log10(jnp.abs(ref_barr)), 1.0)
        # Check if the barrier objective function is increasing too rapidly (according to option obj_max_inc)
        # we only want to execute trial_barr - ref_barr if trial_barr > ref_barr, therefore
        # I ALWAYS calculate the absolute value and select whether or not to use it later
        log_check_fails = (jnp.log10(jnp.abs(trial_barr - ref_barr)) > p["obj_max_inc"] + basval)
        safeguard_is_active = (~in_restoration) & (trial_barr > ref_barr)
        reject_due_to_safeguard = safeguard_is_active & log_check_fails
        is_theta_reduced = compare_le(trial_theta, (1.0 - p["gamma_theta"]) * ref_theta, ref_theta)
        is_barr_reduced = compare_le(trial_barr - ref_barr, -p["gamma_phi"] * ref_theta, ref_barr)
        filter_is_satisfied = is_theta_reduced | is_barr_reduced
        return (~reject_due_to_safeguard) & filter_is_satisfied

    # trace share compatible
    def is_acceptable_to_current_filter(trial_barr, trial_theta, filter):
        # IPOPT requires acceptable to ALL entries (not ANY)
        # For each entry: trial must be better in at least one dimension (barr OR theta)
        # If ANY entry rejects (trial worse in BOTH dimensions), return False
        barr_acceptable = trial_barr <= filter[:, 2]
        theta_acceptable = trial_theta <= filter[:, 1]
        elementwise_acceptable = jnp.logical_or(barr_acceptable, theta_acceptable)
        # Empty entries (inf) always accept, so jnp.all works correctly
        acceptable = jnp.all(elementwise_acceptable)
        return acceptable

    # trace share compatible
    def augment_raw_filter(F, val_theta, val_phi, iter_count):
        dominate_existing = (F[:, 1] >= val_theta) & (F[:, 2] >= val_phi)
        # Preserve column 0 (iteration index), only clear columns 1 and 2 to inf
        replacement = jnp.column_stack([F[:, 0], jnp.full(F.shape[0], jnp.inf), jnp.full(F.shape[0], jnp.inf)])
        F_cleaned = jnp.where(dominate_existing.T, replacement, F)
        is_empty = F_cleaned[:, 1] == jnp.inf
        target_idx = jnp.argmax(is_empty)
        new_row = jnp.array([iter_count, val_theta, val_phi]).reshape(1, 3)
        should_write = is_empty[target_idx]
        # Update unconditionally, then select
        F_updated = jax.lax.dynamic_update_slice(F_cleaned, new_row, (target_idx, 0))
        # Choose between updated and original based on should_write
        F_augmented = jnp.where(should_write, F_updated, F_cleaned)
        return F_augmented
    
    # trace share compatible
    def augment_ls_filter(F, ref_barr, ref_theta, iter_count):
        val_phi = ref_barr - p["gamma_phi"] * ref_theta
        val_theta = (1 - p["gamma_theta"]) * ref_theta
        return augment_raw_filter(F, val_theta, val_phi, iter_count)

    # trace share compatible
    def ls_reset(ls):
        cleared_filter = jnp.full_like(ls.filter.F, jnp.inf)
        # Preserve the iteration index column (first column should be 0, 1, 2, ...)
        cleared_filter = cleared_filter.at[:, 0].set(jnp.arange(ls.filter.F.shape[0]))
        return eqx.tree_at(
            lambda t: (
                t.filter.last_rejection_due_to_filter,
                t.filter.count_successive_filter_rejections,
                t.filter.F
            ), ls,
            (
                jnp.array([[0]]),
                jnp.array([[0]]),
                cleared_filter
            )
        )

    # trace share compatible
    def check_acceptability_of_trial_point(
            trial_barr, trial_theta, ref_barr, ref_theta, ref_gBD, theta_min, theta_max, alpha_pr_test, 
            last_rejection_due_to_filter, count_successive_filter_rejections, n_filter_resets, filter, in_restoration
        ):

        # compute all acceptance checks as boolean values (theta_max can == 0 potentially)
        accept_sanity = (theta_max <= 0) | (trial_theta <= theta_max)
        use_armijo_cond = (
            (alpha_pr_test > 0.)
            & is_ftype(ref_theta, ref_gBD, alpha_pr_test)
            & (ref_theta <= theta_min)
        )
        accept_primary = jnp.where(
            use_armijo_cond,
            armijo_holds(trial_barr, ref_barr, ref_gBD, alpha_pr_test),
            is_acceptable_to_current_iterate(trial_barr, trial_theta, ref_barr, ref_theta, in_restoration)
        )
        accept_historical = is_acceptable_to_current_filter(trial_barr, trial_theta, filter)

        # we may not reject, but if we do then we note its cause
        rejection_due_to_filter = accept_sanity & accept_primary & ~accept_historical

        # only if all conditions are met do we accept
        final_accept = accept_sanity & accept_primary & accept_historical

        # Debug logging
        # log_accept_check(trial_theta, ref_theta, trial_barr, ref_barr, alpha_pr_test, in_restoration, accept_sanity, accept_primary, accept_historical, final_accept)

        # filter reset heuristics
        if p["max_filter_resets"] > 0:
            # max_filter_resets is a static param
            count_after_rejection = jnp.where(
                last_rejection_due_to_filter,
                count_successive_filter_rejections + 1,
                0,
            )
            reset_is_triggered = (
                (count_after_rejection >= p["filter_reset_trigger"])
                & (n_filter_resets < p["max_filter_resets"])
                & (p["max_filter_resets"] > 0) # Handles static option
            )
            reset_filter = jnp.vstack([jnp.hstack([theta_max, p["varphi_max"]])]*p["filter_size"])
            reset_filter = jnp.hstack([jnp.arange(-p["filter_size"],0)[:,None], reset_filter])

            final_filter, final_count, final_n_resets = jax.lax.cond(
                reset_is_triggered.squeeze(),
                lambda: (reset_filter, jnp.zeros_like(count_after_rejection), n_filter_resets + 1),
                lambda: (filter, count_after_rejection, n_filter_resets)
            )
        else:
            final_filter = filter
            final_count = count_successive_filter_rejections
            final_n_resets = n_filter_resets

        return final_accept, final_filter, final_count, final_n_resets, rejection_due_to_filter

    # =========================================================================
    # Least squares multiplier functions (for resto exit)
    # System: [I   0  J_c  J_d] [dx ]   [rhs_x]
    #         [0   I  0   -I  ] [ds ] = [rhs_s]
    #         [Jc' 0  0    0  ] [y_c]   [0    ]
    #         [Jd'-I  0    0  ] [y_d]   [0    ]
    # =========================================================================
    def calc_ls_mults_RHS(jac_f, z_L, z_U, v_L, v_U):
        """Build RHS for least squares multiplier calculation.
        Uses original problem dimensions (not resto expanded).
        """
        # rhs_x = -grad_f + z_L - z_U (original nx)
        rhs_x = (
            -jac_f[:nx]
            + spu.expand_vector(ind_x_L, (nx, 1), z_L)
            - spu.expand_vector(ind_x_U, (nx, 1), z_U)
        )
        # rhs_s = v_L - v_U
        rhs_s = spu.expand_vector(ind_d_L, (nyd, 1), v_L) - spu.expand_vector(ind_d_U, (nyd, 1), v_U)
        rhs_c = jnp.zeros([nyc, 1])
        rhs_d = jnp.zeros([nyd, 1])
        return jnp.vstack([rhs_x, rhs_s, rhs_c, rhs_d])

    def calc_ls_mults_LHS(jacobians):
        """Build LHS for least squares multiplier calculation.
        System with W=0, Sigma=0, identity blocks on diagonal.
        Conforms to unified sparsity pattern.
        """
        _, Jc, Jd = jacobians
        Jc = Jc[:nx]  # Original problem jacobians
        Jd = Jd[:nx]

        # LHS: [I  0  Jc  Jd ]
        #      [0  I  0  -I  ]
        #      [0  0  0   0  ]  (delta_c = 0)
        #      [0  0  0   0  ]  (delta_d = 0)
        _ls_diag_pert = 0.0 if p["VALIDATION_MODE"] is True else -1e-13
        LHS_upper_triangular = spu.vstack([
            spu.hstack([spu.eye(nx),          spu.zeros([nx, nyd]),  Jc,                            Jd                                 ]),
            spu.hstack([spu.zeros([nyd, nx]), spu.eye(nyd),           spu.zeros([nyd, nyc]),         -spu.eye(nyd)                      ]),
            spu.hstack([spu.zeros([nyc, nx]), spu.zeros([nyc, nyd]),  _ls_diag_pert * spu.eye(nyc),  spu.zeros([nyc, nyd])             ]),
            spu.hstack([spu.zeros([nyd, nx]), spu.zeros([nyd, nyd]),  spu.zeros([nyd, nyc]),         _ls_diag_pert * spu.eye(nyd)      ]),
        ])

        LHS_upper_triangular = spu.sum_duplicates_to_pattern(ls_coo_indices, LHS_upper_triangular, nse=ls_nnz_triu)
        LHS_upper_triangular = LHS_upper_triangular.sort_indices()
        LHS_upper_triangular = spu.conform_bcoo_to_new_sparsity(ls_coo_indices, LHS_upper_triangular)
        return LHS_upper_triangular

    stqf = SharedTraceQuantityFunctions(
        calc_grad_barrier_obj_s=calc_grad_barrier_obj_s,
        calc_alpha_min=calc_alpha_min,
        calc_grad_lag_s=calc_grad_lag_s,
        calc_vector_to_iterate=calc_vector_to_iterate,
        calc_iterate_to_vector=calc_iterate_to_vector,
        # calc_transform_aug_to_full=calc_transform_aug_to_full,
        is_ftype=is_ftype,
        armijo_holds=armijo_holds,
        is_acceptable_to_current_iterate=is_acceptable_to_current_iterate,
        is_acceptable_to_current_filter=is_acceptable_to_current_filter,
        augment_raw_filter=augment_raw_filter,
        augment_ls_filter=augment_ls_filter,
        ls_reset=ls_reset,
        check_acceptability_of_trial_point=check_acceptability_of_trial_point,
        calc_barrier_constr_viol=calc_barrier_constr_viol,
        calc_ls_mults_RHS=calc_ls_mults_RHS,
        calc_ls_mults_LHS=calc_ls_mults_LHS
    )

    return stqf

# TODO once we rework the initialization
def calc_values_pre_mu(it, ic, cp, fl, fun_outs, jacobians, hessians, quantities, iter_count):

    # inertia correction loading workaround ------------------------------------
    # def load_pert_x(iter_count):
    #     perts = load_scalars(f"{save_dir}/perturbation_{int(iter_count.squeeze())}.txt")
    #     return perts["delta_x"], perts["delta_s"], perts["delta_c"], perts["delta_d"]

    # # Inside JIT
    # _pert_x, _pert_s, _pert_c, _pert_d = jax.pure_callback(
    #     load_pert_x,
    #     (
    #         jax.ShapeDtypeStruct(shape=(), dtype=jnp.float64),  # must specify shape/dtype
    #         jax.ShapeDtypeStruct(shape=(), dtype=jnp.float64),  # must specify shape/dtype
    #         jax.ShapeDtypeStruct(shape=(), dtype=jnp.float64),  # must specify shape/dtype
    #         jax.ShapeDtypeStruct(shape=(), dtype=jnp.float64),  # must specify shape/dtype
    #     ),
    #     iter_count.squeeze(), vmap_method="sequential"
    # )

    # _perts = (_pert_x, _pert_s, _pert_c, _pert_d)
    # Build LHS with zero perturbation - inertia correction adds dxs via diagonal indices
    perts = (jnp.array(0.0), jnp.array(0.0), jnp.zeros(1), jnp.zeros(1))
    f, c, d = fun_outs
    grad_lag_x, slacks, theta, avrg_compl, grad_lag_s = quantities

    # calculate everything we can ahead of time for the mu calc ----------------
    resto = fl.in_restoration.squeeze()
    rnx = cp.nx + cp.nyc * 2 + cp.nyd * 2
    rnxL = cp.nxL + cp.nyc * 2 + cp.nyd * 2
    rind_x_L = jnp.hstack([cp.ind_x_L, cp.ind_np_L + cp.nx])
    rind_x_U = jnp.hstack([cp.ind_x_U, cp.ind_np_U + cp.nx])
    rx_L = jnp.vstack([cp.x_L, cp.np_L])
    rx_U = jnp.vstack([cp.x_U, cp.np_U])

    dms = d - it.s
    grad_lag_s = cp.stqf.calc_grad_lag_s(it)  # does not need function args
    y_nrminf = jnp.maximum(jnp.linalg.norm(it.y_c, ord=jnp.inf), jnp.linalg.norm(it.y_d, ord=jnp.inf))

    # regular padded values
    pad = jnp.zeros([cp.nyc * 2 + cp.nyd * 2, 1])  # how much to pad by
    def pad_iterate_vector(v):
        insert1 = cp.nx
        insert2 = cp.nx+cp.nyd+cp.nyc+cp.nyd+cp.nxL
        return jnp.vstack([v[:insert1], pad, v[insert1:insert2], pad, v[insert2:]])

    # resto specific
    Sigma_x = spu.expand_vector(rind_x_L, (rnx, 1), it.z_L[:rnxL]) / (it.x[:rnx] - rx_L) + spu.expand_vector(rind_x_U, (rnx, 1), it.z_U) / (rx_U - it.x[:rnx])
    Sigma_x_only = Sigma_x[:cp.nx]
    Sigma_nc = Sigma_x[cp.nx:cp.nx+cp.nyc]
    Sigma_pc = Sigma_x[cp.nx+cp.nyc:cp.nx+cp.nyc*2]
    Sigma_nd = Sigma_x[cp.nx+cp.nyc*2:cp.nx+cp.nyc*2+cp.nyd]
    Sigma_pd = Sigma_x[cp.nx+cp.nyc*2+cp.nyd:]

    # Compute initial Sigma_inv with dxs=0 (no perturbation assumed initially)
    # IC will update the RHS when dxs changes, matching IPOPT's behavior
    rSigma_nc_inv = 1 / jnp.maximum(Sigma_nc, 1e-20)
    rSigma_pc_inv = 1 / jnp.maximum(Sigma_pc, 1e-20)
    rSigma_nd_inv = 1 / jnp.maximum(Sigma_nd, 1e-20)
    rSigma_pd_inv = 1 / jnp.maximum(Sigma_pd, 1e-20)

    Sigma_nc_inv, Sigma_pc_inv, Sigma_nd_inv, Sigma_pd_inv = filter_select(
        resto, 
        (
            rSigma_nc_inv,
            rSigma_pc_inv,
            rSigma_nd_inv,
            rSigma_pd_inv
        ), (
            jnp.zeros_like(rSigma_nc_inv),
            jnp.zeros_like(rSigma_pc_inv),
            jnp.zeros_like(rSigma_nd_inv),
            jnp.zeros_like(rSigma_pd_inv)
        )
    )


    rrhs_aff_full = cp.stqf.calc_vector_to_iterate(cp.nstqfr.kkt.calc_resto_red_pd_RHS_aff(it, slacks, grad_lag_x, grad_lag_s, c, dms, Sigma_nc_inv, Sigma_pc_inv, Sigma_nd_inv, Sigma_pd_inv))
    rrhs_cen_full = cp.stqf.calc_vector_to_iterate(cp.nstqfr.kkt.calc_resto_red_pd_RHS_cen(it, slacks, avrg_compl, Sigma_nc_inv, Sigma_pc_inv, Sigma_nd_inv, Sigma_pd_inv))
    raug_LHS_upper_triangular = cp.nstqfr.kkt.calc_resto_red_pd_LHS_given_deltas(it, jacobians, hessians, perts, Sigma_x_only, Sigma_nc_inv, Sigma_pc_inv, Sigma_nd_inv, Sigma_pd_inv)

    # Extract intermediates for IC RHS update (rrhs_aff_full.x is -mod_rhs_x, so negate to get mod_rhs)
    # These are needed because rhs_cR = rhs_c - Sigma_nc_inv*mod_rhs_nc + Sigma_pc_inv*mod_rhs_pc
    # and when IC changes dxs, Sigma_inv changes, so rhs_cR/rhs_dR must be recomputed
    rhs_intermediates_aff = (
        -rrhs_aff_full.x[cp.nx : cp.nx + cp.nyc],                      # mod_rhs_nc
        -rrhs_aff_full.x[cp.nx + cp.nyc : cp.nx + 2*cp.nyc],           # mod_rhs_pc
        -rrhs_aff_full.x[cp.nx + 2*cp.nyc : cp.nx + 2*cp.nyc + cp.nyd], # mod_rhs_nd
        -rrhs_aff_full.x[cp.nx + 2*cp.nyc + cp.nyd :],                  # mod_rhs_pd
        c,   # rhs_c
        dms  # rhs_d
    )

    # regular specific

    # although this says "full" it is the augmented RHS stacked on the bounds mults for full RHS
    rhs_aff_full = cp.stqf.calc_vector_to_iterate(pad_iterate_vector(cp.nstqf.kkt.calc_aug_pd_RHS_aff(it, slacks, grad_lag_x, grad_lag_s, c, dms)))
    rhs_cen_full = cp.stqf.calc_vector_to_iterate(pad_iterate_vector(cp.nstqf.kkt.calc_aug_pd_RHS_cen(it, slacks, avrg_compl)))
    aug_LHS_upper_triangular = cp.nstqf.kkt.calc_aug_pd_LHS_given_deltas(it, jacobians, hessians, perts) # .sum_duplicates()
    # _aug_LHS_upper_triangular = nstqf.kkt.calc_aug_pd_LHS_given_deltas(it, _perts, args) # .sum_duplicates()
    # aug_LHS_upper_triangular = nstqf.kkt.calc_aug_pd_LHS(it, args) # .sum_duplicates()

    # LS mults via dedicated solver (runs every iteration, separate from KKT)
    ls_init_lhs = cp.stqf.calc_ls_mults_LHS(jacobians)
    ls_init_csr = jsparse.BCSR.from_bcoo(ls_init_lhs)
    ls_init_rhs = cp.stqf.calc_ls_mults_RHS(
        jacobians[0], it.z_L[:cp.nxL], it.z_U, it.v_L, it.v_U
    )
    ls_step = cp.ls_refactorize_and_solve(ls_init_rhs.flatten(), ls_init_csr.data)[0][:, None]
    if cp.p["VALIDATION_MODE"] is True:
        ls_step = cp.ls_linear_solve(ls_init_rhs.flatten(), ls_init_csr.data)[0][:, None]
    y_c_init = ls_step[cp.nx + cp.nyd : cp.nx + cp.nyd + cp.nyc]
    y_d_init = ls_step[cp.nx + cp.nyd + cp.nyc : cp.nx + cp.nyd + cp.nyc + cp.nyd]

    # sort out the KKT systems conditionally on resto or not
    rhs_aff, rhs_cen, csr_lhs = filter_select(
        resto, (
            jnp.vstack([rrhs_aff_full.x[: cp.nx], rrhs_aff_full.s, rrhs_aff_full.y_c, rrhs_aff_full.y_d]), # reduced
            jnp.vstack([rrhs_cen_full.x[: cp.nx], rrhs_cen_full.s, rrhs_cen_full.y_c, rrhs_cen_full.y_d]), # reduced
            jsparse.BCSR.from_bcoo(raug_LHS_upper_triangular)  # NOT sum duplicates - reduced
        ), (
            jnp.vstack([rhs_aff_full.x[:cp.nx], rhs_aff_full.s, rhs_aff_full.y_c, rhs_aff_full.y_d]), # augmented
            jnp.vstack([rhs_cen_full.x[:cp.nx], rhs_cen_full.s, rhs_cen_full.y_c, rhs_cen_full.y_d]), # augmented
            jsparse.BCSR.from_bcoo(aug_LHS_upper_triangular) # augmented
        )
    )

    ic = eqx.tree_at(lambda t: t.perturbed_data, ic, csr_lhs.data)

    # if we are init_regular we need to calculate and update the scaling in the args
    

    # For inertia correction: pass raw Sigma values for resto DcR/DdR updates
    # For regular mode, pass zeros (same shape for vmap compatibility)
    ic_Sigma_nc, ic_Sigma_pc, ic_Sigma_nd, ic_Sigma_pd = filter_select(
        resto,
        (Sigma_nc, Sigma_pc, Sigma_nd, Sigma_pd),
        (jnp.zeros_like(Sigma_nc), jnp.zeros_like(Sigma_pc), jnp.zeros_like(Sigma_nd), jnp.zeros_like(Sigma_pd))
    )

    # For inertia correction RHS update: pass intermediates for resto, zeros for regular
    # In regular mode, RHS doesn't depend on dxs, so intermediates aren't needed
    ic_rhs_intermediates = filter_select(
        resto,
        rhs_intermediates_aff,
        tuple(jnp.zeros_like(x) for x in rhs_intermediates_aff)
    )

    # COMMON INERTIA CORRECTION CODE - and common calls to linear solve
    step_aff, ic = solve_with_inertia_correction(csr_lhs, rhs_aff, ic, cp, fl, resto, ic_Sigma_nc, ic_Sigma_pc, ic_Sigma_nd, ic_Sigma_pd, ic_rhs_intermediates)

    # Recompute Sigma_inv with final ic.dxs for resto transformation (reduced -> aug)
    # IPOPT also uses sigma_tilde_inv = 1/(Sigma + delta_x) in its Solve() for the transformation
    Sigma_nc_inv_final = 1 / jnp.maximum(Sigma_nc + ic.dxs, 1e-20)
    Sigma_pc_inv_final = 1 / jnp.maximum(Sigma_pc + ic.dxs, 1e-20)
    Sigma_nd_inv_final = 1 / jnp.maximum(Sigma_nd + ic.dxs, 1e-20)
    Sigma_pd_inv_final = 1 / jnp.maximum(Sigma_pd + ic.dxs, 1e-20)

    # Recompute centering RHS with final Sigma_inv (for resto mode, rhs_cen depends on dxs)
    rrhs_cen_full_final = cp.stqf.calc_vector_to_iterate(cp.nstqfr.kkt.calc_resto_red_pd_RHS_cen(
        it, slacks, avrg_compl, Sigma_nc_inv_final, Sigma_pc_inv_final, Sigma_nd_inv_final, Sigma_pd_inv_final))
    rhs_cen_final, = filter_select(
        resto,
        (jnp.vstack([rrhs_cen_full_final.x[: cp.nx], rrhs_cen_full_final.s, rrhs_cen_full_final.y_c, rrhs_cen_full_final.y_d]),),
        (rhs_cen,)  # regular mode doesn't depend on dxs
    )

    # empirically found we have to refactorize one more time to avoid iterative refinement issues...
    step_cen = cp.linear_solve(rhs_cen_final.flatten(), ic.perturbed_data)[0][:,None]

    if cp.p["VALIDATION_MODE"] is True:
        rrhs_aff_full_final = cp.stqf.calc_vector_to_iterate(cp.nstqfr.kkt.calc_resto_red_pd_RHS_aff(
            it, slacks, grad_lag_x, grad_lag_s, c, dms,
            Sigma_nc_inv_final, Sigma_pc_inv_final, Sigma_nd_inv_final, Sigma_pd_inv_final))
        rhs_aff_final, = filter_select(
            resto,
            (jnp.vstack([rrhs_aff_full_final.x[: cp.nx], rrhs_aff_full_final.s, rrhs_aff_full_final.y_c, rrhs_aff_full_final.y_d]),),
            (rhs_aff,)  # regular mode doesn't depend on dxs
        )

        step_aff = cp.linear_solve(rhs_aff_final.flatten(), ic.perturbed_data)[0][:, None]
    else:
        rrhs_aff_full_final = rrhs_aff_full

    # resto specific step processing - reduced -> aug -> full
    rstep_aff_aug = cp.nstqfr.kkt.calc_transform_red_to_aug(step_aff, rrhs_aff_full_final, Sigma_nc_inv_final, Sigma_pc_inv_final, Sigma_nd_inv_final, Sigma_pd_inv_final)
    rstep_aff_full = cp.nstqfr.calc_transform_aug_to_full(it, rstep_aff_aug, rrhs_aff_full_final, slacks, fl)

    rstep_cen_aug = cp.nstqfr.kkt.calc_transform_red_to_aug(step_cen, rrhs_cen_full_final, Sigma_nc_inv_final, Sigma_pc_inv_final, Sigma_nd_inv_final, Sigma_pd_inv_final)
    rstep_cen_full = cp.nstqfr.calc_transform_aug_to_full(it, rstep_cen_aug, rrhs_cen_full_final, slacks, fl)

    # regular specific step processing - aug -> full

    # empirically this works with cudss - we must refactorize at least once after factorizing and solving for 
    # subsequent solves and iterative refinements to work properly at all. If you just
    # factorize and solve, and reuse that factorization and just solve and iterative refine
    # in the next step, then the iterative refinement diverges your answer to infinity I have found
    # in this case at least. Maybe this is a ME problem, but it seems weird behaviour regardless.
    # step_cen_aug = cp.refactorize_and_linear_solve(rhs_cen.flatten(), ic.perturbed_data)[0][:, None]
    step_aff_aug = jnp.vstack([step_aff[:cp.nx], pad, step_aff[cp.nx:]])
    step_cen_aug = jnp.vstack([step_cen[:cp.nx], pad, step_cen[cp.nx:]])
    step_aff_full = cp.nstqf.calc_transform_aug_to_full(it, step_aff_aug, rhs_aff_full, slacks, fl)
    step_cen_full = cp.nstqf.calc_transform_aug_to_full(it, step_cen_aug, rhs_cen_full, slacks, fl)

    # for structure parity with resto - pad shape AND nse to match resto jacobians  
    step_aff_full, step_cen_full, nlp_error, nlp_constr_viol, barrier_constr_viol, primal_inf = filter_select(
        resto, (
            rstep_aff_full, rstep_cen_full,
            cp.nstqfr.calc_nlp_error(it, grad_lag_x, grad_lag_s, slacks, jnp.array([[0.]]), c, d),
            cp.nstqfr.calc_nlp_constr_viol(c, d),
            cp.stqf.calc_barrier_constr_viol(c, dms),
            cp.nstqfr.calc_primal_inf_L1(c, dms)
        ), (
            step_aff_full, step_cen_full, 
            cp.nstqf.calc_nlp_error(it, grad_lag_x, grad_lag_s, slacks, jnp.array([[0.]]), c, d),
            cp.nstqf.calc_nlp_constr_viol(c, d),
            cp.stqf.calc_barrier_constr_viol(c, dms),
            cp.nstqf.calc_primal_inf_L1(c, dms)
        )
    )

    grad_lag_x_nrm2 = jnp.linalg.norm(grad_lag_x.flatten(), ord=2)
    grad_lag_s_nrm2 = jnp.linalg.norm(grad_lag_s.flatten(), ord=2)
    c_nrm2 = jnp.linalg.norm(c.flatten(), ord=2)
    d_minus_s_nrm2 = jnp.linalg.norm(dms.flatten(), ord=2)

    jac_f = jacobians[0]

    cqpr = CalculatedQuantitiesPreMu(
        f=f, c=c, d=d, 
        jac_f=jac_f,
        # jac_c_data=jac_c.data,
        # jac_c_indices=jac_c.indices, 
        # jac_d_data=jac_d.data,
        # jac_d_indices=jac_d.indices, 
        # hess_f_data=hess_f.data,
        # hess_f_indices=hess_f.indices, 
        # hess_c_data=hess_c.data,
        # hess_c_indices=hess_c.indices, 
        # hess_d_data=hess_d.data,
        # hess_d_indices=hess_d.indices, 
        dms=dms, y_nrminf=y_nrminf, grad_lag_x=grad_lag_x, grad_lag_s=grad_lag_s, 
        slacks=slacks, avrg_compl=avrg_compl, theta=theta,
        grad_lag_x_nrm2=grad_lag_x_nrm2, grad_lag_s_nrm2=grad_lag_s_nrm2, 
        c_nrm2=c_nrm2, d_minus_s_nrm2=d_minus_s_nrm2, nlp_error=nlp_error, 
        nlp_constr_viol=nlp_constr_viol, barrier_constr_viol=barrier_constr_viol, primal_inf=primal_inf, 
        step_aff_full=step_aff_full, step_cen_full=step_cen_full,
        # Use final Sigma_inv values (with ic.dxs) for resto, zeros for regular
        Sigma_pc_inv=jnp.where(resto, Sigma_pc_inv_final, jnp.zeros_like(Sigma_pc_inv_final)),
        Sigma_nd_inv=jnp.where(resto, Sigma_nd_inv_final, jnp.zeros_like(Sigma_nd_inv_final)),
        Sigma_pd_inv=jnp.where(resto, Sigma_pd_inv_final, jnp.zeros_like(Sigma_pd_inv_final)),
        Sigma_nc_inv=jnp.where(resto, Sigma_nc_inv_final, jnp.zeros_like(Sigma_nc_inv_final)),
        y_c_init=y_c_init,
        y_d_init=y_d_init
    )

    return cqpr, ic

def calc_values_post_mu(it, mu, tau, cqpr, ic, cp, fl):

    # lets unify KKT solving across resto/regular
    rhs_full_resto = cp.stqf.calc_vector_to_iterate(
        cp.nstqfr.kkt.calc_resto_red_pd_RHS(it, mu, cqpr.slacks, cqpr.grad_lag_x, cqpr.grad_lag_s, cqpr.c, cqpr.dms, cqpr.Sigma_nc_inv, cqpr.Sigma_pc_inv, cqpr.Sigma_nd_inv, cqpr.Sigma_pd_inv)
    )
    rhs_red_resto = jnp.vstack([rhs_full_resto.x[: cp.nx], rhs_full_resto.s, rhs_full_resto.y_c, rhs_full_resto.y_d])

    pad_reg = jnp.zeros([cp.nyc * 2 + cp.nyd * 2, 1])  # how much to pad by
    def pad_iterate_vector(v):
        insert1 = cp.nx
        insert2 = cp.nx+cp.nyd+cp.nyc+cp.nyd+cp.nxL
        return jnp.vstack([v[:insert1], pad_reg, v[insert1:insert2], pad_reg, v[insert2:]])
    rhs_full_reg = cp.stqf.calc_vector_to_iterate(
        pad_iterate_vector(cp.nstqf.kkt.calc_aug_pd_RHS(it, mu, cqpr.slacks, cqpr.grad_lag_x, cqpr.grad_lag_s, cqpr.c, cqpr.dms))
    )
    rhs_aug_reg = jnp.vstack([rhs_full_reg.x[:cp.nx], rhs_full_reg.s, rhs_full_reg.y_c, rhs_full_reg.y_d])

    # 2-way RHS selection: regular or resto (LS handled by dedicated solver in calc_values_pre_mu)
    rhs = jnp.where(fl.in_restoration.squeeze(), rhs_red_resto, rhs_aug_reg)

    step = cp.linear_solve(rhs.flatten(), ic.perturbed_data)[0][:, None]

    step_resto = step
    step_aug_reg = pad_iterate_vector(step)

    step_aug_resto = cp.nstqfr.kkt.calc_transform_red_to_aug(
        step_resto, rhs_full_resto, cqpr.Sigma_nc_inv, cqpr.Sigma_pc_inv, cqpr.Sigma_nd_inv, cqpr.Sigma_pd_inv
    )
    # nan avoid when computing resto with regular slacks which are padded with zeros?
    step_full_resto = cp.nstqfr.calc_transform_aug_to_full(it, step_aug_resto, rhs_full_resto, cqpr.slacks, fl)
    step_full_it_resto = cp.stqf.calc_vector_to_iterate(step_full_resto)
    slack_derivatives_resto = cp.nstqfr.calc_slack_derivatives(step_full_it_resto)

    step_full_it_reg = cp.stqf.calc_vector_to_iterate(
        cp.nstqf.calc_transform_aug_to_full(it, step_aug_reg, rhs_full_reg, cqpr.slacks, fl)
    )
    sld_reg = cp.nstqf.calc_slack_derivatives(step_full_it_reg)
    slack_derivatives_reg = (jnp.vstack([sld_reg[0], pad_reg]), *sld_reg[1:])

    slack_derivatives = filter_select(fl.in_restoration.squeeze(), slack_derivatives_resto, slack_derivatives_reg)

    alpha_pr = jnp.atleast_2d(jnp.where(
        fl.in_restoration.squeeze(), 
        cp.nstqfr.calc_alpha_pr(tau, cqpr.slacks, slack_derivatives),
        cp.nstqf.calc_alpha_pr(tau, cqpr.slacks, slack_derivatives)
    ))

    jac_f = cqpr.jac_f # assumes that jac_f is dense # jsparse.BCOO(cqpr.jac_f_data, cqpr.jac_f_indices, shape=[cp.nx+cp.nyc*2+cp.nyd*2,1])

    tmp = cp.nstqf.calc_grad_barrier_obj_x(jac_f, mu, cqpr.slacks)
    grad_barrier_obj_x_reg = jnp.vstack([tmp, jnp.zeros([cp.nyc*2+cp.nyd*2,1])])

    barr, grad_barrier_obj_x = filter_select(
        fl.in_restoration.squeeze(), 
        (
            cp.nstqfr.calc_barrier_obj(cqpr.f, mu, cqpr.slacks),
            cp.nstqfr.calc_grad_barrier_obj_x(jac_f, mu, cqpr.slacks)
        ), (
            cp.nstqf.calc_barrier_obj(cqpr.f, mu, cqpr.slacks),
            grad_barrier_obj_x_reg
        )
    )

    grad_barrier_obj_s = cp.stqf.calc_grad_barrier_obj_s(mu, cqpr.slacks)

    # OLD: gBD uses combined step: step_aff + sigma * step_cen (Mehrotra's formula)
    # # where sigma = mu / avrg_compl
    # sigma = mu / cqpr.avrg_compl
    # step_combined = cqpr.step_aff_full + sigma * cqpr.step_cen_full
    # step_combined_it = cp.stqf.calc_vector_to_iterate(step_combined)
    # gBD, rhs_full, step_full_it = filter_select(
    #     fl.in_restoration.squeeze(),
    #     (
    #         cp.nstqfr.calc_grad_barr_T_delta(step_combined_it, grad_barrier_obj_x, grad_barrier_obj_s),
    #         rhs_full_resto,
    #         step_full_it_resto
    #     ), (
    #         cp.nstqf.calc_grad_barr_T_delta(step_combined_it, grad_barrier_obj_x, grad_barrier_obj_s),
    #         rhs_full_reg,
    #         step_full_it_reg
    #     )
    # )

    # NEW: gBD uses actual KKT solve step (matching IPOPT's gradBarrTDelta)
    gBD, rhs_full, step_full_it = filter_select(
        fl.in_restoration.squeeze(),
        (
            cp.nstqfr.calc_grad_barr_T_delta(step_full_it_resto, grad_barrier_obj_x, grad_barrier_obj_s),
            rhs_full_resto,
            step_full_it_resto
        ), (
            cp.nstqf.calc_grad_barr_T_delta(step_full_it_reg, grad_barrier_obj_x, grad_barrier_obj_s),
            rhs_full_reg,
            step_full_it_reg
        )
    )

    cqpo = CalculatedQuantitiesPostMu(
        rhs_aug=rhs, # just for debugging
        step_aug=step, # just for debugging
        rhs=rhs_full,
        step=step_full_it,
        alpha_pr=alpha_pr,
        barr=barr,
        gBD=gBD,
        slack_derivatives=slack_derivatives
    )

    return cqpo

if __name__ == "__main__":
    pass


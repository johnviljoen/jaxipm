import functools as ft

import equinox as eqx
import jax
import jax.experimental.sparse as jsparse
import jax.numpy as jnp
from jax2sympy.sparsify import get_sparsity_pattern
from jax2sympy.sparsify_sym import sparse_jacobian_sym, sparse_hessian_sym
from spineax.cudss.solver import CuDSSSolver

# import jaxipm.quantities
import jaxipm.utils.sparse_utils as spu
from jaxipm.structures import (
    CommonProblem, # the common problem structure across a whole batch
    KKTCondensedFunctions, # condensed KKT closures
    LineSearchFilterState, # there is also a filter in the adaptive mu
    InertiaCorrectionState, # contains the perturbations which we remember
    Iterate, # the primal-dual iterate itself
    IterateFlags, # flags for control flow throughout the algorithm
    LineSearchState, # line search state
    OptimizationState, # the overall optimization state wrapping all other states
    WatchdogState, # the watchdog state
    CalculatedQuantitiesPreMu,
    CalculatedQuantitiesPostMu
)
from jaxipm.quantities import generate_non_shared_trace_functions, generate_shared_trace_functions, calc_values_pre_mu, calc_values_post_mu
from jaxipm.barrier import calc_updated_mu, calc_init_mu


# def calc_least_square_mults(cp, jac_f, jac_c, jac_d, z_L, z_U, v_L, v_U):
#     """
#     Calculate constraint multipliers (y_c, y_d) using least squares.

#     This solves for multipliers that minimize dual infeasibility:
#         min_y ||grad_f - J_c^T y_c - J_d^T y_d + z_L - z_U||^2

#     Used when exiting restoration phase to reinitialize multipliers for the
#     original problem (IPOPT's DefaultIterateInitializer::least_square_mults).

#     Args:
#         cp: CommonProblem
#         jac_f: Dense Jacobian of objective (nx, 1) (just a vector)
#         jac_c: Sparse BCOO Jacobian of equality constraints (nx, nyc)
#         jac_d: Sparse BCOO Jacobian of inequality constraints (nx, nyd)
#         z_L: primal lower bound multipliers (nxL, 1)
#         z_U: primal upper bound multipliers (nxU, 1)
#         v_L: Slack lower bound multipliers (ndL, 1)
#         v_U: Slack upper bound multipliers (ndU, 1)

#     Returns:
#         y_c: Equality constraint multipliers (nyc, 1)
#         y_d: Inequality constraint multipliers (nyd, 1)
#     """
#     rhs_x = (
#         -jac_f
#         + spu.expand_vector(cp.ind_x_L, (cp.nx, 1), z_L)
#         - spu.expand_vector(cp.ind_x_U, (cp.nx, 1), z_U)
#     )
#     rhs_s = spu.expand_vector(cp.ind_d_L, (cp.nyd, 1), v_L) - spu.expand_vector(
#         cp.ind_d_U, (cp.nyd, 1), v_U
#     )
#     rhs_c = jnp.zeros([cp.nyc, 1])
#     rhs_d = jnp.zeros([cp.nyd, 1])
#     RHS = jnp.vstack([rhs_x, rhs_s, rhs_c, rhs_d])

#     # Build LHS: simplified KKT matrix with W=0, Sigma=0
#     # [I    0    J_c   J_d ] [dx ]   [rhs_x]
#     # [0    I    0     -I  ] [ds ] = [rhs_s]
#     # [J_c' 0    0     0   ] [y_c]   [0    ]
#     # [J_d' -I   0     0   ] [y_d]   [0    ]
#     Sigma_x = jnp.zeros([cp.nx])
#     Sigma_s = jnp.zeros([cp.nyd])
#     delta_x = delta_s = 1.0
#     delta_c = delta_d = 0.0

#     LHS_upper_triangular = spu.vstack([
#         spu.hstack([spu.diagflat(Sigma_x) + delta_x * spu.eye(cp.nx), spu.zeros([cp.nx, cp.nyd]), jac_c, jac_d]),
#         spu.hstack([spu.zeros([cp.nyd, cp.nx]), spu.diagflat(Sigma_s) + delta_s * spu.eye(cp.nyd), spu.zeros([cp.nyd, cp.nyc]), -spu.eye(cp.nyd)]),
#         spu.hstack([spu.zeros([cp.nyc, cp.nx]), spu.zeros([cp.nyc, cp.nyd]), -delta_c * spu.eye(cp.nyc), spu.zeros([cp.nyc, cp.nyd])]),
#         spu.hstack([spu.zeros([cp.nyd, cp.nx]), spu.zeros([cp.nyd, cp.nyd]), spu.zeros([cp.nyd, cp.nyc]), -delta_d * spu.eye(cp.nyd)])
#     ]).sum_duplicates(nse=cp.nnz_triu)

#     LHS_upper_triangular = spu.conform_bcoo_to_new_sparsity(cp.coo_indices, LHS_upper_triangular.sort_indices())
#     csr_lhs = jsparse.BCSR.from_bcoo(LHS_upper_triangular)

#     # Solve the system
#     sol, inertia = cp.refactorize_and_linear_solve(RHS.flatten(), csr_lhs.data)
#     sol = sol[:, None]

#     # Extract y_c and y_d from solution
#     y_c = sol[cp.nx + cp.nyd : cp.nx + cp.nyd + cp.nyc]
#     y_d = sol[cp.nx + cp.nyd + cp.nyc : cp.nx + cp.nyd + cp.nyc + cp.nyd]

#     # Magnitude filter - if computed multipliers exceed threshold, reset to zero
#     # This matches IPOPT's behavior in least_square_mults
#     yinitnrm = jnp.maximum(
#         jnp.where(cp.nyc > 0, jnp.max(jnp.abs(y_c)), 0.0),
#         jnp.where(cp.nyd > 0, jnp.max(jnp.abs(y_d)), 0.0)
#     )
#     y_c = jnp.where(yinitnrm > cp.p['constr_mult_init_max'], jnp.zeros_like(y_c), y_c)
#     y_d = jnp.where(yinitnrm > cp.p['constr_mult_init_max'], jnp.zeros_like(y_d), y_d)

#     return y_c, y_d



def initialize_problem_functions(
    f, c, d, nx, nyc, nyd, jac_f_sp, jac_c_sp, jac_d_sp, hes_f_sp, hes_c_sp, hes_d_sp
):
    new_f = lambda x, *f_args: f(x.flatten(), *f_args)
    new_jac_f = lambda x, *f_args: jac_f_sp(x.flatten(), *f_args).T
    new_hess_f = lambda x, *f_args: hes_f_sp(x.flatten(), *f_args)

    # ensure that the new h function provides the correct shaped outputs
    if nyc == 0:
        new_c = lambda x, *c_args: jnp.zeros([0, nyd > 0])
        new_jac_c = lambda x, *c_args: spu.zeros([nx, 0])
        new_hess_c = lambda x, *c_args: spu.zeros([0, nx, nx])
    elif nyc == 1:
        new_c = lambda x, *c_args: c(x.flatten(), *c_args)[None]
        new_jac_c = lambda x, *c_args: jac_c_sp(x.flatten(), *c_args).T
        new_hess_c = lambda x, *c_args: hes_c_sp(x.flatten(), *c_args)
    else:
        new_c = lambda x, *c_args: c(x.flatten(), *c_args)[:, None]
        new_jac_c = lambda x, *c_args: jac_c_sp(x.flatten(), *c_args).T
        new_hess_c = lambda x, *c_args: hes_c_sp(x.flatten(), *c_args)

    # ensure that the new g function provides the correct shaped outputs
    if nyd == 0:
        new_d = lambda x, *d_args: jnp.zeros([0, nyc > 0])
        new_jac_d = lambda x, *d_args: spu.zeros([nx, 0])
        new_hess_d = lambda x, *d_args: spu.zeros([0, nx, nx])
    elif nyd == 1:
        new_d = lambda x, *d_args: d(x.flatten(), *d_args)[None]
        new_jac_d = lambda x, *d_args: jac_d_sp(x.flatten(), *d_args).T
        new_hess_d = lambda x, *d_args: hes_d_sp(x.flatten(), *d_args)
    else:
        new_d = lambda x, *d_args: d(x.flatten(), *d_args)[:, None]
        new_jac_d = lambda x, *d_args: jac_d_sp(x.flatten(), *d_args).T
        new_hess_d = lambda x, *d_args: hes_d_sp(x.flatten(), *d_args)

    return (
        new_f,
        new_jac_f,
        new_hess_f,
        new_c,
        new_jac_c,
        new_hess_c,
        new_d,
        new_jac_d,
        new_hess_d,
    )


# construct common problem performs all preprocessing prior to a batch optimization
def initialize_common_problem(
    _f, _c, _d, x_L, x_U, d_L, d_U, x0, p, function_args=[(), (), ()],
    calc_next_problem=None,
    override_sparse_funcs=None,
):
    """
    f: objective
    c: equality constraints
    d: inequality constraints
    x_l, x_u, d_l, d_u: bounds constraints
    x0: initial primal optimization variable iterate
    p: dict of static parameters defining the optimizer - although we add some entries during post initialization
    """

    # preprocessing ------------------------------------------------------------
    x = x0[:, None]
    nx = x.size
    f_args, c_args, d_args = function_args

    if _f(x, *f_args) is not None:
        # we pass mu, x_ref, dr_x and scaling as the first 4 *args
        f = (
            lambda x, *f_args_: _f(x[:nx], *f_args_[4:]) * f_args_[3]
        )  # we make the fourth arg the scaling
    else:
        f = lambda x, *f_args_: jnp.zeros([0])
    if _c(x, *c_args) is not None:
        # we pass scaling as first *args
        c = (
            lambda x, *c_args_: jnp.hstack(_c(x[:nx], *c_args_[1:])) * c_args_[0]
        )  # we make the zeroth arg the scaling
        nyc = _c(x, *c_args).size
    else:
        nyc = 0
        c = lambda x, *c_args_: jnp.zeros([0])
    if _d(x, *d_args) is not None:
        # we pass scaling as first *args
        d = (
            lambda x, *d_args_: jnp.hstack(_d(x[:nx], *d_args_[1:])) * d_args_[0]
        )  # we make the zeroth arg the scaling
        nyd = _d(x, *d_args).size
    else:
        nyd = 0
        d = lambda x, *d_args_: jnp.zeros([0])

    nxr = nx + 2 * nyc + 2 * nyd

    if x_L is not None:
        x_L = x_L[:, None] - p["bounds_scale"]
        ind_x_L = jnp.where(x_L != -jnp.inf)[0]
        nxL = ind_x_L.size
    else:
        nxL = 0
    if x_U is not None:
        x_U = x_U[:, None] + p["bounds_scale"]
        ind_x_U = jnp.where(x_U != jnp.inf)[0]
        nxU = ind_x_U.size
    else:
        nxU = 0
    if d_L is not None:
        d_L = d_L[:, None] - p["bounds_scale"]
        ind_d_L = jnp.where(d_L != -jnp.inf)[0]
        ndL = ind_d_L.size
    else:
        ndL = 0
    if d_U is not None:
        d_U = d_U[:, None] + p["bounds_scale"]
        ind_d_U = jnp.where(d_U != jnp.inf)[0]
        ndU = ind_d_U.size
    else:
        ndU = 0

    ind_x_LU = jnp.intersect1d(ind_x_L, ind_x_U)
    ind_d_LU = jnp.intersect1d(ind_d_L, ind_d_U)

    # these are the indices of ind_x_L that are damped
    _dampind_x_L = jnp.setdiff1d(ind_x_L, ind_x_U)
    _dampind_x_U = jnp.setdiff1d(ind_x_U, ind_x_L)
    _dampind_d_L = jnp.setdiff1d(ind_d_L, ind_d_U)
    _dampind_d_U = jnp.setdiff1d(ind_d_U, ind_d_L)

    # these are the indices of x that are damped
    dampind_x_L = spu.expand_vector(ind_x_L[_dampind_x_L], (nx,), 1)
    dampind_x_U = spu.expand_vector(ind_x_U[_dampind_x_U], (nx,), 1)
    dampind_d_L = spu.expand_vector(ind_d_L[_dampind_d_L], (nyd,), 1)
    dampind_d_U = spu.expand_vector(ind_d_U[_dampind_d_U], (nyd,), 1)

    # form sparse functions ----------------------------------------------------
    # acquire sparsity patterns from unscaled problems
    jac_f_coo = jnp.array(get_sparsity_pattern(lambda z: _f(z, *f_args), x.flatten(), type="jacobian"), dtype=jnp.int32)
    hes_f_coo = jnp.array(get_sparsity_pattern(lambda z: _f(z, *f_args), x.flatten(), type="hessian"), dtype=jnp.int32)
    jac_c_coo = jnp.array(get_sparsity_pattern(lambda z: _c(z, *c_args), x.flatten(), type="jacobian"), dtype=jnp.int32)
    hes_c_coo = jnp.array(get_sparsity_pattern(lambda z: _c(z, *c_args), x.flatten(), type="hessian"), dtype=jnp.int32)
    jac_d_coo = jnp.array(get_sparsity_pattern(lambda z: _d(z, *d_args), x.flatten(), type="jacobian"), dtype=jnp.int32)
    hes_d_coo = jnp.array(get_sparsity_pattern(lambda z: _d(z, *d_args), x.flatten(), type="hessian"), dtype=jnp.int32)

    # compute unified hess_f sparsity pattern (union of regular pattern + full nxr diagonal for resto)
    # regular hes_f_coo is in [0:nx, 0:nx] block, resto adds full diagonal over [0:nxr, 0:nxr]
    diag_indices_nxr = jnp.stack([jnp.arange(nxr), jnp.arange(nxr)], axis=1).astype(jnp.int32)
    hess_f_coo_indices = jnp.unique(
        jnp.vstack([hes_f_coo, diag_indices_nxr]) if hes_f_coo.size > 0 else diag_indices_nxr,
        axis=0
    )
    # sort by row-major order for sum_duplicates_to_pattern (uses searchsorted)
    sort_idx = jnp.lexsort((hess_f_coo_indices[:, 1], hess_f_coo_indices[:, 0]))
    hess_f_coo_indices = hess_f_coo_indices[sort_idx]
    hess_f_nnz = hess_f_coo_indices.shape[0]

    # reconstruct the raw sparse jacs/hessians using symbolic differentiation
    # Create sample args for tracing (values don't matter, only shapes)
    # f expects: (mu, x_ref, dr_x, scaling, *original_f_args)
    # c expects: (scaling, *original_c_args)
    # d expects: (scaling, *original_d_args)
    sample_f_args = (jnp.ones([1, 1]), jnp.ones(nx), jnp.ones(nx), jnp.array(1.0), *f_args)
    sample_c_args = (jnp.array(1.0), *c_args)
    sample_d_args = (jnp.array(1.0), *d_args)

    # Overrides let callers (e.g. cusadi-backed problems) supply sparse
    # callables instead of deriving them from _f/_c/_d via jax2sympy.
    # Keys: jac_f_sp, jac_c_sp, jac_d_sp, hes_f_sp, hes_c_sp, hes_d_sp, and
    # optionally f, c, d for value callables.
    ovr = override_sparse_funcs or {}

    jac_f_sp = ovr.get("jac_f_sp") or sparse_jacobian_sym(f, x.flatten(), *sample_f_args, coo_pattern=jac_f_coo, out_shape=(1, nx))
    jac_c_sp = ovr.get("jac_c_sp") or sparse_jacobian_sym(c, x.flatten(), *sample_c_args, coo_pattern=jac_c_coo, out_shape=(nyc, nx))
    jac_d_sp = ovr.get("jac_d_sp") or sparse_jacobian_sym(d, x.flatten(), *sample_d_args, coo_pattern=jac_d_coo, out_shape=(nyd, nx))
    hes_f_sp = ovr.get("hes_f_sp") or sparse_hessian_sym(f, x.flatten(), *sample_f_args, coo_pattern=hes_f_coo, out_shape=(1, nx, nx))
    hes_c_sp = ovr.get("hes_c_sp") or sparse_hessian_sym(c, x.flatten(), *sample_c_args, coo_pattern=hes_c_coo, out_shape=(nyc, nx, nx))
    hes_d_sp = ovr.get("hes_d_sp") or sparse_hessian_sym(d, x.flatten(), *sample_d_args, coo_pattern=hes_d_coo, out_shape=(nyd, nx, nx))

    # acquiring callables for regular and resto --------------------------------
    (
        new_f,
        new_jac_f,
        new_hess_f,
        new_c,
        new_jac_c,
        new_hess_c,
        new_d,
        new_jac_d,
        new_hess_d,
    ) = initialize_problem_functions(
        ovr.get("f", f),
        ovr.get("c", c),
        ovr.get("d", d),
        nx,
        nyc,
        nyd,
        jac_f_sp,
        jac_c_sp,
        jac_d_sp,
        hes_f_sp,
        hes_c_sp,
        hes_d_sp,
    )

    # I am going to retrieve the restoration coo for the unreduced, but augmented problem first

    # we can directly get the COOs for the new system here
    _jac_c = jsparse.BCOO([jnp.ones([jac_c_coo.shape[0]]), jac_c_coo], shape=[nx, nyc])
    resto_jac_c_coo = spu.vstack(
        [
            _jac_c,
            -spu.eye(nyc),
            spu.eye(nyc),
            spu.zeros([nyd, nyc]),
            spu.zeros([nyd, nyc]),
        ]
    )

    _jac_d = jsparse.BCOO([jnp.ones([jac_d_coo.shape[0]]), jac_d_coo], shape=[nx, nyd])
    resto_jac_d_coo = spu.vstack(
        [
            _jac_d,
            spu.zeros([nyc, nyd]),
            spu.zeros([nyc, nyd]),
            -spu.eye(nyd),
            spu.eye(nyd),
        ]
    )

    # 2-col COO format [out_idx, x_idx] for scalar objective jacobian
    nxr = nx + nyc * 2 + nyd * 2
    resto_jac_f_x_indices = jnp.arange(nxr, dtype=jnp.int32)[:, None]
    resto_jac_f_coo = jnp.hstack([jnp.zeros((nxr, 1), dtype=jnp.int32), resto_jac_f_x_indices])

    # fast resto conversion functions (avoid recomputing c/d inside traced funcs)
    # These build resto outputs from precomputed regular outputs.
    # c_resto = c + nc - pc, d_resto = d + nd - pd
    # Derivatives: ∂c_resto/∂nc = +I, ∂c_resto/∂pc = -I, etc.
    # Hessians only have entries in original x block (nc/pc/nd/pd are linear).

    # Precompute constant identity blocks for jacobians
    _eye_nyc = spu.eye(nyc)
    _neg_eye_nyc = -spu.eye(nyc)
    _eye_nyd = spu.eye(nyd)
    _neg_eye_nyd = -spu.eye(nyd)
    _zeros_nyc_nyd = spu.zeros([nyc, nyd])
    _zeros_nyd_nyc = spu.zeros([nyd, nyc])
    rho = p["resto_penalty_parameter"]
    _rho_nyc = rho * jnp.ones(nyc)
    _rho_nyd = rho * jnp.ones(nyd)

    def new_f_resto(x, mu, x_ref, dr_x):
        ret1 = p["resto_penalty_parameter"] * x[nx:].sum()
        x_diff = dr_x * (x[:nx] - x_ref)
        eta = p["resto_proximity_weight"] * mu ** p["eta_mu_exponent"]
        ret2 = eta / 2 * x_diff.T @ x_diff
        if p["DEBUG_MODE"]:
            print("WARNING: not evaluating orig obj at resto trial")
        return ret1 + ret2

    def new_c_resto(c, x):
        """Build c_resto from precomputed c: c_resto = c + nc - pc."""
        nc = x[nx : nx + nyc]
        pc = x[nx + nyc : nx + 2 * nyc]
        return c.flatten() + nc - pc

    def new_d_resto(d, x):
        """Build d_resto from precomputed d: d_resto = d + nd - pd."""
        nd = x[nx + 2 * nyc : nx + 2 * nyc + nyd]
        pd = x[nx + 2 * nyc + nyd :]
        return d.flatten() + nd - pd

    def new_jac_f_resto(x, mu, x_ref, dr_x):
        """Compute jac_f_resto: [eta*dr_x²*(x-x_ref), rho, rho, rho, rho]."""
        eta = p["resto_proximity_weight"] * mu.flatten()[0] ** p["eta_mu_exponent"]
        grad_x = eta * dr_x.flatten() ** 2 * (x[:nx].flatten() - x_ref.flatten())
        return jnp.concatenate([grad_x, _rho_nyc, _rho_nyc, _rho_nyd, _rho_nyd])[:, None]

    def new_jac_c_resto(jac_c):
        """Build resto jac_c (nxr, nyc) from regular jac_c (nx, nyc).
        Jacobians stored transposed. Matches resto_jac_c_coo structure.
        """
        return spu.vstack([
            jac_c,           # (nx, nyc)
            _eye_nyc,        # (nyc, nyc) for nc
            _neg_eye_nyc,    # (nyc, nyc) for pc
            _zeros_nyd_nyc,  # (nyd, nyc)
            _zeros_nyd_nyc,  # (nyd, nyc)
        ])

    def new_jac_d_resto(jac_d):
        """Build resto jac_d (nxr, nyd) from regular jac_d (nx, nyd).
        Jacobians stored transposed. Matches resto_jac_d_coo structure.
        """
        return spu.vstack([
            jac_d,           # (nx, nyd)
            _zeros_nyc_nyd,  # (nyc, nyd)
            _zeros_nyc_nyd,  # (nyc, nyd)
            _eye_nyd,        # (nyd, nyd) for nd
            _neg_eye_nyd,    # (nyd, nyd) for pd

        ])

    def new_hess_f_resto(mu, dr_x):
        """Compute hess_f_resto: diag(eta*dr_x**2) embedded in (1, nxr, nxr)."""
        eta = p["resto_proximity_weight"] * mu.flatten()[0] ** p["eta_mu_exponent"]
        diag_vals = eta * dr_x.flatten() ** 2
        diag_mat = spu.diagflat(diag_vals)
        return jsparse.BCOO(
            (diag_mat.data, diag_mat.indices),
            shape=(nxr, nxr)
        )

    def new_hess_c_resto(hess_c):
        """Embed regular hess_c (nyc, nx, nx) into resto shape (nyc, nxr, nxr).
        Since nc/pc are linear in c_resto, hessian only has entries in x block.
        Indices stay the same, only shape changes.
        """
        if hess_c.nse == 0:
            return spu.zeros([nyc, nxr, nxr])
        return jsparse.BCOO(
            (hess_c.data, hess_c.indices),
            shape=(nyc, nxr, nxr)
        )

    def new_hess_d_resto(hess_d):
        """Embed regular hess_d (nyd, nx, nx) into resto shape (nyd, nxr, nxr)."""
        if hess_d.nse == 0:
            return spu.zeros([nyd, nxr, nxr])
        return jsparse.BCOO(
            (hess_d.data, hess_d.indices),
            shape=(nyd, nxr, nxr)
        )
    
    # pre-calculate LHS KKT system structures ----------------------------------
    def calc_lhs_kkt_structure(
        hes_f_coo, hes_d_coo, hes_c_coo, jac_c_coo, jac_d_coo, nx, nyc, nyd
    ):
        hessvp_d_coo = (
            jnp.unique(hes_d_coo[:, 1:], axis=0)
            if hes_d_coo.size > 0
            else jnp.zeros([0, 2]).astype(jnp.int32)
        )
        hessvp_c_coo = (
            jnp.unique(hes_c_coo[:, 1:], axis=0)
            if hes_c_coo.size > 0
            else jnp.zeros([0, 2]).astype(jnp.int32)
        )
        if nyc > 0 and nyd > 0:
            W_coo = jnp.unique(
                jnp.vstack([hes_f_coo, hessvp_c_coo, hessvp_d_coo]), axis=0
            )
        elif nyc > 0:
            W_coo = jnp.unique(jnp.vstack([hes_f_coo, hessvp_c_coo]), axis=0)
        elif nyd > 0:
            W_coo = jnp.unique(jnp.vstack([hes_f_coo, hessvp_d_coo]), axis=0)
        else:
            W_coo = hes_f_coo

        W = spu.triu(jsparse.BCOO((jnp.ones(W_coo.shape[0]), W_coo), shape=[nx, nx]))
        Jc = jsparse.BCOO(
            (jnp.ones(jac_c_coo.shape[0]), jnp.flip(jac_c_coo, axis=1)), shape=[nx, nyc]
        )
        Jd = jsparse.BCOO(
            (jnp.ones(jac_d_coo.shape[0]), jnp.flip(jac_d_coo, axis=1)), shape=[nx, nyd]
        )

        LHS_triu = spu.vstack([
            spu.hstack([spu.add(spu.eye(nx), W), spu.zeros([nx, nyd]), Jc, Jd]),
            spu.hstack([spu.zeros([nyd, nx]), spu.eye(nyd), spu.zeros([nyd, nyc]), spu.eye(nyd)]),
            spu.hstack([spu.zeros([nyc, nx]), spu.zeros([nyc, nyd]), spu.eye(nyc), spu.zeros([nyc, nyd])]),
            spu.hstack([spu.zeros([nyd, nx]), spu.zeros([nyd, nyd]), spu.zeros([nyd, nyc]), spu.eye(nyd)]),
        ]).sum_duplicates().sort_indices()

        csr = jsparse.BCSR.from_bcoo(LHS_triu)

        # we need to easily update the dxs and dcd perturbations during inertia correction later
        diag_indices = spu.find_bcsr_diag_indices(csr).flatten()
        dxs_diag_indices = diag_indices[: nx + nyd]
        dcd_diag_indices = diag_indices[nx + nyd :]
        # split c and d diagonal indices for resto DcR/DdR updates during inertia correction
        dc_diag_indices = dcd_diag_indices[:nyc]
        dd_diag_indices = dcd_diag_indices[nyc:]
        return LHS_triu, csr, dxs_diag_indices, dcd_diag_indices, dc_diag_indices, dd_diag_indices

    LHS_triu, csr, dxs_diag_indices, dcd_diag_indices, dc_diag_indices, dd_diag_indices = calc_lhs_kkt_structure(
        hes_f_coo, hes_c_coo, hes_d_coo, jac_c_coo, jac_d_coo, nx, nyc, nyd
    )

    # LS (least-squares multipliers) solver — dedicated sparsity pattern
    # Structure: [I  0  Jc  Jd]   (no Hessian W, just identity)
    #            [0  I   0  -I]
    #            [0  0   I   0]   (diagonal placeholder for regularization)
    #            [0  0   0   I]
    def calc_lhs_ls_structure(jac_c_coo, jac_d_coo, nx, nyc, nyd):
        Jc_ls = jsparse.BCOO(
            (jnp.ones(jac_c_coo.shape[0]), jnp.flip(jac_c_coo, axis=1)), shape=[nx, nyc]
        )
        Jd_ls = jsparse.BCOO(
            (jnp.ones(jac_d_coo.shape[0]), jnp.flip(jac_d_coo, axis=1)), shape=[nx, nyd]
        )
        LHS_ls_triu = spu.vstack([
            spu.hstack([spu.eye(nx), spu.zeros([nx, nyd]), Jc_ls, Jd_ls]),
            spu.hstack([spu.zeros([nyd, nx]), spu.eye(nyd), spu.zeros([nyd, nyc]), spu.eye(nyd)]),
            spu.hstack([spu.zeros([nyc, nx]), spu.zeros([nyc, nyd]), spu.eye(nyc), spu.zeros([nyc, nyd])]),
            spu.hstack([spu.zeros([nyd, nx]), spu.zeros([nyd, nyd]), spu.zeros([nyd, nyc]), spu.eye(nyd)]),
        ]).sum_duplicates().sort_indices()
        csr_ls = jsparse.BCSR.from_bcoo(LHS_ls_triu)
        return LHS_ls_triu, csr_ls

    LHS_ls_triu, csr_ls = calc_lhs_ls_structure(jac_c_coo, jac_d_coo, nx, nyc, nyd)

    # # Condensed KKT setup (only when kkt_system == "condensed")
    # if p.get("kkt_system", "augmented") == "condensed":
    #     condensed_result = calc_lhs_condensed_kkt_structure(
    #         hes_f_coo, hes_c_coo, hes_d_coo, jac_c_coo, jac_d_coo, nx, nyc, nyd
    #     )
    #     condensed_csr, condensed_dptr, condensed_hptr, condensed_jptr, condensed_nnz, jt_rows_sorted, jt_cols_sorted, jt_colptr, condensed_sort_order = condensed_result

    # # linear solver closure -----------------------------------------------------
    # if p.get("kkt_system", "augmented") == "condensed":
    #     mtype_id = 3  # SPD (positive definite) — Cholesky
    #     solver_csr = condensed_csr
    # else:

    mtype_id = 1  # symmetric indefinite — LDL
    solver_csr = csr
    mview_id = 1  # {0: full, 1: triu, 2: tril}
    device_id = 0
    _linear_solve = CuDSSSolver(solver_csr.indptr, solver_csr.indices, device_id, mtype_id, mview_id)
    # just for debugging
    # jnp.savez("DEBUG_system_structure",
    #     csr_offsets=csr.indptr, csr_columns=csr.indices, device_id=device_id, mtype_id=mtype_id, mview_id=mview_id
    # )
    linear_solve = ft.partial(
        _linear_solve, 
        refactorize_signal=jnp.array([0], dtype=jnp.int32), 
        solve_signal=jnp.array([1], dtype=jnp.int32), 
        ir_nsteps_signal=jnp.array([p["ir_nsteps"]], dtype=jnp.int32)
    )

    # # Build full symmetric CSR structure from upper triangle for sparse residual computation.
    # # Mirror upper triangle COO indices to get lower triangle, deduplicate diagonal.
    # _triu_offsets = solver_csr.indptr
    # _triu_columns = solver_csr.indices
    # _n_kkt = _triu_offsets.shape[0] - 1
    # _triu_rows = jnp.repeat(jnp.arange(_n_kkt, dtype=jnp.int32),
    #                         jnp.diff(_triu_offsets))  # expand offsets to row indices
    # # lower triangle = transpose of strict upper triangle (exclude diagonal)
    # _diag_mask = _triu_rows == _triu_columns
    # _strict_upper_rows = _triu_rows[~_diag_mask]
    # _strict_upper_cols = _triu_columns[~_diag_mask]
    # # full = upper + strict_lower
    # _full_rows = jnp.concatenate([_triu_rows, _strict_upper_cols])
    # _full_cols = jnp.concatenate([_triu_columns, _strict_upper_rows])
    # _full_coo = jsparse.BCOO(
    #     (jnp.ones(_full_rows.shape[0]), jnp.stack([_full_rows, _full_cols], axis=1)),
    #     shape=(_n_kkt, _n_kkt)
    # ).sort_indices()
    # _full_csr = jsparse.BCSR.from_bcoo(_full_coo)
    # # Build scatter index: for each entry in _full_csr, find which triu_csr data
    # # index it maps to. Upper triangle entries map directly; lower triangle entries
    # # map to their transpose. Vectorized via linear index searchsorted.
    # _full_csr_rows = jnp.repeat(jnp.arange(_n_kkt, dtype=jnp.int32),
    #                             jnp.diff(_full_csr.indptr))
    # _full_csr_cols = _full_csr.indices
    # # For each full entry, the triu entry is at (min(r,c), max(r,c))
    # _lookup_rows = jnp.minimum(_full_csr_rows, _full_csr_cols)
    # _lookup_cols = jnp.maximum(_full_csr_rows, _full_csr_cols)
    # # Flat linear index: row * n + col (globally sorted for row-major CSR)
    # _triu_linear = (_triu_rows * _n_kkt + _triu_columns).astype(jnp.int32)
    # _full_lookup_linear = (_lookup_rows * _n_kkt + _lookup_cols).astype(jnp.int32)
    # _scatter_indices = jnp.searchsorted(_triu_linear, _full_lookup_linear).astype(jnp.int32)
    # _full_csr_offsets = _full_csr.indptr
    # _full_csr_columns = _full_csr.indices

    # def _expand_triu_to_full(triu_values):
    #     """Scatter upper-triangle CSR values into full symmetric CSR values."""
    #     return triu_values[_scatter_indices]

    # _no_refac = jnp.array([0], dtype=jnp.int32)
    # _do_solve = jnp.array([1], dtype=jnp.int32)
    # _ir_0 = jnp.array([0], dtype=jnp.int32)
    # _ir_5 = jnp.array([5], dtype=jnp.int32)

    # # def linear_solve(rhs, csr_values):
    # #     """Solve with ir=0 and ir=5, return whichever has smaller sparse residual."""
    # #     x5, inertia = _linear_solve(rhs, csr_values, _no_refac, _do_solve, ir_nsteps_signal=_ir_5)
    # #     x0, _ = _linear_solve(rhs, csr_values, _no_refac, _do_solve, ir_nsteps_signal=_ir_0)

    # #     # sparse residual: ||b - A @ x|| using full symmetric CSR
    # #     full_values = _expand_triu_to_full(csr_values)
    # #     A_full = jsparse.BCSR((full_values, _full_csr_columns, _full_csr_offsets),
    # #                           shape=(_n_kkt, _n_kkt))
    # #     resid_0 = jnp.linalg.norm(rhs - A_full @ x0)
    # #     resid_5 = jnp.linalg.norm(rhs - A_full @ x5)

    # #     x_best = jnp.where(resid_5 < resid_0, x5, x0)
    # #     return x5, inertia # x_best, inertia

    refactorize_and_linear_solve = ft.partial(_linear_solve, refactorize_signal=jnp.array([1], dtype=jnp.int32), solve_signal=jnp.array([1], dtype=jnp.int32))
    refactorize = ft.partial(_linear_solve, refactorize_signal=jnp.array([1], dtype=jnp.int32), solve_signal=jnp.array([0], dtype=jnp.int32))

    # LS dedicated solver (separate cuDSS handle — never contaminates KKT solver)
    _ls_linear_solve = CuDSSSolver(csr_ls.indptr, csr_ls.indices, device_id, mtype_id, mview_id)
    ls_refactorize_and_solve = ft.partial(
        _ls_linear_solve,
        refactorize_signal=jnp.array([1], dtype=jnp.int32),
        solve_signal=jnp.array([1], dtype=jnp.int32),
        ir_nsteps_signal=jnp.array([p["ir_nsteps"]], dtype=jnp.int32)
    )
    # standalone solve (no refactorize) reusing the LS factorization, so cuDSS IR
    # can drive the residual to ~machine precision (see ls_linear_solve usage).
    ls_linear_solve = ft.partial(
        _ls_linear_solve,
        refactorize_signal=jnp.array([0], dtype=jnp.int32),
        solve_signal=jnp.array([1], dtype=jnp.int32),
        ir_nsteps_signal=jnp.array([p["ir_nsteps"]], dtype=jnp.int32)
    )

    # resto bounds -------------------------------------------------------------
    np_L = jnp.zeros([nyc * 2 + nyd * 2, 1])  # exactly zero not pushed to interior
    np_U = jnp.full([nyc * 2 + nyd * 2, 1], jnp.inf)
    ind_np_L = jnp.arange(nyc * 2 + nyd * 2)  # every np is lower bounded
    ind_np_U = jnp.array([], dtype=int)  # no np is upper bounded
    dampind_np_L = jnp.ones(
        nyc * 2 + nyd * 2
    )  # all lower bounds do not have corresponding upper bounds
    dampind_np_U = jnp.zeros(nyc * 2 + nyd * 2)  # no upper bounds exist

    # params post_init ---------------------------------------------------------
    dummy_array = jnp.array([0.0])  # should be instantiated with either f64 of f32
    p["eps"] = float(jnp.finfo(dummy_array).eps)
    p["varphi_max"] = jnp.atleast_2d(jnp.finfo(dummy_array).max)
    obj_scaling_factor = 1.0
    p["mu_min"] = min(
        p["mu_min_default"],
        0.5 * min(p["tol"], float(jnp.abs(p["compl_inf_tol"] * obj_scaling_factor))),
    )
    p["is_square_problem"] = nx == nyc
    p["slack_move"] = p["eps"] ** 0.75  # IpIpoptCalculated_Quantities.cpp
    p["resto_failure_feasibility_threshold"] = 1e2 * p["tol"]  # IpRestoMinC_1Nrm.cpp
    p["tiny_step_tol"] = 10.0 * p["eps"]

    nnz_triu = LHS_triu.data.size
    ls_nnz_triu = LHS_ls_triu.data.size
    nse_Jf = jac_f_coo.shape[0]
    nse_Jc = jac_c_coo.shape[0]
    nse_Jd = jac_d_coo.shape[0]
    nse_rJf = resto_jac_f_coo.shape[0]
    nse_rJc = resto_jac_c_coo.indices.shape[0]
    nse_rJd = resto_jac_d_coo.indices.shape[0]

    # for static shapes for jit later
    # filter to entries within [:nx, :nx] block (don't use slicing - it clips indices!)
    W_mask = (LHS_triu.indices[:, 0] < nx) & (LHS_triu.indices[:, 1] < nx)
    triu_indices = LHS_triu.indices[W_mask]
    off_diag_mask = triu_indices[:, 0] != triu_indices[:, 1]
    W_full_coo_indices = jnp.concatenate([
        triu_indices,
        triu_indices[off_diag_mask][:, ::-1]
    ], axis=0)
    # Sort by row-major order for conform_bcoo_to_new_sparsity (uses searchsorted)
    sort_idx = jnp.lexsort((W_full_coo_indices[:, 1], W_full_coo_indices[:, 0]))
    W_full_coo_indices = W_full_coo_indices[sort_idx]

    W_nnz_triu = triu_indices.shape[0]
    W_nnz = W_full_coo_indices.shape[0]

    # generate regular phase functions - pass the x_only stuff just for structure parity - never used in regular
    funcs = new_f, new_jac_f, new_hess_f, new_c, new_jac_c, new_hess_c, new_d, new_jac_d, new_hess_d
    phase_dims = nx, nxL, x_L, x_U, ind_x_L, ind_x_U, dampind_x_L, dampind_x_U
    general_dims = nx, nxL, nyc, nyd, nxU, ndL, ndU, d_L, d_U, ind_d_L, ind_d_U, dampind_d_L, dampind_d_U, ind_np_L, ind_np_U, ind_x_L, ind_x_U
    # if p.get("kkt_system", "augmented") == "condensed":
    #     condensed_kkt = condensed_dptr, condensed_hptr, condensed_jptr, condensed_nnz, jt_rows_sorted, jt_cols_sorted, jt_colptr, condensed_sort_order
    # else:
    #     condensed_kkt = None
    kkt = LHS_triu.indices, W_full_coo_indices, dxs_diag_indices, dcd_diag_indices, nnz_triu, W_nnz_triu, W_nnz
    nstqf = generate_non_shared_trace_functions(funcs, general_dims, phase_dims, kkt, p, resto=False)

    # generate resto phase functions
    funcs_resto = new_f_resto, new_jac_f_resto, new_hess_f_resto, new_c_resto, new_jac_c_resto, new_hess_c_resto, new_d_resto, new_jac_d_resto, new_hess_d_resto
    phase_dims_resto = nx + nyc * 2 + nyd * 2, nxL + nyc * 2 + nyd * 2, jnp.vstack([x_L, np_L]), jnp.vstack([x_U, np_U]), jnp.hstack([ind_x_L, ind_np_L + nx]), jnp.hstack([ind_x_U, ind_np_U + nx]), jnp.hstack([dampind_x_L, dampind_np_L]), jnp.hstack([dampind_x_U, dampind_np_U])
    nstqfr = generate_non_shared_trace_functions(funcs_resto, general_dims, phase_dims_resto, kkt, p, resto=True)

    # generate shared trace functions
    general_dims_st = nx, nyc, nyd, nxL, nxU, ndL, ndU, ind_x_L, ind_x_U, ind_d_L, ind_d_U, ind_d_L, dampind_d_L, dampind_d_U, ind_np_L, ind_np_U
    stqf = generate_shared_trace_functions(general_dims_st, kkt, p, ls_coo_indices=LHS_ls_triu.indices, ls_nnz_triu=ls_nnz_triu)

    # # build condensed closures
    # def _build_condensed_fns():
    #     def _build_data(W_data, Sigma_x, Sigma_s_pert, Jd_data, delta_x, diag_buffer):
    #         data = jnp.zeros(condensed_nnz)
    #         data = data.at[condensed_hptr[:, 0]].add(W_data[condensed_hptr[:, 1]])
    #         data = data.at[condensed_dptr[:, 0]].add((Sigma_x + delta_x)[condensed_dptr[:, 1]])
    #         data = data.at[condensed_jptr[:, 0]].add(
    #             diag_buffer[condensed_jptr[:, 1]] * Jd_data[condensed_jptr[:, 2]] * Jd_data[condensed_jptr[:, 3]]
    #         )
    #         return data
    #     def _condense(mod_rhs_x, mod_rhs_s, rhs_d_all, Sigma_s_pert, diag_buffer, Jd_data):
    #         buf = diag_buffer.flatten() * (rhs_d_all.flatten() + mod_rhs_s.flatten() / Sigma_s_pert.flatten())
    #         extra = jnp.zeros(nx)
    #         extra = extra.at[jt_rows_sorted].add(Jd_data * buf[jt_cols_sorted])
    #         return mod_rhs_x[:nx].flatten() + extra, buf
    #     def _recover(dx, buf, diag_buffer, Sigma_s_pert, Jd_data, mod_rhs_s):
    #         J_dx = jnp.zeros(nyc + nyd)
    #         J_dx = J_dx.at[jt_cols_sorted].add(Jd_data * dx[jt_rows_sorted])
    #         dy_all = -buf + diag_buffer.flatten() * J_dx
    #         ds_all = (mod_rhs_s.flatten() + dy_all) / Sigma_s_pert.flatten()
    #         return ds_all, dy_all
    #     return KKTCondensedFunctions(build_condensed_data=_build_data, condense_rhs=_condense, recover_step=_recover)

    # throughput mode: default calc_next_problem returns dummy values (never called outside throughput)
    if calc_next_problem is None:
        _initial_user_args = (
            tuple(jnp.asarray(a) for a in function_args[0]),
            tuple(jnp.asarray(a) for a in function_args[1]),
            tuple(jnp.asarray(a) for a in function_args[2]),
        )
        calc_next_problem = lambda key, sol: (jnp.zeros(nx), *_initial_user_args)

    # just to be extremely verbose about everything ----------------------------
    return CommonProblem(
        refactorize=refactorize,
        refactorize_and_linear_solve=refactorize_and_linear_solve,
        linear_solve=linear_solve,
        ls_refactorize_and_solve=ls_refactorize_and_solve,
        ls_linear_solve=ls_linear_solve,
        ls_coo_indices=LHS_ls_triu.indices,
        ls_nnz_triu=ls_nnz_triu,
        coo_indices=LHS_triu.indices,  # for conforming bcoos to original sparsity (a superset)
        W_full_coo_indices=W_full_coo_indices,
        dxs_diag_indices=dxs_diag_indices,
        dcd_diag_indices=dcd_diag_indices,
        dc_diag_indices=dc_diag_indices,
        dd_diag_indices=dd_diag_indices,
        nnz_triu=nnz_triu,
        nse_Jf=nse_Jf,
        nse_Jc=nse_Jc,
        nse_Jd=nse_Jd,
        nse_rJf=nse_rJf,
        nse_rJc=nse_rJc,
        nse_rJd=nse_rJd,
        W_nnz_triu=W_nnz_triu,
        W_nnz=W_nnz,
        hess_f_coo_indices=hess_f_coo_indices,
        hess_f_nnz=hess_f_nnz,
        nx=nx,
        nyc=nyc,
        nyd=nyd,
        nxL=nxL,
        nxU=nxU,
        ndL=ndL,
        ndU=ndU,
        x_L=x_L,
        x_U=x_U,
        d_L=d_L,
        d_U=d_U,
        ind_x_L=ind_x_L,
        ind_x_U=ind_x_U,
        ind_x_LU=ind_x_LU,
        ind_d_L=ind_d_L,
        ind_d_U=ind_d_U,
        ind_d_LU=ind_d_LU,
        dampind_x_L=dampind_x_L,
        dampind_x_U=dampind_x_U,
        dampind_d_L=dampind_d_L,
        dampind_d_U=dampind_d_U,
        np_L=np_L,  # feas resto bounds
        np_U=np_U,  # feas resto bounds
        ind_np_L=ind_np_L,
        ind_np_U=ind_np_U,
        dampind_np_L=dampind_np_L,
        dampind_np_U=dampind_np_U,
        p=p,
        stqf=stqf,
        nstqf=nstqf,
        nstqfr=nstqfr,
        calc_next_problem=calc_next_problem,
    )

def initialize_inertia_correction_state(cp):
    zf = jnp.array(0.0)  # float
    zi = jnp.array(0)  # integer
    return InertiaCorrectionState(
        dxs=zf,
        dcd=zf,
        dxs_old=zf,
        dcd_old=zf,
        jac_degen=zi,
        hess_degen=zi,
        test_status=zi,
        degen_iters=zi,
        inertia=jnp.int32([0, 0]),
        perturbed_data=jnp.zeros([cp.nnz_triu]),
    )

def initialize_line_search_filter_state(theta_min, theta_max, F, cqpr):
    return LineSearchFilterState(
        theta_min,
        theta_max,
        last_rejection_due_to_filter=jnp.array([[0]]),
        count_successive_filter_rejections=jnp.array([[0]]),
        n_filter_resets=jnp.array([[0]]),
        F=F,
        ref_theta=cqpr.theta,
        ref_barr=jnp.array([[0.]]), # dont have the step to calc barr yet # cqpr.barr,
        ref_gBD=jnp.array([[0.]]), # dont have the step to calc gBD yet # cqpr.gBD,
    )

def initialize_line_search_state(it, cqpr, lsfs, cp):
    return LineSearchState(
        acceptable_point=it,
        n_steps=jnp.array([[0]]),
        accept=jnp.array([[0]]),
        it_trial=it,
        trial_step=it, # dummy input # cq.step
        alpha_pr=jnp.array([[0.]]), # dummy input # jnp.atleast_2d(cq.alpha_pr),
        alpha_min=jnp.array([[0.0]]),
        n_filter_resets=jnp.array([[0]]),
        trial_theta=jnp.array([[0.0]]),
        last_obj_val=jnp.array(
            [[-1e50]]
        ),  # see curr_obj_val_ in IpOptErrorConvCheck.cpp
        count_soc=jnp.array([[0]]),
        theta_soc_old=jnp.array([[0.0]]),
        c_soc=cqpr.c,
        dms_soc=cqpr.dms,
        soft_resto_phase_counter=jnp.array([[0]]),
        # for (soft or hard) feas resto step
        satisfies_original_criterion=jnp.array([[0]]),
        count_restorations=jnp.array([[0]]),
        required_infeasibility_reduction=jnp.array([[0.0]])
        if cp.nx == cp.nyc
        else jnp.array([[1e-3]]),  # op.nx == op.nyc means square
        filter=lsfs
    )

def initialize_iterate_flags_state():
    return IterateFlags(
        in_watchdog=jnp.array([[0]]),
        in_soft_resto_phase=jnp.array([[0]]),
        in_restoration=jnp.array([[0]]),
        theta_max_instantiated=jnp.array(
            [[0]]
        ),  # if we enter tiny step first then leave only instantiate theta_max on first regular iterate
        fallback_activated=jnp.array([[0]]),
        tiny_step_last_iter=jnp.array([[0]]),
        skip_first_trial=jnp.array([[0]]),  # set on watchdog timeout to skip first trial point
        soft_resto_entry_requested=jnp.array([[0]]),  # backtrack failed, try soft resto before full resto
        # signals not for execution phase
        free_mu_mode=jnp.array([[1]]),
        tiny_step_flag=jnp.array([[0]]),
        needs_resto_init=jnp.array([[0]]),
        needs_regular_init=jnp.array([[0]]), # 0=normal, 1=init (reset state + LS mults + real KKT solve)
        should_exit_resto=jnp.array([[0]])  # True when restoration converged and returning to regular
    )

def initialize_adaptive_mu_filter_state(cp):
    adfs = jnp.full((cp.p["filter_size"], 3), jnp.inf)
    return adfs.at[:, 0].set(jnp.arange(-cp.p["filter_size"], 0))

def post_initialize_line_search(ls, cqpo, cqpr, cp):
    # Compute alpha_min for the next iteration
    alpha_min = cp.stqf.calc_alpha_min(cqpo.gBD, cqpr.theta, ls.filter.theta_min)
    return eqx.tree_at(
        lambda t: (
            t.alpha_pr,
            t.alpha_min,
            t.trial_step,
            t.trial_theta,
            t.filter.ref_theta,
            t.filter.ref_barr,
            t.filter.ref_gBD
        ), ls, (
            cqpo.alpha_pr,
            alpha_min,
            cqpo.step,
            cqpr.theta,
            cqpr.theta,
            cqpo.barr,
            cqpo.gBD
        )
    )

def initialize_watchdog(mu, it, cqpr, cqpo):
    return WatchdogState(
        shortened_iter=jnp.array([[0]]),
        trial_iter=jnp.array([[0]]),
        alpha_pr_test=cqpo.alpha_pr,
        it=it,
        delta=cqpo.step,
        last_mu=mu,
        theta=cqpr.theta,
        barr=cqpo.barr,
        gBD=cqpo.gBD
    )

# create all necessary functions that are themselves a function of each x0 of the batch
"""compute initial iterate for regular phase"""
# push the primal opt. variables (x, s) to the interior (+ frac) -----------
def push_to_interior(x, s, cp):
    x = x.at[cp.ind_x_L].set(jnp.maximum(x[cp.ind_x_L], cp.x_L[cp.ind_x_L]))
    x = x.at[cp.ind_x_U].set(jnp.minimum(x[cp.ind_x_U], cp.x_U[cp.ind_x_U]))
    s = s.at[cp.ind_d_L].set(jnp.maximum(s[cp.ind_d_L], cp.d_L[cp.ind_d_L]))
    s = s.at[cp.ind_d_U].set(jnp.minimum(s[cp.ind_d_U], cp.d_U[cp.ind_d_U]))

    if (
        cp.p["bound_push"] == 0.0
        and cp.p["bound_frac"] == 0.0
        and cp.p["slack_bound_push"] == 0.0
        and cp.p["slack_bound_frac"] == 0.0
    ):
        return x

    # defaults
    x_shift = 0.0
    s_shift = 0.0

    def calc_shift(
        x_clipped, x_L, x_U, ind_L, ind_U, ind_LU, bound_push, bound_frac
    ):
        full_x_L = jnp.full_like(x_clipped, -jnp.inf).at[ind_L].set(x_L[ind_L])
        full_x_U = jnp.full_like(x_clipped, jnp.inf).at[ind_U].set(x_U[ind_U])
        p = jnp.full_like(x_clipped, jnp.inf)
        p = p.at[ind_L].set(bound_push * jnp.maximum(1.0, jnp.abs(x_L[ind_L])))
        p = p.at[ind_U].set(
            jnp.minimum(
                p[ind_U], bound_push * jnp.maximum(1.0, jnp.abs(x_U[ind_U]))
            )
        )
        frac_margin = bound_frac * (full_x_U[ind_LU] - full_x_L[ind_LU])
        p = p.at[ind_LU].set(jnp.minimum(p[ind_LU], frac_margin))
        shift = jnp.zeros_like(x_clipped)
        shift = shift.at[ind_L].add(
            jnp.maximum(0.0, full_x_L[ind_L] + p[ind_L] - x_clipped[ind_L])
        )
        shift = shift.at[ind_U].add(
            -jnp.maximum(0.0, x_clipped[ind_U] - (full_x_U[ind_U] - p[ind_U]))
        )
        return shift

    if not (cp.p["bound_push"] == 0.0 and cp.p["bound_frac"] == 0.0):
        # calculate just x shift
        x_shift = calc_shift(x, cp.x_L, cp.x_U, cp.ind_x_L, cp.ind_x_U, cp.ind_x_LU, cp.p["bound_push"], cp.p["bound_frac"])

    if not (cp.p["slack_bound_push"] == 0.0 and cp.p["slack_bound_frac"] == 0.0):
        # calculate just s shift, return non-shifted x and shifted s
        s_shift = calc_shift(s, cp.d_L, cp.d_U, cp.ind_d_L, cp.ind_d_U, cp.ind_d_LU, cp.p["slack_bound_push"], cp.p["slack_bound_frac"])

    return x + x_shift, s + s_shift

def compute_scaling(jac_f, jac_c, jac_d, cp):

    # default scaling is 1.0
    df = jnp.ones(jac_f.shape[1])
    dc = jnp.ones(jac_c.shape[1])
    dd = jnp.ones(jac_d.shape[1])
    # objective scaling
    max_grad_f = jnp.linalg.norm(jac_f.data, ord=jnp.inf)
    if cp.p["nlp_scaling_obj_target_gradient"] == 0.0:
        df_trial = cp.p["nlp_scaling_max_gradient"] / max_grad_f
        df_cond = max_grad_f > cp.p["nlp_scaling_max_gradient"]
        df = jnp.where(df_cond, df_trial, df)
    else:
        df_trial = cp.p["nlp_scaling_max_gradient"] / max_grad_f
        df_cond = max_grad_f == 0.0
        df = jnp.where(df_cond, df, df_trial)

    # equality constraint scaling
    jac_c_t = jac_c.T.sum_duplicates(nse=cp.nse_Jc).sort_indices()
    dc = spu.inf_norm_per_row(jac_c_t, cp.nyc)
    arow_max = jnp.linalg.norm(dc, ord=jnp.inf)
    if cp.p["nlp_scaling_constr_target_gradient"] <= 0.0 and cp.p["nlp_scaling_min_value"] > 0.0:
        dc_trial = jnp.maximum(
            jnp.minimum(cp.p["nlp_scaling_max_gradient"] / dc, 1.0),
            cp.p["nlp_scaling_min_value"],
        )
        dc_cond = arow_max > cp.p["nlp_scaling_max_gradient"]
        dc = jnp.where(dc_cond, dc_trial, jnp.ones_like(dc))
    elif cp.p["nlp_scaling_constr_target_gradient"] <= 0.0:
        dc_trial = jnp.minimum(cp.p["nlp_scaling_max_gradient"] / dc, 1.0)
        dc_cond = arow_max > cp.p["nlp_scaling_max_gradient"]
        dc = jnp.where(dc_cond, dc_trial, jnp.ones_like(dc))
    else:
        dc = jnp.full_like(dc, cp.p["nlp_scaling_constr_target_gradient"] / arow_max)

    # inequality constraint scaling
    jac_d_t = jac_d.T.sum_duplicates(nse=cp.nse_Jd).sort_indices()
    dd = spu.inf_norm_per_row(jac_d_t, cp.nyd)
    arow_max = jnp.linalg.norm(dd, ord=jnp.inf)
    if (
        cp.p["nlp_scaling_constr_target_gradient"] <= 0.0
        and cp.p["nlp_scaling_min_value"] > 0.0
    ):
        dd_trial = jnp.maximum(
            jnp.minimum(cp.p["nlp_scaling_max_gradient"] / dd, 1.0),
            cp.p["nlp_scaling_min_value"],
        )
        dd_cond = arow_max > cp.p["nlp_scaling_max_gradient"]
        dd = jnp.where(dd_cond, dd_trial, jnp.ones_like(dd))
    elif cp.p["nlp_scaling_constr_target_gradient"] <= 0.0:
        dd_trial = jnp.minimum(cp.p["nlp_scaling_max_gradient"] / dd, 1.0)
        dd_cond = arow_max > cp.p["nlp_scaling_max_gradient"]
        dd = jnp.where(dd_cond, dd_trial, jnp.ones_like(dd))
    else:
        dd = jnp.full_like(dd, cp.p["nlp_scaling_constr_target_gradient"] / arow_max)

    return df, dc, dd

def initialize_problem_regular(cp, x0, args=[(), (), ()]):
    """perform entire initialization for regular phase"""
    
    # TODO: IPOPT performs initial scaling calculation before push_to_interior - potentially evalutating
    # functions outside of the feasible region - FIX?

    # SCALING ------------------------------------------------------------------
    fcd_default_scale = [1.0, 1.0, 1.0]  # default scaling is added to *args for each function
    args = [(scale, *_args) for scale, _args in zip(fcd_default_scale, args)]
    f_args, c_args, d_args = args
    dummy_resto_f_args = (jnp.zeros([1, 1]), jnp.zeros_like(x0), jnp.zeros_like(x0))
    f_args = (*dummy_resto_f_args, *f_args)

    jac_f = cp.nstqf.calc_jac_f(x0, *f_args)  # (nx, 1)
    jac_c = cp.nstqf.calc_jac_c(x0, *c_args)  # (nx, nyc)
    jac_d = cp.nstqf.calc_jac_d(x0, *d_args)  # (nx, nyd)
    # jacobians0 = (jac_f, jac_c, jac_d)

    df, dc, dd = compute_scaling(jac_f, jac_c, jac_d, cp)

    # we apply scaling through function args for convenience (alternative is complex closures for feas resto)
    # we add dummy feas resto args to maintain the same structure as resto phase
    args = (
        (*f_args[:3], df, *f_args[4:]),
        (dc, *c_args[1:]),
        (dd, *d_args[1:]),
    )

    # initialize iterate regular -----------------------------------------------
    _, _, d_args = args
    x = x0[:, None]
    s = jnp.max(
        jnp.hstack([cp.nstqf.calc_d(x, *d_args), cp.p["Ktol"] * jnp.ones([cp.nyd, 1])]),
        axis=1,
        keepdims=True,
    )
    x, s = push_to_interior(x, s, cp)

    # calculate the remaining multipliers initializations and form iterate -----
    z_L = jnp.ones_like(cp.ind_x_L, dtype=float)[:, None]
    z_U = jnp.ones_like(cp.ind_x_U, dtype=float)[:, None]
    v_L = jnp.ones_like(cp.ind_d_L, dtype=float)[:, None]
    v_U = jnp.ones_like(cp.ind_d_U, dtype=float)[:, None]

    # initialize multipliers
    # these are the SCALED functions now, and x has been pushed to interior
    f_args, c_args, d_args = args
    jac_f = cp.nstqf.calc_jac_f(x, *f_args).todense()
    jac_c = cp.nstqf.calc_jac_c(x, *c_args)
    jac_d = cp.nstqf.calc_jac_d(x, *d_args)
    jacobians = (jac_f, jac_c, jac_d)

    # padded regular jacobians
    pad_rows = cp.nyc * 2 + cp.nyd * 2 
    # reg_jac_f = spu.vstack([jac_f, spu.ones_with_nse([pad_rows, 1], cp.nse_rJf - cp.nse_Jf)]) # pad the regular jacobians
    reg_jac_f = jnp.vstack([jac_f, jnp.zeros([cp.nyc*2+cp.nyd*2,1])])
    reg_jac_c = spu.vstack([jac_c, spu.ones_with_nse([pad_rows, cp.nyc], cp.nse_rJc - cp.nse_Jc)]) # pad the regular jacobians
    reg_jac_d = spu.vstack([jac_d, spu.ones_with_nse([pad_rows, cp.nyd], cp.nse_rJd - cp.nse_Jd)]) # pad the regular jacobians

    hess_f = cp.nstqf.calc_hess_f(x[:cp.nx], *f_args)
    hess_c = cp.nstqf.calc_hess_c(x[:cp.nx], *c_args)
    hess_d = cp.nstqf.calc_hess_d(x[:cp.nx], *d_args)

    rhs_x = (
        -jac_f
        + spu.expand_vector(cp.ind_x_L, (cp.nx, 1), z_L)
        - spu.expand_vector(cp.ind_x_U, (cp.nx, 1), z_U)
    )
    rhs_s = spu.expand_vector(cp.ind_d_L, (cp.nyd, 1), v_L) - spu.expand_vector(
        cp.ind_d_U, (cp.nyd, 1), v_U
    )
    rhs_c = jnp.zeros([cp.nyc, 1])
    rhs_d = jnp.zeros([cp.nyd, 1])
    RHS = jnp.vstack([rhs_x, rhs_s, rhs_c, rhs_d])

    # W = 0. # making this EXPLICIT
    Sigma_x = jnp.zeros([cp.nx])
    Sigma_s = jnp.zeros([cp.nyd])
    delta_x = delta_s = 1.0
    delta_c = delta_d = 0.0

    # we stick to upper triangular representations for memory efficiency
    LHS_upper_triangular = spu.vstack([
        spu.hstack([spu.diagflat(Sigma_x) + delta_x * spu.eye(cp.nx),   spu.zeros([cp.nx, cp.nyd]),                         jac_c,                             jac_d                          ]),
        spu.hstack([spu.zeros([cp.nyd, cp.nx]),                         spu.diagflat(Sigma_s) + delta_s * spu.eye(cp.nyd),  spu.zeros([cp.nyd, cp.nyc]),    -spu.eye(cp.nyd)            ]),
        spu.hstack([spu.zeros([cp.nyc, cp.nx]),                         spu.zeros([cp.nyc, cp.nyd]),                        -delta_c * spu.eye(cp.nyc),     spu.zeros([cp.nyc, cp.nyd]) ]),
        spu.hstack([spu.zeros([cp.nyd, cp.nx]),                         spu.zeros([cp.nyd, cp.nyd]),                        spu.zeros([cp.nyd, cp.nyc]),    -delta_d * spu.eye(cp.nyd)  ])
    ]).sum_duplicates()

    if cp.p["DEBUG_MODE"]:
        jax.debug.print("LHS before conform: {data}", data=LHS_upper_triangular.data)
    LHS_upper_triangular = spu.conform_bcoo_to_new_sparsity(cp.coo_indices, LHS_upper_triangular.sort_indices())
    if cp.p["DEBUG_MODE"]:
        jax.debug.print("LHS after conform: {data}", data=LHS_upper_triangular.data)
    csr_lhs = jsparse.BCSR.from_bcoo(LHS_upper_triangular)

    sol, inertia = cp.refactorize_and_linear_solve(RHS.flatten(), csr_lhs.data)
    sol = sol[:, None]

    if cp.p["VALIDATION_MODE"] is True:
        m = csr_lhs.todense()
        LHS = m + m.T - jnp.diag(jnp.diag(m))
        residual = jnp.linalg.norm(LHS @ sol - RHS)
        print(f"error residual calc_mults: {residual}")

    y_c = sol[cp.nx + cp.nyd : cp.nx + cp.nyd + cp.nyc]
    y_d = sol[cp.nx + cp.nyd + cp.nyc : cp.nx + cp.nyd + cp.nyc + cp.nyd]

    # magnitude filter - if computed multipliers exceed threshold, reset to zero
    if cp.nyc > 0 and cp.nyd > 0:
        yinitnrm = jnp.maximum(jnp.max(jnp.abs(y_c)), jnp.max(jnp.abs(y_d)))
    elif cp.nyc > 0:
        yinitnrm = jnp.max(jnp.abs(y_c))
    elif cp.nyd > 0:
        yinitnrm = jnp.max(jnp.abs(y_d))
    else:
        yinitnrm = jnp.array(0.0)
    y_c = jnp.where(yinitnrm > cp.p['constr_mult_init_max'], jnp.zeros_like(y_c), y_c)
    y_d = jnp.where(yinitnrm > cp.p['constr_mult_init_max'], jnp.zeros_like(y_d), y_d)

    # pad
    x = jnp.vstack([x, jnp.zeros([cp.nyc * 2 + cp.nyd * 2, 1])])
    z_L = jnp.vstack([z_L, jnp.zeros([cp.nyc * 2 + cp.nyd * 2, 1])])

    # form iterate
    it = Iterate(x, s, y_c, y_d, z_L, z_U, v_L, v_U)    

    # Calculate Quantities -----------------------------------------------------

    # initialize flags ahead of calculate quantities as we need free_mu_mode flag for it
    fl = initialize_iterate_flags_state()
    f = jnp.atleast_2d(cp.nstqf.calc_f(x, *f_args))
    c = cp.nstqf.calc_c(x, *c_args)
    d = cp.nstqf.calc_d(x, *d_args)
    fun_outs = (f, c, d)
    
    dms = d - it.s
    theta = cp.nstqf.calc_theta(c, dms)
    theta_max = jnp.atleast_2d(
        1e4 * jnp.maximum(1, theta)
    )  # we cannot ONLY calculate this here as if tiny step is instantiated immediately
    
    # then we cannot instantiate theta min/max until we have escaped tiny step or converged
    theta_min = jnp.atleast_2d(1e-4 * jnp.maximum(1, theta))
    slacks = cp.nstqf.calc_slacks(it)
    initial_average_complementarity = cp.nstqf.calc_avrg_compl(it, slacks)
    mu_max = cp.p["mu_max_fact"] * initial_average_complementarity
    mu_max = jnp.clip(mu_max, min=None, max=cp.p["mu_max_upper"])
    mu = initial_average_complementarity
    F = jnp.vstack([jnp.hstack([theta_max, cp.p["varphi_max"]])] * cp.p["filter_size"])
    tau = jnp.maximum(cp.p["tau_min"], 1 - mu)
    F_iter = jnp.arange(-cp.p["filter_size"], 0)[:, None]
    F = jnp.hstack([F_iter, F])

    # duplicate calculations but only run once
    grad_lag_x = cp.nstqf.calc_grad_lag_x(it, jacobians)
    grad_lag_s = cp.stqf.calc_grad_lag_s(it)

    # Clip to at least 1.0 to avoid division by zero when initial point is feasible
    # (matches IPOPT's Max(1.0, ...) pattern in AdaptiveMuUpdate::lower_mu_safeguard)
    init_dual_inf = jnp.maximum(1.0, cp.nstqf.calc_dual_inf(grad_lag_x, grad_lag_s, ord=1))
    init_primal_inf = jnp.maximum(1.0, cp.nstqf.calc_primal_inf_L1(c, dms))

    # initial inertia correction state
    ic = initialize_inertia_correction_state(cp)

    # unify calculation of jacobians/hessians outside of calc_values_pre_mu
    jac_f, jac_c, jac_d = reg_jac_f, reg_jac_c, reg_jac_d

    # common operations
    jacobians = (jac_f, jac_c, jac_d)
    hessians = (hess_f, hess_c, hess_d)

    # quantities - padded and passed to calc_values_pre_mu
    pad = jnp.zeros([cp.nyc * 2 + cp.nyd * 2, 1])  # how much to pad by
    reg_grad_lag_x = jnp.vstack([cp.nstqf.calc_grad_lag_x(it, jacobians), pad])
    reg_slacks = cp.nstqf.calc_slacks(it)
    reg_slacks = (jnp.vstack([reg_slacks[0], jnp.zeros_like(pad)]), *reg_slacks[1:])

    quantities = (
        reg_grad_lag_x,
        reg_slacks,
        cp.nstqf.calc_theta(c, d - it.s),
        cp.nstqf.calc_avrg_compl(it, reg_slacks),
        cp.stqf.calc_grad_lag_s(it) # grad lag s
    )

    # calculated quantities before mu calc - plus initial inertia correction
    cqpr, ic = calc_values_pre_mu(it, ic, cp, fl, fun_outs, jacobians, hessians, quantities, iter_count=jnp.array([[0]]))

    # line search filter state
    lsfs = initialize_line_search_filter_state(theta_min, theta_max, F, cqpr)

    # need a line search to run calc_updated_mu - even if we don't have all fields calculated yet
    ls = initialize_line_search_state(it, cqpr, lsfs, cp)

    # adaptive mu filter state
    adfs = initialize_adaptive_mu_filter_state(cp)

    # init based on average complementarity
    mu = calc_init_mu(it, cp.nstqf)

    # update barrier parameter seperately here
    df = args[0][3]
    iter_count = jnp.array([[0]])
    mu, tau, ls, adfs, terminate, new_free_mu_mode = calc_updated_mu(it, cqpr, mu, tau, mu_max, init_dual_inf, init_primal_inf, df, fl, adfs, ls, cp, iter_count)
    fl = eqx.tree_at(lambda t: t.free_mu_mode, fl, new_free_mu_mode)

    # calculate step and other post mu quantities
    cqpo = calc_values_post_mu(it, mu, tau, cqpr, ic, cp, fl)

    # update the line search trial step, alpha_pr, and alpha_min now that we have them
    ls = post_initialize_line_search(ls, cqpo, cqpr, cp)

    # setup watchdog structure
    wd = initialize_watchdog(mu, it, cqpr, cqpo)

    # Save slots initialized as copies of active structures
    # These will be overwritten when entering restoration mode
    saved_fl = fl
    saved_wd = wd
    saved_ls = ls
    saved_ic = ic
    saved_adfs = adfs
    saved_mu = mu
    saved_tau = tau
    saved_mu_max = mu_max

    # Compute initial slacks for saved state
    slacks = cp.nstqf.calc_slacks(it)

    # wrap this all in the optimization state object that we pass around
    state = OptimizationState(
        it=it,
        cqpr=cqpr,
        cqpo=cqpo,
        fl=fl,
        wd=wd,
        ls=ls,
        ic=ic,
        adfs=adfs,
        mu=mu,
        tau=tau,
        mu_max=mu_max,
        init_dual_inf=init_dual_inf,
        init_primal_inf=init_primal_inf,
        saved_fl=saved_fl,
        saved_wd=saved_wd,
        saved_ls=saved_ls,
        saved_ic=saved_ic,
        saved_adfs=saved_adfs,
        saved_mu=saved_mu,
        saved_tau=saved_tau,
        saved_mu_max=saved_mu_max,
        saved_init_dual_inf=jnp.zeros_like(init_dual_inf),
        saved_init_primal_inf=jnp.zeros_like(init_primal_inf),
        saved_orig_inf_pr=jnp.zeros_like(init_primal_inf),  # Set when entering restoration
        saved_z_L=it.z_L[:cp.nxL],  # Placeholder, set when entering restoration
        saved_z_U=it.z_U,
        saved_v_L=it.v_L,
        saved_v_U=it.v_U,
        saved_slacks=slacks,  # (sxL, sxU, sdL, sdU)
        resto_tol=jnp.atleast_2d(cp.p["tol"]),  # Initialized to tol, tightened during restoration
        iter_count=jnp.array([[0]]),
        args=args,
    )

    return state

def initialize_skeleton_state(cp, x0, args=[(), (), ()]):
    """
    A function that only defines the structure of the optimization state, all values
    are set to zeros except needs_regular_init which is set to 1. This is passed to 
    problems before they are started, and the warm start functionality fills in the 
    values. This unifies the warm start and cold start paths.
    """

    # SCALING AND ARGS ---------------------------------------------------------
    fcd_default_scale = [jnp.ones([1]), jnp.ones([cp.nyc]), jnp.ones(cp.nyd)]  # default scaling is added to *args for each function
    args = [(scale, *_args) for scale, _args in zip(fcd_default_scale, args)]
    f_args, c_args, d_args = args
    dummy_resto_f_args = (jnp.zeros([1, 1]), jnp.zeros_like(x0), jnp.zeros_like(x0))
    f_args = (*dummy_resto_f_args, *f_args)
    args = (f_args, c_args, d_args)

    # ITERATE ------------------------------------------------------------------
    # x = jnp.zeros([cp.nx + cp.nyd*2 + cp.nyc*2, 1]) # resto phase padding added
    x = jnp.vstack([x0[:, None], jnp.zeros([cp.nyd*2 + cp.nyc*2, 1])])
    s = jnp.zeros([cp.nyd, 1])
    y_c = jnp.zeros([cp.nyc, 1])
    y_d = jnp.zeros([cp.nyd, 1])
    z_L = jnp.zeros([cp.nxL + cp.nyd*2 + cp.nyc*2, 1])
    z_U = jnp.zeros([cp.nxU, 1])
    v_L = jnp.zeros([cp.ndL, 1])
    v_U = jnp.zeros([cp.ndU, 1])
    it = Iterate(x, s, y_c, y_d, z_L, z_U, v_L, v_U)

    # CQPR ---------------------------------------------------------------------
    f = jnp.zeros([1, 1])
    c = jnp.zeros([cp.nyc, 1])
    d = jnp.zeros([cp.nyd, 1])
    jac_f = jnp.zeros((cp.nx + cp.nyd*2 + cp.nyc*2, 1))  # (nx, 1)
    # jac_c = jnp.zeros((cp.nx, cp.nyc))  # (nx, nyc)
    # jac_d = jnp.zeros((cp.nx, cp.nyd))  # (nx, nyd)
    dms = d - it.s
    y_nrminf = jnp.zeros(1).squeeze()
    grad_lag_x=jnp.zeros([cp.nx + cp.nyc*2 + cp.nyd*2, 1]) # padded for resto
    grad_lag_s = jnp.zeros([cp.nyd, 1])
    grad_lag_x_nrm2 = jnp.zeros(1).squeeze()
    grad_lag_s_nrm2 = jnp.zeros(1).squeeze()
    c_nrm2 = jnp.zeros(1).squeeze()
    slacks=(
        jnp.zeros([cp.nxL + cp.nyc*2 + cp.nyd*2, 1]),
        jnp.zeros([cp.nxU, 1]),
        jnp.zeros([cp.ndL, 1]),
        jnp.zeros([cp.ndU, 1])
    ) # (sxL, sxU, sdL, sdU)
    avrg_compl=jnp.zeros([1, 1])
    theta=jnp.zeros([1, 1])
    d_minus_s_nrm2 = jnp.zeros(1).squeeze()
    nlp_error = jnp.zeros(1).squeeze()
    nlp_constr_viol = jnp.zeros([1,1])
    barrier_constr_viol = jnp.zeros([1,1])
    primal_inf = jnp.zeros([1, 1])
    full_step_size = cp.nx + cp.nyd*2 + cp.nyc*2 + cp.nyd + cp.nyc + cp.nyd + cp.nxL + cp.nyd*2 + cp.nyc*2 + cp.nxU + cp.ndL + cp.ndU
    step_aff_full = jnp.zeros([full_step_size, 1])
    step_cen_full = jnp.zeros([full_step_size, 1])
    Sigma_nc_inv = jnp.zeros([cp.nyc, 1])
    Sigma_pc_inv = jnp.zeros([cp.nyc, 1])
    Sigma_nd_inv = jnp.zeros([cp.nyd, 1])
    Sigma_pd_inv = jnp.zeros([cp.nyd, 1])
    y_c_init = jnp.zeros([cp.nyc, 1])
    y_d_init = jnp.zeros([cp.nyd, 1])

    cqpr = CalculatedQuantitiesPreMu(
        f=f,
        c=c,
        d=d,
        jac_f=jac_f,  # (nx, 1)
        dms=dms,
        y_nrminf=y_nrminf,
        grad_lag_x=grad_lag_x,
        grad_lag_s=grad_lag_s,
        slacks=slacks,
        avrg_compl=avrg_compl,
        theta=theta,
        grad_lag_x_nrm2=grad_lag_x_nrm2,
        grad_lag_s_nrm2=grad_lag_s_nrm2,
        c_nrm2=c_nrm2,
        d_minus_s_nrm2=d_minus_s_nrm2,
        nlp_error=nlp_error,
        nlp_constr_viol=nlp_constr_viol,
        barrier_constr_viol=barrier_constr_viol,
        primal_inf=primal_inf,
        step_aff_full=step_aff_full,
        step_cen_full=step_cen_full,
        Sigma_nc_inv=Sigma_nc_inv,
        Sigma_pc_inv=Sigma_pc_inv,
        Sigma_nd_inv=Sigma_nd_inv,
        Sigma_pd_inv=Sigma_pd_inv,
        y_c_init=y_c_init,
        y_d_init=y_d_init,
    )

    # CQPO ---------------------------------------------------------------------
    rhs_aug = jnp.zeros([cp.nx + cp.nyd + cp.nyc + cp.nyd, 1])
    step_aug = jnp.zeros([cp.nx + cp.nyd + cp.nyc + cp.nyd, 1])
    rhs = Iterate(
        x=jnp.zeros([cp.nx + cp.nyc*2 + cp.nyd*2, 1]),
        s=jnp.zeros([cp.nyd, 1]),
        y_c=jnp.zeros([cp.nyc, 1]),
        y_d=jnp.zeros([cp.nyd, 1]),
        z_L=jnp.zeros([cp.nxL + cp.nyc*2 + cp.nyd*2, 1]),
        z_U=jnp.zeros([cp.nxU, 1]),
        v_L=jnp.zeros([cp.ndL, 1]),
        v_U=jnp.zeros([cp.ndU, 1]),
    )
    step = rhs
    alpha_pr = jnp.zeros([1, 1])
    barr = jnp.zeros([1, 1])
    gBD = jnp.zeros([1, 1])
    slack_derivatives = (
        jnp.zeros([cp.nxL + cp.nyc*2 + cp.nyd*2, 1]), 
        jnp.zeros([cp.nxU, 1]), 
        jnp.zeros([cp.ndL, 1]), 
        jnp.zeros([cp.ndU, 1])
    )

    cqpo = CalculatedQuantitiesPostMu(
        rhs_aug=rhs_aug,
        step_aug=step_aug, 
        rhs=rhs, 
        step=step, 
        alpha_pr=alpha_pr, 
        barr=barr, 
        gBD=gBD, 
        slack_derivatives=slack_derivatives
    )

    # FLAGS --------------------------------------------------------------------
    in_watchdog = jnp.array([[0]])
    in_soft_resto_phase = jnp.array([[0]])
    in_restoration = jnp.array([[0]])
    theta_max_instantiated = jnp.array([[0]])
    fallback_activated = jnp.array([[0]])
    tiny_step_last_iter = jnp.array([[0]])
    skip_first_trial = jnp.array([[0]])
    soft_resto_entry_requested = jnp.array([[0]])
    free_mu_mode = jnp.array([[0]])
    tiny_step_flag = jnp.array([[0]])
    needs_resto_init = jnp.array([[0]])
    needs_regular_init = jnp.array([[1]])
    should_exit_resto = jnp.array([[0]])
    fl = IterateFlags(
        in_watchdog=in_watchdog, 
        in_soft_resto_phase=in_soft_resto_phase, 
        in_restoration=in_restoration, 
        theta_max_instantiated=theta_max_instantiated, 
        fallback_activated=fallback_activated, 
        tiny_step_last_iter=tiny_step_last_iter, 
        skip_first_trial=skip_first_trial, 
        soft_resto_entry_requested=soft_resto_entry_requested, 
        free_mu_mode=free_mu_mode, 
        tiny_step_flag=tiny_step_flag, 
        needs_resto_init=needs_resto_init, 
        needs_regular_init=needs_regular_init, 
        should_exit_resto=should_exit_resto
    )

    # WATCHDOG -----------------------------------------------------------------
    shortened_iter = jnp.array([[0]])
    trial_iter = jnp.array([[0]])
    alpha_pr_test = jnp.array([[0.0]])
    it = it
    delta = it
    last_mu = jnp.array([[0.0]])
    theta = jnp.array([[0.0]])
    barr = jnp.array([[0.0]])
    gBD = jnp.array([[0.0]])

    wd = WatchdogState(
        shortened_iter=shortened_iter, 
        trial_iter=trial_iter, 
        alpha_pr_test=alpha_pr_test, 
        it=it, 
        delta=delta, 
        last_mu=last_mu, 
        theta=theta, 
        barr=barr, 
        gBD=gBD
    )

    # LINE SEARCH --------------------------------------------------------------
    theta_min = jnp.array([[0.0]])
    theta_max = jnp.array([[0.0]])
    last_rejection_due_to_filter = jnp.array([[0]])
    count_successive_filter_rejections = jnp.array([[0]])
    n_filter_resets = jnp.array([[0]])
    F = jnp.zeros([cp.p["filter_size"], 3])  # (idx, theta, barr) triplets
    ref_theta = jnp.array([[0.0]])
    ref_barr = jnp.array([[0.0]])
    ref_gBD = jnp.array([[0.0]])
    filter = LineSearchFilterState(
        theta_min=theta_min, 
        theta_max=theta_max, 
        last_rejection_due_to_filter=last_rejection_due_to_filter, 
        count_successive_filter_rejections=count_successive_filter_rejections, 
        n_filter_resets=n_filter_resets, 
        F=F, 
        ref_theta=ref_theta, 
        ref_barr=ref_barr, 
        ref_gBD=ref_gBD
    )
    acceptable_point = it
    n_steps = jnp.array([[0]])
    accept = jnp.array([[0]])
    trial_step = it
    it_trial = it
    alpha_pr = jnp.array([[0.0]])
    alpha_min = jnp.array([[0.0]])
    n_filter_resets = jnp.array([[0]])
    trial_theta = jnp.array([[0.0]])
    last_obj_val = jnp.array([[0.0]])
    count_soc = jnp.array([[0]])
    theta_soc_old = jnp.array([[0.0]])
    c_soc = jnp.zeros([cp.nyc, 1])
    dms_soc = jnp.zeros([cp.nyd, 1])
    soft_resto_phase_counter = jnp.array([[0]])
    satisfies_original_criterion = jnp.array([[0]])
    count_restorations = jnp.array([[0]])
    required_infeasibility_reduction = jnp.array([[0.0]])

    ls = LineSearchState(
        acceptable_point=acceptable_point, 
        n_steps=n_steps, 
        accept=accept, 
        trial_step=trial_step, 
        it_trial=it_trial, 
        alpha_pr=alpha_pr, 
        alpha_min=alpha_min, 
        n_filter_resets=n_filter_resets, 
        trial_theta=trial_theta, 
        last_obj_val=last_obj_val, 
        count_soc=count_soc, 
        theta_soc_old=theta_soc_old, 
        c_soc=c_soc, 
        dms_soc=dms_soc, 
        soft_resto_phase_counter=soft_resto_phase_counter, 
        satisfies_original_criterion=satisfies_original_criterion, 
        count_restorations=count_restorations, 
        required_infeasibility_reduction=required_infeasibility_reduction, 
        filter=filter
    )

    # INERTIA CORRECTION -------------------------------------------------------
    dxs = jnp.array(0.) # weak f64/f32
    dcd = jnp.array(0.) # weak f64/f32
    dxs_old = jnp.array(0.) # weak f64/f32
    dcd_old = jnp.array(0.) # weak f64/f32
    jac_degen = jnp.array(0) # weak i64/i32
    hess_degen = jnp.array(0) # weak i64/i32
    test_status = jnp.array(0) # weak i64/i32
    degen_iters = jnp.array(0) # weak i64/i32
    inertia = jnp.zeros([2], dtype=jnp.int32) # always hard i32
    perturbed_data = jnp.zeros([cp.nnz_triu])
    ic = InertiaCorrectionState(
        dxs=dxs, 
        dcd=dcd, 
        dxs_old=dxs_old, 
        dcd_old=dcd_old, 
        jac_degen=jac_degen, 
        hess_degen=hess_degen, 
        test_status=test_status, 
        degen_iters=degen_iters, 
        inertia=inertia, 
        perturbed_data=perturbed_data
    )

    # FINAL ASSEMBLY -----------------------------------------------------------
    adfs = jnp.zeros_like(ls.filter.F)
    mu = jnp.array([[0.0]])
    tau = jnp.array([[0.0]])
    mu_max = jnp.array([[0.0]])
    init_dual_inf = jnp.array([[0.0]])
    init_primal_inf = jnp.array([[0.0]])
    orig_inf_pr = jnp.array([[0.0]])
    saved_z_L = jnp.zeros([cp.nxL, 1]) # without padding - only save regular phase components
    saved_z_U = jnp.zeros([cp.nxU, 1])
    saved_v_L = jnp.zeros([cp.ndL, 1])
    saved_v_U = jnp.zeros([cp.ndU, 1])
    saved_slacks = (
        jnp.zeros([cp.nxL, 1]), # no padding again - only save regular phase components
        jnp.zeros([cp.nxU, 1]), 
        jnp.zeros([cp.ndL, 1]), 
        jnp.zeros([cp.ndU, 1])
    )
    resto_tol = jnp.array([[0.0]])
    iter_count = jnp.array([[0]])

    skeleton_state = OptimizationState(
        it=it,
        cqpr=cqpr,
        cqpo=cqpo,
        fl=fl,
        wd=wd,
        ls=ls,
        ic=ic,
        adfs=adfs,
        mu=mu,
        tau=tau,
        mu_max=mu_max,
        init_dual_inf=init_dual_inf,
        init_primal_inf=init_primal_inf,
        saved_fl=fl,
        saved_wd=wd,
        saved_ls=ls,
        saved_ic=ic,
        saved_adfs=adfs,
        saved_mu=mu,
        saved_tau=tau,
        saved_mu_max=mu_max,
        saved_init_dual_inf=init_dual_inf,
        saved_init_primal_inf=init_primal_inf,
        saved_orig_inf_pr=orig_inf_pr,
        saved_z_L=saved_z_L,
        saved_z_U=saved_z_U,
        saved_v_L=saved_v_L,
        saved_v_U=saved_v_U,
        saved_slacks=saved_slacks,
        resto_tol=resto_tol,
        iter_count=iter_count,
        args=args,
    )

    return skeleton_state


# Initialize fresh restoration state
def initialize_resto_args(state, cp):
    # form the new function args based on mu
    x_ref = state.it.x[: cp.nx]
    dr_x = jnp.minimum(1, 1 / jnp.abs(x_ref))
    # Preserve the original scaling factor from regular args at position [3]
    # Restoration's new_f_resto doesn't use this, but we need it when returning to regular mode.
    # Also carry through any user f_args (positions 4+) so the resto-regular tuple
    # match in filter_select, the resto cost function ignores them.
    r_f_args = (
        state.mu,
        x_ref.flatten(),
        dr_x.flatten(),
        state.args[0][3],  # Preserve original scaling for return to regular
        *state.args[0][4:],  # Carry user f_args (e.g. per-slot xr in track_avoid)
    )  # NOTE THIS MU IS NOT USED
    r_c_args = state.args[
        1
    ]  # scaling parameters must be passed to part of func
    r_d_args = state.args[
        2
    ]  # scaling parameters must be passed to part of func
    args = [r_f_args, r_c_args, r_d_args]
    # determine scaling of f, c, d if initial values at x0 are extreme
    # get the initial mu for the resto problem
    rmu = jnp.max(jnp.hstack([
        state.mu,
        jnp.linalg.norm(state.cqpr.c, ord=jnp.inf, keepdims=True),
        jnp.linalg.norm(state.cqpr.dms, ord=jnp.inf, keepdims=True),
    ]))
    args = (
        (jnp.array([[rmu]]), *args[0][1:]),
        *args[1:],
    )  # update mu
    return args, rmu

def initialize_iterate_resto(it, cp, rmu, cq):
    """compute initial iterate for feasibility restoration phase"""
    # compute initial values for nc, pc
    def solve_quadratic(a, b):
        ret = a
        ret *= a
        ret += b
        ret = jnp.sqrt(ret)
        ret += a
        return ret

    rho = cp.p["resto_penalty_parameter"]

    a = rmu / (2.0 * rho)
    a += -0.5 * cq.c
    b = cq.c
    b *= rmu / (2.0 * rho)
    nc = solve_quadratic(a, b)
    pc = cq.c
    pc += 1.0 * nc

    # compute initial values for nd, pd
    a = rmu / (2.0 * rho)
    a += -0.5 * cq.dms
    b = cq.dms
    b *= rmu / (2.0 * rho)
    nd = solve_quadratic(a, b)
    pd = cq.dms
    pd += 1.0 * nd

    # form the new x
    x = jnp.vstack([it.x[: cp.nx], nc, pc, nd, pd])
    # leave slacks unchanged from trial it
    s = it.s

    # bound multipliers - we cap them at rho
    z_L = jnp.vstack([jnp.minimum(rho, it.z_L[:cp.nxL]), rmu / nc, rmu / pc, rmu / nd, rmu / pd])
    z_U = jnp.minimum(rho, it.z_U)
    v_L = jnp.minimum(rho, it.v_L)
    v_U = jnp.minimum(rho, it.v_U)
    y_c, y_d = jnp.zeros([cp.nyc, 1]), jnp.zeros([cp.nyd, 1])
    it_new = Iterate(x, s, y_c, y_d, z_L, z_U, v_L, v_U)
    return it_new

def exit_resto(state, cp):
    """Exit restoration mode, restoring regular algorithm state from save slots."""

    # Technically resetting the padding to zero is unecessary, but we do it for debugging clarity
    pad = jnp.zeros([cp.nyc * 2 + cp.nyd * 2, 1])
    it = eqx.tree_at(
        lambda t: (t.x, t.z_L),
        state.it,
        (jnp.vstack([state.it.x[:cp.nx], pad]), jnp.vstack([state.it.z_L[:cp.nxL], pad]))
    )

    # Restore active structures from save slots - simple full reload
    fl = state.saved_fl
    wd = state.saved_wd
    ls = state.saved_ls
    ic = state.saved_ic
    adfs = state.saved_adfs
    mu = state.saved_mu
    mu_max = state.saved_mu_max

    return eqx.tree_at(
        lambda t: (t.it, t.fl, t.wd, t.ls, t.ic, t.adfs, t.mu, t.mu_max),
        state,
        (it, fl, wd, ls, ic, adfs, mu, mu_max)
    )

if __name__ == "__main__":

    # do the setup for validation ----------------------------------------------
    from jaxipm.utils.validation_boilerplate import *
    from jaxipm.search import execute_search, post_process
    # jax.config.update("jax_log_compiles", False)
    # os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    max_iter = 20

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
    #         "maxiter": max_iter,
    #         # "mu_strategy": "monotone", # forces filter entries to happen pretty much
    #         "start_with_resto": "no",  # lets us test feas resto initialization as well
    #     },
    # )

    # common problem across entire batch
    cp = initialize_common_problem(
        f, c, d, x_L, x_U, d_L, d_U, x0, p, [f_args, c_args, d_args]
    )

    skeleton_state = initialize_skeleton_state(cp, x0, args=[f_args, c_args, d_args])

    _execute_search = eqx.filter_jit(lambda s: execute_search(s, cp))
    _post_process = eqx.filter_jit(lambda s, sn: post_process(s, sn, cp))


    state, _ = _post_process(skeleton_state, skeleton_state)
    # state = _execute_search(state)
    # state, _ = _post_process(state, state)

    # _state = initialize_problem_regular(cp, x0, args=[f_args, c_args, d_args])

    def validate_directions(i, cq, it, resto=False):
        if resto is False: # trim off resto parts as needed
            it = eqx.tree_at(lambda t: t.x, it, it.x[: cp.nx, :])  
            it = eqx.tree_at(lambda t: t.z_L, it, it.z_L[: cp.nxL, :])  
            cq = eqx.tree_at(lambda t: t.step.x, cq, cq.step.x[: cp.nx, :])  
            cq = eqx.tree_at(lambda t: t.step.z_L, cq, cq.step.z_L[: cp.nxL, :]) 
        components = ["x", "s", "y_c", "y_d", "z_L", "z_U", "v_L", "v_U"]
        ref_step = Iterate(
            *[load_vector(f"{save_dir}/delta_{i}_{c}.txt") for c in components]
        )
        # ref_step = eqx.tree_at(lambda t: (t.x, t.s), ref_step, (cq.step.x.flatten(), cq.step.s.flatten()))
        diff_step = (jax.tree.map(jnp.ravel, cq.step) ** ω - ref_step**ω).ω
        diff_step_norm = jnp.linalg.norm(jnp.hstack(jax.tree.leaves(diff_step)))
        print(f"diff step norm: {diff_step_norm}")
        # load up the current iterates
        ref_it = Iterate(*[load_vector(f"{save_dir}/iterate_{i}_{c}.txt") for c in components])

        diff_it = (jax.tree.map(jnp.ravel, it) ** ω - ref_it**ω).ω
        diff_it_norm = jnp.linalg.norm(jnp.hstack(jax.tree.leaves(diff_it)))

        # double check RHS - its bang on...
        # np.loadtxt("ipopt_logs/augRhs_x_iter_0_ref_0_count_5.txt") + state.cqpo.rhs.x[:cp.nx].flatten()

        print(f"diff it norm: {diff_it_norm}")

    # from jaxipm.classify import classify
    from jaxipm.tests.load_ipopt_state_regular import load_state, align_reference_state, analyze_diff
    jax.config.update("jax_log_compiles", False)
    from jaxipm.utils.validation_utils import load_vector

    # verify directions for the regular initialized iterate (E2E test)
    validate_directions(i=0, cq=state.cqpo, it=state.it, resto=False)

    # terminate = jnp.array([[0]])
    # while terminate == 0:
    #     state_next = _execute_search(state)
    #     state_next, terminate = _post_process(state, state_next)
    #     state = state_next
    #     print(f"iter: {state.iter_count.squeeze()}")
    #     print(f"dxs: {state.ic.dxs}")

    for i in range(max_iter-1):
        state_next = _execute_search(state)
        state_next, terminate = _post_process(state, state_next)
        state = state_next
        print(f"iter: {state.iter_count.squeeze()}")
        print(f"dxs: {state.ic.dxs}")

        if i >= 5 and i < 34:
            resto = True
            pass
        else:
            resto = False

        if i >= 35:
            validate_directions(i=state.iter_count.squeeze()-1, cq=state.cqpo, it=state.it, resto=resto)
        else:
            validate_directions(i=state.iter_count.squeeze(), cq=state.cqpo, it=state.it, resto=resto)

        pass

    # animate the result from IPOPT and the result from our implementation
    from problems.redundant.quadcopter.plotting import Animator
    x, u = aux[0](state.it.x[:cp.nx])
    r_ = [0.5, 0.5, 0.5, 0.75]
    xc_ = [1., 1., 3., 3.]
    yc_ = [1., 3., 1., 3.]
    cylinder_definitions = [xc_,yc_,r_]
    quad_params = aux[2]
    # x = x.at[:, 2].mul(-1.)
    x = x.at[:, :2].mul(-1.)
    animator = Animator(quad_params, x, np.arange(0, 3, 0.1), x, cylinder_definitions=cylinder_definitions, drawCylinder=True, followQuad=False)
    animator.animate()

    for i in range(20):
        print(f"iter: {i}")
        ref_state = load_state(i, cp)
        ref_state = align_reference_state(state, ref_state, resto=False, soc=False, cp=cp)
        analyze_diff(state, ref_state)# , path_filter="it")

        # jnp.savez(
        #     f"DEBUG_system_{i}", 
        #     csr_data=state.ic.perturbed_data, 
        #     rhs=state.cqpo.rhs_aug, 
        #     our_sol=state.cqpo.step_aug
        #     # sol=ref_state.cqpo.step_aug # havent gotten the load in properly yet - but its ok we can use dense linalg to check correctness (and residuals)
        # )

        state_next = _execute_search(state)
        state_next, terminate = _post_process(state, state_next)
        state = state_next
        validate_directions(i=state.iter_count.squeeze(), cq=state.cqpo, it=state.it, resto=False)
        # pass

    # test restoration initialization
    result = minimize_ipopt(
        fun=obj,
        x0=x0,
        jac=obj_grad,
        hess=obj_hess,
        constraints=constraints,
        bounds=bounds,
        options={
            "disp": 5,
            "maxiter": 5,
            # "mu_strategy": "monotone", # forces filter entries to happen pretty much
            "start_with_resto": "yes",  # lets us test feas resto initialization as well
        },
    )

    # just for validating the resto initialization - but must run IPOPT with start_with_resto == True
    # update iter count and move onto resto
    opt_state = eqx.tree_at(
        lambda t: (
            t.iter_count,
            t.fl.needs_resto_init
        ), 
        state, 
        (
            jnp.array([[1]]),
            jnp.array([[1]])
        )
    )
    resto_opt_state, _ = _post_process(opt_state, state)

    # verify directions for the restoration initialized iterate (E2E test)
    validate_directions(i=1, cq=resto_opt_state.cqpo, it=resto_opt_state.it, resto=True)

    pass

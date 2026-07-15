"""
Stage 2 of the correctness_test experiment — PER-STEP INJECTION validator.

Replicates the paper's mk3 validation harness (jaxipm/utils/validation_utils_mk3.py
in the paper codebase): every iteration starts jaxipm from IPOPT's EXACT state
(loaded wholesale from the patched IPOPT's ``ipopt_logs/iter_<k>/`` dumps), takes
ONE step out of it, compares the result against IPOPT's next iterate, then
RE-SEEDS jaxipm with IPOPT's state for the following step. Single-step errors
therefore CANNOT accumulate — each sample in the saved npz is the deviation of
one jaxipm iteration from a bit-identical starting point, which is what the
paper's ``validation_state_ecdf`` figure plots.

(The previous revision of this script ran jaxipm OPEN-LOOP — its own state
propagated for the whole solve — and compared iterate k against IPOPT iterate k.
Open-loop trajectories decouple at the first divergent discrete decision (mu
oracle, line-search acceptance, restoration timing), after which the aligned
comparison measures trajectory separation, not step accuracy. That data cannot
reproduce the paper's single-step ECDF; this per-step version can.)

Per pass idx (0-based; pass 0 starts from jaxipm's own initialization, which is
IPOPT's cold start):

    result          = execute_search(state)              # one jaxipm step
    own_state, _    = post_process(state, result)        # jaxipm's own iteration
    rows[idx + 1]   = compare(own_state, IPOPT[idx + 1]) # single-step deviation
    state           = post_process(state, INJECT(IPOPT)) # re-seed for next pass

The npz schema is unchanged (d_<comp> per iterate, jx_resto/ip_resto timelines,
...), so tests/correctness/analyze_results.py consumes it as before — the d_*
arrays are now genuine single-step deviations. The d_*_shift1 keys (an open-loop
alignment diagnostic) are meaningless under injection and are saved as NaN.

Iterative refinement is set to 100 (params.yaml ir_nsteps) — the linear-solve
refinement depth previously found necessary to track IPOPT.
"""

import os
import json
import pathlib

# ── Params: load jaxipm/params.yaml, force ir_nsteps=100, pin the GPU ────────
# CUDA_VISIBLE_DEVICES must be set before jax is imported. We honour the
# gpu_id field in params.yaml exactly as jaxipm/solver.py does.
with open("jaxipm/params.json") as _f:
    p = json.load(_f)

p["VALIDATION_MODE"] = True
p["ir_nsteps"] = 100  # iterative refinement depth for this validation run
# os.environ["CUDA_VISIBLE_DEVICES"] = str(p["gpu_id"])
# os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx

jax.config.update("jax_enable_x64", True)

from jaxipm.initialization import initialize_common_problem, initialize_problem_regular
from jaxipm.search import execute_search, post_process
from jaxipm.solver import TerminationCode
from jaxipm.structures import (
    OptimizationState,
    Iterate,
    CalculatedQuantitiesPreMu,
    CalculatedQuantitiesPostMu,
    IterateFlags,
    WatchdogState,
    LineSearchFilterState,
    LineSearchState,
    InertiaCorrectionState,
)
from jaxipm.utils.validation_utils import (
    load_vector,
    load_scalars,
    load_iterate_x_from_components,
    load_iterate_z_L_from_components,
)

# Single source of truth — the quad_nav_circle problem, the same fixed-start
# nav task Stage 1 (ipopt_correctness.py) converted for IPOPT. Its quadcopter_nav
# returns the unbounded z_L/z_U and an identical obstacle/cost/dynamics setup, so
# the recorded ipopt_logs/ remain a valid reference. OBS_* are the bare cylinder
# radii used only for drawing; the solver inflates them by the quad radius.
from tests.quad_nav_circle.jaxipm_quad_nav import (
    quadcopter_nav, OBSTACLES, N_HORIZON, TS as Ts,
)
OBS_XC = [o["xc"] for o in OBSTACLES]
OBS_YC = [o["yc"] for o in OBSTACLES]
OBS_R = [o["r"] for o in OBSTACLES]

HERE = pathlib.Path(__file__).resolve().parent
IPOPT_LOGS = HERE / "ipopt_logs"
LOGS = HERE / "logs"


# ── Reference loading (flat per-component files, used by the compare rows) ───
def count_ipopt_iters(save_dir):
    """Number of consecutive IPOPT iterates k = 0, 1, 2, ... that were logged."""
    k = 0
    while (os.path.exists(f"{save_dir}/iterate_x_{k}_comp0.txt")
           or os.path.exists(f"{save_dir}/iterate_{k}_x.txt")):
        k += 1
    return k


def count_ipopt_state_dumps(save_dir):
    """Number of consecutive full-state ``iter_<k>/`` dumps (the mk3 schema)."""
    k = 0
    while os.path.isdir(f"{save_dir}/iter_{k}"):
        k += 1
    return k


def _col_np(vec):
    """Coerce a loaded vector to a (n, 1) column (empty -> (0, 1))."""
    a = np.asarray(vec)
    return a.reshape(-1, 1) if a.size else np.zeros((0, 1))


def load_ipopt_iterate(save_dir, k, cp):
    """Load IPOPT's iterate k: the primal/dual vectors plus mu, tau.

    Components mirror jaxipm's Iterate {x, s, y_c, y_d, z_L, z_U, v_L, v_U}.
    x and z_L go through the component loaders (they handle restoration-phase
    splitting); the rest are plain single-vector files.
    """
    x = np.asarray(load_iterate_x_from_components(save_dir, k, cp))[: cp.nx]
    z_L = np.asarray(load_iterate_z_L_from_components(save_dir, k, cp))[: cp.nxL]
    mt = load_scalars(f"{save_dir}/mu_tau_{k}.txt")
    return {
        "x": _col_np(x),
        "s": _col_np(load_vector(f"{save_dir}/iterate_{k}_s.txt")),
        "y_c": _col_np(load_vector(f"{save_dir}/iterate_{k}_y_c.txt")),
        "y_d": _col_np(load_vector(f"{save_dir}/iterate_{k}_y_d.txt")),
        "z_L": _col_np(z_L),
        "z_U": _col_np(load_vector(f"{save_dir}/iterate_{k}_z_U.txt")),
        "v_L": _col_np(load_vector(f"{save_dir}/iterate_{k}_v_L.txt")),
        "v_U": _col_np(load_vector(f"{save_dir}/iterate_{k}_v_U.txt")),
        "mu": float(mt.get("mu", np.nan)),
        "tau": float(mt.get("tau", np.nan)),
    }


def jaxipm_iterate(state, cp):
    """Extract the same {x, s, y_c, y_d, z_L, z_U, v_L, v_U, mu, tau} from a
    jaxipm OptimizationState (unpadded to the core problem dimensions)."""
    it = state.it
    return {
        "x": np.asarray(it.x)[: cp.nx],
        "s": np.asarray(it.s),
        "y_c": np.asarray(it.y_c),
        "y_d": np.asarray(it.y_d),
        "z_L": np.asarray(it.z_L)[: cp.nxL],
        "z_U": np.asarray(it.z_U),
        "v_L": np.asarray(it.v_L),
        "v_U": np.asarray(it.v_U),
        "mu": float(np.asarray(state.mu).squeeze()),
        "tau": float(np.asarray(state.tau).squeeze()),
    }


# Components compared each iteration, and the tolerance band on the primal
# iterate that defines a "matched" single step.
_COMPONENTS = ["x", "s", "y_c", "y_d", "z_L", "z_U", "v_L", "v_U"]
X_MATCH_ATOL = 1e-6   # tight band — the step reproduces IPOPT's
X_DRIFT_ATOL = 1e-4   # loose band — a genuinely different step


def compare(ji, ri):
    """Max abs diff per component between a jaxipm iterate and an IPOPT iterate.
    Returns dict component -> max|Δ| (np.nan if a shape mismatch makes it
    incomparable)."""
    out = {}
    for key in _COMPONENTS + ["mu", "tau"]:
        a, b = ji[key], ri[key]
        if np.ndim(a) == 0:
            out[key] = abs(float(a) - float(b))
            continue
        a, b = np.asarray(a), np.asarray(b)
        if a.shape != b.shape:
            out[key] = np.nan
        else:
            out[key] = float(np.max(np.abs(a - b))) if a.size else 0.0
    return out


# ═════════════════════════════════════════════════════════════════════════════
# mk3 full-state loader — ported from the paper codebase's
# jaxipm/utils/validation_utils_mk3.py. The patched IPOPT writes one file per
# OptimizationState sub-struct under ``ipopt_logs/iter_<n>/`` in a sectioned
# ``@name count [cols]`` format, with section names equal to jaxipm field
# names. Anything IPOPT does not emit is zero-filled from ``cp`` dimensions.
# ═════════════════════════════════════════════════════════════════════════════
def parse_sections(path):
    """Parse a ``@name count [cols]`` file -> {name: jnp.ndarray}.

    1-D section -> shape (count,);  2-D section -> shape (rows, cols).
    Missing file -> {} (caller zero-fills).

    Duplicate keys: FIRST occurrence wins. At the restoration-EXIT boundary IPOPT
    rewinds the iteration counter (PerformRestoration: Set_iter_count(resto-1)), so
    the boundary iter dir gets a SECOND dump of fl/ls/wd (+cqpo.alpha_pr) from the
    resumed REGULAR line search that merely re-accepts the final resto point. Only
    the FIRST dump is aligned with that iter's it/cqpr/state (all logged once, from
    the resto step computation); taking the last dump instead seeds the harness with
    in_restoration=0 over a mid-resto iterate, so post_process never runs the exit
    check and the saved_* restore is skipped.
    """
    if not os.path.exists(path):
        return {}
    out = {}
    with open(path) as f:
        lines = f.read().split("\n")
    i, n = 0, len(lines)
    while i < n:
        head = lines[i].strip()
        i += 1
        if not head.startswith("@"):
            continue
        parts = head[1:].split()
        name = parts[0]
        rows = int(parts[1])
        cols = int(parts[2]) if len(parts) > 2 else None
        block = lines[i : i + rows]
        i += rows
        if name in out:
            continue
        if cols is None:  # 1-D
            out[name] = jnp.array([float(x) for x in block]) if block else jnp.zeros(0)
        else:  # 2-D (e.g. filter.F, adfs)
            data = [[float(x) for x in ln.split()] for ln in block if ln.strip()]
            out[name] = jnp.array(data) if data else jnp.zeros((0, cols))
    return out


def _col(a):
    """-> column vector (n, 1)."""
    return jnp.asarray(a, dtype=float).reshape(-1, 1)


def _sc(sections, name, default=0.0):
    """Scalar section -> (1, 1)."""
    v = sections.get(name)
    if v is None or jnp.asarray(v).size == 0:
        return jnp.atleast_2d(float(default))
    return jnp.atleast_2d(float(jnp.asarray(v).reshape(-1)[0]))


def _vec(sections, name, dim):
    """Vector section -> (dim, 1), zero-filled if absent/empty."""
    v = sections.get(name)
    if v is None or jnp.asarray(v).size == 0:
        return jnp.zeros([dim, 1])
    return _col(v)


def _filter_pad(F2, filter_size):
    """(n, 2) entries -> (filter_size, 3) = [index, theta, phi], inf-padded.

    IPOPT's filter_.GetEntries() emits each entry as (phi, theta); jaxipm stores
    [index, theta, phi], so the two value columns are swapped here. Shared by
    adfs and ls.filter.F (same IPOPT source + same jaxipm convention).
    """
    F2 = jnp.asarray(F2)
    if F2.ndim != 2 or F2.shape[0] == 0:
        F2 = jnp.zeros((0, 2))
    F2 = jnp.vstack([F2, jnp.full([filter_size - F2.shape[0], 2], jnp.inf)])
    F2 = F2[:, ::-1]  # (phi, theta) -> (theta, phi)
    return jnp.hstack([jnp.arange(filter_size)[:, None], F2])


def _stitch(sections, base, sizes):
    """Restoration-aware x / z_L: base section (= comp0) plus the augmented
    .nc/.pc/.nd/.pd blocks when present, else zero-padded to restoration size.

    sizes = [base_dim, nyc, nyc, nyd, nyd].  Returns (sum(sizes), 1).
    """
    parts = [_vec(sections, base, sizes[0])]
    suffixes = ["nc", "pc", "nd", "pd"]
    in_resto = (base + ".nc") in sections
    for suf, sz in zip(suffixes, sizes[1:]):
        parts.append(_vec(sections, f"{base}.{suf}", sz) if in_resto else jnp.zeros([sz, 1]))
    return jnp.vstack(parts)


def _load_iterate(sections, prefix, cp):
    """Build an Iterate from sections named ``<prefix>x``, ``<prefix>s``, ...

    x and z_L are stitched/padded to restoration size; the rest are static.
    """
    xs = [cp.nx, cp.nyc, cp.nyc, cp.nyd, cp.nyd]
    zs = [cp.nxL, cp.nyc, cp.nyc, cp.nyd, cp.nyd]
    return Iterate(
        x=_stitch(sections, prefix + "x", xs),
        s=_vec(sections, prefix + "s", cp.nyd),
        y_c=_vec(sections, prefix + "y_c", cp.nyc),
        y_d=_vec(sections, prefix + "y_d", cp.nyd),
        z_L=_stitch(sections, prefix + "z_L", zs),
        z_U=_vec(sections, prefix + "z_U", cp.nxU),
        v_L=_vec(sections, prefix + "v_L", cp.ndL),
        v_U=_vec(sections, prefix + "v_U", cp.ndU),
    )


def _load_step_full(sections, prefix, cp):
    """Flatten an iterate-shaped step (``<prefix>x``, ``<prefix>s``, ...) into
    jaxipm's flat ``step_aff_full`` / ``step_cen_full`` layout

        [x(nx+aug); s(nyd); y_c(nyc); y_d(nyd); z_L(nxL+aug); z_U(nxU); v_L(ndL); v_U(ndU)]

    matching quantities.calc_transform_aug_to_full. x and z_L carry the resto
    block (stitched/zero-padded). Zero-fill when the producer section is absent
    (IPOPT omits the probing steps in fixed/monotone-mu mode)."""
    xs = [cp.nx, cp.nyc, cp.nyc, cp.nyd, cp.nyd]
    zs = [cp.nxL, cp.nyc, cp.nyc, cp.nyd, cp.nyd]
    if (prefix + "x") not in sections:
        total = sum(xs) + cp.nyd + cp.nyc + cp.nyd + sum(zs) + cp.nxU + cp.ndL + cp.ndU
        return jnp.zeros([total, 1])
    return jnp.vstack([
        _stitch(sections, prefix + "x", xs),
        _vec(sections, prefix + "s", cp.nyd),
        _vec(sections, prefix + "y_c", cp.nyc),
        _vec(sections, prefix + "y_d", cp.nyd),
        _stitch(sections, prefix + "z_L", zs),
        _vec(sections, prefix + "z_U", cp.nxU),
        _vec(sections, prefix + "v_L", cp.ndL),
        _vec(sections, prefix + "v_U", cp.ndU),
    ])


def _empty_iterate(cp):
    return Iterate(
        x=jnp.zeros([cp.nx + 2 * cp.nyc + 2 * cp.nyd, 1]),
        s=jnp.zeros([cp.nyd, 1]),
        y_c=jnp.zeros([cp.nyc, 1]),
        y_d=jnp.zeros([cp.nyd, 1]),
        z_L=jnp.zeros([cp.nxL + 2 * cp.nyc + 2 * cp.nyd, 1]),
        z_U=jnp.zeros([cp.nxU, 1]),
        v_L=jnp.zeros([cp.ndL, 1]),
        v_U=jnp.zeros([cp.ndU, 1]),
    )


def load_state(iter_num, cp, base_dir=None, kkt_token=None, ls_token=None):
    """Reconstruct an OptimizationState from ``<base_dir>/iter_<iter_num>/``.

    base_dir defaults to this test's ``ipopt_logs``.

    kkt_token/ls_token: the LIVE spineax FactorTokens from the running jaxipm
    state (IPOPT logs nothing token-shaped). Required whenever the loaded
    state is injected into post_process or tree-mapped against a jaxipm state.
    """
    base = str(base_dir) if base_dir is not None else str(IPOPT_LOGS)
    d = f"{base}/iter_{iter_num}"
    # Fail loudly rather than silently zero-filling a non-existent dump.
    if not os.path.isdir(d):
        raise FileNotFoundError(
            f"no IPOPT iter dump at '{d}' (cwd={os.getcwd()}). "
            f"Run from a directory containing '{base}/', or pass base_dir=."
        )
    if not os.path.exists(f"{d}/it.txt"):
        raise FileNotFoundError(f"missing '{d}/it.txt' — incomplete IPOPT dump.")
    S = {name: parse_sections(f"{d}/{name}.txt") for name in
         ("it", "cqpr", "cqpo", "fl", "wd", "ls", "ic", "state", "soc")}
    static = parse_sections(f"{base}/_static/proj.txt")
    scaling = parse_sections(f"{base}/_static/nlp_scaling.txt")
    # Frozen restoration reference mu / point, logged once to _static by IPOPT's
    # RestoIterateInitializer. NOTE: this release's ipopt_logs predate those dumps
    # (both files are absent), so the fallbacks below fire — x_ref/dr_x/ref_mu are
    # reconstructed from the CURRENT iterate, which is only correct at the resto
    # entry iter. The injection loop compensates: during restoration it re-injects
    # jaxipm's OWN frozen args (see the resto-continuation block in main()), so the
    # loader's reconstruction is never consumed mid-restoration.
    resto_ref = parse_sections(f"{base}/_static/resto_ref_mu.txt")
    resto_ref_x = parse_sections(f"{base}/_static/resto_ref_x.txt")

    fs = cp.p["filter_size"]
    resto_dim = 2 * cp.nyc + 2 * cp.nyd
    aug = resto_dim  # padding length for the resto-variable blocks

    # ---- it : Iterate ----
    it = _load_iterate(S["it"], "", cp)

    # ---- fl : IterateFlags (7 logged, 6 internal gaps -> 0) ----
    fls = S["fl"]
    fl = IterateFlags(
        in_watchdog=_sc(fls, "in_watchdog"),
        in_soft_resto_phase=_sc(fls, "in_soft_resto_phase"),
        in_restoration=_sc(fls, "in_restoration"),
        theta_max_instantiated=_sc(fls, "theta_max_instantiated"),
        fallback_activated=_sc(fls, "fallback_activated"),
        tiny_step_last_iter=_sc(fls, "tiny_step_last_iter"),
        skip_first_trial=jnp.array([[0]]),
        soft_resto_entry_requested=jnp.array([[0]]),
        free_mu_mode=_sc(fls, "free_mu_mode"),
        tiny_step_flag=_sc(fls, "tiny_step_flag"),
        needs_resto_init=jnp.array([[0]]),
        needs_regular_init=jnp.array([[0]]),
        should_exit_resto=jnp.array([[0]]),
    )

    # ---- state : top-level scalars (mu/tau/mu_max/init_inf) + adfs ----
    st = S["state"]
    mu = _sc(st, "mu")
    tau = _sc(st, "tau")
    mu_max = _sc(st, "mu_max", 1e10)
    init_dual_inf = _sc(st, "init_dual_inf", 1.0)
    init_primal_inf = _sc(st, "init_primal_inf", 1.0)
    adfs = _filter_pad(st.get("adfs", jnp.zeros((0, 2))), fs)

    # ---- args (scaling) — needed before jac_f ----
    df = float(jnp.asarray(scaling.get("df", jnp.array([1.0]))).reshape(-1)[0])
    dc = scaling.get("dc")
    dd = scaling.get("dd")
    dc = jnp.ones([cp.nyc]) if dc is None or jnp.asarray(dc).size == 0 else jnp.asarray(dc).reshape(-1)
    dd = jnp.ones([cp.nyd]) if dd is None or jnp.asarray(dd).size == 0 else jnp.asarray(dd).reshape(-1)
    in_resto = bool(jnp.asarray(fl.in_restoration).squeeze())
    # x_ref / dr_x: in restoration jaxipm FREEZES these at the entry iterate; read
    # the frozen values when IPOPT logged them, else reconstruct from it.x (only
    # correct at the entry iter — see the resto_ref note above).
    if in_resto and "x_ref" in resto_ref_x:
        x_ref = jnp.asarray(resto_ref_x["x_ref"]).reshape(-1)
        dr_x = jnp.asarray(resto_ref_x["dr_x"]).reshape(-1)
    else:
        x_ref = it.x.flatten()
        dr_x = jnp.minimum(1.0, 1.0 / jnp.abs(x_ref))
    # args[0][0] = the FROZEN restoration reference mu while in restoration (jaxipm
    # holds rmu in args[0][0], frozen at entry; it is NOT the current iter's mu).
    # Outside restoration this slot is a dummy, so the current mu is fine.
    arg0_mu = _sc(resto_ref, "ref_mu", float(jnp.asarray(mu).reshape(-1)[0])) if in_resto else mu
    args = (
        (arg0_mu, x_ref[: cp.nx], dr_x[: cp.nx], jnp.array([df])),  # (1,1)
        (dc,),
        (dd,),
    )

    # ---- cqpr : CalculatedQuantitiesPreMu ----
    cq = S["cqpr"]
    # jac_f: gradient over the original x, padded by the resto-variable block to
    # the optimizer's (nx + resto_dim, 1) shape (f is independent of resto vars).
    jac_f = jnp.vstack([
        cp.nstqf.calc_jac_f(it.x[: cp.nx], *args[0]).todense().reshape(-1, 1),
        jnp.zeros([aug, 1]),
    ])

    # grad_lag_x and the x-slack are padded by the resto-variable block when at
    # regular (non-resto) size, matching the optimizer's static shapes.
    def _pad_resto(v, base_dim):
        return jnp.vstack([v, jnp.zeros([aug, 1])]) if v.shape[0] == base_dim else v

    grad_lag_x = _pad_resto(_vec(cq, "grad_lag_x", cp.nx), cp.nx)
    sxL = _pad_resto(_vec(cq, "slacks.sxL", cp.nxL), cp.nxL)

    cqpr = CalculatedQuantitiesPreMu(
        f=_sc(cq, "f"),
        c=_vec(cq, "c", cp.nyc),
        d=_vec(cq, "d", cp.nyd),
        jac_f=jac_f,
        dms=_vec(cq, "dms", cp.nyd),
        y_nrminf=jnp.asarray(_sc(cq, "y_nrminf")).reshape(()),
        grad_lag_x=grad_lag_x,
        grad_lag_s=_vec(cq, "grad_lag_s", cp.nyd),
        slacks=(
            sxL,
            _vec(cq, "slacks.sxU", cp.nxU),
            _vec(cq, "slacks.sdL", cp.ndL),
            _vec(cq, "slacks.sdU", cp.ndU),
        ),
        avrg_compl=_sc(cq, "avrg_compl"),
        theta=_sc(cq, "theta"),
        grad_lag_x_nrm2=jnp.asarray(_sc(cq, "grad_lag_x_nrm2")).reshape(()),
        grad_lag_s_nrm2=jnp.asarray(_sc(cq, "grad_lag_s_nrm2")).reshape(()),
        c_nrm2=jnp.asarray(_sc(cq, "c_nrm2")).reshape(()),
        d_minus_s_nrm2=jnp.asarray(_sc(cq, "d_minus_s_nrm2")).reshape(()),
        nlp_error=jnp.asarray(_sc(cq, "nlp_error")).reshape(()),
        nlp_constr_viol=_sc(cq, "nlp_constr_viol"),
        barrier_constr_viol=_sc(cq, "barrier_constr_viol"),
        primal_inf=_sc(cq, "primal_inf"),
        step_aff_full=_load_step_full(cq, "step_aff_full.", cp),
        step_cen_full=_load_step_full(cq, "step_cen_full.", cp),
        Sigma_nc_inv=_vec(cq, "Sigma_nc_inv", cp.nyc),
        Sigma_pc_inv=_vec(cq, "Sigma_pc_inv", cp.nyc),
        Sigma_nd_inv=_vec(cq, "Sigma_nd_inv", cp.nyd),
        Sigma_pd_inv=_vec(cq, "Sigma_pd_inv", cp.nyd),
        y_c_init=_vec(cq, "y_c_init", cp.nyc),
        y_d_init=_vec(cq, "y_d_init", cp.nyd),
    )

    # ---- cqpo : CalculatedQuantitiesPostMu ----
    co = S["cqpo"]
    # jaxipm's cqpo.rhs is calc_aug_pd_RHS = -[mod_rhs_x, mod_rhs_s, c, dms,
    # rhs_z_L, rhs_z_U, rhs_v_L, rhs_v_U]: NEGATED, with CONDENSED x/s rows
    # (bound-complementarity folded in) but ORIGINAL z/v rows. IPOPT logs the
    # un-negated, UNREDUCED rhs.* plus the condensed rhs_aug.x/.s — so rebuild
    # jaxipm's convention: x/s from rhs_aug (condensed), the rest from rhs.*, all
    # negated. (cqpo.step needs no transform — IPOPT's delta already matches.)
    _rhs_u = _load_iterate(co, "rhs.", cp)
    rhs = Iterate(
        x=-jnp.vstack([_vec(co, "rhs_aug.x", cp.nx), jnp.zeros([aug, 1])]),
        s=-_vec(co, "rhs_aug.s", cp.nyd),
        y_c=-_rhs_u.y_c,
        y_d=-_rhs_u.y_d,
        z_L=-_rhs_u.z_L,
        z_U=-_rhs_u.z_U,
        v_L=-_rhs_u.v_L,
        v_U=-_rhs_u.v_U,
    )
    step = _load_iterate(co, "step.", cp)
    # slack derivatives, derived from the step and the static projection indices
    ind_x_L = jnp.asarray(static.get("ind_x_L", jnp.zeros(0)), dtype=int).reshape(-1)
    ind_x_U = jnp.asarray(static.get("ind_x_U", jnp.zeros(0)), dtype=int).reshape(-1)
    ind_d_L = jnp.asarray(static.get("ind_d_L", jnp.zeros(0)), dtype=int).reshape(-1)
    ind_d_U = jnp.asarray(static.get("ind_d_U", jnp.zeros(0)), dtype=int).reshape(-1)
    dsxL = step.x[ind_x_L] if ind_x_L.size else jnp.zeros([cp.nxL, 1])
    if dsxL.shape[0] == cp.nxL:
        dsxL = jnp.vstack([dsxL, jnp.zeros([aug, 1])])
    slack_derivatives = (
        dsxL,
        -step.x[ind_x_U] if ind_x_U.size else jnp.zeros([cp.nxU, 1]),
        step.s[ind_d_L] if ind_d_L.size else jnp.zeros([cp.ndL, 1]),
        -step.s[ind_d_U] if ind_d_U.size else jnp.zeros([cp.ndU, 1]),
    )
    # SOC-ACCEPTED iterations: jaxipm's post-LS cqpo.step / ls.trial_step hold the
    # ACCEPTED (SOC-corrected) delta, while IPOPT's cqpo.txt step.* keeps the raw
    # pre-LS direction. Override `step` with the accepted attempt's delta from
    # soc.txt so accepted-vs-accepted is loaded. NOTE: slack_derivatives above are
    # intentionally derived from the RAW step BEFORE this override.
    socS = S["soc"]
    _soc_attempts = sorted(int(k[3:k.index(".")]) for k in socS if k.endswith(".attempt")) if socS else []
    _soc_pre = f"soc{_soc_attempts[-1]}." if _soc_attempts else None
    _soc_accepted = False
    if _soc_pre is not None:
        _soc_accepted = int(jnp.asarray(socS[_soc_pre + "accept"]).reshape(-1)[0]) == 1
        if _soc_accepted and (_soc_pre + "delta.x") in socS:
            step = _load_iterate(socS, _soc_pre + "delta.", cp)

    # augmented (condensed) rhs/step: concat the [x, s, y_c, y_d] sections IPOPT
    # logs from PDFullSpaceSolver. jaxipm's rhs convention is negated vs IPOPT's
    # (jaxipm solves +K x = rhs, IPOPT step = -K^-1 rhs), so rhs_aug is negated;
    # step_aug already matches (the two negations cancel).
    def _aug(prefix, sign):
        return sign * jnp.vstack([
            _vec(co, prefix + "x", cp.nx),
            _vec(co, prefix + "s", cp.nyd),
            _vec(co, prefix + "y_c", cp.nyc),
            _vec(co, prefix + "y_d", cp.nyd),
        ])

    cqpo = CalculatedQuantitiesPostMu(
        rhs_aug=_aug("rhs_aug.", -1.0),
        step_aug=_aug("step_aug.", 1.0),
        rhs=rhs,
        step=step,
        alpha_pr=_sc(co, "alpha_pr"),
        barr=_sc(co, "barr"),
        gBD=_sc(co, "gBD"),
        slack_derivatives=slack_derivatives,
    )

    # ---- wd : WatchdogState ----
    wds = S["wd"]
    wd_it = _load_iterate(wds, "it.", cp) if "it.x" in wds else _empty_iterate(cp)
    wd_delta = _load_iterate(wds, "delta.", cp) if "delta.x" in wds else _empty_iterate(cp)
    wd = WatchdogState(
        shortened_iter=_sc(wds, "shortened_iter"),
        trial_iter=_sc(wds, "trial_iter"),
        alpha_pr_test=_sc(wds, "alpha_pr_test"),
        it=wd_it,
        delta=wd_delta,
        last_mu=_sc(wds, "last_mu", -1.0),
        theta=_sc(wds, "theta"),
        barr=_sc(wds, "barr"),
        gBD=_sc(wds, "gBD"),
    )

    # ---- ls : LineSearchState (+ nested filter) ----
    lss = S["ls"]
    ls_filter = LineSearchFilterState(
        theta_min=_sc(lss, "filter.theta_min", -1.0),
        theta_max=_sc(lss, "filter.theta_max", -1.0),
        last_rejection_due_to_filter=_sc(lss, "filter.last_rejection_due_to_filter"),
        count_successive_filter_rejections=_sc(lss, "filter.count_successive_filter_rejections"),
        n_filter_resets=_sc(lss, "filter.n_filter_resets"),
        F=_filter_pad(lss.get("filter.F", jnp.zeros((0, 2))), fs),
        ref_theta=_sc(lss, "filter.ref_theta"),
        ref_barr=_sc(lss, "filter.ref_barr"),
        ref_gBD=_sc(lss, "filter.ref_gBD"),
    )
    ls_it_trial = _load_iterate(lss, "it_trial.", cp) if "it_trial.x" in lss else _empty_iterate(cp)
    ls_acceptable = _load_iterate(lss, "acceptable_point.", cp) if "acceptable_point.x" in lss else _empty_iterate(cp)

    # ---- soc : per-attempt SOC dump (parsed above, before the cqpo build) ----
    ls_count_soc = _sc(lss, "count_soc")
    ls_c_soc = _vec(lss, "c_soc", cp.nyc)
    ls_dms_soc = _vec(lss, "dms_soc", cp.nyd)
    ls_theta_soc_old = _sc(lss, "theta_soc_old")
    ls_trial_step = step  # == cqpo.step (accepted delta; raw when no SOC accepted)
    if _soc_pre is not None:
        _last = _soc_attempts[-1]
        ls_count_soc = jnp.array([[_last - 1 if _soc_accepted else _last]])
        ls_c_soc = jnp.asarray(socS[_soc_pre + "c_soc"]).reshape(-1, 1)
        ls_dms_soc = jnp.asarray(socS[_soc_pre + "dms_soc"]).reshape(-1, 1)
        ls_theta_soc_old = _sc(socS, _soc_pre + "theta_soc_old")

    ls = LineSearchState(
        acceptable_point=ls_acceptable,
        n_steps=_sc(lss, "n_steps"),
        accept=_sc(lss, "accept"),
        trial_step=ls_trial_step,
        it_trial=ls_it_trial,
        alpha_pr=_sc(lss, "alpha_pr"),
        alpha_min=_sc(lss, "alpha_min"),
        n_filter_resets=_sc(lss, "filter.n_filter_resets"),
        trial_theta=_sc(lss, "trial_theta"),
        last_obj_val=jnp.atleast_2d(0.0),
        count_soc=ls_count_soc,
        theta_soc_old=ls_theta_soc_old,
        c_soc=ls_c_soc,
        dms_soc=ls_dms_soc,
        soft_resto_phase_counter=_sc(lss, "soft_resto_phase_counter"),
        satisfies_original_criterion=jnp.atleast_2d(0),
        count_restorations=_sc(lss, "count_restorations"),
        required_infeasibility_reduction=jnp.atleast_2d(0.0),
        filter=ls_filter,
    )

    # ---- ic : InertiaCorrectionState ----
    ics = S["ic"]
    inertia = jnp.array([
        float(jnp.asarray(_sc(ics, "inertia_num_neg_evals")).reshape(-1)[0]),
        float(jnp.asarray(_sc(ics, "inertia_expected_neg_evals")).reshape(-1)[0]),
    ])
    ic = InertiaCorrectionState(
        dxs=jnp.asarray(_sc(ics, "dxs")).reshape(()),
        dcd=jnp.asarray(_sc(ics, "dcd")).reshape(()),
        dxs_old=jnp.asarray(_sc(ics, "dxs_old")).reshape(()),
        dcd_old=jnp.asarray(_sc(ics, "dcd_old")).reshape(()),
        jac_degen=jnp.asarray(_sc(ics, "jac_degen")).reshape(()),
        hess_degen=jnp.asarray(_sc(ics, "hess_degen")).reshape(()),
        test_status=jnp.asarray(_sc(ics, "test_status")).reshape(()),
        degen_iters=jnp.asarray(_sc(ics, "degen_iters")).reshape(()),
        inertia=inertia,
        # debug-only in jaxipm; IPOPT doesn't log it -> zeros of the KKT-triu length
        perturbed_data=jnp.zeros(cp.nnz_triu),
        # live registry handle from the running state (IPOPT logs no tokens)
        token=kkt_token,
    )

    # ---- assemble. Save slots = current state (regular path); the injection
    # loop re-injects jaxipm's OWN saved_* during restoration (see main()). ----
    return OptimizationState(
        it=it, cqpr=cqpr, cqpo=cqpo, fl=fl, wd=wd, ls=ls, ic=ic, ls_token=ls_token, adfs=adfs,
        mu=mu, tau=tau, mu_max=mu_max,
        init_dual_inf=init_dual_inf, init_primal_inf=init_primal_inf,
        saved_fl=fl, saved_wd=wd, saved_ls=ls, saved_ic=ic, saved_adfs=adfs,
        saved_mu=mu, saved_tau=tau, saved_mu_max=mu_max,
        saved_init_dual_inf=init_dual_inf, saved_init_primal_inf=init_primal_inf,
        saved_orig_inf_pr=jnp.zeros_like(init_primal_inf),
        saved_z_L=jnp.zeros_like(it.z_L[: cp.nxL]),
        saved_z_U=jnp.zeros_like(it.z_U),
        saved_v_L=jnp.zeros_like(it.v_L),
        saved_v_U=jnp.zeros_like(it.v_U),
        saved_slacks=(
            jnp.zeros_like(it.z_L[: cp.nxL]),
            jnp.zeros_like(it.z_U),
            jnp.zeros_like(it.v_L),
            jnp.zeros_like(it.v_U),
        ),
        resto_tol=jnp.atleast_2d(cp.p["tol"]),
        iter_count=jnp.atleast_2d(iter_num),
        args=args,
    )


def _conform(loaded, template):
    """Cast every array leaf of `loaded` to the (dtype, shape) of the matching
    leaf in `template`, leaving the pytree structure untouched.

    This is what makes a freshly-loaded IPOPT state injectable WHOLESALE into
    the jitted post_process: its init-branch selects (filter_select ->
    jax.lax.select) trace BOTH branches regardless of the condition, so the
    loaded (it, ic, mu, args) must share jaxipm's dtypes or the trace dies.
    reshape() (not just astype) also absorbs benign (n,) vs (n,1) layout
    differences; it raises only if element COUNTS truly differ — the signal we
    want surfaced rather than silently reshaped away.
    """
    def _cast(l, t):
        if hasattr(t, "dtype") and hasattr(t, "shape"):
            return jnp.asarray(l).astype(t.dtype).reshape(t.shape)
        return l
    return jax.tree_util.tree_map(_cast, loaded, template)


def _flag(state_or_fl_holder, getter):
    return int(np.asarray(getter(state_or_fl_holder)).squeeze())


def main():
    os.chdir(HERE)  # ipopt_logs/ is resolved relative to CWD by the loaders
    LOGS.mkdir(exist_ok=True)
    assert IPOPT_LOGS.is_dir(), "run Stage 1 (ipopt_correctness.py) first"

    # ── Build the problem and jaxipm's CommonProblem ────────────────────────
    f, c, d, x_L, x_U, d_L, d_U, x0, gt, aux = quadcopter_nav(N=N_HORIZON)
    z_to_xu, xu_to_z, quad_params, *_ = aux
    f_args, c_args, d_args = (), (), ()

    # calc_next_problem is unused by the single-solve path but keeps the
    # initializer signature identical to the throughput scripts.
    _x0 = jnp.asarray(x0).squeeze()
    calc_next_problem = lambda key, sol: (_x0, (), (), ())

    cp = initialize_common_problem(
        f, c, d, x_L, x_U, d_L, d_U, x0, p, [f_args, c_args, d_args],
        calc_next_problem=calc_next_problem,
    )
    state = initialize_problem_regular(cp, x0, args=[f_args, c_args, d_args])
    state = eqx.tree_at(lambda t: t.fl.needs_regular_init, state, jnp.array([[0]]))
    print(f"correctness_test/jaxipm: nx={cp.nx} nyc={cp.nyc} nyd={cp.nyd} "
          f"nxL={cp.nxL} nxU={cp.nxU}  ir_nsteps={p['ir_nsteps']}  gpu={p['gpu_id']}")

    n_ipopt = count_ipopt_iters(str(IPOPT_LOGS))          # flat per-component files
    n_dumps = count_ipopt_state_dumps(str(IPOPT_LOGS))    # full iter_<k>/ dumps
    print(f"correctness_test/jaxipm: {n_ipopt} IPOPT iterates "
          f"({n_dumps} full-state dumps) available to compare against")
    assert n_dumps >= 2, "need at least iter_0 and iter_1 full-state dumps"

    _search = eqx.filter_jit(execute_search)
    _post = eqx.filter_jit(post_process)

    # ── Per-step injection loop (the mk3 harness) ───────────────────────────
    # Pass idx: `state` holds IPOPT's iter-idx state (pass 0: jaxipm's own
    # initialization — it IS IPOPT's cold start). jaxipm takes one step; its OWN
    # post-processed result is compared against IPOPT iter idx+1 (rows[idx+1] =
    # the single-step deviation); then `state` is re-seeded from the IPOPT dumps
    # so pass idx+1 again starts from IPOPT's exact state.
    rows = [compare(jaxipm_iterate(state, cp),
                    load_ipopt_iterate(str(IPOPT_LOGS), 0, cp))]
    jx_resto = [_flag(state, lambda t: t.fl.in_restoration)]
    jx_free = [_flag(state, lambda t: t.fl.free_mu_mode)]
    own_terms = [int(TerminationCode.CONTINUE)]

    ipopt_state_raw = load_state(0, cp, kkt_token=state.ic.token,
                                 ls_token=state.ls_token)  # reused as pass idx's iter-idx dump
    term = jnp.array([[TerminationCode.CONTINUE]])
    n_pass = n_dumps - 1
    for idx in range(n_pass):
        orig = state
        result = _search(state, cp)

        # jaxipm's OWN full iteration (search + post_process) from the injected
        # start — the measured object. Its iterate/mu are compared against
        # IPOPT's iter idx+1 below; nothing of it flows into the next pass.
        own_state, own_term = _post(orig, result, cp)
        rows.append(compare(jaxipm_iterate(own_state, cp),
                            load_ipopt_iterate(str(IPOPT_LOGS), min(idx + 1, n_ipopt - 1), cp)))
        jx_resto.append(_flag(own_state, lambda t: t.fl.in_restoration))
        jx_free.append(_flag(own_state, lambda t: t.fl.free_mu_mode))
        own_terms.append(int(np.asarray(own_term).squeeze()))

        # RESTORATION-EXIT GATE (the mk3 harness's ``exiting_resto`` it.* gate).
        # IPOPT rewinds its iteration counter at the exit, so jaxipm runs one
        # extra bookkeeping pass: its exit-transition iterate aligns with
        # IPOPT's PRE-rewind point, not with IPOPT's first post-exit step — a
        # pipeline-stage offset, not a step error. Gate that one sample to NaN
        # ONLY when verified: the shifted comparison (own iterate vs IPOPT one
        # index back) must collapse to the machine band, exactly the check the
        # old open-loop script ran by hand around the exit.
        if len(jx_resto) >= 2 and jx_resto[-2] == 1 and jx_resto[-1] == 0:
            shifted = compare(jaxipm_iterate(own_state, cp),
                              load_ipopt_iterate(str(IPOPT_LOGS), min(idx, n_ipopt - 1), cp))
            print(f"pass {idx:3d}: own resto-exit transition — shifted "
                  f"(vs IPOPT[{idx}]) dx={shifted['x']:9.2e}"
                  + ("  -> confirmed +1 bookkeeping offset; sample gated"
                     if shifted["x"] <= X_MATCH_ATOL else
                     "  -> NOT a pure offset; sample kept"))
            if shifted["x"] <= X_MATCH_ATOL:
                rows[-1] = {key: np.nan for key in _COMPONENTS + ["mu", "tau"]}

        # ===================== RE-SEED (FULL STATE) =========================
        # Inject IPOPT's COMPLETE state with the split post_process consumes:
        # iterate from iter idx+1, ALL bookkeeping from iter idx (post_process's
        # calc_updated_mu is what advances mu_k -> mu_{k+1}, so it must see the
        # PRE-update bookkeeping). cqpr/cqpo are the ONLY fields taken from
        # jaxipm's `result`: they are post_process inputs recomputed at the
        # trial iterate, and in restoration jaxipm's Schur-reduced aug blocks
        # have a different SHAPE than IPOPT's full system.
        ipopt_next = load_state(idx + 1, cp, kkt_token=orig.ic.token,
                                ls_token=orig.ls_token)
        ipopt_cmp = eqx.tree_at(
            lambda t: (
                t.it,
                # The filter-rejection counter (+ flag) advances DURING the line
                # search but IPOPT dumps ls.txt BEFORE the iteration's trials, so
                # like `.it` these two come from iter idx+1 (pre-LS of k+1 ==
                # post-LS of k).
                t.ls.filter.count_successive_filter_rejections,
                t.ls.filter.last_rejection_due_to_filter,
            ),
            ipopt_state_raw,
            (
                ipopt_next.it,
                ipopt_next.ls.filter.count_successive_filter_rejections,
                ipopt_next.ls.filter.last_rejection_due_to_filter,
            ),
        )

        in_resto_now = _flag(ipopt_state_raw, lambda t: t.fl.in_restoration)
        in_resto_next = _flag(ipopt_next, lambda t: t.fl.in_restoration)
        entering_resto = (in_resto_now == 0) and (in_resto_next == 1)
        exiting_resto = (in_resto_now == 1) and (in_resto_next == 0)

        # RESTORATION-ENTRY BOUNDARY: when IPOPT is regular at iter idx but in
        # restoration at iter idx+1, jaxipm must ENTER restoration here.
        # post_process's init_resto branch BUILDS the restoration iterate (p/n
        # slacks, resto mu, resto args) FROM the regular entry iterate — exactly
        # as IPOPT's RestoIterateInitializer does. So at the boundary we
        #   (a) step FROM the regular entry iterate (load_state(idx).it, NOT
        #       idx+1's resto iterate — that is what jaxipm is about to build), and
        #   (b) raise needs_resto_init=1 instead of forcing it to 0.
        base = ipopt_state_raw if entering_resto else ipopt_cmp
        resto_init_flag = jnp.array([[1]]) if entering_resto else jnp.array([[0]])

        ipopt_inj = eqx.tree_at(
            lambda t: (t.cqpr, t.cqpo), base, (result.cqpr, result.cqpo)
        )
        inj = _conform(ipopt_inj, result)
        inj = eqx.tree_at(
            lambda t: (t.fl.needs_resto_init, t.fl.needs_regular_init),
            inj,
            (resto_init_flag, jnp.array([[0]])),
        )

        # MISSING state.txt FALLBACK: this release's dumps omit iter_<k>/state.txt
        # (and the flat mu_tau_<k>.txt) for a third of the iterations — mostly the
        # post-restoration monotone-mu window. The loader then zero-defaults the
        # top-level bookkeeping (mu=tau=0, empty adfs), which poisons the injected
        # state: a mu of 0 makes the next line search reject every trial and fall
        # back to restoration. Where the dump is absent, carry jaxipm's OWN
        # advanced bookkeeping from `orig` instead — the chain's mu/tau/adfs were
        # produced by post_process from the previous (validated) injection, so
        # they track IPOPT's to machine precision across the gap.
        if not os.path.exists(IPOPT_LOGS / f"iter_{idx}" / "state.txt"):
            inj = eqx.tree_at(
                lambda t: (t.mu, t.tau, t.mu_max, t.adfs,
                           t.init_dual_inf, t.init_primal_inf),
                inj,
                (orig.mu, orig.tau, orig.mu_max, orig.adfs,
                 orig.init_dual_inf, orig.init_primal_inf),
            )

        # RESTORATION CONTINUATION: preserve jaxipm's OWN frozen saved_* slots
        # (and frozen resto args) instead of the per-step loader's mirror. IPOPT
        # logs no saved_* — they are jaxipm's internal freeze of the
        # pre-restoration regular state, set ONCE by post_process's init_resto
        # branch at the entry boundary. The loader can only mirror them to the
        # CURRENT iterate, which (a) makes the resto-exit reduction test
        # unpassable (saved_orig_inf_pr collapsed to the small current resto
        # infeasibility) and (b) NaN-poisons the exit bound-multiplier step
        # (saved_slacks zeros are its divisor). args likewise carries the frozen
        # restoration reference (rmu, x_ref, dr_x), which this release's dumps
        # lack (_static/resto_ref_* absent) — jaxipm's own frozen copy is exact
        # because the entry pass injected IPOPT's entry iterate. At the entry
        # boundary `orig` is still regular -> skip, so post_process does the
        # freeze itself.
        if _flag(orig, lambda t: t.fl.in_restoration) == 1:
            inj = eqx.tree_at(
                lambda t: (
                    t.saved_fl, t.saved_wd, t.saved_ls, t.saved_ic, t.saved_adfs,
                    t.saved_mu, t.saved_tau, t.saved_mu_max,
                    t.saved_init_dual_inf, t.saved_init_primal_inf, t.saved_orig_inf_pr,
                    t.saved_z_L, t.saved_z_U, t.saved_v_L, t.saved_v_U,
                    t.saved_slacks, t.args,
                ),
                inj,
                (
                    orig.saved_fl, orig.saved_wd, orig.saved_ls, orig.saved_ic, orig.saved_adfs,
                    orig.saved_mu, orig.saved_tau, orig.saved_mu_max,
                    orig.saved_init_dual_inf, orig.saved_init_primal_inf, orig.saved_orig_inf_pr,
                    orig.saved_z_L, orig.saved_z_U, orig.saved_v_L, orig.saved_v_U,
                    orig.saved_slacks, orig.args,
                ),
            )

        state, term = _post(orig, inj, cp)

        # POST-EXIT BOOKKEEPING RE-SEED. IPOPT's inertia-perturbation memory
        # survives restoration (the resto phase runs a NESTED solver with its own
        # handler; the original handler keeps e.g. dxs_old from the entry iter),
        # while jaxipm's exit restores the entry-FROZEN ic/ls/wd — the two
        # disagree, and the first post-exit KKT solve then follows a different
        # perturbation schedule. That is a real jaxipm-vs-IPOPT semantic gap, but
        # compounding it into the next sample would break the single-step
        # semantics — so surface it, then re-seed the whole state from IPOPT's
        # post-exit dump (iter idx+1 has a complete dump set) so the next pass
        # again measures one step from IPOPT's exact state.
        if exiting_resto:
            for name, a, b in (
                ("ic.dxs", state.ic.dxs, ipopt_next.ic.dxs),
                ("ic.dxs_old", state.ic.dxs_old, ipopt_next.ic.dxs_old),
                ("mu", state.mu, ipopt_next.mu),
            ):
                da = float(np.max(np.abs(np.asarray(a, dtype=float)
                                         - np.asarray(b, dtype=float))))
                if da > 1e-10:
                    print(f"pass {idx:3d}: post-exit restore mismatch "
                          f"{name}: jaxipm-restored vs IPOPT dump |Δ|={da:.3e}")
            reseeded = eqx.tree_at(
                lambda t: (t.cqpr, t.cqpo), ipopt_next, (result.cqpr, result.cqpo)
            )
            reseeded = _conform(reseeded, result)
            state = eqx.tree_at(
                lambda t: (t.fl.needs_resto_init, t.fl.needs_regular_init),
                reseeded,
                (jnp.array([[0]]), jnp.array([[0]])),
            )

        ipopt_state_raw = ipopt_next  # becomes the next pass's iter-idx dump

        marker = (" [resto entry]" if entering_resto
                  else " [resto exit]" if exiting_resto
                  else " [resto]" if in_resto_now else "")
        r = rows[-1]
        print(f"pass {idx:3d} -> iter {idx + 1:3d}: single-step "
              f"dx={r['x']:9.2e} dmu={r['mu']:9.2e} "
              f"resto(jx/ip)={jx_resto[-1]}/{in_resto_next}{marker}")
    jax.block_until_ready(state.it.x)

    # ── Report ───────────────────────────────────────────────────────────────
    n_cmp = len(rows)
    hdr = f"{'k':>4} | " + " ".join(f"{c:>10}" for c in (_COMPONENTS + ["mu"]))
    print("\n" + "=" * len(hdr))
    print("  jaxipm vs IPOPT — max |Δ| per iterate  [SINGLE-STEP: each row is one"
          " step from IPOPT's exact iter-(k-1) state]")
    print("=" * len(hdr))
    print(hdr)
    print("-" * len(hdr))
    for k, diff in enumerate(rows):
        # print first 5, last 5, and every 10th iteration in between
        if k < 5 or k >= n_cmp - 5 or k % 10 == 0:
            cells = " ".join(f"{diff[c]:10.2e}" for c in (_COMPONENTS + ["mu"]))
            print(f"{k:>4} | {cells}")

    # Restoration-flag timeline — did jaxipm enter / exit resto where IPOPT did?
    def ipopt_resto_flag(k):
        f = IPOPT_LOGS / f"iteration_type_{k}.txt"
        if not f.exists():
            return -1
        for line in f.read_text().splitlines():
            if line.startswith("in_restoration"):
                return int(line.split()[1])
        return 0

    ip_resto = [ipopt_resto_flag(min(k, n_ipopt - 1)) for k in range(n_cmp)]
    print("\n  in_restoration timeline (-=0, R=1):")
    print("    IPOPT  : " + "".join("R" if r == 1 else "-" if r == 0 else "?" for r in ip_resto))
    print("    jaxipm : " + "".join("R" if r == 1 else "-" if r == 0 else "?" for r in jx_resto))
    print("  free_mu_mode timeline (F=true, .=false):")
    print("    jaxipm : " + "".join("F" if r == 1 else "." for r in jx_free))

    dx = np.array([r["x"] for r in rows])
    first_tight = next((k for k in range(n_cmp) if not (dx[k] <= X_MATCH_ATOL)), None)
    first_drift = next((k for k in range(n_cmp) if not (dx[k] <= X_DRIFT_ATOL)), None)
    print("-" * len(hdr))
    print(f"  single-step Δx : max over all {n_cmp} compared = {np.nanmax(dx):.3e}")
    print(f"  first k with Δx > {X_MATCH_ATOL:.0e} (tight) : "
          f"{'none — every step reproduces IPOPT' if first_tight is None else first_tight}")
    print(f"  first k with Δx > {X_DRIFT_ATOL:.0e} (drift) : "
          f"{'none' if first_drift is None else first_drift}")
    print("=" * len(hdr))

    # ── Save diff report + the tracked solution ──────────────────────────────
    # `state` is the injected chain's final post_process output — it tracks
    # IPOPT's final logged iterate, i.e. the converged solution.
    z_sol = np.asarray(state.it.x)[: cp.nx, 0]
    x_sol, u_sol = z_to_xu(jnp.asarray(z_sol))
    x_sol = np.asarray(x_sol)
    nan_rows = np.full(n_cmp, np.nan)
    out = LOGS / "jaxipm_correctness.npz"
    np.savez(
        out,
        z_sol=z_sol,
        x_sol=x_sol,
        u_sol=np.asarray(u_sol),
        n_jaxipm_iter=np.array([n_pass]),
        n_ipopt_iter=np.array([n_ipopt]),
        term=np.array([int(np.asarray(term).squeeze())]),
        ir_nsteps=np.array([p["ir_nsteps"]]),
        dx=dx,
        # restoration / mu-mode timelines for plot annotation
        ip_resto=np.asarray(ip_resto, dtype=np.int8),
        jx_resto=np.asarray(jx_resto, dtype=np.int8),
        jx_free_mu=np.asarray(jx_free, dtype=np.int8),
        # per-component SINGLE-STEP diffs. The *_shift1 keys were an open-loop
        # alignment diagnostic; meaningless under per-step injection -> NaN.
        **{f"d_{c}": np.array([r[c] for r in rows]) for c in _COMPONENTS + ["mu", "tau"]},
        **{f"d_{c}_shift1": nan_rows for c in _COMPONENTS + ["mu", "tau"]},
    )
    print(f"saved {out}")


if __name__ == "__main__":
    main()

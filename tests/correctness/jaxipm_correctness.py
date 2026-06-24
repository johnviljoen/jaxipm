"""
Stage 2 of the correctness_test experiment.

Runs jaxipm open-loop on the *same* fixed-start quadcopter nav problem that
Stage 1 (ipopt_correctness.py) solved with the patched IPOPT, then validates
jaxipm's iterate trajectory against ipopt_logs/ one iteration at a time.

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


# ── Reference loading ───────────────────────────────────────────────────────
def count_ipopt_iters(save_dir):
    """Number of consecutive IPOPT iterates k = 0, 1, 2, ... that were logged."""
    k = 0
    while (os.path.exists(f"{save_dir}/iterate_x_{k}_comp0.txt")
           or os.path.exists(f"{save_dir}/iterate_{k}_x.txt")):
        k += 1
    return k


def _col(vec):
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
        "x": _col(x),
        "s": _col(load_vector(f"{save_dir}/iterate_{k}_s.txt")),
        "y_c": _col(load_vector(f"{save_dir}/iterate_{k}_y_c.txt")),
        "y_d": _col(load_vector(f"{save_dir}/iterate_{k}_y_d.txt")),
        "z_L": _col(z_L),
        "z_U": _col(load_vector(f"{save_dir}/iterate_{k}_z_U.txt")),
        "v_L": _col(load_vector(f"{save_dir}/iterate_{k}_v_L.txt")),
        "v_U": _col(load_vector(f"{save_dir}/iterate_{k}_v_U.txt")),
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
# iterate that defines a "matched" iteration.
_COMPONENTS = ["x", "s", "y_c", "y_d", "z_L", "z_U", "v_L", "v_U"]
X_MATCH_ATOL = 1e-6   # tight band — true 1-to-1 tracking
X_DRIFT_ATOL = 1e-4   # loose band — first real divergence


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

    n_ipopt = count_ipopt_iters(str(IPOPT_LOGS))
    print(f"correctness_test/jaxipm: {n_ipopt} IPOPT iterates available to compare against")

    # ── Open-loop solve, capturing every iterate ────────────────────────────
    _search = eqx.filter_jit(execute_search)
    _post = eqx.filter_jit(post_process)

    states = [state]
    term = jnp.array([[TerminationCode.CONTINUE]])
    # Generous cap: let jaxipm run to its own convergence even if it needs more
    # iterations than IPOPT did (the +1 bookkeeping iter alone guarantees ≥1
    # extra). The downstream comparison clamps to IPOPT's final iterate after
    # k >= n_ipopt, so deviations drop to zero as jaxipm catches up.
    max_steps = max(200, n_ipopt + 50)
    while int(term.squeeze()) == TerminationCode.CONTINUE and len(states) <= max_steps:
        orig = state
        result = _search(state, cp)
        state, term = _post(orig, result, cp)
        states.append(state)
    jax.block_until_ready(state.it.x)
    n_jaxipm = len(states) - 1
    print(f"correctness_test/jaxipm: open-loop solve ran {n_jaxipm} iterations "
          f"(term code {int(term.squeeze())})")

    # ── Validate iterate-by-iterate against ipopt_logs ──────────────────────
    # Compare every jaxipm iter to IPOPT. For k >= n_ipopt, IPOPT has already
    # converged so we clamp to its final iterate — deviations should decay to
    # zero as jaxipm reaches the same local minimum.
    n_cmp = len(states)
    rows = []
    for k in range(n_cmp):
        ipopt_k = min(k, n_ipopt - 1)
        diff = compare(jaxipm_iterate(states[k], cp),
                       load_ipopt_iterate(str(IPOPT_LOGS), ipopt_k, cp))
        rows.append(diff)

    hdr = f"{'k':>4} | " + " ".join(f"{c:>10}" for c in (_COMPONENTS + ["mu"]))
    print("\n" + "=" * len(hdr))
    print("  jaxipm vs IPOPT — max |Δ| per iterate  [aligned: jaxipm[k] vs IPOPT[k]]")
    print("=" * len(hdr))
    print(hdr)
    print("-" * len(hdr))
    for k, diff in enumerate(rows):
        # print first 5, last 5, and every 10th iteration in between
        if k < 5 or k >= n_cmp - 5 or k % 10 == 0:
            cells = " ".join(f"{diff[c]:10.2e}" for c in (_COMPONENTS + ["mu"]))
            print(f"{k:>4} | {cells}")

    print("\n  Shifted alignment around restoration exit (k = 30..min(n_cmp, 45)-1)")
    s_hdr = f"{'k':>4} | " + " ".join(f"{f'Δx s={s}':>10}" for s in (0, 1, 2, 3))
    print(s_hdr)
    print("-" * len(s_hdr))
    for k in range(30, min(n_cmp, 46)):
        cells = []
        for s in (0, 1, 2, 3):
            j = k + s
            if j >= len(states):
                cells.append("    n/a   ")
            else:
                diff = compare(jaxipm_iterate(states[j], cp),
                               load_ipopt_iterate(str(IPOPT_LOGS), k, cp))
                cells.append(f"{diff['x']:10.2e}")
        print(f"{k:>4} | " + " ".join(cells))

    # Restoration-flag timeline — did jaxipm enter / exit resto where IPOPT did?
    def ipopt_resto_flag(k):
        f = IPOPT_LOGS / f"iteration_type_{k}.txt"
        if not f.exists():
            return -1
        for line in f.read_text().splitlines():
            if line.startswith("in_restoration"):
                return int(line.split()[1])
        return 0

    n_show = n_cmp
    jx_resto = [int(np.asarray(states[k].fl.in_restoration).squeeze()) for k in range(n_show)]
    ip_resto = [ipopt_resto_flag(min(k, n_ipopt - 1)) for k in range(n_show)]
    jx_free = [int(np.asarray(states[k].fl.free_mu_mode).squeeze()) for k in range(n_show)]
    print("\n  in_restoration timeline (-=0, R=1):")
    print("    IPOPT  : " + "".join("R" if r == 1 else "-" if r == 0 else "?" for r in ip_resto))
    print("    jaxipm : " + "".join("R" if r == 1 else "-" if r == 0 else "?" for r in jx_resto))
    print("  free_mu_mode timeline (F=true, .=false):")
    print("    jaxipm : " + "".join("F" if r == 1 else "." for r in jx_free))

    dx = np.array([r["x"] for r in rows])
    first_tight = next((k for k in range(n_cmp) if not (dx[k] <= X_MATCH_ATOL)), None)
    first_drift = next((k for k in range(n_cmp) if not (dx[k] <= X_DRIFT_ATOL)), None)
    print("-" * len(hdr))
    print(f"  primal iterate Δx : max over all {n_cmp} compared = {np.nanmax(dx):.3e}")
    print(f"  first k with Δx > {X_MATCH_ATOL:.0e} (tight) : "
          f"{'none — locked 1-to-1' if first_tight is None else first_tight}")
    print(f"  first k with Δx > {X_DRIFT_ATOL:.0e} (drift) : "
          f"{'none' if first_drift is None else first_drift}")
    print("=" * len(hdr))

    # ── Shifted (jaxipm[k+1] vs IPOPT[k]) — full component sweep ────────────
    # jaxipm takes one extra bookkeeping iteration at restoration exit
    rows_shift = []
    for k in range(n_cmp):
        if k + 1 < len(states):
            ipopt_k = min(k, n_ipopt - 1)
            diff = compare(jaxipm_iterate(states[k + 1], cp),
                           load_ipopt_iterate(str(IPOPT_LOGS), ipopt_k, cp))
        else:
            diff = {key: np.nan for key in _COMPONENTS + ["mu", "tau"]}
        rows_shift.append(diff)

    # ── Save diff report + jaxipm solution ──────────────────────────────────
    z_sol = np.asarray(states[-1].it.x)[: cp.nx, 0]
    x_sol, u_sol = z_to_xu(jnp.asarray(z_sol))
    x_sol = np.asarray(x_sol)
    out = LOGS / "jaxipm_correctness.npz"
    np.savez(
        out,
        z_sol=z_sol,
        x_sol=x_sol,
        u_sol=np.asarray(u_sol),
        n_jaxipm_iter=np.array([n_jaxipm]),
        n_ipopt_iter=np.array([n_ipopt]),
        term=np.array([int(term.squeeze())]),
        ir_nsteps=np.array([p["ir_nsteps"]]),
        dx=dx,
        # restoration / mu-mode timelines for plot annotation
        ip_resto=np.asarray(ip_resto, dtype=np.int8),
        jx_resto=np.asarray(jx_resto, dtype=np.int8),
        jx_free_mu=np.asarray(jx_free, dtype=np.int8),
        # per-component diffs, aligned and shifted
        **{f"d_{c}": np.array([r[c] for r in rows]) for c in _COMPONENTS + ["mu", "tau"]},
        **{f"d_{c}_shift1": np.array([r[c] for r in rows_shift]) for c in _COMPONENTS + ["mu", "tau"]},
    )
    print(f"saved {out}")

    # ── Visualise jaxipm's open-loop plan (3D gif), parity with Stage 1 ─────
    # xs_arr = x_sol.copy()
    # xs_arr[:, :2] *= -1.0
    # t_arr = np.arange(xs_arr.shape[0]) * Ts
    # gif = LOGS / "jaxipm_correctness.gif"
    # BatchedAnimator(
    #     p=quad_params, xs=[xs_arr], t=t_arr,
    #     cylinder_definitions=(OBS_XC, OBS_YC, OBS_R), drawCylinder=True, dt=Ts,
    #     title="correctness_test - jaxipm open-loop plan", save_path=str(gif),
    # ).animate()
    # print(f"saved {gif}")


if __name__ == "__main__":
    main()

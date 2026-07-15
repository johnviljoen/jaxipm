"""
Stage 1 of the correctness_test experiment.

Takes the jaxipm-format quad_nav problem (the same one Stage 2 feeds to
jaxipm), converts it to cyipopt form with
jaxipm.utils.problem_format_utils.custom_to_cyipopt_format, and solves it
with the patched IPOPT (ipopt_logging_mk2: Ipopt + JohnsCustomLogging). The
patched solver dumps full per-iteration internal state — the sectioned
iter_<k>/ + _static/ schema — to ./ipopt_logs/, which Stage 2
(jaxipm_correctness.py) validates against.

Only needed to REGENERATE the reference logs: the repo ships a committed
ipopt_logs/, so Stage 2 runs out of the box. Requires a cyipopt linked
against the patched libipopt (the `jaxipm` conda env). Run from the repo
root:
    PYTHONPATH=. conda run -n jaxipm python tests/correctness/ipopt_correctness.py
"""

import os
import glob
import shutil
import time

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from cyipopt import minimize_ipopt

from jaxipm.utils.problem_format_utils import custom_to_cyipopt_format

# Single source of truth — the same problem Stage 2 feeds to jaxipm. Its
# module-level test_params.json reads are repo-root-relative, so run this
# from the repo root (same convention as Stage 2):
#     PYTHONPATH=. python tests/correctness/ipopt_correctness.py
from tests.quad_nav_circle.jaxipm_quad_nav import quadcopter_nav, N_HORIZON, TS as Ts


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    os.chdir(here)  # the patched IPOPT writes to a relative "ipopt_logs/"
    logs_dir = os.path.join(here, "logs")
    os.makedirs(logs_dir, exist_ok=True)

    # The patched IPOPT will not create its own log directory, and stale files
    # would corrupt Stage 2's per-iteration comparison — start clean. (The
    # iter_<k>/ dumps are directories, hence rmtree rather than per-file.)
    ipopt_logs = os.path.join(here, "ipopt_logs")
    if os.path.isdir(ipopt_logs):
        shutil.rmtree(ipopt_logs)
    os.makedirs(ipopt_logs, exist_ok=True)

    # ── Build the NLP (jaxipm format) and convert to cyipopt ────────────────
    f, c, d, x_L, x_U, d_L, d_U, z_init, gt, aux = quadcopter_nav(N=N_HORIZON)
    z_to_xu, xu_to_z, quad_params, _x0 = aux
    obj, obj_grad, obj_hess, constraints, bounds = custom_to_cyipopt_format(
        f, c, d, x_L, x_U, d_L, d_U, z_init
    )
    x0 = np.asarray(z_init, dtype=np.float64)
    print(f"correctness_test/IPOPT: nx={x0.size}  "
          f"nyc={np.asarray(c(z_init)).size}  nyd={np.asarray(d(z_init)).size}")

    # IPOPT options mirror src/jaxipm/params.yaml so the iterate trajectory is
    # directly comparable. max_iter is generous; we run to natural convergence.
    options = {
        "tol": 1e-8,
        "max_iter": 500,
        "mu_strategy": "adaptive",
        "print_level": 5,
        "print_timing_statistics": "yes",
    }

    # ── Solve ───────────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    result = minimize_ipopt(
        fun=obj,
        x0=x0,
        jac=obj_grad,
        hess=obj_hess,
        constraints=constraints,
        bounds=bounds,
        options=options,
    )
    solve_time = time.perf_counter() - t0

    # Authoritative iteration count: the patched solver writes one
    # iter_<k>/ full-state dump per iteration.
    n_iter_logs = len(glob.glob(os.path.join(ipopt_logs, "iter_[0-9]*")))

    print("\n" + "=" * 64)
    print("  IPOPT correctness-test solve")
    print("=" * 64)
    print(f"  status        : {result.status}  ({result.message})")
    print(f"  success       : {result.success}")
    print(f"  objective     : {float(result.fun):.10e}")
    print(f"  iterations    : {n_iter_logs} (from ipopt_logs/iter_<k>/ dumps)")
    print(f"  solve time    : {solve_time:.3f} s")
    print(f"  ipopt_logs    : dumped to {ipopt_logs}")
    print("=" * 64)

    # ── Save solution ───────────────────────────────────────────────────────
    z_sol = np.asarray(result.x, dtype=np.float64)
    x_sol, u_sol = z_to_xu(jnp.asarray(z_sol))
    x_sol = np.asarray(x_sol)  # (N, 13)
    u_sol = np.asarray(u_sol)  # (N-1, 4)

    out_path = os.path.join(logs_dir, "ipopt_correctness.npz")
    np.savez(
        out_path,
        z_sol=z_sol,
        x_sol=x_sol,
        u_sol=u_sol,
        obj=np.array([float(result.fun)]),
        n_iter=np.array([n_iter_logs]),
        status=np.array([int(result.status)]),
        success=np.array([bool(result.success)]),
        solve_time=np.array([solve_time]),
        N=np.array([N_HORIZON]),
        Ts=np.array([Ts]),
    )
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()

"""
Stage 1 of the correctness_test experiment.

Takes the jaxipm-format problem from problem.py, converts it to cyipopt form
with jaxipm.utils.sif_adapter.custom_to_cyipopt_format, and solves it with the
patched IPOPT (Ipopt 3.14.18 + JohnsCustomLogging). The patched solver dumps
full per-iteration internal state to ./ipopt_logs/, which Stage 2
(jaxipm_correctness.py) validates against.

Run with the `jaxipm` conda env (its cyipopt links the patched libipopt):
    conda run -n jaxipm python ipopt_correctness.py
or simply  ./run.sh
"""

import os
import glob
import time

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from cyipopt import minimize_ipopt

from jaxipm.utils.sif_adapter import custom_to_cyipopt_format
from problems.redundant.quadcopter_nmpc_nav.plotting_quad_nav import BatchedAnimator

# Single source of truth — the same object Stage 2 feeds to jaxipm.
from problem import quadcopter_nav, OBS_XC, OBS_YC, OBS_R, N_HORIZON, Ts


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    os.chdir(here)  # the patched IPOPT writes to a relative "ipopt_logs/"
    logs_dir = os.path.join(here, "logs")
    os.makedirs(logs_dir, exist_ok=True)

    # The patched IPOPT will not create its own log directory, and stale files
    # would corrupt Stage 2's per-iteration comparison — start clean.
    ipopt_logs = os.path.join(here, "ipopt_logs")
    if os.path.isdir(ipopt_logs):
        for fp in glob.glob(os.path.join(ipopt_logs, "*")):
            os.remove(fp)
    os.makedirs(ipopt_logs, exist_ok=True)

    # ── Build the NLP (jaxipm format) and convert to cyipopt ────────────────
    f, c, d, x_L, x_U, d_L, d_U, z_init, gt, aux = quadcopter_nav(N=N_HORIZON)
    z_to_xu, xu_to_z, quad_params = aux
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
    # iteration_type_<k>.txt per iteration.
    n_iter_logs = len(glob.glob(os.path.join(ipopt_logs, "iteration_type_*.txt")))

    print("\n" + "=" * 64)
    print("  IPOPT correctness-test solve")
    print("=" * 64)
    print(f"  status        : {result.status}  ({result.message})")
    print(f"  success       : {result.success}")
    print(f"  objective     : {float(result.fun):.10e}")
    print(f"  iterations    : {n_iter_logs} (from ipopt_logs/iteration_type_*)")
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

    # ── Visualise the open-loop planned trajectory (3D gif) ─────────────────
    xs_arr = x_sol.copy()
    xs_arr[:, :2] *= -1.0  # negate x,y to match the animator's coordinate frame
    t_arr = np.arange(xs_arr.shape[0]) * Ts
    gif_path = os.path.join(logs_dir, "ipopt_correctness.gif")
    animator = BatchedAnimator(
        p=quad_params,
        xs=[xs_arr],
        t=t_arr,
        cylinder_definitions=(OBS_XC, OBS_YC, OBS_R),
        drawCylinder=True,
        dt=Ts,
        title="correctness_test - IPOPT open-loop plan",
        save_path=gif_path,
    )
    animator.animate()
    print(f"saved {gif_path}")


if __name__ == "__main__":
    main()

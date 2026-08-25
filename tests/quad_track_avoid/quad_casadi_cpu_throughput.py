"""CasADi/IPOPT CPU-throughput benchmark for the quad track-avoid problem
(rebuttal experiment). Harness/protocol: tests/casadi_cpu_throughput_common.py
(P single-threaded IPOPT processes, phys/smt pinning sweep, work queue,
setup-excluded wall clock).

Initialization parity with jaxipm_track_avoid.py: the jaxipm run draws its
problems from a pool of N_RUNS_jaxipm random (x0, xr) pairs produced by
initialization.generate_random_inits(..., delta_std=0.5, seed=0) — a numpy
default_rng(0) pipeline, fully reproducible. Every worker here regenerates
the identical pool with the same code and seed, and the sweep solves each
pool entry exactly once (= the same number of solutions from the same
distribution; jaxipm cycles the pool with replacement after its first
batch, so the multisets differ only in which pool members repeat).
Per-solve protocol is identical to the sequential ipopt_track_avoid.py
benchmark: set (x0, xr), solve_cold() from the linear-interpolation cold
guess, same solver options; success = IPOPT converged.

Run from the repo root (defaults cover the full grid, no args needed):
    python -m tests.quad_track_avoid.quad_casadi_cpu_throughput
Optional overrides for smoke tests only:
    python -m tests.quad_track_avoid.quad_casadi_cpu_throughput \
        --vels 1.5 --modes phys --cores 4 --n-total 16
Interrupted sweeps can be continued with --resume (skips completed configs).
"""

# Import order matters: the common module pins OMP/BLAS thread env vars
# before anything pulls in casadi (both here and in spawn'd children).
from tests.casadi_cpu_throughput_common import (
    MIN_SOLVES_PER_WORKER, load_completed, parse_sweep_args, print_summaries,
    run_config,
)

import json
import os

import numpy as np

with open("tests/quad_track_avoid/test_params.json") as f:
    tmp = json.load(f)
N_HORIZON = tmp["N_horizon"]
TS = tmp["Ts"]
N_RUNS_JAXIPM = tmp["N_RUNS_jaxipm"]
AVG_VELS = tmp["avg_vel"]

# Matches jaxipm_track_avoid.py (hardcoded) and ipopt_track_avoid.py
# (env-var default).
DELTA_STD = 0.5


def generate_pool(avg_vel, n_pool):
    """The identical (x0, xr, s) pool jaxipm_track_avoid.py draws from."""
    from tests.quad_track_avoid.initialization import generate_random_inits
    all_x0, all_xr, all_s, _ = generate_random_inits(
        n_pool, N_HORIZON, TS, avg_vel, delta_std=DELTA_STD, seed=0)
    return all_x0, all_xr, all_s


def make_worker(avg_vel, n_pool):
    """Harness contract: build the MPC (untimed) and return solve(idx)."""
    from tests.quad_track_avoid.ipopt_track_avoid import QuadcopterNavMPC
    from tests.quad_track_avoid.initialization import obstacles

    all_x0, all_xr, _ = generate_pool(avg_vel, n_pool)
    mpc = QuadcopterNavMPC(all_x0[0], all_xr[0], N_HORIZON, TS,
                           obstacles=obstacles)

    def solve(idx):
        i = idx % n_pool
        # Cold solve at pool entry i, exactly as the sequential benchmark:
        # set parameters, reset the cold guess, solve.
        mpc.opti.set_value(mpc.init, all_x0[i])
        mpc.opti.set_value(mpc.Xr, all_xr[i].T)
        mpc.x0_default = all_x0[i]   # so solve_cold uses the right cold guess
        try:
            sol = mpc.solve_cold()
            X = mpc.x_sol.T
            try:
                iters = int(sol.stats().get("iter_count", -1))
            except Exception:
                iters = -1
            try:
                obj = float(sol.value(mpc.opti.f))
            except Exception:
                obj = float("nan")
            return X, iters, obj, True
        except RuntimeError:
            try:
                X = np.asarray(mpc.opti.debug.value(mpc.X)).T
                iters = int(mpc.opti.stats().get("iter_count", -1))
            except Exception:
                X = np.full((N_HORIZON, 13), np.nan)
                iters = -1
            try:
                obj = float(mpc.opti.debug.value(mpc.opti.f))
            except Exception:
                obj = float("nan")
            return X, iters, obj, False

    return solve


def main():
    args, modes, mode_cores = parse_sweep_args(
        "Parallel-process CasADi/IPOPT CPU throughput sweep (track avoid). "
        "Zero-arg run covers the full grid; args are smoke-test overrides.",
        extra_args=[("--vels", dict(
            type=str, default=None,
            help=f"comma list of avg_vel values (default: {AVG_VELS})"))],
    )
    vels = ([float(s) for s in args.vels.split(",")]
            if args.vels else AVG_VELS)
    logs_dir = os.path.join(os.path.dirname(__file__), "logs")

    all_summaries = {}
    for avg_vel in vels:
        all_x0, all_xr, all_s = generate_pool(avg_vel, N_RUNS_JAXIPM)
        for mode in modes:
            cores_list = mode_cores[mode]
            print(f"\nCasADi CPU throughput (track): avg_vel={avg_vel}, "
                  f"mode={mode}, cores={cores_list}, "
                  f"N_RUNS_jaxipm={N_RUNS_JAXIPM}")
            summaries = []
            for n_cores in cores_list:
                n_total = (args.n_total if args.n_total
                           else max(N_RUNS_JAXIPM,
                                    MIN_SOLVES_PER_WORKER * n_cores))
                out_path = os.path.join(
                    logs_dir,
                    f"casadi_cpu_throughput_v{avg_vel:.1f}_{mode}"
                    f"_c{n_cores:03d}_results.npz")
                if args.resume:
                    done = load_completed(out_path, n_cores, n_total)
                    if done is not None:
                        print(f"  [{mode} {n_cores:3d}] already complete "
                              f"({done['throughput']:.2f} solves/s) — "
                              f"skipping")
                        summaries.append(done)
                        continue
                pool_idx = np.arange(n_total) % N_RUNS_JAXIPM
                summaries.append(run_config(
                    task_module="tests.quad_track_avoid."
                                "quad_casadi_cpu_throughput",
                    task_kwargs=dict(avg_vel=float(avg_vel),
                                     n_pool=N_RUNS_JAXIPM),
                    mode=mode, n_cores=n_cores, n_total=n_total,
                    traj_shape=(N_HORIZON, 13),
                    out_path=out_path,
                    extra_npz=dict(
                        starts=all_x0[pool_idx],
                        all_x0=all_x0,
                        all_xr=all_xr,
                        all_s=all_s,
                        N=np.array([N_HORIZON]),
                        Ts=np.array([TS]),
                        avg_vel=np.array([avg_vel]),
                    ),
                    desc=f"track v{avg_vel:.1f}"))
            all_summaries[(avg_vel, mode)] = summaries

    print_summaries(all_summaries, "avg_vel")


if __name__ == "__main__":
    main()

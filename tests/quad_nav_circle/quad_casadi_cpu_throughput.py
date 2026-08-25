"""CasADi/IPOPT CPU-throughput benchmark for the quad nav-circle problem
(rebuttal experiment). Harness/protocol: tests/casadi_cpu_throughput_common.py
(P single-threaded IPOPT processes, phys/smt pinning sweep, work queue,
setup-excluded wall clock).

Initialization parity with jaxipm_quad_nav.py: the jaxipm run consumes a
deterministic np.linspace of N_RUNS_jaxipm start angles over each sector
(its calc_next_problem strides the angle index by N_batch mod N_RUNS, so
every angle is solved exactly once — no randomness). This test regenerates
the identical linspace with the same code and solves each angle exactly
once, so both solvers consume the same N_RUNS_jaxipm problems. Per-solve
protocol is identical to the sequential ipopt_quad_nav.py benchmark:
solve_cold(x0_new) with the linear-interpolation cold guess and the same
solver options; success = IPOPT converged.

Run from the repo root (defaults cover the full grid, no args needed):
    python -m tests.quad_nav_circle.quad_casadi_cpu_throughput
Optional overrides for smoke tests only:
    python -m tests.quad_nav_circle.quad_casadi_cpu_throughput \
        --sectors 90 --modes phys --cores 4 --n-total 16
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

with open("tests/quad_nav_circle/test_params.json") as f:
    tmp = json.load(f)
N_HORIZON = tmp["N_horizon"]
TS = tmp["Ts"]
N_RUNS_JAXIPM = tmp["N_RUNS_jaxipm"]
INIT_ANGLES = tmp["init_angles"]

# Matches jaxipm_quad_nav.py / ipopt_quad_nav.py: starts on the circle
# through the default x0=[4,4,0].
RADIUS = float(np.sqrt(4.0**2 + 4.0**2))


def sector_angles(sector_deg, n_runs):
    """The identical angle linspace jaxipm_quad_nav.py iterates over."""
    sector_half_rad = np.deg2rad(sector_deg) / 2.0
    sector_center_rad = np.pi / 4.0
    return np.linspace(sector_center_rad - sector_half_rad,
                       sector_center_rad + sector_half_rad, n_runs)


def x0_from_angle(theta):
    return np.array([
        RADIUS * np.cos(theta), RADIUS * np.sin(theta), 0.0,
        1.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0,
        0.0, 0.0, 0.0,
    ])


def make_worker(sector_deg, n_pool):
    """Harness contract: build the MPC (untimed) and return solve(idx)."""
    from tests.quad_nav_circle.ipopt_quad_nav import QuadcopterNavMPC

    angles = sector_angles(sector_deg, n_pool)
    mpc = QuadcopterNavMPC(x0_from_angle(angles[0]), N=N_HORIZON, Ts=TS)

    def solve(idx):
        x0_i = x0_from_angle(angles[idx % n_pool])
        try:
            sol = mpc.solve_cold(x0_new=x0_i)
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
        "Parallel-process CasADi/IPOPT CPU throughput sweep (nav circle). "
        "Zero-arg run covers the full grid; args are smoke-test overrides.",
        extra_args=[("--sectors", dict(
            type=str, default=None,
            help=f"comma list of sector degrees (default: {INIT_ANGLES})"))],
    )
    sectors = ([int(s) for s in args.sectors.split(",")]
               if args.sectors else INIT_ANGLES)
    logs_dir = os.path.join(os.path.dirname(__file__), "logs")

    all_summaries = {}
    for sector_deg in sectors:
        angles_pool = sector_angles(sector_deg, N_RUNS_JAXIPM)
        starts_pool = np.stack([x0_from_angle(a) for a in angles_pool])
        for mode in modes:
            cores_list = mode_cores[mode]
            print(f"\nCasADi CPU throughput (nav): sector={sector_deg}, "
                  f"mode={mode}, cores={cores_list}, "
                  f"N_RUNS_jaxipm={N_RUNS_JAXIPM}")
            summaries = []
            for n_cores in cores_list:
                n_total = (args.n_total if args.n_total
                           else max(N_RUNS_JAXIPM,
                                    MIN_SOLVES_PER_WORKER * n_cores))
                out_path = os.path.join(
                    logs_dir,
                    f"casadi_cpu_throughput_sector{sector_deg}_{mode}"
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
                    task_module="tests.quad_nav_circle."
                                "quad_casadi_cpu_throughput",
                    task_kwargs=dict(sector_deg=sector_deg,
                                     n_pool=N_RUNS_JAXIPM),
                    mode=mode, n_cores=n_cores, n_total=n_total,
                    traj_shape=(N_HORIZON, 13),
                    out_path=out_path,
                    extra_npz=dict(
                        starts=starts_pool[pool_idx],
                        angles=angles_pool[pool_idx],
                        all_angles=angles_pool,
                        N=np.array([N_HORIZON]),
                        Ts=np.array([TS]),
                        radius=np.array([RADIUS]),
                        sector_deg=np.array([sector_deg]),
                    ),
                    desc=f"nav s{sector_deg}"))
            all_summaries[(sector_deg, mode)] = summaries

    print_summaries(all_summaries, "sector")


if __name__ == "__main__":
    main()

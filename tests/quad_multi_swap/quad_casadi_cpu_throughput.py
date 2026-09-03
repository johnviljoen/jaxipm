"""CasADi/IPOPT CPU-throughput benchmark for the multi-quad rendezvous
problem (rebuttal experiment): how high does aggregate IPOPT throughput get
when P independent single-threaded instances run in parallel processes?

Two pinning modes are swept:
  - "phys": P in {1,2,4,...,64} workers, worker i pinned to logical CPU i.
    On this box logical CPUs 0-63 are the 64 distinct physical cores, so
    every worker owns a whole physical core.
  - "smt":  P in {2,4,...,128} workers packed pairwise onto SMT siblings —
    workers 2j and 2j+1 share physical core j (logical CPUs j and j+off,
    where off is the sibling offset, 64 here). P workers occupy only P/2
    physical cores, so phys-P vs smt-P isolates what hyperthreading buys.

Protocol (mirrors the jaxipm warm-throughput measurement):
  - For each P, a shared work queue holds
        N_total = max(N_RUNS_jaxipm, MIN_SOLVES_PER_WORKER * P)
    identical cold solves. Throughput is a rate (solves/sec), so scaling
    N_total with P keeps every worker busy for >= ~5 solves without
    changing what is measured. N_RUNS_jaxipm matches the jaxipm run's
    solution count (test_params.json).
  - Each solve is protocol-identical to ipopt_multi_swap.py's solve_cold():
    same Opti graph, same solver options (tol 1e-6, max_iter 500), same
    deterministic hover cold guess. The multi_swap initialization is
    deterministic on the jaxipm side too (calc_next_problem ignores its
    key), so both solvers consume identical problems.
  - Setup is excluded, like jaxipm's excluded JIT: each worker builds its
    QuadcopterMultiSwapMPC (whose constructor performs an untimed warmup
    solve), then waits on a start barrier. The parent stamps t0 at barrier
    release; wall = max(worker finish time) - t0.
  - One thread per instance: OMP/BLAS env vars are pinned to 1 before any
    OpenMP-linked library loads.
  - After the barrier everything is asynchronous: workers pull from the
    queue the moment they finish a solve, no further synchronization.

Results are saved per (N_quads, mode, P) in the same style as the existing
casadi_{N_quads}_results.npz (X_all/times/iters/obj_vals/success/starts/
goals/...) plus throughput extras (n_cores, throughput, worker_id, per-solve
t_start/t_end relative to t0, per-worker busy times, cpu_assignment).

Run from the repo root (defaults cover the full grid, no args needed):
    python -m tests.quad_multi_swap.quad_casadi_cpu_throughput
Optional overrides for smoke tests only:
    python -m tests.quad_multi_swap.quad_casadi_cpu_throughput \
        --n-quads 2 --modes phys --cores 1,8 --n-total 16
Interrupted sweeps can be continued with --resume (skips completed configs).
"""

import os

# Must be set before any OpenMP-linked library (casadi/IPOPT/MUMPS/BLAS)
# loads. spawn'd children re-import this module, so they inherit these too.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse
import json
import socket
import sys
from tests.casadi_cpu_throughput_common import (
    IPOPT_STATUS, ipopt_extras, out_suffix, _git_head)
import contextlib
import io
import json
import multiprocessing as mp
import time
import traceback

import numpy as np

PHYS_CORES_DEFAULT = [1, 2, 4, 8, 16, 32, 64]
SMT_CORES_DEFAULT = [2, 4, 8, 16, 32, 64, 128]
MODES_DEFAULT = ["phys", "smt"]
MIN_SOLVES_PER_WORKER = 5
MAX_ITER = 500              # matches ipopt_multi_swap.py solver opts
BARRIER_TIMEOUT_S = 1800    # generous: 128 procs each build + warmup-solve


def _smt_offset():
    """Logical-CPU offset between the two SMT siblings of a physical core
    (cpu0's sibling list is e.g. '0,64' -> 64). Falls back to n_logical/2,
    which is the standard Linux enumeration."""
    try:
        with open("/sys/devices/system/cpu/cpu0/topology/"
                  "thread_siblings_list") as f:
            parts = f.read().strip().replace("-", ",").split(",")
        if len(parts) > 1:
            return int(parts[1])
    except Exception:
        pass
    return (os.cpu_count() or 2) // 2


def _cpu_assignment(mode, n_cores):
    """Logical CPU for each worker id.

    phys: worker i -> CPU i (CPUs 0..63 are distinct physical cores).
    smt:  workers 2j, 2j+1 -> siblings (j, j+off) of physical core j.
    """
    if mode == "phys":
        return [i for i in range(n_cores)]
    off = _smt_offset()
    return [(i // 2) + off * (i % 2) for i in range(n_cores)]


def _worker(worker_id, cpu_id, n_quads, task_q, result_q, barrier):
    """One single-threaded IPOPT instance pinned to logical CPU cpu_id."""
    try:
        os.sched_setaffinity(0, {cpu_id})

        from tests.quad_multi_swap.ipopt_multi_swap import (
            QuadcopterMultiSwapMPC, R_DEFAULT, check_success, swap_pool, tmp)

        # Varied-IC pool (2026-08-31): task idx -> formation rotation
        # phi_(idx % n_pool), the same pool every solver consumes.
        n_pool = tmp["N_RUNS_jaxipm"]
        _, _, pool_starts, pool_goals = swap_pool(n_quads, n_pool, R_DEFAULT)

        # Constructor builds the Opti graph and performs the untimed warmup
        # solve; silence its "Initial solve time" print (P of them is noise).
        with contextlib.redirect_stdout(io.StringIO()):
            mpc = QuadcopterMultiSwapMPC(N_quads=n_quads)
    except Exception:
        barrier.abort()
        result_q.put(("error", worker_id, traceback.format_exc()))
        return

    try:
        barrier.wait(timeout=BARRIER_TIMEOUT_S)
    except Exception:
        result_q.put(("error", worker_id, "barrier broken/timeout"))
        return

    busy = 0.0
    while True:
        idx = task_q.get()
        if idx is None:
            break

        t1 = time.time()
        converged = False
        iters = -1
        obj = np.nan
        X = None
        extras = {}
        try:
            sol = mpc.solve_cold(starts=pool_starts[idx % n_pool],
                                 goals=pool_goals[idx % n_pool])
            t2 = time.time()
            X = np.stack([x.T for x in mpc.x_sols])       # (n_quads, N, 13)
            extras = ipopt_extras(mpc.opti, sol)
            try:
                extras["U"] = np.stack([np.asarray(u).T for u in mpc.u_sols])  # (n_quads, N-1, 4)
            except Exception:
                pass
            try:
                iters = int(sol.stats().get("iter_count", -1))
            except Exception:
                iters = -1
            try:
                obj = float(sol.value(mpc.opti.f))
            except Exception:
                pass
            converged = True
        except RuntimeError:
            t2 = time.time()
            extras = ipopt_extras(mpc.opti, None)
            try:
                X = np.stack([np.asarray(mpc.opti.debug.value(Xv)).T
                              for Xv in mpc.Xs])
                iters = int(mpc.opti.stats().get("iter_count", -1))
            except Exception:
                iters = -1
            try:
                obj = float(mpc.opti.debug.value(mpc.opti.f))
            except Exception:
                pass
        if X is None:
            X = np.full((n_quads, mpc.N, 13), np.nan)

        succ = False
        if converged:
            succ = check_success(X, mpc.goals, iters, max_iter=MAX_ITER)

        busy += t2 - t1
        result_q.put(("solve", idx, worker_id, t1, t2, iters, obj,
                      bool(succ), X, {k: np.asarray(v) for k, v in extras.items()}))

    result_q.put(("done", worker_id, time.time(), busy))


def _out_path(n_quads, mode, n_cores):
    logs_dir = os.path.join(os.path.dirname(__file__), "logs")
    return os.path.join(
        logs_dir,
        f"casadi_cpu_throughput_{n_quads}_{mode}_c{n_cores:03d}{out_suffix()}_results.npz")


def _load_completed(n_quads, mode, n_cores, n_total):
    """Return the summary dict for an already-completed config, or None.

    A saved npz only counts as complete if its solve count matches the
    expected n_total (guards against stale smoke-test files with --n-total
    overrides) — otherwise the config is rerun and the file overwritten.
    """
    path = _out_path(n_quads, mode, n_cores)
    if not os.path.exists(path):
        return None
    try:
        z = np.load(path)
        if int(z["N_RUNS"][0]) != n_total or int(z["n_cores"][0]) != n_cores:
            return None
        return dict(n_cores=n_cores, n_total=n_total,
                    wall=float(z["total_time"][0]),
                    throughput=float(z["throughput"][0]),
                    success_frac=float(z["success"].mean()),
                    mean_solve_ms=float(z["times"].mean() * 1000))
    except Exception:
        return None


def run_config(n_quads, mode, n_cores, n_total):
    """One (N_quads, mode, P) measurement. Returns the summary dict; saves
    the npz."""
    from tests.quad_multi_swap.ipopt_multi_swap import (
        N_HORIZON, TS, R_DEFAULT, start_state, goal_xyz)
    from tqdm import tqdm

    cpu_ids = _cpu_assignment(mode, n_cores)

    ctx = mp.get_context("spawn")
    task_q = ctx.Queue()
    result_q = ctx.Queue()
    barrier = ctx.Barrier(n_cores + 1)   # workers + parent (stamps t0)

    for i in range(n_total):
        task_q.put(i)
    for _ in range(n_cores):
        task_q.put(None)                 # one shutdown sentinel per worker

    procs = [ctx.Process(target=_worker,
                         args=(i, cpu_ids[i], n_quads, task_q, result_q,
                               barrier),
                         daemon=True)
             for i in range(n_cores)]
    for p in procs:
        p.start()

    print(f"  [{mode} {n_cores:3d}] building {n_cores} IPOPT instances "
          f"(untimed setup + warmup solve)...")
    try:
        barrier.wait(timeout=BARRIER_TIMEOUT_S)
    except Exception:
        for p in procs:
            p.terminate()
        # Surface any worker tracebacks that made it into the queue.
        while not result_q.empty():
            msg = result_q.get_nowait()
            if msg[0] == "error":
                print(f"worker {msg[1]} failed:\n{msg[2]}")
        raise RuntimeError(f"setup failed/timed out at {mode} P={n_cores}")
    t0 = time.time()

    X_all = np.full((n_total, n_quads, N_HORIZON, 13), np.nan)
    times = np.zeros(n_total)
    t_start = np.zeros(n_total)
    t_end = np.zeros(n_total)
    iters = np.full(n_total, -1, dtype=np.int32)
    obj_vals = np.full(n_total, np.nan)
    success = np.zeros(n_total, dtype=bool)
    worker_id = np.full(n_total, -1, dtype=np.int32)
    busy_times = np.zeros(n_cores)
    finish_times = np.zeros(n_cores)

    extra_arrays = {}
    n_solved = 0
    n_done = 0
    pbar = tqdm(total=n_total, desc=f"{mode} P={n_cores}", unit="solve")
    while n_done < n_cores:
        msg = result_q.get()
        if msg[0] == "solve":
            _, idx, wid, t1, t2, it, obj, succ, X, extras = msg
            for k, v in extras.items():
                if k not in extra_arrays:
                    fill = -1 if v.dtype.kind in "iu" else np.nan
                    extra_arrays[k] = np.full((n_total,) + v.shape, fill,
                                              dtype=v.dtype if v.dtype.kind in "iu" else np.float64)
                extra_arrays[k][idx] = v
            X_all[idx] = X
            times[idx] = t2 - t1
            t_start[idx] = t1 - t0
            t_end[idx] = t2 - t0
            iters[idx] = it
            obj_vals[idx] = obj
            success[idx] = succ
            worker_id[idx] = wid
            n_solved += 1
            pbar.update(1)
            if n_solved % 10 == 0:
                pbar.set_postfix(succ=f"{success[worker_id >= 0].mean():.3f}")
        elif msg[0] == "done":
            _, wid, t_fin, busy = msg
            finish_times[wid] = t_fin
            busy_times[wid] = busy
            n_done += 1
        elif msg[0] == "error":
            pbar.close()
            for p in procs:
                p.terminate()
            raise RuntimeError(f"worker {msg[1]} failed:\n{msg[2]}")
    pbar.close()

    wall = finish_times.max() - t0
    for p in procs:
        p.join()

    throughput = n_total / wall
    starts_arr = np.array([start_state(i, n_quads, R_DEFAULT)
                           for i in range(n_quads)])
    goals_arr = np.array([goal_xyz(i, n_quads, R_DEFAULT)
                          for i in range(n_quads)])
    from tests.quad_multi_swap.ipopt_multi_swap import swap_pool
    with open("tests/quad_multi_swap/test_params.json") as _f:
        _n_pool = json.load(_f)["N_RUNS_jaxipm"]
    pool_phis, pool_psis, pool_starts, pool_goals = swap_pool(n_quads, _n_pool, R_DEFAULT)

    out_path = _out_path(n_quads, mode, n_cores)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez(
        out_path,
        X_all=X_all,
        times=times,
        t_start=t_start,
        t_end=t_end,
        iters=iters,
        obj_vals=obj_vals,
        success=success,
        worker_id=worker_id,
        worker_busy_times=busy_times,
        cpu_assignment=np.array(cpu_ids, dtype=np.int32),
        mode=np.array([mode]),
        total_time=np.array([wall]),
        throughput=np.array([throughput]),
        n_cores=np.array([n_cores]),
        N=np.array([N_HORIZON]),
        Ts=np.array([TS]),
        N_RUNS=np.array([n_total]),
        ipopt_tol=np.array([float(os.environ.get("IPOPT_TOL", "1e-6"))]),
        metadata=np.array([json.dumps(dict(
            loadavg_at_save=list(os.getloadavg()),
            ipopt_tol=float(os.environ.get("IPOPT_TOL", "1e-6")), mode=mode, n_cores=n_cores,
            n_total=n_total, hostname=socket.gethostname(), commit=_git_head(), argv=sys.argv,
            ipopt_status_legend=IPOPT_STATUS))]),
        **{("U_all" if k == "U" else k): v for k, v in extra_arrays.items()},
        N_quads=np.array([n_quads]),
        R=np.array([R_DEFAULT]),
        starts=starts_arr,
        goals=goals_arr,
        pool_phis=pool_phis,
        pool_psis=pool_psis,
        pool_starts=pool_starts,
        pool_goals=pool_goals,
        inst_idx=np.arange(n_total) % _n_pool,
    )
    print(f"  [{mode} {n_cores:3d}] saved {out_path}")
    print(f"  [{mode} {n_cores:3d}] wall={wall:.2f}s  "
          f"throughput={throughput:.2f} solves/s  "
          f"success={int(success.sum())}/{n_total}  "
          f"mean_solve={times.mean()*1000:.1f}ms  "
          f"mean_busy_frac={(busy_times / wall).mean():.2f}")

    return dict(n_cores=n_cores, n_total=n_total, wall=wall,
                throughput=throughput,
                success_frac=success.mean(),
                mean_solve_ms=times.mean() * 1000)


def main():
    ap = argparse.ArgumentParser(
        description="Parallel-process CasADi/IPOPT CPU throughput sweep "
                    "(multi-quad swap). Zero-arg run covers the full grid; "
                    "args are smoke-test overrides only.")
    ap.add_argument("--n-quads", type=str, default=None,
                    help="comma list, e.g. '2' (default: test_params N_quads)")
    ap.add_argument("--modes", type=str, default=None,
                    help=f"comma list from {{phys,smt}} "
                         f"(default: {','.join(MODES_DEFAULT)})")
    ap.add_argument("--cores", type=str, default=None,
                    help="comma list, applied to every mode (defaults: "
                         f"phys {PHYS_CORES_DEFAULT}, smt {SMT_CORES_DEFAULT})")
    ap.add_argument("--n-total", type=int, default=None,
                    help="override solve count per config (default: "
                         "max(N_RUNS_jaxipm, 5*P))")
    ap.add_argument("--tol", type=float, default=None,
                    help="IPOPT tol override (IPOPT_TOL for workers; output suffix _tol<v>)")
    ap.add_argument("--resume", action="store_true",
                    help="skip configs whose completed npz already exists "
                         "(default: run everything fresh, overwriting)")
    args = ap.parse_args()
    if args.tol is not None:
        os.environ["IPOPT_TOL"] = repr(float(args.tol))
        os.environ.setdefault("CPU_SWEEP_SUFFIX", f"_tol{args.tol:g}")

    with open("tests/quad_multi_swap/test_params.json") as f:
        params = json.load(f)

    n_quads_list = ([int(s) for s in args.n_quads.split(",")]
                    if args.n_quads else params["N_quads"])
    modes = args.modes.split(",") if args.modes else MODES_DEFAULT
    for m in modes:
        assert m in ("phys", "smt"), f"unknown mode {m!r}"

    n_logical = os.cpu_count() or 1
    n_physical = n_logical // 2         # SMT2 box (64 phys / 128 logical)
    mode_cores = {}
    for mode in modes:
        cap = n_physical if mode == "phys" else n_logical
        cores = ([int(s) for s in args.cores.split(",")]
                 if args.cores else
                 (PHYS_CORES_DEFAULT if mode == "phys" else SMT_CORES_DEFAULT))
        mode_cores[mode] = [c for c in cores if c <= cap] or [cap]

    n_runs_jaxipm = params["N_RUNS_jaxipm"]

    all_summaries = {}
    for n_quads in n_quads_list:
        for mode in modes:
            cores_list = mode_cores[mode]
            print(f"\nCasADi CPU throughput: N_quads={n_quads}, mode={mode}, "
                  f"cores={cores_list}, N_RUNS_jaxipm={n_runs_jaxipm}")
            summaries = []
            for n_cores in cores_list:
                n_total = (args.n_total if args.n_total
                           else max(n_runs_jaxipm,
                                    MIN_SOLVES_PER_WORKER * n_cores))
                done = (_load_completed(n_quads, mode, n_cores, n_total)
                        if args.resume else None)
                if done is not None:
                    print(f"  [{mode} {n_cores:3d}] already complete "
                          f"({done['throughput']:.2f} solves/s) — skipping")
                    summaries.append(done)
                    continue
                summaries.append(run_config(n_quads, mode, n_cores, n_total))
            all_summaries[(n_quads, mode)] = summaries

    for (n_quads, mode), summaries in all_summaries.items():
        # Speedup baseline: the phys P=1 run of the same N_quads when
        # available (so smt numbers are comparable), else the mode's first.
        base_summ = all_summaries.get((n_quads, "phys"), summaries)[0]
        base = base_summ["throughput"] / base_summ["n_cores"]
        print(f"\n===== N_quads={n_quads} mode={mode} summary =====")
        print(f"{'cores':>6} {'solves':>7} {'wall[s]':>9} "
              f"{'solves/s':>9} {'speedup':>8} {'succ':>6} {'ms/solve':>9}")
        for s in summaries:
            print(f"{s['n_cores']:>6d} {s['n_total']:>7d} {s['wall']:>9.2f} "
                  f"{s['throughput']:>9.2f} {s['throughput']/base:>8.2f} "
                  f"{s['success_frac']:>6.3f} {s['mean_solve_ms']:>9.1f}")


if __name__ == "__main__":
    main()

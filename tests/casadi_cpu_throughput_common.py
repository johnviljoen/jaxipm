"""
Rebuttal - how far can we push CPU throughput by MPI'ing across all of our servers
cores (both SMT and physical).
"""

import os

# Must be set before any OpenMP-linked library (casadi/IPOPT/MUMPS/BLAS)
# loads — both in the parent and in spawn'd children.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse
import contextlib
import importlib
import io
import multiprocessing as mp
import time
import traceback

import numpy as np

PHYS_CORES_DEFAULT = [1, 2, 4, 8, 16, 32, 64]
SMT_CORES_DEFAULT = [2, 4, 8, 16, 32, 64, 128]
MODES_DEFAULT = ["phys", "smt"]
MIN_SOLVES_PER_WORKER = 5
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


def cpu_assignment(mode, n_cores):
    """Logical CPU for each worker id.

    phys: worker i -> CPU i (CPUs 0..63 are distinct physical cores).
    smt:  workers 2j, 2j+1 -> siblings (j, j+off) of physical core j.
    """
    if mode == "phys":
        return [i for i in range(n_cores)]
    off = _smt_offset()
    return [(i // 2) + off * (i % 2) for i in range(n_cores)]


def _worker(worker_id, cpu_id, task_module, task_kwargs, task_q, result_q,
            barrier):
    """One single-threaded IPOPT instance pinned to logical CPU cpu_id."""
    try:
        os.sched_setaffinity(0, {cpu_id})
        mod = importlib.import_module(task_module)
        # Silence the MPC constructor's "Initial solve time" print (P of
        # them is noise); IPOPT itself is already at print_level 0.
        with contextlib.redirect_stdout(io.StringIO()):
            solve = mod.make_worker(**task_kwargs)
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
        try:
            X, iters, obj, success = solve(idx)
        except Exception:
            result_q.put(("error", worker_id, traceback.format_exc()))
            return
        t2 = time.time()
        busy += t2 - t1
        result_q.put(("solve", idx, worker_id, t1, t2, int(iters),
                      float(obj), bool(success), np.asarray(X)))

    result_q.put(("done", worker_id, time.time(), busy))


def load_completed(out_path, n_cores, n_total):
    """Return the summary dict for an already-completed config, or None.

    A saved npz only counts as complete if its solve count matches the
    expected n_total (guards against stale smoke-test files with --n-total
    overrides) — otherwise the config is rerun and the file overwritten.
    """
    if not os.path.exists(out_path):
        return None
    try:
        z = np.load(out_path)
        if int(z["N_RUNS"][0]) != n_total or int(z["n_cores"][0]) != n_cores:
            return None
        return dict(n_cores=n_cores, n_total=n_total,
                    wall=float(z["total_time"][0]),
                    throughput=float(z["throughput"][0]),
                    success_frac=float(z["success"].mean()),
                    mean_solve_ms=float(z["times"].mean() * 1000))
    except Exception:
        return None


def run_config(*, task_module, task_kwargs, mode, n_cores, n_total,
               traj_shape, out_path, extra_npz=None, desc=""):
    """One (task, mode, P) measurement. Returns the summary dict; saves the
    npz (standard keys + whatever the task passes via extra_npz)."""
    from tqdm import tqdm

    cpu_ids = cpu_assignment(mode, n_cores)

    ctx = mp.get_context("spawn")
    task_q = ctx.Queue()
    result_q = ctx.Queue()
    barrier = ctx.Barrier(n_cores + 1)   # workers + parent (stamps t0)

    for i in range(n_total):
        task_q.put(i)
    for _ in range(n_cores):
        task_q.put(None)                 # one shutdown sentinel per worker

    procs = [ctx.Process(target=_worker,
                         args=(i, cpu_ids[i], task_module, task_kwargs,
                               task_q, result_q, barrier),
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

    X_all = np.full((n_total,) + tuple(traj_shape), np.nan)
    times = np.zeros(n_total)
    t_start = np.zeros(n_total)
    t_end = np.zeros(n_total)
    iters = np.full(n_total, -1, dtype=np.int32)
    obj_vals = np.full(n_total, np.nan)
    success = np.zeros(n_total, dtype=bool)
    worker_id = np.full(n_total, -1, dtype=np.int32)
    busy_times = np.zeros(n_cores)
    finish_times = np.zeros(n_cores)

    n_solved = 0
    n_done = 0
    pbar = tqdm(total=n_total, desc=f"{desc} {mode} P={n_cores}",
                unit="solve")
    while n_done < n_cores:
        msg = result_q.get()
        if msg[0] == "solve":
            _, idx, wid, t1, t2, it, obj, succ, X = msg
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
            if n_solved % 25 == 0:
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
        N_RUNS=np.array([n_total]),
        **(extra_npz or {}),
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


def parse_sweep_args(description, extra_args=()):
    """Common CLI: --modes/--cores/--n-total/--resume plus per-task extras.
    Returns (args, modes, mode_cores). Zero-arg run covers the full grid."""
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--modes", type=str, default=None,
                    help=f"comma list from {{phys,smt}} "
                         f"(default: {','.join(MODES_DEFAULT)})")
    ap.add_argument("--cores", type=str, default=None,
                    help="comma list, applied to every mode (defaults: "
                         f"phys {PHYS_CORES_DEFAULT}, smt {SMT_CORES_DEFAULT})")
    ap.add_argument("--n-total", type=int, default=None,
                    help="override solve count per config (default: "
                         "max(N_RUNS_jaxipm, 5*P))")
    ap.add_argument("--resume", action="store_true",
                    help="skip configs whose completed npz already exists "
                         "(default: run everything fresh, overwriting)")
    for name, kwargs in extra_args:
        ap.add_argument(name, **kwargs)
    args = ap.parse_args()

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

    return args, modes, mode_cores


def print_summaries(all_summaries, variant_name):
    """Per-(variant, mode) tables. Speedup baseline: the phys P=1 per-core
    throughput of the same variant when available (so smt numbers are
    comparable), else the mode's own first row."""
    for (variant, mode), summaries in all_summaries.items():
        base_summ = all_summaries.get((variant, "phys"), summaries)[0]
        base = base_summ["throughput"] / base_summ["n_cores"]
        print(f"\n===== {variant_name}={variant} mode={mode} summary =====")
        print(f"{'cores':>6} {'solves':>7} {'wall[s]':>9} "
              f"{'solves/s':>9} {'speedup':>8} {'succ':>6} {'ms/solve':>9}")
        for s in summaries:
            print(f"{s['n_cores']:>6d} {s['n_total']:>7d} {s['wall']:>9.2f} "
                  f"{s['throughput']:>9.2f} {s['throughput']/base:>8.2f} "
                  f"{s['success_frac']:>6.3f} {s['mean_solve_ms']:>9.1f}")

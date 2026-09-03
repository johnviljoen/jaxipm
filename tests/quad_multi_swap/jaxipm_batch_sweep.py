"""GPU batch-size saturation sweep for jaxipm on the multi-swap problem
(rebuttal experiment): warm throughput (solves/s) vs N_batch on one GPU.

The paper's multi-swap numbers (8.66 / 1.48 solves/s) were measured at
N_batch = 75, chosen to equal N_RUNS — not to saturate the GPU. This sweep
measures where throughput actually saturates. Setup replicates
jaxipm_multi_swap.py exactly (same problem build, same deterministic hover
initialization via calc_next_problem, same warm-measurement protocol:
untimed JIT call, then a timed call); only N_batch varies, with
max_solves = max(N_RUNS_jaxipm, 3 * N_batch) so every slot turns over
several times per measurement.

OOM at a given batch size is caught and ends that variant's sweep (larger
batches would also OOM).

NOTE: run this only while the CPU throughput sweeps are idle — JIT
compilation is CPU-heavy and would pollute their measurement.

Run from the repo root:
    python -m tests.quad_multi_swap.jaxipm_batch_sweep
    python -m tests.quad_multi_swap.jaxipm_batch_sweep --n-quads 2 --batches 75,300
"""

import argparse
import json
import os
from time import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np

# Import the problem module before touching jax config, mirroring the
# original script's execution order (its module top-level runs before the
# __main__ block enables x64).
from tests.quad_multi_swap.jaxipm_multi_swap import (
    N_HORIZON, R, TS, quadcopter_multi_swap, tmp)
from tests.quad_multi_swap.ipopt_multi_swap import check_success
from tests.jaxipm_variance_common import (
    stamp, timed_repeats, variance_summary, save_variance_npz)

BATCHES_DEFAULT = [25, 75, 150, 300, 600]
MAX_SOLVES_MULT = 3


def main():
    ap = argparse.ArgumentParser(
        description="jaxipm GPU batch-size saturation sweep (multi-swap).")
    ap.add_argument("--n-quads", type=str, default=None,
                    help=f"comma list (default: {tmp['N_quads']})")
    ap.add_argument("--batches", type=str, default=None,
                    help=f"comma list (default: {BATCHES_DEFAULT})")
    ap.add_argument("--repeats", type=int, default=1,
                    help="timed repeats per batch config (JIT paid once in "
                         "the warmup call); >1 records per-repeat "
                         "start/stop stamps and saves jaxipm_variance_* "
                         "instead of the sweep npz")
    ap.add_argument("--out-suffix", type=str, default="",
                    help="also save the jaxipm_variance_* npz (with this "
                         "suffix in the name) even when --repeats 1; used "
                         "for fresh-JIT-per-repeat campaigns")
    args = ap.parse_args()
    n_quads_list = ([int(s) for s in args.n_quads.split(",")]
                    if args.n_quads else tmp["N_quads"])
    batches = ([int(s) for s in args.batches.split(",")]
               if args.batches else BATCHES_DEFAULT)
    n_runs_jaxipm = tmp["N_RUNS_jaxipm"]

    with open("jaxipm/params.json") as f:
        p = json.load(f)
    p["DEBUG_MODE"] = False  # no iter_buffer overhead in throughput numbers

    import jax
    import jax.numpy as jnp
    import equinox as eqx

    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_compilation_cache_dir", "/home/john/tmp/jax_cache")
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)

    from jaxipm.solver import solve_throughput, make_batch_state
    from jaxipm.initialization import (
        initialize_common_problem, initialize_problem_regular,
    )

    logs_dir = os.path.join(os.path.dirname(__file__), "logs")
    os.makedirs(logs_dir, exist_ok=True)

    for N_quads in n_quads_list:
        print(f"\n===== batch sweep: multi-swap N_quads={N_quads}, "
              f"batches={batches} =====")
        f, c, d, x_L, x_U, d_L, d_U, x0, gt, aux = quadcopter_multi_swap(
            N_quads=N_quads, N=N_HORIZON, R=R)
        z_to_xu = aux[0]
        goals = aux[7]

        jax.clear_caches()
        f_args = (); c_args = (); d_args = ()
        _user_args = (
            tuple(jnp.asarray(a) for a in f_args),
            tuple(jnp.asarray(a) for a in c_args),
            tuple(jnp.asarray(a) for a in d_args),
        )
        _x0_flat = x0.squeeze()
        calc_next_problem = lambda key, sol: (_x0_flat, *_user_args)

        cp = initialize_common_problem(
            f, c, d, x_L, x_U, d_L, d_U, x0, p,
            [f_args, c_args, d_args],
            calc_next_problem=calc_next_problem)
        state = initialize_problem_regular(cp, x0, args=[f_args, c_args, d_args])
        state = eqx.tree_at(lambda t: t.fl.needs_regular_init, state,
                            jnp.array([[0]]))

        max_iter_per_solve = cp.p["max_iter"]
        rng_key = jax.random.PRNGKey(0)
        _solve_throughput = eqx.filter_jit(solve_throughput, donate="none")

        rows = []
        for N_batch in batches:
            max_solves = max(n_runs_jaxipm, MAX_SOLVES_MULT * N_batch)
            print(f"\n--- N_batch={N_batch} (max_solves={max_solves}) ---")
            try:
                batch_tp = make_batch_state(cp, [state] * N_batch)

                label = f"multi_q{N_quads}_b{N_batch}"

                def run_once():
                    out = _solve_throughput(
                        rng_key, cp, batch_tp, max_solves,
                        max_iter_per_solve=max_iter_per_solve)
                    jax.block_until_ready(out[0])
                    return out

                def n_collected_of(out):
                    return min(int(out[2]), int(out[1].shape[0]))

                wu0 = time()
                stamp(label, "WARMUP START", wu0)
                run_once()
                wu1 = time()
                stamp(label, "WARMUP STOP ", wu1)
                warmup_s = wu1 - wu0
                print(f"warmup (incl. JIT): {warmup_s*1000:.1f} ms")

                rep_rows = timed_repeats(label, args.repeats, run_once,
                                         n_collected_of)
                tp_out = rep_rows[-1]["tp_out"]
                wall = rep_rows[-1]["wall"]
            except Exception as e:
                print(f"N_batch={N_batch} FAILED ({type(e).__name__}): "
                      f"{str(e)[:300]}")
                print("stopping this variant's sweep (larger batches would "
                      "also fail)")
                break

            if len(tp_out) >= 5:
                final_state, solution_buffer, write_idx, _, iter_buffer = tp_out[:5]
            else:
                final_state, solution_buffer, write_idx = tp_out
                iter_buffer = None
            n_collected = min(int(write_idx), int(solution_buffer.shape[0]))
            throughput = n_collected / wall

            # Success check on the collected solutions (post-timing).
            sol_buf_np = np.asarray(solution_buffer[:n_collected, :cp.nx])
            if iter_buffer is not None:
                iters = np.asarray(iter_buffer)[:n_collected].astype(np.int32)
            else:
                iters = np.full(n_collected, -1, dtype=np.int32)
            success = np.zeros(n_collected, dtype=bool)
            for i in range(n_collected):
                xs_i, _ = z_to_xu(sol_buf_np[i])
                X_i = np.stack([np.asarray(x) for x in xs_i])
                success[i] = check_success(X_i, goals, int(iters[i]),
                                           max_iter=max_iter_per_solve)

            print(f"N_batch={N_batch}: wall={wall:.2f}s  "
                  f"collected={n_collected}/{max_solves}  "
                  f"throughput={throughput:.2f} solves/s  "
                  f"success={int(success.sum())}/{n_collected}")
            rows.append(dict(N_batch=N_batch, max_solves=max_solves,
                             wall=wall, warmup_s=warmup_s,
                             n_collected=n_collected, throughput=throughput,
                             success_frac=float(success.mean())
                             if n_collected else 0.0,
                             warmup_t0=wu0, warmup_t1=wu1,
                             rep_rows=rep_rows))

        if not rows:
            print(f"N_quads={N_quads}: no successful batch configs")
            continue

        if args.repeats > 1 or args.out_suffix:
            suffix = f"_{args.out_suffix}" if args.out_suffix else ""
            out_path = os.path.join(
                logs_dir,
                f"jaxipm_variance_{N_quads}{suffix}_results.npz")
            save_variance_npz(
                out_path, rows, N_quads=N_quads, N=N_HORIZON, Ts=TS, R=R,
                success_frac=[r["success_frac"] for r in rows])
            for r in rows:
                variance_summary(f"multi_q{N_quads}_b{r['N_batch']}",
                                 r["rep_rows"])
            continue

        out_path = os.path.join(
            logs_dir, f"jaxipm_batch_sweep_{N_quads}_results.npz")
        np.savez(
            out_path,
            batch_sizes=np.array([r["N_batch"] for r in rows]),
            throughput=np.array([r["throughput"] for r in rows]),
            wall=np.array([r["wall"] for r in rows]),
            warmup_s=np.array([r["warmup_s"] for r in rows]),
            n_collected=np.array([r["n_collected"] for r in rows]),
            max_solves=np.array([r["max_solves"] for r in rows]),
            success_frac=np.array([r["success_frac"] for r in rows]),
            N_quads=np.array([N_quads]),
            N=np.array([N_HORIZON]),
            Ts=np.array([TS]),
            R=np.array([R]),
        )
        print(f"\nsaved {out_path}")
        print(f"===== N_quads={N_quads} batch-sweep summary =====")
        print(f"{'batch':>6} {'solves':>7} {'wall[s]':>9} {'solves/s':>9} "
              f"{'succ':>6}")
        for r in rows:
            print(f"{r['N_batch']:>6d} {r['n_collected']:>7d} "
                  f"{r['wall']:>9.2f} {r['throughput']:>9.2f} "
                  f"{r['success_frac']:>6.3f}")


if __name__ == "__main__":
    main()

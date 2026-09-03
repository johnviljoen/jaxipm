"""GPU batch-size saturation sweep for jaxipm on the track-avoid problem
(rebuttal experiment): warm throughput (solves/s) vs N_batch on one GPU.

The paper's track numbers (32-37 solves/s) were measured at N_batch = 250.
This sweep measures where throughput actually saturates. Setup replicates
jaxipm_track_avoid.py exactly (same generate_random_inits(seed=0) pool,
same per-slot (x0, xr) and warm-start injection, same random-index
calc_next_problem, same warm-measurement protocol); only N_batch varies,
with max_solves = max(N_RUNS_jaxipm, 3 * N_batch). N_batch is capped at
the pool size (first-batch injection consumes pool[:N_batch]).

OOM at a given batch size is caught and ends that variant's sweep.

NOTE: run this only while the CPU throughput sweeps are idle — JIT
compilation is CPU-heavy and would pollute their measurement.

Run from the repo root:
    python -m tests.quad_track_avoid.jaxipm_batch_sweep
    python -m tests.quad_track_avoid.jaxipm_batch_sweep --vels 1.5 --batches 250,1000
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
from tests.quad_track_avoid.jaxipm_track_avoid import (
    N_HORIZON, TS, quadcopter_track_avoid, tmp)
from tests.jaxipm_variance_common import (
    stamp, timed_repeats, variance_summary, save_variance_npz)
from tests.quad_track_avoid.initialization import generate_random_inits

BATCHES_DEFAULT = [125, 250, 500, 1000]
MAX_SOLVES_MULT = 3
DELTA_STD = 0.5   # matches jaxipm_track_avoid.py


def main():
    ap = argparse.ArgumentParser(
        description="jaxipm GPU batch-size saturation sweep (track avoid).")
    ap.add_argument("--vels", type=str, default=None,
                    help=f"comma list (default: {tmp['avg_vel']})")
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
    vels = ([float(s) for s in args.vels.split(",")]
            if args.vels else tmp["avg_vel"])
    batches = ([int(s) for s in args.batches.split(",")]
               if args.batches else BATCHES_DEFAULT)
    N_RUNS = tmp["N_RUNS_jaxipm"]
    batches = [b for b in batches if b <= N_RUNS] or [N_RUNS]

    with open("jaxipm/params.json") as f:
        p = json.load(f)
    p["DEBUG_MODE"] = False  # no iter_buffer overhead in throughput numbers

    import jax
    import jax.numpy as jnp
    import equinox as eqx

    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
    jax.config.update("jax_persistent_cache_enable_xla_caches",
                      "xla_gpu_per_fusion_autotune_cache_dir")

    from jaxipm.solver import solve_throughput, make_batch_state
    from jaxipm.initialization import (
        initialize_common_problem, initialize_problem_regular,
    )

    N_horizon = N_HORIZON
    Ts_ = TS
    w_hover = 522.9847140714692

    def build_warm_start(x0_ic, xr_ref):
        del xr_ref
        end_xyz = jnp.array([0., 0., -2.])
        xyz = jnp.linspace(x0_ic[:3], end_xyz, N_horizon)
        ang_v_angv = jnp.tile(jnp.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
                              (N_horizon, 1))
        states = jnp.hstack([xyz, ang_v_angv])
        inputs = jnp.full((N_horizon - 1, 4), w_hover)
        return jnp.concatenate([states.flatten(), inputs.flatten()])

    logs_dir = os.path.join(os.path.dirname(__file__), "logs")
    os.makedirs(logs_dir, exist_ok=True)

    for AVG_VEL in vels:
        print(f"\n===== batch sweep: track avg_vel={AVG_VEL}, "
              f"batches={batches} =====")
        f, c, d, x_L, x_U, d_L, d_U, z_init, gt, aux = \
            quadcopter_track_avoid(N=N_horizon, avg_vel=AVG_VEL)
        xr_default = aux[3]
        x0_default = aux[4]

        all_x0_np, all_xr_np, all_s_np, _ = generate_random_inits(
            N_RUNS, N_horizon, Ts_, AVG_VEL, delta_std=DELTA_STD, seed=0)
        all_x0 = jnp.asarray(all_x0_np)
        all_xr = jnp.asarray(all_xr_np)

        f_args = (xr_default,)
        c_args = (x0_default,)
        d_args = ()

        def calc_next_problem(rng_key, sol):
            new_idx = jax.random.randint(rng_key, (), 0, N_RUNS)
            new_x0 = all_x0[new_idx]
            new_xr = all_xr[new_idx]
            new_warm = build_warm_start(new_x0, new_xr)
            return (new_warm, (new_xr,), (new_x0,), ())

        cp = initialize_common_problem(f, c, d, x_L, x_U, d_L, d_U, z_init, p,
                                       [f_args, c_args, d_args],
                                       calc_next_problem=calc_next_problem)
        state = initialize_problem_regular(cp, z_init,
                                           args=[f_args, c_args, d_args])
        state = eqx.tree_at(lambda t: t.fl.needs_regular_init, state,
                            jnp.array([[0]]))

        max_iter_per_solve = cp.p["max_iter"]
        rng_key = jax.random.PRNGKey(0)
        _solve_throughput = eqx.filter_jit(solve_throughput, donate="none")

        rows = []
        for N_batch in batches:
            max_solves = max(N_RUNS, MAX_SOLVES_MULT * N_batch)
            print(f"\n--- N_batch={N_batch} (max_solves={max_solves}) ---")
            try:
                batch_tp = make_batch_state(cp, [state] * N_batch)

                first_batch_x0 = all_x0[:N_batch]
                first_batch_xr = all_xr[:N_batch]
                batch_tp = eqx.tree_at(lambda s: s.args[1][1], batch_tp,
                                       first_batch_x0)
                batch_tp = eqx.tree_at(lambda s: s.args[0][4], batch_tp,
                                       first_batch_xr)
                first_batch_warms = jax.vmap(build_warm_start)(
                    first_batch_x0, first_batch_xr)
                new_x = batch_tp.it.x.at[:, :cp.nx, 0].set(first_batch_warms)
                batch_tp = eqx.tree_at(lambda s: s.it.x, batch_tp, new_x)

                label = f"track_v{AVG_VEL:.1f}_b{N_batch}"

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
                _, solution_buffer, write_idx, _, iter_buffer = tp_out[:5]
            else:
                _, solution_buffer, write_idx = tp_out
                iter_buffer = None
            n_collected = min(int(write_idx), int(solution_buffer.shape[0]))
            throughput = n_collected / wall
            if iter_buffer is not None:
                mean_iters = float(np.asarray(iter_buffer)[:n_collected].mean())
            else:
                mean_iters = float("nan")

            print(f"N_batch={N_batch}: wall={wall:.2f}s  "
                  f"collected={n_collected}/{max_solves}  "
                  f"throughput={throughput:.2f} solves/s")
            rows.append(dict(N_batch=N_batch, max_solves=max_solves,
                             wall=wall, warmup_s=warmup_s,
                             n_collected=n_collected, throughput=throughput,
                             mean_iters=mean_iters,
                             warmup_t0=wu0, warmup_t1=wu1,
                             rep_rows=rep_rows))

        if not rows:
            print(f"avg_vel={AVG_VEL}: no successful batch configs")
            continue

        if args.repeats > 1 or args.out_suffix:
            suffix = f"_{args.out_suffix}" if args.out_suffix else ""
            out_path = os.path.join(
                logs_dir,
                f"jaxipm_variance_v{AVG_VEL:.1f}{suffix}_results.npz")
            save_variance_npz(out_path, rows, N=N_horizon, Ts=Ts_,
                              avg_vel=AVG_VEL)
            for r in rows:
                variance_summary(f"track_v{AVG_VEL:.1f}_b{r['N_batch']}",
                                 r["rep_rows"])
            continue

        out_path = os.path.join(
            logs_dir, f"jaxipm_batch_sweep_v{AVG_VEL:.1f}_results.npz")
        np.savez(
            out_path,
            batch_sizes=np.array([r["N_batch"] for r in rows]),
            throughput=np.array([r["throughput"] for r in rows]),
            wall=np.array([r["wall"] for r in rows]),
            warmup_s=np.array([r["warmup_s"] for r in rows]),
            n_collected=np.array([r["n_collected"] for r in rows]),
            max_solves=np.array([r["max_solves"] for r in rows]),
            mean_iters=np.array([r["mean_iters"] for r in rows]),
            N=np.array([N_horizon]),
            Ts=np.array([Ts_]),
            avg_vel=np.array([AVG_VEL]),
        )
        print(f"\nsaved {out_path}")
        print(f"===== avg_vel={AVG_VEL} batch-sweep summary =====")
        print(f"{'batch':>6} {'solves':>7} {'wall[s]':>9} {'solves/s':>9}")
        for r in rows:
            print(f"{r['N_batch']:>6d} {r['n_collected']:>7d} "
                  f"{r['wall']:>9.2f} {r['throughput']:>9.2f}")


if __name__ == "__main__":
    main()

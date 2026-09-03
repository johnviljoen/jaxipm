"""jaxipm ablation / census harness for the RA-L rebuttal (E2, E5, E6, E11).

One process = one (scenario, variant, mode, ir_nsteps, debug) configuration.
Outputs are written with a configuration-encoding filename so nothing here
can overwrite the paper's headline npz files:

  <scenario logs>/jaxipm_ablation_<tag>_<mode>_ir<K>_b<N>[_dbg][_<suffix>]_results.npz

Modes
  hr   (config c, the paper method)  fusion + iteration-level batching:
       one solve_throughput call; converged slots hot-restart on new problems.
  ws   (config b, "fusion only")     solve-level batching: the pool is cut
       into ceil(max_solves / N_batch) batches; each batch runs until its
       SLOWEST member terminates (hot_restarting=False), then the next batch
       starts. Every slot is regular-initialised on its own problem before
       its first iteration (needs_regular_init=1), so per-problem iteration
       counts are those of a properly initialised IPOPT-style solve.

Timing protocol (E3): one untimed warmup call pays JIT, then --repeats
identical timed calls (hr: one solve_throughput call each; ws: the full
batch sequence each) with per-repeat/per-batch epoch stamps. Batch states are
built BEFORE timing (problem construction is excluded, as for the CPU
baselines). --debug turns on DEBUG_MODE, which retains per-problem iteration
counts and termination codes plus a census of every terminal code raised
(including hot-restart-hidden failures) -- it depresses throughput, so quality
runs and timing runs are separate processes.

Run from the repo root, e.g.
  python -m tests.rebuttal.jaxipm_ablation --scenario nav --variants 90 \
      --mode ws --debug --gpu 1
  python -m tests.rebuttal.jaxipm_ablation --scenario track --mode hr \
      --ir-nsteps 100 --debug --gpu 1 --out-suffix e6
"""

import argparse
import dataclasses
import json
import os
import socket
import subprocess
import sys
from time import time

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _git(repo, *args):
    try:
        return subprocess.check_output(["git", "-C", repo, *args],
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def _gpu_uuid(visible):
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,uuid,name", "--format=csv,noheader"],
            text=True)
        rows = [r.split(", ") for r in out.strip().splitlines()]
        idx = int(str(visible).split(",")[0])
        for r in rows:
            if int(r[0]) == idx:
                return r[1], r[2]
    except Exception:
        pass
    return "unknown", "unknown"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", required=True, choices=["nav", "track", "multi"])
    ap.add_argument("--variants", type=str, default=None,
                    help="comma list (nav: sector deg; track: avg_vel; multi: N_quads)")
    ap.add_argument("--mode", choices=["hr", "ws"], default="hr")
    ap.add_argument("--debug", action="store_true",
                    help="DEBUG_MODE=True: per-problem iters/terms + term census")
    ap.add_argument("--ir-nsteps", type=int, default=None,
                    help="iterative refinement depth (default: params.json, i.e. 0)")
    ap.add_argument("--batch", type=int, default=None,
                    help="N_batch (default: paper batch size for the scenario)")
    ap.add_argument("--max-solves", type=int, default=None,
                    help="solutions to collect (default: N_RUNS_jaxipm)")
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--init-first-batch", type=int, default=1, choices=[0, 1],
                    help="hr only: regular-initialise the first batch on its "
                         "injected problems (1, default) or inherit the "
                         "default-x0 initialisation as the paper drivers do (0). "
                         "ws always initialises.")
    ap.add_argument("--gpu", type=str, default="1")
    ap.add_argument("--out-suffix", type=str, default="")
    ap.add_argument("--pool-subsample", action="store_true",
                    help="ws only: draw the max_solves problems evenly across the "
                         "whole pool (np.linspace) instead of the first max_solves "
                         "instances -- use for small-N quality studies (I2)")
    ap.add_argument("--no-save-z", action="store_true",
                    help="omit the full solution vectors from the npz")
    ap.add_argument("--no-persistent-cache", action="store_true",
                    help="do not use the on-disk JAX compilation cache: every "
                         "process compiles from scratch (fresh-JIT timing protocol)")
    ap.add_argument("--pool-order", choices=["paper", "seq"], default="paper",
                    help="hr only: 'paper' = the paper drivers' hot-restart "
                         "schedule (nav nearest-angle stride, track random "
                         "draw; duplicates and gaps in pool coverage); 'seq' = "
                         "sequential pool: first batch = indices 0..N_batch-1, "
                         "every freed slot takes the next unused index, so each "
                         "of max_solves indices is solved exactly once and a "
                         "failed slot counts as a failure against the pool. "
                         "Adds '_seq' to the output name.")
    args = ap.parse_args()
    if args.pool_order == "seq" and args.mode != "hr":
        ap.error("--pool-order seq applies to --mode hr only (ws already injects each index once)")

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", args.gpu)
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.chdir(REPO)
    sys.path.insert(0, REPO)

    from tests.rebuttal.problems import (
        DEFAULT_VARIANTS, build_scenario, load_params, paper_batch_size,
        parse_variant)
    from tests.jaxipm_variance_common import iso, stamp

    variants = ([parse_variant(args.scenario, s) for s in args.variants.split(",")]
                if args.variants else DEFAULT_VARIANTS[args.scenario])

    p = load_params()
    p["DEBUG_MODE"] = bool(args.debug)
    p["hot_restarting"] = (args.mode == "hr")
    seq = (args.pool_order == "seq")
    p["sequential_pool"] = seq            # pool_size is set per variant below
    if args.ir_nsteps is not None:
        p["ir_nsteps"] = int(args.ir_nsteps)
    ir_k = int(p["ir_nsteps"])

    import jax
    import jax.numpy as jnp
    import equinox as eqx

    jax.config.update("jax_enable_x64", True)
    if not args.no_persistent_cache:
        jax.config.update("jax_compilation_cache_dir", "/home/john/tmp/jax_cache")
        jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
        jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
        jax.config.update("jax_persistent_cache_enable_xla_caches",
                          "xla_gpu_per_fusion_autotune_cache_dir")

    from jaxipm.solver import solve_throughput, make_batch_state

    gpu_uuid, gpu_name = _gpu_uuid(os.environ["CUDA_VISIBLE_DEVICES"])
    spineax_repo = "/home/john/code/spineax"
    metadata_common = dict(
        jaxipm_commit=_git(REPO, "rev-parse", "HEAD"),
        jaxipm_dirty=_git(REPO, "status", "--porcelain") != "",
        spineax_commit=_git(spineax_repo, "rev-parse", "HEAD"),
        spineax_branch=_git(spineax_repo, "branch", "--show-current"),
        spineax_dirty=_git(spineax_repo, "status", "--porcelain") != "",
        hostname=socket.gethostname(), gpu_uuid=gpu_uuid, gpu_name=gpu_name,
        cuda_visible_devices=os.environ["CUDA_VISIBLE_DEVICES"],
        python=sys.version, jax=jax.__version__,
        argv=sys.argv, params=p, rng_seed=0,
        persistent_compilation_cache=not args.no_persistent_cache,
    )

    for variant in variants:
        jax.clear_caches()
        # N_batch must be known before building (nav's hot-restart stride).
        N_batch = args.batch or paper_batch_size(args.scenario)
        sc = build_scenario(args.scenario, variant, p, N_batch)
        cp = sc.cp
        max_solves = args.max_solves or sc.N_RUNS
        if seq:
            if N_batch > max_solves:
                raise SystemExit(f"--pool-order seq needs N_batch <= max_solves ({N_batch} > {max_solves})")
            # static: the solver reads pool_size from cp.p at trace time
            cp = eqx.tree_at(lambda c: c.p, cp, dict(cp.p, pool_size=int(max_solves)))
            sc = dataclasses.replace(sc, cp=cp)
        max_iter = int(cp.p["max_iter"])
        label = f"{sc.label}_{args.mode}_ir{ir_k}_b{N_batch}{'_seq' if seq else ''}{'_dbg' if args.debug else ''}"
        print(f"\n===== {label}: max_solves={max_solves} repeats={args.repeats} "
              f"init_first_batch={args.init_first_batch} =====", flush=True)

        rng_key = jax.random.PRNGKey(0)
        _solve = eqx.filter_jit(solve_throughput, donate="none")

        def set_init_flag(batch_tp, val):
            flag = jnp.full_like(batch_tp.fl.needs_regular_init, val)
            return eqx.tree_at(lambda s: s.fl.needs_regular_init, batch_tp, flag)

        # ── build batch states (untimed) ────────────────────────────────────
        t_build0 = time()
        if args.mode == "hr":
            idx0 = np.arange(N_batch) % sc.N_RUNS      # seq: indices 0..N_batch-1 (N_batch <= max_solves)
            bt = sc.inject(make_batch_state(cp, [sc.state] * N_batch), idx0)
            bt = set_init_flag(bt, args.init_first_batch)
            batches = [(idx0, bt, max_solves)]
        else:
            # WS: ONE pristine batch state per config (one spineax token
            # mint -- registry entries persist until spineax.cudss.release());
            # every batch injects its own problems into it and sets
            # needs_regular_init so the first fused iteration re-initialises
            # and re-factorises each slot on its own problem -- the same path a
            # hot-restarted slot takes. Injection happens just before each
            # timed solve call and is excluded from the wall clock.
            bt0 = set_init_flag(make_batch_state(cp, [sc.state] * N_batch), 1)
            jax.block_until_ready(bt0.it.x)
            n_batches = int(np.ceil(max_solves / N_batch))
            if args.pool_subsample:
                pool_seq = np.unique(np.round(np.linspace(0, sc.N_RUNS - 1,
                                                          n_batches * N_batch)).astype(int))
                pool_seq = np.resize(pool_seq, n_batches * N_batch)
            else:
                pool_seq = np.arange(n_batches * N_batch) % sc.N_RUNS
            batches = []
            for k in range(n_batches):
                idx = pool_seq[k * N_batch:(k + 1) * N_batch]
                batches.append((idx, (lambda idx=idx: sc.inject(bt0, idx)), N_batch))
        if args.mode == "hr":
            jax.block_until_ready(batches[-1][1].it.x)
        build_s = time() - t_build0
        print(f"prepared {len(batches)} batch state(s) of {N_batch} in {build_s:.1f}s",
              flush=True)

        def materialize(bt):
            if callable(bt):
                bt = bt()
                jax.block_until_ready(bt.it.x)
            return bt

        def run_batch(bt, n):
            out = _solve(rng_key, cp, bt, n, max_iter_per_solve=max_iter)
            jax.block_until_ready(out[1])
            return out

        def peak_mem_bytes():
            try:
                st = jax.local_devices()[0].memory_stats()
                return int(st.get("peak_bytes_in_use", -1))
            except Exception:
                return -1

        # ── warmup (JIT) ────────────────────────────────────────────────────
        wu0 = time(); stamp(label, "WARMUP START", wu0)
        run_batch(materialize(batches[0][1]), batches[0][2])
        wu1 = time(); stamp(label, "WARMUP STOP ", wu1)
        print(f"warmup (incl. JIT): {wu1 - wu0:.1f}s", flush=True)

        # ── timed repeats ───────────────────────────────────────────────────
        rep_rows = []
        last_outs = None
        for rep in range(1, args.repeats + 1):
            outs = []
            batch_stamps = []
            r0 = time(); stamp(label, f"rep {rep}/{args.repeats} START", r0)
            for (idx, bt, n) in batches:
                bt = materialize(bt)          # untimed (WS lazy build)
                b0 = time()
                out = run_batch(bt, n)
                b1 = time()
                outs.append(out)
                batch_stamps.append((b0, b1))
            r1 = time(); stamp(label, f"rep {rep}/{args.repeats} STOP ", r1)
            n_col = sum(min(int(o[2]), int(o[1].shape[0])) for o in outs)
            wall = sum(b1 - b0 for b0, b1 in batch_stamps)   # solve-only wall
            rep_rows.append(dict(rep=rep, t_start=r0, t_stop=r1, wall=wall,
                                 wall_incl_host=r1 - r0, n_collected=n_col,
                                 throughput=n_col / wall,
                                 batch_t0=[b[0] for b in batch_stamps],
                                 batch_t1=[b[1] for b in batch_stamps]))
            print(f"[VARIANCE] {label} rep {rep}/{args.repeats}: wall={wall:.2f}s "
                  f"(incl. host loop {r1 - r0:.2f}s) collected={n_col} "
                  f"throughput={n_col / wall:.2f} solves/s", flush=True)
            last_outs = outs

        peak_mem = peak_mem_bytes()
        print(f"[MEM] {label}: device peak_bytes_in_use = {peak_mem/2**20:.0f} MiB", flush=True)
        try:
            from spineax import cudss as _cudss
            print(f"[SPINEAX] {label}: registry_size={_cudss.registry_size()} "
                  f"cache_capacity={_cudss.cache_capacity()} rebuild_count={_cudss.rebuild_count()} "
                  f"(rebuilds > 0 => LRU eviction re-factorized live tokens inside the timed loop)",
                  flush=True)
            spineax_stats = dict(registry_size=_cudss.registry_size(),
                                 cache_capacity=_cudss.cache_capacity(),
                                 rebuild_count=_cudss.rebuild_count())
        except Exception as e:  # pragma: no cover
            spineax_stats = dict(error=str(e))
        thr = np.array([r["throughput"] for r in rep_rows])
        print(f"[VARIANCE] {label}: n={len(thr)} mean={thr.mean():.2f} "
              f"std={thr.std(ddof=1) if len(thr) > 1 else 0.0:.2f}", flush=True)

        # ── collect per-problem results from the last repeat ────────────────
        z_list, iters_list, terms_list, bidx_list = [], [], [], []
        blog_list, rlog_list, ebuf_list, brow_list, wlog_list, iclog_list = [], [], [], [], [], []
        term_hist = np.zeros(6, dtype=np.int64)
        n_fused = []
        for b, ((idx, _bt, n), out) in enumerate(zip(batches, last_outs)):
            n_col = min(int(out[2]), int(out[1].shape[0]))
            z = np.asarray(out[1][:n_col, :cp.nx])
            z_list.append(z)
            bidx_list.append(np.full(n_col, b, dtype=np.int32))
            if len(out) >= 5:
                iters_list.append(np.asarray(out[4])[:n_col].astype(np.int32))
                terms_list.append(np.asarray(out[3])[:n_col].astype(np.int32))
                if len(out) >= 7:
                    term_hist += np.asarray(out[5]).astype(np.int64)
                    nf = int(out[6])
                    n_fused.append(nf)
                if len(out) >= 10:
                    blog_list.append(np.asarray(out[7])[:nf])
                    rlog_list.append(np.asarray(out[8])[:nf])
                    ebuf_list.append(np.asarray(out[9])[:n_col])
                    brow_list.append(np.full(nf, b, dtype=np.int32))
                    if len(out) >= 11:
                        wlog_list.append(np.asarray(out[10])[:nf])
                    if len(out) >= 12:
                        iclog_list.append(np.asarray(out[11])[:nf])
            else:
                iters_list.append(np.full(n_col, -1, dtype=np.int32))
                terms_list.append(np.full(n_col, -1, dtype=np.int32))
        z_all = np.concatenate(z_list) if z_list else np.zeros((0, cp.nx))
        instr = {}
        if blog_list:
            instr = dict(branch_log=np.concatenate(blog_list),      # (sum n_fused, N_batch) int8
                         kkt_res_log=np.concatenate(rlog_list),     # (sum n_fused, N_batch) f32
                         kkt_err=np.concatenate(ebuf_list),         # (n_collected, 4)
                         log_batch_of_row=np.concatenate(brow_list))
            if wlog_list:
                instr["kkt_bwd_log"] = np.concatenate(wlog_list)
            if iclog_list:
                instr["ic_log"] = np.concatenate(iclog_list)
                _ic = instr["ic_log"]; _act = _ic >= 0
                print(f"{label}: INERTIA CORRECTION applied (dxs>0) on {100*(_ic[_act] == 1).mean():.1f}% of "
                      f"active slot-iterations", flush=True)
        iters = np.concatenate(iters_list)
        terms = np.concatenate(terms_list)
        batch_of = np.concatenate(bidx_list)
        n_collected = z_all.shape[0]

        pool_idx = np.array([sc.match_pool(z) for z in z_all], dtype=np.int32)
        obj_vals = np.array([sc.objective(z, i) for z, i in zip(z_all, pool_idx)])
        X_list = [sc.to_X(z) for z in z_all]
        X_all = (np.stack(X_list) if X_list else
                 np.zeros((0,) + tuple(sc.to_X(np.asarray(sc.state.it.x[:cp.nx, 0])).shape)))
        success = np.array([sc.success(X, int(it), max_iter)
                            for X, it in zip(X_list, iters)], dtype=bool)

        ok_it = iters[iters >= 0]
        # Success against the POOL (max_solves instances requested), not just
        # against the rows that came back: a member that never terminated
        # (WS cap-hitter) or failed (seq HR) is a failure of the pool instance.
        n_success_pool = int(success.sum())
        print(f"{label}: collected={n_collected}/{max_solves} "
              f"success={n_success_pool}/{n_collected} (collected) = "
              f"{n_success_pool}/{max_solves} ({100.0 * n_success_pool / max_solves:.2f}% of pool) "
              f"unique_pool={len(np.unique(pool_idx))}/{sc.N_RUNS} "
              f"obj mean={obj_vals.mean() if n_collected else float('nan'):.3f}",
              flush=True)
        if seq:
            u, cnt = np.unique(pool_idx, return_counts=True)
            dup = int((cnt > 1).sum()); missing = int(max_solves - u.size)
            print(f"{label}: SEQ CHECK duplicates={dup} missing={missing} "
                  f"(expected 0 / {max_solves - n_collected})", flush=True)
        if ok_it.size:
            print(f"{label}: iters mean={ok_it.mean():.1f} median={np.median(ok_it):.0f} "
                  f"max={ok_it.max()}  term codes (1/5 among collected): "
                  f"{np.bincount(terms[terms >= 0], minlength=6).tolist()}", flush=True)
        if instr:
            bl = instr["branch_log"]; act = bl >= 0
            codes, cnt = np.unique(bl[act], return_counts=True)
            distinct = np.array([len(np.unique(row[row >= 0])) for row in bl])
            rl = instr["kkt_res_log"]; rl = rl[np.isfinite(rl)]
            print(f"{label}: BRANCH CENSUS over {act.sum()} slot-iterations: "
                  f"{dict(zip(codes.tolist(), cnt.tolist()))}  "
                  f"(0 full-step,1 SOC,2 WD,3 TS,4 SFR,5 resto-entry,6 backtracked,7 init,+8 in-resto); "
                  f"distinct branches/iteration mean={distinct.mean():.2f} max={distinct.max()}",
                  flush=True)
            if rl.size:
                print(f"{label}: KKT step residual ||Kd-r||/||r||: median={np.median(rl):.2e} "
                      f"p95={np.percentile(rl, 95):.2e} max={rl.max():.2e}", flush=True)
            ke = instr["kkt_err"]
            print(f"{label}: KKT error at termination (overall/dual/constr/compl) median: "
                  f"{np.nanmedian(ke, axis=0)}  max: {np.nanmax(ke, axis=0)}", flush=True)
        if args.debug and len(last_outs[0]) >= 7:
            print(f"{label}: TERM CENSUS (all slot-iterations, codes 0..5 = "
                  f"continue/converged/max_iter/tiny_step/resto_fail/acceptable): "
                  f"{term_hist.tolist()}  fused iterations per batch: {n_fused}",
                  flush=True)

        # ── save ────────────────────────────────────────────────────────────
        os.makedirs(sc.logs_dir, exist_ok=True)
        suffix = f"_{args.out_suffix}" if args.out_suffix else ""
        out_path = os.path.join(
            sc.logs_dir,
            f"jaxipm_ablation_{sc.tag}_{args.mode}_ir{ir_k}_b{N_batch}"
            f"{'_seq' if seq else ''}{'_dbg' if args.debug else ''}{suffix}_results.npz")
        meta = dict(metadata_common, scenario=args.scenario, variant=variant,
                    mode=args.mode, ir_nsteps=ir_k, debug=bool(args.debug),
                    pool_order=args.pool_order,
                    N_batch=N_batch, max_solves=max_solves, max_iter=max_iter,
                    tol=p["tol"], init_first_batch=args.init_first_batch,
                    pool_subsample=bool(args.pool_subsample),
                    n_batches=len(batches), build_s=build_s,
                    warmup_start=wu0, warmup_stop=wu1,
                    warmup_start_iso=iso(wu0), warmup_stop_iso=iso(wu1),
                    scenario_meta=sc.meta, spineax_stats=spineax_stats)
        np.savez(
            out_path,
            **({} if args.no_save_z else {"z_all": z_all}),
            X_all=X_all, iters=iters, terms=terms, obj_vals=obj_vals,
            success=success, pool_idx=pool_idx, batch_of=batch_of,
            term_hist=term_hist,
            **instr,
            n_fused_iters=np.array(n_fused if n_fused else [-1] * len(batches)),
            rep_index=np.array([r["rep"] for r in rep_rows]),
            rep_wall=np.array([r["wall"] for r in rep_rows]),
            rep_wall_incl_host=np.array([r["wall_incl_host"] for r in rep_rows]),
            rep_n_collected=np.array([r["n_collected"] for r in rep_rows]),
            rep_throughput=np.array([r["throughput"] for r in rep_rows]),
            rep_start_epoch=np.array([r["t_start"] for r in rep_rows]),
            rep_stop_epoch=np.array([r["t_stop"] for r in rep_rows]),
            rep_batch_t0=np.array([r["batch_t0"] for r in rep_rows]),
            rep_batch_t1=np.array([r["batch_t1"] for r in rep_rows]),
            total_time=np.array([rep_rows[-1]["wall"]]),
            peak_mem_bytes=np.array([peak_mem]),
            N_RUNS=np.array([max_solves]),
            N_BATCH=np.array([N_batch]),
            n_success_pool=np.array([n_success_pool]),
            pool_success_rate=np.array([n_success_pool / max_solves]),
            metadata=np.array([json.dumps(meta, default=str)]),
            **sc.extra_npz,
        )
        print(f"saved {out_path}", flush=True)


if __name__ == "__main__":
    main()

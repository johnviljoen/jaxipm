"""DAgger on quad track-avoid with a batched NLP expert (memoryless student).

Round 0  : behaviour cloning from closed-loop EXPERT rollouts (n_roll starts x T
           steps); the first n_heldout starts are paper-pool members used only
           as the expert reference / gate, the rest are training data.
Rounds 1+: roll out the STUDENT from n_roll fresh starts, label every visited
           (state, phase, time) with one expert call (pool = n_roll*T,
           warm-started from the student's own rollout), aggregate, retrain
           from scratch, evaluate on the held-out starts.

Experts (--expert), all solving the identical NLP:
  il      jaxipm, iteration-level batching (paper method, solve_throughput + hot restart)
  sl      jaxipm, solve-level batching (batches run until their slowest member terminates)
  madnlp  MadNLP + ExaModels on the GPU (cuDSS), sequential solves (paper baseline)

Run from the repo root:  tests/imitation_track/run.sh [GPU]   (one run)
                         tests/imitation_track/campaign.sh [GPU]   (5 x {il, madnlp, sl} -> logs/runs/)
"""

import argparse
import json
import os
import socket
import subprocess
from time import time

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = os.path.join(os.path.dirname(__file__), "logs")


def _gpu_info():
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    try:
        q = subprocess.run(["nvidia-smi", "--query-gpu=index,name,uuid", "--format=csv,noheader"],
                           capture_output=True, text=True, timeout=10).stdout.strip().splitlines()
        rows = [r for r in q if cvd == "" or r.split(",")[0].strip() in cvd.split(",")]
    except Exception:
        rows = []
    return dict(cuda_visible_devices=cvd, nvidia_smi=rows, hostname=socket.gethostname())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--expert", choices=["il", "sl", "madnlp"], default="il")
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--n-roll", type=int, default=256)
    ap.add_argument("--T", type=int, default=50)
    ap.add_argument("--avg-vel", type=float, default=1.0, help="reference phase rate (paper: 1.0/1.5/2.0)")
    ap.add_argument("--n-heldout", type=int, default=32)
    ap.add_argument("--batch", type=int, default=500, help="jaxipm batch size (il/sl)")
    ap.add_argument("--train-steps", type=int, default=10000)
    ap.add_argument("--train-batch", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--margin", type=float, default=0.1, help="planner obstacle margin [m] (eval uses the true radius)")
    ap.add_argument("--k-start", type=int, default=1, help="first horizon stage with obstacle constraints")
    ap.add_argument("--r-u", type=float, default=1e-5, help="expert input cost weight on (u-hover)^2 (paper: 0 = bang-bang; 1e-5 halves iterations and failures)")
    ap.add_argument("--madnlp-tol", type=float, default=1e-6, help="MadNLP tolerance (paper driver: 1e-6)")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    os.chdir(REPO)

    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from tests.imitation_track import sim, policy as pol
    from tests.imitation_track.expert import Expert, expert_rollout, rollout_warm_starts, match_rows

    os.makedirs(args.out, exist_ok=True)
    key = jax.random.PRNGKey(args.seed)
    T, R, v = args.T, args.n_roll, args.avg_vel
    hx0, hs0 = sim.heldout_starts(args.n_heldout, v)
    metrics = dict(args=vars(args), expert=args.expert, gpu=_gpu_info(), jax=jax.__version__,
                   jax_device=str(jax.devices()[0]), rounds=[])
    t_start = time()

    def make_expert(pool_size, batch_size):
        if args.expert == "madnlp":
            from tests.imitation_track.expert_madnlp import MadNLPExpert
            return MadNLPExpert(pool_size=pool_size, v=v, margin=args.margin, k_start=args.k_start, r_u=args.r_u,
                                tol=args.madnlp_tol, log_path=os.path.join(args.out, "madnlp_server.log"))
        return Expert(pool_size=pool_size, batch_size=batch_size, v=v, margin=args.margin, k_start=args.k_start,
                      r_u=args.r_u, mode=args.expert)

    def dump():
        metrics["wall_total_s"] = time() - t_start
        with open(os.path.join(args.out, "metrics.json"), "w") as fh:
            json.dump(metrics, fh, indent=2, default=float)

    def feats_of_rollout(X, s0s):
        """features for every visited state X[:, :T] of rollouts with phases s0s -> (B*T, FEAT_DIM)"""
        B = X.shape[0]
        Xv = jnp.asarray(X[:, :T].reshape(-1, sim.NX))
        S = jnp.repeat(jnp.asarray(s0s), T)
        Tt = jnp.tile(jnp.arange(T) * sim.TS, B)
        return np.asarray(sim.features_batch(Xv, S, Tt, v)), np.asarray(S), np.asarray(Tt)

    def eval_student(policy, tag):
        X, U = jax.jit(lambda x, s: sim.rollout(jax.vmap(policy), x, s, T, v))(jnp.asarray(hx0), jnp.asarray(hs0))
        X, U = np.asarray(X), np.asarray(U)
        ev = sim.evaluate(X, hs0, v)
        print(f"  [{tag}] held-out: success {ev['success'].sum()}/{len(hx0)}  finite {ev['finite'].sum()}  "
              f"median tail RMS err {np.median(ev['rms_tail']):.3f} m  min clearance {ev['min_clearance'].min():+.3f} m", flush=True)
        return X, U, ev

    def summary(ev):
        return dict(student_success=int(ev["success"].sum()), student_finite=int(ev["finite"].sum()),
                    student_median_rms_tail=float(np.median(ev["rms_tail"])),
                    student_min_clearance=float(ev["min_clearance"].min()))

    def iter_stats(iters):
        iters = np.asarray(iters)
        ok = iters[iters >= 0] if iters.size else iters
        return dict(iters_mean=float(ok.mean()) if ok.size else float("nan"),
                    iters_max=int(ok.max()) if ok.size else -1)

    # ── round 0 ──────────────────────────────────────────────────────────────
    print(f"=== round 0: expert [{args.expert}] closed-loop rollouts, {R} starts x {T} steps, avg_vel {v} ===", flush=True)
    t_round = time()
    tx0, ts0 = sim.sample_starts(R - args.n_heldout, v, seed=10_000 + args.seed)
    x0s = np.concatenate([hx0, tx0]); s0s = np.concatenate([hs0, ts0])
    ex_small = make_expert(R, R)
    t0 = time()
    Xe, Ue, OKe, step_times = expert_rollout(ex_small, x0s, s0s, T)
    t_round0 = time() - t0
    ev_e = sim.evaluate(Xe[:args.n_heldout], hs0, v)
    frac_bounds = float(np.mean((Ue < sim.U_MIN + 1) | (Ue > sim.U_MAX - 1)))
    print(f"  [expert] held-out: success {ev_e['success'].sum()}/{args.n_heldout}  median tail RMS err "
          f"{np.median(ev_e['rms_tail']):.3f} m  min clearance {ev_e['min_clearance'].min():+.3f} m  "
          f"labelled {OKe.mean()*100:.1f}%  at motor bounds {frac_bounds*100:.1f}%  wall {t_round0:.0f}s", flush=True)
    np.savez(os.path.join(args.out, "expert_heldout.npz"), X=Xe[:args.n_heldout], U=Ue[:args.n_heldout],
             ok=OKe[:args.n_heldout], x0=hx0, s0=hs0, avg_vel=v)
    np.savez(os.path.join(args.out, "expert_rollouts_round0.npz"), X=Xe, U=Ue, ok=OKe, x0=x0s, s0=s0s,
             step_times=step_times, avg_vel=v)
    startup_s = getattr(ex_small, "startup_s", None)
    if hasattr(ex_small, "close"):
        ex_small.close()
    del ex_small

    tr = slice(args.n_heldout, None)
    F_all, _, _ = feats_of_rollout(Xe[tr], s0s[tr])
    mask = OKe[tr].reshape(-1)
    DF = F_all[mask]; DU = Ue[tr].reshape(-1, sim.NU)[mask]
    u_scale = pol.u_scale_from_data(DU)
    print(f"  dataset: {DF.shape[0]} labels, u_scale = {u_scale:.1f}", flush=True)
    key, k = jax.random.split(key)
    t0 = time()
    student, hist = pol.train(k, DF, DU, u_scale, steps=args.train_steps, batch=args.train_batch)
    t_train = time() - t0
    pol.save(student, os.path.join(args.out, "policy_round0.eqx"))
    Xs, Us, ev_s = eval_student(student, "student r0")
    np.savez(os.path.join(args.out, "dataset_round0.npz"), F=DF, U=DU)
    metrics["rounds"].append(dict(round=0, n_labels=int(DF.shape[0]), new_labels=int(DF.shape[0]),
                                  label_wall_s=float(step_times.sum()), round0_wall_s=t_round0,
                                  step_times=[float(t) for t in step_times], expert_startup_s=startup_s,
                                  expert_labelled_frac=float(OKe.mean()),
                                  train_wall_s=t_train, round_wall_s=time() - t_round,
                                  final_loss=hist[-1][1], u_scale=u_scale, expert_frac_at_bounds=frac_bounds,
                                  expert_success=int(ev_e["success"].sum()),
                                  expert_median_rms_tail=float(np.median(ev_e["rms_tail"])), **summary(ev_s)))
    dump()

    # ── rounds 1..K ──────────────────────────────────────────────────────────
    P = R * T
    ex_big = make_expert(P, args.batch)
    for r in range(1, args.rounds + 1):
        print(f"=== round {r}: student rollouts -> {P} expert [{args.expert}] labels ===", flush=True)
        t_round = time()
        key, k2 = jax.random.split(key)
        rx0, rs0 = sim.sample_starts(R, v, seed=20_000 + 100 * args.seed + r)
        X, U = jax.jit(lambda x, s: sim.rollout(jax.vmap(student), x, s, T, v))(jnp.asarray(rx0), jnp.asarray(rs0))
        X, U = np.asarray(X), np.asarray(U)
        ev_r = sim.evaluate(X, rs0, v)
        print(f"  student rollouts: success {ev_r['success'].sum()}/{R}  finite {ev_r['finite'].sum()}  "
              f"median tail RMS err {np.median(ev_r['rms_tail']):.3f} m  min clearance {ev_r['min_clearance'].min():+.3f}", flush=True)
        visited = X[:, :T].reshape(-1, sim.NX)
        F_vis, S_vis, T_vis = feats_of_rollout(X, rs0)
        n_valid = int(sim.valid_states(visited).sum())
        if n_valid < 0.05 * P:
            print(f"  ABORT: only {n_valid}/{P} visited states are valid (student diverged)", flush=True)
            metrics["aborted_round"] = r
            break
        res = ex_big.label(visited, S_vis, T_vis, warm=rollout_warm_starts(X, U))
        print(f"  labelled {res['n']}/{P} ({res['n_invalid']} invalid states) in {res['elapsed']:.1f}s "
              f"= {res['n']/max(res['elapsed'],1e-9):.1f} solves/s; iters mean {res['iters'].mean() if res['n'] else float('nan'):.1f}; "
              f"census {res['term_hist'][1:].tolist()}", flush=True)
        Fnew = F_vis[match_rows(res["X0"], visited)] if res["n"] else np.zeros((0, sim.FEAT_DIM))
        DF = np.concatenate([DF, Fnew]); DU = np.concatenate([DU, res["U0"]])
        u_scale = max(u_scale, pol.u_scale_from_data(DU))
        t0 = time()
        student, hist = pol.train(k2, DF, DU, u_scale, steps=args.train_steps, batch=args.train_batch)
        t_train = time() - t0
        pol.save(student, os.path.join(args.out, f"policy_round{r}.eqx"))
        Xs, Us, ev_s = eval_student(student, f"student r{r}")
        np.savez(os.path.join(args.out, f"dataset_round{r}.npz"), F=Fnew, U=res["U0"])
        np.savez(os.path.join(args.out, f"student_rollouts_round{r}.npz"), X=X, U=U, x0=rx0, s0=rs0)
        metrics["rounds"].append(dict(round=r, n_labels=int(DF.shape[0]), new_labels=int(res["n"]),
                                      n_invalid=res["n_invalid"], term_census=[int(c) for c in res["term_hist"]],
                                      rollout_success=int(ev_r["success"].sum()), label_wall_s=res["elapsed"],
                                      label_wall_io_s=res.get("elapsed_io"),
                                      madnlp_status_census=({int(k): int(c) for k, c in zip(*np.unique(res["madnlp_status"], return_counts=True))}
                                                            if "madnlp_status" in res else None),
                                      solves_per_s=res["n"] / max(res["elapsed"], 1e-9), **iter_stats(res["iters"]),
                                      train_wall_s=t_train, round_wall_s=time() - t_round,
                                      final_loss=hist[-1][1], u_scale=u_scale, **summary(ev_s)))
        dump()
    if hasattr(ex_big, "close"):
        ex_big.close()

    pol.save(student, os.path.join(args.out, "policy_final.eqx"))
    np.savez(os.path.join(args.out, "student_heldout.npz"), X=Xs, U=Us, x0=hx0, s0=hs0, avg_vel=v)
    metrics["label_wall_total_s"] = float(sum(m["label_wall_s"] for m in metrics["rounds"]))
    metrics["train_wall_total_s"] = float(sum(m["train_wall_s"] for m in metrics["rounds"]))
    dump()
    print("\nround  labels  student success  finite  median tail RMS  min clear   label s   train s")
    for m in metrics["rounds"]:
        print(f"{m['round']:5d} {m['n_labels']:7d} {m['student_success']:9d}/{args.n_heldout}   {m['student_finite']:5d}   "
              f"{m['student_median_rms_tail']:10.3f}   {m['student_min_clearance']:+7.3f}   {m['label_wall_s']:7.1f}   {m['train_wall_s']:6.1f}")
    print(f"total wall {time()-t_start:.0f}s (labelling {metrics['label_wall_total_s']:.0f}s, training "
          f"{metrics['train_wall_total_s']:.0f}s); outputs in {args.out}")


if __name__ == "__main__":
    main()

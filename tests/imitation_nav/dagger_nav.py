"""DAgger on the quad nav scenario with jaxipm as the batched expert.

Round 0  : behaviour cloning from closed-loop EXPERT rollouts (n_roll starts,
           T steps). The first n_heldout rollouts start from the held-out
           paper-linspace angles and are used only as the expert reference
           for the render / gate; the rest are training data.
Rounds 1+: roll out the current STUDENT from n_roll fresh starts, label every
           visited state with one solve_throughput call (pool = n_roll * T),
           aggregate, retrain from scratch, evaluate on the held-out starts.

Everything (rollouts, labelling, training) runs in one process on one GPU.
Run from the repo root:
    tests/imitation_nav/run.sh            # GPU 1, no preallocation
"""

import argparse
import json
import os
from time import time

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = os.path.join(os.path.dirname(__file__), "logs")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=8, help="DAgger rounds after BC round 0")
    ap.add_argument("--n-roll", type=int, default=256, help="rollouts per round")
    ap.add_argument("--T", type=int, default=50, help="episode length (10 Hz steps)")
    ap.add_argument("--sector", type=float, default=90.0)
    ap.add_argument("--n-heldout", type=int, default=32)
    ap.add_argument("--batch", type=int, default=500, help="solver batch for the big pool")
    ap.add_argument("--train-steps", type=int, default=10000)
    ap.add_argument("--train-batch", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--margin", type=float, default=0.1, help="planner obstacle margin [m] on top of the inflated radius (eval uses the true radius)")
    ap.add_argument("--k-start", type=int, default=1, help="first horizon stage with obstacle constraints (stage 0 is the measured state)")
    ap.add_argument("--use-prev-action", action="store_true",
                    help="feed the previous action to the student (off: it copies itself and diverges)")
    ap.add_argument("--r-u", type=float, default=3e-5, help="expert input cost weight on (u - hover)^2 (paper: 0 = bang-bang)")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    os.chdir(REPO)
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from tests.imitation_nav import sim, policy as pol
    from tests.imitation_nav.expert import Expert, expert_rollout, rollout_warm_starts, prev_actions, match_rows

    os.makedirs(args.out, exist_ok=True)
    key = jax.random.PRNGKey(args.seed)
    T, R = args.T, args.n_roll
    heldout = sim.arc_starts(sim.heldout_angles(args.n_heldout, args.sector))
    metrics = dict(args=vars(args), rounds=[])
    t_start = time()

    def dump():
        metrics["wall_total_s"] = time() - t_start
        with open(os.path.join(args.out, "metrics.json"), "w") as fh:
            json.dump(metrics, fh, indent=2, default=float)

    def eval_student(policy, tag):
        X, U = jax.jit(lambda x: sim.rollout(jax.vmap(policy), x, T))(jnp.asarray(heldout))
        X, U = np.asarray(X), np.asarray(U)
        ev = sim.evaluate(X)
        print(f"  [{tag}] held-out: success {ev['success'].sum()}/{len(heldout)}  "
              f"reached {ev['reached'].sum()}  median final dist {np.median(ev['final_dist']):.3f} m  "
              f"min clearance {ev['min_clearance'].min():+.3f} m", flush=True)
        return X, U, ev

    # ── Round 0: closed-loop expert rollouts (gate + BC data) ────────────────
    print(f"=== round 0: expert closed-loop rollouts, {R} starts x {T} steps ===", flush=True)
    key, k = jax.random.split(key)
    ang_train = sim.sample_angles(k, R - args.n_heldout, args.sector)
    x0s = np.concatenate([heldout, sim.arc_starts(ang_train)])
    ex_small = Expert(pool_size=R, batch_size=R, margin=args.margin, k_start=args.k_start, r_u=args.r_u)
    t0 = time()
    Xe, Ue, OKe, step_times = expert_rollout(ex_small, x0s, T)
    t_round0 = time() - t0
    ev_e = sim.evaluate(Xe[:args.n_heldout])
    print(f"  [expert] held-out: success {ev_e['success'].sum()}/{args.n_heldout}  "
          f"median final dist {np.median(ev_e['final_dist']):.3f} m  "
          f"min clearance {ev_e['min_clearance'].min():+.3f} m  labelled {OKe.mean()*100:.1f}%  "
          f"wall {t_round0:.0f}s", flush=True)
    np.savez(os.path.join(args.out, "expert_heldout.npz"), X=Xe[:args.n_heldout], U=Ue[:args.n_heldout],
             ok=OKe[:args.n_heldout], starts=heldout)
    del ex_small

    frac_bounds = float(np.mean((Ue < sim.U_MIN + 1) | (Ue > sim.U_MAX - 1)))
    print(f"  [expert] controls at motor bounds: {frac_bounds*100:.1f}%", flush=True)
    tr = slice(args.n_heldout, None)
    mask = OKe[tr].reshape(-1)
    DX = Xe[tr, :T].reshape(-1, sim.NX)[mask]
    DP = prev_actions(Ue[tr]).reshape(-1, sim.NU)[mask]
    DU = Ue[tr].reshape(-1, sim.NU)[mask]
    u_scale = pol.u_scale_from_data(DU)
    print(f"  dataset: {DX.shape[0]} labels, u_scale = {u_scale:.1f}", flush=True)

    key, k = jax.random.split(key)
    t0 = time()
    student, hist = pol.train(k, DX, DP, DU, u_scale, steps=args.train_steps, batch=args.train_batch, use_prev=args.use_prev_action)
    t_train = time() - t0
    pol.save(student, os.path.join(args.out, "policy_round0.eqx"))
    Xs, Us, ev_s = eval_student(student, "student r0")
    np.savez(os.path.join(args.out, "dataset_round0.npz"), X=DX, Up=DP, U=DU)
    metrics["rounds"].append(dict(
        round=0, n_labels=int(DX.shape[0]), new_labels=int(DX.shape[0]),
        label_wall_s=float(step_times.sum()), round0_wall_s=t_round0, train_wall_s=t_train,
        final_loss=hist[-1][1], u_scale=u_scale, expert_frac_at_bounds=frac_bounds,
        expert_success=int(ev_e["success"].sum()), student_success=int(ev_s["success"].sum()),
        student_reached=int(ev_s["reached"].sum()),
        student_median_final_dist=float(np.median(ev_s["final_dist"])),
        student_min_clearance=float(ev_s["min_clearance"].min())))
    dump()

    # ── Rounds 1..K: DAgger ──────────────────────────────────────────────────
    P = R * T
    ex_big = Expert(pool_size=P, batch_size=args.batch, margin=args.margin, k_start=args.k_start, r_u=args.r_u)
    for r in range(1, args.rounds + 1):
        print(f"=== round {r}: student rollouts -> {P} expert labels ===", flush=True)
        key, k1, k2 = jax.random.split(key, 3)
        x0s = jnp.asarray(sim.arc_starts(sim.sample_angles(k1, R, args.sector)))
        X, U = jax.jit(lambda x: sim.rollout(jax.vmap(student), x, T))(x0s)
        X, U = np.asarray(X), np.asarray(U)
        visited = X[:, :T].reshape(-1, sim.NX)
        ev_r = sim.evaluate(X)
        n_valid = int(sim.valid_states(visited).sum())
        if n_valid < 0.05 * P:
            print(f"  ABORT: only {n_valid}/{P} visited states are valid (student diverged); "
                  f"further rounds would add no labels", flush=True)
            metrics["aborted_round"] = r
            break
        print(f"  student rollouts: success {ev_r['success'].sum()}/{R}  median final dist "
              f"{np.median(ev_r['final_dist']):.3f} m  min clearance {ev_r['min_clearance'].min():+.3f}", flush=True)
        # warm start every NLP from the student's own future trajectory (exact dynamics)
        res = ex_big.label(visited, warm=rollout_warm_starts(X, U))
        # previous action of each labelled row (rows are unordered: match back to the pool)
        Pprev = prev_actions(U).reshape(-1, sim.NU)[match_rows(res["X0"], visited)]
        print(f"  labelled {res['n']}/{P} ({res['n_invalid']} invalid states) in {res['elapsed']:.1f}s "
              f"= {res['n']/res['elapsed']:.1f} solves/s; iters mean {res['iters'].mean():.1f}", flush=True)
        DX = np.concatenate([DX, res["X0"]]); DP = np.concatenate([DP, Pprev]); DU = np.concatenate([DU, res["U0"]])
        u_scale = max(u_scale, pol.u_scale_from_data(DU))
        t0 = time()
        student, hist = pol.train(k2, DX, DP, DU, u_scale, steps=args.train_steps, batch=args.train_batch, use_prev=args.use_prev_action)
        t_train = time() - t0
        pol.save(student, os.path.join(args.out, f"policy_round{r}.eqx"))
        Xs, Us, ev_s = eval_student(student, f"student r{r}")
        np.savez(os.path.join(args.out, f"dataset_round{r}.npz"), X=res["X0"], Up=Pprev, U=res["U0"])
        metrics["rounds"].append(dict(
            round=r, n_labels=int(DX.shape[0]), new_labels=int(res["n"]), n_invalid=res["n_invalid"],
            term_census=[int(v) for v in res["term_hist"]],
            rollout_success=int(ev_r["success"].sum()),
            label_wall_s=res["elapsed"], solves_per_s=res["n"] / res["elapsed"],
            iters_mean=float(res["iters"].mean()), train_wall_s=t_train, final_loss=hist[-1][1],
            u_scale=u_scale, student_success=int(ev_s["success"].sum()),
            student_reached=int(ev_s["reached"].sum()),
            student_median_final_dist=float(np.median(ev_s["final_dist"])),
            student_min_clearance=float(ev_s["min_clearance"].min())))
        dump()

    pol.save(student, os.path.join(args.out, "policy_final.eqx"))
    np.savez(os.path.join(args.out, "student_heldout.npz"), X=Xs, U=Us, starts=heldout)
    print("\nround  labels  student success  median dist  min clear   label s   train s")
    for m in metrics["rounds"]:
        print(f"{m['round']:5d} {m['n_labels']:7d} {m['student_success']:9d}/{args.n_heldout}      "
              f"{m['student_median_final_dist']:7.3f}   {m['student_min_clearance']:+7.3f}   "
              f"{m['label_wall_s']:7.1f}   {m['train_wall_s']:6.1f}")
    print(f"total wall {time()-t_start:.0f}s; outputs in {args.out}")


if __name__ == "__main__":
    main()

"""Inference-time comparison: learned MLP policy vs jaxipm receding-horizon solve.

Two numbers, both after an untimed JIT warmup, `--repeats` timed calls each:
  batched : the paper's 2000 nav arc states in one call
            (MLP vmap at batch 2000 vs solve_throughput at --batch, DEBUG off)
  single  : one state -> one control (MLP batch 1 vs jaxipm pool 1 / batch 1)
Writes tests/imitation_nav/logs/timing.json.
"""

import argparse
import json
import os
from time import time

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = os.path.join(os.path.dirname(__file__), "logs")


def _stats(ts):
    ts = np.asarray(ts)
    return dict(mean_s=float(ts.mean()), std_s=float(ts.std(ddof=1) if ts.size > 1 else 0.0),
                min_s=float(ts.min()), repeats=int(ts.size), all_s=[float(t) for t in ts])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default=os.path.join(OUT, "policy_final.eqx"))
    ap.add_argument("--sector", type=float, default=90.0)
    ap.add_argument("--batch", type=int, default=500)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--out", default=os.path.join(OUT, "timing.json"))
    args = ap.parse_args()
    os.chdir(REPO)

    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from tests.imitation_nav import sim, policy as pol
    from tests.imitation_nav.expert import Expert

    student = pol.load(args.policy)
    arc = sim.arc_starts(sim.paper_angles(args.sector))          # (2000, 13)
    n_arc = arc.shape[0]
    out = dict(n_arc=n_arc, batch=args.batch, repeats=args.repeats)

    # ── MLP ──────────────────────────────────────────────────────────────────
    f_batch = student.batched()
    xa = jnp.asarray(arc); ua = jnp.full((n_arc, sim.NU), sim.W_HOVER)
    jax.block_until_ready(f_batch(xa, ua))
    ts = []
    for _ in range(args.repeats):
        t0 = time(); jax.block_until_ready(f_batch(xa, ua)); ts.append(time() - t0)
    out["mlp_batched"] = _stats(ts)
    x1 = jnp.asarray(arc[:1]); u1 = ua[:1]
    jax.block_until_ready(f_batch(x1, u1))
    ts = []
    for _ in range(args.repeats * 20):
        t0 = time(); jax.block_until_ready(f_batch(x1, u1)); ts.append(time() - t0)
    out["mlp_single"] = _stats(ts)
    print(f"MLP   batched {n_arc}: {out['mlp_batched']['mean_s']*1e3:.3f} ms  "
          f"single: {out['mlp_single']['mean_s']*1e6:.1f} us", flush=True)

    # ── jaxipm batched (DEBUG off: this is a timing run) ─────────────────────
    ex = Expert(pool_size=n_arc, batch_size=args.batch, debug=False)
    ex.label(arc)                                                 # warmup
    ts, ns = [], []
    for _ in range(args.repeats):
        res = ex.label(arc); ts.append(res["elapsed"]); ns.append(res["n"])
    out["jaxipm_batched"] = dict(_stats(ts), n_collected=ns,
                                 solves_per_s=float(np.mean(ns) / np.mean(ts)))
    print(f"jaxipm batched {n_arc} @ batch {args.batch}: {np.mean(ts):.2f} s "
          f"({out['jaxipm_batched']['solves_per_s']:.1f} solves/s, collected {ns})", flush=True)
    del ex

    # ── jaxipm single ────────────────────────────────────────────────────────
    ex1 = Expert(pool_size=1, batch_size=1, debug=False)
    one = arc[n_arc // 2:n_arc // 2 + 1]
    ex1.label(one)
    ts = []
    for _ in range(args.repeats):
        ts.append(ex1.label(one)["elapsed"])
    out["jaxipm_single"] = _stats(ts)
    print(f"jaxipm single: {np.mean(ts)*1e3:.1f} ms", flush=True)

    out["speedup_batched"] = out["jaxipm_batched"]["mean_s"] / out["mlp_batched"]["mean_s"]
    out["speedup_single"] = out["jaxipm_single"]["mean_s"] / out["mlp_single"]["mean_s"]
    print(f"MLP faster by {out['speedup_batched']:.0f}x (batched {n_arc}) and "
          f"{out['speedup_single']:.0f}x (single state)")
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)
    print("wrote", args.out)


if __name__ == "__main__":
    main()

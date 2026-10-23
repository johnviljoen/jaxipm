"""Inference-time comparison on track-avoid: learned MLP vs jaxipm receding-horizon solve.

batched : the paper's 1000-start pool (features at t = 0) in one call,
          MLP vmap vs solve_throughput at --batch (DEBUG off)
single  : one state -> one control (MLP batch 1 vs jaxipm pool 1 / batch 1)
Writes logs/timing.json.
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
    ap.add_argument("--avg-vel", type=float, default=None, help="default: from logs/metrics.json")
    ap.add_argument("--batch", type=int, default=250)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--out", default=os.path.join(OUT, "timing.json"))
    args = ap.parse_args()
    os.chdir(REPO)
    if args.avg_vel is None:
        with open(os.path.join(OUT, "metrics.json")) as fh:
            args.avg_vel = float(json.load(fh)["args"]["avg_vel"])
    v = args.avg_vel

    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from tests.imitation_track import sim, policy as pol
    from tests.imitation_track.expert import Expert

    student = pol.load(args.policy)
    x0, s0 = sim.paper_pool(v)
    n = x0.shape[0]
    t0s = np.zeros(n)
    F = sim.features_batch(jnp.asarray(x0), jnp.asarray(s0), jnp.asarray(t0s), v)
    out = dict(n_pool=n, batch=args.batch, repeats=args.repeats, avg_vel=v)

    f_batch = student.batched()
    jax.block_until_ready(f_batch(F))
    ts = []
    for _ in range(args.repeats):
        t0 = time(); jax.block_until_ready(f_batch(F)); ts.append(time() - t0)
    out["mlp_batched"] = _stats(ts)
    F1 = F[:1]
    jax.block_until_ready(f_batch(F1))
    ts = []
    for _ in range(args.repeats * 20):
        t0 = time(); jax.block_until_ready(f_batch(F1)); ts.append(time() - t0)
    out["mlp_single"] = _stats(ts)
    print(f"MLP   batched {n}: {out['mlp_batched']['mean_s']*1e3:.3f} ms  single: {out['mlp_single']['mean_s']*1e6:.1f} us", flush=True)

    ex = Expert(pool_size=n, batch_size=args.batch, v=v, debug=False)
    ex.label(x0, s0, t0s)
    ts, ns = [], []
    for _ in range(args.repeats):
        res = ex.label(x0, s0, t0s); ts.append(res["elapsed"]); ns.append(res["n"])
    out["jaxipm_batched"] = dict(_stats(ts), n_collected=ns, solves_per_s=float(np.mean(ns) / np.mean(ts)))
    print(f"jaxipm batched {n} @ batch {args.batch}: {np.mean(ts):.2f} s ({out['jaxipm_batched']['solves_per_s']:.1f} solves/s, collected {ns})", flush=True)
    del ex

    ex1 = Expert(pool_size=1, batch_size=1, v=v, debug=False)
    one = (x0[n // 2:n // 2 + 1], s0[n // 2:n // 2 + 1], t0s[:1])
    ex1.label(*one)
    ts = [ex1.label(*one)["elapsed"] for _ in range(args.repeats)]
    out["jaxipm_single"] = _stats(ts)
    print(f"jaxipm single: {np.mean(ts)*1e3:.1f} ms", flush=True)

    out["speedup_batched"] = out["jaxipm_batched"]["mean_s"] / out["mlp_batched"]["mean_s"]
    out["speedup_single"] = out["jaxipm_single"]["mean_s"] / out["mlp_single"]["mean_s"]
    print(f"MLP faster by {out['speedup_batched']:.0f}x (batched {n}) and {out['speedup_single']:.0f}x (single state)")
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)
    print("wrote", args.out)


if __name__ == "__main__":
    main()

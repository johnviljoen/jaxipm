"""Shared helpers for the jaxipm GPU variance campaign.

Protocol (agreed 2026-08-13): per config, one process pays JIT once in an
untimed warmup call, then runs N identical timed solve_throughput calls
(same rng_key, same initial batch state). Every warmup and every repeat
gets wall-clock start/stop stamps (epoch + local ISO), printed and saved
in the npz, so externally detected GPU contamination can be
cross-referenced against individual repeats after the fact.
"""

import time
from datetime import datetime

import numpy as np


def iso(t):
    return datetime.fromtimestamp(t).astimezone().isoformat(
        timespec="milliseconds")


def stamp(label, event, t):
    print(f"[VARIANCE] {label} {event} epoch={t:.3f} ({iso(t)})", flush=True)


def timed_repeats(label, repeats, run_once, n_collected_of):
    """Run `repeats` identical timed calls; run_once() must block until the
    device results are ready and return them."""
    rows = []
    for rep in range(1, repeats + 1):
        t0 = time.time()
        stamp(label, f"rep {rep}/{repeats} START", t0)
        tp_out = run_once()
        t1 = time.time()
        stamp(label, f"rep {rep}/{repeats} STOP ", t1)
        n = n_collected_of(tp_out)
        wall = t1 - t0
        rows.append(dict(rep=rep, t_start=t0, t_stop=t1, wall=wall,
                         n_collected=n, throughput=n / wall, tp_out=tp_out))
        print(f"[VARIANCE] {label} rep {rep}/{repeats}: wall={wall:.2f}s  "
              f"collected={n}  throughput={n / wall:.2f} solves/s",
              flush=True)
    return rows


def variance_summary(label, rep_rows):
    thr = np.array([r["throughput"] for r in rep_rows])
    mean = thr.mean()
    std = thr.std(ddof=1) if len(thr) > 1 else 0.0
    print(f"[VARIANCE] {label}: n={len(thr)}  mean={mean:.2f}  "
          f"std={std:.2f}  cv={100 * std / mean:.1f}%  "
          f"reps={np.array2string(thr, precision=2)}", flush=True)
    return mean, std


def save_variance_npz(out_path, batch_rows, **meta):
    """batch_rows: dicts with N_batch, max_solves, warmup_t0, warmup_t1 and
    rep_rows (from timed_repeats). Flattened so every repeat keeps its own
    start/stop stamps."""
    flat = [(br, r) for br in batch_rows for r in br["rep_rows"]]
    np.savez(
        out_path,
        rep_batch=np.array([br["N_batch"] for br, r in flat]),
        rep_max_solves=np.array([br["max_solves"] for br, r in flat]),
        rep_index=np.array([r["rep"] for br, r in flat]),
        rep_start_epoch=np.array([r["t_start"] for br, r in flat]),
        rep_stop_epoch=np.array([r["t_stop"] for br, r in flat]),
        rep_start_iso=np.array([iso(r["t_start"]) for br, r in flat]),
        rep_stop_iso=np.array([iso(r["t_stop"]) for br, r in flat]),
        rep_wall=np.array([r["wall"] for br, r in flat]),
        rep_n_collected=np.array([r["n_collected"] for br, r in flat]),
        rep_throughput=np.array([r["throughput"] for br, r in flat]),
        warmup_batch=np.array([br["N_batch"] for br in batch_rows]),
        warmup_start_epoch=np.array([br["warmup_t0"] for br in batch_rows]),
        warmup_stop_epoch=np.array([br["warmup_t1"] for br in batch_rows]),
        warmup_start_iso=np.array([iso(br["warmup_t0"])
                                   for br in batch_rows]),
        warmup_stop_iso=np.array([iso(br["warmup_t1"])
                                  for br in batch_rows]),
        **{k: np.atleast_1d(np.asarray(v)) for k, v in meta.items()},
    )
    print(f"[VARIANCE] saved {out_path}", flush=True)

#!/usr/bin/env python
"""IPOPT default configuration (one IPOPT/MUMPS process using the whole CPU, cold solves),
tol 1e-8: throughput and cost mean ± sd over repeats.

Reads tests/*/logs/casadi_<tag>_tol1e-08_rep<r>_results.npz (paper pool sizes N_RUNS_seq,
R repeats) and casadi_<tag>_pool_tol1e-08_results.npz (one pass over the full jaxipm pool);
compares with the paper's casadi_<tag>_results.npz (tol 1e-6, single run).
Writes results/ipopt_default_repeats.md / .json.
"""
import glob, json, os, re
import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RES = os.path.join(REPO, "tests", "rebuttal", "results")
SC = [("nav 90°", "tests/quad_nav_circle/logs", "sector90"),
      ("nav 180°", "tests/quad_nav_circle/logs", "sector180"),
      ("track v̄=1.0", "tests/quad_track_avoid/logs", "v1.0"),
      ("track v̄=1.5", "tests/quad_track_avoid/logs", "v1.5"),
      ("track v̄=2.0", "tests/quad_track_avoid/logs", "v2.0"),
      ("multi N=2", "tests/quad_multi_swap/logs", "2"),
      ("multi N=4", "tests/quad_multi_swap/logs", "4")]


def load(f):
    d = np.load(f, allow_pickle=True)
    n = int(d["N_RUNS"][0]); tt = float(d["total_time"][0])
    obj = np.asarray(d["obj_vals"], float); ok = np.asarray(d["success"], bool) & np.isfinite(obj)
    return dict(n=n, total_time=tt, throughput=n / tt, mean_solve=float(np.mean(d["times"])),
                obj=obj[ok], success=float(ok.mean()), iters=float(np.mean(np.asarray(d["iters"])[ok])) if ok.any() else np.nan,
                load=(d["loadavg_at_save"].tolist() if "loadavg_at_save" in d.files else None),
                tol=(float(d["ipopt_tol"][0]) if "ipopt_tol" in d.files else 1e-6))


def ms(v, fmt="{:.2f}"):
    v = np.asarray(v, float)
    return (fmt + " ± " + fmt).format(v.mean(), v.std(ddof=1) if v.size > 1 else 0.0) if v.size else "–"


def main():
    out, lines = {}, ["# IPOPT default configuration (one IPOPT/MUMPS process, whole CPU, cold solves) — tol 1e-8, repeats", "",
                      "Same driver and pool sizes as the paper's IPOPT column (N_RUNS_seq = 80 nav / 50 track / 50 multi), "
                      "tol 1e-8, R independent repeats; throughput = N_RUNS / total wall. Host 5-min load average at save time "
                      "listed per repeat (other users' jobs were running). `pool` = one pass over the full jaxipm pool "
                      "(2000/1000/75) with inputs saved for the ‖c‖∞ column. Paper column = original tol 1e-6 single run.", "",
                      "| scenario | R | throughput mean ± sd [solves/s] | per-repeat | load (5 min) per repeat | mean solve [ms] | iters mean | cost mean ± sd (pooled over repeats) | success | pool pass: throughput / cost mean ± sd (n) | paper (1e-6): throughput / cost mean ± sd (n) |",
                      "|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, logs, tag in SC:
        reps = sorted(glob.glob(os.path.join(REPO, logs, f"casadi_{tag}_tol1e-08_rep*_results.npz")),
                      key=lambda f: int(re.findall(r"rep(\d+)", f)[0]))
        R = [load(f) for f in reps]
        pool = glob.glob(os.path.join(REPO, logs, f"casadi_{tag}_pool_tol1e-08_results.npz"))
        paper = glob.glob(os.path.join(REPO, logs, f"casadi_{tag}_results.npz"))
        P = load(pool[0]) if pool else None; Q = load(paper[0]) if paper else None
        thr = np.array([r["throughput"] for r in R]); cost_all = np.concatenate([r["obj"] for r in R]) if R else np.array([])
        row = dict(R=len(R), throughput=thr.tolist(), throughput_mean=float(thr.mean()) if thr.size else None,
                   throughput_sd=float(thr.std(ddof=1)) if thr.size > 1 else None,
                   cost_mean=float(cost_all.mean()) if cost_all.size else None,
                   cost_sd=float(cost_all.std(ddof=1)) if cost_all.size > 1 else None,
                   loads=[r["load"][1] if r["load"] else None for r in R],
                   pool=(dict(throughput=P["throughput"], cost_mean=float(P["obj"].mean()), cost_sd=float(P["obj"].std(ddof=1)), n=int(P["obj"].size)) if P else None),
                   paper=(dict(throughput=Q["throughput"], cost_mean=float(Q["obj"].mean()), cost_sd=float(Q["obj"].std(ddof=1)), n=int(Q["obj"].size), tol=Q["tol"]) if Q else None))
        out[name] = row
        lines.append(f"| {name} | {len(R)} | {ms(thr)} | " + ", ".join(f"{t:.2f}" for t in thr) + " | " +
                     ", ".join(f"{l:.0f}" if l is not None else "?" for l in row["loads"]) +
                     f" | {ms([r['mean_solve']*1e3 for r in R], '{:.0f}')} | {ms([r['iters'] for r in R], '{:.1f}')} | {ms(cost_all, '{:.3f}')} | " +
                     (f"{100*np.mean([r['success'] for r in R]):.1f}%" if R else "–") + " | " +
                     (f"{P['throughput']:.2f} / {P['obj'].mean():.3f} ± {P['obj'].std(ddof=1):.3f} ({P['obj'].size})" if P else "–") + " | " +
                     (f"{Q['throughput']:.2f} / {Q['obj'].mean():.3f} ± {Q['obj'].std(ddof=1):.3f} ({Q['obj'].size})" if Q else "–") + " |")
    os.makedirs(RES, exist_ok=True)
    with open(os.path.join(RES, "ipopt_default_repeats.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(RES, "ipopt_default_repeats.json"), "w") as f:
        json.dump(out, f, indent=1)
    print("\n".join(lines))


if __name__ == "__main__":
    main()

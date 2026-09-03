#!/usr/bin/env python
"""Run D repeats: IPOPT P=64 physical, tol 1e-8, R repeats per scenario -> mean ± sd.

Reads tests/*/logs/casadi_cpu_throughput_<tag>_phys_c064_tol1e-08_<set>_rep<r>_results.npz
(sets: 'now' = run under whatever host load there was, 'quiet' = run when the 5-min load
average was < 8) plus the single sweep file *_c064_tol1e-08_results.npz, and writes
results/D_p64_repeats.md / .json with throughput mean ± sd, per-repeat values and the
host load average stamped at save time (metadata.loadavg_at_save).
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
JAXIPM_A = {"nav 90°": (101.69, 2.96), "nav 180°": (105.82, 6.32), "track v̄=1.0": (35.75, 1.20),
            "track v̄=1.5": (31.60, 0.76), "track v̄=2.0": (30.94, 0.50), "multi N=2": (8.72, 0.28),
            "multi N=4": (1.35, 0.06)}


def load(f):
    d = np.load(f, allow_pickle=True)
    thr = float(d["throughput"][0]) if "throughput" in d.files else float(d["N_RUNS"][0] / d["wall"][0])
    succ = float(np.mean(d["success"])) if "success" in d.files else np.nan
    meta = json.loads(str(d["metadata"][0])) if "metadata" in d.files else {}
    return thr, succ, meta.get("loadavg_at_save")


def main():
    out, lines = {}, ["# Run D repeats — IPOPT, P = 64 physical cores, tol 1e-8 (mean ± sd over repeats)", "",
                      "Same protocol as the sweep (one pinned single-thread process per core, whole pool per repeat). "
                      "`now` = repeats run 2026-08-27 15:00 while other users loaded the host (5-min load ≈ 55); "
                      "`quiet` = repeats run when the 5-min load average dropped below 8. Host load average "
                      "(1/5/15 min) is stamped into every file at save time. jaxipm column = run A (ILB on, 5 fresh-process repeats).", ""]
    sets = ["now", "quiet"]
    hdr = "| scenario | sweep single run | " + " | ".join(f"{s}: mean ± sd (n) | {s}: per-repeat | {s}: load avg (5 min) at save" for s in sets) + " | jaxipm A | node/jaxipm (best set) |"
    lines += [hdr, "|---|---|" + "---|---|---|" * len(sets) + "---|---|"]
    for name, logs, tag in SC:
        row = {"scenario": name}
        single = glob.glob(os.path.join(REPO, logs, f"casadi_cpu_throughput_{tag}_phys_c064_tol1e-08_results.npz"))
        row["sweep_single"] = load(single[0])[0] if single else None
        cells = [f"| {name} | " + (f"{row['sweep_single']:.1f}" if single else "–")]
        best = None
        for s in sets:
            fs = sorted(glob.glob(os.path.join(REPO, logs, f"casadi_cpu_throughput_{tag}_phys_c064_tol1e-08_{s}_rep*_results.npz")),
                        key=lambda f: int(re.findall(r"rep(\d+)", f)[0]))
            vals = [load(f) for f in fs]
            thr = np.array([v[0] for v in vals]); loads = [v[2][1] if v[2] else None for v in vals]
            row[s] = dict(throughput=thr.tolist(), mean=float(thr.mean()) if thr.size else None,
                          sd=float(thr.std(ddof=1)) if thr.size > 1 else None,
                          success=[v[1] for v in vals], loadavg5=loads)
            if thr.size:
                cells.append(f"{thr.mean():.1f} ± {thr.std(ddof=1) if thr.size > 1 else 0:.1f} ({thr.size})")
                cells.append(", ".join(f"{t:.1f}" for t in thr))
                cells.append(", ".join(f"{l:.0f}" if l is not None else "?" for l in loads))
                if best is None or (s == "quiet"):
                    best = thr.mean()
            else:
                cells += ["–", "–", "–"]
        ja = JAXIPM_A[name]
        cells.append(f"{ja[0]:.2f} ± {ja[1]:.2f}")
        cells.append(f"{best/ja[0]:.2f}×" if best else "–")
        lines.append(" | ".join(cells) + " |")
        out[name] = row
    os.makedirs(RES, exist_ok=True)
    with open(os.path.join(RES, "D_p64_repeats.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(RES, "D_p64_repeats.json"), "w") as f:
        json.dump(out, f, indent=1)
    print("\n".join(lines))


if __name__ == "__main__":
    main()

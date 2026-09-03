"""Run E -- MadNLP on the jaxipm pool (tol 1e-8, 5 repeats): mean ± std solves/s.

Per-repeat walls are recovered from the saved stop stamps: the first two MadNLP
driver launches (nav, track) had the repeat loop placed after `t0 = time()`, so
their logged walls are cumulative; wall_k = stop_k - stop_{k-1} (wall_1 =
stop_1 - start_1) is exact for every file. Reports the paper protocol
(wall incl. per-instance ExaModel build) and, where saved, solve-only wall.
Run from the repo root:  python -m tests.rebuttal.analyze_E
"""
import glob, json, os
import numpy as np
RES = os.path.join(os.path.dirname(__file__), "results")
FILES = [("nav 90°", "tests/quad_nav_circle/logs/madnlp_sector90_pool_tol1e-8_results.npz"),
         ("nav 180°", "tests/quad_nav_circle/logs/madnlp_sector180_pool_tol1e-8_results.npz"),
         ("track v̄=1.0", "tests/quad_track_avoid/logs/madnlp_v1.0_pool_tol1e-8_results.npz"),
         ("track v̄=1.5", "tests/quad_track_avoid/logs/madnlp_v1.5_pool_tol1e-8_results.npz"),
         ("track v̄=2.0", "tests/quad_track_avoid/logs/madnlp_v2.0_pool_tol1e-8_results.npz"),
         ("multi N=2", "tests/quad_multi_swap/logs/madnlp_2_pool_tol1e-8_results.npz"),
         ("multi N=4", "tests/quad_multi_swap/logs/madnlp_4_pool_tol1e-8_results.npz")]
def main():
    os.makedirs(RES, exist_ok=True)
    md = ["# Run E — MadNLP (ExaModels + MadNLPGPU/cuDSS), jaxipm pool, tol 1e-8, 5 repeats\n",
          "| config | n | success | iters mean/max | wall per rep [s] | solves/s (paper protocol: build + solve) | solve-only solves/s | status codes |\n|---|---|---|---|---|---|---|---|"]
    out = {}
    for label, f in FILES:
        if not os.path.exists(f):
            md.append(f"| {label} | – | – | – | – | – | – | (missing) |"); continue
        z = np.load(f, allow_pickle=True); n = int(z["N_RUNS"][0])
        st, sp = np.asarray(z["rep_start_epoch"]), np.asarray(z["rep_stop_epoch"])
        walls = np.diff(np.concatenate([[st[0]], sp]))
        thr = n / walls
        so = (n / np.asarray(z["rep_solve_wall"])) if "rep_solve_wall" in z.files and np.asarray(z["rep_solve_wall"]).max() > 0 else None
        meta = json.load(open(f + ".meta.json")) if os.path.exists(f + ".meta.json") else {}
        codes, cnt = np.unique(z["status_codes"], return_counts=True)
        legend = meta.get("madnlp_status_legend", {})
        codes_s = ", ".join(f"{legend.get(str(int(c)), int(c))}:{int(k)}" for c, k in zip(codes, cnt))
        it = z["iters"]
        md.append(f"| {label} | {n} | {z['success'].mean():.4f} | {it.mean():.1f}/{it.max()} | "
                  f"{' / '.join(f'{w:.0f}' for w in walls)} | **{thr.mean():.3f} ± {thr.std(ddof=1):.3f}** | "
                  f"{(f'{so.mean():.3f} ± {so.std(ddof=1):.3f}') if so is not None else '–'} | {codes_s} |")
        out[label] = dict(n=n, walls=walls.tolist(), throughput=thr.tolist(), success=float(z["success"].mean()),
                          solve_only=so.tolist() if so is not None else None)
    open(os.path.join(RES, "E_madnlp.md"), "w").write("\n".join(md) + "\n")
    json.dump(out, open(os.path.join(RES, "E_madnlp.json"), "w"), indent=1)
    print("\n".join(md))
if __name__ == "__main__":
    main()

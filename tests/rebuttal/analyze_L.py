"""Run L -- order statistics vs the measured ILB gain (A / B).

Predicted ratio  = E[max_{i<=N} K_i] / E[K]  from the per-problem iteration
distribution K of the jaxipm DEBUG census runs (exact from the empirical CDF).
Measured ratio   = A throughput (ILB on, fresh-JIT reps) / B throughput (ILB off).
Also reports the fused-iteration ratio directly (WS fused iterations summed over
batches / HR fused iterations), which is what the order statistic actually
predicts; the throughput ratio additionally contains per-iteration cost changes.

Run from the repo root:  python -m tests.rebuttal.analyze_L
"""
import glob, json, os
import numpy as np
HERE = os.path.dirname(__file__); RES = os.path.join(HERE, "results")
SCEN = [("nav", "tests/quad_nav_circle/logs", [("sector90", 500), ("sector180", 500)]),
        ("track", "tests/quad_track_avoid/logs", [("v1.0", 250), ("v1.5", 250), ("v2.0", 250)]),
        ("multi", "tests/quad_multi_swap/logs", [("2", 75), ("4", 75)])]

def emax_ratio(K, N):
    K = np.asarray(K, float); K = K[K >= 0]
    vals, counts = np.unique(K, return_counts=True); F = np.cumsum(counts) / K.size
    Fm1 = np.concatenate([[0.0], F[:-1]]); emax = float(np.sum(vals * (F**N - Fm1**N)))
    return emax / K.mean(), emax, K.mean()

def thr(paths, key="rep_throughput"):
    v = np.concatenate([np.asarray(np.load(p, allow_pickle=True)[key]).reshape(-1) for p in paths]) if paths else np.array([])
    return v

def main():
    os.makedirs(RES, exist_ok=True)
    md = ["# Run L — order statistics vs measured ILB gain\n",
          "| scenario | N | E[K] (source) | E[max_N K] | predicted ratio | predicted incl. cap hits | fused-iter ratio WS/HR (dbg) | A solves/s | B solves/s (in-proc) | B fresh | measured A/B |\n|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, logs, variants in SCEN:
        for tag, N in variants:
            A = thr(sorted(glob.glob(f"{logs}/jaxipm_variance_{tag}_fresh_rep*_results.npz")))
            B = thr(sorted(glob.glob(f"{logs}/jaxipm_ablation_{tag}_ws_ir0_b{N}_results.npz")))
            Bf = thr(sorted(glob.glob(f"{logs}/jaxipm_ablation_{tag}_ws_ir0_b{N}_fresh_rep*_results.npz")))
            hr_dbg = sorted(glob.glob(f"{logs}/jaxipm_ablation_{tag}_hr_ir0_b{N}_dbg_results.npz"))
            ws_dbg = sorted(glob.glob(f"{logs}/jaxipm_ablation_{tag}_ws_ir0_b{N}_dbg_results.npz"))
            src, pred, em, ek = "–", float("nan"), float("nan"), float("nan")
            if hr_dbg:
                Z = np.load(hr_dbg[0], allow_pickle=True); pred, em, ek = emax_ratio(Z["iters"], N); src = "jaxipm HR dbg"
            pred_cap = "–"
            if hr_dbg and ws_dbg:
                Zw = np.load(ws_dbg[0], allow_pickle=True)
                th = Zw["term_hist"]; n_cap = int(th[2]) if th.size >= 3 else 0
                K = np.asarray(np.load(hr_dbg[0], allow_pickle=True)["iters"], float)
                Kc = np.concatenate([K, np.full(n_cap, 500.0)])   # WS census: cap-hitters as K = 500
                r_c, _, _ = emax_ratio(Kc, N)
                Kw = np.concatenate([np.asarray(Zw["iters"], float), np.full(n_cap, 500.0)])  # WS census K (all pool instances once)
                r_w, _, ekw = emax_ratio(Kw, N)
                pred_cap = f"{r_c:.2f} (+{n_cap} cap) / WS-K: {r_w:.2f} (E[K]={ekw:.1f})"
            fr = "–"
            if hr_dbg and ws_dbg:
                nh = np.asarray(np.load(hr_dbg[0], allow_pickle=True)["n_fused_iters"]).sum()
                nw = np.asarray(np.load(ws_dbg[0], allow_pickle=True)["n_fused_iters"]).sum()
                fr = f"{nw / max(nh, 1):.2f} ({nw}/{nh})"
            f = lambda v: f"{v.mean():.2f} ± {v.std(ddof=1) if v.size > 1 else 0:.2f} (n={v.size})" if v.size else "–"
            meas = f"{A.mean() / B.mean():.2f}" if A.size and B.size else "–"
            md.append(f"| {name} {tag} | {N} | {ek:.1f} ({src}) | {em:.1f} | {pred:.2f} | {pred_cap} | {fr} | {f(A)} | {f(B)} | {f(Bf)} | {meas} |")
    open(os.path.join(RES, "L_orderstats_vs_measured.md"), "w").write("\n".join(md) + "\n")
    print("\n".join(md))

if __name__ == "__main__":
    main()

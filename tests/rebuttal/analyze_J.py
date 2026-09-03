#!/usr/bin/env python
"""Run J analysis: per-step injection correctness across nav pool instances.

Reads tests/correctness/logs/jaxipm_correctness[_inst<k>].npz (per-step deviation of
the jaxipm step from IPOPT's, seeded from IPOPT's exact state; jx_branch = branch
jaxipm's fused body took at that step) and writes
  results/J_correctness.md, figures/J_step_ecdf.{pdf,png}
Caveat (John, 2026-08-27): the comparison is stochastic — poorly conditioned KKT
solves (cuDSS static pivoting) make some steps differ run-to-run; report the spread.
"""
import glob, os, re
import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
LOGS = os.path.join(REPO, "tests", "correctness", "logs")
RES = os.path.join(REPO, "tests", "rebuttal", "results")
FIG = os.path.join(REPO, "tests", "rebuttal", "figures")
BR = {0: "full step", 1: "SOC", 2: "watchdog", 3: "tiny step", 4: "SFR", 5: "resto entry",
      6: "backtracked", 7: "init", -1: "n/a"}
COL = {0: "#0072B2", 1: "#E69F00", 2: "#56B4E9", 3: "#009E73", 4: "#F0E442", 5: "#D55E00",
       6: "#CC79A7", 7: "#999999", -1: "#000000"}
X_MATCH, X_DRIFT = 1e-6, 1e-4


def main():
    files = sorted(glob.glob(os.path.join(LOGS, "jaxipm_correctness*.npz")),
                   key=lambda f: (0 if "inst" not in f else 1, int(re.findall(r"inst(\d+)", f)[0]) if "inst" in f else -1))
    rows, allx, allb, allr = [], [], [], []
    for f in files:
        d = np.load(f, allow_pickle=True)
        inst = re.findall(r"inst(\d+)", f)
        inst = inst[0] if inst else "orig"
        dx = np.asarray(d["dx"], float); n = dx.size
        br = np.asarray(d["jx_branch"]) if "jx_branch" in d.files else np.full(n, -1)
        ipr, jxr = np.asarray(d["ip_resto"]), np.asarray(d["jx_resto"])
        resto_agree = float(np.mean(ipr[:n] == jxr[:n])) if n else np.nan
        base = br % 8; base[br < 0] = -1
        cnt = {BR[k]: int(np.sum(base == k)) for k in sorted(set(base.tolist()))}
        first_tight = next((k for k in range(n) if not (dx[k] <= X_MATCH)), None)
        first_drift = next((k for k in range(n) if not (dx[k] <= X_DRIFT)), None)
        rows.append(dict(inst=inst, n_ipopt=int(d["n_ipopt_iter"][0]), n_cmp=n,
                         med=float(np.nanmedian(dx)), p90=float(np.nanpercentile(dx, 90)),
                         mx=float(np.nanmax(dx)), tight=float(np.mean(dx <= X_MATCH)),
                         drift=float(np.mean(dx > X_DRIFT)), first_tight=first_tight,
                         first_drift=first_drift, resto_agree=resto_agree, cnt=cnt,
                         term=int(d["term"][0]), in_resto_frac=float(np.mean(base >= 0) and np.mean(br >= 8))))
        allx.append(dx); allb.append(base); allr.append(br)
    lines = ["# Run J — per-step injection correctness across nav-circle pool instances", "",
             "jaxipm is seeded from IPOPT's exact internal state at every iteration k, takes one step, "
             "and Δx = ‖x_jaxipm(k+1) − x_IPOPT(k+1)‖_∞ is recorded (ir_nsteps=100, VALIDATION_MODE). "
             "Branch = the branch jaxipm's fused body selected for that step (DEBUG_MODE branch id). "
             f"tight: Δx ≤ {X_MATCH:.0e}; drift: Δx > {X_DRIFT:.0e}. "
             "Caveat: the KKT solves are poorly conditioned on some steps, so individual Δx values "
             "vary run-to-run (cuDSS static pivoting is not deterministic); the tight fraction and the "
             "median are the stable statistics.", "",
             "| instance | IPOPT iters | steps compared | median Δx | p90 Δx | max Δx | tight frac | drift frac | first tight-miss | first drift | resto-flag agreement | branches (jaxipm) |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['inst']} | {r['n_ipopt']} | {r['n_cmp']} | {r['med']:.1e} | {r['p90']:.1e} | {r['mx']:.1e} | "
                     f"{100*r['tight']:.1f}% | {100*r['drift']:.1f}% | {r['first_tight']} | {r['first_drift']} | "
                     f"{100*r['resto_agree']:.0f}% | " + ", ".join(f"{k} {v}" for k, v in r['cnt'].items()) + " |")
    if rows:
        X = np.concatenate(allx); B = np.concatenate(allb)
        lines += ["", f"**All instances pooled:** {X.size} steps, median Δx {np.nanmedian(X):.1e}, "
                  f"tight {100*np.mean(X <= X_MATCH):.1f}%, drift {100*np.mean(X > X_DRIFT):.1f}%.", "",
                  "| branch | steps | median Δx | tight frac | drift frac |", "|---|---|---|---|---|"]
        for k in sorted(set(B.tolist())):
            m = B == k
            lines.append(f"| {BR[k]} | {m.sum()} | {np.nanmedian(X[m]):.1e} | {100*np.mean(X[m] <= X_MATCH):.1f}% | {100*np.mean(X[m] > X_DRIFT):.1f}% |")
    os.makedirs(RES, exist_ok=True)
    with open(os.path.join(RES, "J_correctness.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    if not rows:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    X = np.concatenate(allx); B = np.concatenate(allb)
    ok = np.isfinite(X) & (X > 0)
    xs = np.sort(X[ok]); ys = np.arange(1, xs.size + 1) / xs.size
    ax.plot(xs, ys, color="k", lw=1, zorder=1)
    order = np.argsort(X[ok]); Bo = B[ok][order]
    for k in sorted(set(Bo.tolist())):
        m = Bo == k
        ax.scatter(xs[m], ys[m], s=9, color=COL[k], label=f"{BR[k]} ({m.sum()})", zorder=2)
    ax.axvline(X_MATCH, ls=":", color="gray", lw=0.8); ax.axvline(X_DRIFT, ls="--", color="gray", lw=0.8)
    ax.set_xscale("log"); ax.set_xlabel("single-step Δx (∞-norm) vs IPOPT"); ax.set_ylabel("ECDF")
    ax.set_title(f"{len(rows)} nav instances, {xs.size} injected steps", fontsize=8)
    ax.legend(fontsize=6, loc="upper left")
    fig.tight_layout(); os.makedirs(FIG, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG, f"J_step_ecdf.{ext}"), dpi=200)
    print("figure -> tests/rebuttal/figures/J_step_ecdf.pdf")


if __name__ == "__main__":
    main()

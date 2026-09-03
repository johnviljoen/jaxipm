#!/usr/bin/env python
"""Paper Figure 6 (validation_state_ecdf.pdf) re-rendered from the ORIGINAL data with a legend.

Provenance of the paper figure (byte-identical md5 71ff849a...):
  script : /home/john/code/jaxipm/src/problems/correctness_test/plot_validation_state_diffs.py  (2026-06-14)
  data   : /home/john/code/jaxipm/data/validation_runs/run_2026060{9,10}_*/iter_<k>.eqx  (5 runs x 93 iters,
           per-step injection from IPOPT's exact state, ir_nsteps=100, git fd6c048 / 4ec84e8)
  metric : element-wise isclose score max|a-b|/(1+|b|) over every compared leaf of the full state pair
           (analyze_diff gating), last 10% of iterations dropped -> 5 x 84 = 420 samples.
The samples (deviation + per-step branch flags read from the archived jaxipm/IPOPT state pairs) are
re-collected by fig6_collect_paper_samples.py (run in the `jaxipm` env) into
results/fig6_paper_samples.npz, which this script plots.

Two outputs in tests/correctness/figures/:
  validation_state_ecdf_paper_legend.pdf    identical points/colours to the paper, legend added
  validation_state_ecdf_paper_branches.pdf  same data, categories split by branch (release DEBUG
                                            branch_id semantics: SOC = SOC loop ran and its trial was
                                            accepted; backtracking = ls.n_steps > 0 after the step)
"""
import os, sys, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
FIG = os.path.join(REPO, "tests", "correctness", "figures")
NPZ = os.path.join(REPO, "tests", "rebuttal", "results", "fig6_paper_samples.npz")
FLOOR = 1e-18
sys.path.insert(0, "/home/john/code/jaxipm/src/problems")
try:
    from _paper_fonts import apply_paper_fonts, PDF_METADATA
except Exception:  # fallback: plain matplotlib
    apply_paper_fonts = None; PDF_METADATA = None
matplotlib.rcParams["font.family"] = "serif"
matplotlib.rcParams["font.serif"] = ["Palatino", "P052", "Palatino Linotype", "URW Palladio L", "DejaVu Serif"]
matplotlib.rcParams["mathtext.fontset"] = "dejavuserif"

PAPER_COLORS = [("regular", "tab:blue"), ("restoration", "tab:red"), ("restoration entry", "tab:green"),
                ("restoration exit", "tab:purple"), ("second order correction", "tab:orange")]
BRANCH_COLORS = [("full step", "#0072B2"), ("full step (resto)", "#E69F00"),
                 ("second-order correction", "#D55E00"), ("backtracking", "#CC79A7"),
                 ("watchdog", "#56B4E9"), ("tiny step", "#000000"), ("soft feasibility restoration", "#8B4513"),
                 ("restoration entry", "#009E73"), ("restoration exit", "tab:purple")]


def branch_cats(d):
    """Release branch semantics applied to the archived post-step state (priority SFR > TS > WD > entry > SOC > BT > full)."""
    out = []
    for i in range(d["dev"].size):
        if d["exiting"][i]:
            c = "restoration exit"
        elif d["jx_sfr"][i] > 0:
            c = "soft feasibility restoration"
        elif d["jx_ts_flag"][i] > 0 or d["jx_ts_last"][i] > 0:
            c = "tiny step"
        elif d["jx_in_wd"][i] > 0 or d["jx_wd_trial"][i] > 0:
            c = "watchdog"
        elif d["entering"][i]:
            c = "restoration entry"
        elif d["soc_ran"][i] and d["jx_n_steps"][i] == 0:
            c = "second-order correction"
        elif d["jx_n_steps"][i] > 0:
            c = "backtracking"
        else:
            c = "full step (resto)" if d["resto"][i] else "full step"
        out.append(c)
    return np.array(out)


def render(dev, cats, colors, out_path, legend_loc="lower right"):
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    keep = np.isfinite(dev)
    v = np.where(dev[keep] <= 0, FLOOR, dev[keep])
    order = np.argsort(v, kind="stable")
    x = v[order]; y = np.arange(1, x.size + 1) / x.size; cs = cats[keep][order]
    ax.semilogx(x, y, color="0.6", lw=1.2, drawstyle="steps-post", zorder=1)
    for cat, color in colors:
        sel = cs == cat
        if sel.any():
            ax.semilogx(x[sel], y[sel], ls="none", marker="o", ms=4, color=color, zorder=2,
                        label=f"{cat} ({int(sel.sum())})")
    ax.set_xlabel("scaled single-step deviation"); ax.set_ylabel("empirical CDF")
    ax.set_xlim(x.min() / 2, x.max() * 2); ax.set_ylim(0.0, 1.02)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.grid(True, which="both", axis="x", alpha=0.25); ax.grid(True, which="major", axis="y", alpha=0.15)
    ax.legend(loc=legend_loc, frameon=True, facecolor="white", edgecolor="black", framealpha=1.0, fancybox=False)
    if apply_paper_fonts is not None:
        apply_paper_fonts(fig, frac=1.0, tight=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    fig.savefig(os.path.splitext(out_path)[0] + ".pdf", metadata=PDF_METADATA)
    print("saved", out_path, {c: int((cs == c).sum()) for c, _ in colors if (cs == c).any()})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default=NPZ)
    args = ap.parse_args()
    d = np.load(args.npz)
    dev = np.asarray(d["dev"], float)
    os.makedirs(FIG, exist_ok=True)
    render(dev, np.asarray(d["cat"]), PAPER_COLORS, os.path.join(FIG, "validation_state_ecdf_paper_legend.png"))
    render(dev, branch_cats(d), BRANCH_COLORS, os.path.join(FIG, "validation_state_ecdf_paper_branches.png"))


if __name__ == "__main__":
    main()

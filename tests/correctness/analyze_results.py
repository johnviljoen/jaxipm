"""Paper figure for correctness: the single-step state-deviation ECDF.

Replicates the plotting of ``validation_state_ecdf.pdf`` from the jaxipm paper
codebase (correctness_test/plot_validation_state_diffs.py): an empirical CDF of
the per-step jaxipm-vs-IPOPT full-state deviation, each sample coloured by the
kind of IPM step it was (regular / restoration / the entry & exit boundary
passes). Exact-zero (bit-for-bit) steps clamp to the left edge of the log axis.

DATA-SOURCE NOTE. The paper figure aggregates MANY validation runs from the mk3
eqx state-pair archive (data/validation_runs/*.eqx + analyze_diff), neither of
which ships in this release. This script instead sources the SAME ECDF from the
single archived correctness run that jaxipm_correctness.py produces
(logs/jaxipm_correctness.npz): one sample per IPM iteration, the deviation being
the max over every saved state component (d_x, d_s, d_y_*, d_z_*, d_v_*, d_mu,
d_tau). That is the release-available analog of "max over every compared leaf".
Two consequences vs. the paper figure: (1) the metric is the raw inf-norm diff
(the paper's --metric abs), since the release npz carries no isclose scaling;
(2) it is one run, not the multi-run aggregate.

Self-contained: its own apply_paper_fonts is embedded below — no shared helper.

Run from the repo root:
    python -m tests.correctness.analyze_results
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
from matplotlib.legend import Legend

# Palatino for all text (URW "P052" clone is used when Palatino proper is absent).
matplotlib.rcParams["font.family"] = "serif"
matplotlib.rcParams["font.serif"] = ["Palatino", "P052", "Palatino Linotype",
                                     "URW Palladio L", "DejaVu Serif"]
matplotlib.rcParams["mathtext.fontset"] = "dejavuserif"
# Embed TrueType (Type 42) fonts in PDF output (Type 3 breaks Overleaf/PDF-A).
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

# ── Embedded paper-font scheme (own copy; see jaxipm _paper_fonts.py) ─────────
RHO = {"label": 2.10, "title": 2.10, "tick": 1.80, "legend": 1.80}
TIGHT_PAD_IN = 0.1
PDF_METADATA = {"Creator": None, "Producer": None, "CreationDate": None}


def apply_paper_fonts(fig, frac, tight=True, iters=3, verbose=True):
    """Set label/title/tick/legend sizes for LaTeX inclusion at frac*linewidth."""
    sizes = None
    for _ in range(max(1, iters if tight else 1)):
        if tight:
            fig.canvas.draw()
            w_in = (fig.get_tightbbox(fig.canvas.get_renderer()).width
                    + 2 * TIGHT_PAD_IN)
        else:
            w_in = float(fig.get_size_inches()[0])
        sizes = {k: v * w_in / frac for k, v in RHO.items()}
        for ax in fig.get_axes():
            ax.xaxis.label.set_size(sizes["label"])
            ax.yaxis.label.set_size(sizes["label"])
            ax.title.set_size(sizes["title"])
            for name in ("xaxis", "yaxis", "zaxis"):
                axis = getattr(ax, name, None)
                if axis is not None:
                    axis.set_tick_params(labelsize=sizes["tick"])
                    if name == "zaxis":
                        ax.zaxis.label.set_size(sizes["label"])
            for leg in [a for a in ax.get_children() if isinstance(a, Legend)]:
                for t in leg.get_texts():
                    t.set_fontsize(sizes["legend"])
    if verbose:
        print(f"[paper_fonts] frac={frac}  saved-width={w_in:.2f}in  "
              + "  ".join(f"{k}={v:.1f}pt" for k, v in sizes.items()))
    return sizes


# ── Paths / config ───────────────────────────────────────────────────────────
HERE = os.path.dirname(os.path.abspath(__file__))
LOGS = os.path.join(HERE, "logs")
FIGURES = os.path.join(HERE, "figures")
os.makedirs(FIGURES, exist_ok=True)

NPZ = os.path.join(LOGS, os.environ.get("CORR_NPZ", "jaxipm_correctness.npz"))
OUT_PATH = os.path.join(FIGURES, os.environ.get("CORR_FIG", "validation_state_ecdf.pdf"))

# State components saved per iteration in jaxipm_correctness.npz (d_<comp>).
COMPONENTS = ["x", "s", "y_c", "y_d", "z_L", "z_U", "v_L", "v_U", "mu", "tau"]

FLOOR = 1e-18          # log-axis floor: exact zeros show at the left edge
TAIL_FRAC = 0.10       # drop the last 10% of iters (near-convergence ill-cond.)
MASK_RESTO_EXIT = True  # exclude the resto-EXIT boundary iter (pipeline offset;
                        # matches the mk3 viewer's gating of it.* at that pass)

BRANCH_COLORS = None  # set in main() when branch ids are available
CAT_COLORS = [
    ("regular", "tab:blue"),
    ("restoration", "tab:red"),
    ("restoration entry", "tab:green"),
    ("restoration exit", "tab:purple"),
]


def categorize(jx_resto):
    """Per-iter step category from the jaxipm in_restoration timeline."""
    n = jx_resto.size
    cats = []
    for k in range(n):
        if jx_resto[k] == 1:
            entry = (k == 0) or (jx_resto[k - 1] == 0)
            exit_ = (k == n - 1) or (jx_resto[k + 1] == 0)
            if entry:
                cats.append("restoration entry")
            elif exit_:
                cats.append("restoration exit")
            else:
                cats.append("restoration")
        else:
            cats.append("regular")
    return np.array(cats)


def main():
    if not os.path.exists(NPZ):
        raise SystemExit(f"{NPZ} not found "
                         "(run `python -m tests.correctness.jaxipm_correctness` first)")
    d = np.load(NPZ)

    # Full-state single-step deviation per iter = max over saved components.
    comp = np.vstack([np.asarray(d[f"d_{c}"], float) for c in COMPONENTS])  # (C, n)
    dev = np.nanmax(comp, axis=0)                                            # (n,)
    jx_resto = np.asarray(d["jx_resto"]).astype(int)
    cats = categorize(jx_resto)
    # Branch-type legend (paper Figure 6): use the per-step branch id when the npz
    # was produced with DEBUG_MODE (CORR_DEBUG=1): 0 full step, 1 SOC, 2 watchdog,
    # 3 tiny step, 4 SFR, 5 restoration entry, 6 backtracked; +8 = inside restoration.
    if "jx_branch" in d.files and np.any(np.asarray(d["jx_branch"]) >= 0):
        br = np.asarray(d["jx_branch"]).astype(int)
        base = np.where(br >= 0, br % 8, -1)
        names = {0: "full step", 1: "second-order correction", 2: "watchdog", 3: "tiny step",
                 4: "soft feasibility restoration", 5: "restoration entry", 6: "backtracking"}
        cats = np.array([("restoration exit" if c == "restoration exit" else
                          ("initial point" if b < 0 else
                           names.get(b, "regular") + (" (resto)" if r == 1 and b in (0, 1, 6) else "")))
                         for c, b, r in zip(cats, base, jx_resto)])
        global BRANCH_COLORS
        BRANCH_COLORS = [("full step", "#0072B2"), ("full step (resto)", "#E69F00"),
                         ("second-order correction", "#D55E00"), ("second-order correction (resto)", "#F0E442"),
                         ("backtracking", "#CC79A7"), ("backtracking (resto)", "#999999"),
                         ("watchdog", "#56B4E9"), ("tiny step", "#000000"),
                         ("soft feasibility restoration", "#8B4513"), ("restoration entry", "#009E73"),
                         ("restoration exit", "tab:purple"), ("initial point", "0.5")]

    n = dev.size
    keep = np.isfinite(dev)
    # Drop the near-convergence tail (degraded KKT conditioning).
    n_keep = max(1, int(round((1.0 - TAIL_FRAC) * n)))
    tail = np.arange(n) >= n_keep
    keep &= ~tail
    # Mask the resto-exit boundary iter (known pipeline-stage offset).
    if MASK_RESTO_EXIT:
        keep &= cats != "restoration exit"

    dev_k = dev[keep]
    cats_k = cats[keep]
    if dev_k.size == 0:
        raise SystemExit("no samples left after masking; check the npz")

    # ── ECDF (one combined curve; markers coloured by step category) ─────────
    v = np.where(dev_k <= 0, FLOOR, dev_k)
    order = np.argsort(v, kind="stable")
    x = v[order]
    y = np.arange(1, x.size + 1) / x.size
    cat_sorted = cats_k[order]

    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    ax.semilogx(x, y, color="0.6", lw=1.2, drawstyle="steps-post", zorder=1)
    base_colors = BRANCH_COLORS if BRANCH_COLORS is not None else list(CAT_COLORS)
    known = [c for c, _ in base_colors]
    extra = [c for c in dict.fromkeys(cat_sorted.tolist()) if c not in known]
    palette = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00",
               "#F0E442", "#999999", "#000000"]
    cat_colors = list(base_colors) + [(c, palette[i % len(palette)]) for i, c in enumerate(extra)]
    for cat, color in cat_colors:
        sel = cat_sorted == cat
        if sel.any():
            ax.semilogx(x[sel], y[sel], ls="none", marker="o", ms=4,
                        color=color, zorder=2, label=f"{cat} ({int(sel.sum())})")

    ax.set_xlabel("single-step deviation")
    ax.set_ylabel("empirical CDF")
    ax.set_xlim(x.min() / 2, x.max() * 2)
    ax.set_ylim(0.0, 1.02)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, which="both", axis="x", alpha=0.25)
    ax.grid(True, which="major", axis="y", alpha=0.15)
    ax.legend(loc="lower right")

    # Console digest: bit-for-bit fraction + the worst few samples.
    bitfrac = float(np.mean(dev_k <= 0))
    print(f"samples: {dev_k.size}  bit-for-bit fraction: {bitfrac:.3f}")
    worst = np.argsort(-dev_k)[:8]
    print("worst samples (deviation, category):")
    for i in worst:
        print(f"  {dev_k[i]:11.3e}  {cats_k[i]}")

    apply_paper_fonts(fig, frac=1.0, tight=False)
    fig.tight_layout()
    fig.savefig(OUT_PATH, metadata=PDF_METADATA)
    plt.close(fig)
    print(f"saved {OUT_PATH}")


if __name__ == "__main__":
    main()

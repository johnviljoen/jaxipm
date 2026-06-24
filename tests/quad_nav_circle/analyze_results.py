"""Paper figure for quad_nav_circle: the 2x2 fan + iteration-count panel.

Replicates ``jaxipm_fan_iters_2x2.pdf`` from the jaxipm paper codebase
(quadcopter_nmpc_nav_circle/plot_fan_iters_2x2.py): top row = jaxipm start-fan
plots for the 90 and 180 degree sectors, bottom row = IPM iteration-count
histograms. jaxipm hot-restart results only (the paper's headline configuration).

Reads logs/jaxipm_sector{90,180}_results.npz (produced by jaxipm_quad_nav.py)
and writes figures/jaxipm_fan_iters_2x2.pdf. Obstacle geometry is loaded from the
shared test_params.json. Self-contained: its own apply_paper_fonts is embedded
below — no shared helper module.

Run from the repo root:
    python -m tests.quad_nav_circle.analyze_results
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.ticker import MultipleLocator
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
# Scrub identifying PDF metadata for double-blind submission.
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

FILES = {
    90:  os.path.join(LOGS, "jaxipm_sector90_results.npz"),
    180: os.path.join(LOGS, "jaxipm_sector180_results.npz"),
}
OUT_PATH = os.path.join(FIGURES, "jaxipm_fan_iters_2x2.pdf")

COLOR = "tab:green"                              # jaxipm color
FAN_HIST_COLOR = {90: "#1f77b4", 180: "#d62728"}
N_TRAJ = 80

# Obstacles from the shared test_params.json (bare cylinder radius for drawing).
with open(os.path.join(HERE, "test_params.json")) as _f:
    _obs = json.load(_f)["obstacles"]
OBS_XC = [o["xc"] for o in _obs]
OBS_YC = [o["yc"] for o in _obs]
OBS_R = [o["r"] for o in _obs]


# ── Helpers ──────────────────────────────────────────────────────────────────
def _despine(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def add_dir_arrows_2d(ax, xs, ys, color, n_arrows=3, mutation_scale=13, alpha=1.0):
    """Overlay direction arrowheads ON a 2D polyline (heads hug the curve)."""
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    n = xs.size
    if n < 3:
        return
    arc = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(xs), np.diff(ys)))])
    if arc[-1] <= 0:
        return
    eps = 0.005 * arc[-1]
    targets = arc[-1] * (np.arange(1, n_arrows + 1) / (n_arrows + 1.0))
    for t in targets:
        i = int(np.clip(np.searchsorted(arc, t), 1, n - 2))
        dx, dy = xs[i + 1] - xs[i - 1], ys[i + 1] - ys[i - 1]
        nrm = np.hypot(dx, dy)
        if nrm < 1e-12:
            continue
        dx, dy = dx / nrm, dy / nrm
        ax.annotate("", xy=(xs[i] + eps * dx, ys[i] + eps * dy),
                    xytext=(xs[i] - eps * dx, ys[i] - eps * dy),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=0, alpha=alpha,
                                    mutation_scale=mutation_scale, shrinkA=0, shrinkB=0),
                    zorder=6)


def load(path):
    d = np.load(path)
    return {k: np.asarray(d[k]) for k in d.keys()}


def draw_fan(ax, r, sec_deg):
    ax.set_aspect("equal")
    starts = r["starts"]
    ang_starts = np.arctan2(starts[:, 1], starts[:, 0])
    if "all_angles" in r:
        sec_lo_s, sec_hi_s = float(r["all_angles"].min()), float(r["all_angles"].max())
    else:
        sec_lo_s, sec_hi_s = float(ang_starts.min()), float(ang_starts.max())

    n_dots = min(N_TRAJ, starts.shape[0])
    dot_edges = np.linspace(sec_lo_s, sec_hi_s, n_dots + 1)
    dot_idxs = []
    for i in range(n_dots):
        lo, hi = dot_edges[i], dot_edges[i + 1]
        mask = (ang_starts >= lo) & (ang_starts <= hi) if i == n_dots - 1 \
               else (ang_starts >= lo) & (ang_starts < hi)
        cands = np.where(mask)[0]
        if cands.size:
            dot_idxs.append(int(cands[0]))
    dot_idxs = np.array(dot_idxs, dtype=int)
    ax.plot(starts[dot_idxs, 0], starts[dot_idxs, 1], ".",
            color="lightgray", markersize=2, label="starts")

    kept = []
    succ_idxs = np.where(r["success"])[0]
    if succ_idxs.size:
        ang_succ = r["angles"][succ_idxs] if "angles" in r \
                   else np.arctan2(r["X_all"][succ_idxs, 0, 1], r["X_all"][succ_idxs, 0, 0])
        sec_lo, sec_hi = (float(r["all_angles"].min()), float(r["all_angles"].max())) \
                        if "all_angles" in r else (float(ang_succ.min()), float(ang_succ.max()))
        bin_edges = np.linspace(sec_lo, sec_hi, N_TRAJ + 1)
        for i in range(N_TRAJ):
            lo, hi = bin_edges[i], bin_edges[i + 1]
            mask = (ang_succ >= lo) & (ang_succ <= hi) if i == N_TRAJ - 1 \
                   else (ang_succ >= lo) & (ang_succ < hi)
            cands = succ_idxs[mask]
            if cands.size:
                traj = r["X_all"][int(cands[0])]
                kept.append(traj)
                ax.plot(traj[:, 0], traj[:, 1], "-", color=COLOR, lw=0.6, alpha=0.4)

    # One faint midpoint arrowhead per plotted trajectory.
    for traj in kept:
        add_dir_arrows_2d(ax, traj[:, 0], traj[:, 1], color=COLOR,
                          n_arrows=1, alpha=0.4, mutation_scale=16)

    for xc, yc, rad in zip(OBS_XC, OBS_YC, OBS_R):
        ax.add_patch(Circle((xc, yc), rad, color="red", alpha=0.25))
    ax.plot(0, 0, "kx", markersize=10, label="goal")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title(f"jaxipm — {sec_deg}° sector")
    ax.legend(loc="upper right")
    _despine(ax)


def draw_iter_hist(ax, r, sec_deg, bins):
    color = FAN_HIST_COLOR[sec_deg]
    ax.hist(r["iters"], bins=bins, histtype="stepfilled",
            color=color, alpha=0.55, edgecolor=color, linewidth=1.2)
    ax.set_xlabel("IPM iterations")
    ax.set_ylabel("count")
    ax.xaxis.set_major_locator(MultipleLocator(40))
    _despine(ax)


def main():
    missing = [p for p in FILES.values() if not os.path.exists(p)]
    if missing:
        raise SystemExit("missing jaxipm sector results: " + ", ".join(missing)
                         + "\n(run `python -m tests.quad_nav_circle.jaxipm_quad_nav` first)")
    data = {sd: load(p) for sd, p in FILES.items()}

    i_min = min(int(d["iters"].min()) for d in data.values())
    i_max = max(int(d["iters"].max()) for d in data.values())
    hist_bins = np.arange(i_min, i_max + 2) - 0.5

    fig, axes = plt.subplots(2, 2, figsize=(13, 11))
    draw_fan(axes[0, 0], data[90], 90)
    draw_fan(axes[0, 1], data[180], 180)
    draw_iter_hist(axes[1, 0], data[90], 90, hist_bins)
    draw_iter_hist(axes[1, 1], data[180], 180, hist_bins)

    # Share x/y ranges across the two histograms for direct comparison.
    xmin, xmax = i_min - 0.5, i_max + 0.5
    ymax = max(axes[1, 0].get_ylim()[1], axes[1, 1].get_ylim()[1])
    for ax in (axes[1, 0], axes[1, 1]):
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(0, ymax)

    # Fonts first (tight=False: sizes depend only on figsize), layout second.
    apply_paper_fonts(fig, frac=1.0, tight=False)
    fig.tight_layout()

    # Equal-aspect fan plots are narrower than their cell; match each bottom
    # histogram's horizontal extent to the fan plot above it.
    fig.canvas.draw()
    for col in range(2):
        top_pos = axes[0, col].get_position()
        bot_pos = axes[1, col].get_position()
        axes[1, col].set_position([top_pos.x0, bot_pos.y0, top_pos.width, bot_pos.height])

    fig.savefig(OUT_PATH, metadata=PDF_METADATA)
    plt.close(fig)
    print(f"saved {OUT_PATH}")


if __name__ == "__main__":
    main()

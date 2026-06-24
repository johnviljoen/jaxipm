"""Paper figure for quad_multi_swap: the jaxipm 2-panel rendezvous fan (n=2, n=4).

Replicates ``multi_swap_fan_jaxipm_n2_n4.pdf`` from the jaxipm paper codebase
(_render_fan_3panel.py::render_2panel_jaxipm_multi_swap): jaxipm-only, two panels
(N_quads=2 and N_quads=4), a few runs each with solid per-quad trajectories,
direction arrows, hollow start circles and goal crosses. The n=2 head-on swap is
drawn in X-Z (avoidance happens in altitude); n=4 in X-Y.

Reads logs/jaxipm_{2,4}_results.npz (produced by jaxipm_multi_swap.py) and writes
figures/multi_swap_fan_jaxipm_n2_n4.pdf. Self-contained: its own apply_paper_fonts
is embedded below — no shared helper module.

Run from the repo root:
    python -m tests.quad_multi_swap.analyze_results
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

OUT_PATH = os.path.join(FIGURES, "multi_swap_fan_jaxipm_n2_n4.pdf")
N_TRAJ = 3  # a few runs only, matching the paper's final-frame styling


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


def load_multi_swap(n_quads):
    path = os.path.join(LOGS, f"jaxipm_{n_quads}_results.npz")
    if not os.path.exists(path):
        return None
    d = np.load(path)
    return {"X_all": d["X_all"],            # (N_RUNS, n_quads, N, 13)
            "success": d["success"].astype(bool),
            "N_RUNS": int(d["N_RUNS"][0]),
            "starts": d["starts"],          # (n_quads, 13)
            "goals": d["goals"]}            # (n_quads, 3)


def main():
    results = [load_multi_swap(2), load_multi_swap(4)]
    valid = [r for r in results if r is not None and r["success"].any()]
    if len(valid) < 2:
        raise SystemExit("need both jaxipm_2 and jaxipm_4 results in logs/ "
                         "(run `python -m tests.quad_multi_swap.jaxipm_multi_swap` first)")

    cmap = plt.cm.get_cmap("tab10")
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), squeeze=False)
    axes = axes[0]
    for ax, r in zip(axes, valid):
        ax.set_aspect("equal")
        succ_idxs = np.where(r["success"])[0]
        # A few runs spread across the batch (not just the tail) for variety.
        if succ_idxs.size > N_TRAJ:
            sample = succ_idxs[np.linspace(0, succ_idxs.size - 1, N_TRAJ).astype(int)]
        else:
            sample = succ_idxs
        n_quads = r["X_all"].shape[1]

        # Pick projection: xy by default, xz when y-range is degenerate
        # (e.g. n=2 head-on swap where avoidance happens in altitude).
        x_range = np.ptp(r["X_all"][..., 0])
        y_range = np.ptp(r["X_all"][..., 1])
        use_xz = (x_range > 0) and (y_range / x_range < 0.01)
        axis_idx = 2 if use_xz else 1
        sign = -1.0 if use_xz else 1.0   # NED: negative z is up
        ax_label = "Z" if use_xz else "Y"

        for bi in sample:
            for q in range(n_quads):
                traj = r["X_all"][bi, q]
                ax.plot(traj[:, 0], sign * traj[:, axis_idx], "-",
                        color=cmap(q % 10), lw=1.6, alpha=0.55)
                add_dir_arrows_2d(ax, traj[:, 0], sign * traj[:, axis_idx],
                                  color=cmap(q % 10), n_arrows=4,
                                  mutation_scale=26, alpha=0.9)

        starts = r["starts"]
        for q in range(n_quads):
            ax.plot(starts[q, 0], sign * starts[q, axis_idx], "o",
                    markersize=6, markerfacecolor="none",
                    markeredgecolor=cmap(q % 10), markeredgewidth=1.2, zorder=5)
        for q in range(n_quads):
            g = r["goals"][q]
            ax.plot(g[0], sign * g[axis_idx], "x", markersize=10,
                    color=cmap(q % 10), markeredgewidth=1.6, zorder=5)

        ax.set_xlabel("X")
        ax.set_ylabel(ax_label)

        # n=2 (XZ) has a tiny Z range vs X; widen plotted Z to match the X span
        # (keeping equal aspect) so this panel renders at the same size as n=4.
        if use_xz:
            x0, x1 = ax.get_xlim()
            y0, y1 = ax.get_ylim()
            span = x1 - x0
            yc = 0.5 * (y0 + y1)
            ax.set_ylim(yc - span / 2.0, yc + span / 2.0)

        _despine(ax)
        legend_handles = [
            plt.Line2D([0], [0], marker="o", color="none", markerfacecolor="none",
                       markeredgecolor="black", markeredgewidth=1.2,
                       markersize=6, label="start"),
            plt.Line2D([0], [0], marker="x", color="none", markeredgecolor="black",
                       markeredgewidth=1.6, markersize=10, label="goal"),
        ]
        ax.legend(handles=legend_handles, loc="upper right")

    fig.tight_layout()
    # Included in the paper at width=\textwidth.
    apply_paper_fonts(fig, frac=1.0, tight=True)
    fig.savefig(OUT_PATH, format="pdf", bbox_inches="tight", metadata=PDF_METADATA)
    plt.close(fig)
    print(f"saved {OUT_PATH}")


if __name__ == "__main__":
    main()

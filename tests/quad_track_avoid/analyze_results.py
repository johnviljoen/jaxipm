"""Analyze results from random-init quadcopter tracking with obstacle avoidance.

Loads <solver>_v<avg_vel>_results.npz from this directory's logs/, prints a
per-velocity summary table, and produces the static paper figures:
  - per-solve time / iteration-count distributions
  - top-down overlay of one representative trajectory per solver
  - per-solver fan plot (random starts + a sampled bundle of trajectories)

Obstacle geometry and the pringle reference parameters are loaded from the
shared tests/quad_track_avoid/test_params.json (via initialization.py), the same
single source of truth the solver scripts use. No animation / GIF output.

Run from the repo root so the `tests.*` import resolves:
    python -m tests.quad_track_avoid.analyze_results
"""

import os
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless: write PNGs, never open a display
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# Single source of truth — obstacles + pringle params from test_params.json.
from tests.quad_track_avoid.initialization import (
    obstacles, PRINGLE_A, PRINGLE_B, PRINGLE_C,
)

HERE = os.path.dirname(os.path.abspath(__file__))
LOGS_DIR = os.path.join(HERE, "logs")        # input: <solver>_v<vel>_results.npz
FIGURES_DIR = os.path.join(HERE, "figures")  # output: the generated figures
os.makedirs(FIGURES_DIR, exist_ok=True)

SOLVERS = ["casadi", "madnlp", "jaxipm"]

def load_results(name, avg_vel):
    """Load `<name>_v<avg_vel>_results.npz`."""
    path = os.path.join(LOGS_DIR, f"{name}_v{avg_vel:.1f}_results.npz")
    if not os.path.exists(path):
        print(f"  [skip] {name} v={avg_vel}: {path} not found")
        return None
    data = np.load(path)
    out = {
        "name": name,
        "avg_vel": float(avg_vel),
        "X_all": data["X_all"],
        "times": data["times"],
        "iters": data["iters"],
        "success": data["success"].astype(bool),
        "total_time": float(data["total_time"][0]),
        "N": int(data["N"][0]),
        "Ts": float(data["Ts"][0]),
        "N_RUNS": int(data["N_RUNS"][0]),
    }
    for k in ("starts", "all_x0", "all_xr", "all_s", "xr"):
        if k in data.files:
            out[k] = data[k]
    return out


def discover_avg_vels():
    """Find all avg_vel values for which any solver has results."""
    pat = re.compile(r"^(?:casadi|madnlp|jaxipm|cusadi_jaxipm)_v(\d+\.\d+)_results\.npz$")
    found = set()
    for fn in os.listdir(LOGS_DIR):
        m = pat.match(fn)
        if m:
            found.add(float(m.group(1)))
    return sorted(found)


def print_summary(results):
    print()
    print("=" * 90)
    print(f"{'Solver':<10} {'N_RUNS':>8} {'success':>10} {'total(s)':>12} "
          f"{'mean(ms)':>12} {'med(ms)':>12} {'mean_iter':>12}")
    print("-" * 90)
    for r in results:
        if r is None:
            continue
        succ_mask = r["success"]
        n_succ = int(succ_mask.sum())
        n_runs = r["N_RUNS"]
        succ_str = f"{n_succ}/{n_runs}"
        if n_succ > 0:
            t_succ = r["times"][succ_mask]
            mean_ms = t_succ.mean() * 1000
            med_ms = float(np.median(t_succ)) * 1000
            iters_succ = r["iters"][succ_mask]
            iters_real = iters_succ[iters_succ >= 0]
            mean_iter = float(iters_real.mean()) if iters_real.size > 0 else float("nan")
        else:
            mean_ms = med_ms = mean_iter = float("nan")
        iter_str = f"{mean_iter:.1f}" if not np.isnan(mean_iter) else "n/a"
        print(f"{r['name']:<10} {n_runs:>8} {succ_str:>10} "
              f"{r['total_time']:>12.2f} {mean_ms:>12.3f} {med_ms:>12.3f} {iter_str:>12}")
    print("=" * 90)
    print()


def plot_distributions(results, save_path):
    valid = [r for r in results if r is not None and r["success"].any()]
    if not valid:
        return
    colors = {"casadi": "tab:blue", "madnlp": "tab:orange", "jaxipm": "tab:green"}

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for r in valid:
        succ = r["success"]
        c = colors.get(r["name"], None)
        times_ms = r["times"][succ] * 1000
        iters_valid = r["iters"][succ]

        if times_ms.std() < 1e-9 or np.unique(times_ms).size < 3:
            axes[0].axvline(times_ms.mean(), color=c, lw=2,
                            label=f"{r['name']} (≈{times_ms.mean():.2f} ms, parallel)")
        else:
            axes[0].hist(times_ms, bins=50, histtype="step", lw=2, color=c, label=r["name"])

        iters_pos = iters_valid[iters_valid >= 0]
        if iters_pos.size == 0:
            continue
        if np.unique(iters_pos).size < 3:
            axes[1].axvline(iters_pos.mean(), color=c, lw=2,
                            label=f"{r['name']} (≈{iters_pos.mean():.0f})")
        else:
            axes[1].hist(iters_pos, bins=30, histtype="step", lw=2, color=c, label=r["name"])

    axes[0].set_xscale("log")
    axes[0].set_xlabel("solve time (ms)")
    axes[0].set_ylabel("count")
    axes[0].set_title("Per-solve time distribution")
    axes[0].legend()
    axes[1].set_xlabel("iterations")
    axes[1].set_ylabel("count")
    axes[1].set_title("Iteration count distribution")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
    print(f"Saved {save_path}")


def _pringle_curve(n=400):
    s = np.linspace(0.0, 2.0 * np.pi, n)
    return (PRINGLE_A * np.cos(s),
            PRINGLE_B * np.sin(s),
            PRINGLE_C * np.cos(2.0 * s))


def _pick_subsample_by_arc(angles, succ_idxs, n):
    """Return n indices into succ_idxs, one per uniform bin over angles' min/max."""
    if succ_idxs.size == 0:
        return np.empty(0, dtype=int)
    lo, hi = float(np.min(angles)), float(np.max(angles))
    if hi <= lo:
        return succ_idxs[: min(n, succ_idxs.size)]
    edges = np.linspace(lo, hi, n + 1)
    out = []
    for i in range(n):
        l, h = edges[i], edges[i + 1]
        mask = (angles >= l) & (angles <= h if i == n - 1 else angles < h)
        cands = succ_idxs[mask]
        if cands.size > 0:
            out.append(int(cands[0]))
    return np.array(out, dtype=int)


def plot_trajectory_fan(results, save_path, n_traj=80):
    """Per-solver fan plot: pringle reference + random starts + sampled trajectories."""
    valid = [r for r in results if r is not None and r["success"].any() and "starts" in r]
    if not valid:
        return

    fig, axes = plt.subplots(1, len(valid), figsize=(6 * len(valid), 6), squeeze=False)
    axes = axes[0]
    colors = {"casadi": "tab:blue", "madnlp": "tab:orange", "jaxipm": "tab:green"}

    px, py, _ = _pringle_curve()

    for ax, r in zip(axes, valid):
        ax.set_aspect("equal")
        c = colors.get(r["name"], "k")

        # Pringle reference (full loop) as faint dotted curve
        ax.plot(px, py, ":", color="gray", lw=1, label="pringle ref")

        # Subsample starts to ~n_traj uniform pringle-arc bins.
        starts = r["starts"]
        # Use the saved pringle parameter `s` if present (most accurate); else
        # derive an angle from xy.
        if "all_s" in r and r["all_s"].shape[0] >= starts.shape[0]:
            ang_starts = r["all_s"][: starts.shape[0]]
        else:
            ang_starts = np.arctan2(starts[:, 1], starts[:, 0])
        succ = r["success"]
        succ_idxs = np.where(succ)[0]

        dot_idxs = _pick_subsample_by_arc(ang_starts[succ_idxs], succ_idxs, n_traj)
        if dot_idxs.size > 0:
            ax.plot(starts[dot_idxs, 0], starts[dot_idxs, 1], ".",
                    color="lightgray", markersize=2,
                    label=f"starts (N={r['N_RUNS']})")
            for idx in dot_idxs:
                traj = r["X_all"][idx]
                ax.plot(traj[:, 0], traj[:, 1], "-", color=c, lw=0.6, alpha=0.4)

        # Obstacles at t=0
        for obs in obstacles:
            xc = obs["xc0"] + obs["ax"] * np.sin(obs["px"])
            yc = obs["yc0"] + obs["ay"] * np.sin(obs["py"])
            ax.add_patch(Circle((xc, yc), obs["r"], color="red", alpha=0.25))

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_title(f"{r['name']} — {dot_idxs.size} sampled trajectories")
        ax.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
    print(f"Saved {save_path}")


def plot_overlay_top_down(results, save_path):
    """Show one representative trajectory per solver overlaid in 2D."""
    valid = [r for r in results if r is not None and r["success"].any()]
    if not valid:
        return
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect("equal")

    px, py, _ = _pringle_curve()
    ax.plot(px, py, ":", color="gray", lw=1, label="pringle ref")

    colors = {"casadi": "tab:blue", "madnlp": "tab:orange", "jaxipm": "tab:green"}
    for r in valid:
        idx = int(np.argmax(r["success"]))
        traj = r["X_all"][idx]
        c = colors.get(r["name"], None)
        ax.plot(traj[:, 0], traj[:, 1], "-", color=c, lw=1.5, label=r["name"])
        ax.plot(traj[0, 0], traj[0, 1], "o", color=c, markersize=6)

    for obs in obstacles:
        xc = obs["xc0"] + obs["ax"] * np.sin(obs["px"])
        yc = obs["yc0"] + obs["ay"] * np.sin(obs["py"])
        ax.add_patch(Circle((xc, yc), obs["r"], color="red", alpha=0.2))

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("Top-down trajectories (first successful solve, t=0 obstacles)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
    print(f"Saved {save_path}")


if __name__ == "__main__":
    env_v = os.environ.get("AVG_VEL")
    if env_v is not None:
        avg_vels = [float(env_v)]
    else:
        avg_vels = discover_avg_vels()
    if not avg_vels:
        print("No vel-suffixed result files found; nothing to analyze.")
    for v in avg_vels:
        print(f"\n>>> avg_vel = {v}")
        results = [load_results(name, v) for name in SOLVERS]
        print_summary(results)
        plot_distributions(results, os.path.join(FIGURES_DIR, f"distributions_v{v:.1f}.png"))
        plot_overlay_top_down(results, os.path.join(FIGURES_DIR, f"overlay_top_down_v{v:.1f}.png"))
        plot_trajectory_fan(results, os.path.join(FIGURES_DIR, f"trajectory_fan_v{v:.1f}.png"))

    print("\nAnalysis complete.")

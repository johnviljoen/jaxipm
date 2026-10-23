"""Render the learned track-avoid policy (and the expert) from the held-out starts.

Inputs : logs/student_heldout.npz, logs/expert_heldout.npz (from dagger_track.py)
Outputs: logs|figures/student_track.gif   3D animation: student quads, moving obstacles, reference curve
         logs|figures/track_overlay.png   top-down expert vs student, obstacle paths, + tracking-error panel
         logs|figures/render_summary.json
Plotted altitude is -z (NED).
"""

import argparse
import json
import os
import shutil

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = os.path.join(os.path.dirname(__file__), "logs")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", default=OUT)
    ap.add_argument("--figures", default=os.path.join(os.path.dirname(__file__), "figures"))
    ap.add_argument("--policy", default=None, help="re-roll the student from this policy instead of the saved npz")
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--err-tol", type=float, default=1.0, help="success: tail RMS tracking error below this [m]")
    args = ap.parse_args()
    os.chdir(REPO)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from tests.imitation_track import sim
    from tests.imitation_nav.plotting_batched import BatchedAnimator

    exp = np.load(os.path.join(args.logs, "expert_heldout.npz"))
    Xe, x0, s0, v = exp["X"], exp["x0"], exp["s0"], float(exp["avg_vel"])
    T = Xe.shape[1] - 1
    if args.policy:
        from tests.imitation_track import policy as pol
        student = pol.load(args.policy)
        Xs, _ = jax.jit(lambda x, s: sim.rollout(jax.vmap(student), x, s, T, v))(jnp.asarray(x0), jnp.asarray(s0))
        Xs = np.asarray(Xs)
    else:
        Xs = np.load(os.path.join(args.logs, "student_heldout.npz"))["X"]
    n = Xs.shape[0]
    ev_s, ev_e = sim.evaluate(Xs, s0, v, err_tol=args.err_tol), sim.evaluate(Xe, s0, v, err_tol=args.err_tol)
    summary = {k: dict(success=int(e["success"].sum()), finite=int(e["finite"].sum()), n=n, err_tol=args.err_tol,
                       median_rms_tail=float(np.median(e["rms_tail"])), max_rms_tail=float(e["rms_tail"].max()),
                       min_clearance=float(e["min_clearance"].min()))
               for k, e in (("student", ev_s), ("expert", ev_e))}
    print(json.dumps(summary, indent=2))
    os.makedirs(args.figures, exist_ok=True)
    for d in (args.logs, args.figures):
        with open(os.path.join(d, "render_summary.json"), "w") as fh:
            json.dump(summary, fh, indent=2)

    # ── 2D overlay + error panel ─────────────────────────────────────────────
    ref = sim.ref_positions(s0, T, v)
    oc = sim.obstacle_path(T)                                           # (T+1, N_OBS, 2)
    ss = np.linspace(0, 2 * np.pi, 400)
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13, 6), gridspec_kw=dict(width_ratios=[1.15, 1]))
    ax.plot(sim.PRINGLE_A * np.cos(ss), sim.PRINGLE_B * np.sin(ss), "k--", lw=1, alpha=0.5, label="reference path")
    for j in range(sim.N_OBS):
        ax.plot(oc[:, j, 0], oc[:, j, 1], color="0.6", lw=1, alpha=0.8)
        ax.add_patch(Circle(oc[0, j], sim.OBS_R[j], color="0.75", alpha=0.6, zorder=1))
        ax.add_patch(Circle(oc[-1, j], sim.OBS_R[j], color="0.45", alpha=0.6, zorder=2))
    for i in range(n):
        ax.plot(Xe[i, :, 0], Xe[i, :, 1], color="C0", lw=1.0, alpha=0.6, zorder=3, label="expert (jaxipm MPC)" if i == 0 else None)
        ax.plot(Xs[i, :, 0], Xs[i, :, 1], color="C3", lw=1.2, alpha=0.8, zorder=4, label="student (MLP)" if i == 0 else None)
    ax.scatter(x0[:, 0], x0[:, 1], s=14, color="k", zorder=5, label="starts")
    ax.set_aspect("equal"); ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    ax.set_title(f"track-avoid (avg_vel {v}): student {summary['student']['success']}/{n} vs expert "
                 f"{summary['expert']['success']}/{n} succeed\nobstacles: light = t 0, dark = t {T*sim.TS:.0f} s")
    ax.legend(loc="upper right", fontsize=8)
    t = np.arange(T + 1) * sim.TS
    for i in range(n):
        ax2.plot(t, ev_e["err"][i], color="C0", lw=0.8, alpha=0.5)
        ax2.plot(t, np.where(np.isfinite(Xs[i]).all(-1), ev_s["err"][i], np.nan), color="C3", lw=0.8, alpha=0.6)
    ax2.set_xlabel("time [s]"); ax2.set_ylabel("position error to reference [m]"); ax2.set_ylim(0, 3)
    ax2.set_title("tracking error (blue expert, red student)")
    fig.tight_layout()
    for d in (args.logs, args.figures):
        png = os.path.join(d, "track_overlay.png"); fig.savefig(png, dpi=160); print("wrote", png)

    # ── 3D gif of the student with moving obstacles ──────────────────────────
    Xp = Xs.copy(); Xp[:, :, 2] *= -1.0
    Xp = np.where(np.isfinite(Xp), Xp, np.nan)
    cyl = np.concatenate([oc, np.broadcast_to(sim.OBS_R[None, :, None], (T + 1, sim.N_OBS, 1))], axis=-1)
    curve = np.stack([sim.PRINGLE_A * np.cos(ss), sim.PRINGLE_B * np.sin(ss), -sim.PRINGLE_C * np.cos(2 * ss)], axis=1)
    gif = os.path.join(args.logs, "student_track.gif")
    anim = BatchedAnimator(sim.quad_params, [Xp[i] for i in range(n)], t, cylinder_traj=cyl, ref_curve=curve,
                           drawCylinder=True, save_path=gif, title="learned MLP policy", alpha=0.5,
                           x_lim=(-5.5, 5.5), y_lim=(-5.5, 5.5), z_lim=(-3.0, 3.0), dt=sim.TS)
    anim.fps = args.fps
    anim.animate()
    shutil.copy(gif, os.path.join(args.figures, "student_track.gif"))
    print("wrote", gif)


if __name__ == "__main__":
    main()

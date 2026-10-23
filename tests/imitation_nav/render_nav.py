"""Render the learned policy (and the expert) from the held-out arc starts.

Inputs (from dagger_nav.py):  logs/student_heldout.npz, logs/expert_heldout.npz
Outputs:
  logs/student_nav.gif      3D animation of the student closed loop (BatchedAnimator)
  logs/nav_overlay.png      top-down expert vs student trajectories + obstacles
  logs/render_summary.json  success table for both
Plotted altitude is -z (NED).
"""

import argparse
import json
import os

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = os.path.join(os.path.dirname(__file__), "logs")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", default=OUT)
    ap.add_argument("--policy", default=None, help="re-roll the student from this policy instead of the saved npz")
    ap.add_argument("--T", type=int, default=50)
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--figures", default=os.path.join(os.path.dirname(__file__), "figures"),
                    help="copy of the png/gif outside the gitignored logs dir")
    args = ap.parse_args()
    os.chdir(REPO)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from tests.imitation_nav import sim
    from tests.imitation_nav.plotting_batched import BatchedAnimator

    exp = np.load(os.path.join(args.logs, "expert_heldout.npz"))
    Xe, starts = exp["X"], exp["starts"]
    if args.policy:
        from tests.imitation_nav import policy as pol
        student = pol.load(args.policy)
        Xs, _ = jax.jit(lambda x: sim.rollout(jax.vmap(student), x, args.T))(jnp.asarray(starts))
        Xs = np.asarray(Xs)
    else:
        Xs = np.load(os.path.join(args.logs, "student_heldout.npz"))["X"]
    n = Xs.shape[0]
    ev_s, ev_e = sim.evaluate(Xs), sim.evaluate(Xe)
    summary = {k: dict(success=int(v["success"].sum()), reached=int(v["reached"].sum()), n=n,
                       median_final_dist=float(np.median(v["final_dist"])),
                       max_final_dist=float(v["final_dist"].max()),
                       min_clearance=float(v["min_clearance"].min()))
               for k, v in (("student", ev_s), ("expert", ev_e))}
    print(json.dumps(summary, indent=2))
    with open(os.path.join(args.logs, "render_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)

    # ── 2D top-down overlay ──────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    for xc, yc, r in sim.OBS:
        ax.add_patch(Circle((xc, yc), r, color="0.75", zorder=1))
        ax.add_patch(Circle((xc, yc), r - sim.QUAD_RADIUS, color="0.45", zorder=2))
    for i in range(n):
        ax.plot(Xe[i, :, 0], Xe[i, :, 1], color="C0", lw=1.0, alpha=0.6, zorder=3,
                label="expert (jaxipm MPC)" if i == 0 else None)
        ax.plot(Xs[i, :, 0], Xs[i, :, 1], color="C3", lw=1.2, alpha=0.8, zorder=4,
                label="student (MLP)" if i == 0 else None)
    ax.scatter(starts[:, 0], starts[:, 1], s=14, color="k", zorder=5, label="starts")
    ax.scatter([0], [0], marker="*", s=160, color="gold", edgecolor="k", zorder=6, label="goal")
    ax.set_aspect("equal"); ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    ax.set_title(f"nav: student {summary['student']['success']}/{n} vs expert "
                 f"{summary['expert']['success']}/{n} succeed")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    os.makedirs(args.figures, exist_ok=True)
    for d in (args.logs, args.figures):
        png = os.path.join(d, "nav_overlay.png")
        fig.savefig(png, dpi=160)
        print("wrote", png)

    # ── 3D gif of the student ────────────────────────────────────────────────
    Xp = Xs.copy(); Xp[:, :, 2] *= -1.0                                # altitude up
    xc_ = [o[0] for o in sim.OBS]; yc_ = [o[1] for o in sim.OBS]; r_ = [o[2] for o in sim.OBS]
    t = np.arange(Xp.shape[1]) * sim.TS
    gif = os.path.join(args.logs, "student_nav.gif")
    anim = BatchedAnimator(sim.quad_params, [Xp[i] for i in range(n)], t,
                           cylinder_definitions=[xc_, yc_, r_], drawCylinder=True,
                           save_path=gif, title="learned MLP policy", alpha=0.5,
                           x_lim=(-1.0, 6.5), y_lim=(-1.0, 6.5), z_lim=(-0.5, 3.0), dt=sim.TS)
    anim.fps = args.fps
    anim.animate()
    print("wrote", gif)
    import shutil
    shutil.copy(gif, os.path.join(args.figures, "student_nav.gif"))
    shutil.copy(os.path.join(args.logs, "render_summary.json"), os.path.join(args.figures, "render_summary.json"))


if __name__ == "__main__":
    main()

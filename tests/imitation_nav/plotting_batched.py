import matplotlib.pyplot as plt
from matplotlib import animation
import jax.numpy as jnp
import numpy as np

from tests.imitation_nav import geometry


class BatchedAnimator:
    """Animator for visualizing multiple quadcopter trajectories from a batch solve."""

    def __init__(
            self,
            p,
            xs,  # List/array of state trajectories, shape (num_quads, num_steps, state_dim)
            t,
            rs=None,  # Optional: list of reference trajectories, or single reference for all
            state_predictions=None,
            cylinder_definitions=None,
            drawCylinder=False,
            followQuad=False,
            max_frames=500,
            dt=0.1,
            save_path='data/gifs/batch_test.gif',
            title='Batch Solve',
            colors=None,  # Optional: list of colors for each quad
            alpha=0.7,  # Transparency for quad trails
            x_lim=None,
            y_lim=None,
            z_lim=None,
            cylinder_traj=None,   # (num_steps, n_obs, 3) [xc, yc, r] per time step (moving obstacles)
            ref_curve=None,       # (M, 3) static reference curve drawn once
        ):

        self.num_quads = len(xs)
        num_steps = len(t)

        def compute_render_interval(num_steps, max_frames):
            render_interval = 1
            while num_steps / render_interval > max_frames:
                render_interval *= 2
            return render_interval

        render_interval = compute_render_interval(num_steps, max_frames)
        self.viewing_range = 1.0
        self.camera_z_offset = 0.5
        self.save_path = save_path
        self.dt = dt

        self.rps = None
        self.cylinders = None
        self.drawCylinder = drawCylinder
        self.followQuad = followQuad
        self.preds = None

        # Handle cylinder definitions
        if cylinder_definitions is not None:
            xc_, yc_, r_ = cylinder_definitions
            self.cylinder_definitions = [(xc, yc, r) for xc, yc, r in zip(xc_, yc_, r_)]
        else:
            self.cylinder_definitions = []

        self.p = p
        self.t = t[::render_interval]
        self.cyl_traj = None if cylinder_traj is None else np.asarray(cylinder_traj)[::render_interval]
        self.ref_curve = None if ref_curve is None else np.asarray(ref_curve)
        self.fps = 12

        # Process all trajectories - ensure numpy arrays
        self.xs = [np.asarray(x)[::render_interval, :] for x in xs]

        # Handle references - can be single ref for all or list of refs
        if rs is None:
            self.rs = [self.xs[0] for _ in range(self.num_quads)]  # Use first traj as dummy ref
        elif isinstance(rs, list) and len(rs) == self.num_quads:
            self.rs = [np.asarray(r)[::render_interval, :] for r in rs]
        else:
            # Single reference for all quads
            r_arr = np.asarray(rs)[::render_interval, :]
            self.rs = [r_arr for _ in range(self.num_quads)]

        # Generate colors if not provided
        if colors is None:
            cmap = plt.get_cmap('tab10')
            self.colors = [cmap(i % 10) for i in range(self.num_quads)]
        else:
            self.colors = colors

        self.alpha = alpha

        # Instantiate the figure
        self.fig = plt.figure(figsize=(10, 8))
        self.ax = self.fig.add_subplot(projection='3d')

        # Initialize lines for each quadcopter
        self.lines1 = []  # Red arm
        self.lines2 = []  # Blue arm
        self.lines3 = []  # Trail (dashed)

        for i in range(self.num_quads):
            color = self.colors[i]
            line1, = self.ax.plot([], [], [], lw=2, color=color)
            line2, = self.ax.plot([], [], [], lw=2, color=color, alpha=0.7)
            line3, = self.ax.plot([], [], [], '--', lw=1, color=color, alpha=self.alpha)
            self.lines1.append(line1)
            self.lines2.append(line2)
            self.lines3.append(line3)

        # Compute global bounds across all trajectories
        extra_each_side = 0.5

        all_x = np.concatenate([x[:, 0] for x in self.xs])
        all_y = np.concatenate([x[:, 1] for x in self.xs])
        all_z = np.concatenate([x[:, 2] for x in self.xs])

        x_min, x_max = all_x.min(), all_x.max()
        y_min, y_max = all_y.min(), all_y.max()
        z_min, z_max = all_z.min(), all_z.max()

        max_range = 0.5 * np.array([x_max - x_min, y_max - y_min, z_max - z_min]).max() + extra_each_side
        mid_x = 0.5 * (x_max + x_min)
        mid_y = 0.5 * (y_max + y_min)
        mid_z = 0.5 * (z_max + z_min)

        if x_lim is not None:
            self.ax.set_xlim3d(x_lim)
        else:
            self.ax.set_xlim3d([mid_x - max_range, mid_x + max_range])
        self.ax.set_xlabel('X')

        if y_lim is not None:
            self.ax.set_ylim3d(y_lim)
        else:
            self.ax.set_ylim3d([mid_y - max_range, mid_y + max_range])
        self.ax.set_ylabel('Y')

        if z_lim is not None:
            self.ax.set_zlim3d(z_lim)
        else:
            self.ax.set_zlim3d([mid_z - max_range, mid_z + max_range])
        self.ax.set_zlabel('Altitude')

        if self.ref_curve is not None:
            self.ax.plot(self.ref_curve[:, 0], self.ref_curve[:, 1], self.ref_curve[:, 2], 'k--', lw=1, alpha=0.5)
        self.title_time = self.ax.text2D(0.05, 0.95, "", transform=self.ax.transAxes)
        self.ax.text2D(0.95, 0.95, f"{title} (n={self.num_quads})", transform=self.ax.transAxes, horizontalalignment='right')

    def draw_cylinder(self, x_center=0, y_center=0, z_center=0, radius=1, depth=1, resolution=100):
        z = jnp.linspace(z_center - depth, z_center + depth, resolution)
        theta = jnp.linspace(0, 2 * jnp.pi, resolution)
        theta_grid, z_grid = jnp.meshgrid(theta, z)
        x_grid = radius * jnp.cos(theta_grid) + x_center
        y_grid = radius * jnp.sin(theta_grid) + y_center
        return self.ax.plot_surface(x_grid, y_grid, z_grid, alpha=0.5, rstride=5, cstride=5, color='b')

    def update_lines(self, k):
        tk = self.t[k]

        for i in range(self.num_quads):
            x = self.xs[i]  # Already numpy from __init__
            x_0k = x[0:k + 1]
            xk = x[k]

            qk = xk[3:7]

            R = np.asarray(geometry.quaternion_to_rotation_matrix(qk))

            arm_offsets = np.array([
                [self.p["dxm"], -self.p["dym"], self.p["dzm"]],
                [0, 0, 0],
                [self.p["dxm"], self.p["dym"], self.p["dzm"]],
                [-self.p["dxm"], self.p["dym"], self.p["dzm"]],
                [0, 0, 0],
                [-self.p["dxm"], -self.p["dym"], self.p["dzm"]]
            ]).T
            motor_points = R @ arm_offsets
            motor_points = motor_points + np.asarray(xk[0:3])[:, None]

            self.lines1[i].set_data(motor_points[0, 0:3], motor_points[1, 0:3])
            self.lines1[i].set_3d_properties(motor_points[2, 0:3])
            self.lines2[i].set_data(motor_points[0, 3:6], motor_points[1, 3:6])
            self.lines2[i].set_3d_properties(motor_points[2, 3:6])
            self.lines3[i].set_data(x_0k[:, 0], x_0k[:, 1])
            self.lines3[i].set_3d_properties(x_0k[:, 2])

        # Draw cylinders
        if self.cyl_traj is not None:
            self.cylinder_definitions = [(float(xc), float(yc), float(r)) for xc, yc, r in self.cyl_traj[min(k, len(self.cyl_traj) - 1)]]
        if self.drawCylinder and self.cylinder_definitions:
            if self.cylinders is not None:
                for cyl_obj in self.cylinders:
                    cyl_obj.remove()
            cyl_objs = []
            for cyl_def in self.cylinder_definitions:
                xc, yc, r, zc, depth = (*cyl_def, 1.0, 1.0) if len(cyl_def) == 3 else cyl_def
                cyl_obj = self.draw_cylinder(x_center=xc, y_center=yc, z_center=zc, radius=r, depth=depth)
                cyl_objs.append(cyl_obj)
            self.cylinders = cyl_objs

        self.title_time.set_text(u"Time = {:.2f} s ".format(tk))

        if self.followQuad:
            # Follow the centroid of all quads
            positions = np.array([self.xs[i][k, 0:3] for i in range(self.num_quads)])
            centroid = positions.mean(axis=0)

            self.ax.set_xlim(centroid[0] - self.viewing_range, centroid[0] + self.viewing_range)
            self.ax.set_ylim(centroid[1] - self.viewing_range, centroid[1] + self.viewing_range)
            self.ax.set_zlim(centroid[2] - self.viewing_range, centroid[2] + self.viewing_range)

    def ini_plot(self):
        for i in range(self.num_quads):
            self.lines1[i].set_data(np.empty([1]), np.empty([1]))
            self.lines1[i].set_3d_properties(np.empty([1]))
            self.lines2[i].set_data(np.empty([1]), np.empty([1]))
            self.lines2[i].set_3d_properties(np.empty([1]))
            self.lines3[i].set_data(np.empty([1]), np.empty([1]))
            self.lines3[i].set_3d_properties(np.empty([1]))

        return self.lines1 + self.lines2 + self.lines3

    def animate(self):
        line_ani = animation.FuncAnimation(
            self.fig,
            self.update_lines,
            init_func=self.ini_plot,
            frames=len(self.t) - 1,
            interval=(self.dt * 10),
            blit=False)

        line_ani.save(self.save_path, dpi=120, fps=self.fps)
        return line_ani

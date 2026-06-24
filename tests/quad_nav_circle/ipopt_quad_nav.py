import time, json
import casadi as ca
import numpy as np
from tests.quad_casadi_dynamics import f_ca

with open("tests/quad_params.json") as f:
    quad_params = json.load(f)
    quad_params["IB"] = np.diag(np.array([quad_params["IB00"], quad_params["IB11"], quad_params["IB22"]]))

# post init useful parameters for quad
quad_params["B0"] = np.array([
    [quad_params["kTh"], quad_params["kTh"], quad_params["kTh"], quad_params["kTh"]],
    [quad_params["dym"]*quad_params["kTh"], -quad_params["dym"]*quad_params["kTh"], -quad_params["dym"]*quad_params["kTh"], quad_params["dym"]*quad_params["kTh"]],
    [quad_params["dxm"]*quad_params["kTh"], quad_params["dxm"]*quad_params["kTh"], -quad_params["dxm"]*quad_params["kTh"], -quad_params["dxm"]*quad_params["kTh"]],
    [-quad_params["kTo"], quad_params["kTo"], -quad_params["kTo"], quad_params["kTo"]]]) # actuation matrix

with open("tests/quad_nav_circle/test_params.json") as f:
    tmp = json.load(f)
    # JSON has no infinity, so `null` denotes an unbounded (±inf) entry.
    quad_params["x_lb"] = np.array([-np.inf if v is None else v for v in tmp["x_lb"]])
    quad_params["x_ub"] = np.array([ np.inf if v is None else v for v in tmp["x_ub"]])

    # Horizon / timestep
    N_HORIZON = tmp["N_horizon"]
    TS = tmp["Ts"]

    # Obstacles (base radius `r`; the quad's own radius is added on in the MPC)
    OBSTACLES = tmp["obstacles"]
    QUAD_RADIUS = tmp["quad_radius"]

class QuadcopterNavMPC:
    """
    CasADi MPC for quadcopter navigation with obstacle avoidance.

    State: x = [x, y, z, q0, q1, q2, q3, xd, yd, zd, p, q, r] (13 states)
    Control: u = [w0, w1, w2, w3] (4 motor speeds in rad/s)

    Objective: Minimize sum of Q*x^2 (drive to origin)
    Subject to:
        - Quadcopter dynamics (Euler integration)
        - Initial condition constraint
        - State bounds (position, velocity, angular rates)
        - Control bounds (motor speeds)
        - Cylinder obstacle avoidance
    """

    def __init__(self, x0: np.ndarray = None, N: int = 30, Ts: float = 0.1):
        """
        Initialize and solve the problem once.

        Args:
            x0: Initial state (13,). Defaults to [4, 4, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0].
            N: Horizon length.
            Ts: Timestep.
        """
        self.nx = 13
        self.nu = 4
        self.N = N
        self.Ts = Ts
        self.qp = quad_params

        # Cost weights (same as JAX version)
        # Q = [1,1,1,0,0,0,0,1,1,1,1,1,1] for states
        self.Q = np.array([1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])

        # Default initial condition
        if x0 is None:
            x0 = np.array([4., 4., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
        self.x0_default = x0

        # Obstacle parameters (from test_params.json)
        self.xc_ = [o["xc"] for o in OBSTACLES]
        self.yc_ = [o["yc"] for o in OBSTACLES]
        self.r_ = [o["r"] for o in OBSTACLES]
        self.quad_radius = QUAD_RADIUS
        self.r__ = [r + self.quad_radius for r in self.r_]

        # Create optimizer
        self.opti = ca.Opti()

        # Decision variables
        self.X = self.opti.variable(self.nx, N)
        self.U = self.opti.variable(self.nu, N - 1)

        # Parameter for initial condition (can change at runtime)
        self.init = self.opti.parameter(self.nx, 1)

        # Cost function: sum of Q * x^2
        cost = ca.MX(0)
        for k in range(N):
            for i in range(self.nx):
                cost += self.Q[i] * self.X[i, k] ** 2
        self.opti.minimize(cost)

        # Initial condition constraint (parameterized)
        self.opti.subject_to(self.X[:, 0] == self.init)

        # Dynamics constraints (Euler integration): x[k+1] = x[k] + f(x[k], u[k]) * Ts
        for k in range(N - 1):
            xdot = f_ca(self.X[:, k], self.U[:, k], self.qp)
            x_next = self.X[:, k] + xdot * Ts
            self.opti.subject_to(self.X[:, k + 1] == x_next)

        # State bounds (for non-inf bounds)
        x_lb = self.qp["x_lb"]
        x_ub = self.qp["x_ub"]
        for k in range(N):
            for i in range(self.nx):
                if not np.isinf(x_lb[i]):
                    self.opti.subject_to(self.X[i, k] >= x_lb[i])
                if not np.isinf(x_ub[i]):
                    self.opti.subject_to(self.X[i, k] <= x_ub[i])

        # Control bounds (motor speeds)
        self.opti.subject_to(self.opti.bounded(
            self.qp["minWmotor"], self.U, self.qp["maxWmotor"]
        ))

        # Obstacle avoidance constraints
        # (x - xc)^2 + (y - yc)^2 >= r^2 * multiplier
        for k in range(N - 1):
            multiplier = 1 + k * Ts * 0.1
            for xc, yc, r in zip(self.xc_, self.yc_, self.r__):
                dist_sq = (self.X[0, k] - xc) ** 2 + (self.X[1, k] - yc) ** 2
                self.opti.subject_to(dist_sq >= r ** 2 * multiplier)

        # Solver options
        opts = {
            'ipopt.print_level': 0,
            'print_time': 0,
            'ipopt.tol': 1e-6,
            'ipopt.warm_start_init_point': 'yes',
        }
        self.opti.solver('ipopt', opts)

        # Set initial condition parameter
        self.opti.set_value(self.init, x0)

        # Initial guess: linear interpolation for position, hover for controls
        start_xyz = x0[:3]
        end_xyz = np.array([0., 0., -2.])
        xyz_traj = np.linspace(start_xyz, end_xyz, N).T

        # Initialize full state trajectory
        x_init = np.zeros((self.nx, N))
        x_init[:3, :] = xyz_traj
        x_init[3, :] = 1.0  # q0 = 1 (identity quaternion)
        # Other states start at 0

        # Hover motor speed
        w_hover = 522.9847140714692
        u_init = np.ones((self.nu, N - 1)) * w_hover

        self.opti.set_initial(self.X, x_init)
        self.opti.set_initial(self.U, u_init)

        # Solve once to initialize
        t1 = time.time()
        sol = self.opti.solve()
        t2 = time.time()
        print(f"Initial solve time: {(t2 - t1) * 1000:.3f} ms")
        self.x_sol = sol.value(self.X)
        self.u_sol = sol.value(self.U)

    def solve_cold(self, x0_new=None):
        """Re-solve from a cold initial guess; optionally override the IC."""
        if x0_new is None:
            x0_new = self.x0_default

        start_xyz = x0_new[:3]
        end_xyz = np.array([0., 0., -2.])
        xyz_traj = np.linspace(start_xyz, end_xyz, self.N).T

        x_init = np.zeros((self.nx, self.N))
        x_init[:3, :] = xyz_traj
        x_init[3, :] = 1.0

        w_hover = 522.9847140714692
        u_init = np.ones((self.nu, self.N - 1)) * w_hover

        self.opti.set_value(self.init, x0_new)
        self.opti.set_initial(self.X, x_init)
        self.opti.set_initial(self.U, u_init)

        sol = self.opti.solve()
        self.x_sol = sol.value(self.X)
        self.u_sol = sol.value(self.U)
        return sol

    def __call__(self, x: np.ndarray):
        """
        Solve with warm starting from previous solution.

        Args:
            x: New initial state (13,)

        Returns:
            First control input (4,)
        """
        # Set new initial condition
        self.opti.set_value(self.init, x)

        # Warm start: shift previous solution
        old_x_sol = self.x_sol[:, 1:]
        x_warm = np.hstack([old_x_sol, old_x_sol[:, -1:]])

        old_u_sol = self.u_sol[:, 1:]
        u_warm = np.hstack([old_u_sol, old_u_sol[:, -1:]])

        self.opti.set_initial(self.X, x_warm)
        self.opti.set_initial(self.U, u_warm)

        # Solve
        sol = self.opti.solve()
        self.x_sol = sol.value(self.X)
        self.u_sol = sol.value(self.U)

        return self.u_sol[:, 0]

    def get_solution(self):
        """Return current solution."""
        return self.x_sol, self.u_sol

    def get_cost(self):
        """Return current optimal cost."""
        return self.opti.value(self.opti.f)

    def simulate_step(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Simulate one step of dynamics (for testing)."""
        # Use numpy version for simulation
        from tests.quad_jax_dynamics import f_jnp
        import jax.numpy as jnp
        x_jnp = jnp.array(x)
        u_jnp = jnp.array(u)
        xdot = f_jnp(x_jnp, u_jnp, self.qp)
        return np.array(x + np.array(xdot) * self.Ts)


def time_warm_start():
    """Time warm start calls of the quadcopter MPC, including per-iteration timing."""
    print("=" * 60)
    print("Timing quadcopter MPC warm start calls")
    print("=" * 60)

    # Initial state
    x0 = np.array([4., 4., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.])

    print("\nCold start (initialization)...")
    t0 = time.perf_counter()
    mpc = QuadcopterNavMPC(x0)
    cold_time = time.perf_counter() - t0
    cold_iters = mpc.opti.stats()['iter_count']
    print(f"Cold start time: {cold_time * 1000:.3f} ms")
    print(f"Cold start iterations: {cold_iters}")
    print(f"Cold start time/iter: {cold_time * 1000 / cold_iters:.3f} ms")
    print(f"Initial cost: {mpc.get_cost():.6f}")

    # Simulate a few steps with warm starting
    x_current = x0.copy()
    n_steps = 100
    warm_times = []
    warm_iters = []
    trajectory = [x_current.copy()]

    print(f"\nRunning {n_steps} warm start solves...")
    for i in range(n_steps):
        # Apply first control and simulate one step
        u = mpc.u_sol[:, 0]
        x_next = mpc.simulate_step(x_current, u)

        # Time the warm start solve
        t0 = time.perf_counter()
        mpc(x_next)
        warm_time = time.perf_counter() - t0
        iters = mpc.opti.stats()['iter_count']

        warm_times.append(warm_time)
        warm_iters.append(iters)

        x_current = x_next
        trajectory.append(x_current.copy())

    warm_times_ms = [wt * 1000 for wt in warm_times]
    time_per_iter = [wt / it if it > 0 else 0 for wt, it in zip(warm_times_ms, warm_iters)]

    print(f"\nWarm start times (ms):")
    print(f"  {'Step':<6} {'Time (ms)':<12} {'Iters':<8} {'ms/iter':<10}")
    print(f"  {'-'*36}")
    for i, (wt, it, tpi) in enumerate(zip(warm_times_ms, warm_iters, time_per_iter)):
        print(f"  {i + 1:<6} {wt:<12.3f} {it:<8} {tpi:<10.3f}")

    print(f"\nStatistics:")
    print(f"  Total time:       {sum(warm_times_ms):.3f} ms")
    print(f"  Mean time:        {np.mean(warm_times_ms):.3f} ms")
    print(f"  Std time:         {np.std(warm_times_ms):.3f} ms")
    print(f"  Total iterations: {sum(warm_iters)}")
    print(f"  Mean iterations:  {np.mean(warm_iters):.1f}")
    print(f"  Mean ms/iter:     {np.mean(time_per_iter):.3f} ms")
    print(f"  Min ms/iter:      {np.min(time_per_iter):.3f} ms")
    print(f"  Max ms/iter:      {np.max(time_per_iter):.3f} ms")

    print(f"\nFinal position: {x_current[:3]}")
    print(f"Final cost: {mpc.get_cost():.6f}")

    # Animate the trajectory
    # xs_arr = np.array(trajectory)  # (n_steps+1, nx)
    # xs_arr[:, :2] *= -1.  # negate x and y to match animator coordinate convention
    # t_arr = np.arange(len(trajectory)) * mpc.Ts
    # animator = BatchedAnimator(
    #     p=quad_params,
    #     xs=[xs_arr],
    #     t=t_arr,
    #     cylinder_definitions=(mpc.xc_, mpc.yc_, mpc.r_),
    #     drawCylinder=True,
    #     dt=mpc.Ts,
    #     title='CasADi Quadcopter Nav MPC',
    #     save_path='casadi_quad_nav.gif',
    # )
    # animator.animate()

    return warm_times, warm_iters


if __name__ == "__main__":

    import os
    import time as time_mod
    from tqdm import tqdm

    # Circle of starts: linspace angles in a sector centered at π/4 (45°).
    # SECTOR_DEG controls the total arc angular width in degrees.
    #   SECTOR_DEG=90  → angles [0, π/2]     (the original "corner" sector)
    #   SECTOR_DEG=180 → angles [-π/4, 3π/4] (wider half-plane, crosses axes)
    RADIUS = float(np.sqrt(4.0**2 + 4.0**2))  # ~5.657 m, matches default x0=[4,4,0]
    N_horizon = N_HORIZON
    Ts = TS
    N_RUNS = tmp["N_RUNS_seq"]
    for SECTOR_DEG in tmp["init_angles"]:

        sector_half_rad = np.deg2rad(SECTOR_DEG) / 2.0
        sector_center_rad = np.pi / 4.0  # 45° — matches the default x0 direction
        angles = np.linspace(sector_center_rad - sector_half_rad,
                             sector_center_rad + sector_half_rad, N_RUNS)

        def x0_from_angle(theta):
            return np.array([
                RADIUS * np.cos(theta), RADIUS * np.sin(theta), 0.0,
                1.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0,
                0.0, 0.0, 0.0,
            ])

        x0_default = x0_from_angle(angles[0])

        print(f"CasADi nav: building MPC and warming up "
              f"(N={N_horizon}, N_RUNS={N_RUNS}, R={RADIUS:.3f})")
        mpc = QuadcopterNavMPC(x0_default, N=N_horizon, Ts=Ts)

        X_all = np.full((N_RUNS, N_horizon, 13), np.nan, dtype=np.float64)
        times = np.zeros(N_RUNS, dtype=np.float64)
        iters = np.zeros(N_RUNS, dtype=np.int32)
        obj_vals = np.full(N_RUNS, np.nan, dtype=np.float64)
        success = np.zeros(N_RUNS, dtype=bool)
        starts = np.zeros((N_RUNS, 13), dtype=np.float64)

        t0 = time_mod.time()
        pbar = tqdm(range(N_RUNS), desc="CasADi nav", unit="solve")
        for i in pbar:
            x0_i = x0_from_angle(angles[i])
            starts[i] = x0_i
            t1 = time_mod.time()
            try:
                sol = mpc.solve_cold(x0_new=x0_i)
                t2 = time_mod.time()
                X_all[i] = mpc.x_sol.T
                times[i] = t2 - t1
                try:
                    iters[i] = sol.stats().get("iter_count", -1)
                except Exception:
                    iters[i] = -1
                try:
                    obj_vals[i] = float(sol.value(mpc.opti.f))
                except Exception:
                    pass
                success[i] = True
            except RuntimeError:
                t2 = time_mod.time()
                times[i] = t2 - t1
                try:
                    X_all[i] = np.asarray(mpc.opti.debug.value(mpc.X)).T
                    iters[i] = mpc.opti.stats().get("iter_count", -1)
                except Exception:
                    iters[i] = -1
                try:
                    obj_vals[i] = float(mpc.opti.debug.value(mpc.opti.f))
                except Exception:
                    pass
                success[i] = False

            if (i + 1) % 10 == 0:
                pbar.set_postfix(succ=f"{success[:i+1].mean():.3f}")

        total_time = time_mod.time() - t0

        logs_dir = os.path.join(os.path.dirname(__file__), "logs")
        os.makedirs(logs_dir, exist_ok=True)
        out_path = os.path.join(logs_dir, f"casadi_sector{int(SECTOR_DEG)}_results.npz")
        np.savez(
            out_path,
            X_all=X_all,
            times=times,
            iters=iters,
            obj_vals=obj_vals,
            success=success,
            starts=starts,
            angles=angles,
            total_time=np.array([total_time]),
            N=np.array([N_horizon]),
            Ts=np.array([Ts]),
            N_RUNS=np.array([N_RUNS]),
            radius=np.array([RADIUS]),
            sector_deg=np.array([SECTOR_DEG]),
        )
        print(f"CasADi nav: saved {out_path}")
        if success.any():
            print(f"CasADi nav: total_time={total_time:.2f}s, success={success.sum()}/{N_RUNS}, "
                  f"mean_solve={times[success].mean()*1000:.2f}ms")
        else:
            print(f"CasADi nav: total_time={total_time:.2f}s, success=0/{N_RUNS}")

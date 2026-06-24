import time, json
import casadi as ca
import numpy as np
from tests.quad_casadi_dynamics import f_ca
from tests.quad_track_avoid.initialization import generate_random_inits, obstacles

with open("tests/quad_params.json") as f:
    quad_params = json.load(f)
    quad_params["IB"] = np.diag(np.array([quad_params["IB00"], quad_params["IB11"], quad_params["IB22"]]))

# post init useful parameters for quad
quad_params["B0"] = np.array([
    [quad_params["kTh"], quad_params["kTh"], quad_params["kTh"], quad_params["kTh"]],
    [quad_params["dym"]*quad_params["kTh"], -quad_params["dym"]*quad_params["kTh"], -quad_params["dym"]*quad_params["kTh"], quad_params["dym"]*quad_params["kTh"]],
    [quad_params["dxm"]*quad_params["kTh"], quad_params["dxm"]*quad_params["kTh"], -quad_params["dxm"]*quad_params["kTh"], -quad_params["dxm"]*quad_params["kTh"]],
    [-quad_params["kTo"], quad_params["kTo"], -quad_params["kTo"], quad_params["kTo"]]]) # actuation matrix

with open("tests/quad_track_avoid/test_params.json") as f:
    tmp = json.load(f)
    # JSON has no infinity, so `null` denotes an unbounded (±inf) entry.
    quad_params["x_lb"] = np.array([-np.inf if v is None else v for v in tmp["x_lb"]])
    quad_params["x_ub"] = np.array([ np.inf if v is None else v for v in tmp["x_ub"]])

    # Horizon / timestep
    N_HORIZON = tmp["N_horizon"]
    TS = tmp["Ts"]

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

    def __init__(self, x0: np.ndarray, xr0=None, N: int = 30, Ts: float = 0.1, obstacles=None):
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
        self.x0_default = x0
 
        # Create optimizer
        self.opti = ca.Opti()

        # Decision variables
        self.X = self.opti.variable(self.nx, N)
        self.U = self.opti.variable(self.nu, N - 1)

        # Parameter for initial condition (can change at runtime)
        self.init = self.opti.parameter(self.nx, 1)
        self.Xr = self.opti.parameter(self.nx, N)

        # Cost function: sum of Q * x^2 - we ignore input costs
        cost = ca.MX(0)
        for k in range(N):
            for i in range(self.nx):
                cost += self.Q[i] * (self.X[i, k] - self.Xr[i, k]) ** 2
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

        # Quaternion norm band: 0.5 <= ||q||^2 <= 1.5. Matches the two-sided
        # band that jaxipm_track_avoid applies via d_L/d_U. The lower bound
        # is non-convex but IPOPT's IPM handles it fine on this problem.
        for k in range(N):
            q = self.X[3:7, k]
            q_norm2 = q[0]**2 + q[1]**2 + q[2]**2 + q[3]**2
            self.opti.subject_to(q_norm2 >= 0.5)
            self.opti.subject_to(q_norm2 <= 1.5)

        # Cylinder obstacle avoidance constraints (time-varying)
        if obstacles is not None:
            for obs in obstacles:
                for k in range(N):
                    tk = k * Ts
                    xc = obs["xc0"] + obs["ax"] * np.sin(obs["fx"] * tk + obs["px"])
                    yc = obs["yc0"] + obs["ay"] * np.sin(obs["fy"] * tk + obs["py"])
                    self.opti.subject_to(
                        (self.X[0, k] - xc)**2 + (self.X[1, k] - yc)**2 >= obs["r"]**2
                    )

        # Control bounds (motor speeds)
        self.opti.subject_to(self.opti.bounded(
            self.qp["minWmotor"], self.U, self.qp["maxWmotor"]
        ))

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

        if xr0 is None:
            xr0 = np.vstack([x0]*N).T
        else:
            xr0 = xr0.T
        self.opti.set_value(self.Xr, xr0)

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

    def solve_cold(self):
        """Re-solve the problem from the original cold initial guess."""
        # Reset to the same linear-interpolation cold initial guess used in __init__
        start_xyz = self.x0_default[:3]
        end_xyz = np.array([0., 0., -2.])
        xyz_traj = np.linspace(start_xyz, end_xyz, self.N).T

        x_init = np.zeros((self.nx, self.N))
        x_init[:3, :] = xyz_traj
        x_init[3, :] = 1.0

        w_hover = 522.9847140714692
        u_init = np.ones((self.nu, self.N - 1)) * w_hover

        self.opti.set_initial(self.X, x_init)
        self.opti.set_initial(self.U, u_init)

        sol = self.opti.solve()
        self.x_sol = sol.value(self.X)
        self.u_sol = sol.value(self.U)
        return sol

    def __call__(self, x: np.ndarray, xr: np.ndarray):
        
        self.opti.set_value(self.init, x)
        self.opti.set_value(self.Xr, xr)

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

        return self.u_sol[:,0]

def get_xr_traj(t0, N, Ts, average_vel=0.5, A=4, B=4, C=2):
    """
    Generate a pringle (hyperbolic paraboloid) trajectory.

    Parametric path:
        x(t) = A * cos(s)
        y(t) = B * sin(s)
        z(t) = C * cos(2s)
    where s = t * average_vel.
    """

    def YPRToQuat_np(r1, r2, r3):
        cr1 = np.cos(0.5*r1)
        cr2 = np.cos(0.5*r2)
        cr3 = np.cos(0.5*r3)
        sr1 = np.sin(0.5*r1)
        sr2 = np.sin(0.5*r2)
        sr3 = np.sin(0.5*r3)
        q0 = cr1*cr2*cr3 + sr1*sr2*sr3
        q1 = cr1*cr2*sr3 - sr1*sr2*cr3
        q2 = cr1*sr2*cr3 + sr1*cr2*sr3
        q3 = sr1*cr2*cr3 - cr1*sr2*sr3
        q = np.array([q0, q1, q2, q3])
        q = q / np.linalg.norm(q)
        return q

    states = []
    for t_abs in np.arange(t0, t0 + N * Ts, Ts):
        s = t_abs * average_vel

        # Position on pringle
        x = A * np.cos(s)
        y = B * np.sin(s)
        z = C * np.cos(2 * s)

        # Analytical velocities (chain rule: dx/dt = dx/ds * ds/dt)
        xdot = -A * np.sin(s) * average_vel
        ydot =  B * np.cos(s) * average_vel
        zdot = -2 * C * np.sin(2 * s) * average_vel

        # Yaw from velocity direction
        yaw = np.arctan2(ydot, xdot)
        q0, q1, q2, q3 = YPRToQuat_np(yaw, 0, 0)

        p, q, r = 0, 0, 0

        states.append(np.array([
            x, y, z,
            q0, q1, q2, q3,
            xdot, ydot, zdot,
            p, q, r
        ]))

    return np.vstack(states)

if __name__ == "__main__":

    import os
    import time as time_mod
    from tqdm import tqdm

    N = N_HORIZON
    Ts = TS
    N_RUNS = tmp["N_RUNS_seq"]
    DELTA_STD = float(os.environ.get("DELTA_STD", "0.5"))

    for AVG_VEL in tmp["avg_vel"]:
        print(f"\n================ avg_vel = {AVG_VEL} ================")
        print(f"CasADi: generating {N_RUNS} random initial conditions (delta_std={DELTA_STD})")
        all_x0_np, all_xr_np, all_s_np, n_rejects = generate_random_inits(
            N_RUNS, N, Ts, AVG_VEL, delta_std=DELTA_STD, seed=0)
        print(f"CasADi: rejected {n_rejects} samples (kept {N_RUNS})")

        # Build MPC with the first sample as the seed problem (constructor transposes
        # xr internally, so pass the (N, nx) form).
        print(f"CasADi: building MPC and warming up (N={N}, N_RUNS={N_RUNS})")
        mpc = QuadcopterNavMPC(all_x0_np[0], all_xr_np[0], N, Ts, obstacles=obstacles)

        X_all = np.full((N_RUNS, N, 13), np.nan, dtype=np.float64)
        times = np.zeros(N_RUNS, dtype=np.float64)
        iters = np.zeros(N_RUNS, dtype=np.int32)
        obj_vals = np.full(N_RUNS, np.nan, dtype=np.float64)
        success = np.zeros(N_RUNS, dtype=bool)
        starts = np.zeros((N_RUNS, 13), dtype=np.float64)

        t0 = time_mod.time()
        pbar = tqdm(range(N_RUNS), desc=f"CasADi v{AVG_VEL}", unit="solve")
        for i in pbar:
            x0_i = all_x0_np[i]
            xr_i = all_xr_np[i]  # (N, 13)
            starts[i] = x0_i

            # Cold solve at this random (x0, xr): set parameters, reset cold guess, solve.
            mpc.opti.set_value(mpc.init, x0_i)
            mpc.opti.set_value(mpc.Xr, xr_i.T)
            mpc.x0_default = x0_i  # so solve_cold uses correct cold guess

            t1 = time_mod.time()
            try:
                sol = mpc.solve_cold()
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
        out_path = os.path.join(logs_dir, f"casadi_v{AVG_VEL:.1f}_results.npz")
        np.savez(
            out_path,
            X_all=X_all,
            times=times,
            iters=iters,
            obj_vals=obj_vals,
            success=success,
            starts=starts,
            all_x0=all_x0_np,
            all_xr=all_xr_np,
            all_s=all_s_np,
            total_time=np.array([total_time]),
            N=np.array([N]),
            Ts=np.array([Ts]),
            N_RUNS=np.array([N_RUNS]),
            avg_vel=np.array([AVG_VEL]),
        )
        print(f"CasADi: saved {out_path}")
        print(f"CasADi: total_time={total_time:.2f}s, success={success.sum()}/{N_RUNS}, "
              f"mean_solve={times[success].mean()*1000:.2f}ms" if success.any() else
              f"CasADi: total_time={total_time:.2f}s, success=0/{N_RUNS}")

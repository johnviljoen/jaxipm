"""CasADi/IPOPT reference solver for the multi-quadcopter rendezvous problem.

N_quads quads start evenly spaced on a circle of radius R, z=0, and must
each fly to the diametrically opposite point while avoiding pairwise
collisions. Same 13-DOF quad dynamics used throughout this repo.

Per-quad layout: `x_i` (N, 13) state, `u_i` (N-1, 4) motor speeds. The
decision variable is stacked as:

    z = [x_0.flat, x_1.flat, ..., x_{Nq-1}.flat,
         u_0.flat, u_1.flat, ..., u_{Nq-1}.flat]

Cost: per-quad Q-weighted quadratic in (state - goal). Because goals are
non-zero, the cost is `Q * (x - goal)^2`, NOT `Q * x^2` (gauntlet/nav bug
trap).

Pairwise collision: ||pos_i[k] - pos_j[k]||^2 >= min_dist^2 for every
pair i<j and every stage k=0..N-2. No per-stage margin multiplier.

Cold guess: hover at each quad's start (feasible because starts are
2*R*sin(pi/N_quads) apart, which exceeds min_dist for any sensible R).
"""

import time, json
import casadi as ca
import numpy as np
from tests.quad_casadi_dynamics import f_ca

with open("tests/quad_params.json") as f:
    quad_params = json.load(f)
    quad_params["IB"] = np.diag(np.array([quad_params["IB00"], quad_params["IB11"], quad_params["IB22"]]))

quad_params["B0"] = np.array([
    [quad_params["kTh"], quad_params["kTh"], quad_params["kTh"], quad_params["kTh"]],
    [quad_params["dym"]*quad_params["kTh"], -quad_params["dym"]*quad_params["kTh"], -quad_params["dym"]*quad_params["kTh"], quad_params["dym"]*quad_params["kTh"]],
    [quad_params["dxm"]*quad_params["kTh"], quad_params["dxm"]*quad_params["kTh"], -quad_params["dxm"]*quad_params["kTh"], -quad_params["dxm"]*quad_params["kTh"]],
    [-quad_params["kTo"], quad_params["kTo"], -quad_params["kTo"], quad_params["kTo"]]])

# Multi-swap geometry (module-level so JAX and cusadi variants can import).
# R and N_quads are parametric; the helpers below build starts / goals.
def start_state(i: int, N_quads: int, R: float) -> np.ndarray:
    """Quad i's 13-state start: position on circle, identity quaternion,
    zero velocity + angular rate."""
    theta = 2.0 * np.pi * i / N_quads
    return np.array([
        R * np.cos(theta), R * np.sin(theta), 0.0,
        1.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0,
        0.0, 0.0, 0.0,
    ])


def goal_xyz(i: int, N_quads: int, R: float) -> np.ndarray:
    """Quad i's goal position (diametrically opposite start)."""
    theta = 2.0 * np.pi * i / N_quads + np.pi
    return np.array([R * np.cos(theta), R * np.sin(theta), 0.0])


# Test-specific parameters from test_params.json (horizon, timestep, geometry,
# collision, and the R-independent state-bound pieces).
with open("tests/quad_multi_swap/test_params.json") as f:
    tmp = json.load(f)
N_HORIZON = tmp["N_horizon"]
TS = tmp["Ts"]
R_DEFAULT = tmp["R"]
XY_MARGIN = tmp["xy_margin"]
Z_BOUND = tmp["z_bound"]
VEL_RATE_BOUND = tmp["vel_rate_bound"]

# Collision parameters
QUAD_RADIUS = tmp["quad_radius"]
SAFETY = tmp["safety"]
# Doubled from (2*r_quad + safety) = 0.5 to 1.0 for a more substantial
# avoidance cushion between quads.
MIN_DIST = 2.0 * (2 * QUAD_RADIUS + SAFETY)   # 1.0 m centre-centre


class QuadcopterMultiSwapMPC:
    """CasADi MPC for the multi-quad rendezvous: N_quads swap sides of a
    circle while avoiding each other."""

    def __init__(self, N_quads: int = 2, N: int = N_HORIZON, Ts: float = TS, R: float = R_DEFAULT):
        self.N_quads = N_quads
        self.N = N
        self.Ts = Ts
        self.R = R
        self.qp = quad_params

        self.nx_dim = 13
        self.nu_dim = 4

        # Q: position 1.0, linear velocity 0.1 (relaxed), angular rates 1.0.
        # Quaternion entries unweighted.
        self.Q = np.array([1, 1, 1, 0, 0, 0, 0, 0.1, 0.1, 0.1, 1, 1, 1])

        self.starts = [start_state(i, N_quads, R) for i in range(N_quads)]
        self.goals = [goal_xyz(i, N_quads, R) for i in range(N_quads)]

        # Per-quad 13-vector goal (zeros outside position slots so Q-weighted
        # quadratic cost only penalises position tracking).
        self.goal_vec = [
            np.hstack([self.goals[i], np.zeros(10)]) for i in range(N_quads)
        ]

        self.opti = ca.Opti()

        self.Xs = [self.opti.variable(self.nx_dim, N) for _ in range(N_quads)]
        self.Us = [self.opti.variable(self.nu_dim, N - 1) for _ in range(N_quads)]

        self.inits = [self.opti.parameter(self.nx_dim, 1) for _ in range(N_quads)]

        # Cost: sum_i sum_k Q * (x_i[k] - goal_i)^2
        cost = ca.MX(0)
        for i in range(N_quads):
            gvec = ca.DM(self.goal_vec[i])
            for k in range(N):
                dx = self.Xs[i][:, k] - gvec
                for s in range(self.nx_dim):
                    cost += self.Q[s] * dx[s] ** 2
        self.opti.minimize(cost)

        # Initial condition + Euler dynamics per quad.
        for i in range(N_quads):
            self.opti.subject_to(self.Xs[i][:, 0] == self.inits[i])
            for k in range(N - 1):
                xdot = f_ca(self.Xs[i][:, k], self.Us[i][:, k], self.qp)
                x_next = self.Xs[i][:, k] + xdot * Ts
                self.opti.subject_to(self.Xs[i][:, k + 1] == x_next)

        # State bounds (per-quad, per-stage).
        # x, y in ±(R+XY_MARGIN); z in ±Z_BOUND; q free; vel & rate in ±VEL_RATE_BOUND.
        vrb = VEL_RATE_BOUND
        x_lb = np.array([-(R + XY_MARGIN), -(R + XY_MARGIN), -Z_BOUND,
                         -np.inf, -np.inf, -np.inf, -np.inf,
                         -vrb, -vrb, -vrb,
                         -vrb, -vrb, -vrb])
        x_ub = np.array([ (R + XY_MARGIN),  (R + XY_MARGIN),  Z_BOUND,
                          np.inf,  np.inf,  np.inf,  np.inf,
                          vrb,  vrb,  vrb,
                          vrb,  vrb,  vrb])
        self.x_lb = x_lb
        self.x_ub = x_ub

        for i in range(N_quads):
            for k in range(N):
                for s in range(self.nx_dim):
                    if not np.isinf(x_lb[s]):
                        self.opti.subject_to(self.Xs[i][s, k] >= x_lb[s])
                    if not np.isinf(x_ub[s]):
                        self.opti.subject_to(self.Xs[i][s, k] <= x_ub[s])

        # Motor bounds on every control slot of every quad.
        for i in range(N_quads):
            self.opti.subject_to(self.opti.bounded(
                self.qp["minWmotor"], self.Us[i], self.qp["maxWmotor"]
            ))

        # Pairwise collision avoidance.
        min_dist_sq = MIN_DIST ** 2
        for i in range(N_quads):
            for j in range(i + 1, N_quads):
                for k in range(N - 1):
                    diff = self.Xs[i][0:3, k] - self.Xs[j][0:3, k]
                    norm_sq = diff[0] ** 2 + diff[1] ** 2 + diff[2] ** 2
                    self.opti.subject_to(norm_sq >= min_dist_sq)

        opts = {
            'ipopt.print_level': 0,
            'print_time': 0,
            'ipopt.tol': 1e-6,
            'ipopt.warm_start_init_point': 'yes',
            'ipopt.max_iter': 500,
        }
        self.opti.solver('ipopt', opts)

        for i in range(N_quads):
            self.opti.set_value(self.inits[i], self.starts[i])

        self._set_hover_guess()

        t1 = time.time()
        sol = self.opti.solve()
        t2 = time.time()
        print(f"Initial solve time: {(t2 - t1) * 1000:.3f} ms")
        self.x_sols = [sol.value(X) for X in self.Xs]
        self.u_sols = [sol.value(U) for U in self.Us]

    def _set_hover_guess(self):
        """Hover at each quad's start (feasible; IPM skips restoration)."""
        w_hover = 522.9847140714692
        for i in range(self.N_quads):
            x_init_i = np.tile(self.starts[i][:, None], (1, self.N))
            u_init_i = np.ones((self.nu_dim, self.N - 1)) * w_hover
            self.opti.set_initial(self.Xs[i], x_init_i)
            self.opti.set_initial(self.Us[i], u_init_i)

    def solve_cold(self):
        """Re-solve from the constant-hover cold guess (no warm-starting)."""
        for i in range(self.N_quads):
            self.opti.set_value(self.inits[i], self.starts[i])
        self._set_hover_guess()
        sol = self.opti.solve()
        self.x_sols = [sol.value(X) for X in self.Xs]
        self.u_sols = [sol.value(U) for U in self.Us]
        return sol

    def get_solution(self):
        return self.x_sols, self.u_sols

    def get_cost(self):
        return self.opti.value(self.opti.f)


def check_success(X_quad_traj: np.ndarray, goals: list, iter_count: int,
                  max_iter: int = 500,
                  goal_tol: float = 0.3,
                  collision_tol: float = 0.8 * MIN_DIST) -> bool:
    """Real success check: every quad ended close to goal, no pairwise
    collision violation, iter count not at cap.

    Args:
        X_quad_traj: (N_quads, N, 13) trajectory of one solve.
        goals: list of 3-vec goal positions per quad.
        iter_count: number of IPM iterations used.
        max_iter: iteration cap (iter_count == this ⇒ failed to converge).
        goal_tol: max allowed distance at final timestep for success.
        collision_tol: min allowed pairwise distance over horizon
            (default scales with MIN_DIST so it tracks changes to the
            avoidance constraint).
    """
    N_quads = X_quad_traj.shape[0]
    # Goal check.
    pos_errs = np.array([
        np.linalg.norm(X_quad_traj[i, -1, :3] - goals[i])
        for i in range(N_quads)
    ])
    goal_ok = pos_errs.max() < goal_tol

    # Collision check.
    min_d = np.inf
    for i in range(N_quads):
        for j in range(i + 1, N_quads):
            d = np.linalg.norm(
                X_quad_traj[i, :, :3] - X_quad_traj[j, :, :3], axis=-1
            )
            min_d = min(min_d, d.min())
    no_collision = min_d > collision_tol

    # iter_count == -1 means iter info was not retained (happens when
    # N_RUNS > N_BATCH because final_state.iter_count is per-slot, not
    # per-collected-problem). In that case fall back to goal+collision.
    iters_ok = (iter_count == -1) or (0 <= iter_count < max_iter)

    return bool(goal_ok and no_collision and iters_ok)


if __name__ == "__main__":
    import os
    import time as time_mod
    from tqdm import tqdm

    N_horizon = N_HORIZON # int(os.environ.get("N", str(N_HORIZON)))
    Ts = TS
    R = R_DEFAULT # float(os.environ.get("R", str(R_DEFAULT)))
    N_RUNS = tmp["N_RUNS_seq"]

    for N_quads in tmp["N_quads"]:
        print(f"CasADi multi-swap: N_quads={N_quads}, N={N_horizon}, R={R}, N_RUNS={N_RUNS}")
        mpc = QuadcopterMultiSwapMPC(N_quads=N_quads, N=N_horizon, Ts=Ts, R=R)

        X_all = np.full((N_RUNS, N_quads, N_horizon, 13), np.nan, dtype=np.float64)
        times = np.zeros(N_RUNS, dtype=np.float64)
        iters = np.zeros(N_RUNS, dtype=np.int32)
        obj_vals = np.full(N_RUNS, np.nan, dtype=np.float64)
        success = np.zeros(N_RUNS, dtype=bool)

        t0 = time_mod.time()
        pbar = tqdm(range(N_RUNS), desc="CasADi multi-swap", unit="solve")
        for i in pbar:
            t1 = time_mod.time()
            converged = False
            try:
                sol = mpc.solve_cold()
                t2 = time_mod.time()
                times[i] = t2 - t1
                for q in range(N_quads):
                    X_all[i, q] = mpc.x_sols[q].T
                try:
                    iters[i] = sol.stats().get("iter_count", -1)
                except Exception:
                    iters[i] = -1
                try:
                    obj_vals[i] = float(sol.value(mpc.opti.f))
                except Exception:
                    pass
                converged = True
            except RuntimeError:
                t2 = time_mod.time()
                times[i] = t2 - t1
                try:
                    for q in range(N_quads):
                        X_all[i, q] = np.asarray(mpc.opti.debug.value(mpc.Xs[q])).T
                    iters[i] = mpc.opti.stats().get("iter_count", -1)
                except Exception:
                    iters[i] = -1
                try:
                    obj_vals[i] = float(mpc.opti.debug.value(mpc.opti.f))
                except Exception:
                    pass

            if converged:
                success[i] = check_success(X_all[i], mpc.goals, int(iters[i]),
                                            max_iter=500)

            if (i + 1) % 10 == 0:
                pbar.set_postfix(succ=f"{success[:i+1].mean():.3f}")

        total_time = time_mod.time() - t0

        # Persist goals + starts so analyze_results.py can draw them.
        starts_arr = np.array(mpc.starts)
        goals_arr = np.array(mpc.goals)

        logs_dir = os.path.join(os.path.dirname(__file__), "logs")
        os.makedirs(logs_dir, exist_ok=True)
        out_path = os.path.join(logs_dir, f"casadi_{N_quads}_results.npz")
        np.savez(
            out_path,
            X_all=X_all,
            times=times,
            iters=iters,
            obj_vals=obj_vals,
            success=success,
            total_time=np.array([total_time]),
            N=np.array([N_horizon]),
            Ts=np.array([Ts]),
            N_RUNS=np.array([N_RUNS]),
            N_quads=np.array([N_quads]),
            R=np.array([R]),
            starts=starts_arr,
            goals=goals_arr,
        )
        print(f"CasADi multi-swap: saved {out_path}")
        if success.any():
            print(f"CasADi multi-swap: total_time={total_time:.2f}s, "
                  f"success={success.sum()}/{N_RUNS}, "
                  f"mean_solve={times[success].mean()*1000:.2f}ms, "
                  f"mean_iters={iters[success].mean():.1f}")
        else:
            print(f"CasADi multi-swap: total_time={total_time:.2f}s, success=0/{N_RUNS}")

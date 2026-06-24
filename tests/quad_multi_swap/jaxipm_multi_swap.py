"""Pure-JAX multi-quadcopter rendezvous problem. Returns the standard
(f, c, d, bounds, z_init) tuple consumed by jaxipm; imported by
cusadi_jaxipm_multi_swap.py as numerical ground truth for parity checks.

Mirrors casadi_multi_swap.py exactly:
  - same 13-DOF quadcopter dynamics (rotor dynamics removed)
  - same Q: position 1.0, linear velocity 0.1, angular rates 1.0
  - N_quads quads evenly spaced on a circle of radius R; goals are
    diametrically opposite points (z=0)
  - cost is `Q * (x - goal)^2` per quad (goal ≠ 0!)
  - pairwise collision: ||pos_i[k] - pos_j[k]||^2 - min_dist^2 >= 0
  - hover cold guess (feasible because starts exceed min_dist apart)
  - no per-stage margin multiplier
"""
import json
import numpy as np
import jax.numpy as jnp
from tests.quad_jax_dynamics import f_jnp

# Test-specific parameters from test_params.json (geometry, collision, horizon,
# timestep, and the R-independent state-bound pieces).
with open("tests/quad_multi_swap/test_params.json") as f:
    tmp = json.load(f)
N_HORIZON = tmp["N_horizon"]
TS = tmp["Ts"]
R = tmp["R"]
XY_MARGIN = tmp["xy_margin"]
Z_BOUND = tmp["z_bound"]
VEL_RATE_BOUND = tmp["vel_rate_bound"]

# Collision parameters (match casadi_multi_swap.py).
QUAD_RADIUS = tmp["quad_radius"]
SAFETY = tmp["safety"]
# Doubled from (2*r_quad + safety) = 0.5 to 1.0 for a more substantial
# avoidance cushion between quads.
MIN_DIST = 2.0 * (2 * QUAD_RADIUS + SAFETY)   # 1.0 m centre-centre

with open("tests/quad_params.json") as f:
    quad_params = json.load(f)
    quad_params["IB"] = jnp.diag(jnp.array([quad_params["IB00"], quad_params["IB11"], quad_params["IB22"]]))


quad_params["B0"] = np.array([
    [quad_params["kTh"]]*4,
    [quad_params["dym"]*quad_params["kTh"], -quad_params["dym"]*quad_params["kTh"],
        -quad_params["dym"]*quad_params["kTh"], quad_params["dym"]*quad_params["kTh"]],
    [quad_params["dxm"]*quad_params["kTh"], quad_params["dxm"]*quad_params["kTh"],
        -quad_params["dxm"]*quad_params["kTh"], -quad_params["dxm"]*quad_params["kTh"]],
    [-quad_params["kTo"], quad_params["kTo"], -quad_params["kTo"], quad_params["kTo"]]])

# State bounds (per-quad, per-stage).
quad_params["x_lb"] = np.array([
    -(R + XY_MARGIN), -(R + XY_MARGIN), -Z_BOUND,
    -np.inf, -np.inf, -np.inf, -np.inf,
    -VEL_RATE_BOUND, -VEL_RATE_BOUND, -VEL_RATE_BOUND,
    -VEL_RATE_BOUND, -VEL_RATE_BOUND, -VEL_RATE_BOUND,
])
quad_params["x_ub"] = np.array([
    (R + XY_MARGIN), (R + XY_MARGIN), Z_BOUND,
    np.inf, np.inf, np.inf, np.inf,
    VEL_RATE_BOUND, VEL_RATE_BOUND, VEL_RATE_BOUND,
    VEL_RATE_BOUND, VEL_RATE_BOUND, VEL_RATE_BOUND,
])


def start_state(i: int, N_quads: int, R: float) -> np.ndarray:
    theta = 2.0 * np.pi * i / N_quads
    return np.array([
        R * np.cos(theta), R * np.sin(theta), 0.0,
        1.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0,
        0.0, 0.0, 0.0,
    ])


def goal_xyz(i: int, N_quads: int, R: float) -> np.ndarray:
    theta = 2.0 * np.pi * i / N_quads + np.pi
    return np.array([R * np.cos(theta), R * np.sin(theta), 0.0])


def quadcopter_multi_swap(N_quads: int = 2, N: int = N_HORIZON, R: float = R):
    nx_dim, nu_dim, Ts = 13, 4, TS

    stride_x = N * nx_dim
    stride_u = (N - 1) * nu_dim

    def xu_to_z(xs, us):
        """xs: list of (N, 13) arrays; us: list of (N-1, 4) arrays."""
        return jnp.hstack(
            [x.flatten() for x in xs] + [u.flatten() for u in us]
        )

    def z_to_xu(z):
        xs = [z[i*stride_x:(i+1)*stride_x].reshape(N, nx_dim)
              for i in range(N_quads)]
        u_base = N_quads * stride_x
        us = [z[u_base + i*stride_u: u_base + (i+1)*stride_u].reshape(N-1, nu_dim)
              for i in range(N_quads)]
        return xs, us

    # Per-quad starts / goals.
    starts = [start_state(i, N_quads, R) for i in range(N_quads)]
    goals = [goal_xyz(i, N_quads, R) for i in range(N_quads)]
    # Full 13-vector goal (zeros outside position slots).
    goal_full = [np.hstack([goals[i], np.zeros(10)]) for i in range(N_quads)]

    starts_j = [jnp.asarray(s) for s in starts]
    goal_full_j = [jnp.asarray(g) for g in goal_full]

    # Hover cold guess.
    w_hover = 522.9847140714692
    state0_list = [jnp.tile(starts_j[i], (N, 1)) for i in range(N_quads)]
    input0_list = [jnp.ones((N - 1, nu_dim)) * w_hover for _ in range(N_quads)]
    z_init = xu_to_z(state0_list, input0_list)

    def f(z):
        """Sum of per-quad Q-weighted quadratic in (state - goal_i)."""
        Q = jnp.array([1, 1, 1, 0, 0, 0, 0, 0.1, 0.1, 0.1, 1, 1, 1])
        xs, _ = z_to_xu(z)
        total = jnp.asarray(0.0)
        for i in range(N_quads):
            dx = xs[i] - goal_full_j[i]   # (N, 13) broadcast
            total = total + jnp.sum(Q * dx ** 2)
        return total

    def c(z):
        xs, us = z_to_xu(z)
        constraints = []
        for i in range(N_quads):
            constraints.append(xs[i][0] - starts_j[i])
            for k in range(N - 1):
                x_next = xs[i][k] + f_jnp(xs[i][k], us[i][k], quad_params) * Ts
                constraints.append(xs[i][k + 1] - x_next)
        return jnp.concatenate(constraints).flatten()

    min_dist_sq = MIN_DIST ** 2

    def d(z):
        xs, _ = z_to_xu(z)
        cl = []
        for i in range(N_quads):
            for j in range(i + 1, N_quads):
                for k in range(N - 1):
                    diff = xs[i][k, :3] - xs[j][k, :3]
                    cl.append(jnp.sum(diff ** 2) - min_dist_sq)
        return jnp.hstack(cl)

    gt = None

    # Variable bounds.
    x_lb = quad_params["x_lb"]
    x_ub = quad_params["x_ub"]
    finite_lb = np.where(~np.isinf(x_lb))[0]
    finite_ub = np.where(~np.isinf(x_ub))[0]

    z_lb = np.full(z_init.size, -np.inf, dtype=np.float64)
    z_ub = np.full(z_init.size,  np.inf, dtype=np.float64)

    for i in range(N_quads):
        base = i * stride_x
        for k in range(N):
            for s in finite_lb:
                z_lb[base + k * nx_dim + int(s)] = x_lb[int(s)]
            for s in finite_ub:
                z_ub[base + k * nx_dim + int(s)] = x_ub[int(s)]

    u_base = N_quads * stride_x
    for i in range(N_quads):
        base = u_base + i * stride_u
        for k in range(N - 1):
            for m in range(nu_dim):
                z_lb[base + k * nu_dim + m] = quad_params["minWmotor"]
                z_ub[base + k * nu_dim + m] = quad_params["maxWmotor"]

    x_L = jnp.asarray(z_lb)
    x_U = jnp.asarray(z_ub)

    d_L = jnp.zeros(d(z_init).shape)
    d_U = jnp.ones(d(z_init).shape) * jnp.inf

    aux = [z_to_xu, xu_to_z, quad_params, N_quads, N, R, starts, goals]

    return f, c, d, x_L, x_U, d_L, d_U, z_init, gt, aux


if __name__ == "__main__":
    import os
    from time import time

    with open("jaxipm/params.json") as _pf:
        p = json.load(_pf)

    # if os.environ.get("HOT_RESTART", "1") == "0":
    #     p["hot_restarting"] = False
    #     print("[jaxipm] hot_restarting DISABLED via HOT_RESTART=0")
    # if os.environ.get("JAXIPM_DEBUG", "0") == "1":
    #     p["DEBUG_MODE"] = True
    #     print("[jaxipm] DEBUG_MODE=true via JAXIPM_DEBUG=1 — iter_buffer enabled")

    # os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    # os.environ['CUDA_VISIBLE_DEVICES'] = str(p["gpu_id"])

    import jax
    import equinox as eqx

    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_compilation_cache_dir", "/home/john/tmp/jax_cache")
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)

    from jaxipm.solver import solve_throughput
    from jaxipm.initialization import (
        initialize_common_problem, initialize_problem_regular,
    )

    from tests.quad_multi_swap.ipopt_multi_swap import check_success

    N_horizon = N_HORIZON # int(os.environ.get("N", str(N_HORIZON)))
    # R = float(os.environ.get("R", str(R)))
    Ts_ = TS

    for N_quads in tmp["N_quads"]:
        print(f"JAXIPM multi-swap: N_quads={N_quads}, N={N_horizon}, R={R}")
        f, c, d, x_L, x_U, d_L, d_U, x0, gt, aux = quadcopter_multi_swap(
            N_quads=N_quads, N=N_horizon, R=R)
        z_to_xu, xu_to_z, qp, _, _, _, starts, goals = aux

        print(f"JAXIPM multi-swap: nx={x0.size}, nyc={c(x0).size}, nyd={d(x0).size}")

        jax.clear_caches()
        f_args = (); c_args = (); d_args = ()
        _user_args = (
            tuple(jnp.asarray(a) for a in f_args),
            tuple(jnp.asarray(a) for a in c_args),
            tuple(jnp.asarray(a) for a in d_args),
        )
        _x0_flat = x0.squeeze()
        calc_next_problem = lambda key, sol: (_x0_flat, *_user_args)

        cp = initialize_common_problem(
            f, c, d, x_L, x_U, d_L, d_U, x0, p,
            [f_args, c_args, d_args],
            calc_next_problem=calc_next_problem)
        state = initialize_problem_regular(cp, x0, args=[f_args, c_args, d_args])
        state = eqx.tree_at(lambda t: t.fl.needs_regular_init, state, jnp.array([[0]]))

        print("\n--- Testing solve_throughput (pure-JAX multi-swap) ---")
        N_batch = tmp["batch_size"]
        max_solves = tmp["N_RUNS_jaxipm"]
        max_iter_per_solve = cp.p["max_iter"] # int(os.environ.get("MAX_ITER", "500"))

        def stack_states_tp(states):
            def stack_leaves(*leaves):
                if eqx.is_array(leaves[0]):
                    return jnp.stack(leaves)
                return leaves[0]
            return jax.tree.map(stack_leaves, *states)

        batch_tp = stack_states_tp([state] * N_batch)

        rng_key = jax.random.PRNGKey(0)
        _solve_throughput = eqx.filter_jit(solve_throughput, donate="none")

        t1 = time()
        tp_out = _solve_throughput(rng_key, cp, batch_tp, max_solves,
                                   max_iter_per_solve=max_iter_per_solve)
        jax.block_until_ready(tp_out[0])
        t2 = time()
        print(f"Throughput (incl. JIT): {(t2-t1)*1000:.1f} ms")

        t1 = time()
        tp_out = _solve_throughput(rng_key, cp, batch_tp, max_solves,
                                   max_iter_per_solve=max_iter_per_solve)
        jax.block_until_ready(tp_out[0])
        t2 = time()
        total_time = t2 - t1
        print(f"Throughput (warm):      {total_time*1000:.1f} ms")

        if len(tp_out) == 5:
            final_state, solution_buffer, write_idx, term_buffer, iter_buffer = tp_out
        else:
            final_state, solution_buffer, write_idx = tp_out
            iter_buffer = None
            term_buffer = None
        n_collected = min(int(write_idx), int(solution_buffer.shape[0]))
        print(f"JAXIPM multi-swap: collected {n_collected} (write_idx {int(write_idx)})")

        sol_buf_np = np.asarray(solution_buffer[:n_collected, :cp.nx])
        iter_counts_np = np.asarray(final_state.iter_count.squeeze()).reshape(-1)

        X_all = np.full((n_collected, N_quads, N_horizon, 13), np.nan, dtype=np.float64)
        for i in range(n_collected):
            xs_i, _ = aux[0](sol_buf_np[i])
            for q in range(N_quads):
                X_all[i, q] = np.asarray(xs_i[q])

        # Per-problem objective values (post-hoc f evaluation on saved z)
        obj_vals = np.zeros(n_collected, dtype=np.float64)
        for i in range(n_collected):
            z_i = jnp.asarray(sol_buf_np[i])
            obj_vals[i] = float(f(z_i))

        times = np.full(n_collected, total_time / max(n_collected, 1), dtype=np.float64)
        if iter_buffer is not None:
            iters = np.asarray(iter_buffer)[:n_collected].astype(np.int32)
        elif iter_counts_np.size >= n_collected:
            iters = iter_counts_np[:n_collected].astype(np.int32)
        else:
            iters = np.full(n_collected, -1, dtype=np.int32)

        # Real success check per solve (don't trust hardcoded true).
        success = np.zeros(n_collected, dtype=bool)
        for i in range(n_collected):
            success[i] = check_success(X_all[i], goals, int(iters[i]),
                                       max_iter=max_iter_per_solve)

        starts_arr = np.array(starts)
        goals_arr = np.array(goals)

        logs_dir = os.path.join(os.path.dirname(__file__), "logs")
        os.makedirs(logs_dir, exist_ok=True)
        out_path = os.path.join(logs_dir, f"jaxipm_{N_quads}_results.npz")
        if term_buffer is not None:
            terms = np.asarray(term_buffer)[:n_collected].astype(np.int32)
        else:
            terms = np.full(n_collected, -1, dtype=np.int32)

        np.savez(out_path, X_all=X_all, times=times, iters=iters, obj_vals=obj_vals,
                 terms=terms,
                 success=success,
                 total_time=np.array([total_time]), N=np.array([N_horizon]),
                 Ts=np.array([Ts_]), N_RUNS=np.array([max_solves]),
                 N_quads=np.array([N_quads]), R=np.array([R]),
                 starts=starts_arr, goals=goals_arr)
        print(f"JAXIPM multi-swap: saved {out_path}")
        print(f"JAXIPM multi-swap: total_time={total_time:.2f}s, "
              f"collected={n_collected}/{max_solves}, "
              f"success={int(success.sum())}/{n_collected}")

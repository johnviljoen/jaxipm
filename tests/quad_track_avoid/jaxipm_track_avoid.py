import json
import numpy as np
import jax.numpy as jnp
from tests.quad_jax_dynamics import f_jnp
from tests.quad_track_avoid.initialization import build_xr_from_s, generate_random_inits, obstacles

with open("tests/quad_params.json") as f:
    quad_params = json.load(f)
    quad_params["IB"] = jnp.diag(jnp.array([quad_params["IB00"], quad_params["IB11"], quad_params["IB22"]]))

quad_params["B0"] = np.array([
    [quad_params["kTh"], quad_params["kTh"], quad_params["kTh"], quad_params["kTh"]],
    [quad_params["dym"]*quad_params["kTh"], -quad_params["dym"]*quad_params["kTh"], -quad_params["dym"]*quad_params["kTh"], quad_params["dym"]*quad_params["kTh"]],
    [quad_params["dxm"]*quad_params["kTh"], quad_params["dxm"]*quad_params["kTh"], -quad_params["dxm"]*quad_params["kTh"], -quad_params["dxm"]*quad_params["kTh"]],
    [-quad_params["kTo"], quad_params["kTo"], -quad_params["kTo"], quad_params["kTo"]]
])

with open("tests/quad_track_avoid/test_params.json") as f:
    tmp = json.load(f)
    # JSON has no infinity, so `null` denotes an unbounded (±inf) entry.
    quad_params["x_lb"] = np.array([-np.inf if v is None else v for v in tmp["x_lb"]])
    quad_params["x_ub"] = np.array([ np.inf if v is None else v for v in tmp["x_ub"]])

    # Horizon / timestep
    N_HORIZON = tmp["N_horizon"]
    TS = tmp["Ts"]

def quadcopter_track_avoid(N=300, avg_vel=1.5):
    nx, nu, Ts = 13, 4, TS

    # ── z <-> (x, u) helpers ────────────────────────────────────────────────
    def xu_to_z(x, u):
        return jnp.hstack([x.flatten(), u.flatten()])

    def z_to_xu(z):
        x = z[:N*nx].reshape(N, nx)
        u = z[N*nx:].reshape(N-1, nu)
        return x, u

    # ── Default reference (s_start=0) used at problem-build time ────────────
    # Per-slot xr is injected at solve time as f_args[1]; this default makes
    # the function callable for the cyipopt adapter and for cusadi codegen.
    xr_default_np = build_xr_from_s(0.0, N, Ts, float(avg_vel))
    xr_default = jnp.array(xr_default_np)
    x0_default = jnp.array(xr_default_np[0])  # start at pringle t=0 by default

    # ── Precompute obstacle positions at each timestep ──────────────────────
    obs_positions = []
    for k in range(N):
        tk = k * Ts
        for obs in obstacles:
            xc = obs["xc0"] + obs["ax"] * np.sin(obs["fx"] * tk + obs["px"])
            yc = obs["yc0"] + obs["ay"] * np.sin(obs["fy"] * tk + obs["py"])
            obs_positions.append((xc, yc, obs["r"]**2))
    obs_xc = jnp.array([p[0] for p in obs_positions])
    obs_yc = jnp.array([p[1] for p in obs_positions])
    obs_r2 = jnp.array([p[2] for p in obs_positions])
    n_obs = len(obstacles)

    # ── Initial guess (warm start z_init for cyipopt path) ──────────────────
    # Match casadi/madnlp: linspace from x0 to a fixed (0, 0, -2) hover, with
    # identity quaternion and zero velocities/angular rates.
    start_xyz = jnp.array(x0_default[:3])
    end_xyz = jnp.array([0., 0., -2.])
    xyz = jnp.linspace(start_xyz, end_xyz, N)
    ang_v_angv = jnp.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]] * N)
    state0 = jnp.hstack([xyz, ang_v_angv])
    input0 = jnp.array([[522.9847140714692]*4]*(N-1))
    z_init = xu_to_z(state0, input0)

    # ── Objective: tracking cost (xr is per-slot runtime arg) ───────────────
    def f(z, xr=xr_default):
        Q = jnp.array([1,1,1,0,0,0,0,1,1,1,1,1,1])
        x, u = z_to_xu(z)
        cost = jnp.sum(Q * (x - xr)**2)
        return cost

    # ── Euler step ──────────────────────────────────────────────────────────
    def euler_step(x, u, qp=quad_params, dt=Ts):
        return x + dt * f_jnp(x, u, qp)

    # ── Equality constraints: initial condition + dynamics ──────────────────
    # x0 is the per-slot runtime arg (mirrors nav_circle's x0_ic pattern).
    def c(z, x0=x0_default, N=N, Ts=Ts):
        x, u = z_to_xu(z)
        constraints = []
        constraints.append(x[0] - x0)
        for k in range(N-1):
            x_next = euler_step(x[k], u[k])
            constraints.append(x[k+1] - x_next)
        return jnp.concatenate(constraints).flatten()

    # ── Inequality constraints: obstacle avoidance + two-sided quaternion ───
    def d(z, N=N):
        x, u = z_to_xu(z)
        x_pos = jnp.repeat(x[:, 0], n_obs)
        y_pos = jnp.repeat(x[:, 1], n_obs)
        obs_vals = (x_pos - obs_xc)**2 + (y_pos - obs_yc)**2 - obs_r2
        q_norm2 = x[:,3]**2 + x[:,4]**2 + x[:,5]**2 + x[:,6]**2
        return jnp.concatenate([obs_vals, q_norm2])

    gt = None

    # ── Variable bounds ─────────────────────────────────────────────────────
    dummy_z = jnp.arange(z_init.size)
    dummy_x, dummy_u = z_to_xu(dummy_z)
    x_u_indices = dummy_x[:, jnp.where(~jnp.isinf(quad_params["x_ub"]))[0]]
    x_l_indices = dummy_x[:, jnp.where(~jnp.isinf(quad_params["x_lb"]))[0]]
    u_u_indices = dummy_u[:, jnp.array([0,1,2,3])]
    u_l_indices = dummy_u[:, jnp.array([0,1,2,3])]
    z_l_i = jnp.hstack([*x_l_indices, *u_l_indices])
    z_u_i = jnp.hstack([*x_u_indices, *u_u_indices])
    x_l = jnp.zeros_like(x_l_indices) + quad_params["x_lb"][jnp.where(~jnp.isinf(quad_params["x_lb"]))[0]]
    x_u = jnp.zeros_like(x_u_indices) + quad_params["x_ub"][jnp.where(~jnp.isinf(quad_params["x_ub"]))[0]]
    u_l = jnp.zeros_like(u_l_indices) + quad_params["minWmotor"]
    u_u = jnp.zeros_like(u_u_indices) + quad_params["maxWmotor"]
    z_l = xu_to_z(x_l, u_l)
    z_u = xu_to_z(x_u, u_u)
    x_L = jnp.ones_like(z_init) * -jnp.inf
    x_L = x_L.at[z_l_i].set(z_l)
    x_U = jnp.ones_like(z_init) * jnp.inf
    x_U = x_U.at[z_u_i].set(z_u)

    d_L = jnp.concatenate([jnp.zeros(N * n_obs), jnp.full(N, 0.5)])
    d_U = jnp.concatenate([jnp.full(N * n_obs, jnp.inf), jnp.full(N, 1.5)])

    aux = [z_to_xu, xu_to_z, quad_params, xr_default, x0_default]

    return f, c, d, x_L, x_U, d_L, d_U, z_init, gt, aux


if __name__ == "__main__":

    import os
    with open("jaxipm/params.json") as f:
        p = json.load(f)

    # Allow disabling hot-restart from the env (for ablation runs).
    # if os.environ.get("HOT_RESTART", "1") == "0":
    #     p["hot_restarting"] = False
    #     print("[jaxipm] hot_restarting DISABLED via HOT_RESTART=0")
    # # DEBUG_MODE retains per-problem iter_buffer (otherwise iter_count is per-slot).
    # if os.environ.get("JAXIPM_DEBUG", "0") == "1":
    #     p["DEBUG_MODE"] = True
    #     print("[jaxipm] DEBUG_MODE=true via JAXIPM_DEBUG=1 — iter_buffer enabled")

    from time import time
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(p["gpu_id"])

    import jax
    import equinox as eqx

    jax.config.update("jax_enable_x64", True)
    # jax.config.update("jax_compilation_cache_dir", "/home/john/tmp/jax_cache")
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
    jax.config.update("jax_persistent_cache_enable_xla_caches", "xla_gpu_per_fusion_autotune_cache_dir")

    from jaxipm.utils.problem_format_utils import custom_to_cyipopt_format
    from jaxipm.solver import solve_throughput
    from jaxipm.initialization import initialize_common_problem, initialize_problem_regular

    N_horizon = N_HORIZON
    Ts_ = TS
    N_batch = tmp["batch_size"]
    N_RUNS = tmp["N_RUNS_jaxipm"]
    DELTA_STD = 0.5 # float(os.environ.get("DELTA_STD", "0.5"))

    for AVG_VEL in tmp["avg_vel"]:
        print(f"\n================ avg_vel = {AVG_VEL} ================")
        f, c, d, x_L, x_U, d_L, d_U, z_init, gt, aux = quadcopter_track_avoid(N=N_horizon, avg_vel=AVG_VEL)
        z_to_xu_fn, _, quad_params_local, xr_default, x0_default = aux

        # ── Pre-generate N_RUNS random valid (x0, xr) pairs (numpy + rejection) ─
        print(f"--- Generating {N_RUNS} random initial conditions (delta_std={DELTA_STD}) ---")
        all_x0_np, all_xr_np, all_s_np, n_rejects = generate_random_inits(
            N_RUNS, N_horizon, Ts_, AVG_VEL, delta_std=DELTA_STD, seed=0)
        print(f"  rejected {n_rejects} samples (kept {N_RUNS})")
        all_x0 = jnp.asarray(all_x0_np)
        all_xr = jnp.asarray(all_xr_np)

        # User args: f_args[1] = xr (default), c_args[1] = x0 (default)
        f_args = (xr_default,)
        c_args = (x0_default,)
        d_args = ()

        obj, obj_grad, obj_hess, constraints, bounds = custom_to_cyipopt_format(f, c, d, x_L, x_U, d_L, d_U, z_init)

        # jax.clear_caches()

        nx_total = N_horizon * 13 + (N_horizon - 1) * 4  # 506
        w_hover = 522.9847140714692

        def build_warm_start(x0_ic, xr_ref):
            # Match casadi/madnlp: linspace x,y,z from x0 to a fixed (0, 0, -2),
            # identity quaternion, zero velocities/angular rates, hover controls.
            # (xr_ref is unused but kept for the calc_next_problem signature.)
            del xr_ref
            end_xyz = jnp.array([0., 0., -2.])
            xyz = jnp.linspace(x0_ic[:3], end_xyz, N_horizon)
            ang_v_angv = jnp.tile(jnp.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
                                  (N_horizon, 1))
            states = jnp.hstack([xyz, ang_v_angv])  # (N, 13)
            inputs = jnp.full((N_horizon - 1, 4), w_hover)
            return jnp.concatenate([states.flatten(), inputs.flatten()])

        def calc_next_problem(rng_key, sol):
            # Pick a random idx into the pre-generated array. Sol is unused
            # (random sampling already provides diversity; no need to derive idx).
            new_idx = jax.random.randint(rng_key, (), 0, N_RUNS)
            new_x0 = all_x0[new_idx]
            new_xr = all_xr[new_idx]
            new_warm = build_warm_start(new_x0, new_xr)
            return (new_warm, (new_xr,), (new_x0,), ())

        cp = initialize_common_problem(f, c, d, x_L, x_U, d_L, d_U, z_init, p,
                                       [f_args, c_args, d_args],
                                       calc_next_problem=calc_next_problem)
        state = initialize_problem_regular(cp, z_init, args=[f_args, c_args, d_args])
        state = eqx.tree_at(lambda t: t.fl.needs_regular_init, state, jnp.array([[0]]))

        print("--- Testing solve_throughput ---")
        max_solves = N_RUNS

        def stack_states_tp(states):
            def stack_leaves(*leaves):
                if eqx.is_array(leaves[0]):
                    return jnp.stack(leaves)
                else:
                    return leaves[0]
            return jax.tree.map(stack_leaves, *states)

        batch_tp = stack_states_tp([state] * N_batch)

        # Inject the first N_batch unique (x0, xr) into the per-slot args.
        # Framework layout:
        #   state.args[0] (f_args) = (mu, x_ref_dummy, dr_x_dummy, scaling, *user_f_args)
        #   state.args[1] (c_args) = (scaling, *user_c_args)
        # so user xr lives at args[0][4] and user x0 lives at args[1][1].
        first_batch_x0 = all_x0[:N_batch]      # (N_batch, 13)
        first_batch_xr = all_xr[:N_batch]      # (N_batch, N_horizon, 13)
        batch_tp = eqx.tree_at(lambda s: s.args[1][1], batch_tp, first_batch_x0)
        batch_tp = eqx.tree_at(lambda s: s.args[0][4], batch_tp, first_batch_xr)

        # Pre-populate per-slot warm starts.
        first_batch_warms = jax.vmap(build_warm_start)(first_batch_x0, first_batch_xr)
        new_x = batch_tp.it.x.at[:, :cp.nx, 0].set(first_batch_warms)
        batch_tp = eqx.tree_at(lambda s: s.it.x, batch_tp, new_x)

        rng_key = jax.random.PRNGKey(0)
        _solve_throughput = eqx.filter_jit(solve_throughput, donate="none")

        print(f"JAXIPM: JIT warmup ({max_solves} solves) — first call includes compilation")
        t1 = time()
        tp_out = _solve_throughput(
            rng_key, cp, batch_tp, max_solves, max_iter_per_solve=cp.p["max_iter"]
        )
        jax.block_until_ready(tp_out[0])
        t2 = time()
        print(f"JAXIPM: warmup wall time {(t2-t1)*1000:.1f} ms")

        print(f"JAXIPM: running {max_solves} solves (warm)")
        t1 = time()
        tp_out = _solve_throughput(
            rng_key, cp, batch_tp, max_solves, max_iter_per_solve=cp.p["max_iter"]
        )
        jax.block_until_ready(tp_out[0])
        t2 = time()
        total_time = t2 - t1
        print(f"JAXIPM: warm wall time {total_time*1000:.1f} ms")

        # final_state, solution_buffer, write_idx, term_buffer, iter_buffer = tp_out
        final_state, solution_buffer, write_idx = tp_out
        iter_buffer = None
        term_buffer = None

        n_collected = min(int(write_idx), int(solution_buffer.shape[0]))
        print(f"JAXIPM: collected {n_collected} (write_idx reported {int(write_idx)})")

        # ── Save results in shared schema ───────────────────────────────────────
        sol_buf_np = np.asarray(solution_buffer[:n_collected, :cp.nx])
        iter_counts_np = np.asarray(final_state.iter_count.squeeze()).reshape(-1)

        X_all = np.full((n_collected, N_horizon, 13), np.nan, dtype=np.float64)
        U_all = np.full((n_collected, N_horizon - 1, 4), np.nan, dtype=np.float64)
        starts_executed = np.zeros((n_collected, 13), dtype=np.float64)
        for i in range(n_collected):
            x_i, u_i = z_to_xu_fn(sol_buf_np[i])
            X_all[i] = np.asarray(x_i)
            U_all[i] = np.asarray(u_i)
            starts_executed[i] = X_all[i, 0]  # the IC, recoverable from x[0]

        # ── Per-problem objective values (post-hoc evaluation of f on saved z) ──
        # Slot ordering during hot-restart cycling is non-deterministic; recover
        # each collection's (x0, xr) pairing by nearest-start matching against
        # the random-init pool. Position-only matching is unambiguous because
        # each slot's IC is sampled from a continuous distribution.
        obj_vals = np.zeros(n_collected, dtype=np.float64)
        for i in range(n_collected):
            x0_i = X_all[i, 0]
            d_to_pool = np.linalg.norm(all_x0_np - x0_i, axis=1)
            idx = int(np.argmin(d_to_pool))
            z_i = jnp.asarray(sol_buf_np[i])
            xr_i = jnp.asarray(all_xr_np[idx])
            obj_vals[i] = float(f(z_i, xr_i))

        times = np.full(n_collected, total_time / max(n_collected, 1), dtype=np.float64)
        if iter_buffer is not None:
            iters = np.asarray(iter_buffer)[:n_collected].astype(np.int32)
        elif iter_counts_np.size >= n_collected:
            iters = iter_counts_np[:n_collected].astype(np.int32)
        else:
            iters = np.full(n_collected, -1, dtype=np.int32)
        success = np.ones(n_collected, dtype=bool)

        logs_dir = os.path.join(os.path.dirname(__file__), "logs")
        os.makedirs(logs_dir, exist_ok=True)
        out_path = os.path.join(logs_dir, f"jaxipm_v{AVG_VEL:.1f}_results.npz")
        if term_buffer is not None:
            terms = np.asarray(term_buffer)[:n_collected].astype(np.int32)
        else:
            terms = np.full(n_collected, -1, dtype=np.int32)

        np.savez(
            out_path,
            X_all=X_all,
            U_all=U_all,
            times=times,
            iters=iters,
            obj_vals=obj_vals,
            terms=terms,
            success=success,
            starts=starts_executed,
            all_x0=all_x0_np,        # (N_RUNS, 13) — full pool of pre-generated starts
            all_xr=all_xr_np,        # (N_RUNS, N, 13) — full pool of references
            all_s=all_s_np,          # (N_RUNS,) — pringle parameter for each
            total_time=np.array([total_time]),
            N=np.array([N_horizon]),
            Ts=np.array([Ts_]),
            N_RUNS=np.array([max_solves]),
            N_BATCH=np.array([N_batch]),
            avg_vel=np.array([AVG_VEL]),
        )
        print(f"JAXIPM: saved {out_path}")
        print(f"JAXIPM: total_time={total_time:.2f}s, collected={n_collected}/{max_solves}")

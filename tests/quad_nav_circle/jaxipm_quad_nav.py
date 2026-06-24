import json
import numpy as np
import jax.numpy as jnp
from tests.quad_jax_dynamics import f_jnp

with open("tests/quad_params.json") as f:
    quad_params = json.load(f)
    quad_params["IB"] = jnp.diag(jnp.array([quad_params["IB00"], quad_params["IB11"], quad_params["IB22"]]))

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

    # Obstacles (base radius `r`; the quad's own radius is added on below)
    OBSTACLES = tmp["obstacles"]
    QUAD_RADIUS = tmp["quad_radius"]

def quadcopter_nav(N=30):
    nx, nu, Ts = 13, 4, TS

    def xu_to_z(x, u):
        return jnp.hstack([x.flatten(), u.flatten()])

    def z_to_xu(z):
        x = z[:N*nx].reshape(N, nx)
        u = z[N*nx:].reshape(N-1, nu)
        return x, u
    
    start_xyz = jnp.array([4,4,0])
    end_xyz = jnp.array([0,0,-2])
    xyz = jnp.linspace(start_xyz, end_xyz, N)

    # make initial optimization iterate a linear interpolation between start and end
    ang_v_angv = jnp.array([[1,0,0,0,0,0,0,0,0,0]]*N)
    state0 = jnp.hstack([xyz, ang_v_angv])
    input0 = jnp.array([[522.9847140714692]*4]*(N-1))
    z_init = xu_to_z(state0, input0)

    def f(z):
        Q = jnp.array([1,1,1,0, 0, 0, 0, 1, 1, 1, 1,1,1])
        R = jnp.array([0,   0,   0,   0])
        x, u = z_to_xu(z)
        cost = jnp.sum(Q * x**2) # + jnp.sum(R * (u**2))
        return cost    

    # Equality constraints — x0_ic is a runtime c_arg so we can vary it per element
    def c(z, x0_ic=state0[0], N=N, Ts=Ts):
        x, u = z_to_xu(z)
        constraints = []
        constraints.append(x[0] - x0_ic)
        for k in range(N-1):
            x_next = x[k] + f_jnp(x[k], u[k], quad_params) * Ts
            constraints.append(x[k+1] - x_next)
        return jnp.concatenate(constraints).flatten()
    
    xc_ = [o["xc"] for o in OBSTACLES]
    yc_ = [o["yc"] for o in OBSTACLES]
    r_ = [o["r"] for o in OBSTACLES]
    quad_radius = QUAD_RADIUS
    r__ = [i+quad_radius for i in r_] # base radius + the quad's own radius

    # Inequality constraints
    def d(z, N=N):
        x, u = z_to_xu(z)
        cl = []
        for k in range(N-1):
            multiplier = 1 + k*Ts * 0.1
            for (xc, yc, r) in zip(xc_, yc_, r__):
                # these terms >= 0
                cl.append((x[k,0] - xc)**2 + (x[k,1] - yc)**2 - r**2 * multiplier)
        return jnp.hstack(cl)
    
    gt = None

    # optimization variable bounds
    z_L = jnp.ones_like(z_init)*-jnp.inf
    z_U = jnp.ones_like(z_init)*jnp.inf

    # # define optimization variable bounds
    dummy_z = jnp.arange(z_init.size) # get arange of each z to map to xu
    dummy_x, dummy_u = z_to_xu(dummy_z)
    x_u_indices = dummy_x[:,jnp.where(~jnp.isinf(quad_params["x_ub"]))[0]]
    x_l_indices = dummy_x[:,jnp.where(~jnp.isinf(quad_params["x_lb"]))[0]]
    u_u_indices = dummy_u[:,jnp.array([0,1,2,3])]
    u_l_indices = dummy_u[:,jnp.array([0,1,2,3])]
    z_l_i = jnp.hstack([*x_l_indices, *u_l_indices])
    z_u_i = jnp.hstack([*x_u_indices, *u_u_indices])
    x_l = jnp.zeros_like(x_l_indices) + quad_params["x_lb"][jnp.where(~jnp.isinf(quad_params["x_lb"]))[0]]
    x_u = jnp.zeros_like(x_u_indices) + quad_params["x_ub"][jnp.where(~jnp.isinf(quad_params["x_ub"]))[0]]
    u_l = jnp.zeros_like(u_l_indices) + quad_params["minWmotor"]
    u_u = jnp.zeros_like(u_u_indices) + quad_params["maxWmotor"]
    z_l = xu_to_z(x_l, u_l)
    z_u = xu_to_z(x_u, u_u)
    x_L = jnp.ones_like(z_init)*-jnp.inf
    x_L = x_L.at[z_l_i].set(z_l)
    x_U = jnp.ones_like(z_init)*jnp.inf
    x_U = x_U.at[z_u_i].set(z_u)

    # slack variable bounds
    d_L = jnp.zeros(d(z_init).shape) # only one sided general inequality constraints
    d_U = jnp.ones(d(z_init).shape)*jnp.inf # only one sided general inequality constraints

    aux = [z_to_xu, xu_to_z, quad_params, state0[0]]

    return f, c, d, z_L, z_U, d_L, d_U, z_init, gt, aux

if __name__ == "__main__":
   
    with open("jaxipm/params.json") as f:
        p = json.load(f)

    # import os
    # if os.environ.get("HOT_RESTART", "1") == "0":
    #     p["hot_restarting"] = False
    #     print("[jaxipm] hot_restarting DISABLED via HOT_RESTART=0")
    # if os.environ.get("JAXIPM_DEBUG", "0") == "1":
    #     p["DEBUG_MODE"] = True
    #     print("[jaxipm] DEBUG_MODE=true via JAXIPM_DEBUG=1 — iter_buffer enabled")
    # from time import time
    # os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    # os.environ['CUDA_VISIBLE_DEVICES'] = str(p["gpu_id"])

    import jax
    import equinox as eqx
    from time import time
    import os

    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_compilation_cache_dir", "/home/john/tmp/jax_cache")
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
    jax.config.update("jax_persistent_cache_enable_xla_caches", "xla_gpu_per_fusion_autotune_cache_dir")

    # translation tooling
    from jaxipm.utils.problem_format_utils import custom_to_cyipopt_format
    from jaxipm.solver import solve_throughput
    from jaxipm.initialization import initialize_common_problem, initialize_problem_regular

    f, c, d, x_L, x_U, d_L, d_U, x0, gt, aux = quadcopter_nav()
    z_to_xu_fn, _, quad_params_local, default_x0_ic = aux

    # ── Set up the angle linspace and per-angle x0 array ────────────────────
    # Sector centered at π/4 (45°); SECTOR_DEG controls total arc width.
    #   SECTOR_DEG=90  → [0, π/2]   (original "corner" sector)
    #   SECTOR_DEG=180 → [-π/4, 3π/4]
    N_RUNS = tmp["N_RUNS_jaxipm"]
    N_batch = tmp["batch_size"]
    N_horizon = N_HORIZON  # MPC horizon (matches quadcopter_nav internal N)
    RADIUS = float(np.sqrt(4.0**2 + 4.0**2))  # ~5.657, matches default x0=[4,4,0]
    for SECTOR_DEG in tmp["init_angles"]:

        sector_half_rad = np.deg2rad(SECTOR_DEG) / 2.0
        sector_center_rad = np.pi / 4.0
        angles_np = np.linspace(sector_center_rad - sector_half_rad,
                                sector_center_rad + sector_half_rad, N_RUNS)
        all_x0_starts_np = np.zeros((N_RUNS, 13), dtype=np.float64)
        all_x0_starts_np[:, 0] = RADIUS * np.cos(angles_np)
        all_x0_starts_np[:, 1] = RADIUS * np.sin(angles_np)
        all_x0_starts_np[:, 3] = 1.0  # q0 = 1
        all_x0_starts = jnp.asarray(all_x0_starts_np)

        # User c_args carry the runtime IC
        c_args = (all_x0_starts[0],)
        f_args = ()
        d_args = ()

        obj, obj_grad, obj_hess, constraints, bounds = custom_to_cyipopt_format(f, c, d, x_L, x_U, d_L, d_U, x0)
        save_dir = "ipopt_logs"

        jax.clear_caches()
        _user_args = (
            tuple(jnp.asarray(a) for a in f_args),
            tuple(jnp.asarray(a) for a in c_args),
            tuple(jnp.asarray(a) for a in d_args),
        )

        nx_horizon = 13
        nu_horizon = 4
        nx_total = N_horizon * nx_horizon + (N_horizon - 1) * nu_horizon  # 506
        w_hover = 522.9847140714692
        end_xyz = jnp.array([0.0, 0.0, -2.0])
        ts_lin = jnp.linspace(0.0, 1.0, N_horizon)[:, None]

        def build_warm_start(x0_ic):
            start_xyz = x0_ic[:3]
            xyz = (1 - ts_lin) * start_xyz + ts_lin * end_xyz             # (N, 3)
            states = jnp.zeros((N_horizon, nx_horizon)).at[:, :3].set(xyz).at[:, 3].set(1.0)
            inputs = jnp.full((N_horizon - 1, nu_horizon), w_hover)
            return jnp.concatenate([states.flatten(), inputs.flatten()])  # (nx_total,)

        sector_lo = sector_center_rad - sector_half_rad
        sector_hi = sector_center_rad + sector_half_rad

        def calc_next_problem(key, sol):
            # sol shape: (cp.nx,) per element. sol[0:13] is x[t=0] = previous x0_ic
            theta = jnp.arctan2(sol[1], sol[0])
            theta = jnp.clip(theta, sector_lo, sector_hi)
            # Map theta ∈ [sector_lo, sector_hi] → idx ∈ [0, N_RUNS-1]
            idx = jnp.round((theta - sector_lo) / (sector_hi - sector_lo)
                            * (N_RUNS - 1)).astype(jnp.int32)
            new_idx = (idx + N_batch) % N_RUNS
            new_x0_ic = all_x0_starts[new_idx]
            new_warm = build_warm_start(new_x0_ic)
            return (new_warm, (), (new_x0_ic,), ())

        cp = initialize_common_problem(f, c, d, x_L, x_U, d_L, d_U, x0, p, [f_args, c_args, d_args],
                                       calc_next_problem=calc_next_problem)
        state = initialize_problem_regular(cp, x0, args=[f_args, c_args, d_args])
        state = eqx.tree_at(lambda t: t.fl.needs_regular_init, state, jnp.array([[0]]))

        print("\n--- Testing solve_throughput ---")
        max_solves = N_RUNS

        def stack_states_tp(states):
            def stack_leaves(*leaves):
                if eqx.is_array(leaves[0]):
                    return jnp.stack(leaves)
                else:
                    return leaves[0]
            return jax.tree.map(stack_leaves, *states)

        batch_tp = stack_states_tp([state] * N_batch)

        # Inject the first N_batch unique x0_ic values into c_args[1] of each element
        first_batch_x0 = all_x0_starts[:N_batch]  # (N_batch, 13)
        batch_tp = eqx.tree_at(
            lambda s: s.args[1][1],
            batch_tp,
            first_batch_x0,
        )

        # Pre-populate the per-element warm starts (linear interp from each x0_ic to origin)
        first_batch_warms = jax.vmap(build_warm_start)(first_batch_x0)  # (N_batch, nx_total)
        new_x = batch_tp.it.x.at[:, :cp.nx, 0].set(first_batch_warms)
        batch_tp = eqx.tree_at(lambda s: s.it.x, batch_tp, new_x)

        rng_key = jax.random.PRNGKey(0)
        _solve_throughput = eqx.filter_jit(solve_throughput, donate="none")

        # Warmup (includes JIT compilation)
        t1 = time()
        tp_out = _solve_throughput(
            rng_key, cp, batch_tp, max_solves, max_iter_per_solve=500
        )
        jax.block_until_ready(tp_out[0])
        t2 = time()
        print(f"Throughput (incl. JIT): {(t2-t1)*1000:.1f} ms")

        # Timed run
        t1 = time()
        tp_out = _solve_throughput(
            rng_key, cp, batch_tp, max_solves, max_iter_per_solve=500
        )
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
        print(f"JAXIPM nav: collected {n_collected} (write_idx reported {int(write_idx)})")

        # ── Save results in shared schema ───────────────────────────────────────
        Ts_ = TS
        sol_buf_np = np.asarray(solution_buffer[:n_collected, :cp.nx])
        iter_counts_np = np.asarray(final_state.iter_count.squeeze()).reshape(-1)

        X_all = np.full((n_collected, N_horizon, 13), np.nan, dtype=np.float64)
        for i in range(n_collected):
            x_i, _ = z_to_xu_fn(sol_buf_np[i])
            X_all[i] = np.asarray(x_i)

        # Recover the actual angle of each saved trajectory from x[0]
        starts_executed = X_all[:, 0, :]                # (n_collected, 13)
        angles_executed = np.arctan2(starts_executed[:, 1], starts_executed[:, 0])

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
        success = np.ones(n_collected, dtype=bool)

        logs_dir = os.path.join(os.path.dirname(__file__), "logs")
        os.makedirs(logs_dir, exist_ok=True)
        out_path = os.path.join(logs_dir, f"jaxipm_sector{int(SECTOR_DEG)}_results.npz")
        if term_buffer is not None:
            terms = np.asarray(term_buffer)[:n_collected].astype(np.int32)
        else:
            terms = np.full(n_collected, -1, dtype=np.int32)

        np.savez(
            out_path,
            X_all=X_all,
            times=times,
            iters=iters,
            obj_vals=obj_vals,
            terms=terms,
            success=success,
            starts=starts_executed,
            angles=angles_executed,
            all_angles=angles_np,
            total_time=np.array([total_time]),
            N=np.array([N_horizon]),
            Ts=np.array([Ts_]),
            N_RUNS=np.array([max_solves]),
            radius=np.array([RADIUS]),
            sector_deg=np.array([SECTOR_DEG]),
        )
        print(f"JAXIPM nav: saved {out_path}")
        print(f"JAXIPM nav: total_time={total_time:.2f}s, collected={n_collected}/{max_solves}")

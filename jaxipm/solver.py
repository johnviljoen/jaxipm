# We execute the solver from here


import equinox as eqx
import jax
import jax.numpy as jnp

from jaxipm.search import execute_search, post_process

class TerminationCode:
    CONTINUE = 0
    CONVERGED = 1
    MAX_ITER_EXCEEDED = 2
    TINY_STEP_BREAK = 3
    RESTORATION_FAILURE = 4
    ACCEPTABLE_POINT = 5

def save_kkt_systems(state, cp):
    lhs_coo_indices = cp.coo_indices
    lhs_data = state.ic.perturbed_data
    rhs_it = state.cqpo.rhs
    rhs = jnp.vstack([rhs_it.x[:cp.nx], rhs_it.s, rhs_it.y_c, rhs_it.y_d]).flatten()
    filename = f"data/kkt_systems/iter_{state.iter_count.squeeze()}.npz"
    jnp.savez(filename, lhs_coo_indices=lhs_coo_indices, lhs_data=lhs_data, rhs=rhs)

def solve(cp, state, max_iter=None, fill=None, debug=False):
    if max_iter is None: max_iter = cp.p["max_iter"]
    def cond(carry):
        state, term, i = carry
        return ((term == TerminationCode.CONTINUE) & (i < max_iter)).squeeze()
    def body(carry):
        state, _, i = carry
        orig = state
        result = execute_search(state, cp)
        state, term = post_process(orig, result, cp)
        if debug is True:
            jax.debug.callback(save_kkt_systems, state, cp)
        return state, term, i + 1
    state, term, _ = jax.lax.while_loop(cond, body, (state, jnp.array([[TerminationCode.CONTINUE]]), 0))
    return state, term

def solve_batched(cp, batch_state, max_iter=None):
    fill = jax.tree.map(lambda x: x[0], batch_state)
    solve_vmapped = eqx.filter_vmap(solve, in_axes=(None, 0, None, None), axis_name="batch")
    return solve_vmapped(cp, batch_state, max_iter, fill)

def solve_throughput(key, cp, batch_state, max_solves, max_iter_per_solve=None):
    """Batched throughput solver: elements that converge auto-restart with new problems.

    Solutions are collected into a flat buffer of known size (max_solves, nx).
    Each iteration, converged elements have their solutions written at write_idx
    via prefix-sum scatter, then post_process triggers their re-initialization.

    Args:
        key: JAX PRNGKey for generating new problems on restart
        cp: CommonProblem (shared, not vmapped)
        batch_state: batched OptimizationState (vmapped dim 0)
        max_solves: total solutions to collect across entire batch
        max_iter_per_solve: safety cap on total iterations (default: enough headroom)

    Returns:
        (final_state, solution_buffer, write_idx)
    """
    if max_iter_per_solve is None:
        max_iter_per_solve = cp.p["max_iter"]

    batch_size = jax.tree.leaves(batch_state)[0].shape[0]
    safety_budget = max_iter_per_solve * (max_solves // batch_size + 2)

    solution_buffer = jnp.zeros((max_solves, cp.nx))
    write_idx = jnp.array(0)

    vmapped_search = eqx.filter_vmap(execute_search, in_axes=(0, None))
    vmapped_post = eqx.filter_vmap(post_process, in_axes=(0, 0, None))

    if cp.p["DEBUG_MODE"]:
        term_buffer = jnp.zeros(max_solves, dtype=jnp.int32)
        iter_buffer = jnp.zeros(max_solves, dtype=jnp.int32)
        # `done_mask` latches "this slot has reached any terminal code since its
        # last (re-init or start)". Used both to (a) suppress re-scattering of
        # post-convergence drifted state and (b) exit the loop early in WS mode
        # once every slot is finished. In HR mode the latch is cleared on
        # re-seed so the slot can scatter again on its next problem.
        done_mask0 = jnp.zeros(batch_size, dtype=jnp.bool_)

        def cond(carry):
            state, buf, tbuf, ibuf, done, widx, i, rng_key = carry
            # Exit when buffer full, safety budget exhausted, or every slot
            # has reached a terminal state and no more progress is possible.
            return (widx < max_solves) & (i < safety_budget) & (~done.all())

        def body(carry):
            state, buf, tbuf, ibuf, done, widx, i, rng_key = carry
            orig = state
            result = vmapped_search(state, cp)
            state, terminate = vmapped_post(orig, result, cp)

            # Scatter only the FIRST time a slot reaches a successful terminal
            # code (1 or 5) since its last (re-init / start). The `done` latch
            # blocks re-scattering of slots whose post-convergence state may
            # drift on subsequent IPM steps.
            term_codes = terminate.squeeze()
            iter_counts = state.iter_count.squeeze()
            solved_mask = ((term_codes == 1) | (term_codes == 5)) & ~done
            solutions = state.it.x[:, :cp.nx, 0]  # (batch_size, nx)

            # Prefix-sum scatter: compute write position for each solved element
            offsets = jnp.cumsum(solved_mask) - 1  # 0-based offset among solved elements
            write_positions = widx + offsets  # global buffer positions

            # Vectorized scatter: unsolved elements target a dummy position and get masked out
            safe_positions = jnp.where(solved_mask, write_positions, max_solves - 1)
            buf = buf.at[safe_positions].set(
                jnp.where(solved_mask[:, None], solutions, buf[safe_positions])
            )
            tbuf = tbuf.at[safe_positions].set(
                jnp.where(solved_mask, term_codes, tbuf[safe_positions])
            )
            ibuf = ibuf.at[safe_positions].set(
                jnp.where(solved_mask, iter_counts, ibuf[safe_positions])
            )
            widx = widx + solved_mask.sum()

            # --- Restart terminated elements with new problems (HR mode only) ---
            # In WS mode (hot_restarting=false) slots stay on their original problem
            # after termination — search.py's init_regular machinery is what makes
            # re-seeding sound, and that's only triggered when hot_restarting is on.
            if cp.p["hot_restarting"]:
                should_restart = (terminate > 0).squeeze()  # (batch_size,)

                # Split rng: 1 new root + batch_size subkeys
                rng_key, *subkeys = jax.random.split(rng_key, batch_size + 1)
                subkeys = jnp.stack(subkeys)  # (batch_size, 2)

                # Get next problems for ALL elements (select only where needed)
                next_x0, next_f, next_c, next_d = jax.vmap(cp.calc_next_problem)(
                    subkeys, state.it.x[:, :cp.nx, 0]
                )

                # Splice new user args into existing prefix structure
                f_curr, c_curr, d_curr = state.args
                next_args = (
                    (*f_curr[:4], *next_f),
                    (c_curr[0], *next_c),
                    (d_curr[0], *next_d),
                )

                # Only apply to restarting elements
                new_args = jax.tree.map(
                    lambda n, o: jnp.where(
                        jnp.reshape(should_restart, (-1,) + (1,) * (n.ndim - 1)), n, o
                    ),
                    next_args, state.args
                )
                new_x = jnp.where(
                    should_restart[:, None, None],
                    state.it.x.at[:, :cp.nx].set(next_x0[:, :, None]),
                    state.it.x
                )
                state = eqx.tree_at(lambda t: (t.it.x, t.args), state, (new_x, new_args))
                # Clear latch for re-seeded slots so they can scatter again.
                next_done = (done | (term_codes > 0)) & ~should_restart
            else:
                # WS mode: latch stays True once set (any terminal code).
                next_done = done | (term_codes > 0)

            return state, buf, tbuf, ibuf, next_done, widx, i + 1, rng_key

        final_state, solution_buffer, term_buffer, iter_buffer, _, write_idx, _, _ = jax.lax.while_loop(
            cond, body,
            (batch_state, solution_buffer, term_buffer, iter_buffer,
             done_mask0, write_idx, 0, key)
        )
        return final_state, solution_buffer, write_idx, term_buffer, iter_buffer

    else:
        done_mask0 = jnp.zeros(batch_size, dtype=jnp.bool_)

        def cond(carry):
            state, buf, done, widx, i, rng_key = carry
            return (widx < max_solves) & (i < safety_budget) & (~done.all())

        def body(carry):
            state, buf, done, widx, i, rng_key = carry
            orig = state
            result = vmapped_search(state, cp)
            state, terminate = vmapped_post(orig, result, cp)

            term_codes = terminate.squeeze()
            solved_mask = ((term_codes == 1) | (term_codes == 5)) & ~done
            solutions = state.it.x[:, :cp.nx, 0]  # (batch_size, nx)

            # Prefix-sum scatter: compute write position for each solved element
            offsets = jnp.cumsum(solved_mask) - 1  # 0-based offset among solved elements
            write_positions = widx + offsets  # global buffer positions

            # Vectorized scatter: unsolved elements target a dummy position and get masked out
            safe_positions = jnp.where(solved_mask, write_positions, max_solves - 1)
            buf = buf.at[safe_positions].set(
                jnp.where(solved_mask[:, None], solutions, buf[safe_positions])
            )
            widx = widx + solved_mask.sum()

            # --- Restart terminated elements with new problems (HR mode only) ---
            if cp.p["hot_restarting"]:
                should_restart = (terminate > 0).squeeze()  # (batch_size,)

                # Split rng: 1 new root + batch_size subkeys
                rng_key, *subkeys = jax.random.split(rng_key, batch_size + 1)
                subkeys = jnp.stack(subkeys)  # (batch_size, 2)

                # Get next problems for ALL elements (select only where needed)
                next_x0, next_f, next_c, next_d = jax.vmap(cp.calc_next_problem)(
                    subkeys, state.it.x[:, :cp.nx, 0]
                )

                # Splice new user args into existing prefix structure
                f_curr, c_curr, d_curr = state.args
                next_args = (
                    (*f_curr[:4], *next_f),
                    (c_curr[0], *next_c),
                    (d_curr[0], *next_d),
                )

                # Only apply to restarting elements
                new_args = jax.tree.map(
                    lambda n, o: jnp.where(
                        jnp.reshape(should_restart, (-1,) + (1,) * (n.ndim - 1)), n, o
                    ),
                    next_args, state.args
                )
                new_x = jnp.where(
                    should_restart[:, None, None],
                    state.it.x.at[:, :cp.nx].set(next_x0[:, :, None]),
                    state.it.x
                )
                state = eqx.tree_at(lambda t: (t.it.x, t.args), state, (new_x, new_args))
                next_done = (done | (term_codes > 0)) & ~should_restart
            else:
                next_done = done | (term_codes > 0)

            return state, buf, next_done, widx, i + 1, rng_key

        final_state, solution_buffer, _, write_idx, _, _ = jax.lax.while_loop(
            cond, body,
            (batch_state, solution_buffer, done_mask0, write_idx, 0, key)
        )
        return final_state, solution_buffer, write_idx

if __name__ == "__main__":
    from jaxipm.utils.validation_boilerplate import *
    from jaxipm.initialization import initialize_common_problem, initialize_skeleton_state, initialize_problem_regular
    from time import time

    # if os.path.exists("ipopt_logs"):
    #     shutil.rmtree("ipopt_logs")

    # result = minimize_ipopt(
    #     fun=obj,
    #     x0=x0,
    #     jac=obj_grad,
    #     hess=obj_hess,
    #     constraints=constraints,
    #     bounds=bounds,
    #     options={
    #         "disp": 5,
    #         "maxiter": 23,
    #         # "mu_strategy": "monotone", # forces filter entries to happen pretty much
    #         # "start_with_resto": "yes",  # lets us test feas resto initialization as well
    #     },
    # )

    # we are holding onto some data we dont want to hold onto
    jax.clear_caches()

    _user_args = (
        tuple(jnp.asarray(a) for a in f_args),
        tuple(jnp.asarray(a) for a in c_args),
        tuple(jnp.asarray(a) for a in d_args),
    )
    _x0_flat = x0.squeeze()
    calc_next_problem = lambda key, sol: (_x0_flat, *_user_args)

    cp = initialize_common_problem(f, c, d, x_L, x_U, d_L, d_U, x0, p, [f_args, c_args, d_args],
                                   calc_next_problem=calc_next_problem)
    # state = initialize_skeleton_state(cp, x0, args=[f_args, c_args, d_args])
    state = initialize_problem_regular(cp, x0, args=[f_args, c_args, d_args])
    state = eqx.tree_at(lambda t: t.fl.needs_regular_init, state, jnp.array([[0]]))

    # _solve = eqx.filter_jit(solve)
    # out = _solve(cp, state, max_iter=100, debug=True)

    # with jax.profiler.trace("tmp/single_nmpc", create_perfetto_trace=True):
    #     # Run the operations to be profiled
    #     t1 = time()
    #     out = _solve(cp, state, max_iter=100)
    #     jax.block_until_ready(out)
    #     t2 = time()
    #     print(F"TIME: {(t2-t1)/20}")

    # TESTING THROUGHPUT -------------------------------------------------------

    print("\n--- Testing solve_throughput ---")
    N = 2000
    max_solves = 10000

    def stack_states_tp(states):
        def stack_leaves(*leaves):
            if eqx.is_array(leaves[0]):
                return jnp.stack(leaves)
            else:
                return leaves[0]
        return jax.tree.map(stack_leaves, *states)

    batch_tp = stack_states_tp([state] * N)

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
    print(f"Throughput (warm):      {(t2-t1)*1000:.1f} ms")

    final_state, solution_buffer, write_idx = tp_out

    print(f"Solutions collected: {write_idx}")
    print(f"Solution buffer shape: {solution_buffer.shape}")
    print(f"Final iter counts: {final_state.iter_count.squeeze()}")

    # Diagnostic: inspect collected solutions
    n_valid = min(int(write_idx), max_solves)

    # unique_terms, term_counts = jnp.unique(terms, return_counts=True)
    # print(f"\n--- DEBUG: Solution diagnostics ---")
    # print(f"Term code distribution: { {int(k): int(v) for k, v in zip(unique_terms, term_counts)} }")
    # print(f"Iter count range: [{int(iters.min())}, {int(iters.max())}], mean: {float(iters.mean()):.1f}")
    # suspicious = (iters <= 1)
    # if suspicious.any():
    #     idxs = jnp.where(suspicious)[0]
    #     print(f"WARNING: {int(suspicious.sum())} solutions with iter_count <= 1 at buffer indices: {idxs[:20]}")
    #     print(f"  Their term codes: {terms[idxs[:20]]}")
    #     pass

    # POST-HOC KKT OPTIMALITY CHECK -------------------------------------------
    # Check 3 independent conditions (without reconstructing bound multipliers):
    #   1. Primal feasibility: ||c(x)||, d bound violations
    #   2. Stationarity: LS residual — do multipliers exist making grad_L ≈ 0?
    #   3. Bound feasibility: x_L <= x <= x_U
    print("\n--- Post-hoc KKT optimality check ---")

    _mu_zero = jnp.zeros((1, 1))
    _x_ref = jnp.zeros(cp.nx)
    _dr_x = jnp.ones(cp.nx)
    _df = jnp.array(1.0)
    _dc = jnp.ones(cp.nyc)
    _dd = jnp.ones(cp.nyd)
    _f_args = (_mu_zero, _x_ref, _dr_x, _df)
    _c_args = (_dc,)
    _d_args = (_dd,)
    _ny = cp.nyc + cp.nyd

    def check_solution_kkt(x_flat):
        x_col = x_flat[:, None]  # (nx,) -> (nx, 1)

        # 1. Primal feasibility — constraint violation
        c_val = cp.nstqf.calc_c(x_col, *_c_args)   # (nyc, 1)
        d_val = cp.nstqf.calc_d(x_col, *_d_args)   # (nyd, 1)
        constr_viol = cp.nstqf.calc_nlp_constr_viol(c_val, d_val).squeeze()

        # 2. Stationarity — LS residual via CG (sparse @ dense only)
        jac_f = cp.nstqf.calc_jac_f(x_col, *_f_args)   # sparse
        jac_c = cp.nstqf.calc_jac_c(x_col, *_c_args)   # sparse
        jac_d = cp.nstqf.calc_jac_d(x_col, *_d_args)   # sparse
        jc_nx = jac_c[:cp.nx]
        jd_nx = jac_d[:cp.nx]
        grad_f = jac_f[:cp.nx].todense()   # (nx, 1) — gradient vector, tiny

        # Solve [Jc^T | Jd^T] @ [y_c; y_d] = -grad_f  via CG on normal equations
        neg_gf = -grad_f
        ATb = jnp.vstack([jc_nx.T @ neg_gf, jd_nx.T @ neg_gf]).squeeze()

        def ata_matvec(v):
            v_c, v_d = v[:cp.nyc, None], v[cp.nyc:, None]
            Av = jc_nx @ v_c + jd_nx @ v_d
            return jnp.vstack([jc_nx.T @ Av, jd_nx.T @ Av]).squeeze()

        y_all, _ = jax.scipy.sparse.linalg.cg(ata_matvec, ATb, maxiter=_ny)
        y_c = y_all[:cp.nyc, None]
        y_d = y_all[cp.nyc:, None]

        # Stationarity residual: component of grad_f not in range([Jc^T | Jd^T])
        residual = grad_f + jc_nx @ y_c + jd_nx @ y_d   # (nx, 1)
        stationarity = jnp.linalg.norm(residual, ord=jnp.inf).squeeze()

        # 3. Bound feasibility
        if cp.nxL > 0:
            x_viol_L = jnp.max(jnp.maximum(cp.x_L[cp.ind_x_L, None] - x_col[cp.ind_x_L], 0.0))
        else:
            x_viol_L = 0.0
        if cp.nxU > 0:
            x_viol_U = jnp.max(jnp.maximum(x_col[cp.ind_x_U] - cp.x_U[cp.ind_x_U, None], 0.0))
        else:
            x_viol_U = 0.0
        bound_viol = jnp.maximum(x_viol_L, x_viol_U)

        return constr_viol, stationarity, bound_viol

    valid_solutions = solution_buffer[:n_valid]
    constr_viols, stationarity, bound_viols = jax.vmap(check_solution_kkt)(valid_solutions)

    tol_cv = p["constr_viol_tol"]  # 1e-4
    tol_di = p["dual_inf_tol"]     # 1.0
    print(f"Tolerances: constr_viol={tol_cv}, dual_inf(stationarity)={tol_di}")
    print(f"  constr_viol:   max={float(constr_viols.max()):.6e}, mean={float(constr_viols.mean()):.6e}")
    print(f"  stationarity:  max={float(stationarity.max()):.6e}, mean={float(stationarity.mean()):.6e}")
    print(f"  bound_viol:    max={float(bound_viols.max()):.6e}, mean={float(bound_viols.mean()):.6e}")

    bad_cv = constr_viols > tol_cv
    bad_st = stationarity > tol_di
    bad_bv = bound_viols > tol_cv
    bad_any = bad_cv | bad_st | bad_bv
    print(f"\nFailing: {int(bad_any.sum())} / {n_valid}  "
          f"(constr={int(bad_cv.sum())}, station={int(bad_st.sum())}, bounds={int(bad_bv.sum())})")

    if bad_any.any():
        bad_idxs = jnp.where(bad_any)[0]
        print(f"\n  All failing indices: {bad_idxs[:50]}")
        print(f"    constr_viol:  {constr_viols[bad_idxs[:50]]}")
        print(f"    stationarity: {stationarity[bad_idxs[:50]]}")
        print(f"    bound_viol:   {bound_viols[bad_idxs[:50]]}")
        # Note: index max_solves-1 (={max_solves-1}) is the scatter dummy position
        print(f"    (index {max_solves-1} is the scatter dummy — ignore it)")

    # TESTING BATCH SOLVE WITHOUT RESTARTS -------------------------------------

    # print("\n--- Testing solve_batched ---")
    # N = 2000

    # max_iter = 500

    # def stack_states(states):
    #     def stack_leaves(*leaves):
    #         if eqx.is_array(leaves[0]):
    #             return jnp.stack(leaves)
    #         else:
    #             return leaves[0]
    #     return jax.tree.map(stack_leaves, *states)

    # batch_state = stack_states([state] * N)
    # _solve_batched = eqx.filter_jit(solve_batched)
    # batch_final, batch_terminate = _solve_batched(cp, batch_state, max_iter=max_iter)

    # with jax.profiler.trace("tmp/quad_batch_2000", create_perfetto_trace=True):
    #     t1 = time()
    #     batch_final_state, batch_terminate = _solve_batched(cp, batch_state, max_iter=max_iter)
    #     jax.block_until_ready(batch_final_state)
    #     t2 = time()
    #     print(f"TIME: {(t2-t1)}")

    batch_final_state = final_state

    # print(f"Batch terminate codes: {batch_terminate.squeeze()}")
    # print(f"Batch iter counts: {batch_final.iter_count.squeeze()}")

    # animate the result from IPOPT and the result from our implementation
    # from problems.quadcopter.plotting import Animator
    # x, u = aux[0](solution_buffer[0,:cp.nx])
    # r_ = [0.5, 0.5, 0.5, 0.75]
    # xc_ = [1., 1., 3., 3.]
    # yc_ = [1., 3., 1., 3.]
    # cylinder_definitions = [xc_,yc_,r_]
    # quad_params = aux[2]
    # # x = x.at[:, 2].mul(-1.)
    # x = x.at[:, :2].mul(-1.)
    # animator = Animator(quad_params, x, np.arange(0, 3, 0.1), x, cylinder_definitions=cylinder_definitions, drawCylinder=True, followQuad=False)
    # animator.animate()

    # Filter out non-converged solutions before plotting
    good_mask = ~bad_any
    good_indices = jnp.where(good_mask)[0]
    solution_buffer = solution_buffer[good_indices]
    print(f"\nPlotting {solution_buffer.shape[0]} good solutions (removed {int(bad_any.sum())})")

    from problems.redundant.quadcopter.plotting_batched import BatchedAnimator
    # Extract all trajectories from batch solve
    num_batch = solution_buffer.shape[0]
    xs = []
    for i in range(num_batch):
        x_i, u_i = aux[0](solution_buffer[i, :cp.nx])
        x_i = x_i.at[:, :2].mul(-1.)
        xs.append(x_i)

    r_ = [0.5, 0.5, 0.5, 0.75]
    xc_ = [1., 1., 3., 3.]
    yc_ = [1., 3., 1., 3.]
    cylinder_definitions = [xc_, yc_, r_]
    quad_params = aux[2]

    animator = BatchedAnimator(
        quad_params,
        xs,  # List of all trajectories
        np.arange(0, 3, 0.1),
        cylinder_definitions=cylinder_definitions,
        drawCylinder=True,
        followQuad=False,
        save_path='data/gifs/batch_quads_0IR_10000_sol.gif',
        title='Batch Solve',
        x_lim=(-5, 0),
        y_lim=(-5, 0),
        z_lim=(-3, 3)
    )
    animator.animate()

    pass

    # basic NMPC 2000 batch
    # 0.05548900365829468s per iteration 
    # (~30x solutions per second compared to IPOPT sparse at ~0.00083s per iter)

    # quadcopter NMPC 2000 batch
    # 1.1774216771125794s per iteration
    # (~10x solutions per second compared to IPOPT sparse at ~0.006s per iter) 
    # (different problems though - we are dominated by inertia correction refactorizations)

    # post expensive function call rework and resto initialization introduction again
    # 0.06772053241729736s per iter (some down to 0.032s, but we have some unecessary looping...)

    # after fixing the directions after not refactorizing 
    # 0.03811889886856079s per iter (43x)

    # lets try the QUAD at 2000 (only used 12GB of the 44, also was using GPU 260W/350W - not bad!)
    # 0.8686208367347718s per iteration

    # After fixing the algorithm with resto and stuff we have for QUAD with 2000 batch - with 5 IR steps (50 was what we needed to match IPOPT)
    # 0.5335468053817749s per iteration 
    # 0.5090104460716247s per iteration (second run)
    # 0.514044189453125s per iteration (third run)

    # IPOPT baseline is: 0.006s per iteration

    # Quad with 4000 batch
    # 1.153463399410248
    # 1.1548263072967528
    # 1.163555908203125
    
    # Quad with 1000 batch
    # 0.2557172179222107
    # 0.2475130081176758
    # 0.2360561966896057

    # Quad with 500 batch
    # 0.11297622919082642
    # 0.1081244945526123s
    # 0.10513460636138916s

    # going to 100 steps in 1000 batch
    # 
    
    # JOHN YOU ARE HERE
    """
    The quadcopter on casadi IPOPT on first call on cold start gets:
    Initial solve time: 1459.163 ms

    I am not sure how many iterations this took, but our batch solve with 2000
    instances took: TIME: 179378.8239955902s ms

    At face value this means we are getting (1459 / 179378) * 2000 = 16.27x more
    solutions per second than IPOPT on this problem. The nuance lies in the number
    of iterations per element in the batch optimization solve:

    iter_count.mean()
    Array(129.266, dtype=float64)
    iter_count.max()
    Array(500, dtype=int64)

    Therefore - if we had restarted problems on the fly when they completed (assuming no
    additional overhead) we would have gotten approximately 16.27 * (500 / 129.266) = 62.97x
    more solutions per second than IPOPT on this problem.

    The MadNLP solution to this problem (at tol == 1e-8) takes 1191.000831ms. That would 
    mean if we did hot restarting in the jaxipm (and it had no overhead, and the 
    assumption that solving the same problem over and over again is still representative 
    of batch performance), we would be 51.37x faster than MadNLP at generating solutions 
    for this quadcopter problem.

    Some more MadNLP output data.
    MadNLP setup: MadNLP version 0.8.12, running with cuDSS v0.7.1, L40s GPU
    MadNLP Cold run (includes JIT): 
    (1) ~ 104000 ms
    (2) 103984.10364599999 ms
    (3) 108317.035859 ms
    (4) 107238.362494 ms
    MadNLP JITTED warm runs:
    (1) 1191.000831 ms
    (2) 1456.887153 ms
    (3) 1154.01196 ms
    (4) 1192.291881 ms

    """

    pass
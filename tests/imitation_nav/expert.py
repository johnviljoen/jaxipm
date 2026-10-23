"""jaxipm receding-horizon expert for the nav scenario.

label(states) re-solves the full 30-step nav NLP (tests/quad_nav_circle) with
x0_ic = each state and returns u[0] of every converged solve. The states are
pushed through solve_throughput's sequential-pool hot-restart mode: every
pool index is injected exactly once, converged slots are refilled from the
pool, and the pool itself is an ARRAY FIELD of the reseed callable (an
equinox Module), so a new pool each call is traced, not recompiled. One
Expert = one (pool_size, batch_size) compile.

Returned rows are the solver's own x[0] (== injected x0_ic to tolerance) and
u[0], so for an unordered pool no index bookkeeping is needed. For ordered
labels (closed-loop expert rollouts) `label_ordered` matches rows back to
pool members by nearest neighbour on the 13 injected entries.
"""

import json
import os
from time import time

import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx

from jaxipm.initialization import initialize_common_problem, initialize_problem_regular
from jaxipm.solver import solve_throughput, make_batch_state
from tests.quad_nav_circle.jaxipm_quad_nav import quadcopter_nav, N_HORIZON
from tests.imitation_nav.sim import NX, NU, W_HOVER, WARM_END, arc_starts, valid_states

U0_SLICE = slice(N_HORIZON * NX, N_HORIZON * NX + NU)   # u[0] inside z = [x(30x13), u(29x4)]
_TS_LIN = jnp.linspace(0.0, 1.0, N_HORIZON)[:, None]
_END = jnp.asarray(WARM_END)


def build_warm_start(x0_ic):
    """Linear position interpolation to the goal, identity attitude, hover inputs."""
    xyz = (1 - _TS_LIN) * x0_ic[:3] + _TS_LIN * _END
    states = jnp.zeros((N_HORIZON, NX)).at[:, :3].set(xyz).at[:, 3].set(1.0)
    inputs = jnp.full((N_HORIZON - 1, NU), W_HOVER)
    return jnp.concatenate([states.flatten(), inputs.flatten()])


def nav_nlp(margin=0.0, k_start=0, r_u=0.0):
    """The paper's nav NLP (tests/quad_nav_circle) with three optional MPC knobs:
    `margin` [m] added to the inflated obstacle radius the PLANNER sees,
    `k_start`: first horizon stage the obstacle constraint applies to (the
    paper uses 0; stage 0 is the fixed measured state in closed loop), and
    `r_u`: weight of an input cost r_u * sum((u - w_hover)^2) (paper: 0, which
    makes the optimal controls bang-bang).
    margin=0, k_start=0, r_u=0 reproduces quadcopter_nav() exactly.
    """
    from tests.quad_nav_circle.jaxipm_quad_nav import OBSTACLES, QUAD_RADIUS, TS
    f, c, d, x_L, x_U, d_L, d_U, z0, gt, aux = quadcopter_nav()
    if margin == 0.0 and k_start == 0 and r_u == 0.0:
        return f, c, d, x_L, x_U, d_L, d_U, z0, gt, aux
    z_to_xu = aux[0]
    if r_u > 0.0:
        f0 = f
        def f(z):
            _, u = z_to_xu(z)
            return f0(z) + r_u * jnp.sum((u - W_HOVER) ** 2)
    obs = [(o["xc"], o["yc"], o["r"] + QUAD_RADIUS + margin) for o in OBSTACLES]

    def d2(z):
        x, u = z_to_xu(z)
        cl = []
        for k in range(k_start, N_HORIZON - 1):
            multiplier = 1 + k * TS * 0.1
            for (xc, yc, r) in obs:
                cl.append((x[k, 0] - xc) ** 2 + (x[k, 1] - yc) ** 2 - r ** 2 * multiplier)
        return jnp.hstack(cl)

    n_d = d2(z0).shape[0]
    return f, c, d2, x_L, x_U, jnp.zeros(n_d), jnp.full(n_d, jnp.inf), z0, gt, aux


class PoolReseed(eqx.Module):
    """calc_next_problem_seq with the pool AND its warm starts as traced leaves."""
    pool: jax.Array   # (pool_size, 13)  injected x0_ic
    warm: jax.Array   # (pool_size, nz)  initial iterate z per member

    def __call__(self, idx, sol):
        return (self.warm[idx], (), (self.pool[idx],), ())


def shift_plan(z):
    """Standard MPC warm start: drop stage 0, repeat the last stage. z (nz,) or (P, nz)."""
    z = jnp.asarray(z)
    x = z[..., :N_HORIZON * NX].reshape(*z.shape[:-1], N_HORIZON, NX)
    u = z[..., N_HORIZON * NX:].reshape(*z.shape[:-1], N_HORIZON - 1, NU)
    x = jnp.concatenate([x[..., 1:, :], x[..., -1:, :]], axis=-2)
    u = jnp.concatenate([u[..., 1:, :], u[..., -1:, :]], axis=-2)
    return jnp.concatenate([x.reshape(*z.shape[:-1], -1), u.reshape(*z.shape[:-1], -1)], axis=-1)


def rollout_warm_starts(X, U):
    """Warm starts for every visited state of a closed-loop rollout.

    X (B, T+1, 13), U (B, T, 4) -> Z (B*T, nz): for state (b, t) the next 30
    states / 29 inputs of the SAME rollout, padded by holding the last state
    and input. These satisfy the Euler dynamics constraints exactly (same
    model) wherever no padding is involved. Row order = (b major, t minor),
    matching X[:, :T].reshape(-1, 13).
    """
    X = np.asarray(X); U = np.asarray(U)
    B, T1, _ = X.shape; T = T1 - 1
    Xp = np.concatenate([X, np.repeat(X[:, -1:], N_HORIZON, axis=1)], axis=1)
    Up = np.concatenate([U, np.repeat(U[:, -1:], N_HORIZON, axis=1)], axis=1)
    rows = []
    for t in range(T):
        xs = Xp[:, t:t + N_HORIZON].reshape(B, -1)
        us = Up[:, t:t + N_HORIZON - 1].reshape(B, -1)
        rows.append(np.concatenate([xs, us], axis=1))
    Z = np.stack(rows, axis=1)            # (B, T, nz)
    return Z.reshape(B * T, -1)


class Expert:
    def __init__(self, pool_size, batch_size, debug=True, params=None, max_iter_per_solve=500,
                 init_first_batch=1, margin=0.0, k_start=0, r_u=0.0, cache_dir="/home/john/tmp/jax_cache"):
        assert batch_size <= pool_size
        if cache_dir:
            jax.config.update("jax_compilation_cache_dir", cache_dir)
            jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
            jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
        self.pool_size, self.batch_size = int(pool_size), int(batch_size)
        self.max_iter_per_solve = max_iter_per_solve
        if params is None:
            with open("jaxipm/params.json") as fh:
                params = json.load(fh)
        p = dict(params)
        p["DEBUG_MODE"] = bool(debug)
        p["hot_restarting"] = True
        p["sequential_pool"] = True
        p["pool_size"] = self.pool_size
        self.p = p
        self.debug = bool(debug)

        f, c, d, x_L, x_U, d_L, d_U, z0, _, aux = nav_nlp(margin=margin, k_start=k_start, r_u=r_u)
        self.f, self.z_to_xu = f, aux[0]
        dummy_pool = jnp.asarray(arc_starts(np.full(self.pool_size, np.pi / 4)))
        dummy_warm = jax.vmap(build_warm_start)(dummy_pool)
        reseed = PoolReseed(dummy_pool, dummy_warm)
        calc_next = lambda key, sol: reseed(jnp.int32(0), sol)   # unused in seq mode
        c_args = (dummy_pool[0],)
        self.cp = initialize_common_problem(f, c, d, x_L, x_U, d_L, d_U, z0, p,
                                            [(), c_args, ()],
                                            calc_next_problem=calc_next,
                                            calc_next_problem_seq=reseed)
        state = initialize_problem_regular(self.cp, z0, args=[(), c_args, ()])
        state = eqx.tree_at(lambda t: t.fl.needs_regular_init, state, jnp.array([[init_first_batch]]))
        # one token per Expert; problems are injected per call (hot-restart path)
        self.batch_tp = make_batch_state(self.cp, [state] * self.batch_size)
        self._solve = eqx.filter_jit(solve_throughput, donate="none")
        self._warm = jax.jit(jax.vmap(build_warm_start))
        self.key = jax.random.PRNGKey(0)
        self.compiled = False
        self.last = {}

    # ------------------------------------------------------------------ #
    def _inject(self, pool, warm):
        cp = eqx.tree_at(lambda c: (c.calc_next_problem_seq.pool, c.calc_next_problem_seq.warm),
                         self.cp, (pool, warm))
        first = pool[:self.batch_size]
        bt = eqx.tree_at(lambda s: s.args[1][1], self.batch_tp, first)
        new_x = bt.it.x.at[:, :cp.nx, 0].set(warm[:self.batch_size])
        bt = eqx.tree_at(lambda s: s.it.x, bt, new_x)
        return cp, bt

    def _run(self, pool, warm):
        cp, bt = self._inject(pool, warm)
        self.key, sub = jax.random.split(self.key)
        t0 = time()
        out = self._solve(sub, cp, bt, self.pool_size, max_iter_per_solve=self.max_iter_per_solve)
        jax.block_until_ready(out[1])
        return out, time() - t0

    def label(self, states, warm=None, return_z=False):
        """Unordered labels. states (P,13) numpy, P == pool_size.

        Returns dict(X0 (n,13), U0 (n,4), terms (n,), iters (n,), n, elapsed,
        n_invalid). Rows are converged solves only; invalid inputs (non-finite
        or outside the NLP state box) are replaced by a valid dummy and their
        rows dropped.
        """
        states = np.asarray(states, dtype=np.float64)
        assert states.shape == (self.pool_size, NX), states.shape
        ok_in = valid_states(states)
        safe = states.copy()
        safe[~ok_in] = arc_starts([np.pi / 4])[0]
        pool = jnp.asarray(safe)
        if warm is None:
            warm = self._warm(pool)
        else:
            warm = jnp.asarray(np.asarray(warm, dtype=np.float64))
            assert warm.shape == (self.pool_size, self.cp.nx), warm.shape
            warm = jnp.where(jnp.asarray(ok_in)[:, None], warm, self._warm(pool))
        if not self.compiled:
            t0 = time()
            self._run(pool, warm)                 # JIT warmup (untimed)
            self.compiled = True
            print(f"    [expert] compiled pool={self.pool_size} batch={self.batch_size} "
                  f"in {time()-t0:.1f}s", flush=True)
        out, elapsed = self._run(pool, warm)
        n = int(min(int(out[2]), self.pool_size))
        sols = np.asarray(out[1][:n])
        X0 = sols[:, :NX]
        U0 = sols[:, U0_SLICE]
        if self.debug:
            terms = np.asarray(out[3][:n]).astype(np.int32)
            iters = np.asarray(out[4][:n]).astype(np.int32)
        else:
            terms = np.full(n, -1, np.int32)
            iters = np.full(n, -1, np.int32)
        if (~ok_in).any():
            dummy = arc_starts([np.pi / 4])[0]
            keep = np.linalg.norm(X0 - dummy, axis=1) > 1e-6
            X0, U0, terms, iters = X0[keep], U0[keep], terms[keep], iters[keep]
            n = int(keep.sum())
        self.last = dict(n=n, elapsed=elapsed, n_invalid=int((~ok_in).sum()))
        out_d = dict(X0=X0, U0=U0, terms=terms, iters=iters, n=n, elapsed=elapsed,
                     n_invalid=int((~ok_in).sum()))
        if return_z:
            out_d["Z"] = sols if not (~ok_in).any() else sols[keep]
        if self.debug:
            out_d["term_hist"] = np.asarray(out[5]).astype(int)   # census of every terminal code (1..5)
        return out_d

    def label_ordered(self, states, warm=None):
        """Ordered labels for closed-loop use.

        Returns Z (P, nz) = the converged plan per pool member (NaN rows for
        failed / invalid members), ok (P,) bool, and the raw label() result.
        """
        states = np.asarray(states, dtype=np.float64)
        res = self.label(states, warm=warm, return_z=True)
        P = states.shape[0]
        Z = np.full((P, self.cp.nx), np.nan)
        ok = np.zeros(P, dtype=bool)
        if res["n"] > 0:
            # nearest pool member for each returned row (chunked to bound memory)
            d2 = _chunked_nn(res["X0"], states)
            for row, j in enumerate(d2):
                if not ok[j]:
                    Z[j] = res["Z"][row]
                    ok[j] = True
        return Z, ok, res


def _chunked_nn(Q, P, chunk=512):
    """argmin_j ||Q_i - P_j|| for each query row i, computed on the GPU in chunks."""
    Pj = jnp.asarray(P, dtype=jnp.float32)
    out = []
    for s in range(0, Q.shape[0], chunk):
        q = jnp.asarray(Q[s:s + chunk], dtype=jnp.float32)
        d = jnp.sum((q[:, None, :] - Pj[None, :, :]) ** 2, axis=-1)
        out.append(np.asarray(jnp.argmin(d, axis=1)))
    return np.concatenate(out) if out else np.zeros(0, int)


def expert_rollout(expert, x0s, T, verbose=True):
    """Closed-loop expert: T sequential batched re-solves from x0s (P,13).

    Returns X (P,T+1,13), U (P,T,4), ok (P,T) and per-step solve times.
    """
    from tests.imitation_nav.sim import step
    _step = jax.jit(jax.vmap(step))
    _shift = jax.jit(shift_plan)
    P = x0s.shape[0]
    X = np.zeros((P, T + 1, NX)); U = np.zeros((P, T, NU)); OK = np.zeros((P, T), bool)
    X[:, 0] = x0s
    times = []
    plan = np.asarray(jax.vmap(build_warm_start)(jnp.asarray(x0s)))   # (P, nz)
    for t in range(T):
        Z, ok, res = expert.label_ordered(X[:, t], warm=plan)
        # failed members fall back on their previous plan (already shifted)
        plan = np.where(ok[:, None], Z, plan)
        u = plan[:, U0_SLICE]
        U[:, t], OK[:, t] = u, ok
        times.append(res["elapsed"])
        X[:, t + 1] = np.asarray(_step(jnp.asarray(X[:, t]), jnp.asarray(u)))
        plan = np.asarray(_shift(jnp.asarray(plan)))
        if verbose and (t % 10 == 0 or t == T - 1):
            print(f"    [expert rollout] step {t+1:3d}/{T}  labelled {ok.sum()}/{P}  "
                  f"solve {res['elapsed']:.2f}s", flush=True)
    return X, U, OK, np.asarray(times)


def prev_actions(U):
    """U (B, T, 4) -> Uprev (B, T, 4): action applied one step earlier, hover at t=0."""
    U = np.asarray(U)
    return np.concatenate([np.full_like(U[:, :1], W_HOVER), U[:, :-1]], axis=1)


def match_rows(X0, pool):
    """Pool index of each returned row (rows equal the injected state to solver tol)."""
    return _chunked_nn(np.asarray(X0), np.asarray(pool), chunk=256)

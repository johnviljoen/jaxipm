"""jaxipm receding-horizon expert for track-avoid.

Per pool member the NLP takes three runtime arguments: the reference horizon
xr (f_arg), the measured state x0 (c_arg) and the obstacle centres over the
horizon (d_arg, absolute time t .. t + 29 Ts). All of them plus the warm
start live as array fields of the sequential-pool reseed Module, so a new
pool is traced, not recompiled. Mechanics (sequential pool, hot restart,
rollout / shifted-plan warm starts, row matching) are as in
tests/imitation_nav/expert.py, whose generic helpers are reused.
"""

import json
from time import time

import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx

from jaxipm.initialization import initialize_common_problem, initialize_problem_regular
from jaxipm.solver import solve_throughput, make_batch_state
from tests.quad_track_avoid.jaxipm_track_avoid import quadcopter_track_avoid, N_HORIZON
from tests.imitation_nav.expert import shift_plan, rollout_warm_starts, prev_actions, match_rows, _chunked_nn
from tests.imitation_track.sim import (NX, NU, W_HOVER, N_OBS, OBS_R, TS, ref_horizon, obstacle_horizon,
                                       valid_states, step)

U0_SLICE = slice(N_HORIZON * NX, N_HORIZON * NX + NU)


def warm_from_ref(xr):
    """Warm start = the reference states themselves + hover inputs. xr (N,13) -> (nz,)."""
    return jnp.concatenate([xr.reshape(-1), jnp.full((N_HORIZON - 1) * NU, W_HOVER)])


def track_nlp(v, margin=0.0, k_start=0, r_u=0.0):
    """The paper's track-avoid NLP with obstacle centres as a runtime argument
    (closed loop needs absolute time) and the same optional knobs as nav:
    planner obstacle margin [m], first constrained stage, input cost weight."""
    f0, c, _, x_L, x_U, _, _, z_init, gt, aux = quadcopter_track_avoid(N=N_HORIZON, avg_vel=v)
    z_to_xu = aux[0]
    r2 = jnp.tile((jnp.asarray(OBS_R) + margin) ** 2, N_HORIZON)      # (N*N_OBS,), stage-major

    def d(z, obs_xy):                                 # obs_xy (N*N_OBS, 2) stage-major
        x, u = z_to_xu(z)
        xs = jnp.repeat(x[:, 0], N_OBS); ys = jnp.repeat(x[:, 1], N_OBS)
        vals = (xs - obs_xy[:, 0]) ** 2 + (ys - obs_xy[:, 1]) ** 2 - r2
        vals = vals[k_start * N_OBS:]
        qn = jnp.sum(x[:, 3:7] ** 2, axis=1)
        return jnp.concatenate([vals, qn])

    if r_u > 0.0:
        def f(z, xr):
            _, u = z_to_xu(z)
            return f0(z, xr) + r_u * jnp.sum((u - W_HOVER) ** 2)
    else:
        f = f0
    n_vals = (N_HORIZON - k_start) * N_OBS
    d_L = jnp.concatenate([jnp.zeros(n_vals), jnp.full(N_HORIZON, 0.5)])
    d_U = jnp.concatenate([jnp.full(n_vals, jnp.inf), jnp.full(N_HORIZON, 1.5)])
    return f, c, d, x_L, x_U, d_L, d_U, z_init, gt, aux


class PoolReseed(eqx.Module):
    x0: jax.Array     # (P, 13)
    xr: jax.Array     # (P, N, 13)
    obs: jax.Array    # (P, N*N_OBS, 2)
    warm: jax.Array   # (P, nz)

    def __call__(self, idx, sol):
        return (self.warm[idx], (self.xr[idx],), (self.x0[idx],), (self.obs[idx],))


class Expert:
    """jaxipm expert. mode="il": iteration-level batching (the paper method:
    one solve_throughput call, converged slots hot-restart on the next pool
    member). mode="sl": solve-level batching (jaxipm-SL in the paper tables):
    the pool is cut into batches of batch_size, each batch runs until its
    slowest member terminates (hot_restarting=False), then the next batch."""

    def __init__(self, pool_size, batch_size, v, debug=True, params=None, max_iter_per_solve=500,
                 init_first_batch=1, margin=0.0, k_start=0, r_u=0.0, cache_dir="/home/john/tmp/jax_cache",
                 mode="il"):
        assert batch_size <= pool_size
        assert mode in ("il", "sl"), mode
        self.mode = mode
        if cache_dir:
            jax.config.update("jax_compilation_cache_dir", cache_dir)
            jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
            jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
        self.pool_size, self.batch_size, self.v = int(pool_size), int(batch_size), float(v)
        self.max_iter_per_solve = max_iter_per_solve
        if params is None:
            with open("jaxipm/params.json") as fh:
                params = json.load(fh)
        p = dict(params)
        p["DEBUG_MODE"] = bool(debug)
        p["hot_restarting"] = (mode == "il")
        p["sequential_pool"] = (mode == "il"); p["pool_size"] = self.pool_size
        # sl: the pool is padded to a whole number of batches; pad rows are
        # flagged invalid in label() and dropped like invalid states
        self.n_pad = (-self.pool_size) % self.batch_size if mode == "sl" else 0
        self.p, self.debug = p, bool(debug)

        f, c, d, x_L, x_U, d_L, d_U, z0, _, aux = track_nlp(v, margin=margin, k_start=k_start, r_u=r_u)
        self.f, self.z_to_xu = f, aux[0]
        self._xr = jax.jit(jax.vmap(ref_horizon, in_axes=(0, 0, None)), static_argnums=2)
        self._obs = jax.jit(jax.vmap(lambda t: obstacle_horizon(t).reshape(-1, 2)))
        self._warm = jax.jit(jax.vmap(warm_from_ref))
        P = self.pool_size
        s_dummy = jnp.zeros(P); t_dummy = jnp.zeros(P)
        xr_d = self._xr(s_dummy, t_dummy, self.v); obs_d = self._obs(t_dummy)
        x0_d = xr_d[:, 0]; warm_d = self._warm(xr_d)
        reseed = PoolReseed(x0_d, xr_d, obs_d, warm_d)
        calc_next = lambda key, sol: reseed(jnp.int32(0), sol)        # unused in seq mode
        args = [(xr_d[0],), (x0_d[0],), (obs_d[0],)]
        self.cp = initialize_common_problem(f, c, d, x_L, x_U, d_L, d_U, z0, p, args,
                                            calc_next_problem=calc_next, calc_next_problem_seq=reseed)
        state = initialize_problem_regular(self.cp, z0, args=args)
        state = eqx.tree_at(lambda t: t.fl.needs_regular_init, state, jnp.array([[init_first_batch]]))
        self.batch_tp = make_batch_state(self.cp, [state] * self.batch_size)
        self._solve = eqx.filter_jit(solve_throughput, donate="none")
        self.key = jax.random.PRNGKey(0)
        self.compiled = False

    # ------------------------------------------------------------------ #
    def _inject(self, x0, xr, obs, warm):
        """First batch of problems + warm starts into the pristine batch state."""
        B = self.batch_size
        bt = eqx.tree_at(lambda s: (s.args[0][4], s.args[1][1], s.args[2][1]), self.batch_tp,
                         (xr[:B], x0[:B], obs[:B]))
        return eqx.tree_at(lambda s: s.it.x, bt, bt.it.x.at[:, :self.cp.nx, 0].set(warm[:B]))

    def _run(self, x0, xr, obs, warm):
        if self.mode == "il":
            cp = eqx.tree_at(lambda c: (c.calc_next_problem_seq.x0, c.calc_next_problem_seq.xr,
                                        c.calc_next_problem_seq.obs, c.calc_next_problem_seq.warm),
                             self.cp, (x0, xr, obs, warm))
            bt = self._inject(x0, xr, obs, warm)
            self.key, sub = jax.random.split(self.key)
            t0 = time()
            out = self._solve(sub, cp, bt, self.pool_size, max_iter_per_solve=self.max_iter_per_solve)
            jax.block_until_ready(out[1])
            return out, time() - t0
        # sl: one solve_throughput call per batch (max_solves = batch_size, no
        # restarts); outputs concatenated in collection order like the il buffer
        B = self.batch_size
        P = x0.shape[0]
        assert P % B == 0, (P, B)
        sols, terms, iters, hist, n_tot = [], [], [], np.zeros(6, np.int64), 0
        t0 = time()
        for s in range(0, P, B):
            sl = slice(s, s + B)
            bt = self._inject(x0[sl], xr[sl], obs[sl], warm[sl])
            self.key, sub = jax.random.split(self.key)
            out = self._solve(sub, self.cp, bt, B, max_iter_per_solve=self.max_iter_per_solve)
            n = int(out[2])
            sols.append(np.asarray(out[1][:n])); n_tot += n
            if self.debug:
                terms.append(np.asarray(out[3][:n])); iters.append(np.asarray(out[4][:n]))
                hist += np.asarray(out[5]).astype(np.int64)
        elapsed = time() - t0
        cat = lambda l, d: np.concatenate(l) if l else np.zeros((0,) + d)
        return (None, cat(sols, (self.cp.nx,)), n_tot, cat(terms, ()), cat(iters, ()), hist), elapsed

    def label(self, x0s, s0s, ts, warm=None, return_z=False):
        """Unordered labels for pool members (x0s (P,13), s0s (P,), ts (P,) sim time).

        Returns dict(X0, U0, terms, iters, n, elapsed, n_invalid[, Z, term_hist]).
        Invalid states (non-finite / outside the state box) are replaced by their
        own reference state and the resulting rows dropped.
        """
        x0s = np.asarray(x0s, dtype=np.float64); s0s = np.asarray(s0s, dtype=np.float64)
        ts = np.asarray(ts, dtype=np.float64)
        assert x0s.shape == (self.pool_size, NX), x0s.shape
        ok_in = valid_states(x0s)
        if self.n_pad:   # sl: pad to whole batches with invalid (dropped) copies of the last row
            pad = lambda a: np.concatenate([a, np.repeat(a[-1:], self.n_pad, axis=0)])
            x0s, s0s, ts = pad(x0s), pad(s0s), pad(ts)
            ok_in = np.concatenate([ok_in, np.zeros(self.n_pad, bool)])
            if warm is not None:
                warm = pad(np.asarray(warm, dtype=np.float64))
        xr = self._xr(jnp.asarray(s0s), jnp.asarray(ts), self.v)
        obs = self._obs(jnp.asarray(ts))
        safe = np.where(ok_in[:, None], x0s, np.asarray(xr[:, 0]))
        x0 = jnp.asarray(safe)
        warm_ref = self._warm(xr)
        if warm is None:
            warm = warm_ref
        else:
            warm = jnp.asarray(np.asarray(warm, dtype=np.float64))
            assert warm.shape == (x0.shape[0], self.cp.nx), warm.shape
            warm = jnp.where(jnp.asarray(ok_in)[:, None], warm, warm_ref)
        if not self.compiled:
            t0 = time()
            B = self.batch_size if self.mode == "sl" else x0.shape[0]   # sl: one batch suffices
            self._run(x0[:B], xr[:B], obs[:B], warm[:B]); self.compiled = True
            print(f"    [expert {self.mode}] compiled pool={self.pool_size} batch={self.batch_size} in {time()-t0:.1f}s", flush=True)
        out, elapsed = self._run(x0, xr, obs, warm)
        n = int(min(int(out[2]), x0.shape[0]))
        sols = np.asarray(out[1][:n])
        X0, U0 = sols[:, :NX], sols[:, U0_SLICE]
        terms = np.asarray(out[3][:n]).astype(np.int32) if self.debug else np.full(n, -1, np.int32)
        iters = np.asarray(out[4][:n]).astype(np.int32) if self.debug else np.full(n, -1, np.int32)
        keep = np.ones(n, bool)
        if (~ok_in).any():
            # rows that solved a substituted reference state are not labels of real states
            bad = np.asarray(xr[:, 0])[~ok_in]
            dmin = _chunked_min_dist(X0, bad) if n else np.zeros(0)
            keep = dmin > 1e-6
            X0, U0, terms, iters, sols = X0[keep], U0[keep], terms[keep], iters[keep], sols[keep]
            n = int(keep.sum())
        res = dict(X0=X0, U0=U0, terms=terms, iters=iters, n=n, elapsed=elapsed,
                   n_invalid=int((~ok_in).sum()) - self.n_pad)
        if return_z:
            res["Z"] = sols
        if self.debug:
            res["term_hist"] = np.asarray(out[5]).astype(int)
        return res

    def label_ordered(self, x0s, s0s, ts, warm=None):
        """Ordered plans per pool member: Z (P, nz) (NaN rows for failures), ok (P,), res."""
        x0s = np.asarray(x0s, dtype=np.float64)
        res = self.label(x0s, s0s, ts, warm=warm, return_z=True)
        P = x0s.shape[0]
        Z = np.full((P, self.cp.nx), np.nan); ok = np.zeros(P, bool)
        if res["n"] > 0:
            for row, j in enumerate(match_rows(res["X0"], x0s)):
                if not ok[j]:
                    Z[j] = res["Z"][row]; ok[j] = True
        return Z, ok, res


def _chunked_min_dist(Q, P, chunk=512):
    """min_j ||Q_i - P_j|| per query row (GPU, chunked over queries)."""
    Pj = jnp.asarray(P)
    out = []
    for s in range(0, Q.shape[0], chunk):
        q = jnp.asarray(Q[s:s + chunk])
        out.append(np.asarray(jnp.min(jnp.linalg.norm(q[:, None, :] - Pj[None, :, :], axis=-1), axis=1)))
    return np.concatenate(out) if out else np.zeros(0)


def expert_rollout(expert, x0s, s0s, T, verbose=True):
    """Closed-loop expert with shifted-plan warm starts. Returns X (P,T+1,13), U (P,T,4), ok (P,T), times."""
    _step = jax.jit(jax.vmap(step)); _shift = jax.jit(shift_plan)
    P = x0s.shape[0]
    X = np.zeros((P, T + 1, NX)); U = np.zeros((P, T, NU)); OK = np.zeros((P, T), bool)
    X[:, 0] = x0s
    times = []
    plan = np.asarray(expert._warm(expert._xr(jnp.asarray(s0s), jnp.zeros(P), expert.v)))
    for t in range(T):
        Z, ok, res = expert.label_ordered(X[:, t], s0s, np.full(P, t * TS), warm=plan)
        plan = np.where(ok[:, None], Z, plan)
        u = plan[:, U0_SLICE]
        U[:, t], OK[:, t] = u, ok
        times.append(res["elapsed"])
        X[:, t + 1] = np.asarray(_step(jnp.asarray(X[:, t]), jnp.asarray(u)))
        plan = np.asarray(_shift(jnp.asarray(plan)))
        if verbose and (t % 10 == 0 or t == T - 1):
            print(f"    [expert rollout] step {t+1:3d}/{T}  labelled {ok.sum()}/{P}  solve {res['elapsed']:.2f}s", flush=True)
    return X, U, OK, np.asarray(times)

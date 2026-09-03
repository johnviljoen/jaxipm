"""Scenario builders shared by the rebuttal harnesses (tests/rebuttal/).

Each builder reproduces the problem setup of the corresponding paper driver /
batch-sweep script exactly (same pool of problem instances, same order, same
per-slot injection and warm start) and returns a `Scenario` that the harnesses
drive generically:

    sc = build_scenario("nav", 90, p, N_batch)
    batch_tp = sc.inject(make_batch_state(sc.cp, [sc.state] * N_batch), idx)
    ...
    obj = sc.objective(z, pool_idx);  X = sc.to_X(z);  ok = sc.success(X, iters)

Pool semantics (identical to the paper drivers):
  nav   : N_RUNS start angles on a circle (deterministic linspace over the
          sector); hot-restart strides the angle index by N_batch.
  track : N_RUNS random (x0, xr) pairs from generate_random_inits(seed=0);
          hot-restart strides N_batch through the pool like nav (was: random draw).
  multi : N_RUNS varied-IC instances (2026-08-31 v3): start formation rotated
          by phi_k in [-5deg, +5deg) around the static starts, goal
          formation INDEPENDENTLY at psi_k = phi_k + delta_k, delta_k in
          [-5deg, +5deg) around antipodal (low-discrepancy strides);
          hot-restart strides the pool index by N_batch like nav.

Nothing here touches jax config -- the caller enables x64 before building.
"""

import json
import os
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np


@dataclass
class Scenario:
    name: str                 # "nav" | "track" | "multi"
    variant: Any              # 90 | 1.5 | 2 ...
    tag: str                  # filename fragment matching the paper npz names
    label: str                # human label e.g. "nav_s90"
    logs_dir: str
    N_RUNS: int               # pool size (== paper N_RUNS_jaxipm)
    N_batch_paper: int        # batch size used for the paper numbers
    cp: Any
    state: Any
    f: Callable
    inject: Callable          # (batch_tp, pool_idx: np.ndarray) -> batch_tp
    objective: Callable       # (z_np, pool_idx) -> float
    match_pool: Callable      # (z_np) -> pool idx of the instance this z solved
    to_X: Callable            # (z_np) -> state trajectory array
    success: Callable         # (X, iters, max_iter) -> bool
    meta: dict = field(default_factory=dict)
    extra_npz: dict = field(default_factory=dict)
    # post-hoc quality evaluation on a saved z (user-level functions, unscaled)
    c_fn: Callable = None          # (z_np, pool_idx) -> equality residuals c(z)
    d_fn: Callable = None          # (z_np, pool_idx) -> inequality values d(z)
    bounds: tuple = None           # (x_L, x_U, d_L, d_U) as numpy arrays
    c_batch: Callable = None       # (Z (n,nz), idx (n,)) -> c residuals (n, nyc), jitted+vmapped
    d_batch: Callable = None       # (Z) -> d values (n, nyd)
    match_batch: Callable = None   # (Z) -> pool idx (n,)



def load_params():
    with open("jaxipm/params.json") as fh:
        return json.load(fh)


TEST_PARAMS = {"nav": "tests/quad_nav_circle/test_params.json",
               "track": "tests/quad_track_avoid/test_params.json",
               "multi": "tests/quad_multi_swap/test_params.json"}


def paper_batch_size(name):
    with open(TEST_PARAMS[name]) as fh:
        return int(json.load(fh)["batch_size"])


# ─────────────────────────────────────────────────────────────────────────────
def build_nav(sector_deg, p, N_batch):
    import jax
    import jax.numpy as jnp
    import equinox as eqx
    from tests.quad_nav_circle.jaxipm_quad_nav import (
        N_HORIZON, TS, quadcopter_nav, tmp)
    from jaxipm.initialization import (
        initialize_common_problem, initialize_problem_regular)

    N_RUNS = tmp["N_RUNS_jaxipm"]
    f, c, d, x_L, x_U, d_L, d_U, x0, gt, aux = quadcopter_nav()
    z_to_xu = aux[0]
    RADIUS = float(np.sqrt(4.0**2 + 4.0**2))
    sector_half = np.deg2rad(sector_deg) / 2.0
    center = np.pi / 4.0
    angles_np = np.linspace(center - sector_half, center + sector_half, N_RUNS)
    all_x0_np = np.zeros((N_RUNS, 13))
    all_x0_np[:, 0] = RADIUS * np.cos(angles_np)
    all_x0_np[:, 1] = RADIUS * np.sin(angles_np)
    all_x0_np[:, 3] = 1.0
    all_x0 = jnp.asarray(all_x0_np)
    lo, hi = center - sector_half, center + sector_half

    nx_h, nu_h = 13, 4
    w_hover = 522.9847140714692
    end_xyz = jnp.array([0.0, 0.0, -2.0])
    ts_lin = jnp.linspace(0.0, 1.0, N_HORIZON)[:, None]

    def build_warm_start(x0_ic):
        xyz = (1 - ts_lin) * x0_ic[:3] + ts_lin * end_xyz
        states = (jnp.zeros((N_HORIZON, nx_h)).at[:, :3].set(xyz)
                  .at[:, 3].set(1.0))
        inputs = jnp.full((N_HORIZON - 1, nu_h), w_hover)
        return jnp.concatenate([states.flatten(), inputs.flatten()])

    def calc_next_problem(key, sol):
        theta = jnp.clip(jnp.arctan2(sol[1], sol[0]), lo, hi)
        idx = jnp.round((theta - lo) / (hi - lo) * (N_RUNS - 1)).astype(jnp.int32)
        new_idx = (idx + N_batch) % N_RUNS
        new_x0_ic = all_x0[new_idx]
        return (build_warm_start(new_x0_ic), (), (new_x0_ic,), ())

    def calc_next_problem_seq(idx, sol):     # sequential pool: explicit index
        new_x0_ic = all_x0[idx]
        return (build_warm_start(new_x0_ic), (), (new_x0_ic,), ())

    f_args, c_args, d_args = (), (all_x0[0],), ()
    cp = initialize_common_problem(f, c, d, x_L, x_U, d_L, d_U, x0, p,
                                   [f_args, c_args, d_args],
                                   calc_next_problem=calc_next_problem,
                                   calc_next_problem_seq=calc_next_problem_seq)
    state = initialize_problem_regular(cp, x0, args=[f_args, c_args, d_args])
    state = eqx.tree_at(lambda t: t.fl.needs_regular_init, state,
                        jnp.array([[0]]))

    def inject(batch_tp, idx):
        idx = np.asarray(idx)
        bx0 = all_x0[idx]
        batch_tp = eqx.tree_at(lambda s: s.args[1][1], batch_tp, bx0)
        warms = jax.vmap(build_warm_start)(bx0)
        new_x = batch_tp.it.x.at[:, :cp.nx, 0].set(warms)
        return eqx.tree_at(lambda s: s.it.x, batch_tp, new_x)

    def objective(z, idx):
        return float(f(jnp.asarray(z)))

    def match_pool(z):
        x, _ = z_to_xu(jnp.asarray(z))
        x0_ = np.asarray(x[0])
        theta = np.arctan2(x0_[1], x0_[0])
        return int(np.abs(angles_np - theta).argmin())

    def to_X(z):
        x, _ = z_to_xu(jnp.asarray(z))
        return np.asarray(x)

    def success(X, iters, max_iter):
        return bool(np.isfinite(X).all() and (iters < 0 or iters < max_iter))

    def c_fn(z, idx):
        return np.asarray(c(jnp.asarray(z), all_x0[int(idx)]))

    def d_fn(z, idx):
        return np.asarray(d(jnp.asarray(z)))

    c_batch = jax.jit(jax.vmap(lambda z, i: c(z, all_x0[i])))
    d_batch = jax.jit(jax.vmap(lambda z: d(z)))
    def match_batch(Z):
        th = np.arctan2(np.asarray(Z)[:, 1], np.asarray(Z)[:, 0])
        return np.abs(angles_np[None, :] - th[:, None]).argmin(axis=1)

    return Scenario(
        c_fn=c_fn, d_fn=d_fn, c_batch=c_batch, d_batch=d_batch, match_batch=match_batch,
        bounds=tuple(np.asarray(b) for b in (x_L, x_U, d_L, d_U)),
        name="nav", variant=sector_deg, tag=f"sector{int(sector_deg)}",
        label=f"nav_s{int(sector_deg)}",
        logs_dir=os.path.join("tests", "quad_nav_circle", "logs"),
        N_RUNS=N_RUNS, N_batch_paper=tmp["batch_size"], cp=cp, state=state,
        f=f, inject=inject, objective=objective, match_pool=match_pool,
        to_X=to_X, success=success,
        meta=dict(N=N_HORIZON, Ts=TS, radius=RADIUS, sector_deg=sector_deg),
        extra_npz=dict(all_angles=angles_np, all_x0=all_x0_np,
                       N=np.array([N_HORIZON]), Ts=np.array([TS]),
                       radius=np.array([RADIUS]),
                       sector_deg=np.array([sector_deg])),
    )


# ─────────────────────────────────────────────────────────────────────────────

def nav_pool_instance(sector_deg, k):
    """(x0_ic, z_warm) of nav-circle pool member k -- identical formulas to build_nav
    (used by the run-J correctness extension to pick pool instances)."""
    import jax.numpy as jnp
    from tests.quad_nav_circle.jaxipm_quad_nav import N_HORIZON, tmp
    N_RUNS = tmp["N_RUNS_jaxipm"]
    RADIUS = float(np.sqrt(4.0**2 + 4.0**2))
    sector_half = np.deg2rad(sector_deg) / 2.0
    center = np.pi / 4.0
    angles_np = np.linspace(center - sector_half, center + sector_half, N_RUNS)
    x0 = np.zeros(13); x0[0] = RADIUS * np.cos(angles_np[k]); x0[1] = RADIUS * np.sin(angles_np[k]); x0[3] = 1.0
    w_hover = 522.9847140714692
    end_xyz = np.array([0.0, 0.0, -2.0])
    ts_lin = np.linspace(0.0, 1.0, N_HORIZON)[:, None]
    xyz = (1 - ts_lin) * x0[:3] + ts_lin * end_xyz
    states = np.zeros((N_HORIZON, 13)); states[:, :3] = xyz; states[:, 3] = 1.0
    inputs = np.full((N_HORIZON - 1, 4), w_hover)
    return x0, np.concatenate([states.flatten(), inputs.flatten()])

def build_track(avg_vel, p, N_batch):
    import jax
    import jax.numpy as jnp
    import equinox as eqx
    from tests.quad_track_avoid.jaxipm_track_avoid import (
        N_HORIZON, TS, quadcopter_track_avoid, tmp)
    from tests.quad_track_avoid.initialization import generate_random_inits
    from jaxipm.initialization import (
        initialize_common_problem, initialize_problem_regular)

    DELTA_STD = 0.5   # matches jaxipm_track_avoid.py
    N_RUNS = tmp["N_RUNS_jaxipm"]
    f, c, d, x_L, x_U, d_L, d_U, z_init, gt, aux = quadcopter_track_avoid(
        N=N_HORIZON, avg_vel=avg_vel)
    z_to_xu, _, _, xr_default, x0_default = aux
    all_x0_np, all_xr_np, all_s_np, _ = generate_random_inits(
        N_RUNS, N_HORIZON, TS, avg_vel, delta_std=DELTA_STD, seed=0)
    all_x0_np = np.asarray(all_x0_np)
    all_xr_np = np.asarray(all_xr_np)
    all_s_np = np.asarray(all_s_np)
    all_x0 = jnp.asarray(all_x0_np)
    all_xr = jnp.asarray(all_xr_np)
    w_hover = 522.9847140714692

    def build_warm_start(x0_ic, xr_ref):
        del xr_ref
        end_xyz = jnp.array([0., 0., -2.])
        xyz = jnp.linspace(x0_ic[:3], end_xyz, N_HORIZON)
        ang = jnp.tile(jnp.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0]), (N_HORIZON, 1))
        states = jnp.hstack([xyz, ang])
        inputs = jnp.full((N_HORIZON - 1, 4), w_hover)
        return jnp.concatenate([states.flatten(), inputs.flatten()])

    def calc_next_problem(rng_key, sol):
        # nav-style stride (matches jaxipm_track_avoid.py): decode the solved
        # pool member from the converged x0, step N_batch, wrap at the end.
        cur_idx = jnp.argmin(jnp.sum((all_x0 - sol[None, :13]) ** 2, axis=1))
        new_idx = (cur_idx + N_batch) % N_RUNS
        new_x0, new_xr = all_x0[new_idx], all_xr[new_idx]
        return (build_warm_start(new_x0, new_xr), (new_xr,), (new_x0,), ())

    def calc_next_problem_seq(idx, sol):     # sequential pool: explicit index
        new_x0, new_xr = all_x0[idx], all_xr[idx]
        return (build_warm_start(new_x0, new_xr), (new_xr,), (new_x0,), ())

    f_args, c_args, d_args = (xr_default,), (x0_default,), ()
    cp = initialize_common_problem(f, c, d, x_L, x_U, d_L, d_U, z_init, p,
                                   [f_args, c_args, d_args],
                                   calc_next_problem=calc_next_problem,
                                   calc_next_problem_seq=calc_next_problem_seq)
    state = initialize_problem_regular(cp, z_init,
                                       args=[f_args, c_args, d_args])
    state = eqx.tree_at(lambda t: t.fl.needs_regular_init, state,
                        jnp.array([[0]]))

    def inject(batch_tp, idx):
        idx = np.asarray(idx)
        bx0, bxr = all_x0[idx], all_xr[idx]
        batch_tp = eqx.tree_at(lambda s: s.args[1][1], batch_tp, bx0)
        batch_tp = eqx.tree_at(lambda s: s.args[0][4], batch_tp, bxr)
        warms = jax.vmap(build_warm_start)(bx0, bxr)
        new_x = batch_tp.it.x.at[:, :cp.nx, 0].set(warms)
        return eqx.tree_at(lambda s: s.it.x, batch_tp, new_x)

    def objective(z, idx):
        return float(f(jnp.asarray(z), jnp.asarray(all_xr_np[int(idx)])))

    def match_pool(z):
        x, _ = z_to_xu(jnp.asarray(z))
        x0_ = np.asarray(x[0])
        return int(np.linalg.norm(all_x0_np - x0_, axis=1).argmin())

    def to_X(z):
        x, _ = z_to_xu(jnp.asarray(z))
        return np.asarray(x)

    def success(X, iters, max_iter):
        return bool(np.isfinite(X).all() and (iters < 0 or iters < max_iter))

    def c_fn(z, idx):
        return np.asarray(c(jnp.asarray(z), all_x0[int(idx)]))

    def d_fn(z, idx):
        return np.asarray(d(jnp.asarray(z)))

    c_batch = jax.jit(jax.vmap(lambda z, i: c(z, all_x0[i])))
    d_batch = jax.jit(jax.vmap(lambda z: d(z)))
    def match_batch(Z):
        Z = np.asarray(Z)[:, :13]
        return np.linalg.norm(all_x0_np[None, :, :] - Z[:, None, :], axis=2).argmin(axis=1)

    return Scenario(
        c_fn=c_fn, d_fn=d_fn, c_batch=c_batch, d_batch=d_batch, match_batch=match_batch,
        bounds=tuple(np.asarray(b) for b in (x_L, x_U, d_L, d_U)),
        name="track", variant=avg_vel, tag=f"v{avg_vel:.1f}",
        label=f"track_v{avg_vel:.1f}",
        logs_dir=os.path.join("tests", "quad_track_avoid", "logs"),
        N_RUNS=N_RUNS, N_batch_paper=tmp["batch_size"], cp=cp, state=state,
        f=f, inject=inject, objective=objective, match_pool=match_pool,
        to_X=to_X, success=success,
        meta=dict(N=N_HORIZON, Ts=TS, avg_vel=avg_vel, delta_std=DELTA_STD,
                  init_seed=0),
        extra_npz=dict(all_x0=all_x0_np, all_xr=all_xr_np, all_s=all_s_np,
                       N=np.array([N_HORIZON]), Ts=np.array([TS]),
                       avg_vel=np.array([avg_vel])),
    )


# ─────────────────────────────────────────────────────────────────────────────
def build_multi(n_quads, p, N_batch):
    import jax
    import jax.numpy as jnp
    import equinox as eqx
    from tests.quad_multi_swap.jaxipm_multi_swap import (
        N_HORIZON, R, TS, multi_swap_pool, quadcopter_multi_swap_pool, tmp)
    from tests.quad_multi_swap.ipopt_multi_swap import check_success
    from jaxipm.initialization import (
        initialize_common_problem, initialize_problem_regular)

    # Varied-IC pool (2026-08-31): N_RUNS formation rotations phi_k = 2*pi*k/N_RUNS
    # around the circle (starts at 2*pi*i/Nq + phi_k, goals antipodal), matching
    # the nav test's deterministic angle pool. Instance geometry enters f/c as
    # runtime args; hot-restart strides the pool index by N_batch like nav.
    N_RUNS = tmp["N_RUNS_jaxipm"]
    f, c, d, x_L, x_U, d_L, d_U, x0, gt, aux = quadcopter_multi_swap_pool(
        N_quads=n_quads, N=N_HORIZON, R=R)
    z_to_xu, xu_to_z = aux[0], aux[1]
    phis_np, psis_np, starts_np, goals_np, goals_full_np = multi_swap_pool(n_quads, N_RUNS, R)
    all_starts = jnp.asarray(starts_np.reshape(N_RUNS, -1))       # (N_RUNS, Nq*13)
    all_goalsf = jnp.asarray(goals_full_np.reshape(N_RUNS, -1))   # (N_RUNS, Nq*13)
    all_phis = jnp.asarray(phis_np)

    nx_dim, nu_dim = 13, 4
    w_hover = 522.9847140714692

    def build_warm_start(starts_flat):
        """Hover at each quad's (rotated) start -- the same cold guess every
        solver uses, per instance."""
        s = starts_flat.reshape(n_quads, nx_dim)
        states = [jnp.tile(s[i][None, :], (N_HORIZON, 1)) for i in range(n_quads)]
        inputs = [jnp.full((N_HORIZON - 1, nu_dim), w_hover) for _ in range(n_quads)]
        return xu_to_z(states, inputs)

    def idx_from_sol(sol):
        # quad 0's first-state xy encodes phi (its base angle is 0); nearest
        # pool angle on the circle (IC is an equality constraint -> exact hit)
        phi = jnp.arctan2(sol[1], sol[0])
        dist = jnp.abs(jnp.mod(all_phis - phi + jnp.pi, 2.0 * jnp.pi) - jnp.pi)
        return jnp.argmin(dist).astype(jnp.int32)

    def calc_next_problem(key, sol):
        new_idx = (idx_from_sol(sol) + N_batch) % N_RUNS
        return (build_warm_start(all_starts[new_idx]),
                (all_goalsf[new_idx],), (all_starts[new_idx],), ())

    def calc_next_problem_seq(idx, sol):     # sequential pool: explicit index
        return (build_warm_start(all_starts[idx]),
                (all_goalsf[idx],), (all_starts[idx],), ())

    f_args, c_args, d_args = (all_goalsf[0],), (all_starts[0],), ()
    cp = initialize_common_problem(f, c, d, x_L, x_U, d_L, d_U, x0, p,
                                   [f_args, c_args, d_args],
                                   calc_next_problem=calc_next_problem,
                                   calc_next_problem_seq=calc_next_problem_seq)
    state = initialize_problem_regular(cp, x0, args=[f_args, c_args, d_args])
    state = eqx.tree_at(lambda t: t.fl.needs_regular_init, state,
                        jnp.array([[0]]))

    def inject(batch_tp, idx):
        idx = np.asarray(idx)
        bg, bs = all_goalsf[idx], all_starts[idx]
        batch_tp = eqx.tree_at(lambda s: s.args[0][4], batch_tp, bg)
        batch_tp = eqx.tree_at(lambda s: s.args[1][1], batch_tp, bs)
        warms = jax.vmap(build_warm_start)(bs)
        new_x = batch_tp.it.x.at[:, :cp.nx, 0].set(warms)
        return eqx.tree_at(lambda s: s.it.x, batch_tp, new_x)

    def objective(z, idx):
        return float(f(jnp.asarray(z), all_goalsf[int(idx)]))

    def _idx_np(x0_row):
        phi = np.arctan2(x0_row[1], x0_row[0])
        dist = np.abs(np.mod(phis_np - phi + np.pi, 2.0 * np.pi) - np.pi)
        return int(np.argmin(dist))

    def match_pool(z):
        xs, _ = z_to_xu(jnp.asarray(z))
        return _idx_np(np.asarray(xs[0][0]))

    def to_X(z):
        xs, _ = z_to_xu(jnp.asarray(z))
        return np.stack([np.asarray(x) for x in xs])   # (N_quads, N, 13)

    def success(X, iters, max_iter):
        k = _idx_np(np.asarray(X)[0, 0])   # instance from quad-0 start row
        return check_success(X, list(goals_np[k]), int(iters), max_iter=max_iter)

    def c_fn(z, idx):
        return np.asarray(c(jnp.asarray(z), all_starts[int(idx)]))

    def d_fn(z, idx):
        return np.asarray(d(jnp.asarray(z)))

    c_batch = jax.jit(jax.vmap(lambda z, i: c(z, all_starts[i])))
    d_batch = jax.jit(jax.vmap(lambda z: d(z)))

    def match_batch(Z):
        Z = np.asarray(Z)
        phi = np.arctan2(Z[:, 1], Z[:, 0])
        dist = np.abs(np.mod(phis_np[None, :] - phi[:, None] + np.pi,
                             2.0 * np.pi) - np.pi)
        return dist.argmin(axis=1)

    return Scenario(
        c_fn=c_fn, d_fn=d_fn, c_batch=c_batch, d_batch=d_batch, match_batch=match_batch,
        bounds=tuple(np.asarray(b) for b in (x_L, x_U, d_L, d_U)),
        name="multi", variant=n_quads, tag=f"{int(n_quads)}",
        label=f"multi_q{int(n_quads)}",
        logs_dir=os.path.join("tests", "quad_multi_swap", "logs"),
        N_RUNS=N_RUNS, N_batch_paper=tmp["batch_size"], cp=cp, state=state,
        f=lambda z: f(z, all_goalsf[match_pool(z)]),
        inject=inject, objective=objective, match_pool=match_pool,
        to_X=to_X, success=success,
        meta=dict(N=N_HORIZON, Ts=TS, R=R, N_quads=n_quads,
                  pool="rot", n_pool=N_RUNS),
        extra_npz=dict(pool_phis=phis_np, pool_psis=psis_np, pool_starts=starts_np,
                       pool_goals=goals_np,
                       N=np.array([N_HORIZON]), Ts=np.array([TS]),
                       R=np.array([R]), N_quads=np.array([n_quads])),
    )


BUILDERS = {"nav": build_nav, "track": build_track, "multi": build_multi}
DEFAULT_VARIANTS = {"nav": [90, 180], "track": [1.0, 1.5, 2.0],
                    "multi": [2, 4]}


def parse_variant(name, s):
    return {"nav": int, "track": float, "multi": int}[name](s)


def build_scenario(name, variant, p, N_batch):
    return BUILDERS[name](variant, p, N_batch)

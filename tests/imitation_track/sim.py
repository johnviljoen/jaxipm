"""Closed-loop simulator, reference, obstacles and student features for the
track-avoid scenario (tests/quad_track_avoid).

Reference: the "pringle" curve x = A cos s, y = B sin s, z = C cos 2s with
s advancing at AVG_VEL rad/s. Obstacles: three cylinders whose centres
oscillate in absolute time. Both are functions of the episode's start phase
s0 and the simulation time t, which closed loop makes explicit (the paper's
open-loop problem always starts at t = 0).

World model = the NLP's forward-Euler step (Ts = 0.1 s), as for nav.
Import from the repo root (the track module loads JSON by relative path).
"""

import numpy as np
import jax
import jax.numpy as jnp

from tests.quad_jax_dynamics import f_jnp
from tests.quad_track_avoid.jaxipm_track_avoid import quad_params, TS, N_HORIZON, tmp as TRACK_PARAMS
from tests.quad_track_avoid.initialization import (
    obstacles as OBSTACLES, PRINGLE_A, PRINGLE_B, PRINGLE_C, generate_random_inits)

NX, NU = 13, 4
W_HOVER = 522.9847140714692
U_MIN = float(quad_params["minWmotor"])
U_MAX = float(quad_params["maxWmotor"])
X_LB = np.asarray(quad_params["x_lb"], dtype=np.float64)
X_UB = np.asarray(quad_params["x_ub"], dtype=np.float64)
N_OBS = len(OBSTACLES)
OBS_R = np.array([o["r"] for o in OBSTACLES])          # (N_OBS,), no quad inflation in the paper
_OBS = {k: jnp.array([o[k] for o in OBSTACLES]) for k in ("xc0", "yc0", "ax", "ay", "fx", "fy", "px", "py")}
DELTA_STD = 0.5                                       # paper's start offset std [m]
N_POOL_PAPER = int(TRACK_PARAMS["N_RUNS_jaxipm"])     # 1000
FEAT_STAGES = jnp.arange(0, N_HORIZON, 3)             # 10 reference points fed to the student
OBS_STAGES = jnp.array([0, N_HORIZON // 2, N_HORIZON - 1])
FEAT_DIM = NX + 2 * 3 * FEAT_STAGES.shape[0] + 2 * N_OBS * OBS_STAGES.shape[0]   # 13 + 60 + 18 = 91
POS_SCALE, VEL_SCALE = 6.0, 5.0


def step(x, u):
    """Euler step as in the NLP, then quaternion renormalisation (the plant keeps a
    unit quaternion; un-normalised Euler inflates |q| monotonically and would push
    long episodes into the NLP's |q|^2 <= 1.5 bound)."""
    xn = x + f_jnp(x, u, quad_params) * TS
    q = xn[3:7]
    return xn.at[3:7].set(q / jnp.linalg.norm(q))


# ── reference and obstacles ──────────────────────────────────────────────────
def ref_state(s, v):
    """Full 13-state pringle reference at phase s (jnp; matches pringle_state_np)."""
    x_ = PRINGLE_A * jnp.cos(s); y_ = PRINGLE_B * jnp.sin(s); z_ = PRINGLE_C * jnp.cos(2 * s)
    xd = -PRINGLE_A * jnp.sin(s) * v; yd = PRINGLE_B * jnp.cos(s) * v; zd = -2 * PRINGLE_C * jnp.sin(2 * s) * v
    yaw = jnp.arctan2(yd, xd)
    return jnp.array([x_, y_, z_, jnp.cos(0.5 * yaw), 0.0, 0.0, jnp.sin(0.5 * yaw), xd, yd, zd, 0.0, 0.0, 0.0])


def ref_horizon(s0, t, v):
    """(N_HORIZON, 13) reference for an NLP whose stage 0 is at time t."""
    s = s0 + (t + jnp.arange(N_HORIZON) * TS) * v
    return jax.vmap(ref_state, in_axes=(0, None))(s, v)


def obstacle_centers(t):
    """(N_OBS, 2) obstacle centres at absolute time t."""
    xc = _OBS["xc0"] + _OBS["ax"] * jnp.sin(_OBS["fx"] * t + _OBS["px"])
    yc = _OBS["yc0"] + _OBS["ay"] * jnp.sin(_OBS["fy"] * t + _OBS["py"])
    return jnp.stack([xc, yc], axis=1)


def obstacle_horizon(t):
    """(N_HORIZON, N_OBS, 2) obstacle centres at t + k Ts."""
    return jax.vmap(obstacle_centers)(t + jnp.arange(N_HORIZON) * TS)


# ── starts ───────────────────────────────────────────────────────────────────
def paper_pool(v):
    """The paper's 1000 (x0, s0) pairs for this avg_vel (seed 0)."""
    x0, _, s0, _ = generate_random_inits(N_POOL_PAPER, N_HORIZON, TS, v, delta_std=DELTA_STD, seed=0)
    return np.asarray(x0), np.asarray(s0)


def heldout_starts(n, v):
    """n evenly spaced members of the paper pool (held out from training)."""
    x0, s0 = paper_pool(v)
    idx = np.round(np.linspace(0, x0.shape[0] - 1, n)).astype(int)
    return x0[idx], s0[idx]


def sample_starts(n, v, seed):
    """n fresh random starts with the paper's recipe (offset at rest, obstacle rejection)."""
    x0, _, s0, _ = generate_random_inits(n, N_HORIZON, TS, v, delta_std=DELTA_STD, seed=int(seed))
    return np.asarray(x0), np.asarray(s0)


# ── student features ─────────────────────────────────────────────────────────
def features(x, s0, t, v):
    """(FEAT_DIM,) normalised input: own state, reference ahead (relative
    positions + velocities at 10 stages), obstacle centres relative to the
    quad now, mid-horizon and end-of-horizon."""
    xr = ref_horizon(s0, t, v)[FEAT_STAGES]                      # (10, 13)
    obs = obstacle_horizon(t)[OBS_STAGES]                         # (3, N_OBS, 2)
    return jnp.concatenate([
        x[:3] / POS_SCALE, x[3:7], x[7:] / VEL_SCALE,
        ((xr[:, :3] - x[:3]) / POS_SCALE).ravel(),
        (xr[:, 7:10] / VEL_SCALE).ravel(),
        ((obs - x[:2]) / POS_SCALE).ravel(),
    ])


features_batch = jax.jit(jax.vmap(features, in_axes=(0, 0, 0, None)), static_argnums=3)


def rollout(policy_feat, x0s, s0s, T, v):
    """Closed loop under u = policy_feat(F) with F = features(x, s0, t).

    Returns X (B, T+1, 13), U (B, T, 4). Jit with the policy closed over.
    """
    def body(X, k):
        t = k * TS
        F = jax.vmap(features, in_axes=(0, 0, None, None))(X, s0s, t, v)
        U = policy_feat(F)
        Xn = jax.vmap(step)(X, U)
        return Xn, (Xn, U)
    _, (Xs, Us) = jax.lax.scan(body, x0s, jnp.arange(T))
    X = jnp.concatenate([x0s[:, None], jnp.swapaxes(Xs, 0, 1)], axis=1)
    return X, jnp.swapaxes(Us, 0, 1)


# ── evaluation ───────────────────────────────────────────────────────────────
def ref_positions(s0s, T, v):
    """(B, T+1, 3) reference positions at t_k = k Ts."""
    t = np.arange(T + 1) * TS
    s = np.asarray(s0s)[:, None] + t[None, :] * v
    return np.stack([PRINGLE_A * np.cos(s), PRINGLE_B * np.sin(s), PRINGLE_C * np.cos(2 * s)], axis=-1)


def obstacle_path(T):
    """(T+1, N_OBS, 2) obstacle centres at t_k (numpy)."""
    return np.asarray(jax.vmap(obstacle_centers)(jnp.arange(T + 1) * TS))


def valid_states(X):
    X = np.asarray(X)
    return np.isfinite(X).all(axis=-1) & ((X >= X_LB) & (X <= X_UB)).all(axis=-1)


def min_clearance(X):
    """Min over time of distance to each MOVING obstacle minus its radius, (B,)."""
    X = np.asarray(X)
    T1 = X.shape[1]
    oc = obstacle_path(T1 - 1)                                  # (T1, N_OBS, 2)
    d = np.linalg.norm(X[:, :, None, :2] - oc[None], axis=-1) - OBS_R[None, None]
    return d.min(axis=(1, 2))


def evaluate(X, s0s, v, tail=20, err_tol=0.5):
    """Per-trajectory metrics. success = finite, no collision, RMS position
    error over the last `tail` steps below err_tol [m]."""
    X = np.asarray(X)
    fin = np.isfinite(X).all(axis=(1, 2))
    Xf = np.where(fin[:, None, None], X, 0.0)
    err = np.linalg.norm(Xf[:, :, :3] - ref_positions(s0s, X.shape[1] - 1, v), axis=-1)   # (B, T+1)
    rms_tail = np.sqrt(np.mean(err[:, -tail:] ** 2, axis=1))
    rms_tail = np.where(fin, rms_tail, np.inf)
    clear = np.where(fin, min_clearance(Xf), -np.inf)
    success = fin & (clear > 0.0) & (rms_tail < err_tol)
    return dict(success=success, finite=fin, rms_tail=rms_tail, err=err, min_clearance=clear,
                final_err=np.where(fin, err[:, -1], np.inf))

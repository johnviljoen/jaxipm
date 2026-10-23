"""Closed-loop quadcopter simulator for the nav scenario (tests/quad_nav_circle).

The world model is the SAME forward-Euler step the nav NLP uses in its
equality constraints (x_{k+1} = x_k + f(x_k, u_k) * Ts, Ts = 0.1 s), so the
receding-horizon expert sees no model mismatch and the student is judged in
the world it was taught in.

Start states are at rest with identity quaternion on the paper's arc of
radius sqrt(32) around the goal (origin, 2 m up = NED z = -2), inside the
sector centred at 45 deg.

Import from the repo root (the nav module loads JSON by relative path).
"""

import numpy as np
import jax
import jax.numpy as jnp

from tests.quad_jax_dynamics import f_jnp
from tests.quad_nav_circle.jaxipm_quad_nav import (
    quad_params, TS, N_HORIZON, OBSTACLES, QUAD_RADIUS, tmp as NAV_PARAMS)

NX, NU = 13, 4
W_HOVER = 522.9847140714692
GOAL = np.array([0.0, 0.0, 0.0])        # the objective penalises x^2: minimum is the origin
WARM_END = np.array([0.0, 0.0, -2.0])   # the paper driver warm-start endpoint (kept as is)
RADIUS = float(np.sqrt(4.0 ** 2 + 4.0 ** 2))
SECTOR_CENTER = np.pi / 4.0
U_MIN = float(quad_params["minWmotor"])
U_MAX = float(quad_params["maxWmotor"])
X_LB = np.asarray(quad_params["x_lb"], dtype=np.float64)
X_UB = np.asarray(quad_params["x_ub"], dtype=np.float64)
OBS = [(o["xc"], o["yc"], o["r"] + QUAD_RADIUS) for o in OBSTACLES]  # inflated


def step(x, u):
    """One 10 Hz Euler step of the quad (x: (13,), u: (4,) motor speeds)."""
    return x + f_jnp(x, u, quad_params) * TS


def sector_bounds(sector_deg):
    h = np.deg2rad(sector_deg) / 2.0
    return SECTOR_CENTER - h, SECTOR_CENTER + h


def arc_starts(angles):
    """At-rest start states on the arc, (n, 13) float64."""
    angles = np.asarray(angles, dtype=np.float64)
    x0 = np.zeros((angles.size, NX), dtype=np.float64)
    x0[:, 0] = RADIUS * np.cos(angles)
    x0[:, 1] = RADIUS * np.sin(angles)
    x0[:, 3] = 1.0
    return x0


def paper_angles(sector_deg):
    """The paper's deterministic linspace over the sector (N_RUNS_jaxipm pts)."""
    lo, hi = sector_bounds(sector_deg)
    return np.linspace(lo, hi, int(NAV_PARAMS["N_RUNS_jaxipm"]))


def heldout_angles(n, sector_deg):
    """n evenly spaced members of the paper linspace (held out from training)."""
    ang = paper_angles(sector_deg)
    idx = np.round(np.linspace(0, ang.size - 1, n)).astype(int)
    return ang[idx]


def sample_angles(key, n, sector_deg):
    lo, hi = sector_bounds(sector_deg)
    return np.asarray(jax.random.uniform(key, (n,), minval=lo, maxval=hi, dtype=jnp.float64))


def rollout(policy_batch, x0s, T):
    """Closed loop under a batched policy  u = policy_batch(X, Uprev)  (X: (B,13), Uprev: (B,4)).

    The previous action starts at hover. Returns X (B, T+1, 13) and U (B, T, 4).
    Jit this with the policy closed over.
    """
    def body(carry, _):
        X, Up = carry
        U = policy_batch(X, Up)
        Xn = jax.vmap(step)(X, U)
        return (Xn, U), (Xn, U)
    u0 = jnp.full((x0s.shape[0], NU), W_HOVER, dtype=x0s.dtype)
    _, (Xs, Us) = jax.lax.scan(body, (x0s, u0), None, length=T)
    X = jnp.concatenate([x0s[:, None], jnp.swapaxes(Xs, 0, 1)], axis=1)
    return X, jnp.swapaxes(Us, 0, 1)


def valid_states(X):
    """Finite and inside the NLP's state box (an x0_ic outside it is infeasible)."""
    X = np.asarray(X)
    fin = np.isfinite(X).all(axis=-1)
    inb = ((X >= X_LB) & (X <= X_UB)).all(axis=-1)
    return fin & inb


def min_clearance(X):
    """Min over time of (distance to obstacle centre - inflated radius), (B,)."""
    X = np.asarray(X)
    cl = np.full(X.shape[:-1], np.inf)
    for xc, yc, r in OBS:
        cl = np.minimum(cl, np.hypot(X[..., 0] - xc, X[..., 1] - yc) - r)
    return cl.min(axis=-1)


def evaluate(X, goal_tol=0.3):
    """Per-trajectory success metrics for X (B, T+1, 13)."""
    X = np.asarray(X)
    fin = np.isfinite(X).all(axis=(1, 2))
    Xf = np.where(fin[:, None, None], X, 0.0)
    final_dist = np.where(fin, np.linalg.norm(Xf[:, -1, :3] - GOAL, axis=-1), np.inf)
    clear = min_clearance(Xf)
    reached = fin & (final_dist < goal_tol)
    success = reached & (clear > 0.0)
    return dict(success=success, reached=reached, final_dist=final_dist,
                min_clearance=clear, finite=fin)

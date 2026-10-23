"""Student policy for the nav scenario: (13-state, previous action) -> 4 motor speeds.

Input normalisation is fixed (position / 6, quaternion raw, velocities and
rates / 5, previous action as (u - hover) / 402). The previous action lets
the policy commit to one side of an obstacle where the expert's labels are
multimodal (left/right), which a memoryless regressor would average. Output is  u = W_HOVER + u_scale * tanh(mlp(x))  so the policy
can never leave the motor box. Training is MSE in the normalised output
space, Adam with warmup + cosine decay. The network is float32; the
simulator and solver stay float64, so __call__ casts in and out.
"""

import json
import os

import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
import optax

from tests.imitation_nav.sim import W_HOVER, U_MIN, U_MAX, NX, NU

POS_SCALE = 6.0
VEL_SCALE = 5.0
U_SCALE_MAX = min(W_HOVER - U_MIN, U_MAX - W_HOVER)   # 402.0: keeps u inside the motor box


def normalize_x(x, u_prev, use_prev):
    parts = [x[..., :3] / POS_SCALE, x[..., 3:7], x[..., 7:] / VEL_SCALE]
    if use_prev:
        parts.append((u_prev - W_HOVER) / U_SCALE_MAX)
    return jnp.concatenate(parts, axis=-1)


class Policy(eqx.Module):
    mlp: eqx.nn.MLP
    u_scale: float = eqx.field(static=True)
    use_prev: bool = eqx.field(static=True)   # previous-action input (default off: it
                                               # makes the student copy itself and run away)

    def __init__(self, key, u_scale=200.0, width=256, depth=3, use_prev=False):
        self.use_prev = bool(use_prev)
        self.mlp = eqx.nn.MLP(NX + (NU if use_prev else 0), NU, width, depth,
                              activation=jax.nn.gelu, key=key)
        self.u_scale = float(u_scale)

    def normalized_out(self, x, u_prev):
        """tanh(mlp(x[, u_prev])) in (-1, 1)^4 for a single float64 state (13,) and action (4,)."""
        return jnp.tanh(self.mlp(normalize_x(x, u_prev, self.use_prev).astype(jnp.float32)))

    def __call__(self, x, u_prev):
        """Single state (13,) + previous action (4,) float64 -> motor speeds (4,) float64."""
        return W_HOVER + self.u_scale * self.normalized_out(x, u_prev).astype(jnp.float64)

    def batched(self):
        """A jitted ((B,13), (B,4)) -> (B,4) function for closed-loop rollouts/timing."""
        return eqx.filter_jit(jax.vmap(self))


def u_scale_from_data(U, margin=1.1):
    dev = float(np.abs(np.asarray(U) - W_HOVER).max())
    return float(min(margin * dev, U_SCALE_MAX))


def train(key, X, Up, U, u_scale, steps=5000, batch=1024, lr=1e-3, width=256, depth=3, log_every=500,
          use_prev=False):
    """Train a fresh Policy on (X (n,13), Up (n,4) previous action, U (n,4) label) by MSE on tanh outputs."""
    k_init, k_data = jax.random.split(key)
    policy = Policy(k_init, u_scale=u_scale, width=width, depth=depth, use_prev=use_prev)
    X = jnp.asarray(np.asarray(X), dtype=jnp.float64)
    Up = jnp.asarray(np.asarray(Up), dtype=jnp.float64)
    T = jnp.clip((jnp.asarray(np.asarray(U), dtype=jnp.float64) - W_HOVER) / u_scale, -0.999, 0.999)
    T = T.astype(jnp.float32)
    n = X.shape[0]

    sched = optax.warmup_cosine_decay_schedule(0.0, lr, min(200, steps // 10), steps, lr * 1e-2)
    opt = optax.adam(sched)
    opt_state = opt.init(eqx.filter(policy, eqx.is_array))

    def loss_fn(pol, xb, ub, tb):
        pred = jax.vmap(pol.normalized_out)(xb, ub)
        return jnp.mean((pred - tb) ** 2)

    @eqx.filter_jit
    def train_step(pol, opt_state, k):
        idx = jax.random.randint(k, (batch,), 0, n)
        loss, grads = eqx.filter_value_and_grad(loss_fn)(pol, X[idx], Up[idx], T[idx])
        updates, opt_state = opt.update(grads, opt_state, eqx.filter(pol, eqx.is_array))
        pol = eqx.apply_updates(pol, updates)
        return pol, opt_state, loss

    hist = []
    keys = jax.random.split(k_data, steps)
    for i in range(steps):
        policy, opt_state, loss = train_step(policy, opt_state, keys[i])
        if (i + 1) % log_every == 0 or i == 0:
            hist.append((i + 1, float(loss)))
            print(f"    train step {i+1:5d}/{steps}  mse(norm) = {float(loss):.3e}", flush=True)
    return policy, hist


def save(policy, path):
    with open(path + ".json", "w") as fh:
        json.dump(dict(u_scale=policy.u_scale, use_prev=policy.use_prev,
                       width=policy.mlp.layers[0].out_features,
                       depth=len(policy.mlp.layers) - 1), fh)
    eqx.tree_serialise_leaves(path, policy)


def load(path):
    with open(path + ".json") as fh:
        meta = json.load(fh)
    template = Policy(jax.random.PRNGKey(0), u_scale=meta["u_scale"],
                      width=meta["width"], depth=meta["depth"], use_prev=meta.get("use_prev", False))
    return eqx.tree_deserialise_leaves(path, template)

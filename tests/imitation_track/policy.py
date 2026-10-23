"""Memoryless student policy for track-avoid: features (FEAT_DIM,) -> 4 motor speeds.

u = W_HOVER + u_scale * tanh(mlp(F)); MSE in the normalised output space; Adam
with warmup + cosine decay; float32 network, float64 interface.
"""

import json

import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
import optax

from tests.imitation_track.sim import W_HOVER, U_MIN, U_MAX, NU, FEAT_DIM

U_SCALE_MAX = min(W_HOVER - U_MIN, U_MAX - W_HOVER)   # 402.0


class Policy(eqx.Module):
    mlp: eqx.nn.MLP
    u_scale: float = eqx.field(static=True)

    def __init__(self, key, u_scale=200.0, width=256, depth=3, in_dim=FEAT_DIM):
        self.mlp = eqx.nn.MLP(in_dim, NU, width, depth, activation=jax.nn.gelu, key=key)
        self.u_scale = float(u_scale)

    def normalized_out(self, F):
        return jnp.tanh(self.mlp(F.astype(jnp.float32)))

    def __call__(self, F):
        """Single feature vector (FEAT_DIM,) float64 -> motor speeds (4,) float64."""
        return W_HOVER + self.u_scale * self.normalized_out(F).astype(jnp.float64)

    def batched(self):
        return eqx.filter_jit(jax.vmap(self))


def u_scale_from_data(U, margin=1.1):
    dev = float(np.abs(np.asarray(U) - W_HOVER).max())
    return float(min(margin * dev, U_SCALE_MAX))


def train(key, F, U, u_scale, steps=5000, batch=1024, lr=1e-3, width=256, depth=3, log_every=1000):
    k_init, k_data = jax.random.split(key)
    policy = Policy(k_init, u_scale=u_scale, width=width, depth=depth, in_dim=F.shape[1])
    F = jnp.asarray(np.asarray(F), dtype=jnp.float32)
    T = jnp.clip((jnp.asarray(np.asarray(U), dtype=jnp.float64) - W_HOVER) / u_scale, -0.999, 0.999).astype(jnp.float32)
    n = F.shape[0]
    sched = optax.warmup_cosine_decay_schedule(0.0, lr, min(200, steps // 10), steps, lr * 1e-2)
    opt = optax.adam(sched)
    opt_state = opt.init(eqx.filter(policy, eqx.is_array))

    def loss_fn(pol, fb, tb):
        return jnp.mean((jax.vmap(pol.normalized_out)(fb) - tb) ** 2)

    @eqx.filter_jit
    def train_step(pol, opt_state, k):
        idx = jax.random.randint(k, (batch,), 0, n)
        loss, grads = eqx.filter_value_and_grad(loss_fn)(pol, F[idx], T[idx])
        updates, opt_state = opt.update(grads, opt_state, eqx.filter(pol, eqx.is_array))
        return eqx.apply_updates(pol, updates), opt_state, loss

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
        json.dump(dict(u_scale=policy.u_scale, in_dim=policy.mlp.layers[0].in_features,
                       width=policy.mlp.layers[0].out_features, depth=len(policy.mlp.layers) - 1), fh)
    eqx.tree_serialise_leaves(path, policy)


def load(path):
    with open(path + ".json") as fh:
        m = json.load(fh)
    template = Policy(jax.random.PRNGKey(0), u_scale=m["u_scale"], width=m["width"],
                      depth=m["depth"], in_dim=m["in_dim"])
    return eqx.tree_deserialise_leaves(path, template)

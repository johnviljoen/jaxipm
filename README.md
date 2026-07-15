# jaxipm

We present our jaxipm work from our paper: [Scaling Nonlinear Optimization: Many Problems, One GPU](https://arxiv.org/abs/2606.26341)

# Installation

Requirements:
* An NVIDIA GPU of **Turing generation (compute capability 7.5) or newer**
* **CUDA 13**
* **Python 3.12 or newer**
* Linux x86-64 only

jaxipm uses [spineax](https://github.com/johnviljoen/spineax) (cuDSS bindings) for its batched sparse linear solves, and [jax2sympy](https://github.com/johnviljoen/jax2sympy) for sparse symbolic derivatives.

pip:
```bash
pip install jaxipm
```
uv:
```bash
uv pip install jaxipm
```

> Using a uv-managed project instead? Just `uv add jaxipm`.

The IPOPT cross-check (`tests/correctness/jaxipm_correctness.py`) runs out of the box against the reference logs committed in `tests/correctness/ipopt_logs/`. Regenerating those logs (`tests/correctness/ipopt_correctness.py`) is only needed if the test problem changes, and additionally requires `casadi` and `cyipopt` built against our logging-instrumented IPOPT fork.

# Usage

A minimal constrained NLP — minimize `(x0-2)^2 + (x1-1)^2` subject to `x0^2 + x1^2 <= 1` (solution: the projection of `(2, 1)` onto the unit disk). The objective `f` returns a scalar; equality constraints `c` and inequality constraints `d` return 1-D arrays, or `None` if absent; `d_L <= d(x) <= d_U` and `x_L <= x <= x_U` are the bounds. Solver parameters are IPOPT-style; `jaxipm.default_params()` returns the defaults (shipped as `jaxipm/params.json`):

```python
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from jaxipm import default_params
from jaxipm.initialization import initialize_common_problem, initialize_problem_regular
from jaxipm.solver import solve

p = default_params()

f = lambda x: (x[0] - 2.0) ** 2 + (x[1] - 1.0) ** 2   # objective (scalar)
c = lambda x: None                                     # equality constraints (none)
d = lambda x: jnp.array([x[0] ** 2 + x[1] ** 2])       # inequality constraints (1-D)

x_L = jnp.array([-10.0, -10.0])                        # x_L <= x <= x_U
x_U = jnp.array([10.0, 10.0])
d_L = jnp.array([-jnp.inf])                            # d_L <= d(x) <= d_U
d_U = jnp.array([1.0])
x0 = jnp.array([0.0, 0.0])                             # initial iterate

cp = initialize_common_problem(f, c, d, x_L, x_U, d_L, d_U, x0, p)
state = initialize_problem_regular(cp, x0)
state, term = solve(cp, state)

print(state.it.x[:2].ravel())   # [0.894 0.447] = (2, 1) / sqrt(5)
```

The whole pipeline composes with `jax.vmap`: parameterize the problem through the function args, vmap over initialization + solve, and a batch of problems becomes ONE block-diagonal KKT factorization per iteration (the point of this project — see the paper). Solving a batch of projections onto the unit disk, one per target:

```python
import equinox as eqx

f = lambda x, target: jnp.sum((x - target) ** 2)       # objective, parameterized
targets = jnp.array([[2.0, 1.0], [-1.5, 0.5], [0.3, 0.1]])

cp = initialize_common_problem(f, c, d, x_L, x_U, d_L, d_U, x0, p,
                               [(targets[0],), (), ()])   # sample args: structure is analyzed ONCE

def solve_one(target):
    state = initialize_problem_regular(cp, x0, args=[(target,), (), ()])
    state, term = solve(cp, state)
    return state.it.x[:2, 0], term

xs, terms = eqx.filter_vmap(solve_one)(targets)
# xs: the two exterior targets projected onto the disk, the interior one returned exactly
```

Finally, `solve_throughput` — the entry point behind the paper's results — adds **iteration-level batching**: every batch slot runs its own problem, and the moment a slot converges its solution is scattered into a result buffer and the slot is hot-restarted with a fresh problem drawn from an rng key, so the GPU never idles on converged slots. The problem family below parameterizes both the objective (`target`) and the constraint (`r2`), starts every slot from its own `x0`, and refills converged slots via `calc_next_problem`:

```python
from jaxipm.solver import solve_throughput

# problem FAMILY: minimize ||x - target||^2  s.t.  ||x||^2 <= r^2
f = lambda x, target: jnp.sum((x - target) ** 2)        # parameterized objective
c = lambda x: None
d = lambda x, r2: jnp.array([x[0]**2 + x[1]**2 - r2])   # parameterized constraint, d(x) <= 0
d_L = jnp.array([-jnp.inf]); d_U = jnp.array([0.0])

def sample_problem(key):
    k1, k2, k3 = jax.random.split(key, 3)
    target = jax.random.uniform(k1, (2,), minval=-2.0, maxval=2.0)
    r2 = jax.random.uniform(k2, (), minval=0.25, maxval=1.0)
    x0 = 0.1 * jax.random.normal(k3, (2,))
    return x0, target, r2

def calc_next_problem(rng_key, sol):
    """Refill a converged slot: fresh target/radius, warm-started near the
    slot's previous solution."""
    x0, target, r2 = sample_problem(rng_key)
    return 0.5 * sol + 0.5 * x0, (target,), (), (r2,)

key = jax.random.PRNGKey(0)
B, MAX_SOLVES = 4, 12
x0s, targets, r2s = jax.vmap(sample_problem)(jax.random.split(key, B))

cp = initialize_common_problem(f, c, d, x_L, x_U, d_L, d_U, x0s[0], p,
                               [(targets[0],), (), (r2s[0],)],
                               calc_next_problem=calc_next_problem)

def init_one(x0, target, r2):   # per-slot x0 AND per-slot problem parameters
    state = initialize_problem_regular(cp, x0, args=[(target,), (), (r2,)])
    return eqx.tree_at(lambda s: s.fl.needs_regular_init, state, jnp.array([[0]]))

batch = eqx.filter_vmap(init_one)(x0s, targets, r2s)

final_state, solutions, n = eqx.filter_jit(solve_throughput)(
    jax.random.PRNGKey(1), cp, batch, MAX_SOLVES)
# solutions: (MAX_SOLVES, nx) buffer — 12 solved problems from a 4-slot batch
```

See `tests/quad_track_avoid/` and `tests/quad_multi_swap/` for the full-scale versions of this workflow used for the paper's results (quadcopter trajectory optimization, 1000s of solves).

# Research FAQ's

## Magic Numbers

The only "magic numbers" we have in the results presented is in the batch size and number of optimization results expected for each problem (beyond the parameters for the optimization, which we borrow from IPOPT, and have proven robust in IPOPTs case). We chose the number of optimization results to be small enough such that it wouldnt take forever on our hardware (1x L40s GPU), and chose the batch sizes to be roughly 1/4 of this. We chose the batch size to be roughly 1/4 to demonstrate the necessity of the iteration-level batching, which has a more pronounced effect when optimizations have diverged a bit, which is more the case when we need to do a few resets along the way. In my experimentation the throughput results were'nt very sensitive to batch size, but feel free to experiment (just takes a while to compile and run - but I am working on the compilation side of things ;) )!

## Why is Throughput Higher than GPU-Accelerated Sequential MadNLP?

This may not be obvious to the outsider so I just wanted to take a moment to mention it here. MadNLP does a great job GPU accelerating both the evaluation of sparse derivatives which constitute the KKT matrices (via examodels), and in the solving of the sparse symmetric positive definite condensed KKT matrix reformulation via cuDSS. We also use cuDSS in this work, but we solve the symmetric indefinite KKT matrix formulation used by IPOPT. When MadNLP calls cuDSS, a given problem is not guaranteed to fully saturate the GPU - leaving performance on the table. However, when we batch solve our symmetric indefinite KKT matrix (the same form as IPOPT uses), we almost certainly fully saturate the GPU (and if we don't then we can simply increase the batch size until we do). This means we get more linear solves per second from the GPU than the non GPU-batched (but still GPU-accelerated) MadNLP implementation.

## Why Don't we use the MadNLP Positive Definite KKT Formulation?

For this work we decided to use cuDSS for our sparse linear solves, as it is the fastest currently available direct sparse set of linear solvers on GPU that exist today. Both the IPOPT and MadNLP KKT matrix formulations require inertia correction (perturb the diagonal such that we find directions to local minima instead of saddles and maxima). MadNLP does this by detecting failure of positive definite symmetric linear solves in cuDSS, and inertia correcting. This does not work in the batched case because a batch of linear solves on cuDSS is seen as a block diagonal set of linear systems, and if any one of them fails, the whole batch fails, without a way to detect which element of the batch failed. Therefore we use the cuDSS symmetric indefinite linear solver, which gives us explicit matrix inertias for each linear solve in the batch, and allows us to inertia correct them individually. That being SAID - I would like to implement something like BaSpaCho to enable the MadNLP KKT formulation in (GPU-)batch in the future!

# Citation

```
@article{viljoen2026scaling,
  title={Scaling Nonlinear Optimization: Many Problems One GPU},
  author={Viljoen, John and Haffner, Johanna and Tomizuka, Masayoshi and Mehr, Negar},
  journal={arXiv preprint arXiv:2606.26341},
  year={2026}
}
```

"""Run I3 -- iterative refinement on a BATCHED KKT system: per-member residual
at every refinement step, and whether one member's blow-up touches the others.

Protocol
  1. Build the paper batch for a scenario, run `k` fused iterations of the
     real solver (hot-restart on, ir_nsteps = 0, exactly as in run A).
  2. Take the batch's CURRENT factorized KKT matrices (ic.token, i.e. the
     inertia-corrected values the solver just used) and Newton right-hand
     sides (cqpo.rhs), and re-solve them with the SAME refinement loop spineax
     runs for ir_nsteps > 0 (x <- x + K^{-1}(r - K x), fixed step count, no
     convergence test), logging  ||r - K x_j||_2 / ||r||_2  PER MEMBER at every
     step j = 0..J.
  3. Coupling test: pick the member with the worst final residual ("bad"),
     re-mint a batch token WITHOUT it, rerun the same loop on the remaining
     members, and compare their per-step residuals with the with-bad-member
     run (max abs relative difference). Block-diagonal factorization + a
     JAX-side blockwise loop predicts zero coupling; a non-zero difference (or
     NaN propagation) would indicate a batch-global quantity.

Repeats 2-3 over several k (fused-iteration snapshots) because refinement
behaviour depends on the conditioning of that iteration's K.

Output: tests/rebuttal/results/i3_<scenario>_<variant>.npz + a printed table.
Run from the repo root:
  python -m tests.rebuttal.i3_refinement_divergence --scenario nav --variant 90 \
      --snapshots 5,20,60 --steps 20 --gpu 2
"""

import argparse
import json
import os
import sys

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", required=True, choices=["nav", "track", "multi"])
    ap.add_argument("--variant", required=True)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--snapshots", type=str, default="5,20,60",
                    help="fused-iteration indices at which to grab K, r")
    ap.add_argument("--steps", type=int, default=20, help="refinement steps J")
    ap.add_argument("--gpu", type=str, default="2")
    args = ap.parse_args()
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", args.gpu)
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.chdir(REPO); sys.path.insert(0, REPO)

    import jax, jax.numpy as jnp, equinox as eqx
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_compilation_cache_dir", "/home/john/tmp/jax_cache")
    from spineax import cudss
    from spineax.cudss.solver import _matvec
    from jaxipm.solver import make_batch_state
    from jaxipm.search import execute_search, post_process
    from tests.rebuttal.problems import build_scenario, load_params, parse_variant, paper_batch_size

    variant = parse_variant(args.scenario, args.variant)
    p = load_params(); p["DEBUG_MODE"] = False; p["hot_restarting"] = True; p["ir_nsteps"] = 0
    N_batch = args.batch or paper_batch_size(args.scenario)
    sc = build_scenario(args.scenario, variant, p, N_batch); cp = sc.cp
    bt = sc.inject(make_batch_state(cp, [sc.state] * N_batch), np.arange(N_batch) % sc.N_RUNS)
    bt = eqx.tree_at(lambda s: s.fl.needs_regular_init, bt, jnp.ones_like(bt.fl.needs_regular_init))

    vsearch = eqx.filter_vmap(execute_search, in_axes=(0, None))
    vpost = eqx.filter_vmap(post_process, in_axes=(0, 0, None))

    @eqx.filter_jit
    def fused_iters(state, k):
        def body(s, _):
            res = vsearch(s, cp)
            s2, term = vpost(s, res, cp)
            return s2, term.reshape(-1)
        return jax.lax.scan(body, state, None, length=k)

    def reduced(vec_it):
        return jnp.concatenate([vec_it.x[:, :cp.nx, 0], vec_it.s[:, :, 0],
                                vec_it.y_c[:, :, 0], vec_it.y_d[:, :, 0]], axis=1)

    from functools import partial
    @partial(jax.jit, static_argnums=2)
    def refine_log(token, r, J):
        # spineax _refined_solve semantics: x0 = K^{-1} r; x_{j+1} = x_j + K^{-1}(r - K x_j)
        x = cudss.solve(token, r, ir_nsteps=0)
        def step(carry, _):
            x = carry
            res = r - _matvec(token, x)
            rn = jnp.linalg.norm(res, axis=-1) / jnp.linalg.norm(r, axis=-1)
            x = x + cudss.solve(token, res, ir_nsteps=0)
            return x, rn
        x, rns = jax.lax.scan(step, x, None, length=J)
        final = jnp.linalg.norm(r - _matvec(token, x), axis=-1) / jnp.linalg.norm(r, axis=-1)
        return jnp.concatenate([rns, final[None]], axis=0)   # (J+1, B): residual BEFORE step j, then final

    def remint(values):
        return jax.vmap(lambda v: cudss.factorize(
            cudss.analyze(v, cp.solver_indptr, cp.solver_indices, mtype_id=1, mview_id=1), v))(values)

    snaps = [int(s) for s in args.snapshots.split(",")]
    out = {}
    state = bt; done_k = 0
    rows = []
    for k in snaps:
        state, terms = fused_iters(state, k - done_k); done_k = k
        jax.block_until_ready(state.it.x)
        token = state.ic.token
        r = reduced(state.cqpo.rhs)
        in_resto = np.asarray(state.fl.in_restoration).reshape(-1) > 0
        R = np.asarray(refine_log(token, r, args.steps))            # (J+1, B)
        bad = int(np.nanargmax(np.where(np.isfinite(R[-1]), R[-1], np.inf)))
        growth = R[-1] / np.maximum(R[0], 1e-300)
        n_grow = int(np.sum(growth > 1.0)); n_nonfinite = int(np.sum(~np.isfinite(R[-1])))
        # CONTROLS for the coupling test: (a) same token, same rhs, run again
        # (run-to-run nondeterminism); (b) the FULL batch re-minted (re-mint
        # variance). Both compared on members converging in both runs.
        R_same = np.asarray(refine_log(token, r, args.steps))
        R_full = np.asarray(refine_log(remint(token.values), r, args.steps))
        def _cmp(Ra, Rb):
            conv = (Ra[-1] < 1e-8) & (Rb[-1] < 1e-8)
            rel = np.abs(Rb[:, conv] - Ra[:, conv]) / np.maximum(np.abs(Ra[:, conv]), 1e-300)
            return dict(n_conv=int(conv.sum()), rel_med=float(np.median(rel)) if conv.any() else float('nan'),
                        rel_max=float(rel.max()) if conv.any() else float('nan'),
                        n_div_a=int((Ra[-1] >= 1e-8).sum()), n_div_b=int((Rb[-1] >= 1e-8).sum()))
        c_same, c_full = _cmp(R, R_same), _cmp(R, R_full)
        print(f"   k={k}: CONTROL same-token rerun: conv {c_same['n_conv']} rel-diff median {c_same['rel_med']:.2e} max {c_same['rel_max']:.2e} diverging {c_same['n_div_a']} vs {c_same['n_div_b']} | "
              f"full re-mint: conv {c_full['n_conv']} rel-diff median {c_full['rel_med']:.2e} max {c_full['rel_max']:.2e} diverging {c_full['n_div_a']} vs {c_full['n_div_b']}", flush=True)
        out[f"R_same_k{k}"] = R_same; out[f"R_full_k{k}"] = R_full
        # coupling test: drop the worst member, re-mint, rerun
        keep = np.array([i for i in range(N_batch) if i != bad])
        tok2 = remint(token.values[keep])
        R2 = np.asarray(refine_log(tok2, r[keep], args.steps))
        R1k = R[:, keep]
        with np.errstate(invalid="ignore", divide="ignore"):
            rel = np.abs(R2 - R1k) / np.maximum(np.abs(R1k), 1e-300)
        coupling = float(np.nanmax(rel[np.isfinite(R1k) & np.isfinite(R2)])) if np.isfinite(R1k).any() else float("nan")
        c_drop = _cmp(R1k, R2)
        print(f"   k={k}: DROP-WORST (coupling) on converging members: conv {c_drop['n_conv']} rel-diff median {c_drop['rel_med']:.2e} max {c_drop['rel_max']:.2e} diverging {c_drop['n_div_a']} vs {c_drop['n_div_b']}", flush=True)
        rows.append(dict(k=k, res0_med=float(np.nanmedian(R[0])), res0_max=float(np.nanmax(R[0])),
                         resJ_med=float(np.nanmedian(R[-1])), resJ_max=float(np.nanmax(R[-1][np.isfinite(R[-1])])) if np.isfinite(R[-1]).any() else float("nan"),
                         n_grow=n_grow, n_nonfinite=n_nonfinite, bad=bad, bad_in_resto=bool(in_resto[bad]),
                         bad_traj=R[:, bad].tolist(), coupling_max_rel_diff=coupling,
                         n_in_resto=int(in_resto.sum()), ctrl_same=c_same, ctrl_full=c_full, ctrl_drop=c_drop))
        out[f"R_k{k}"] = R; out[f"R_nobad_k{k}"] = R2; out[f"in_resto_k{k}"] = in_resto
        print(f"k={k:4d}: rel residual step0 median {rows[-1]['res0_med']:.2e} max {rows[-1]['res0_max']:.2e} | "
              f"after {args.steps} steps median {rows[-1]['resJ_med']:.2e} max {rows[-1]['resJ_max']:.2e} | "
              f"members growing {n_grow}/{N_batch}, non-finite {n_nonfinite} | worst member {bad} "
              f"(in_resto={in_resto[bad]}) traj {np.array2string(R[:, bad], precision=1)} | "
              f"coupling max|Δ|/|R| without worst member = {coupling:.2e}", flush=True)
    os.makedirs("tests/rebuttal/results", exist_ok=True)
    path = f"tests/rebuttal/results/i3_{sc.label}_b{N_batch}.npz"
    np.savez(path, rows=np.array([json.dumps(r) for r in rows]), steps=np.array([args.steps]),
             N_batch=np.array([N_batch]), **out)
    print("saved", path)


if __name__ == "__main__":
    main()

"""Trace one instance iteration-by-iteration at a given batch size to diagnose
the zero-rhs stagnation seen at small N (track v2.0). Prints per fused iteration:
branch, mu, scaled KKT error components, ||rhs||, ||step||, df (objective scaling),
in_restoration, dxs (IC perturbation), for the traced slot.
  python -m tests.rebuttal.debug_stagnation --scenario track --variant 2.0 --pool-idx 21 --batch 1 --iters 60 --gpu 1
"""
import argparse, os, sys
import numpy as np
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", required=True); ap.add_argument("--variant", required=True)
    ap.add_argument("--pool-idx", type=int, default=21); ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--iters", type=int, default=60); ap.add_argument("--gpu", default="1")
    ap.add_argument("--fill", type=str, default="linspace", help="other slots: 'linspace' pool instances or 'same'")
    args = ap.parse_args()
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", args.gpu); os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.chdir(REPO); sys.path.insert(0, REPO)
    import jax, jax.numpy as jnp, equinox as eqx
    jax.config.update("jax_enable_x64", True); jax.config.update("jax_compilation_cache_dir", "/home/john/tmp/jax_cache")
    from jaxipm.solver import make_batch_state
    from spineax import cudss
    from jaxipm.search import execute_search, post_process
    from tests.rebuttal.problems import build_scenario, load_params, parse_variant
    p = load_params(); p["DEBUG_MODE"] = True; p["hot_restarting"] = False
    sc = build_scenario(args.scenario, parse_variant(args.scenario, args.variant), p, args.batch); cp = sc.cp
    N = args.batch
    idx = np.full(N, args.pool_idx) if args.fill == "same" else np.round(np.linspace(0, sc.N_RUNS - 1, N)).astype(int)
    idx[0] = args.pool_idx
    bt = sc.inject(make_batch_state(cp, [sc.state] * N), idx)
    bt = eqx.tree_at(lambda s: s.fl.needs_regular_init, bt, jnp.ones_like(bt.fl.needs_regular_init))
    vsearch = eqx.filter_jit(eqx.filter_vmap(execute_search, in_axes=(0, None)))
    vpost = eqx.filter_jit(eqx.filter_vmap(post_process, in_axes=(0, 0, None)))
    def kkt(state):
        def one(it, mu, cqpr, a):
            conv, (oe, di, cv, ci) = cp.nstqf.calc_check_converged(it, mu, cqpr.grad_lag_x, cqpr.grad_lag_s, cqpr.slacks, cqpr.c, cqpr.d, a)
            return jnp.stack([jnp.reshape(oe, ()), jnp.reshape(di, ()), jnp.reshape(cv, ()), jnp.reshape(ci, ())])
        return np.asarray(eqx.filter_vmap(one, in_axes=(0, 0, 0, 0))(state.it, state.mu, state.cqpr, state.args))
    state = bt
    print(f"tracing slot 0 = pool {args.pool_idx} in a batch of {N} ({args.fill})")
    print(f"{'it':>3} {'br':>3} {'term':>4} {'mu':>9} {'kkt_err':>9} {'dual':>9} {'cviol':>9} {'compl':>9} {'|rhs|':>9} {'|step|':>9} {'df':>9} {'dxs':>8} {'resto':>5} {'obj':>10}")
    for k in range(args.iters):
        res = vsearch(state, cp); state, term = vpost(state, res, cp)
        e = kkt(state)[0]
        rh = state.cqpo.rhs; st = state.cqpo.step
        rn = float(jnp.linalg.norm(jnp.concatenate([rh.x[0, :cp.nx, 0], rh.s[0, :, 0], rh.y_c[0, :, 0], rh.y_d[0, :, 0]])))
        sn = float(jnp.linalg.norm(jnp.concatenate([st.x[0, :cp.nx, 0], st.s[0, :, 0], st.y_c[0, :, 0], st.y_d[0, :, 0]])))
        df = float(np.asarray(state.args[0][3]).reshape(-1)[0]) if np.asarray(state.args[0][3]).size >= 1 else float("nan")
        obj = sc.objective(np.asarray(state.it.x[0, :cp.nx, 0]), idx[0])
        try:   # cuDSS factorization diagnostics for slot 0: inertia readout, perturbed pivots, tiny |diag|
            qd = cudss.query(state.ic.token); n_sys = state.ic.token.values.shape[-1] if False else None
            diag = np.asarray(qd["diag"]).reshape(N, -1)[0]; npiv = int(np.asarray(qd["npivots"]).reshape(-1)[0])
            inert = f"{int((diag >= 1e-13).sum())}/{int((diag <= -1e-13).sum())}"; tiny = int((np.abs(diag) <= 1e-13).sum()); at_eps = int((np.abs(np.abs(diag) - 1e-13) < 1e-20).sum())
        except Exception as e:
            inert, npiv, tiny, at_eps = "?", -1, -1, -1
        extra = f" inertia(+/-)={inert} npiv={npiv} |d|<=1e-13:{tiny} |d|==1e-13:{at_eps} expected=({cp.nx + int(state.cqpo.step.s.shape[1])},{int(state.cqpo.step.y_c.shape[1]) + int(state.cqpo.step.y_d.shape[1])})"
        print(f"{k:>3} {int(state.fl.branch_id[0].reshape(-1)[0]):>3} {int(term[0].reshape(-1)[0]):>4} {float(state.mu[0].reshape(-1)[0]):>9.2e} "
              f"{e[0]:>9.2e} {e[1]:>9.2e} {e[2]:>9.2e} {e[3]:>9.2e} {rn:>9.2e} {sn:>9.2e} {df:>9.2e} "
              f"{float(state.ic.dxs[0].reshape(-1)[0]):>8.1e} {int(state.fl.in_restoration[0].reshape(-1)[0]):>5} {obj:>10.3f}" + extra, flush=True)
        if int(term[0].reshape(-1)[0]) > 0: print("terminated"); break
if __name__ == "__main__":
    main()

"""Run I1 cross-check: how accurate is ONE batched cuDSS solve (ir_nsteps=0)?

Takes the paper batch (or --batch N), runs k fused iterations of the real
solver, extracts every member's factorized KKT matrix (ic.token.values, the
inertia-corrected values the solver used) and Newton rhs (cqpo.rhs), then
(a) re-solves fresh with cudss.solve(ir_nsteps=0) and reports the relative
    residual ||K x - r|| / ||r|| per member (independent of the solver's own
    step bookkeeping),
(b) solves the same systems on the CPU with SciPy sparse LU (partial pivoting,
    reference accuracy) and reports both the reference residual and the
    relative error ||x_cudss - x_ref|| / ||x_ref||,
(c) reports the solver's own stored step residual for the same iteration, to
    validate the DEBUG-mode kkt_res_log pairing.
Run from the repo root, e.g.
  python -m tests.rebuttal.i1_solve_accuracy --scenario nav --variant 90 --batch 8 --snapshots 3,10 --gpu 2
"""
import argparse, os, sys, json
import numpy as np
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", required=True); ap.add_argument("--variant", required=True)
    ap.add_argument("--batch", type=int, default=None); ap.add_argument("--snapshots", default="3,10,30")
    ap.add_argument("--gpu", default="2"); ap.add_argument("--ref-members", type=int, default=8,
                    help="how many members to check against SciPy (CPU cost)")
    args = ap.parse_args()
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", args.gpu); os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.chdir(REPO); sys.path.insert(0, REPO)
    import jax, jax.numpy as jnp, equinox as eqx
    import scipy.sparse as sp, scipy.sparse.linalg as spla
    jax.config.update("jax_enable_x64", True); jax.config.update("jax_compilation_cache_dir", "/home/john/tmp/jax_cache")
    from spineax import cudss
    from spineax.cudss.solver import _matvec
    from jaxipm.solver import make_batch_state
    from jaxipm.search import execute_search, post_process
    from tests.rebuttal.problems import build_scenario, load_params, parse_variant, paper_batch_size
    variant = parse_variant(args.scenario, args.variant)
    p = load_params(); p["DEBUG_MODE"] = False; p["hot_restarting"] = True; p["ir_nsteps"] = 0
    N = args.batch or paper_batch_size(args.scenario)
    sc = build_scenario(args.scenario, variant, p, N); cp = sc.cp
    bt = sc.inject(make_batch_state(cp, [sc.state] * N), np.arange(N) % sc.N_RUNS)
    bt = eqx.tree_at(lambda s: s.fl.needs_regular_init, bt, jnp.ones_like(bt.fl.needs_regular_init))
    vsearch = eqx.filter_vmap(execute_search, in_axes=(0, None)); vpost = eqx.filter_vmap(post_process, in_axes=(0, 0, None))
    @eqx.filter_jit
    def fused(state, k):
        def body(s, _):
            res = vsearch(s, cp); s2, _t = vpost(s, res, cp); return s2, None
        return jax.lax.scan(body, state, None, length=k)[0]
    def reduced(v):
        return jnp.concatenate([v.x[:, :cp.nx, 0], v.s[:, :, 0], v.y_c[:, :, 0], v.y_d[:, :, 0]], axis=1)
    indptr = np.asarray(cp.solver_indptr); indices = np.asarray(cp.solver_indices); n = indptr.size - 1
    def full_csr(vals):
        A = sp.csr_matrix((vals, indices, indptr), shape=(n, n))
        D = sp.diags(A.diagonal()); return (A + A.T - D).tocsc()
    state = bt; done = 0; rows = []
    for k in [int(s) for s in args.snapshots.split(",")]:
        state = fused(state, k - done); done = k; jax.block_until_ready(state.it.x)
        token = state.ic.token; r = reduced(state.cqpo.rhs); d_stored = reduced(state.cqpo.step)
        in_resto = np.asarray(state.fl.in_restoration).reshape(-1) > 0
        x = cudss.solve(token, r, ir_nsteps=0)
        rn = lambda xx: np.asarray(jnp.linalg.norm(r - _matvec(token, xx), axis=-1) / jnp.linalg.norm(r, axis=-1))
        res_fresh, res_stored = rn(x), rn(d_stored)
        x2 = cudss.solve(token, r, ir_nsteps=2); res_ir2 = rn(x2)
        # CONTROLS: (i) re-mint a fresh symmetric-LDL token from token.values (rules out stale
        # cached factors in the registry); (ii) a general-LU token (mtype 0, full pattern).
        vals_j = token.values
        tok_fresh = jax.vmap(lambda v: cudss.factorize(cudss.analyze(v, cp.solver_indptr, cp.solver_indices, mtype_id=1, mview_id=1), v))(vals_j)
        res_fresh_mint = rn(cudss.solve(tok_fresh, r, ir_nsteps=0))
        # general LU on the symmetrically expanded matrix
        Af = [full_csr(np.asarray(vals_j[b])).tocsr() for b in range(N)]
        indptr_f, indices_f = Af[0].indptr, Af[0].indices
        vals_f = jnp.asarray(np.stack([A.data for A in Af]))
        tok_lu = jax.vmap(lambda v: cudss.factorize(cudss.analyze(v, jnp.asarray(indptr_f), jnp.asarray(indices_f), mtype_id=0, mview_id=0), v))(vals_f)
        x_lu = np.asarray(cudss.solve(tok_lu, r, ir_nsteps=0))
        res_lu = np.array([np.linalg.norm(Af[b] @ x_lu[b] - np.asarray(r)[b]) / np.linalg.norm(np.asarray(r)[b]) for b in range(N)])
        vals = np.asarray(token.values); rnp = np.asarray(r); xnp = np.asarray(x)
        # cuDSS factorization data: perturbed-pivot count and pivot magnitudes (static pivoting check)
        try:
            qd = cudss.query(token)
            npiv = int(np.asarray(qd["npivots"]).reshape(-1)[0])
            diag = np.abs(np.asarray(qd["diag"]).reshape(N, -1))
            piv_min = float(diag.min()); piv_tiny = int((diag < 1e-10).sum())
        except Exception as e:
            npiv, piv_min, piv_tiny = -1, float("nan"), -1
        # per-KKT-block residual localisation (x | s | y_c | y_d rows)
        blk = np.cumsum([0, cp.nx, int(state.cqpo.step.s.shape[1]), int(state.cqpo.step.y_c.shape[1]), int(state.cqpo.step.y_d.shape[1])])
        resvec = np.asarray(r - _matvec(token, x)); rnorm = np.linalg.norm(rnp, axis=1)
        blk_res = [float(np.median(np.linalg.norm(resvec[:, blk[i]:blk[i+1]], axis=1) / rnorm)) for i in range(4)]
        print(f"   k={k}: cuDSS npivots(perturbed)={npiv} min|pivot|={piv_min:.2e} #|pivot|<1e-10 per batch={piv_tiny} | "
              f"median block residual share x/s/y_c/y_d = {blk_res[0]:.1e}/{blk_res[1]:.1e}/{blk_res[2]:.1e}/{blk_res[3]:.1e}", flush=True)
        ref_res, ref_err, ref_cond_est = [], [], []
        for b in range(min(args.ref_members, N)):
            A = full_csr(vals[b]); lu = spla.splu(A); xr = lu.solve(rnp[b])
            ref_res.append(np.linalg.norm(A @ xr - rnp[b]) / np.linalg.norm(rnp[b]))
            ref_err.append(np.linalg.norm(xnp[b] - xr) / np.linalg.norm(xr))
            # residual of the cuDSS solution against the SciPy-assembled matrix (checks _matvec symmetric expansion)
            ref_cond_est.append(np.linalg.norm(A @ xnp[b] - rnp[b]) / np.linalg.norm(rnp[b]))
        row = dict(k=k, N=N, n_in_resto=int(in_resto.sum()),
                   fresh_med=float(np.median(res_fresh)), fresh_p95=float(np.percentile(res_fresh, 95)), fresh_max=float(res_fresh.max()),
                   stored_med=float(np.median(res_stored)), stored_max=float(res_stored.max()),
                   ir2_med=float(np.median(res_ir2)), ir2_max=float(res_ir2.max()),
                   fresh_mint_med=float(np.median(res_fresh_mint)), fresh_mint_max=float(res_fresh_mint.max()),
                   general_lu_med=float(np.median(res_lu)), general_lu_max=float(res_lu.max()),
                   scipy_res_max=float(max(ref_res)), cudss_vs_scipy_err_med=float(np.median(ref_err)), cudss_vs_scipy_err_max=float(max(ref_err)),
                   cudss_res_on_scipy_matrix_max=float(max(ref_cond_est)))
        rows.append(row)
        print(f"k={k:3d} N={N}: fresh cuDSS rel-res median {row['fresh_med']:.2e} p95 {row['fresh_p95']:.2e} max {row['fresh_max']:.2e} | "
              f"stored-step res median {row['stored_med']:.2e} max {row['stored_max']:.2e} | ir=2 median {row['ir2_med']:.2e} max {row['ir2_max']:.2e} | "
              f"REMINT sym-LDL res median {row['fresh_mint_med']:.2e} max {row['fresh_mint_max']:.2e} | cuDSS general-LU res median {row['general_lu_med']:.2e} max {row['general_lu_max']:.2e} | SciPy LU res max {row['scipy_res_max']:.1e}; ||x_cudss-x_ref||/||x_ref|| median {row['cudss_vs_scipy_err_med']:.2e} max {row['cudss_vs_scipy_err_max']:.2e} | "
              f"cuDSS res on SciPy matrix max {row['cudss_res_on_scipy_matrix_max']:.2e} | in_resto {int(in_resto.sum())}", flush=True)
    os.makedirs("tests/rebuttal/results", exist_ok=True)
    out = f"tests/rebuttal/results/i1_solve_accuracy_{sc.label}_b{N}.json"
    json.dump(rows, open(out, "w"), indent=1); print("saved", out)

if __name__ == "__main__":
    main()

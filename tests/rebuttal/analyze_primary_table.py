#!/usr/bin/env python
"""Primary-table quality columns for every solver: final cost mean ± sd over the pool and
constraint violation at the returned point (‖c‖∞ equality residual, max inequality violation,
max bound violation), evaluated with the SAME jaxipm-format c/d/bounds for every solver.

Sources per scenario (first existing wins):
  IPOPT default (one MUMPS-parallel IPOPT process, tol 1e-8, jaxipm pool, U saved):
        casadi_<tag>_pool_tol1e-08_results.npz   (fallback: paper casadi_<tag>_results.npz, tol 1e-6, cost only)
  IPOPT P=64 (run D, tol 1e-8):  casadi_cpu_throughput_<tag>_phys_c064_tol1e-08_results.npz  (X_all, U_all)
  MadNLP (run E pool, tol 1e-8): madnlp_<tag>_pool_tol1e-8_z_results.npz (z_all) (fallback: ..._pool_tol1e-8, cost only)
  jaxipm-IL (ILB on):  cost from the paper npz jaxipm_<tag>_results.npz; violations from the
                       run-F/G census file jaxipm_ablation_<tag>_hr_ir0_b<N>_dbg_results.npz (z_all)
  jaxipm-WS (ILB off): jaxipm_ablation_<tag>_ws_ir0_b<N>_dbg_results.npz (z_all, obj_vals)
Writes results/primary_table_quality.md/.json.  Needs a GPU visible (scenario build uses spineax).
"""
import argparse, glob, json, os, sys

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RES = os.path.join(REPO, "tests", "rebuttal", "results")
SC = [("nav", 90, "nav 90°", "tests/quad_nav_circle/logs", "sector90"),
      ("nav", 180, "nav 180°", "tests/quad_nav_circle/logs", "sector180"),
      ("track", 1.0, "track v̄=1.0", "tests/quad_track_avoid/logs", "v1.0"),
      ("track", 1.5, "track v̄=1.5", "tests/quad_track_avoid/logs", "v1.5"),
      ("track", 2.0, "track v̄=2.0", "tests/quad_track_avoid/logs", "v2.0"),
      ("multi", 2, "multi N=2", "tests/quad_multi_swap/logs", "2"),
      ("multi", 4, "multi N=4", "tests/quad_multi_swap/logs", "4")]


def ms(v):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    return (f"{v.mean():.3f} ± {v.std(ddof=1) if v.size > 1 else 0:.3f}", int(v.size)) if v.size else ("–", 0)


def z_from_xu(X, U):
    """jaxipm z layout: nav/track [x.flatten(), u.flatten()]; multi [x_q.flatten()..., u_q.flatten()...]."""
    X = np.asarray(X); U = np.asarray(U)
    if X.ndim == 2:
        return np.concatenate([X.reshape(-1), U.reshape(-1)])
    return np.concatenate([x.reshape(-1) for x in X] + [u.reshape(-1) for u in U])


def violations(sc, zs, ok, chunk=256):
    """(eq_inf, ineq_viol, bound_viol) arrays over rows with ok=True (NaN otherwise); batched + jitted."""
    x_L, x_U, d_L, d_U = [np.asarray(a, float).reshape(-1) for a in sc.bounds]
    n = len(zs); eq = np.full(n, np.nan); ineq = np.full(n, np.nan); bnd = np.full(n, np.nan)
    Z = np.stack([np.asarray(z, float).reshape(-1) for z in zs]) if n else np.zeros((0, x_L.size))
    good = np.asarray(ok, bool) & np.all(np.isfinite(Z), axis=1)
    rows = np.flatnonzero(good)
    for k in range(0, rows.size, chunk):
        r = rows[k:k + chunk]; Zc = Z[r]
        idx = np.asarray(sc.match_batch(Zc), dtype=np.int32)
        c = np.asarray(sc.c_batch(Zc, idx), float).reshape(len(r), -1)
        d = np.asarray(sc.d_batch(Zc), float).reshape(len(r), -1)
        eq[r] = np.max(np.abs(c), axis=1) if c.shape[1] else 0.0
        ineq[r] = np.maximum(0.0, np.max(np.maximum(d_L[None, :] - d, d - d_U[None, :]), axis=1)) if d.shape[1] else 0.0
        bnd[r] = np.maximum(0.0, np.max(np.maximum(x_L[None, :] - Zc, Zc - x_U[None, :]), axis=1))
    return eq, ineq, bnd


def qstr(a):
    a = np.asarray(a, float); a = a[np.isfinite(a)]
    if not a.size:
        return "–"
    return f"{np.median(a):.1e} / {np.mean(a):.1e} / {np.max(a):.1e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", default="5")
    ap.add_argument("--no-viol", action="store_true", help="cost columns only (no scenario build)")
    args = ap.parse_args()
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", args.gpu)
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.chdir(REPO); sys.path.insert(0, REPO)
    if not args.no_viol:
        import jax
        jax.config.update("jax_enable_x64", True)
        from tests.rebuttal.problems import build_scenario, load_params, paper_batch_size
        p = load_params()

    rows, out = [], {}
    for fam, var, name, logs, tag in SC:
        logs = os.path.join(REPO, logs)
        sc = None
        if not args.no_viol:
            sc = build_scenario(fam, var, p, paper_batch_size(fam))
        N = {"nav": 500, "track": 250, "multi": 75}[fam]
        solvers = {}

        def rec(key, obj, ok, zs=None, note=""):
            o = np.asarray(obj, float); okm = np.asarray(ok, bool) & np.isfinite(o)
            cost, n = ms(o[okm])
            r = dict(n=int(o.size), success=float(np.mean(okm)) if o.size else np.nan, cost=cost,
                     cost_mean=float(np.nanmean(o[okm])) if okm.any() else None,
                     cost_sd=float(np.nanstd(o[okm], ddof=1)) if okm.sum() > 1 else None, note=note)
            if zs is not None and sc is not None:
                eq, ineq, bnd = violations(sc, zs, okm)
                r.update(eq_inf=qstr(eq), ineq=qstr(ineq), bound=qstr(bnd),
                         eq_inf_median=float(np.nanmedian(eq)), eq_inf_max=float(np.nanmax(eq)),
                         ineq_max=float(np.nanmax(ineq)), bound_max=float(np.nanmax(bnd)))
            else:
                r.update(eq_inf="–", ineq="–", bound="–")
            solvers[key] = r

        # IPOPT default
        f = os.path.join(logs, f"casadi_{tag}_pool_tol1e-08_results.npz")
        if os.path.exists(f):
            d = np.load(f, allow_pickle=True)
            zs = [z_from_xu(X, U) for X, U in zip(d["X_all"], d["U_all"])]
            rec("IPOPT default (MUMPS, tol 1e-8, pool)", d["obj_vals"], d["success"], zs, "one process, all cores")
        else:
            f = os.path.join(logs, f"casadi_{tag}_results.npz")
            if os.path.exists(f):
                d = np.load(f, allow_pickle=True)
                rec("IPOPT default (paper, tol 1e-6)", d["obj_vals"], d["success"], None, "paper npz, X only")
        # IPOPT P=64
        f = os.path.join(logs, f"casadi_cpu_throughput_{tag}_phys_c064_tol1e-08_results.npz")
        if os.path.exists(f):
            d = np.load(f, allow_pickle=True)
            zs = [z_from_xu(X, U) for X, U in zip(d["X_all"], d["U_all"])]
            rec("IPOPT P=64 (run D, tol 1e-8)", d["obj_vals"], d["success"], zs)
        # MadNLP
        f = os.path.join(logs, f"madnlp_{tag}_pool_tol1e-8_z_results.npz")
        if os.path.exists(f):
            d = np.load(f, allow_pickle=True)
            zs = [np.asarray(z) for z in d["z_all"]]
            rec("MadNLP (tol 1e-8, pool)", d["obj_vals"], d["success"], zs, "‖c‖∞ = 2e-8 on the x0 block: MadNLP bound-relaxation of the fixed initial state (dynamics rows ~1e-15); z layout verified")
        else:
            f = os.path.join(logs, f"madnlp_{tag}_pool_tol1e-8_results.npz")
            if os.path.exists(f):
                d = np.load(f, allow_pickle=True)
                rec("MadNLP (tol 1e-8, pool)", d["obj_vals"], d["success"], None, "X only (z run pending)")
        # jaxipm-IL: cost from paper npz, violations from the HR census file
        f = os.path.join(logs, f"jaxipm_{tag}_results.npz")
        fh = os.path.join(logs, f"jaxipm_ablation_{tag}_hr_ir0_b{N}_dbg_results.npz")
        if os.path.exists(f):
            d = np.load(f, allow_pickle=True)
            zs = None; note = "paper npz (cost)"
            if os.path.exists(fh):
                dh = np.load(fh, allow_pickle=True)
                rec("jaxipm-IL (ILB on)", d["obj_vals"], d["success"], None, note)
                zs = [np.asarray(z) for z in dh["z_all"]]
                rec("jaxipm-IL census run (HR dbg)", dh["obj_vals"], dh["success"], zs, "same config, DEBUG run; violations from z_all")
            else:
                rec("jaxipm-IL (ILB on)", d["obj_vals"], d["success"], None, note)
        # jaxipm-WS
        fw = os.path.join(logs, f"jaxipm_ablation_{tag}_ws_ir0_b{N}_dbg_results.npz")
        if os.path.exists(fw):
            dw = np.load(fw, allow_pickle=True)
            rec("jaxipm-WS (ILB off, dbg)", dw["obj_vals"], dw["success"], [np.asarray(z) for z in dw["z_all"]])
        out[name] = solvers
        for k, r in solvers.items():
            rows.append(f"| {name} | {k} | {r['n']} | {100*r['success']:.1f}% | {r['cost']} | {r['eq_inf']} | {r['ineq']} | {r['bound']} | {r['note']} |")
        print(name, {k: (r["cost"], r["eq_inf"]) for k, r in solvers.items()}, flush=True)

    lines = ["# Primary-table quality columns: final cost (mean ± sd over the pool) and violations at the returned point", "",
             "Violations evaluated with jaxipm's own c(z)/d(z)/bounds for every solver (same functions, same scaling). "
             "Columns: ‖c‖∞ = max |equality residual| (dynamics + initial condition), ineq = max(d_L−d, d−d_U, 0), "
             "bound = max(x_L−z, z−x_U, 0); each reported as median / mean / max over successful solves. "
             "Cost sd is the spread over the problem pool (not over repeats). IPOPT P=64 = run D (pinned processes, tol 1e-8). "
             "IPOPT default = one IPOPT/MUMPS process using all cores (tol 1e-8, same pool). MadNLP = run E pool run.", "",
             "| scenario | solver | n | success | cost mean ± sd | ‖c‖∞ med/mean/max | ineq viol | bound viol | note |",
             "|---|---|---|---|---|---|---|---|---|"] + rows
    os.makedirs(RES, exist_ok=True)
    with open(os.path.join(RES, "primary_table_quality.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(RES, "primary_table_quality.json"), "w") as f:
        json.dump(out, f, indent=1, default=str)
    print("\n".join(lines))


if __name__ == "__main__":
    main()

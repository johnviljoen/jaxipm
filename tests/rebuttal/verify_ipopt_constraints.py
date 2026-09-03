"""Standalone verifier: recompute the full-data table's constraint-violation
entries with IPOPT's OWN constraint model (CasADi f_ca dynamics + CasADi
collision/bound algebra) instead of jaxipm's functions, and diff against the
numbers make_paper_tables stored (which used jaxipm's c_batch/d_batch).

metrics (identical formulas to analyze_primary_table.violations):
  eq_inf   = max_j |c(z)_j|                              (equality residual)
  ineq     = max(0, max_j (d_L - d, d - d_U))            (collision inequality)
  bound    = max(0, max(x_L - z, z - x_U))               (variable bounds)
aggregated over the SUCCESSFUL pool rows: eq_inf_max, eq_inf_mean(nanmean),
ineq_max, bound_max -- exactly the quality() reductions.
"""
import json, os, sys, glob
import numpy as np
import casadi as ca

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
os.chdir(REPO); sys.path.insert(0, REPO)
from tests.quad_casadi_dynamics import f_ca
from tests.quad_multi_swap.ipopt_multi_swap import quad_params as QP, MIN_DIST
from tests.quad_multi_swap.jaxipm_multi_swap import (multi_swap_pool, N_HORIZON, TS, R,
                                                     XY_MARGIN, Z_BOUND, VEL_RATE_BOUND)

def build_ca_checks(nq):
    N, nx, nu, Ts = N_HORIZON, 13, 4, TS
    sx, su = N*nx, (N-1)*nu
    nz = nq*sx + nq*su
    z = ca.SX.sym("z", nz)
    st = ca.SX.sym("st", nq*nx)                 # per-quad start states (flat)
    xs = [ca.reshape(z[i*sx:(i+1)*sx], nx, N) for i in range(nq)]     # (nx,N) col k = stage k
    ub = nq*sx
    us = [ca.reshape(z[ub+i*su:ub+(i+1)*su], nu, N-1) for i in range(nq)]
    # equality c : IC + Euler dynamics with the CasADi dynamics f_ca (IPOPT's own)
    ceq = []
    for i in range(nq):
        ceq.append(xs[i][:, 0] - st[i*nx:(i+1)*nx])
        for k in range(N-1):
            xnext = xs[i][:, k] + f_ca(xs[i][:, k], us[i][:, k], QP)*Ts
            ceq.append(xs[i][:, k+1] - xnext)
    c = ca.vertcat(*ceq)
    # inequality d : pairwise collision  ||pos_i-pos_j||^2 - min_dist^2
    dd = []
    md2 = MIN_DIST**2
    for i in range(nq):
        for j in range(i+1, nq):
            for k in range(N-1):
                diff = xs[i][0:3, k] - xs[j][0:3, k]
                dd.append(ca.dot(diff, diff) - md2)
    d = ca.vertcat(*dd)
    return (ca.Function("c", [z, st], [c]), ca.Function("d", [z], [d]), nz)

def z_bounds(nq):
    N, nx, nu = N_HORIZON, 13, 4
    sx, su = N*nx, (N-1)*nu
    xlb = np.array([-(R+XY_MARGIN), -(R+XY_MARGIN), -Z_BOUND, -np.inf,-np.inf,-np.inf,-np.inf,
                    -VEL_RATE_BOUND]*1 + [-VEL_RATE_BOUND]*5)
    xub = np.array([ (R+XY_MARGIN),  (R+XY_MARGIN),  Z_BOUND,  np.inf, np.inf, np.inf, np.inf,
                     VEL_RATE_BOUND]*1 + [VEL_RATE_BOUND]*5)
    zlb = np.full(nq*sx+nq*su, -np.inf); zub = np.full(nq*sx+nq*su, np.inf)
    for i in range(nq):
        for k in range(N):
            zlb[i*sx+k*nx:i*sx+k*nx+nx] = xlb; zub[i*sx+k*nx:i*sx+k*nx+nx] = xub
    ub0 = nq*sx
    for i in range(nq):
        for k in range(N-1):
            b = ub0+i*su+k*nu
            zlb[b:b+nu] = QP["minWmotor"]; zub[b:b+nu] = QP["maxWmotor"]
    return zlb, zub

def z_from_xu(X, U):
    X, U = np.asarray(X), np.asarray(U)
    return np.concatenate([x.reshape(-1) for x in X] + [u.reshape(-1) for u in U])

def viol_ca(nq, Z, ok):
    cfn, dfn, nz = build_ca_checks(nq)
    phis, psis, starts, goals, gf = multi_swap_pool(nq, 75, R)
    all_st = starts.reshape(75, -1)               # (75, nq*13)
    dphi_ref = phis
    zlb, zub = z_bounds(nq)
    n = len(Z); eq = np.full(n, np.nan); iq = np.full(n, np.nan); bd = np.full(n, np.nan)
    good = np.asarray(ok, bool) & np.all(np.isfinite(Z), axis=1)
    dl, du = 0.0, np.inf
    for r in np.flatnonzero(good):
        zc = Z[r]
        phi = np.arctan2(zc[1], zc[0])
        k = int(np.argmin(np.abs(np.mod(phis - phi + np.pi, 2*np.pi) - np.pi)))
        c = np.asarray(cfn(zc, all_st[k])).reshape(-1)
        d = np.asarray(dfn(zc)).reshape(-1)
        eq[r] = np.max(np.abs(c))
        iq[r] = max(0.0, np.max(np.maximum(dl - d, d - du)))
        bd[r] = max(0.0, np.max(np.maximum(zlb - zc, zc - zub)))
    return eq, iq, bd

L = "tests/quad_multi_swap/logs"
SRC = {2: {}, 4: {}}
for q in (2, 4):
    t = str(q)
    SRC[q] = {
      "IPOPT-default": (f"{L}/casadi_{t}_rot_pool_tol1e-08_results.npz", "xu"),
      "IPOPT-64":      (f"{L}/casadi_cpu_throughput_{t}_phys_c064_rot_tol1e-08_now_rep1_results.npz", "xu"),
      "MadNLP":        (f"{L}/madnlp_{t}_rot_pool_tol1e-8_results.npz", "z"),
      "jaxipm-SL":     (f"{L}/jaxipm_ablation_{t}_ws_ir0_b75_dbg_rot300_results.npz", "z"),
      "jaxipm-IL":     (f"{L}/jaxipm_ablation_{t}_hr_ir0_b75_dbg_rot300_results.npz", "z"),
    }

tbl = json.load(open("tests/rebuttal/results/paper_tables.json"))
scen_name = {2: "multi N=2", 4: "multi N=4"}
print(f"{'scenario':10s} {'solver':13s} {'metric':9s} {'CasADi(IPOPT)':>16s} {'table(jaxipm)':>16s} {'|diff|':>10s}")
maxdiff = 0.0
for q in (2, 4):
    for solver, (f, kind) in SRC[q].items():
        d = np.load(f, allow_pickle=True)
        ok = np.asarray(d["success"], bool)
        if kind == "xu":
            Z = np.stack([z_from_xu(X, U) for X, U in zip(d["X_all"], d["U_all"])])
        else:
            Z = np.asarray(d["z_all"], float)
        eq, iq, bd = viol_ca(q, Z, ok)
        got = {"eq_inf_max": np.nanmax(eq), "eq_inf_mean": np.nanmean(eq),
               "ineq_max": np.nanmax(iq), "bound_max": np.nanmax(bd)}
        tq = tbl[scen_name[q]]["rows"][solver]["q"]
        for m in ("eq_inf_max", "eq_inf_mean", "ineq_max", "bound_max"):
            tv = tq.get(m); gv = float(got[m])
            diff = abs(gv - tv) if tv is not None else float("nan")
            reld = diff / abs(tv) if tv else diff
            maxdiff = max(maxdiff, reld if np.isfinite(reld) else 0.0)
            print(f"{scen_name[q]:10s} {solver:13s} {m:9s} {gv:16.3e} {tv if tv is not None else float('nan'):16.3e} {reld:10.2e}")
print(f"\nMAX RELATIVE DIFFERENCE across all multi entries: {maxdiff:.3e}")

"""Verify nav+track full-data violation entries with IPOPT's CasADi constraint
model (f_ca dynamics + CasADi obstacle/quaternion/bound algebra), diff vs the
jaxipm-computed table numbers, and cross-check CasADi-vs-jaxipm per element."""
import json, os, sys
import numpy as np
import casadi as ca
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
os.chdir(REPO); sys.path.insert(0, REPO)
import jax; jax.config.update("jax_enable_x64", True)
from tests.quad_casadi_dynamics import f_ca
import json as _j
QP = _j.load(open("tests/quad_params.json"))
import numpy as _np
QP["IB"] = ca.DM(_np.diag([QP["IB00"], QP["IB11"], QP["IB22"]]))
from tests.rebuttal.problems import build_scenario, load_params

def zxu(z, N, nx=13, nu=4):
    x = ca.reshape(z[:N*nx], nx, N); u = ca.reshape(z[N*nx:], nu, N-1); return x, u

def nav_cd(N, Ts):
    import json
    tmp = json.load(open("tests/quad_nav_circle/test_params.json"))
    OB = tmp["obstacles"]; QR = tmp["quad_radius"]
    nz = N*13 + (N-1)*4
    z = ca.SX.sym("z", nz); x0 = ca.SX.sym("x0", 13)
    x, u = zxu(z, N)
    ceq = [x[:,0]-x0]
    for k in range(N-1):
        ceq.append(x[:,k+1] - (x[:,k] + f_ca(x[:,k], u[:,k], QP)*Ts))
    dl = []
    for k in range(N-1):
        m = 1 + k*Ts*0.1
        for o in OB:
            r = o["r"] + QR                       # base radius + quad's own radius (matches jaxipm r__)
            dl.append((x[0,k]-o["xc"])**2 + (x[1,k]-o["yc"])**2 - r**2*m)
    return ca.Function("c",[z,x0],[ca.vertcat(*ceq)]), ca.Function("d",[z],[ca.vertcat(*dl)])

def track_cd(N, Ts):
    from tests.quad_track_avoid.initialization import obstacles
    nz = N*13 + (N-1)*4
    z = ca.SX.sym("z", nz); x0 = ca.SX.sym("x0", 13)
    x, u = zxu(z, N)
    ceq = [x[:,0]-x0]
    for k in range(N-1):
        ceq.append(x[:,k+1] - (x[:,k] + f_ca(x[:,k], u[:,k], QP)*Ts))
    obs = []
    for k in range(N):
        tk = k*Ts
        for o in obstacles:
            xc = o["xc0"]+o["ax"]*np.sin(o["fx"]*tk+o["px"])
            yc = o["yc0"]+o["ay"]*np.sin(o["fy"]*tk+o["py"])
            obs.append((x[0,k]-xc)**2 + (x[1,k]-yc)**2 - o["r"]**2)
    qn = [x[3,k]**2+x[4,k]**2+x[5,k]**2+x[6,k]**2 for k in range(N)]
    return ca.Function("c",[z,x0],[ca.vertcat(*ceq)]), ca.Function("d",[z],[ca.vertcat(*obs+qn)])

def z_from_xu(X, U):
    X, U = np.asarray(X), np.asarray(U)
    if X.ndim == 2: return np.concatenate([X.reshape(-1), U.reshape(-1)])
    return np.concatenate([x.reshape(-1) for x in X]+[u.reshape(-1) for u in U])

p = load_params()
CFG = {
 "nav 90°":  ("nav", 90, nav_cd, "tests/quad_nav_circle/logs", "sector90", 500),
 "nav 180°": ("nav", 180, nav_cd, "tests/quad_nav_circle/logs", "sector180", 500),
 "track v̄=1.0": ("track", 1.0, track_cd, "tests/quad_track_avoid/logs", "v1.0", 250),
 "track v̄=1.5": ("track", 1.5, track_cd, "tests/quad_track_avoid/logs", "v1.5", 250),
 "track v̄=2.0": ("track", 2.0, track_cd, "tests/quad_track_avoid/logs", "v2.0", 250),
}
def solvers(L, tag, N):
    return {
     "IPOPT-default": (f"{L}/casadi_{tag}_pool_tol1e-08_results.npz","xu"),
     "IPOPT-64":      (f"{L}/casadi_cpu_throughput_{tag}_phys_c064_tol1e-08_now_rep1_results.npz","xu"),
     "MadNLP":        (f"{L}/madnlp_{tag}_pool_tol1e-8_results.npz","z"),
     "jaxipm-SL":     (f"{L}/jaxipm_ablation_{tag}_ws_ir0_b{N}_dbg_reld_results.npz","z"),
     "jaxipm-IL":     (f"{L}/jaxipm_ablation_{tag}_hr_ir0_b{N}_dbg_reld_results.npz","z"),
    }
tbl = json.load(open("tests/rebuttal/results/paper_tables.json"))
print(f"{'scenario':12s} {'solver':13s} {'metric':11s} {'CasADi':>13s} {'table':>13s} {'reldiff':>10s}")
maxdiff = 0.0; xchk = 0.0
for scen,(fam,var,cdfn,L,tag,N) in CFG.items():
    sc = build_scenario(fam, var, p, N)
    Ncfg = sc.meta["N"]; Ts = sc.meta["Ts"]
    cfn, dfn = cdfn(Ncfg, Ts)
    all_x0 = np.asarray(sc.extra_npz["all_x0"])
    x_L,x_U,d_L,d_U = [np.asarray(a,float).reshape(-1) for a in sc.bounds]
    for solver,(f,kind) in solvers(L,tag,N).items():
        if not os.path.exists(f):
            print(f"{scen:12s} {solver:13s} MISSING {f}"); continue
        d = np.load(f, allow_pickle=True); ok = np.asarray(d["success"],bool)
        if kind=="xu":
            Z = np.stack([z_from_xu(X,U) for X,U in zip(d["X_all"],d["U_all"])])
        elif "z_all" in d.files and np.asarray(d["z_all"]).size:
            Z = np.asarray(d["z_all"],float)
        else:                                  # MadNLP nav/track: solutions in a separate _z file
            fz = f.replace("_results.npz","_z_results.npz")
            dz = np.load(fz, allow_pickle=True); Z = np.asarray(dz["z_all"],float)
            ok = np.asarray(dz["success"],bool)
        good = ok & np.all(np.isfinite(Z),axis=1)
        idxs = np.asarray(sc.match_batch(Z)).astype(int)
        eq=np.full(len(Z),np.nan); iq=np.full(len(Z),np.nan); bd=np.full(len(Z),np.nan)
        jc = sc.c_batch(Z[good], idxs[good]); jd = sc.d_batch(Z[good])
        jc=np.asarray(jc); jd=np.asarray(jd)
        gi=0
        for r in np.flatnonzero(good):
            zc=Z[r]; c=np.asarray(cfn(zc, all_x0[idxs[r]])).reshape(-1); dv=np.asarray(dfn(zc)).reshape(-1)
            xchk=max(xchk, np.max(np.abs(c-jc[gi])), np.max(np.abs(dv-jd[gi]))); gi+=1
            eq[r]=np.max(np.abs(c)); iq[r]=max(0.0,np.max(np.maximum(d_L-dv,dv-d_U)))
            bd[r]=max(0.0,np.max(np.maximum(x_L-zc,zc-x_U)))
        got={"eq_inf_max":np.nanmax(eq),"eq_inf_mean":np.nanmean(eq),"ineq_max":np.nanmax(iq),"bound_max":np.nanmax(bd)}
        tq=tbl[scen]["rows"][solver]["q"]
        for m in ("eq_inf_max","eq_inf_mean","ineq_max","bound_max"):
            tv=tq.get(m); gv=float(got[m]); rd=abs(gv-tv)/abs(tv) if tv else abs(gv-(tv or 0))
            maxdiff=max(maxdiff, rd if np.isfinite(rd) else 0.0)
            print(f"{scen:12s} {solver:13s} {m:11s} {gv:13.3e} {(tv if tv is not None else float('nan')):13.3e} {rd:10.2e}")
print(f"\nMAX RELATIVE DIFF vs table: {maxdiff:.3e}")
print(f"MAX |CasADi - jaxipm| per constraint element over ALL solutions: {xchk:.3e}")

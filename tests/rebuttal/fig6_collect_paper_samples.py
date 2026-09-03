"""Re-collect the paper Figure-6 samples (June validation_runs archives, isclose metric,
tail_frac 0.10) exactly as plot_validation_state_diffs.collect_samples does, and
additionally record per-sample branch flags from the archived jaxipm/IPOPT state pair
so a branch-type legend can be added.  Writes samples.npz."""
import os, sys, pathlib
os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, "/home/john/code/jaxipm/src/problems/correctness_test")
import numpy as np
import plot_validation_state_diffs as P

def sc(x):
    try: return float(np.asarray(x).reshape(-1)[0])
    except Exception: return np.nan

rows = []
for r in P.list_runs(P.DEFAULT_ROOT):
    iters = P.list_iters(r)
    n_keep = max(1, int(round(0.9 * len(iters))))
    iters = iters[:n_keep]
    prev = None
    for k in iters + [None]:
        cur = None
        if k is not None:
            jx, ip = P.load_pair(r, k)
            cur = (k, jx, ip, int(np.asarray(jx.fl.in_restoration).squeeze()))
        if prev is not None:
            pk, pjx, pip, presto = prev
            exiting = presto == 1 and cur is not None and cur[3] == 0
            recs = P.analyze_diff(pjx, pip, print_diffs=False, exiting_resto=exiting)
            compared = [(P.leaf_metric(d, s, e, "isclose"), path) for path, st, d, s, e in recs if st == "compared"]
            m, p = max(compared) if compared else (0.0, "")
            entering = int(np.asarray(pjx.fl.fallback_activated).squeeze()) == 1
            soc_ran = float(np.max(np.abs(np.asarray(pip.ls.c_soc)))) > 0.0
            cat = ("restoration entry" if entering else "restoration exit" if exiting else
                   "second order correction" if soc_ran else "restoration" if presto else "regular")
            rows.append(dict(run=r.name, iter=pk, dev=m, worst=p, cat=cat, resto=presto, entering=int(entering),
                             exiting=int(exiting), soc_ran=int(soc_ran),
                             jx_n_steps=sc(pjx.ls.n_steps), jx_count_soc=sc(pjx.ls.count_soc), jx_alpha_pr=sc(pjx.ls.alpha_pr),
                             jx_accept=sc(pjx.ls.accept), jx_in_wd=sc(pjx.fl.in_watchdog), jx_wd_trial=sc(pjx.wd.trial_iter),
                             jx_wd_short=sc(pjx.wd.shortened_iter), jx_ts_flag=sc(pjx.fl.tiny_step_flag),
                             jx_ts_last=sc(pjx.fl.tiny_step_last_iter), jx_sfr=sc(pjx.fl.in_soft_resto_phase),
                             jx_free_mu=sc(pjx.fl.free_mu_mode), jx_skip_first=sc(pjx.fl.skip_first_trial),
                             ip_n_steps=sc(pip.ls.n_steps), ip_count_soc=sc(pip.ls.count_soc), ip_alpha_pr=sc(pip.ls.alpha_pr),
                             ip_in_wd=sc(pip.fl.in_watchdog), ip_wd_trial=sc(pip.wd.trial_iter), ip_wd_short=sc(pip.wd.shortened_iter),
                             ip_ts_flag=sc(pip.fl.tiny_step_flag), ip_ts_last=sc(pip.fl.tiny_step_last_iter),
                             ip_fallback=sc(pip.fl.fallback_activated), ip_resto=sc(pip.fl.in_restoration)))
            print(r.name, pk, f"{m:.3e}", cat, rows[-1]["jx_n_steps"], rows[-1]["ip_n_steps"], flush=True)
        prev = cur
keys = list(rows[0].keys())
np.savez(pathlib.Path(__file__).parent / "samples.npz", **{k: np.array([row[k] for row in rows]) for k in keys})
print("saved", len(rows), "samples")

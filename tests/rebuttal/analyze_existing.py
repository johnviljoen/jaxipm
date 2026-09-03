"""What the EXISTING logs already say about E4 (paired final cost), E5
(termination census) and E11 (order-statistics prediction) -- no solver runs.

Sources
  IPOPT  : logs/casadi_cpu_throughput_<tag>_phys_c001_results.npz  (E1 sweep,
           P=1: the whole jaxipm pool solved once by one pinned single-thread
           IPOPT; tol=1e-6, max_iter=500) and the paper's sequential
           logs/casadi_<tag>_results.npz (n = N_RUNS_seq).
  MadNLP : logs/madnlp_<tag>_results.npz (paper run, 2026-06-24).
  jaxipm : logs/jaxipm_<tag>_results.npz (paper drivers). Per-problem iters /
           termination codes are only present for multi-swap (the nav/track
           files were written with DEBUG_MODE off -> iters = terms = -1).
           Ablation outputs (jaxipm_ablation_*_dbg_results.npz) are picked up
           when present and supersede the paper files for E5/E11.

Writes tests/rebuttal/results/existing_{e4_cost,e5_census,e11_orderstats}.md
Run from the repo root:  python -m tests.rebuttal.analyze_existing
"""

import glob
import os

import numpy as np

try:
    from scipy import stats as sps
except Exception:      # pragma: no cover
    sps = None

HERE = os.path.dirname(__file__)
RES_DIR = os.path.join(HERE, "results")
SUFFIX = os.environ.get("CPU_SWEEP_SUFFIX", "")   # "" = tol 1e-6 sweep; "_tol1e-08" = run D rerun

SCENARIOS = [
    ("nav", "tests/quad_nav_circle/logs", [90, 180], lambda v: f"sector{v}",
     lambda v: f"nav {v}°", 500),
    ("track", "tests/quad_track_avoid/logs", [1.0, 1.5, 2.0], lambda v: f"v{v:.1f}",
     lambda v: f"track v̄={v:.1f}", 250),
    ("multi", "tests/quad_multi_swap/logs", [2, 4], lambda v: f"{v}",
     lambda v: f"multi N={v}", 75),
]


def _load(path):
    return np.load(path, allow_pickle=True) if os.path.exists(path) else None


def iter_stats(it, cap=500):
    it = np.asarray(it)
    it = it[it >= 0]
    if it.size == 0:
        return None
    return dict(n=int(it.size), mean=float(it.mean()), median=float(np.median(it)),
                p95=float(np.percentile(it, 95)), max=int(it.max()),
                at_cap=float((it >= cap).mean()))


def emax_ratio(K, N):
    """E[max of N iid draws from the empirical distribution of K] / E[K]."""
    K = np.asarray(K, dtype=float)
    K = K[K >= 0]
    vals, counts = np.unique(K, return_counts=True)
    F = np.cumsum(counts) / K.size
    Fm1 = np.concatenate([[0.0], F[:-1]])
    emax = float(np.sum(vals * (F**N - Fm1**N)))
    return emax / K.mean(), emax


def pair_nav(J, C):
    pool = J["all_angles"]
    ji = np.abs(J["angles"][:, None] - pool[None, :]).argmin(1)
    ci = np.abs(C["angles"][:, None] - pool[None, :]).argmin(1)
    co = np.full(pool.size, np.nan); co[ci] = C["obj_vals"]
    # jaxipm may solve an angle more than once: average duplicates.
    jo = np.full(pool.size, np.nan)
    for k in np.unique(ji):
        jo[k] = J["obj_vals"][ji == k].mean()
    m = ~np.isnan(jo) & ~np.isnan(co)
    return jo[m], co[m], int(m.sum()), len(np.unique(ji))


def pair_track(J, C):
    pool = C["all_x0"]
    ji = np.linalg.norm(J["starts"][:, None, :] - pool[None, :, :], axis=2).argmin(1)
    ci = np.linalg.norm(C["starts"][:, None, :] - pool[None, :, :], axis=2).argmin(1)
    co = np.full(len(pool), np.nan); co[ci] = C["obj_vals"]
    jo = np.full(len(pool), np.nan)
    for k in np.unique(ji):
        jo[k] = J["obj_vals"][ji == k].mean()
    m = ~np.isnan(jo) & ~np.isnan(co)
    return jo[m], co[m], int(m.sum()), len(np.unique(ji))


def paired_report(jo, co, label):
    d = jo - co
    rel = d / np.abs(co)
    out = [f"\n### {label}: paired final objective, jaxipm − IPOPT (n = {d.size})\n"]
    out.append(f"- mean objective: jaxipm {jo.mean():.3f}, IPOPT {co.mean():.3f}; "
               f"median jaxipm {np.median(jo):.3f}, IPOPT {np.median(co):.3f}")
    out.append(f"- mean paired difference {d.mean():+.4f} (std {d.std(ddof=1):.4f}); "
               f"median {np.median(d):+.5f}; mean relative gap {100*rel.mean():+.3f}% "
               f"(median {100*np.median(rel):+.4f}%)")
    rng = np.random.default_rng(0)
    boots = np.array([rng.choice(rel, rel.size).mean() for _ in range(4000)])
    out.append(f"- 95% bootstrap CI of the mean relative gap: "
               f"[{100*np.percentile(boots, 2.5):+.3f}%, {100*np.percentile(boots, 97.5):+.3f}%]")
    tol = 1e-4
    out.append(f"- jaxipm lower by >{tol:g} rel: {100*(rel < -tol).mean():.1f}%; "
               f"equal within {tol:g}: {100*(np.abs(rel) <= tol).mean():.1f}%; "
               f"jaxipm higher by >{tol:g}: {100*(rel > tol).mean():.1f}%")
    q = np.percentile(rel, [0, 1, 5, 50, 95, 99, 100])
    out.append("- relative-gap quantiles (0/1/5/50/95/99/100 %): " +
               ", ".join(f"{100*x:+.3f}%" for x in q))
    big = np.abs(rel) > 0.01
    out.append(f"- instances with |rel gap| > 1% (different local optimum or failed "
               f"solve): {int(big.sum())} ({100*big.mean():.2f}%)")
    if sps is not None:
        try:
            w = sps.wilcoxon(d, zero_method="wilcox")
            t = sps.ttest_rel(jo, co)
            out.append(f"- Wilcoxon signed-rank p = {w.pvalue:.3g}; paired t-test "
                       f"p = {t.pvalue:.3g} (t = {t.statistic:+.2f})")
            if big.any():
                w2 = sps.wilcoxon(d[~big], zero_method="wilcox")
                out.append(f"- excluding |rel gap| > 1%: mean rel gap "
                           f"{100*rel[~big].mean():+.4f}%, Wilcoxon p = {w2.pvalue:.3g}")
        except Exception as e:  # pragma: no cover
            out.append(f"- (scipy tests failed: {e})")
    return out


def main():
    os.makedirs(RES_DIR, exist_ok=True)
    e4, e5, e11 = [], [], []
    e4.append("# E4 (existing data) — paired final-cost comparison, jaxipm vs IPOPT\n"
              "Pairs are formed on the SAME problem instance (nav: start angle; "
              "track: start state x0 → pool index). IPOPT values come from the E1 "
              "P=1 sweep (tol 1e-6); jaxipm values from the paper-driver npz "
              "(tol 1e-8, ir_nsteps 0). Constraint-violation / KKT-residual "
              "columns need the full z (not saved by the paper drivers) and come "
              "from the ablation runs.\n")
    e5.append("# E5 (existing data) — termination census\n"
              "`at_cap` = fraction of solves that used max_iter = 500 iterations. "
              "IPOPT/MadNLP `success` = solver reported success (CasADi raises on "
              "any non-success return status, so 'Solved To Acceptable Level' "
              "counts as a failure here). jaxipm per-problem codes exist only "
              "where DEBUG_MODE was on.\n")
    e5.append("| scenario | solver | source | n | success | iters mean | median | p95 | max | at_cap |\n"
              "|---|---|---|---|---|---|---|---|---|---|")
    e11.append("# E11 (existing data) — order-statistics prediction of the ILB gain\n"
               "Predicted (b)→(c) throughput ratio = E[max_{i≤N} K_i] / E[K] with N = "
               "paper batch size, using the EMPIRICAL per-problem iteration "
               "distribution K (exact from the empirical CDF, i.i.d. draws with "
               "replacement). Rows marked *IPOPT proxy* use IPOPT's iteration "
               "counts on the same pool because the paper jaxipm files lack "
               "per-problem counts; jaxipm rows appear once ablation *_dbg runs exist.\n")
    e11.append("| scenario | K source | n | E[K] | E[max_N K] | N | predicted ratio |\n"
               "|---|---|---|---|---|---|---|")

    for name, logs, variants, tagf, labf, N_batch in SCENARIOS:
        for v in variants:
            tag = tagf(v)
            C1 = _load(os.path.join(logs, f"casadi_cpu_throughput_{tag}_phys_c001{SUFFIX}_results.npz"))
            Cs = _load(os.path.join(logs, f"casadi_{tag}_results.npz"))
            M = _load(os.path.join(logs, f"madnlp_{tag}_results.npz"))
            J = _load(os.path.join(logs, f"jaxipm_{tag}_results.npz"))
            abl = sorted(glob.glob(os.path.join(logs, f"jaxipm_ablation_{tag}_*_dbg*_results.npz")))

            # ---- E5 ----
            for solver, src, Z in (("IPOPT", f"D P=1 sweep{SUFFIX or ' (tol 1e-6)'}", C1),
                                   ("IPOPT", "paper sequential", Cs),
                                   ("MadNLP", "paper", M),
                                   ("jaxipm", "paper driver", J)):
                if Z is None:
                    continue
                st = iter_stats(Z["iters"])
                succ = float(Z["success"].mean()) if "success" in Z.files else float("nan")
                n = int(Z["success"].size) if "success" in Z.files else int(Z["iters"].size)
                if st is None:
                    e5.append(f"| {labf(v)} | {solver} | {src} | {n} | {succ:.4f} | – | – | – | – | – |")
                else:
                    e5.append(f"| {labf(v)} | {solver} | {src} | {n} | {succ:.4f} | "
                              f"{st['mean']:.1f} | {st['median']:.0f} | {st['p95']:.0f} | "
                              f"{st['max']} | {st['at_cap']:.4f} |")
            for path in abl:
                Z = np.load(path, allow_pickle=True)
                st = iter_stats(Z["iters"])
                th = Z["term_hist"] if "term_hist" in Z.files else None
                src = os.path.basename(path).replace(f"jaxipm_ablation_{tag}_", "").replace("_results.npz", "")
                extra = ""
                if th is not None and th.sum() > 0:
                    extra = (f" census(conv/max_iter/tiny/resto_fail/acceptable)="
                             f"{th[1]}/{th[2]}/{th[3]}/{th[4]}/{th[5]}")
                if st:
                    e5.append(f"| {labf(v)} | jaxipm | ablation {src}{extra} | {int(Z['success'].size)} | "
                              f"{float(Z['success'].mean()):.4f} | {st['mean']:.1f} | "
                              f"{st['median']:.0f} | {st['p95']:.0f} | {st['max']} | {st['at_cap']:.4f} |")

            # ---- E11 ----
            if C1 is not None:
                r, em = emax_ratio(C1["iters"], N_batch)
                K = C1["iters"][C1["iters"] >= 0]
                e11.append(f"| {labf(v)} | IPOPT proxy (E1 P=1) | {K.size} | {K.mean():.1f} | "
                           f"{em:.1f} | {N_batch} | {r:.2f} |")
            if J is not None and (J["iters"] >= 0).any():
                r, em = emax_ratio(J["iters"], N_batch)
                K = J["iters"][J["iters"] >= 0]
                e11.append(f"| {labf(v)} | jaxipm paper driver | {K.size} | {K.mean():.1f} | "
                           f"{em:.1f} | {N_batch} | {r:.2f} |")
            for path in abl:
                Z = np.load(path, allow_pickle=True)
                if (Z["iters"] >= 0).any():
                    r, em = emax_ratio(Z["iters"], int(Z["N_BATCH"][0]))
                    K = Z["iters"][Z["iters"] >= 0]
                    src = os.path.basename(path).replace(f"jaxipm_ablation_{tag}_", "").replace("_results.npz", "")
                    nf = Z["n_fused_iters"] if "n_fused_iters" in Z.files else None
                    meas = ""
                    if nf is not None and (nf >= 0).all() and "ws" in src:
                        meas = f" (measured per-batch max: mean {nf.mean():.1f})"
                    e11.append(f"| {labf(v)} | jaxipm ablation {src}{meas} | {K.size} | {K.mean():.1f} | "
                               f"{em:.1f} | {int(Z['N_BATCH'][0])} | {r:.2f} |")

            # ---- E4 ----
            if J is not None and C1 is not None:
                if name == "nav":
                    jo, co, n, uniq = pair_nav(J, C1)
                    e4 += paired_report(jo, co, f"{labf(v)}  [jaxipm solved {uniq}/{len(J['all_angles'])} distinct pool instances]")
                elif name == "track":
                    jo, co, n, uniq = pair_track(J, C1)
                    e4 += paired_report(jo, co, f"{labf(v)}  [jaxipm solved {uniq}/{len(C1['all_x0'])} distinct pool instances]")
                else:
                    jo, co = J["obj_vals"], C1["obj_vals"]
                    e4.append(f"\n### {labf(v)}: one fixed problem solved {jo.size}× (jaxipm) / "
                              f"{co.size}× (IPOPT)\n")
                    e4.append(f"- IPOPT objective {co.mean():.4f} (std {co.std():.2e}); jaxipm "
                              f"median {np.median(jo):.4f}, mean {jo.mean():.4f}, min {jo.min():.4f}, "
                              f"max {jo.max():.4f}; jaxipm within 1e-4 rel of IPOPT: "
                              f"{100*(np.abs(jo/co.mean()-1) < 1e-4).mean():.1f}%; "
                              f"jaxipm success flags {int(J['success'].sum())}/{jo.size}")
            for path in abl:
                Z = np.load(path, allow_pickle=True)
                if C1 is None:
                    continue
                src = os.path.basename(path).replace(f"jaxipm_ablation_{tag}_", "").replace("_results.npz", "")
                if name == "multi":
                    jo, co = Z["obj_vals"], C1["obj_vals"]
                    e4.append(f"\n### {labf(v)} ablation {src}: IPOPT {co.mean():.4f}; jaxipm median "
                              f"{np.median(jo):.4f} mean {jo.mean():.4f} min {jo.min():.4f} max {jo.max():.4f}; "
                              f"success {int(Z['success'].sum())}/{jo.size}")
                    continue
                pool_idx = Z["pool_idx"]
                if name == "nav":
                    ci = np.abs(C1["angles"][:, None] - Z["all_angles"][None, :]).argmin(1)
                else:
                    ci = np.linalg.norm(C1["starts"][:, None, :] - Z["all_x0"][None, :, :], axis=2).argmin(1)
                co = np.full(len(Z["all_x0"]), np.nan); co[ci] = C1["obj_vals"]
                jo = np.full(len(Z["all_x0"]), np.nan)
                for k in np.unique(pool_idx):
                    jo[k] = Z["obj_vals"][pool_idx == k].mean()
                m = ~np.isnan(jo) & ~np.isnan(co)
                e4 += paired_report(jo[m], co[m], f"{labf(v)} ablation {src}  "
                                    f"[{len(np.unique(pool_idx))} distinct instances]")

    sfx = SUFFIX.replace("_", "-")
    for fname, lines in ((f"existing_e4_cost{sfx}.md", e4), (f"existing_e5_census{sfx}.md", e5),
                         (f"existing_e11_orderstats{sfx}.md", e11)):
        with open(os.path.join(RES_DIR, fname), "w") as fh:
            fh.write("\n".join(lines) + "\n")
        print("\n".join(lines))
        print()


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Generate the three paper throughput tables (nav / track / multi) straight from the
result npz files -- no hand transcription -- and diff them cell by cell against the
rows currently in the paper (results/paper_tables_current.tex).

Every row: throughput = SUCCESSFUL solves / time to the last successful solve, mean +- sd over
5 repeats (IPOPT-64: per-solve end stamps; jaxipm batches: full call wall, i.e. conservative); success = fraction of the POOL
solved to tolerance; cost = mean +- sd of the final objective over successful solves;
||c||inf = max equality-constraint violation at the returned point, evaluated with
jaxipm's own constraint function for every solver (needs a GPU; --no-viol skips it);
iters = mean / max over successful solves; instances = solves/s / IPOPT-default's solves/s
(equiv. single-instance count, with IPOPT-default as the one-instance baseline; IPOPT-64's
value comes out below 64 to the extent the 64-process run scales sub-linearly off one instance).

Sources (tag = sector90 | v1.0 | 2 ..., N = paper batch, V = baseline variant suffix):
  IPOPT-default  thr: casadi_<tag><V>_tol1e-08_rep{r}          (N_RUNS / total_time, paper N_RUNS_seq)
                 qual: casadi_<tag><V>_pool_tol1e-08            (full pool pass, X_all/U_all)
  IPOPT-64       thr: casadi_cpu_throughput_<tag>_phys_c064<V>_tol1e-08_now_rep{r}  (qual from rep1)
  MadNLP         thr: madnlp_<tag><V>_pool_tol1e-8              (walls from stop stamps, 5 in-process reps)
                 qual: same file if it holds z_all, else madnlp_<tag>_pool_tol1e-8_z
  jaxipm-SL      thr: jaxipm_ablation_<tag>_ws_ir0_b<N>_fresh_rep{k}   qual: ..._ws_ir0_b<N>_dbg
  jaxipm-IL      nav/multi: thr jaxipm_variance_<tag>_fresh_rep{k} (run A), qual ..._hr_ir0_b<N>_dbg
                 track:     thr jaxipm_ablation_<tag>_hr_ir0_b<N>_rel_fresh_rep{k} (nav-style stride,
                            2026-08-29 rerun), qual ..._hr_ir0_b<N>_dbg_rel
V = "_nobounds" for nav: the jaxipm nav model (quadcopter_nav) has +-inf variable bounds,
so the nav baselines are recomputed without state/motor bounds (2026-08-29).

Usage (repo root):  python -m tests.rebuttal.make_paper_tables --gpu 6 [--no-viol]
Writes results/paper_tables.tex / .md / .json and prints the cell diff.
"""
import argparse, glob, json, math, os, re, sys

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RES = os.path.join(REPO, "tests", "rebuttal", "results")

# (family, variant, name, logs dir, tag, N_batch, N_RUNS, LaTeX group header)
SC = [("nav", 90, "nav 90°", "tests/quad_nav_circle/logs", "sector90", 500, 2000, r"$90^\circ$ sector"),
      ("nav", 180, "nav 180°", "tests/quad_nav_circle/logs", "sector180", 500, 2000, r"$180^\circ$ sector"),
      ("track", 1.0, "track v̄=1.0", "tests/quad_track_avoid/logs", "v1.0", 250, 1000, r"$\bar{v} = 1.0$ m/s"),
      ("track", 1.5, "track v̄=1.5", "tests/quad_track_avoid/logs", "v1.5", 250, 1000, r"$\bar{v} = 1.5$ m/s"),
      ("track", 2.0, "track v̄=2.0", "tests/quad_track_avoid/logs", "v2.0", 250, 1000, r"$\bar{v} = 2.0$ m/s"),
      ("multi", 2, "multi N=2", "tests/quad_multi_swap/logs", "2", 75, 75, "2 quadrotors"),
      ("multi", 4, "multi N=4", "tests/quad_multi_swap/logs", "4", 75, 75, "4 quadrotors")]
TABLES = [("nav", "tab:nav-throughput"), ("track", "tab:track-avoid-throughput"),
          ("multi", "tab:multi-swap-throughput")]
ROWS = ["IPOPT-default", "IPOPT-64", "MadNLP", "jaxipm-SL", "jaxipm-IL"]
# nav variant: "bounded" (2026-08-30: jaxipm nav model fixed, jaxipm reruns `relb`, ORIGINAL bounded
# baselines) or "nobounds" (2026-08-29: jaxipm unbounded run A, baselines recomputed without bounds).
NAV_VARIANT = "bounded"
BASELINE_VARIANT = {"nav": "", "track": "", "multi": "_rot"}   # _rot = varied-IC rotation pool (2026-08-31)
IL_SOURCE = {"nav": "reld", "track": "reld", "multi": "rot300"}    # runA | stride (rel) | relb
SL_SOURCE = {"nav": "reld", "track": "reld", "multi": "rot300"}        # old | relb


# ── formatting (paper conventions) ───────────────────────────────────────────
def fmt_pm(m, s, dp):
    return f"{m:.{dp}f}\\,$\\pm$\\,{s:.{dp}f}"


def fmt_thr(m, s):
    return fmt_pm(m, s, 3 if m < 1 else (2 if m < 100 else 1))


def fmt_instances(v):
    if v is None:
        return "–"
    return f"{v:.2f}" if v < 10 else f"{v:.1f}"


def fmt_sci(v):
    if v is None or not np.isfinite(v):
        return "–"
    if v == 0:
        return "$0$"
    e = int(math.floor(math.log10(abs(v)))); mant = v / 10 ** e
    if round(mant, 1) >= 10:
        mant /= 10; e += 1
    return f"${mant:.1f}\\!\\times\\!10^{{{e}}}$"


def fmt_succ(k, n):
    if k >= n:
        return "100"
    return f"{100.0 * k / n:.2f}".rstrip("0").rstrip(".")


def bold(s):
    return "{\\boldmath" + s + "}" if s.startswith("$") else "\\textbf{" + s + "}"


# ── loaders ──────────────────────────────────────────────────────────────────
def npz(path):
    return np.load(path, allow_pickle=True) if path and os.path.exists(path) else None


def z_from_xu(X, U):
    X = np.asarray(X); U = np.asarray(U)
    if X.ndim == 2:
        return np.concatenate([X.reshape(-1), U.reshape(-1)])
    return np.concatenate([x.reshape(-1) for x in X] + [u.reshape(-1) for u in U])


def quality(d, N_pool, sc, viol_fn, zkind, rel):
    """cost / success-vs-pool / iters / ||c||inf from one result file."""
    obj = np.asarray(d["obj_vals"], float); ok = np.asarray(d["success"], bool) & np.isfinite(obj)
    it = np.asarray(d["iters"]).ravel()
    n_ok = int(ok.sum())
    # Pool denominator = the file's own requested-solve count when recorded
    # (fix 2026-08-31: multi's SC N_pool=75 under-counted the 300-solve
    # campaign, hiding WS cap-hitters; nav/track files carry the same key and
    # are unchanged by this).
    n_pool = int(d["N_RUNS"][0]) if "N_RUNS" in getattr(d, "files", []) else N_pool
    n_pool = max(n_pool, obj.size) if zkind == "xu64" else n_pool
    r = dict(file=rel, n_rows=int(obj.size), n_success=n_ok, n_pool=int(n_pool),
             cost_mean=float(obj[ok].mean()), cost_sd=float(obj[ok].std(ddof=1)) if n_ok > 1 else 0.0,
             iters_mean=float(it[ok].mean()), iters_max=int(it[ok].max()), eq_inf_max=None)
    if viol_fn is not None and sc is not None:
        if zkind in ("xu", "xu64") and "U_all" in d.files:
            zs = [z_from_xu(X, U) for X, U in zip(d["X_all"], d["U_all"])]
        elif zkind == "z" and "z_all" in d.files and np.asarray(d["z_all"]).size:
            zs = [np.asarray(z) for z in d["z_all"]]
        else:
            zs = None
        if zs is not None:
            eq, ineq, bnd = viol_fn(sc, zs, ok)
            r.update(eq_inf_max=float(np.nanmax(eq)), eq_inf_mean=float(np.nanmean(eq)), eq_inf_median=float(np.nanmedian(eq)),
                     ineq_max=float(np.nanmax(ineq)), bound_max=float(np.nanmax(bnd)))
        else:
            r["note"] = "no solution vectors in file: ||c||inf not evaluated"
    return r


def succ_per_s(d):
    """Throughput = successful solves / time at which the LAST SUCCESSFUL solve finished.
    Mirrors jaxipm-IL's protocol (the clock stops when the successes are in; a diverging
    straggler is abandoned). Uses per-solve end stamps (t_end) when the file has them
    (process-parallel IPOPT runs); otherwise successes / total wall (sequential runs,
    where the two coincide unless a failure is the last solve)."""
    ok = np.asarray(d["success"], bool)
    if "t_end" in d.files:
        return float(ok.sum() / np.asarray(d["t_end"])[ok].max())
    return float(ok.sum() / float(d["total_time"][0]))


def succ_per_total_wall(d):
    return float(np.asarray(d["success"], bool).sum() / float(d["total_time"][0]))


def thr_stats(vals):
    vals = [float(v) for v in vals]
    return (np.mean(vals), np.std(vals, ddof=1) if len(vals) > 1 else 0.0) if vals else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gpu", default="6")
    ap.add_argument("--no-viol", action="store_true")
    ap.add_argument("--out", default="paper_tables")
    ap.add_argument("--nav-variant", choices=["bounded", "nobounds"], default="bounded")
    args = ap.parse_args()
    global NAV_VARIANT
    NAV_VARIANT = args.nav_variant
    if NAV_VARIANT == "nobounds":
        BASELINE_VARIANT["nav"] = "_nobounds"; IL_SOURCE["nav"] = "runA"; SL_SOURCE["nav"] = "old"
    os.chdir(REPO); sys.path.insert(0, REPO)
    viol_fn = build = p = None
    if not args.no_viol:
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", args.gpu)
        os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
        import jax
        jax.config.update("jax_enable_x64", True)
        from tests.rebuttal.problems import build_scenario, load_params
        from tests.rebuttal.analyze_primary_table import violations
        viol_fn, build, p = violations, build_scenario, load_params()

    table, notes = {}, []
    for fam, var, scen, logs, tag, N, N_RUNS, header in SC:
        L = os.path.join(REPO, logs); V = BASELINE_VARIANT[fam]
        rel = lambda f: os.path.relpath(f, REPO)
        sc = build(fam, var, p, N) if build else None
        rows = {}

        def add(row, thr_vals, thr_files, qfile, zkind):
            q = npz(qfile)
            r = dict(thr=thr_stats(thr_vals), n_reps=len(thr_vals), thr_files=[rel(f) for f in thr_files])
            if q is not None:
                r["q"] = quality(q, N_RUNS, sc, viol_fn, zkind, rel(qfile))
            else:
                r["q"] = None; notes.append(f"{scen} {row}: quality file missing ({rel(qfile) if qfile else '-'})")
            if len(thr_vals) != 5:
                notes.append(f"{scen} {row}: {len(thr_vals)} throughput repeats (expected 5)")
            if r["q"] and r["q"]["n_success"] < r["q"]["n_pool"]:
                r["q"]["n_fail"] = r["q"]["n_pool"] - r["q"]["n_success"]
                notes.append(f"{scen} {row}: {r['q']['n_fail']} of {r['q']['n_pool']} pool instances NOT solved "
                             f"(excluded from cost/||c||/iters and from the throughput count; clock stops at the last "
                             f"successful solve for process-parallel IPOPT, full call wall for jaxipm batches)")
            rows[row] = r

        # IPOPT-default: 5 timed repeats on N_RUNS_seq + full pool pass for quality
        fs = sorted(glob.glob(os.path.join(L, f"casadi_{tag}{V}_tol1e-08_rep*_results.npz")))
        add("IPOPT-default", [succ_per_s(np.load(f, allow_pickle=True)) for f in fs], fs,
            os.path.join(L, f"casadi_{tag}{V}_pool_tol1e-08_results.npz"), "xu")
        # IPOPT-64
        fs = sorted(glob.glob(os.path.join(L, f"casadi_cpu_throughput_{tag}_phys_c064{V}_tol1e-08_now_rep*_results.npz")))
        add("IPOPT-64", [succ_per_s(np.load(f, allow_pickle=True)) for f in fs], fs, fs[0] if fs else None, "xu64")
        rows["IPOPT-64"]["thr_total_wall"] = thr_stats([succ_per_total_wall(np.load(f, allow_pickle=True)) for f in fs])
        # MadNLP: one file, 5 in-process repeats; walls from stop stamps (rep_throughput in the file is cumulative)
        f = os.path.join(L, f"madnlp_{tag}{V}_pool_tol1e-8_results.npz"); d = npz(f); vals = []
        if d is not None:
            st, sp = np.asarray(d["rep_start_epoch"]), np.asarray(d["rep_stop_epoch"])
            walls = np.diff(np.concatenate([[st[0]], sp])); vals = (int(np.asarray(d["success"], bool).sum()) / walls).tolist()
        fz = f if (d is not None and "z_all" in d.files and np.asarray(d["z_all"]).size) else os.path.join(L, f"madnlp_{tag}_pool_tol1e-8_z_results.npz")
        add("MadNLP", vals, [f] if d is not None else [], fz, "z")
        if d is not None and fz != f and rows["MadNLP"]["q"] is not None:   # iters from the timed run (as in the paper), z-run only for ||c||inf
            it = np.asarray(d["iters"]).ravel(); ok = np.asarray(d["success"], bool)
            rows["MadNLP"]["q"].update(iters_mean=float(it[ok].mean()), iters_max=int(it[ok].max()), iters_file=rel(f))
        # jaxipm-SL
        sfx = "" if SL_SOURCE[fam] == "old" else f"{SL_SOURCE[fam]}_"
        fs = sorted(glob.glob(os.path.join(L, f"jaxipm_ablation_{tag}_ws_ir0_b{N}_{sfx}fresh_rep*_results.npz")))
        add("jaxipm-SL", [float(np.load(f)["rep_throughput"][-1]) for f in fs], fs,
            os.path.join(L, f"jaxipm_ablation_{tag}_ws_ir0_b{N}_dbg{'_' + SL_SOURCE[fam] if sfx else ''}_results.npz"), "z")
        # jaxipm-IL
        if IL_SOURCE[fam] == "runA":
            fs = sorted(glob.glob(os.path.join(L, f"jaxipm_variance_{tag}_fresh_rep*_results.npz")))
            qf = os.path.join(L, f"jaxipm_ablation_{tag}_hr_ir0_b{N}_dbg_results.npz")
        else:
            tagsfx = {"stride": "rel", "relb": "relb", "relc": "relc", "reld": "reld", "relcp375": "relcp375", "relcp300": "relcp300", "rot300": "rot300"}[IL_SOURCE[fam]]
            fs = sorted(glob.glob(os.path.join(L, f"jaxipm_ablation_{tag}_hr_ir0_b{N}_{tagsfx}_fresh_rep*_results.npz")))
            qf = os.path.join(L, f"jaxipm_ablation_{tag}_hr_ir0_b{N}_dbg_{tagsfx}_results.npz")
        add("jaxipm-IL", [float(np.load(f)["rep_throughput"][-1]) for f in fs], fs, qf, "z")

        if viol_fn is None:   # --no-viol: reuse ||c||inf from the previous GPU pass for rows whose data is unchanged
            try:
                pq = json.load(open(os.path.join(RES, "primary_table_quality.json")))[scen]
                prefix = {"IPOPT-default": "IPOPT default", "IPOPT-64": "IPOPT P=64", "MadNLP": "MadNLP",
                          "jaxipm-SL": "jaxipm-WS", "jaxipm-IL": "jaxipm-IL census"}
                for row, r in rows.items():
                    unchanged = not ((V and row in ("IPOPT-default", "IPOPT-64", "MadNLP")) or (row == "jaxipm-IL" and IL_SOURCE[fam] != "runA"))
                    if unchanged and r["q"] is not None:
                        for k, v in pq.items():
                            if k.startswith(prefix[row]) and v.get("eq_inf_max") is not None:
                                r["q"]["eq_inf_max"] = float(v["eq_inf_max"]); r["q"]["eq_inf_source"] = "primary_table_quality.json"; break
            except Exception as e:
                notes.append(f"{scen}: could not reuse previous ||c||inf ({e})")
        baseline = rows["IPOPT-default"]["thr"][0] if rows["IPOPT-default"]["thr"] else None
        for row, r in rows.items():
            r["instances"] = r["thr"][0] / baseline if (r["thr"] and baseline) else None
        table[scen] = dict(rows=rows, header=header, fam=fam, baseline_variant=V)

    # ── cells / LaTeX ────────────────────────────────────────────────────────
    def cells(r):
        q = r.get("q")
        return [fmt_thr(*r["thr"]) if r["thr"] else "–", fmt_instances(r.get("instances")),
                fmt_succ(q["n_success"], q["n_pool"]) if q else "–",
                fmt_pm(q["cost_mean"], q["cost_sd"], 1) if q else "–",
                fmt_sci(q["eq_inf_max"]) if q and q.get("eq_inf_max") is not None else "–",
                f"{q['iters_mean']:.1f} / {q['iters_max']}" if q else "–"]

    captions = {
        "nav": (r"Navigation. Throughput is the mean and standard deviation over five repeated runs; "
                r"instances is each solver's throughput as a multiple of one IPOPT-default instance's throughput; "
                r"succ.\ is the percentage of the pool solved to tolerance; cost is the mean and standard deviation "
                r"of the final objective over the successfully solved pool instances; $\lVert c \rVert_\infty$ is the "
                r"largest equality-constraint violation at the returned point over the pool, evaluated with the same "
                r"constraint function for every solver; iters is the mean and maximum iteration count. All solvers "
                r"draw from the same pool of parameterizations and initial conditions and start from the same cold "
                r"initial guess at tol $10^{-8}$."),
        "track": (r"Per-solve throughput and solution quality on the quadrotor reference-tracking problem with three "
                  r"time-varying obstacles. We control difficulty by varying the average velocity $\bar{v}$ of the "
                  r"reference trajectory. Columns as in Table~\ref{tab:nav-throughput}."),
        "multi": (r"Per-solve throughput and solution quality on the multi-quadrotor consensus problem, where every "
                  r"quadrotor maintains a minimum separation from every other in the fleet. Columns as in "
                  r"Table~\ref{tab:nav-throughput}."),
    }
    tex, md = [], ["# Paper tables (generated by tests/rebuttal/make_paper_tables.py)", ""]
    for fam, label in TABLES:
        tex += [r"\begin{table}[t]", r"\centering", r"\setlength{\tabcolsep}{2.5pt}", r"{\scriptsize",
                r"\begin{tabular}{@{}l r r r r r r@{}}", r"\toprule",
                r"Solver & solves/s & instances & succ.\ [\%] & cost & $\lVert c \rVert_\infty$ & iters mean/max \\", r"\midrule"]
        md += [f"## {fam}", "", "| scenario | solver | solves/s | instances | succ. [%] | cost | ‖c‖∞ | iters mean/max |", "|---|---|---|---|---|---|---|---|"]
        groups = [s for s in SC if s[0] == fam]
        for gi, (_, var, scen, logs, tag, N, N_RUNS, header) in enumerate(groups):
            tex.append(r"\multicolumn{7}{@{}l}{\textit{" + header + r"}} \\")
            for row in ROWS:
                c = cells(table[scen]["rows"][row])
                md.append(f"| {scen} | {row} | " + " | ".join(x.replace("\\,$\\pm$\\,", " ± ") for x in c) + " |")
                if row.startswith("jaxipm"):
                    tex.append(bold(row) + " & " + " & ".join(bold(x) if x != "–" else x for x in c) + r" \\")
                else:
                    tex.append(row + " & " + " & ".join(c) + r" \\")
            tex.append(r"\midrule" if gi < len(groups) - 1 else r"\bottomrule")
        tex += [r"\end{tabular}}", r"\caption{" + captions[fam] + "}", r"\label{" + label + "}", r"\end{table}", ""]
        md.append("")
    os.makedirs(RES, exist_ok=True)
    open(os.path.join(RES, args.out + ".tex"), "w").write("\n".join(tex))
    open(os.path.join(RES, args.out + ".md"), "w").write("\n".join(md) + "\n" + "".join("- NOTE: " + n + "\n" for n in notes))
    json.dump(table, open(os.path.join(RES, args.out + ".json"), "w"), indent=1, default=str)
    print("\n".join(md))
    for n in notes:
        print("NOTE:", n)

    # ── speedups (jaxipm / baseline) ────────────────────────────────────────
    print("\n## Speedups (throughput ratio, new numbers)")
    print("| scenario | IL/IPOPT-def | IL/IPOPT-64 | IL/MadNLP | SL/IPOPT-def | SL/IPOPT-64 | SL/MadNLP |")
    print("|---|---|---|---|---|---|---|")
    for scen, t in table.items():
        R = t["rows"]; f = lambda a, b: (f"{R[a]['thr'][0] / R[b]['thr'][0]:.1f}x" if R[a]["thr"] and R[b]["thr"] else "–")
        print(f"| {scen} | {f('jaxipm-IL','IPOPT-default')} | {f('jaxipm-IL','IPOPT-64')} | {f('jaxipm-IL','MadNLP')} | "
              f"{f('jaxipm-SL','IPOPT-default')} | {f('jaxipm-SL','IPOPT-64')} | {f('jaxipm-SL','MadNLP')} |")

    # ── diff against the current paper rows ──────────────────────────────────
    cur = os.path.join(RES, "paper_tables_current.tex")
    if os.path.exists(cur):
        def strip(s):
            return re.sub(r"\\textbf\{|\{\\boldmath", "", s).replace("{", "").replace("}", "").strip()
        old, order, fam, gi, ri = {}, [], None, 0, 0
        for line in open(cur):
            line = line.strip()
            if line.startswith("%"):
                fam, gi, ri = line[1:].strip(), 0, 0; continue
            if not line or "&" not in line:
                continue
            parts = [strip(x.replace(r"\\", "")) for x in line.split("&")]
            scen = [s for s in SC if s[0] == fam][gi][2]
            old[(scen, ROWS[ri])] = parts[1:]; order.append((scen, ROWS[ri])); ri += 1
            if ri == len(ROWS):
                ri, gi = 0, gi + 1
        names = ["solves/s", "instances", "cost", "‖c‖∞", "iters"]
        print("\n## Cell diff vs paper_tables_current.tex (old -> new); succ. column is new and not diffed")
        nd = 0
        for key in order:
            new = cells(table[key[0]]["rows"][key[1]]); newc = [new[0], new[1], new[3], new[4], new[5]]
            for nm, o, n in zip(names, old[key], newc):
                if strip(o) != strip(n):
                    nd += 1; print(f"  {key[0]:<13} {key[1]:<14} {nm:<9} {strip(o):>22}  ->  {strip(n)}")
        print(f"  {nd} cell(s) differ")


if __name__ == "__main__":
    main()

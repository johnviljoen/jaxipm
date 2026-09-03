"""Runs F / G / H / I1 from the instrumented jaxipm ablation files.

Reads every tests/*/logs/jaxipm_ablation_*_dbg*_results.npz (DEBUG runs of
tests/rebuttal/jaxipm_ablation.py) and produces

  results/census_F_branches.md   branch activation: fraction of slot-iterations
                                 and fraction of SOLVES in which each branch
                                 fires; distinct branches active per fused
                                 iteration (mean / quantiles / max)
  results/census_G_termination.md termination census per config (codes 1/5
                                 among collected, hidden 2/3/4 from term_hist,
                                 iteration statistics, at-cap fraction)
  results/census_H_kkt.md        KKT error components at termination
                                 (overall / dual / constr / compl, scaled as in
                                 the convergence test) -- quantiles
  results/census_I1_residual.md  per-iteration relative KKT step residual
                                 ||K d - r|| / ||r|| -- quantiles, and its
                                 dependence on iteration index

Branch codes (IterateFlags.branch_id + solver.py DEBUG log):
  0 first trial step accepted   6 backtracked line search   1 SOC accepted   2 watchdog
  3 tiny step   4 soft restoration   5 full-restoration entry   7 hot-restart
  init iteration   +8 = iteration executed inside the restoration phase
  -1 = slot inactive (WS mode, already terminated)

Run from the repo root:  python -m tests.rebuttal.analyze_census
"""

import glob
import json
import os

import numpy as np

HERE = os.path.dirname(__file__)
RES_DIR = os.path.join(HERE, "results")
LOGS = ["tests/quad_nav_circle/logs", "tests/quad_track_avoid/logs",
        "tests/quad_multi_swap/logs"]
NAMES = {0: "full-step", 1: "SOC", 2: "WD", 3: "TS", 4: "SFR", 5: "resto-entry", 6: "backtracked", 7: "init"}


def code_name(c):
    base, resto = c % 8, c >= 8
    return NAMES.get(base, str(base)) + ("(in-resto)" if resto else "")


def solve_segments(col):
    """Split one slot's branch column into solves: a new solve starts at each
    init iteration (7) or after a -1 gap. Returns list of code arrays."""
    segs, cur = [], []
    for c in col:
        if c == -1:
            if cur:
                segs.append(np.array(cur)); cur = []
            continue
        if c == 7 and cur:
            segs.append(np.array(cur)); cur = []
        cur.append(c)
    if cur:
        segs.append(np.array(cur))
    return segs


def q(a, pct=(50, 95, 100)):
    a = np.asarray(a, dtype=float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return "–"
    return " / ".join(f"{np.percentile(a, p):.2e}" for p in pct)


def main():
    os.makedirs(RES_DIR, exist_ok=True)
    F = ["# Run F — branch activation census (jaxipm, DEBUG runs)\n",
         "Per config: fraction of active slot-iterations taking each branch, "
         "fraction of solves in which the branch fires at least once, and the "
         "number of DISTINCT branches simultaneously active per fused iteration.\n"]
    G = ["# Run G — termination census (jaxipm, DEBUG runs)\n",
         "| config | collected | success | code1 conv | code5 acceptable | hidden max_iter | "
         "hidden tiny-step | hidden resto-fail | iters mean | median | p95 | max | at cap |\n"
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    H = ["# Run H — KKT error components at termination (jaxipm, DEBUG runs)\n",
         "Scaled quantities exactly as used in the convergence test "
         "(overall_error ≤ tol=1e-8 declares convergence). median / p95 / max.\n",
         "| config | n | overall_error | dual_inf/s_d | constr_viol | compl/s_c |\n|---|---|---|---|---|---|"]
    I1 = ["# Run I1 — relative KKT step residual ‖KΔ−r‖/‖r‖ per fused iteration (ir_nsteps as in file)\n",
          "Same factorized K (token.values after inertia correction) as used for "
          "the step; restoration-phase iterations excluded. median / p95 / max; "
          "'early' = first 10 iterations of a solve, 'late' = iterations ≥ 30.\n",
          "| config | ir | slot-iters | all | early | late | frac > 1e-6 | frac > 1e-3 |\n|---|---|---|---|---|---|---|---|"]
    summary = {}
    for logs in LOGS:
        for path in sorted(glob.glob(os.path.join(logs, "jaxipm_ablation_*_dbg*_results.npz"))):
            Z = np.load(path, allow_pickle=True)
            cfg = os.path.basename(path).replace("jaxipm_ablation_", "").replace("_results.npz", "")
            meta = json.loads(str(Z["metadata"][0])) if "metadata" in Z.files else {}
            ir = meta.get("ir_nsteps", "?")
            iters, terms, th = Z["iters"], Z["terms"], Z["term_hist"]
            n = iters.size
            ok = iters[iters >= 0]
            max_iter = meta.get("max_iter", 500)
            G.append(f"| {cfg} | {n} | {Z['success'].mean():.4f} | {int((terms == 1).sum())} | "
                     f"{int((terms == 5).sum())} | {int(th[2])} | {int(th[3])} | {int(th[4])} | "
                     f"{ok.mean():.1f} | {np.median(ok):.0f} | {np.percentile(ok, 95):.0f} | "
                     f"{ok.max()} | {(ok >= max_iter).mean():.4f} |")
            summary[cfg] = dict(n=n, success=float(Z["success"].mean()),
                                term_hist=th.tolist(), iters_mean=float(ok.mean()),
                                iters_max=int(ok.max()))
            if "branch_log" not in Z.files:
                continue
            bl = Z["branch_log"]                     # (iters, slots)
            act = bl >= 0
            n_act = int(act.sum())
            codes, cnt = np.unique(bl[act], return_counts=True)
            # per-solve activation
            segs = [s for col in bl.T for s in solve_segments(col)]
            n_solves = len(segs)
            per_solve = {int(c): sum(1 for s in segs if (s == c).any()) for c in codes}
            distinct = np.array([len(np.unique(row[row >= 0])) for row in bl if (row >= 0).any()])
            F.append(f"\n## {cfg}  (ir={ir}; {n_act} active slot-iterations, {n_solves} solve segments, "
                     f"{bl.shape[0]} fused iterations, batch {bl.shape[1]})\n")
            F.append("| branch | code | slot-iterations | % of iterations | solves with ≥1 | % of solves |\n"
                     "|---|---|---|---|---|---|")
            for c, k in zip(codes.tolist(), cnt.tolist()):
                F.append(f"| {code_name(c)} | {c} | {k} | {100*k/n_act:.2f}% | {per_solve[c]} | "
                         f"{100*per_solve[c]/max(n_solves,1):.1f}% |")
            F.append(f"\nDistinct branches active per fused iteration: mean {distinct.mean():.2f}, "
                     f"median {np.median(distinct):.0f}, p95 {np.percentile(distinct, 95):.0f}, "
                     f"max {distinct.max()}; iterations with ≥2 distinct branches: "
                     f"{100*(distinct >= 2).mean():.1f}%, ≥3: {100*(distinct >= 3).mean():.1f}%")
            summary[cfg].update(branch_iter_frac={code_name(c): k / n_act for c, k in zip(codes.tolist(), cnt.tolist())},
                                branch_solve_frac={code_name(c): per_solve[c] / max(n_solves, 1) for c in codes.tolist()},
                                distinct_mean=float(distinct.mean()), distinct_max=int(distinct.max()))
            # H
            ke = Z["kkt_err"]
            H.append(f"| {cfg} | {ke.shape[0]} | {q(ke[:, 0])} | {q(ke[:, 1])} | {q(ke[:, 2])} | {q(ke[:, 3])} |")
            # I1 -- iteration index within the solve for early/late split
            rl = Z["kkt_res_log"]
            it_idx = np.full(bl.shape, -1)
            for j in range(bl.shape[1]):
                k = 0
                for i in range(bl.shape[0]):
                    c = bl[i, j]
                    if c == -1:
                        continue
                    if c == 7:
                        k = 0
                    it_idx[i, j] = k
                    k += 1
            fin = np.isfinite(rl)
            early = fin & (it_idx >= 0) & (it_idx < 10)
            late = fin & (it_idx >= 30)
            I1.append(f"| {cfg} | {ir} | {int(fin.sum())} | {q(rl[fin])} | {q(rl[early])} | {q(rl[late])} | "
                      f"{(rl[fin] > 1e-6).mean():.4f} | {(rl[fin] > 1e-3).mean():.4f} |")
    for fname, lines in (("census_F_branches.md", F), ("census_G_termination.md", G),
                         ("census_H_kkt.md", H), ("census_I1_residual.md", I1)):
        with open(os.path.join(RES_DIR, fname), "w") as fh:
            fh.write("\n".join(lines) + "\n")
        print("\n".join(lines)); print()
    with open(os.path.join(RES_DIR, "census_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1)


if __name__ == "__main__":
    main()

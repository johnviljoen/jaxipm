"""E1 -- embarrassingly-parallel CPU baseline vs jaxipm (rebuttal, R11-1).

Reads the multi-process IPOPT sweeps produced by
tests/*/quad_casadi_cpu_throughput.py (P single-threaded, core-pinned IPOPT
processes; logs/casadi_cpu_throughput_<variant>_<mode>_c<P>_results.npz) and
the jaxipm fresh-JIT variance campaign
(logs/jaxipm_variance_<variant>_fresh_rep<k>_results.npz, one process per
repeat, GPU, paper batch sizes, ir_nsteps=0).

Writes
  tests/rebuttal/figures/e1_cpu_scaling.pdf        solves/s vs P, per scenario
  tests/rebuttal/figures/e1_cpu_efficiency.pdf     per-core efficiency vs P
  tests/rebuttal/results/e1_cpu_scaling.md         tables for the response
  tests/rebuttal/results/e1_cpu_scaling.json       machine-readable summary

Run from the repo root:  python -m tests.rebuttal.analyze_e1_cpu_scaling
"""

import glob
import json
import os
import re

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
FIG_DIR = os.path.join(HERE, "figures")
RES_DIR = os.path.join(HERE, "results")

SCENARIOS = [
    ("nav", "tests/quad_nav_circle/logs", "sector", [90, 180],
     lambda v: f"sector{v}", lambda v: f"nav {v}°"),
    ("track", "tests/quad_track_avoid/logs", "avg_vel", [1.0, 1.5, 2.0],
     lambda v: f"v{v:.1f}", lambda v: f"track $\\bar v$={v:.1f}"),
    ("multi", "tests/quad_multi_swap/logs", "N_quads", [2, 4],
     lambda v: f"{v}", lambda v: f"multi N={v}"),
]

# Okabe-Ito (CVD-safe), fixed assignment by variant slot -- never cycled.
COLORS = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9"]

matplotlib.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Palatino", "P052", "Palatino Linotype", "URW Palladio L",
                   "DejaVu Serif"],
    "mathtext.fontset": "dejavuserif",
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#DDDDDD", "grid.linewidth": 0.5,
    "axes.edgecolor": "#888888", "xtick.color": "#444444", "ytick.color": "#444444",
    "font.size": 9,
})


SUFFIX = os.environ.get("CPU_SWEEP_SUFFIX", "")   # "" = 2026-08-03 sweep (tol 1e-6); "_tol1e-08" = run D rerun


def load_cpu(logs_dir, tag):
    rows = {}
    for path in sorted(glob.glob(os.path.join(
            logs_dir, f"casadi_cpu_throughput_{tag}_*_c*{SUFFIX}_results.npz"))):
        m = re.search(r"_(phys|smt)_c(\d+)" + re.escape(SUFFIX) + r"_results\.npz$", path)
        if not m:
            continue
        mode, P = m.group(1), int(m.group(2))
        z = np.load(path)
        rows[(mode, P)] = dict(
            mode=mode, P=P, throughput=float(z["throughput"][0]),
            wall=float(z["total_time"][0]), n_total=int(z["N_RUNS"][0]),
            success=float(z["success"].mean()),
            ms_per_solve=float(z["times"].mean() * 1e3),
            iters_mean=float(z["iters"][z["iters"] >= 0].mean()),
            iters_max=int(z["iters"].max()),
            busy_frac=float((z["worker_busy_times"] / z["total_time"][0]).mean()),
            file=os.path.relpath(path),
        )
    return rows


def load_jaxipm_fresh(logs_dir, tag):
    thr = []
    for path in sorted(glob.glob(os.path.join(
            logs_dir, f"jaxipm_variance_{tag}_fresh_rep*_results.npz"))):
        z = np.load(path)
        thr.extend(np.asarray(z["rep_throughput"]).tolist())
    thr = np.array(thr)
    if thr.size == 0:
        return None
    return dict(n=int(thr.size), mean=float(thr.mean()),
                std=float(thr.std(ddof=1)) if thr.size > 1 else 0.0,
                reps=thr.tolist())


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(RES_DIR, exist_ok=True)
    summary = {}
    md = ["# E1 — embarrassingly-parallel IPOPT baseline vs jaxipm\n",
          "Protocol: P independent single-threaded CasADi/IPOPT processes "
          "(OMP/BLAS threads = 1), each pinned to one logical CPU "
          "(`phys`: P distinct physical cores; `smt`: both SMT siblings of "
          "P/2 physical cores), shared work queue over the same problem pool "
          "jaxipm consumes, wall clock excludes process construction and one "
          "warm-up solve per worker. IPOPT `tol=1e-6`, `max_iter=500` "
          "(2026-08-03 sweep) or `tol=1e-8` (run D rerun, suffix _tol1e-08; "
          f"this file: {SUFFIX or 'tol 1e-6'}). jaxipm: L40S, paper batch size, "
          "`ir_nsteps=0`, `tol=1e-8`, mean ± std over 5 fresh-process repeats "
          "(2026-08-13). CPU sweep measured 2026-08-03 on iconlab "
          "(2× EPYC 9354, 64 physical / 128 logical).\n"]

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5))
    fig2, axes2 = plt.subplots(1, 3, figsize=(7.2, 2.5))
    for (name, logs_dir, vname, variants, tagf, labf), ax, ax2 in zip(
            SCENARIOS, axes, axes2):
        summary[name] = {}
        md.append(f"\n## {name}\n")
        for vi, v in enumerate(variants):
            tag = tagf(v)
            cpu = load_cpu(logs_dir, tag)
            gpu = load_jaxipm_fresh(logs_dir, tag)
            if not cpu:
                md.append(f"\n*{labf(v)}: no CPU sweep data*\n")
                continue
            base = cpu[("phys", 1)]["throughput"]
            col = COLORS[vi]
            for mode, ls in (("phys", "-"), ("smt", "--")):
                Ps = sorted(P for (m, P) in cpu if m == mode)
                if not Ps:
                    continue
                thr = np.array([cpu[(mode, P)]["throughput"] for P in Ps])
                eff = thr / (base * np.array(Ps))
                ax.plot(Ps, thr, ls, color=col, lw=1.4, marker="o", ms=3,
                        label=f"IPOPT {labf(v)} ({mode})")
                ax2.plot(Ps, eff, ls, color=col, lw=1.4, marker="o", ms=3,
                         label=f"{labf(v)} ({mode})")
            if gpu:
                ax.axhline(gpu["mean"], color=col, lw=1.0, ls=":",
                           label=f"jaxipm {labf(v)}")

            # crossover: smallest P at which IPOPT throughput >= jaxipm mean
            cross = {}
            for mode in ("phys", "smt"):
                Ps = sorted(P for (m, P) in cpu if m == mode)
                if gpu:
                    hit = [P for P in Ps if cpu[(mode, P)]["throughput"] >= gpu["mean"]]
                    cross[mode] = hit[0] if hit else None
            p64 = cpu.get(("phys", 64), {}).get("throughput")
            s128 = cpu.get(("smt", 128), {}).get("throughput")
            summary[name][str(v)] = dict(
                cpu={f"{m}_{P}": r for (m, P), r in cpu.items()},
                jaxipm=gpu, p1_throughput=base, phys64_throughput=p64,
                smt128_throughput=s128, crossover_P=cross,
                ratio_phys64_over_jaxipm=(p64 / gpu["mean"]) if (gpu and p64) else None,
                ratio_smt128_over_jaxipm=(s128 / gpu["mean"]) if (gpu and s128) else None,
                ratio_jaxipm_over_p1=(gpu["mean"] / base) if gpu else None,
            )

            md.append(f"\n### {labf(v)}\n")
            if gpu:
                md.append(f"jaxipm (GPU, fresh-JIT ×{gpu['n']}): "
                          f"**{gpu['mean']:.2f} ± {gpu['std']:.2f} solves/s**  ·  "
                          f"IPOPT P=1 pinned: {base:.2f} solves/s "
                          f"(jaxipm/P1 = {gpu['mean']/base:.1f}×)  ·  "
                          f"IPOPT P=64 phys: {p64:.1f} solves/s "
                          f"(= {p64/gpu['mean']:.2f}× jaxipm)  ·  "
                          f"P=128 SMT: {s128:.1f} solves/s "
                          f"(= {s128/gpu['mean']:.2f}× jaxipm)  ·  "
                          f"IPOPT matches jaxipm from P = {cross['phys']} phys / "
                          f"{cross['smt']} smt\n")
            md.append("\n| mode | P | solves | wall [s] | solves/s | speedup vs P=1 "
                      "| per-core eff. | success | ms/solve | iters mean/max | busy |\n"
                      "|---|---|---|---|---|---|---|---|---|---|---|")
            for mode in ("phys", "smt"):
                for P in sorted(P for (m, P) in cpu if m == mode):
                    r = cpu[(mode, P)]
                    md.append(f"| {mode} | {P} | {r['n_total']} | {r['wall']:.2f} | "
                              f"{r['throughput']:.2f} | {r['throughput']/base:.2f}× | "
                              f"{r['throughput']/(base*P):.2f} | {r['success']:.3f} | "
                              f"{r['ms_per_solve']:.1f} | {r['iters_mean']:.1f}/{r['iters_max']} | "
                              f"{r['busy_frac']:.2f} |")
            md.append("")

        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("IPOPT processes $P$")
        ax.set_title(name, fontsize=9)
        ax2.set_xscale("log", base=2)
        ax2.set_ylim(0, 1.1)
        ax2.set_xlabel("IPOPT processes $P$")
        ax2.set_title(name, fontsize=9)
    axes[0].set_ylabel("throughput [solves/s]")
    axes2[0].set_ylabel("per-core efficiency  $T(P)\\,/\\,(P\\,T(1))$")
    for a in axes:
        a.legend(fontsize=5.0, frameon=False, loc="lower right", handlelength=2.0,
                 labelspacing=0.25)
    for a in axes2:
        a.legend(fontsize=5.0, frameon=False, loc="lower left", handlelength=2.0,
                 labelspacing=0.25)
    for f_, name_ in ((fig, "e1_cpu_scaling"), (fig2, "e1_cpu_efficiency")):
        f_.tight_layout(pad=0.3)
        f_.savefig(os.path.join(FIG_DIR, f"{name_}{SUFFIX}.pdf"))
        f_.savefig(os.path.join(FIG_DIR, f"{name_}{SUFFIX}.png"), dpi=200)

    # Headline table for the response letter
    md.append("\n## Headline comparison (one L40S vs one 2-socket EPYC 9354 node)\n")
    md.append("| scenario | jaxipm GPU [solves/s] | IPOPT P=1 | jaxipm / P=1 | "
              "IPOPT 64 phys | IPOPT 128 SMT | 64-phys / jaxipm | 128-SMT / jaxipm | "
              "P at which IPOPT ≥ jaxipm (phys/smt) |\n|---|---|---|---|---|---|---|---|---|")
    for name, _, _, variants, tagf, labf in SCENARIOS:
        for v in variants:
            s = summary[name].get(str(v))
            if not s or not s["jaxipm"]:
                continue
            g = s["jaxipm"]
            md.append(f"| {labf(v).replace(chr(36)+chr(92)+chr(98)+chr(97)+chr(114)+chr(32)+chr(118)+chr(36), chr(118)+chr(772))} | {g['mean']:.2f} ± {g['std']:.2f} | "
                      f"{s['p1_throughput']:.2f} | {s['ratio_jaxipm_over_p1']:.1f}× | "
                      f"{s['phys64_throughput']:.1f} | {s['smt128_throughput']:.1f} | "
                      f"{s['ratio_phys64_over_jaxipm']:.2f}× | "
                      f"{s['ratio_smt128_over_jaxipm']:.2f}× | "
                      f"{s['crossover_P']['phys']} / {s['crossover_P']['smt']} |")
    tagout = SUFFIX.replace("_", "") or "tol1e-6"
    with open(os.path.join(RES_DIR, f"e1_cpu_scaling_{tagout}.md"), "w") as fh:
        fh.write("\n".join(md).replace("$\\bar v$", "v\u0304") + "\n")
    with open(os.path.join(RES_DIR, f"e1_cpu_scaling_{tagout}.json"), "w") as fh:
        json.dump(summary, fh, indent=1, default=str)
    print("\n".join(md[-12:]))
    print(f"\nwrote {RES_DIR}/e1_cpu_scaling.{{md,json}} and {FIG_DIR}/e1_cpu_*.pdf")


if __name__ == "__main__":
    main()

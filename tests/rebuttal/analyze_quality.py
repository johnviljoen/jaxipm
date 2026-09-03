"""E4 -- post-hoc solution quality of saved jaxipm ablation solutions.

For every tests/*/logs/jaxipm_ablation_*_results.npz that contains `z_all`,
re-evaluates the USER-LEVEL (unscaled) problem functions at the returned point:

  eq_inf   = ||c(z)||_inf          eq_1 = ||c(z)||_1
  ineq_viol= max(d_L - d(z), d(z) - d_U, 0)_inf
  bound_viol = max(x_L - z, z - x_U, 0)_inf
  obj      = f(z)   (recomputed; must match obj_vals)

Stationarity / complementarity need the multipliers, which solve_throughput
does not scatter; they are reported by the correctness harness
(tests/correctness) on single problems instead.

Writes tests/rebuttal/results/e4_quality.md (one row per file: distribution
quantiles) and, per file, a sidecar <file>.quality.npz with the per-solve
columns. Run from the repo root (GPU not needed but jax must import):
    python -m tests.rebuttal.analyze_quality [--gpu -1]
"""

import argparse
import glob
import json
import os
import re

import numpy as np

HERE = os.path.dirname(__file__)
RES_DIR = os.path.join(HERE, "results")

LOGS = {"nav": "tests/quad_nav_circle/logs", "track": "tests/quad_track_avoid/logs",
        "multi": "tests/quad_multi_swap/logs"}
TAG_RE = {"nav": r"jaxipm_ablation_sector(\d+)_", "track": r"jaxipm_ablation_v([\d.]+)_",
          "multi": r"jaxipm_ablation_(\d+)_"}


def q(a):
    a = np.asarray(a, dtype=float)
    return (f"{np.median(a):.2e} / {np.percentile(a, 95):.2e} / {a.max():.2e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=str, default="",
                    help="CUDA_VISIBLE_DEVICES ('' = CPU-only jax)")
    ap.add_argument("--only", type=str, default=None, help="substring filter on filenames")
    args = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ.setdefault("JAX_PLATFORMS", "cpu" if args.gpu == "" else "cuda,cpu")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

    import jax
    jax.config.update("jax_enable_x64", True)
    from tests.rebuttal.problems import build_scenario, load_params, parse_variant

    os.makedirs(RES_DIR, exist_ok=True)
    md = ["# E4 — constraint violation at the returned point (jaxipm ablation runs)\n",
          "median / p95 / max over collected solves. Tolerances for reference: "
          "jaxipm tol 1e-8 on the SCALED KKT error; `constr_viol_tol` 1e-4 "
          "(acceptable-point criterion).\n",
          "| file | n | success | obj mean | ‖c‖∞ | ‖c‖₁ | ineq viol | bound viol | "
          "obj recompute max|Δ| |\n|---|---|---|---|---|---|---|---|---|"]
    p = load_params()
    cache = {}
    for name, logs in LOGS.items():
        for path in sorted(glob.glob(os.path.join(logs, "jaxipm_ablation_*_results.npz"))):
            if args.only and args.only not in path:
                continue
            Z = np.load(path, allow_pickle=True)
            if "z_all" not in Z.files or Z["z_all"].shape[0] == 0:
                continue
            m = re.search(TAG_RE[name], os.path.basename(path))
            variant = parse_variant(name, m.group(1))
            key = (name, variant)
            if key not in cache:
                cache[key] = build_scenario(name, variant, p, N_batch=1)
            sc = cache[key]
            x_L, x_U, d_L, d_U = sc.bounds
            z_all, pool_idx = Z["z_all"], Z["pool_idx"]
            n = z_all.shape[0]
            eq_inf = np.zeros(n); eq_1 = np.zeros(n); ineq = np.zeros(n)
            bnd = np.zeros(n); obj = np.zeros(n)
            for i in range(n):
                z = z_all[i]
                c = sc.c_fn(z, pool_idx[i]); d = sc.d_fn(z, pool_idx[i])
                eq_inf[i] = np.abs(c).max(); eq_1[i] = np.abs(c).sum()
                ineq[i] = np.maximum(np.maximum(d_L - d, d - d_U), 0).max()
                bnd[i] = np.maximum(np.maximum(x_L - z, z - x_U), 0).max()
                obj[i] = sc.objective(z, pool_idx[i])
            dobj = np.abs(obj - Z["obj_vals"]).max()
            base = os.path.relpath(path)
            np.savez(path.replace("_results.npz", "_results.quality.npz"),
                     eq_inf=eq_inf, eq_1=eq_1, ineq_viol=ineq, bound_viol=bnd, obj=obj,
                     pool_idx=pool_idx, source=np.array([base]))
            md.append(f"| {os.path.basename(path)} | {n} | {Z['success'].mean():.4f} | "
                      f"{obj.mean():.3f} | {q(eq_inf)} | {q(eq_1)} | {q(ineq)} | {q(bnd)} | "
                      f"{dobj:.1e} |")
            print(md[-1], flush=True)
    with open(os.path.join(RES_DIR, "e4_quality.md"), "w") as fh:
        fh.write("\n".join(md) + "\n")
    print(f"wrote {RES_DIR}/e4_quality.md")


if __name__ == "__main__":
    main()

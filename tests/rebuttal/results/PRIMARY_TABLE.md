# PRIMARY TABLE — consolidated cells (2026-08-27 20:40, hand-assembled from the result files)

All solvers: tol 1e-8, iteration cap 500, FP64, same problem pool (nav 2000 / track 1000 / multi 75). Throughput ± = repeat-to-repeat sd (≥5 repeats); cost ± = sd over the problem pool; ‖c‖∞ = max |equality residual| at the returned point evaluated with jaxipm's own constraint function for every solver (median / mean / max over the pool). Sources: run A (`jaxipm_variance_*_fresh_rep*`), run B fresh (`jaxipm_ablation_*_ws_*_fresh_rep*`), run D `D_p64_repeats.md`, `ipopt_default_repeats.md`, run E `E_madnlp.md`, `primary_table_quality.md` (rebuilt 20:45 with all seven MadNLP reruns; agrees with the values transcribed earlier).

## Throughput [solves/s], mean ± sd over repeats

| scenario | IPOPT default (1 proc, MUMPS ∥, whole CPU) ×5 | IPOPT P=64 pinned ×5 (`now` set, loaded host) | IPOPT P=64 single sweep run | MadNLP (GPU) ×5 | jaxipm-WS (ILB off) ×5 fresh | jaxipm-IL (ILB on) ×5 fresh |
|---|---|---|---|---|---|---|
| nav 90° | 6.61 ± 0.39 | 280.7 ± 13.9 | 350.2 | 2.250 ± 0.282 | 15.62 ± 1.63 | 101.69 ± 2.96 |
| nav 180° | 6.99 ± 0.28 | 310.9 ± 14.8 | 353.5 | 2.744 ± 0.022 | 15.68 ± 2.31 | 105.82 ± 6.32 |
| track v̄=1.0 | 6.06 ± 0.30 | 234.6 ± 9.5 | 259.7 | 2.970 ± 0.989 | 4.78 ± 0.16 | 35.75 ± 1.20 |
| track v̄=1.5 | 4.61 ± 0.27 | 185.3 ± 3.8 | 221.2 | 2.016 ± 0.012 | 4.52 ± 0.16 | 31.60 ± 0.76 |
| track v̄=2.0 | 2.55 ± 0.14 | 105.4 ± 3.7 | 125.6 | 1.649 ± 0.002 | 5.36 ± 0.22 | 30.94 ± 0.50 |
| multi N=2 | 0.33 ± 0.01 | 13.5 ± 0.7 | 15.4 | 0.945 ± 0.066 | 12.01 ± 0.67 | 8.72 ± 0.28 |
| multi N=4 | 0.11 ± 0.00 | 5.2 ± 0.2 | 5.6 | 0.428 ± 0.005 | 2.01 ± 0.14 | 1.35 ± 0.06 |

Host-load caveat: the IPOPT-default and P=64 repeats ran while other users loaded the host (5-min load 60–150; stamped in each npz as `loadavg_at_save`); the overnight P=64 sweep single run (22:36–00:53) saw less contention and sits 10–20 % higher. Within-set sd is 2–6 % for all CPU configs.

## Final cost, mean ± sd over the pool (successful solves), and ‖c‖∞ at the returned point

| scenario | IPOPT (default ≡ P=64: identical solutions) cost | ‖c‖∞ IPOPT | MadNLP cost | ‖c‖∞ MadNLP | jaxipm-IL cost | ‖c‖∞ jaxipm-IL | jaxipm-WS cost | ‖c‖∞ jaxipm-WS |
|---|---|---|---|---|---|---|---|---|
| nav 90° | 516.953 ± 20.286 | 1.6e-12 / 3.9e-10 / 8.9e-09 | 516.994 ± 20.302 | 2.0e-08 / 2.0e-08 / 2.0e-08 | 516.479 ± 20.225 | 3.5e-13 / 1.5e-11 / 7.7e-09 | 516.164 ± 20.159 | 3.4e-13 / 1.6e-11 / 9.7e-09 |
| nav 180° | 507.571 ± 17.143 | 9.0e-13 / 2.0e-10 / 9.9e-09 | 507.591 ± 17.162 | 2.0e-08 / 2.0e-08 / 2.0e-08 | 506.779 ± 16.479 | 3.4e-13 / 1.5e-11 / 9.9e-09 | 506.816 ± 17.073 | 3.3e-13 / 1.8e-11 / 8.3e-09 |
| track v̄=1.0 | 265.709 ± 98.667 | 8.7e-11 / 7.8e-10 / 9.7e-09 | 261.701 ± 92.033 | 2.0e-08 / 2.0e-08 / 2.0e-08 | 245.593 ± 75.545 | 1.0e-13 / 1.2e-11 / 2.2e-09 | 249.396 ± 79.394 | 9.5e-14 / 1.8e-11 / 7.3e-09 |
| track v̄=1.5 | 556.694 ± 140.884 | 6.9e-11 / 9.0e-10 / 9.8e-09 | 555.812 ± 138.326 | 2.0e-08 / 2.0e-08 / 2.0e-08 | 545.524 ± 123.372 | 1.5e-13 / 2.5e-11 / 2.4e-09 | 543.564 ± 123.421 | 1.4e-13 / 2.7e-11 / 3.4e-09 |
| track v̄=2.0 | 1146.663 ± 126.935 | 3.6e-12 / 2.1e-10 / 4.6e-09 | 1147.484 ± 121.172 | 2.0e-08 / 2.0e-08 / 2.0e-08 | 1146.953 ± 131.923 | 2.4e-13 / 2.7e-11 / 8.0e-09 | 1142.525 ± 128.938 | 2.3e-13 / 1.9e-11 / 2.6e-09 |
| multi N=2 | 756.311 ± 0.000 | 3.1e-11 / 3.1e-11 / 3.1e-11 | 756.311 ± 0.000 | 2.0e-08 / 2.0e-08 / 2.0e-08 | 756.311 ± 0.000 | 2.8e-13 / 6.8e-13 / 9.7e-12 | 756.311 ± 0.000 | 2.2e-13 / 6.7e-13 / 1.0e-11 |
| multi N=4 | 1537.688 ± 0.000 | 9.2e-12 / 9.2e-12 / 9.2e-12 | 1537.704 ± 0.014 | 2.0e-08 / 2.0e-08 / 2.0e-08 | 1537.702 ± 0.014 | 3.1e-12 / 1.1e-11 / 1.1e-10 | 1537.703 ± 0.014 | 1.2e-12 / 7.7e-12 / 4.8e-11 |

## Inequality and bound violations (median / mean / max over the pool)

| scenario | solver | ineq viol max(d_L−d, d−d_U, 0) | bound viol max(x_L−z, z−x_U, 0) |
|---|---|---|---|
| nav 90° | IPOPT (both configs) | 9.9e-09 / 8.7e-09 / 1.0e-08 | 0.0e+00 / 0.0e+00 / 0.0e+00 |
| nav 90° | MadNLP | 2.0e-08 / 1.8e-08 / 2.0e-08 | 0.0e+00 / 0.0e+00 / 0.0e+00 |
| nav 90° | jaxipm-IL | 1.0e-08 / 9.0e-09 / 1.0e-08 | 0.0e+00 / 0.0e+00 / 0.0e+00 |
| nav 90° | jaxipm-WS | 1.0e-08 / 8.8e-09 / 1.0e-08 | 0.0e+00 / 0.0e+00 / 0.0e+00 |
| nav 180° | IPOPT (both configs) | 0.0e+00 / 4.4e-09 / 1.0e-08 | 0.0e+00 / 0.0e+00 / 0.0e+00 |
| nav 180° | MadNLP | 0.0e+00 / 8.8e-09 / 2.0e-08 | 0.0e+00 / 0.0e+00 / 0.0e+00 |
| nav 180° | jaxipm-IL | 0.0e+00 / 4.6e-09 / 1.0e-08 | 0.0e+00 / 0.0e+00 / 0.0e+00 |
| nav 180° | jaxipm-WS | 0.0e+00 / 4.4e-09 / 1.0e-08 | 0.0e+00 / 0.0e+00 / 0.0e+00 |
| track v̄=1.0 | IPOPT (both configs) | 9.9e-09 / 9.1e-09 / 1.0e-08 | 8.9e-06 / 8.0e-06 / 9.2e-06 |
| track v̄=1.0 | MadNLP | 2.0e-08 / 1.8e-08 / 2.0e-08 | 9.1e-06 / 8.3e-06 / 9.2e-06 |
| track v̄=1.0 | jaxipm-IL | 1.0e-08 / 8.7e-09 / 1.0e-08 | 8.9e-09 / 7.9e-09 / 9.6e-09 |
| track v̄=1.0 | jaxipm-WS | 1.0e-08 / 8.8e-09 / 1.0e-08 | 8.9e-09 / 7.9e-09 / 9.7e-09 |
| track v̄=1.5 | IPOPT (both configs) | 9.9e-09 / 1.0e-08 / 1.5e-08 | 9.1e-06 / 8.9e-06 / 9.2e-06 |
| track v̄=1.5 | MadNLP | 2.0e-08 / 2.1e-08 / 3.0e-08 | 9.2e-06 / 9.1e-06 / 9.2e-06 |
| track v̄=1.5 | jaxipm-IL | 1.0e-08 / 1.0e-08 / 1.0e-08 | 9.6e-09 / 9.3e-09 / 9.9e-09 |
| track v̄=1.5 | jaxipm-WS | 1.0e-08 / 9.9e-09 / 1.0e-08 | 9.5e-09 / 9.3e-09 / 9.9e-09 |
| track v̄=2.0 | IPOPT (both configs) | 1.5e-08 / 1.5e-08 / 1.5e-08 | 9.2e-06 / 9.2e-06 / 9.2e-06 |
| track v̄=2.0 | MadNLP | 3.0e-08 / 3.0e-08 / 3.0e-08 | 9.2e-06 / 9.2e-06 / 9.2e-06 |
| track v̄=2.0 | jaxipm-IL | 1.0e-08 / 1.0e-08 / 1.0e-08 | 9.8e-09 / 9.8e-09 / 1.0e-08 |
| track v̄=2.0 | jaxipm-WS | 1.0e-08 / 1.0e-08 / 1.0e-08 | 9.8e-09 / 9.8e-09 / 1.0e-08 |
| multi N=2 | IPOPT (both configs) | 9.5e-09 / 9.5e-09 / 9.5e-09 | 9.1e-06 / 9.1e-06 / 9.1e-06 |
| multi N=2 | MadNLP | 2.0e-08 / 2.0e-08 / 2.0e-08 | 9.1e-06 / 9.1e-06 / 9.2e-06 |
| multi N=2 | jaxipm-IL | 1.0e-08 / 1.0e-08 / 1.0e-08 | 9.4e-09 / 9.4e-09 / 9.4e-09 |
| multi N=2 | jaxipm-WS | 1.0e-08 / 1.0e-08 / 1.0e-08 | 9.4e-09 / 9.4e-09 / 9.4e-09 |
| multi N=4 | IPOPT (both configs) | 9.8e-09 / 9.8e-09 / 9.8e-09 | 9.1e-06 / 9.1e-06 / 9.1e-06 |
| multi N=4 | MadNLP | 2.0e-08 / 2.0e-08 / 2.0e-08 | 9.1e-06 / 9.1e-06 / 9.2e-06 |
| multi N=4 | jaxipm-IL | 1.0e-08 / 1.0e-08 / 1.0e-08 | 9.5e-09 / 9.5e-09 / 9.5e-09 |
| multi N=4 | jaxipm-WS | 1.0e-08 / 1.0e-08 / 1.0e-08 | 9.5e-09 / 9.5e-09 / 9.5e-09 |

## Footnotes for the table

1. IPOPT default and IPOPT P=64 return bit-identical solutions (same solver, tolerance, pool), so one cost/violation column serves both.
2. jaxipm-IL cost is taken from the census run of the paper configuration (fixes the obj=0 scatter-collision outlier that inflates the paper npz sd, e.g. nav 90° 517.16 ± 24.06 → 516.48 ± 20.23). Success 100 % for every solver at tol 1e-8.
3. MadNLP's ‖c‖∞ is exactly 2e-8 on every problem: it enforces the fixed initial state as a relaxed variable bound (bound_relax_factor ≈ 1e-8); its dynamics rows are satisfied to ~1e-15. IPOPT and jaxipm enforce x₀ as an equality.
4. Bound violations of 9e-6 for IPOPT and MadNLP on track/multi are their bound relaxation on the ±900 rad/s motor-speed bounds; jaxipm's are ≤ 1e-8. Inequality violations are ≤ 1e-8 (MadNLP 2–3e-8) for all.
5. Paired cost tests (jaxipm vs IPOPT on identical instances): `existing_e4_cost.md` — jaxipm ≈ 0.1 % lower on nav (96–99 % of instances), identical medians on track with different local optima on 11–27 % of instances.
6. Multi-swap is one fixed problem solved 75× (pool sd = 0 by construction).
7. Order statistics / ILB explanation: `L_orderstats_vs_measured.md`; ladder (N=1 → ILB off → ILB on): `C_ladder_and_sweeps.md`.

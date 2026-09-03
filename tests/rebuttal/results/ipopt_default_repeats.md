# IPOPT default configuration (one IPOPT/MUMPS process, whole CPU, cold solves) — tol 1e-8, repeats

Same driver and pool sizes as the paper's IPOPT column (N_RUNS_seq = 80 nav / 50 track / 50 multi), tol 1e-8, R independent repeats; throughput = N_RUNS / total wall. Host 5-min load average at save time listed per repeat (other users' jobs were running). `pool` = one pass over the full jaxipm pool (2000/1000/75) with inputs saved for the ‖c‖∞ column. Paper column = original tol 1e-6 single run.

| scenario | R | throughput mean ± sd [solves/s] | per-repeat | load (5 min) per repeat | mean solve [ms] | iters mean | cost mean ± sd (pooled over repeats) | success | pool pass: throughput / cost mean ± sd (n) | paper (1e-6): throughput / cost mean ± sd (n) |
|---|---|---|---|---|---|---|---|---|---|---|
| nav 90° | 5 | 6.61 ± 0.39 | 5.96, 6.68, 6.72, 6.99, 6.72 | 98, 91, 87, 85, 94 | 151 ± 9 | 23.4 ± 0.0 | 516.959 ± 20.332 | 100.0% | 6.44 / 516.953 ± 20.286 (2000) | 7.87 / 516.959 ± 20.435 (80) |
| nav 180° | 5 | 6.99 ± 0.28 | 6.49, 7.10, 7.12, 7.18, 7.04 | 99, 90, 88, 85, 94 | 143 ± 6 | 21.5 ± 0.0 | 507.536 ± 17.071 | 100.0% | 7.33 / 507.571 ± 17.143 (2000) | 8.76 / 507.536 ± 17.157 (80) |
| track v̄=1.0 | 5 | 6.06 ± 0.30 | 5.57, 6.05, 6.13, 6.41, 6.13 | 103, 91, 87, 83, 92 | 165 ± 9 | 24.2 ± 0.0 | 262.504 ± 80.122 | 100.0% | 5.40 / 265.709 ± 98.667 (1000) | 7.12 / 262.504 ± 80.774 (50) |
| track v̄=1.5 | 5 | 4.61 ± 0.27 | 4.52, 4.91, 4.48, 4.85, 4.28 | 103, 89, 85, 81, 90 | 217 ± 13 | 29.4 ± 0.0 | 537.435 ± 117.023 | 100.0% | 4.43 / 556.694 ± 140.884 (1000) | 5.42 / 537.435 ± 117.975 (50) |
| track v̄=2.0 | 5 | 2.55 ± 0.14 | 2.39, 2.61, 2.54, 2.75, 2.47 | 103, 88, 82, 79, 89 | 392 ± 21 | 49.3 ± 0.0 | 1134.160 ± 123.430 | 100.0% | 2.65 / 1146.663 ± 126.935 (1000) | 3.00 / 1134.160 ± 124.434 (50) |
| multi N=2 | 5 | 0.33 ± 0.01 | 0.33, 0.33, 0.34, 0.35, 0.32 | 101, 81, 70, 81, 88 | 3005 ± 118 | 178.0 ± 0.0 | 756.311 ± 0.000 | 100.0% | 0.34 / 756.311 ± 0.000 (75) | 0.40 / 756.311 ± 0.000 (50) |
| multi N=4 | 5 | 0.11 ± 0.00 | 0.11, 0.11, 0.11, 0.11, 0.11 | 94, 86, 85, 96, 86 | 9129 ± 281 | 217.0 ± 0.0 | 1537.688 ± 0.000 | 100.0% | 0.11 / 1537.688 ± 0.000 (75) | 0.13 / 1537.688 ± 0.000 (50) |

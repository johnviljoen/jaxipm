# Run J — per-step injection correctness across nav-circle pool instances

jaxipm is seeded from IPOPT's exact internal state at every iteration k, takes one step, and Δx = ‖x_jaxipm(k+1) − x_IPOPT(k+1)‖_∞ is recorded (ir_nsteps=100, VALIDATION_MODE). Branch = the branch jaxipm's fused body selected for that step (DEBUG_MODE branch id). tight: Δx ≤ 1e-06; drift: Δx > 1e-04. Caveat: the KKT solves are poorly conditioned on some steps, so individual Δx values vary run-to-run (cuDSS static pivoting is not deterministic); the tight fraction and the median are the stable statistics.

| instance | IPOPT iters | steps compared | median Δx | p90 Δx | max Δx | tight frac | drift frac | first tight-miss | first drift | resto-flag agreement | branches (jaxipm) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| orig | 94 | 94 | 2.9e-14 | 6.5e-13 | 5.4e+02 | 98.9% | 1.1% | 36 | 36 | 99% | n/a 94 |
| orig | 94 | 94 | 7.8e-15 | 6.1e-13 | 5.4e+02 | 98.9% | 1.1% | 36 | 36 | 99% | n/a 1, full step 83, SOC 3, resto entry 1, backtracked 6 |
| 0 | 10 | 10 | 3.4e-13 | 3.4e-11 | 3.4e-10 | 100.0% | 0.0% | None | None | 100% | n/a 1, full step 8, SOC 1 |
| 182 | 18 | 18 | 1.7e-13 | 1.0e-12 | 1.6e-10 | 100.0% | 0.0% | None | None | 100% | n/a 1, full step 12, SOC 1, backtracked 4 |
| 364 | 13 | 13 | 4.3e-13 | 2.2e-10 | 4.1e-10 | 100.0% | 0.0% | None | None | 100% | n/a 1, full step 11, SOC 1 |
| 545 | 43 | 43 | 3.4e-13 | 2.0e-10 | 3.4e+02 | 97.7% | 2.3% | 5 | 5 | 100% | n/a 1, full step 26, SOC 9, backtracked 7 |
| 727 | 15 | 15 | 5.1e-13 | 6.8e-12 | 2.9e+02 | 93.3% | 6.7% | 2 | 2 | 93% | n/a 1, full step 13, resto entry 1 |
| 909 | 17 | 17 | 3.4e-13 | 2.0e-12 | 1.1e-09 | 100.0% | 0.0% | None | None | 100% | n/a 1, full step 15, SOC 1 |
| 1090 | 18 | 18 | 3.0e-13 | 2.8e-12 | 1.8e-10 | 100.0% | 0.0% | None | None | 100% | n/a 1, full step 16, SOC 1 |
| 1272 | 18 | 18 | 2.3e-13 | 7.5e-12 | 3.9e-09 | 100.0% | 0.0% | None | None | 100% | n/a 1, full step 14, SOC 2, backtracked 1 |
| 1454 | 17 | 17 | 8.5e-13 | 1.8e-10 | 1.8e-09 | 100.0% | 0.0% | None | None | 100% | n/a 1, full step 10, SOC 5, backtracked 1 |
| 1636 | 18 | 18 | 2.8e-13 | 9.3e-13 | 3.7e-10 | 100.0% | 0.0% | None | None | 100% | n/a 1, full step 15, SOC 1, backtracked 1 |
| 1818 | 14 | 14 | 2.6e-13 | 3.5e-12 | 1.9e-11 | 100.0% | 0.0% | None | None | 100% | n/a 1, full step 8, SOC 2, backtracked 3 |
| 1999 | 10 | 10 | 3.7e-13 | 7.8e-02 | 7.8e-01 | 90.0% | 10.0% | 9 | 9 | 90% | n/a 1, full step 7, SOC 1, resto entry 1 |

**All instances pooled:** 399 steps, median Δx 1.1e-13, tight 98.7%, drift 1.3%.

| branch | steps | median Δx | tight frac | drift frac |
|---|---|---|---|---|
| n/a | 107 | 4.4e-16 | 99.1% | 0.9% |
| full step | 238 | 2.0e-13 | 100.0% | 0.0% |
| SOC | 28 | 1.1e-10 | 100.0% | 0.0% |
| resto entry | 3 | 7.8e-01 | 33.3% | 66.7% |
| backtracked | 23 | 1.1e-13 | 91.3% | 8.7% |

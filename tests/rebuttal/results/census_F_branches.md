# Run F — branch activation census (jaxipm, DEBUG runs)

Per config: fraction of active slot-iterations taking each branch, fraction of solves in which the branch fires at least once, and the number of DISTINCT branches simultaneously active per fused iteration.


## sector180_hr_ir0_b500_dbg  (ir=0; 111500 active slot-iterations, 2491 solve segments, 223 fused iterations, batch 500)

| branch | code | slot-iterations | % of iterations | solves with ≥1 | % of solves |
|---|---|---|---|---|---|
| full-step | 0 | 74553 | 66.86% | 2476 | 99.4% |
| SOC | 1 | 23772 | 21.32% | 2337 | 93.8% |
| WD | 2 | 36 | 0.03% | 7 | 0.3% |
| SFR | 4 | 2 | 0.00% | 1 | 0.0% |
| resto-entry | 5 | 56 | 0.05% | 10 | 0.4% |
| backtracked | 6 | 10562 | 9.47% | 1637 | 65.7% |
| init | 7 | 2491 | 2.23% | 2491 | 100.0% |
| full-step(in-resto) | 8 | 28 | 0.03% | 9 | 0.4% |

Distinct branches active per fused iteration: mean 4.35, median 4, p95 6, max 7; iterations with ≥2 distinct branches: 99.6%, ≥3: 97.3%

## sector180_ws_ir0_b1_dbg_i2  (ir=0; 907 active slot-iterations, 48 solve segments, 907 fused iterations, batch 1)

| branch | code | slot-iterations | % of iterations | solves with ≥1 | % of solves |
|---|---|---|---|---|---|
| full-step | 0 | 782 | 86.22% | 48 | 100.0% |
| SOC | 1 | 48 | 5.29% | 39 | 81.2% |
| backtracked | 6 | 29 | 3.20% | 18 | 37.5% |
| init | 7 | 48 | 5.29% | 48 | 100.0% |

Distinct branches active per fused iteration: mean 1.00, median 1, p95 1, max 1; iterations with ≥2 distinct branches: 0.0%, ≥3: 0.0%

## sector180_ws_ir0_b32_dbg_i2  (ir=0; 3862 active slot-iterations, 64 solve segments, 556 fused iterations, batch 32)

| branch | code | slot-iterations | % of iterations | solves with ≥1 | % of solves |
|---|---|---|---|---|---|
| full-step | 0 | 2287 | 59.22% | 64 | 100.0% |
| SOC | 1 | 891 | 23.07% | 55 | 85.9% |
| WD | 2 | 19 | 0.49% | 2 | 3.1% |
| resto-entry | 5 | 2 | 0.05% | 2 | 3.1% |
| backtracked | 6 | 596 | 15.43% | 46 | 71.9% |
| init | 7 | 64 | 1.66% | 64 | 100.0% |
| full-step(in-resto) | 8 | 3 | 0.08% | 1 | 1.6% |

Distinct branches active per fused iteration: mean 1.96, median 2, p95 3, max 4; iterations with ≥2 distinct branches: 60.6%, ≥3: 35.1%

## sector180_ws_ir0_b500_dbg  (ir=0; 126428 active slot-iterations, 2000 solve segments, 1557 fused iterations, batch 500)

| branch | code | slot-iterations | % of iterations | solves with ≥1 | % of solves |
|---|---|---|---|---|---|
| full-step | 0 | 75384 | 59.63% | 2000 | 100.0% |
| SOC | 1 | 30528 | 24.15% | 1899 | 95.0% |
| WD | 2 | 206 | 0.16% | 33 | 1.6% |
| SFR | 4 | 11 | 0.01% | 4 | 0.2% |
| resto-entry | 5 | 33 | 0.03% | 7 | 0.3% |
| backtracked | 6 | 18080 | 14.30% | 1549 | 77.5% |
| init | 7 | 2000 | 1.58% | 2000 | 100.0% |
| full-step(in-resto) | 8 | 180 | 0.14% | 25 | 1.2% |
| SOC(in-resto) | 9 | 1 | 0.00% | 1 | 0.1% |
| backtracked(in-resto) | 14 | 5 | 0.00% | 3 | 0.1% |

Distinct branches active per fused iteration: mean 2.68, median 3, p95 4, max 5; iterations with ≥2 distinct branches: 83.8%, ≥3: 70.2%

## sector180_ws_ir0_b8_dbg_i2  (ir=0; 1784 active slot-iterations, 48 solve segments, 640 fused iterations, batch 8)

| branch | code | slot-iterations | % of iterations | solves with ≥1 | % of solves |
|---|---|---|---|---|---|
| full-step | 0 | 1110 | 62.22% | 48 | 100.0% |
| SOC | 1 | 399 | 22.37% | 42 | 87.5% |
| WD | 2 | 16 | 0.90% | 1 | 2.1% |
| backtracked | 6 | 209 | 11.72% | 20 | 41.7% |
| init | 7 | 48 | 2.69% | 48 | 100.0% |
| full-step(in-resto) | 8 | 2 | 0.11% | 1 | 2.1% |

Distinct branches active per fused iteration: mean 1.29, median 1, p95 2, max 3; iterations with ≥2 distinct branches: 25.6%, ≥3: 3.1%

## sector180_ws_ir100_b1_dbg_i2  (ir=100; 800 active slot-iterations, 48 solve segments, 800 fused iterations, batch 1)

| branch | code | slot-iterations | % of iterations | solves with ≥1 | % of solves |
|---|---|---|---|---|---|
| full-step | 0 | 634 | 79.25% | 48 | 100.0% |
| SOC | 1 | 44 | 5.50% | 40 | 83.3% |
| resto-entry | 5 | 2 | 0.25% | 1 | 2.1% |
| backtracked | 6 | 72 | 9.00% | 21 | 43.8% |
| init | 7 | 48 | 6.00% | 48 | 100.0% |

Distinct branches active per fused iteration: mean 1.00, median 1, p95 1, max 1; iterations with ≥2 distinct branches: 0.0%, ≥3: 0.0%

## sector180_ws_ir100_b32_dbg_i2  (ir=100; 5181 active slot-iterations, 64 solve segments, 1002 fused iterations, batch 32)

| branch | code | slot-iterations | % of iterations | solves with ≥1 | % of solves |
|---|---|---|---|---|---|
| full-step | 0 | 2029 | 39.16% | 64 | 100.0% |
| SOC | 1 | 193 | 3.73% | 60 | 93.8% |
| WD | 2 | 30 | 0.58% | 7 | 10.9% |
| SFR | 4 | 5 | 0.10% | 3 | 4.7% |
| resto-entry | 5 | 280 | 5.40% | 33 | 51.6% |
| backtracked | 6 | 1132 | 21.85% | 60 | 93.8% |
| init | 7 | 64 | 1.24% | 64 | 100.0% |
| full-step(in-resto) | 8 | 1382 | 26.67% | 16 | 25.0% |
| SOC(in-resto) | 9 | 13 | 0.25% | 3 | 4.7% |
| resto-entry(in-resto) | 13 | 12 | 0.23% | 2 | 3.1% |
| backtracked(in-resto) | 14 | 41 | 0.79% | 7 | 10.9% |

Distinct branches active per fused iteration: mean 2.55, median 2, p95 4, max 6; iterations with ≥2 distinct branches: 99.2%, ≥3: 38.3%

## sector180_ws_ir100_b8_dbg_i2  (ir=100; 2313 active slot-iterations, 48 solve segments, 1236 fused iterations, batch 8)

| branch | code | slot-iterations | % of iterations | solves with ≥1 | % of solves |
|---|---|---|---|---|---|
| full-step | 0 | 997 | 43.10% | 48 | 100.0% |
| SOC | 1 | 64 | 2.77% | 38 | 79.2% |
| resto-entry | 5 | 74 | 3.20% | 6 | 12.5% |
| backtracked | 6 | 220 | 9.51% | 25 | 52.1% |
| init | 7 | 48 | 2.08% | 48 | 100.0% |
| full-step(in-resto) | 8 | 427 | 18.46% | 3 | 6.2% |
| SOC(in-resto) | 9 | 2 | 0.09% | 1 | 2.1% |
| resto-entry(in-resto) | 13 | 475 | 20.54% | 1 | 2.1% |
| backtracked(in-resto) | 14 | 6 | 0.26% | 2 | 4.2% |

Distinct branches active per fused iteration: mean 1.31, median 1, p95 2, max 3; iterations with ≥2 distinct branches: 29.9%, ≥3: 0.8%

## sector90_hr_ir0_b500_dbg_iclog  (ir=0; 109500 active slot-iterations, 2493 solve segments, 219 fused iterations, batch 500)

| branch | code | slot-iterations | % of iterations | solves with ≥1 | % of solves |
|---|---|---|---|---|---|
| full-step | 0 | 77105 | 70.42% | 2480 | 99.5% |
| SOC | 1 | 19641 | 17.94% | 2159 | 86.6% |
| SFR | 4 | 16 | 0.01% | 3 | 0.1% |
| resto-entry | 5 | 94 | 0.09% | 20 | 0.8% |
| backtracked | 6 | 10065 | 9.19% | 2042 | 81.9% |
| init | 7 | 2493 | 2.28% | 2493 | 100.0% |
| full-step(in-resto) | 8 | 84 | 0.08% | 17 | 0.7% |
| backtracked(in-resto) | 14 | 2 | 0.00% | 2 | 0.1% |

Distinct branches active per fused iteration: mean 4.58, median 5, p95 6, max 7; iterations with ≥2 distinct branches: 99.5%, ≥3: 99.5%

## sector90_hr_ir0_b500_dbg  (ir=0; 109000 active slot-iterations, 2497 solve segments, 218 fused iterations, batch 500)

| branch | code | slot-iterations | % of iterations | solves with ≥1 | % of solves |
|---|---|---|---|---|---|
| full-step | 0 | 76724 | 70.39% | 2473 | 99.0% |
| SOC | 1 | 19562 | 17.95% | 2142 | 85.8% |
| WD | 2 | 9 | 0.01% | 2 | 0.1% |
| SFR | 4 | 13 | 0.01% | 3 | 0.1% |
| resto-entry | 5 | 121 | 0.11% | 24 | 1.0% |
| backtracked | 6 | 10002 | 9.18% | 2015 | 80.7% |
| init | 7 | 2497 | 2.29% | 2497 | 100.0% |
| full-step(in-resto) | 8 | 71 | 0.07% | 16 | 0.6% |
| backtracked(in-resto) | 14 | 1 | 0.00% | 1 | 0.0% |

Distinct branches active per fused iteration: mean 4.76, median 5, p95 6, max 7; iterations with ≥2 distinct branches: 99.5%, ≥3: 99.5%

## sector90_hr_ir0_b8_dbg_smoke4  (ir=0; 448 active slot-iterations, 30 solve segments, 56 fused iterations, batch 8)

| branch | code | slot-iterations | % of iterations | solves with ≥1 | % of solves |
|---|---|---|---|---|---|
| SOC | 1 | 418 | 93.30% | 28 | 93.3% |
| init | 7 | 30 | 6.70% | 30 | 100.0% |

Distinct branches active per fused iteration: mean 1.20, median 1, p95 2, max 2; iterations with ≥2 distinct branches: 19.6%, ≥3: 0.0%

## sector90_ws_ir0_b1_dbg_i2  (ir=0; 1180 active slot-iterations, 48 solve segments, 1180 fused iterations, batch 1)

| branch | code | slot-iterations | % of iterations | solves with ≥1 | % of solves |
|---|---|---|---|---|---|
| full-step | 0 | 971 | 82.29% | 48 | 100.0% |
| SOC | 1 | 66 | 5.59% | 38 | 79.2% |
| resto-entry | 5 | 17 | 1.44% | 3 | 6.2% |
| backtracked | 6 | 74 | 6.27% | 33 | 68.8% |
| init | 7 | 48 | 4.07% | 48 | 100.0% |
| full-step(in-resto) | 8 | 4 | 0.34% | 1 | 2.1% |

Distinct branches active per fused iteration: mean 1.00, median 1, p95 1, max 1; iterations with ≥2 distinct branches: 0.0%, ≥3: 0.0%

## sector90_ws_ir0_b1_dbg_iclog  (ir=0; 1200 active slot-iterations, 48 solve segments, 1200 fused iterations, batch 1)

| branch | code | slot-iterations | % of iterations | solves with ≥1 | % of solves |
|---|---|---|---|---|---|
| full-step | 0 | 976 | 81.33% | 48 | 100.0% |
| SOC | 1 | 62 | 5.17% | 33 | 68.8% |
| resto-entry | 5 | 11 | 0.92% | 1 | 2.1% |
| backtracked | 6 | 101 | 8.42% | 35 | 72.9% |
| init | 7 | 48 | 4.00% | 48 | 100.0% |
| full-step(in-resto) | 8 | 2 | 0.17% | 1 | 2.1% |

Distinct branches active per fused iteration: mean 1.00, median 1, p95 1, max 1; iterations with ≥2 distinct branches: 0.0%, ≥3: 0.0%

## sector90_ws_ir0_b32_dbg_i2  (ir=0; 3315 active slot-iterations, 64 solve segments, 610 fused iterations, batch 32)

| branch | code | slot-iterations | % of iterations | solves with ≥1 | % of solves |
|---|---|---|---|---|---|
| full-step | 0 | 2249 | 67.84% | 64 | 100.0% |
| SOC | 1 | 607 | 18.31% | 54 | 84.4% |
| SFR | 4 | 2 | 0.06% | 1 | 1.6% |
| resto-entry | 5 | 15 | 0.45% | 1 | 1.6% |
| backtracked | 6 | 377 | 11.37% | 57 | 89.1% |
| init | 7 | 64 | 1.93% | 64 | 100.0% |
| full-step(in-resto) | 8 | 1 | 0.03% | 1 | 1.6% |

Distinct branches active per fused iteration: mean 1.58, median 1, p95 3, max 3; iterations with ≥2 distinct branches: 40.8%, ≥3: 17.2%

## sector90_ws_ir0_b32_dbg_iclog  (ir=0; 4978 active slot-iterations, 96 solve segments, 582 fused iterations, batch 32)

| branch | code | slot-iterations | % of iterations | solves with ≥1 | % of solves |
|---|---|---|---|---|---|
| full-step | 0 | 3329 | 66.87% | 96 | 100.0% |
| SOC | 1 | 1017 | 20.43% | 78 | 81.2% |
| WD | 2 | 2 | 0.04% | 1 | 1.0% |
| resto-entry | 5 | 5 | 0.10% | 2 | 2.1% |
| backtracked | 6 | 524 | 10.53% | 79 | 82.3% |
| init | 7 | 96 | 1.93% | 96 | 100.0% |
| full-step(in-resto) | 8 | 5 | 0.10% | 2 | 2.1% |

Distinct branches active per fused iteration: mean 1.91, median 2, p95 3, max 4; iterations with ≥2 distinct branches: 55.8%, ≥3: 33.5%

## sector90_ws_ir0_b500_dbg  (ir=0; 119010 active slot-iterations, 2000 solve segments, 1418 fused iterations, batch 500)

| branch | code | slot-iterations | % of iterations | solves with ≥1 | % of solves |
|---|---|---|---|---|---|
| full-step | 0 | 75177 | 63.17% | 2000 | 100.0% |
| SOC | 1 | 24999 | 21.01% | 1809 | 90.5% |
| WD | 2 | 216 | 0.18% | 30 | 1.5% |
| SFR | 4 | 8 | 0.01% | 5 | 0.2% |
| resto-entry | 5 | 82 | 0.07% | 20 | 1.0% |
| backtracked | 6 | 15936 | 13.39% | 1722 | 86.1% |
| init | 7 | 2000 | 1.68% | 2000 | 100.0% |
| full-step(in-resto) | 8 | 579 | 0.49% | 24 | 1.2% |
| backtracked(in-resto) | 14 | 13 | 0.01% | 12 | 0.6% |

Distinct branches active per fused iteration: mean 2.97, median 3, p95 5, max 5; iterations with ≥2 distinct branches: 87.2%, ≥3: 78.6%

## sector90_ws_ir0_b8_dbg_i2  (ir=0; 1222 active slot-iterations, 48 solve segments, 286 fused iterations, batch 8)

| branch | code | slot-iterations | % of iterations | solves with ≥1 | % of solves |
|---|---|---|---|---|---|
| full-step | 0 | 986 | 80.69% | 48 | 100.0% |
| SOC | 1 | 107 | 8.76% | 41 | 85.4% |
| resto-entry | 5 | 1 | 0.08% | 1 | 2.1% |
| backtracked | 6 | 80 | 6.55% | 32 | 66.7% |
| init | 7 | 48 | 3.93% | 48 | 100.0% |

Distinct branches active per fused iteration: mean 1.29, median 1, p95 3, max 3; iterations with ≥2 distinct branches: 22.7%, ≥3: 5.9%

## sector90_ws_ir0_b8_dbg_smoke5  (ir=0; 341 active slot-iterations, 16 solve segments, 50 fused iterations, batch 8)

| branch | code | slot-iterations | % of iterations | solves with ≥1 | % of solves |
|---|---|---|---|---|---|
| full-step | 0 | 304 | 89.15% | 16 | 100.0% |
| SOC | 1 | 11 | 3.23% | 10 | 62.5% |
| backtracked | 6 | 10 | 2.93% | 9 | 56.2% |
| init | 7 | 16 | 4.69% | 16 | 100.0% |

Distinct branches active per fused iteration: mean 1.26, median 1, p95 2, max 3; iterations with ≥2 distinct branches: 22.0%, ≥3: 4.0%

## sector90_ws_ir0_b8_dbg_smoke6  (ir=0; 346 active slot-iterations, 16 solve segments, 56 fused iterations, batch 8)

| branch | code | slot-iterations | % of iterations | solves with ≥1 | % of solves |
|---|---|---|---|---|---|
| full-step | 0 | 300 | 86.71% | 16 | 100.0% |
| SOC | 1 | 16 | 4.62% | 11 | 68.8% |
| backtracked | 6 | 14 | 4.05% | 10 | 62.5% |
| init | 7 | 16 | 4.62% | 16 | 100.0% |

Distinct branches active per fused iteration: mean 1.39, median 1, p95 2, max 3; iterations with ≥2 distinct branches: 35.7%, ≥3: 3.6%

## sector90_ws_ir100_b1_dbg_i2  (ir=100; 1552 active slot-iterations, 48 solve segments, 1552 fused iterations, batch 1)

| branch | code | slot-iterations | % of iterations | solves with ≥1 | % of solves |
|---|---|---|---|---|---|
| full-step | 0 | 1269 | 81.77% | 48 | 100.0% |
| SOC | 1 | 73 | 4.70% | 39 | 81.2% |
| resto-entry | 5 | 25 | 1.61% | 2 | 4.2% |
| backtracked | 6 | 103 | 6.64% | 35 | 72.9% |
| init | 7 | 48 | 3.09% | 48 | 100.0% |
| full-step(in-resto) | 8 | 34 | 2.19% | 2 | 4.2% |

Distinct branches active per fused iteration: mean 1.00, median 1, p95 1, max 1; iterations with ≥2 distinct branches: 0.0%, ≥3: 0.0%

## sector90_ws_ir100_b32_dbg_i2  (ir=100; 4866 active slot-iterations, 64 solve segments, 1002 fused iterations, batch 32)

| branch | code | slot-iterations | % of iterations | solves with ≥1 | % of solves |
|---|---|---|---|---|---|
| full-step | 0 | 2022 | 41.55% | 64 | 100.0% |
| SOC | 1 | 189 | 3.88% | 56 | 87.5% |
| WD | 2 | 15 | 0.31% | 3 | 4.7% |
| SFR | 4 | 3 | 0.06% | 3 | 4.7% |
| resto-entry | 5 | 278 | 5.71% | 41 | 64.1% |
| backtracked | 6 | 903 | 18.56% | 64 | 100.0% |
| init | 7 | 64 | 1.32% | 64 | 100.0% |
| full-step(in-resto) | 8 | 1348 | 27.70% | 21 | 32.8% |
| SOC(in-resto) | 9 | 6 | 0.12% | 3 | 4.7% |
| resto-entry(in-resto) | 13 | 5 | 0.10% | 1 | 1.6% |
| backtracked(in-resto) | 14 | 33 | 0.68% | 7 | 10.9% |

Distinct branches active per fused iteration: mean 2.01, median 2, p95 4, max 6; iterations with ≥2 distinct branches: 51.5%, ≥3: 28.6%

## sector90_ws_ir100_b8_dbg_i2  (ir=100; 1299 active slot-iterations, 48 solve segments, 336 fused iterations, batch 8)

| branch | code | slot-iterations | % of iterations | solves with ≥1 | % of solves |
|---|---|---|---|---|---|
| full-step | 0 | 926 | 71.29% | 48 | 100.0% |
| SOC | 1 | 75 | 5.77% | 43 | 89.6% |
| WD | 2 | 4 | 0.31% | 1 | 2.1% |
| resto-entry | 5 | 20 | 1.54% | 10 | 20.8% |
| backtracked | 6 | 226 | 17.40% | 42 | 87.5% |
| init | 7 | 48 | 3.70% | 48 | 100.0% |

Distinct branches active per fused iteration: mean 1.49, median 1, p95 3, max 4; iterations with ≥2 distinct branches: 36.3%, ≥3: 11.6%

## v1.0_hr_ir0_b250_dbg  (ir=0; 53750 active slot-iterations, 1248 solve segments, 215 fused iterations, batch 250)

| branch | code | slot-iterations | % of iterations | solves with ≥1 | % of solves |
|---|---|---|---|---|---|
| full-step | 0 | 45165 | 84.03% | 1242 | 99.5% |
| SOC | 1 | 5227 | 9.72% | 735 | 58.9% |
| WD | 2 | 8 | 0.01% | 2 | 0.2% |
| SFR | 4 | 2 | 0.00% | 1 | 0.1% |
| resto-entry | 5 | 2 | 0.00% | 2 | 0.2% |
| backtracked | 6 | 2086 | 3.88% | 545 | 43.7% |
| init | 7 | 1248 | 2.32% | 1248 | 100.0% |
| full-step(in-resto) | 8 | 12 | 0.02% | 2 | 0.2% |

Distinct branches active per fused iteration: mean 3.99, median 4, p95 5, max 6; iterations with ≥2 distinct branches: 99.5%, ≥3: 98.6%

## v1.0_ws_ir0_b1_dbg_i2  (ir=0; 1629 active slot-iterations, 48 solve segments, 1629 fused iterations, batch 1)

| branch | code | slot-iterations | % of iterations | solves with ≥1 | % of solves |
|---|---|---|---|---|---|
| full-step | 0 | 1506 | 92.45% | 48 | 100.0% |
| SOC | 1 | 34 | 2.09% | 24 | 50.0% |
| resto-entry | 5 | 3 | 0.18% | 1 | 2.1% |
| backtracked | 6 | 27 | 1.66% | 17 | 35.4% |
| init | 7 | 48 | 2.95% | 48 | 100.0% |
| full-step(in-resto) | 8 | 11 | 0.68% | 1 | 2.1% |

Distinct branches active per fused iteration: mean 1.00, median 1, p95 1, max 1; iterations with ≥2 distinct branches: 0.0%, ≥3: 0.0%

## v1.0_ws_ir0_b250_dbg  (ir=0; 58010 active slot-iterations, 1000 solve segments, 2004 fused iterations, batch 250)

| branch | code | slot-iterations | % of iterations | solves with ≥1 | % of solves |
|---|---|---|---|---|---|
| full-step | 0 | 42834 | 73.84% | 1000 | 100.0% |
| SOC | 1 | 8030 | 13.84% | 652 | 65.2% |
| WD | 2 | 182 | 0.31% | 15 | 1.5% |
| SFR | 4 | 3 | 0.01% | 1 | 0.1% |
| resto-entry | 5 | 11 | 0.02% | 6 | 0.6% |
| backtracked | 6 | 4925 | 8.49% | 501 | 50.1% |
| init | 7 | 1000 | 1.72% | 1000 | 100.0% |
| full-step(in-resto) | 8 | 1018 | 1.75% | 13 | 1.3% |
| backtracked(in-resto) | 14 | 7 | 0.01% | 6 | 0.6% |

Distinct branches active per fused iteration: mean 2.69, median 3, p95 4, max 5; iterations with ≥2 distinct branches: 87.2%, ≥3: 66.0%

## v1.0_ws_ir0_b32_dbg_i2  (ir=0; 4673 active slot-iterations, 64 solve segments, 653 fused iterations, batch 32)

| branch | code | slot-iterations | % of iterations | solves with ≥1 | % of solves |
|---|---|---|---|---|---|
| full-step | 0 | 2930 | 62.70% | 64 | 100.0% |
| SOC | 1 | 988 | 21.14% | 44 | 68.8% |
| WD | 2 | 8 | 0.17% | 2 | 3.1% |
| resto-entry | 5 | 3 | 0.06% | 2 | 3.1% |
| backtracked | 6 | 673 | 14.40% | 34 | 53.1% |
| init | 7 | 64 | 1.37% | 64 | 100.0% |
| full-step(in-resto) | 8 | 7 | 0.15% | 1 | 1.6% |

Distinct branches active per fused iteration: mean 2.06, median 2, p95 3, max 4; iterations with ≥2 distinct branches: 71.1%, ≥3: 34.3%

## v1.0_ws_ir0_b8_dbg_i2  (ir=0; 1733 active slot-iterations, 48 solve segments, 334 fused iterations, batch 8)

| branch | code | slot-iterations | % of iterations | solves with ≥1 | % of solves |
|---|---|---|---|---|---|
| full-step | 0 | 1577 | 91.00% | 48 | 100.0% |
| SOC | 1 | 68 | 3.92% | 17 | 35.4% |
| resto-entry | 5 | 1 | 0.06% | 1 | 2.1% |
| backtracked | 6 | 27 | 1.56% | 19 | 39.6% |
| init | 7 | 48 | 2.77% | 48 | 100.0% |
| full-step(in-resto) | 8 | 12 | 0.69% | 1 | 2.1% |

Distinct branches active per fused iteration: mean 1.30, median 1, p95 2, max 3; iterations with ≥2 distinct branches: 29.3%, ≥3: 0.6%

## v1.0_ws_ir100_b1_dbg_i2  (ir=100; 1559 active slot-iterations, 48 solve segments, 1559 fused iterations, batch 1)

| branch | code | slot-iterations | % of iterations | solves with ≥1 | % of solves |
|---|---|---|---|---|---|
| full-step | 0 | 1455 | 93.33% | 48 | 100.0% |
| SOC | 1 | 25 | 1.60% | 17 | 35.4% |
| resto-entry | 5 | 1 | 0.06% | 1 | 2.1% |
| backtracked | 6 | 30 | 1.92% | 19 | 39.6% |
| init | 7 | 48 | 3.08% | 48 | 100.0% |

Distinct branches active per fused iteration: mean 1.00, median 1, p95 1, max 1; iterations with ≥2 distinct branches: 0.0%, ≥3: 0.0%

## v1.0_ws_ir100_b32_dbg_i2  (ir=100; 4543 active slot-iterations, 64 solve segments, 746 fused iterations, batch 32)

| branch | code | slot-iterations | % of iterations | solves with ≥1 | % of solves |
|---|---|---|---|---|---|
| full-step | 0 | 2338 | 51.46% | 64 | 100.0% |
| SOC | 1 | 97 | 2.14% | 43 | 67.2% |
| WD | 2 | 52 | 1.14% | 11 | 17.2% |
| SFR | 4 | 3 | 0.07% | 2 | 3.1% |
| resto-entry | 5 | 217 | 4.78% | 40 | 62.5% |
| backtracked | 6 | 1710 | 37.64% | 63 | 98.4% |
| init | 7 | 64 | 1.41% | 64 | 100.0% |
| full-step(in-resto) | 8 | 44 | 0.97% | 8 | 12.5% |
| SOC(in-resto) | 9 | 3 | 0.07% | 1 | 1.6% |
| resto-entry(in-resto) | 13 | 4 | 0.09% | 2 | 3.1% |
| backtracked(in-resto) | 14 | 11 | 0.24% | 5 | 7.8% |

Distinct branches active per fused iteration: mean 1.78, median 1, p95 4, max 6; iterations with ≥2 distinct branches: 43.4%, ≥3: 24.0%

## v1.0_ws_ir100_b8_dbg_i2  (ir=100; 1757 active slot-iterations, 48 solve segments, 371 fused iterations, batch 8)

| branch | code | slot-iterations | % of iterations | solves with ≥1 | % of solves |
|---|---|---|---|---|---|
| full-step | 0 | 1503 | 85.54% | 48 | 100.0% |
| SOC | 1 | 34 | 1.94% | 22 | 45.8% |
| resto-entry | 5 | 20 | 1.14% | 14 | 29.2% |
| backtracked | 6 | 152 | 8.65% | 38 | 79.2% |
| init | 7 | 48 | 2.73% | 48 | 100.0% |

Distinct branches active per fused iteration: mean 1.42, median 1, p95 2, max 3; iterations with ≥2 distinct branches: 38.0%, ≥3: 3.5%

## v1.5_hr_ir0_b250_dbg  (ir=0; 60500 active slot-iterations, 1247 solve segments, 242 fused iterations, batch 250)

| branch | code | slot-iterations | % of iterations | solves with ≥1 | % of solves |
|---|---|---|---|---|---|
| full-step | 0 | 48082 | 79.47% | 1241 | 99.5% |
| SOC | 1 | 5873 | 9.71% | 717 | 57.5% |
| WD | 2 | 74 | 0.12% | 8 | 0.6% |
| TS | 3 | 1445 | 2.39% | 14 | 1.1% |
| SFR | 4 | 8 | 0.01% | 2 | 0.2% |
| resto-entry | 5 | 17 | 0.03% | 9 | 0.7% |
| backtracked | 6 | 3700 | 6.12% | 658 | 52.8% |
| init | 7 | 1247 | 2.06% | 1247 | 100.0% |
| full-step(in-resto) | 8 | 53 | 0.09% | 4 | 0.3% |
| backtracked(in-resto) | 14 | 1 | 0.00% | 1 | 0.1% |

Distinct branches active per fused iteration: mean 5.38, median 5, p95 7, max 8; iterations with ≥2 distinct branches: 99.6%, ≥3: 99.6%

## v1.5_ws_ir0_b1_dbg_i2  (ir=0; 3552 active slot-iterations, 48 solve segments, 3552 fused iterations, batch 1)

| branch | code | slot-iterations | % of iterations | solves with ≥1 | % of solves |
|---|---|---|---|---|---|
| full-step | 0 | 3261 | 91.81% | 48 | 100.0% |
| SOC | 1 | 38 | 1.07% | 23 | 47.9% |
| WD | 2 | 29 | 0.82% | 2 | 4.2% |
| TS | 3 | 11 | 0.31% | 3 | 6.2% |
| resto-entry | 5 | 5 | 0.14% | 2 | 4.2% |
| backtracked | 6 | 159 | 4.48% | 24 | 50.0% |
| init | 7 | 48 | 1.35% | 48 | 100.0% |
| full-step(in-resto) | 8 | 1 | 0.03% | 1 | 2.1% |

Distinct branches active per fused iteration: mean 1.00, median 1, p95 1, max 1; iterations with ≥2 distinct branches: 0.0%, ≥3: 0.0%

## v1.5_ws_ir0_b250_dbg  (ir=0; 70051 active slot-iterations, 1000 solve segments, 2004 fused iterations, batch 250)

| branch | code | slot-iterations | % of iterations | solves with ≥1 | % of solves |
|---|---|---|---|---|---|
| full-step | 0 | 48051 | 68.59% | 1000 | 100.0% |
| SOC | 1 | 7668 | 10.95% | 598 | 59.8% |
| WD | 2 | 274 | 0.39% | 26 | 2.6% |
| TS | 3 | 4000 | 5.71% | 18 | 1.8% |
| SFR | 4 | 10 | 0.01% | 4 | 0.4% |
| resto-entry | 5 | 29 | 0.04% | 15 | 1.5% |
| backtracked | 6 | 6758 | 9.65% | 564 | 56.4% |
| init | 7 | 1000 | 1.43% | 1000 | 100.0% |
| full-step(in-resto) | 8 | 1938 | 2.77% | 17 | 1.7% |
| SOC(in-resto) | 9 | 1 | 0.00% | 1 | 0.1% |
| resto-entry(in-resto) | 13 | 316 | 0.45% | 2 | 0.2% |
| backtracked(in-resto) | 14 | 6 | 0.01% | 4 | 0.4% |

Distinct branches active per fused iteration: mean 4.04, median 4, p95 6, max 7; iterations with ≥2 distinct branches: 99.8%, ≥3: 95.3%

## v1.5_ws_ir0_b32_dbg_i2  (ir=0; 4319 active slot-iterations, 64 solve segments, 866 fused iterations, batch 32)

| branch | code | slot-iterations | % of iterations | solves with ≥1 | % of solves |
|---|---|---|---|---|---|
| full-step | 0 | 3602 | 83.40% | 64 | 100.0% |
| SOC | 1 | 388 | 8.98% | 40 | 62.5% |
| WD | 2 | 2 | 0.05% | 1 | 1.6% |
| TS | 3 | 3 | 0.07% | 1 | 1.6% |
| resto-entry | 5 | 1 | 0.02% | 1 | 1.6% |
| backtracked | 6 | 251 | 5.81% | 31 | 48.4% |
| init | 7 | 64 | 1.48% | 64 | 100.0% |
| full-step(in-resto) | 8 | 8 | 0.19% | 1 | 1.6% |

Distinct branches active per fused iteration: mean 1.46, median 1, p95 3, max 4; iterations with ≥2 distinct branches: 31.5%, ≥3: 14.4%

## v1.5_ws_ir0_b8_dbg_i2  (ir=0; 3642 active slot-iterations, 48 solve segments, 1218 fused iterations, batch 8)

| branch | code | slot-iterations | % of iterations | solves with ≥1 | % of solves |
|---|---|---|---|---|---|
| full-step | 0 | 1889 | 51.87% | 48 | 100.0% |
| SOC | 1 | 43 | 1.18% | 24 | 50.0% |
| TS | 3 | 1557 | 42.75% | 4 | 8.3% |
| resto-entry | 5 | 30 | 0.82% | 2 | 4.2% |
| backtracked | 6 | 75 | 2.06% | 25 | 52.1% |
| init | 7 | 48 | 1.32% | 48 | 100.0% |

Distinct branches active per fused iteration: mean 1.37, median 1, p95 2, max 3; iterations with ≥2 distinct branches: 34.8%, ≥3: 2.0%

## v1.5_ws_ir100_b1_dbg_i2  (ir=100; 2539 active slot-iterations, 48 solve segments, 2539 fused iterations, batch 1)

| branch | code | slot-iterations | % of iterations | solves with ≥1 | % of solves |
|---|---|---|---|---|---|
| full-step | 0 | 2135 | 84.09% | 48 | 100.0% |
| SOC | 1 | 44 | 1.73% | 18 | 37.5% |
| WD | 2 | 1 | 0.04% | 1 | 2.1% |
| TS | 3 | 198 | 7.80% | 1 | 2.1% |
| resto-entry | 5 | 4 | 0.16% | 2 | 4.2% |
| backtracked | 6 | 63 | 2.48% | 21 | 43.8% |
| init | 7 | 48 | 1.89% | 48 | 100.0% |
| full-step(in-resto) | 8 | 41 | 1.61% | 1 | 2.1% |
| backtracked(in-resto) | 14 | 5 | 0.20% | 1 | 2.1% |

Distinct branches active per fused iteration: mean 1.00, median 1, p95 1, max 1; iterations with ≥2 distinct branches: 0.0%, ≥3: 0.0%

## v1.5_ws_ir100_b32_dbg_i2  (ir=100; 3765 active slot-iterations, 64 solve segments, 629 fused iterations, batch 32)

| branch | code | slot-iterations | % of iterations | solves with ≥1 | % of solves |
|---|---|---|---|---|---|
| full-step | 0 | 2565 | 68.13% | 64 | 100.0% |
| SOC | 1 | 92 | 2.44% | 43 | 67.2% |
| WD | 2 | 12 | 0.32% | 4 | 6.2% |
| TS | 3 | 108 | 2.87% | 1 | 1.6% |
| SFR | 4 | 2 | 0.05% | 2 | 3.1% |
| resto-entry | 5 | 74 | 1.97% | 32 | 50.0% |
| backtracked | 6 | 675 | 17.93% | 58 | 90.6% |
| init | 7 | 64 | 1.70% | 64 | 100.0% |
| full-step(in-resto) | 8 | 33 | 0.88% | 6 | 9.4% |
| SOC(in-resto) | 9 | 1 | 0.03% | 1 | 1.6% |
| resto-entry(in-resto) | 13 | 135 | 3.59% | 3 | 4.7% |
| backtracked(in-resto) | 14 | 4 | 0.11% | 3 | 4.7% |

Distinct branches active per fused iteration: mean 1.55, median 1, p95 4, max 5; iterations with ≥2 distinct branches: 32.1%, ≥3: 16.1%

## v1.5_ws_ir100_b8_dbg_i2  (ir=100; 2695 active slot-iterations, 48 solve segments, 1229 fused iterations, batch 8)

| branch | code | slot-iterations | % of iterations | solves with ≥1 | % of solves |
|---|---|---|---|---|---|
| full-step | 0 | 1814 | 67.31% | 48 | 100.0% |
| SOC | 1 | 42 | 1.56% | 23 | 47.9% |
| WD | 2 | 90 | 3.34% | 2 | 4.2% |
| resto-entry | 5 | 10 | 0.37% | 7 | 14.6% |
| backtracked | 6 | 456 | 16.92% | 33 | 68.8% |
| init | 7 | 48 | 1.78% | 48 | 100.0% |
| full-step(in-resto) | 8 | 147 | 5.45% | 1 | 2.1% |
| SOC(in-resto) | 9 | 13 | 0.48% | 1 | 2.1% |
| resto-entry(in-resto) | 13 | 23 | 0.85% | 1 | 2.1% |
| backtracked(in-resto) | 14 | 52 | 1.93% | 1 | 2.1% |

Distinct branches active per fused iteration: mean 1.13, median 1, p95 2, max 4; iterations with ≥2 distinct branches: 11.2%, ≥3: 1.9%

## v2.0_hr_ir0_b250_dbg  (ir=0; 65250 active slot-iterations, 1247 solve segments, 261 fused iterations, batch 250)

| branch | code | slot-iterations | % of iterations | solves with ≥1 | % of solves |
|---|---|---|---|---|---|
| full-step | 0 | 58117 | 89.07% | 1241 | 99.5% |
| SOC | 1 | 924 | 1.42% | 449 | 36.0% |
| WD | 2 | 271 | 0.42% | 17 | 1.4% |
| TS | 3 | 1913 | 2.93% | 38 | 3.0% |
| SFR | 4 | 12 | 0.02% | 5 | 0.4% |
| resto-entry | 5 | 44 | 0.07% | 26 | 2.1% |
| backtracked | 6 | 2652 | 4.06% | 524 | 42.0% |
| init | 7 | 1247 | 1.91% | 1247 | 100.0% |
| full-step(in-resto) | 8 | 70 | 0.11% | 15 | 1.2% |

Distinct branches active per fused iteration: mean 5.70, median 6, p95 7, max 8; iterations with ≥2 distinct branches: 99.6%, ≥3: 98.9%

## v2.0_ws_ir0_b1_dbg_i2  (ir=0; 19481 active slot-iterations, 48 solve segments, 19481 fused iterations, batch 1)

| branch | code | slot-iterations | % of iterations | solves with ≥1 | % of solves |
|---|---|---|---|---|---|
| full-step | 0 | 17518 | 89.92% | 48 | 100.0% |
| SOC | 1 | 38 | 0.20% | 21 | 43.8% |
| WD | 2 | 169 | 0.87% | 5 | 10.4% |
| TS | 3 | 766 | 3.93% | 17 | 35.4% |
| resto-entry | 5 | 22 | 0.11% | 10 | 20.8% |
| backtracked | 6 | 875 | 4.49% | 29 | 60.4% |
| init | 7 | 48 | 0.25% | 48 | 100.0% |
| full-step(in-resto) | 8 | 45 | 0.23% | 3 | 6.2% |

Distinct branches active per fused iteration: mean 1.00, median 1, p95 1, max 1; iterations with ≥2 distinct branches: 0.0%, ≥3: 0.0%

## v2.0_ws_ir0_b250_dbg  (ir=0; 112731 active slot-iterations, 1000 solve segments, 2004 fused iterations, batch 250)

| branch | code | slot-iterations | % of iterations | solves with ≥1 | % of solves |
|---|---|---|---|---|---|
| full-step | 0 | 93814 | 83.22% | 1000 | 100.0% |
| SOC | 1 | 775 | 0.69% | 416 | 41.6% |
| WD | 2 | 1446 | 1.28% | 26 | 2.6% |
| TS | 3 | 7847 | 6.96% | 93 | 9.3% |
| SFR | 4 | 7 | 0.01% | 3 | 0.3% |
| resto-entry | 5 | 102 | 0.09% | 46 | 4.6% |
| backtracked | 6 | 7670 | 6.80% | 494 | 49.4% |
| init | 7 | 1000 | 0.89% | 1000 | 100.0% |
| full-step(in-resto) | 8 | 69 | 0.06% | 12 | 1.2% |
| backtracked(in-resto) | 14 | 1 | 0.00% | 1 | 0.1% |

Distinct branches active per fused iteration: mean 3.68, median 4, p95 5, max 6; iterations with ≥2 distinct branches: 99.8%, ≥3: 99.4%

## v2.0_ws_ir0_b250_dbg_sub250  (ir=0; 28815 active slot-iterations, 250 solve segments, 501 fused iterations, batch 250)

| branch | code | slot-iterations | % of iterations | solves with ≥1 | % of solves |
|---|---|---|---|---|---|
| full-step | 0 | 23472 | 81.46% | 250 | 100.0% |
| SOC | 1 | 196 | 0.68% | 99 | 39.6% |
| WD | 2 | 330 | 1.15% | 8 | 3.2% |
| TS | 3 | 2681 | 9.30% | 22 | 8.8% |
| SFR | 4 | 6 | 0.02% | 2 | 0.8% |
| resto-entry | 5 | 22 | 0.08% | 11 | 4.4% |
| backtracked | 6 | 1827 | 6.34% | 113 | 45.2% |
| init | 7 | 250 | 0.87% | 250 | 100.0% |
| full-step(in-resto) | 8 | 31 | 0.11% | 3 | 1.2% |

Distinct branches active per fused iteration: mean 3.76, median 4, p95 5, max 6; iterations with ≥2 distinct branches: 99.8%, ≥3: 99.2%

## v2.0_ws_ir0_b32_dbg_i2  (ir=0; 9893 active slot-iterations, 64 solve segments, 1002 fused iterations, batch 32)

| branch | code | slot-iterations | % of iterations | solves with ≥1 | % of solves |
|---|---|---|---|---|---|
| full-step | 0 | 8440 | 85.31% | 64 | 100.0% |
| SOC | 1 | 26 | 0.26% | 17 | 26.6% |
| WD | 2 | 89 | 0.90% | 2 | 3.1% |
| TS | 3 | 776 | 7.84% | 10 | 15.6% |
| resto-entry | 5 | 8 | 0.08% | 4 | 6.2% |
| backtracked | 6 | 478 | 4.83% | 31 | 48.4% |
| init | 7 | 64 | 0.65% | 64 | 100.0% |
| full-step(in-resto) | 8 | 12 | 0.12% | 1 | 1.6% |

Distinct branches active per fused iteration: mean 2.31, median 2, p95 3, max 4; iterations with ≥2 distinct branches: 92.6%, ≥3: 38.0%

## v2.0_ws_ir0_b48_dbg_sub48  (ir=0; 5052 active slot-iterations, 48 solve segments, 501 fused iterations, batch 48)

| branch | code | slot-iterations | % of iterations | solves with ≥1 | % of solves |
|---|---|---|---|---|---|
| full-step | 0 | 4532 | 89.71% | 48 | 100.0% |
| SOC | 1 | 28 | 0.55% | 18 | 37.5% |
| WD | 2 | 60 | 1.19% | 1 | 2.1% |
| TS | 3 | 19 | 0.38% | 3 | 6.2% |
| resto-entry | 5 | 1 | 0.02% | 1 | 2.1% |
| backtracked | 6 | 364 | 7.21% | 24 | 50.0% |
| init | 7 | 48 | 0.95% | 48 | 100.0% |

Distinct branches active per fused iteration: mean 1.89, median 2, p95 2, max 4; iterations with ≥2 distinct branches: 85.6%, ≥3: 3.4%

## v2.0_ws_ir0_b8_dbg_i2  (ir=0; 19012 active slot-iterations, 48 solve segments, 3006 fused iterations, batch 8)

| branch | code | slot-iterations | % of iterations | solves with ≥1 | % of solves |
|---|---|---|---|---|---|
| full-step | 0 | 16419 | 86.36% | 48 | 100.0% |
| SOC | 1 | 44 | 0.23% | 21 | 43.8% |
| WD | 2 | 308 | 1.62% | 4 | 8.3% |
| TS | 3 | 552 | 2.90% | 21 | 43.8% |
| SFR | 4 | 3 | 0.02% | 1 | 2.1% |
| resto-entry | 5 | 37 | 0.19% | 9 | 18.8% |
| backtracked | 6 | 1587 | 8.35% | 36 | 75.0% |
| init | 7 | 48 | 0.25% | 48 | 100.0% |
| full-step(in-resto) | 8 | 14 | 0.07% | 3 | 6.2% |

Distinct branches active per fused iteration: mean 1.70, median 2, p95 3, max 4; iterations with ≥2 distinct branches: 64.4%, ≥3: 5.9%

## v2.0_ws_ir100_b1_dbg_i2  (ir=100; 19006 active slot-iterations, 48 solve segments, 19006 fused iterations, batch 1)

| branch | code | slot-iterations | % of iterations | solves with ≥1 | % of solves |
|---|---|---|---|---|---|
| full-step | 0 | 11896 | 62.59% | 48 | 100.0% |
| SOC | 1 | 35 | 0.18% | 22 | 45.8% |
| WD | 2 | 1330 | 7.00% | 29 | 60.4% |
| TS | 3 | 16 | 0.08% | 4 | 8.3% |
| SFR | 4 | 10 | 0.05% | 1 | 2.1% |
| resto-entry | 5 | 41 | 0.22% | 13 | 27.1% |
| backtracked | 6 | 5476 | 28.81% | 36 | 75.0% |
| init | 7 | 48 | 0.25% | 48 | 100.0% |
| full-step(in-resto) | 8 | 136 | 0.72% | 9 | 18.8% |
| SOC(in-resto) | 9 | 1 | 0.01% | 1 | 2.1% |
| backtracked(in-resto) | 14 | 17 | 0.09% | 5 | 10.4% |

Distinct branches active per fused iteration: mean 1.00, median 1, p95 1, max 1; iterations with ≥2 distinct branches: 0.0%, ≥3: 0.0%

## v2.0_ws_ir100_b32_dbg_i2  (ir=100; 9251 active slot-iterations, 64 solve segments, 1002 fused iterations, batch 32)

| branch | code | slot-iterations | % of iterations | solves with ≥1 | % of solves |
|---|---|---|---|---|---|
| full-step | 0 | 7051 | 76.22% | 64 | 100.0% |
| SOC | 1 | 82 | 0.89% | 30 | 46.9% |
| WD | 2 | 299 | 3.23% | 5 | 7.8% |
| TS | 3 | 32 | 0.35% | 2 | 3.1% |
| SFR | 4 | 41 | 0.44% | 4 | 6.2% |
| resto-entry | 5 | 72 | 0.78% | 9 | 14.1% |
| backtracked | 6 | 1329 | 14.37% | 44 | 68.8% |
| init | 7 | 64 | 0.69% | 64 | 100.0% |
| full-step(in-resto) | 8 | 235 | 2.54% | 5 | 7.8% |
| SOC(in-resto) | 9 | 10 | 0.11% | 1 | 1.6% |
| resto-entry(in-resto) | 13 | 10 | 0.11% | 2 | 3.1% |
| backtracked(in-resto) | 14 | 26 | 0.28% | 1 | 1.6% |

Distinct branches active per fused iteration: mean 2.49, median 3, p95 3, max 4; iterations with ≥2 distinct branches: 92.9%, ≥3: 52.2%

## v2.0_ws_ir100_b8_dbg_i2  (ir=100; 19291 active slot-iterations, 48 solve segments, 3006 fused iterations, batch 8)

| branch | code | slot-iterations | % of iterations | solves with ≥1 | % of solves |
|---|---|---|---|---|---|
| full-step | 0 | 11004 | 57.04% | 48 | 100.0% |
| SOC | 1 | 101 | 0.52% | 24 | 50.0% |
| WD | 2 | 1359 | 7.04% | 27 | 56.2% |
| TS | 3 | 23 | 0.12% | 3 | 6.2% |
| SFR | 4 | 12 | 0.06% | 3 | 6.2% |
| resto-entry | 5 | 129 | 0.67% | 16 | 33.3% |
| backtracked | 6 | 5784 | 29.98% | 36 | 75.0% |
| init | 7 | 48 | 0.25% | 48 | 100.0% |
| full-step(in-resto) | 8 | 159 | 0.82% | 10 | 20.8% |
| resto-entry(in-resto) | 13 | 662 | 3.43% | 3 | 6.2% |
| backtracked(in-resto) | 14 | 10 | 0.05% | 4 | 8.3% |

Distinct branches active per fused iteration: mean 2.46, median 2, p95 4, max 5; iterations with ≥2 distinct branches: 83.9%, ≥3: 49.6%

## 2_hr_ir0_b75_dbg  (ir=0; 14250 active slot-iterations, 149 solve segments, 190 fused iterations, batch 75)

| branch | code | slot-iterations | % of iterations | solves with ≥1 | % of solves |
|---|---|---|---|---|---|
| full-step | 0 | 13267 | 93.10% | 149 | 100.0% |
| SOC | 1 | 504 | 3.54% | 141 | 94.6% |
| backtracked | 6 | 330 | 2.32% | 127 | 85.2% |
| init | 7 | 149 | 1.05% | 149 | 100.0% |

Distinct branches active per fused iteration: mean 2.57, median 3, p95 4, max 4; iterations with ≥2 distinct branches: 84.7%, ≥3: 63.2%

## 2_ws_ir0_b1_dbg_i2  (ir=0; 5000 active slot-iterations, 48 solve segments, 5000 fused iterations, batch 1)

| branch | code | slot-iterations | % of iterations | solves with ≥1 | % of solves |
|---|---|---|---|---|---|
| full-step | 0 | 4665 | 93.30% | 48 | 100.0% |
| SOC | 1 | 121 | 2.42% | 43 | 89.6% |
| SFR | 4 | 2 | 0.04% | 1 | 2.1% |
| resto-entry | 5 | 6 | 0.12% | 4 | 8.3% |
| backtracked | 6 | 158 | 3.16% | 45 | 93.8% |
| init | 7 | 48 | 0.96% | 48 | 100.0% |

Distinct branches active per fused iteration: mean 1.00, median 1, p95 1, max 1; iterations with ≥2 distinct branches: 0.0%, ≥3: 0.0%

## 2_ws_ir0_b32_dbg_i2  (ir=0; 10874 active slot-iterations, 64 solve segments, 1002 fused iterations, batch 32)

| branch | code | slot-iterations | % of iterations | solves with ≥1 | % of solves |
|---|---|---|---|---|---|
| full-step | 0 | 10348 | 95.16% | 64 | 100.0% |
| SOC | 1 | 260 | 2.39% | 62 | 96.9% |
| backtracked | 6 | 202 | 1.86% | 64 | 100.0% |
| init | 7 | 64 | 0.59% | 64 | 100.0% |

Distinct branches active per fused iteration: mean 1.23, median 1, p95 2, max 3; iterations with ≥2 distinct branches: 18.2%, ≥3: 4.9%

## 2_ws_ir0_b75_dbg  (ir=0; 10345 active slot-iterations, 75 solve segments, 183 fused iterations, batch 75)

| branch | code | slot-iterations | % of iterations | solves with ≥1 | % of solves |
|---|---|---|---|---|---|
| full-step | 0 | 9710 | 93.86% | 75 | 100.0% |
| SOC | 1 | 330 | 3.19% | 75 | 100.0% |
| backtracked | 6 | 230 | 2.22% | 73 | 97.3% |
| init | 7 | 75 | 0.72% | 75 | 100.0% |

Distinct branches active per fused iteration: mean 1.99, median 2, p95 3, max 3; iterations with ≥2 distinct branches: 66.1%, ≥3: 32.8%

## 2_ws_ir0_b8_dbg_i2  (ir=0; 5567 active slot-iterations, 48 solve segments, 803 fused iterations, batch 8)

| branch | code | slot-iterations | % of iterations | solves with ≥1 | % of solves |
|---|---|---|---|---|---|
| full-step | 0 | 5137 | 92.28% | 48 | 100.0% |
| SOC | 1 | 167 | 3.00% | 45 | 93.8% |
| resto-entry | 5 | 1 | 0.02% | 1 | 2.1% |
| backtracked | 6 | 214 | 3.84% | 48 | 100.0% |
| init | 7 | 48 | 0.86% | 48 | 100.0% |

Distinct branches active per fused iteration: mean 1.38, median 1, p95 2, max 3; iterations with ≥2 distinct branches: 34.2%, ≥3: 3.6%

## 2_ws_ir100_b1_dbg_i2  (ir=100; 6320 active slot-iterations, 48 solve segments, 6320 fused iterations, batch 1)

| branch | code | slot-iterations | % of iterations | solves with ≥1 | % of solves |
|---|---|---|---|---|---|
| full-step | 0 | 5596 | 88.54% | 48 | 100.0% |
| SOC | 1 | 143 | 2.26% | 43 | 89.6% |
| SFR | 4 | 2 | 0.03% | 1 | 2.1% |
| resto-entry | 5 | 77 | 1.22% | 29 | 60.4% |
| backtracked | 6 | 450 | 7.12% | 48 | 100.0% |
| init | 7 | 48 | 0.76% | 48 | 100.0% |
| full-step(in-resto) | 8 | 3 | 0.05% | 1 | 2.1% |
| backtracked(in-resto) | 14 | 1 | 0.02% | 1 | 2.1% |

Distinct branches active per fused iteration: mean 1.00, median 1, p95 1, max 1; iterations with ≥2 distinct branches: 0.0%, ≥3: 0.0%

## 2_ws_ir100_b32_dbg_i2  (ir=100; 23990 active slot-iterations, 64 solve segments, 1002 fused iterations, batch 32)

| branch | code | slot-iterations | % of iterations | solves with ≥1 | % of solves |
|---|---|---|---|---|---|
| full-step | 0 | 15482 | 64.54% | 64 | 100.0% |
| SOC | 1 | 370 | 1.54% | 64 | 100.0% |
| WD | 2 | 20 | 0.08% | 7 | 10.9% |
| SFR | 4 | 7 | 0.03% | 5 | 7.8% |
| resto-entry | 5 | 314 | 1.31% | 46 | 71.9% |
| backtracked | 6 | 6551 | 27.31% | 64 | 100.0% |
| init | 7 | 64 | 0.27% | 64 | 100.0% |
| full-step(in-resto) | 8 | 300 | 1.25% | 14 | 21.9% |
| SOC(in-resto) | 9 | 56 | 0.23% | 9 | 14.1% |
| resto-entry(in-resto) | 13 | 670 | 2.79% | 18 | 28.1% |
| backtracked(in-resto) | 14 | 156 | 0.65% | 19 | 29.7% |

Distinct branches active per fused iteration: mean 3.25, median 3, p95 5, max 7; iterations with ≥2 distinct branches: 91.7%, ≥3: 69.3%

## 2_ws_ir100_b8_dbg_i2  (ir=100; 12530 active slot-iterations, 48 solve segments, 2425 fused iterations, batch 8)

| branch | code | slot-iterations | % of iterations | solves with ≥1 | % of solves |
|---|---|---|---|---|---|
| full-step | 0 | 8283 | 66.11% | 48 | 100.0% |
| SOC | 1 | 345 | 2.75% | 48 | 100.0% |
| WD | 2 | 42 | 0.34% | 13 | 27.1% |
| SFR | 4 | 4 | 0.03% | 3 | 6.2% |
| resto-entry | 5 | 291 | 2.32% | 44 | 91.7% |
| backtracked | 6 | 2890 | 23.06% | 48 | 100.0% |
| init | 7 | 48 | 0.38% | 48 | 100.0% |
| full-step(in-resto) | 8 | 173 | 1.38% | 10 | 20.8% |
| SOC(in-resto) | 9 | 35 | 0.28% | 3 | 6.2% |
| resto-entry(in-resto) | 13 | 344 | 2.75% | 7 | 14.6% |
| backtracked(in-resto) | 14 | 75 | 0.60% | 7 | 14.6% |

Distinct branches active per fused iteration: mean 1.75, median 2, p95 3, max 4; iterations with ≥2 distinct branches: 53.6%, ≥3: 19.0%

## 4_hr_ir0_b75_dbg  (ir=0; 21150 active slot-iterations, 149 solve segments, 282 fused iterations, batch 75)

| branch | code | slot-iterations | % of iterations | solves with ≥1 | % of solves |
|---|---|---|---|---|---|
| full-step | 0 | 20477 | 96.82% | 149 | 100.0% |
| SOC | 1 | 334 | 1.58% | 129 | 86.6% |
| SFR | 4 | 2 | 0.01% | 1 | 0.7% |
| resto-entry | 5 | 2 | 0.01% | 2 | 1.3% |
| backtracked | 6 | 186 | 0.88% | 95 | 63.8% |
| init | 7 | 149 | 0.70% | 149 | 100.0% |

Distinct branches active per fused iteration: mean 2.13, median 2, p95 4, max 4; iterations with ≥2 distinct branches: 72.7%, ≥3: 34.8%

## 4_ws_ir0_b1_dbg_i2  (ir=0; 9311 active slot-iterations, 48 solve segments, 9311 fused iterations, batch 1)

| branch | code | slot-iterations | % of iterations | solves with ≥1 | % of solves |
|---|---|---|---|---|---|
| full-step | 0 | 9044 | 97.13% | 48 | 100.0% |
| SOC | 1 | 132 | 1.42% | 45 | 93.8% |
| backtracked | 6 | 87 | 0.93% | 41 | 85.4% |
| init | 7 | 48 | 0.52% | 48 | 100.0% |

Distinct branches active per fused iteration: mean 1.00, median 1, p95 1, max 1; iterations with ≥2 distinct branches: 0.0%, ≥3: 0.0%

## 4_ws_ir0_b32_dbg_i2  (ir=0; 14048 active slot-iterations, 64 solve segments, 527 fused iterations, batch 32)

| branch | code | slot-iterations | % of iterations | solves with ≥1 | % of solves |
|---|---|---|---|---|---|
| full-step | 0 | 13694 | 97.48% | 64 | 100.0% |
| SOC | 1 | 165 | 1.17% | 57 | 89.1% |
| backtracked | 6 | 125 | 0.89% | 55 | 85.9% |
| init | 7 | 64 | 0.46% | 64 | 100.0% |

Distinct branches active per fused iteration: mean 1.41, median 1, p95 2, max 3; iterations with ≥2 distinct branches: 35.9%, ≥3: 4.9%

## 4_ws_ir0_b75_dbg  (ir=0; 16437 active slot-iterations, 75 solve segments, 269 fused iterations, batch 75)

| branch | code | slot-iterations | % of iterations | solves with ≥1 | % of solves |
|---|---|---|---|---|---|
| full-step | 0 | 15963 | 97.12% | 75 | 100.0% |
| SOC | 1 | 242 | 1.47% | 71 | 94.7% |
| resto-entry | 5 | 1 | 0.01% | 1 | 1.3% |
| backtracked | 6 | 156 | 0.95% | 68 | 90.7% |
| init | 7 | 75 | 0.46% | 75 | 100.0% |

Distinct branches active per fused iteration: mean 1.79, median 2, p95 3, max 4; iterations with ≥2 distinct branches: 60.2%, ≥3: 18.2%

## 4_ws_ir0_b8_dbg_i2  (ir=0; 10193 active slot-iterations, 48 solve segments, 1433 fused iterations, batch 8)

| branch | code | slot-iterations | % of iterations | solves with ≥1 | % of solves |
|---|---|---|---|---|---|
| full-step | 0 | 9918 | 97.30% | 48 | 100.0% |
| SOC | 1 | 148 | 1.45% | 45 | 93.8% |
| backtracked | 6 | 79 | 0.78% | 38 | 79.2% |
| init | 7 | 48 | 0.47% | 48 | 100.0% |

Distinct branches active per fused iteration: mean 1.13, median 1, p95 2, max 3; iterations with ≥2 distinct branches: 12.4%, ≥3: 0.9%

## 4_ws_ir100_b1_dbg_i2  (ir=100; 21271 active slot-iterations, 48 solve segments, 21271 fused iterations, batch 1)

| branch | code | slot-iterations | % of iterations | solves with ≥1 | % of solves |
|---|---|---|---|---|---|
| full-step | 0 | 11352 | 53.37% | 48 | 100.0% |
| SOC | 1 | 226 | 1.06% | 44 | 91.7% |
| WD | 2 | 172 | 0.81% | 25 | 52.1% |
| TS | 3 | 1 | 0.00% | 1 | 2.1% |
| SFR | 4 | 9 | 0.04% | 6 | 12.5% |
| resto-entry | 5 | 205 | 0.96% | 42 | 87.5% |
| backtracked | 6 | 8132 | 38.23% | 45 | 93.8% |
| init | 7 | 48 | 0.23% | 48 | 100.0% |
| full-step(in-resto) | 8 | 356 | 1.67% | 31 | 64.6% |
| SOC(in-resto) | 9 | 2 | 0.01% | 2 | 4.2% |
| resto-entry(in-resto) | 13 | 546 | 2.57% | 5 | 10.4% |
| backtracked(in-resto) | 14 | 222 | 1.04% | 30 | 62.5% |

Distinct branches active per fused iteration: mean 1.00, median 1, p95 1, max 1; iterations with ≥2 distinct branches: 0.0%, ≥3: 0.0%

## 4_ws_ir100_b32_dbg_i2  (ir=100; 29272 active slot-iterations, 64 solve segments, 1002 fused iterations, batch 32)

| branch | code | slot-iterations | % of iterations | solves with ≥1 | % of solves |
|---|---|---|---|---|---|
| full-step | 0 | 17189 | 58.72% | 64 | 100.0% |
| SOC | 1 | 343 | 1.17% | 60 | 93.8% |
| WD | 2 | 52 | 0.18% | 16 | 25.0% |
| SFR | 4 | 13 | 0.04% | 4 | 6.2% |
| resto-entry | 5 | 253 | 0.86% | 54 | 84.4% |
| backtracked | 6 | 10381 | 35.46% | 60 | 93.8% |
| init | 7 | 64 | 0.22% | 64 | 100.0% |
| full-step(in-resto) | 8 | 222 | 0.76% | 26 | 40.6% |
| SOC(in-resto) | 9 | 39 | 0.13% | 14 | 21.9% |
| SFR(in-resto) | 12 | 1 | 0.00% | 1 | 1.6% |
| resto-entry(in-resto) | 13 | 618 | 2.11% | 19 | 29.7% |
| backtracked(in-resto) | 14 | 97 | 0.33% | 25 | 39.1% |

Distinct branches active per fused iteration: mean 3.02, median 3, p95 5, max 9; iterations with ≥2 distinct branches: 90.3%, ≥3: 63.2%

## 4_ws_ir100_b8_dbg_i2  (ir=100; 23312 active slot-iterations, 48 solve segments, 3006 fused iterations, batch 8)

| branch | code | slot-iterations | % of iterations | solves with ≥1 | % of solves |
|---|---|---|---|---|---|
| full-step | 0 | 13708 | 58.80% | 48 | 100.0% |
| SOC | 1 | 299 | 1.28% | 46 | 95.8% |
| WD | 2 | 58 | 0.25% | 18 | 37.5% |
| TS | 3 | 1 | 0.00% | 1 | 2.1% |
| SFR | 4 | 12 | 0.05% | 4 | 8.3% |
| resto-entry | 5 | 125 | 0.54% | 36 | 75.0% |
| backtracked | 6 | 7875 | 33.78% | 46 | 95.8% |
| init | 7 | 48 | 0.21% | 48 | 100.0% |
| full-step(in-resto) | 8 | 147 | 0.63% | 12 | 25.0% |
| SOC(in-resto) | 9 | 17 | 0.07% | 7 | 14.6% |
| resto-entry(in-resto) | 13 | 964 | 4.14% | 11 | 22.9% |
| backtracked(in-resto) | 14 | 58 | 0.25% | 11 | 22.9% |

Distinct branches active per fused iteration: mean 2.22, median 2, p95 4, max 6; iterations with ≥2 distinct branches: 79.5%, ≥3: 36.2%

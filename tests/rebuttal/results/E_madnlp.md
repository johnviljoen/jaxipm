# Run E — MadNLP (ExaModels + MadNLPGPU/cuDSS), jaxipm pool, tol 1e-8, 5 repeats

| config | n | success | iters mean/max | wall per rep [s] | solves/s (paper protocol: build + solve) | solve-only solves/s | status codes |
|---|---|---|---|---|---|---|---|
| nav 90° | 2000 | 1.0000 | 43.3/125 | 726 / 945 / 936 / 943 / 943 | **2.250 ± 0.282** | 2.266 ± 0.288 | SOLVE_SUCCEEDED:2000 |
| nav 180° | 2000 | 1.0000 | 32.2/184 | 739 / 728 / 726 / 725 / 726 | **2.744 ± 0.022** | 2.765 ± 0.020 | SOLVE_SUCCEEDED:2000 |
| track v̄=1.0 | 1000 | 1.0000 | 38.0/84 | 211 / 388 / 402 / 395 / 397 | **2.970 ± 0.989** | 3.000 ± 1.017 | SOLVE_SUCCEEDED:1000 |
| track v̄=1.5 | 1000 | 1.0000 | 47.4/116 | 499 / 498 / 494 / 497 / 492 | **2.016 ± 0.012** | 2.027 ± 0.012 | SOLVE_SUCCEEDED:1000 |
| track v̄=2.0 | 1000 | 1.0000 | 57.8/136 | 606 / 606 / 607 / 607 / 606 | **1.649 ± 0.002** | 1.657 ± 0.003 | SOLVE_SUCCEEDED:1000 |
| multi N=2 | 75 | 1.0000 | 147.9/174 | 91 / 78 / 77 / 77 / 76 | **0.945 ± 0.066** | 0.949 ± 0.065 | SOLVE_SUCCEEDED:75 |
| multi N=4 | 75 | 1.0000 | 282.9/328 | 173 / 176 / 173 / 177 / 177 | **0.428 ± 0.005** | 0.430 ± 0.005 | SOLVE_SUCCEEDED:75 |

# E5 (existing data) — termination census
`at_cap` = fraction of solves that used max_iter = 500 iterations. IPOPT/MadNLP `success` = solver reported success (CasADi raises on any non-success return status, so 'Solved To Acceptable Level' counts as a failure here). jaxipm per-problem codes exist only where DEBUG_MODE was on.

| scenario | solver | source | n | success | iters mean | median | p95 | max | at_cap |
|---|---|---|---|---|---|---|---|---|---|
| nav 90° | IPOPT | E1 P=1 sweep | 2000 | 1.0000 | 22.8 | 22 | 31 | 73 | 0.0000 |
| nav 90° | IPOPT | paper sequential | 80 | 1.0000 | 22.5 | 22 | 30 | 34 | 0.0000 |
| nav 90° | MadNLP | paper | 80 | 1.0000 | 45.0 | 40 | 90 | 103 | 0.0000 |
| nav 90° | jaxipm | paper driver | 2000 | 1.0000 | – | – | – | – | – |
| nav 180° | IPOPT | E1 P=1 sweep | 2000 | 1.0000 | 20.5 | 19 | 28 | 70 | 0.0000 |
| nav 180° | IPOPT | paper sequential | 80 | 1.0000 | 20.6 | 19 | 28 | 36 | 0.0000 |
| nav 180° | MadNLP | paper | 80 | 1.0000 | 30.4 | 22 | 63 | 74 | 0.0000 |
| nav 180° | jaxipm | paper driver | 2000 | 1.0000 | – | – | – | – | – |
| track v̄=1.0 | IPOPT | E1 P=1 sweep | 1000 | 1.0000 | 24.5 | 24 | 38 | 76 | 0.0000 |
| track v̄=1.0 | IPOPT | paper sequential | 50 | 1.0000 | 23.3 | 23 | 33 | 41 | 0.0000 |
| track v̄=1.0 | MadNLP | paper | 50 | 1.0000 | 31.0 | 27 | 51 | 67 | 0.0000 |
| track v̄=1.0 | jaxipm | paper driver | 1000 | 1.0000 | – | – | – | – | – |
| track v̄=1.5 | IPOPT | E1 P=1 sweep | 1000 | 1.0000 | 29.2 | 28 | 41 | 70 | 0.0000 |
| track v̄=1.5 | IPOPT | paper sequential | 50 | 1.0000 | 28.2 | 27 | 41 | 45 | 0.0000 |
| track v̄=1.5 | MadNLP | paper | 50 | 1.0000 | 39.7 | 36 | 62 | 76 | 0.0000 |
| track v̄=1.5 | jaxipm | paper driver | 1000 | 1.0000 | – | – | – | – | – |
| track v̄=2.0 | IPOPT | E1 P=1 sweep | 1000 | 1.0000 | 46.7 | 43 | 71 | 124 | 0.0000 |
| track v̄=2.0 | IPOPT | paper sequential | 50 | 1.0000 | 47.9 | 44 | 73 | 85 | 0.0000 |
| track v̄=2.0 | MadNLP | paper | 50 | 1.0000 | 44.8 | 43 | 58 | 87 | 0.0000 |
| track v̄=2.0 | jaxipm | paper driver | 1000 | 1.0000 | – | – | – | – | – |
| multi N=2 | IPOPT | E1 P=1 sweep | 75 | 1.0000 | 177.0 | 177 | 177 | 177 | 0.0000 |
| multi N=2 | IPOPT | paper sequential | 50 | 1.0000 | 177.0 | 177 | 177 | 177 | 0.0000 |
| multi N=2 | MadNLP | paper | 50 | 1.0000 | 147.6 | 149 | 166 | 182 | 0.0000 |
| multi N=2 | jaxipm | paper driver | 75 | 0.9867 | 34.8 | 34 | 57 | 165 | 0.0000 |
| multi N=4 | IPOPT | E1 P=1 sweep | 75 | 1.0000 | 205.0 | 205 | 205 | 205 | 0.0000 |
| multi N=4 | IPOPT | paper sequential | 50 | 1.0000 | 216.0 | 216 | 216 | 216 | 0.0000 |
| multi N=4 | MadNLP | paper | 50 | 1.0000 | 283.1 | 282 | 312 | 317 | 0.0000 |
| multi N=4 | jaxipm | paper driver | 75 | 0.9867 | 68.3 | 64 | 115 | 284 | 0.0000 |

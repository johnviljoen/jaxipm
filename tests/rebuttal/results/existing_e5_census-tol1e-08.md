# E5 (existing data) — termination census
`at_cap` = fraction of solves that used max_iter = 500 iterations. IPOPT/MadNLP `success` = solver reported success (CasADi raises on any non-success return status, so 'Solved To Acceptable Level' counts as a failure here). jaxipm per-problem codes exist only where DEBUG_MODE was on.

| scenario | solver | source | n | success | iters mean | median | p95 | max | at_cap |
|---|---|---|---|---|---|---|---|---|---|
| nav 90° | IPOPT | D P=1 sweep_tol1e-08 | 2000 | 1.0000 | 23.9 | 23 | 32 | 74 | 0.0000 |
| nav 90° | IPOPT | paper sequential | 80 | 1.0000 | 22.5 | 22 | 30 | 34 | 0.0000 |
| nav 90° | MadNLP | paper | 80 | 1.0000 | 45.0 | 40 | 90 | 103 | 0.0000 |
| nav 90° | jaxipm | paper driver | 2000 | 1.0000 | – | – | – | – | – |
| nav 90° | jaxipm | ablation hr_ir0_b500_dbg_iclog census(conv/max_iter/tiny/resto_fail/acceptable)=2002/0/0/0/0 | 2000 | 1.0000 | 44.6 | 34 | 106 | 198 | 0.0000 |
| nav 90° | jaxipm | ablation hr_ir0_b500_dbg census(conv/max_iter/tiny/resto_fail/acceptable)=2008/0/0/0/0 | 2000 | 1.0000 | 44.4 | 33 | 102 | 214 | 0.0000 |
| nav 90° | jaxipm | ablation hr_ir0_b8_dbg_smoke4 census(conv/max_iter/tiny/resto_fail/acceptable)=24/0/0/0/0 | 24 | 1.0000 | 17.2 | 17 | 18 | 18 | 0.0000 |
| nav 90° | jaxipm | ablation ws_ir0_b1_dbg_i2 census(conv/max_iter/tiny/resto_fail/acceptable)=48/0/0/0/0 | 48 | 1.0000 | 23.6 | 22 | 35 | 89 | 0.0000 |
| nav 90° | jaxipm | ablation ws_ir0_b1_dbg_iclog census(conv/max_iter/tiny/resto_fail/acceptable)=48/0/0/0/0 | 48 | 1.0000 | 24.0 | 22 | 45 | 83 | 0.0000 |
| nav 90° | jaxipm | ablation ws_ir0_b32_dbg_i2 census(conv/max_iter/tiny/resto_fail/acceptable)=63/1/0/0/0 | 63 | 1.0000 | 43.7 | 30 | 97 | 187 | 0.0000 |
| nav 90° | jaxipm | ablation ws_ir0_b32_dbg_iclog census(conv/max_iter/tiny/resto_fail/acceptable)=96/0/0/0/0 | 96 | 1.0000 | 50.9 | 34 | 122 | 308 | 0.0000 |
| nav 90° | jaxipm | ablation ws_ir0_b500_dbg census(conv/max_iter/tiny/resto_fail/acceptable)=2000/0/0/0/0 | 2000 | 1.0000 | 58.5 | 40 | 150 | 465 | 0.0000 |
| nav 90° | jaxipm | ablation ws_ir0_b8_dbg_i2 census(conv/max_iter/tiny/resto_fail/acceptable)=48/0/0/0/0 | 48 | 1.0000 | 24.5 | 22 | 36 | 135 | 0.0000 |
| nav 90° | jaxipm | ablation ws_ir0_b8_dbg_smoke5 census(conv/max_iter/tiny/resto_fail/acceptable)=16/0/0/0/0 | 16 | 1.0000 | 20.3 | 20 | 24 | 25 | 0.0000 |
| nav 90° | jaxipm | ablation ws_ir0_b8_dbg_smoke6 census(conv/max_iter/tiny/resto_fail/acceptable)=16/0/0/0/0 | 16 | 1.0000 | 20.6 | 20 | 27 | 27 | 0.0000 |
| nav 90° | jaxipm | ablation ws_ir100_b1_dbg_i2 census(conv/max_iter/tiny/resto_fail/acceptable)=47/1/0/0/0 | 47 | 1.0000 | 21.4 | 19 | 38 | 103 | 0.0000 |
| nav 90° | jaxipm | ablation ws_ir100_b32_dbg_i2 census(conv/max_iter/tiny/resto_fail/acceptable)=61/3/0/0/0 | 61 | 1.0000 | 54.1 | 31 | 146 | 289 | 0.0000 |
| nav 90° | jaxipm | ablation ws_ir100_b8_dbg_i2 census(conv/max_iter/tiny/resto_fail/acceptable)=48/0/0/0/0 | 48 | 1.0000 | 26.1 | 20 | 47 | 182 | 0.0000 |
| nav 180° | IPOPT | D P=1 sweep_tol1e-08 | 2000 | 1.0000 | 21.5 | 20 | 30 | 71 | 0.0000 |
| nav 180° | IPOPT | paper sequential | 80 | 1.0000 | 20.6 | 19 | 28 | 36 | 0.0000 |
| nav 180° | MadNLP | paper | 80 | 1.0000 | 30.4 | 22 | 63 | 74 | 0.0000 |
| nav 180° | jaxipm | paper driver | 2000 | 1.0000 | – | – | – | – | – |
| nav 180° | jaxipm | ablation hr_ir0_b500_dbg census(conv/max_iter/tiny/resto_fail/acceptable)=2004/0/0/0/0 | 2000 | 1.0000 | 44.9 | 33 | 110 | 214 | 0.0000 |
| nav 180° | jaxipm | ablation ws_ir0_b1_dbg_i2 census(conv/max_iter/tiny/resto_fail/acceptable)=48/0/0/0/0 | 48 | 1.0000 | 17.9 | 16 | 25 | 43 | 0.0000 |
| nav 180° | jaxipm | ablation ws_ir0_b32_dbg_i2 census(conv/max_iter/tiny/resto_fail/acceptable)=64/0/0/0/0 | 64 | 1.0000 | 59.3 | 39 | 180 | 337 | 0.0000 |
| nav 180° | jaxipm | ablation ws_ir0_b500_dbg census(conv/max_iter/tiny/resto_fail/acceptable)=1999/1/0/0/0 | 1999 | 1.0000 | 62.0 | 42 | 166 | 427 | 0.0000 |
| nav 180° | jaxipm | ablation ws_ir0_b8_dbg_i2 census(conv/max_iter/tiny/resto_fail/acceptable)=48/0/0/0/0 | 48 | 1.0000 | 36.2 | 19 | 111 | 240 | 0.0000 |
| nav 180° | jaxipm | ablation ws_ir100_b1_dbg_i2 census(conv/max_iter/tiny/resto_fail/acceptable)=48/0/0/0/0 | 48 | 1.0000 | 15.7 | 12 | 31 | 70 | 0.0000 |
| nav 180° | jaxipm | ablation ws_ir100_b32_dbg_i2 census(conv/max_iter/tiny/resto_fail/acceptable)=60/4/0/0/0 | 60 | 1.0000 | 52.0 | 26 | 160 | 428 | 0.0000 |
| nav 180° | jaxipm | ablation ws_ir100_b8_dbg_i2 census(conv/max_iter/tiny/resto_fail/acceptable)=46/2/0/0/0 | 46 | 1.0000 | 27.5 | 14 | 76 | 316 | 0.0000 |
| track v̄=1.0 | IPOPT | D P=1 sweep_tol1e-08 | 1000 | 1.0000 | 25.4 | 25 | 39 | 77 | 0.0000 |
| track v̄=1.0 | IPOPT | paper sequential | 50 | 1.0000 | 23.3 | 23 | 33 | 41 | 0.0000 |
| track v̄=1.0 | MadNLP | paper | 50 | 1.0000 | 31.0 | 27 | 51 | 67 | 0.0000 |
| track v̄=1.0 | jaxipm | paper driver | 1000 | 1.0000 | – | – | – | – | – |
| track v̄=1.0 | jaxipm | ablation hr_ir0_b250_dbg census(conv/max_iter/tiny/resto_fail/acceptable)=1001/0/0/0/0 | 1000 | 1.0000 | 43.4 | 37 | 89 | 187 | 0.0000 |
| track v̄=1.0 | jaxipm | ablation ws_ir0_b1_dbg_i2 census(conv/max_iter/tiny/resto_fail/acceptable)=48/0/0/0/0 | 48 | 1.0000 | 32.9 | 32 | 54 | 57 | 0.0000 |
| track v̄=1.0 | jaxipm | ablation ws_ir0_b250_dbg census(conv/max_iter/tiny/resto_fail/acceptable)=993/7/0/0/0 | 993 | 1.0000 | 53.9 | 40 | 134 | 492 | 0.0000 |
| track v̄=1.0 | jaxipm | ablation ws_ir0_b32_dbg_i2 census(conv/max_iter/tiny/resto_fail/acceptable)=64/0/0/0/0 | 64 | 1.0000 | 72.0 | 42 | 249 | 335 | 0.0000 |
| track v̄=1.0 | jaxipm | ablation ws_ir0_b8_dbg_i2 census(conv/max_iter/tiny/resto_fail/acceptable)=48/0/0/0/0 | 48 | 1.0000 | 35.1 | 32 | 57 | 61 | 0.0000 |
| track v̄=1.0 | jaxipm | ablation ws_ir100_b1_dbg_i2 census(conv/max_iter/tiny/resto_fail/acceptable)=48/0/0/0/0 | 48 | 1.0000 | 31.5 | 28 | 52 | 59 | 0.0000 |
| track v̄=1.0 | jaxipm | ablation ws_ir100_b32_dbg_i2 census(conv/max_iter/tiny/resto_fail/acceptable)=63/1/0/0/0 | 63 | 1.0000 | 63.2 | 46 | 167 | 244 | 0.0000 |
| track v̄=1.0 | jaxipm | ablation ws_ir100_b8_dbg_i2 census(conv/max_iter/tiny/resto_fail/acceptable)=48/0/0/0/0 | 48 | 1.0000 | 35.6 | 31 | 59 | 78 | 0.0000 |
| track v̄=1.5 | IPOPT | D P=1 sweep_tol1e-08 | 1000 | 1.0000 | 30.3 | 29 | 42 | 71 | 0.0000 |
| track v̄=1.5 | IPOPT | paper sequential | 50 | 1.0000 | 28.2 | 27 | 41 | 45 | 0.0000 |
| track v̄=1.5 | MadNLP | paper | 50 | 1.0000 | 39.7 | 36 | 62 | 76 | 0.0000 |
| track v̄=1.5 | jaxipm | paper driver | 1000 | 1.0000 | – | – | – | – | – |
| track v̄=1.5 | jaxipm | ablation hr_ir0_b250_dbg census(conv/max_iter/tiny/resto_fail/acceptable)=1002/0/0/0/0 | 1000 | 1.0000 | 44.5 | 37 | 91 | 226 | 0.0000 |
| track v̄=1.5 | jaxipm | ablation ws_ir0_b1_dbg_i2 census(conv/max_iter/tiny/resto_fail/acceptable)=44/4/0/0/0 | 44 | 1.0000 | 34.2 | 32 | 48 | 55 | 0.0000 |
| track v̄=1.5 | jaxipm | ablation ws_ir0_b250_dbg census(conv/max_iter/tiny/resto_fail/acceptable)=968/32/0/0/0 | 968 | 1.0000 | 54.8 | 39 | 134 | 479 | 0.0000 |
| track v̄=1.5 | jaxipm | ablation ws_ir0_b32_dbg_i2 census(conv/max_iter/tiny/resto_fail/acceptable)=63/1/0/0/0 | 63 | 1.0000 | 59.6 | 39 | 153 | 364 | 0.0000 |
| track v̄=1.5 | jaxipm | ablation ws_ir0_b8_dbg_i2 census(conv/max_iter/tiny/resto_fail/acceptable)=44/4/0/0/0 | 44 | 1.0000 | 36.2 | 32 | 59 | 74 | 0.0000 |
| track v̄=1.5 | jaxipm | ablation ws_ir100_b1_dbg_i2 census(conv/max_iter/tiny/resto_fail/acceptable)=46/2/0/0/0 | 46 | 1.0000 | 32.4 | 30 | 53 | 65 | 0.0000 |
| track v̄=1.5 | jaxipm | ablation ws_ir100_b32_dbg_i2 census(conv/max_iter/tiny/resto_fail/acceptable)=63/1/0/0/0 | 63 | 1.0000 | 50.8 | 41 | 98 | 127 | 0.0000 |
| track v̄=1.5 | jaxipm | ablation ws_ir100_b8_dbg_i2 census(conv/max_iter/tiny/resto_fail/acceptable)=46/2/0/0/0 | 46 | 1.0000 | 35.8 | 30 | 60 | 88 | 0.0000 |
| track v̄=2.0 | IPOPT | D P=1 sweep_tol1e-08 | 1000 | 1.0000 | 47.8 | 44 | 73 | 125 | 0.0000 |
| track v̄=2.0 | IPOPT | paper sequential | 50 | 1.0000 | 47.9 | 44 | 73 | 85 | 0.0000 |
| track v̄=2.0 | MadNLP | paper | 50 | 1.0000 | 44.8 | 43 | 58 | 87 | 0.0000 |
| track v̄=2.0 | jaxipm | paper driver | 1000 | 1.0000 | – | – | – | – | – |
| track v̄=2.0 | jaxipm | ablation hr_ir0_b250_dbg census(conv/max_iter/tiny/resto_fail/acceptable)=1002/0/0/0/0 | 1000 | 1.0000 | 46.6 | 43 | 75 | 163 | 0.0000 |
| track v̄=2.0 | jaxipm | ablation ws_ir0_b1_dbg_i2 census(conv/max_iter/tiny/resto_fail/acceptable)=10/38/0/0/0 | 10 | 1.0000 | 43.3 | 38 | 63 | 63 | 0.0000 |
| track v̄=2.0 | jaxipm | ablation ws_ir0_b250_dbg census(conv/max_iter/tiny/resto_fail/acceptable)=858/142/0/0/0 | 858 | 1.0000 | 47.5 | 44 | 79 | 134 | 0.0000 |
| track v̄=2.0 | jaxipm | ablation ws_ir0_b250_dbg_sub250 census(conv/max_iter/tiny/resto_fail/acceptable)=213/37/0/0/0 | 213 | 1.0000 | 47.3 | 44 | 77 | 177 | 0.0000 |
| track v̄=2.0 | jaxipm | ablation ws_ir0_b32_dbg_i2 census(conv/max_iter/tiny/resto_fail/acceptable)=49/15/0/0/0 | 49 | 1.0000 | 47.5 | 43 | 74 | 114 | 0.0000 |
| track v̄=2.0 | jaxipm | ablation ws_ir0_b48_dbg_sub48 census(conv/max_iter/tiny/resto_fail/acceptable)=42/6/0/0/0 | 42 | 1.0000 | 47.7 | 43 | 72 | 81 | 0.0000 |
| track v̄=2.0 | jaxipm | ablation ws_ir0_b8_dbg_i2 census(conv/max_iter/tiny/resto_fail/acceptable)=11/37/0/0/0 | 11 | 1.0000 | 42.2 | 41 | 54 | 56 | 0.0000 |
| track v̄=2.0 | jaxipm | ablation ws_ir100_b1_dbg_i2 census(conv/max_iter/tiny/resto_fail/acceptable)=11/37/0/0/0 | 11 | 1.0000 | 41.6 | 39 | 52 | 52 | 0.0000 |
| track v̄=2.0 | jaxipm | ablation ws_ir100_b32_dbg_i2 census(conv/max_iter/tiny/resto_fail/acceptable)=51/13/0/0/0 | 51 | 1.0000 | 52.7 | 43 | 102 | 177 | 0.0000 |
| track v̄=2.0 | jaxipm | ablation ws_ir100_b8_dbg_i2 census(conv/max_iter/tiny/resto_fail/acceptable)=11/37/0/0/0 | 11 | 1.0000 | 67.5 | 41 | 191 | 323 | 0.0000 |
| multi N=2 | IPOPT | D P=1 sweep_tol1e-08 | 75 | 1.0000 | 178.0 | 178 | 178 | 178 | 0.0000 |
| multi N=2 | IPOPT | paper sequential | 50 | 1.0000 | 177.0 | 177 | 177 | 177 | 0.0000 |
| multi N=2 | MadNLP | paper | 50 | 1.0000 | 147.6 | 149 | 166 | 182 | 0.0000 |
| multi N=2 | jaxipm | paper driver | 75 | 0.9867 | 34.8 | 34 | 57 | 165 | 0.0000 |
| multi N=2 | jaxipm | ablation hr_ir0_b75_dbg census(conv/max_iter/tiny/resto_fail/acceptable)=75/0/0/0/0 | 75 | 1.0000 | 133.3 | 131 | 160 | 189 | 0.0000 |
| multi N=2 | jaxipm | ablation ws_ir0_b1_dbg_i2 census(conv/max_iter/tiny/resto_fail/acceptable)=48/0/0/0/0 | 48 | 1.0000 | 103.2 | 102 | 125 | 150 | 0.0000 |
| multi N=2 | jaxipm | ablation ws_ir0_b32_dbg_i2 census(conv/max_iter/tiny/resto_fail/acceptable)=58/6/0/0/0 | 58 | 1.0000 | 134.7 | 132 | 157 | 178 | 0.0000 |
| multi N=2 | jaxipm | ablation ws_ir0_b75_dbg census(conv/max_iter/tiny/resto_fail/acceptable)=75/0/0/0/0 | 75 | 1.0000 | 136.9 | 133 | 168 | 182 | 0.0000 |
| multi N=2 | jaxipm | ablation ws_ir0_b8_dbg_i2 census(conv/max_iter/tiny/resto_fail/acceptable)=48/0/0/0/0 | 48 | 1.0000 | 115.0 | 115 | 137 | 143 | 0.0000 |
| multi N=2 | jaxipm | ablation ws_ir100_b1_dbg_i2 census(conv/max_iter/tiny/resto_fail/acceptable)=48/0/0/0/0 | 48 | 1.0000 | 130.7 | 126 | 184 | 200 | 0.0000 |
| multi N=2 | jaxipm | ablation ws_ir100_b32_dbg_i2 census(conv/max_iter/tiny/resto_fail/acceptable)=47/17/0/0/0 | 47 | 1.0000 | 328.2 | 329 | 415 | 470 | 0.0000 |
| multi N=2 | jaxipm | ablation ws_ir100_b8_dbg_i2 census(conv/max_iter/tiny/resto_fail/acceptable)=47/1/0/0/0 | 47 | 1.0000 | 254.9 | 242 | 345 | 474 | 0.0000 |
| multi N=4 | IPOPT | D P=1 sweep_tol1e-08 | 75 | 1.0000 | 207.0 | 207 | 207 | 207 | 0.0000 |
| multi N=4 | IPOPT | paper sequential | 50 | 1.0000 | 216.0 | 216 | 216 | 216 | 0.0000 |
| multi N=4 | MadNLP | paper | 50 | 1.0000 | 283.1 | 282 | 312 | 317 | 0.0000 |
| multi N=4 | jaxipm | paper driver | 75 | 0.9867 | 68.3 | 64 | 115 | 284 | 0.0000 |
| multi N=4 | jaxipm | ablation hr_ir0_b75_dbg census(conv/max_iter/tiny/resto_fail/acceptable)=75/0/0/0/0 | 75 | 1.0000 | 216.4 | 215 | 253 | 281 | 0.0000 |
| multi N=4 | jaxipm | ablation ws_ir0_b1_dbg_i2 census(conv/max_iter/tiny/resto_fail/acceptable)=48/0/0/0/0 | 48 | 1.0000 | 193.0 | 196 | 224 | 236 | 0.0000 |
| multi N=4 | jaxipm | ablation ws_ir0_b32_dbg_i2 census(conv/max_iter/tiny/resto_fail/acceptable)=64/0/0/0/0 | 64 | 1.0000 | 218.5 | 223 | 256 | 273 | 0.0000 |
| multi N=4 | jaxipm | ablation ws_ir0_b75_dbg census(conv/max_iter/tiny/resto_fail/acceptable)=75/0/0/0/0 | 75 | 1.0000 | 218.2 | 222 | 256 | 268 | 0.0000 |
| multi N=4 | jaxipm | ablation ws_ir0_b8_dbg_i2 census(conv/max_iter/tiny/resto_fail/acceptable)=48/0/0/0/0 | 48 | 1.0000 | 211.4 | 215 | 242 | 258 | 0.0000 |
| multi N=4 | jaxipm | ablation ws_ir100_b1_dbg_i2 census(conv/max_iter/tiny/resto_fail/acceptable)=17/31/0/0/0 | 17 | 0.8235 | 336.6 | 387 | 485 | 494 | 0.0000 |
| multi N=4 | jaxipm | ablation ws_ir100_b8_dbg_i2 census(conv/max_iter/tiny/resto_fail/acceptable)=11/37/0/0/0 | 11 | 0.9091 | 433.1 | 482 | 496 | 496 | 0.0000 |

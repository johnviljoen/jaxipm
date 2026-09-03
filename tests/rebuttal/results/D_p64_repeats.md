# Run D repeats — IPOPT, P = 64 physical cores, tol 1e-8 (mean ± sd over repeats)

Same protocol as the sweep (one pinned single-thread process per core, whole pool per repeat). `now` = repeats run 2026-08-27 15:00 while other users loaded the host (5-min load ≈ 55); `quiet` = repeats run when the 5-min load average dropped below 8. Host load average (1/5/15 min) is stamped into every file at save time. jaxipm column = run A (ILB on, 5 fresh-process repeats).

| scenario | sweep single run | now: mean ± sd (n) | now: per-repeat | now: load avg (5 min) at save | quiet: mean ± sd (n) | quiet: per-repeat | quiet: load avg (5 min) at save | jaxipm A | node/jaxipm (best set) |
|---|---|---|---|---|---|---|---|---|---|
| nav 90° | 350.2 | 280.7 ± 13.9 (5) | 257.2, 286.7, 293.7, 282.5, 283.5 | 58, 108, 132, 139, 151 | – | – | – | 101.69 ± 2.96 | 2.76× |
| nav 180° | 353.5 | 310.9 ± 14.8 (5) | 284.8, 315.3, 318.4, 315.1, 320.8 | 62, 113, 132, 140, 151 | – | – | – | 105.82 ± 6.32 | 2.94× |
| track v̄=1.0 | 259.7 | 234.6 ± 9.5 (5) | 221.3, 242.1, 240.0, 241.8, 228.0 | 64, 113, 131, 138, 154 | – | – | – | 35.75 ± 1.20 | 6.56× |
| track v̄=1.5 | 221.2 | 185.3 ± 3.8 (5) | 184.3, 184.8, 187.8, 189.9, 179.9 | 66, 114, 132, 138, 154 | – | – | – | 31.60 ± 0.76 | 5.87× |
| track v̄=2.0 | 125.6 | 105.4 ± 3.7 (5) | 108.9, 109.2, 105.8, 101.5, 101.8 | 71, 116, 133, 140, 155 | – | – | – | 30.94 ± 0.50 | 3.41× |
| multi N=2 | 15.4 | 13.5 ± 0.7 (5) | 14.4, 13.2, 13.9, 13.5, 12.6 | ?, ?, ?, ?, ? | – | – | – | 8.72 ± 0.28 | 1.55× |
| multi N=4 | 5.6 | 5.2 ± 0.2 (5) | 5.4, 5.3, 5.3, 5.2, 4.9 | ?, ?, ?, ?, ? | – | – | – | 1.35 ± 0.06 | 3.87× |

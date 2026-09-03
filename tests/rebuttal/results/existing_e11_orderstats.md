# E11 (existing data) — order-statistics prediction of the ILB gain
Predicted (b)→(c) throughput ratio = E[max_{i≤N} K_i] / E[K] with N = paper batch size, using the EMPIRICAL per-problem iteration distribution K (exact from the empirical CDF, i.i.d. draws with replacement). Rows marked *IPOPT proxy* use IPOPT's iteration counts on the same pool because the paper jaxipm files lack per-problem counts; jaxipm rows appear once ablation *_dbg runs exist.

| scenario | K source | n | E[K] | E[max_N K] | N | predicted ratio |
|---|---|---|---|---|---|---|
| nav 90° | IPOPT proxy (E1 P=1) | 2000 | 22.8 | 56.2 | 500 | 2.46 |
| nav 180° | IPOPT proxy (E1 P=1) | 2000 | 20.5 | 53.3 | 500 | 2.60 |
| track v̄=1.0 | IPOPT proxy (E1 P=1) | 1000 | 24.5 | 60.7 | 250 | 2.48 |
| track v̄=1.5 | IPOPT proxy (E1 P=1) | 1000 | 29.2 | 60.5 | 250 | 2.07 |
| track v̄=2.0 | IPOPT proxy (E1 P=1) | 1000 | 46.7 | 100.5 | 250 | 2.15 |
| multi N=2 | IPOPT proxy (E1 P=1) | 75 | 177.0 | 177.0 | 75 | 1.00 |
| multi N=2 | jaxipm paper driver | 75 | 34.8 | 127.3 | 75 | 3.66 |
| multi N=4 | IPOPT proxy (E1 P=1) | 75 | 205.0 | 205.0 | 75 | 1.00 |
| multi N=4 | jaxipm paper driver | 75 | 68.3 | 224.9 | 75 | 3.29 |

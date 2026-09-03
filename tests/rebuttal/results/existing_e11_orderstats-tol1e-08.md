# E11 (existing data) — order-statistics prediction of the ILB gain
Predicted (b)→(c) throughput ratio = E[max_{i≤N} K_i] / E[K] with N = paper batch size, using the EMPIRICAL per-problem iteration distribution K (exact from the empirical CDF, i.i.d. draws with replacement). Rows marked *IPOPT proxy* use IPOPT's iteration counts on the same pool because the paper jaxipm files lack per-problem counts; jaxipm rows appear once ablation *_dbg runs exist.

| scenario | K source | n | E[K] | E[max_N K] | N | predicted ratio |
|---|---|---|---|---|---|---|
| nav 90° | IPOPT proxy (E1 P=1) | 2000 | 23.9 | 57.6 | 500 | 2.41 |
| nav 90° | jaxipm ablation hr_ir0_b500_dbg_iclog | 2000 | 44.6 | 186.5 | 500 | 4.18 |
| nav 90° | jaxipm ablation hr_ir0_b500_dbg | 2000 | 44.4 | 192.5 | 500 | 4.34 |
| nav 90° | jaxipm ablation hr_ir0_b8_dbg_smoke4 | 24 | 17.2 | 18.0 | 8 | 1.05 |
| nav 90° | jaxipm ablation ws_ir0_b1_dbg_i2 (measured per-batch max: mean 24.6) | 48 | 23.6 | 23.6 | 1 | 1.00 |
| nav 90° | jaxipm ablation ws_ir0_b1_dbg_iclog (measured per-batch max: mean 25.0) | 48 | 24.0 | 24.0 | 1 | 1.00 |
| nav 90° | jaxipm ablation ws_ir0_b32_dbg_i2 (measured per-batch max: mean 305.0) | 63 | 43.7 | 138.8 | 32 | 3.18 |
| nav 90° | jaxipm ablation ws_ir0_b32_dbg_iclog (measured per-batch max: mean 194.0) | 96 | 50.9 | 178.9 | 32 | 3.52 |
| nav 90° | jaxipm ablation ws_ir0_b500_dbg (measured per-batch max: mean 354.5) | 2000 | 58.5 | 397.1 | 500 | 6.79 |
| nav 90° | jaxipm ablation ws_ir0_b8_dbg_i2 (measured per-batch max: mean 47.7) | 48 | 24.5 | 47.7 | 8 | 1.95 |
| nav 90° | jaxipm ablation ws_ir0_b8_dbg_smoke5 (measured per-batch max: mean 25.0) | 16 | 20.3 | 23.7 | 8 | 1.17 |
| nav 90° | jaxipm ablation ws_ir0_b8_dbg_smoke6 (measured per-batch max: mean 28.0) | 16 | 20.6 | 26.0 | 8 | 1.26 |
| nav 90° | jaxipm ablation ws_ir100_b1_dbg_i2 (measured per-batch max: mean 32.3) | 47 | 21.4 | 21.4 | 1 | 1.00 |
| nav 90° | jaxipm ablation ws_ir100_b32_dbg_i2 (measured per-batch max: mean 501.0) | 61 | 54.1 | 213.1 | 32 | 3.94 |
| nav 90° | jaxipm ablation ws_ir100_b8_dbg_i2 (measured per-batch max: mean 56.0) | 48 | 26.1 | 62.1 | 8 | 2.38 |
| nav 180° | IPOPT proxy (E1 P=1) | 2000 | 21.5 | 55.0 | 500 | 2.56 |
| nav 180° | jaxipm ablation hr_ir0_b500_dbg | 2000 | 44.9 | 203.5 | 500 | 4.53 |
| nav 180° | jaxipm ablation ws_ir0_b1_dbg_i2 (measured per-batch max: mean 18.9) | 48 | 17.9 | 17.9 | 1 | 1.00 |
| nav 180° | jaxipm ablation ws_ir0_b32_dbg_i2 (measured per-batch max: mean 278.0) | 64 | 59.3 | 252.7 | 32 | 4.26 |
| nav 180° | jaxipm ablation ws_ir0_b500_dbg (measured per-batch max: mean 389.2) | 1999 | 62.0 | 382.8 | 500 | 6.17 |
| nav 180° | jaxipm ablation ws_ir0_b8_dbg_i2 (measured per-batch max: mean 106.7) | 48 | 36.2 | 111.6 | 8 | 3.09 |
| nav 180° | jaxipm ablation ws_ir100_b1_dbg_i2 (measured per-batch max: mean 16.7) | 48 | 15.7 | 15.7 | 1 | 1.00 |
| nav 180° | jaxipm ablation ws_ir100_b32_dbg_i2 (measured per-batch max: mean 501.0) | 60 | 52.0 | 320.0 | 32 | 6.16 |
| nav 180° | jaxipm ablation ws_ir100_b8_dbg_i2 (measured per-batch max: mean 206.0) | 46 | 27.5 | 93.0 | 8 | 3.38 |
| track v̄=1.0 | IPOPT proxy (E1 P=1) | 1000 | 25.4 | 61.7 | 250 | 2.43 |
| track v̄=1.0 | jaxipm ablation hr_ir0_b250_dbg | 1000 | 43.4 | 161.2 | 250 | 3.71 |
| track v̄=1.0 | jaxipm ablation ws_ir0_b1_dbg_i2 (measured per-batch max: mean 33.9) | 48 | 32.9 | 32.9 | 1 | 1.00 |
| track v̄=1.0 | jaxipm ablation ws_ir0_b250_dbg (measured per-batch max: mean 501.0) | 993 | 53.9 | 404.6 | 250 | 7.51 |
| track v̄=1.0 | jaxipm ablation ws_ir0_b32_dbg_i2 (measured per-batch max: mean 326.5) | 64 | 72.0 | 294.7 | 32 | 4.09 |
| track v̄=1.0 | jaxipm ablation ws_ir0_b8_dbg_i2 (measured per-batch max: mean 55.7) | 48 | 35.1 | 54.0 | 8 | 1.54 |
| track v̄=1.0 | jaxipm ablation ws_ir100_b1_dbg_i2 (measured per-batch max: mean 32.5) | 48 | 31.5 | 31.5 | 1 | 1.00 |
| track v̄=1.0 | jaxipm ablation ws_ir100_b32_dbg_i2 (measured per-batch max: mean 373.0) | 63 | 63.2 | 204.6 | 32 | 3.24 |
| track v̄=1.0 | jaxipm ablation ws_ir100_b8_dbg_i2 (measured per-batch max: mean 61.8) | 48 | 35.6 | 58.1 | 8 | 1.63 |
| track v̄=1.5 | IPOPT proxy (E1 P=1) | 1000 | 30.3 | 61.6 | 250 | 2.03 |
| track v̄=1.5 | jaxipm ablation hr_ir0_b250_dbg | 1000 | 44.5 | 195.4 | 250 | 4.39 |
| track v̄=1.5 | jaxipm ablation ws_ir0_b1_dbg_i2 (measured per-batch max: mean 74.0) | 44 | 34.2 | 34.2 | 1 | 1.00 |
| track v̄=1.5 | jaxipm ablation ws_ir0_b250_dbg (measured per-batch max: mean 501.0) | 968 | 54.8 | 423.9 | 250 | 7.74 |
| track v̄=1.5 | jaxipm ablation ws_ir0_b32_dbg_i2 (measured per-batch max: mean 433.0) | 63 | 59.6 | 254.7 | 32 | 4.27 |
| track v̄=1.5 | jaxipm ablation ws_ir0_b8_dbg_i2 (measured per-batch max: mean 203.0) | 44 | 36.2 | 57.1 | 8 | 1.58 |
| track v̄=1.5 | jaxipm ablation ws_ir100_b1_dbg_i2 (measured per-batch max: mean 52.9) | 46 | 32.4 | 32.4 | 1 | 1.00 |
| track v̄=1.5 | jaxipm ablation ws_ir100_b32_dbg_i2 (measured per-batch max: mean 314.5) | 63 | 50.8 | 115.7 | 32 | 2.28 |
| track v̄=1.5 | jaxipm ablation ws_ir100_b8_dbg_i2 (measured per-batch max: mean 204.8) | 46 | 35.8 | 59.5 | 8 | 1.66 |
| track v̄=2.0 | IPOPT proxy (E1 P=1) | 1000 | 47.8 | 101.3 | 250 | 2.12 |
| track v̄=2.0 | jaxipm ablation hr_ir0_b250_dbg | 1000 | 46.6 | 130.8 | 250 | 2.80 |
| track v̄=2.0 | jaxipm ablation ws_ir0_b1_dbg_i2 (measured per-batch max: mean 405.9) | 10 | 43.3 | 43.3 | 1 | 1.00 |
| track v̄=2.0 | jaxipm ablation ws_ir0_b250_dbg (measured per-batch max: mean 501.0) | 858 | 47.5 | 114.1 | 250 | 2.40 |
| track v̄=2.0 | jaxipm ablation ws_ir0_b250_dbg_sub250 (measured per-batch max: mean 501.0) | 213 | 47.3 | 158.4 | 250 | 3.35 |
| track v̄=2.0 | jaxipm ablation ws_ir0_b32_dbg_i2 (measured per-batch max: mean 501.0) | 49 | 47.5 | 94.4 | 32 | 1.99 |
| track v̄=2.0 | jaxipm ablation ws_ir0_b48_dbg_sub48 (measured per-batch max: mean 501.0) | 42 | 47.7 | 78.3 | 48 | 1.64 |
| track v̄=2.0 | jaxipm ablation ws_ir0_b8_dbg_i2 (measured per-batch max: mean 501.0) | 11 | 42.2 | 53.0 | 8 | 1.26 |
| track v̄=2.0 | jaxipm ablation ws_ir100_b1_dbg_i2 (measured per-batch max: mean 396.0) | 11 | 41.6 | 41.6 | 1 | 1.00 |
| track v̄=2.0 | jaxipm ablation ws_ir100_b32_dbg_i2 (measured per-batch max: mean 501.0) | 51 | 52.7 | 145.4 | 32 | 2.76 |
| track v̄=2.0 | jaxipm ablation ws_ir100_b8_dbg_i2 (measured per-batch max: mean 501.0) | 11 | 67.5 | 198.3 | 8 | 2.94 |
| multi N=2 | IPOPT proxy (E1 P=1) | 75 | 178.0 | 178.0 | 75 | 1.00 |
| multi N=2 | jaxipm paper driver | 75 | 34.8 | 127.3 | 75 | 3.66 |
| multi N=2 | jaxipm ablation hr_ir0_b75_dbg | 75 | 133.3 | 183.2 | 75 | 1.37 |
| multi N=2 | jaxipm ablation ws_ir0_b1_dbg_i2 (measured per-batch max: mean 104.2) | 48 | 103.2 | 103.2 | 1 | 1.00 |
| multi N=2 | jaxipm ablation ws_ir0_b32_dbg_i2 (measured per-batch max: mean 501.0) | 58 | 134.7 | 171.5 | 32 | 1.27 |
| multi N=2 | jaxipm ablation ws_ir0_b75_dbg (measured per-batch max: mean 183.0) | 75 | 136.9 | 179.6 | 75 | 1.31 |
| multi N=2 | jaxipm ablation ws_ir0_b8_dbg_i2 (measured per-batch max: mean 133.8) | 48 | 115.0 | 133.4 | 8 | 1.16 |
| multi N=2 | jaxipm ablation ws_ir100_b1_dbg_i2 (measured per-batch max: mean 131.7) | 48 | 130.7 | 130.7 | 1 | 1.00 |
| multi N=2 | jaxipm ablation ws_ir100_b32_dbg_i2 (measured per-batch max: mean 501.0) | 47 | 328.2 | 447.6 | 32 | 1.36 |
| multi N=2 | jaxipm ablation ws_ir100_b8_dbg_i2 (measured per-batch max: mean 404.2) | 47 | 254.9 | 361.0 | 8 | 1.42 |
| multi N=4 | IPOPT proxy (E1 P=1) | 75 | 207.0 | 207.0 | 75 | 1.00 |
| multi N=4 | jaxipm paper driver | 75 | 68.3 | 224.9 | 75 | 3.29 |
| multi N=4 | jaxipm ablation hr_ir0_b75_dbg | 75 | 216.4 | 276.5 | 75 | 1.28 |
| multi N=4 | jaxipm ablation ws_ir0_b1_dbg_i2 (measured per-batch max: mean 194.0) | 48 | 193.0 | 193.0 | 1 | 1.00 |
| multi N=4 | jaxipm ablation ws_ir0_b32_dbg_i2 (measured per-batch max: mean 263.5) | 64 | 218.5 | 266.3 | 32 | 1.22 |
| multi N=4 | jaxipm ablation ws_ir0_b75_dbg (measured per-batch max: mean 269.0) | 75 | 218.2 | 265.6 | 75 | 1.22 |
| multi N=4 | jaxipm ablation ws_ir0_b8_dbg_i2 (measured per-batch max: mean 238.8) | 48 | 211.4 | 239.1 | 8 | 1.13 |
| multi N=4 | jaxipm ablation ws_ir100_b1_dbg_i2 (measured per-batch max: mean 443.1) | 17 | 336.6 | 336.6 | 1 | 1.00 |
| multi N=4 | jaxipm ablation ws_ir100_b8_dbg_i2 (measured per-batch max: mean 501.0) | 11 | 433.1 | 495.2 | 8 | 1.14 |

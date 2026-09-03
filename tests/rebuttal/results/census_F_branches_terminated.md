# Branch activation census restricted to solves that TERMINATED inside the census window (ILB on, paper batch sizes)

Same data as `census_F_branches.md` (branch id per member per fused iteration, DEBUG runs), but slot timelines are cut into solves at each restart, and solves still running when the buffer filled (the census run stops when max_solves solutions are collected; unfinished slots are dropped) are excluded. Those unfinished solves are the stagnated members of the inertia-correction defect (STATUS 22–24): they take the tiny-step branch every iteration and trigger the watchdog, so including them inflates (3) and (4). (6) = restoration entry + iterations inside restoration; distinct = mean / max number of different branches active in one fused iteration (unchanged: computed over all active members).

| Problem | solves (terminated / total) | full | (1) | (2) | (3) | (4) | (5) | (6) | restart | distinct | TS/WD iterations in unfinished solves |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Sector 90° | 1997 / 2497 | 72.08 | 8.67 | 16.93 | 0.00 | 0.00 | 0.01 | 0.10 | 2.20 | 4.4 / 6 | 0 TS, 5 WD |
| Sector 180° | 1991 / 2491 | 68.93 | 8.72 | 20.12 | 0.01 | 0.00 | 0.00 | 0.04 | 2.18 | 4.2 / 6 | 0 TS, 25 WD |
| v̄ = 1.0 | 998 / 1248 | 86.30 | 3.06 | 8.36 | 0.00 | 0.00 | 0.00 | 0.02 | 2.25 | 3.9 / 5 | 0 TS, 8 WD |
| v̄ = 1.5 | 997 / 1247 | 86.38 | 3.89 | 7.49 | 0.00 | 0.00 | 0.02 | 0.02 | 2.20 | 5.2 / 7 | 1445 TS, 74 WD |
| v̄ = 2.0 | 997 / 1247 | 94.35 | 1.95 | 1.36 | 0.05 | 0.00 | 0.01 | 0.17 | 2.10 | 5.5 / 8 | 1913 TS, 247 WD |
| N_quads = 2 | 74 / 149 | 93.79 | 1.94 | 3.52 | 0.00 | 0.00 | 0.00 | 0.00 | 0.75 | 2.6 / 4 | 0 TS, 0 WD |
| N_quads = 4 | 74 / 149 | 97.19 | 0.83 | 1.50 | 0.00 | 0.00 | 0.01 | 0.00 | 0.46 | 2.1 / 4 | 0 TS, 0 WD |

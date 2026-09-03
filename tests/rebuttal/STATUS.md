# RA-L-26-3388 rebuttal — run-manifest status (2026-08-25, late evening)

Tracks the **run manifest A–L** (the rewritten `EXPERIMENTS.md`). Earlier E-numbered
work has been re-mapped; nothing was built for CusADi / monolithic ExaModels / CUTEst /
a lockstep solver. Raw results first; interpretation is marked.

## 0. Findings that change drafted answers

1. **Run D exists (2026-08-03) and the CPU node wins.** Pinned single-thread IPOPT,
   P = 64 physical: nav 308 / 351 solves/s vs jaxipm 101.7 / 105.8; track 269 /
   228 / 138 vs 35.8 / 31.6 / 30.9; multi 15.7 / 6.0 vs 8.72 / 1.35. Crossover at
   P = 8–32 physical cores; per-core efficiency ≥ 0.65 to P = 64; SMT +10–30%.
   `results/e1_cpu_scaling.md`, `figures/e1_cpu_scaling.pdf`. *Interpretation:* the
   headline "25×" is single-core; R11-1 must be reframed (per device / per watt /
   in-pipeline), and the "MUMPS uses every core" text dropped.
   ⚠ D was run at **`ipopt.tol = 1e-6`** (all three `tests/*/ipopt_*.py`), not the
   manifest's 1e-8. **Decision (John, 2026-08-26 08:30): rerun at 1e-8.** Queued
   (`run_queue.sh d_cpu`, `--tol 1e-8`, output suffix `_tol1e-08`, now also saving
   U, IPOPT return-status code and λ_g per solve for G and H).
2. **Paired final cost (H, existing data).** nav: jaxipm lower than IPOPT on 96–99%
   of matched instances (median −0.13%; mean-gap CI 90° [−0.085, +0.061]%, 180°
   [−0.230, −0.107]%). track: medians identical; 11–27% of instances in a
   different local optimum, both directions; v̄=2.0 mean gap +0.43%, CI [−0.11,
   +1.14]%, Wilcoxon p = 0.07. `results/existing_e4_cost.md`.
3. **Termination (G).** No solver hits the 500 cap. jaxipm multi-swap fails 1/75 in
   both N (collapsed solutions) — the letter's "all converged" is false there.
   nav 90° HR census (tonight): 2000/2000 code 1, no hidden failures, iterations
   mean 44.3 / median 33 / max 200 vs IPOPT 22.8 mean on the same pool.
4. **Iteration inflation vs IPOPT (44 vs 22.8 mean at N=500) — cause open.** The
   earlier "batch-size dependent" reading was confounded: the N=8 smoke solved
   only pool instances 0–23 (all near θ≈0, easy), not a sample of the sector.
   The I1 cross-check shows the cuDSS step inaccuracy is the same at N=1 and
   N=8 (item 9), so batch size is not the variable; I2 (small N, ir 0 vs 100,
   on a proper sample of instances) is the test that isolates the linear solve.
5. **Batch sweep already contradicts "occupancy-limited" (C, ILB-on).** nav 90°:
   83.6 / 94.8 / 98.2 / 88.0 / 94.9 solves/s at N = 125 … 2000; track v̄=1.5 27.2 /
   30.8 / 32.6 / 30.2 at 125 … 1000; multi N=4 0.20 / 1.31 / 1.05 / 0.85 / 0.74 at
   25 … 600. Throughput peaks at 250–500 and falls beyond.
6. **Bug fixed in `solve_throughput`** (both paths): the dummy scatter index
   `max_solves − 1` collided with the final real write, so the last collected row
   of every call could be zeros/stale (the obj = 0 outlier in the paper's nav
   npz). Now an out-of-bounds index with `mode="drop"`; verified.
7. **spineax factor registry is an LRU of capacity 8** (`SPINEAX_FACTOR_CACHE`);
   overflow evicts and silently re-factorizes on next use. Re-minting a batch
   token per WS batch therefore thrashes the cache inside the timed loop. The WS
   harness now mints one token per config and injects problems per batch (the
   hot-restart path); `registry_size` / `rebuild_count` are logged per run.
8. **Run B, nav 90° (fixed harness, 5 reps): ILB-off 15.32 ± 1.55 solves/s**
   (16.0 / 14.2 / 13.9 / 17.7 / 14.8) vs ILB-on 101.7 ± 3.0 → measured ILB gain
   6.6×, against the order-statistics prediction of 2.46. `collected` is
   1998–2000 of 2000 per repeat: 0–2 members per repeat terminate on a failure
   code in WS mode (the WS-dbg census will identify which code). *Interpretation
   (pending the census):* the gap between 6.6× and 2.5× is the work converged
   members keep doing while the slowest member finishes, plus longer tails —
   check the per-batch `n_fused_iters` in the dbg run.
9. **cuDSS single solve is ~1 % inaccurate at `ir_nsteps=0` — and it is not a
   batching effect.** `i1_solve_accuracy.py` (nav 90°, real solver state after k
   fused iterations, same factorized K and rhs as the solver): relative residual
   ‖Kx−r‖/‖r‖ of a fresh `cudss.solve` — N=8, k=3: median 6.4e-3, max 9.7e-3;
   N=1, k=3: 5.0e-3; N=8, k=30: median 2.8e-2, max 8.4e-2. SciPy LU on the
   identical matrix: 2e-15. Solution error vs SciPy: 0.3–1 % at k=3, up to 34 %
   at k=30. Two Richardson steps (ir=2): 1e-7 (k=3), 2e-5 (k=30). The solver's
   own stored step reproduces these residuals (DEBUG `kkt_res_log` validated).
   **Controls (N=8, k=3):** re-minted fresh symmetric-LDLᵀ token 5.7e-3 (max
   8.6e-3) — not stale factors; cuDSS **general LU** (mtype 0, full pattern) on
   the same matrices 9.6e-3 (max 1.2e-2) — not the LDLᵀ path either. N=500,
   k=3: median 8.9e-4, max 8.7e-2, solution error median 2.7 %, max 10 %; with
   ir=2 one member's refinement **diverged (residual 5.1)** — the I3 phenomenon
   seen for free. N=1, k=10: residual 2.4 (step useless), ir=2 → 7.9e-3.
   This is the evidenced form of R8-6a / R8-7a / R11-2(8): every Newton
   direction is 0.3–30 % wrong, not batch dependent, worse as μ → 0.
   **Mechanism (measured):** `cudss.query` reports **136 perturbed pivots per
   KKT system** (1088 per batch of 8), every one clamped to exactly |pivot| =
   1e-13 — cuDSS static pivoting at its default epsilon, with internal
   refinement disabled by spineax. The residual sits in the x-block (5e-3)
   and is ~1e-9 / 1e-15 in the y_c / y_d blocks. Late in the solve (N=500,
   k=10 / 30) the relative residual explodes (median 4e-4 / 0.43, max 14 / 376)
   and ir=2 diverges for some members (max 6e4 / 7e8) — but ‖r‖ → 0 there, so
   the DEBUG log now also records the backward error
   ‖KΔ−r‖/(‖K‖_F‖Δ‖+‖r‖) to separate solver error from ill-conditioning.
   First backward-error numbers (nav 90°, N=8, 16 solves, all iterations):
   median 6.5e-12, p95 3.8e-7, max 1.3e-6 — i.e. the static-pivot perturbation
   costs up to ~1e-6 backward error, which ill-conditioning (‖K‖‖Δ‖/‖r‖ up to
   ~1e6 late in a solve) amplifies into the O(1) relative-to-rhs residuals,
   whereas a pivoted LU (SciPy) stays at 1e-15 on the same systems.
10. **Run B nav 180° decays within one process:** reps 13.2 / 14.5 / 10.0 /
   5.4 / 7.1 solves/s (mean 10.06 ± 3.86), `collected` 1997–2000. Only my
   process was on GPU 1. This is the same in-process decay the A campaign saw
   for nav (JIT-once 112 → 87) and avoided with one process per repeat, so B
   will additionally be measured fresh-process (`timing_ws_fresh` stage) and
   the two protocols reported side by side.

11. **Run I3 (nav 90°, N=500, `i3_refinement_divergence.py`, 20 Richardson
   steps on the solver's own factorized K and rhs):** refinement DIVERGES for
   17/500 members at fused iteration k=3 (5 of them by >1e6), 149/500 at k=10,
   242/500 at k=30, growing ~10× per step (worst trajectories reach 1e17 /
   1e52 / 1e71 in 20 steps) while the median member converges to 1e-16
   (459/500 below 1e-10 at k=3). No member goes non-finite within 20 steps.
   Divergence = Richardson iteration with ‖I − K K̃⁻¹‖ > 1 for members whose
   static-pivot perturbation (item 9) is too large — a per-member property.
   **Coupling test result:** on members that converge in both runs, dropping
   the worst member changes the residual trajectories by median 23–27 % (max
   1e5) and the diverging set by ±5 — but the CONTROL of simply running the
   same token twice already gives median 20 % and ±1–2 in the diverging set,
   and a full re-mint of the same batch gives the same. So (i) the batched
   cuDSS solve is run-to-run **nondeterministic** at the ~20 % level in the
   refinement residual (this also explains B's varying `collected` counts), and
   (ii) no cross-member coupling is detectable above that noise. The letter
   must therefore NOT say a failing member "corrupts a shared quantity": in
   spineax's JAX-side loop there is no global reduction at all; refinement is
   unusable in batch because a fixed step count cannot be right for a batch in
   which 3–50 % of members have an unstable (static-pivot-perturbed) factor
   while the rest converge — and there is no per-member stopping test.
12. **Run B track v̄=1.0: 4.41 ± 0.83 solves/s** (A 35.75 → 8.1×), but the
   spineax LRU (capacity 8) logged **39 rebuilds inside the timed loop**
   (registry at capacity). All subsequent stages run with
   `SPINEAX_FACTOR_CACHE=64` (set in `run_queue.sh`); B's rebuild counts are in
   the per-run logs and the effect (≈ tens of extra batched factorizations per
   ~1600) is small but nonzero — the fresh-process B rerun will use 64.

13. **Run B track v̄=1.5: 4.5 solves/s with only 958–968 / 1000 collected per
   repeat** — 3–4 % of members terminate on a non-success code in ILB-off mode
   (v̄=1.0: 4.41 ± 0.83). In ILB-on mode such failures are silently replaced;
   the WS-dbg census (`term_hist`) will give the codes. This is a G-relevant
   fact for jaxipm regardless of mode.

14. **Run F/G/H/I1 at paper scale — nav 90° HR, N=500, 2000 solves (08:17–08:40):**
   G: 2000/2000 code-1 converged, zero hidden failures (term census
   [106992, 2008, 0, 0, 0, 0]); iterations median 33 / p90 83 / p99 146 / max
   214; KKT error at termination max 9.99e-9. F: over 109 000 slot-iterations —
   full step 70.4 %, SOC 17.9 %, backtracked LS 9.2 %, hot-restart init 2.3 %,
   full-restoration entry 121 (0.11 %), in-restoration iterations 72, soft
   restoration 13, watchdog 9, **tiny step 0**. Distinct branches active per
   fused iteration: mean 4.76, max 7; histogram over 218 fused iterations
   (1…7 distinct): 1, 0, 9, 74, 97, 30, 7. → "six of seven branches fire;
   tiny-step never; ≥4 distinct branches are active in 95 % of fused
   iterations." I1: backward error of the Newton step median 3.3e-15, p90
   4.8e-8, p99 8.3e-7, max 6.5e-5 (20 % of steps above 1e-8); relative-to-rhs
   residual median 0.13, p90 1.65. L: E[max₅₀₀ K]/E[K] from this run = 4.5
   (measured A/B = 6.6).

15. **Census, remaining HR configs (12:00–13:00):** track v̄=1.5: 1000/1000
   converged, 0 hidden failures, iters 44.5/37/226, branches full 79.5 % / SOC
   9.7 % / backtrack 6.1 % / **tiny-step 2.4 %** / WD 74 / SFR 8 / resto-entry
   17 / in-resto 54, distinct per fused iteration mean 5.38 max **8**; track
   v̄=2.0: 1000/1000, 0 hidden, iters 46.6/43/163, full 89.1 % / SOC 1.4 % /
   backtrack 4.1 % / tiny-step 2.9 % / WD 271 / SFR 12 / resto-entry 44 /
   in-resto 70, distinct mean 5.70 max 8; multi N=2: 75/75, 0 hidden, iters
   133.3/131/189 (matches the paper table's 131.6 — the 34.8 in the Aug-10
   paper-driver npz came from the un-initialised first batch; item 6's
   comparability worry is resolved), branches full 93 % / SOC 3.5 % /
   backtrack 2.3 %, distinct mean 2.57 max 4. → In track-avoid **all seven
   branch types fire**; in nav six of seven (no tiny step).

16. **I2 first point (13:20): nav 90°, N=1, ir=0, 48 instances spread over the
   pool (WS, DEBUG): 48/48 converged, iterations mean 23.6 / median 22 / max 89**
   — IPOPT on the same pool: 22.8 mean. At N=500 (HR census) the mean is 44.4.
   So the per-problem iteration inflation is a property of the batched
   setting, not of the single cuDSS solve (whose accuracy is N-independent,
   item 9). Candidates: hot-restart re-initialisation dynamics, inertia-
   correction coupling through the batched factorization, or refinement-free
   steps interacting with restoration more often in large batches. The
   N ∈ {1, 8, 32} × ir ∈ {0, 100} grid (GPU 3) and the N=1 HR rung (GPU 2)
   separate these.

17. **WS-dbg census (14:00) — the failure tail is real and ILB-on hides it.**
   ILB-off, DEBUG, paper batch sizes: nav 90° 2000/2000 converged (iters mean
   58.5, max 465), nav 180° 1999/2000 (1 cap hit; max 427), track v̄=1.0
   993/1000 (**7 hit the 500 cap**; max 492), v̄=1.5 968/1000 (**32 cap**),
   v̄=2.0 858/1000 (**142 cap**), multi N=2 75/75. Every ILB-off failure is
   code 2 (max_iter); no tiny-step / restoration failures. The ILB-on census
   shows 0 failures because a slot still mid-solve when the solution buffer
   fills is dropped unreported, and the HR runs last only 218–261 fused
   iterations, so the cap is unreachable there. IPOPT (tol 1e-6) solved all
   1000 track v̄=2.0 instances with max 124 iterations. → G must report the
   cap-hit fractions from the ILB-off census; A's throughput excludes the
   instances that never finish.
18. **Run C bottom rung, nav 90° N=1 (ILB on, 3 reps): 9.82 / 9.47 / 9.58
   solves/s** (2000/2000 solved, every pool instance once). With A = 101.7 and
   B = 15.3: fusion + solve-level batching (2)/(1) = 1.6×, ILB (3)/(2) = 6.6×,
   total 10.6× on identical code; IPOPT P=1 pinned is 7.3.
19. **I2, N=1, ir 0 vs 100 (48 pool-spread instances, WS):** nav 90° iters
   23.6 → 21.4 (median 22 → 19), one refinement blow-up (residual 2.6e35)
   yet 47/48 converged; nav 180° 17.9; track v̄=1.0 32.9 (48/48), v̄=1.5 34.2
   (44/48), **v̄=2.0: only 10/48 converged, 38 hit the cap** (vs 14 % at
   N=250); multi N=2 103, N=4 193 (IPOPT 177 / 205). Refinement barely moves
   the iteration count at N=1; the track v̄=2.0 N=1 collapse is either the
   single-token spineax path or the hard tail — N=8 point pending.
   Mechanism hypothesis under test (ic_log added to DEBUG): cuDSS's static
   pivots are clamped to exactly ±1e-13, spineax's inertia threshold is
   1e-13, so perturbed pivots' signs decide the inertia count and may trigger
   spurious Hessian perturbations.

20. **Census stage complete (14:46) — tables in `results/census_{F,G,H,I1}*.md`,
   `results/L_orderstats_vs_measured.md`.** Highlights: F — distinct branches per
   fused iteration, ILB-on: nav 4.76 / 4.35, track 3.99 / 5.38 / 5.70, multi 2.57 /
   2.13 (≥3 distinct in 97–99.6 % of nav/track iterations); ILB-off batches are
   less diverse (2.7–4.0) because converged members idle. I1 — fraction of Newton
   steps with relative residual > 1e-6 at ir=0: 0.84–0.98 in every batched
   config; with ir=100 (N=1) it is 0.008. L — with jaxipm's own K:
   predicted 3.7–4.5 (nav/track) vs measured fused-iteration ratio WS/HR
   6.5–9.3 and throughput ratio 5.6–10.5; **every track ILB-off batch ran to
   the 501-iteration cap** (2004 fused iterations / 4 batches) because ≥1
   member hit max_iter, so the order statistic must include the cap-hitters
   (added as a column: predicted incl. cap hits). multi: WS/HR fused ratio
   0.95 — solve-level batching wins on identical problems, as the paper's
   own WS rows showed. Using the ILB-off census's own K (all pool instances
   once, cap-hitters at 500) the prediction is nav 6.79 / 6.57 vs measured
   6.50 / 6.98, track 8.48 / 7.24 / 4.48 vs 9.32 / 8.28 / 7.68 — the formula
   holds where the tail is finite; at v̄=2.0 the measured ratio is inflated
   because ILB-on never finishes the 14 % hard instances that ILB-off runs to
   the cap (HR's E[K] = 46.6 vs ILB-off's 111.7 on the same pool). That is a
   fairness caveat for A vs D as well. Fresh-process B first points: nav 180° 19.7 (in-process
   mean 10.1) — the in-process decay confirmed; nav 90° 15.1 vs 15.3.

21. **Spurious inertia corrections (16:50) — a primary mechanism for the
   iteration inflation.** New DEBUG log `ic_log` (Hessian perturbation dxs > 0
   applied on the step): nav 90° N=8: **80.6 %** of slot-iterations; I2 N=1
   with ir=100: track v̄=1.0 59.1 %, v̄=1.5 70.7 %, v̄=2.0 94.1 %, multi N=2
   89.2 %. IPOPT (tol 1e-8, print_level 5, 12 nav instances spread over the
   pool): regularization on **22.7 %** of iterations, and 0 % on 8 of the 12
   instances. Refinement depth does not change jaxipm's rate, so the cause is
   the inertia readout, not the solve: spineax counts inertia by the sign of
   the LDLᵀ diagonal with threshold 1e-13 (`inertia()`), while cuDSS's static
   pivoting clamps 136 pivots per KKT system to exactly ±1e-13 (item 9). The
   IC loop then perturbs the Hessian on iterations IPOPT would not, damping
   the Newton step. **Update 17:30 — not supported as the inflation driver:**
   nav 90° perturbation rate vs batch size: N=1 37.8 % (iters 24.0), N=8 78.8 %
   (24.5), N=32 25.5 % (50.9), N=500 HR 23.1 % (44.6) — at the paper batch size
   the rate equals IPOPT's 22.7 %, and the rate does not track the iteration
   inflation, which appears between N=8 and N=32. What does track N is the
   batched solve's solution error from the I1 check (‖x−x_ref‖/‖x_ref‖ median:
   N=1 0.40 %, N=8 0.32 %, N=500 2.7 %; max 1 % vs 10 %). Decisive test queued
   in the I2 grid: N=32 with ir=100 — if iterations return to ~24, step
   accuracy of the expanded block-diagonal factorization is the cause.

22. **Track v̄=2.0 small-batch failures are not a solve-accuracy or N=1-path
   effect:** WS, 48 pool-spread instances: N=1 ir=0 → 10/48 converged (38 cap),
   N=8 ir=0 → 11/48 (37 cap), N=1 ir=100 → 11/48 (37 cap). Full-pool ILB-off
   census at N=250: 858/1000 (14 % cap). Discriminating runs queued on GPU 1:
   the same 48 instances as one batch of 48, and a 250-instance subsample at
   N=250.

23. **The cap hits are a stagnation defect, not hard instances (18:20).** The
   same 48 track v̄=2.0 instances: as one batch of 48 → 42/48 converged; a
   250-instance subsample at N=250 → 213/250 (15 %, like the full census);
   every instance that converged at N=1 also converged at N=48. Signature of
   every cap-hitter, at every N (N=8: 35/37, N=250 census: 133/142 below 1e-25,
   the rest ~1e-20): an accepted full step on every iteration with a Newton
   step of backward error ~1e-35 — zero right-hand side, zero step, iterate
   never moves, termination test never satisfied. Inside a batch where the
   same instance converges, its steps have backward error ~1e-11. This is a
   batch-size-dependent solver defect (small N worse: 77 % at N=1/8, 12 % at
   N=48, 15 % at N=250 for v̄=2.0; 0.7 % v̄=1.0; 0 % nav) — it falsifies the
   "fusion does not change the algorithm" premise (run J) until fixed, and it
   means the ILB-off throughput numbers (B) are depressed by defect-induced
   501-iteration batches. Per-iteration trace of pool instance 21 at N=1 and
   N=48 running (`debug_stagnation.py`).

24. **ROOT CAUSE of the stagnation (18:45, `debug_stagnation.py`, track v̄=2.0
   pool instance 21):** at N=1 the solve is one step from convergence at
   iteration 43 (μ = 1e-11 floor, KKT error 5.8e-6, constraint violation 1e-6)
   when the inertia-correction loop runs away: δ_x jumps to 3.6e20 (past
   `max_hessian_perturbation` = 1e20), the step norm freezes at 1.8e7, the dual
   infeasibility grows by 1.1e7 per iteration, the objective is frozen at
   1108.535 — and every one of the remaining 450 steps is accepted as a full
   step. In the batch of 48 the same instance reaches μ = 1e-11 at iteration 55
   with δ_x = 0 and converges (KKT error 2e-6 → done). Mechanism: near
   convergence the Schur-complement pivots of the constraint block are tiny;
   cuDSS's static pivoting clamps them to ±1e-13 with a numerically arbitrary
   sign; spineax counts inertia by the sign of those diagonals with threshold
   1e-13, so the inertia test fails spuriously; the IC loop escalates δ by 8×
   per trip until it overflows the cap, and jaxipm keeps iterating with the
   resulting garbage step instead of declaring failure as IPOPT does when its
   perturbation limit is hit. Batch size enters only through which pivots get
   perturbed in the expanded block-diagonal factorization (and its run-to-run
   nondeterminism), which is why the failure rate varies 77 % → 12 % with N
   and why the same instance succeeds in one batch and not another. This one
   mechanism accounts for: the ILB-off cap hits (G), the extra IC
   perturbations vs IPOPT (23–94 % vs 23 %), a large part of the iteration
   inflation, and it is what ILB-on silently hides. Pivot readout (`cudss.query`, track v̄=2.0 N=1, every iteration): 206–207
   of the 1136 pivots per KKT system are clamped to exactly |d| = 1e-13
   (npivots = 206–207), i.e. 18 % of the diagonal; the inertia count (626/510
   expected) is decided by the signs of those clamped entries. Two N=1 runs of
   the same instance follow different trajectories (the second had not reached
   the runaway by iteration 48): the batched cuDSS factorization is run-to-run
   nondeterministic, so the failure is a per-run event with a batch-size-
   dependent probability. Fix path (not applied —
   changes the solver): treat δ overflow as step failure → restoration/abort;
   make the inertia test robust to clamped pivots (e.g. exclude |d| == 1e-13
   pivots, or refine the Schur block), or set cuDSS pivot options.

25. **Run E complete (19:00, `results/E_madnlp.md`):** MadNLP (ExaModels +
   MadNLPGPU/cuDSS, SparseCondensedKKT, tol 1e-8) on the full jaxipm pool, 5
   repeats, 100 % SOLVE_SUCCEEDED everywhere: nav 90° 2.25 ± 0.28, 180° 2.74 ±
   0.02; track v̄=1.0 2.97 ± 0.99, 1.5 2.02 ± 0.01, 2.0 1.65 ± 0.00; multi N=2
   0.945 ± 0.07, N=4 0.428 ± 0.01 solves/s (paper protocol: wall includes the
   per-instance ExaModel build; solve-only differs by < 1 %). Per-repeat walls
   recovered from stop stamps (the first two launches had the repeat loop
   after `t0`). Iterations: nav 43.3 / 32.2 (max 125 / 184), track 38.0 / 47.4 /
   57.8, multi 148 / 283 — i.e. MadNLP's counts are at jaxipm's level (~44 on
   nav), about 2× IPOPT/MUMPS's 22.8 on the same pool: the other cuDSS-based
   solver shows the same inflation, which supports the linear-solver reading
   of R8-6a independently of jaxipm's own defect (item 24).

26. **Run C ladder (partial, `results/C_ladder_and_sweeps.md`, 19:10):**
   (1) jaxipm N=1 ILB-on: nav 9.62 ± 0.18 / 11.46 ± 0.25, track 5.52 ± 0.06 /
   2.15 ± 0.22 solves/s (IPOPT P=1 pinned, tol 1e-6: 7.26 / 8.16 / 6.33 / 4.72);
   N=1 ILB-off 7.02 / 8.39 / 4.63 / 1.97. (2) ILB-off at paper N, fresh-process:
   nav 14.6 ± 1.3 / 16.1 ± 3.1, track 4.86 / 4.59 / 5.25, multi 11.7 / 1.94.
   Ratios: (2)/(1) = 1.6 / 0.9 / 0.8 / 2.1 — fusion with solve-level batching
   gives almost nothing (the batch waits for its slowest — and, per item 24,
   often stagnating — member); (3)/(2) = 6.6 / 10.5 / 8.1 / 7.1 / 5.6; total
   (3)/(1) = 10.6 / 9.2 / 6.5 / 14.7 on identical code and GPU. multi: ILB-off
   beats ILB-on (0.79× / 0.65×). Remaining: N=1 for track v̄=2.0 and multi,
   ILB-off sweep tail, fresh-B tail, then D (CPU).

27. **I2 at N=32 (19:45): refinement does NOT cure the iteration inflation.**
   nav 90°, WS, 64 pool-spread instances: ir=0 → iters mean 43.7 / median 30 /
   max 187 (63/64 converged, IC rate 24.6 %); **ir=100 → mean 54.1 / median 31 /
   max 289 (IC rate 50.4 %)**. nav 180° ir=0: 59.3 / 39 / 337; track v̄=1.0
   ir=0: 72.0 / 42 / 335. Across N: medians 22 (N=1) → 22 (N=8) → 30–31 (N=32)
   → 33–34 (N=500), means 24 → 24.5 → 44–72 → 44–47, maxima 89–135 → 187–337.
   So the inflation vs IPOPT (22.8) is a tail phenomenon that (a) is absent at
   N ≤ 8, (b) is unaffected by solving every KKT system to 1e-15, and (c)
   coincides with the inertia-noise events of item 24 (runaways that recover
   after a large δ, or don't). The letter's claim "cuDSS solve accuracy causes
   the flat/inflated iteration counts" must be rewritten as: the static-pivot
   perturbation corrupts the *inertia signal*, and the IC loop's response
   (spurious δ_x, occasionally runaway) lengthens the tail of the iteration
   distribution and inflates the mean; the step's own accuracy (0.3–3 %) is
   a second-order effect (N=1: ir=100 changes 23.6 → 21.4). MadNLP shows the
   same ~2× inflation through the same library (item 25).

28. **I2 grid complete (22:00, `results/census_G_termination.md` rows `_i2`):**
   N ∈ {1, 8, 32} × ir ∈ {0, 100}, WS, pool-spread instances. nav/track: ir=100
   never lowers the mean iteration count at N ≥ 8 (nav 90° N=32: 43.7 → 54.1;
   nav 180° N=32: 59.3 → 52.0 with 4 cap hits; track v̄=1.0 N=32: 72.0 → 63.2;
   v̄=1.5: 59.6 → 50.8), and at N=1 it changes it by ≤ 2 iterations. The
   inflation is entirely the tail: medians 22 → 22 → 30 (nav 90°), maxima
   89 → 135 → 187. **multi-swap: refinement is actively destructive in batch**
   — N=2: ir=0 135 iterations, ir=100 328 with 17/64 cap hits at N=32;
   N=4: ir=0 0 cap hits, ir=100 31/48 (N=1) and 37/48 (N=8) cap hits — the
   per-member divergence of item 11 wrecking the solve. So "IR cannot be used
   in batch" is confirmed, with the per-member mechanism, and "IR would fix
   the iteration counts" is refuted. Ladder caveat: the N=1 ILB-on rung for
   track v̄=2.0 (0.13 solves/s) and multi N=2 (0.61) are dominated by the
   stagnation defect (77 % of solves burn 500 iterations), so their (2)/(1)
   ratios (44×, 18×) are artefacts of the defect, not of batching.

## 1. Manifest status

| Run | Status | Data | Next |
|---|---|---|---|
| A ILB-on ≥5 reps | **done** 2026-08-13 | `logs/jaxipm_variance_<tag>_fresh_rep{1..5}` (+ JIT-once ×5) | optional `run_queue.sh 1 timing_hr` for same-code-path parity with B/C |
| B ILB-off ≥5 reps | **running** (relaunched after fix 7) | `logs/jaxipm_ablation_<tag>_ws_ir0_b<N>_results.npz` | — |
| C sweep × {on,off}, N=1, peak mem | partial (on-only, no N=1, no mem) | `logs/jaxipm_batch_sweep_*` | `run_queue.sh 1 n1`; WS sweep; on-sweep with mem |
| D IPOPT core sweep | done at 1e-6; **rerun at 1e-8 queued** (CPU, after census) | `logs/casadi_cpu_throughput_*[_tol1e-08]` | analysis script needs the suffix switch |
| E MadNLP ≥5 reps | **done** 19:00 (jaxipm pool, tol 1e-8, 5 reps) | `logs/madnlp_<tag>_pool_tol1e-8_results.npz` + `.meta.json`, `results/E_madnlp.md` | old single run `logs/madnlp_<tag>` (n=80/50, track tol 1e-6) | `MADNLP_TOL=1e-8 MADNLP_N_RUNS=<pool> MADNLP_REPEATS=5 MADNLP_OUT_SUFFIX=_e julia tests/<s>/madnlp_*.jl` (~2 h GPU) |
| F branch census | instrumented, smoke-testing | — | 7 HR-dbg + 7 WS-dbg runs (`census`, `ws` stages) → `analyze_census.py` |
| G termination census | jaxipm 1/7 (nav 90° HR); IPOPT/MadNLP from existing | `results/existing_e5_census.md`, nav dbg npz | same runs as F |
| H quality + paired test | paired cost done; violations/KKT errors from dbg runs | `results/existing_e4_cost.md` | `analyze_quality.py`, `analyze_census.py`; IPOPT side needs D rerun |
| I1 batch KKT residual | instrumented | — | same runs as F |
| I2 N small, ir 0 vs 100 | half (correctness, 1 problem) | `tests/correctness/logs` | `jaxipm_ablation --batch {1,8,32} --ir-nsteps {0,100} --debug --max-solves 50` |
| I3 batch refinement divergence | script written, untested | `i3_refinement_divergence.py` | run on GPU 2 after smoke |
| J per-step injection | done for 1 problem | `tests/correctness/` | more problems + branch legend (after F) |
| K fusion profile | **not started** | — | last |
| L order statistics | proxy (IPOPT K): predicted 2.46 / 2.60 nav, 2.07–2.48 track | `results/existing_e11_orderstats.md` | recompute from jaxipm dbg K; compare with A/B |

## 2. Code changes (uncommitted)

- `jaxipm/solver.py`: scatter-collision fix (both paths); DEBUG path returns
  `(state, buf, write_idx, term_buffer, iter_buffer, term_hist[6], n_fused_iters,
  branch_log[i,b], kkt_res_log[i,b], kkt_err[k,4])`.
- `jaxipm/structures.py`, `initialization.py`, `search.py`: `IterateFlags.branch_id`
  (written only when `DEBUG_MODE`; throughput path unchanged).
- Drivers: `HOT_RESTART=0`, `JAXIPM_DEBUG=1`, `JAXIPM_IR_NSTEPS`, `JAXIPM_OUT_SUFFIX`
  hooks; nav driver on `make_batch_state`; `tp_out[:5]` unpacking.
- MadNLP drivers: `MADNLP_TOL / N_RUNS / REPEATS / OUT_SUFFIX`, per-repeat stamps,
  MadNLP status codes, metadata block (parse-checked, not yet run).
- `tests/rebuttal/`: `problems.py`, `jaxipm_ablation.py`, `run_queue.sh`
  (stages census / ws / timing / timing_ws / timing_hr / n1 / ir),
  `analyze_e1_cpu_scaling.py`, `analyze_existing.py`, `analyze_quality.py`,
  `analyze_census.py`, `i3_refinement_divergence.py`.

## 3. Queue (22:36 Aug 26: all GPU stages done; D running on the CPU, launched directly after its waiter was lost; ETA ≈ 01:10)

**Server rebooted ~11:10 on 2026-08-26** (all GPUs cleared). Before that the census stage had
finished nav 90°/180° and track v̄=1.0 (HR dbg). Relaunched 11:47 via the idempotent
`tests/rebuttal/resume_chain.sh 1` (skips configs whose npz exists; log
`tests/rebuttal/logs_chain_20260826_1147.log`): remaining census HR+WS dbg → analyses →
**D rerun (CPU, tol 1e-8)** → N=1 rung → I2 → fresh-process B → ILB-off sweep → **MadNLP (E)**.
B done 02:15; I3 done.

**Parallelised 13:25 (John: spread across idle GPUs):** GPU 1 census_ws → analyses (after the running multi N=4 HR census); GPU 2 N=1 rung; GPU 3 I2; GPU 4 ILB-off sweep; GPU 5 fresh-process B (alone on its GPU); GPU 6 MadNLP; then D on the CPU once every GPU worker has exited. Logs in . Caveat recorded: timing stages ran concurrently with JIT compiles on other GPUs (host CPU contention possible) — per-repeat stamps allow cross-checking; fresh-process B is the protected set.

**Overnight 2026-08-26/27 (John: "continue the experiments to completion — server largely to yourself"):**
`tests/rebuttal/overnight_0827.sh` waits for the D chain (pid 484116, tol 1e-8 CPU sweep, running
since 22:36) to exit, then: `analyze_e1_cpu_scaling` + `analyze_existing` with `CPU_SWEEP_SUFFIX=_tol1e-08`,
run-H violations (`analyze_quality.py`, CPU) in parallel with a K-profiler smoke test (GPU 4), then (John, 23:20: **GPUs 5 and 6 only** — someone else uses the box overnight) two serialized chains: GPU 5 = K smoke → `k_profile` → `sweep_hr`; GPU 6 = `n1 i2` → `sweep_ws_ext` → `run_J.sh` (12 nav pool instances,
per-step injection with branch ids; best effort — John warns the comparison is stochastic because of
poorly conditioned solves). Logs: `tests/rebuttal/logs_par_20260827/`. New code: `jaxipm/utils/kscope.py`
+ `jax.named_scope("K_*")` markers in search.py/quantities.py (metadata only, zero runtime effect),
`tests/rebuttal/k_fusion_profile.py`, `analyze_J.py`, `run_J.sh`, `CORR_INSTANCE` hooks in
`tests/correctness/*_correctness.py` (per-instance `ipopt_logs_inst<k>/`, `jaxipm_correctness_inst<k>.npz`,
`jx_branch`), `tests/rebuttal/problems.nav_pool_instance`.

29. **Run D at tol 1e-8 (2026-08-27 00:53, `results/e1_cpu_scaling_tol1e-08.md`, raw
    `casadi_cpu_throughput_<tag>_{phys,smt}_c<P>_tol1e-08_results.npz`, 98 files, 100 % success
    everywhere):** the tolerance change does not move the picture. P=1: nav 7.66/8.44, track
    6.51/5.18/2.99, multi 0.40/0.14 solves/s (jaxipm / P=1 = 13.3×, 12.5×, 5.5×, 6.1×, 10.3×,
    22.1×, 9.6×). 64 physical cores: nav 350/354, track 260/221/126, multi 15.4/5.6 solves/s;
    128 SMT: 362/431, 291/266/128, 18.7/7.9. Node / jaxipm = 3.4×/3.3×/7.3×/7.0×/4.1×/1.8×/4.1×
    (phys) and 3.6×/4.1×/8.2×/8.4×/4.1×/2.2×/5.9× (SMT). Crossover P = 8–32 phys. Compared with the
    tol-1e-6 sweep (item 3): nav +14 %, track −4/−3/−9 %, multi −2/−7 % at P=64 — IPOPT's
    per-solve cost is barely tolerance-sensitive here (it converges quadratically once close).
    These are the numbers to quote for R11-1; the 1e-6 sweep is superseded.

30. **Figure 6 (correctness ECDF) with the branch-type legend restored (2026-08-27 01:35):**
    the paper problem was re-run through the per-step injection test with `DEBUG_MODE` on
    (`CORR_DEBUG=1`, output `tests/correctness/logs/jaxipm_correctness_branches.npz`, paper npz
    untouched) so every injected step carries the branch jaxipm's fused body took. Figure:
    `tests/correctness/figures/validation_state_ecdf_branches.pdf` (paper fonts/layout;
    `validation_state_ecdf.pdf` = legacy 3-category version, July original kept as
    `.jul15-bak.pdf`). Legend counts over the 84 plotted steps: full step 46, full step in
    restoration 27, backtracking 6, second-order correction 3, restoration entry 1, initial
    point 1. Digest: median Δx ≈ 1e-12, 95 % of steps ≤ 1e-9; restoration entry/exit
    timeline identical to IPOPT's (iters 6–34); the single large outlier (Δx = 5.4e+02, a
    backtracking step at k = 36) is the step immediately after restoration exit — the known
    resto-exit pipeline offset (the script masks the exit iterate itself but not the step
    after it). The two ~1e-8 points are SOC steps. Re-running changes individual small Δx
    values slightly (nondeterministic cuDSS pivoting), not the branch mix.

31. **Overnight 08-27 outcome (GPUs 5+6 only) — partially contaminated.** At 01:43 another user
    started four ~10 GB training jobs on **GPU 5** (still running at 10:00; host load average ≈ 80).
    Everything timed on GPU 5 after 01:43 shares the GPU with them: the ILB-on harness sweep
    (`sweep_hr`) is therefore clean only for nav N ≤ 1000 (run 00:55–01:40: 97 / 108 / 111 / 114
    solves/s for N = 125/250/500/1000 at 90°, i.e. above the Aug-10 sweep's 84–98 — the scatter fix
    and warm cache), and **contaminated for track (17–31 vs 33–37 on Aug-10) and multi (4.3 vs 8.3)**;
    its peak-memory column is still valid (memory does not depend on co-tenants). Failures: nav HR
    N = 2000 → CUDA OOM (co-tenants hold 40 GB), nav HR N = 4000/8000, track HR N = 1000/2000 and
    nav WS N = 8000 (rep 2, after rep 1 ran at 12.5 solves/s) → `cuDSS analysis` alloc failure
    (status 2), multi N=4 HR N = 25 → `free(): invalid pointer` after an 11-min XLA compile (crash,
    not reproduced). **GPU 6 was exclusive** (only my process listed): the ILB-off extension is clean —
    nav N = 4000: 12.2 / 12.5 solves/s (10.9 GB), track N = 2000: 4.14 / 3.92 / 4.05, N = 4000:
    3.76 / 3.71 (13 GB) — ILB-off is flat-to-falling from N ≈ 125 upward, as the order statistic
    predicts (L). N=1 rung completed for multi N=4: 0.46 (ILB-on) / 0.43 (ILB-off) solves/s, so the
    ladder is now complete for all seven configs; multi N=4 (2)/(1) = 4.5×, (3)/(2) = 0.65× (ILB is
    *slower* than solve-level batching for both multi configs, 0.79× and 0.65×). I2 multi N=4,
    N = 32, ir = 100: 48 cap hits / 16 converged out of 64 (ir = 0: 0 cap hits) — refinement in batch
    is destructive, as for N=2. **Action needed:** re-run `sweep_hr` for track and multi (and nav
    N ≥ 2000) on an exclusive GPU (~3 h); until then the ILB-on curve is the Aug-10 sweep (1 rep,
    clean) and the harness run supplies peak memory only. Remaining queue (GPU 6, serialized):
    WS track 2.0 N = 4000 → WS multi N = 1200/2400 → J (12 instances) → K → run-H violations.
    → John (10:10): **re-run queued on GPU 6** via `tests/rebuttal/ilb_on_rerun.sh` — starts when the
    GPU-6 queue (WS ext → J → K → run-H) reports done; contaminated npz go to
    `logs/contaminated_gpu5_0827/`, nav N ≤ 1000 kept; log `logs_par_20260827/gpu6_sweep_hr_rerun.log`.

32. **Run D repeats — IPOPT P=64 phys × 5, tol 1e-8, all seven scenarios (`now` set, 15:00–15:11,
    `results/D_p64_repeats.md`, files `*_phys_c064_tol1e-08_now_rep<r>_results.npz`, 100 % success):**
    nav 280.7 ± 13.9 / 310.9 ± 14.8, track 234.6 ± 9.5 / 185.3 ± 3.8 / 105.4 ± 3.7, multi 13.5 ± 0.7 /
    5.2 ± 0.2 solves/s → node/jaxipm 2.8× / 2.9× / 6.6× / 5.9× / 3.4× / 1.55× / 3.9×. These ran on a
    loaded host (other users; 5-min load average 58 → 151 across the five repeats, stamped in each
    file as `metadata.loadavg_at_save`) and sit 10–20 % below the overnight single-run sweep values
    (350/354, 260/221/126, 15.4/5.6; STATUS 29), which is the contention, not the variance: within-set
    sd is 2–5 %. A `quiet` set (same command, fires automatically when the 5-min load < 8,
    `tests/rebuttal/logs_par_20260827/d_repeats_quiet.log`) will give the clean ±. Recommendation for
    the cells: quote the quiet set once it exists; until then the sweep value with the `now` sd.

33. **Afternoon 08-27 fixes/queue.** (a) Run J: all 12 jaxipm-side runs failed at 15:44–17:19 —
    the correctness loader (`load_state`) built the state's `args` without the instance's `x0_ic`
    constraint argument, so the injected pytree did not match jaxipm's; fixed (`_USER_C_ARGS`),
    IPOPT dumps for all 12 instances (`tests/correctness/ipopt_logs_inst<k>/`) are valid and the
    jaxipm half is re-running on GPU 6 (`logs_par_20260827/gpu6_J2.log`). (b) Run K: the smoke run
    compiled and scoped fine (84 094 of 96 566 HLO instructions carry a `K_` scope) but the trace
    parser expected an "XLA Ops" line; GPU traces here are per-stream with kernel names — parser
    rewritten (fusion kernels → HLO op_name scopes; cuDSS kernels classified factorize / solve /
    analysis by name; memcpy separate) and validated on the smoke trace (N=8: cuDSS factorize 15 %,
    solve 6 %, analysis 35 % (!), memcpy 6 %, XLA 38 % of device time — the analysis share at N=8
    suggests re-analysis inside the loop; the real run logs spineax rebuild counters to settle it).
    K re-queued on GPU 6 after the ILB-on re-run (`k2_waiter.log`). (c) Run H violations:
    `analyze_quality.py` needs the GPU (spineax FFI has no host handler) — running on GPU 5
    (`analyze_quality_gpu5.log`). GPU-6 order now: J2 → clean ILB-on sweep → K.

34. **Primary-table gaps (John, 08-27 18:00 — highest priority):** (i) cost sd for IPOPT-default,
    MadNLP and jaxipm-IL; (ii) ‖c‖∞ at the returned point for MadNLP and both IPOPT configs (as
    already computed for jaxipm); (iii) throughput mean ± sd for the IPOPT-default configuration
    (one IPOPT/MUMPS process on the whole CPU) on all three tests. Work: `analyze_primary_table.py`
    → `results/primary_table_quality.md` (cost mean ± sd over the pool for every solver from the
    existing npz; ‖c‖∞ / inequality / bound violations with jaxipm's own c, d, bounds for every
    solver whose inputs U are on disk). The paper drivers saved only X, so two baseline reruns were
    added to get U: IPOPT-default at tol 1e-8 on the full jaxipm pool with `U_all` saved
    (`casadi_<tag>_pool_tol1e-08_results.npz`, CPU, `IPOPT_OUT_SUFFIX`/`IPOPT_N_RUNS` hooks) and
    MadNLP tol 1e-8 pool with the raw solution vector saved (`madnlp_<tag>_pool_tol1e-8_z_results.npz`,
    `z_all`, GPU 1). IPOPT-default throughput repeats: `ipopt_default_repeats.sh` (5 × the paper
    protocol, N_RUNS_seq pool, tol 1e-8, host load stamped) → `analyze_ipopt_default.py` →
    `results/ipopt_default_repeats.md`. GPU 5 vacated at John's request (co-tenants); GPUs 1 and 6 in use.
    The quiet-host D-repeat waiter was stopped so it cannot collide with the IPOPT-default timing.

35. **IPOPT-default throughput repeats (one IPOPT/MUMPS process on the whole CPU, tol 1e-8, paper pool
    sizes 80/50/50, 5 repeats, 18:45–19:52; `results/ipopt_default_repeats.md`, files
    `casadi_<tag>_tol1e-08_rep{1..5}_results.npz`; 100 % success):** nav 6.61 ± 0.39 / 6.99 ± 0.28,
    track 6.06 ± 0.30 / 4.61 ± 0.27 / 2.55 ± 0.14, multi 0.33 ± 0.01 / 0.11 ± 0.00 solves/s
    (paper column, tol 1e-6, single run: 7.87 / 8.76 / 7.12 / 5.42 / 3.00 / 0.40 / 0.13). Host 5-min
    load average 79–103 during every repeat (other users), so the level is ~15–25 % below the paper
    run; the sd (4–6 %) is the repeat-to-repeat spread under that load. Iteration counts are
    identical across repeats (deterministic). Cost over the paper pool at 1e-8 equals the paper's
    to three decimals (e.g. nav 90° 516.959 ± 20.332); the full-pool pass (n = 2000/1000/75)
    gives 516.953 ± 20.286 etc. — see `results/primary_table_quality.md` for all solvers.

36. **Primary-table quality columns (`results/primary_table_quality.md`, 20:05; rebuilds automatically
    once the MadNLP z-rerun finishes for track 2.0 / multi N=2):** cost mean ± sd over the pool and
    violations at the returned point with jaxipm's own c/d/bounds for every solver. Findings:
    (a) IPOPT-default and IPOPT P=64 return bit-identical solutions (same solver, tol, pool) —
    quote one cost column for both. (b) Cost sd is the pool spread (nav ≈ 20, track ≈ 80–150,
    multi 0); jaxipm-IL's paper npz sd is inflated by the obj=0 scatter-collision outlier (nav 90°
    517.16 ± 24.06 vs 516.48 ± 20.23 in the census run with the fix) — quote the census-run row for
    jaxipm-IL cost. Paired differences remain as in `existing_e4_cost.md` (jaxipm ≈ 0.1 % lower on
    nav, different local optima on 11–27 % of track). (c) ‖c‖∞ (equality residual): IPOPT median
    1e-12–1e-10 / max ≤ 1e-8; jaxipm-IL median 1e-13–3e-12 / max ≤ 1e-8; MadNLP exactly 2.0e-8 on
    every problem = its bound relaxation of the fixed initial state (x₀ is a relaxed variable
    bound in the ExaModels model; dynamics rows ~1e-15) — verified by constraint block. (d)
    Inequality violation ≤ 1e-8 for all (tolerance-level), MadNLP 2–3e-8. (e) Bound violation:
    jaxipm ≤ 1e-8; IPOPT 9e-6 on track/multi (IPOPT's `bound_relax_factor` 1e-8 relative to the
    ±900 rad/s motor bounds), MadNLP the same 9e-6; nav has no active bounds (0). (f) Success
    100 % everywhere at tol 1e-8 for IPOPT/MadNLP; jaxipm-IL paper npz 98.7 % on multi is the
    scatter-collision artefact (census run 100 %).

37. **Run J (12 nav pool instances, per-step injection with branch ids; 20:20; `results/J_correctness.md`,
    `figures/J_step_ecdf.pdf`):** 399 injected steps; median Δx 1.1e-13, 98.7 % of steps within
    1e-6 of IPOPT's, 1.3 % drift. By branch: full step 238 steps (100 % tight, median 2e-13), SOC 28
    (100 %, 1e-10), backtracking 23 (91 %), restoration entry 3 (33 %). The 5 drifting steps are
    all at restoration entry/exit boundaries (instances 545 k=5, 727 k=2, 1999 k=9 — the same
    pipeline-offset mechanism as the paper problem's k=36 outlier) — not a solver disagreement
    inside a phase. Ten of twelve instances reproduce IPOPT at every step; restoration-flag
    timelines agree on 11/12. John's caveat stands: individual Δx values vary run-to-run with cuDSS
    pivoting; the tight fraction and the branch-wise medians are the stable statistics.
    GPU-6 chain now on the clean ILB-on sweep (no co-tenants on GPU 6 at start), then K.

38. **Branch-activation table (paper `tab:branch-activation`) — the tiny-step (4) and watchdog (3)
    columns are the stagnation defect, not algorithmic branches (08-27 21:20,
    `results/census_F_branches_terminated.md`).** Cutting the census timelines into solves shows that
    every solve containing a tiny-step iteration (track 1.5: 14, track 2.0: 38) and nearly every
    solve containing a watchdog iteration (8/8, 14/17) never terminated inside the census window —
    they are the zero-rhs stagnated members (IC-runaway, STATUS 22–24), which take the tiny-step
    branch every iteration (1545 of 1913 TS iterations are followed by another TS) and trip the
    watchdog after 10 shortened iterations. Over solves that terminated (2000/1000/75 per config),
    tiny step is 0.00 % everywhere and watchdog ≤ 0.05 %; full 69–97 %, backtracking 1–9 %, SOC
    1.4–20 %, restoration ≤ 0.2 %, restart 0.5–2.3 %; distinct branches per fused iteration
    unchanged (mean 2.1–5.5, max 4–8). Recommendation: publish the terminated-solves table (or set
    (3)/(4) to "0 / 0.0x" with a footnote that the non-zero raw values come from the stagnated
    members that are excluded from the throughput and quality statistics anyway).

## 39. Paper Figure 6 provenance + legend (Aug 28)
- Paper `validation_state_ecdf.pdf` (md5 71ff849a…) was NOT made by `tests/correctness/analyze_results.py`; it is
  `/home/john/code/jaxipm/src/problems/correctness_test/plot_validation_state_diffs.py` (2026-06-14) over the archived
  state-pair runs `/home/john/code/jaxipm/data/validation_runs/run_2026060{9,10}_*` (5 runs × 93 iters, ir_nsteps=100,
  isclose metric max|a−b|/(1+|b|) over all compared leaves, last 10% dropped → 420 samples). Reproduced bit-for-bit in shape.
- Re-rendered with legends from the same data: `tests/correctness/figures/validation_state_ecdf_paper_legend.pdf`
  (paper categories: regular 230, restoration 140, entry 5, exit 5, SOC 40) and `..._paper_branches.pdf`
  (release branch semantics: full step 230, full (resto) 140, SOC accepted 15, backtracking 25 (SOC tried→rejected→BT),
  entry 5, exit 5; watchdog/tiny-step/SFR 0 — same as the Aug 27 rerun). Scripts: `tests/rebuttal/fig6_collect_paper_samples.py`
  (jaxipm env, reads the archives) → `results/fig6_paper_samples.npz` → `tests/rebuttal/fig6_paper_legend.py`.
- Note: the June IPOPT trace (93 iters) and the July/Aug one (94 iters) are different IPOPT runs of the same problem
  (solutions agree to 3e-8), so the Aug `jx_branch` ids were not copied across; branches were read from the archived states.

## 40. Run K complete (Aug 28, GPU 6) — attribution fix + a real solver hotspot found
- First K pass (12:05–12:55) mis-attributed kernels: XLA names the instruction `foo.52` but the profiler reports the
  kernel as `foo_52`; the old lookup stripped the suffix (→ "unscoped" 53–64% for track/multi, or the wrong fusion).
  Fixed (`_N`→`.N`), old outputs in `results/K_old_lookup_0828/`, `logs/ktrace_old_lookup_0828/`; the compiled HLO is
  now saved next to each trace (`logs/ktrace_<label>/compiled_hlo.txt`). Re-run 13:09–14:02 → `results/K_fusion_profile.md`.
- Device time per fused iteration (unscoped now ≤2.5%): nav90 b500 78 ms (cuDSS factorize 41%, solves 10%, analysis 7%;
  branch-unique work 22%); nav90 b1 3.6 ms device / 12 ms wall (launch-bound, 1442 kernels); track1.5 b250 116 ms
  (branch-unique 68%!); multi2 b75 54 ms (branch-unique 58%).
- **Why track/multi are dominated by the SOC/SFR/quantities regions: one kernel, `jit(norm)/reduce_sum` over an
  `[N, nxL_resto, nxL_resto]` tensor** (track 250×1406×1406, multi 75×2390×2390) = 59% / 48% of device time.
  Source: `jaxipm/quantities.py:625-628`, `calc_barrier_obj` damping term
  `jnp.linalg.norm(sxL[:nxL] * sxL_mask, ord=1)`: `sxL` is a column `[nxL,1]`, the mask is 1-D `[nxL]` → broadcasts to an
  `[nxL,nxL]` outer product, and `ord=1` on a 2-D array is the matrix 1-norm = max_j Σ_i |s_i m_j| = ‖s‖₁·max(mask).
  It bites in the RESTORATION-phase quantities (`nstqfr`, nxL = 386+2·510 = 1406 for track), which the fused body
  evaluates for every lane at every SOC/BT/SFR trial (filter_select computes both branches).
  Numerics: intended Σ_{one-sided} s_i, computed ‖s‖₁ if any bound is one-sided. nav: x has no bounds, in resto all
  lower bounds (p,n) are one-sided → mask all ones → computed == intended (this is why the IPOPT injection test matches).
  track/multi: two-sided x bounds are also summed into the resto damping term (κ_d=1e-5·μ·Σ two-sided slacks): a small
  deviation from IPOPT's resto barrier objective, only inside restoration. Regular phase: nav nxL=0; track/multi
  sxL mask all-zero (x*0 is not folded by XLA for floats) → 386²/772² per trial, minor.
  One-line fix (`sxL[:nxL].reshape(-1) * sxL_mask` etc.) would cut track/multi per-iteration device time roughly 2×;
  NOT applied — it changes the solver mid-rebuttal (all timings + resto numerics); John's call.

#!/usr/bin/env bash
# Run every solver test. Sequential (all GPU-bound), continues past failures,
# streams each run's output to the terminal, prints a PASS/FAIL summary.
# Lives in tests/ but runs from the repo root so the `tests.*` imports resolve.
# Per-run results (timings, success, iters) are saved by each solver into its
# own logs/<solver>_*.npz — this script keeps no separate log files.
#
#   tests/run_all_tests.sh            # default GPU 0
#   CUDA_VISIBLE_DEVICES=GPU-xxxx tests/run_all_tests.sh    # pin a GPU/UUID
#   PY=/path/python JL=/path/julia tests/run_all_tests.sh   # pin interpreters
set -u
cd "$(dirname "$0")/.."                     # repo root (script lives in tests/)

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
PY="${PY:-python}"
JL="${JL:-julia}"
echo "GPU=$CUDA_VISIBLE_DEVICES"

declare -a RESULTS
run() {                                    # run <name> <command...>
  local name="$1"; shift
  echo "[$(date +%H:%M:%S)] >>> $name"
  if "$@"; then
    RESULTS+=("PASS  $name"); echo "    PASS"
  else
    RESULTS+=("FAIL  $name"); echo "    FAIL"
  fi
}

# ── Python solvers: module form so `tests.*` imports resolve from repo root ──
for prob in quad_nav_circle quad_multi_swap quad_track_avoid; do
  for solver in ipopt jaxipm; do
    mod=$(ls tests/$prob/${solver}_*.py | head -1 | sed 's#/#.#g; s#\.py$##')
    run "${prob}.${solver}" $PY -m "$mod"
  done
done

# ── Correctness: Stage 2 validation only (ipopt_logs/ already present) ───────
run "correctness.jaxipm" $PY -m tests.correctness.jaxipm_correctness

# ── MadNLP / Julia solvers ──────────────────────────────────────────────────
run "quad_nav_circle.madnlp"  $JL tests/quad_nav_circle/madnlp_quad.jl
run "quad_multi_swap.madnlp"  $JL tests/quad_multi_swap/madnlp_multi_swap.jl
run "quad_track_avoid.madnlp" $JL tests/quad_track_avoid/madnlp_track_avoid.jl

# ── Figures: regenerate the paper figures from the fresh logs/ data ──────────
# Runs after all solvers so each analyzer sees complete data (track_avoid reads
# casadi/madnlp/jaxipm). correctness is omitted on purpose — its analyze_results
# is unresolved (open-loop data can't reproduce the paper's single-step ECDF).
for prob in quad_nav_circle quad_multi_swap quad_track_avoid; do
  run "${prob}.figures" $PY -m "tests.${prob}.analyze_results"
done

echo; echo "===== SUMMARY ====="
printf '%s\n' "${RESULTS[@]}"

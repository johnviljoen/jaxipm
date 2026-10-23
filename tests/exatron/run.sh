#!/usr/bin/env bash
# Run ExaTron (slack + augmented Lagrangian) on all three quadrotor tasks, one
# Julia process per variant in parallel (the sparse TRON path is single-threaded
# CPU; the GPU is not used — ExaTron's GPU kernel is limited to n <= 32, see README).
#   tests/exatron/run.sh [GPU=1]
set -u
cd "$(dirname "$0")/../.."
export CUDA_VISIBLE_DEVICES="${1:-1}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
JL="${JL:-julia}"
LOGS=tests/exatron/logs; mkdir -p "$LOGS"
pids=()
for spec in "nav 90" "nav 180" "track 1.0" "track 1.5" "track 2.0" "multi 2" "multi 4"; do
  set -- $spec
  $JL tests/exatron/exatron_quad.jl "$1" "$2" > "$LOGS/exatron_$1_$2.log" 2>&1 &
  pids+=($!)
  echo "started $spec (pid $!)"
done
for p in "${pids[@]}"; do wait "$p"; done
python -m tests.exatron.analyze_results

#!/usr/bin/env bash
# Run J extension: per-step injection correctness on a sample of nav-circle pool
# instances, with the branch taken at every step (DEBUG_MODE branch ids).
#   tests/rebuttal/run_J.sh [GPU] [instance ...]
set -u
cd "$(dirname "$0")/../.."
GPU="${1:-5}"; shift || true
INST="${*:-0 182 364 545 727 909 1090 1272 1454 1636 1818 1999}"
PY1=/home/john/.conda/envs/jaxipm/bin/python          # patched IPOPT (dumps per-iteration state)
PY2=/home/john/.conda/envs/jaxipm_latest/bin/python
LOGD=tests/rebuttal/logs; mkdir -p "$LOGD"
log() { echo "[$(date +%F' '%T)] $*"; }
for k in $INST; do
  if [ ! -d "tests/correctness/ipopt_logs_inst$k" ]; then
    log ">>> J stage 1 (IPOPT) instance $k"
    PYTHONPATH=. CORR_INSTANCE=$k OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=$GPU XLA_PYTHON_CLIENT_PREALLOCATE=false "$PY1" tests/correctness/ipopt_correctness.py > "$LOGD/J_ipopt_inst$k.log" 2>&1 \
      || { log "    FAIL stage 1 inst $k"; tail -3 "$LOGD/J_ipopt_inst$k.log"; continue; }
    grep -E "iterations|status" "$LOGD/J_ipopt_inst$k.log" | sed 's/^/    /'
  fi
  if [ ! -f "tests/correctness/logs/jaxipm_correctness_inst$k.npz" ]; then
    log ">>> J stage 2 (jaxipm injection) instance $k"
    PYTHONPATH=. CORR_INSTANCE=$k CUDA_VISIBLE_DEVICES=$GPU XLA_PYTHON_CLIENT_PREALLOCATE=false \
      "$PY2" tests/correctness/jaxipm_correctness.py > "$LOGD/J_jaxipm_inst$k.log" 2>&1 \
      || { log "    FAIL stage 2 inst $k"; tail -3 "$LOGD/J_jaxipm_inst$k.log"; continue; }
    grep -E "single-step|first k with|saved" "$LOGD/J_jaxipm_inst$k.log" | sed 's/^/    /'
  fi
done
"$PY2" -m tests.rebuttal.analyze_J > "$LOGD/analyze_J.log" 2>&1 && log "J analysis done" || log "J analysis FAILED"
log "J DONE"

#!/usr/bin/env bash
# Overnight 2026-08-27 (John: GPUs 5 and 6 ONLY). Wait for the D (CPU, tol 1e-8) chain to exit,
# run the CPU-side analyses, then two serialized GPU chains.
set -u
cd "$(dirname "$0")/../.."
PY=/home/john/.conda/envs/jaxipm_latest/bin/python
LOGD=tests/rebuttal/logs_par_20260827; mkdir -p "$LOGD"
log() { echo "[$(date +%F' '%T)] $*"; }
DPID="${1:-484116}"
while kill -0 "$DPID" 2>/dev/null; do sleep 60; done
log "D chain (pid $DPID) exited"
export SPINEAX_FACTOR_CACHE=64
CPU_SWEEP_SUFFIX=_tol1e-08 "$PY" -m tests.rebuttal.analyze_e1_cpu_scaling > "$LOGD/analyze_e1_tol1e-08.log" 2>&1; log "analyze_e1 tol1e-08 exit $?"
CPU_SWEEP_SUFFIX=_tol1e-08 "$PY" -m tests.rebuttal.analyze_existing > "$LOGD/analyze_existing_tol1e-08.log" 2>&1; log "analyze_existing tol1e-08 exit $?"
# K smoke (GPU 1) in parallel with run-H violations (CPU)
"$PY" -m tests.rebuttal.k_fusion_profile --scenario nav --variants 90 --batch 8 --warm-solves 16 \
      --profile-iters 3 --timing-iters 5 --gpu 5 --out-suffix smoke > "$LOGD/k_smoke.log" 2>&1 &
KS=$!
"$PY" -m tests.rebuttal.analyze_quality > "$LOGD/analyze_quality.log" 2>&1; log "analyze_quality exit $?"
wait $KS; KRC=$?; log "K smoke exit $KRC"
rm -f tests/rebuttal/results/K_fusion_profile_nav_s90_b8_smoke.json
# ---- GPU 5 chain: K profile -> ILB-on sweep ----
( if [ "$KRC" -eq 0 ]; then bash tests/rebuttal/resume_chain.sh 5 k_profile > "$LOGD/gpu5_k_profile.log" 2>&1; log "gpu5 k_profile done";
  else log "K smoke failed -- k_profile NOT run"; fi
  bash tests/rebuttal/resume_chain.sh 5 sweep_hr > "$LOGD/gpu5_sweep_hr.log" 2>&1; log "gpu5 sweep_hr done" ) &
# ---- GPU 6 chain: N=1 rung + I2 -> ILB-off extension -> J ----
( bash tests/rebuttal/resume_chain.sh 6 n1 i2 > "$LOGD/gpu6_n1_i2.log" 2>&1; log "gpu6 n1 i2 done"
  bash tests/rebuttal/resume_chain.sh 6 sweep_ws_ext > "$LOGD/gpu6_sweep_ws_ext.log" 2>&1; log "gpu6 sweep_ws_ext done"
  if grep -q '^EXIT 0' "$LOGD/J_ipopt_inst1000.log" 2>/dev/null; then bash tests/rebuttal/run_J.sh 6 > "$LOGD/gpu6_J.log" 2>&1; log "gpu6 J done";
  else log "J stage-1 test did not pass -- J NOT run"; fi ) &
log "GPU chains launched (GPUs 5 and 6 only)"
wait
log "GPU chains done"
"$PY" -m tests.rebuttal.analyze_C > "$LOGD/analyze_C_final.log" 2>&1; log "analyze_C exit $?"
"$PY" -m tests.rebuttal.analyze_census > "$LOGD/analyze_census_final.log" 2>&1; log "analyze_census exit $?"
"$PY" -m tests.rebuttal.analyze_L > "$LOGD/analyze_L_final.log" 2>&1; log "analyze_L exit $?"
log "OVERNIGHT DONE"

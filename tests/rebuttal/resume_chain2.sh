#!/usr/bin/env bash
# Idempotent resume of the rebuttal run chain after an interruption/reboot.
# Every jaxipm ablation config is skipped when its output npz already exists;
# the CPU sweep uses --resume; MadNLP configs are skipped when their npz exists.
#   tests/rebuttal/resume_chain.sh [GPU] [STAGE...]
#   stages: census_hr census_ws analyses d_cpu n1 i2 fresh sweep_ws madnlp
set -u
cd "$(dirname "$0")/../.."
GPU="${1:-1}"; shift || true
STAGES="${*:-census_hr census_ws analyses d_cpu n1 i2 fresh sweep_ws madnlp}"
PY="${PY:-/home/john/.conda/envs/jaxipm_latest/bin/python}"
JL="${JL:-/home/john/.juliaup/bin/julia}"
export SPINEAX_FACTOR_CACHE="${SPINEAX_FACTOR_CACHE:-64}"
LOGD=tests/rebuttal/logs; mkdir -p "$LOGD"
log() { echo "[$(date +%F' '%T)] $*"; }

# abl <logname> <expected-npz> <args...>
abl() {
  local name="$1" expect="$2"; shift 2
  if [ -f "$expect" ]; then log "skip $name (exists: $expect)"; return; fi
  # per-config lock so several workers can share a stage without duplicating a run
  if ! mkdir "$LOGD/.lock_$name" 2>/dev/null; then log "skip $name (locked by another worker)"; return; fi
  log ">>> $name : $*"
  "$PY" -m tests.rebuttal.jaxipm_ablation "$@" --gpu "$GPU" > "$LOGD/$name.log" 2>&1 \
    && grep -E "^\[VARIANCE\].*: n=|collected=|BRANCH|TERM CENSUS|^saved" "$LOGD/$name.log" | sed 's/^/    /' \
    || { echo "    FAIL (exit $?) -- see $LOGD/$name.log"; tail -4 "$LOGD/$name.log" | sed 's/^/    /'; }
}
NAV=tests/quad_nav_circle/logs; TRK=tests/quad_track_avoid/logs; MUL=tests/quad_multi_swap/logs

stage_census_hr() {
# ---- census F/G/H/I1: HR dbg then WS dbg, per variant ----
for m in hr; do
  for v in 90 180; do abl "census_nav${v}_${m}_dbg" "$NAV/jaxipm_ablation_sector${v}_${m}_ir0_b500_dbg_results.npz" --scenario nav --variants "$v" --mode "$m" --debug; done
  for v in 1.0 1.5 2.0; do abl "census_track${v}_${m}_dbg" "$TRK/jaxipm_ablation_v${v}_${m}_ir0_b250_dbg_results.npz" --scenario track --variants "$v" --mode "$m" --debug; done
  for v in 2 4; do abl "census_multi${v}_${m}_dbg" "$MUL/jaxipm_ablation_${v}_${m}_ir0_b75_dbg_results.npz" --scenario multi --variants "$v" --mode "$m" --debug; done
done
}

stage_census_ws() {
# ---- census F/G/H/I1: HR dbg then WS dbg, per variant ----
for m in ws; do
  for v in 90 180; do abl "census_nav${v}_${m}_dbg" "$NAV/jaxipm_ablation_sector${v}_${m}_ir0_b500_dbg_results.npz" --scenario nav --variants "$v" --mode "$m" --debug; done
  for v in 1.0 1.5 2.0; do abl "census_track${v}_${m}_dbg" "$TRK/jaxipm_ablation_v${v}_${m}_ir0_b250_dbg_results.npz" --scenario track --variants "$v" --mode "$m" --debug; done
  for v in 2 4; do abl "census_multi${v}_${m}_dbg" "$MUL/jaxipm_ablation_${v}_${m}_ir0_b75_dbg_results.npz" --scenario multi --variants "$v" --mode "$m" --debug; done
done
}

stage_analyses() {
"$PY" -m tests.rebuttal.analyze_census > "$LOGD/analyze_census_$(date +%m%d_%H%M).log" 2>&1
"$PY" -m tests.rebuttal.analyze_L > "$LOGD/analyze_L_$(date +%m%d_%H%M).log" 2>&1
log "census + analyses done"

}

stage_d_cpu() {
# ---- run D at tol 1e-8 (CPU only; GPU idle during this) ----
for s in quad_nav_circle quad_track_avoid quad_multi_swap; do
  log ">>> d_cpu $s"
  "$PY" -m "tests.$s.quad_casadi_cpu_throughput" --tol 1e-8 --resume > "$LOGD/d_cpu_${s}_tol1e-8.log" 2>&1 \
    && grep -E "wall=" "$LOGD/d_cpu_${s}_tol1e-8.log" | sed 's/^/    /' \
    || { echo "    FAIL -- see $LOGD/d_cpu_${s}_tol1e-8.log"; tail -5 "$LOGD/d_cpu_${s}_tol1e-8.log" | sed 's/^/    /'; }
done
CPU_SWEEP_SUFFIX=_tol1e-08 "$PY" -m tests.rebuttal.analyze_e1_cpu_scaling > "$LOGD/analyze_D_tol1e-8.log" 2>&1
CPU_SWEEP_SUFFIX=_tol1e-08 "$PY" -m tests.rebuttal.analyze_existing > "$LOGD/analyze_existing_tol1e-8.log" 2>&1
log "D done"

}

stage_n1() {
# ---- run C bottom rung: N = 1, both modes ----
for v in 90 180; do for m in hr ws; do abl "n1_nav${v}_${m}" "$NAV/jaxipm_ablation_sector${v}_${m}_ir0_b1_results.npz" --scenario nav --variants "$v" --mode "$m" --batch 1 --repeats 3 --no-save-z; done; done
for v in 1.0 1.5 2.0; do for m in hr ws; do abl "n1_track${v}_${m}" "$TRK/jaxipm_ablation_v${v}_${m}_ir0_b1_results.npz" --scenario track --variants "$v" --mode "$m" --batch 1 --repeats 3 --no-save-z; done; done
for v in 2 4; do for m in hr ws; do abl "n1_multi${v}_${m}" "$MUL/jaxipm_ablation_${v}_${m}_ir0_b1_results.npz" --scenario multi --variants "$v" --mode "$m" --batch 1 --repeats 3 --no-save-z; done; done

}

stage_i2() {
# ---- run I2: small N, ir 0 vs 100, pool-subsampled, DEBUG ----
for n in 1 8 32; do for k in 0 100; do
  for v in 90 180; do abl "i2_nav${v}_b${n}_ir${k}" "$NAV/jaxipm_ablation_sector${v}_ws_ir${k}_b${n}_dbg_i2_results.npz" --scenario nav --variants "$v" --mode ws --pool-subsample --debug --batch "$n" --max-solves 48 --ir-nsteps "$k" --out-suffix i2; done
  for v in 1.0 1.5 2.0; do abl "i2_track${v}_b${n}_ir${k}" "$TRK/jaxipm_ablation_v${v}_ws_ir${k}_b${n}_dbg_i2_results.npz" --scenario track --variants "$v" --mode ws --pool-subsample --debug --batch "$n" --max-solves 48 --ir-nsteps "$k" --out-suffix i2; done
  for v in 2 4; do abl "i2_multi${v}_b${n}_ir${k}" "$MUL/jaxipm_ablation_${v}_ws_ir${k}_b${n}_dbg_i2_results.npz" --scenario multi --variants "$v" --mode ws --pool-subsample --debug --batch "$n" --max-solves 48 --ir-nsteps "$k" --out-suffix i2; done
done; done

}

stage_fresh() {
# ---- run B fresh-process repeats ----
for r in 1 2 3 4 5; do
  for v in 90 180; do abl "fresh_nav${v}_ws_rep${r}" "$NAV/jaxipm_ablation_sector${v}_ws_ir0_b500_fresh_rep${r}_results.npz" --scenario nav --variants "$v" --mode ws --repeats 1 --no-save-z --out-suffix "fresh_rep${r}"; done
  for v in 1.0 1.5 2.0; do abl "fresh_track${v}_ws_rep${r}" "$TRK/jaxipm_ablation_v${v}_ws_ir0_b250_fresh_rep${r}_results.npz" --scenario track --variants "$v" --mode ws --repeats 1 --no-save-z --out-suffix "fresh_rep${r}"; done
  for v in 2 4; do abl "fresh_multi${v}_ws_rep${r}" "$MUL/jaxipm_ablation_${v}_ws_ir0_b75_fresh_rep${r}_results.npz" --scenario multi --variants "$v" --mode ws --repeats 1 --no-save-z --out-suffix "fresh_rep${r}"; done
done

}

stage_sweep_ws() {
# ---- run C: ILB-off batch sweep ----
for n in 125 250 500 1000 2000; do
  for v in 90 180; do abl "sweep_nav${v}_ws_b${n}" "$NAV/jaxipm_ablation_sector${v}_ws_ir0_b${n}_results.npz" --scenario nav --variants "$v" --mode ws --batch "$n" --repeats 3 --no-save-z; done
  for v in 1.0 1.5 2.0; do [ "$n" -le 1000 ] && abl "sweep_track${v}_ws_b${n}" "$TRK/jaxipm_ablation_v${v}_ws_ir0_b${n}_results.npz" --scenario track --variants "$v" --mode ws --batch "$n" --repeats 3 --no-save-z; done
done
for n in 25 75 150 300 600; do for v in 2 4; do abl "sweep_multi${v}_ws_b${n}" "$MUL/jaxipm_ablation_${v}_ws_ir0_b${n}_results.npz" --scenario multi --variants "$v" --mode ws --batch "$n" --repeats 3 --no-save-z; done; done

}

stage_madnlp() {
# ---- run E: MadNLP on the jaxipm pool ----
for spec in "quad_nav_circle madnlp_quad.jl 2000 madnlp_sector90_pool_tol1e-8_results.npz" "quad_track_avoid madnlp_track_avoid.jl 1000 madnlp_v1.0_pool_tol1e-8_results.npz" "quad_multi_swap madnlp_multi_swap.jl 75 madnlp_2_pool_tol1e-8_results.npz"; do
  set -- $spec; d=$1; f=$2; n=$3; first=$4
  if [ -f "tests/$d/logs/$first" ]; then log "skip madnlp $d (exists)"; continue; fi
  log ">>> madnlp $d (N_RUNS=$n, tol 1e-8, 5 reps)"
  CUDA_VISIBLE_DEVICES="$GPU" MADNLP_TOL=1e-8 MADNLP_N_RUNS="$n" MADNLP_REPEATS=5 MADNLP_OUT_SUFFIX="_pool_tol1e-8" "$JL" "tests/$d/$f" > "$LOGD/madnlp_${d}.log" 2>&1 \
    && grep -E "VARIANCE\] throughput|saved" "$LOGD/madnlp_${d}.log" | sed 's/^/    /' \
    || { echo "    FAIL -- see $LOGD/madnlp_${d}.log"; tail -5 "$LOGD/madnlp_${d}.log" | sed 's/^/    /'; }
done
}

for st in $STAGES; do
  log "===== stage $st (GPU $GPU) ====="
  "stage_$st" || log "stage $st returned $?"
done
"$PY" -m tests.rebuttal.analyze_L > "$LOGD/analyze_L_final_$(date +%m%d_%H%M).log" 2>&1
log "CHAIN COMPLETE"

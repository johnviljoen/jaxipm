#!/usr/bin/env bash
# Run D repeats: IPOPT P=64 physical, tol 1e-8, all seven scenarios, R repeats
# (John, 08-27: "repeats are nearly free ... the ± drops straight into the cells").
#   tests/rebuttal/d_repeats.sh <suffix-tag> [R]      -> files *_phys_c064_tol1e-08_<tag>_rep<r>_results.npz
set -u
cd "$(dirname "$0")/../.."
TAG="${1:-now}"; R="${2:-5}"
PY=/home/john/.conda/envs/jaxipm_latest/bin/python
LOGD=tests/rebuttal/logs_par_20260827; mkdir -p "$LOGD"
log() { echo "[$(date +%F' '%T)] $*"; }
log "D repeats tag=$TAG R=$R  loadavg=$(cut -d' ' -f1-3 /proc/loadavg)"
for r in $(seq 1 "$R"); do
  for m in quad_nav_circle quad_track_avoid quad_multi_swap; do
    CPU_SWEEP_SUFFIX="_tol1e-08_${TAG}_rep${r}" "$PY" -m tests.$m.quad_casadi_cpu_throughput --tol 1e-8 --modes phys --cores 64 --resume \
      > "$LOGD/d_rep_${TAG}_${m}_r${r}.log" 2>&1 || log "FAIL $m rep $r"
    grep -E 'throughput=' "$LOGD/d_rep_${TAG}_${m}_r${r}.log" | sed "s/^/    rep$r $m /" | cut -c1-150
  done
  log "rep $r done  loadavg=$(cut -d' ' -f1-3 /proc/loadavg)"
done
log "D REPEATS DONE ($TAG)"

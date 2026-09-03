#!/usr/bin/env bash
# IPOPT default config (one IPOPT/MUMPS process using the whole CPU), tol 1e-8, R repeats of
# every scenario at the paper pool sizes (N_RUNS_seq) -> casadi_<tag>_tol1e-08_rep<r>_results.npz
set -u
cd "$(dirname "$0")/../.."
R="${1:-5}"
PY=/home/john/.conda/envs/jaxipm_latest/bin/python
LOGD=tests/rebuttal/logs_par_20260827; mkdir -p "$LOGD"
log() { echo "[$(date +%F' '%T)] $*"; }
until grep -q 'IPOPT-DEFAULT DONE' "$LOGD/ipopt_default_chain.log" 2>/dev/null; do sleep 60; done
export IPOPT_TOL=1e-8 PYTHONPATH=.
for r in $(seq 1 "$R"); do
  for m in "quad_nav_circle ipopt_quad_nav" "quad_track_avoid ipopt_track_avoid" "quad_multi_swap ipopt_multi_swap"; do
    set -- $m
    log "IPOPT default rep $r $1  loadavg=$(cut -d' ' -f1-3 /proc/loadavg)"
    IPOPT_OUT_SUFFIX="_tol1e-08_rep${r}" "$PY" tests/$1/$2.py > "$LOGD/ipopt_default_rep${r}_$1.log" 2>&1 || log "  FAIL"
    grep -E 'saved' "$LOGD/ipopt_default_rep${r}_$1.log" | sed 's/^/    /' | cut -c1-120
  done
done
"$PY" -m tests.rebuttal.analyze_ipopt_default > "$LOGD/analyze_ipopt_default.log" 2>&1; log "analysis exit $?"
log "IPOPT-DEFAULT REPEATS DONE"

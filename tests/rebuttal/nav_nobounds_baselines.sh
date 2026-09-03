#!/usr/bin/env bash
# Nav baselines recomputed WITHOUT state/motor bounds (parity with jaxipm's nav
# model, whose quadcopter_nav returns +-inf bounds). tol 1e-8, same pool.
#   cpu    : IPOPT-default x5 (N_RUNS_seq=80, paper protocol) + full pool pass
#            (2000, U_all saved) ; IPOPT P=64 pinned x5 ("now" protocol)
#   madnlp : MadNLP nav 90/180 on GPU ${GPU:-3}, 2000 pool, one compile + 5 repeats
set -u
cd "$(dirname "$0")/../.."
PY="${PY:-/home/john/.conda/envs/jaxipm_release/bin/python}"
JL="${JL:-/home/john/.juliaup/bin/julia}"
LOGD=tests/rebuttal/logs_nobounds_20260829; mkdir -p "$LOGD"
log() { echo "[$(date +%F' '%T)] $*"; }
export NAV_NO_BOUNDS=1 IPOPT_TOL=1e-8 PYTHONPATH=.
case "${1:-cpu}" in
  cpu)
    for r in 1 2 3 4 5; do
      f=tests/quad_nav_circle/logs/casadi_sector90_nobounds_tol1e-08_rep${r}_results.npz
      if [ -f "$f" ]; then log "skip default rep$r"; else
        log "IPOPT default rep $r  load=$(cut -d' ' -f1-3 /proc/loadavg)"
        IPOPT_OUT_SUFFIX="_nobounds_tol1e-08_rep${r}" "$PY" tests/quad_nav_circle/ipopt_quad_nav.py > "$LOGD/ipopt_default_rep${r}.log" 2>&1 || log "  FAIL"
        grep -h "saved\|throughput" "$LOGD/ipopt_default_rep${r}.log" | cut -c1-140; fi
    done
    f=tests/quad_nav_circle/logs/casadi_sector90_nobounds_pool_tol1e-08_results.npz
    if [ -f "$f" ]; then log "skip pool pass"; else
      log "IPOPT default POOL pass (2000)  load=$(cut -d' ' -f1-3 /proc/loadavg)"
      IPOPT_N_RUNS=2000 IPOPT_OUT_SUFFIX="_nobounds_pool_tol1e-08" "$PY" tests/quad_nav_circle/ipopt_quad_nav.py > "$LOGD/ipopt_default_pool.log" 2>&1 || log "  FAIL"; fi
    for r in 1 2 3 4 5; do
      log "IPOPT P=64 rep $r  load=$(cut -d' ' -f1-3 /proc/loadavg)"
      CPU_SWEEP_SUFFIX="_nobounds_tol1e-08_now_rep${r}" "$PY" -m tests.quad_nav_circle.quad_casadi_cpu_throughput --tol 1e-8 --modes phys --cores 64 --resume > "$LOGD/d64_rep${r}.log" 2>&1 || log "  FAIL"
      grep -h "throughput=" "$LOGD/d64_rep${r}.log" | cut -c1-140
    done
    log "CPU BASELINES DONE" ;;
  madnlp)
    log "MadNLP nav no-bounds on GPU ${GPU:-3}  load=$(cut -d' ' -f1-3 /proc/loadavg)"
    CUDA_VISIBLE_DEVICES="${GPU:-3}" MADNLP_TOL=1e-8 MADNLP_N_RUNS=2000 MADNLP_REPEATS=5 MADNLP_NO_BOUNDS=1 MADNLP_OUT_SUFFIX="_nobounds_pool_tol1e-8" "$JL" tests/quad_nav_circle/madnlp_quad.jl > "$LOGD/madnlp_nav.log" 2>&1 || log "  FAIL"
    grep -hE "VARIANCE|saved|DISABLED" "$LOGD/madnlp_nav.log" | cut -c1-140
    log "MADNLP DONE" ;;
esac

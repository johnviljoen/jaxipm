#!/usr/bin/env bash
# Clean ILB-on sweep re-run on GPU 6 after the overnight queue (STATUS item 31): the
# 08-27 GPU-5 run shared the GPU with another user's jobs from 01:43. Contaminated files are
# moved aside (never deleted); nav N<=1000 (run before 01:43) is kept.
set -u
cd "$(dirname "$0")/../.."
LOGD=tests/rebuttal/logs_par_20260827
log() { echo "[$(date +%F' '%T)] $*"; }
until grep -q 'K-AND-QUALITY DONE' "$LOGD/k_quality_waiter.log" 2>/dev/null; do sleep 120; done
log "queue finished -- moving contaminated ILB-on sweep files aside"
for d in tests/quad_track_avoid/logs tests/quad_multi_swap/logs tests/quad_nav_circle/logs; do
  mkdir -p "$d/contaminated_gpu5_0827"
done
mv tests/quad_track_avoid/logs/jaxipm_ablation_v*_hr_ir0_b{125,250,500,1000,2000}_results.npz tests/quad_track_avoid/logs/contaminated_gpu5_0827/ 2>/dev/null
mv tests/quad_multi_swap/logs/jaxipm_ablation_{2,4}_hr_ir0_b{25,75,150,300,600,1200}_results.npz tests/quad_multi_swap/logs/contaminated_gpu5_0827/ 2>/dev/null
ls tests/quad_track_avoid/logs/contaminated_gpu5_0827 tests/quad_multi_swap/logs/contaminated_gpu5_0827 | sed 's/^/    /'
echo "co-tenants on GPU 6 at start:"; nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader | grep c0d08954 || echo "    none"
export SPINEAX_FACTOR_CACHE=64
bash tests/rebuttal/resume_chain.sh 6 sweep_hr > "$LOGD/gpu6_sweep_hr_rerun.log" 2>&1
log "sweep_hr rerun exit $?"
echo "co-tenants on GPU 6 at end:"; nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader | grep c0d08954 || echo "    none"
/home/john/.conda/envs/jaxipm_latest/bin/python -m tests.rebuttal.analyze_C > "$LOGD/analyze_C_rerun.log" 2>&1; log "analyze_C exit $?"
log "ILB-ON RERUN DONE"

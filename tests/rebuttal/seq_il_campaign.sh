#!/usr/bin/env bash
# Paper-table jaxipm campaign, env jaxipm_release, GPU 6, FRESH JIT per process
# (persistent compilation cache disabled). Per config and per mode
# (hr = IL with sequential pool, every index once; ws = SL, one pass by
# construction): 5 fresh-process timed runs (own JIT + one untimed warmup
# call + one timed call, non-DEBUG) then 1 DEBUG run for the quality columns. Idempotent:
# a launch is skipped when its npz already exists. GPU 6 only (John).
set -u
cd "$(dirname "$0")/../.."
PY="${PY:-/home/john/.conda/envs/jaxipm_release/bin/python}"
GPU="${GPU:-6}"
REPS="${REPS:-5}"
GPU_UUID=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' -v g="$GPU" '$1==g{print $2}')
LOGDIR=tests/rebuttal/logs; mkdir -p "$LOGDIR"
STAMP=$(date +%Y%m%d_%H%M)
MAIN="$LOGDIR/seq_il_campaign_${STAMP}_gpu${GPU}.log"
echo "campaign start $(date -Is) GPU=$GPU ($GPU_UUID) PY=$PY" | tee -a "$MAIN"

cotenants() {  # foreign processes on our GPU (anything not this campaign)
  nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader \
    | grep "$GPU_UUID" | grep -v "jaxipm_ablation" || true
}

launch() {  # scenario variant tag N_batch kind(rep_k|dbg) logs mode(hr|ws)
  local sc=$1 var=$2 tag=$3 nb=$4 kind=$5 logs=$6 mode=$7
  local suffix args out seqtag order
  # hr = IL (sequential pool, every index once); ws = SL (already one pass).
  # 'rel' in the suffix = jaxipm_release env + fresh JIT (no persistent cache).
  # POOL_ORDER=seq (every index once, '_seq' in the name) or paper (nav-style stride, wrap at the end)
  if [ "$mode" = hr ]; then order="--pool-order ${POOL_ORDER:-seq}"; seqtag=""; [ "${POOL_ORDER:-seq}" = seq ] && seqtag="_seq"; else order=""; seqtag=""; fi
  # SUFFIX tags the run set (rel = jaxipm_release + fresh JIT; relb = same, nav bounds fixed 2026-08-30)
  if [ "$kind" = dbg ]; then
    suffix="${SUFFIX:-rel}"; args="--debug --out-suffix $suffix"
    out="$logs/jaxipm_ablation_${tag}_${mode}_ir0_b${nb}${seqtag}_dbg_${suffix}_results.npz"
  else
    suffix="${SUFFIX:-rel}_fresh_${kind}"; args="--out-suffix $suffix --no-save-z"
    out="$logs/jaxipm_ablation_${tag}_${mode}_ir0_b${nb}${seqtag}_${suffix}_results.npz"
  fi
  if [ -f "$out" ]; then echo "skip (exists) $out" | tee -a "$MAIN"; return; fi
  local ct; ct=$(cotenants)
  echo "$(date -Is) LAUNCH $sc $var $mode $kind  cotenants_before=[${ct//$'\n'/;}] load=$(cut -d' ' -f2 /proc/loadavg)" | tee -a "$MAIN"
  local l="$LOGDIR/seq_il_${tag}_${mode}_${kind}_${STAMP}.log"
  $PY -m tests.rebuttal.jaxipm_ablation --scenario "$sc" --variants "$var" --mode "$mode" \
      $order --gpu "$GPU" --repeats 1 --no-persistent-cache $args > "$l" 2>&1
  local rc=$?
  ct=$(cotenants)
  echo "$(date -Is) DONE   $sc $var $mode $kind rc=$rc cotenants_after=[${ct//$'\n'/;}] $(grep -h '\[VARIANCE\].*rep 1/1\|SEQ CHECK\|% of pool' "$l" | tr '\n' ' ')" | tee -a "$MAIN"
}

# scenario variant tag N_batch logs_dir
CONFIGS=(
  "nav 90 sector90 500 tests/quad_nav_circle/logs"
  "nav 180 sector180 500 tests/quad_nav_circle/logs"
  "track 1.0 v1.0 250 tests/quad_track_avoid/logs"
  "track 1.5 v1.5 250 tests/quad_track_avoid/logs"
  "track 2.0 v2.0 250 tests/quad_track_avoid/logs"
  "multi 2 2 75 tests/quad_multi_swap/logs"
  "multi 4 4 75 tests/quad_multi_swap/logs"
)
for c in "${CONFIGS[@]}"; do
  set -- $c
  [ -n "${SCEN:-}" ] && [ "$1" != "$SCEN" ] && continue
  [ -n "${VAR:-}" ] && [ "$2" != "$VAR" ] && continue
  for mode in ${MODES:-hr ws}; do
    if [ "${REVERSE:-0}" = 1 ]; then   # helper worker on a second GPU: run the queue backwards
      launch "$1" "$2" "$3" "$4" dbg "$5" "$mode"
      for k in $(seq "$REPS" -1 1); do launch "$1" "$2" "$3" "$4" "rep$k" "$5" "$mode"; done
    else
      for k in $(seq 1 "$REPS"); do launch "$1" "$2" "$3" "$4" "rep$k" "$5" "$mode"; done
      [ "${NODBG:-0}" = 1 ] || launch "$1" "$2" "$3" "$4" dbg "$5" "$mode"
    fi
  done
done
echo "campaign end $(date -Is)" | tee -a "$MAIN"

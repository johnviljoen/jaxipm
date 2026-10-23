#!/usr/bin/env bash
# DAgger track-avoid campaign: N repeats x {il, madnlp, sl} experts, interleaved
# (il, madnlp, sl, il, madnlp, sl, ...). Repeat k uses --seed k for every expert
# (paired: identical starts, rollout seeds and network init across experts).
# Each run = dagger_track -> timing_track -> render_track into logs/runs/<expert>/run<k>/
# (metrics.json, per-round policies/datasets/rollouts, expert_heldout.npz,
# student_heldout.npz, timing.json, figures/{track_overlay.png, student_track.gif,
# render_summary.json}, logs, done.json). Idempotent: a run whose done.json
# reports a successful dagger stage is skipped.
# Usage (from anywhere):  tests/imitation_track/campaign.sh [GPU=6] [N_REPEATS=5] [extra dagger args...]
set -u
GPU="${1:-6}"; N="${2:-5}"; shift 2 2>/dev/null || shift $# 
export CUDA_VISIBLE_DEVICES="$GPU"
export XLA_PYTHON_CLIENT_PREALLOCATE=false      # dynamic allocation, never grab the card
PY="${PY:-/home/john/.conda/envs/jaxipm_release/bin/python}"
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO"
RUNS="tests/imitation_track/logs/runs"   # under logs/: gitignored; mkdir -p "$RUNS"
MAIN="$RUNS/campaign_gpu${GPU}.log"
echo "campaign start $(date -Is) GPU=$GPU repeats=$N PY=$PY extra=[$*]" | tee -a "$MAIN"
for k in $(seq 1 "$N"); do
  for ex in il madnlp sl; do
    out="$RUNS/$ex/run$k"
    if [ -f "$out/done.json" ] && grep -q '"dagger_rc": 0' "$out/done.json"; then
      echo "skip $out (done)" | tee -a "$MAIN"; continue
    fi
    mkdir -p "$out"
    echo "[$(date -Is)] START $ex run$k" | tee -a "$MAIN"
    t0=$(date +%s)
    "$PY" -m tests.imitation_track.dagger_track --expert "$ex" --seed "$k" --out "$out" "$@" > "$out/dagger.log" 2>&1; rc_d=$?
    t1=$(date +%s)
    "$PY" -m tests.imitation_track.timing_track --policy "$out/policy_final.eqx" --avg-vel 1.0 --out "$out/timing.json" > "$out/timing.log" 2>&1; rc_t=$?
    t2=$(date +%s)
    "$PY" -m tests.imitation_track.render_track --logs "$out" --figures "$out/figures" > "$out/render.log" 2>&1; rc_r=$?
    t3=$(date +%s)
    printf '{"expert": "%s", "run": %d, "seed": %d, "gpu": "%s", "dagger_rc": %d, "timing_rc": %d, "render_rc": %d, "dagger_wall_s": %d, "timing_wall_s": %d, "render_wall_s": %d, "finished": "%s"}\n' \
      "$ex" "$k" "$k" "$GPU" "$rc_d" "$rc_t" "$rc_r" $((t1-t0)) $((t2-t1)) $((t3-t2)) "$(date -Is)" > "$out/done.json"
    echo "[$(date -Is)] END $ex run$k rc dagger/timing/render=$rc_d/$rc_t/$rc_r  dagger $((t1-t0))s timing $((t2-t1))s render $((t3-t2))s" | tee -a "$MAIN"
  done
done
echo "campaign end $(date -Is)" | tee -a "$MAIN"

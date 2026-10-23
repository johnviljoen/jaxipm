#!/usr/bin/env bash
# DAgger imitation pipeline on track-avoid: train -> timing -> render.
# Usage (from anywhere):  tests/imitation_track/run.sh [GPU] [extra dagger_track args...]
set -u
GPU="${1:-0}"; shift || true
export CUDA_VISIBLE_DEVICES="$GPU"
export XLA_PYTHON_CLIENT_PREALLOCATE=false      # dynamic allocation, never grab the card
PY="${PY:-/home/john/.conda/envs/jaxipm_release/bin/python}"
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO"
LOGD="tests/imitation_track/logs"; mkdir -p "$LOGD"
echo "GPU=$GPU PY=$PY"
"$PY" -m tests.imitation_track.dagger_track "$@" 2>&1 | tee "$LOGD/dagger.log"
"$PY" -m tests.imitation_track.timing_track     2>&1 | tee "$LOGD/timing.log"
"$PY" -m tests.imitation_track.render_track     2>&1 | tee "$LOGD/render.log"

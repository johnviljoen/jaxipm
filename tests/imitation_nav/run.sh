#!/usr/bin/env bash
# DAgger imitation pipeline on the nav scenario: train -> timing -> render.
# Usage (from anywhere):  tests/imitation_nav/run.sh [GPU] [extra dagger_nav args...]
set -u
GPU="${1:-1}"; shift || true
export CUDA_VISIBLE_DEVICES="$GPU"
export XLA_PYTHON_CLIENT_PREALLOCATE=false      # dynamic allocation, never grab the card
PY="${PY:-/home/john/.conda/envs/jaxipm_release/bin/python}"
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO"
LOGD="tests/imitation_nav/logs"; mkdir -p "$LOGD"
echo "GPU=$GPU PY=$PY"
"$PY" -m tests.imitation_nav.dagger_nav "$@" 2>&1 | tee "$LOGD/dagger.log"
"$PY" -m tests.imitation_nav.timing_nav     2>&1 | tee "$LOGD/timing.log"
"$PY" -m tests.imitation_nav.render_nav     2>&1 | tee "$LOGD/render.log"

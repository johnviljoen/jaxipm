#!/usr/bin/env bash
# Varied-IC multi-swap campaign (rotation pool, 2026-08-31).
# usage: GPU=<n> SLOTS="2:hr:rep1 4:ws:rep3 2:hr:dbg ..." bash run_multi_rot.sh
# kind = repN (fresh-JIT timed) or dbg. Suffix rot300, max_solves 300, skip-if-exists.
set -u
cd "$(dirname "$0")/../.."
PY=/home/john/.conda/envs/jaxipm_release/bin/python
L=tests/rebuttal/logs; D=tests/quad_multi_swap/logs
for slot in $SLOTS; do
  IFS=: read -r v m k <<< "$slot"
  order=""; [ "$m" = hr ] && order="--pool-order paper"
  if [ "$k" = dbg ]; then
    out="$D/jaxipm_ablation_${v}_${m}_ir0_b75_dbg_rot300_results.npz"
    args="--debug --out-suffix rot300"
  else
    out="$D/jaxipm_ablation_${v}_${m}_ir0_b75_rot300_fresh_${k}_results.npz"
    args="--out-suffix rot300_fresh_${k} --no-save-z"
  fi
  [ -f "$out" ] && { echo "skip $out"; continue; }
  $PY -u -m tests.rebuttal.jaxipm_ablation --scenario multi --variants $v --mode $m $order \
    --gpu "$GPU" --repeats 1 --no-persistent-cache --max-solves 300 $args \
    > $L/rot300_${v}_${m}_${k}.log 2>&1
  echo "multi q$v $m $k rc=$? $(grep -oE 'throughput=[0-9.]+' $L/rot300_${v}_${m}_${k}.log | tail -1)"
done

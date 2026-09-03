#!/usr/bin/env bash
# Multi-swap 375-solve campaign half: usage GPU=<n> MODE=<hr|ws> bash relcp375_half.sh
# 5 fresh-JIT timed reps per variant (q2,q4) then DEBUG per variant. Skips existing npz.
set -u
cd "$(dirname "$0")/../.."
PY=/home/john/.conda/envs/jaxipm_release/bin/python
L=tests/rebuttal/logs; D=tests/quad_multi_swap/logs
order=""; [ "$MODE" = hr ] && order="--pool-order paper"
for r in 1 2 3 4 5; do
  for v in 2 4; do
    out="$D/jaxipm_ablation_${v}_${MODE}_ir0_b75_relcp375_fresh_rep${r}_results.npz"
    [ -f "$out" ] && { echo "skip $out"; continue; }
    $PY -u -m tests.rebuttal.jaxipm_ablation --scenario multi --variants $v --mode $MODE $order \
      --gpu "$GPU" --repeats 1 --no-persistent-cache --max-solves 375 \
      --out-suffix relcp375_fresh_rep$r --no-save-z > $L/relcp375_multi_${v}_${MODE}_rep${r}.log 2>&1
    echo "multi q$v $MODE rep$r rc=$?"
  done
done
for v in 2 4; do
  out="$D/jaxipm_ablation_${v}_${MODE}_ir0_b75_dbg_relcp375_results.npz"
  [ -f "$out" ] && { echo "skip $out"; continue; }
  $PY -u -m tests.rebuttal.jaxipm_ablation --scenario multi --variants $v --mode $MODE $order \
    --gpu "$GPU" --repeats 1 --no-persistent-cache --max-solves 375 \
    --debug --out-suffix relcp375 > $L/relcp375_multi_${v}_${MODE}_dbg.log 2>&1
  echo "multi q$v $MODE dbg rc=$?"
done

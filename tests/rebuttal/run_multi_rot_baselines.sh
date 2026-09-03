#!/usr/bin/env bash
# Varied-IC (v4, ±5° sectors) multi-swap baseline chain — run AFTER the jaxipm
# fresh runs (needs a quiet host for the CPU stages). Stages are serialized so
# no timed stage shares host CPU or GPU with another.
#   Stage 1: MadNLP GPU (75-pool × 5 in-process reps)      -> madnlp_<q>_rot_pool_tol1e-8
#   Stage 2: IPOPT-default ×5 reps (N_RUNS_seq=50) + pool  -> casadi_<q>_rot_tol1e-08_rep<r>,
#            quality pass (75 instances)                      casadi_<q>_rot_pool_tol1e-08
#   Stage 3: IPOPT-64 pinned ×5 reps (320 solves)          -> casadi_cpu_throughput_<q>_phys_c064_rot_tol1e-08_now_rep<r>
# usage: GPU=<n> bash run_multi_rot_baselines.sh
set -u
cd "$(dirname "$0")/../.."
PY=/home/john/.conda/envs/jaxipm_release/bin/python
L=tests/rebuttal/logs
log() { echo "[$(date +%F' '%T)] $*"; }

log "STAGE 1: MadNLP GPU $GPU (75-pool x 5 reps, tol 1e-8)"
( cd tests/quad_multi_swap && \
  CUDA_VISIBLE_DEVICES=$GPU MADNLP_TOL=1e-8 MADNLP_N_RUNS=75 MADNLP_REPEATS=5 \
  MADNLP_OUT_SUFFIX=_rot_pool_tol1e-8 julia madnlp_multi_swap.jl ) \
  > $L/rot_madnlp_pool.log 2>&1
log "  madnlp rc=$? $(grep -oE 'throughput mean=[0-9.]+ std=[0-9.]+' $L/rot_madnlp_pool.log | tail -2 | tr '\n' ' ')"

log "STAGE 2: IPOPT-default x5 reps + 75-pool quality pass (tol 1e-8)"
export PYTHONPATH=. IPOPT_TOL=1e-8
for r in 1 2 3 4 5; do
  IPOPT_OUT_SUFFIX="_rot_tol1e-08_rep${r}" $PY tests/quad_multi_swap/ipopt_multi_swap.py \
    > $L/rot_ipopt_default_rep${r}.log 2>&1
  log "  ipopt-default rep$r rc=$? $(grep -oE 'total_time=[0-9.]+s, success=[0-9]+/[0-9]+' $L/rot_ipopt_default_rep${r}.log | tr '\n' ' ')"
done
IPOPT_N_RUNS=75 IPOPT_OUT_SUFFIX="_rot_pool_tol1e-08" $PY tests/quad_multi_swap/ipopt_multi_swap.py \
  > $L/rot_ipopt_default_pool.log 2>&1
log "  ipopt-default pool rc=$? $(grep -oE 'success=[0-9]+/[0-9]+' $L/rot_ipopt_default_pool.log | tr '\n' ' ')"

log "STAGE 3: IPOPT-64 pinned x5 reps (tol 1e-8)"
for r in 1 2 3 4 5; do
  CPU_SWEEP_SUFFIX="_rot_tol1e-08_now_rep${r}" $PY -m tests.quad_multi_swap.quad_casadi_cpu_throughput \
    --modes phys --cores 64 --tol 1e-8 > $L/rot_ipopt64_rep${r}.log 2>&1
  log "  ipopt-64 rep$r rc=$? $(grep -oE 'throughput=[0-9.]+ solves/s' $L/rot_ipopt64_rep${r}.log | tr '\n' ' ')"
done
log "BASELINE CHAIN DONE"

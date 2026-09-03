#!/usr/bin/env bash
# Sequential GPU run queue for the rebuttal ablations (E2 / E5 / E6 / E11).
# One GPU, one process at a time; CPU sweeps must NOT run concurrently.
#
#   tests/rebuttal/run_queue.sh [GPU] [STAGE...]
#     GPU    : CUDA device index (default 1)
#     STAGE  : any of  census  ws  timing  timing_ws  timing_hr  n1  i2  sweep_ws  sweep_hr  ir  d_cpu  madnlp
#              (default: census ws timing ir, in order)
#
# Every run writes jaxipm_ablation_<tag>_<mode>_ir<K>_b<N>[_dbg]_results.npz
# into the scenario's logs/ dir plus a per-run log under tests/rebuttal/logs/.
set -u
cd "$(dirname "$0")/../.."
GPU="${1:-1}"; shift || true
STAGES="${*:-census ws timing ir}"
PY="${PY:-/home/john/.conda/envs/jaxipm_latest/bin/python}"
export SPINEAX_FACTOR_CACHE="${SPINEAX_FACTOR_CACHE:-64}"   # LRU capacity; default 8 evicts+refactorizes live tokens inside timed loops
LOGD=tests/rebuttal/logs; mkdir -p "$LOGD"
SCEN="nav track multi"

run() {  # run <logname> <args...>
  local name="$1"; shift
  echo "[$(date +%F' '%T)] >>> $name : $*"
  if "$PY" -m tests.rebuttal.jaxipm_ablation "$@" --gpu "$GPU" > "$LOGD/$name.log" 2>&1; then
    grep -E "^\[VARIANCE\].*: n=|collected=|iters mean|TERM CENSUS|^saved" "$LOGD/$name.log" | sed 's/^/    /'
  else
    echo "    FAIL (exit $?) -- see $LOGD/$name.log"; tail -5 "$LOGD/$name.log" | sed 's/^/    /'
  fi
}

for st in $STAGES; do
  case "$st" in
    census)   # E5 + E11 + E2(c) quality: HR, DEBUG on, ir=0
      for s in $SCEN; do run "census_${s}_hr_dbg" --scenario "$s" --mode hr --debug; done ;;
    ws)       # E2(b) fusion-only quality: WS, DEBUG on, ir=0
      for s in $SCEN; do run "census_${s}_ws_dbg" --scenario "$s" --mode ws --debug; done ;;
    timing)   # runs A (hr) and B (ws): DEBUG off, 5 timed repeats each
      for s in $SCEN; do
        run "timing_${s}_hr" --scenario "$s" --mode hr --repeats 5 --no-save-z
        run "timing_${s}_ws" --scenario "$s" --mode ws --repeats 5 --no-save-z
      done ;;
    timing_ws)  # run B only (ILB off), DEBUG off, 5 timed repeats
      for s in $SCEN; do run "timing_${s}_ws" --scenario "$s" --mode ws --repeats 5 --no-save-z; done ;;
    timing_ws_fresh)  # run B with one process per repeat (A's fresh-JIT protocol), 5 launches
      for s in $SCEN; do for r in 1 2 3 4 5; do
        run "timing_${s}_ws_fresh_rep${r}" --scenario "$s" --mode ws --repeats 1 --no-save-z --out-suffix "fresh_rep${r}"
      done; done ;;
    timing_hr)  # run A via the harness (same code path as B/C), DEBUG off, 5 repeats
      for s in $SCEN; do run "timing_${s}_hr" --scenario "$s" --mode hr --repeats 5 --no-save-z; done ;;
    n1)         # run C bottom rung: N = 1, both ILB settings, DEBUG off, 3 repeats
      for s in $SCEN; do
        run "n1_${s}_hr" --scenario "$s" --mode hr --batch 1 --repeats 3 --no-save-z
        run "n1_${s}_ws" --scenario "$s" --mode ws --batch 1 --repeats 3 --no-save-z
      done ;;
    i2)       # run I2: small N, ir 0 vs 100, DEBUG on, 48 solves per config
      for s in $SCEN; do for n in 1 8 32; do for k in 0 100; do
        run "i2_${s}_ws_b${n}_ir${k}_dbg" --scenario "$s" --mode ws --pool-subsample --debug --batch "$n" --max-solves 48 --ir-nsteps "$k" --out-suffix i2
      done; done; done ;;
    sweep_ws) # run C: ILB-off batch sweep, DEBUG off, 3 repeats, peak memory logged
      for n in 125 250 500 1000 2000; do for s in nav track; do
        run "sweep_${s}_ws_b${n}" --scenario "$s" --mode ws --batch "$n" --repeats 3 --no-save-z
      done; done
      for n in 25 75 150 300 600; do run "sweep_multi_ws_b${n}" --scenario multi --mode ws --batch "$n" --repeats 3 --no-save-z; done ;;
    sweep_hr) # run C: ILB-on batch sweep through the same harness (peak memory), 3 repeats
      for n in 125 250 500 1000 2000 4000; do for s in nav track; do
        run "sweep_${s}_hr_b${n}" --scenario "$s" --mode hr --batch "$n" --repeats 3 --no-save-z
      done; done
      for n in 25 75 150 300 600 1200; do run "sweep_multi_hr_b${n}" --scenario multi --mode hr --batch "$n" --repeats 3 --no-save-z; done ;;
    d_cpu)    # run D rerun at tol 1e-8 (CPU only -- never concurrent with GPU stages)
      for s in quad_nav_circle quad_track_avoid quad_multi_swap; do
        echo "[$(date +%F' '%T)] >>> d_cpu $s"
        "$PY" -m "tests.$s.quad_casadi_cpu_throughput" --tol 1e-8 --resume > "$LOGD/d_cpu_${s}_tol1e-8.log" 2>&1 \
          && grep -E "wall=|summary" "$LOGD/d_cpu_${s}_tol1e-8.log" | tail -16 | sed 's/^/    /' \
          || { echo "    FAIL -- see $LOGD/d_cpu_${s}_tol1e-8.log"; tail -5 "$LOGD/d_cpu_${s}_tol1e-8.log" | sed 's/^/    /'; }
      done ;;
    madnlp)   # run E: MadNLP, jaxipm pool, tol 1e-8, 5 repeats (GPU)
      JL="${JL:-/home/john/.juliaup/bin/julia}"
      for spec in "quad_nav_circle madnlp_quad.jl 2000" "quad_track_avoid madnlp_track_avoid.jl 1000" "quad_multi_swap madnlp_multi_swap.jl 75"; do
        set -- $spec; d=$1; f=$2; n=$3
        echo "[$(date +%F' '%T)] >>> madnlp $d (N_RUNS=$n, tol 1e-8, 5 reps)"
        CUDA_VISIBLE_DEVICES="$GPU" MADNLP_TOL=1e-8 MADNLP_N_RUNS="$n" MADNLP_REPEATS=5 MADNLP_OUT_SUFFIX="_pool_tol1e-8" \
          "$JL" "tests/$d/$f" > "$LOGD/madnlp_${d}.log" 2>&1 \
          && grep -E "VARIANCE|saved" "$LOGD/madnlp_${d}.log" | tail -8 | sed 's/^/    /' \
          || { echo "    FAIL -- see $LOGD/madnlp_${d}.log"; tail -8 "$LOGD/madnlp_${d}.log" | sed 's/^/    /'; }
      done ;;
    ir)       # refinement sweep in batch: HR, DEBUG on, ir in {2,10,100}
      for k in 2 10 100; do
        for s in $SCEN; do run "ir${k}_${s}_hr_dbg" --scenario "$s" --mode hr --debug --ir-nsteps "$k"; done
      done ;;
    *) echo "unknown stage $st";;
  esac
done
echo "[$(date +%F' '%T)] queue done"

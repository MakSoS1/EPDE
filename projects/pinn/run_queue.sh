#!/bin/bash
# The whole queue after the no-bias change, in order: recalculated Stages 0/1/1b
# (GPU, ~23 h), the Stage-2 grid (GPU, ~46 h), then the basis cells (CPU, ~2 h,
# only once no GPU work is timed). Each part logs to results/<part>.log.
cd "$(dirname "$0")"
# 0. patch neutrality, before anything is spent: under the OLD candidate rule the
#    patched code reproduces the Stage-0 B rows and the Stage-1 S1 rows bitwise
{
  for s in ac duffing burgers; do
    python gate.py drive --system $s --arms B-ctl --seeds 0-2 --tag stage2_legacy --legacy-intercept
  done
  for s in ac duffing; do
    python gate.py drive --system $s --arms S1 --seeds 0 --mode trace --tag stage2_legacy --legacy-intercept
  done
  python gate.py neutrality --tag stage2_legacy
} > results/neutrality.log 2>&1 || { echo "patch NOT neutral: see results/neutrality.log"; exit 1; }
bash run_recalc_gpu.sh > results/recalc_gpu.log 2>&1
grep -q RECALC_GPU_DONE results/recalc_gpu.log || { echo "recalc stopped early"; exit 1; }
bash run_stage2_gpu.sh > results/stage2.log 2>&1
grep -q STAGE2_ALL_DONE results/stage2.log || { echo "stage 2 stopped early"; exit 1; }
bash run_recalc_basis.sh > results/recalc_basis.log 2>&1
echo QUEUE_DONE

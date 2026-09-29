#!/bin/bash
# The frozen basis cells of Stages 1 / 1c under the no-bias rule (CPU, run only
# when no GPU batch is timed). Read with  python analyze.py --tags r_basis
cd "$(dirname "$0")"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=""
set -x
ARMS=BL,BL-n1,BL-n2,BL-n3,BL-n4,BL-r5,BL-data,BL-noprec,BL-t4,BL-c3,BL-sinv-live,BL-sinv-data,BL-pred,BL-sinv-live-f,BL-sinv-data-f
for s in duffing burgers ac; do
  python gate.py drive --system $s --arms $ARMS --seeds 0 --tag r_basis
done
echo RECALC_BASIS_DONE

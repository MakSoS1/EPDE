#!/bin/bash
# Recalculation of Stages 0 / 1 / 1b under the no-bias rule (PI, Sep 29: "Make
# sure that there is no bias in pde. It is only there if equation has it").
# Every row of the old tags was refit with a FORCED free coefficient; these
# rerun the same network arms with a free term only where the form writes one
# (row 'intercept_rule' = 'from_form'). One code state, tags r_*. Read with
#   python analyze.py --tags r_baseline,r_start,r_stage1,r_stage1b,r_end
# Shared data fits depend on the data only, so the existing ones are reused.
# Needs: results/ moved here, the Stage-2 adapter patches applied, suite green.
cd "$(dirname "$0")"
set -x
for s in ac duffing burgers; do
  python gate.py prefit --system $s --arm S9 --seeds 0-2
  python gate.py prefit --system $s --arm S10 --seeds 0-2
done
# Stage 0: the shipped recipe and its duplicate
for s in ac duffing burgers; do
  python gate.py drive --system $s --arms B,B-ctl --seeds 0-2 --tag r_baseline
done
for s in ac duffing burgers; do
  python gate.py drive --system $s --arms B-ctl --seeds 0-2 --tag r_start
done
# Stage 1: the lr-free network arms (trace), null checks on seed 0
for s in ac duffing burgers; do
  python gate.py drive --system $s --arms B-4k,S0 --seeds 0-2 --mode trace --tag r_stage1
  python gate.py drive --system $s --arms S0-nolr,S3-null --seeds 0 --mode trace --tag r_stage1
  python gate.py drive --system $s --arms S1,S2,S3,S4,S6,S7,S8,S9,B9 --seeds 0-2 --mode trace --tag r_stage1
done
# Stage 1b/1c: the other lr-free routes
for s in ac duffing burgers; do
  python gate.py drive --system $s --arms S1-cap --seeds 0-2 --tag r_stage1b
  python gate.py drive --system $s --arms S10,S7f,S8f,P1,P2,D1,D2,S0-32 --seeds 0-2 --mode trace --tag r_stage1b
done
for s in ac duffing burgers; do
  python gate.py drive --system $s --arms S5-32 --seeds 0-2 --tag r_stage1b
done
for s in ac duffing burgers; do
  python gate.py drive --system $s --arms B-ctl --seeds 0-2 --tag r_end
done
echo RECALC_GPU_DONE

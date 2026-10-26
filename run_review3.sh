#!/bin/bash
# Third-review analyses (2026-10-18). Methods are run exactly as frozen.
#  A  prior-generator mismatch on the fourth fresh test set (930000+)
#  B  signed vs clipped valuation of CauProbe-2 on second-test plants (analysis)
#  C  detector-threshold sensitivity on the second-test broker-fault plants
set -u
cd /mnt/d/Experiments/CauSLM-Edge
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mkdir -p logs/v4
MA=oracle,greedy,cauprobe-noprobe,cauprobe,cauprobe-v2,thompson
(
  for sc in mismatch mismatch_lpwan; do
    R=1; [ "$sc" = mismatch ] && R=2
    taskset -c 0-39 /root/wpy experiments/cp_e1_commission.py --outdir test4 --tag "$sc" \
        --scenario "$sc" --plants test4 --runs "$R" --methods "$MA" --days 14 \
        --workers 40 --resume >> "logs/v4/test4_$sc.log" 2>&1
    echo "$(date) finished test4 $sc" >> logs/v4/grid.log
  done
) &
(
  for sc in nominal lpwan; do
    R=1; [ "$sc" = nominal ] && R=2
    taskset -c 40-55 /root/wpy experiments/cp_e1_commission.py --outdir ablate2 --tag "$sc" \
        --scenario "$sc" --plants test2 --runs "$R" --methods "cauprobe-v2@fl_signed=1" \
        --days 14 --workers 16 --resume >> "logs/v4/ablate2_$sc.log" 2>&1
    echo "$(date) finished signed $sc" >> logs/v4/grid.log
  done
  taskset -c 40-55 /root/wpy experiments/cp_safety.py --sens >> logs/v4/safety_sens.log 2>&1
  echo "$(date) finished safety sens" >> logs/v4/grid.log
) &
wait
echo "$(date) REVIEW3_DONE" >> logs/v4/grid.log

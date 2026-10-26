#!/bin/bash
# Second iteration: evaluation on the SECOND fresh test set (seeds 910000+,
# out-of-library 960000+), created after CauProbe-2 was frozen. CauProbe-1
# (frozen), CauProbe-2 and the rivals are run once, paired on the same plants
# and noise streams. Every stage appends to its own JSONL and can be resumed.
set -u
cd /mnt/d/Experiments/CauSLM-Edge
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
W=${WORKERS:-56}
M=oracle,rule,greedy,cauprobe-noprobe,cauprobe,cauprobe-v2,cauprobe-v2-noask,thompson,ids,identify

run () {  # tag scenario plants runs
  taskset -c 0-55 /root/wpy experiments/cp_e1_commission.py --outdir test2 --tag "$1" \
      --scenario "$2" --plants "$3" --runs "$4" --methods "$M" --days 14 \
      --workers "$W" --resume >> "logs/v2/test2_$1.log" 2>&1
  echo "$(date) finished $1" >> logs/v2/test2_grid.log
}

run nominal nominal test2 2
for sc in noisy lpwan noisy_lpwan poor_docs shift; do
  run "$sc" "$sc" test2 1
done
run ool nominal ool2 2
taskset -c 0-55 /root/wpy experiments/cp_e7_price.py --set test2 --workers "$W" \
    >> logs/v2/test2_price.log 2>&1
echo "$(date) finished price" >> logs/v2/test2_grid.log
taskset -c 0-55 /root/wpy experiments/cp_voi_ask2.py >> logs/v2/test2_voi.log 2>&1
echo "$(date) TEST2_GRID_DONE" >> logs/v2/test2_grid.log

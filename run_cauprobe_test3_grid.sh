#!/bin/bash
# Third fresh test set (seeds 920000+), created after the fleet-calibrated
# expert was built from earlier sites' interventional labels and frozen
# (results/cauprobe/test3/FROZEN.txt). CauProbe-2 and expert-plus-greedy
# control are run with the expert as calibrated on training-vocabulary notes
# and with the fleet-calibrated expert, against greedy control on the same
# belief, paired on plants and noise streams. Each stage can be resumed.
set -u
cd /mnt/d/Experiments/CauSLM-Edge
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
W=${WORKERS:-40}
CORES=${CORES:-0-39}
M=oracle,greedy,cauprobe-noprobe,cauprobe-noprobe-fleet,cauprobe-v2,cauprobe-v2-fleet

run () {  # tag scenario runs
  taskset -c "$CORES" /root/wpy experiments/cp_e1_commission.py --outdir test3 --tag "$1" \
      --scenario "$2" --plants test3 --runs "$3" --methods "$M" --days 14 \
      --workers "$W" --resume >> "logs/v3/test3_$1.log" 2>&1
  echo "$(date) finished $1" >> logs/v3/test3_grid.log
}

run nominal nominal 2
run poor_docs poor_docs 1
run lpwan lpwan 1
run noisy noisy 1
WORKERS=$W taskset -c "$CORES" /root/wpy experiments/cp_voi_ask3.py >> logs/v3/test3_voi.log 2>&1
echo "$(date) TEST3_GRID_DONE" >> logs/v3/test3_grid.log

#!/bin/bash
# Development check of the fleet-calibrated expert on the 16 dev plants
# (outside the fleet and every test set): does anything break, and what does
# it do? Not used for any decision about the fleet expert, which is fixed by
# construction (cp_build_fleet_expert.py).
set -u
cd /mnt/d/Experiments/CauSLM-Edge
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
M=greedy,cauprobe-noprobe,cauprobe-noprobe-fleet,cauprobe-v2,cauprobe-v2-fleet
for sc in nominal lpwan poor_docs; do
  taskset -c 0-55 /root/wpy experiments/cp_e1_commission.py --outdir fleet_dev \
      --tag "$sc" --scenario "$sc" --plants dev --runs 1 --methods "$M" \
      --days 14 --workers 56 --resume >> logs/v3/fleet_dev.log 2>&1
done
echo "$(date) FLEET_DEV_DONE" >> logs/v3/fleet_dev.log

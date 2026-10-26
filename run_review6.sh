#!/bin/bash
# Sixth-review analyses (2026-10-19), beside the powered probe audit (56
# workers): at most 4 workers here (machine limit 60).
#  0  frozen-code check after adding the analysis-only fs_audit flag
#  A  posterior-dependent vs frozen admissible sets in the valuation
set -u
cd /mnt/d/Experiments/CauSLM-Edge
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mkdir -p logs/v7
/root/wpy experiments/cp_verify_frozen.py --replay > logs/v7/verify_frozen.log 2>&1
echo "$(date) finished verify ($?)" >> logs/v7/grid.log
/root/wpy experiments/cp_fs_audit.py --workers 4 > logs/v7/fs_audit.log 2>&1
echo "$(date) finished fs_audit ($?)" >> logs/v7/grid.log
#  B  BOPTEST with the second iteration (library iteration v3, as reported),
#     the same ten weeks and seeds as the reported runs
CP_BOPTEST_PRIOR=v3 LD_LIBRARY_PATH=/root/miniforge3/envs/bop/lib /root/wbop experiments/cp_e6_boptest.py \
    --days 7 --starts 155,165,175,185,195,205,215,225,235,245 --methods cauprobe-v2,cauprobe-v2+expert \
    --workers 4 --tag v3ctrl_v2 > logs/v7/boptest_v2.log 2>&1
echo "$(date) finished boptest v2 ($?)" >> logs/v7/grid.log
#  C  reproducibility check: one stored BOPTEST week of the first iteration
CP_BOPTEST_PRIOR=v3 LD_LIBRARY_PATH=/root/miniforge3/envs/bop/lib /root/wbop experiments/cp_e6_boptest.py \
    --days 7 --starts 155 --methods cauprobe,greedy --workers 2 --tag v3ctrl_check > logs/v7/boptest_check.log 2>&1
echo "$(date) finished boptest check ($?)" >> logs/v7/grid.log

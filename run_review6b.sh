#!/bin/bash
# Rerun of the admissible-set audit with per-valuation records (4 workers).
set -u
cd /mnt/d/Experiments/CauSLM-Edge
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
/root/wpy experiments/cp_fs_audit.py --workers 4 > logs/v7/fs_audit2.log 2>&1
echo "$(date) finished fs_audit2 ($?)" >> logs/v7/grid.log

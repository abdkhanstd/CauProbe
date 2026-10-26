#!/bin/bash
# Powered replicated probe audit (fifth review round, 2026-10-18).
# Every decision of the extended audit (cp_audit2.py: 157 decisions of
# CauProbe-2 on 32 second-test plants, Wi-Fi and LPWAN costs) is replayed 36
# times with independent future noise (cp_audit3.py's reseed scheme, so
# replicates 0-5 reproduce the 48 x 6 run). R = 36 is fixed in advance by the
# power calculation (experiments/cp_audit3_power.py): reliability of a
# decision's mean >= 0.7 at the ICC of the 48 x 6 run. Jobs hold 3 replicates
# and are queued block by block, so a balanced design exists at every multiple
# of 3; if the run is cut, the analysis uses the largest balanced R (>= 24).
# Restart after an interruption with the same command (--resume);
# WORKERS=32 bash run_audit3_powered.sh uses fewer workers (never more than 56).
set -u
cd /mnt/d/Experiments/CauSLM-Edge
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mkdir -p logs/v5
/root/wpy experiments/cp_audit3.py --n 999 --reps 36 --chunk 3 --workers ${WORKERS:-56} --resume \
    --out results/cauprobe/test2/probe_audit3_powered.jsonl >> logs/v5/audit3_powered.log 2>&1
echo "$(date) AUDIT3_POWERED_DONE" >> logs/v5/audit3_powered.log

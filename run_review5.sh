#!/bin/bash
# Fifth-review analyses (2026-10-18), run beside the powered probe audit
# (56 workers), so at most 4 workers here: the machine's limit is 60.
#  0  frozen-code check after adding the analysis-only fl_tau_scale flag
#  A  embedded profile of the decision procedure (development plants)
#  B  sensitivity of CauProbe-2 to mis-estimated free-learning horizons:
#     every estimated tau_s scaled by 0.5 or 2 (second-test plants, analysis)
set -u
cd /mnt/d/Experiments/CauSLM-Edge
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mkdir -p logs/v6
/root/wpy experiments/cp_verify_frozen.py --replay > logs/v6/verify_frozen.log 2>&1
echo "$(date) finished verify ($?)" >> logs/v6/grid.log
/root/wpy experiments/cp_profile.py --workers 4 > logs/v6/profile.log 2>&1
echo "$(date) finished profile" >> logs/v6/grid.log
for sc in nominal lpwan; do
  /root/wpy experiments/cp_e1_commission.py --outdir tausens2 --tag "$sc" \
      --scenario "$sc" --plants test2 --runs 1 \
      --methods "cauprobe-v2@fl_tau_scale=0.5,cauprobe-v2@fl_tau_scale=2.0" \
      --days 14 --workers 4 --resume >> "logs/v6/tausens_$sc.log" 2>&1
  echo "$(date) finished tausens $sc" >> logs/v6/grid.log
done
echo "$(date) REVIEW5_DONE" >> logs/v6/grid.log
# A again, with the step index of the slowest steps recorded
/root/wpy experiments/cp_profile.py --workers 4 > logs/v6/profile2.log 2>&1
echo "$(date) finished profile2" >> logs/v6/grid.log

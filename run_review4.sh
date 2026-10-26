#!/bin/bash
# Fourth-review analyses (2026-10-18).
#  A  approximate dual control by Bayes-adaptive rollouts ("dual-rollout"),
#     settings fixed a priori to CauProbe-2's own rollouts (3 days, 8 truths
#     per structure, same candidates and probe lengths), run on the second
#     fresh test set and paired with the stored runs there
#  B  extended paired probe audit with decision-state features and the
#     rollout comparator's direct value of each audited probe
set -u
cd /mnt/d/Experiments/CauSLM-Edge
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mkdir -p logs/v5
(
  for sc in nominal lpwan noisy noisy_lpwan shift; do
    R=1; [ "$sc" = nominal ] && R=2
    M=dual-rollout
    if [ "$sc" = nominal ] || [ "$sc" = lpwan ]; then M=dual-rollout,dual-rollout+ask; fi
    taskset -c 0-39 /root/wpy experiments/cp_e1_commission.py --outdir dual2 --tag "$sc" \
        --scenario "$sc" --plants test2 --runs "$R" --methods "$M" --days 14 \
        --workers 40 --resume >> "logs/v5/dual2_$sc.log" 2>&1
    echo "$(date) finished dual2 $sc" >> logs/v5/grid.log
  done
) &
(
  taskset -c 40-63 /root/wpy experiments/cp_audit2.py --workers 24 >> logs/v5/audit2.log 2>&1
  echo "$(date) finished audit2" >> logs/v5/grid.log
) &
wait
echo "$(date) REVIEW4_DONE" >> logs/v5/grid.log

# C  replicated paired audit (run afterwards, from the decisions audited in B):
#    48 decisions x 6 independent future-noise replicates
# taskset -c 0-47 /root/wpy experiments/cp_audit3.py --workers 48 >> logs/v5/audit3.log 2>&1

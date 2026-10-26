#!/bin/bash
# Final evaluation on 48 fresh test plants (seeds 900000+), method frozen.
# Every stage appends to its own JSONL and can be resumed.
set -u
cd /mnt/d/Experiments/CauSLM-Edge
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
W=${WORKERS:-44}
CORE=oracle,rule,linucb,advice-plan,greedy,thompson,ids,identify,cauprobe-noask,cauprobe-noprobe,cauprobe
ALL=$CORE,fallback,nominal,log-naive,thompson-unsafe,cauprobe-nosafety,cauprobe+logs-naive,cauprobe-particle,cauprobe-nodur,cauprobe-nokappa,cauprobe-h1,cauprobe-h7,cauprobe-expert:tfidf-lr,cauprobe-expert:slm-0.5b-lora-uncal,cauprobe-expert:zeroshot-1.5b

run () {  # tag scenario plants runs methods
  /root/wpy experiments/cp_e1_commission.py --tag "$1" --scenario "$2" \
      --plants "$3" --runs "$4" --methods "$5" --days 14 --workers "$W" --resume \
      >> "logs/cp_test_$1.log" 2>&1
  echo "$(date) finished $1" >> logs/cp_test_grid.log
}

run test_main nominal test 2 "$ALL"
for sc in noisy lpwan noisy_lpwan poor_docs shift; do
  run "test_$sc" "$sc" test 1 "$CORE"
done
run test_ool nominal ool 2 "$CORE"
/root/wpy experiments/cp_e7_price.py --workers "$W" >> logs/cp_test_price.log 2>&1
echo "$(date) finished price" >> logs/cp_test_grid.log
W=16 /root/wpy experiments/cp_probe_timing.py >> logs/cp_test_timing.log 2>&1
echo "$(date) TEST_GRID_DONE" >> logs/cp_test_grid.log

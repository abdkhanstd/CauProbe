#!/bin/bash
# Paired counterfactual audit of probe decisions (cp_audit_probes.py), started
# once the safety analysis has released its cores.
cd /mnt/d/Experiments/CauSLM-Edge
while pgrep -f 'cp_safet[y]' > /dev/null; do sleep 30; done
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
taskset -c 40-63 /root/wpy experiments/cp_audit_probes.py --workers 24 >> logs/v3/audit.log 2>&1
echo "$(date) AUDIT_DONE" >> logs/v3/audit.log

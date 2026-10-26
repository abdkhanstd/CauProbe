# Frozen code states

`test2/` and `test3/` hold the exact sources whose checksums were recorded in
`results/cauprobe/test2/FROZEN.txt` and `results/cauprobe/test3/FROZEN.txt`
before those fresh test sets were created (paths inside each folder mirror the
repository).

Code was added to the controller after these freezes: decision records for the
probe audits, the signed-gain ablation (`fl_signed`), the horizon-scaling
sensitivity (`fl_tau_scale`), the frozen-admissible-set audit (`fs_audit`), the approximate dual-control comparator
(`objective="rollout"`), and the plant generator's widened ranges for the
fourth test set (`widen`). Each is switched off unless selected, so the current
files no longer match the recorded checksums. The frozen versions were
recovered by undoing, newest first, every recorded edit made after each
freeze. Every recovered file matches its recorded checksum.

`test2_to_current.diff` and `test3_to_current.diff` are the unified diffs
from each frozen state to the current code.

To check the checksums and replay a stored second-test run of CauProbe-1 and
CauProbe-2 with both the frozen and the current code, run:

    python experiments/cp_verify_frozen.py --replay

All four reruns reproduce the stored returns bit for bit (-2041.784002891706
and -2041.6754328037687).

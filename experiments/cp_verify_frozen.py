"""Verify the frozen code states behind the fresh test sets.

frozen/test2 and frozen/test3 hold the exact sources whose checksums were
recorded in results/cauprobe/test2/FROZEN.txt and test3/FROZEN.txt before those
test sets were created (recovered by undoing, newest first, every later edit).
This script
  1. checks every snapshot file against the recorded md5;
  2. writes unified diffs from each snapshot to the current sources
     (frozen/test2_to_current.diff, frozen/test3_to_current.diff): the later
     additions are an analysis-only flag, decision records, the rollout
     comparator and the plant generator's `widen` option, all off by default;
  3. with --replay, reruns one stored test-2 run of CauProbe-1 and CauProbe-2
     with the frozen test-2 code and with the current code and compares the
     returns with the stored ones bit for bit.
"""
from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / "frozen"
SEED, RECORD = 9100030, ROOT / "results" / "cauprobe" / "test2" / "e1_nominal.jsonl"


def recorded(path):
    out = {}
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) == 2 and len(parts[0]) == 32:
            out[parts[1].lstrip("*")] = parts[0]
    return out


def check(name):
    rec = recorded(ROOT / "results" / "cauprobe" / name / "FROZEN.txt")
    ok = True
    diff = []
    for rel, h in rec.items():
        snap = FROZEN / name / rel
        got = hashlib.md5(snap.read_bytes()).hexdigest()
        print(f"  {name} {rel:45s} {'OK' if got == h else 'MISMATCH'}")
        ok &= got == h
        cur = ROOT / rel
        if rel.endswith(".py") and cur.read_bytes() != snap.read_bytes():
            diff += difflib.unified_diff(snap.read_text().splitlines(True), cur.read_text().splitlines(True),
                                         f"frozen/{name}/{rel}", rel)
    (FROZEN / f"{name}_to_current.diff").write_text("".join(diff))
    return ok


REPLAY = r'''
import sys, json
root, frozen, method = sys.argv[1], sys.argv[2], sys.argv[3]
sys.path.insert(0, root + "/experiments"); sys.path.insert(0, root)
import causlm, causlm.probe
if frozen != "-":
    causlm.probe.__path__.insert(0, frozen + "/causlm/probe")
    sys.path.insert(0, frozen + "/experiments")
import cp_e1_commission as E
import causlm.probe.agent as A
rows = [json.loads(l) for l in open(root + "/results/cauprobe/test2/e1_nominal.jsonl") if l.strip()]
r0 = [r for r in rows if r["method"] == method and r["seed"] == %d][0]
out = E.job((method, r0["seed"] // 10, r0["combo"], None, r0["seed"] %% 10, 14,
             {"energy_j": 40.0, "latency_ms": 6000.0, "price_per_kj": 0.4}, "nominal"))
print(json.dumps({"agent": A.__file__, "stored": r0["return"], "rerun": out.get("return")}))
''' % SEED


def replay():
    procs = {}
    for label, fz in (("frozen test-2 code", str(FROZEN / "test2")), ("current code", "-")):
        for m in ("cauprobe", "cauprobe-v2"):
            procs[(label, m)] = subprocess.Popen([sys.executable, "-c", REPLAY, str(ROOT), fz, m],
                                                 stdout=subprocess.PIPE, text=True)
    ok = True
    for (label, m), p in procs.items():
        r = json.loads(p.communicate()[0].strip().splitlines()[-1])
        same = r["rerun"] == r["stored"]
        ok &= same
        print(f"  {label:20s} {m:12s} stored {r['stored']!r} rerun {r['rerun']!r} "
              f"{'BIT-IDENTICAL' if same else 'DIFFERENT'}  ({Path(r['agent']).relative_to(ROOT)})")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replay", action="store_true")
    a = ap.parse_args()
    ok = check("test2") & check("test3")
    if a.replay:
        ok &= replay()
    print("ALL CHECKS PASSED" if ok else "CHECK FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

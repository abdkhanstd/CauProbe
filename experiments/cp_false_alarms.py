"""False alarms of the misfit detector on the in-library test runs.

The detector's threshold (cusum_h = 21.6) was chosen on 64 seven-day
in-library development runs (cp_calib_detector.py), none of which it alarms
on. The test plants played no part in that choice, so their in-library runs
give an independent estimate of the false-alarm rate: every alarm there is
false. Counted over the 14-day runs of every controller that runs the
detector (peak CUSUM > 0) on the first three test sets, in the scenarios whose
plants stay inside the library (nominal, noisy, LPWAN, noisy + LPWAN, poor
documentation); structural change, prior mismatch and out-of-library plants
are left out, since an alarm there can be correct.

    python experiments/cp_false_alarms.py   # writes results/cauprobe/false_alarms.json
"""
from __future__ import annotations

import json

import numpy as np

from cp_common import CP_RESULTS

IN_LIBRARY = ("nominal", "noisy", "lpwan", "noisy_lpwan", "poor_docs")
NOISY = ("noisy", "noisy_lpwan")
FILES = {"test1": "e1_test_{}.jsonl", "test2": "test2/e1_{}.jsonl", "test3": "test3/e1_{}.jsonl"}
CAL_TARGET = 0.025          # weekly false-alarm target of the calibration


def rows():
    out = []
    for st, pat in FILES.items():
        for sc in IN_LIBRARY:
            p = CP_RESULTS / pat.format("main" if (st == "test1" and sc == "nominal") else sc)
            if not p.exists():
                continue
            for line in p.read_text().splitlines():
                if not line.strip():
                    continue
                r = json.loads(line)
                if "error" in r or r.get("ool") is not None or r.get("cusum_max", 0) <= 0:
                    continue
                r["_set"], r["_scen"] = st, sc
                out.append(r)
    return out


def rate(rs):
    k = sum(r["misspec_alarms"] > 0 for r in rs)
    return {"runs": len(rs), "with_alarm": k, "rate": k / len(rs) if rs else None}


def main():
    rs = rows()
    alarmed = [r for r in rs if r["misspec_alarms"] > 0]
    out = {
        "all": rate(rs),
        "normal_telemetry": rate([r for r in rs if r["_scen"] not in NOISY]),
        "noisy_telemetry": rate([r for r in rs if r["_scen"] in NOISY]),
        "cauprobe-v2": rate([r for r in rs if r["method"] == "cauprobe-v2"]),
        "by_scenario": {sc: rate([r for r in rs if r["_scen"] == sc]) for sc in IN_LIBRARY},
        "by_set": {st: rate([r for r in rs if r["_set"] == st]) for st in FILES},
        "alarms_per_alarmed_run": {"median": float(np.median([r["misspec_alarms"] for r in alarmed])),
                                   "max": int(max(r["misspec_alarms"] for r in alarmed))},
        "fallback_steps_per_alarmed_run_median": float(np.median([r["fallback_steps"] for r in alarmed])),
        # the weekly target expressed per 14-day run
        "target_per_14_days": 1 - (1 - CAL_TARGET) ** 2,
        # one-sided 95% upper bound on a rate after 0 alarms in 64 runs
        "dev_0_of_64_upper95": 1 - 0.05 ** (1 / 64),
    }
    (CP_RESULTS / "false_alarms.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("normal_telemetry", "noisy_telemetry", "cauprobe-v2",
                                          "alarms_per_alarmed_run",
                                          "fallback_steps_per_alarmed_run_median",
                                          "target_per_14_days", "dev_0_of_64_upper95")}, indent=1))


if __name__ == "__main__":
    main()

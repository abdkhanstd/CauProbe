"""Calibrate the thermal out-of-library detector on in-library dev plants,
then check it in shadow mode on the 64 evaluation plants.

Dev plants (seeds 830_000+) are used for nothing else. The threshold is the
smallest h with at most 2.5% false alarms over a 14-day deployment, rounded
up. Shadow mode computes the statistic but never acts, so the evaluation-plant
check shows whether including the detector could have changed any reported
result: if it never crosses h there, it could not.
"""
from __future__ import annotations

import json
import os
from multiprocessing import get_context

import numpy as np

from cp_common import CP_RESULTS, EVAL_PLANTS
from causlm.probe.agent import CauProbeAgent
from causlm.probe.baselines import greedy_cfg
from causlm.probe.plants import ALL_COMBOS, sample_plant
from causlm.probe.run import run_commissioning


def job(a):
    seed, combo, days = a
    spec = sample_plant(seed, combo)
    ag = CauProbeAgent(greedy_cfg(thermal_detect=True, thermal_shadow=True),
                       seed=seed, name="greedy")
    out = run_commissioning(spec, ag, days=days, seed=seed * 10)
    return {"seed": seed, "th_cusum_max": out["th_cusum_max"]}


def main():
    dev = [(830_000 + i, ALL_COMBOS[i % 16], 14) for i in range(48)]
    ev = [(s, c, 14) for s, c in EVAL_PLANTS]
    with get_context("fork").Pool(int(os.environ.get("W", "16"))) as pool:
        r_dev = pool.map(job, dev)
        r_ev = pool.map(job, ev)
    d = np.array([r["th_cusum_max"] for r in r_dev])
    e = np.array([r["th_cusum_max"] for r in r_ev])
    h = None
    for x in np.sort(np.concatenate([d, [1e9]])):
        if (d > x).mean() <= 0.025:
            h = float(np.ceil(x + 1.0))
            break
    out = {"threshold": h, "dev_far": float((d > h).mean()),
           "eval_alarms": int((e > h).sum()), "dev_max": float(d.max()),
           "eval_max": float(e.max()), "dev": d.tolist(), "eval": e.tolist()}
    (CP_RESULTS / "thermal_detector_calibration.json").write_text(json.dumps(out, indent=1))
    print({k: out[k] for k in ("threshold", "dev_far", "eval_alarms", "dev_max",
                               "eval_max")})


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main()

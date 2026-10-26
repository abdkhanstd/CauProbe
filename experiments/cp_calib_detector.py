"""Calibrate the out-of-library detector on dev plants (seeds 800_000+).

The detector's CUSUM is run with no alarm (h = inf) and its peak is recorded
on (i) in-library plants under nominal and noisy telemetry -- every alarm
there would be false -- and (ii) out-of-library plants, where an alarm is the
correct outcome. The reported threshold is the smallest h whose false-alarm
rate over a 7-day deployment is at most 2.5%; its detection rate on the
out-of-library dev plants is reported with it. None of these plants is used
anywhere else.
"""
from __future__ import annotations

import json
import os
from multiprocessing import get_context

import numpy as np

from cp_common import CP_RESULTS
from causlm.probe.agent import CauProbeAgent
from causlm.probe.baselines import greedy_cfg
from causlm.probe.plants import ALL_COMBOS, sample_plant
from causlm.probe.run import run_commissioning

NOISY = {"sigma": (0.35, 0.55), "burst_p": 0.10, "burst": (2.0, 4.0)}


def job(a):
    seed, combo, ool, scen = a
    spec = sample_plant(seed, combo, ool=ool, scenario=NOISY if scen == "noisy" else None)
    ag = CauProbeAgent(greedy_cfg(cusum_h=float("inf")), seed=seed, name="greedy")
    out = run_commissioning(spec, ag, days=7, seed=seed * 10)
    return {"seed": seed, "ool": ool, "scen": scen, "cusum_max": out["cusum_max"]}


def main():
    jobs = []
    for i in range(32):
        c = ALL_COMBOS[i % 16]
        jobs.append((800_000 + i, c, None, "nominal"))
        jobs.append((800_100 + i, c, None, "noisy"))
    for i in range(8):
        jobs.append((800_200 + i, ALL_COMBOS[(3 * i) % 16], "tsch", "nominal"))
        jobs.append((800_300 + i, ALL_COMBOS[(5 * i + 1) % 16], "congested_wan",
                     "nominal"))
        jobs.append((800_400 + i, ALL_COMBOS[(7 * i + 2) % 16], "broker_fault",
                     "nominal"))
    with get_context("fork").Pool(40) as pool:
        res = pool.map(job, jobs)
    inl = np.array([r["cusum_max"] for r in res if r["ool"] is None])
    ool = np.array([r["cusum_max"] for r in res if r["ool"] is not None])
    hs = np.sort(np.unique(np.concatenate([inl, ool, [1e9]])))
    best = None
    for h in hs:
        far = float((inl > h).mean())
        if far <= 0.025:
            best = h
            break
    det = float((ool > best).mean())
    out = {"threshold": float(best) + 1.0, "false_alarm_rate": float((inl > best).mean()),
           "detection_rate": det, "in_library_cusum": inl.tolist(),
           "ool_cusum": ool.tolist(), "rows": res}
    (CP_RESULTS / "detector_calibration.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("threshold", "false_alarm_rate",
                                          "detection_rate")}))
    print("in-library peak CUSUM quantiles", np.percentile(inl, [50, 90, 97.5, 100]).round(1))
    print("out-of-library peak CUSUM quantiles", np.percentile(ool, [0, 25, 50, 100]).round(1))
    for kind in ("tsch", "congested_wan", "broker_fault"):
        v = np.array([r["cusum_max"] for r in res if r["ool"] == kind])
        print(f"  {kind:14s} detection {float((v > best).mean()):.2f} peak median {np.median(v):.1f}")


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main()

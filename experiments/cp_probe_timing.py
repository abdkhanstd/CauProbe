"""When does CauProbe buy information? Per-day probes, asks and EVPI.

Runs CauProbe with step recording on 16 evaluation plants (one per mechanism
combination, nominal scenario, 14 days) and writes per-day probe counts, the
day of the expert invocation, the night step tests, and the trace of the
self-calibration ratio.
"""
from __future__ import annotations

import json
import os
from multiprocessing import get_context

import numpy as np

import cp_e1_commission as E
from cp_common import CP_RESULTS, TEST_PLANTS
from causlm.probe.plants import sample_plant
from causlm.probe.run import AskCost, run_commissioning


def job(a):
    seed, combo = a
    spec = sample_plant(seed, tuple(combo))
    ag, z = E.make_agent("cauprobe", spec, f"eval:{seed}", seed * 10)
    out = run_commissioning(spec, ag, days=14, seed=seed * 10, expert_out=z,
                            ask=AskCost(), record_steps=True)
    st = out.pop("_steps")
    probe = np.array([s["probe"] for s in st]).reshape(14, -1).sum(1)
    return {"plant": spec.name, "probes_per_day": probe.tolist(),
            "asks": out["asks"], "th_probes": out["th_probes"],
            "kappa_trace": [list(map(float, k)) for k in ag.kappa_trace],
            "final_kappa": float(ag.kappa)}


def main():
    plants = sorted(TEST_PLANTS, key=lambda p: (p[0] - 900_000) // 16)[:16]
    with get_context("fork").Pool(int(os.environ.get("W", "16"))) as pool:
        res = pool.map(job, plants)
    (CP_RESULTS / "probe_timing.json").write_text(json.dumps(res, indent=1))
    P = np.array([r["probes_per_day"] for r in res])
    print("mean probes per day:", P.mean(0).round(1).tolist())
    print("share of probes in first 2 days: %.2f" % (P[:, :2].sum() / max(P.sum(), 1)))


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main()

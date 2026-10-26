"""Predicted versus realised value of reading the commissioning notes.

For the 32 price-sweep test plants, record the value of asking that CauProbe
predicts before its first action (the quantity it compares with the price of a
call) and set it against the realised value of the reading: ask-always minus
greedy control on the same plant and noise stream over the same 7 days
(results/cauprobe/e7_price.jsonl, Wi-Fi costs, lowest price). Writes
results/cauprobe/voi_ask.json.
"""
from __future__ import annotations

import json
from collections import defaultdict

import numpy as np

import cp_e1_commission as E
from cp_common import CP_RESULTS, TEST_PLANTS
from causlm.probe import mech
from causlm.probe.agent import CauProbeAgent, ProbeConfig
from causlm.probe.plants import sample_plant
from causlm.probe.run import AskCost, run_commissioning


def main():
    mech.set_switch_costs(0.3, 0.05)
    plants = sorted(TEST_PLANTS, key=lambda p: (p[0] - 900_000) // 16)[:32]
    ex = E.load_expert("slm-0.5b-lora")
    pred = {}
    for seed, combo in plants:
        spec = sample_plant(seed, tuple(combo))
        ag = CauProbeAgent(ProbeConfig(), seed=seed * 10, expert=ex, name="cauprobe")
        seen = []
        f = ag._voi_ask

        def wrapped(f=f, seen=seen):
            v = f()
            seen.append(v)
            return v

        ag._voi_ask = wrapped
        z = E.expert_outputs()[f"eval:{seed}"]["slm-0.5b-lora"]
        run_commissioning(spec, ag, days=1, seed=seed * 10, expert_out=z,
                          ask=AskCost(energy_j=25.0, latency_ms=6000.0, price_per_kj=0.4))
        pred[seed] = seen[0] if seen else None
    rows = [json.loads(l) for l in (CP_RESULTS / "e7_price.jsonl").read_text().splitlines()]
    base = {r["plant"] + str(r["seed"]): r["return"] for r in rows
            if r["method"] == "greedy" and r["switch"] == "wifi"}
    real = {}
    for r in rows:
        if r["method"] == "ask-always" and r["switch"] == "wifi" and r["price"] == 0.01:
            real[r["seed"] // 10] = r["return"] - base[r["plant"] + str(r["seed"])] + 0.01
    s = sorted(set(pred) & set(real))
    p = np.array([pred[k] for k in s])
    q = np.array([real[k] for k in s])
    out = {"n": len(s), "pred_mean": float(p.mean()), "pred_median": float(np.median(p)),
           "pred_min": float(p.min()), "real_mean": float(q.mean()),
           "real_median": float(np.median(q)), "ratio_of_means": float(p.mean() / q.mean()),
           "corr": float(np.corrcoef(p, q)[0, 1])}
    (CP_RESULTS / "voi_ask.json").write_text(json.dumps(out, indent=1))
    print(out)


if __name__ == "__main__":
    main()

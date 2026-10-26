"""Predicted versus realised value of reading the notes, with and without
fleet calibration of the expert (third test set).

For 32 plants of the third test set (two per mechanism combination), under
Wi-Fi and LPWAN reconfiguration costs: the value of reading the notes that
CauProbe-2 predicts before its first action, once with the expert calibrated
on training-vocabulary notes and once with the fleet-calibrated expert
(cp_build_fleet_expert.py), set against the realised value of the reading
read through the same expert -- ask-always (read once, never probe) minus
greedy control on the same plant and noise stream over 7 days, price added
back. Same design as cp_voi_ask2.py. Writes results/cauprobe/test3/voi_ask.json
and the 7-day runs to results/cauprobe/test3/e7_ask.jsonl.
"""
from __future__ import annotations

import json
import os
from multiprocessing import get_context

import numpy as np

import cp_e1_commission as E
from cp_common import CP_RESULTS, TEST3_PLANTS
from causlm.probe import mech
from causlm.probe.agent import CauProbeAgent, ProbeConfig
from causlm.probe.baselines import greedy_cfg
from causlm.probe.plants import GatewayPlant, sample_plant
from causlm.probe.run import AskCost, run_commissioning

OUT = CP_RESULTS / "test3"
SWITCH = {"wifi": (0.3, 0.05), "lpwan": (1.5, 0.2)}
EX = "slm-0.5b-lora"
PRICE = 0.01


def ask_cost():
    return AskCost(energy_j=PRICE / 0.4 * 1000.0, latency_ms=6000.0, price_per_kj=0.4)


def run7(a):
    method, seed, combo, sw = a
    mech.set_switch_costs(*SWITCH[sw])
    spec = sample_plant(seed, tuple(combo))
    z = E.expert_outputs()[f"eval:{seed}"][EX]
    s = seed * 10
    if method == "greedy":
        ag, z = CauProbeAgent(greedy_cfg(), seed=s, name="greedy"), None
    else:
        ex = E.load_expert(EX + ("-fleet" if method.endswith("-fleet") else ""))
        ag = CauProbeAgent(ProbeConfig(use_probe=False, force_ask=True), seed=s,
                           expert=ex, name=method)
    out = run_commissioning(spec, ag, days=7, seed=s, expert_out=z, ask=ask_cost())
    out.update({"method": method, "switch": sw, "price": PRICE})
    return out


def predicted(a):
    seed, combo, sw = a
    mech.set_switch_costs(*SWITCH[sw])
    spec = sample_plant(seed, tuple(combo))
    res = {}
    for tag, suffix in (("orig", ""), ("fleet", "-fleet")):
        ag, _ = E.make_agent("cauprobe-v2" + suffix, spec, f"eval:{seed}", seed * 10)
        plant = GatewayPlant(spec, seed=seed * 10)
        ag.cfg.ask_price = ask_cost().price
        ag.offer_text()
        ag.set_forecast(plant.forecast_day)
        ag._plan_day(None)               # what act() does first at midnight
        res[tag] = float(ag._voi_ask())
        if tag == "orig":
            p = ag._ask_parts
            res["v1"] = float(p["net_v1"] + p["th_v1"])
    return seed, sw, res


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    plants = sorted(TEST3_PLANTS, key=lambda p: (p[0] - 920_000) // 16)[:32]
    jobs = [(m, s, c, sw) for sw in SWITCH for s, c in plants
            for m in ("greedy", "ask-always", "ask-always-fleet")]
    W = int(os.environ.get("WORKERS", "40"))
    ctx = get_context("fork")
    with ctx.Pool(W, maxtasksperchild=4) as pool:
        rows = pool.map(run7, jobs)
        preds = pool.map(predicted, [(s, c, sw) for sw in SWITCH for s, c in plants])
    with open(OUT / "e7_ask.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    out = {}
    for sw in SWITCH:
        base = {r["seed"]: r["return"] for r in rows if r["method"] == "greedy" and r["switch"] == sw}
        real = {tag: {r["seed"] // 10: r["return"] - base[r["seed"]] + PRICE for r in rows
                      if r["method"] == m and r["switch"] == sw}
                for tag, m in (("orig", "ask-always"), ("fleet", "ask-always-fleet"))}
        pred = {s: p for s, w, p in preds if w == sw}
        S = sorted(pred)
        res = {"n": len(S)}
        for tag in ("orig", "fleet"):
            q = np.array([real[tag][k] for k in S])
            p = np.array([pred[k][tag] for k in S])
            res[tag] = {"real_mean": float(q.mean()), "real_median": float(np.median(q)),
                        "pred_mean": float(p.mean()), "pred_median": float(np.median(p)),
                        "ratio_of_means": float(p.mean() / q.mean()) if q.mean() else None,
                        "corr": float(np.corrcoef(p, q)[0, 1])}
        pv1 = np.array([pred[k]["v1"] for k in S])
        res["v1_pred_mean"] = float(pv1.mean())
        res["per_plant"] = {str(k): {**pred[k], "real_orig": real["orig"][k],
                                     "real_fleet": real["fleet"][k]} for k in S}
        out[sw] = res
        print(sw, {k: v for k, v in res.items() if k != "per_plant"}, flush=True)
    (OUT / "voi_ask.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main()

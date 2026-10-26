"""Predicted versus realised value of reading the notes, both iterations.

For the 32 price-sweep plants of the second test set, the value of asking each
iteration predicts before its first action (what it compares with the price of
a call) -- the first iteration's and the second's are computed on the same
belief, by the second iteration's agent, which records both -- set against the
realised value of the reading: ask-always minus greedy control on the same
plant and noise stream over the same 7 days (results/cauprobe/test2/
e7_price.jsonl, lowest price, price added back). Writes
results/cauprobe/test2/voi_ask.json. Same design as cp_voi_ask.py.
"""
from __future__ import annotations

import json

import numpy as np

import cp_e1_commission as E
from cp_common import CP_RESULTS, TEST2_PLANTS
from causlm.probe import mech
from causlm.probe.plants import GatewayPlant, sample_plant
from causlm.probe.run import AskCost

OUT = CP_RESULTS / "test2"


def predicted(seed, combo, sw):
    """Both iterations' predicted value of the first reading on one plant."""
    mech.set_switch_costs(*sw)
    spec = sample_plant(seed, tuple(combo))
    ag, z = E.make_agent("cauprobe-v2", spec, f"eval:{seed}", seed * 10)
    plant = GatewayPlant(spec, seed=seed * 10)
    ag.cfg.ask_price = AskCost(energy_j=25.0).price
    ag.offer_text()
    ag.set_forecast(plant.forecast_day)
    ag._plan_day(None)              # what act() does first at midnight
    v2 = ag._voi_ask()
    p = ag._ask_parts
    return {"v1": p["net_v1"] + p["th_v1"], "v2": float(v2),
            "v2_net": p["net"], "v2_th": float(p["th"])}


def main():
    plants = sorted(TEST2_PLANTS, key=lambda p: (p[0] - 910_000) // 16)[:32]
    rows = [json.loads(l) for l in (OUT / "e7_price.jsonl").read_text().splitlines()]
    out = {}
    for sw_name, sw in (("wifi", (0.3, 0.05)), ("lpwan", (1.5, 0.2))):
        pred = {s: predicted(s, c, sw) for s, c in plants}
        base = {r["seed"]: r["return"] for r in rows
                if r["method"] == "greedy" and r["switch"] == sw_name}
        lo = min(r["price"] for r in rows if r["method"] == "ask-always")
        real = {r["seed"] // 10: r["return"] - base[r["seed"]] + lo
                for r in rows if r["method"] == "ask-always"
                and r["switch"] == sw_name and r["price"] == lo}
        s = sorted(set(pred) & set(real))
        q = np.array([real[k] for k in s])
        res = {"n": len(s), "real_mean": float(q.mean()),
               "real_median": float(np.median(q))}
        for it in ("v1", "v2"):
            p = np.array([pred[k][it] for k in s])
            res[it] = {"pred_mean": float(p.mean()), "pred_median": float(np.median(p)),
                       "pred_min": float(p.min()), "pred_max": float(p.max()),
                       "ratio_of_means": float(p.mean() / q.mean()) if q.mean() else None,
                       "corr": float(np.corrcoef(p, q)[0, 1])}
        res["per_plant"] = {str(k): {**pred[k], "real": real[k]} for k in s}
        out[sw_name] = res
        print(sw_name, {k: v for k, v in res.items() if k != "per_plant"})
    # the summary the paper quotes is the Wi-Fi one (as in cp_voi_ask.py)
    out.update({"real_mean": out["wifi"]["real_mean"]})
    (OUT / "voi_ask.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()

"""Predicted versus realised value of reading the notes on a common horizon.

cp_voi_ask2.py / cp_voi_ask3.py set the value of a reading that each iteration
predicts before its first action -- valued over the controller's horizon,
min(h_days = 30, durability) days -- against the value realised over 7 days.
Here both are put on the same horizon H: the predictions are recomputed with
the horizon capped at H (cfg.h_days = H; this caps the first iteration's
durability horizon and the second iteration's free-learning horizons alike,
and changes nothing else), for H = 7, 14 and 30 (30 reproduces the stored
values), and the realised value is measured over the same H days:
  - 7 days: ask-always minus greedy control, 7-day price-sweep runs
    (test2/e7_price.jsonl at the lowest price, test3/e7_ask.jsonl);
  - 7 and 14 days: expert-plus-greedy control (reads once, never probes) minus
    greedy control in the stored 14-day grids, from the per-day returns
    (test2|test3/e1_nominal.jsonl run 0 for Wi-Fi, e1_lpwan.jsonl for LPWAN).
For the third set also the fleet-calibrated expert's prediction (realised:
ask-always-fleet, cauprobe-noprobe-fleet). Same 32 plants per set as the stored audits. Analysis only. Writes
results/cauprobe/voi_horizon.json.
"""
from __future__ import annotations

import json
import os
from multiprocessing import get_context

import numpy as np

import cp_e1_commission as E
from cp_common import CP_RESULTS, TEST2_PLANTS, TEST3_PLANTS
from causlm.probe import mech
from causlm.probe.plants import GatewayPlant, sample_plant
from causlm.probe.run import AskCost

SWITCH = {"wifi": (0.3, 0.05), "lpwan": (1.5, 0.2)}
HORIZONS = (7, 14, 30)
SETS = {"test2": (TEST2_PLANTS, 910_000, AskCost(energy_j=25.0).price),
        "test3": (TEST3_PLANTS, 920_000, 0.01)}


def predicted(a):
    """Both iterations' predicted value of the first reading, horizon capped at H."""
    tset, seed, combo, sw, H = a
    mech.set_switch_costs(*SWITCH[sw])
    spec = sample_plant(seed, tuple(combo))
    ag, _ = E.make_agent("cauprobe-v2", spec, f"eval:{seed}", seed * 10)
    ag.cfg.h_days = float(H)
    plant = GatewayPlant(spec, seed=seed * 10)
    ag.cfg.ask_price = SETS[tset][2]
    ag.offer_text()
    ag.set_forecast(plant.forecast_day)
    ag._plan_day(None)              # what act() does first at midnight
    v2 = float(ag._voi_ask())
    p = ag._ask_parts
    out = {"v1": float(p["net_v1"] + p["th_v1"]), "v2": v2}
    if tset == "test3":         # the fleet-calibrated expert (cp_voi_ask3.py)
        af, _ = E.make_agent("cauprobe-v2-fleet", spec, f"eval:{seed}", seed * 10)
        af.cfg.h_days = float(H)
        af.cfg.ask_price = SETS[tset][2]
        af.offer_text()
        af.set_forecast(GatewayPlant(spec, seed=seed * 10).forecast_day)
        af._plan_day(None)
        out["v2fleet"] = float(af._voi_ask())
    return tset, seed, sw, H, out


def realised_grid(tset, sw, plants, method="cauprobe-noprobe"):
    """Expert-plus-greedy minus greedy, run 0, cumulative over days 1..14."""
    f = CP_RESULTS / tset / ("e1_nominal.jsonl" if sw == "wifi" else "e1_lpwan.jsonl")
    rows = [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
    by = {(r["method"], r["seed"]): r for r in rows if r["run"] == 0}
    out = {}
    for s in plants:
        a, b = by.get((method, s * 10)), by.get(("greedy", s * 10))
        if a and b:
            out[s] = np.cumsum(np.array(a["return_per_day"]) - np.array(b["return_per_day"]))
    return out


def realised_7(tset, sw, method="ask-always"):
    if tset == "test2":
        rows = [json.loads(l) for l in (CP_RESULTS / "test2" / "e7_price.jsonl").read_text().splitlines()]
        base = {r["seed"]: r["return"] for r in rows if r["method"] == "greedy" and r["switch"] == sw}
        lo = min(r["price"] for r in rows if r["method"] == "ask-always")
        return {r["seed"] // 10: r["return"] - base[r["seed"]] + lo for r in rows
                if r["method"] == "ask-always" and r["switch"] == sw and r["price"] == lo}
    rows = [json.loads(l) for l in (CP_RESULTS / "test3" / "e7_ask.jsonl").read_text().splitlines()]
    base = {r["seed"]: r["return"] for r in rows if r["method"] == "greedy" and r["switch"] == sw}
    return {r["seed"] // 10: r["return"] - base[r["seed"]] + r["price"] for r in rows
            if r["method"] == method and r["switch"] == sw}


def main():
    plants = {t: sorted(P, key=lambda p: (p[0] - off) // 16)[:32] for t, (P, off, _) in SETS.items()}
    jobs = [(t, s, c, sw, H) for t in SETS for s, c in plants[t] for sw in SWITCH for H in HORIZONS]
    W = min(int(os.environ.get("WORKERS", "32")), 56)
    with get_context("fork").Pool(W, maxtasksperchild=4) as pool:
        res = pool.map(predicted, jobs)
    pred = {}
    for t, s, sw, H, v in res:
        pred.setdefault((t, sw), {}).setdefault(s, {})[H] = v
    out = {}
    for t in SETS:
        stored = json.loads((CP_RESULTS / t / "voi_ask.json").read_text())
        for sw in SWITCH:
            S = sorted(pred[(t, sw)])
            g = realised_grid(t, sw, S)
            r7 = realised_7(t, sw)
            cell = {"n": len(S)}
            for it in ("v1", "v2"):
                for H in HORIZONS:
                    cell[f"pred_{it}_H{H}"] = float(np.mean([pred[(t, sw)][s][H][it] for s in S]))
            # reproduction of the stored 30-day predictions
            st = stored[sw]["per_plant"]
            k2 = "v2" if t == "test2" else "orig"
            cell["max_abs_diff_vs_stored_H30"] = float(max(
                max(abs(pred[(t, sw)][s][30]["v1"] - st[str(s)]["v1"]),
                    abs(pred[(t, sw)][s][30]["v2"] - st[str(s)][k2])) for s in S))
            Sg = [s for s in S if s in g]
            cell["n_grid"] = len(Sg)
            cell["real_e7"] = float(np.mean([r7[s] for s in S]))
            cell["real_grid_7d"] = float(np.mean([g[s][6] for s in Sg]))
            cell["real_grid_14d"] = float(np.mean([g[s][13] for s in Sg]))
            cell["real_grid_days8_14"] = float(np.mean([g[s][13] - g[s][6] for s in Sg]))
            cell["real_grid_by_day"] = [float(np.mean([g[s][d] for s in Sg])) for d in range(14)]
            for it in ("v1", "v2"):
                cell[f"ratio_{it}_7d"] = cell[f"pred_{it}_H7"] / cell["real_e7"]
                cell[f"ratio_{it}_14d"] = cell[f"pred_{it}_H14"] / cell["real_grid_14d"]
                cell[f"ratio_{it}_H30_vs_7d_old"] = cell[f"pred_{it}_H30"] / cell["real_e7"]
            if t == "test3":
                gf, rf = realised_grid(t, sw, S, "cauprobe-noprobe-fleet"), realised_7(t, sw, "ask-always-fleet")
                for H in HORIZONS:
                    cell[f"pred_v2fleet_H{H}"] = float(np.mean([pred[(t, sw)][s][H]["v2fleet"] for s in S]))
                cell["max_abs_diff_vs_stored_fleet_H30"] = float(max(
                    abs(pred[(t, sw)][s][30]["v2fleet"] - st[str(s)]["fleet"]) for s in S))
                cell["real_fleet_e7"] = float(np.mean([rf[s] for s in S]))
                cell["real_fleet_grid_14d"] = float(np.mean([gf[s][13] for s in S if s in gf]))
                cell["ratio_v2fleet_7d"] = cell["pred_v2fleet_H7"] / cell["real_fleet_e7"]
                cell["ratio_v2fleet_14d"] = cell["pred_v2fleet_H14"] / cell["real_fleet_grid_14d"]
            out[f"{t}|{sw}"] = cell
            print(t, sw, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in cell.items()
                          if k != "real_grid_by_day"}, flush=True)
    (CP_RESULTS / "voi_horizon.json").write_text(json.dumps(out, indent=1))


def predicted_test1(a):
    """The first iteration's prediction on the first test set, as in
    cp_voi_ask.py (CauProbe-1, one-day run, first valuation), horizon capped."""
    seed, combo, H = a
    from causlm.probe.agent import CauProbeAgent, ProbeConfig
    from causlm.probe.run import run_commissioning
    mech.set_switch_costs(0.3, 0.05)
    spec = sample_plant(seed, tuple(combo))
    ag = CauProbeAgent(ProbeConfig(h_days=float(H)), seed=seed * 10,
                       expert=E.load_expert("slm-0.5b-lora"), name="cauprobe")
    seen, f = [], ag._voi_ask

    def wrapped(f=f, seen=seen):
        v = f()
        seen.append(v)
        return v

    ag._voi_ask = wrapped
    z = E.expert_outputs()[f"eval:{seed}"]["slm-0.5b-lora"]
    run_commissioning(spec, ag, days=1, seed=seed * 10, expert_out=z,
                      ask=AskCost(energy_j=25.0, latency_ms=6000.0, price_per_kj=0.4))
    return seed, H, float(seen[0])


def main_test1():
    from cp_common import TEST_PLANTS
    plants = sorted(TEST_PLANTS, key=lambda p: (p[0] - 900_000) // 16)[:32]
    W = min(int(os.environ.get("WORKERS", "32")), 56)
    with get_context("fork").Pool(W, maxtasksperchild=4) as pool:
        res = pool.map(predicted_test1, [(s, c, H) for s, c in plants for H in HORIZONS])
    pred = {}
    for s, H, v in res:
        pred.setdefault(s, {})[H] = v
    S = sorted(pred)
    rows = [json.loads(l) for l in (CP_RESULTS / "e7_price.jsonl").read_text().splitlines()]
    base = {r["plant"] + str(r["seed"]): r["return"] for r in rows
            if r["method"] == "greedy" and r["switch"] == "wifi"}
    r7 = {r["seed"] // 10: r["return"] - base[r["plant"] + str(r["seed"])] + 0.01 for r in rows
          if r["method"] == "ask-always" and r["switch"] == "wifi" and r["price"] == 0.01}
    g = {}
    grid = [json.loads(l) for l in (CP_RESULTS / "e1_test_main.jsonl").read_text().splitlines() if l.strip()]
    by = {(r["method"], r["seed"]): r for r in grid if r["run"] == 0}
    for s in S:
        a, b = by.get(("cauprobe-noprobe", s * 10)), by.get(("greedy", s * 10))
        if a and b:
            g[s] = np.cumsum(np.array(a["return_per_day"]) - np.array(b["return_per_day"]))
    Sg = [s for s in S if s in g]
    stored = json.loads((CP_RESULTS / "voi_ask.json").read_text())
    cell = {"n": len(S), "n_grid": len(Sg),
            **{f"pred_v1_H{H}": float(np.mean([pred[s][H] for s in S])) for H in HORIZONS},
            "stored_pred_mean_H30": stored["pred_mean"],
            "real_e7": float(np.mean([r7[s] for s in S])),
            "real_grid_7d": float(np.mean([g[s][6] for s in Sg])),
            "real_grid_14d": float(np.mean([g[s][13] for s in Sg])),
            "real_grid_by_day": [float(np.mean([g[s][d] for s in Sg])) for d in range(14)]}
    cell["ratio_v1_7d"] = cell["pred_v1_H7"] / cell["real_e7"]
    cell["ratio_v1_14d"] = cell["pred_v1_H14"] / cell["real_grid_14d"]
    print({k: v for k, v in cell.items() if k != "real_grid_by_day"}, flush=True)
    (CP_RESULTS / "voi_horizon_test1.json").write_text(json.dumps(cell, indent=1))


if __name__ == "__main__":
    import sys
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main_test1() if "--set" in sys.argv and "test1" in sys.argv else main()

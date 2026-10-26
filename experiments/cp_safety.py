"""What the safety machinery does under repeated faults (out-of-library).

The intermittently failing broker of the out-of-library set loses two-thirds
of its service rate in a random 12-18% of hours. For the eight such plants of
the second test set this re-runs the controllers with a per-step record of
the fault state, the detector's alarm state and the deadline misses, and
reports: misses inside and outside fault hours; per fault episode, the delay
until the detector raised an alarm (if it did) and the misses before it did;
the share of fault and non-fault hours spent in the fallback; recovery (misses
in the hour after an episode ends); and, as the reference the fallback is
checked against, the fixed fallback configuration and the rule-based
incumbent on the same plants and noise. Writes results/cauprobe/test2/safety.json.
"""
from __future__ import annotations

import json
from multiprocessing import get_context

import numpy as np

import cp_e1_commission as E
from cp_common import CP_RESULTS, OOL_TEST2_PLANTS
from causlm.probe import mech
from causlm.probe.plants import GatewayPlant, sample_plant
from causlm.probe.run import AskCost, STEPS_PER_DAY

DAYS = 14
METHODS = ["rule", "fallback", "greedy", "cauprobe", "cauprobe-v2"]
# detector sensitivity (--sens): the frozen controller with lower CUSUM
# thresholds; the in-library false-alarm rate of each threshold comes from
# the detector calibration runs (results/cauprobe/detector_calibration.json)
SENS = ["cauprobe-v2@cusum_h=5", "cauprobe-v2@cusum_h=10", "cauprobe-v2@cusum_h=15"]


def job(a):
    method, seed, combo, kind = a
    mech.set_switch_costs(0.3, 0.05)
    spec = sample_plant(seed, tuple(combo), ool=kind)
    agent, z = E.make_agent(method, spec, f"ool:{seed}", seed * 10)
    plant = GatewayPlant(spec, seed=seed * 10)
    ask = AskCost()
    if hasattr(agent, "cfg"):
        agent.cfg.ask_price = ask.price
    if z is not None and hasattr(agent, "offer_text"):
        agent.offer_text()
    obs = None
    T = DAYS * STEPS_PER_DAY
    fault = np.zeros(T, bool)
    alarm = np.zeros(T, bool)
    fb = np.zeros(T, bool)
    miss = np.zeros(T, bool)
    for t in range(T):
        if t % STEPS_PER_DAY == 0 and hasattr(agent, "set_forecast"):
            agent.set_forecast(plant.forecast_day)
        a, sp, do_ask, info = agent.act(obs)
        busy = ask.latency_ms if do_ask else 0.0
        obs, r, diag = plant.step(a, sp, busy)
        if do_ask:
            agent.receive_expert(z)
        agent.observe(obs, busy / (mech.DT * 1000.0))
        fault[t] = bool(getattr(plant, "_fault", False))
        alarm[t] = bool(getattr(agent, "misspec", False))
        fb[t] = bool(info.get("fallback", False))
        miss[t] = bool(diag["viol_net"])
    # fault episodes: maximal runs of fault steps
    ep, t = [], 0
    while t < T:
        if fault[t]:
            u = t
            while u < T and fault[u]:
                u += 1
            ep.append((t, u))
            t = u
        else:
            t += 1
    delays, before = [], []
    for s, u in ep:
        hit = np.flatnonzero(alarm[s:u])
        delays.append(int(hit[0]) if len(hit) else None)
        before.append(int(miss[s:s + (hit[0] if len(hit) else u - s)].sum()))
    after = [int(miss[u:u + 12].sum()) for _, u in ep if u + 12 <= T]
    return {"method": method, "seed": seed, "n_steps": T, "fault_steps": int(fault.sum()),
            "episodes": len(ep), "misses": int(miss.sum()),
            "misses_fault": int(miss[fault].sum()), "misses_clear": int(miss[~fault].sum()),
            "fallback_fault": float(fb[fault].mean()) if fault.any() else None,
            "fallback_clear": float(fb[~fault].mean()),
            "alarm_fault": float(alarm[fault].mean()) if fault.any() else None,
            "alarm_clear": float(alarm[~fault].mean()),
            "delays": delays, "misses_before_alarm": before, "misses_hour_after": after,
            "alarms": int(getattr(agent, "n_misspec_alarms", 0))}


def main():
    import sys
    sens = "--sens" in sys.argv
    methods = SENS if sens else METHODS
    plants = [(s, c, k) for s, c, k in OOL_TEST2_PLANTS if k == "broker_fault"]
    jobs = [(m, s, c, k) for m in methods for s, c, k in plants]
    with get_context("fork").Pool(16, maxtasksperchild=2) as pool:
        rows = pool.map(job, jobs)
    out = {}
    for m in methods:
        rr = [r for r in rows if r["method"] == m]
        d = [x for r in rr for x in r["delays"]]
        det = [x for x in d if x is not None]
        out[m] = {
            "misses": float(np.mean([r["misses"] for r in rr])),
            "misses_fault": float(np.mean([r["misses_fault"] for r in rr])),
            "misses_clear": float(np.mean([r["misses_clear"] for r in rr])),
            "miss_rate_fault": float(sum(r["misses_fault"] for r in rr)
                                     / sum(r["fault_steps"] for r in rr)),
            "miss_rate_clear": float(sum(r["misses_clear"] for r in rr)
                                     / sum(r["n_steps"] - r["fault_steps"] for r in rr)),
            "fallback_fault": float(np.mean([r["fallback_fault"] for r in rr])),
            "fallback_clear": float(np.mean([r["fallback_clear"] for r in rr])),
            "episodes": int(sum(r["episodes"] for r in rr)),
            "episodes_detected": len(det),
            "detect_delay_median_steps": float(np.median(det)) if det else None,
            "misses_before_alarm_mean": float(np.mean([x for r in rr for x in r["misses_before_alarm"]])),
            "misses_hour_after_mean": float(np.mean([x for r in rr for x in r["misses_hour_after"]])),
            "alarms": float(np.mean([r["alarms"] for r in rr])),
        }
        print(m, out[m], flush=True)
    (CP_RESULTS / "test2" / ("safety_sens.json" if sens else "safety.json")).write_text(json.dumps(
        {"summary": out, "rows": rows}, indent=1, default=float))


if __name__ == "__main__":
    main()

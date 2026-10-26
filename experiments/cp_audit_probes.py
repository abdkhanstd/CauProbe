"""Predicted versus realised value of individual probe decisions.

The decisive check of a value-of-information controller: does a probe it
chooses actually pay what it predicted, once the controller's own continued
learning is accounted for? At a probe decision the agent and the plant
(including the plant's random state) are copied. Branch A carries out the
probe as chosen; branch B takes the greedy action at that step instead. Both
branches then continue under the same policy -- probing again later if the
policy wants to -- to the end of the 14-day run, so the paired difference in
return is the realised value of that one decision, net of its cost and of
everything ordinary operation would have learned anyway. The predicted value
is the agent's own: the score advantage of the chosen probe over staying
greedy, times its length (gross VOI also recorded).

Both frozen iterations, on 16 plants of the second test set (one per
mechanism combination), under Wi-Fi and LPWAN reconfiguration costs; up to two
probe decisions per run (the first, and one drawn at random from the first
seven days). Writes results/cauprobe/test2/probe_audit.jsonl.
"""
from __future__ import annotations

import argparse
import copy
import json
from multiprocessing import get_context

import numpy as np

import cp_e1_commission as E
from cp_common import CP_RESULTS, TEST2_PLANTS
from causlm.probe import mech
from causlm.probe.plants import GatewayPlant, sample_plant
from causlm.probe.run import AskCost, STEPS_PER_DAY

SWITCH = {"wifi": (0.3, 0.05), "lpwan": (1.5, 0.2)}
DAYS = 14
OUT = CP_RESULTS / "test2" / "probe_audit.jsonl"


def step(agent, plant, obs, t, z, ask, force_greedy=False):
    """One step of run_commissioning's loop (no regime change, no ticket)."""
    if t % STEPS_PER_DAY == 0:
        agent.set_forecast(plant.forecast_day)
    if force_greedy:
        up = agent.cfg.use_probe
        agent.cfg.use_probe = False
        a, sp, do_ask, info = agent.act(obs)
        agent.cfg.use_probe = up
    else:
        a, sp, do_ask, info = agent.act(obs)
    busy = ask.latency_ms if do_ask else 0.0
    obs, r, diag = plant.step(a, sp, busy)
    if do_ask:
        r -= ask.price
        agent.receive_expert(z)
    agent.observe(obs, busy / (300.0 * 1000.0))
    return obs, r, info


def rollout(agent, plant, obs, t0, z, ask, force_first):
    tot = 0.0
    for t in range(t0, DAYS * STEPS_PER_DAY):
        obs, r, _ = step(agent, plant, obs, t, z, ask, force_greedy=(force_first and t == t0))
        tot += r
    return tot


def job(a):
    method, seed, combo, sw = a
    mech.set_switch_costs(*SWITCH[sw])
    spec = sample_plant(seed, tuple(combo))
    agent, z = E.make_agent(method, spec, f"eval:{seed}", seed * 10)
    plant = GatewayPlant(spec, seed=seed * 10)
    ask = AskCost(energy_j=40.0, latency_ms=6000.0, price_per_kj=0.4)
    if hasattr(agent, "bind_plant"):
        agent.bind_plant(plant)
    agent.cfg.ask_price = ask.price
    agent.offer_text()
    rng = np.random.default_rng(seed + 77)
    # first pass: find the probe decisions (new probes, not held ones)
    a0, p0 = copy.deepcopy(agent), copy.deepcopy(plant)
    obs, dec = None, []
    for t in range(DAYS * STEPS_PER_DAY):
        held = agent._commit is not None and t < agent._commit[1]
        obs, r, info = step(agent, plant, obs, t, z, ask)
        if info.get("probe") and not held:
            dec.append(t)
    early = [t for t in dec if t < 7 * STEPS_PER_DAY]
    pick = sorted(set(early[:1] + list(rng.choice(early[1:], size=min(1, len(early[1:])),
                                                  replace=False)) if len(early) > 1 else early[:1]))
    out = []
    agent, plant, obs = a0, p0, None
    for t in range(DAYS * STEPS_PER_DAY):
        if t in pick:
            A, PA = copy.deepcopy(agent), copy.deepcopy(plant)
            B, PB = copy.deepcopy(agent), copy.deepcopy(plant)
            # replay the decision step in A to read the prediction, then go on
            if t % STEPS_PER_DAY == 0:
                A.set_forecast(PA.forecast_day)
            aA, spA, askA, infoA = A.act(obs)
            pred_net = float(getattr(A, "_last_net", np.nan))
            pred_voi = float(getattr(A, "_last_voi", np.nan))
            m = int(A._commit[1] - t) if A._commit is not None else 1
            busy = ask.latency_ms if askA else 0.0
            oA, rA, _ = PA.step(aA, spA, busy)
            if askA:
                rA -= ask.price
                A.receive_expert(z)
            A.observe(oA, busy / (300.0 * 1000.0))
            retA = rA + rollout(A, PA, oA, t + 1, z, ask, False)
            retB = rollout(B, PB, obs, t, z, ask, True)
            out.append({"method": method, "seed": seed, "combo": list(combo),
                        "switch": sw, "t": t, "day": t // STEPS_PER_DAY,
                        "probe_action": int(aA), "m": m, "probe": bool(infoA.get("probe")),
                        "pred_net": pred_net, "pred_voi": pred_voi,
                        "realised": float(retA - retB)})
        obs, r, info = step(agent, plant, obs, t, z, ask)
        if t >= max(pick, default=-1):
            break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=56)
    ap.add_argument("--n_plants", type=int, default=16)
    a = ap.parse_args()
    plants = TEST2_PLANTS[: a.n_plants]
    jobs = [(m, s, c, sw) for sw in SWITCH for m in ("cauprobe", "cauprobe-v2")
            for s, c in plants]
    with get_context("fork").Pool(a.workers, maxtasksperchild=2) as pool, \
            open(OUT, "w") as f:
        for i, res in enumerate(pool.imap_unordered(job, jobs)):
            for r in res:
                f.write(json.dumps(r) + "\n")
            f.flush()
            print(f"{i + 1}/{len(jobs)}", flush=True)


if __name__ == "__main__":
    main()

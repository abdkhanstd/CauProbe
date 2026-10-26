"""E7: who should pay for which knowledge? Ask price x reconfiguration price.

For each price of one expert invocation (reward units; at the IT-energy price
of 0.4 per kJ, 0.016 is a ~40 J CPU call and 0.15 a ~376 J GPU call; the larger
values stand for energy-harvesting gateways or metered cloud models) and each
reconfiguration price (Wi-Fi-like 0.3 vs LPWAN-like 1.5), compare:

  cauprobe          prices both sources and buys what pays
  ask-always        reads the text whatever it costs, never probes
  probe-only        never reads the text
  greedy            neither

on 32 evaluation plants for 7 days. The claim under test is that the two
sources are substitutes and the adaptive agent tracks the cheaper one.
"""
from __future__ import annotations

import argparse
import json
import os
from multiprocessing import get_context

import numpy as np

import cp_e1_commission as E
from cp_common import CP_RESULTS, TEST_PLANTS
from causlm.probe import mech
from causlm.probe.agent import CauProbeAgent, ProbeConfig
from causlm.probe.baselines import greedy_cfg
from causlm.probe.plants import sample_plant
from causlm.probe.run import AskCost, run_commissioning

PRICES = [0.01, 0.1, 1.0, 3.0, 10.0, 30.0]
SWITCH = {"wifi": (0.3, 0.05), "lpwan": (1.5, 0.2)}


def job(a):
    method, seed, combo, price, sw = a
    mech.set_switch_costs(*SWITCH[sw])
    spec = sample_plant(seed, tuple(combo))
    key = f"eval:{seed}"
    z = E.expert_outputs()[key]["slm-0.5b-lora"]
    ex = E.load_expert("slm-0.5b-lora")
    s = seed * 10
    if method == "cauprobe":
        ag = CauProbeAgent(ProbeConfig(), seed=s, expert=ex, name=method)
    elif method in ("cauprobe-v2", "probe-only-v2"):
        # the second iteration (cp_e1_commission.make_agent, frozen config)
        ag, _ = E.make_agent("cauprobe-v2" if method == "cauprobe-v2"
                             else "cauprobe-v2-noask", spec, key, s)
        ag.name = method
        if method == "probe-only-v2":
            z = None
    elif method == "ask-always":
        ag = CauProbeAgent(ProbeConfig(use_probe=False, force_ask=True), seed=s,
                           expert=ex, name=method)
    elif method == "probe-only":
        ag = CauProbeAgent(ProbeConfig(use_ask=False), seed=s, name=method)
        z = None
    else:
        ag = CauProbeAgent(greedy_cfg(), seed=s, name="greedy")
        z = None
    # price expressed through the energy per call at the IT-energy price
    ask = AskCost(energy_j=price / 0.4 * 1000.0, latency_ms=6000.0,
                  price_per_kj=0.4)
    out = run_commissioning(spec, ag, days=7, seed=s, expert_out=z, ask=ask)
    out.update({"method": method, "price": price, "switch": sw})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=44)
    ap.add_argument("--n_plants", type=int, default=32)
    ap.add_argument("--set", default="test", choices=["test", "test2"])
    a = ap.parse_args()
    # held-out plants only, two per mechanism combination
    if a.set == "test2":
        # second fresh test set: both iterations, plus the second's probe-only
        from cp_common import TEST2_PLANTS
        plants = sorted(TEST2_PLANTS, key=lambda p: (p[0] - 910_000) // 16)[: a.n_plants]
        base_m, priced = ("probe-only", "probe-only-v2", "greedy"),             ("cauprobe", "cauprobe-v2", "ask-always")
        out_p = CP_RESULTS / "test2" / "e7_price.jsonl"
        out_p.parent.mkdir(parents=True, exist_ok=True)
    else:
        plants = sorted(TEST_PLANTS, key=lambda p: (p[0] - 900_000) // 16)[: a.n_plants]
        base_m, priced = ("probe-only", "greedy"), ("cauprobe", "ask-always")
        out_p = CP_RESULTS / "e7_price.jsonl"
    jobs = []
    for sw in SWITCH:
        for s, c in plants:
            for m in base_m:
                jobs.append((m, s, c, 0.0, sw))
            for price in PRICES:
                for m in priced:
                    jobs.append((m, s, c, price, sw))
    with get_context("fork").Pool(a.workers, maxtasksperchild=4) as pool, \
            open(out_p, "w") as f:
        for i, r in enumerate(pool.imap_unordered(job, jobs)):
            f.write(json.dumps(r) + "\n")
            f.flush()
            if (i + 1) % 50 == 0:
                print(f"{i+1}/{len(jobs)}", flush=True)


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main()

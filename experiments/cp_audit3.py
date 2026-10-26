"""Is a probe's realised value predictable at all? Replicated paired audit.

A single paired fork (cp_audit_probes.py, cp_audit2.py) measures one draw of a
probe decision's value: the future noise after the decision is one sample.
Here, for 48 of the audited decisions of CauProbe-2 (24 under Wi-Fi and 24
under LPWAN costs, drawn at random), the run is replayed to the same decision
state and the probe-versus-greedy fork is repeated with REPS independent
future noise streams (both branches of a replicate share theirs). The mean
over replicates estimates the decision's expected value; the spread is the
noise no decision-time score can foresee. Writes
results/cauprobe/test2/probe_audit3.jsonl.

Powered extension (--reps, --chunk, --out): with the 48 x 6 run only ~6% of a
single draw's variance lay between decisions, so a 6-replicate mean has a
reliability of ~0.3. `--n 999 --reps 36 --chunk 6` replays every audited
decision of cp_audit2.py (157) 36 times. Replicate k is reseeded exactly as
before (so replicates 0-5 reproduce the 48 x 6 run); each job covers one
decision and one block of `chunk` replicates, and blocks are queued block by
block, so the file holds a balanced design (every decision with the same
number of replicates) whenever a block is complete. One line per job.
"""
from __future__ import annotations

import argparse
import copy
import json
import time
from multiprocessing import get_context

import numpy as np

import cp_e1_commission as E
from cp_common import CP_RESULTS
from causlm.probe import mech
from causlm.probe.plants import GatewayPlant, sample_plant
from causlm.probe.run import AskCost, STEPS_PER_DAY
from cp_audit_probes import DAYS, SWITCH, rollout, step

SRC = CP_RESULTS / "test2" / "probe_audit2.jsonl"
OUT = CP_RESULTS / "test2" / "probe_audit3.jsonl"
REPS = 6


def reseed(agent, plant, k):
    plant.rng = np.random.default_rng(10_000_019 * (k + 1) + plant.spec.seed)
    agent.rng = np.random.default_rng(20_000_003 * (k + 1) + plant.spec.seed)
    agent._fl_rng = np.random.default_rng(30_000_001 * (k + 1) + plant.spec.seed)


def job(a):
    seed, combo, sw, t_dec, pred_net, k0, k1 = a
    t0 = time.time()
    mech.set_switch_costs(*SWITCH[sw])
    spec = sample_plant(seed, tuple(combo))
    agent, z = E.make_agent("cauprobe-v2", spec, f"eval:{seed}", seed * 10)
    plant = GatewayPlant(spec, seed=seed * 10)
    ask = AskCost(energy_j=40.0, latency_ms=6000.0, price_per_kj=0.4)
    agent.cfg.ask_price = ask.price
    agent.offer_text()
    obs = None
    for t in range(t_dec):                     # replay to the decision state
        obs, r, info = step(agent, plant, obs, t, z, ask)
    vals, probed = [], []
    for k in range(k0, k1):
        A, PA = copy.deepcopy(agent), copy.deepcopy(plant)
        B, PB = copy.deepcopy(agent), copy.deepcopy(plant)
        # the decision step itself with the original random state, so that A
        # takes exactly the audited probe; then both branches get the same
        # fresh future noise for this replicate
        oA, rA, infoA = step(A, PA, obs, t_dec, z, ask)
        oB, rB, _ = step(B, PB, obs, t_dec, z, ask, force_greedy=True)
        probed.append(bool(infoA.get("probe")))
        reseed(A, PA, k)
        reseed(B, PB, k)
        retA = rA + rollout(A, PA, oA, t_dec + 1, z, ask, False)
        retB = rB + rollout(B, PB, oB, t_dec + 1, z, ask, False)
        vals.append(float(retA - retB))
    return {"seed": seed, "combo": list(combo), "switch": sw, "t": t_dec,
            "pred_net": pred_net, "k0": k0, "probed": probed, "realised_reps": vals,
            "wall_s": time.time() - t0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--n", type=int, default=48)       # 999: every decision
    ap.add_argument("--reps", type=int, default=REPS)
    ap.add_argument("--chunk", type=int, default=0)    # replicates per job (0: all)
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()
    chunk = a.chunk or a.reps
    rows = [json.loads(l) for l in SRC.read_text().splitlines() if l.strip()]
    rng = np.random.default_rng(5)
    pick = []
    for sw in ("wifi", "lpwan"):
        rr = [r for r in rows if r["switch"] == sw and np.isfinite(r["pred_net"])]
        for i in rng.choice(len(rr), size=min(a.n // 2, len(rr)), replace=False):
            r = rr[i]
            pick.append((r["seed"], r["combo"], sw, r["t"], r["pred_net"]))
    done = set()
    if a.resume:
        try:
            with open(a.out) as f:
                done = {(d["seed"], d["switch"], d["t"], d["k0"])
                        for d in map(json.loads, filter(str.strip, f))}
        except FileNotFoundError:
            pass
    jobs = [(*p, k0, min(k0 + chunk, a.reps)) for k0 in range(0, a.reps, chunk)
            for p in pick if (p[0], p[2], p[3], k0) not in done]
    print(f"{len(pick)} decisions x {a.reps} replicates, {len(jobs)} jobs -> {a.out}",
          flush=True)
    t0 = time.time()
    with get_context("fork").Pool(a.workers, maxtasksperchild=1) as pool, \
            open(a.out, "a" if a.resume else "w") as f:
        for i, res in enumerate(pool.imap_unordered(job, jobs)):
            f.write(json.dumps(res) + "\n")
            f.flush()
            print(f"{i + 1}/{len(jobs)} k0={res['k0']} {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

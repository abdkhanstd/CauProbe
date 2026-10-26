"""Development diagnostic for the second iteration (dev plants only).

Runs a method on the 16 development plants (seeds 500_000-500_015) and records,
for every probe it starts, what the controller predicted: per-day gain,
information horizon, kappa, EVPI and the structure posterior on factor B; and,
at every first ask, the predicted value of reading the notes. Never touches the
test plants.

  cp_v2_diag.py --method cauprobe --runs 2 --tag diag_v1
"""
from __future__ import annotations

import argparse
import json
import os
from multiprocessing import get_context

import numpy as np

import cp_e1_commission as E
from cp_common import CP_RESULTS, DEV_EVAL_SEEDS, EVAL_PLANTS
from causlm.probe import mech
from causlm.probe.plants import sample_plant
from causlm.probe.run import AskCost, run_commissioning

OUT = CP_RESULTS / "v2_dev"
OUT.mkdir(parents=True, exist_ok=True)


def job(a):
    method, seed, combo, run, days, scen = a
    sc = E.SCENARIOS[scen]
    E.NOTES = sc.get("notes", "nominal")
    mech.set_switch_costs(*sc.get("switch", (0.3, 0.05)))
    spec = sample_plant(seed, tuple(combo), scenario=sc.get("plant"))
    key = f"eval:{seed}"
    ag, z = E.make_agent(method, spec, key, seed * 10 + run)
    rec = {"probes": [], "asks": []}
    if hasattr(ag, "_kg_action"):
        f = ag._kg_action

        def kg(*args, f=f, ag=ag, **kw):
            a, m = f(*args, **kw)
            a_g = args[-1]
            if a != a_g:
                rec["probes"].append({
                    "t": ag.t, "a": int(a), "m": int(m),
                    "gain": float(ag._last_pred), "h_eff": float(ag._h_eff),
                    "kappa": float(ag.kappa), "evpi": float(ag._evpi_last),
                    "tau": getattr(ag, "_last_tau", None),
                    "pB1": float(ag.nb.factor_marginals()["B"])})
            return a, m
        ag._kg_action = kg
    if hasattr(ag, "_voi_ask"):
        g = ag._voi_ask

        def va(g=g, ag=ag):
            v = g()
            rec["asks"].append({"t": ag.t, "voi": float(v),
                                "parts": getattr(ag, "_ask_parts", None)})
            return v
        ag._voi_ask = va
    out = run_commissioning(spec, ag, days=days, seed=seed * 10 + run,
                            expert_out=z, ask=AskCost(energy_j=40.0))
    out.update({"run": run, "method": method, "scenario": scen, "diag": rec})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", default="cauprobe")
    ap.add_argument("--runs", type=int, default=2)
    ap.add_argument("--days", type=int, default=14)
    ap.add_argument("--scenario", default="nominal")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--workers", type=int, default=32)
    a = ap.parse_args()
    plants = [(s, c) for s, c in EVAL_PLANTS if s in DEV_EVAL_SEEDS]
    jobs = [(m, s, c, r, a.days, a.scenario) for m in a.methods.split(",")
            for s, c in plants for r in range(a.runs)]
    with get_context("fork").Pool(a.workers, maxtasksperchild=2) as pool, \
            open(OUT / f"{a.tag}.jsonl", "w") as f:
        for r in pool.imap_unordered(job, jobs):
            r.pop("_steps", None)
            f.write(json.dumps(r) + "\n")
            f.flush()
            print(r["method"], r["seed"], round(r["return"], 1),
                  r["probe_steps"], flush=True)


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main()

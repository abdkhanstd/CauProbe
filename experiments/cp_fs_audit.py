"""Do posterior-dependent admissible sets matter to the valuation?

The implemented valuation recomputes the safety filter's admissible set for
every counterfactual posterior; the theory freezes it. With the analysis flag
fs_audit, every probe valuation also values the same candidates with the set
frozen at the current posterior (Proposition 1's setting) and logs both. This
script re-runs CauProbe-1 and CauProbe-2 on one second-test plant per
mechanism combination (16 plants; first run; Wi-Fi costs, and LPWAN costs for
CauProbe-2) with the flag on, checks that every run reproduces its stored
return bit for bit (the flag changes no decision), and reports
  - how often an implemented gain E[V(beta')] - V(beta) is negative, and by
    how much relative to EVPI (Prop. 1's lower bound needs frozen sets),
  - whether any gain exceeds EVPI (the upper bound, i.e. the exact screen,
    holds for the implementation as well),
  - how often the frozen-set valuation would have chosen differently.
Writes results/cauprobe/fs_audit.json.
"""
from __future__ import annotations

import argparse
import json
from multiprocessing import get_context

import numpy as np

import cp_e1_commission as E
from cp_common import CP_RESULTS, TEST2_PLANTS
from causlm.probe import mech
from causlm.probe.plants import sample_plant
from causlm.probe.run import AskCost, run_commissioning

PLANTS = TEST2_PLANTS[:16]
SWITCH = {"nominal": (0.3, 0.05), "lpwan": (1.5, 0.2)}
JOBS = [(m, sc, s, c) for s, c in PLANTS for m, sc in
        (("cauprobe", "nominal"), ("cauprobe-v2", "nominal"), ("cauprobe-v2", "lpwan"))]


def job(a):
    method, sc, seed, combo = a
    mech.set_switch_costs(*SWITCH[sc])
    E.NOTES = "nominal"
    spec = sample_plant(seed, tuple(combo))
    agent, z = E.make_agent(method, spec, f"eval:{seed}", seed * 10)
    agent.cfg.fs_audit = True
    out = run_commissioning(spec, agent, days=14, seed=seed * 10, expert_out=z,
                            ask=AskCost(energy_j=40.0, latency_ms=6000.0, price_per_kj=0.4))
    return {"method": method, "scenario": sc, "seed": seed * 10, "return": out["return"],
            "log": getattr(agent, "_fs_log", [])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    stored = {}
    for sc in SWITCH:
        for l in (CP_RESULTS / "test2" / f"e1_{sc}.jsonl").read_text().splitlines():
            if l.strip():
                r = json.loads(l)
                stored[(r["method"], sc, r["seed"])] = r["return"]
    runs = []
    with get_context("fork").Pool(a.workers, maxtasksperchild=1) as pool:
        for r in pool.imap_unordered(job, JOBS):
            r["identical"] = r["return"] == stored.get((r["method"], r["scenario"], r["seed"]))
            runs.append(r)
            print(r["method"], r["scenario"], r["seed"], len(r["log"]), "valuations, identical:",
                  r["identical"], flush=True)
    out = {"runs_identical": int(sum(r["identical"] for r in runs)), "runs": len(runs), "by": {}}
    for m, sc in (("cauprobe", "nominal"), ("cauprobe-v2", "nominal"), ("cauprobe-v2", "lpwan")):
        L = [e for r in runs if r["method"] == m and r["scenario"] == sc for e in r["log"]]
        if not L:
            continue
        neg = [e for e in L if e["n_negative"] > 0]
        over = [e for e in L if e["max_over_evpi"] > 1e-9]
        out["by"][f"{m}|{sc}"] = {
            "valuations_over_evpi": len(over),
            "valuations_over_evpi_frozen": int(sum(e["max_over_evpi_frozen"] > 1e-9 for e in L)),
            "max_over_evpi_rel": float(max((e["max_over_evpi"] / max(e["evpi"], 1e-9) for e in over),
                                           default=0.0)),
            "median_over_evpi": float(np.median([e["max_over_evpi"] for e in over])) if over else 0.0,
            "probe_decision_differs": int(sum(e["probe"] != e["probe_frozen"] for e in L)),
            "valuations": len(L),
            "candidate_values": int(sum(e["n"] for e in L)),
            "valuations_with_negative_gain": len(neg),
            "negative_candidate_values": int(sum(e["n_negative"] for e in L)),
            "negative_candidate_values_frozen": int(sum(e["n_negative_frozen"] for e in L)),
            "worst_negative_gain": float(min(e["min_gain"] for e in L)),
            "worst_negative_gain_rel_evpi": float(min(e["min_gain"] / max(e["evpi"], 1e-9) for e in L)),
            "median_negative_rel_evpi": float(np.median([e["min_gain"] / max(e["evpi"], 1e-9)
                                                         for e in neg])) if neg else 0.0,
            "max_gain_over_evpi": float(max(e["max_over_evpi"] for e in L)),
            "max_gain_over_evpi_frozen": float(max(e["max_over_evpi_frozen"] for e in L)),
            "same_choice": float(np.mean([e["choice"] == e["choice_frozen"] for e in L])),
            "same_probe_decision": float(np.mean([e["probe"] == e["probe_frozen"] for e in L])),
            "probes_impl": int(sum(e["probe"] for e in L)),
            "probes_frozen": int(sum(e["probe_frozen"] for e in L)),
        }
    (CP_RESULTS / "fs_audit.json").write_text(json.dumps(out, indent=1))
    with open(CP_RESULTS / "fs_audit_log.jsonl", "w") as f:
        for r in sorted(runs, key=lambda r: (r["method"], r["scenario"], r["seed"])):
            f.write(json.dumps(r) + "\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()

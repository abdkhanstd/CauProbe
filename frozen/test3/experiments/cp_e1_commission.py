"""E1/E2/E3: commissioning on held-out plants, every method, paired seeds.

Each (plant, run) pair is one gateway commissioned for `--days` days. All
methods see the same plant, the same exogenous days and the same noise stream
(`seed = plant_seed * 10 + run`), so per-plant differences are paired.

Usage
  cp_e1_commission.py --methods all --days 14 --runs 2 --workers 48
Writes results/cauprobe/e1_<tag>.jsonl, one JSON line per (method, plant, run),
appended as runs finish so a long job can be resumed with --resume.
"""
from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from multiprocessing import get_context

import numpy as np

from cp_common import (CP_RESULTS, DEV_EVAL_SEEDS, EVAL_PLANTS, EXPERT_DIR,
                       HELDOUT_PLANTS, OOL_PLANTS, TEST_PLANTS)
from causlm.probe.agent import CauProbeAgent, ProbeConfig
from causlm.probe.baselines import (FixedAgent, IDSAgent, LinUCBAgent,
                                    NominalModelAgent, OracleAgent, RuleAgent,
                                    ThompsonAgent, advice_from_expert,
                                    greedy_cfg)
from causlm.probe.expert import ExpertModel
from causlm.probe.plants import (FALLBACK_NET, FALLBACK_SCHED, generate_logs,
                                 sample_plant)
from causlm.probe.run import AskCost, run_commissioning

_OUTPUTS = {}
NOTES = "nominal"

# Deployment scenarios. Each changes one thing a real site can differ in.
SCENARIOS = {
    "nominal": {},
    "noisy": {"plant": {"sigma": (0.35, 0.55), "burst_p": 0.10,
                        "burst": (2.0, 4.0)}},
    "lpwan": {"switch": (1.5, 0.2)},
    "noisy_lpwan": {"plant": {"sigma": (0.35, 0.55), "burst_p": 0.10,
                              "burst": (2.0, 4.0)}, "switch": (1.5, 0.2)},
    "poor_docs": {"notes": "poor_docs"},
    "shift": {"regime": {"day": 3}},
}


def expert_outputs():
    if NOTES not in _OUTPUTS:
        merged = {}
        if NOTES == "nominal":
            # every expert scored on the nominal notes (cp_expert.py) ...
            d = json.loads((CP_RESULTS / "expert_outputs.json").read_text())
            for k in d["keys"]:
                merged.setdefault(k, {}).update(d["outputs"][k])
        # ... plus the scenario's own scoring (identical notes for nominal,
        # different notes for poor_docs) and the change tickets
        p = CP_RESULTS / f"expert_outputs_{NOTES}.json"
        if p.exists():
            d = json.loads(p.read_text())
            for k in d["keys"]:
                merged.setdefault(k, {}).update(d["outputs"][k])
        # ... plus the notes of the fresh out-of-library test plants, and of
        # the second test set (cp_expert_score_test2.py)
        extra = [CP_RESULTS / f"expert_outputs_test2_{NOTES}.json",
                 CP_RESULTS / f"expert_outputs_test3_{NOTES}.json"]
        if NOTES == "nominal":
            extra += [CP_RESULTS / "expert_outputs_ool_test.json",
                      CP_RESULTS / "expert_outputs_ool_test2.json"]
        for p in extra:
            if p.exists():
                d = json.loads(p.read_text())
                for k in d["keys"]:
                    merged.setdefault(k, {}).update(d["outputs"][k])
        _OUTPUTS[NOTES] = merged
    return _OUTPUTS[NOTES]


def load_expert(name):
    p = EXPERT_DIR / f"{name}.json"
    return ExpertModel.from_json(p) if p.exists() else None


def make_agent(method, spec, key, seed):
    """Factory for every method in the study."""
    ex_name = "slm-0.5b-lora"
    if method.endswith("-fleet"):
        # the same model output, read through the fleet-calibrated expert
        # (cp_build_fleet_expert.py): temperatures and logit bank refitted on
        # interventional labels from earlier sites
        ag, z = make_agent(method[:-len("-fleet")], spec, key, seed)
        ag.expert = load_expert(ex_name + "-fleet")
        assert ag.expert is not None, "run cp_build_fleet_expert.py first"
        ag.name = method
        return ag, z
    if method == "oracle":
        return OracleAgent(spec, seed=seed), None
    if method == "fallback":
        return FixedAgent(FALLBACK_NET, FALLBACK_SCHED, "fallback"), None
    if method == "rule":
        return RuleAgent(), None
    if method == "nominal":
        return NominalModelAgent(seed=seed), None
    if method == "greedy":
        return CauProbeAgent(greedy_cfg(), seed=seed, name="greedy"), None
    if method == "thompson":
        return ThompsonAgent(safe=True, seed=seed), None
    if method == "thompson-unsafe":
        return ThompsonAgent(safe=False, seed=seed), None
    if method == "ids":
        return IDSAgent(seed=seed), None
    if method == "linucb":
        return LinUCBAgent(seed=seed), None
    if method == "identify":
        return CauProbeAgent(ProbeConfig(use_ask=False, objective="identify"),
                             seed=seed, name="identify"), None
    if method == "log-naive":
        ag = CauProbeAgent(greedy_cfg(), seed=seed, name="log-naive")
        ag.preload_logs(generate_logs(spec, days=14, seed=seed))
        return ag, None
    if method == "cauprobe+logs-naive":
        ag = CauProbeAgent(ProbeConfig(use_ask=False), seed=seed,
                           name="cauprobe+logs-naive")
        ag.preload_logs(generate_logs(spec, days=14, seed=seed))
        return ag, None
    if method == "cauprobe-noask":
        return CauProbeAgent(ProbeConfig(use_ask=False), seed=seed,
                             name="cauprobe-noask"), None
    if method == "cauprobe-noprobe":
        return CauProbeAgent(ProbeConfig(use_probe=False), seed=seed,
                             expert=load_expert(ex_name),
                             name="cauprobe-noprobe"), expert_outputs()[key][ex_name]
    if method == "cauprobe-nosafety":
        return CauProbeAgent(ProbeConfig(use_ask=False, use_safety=False),
                             seed=seed, name="cauprobe-nosafety"), None
    if method in ("cauprobe-particle", "cauprobe-nodur", "cauprobe-nokappa"):
        kw = {"cauprobe-particle": {"voi_level": "particle", "durability": False},
              "cauprobe-nodur": {"durability": False},
              "cauprobe-nokappa": {"self_calibrate": False}}[method]
        return CauProbeAgent(ProbeConfig(use_ask=False, **kw), seed=seed,
                             name=method), None
    if method.startswith("cauprobe-h"):
        h = float(method[len("cauprobe-h"):])
        return CauProbeAgent(ProbeConfig(use_ask=False, h_days=h), seed=seed,
                             name=method), None
    if method == "cauprobe":
        return CauProbeAgent(ProbeConfig(), seed=seed, expert=load_expert(ex_name),
                             name="cauprobe"), expert_outputs()[key][ex_name]
    if method.startswith("cauprobe-v2"):
        # second iteration: value of information net of free learning.
        # Frozen 2026-10-17 after one validation run (results/cauprobe/
        # validation_v2); the configuration below is what was tested.
        # "cauprobe-v2-noask" never reads the notes; a suffix "@k=v+k=v"
        # overrides ProbeConfig fields (development variants only)
        base, _, over = method.partition("@")
        kw = {"free_learning": True, "reopen_rate": True, "probe_return": True,
              "fl_tail": "geometric", "fl_days": 3}
        for item in filter(None, over.split("+")):
            k, v = item.split("=")
            kw[k] = type(getattr(ProbeConfig, k))(v) if not isinstance(
                getattr(ProbeConfig, k), bool) else v in ("1", "True", "true")
        if base == "cauprobe-v2-noask":
            return CauProbeAgent(ProbeConfig(use_ask=False, **kw), seed=seed,
                                 name=method), None
        assert base == "cauprobe-v2", method
        return CauProbeAgent(ProbeConfig(**kw), seed=seed,
                             expert=load_expert(ex_name),
                             name=method), expert_outputs()[key][ex_name]
    if method.startswith("cauprobe-expert:"):
        en = method.split(":", 1)[1]
        return CauProbeAgent(ProbeConfig(), seed=seed, expert=load_expert(en),
                             name=method), expert_outputs()[key][en.replace("-uncal", "")]
    if method == "advice-plan":
        z = expert_outputs()[key][ex_name]
        adv = advice_from_expert(z, seed=seed)
        return LinUCBAgent(seed=seed, advice=adv), None
    raise ValueError(method)


ALL = ["oracle", "rule", "fallback", "nominal", "linucb", "advice-plan",
       "greedy", "log-naive", "thompson", "thompson-unsafe", "ids", "identify",
       "cauprobe-noask", "cauprobe-noprobe", "cauprobe", "cauprobe-nosafety",
       "cauprobe+logs-naive"]


def job(args):
    method, seed, combo, ool, run, days, ask_cfg, scen = args
    global NOTES
    try:
        sc = SCENARIOS[scen]
        NOTES = sc.get("notes", "nominal")
        from causlm.probe import mech
        mech.set_switch_costs(*sc.get("switch", (0.3, 0.05)))
        spec = sample_plant(seed, tuple(combo), ool=ool,
                            scenario=sc.get("plant"))
        key = f"{'ool' if ool else 'eval'}:{seed}"
        agent, z = make_agent(method, spec, key, seed * 10 + run)
        regime = None
        if "regime" in sc:
            # the radio is swapped mid-deployment: A flips. Half the sites
            # file a maintenance ticket saying so; the other half do not.
            rng = np.random.default_rng(seed + 41)
            newA = 1 - spec.combo[0]
            upd = {"A": newA,
                   "t_wake": float(rng.uniform(90.0, 170.0)) if newA == 1 else 0.0}
            regime = {"t": sc["regime"]["day"] * 288, "net": upd}
            if seed % 2 == 0:
                outs = expert_outputs()
                regime["ticket"] = outs.get(f"ticket:{seed}", {}).get(
                    "slm-0.5b-lora")
        out = run_commissioning(spec, agent, days=days, seed=seed * 10 + run,
                                expert_out=z, ask=AskCost(**ask_cfg),
                                regime=regime)
        out["run"] = run
        out["method"] = method
        out["scenario"] = scen
        out["ticket"] = bool(regime and regime.get("ticket") is not None)
        return out
    except Exception as e:
        return {"method": method, "plant_seed": seed, "run": run,
                "error": repr(e), "tb": traceback.format_exc()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", default="all")
    ap.add_argument("--days", type=int, default=14)
    ap.add_argument("--runs", type=int, default=2)
    ap.add_argument("--plants", default="eval")       # eval | ool | eval:16
    ap.add_argument("--workers", type=int, default=40)
    ap.add_argument("--tag", default="main")
    ap.add_argument("--outdir", default="")         # sub-directory of CP_RESULTS
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--ask_j", type=float, default=40.0)
    ap.add_argument("--ask_ms", type=float, default=6000.0)
    ap.add_argument("--price_per_kj", type=float, default=0.4)
    ap.add_argument("--scenario", default="nominal", choices=list(SCENARIOS))
    args = ap.parse_args()
    methods = ALL if args.methods == "all" else args.methods.split(",")
    if args.plants == "dev":
        plants = [(s, c, None) for s, c in EVAL_PLANTS if s in DEV_EVAL_SEEDS]
    elif args.plants == "test":
        plants = [(s, c, None) for s, c in TEST_PLANTS]
    elif args.plants == "val":
        # the 48 validation plants (evaluation plants minus the dev plants)
        plants = [(s, c, None) for s, c in HELDOUT_PLANTS]
    elif args.plants == "test2":
        # the second fresh test set (second iteration frozen beforehand)
        from cp_common import TEST2_PLANTS
        plants = [(s, c, None) for s, c in TEST2_PLANTS]
    elif args.plants == "test3":
        # the third fresh test set (fleet calibration frozen beforehand)
        from cp_common import TEST3_PLANTS
        plants = [(s, c, None) for s, c in TEST3_PLANTS]
    elif args.plants == "ool2":
        from cp_common import OOL_TEST2_PLANTS
        plants = [(s, c, k) for s, c, k in OOL_TEST2_PLANTS]
    elif args.plants == "oolval":
        # validation out-of-library plants with scored notes (TSCH, backhaul)
        from cp_common import OOL_VAL_PLANTS
        plants = [(s, c, k) for s, c, k in OOL_VAL_PLANTS if k != "broker_fault"]
    elif args.plants.startswith("eval"):
        plants = [(s, c, None) for s, c in EVAL_PLANTS]
        if ":" in args.plants:
            n = int(args.plants.split(":")[1])
            # one plant per combination first, so a small pilot covers all 16
            plants = sorted(plants, key=lambda p: (p[0] - 500_000) // 16)[:n]
    else:
        plants = [(s, c, k) for s, c, k in OOL_PLANTS]
    ask_cfg = {"energy_j": args.ask_j, "latency_ms": args.ask_ms,
               "price_per_kj": args.price_per_kj}
    out_path = CP_RESULTS / args.outdir / f"e1_{args.tag}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if args.resume and out_path.exists():
        for line in out_path.read_text().splitlines():
            d = json.loads(line)
            if "error" not in d:
                done.add((d["method"], d["seed"] // 10, d["run"]))
    jobs = [(m, s, c, k, r, args.days, ask_cfg, args.scenario)
            for r in range(args.runs) for s, c, k in plants for m in methods
            if (m, s, r) not in done]
    print(f"{len(jobs)} jobs -> {out_path}", flush=True)
    t0 = time.time()
    ctx = get_context("fork")
    n_ok = 0
    with ctx.Pool(args.workers, maxtasksperchild=4) as pool, \
            open(out_path, "a") as f:
        for i, res in enumerate(pool.imap_unordered(job, jobs)):
            if "error" in res:
                print("ERROR", res["method"], res.get("plant_seed"), res["error"],
                      flush=True)
                print(res["tb"], flush=True)
                continue
            res.pop("_steps", None)
            f.write(json.dumps(res) + "\n")
            f.flush()
            n_ok += 1
            if (i + 1) % 10 == 0 or i + 1 == len(jobs):
                print(f"  {i+1}/{len(jobs)} done ({time.time()-t0:.0f}s)",
                      flush=True)
    print("finished", n_ok, "ok")


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main()

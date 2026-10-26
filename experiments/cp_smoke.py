"""Smoke test: every agent on a couple of plants for a short horizon."""
from __future__ import annotations

import sys
import time

import numpy as np

from cp_common import EVAL_PLANTS
from causlm.probe.agent import CauProbeAgent, ProbeConfig
from causlm.probe.baselines import (FixedAgent, IDSAgent, LinUCBAgent,
                                    NominalModelAgent, OracleAgent, RuleAgent,
                                    ThompsonAgent, greedy_cfg)
from causlm.probe.plants import FALLBACK_NET, FALLBACK_SCHED, sample_plant
from causlm.probe.run import run_commissioning


def main(days=2, which=None):
    for seed, combo in EVAL_PLANTS[:2]:
        spec = sample_plant(seed, combo)
        agents = {
            "oracle": lambda: OracleAgent(spec, seed=1),
            "fixed-fallback": lambda: FixedAgent(FALLBACK_NET, FALLBACK_SCHED),
            "rule": lambda: RuleAgent(),
            "greedy": lambda: CauProbeAgent(greedy_cfg(), seed=1, name="greedy"),
            "cauprobe-noask": lambda: CauProbeAgent(ProbeConfig(use_ask=False), seed=1),
            "thompson": lambda: ThompsonAgent(seed=1),
            "ids": lambda: IDSAgent(seed=1),
            "linucb": lambda: LinUCBAgent(seed=1),
            "nominal": lambda: NominalModelAgent(seed=1),
            "identify": lambda: CauProbeAgent(ProbeConfig(use_ask=False, objective="identify"), seed=1, name="identify"),
        }
        print("plant", spec.name, flush=True)
        for name, mk in agents.items():
            if which and name not in which:
                continue
            t0 = time.time()
            out = run_commissioning(spec, mk(), days=days, seed=seed * 10)
            print(f"  {name:16s} return {out['return']:9.2f} net {out['r_net']:9.2f} "
                  f"th {out['r_th']:7.2f} viol {out['viol_net']:3d}/{out['viol_th']:3d} "
                  f"probe {out['probe_steps']:4d} fb {out['fallback_steps']:4d} "
                  f"thp {out['th_probes']} post {out.get('final_factor_post')} "
                  f"mass {out.get('final_mass_post')} ({time.time()-t0:.1f}s)",
                  flush=True)


if __name__ == "__main__":
    d = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    which = sys.argv[2].split(",") if len(sys.argv) > 2 else None
    main(d, which)

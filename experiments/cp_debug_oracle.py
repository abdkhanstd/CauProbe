"""Why does the known-plant reference trail the learners? Step-level diff."""
import collections

import numpy as np

from cp_common import EVAL_PLANTS
from causlm.probe import mech
from causlm.probe.agent import CauProbeAgent
from causlm.probe.baselines import OracleAgent, greedy_cfg
from causlm.probe.plants import sample_plant
from causlm.probe.run import run_commissioning

tot = collections.defaultdict(float)
for seed, combo in EVAL_PLANTS[:6]:
    spec = sample_plant(seed, combo)
    res = {}
    for name, mk in (("oracle", lambda: OracleAgent(spec, seed=1)),
                     ("oracle-expocc", lambda: OracleAgent(spec, seed=1)),
                     ("greedy", lambda: CauProbeAgent(greedy_cfg(), seed=1,
                                                      name="greedy"))):
        ag = mk()
        if name == "oracle-expocc":
            ag.bind_plant = lambda plant: None
        res[name] = run_commissioning(spec, ag, days=3, seed=seed * 10,
                                      record_steps=True)
    o, g = res["oracle"]["_steps"], res["greedy"]["_steps"]
    ro = np.array([s["r"] for s in o]); rg = np.array([s["r"] for s in g])
    ao = np.array([s["a"] for s in o]); ag_ = np.array([s["a"] for s in g])
    lo = np.array([s["lat"] for s in o]); lg = np.array([s["lat"] for s in g])
    print(spec.name, "oracle %.1f oracle-expocc %.1f greedy %.1f | switches o=%d g=%d"
          % (ro.sum(), res["oracle-expocc"]["return"], rg.sum(),
             res["oracle"]["n_switches"], res["greedy"]["n_switches"]))
    print("   mean lat o=%.0f g=%.0f  p95 o=%.0f g=%.0f  r_net o=%.1f g=%.1f r_th o=%.1f g=%.1f"
          % (lo.mean(), lg.mean(), np.percentile(lo, 95), np.percentile(lg, 95),
             res["oracle"]["r_net"], res["greedy"]["r_net"],
             res["oracle"]["r_th"], res["greedy"]["r_th"]))
    co = collections.Counter(map(int, ao)).most_common(3)
    cg = collections.Counter(map(int, ag_)).most_common(3)
    print("   top actions o", [(tuple(mech.NET_ACTIONS[a]), n) for a, n in co])
    print("   top actions g", [(tuple(mech.NET_ACTIONS[a]), n) for a, n in cg])
    for k in ("oracle", "oracle-expocc", "greedy"):
        tot[k] += res[k]["return"]
print({k: round(v, 1) for k, v in tot.items()})

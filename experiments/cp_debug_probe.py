"""Trace CauProbe's structure posterior and probes on one plant."""
import sys

import numpy as np

from cp_common import EVAL_PLANTS
from causlm.probe import mech
from causlm.probe.agent import CauProbeAgent, ProbeConfig
from causlm.probe.plants import GatewayPlant, sample_plant

i = int(sys.argv[1]) if len(sys.argv) > 1 else 14
days = int(sys.argv[2]) if len(sys.argv) > 2 else 2
seed, combo = EVAL_PLANTS[i]
spec = sample_plant(seed, combo)
print(spec.name, {k: round(float(v), 3) for k, v in spec.net.items()})
ag = CauProbeAgent(ProbeConfig(use_ask=False), seed=seed * 10)
plant = GatewayPlant(spec, seed=seed * 10)
obs = None
names = ["".join(f"{x}{v}" for x, v in zip("ABC", c)) for c in ag.nb.combos]
for t in range(days * 288):
    if t % 288 == 0:
        ag.set_forecast(plant.forecast_day)
    a, sp, ask, info = ag.act(obs)
    obs, r, diag = plant.step(a, sp)
    ag.observe(obs)
    if info.get("probe") or t % 24 == 0:
        cp = ag.nb.combo_post()
        top = np.argsort(-cp)[:3]
        print(f"t={t:4d} h={t%288/12:5.1f} a={tuple(mech.NET_ACTIONS[a])} "
              f"{'PROBE' if info.get('probe') else '     '} lat={obs['latency_ms']:6.0f} "
              f"evpi={info.get('evpi', float('nan')):7.3f} kappa={ag.kappa:.2f} "
              + " ".join(f"{names[j]}:{cp[j]:.2f}" for j in top))

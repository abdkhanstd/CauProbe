"""Dev-plant diagnostic: why does a correct structural belief lose to a wrong one?

Runs the reference, greedy and CauProbe on development plant 500001 (run 0)
and compares offload choices, rewards and the parameters of the particles the
posterior concentrated on. Development plants only.
"""
import collections

import numpy as np

import cp_e1_commission as E
from causlm.probe import mech
from causlm.probe.plants import sample_plant
from causlm.probe.run import AskCost, run_commissioning

seed = 500_001
spec = sample_plant(seed, (0, 0, 0, 1))
print("true:", {k: round(float(v), 2) for k, v in spec.net.items()})
res = {}
agents = {}
for m in ("oracle", "greedy", "cauprobe"):
    ag, z = E.make_agent(m, spec, f"eval:{seed}", seed * 10)
    out = run_commissioning(spec, ag, days=14, seed=seed * 10, expert_out=z,
                            ask=AskCost(), record_steps=True)
    res[m], agents[m] = out, ag
    st = out["_steps"]
    a = np.array([s["a"] for s in st])
    o = mech.NET_ACTIONS[a][:, 2]
    f = mech.NET_ACTIONS[a][:, 0]
    d = mech.NET_ACTIONS[a][:, 1]
    lat = np.array([s["lat"] for s in st])
    print(f"{m:9s} return {out['return']:9.1f} net {out['r_net']:9.1f} "
          f"offload mean {o.mean():.2f} f mean {f.mean():.2f} d mean {d.mean():.2f} "
          f"lat median {np.median(lat):.0f} p95 {np.percentile(lat, 95):.0f} "
          f"switches {out['n_switches']}")
    top = collections.Counter(map(tuple, mech.NET_ACTIONS[a].tolist())).most_common(3)
    print("          top actions", top)
for m in ("greedy", "cauprobe"):
    nb = agents[m].nb
    w = nb.w
    i = np.argsort(-w)[:3]
    print(m, "ESS %.1f" % nb.ess(), "top particles:")
    for j in i:
        print("   w=%.3f" % w[j], nb.combos[nb.combo_id[j]],
              {k: round(float(nb.P[k][j]), 2) for k in
               ("mu", "bg_base", "bg_amp", "k_occ", "p_base", "c_cpu", "rtt", "n_dev", "sigma")})

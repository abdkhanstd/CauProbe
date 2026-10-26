"""Lag-1 autocorrelation of the known-plant model's log-latency residuals.

Run on dev plants (seeds 810_000+): the reference controller knows every
parameter, so whatever autocorrelation its residuals show comes from what the
gateway cannot observe (occupancy-driven tenant load), not from estimation.
"""
import numpy as np

from cp_common import ROOT  # noqa: F401
from causlm.probe.baselines import OracleAgent
from causlm.probe.belief import expected_occ
from causlm.probe import mech
from causlm.probe.plants import ALL_COMBOS, sample_plant
from causlm.probe.run import run_commissioning

acf = []
for i in range(16):
    spec = sample_plant(810_000 + i, ALL_COMBOS[i])
    ag = OracleAgent(spec, seed=1)
    out = run_commissioning(spec, ag, days=2, seed=8_100_000 + i, record_steps=True)
    st = out["_steps"]
    P = {k: np.atleast_1d(np.asarray(v, float)) for k, v in spec.net.items()}
    res = []
    prev_pdr = 0.93
    for s_ in st:
        h = (s_["t"] % 288) / 12.0
        f, d, o = mech.NET_ACTIONS[s_["a"]]
        lat, _, _ = mech.net_outcomes(P, f, d, o, h, expected_occ(h), s_["pdr"])
        res.append(np.log(s_["lat"]) - np.log(float(lat[0])))
    r = np.asarray(res)
    r = r - r.mean()
    a1 = float((r[1:] * r[:-1]).sum() / (r * r).sum())
    acf.append(a1)
    print(spec.name, "lag-1 acf %.2f  resid sd %.2f" % (a1, r.std()))
print("median lag-1 autocorrelation %.2f" % np.median(acf))

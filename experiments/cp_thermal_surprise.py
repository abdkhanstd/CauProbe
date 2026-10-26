"""Would a thermal out-of-library detector have flagged BOPTEST?

Excess surprise of the thermal belief's one-step predictions (air temperature
and, where the controller is unknown, the controller's power response):
-log predictive density minus the predictive's entropy, per step. Recorded for
the greedy agent on (a) in-library dev plants of the simulated family and (b)
BOPTEST bestest_air. Only a diagnostic: it decides whether a thermal detector
is feasible, it does not tune one.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
from scipy.special import logsumexp
from scipy.stats import t as T4

from cp_common import CP_RESULTS
from causlm.probe import mech
from causlm.probe.belief import student_t_logpdf, expected_occ

H4 = float(T4(4).entropy())


def excess(tb, prev, obs):
    """Excess surprise of the mixture predictive for this transition."""
    ta_pred, _ = tb.predict_next(prev["t_air"], obs["t_out"], obs["hvac_kw"],
                                 obs["hour"])
    occ_sd = mech.office_gate(obs["hour"]) * 3.0
    sc = np.sqrt(tb.sigma ** 2 + (tb.P["occ_gain"] * occ_sd * mech.DT
                                  / tb.P["C_air"]) ** 2)
    w = tb.w
    lp = logsumexp(np.log(w + 1e-300) + student_t_logpdf(obs["t_air"] - ta_pred, sc))
    ex_t = -lp - float(w @ (np.log(sc) + H4))
    hv = mech.hvac_power(prev["t_air"], obs["setpoint"], tb.P)
    sh = 0.1 + 0.1 * hv
    lp2 = logsumexp(np.log(w + 1e-300) + student_t_logpdf(obs["hvac_kw"] - hv, sh))
    ex_h = -lp2 - float(w @ (np.log(sh) + H4))
    return ex_t, ex_h


def gateway(n=8, days=3):
    from causlm.probe.agent import CauProbeAgent
    from causlm.probe.baselines import greedy_cfg
    from causlm.probe.plants import ALL_COMBOS, GatewayPlant, sample_plant
    out = []
    for i in range(n):
        spec = sample_plant(820_000 + i, ALL_COMBOS[(5 * i) % 16])
        ag = CauProbeAgent(greedy_cfg(), seed=i)
        plant = GatewayPlant(spec, seed=8_200_000 + i)
        obs, ext, exh = None, [], []
        for t in range(days * 288):
            if t % 288 == 0:
                ag.set_forecast(plant.forecast_day)
            a, sp, _, _ = ag.act(obs)
            prev = obs
            obs, _, _ = plant.step(a, sp)
            if prev is not None and ag.tb.t_mass is not None and t > 288:
                e1, e2 = excess(ag.tb, prev, obs)
                ext.append(e1)
                exh.append(e2)
            ag.observe(obs)
        out.append({"plant": spec.name, "ex_t": float(np.mean(ext)),
                    "ex_h": float(np.mean(exh))})
        print(out[-1], flush=True)
    return out


if __name__ == "__main__":
    which = sys.argv[1]
    if which == "gateway":
        res = gateway()
    else:
        import cp_e6_boptest as E
        os.environ["CP_BOPTEST_PRIOR"] = "v3"
        res = []
        orig_update = None

        # wrap the thermal update to record excess surprise before each update
        from causlm.probe import belief as B
        rec = {"t": [], "h": []}
        _upd = B.ThermalBelief.update

        def upd(self, obs):
            if self._last is not None and self.t_mass is not None and \
                    "setpoint" in obs:
                e1, e2 = excess(self, self._last, obs)
                rec["t"].append(e1)
                rec["h"].append(e2)
            return _upd(self, obs)
        B.ThermalBelief.update = upd
        r = E.run(("greedy", 205, 3, 1000, None))
        res = {"ex_t": float(np.mean(rec["t"][288:])),
               "ex_h": float(np.mean(rec["h"][288:])), "reward": r["reward"]}
        print(res)
    (CP_RESULTS / f"thermal_surprise_{which}.json").write_text(json.dumps(res, indent=1))

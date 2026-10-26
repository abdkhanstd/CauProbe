"""How much does the 3-day rollout with geometric extrapolation misjudge the
free-learning horizon?

CauProbe-2 estimates tau_s, the days for which greedy operation alone keeps
paying the regret of not knowing structure s, from 3-day rollouts whose last
daily decay rate is extrapolated geometrically over the rest of the horizon.
Here, at every estimate CauProbe-2 makes in the first three days on 16
second-test plants (one per mechanism combination, nominal scenario), the
same quantity is also computed from 12-day rollouts (same model, same number
of truths), and the two are compared per live structure, weighted by the
structure's posterior mass. Analysis only: the extra rollouts draw from the
agent's random stream, so the runs are not the stored test runs. Writes
results/cauprobe/test2/tau_extrapolation.json.
"""
from __future__ import annotations

import json
from multiprocessing import get_context

import numpy as np

import cp_e1_commission as E
from cp_common import CP_RESULTS, TEST2_PLANTS
from causlm.probe import mech
from causlm.probe.agent import CauProbeAgent
from causlm.probe.plants import sample_plant
from causlm.probe.run import AskCost, run_commissioning

MAX_CALLS = 6


def job(a):
    seed, combo = a
    mech.set_switch_costs(0.3, 0.05)
    spec = sample_plant(seed, tuple(combo))
    ag, z = E.make_agent("cauprobe-v2", spec, f"eval:{seed}", seed * 10)
    rec = []
    orig = CauProbeAgent._free_horizon

    def wrapped(self, w, live, beta, Mw, U, PV, h_eff, hour, R=None):
        t3 = orig(self, w, live, beta, Mw, U, PV, h_eff, hour, R=R)
        if len(rec) < MAX_CALLS and R is None:
            d0 = self.cfg.fl_days
            self.cfg.fl_days = 12
            t12 = orig(self, w, live, beta, Mw, U, PV, h_eff, hour, R=R)
            self.cfg.fl_days = d0
            rec.append({"t": int(self.t), "h_eff": float(h_eff), "beta": beta.tolist(),
                        "tau3": t3.tolist(), "tau12": t12.tolist()})
        return t3

    ag._free_horizon = wrapped.__get__(ag, CauProbeAgent)
    run_commissioning(spec, ag, days=3, seed=seed * 10, expert_out=z, ask=AskCost())
    return {"seed": seed, "combo": list(combo), "calls": rec}


def main():
    plants = TEST2_PLANTS[:16]
    with get_context("fork").Pool(8) as pool:
        res = pool.map(job, plants)
    rel, absd, wsum = [], [], []
    for r in res:
        for c in r["calls"]:
            b = np.array(c["beta"])
            t3, t12 = np.array(c["tau3"]), np.array(c["tau12"])
            m = b >= 1e-3
            for bi, a3, a12 in zip(b[m], t3[m], t12[m]):
                rel.append((a3 - a12) / max(a12, 1e-6))
                absd.append(a3 - a12)
                wsum.append(bi)
    rel, absd, w = np.array(rel), np.array(absd), np.array(wsum)
    # mass-weighted mean horizon under each estimate, per call
    ratio = []
    for r in res:
        for c in r["calls"]:
            b = np.array(c["beta"])
            ratio.append(float(b @ np.array(c["tau3"])) / max(float(b @ np.array(c["tau12"])), 1e-9))
    out = {"n_calls": int(sum(len(r["calls"]) for r in res)), "n_struct": int(len(rel)),
           "median_rel_err": float(np.median(rel)),
           "mean_abs_err_days": float(np.mean(np.abs(absd))),
           "weighted_mean_rel_err": float(np.sum(w * rel) / np.sum(w)),
           "share_within_25pct": float(np.mean(np.abs(rel) <= 0.25)),
           "mass_weighted_tau_ratio_median": float(np.median(ratio)),
           "mass_weighted_tau_ratio_iqr": (float(np.percentile(ratio, 25)),
                                           float(np.percentile(ratio, 75))),
           "rows": res}
    (CP_RESULTS / "test2" / "tau_extrapolation.json").write_text(json.dumps(out, indent=1))
    print({k: v for k, v in out.items() if k != "rows"})


if __name__ == "__main__":
    main()

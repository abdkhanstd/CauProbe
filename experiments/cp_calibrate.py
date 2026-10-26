"""Sanity checks on the plant family before any agent is run.

For every mechanism combination, which network action is optimal at each time
of day, how much is lost by using the optimum of a *different* mechanism, and
which thermal schedule is optimal for heavy versus light mass. If the optimal
actions did not differ between mechanisms, nothing in this paper would be
decision-relevant; this script is the check that they do, and by how much.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from causlm.config import RESULTS
from causlm.probe import mech
from causlm.probe.belief import NetBelief, ThermalBelief, expected_occ
from causlm.probe.plants import sample_net_params, sample_th_params, FALLBACK_NET
from causlm.probe.agent import CauProbeAgent, ProbeConfig, CTX_HOURS


def main():
    rng = np.random.default_rng(0)
    agent = CauProbeAgent(ProbeConfig(), seed=0)
    out = {"net": {}, "thermal": {}}
    best = {}
    tables = {}
    for combo in [(a, b, c) for a in (0, 1) for b in (0, 1) for c in (0, 1)]:
        P = sample_net_params(rng, *combo, n=64)
        agent.nb = NetBelief.from_params({k: np.median(v) for k, v in P.items()})
        Us = []
        for h in CTX_HOURS:
            R, pv, mu, s = agent._net_tables(h, 0.93)
            R = np.where(pv[:, 0] <= 0.02, R[:, 0], -np.inf)
            Us.append(R)
        U = np.stack(Us, 1)                                    # (A, K)
        tables[combo] = U
        best[combo] = U.argmax(0)
        out["net"][str(combo)] = {
            "best_actions_by_ctx": [mech.NET_ACTIONS[i].tolist()
                                    for i in best[combo]],
            "best_value_per_day": float(24 * U.max(0).sum()),
            "fallback_value_per_day": float(24 * U[FALLBACK_NET].sum()),
        }
    # cross-mechanism loss: value of acting on combo' optimum in combo
    combos = list(tables)
    loss = np.zeros((len(combos), len(combos)))
    for i, c in enumerate(combos):
        U = tables[c]
        for j, c2 in enumerate(combos):
            pick = best[c2]
            v = np.array([U[pick[k], k] for k in range(U.shape[1])])
            v = np.where(np.isfinite(v), v, U.min(0) - 4.0)
            loss[i, j] = 24 * (U.max(0).sum() - v.sum())
    out["net_cross_loss_per_day"] = {"combos": [list(c) for c in combos],
                                    "loss": loss.round(2).tolist()}
    # thermal: optimal schedule per mass class
    fc = 27.0 + 5.5 * np.sin(2 * np.pi * (np.arange(288) / 12.0 - 9.0) / 24.0)
    Uth = {}
    for D in (0, 1):
        P = sample_th_params(rng, D, n=64)
        tb = ThermalBelief.from_params({k: np.median(v) for k, v in P.items()})
        tb.t_mass = np.full(1, 26.0)
        # two days so the slab state is not an artefact of the initial guess
        U1 = tb.simulate_day(mech.SCHED_SP, fc, 26.0)[:, 0]
        Uth[D] = U1
        out["thermal"][f"D{D}"] = {
            "best": mech.SCHEDULES[int(np.argmax(U1))],
            "values": U1.round(2).tolist()}
    out["thermal"]["cross_loss"] = {
        "heavy_using_light_best": float(Uth[0].max() - Uth[0][int(np.argmax(Uth[1]))]),
        "light_using_heavy_best": float(Uth[1].max() - Uth[1][int(np.argmax(Uth[0]))]),
    }
    p = RESULTS / "cp_calibration.json"
    p.write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "net"}, indent=1))
    for c, d in out["net"].items():
        print(c, "best/day %.2f fallback/day %.2f" % (d["best_value_per_day"],
                                                     d["fallback_value_per_day"]))
        print("   ", d["best_actions_by_ctx"])


if __name__ == "__main__":
    main()

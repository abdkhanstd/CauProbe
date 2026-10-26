"""What predicts whether a probe pays? An extended paired audit.

As in cp_audit_probes.py, at a probe decision of CauProbe-2 the agent and the
plant are copied, the probe is carried out in one copy and the greedy action
taken in the other, and both continue under the same policy to the end of the
14-day run; the paired difference in return is the decision's realised value.
In addition, the decision state is recorded -- EVPI, the posterior mass of the
leading structure, the entropy of the structure posterior, how undecided the
processing-path factor (B) is, the effective horizon, kappa, the day, the probe
length, which knob the probe changes -- and the probe is also valued
*directly* by the rollout comparator (approximate dual control: the simulated
return of taking it and continuing greedily with Bayesian updating, minus that
of continuing greedily now), at the same state.

32 second-test plants (two per mechanism combination), Wi-Fi and LPWAN
reconfiguration costs, the first probe decision and up to two more drawn from
the first seven days. Writes results/cauprobe/test2/probe_audit2.jsonl.
"""
from __future__ import annotations

import argparse
import copy
import json
from multiprocessing import get_context

import numpy as np

import cp_e1_commission as E
from cp_common import CP_RESULTS, TEST2_PLANTS
from causlm.probe import mech
from causlm.probe.plants import GatewayPlant, sample_plant
from causlm.probe.run import AskCost, STEPS_PER_DAY
from cp_audit_probes import DAYS, SWITCH, rollout, step

OUT = CP_RESULTS / "test2" / "probe_audit2.jsonl"


def state_features(ag, info):
    fp = ag.nb.factor_marginals()
    w = ag.nb.w
    cid = ag.nb.combo_id
    beta = np.bincount(cid, weights=w, minlength=8)
    beta = beta / beta.sum()
    ent = float(-(beta[beta > 0] * np.log(beta[beta > 0])).sum())
    return {"evpi": float(info.get("evpi", np.nan)), "max_beta": float(beta.max()),
            "entropy": ent, "b_undecided": float(0.5 - abs(fp["B"] - 0.5)),
            "a_undecided": float(0.5 - abs(fp["A"] - 0.5)),
            "c_undecided": float(0.5 - abs(fp["C"] - 0.5)),
            "h_eff": float(getattr(ag, "_h_eff", np.nan)), "kappa": float(ag.kappa)}


def job(a):
    seed, combo, sw = a
    mech.set_switch_costs(*SWITCH[sw])
    spec = sample_plant(seed, tuple(combo))
    agent, z = E.make_agent("cauprobe-v2", spec, f"eval:{seed}", seed * 10)
    plant = GatewayPlant(spec, seed=seed * 10)
    ask = AskCost(energy_j=40.0, latency_ms=6000.0, price_per_kj=0.4)
    agent.cfg.ask_price = ask.price
    agent.offer_text()
    rng = np.random.default_rng(seed + 91)
    a0, p0 = copy.deepcopy(agent), copy.deepcopy(plant)
    obs, dec = None, []
    for t in range(DAYS * STEPS_PER_DAY):
        held = agent._commit is not None and t < agent._commit[1]
        obs, r, info = step(agent, plant, obs, t, z, ask)
        if info.get("probe") and not held:
            dec.append(t)
    early = [t for t in dec if t < 7 * STEPS_PER_DAY]
    pick = sorted(set(early[:1] + (list(rng.choice(early[1:], size=min(2, len(early) - 1),
                                                    replace=False)) if len(early) > 1 else [])))
    out = []
    agent, plant, obs = a0, p0, None
    for t in range(DAYS * STEPS_PER_DAY):
        if t in pick:
            A, PA = copy.deepcopy(agent), copy.deepcopy(plant)
            B, PB = copy.deepcopy(agent), copy.deepcopy(plant)
            Rd = copy.deepcopy(agent)
            if t % STEPS_PER_DAY == 0:
                A.set_forecast(PA.forecast_day)
            prev = int(obs["net_idx"]) if obs is not None else None
            aA, spA, askA, infoA = A.act(obs)
            m = int(A._commit[1] - t) if A._commit is not None else 1
            feats = state_features(A, infoA)
            pred_net = float(getattr(A, "_last_net", np.nan))
            # direct (rollout) value of the same probe at the same state
            Rd.cfg.objective = "rollout"
            Rd.cfg.free_learning = False
            Rd._rd_probe = (aA, m)
            if t % STEPS_PER_DAY == 0:
                Rd.set_forecast(PA.forecast_day)
            Rd.act(obs)
            direct = float(getattr(Rd, "_rd_probe_value", np.nan))
            # the knob the probe moves relative to where the gateway was
            knob = None
            if prev is not None:
                f0, d0, o0 = mech.NET_ACTIONS[prev]
                f1, d1, o1 = mech.NET_ACTIONS[aA]
                knob = "".join(x for x, ch in (("f", f0 != f1), ("d", d0 != d1), ("o", o0 != o1)) if ch)
            busy = ask.latency_ms if askA else 0.0
            oA, rA, _ = PA.step(aA, spA, busy)
            if askA:
                rA -= ask.price
                A.receive_expert(z)
            A.observe(oA, busy / (300.0 * 1000.0))
            retA = rA + rollout(A, PA, oA, t + 1, z, ask, False)
            retB = rollout(B, PB, obs, t, z, ask, True)
            out.append({"seed": seed, "combo": list(combo), "switch": sw, "t": t,
                        "day": t // STEPS_PER_DAY, "m": m, "knob": knob,
                        "pred_net": pred_net, "direct": direct,
                        "realised": float(retA - retB), **feats})
        obs, r, info = step(agent, plant, obs, t, z, ask)
        if t >= max(pick, default=-1):
            break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=24)
    a = ap.parse_args()
    plants = sorted(TEST2_PLANTS, key=lambda p: (p[0] - 910_000) // 16)[:32]
    jobs = [(s, c, sw) for sw in SWITCH for s, c in plants]
    with get_context("fork").Pool(a.workers, maxtasksperchild=2) as pool, \
            open(OUT, "w") as f:
        for i, res in enumerate(pool.imap_unordered(job, jobs)):
            for r in res:
                f.write(json.dumps(r) + "\n")
            f.flush()
            print(f"{i + 1}/{len(jobs)}", flush=True)


if __name__ == "__main__":
    main()

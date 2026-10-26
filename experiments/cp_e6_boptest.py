"""E6: the thermal side of CauProbe on BOPTEST bestest_air (Modelica, in-process).

bestest_air is BESTEST Case 900 -- concrete-block walls and slab, i.e. heavy --
a 48 m2 two-person office in Denver with a fan-coil unit. Nothing about it is in
the agent's library: the zone, the fan coil and its PI controller are Modelica,
and the agent plans with a two-node RC grey box whose prior is set only from
the floor area (a 48 m2 office), with both a heavy and a light class.

Methods (all cooling-setpoint schedules, heating left to BOPTEST's baseline):
  boptest-baseline   BOPTEST's own controller (no overwrite)
  incumbent          24 C occupied / 28 C setback, no pre-cooling
  nominal            best template for a fixed mid-range grey box
  greedy             Bayes template on the learned grey-box belief
  thompson           template optimal for one posterior sample per day
  cauprobe           Bayes template + night step tests valued by VOI
  cauprobe+expert    as cauprobe, with the expert's reading of site notes

Metrics: BOPTEST's own KPIs (tdis_tot in K h per zone, ener_tot in kWh/m2,
cost_tot under its 'dynamic' price scenario) and the gateway's time-of-use cost.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from multiprocessing import get_context

import numpy as np

from cp_common import CP_RESULTS, EXPERT_DIR
from causlm.probe import mech
from causlm.probe.agent import CauProbeAgent, ProbeConfig
from causlm.probe.baselines import ThompsonAgent, greedy_cfg
from causlm.probe.belief import ThermalBelief
from causlm.probe.boptest_plant import BOPTEST_ROOT, _cwd, load_testcase
from causlm.probe.mech import DT, STEPS_PER_DAY

W_TDIS = 1.0          # reward units per K h of discomfort (occupied)
W_E = mech.W.w_energy

# Site prior for a 48 m2 office: capacities and conductances scaled from area,
# wide enough to contain both a lightweight and a heavyweight construction.
PER_CLASS = {0: {"C_mass": (5e6, 4e7), "R_mass": (5e-4, 5e-3)},
             1: {"C_mass": (3e5, 3e6), "R_mass": (5e-3, 5e-2)}}
COMMON = {"C_air": (2e5, 2e6), "R_out": (5e-3, 3e-2), "solar": (300.0, 3000.0),
          "occ_gain": (100.0, 400.0), "hvac_max": (0.3, 3.0), "cop": (2.0, 5.0),
          "kp": (0.2, 3.0)}
# v2 prior: the zone controller's tuning is unknown too, so its gain and dead
# band are learned (a PI fan coil looks like a high-gain, zero-dead-band P loop)
if os.environ.get("CP_BOPTEST_PRIOR", "v2") in ("v2", "v3", "v4"):
    COMMON = dict(COMMON, kp=(0.2, 10.0), db=(0.0, 0.6))

# Templates suited to BOPTEST's comfort band (21-24 C occupied).
B_SCHEDULES = []
for s_occ in (23.0, 23.5, 24.0):
    for pre in ("none", "night21", "night20", "morning"):
        for relax in (0.0, 0.5):
            for early in (False, True):
                B_SCHEDULES.append({"s_occ": s_occ, "pre": pre, "relax": relax,
                                    "early": early})


def b_setpoints(sc):
    h = np.arange(STEPS_PER_DAY) / 12.0
    sp = np.full(STEPS_PER_DAY, 28.0)
    occ = (h >= 7.0) & (h < 19.0)
    sp[occ] = sc["s_occ"]
    sp[(h >= 14.0) & (h < 19.0)] = sc["s_occ"] + sc["relax"]
    if sc["pre"] == "night21":
        sp[(h >= 3.0) & (h < 7.0)] = 21.0
    elif sc["pre"] == "night20":
        sp[(h >= 3.0) & (h < 7.0)] = 20.0
    elif sc["pre"] == "morning":
        sp[(h >= 7.0) & (h < 14.0)] = sc["s_occ"] - 1.5
    if sc["early"]:
        sp[h < 3.0] = 20.0
    return sp


B_SP = np.stack([b_setpoints(s) for s in B_SCHEDULES])
INCUMBENT = B_SCHEDULES.index({"s_occ": 24.0, "pre": "none", "relax": 0.0,
                               "early": False})


def occupied(hour):
    h = np.asarray(hour, float)
    return ((h >= 8.0) & (h < 18.0)).astype(float)


def reward_fn(hv, hour, ta, occ):
    dt_h = DT / 3600.0
    dis = occupied(hour) * (np.maximum(ta - 24.0, 0) + np.maximum(21.0 - ta, 0))
    return -(W_E * mech.tariff(hour) * hv * dt_h + W_TDIS * dis * dt_h)


def _payload(r):
    """BOPTEST's TestCase methods return (status, message, payload)."""
    if isinstance(r, tuple):
        return r[2]
    return r["payload"] if isinstance(r, dict) and "payload" in r else r


def make_agent(method, seed, expert_z=None):
    tb_kw = dict(per_combo=256, eta=0.5, hazard=2e-5, seed=seed + 1, occ_prior=2.0,
                 ranges=(PER_CLASS, COMMON), reward_fn=reward_fn)
    if method == "thompson":
        ag = ThompsonAgent(seed=seed)
    elif method == "greedy":
        ag = CauProbeAgent(greedy_cfg(), seed=seed, name="greedy")
    elif method.startswith("cauprobe"):
        ex = None
        if method.endswith("+expert"):
            from causlm.probe.expert import ExpertModel
            ex = ExpertModel.from_json(EXPERT_DIR / "slm-0.5b-lora.json")
        # "cauprobe-v2": the second iteration, frozen settings as in
        # cp_e1_commission.py (on BOPTEST only its thermal side acts: the
        # night step test is valued net of free learning)
        kw = ({"free_learning": True, "reopen_rate": True, "probe_return": True,
               "fl_tail": "geometric", "fl_days": 3} if method.startswith("cauprobe-v2") else {})
        ag = CauProbeAgent(ProbeConfig(use_ask=False, **kw), seed=seed, expert=ex,
                           name=method)
    elif method == "nominal":
        ag = CauProbeAgent(greedy_cfg(), seed=seed, name="nominal")
    else:
        return None
    ag.tb = ThermalBelief(**tb_kw)
    if method == "nominal":
        # a fixed mid-range model, never revised
        P = {k: np.atleast_1d(np.median(v)) for k, v in ag.tb.P.items()}
        P["D"] = np.atleast_1d(0.0)
        ag.tb = ThermalBelief.from_params(P, occ_prior=2.0, reward_fn=reward_fn)
        ag.tb.eta = 0.0
    ag.sched_sp = B_SP
    ag.schedules = B_SCHEDULES
    # v4: the thermal out-of-library detector, threshold calibrated on
    # in-library dev plants only, hands control to the incumbent schedule
    if os.environ.get("CP_BOPTEST_PRIOR", "v2") == "v4":
        ag.cfg.thermal_detect = True
        ag.cfg.th_h = float(os.environ.get("CP_TH_H", "20.0"))
        ag.fallback_sched = INCUMBENT
    # v3: the controller's logged response enters the likelihood
    ag.tb.fit_controller = os.environ.get("CP_BOPTEST_PRIOR", "v2") in ("v3", "v4")
    if method.endswith("+expert") and expert_z is not None:
        z = expert_z["D"]
        ag.tb.logw = ag.tb.logw + ag.expert.loglik("D", z, ag.tb.factor_values())
        ag.tb.logw -= np.logaddexp.reduce(ag.tb.logw)
    return ag


def run(args):
    method, start_day, days, seed, expert_z = args
    t0 = time.time()
    import logging
    logging.disable(logging.INFO)
    tc = load_testcase("bestest_air")
    with _cwd(BOPTEST_ROOT):
        sc = tc.set_scenario({"electricity_price": "dynamic", "time_period": None,
                              "temperature_uncertainty": None,
                              "solar_uncertainty": None, "seed": None})
        if isinstance(sc, tuple) and sc[0] != 200:
            raise RuntimeError(f"BOPTEST rejected the scenario: {sc[1]}")
        tc.set_step(DT)
        tc.initialize(start_day * 86400.0, 86400.0)
    ag = make_agent(method, seed, expert_z)
    rec = {"t_air": [], "p_kw": [], "hour": [], "sp": [], "t_out": []}
    obs = None
    sched_log = []
    for k in range(days * STEPS_PER_DAY):
        h = (k % STEPS_PER_DAY) / 12.0
        if k % STEPS_PER_DAY == 0 and ag is not None:
            with _cwd(BOPTEST_ROOT):
                fc = tc.get_forecast(["TDryBul"], horizon=86400 - DT, interval=DT)
            fc = np.asarray(_payload(fc)["TDryBul"], float) - 273.15
            ag.set_forecast(fc[:STEPS_PER_DAY])
            if obs is not None:
                ag.last_obs = obs
            if isinstance(ag, ThompsonAgent):
                ag._plan_day_ts(obs)
            else:
                ag._plan_day(obs)
            sched_log.append(B_SCHEDULES[ag.sched])
        if method == "boptest-baseline":
            u = {}
        elif method == "incumbent":
            u = {"con_oveTSetCoo_u": B_SP[INCUMBENT][k % STEPS_PER_DAY] + 273.15,
                 "con_oveTSetCoo_activate": 1}
        else:
            sp = float(ag.day_sp[k % STEPS_PER_DAY])
            u = {"con_oveTSetCoo_u": sp + 273.15, "con_oveTSetCoo_activate": 1}
        with _cwd(BOPTEST_ROOT):
            y = tc.advance(u)
        y = _payload(y)
        t_air = float(y["zon_reaTRooAir_y"]) - 273.15
        p_kw = (float(y["fcu_reaPCoo_y"]) + float(y["fcu_reaPFan_y"])) / 1000.0
        t_out = float(y["zon_weaSta_reaWeaTDryBul_y"]) - 273.15
        sp_now = (float(ag.day_sp[k % STEPS_PER_DAY]) if ag is not None
                  else float(B_SP[INCUMBENT][k % STEPS_PER_DAY]))
        obs = {"t_air": t_air, "hvac_kw": p_kw, "t_out": t_out, "hour": h,
               "setpoint": sp_now}
        if ag is not None:
            ag._thermal_watch(obs)
            ag.tb.update(obs)
            ag.last_obs = obs
        for key, v in (("t_air", t_air), ("p_kw", p_kw), ("hour", h),
                       ("t_out", t_out)):
            rec[key].append(v)
    with _cwd(BOPTEST_ROOT):
        kp = tc.get_kpis()
    kp = _payload(kp)
    ta, pk, hh = map(np.asarray, (rec["t_air"], rec["p_kw"], rec["hour"]))
    dt_h = DT / 3600.0
    tou = float(np.sum(mech.tariff(hh) * pk * dt_h))
    tdis = float(np.sum(occupied(hh) * (np.maximum(ta - 24, 0)
                                        + np.maximum(21 - ta, 0)) * dt_h))
    out = {"method": method, "start_day": start_day, "days": days, "seed": seed,
           "kpis": kp, "tou_kwh_weighted": tou, "tdis_kh": tdis,
           "energy_kwh": float(pk.sum() * dt_h),
           "reward": float(-(W_E * tou + W_TDIS * tdis)),
           "th_probes": int(getattr(ag, "n_th_probes", 0)) if ag else 0,
           "schedules": sched_log,
           "prior": os.environ.get("CP_BOPTEST_PRIOR", "v2"),
           "th_alarms": int(getattr(ag, "n_th_alarms", 0)) if ag else 0,
           "th_cusum_max": float(getattr(ag, "th_cusum_max", 0.0)) if ag else 0.0,
           "mass_post_heavy": float(ag.tb.mass_post()) if ag else None,
           "wall_s": time.time() - t0}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=7)
    ap.add_argument("--starts", default="160,175,190,205,220,235")
    ap.add_argument("--methods", default="boptest-baseline,incumbent,nominal,"
                    "greedy,thompson,cauprobe,cauprobe+expert")
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--tag", default="main")
    a = ap.parse_args()
    # what the trained expert says about a site described like bestest_air
    ez = None
    p = CP_RESULTS / "expert_outputs_boptest.json"
    if p.exists():
        ez = json.loads(p.read_text())["z"]
    jobs = [(m, int(s), a.days, 1000 + i, ez)
            for i, s in enumerate(a.starts.split(",")) for m in a.methods.split(",")]
    out_p = CP_RESULTS / f"e6_boptest_{a.tag}.jsonl"
    # one job per process: PyFMI result objects must not outlive their run
    with get_context("spawn").Pool(a.workers, maxtasksperchild=1) as pool,             open(out_p, "a") as f:
        for r in pool.imap_unordered(run, jobs):
            f.write(json.dumps(r) + "\n")
            f.flush()
            print(r["method"], r["start_day"], "reward %.2f tdis %.2f tou %.2f probes %d heavy %s (%.0fs)"
                  % (r["reward"], r["tdis_kh"], r["tou_kwh_weighted"], r["th_probes"],
                     r["mass_post_heavy"], r["wall_s"]), flush=True)


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main()

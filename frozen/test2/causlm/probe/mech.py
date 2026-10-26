"""Mechanism equations of the gateway plant family.

Every function here is vectorised over a leading particle axis, and the same
functions drive both the true plant (one parameter vector) and the agent's
hypothesis particles (thousands of them). Sharing them is deliberate: an
in-library plant is then exactly representable by some particle, and the
misspecification experiments break that link explicitly (out-of-library
mechanisms in `plants.py`, and the BOPTEST Modelica plant).

Mechanism factors
-----------------
A  radio MAC          0 aggregating/contention: a lower duty cycle batches
                        frames, so it removes broker load and adds no delay
                      1 sleep MAC / low-power listening: a lower duty cycle adds
                        a rendezvous delay ~ T_w (1/d - 1) / 2 per frame and
                        leaves the offered load unchanged
B  processing         0 edge-bound: local processing saturates with load, the
                        cloud round trip is short, so offloading relieves latency
                      1 WAN-bound: local processing is cheap, the backhaul is
                        cellular/long-haul, so offloading adds 150-300 ms
C  broker load        0 self-load: many own devices on a dedicated broker, so the
                        publish rate dominates the queue
                      1 tenant-dominated: few own devices on a shared broker, so
                        the publish rate barely moves latency and should be set
                        for freshness instead
D  thermal mass       0 heavy (exposed slab): pre-cooling shifts load off-peak
                      1 light (timber / lightweight partitions): it does not

Each factor flips the sign or the materiality of one knob's effect on the
reward, which is what makes it decision-relevant rather than merely structural.
"""
from __future__ import annotations

import itertools
from functools import lru_cache
from typing import Dict

import numpy as np

DT = 300.0                  # control step, s
STEPS_PER_DAY = 288

# ----------------------------------------------------------------- actions
F_LEVELS = np.array([0.1, 0.25, 0.5, 1.0])         # publish rate, Hz
D_LEVELS = np.array([0.25, 0.5, 1.0])              # radio duty cycle
O_LEVELS = np.array([0.0, 0.5, 1.0])               # offload fraction
NET_ACTIONS = np.array(list(itertools.product(F_LEVELS, D_LEVELS, O_LEVELS)))
N_NET = len(NET_ACTIONS)                           # 36 joint network actions


@lru_cache(maxsize=None)
def net_action_index(f: float, d: float, o: float) -> int:
    hit = np.flatnonzero((np.abs(NET_ACTIONS[:, 0] - f) < 1e-9)
                         & (np.abs(NET_ACTIONS[:, 1] - d) < 1e-9)
                         & (np.abs(NET_ACTIONS[:, 2] - o) < 1e-9))
    return int(hit[0])


# ----------------------------------------------------------------- reward
class W:
    """Reward weights, shared by the plant, the agent and every baseline."""
    lat_sla_ms = 150.0
    lat_cap_ms = 500.0
    w_lat = 0.003            # per ms above SLA, saturating at lat_cap_ms
    w_aoi = 0.12             # per second of age of information
    w_pdr = 2.0              # per unit of packet loss
    w_it = 0.4               # per kJ of gateway IT energy
    w_energy = 0.25          # per tariff-weighted kWh of zone electricity
    w_comfort = 0.55         # per unit |PMV| above the comfort band, occupied
    comfort_band = 0.5
    comfort_occ_full = 3.0
    # hard constraints ("violations"), counted separately from the reward
    lat_hard_ms = 800.0      # alarm-delivery deadline
    pdr_hard = 0.5
    t_hard_hi = 28.0         # occupied-zone temperature band
    t_hard_lo = 18.0
    w_violation = 4.0        # per violating step, charged in the reward too
    # Reconfiguring the fleet is not free: a new publish rate or duty cycle
    # has to be pushed to every device over the downlink (and sleepy devices
    # only pick it up at their next wake-up). Changing the offload split is a
    # local routing change and nearly free.
    w_switch_fd = 0.3
    w_switch_o = 0.05


def set_switch_costs(fd: float, o: float):
    """Scenario hook: reconfiguration prices (e.g. an LPWAN fleet, where every
    downlink is airtime-limited, pays far more than a Wi-Fi fleet)."""
    W.w_switch_fd = float(fd)
    W.w_switch_o = float(o)


def switch_cost(a_prev, a):
    """Reconfiguration cost of moving the fleet from action a_prev to a."""
    if a_prev is None:
        return 0.0
    p, q = NET_ACTIONS[a_prev], NET_ACTIONS[a]
    c = 0.0
    if p[0] != q[0] or p[1] != q[1]:
        c += W.w_switch_fd
    if p[2] != q[2]:
        c += W.w_switch_o
    return c


def switch_cost_vec(a_prev):
    """Switch cost from a_prev to every joint action, (N_NET,)."""
    if a_prev is None:
        return np.zeros(N_NET)
    p = NET_ACTIONS[a_prev]
    fd = (NET_ACTIONS[:, 0] != p[0]) | (NET_ACTIONS[:, 1] != p[1])
    o = NET_ACTIONS[:, 2] != p[2]
    return W.w_switch_fd * fd + W.w_switch_o * o


def tariff(hour):
    h = np.asarray(hour, float)
    return np.where((h >= 14.0) & (h < 20.0), 2.4,
                    np.where((h >= 7.0) & (h < 14.0), 1.0, 0.55))


def day_shape(hour):
    h = np.asarray(hour, float)
    return np.maximum(0.0, np.sin(np.pi * (h - 6.0) / 14.0))


def office_gate(hour):
    h = np.asarray(hour, float)
    on = ((h > 7.5) & (h < 19.0)).astype(float)
    return on * (0.65 + 0.35 * np.sin(np.pi * np.clip((h - 7.5) / 11.5, 0, 1)))


# ----------------------------------------------------------------- network
NET_KEYS = ("A", "B", "C", "n_dev", "mu", "bg_base", "bg_amp", "k_occ",
            "p_base", "c_cpu", "rtt", "t_wake", "s_frame", "sigma")


def own_load(P: Dict[str, np.ndarray], f, d):
    A = P["A"]
    mult = np.where(A == 0, 0.35 + 0.65 * d, 1.0)
    return P["n_dev"] * f * mult


def bg_load(P, hour, occ):
    return P["bg_base"] + P["bg_amp"] * day_shape(hour) + P["k_occ"] * occ


def net_outcomes(P: Dict[str, np.ndarray], f, d, o, hour, occ, pdr,
                 busy: float = 0.0, burst: float = 1.0):
    """Mean latency (ms), gateway IT energy (J/step) and age of information (s).

    `f, d, o` broadcast against the particle axis, so one call can score every
    particle under one action or one particle under every action.
    """
    pdr = np.clip(pdr, 0.05, 1.0)
    lam = (own_load(P, f, d) + burst * bg_load(P, hour, occ)) / pdr
    mu = P["mu"] * (1.0 - busy)
    rho = lam / mu
    wait = 1000.0 / (mu * np.maximum(1.0 - rho, 0.03))
    local = P["p_base"] / np.maximum(1.0 - (1.0 - o) * lam / P["c_cpu"], 0.08)
    proc = (1.0 - o) * local + o * P["rtt"]
    A = P["A"]
    sleep = np.where(A == 1, P["t_wake"] * (1.0 / d - 1.0) / 2.0,
                     np.where(A == 2, P["s_frame"] / d, 0.0))
    lat = wait + proc + sleep
    mult = np.where(A == 0, 0.35 + 0.65 * d, 1.0)
    p_radio = np.where(A == 0, 0.25 + 0.9 * f * (P["n_dev"] / 24.0) * mult,
                       0.25 + 1.2 * d + 0.6 * f * P["n_dev"] / 24.0)
    p_cpu = 1.8 + 1.4 * (1.0 - o) * lam / 40.0 + 0.35 * o * lam / 40.0
    it_j = DT * (p_radio + p_cpu)
    aoi = 1.0 / (2.0 * f * pdr) + lat / 1000.0
    return lat, it_j, aoi


def net_reward(lat, it_j, aoi, pdr):
    w = W
    lat_pen = w.w_lat * np.clip(lat - w.lat_sla_ms, 0.0, w.lat_cap_ms)
    # The hard constraint is the alarm-delivery deadline. Delivery-ratio dips
    # are charged in the reward but are not a safety violation: they come
    # from people shadowing the radio, which no gateway knob controls.
    viol = lat > w.lat_hard_ms
    r = -(lat_pen + w.w_aoi * aoi + w.w_pdr * (1.0 - pdr)
          + w.w_it * it_j / 1000.0 + w.w_violation * viol)
    return r, viol


def pdr_from_rssi(rssi):
    return np.clip(1.0 / (1.0 + np.exp(-0.22 * (rssi + 80.0))), 0.02, 0.999)


# ----------------------------------------------------------------- thermal
TH_KEYS = ("D", "C_air", "C_mass", "R_out", "R_mass", "solar", "occ_gain",
           "hvac_max", "cop", "kp")


def solar_gain(P, hour):
    h = np.asarray(hour, float)
    return P["solar"] * np.maximum(0.0, np.sin(np.pi * (h - 6.5) / 12.0))


def hvac_power(t_air, setpoint, P):
    """Cooling-only proportional zone controller (summer operation).

    Returns electrical kW (>= 0). The thermal power removed is cop * kW.
    """
    err = t_air - setpoint
    p = np.minimum(np.maximum(err, 0.0) * P["kp"], P["hvac_max"])
    # dead band of the zone controller: 0.4 K in the simulated plants; a
    # learned parameter where the controller's tuning is unknown (BOPTEST)
    db = P["db"] if "db" in P else 0.4
    return np.where(err >= db, p, 0.0)


def thermal_step(P, t_air, t_mass, t_out, hvac_kw, occ, hour, substeps: int = 2):
    """One 5-minute step of the 2R2C zone, explicit Euler in `substeps`."""
    dt = DT / substeps
    q_free = P["occ_gain"] * occ + solar_gain(P, hour)
    q_hvac = -P["cop"] * hvac_kw * 1000.0
    for _ in range(substeps):
        d_air = ((t_out - t_air) / P["R_out"] + (t_mass - t_air) / P["R_mass"]
                 + q_free + q_hvac) / P["C_air"] * dt
        d_mass = ((t_air - t_mass) / P["R_mass"]) / P["C_mass"] * dt
        t_air = np.clip(t_air + d_air, 5.0, 45.0)
        t_mass = np.clip(t_mass + d_mass, 5.0, 45.0)
    return t_air, t_mass


_PMV_POLY = None


def pmv_fast(t_air):
    """Cubic fit of ISO 7730 PMV in air temperature (rh 45%, 0.12 m/s,
    1.2 met, 0.6 clo). Used by the agent's planner; the plant itself evaluates
    the full fixed-point PMV with its actual humidity."""
    global _PMV_POLY
    if _PMV_POLY is None:
        from ..env.comfort import pmv_ppd
        ts = np.linspace(14, 36, 89)
        pm = np.array([pmv_ppd(t, rh=45.0, vel=0.12, met=1.2, clo=0.6)[0]
                       for t in ts])
        _PMV_POLY = np.polyfit(ts, pm, 3)
    return np.clip(np.polyval(_PMV_POLY, t_air), -3.5, 3.5)


def thermal_reward(e_kwh, hour, pmv, occ, t_air):
    w = W
    occ_w = np.minimum(1.0, occ / w.comfort_occ_full)
    comfort = occ_w * np.maximum(0.0, np.abs(pmv) - w.comfort_band)
    viol = (occ >= 1.0) & ((t_air > w.t_hard_hi) | (t_air < w.t_hard_lo))
    r = -(w.w_energy * tariff(hour) * e_kwh + w.w_comfort * comfort
          + w.w_violation * viol)
    return r, viol


def zone_energy_kwh(hvac_kw, occ):
    dt_h = DT / 3600.0
    return hvac_kw * dt_h + 0.9 * dt_h + 0.095 * occ * dt_h


# ----------------------------------------------------------------- schedules
# A daily thermal policy is a schedule template, chosen at midnight. Summer,
# cooling only: the unoccupied setback is warm, and "pre-cooling" lowers the
# setpoint ahead of the 14:00-20:00 peak tariff so heavy mass can be charged
# with cold while electricity is cheap.
SCHEDULES = []
for s_occ in (23.0, 24.0, 25.0):
    for pre in ("none", "night21", "night20", "morning"):
        for relax in (0.0, 1.0):
            for early in (False, True):
                SCHEDULES.append({"s_occ": s_occ, "pre": pre, "relax": relax,
                                  "early": early})
N_SCHED = len(SCHEDULES)


def schedule_setpoints(sched: Dict) -> np.ndarray:
    """Setpoint for each of the 288 steps of a day under a template."""
    h = np.arange(STEPS_PER_DAY) / 12.0
    sp = np.full(STEPS_PER_DAY, 28.0)                  # night setback
    occ_h = (h >= 7.0) & (h < 19.0)
    sp[occ_h] = sched["s_occ"]
    peak = (h >= 14.0) & (h < 19.0)
    sp[peak] = sched["s_occ"] + sched["relax"]
    if sched["pre"] == "night21":
        sp[(h >= 3.0) & (h < 7.0)] = 21.0
    elif sched["pre"] == "night20":
        sp[(h >= 3.0) & (h < 7.0)] = 20.0
    elif sched["pre"] == "morning":
        sp[(h >= 7.0) & (h < 14.0)] = sched["s_occ"] - 1.5
    if sched.get("early"):
        # 00:00-03:00 at 20 C: the same profile as the thermal probe, offered
        # to every method as a control option so that the probe's measured
        # value is information only, never an extra control move
        sp[h < 3.0] = 20.0
    return sp


SCHED_SP = np.stack([schedule_setpoints(s) for s in SCHEDULES])   # (N_SCHED, 288)

# Night step test used as the thermal probe: hold 20 C from 00:00 to 03:00,
# when the zone is empty and the tariff is off-peak.
PROBE_SP = np.full(STEPS_PER_DAY, np.nan)
PROBE_SP[: 36] = 20.0

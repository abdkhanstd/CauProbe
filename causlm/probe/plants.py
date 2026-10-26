"""The gateway plant family: true simulators the agent is evaluated on.

A plant is a mechanism combination (A, B, C, D) plus continuous parameters drawn
from per-mechanism ranges. Exogenous drivers are replayed from real traces:
outdoor temperature from TON_IoT weather telemetry (rescaled to a summer
range) and the occupancy shape from the Intel Berkeley Lab light channel.
Occupancy is never observed by the gateway: it confounds the thermal load, the
radio link and the tenant traffic on the broker, and it drove the incumbent
system's knob settings in the historical logs.

Out-of-library plants (`ool=True`) swap in mechanisms that are not in the
agent's hypothesis library: a TSCH-style scheduled MAC whose delay is a slot
frame divided by the duty cycle, and a backhaul whose round trip grows with
load. They exist to test whether the agent notices that none of its
hypotheses fits and falls back.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from ..config import DATA_PROC
from . import mech
from .mech import DT, STEPS_PER_DAY, W

EXO_CACHE = DATA_PROC / "probe_exogenous.npz"

# Parameter ranges per mechanism value. The agent's prior over particles is
# drawn from exactly these ranges, so an in-library plant is representable.
NET_RANGES = {
    ("C", 0): {"n_dev": (32, 48), "bg_base": (1.0, 3.0), "bg_amp": (2.0, 5.0),
               "k_occ": (0.2, 0.5)},
    ("C", 1): {"n_dev": (5, 10), "bg_base": (8.0, 13.0), "bg_amp": (6.0, 10.0),
               "k_occ": (0.4, 0.9)},
    ("B", 0): {"p_base": (8.0, 14.0), "c_cpu": (30.0, 45.0), "rtt": (22.0, 40.0)},
    ("B", 1): {"p_base": (2.0, 5.0), "c_cpu": (150.0, 250.0),
               "rtt": (160.0, 280.0)},
    ("A", 0): {"t_wake": (0.0, 0.0)},
    ("A", 1): {"t_wake": (90.0, 170.0)},
}
NET_COMMON = {"mu": (34.0, 46.0), "sigma": (0.15, 0.30)}
TH_RANGES = {
    0: {"C_mass": (4e7, 8e7), "R_mass": (2.5e-4, 4.0e-4)},
    1: {"C_mass": (2e6, 5e6), "R_mass": (8e-4, 1.6e-3)},
}
TH_COMMON = {"C_air": (3.5e6, 6.0e6), "R_out": (2.5e-3, 4.5e-3),
             "solar": (3000.0, 6000.0), "occ_gain": (180.0, 260.0),
             "hvac_max": (12.0, 18.0), "cop": (2.8, 3.4), "kp": (4.0, 4.0)}

ALL_COMBOS = [(a, b, c, d) for a in (0, 1) for b in (0, 1) for c in (0, 1)
              for d in (0, 1)]


def _u(rng, lo_hi):
    lo, hi = lo_hi
    return float(lo if hi == lo else rng.uniform(lo, hi))


def sample_net_params(rng, A, B, C, n=None) -> Dict[str, np.ndarray]:
    """Draw network parameters; vectorised when `n` is given (particles)."""
    size = n or 1
    P = {"A": np.full(size, A), "B": np.full(size, B), "C": np.full(size, C)}
    for spec in (NET_RANGES[("C", C)], NET_RANGES[("B", B)],
                 NET_RANGES[("A", A)], NET_COMMON):
        for k, (lo, hi) in spec.items():
            P[k] = rng.uniform(lo, hi, size) if hi > lo else np.full(size, lo)
    P["s_frame"] = np.zeros(size)
    if n is None:
        return {k: v[0] for k, v in P.items()}
    return P


def sample_th_params(rng, D, n=None):
    size = n or 1
    P = {"D": np.full(size, D)}
    for spec in (TH_RANGES[D], TH_COMMON):
        for k, (lo, hi) in spec.items():
            P[k] = rng.uniform(lo, hi, size) if hi > lo else np.full(size, lo)
    if n is None:
        return {k: v[0] for k, v in P.items()}
    return P


@dataclass
class PlantSpec:
    combo: tuple                       # (A, B, C, D)
    net: Dict[str, float]
    th: Dict[str, float]
    occ_max: float
    rssi_base: float
    rssi_occ: float
    ool: Optional[str] = None          # out-of-library mechanism, if any
    seed: int = 0
    burst_p: float = 0.06              # share of steps with a tenant surge
    burst_lo: float = 2.0
    burst_hi: float = 3.5

    @property
    def name(self):
        a, b, c, d = self.combo
        return f"A{a}B{b}C{c}D{d}" + (f"-{self.ool}" if self.ool else "")


def sample_plant(seed: int, combo=None, ool: Optional[str] = None,
                 scenario: Optional[Dict] = None) -> PlantSpec:
    rng = np.random.default_rng(seed)
    if combo is None:
        combo = ALL_COMBOS[int(rng.integers(len(ALL_COMBOS)))]
    A, B, C, D = combo
    net = sample_net_params(rng, A, B, C)
    th = sample_th_params(rng, D)
    scenario = scenario or {}
    if scenario.get("widen"):
        # Prior-generator mismatch: every continuous parameter is redrawn
        # from its range widened by `widen` times its width on each side, so
        # that a large share of plants lie outside the support the agent's
        # particles are drawn from. A dedicated stream keeps everything else
        # as in the nominal plant; signs and orders of magnitude are kept.
        f = float(scenario["widen"])
        rw = np.random.default_rng(seed + 53)
        for P, specs in ((net, (NET_RANGES[("C", C)], NET_RANGES[("B", B)],
                                NET_RANGES[("A", A)], NET_COMMON)),
                         (th, (TH_RANGES[D], TH_COMMON))):
            for spec in specs:
                for k, (lo, hi) in spec.items():
                    if hi <= lo:
                        continue
                    a, b = lo - f * (hi - lo), hi + f * (hi - lo)
                    if lo > 0:
                        a = max(a, lo / 2.0)
                    if hi < 0:
                        b = min(b, hi / 2.0)
                    P[k] = float(rw.uniform(a, b))
    if "sigma" in scenario:
        # draw from a dedicated stream so the other parameters are unchanged
        net["sigma"] = float(np.random.default_rng(seed + 31).uniform(
            *scenario["sigma"]))
    if ool == "tsch":
        # Scheduled MAC: delay is one slot frame per duty-cycle period, and the
        # offered load is unaffected. Not representable by A0 or A1.
        net["A"] = 2
        net["s_frame"] = float(rng.uniform(40.0, 70.0))
    elif ool == "congested_wan":
        # A backhaul whose round trip grows with the gateway's own upload.
        net["rtt_load"] = float(rng.uniform(4.0, 7.0))
    elif ool == "broker_fault":
        # Failing broker hardware: in a random 15% of hours the service rate
        # collapses to a third. Nothing in the library produces hour-long
        # surges that no knob explains.
        net["fault_p"] = float(rng.uniform(0.12, 0.18))
    return PlantSpec(combo=tuple(combo), net=net, th=th,
                     occ_max=float(rng.uniform(8, 16)),
                     rssi_base=float(rng.uniform(-60, -50)),
                     rssi_occ=float(rng.uniform(-1.4, -0.7)), ool=ool, seed=seed,
                     burst_p=float(scenario.get("burst_p", 0.06)),
                     burst_lo=float(scenario.get("burst", (2.0, 3.5))[0]),
                     burst_hi=float(scenario.get("burst", (2.0, 3.5))[1]))


# ----------------------------------------------------------------- exogenous
_EXO = None


def load_exogenous():
    """Real exogenous traces, cached: outdoor temperature and occupancy shape."""
    global _EXO
    if _EXO is not None:
        return _EXO
    if EXO_CACHE.exists():
        z = np.load(EXO_CACHE)
        _EXO = {"outdoor": z["outdoor"], "occ": z["occ"]}
        return _EXO
    from ..data.real_traces import load_intel_lab, load_ton_iot
    ton, _ = load_ton_iot(devices=("weather",))
    w = ton["weather"]
    out = w.loc[w["type"] == "normal", "temperature"].to_numpy(float)
    out = out[np.isfinite(out)]
    lo, hi = np.quantile(out, [0.02, 0.98])
    out = np.clip((out - lo) / max(hi - lo, 1e-6), 0, 1)
    intel, _ = load_intel_lab()
    light = intel["light"].to_numpy(float)
    light = light[np.isfinite(light)]
    q = np.quantile(light, [0.05, 0.95])
    occ = np.clip((light - q[0]) / max(q[1] - q[0], 1e-6), 0, 1)
    np.savez_compressed(EXO_CACHE, outdoor=out.astype(np.float32),
                        occ=occ.astype(np.float32))
    _EXO = {"outdoor": out, "occ": occ}
    return _EXO


def sample_day_exogenous(rng, spec: PlantSpec):
    """One day of outdoor temperature (C), true occupancy and RSSI noise."""
    exo = load_exogenous()
    T = STEPS_PER_DAY
    h = np.arange(T) / 12.0
    s = int(rng.integers(0, len(exo["outdoor"]) - T - 1))
    rep = exo["outdoor"][s:s + T]
    base = rng.uniform(23.0, 29.0)                 # daily mean, summer
    amp = rng.uniform(4.0, 7.5)
    t_out = (base + amp * np.sin(2 * np.pi * (h - 9.0) / 24.0)
             + 3.0 * (rep - rep.mean()))
    s2 = int(rng.integers(0, len(exo["occ"]) - T - 1))
    shape = exo["occ"][s2:s2 + T]
    occ = np.round(np.clip(spec.occ_max * shape * mech.office_gate(h)
                           * rng.uniform(0.7, 1.1)
                           + rng.normal(0, 0.4, T), 0, None))
    return t_out, occ


# ----------------------------------------------------------------- plant
class GatewayPlant:
    """Multi-day simulator of one gateway: broker/radio network + one zone.

    `step(net_idx, setpoint, slm_busy_ms)` advances five minutes and returns
    the observation the gateway sees (no occupancy, no slab temperature), the
    reward and a diagnostics dict for the evaluator.
    """

    def __init__(self, spec: PlantSpec, seed: int = 0):
        self.spec = spec
        self.rng = np.random.default_rng(seed)
        self.P_net = {k: np.asarray(v, float) for k, v in spec.net.items()}
        self.P_th = {k: np.asarray(v, float) for k, v in spec.th.items()}
        self.day = -1
        self.t = 0
        self.t_air = 26.0
        self.t_mass = 26.0
        self.rh = 50.0
        self.pdr = 0.95
        self.prev_net = None
        self._new_day()

    def _new_day(self):
        self.day += 1
        self.t_out_day, self.occ_day = sample_day_exogenous(self.rng, self.spec)
        # the forecast the gateway gets for today: smooth error, ~1 K
        err = np.cumsum(self.rng.normal(0, 0.08, STEPS_PER_DAY))
        self.forecast_day = self.t_out_day + (err - err.mean()) \
            + self.rng.normal(0, 0.6)
        if self.day == 0:
            self.t_air = float(self.t_out_day[0] + 1.5)
            self.t_mass = self.t_air

    @property
    def hour(self):
        return (self.t % STEPS_PER_DAY) / 12.0

    def regime_change(self, **net_updates):
        """Mid-deployment structural change, e.g. A: 0 -> 1 after a radio swap."""
        for k, v in net_updates.items():
            self.P_net[k] = np.asarray(float(v))
        self.spec = copy.deepcopy(self.spec)
        self.spec.net.update(net_updates)

    def step(self, net_idx: int, setpoint: float, slm_busy_ms: float = 0.0):
        k = self.t % STEPS_PER_DAY
        h = k / 12.0
        occ = float(self.occ_day[k])
        t_out = float(self.t_out_day[k])
        f, d, o = mech.NET_ACTIONS[net_idx]
        Pn = self.P_net

        # radio link: occupancy shadows the fleet
        rssi = (self.spec.rssi_base + self.spec.rssi_occ * occ
                + self.rng.normal(0, 1.2))
        pdr = float(mech.pdr_from_rssi(rssi))
        # other tenants' traffic is bursty: a few percent of steps see a
        # two- to three-and-a-half-fold surge
        burst = (float(self.rng.uniform(self.spec.burst_lo, self.spec.burst_hi))
                 if self.rng.random() < self.spec.burst_p else 1.0)
        busy = min(0.6, slm_busy_ms / (DT * 1000.0))
        if self.spec.ool == "broker_fault":
            if k % 12 == 0:
                self._fault = self.rng.random() < float(Pn["fault_p"])
            if getattr(self, "_fault", False):
                busy = max(busy, 0.65)
        lat, it_j, aoi = mech.net_outcomes(Pn, f, d, o, h, occ, pdr, busy=busy,
                                           burst=burst)
        lat = float(lat)
        if self.spec.ool == "congested_wan":
            lam_up = float(mech.own_load(Pn, f, d))
            lat += o * Pn["rtt_load"] * lam_up
        lat *= float(np.exp(self.rng.normal(0, float(Pn["sigma"]))))
        it_j = float(it_j) + self.rng.normal(0, 6.0)
        aoi = 1.0 / (2.0 * f * pdr) + lat / 1000.0
        r_net, v_net = mech.net_reward(lat, it_j, aoi, pdr)
        sw = mech.switch_cost(self.prev_net, int(net_idx))
        r_net = float(r_net) - sw
        self.prev_net = int(net_idx)

        # thermal zone
        hvac_kw = float(mech.hvac_power(self.t_air, setpoint, self.P_th))
        self.t_air, self.t_mass = (float(x) for x in mech.thermal_step(
            self.P_th, self.t_air, self.t_mass, t_out, hvac_kw, occ, h))
        self.rh = float(np.clip(self.rh + 0.09 * occ - 0.006 * (self.rh - 55.0)
                                - 0.35 * hvac_kw * 0.1
                                + self.rng.normal(0, 0.25), 25, 80))
        from ..env.comfort import pmv_ppd
        pmv = pmv_ppd(self.t_air, rh=self.rh, vel=0.12, met=1.2, clo=0.6)[0]
        e_kwh = float(mech.zone_energy_kwh(hvac_kw, occ))
        r_th, v_th = mech.thermal_reward(e_kwh, h, pmv, occ, self.t_air)
        t_meas = self.t_air + self.rng.normal(0, 0.1)

        obs = {"t": self.t, "day": self.day, "hour": h,
               "tariff": float(mech.tariff(h)), "latency_ms": lat,
               "pdr": pdr, "it_j": it_j, "aoi": float(aoi),
               "t_air": t_meas, "hvac_kw": hvac_kw, "t_out": t_out,
               "e_kwh": e_kwh, "net_idx": int(net_idx),
               "setpoint": float(setpoint)}
        diag = {"r_net": float(r_net), "r_th": float(r_th),
                "viol_net": bool(v_net), "viol_th": bool(v_th),
                "occ": occ, "pmv": float(pmv), "t_mass": self.t_mass,
                "switch": sw,
                "burst": burst > 1.0}
        self.t += 1
        if self.t % STEPS_PER_DAY == 0:
            self._new_day()
        return obs, float(r_net + r_th), diag


# ----------------------------------------------------------------- incumbents
LOG_JITTER = 0.03


def logging_policy(rng, hour, occ):
    """The incumbent system that produced the historical logs.

    Sensors publish on events, so the effective publish rate rises with
    activity in the zone; the facilities team raised the duty cycle and turned
    on cloud analytics when the floor was busy. Every one of those choices
    tracks occupancy, which the gateway never sees: the logs are confounded by
    construction, exactly as real BMS logs are.
    """
    busy = occ > 2
    f = 0.5 if busy else 0.1
    if occ > 8:
        f = 1.0
    d = 1.0 if busy else 0.5
    o = 1.0 if occ > 4 else 0.0
    # occasional manual overrides by the facilities team; otherwise the
    # incumbent is a deterministic function of activity, like a real BMS
    if rng.random() < LOG_JITTER:
        f = float(rng.choice(mech.F_LEVELS))
    if rng.random() < LOG_JITTER:
        d = float(rng.choice(mech.D_LEVELS))
    if rng.random() < LOG_JITTER:
        o = float(rng.choice(mech.O_LEVELS))
    return mech.net_action_index(f, d, o)


INCUMBENT_SCHEDULE = mech.SCHEDULES.index({"s_occ": 24.0, "pre": "none",
                                           "relax": 0.0, "early": False})


def generate_logs(spec: PlantSpec, days: int = 14, seed: int = 0) -> List[Dict]:
    """Historical BMS logs: the incumbent policy on this plant, before handover."""
    plant = GatewayPlant(spec, seed=seed + 7_000_000)
    rng = np.random.default_rng(seed + 9_000_000)
    sp = mech.SCHED_SP[INCUMBENT_SCHEDULE]
    out = []
    for _ in range(days * STEPS_PER_DAY):
        k = plant.t % STEPS_PER_DAY
        occ = float(plant.occ_day[k])
        a = logging_policy(rng, plant.hour, occ)
        obs, r, diag = plant.step(a, sp[k])
        out.append(obs)
    return out


class RuleController:
    """What an integrator would deploy: fixed configuration plus the usual
    hot-queue backoff (drop the publish rate after three late steps, restore it
    after an hour of quiet). Assumes a Wi-Fi-like MAC."""
    name = "rule-based"

    def __init__(self):
        self.reset()

    def reset(self):
        self.f_i = 1          # 0.25 Hz
        self.late = 0
        self.quiet = 0

    def act(self, obs):
        if obs is not None:
            if obs["latency_ms"] > W.lat_sla_ms:
                self.late += 1
                self.quiet = 0
            else:
                self.quiet += 1
                self.late = 0
            if self.late >= 3 and self.f_i > 0:
                self.f_i -= 1
                self.late = 0
            if self.quiet >= 12 and self.f_i < 1:
                self.f_i += 1
                self.quiet = 0
        f = mech.F_LEVELS[self.f_i]
        return mech.net_action_index(f, 1.0, 0.5), INCUMBENT_SCHEDULE


# Verified-conservative fallback: lowest publish rate (least load in every
# mechanism), full duty cycle (no sleep delay in any MAC), half offload (hedges
# the edge- and WAN-bound cases), and the incumbent thermal schedule.
FALLBACK_NET = mech.net_action_index(0.1, 1.0, 0.5)
FALLBACK_SCHED = INCUMBENT_SCHEDULE

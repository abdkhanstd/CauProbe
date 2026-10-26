"""IoTGatewayEnv: a smart-building IoT gateway as a Gymnasium environment.

Physics
-------
* zone temperature: second-order (2R2C) resistance-capacitance model driven by
  outdoor temperature, solar gain, occupant sensible gain and HVAC thermal power;
* humidity: occupant latent gain, outdoor coupling and HVAC dehumidification;
* radio: RSSI degrades with occupancy (body shadowing) and with device ageing,
  packet-delivery ratio follows the usual logistic link model;
* broker: M/M/1 sojourn time on the gateway MQTT queue, with the service rate
  scaled by the delivery ratio.

Exogenous drivers are replayed from **real** traces where available
(TON_IoT weather for outdoor temperature, Intel Berkeley Lab light for the
occupancy profile), which is what makes the confounding realistic rather than
hand-drawn.

Confounding
-----------
`occupancy` drives HVAC load *and* Wi-Fi RSSI *and* the MQTT publish rate.
In `hidden_confounder` mode it is removed from the observation, which is
exactly the situation that makes purely correlational controllers infer the
spurious edge wifi_rssi -> indoor_temp.

Interventions
-------------
`counterfactual_ate()` rolls the same environment forward twice under
*common random numbers*, so the difference between the two arms is the true
average treatment effect of the intervention, free of Monte-Carlo noise. This
is the ground truth used to score the SLM's `predicted_ATE` field.
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from ..config import (NKPI, NVAR, VARS, VARS_ACT, VARS_KPI, NetworkConfig,
                      RewardWeights, SLMCostConfig, ThermalConfig)
from .comfort import pmv_ppd

# Discrete action levels. MultiDiscrete keeps the joint action space small
# enough for tabular-ish exploration while still covering the five knobs.
ACTION_LEVELS = {
    "hvac_setpoint": [18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0],
    "sample_rate_hz": [0.1, 0.25, 0.5, 1.0],
    "duty_cycle": [0.25, 0.5, 1.0],
    "offload_frac": [0.0, 0.5, 1.0],
    "isolate": [0, 1],
    "query_slm": [0, 1],
}
ACTION_KEYS = list(ACTION_LEVELS)
ACTION_NVEC = [len(ACTION_LEVELS[k]) for k in ACTION_KEYS]


@dataclass
class EnvOptions:
    episode_steps: int = 288
    hidden_confounder: bool = False
    drift: bool = False
    attack: bool = False
    attack_rate: float = 0.12           # fraction of steps under attack
    attack_families: Tuple[str, ...] = ("ddos", "injection", "backdoor",
                                        "password", "ransomware", "xss",
                                        "scanning")
    use_real_exogenous: bool = True
    obs_history: int = 1
    thermal: ThermalConfig = field(default_factory=ThermalConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    reward: RewardWeights = field(default_factory=RewardWeights)
    slm_cost: SLMCostConfig = field(default_factory=SLMCostConfig)
    seed: int = 0


class _Exogenous:
    """Outdoor temperature / occupancy / attack schedule, real where possible."""

    def __init__(self, opts: EnvOptions, rng: np.random.Generator):
        self.opts, self.rng = opts, rng
        self.outdoor: Optional[np.ndarray] = None
        self.occ_profile: Optional[np.ndarray] = None
        self.attack_series: Optional[np.ndarray] = None
        self.attack_types: Optional[np.ndarray] = None
        self.source = "synthetic"
        if opts.use_real_exogenous:
            self._load_real()

    def _load_real(self):
        try:
            from ..data.real_traces import load_intel_lab, load_ton_iot
            ton, _ = load_ton_iot(devices=("weather",))
            w = ton["weather"]
            self.outdoor = w.loc[w["type"] == "normal", "temperature"].to_numpy(float)
            self.outdoor = self.outdoor[np.isfinite(self.outdoor)]
            # keep a plausible outdoor range: the raw telemetry is unitless-ish
            lo, hi = np.quantile(self.outdoor, [0.02, 0.98])
            self.outdoor = np.clip(self.outdoor, lo, hi)
            self.outdoor = 2.0 + 32.0 * (self.outdoor - lo) / max(hi - lo, 1e-6)

            intel, _ = load_intel_lab()
            light = intel["light"].to_numpy(float)
            q = np.quantile(light, [0.05, 0.95])
            self.occ_profile = np.round(
                np.clip((light - q[0]) / max(q[1] - q[0], 1e-6), 0, 1) * 12.0)

            if self.opts.attack:
                fams = list(self.opts.attack_families)
                ton_a, _ = load_ton_iot(devices=("thermostat", "fridge",
                                                 "motion_light", "weather"))
                lab, typ = [], []
                for dev, df in ton_a.items():
                    sel = df[df["type"].isin(fams)]
                    lab.append(sel["label"].to_numpy(int))
                    typ.append(sel["type"].to_numpy(object))
                self.attack_series = np.concatenate(lab) if lab else None
                self.attack_types = np.concatenate(typ) if typ else None
            self.source = "real (TON_IoT weather + Intel Lab light)"
        except Exception:
            self.outdoor = None
            self.occ_profile = None
            self.source = "synthetic (real traces unavailable)"

    def sample_episode(self, T: int):
        rng = self.rng
        if self.outdoor is not None and len(self.outdoor) > T + 2:
            s = rng.integers(0, len(self.outdoor) - T - 1)
            out = self.outdoor[s:s + T].copy()
        else:
            tt = np.arange(T)
            out = 14.0 + 13.0 * np.sin(2 * np.pi * (tt / 288.0) - 1.9) \
                + rng.normal(0, 0.6, T).cumsum() * 0.05
        # a smooth diurnal envelope so the replayed values are ordered like a day
        day = np.sin(2 * np.pi * np.arange(T) / 288.0 - 1.9)
        out = 0.55 * out + 0.45 * (out.mean() + 11.0 * day)

        if self.occ_profile is not None and len(self.occ_profile) > T + 2:
            s = rng.integers(0, len(self.occ_profile) - T - 1)
            occ = self.occ_profile[s:s + T].copy()
        else:
            occ = np.zeros(T)
        # office schedule gate: essentially nobody at night
        hod = (np.arange(T) % 288) / 12.0
        gate = ((hod > 7.5) & (hod < 19.0)).astype(float)
        gate = gate * (0.65 + 0.35 * np.sin(np.pi * np.clip((hod - 7.5) / 11.5, 0, 1)))
        occ = np.round(np.clip(occ * gate + rng.normal(0, 0.35, T), 0, 12))

        atk = np.zeros(T, int)
        atk_type = np.array([""] * T, dtype=object)
        if self.opts.attack:
            n_bursts = max(1, int(self.opts.attack_rate * T / 12))
            for _ in range(n_bursts):
                st = int(rng.integers(0, max(T - 14, 1)))
                ln = int(rng.integers(6, 16))
                fam = (str(rng.choice(self.attack_types))
                       if self.attack_types is not None and len(self.attack_types)
                       else str(rng.choice(list(self.opts.attack_families))))
                atk[st:st + ln] = 1
                atk_type[st:st + ln] = fam
        return out, occ, atk, atk_type


class IoTGatewayEnv(gym.Env):
    """Gymnasium environment for causal intervention selection on an IoT gateway."""

    metadata = {"render_modes": []}

    def __init__(self, options: Optional[EnvOptions] = None):
        super().__init__()
        self.opts = options or EnvOptions()
        self.tc = self.opts.thermal
        self.nc = self.opts.network
        self.rw = self.opts.reward
        self.sc = self.opts.slm_cost
        self.rng = np.random.default_rng(self.opts.seed)
        self.exo = _Exogenous(self.opts, self.rng)

        self.action_space = spaces.MultiDiscrete(ACTION_NVEC)
        self._obs_names = self._make_obs_names()
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(len(self._obs_names),), dtype=np.float32)
        self.history: List[np.ndarray] = []
        self.reset(seed=self.opts.seed)

    # ---------------------------------------------------------------- setup
    def _make_obs_names(self) -> List[str]:
        names = [v for v in VARS_KPI]
        if self.opts.hidden_confounder:
            names.remove("occupancy")
        names += ["pdr", "queue_util", "tariff", "tod_sin", "tod_cos", "setpoint",
                  "sample_rate", "duty_cycle", "offload", "n_isolated",
                  "slm_cost_last", "steps_since_slm", "suspicion"]
        return names

    def reset(self, *, seed: Optional[int] = None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            self.exo.rng = self.rng
        T = self.opts.episode_steps
        self.T = T
        self.t = 0
        self.outdoor, self.occ, self.atk, self.atk_type = self.exo.sample_episode(T)

        self.t_air = float(self.outdoor[0] + self.rng.normal(2.0, 0.6))
        self.t_mass = self.t_air + self.rng.normal(0.0, 0.3)
        self.rh = float(np.clip(self.rng.normal(45.0, 4.0), 25, 70))
        self.setpoint = 22.0
        self.sample_rate = 0.5
        self.duty_cycle = 1.0
        self.offload = 0.0
        self.isolated = np.zeros(self.nc.n_devices, bool)
        self.suspicion = np.zeros(self.nc.n_devices)
        self.energy_cum = 0.0
        self.energy_cost_cum = 0.0
        self.it_energy_cum = 0.0
        self.slm_queries = 0
        self.slm_energy_j = 0.0
        self.slm_cost_last = 0.0
        self.steps_since_slm = 0
        self.rssi_drift = 0.0
        self.r_out_drift = 0.0
        self.service_drift = 0.0
        self.history = []
        self.episode_log: List[Dict] = []
        self._last_kpis = self._kpis()
        return self._obs(), {"exogenous_source": self.exo.source}

    # ------------------------------------------------------------- dynamics
    def _decode(self, action) -> Dict:
        a = np.asarray(action).ravel().astype(int)
        return {k: ACTION_LEVELS[k][int(a[i])] for i, k in enumerate(ACTION_KEYS)}

    def _radio(self, occ: float, atk: int, atk_type: str):
        nc = self.nc
        rssi = (nc.base_rssi + nc.rssi_occ_slope * occ + self.rssi_drift
                + self.rng.normal(0, nc.rssi_noise))
        if atk and atk_type in ("ddos", "scanning"):
            rssi -= 4.5              # channel contention from flood traffic
        pdr = 1.0 / (1.0 + math.exp(-nc.pdr_rssi_slope * (rssi - nc.pdr_rssi_mid)))
        return float(rssi), float(np.clip(pdr, 0.02, 0.999))

    def _tariff(self) -> float:
        """Time-of-use price multiplier for the current step."""
        rw = self.rw
        hod = (self.t % 288) / 12.0
        if rw.tou_peak_hours[0] <= hod < rw.tou_peak_hours[1]:
            return rw.tou_peak
        if rw.tou_mid_hours[0] <= hod < rw.tou_mid_hours[1]:
            return rw.tou_mid
        return rw.tou_offpeak

    def _background_load(self) -> float:
        """Traffic other tenants put on the same broker.

        A gateway rarely owns its uplink. The background load follows the
        working day and occasionally bursts, so the sampling rate that keeps
        the queue inside its service objective changes within the episode --
        which is what makes the network knobs worth controlling at all.
        """
        nc = self.nc
        hod = (self.t % 288) / 12.0
        day = max(0.0, math.sin(math.pi * (hod - 6.0) / 14.0))
        lam = nc.bg_load_mean + nc.bg_load_amp * day
        if self.rng.random() < nc.bg_burst_p:
            lam *= nc.bg_burst_mult
        return float(lam)

    def _queue(self, pdr: float, atk: int, atk_type: str,
               slm_busy_ms: float = 0.0):
        nc = self.nc
        active = int((~self.isolated).sum())
        lam = active * self.sample_rate * self.duty_cycle + self._background_load()
        if atk and atk_type in ("ddos", "scanning", "password"):
            lam *= self.rng.uniform(2.6, 5.0)
        # Link-layer ARQ: a lost frame is retransmitted, so a weak link
        # multiplies the *offered* load rather than silently dropping it. This
        # is the mechanism that makes wifi_rssi a genuine cause of latency.
        lam = lam / max(pdr, 0.05)
        # A gateway SLM call occupies the shared processor while it runs, so
        # the broker is starved for that fraction of the control period and the
        # backlog it builds has to be drained afterwards.
        busy = min(0.6, self.sc.cpu_contention * slm_busy_ms
                   / (self.tc.dt * 1000.0))
        mu = max(nc.service_rate_hz * (1.0 - busy) - self.service_drift, 0.15)
        rho = lam / mu
        if rho >= 0.985:
            wait_ms = 1000.0 / (mu * 0.015)
        else:
            wait_ms = 1000.0 / (mu * (1.0 - rho))
        proc = (1 - self.offload) * nc.edge_proc_ms + self.offload * (
            nc.cloud_rtt_ms + abs(self.rng.normal(0, nc.cloud_rtt_jitter)))
        return float(wait_ms + proc), float(min(rho, 1.2)), float(lam)

    def _thermal_step(self, occ: float, t_out: float, hvac_kw: float, cooling: bool):
        tc = self.tc
        dt = tc.dt
        hod = ((self.t % 288) / 12.0)
        solar = tc.solar_gain_w * max(0.0, math.sin(math.pi * (hod - 6.5) / 12.0))
        q_occ = (tc.occ_gain_w + tc.plug_load_w) * occ
        q_hvac = tc.hvac_cop * hvac_kw * 1000.0   # hvac_kw is already signed
        r_out = tc.R_out * (1.0 + self.r_out_drift)
        dTa = ((t_out - self.t_air) / r_out
               + (self.t_mass - self.t_air) / tc.R_mass
               + q_occ + solar + q_hvac) / tc.C_air * dt
        dTm = ((self.t_air - self.t_mass) / tc.R_mass) / tc.C_mass * dt
        self.t_air = float(np.clip(self.t_air + dTa, 5.0, 45.0))
        self.t_mass = float(np.clip(self.t_mass + dTm, 5.0, 45.0))

        # humidity: occupant latent gain, outdoor coupling, HVAC dehumidification
        d_rh = (0.090 * occ - 0.06 * (self.rh - 42.0) * 0.1
                + 0.55 * hvac_kw            # cooling (negative) dehumidifies
                + self.rng.normal(0, 0.25))
        self.rh = float(np.clip(self.rh + d_rh, 18.0, 85.0))

    def _hvac_command(self) -> Tuple[float, bool]:
        """Proportional zone controller tracking the commanded setpoint.

        Returns the *signed* thermal power in kW: negative means the plant is
        cooling, positive means heating. Building management systems log the
        signed value, and keeping the sign is what makes the edge
        hvac_power -> indoor_temp identifiable at all: an unsigned magnitude
        is a symmetric function of the tracking error and is therefore
        uncorrelated with the effect it causes.
        """
        err = self.t_air - self.setpoint
        kp = 4.0
        p = min(abs(err) * kp, self.tc.hvac_max_kw)
        if abs(err) < 0.4:
            p = 0.0
        return float(-p if err > 0 else p), (err > 0)

    def step(self, action):
        act = self._decode(action)
        self.setpoint = act["hvac_setpoint"]
        self.sample_rate = act["sample_rate_hz"]
        self.duty_cycle = act["duty_cycle"]
        self.offload = act["offload_frac"]

        if act["isolate"] and self.suspicion.max() > 0.5:
            self.isolated[int(np.argmax(self.suspicion))] = True
        if self.rng.random() < 0.02:                # slow re-admission
            idx = np.flatnonzero(self.isolated)
            if len(idx):
                self.isolated[self.rng.choice(idx)] = False

        occ = float(self.occ[self.t])
        t_out = float(self.outdoor[self.t])
        atk = int(self.atk[self.t])
        atk_type = str(self.atk_type[self.t])

        if self.opts.drift:
            self.rssi_drift -= 0.011                    # antenna / battery ageing
            self.r_out_drift += 0.0016                  # envelope / seasonal drift
            self.service_drift += 0.006                 # broker degradation

        hvac_kw, cooling = self._hvac_command()
        self._thermal_step(occ, t_out, hvac_kw, cooling)
        rssi, pdr = self._radio(occ, atk, atk_type)
        slm_busy_ms = self.sc.latency_ms if act["query_slm"] else 0.0
        latency, rho, lam = self._queue(pdr, atk, atk_type, slm_busy_ms)

        # attacker-controlled sensor tampering corrupts the measured temperature
        t_meas = self.t_air
        if atk and atk_type in ("injection", "backdoor", "ransomware", "xss"):
            self.suspicion = np.clip(
                self.suspicion + self.rng.random(self.nc.n_devices) * 0.22, 0, 1)
            t_meas = self.t_air + self.rng.normal(0, 3.5)
        else:
            self.suspicion *= 0.9
        if atk and atk_type in ("ddos", "scanning"):
            self.suspicion = np.clip(self.suspicion + 0.12, 0, 1)

        # energy: HVAC + radio + local compute + cloud offload + SLM
        dt_h = self.tc.dt / 3600.0
        tariff = self._tariff()
        e_hvac = abs(hvac_kw) * dt_h                      # metered magnitude
        base_load = 0.9 * dt_h                            # lighting / standby
        plug = self.tc.plug_load_w * occ / 1000.0 * dt_h   # occupant desk load
        e_step = e_hvac + base_load + plug + abs(self.rng.normal(0, 0.0012))

        # Gateway IT energy, in Joules, on its own budget. Radio energy scales
        # with publish rate x duty cycle x active devices; local inference costs
        # more than shipping the work to the cloud; the SLM adds its measured
        # per-invocation Joules on top.
        p_radio = (0.42 * self.sample_rate * self.duty_cycle
                   * int((~self.isolated).sum()) / self.nc.n_devices)
        p_compute = (1 - self.offload) * 1.35 + self.offload * 0.42
        it_j = (p_radio + p_compute + self.sc.idle_power_w) * self.tc.dt
        it_j += self.rng.normal(0, 4.0)

        slm_j = 0.0
        decision_lat = 2.0
        if act["query_slm"]:
            self.slm_queries += 1
            slm_j = self.sc.energy_j
            self.slm_energy_j += slm_j
            decision_lat += self.sc.latency_ms
            self.steps_since_slm = 0
            self.slm_cost_last = slm_j
        else:
            self.steps_since_slm += 1
            self.slm_cost_last *= 0.8
        it_j += slm_j
        e_cost = tariff * e_step
        self.energy_cum += e_step
        self.energy_cost_cum += e_cost
        self.it_energy_cum += it_j

        pmv, ppd = pmv_ppd(self.t_air, rh=self.rh, vel=0.12, met=1.2, clo=0.7)

        rw = self.rw
        # Discomfort only counts when somebody is there to feel it. Charging a
        # comfort penalty in an empty zone at 3 a.m. removes exactly the lever
        # -- unoccupied setback and pre-cooling into the off-peak tariff --
        # that a building controller exists to use.
        occ_w = min(1.0, occ / rw.comfort_occ_full)
        comfort_pen = occ_w * max(0.0, abs(pmv) - rw.comfort_band)
        lat_pen = min(max(0.0, latency - rw.latency_sla_ms),
                      rw.latency_pen_cap_ms)
        safety = 0.0
        if self.t_air < 16.0 or self.t_air > 30.0:
            safety += 1.0
        if pdr < 0.5:
            safety += 1.0
        reward = -(rw.w_energy * e_cost
                   + rw.w_comfort * comfort_pen
                   + rw.w_latency * lat_pen
                   + rw.w_pdr * (1.0 - pdr)
                   + rw.w_safety * safety
                   + rw.lambda_slm * slm_j / 1000.0
                   + rw.w_it_energy * it_j / 1000.0
                   + rw.w_decision_lat * decision_lat / 1000.0)

        kpi = dict(occupancy=occ, outdoor_temp=t_out, indoor_temp=self.t_air,
                   humidity=self.rh, hvac_power=hvac_kw, wifi_rssi=rssi,
                   latency_ms=latency, energy_kwh=e_step, it_energy_j=it_j,
                   tariff=tariff, energy_cost=e_cost,
                   hvac_setpoint=self.setpoint, sample_rate_hz=self.sample_rate,
                   duty_cycle=self.duty_cycle, offload_frac=self.offload)
        self.history.append(np.array([kpi[v] for v in VARS], np.float32))
        self._last_kpis = dict(kpi, pdr=pdr, pmv=pmv, ppd=ppd, queue_util=rho,
                               offered_load=lam, attack=atk, attack_type=atk_type,
                               t_meas=t_meas, slm_energy_j=slm_j,
                               n_isolated=int(self.isolated.sum()),
                               decision_latency_ms=decision_lat,
                               sla_violation=int(latency > rw.latency_sla_ms),
                               safety=safety)
        self.episode_log.append(dict(self._last_kpis, t=self.t, reward=reward,
                                     query_slm=int(act["query_slm"]),
                                     setpoint=self.setpoint))
        self.t += 1
        done = self.t >= self.T
        return self._obs(), float(reward), done, False, dict(self._last_kpis)

    # ------------------------------------------------------------- observation
    def _kpis(self) -> Dict:
        return dict(occupancy=0.0, outdoor_temp=self.outdoor[0],
                    indoor_temp=self.t_air, humidity=self.rh, hvac_power=0.0,
                    wifi_rssi=self.nc.base_rssi, latency_ms=20.0, energy_kwh=0.0,
                    it_energy_j=900.0, tariff=1.0, energy_cost=0.0,
                    pdr=0.95, queue_util=0.2, pmv=0.0, ppd=5.0)

    def _obs(self) -> np.ndarray:
        k = self._last_kpis
        hod = ((self.t % 288) / 288.0) * 2 * math.pi
        vals = []
        for v in VARS_KPI:
            if v == "occupancy" and self.opts.hidden_confounder:
                continue
            vals.append(k.get(v, 0.0))
        vals += [k.get("pdr", 0.9), k.get("queue_util", 0.2),
                 self._tariff(),
                 math.sin(hod), math.cos(hod), self.setpoint, self.sample_rate,
                 self.duty_cycle, self.offload, float(self.isolated.sum()),
                 self.slm_cost_last, min(self.steps_since_slm, 50) / 50.0,
                 float(self.suspicion.max())]
        return np.asarray(vals, np.float32)

    # ------------------------------------------------------------ interfaces
    def get_window(self, w: int = 288) -> np.ndarray:
        """Last `w` steps of the canonical 8-variable KPI matrix, for PCMCI+."""
        if not self.history:
            return np.zeros((0, NVAR), np.float32)
        H = np.stack(self.history[-w:])
        if self.opts.hidden_confounder:
            # the confounder is genuinely absent from the data the gateway sees
            H = np.delete(H, VARS.index("occupancy"), axis=1)
        return H

    def observed_vars(self) -> List[str]:
        return [v for v in VARS
                if not (self.opts.hidden_confounder and v == "occupancy")]

    def observed_idx(self) -> List[int]:
        return [i for i, v in enumerate(VARS)
                if not (self.opts.hidden_confounder and v == "occupancy")]

    def snapshot(self):
        """Deep copy of everything needed to replay the environment exactly."""
        return copy.deepcopy({
            "t": self.t, "t_air": self.t_air, "t_mass": self.t_mass,
            "rh": self.rh, "setpoint": self.setpoint,
            "sample_rate": self.sample_rate, "duty_cycle": self.duty_cycle,
            "offload": self.offload, "isolated": self.isolated,
            "suspicion": self.suspicion, "energy_cum": self.energy_cum,
            "energy_cost_cum": self.energy_cost_cum,
            "it_energy_cum": self.it_energy_cum,
            "slm_queries": self.slm_queries, "slm_energy_j": self.slm_energy_j,
            "slm_cost_last": self.slm_cost_last,
            "steps_since_slm": self.steps_since_slm,
            "rssi_drift": self.rssi_drift, "r_out_drift": self.r_out_drift,
            "service_drift": self.service_drift,
            "rng": self.rng.bit_generator.state,
            "history_len": len(self.history), "last_kpis": self._last_kpis,
        })

    def restore(self, s):
        s = copy.deepcopy(s)
        self.t = s["t"]; self.t_air = s["t_air"]; self.t_mass = s["t_mass"]
        self.rh = s["rh"]; self.setpoint = s["setpoint"]
        self.sample_rate = s["sample_rate"]; self.duty_cycle = s["duty_cycle"]
        self.offload = s["offload"]; self.isolated = s["isolated"]
        self.suspicion = s["suspicion"]; self.energy_cum = s["energy_cum"]
        self.energy_cost_cum = s["energy_cost_cum"]
        self.it_energy_cum = s["it_energy_cum"]
        self.slm_queries = s["slm_queries"]; self.slm_energy_j = s["slm_energy_j"]
        self.slm_cost_last = s["slm_cost_last"]
        self.steps_since_slm = s["steps_since_slm"]
        self.rssi_drift = s["rssi_drift"]; self.r_out_drift = s["r_out_drift"]
        self.service_drift = s["service_drift"]
        self.rng.bit_generator.state = s["rng"]
        self.history = self.history[:s["history_len"]]
        self._last_kpis = s["last_kpis"]

    # ------------------------------------------------- interventional oracle
    def counterfactual_ate(self, action_do, action_ref, horizon: int = 12,
                           targets=("energy_kwh", "indoor_temp", "latency_ms",
                                    "hvac_power", "wifi_rssi"),
                           n_mc: int = 24) -> Dict[str, float]:
        """True ATE of do(action_do) vs do(action_ref) over `horizon` steps.

        Both arms start from the identical environment state and identical RNG
        stream (common random numbers), so the returned difference is the causal
        effect and not sampling noise. Averaged over `n_mc` restarts of the RNG
        so that the estimate covers the stochastic part of the dynamics too.
        """
        base = self.snapshot()
        acc = {t: 0.0 for t in targets}
        acc["pmv"] = 0.0
        for m in range(n_mc):
            seed_state = np.random.default_rng(1000 + m).bit_generator.state
            arms = {}
            for name, a in (("do", action_do), ("ref", action_ref)):
                self.restore(base)
                self.rng.bit_generator.state = copy.deepcopy(seed_state)
                sums = {t: 0.0 for t in targets}
                sums["pmv"] = 0.0
                n = 0
                for _ in range(horizon):
                    if self.t >= self.T:
                        break
                    _, _, done, _, info = self.step(a)
                    for t in targets:
                        sums[t] += info.get(t, 0.0)
                    sums["pmv"] += info.get("pmv", 0.0)
                    n += 1
                    if done:
                        break
                arms[name] = {k: v / max(n, 1) for k, v in sums.items()}
            for k in acc:
                acc[k] += arms["do"][k] - arms["ref"][k]
        self.restore(base)
        return {k: v / n_mc for k, v in acc.items()}

    def episode_summary(self) -> Dict[str, float]:
        if not self.episode_log:
            return {}
        L = self.episode_log
        f = lambda k: np.array([d[k] for d in L], float)
        lat = f("latency_ms")
        return {
            "return": float(f("reward").sum()),
            "energy_kwh": float(f("energy_kwh").sum()),
            "energy_cost": float(f("energy_cost").sum()),
            "peak_kwh": float(f("energy_kwh")[f("tariff") > 2.0].sum()),
            "hvac_kwh": float((np.abs(f("hvac_power")) * self.tc.dt / 3600.0).sum()),
            "cooling_frac": float((f("hvac_power") < 0).mean()),
            "mean_abs_pmv": float(np.abs(f("pmv")).mean()),
            "mean_ppd": float(f("ppd").mean()),
            "comfort_viol_pct": float((np.abs(f("pmv")) > self.rw.comfort_band).mean() * 100),
            "latency_p50": float(np.percentile(lat, 50)),
            "latency_p95": float(np.percentile(lat, 95)),
            "sla_viol_pct": float(f("sla_violation").mean() * 100),
            "mean_pdr": float(f("pdr").mean()),
            "safety_viol_pct": float((f("safety") > 0).mean() * 100),
            "slm_queries": int(f("query_slm").sum()),
            "slm_query_rate": float(f("query_slm").mean()),
            "it_energy_j": float(self.it_energy_cum),
            "it_energy_wh": float(self.it_energy_cum / 3600.0),
            "slm_energy_j": float(self.slm_energy_j),
            "slm_share_of_it_pct": float(100.0 * self.slm_energy_j
                                         / max(self.it_energy_cum, 1e-9)),
            "gateway_slm_j_per_step": float(self.slm_energy_j / max(len(L), 1)),
            "decision_latency_mean_ms": float(f("decision_latency_ms").mean()),
            "attack_steps": int(f("attack").sum()),
        }

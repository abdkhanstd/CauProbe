"""Global configuration: canonical variable set, physical constants, paths.

Everything that experiments share lives here so that the Gym environment, the
causal-discovery stage, the SLM prompts and the paper tables all agree on
variable names and ordering.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROC = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
WEIGHTS = ROOT / "weights"
FIGS = ROOT / "paper" / "figs"
for _p in (DATA_PROC, RESULTS, WEIGHTS, FIGS):
    _p.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------
# Canonical IoT-gateway variable set.
# Order matters: it fixes the row/column order of every adjacency matrix.
# --------------------------------------------------------------------------
VARS_KPI = [
    "occupancy",      # people present in the zone (exogenous confounder)
    "outdoor_temp",   # outdoor air temperature (exogenous confounder)
    "indoor_temp",    # zone air temperature
    "humidity",       # zone relative humidity
    "hvac_power",     # signed HVAC thermal power (kW; <0 cooling, >0 heating)
    "wifi_rssi",      # mean received signal strength of the sensor fleet (dBm)
    "latency_ms",     # end-to-end MQTT publish-to-broker latency
    "energy_kwh",     # zone electrical energy for the control step (kWh)
    "it_energy_j",    # gateway IT energy for the step (J): radio + compute + SLM
    "tariff",         # time-of-use electricity price multiplier (exogenous)
    "energy_cost",    # what the zone energy actually costs this step
]
# The gateway logs its own control signals, so they are observed variables of
# the causal system, not hidden exogenous noise. Including them is what makes
# the ground-truth graph identifiable; leaving them out would make every
# actuator an unobserved common cause of latency and energy.
VARS_ACT = ["hvac_setpoint", "sample_rate_hz", "duty_cycle", "offload_frac"]
VARS = VARS_KPI + VARS_ACT
NKPI = len(VARS_KPI)
NVAR = len(VARS)
VIDX = {v: i for i, v in enumerate(VARS)}

# Variables the gateway can intervene on, and their admissible ranges.
ACTUATORS = {
    "hvac_setpoint": (18.0, 27.0),
    "sample_rate_hz": (0.1, 1.0),
    "duty_cycle": (0.1, 1.0),
    "offload_frac": (0.0, 1.0),
}

# Human-readable descriptions handed to the SLM inside the prompt.
VAR_DESC = {
    "occupancy": "number of people in the zone (0-12), exogenous",
    "outdoor_temp": "outdoor air temperature in C, exogenous",
    "indoor_temp": "zone air temperature in C",
    "humidity": "zone relative humidity in %",
    "hvac_power": "signed HVAC thermal power in kW, negative = cooling",
    "wifi_rssi": "mean Wi-Fi RSSI of the sensor fleet in dBm",
    "latency_ms": "MQTT publish-to-broker latency in ms",
    "energy_kwh": "zone electrical energy for the control step in kWh",
    "it_energy_j": "gateway IT energy for the step in J (radio+compute+SLM)",
    "tariff": "time-of-use electricity price multiplier, exogenous",
    "energy_cost": "zone energy cost for the step = tariff x energy_kwh",
    "hvac_setpoint": "commanded zone setpoint in C (actuator)",
    "sample_rate_hz": "per-device sensor publish rate in Hz (actuator)",
    "duty_cycle": "radio duty cycle, 0-1 (actuator)",
    "offload_frac": "fraction of inference sent to the cloud, 0-1 (actuator)",
}

# Ground-truth structural graph of the gateway environment, read directly off
# the equations in causlm/env/gateway_env.py. Entry (u, v) means u -> v.
TRUE_ENV_EDGES = [
    ("occupancy", "indoor_temp"), ("occupancy", "humidity"),
    ("occupancy", "wifi_rssi"), ("occupancy", "energy_kwh"),
    ("outdoor_temp", "indoor_temp"),
    ("indoor_temp", "hvac_power"),
    ("hvac_setpoint", "hvac_power"),
    ("hvac_power", "indoor_temp"), ("hvac_power", "humidity"),
    ("hvac_power", "energy_kwh"),
    ("wifi_rssi", "latency_ms"),
    ("tariff", "energy_cost"), ("energy_kwh", "energy_cost"),
    ("sample_rate_hz", "latency_ms"), ("sample_rate_hz", "it_energy_j"),
    ("duty_cycle", "latency_ms"), ("duty_cycle", "it_energy_j"),
    ("offload_frac", "latency_ms"), ("offload_frac", "it_energy_j"),
]


# CausalTime-IoT is a *time-series discovery* benchmark generated from real
# traces. Public IoT archives contain no actuator command logs, so it is built
# over the eight KPI channels that do have real provenance. The gateway control
# benchmark, which does observe its own actuators, uses the full VARS list.
CT_IOT_VARS = ["occupancy", "outdoor_temp", "indoor_temp", "humidity",
               "hvac_power", "wifi_rssi", "latency_ms", "energy_kwh"]
N_CT_IOT = len(CT_IOT_VARS)


def true_env_graph():
    """(NVAR, NVAR) float adjacency of the environment's structural equations."""
    import numpy as _np
    G = _np.zeros((NVAR, NVAR), dtype=_np.float32)
    for u, v in TRUE_ENV_EDGES:
        G[VIDX[u], VIDX[v]] = 1.0
    return G


@dataclass
class ThermalConfig:
    """Second-order RC model of a single conditioned zone."""
    C_air: float = 4.8e6      # J/K, air + light furniture capacitance
    C_mass: float = 3.6e7     # J/K, structural mass capacitance
    R_out: float = 3.6e-3     # K/W, air <-> outdoor resistance
    R_mass: float = 3.6e-4    # K/W, air <-> mass resistance
    hvac_cop: float = 3.1     # coefficient of performance
    hvac_max_kw: float = 16.0  # maximum electrical power draw
    occ_gain_w: float = 130.0 # sensible heat gain per occupant, W
    plug_load_w: float = 95.0 # desk equipment drawn per occupant, W (heats the
                              # zone AND shows up on the electricity meter)
    solar_gain_w: float = 5200.0 # peak solar gain, W
    dt: float = 300.0         # control step, seconds (5 min)


@dataclass
class NetworkConfig:
    """M/M/1-style MQTT broker queue on the gateway."""
    n_devices: int = 24
    service_rate_hz: float = 22.0   # broker packets/s the gateway can serve
    base_rssi: float = -52.0        # dBm at nominal conditions
    rssi_occ_slope: float = -1.15   # dBm lost per occupant (body shadowing)
    rssi_noise: float = 1.4
    pdr_rssi_mid: float = -80.0     # dBm where PDR = 0.5
    pdr_rssi_slope: float = 0.22
    bg_load_mean: float = 5.0       # other tenants' traffic on the same broker
    bg_load_amp: float = 4.5        # diurnal swing of that background load
    bg_burst_p: float = 0.05        # probability a step sees a traffic burst
    bg_burst_mult: float = 2.6
    cloud_rtt_ms: float = 48.0
    cloud_rtt_jitter: float = 9.0
    edge_proc_ms: float = 3.2


@dataclass
class SLMCostConfig:
    """Measured cost of one SLM invocation on the gateway.

    Defaults are placeholders; `experiments/profile_slm.py` overwrites them
    with values measured on the machine that runs the experiments.
    """
    latency_ms: float = 210.0       # measured wall-clock per invocation
    energy_j: float = 4.6           # measured Joules per invocation
    idle_power_w: float = 3.0       # gateway idle draw
    cpu_contention: float = 1.0     # fraction of gateway CPU the SLM takes
                                    # while running (1.0 = fully saturating)


@dataclass
class RewardWeights:
    """Scalarisation of the multi-objective gateway QoS problem."""
    w_energy: float = 0.25      # per unit of tariff-weighted energy cost
    w_comfort: float = 0.55     # per unit |PMV|
    w_latency: float = 0.0012   # per ms above the SLA (saturating)
    latency_pen_cap_ms: float = 400.0
    w_pdr: float = 2.0          # per unit packet loss
    w_safety: float = 6.0       # constraint-violation penalty
    lambda_slm: float = 0.25    # price of one kilojoule of gateway inference
                                # energy, in reward units. Because the charge
                                # is proportional to the *measured* Joules per
                                # invocation, a bigger or slower model is
                                # automatically more expensive to consult;
                                # lambda is swept in the ablations.
    w_it_energy: float = 0.02   # price of one kilojoule of gateway IT energy
                                # (radio, local inference, cloud offload)
    latency_sla_ms: float = 120.0
    # Time-of-use tariff: flat overnight, mid-peak in the morning, peak in the
    # late afternoon. This is what makes the setpoint decision genuinely
    # time-dependent -- without it a single well-chosen constant setpoint is
    # close to optimal and there is nothing for a controller to learn.
    tou_offpeak: float = 0.55
    tou_mid: float = 1.0
    tou_peak: float = 2.4
    tou_peak_hours: tuple = (14.0, 20.0)
    tou_mid_hours: tuple = (7.0, 14.0)
    comfort_band: float = 0.5   # |PMV| considered comfortable
    comfort_occ_full: float = 3.0   # occupants at which comfort counts fully
    w_decision_lat: float = 0.01    # per second of control-decision latency


@dataclass
class ExperimentConfig:
    seed: int = 0
    episode_steps: int = 288          # 24 h at 5-min steps
    train_episodes: int = 120
    eval_episodes: int = 30
    window: int = 288                 # sliding window handed to PCMCI+
    tau_max: int = 3
    pc_alpha: float = 0.05
    thermal: ThermalConfig = field(default_factory=ThermalConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    slm_cost: SLMCostConfig = field(default_factory=SLMCostConfig)
    reward: RewardWeights = field(default_factory=RewardWeights)


DEFAULT = ExperimentConfig()


def load_slm_cost(model_key: str = "qwen2.5-0.5b", quant: str = "none",
                  adapter: bool = True,
                  path: Optional[Path] = None) -> SLMCostConfig:
    """Fill SLMCostConfig from the measured single-invocation profile.

    Falls back to the dataclass defaults, with a warning, when the profile has
    not been produced yet -- but every number reported in the paper comes from
    a run where it had been.
    """
    import json
    path = Path(path or (RESULTS / "slm_cost_profile.json"))
    if not path.exists():
        print(f"  [warn] {path} missing; using default SLM cost "
              f"({DEFAULT.slm_cost.energy_j} J, {DEFAULT.slm_cost.latency_ms} ms)")
        return SLMCostConfig()
    prof = json.loads(path.read_text())
    key = f"{model_key}/{quant}" + ("+lora" if adapter else "")
    row = prof.get(key)
    if row is None or "latency_ms" not in row:
        for k, v in prof.items():
            if isinstance(v, dict) and "latency_ms" in v and model_key in k:
                print(f"  [warn] no profile for {key}; charging {k} instead")
                key, row = k, v
                break
    if row is None:
        print(f"  [warn] no measured profile for {key} and no near match; "
              f"charging placeholder defaults ({DEFAULT.slm_cost.energy_j} J, "
              f"{DEFAULT.slm_cost.latency_ms} ms). Any result that depends on "
              f"invocation cost is meaningless until profile_slm.py is run "
              f"for this configuration.")
        return SLMCostConfig()
    return SLMCostConfig(latency_ms=float(row["latency_ms"]),
                         energy_j=float(row["energy_j"]))

# Models used for the SLM stage. Kept here so ablations can iterate over them.
SLM_MODELS = {
    "qwen2.5-0.5b": "Qwen/Qwen2.5-0.5B-Instruct",
    "qwen2.5-1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
    "llama3.2-1b": "meta-llama/Llama-3.2-1B-Instruct",
}

HF_HOME = os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface"))

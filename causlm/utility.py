"""The one definition of "is this intervention worth taking".

The teacher, the CPU fallback policy, the bandit's prior injection and the
offline scoring in Experiment 2 all have to agree on this, or the model is
distilled towards one objective and evaluated against another.

It mirrors `IoTGatewayEnv`'s reward exactly: comfort is scored on the
*resulting* discomfort rather than on the size of the change, and latency is
scored through the saturating SLA penalty rather than as a raw magnitude. An
earlier version used `max(delta_latency, 0)`, which silently gave no credit for
relieving a saturated broker queue -- the single most valuable thing the
sampling-rate knob can do.
"""
from __future__ import annotations

from typing import Dict, Optional

from .config import DEFAULT, RewardWeights


def latency_penalty(lat_ms: float, rw: RewardWeights) -> float:
    return rw.w_latency * min(max(0.0, lat_ms - rw.latency_sla_ms),
                              rw.latency_pen_cap_ms)


def comfort_penalty(pmv: float, rw: RewardWeights) -> float:
    return rw.w_comfort * max(0.0, abs(pmv) - rw.comfort_band)


def intervention_utility(ate: Dict[str, float], kpis: Optional[Dict] = None,
                         rw: Optional[RewardWeights] = None) -> float:
    """Expected per-step reward improvement from applying an intervention.

    `ate` holds the average treatment effects of the intervention against
    holding the current settings; `kpis` is the current operating point, which
    is what makes the comfort and latency terms directional.
    """
    rw = rw or DEFAULT.reward
    kpis = kpis or {}
    pmv0 = float(kpis.get("pmv", 0.0))
    lat0 = float(kpis.get("latency_ms", rw.latency_sla_ms))
    pdr0 = float(kpis.get("pdr", 1.0))

    tariff = float(kpis.get("tariff", 1.0))
    d_energy = rw.w_energy * (ate.get("energy_cost")
                              if "energy_cost" in ate
                              else tariff * ate.get("energy_kwh", 0.0))
    d_comfort = (comfort_penalty(pmv0 + ate.get("pmv", 0.0), rw)
                 - comfort_penalty(pmv0, rw))
    d_latency = (latency_penalty(lat0 + ate.get("latency_ms", 0.0), rw)
                 - latency_penalty(lat0, rw))
    d_it = 0.004 * ate.get("it_energy_j", 0.0) / 1000.0
    d_pdr = 0.0
    if "wifi_rssi" in ate:
        # RSSI moves the delivery ratio through the usual logistic link
        import math
        s = DEFAULT.network.pdr_rssi_slope
        r0 = float(kpis.get("wifi_rssi", DEFAULT.network.base_rssi))
        p1 = 1.0 / (1.0 + math.exp(-s * (r0 + ate["wifi_rssi"]
                                         - DEFAULT.network.pdr_rssi_mid)))
        d_pdr = rw.w_pdr * ((1.0 - p1) - (1.0 - pdr0))
    return -(d_energy + d_comfort + d_latency + d_it + d_pdr)

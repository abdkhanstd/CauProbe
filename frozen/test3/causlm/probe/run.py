"""One driver for every method: commission a gateway on a plant, record it.

The driver owns the plant, the clock, the commissioning text and the price of
asking. An agent only ever sees the observation dict the plant emits (no
occupancy, no slab temperature, no mechanism labels), plus the expert's
reading when it chose to pay for one.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import numpy as np

from .mech import DT, STEPS_PER_DAY
from .plants import GatewayPlant, PlantSpec


@dataclass
class AskCost:
    """Price of one expert invocation on the target device."""
    energy_j: float = 40.0         # Joules per call
    latency_ms: float = 6000.0     # CPU-busy time per call (contention)
    price_per_kj: float = 0.4      # reward units per kJ, = the IT-energy price
    device: str = "emulated"

    @property
    def price(self):
        return self.price_per_kj * self.energy_j / 1000.0


def run_commissioning(spec: PlantSpec, agent, days: int = 14, seed: int = 0,
                      expert_out: Optional[Dict[str, float]] = None,
                      ask: AskCost = AskCost(), regime: Optional[Dict] = None,
                      record_steps: bool = False) -> Dict:
    plant = GatewayPlant(spec, seed=seed)
    if hasattr(agent, "bind_plant"):
        agent.bind_plant(plant)
    if hasattr(agent, "cfg"):
        agent.cfg.ask_price = ask.price
    if expert_out is not None and hasattr(agent, "offer_text"):
        agent.offer_text()
    obs = None
    is_ticket = False
    T = days * STEPS_PER_DAY
    R = np.zeros(T)
    Rn = np.zeros(T)
    Rt = np.zeros(T)
    vn = np.zeros(T, bool)
    vt = np.zeros(T, bool)
    probe = np.zeros(T, bool)
    fb = np.zeros(T, bool)
    sw = np.zeros(T)
    asks = []
    steps = []
    t0 = time.perf_counter()
    for t in range(T):
        k = t % STEPS_PER_DAY
        if k == 0 and hasattr(agent, "set_forecast"):
            agent.set_forecast(plant.forecast_day)
        if regime is not None and t == regime["t"]:
            plant.regime_change(**regime["net"])
            if hasattr(agent, "on_regime"):
                agent.on_regime(plant.spec)
            if regime.get("ticket") is not None:
                expert_out = regime["ticket"]
                is_ticket = True
                if hasattr(agent, "offer_text"):
                    agent.offer_text()
        a, sp, do_ask, info = agent.act(obs)
        busy_ms = ask.latency_ms if do_ask else 0.0
        obs, r, diag = plant.step(a, sp, busy_ms)
        if do_ask:
            r -= ask.price
            asks.append(t)
            agent.receive_expert(expert_out, is_ticket=is_ticket)
        agent.observe(obs, busy_ms / (DT * 1000.0))
        R[t], Rn[t], Rt[t] = r, diag["r_net"], diag["r_th"]
        vn[t], vt[t] = diag["viol_net"], diag["viol_th"]
        probe[t] = bool(info.get("probe", False))
        fb[t] = bool(info.get("fallback", False))
        sw[t] = diag.get("switch", 0.0)
        if record_steps:
            steps.append({"t": t, "a": int(a), "sp": sp, "r": r,
                          "lat": obs["latency_ms"], "pdr": obs["pdr"],
                          "t_air": obs["t_air"], "occ": diag["occ"],
                          "probe": probe[t], "ask": do_ask})
    days_r = R.reshape(days, -1).sum(1)
    out = {
        "plant": spec.name, "combo": list(spec.combo), "ool": spec.ool,
        "seed": seed, "method": getattr(agent, "name", type(agent).__name__),
        "return": float(R.sum()), "return_per_day": days_r.tolist(),
        "r_net": float(Rn.sum()), "r_th": float(Rt.sum()),
        "viol_net": int(vn.sum()), "viol_th": int(vt.sum()),
        "viol_per_day": (vn | vt).reshape(days, -1).sum(1).tolist(),
        "probe_steps": int(probe.sum()), "fallback_steps": int(fb.sum()),
        "switch_cost": float(sw.sum()), "n_switches": int((sw > 0).sum()),
        "asks": asks, "ask_j": float(len(asks) * ask.energy_j),
        "th_probes": int(getattr(agent, "n_th_probes", 0)),
        "misspec_flag": bool(getattr(agent, "misspec", False)),
        "misspec_alarms": int(getattr(agent, "n_misspec_alarms", 0)),
        "cusum_max": float(getattr(agent, "cusum_max", 0.0)),
        "th_cusum_max": float(getattr(agent, "th_cusum_max", 0.0)),
        "th_alarms": int(getattr(agent, "n_th_alarms", 0)),
        "kappa": float(getattr(agent, "kappa", 1.0)),
        "wall_s": time.perf_counter() - t0,
    }
    if hasattr(agent, "nb") and getattr(agent.nb, "n", 0) > 1:
        out["final_factor_post"] = agent.nb.factor_marginals()
        out["final_mass_post"] = agent.tb.mass_post()
    if record_steps:
        out["_steps"] = steps
    return out

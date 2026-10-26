"""Compact text serialisation of the gateway's causal state.

The small model never sees the raw multivariate series. It sees a token-cheap
description of what the statistical stage already found:

  * the variable list,
  * the PCMCI+ skeleton as `parent>child strength/significance` items,
  * detected change points and the current drift / attack indicators,
  * device metadata and topology,
  * the last k KPI readings at three significant digits,
  * the candidate interventions it is allowed to choose between.

Keeping the prompt short is not cosmetic. Prompt length sets both the latency
and the Joule cost of every invocation, which the meta-controller then has to
pay for, so the deployed serialisation uses short aliases, three significant
digits and a one-character significance code. `compact=False` keeps the verbose
rendering for the prompt ablation, which measures what the extra tokens buy.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

from ..config import ACTUATORS, VAR_DESC, VARS

SYSTEM_PROMPT = (
    "Causal world model for an IoT gateway. G is a statistical skeleton that "
    "may hold spurious edges from unobserved confounders. Reply with ONE JSON "
    "object: do, predicted_ATE, confounders, spurious_edges, safety, explain."
)

ALIAS = {"occupancy": "occ", "outdoor_temp": "otmp", "indoor_temp": "itmp",
         "humidity": "hum", "hvac_power": "hvac", "wifi_rssi": "rssi",
         "latency_ms": "lat", "energy_kwh": "kwh", "it_energy_j": "itj",
         "tariff": "tar", "energy_cost": "cost",
         "hvac_setpoint": "sp", "sample_rate_hz": "srt", "duty_cycle": "dc",
         "offload_frac": "off"}
UNALIAS = {v: k for k, v in ALIAS.items()}


def _a(v: str) -> str:
    return ALIAS.get(v, v)


def _p_code(v: float) -> str:
    """One-character significance code in place of a full p-value."""
    if v < 1e-6:
        return "0"
    if v < 1e-3:
        return "3"
    if v < 0.01:
        return "2"
    if v < 0.05:
        return "1"
    return "9"


def _q(x: float, nd: int = 2) -> str:
    return f"{x:.{nd}f}".rstrip("0").rstrip(".") or "0"


def _sig(x: float, nd: int = 3) -> str:
    """Three significant digits, no trailing zeros, no leading zero."""
    if x == 0 or not np.isfinite(x):
        return "0"
    s = f"{x:.{nd}g}"
    if s.startswith("0."):
        s = s[1:]
    elif s.startswith("-0."):
        s = "-" + s[2:]
    return s


def summarise_graph(A: np.ndarray, pvals: Optional[np.ndarray] = None,
                    scores: Optional[np.ndarray] = None,
                    var_names: Sequence[str] = VARS, max_edges: int = 16,
                    compact: bool = True) -> str:
    """Render an adjacency matrix as edges ordered by strength."""
    idx = []
    n = len(var_names)
    for i in range(n):
        for j in range(n):
            if i == j or A[i, j] <= 0:
                continue
            s = float(scores[i, j]) if scores is not None else 1.0
            idx.append((s, i, j))
    idx.sort(reverse=True)
    items = []
    for s, i, j in idx[:max_edges]:
        u, v = var_names[i], var_names[j]
        if compact:
            p = f"/{_p_code(float(pvals[i, j]))}" if pvals is not None else ""
            items.append(f"{_a(u)}>{_a(v)} {_q(s)}{p}")
        else:
            p = f", p={float(pvals[i, j]):.3g}" if pvals is not None else ""
            items.append(f"{u}->{v} (rho={s:.2f}{p})")
    if not items:
        return "-" if compact else "none"
    return (" " if compact else "; ").join(items)


def change_points(X: np.ndarray, var_names: Sequence[str] = VARS,
                  z_thresh: float = 3.0, min_seg: int = 12) -> List[str]:
    """Cheap CUSUM-style change-point flags, one entry per shifted variable."""
    out = []
    X = np.asarray(X)
    if X.ndim != 2 or X.shape[0] < 2 * min_seg:
        return out
    half = X.shape[0] // 2
    for j, v in enumerate(var_names[:X.shape[1]]):
        c = X[:, j]
        if not np.isfinite(c).all():
            continue
        a, b = c[:half], c[half:]
        sd = np.sqrt((a.var() + b.var()) / 2) + 1e-8
        z = abs(b.mean() - a.mean()) / (sd / np.sqrt(min(len(a), len(b))) + 1e-8)
        if z > z_thresh:
            out.append(f"{v} shift {'up' if b.mean() > a.mean() else 'down'} "
                       f"(z={z:.1f})")
    return out


def build_prompt(A: np.ndarray, X: np.ndarray, *,
                 pvals: Optional[np.ndarray] = None,
                 scores: Optional[np.ndarray] = None,
                 var_names: Sequence[str] = VARS,
                 last_k: int = 3,
                 device_meta: Optional[Dict] = None,
                 candidates: Optional[List[Dict[str, float]]] = None,
                 regime: Optional[Dict] = None,
                 compact: bool = True,
                 include_graph: bool = True,
                 include_pvals: bool = True,
                 include_changepoints: bool = True,
                 include_kpis: bool = True) -> str:
    """Assemble the user-side prompt handed to the small model."""
    obs = list(var_names)
    X = np.asarray(X)
    parts: List[str] = []

    if not compact:
        parts.append("VARS: " + "; ".join(f"{v}={VAR_DESC.get(v, v)}"
                                          for v in obs))
        if include_graph:
            parts.append("STAT_GRAPH: " + summarise_graph(
                A, pvals if include_pvals else None, scores, obs,
                compact=False))
        if include_changepoints:
            cp = change_points(X, obs)
            parts.append("CHANGE_POINTS: " + ("; ".join(cp) if cp else "none"))
        if include_kpis and X.size:
            tail = X[-last_k:]
            rows = [" ".join(_q(float(tail[t, j])) for j in range(tail.shape[1]))
                    for t in range(tail.shape[0])]
            parts.append(f"LAST_{tail.shape[0]}_KPI[{','.join(obs)}]: "
                         + " ; ".join(rows))
        if device_meta:
            parts.append("TOPOLOGY: " + "; ".join(
                f"{k}={v}" for k, v in device_meta.items()))
        if regime:
            parts.append("REGIME: " + "; ".join(f"{k}={v}"
                                                for k, v in regime.items()))
        parts.append("ACTUATORS: " + "; ".join(
            f"{k} in [{_q(lo)},{_q(hi)}]" for k, (lo, hi) in ACTUATORS.items()))
        if candidates:
            parts.append("CANDIDATES: " + " | ".join(
                ",".join(f"{k}={_q(v)}" for k, v in c.items())
                for c in candidates))
        parts.append(
            "TASK: list confounders and spurious edges in STAT_GRAPH, then "
            "choose one intervention from CANDIDATES and predict its ATE on "
            "energy_cost, indoor_temp, latency_ms, pmv. JSON only.")
        return "\n".join(parts)

    # ------------------------------------------------------------- compact
    parts.append("V " + ",".join(_a(v) for v in obs))
    if include_graph:
        parts.append("G " + summarise_graph(
            A, pvals if include_pvals else None, scores, obs, compact=True))
    if include_changepoints:
        cp = change_points(X, obs)
        parts.append("CP " + (" ".join(
            f"{_a(c.split()[0])}{'+' if ' up ' in c else '-'}" for c in cp)
            if cp else "-"))
    if include_kpis and X.size:
        tail = X[-last_k:]
        rows = [",".join(_sig(float(tail[t, j])) for j in range(tail.shape[1]))
                for t in range(tail.shape[0])]
        parts.append("K " + " | ".join(rows))
    bits = []
    if device_meta:
        bits.append(f"dev{device_meta.get('n_devices', 24)} mqtt wifi")
    if regime:
        bits.append(f"drift{1 if regime.get('drift') == 'yes' else 0}"
                    f" atk{1 if regime.get('attack') == 'yes' else 0}"
                    f" occsens{0 if regime.get('occupancy_sensor') == 'absent' else 1}")
    if bits:
        parts.append("D " + " ".join(bits))
    parts.append("A sp[18,27] srt[.1,1] dc[.1,1] off[0,1]")
    if candidates:
        parts.append("C " + " | ".join(
            ",".join(f"{_a(k)}={_sig(v)}" for k, v in c.items())
            for c in candidates))
    parts.append("Q repair G, pick one C, give ATE on cost,itmp,lat,pmv")
    return "\n".join(parts)

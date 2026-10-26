"""Preprocessing applied before every causal-discovery method, identically.

Two gateway KPIs are heavy-tailed by construction -- an M/M/1 sojourn time
behaves like 1/(mu - lambda) and blows up near saturation, and gateway IT
energy inherits that tail through compute contention. Feeding raw values to a
partial-correlation test lets a handful of saturated steps dominate every
statistic. We therefore apply a signed log1p to those two channels only, and
apply it for *all* methods so the comparison stays fair.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from ..config import VARS

HEAVY_TAILED = ("latency_ms", "it_energy_j")


def preprocess_for_discovery(X: np.ndarray, var_names: Sequence[str] = None,
                             log_heavy_tailed: bool = True,
                             standardise: bool = True) -> np.ndarray:
    names = list(var_names) if var_names is not None else list(VARS)
    Z = np.asarray(X, np.float64).copy()
    if log_heavy_tailed:
        for v in HEAVY_TAILED:
            if v in names:
                j = names.index(v)
                Z[:, j] = np.sign(Z[:, j]) * np.log1p(np.abs(Z[:, j]))
    Z = np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0)
    if standardise:
        Z = (Z - Z.mean(0)) / (Z.std(0) + 1e-8)
    return Z

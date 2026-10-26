"""Tabular ATE regressor used by the CPU fallback policy.

Kept in the package rather than in the training script so the pickled model
resolves its class on load.
"""
from __future__ import annotations

import numpy as np

ATE_KEYS = ["energy_cost", "indoor_temp", "latency_ms", "pmv"]
CTX_KPIS = ["indoor_temp", "outdoor_temp", "pmv", "latency_ms", "pdr",
            "occupancy", "humidity", "hvac_power", "it_energy_j",
            "hvac_setpoint", "sample_rate_hz", "duty_cycle", "offload_frac",
            "queue_util", "suspicion", "tariff"]


class MultiATE:
    """One gradient-boosted regressor per ATE target behind a single predict()."""

    def __init__(self, models, targets=None, ctx=None):
        self.models = models
        self.targets = list(targets or ATE_KEYS)
        self.ctx = list(ctx or CTX_KPIS)

    def predict(self, X):
        return np.stack([m.predict(X) for m in self.models], axis=1)

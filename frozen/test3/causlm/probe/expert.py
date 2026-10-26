"""A text-reading expert as a calibrated, noisy observation of the plant.

The on-device language model reads the commissioning text and returns, for
each mechanism factor, a probability that the factor takes value 1. The agent
does not take that number at face value. It treats the model's logit `z` as an
observation whose likelihood under a hypothesis with factor value `v` is

    p(z | v=1) proportional to sigmoid(tau_F * z),  p(z | v=0) to sigmoid(-tau_F * z),

with a per-factor temperature `tau_F` fitted on held-out validation plants by
minimising negative log-likelihood (temperature scaling). An expert that is
overconfident gets tau < 1 and is trusted less; an uninformative one gets
tau ~ 0 and is, correctly, worth nothing.

To value an invocation *before* making it, the agent needs the distribution
of what the expert would say. That is the empirical bank of validation logits,
conditioned on the true factor value: sampling from it reproduces the expert's
real error profile, including the plants whose text never mentions a factor.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

import numpy as np
from scipy.optimize import minimize_scalar

FACTORS = ("A", "B", "C", "D")


def _log_sigmoid(x):
    return -np.logaddexp(0.0, -x)


@dataclass
class ExpertModel:
    name: str
    tau: Dict[str, float]
    bank: Dict[str, Dict[int, np.ndarray]]     # factor -> value -> logits

    def loglik(self, F: str, z: Optional[float], vals: np.ndarray) -> np.ndarray:
        if z is None or not np.isfinite(z):
            return np.zeros(len(vals))
        t = self.tau.get(F, 1.0) * float(z)
        return np.where(np.asarray(vals) == 1, _log_sigmoid(t), _log_sigmoid(-t))

    def sample_output(self, truth: Dict[str, int], rng) -> Dict[str, float]:
        out = {}
        for F in FACTORS:
            b = self.bank[F][int(truth[F])]
            out[F] = float(b[rng.integers(len(b))]) if len(b) else 0.0
        return out

    def to_json(self, path: Path):
        Path(path).write_text(json.dumps({
            "name": self.name, "tau": self.tau,
            "bank": {F: {str(v): self.bank[F][v].tolist() for v in (0, 1)}
                     for F in FACTORS}}))

    @staticmethod
    def from_json(path: Path) -> "ExpertModel":
        d = json.loads(Path(path).read_text())
        bank = {F: {int(v): np.asarray(x, float) for v, x in d["bank"][F].items()}
                for F in FACTORS}
        return ExpertModel(d["name"], {k: float(v) for k, v in d["tau"].items()},
                           bank)


def fit_temperature(z: np.ndarray, y: np.ndarray) -> float:
    """Temperature tau minimising the NLL of y in {0,1} given logits z."""
    z = np.asarray(z, float)
    y = np.asarray(y, float)

    def nll(t):
        return -np.mean(y * _log_sigmoid(t * z) + (1 - y) * _log_sigmoid(-t * z))
    res = minimize_scalar(nll, bounds=(0.0, 5.0), method="bounded")
    return float(res.x)


def build_expert(name: str, z_val: Dict[str, np.ndarray],
                 y_val: Dict[str, np.ndarray], calibrate: bool = True
                 ) -> ExpertModel:
    """From validation logits and truths, the calibrated expert model."""
    tau, bank = {}, {}
    for F in FACTORS:
        z, y = np.asarray(z_val[F], float), np.asarray(y_val[F], int)
        tau[F] = fit_temperature(z, y) if calibrate else 1.0
        bank[F] = {v: z[y == v] for v in (0, 1)}
    return ExpertModel(name, tau, bank)


def expected_calibration_error(p: np.ndarray, y: np.ndarray, bins: int = 10):
    p = np.asarray(p, float)
    y = np.asarray(y, float)
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi) if hi < 1 else (p >= lo) & (p <= hi)
        if m.any():
            ece += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(ece)

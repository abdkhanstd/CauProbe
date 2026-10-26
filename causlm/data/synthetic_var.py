"""Tigramite-style synthetic structural VAR processes with a known lagged DAG.

Used as the controlled sanity benchmark for the causal-discovery stage: the
ground-truth summary graph is exactly the union of the non-zero lagged
coefficients, so F1/SHD are unambiguous.
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np


def make_var_dataset(n_vars: int = 8, T: int = 1500, tau_max: int = 3,
                     density: float = 0.22, coef_range=(0.25, 0.55),
                     nonlinear: float = 0.0, noise_std: float = 0.35,
                     n_hidden: int = 0, seed: int = 0
                     ) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Simulate a stable structural VAR.

    Returns
    -------
    X : (T, n_vars) observed series (hidden confounders are dropped)
    G : (n_vars, n_vars) summary adjacency, G[i, j] = 1 iff i -> j at some lag
    info : link coefficients and the identity of the hidden variables
    """
    rng = np.random.default_rng(seed)
    n_total = n_vars + n_hidden
    links: Dict[Tuple[int, int, int], float] = {}

    # self-loops keep the process autocorrelated, which is what makes naive
    # Granger tests fail
    for i in range(n_total):
        links[(i, i, 1)] = rng.uniform(0.25, 0.5)

    for j in range(n_total):
        for i in range(n_total):
            if i == j:
                continue
            if rng.random() < density:
                lag = int(rng.integers(1, tau_max + 1))
                c = rng.uniform(*coef_range) * rng.choice([-1.0, 1.0])
                links[(i, j, lag)] = c

    # hidden variables act as confounders: force them to have >= 2 children
    for h in range(n_vars, n_total):
        children = rng.choice(n_vars, size=min(3, n_vars), replace=False)
        for c in children:
            links[(h, int(c), int(rng.integers(1, tau_max + 1)))] = \
                rng.uniform(0.4, 0.7) * rng.choice([-1.0, 1.0])

    # rescale to keep the spectral radius below 1
    for lag in range(1, tau_max + 1):
        A = np.zeros((n_total, n_total))
        for (i, j, l), c in links.items():
            if l == lag:
                A[j, i] = c
        r = max(abs(np.linalg.eigvals(A)))
        if r > 0.85:
            for k in list(links):
                if k[2] == lag:
                    links[k] *= 0.85 / r

    burn = 200
    X = np.zeros((T + burn, n_total))
    for t in range(tau_max, T + burn):
        for j in range(n_total):
            v = 0.0
            for (i, jj, lag), c in links.items():
                if jj != j:
                    continue
                z = X[t - lag, i]
                v += c * (z + nonlinear * np.tanh(z) ** 2)
            X[t, j] = v + rng.normal(0, noise_std)
    X = X[burn:]

    G = np.zeros((n_vars, n_vars), dtype=np.float32)
    for (i, j, _), c in links.items():
        if i < n_vars and j < n_vars and abs(c) > 1e-8 and i != j:
            G[i, j] = 1.0

    info = {"links": {f"{i}->{j}@{l}": float(c) for (i, j, l), c in links.items()},
            "n_hidden": n_hidden, "hidden_idx": list(range(n_vars, n_total)),
            "tau_max": tau_max, "nonlinear": nonlinear}
    return X[:, :n_vars].astype(np.float32), G, info

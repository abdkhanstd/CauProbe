"""Statistical ATE estimation from an observational window plus a graph.

Given a (possibly wrong) DAG, the treatment's parents form a valid backdoor
adjustment set, so a linear model of the outcome on the treatment and that set
identifies the effect. This is the natural non-LLM comparison for the small
model's `predicted_ATE` field, and running it on the *true* graph gives the
oracle upper bound that separates "the graph was wrong" from "the estimator
was wrong".
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np


def _parents(G: np.ndarray, j: int) -> List[int]:
    return [int(i) for i in np.flatnonzero(G[:, j] > 0) if i != j]


def backdoor_ate(X: np.ndarray, names: Sequence[str], G: np.ndarray,
                 treat: str, outcomes: Sequence[str], v_do: float,
                 v_ref: float, lag: int = 1, ridge: float = 1e-3
                 ) -> Dict[str, float]:
    """Linear backdoor adjustment for do(treat = v_do) vs do(treat = v_ref).

    Regresses each outcome at time t on the treatment and its parents at
    t - `lag`; the treatment coefficient times the contrast is the estimate.
    """
    names = list(names)
    if treat not in names:
        return {o: 0.0 for o in outcomes}
    ti = names.index(treat)
    Z = np.asarray(X, float)
    mu, sd = Z.mean(0), Z.std(0) + 1e-8
    pa = _parents(G, ti)
    cols = [ti] + [p for p in pa if p != ti]
    A = Z[:-lag][:, cols]
    A = (A - mu[cols]) / sd[cols]
    A = np.c_[A, np.ones(len(A))]
    out = {}
    for o in outcomes:
        if o not in names:
            out[o] = 0.0
            continue
        oi = names.index(o)
        y = Z[lag:, oi]
        try:
            AtA = A.T @ A + ridge * np.eye(A.shape[1])
            beta = np.linalg.solve(AtA, A.T @ y)
        except np.linalg.LinAlgError:
            out[o] = 0.0
            continue
        out[o] = float(beta[0] * (v_do - v_ref) / sd[ti])
    return out


def naive_regression_ate(X: np.ndarray, names: Sequence[str], treat: str,
                         outcomes: Sequence[str], v_do: float, v_ref: float,
                         lag: int = 1) -> Dict[str, float]:
    """No adjustment at all: the correlational answer, kept as a floor."""
    n = len(names)
    G = np.zeros((n, n))
    return backdoor_ate(X, names, G, treat, outcomes, v_do, v_ref, lag)


def ate_metrics(pred: Dict[str, float], true: Dict[str, float],
                scale: Optional[Dict[str, float]] = None) -> Dict[str, float]:
    """MAE / RMSE / sign agreement, both raw and scale-normalised."""
    keys = [k for k in true if k in pred]
    if not keys:
        return {}
    p = np.array([pred[k] for k in keys], float)
    t = np.array([true[k] for k in keys], float)
    out = {"mae": float(np.abs(p - t).mean()),
           "rmse": float(np.sqrt(((p - t) ** 2).mean()))}
    sig = (np.sign(p) == np.sign(t)) | ((np.abs(t) < 1e-4) & (np.abs(p) < 1e-4))
    out["sign_acc"] = float(sig.mean())
    if scale:
        s = np.array([max(scale.get(k, 1.0), 1e-8) for k in keys])
        out["nmae"] = float((np.abs(p - t) / s).mean())
    return out

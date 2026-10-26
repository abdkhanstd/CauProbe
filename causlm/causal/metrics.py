"""Graph-recovery metrics: precision / recall / F1 / SHD / AUROC / SID-lite."""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np


def _offdiag_mask(n: int) -> np.ndarray:
    m = ~np.eye(n, dtype=bool)
    return m


def shd(pred: np.ndarray, true: np.ndarray, ignore_diag: bool = True) -> int:
    """Structural Hamming distance between two directed adjacency matrices.

    A reversed edge costs 1 (not 2): it is one operation away from correct.
    """
    p = (np.asarray(pred) > 0).astype(int).copy()
    t = (np.asarray(true) > 0).astype(int).copy()
    if ignore_diag:
        np.fill_diagonal(p, 0)
        np.fill_diagonal(t, 0)
    n = p.shape[0]
    d = 0
    seen = set()
    for i in range(n):
        for j in range(n):
            if i == j or (j, i) in seen:
                continue
            seen.add((i, j))
            pe, pr = p[i, j], p[j, i]
            te, tr = t[i, j], t[j, i]
            if (pe, pr) == (te, tr):
                continue
            if (pe, pr) == (tr, te) and (te + tr) == 1:
                d += 1            # reversed
            else:
                d += abs(pe - te) + abs(pr - tr)
    return int(d)


def graph_metrics(pred: np.ndarray, true: np.ndarray,
                  scores: Optional[np.ndarray] = None,
                  ignore_diag: bool = True) -> Dict[str, float]:
    p = (np.asarray(pred) > 0).astype(int)
    t = (np.asarray(true) > 0).astype(int)
    n = p.shape[0]
    m = _offdiag_mask(n) if ignore_diag else np.ones_like(p, bool)
    pv, tv = p[m], t[m]
    tp = int((pv & tv).sum())
    fp = int((pv & ~tv).sum())
    fn = int((~pv & tv).sum())
    tn = int((~pv & ~tv).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    out = {"precision": prec, "recall": rec, "f1": f1,
           "shd": float(shd(p, t, ignore_diag)), "tp": tp, "fp": fp,
           "fn": fn, "tn": tn,
           "fdr": fp / (tp + fp) if tp + fp else 0.0,
           "tpr": rec,
           "n_pred_edges": int(pv.sum()), "n_true_edges": int(tv.sum())}
    if scores is not None:
        out["auroc"] = _auroc(np.asarray(scores)[m], tv)
        out["auprc"] = _auprc(np.asarray(scores)[m], tv)
    return out


def _auroc(s: np.ndarray, y: np.ndarray) -> float:
    if y.sum() == 0 or y.sum() == len(y):
        return float("nan")
    order = np.argsort(-s)
    y = y[order]
    tps = np.cumsum(y)
    fps = np.cumsum(1 - y)
    tpr = tps / max(y.sum(), 1)
    fpr = fps / max((1 - y).sum(), 1)
    return float(np.trapz(np.r_[0, tpr, 1], np.r_[0, fpr, 1]))


def _auprc(s: np.ndarray, y: np.ndarray) -> float:
    if y.sum() == 0:
        return float("nan")
    order = np.argsort(-s)
    y = y[order]
    tps = np.cumsum(y)
    prec = tps / np.arange(1, len(y) + 1)
    rec = tps / y.sum()
    return float(np.trapz(np.r_[1, prec], np.r_[0, rec]))


def adjacency_from_scores(scores: np.ndarray, k: Optional[int] = None,
                          thresh: Optional[float] = None) -> np.ndarray:
    """Binarise a score matrix either by top-k or by a threshold."""
    s = np.asarray(scores, float).copy()
    np.fill_diagonal(s, -np.inf)
    A = np.zeros_like(s)
    if k is not None:
        flat = np.argsort(-s, axis=None)[:k]
        A.flat[flat] = 1.0
    else:
        A = (s > (thresh if thresh is not None else 0.0)).astype(float)
    np.fill_diagonal(A, 0.0)
    return A

"""Temporal causal-discovery baselines, all returning (adjacency, scores, info).

Convention everywhere: `A[i, j] == 1` means "variable i causes variable j"
in the summary graph (aggregated over lags 1..tau_max).

Implemented / wrapped
---------------------
pcmci+        Runge et al. -- tigramite `run_pcmciplus`, the main statistical
              skeleton used by CauSLM-Edge.
granger       Multivariate Granger causality via VAR Wald tests (statsmodels).
var-lingam    VARLiNGAM (causal-learn), non-Gaussian linear SEM.
dynotears     Continuous NOTEARS objective extended to lagged data, solved with
              an augmented-Lagrangian loop (our own implementation).
pc            PC with Fisher-Z on the time-lagged design matrix (causal-learn).
tf-prior      Transformer forecaster + gradient attribution + attention masking
              for prior knowledge, following the design of Huang et al. 2025
              ("Transforming Causality", arXiv:2508.15928). Re-implemented here
              because no reference code was released.
"""
from __future__ import annotations

import warnings
from typing import Dict, Optional, Tuple

import numpy as np

warnings.filterwarnings("ignore")


# ------------------------------------------------------------------ PCMCI+
def run_pcmci_plus(X: np.ndarray, tau_max: int = 3, pc_alpha: float = 0.05,
                   ci_test: str = "parcorr", var_names=None, tau_min: int = 0,
                   return_pvals: bool = True, **_
                   ) -> Tuple[np.ndarray, np.ndarray, Dict]:
    from tigramite import data_processing as pp
    from tigramite.pcmci import PCMCI

    N = X.shape[1]
    if ci_test == "parcorr":
        from tigramite.independence_tests.parcorr import ParCorr
        cond = ParCorr(significance="analytic")
    elif ci_test == "robust_parcorr":
        from tigramite.independence_tests.robust_parcorr import RobustParCorr
        cond = RobustParCorr(significance="analytic")
    elif ci_test == "gpdc":
        from tigramite.independence_tests.gpdc import GPDC
        cond = GPDC(significance="analytic")
    elif ci_test == "cmiknn":
        from tigramite.independence_tests.cmiknn import CMIknn
        cond = CMIknn(significance="shuffle_test", sig_samples=200)
    else:
        raise ValueError(ci_test)

    df = pp.DataFrame(np.asarray(X, float),
                      var_names=var_names or [f"x{i}" for i in range(N)])
    pcmci = PCMCI(dataframe=df, cond_ind_test=cond, verbosity=0)
    res = pcmci.run_pcmciplus(tau_min=tau_min, tau_max=tau_max,
                              pc_alpha=pc_alpha)

    g = res["graph"]                       # (N, N, tau_max+1) of link strings
    val = np.abs(np.nan_to_num(res["val_matrix"]))
    pmat = np.nan_to_num(res["p_matrix"], nan=1.0)

    A = np.zeros((N, N), np.float32)
    S = np.zeros((N, N), np.float32)
    P = np.ones((N, N), np.float32)
    directed = ("-->", "o->", "+->")
    undirected = ("o-o", "x-x", "---", "+-+")
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            for t in range(tau_min, tau_max + 1):
                lnk = g[i, j, t]
                if lnk in directed:
                    A[i, j] = 1.0
                elif t == 0 and lnk in undirected:
                    # PCMCI+ leaves some contemporaneous links unoriented; the
                    # skeleton still carries the adjacency, which is what the
                    # SLM is asked to orient.
                    A[i, j] = max(A[i, j], 0.5)
            lo = max(tau_min, 0)
            S[i, j] = val[i, j, lo:tau_max + 1].max()
            P[i, j] = pmat[i, j, lo:tau_max + 1].min()
    A = (A > 0).astype(np.float32)
    info = {"method": "pcmci+", "ci_test": ci_test, "pc_alpha": pc_alpha,
            "tau_max": tau_max, "tau_min": tau_min}
    if return_pvals:
        info["pvals"] = P
        info["lag_graph"] = g
        info["val_matrix"] = val
    return A, S, info


# ----------------------------------------------------------------- Granger
def run_granger(X: np.ndarray, tau_max: int = 3, alpha: float = 0.05,
                **_) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Multivariate Granger causality: Wald test on the VAR coefficients."""
    from statsmodels.tsa.api import VAR
    X = np.asarray(X, float)
    N = X.shape[1]
    model = VAR(X)
    res = model.fit(maxlags=tau_max, trend="c")
    A = np.zeros((N, N), np.float32)
    S = np.zeros((N, N), np.float32)
    P = np.ones((N, N), np.float32)
    for j in range(N):
        for i in range(N):
            if i == j:
                continue
            try:
                t = res.test_causality(j, [i], kind="wald")
                P[i, j] = t.pvalue
                S[i, j] = t.test_statistic
                A[i, j] = float(t.pvalue < alpha)
            except Exception:
                pass
    return A, S, {"method": "granger", "alpha": alpha, "tau_max": tau_max,
                  "pvals": P}


# --------------------------------------------------------------- VARLiNGAM
def run_var_lingam(X: np.ndarray, tau_max: int = 3, thresh: float = 0.05,
                   **_) -> Tuple[np.ndarray, np.ndarray, Dict]:
    from causallearn.search.FCMBased.lingam import VARLiNGAM
    X = np.asarray(X, float)
    N = X.shape[1]
    m = VARLiNGAM(lags=tau_max, criterion=None)
    m.fit(X)
    B = np.array(m.adjacency_matrices_)      # (lags+1, N, N), B[k][j, i]: i->j
    S = np.abs(B[1:]).max(axis=0).T          # lagged only -> (i, j)
    np.fill_diagonal(S, 0.0)
    A = (S > thresh * S.max()).astype(np.float32) if S.max() > 0 else np.zeros_like(S)
    return A.astype(np.float32), S.astype(np.float32), {
        "method": "var-lingam", "tau_max": tau_max, "thresh": thresh}


# --------------------------------------------------------------- DYNOTEARS
def run_dynotears(X: np.ndarray, tau_max: int = 3, lambda_w: float = 0.05,
                  lambda_a: float = 0.05, max_iter: int = 80,
                  h_tol: float = 1e-8, w_thresh: float = 0.05,
                  **_) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """DYNOTEARS-style continuous structure learning for dynamic Bayes nets.

    Minimises  ||X - X W - Y A||_F^2 / 2n + l1 penalties  subject to the
    NOTEARS acyclicity constraint h(W) = tr(e^{W o W}) - d = 0, solved with an
    augmented-Lagrangian outer loop and L-BFGS-B inner solves. We report the
    *lagged* block A (aggregated over lags), which is the temporal analogue of
    the summary graph used by the other methods.
    """
    from scipy.linalg import expm
    from scipy.optimize import minimize

    X = np.asarray(X, float)
    T, d = X.shape
    p = tau_max
    Xc = X[p:]
    Y = np.concatenate([X[p - k - 1:T - k - 1] for k in range(p)], axis=1)  # (n, d*p)
    n = Xc.shape[0]
    Xc = (Xc - Xc.mean(0)) / (Xc.std(0) + 1e-8)
    Y = (Y - Y.mean(0)) / (Y.std(0) + 1e-8)
    dp = d * p

    def unpack(w):
        W = (w[:d * d] - w[d * d:2 * d * d]).reshape(d, d)
        A = (w[2 * d * d:2 * d * d + dp * d]
             - w[2 * d * d + dp * d:]).reshape(dp, d)
        return W, A

    def loss(w, rho, alpha):
        W, A = unpack(w)
        R = Xc - Xc @ W - Y @ A
        l = 0.5 / n * (R ** 2).sum()
        E = expm(W * W)
        h = np.trace(E) - d
        obj = (l + lambda_w * np.abs(W).sum() + lambda_a * np.abs(A).sum()
               + 0.5 * rho * h * h + alpha * h)
        GW = -1.0 / n * Xc.T @ R + (rho * h + alpha) * E.T * W * 2
        GA = -1.0 / n * Y.T @ R
        g = np.concatenate([GW.ravel() + lambda_w, -GW.ravel() + lambda_w,
                            GA.ravel() + lambda_a, -GA.ravel() + lambda_a])
        return obj, g

    w = np.zeros(2 * (d * d + dp * d))
    rho, alpha, h_prev = 1.0, 0.0, np.inf
    bounds = [(0, None)] * len(w)
    for _ in range(max_iter):
        sol = minimize(loss, w, args=(rho, alpha), jac=True, method="L-BFGS-B",
                       bounds=bounds, options={"maxiter": 120})
        w = sol.x
        W, _A = unpack(w)
        h = np.trace(expm(W * W)) - d
        alpha += rho * h
        if h > 0.25 * h_prev:
            rho *= 10
        h_prev = h
        if h <= h_tol or rho > 1e16:
            break

    W, A = unpack(w)
    Al = np.abs(A).reshape(p, d, d).max(axis=0)     # (i, j) lagged strength
    S = Al + np.abs(W)                              # contemporaneous + lagged
    np.fill_diagonal(S, 0.0)
    Ab = (S > w_thresh).astype(np.float32)
    return Ab, S.astype(np.float32), {"method": "dynotears", "tau_max": tau_max,
                                      "lambda_w": lambda_w, "h": float(h)}


# ---------------------------------------------------------------------- PC
def run_pc(X: np.ndarray, tau_max: int = 3, alpha: float = 0.05,
           **_) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """PC (Fisher-Z) on the lag-augmented design matrix, then read off i(t-k)->j(t)."""
    from causallearn.search.ConstraintBased.PC import pc
    X = np.asarray(X, float)
    T, d = X.shape
    p = tau_max
    Z = np.concatenate([X[p - k:T - k] for k in range(p + 1)], axis=1)  # lag0..lagp
    cg = pc(Z, alpha=alpha, indep_test="fisherz", show_progress=False)
    G = cg.G.graph                       # causal-learn: -1/1 endpoint encoding
    m = Z.shape[1]
    A = np.zeros((d, d), np.float32)
    S = np.zeros((d, d), np.float32)
    for a in range(m):
        for b in range(m):
            if a == b:
                continue
            # directed a -> b in causal-learn is G[b, a] == 1 and G[a, b] == -1
            directed = (G[b, a] == 1 and G[a, b] == -1)
            adjacent = (G[a, b] != 0 or G[b, a] != 0)
            la, va = divmod(a, d) if False else (a // d, a % d)
            lb, vb = b // d, b % d
            if va == vb:
                continue
            if la > 0 and lb == 0:       # past -> present
                if directed or adjacent:
                    A[va, vb] = 1.0
                    S[va, vb] = max(S[va, vb], 1.0 if directed else 0.5)
    return A, S, {"method": "pc", "alpha": alpha, "tau_max": tau_max}


# ---------------------------------------------------- Transformer + priors
def run_transformer_prior(X: np.ndarray, tau_max: int = 3, epochs: int = 220,
                          hidden: int = 96, heads: int = 4, layers: int = 2,
                          prior_forbidden: Optional[np.ndarray] = None,
                          density: Optional[float] = None, device: str = "cuda",
                          seed: int = 0, **_) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Transformer forecaster + gradient attribution, with forbidden-edge masking.

    Re-implementation of the published design (Huang, Yao & Divakaran 2025):
    a multi-layer Transformer forecasts x(t) from x(t-tau..t-1); causal
    structure and lags are read out by gradient attribution; user priors are
    injected as attention masks that persistently forbid links.
    """
    import torch
    import torch.nn as nn
    dev = device if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed)
    X = np.asarray(X, np.float32)
    T, d = X.shape
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Z = (X - mu) / sd
    xs = np.stack([Z[t - tau_max:t] for t in range(tau_max, T)])
    ys = Z[tau_max:]

    class TF(nn.Module):
        def __init__(self):
            super().__init__()
            self.inp = nn.Linear(d, hidden)
            self.pos = nn.Parameter(torch.randn(1, tau_max, hidden) * 0.02)
            enc = nn.TransformerEncoderLayer(hidden, heads, hidden * 2,
                                             batch_first=True, dropout=0.05)
            self.enc = nn.TransformerEncoder(enc, layers)
            self.out = nn.Linear(hidden * tau_max, d)

        def forward(self, x):
            h = self.enc(self.inp(x) + self.pos)
            return self.out(h.reshape(x.shape[0], -1))

    net = TF().to(dev)
    xt = torch.from_numpy(xs).to(dev)
    yt = torch.from_numpy(ys).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    n = len(xt)
    for _ in range(epochs):
        perm = torch.randperm(n, device=dev)
        for i in range(0, n, 256):
            idx = perm[i:i + 256]
            opt.zero_grad()
            nn.functional.mse_loss(net(xt[idx]), yt[idx]).backward()
            opt.step()
        sch.step()

    net.eval()
    probe = xt[torch.randperm(n, device=dev)[:min(2048, n)]].clone().requires_grad_(True)
    S = np.zeros((d, d), np.float32)
    lag_pick = np.zeros((d, d), np.int32)
    for j in range(d):
        if probe.grad is not None:
            probe.grad = None
        g, = torch.autograd.grad(net(probe)[:, j].sum(), probe, retain_graph=True)
        gm = g.abs().mean(0).cpu().numpy()          # (tau, d)
        S[:, j] = gm.max(0)
        lag_pick[:, j] = tau_max - gm.argmax(0)
    np.fill_diagonal(S, 0.0)
    if prior_forbidden is not None:
        S = S * (1.0 - np.asarray(prior_forbidden, np.float32))
    Sn = S / (S.max(axis=0, keepdims=True) + 1e-12)
    if density is None:
        thr = np.quantile(Sn[~np.eye(d, dtype=bool)], 0.72)
        A = (Sn > thr).astype(np.float32)
    else:
        k = int(round(density * d * (d - 1)))
        from .metrics import adjacency_from_scores
        A = adjacency_from_scores(Sn, k=k)
    np.fill_diagonal(A, 0.0)
    return A, Sn.astype(np.float32), {"method": "tf-prior", "tau_max": tau_max,
                                      "lags": lag_pick}


DISCOVERY_METHODS = {
    "pcmci+": run_pcmci_plus,
    "granger": run_granger,
    "var-lingam": run_var_lingam,
    "dynotears": run_dynotears,
    "pc": run_pc,
    "tf-prior": run_transformer_prior,
}


def discover(method: str, X: np.ndarray, **kw):
    if method not in DISCOVERY_METHODS:
        raise KeyError(f"unknown method {method}; have {list(DISCOVERY_METHODS)}")
    return DISCOVERY_METHODS[method](X, **kw)

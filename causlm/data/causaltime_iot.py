"""CausalTime-IoT: an IoT-gateway instance of the CausalTime generation protocol.

The official CausalTime release (Cheng et al., ICLR 2024) ships three domains --
medical, pm25, traffic -- none of which contains smart-building / gateway
variables. We therefore reproduce the CausalTime *pipeline* on real IoT traces
so that the benchmark speaks the same variable vocabulary as the gateway
environment. The pipeline has four stages, following the original paper:

  1. fit a deep autoregressive model to real multivariate IoT observations;
  2. read a hypothesised causal graph out of the fitted network by
     gradient-based importance analysis, then prune it;
  3. **re-fit the network under a hard input mask given by that graph**, which
     makes the graph exact by construction rather than merely hypothesised;
  4. split each series into a causal term (the masked network), a residual term
     (an autoregressive process fitted with a normalising flow) and a noise
     term, and unroll the split model to generate new series.

The real seed traces are:

  indoor_temp, humidity  <- Intel Berkeley Lab motes (measured)
  occupancy              <- Intel Berkeley Lab light channel (measured, used as
                            an occupancy proxy after a workday-threshold map)
  outdoor_temp           <- TON_IoT weather telemetry (measured)
  hvac_power, energy_kwh <- UK-DALE house-1 aggregate demand (measured)
  wifi_rssi              <- Intel Lab per-step delivery ratio pushed through the
                            inverse of the standard logistic PDR(RSSI) link model
  latency_ms             <- Intel Lab per-step report count pushed through an
                            M/M/1 sojourn-time formula

Series from different deployments are aligned by sample index, not wall-clock
time; this is stated in the paper because the deployments are not contemporaneous.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from ..config import CT_IOT_VARS as VARS
from ..config import DATA_PROC
from ..config import N_CT_IOT as NVAR
from ..utils import save_json, set_seed
from .real_traces import (DatasetUnavailable, load_intel_lab, load_ton_iot,
                          load_ukdale)

CT_IOT_DIR = DATA_PROC / "causaltime_iot"


# ----------------------------------------------------------------- seed data
def build_seed_frame(n_steps: int = 8000, seed: int = 0
                     ) -> Tuple[np.ndarray, Dict]:
    """Assemble the real 8-channel seed matrix, column order = config.VARS."""
    rng = np.random.default_rng(seed)
    prov: Dict[str, str] = {}
    cols: Dict[str, np.ndarray] = {}

    intel, im = load_intel_lab()
    prov["intel_lab"] = im["source"]
    T = min(n_steps, len(intel))
    intel = intel.iloc[:T]

    cols["indoor_temp"] = intel["temperature"].to_numpy(np.float64)
    cols["humidity"] = intel["humidity"].to_numpy(np.float64)

    # Occupancy proxy from the light channel: lux normalised into a 0-12 count.
    light = intel["light"].to_numpy(np.float64)
    lq = np.quantile(light, [0.05, 0.95])
    occ = np.clip((light - lq[0]) / max(lq[1] - lq[0], 1e-6), 0, 1) * 12.0
    cols["occupancy"] = np.round(occ)

    # RSSI from the measured delivery ratio, inverting PDR = sigmoid(a*(r - m)).
    pdr = np.clip(intel["pdr_proxy"].to_numpy(np.float64), 0.02, 0.98)
    a, m = 0.22, -80.0
    cols["wifi_rssi"] = m + np.log(pdr / (1 - pdr)) / a

    # Latency from the measured offered load through an M/M/1 sojourn time.
    lam = intel["n_reports"].to_numpy(np.float64) / 300.0     # packets/s
    mu = max(np.quantile(lam, 0.995) * 1.35, 1e-3)
    rho = np.clip(lam / mu, 0.0, 0.97)
    cols["latency_ms"] = 1000.0 / (mu * (1.0 - rho))

    # Outdoor temperature from TON_IoT weather telemetry (normal rows only).
    ton, tm = load_ton_iot(devices=("weather",))
    prov["ton_iot"] = tm["source"]
    w = ton["weather"]
    w = w[w["type"] == "normal"]
    ot = w["temperature"].to_numpy(np.float64)
    ot = ot[np.isfinite(ot)]
    ot = _tile_to(ot, T)
    cols["outdoor_temp"] = ot

    # HVAC power and energy from UK-DALE aggregate demand.
    try:
        uk, um = load_ukdale(house=1, channels=(1,))
        prov["ukdale"] = um["source"]
        p = uk.iloc[:, 1].to_numpy(np.float64) / 1000.0       # W -> kW
        p = _tile_to(p, T)
    except DatasetUnavailable as e:               # pragma: no cover
        prov["ukdale"] = f"UNAVAILABLE ({e}); HVAC power synthesised from RC model"
        p = 1.8 + 1.2 * np.sin(np.arange(T) * 2 * np.pi / 288) + rng.normal(0, .25, T)
        p = np.clip(p, 0.05, None)
    cols["hvac_power"] = p
    cols["energy_kwh"] = p * (300.0 / 3600.0)

    X = np.stack([cols[v] for v in VARS], axis=1).astype(np.float32)
    X = _clean(X)
    meta = {"n_steps": int(X.shape[0]), "vars": VARS, "provenance": prov,
            "channel_sources": {
                "indoor_temp": "Intel Lab temperature (measured)",
                "humidity": "Intel Lab humidity (measured)",
                "occupancy": "Intel Lab light -> 0-12 occupant proxy",
                "outdoor_temp": "TON_IoT weather temperature (measured, normal rows)",
                "hvac_power": "UK-DALE house-1 aggregate demand (measured)",
                "energy_kwh": "UK-DALE aggregate demand integrated over 300 s",
                "wifi_rssi": "Intel Lab delivery ratio -> inverse logistic PDR(RSSI)",
                "latency_ms": "Intel Lab report rate -> M/M/1 sojourn time"}}
    return X, meta


def _tile_to(x: np.ndarray, T: int) -> np.ndarray:
    if len(x) >= T:
        return x[:T]
    reps = int(np.ceil(T / max(len(x), 1)))
    return np.tile(x, reps)[:T]


def _clean(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, np.float32)
    for j in range(X.shape[1]):
        c = X[:, j]
        bad = ~np.isfinite(c)
        if bad.any():
            c[bad] = np.interp(np.flatnonzero(bad), np.flatnonzero(~bad), c[~bad])
        lo, hi = np.quantile(c, [0.001, 0.999])
        X[:, j] = np.clip(c, lo, hi)
    return X


# --------------------------------------------------------------- causal net
class MaskedCausalNet(nn.Module):
    """One MLP head per node over a lag window, with a hard input mask.

    input  : (B, tau, N)
    mask   : (N, tau, N) -- mask[j] selects which (lag, parent) entries node j
             is allowed to read. Zeroing an entry removes the edge exactly.
    output : (B, N)
    """

    def __init__(self, n_vars: int, tau: int, hidden: int = 96):
        super().__init__()
        self.n, self.tau = n_vars, tau
        d_in = n_vars * tau
        self.heads = nn.ModuleList([
            nn.Sequential(nn.Linear(d_in, hidden), nn.GELU(),
                          nn.Linear(hidden, hidden), nn.GELU(),
                          nn.Linear(hidden, 1))
            for _ in range(n_vars)])
        self.register_buffer("mask", torch.ones(n_vars, tau, n_vars))

    def set_mask(self, mask: torch.Tensor):
        self.mask.copy_(mask.to(self.mask.device, self.mask.dtype))

    def forward(self, x):                     # x: (B, tau, N)
        outs = []
        for j, head in enumerate(self.heads):
            xm = x * self.mask[j].unsqueeze(0)
            outs.append(head(xm.reshape(x.shape[0], -1)))
        return torch.cat(outs, dim=1)


def _windows(X: np.ndarray, tau: int):
    T = X.shape[0]
    xs = np.stack([X[t - tau:t] for t in range(tau, T)])
    ys = X[tau:]
    return xs.astype(np.float32), ys.astype(np.float32)


def _train_net(net, xs, ys, epochs=140, bs=256, lr=2e-3, device="cuda",
               verbose=False, input_noise: float = 0.0):
    """Fit the per-node heads.

    `input_noise` injects Gaussian perturbations into the lag window during
    training. The network is fitted with teacher forcing but generation runs
    free, so without this the map is only accurate on the data manifold and
    its own small errors compound into a divergent rollout. Denoising the
    inputs makes it contractive in a neighbourhood of the data, which is what
    a stable generator needs.
    """
    net = net.to(device)
    xt = torch.from_numpy(xs).to(device)
    yt = torch.from_numpy(ys).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    n = len(xt)
    for ep in range(epochs):
        perm = torch.randperm(n, device=device)
        tot = 0.0
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            xb = xt[idx]
            if input_noise > 0:
                xb = xb + torch.randn_like(xb) * input_noise
            loss = nn.functional.mse_loss(net(xb), yt[idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            tot += loss.item() * len(idx)
        sched.step()
        if verbose and ep % 40 == 0:
            print(f"    epoch {ep:3d} mse {tot / n:.5f}")
    return net


def _importance(net, xs, device="cuda", n_probe=2048) -> np.ndarray:
    """Mean |d y_j / d x_{i,lag}| over the data -> (N, tau, N) importance."""
    net.eval()
    idx = np.random.default_rng(0).choice(len(xs), min(n_probe, len(xs)), replace=False)
    x = torch.from_numpy(xs[idx]).to(device).requires_grad_(True)
    imp = torch.zeros(net.n, net.tau, net.n, device=device)
    for j in range(net.n):
        if x.grad is not None:
            x.grad = None
        out = net(x)[:, j].sum()
        g, = torch.autograd.grad(out, x, retain_graph=True)
        imp[j] = g.abs().mean(0)
    return imp.detach().cpu().numpy()


# ------------------------------------------------------------ residual flow
class RealNVP(nn.Module):
    """Small RealNVP density model for the residual vector.

    CausalTime uses a normalising flow so that generated residuals keep the
    heavy tails and cross-channel dependence of real sensor noise; a Gaussian
    would wash both out.
    """

    def __init__(self, dim: int, n_blocks: int = 6, hidden: int = 96):
        super().__init__()
        self.dim = dim
        self.n_blocks = n_blocks
        masks = []
        for b in range(n_blocks):
            m = torch.zeros(dim)
            m[b % 2::2] = 1.0
            masks.append(m)
        self.register_buffer("masks", torch.stack(masks))
        self.s = nn.ModuleList([nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, hidden),
            nn.GELU(), nn.Linear(hidden, dim), nn.Tanh()) for _ in range(n_blocks)])
        self.t = nn.ModuleList([nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, hidden),
            nn.GELU(), nn.Linear(hidden, dim)) for _ in range(n_blocks)])

    def forward(self, x):
        """x -> z, log|det J|"""
        logdet = torch.zeros(x.shape[0], device=x.device)
        z = x
        for b in range(self.n_blocks):
            m = self.masks[b]
            zm = z * m
            s = self.s[b](zm) * (1 - m)
            t = self.t[b](zm) * (1 - m)
            z = zm + (1 - m) * (z * torch.exp(s) + t)
            logdet = logdet + s.sum(1)
        return z, logdet

    def inverse(self, z):
        x = z
        for b in reversed(range(self.n_blocks)):
            m = self.masks[b]
            xm = x * m
            s = self.s[b](xm) * (1 - m)
            t = self.t[b](xm) * (1 - m)
            x = xm + (1 - m) * ((x - t) * torch.exp(-s))
        return x

    def log_prob(self, x):
        z, logdet = self.forward(x)
        return (-0.5 * (z ** 2 + np.log(2 * np.pi)).sum(1) + logdet)

    @torch.no_grad()
    def sample(self, n, device="cuda"):
        z = torch.randn(n, self.dim, device=device)
        return self.inverse(z)


def _fit_flow(R: np.ndarray, epochs=400, bs=512, lr=1e-3, device="cuda", seed=0):
    torch.manual_seed(seed)
    mu, sd = R.mean(0), R.std(0) + 1e-6
    Z = (R - mu) / sd
    flow = RealNVP(R.shape[1]).to(device)
    x = torch.from_numpy(Z.astype(np.float32)).to(device)
    opt = torch.optim.AdamW(flow.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    for ep in range(epochs):
        perm = torch.randperm(len(x), device=device)
        for i in range(0, len(x), bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            loss = -flow.log_prob(x[idx]).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(flow.parameters(), 5.0)
            opt.step()
        sched.step()
    return flow, mu.astype(np.float32), sd.astype(np.float32)


# ---------------------------------------------------------------- pipeline
class CausalTimeIoT:
    """Fitted generator + exact ground-truth graph + interventional oracle."""

    def __init__(self, tau: int = 3, hidden: int = 96, device: str = "cuda",
                 seed: int = 0, input_noise: float = 0.20):
        self.tau, self.hidden, self.seed = tau, hidden, seed
        self.input_noise = input_noise
        self.innov_scale = np.ones(NVAR, np.float32)   # set by calibrate()
        self.out_scale = np.ones(NVAR, np.float32)
        self.out_shift = None
        self.device = device if torch.cuda.is_available() else "cpu"
        self.net: Optional[MaskedCausalNet] = None
        self.flow: Optional[RealNVP] = None
        self.G: Optional[np.ndarray] = None          # (N, N) summary graph
        self.G_lag: Optional[np.ndarray] = None      # (N, tau, N) lagged graph
        self.mu = self.sd = None
        self.res_ar: Optional[np.ndarray] = None     # AR(1) coeff per residual

    # -- stage 1-3 -------------------------------------------------------
    def fit(self, X: np.ndarray, density_target: float = 0.30,
            epochs: int = 140, verbose: bool = True):
        set_seed(self.seed)
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-6
        Z = ((X - self.mu) / self.sd).astype(np.float32)
        xs, ys = _windows(Z, self.tau)
        n_tr = int(0.8 * len(xs))

        if verbose:
            print("  [1/4] fitting unmasked autoregressive network ...")
        net = MaskedCausalNet(NVAR, self.tau, self.hidden)
        net = _train_net(net, xs[:n_tr], ys[:n_tr], epochs=epochs,
                         device=self.device, verbose=verbose)

        if verbose:
            print("  [2/4] gradient importance -> hypothesised graph ...")
        imp = _importance(net, xs[:n_tr], device=self.device)   # (N, tau, N)
        # normalise per target so every node keeps its own strongest parents
        flat = imp.reshape(NVAR, -1)
        flat = flat / (flat.max(axis=1, keepdims=True) + 1e-12)
        k = max(1, int(round(density_target * self.tau * NVAR)))
        G_lag = np.zeros_like(flat)
        for j in range(NVAR):
            top = np.argsort(-flat[j])[:k]
            G_lag[j, top] = 1.0
        G_lag = G_lag.reshape(NVAR, self.tau, NVAR)
        # every node keeps its own lag-1 self edge (physical persistence)
        for j in range(NVAR):
            G_lag[j, -1, j] = 1.0
        self.G_lag = G_lag
        G = (G_lag.sum(1) > 0).astype(np.float32)      # G[j, i]: i -> j
        self.G = G.T.copy()                            # store as G[i, j]: i -> j
        np.fill_diagonal(self.G, 0.0)

        if verbose:
            n_e = int(self.G.sum())
            print(f"        graph: {n_e} off-diagonal edges "
                  f"(density {n_e / (NVAR * (NVAR - 1)):.2f})")
            print("  [3/4] re-fitting under the hard mask (graph becomes exact) ...")
        net2 = MaskedCausalNet(NVAR, self.tau, self.hidden)
        net2.set_mask(torch.from_numpy(G_lag).float())
        net2 = _train_net(net2, xs[:n_tr], ys[:n_tr], epochs=epochs,
                          device=self.device, verbose=verbose,
                          input_noise=self.input_noise)
        self.net = net2

        # -- stage 4: residuals + flow ----------------------------------
        with torch.no_grad():
            pred = net2(torch.from_numpy(xs).to(self.device)).cpu().numpy()
        R = ys - pred
        self.res_ar = np.array([_ar1(R[:, j]) for j in range(NVAR)], np.float32)
        innov = R[1:] - self.res_ar[None, :] * R[:-1]
        if verbose:
            print("  [4/4] fitting RealNVP on residual innovations ...")
        self.flow, self.rmu, self.rsd = _fit_flow(innov, device=self.device,
                                                  seed=self.seed)
        self.fit_stats = {
            "train_mse": float(((pred[:n_tr] - ys[:n_tr]) ** 2).mean()),
            "test_mse": float(((pred[n_tr:] - ys[n_tr:]) ** 2).mean()),
            "residual_ar1": self.res_ar.tolist(),
            "n_edges": int(self.G.sum()),
        }
        return self

    # -- generation ------------------------------------------------------
    # Free-running an autoregressive model that was trained with teacher
    # forcing is only conditionally stable: small errors feed back through the
    # residual AR term and compound. We clamp both the residual and the state
    # to a physically wide but finite band in standardised units, which is the
    # usual remedy and costs nothing when the rollout is already stable.
    CLAMP_X = 5.0
    CLAMP_R = 6.0
    CLAMP_INNOV = 5.0

    @torch.no_grad()
    def generate(self, n_samples: int = 480, T: int = 40, seed: int = 0,
                 warmup: int = 64, seed_X: Optional[np.ndarray] = None
                 ) -> Tuple[np.ndarray, np.ndarray]:
        """Unroll the split model. Returns (n_samples, T, 2N) like CausalTime."""
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        dev = self.device
        N, tau = NVAR, self.tau
        total = warmup + T
        out = np.zeros((n_samples, total, N), np.float32)
        res_out = np.zeros((n_samples, total, N), np.float32)

        if seed_X is not None:
            Zseed = ((seed_X - self.mu) / self.sd).astype(np.float32)
            starts = rng.integers(0, max(len(Zseed) - tau - 1, 1), n_samples)
            hist = np.stack([Zseed[s:s + tau] for s in starts])
        else:
            hist = rng.normal(0, 1, (n_samples, tau, N)).astype(np.float32)

        h = torch.from_numpy(hist).to(dev)
        r = torch.zeros(n_samples, N, device=dev)
        ar = torch.from_numpy(self.res_ar).to(dev)
        rmu = torch.from_numpy(self.rmu).to(dev)
        # Per-channel innovation scale, set by `calibrate()`. Training the
        # causal net with input noise makes it contractive, so a free-running
        # rollout is smoother and narrower than the data it was fitted to.
        # Scaling the innovations restores the one-step variability; it is a
        # per-variable change to the noise term, so the graph is untouched.
        rsd = torch.from_numpy(self.rsd * self.innov_scale).to(dev)
        for t in range(total):
            causal = self.net(h)
            z = self.flow.sample(n_samples, device=dev)
            z = torch.clamp(z, -self.CLAMP_INNOV, self.CLAMP_INNOV)
            innov = z * rsd + rmu
            r = torch.clamp(ar.unsqueeze(0) * r + innov,
                            -self.CLAMP_R, self.CLAMP_R)
            x = torch.clamp(causal + r, -self.CLAMP_X, self.CLAMP_X)
            out[:, t] = x.cpu().numpy()
            res_out[:, t] = r.cpu().numpy()
            h = torch.cat([h[:, 1:], x.unsqueeze(1)], dim=1)

        out = out[:, warmup:]
        res_out = res_out[:, warmup:]
        # Affine calibration to the seed corpus's standardised moments. The
        # rollout drifts, so its per-channel mean and spread need not match the
        # data even when its dynamics do. The correction is one scale and one
        # offset per variable, which is invertible and applied to each variable
        # separately, so every edge of the ground-truth graph is preserved.
        if self.out_shift is not None:
            out = out * self.out_scale[None, None, :] + self.out_shift[None, None, :]
        data_std = np.concatenate([out, res_out], axis=2)          # (S, T, 2N)
        data = data_std.copy()
        data[:, :, :N] = data[:, :, :N] * self.sd + self.mu        # de-standardise
        return data.astype(np.float32), self.G.copy()

    # -- interventional oracle ------------------------------------------
    @torch.no_grad()
    def true_ate(self, var: str, v_do: float, v_ref: float,
                 targets: List[str], horizon: int = 12, n_mc: int = 512,
                 seed: int = 0, seed_X: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Ground-truth ATE of do(var = v_do) vs do(var = v_ref).

        Both arms are rolled out with *common random numbers*: the residual
        innovations are drawn once and reused, so the difference isolates the
        causal effect rather than Monte-Carlo noise.
        """
        dev, N, tau = self.device, NVAR, self.tau
        j_do = VARS.index(var)
        rng = np.random.default_rng(seed)
        torch.manual_seed(seed)

        if seed_X is not None:
            Zseed = ((seed_X - self.mu) / self.sd).astype(np.float32)
            starts = rng.integers(0, max(len(Zseed) - tau - 1, 1), n_mc)
            hist0 = torch.from_numpy(np.stack([Zseed[s:s + tau] for s in starts])).to(dev)
        else:
            hist0 = torch.randn(n_mc, tau, N, device=dev)

        innovs = [torch.clamp(self.flow.sample(n_mc, device=dev),
                              -self.CLAMP_INNOV, self.CLAMP_INNOV)
                  * torch.from_numpy(self.rsd).to(dev)
                  + torch.from_numpy(self.rmu).to(dev) for _ in range(horizon)]
        ar = torch.from_numpy(self.res_ar).to(dev)

        def roll(val_raw):
            val = (val_raw - self.mu[j_do]) / self.sd[j_do]
            h = hist0.clone()
            h[:, :, j_do] = float(val)
            r = torch.zeros(n_mc, N, device=dev)
            acc = torch.zeros(n_mc, N, device=dev)
            for t in range(horizon):
                r = torch.clamp(ar.unsqueeze(0) * r + innovs[t],
                                -self.CLAMP_R, self.CLAMP_R)
                x = torch.clamp(self.net(h) + r, -self.CLAMP_X, self.CLAMP_X)
                x[:, j_do] = float(val)          # hard intervention holds
                acc += x
                h = torch.cat([h[:, 1:], x.unsqueeze(1)], dim=1)
            return acc / horizon

        a = roll(v_do).cpu().numpy() * self.sd + self.mu
        b = roll(v_ref).cpu().numpy() * self.sd + self.mu
        d = (a - b).mean(0)
        return {t: float(d[VARS.index(t)]) for t in targets}

    # -- persistence -----------------------------------------------------
    def calibrate(self, seed_X: np.ndarray, n_samples: int = 600, T: int = 40,
                  rounds: int = 6, seed: int = 0, verbose: bool = False):
        """Match the rollout's one-step variability and marginals to the data.

        Two things go wrong when a teacher-forced autoregressive model is run
        free. It is contractive, because it was trained on noised inputs, so
        its trajectories are smoother than the data; and it drifts, so its
        per-channel mean and spread need not match. Both are per-variable
        properties of the noise and scale, not of the structure, so correcting
        them leaves the ground-truth graph exact.

        We fit one innovation multiplier per channel by matching the lag-1
        autocorrelation of the generated series to the real one -- a smoother
        series has higher lag-1 autocorrelation, so the multiplier moves
        monotonically against the error -- and then one affine correction per
        channel so the marginals line up.
        """
        N = NVAR
        Zs = ((seed_X - self.mu) / self.sd).astype(np.float32)
        # The target statistics must be estimated exactly as they will be on
        # the generated corpus: per window, over windows of the same length.
        # Lag-1 autocorrelation on a long series and on a 40-step window are
        # different quantities, and matching one against the other drives the
        # calibration the wrong way.
        rng_c = np.random.default_rng(seed)
        starts = rng_c.integers(0, max(len(Zs) - T - 1, 1), n_samples)
        rw = np.stack([Zs[st:st + T] for st in starts])
        real_acf = np.array([np.nanmean([
            np.corrcoef(w[:-1, j], w[1:, j])[0, 1]
            if w[:, j].std() > 1e-8 else np.nan for w in rw])
            for j in range(N)], np.float32)
        real_acf = np.nan_to_num(real_acf)
        real_sd = rw.reshape(-1, N).std(0)

        def gen_stats(scale):
            self.innov_scale = scale.astype(np.float32)
            self.out_shift = None                       # raw, uncalibrated
            self.out_scale = np.ones(N, np.float32)
            d, _ = self.generate(n_samples=n_samples, T=T, seed=seed,
                                 seed_X=seed_X)
            X = (d[:, :, :N] - self.mu) / self.sd       # back to standardised
            acf = np.array([np.nanmean([
                np.corrcoef(w[:-1, j], w[1:, j])[0, 1]
                if w[:, j].std() > 1e-8 else np.nan for w in X])
                for j in range(N)], np.float32)
            return np.nan_to_num(acf), X

        scale = np.ones(N, np.float32)
        for it in range(rounds):
            acf, X = gen_stats(scale)
            err = acf - real_acf                 # >0 means too smooth
            if np.all(np.abs(err) < 0.02):
                break
            # too smooth -> more innovation noise; damped multiplicative step
            scale = np.clip(scale * np.exp(np.clip(err, -0.5, 0.5) * 1.5),
                            0.2, 12.0).astype(np.float32)
            if verbose:
                print(f"    calib {it}: max|acf err| {np.abs(err).max():.3f}")
        self.innov_scale = scale

        acf, X = gen_stats(scale)
        g_mu = X.reshape(-1, N).mean(0)
        g_sd = X.reshape(-1, N).std(0) + 1e-8
        self.out_scale = (1.0 / g_sd).astype(np.float32)
        self.out_shift = (-g_mu / g_sd).astype(np.float32)
        self.calib_stats = {
            "real_acf1": real_acf.tolist(), "gen_acf1": acf.tolist(),
            "real_sd_windowed": real_sd.tolist(),
            "innov_scale": scale.tolist(),
            "out_scale": self.out_scale.tolist(),
            "out_shift": self.out_shift.tolist()}
        if verbose:
            print(f"    innovation scale: "
                  f"{np.round(scale, 2).tolist()}")
        return self.calib_stats

    def save(self, path: Path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save({"net": self.net.state_dict(), "flow": self.flow.state_dict(),
                    "mask": self.net.mask.cpu(), "tau": self.tau,
                    "hidden": self.hidden, "mu": self.mu, "sd": self.sd,
                    "rmu": self.rmu, "rsd": self.rsd, "res_ar": self.res_ar,
                    "G": self.G, "G_lag": self.G_lag,
                    "innov_scale": self.innov_scale,
                    "out_scale": self.out_scale, "out_shift": self.out_shift},
                   path / "causaltime_iot_generator.pt")
        save_json(self.fit_stats, path / "fit_stats.json")

    @classmethod
    def load(cls, path: Path, device: str = "cuda"):
        d = torch.load(Path(path) / "causaltime_iot_generator.pt",
                       map_location="cpu", weights_only=False)
        o = cls(tau=d["tau"], hidden=d["hidden"], device=device)
        o.net = MaskedCausalNet(NVAR, d["tau"], d["hidden"])
        o.net.load_state_dict(d["net"])
        o.net.set_mask(d["mask"])
        o.net = o.net.to(o.device).eval()
        o.flow = RealNVP(NVAR).to(o.device)
        o.flow.load_state_dict(d["flow"])
        o.flow.eval()
        o.mu, o.sd, o.rmu, o.rsd = d["mu"], d["sd"], d["rmu"], d["rsd"]
        o.res_ar, o.G, o.G_lag = d["res_ar"], d["G"], d["G_lag"]
        o.innov_scale = d.get("innov_scale", np.ones(NVAR, np.float32))
        o.out_scale = d.get("out_scale", np.ones(NVAR, np.float32))
        o.out_shift = d.get("out_shift")
        o.fit_stats = {}
        return o


def _ar1(x: np.ndarray) -> float:
    x = x - x.mean()
    d = (x[:-1] ** 2).sum()
    return float(np.clip((x[1:] * x[:-1]).sum() / (d + 1e-12), -0.95, 0.95))


# ------------------------------------------------------------------- API
def build_causaltime_iot(n_steps: int = 8000, n_samples: int = 480, T: int = 40,
                         tau: int = 3, seed: int = 0, out_dir: Path = CT_IOT_DIR,
                         verbose: bool = True):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    X, meta = build_seed_frame(n_steps=n_steps, seed=seed)
    if verbose:
        print(f"  seed frame {X.shape} from real traces")
    gen = CausalTimeIoT(tau=tau, seed=seed).fit(X, verbose=verbose)
    # Calibrate before generating. A teacher-forced model run free is
    # contractive and drifts, so without this the corpus is separable from the
    # real traces on its marginals alone -- which is what our first release
    # was. The correction is per-variable and invertible, so the ground-truth
    # graph is untouched.
    if verbose:
        print("  [5/5] calibrating the rollout to the seed corpus ...")
    calib = gen.calibrate(X, n_samples=min(n_samples, 600), T=T, seed=seed,
                          verbose=verbose)
    data, G = gen.generate(n_samples=n_samples, T=T, seed=seed, seed_X=X)
    np.save(out_dir / "gen_data.npy", data)
    np.save(out_dir / "graph.npy", G)
    np.save(out_dir / "seed_real.npy", X)
    gen.save(out_dir)
    meta.update(gen.fit_stats)
    meta["calibration"] = calib
    meta["gen_data_shape"] = list(data.shape)
    meta["graph_shape"] = list(G.shape)
    save_json(meta, out_dir / "meta.json")
    return data, G, meta, gen


def load_causaltime_iot(out_dir: Path = CT_IOT_DIR):
    out_dir = Path(out_dir)
    if not (out_dir / "gen_data.npy").exists():
        raise DatasetUnavailable("CausalTime-IoT not built; run "
                                 "experiments/build_datasets.py")
    data = np.load(out_dir / "gen_data.npy")
    G = np.load(out_dir / "graph.npy")
    meta = json.loads((out_dir / "meta.json").read_text())
    return data, G, meta

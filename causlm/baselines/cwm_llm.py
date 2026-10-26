"""CWM-LLM: a causal world model the language model can query.

Re-implementation of the design in "Language Agents Meet Causality -- Bridging
LLMs and Causal World Models" (Gkountouras, Lindemann, Lippe, Gavves & Titov,
ICLR 2025, arXiv:2410.19923), adapted to this environment. Their framework
learns a causal world model whose variables are tied to natural-language names
and lets the LLM query it as a simulator; planning quality comes from the world
model, and the LLM does the reasoning on top of it.

Differences from CauSLM-Edge, which are the point of the comparison:
  * the world model is learned from the observation window rather than read out
    of a statistical skeleton, and no spurious-edge repair step exists;
  * the language model is invoked on every decision -- there is no cost model
    for the call itself.
"""
from __future__ import annotations

from collections import deque
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from ..config import VARS
from ..env import ACTION_LEVELS
from ..rl.bandit import KNOB_ARMS
from ..slm.schema import CausalAdvice
from ..slm.teacher import CANDIDATES

ACTION_KEYS = list(ACTION_LEVELS)


class _WorldModel(torch.nn.Module):
    """Small learned dynamics model over the observed variables."""

    def __init__(self, n: int, tau: int = 3, hidden: int = 128):
        super().__init__()
        self.n, self.tau = n, tau
        self.net = torch.nn.Sequential(
            torch.nn.Linear(n * tau, hidden), torch.nn.GELU(),
            torch.nn.Linear(hidden, hidden), torch.nn.GELU(),
            torch.nn.Linear(hidden, n))

    def forward(self, x):
        return self.net(x.reshape(x.shape[0], -1))


class CWMLLMAgent:
    """Causal world model + LLM planner, queried every step."""

    def __init__(self, policy, var_names: Sequence[str] = None,
                 window: int = 288, refit_period: int = 48, tau: int = 3,
                 device: str = "cuda", seed: int = 0, n_candidates: int = 8,
                 query_every: int = 1):
        self.policy = policy
        self.var_names = list(var_names or VARS)
        self.window, self.refit_period, self.tau = window, refit_period, tau
        self.device = device if torch.cuda.is_available() else "cpu"
        self.rng = np.random.default_rng(seed)
        self.n_candidates = n_candidates
        self.query_every = query_every
        self.reset()

    def reset(self, full: bool = False):
        self.win = deque(maxlen=self.window)
        self.wm = None
        self.mu = self.sd = None
        self.t = 0
        self.n_queries = 0
        self.slm_energy_j = 0.0
        self.slm_latency_ms = 0.0
        self.measured_slm_energy_j = 0.0
        self.measured_slm_latency_ms = 0.0
        self.unit_energy_j = None
        self.unit_latency_ms = None
        self.invalid_json = 0
        self.knobs = {"hvac_setpoint": 22.0, "sample_rate_hz": 0.25,
                      "duty_cycle": 0.5, "offload_frac": 0.0}
        self._queried = False
        self._sims = {}
        self._env = None
        self._kpis = {}

    # ------------------------------------------------------------ world model
    def _fit_wm(self):
        if len(self.win) < 96:
            return
        X = np.stack(self.win).astype(np.float32)
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-6
        Z = (X - self.mu) / self.sd
        xs = np.stack([Z[t - self.tau:t] for t in range(self.tau, len(Z))])
        ys = Z[self.tau:]
        wm = _WorldModel(Z.shape[1], self.tau).to(self.device)
        xt = torch.from_numpy(xs).to(self.device)
        yt = torch.from_numpy(ys).to(self.device)
        opt = torch.optim.AdamW(wm.parameters(), lr=3e-3, weight_decay=1e-4)
        for _ in range(180):
            opt.zero_grad()
            torch.nn.functional.mse_loss(wm(xt), yt).backward()
            opt.step()
        self.wm = wm.eval()

    @torch.no_grad()
    def _simulate(self, cand: Dict[str, float], horizon: int = 8) -> Dict[str, float]:
        """Roll the world model forward with the candidate value held fixed."""
        if self.wm is None or len(self.win) < self.tau + 1:
            return {}
        Z = (np.stack(self.win)[-self.tau:] - self.mu) / self.sd
        h = torch.from_numpy(Z.astype(np.float32)).to(self.device)[None]
        idx = {v: i for i, v in enumerate(self.var_names)}
        acc_do = np.zeros(len(self.var_names))
        acc_ref = np.zeros(len(self.var_names))
        for arm, target in (("do", cand), ("ref", {})):
            hh = h.clone()
            acc = np.zeros(len(self.var_names))
            for _ in range(horizon):
                x = self.wm(hh)
                for k, v in target.items():
                    if k in idx:
                        x[0, idx[k]] = (v - self.mu[idx[k]]) / self.sd[idx[k]]
                x = torch.clamp(x, -5, 5)
                acc += x[0].cpu().numpy()
                hh = torch.cat([hh[:, 1:], x[None]], dim=1)
            acc = acc / horizon * self.sd + self.mu
            if arm == "do":
                acc_do = acc
            else:
                acc_ref = acc
        d = acc_do - acc_ref
        return {v: float(d[i]) for i, v in enumerate(self.var_names)}

    # ------------------------------------------------------------------ acting
    def act(self, env, kpis):
        ctx = self.act_prepare(env, kpis)
        res = self.policy.advise(ctx) if ctx is not None else None
        return self.act_finish(res)

    def act_prepare(self, env, kpis):
        if kpis.get("_row") is not None:
            self.win.append(kpis["_row"])
        if self.t % self.refit_period == 0:
            self._fit_wm()

        queried = (self.wm is not None and self.t % self.query_every == 0
                   and len(self.win) >= 32)
        self._queried = queried
        self._sims = {}
        self._env = env
        self._kpis = kpis
        if not queried:
            return None
        ci = self.rng.choice(len(CANDIDATES),
                             min(self.n_candidates, len(CANDIDATES)),
                             replace=False)
        cands = [CANDIDATES[int(i)] for i in ci]
        for c in cands:
            d = self._simulate(c)
            if d:
                self._sims[tuple(sorted(c.items()))] = d
        # the world model's own rollouts stand in for a statistical skeleton
        n = len(self.var_names)
        return {"A": np.zeros((n, n)), "X": np.stack(self.win),
                "var_names": self.var_names, "candidates": cands,
                "device_meta": {"n_devices": env.nc.n_devices,
                                "broker": "mqtt", "wm": "learned-CRL"},
                "regime": {"world_model": "fitted", "sim_horizon": "8"},
                "kpis": kpis}

    def act_finish(self, result):
        env, kpis, queried = self._env, self._kpis, self._queried
        if queried:
            adv, cost = result if result is not None else (None, None)
            self.n_queries += 1
            if cost is not None:
                self.measured_slm_energy_j += cost.energy_j
            if self.unit_energy_j is None:
                sc = getattr(env, "sc", None)
                self.unit_energy_j = float(getattr(sc, "energy_j", 0.0))
                self.unit_latency_ms = float(getattr(sc, "latency_ms", 0.0))
            self.slm_energy_j += self.unit_energy_j
            self.slm_latency_ms += self.unit_latency_ms
            if adv is None or not adv.valid:
                self.invalid_json += 1
            if adv is not None and adv.valid and adv.do:
                for k, v in adv.do.items():
                    if k in self.knobs:
                        self.knobs[k] = float(v)
            else:
                # no usable recommendation: fall back on the world model's own
                # best-scoring candidate rather than freezing the knobs
                if self._sims:
                    best = max(self._sims, key=lambda k: _util(self._sims[k]))
                    for kk, vv in best:
                        if kk in self.knobs:
                            self.knobs[kk] = float(vv)

        a = []
        for k in ACTION_KEYS:
            if k in self.knobs:
                lv = np.asarray(ACTION_LEVELS[k])
                a.append(int(np.argmin(np.abs(lv - self.knobs[k]))))
            elif k == "isolate":
                a.append(1 if kpis.get("suspicion", 0.0) > 0.75 else 0)
            elif k == "query_slm":
                a.append(int(queried))
            else:
                a.append(0)
        self.t += 1
        return a, {"queried": queried}

    def observe(self, *a, **kw):
        pass

    def summary(self):
        return {"n_queries": self.n_queries, "slm_energy_j": self.slm_energy_j,
                "slm_j_per_query": self.slm_energy_j / max(self.n_queries, 1),
                "slm_ms_per_query": self.slm_latency_ms / max(self.n_queries, 1),
                "invalid_json": self.invalid_json,
                "invalid_json_rate": self.invalid_json / max(self.n_queries, 1),
                "query_rate": self.n_queries / max(self.t, 1), "gate": "always"}


def _util(d: Dict[str, float]) -> float:
    return -(1.0 * d.get("energy_kwh", 0.0)
             + 0.0012 * max(d.get("latency_ms", 0.0), 0.0)
             + 0.004 * max(d.get("it_energy_j", 0.0), 0.0)
             + 0.15 * abs(d.get("indoor_temp", 0.0)))

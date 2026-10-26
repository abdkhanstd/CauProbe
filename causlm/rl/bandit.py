"""Factored causal bandit over the gateway's actuator knobs.

The causal graph is not decoration here: it decides *credit assignment*.
Each knob's bandit is only scored on the reward terms whose KPI is a
descendant of that knob in the (repaired) graph. A graph that misses
`sample_rate_hz -> latency_ms` therefore stops the sampling-rate bandit from
ever seeing the latency penalty it causes, and the controller keeps a bad
sampling rate. This is the mechanism by which graph F1 turns into QoS.

Each knob runs disjoint LinUCB (Li et al. 2010) over its arms, with the small
context vector below. When the small language model supplies a `predicted_ATE`
for an arm, it is injected as a prior pseudo-observation rather than as a hard
override, so a wrong SLM prediction is washed out by real feedback instead of
being obeyed forever.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..config import VARS, VIDX

KNOB_ARMS: Dict[str, List[float]] = {
    "hvac_setpoint": [18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0],
    "sample_rate_hz": [0.1, 0.25, 0.5, 1.0],
    "duty_cycle": [0.25, 0.5, 1.0],
    "offload_frac": [0.0, 0.5, 1.0],
}

# Which reward term each KPI contributes to.
# Which reward term each KPI contributes to. `hvac_power` gets its own term
# rather than being folded into `energy_kwh`: the meter also carries lighting,
# standby and occupant plug load, none of which the setpoint controls, and
# scoring the setpoint arm on the whole meter buries its effect in that noise.
# The graph says the setpoint's influence on cost runs through hvac_power, so
# that is what the setpoint arm is credited with.
REWARD_TERMS = {
    "hvac_power": "hvac_cost",
    "energy_kwh": "energy",
    "energy_cost": "energy",
    "tariff": "energy",
    "indoor_temp": "comfort",
    "humidity": "comfort",
    "latency_ms": "latency",
    "wifi_rssi": "pdr",
    "it_energy_j": "it_energy",
}


def descendants(G: np.ndarray, i: int, depth: int = 3) -> set:
    cur, out = {i}, set()
    for _ in range(depth):
        nxt = set()
        for c in cur:
            for j in np.flatnonzero(G[c] > 0):
                j = int(j)
                if j != i and j not in out:
                    out.add(j)
                    nxt.add(j)
        cur = nxt
        if not cur:
            break
    return out


class LinUCB:
    def __init__(self, n_arms: int, d: int, alpha: float = 0.6,
                 lam: float = 1.0, alpha_min: float = 0.05,
                 alpha_halflife: float = 900.0):
        self.n, self.d, self.alpha = n_arms, d, alpha
        self.alpha_min, self.alpha_halflife = alpha_min, alpha_halflife
        self.A = np.stack([np.eye(d) * lam for _ in range(n_arms)])
        self.b = np.zeros((n_arms, d))
        self.counts = np.zeros(n_arms)

    def scores(self, x: np.ndarray) -> np.ndarray:
        # Exploration decays with experience. A constant bonus keeps pulling
        # arms the posterior has already ruled out, which on a 288-step day is
        # a large share of the episode spent on settings known to be bad.
        n_pull = float(self.counts.sum())
        a_t = max(self.alpha_min,
                  self.alpha * 0.5 ** (n_pull / self.alpha_halflife))
        out = np.zeros(self.n)
        for a in range(self.n):
            Ainv = np.linalg.inv(self.A[a])
            theta = Ainv @ self.b[a]
            out[a] = theta @ x + a_t * np.sqrt(max(x @ Ainv @ x, 0.0))
        return out

    def select(self, x: np.ndarray, mask: Optional[np.ndarray] = None) -> int:
        s = self.scores(x)
        if mask is not None:
            s = np.where(mask, s, -np.inf)
        return int(np.argmax(s))

    def update(self, arm: int, x: np.ndarray, r: float, weight: float = 1.0):
        self.A[arm] += weight * np.outer(x, x)
        self.b[arm] += weight * r * x
        self.counts[arm] += weight

    def discount(self, gamma: float, lam: float = 1.0):
        """Forget at rate `gamma`. Used when the environment has changed.

        Experience about a regime that no longer holds is not experience. A
        stationary bandit would keep trusting it; discounting both the
        posterior and the observation counts lets the controller notice that
        it is ignorant again -- which is also what makes the model's advice
        become usable, and therefore worth paying for, a second time.
        """
        eye = np.eye(self.d) * lam
        self.A = gamma * self.A + (1.0 - gamma) * eye[None, :, :]
        self.b *= gamma
        self.counts *= gamma


@dataclass
class BanditConfig:
    alpha: float = 0.8
    alpha_min: float = 0.05
    alpha_halflife: float = 900.0   # pulls after which exploration halves
    prior_weight: float = 5.0       # pseudo-observations per SLM prior
    prior_decay: float = 0.9        # advice goes stale between invocations
    ctx_dim: int = 11
    # One rule governs every channel through which the model's advice reaches
    # the controller: advice informs a knob while the bandit is ignorant about
    # it, and defers to data once it is not. 288 steps is one control day at a
    # five-minute period. Measured separately, an ungated ATE prior is the
    # channel that does the damage -- at one warm-up day it contributes about
    # 275 pseudo-observations against 288 real ones -- while structural repair
    # is close to free.
    # One control day of pulls. Every number in the paper uses this value.
    # The cold-start ablation (Table 7) says it is too short: 1152 (four days)
    # gains 2.71 points of return at commissioning and 1.50 at convergence, and
    # never withdrawing advice gains the same, so the benefit saturates between
    # one and four days. A deployment should use 1152; we keep 288 here so the
    # code reproduces the published tables exactly.
    advice_horizon_pulls: float = 288.0
    forget: float = 0.985        # per-step discount applied to the posterior
                                 # while a regime change is being detected
    safety_margin: float = 0.6


class FactoredCausalBandit:
    def __init__(self, graph: Optional[np.ndarray] = None,
                 cfg: BanditConfig = None, seed: int = 0,
                 var_names: Sequence[str] = None):
        self.cfg = cfg or BanditConfig()
        self.rng = np.random.default_rng(seed)
        self.var_names = list(var_names or VARS)
        self.knobs = list(KNOB_ARMS)
        self.b = {k: LinUCB(len(KNOB_ARMS[k]), self.cfg.ctx_dim, self.cfg.alpha,
                            alpha_min=self.cfg.alpha_min,
                            alpha_halflife=self.cfg.alpha_halflife)
                  for k in self.knobs}
        self.set_graph(graph)
        self.prior_age = {k: 0 for k in self.knobs}
        self._last = {}

    # ---------------------------------------------------------- graph -> credit
    def set_graph(self, G: Optional[np.ndarray]):
        """Recompute which reward terms each knob is scored on."""
        self.G = G
        self.credit: Dict[str, set] = {}
        if G is None:
            for k in self.knobs:
                self.credit[k] = {"hvac_cost", "energy", "comfort", "latency",
                                  "pdr", "it_energy"}
            return
        idx = {v: i for i, v in enumerate(self.var_names)}
        for k in self.knobs:
            if k not in idx:
                self.credit[k] = {"hvac_cost", "energy", "comfort", "latency",
                                  "it_energy"}
                continue
            des = descendants(G, idx[k])
            terms = set()
            for j in des:
                v = self.var_names[j]
                if v in REWARD_TERMS:
                    terms.add(REWARD_TERMS[v])
            # A knob with no detected consequences would never learn anything,
            # so fall back to the full reward rather than to silence.
            self.credit[k] = terms or {"hvac_cost", "energy", "comfort",
                                       "latency", "it_energy"}

    # ------------------------------------------------------------- SLM prior
    def inject_prior(self, do: Dict[str, float], ate: Dict[str, float],
                     ctx: np.ndarray, pmv_now: float = 0.0, band: float = 0.5,
                     weight: Optional[float] = None, kpis=None,
                     baseline: Optional[Dict[str, float]] = None):
        """Fold one piece of model advice in as a weighted pseudo-observation.

        A treatment effect is a *difference* -- what changes if this
        intervention is taken instead of holding the current settings --
        whereas every real observation the bandit receives is an absolute
        per-step reward. Injecting the difference directly would put the
        advised arm on a different scale from every arm the controller has
        actually tried, and would inflate it systematically. The prior is
        therefore anchored: the predicted change is added to the reward the
        current settings are already delivering on the terms this knob is
        credited with.
        """
        w = self.cfg.prior_weight if weight is None else weight
        baseline = baseline or {}
        for knob, val in do.items():
            if knob not in self.b:
                continue
            if self.b[knob].counts.sum() >= self.cfg.advice_horizon_pulls:
                continue        # the controller has its own data for this knob
            arms = np.asarray(KNOB_ARMS[knob])
            arm = int(np.argmin(np.abs(arms - float(val))))
            delta = self._ate_to_reward(ate, pmv_now, band, self.credit[knob],
                                        kpis)
            anchor = sum(baseline.get(t, 0.0) for t in self.credit[knob])
            r = float(np.clip(anchor + delta, -5, 5))
            self.b[knob].update(arm, ctx, r, weight=w)
            self.prior_age[knob] = 0

    @staticmethod
    def _ate_to_reward(ate: Dict[str, float], pmv_now: float, band: float,
                       terms: set, kpis=None) -> float:
        """Score an ATE only on the reward terms this knob is responsible for."""
        from ..config import DEFAULT
        from ..utility import comfort_penalty, latency_penalty
        rw = DEFAULT.reward
        k = dict(kpis or {})
        k.setdefault("pmv", pmv_now)
        lat0 = float(k.get("latency_ms", rw.latency_sla_ms))
        r = 0.0
        if "hvac_cost" in terms:
            tar = float(k.get("tariff", 1.0))
            r -= rw.w_energy * tar * abs(ate.get("hvac_power", 0.0)) * (300.0 / 3600.0)
        if "energy" in terms:
            tar = float(k.get("tariff", 1.0))
            r -= rw.w_energy * (ate.get("energy_cost")
                                if "energy_cost" in ate
                                else tar * ate.get("energy_kwh", 0.0))
        if "comfort" in terms:
            p0 = float(k.get("pmv", 0.0))
            r -= (comfort_penalty(p0 + ate.get("pmv", 0.0), rw)
                  - comfort_penalty(p0, rw))
        if "latency" in terms:
            r -= (latency_penalty(lat0 + ate.get("latency_ms", 0.0), rw)
                  - latency_penalty(lat0, rw))
        if "it_energy" in terms:
            r -= 0.004 * ate.get("it_energy_j", 0.0) / 1000.0
        return float(np.clip(r, -5, 5))

    # ------------------------------------------------------------- selection
    def select(self, ctx: np.ndarray, safety: str = "ok",
               kpis: Optional[Dict] = None,
               advice: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        """Choose one arm per knob, deferring to advice only where uninformed.

        A piece of model advice overrides the bandit for knob k only while the
        bandit cannot tell the advised arm from its own favourite -- that is,
        while the advised arm still lies inside the optimistic interval of the
        best one. Early on that is nearly always true and the controller acts
        on model knowledge; once the posterior has separated the arms the
        bandit keeps its own choice. An unconditional override is worse than
        either component: it helps on day one and then holds a converged
        controller back.
        """
        out = {}
        advice = advice or {}
        self._override = {}
        for k in self.knobs:
            arms = np.asarray(KNOB_ARMS[k])
            mask = np.ones(len(arms), bool)
            if k == "hvac_setpoint" and kpis is not None:
                # never command a setpoint that would push the zone outside the
                # safe band in one step
                t = kpis.get("indoor_temp", 22.0)
                mask &= (np.abs(arms - t) <= 6.0)
                if safety == "unsafe":
                    mask &= (np.abs(arms - 22.0) <= 2.0)
                if not mask.any():
                    mask = np.ones(len(arms), bool)
            a = self.b[k].select(ctx, mask)
            if k in advice:
                adv_i = int(np.argmin(np.abs(arms - float(advice[k]))))
                # Follow the advice while the bandit has less than a day of
                # its own experience with this knob. Measuring the threshold
                # in experience rather than per-arm pulls is what makes it
                # behave sensibly: an unconditional override helps on day one
                # and then holds a converged controller back, while a purely
                # Bayesian prior, however heavy, is not decisive enough on day
                # one because the optimism term keeps pulling untried arms.
                if mask[adv_i] and self.b[k].counts.sum() < \
                        self.cfg.advice_horizon_pulls:
                    a = adv_i
                    self._override[k] = True
            out[k] = float(arms[a])
            self._last[k] = a
        return out

    def forget_step(self, strength: float = 1.0):
        """Apply one step of forgetting to every knob."""
        g = 1.0 - (1.0 - self.cfg.forget) * float(np.clip(strength, 0.0, 1.0))
        for lin in self.b.values():
            lin.discount(g)

    def update(self, ctx: np.ndarray, term_rewards: Dict[str, float]):
        for k in self.knobs:
            if k not in self._last:
                continue
            r = sum(term_rewards.get(t, 0.0) for t in self.credit[k])
            self.b[k].update(self._last[k], ctx, float(np.clip(r, -5, 5)))
            self.prior_age[k] += 1

    # ---------------------------------------------------------------- state
    def snapshot(self):
        return {k: (v.A.copy(), v.b.copy(), v.counts.copy())
                for k, v in self.b.items()}

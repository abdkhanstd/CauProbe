"""When is it worth paying for an SLM invocation?

Invoking the on-gateway model is itself an intervention, `do(QuerySLM)`, with a
measured Joule and millisecond price. The meta-controller estimates the
expected value of the information the call would return and fires only when
that exceeds `lambda * cost`.

The value model is a small ridge regressor trained online on a directly
observable target: for every actual invocation, the realised advantage of the
next H steps against the counterfactual of having reused the cached plan. That
target is available for free at run time, because the agent already keeps the
cached plan and can score it in hindsight.

Gate baselines implemented for comparison
-----------------------------------------
always / never / periodic / random    -- the obvious controls.
EntropyGate  -- fire when the small model's own predictive uncertainty is high,
                calibrated with isotonic regression, in the style of
                UCCI (Kotte 2026, arXiv:2605.18796).
MarginGate   -- lightweight on-device margin gate with a risk-controlled
                acceptance threshold, in the style of CR^2
                (Xue et al. 2026, arXiv:2605.12001).
EVOIGate     -- ours.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np


# --------------------------------------------------------------- features
FEATURE_NAMES = [
    "staleness",          # steps since the last invocation, capped and scaled
    "drift_z",            # standardised mean shift between window halves
    "graph_churn",        # fraction of edges that changed at the last refit
    "suspicion",          # max device suspicion score (attack indicator)
    "comfort_slack",      # |PMV| excess over the comfort band
    "latency_slack",      # latency over SLA, log-scaled
    "pdr_slack",          # 1 - packet delivery ratio
    "plan_age_reward",    # reward trend since the cached plan was installed
    "occupancy_var",      # short-window variance of the occupancy proxy
    "hour_sin", "hour_cos",
    "bias",
]
NFEAT = len(FEATURE_NAMES)


def build_features(state: Dict) -> np.ndarray:
    f = np.zeros(NFEAT, np.float64)
    f[0] = min(state.get("staleness", 0), 60) / 60.0
    f[1] = np.tanh(state.get("drift_z", 0.0) / 4.0)
    f[2] = state.get("graph_churn", 0.0)
    f[3] = state.get("suspicion", 0.0)
    f[4] = np.tanh(max(0.0, abs(state.get("pmv", 0.0)) - 0.5))
    f[5] = np.tanh(np.log1p(max(0.0, state.get("latency_ms", 0.0) - 120.0)) / 4.0)
    f[6] = 1.0 - state.get("pdr", 1.0)
    f[7] = np.tanh(state.get("reward_trend", 0.0))
    f[8] = np.tanh(state.get("occ_var", 0.0) / 8.0)
    h = 2 * np.pi * (state.get("t", 0) % 288) / 288.0
    f[9], f[10] = np.sin(h), np.cos(h)
    f[11] = 1.0
    return f


# ------------------------------------------------------------------ gates
class MetaController:
    name = "base"
    trainable = False

    def should_query(self, state: Dict) -> bool:
        raise NotImplementedError

    def observe(self, state: Dict, queried: bool, realised_gain: float):
        pass

    def reset_episode(self):
        """Clear per-episode bookkeeping but keep everything that was learned.

        Episode time restarts at zero, so any state that compares against a
        step index has to be cleared here; leaving it stale silently disables
        the gate for the rest of the run.
        """
        pass

    def stats(self) -> Dict:
        return {}


class AlwaysOn(MetaController):
    name = "always"

    def __init__(self, **kw):
        pass

    def should_query(self, state):
        return True


class NeverQuery(MetaController):
    name = "never"

    def __init__(self, **kw):
        pass

    def should_query(self, state):
        return False


class PeriodicQuery(MetaController):
    name = "periodic"

    def __init__(self, period: int = 12, **kw):
        self.period = period
        self.name = f"periodic-{period}"

    def should_query(self, state):
        return (state.get("t", 0) % self.period) == 0


class RandomQuery(MetaController):
    name = "random"

    def __init__(self, p: float = 0.1, seed: int = 0, **kw):
        self.p, self.rng = p, np.random.default_rng(seed)
        self.name = f"random-{p:g}"

    def should_query(self, state):
        return bool(self.rng.random() < self.p)


class _Isotonic:
    """Monotone calibration map, used by EntropyGate."""

    def __init__(self):
        self.x = None
        self.y = None

    def fit(self, x, y):
        from sklearn.isotonic import IsotonicRegression
        if len(x) < 12:
            return
        ir = IsotonicRegression(out_of_bounds="clip")
        ir.fit(np.asarray(x, float), np.asarray(y, float))
        self.model = ir
        self.x = True

    def __call__(self, v):
        if self.x is None:
            return float(v)
        return float(self.model.predict([v])[0])


class EntropyGate(MetaController):
    """UCCI-style: escalate when the cheap path's calibrated error probability
    is high enough that the expected loss exceeds the invocation price."""
    name = "entropy-gate"
    trainable = True

    def __init__(self, lam: float = 1.0, cost: float = 0.05,
                 warmup: int = 40, seed: int = 0, **kw):
        self.lam, self.cost, self.warmup = lam, cost, warmup
        self.cal = _Isotonic()
        self.xs: List[float] = []
        self.ys: List[float] = []
        self.n = 0

    @staticmethod
    def _uncertainty(state: Dict) -> float:
        # cheap proxy for "the cached causal plan may be wrong here"
        return float(np.tanh(0.5 * abs(state.get("drift_z", 0.0))
                             + 2.0 * state.get("graph_churn", 0.0)
                             + state.get("suspicion", 0.0)))

    def should_query(self, state):
        self.n += 1
        u = self._uncertainty(state)
        if self.n < self.warmup:
            return bool(self.n % 8 == 0)
        p_err = self.cal(u)
        return bool(p_err * self.lam > self.cost)

    def observe(self, state, queried, realised_gain):
        if not queried:
            return
        self.xs.append(self._uncertainty(state))
        self.ys.append(1.0 if realised_gain > 0 else 0.0)
        if len(self.xs) % 20 == 0:
            self.cal.fit(self.xs, self.ys)

    def stats(self):
        return {"calibration_points": len(self.xs)}


class MarginGate(MetaController):
    """CR^2-style: a linear margin score on frozen features plus a
    risk-controlled acceptance threshold chosen to hold the false-acceptance
    rate at `risk`."""
    name = "margin-gate"
    trainable = True

    def __init__(self, risk: float = 0.15, warmup: int = 48, seed: int = 0,
                 **kw):
        self.risk, self.warmup = risk, warmup
        self.w = np.zeros(NFEAT)
        self.A = np.eye(NFEAT) * 1.0
        self.b = np.zeros(NFEAT)
        self.scores: List[float] = []
        self.n = 0
        self.thresh = 0.0

    def _score(self, state):
        return float(self.w @ build_features(state))

    def should_query(self, state):
        self.n += 1
        if self.n < self.warmup:
            return bool(self.n % 6 == 0)
        return bool(self._score(state) > self.thresh)

    def observe(self, state, queried, realised_gain):
        if not queried:
            return
        x = build_features(state)
        self.A += np.outer(x, x)
        self.b += realised_gain * x
        self.w = np.linalg.solve(self.A, self.b)
        self.scores.append(self._score(state))
        if len(self.scores) >= 16:
            # conformal-style threshold: accept only the top (1 - risk) tail
            self.thresh = float(np.quantile(self.scores, self.risk))

    def stats(self):
        return {"threshold": self.thresh, "n_obs": len(self.scores)}


@dataclass
class EVOIConfig:
    lam: float = 1.0            # price multiplier on the invocation cost
    ridge: float = 1.0
    warmup: int = 40
    explore_p: float = 0.10     # forced exploration so the value model sees
                                # states it would otherwise never query in
    explore_decay: float = 250.0  # observations after which forced exploration
                                  # has fallen to half; without this the gate
                                  # keeps paying an exploration tax forever
    explore_p_min: float = 0.004
    min_gap: int = 2            # never fire twice within this many steps
    horizon: int = 12


class EVOIGate(MetaController):
    """Ours: fire when the estimated expected value of information exceeds the
    price of the invocation.

    The value model is online ridge regression on the features above, trained
    on the realised H-step advantage of having queried. Forced exploration
    keeps the regression identifiable: without it the gate only ever observes
    outcomes in states it already believed were worth querying.
    """
    name = "evoi"
    trainable = True

    def __init__(self, cfg: EVOIConfig = None, cost: float = 0.05,
                 seed: int = 0, **kw):
        self.cfg = cfg or EVOIConfig()
        self.cost = cost
        self.rng = np.random.default_rng(seed)
        self.A = np.eye(NFEAT) * self.cfg.ridge
        self.b = np.zeros(NFEAT)
        self.w = np.zeros(NFEAT)
        self.n = 0
        self.n_obs = 0
        self.last_fire = -(10 ** 6)
        self.name = f"evoi(lam={self.cfg.lam:g})"

    def reset_episode(self):
        self.last_fire = -(10 ** 6)

    def _explore_p(self) -> float:
        """Forced exploration, decaying with the evidence already collected.

        A constant rate is a permanent tax: in a settled regime the gate would
        keep buying invocations it has already learned are not worth buying.
        Decaying it lets the query rate fall toward zero where the value model
        is confident, while still recovering when a regime change makes the
        model wrong again -- the value estimate rises and the gate fires on
        merit rather than on the coin flip.
        """
        c = self.cfg
        return max(c.explore_p_min,
                   c.explore_p * 0.5 ** (self.n_obs / c.explore_decay))

    def value(self, state) -> float:
        return float(self.w @ build_features(state))

    def should_query(self, state):
        self.n += 1
        t = state.get("t", self.n)
        if t - self.last_fire < self.cfg.min_gap:
            return False
        if self.n_obs < self.cfg.warmup:
            fire = bool(self.rng.random() < 0.18)
        elif self.rng.random() < self._explore_p():
            fire = True
        else:
            fire = bool(self.value(state) > self.cfg.lam * self.cost)
        if fire:
            self.last_fire = t
        return fire

    def observe(self, state, queried, realised_gain):
        if not queried:
            return
        x = build_features(state)
        self.A += np.outer(x, x)
        self.b += realised_gain * x
        self.w = np.linalg.solve(self.A, self.b)
        self.n_obs += 1

    def stats(self):
        return {"n_obs": self.n_obs, "explore_p": self._explore_p(),
                "weights": dict(zip(FEATURE_NAMES, self.w.round(4).tolist()))}


META_GATES = {
    "always": AlwaysOn,
    "never": NeverQuery,
    "periodic": PeriodicQuery,
    "random": RandomQuery,
    "entropy-gate": EntropyGate,
    "margin-gate": MarginGate,
    "evoi": EVOIGate,
}

"""Every comparison method, behind one interface: act(obs) / observe(obs).

Methods that share CauProbe's belief (greedy, Thompson sampling, IDS,
identify-then-act, log-naive, nominal model, oracle) differ from it *only* in
the decision rule, so a difference in outcome is attributable to that rule and
not to a better model. Methods that do not (fixed, rule-based, LinUCB,
advice-as-plan) are the practical alternatives a deployment would actually
consider.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from scipy.special import logsumexp

from . import mech
from .agent import CauProbeAgent, ProbeConfig
from .belief import NetBelief, ThermalBelief, expected_occ, student_t_logpdf
from .mech import N_NET, N_SCHED, NET_ACTIONS, SCHED_SP, STEPS_PER_DAY
from .plants import (FALLBACK_NET, FALLBACK_SCHED, INCUMBENT_SCHEDULE,
                     RuleController, sample_net_params, sample_th_params)


# ------------------------------------------------------------ fixed / rules
class FixedAgent:
    def __init__(self, net_idx: int, sched: int, name: str = "fixed"):
        self.net_idx, self.sched, self.name = net_idx, sched, name
        self.t = 0
        self.n_asks = self.n_probe_steps = self.n_fallback = self.n_th_probes = 0

    def act(self, obs):
        k = self.t % STEPS_PER_DAY
        return self.net_idx, float(SCHED_SP[self.sched][k]), False, {}

    def observe(self, obs, busy=0.0):
        self.t += 1


class RuleAgent(FixedAgent):
    def __init__(self):
        super().__init__(FALLBACK_NET, INCUMBENT_SCHEDULE, "rule-based")
        self.rc = RuleController()

    def act(self, obs):
        a, s = self.rc.act(obs)
        return a, float(SCHED_SP[s][self.t % STEPS_PER_DAY]), False, {}


# ------------------------------------------------------------ same belief
def greedy_cfg(**kw):
    return ProbeConfig(use_ask=False, use_probe=False, use_thermal_probe=False,
                       **kw)


class OracleAgent(CauProbeAgent):
    """Known-plant reference: the same decision rule with the plant's true
    mechanisms and parameters, and otherwise the same information as every
    other method (expected occupancy, no view of noise or bursts)."""
    name = "oracle"

    def __init__(self, spec, seed=0):
        super().__init__(greedy_cfg(misspec_detect=False), seed=seed)
        self.nb = NetBelief.from_params(spec.net, occ_prior=self.cfg.occ_prior)
        self.tb = ThermalBelief.from_params(spec.th, occ_prior=self.cfg.occ_prior)

    def on_regime(self, spec):
        """The reference knows the plant, including when it is rebuilt."""
        self.nb = NetBelief.from_params(spec.net, occ_prior=self.cfg.occ_prior)
        self._U_key = None

    def bind_plant_privileged(self, plant):
        """Not used for the reported reference: with the true occupancy the
        myopic controller chases occupancy fluctuations and pays for the
        reconfigurations, which makes it a worse reference, not a better one
        (experiments/cp_debug_oracle.py)."""
        self.occ_override = lambda hour: float(
            plant.occ_day[int(round(hour * 12)) % STEPS_PER_DAY])
        self.occ_day_ref = lambda: plant.occ_day


class NominalModelAgent(CauProbeAgent):
    """An engineer's model: the integrator's default assumptions (Wi-Fi-like
    contention MAC, edge-bound processing, self-loaded broker, heavy mass)
    with mid-range parameters, never revised. Model-predictive control on a
    fixed model."""
    name = "nominal-model"

    def __init__(self, seed=0, combo=(0, 0, 0, 0)):
        super().__init__(greedy_cfg(misspec_detect=False), seed=seed)
        rng = np.random.default_rng(12345)
        a, b, c, d = combo
        Pn = sample_net_params(rng, a, b, c, n=256)
        Pt = sample_th_params(rng, d, n=256)
        self.nb = NetBelief.from_params({k: np.median(v) for k, v in Pn.items()},
                                        occ_prior=self.cfg.occ_prior)
        self.tb = ThermalBelief.from_params({k: np.median(v) for k, v in Pt.items()},
                                            occ_prior=self.cfg.occ_prior)
        self.nb.eta = 0.0
        self.tb.eta = 0.0


class ThompsonAgent(CauProbeAgent):
    """Posterior sampling on the same belief: act optimally for one sampled
    hypothesis, resampled every hour (network) and every day (thermal)."""
    name = "thompson"

    def __init__(self, safe: bool = True, seed=0, **kw):
        super().__init__(greedy_cfg(use_safety=safe, **kw), seed=seed)
        self.name = "thompson" + ("" if safe else "-unsafe")
        self._i = None

    def act(self, obs):
        k = self.t % STEPS_PER_DAY
        if k == 0:
            self._plan_day_ts(obs)
        sp = float(self.day_sp[k])
        if self.misspec:
            self.n_fallback += 1
            return FALLBACK_NET, sp, False, {"fallback": True}
        if self._i is None or self.t % 12 == 0:
            self._i = int(self.rng.choice(self.nb.n, p=self.nb.w))
        idx = np.array([self._i])
        pdr = obs["pdr"] if obs is not None else 0.93
        R, pv, _, _ = self._net_tables(k / 12.0, pdr, idx=idx)
        prev = int(obs["net_idx"]) if obs is not None else None
        r = R[:, 0] - mech.switch_cost_vec(prev) / self.cfg.dwell
        if self.cfg.use_safety:
            idx2, w = self._active()
            _, pv2, _, _ = self._net_tables(k / 12.0, pdr, idx=idx2)
            ok = self._admissible(pv2 @ w)
            if not ok.any():
                self.n_fallback += 1
                return FALLBACK_NET, sp, False, {"fallback": True}
            r = np.where(ok, r, -np.inf)
        return int(np.argmax(r)), sp, False, {}

    def _plan_day_ts(self, obs):
        if self.th_misspec:
            self.sched = self.fallback_sched
            self.day_sp = self.sched_sp[self.sched].copy()
            return
        Uth = self._th_values()
        i = int(self.rng.choice(self.tb.n, p=self.tb.w))
        self.sched = int(np.argmax(Uth[i]))
        self.day_sp = self.sched_sp[self.sched].copy()


class IDSAgent(CauProbeAgent):
    """Information-directed sampling (deterministic variant): minimise
    expected-regret^2 / information gain about the optimal action, on the
    same belief and the same admissible set."""
    name = "ids"

    def __init__(self, seed=0, **kw):
        super().__init__(greedy_cfg(**kw), seed=seed)

    def act(self, obs):
        k = self.t % STEPS_PER_DAY
        if k == 0:
            self._plan_day(obs)
        sp = float(self.day_sp[k])
        if self.misspec:
            self.n_fallback += 1
            return FALLBACK_NET, sp, False, {"fallback": True}
        idx, w = self._active()
        pdr = obs["pdr"] if obs is not None else 0.93
        R, pv, mu, s = self._net_tables(k / 12.0, pdr, idx=idx)
        ok = self._admissible(pv @ w)
        if not ok.any():
            self.n_fallback += 1
            return FALLBACK_NET, sp, False, {"fallback": True}
        prev = int(obs["net_idx"]) if obs is not None else None
        R = R - mech.switch_cost_vec(prev)[:, None] / self.cfg.dwell
        Rm = np.where(ok[:, None], R, -np.inf)
        astar = Rm.argmax(0)                               # per particle
        regret = (w * Rm.max(0)).sum() - R @ w             # (A,)
        pA = np.bincount(astar, weights=w, minlength=N_NET)
        H0 = -np.sum(pA * np.log(pA + 1e-12))
        if H0 < 1e-3:
            return int(np.argmax(np.where(ok, R @ w, -np.inf))), sp, False, {}
        M = self.cfg.mc
        j = self.rng.choice(len(w), size=M, p=w)
        eps = self.rng.standard_normal(M)
        best, best_ratio = None, np.inf
        for a in np.flatnonzero(ok):
            y = mu[a, j] + s[0, j] * eps
            ll = student_t_logpdf(y[:, None] - mu[a][None, :], s[0][None, :])
            lw = np.log(w)[None, :] + self.nb.eta * ll
            W2 = np.exp(lw - logsumexp(lw, axis=1, keepdims=True))
            pc = np.stack([np.bincount(astar, weights=r, minlength=N_NET)
                           for r in W2])
            g = H0 + np.mean(np.sum(pc * np.log(pc + 1e-12), 1))
            ratio = max(regret[a], 0.0) ** 2 / max(g, 1e-9)
            if ratio < best_ratio:
                best, best_ratio = int(a), ratio
        a = best if best is not None else int(np.argmax(np.where(ok, R @ w,
                                                                 -np.inf)))
        if a != int(np.argmax(np.where(ok, R @ w, -np.inf))):
            self.n_probe_steps += 1
        return a, sp, False, {}


# ------------------------------------------------------------ bandits
class LinUCBAgent:
    """Joint-arm LinUCB over the 36 network actions, plus UCB1 over the daily
    thermal schedule templates. The reward it learns from is the part of the
    reward the gateway can observe."""
    name = "linucb"

    def __init__(self, alpha: float = 0.6, seed: int = 0, advice=None):
        self.alpha = alpha
        self.d = 7
        self.A = np.stack([np.eye(self.d) for _ in range(N_NET)])
        self.b = np.zeros((N_NET, self.d))
        self.rng = np.random.default_rng(seed)
        self.t = 0
        self.ns = np.zeros(N_SCHED)
        self.rs = np.zeros(N_SCHED)
        self.sched = INCUMBENT_SCHEDULE
        self.day_r = 0.0
        self._x = None
        self._a = None
        self.advice = advice or {}
        self.n_asks = self.n_probe_steps = self.n_fallback = self.n_th_probes = 0

    def _ctx(self, obs, k):
        h = 2 * np.pi * k / STEPS_PER_DAY
        if obs is None:
            return np.array([1, np.sin(h), np.cos(h), 1.0, 0.0, 0.93, 0.0])
        return np.array([1.0, np.sin(h), np.cos(h), obs["tariff"] / 2.4,
                         np.log(obs["latency_ms"]) / 6.0, obs["pdr"],
                         obs["it_j"] / 1500.0])

    def act(self, obs):
        k = self.t % STEPS_PER_DAY
        if k == 0:
            if self.t > 0:
                self.ns[self.sched] += 1
                self.rs[self.sched] += self.day_r
            self.day_r = 0.0
            day = self.t // STEPS_PER_DAY
            if day == 0 and "sched" in self.advice:
                self.sched = self.advice["sched"]
            elif (self.ns == 0).any() and day < N_SCHED:
                self.sched = int(self.rng.choice(np.flatnonzero(self.ns == 0)))
            else:
                n = np.maximum(self.ns, 1)
                ucb = self.rs / n + 2.0 * np.sqrt(np.log(day + 1) / n)
                self.sched = int(np.argmax(ucb))
        x = self._ctx(obs, k)
        if "net" in self.advice and self.t < self.advice.get("net_steps", 0):
            a = self.advice["net"]
        else:
            Ainv = np.linalg.inv(self.A)
            theta = np.einsum("aij,aj->ai", Ainv, self.b)
            ucb = theta @ x + self.alpha * np.sqrt(
                np.einsum("i,aij,j->a", x, Ainv, x))
            a = int(np.argmax(ucb))
        self._x, self._a = x, a
        return a, float(SCHED_SP[self.sched][k]), False, {}

    def observe(self, obs, busy=0.0):
        r, _ = mech.net_reward(obs["latency_ms"], obs["it_j"], obs["aoi"],
                               obs["pdr"])
        self.A[self._a] += np.outer(self._x, self._x)
        self.b[self._a] += float(r) * self._x
        # observable thermal signal: tariff-weighted energy and a PMV proxy
        occ = expected_occ(obs["hour"])
        rt, _ = mech.thermal_reward(obs["e_kwh"], obs["hour"],
                                    mech.pmv_fast(obs["t_air"]), occ, obs["t_air"])
        self.day_r += float(rt)
        self.t += 1


def advice_from_expert(z: Dict[str, float], seed: int = 0):
    """The prior paper's use of the model: take its most likely mechanism,
    compute the best action for that mechanism under nominal parameters, and
    install it as a plan."""
    combo = tuple(int(z[F] > 0) for F in "ABCD")
    nom = NominalModelAgent(seed=seed, combo=combo)
    R, pv, _, _ = nom._net_tables(12.0, 0.9)
    ok = nom._admissible(pv[:, 0])
    net = int(np.argmax(np.where(ok, R[:, 0], -np.inf)))
    fc = np.full(STEPS_PER_DAY, 27.0) + 5.0 * np.sin(
        2 * np.pi * (np.arange(STEPS_PER_DAY) / 12.0 - 9.0) / 24.0)
    nom.tb.t_mass = np.full(1, 26.0)
    U = nom.tb.simulate_day(SCHED_SP, fc, 26.0)[:, 0]
    return {"net": net, "net_steps": STEPS_PER_DAY, "sched": int(np.argmax(U))}

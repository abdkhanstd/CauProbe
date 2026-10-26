"""Posterior over structural hypotheses, as weighted particles.

A particle is a complete structural causal model of one subsystem: a mechanism
choice (which edges exist and with which sign) plus the continuous parameters
of its equations. The weights are updated from data the gateway generated
itself -- its own actions given the context it observed -- which is
interventional data, so the update is unconfounded. Historical logs produced by
the incumbent system are *not* interventional (the incumbent's knob settings
tracked occupancy, which the gateway never sees) and are only used by the
`log-naive` baseline, to show what goes wrong when they are treated as if they
were.

Two practical departures from textbook Bayes, both standard and both reported:
the likelihood is tempered (a power posterior, exponent `eta`), which keeps the
posterior from collapsing onto a slightly wrong particle under model
misspecification; and a fixed-share step mixes a little prior mass back in
each step, which is the Bayesian way of saying the plant might change.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from scipy.special import gammaln, logsumexp

from . import mech
from .mech import STEPS_PER_DAY
from .plants import ALL_COMBOS, sample_net_params, sample_th_params

NET_COMBOS = sorted({(a, b, c) for a, b, c, _ in ALL_COMBOS})
TH_COMBOS = [0, 1]


def student_t_logpdf(r, scale, nu=4.0):
    z = r / scale
    return (gammaln((nu + 1) / 2) - gammaln(nu / 2) - 0.5 * np.log(nu * np.pi)
            - np.log(scale) - (nu + 1) / 2 * np.log1p(z * z / nu))


def expected_occ(hour, occ_prior: float = 10.0):
    return occ_prior * mech.office_gate(hour)


class _ParticleSet:
    """Shared machinery: weights, tempering, fixed share, rejuvenation."""

    def __init__(self, n: int, eta: float, hazard: float, seed: int):
        self.n = n
        self.eta = eta
        self.hazard = hazard
        self.rng = np.random.default_rng(seed)
        self.logw = np.full(n, -np.log(n))
        self.log_prior = self.logw.copy()

    @property
    def w(self):
        return np.exp(self.logw - logsumexp(self.logw))

    def ess(self):
        w = self.w
        return 1.0 / np.sum(w * w)

    def _apply(self, loglik):
        self.logw = self.logw + self.eta * loglik
        self.logw -= logsumexp(self.logw)
        if self.hazard > 0:
            self.logw = np.logaddexp(np.log1p(-self.hazard) + self.logw,
                                     np.log(self.hazard) + self.log_prior)
            self.logw -= logsumexp(self.logw)

    def posterior_after(self, loglik):
        """Weights after a hypothetical observation, without committing it."""
        lw = self.logw[None, :] + self.eta * loglik
        return np.exp(lw - logsumexp(lw, axis=1, keepdims=True))


class NetBelief(_ParticleSet):
    """Particles over network mechanisms (A, B, C) and their parameters."""

    def __init__(self, per_combo: int = 128, eta: float = 0.5,
                 hazard: float = 2e-4, seed: int = 0, occ_prior: float = 10.0,
                 noise_extra: float = 0.22, combos=None):
        combos = combos or NET_COMBOS
        n = per_combo * len(combos)
        super().__init__(n, eta, hazard, seed)
        parts = [sample_net_params(self.rng, a, b, c, n=per_combo)
                 for a, b, c in combos]
        self.P = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
        # The gateway does not know how noisy its telemetry will be: the
        # particle prior on the noise level is wider than any plant's.
        self.P["sigma"] = self.rng.uniform(0.10, 0.60, n)
        self.combo_id = np.repeat(np.arange(len(combos)), per_combo)
        self.combos = combos
        self.occ_prior = occ_prior
        self.noise_extra = noise_extra
        self.pdr_by_hour = np.full(24, 0.93)
        self._pdr_n = np.zeros(24)

    @classmethod
    def from_params(cls, P: Dict, **kw):
        """A single-particle 'belief': the true plant (oracle) or a nominal model."""
        b = cls(per_combo=1, **kw)
        b.P = {k: np.atleast_1d(np.asarray(v, float)) for k, v in P.items()}
        b.combos = [(int(b.P["A"][0]), int(b.P["B"][0]), int(b.P["C"][0]))]
        b.combo_id = np.zeros(1, int)
        b.n = 1
        b.logw = np.zeros(1)
        b.log_prior = np.zeros(1)
        return b

    # ------------------------------------------------------------ prediction
    def predict(self, a_idx, hour, pdr, busy=0.0, P=None):
        """Mean log-latency and IT energy of every particle under action a."""
        P = self.P if P is None else P
        f, d, o = mech.NET_ACTIONS[a_idx]
        lat, it_j, aoi = mech.net_outcomes(P, f, d, o, hour,
                                           expected_occ(hour, self.occ_prior),
                                           pdr, busy=busy)
        return np.log(lat), it_j

    def obs_scale(self):
        return np.sqrt(self.P["sigma"] ** 2 + self.noise_extra ** 2)

    def loglik(self, obs, busy=0.0):
        mu, it_pred = self.predict(obs["net_idx"], obs["hour"], obs["pdr"], busy)
        ll = student_t_logpdf(np.log(obs["latency_ms"]) - mu, self.obs_scale())
        ll = ll + student_t_logpdf(obs["it_j"] - it_pred, 60.0)
        return ll

    def update(self, obs, busy=0.0):
        hb = int(obs["hour"]) % 24
        self._pdr_n[hb] += 1
        self.pdr_by_hour[hb] += (obs["pdr"] - self.pdr_by_hour[hb]) \
            / min(self._pdr_n[hb], 50.0)
        self._apply(self.loglik(obs, busy))

    # ------------------------------------------------------------ marginals
    def factor_marginals(self):
        w = self.w
        out = {}
        for fi, name in enumerate("ABC"):
            vals = np.array([c[fi] for c in self.combos])[self.combo_id]
            out[name] = float(w[vals == 1].sum())
        return out

    def combo_post(self):
        return np.bincount(self.combo_id, weights=self.w,
                           minlength=len(self.combos))

    def factor_values(self, name):
        fi = "ABC".index(name)
        return np.array([c[fi] for c in self.combos])[self.combo_id]


class ThermalBelief(_ParticleSet):
    """Particles over thermal mass (D) and the zone's 2R2C parameters.

    Each particle carries its own estimate of the unobserved slab temperature,
    propagated through its own mass equation from the measured air
    temperature: given the parameters, the slab state is a deterministic
    function of the air-temperature history, so no inner filter is needed.
    """

    def __init__(self, per_combo: int = 256, eta: float = 0.5,
                 hazard: float = 1e-4, seed: int = 0, occ_prior: float = 10.0,
                 sigma: float = 0.12, ranges=None, reward_fn=None):
        n = per_combo * 2
        super().__init__(n, eta, hazard, seed)
        if ranges is None:
            parts = [sample_th_params(self.rng, D, n=per_combo) for D in (0, 1)]
        else:
            # site-specific prior (e.g. a 48 m2 BOPTEST zone instead of a floor)
            per_class, common = ranges
            parts = []
            for D in (0, 1):
                P = {"D": np.full(per_combo, D)}
                for spec in (per_class[D], common):
                    for k, (lo, hi) in spec.items():
                        P[k] = self.rng.uniform(lo, hi, per_combo)
                parts.append(P)
        self.reward_fn = reward_fn
        self.fit_controller = False
        self.P = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
        self.combo_id = np.repeat(np.arange(2), per_combo)
        self.occ_prior = occ_prior
        self.sigma = sigma
        self.t_mass = None
        self._last = None

    @classmethod
    def from_params(cls, P: Dict, **kw):
        b = cls(per_combo=1, **kw)
        b.P = {k: np.atleast_1d(np.asarray(v, float)) for k, v in P.items()}
        b.combo_id = np.array([int(b.P["D"][0])])
        b.n = 1
        b.logw = np.zeros(1)
        b.log_prior = np.zeros(1)
        return b

    def predict_next(self, t_air, t_out, hvac_kw, hour):
        occ = expected_occ(hour, self.occ_prior)
        ta, tm = mech.thermal_step(self.P, t_air, self.t_mass, t_out, hvac_kw,
                                   occ, hour)
        return ta, tm

    def update(self, obs):
        """Score the transition from the previous observation to this one."""
        if self._last is None or self.t_mass is None:
            self.t_mass = np.full(self.n, obs["t_air"])
            self._last = obs
            return
        # obs[k] reports the air temperature *after* step k, which was driven
        # by step k's outdoor temperature, HVAC power and hour.
        prev = self._last
        ta_pred, tm_new = self.predict_next(prev["t_air"], obs["t_out"],
                                            obs["hvac_kw"], obs["hour"])
        occ_sd = mech.office_gate(obs["hour"]) * 3.0
        scale = np.sqrt(self.sigma ** 2 + (self.P["occ_gain"] * occ_sd
                                           * mech.DT / self.P["C_air"]) ** 2)
        ll = student_t_logpdf(obs["t_air"] - ta_pred, scale)
        if self.fit_controller and "setpoint" in obs:
            # The zone controller's response is logged too: the HVAC power it
            # drew given the temperature and the setpoint. Where the controller
            # is unknown (BOPTEST), this is the only evidence about its gain
            # and dead band, which the planner needs; conditioning on the
            # measured power alone leaves them at the prior.
            hv = mech.hvac_power(prev["t_air"], obs["setpoint"], self.P)
            ll = ll + student_t_logpdf(obs["hvac_kw"] - hv, 0.1 + 0.1 * hv)
        self._apply(ll)
        # re-anchor each particle's slab estimate on the measured air temperature
        self.t_mass = tm_new
        self._last = obs

    def mass_post(self):
        return float(self.w[self.combo_id == 0].sum())     # P(heavy)

    def factor_values(self, name="D"):
        return self.combo_id.copy()

    # ------------------------------------------------------------ planning
    def simulate_day(self, setpoints, t_out, t_air0, t_mass0=None, P=None,
                     occ=None, hours=None):
        """Roll every particle forward under one or more setpoint profiles.

        `setpoints` is (S, T); returns per-particle-per-schedule total reward
        (S, N) using the planner's PMV approximation and expected occupancy.
        """
        P = self.P if P is None else P
        S, T = setpoints.shape
        N = len(P["C_air"])
        hours = np.arange(T) / 12.0 if hours is None else hours
        occ = expected_occ(hours, self.occ_prior) if occ is None else occ
        Pb = {k: np.broadcast_to(v, (S, N)) for k, v in P.items()}
        ta = np.broadcast_to(np.asarray(t_air0, float), (S, N)).copy()
        tm = (np.broadcast_to(np.asarray(t_mass0 if t_mass0 is not None
                                         else self.t_mass), (S, N)).copy())
        tot = np.zeros((S, N))
        for k in range(T):
            sp = setpoints[:, k][:, None]
            hv = mech.hvac_power(ta, sp, Pb)
            ta, tm = mech.thermal_step(Pb, ta, tm, t_out[k], hv, occ[k],
                                       hours[k])
            if self.reward_fn is not None:
                tot += self.reward_fn(hv, hours[k], ta, occ[k])
                continue
            e = mech.zone_energy_kwh(hv, occ[k])
            r, _ = mech.thermal_reward(e, hours[k], mech.pmv_fast(ta), occ[k], ta)
            tot += r
        return tot

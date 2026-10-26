"""CauProbe: act, ask, probe or fall back, by decision-relevant value of information.

Per step the network side does the following:

1. score every joint network action under every particle at the current
   context (hour, measured delivery ratio): expected reward, and probability
   of a hard violation;
2. keep only actions whose posterior violation probability is <= delta; if
   none survives, or the plant has been flagged as out-of-library, use the
   verified fallback;
3. screen: the value of *any* information is bounded by the expected value of
   perfect information, so if H * EVPI is negligible, act greedily and spend
   nothing on learning (this is what makes the steady state free);
4. otherwise pick the action maximising
       Q(a) + max_m VOI_m(a) / m,
   where VOI_m(a) is the gain in the value of every remaining decision (all
   hours of the day, over a durability horizon of H days) from observing m
   repetitions of a. With m in {1, 4} this is the KG(*) rule of Frazier &
   Powell, which corrects the single-sample under-valuation of information.
   A "probe" is simply an action taken for this second term.

Asking the language model is a separate information action, available while
there is unread text. Its value is computed the same way, with the expert's
calibrated output distribution in place of a plant observation, and it is
taken iff that value exceeds its price in Joules (plus contention).

The thermal side decides once a day, at midnight: which schedule template to
run, and whether to spend the first three hours of the night on a step test
whose response separates heavy from light construction.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
from scipy.special import logsumexp
from scipy.stats import norm

from . import mech
from .belief import NetBelief, ThermalBelief, expected_occ, student_t_logpdf
from .mech import N_NET, N_SCHED, NET_ACTIONS, SCHED_SP, STEPS_PER_DAY
from .plants import FALLBACK_NET, FALLBACK_SCHED

GH_X, GH_W = np.polynomial.hermite_e.hermegauss(3)      # N(0,1) quadrature
GH_W = GH_W / GH_W.sum()
GH5_X, GH5_W = np.polynomial.hermite_e.hermegauss(5)
GH5_W = GH5_W / GH5_W.sum()
from scipy.stats import t as _t4
T4_ENTROPY = float(_t4(4).entropy())                    # standard Student-t, nu=4
BURST_P = 0.06                                          # tenant bursts
BURST_MULT = 2.75
CTX_HOURS = np.arange(1.0, 24.0, 2.0)                   # 12 two-hour contexts
CTX_STEPS = 24                                          # steps per context


@dataclass
class ProbeConfig:
    h_days: float = 30.0          # durability of structural knowledge
    delta: float = 0.01           # admissible violation probability per step
    dwell: float = 12.0           # steps a reconfiguration is amortised over
    use_ask: bool = True
    use_probe: bool = True        # VOI-driven network actions
    use_thermal_probe: bool = True
    use_safety: bool = True
    screen_eps: float = 1e-3      # skip VOI when H*EVPI is below this
    mc: int = 32
    kg_m: tuple = (1, 4, 12)     # probe durations considered (KG*)
    quad_particles: int = 48
    n_cand: int = 8
    misspec_detect: bool = True
    cusum_drift: float = 0.5      # nats of excess surprise tolerated per step
    cusum_h: float = 21.6         # alarm threshold: 1.6% false alarms per week
                                  # on dev plants (cp_calib_detector.py)
    surprise_clip: float = 4.0
    misspec_exit: float = 0.25    # mean excess surprise to leave fallback
    ticket_reset: float = 0.5     # prior mass a change ticket re-injects
    self_calibrate: bool = True   # scale VOI by realised/predicted information
    kappa_mode: str = "daily"     # "daily" audit | "probe" (per-probe check)
    kappa_delay: int = 144        # steps after a probe before its information
                                  # is judged (or until the next probe starts)
    voi_level: str = "structure"  # "structure" | "particle" (ablation)
    force_ask: bool = False       # ablation: read every text whatever it costs
    # thermal out-of-library detector (off by default; see paper Sec. 7)
    thermal_detect: bool = False
    thermal_shadow: bool = False  # compute and record, never act
    th_drift: float = 0.1         # nats of excess surprise tolerated
    th_clip: float = 3.0
    th_h: float = 20.0            # alarm threshold (calibrated on dev plants)
    th_exit: float = -0.02        # mean excess surprise to leave fallback
    durability: bool = True       # hazard-aware information horizon
    ask_price: float = 0.15       # reward units per invocation (set by driver)
    objective: str = "decision"   # "decision" | "identify"
    identify_target: float = 0.95
    per_combo_net: int = 128
    per_combo_th: int = 256
    eta: float = 0.5
    occ_prior: float = 10.0
    hazard_net: float = 4e-5      # prior rate of structural change per step
    hazard_th: float = 2e-5
    # Second iteration (v2, developed after the first test runs; off by
    # default so that the first iteration stays reproducible). Every source is
    # valued net of what greedy operation would reveal anyway: the gain of
    # knowing that structure c holds is counted only for as long as greedy
    # control, if c were true, would still be acting on a wrong decision.
    # That time is estimated by simulated greedy rollouts under the current
    # posterior (_free_horizon).
    free_learning: bool = False
    fl_days: int = 2              # length of the simulated rollouts
    fl_truths: int = 8            # simulated truths per live structure
    fl_min_mass: float = 1e-3     # structures below this get no credit
    fl_obs: int = 8               # simulated observations per two-hour block
    fl_refresh: int = 24          # steps between re-estimates of the horizon
    fl_tail: str = "hold"         # beyond the rollout: "hold" the last day's
                                  # regret, or extrapolate each rollout's own
                                  # decay ("geometric")
    fl_ask_truths: int = 32       # rollouts per structure when valuing a reading
    fl_ask_mc: int = 64           # simulated readings per structure
    # Information is also lost to what operation does: where the posterior
    # re-opens a question faster than the change hazard says it should (a
    # library that does not quite fit the site), ruling it out lasts less.
    # v2 measures the drift of the off-leading structural mass over steps
    # without probes (an exponential average over reopen_tau steps) and uses
    # it in the durability horizon whenever it exceeds the prior hazard.
    reopen_rate: bool = False
    reopen_tau: float = 288.0
    # A probe is valued as an excursion: switch in, hold m steps, switch back.
    # The first iteration did not execute the switch back -- after a probe the
    # switching-aware greedy rule decided from the probe's configuration, and
    # where returning gained less per step than its amortised cost (LPWAN
    # prices) the gateway stayed there. v2 returns.
    probe_return: bool = False


class CauProbeAgent:
    name = "cauprobe"

    def __init__(self, cfg: ProbeConfig = None, expert=None, seed: int = 0,
                 name: Optional[str] = None):
        self.cfg = cfg or ProbeConfig()
        self.expert = expert            # calibrated expert model (see expert.py)
        self.rng = np.random.default_rng(seed)
        self.nb = NetBelief(per_combo=self.cfg.per_combo_net, eta=self.cfg.eta,
                            hazard=self.cfg.hazard_net, seed=seed,
                            occ_prior=self.cfg.occ_prior)
        self.tb = ThermalBelief(per_combo=self.cfg.per_combo_th,
                                eta=self.cfg.eta, hazard=self.cfg.hazard_th,
                                seed=seed + 1, occ_prior=self.cfg.occ_prior)
        self._commit = None
        self.occ_override = None      # privileged reference controller only
        # Self-calibration of the value of information. Under a correctly
        # specified model the expected drop in EVPI over a probe equals the
        # probe's predicted VOI (the posterior is a martingale). kappa is the
        # running ratio realised / predicted; it scales every VOI, so a model
        # whose probes keep under-delivering stops paying for them.
        self.kappa = 1.0
        self._h_eff = self.cfg.h_days
        self.th_cusum = 0.0
        self.th_cusum_max = 0.0
        self._th_surp = 0.0
        self.th_misspec = False
        self.n_th_alarms = 0
        self.fallback_sched = FALLBACK_SCHED
        self._kap_r = 0.0
        self._kap_p = 0.0
        self._kap_c0 = None
        self._probe_rec = None
        self._last_pred = 0.0
        self._day_pred = 0.0
        self._evpi_day0 = None
        self._evpi_last = 0.0
        self.kappa_trace = []
        if name:
            self.name = name
        self.unread_text = False
        self.last_obs = None
        self.t = 0
        self.sched = FALLBACK_SCHED
        # the daily thermal templates; a site can supply its own set
        self.sched_sp = SCHED_SP
        self.schedules = mech.SCHEDULES
        self.day_sp = SCHED_SP[FALLBACK_SCHED].copy()
        self.cusum = 0.0
        self.misspec = False
        self.log: List[Dict] = []
        self.n_probe_steps = 0
        self.n_asks = 0
        self.n_fallback = 0
        self.n_th_probes = 0
        self._U = None
        self._U_key = None
        self._MU = None
        self._tau = None              # v2: cached free-learning horizons
        self._fl_rng = np.random.default_rng(seed + 7)
        self._reopen = {"net": self.cfg.hazard_net, "th": self.cfg.hazard_th}
        self._probing = False
        self._home = None             # v2: configuration a probe returns to

    # ------------------------------------------------------------ lifecycle
    def offer_text(self):
        """New text (commissioning notes, a maintenance ticket) is available."""
        self.unread_text = True

    def preload_logs(self, logs):
        """Used only by the log-naive variant: treat logged knob settings as
        if they had been interventions. They were not."""
        for o in logs:
            self.nb.update(o)
            self.tb.update(o)
        self.tb._last = None

    # ------------------------------------------------------------ net values
    def _net_tables(self, hour, pdr, P=None, idx=None):
        """Expected reward and violation probability, (N_NET, n_particles).

        The predictive is a two-component mixture: ordinary steps, and the
        tenant-traffic bursts every shared broker sees on a few percent of
        steps (BURST_P, BURST_MULT are part of the hypothesis library, not
        learned). Bursts are what push a near-saturated queue past the
        deadline, so leaving them out would make the safety filter blind.
        `mu`, `s` returned are the ordinary-step log-latency moments, which is
        what the likelihood and the value-of-information nodes use.
        """
        nb = self.nb
        Psub = nb.P if idx is None else {k: v[idx] for k, v in nb.P.items()}
        f = NET_ACTIONS[:, 0][:, None]
        d = NET_ACTIONS[:, 1][:, None]
        o = NET_ACTIONS[:, 2][:, None]
        occ = expected_occ(hour, nb.occ_prior)
        if self.occ_override is not None:
            occ = self.occ_override(hour)
        lat, it_j, aoi = mech.net_outcomes(Psub, f, d, o, hour, occ, pdr)
        lat_b, it_b, _ = mech.net_outcomes(Psub, f, d, o, hour, occ, pdr,
                                           burst=BURST_MULT)
        mu = np.log(lat)
        mu_b = np.log(lat_b)
        s = np.sqrt(Psub["sigma"] ** 2 + nb.noise_extra ** 2)[None, :]
        lh = np.log(mech.W.lat_hard_ms)
        R = np.zeros_like(mu)
        pv = np.zeros_like(mu)
        for m_, e_, pw in ((mu, it_j, 1.0 - BURST_P), (mu_b, it_b, BURST_P)):
            for x, w in zip(GH_X, GH_W):
                L = np.exp(m_ + s * x)
                r, _ = mech.net_reward(L, e_, 1.0 / (2 * f * pdr) + L / 1000.0,
                                       pdr)
                # violation penalty is charged through the probability below
                R += pw * w * (r + mech.W.w_violation * (L > mech.W.lat_hard_ms))
            pv += pw * (1.0 - norm.cdf((lh - m_) / s))
        R -= mech.W.w_violation * pv
        return R, pv, mu, s

    def _context_tables(self, idx):
        """U[a, k, i] and violation probs for the 12 daily contexts."""
        key = (len(idx), int(idx.sum()) if len(idx) else 0,
               tuple(np.round(self.nb.pdr_by_hour[::2], 2)))
        if self._U_key == key:
            return self._U
        Us, Ps, Ms = [], [], []
        for h in CTX_HOURS:
            R, pv, mu, s = self._net_tables(h, self.nb.pdr_by_hour[int(h)],
                                            idx=idx)
            Us.append(R)
            Ps.append(pv)
            Ms.append(mu)
        self._U = (np.stack(Us, 1), np.stack(Ps, 1))      # (A, K, n)
        self._MU = (np.stack(Ms, 1), s[0])                # log-latency means
        self._U_key = key
        return self._U

    def _value_net(self, W, U, PV):
        """Per-day decision value of belief rows W (B, n) over all contexts."""
        B = W.shape[0]
        A, K, n = U.shape
        Q = W @ U.reshape(A * K, n).T               # (B, A*K)
        Q = Q.reshape(B, A, K)
        if self.cfg.use_safety:
            P = (W @ PV.reshape(A * K, n).T).reshape(B, A, K)
            lim = np.maximum(self.cfg.delta, P.min(1, keepdims=True)
                             + self.cfg.delta)
            Q = np.where(P <= lim, Q, -1e9)
        return CTX_STEPS * Q.max(1).sum(1)           # (B,)

    def _greedy_actions(self, W, U, PV):
        """The action _value_net's maximum picks, per belief row and
        context, (B, K)."""
        B = W.shape[0]
        A, K, n = U.shape
        Q = (W @ U.reshape(A * K, n).T).reshape(B, A, K)
        if self.cfg.use_safety:
            P = (W @ PV.reshape(A * K, n).T).reshape(B, A, K)
            lim = np.maximum(self.cfg.delta, P.min(1, keepdims=True)
                             + self.cfg.delta)
            Q = np.where(P <= lim, Q, -1e9)
        return Q.argmax(1)

    def _truth_value(self, acts, Uc, c):
        """Per-day value of action rows acts (B, K) under structure c (B,)."""
        K = Uc.shape[1]
        return CTX_STEPS * Uc[acts, np.arange(K)[None, :], c[:, None]].sum(1)

    def _free_horizon(self, w, live, beta, Mw, U, PV, h_eff, hour, R=None):
        """v2: for each live structure c, the number of days (within h_eff)
        for which greedy operation alone would still be paying the decision
        regret of not knowing that c holds.

        Monte Carlo over the agent's own model: draw truths from each
        structure's particles, run greedy control on the posterior for
        fl_days days at the resolution of the twelve two-hour contexts
        (simulated log-latency observations, the agent's own tempered
        Student-t update and change hazard), and record the regret each day
        against acting with the structure known. rho_c(d), the fraction of
        today's regret still paid on day d, integrates to the horizon; what
        greedy has not resolved by the last simulated day it is assumed never
        to resolve (lock-in keeps the full horizon). Structures holding less
        than fl_min_mass get no credit (tau = h_eff), as in the first
        iteration."""
        cfg = self.cfg
        MU, S = self._MU
        A, K, n = U.shape
        C = len(live)
        R = R or cfg.fl_truths
        sim = np.flatnonzero(beta >= cfg.fl_min_mass)
        tau = np.full(C, float(h_eff))
        if len(sim) == 0:
            return tau
        u = (np.arange(R) + 0.5) / R
        J = np.empty((len(sim), R), int)
        for i, c in enumerate(sim):
            cw = np.cumsum(Mw[c])
            J[i] = np.minimum(np.searchsorted(cw, u * cw[-1]), n - 1)
        J = J.ravel()
        cr = np.repeat(sim, R)
        Uc, PVc = self._aggregate(U, Mw), self._aggregate(PV, Mw)
        a_star = self._greedy_actions(np.eye(C), Uc, PVc)          # (C, K)
        a0 = self._greedy_actions(w[None, :], U, PV)[0]             # (K,)
        UJ = U[:, :, J]                                             # (A, K, nR)
        kk = np.arange(K)
        rr = np.arange(len(J))
        best = UJ[a_star[cr], kk[None, :], rr[:, None]]             # (nR, K)
        reg0 = CTX_STEPS * (best - UJ[a0, kk, :].T).sum(1)          # no learning
        logW = np.repeat(np.log(np.maximum(w, 1e-300))[None, :], len(J), 0)
        hb = 1.0 - (1.0 - cfg.hazard_net) ** CTX_STEPS
        lp = -np.log(self.nb.n)
        k0 = int(hour // 2) % K
        D = cfg.fl_days
        reg = np.zeros((len(J), D))
        wt = CTX_STEPS / cfg.fl_obs
        Ut = np.ascontiguousarray(U.transpose(1, 2, 0))             # (K, n, A)
        PVt = np.ascontiguousarray(PV.transpose(1, 2, 0))
        logS = np.log(S)
        for b in range(K * D):
            k = (k0 + b) % K
            Wb = np.exp(logW - logsumexp(logW, axis=1, keepdims=True))
            Q = Wb @ Ut[k]                                          # (nR, A)
            if cfg.use_safety:
                P = Wb @ PVt[k]
                lim = np.maximum(cfg.delta, P.min(1, keepdims=True) + cfg.delta)
                Q = np.where(P <= lim, Q, -1e9)
            a = Q.argmax(1)
            reg[:, b // K] += CTX_STEPS * (best[:, k] - UJ[a, k, rr])
            y = (MU[a, k, J][:, None]
                 + S[J][:, None] * self._fl_rng.standard_normal((len(J), cfg.fl_obs)))
            # Student-t (nu = 4) log-likelihood of the block, up to a constant
            z = (y[:, :, None] - MU[a, k, :][:, None, :]) / S[None, None, :]
            ll = -(2.5 * np.log1p(0.25 * z * z).sum(1)
                   + cfg.fl_obs * logS[None, :]) * wt
            logW = logW + self.nb.eta * ll
            logW -= logsumexp(logW, axis=1, keepdims=True)
            logW = np.logaddexp(np.log1p(-hb) + logW, np.log(hb) + lp)
        days = np.clip(h_eff - np.arange(D), 0.0, 1.0)
        T = max(0.0, h_eff - D)
        if cfg.fl_tail == "geometric" and D >= 2:
            # beyond the rollout, each structure's remaining regret keeps
            # decaying at its last simulated daily rate: learning that is
            # under way continues, a locked-in share stays locked
            for c in sim:
                m = cr == c
                den = reg0[m].sum()
                if den <= 1e-9:
                    continue
                rho = np.clip(reg[m].sum(0) / den, 0.0, 1.0)
                q = float(np.clip(rho[-1] / rho[-2], 0.0, 1.0)) if rho[-2] > 1e-9 else 0.0
                geo = q * (1.0 - q ** T) / (1.0 - q) if q < 1.0 - 1e-9 else T
                tau[c] = float(rho @ days + rho[-1] * geo)
            return tau
        for c in sim:
            m = cr == c
            den = reg0[m].sum()
            rho = (np.clip(reg[m].sum(0) / den, 0.0, 1.0) if den > 1e-9
                   else np.ones(D))
            tau[c] = float(rho @ days + rho[-1] * T)
        return tau

    def _tau_for(self, w, live, beta, Mw, U, PV, h_eff, hour):
        """Cached free-learning horizons, re-estimated every fl_refresh steps
        or when the structure posterior has moved."""
        c = self._tau
        if (c is not None and self.t - c["t"] < self.cfg.fl_refresh
                and np.array_equal(c["live"], live)
                and np.abs(c["beta"] - beta).sum() < 0.2
                and abs(c["h"] - h_eff) <= 0.2 * max(c["h"], 1e-9)):
            return c["tau"]
        tau = self._free_horizon(w, live, beta, Mw, U, PV, h_eff, hour)
        self._tau = {"t": self.t, "live": live.copy(), "beta": beta.copy(),
                     "h": h_eff, "tau": tau}
        return tau

    def _struct(self, w):
        """Collapse the active particles onto mechanism structures.

        Returns the live structures, their posterior mass beta, and the
        (C, n) membership matrix holding each particle's weight *within* its
        structure. Value-of-information calculations run on beta only: they
        price knowledge of which mechanism holds, while the continuous
        parameters inside a mechanism are marginalised and keep being learned
        from whatever the controller does anyway.
        """
        cid = self._cid
        C = len(self.nb.combos)
        if self.cfg.voi_level == "particle":
            # ablation: every particle its own "structure", so the value of
            # information also prices the continuous parameters
            cid = np.arange(len(w))
            C = len(w)
        beta = np.bincount(cid, weights=w, minlength=C)
        live = np.flatnonzero(beta > 0)
        pos = np.full(C, -1)
        pos[live] = np.arange(len(live))
        M = np.zeros((len(live), len(w)))
        M[pos[cid], np.arange(len(w))] = w / beta[cid]
        return live, beta[live], M

    @staticmethod
    def _aggregate(U, M):
        """Within-structure posterior-mean tables, (A, K, C)."""
        return np.einsum("akn,cn->akc", U, M)

    def _active(self, eps=1e-6):
        w = self.nb.w
        idx = np.flatnonzero(w > eps * w.max())
        return idx, w[idx] / w[idx].sum()

    # ------------------------------------------------------------ acting
    def act(self, obs):
        out = self._act(obs)
        self._probing = bool(out[3].get("probe", False))
        return out

    def _act(self, obs):
        cfg = self.cfg
        k = self.t % STEPS_PER_DAY
        info = {"probe": False, "ask": False, "fallback": False,
                "th_probe": False}
        if k == 0:
            if self.t > 0 and self.cfg.kappa_mode == "daily":
                self._daily_audit()
            self._plan_day(obs)
        sp = float(self.day_sp[k])

        ask = False
        if cfg.use_ask and self.unread_text and self.expert is not None:
            voi_ask = self._voi_ask()
            info["voi_ask"] = voi_ask
            if voi_ask > cfg.ask_price or cfg.force_ask:
                ask = True
            else:
                self.unread_text = False         # judged not worth reading
        info["ask"] = ask

        hour = k / 12.0
        pdr = obs["pdr"] if obs is not None else 0.93
        prev = int(obs["net_idx"]) if obs is not None else None
        if self.misspec:
            self.n_fallback += 1
            info["fallback"] = True
            if cfg.probe_return:
                self._home = None
            return FALLBACK_NET, sp, ask, info

        idx, w = self._active()
        self._cid = self.nb.combo_id[idx]
        R, pv, mu, s = self._net_tables(hour, pdr, idx=idx)
        base = R @ w
        s_in = mech.switch_cost_vec(prev)
        P = pv @ w
        ok = self._admissible(P)
        if not ok.any():
            self.n_fallback += 1
            info["fallback"] = True
            self._commit = None
            if cfg.probe_return:
                self._home = None
            return FALLBACK_NET, sp, ask, info
        # Switching-aware greedy, shared by every belief-based controller:
        # a reconfiguration is amortised over the dwell time it buys, so
        # the controller does not dither when the best setting flickers.
        s_ref = s_in
        if cfg.probe_return and self._home is not None:
            # v2: a probe is an excursion -- it was priced with the switch
            # back -- from the path greedy control would have followed. That
            # path is kept as a shadow configuration, moved by the greedy
            # rule itself, and the gateway returns to it when the probe ends.
            s_ref = mech.switch_cost_vec(self._home)
        Qm = np.where(ok, base - s_ref / cfg.dwell, -np.inf)
        a_greedy = int(np.argmax(Qm))
        # a probe is held for the duration it was valued at, while admissible
        if self._commit is not None:
            a_c, until = self._commit
            if self.t < until and ok[a_c]:
                if cfg.probe_return:
                    self._home = a_greedy
                self.n_probe_steps += int(a_c != a_greedy)
                info["probe"] = bool(a_c != a_greedy)
                info["P_viol"] = float(P[a_c])
                return a_c, sp, ask, info
            self._commit = None
        home, self._home = a_greedy, None
        a, m = a_greedy, 1
        if cfg.objective == "identify":
            a = self._identify_action(idx, w, mu, s, ok, a_greedy)
        elif cfg.use_probe:
            U, PV = self._context_tables(idx)
            live, beta, Mw = self._struct(w)
            Uc, PVc = self._aggregate(U, Mw), self._aggregate(PV, Mw)
            evpi = self._evpi(beta, Uc, PVc)
            info["evpi"] = evpi
            self._evpi_last = evpi
            if self._evpi_day0 is None:
                self._evpi_day0 = evpi
            # Exact screen: no observation is worth more than perfect
            # information, so a deviation from greedy can only pay if
            # H*EVPI exceeds what it costs (lost reward + reconfiguration in
            # and back out). When no action passes, skip the computation.
            self._h_eff = self._durability(beta, cfg.hazard_net)
            vmax = self._h_eff * evpi
            s_out = mech.switch_cost_vec(a_greedy)

            def passing(vmax):
                num = vmax - s_in - s_out
                ub = base + np.where(num > 0, num, num / max(cfg.kg_m))
                pas = ok & (ub > Qm[a_greedy] + 1e-9)
                pas[a_greedy] = False
                return pas
            pas = passing(vmax)
            tau = None
            if cfg.free_learning and vmax > cfg.screen_eps and pas.any():
                # v2: the same screen, net of free learning. No information
                # about structure c is worth more than the regret of not
                # knowing c, paid for as long as greedy would keep paying it.
                tau = self._tau_for(w, live, beta, Mw, U, PV, self._h_eff,
                                    hour)
                a0 = self._greedy_actions(beta[None, :], Uc, PVc)[0]
                kk = np.arange(Uc.shape[1])
                r_c = CTX_STEPS * (Uc.max(0) - Uc[a0, kk, :]).sum(0)
                vmax = float(beta @ (r_c * tau))
                info["vmax_fl"] = vmax
                pas = passing(vmax)
            info["screened"] = not pas.any()
            if vmax > cfg.screen_eps and pas.any():
                a, m = self._kg_action(w, beta, Mw, Uc, PVc, mu, s, base,
                                       s_in, s_out, np.flatnonzero(pas),
                                       a_greedy, live=live, tau=tau)
                if a != a_greedy:
                    self._day_pred += self._last_pred
                if a != a_greedy and self.cfg.kappa_mode == "probe":
                    # A probe's information is judged by what is left of it
                    # when the next probe starts (or a while after it ended),
                    # not the moment it ends: knowledge that evaporates within
                    # hours is not what its value was computed for.
                    if self._probe_rec is not None:
                        self._settle_probe()
                    self._probe_rec = {"t_end": self.t + m, "evpi0": evpi,
                                       "pred": self._last_pred}
        if a != a_greedy:
            self.n_probe_steps += 1
            info["probe"] = True
            if cfg.probe_return:
                self._home = home
            if m > 1:
                self._commit = (a, self.t + m)
        info["P_viol"] = float(P[a])
        return a, sp, ask, info

    def _durability(self, beta, hazard):
        """How long (in days) resolving the current structural uncertainty
        stays resolved. Under the change-point prior the posterior mass off
        the leading structure, m, is replenished at the hazard rate, so ruling
        it out lasts about m / hazard steps; a genuine ambiguity keeps the full
        horizon, residual hazard mass is worth hours, not weeks."""
        if not self.cfg.durability:
            return self.cfg.h_days
        m = float(1.0 - np.max(beta))
        if self.cfg.reopen_rate:
            # v2: the measured re-opening rate, where it exceeds the prior's
            side = "net" if hazard == self.cfg.hazard_net else "th"
            hazard = max(hazard, self._reopen[side])
        return min(self.cfg.h_days, m / max(hazard, 1e-12) / STEPS_PER_DAY)

    @staticmethod
    def _offmass(b):
        """Posterior mass off the leading structure of a particle belief."""
        return 1.0 - float(np.bincount(b.combo_id, weights=b.w).max())

    def _admissible(self, P):
        """Relative chance constraint. An action is admissible if its
        posterior probability of missing the deadline is at most delta above
        that of the safest available action (or below delta outright). Some
        deadline misses -- tenant bursts on a saturated shared broker -- no
        action can prevent; what the gateway must not do is add risk the
        safest setting would not have taken (cf. conservative bandits)."""
        if not self.cfg.use_safety:
            return np.ones(len(P), bool)
        return P <= max(self.cfg.delta, float(P.min()) + self.cfg.delta)

    def _pos_live(self, live):
        if self.cfg.voi_level == "particle":
            return np.arange(len(self._cid))
        pos = np.full(len(self.nb.combos), -1)
        pos[live] = np.arange(len(live))
        return pos[self._cid]

    def _evpi(self, w, U, PV):
        best_each = CTX_STEPS * U.max(0).sum(0)            # (n,) value if known
        return float(w @ best_each - self._value_net(w[None, :], U, PV)[0])

    def _outcome_nodes(self, w):
        """Quadrature over 'what would I observe': every active particle (or a
        systematic resample when there are many) times Gauss-Hermite nodes on
        the observation noise. Deterministic, so the value of information is
        not inflated by Monte-Carlo noise and by the max over candidates."""
        n = len(w)
        Q = self.cfg.quad_particles
        if n <= Q:
            return np.arange(n), w
        u = (np.arange(Q) + 0.5) / Q
        J = np.minimum(np.searchsorted(np.cumsum(w), u), n - 1)
        return J, np.full(Q, 1.0 / Q)

    def _kg_action(self, w, beta, Mw, Uc, PVc, mu, s, base, s_in, s_out,
                   cand, a_greedy, live=None, tau=None):
        """KG(*) over mechanism structures, with switching costs.

        For each candidate action a and probe length m: quadrature over what
        would be observed (every active particle, Gauss-Hermite nodes on the
        noise), the structure-level likelihood of that observation (a mixture
        over each structure's particles), the updated structure posterior, and
        the resulting per-day decision value. The score per step is
            base[a] + H * (E[V(beta')] - V(beta)) / m - reconfiguration / m.
        With free-learning horizons tau (v2), the gain is split by the
        structure that generated the observation (by the tower property,
        E[V(beta')] - V(beta) is the mean over outcome nodes of the true
        structure's value of the updated decision minus that of the current
        one), and each part is counted for tau[c] days instead of H. A
        structure under which the observation would on average worsen the
        decision is not counted against it: the controller would then
        re-learn from its own operation, and the rollouts value only what
        greedy operation alone would take to learn.
        """
        cfg = self.cfg
        if len(cand) > cfg.n_cand:
            cand = cand[np.argsort(-(base - s_in)[cand])[: cfg.n_cand]]
        cand = np.concatenate([[a_greedy], cand]).astype(int)
        V0 = self._value_net(beta[None, :], Uc, PVc)[0]
        J, pj = self._outcome_nodes(w)
        x, wx = GH5_X, GH5_W
        ms = np.asarray(cfg.kg_m, float)
        nC, nM, nJ, nX = len(cand), len(ms), len(J), len(x)
        sc = s[0][None, :] / np.sqrt(ms)[:, None]                    # (M, n)
        sc_rows = np.repeat(sc, nJ * nX, axis=0)                     # (MJX, n)
        lb0 = np.log(beta)[None, :]
        EV = np.empty((nC, nM))
        if tau is not None:
            cJ = self._pos_live(live)[J]                     # truth structure
            cR = np.tile(np.repeat(cJ, nX), nM)              # rows (m, j, x)
            onehot = np.eye(len(live))[cJ]                   # (J, C)
            a0 = self._greedy_actions(beta[None, :], Uc, PVc)
            V0c = self._truth_value(np.repeat(a0, len(cR), 0), Uc, cR)
            EVt = np.empty((nC, nM))
        for ci, a in enumerate(cand):
            y = (mu[a][None, J, None] + sc[:, J, None] * x[None, None, :])
            ll = student_t_logpdf(y.reshape(-1)[:, None] - mu[a][None, :],
                                  sc_rows)                           # (R, n)
            mx = ll.max(1, keepdims=True)
            Lc = np.log(np.exp(ll - mx) @ Mw.T + 1e-300) + mx         # (R, C)
            lb = lb0 + self.nb.eta * Lc
            B2 = np.exp(lb - logsumexp(lb, axis=1, keepdims=True))
            Vb = self._value_net(B2, Uc, PVc).reshape(nM, nJ, nX)
            EV[ci] = np.einsum("mjx,j,x->m", Vb, pj, wx)
            if tau is not None:
                acts = self._greedy_actions(B2, Uc, PVc)
                gt = (self._truth_value(acts, Uc, cR) - V0c).reshape(nM, nJ, nX)
                # beta_c * g_c for every structure c: the expected gain when
                # c holds, weighted by its posterior mass
                Gc = np.einsum("mjx,j,x,jc->mc", gt, pj, wx, onehot)
                EVt[ci] = np.maximum(Gc, 0.0) @ tau
        gain = np.maximum(EV - V0, 0.0)                              # (C, M)
        if tau is None:
            voi = self._h_eff * self.kappa * gain
        else:
            voi = self.kappa * EVt
            self._last_tau = {"tau": tau.tolist(), "beta": beta.tolist(),
                              "voi": float(voi.max()),
                              "voi_v1": float(self._h_eff * self.kappa
                                              * gain.max())}
        so = s_out[cand].copy()
        so[0] = 0.0
        # a probe pays its reconfiguration in and back out within its m steps;
        # staying with the greedy action is charged exactly as the greedy rule
        # charges it (amortised over the dwell time), whatever m is
        pen = (s_in[cand] + so)[:, None] / ms[None, :]
        pen[0, :] = s_in[a_greedy] / cfg.dwell
        score = base[cand][:, None] + voi / ms[None, :] - pen
        c, mi = np.unravel_index(int(np.argmax(score)), score.shape)
        self._last_pred = float(gain[c, mi])
        return int(cand[c]), int(ms[mi])

    def _identify_action(self, idx, w, mu, s, ok, a_greedy):
        """Ablation: probe for the *mechanism*, not for the decision.

        While no mechanism combination holds `identify_target` of the mass,
        take the admissible action with the largest mutual information about
        the combination; then exploit.
        """
        cid = self.nb.combo_id[idx]
        post = np.bincount(cid, weights=w, minlength=len(self.nb.combos))
        if post.max() >= self.cfg.identify_target:
            return a_greedy
        J, pj = self._outcome_nodes(w)
        x, wx = GH5_X, GH5_W
        H0 = -np.sum(post * np.log(post + 1e-12))
        best, best_ig = a_greedy, -1.0
        for a in np.flatnonzero(ok):
            y = (mu[a, J][:, None] + s[0, J][:, None] * x[None, :]).ravel()
            ll = student_t_logpdf(y[:, None] - mu[a][None, :], s[0][None, :])
            lw = np.log(w)[None, :] + self.nb.eta * ll
            W2 = np.exp(lw - logsumexp(lw, axis=1, keepdims=True))
            pc = np.stack([np.bincount(cid, weights=r, minlength=len(post))
                           for r in W2])
            Hc = -np.sum(pc * np.log(pc + 1e-12), 1).reshape(len(J), len(x))
            H1 = float(pj @ (Hc @ wx))
            if H0 - H1 > best_ig:
                best, best_ig = int(a), H0 - H1
        return best

    # ------------------------------------------------------------ asking
    def _voi_ask(self):
        """Value of reading the unread text with the expert, both subsystems."""
        ex = self.expert
        cfg = self.cfg
        idx, w = self._active()
        self._cid = self.nb.combo_id[idx]
        U, PV = self._context_tables(idx)
        live, beta, Mw = self._struct(w)
        Uc, PVc = self._aggregate(U, Mw), self._aggregate(PV, Mw)
        V0n = self._value_net(beta[None, :], Uc, PVc)[0]
        tw = self.tb.w
        Uth = self._th_values()
        V0t = float(np.max(tw @ Uth))
        M = cfg.mc
        jn = self.rng.choice(len(w), size=M, p=w)
        jt = self.rng.choice(len(tw), size=M, p=tw)
        fac_net = {F: self.nb.factor_values(F)[idx] for F in "ABC"}
        fac_th = self.tb.factor_values()
        Wn = np.empty((M, len(w)))
        Wt = np.empty((M, len(tw)))
        for m in range(M):
            truth = {F: int(fac_net[F][jn[m]]) for F in "ABC"}
            truth["D"] = int(fac_th[jt[m]])
            z = ex.sample_output(truth, self.rng)
            lln = sum(ex.loglik(F, z[F], fac_net[F]) for F in "ABC")
            llt = ex.loglik("D", z["D"], fac_th)
            a = np.log(w) + lln
            Wn[m] = np.exp(a - logsumexp(a))
            b = np.log(tw + 1e-300) + llt
            Wt[m] = np.exp(b - logsumexp(b))
        pos = self._pos_live(live)
        Bn = np.stack([np.bincount(pos, weights=r, minlength=len(live))
                       for r in Wn])
        dn = self._value_net(Bn, Uc, PVc).mean() - V0n
        dt = float(np.mean(np.max(Wt @ Uth, axis=1))) - V0t
        h_n = self._durability(beta, cfg.hazard_net)
        bt = np.bincount(self.tb.combo_id, weights=tw, minlength=2)
        h_t = self._durability(bt, cfg.hazard_th)
        self._ask_parts = {"net_v1": float(h_n * dn), "th_v1": float(h_t * dt)}
        if not cfg.free_learning:
            return h_n * dn + h_t * dt
        return self._voi_ask_v2(w, live, beta, Mw, U, PV, Uc, PVc, h_n, h_t,
                                fac_net, fac_th, Uth)

    def _voi_ask_v2(self, w, live, beta, Mw, U, PV, Uc, PVc, h_n, h_t,
                    fac_net, fac_th, Uth):
        """v2: value of a reading, net of free learning.

        Stratified by structure: for every live structure c, fl_ask_mc
        readings simulated from the expert's error bank as if c held, the
        updated posterior's decision, and its value *under c* against the
        current decision; that gain counts for tau[c] days, the time greedy
        operation would take to find c out by itself (rollouts as in
        _free_horizon, more of them, since this is valued once). Likewise
        for the thermal mass class.
        """
        cfg = self.cfg
        ex = self.expert
        rng = self._fl_rng
        hour = (self.t % STEPS_PER_DAY) / 12.0
        tau = self._free_horizon(w, live, beta, Mw, U, PV, h_n, hour,
                                 R=cfg.fl_ask_truths)
        pos = self._pos_live(live)
        C, M = len(live), cfg.fl_ask_mc
        a0 = np.repeat(self._greedy_actions(beta[None, :], Uc, PVc), M, 0)
        lw = np.log(w)
        g = np.zeros(C)
        for c in range(C):
            A_, B_, C_ = self.nb.combos[live[c]]
            truth = {"A": A_, "B": B_, "C": C_, "D": 0}
            Bn = np.empty((M, C))
            for m in range(M):
                z = ex.sample_output(truth, rng)
                a = lw + sum(ex.loglik(F, z[F], fac_net[F]) for F in "ABC")
                Bn[m] = np.bincount(pos, weights=np.exp(a - logsumexp(a)),
                                    minlength=C)
            cc = np.full(M, c)
            g[c] = float(np.mean(
                self._truth_value(self._greedy_actions(Bn, Uc, PVc), Uc, cc)
                - self._truth_value(a0, Uc, cc)))
        vn = float(beta @ (np.maximum(g, 0.0) * tau))
        tw = self.tb.w
        tau_t, Ucl, cls_pos = self._free_horizon_th(h_t, at_start=True,
                                                    R=cfg.fl_ask_truths)
        bt = np.bincount(self.tb.combo_id, weights=tw, minlength=2)
        s0 = int(np.argmax(tw @ Uth))
        ltw = np.log(tw + 1e-300)
        vt = 0.0
        for d in np.flatnonzero(cls_pos >= 0):
            s1 = np.empty(M, int)
            for m in range(M):
                z = ex.sample_output({"A": 0, "B": 0, "C": 0, "D": int(d)}, rng)
                b = ltw + ex.loglik("D", z["D"], fac_th)
                s1[m] = int(np.argmax(np.exp(b - logsumexp(b)) @ Uth))
            r = cls_pos[d]
            vt += bt[d] * tau_t[r] * max(0.0, float(np.mean(Ucl[r, s1]
                                                            - Ucl[r, s0])))
        self._ask_parts.update({"net": vn, "th": vt, "tau": tau.tolist(),
                                "gain": g.tolist(), "beta": beta.tolist(),
                                "tau_th": tau_t.tolist()})
        return max(0.0, vn + vt)

    def receive_expert(self, z: Dict[str, float], is_ticket: bool = False):
        """Fold one expert reading into both beliefs (calibrated likelihood).

        A maintenance ticket is first of all evidence that *something changed*:
        before its content is weighed, part of the prior is mixed back in, so
        a belief made certain by weeks of data can actually move.
        """
        ex = self.expert
        if is_ticket:
            rho = self.cfg.ticket_reset
            for b in (self.nb, self.tb):
                b.logw = np.logaddexp(np.log1p(-rho) + b.logw,
                                      np.log(rho) + b.log_prior)
                b.logw -= logsumexp(b.logw)
        ll = sum(ex.loglik(F, z[F], self.nb.factor_values(F)) for F in "ABC")
        self.nb.logw = self.nb.logw + ll
        self.nb.logw -= logsumexp(self.nb.logw)
        llt = ex.loglik("D", z["D"], self.tb.factor_values())
        self.tb.logw = self.tb.logw + llt
        self.tb.logw -= logsumexp(self.tb.logw)
        self.unread_text = False
        self.n_asks += 1
        self._U_key = None
        self._th_cache = None

    # ------------------------------------------------------------ thermal
    def _th_values(self):
        c = getattr(self, "_th_cache", None)
        if c is not None:
            return c
        tb = self.tb
        t_air = self.last_obs["t_air"] if self.last_obs else 26.0
        fc = getattr(self, "forecast", None)
        if fc is None:
            fc = np.full(STEPS_PER_DAY, 27.0)
        if tb.t_mass is None:
            tb.t_mass = np.full(tb.n, t_air)
        occ = getattr(self, "occ_day_ref", None)
        self._th_cache = tb.simulate_day(self.sched_sp, fc, t_air,
                                         occ=None if occ is None else occ()).T
        return self._th_cache

    def _free_horizon_th(self, h_th, at_start, R=None):
        """v2, thermal side: for each thermal class (heavy, light), the days
        of schedule regret greedy operation would still pay within h_th.

        One simulated day of the Bayes schedule under the agent's own model:
        truths drawn from each class, their air temperature simulated and
        measured, and the belief's own one-step update along it; then the
        schedule the updated belief would choose. rho_c is the fraction of
        today's regret still paid tomorrow, held for the rest of the horizon.
        Information that arrives before today's schedule is fixed
        (at_start: a reading at midnight) also saves today's regret; a night
        step test only informs tomorrow's schedule.
        Returns (tau per live class, class-mean values (C, S), class->row)."""
        cfg = self.cfg
        tb = self.tb
        w = tb.w
        Uth = self._th_values()                                   # (N, S)
        N = Uth.shape[0]
        cls = tb.combo_id
        beta = np.bincount(cls, weights=w, minlength=2)
        live = np.flatnonzero(beta > 1e-12)
        cls_pos = np.full(2, -1)
        cls_pos[live] = np.arange(len(live))
        R = R or cfg.fl_truths
        u = (np.arange(R) + 0.5) / R
        Mw = np.zeros((len(live), N))
        J = []
        for r_, c in enumerate(live):
            Mw[r_, cls == c] = w[cls == c] / beta[c]
            cw = np.cumsum(Mw[r_])
            J.append(np.minimum(np.searchsorted(cw, u * cw[-1]), N - 1))
        J = np.concatenate(J)
        cr = np.repeat(np.arange(len(live)), R)
        Ucl = Mw @ Uth                                            # (C, S)
        s_star = Ucl.argmax(1)
        s0 = int(np.argmax(w @ Uth))
        sp = self.sched_sp[s0]
        fc = getattr(self, "forecast", None)
        if fc is None:
            fc = np.full(STEPS_PER_DAY, 27.0)
        t_air = self.last_obs["t_air"] if self.last_obs else 26.0
        if tb.t_mass is None:
            tb.t_mass = np.full(tb.n, t_air)
        P = tb.P
        PJ = {k: v[J] for k, v in P.items()}
        ta = np.full(len(J), t_air)
        tmJ = tb.t_mass[J].copy()
        tm = np.repeat(tb.t_mass[None, :], len(J), 0)             # (nR, N)
        y_prev = ta.copy()
        ll = np.zeros((len(J), N))
        for k in range(STEPS_PER_DAY):
            h = k / 12.0
            occ = expected_occ(h, tb.occ_prior)
            hv = mech.hvac_power(ta, sp[k], PJ)
            ta, tmJ = mech.thermal_step(PJ, ta, tmJ, fc[k], hv, occ, h)
            occ_sd = mech.office_gate(h) * 3.0
            scJ = np.sqrt(tb.sigma ** 2 + (PJ["occ_gain"] * occ_sd * mech.DT
                                           / PJ["C_air"]) ** 2)
            y = ta + scJ * self._fl_rng.standard_normal(len(J))
            pred, tm = mech.thermal_step(P, y_prev[:, None], tm, fc[k],
                                         hv[:, None], occ, h)
            sc = np.sqrt(tb.sigma ** 2 + (P["occ_gain"] * occ_sd * mech.DT
                                          / P["C_air"]) ** 2)
            ll += student_t_logpdf(y[:, None] - pred, sc[None, :])
            y_prev = y
        lw = np.log(w + 1e-300)[None, :] + tb.eta * ll
        W1 = np.exp(lw - logsumexp(lw, axis=1, keepdims=True))
        s1 = np.argmax(W1 @ Uth, axis=1)
        UJ = Uth[J]
        rr = np.arange(len(J))
        bestJ = UJ[rr, s_star[cr]]
        reg0 = bestJ - UJ[:, s0]
        reg1 = bestJ - UJ[rr, s1]
        tau = np.empty(len(live))
        for c in range(len(live)):
            m = cr == c
            den = reg0[m].sum()
            rho = float(np.clip(reg1[m].sum() / den, 0.0, 1.0)) if den > 1e-9 else 1.0
            tau[c] = ((min(1.0, h_th) if at_start else 0.0)
                      + rho * max(0.0, h_th - 1.0))
        return tau, Ucl, cls_pos

    def set_forecast(self, forecast):
        self.forecast = np.asarray(forecast, float)
        self._th_cache = None

    def _thermal_watch(self, obs):
        """Excess surprise of the thermal belief's one-step prediction of
        the air temperature, in a clipped CUSUM: the thermal counterpart of
        the latency detector. Call before the belief update."""
        cfg = self.cfg
        tb = self.tb
        if not cfg.thermal_detect or tb._last is None or tb.t_mass is None:
            return
        prev = tb._last
        ta_pred, _ = tb.predict_next(prev["t_air"], obs["t_out"],
                                     obs["hvac_kw"], obs["hour"])
        occ_sd = mech.office_gate(obs["hour"]) * 3.0
        sc = np.sqrt(tb.sigma ** 2 + (tb.P["occ_gain"] * occ_sd * mech.DT
                                      / tb.P["C_air"]) ** 2)
        w = tb.w
        lp = logsumexp(np.log(w + 1e-300)
                       + student_t_logpdf(obs["t_air"] - ta_pred, sc))
        ex = min(-lp - float(w @ (np.log(sc) + T4_ENTROPY)), cfg.th_clip)
        self.th_cusum = max(0.0, self.th_cusum + ex - cfg.th_drift)
        self.th_cusum_max = max(self.th_cusum_max, self.th_cusum)
        self._th_surp = 0.99 * self._th_surp + 0.01 * ex
        if cfg.thermal_shadow:
            return
        if not self.th_misspec and self.th_cusum > cfg.th_h:
            self.th_misspec = True
            self.n_th_alarms += 1
        elif self.th_misspec and self._th_surp < cfg.th_exit:
            self.th_misspec = False
            self.th_cusum = 0.0

    def _plan_day(self, obs):
        cfg = self.cfg
        tb = self.tb
        if self.th_misspec:
            # the grey-box library does not explain this zone: run the
            # verified schedule and buy no thermal information
            self.sched = self.fallback_sched
            self.day_sp = self.sched_sp[self.sched].copy()
            return
        Uth = self._th_values()                       # (N, S)
        w = tb.w
        Q = w @ Uth
        self.sched = int(np.argmax(Q))
        self.day_sp = self.sched_sp[self.sched].copy()
        if not (cfg.use_thermal_probe and cfg.use_probe):
            return
        # Night step test: 20 C from 00:00 to 03:00 against the plan without
        # it. Every method can also choose that profile as a control option
        # (the "early" schedule variants); if the Bayes schedule already
        # includes it, the test is free and nothing needs deciding.
        if self.schedules[self.sched].get("early"):
            return
        t_air = obs["t_air"] if obs else 26.0
        fc = self.forecast
        sp_probe = self.day_sp.copy()
        sp_probe[:36] = 20.0
        both = np.stack([self.day_sp, sp_probe])
        Up = tb.simulate_day(both, fc, t_air)          # (2, N)
        cost = float(w @ (Up[0] - Up[1]))              # >0 means probe costs
        # predicted air-temperature response over the first four hours
        traj = self._trajectories(sp_probe[:48], fc[:48], t_air)   # (N, 48)
        # value it on the *structure* (heavy vs light mass), with the zone's
        # continuous parameters marginalised inside each class
        n = len(w)
        cls = tb.combo_id
        beta = np.bincount(cls, weights=w, minlength=2)
        live = np.flatnonzero(beta > 1e-12)
        Mw = np.zeros((len(live), n))
        for r, c in enumerate(live):
            Mw[r, cls == c] = w[cls == c] / beta[c]
        Uc = Mw @ Uth                                  # (C, S)
        M = cfg.mc
        j = self.rng.choice(n, size=M, p=w)
        noise = self.rng.normal(0, tb.sigma, (M, 48))
        Y = traj[j] + noise
        ll = np.stack([student_t_logpdf(Y[m][None, :] - traj, tb.sigma).sum(1)
                       for m in range(M)])            # (M, N)
        mx = ll.max(1, keepdims=True)
        Lc = np.log(np.exp(ll - mx) @ Mw.T + 1e-300) + mx            # (M, C)
        lb = np.log(beta[live])[None, :] + tb.eta * Lc
        B2 = np.exp(lb - logsumexp(lb, axis=1, keepdims=True))
        h_th = self._durability(beta[live], cfg.hazard_th)
        if cfg.free_learning:
            # v2: the step test informs tomorrow's schedule, and so does
            # today's ordinary operation; count each class's gain only for
            # the regret greedy scheduling would still leave from tomorrow on
            tau_t, _, cls_pos = self._free_horizon_th(h_th, at_start=False)
            ct = cls_pos[cls[j]]                     # truth class of each draw
            s_b = np.argmax(B2 @ Uc, axis=1)
            s_0 = int(np.argmax(beta[live] @ Uc))
            g = Uc[ct, s_b] - Uc[ct, s_0]
            G = np.bincount(ct, weights=g, minlength=len(live)) / len(g)
            voi = float(np.maximum(G, 0.0) @ tau_t)
        else:
            voi = h_th * (float(np.mean(np.max(B2 @ Uc, axis=1)))
                          - float(np.max(beta[live] @ Uc)))
        if voi > cost:
            self.day_sp = sp_probe
            self.n_th_probes += 1

    def _trajectories(self, sp, fc, t_air):
        tb = self.tb
        P = tb.P
        ta = np.full(tb.n, t_air)
        tm = tb.t_mass.copy()
        out = np.empty((tb.n, len(sp)))
        for k in range(len(sp)):
            h = k / 12.0
            hv = mech.hvac_power(ta, sp[k], P)
            ta, tm = mech.thermal_step(P, ta, tm, fc[k], hv,
                                       expected_occ(h, tb.occ_prior), h)
            out[:, k] = ta
        return out

    # ------------------------------------------------------------ learning
    def _daily_audit(self):
        """Durable self-calibration. Under a correct model the drop in EVPI
        over a day is at least the predicted value of the probes bought that
        day (ordinary operation only adds information). Where structural
        ambiguity keeps regenerating -- the model says every probe resolves
        it, the data keep re-opening it -- EVPI does not fall from day to day,
        and kappa, which scales every value of information, falls instead."""
        pred = self._day_pred
        if self._evpi_day0 is not None and pred > 0:
            realised = self._evpi_day0 - self._evpi_last
            if self._kap_c0 is None:
                self._kap_c0 = 3.0 * max(pred, 1e-6)
            self._kap_r += realised
            self._kap_p += pred
            if self.cfg.self_calibrate:
                c0 = self._kap_c0
                self.kappa = float(np.clip((c0 + self._kap_r) / (c0 + self._kap_p),
                                           0.05, 1.0))
            self.kappa_trace.append((self.t, pred, realised, self.kappa))
        self._day_pred = 0.0
        self._evpi_day0 = self._evpi_last

    def _settle_probe(self):
        """Close a probe record: compare realised with predicted EVPI drop."""
        rec = self._probe_rec
        self._probe_rec = None
        idx, w = self._active()
        self._cid = self.nb.combo_id[idx]
        U, PV = self._context_tables(idx)
        live, beta, Mw = self._struct(w)
        evpi1 = self._evpi(beta, self._aggregate(U, Mw), self._aggregate(PV, Mw))
        realised = rec["evpi0"] - evpi1
        if self._kap_c0 is None:
            self._kap_c0 = 3.0 * max(rec["pred"], 1e-6)
        self._kap_r += realised
        self._kap_p += rec["pred"]
        c0 = self._kap_c0
        if self.cfg.self_calibrate:
            self.kappa = float(np.clip((c0 + self._kap_r) / (c0 + self._kap_p),
                                       0.05, 1.0))
        self.kappa_trace.append((self.t, rec["pred"], realised, self.kappa))

    def observe(self, obs, busy=0.0):
        cfg = self.cfg
        if cfg.misspec_detect:
            idx, w = self._active()
            mu, _ = self.nb.predict(obs["net_idx"], obs["hour"], obs["pdr"],
                                    busy, P={k: v[idx] for k, v in
                                             self.nb.P.items()})
            s = np.sqrt(self.nb.P["sigma"][idx] ** 2 + self.nb.noise_extra ** 2)
            lp = logsumexp(np.log(w) + student_t_logpdf(
                np.log(obs["latency_ms"]) - mu, s))
            # Excess surprise: the observation's surprise minus what the
            # model's own predictive expects it to be (its entropy), clipped
            # so that a single tenant burst cannot carry the statistic. Zero
            # mean when the library explains the plant, whatever the noise.
            H = float(w @ (np.log(s) + T4_ENTROPY))
            ex = min(-lp - H, cfg.surprise_clip)
            self.cusum = max(0.0, self.cusum + ex - cfg.cusum_drift)
            self.cusum_max = max(getattr(self, "cusum_max", 0.0), self.cusum)
            self._surp = 0.97 * getattr(self, "_surp", 0.0) + 0.03 * ex
            if not self.misspec and self.cusum > cfg.cusum_h:
                self.misspec = True
                self.n_misspec_alarms = getattr(self, "n_misspec_alarms", 0) + 1
            elif self.misspec and self._surp < cfg.misspec_exit:
                # the library explains the data again (e.g. after a structural
                # change the posterior has caught up with): leave fallback
                self.misspec = False
                self.cusum = 0.0
        if cfg.reopen_rate:
            m_net, m_th = self._offmass(self.nb), self._offmass(self.tb)
        self.nb.update(obs, busy)
        self._thermal_watch(obs)
        self.tb.update(obs)
        if cfg.reopen_rate and not self._probing:
            # drift of the off-leading mass under ordinary operation
            a = 1.0 / cfg.reopen_tau
            for side, b, m0 in (("net", self.nb, m_net), ("th", self.tb, m_th)):
                self._reopen[side] += a * (self._offmass(b) - m0
                                           - self._reopen[side])
        self.last_obs = obs
        self.t += 1
        if self._probe_rec is not None and self.cfg.kappa_mode == "probe" and                 self.t >= self._probe_rec["t_end"] + self.cfg.kappa_delay:
            self._settle_probe()
        if self.t % 24 == 0:
            self._U_key = None

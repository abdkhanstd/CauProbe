"""CauSLM-Edge: the full on-gateway agent.

Per control step the agent:

  1. appends the new KPI row to a sliding window;
  2. every `graph_period` steps refits the PCMCI+ skeleton on that window
     (this is the cheap statistical stage, and it runs whether or not the
     language model is invoked);
  3. asks the meta-controller whether the expected value of a fresh SLM
     invocation beats its measured price;
  4. if so, serialises the skeleton into the prompt, invokes the model, parses
     the JSON, deletes the edges it flagged as spurious, and folds its
     predicted ATE into the bandit as a prior;
  5. selects actuator settings with the factored causal bandit, whose credit
     assignment comes from the current (possibly repaired) graph;
  6. scores the realised H-step advantage of the last invocation and hands it
     back to the meta-controller as a training target.

Setting `policy=None` and `gate="never"` reduces this to the pure
PCMCI+ + causal-bandit baseline, which is how the ablations are run: every
comparison in the paper goes through this one code path.
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..causal import discover, preprocess_for_discovery
from ..config import VARS
from ..env import ACTION_LEVELS, IoTGatewayEnv
from ..slm.prompts import build_prompt
from ..slm.schema import CausalAdvice
from ..slm.teacher import CANDIDATES
from .bandit import KNOB_ARMS, FactoredCausalBandit
from .meta import META_GATES, MetaController, NeverQuery

ACTION_KEYS = list(ACTION_LEVELS)


@dataclass
class AgentConfig:
    window: int = 288
    graph_period: int = 48
    discovery: str = "pcmci+"        # or granger / dynotears / none
    tau_max: int = 3
    pc_alpha: float = 0.05
    repair_with_slm: bool = True
    repair_min_flags: int = 2        # invocations that must independently flag
                                     # an edge before it is deleted
    use_graph_credit: bool = True
    evoi_horizon: int = 12
    plan_ttl: int = 24               # how long one piece of advice stays in force
    use_plan_override: bool = True   # act on the advice, not only prior it
    n_candidates: int = 8
    slm_batch: bool = False
    warm_start_steps: int = 24
    seed: int = 0


class CauSLMAgent:
    def __init__(self, cfg: AgentConfig = None, policy=None,
                 gate: MetaController | str = "never",
                 gate_kwargs: Optional[Dict] = None,
                 var_names: Optional[Sequence[str]] = None, seed: int = 0):
        self.cfg = cfg or AgentConfig()
        self.policy = policy
        self.rng = np.random.default_rng(seed)
        if isinstance(gate, str):
            gate = META_GATES[gate](**(gate_kwargs or {}))
        self.gate = gate
        self.var_names = list(var_names or VARS)
        self.reset(full=True)

    # -------------------------------------------------------------- lifecycle
    def reset(self, full: bool = False):
        """Clear per-episode state.

        The bandit posterior and the meta-controller's value model are *not*
        cleared: on a real gateway they accumulate across days, and resetting
        them every episode would measure cold-start behaviour instead of
        steady-state control. `full=True` restores a fresh agent.
        """
        self.win: deque = deque(maxlen=self.cfg.window)
        self.G_stat = None
        self.G_repaired = None
        self.scores = None
        self.pvals = None
        self.prev_graph = None
        self.graph_churn = 0.0
        self.advice: Optional[CausalAdvice] = None
        if full or not hasattr(self, "known_spurious"):
            self.known_spurious: set = set()
            self.known_reversed: set = set()
            self.flag_counts: Dict[tuple, int] = {}
        self.plan: Dict[str, tuple] = {}
        self._pending_state = None
        self._queried = False
        self.last_terms: Dict[str, float] = {}
        self.last_query_t = -10 ** 6
        self.pending: List[Dict] = []
        self.reward_hist: deque = deque(maxlen=64)
        self.t = 0
        self.n_queries = 0
        # Two accountings, deliberately separate. `slm_energy_j` is
        # queries x the profiled single-invocation cost, which is the number
        # the paper reports: it is a clean measurement taken once on an
        # exclusively-held device. `measured_*` is what the wall clock and the
        # device-wide NVML counter said during this run, which on a shared GPU
        # also charges whatever else was resident, and is kept only as a
        # diagnostic.
        self.slm_energy_j = 0.0
        self.slm_latency_ms = 0.0
        self.measured_slm_energy_j = 0.0
        self.measured_slm_latency_ms = 0.0
        self.unit_energy_j = None
        self.unit_latency_ms = None
        self.invalid_json = 0
        self.n_overrides = 0
        self.forget_events = 0
        self.discovery_time_s = 0.0
        self.log: List[Dict] = []
        self._names_bound = False
        if hasattr(self, "gate") and self.gate is not None:
            self.gate.reset_episode()
        if full or not hasattr(self, "bandit"):
            self.bandit = FactoredCausalBandit(None, seed=self.cfg.seed,
                                               var_names=self.var_names)

    # ----------------------------------------------------------- causal stage
    def _refit_graph(self):
        if self.cfg.discovery in (None, "none") or len(self.win) < 64:
            return
        X = preprocess_for_discovery(np.stack(self.win), self.var_names)
        t0 = time.perf_counter()
        try:
            A, S, info = discover(self.cfg.discovery, X, tau_max=self.cfg.tau_max,
                                  pc_alpha=self.cfg.pc_alpha)
        except Exception:
            return
        self.discovery_time_s += time.perf_counter() - t0
        if self.prev_graph is not None:
            self.graph_churn = float(np.mean(np.abs(A - self.prev_graph)))
        self.prev_graph = A.copy()
        self.G_stat, self.scores = A, S
        self.pvals = info.get("pvals")
        # Structural knowledge is durable; the statistics are not. Edges the
        # model has already ruled out stay ruled out across refits, which is
        # what makes it viable to consult the model rarely: the gateway keeps
        # the correction without having to re-ask for it.
        self.G_repaired = self._strip_known(A)
        if self.cfg.use_graph_credit:
            self.bandit.set_graph(self.G_repaired)

    def _strip_known(self, A: np.ndarray) -> np.ndarray:
        G = np.asarray(A, float).copy()
        idx = {v: i for i, v in enumerate(self.var_names)}
        for u, v in self.known_spurious:
            if u in idx and v in idx:
                G[idx[u], idx[v]] = 0.0
        for u, v in self.known_reversed:
            if u in idx and v in idx:
                G[idx[v], idx[u]] = max(G[idx[v], idx[u]], 1.0)
        return G

    def _apply_repair(self, adv: CausalAdvice):
        if self.G_stat is None or not self.cfg.repair_with_slm:
            return
        idx = {v: i for i, v in enumerate(self.var_names)}
        for u, v in adv.spurious_edges:
            if u not in idx or v not in idx:
                continue
            # Deleting an edge is not symmetric with keeping one. A surplus
            # edge only adds noise to a knob's reward signal; a missing edge
            # removes a cost the knob causes from its view entirely, and the
            # controller then drives that cost up freely. Because the model's
            # per-flag precision is well below one, an edge is deleted only
            # once independent invocations have agreed on it.
            n = self.flag_counts.get((u, v), 0) + 1
            self.flag_counts[(u, v)] = n
            if n < self.cfg.repair_min_flags:
                continue
            self.known_spurious.add((u, v))
            # A flagged edge whose reverse is absent from the skeleton but
            # whose association is strong is a reversal, not a phantom: the
            # dependence is real and only its direction was wrong.
            if (self.G_stat[idx[v], idx[u]] == 0 and self.scores is not None
                    and self.scores[idx[u], idx[v]] > 0.15):
                self.known_reversed.add((u, v))
        self.G_repaired = self._strip_known(self.G_stat)
        if self.cfg.use_graph_credit:
            self.bandit.set_graph(self.G_repaired)

    # ------------------------------------------------------------- meta state
    def _meta_state(self, kpis: Dict) -> Dict:
        occ_var = 0.0
        drift_z = 0.0
        if len(self.win) >= 48:
            W = np.stack(self.win)
            h = len(W) // 2
            a, b = W[:h], W[h:]
            sd = (a.std(0) + b.std(0)) / 2 + 1e-8
            drift_z = float(np.max(np.abs(b.mean(0) - a.mean(0)) / sd
                                   * np.sqrt(h)))
            if "occupancy" in self.var_names:
                occ_var = float(W[-24:, self.var_names.index("occupancy")].var())
        rt = 0.0
        if len(self.reward_hist) >= 8:
            r = np.asarray(self.reward_hist)
            rt = float(r[-4:].mean() - r[:-4].mean())
        return {"t": self.t, "staleness": self.t - self.last_query_t,
                "drift_z": drift_z, "graph_churn": self.graph_churn,
                "suspicion": float(kpis.get("suspicion", 0.0)),
                "pmv": float(kpis.get("pmv", 0.0)),
                "latency_ms": float(kpis.get("latency_ms", 0.0)),
                "pdr": float(kpis.get("pdr", 1.0)),
                "reward_trend": rt, "occ_var": occ_var}

    def _ctx(self, kpis: Dict) -> np.ndarray:
        """Context for the bandit.

        The tariff and the occupancy state belong here: the right setpoint at
        a peak-price hour in an empty zone is not the right setpoint at an
        off-peak hour with forty people in it, and a context that cannot see
        the difference forces one compromise arm for both.
        """
        hod = (self.t % 288) / 12.0
        h = 2 * np.pi * hod / 24.0
        tar = float(kpis.get("tariff", 1.0))
        occ = float(kpis.get("occupancy", 0.0)) if "occupancy" in self.var_names \
            else (1.0 if 7.5 <= hod <= 19.0 else 0.0) * 6.0
        return np.array([
            np.tanh((kpis.get("indoor_temp", 22.0) - 22.0) / 5.0),
            np.tanh((kpis.get("outdoor_temp", 18.0) - 18.0) / 10.0),
            np.tanh(kpis.get("pmv", 0.0)),
            np.tanh(np.log1p(max(kpis.get("latency_ms", 50.0), 0.0)) / 6.0),
            1.0 - kpis.get("pdr", 1.0),
            np.tanh(occ / 8.0),
            float(tar > 2.0), float(tar < 0.7),
            np.sin(h), np.cos(h), 1.0], float)

    # ---------------------------------------------------------------- acting
    def act(self, env: IoTGatewayEnv, kpis: Dict) -> Tuple[List[int], Dict]:
        """Sequential convenience wrapper around the two-phase interface."""
        ctx = self.act_prepare(env, kpis)
        res = None
        if ctx is not None:
            res = self.policy.advise(ctx)
        return self.act_finish(res)

    # The step is split in two so that a lockstep runner can collect the
    # pending invocations of many episodes and issue them as one batch. The
    # cost charged to the environment is unaffected: it comes from the
    # single-invocation profile in SLMCostConfig, not from the batched call.
    def act_prepare(self, env: IoTGatewayEnv, kpis: Dict) -> Optional[Dict]:
        if not self._names_bound:
            names = list(env.observed_vars())
            if names != self.var_names:
                self.var_names = names
                self.bandit.var_names = names
                self.bandit.set_graph(None)
            self._names_bound = True
        if kpis.get("_row") is not None:
            self.win.append(kpis["_row"])
        if self.t % self.cfg.graph_period == 0:
            self._refit_graph()

        ms = self._meta_state(kpis)
        ctx = self._ctx(kpis)
        # A detected regime change makes the controller's own history less
        # relevant, so it is discounted. This is what lets the model's advice
        # become usable -- and therefore worth its price -- more than once in
        # the lifetime of a deployment.
        # The trigger has to be a property of the *world*, not of the
        # estimator watching it. A mean shift between the two halves of the
        # window is not one -- in a building those halves are day and night,
        # so that statistic is large every step. Skeleton churn is not one
        # either, and using it was actively harmful: a discovery method whose
        # output is unstable, such as Granger here, then forgets continuously
        # and never converges, which penalises the baseline for a property of
        # its estimator. Device suspicion is a direct sensor-level signal of a
        # changed world, so that is what we use.
        shift = float(ms["suspicion"] > 0.6)
        if shift > 0.0:
            self.bandit.forget_step(shift)
        self.forget_events += int(shift > 0.0)
        self._pending_state = (ms, ctx, kpis, env)

        queried = False
        if (self.policy is not None and self.t >= self.cfg.warm_start_steps
                and len(self.win) >= 32):
            queried = bool(self.gate.should_query(ms))
        self._queried = queried
        if not queried:
            return None

        G_in = (self.G_stat if self.G_stat is not None
                else np.zeros((len(self.var_names), len(self.var_names)),
                              np.float32))
        cand_idx = self.rng.choice(len(CANDIDATES),
                                   min(self.cfg.n_candidates, len(CANDIDATES)),
                                   replace=False)
        cands = [CANDIDATES[int(i)] for i in cand_idx]
        return {
            "A": G_in, "X": np.stack(self.win), "pvals": self.pvals,
            "scores": self.scores, "var_names": self.var_names,
            "candidates": cands,
            "device_meta": {"n_devices": env.nc.n_devices, "broker": "mqtt",
                            "link": "wifi2.4"},
            "regime": {"drift": "yes" if env.opts.drift else "no",
                       "attack": "yes" if ms["suspicion"] > 0.3 else "no",
                       "occupancy_sensor": ("absent" if
                                            env.opts.hidden_confounder
                                            else "present")},
            "features": ctx, "kpis": kpis}

    def act_finish(self, result) -> Tuple[List[int], Dict]:
        ms, ctx, kpis, env = self._pending_state
        queried = self._queried
        if queried:
            adv, cost = result if result is not None else (None, None)
            self.n_queries += 1
            if cost is not None:
                self.measured_slm_energy_j += cost.energy_j
                self.measured_slm_latency_ms += cost.latency_ms
            if self.unit_energy_j is None:
                sc = getattr(env, "sc", None)
                self.unit_energy_j = float(getattr(sc, "energy_j", 0.0))
                self.unit_latency_ms = float(getattr(sc, "latency_ms", 0.0))
            self.slm_energy_j += self.unit_energy_j
            self.slm_latency_ms += self.unit_latency_ms
            if adv is None or not adv.valid:
                self.invalid_json += 1
            else:
                self.advice = adv
                self._apply_repair(adv)
                self.bandit.inject_prior(adv.do, adv.predicted_ATE, ctx,
                                         pmv_now=kpis.get("pmv", 0.0),
                                         kpis=kpis,
                                         baseline=self.last_terms)
                if self.cfg.use_plan_override and adv.safety != "unsafe":
                    # Plans accumulate per knob rather than replacing each
                    # other. One invocation names one intervention, so a
                    # single plan would pin one actuator and leave the other
                    # three to an untrained bandit -- which is exactly the
                    # exploration cost the model is supposed to remove. Each
                    # knob keeps its own expiry and falls back to the bandit
                    # when its advice goes stale.
                    for k, v in adv.do.items():
                        if k in KNOB_ARMS:
                            self.plan[k] = (float(v),
                                            self.t + self.cfg.plan_ttl)
            self.last_query_t = self.t
            self.pending.append({"t": self.t, "state": ms,
                                 "baseline": float(np.mean(self.reward_hist))
                                 if self.reward_hist else 0.0, "acc": [],
                                 "n": 0})

        safety = self.advice.safety if self.advice else "ok"
        live = {}
        for k in list(self.plan):
            v, exp = self.plan[k]
            if self.t > exp:
                del self.plan[k]
            else:
                live[k] = v
        knobs = self.bandit.select(ctx, safety=safety, kpis=kpis,
                                   advice=live if self.cfg.use_plan_override
                                   else None)
        self.n_overrides += len(getattr(self.bandit, "_override", {}))
        action = self._to_action(knobs, query=queried, kpis=kpis)
        self._last_ctx = ctx
        return action, {"queried": queried, "knobs": knobs, "safety": safety,
                        "meta_state": ms}

    def _to_action(self, knobs: Dict[str, float], query: bool,
                   kpis: Dict) -> List[int]:
        a = []
        for k in ACTION_KEYS:
            if k in knobs:
                lv = np.asarray(ACTION_LEVELS[k])
                a.append(int(np.argmin(np.abs(lv - knobs[k]))))
            elif k == "isolate":
                a.append(1 if kpis.get("suspicion", 0.0) > 0.75 else 0)
            elif k == "query_slm":
                a.append(int(query))
            else:
                a.append(0)
        return a

    # -------------------------------------------------------------- feedback
    def observe(self, reward: float, kpis: Dict, rw):
        self.reward_hist.append(reward)
        tariff = float(kpis.get("tariff", 1.0))
        terms = {
            "hvac_cost": -rw.w_energy * tariff * abs(kpis.get("hvac_power", 0.0))
            * (300.0 / 3600.0),
            "energy": -rw.w_energy * kpis.get(
                "energy_cost", kpis.get("energy_kwh", 0.0)),
            "comfort": -rw.w_comfort * max(0.0, abs(kpis.get("pmv", 0.0))
                                           - rw.comfort_band),
            "latency": -rw.w_latency * min(max(0.0, kpis.get("latency_ms", 0.0)
                                               - rw.latency_sla_ms),
                                           rw.latency_pen_cap_ms),
            "pdr": -rw.w_pdr * (1.0 - kpis.get("pdr", 1.0)),
            "it_energy": -0.004 * kpis.get("it_energy_j", 0.0) / 1000.0,
        }
        self.last_terms = terms
        self.bandit.update(self._last_ctx, terms)

        # score every open invocation over its evaluation horizon
        done = []
        for p in self.pending:
            p["acc"].append(reward)
            p["n"] += 1
            if p["n"] >= self.cfg.evoi_horizon:
                gain = float(np.mean(p["acc"]) - p["baseline"])
                self.gate.observe(p["state"], True, gain)
                done.append(p)
        for p in done:
            self.pending.remove(p)
        self.t += 1

    def summary(self) -> Dict:
        return {"n_queries": self.n_queries,
                "slm_energy_j": self.slm_energy_j,
                "slm_latency_ms_total": self.slm_latency_ms,
                "slm_j_per_query": self.slm_energy_j / max(self.n_queries, 1),
                "slm_ms_per_query": self.slm_latency_ms / max(self.n_queries, 1),
                "measured_slm_energy_j": self.measured_slm_energy_j,
                "measured_slm_ms_per_query": (self.measured_slm_latency_ms
                                              / max(self.n_queries, 1)),
                "invalid_json": self.invalid_json,
                "invalid_json_rate": self.invalid_json / max(self.n_queries, 1),
                "n_overrides": self.n_overrides,
                "forget_events": self.forget_events,
                "override_rate": self.n_overrides / max(self.t, 1),
                "discovery_time_s": self.discovery_time_s,
                "query_rate": self.n_queries / max(self.t, 1),
                "gate": getattr(self.gate, "name", "?"),
                "gate_stats": self.gate.stats()}

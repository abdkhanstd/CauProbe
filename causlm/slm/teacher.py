"""Oracle-interventional teacher for distilling the causal world model.

No external LLM API is used. The teacher's answers are *derived*, not
generated, which removes the usual worry that a distilled student inherits a
teacher's hallucinations:

  * `spurious_edges` -- the false positives that PCMCI+ actually produced on
    this window, compared against the environment's structural equations;
  * `confounders`    -- the variables in the true graph that are common causes
    of the flagged pairs;
  * `do`             -- the candidate intervention with the highest true
    utility, found by rolling the simulator forward under each candidate;
  * `predicted_ATE`  -- the true average treatment effect of that candidate,
    computed with common random numbers;
  * `safety`         -- a constraint check on the resulting trajectory;
  * `explain`        -- a factual sentence assembled from the above.

Everything the student is trained to say is therefore checkable against the
simulator that produced it.
"""
from __future__ import annotations

import itertools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..causal import discover, preprocess_for_discovery
from ..config import ACTUATORS, VARS, true_env_graph
from ..env import ACTION_LEVELS, EnvOptions, IoTGatewayEnv
from .prompts import SYSTEM_PROMPT, build_prompt
from .schema import CausalAdvice, advice_to_json

# The model predicts the tariff-weighted cost rather than raw kWh: that is
# the quantity the reward charges, and asking for both would only lengthen
# the output the gateway has to pay to generate.
ATE_KEYS = ["energy_cost", "indoor_temp", "latency_ms", "pmv"]

# Candidate intervention menu offered to the model at every invocation.
CANDIDATES: List[Dict[str, float]] = (
    [{"hvac_setpoint": v} for v in (20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0)]
    + [{"sample_rate_hz": v} for v in (0.1, 0.25, 0.5, 1.0)]
    + [{"duty_cycle": v} for v in (0.25, 0.5, 1.0)]
    + [{"offload_frac": v} for v in (0.0, 0.5, 1.0)]
)


def _apply_candidate(base_action: Sequence[int], cand: Dict[str, float]):
    a = list(base_action)
    for k, v in cand.items():
        levels = ACTION_LEVELS[k]
        a[list(ACTION_LEVELS).index(k)] = int(np.argmin(np.abs(np.asarray(levels) - v)))
    return a


from ..utility import intervention_utility


def utility(ate: Dict[str, float], pmv_now: float = 0.0, kpis=None, **_) -> float:
    """Thin alias so existing call sites keep working."""
    k = dict(kpis or {})
    k.setdefault("pmv", pmv_now)
    return intervention_utility(ate, k)


def _ancestors(G: np.ndarray, j: int, depth: int = 3) -> Dict[int, int]:
    """Map ancestor index -> shortest directed-path length into j."""
    cur, anc = {j}, {}
    for d in range(1, depth + 1):
        nxt = set()
        for c in cur:
            for i in np.flatnonzero(G[:, c] > 0):
                i = int(i)
                if i != j and i not in anc:
                    anc[i] = d
                    nxt.add(i)
        cur = nxt
        if not cur:
            break
    return anc


def _descendants(G: np.ndarray, i: int, depth: int = 3) -> set:
    cur, des = {i}, set()
    for _ in range(depth):
        nxt = set()
        for c in cur:
            for j in np.flatnonzero(G[c] > 0):
                j = int(j)
                if j != i and j not in des:
                    des.add(j)
                    nxt.add(j)
        cur = nxt
        if not cur:
            break
    return des


def label_graph_errors(A: np.ndarray, obs_names: Sequence[str],
                       G_full: np.ndarray, full_names: Sequence[str]):
    """Split PCMCI+'s output into spurious edges, confounders and misses.

    The confounder search runs on the *full* structural graph, including
    variables the gateway cannot measure. That is deliberate: when the
    occupancy sensor is absent, the correct answer to "why is there an edge
    wifi_rssi -> indoor_temp?" is "occupancy", a variable that does not appear
    in the data at all. A confounder must be a common ancestor of both
    endpoints that is not itself a descendant of either.
    """
    fidx = {v: i for i, v in enumerate(full_names)}
    spurious, reversed_edges, missed = [], [], []
    conf_votes: Dict[str, int] = {}
    n = len(obs_names)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            u, v = obs_names[i], obs_names[j]
            fi, fj = fidx[u], fidx[v]
            true_ij = G_full[fi, fj] > 0
            if A[i, j] > 0 and not true_ij:
                spurious.append((u, v))
                if G_full[fj, fi] > 0:
                    reversed_edges.append((u, v))
                    continue
                au, av = _ancestors(G_full, fi), _ancestors(G_full, fj)
                des = _descendants(G_full, fi) | _descendants(G_full, fj)
                shared = [(au[k] + av[k], k) for k in set(au) & set(av)
                          if k not in des and k not in (fi, fj)]
                for _, k in sorted(shared)[:2]:
                    name = full_names[k]
                    conf_votes[name] = conf_votes.get(name, 0) + 1
            if A[i, j] == 0 and true_ij:
                missed.append((u, v))
    confounders = [k for k, _ in sorted(conf_votes.items(),
                                        key=lambda kv: -kv[1])][:4]
    return spurious, confounders, missed, reversed_edges


@dataclass
class TeacherSample:
    prompt: str
    target: str
    meta: Dict


def make_sample(seed: int, window: int = 288, regime: Optional[Dict] = None,
                horizon: int = 8, n_mc: int = 6, tau_max: int = 3,
                pc_alpha: float = 0.05, n_candidates: int = 8,
                persist: int = 4, store_window: bool = False
                ) -> Optional[TeacherSample]:
    rng = np.random.default_rng(seed)
    regime = regime or {}
    opts = EnvOptions(episode_steps=window + horizon * 4 + 20,
                      hidden_confounder=regime.get("hidden_confounder", False),
                      drift=regime.get("drift", False),
                      attack=regime.get("attack", False), seed=seed)
    env = IoTGatewayEnv(opts)
    env.reset(seed=seed)
    # Seed the action space too: without this the exploratory behaviour policy
    # is drawn from an unseeded global stream and the window cannot be
    # reconstructed afterwards for evaluation.
    env.action_space.seed(seed)

    a = env.action_space.sample()
    a[5] = 0
    for t in range(window):
        if t % persist == 0:
            a = env.action_space.sample()
            a[5] = 0
        env.step(a)

    names = env.observed_vars()
    Xraw = env.get_window(window)
    if Xraw.shape[0] < window // 2:
        return None
    Z = preprocess_for_discovery(Xraw, names)
    try:
        A, S, info = discover("pcmci+", Z, tau_max=tau_max, pc_alpha=pc_alpha)
    except Exception:
        return None

    Gfull = true_env_graph()
    keep = [VARS.index(v) for v in names]
    G = Gfull[np.ix_(keep, keep)]
    spurious, confounders, missed, reversed_edges = label_graph_errors(
        A, names, Gfull, VARS)

    cand_idx = rng.choice(len(CANDIDATES), size=min(n_candidates, len(CANDIDATES)),
                          replace=False)
    cands = [CANDIDATES[int(i)] for i in cand_idx]
    ref = list(a)
    ref[5] = 0
    pmv_now = float(env._last_kpis.get("pmv", 0.0))
    cands = cands + [{}]                     # "leave the gateway alone"
    best, best_u, best_ate = None, -np.inf, None
    all_ate = {}
    for c in cands:
        act = _apply_candidate(ref, c)
        try:
            ate = env.counterfactual_ate(
                act, ref, horizon=horizon, n_mc=n_mc,
                targets=("energy_cost", "energy_kwh", "indoor_temp",
                         "latency_ms", "it_energy_j", "hvac_power",
                         "wifi_rssi", "tariff"))
        except Exception:
            continue
        u = utility(ate, pmv_now=pmv_now, kpis=env._last_kpis)
        all_ate[json.dumps(c, sort_keys=True)] = {k: float(v) for k, v in ate.items()}
        if u > best_u:
            best, best_u, best_ate = c, u, ate
    if best is None:
        return None

    k = env._last_kpis
    t_after = k["indoor_temp"] + best_ate.get("indoor_temp", 0.0)
    pmv_after = k.get("pmv", 0.0) + best_ate.get("pmv", 0.0)
    if t_after < 17.0 or t_after > 29.0 or k.get("pdr", 1.0) < 0.55:
        safety = "unsafe"
    elif abs(pmv_after) > 1.0 or t_after < 18.5 or t_after > 27.5:
        safety = "caution"
    else:
        safety = "ok"

    if best:
        knob, val = next(iter(best.items()))
        do = {knob: float(val)}
        act_txt = (f"Setting {knob} to {val:g} changes energy cost by "
                   f"{best_ate.get('energy_cost', 0.0):+.3f} and PMV by "
                   f"{best_ate.get('pmv', 0.0):+.2f}")
    else:
        cur = {k: float(env._last_kpis.get(k, 0.0)) for k in ("hvac_setpoint",)}
        do = cur
        act_txt = "No candidate beats the current settings, so hold them"
    if reversed_edges:
        u0, v0 = reversed_edges[0]
        why = f"{u0}->{v0} points the wrong way; the true direction is {v0}->{u0}"
    elif spurious and confounders:
        u0, v0 = spurious[0]
        why = f"{u0}->{v0} is not causal; {confounders[0]} drives both"
    elif spurious:
        u0, v0 = spurious[0]
        why = f"{u0}->{v0} is not causal"
    else:
        why = "the skeleton is consistent with the physics"
    explain = f"{why}. {act_txt}."

    adv = CausalAdvice(
        do=do,
        predicted_ATE={kk: float(best_ate.get(kk, 0.0)) for kk in ATE_KEYS},
        confounders=confounders, spurious_edges=spurious[:6],
        safety=safety, explain=explain)

    prompt = build_prompt(A, Xraw, pvals=info.get("pvals"), scores=S,
                          var_names=names, candidates=cands,
                          device_meta={"n_devices": opts.network.n_devices,
                                       "broker": "mqtt", "link": "wifi2.4"},
                          regime={"drift": "yes" if opts.drift else "no",
                                  "attack": "yes" if opts.attack else "no",
                                  "occupancy_sensor": ("absent" if
                                                       opts.hidden_confounder
                                                       else "present")})
    meta = {"seed": seed, "regime": regime, "n_spurious": len(spurious),
            "n_missed": len(missed), "n_reversed": len(reversed_edges),
            "best_utility": float(best_u), "pmv_now": pmv_now,
            "true_confounders": confounders,
            "true_spurious": [list(e) for e in spurious],
            "true_missed": [list(e) for e in missed],
            "all_ate": all_ate, "candidates": cands,
            "stat_edges": int(A.sum()), "true_edges": int(G.sum()),
            "kpis": {kk: float(vv) for kk, vv in k.items()
                     if isinstance(vv, (int, float))},
            "graph_pred": A.tolist(), "graph_true": G.tolist(),
            "var_names": names}
    if store_window:
        meta["window"] = np.round(Xraw, 4).tolist()
        meta["scores"] = np.round(S, 4).tolist()
    return TeacherSample(prompt=prompt, target=advice_to_json(adv), meta=meta)


def _worker(args):
    seed, regime, kw = args
    try:
        return make_sample(seed, regime=regime, **kw)
    except Exception:
        return None


def generate_dataset(n: int = 6000, out: Path = None, n_jobs: int = 40,
                     regimes: Optional[List[Dict]] = None, seed0: int = 10_000,
                     **kw) -> List[TeacherSample]:
    """Build the distillation corpus in parallel across worker processes."""
    from multiprocessing import Pool
    regimes = regimes or [
        {},                                                    # nominal
        {"hidden_confounder": True},
        {"drift": True},
        {"attack": True},
        {"hidden_confounder": True, "drift": True},
        {"hidden_confounder": True, "attack": True},
        {"drift": True, "attack": True},
        {"hidden_confounder": True, "drift": True, "attack": True},
    ]
    jobs = [(seed0 + i, regimes[i % len(regimes)], kw) for i in range(n)]
    out_samples = []
    with Pool(n_jobs) as pool:
        for i, s in enumerate(pool.imap_unordered(_worker, jobs, chunksize=4)):
            if s is not None:
                out_samples.append(s)
            if (i + 1) % 250 == 0:
                print(f"    {i + 1}/{n} generated ({len(out_samples)} kept)",
                      flush=True)
    if out is not None:
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            for s in out_samples:
                f.write(json.dumps({"prompt": s.prompt, "target": s.target,
                                    "meta": s.meta}) + "\n")
    return out_samples

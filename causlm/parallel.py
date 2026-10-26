"""Lockstep evaluation of many episodes with batched language-model calls.

Running one episode at a time makes the language-model rows unaffordable: a
single always-on day is 288 sequential `generate` calls, and Hugging Face
generation is dominated by per-token Python overhead rather than by arithmetic.
Stepping P episodes together and issuing their pending invocations as one batch
recovers roughly an order of magnitude.

This changes only how fast the evaluation runs, never what it reports. The
environment charges `do(QuerySLM)` from `SLMCostConfig`, which is filled in from
the *single*-invocation profile measured by `experiments/profile_slm.py`, so the
Joules and milliseconds in the paper are the ones a gateway issuing one call at
a time would actually pay.

Learning still accumulates across days. The bandit posteriors and the
meta-controller's value model are Gaussian sufficient statistics of the form
`A += x x^T`, `b += r x`, so the increments collected by P independent copies
can be summed back into the master agent exactly, which is what `_merge` does.
"""
from __future__ import annotations

import copy
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from .config import VARS
from .env import EnvOptions, IoTGatewayEnv
from .runner import aggregate


def _row_for(env: IoTGatewayEnv, kp: Dict) -> np.ndarray:
    names = env.observed_vars()
    return np.array([kp[v] for v in names], np.float32)


#: attributes that must be shared by reference rather than copied -- the
#: language model itself (a CUDA module is not deep-copyable and there is no
#: reason to have P of them) and any transient handle to an environment.
_SHARED = ("policy", "_pending_state", "_env", "wm")


def _clone(master, seed: int):
    stash = {}
    for k in _SHARED:
        if hasattr(master, k):
            stash[k] = getattr(master, k)
            setattr(master, k, None)
    try:
        a = copy.deepcopy(master)
    finally:
        for k, v in stash.items():
            setattr(master, k, v)
    for k, v in stash.items():
        setattr(a, k, v)
    try:
        a.reset(full=False)
    except TypeError:
        a.reset()
    except AttributeError:
        pass
    for k, v in stash.items():          # reset() may have cleared them again
        if getattr(a, k, None) is None and k in ("policy",):
            setattr(a, k, v)
    if hasattr(a, "rng"):
        a.rng = np.random.default_rng(seed)
    return a


def _snapshot_stats(agent):
    """Sufficient statistics that a batch of clones may add to.

    Agents without a bandit or a learning gate (rule-based, PPO, the world-model
    baseline) simply have nothing to merge, and the runner still works.
    """
    if not hasattr(agent, "bandit"):
        return {}
    st = {"bandit": {k: (v.A.copy(), v.b.copy(), v.counts.copy())
                     for k, v in agent.bandit.b.items()}}
    g = getattr(agent, "gate", None)
    if g is None:
        return st
    if hasattr(g, "A") and hasattr(g, "b"):
        st["gate"] = (np.array(g.A, copy=True), np.array(g.b, copy=True),
                      getattr(g, "n_obs", 0))
    if hasattr(g, "xs"):
        st["gate_cal"] = (list(g.xs), list(g.ys))
    return st


def _merge(master, clones, base):
    """Fold each clone's *increment* in sufficient statistics into the master."""
    if not base:
        for c in clones:
            for f in ("n_queries", "slm_energy_j", "slm_latency_ms",
                      "measured_slm_energy_j", "measured_slm_latency_ms",
                      "invalid_json", "n_overrides", "forget_events",
                      "discovery_time_s"):
                if hasattr(master, f) and hasattr(c, f):
                    setattr(master, f, getattr(master, f) + getattr(c, f))
        return
    for k, lin in master.bandit.b.items():
        A0, b0, c0 = base["bandit"][k]
        for c in clones:
            cl = c.bandit.b.get(k)
            if cl is None:
                continue
            lin.A += cl.A - A0
            lin.b += cl.b - b0
            lin.counts += cl.counts - c0
    g = getattr(master, "gate", None)
    if g is not None and "gate" in base and hasattr(g, "A"):
        A0, b0, n0 = base["gate"]
        for c in clones:
            gc = c.gate
            if not hasattr(gc, "A"):
                continue
            g.A = np.asarray(g.A) + (np.asarray(gc.A) - A0)
            g.b = np.asarray(g.b) + (np.asarray(gc.b) - b0)
            if hasattr(g, "n_obs"):
                g.n_obs += max(getattr(gc, "n_obs", 0) - n0, 0)
        try:
            g.w = np.linalg.solve(g.A, g.b)
        except np.linalg.LinAlgError:
            pass
    if g is not None and "gate_cal" in base and hasattr(g, "xs"):
        x0, y0 = base["gate_cal"]
        for c in clones:
            g.xs.extend(c.gate.xs[len(x0):])
            g.ys.extend(c.gate.ys[len(y0):])
        if hasattr(g, "cal") and len(g.xs) >= 12:
            g.cal.fit(g.xs, g.ys)
    # per-episode counters that the caller reports
    for c in clones:
        for f in ("n_queries", "slm_energy_j", "slm_latency_ms",
                  "measured_slm_energy_j", "measured_slm_latency_ms",
                  "invalid_json", "n_overrides", "forget_events",
                      "discovery_time_s"):
            if hasattr(master, f) and hasattr(c, f):
                setattr(master, f, getattr(master, f) + getattr(c, f))


def run_batch(master, seeds: Sequence[int], env_options: EnvOptions,
              policy=None, collect_traj: bool = False,
              merge_back: bool = True, factory=None) -> List[Dict]:
    """Run len(seeds) episodes in lockstep, batching the invocations.

    Agents that carry learned state (the bandit family) are cloned from the
    master and merged back afterwards. Agents that do not -- a rule-based
    supervisor, a frozen PPO policy -- are simply rebuilt from the factory,
    which also avoids deep-copying a torch module.
    """
    learns = hasattr(master, "bandit")
    base = _snapshot_stats(master) if (merge_back and learns) else None
    if learns or factory is None:
        agents = [_clone(master, 7919 * s + 13) for s in seeds]
    else:
        agents = [factory(s) for s in seeds]
    envs = []
    for s in seeds:
        o = EnvOptions(**{**env_options.__dict__, "seed": s})
        e = IoTGatewayEnv(o)
        e.reset(seed=s)
        envs.append(e)

    kpis = []
    for e in envs:
        k = dict(e._last_kpis)
        k["suspicion"] = float(e.suspicion.max())
        k["_row"] = None
        kpis.append(k)

    T = envs[0].T
    done = [False] * len(envs)
    trajs = [[] for _ in envs]
    for t in range(T):
        ctxs, owners = [], []
        for i, (a, e) in enumerate(zip(agents, envs)):
            if done[i] or not hasattr(a, "act_prepare"):
                continue
            c = a.act_prepare(e, kpis[i])
            if c is not None:
                ctxs.append(c)
                owners.append(i)
        results: Dict[int, object] = {}
        if ctxs:
            pol = policy if policy is not None else agents[owners[0]].policy
            out = pol.advise_batch(ctxs)
            for i, r in zip(owners, out):
                results[i] = r
        for i, (a, e) in enumerate(zip(agents, envs)):
            if done[i]:
                continue
            if hasattr(a, "act_finish"):
                action, ainfo = a.act_finish(results.get(i))
            else:
                action, ainfo = a.act(e, kpis[i])
            _, r, d, _, kp = e.step(action)
            k = dict(kp)
            k["suspicion"] = float(e.suspicion.max())
            k["_row"] = _row_for(e, kp)
            kpis[i] = k
            if hasattr(a, "observe"):
                a.observe(r, k, e.rw)
            if collect_traj:
                trajs[i].append({"t": t, "reward": r,
                                 **{kk: vv for kk, vv in kp.items()
                                    if isinstance(vv, (int, float))},
                                 "queried": ainfo.get("queried", False)})
            done[i] = d
        if all(done):
            break

    rows = []
    for i, (a, e) in enumerate(zip(agents, envs)):
        row = e.episode_summary()
        if hasattr(a, "summary"):
            row.update({f"agent_{k}": v for k, v in a.summary().items()})
        if collect_traj:
            row["_traj"] = trajs[i]
        rows.append(row)
    if merge_back and learns:
        _merge(master, agents, base)
    return rows


def evaluate_parallel(agent_factory: Callable[[int], object],
                      n_episodes: int = 20,
                      env_options: Optional[EnvOptions] = None,
                      seed0: int = 5000, warmup_episodes: int = 0,
                      warmup_seed0: int = 90_000, batch: int = 16,
                      warmup_batch: int = 8,
                      policy=None, collect_traj: bool = False,
                      verbose: bool = False, shared_agent: bool = True,
                      warm_master=None) -> Dict:
    """Evaluate `agent_factory` on `n_episodes` held-out days.

    `warm_master` lets several configurations start from one already-warmed
    controller. That is both cheaper and a cleaner comparison: when the thing
    under test is the invocation gate, every gate should be handed the same
    bandit posterior rather than each warming its own.
    """
    opts = env_options or EnvOptions()
    if warm_master is not None:
        master = _clone(warm_master, seed0)
        proto = agent_factory(seed0)
        # take the configuration under test, keep the learned state
        for attr in ("policy", "gate", "cfg"):
            if hasattr(proto, attr):
                setattr(master, attr, getattr(proto, attr))
        # The bandit's own config is part of the configuration under test --
        # the advice horizon and the ATE prior weight live there. Only the
        # config is taken; the learned posterior stays the master's, which is
        # the whole point of sharing a warmed controller. Without this the
        # bandit-config ablations silently re-run the reference: they returned
        # bit-for-bit identical numbers, which is how the bug was found.
        if hasattr(proto, "bandit") and hasattr(master, "bandit"):
            if hasattr(proto.bandit, "cfg"):
                master.bandit.cfg = proto.bandit.cfg
        master.reset(full=False)
        master.n_queries = 0
        master.slm_energy_j = 0.0
        master.slm_latency_ms = 0.0
        master.measured_slm_energy_j = 0.0
        master.measured_slm_latency_ms = 0.0
        master.invalid_json = 0
        master.discovery_time_s = 0.0
    else:
        master = agent_factory(seed0)

    # Clones inside one warm-up batch do not see each other's updates, so the
    # warm-up uses a smaller batch than the evaluation: it trades a little
    # speed for learning that tracks the sequential reference closely.
    wb = max(1, min(warmup_batch, batch))
    for start in range(0, warmup_episodes, wb):
        seeds = list(range(warmup_seed0 + start,
                           warmup_seed0 + min(start + wb, warmup_episodes)))
        run_batch(master, seeds, opts, policy=policy, merge_back=True,
                  factory=agent_factory)
        if verbose:
            print(f"    warmup {min(start + wb, warmup_episodes)}"
                  f"/{warmup_episodes}", flush=True)

    rows = []
    for start in range(0, n_episodes, batch):
        seeds = list(range(seed0 + start,
                           seed0 + min(start + batch, n_episodes)))
        rows += run_batch(master, seeds, opts, policy=policy,
                          collect_traj=collect_traj, merge_back=False,
                          factory=agent_factory)
        if verbose:
            print(f"    eval {len(rows)}/{n_episodes}", flush=True)
    out = aggregate(rows)
    out["_master"] = master
    return out


def warm_up(agent_factory, episodes: int, env_options: Optional[EnvOptions] = None,
            seed0: int = 90_000, batch: int = 8, policy=None, verbose=False):
    """Warm one controller on `episodes` days and hand it back.

    Used when several configurations should be compared from an identical
    learned state -- the gate study in Experiment 5, for instance.
    """
    opts = env_options or EnvOptions()
    master = agent_factory(seed0)
    for start in range(0, episodes, batch):
        seeds = list(range(seed0 + start,
                           seed0 + min(start + batch, episodes)))
        run_batch(master, seeds, opts, policy=policy, merge_back=True,
                  factory=agent_factory)
        if verbose:
            print(f"    warm {min(start + batch, episodes)}/{episodes}",
                  flush=True)
    return master

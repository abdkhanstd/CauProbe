"""Episode driver shared by every method in the paper.

`run_episode` is the single code path: model-free baselines, the statistical
causal controller, the language-model variants and the full CauSLM-Edge agent
all go through it, so no comparison can be won by a difference in the harness.
"""
from __future__ import annotations

import time
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from .config import VARS
from .env import ACTION_LEVELS, EnvOptions, IoTGatewayEnv

ACTION_KEYS = list(ACTION_LEVELS)


def run_episode(env: IoTGatewayEnv, agent, seed: int = 0,
                collect_traj: bool = False) -> Dict:
    obs, info = env.reset(seed=seed)
    if hasattr(agent, "reset"):
        agent.reset()
    kpis = dict(env._last_kpis)
    kpis["suspicion"] = float(env.suspicion.max())
    kpis["_row"] = None
    traj = []
    t0 = time.perf_counter()
    total_r = 0.0
    for t in range(env.T):
        action, ainfo = agent.act(env, kpis)
        obs, r, done, _, kp = env.step(action)
        total_r += r
        kpis = dict(kp)
        kpis["suspicion"] = float(env.suspicion.max())
        kpis["_row"] = np.array([kp[v] for v in VARS], np.float32) \
            if not env.opts.hidden_confounder else \
            np.array([kp[v] for v in VARS if v != "occupancy"], np.float32)
        if hasattr(agent, "observe"):
            agent.observe(r, kpis, env.rw)
        if collect_traj:
            traj.append({"t": t, "reward": r, **{k: v for k, v in kp.items()
                                                 if isinstance(v, (int, float))},
                         "queried": ainfo.get("queried", False)})
        if done:
            break
    out = env.episode_summary()
    out["wall_time_s"] = time.perf_counter() - t0
    if hasattr(agent, "summary"):
        out.update({f"agent_{k}": v for k, v in agent.summary().items()})
    if collect_traj:
        out["_traj"] = traj
    return out


def evaluate(agent_factory: Callable[[int], object], n_episodes: int = 20,
             env_options: Optional[EnvOptions] = None, seed0: int = 5000,
             collect_traj: bool = False, verbose: bool = False,
             warmup_episodes: int = 0, warmup_seed0: int = 90_000,
             shared_agent: bool = True) -> Dict:
    """Optionally let the agent learn on `warmup_episodes` days first.

    Warm-up days use a disjoint seed range from the evaluation days, so
    nothing is scored on data the agent adapted to.
    """
    base = dict(env_options.__dict__) if env_options else {}
    agent = agent_factory(seed0) if shared_agent else None
    for i in range(warmup_episodes):
        opts = EnvOptions(**{**base, "seed": warmup_seed0 + i})
        a = agent if shared_agent else agent_factory(warmup_seed0 + i)
        run_episode(IoTGatewayEnv(opts), a, seed=warmup_seed0 + i)
        if verbose and (i + 1) % 5 == 0:
            print(f"    warmup {i+1}/{warmup_episodes}", flush=True)
    rows = []
    for i in range(n_episodes):
        opts = EnvOptions(**{**base, "seed": seed0 + i})
        env = IoTGatewayEnv(opts)
        a = agent if shared_agent else agent_factory(seed0 + i)
        rows.append(run_episode(env, a, seed=seed0 + i,
                                collect_traj=collect_traj))
        if verbose:
            print(f"    ep {i}: return {rows[-1]['return']:.2f}", flush=True)
    return aggregate(rows)


def aggregate(rows: List[Dict]) -> Dict:
    keys = [k for k in rows[0] if not k.startswith("_")
            and isinstance(rows[0][k], (int, float, np.integer, np.floating))]
    out = {}
    for k in keys:
        v = np.array([r[k] for r in rows], float)
        out[k] = float(v.mean())
        out[k + "_std"] = float(v.std(ddof=1)) if len(v) > 1 else 0.0
        out[k + "_sem"] = float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0
    out["n_episodes"] = len(rows)
    out["_rows"] = rows
    return out


# ------------------------------------------------------------------ agents
class FixedAgent:
    """Constant actuator settings. Used as the trivial control."""

    def __init__(self, knobs: Dict[str, float], query: bool = False):
        self.knobs, self.query = knobs, query

    def reset(self):
        pass

    def act(self, env, kpis):
        a = []
        for k in ACTION_KEYS:
            if k in self.knobs:
                lv = np.asarray(ACTION_LEVELS[k])
                a.append(int(np.argmin(np.abs(lv - self.knobs[k]))))
            elif k == "query_slm":
                a.append(int(self.query))
            else:
                a.append(0)
        return a, {"queried": self.query}

    def summary(self):
        return {}


class RuleBasedAgent:
    """ASHRAE-style deadband supervisor with a simple network guard.

    Setpoint follows an occupied / unoccupied schedule; the sampling rate is
    backed off when the broker queue saturates; offloading is enabled when the
    queue stays hot. This is the controller a building operator would actually
    deploy, and it is the reference every learned method has to beat.
    """

    def __init__(self, occupied_sp: float = 23.0, setback: float = 3.0,
                 seed: int = 0):
        self.occ_sp, self.setback = occupied_sp, setback
        self.reset()

    def reset(self):
        self.hot = 0
        self.t = 0
        self.oat_ema = None
        self.cooling_season = True

    def act(self, env, kpis):
        hod = (self.t % 288) / 12.0
        occupied = 7.5 <= hod <= 19.0
        # Unoccupied setback has to move *away* from the load, not always up:
        # setting back to 26 C on a cold night makes the plant heat harder.
        # The direction follows a seasonal changeover on a slow outdoor-air
        # average with a dead band, exactly so the setpoint cannot chatter
        # between heating and cooling as the instantaneous temperature crosses
        # the zone -- a controller that did that would burn energy in both
        # directions on the same night.
        oat = float(kpis.get("outdoor_temp", 18.0))
        self.oat_ema = oat if self.oat_ema is None else \
            0.995 * self.oat_ema + 0.005 * oat
        if self.cooling_season and self.oat_ema < 16.0:
            self.cooling_season = False
        elif not self.cooling_season and self.oat_ema > 20.0:
            self.cooling_season = True
        sp = self.occ_sp
        if not occupied:
            sp = self.occ_sp + (self.setback if self.cooling_season
                                else -self.setback)
            sp = float(min(max(sp, 18.0), 27.0))
        # Network guard: an engineered gateway does not publish at a rate the
        # broker cannot serve. The nominal rate is already conservative and it
        # backs off further whenever the queue runs hot, which is what a
        # commissioned deployment does.
        lat = kpis.get("latency_ms", 50.0)
        self.hot = self.hot + 1 if lat > env.rw.latency_sla_ms \
            else max(0, self.hot - 1)
        sr = 0.1 if self.hot > 2 else (0.25 if occupied else 0.1)
        dc = 0.25 if self.hot > 2 else (0.5 if occupied else 0.25)
        off = 0.5 if self.hot > 6 else 0.0
        knobs = {"hvac_setpoint": sp, "sample_rate_hz": sr, "duty_cycle": dc,
                 "offload_frac": off}
        a = []
        for k in ACTION_KEYS:
            if k in knobs:
                lv = np.asarray(ACTION_LEVELS[k])
                a.append(int(np.argmin(np.abs(lv - knobs[k]))))
            elif k == "isolate":
                a.append(1 if kpis.get("suspicion", 0.0) > 0.75 else 0)
            else:
                a.append(0)
        self.t += 1
        return a, {"queried": False}

    def observe(self, *a, **kw):
        pass

    def summary(self):
        return {}


class SB3Agent:
    """Wraps a trained stable-baselines3 policy in the same interface."""

    def __init__(self, model, deterministic: bool = True):
        self.model, self.det = model, deterministic
        self._obs = None

    def reset(self):
        self._obs = None

    def act(self, env, kpis):
        obs = env._obs()
        a, _ = self.model.predict(obs, deterministic=self.det)
        return list(np.asarray(a).ravel().astype(int)), {"queried": bool(a[-1])}

    def observe(self, *a, **kw):
        pass

    def summary(self):
        return {}

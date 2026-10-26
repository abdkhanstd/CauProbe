"""Counter-Dyna style data-efficient model-based RL.

Re-implementation of the idea in "Counter-Dyna: Data-Efficient RL-Based HVAC
Control using Counterfactual Building Models" (Ruiz de Vargas, Raisch, Nagy,
Pinson & Goebel, 2026, arXiv:2605.04555). The insight there is that a Dyna
surrogate should not waste capacity predicting the parts of the state the
controller cannot affect -- weather, prices, occupancy -- because those are
invariant to the action and can simply be replayed. The surrogate therefore
models only the controllable sub-state, conditioned on replayed exogenous
inputs, and PPO is trained on a mixture of real and surrogate rollouts.

Here the exogenous channels are outdoor temperature and occupancy, both of
which come from real traces, so the same trick applies directly.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn

from ..config import VARS
from ..env import ACTION_LEVELS, EnvOptions, IoTGatewayEnv

ACTION_KEYS = list(ACTION_LEVELS)
EXOGENOUS = ("outdoor_temp", "occupancy")


class CounterfactualSurrogate(nn.Module):
    """Predicts the *controllable* next state and reward from (s, a, exo)."""

    def __init__(self, d_obs: int, d_act: int, d_ctrl: int, hidden: int = 256):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(d_obs + d_act + 2, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU())
        self.head_s = nn.Linear(hidden, d_ctrl)
        self.head_r = nn.Linear(hidden, 1)

    def forward(self, s, a, exo):
        h = self.body(torch.cat([s, a, exo], dim=-1))
        return self.head_s(h), self.head_r(h).squeeze(-1)


def collect_real(env_options: EnvOptions, n_episodes: int, seed: int = 0,
                 policy=None) -> Dict[str, np.ndarray]:
    S, A, S2, R, E = [], [], [], [], []
    for i in range(n_episodes):
        o = EnvOptions(**{**env_options.__dict__, "seed": seed + i})
        env = IoTGatewayEnv(o)
        obs, _ = env.reset(seed=seed + i)
        env.action_space.seed(seed + i)
        for t in range(env.T):
            a = (env.action_space.sample() if policy is None
                 else policy.predict(obs, deterministic=False)[0])
            a = np.asarray(a).ravel().astype(int)
            nxt, r, done, _, info = env.step(a)
            S.append(obs)
            A.append(_onehot(a))
            S2.append(nxt)
            R.append(r)
            E.append([info.get("outdoor_temp", 0.0), info.get("occupancy", 0.0)])
            obs = nxt
            if done:
                break
    return {k: np.asarray(v, np.float32) for k, v in
            dict(S=S, A=A, S2=S2, R=R, E=E).items()}


def _onehot(a):
    out = []
    for i, k in enumerate(ACTION_KEYS):
        n = len(ACTION_LEVELS[k])
        v = np.zeros(n, np.float32)
        v[int(a[i])] = 1.0
        out.append(v)
    return np.concatenate(out)


def fit_surrogate(data: Dict[str, np.ndarray], ctrl_idx: List[int],
                  epochs: int = 260, device: str = "cpu", seed: int = 0):
    torch.manual_seed(seed)
    S, A, S2, R, E = (data[k] for k in ("S", "A", "S2", "R", "E"))
    mu, sd = S.mean(0), S.std(0) + 1e-6
    emu, esd = E.mean(0), E.std(0) + 1e-6
    rmu, rsd = R.mean(), R.std() + 1e-6
    Zs, Zs2 = (S - mu) / sd, (S2 - mu) / sd
    Ze = (E - emu) / esd
    Zr = (R - rmu) / rsd
    m = CounterfactualSurrogate(S.shape[1], A.shape[1], len(ctrl_idx)).to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=2e-3, weight_decay=1e-5)
    t = lambda x: torch.from_numpy(np.asarray(x, np.float32)).to(device)
    Zs_t, A_t, Ze_t = t(Zs), t(A), t(Ze)
    Y_t, R_t = t(Zs2[:, ctrl_idx]), t(Zr)
    n = len(Zs_t)
    for ep in range(epochs):
        perm = torch.randperm(n, device=device)
        for i in range(0, n, 1024):
            idx = perm[i:i + 1024]
            opt.zero_grad()
            ps, pr = m(Zs_t[idx], A_t[idx], Ze_t[idx])
            loss = (nn.functional.mse_loss(ps, Y_t[idx])
                    + 0.5 * nn.functional.mse_loss(pr, R_t[idx]))
            loss.backward()
            opt.step()
    stats = {"mu": mu, "sd": sd, "emu": emu, "esd": esd, "rmu": float(rmu),
             "rsd": float(rsd), "ctrl_idx": ctrl_idx}
    with torch.no_grad():
        ps, pr = m(Zs_t, A_t, Ze_t)
        stats["state_mse"] = float(nn.functional.mse_loss(ps, Y_t))
        stats["reward_mse"] = float(nn.functional.mse_loss(pr, R_t))
    return m.eval(), stats


class SurrogateEnv(gym.Env):
    """Surrogate environment that replays the real exogenous inputs.

    Only the controllable part of the state is predicted; weather and occupancy
    are read straight from the logged trajectories, which is the invariance the
    Counter-Dyna construction exploits.
    """
    metadata = {"render_modes": []}

    def __init__(self, model, stats, real_obs: np.ndarray, real_exo: np.ndarray,
                 action_space, observation_space, horizon: int = 48,
                 device: str = "cpu", seed: int = 0):
        super().__init__()
        self.m, self.st = model, stats
        self.real_obs, self.real_exo = real_obs, real_exo
        self.action_space = action_space
        self.observation_space = observation_space
        self.horizon, self.device = horizon, device
        self.rng = np.random.default_rng(seed)
        self.metadata = {}
        self.spec = None
        self.render_mode = None

    def reset(self, *, seed=None, options=None):
        i = self.rng.integers(0, len(self.real_obs) - self.horizon - 1)
        self.i0 = int(i)
        self.k = 0
        self.s = self.real_obs[self.i0].copy()
        return self.s.astype(np.float32), {}

    def step(self, a):
        st = self.st
        zs = (self.s - st["mu"]) / st["sd"]
        exo = self.real_exo[self.i0 + self.k]
        ze = (exo - st["emu"]) / st["esd"]
        with torch.no_grad():
            ps, pr = self.m(
                torch.tensor(zs[None], dtype=torch.float32),
                torch.tensor(_onehot(np.asarray(a).ravel().astype(int))[None]),
                torch.tensor(ze[None], dtype=torch.float32))
        nxt = ((self.s - st["mu"]) / st["sd"]).copy()
        nxt[st["ctrl_idx"]] = np.clip(ps[0].numpy(), -8, 8)
        s2 = nxt * st["sd"] + st["mu"]
        r = float(pr[0].numpy()) * st["rsd"] + st["rmu"]
        self.s = s2
        self.k += 1
        return s2.astype(np.float32), r, self.k >= self.horizon, False, {}

    def close(self):
        pass


def train_counter_dyna(env_options: EnvOptions, real_episodes: int = 35,
                       sur_steps: int = 200_000, real_steps: int = 60_000,
                       seed: int = 0, device: str = "cpu",
                       save_path: Optional[Path] = None, verbose: int = 0):
    """Fit the counterfactual surrogate, pretrain PPO on it, fine-tune on the
    real environment. `real_episodes` is the interaction budget the method is
    allowed, which is the quantity the original paper minimises."""
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    probe = IoTGatewayEnv(env_options)
    obs_names = probe._obs_names
    ctrl_idx = [i for i, n in enumerate(obs_names) if n not in EXOGENOUS]

    data = collect_real(env_options, real_episodes, seed=seed)
    m, stats = fit_surrogate(data, ctrl_idx, device=device, seed=seed)

    venv = DummyVecEnv([lambda i=i: SurrogateEnv(
        m, stats, data["S"], data["E"], probe.action_space,
        probe.observation_space, seed=seed * 100 + i) for i in range(8)])
    venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=15.0)
    model = PPO("MlpPolicy", venv, seed=seed, verbose=verbose, device=device,
                n_steps=256, batch_size=512, n_epochs=8, learning_rate=3e-4,
                gamma=0.985, policy_kwargs=dict(net_arch=[128, 128]))
    model.learn(total_timesteps=sur_steps, progress_bar=False)

    from .sb3 import _make_env
    rvenv = DummyVecEnv([_make_env(env_options, seed * 1000 + i)
                         for i in range(8)])
    rvenv = VecNormalize(rvenv, norm_obs=True, norm_reward=True, clip_obs=15.0)
    model.set_env(rvenv)
    model.learn(total_timesteps=real_steps, progress_bar=False,
                reset_num_timesteps=False)
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(save_path))
        rvenv.save(str(save_path) + "_vecnorm.pkl")
    return model, rvenv, stats


class CounterDynaAgent:
    def __init__(self, model, vecnorm=None):
        from .sb3 import SB3NormAgent
        self.inner = SB3NormAgent(model, vecnorm)

    def reset(self, full: bool = False):
        self.inner.reset()

    def act(self, env, kpis):
        return self.inner.act(env, kpis)

    def observe(self, *a, **kw):
        pass

    def summary(self):
        return {}

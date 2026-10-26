"""Model-free deep RL baselines (PPO, A2C, recurrent-free) via stable-baselines3."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np

from ..config import WEIGHTS
from ..env import EnvOptions, IoTGatewayEnv

SB3_ALGOS = ("ppo", "a2c")


def _make_env(opts: EnvOptions, seed: int):
    def _f():
        o = EnvOptions(**{**opts.__dict__, "seed": seed})
        env = IoTGatewayEnv(o)
        env.reset(seed=seed)
        return env
    return _f


def train_sb3(algo: str = "ppo", total_steps: int = 300_000,
              env_options: Optional[EnvOptions] = None, n_envs: int = 8,
              seed: int = 0, save_path: Optional[Path] = None,
              device: str = "cpu", verbose: int = 0, **kw):
    from stable_baselines3 import A2C, PPO
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    opts = env_options or EnvOptions()
    venv = DummyVecEnv([_make_env(opts, seed * 1000 + i) for i in range(n_envs)])
    venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=15.0)

    common = dict(policy="MlpPolicy", env=venv, seed=seed, verbose=verbose,
                  device=device)
    if algo == "ppo":
        model = PPO(**common, n_steps=512, batch_size=512, n_epochs=8,
                    learning_rate=3e-4, gamma=0.985, gae_lambda=0.95,
                    ent_coef=0.005, clip_range=0.2,
                    policy_kwargs=dict(net_arch=[128, 128]), **kw)
    elif algo == "a2c":
        model = A2C(**common, n_steps=32, learning_rate=7e-4, gamma=0.985,
                    ent_coef=0.01, policy_kwargs=dict(net_arch=[128, 128]), **kw)
    else:
        raise ValueError(algo)
    model.learn(total_timesteps=total_steps, progress_bar=False)
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(save_path))
        venv.save(str(save_path) + "_vecnorm.pkl")
    return model, venv


class SB3NormAgent:
    """SB3 policy plus the observation normaliser it was trained with."""

    def __init__(self, model, vecnorm=None, deterministic: bool = True):
        self.model, self.vn, self.det = model, vecnorm, deterministic

    def reset(self, full: bool = False):
        pass

    def act(self, env, kpis):
        obs = env._obs()
        if self.vn is not None:
            obs = np.clip((obs - self.vn.obs_rms.mean)
                          / np.sqrt(self.vn.obs_rms.var + 1e-8), -15, 15)
        a, _ = self.model.predict(obs.astype(np.float32), deterministic=self.det)
        a = list(np.asarray(a).ravel().astype(int))
        return a, {"queried": bool(a[-1])}

    def observe(self, *a, **kw):
        pass

    def summary(self):
        return {}

"""Small shared helpers: seeding, JSON/NPZ IO, timing, energy metering."""
from __future__ import annotations

import json
import os
import random
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict

import numpy as np


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def save_json(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=_default)


def _default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    return str(o)


def load_json(path: str | Path) -> Any:
    with open(path) as f:
        return json.load(f)


@contextmanager
def timer(store: Dict[str, float], key: str):
    t0 = time.perf_counter()
    yield
    store[key] = store.get(key, 0.0) + (time.perf_counter() - t0)


class EnergyMeter:
    """Wall-clock + NVML joule counter around a block of GPU work.

    NVML exposes a monotonic millijoule counter on modern NVIDIA parts, so the
    energy figures reported for the SLM are measured, not modelled. When the
    counter is unavailable (CPU-only run) we fall back to
    `elapsed_seconds * assumed_power_w` and flag the result as estimated.
    """

    def __init__(self, device_index: int = 0, cpu_power_w: float = 15.0):
        self.device_index = device_index
        self.cpu_power_w = cpu_power_w
        self.handle = None
        self.mode = "estimated"
        try:
            import pynvml
            pynvml.nvmlInit()
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
            pynvml.nvmlDeviceGetTotalEnergyConsumption(self.handle)
            self._nvml = pynvml
            self.mode = "measured_nvml"
        except Exception:
            self.handle = None

    def _energy_mj(self) -> float | None:
        if self.handle is None:
            return None
        try:
            return float(self._nvml.nvmlDeviceGetTotalEnergyConsumption(self.handle))
        except Exception:
            return None

    @contextmanager
    def measure(self):
        res: Dict[str, float] = {}
        e0 = self._energy_mj()
        t0 = time.perf_counter()
        try:
            yield res
        finally:
            dt = time.perf_counter() - t0
            e1 = self._energy_mj()
            res["latency_ms"] = dt * 1e3
            if e0 is not None and e1 is not None:
                res["energy_j"] = (e1 - e0) / 1e3
                res["mode"] = "measured_nvml"
            else:
                res["energy_j"] = dt * self.cpu_power_w
                res["mode"] = "estimated_cpu"


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60, 60)))


def zscore(x: np.ndarray, axis: int = 0, eps: float = 1e-8):
    mu = x.mean(axis=axis, keepdims=True)
    sd = x.std(axis=axis, keepdims=True) + eps
    return (x - mu) / sd, mu, sd

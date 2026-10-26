"""BOPTEST bestest_air as an independent thermal plant.

BOPTEST (Blum et al., J. Build. Perf. Sim. 2021) emulates buildings with
Modelica models compiled to FMUs. We run its `TestCase` class in-process
(through PyFMI, the same code the BOPTEST web service runs) on the
`bestest_air` case: a single 48 m2 zone served by a fan-coil unit, with its own
weather, occupancy and internal-gain schedules. None of the agent's 2R2C
hypotheses is the true model of this plant, which is the point: it tests the
thermal side of the method under misspecification.

The adapter exposes the same observation fields the agent uses on the
simulated gateway (air temperature, HVAC electrical power, outdoor temperature,
hour) and accepts the cooling setpoint the agent's schedule asks for.
"""
from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from . import mech
from .mech import DT, STEPS_PER_DAY

BOPTEST_ROOT = Path(os.environ.get(
    "BOPTEST_ROOT", "/mnt/d/Experiments/third_party/dl/boptest"))


def _workdir() -> Path:
    """A private working directory per process. BOPTEST and PyFMI write their
    result files into the working directory (the constructor already runs a
    warm-up simulation), so parallel processes must not share one."""
    global _WD
    if _WD is not None:
        return _WD
    import uuid
    d = Path(f"/tmp/boptest_{os.getpid()}_{uuid.uuid4().hex[:8]}")
    if not d.exists():
        d.mkdir(parents=True)
        for item in BOPTEST_ROOT.iterdir():
            (d / item.name).symlink_to(item)
    _WD = d
    return d


_WD = None


@contextmanager
def _cwd(p):
    old = os.getcwd()
    os.chdir(_workdir() if Path(p) == BOPTEST_ROOT else p)
    try:
        yield
    finally:
        os.chdir(old)


class _MemFMU:
    """Proxy around a PyFMI model whose default simulation options keep
    results in memory. BOPTEST's constructor runs a warm-up simulation and
    immediately reads it back from a result file; under heavy CPU load PyFMI's
    file-consistency check can fire on that file, so file results are avoided
    from the very first simulation."""

    def __init__(self, fmu):
        self._fmu = fmu

    def simulate_options(self):
        opts = self._fmu.simulate_options()
        opts["result_handling"] = "memory"
        return opts

    def __getattr__(self, k):
        return getattr(self._fmu, k)


def load_testcase(name: str = "bestest_air"):
    if str(BOPTEST_ROOT) not in sys.path:
        sys.path.insert(0, str(BOPTEST_ROOT))
    with _cwd(BOPTEST_ROOT):
        import pyfmi
        import testcase as _tcmod
        _tcmod.load_fmu = lambda path, *a, **k: _MemFMU(pyfmi.load_fmu(path, *a, **k))
        tc = _tcmod.TestCase(fmupath=f"testcases/{name}/models/wrapped.fmu")
    # Keep PyFMI's results in memory: by default every process would write
    # the same result file in the shared working directory.
    tc.options["result_handling"] = "memory"
    return tc


class BoptestThermalPlant:
    """Day-by-day control of bestest_air through cooling-setpoint overrides.

    Observation per 5-minute step: zone air temperature, electrical HVAC power
    (cooling coil + fan, kW), outdoor dry-bulb temperature, hour of day.
    Reward per step uses the same tariff and comfort weights as the gateway
    plant, with BOPTEST's own occupancy schedule for the comfort weight.
    """

    def __init__(self, start_day: int = 153, warmup_days: int = 1,
                 heat_sp: float = 15.0, scenario_price: str = "highly_dynamic"):
        self.tc = load_testcase("bestest_air")
        with _cwd(BOPTEST_ROOT):
            self.tc.set_step(DT)
            self.tc.initialize(start_day * 86400.0, warmup_days * 86400.0)
        self.heat_sp = heat_sp + 273.15
        self.t = 0
        self.last = self._meas()

    def _meas(self):
        m = self.tc.get_measurements() if False else None
        return m

    def step(self, cool_sp_c: float):
        u = {"con_oveTSetCoo_u": cool_sp_c + 273.15,
             "con_oveTSetCoo_activate": 1,
             "con_oveTSetHea_u": self.heat_sp,
             "con_oveTSetHea_activate": 1}
        with _cwd(BOPTEST_ROOT):
            y = self.tc.advance(u)
        t_air = float(y["zon_reaTRooAir_y"]) - 273.15
        p_kw = (float(y["fcu_reaPCoo_y"]) + float(y["fcu_reaPFan_y"])) / 1000.0
        t_out = float(y.get("zon_weaSta_reaWeaTDryBul_y", np.nan)) - 273.15
        occ = float(y.get("zon_reaOcc_y", np.nan)) if "zon_reaOcc_y" in y else np.nan
        k = self.t % STEPS_PER_DAY
        h = k / 12.0
        self.t += 1
        return {"t_air": t_air, "hvac_kw": p_kw, "t_out": t_out, "hour": h,
                "raw": y, "occ": occ}

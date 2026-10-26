"""One day of bestest_air under the incumbent schedule: hourly temperature and power."""
import logging

import numpy as np

from cp_common import ROOT  # noqa: F401
from causlm.probe.boptest_plant import BOPTEST_ROOT, _cwd, load_testcase

logging.disable(logging.INFO)
tc = load_testcase("bestest_air")


def P(r):
    return r[2] if isinstance(r, tuple) else r


with _cwd(BOPTEST_ROOT):
    print(P(tc.set_scenario({"electricity_price": "dynamic"})) and "scenario ok")
    tc.set_step(300.0)
    tc.initialize(190 * 86400.0, 86400.0)
    for k in range(288):
        h = k / 12.0
        sp = 24.0 if 7 <= h < 19 else 28.0
        y = P(tc.advance({"con_oveTSetCoo_u": sp + 273.15,
                          "con_oveTSetCoo_activate": 1}))
        if k % 12 == 0:
            print(f"h={h:4.1f} sp={sp} Tz={y['zon_reaTRooAir_y']-273.15:5.2f} "
                  f"Tout={y['zon_weaSta_reaWeaTDryBul_y']-273.15:5.1f} "
                  f"Pcoo={y['fcu_reaPCoo_y']:7.1f}W Pfan={y['fcu_reaPFan_y']:6.1f}W "
                  f"Phea={y['fcu_reaPHea_y']:7.1f}W flow={y['fcu_reaFloSup_y']:.3f}")
    print({k: v for k, v in P(tc.get_kpis()).items()})

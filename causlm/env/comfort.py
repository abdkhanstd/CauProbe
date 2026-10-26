"""Fanger PMV / PPD thermal comfort (ISO 7730 / ASHRAE 55).

Implemented directly from the standard rather than approximated, so the comfort
column in the results tables is a recognised index and not an ad-hoc
temperature deviation.
"""
from __future__ import annotations

import math


def pmv_ppd(ta: float, tr: float | None = None, vel: float = 0.1,
            rh: float = 50.0, met: float = 1.2, clo: float = 0.7,
            wme: float = 0.0):
    """Predicted Mean Vote and Predicted Percentage Dissatisfied.

    Parameters
    ----------
    ta  : air temperature, C
    tr  : mean radiant temperature, C (defaults to ta)
    vel : relative air velocity, m/s
    rh  : relative humidity, %
    met : metabolic rate, met
    clo : clothing insulation, clo
    """
    if tr is None:
        tr = ta
    pa = rh * 10.0 * math.exp(16.6536 - 4030.183 / (ta + 235.0))
    icl = 0.155 * clo
    m = met * 58.15
    w = wme * 58.15
    mw = m - w

    fcl = 1.0 + 1.29 * icl if icl <= 0.078 else 1.05 + 0.645 * icl
    hcf = 12.1 * math.sqrt(max(vel, 0.0))
    taa = ta + 273.0
    tra = tr + 273.0

    # first guess for the clothing surface temperature
    tcla = taa + (35.5 - ta) / (3.5 * icl + 0.1)
    p1 = icl * fcl
    p2 = p1 * 3.96
    p3 = p1 * 100.0
    p4 = p1 * taa
    p5 = 308.7 - 0.028 * mw + p2 * (tra / 100.0) ** 4
    xn = tcla / 100.0
    xf = xn
    eps = 1e-5
    for _ in range(150):
        xf = (xf + xn) / 2.0
        hcn = 2.38 * abs(100.0 * xf - taa) ** 0.25
        hc = max(hcf, hcn)
        xn = (p5 + p4 * hc - p2 * xf ** 4) / (100.0 + p3 * hc)
        if abs(xn - xf) <= eps:
            break
    tcl = 100.0 * xn - 273.0

    hl1 = 3.05e-3 * (5733.0 - 6.99 * mw - pa)               # skin diffusion
    hl2 = 0.42 * (mw - 58.15) if mw > 58.15 else 0.0        # sweating
    hl3 = 1.7e-5 * m * (5867.0 - pa)                        # latent respiration
    hl4 = 0.0014 * m * (34.0 - ta)                          # dry respiration
    hl5 = 3.96 * fcl * (xn ** 4 - (tra / 100.0) ** 4)       # radiation
    hl6 = fcl * max(hcf, 2.38 * abs(100.0 * xn - taa) ** 0.25) * (tcl - ta)

    ts = 0.303 * math.exp(-0.036 * m) + 0.028
    pmv = ts * (mw - hl1 - hl2 - hl3 - hl4 - hl5 - hl6)
    pmv = max(-3.5, min(3.5, pmv))
    ppd = 100.0 - 95.0 * math.exp(-0.03353 * pmv ** 4 - 0.2179 * pmv ** 2)
    return pmv, ppd

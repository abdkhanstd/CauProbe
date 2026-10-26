"""Power calculation for the powered replicated probe audit.

From the first replicated audit (cp_audit3.py: 48 decisions x 6 replicates):
the intraclass correlation (share of a single draw's variance between
decisions) fixes the reliability of an R-replicate mean by Spearman-Brown,
rel(R) = R icc / (1 + (R - 1) icc), and so the number of replicates needed for
rel >= 0.7. A simulation then gives, per design (decisions n, replicates R),
the spread of the observed Spearman correlation between a decision-time score
and the decisions' R-replicate means, and the power to exclude zero, when the
score correlates rho in {0, 0.2, 0.3} with the decisions' true expected value.
Expected values are normal; the noise is resampled from the within-decision
residuals of the 48 x 6 run (heavy tails kept) and scaled to the ICC.
Clustering of decisions within plants is ignored (power slightly optimistic).
Writes results/cauprobe/test2/probe_audit3_power.json.
"""
from __future__ import annotations

import json

import numpy as np
from scipy.stats import spearmanr

from cp_common import CP_RESULTS

P3 = CP_RESULTS / "test2" / "probe_audit3.jsonl"
OUT = CP_RESULTS / "test2" / "probe_audit3_power.json"
DESIGNS = [(48, 6), (48, 36), (96, 36), (157, 24), (157, 30), (157, 36)]


def rel(R, icc):
    return R * icc / (1 + (R - 1) * icc)


def main():
    V = np.array([json.loads(l)["realised_reps"] for l in P3.read_text().splitlines()
                  if l.strip()])
    n, R = V.shape
    M = V.mean(1)
    msw = V.var(1, ddof=1).mean()
    icc = max(0.0, (R * M.var(ddof=1) - msw) / R) / (max(0.0, (R * M.var(ddof=1) - msw) / R) + msw)
    rk = np.argsort(np.argsort(V.ravel())).reshape(V.shape).astype(float)
    w_r = rk.var(1, ddof=1).mean()
    b_r = max(0.0, (R * rk.mean(1).var(ddof=1) - w_r) / R)
    icc_r = b_r / (b_r + w_r)
    iccs = {"ci_low_0.03": 0.03, "anova": icc, "rank": icc_r, "ci_high_0.134": 0.134}
    out = {"source": f"{n} decisions x {R} replicates", "icc": iccs,
           "reliability": {k: {str(r): rel(r, c) for r in (6, 12, 24, 30, 36, 48)}
                           for k, c in iccs.items()},
           "reps_for_reliability_0.7": {k: 0.7 * (1 - c) / (c * 0.3) for k, c in iccs.items()},
           "simulation": {}}
    rng = np.random.default_rng(1)
    res = (V - M[:, None]).ravel()
    res = res[np.abs(res) > 0]
    res = res / res.std()
    for name in ("anova", "rank"):
        c = iccs[name]
        for nd, Rr in DESIGNS:
            cell = {"reliability": rel(Rr, c)}
            for rho in (0.0, 0.2, 0.3):
                rs = []
                for _ in range(2000):
                    mu = rng.standard_normal(nd)
                    s = rho * mu + np.sqrt(1 - rho ** 2) * rng.standard_normal(nd)
                    noise = rng.choice(res, size=(nd, Rr)) * np.sqrt((1 - c) / c)
                    rs.append(spearmanr(s, mu + noise.mean(1)).correlation)
                rs = np.array(rs)
                se = 1.06 / np.sqrt(nd - 3)        # Fieller et al. for Spearman
                cell[f"rho={rho}"] = {
                    "median": float(np.median(rs)),
                    "q025": float(np.percentile(rs, 2.5)),
                    "q975": float(np.percentile(rs, 97.5)),
                    "power_excl_0": float(np.mean(np.arctanh(rs) - 1.96 * se > 0))}
            out["simulation"][f"{name}:n={nd},R={Rr}"] = cell
            print(name, nd, Rr, json.dumps(cell), flush=True)
    OUT.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()

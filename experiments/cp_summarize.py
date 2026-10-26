"""Summaries of commissioning runs: regret against the oracle, paired CIs.

Regret is computed per (plant, run) against the oracle on the identical plant
and noise stream, so it is a paired quantity. Confidence intervals are
percentile bootstraps over plants (runs averaged within plant first, since
runs on one plant are not independent). Pairwise tests against CauProbe use
the paired bootstrap of per-plant differences with Holm correction.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict

import numpy as np

from cp_common import CP_RESULTS


def load(tag):
    rows = [json.loads(l) for l in (CP_RESULTS / f"e1_{tag}.jsonl").read_text().splitlines()]
    return rows


def boot_ci(x, n=10000, seed=0):
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    if len(x) < 2:
        return float(x.mean()), float("nan"), float("nan")
    b = rng.choice(x, (n, len(x))).mean(1)
    return float(x.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def paired_p(d, n=20000, seed=0):
    d = np.asarray(d, float)
    rng = np.random.default_rng(seed)
    b = rng.choice(d, (n, len(d))).mean(1)
    p = 2 * min((b <= 0).mean(), (b >= 0).mean())
    return float(max(p, 1.0 / n))


def holm(ps):
    ps = np.asarray(ps)
    order = np.argsort(ps)
    adj = np.empty_like(ps)
    m = len(ps)
    run = 0.0
    for i, k in enumerate(order):
        run = max(run, (m - i) * ps[k])
        adj[k] = min(1.0, run)
    return adj


def table(rows, ref="cauprobe", days=None):
    by = defaultdict(dict)       # method -> (plant, run) -> row
    for r in rows:
        by[r["method"]][(r["plant"], r["seed"])] = r
    if "oracle" not in by:
        raise SystemExit("no oracle rows")
    orc = by["oracle"]
    out = {}
    for m, d in by.items():
        keys = sorted(set(d) & set(orc))
        if not keys:
            continue
        per_plant = defaultdict(list)
        for k in keys:
            rr, oo = d[k], orc[k]
            if days:
                reg = sum(oo["return_per_day"][:days]) - sum(rr["return_per_day"][:days])
            else:
                reg = oo["return"] - rr["return"]
            per_plant[k[1] // 10].append({
                "regret": reg,
                "reg_net": oo["r_net"] - rr["r_net"],
                "reg_th": oo["r_th"] - rr["r_th"],
                "viol": rr["viol_net"] + rr["viol_th"],
                "probe": rr["probe_steps"], "fb": rr["fallback_steps"],
                "asks": len(rr["asks"]), "thp": rr["th_probes"],
                "day1": oo["return_per_day"][0] - rr["return_per_day"][0]})
        agg = {k2: np.array([np.mean([x[k2] for x in v]) for v in per_plant.values()])
               for k2 in per_plant[next(iter(per_plant))][0]}
        out[m] = {k2: boot_ci(v) for k2, v in agg.items()}
        out[m]["n_plants"] = len(per_plant)
        out[m]["_per_plant_regret"] = dict(zip(per_plant.keys(), agg["regret"]))
    if ref in out:
        refp = out[ref]["_per_plant_regret"]
        ms = [m for m in out if m != ref]
        ps, ds = [], []
        for m in ms:
            pp = out[m]["_per_plant_regret"]
            common = sorted(set(pp) & set(refp))
            d = np.array([pp[k] - refp[k] for k in common])
            ds.append(boot_ci(d))
            ps.append(paired_p(d) if len(d) > 1 else 1.0)
        for m, d, p, pa in zip(ms, ds, ps, holm(ps)):
            out[m]["vs_ref"] = {"diff": d, "p": p, "p_holm": float(pa)}
    return out


def show(out, ref="cauprobe"):
    order = sorted(out, key=lambda m: out[m]["regret"][0])
    print(f"{'method':24s} {'regret [95% CI]':>28s} {'net':>8s} {'th':>7s} "
          f"{'viol':>6s} {'probe':>6s} {'fb':>5s} {'asks':>4s} {'day1':>7s}  vs-{ref}")
    for m in order:
        o = out[m]
        r = o["regret"]
        vs = o.get("vs_ref")
        vs_s = (f"{vs['diff'][0]:+8.1f} p={vs['p_holm']:.3g}" if vs else "")
        print(f"{m:24s} {r[0]:9.1f} [{r[1]:8.1f},{r[2]:8.1f}] "
              f"{o['reg_net'][0]:8.1f} {o['reg_th'][0]:7.2f} {o['viol'][0]:6.1f} "
              f"{o['probe'][0]:6.1f} {o['fb'][0]:5.0f} {o['asks'][0]:4.1f} "
              f"{o['day1'][0]:7.1f}  {vs_s}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("tag")
    ap.add_argument("--ref", default="cauprobe")
    ap.add_argument("--days", type=int, default=None)
    a = ap.parse_args()
    out = table(load(a.tag), ref=a.ref, days=a.days)
    show(out, a.ref)

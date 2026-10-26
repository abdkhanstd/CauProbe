"""Paired comparison of the second iteration with its rivals (dev/validation).

Reads any number of E1-style JSONL files (rows of the same plants and noise
streams), and for every method reports the per-plant mean return difference
to greedy control and to expert+greedy (CauProbe without probing), with
bootstrap CIs and paired p-values over plants, plus probe steps, asks,
reconfiguration cost and the wrong-mechanism rate. Used only on the dev and
validation plants while the second iteration was developed.

  cp_v2_compare.py results/cauprobe/validation_v2/e1_nominal.jsonl [...]
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict

import numpy as np

from cp_common import TEST2_SEEDS_GUARD, TEST_SEEDS
from cp_summarize import boot_ci, paired_p


def wrong(r):
    fp = r.get("final_factor_post")
    if not fp:
        return None
    c = list(r["combo"])
    if r.get("scenario") == "shift":
        c[0] = 1 - c[0]
    return any((fp[F] > 0.5) != bool(c[i]) for i, F in enumerate("ABC"))


def main(paths, refs=("greedy", "cauprobe-noprobe")):
    by = defaultdict(dict)
    for p in paths:
        for line in open(p):
            r = json.loads(line)
            if "error" in r:
                continue
            assert r["seed"] // 10 not in TEST_SEEDS, "test plant in dev data"
            assert r["seed"] // 10 not in TEST2_SEEDS_GUARD, "test plant in dev data"
            by[r["method"]].setdefault(r["seed"], r)
    out = {}
    for m in sorted(by):
        rr = list(by[m].values())
        st = {"n": len(rr),
              "probes": float(np.mean([r["probe_steps"] for r in rr])),
              "asks": float(np.mean([len(r["asks"]) for r in rr])),
              "switch_cost": float(np.mean([r["switch_cost"] for r in rr])),
              "wrong": float(np.mean([bool(wrong(r)) for r in rr]))}
        for ref in refs:
            if ref not in by or ref == m:
                continue
            acc = defaultdict(list)
            for s, r in by[m].items():
                if s in by[ref]:
                    acc[s // 10].append(r["return"] - by[ref][s]["return"])
            d = np.array([np.mean(v) for v in acc.values()])
            if len(d) > 1:
                st[f"vs_{ref}"] = {"ci": boot_ci(d), "p": paired_p(d),
                                   "n_plants": len(d)}
        out[m] = st
        line = f"{m:30s} n={st['n']:3d} probes {st['probes']:6.1f} asks {st['asks']:.2f} " \
               f"sw {st['switch_cost']:6.1f} wrong {st['wrong']:.3f}"
        for ref in refs:
            v = st.get(f"vs_{ref}")
            if v:
                c = v["ci"]
                line += f" | vs {ref}: {c[0]:+6.1f} [{c[1]:+.1f},{c[2]:+.1f}] p={v['p']:.3f}"
        print(line)
    return out


if __name__ == "__main__":
    main(sys.argv[1:])

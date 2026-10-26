"""Digest of the test-set results: every number the results text quotes.

Prints, for the fresh test plants: per-scenario paired differences of the
belief-based methods against CauProbe, per-structure CauProbe-minus-greedy,
lock-in rescues (runs where greedy ends with a wrong mechanism and CauProbe
does not), tail metrics, out-of-library outcomes by mechanism type, and the
structural-change scenario split by ticket. Writes results/cauprobe/digest.json.
"""
from __future__ import annotations

import json
from collections import defaultdict

import numpy as np

from cp_common import CP_RESULTS, testrows
from cp_summarize import boot_ci, paired_p

SCEN = ["main", "noisy", "lpwan", "noisy_lpwan", "poor_docs", "shift"]
BEL = ["cauprobe", "cauprobe-noask", "cauprobe-noprobe", "greedy", "thompson",
       "ids", "identify"]


def load(tag):
    p = CP_RESULTS / f"e1_test_{tag}.jsonl"
    if not p.exists():
        return []
    rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    return rows if tag == "ool" else testrows(rows)


def index(rows):
    by = defaultdict(dict)
    for r in rows:
        by[r["method"]][r["seed"]] = r
    return by


def per_plant(by, m, key):
    acc = defaultdict(list)
    for s, r in by[m].items():
        acc[s // 10].append(key(r))
    return {p: float(np.mean(v)) for p, v in acc.items()}


def wrong_mech(r):
    fp = r.get("final_factor_post")
    if not fp:
        return None
    c = list(r["combo"])
    if r.get("scenario") == "shift":
        c[0] = 1 - c[0]          # the radio was swapped on day 3: A flipped
    return any((fp[F] > 0.5) != bool(c[i]) for i, F in enumerate("ABC"))


def main():
    out = {}
    for sc in SCEN:
        rows = load(sc)
        if not rows:
            continue
        by = index(rows)
        if "oracle" not in by or "cauprobe" not in by:
            continue
        orc = by["oracle"]
        d = {}
        ref = per_plant(by, "cauprobe", lambda r: orc[r["seed"]]["return"] - r["return"])
        for m in BEL:
            if m not in by:
                continue
            g = per_plant(by, m, lambda r: orc[r["seed"]]["return"] - r["return"])
            common = sorted(set(g) & set(ref))
            diff = np.array([g[k] - ref[k] for k in common])
            arr = np.array(list(g.values()))
            q = np.sort(arr)[::-1]
            d[m] = {"gap": boot_ci(arr), "diff_vs_cauprobe": boot_ci(diff),
                    "p_vs_cauprobe": paired_p(diff) if m != "cauprobe" else None,
                    "cvar10": float(q[:max(1, len(q) // 10)].mean()),
                    "max": float(arr.max()),
                    "probes": float(np.mean([r["probe_steps"] for r in by[m].values()])),
                    "misses": float(np.mean([r["viol_net"] + r["viol_th"] for r in by[m].values()])),
                    "wrong_final": float(np.mean([bool(wrong_mech(r)) for r in by[m].values()
                                                  if wrong_mech(r) is not None] or [0]))}
        # lock-in rescues: greedy wrong, CauProbe right, same plant and noise
        resc, harm = [], []
        for s, c in by["cauprobe"].items():
            g = by["greedy"].get(s)
            if g is None:
                continue
            dd = c["return"] - g["return"]
            if wrong_mech(g) and not wrong_mech(c):
                resc.append(dd)
            if wrong_mech(c) and not wrong_mech(g):
                harm.append(dd)
        d["_lockin"] = {"n_rescue_runs": len(resc),
                        "mean_gain_rescue": float(np.mean(resc)) if resc else None,
                        "n_reverse_runs": len(harm),
                        "mean_gain_reverse": float(np.mean(harm)) if harm else None,
                        "n_runs": len(by["cauprobe"])}
        # per structure (A,B,C) mean CauProbe minus greedy return
        st = defaultdict(list)
        for s, c in by["cauprobe"].items():
            g = by["greedy"].get(s)
            if g is not None:
                st["A%dB%dC%d" % tuple(c["combo"][:3])].append(c["return"] - g["return"])
        d["_per_structure"] = {k: float(np.mean(v)) for k, v in sorted(st.items())}
        if sc == "shift":
            tk = defaultdict(lambda: defaultdict(list))
            for m in BEL:
                for s, r in by.get(m, {}).items():
                    o = orc.get(s)
                    if o is None:
                        continue
                    late = sum(o["return_per_day"][3:]) - sum(r["return_per_day"][3:])
                    tk[m]["ticket" if r.get("ticket") else "no_ticket"].append(late)
            d["_shift_after_change"] = {m: {k: float(np.mean(v)) for k, v in t.items()}
                                        for m, t in tk.items()}
        out[sc] = d
        print(f"== {sc}")
        for m in BEL:
            if m in d:
                x = d[m]
                print(f"  {m:18s} gap {x['gap'][0]:8.1f}  vsCP {x['diff_vs_cauprobe'][0]:+7.1f} "
                      f"p={x['p_vs_cauprobe'] if x['p_vs_cauprobe'] is None else round(x['p_vs_cauprobe'], 4)} "
                      f"cvar10 {x['cvar10']:7.1f} probes {x['probes']:6.1f} misses {x['misses']:5.1f} "
                      f"wrong {x['wrong_final']:.2f}")
        print("  lock-in:", d["_lockin"])
        print("  per-structure CauProbe-greedy:", {k: round(v, 1) for k, v in d["_per_structure"].items()})
        if "_shift_after_change" in d:
            print("  shift (gap after day 3):", {m: {k: round(v, 1) for k, v in t.items()}
                                                for m, t in d["_shift_after_change"].items()})
    rows = load("ool")
    if rows:
        by = index(rows)
        o = {}
        for m in BEL + ["rule", "oracle"]:
            if m not in by:
                continue
            kinds = defaultdict(list)
            for r in by[m].values():
                kinds[r["ool"]].append(r)
            o[m] = {k: {"return": float(np.mean([x["return"] for x in v])),
                        "misses": float(np.mean([x["viol_net"] for x in v])),
                        "fallback": float(np.mean([x["fallback_steps"] for x in v])),
                        "alarms": float(np.mean([x.get("misspec_alarms", 0) > 0 for x in v]))}
                    for k, v in kinds.items()}
        out["ool"] = o
        print("== ool")
        for m, k in o.items():
            print(" ", m, {kk: {a: round(b, 2) for a, b in vv.items()} for kk, vv in k.items()})
    (CP_RESULTS / "digest.json").write_text(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()

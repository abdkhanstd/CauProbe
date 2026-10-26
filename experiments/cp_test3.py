"""Third fresh test set: does a fleet-calibrated expert change commissioning?

Reads results/cauprobe/test3/ (run_cauprobe_test3_grid.sh). For each scenario:
return gained over greedy control on the same plant and noise stream for
CauProbe-2 and expert-plus-greedy control, each with the expert calibrated on
training-vocabulary notes and with the fleet-calibrated expert; the paired
effect of fleet calibration (fleet minus original, bootstrap CI over plants,
Holm over the 8 fleet comparisons); probe steps; and the share of runs ending
on a wrong network mechanism. Also the value-of-a-reading audit
(cp_voi_ask3.py). Writes results/cauprobe/test3/summary.json and
paper/tables_cp_test3.tex.
"""
from __future__ import annotations

import json
from collections import defaultdict

import numpy as np

from cp_common import CP_RESULTS, ROOT, TEST3_SEEDS
from cp_summarize import boot_ci, holm, paired_p

T3 = CP_RESULTS / "test3"
SCEN = ["nominal", "poor_docs", "lpwan", "noisy"]
LAB = {"nominal": "nominal", "poor_docs": "poor docs", "lpwan": "LPWAN",
       "noisy": "noisy"}
M = ["cauprobe-v2", "cauprobe-v2-fleet", "cauprobe-noprobe", "cauprobe-noprobe-fleet"]


def rows(sc):
    p = T3 / f"e1_{sc}.jsonl"
    if not p.exists():
        return []
    rr = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    return [r for r in rr if "error" not in r and r["seed"] // 10 in TEST3_SEEDS]


def pdiff(b, m1, m2):
    acc = defaultdict(list)
    for s, r in b[m1].items():
        if s in b[m2]:
            acc[s // 10].append(r["return"] - b[m2][s]["return"])
    d = np.array([np.mean(v) for v in acc.values()])
    return {"ci": boot_ci(d), "p": paired_p(d), "n": len(d)}


def wrong(r):
    fp = r.get("final_factor_post")
    if not fp:
        return None
    return any((fp[F] > 0.5) != bool(r["combo"][i]) for i, F in enumerate("ABC"))


def main():
    S, keys, ps = {}, [], []
    for sc in SCEN:
        rr = rows(sc)
        if not rr:
            continue
        b = defaultdict(dict)
        for r in rr:
            b[r["method"]][r["seed"]] = r
        d = {"n_runs": {m: len(b[m]) for m in b}}
        for m in M:
            if m in b and "greedy" in b:
                d[f"{m}|greedy"] = pdiff(b, m, "greedy")
                d[f"probes:{m}"] = float(np.mean([r["probe_steps"] for r in b[m].values()]))
                w = [wrong(r) for r in b[m].values() if wrong(r) is not None]
                d[f"wrong:{m}"] = float(np.mean(w)) if w else None
        w = [wrong(r) for r in b["greedy"].values() if wrong(r) is not None]
        d["wrong:greedy"] = float(np.mean(w)) if w else None
        for base in ("cauprobe-v2", "cauprobe-noprobe"):
            if base in b and base + "-fleet" in b:
                k = f"{base}-fleet|{base}"
                d[k] = pdiff(b, base + "-fleet", base)
                keys.append((sc, k))
                ps.append(d[k]["p"])
        if "cauprobe-v2" in b and "cauprobe-noprobe" in b:
            d["cauprobe-v2|cauprobe-noprobe"] = pdiff(b, "cauprobe-v2", "cauprobe-noprobe")
        S[sc] = d
    for (sc, k), ph in zip(keys, holm(ps) if ps else []):
        S[sc][k]["p_holm"] = float(ph)
    # the replication of the second iteration on fresh plants: its two key
    # comparisons, Holm-corrected across the scenarios of this set
    for k in ("cauprobe-v2|greedy", "cauprobe-v2|cauprobe-noprobe"):
        ks = [sc for sc in S if k in S[sc]]
        for sc, ph in zip(ks, holm([S[sc][k]["p"] for sc in ks])):
            S[sc][k]["p_holm"] = float(ph)
    va = T3 / "voi_ask.json"
    V = json.loads(va.read_text()) if va.exists() else None
    out = {"scen": S, "voi_ask": V}
    (T3 / "summary.json").write_text(json.dumps(out, indent=1, default=float))
    for sc, d in S.items():
        print("==", sc, d["n_runs"])
        for k, v in d.items():
            if isinstance(v, dict) and "ci" in v:
                print(f"  {k:40s} {v['ci'][0]:+7.1f} [{v['ci'][1]:+.1f}, {v['ci'][2]:+.1f}] "
                      f"p={v['p']:.3f} holm={v.get('p_holm')}")
        print("  probes", {m: round(d.get(f'probes:{m}', -1), 1) for m in M})
        print("  wrong", {m: d.get(f'wrong:{m}') for m in M + ['greedy']})
    if V:
        for sw in ("wifi", "lpwan"):
            if sw in V:
                print(sw, {k: v for k, v in V[sw].items() if k != "per_plant"})
    table(S, V)


def cell(x, nd=1):
    return f"{x:+.{nd}f}"


def table(S, V):
    cols = [sc for sc in SCEN if sc in S]
    L = [r"\begin{table*}[t]", r"\centering",
         r"\caption{Third fresh test set (48 plants, 14 days; two runs per plant "
         r"in the nominal scenario, one in the others): the same controllers with "
         r"the expert calibrated on training-vocabulary notes and with the "
         r"fleet-calibrated expert. Top: return gained over greedy control on the "
         r"same plant and noise stream. Middle: the paired effect of fleet "
         r"calibration (fleet minus original), mean [95\% bootstrap CI], "
         r"$p_{\mathrm{Holm}}$ over the eight comparisons. Bottom: probe steps per "
         r"run and runs ending on a wrong network mechanism.}",
         r"\label{tab:test3}", r"\footnotesize", r"\setlength{\tabcolsep}{3pt}",
         r"\begin{tabular}{l" + "r" * len(cols) + "}", r"\toprule",
         " & " + " & ".join(LAB[c] for c in cols) + r" \\", r"\midrule",
         r"\multicolumn{" + str(len(cols) + 1) + r"}{l}{\emph{Return gained over greedy control}} \\"]
    names = {"cauprobe-v2": r"\ours-2", "cauprobe-v2-fleet": r"\ours-2, fleet expert",
             "cauprobe-noprobe": "expert + greedy",
             "cauprobe-noprobe-fleet": "expert + greedy, fleet expert"}
    for m in M:
        L.append(names[m] + " & " + " & ".join(
            cell(S[c][f"{m}|greedy"]["ci"][0]) if f"{m}|greedy" in S[c] else "--"
            for c in cols) + r" \\")
    L.append(r"\midrule")
    L.append(r"\multicolumn{" + str(len(cols) + 1) + r"}{l}{\emph{Effect of fleet calibration}} \\")
    for base, nm in (("cauprobe-v2", r"\ours-2"), ("cauprobe-noprobe", "expert + greedy")):
        k = f"{base}-fleet|{base}"
        cells = []
        for c in cols:
            x = S[c].get(k)
            if x is None:
                cells.append("--")
                continue
            m, lo, hi = x["ci"]
            cells.append(f"{m:+.1f} [{lo:+.1f}, {hi:+.1f}] ({x['p_holm']:.2f})")
        L.append(nm + " & " + " & ".join(cells) + r" \\")
    L.append(r"\midrule")
    L.append(r"\multicolumn{" + str(len(cols) + 1) + r"}{l}{\emph{Probe steps per run; runs on a wrong mechanism}} \\")
    for m in ("cauprobe-v2", "cauprobe-v2-fleet"):
        L.append(names[m] + " & " + " & ".join(
            f"{S[c].get('probes:' + m, float('nan')):.0f}; {100 * (S[c].get('wrong:' + m) or 0):.0f}\\%"
            for c in cols) + r" \\")
    for m in ("cauprobe-noprobe", "cauprobe-noprobe-fleet", "greedy"):
        nm = names.get(m, "greedy (same belief)")
        L.append(nm + " & " + " & ".join(
            f"--; {100 * (S[c].get('wrong:' + m) or 0):.0f}\\%" for c in cols) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""]
    (ROOT / "paper" / "tables_cp_test3.tex").write_text("\n".join(L))


if __name__ == "__main__":
    main()

"""Approximate dual control by Bayes-adaptive rollouts against CauProbe-2.

The comparator ("dual-rollout", objective="rollout" in agent.py) values every
candidate probe directly by the simulated return of taking it and continuing
greedy control with Bayesian updating, against continuing greedily now. Its
settings were fixed in advance to CauProbe-2's own rollouts (3 days, 8 truths
per structure, the same candidates, probe lengths, screen, safety filter and
probe execution). It was run on the second fresh test set
(results/cauprobe/dual2) and is paired here with the stored runs of the
same plants and noise streams (results/cauprobe/test2).

  probing only   dual-rollout       vs CauProbe-2 without the expert
  with reading   dual-rollout+ask   vs CauProbe-2
  both           vs greedy control; probes, lock-in, compute per run
Holm over all comparisons against CauProbe-2. Writes
results/cauprobe/dual.json and paper/tables_cp_dual.tex.
"""
from __future__ import annotations

import json
from collections import defaultdict

import numpy as np

from cp_common import CP_RESULTS, ROOT
from cp_summarize import boot_ci, holm, paired_p

SCEN = ["nominal", "lpwan", "noisy", "noisy_lpwan", "shift"]
LAB = {"nominal": "nominal", "lpwan": "LPWAN", "noisy": "noisy",
       "noisy_lpwan": "noisy+LPWAN", "shift": r"struct.\ change"}


def load(p):
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()
            and "error" not in l]


def wrong(r, sc):
    fp = r.get("final_factor_post")
    if not fp:
        return None
    c = list(r["combo"])
    if sc == "shift":
        c[0] = 1 - c[0]
    return any((fp[F] > 0.5) != bool(c[i]) for i, F in enumerate("ABC"))


def pdiff(a, b):
    acc = defaultdict(list)
    for s, r in a.items():
        if s in b:
            acc[s // 10].append(r["return"] - b[s]["return"])
    d = np.array([np.mean(v) for v in acc.values()])
    return {"ci": boot_ci(d), "p": paired_p(d), "n": len(d)}


def main():
    out, keys, ps = {}, [], []
    for sc in SCEN:
        by = defaultdict(dict)
        for r in load(CP_RESULTS / "test2" / f"e1_{sc}.jsonl"):
            by[r["method"]][r["seed"]] = r
        for r in load(CP_RESULTS / "dual2" / f"e1_{sc}.jsonl"):
            by[r["method"]][r["seed"]] = r
        d = {}
        pairs = [("dual-rollout", "cauprobe-v2-noask"), ("dual-rollout", "greedy"),
                 ("cauprobe-v2-noask", "greedy")]
        if "dual-rollout+ask" in by:
            pairs += [("dual-rollout+ask", "cauprobe-v2"), ("dual-rollout+ask", "cauprobe-noprobe"),
                      ("cauprobe-v2", "greedy")]
        for a, b in pairs:
            if a in by and b in by:
                d[f"{a}|{b}"] = pdiff(by[a], by[b])
                if b in ("cauprobe-v2-noask", "cauprobe-v2"):
                    keys.append((sc, f"{a}|{b}"))
                    ps.append(d[f"{a}|{b}"]["p"])
        for m in ("dual-rollout", "dual-rollout+ask", "cauprobe-v2-noask", "cauprobe-v2",
                  "cauprobe-noprobe", "greedy"):
            if m not in by:
                continue
            w = [bool(wrong(r, sc)) for r in by[m].values() if wrong(r, sc) is not None]
            d[f"stats:{m}"] = {"probes": float(np.mean([r["probe_steps"] for r in by[m].values()])),
                               "wrong": float(np.mean(w)) if w else None,
                               "wall_s": float(np.mean([r["wall_s"] for r in by[m].values()])),
                               "n": len(by[m])}
        out[sc] = d
    for (sc, k), ph in zip(keys, holm(ps)):
        out[sc][k]["p_holm"] = float(ph)
    (CP_RESULTS / "dual.json").write_text(json.dumps(out, indent=1, default=float))
    for sc, d in out.items():
        print("==", sc)
        for k, v in d.items():
            if "ci" in v:
                print(f"  {k:40s} {v['ci'][0]:+7.1f} [{v['ci'][1]:+.1f}, {v['ci'][2]:+.1f}] p={v['p']:.3f} holm={v.get('p_holm')}")
            else:
                print(f"  {k:40s} {v}")
    table(out)


def table(o):
    cols = [sc for sc in SCEN if sc in o]
    L = [r"\begin{table*}[t]", r"\centering",
         r"\caption{\ours-2 against approximate dual control (``dual'') by Bayes-adaptive "
         r"rollouts on the second fresh test set (48 plants; two runs per plant in "
         r"the nominal scenario, one in the others). The comparator values each "
         r"probe directly by the simulated return of taking it and continuing to "
         r"learn, with settings fixed in advance to \ours-2's own rollouts. Paired "
         r"differences in 14-day return, mean [95\% CI]; $p_{\mathrm{Holm}}$ over "
         r"the comparisons with \ours-2. Bottom: probe steps per run, runs ending "
         r"on a wrong mechanism, and compute per two-week run on one host core.}",
         r"\label{tab:dual}", r"\footnotesize", r"\setlength{\tabcolsep}{2.5pt}",
         r"\begin{tabular}{l" + "r" * len(cols) + "}", r"\toprule",
         " & " + " & ".join(LAB[c] for c in cols) + r" \\", r"\midrule",
         r"\multicolumn{" + str(len(cols) + 1) + r"}{l}{\emph{Probing only (no expert)}} \\"]

    def cell(x):
        if x is None:
            return "--"
        m, lo, hi = x["ci"]
        return f"{m:+.1f} [{lo:+.0f}, {hi:+.0f}]".replace("-", "$-$")

    def pcell(x):
        if x is None or "p_holm" not in x:
            return "--"
        ph = x["p_holm"]
        return f"{ph:.3f}" if ph >= 0.001 else "$<$0.001"

    def rows(spec):
        for k, lab in spec:
            L.append(lab + " & " + " & ".join(cell(o[c].get(k)) for c in cols) + r" \\")
            if any("p_holm" in (o[c].get(k) or {}) for c in cols):
                L.append(r"\quad $p_{\mathrm{Holm}}$ & " + " & ".join(pcell(o[c].get(k)) for c in cols) + r" \\")

    rows((("dual-rollout|cauprobe-v2-noask", r"dual $-$ \ours-2"),
          ("dual-rollout|greedy", r"dual $-$ greedy"),
          ("cauprobe-v2-noask|greedy", r"\ours-2 $-$ greedy")))
    L.append(r"\midrule")
    L.append(r"\multicolumn{" + str(len(cols) + 1) + r"}{l}{\emph{With the expert}} \\")
    rows((("dual-rollout+ask|cauprobe-v2", r"dual $-$ \ours-2"),
          ("dual-rollout+ask|cauprobe-noprobe", r"dual $-$ expert+greedy"),
          ("cauprobe-v2|greedy", r"\ours-2 $-$ greedy")))
    L.append(r"\midrule")
    L.append(r"\multicolumn{" + str(len(cols) + 1) + r"}{l}{\emph{Probe steps; runs ending on a wrong mechanism; compute per run}} \\")
    for m, lab in (("dual-rollout", "dual"), ("cauprobe-v2-noask", r"\ours-2 (no expert)")):
        cells = []
        for c in cols:
            s = o[c].get(f"stats:{m}")
            cells.append("--" if s is None else
                         f"{s['probes']:.0f}; {100 * s['wrong']:.0f}\\%; {s['wall_s']:.0f}\\,s")
        L.append(lab + " & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""]
    (ROOT / "paper" / "tables_cp_dual.tex").write_text("\n".join(L))


if __name__ == "__main__":
    main()

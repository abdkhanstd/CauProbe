"""Analyses added for the third review round (2026-10-18).

  signed    CauProbe-2 with per-structure gains counted signed instead of
            clipped at zero (results/cauprobe/ablate2), against the frozen
            CauProbe-2 runs on the same second-test plants and noise streams
            (results/cauprobe/test2): paired return, probes, lock-in
  mismatch  the fourth fresh test set (results/cauprobe/test4), plants drawn
            from parameter ranges twice as wide as the agent's prior: gap to
            the known-plant reference, paired differences against greedy and
            expert-plus-greedy control, CVaR, lock-in, misfit alarms
  safety    detector-threshold sensitivity on the broker-fault plants, with
            the in-library false-alarm rate of each threshold from the
            detector calibration runs
Writes results/cauprobe/review3.json and paper/tables_cp_review3.tex.
"""
from __future__ import annotations

import json
from collections import defaultdict

import numpy as np

from cp_common import CP_RESULTS, ROOT, TEST4_SEEDS
from cp_summarize import boot_ci, holm, paired_p


def load(p, seeds=None):
    rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    rows = [r for r in rows if "error" not in r]
    if seeds is not None:
        rows = [r for r in rows if r["seed"] // 10 in seeds]
    return rows


def wrong(r):
    fp = r.get("final_factor_post")
    if not fp:
        return None
    return any((fp[F] > 0.5) != bool(r["combo"][i]) for i, F in enumerate("ABC"))


def pdiff(a, b):
    """Paired difference of return, a minus b, per plant (runs averaged)."""
    acc = defaultdict(list)
    for s, r in a.items():
        if s in b:
            acc[s // 10].append(r["return"] - b[s]["return"])
    d = np.array([np.mean(v) for v in acc.values()])
    return {"ci": boot_ci(d), "p": paired_p(d), "n": len(d)}


def signed():
    out = {}
    for sc in ("nominal", "lpwan"):
        base = {r["seed"]: r for r in load(CP_RESULTS / "test2" / f"e1_{sc}.jsonl")
                if r["method"] == "cauprobe-v2"}
        sg = {r["seed"]: r for r in load(CP_RESULTS / "ablate2" / f"e1_{sc}.jsonl")}
        greedy = {r["seed"]: r for r in load(CP_RESULTS / "test2" / f"e1_{sc}.jsonl")
                  if r["method"] == "greedy"}
        out[sc] = {"signed_minus_clipped": pdiff(sg, base),
                   "signed_minus_greedy": pdiff(sg, greedy),
                   "clipped_minus_greedy": pdiff(base, greedy),
                   "probes": (float(np.mean([r["probe_steps"] for r in base.values()])),
                              float(np.mean([r["probe_steps"] for r in sg.values()]))),
                   "asks": (float(np.mean([len(r["asks"]) for r in base.values()])),
                            float(np.mean([len(r["asks"]) for r in sg.values()]))),
                   "wrong": (float(np.mean([bool(wrong(r)) for r in base.values()])),
                             float(np.mean([bool(wrong(r)) for r in sg.values()]))),
                   "identical_runs": int(sum(1 for s in sg if s in base and
                                             sg[s]["return"] == base[s]["return"])),
                   "n_runs": len(sg)}
    return out


def cvar(g, q=0.1):
    g = np.sort(np.asarray(g))[::-1]
    return float(g[:max(1, int(round(q * len(g))))].mean())


def mismatch():
    out = {}
    M = ["cauprobe-v2", "cauprobe-noprobe", "cauprobe", "thompson", "greedy"]
    keys, ps = [], []
    for sc in ("mismatch", "mismatch_lpwan"):
        rows = load(CP_RESULTS / "test4" / f"e1_{sc}.jsonl", TEST4_SEEDS)
        by = defaultdict(dict)
        for r in rows:
            by[r["method"]][r["seed"]] = r
        orc = by["oracle"]
        d = {}
        for m in M:
            gp = defaultdict(list)
            for s, r in by[m].items():
                gp[s // 10].append(orc[s]["return"] - r["return"])
            g = [np.mean(v) for v in gp.values()]
            w = [bool(wrong(r)) for r in by[m].values() if wrong(r) is not None]
            d[m] = {"gap": boot_ci(g), "cvar10": cvar(g), "wrong": float(np.mean(w)) if w else None,
                    "probes": float(np.mean([r["probe_steps"] for r in by[m].values()])),
                    "alarms": float(np.mean([r.get("misspec_alarms", 0) > 0 for r in by[m].values()])),
                    "fallback": float(np.mean([r["fallback_steps"] for r in by[m].values()]))}
        for a, b in (("cauprobe-v2", "greedy"), ("cauprobe-v2", "cauprobe-noprobe"),
                     ("cauprobe-v2", "cauprobe"), ("cauprobe-noprobe", "greedy")):
            d[f"{a}|{b}"] = pdiff(by[a], by[b])
            if a == "cauprobe-v2":
                keys.append((sc, f"{a}|{b}"))
                ps.append(d[f"{a}|{b}"]["p"])
        out[sc] = d
    for (sc, k), ph in zip(keys, holm(ps)):
        out[sc][k]["p_holm"] = float(ph)
    return out


def detector():
    cal = json.loads((CP_RESULTS / "detector_calibration.json").read_text())
    c = np.array(cal["in_library_cusum"])
    sens = json.loads((CP_RESULTS / "test2" / "safety_sens.json").read_text())["summary"]
    base = json.loads((CP_RESULTS / "test2" / "safety.json").read_text())["summary"]
    rows = {"21.6": base["cauprobe-v2"]}
    for h in ("15", "10", "5"):
        rows[h] = sens[f"cauprobe-v2@cusum_h={h}"]
    out = {}
    for h, r in rows.items():
        out[h] = {"false_alarm_in_library": float((c > float(h)).mean()),
                  "misses": r["misses"], "misses_fault": r["misses_fault"],
                  "delay": r["detect_delay_median_steps"],
                  "pre_alarm": r["misses_before_alarm_mean"],
                  "fallback_fault": r["fallback_fault"], "fallback_clear": r["fallback_clear"]}
    out["incumbent_misses"] = base["rule"]["misses"]
    out["fixed_fallback_misses"] = base["fallback"]["misses"]
    return out


def main():
    R = {"signed": signed(), "mismatch": mismatch(), "detector": detector()}
    (CP_RESULTS / "review3.json").write_text(json.dumps(R, indent=1, default=float))
    print(json.dumps(R, indent=1, default=float)[:9000])
    tables(R)


def tables(R):
    nm = {"cauprobe-v2": r"\ours-2", "cauprobe-noprobe": "expert + greedy",
          "cauprobe": r"\ours-1", "thompson": "Thompson sampling",
          "greedy": "greedy (same belief)"}
    M = R["mismatch"]
    L = [r"\begin{table*}[t]", r"\centering",
         r"\caption{Prior-generator mismatch: the fourth fresh test set (48 plants, "
         r"14 days; two runs per plant under Wi-Fi costs, one under LPWAN costs), "
         r"every continuous plant parameter drawn from a range twice as wide as "
         r"the agent's prior. Gap to the known-plant reference (mean [95\% CI]; "
         r"lower is better), worst-decile mean gap, runs ending on a wrong network "
         r"mechanism, probe steps per run and share of runs in which the misfit "
         r"detector fired. Bottom: paired differences in return for \ours-2, "
         r"$p_{\mathrm{Holm}}$ over the six comparisons.}",
         r"\label{tab:mismatch}", r"\footnotesize", r"\setlength{\tabcolsep}{3pt}",
         r"\begin{tabular}{lrrrrrrrrrr}", r"\toprule",
         r"& \multicolumn{5}{c}{Wi-Fi reconfiguration} & \multicolumn{5}{c}{LPWAN reconfiguration} \\",
         r"\cmidrule(lr){2-6}\cmidrule(lr){7-11}",
         r"Controller & gap & CVaR$_{10}$ & wrong & probes & alarm & gap & CVaR$_{10}$ & wrong & probes & alarm \\",
         r"\midrule"]
    for m in ("cauprobe-v2", "cauprobe-noprobe", "cauprobe", "thompson", "greedy"):
        cells = []
        for sc in ("mismatch", "mismatch_lpwan"):
            x = M[sc][m]
            g, lo, hi = x["gap"]
            cells += [f"{g:.1f} [{lo:.0f}, {hi:.0f}]", f"{x['cvar10']:.0f}",
                      f"{100 * x['wrong']:.0f}\\%" if x["wrong"] is not None else "--",
                      f"{x['probes']:.0f}", f"{100 * x['alarms']:.0f}\\%"]
        L.append(nm[m] + " & " + " & ".join(cells) + r" \\")
    L.append(r"\midrule")
    for k, lab in (("cauprobe-v2|greedy", r"\ours-2 $-$ greedy"),
                   ("cauprobe-v2|cauprobe-noprobe", r"\ours-2 $-$ expert + greedy"),
                   ("cauprobe-v2|cauprobe", r"\ours-2 $-$ \ours-1")):
        cells = []
        for sc in ("mismatch", "mismatch_lpwan"):
            x = M[sc][k]
            m_, lo, hi = x["ci"]
            cells.append(r"\multicolumn{5}{c}{" + f"{m_:+.1f} [{lo:+.1f}, {hi:+.1f}], "
                         + f"$p_{{\\mathrm{{Holm}}}}$ = {x['p_holm']:.3f}" + "}")
        L.append(lab + " & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""]
    (ROOT / "paper" / "tables_cp_review3.tex").write_text("\n".join(L))


if __name__ == "__main__":
    main()

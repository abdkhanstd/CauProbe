"""How sensitive is CauProbe-2 to mis-estimated free-learning horizons?

Every estimated horizon tau_s (probes, readings, night step tests) is
multiplied by 0.5 or 2 (capped at the information horizon; agent flag
fl_tau_scale, analysis only), and the controller is re-run on the second-test
plants (results/cauprobe/tausens2; first run of each plant, Wi-Fi and LPWAN
costs), paired with the frozen CauProbe-2 runs on the same plants and noise
streams (results/cauprobe/test2). Reports paired return (scaled minus
frozen, and minus greedy control), probe steps, share of runs that read the
notes, lock-in, and the share of runs whose return is unchanged. Holm over
the four comparisons with the frozen controller. Writes
results/cauprobe/tausens.json and paper/tables_cp_tausens.tex.
"""
from __future__ import annotations

import json

import numpy as np

from cp_common import CP_RESULTS, ROOT
from cp_review3 import load, pdiff, wrong
from cp_summarize import holm

SCALES = ("0.5", "2.0")
SCEN = (("nominal", "Wi-Fi"), ("lpwan", "LPWAN"))


def main():
    out, keys, ps = {}, [], []
    for sc, _ in SCEN:
        t2 = load(CP_RESULTS / "test2" / f"e1_{sc}.jsonl")
        base = {r["seed"]: r for r in t2 if r["method"] == "cauprobe-v2" and r["run"] == 0}
        greedy = {r["seed"]: r for r in t2 if r["method"] == "greedy" and r["run"] == 0}
        ts = load(CP_RESULTS / "tausens2" / f"e1_{sc}.jsonl")
        d = {"frozen": {"probes": float(np.mean([r["probe_steps"] for r in base.values()])),
                        "read": float(np.mean([len(r["asks"]) > 0 for r in base.values()])),
                        "wrong": float(np.mean([bool(wrong(r)) for r in base.values()])),
                        "vs_greedy": pdiff(base, greedy)}}
        for s in SCALES:
            m = {r["seed"]: r for r in ts if r["method"] == f"cauprobe-v2@fl_tau_scale={s}"}
            if not m:
                continue
            d[s] = {"vs_frozen": pdiff(m, base), "vs_greedy": pdiff(m, greedy),
                    "probes": float(np.mean([r["probe_steps"] for r in m.values()])),
                    "read": float(np.mean([len(r["asks"]) > 0 for r in m.values()])),
                    "wrong": float(np.mean([bool(wrong(r)) for r in m.values()])),
                    "unchanged": float(np.mean([m[k]["return"] == base[k]["return"]
                                                for k in m if k in base])),
                    "n": len(m)}
            keys.append((sc, s))
            ps.append(d[s]["vs_frozen"]["p"])
        out[sc] = d
    for (sc, s), ph in zip(keys, holm(ps)):
        out[sc][s]["vs_frozen"]["p_holm"] = float(ph)
    (CP_RESULTS / "tausens.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps(out, indent=1, default=float))

    def ci(x):
        m, lo, hi = x["ci"]
        return f"{m:+.1f} [{lo:+.1f}, {hi:+.1f}]".replace("-", "$-$")

    L = [r"\begin{table}[t]", r"\centering",
         r"\caption{Sensitivity of \ours-2 to mis-estimated free-learning horizons: "
         r"every estimated $\tau_s$ multiplied by 0.5 or 2 (capped at the information "
         r"horizon), second-test plants, first run of each plant. Paired differences in "
         r"14-day return against the frozen controller (95\% CI; $p_{\mathrm{Holm}}$ over "
         r"the four comparisons) and against greedy control; probe steps per run; share "
         r"of runs that read the notes; runs ending on a wrong mechanism.}",
         r"\label{tab:tausens}", r"\footnotesize", r"\setlength{\tabcolsep}{3pt}",
         r"\begin{tabular}{llrrrrr}", r"\toprule",
         r"Costs & $\tau$ scale & vs frozen ($p_{\mathrm{Holm}}$) & vs greedy & probes & read & wrong \\",
         r"\midrule"]
    for sc, lab in SCEN:
        d = out[sc]
        f = d["frozen"]
        L.append(f"{lab} & 1 (frozen) & -- & {ci(f['vs_greedy'])} & {f['probes']:.0f} & "
                 f"{100 * f['read']:.0f}\\% & {100 * f['wrong']:.0f}\\% \\\\")
        for s in SCALES:
            if s not in d:
                continue
            x = d[s]
            L.append(f" & {float(s):g} & {ci(x['vs_frozen'])} ({x['vs_frozen']['p_holm']:.2f}) & "
                     f"{ci(x['vs_greedy'])} & {x['probes']:.0f} & {100 * x['read']:.0f}\\% & "
                     f"{100 * x['wrong']:.0f}\\% \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    (ROOT / "paper" / "tables_cp_tausens.tex").write_text("\n".join(L))


if __name__ == "__main__":
    main()

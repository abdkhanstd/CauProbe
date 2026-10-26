"""Pooled analysis of the learning-aware controller over both fresh test sets.

The second and third test sets were never used for any decision about
CauProbe-2, so together they give ten scenario-sets of paired evidence (six
and four scenarios, 48 plants each). A plant appears in every scenario of its
set, so uncertainty is estimated by a cluster bootstrap that resamples plants
(within each set) and keeps all of a plant's scenarios together.

Reported, for CauProbe-2 against expert-plus-greedy control and against greedy
control on the same belief:
  mean      paired difference in 14-day return, 95% CI, and the TOST
            equivalence margin -- the smallest +/- margin the 90% CI fits in
  tail      CVaR of the gap to the known-plant reference over the worst 10%
            and 5% of plant-scenarios, and the paired CI of its difference
  severe    share of plant-scenarios whose gap to the reference exceeds 50 and
            100 reward units, and the paired CI of the difference
  lock-in   share of runs ending on a wrong network mechanism, with the
            discordant counts and an exact McNemar test
Writes results/cauprobe/pooled.json and paper/tables_cp_pooled.tex.
"""
from __future__ import annotations

import json
from collections import defaultdict
from math import comb

import numpy as np

from cp_common import CP_RESULTS, ROOT

SETS = {"test2": ["nominal", "noisy", "lpwan", "noisy_lpwan", "poor_docs", "shift"],
        "test3": ["nominal", "poor_docs", "lpwan", "noisy"]}
METHODS = ["cauprobe-v2", "cauprobe-noprobe", "greedy", "cauprobe"]
B = 10000


def load():
    """(set, scenario, plant) -> method -> {gap, ret, wrong-rate}."""
    D = defaultdict(dict)
    W = defaultdict(lambda: defaultdict(list))
    for st, scen in SETS.items():
        for sc in scen:
            p = CP_RESULTS / st / f"e1_{sc}.jsonl"
            rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
            rows = [r for r in rows if "error" not in r]
            orc = {r["seed"]: r for r in rows if r["method"] == "oracle"}
            acc = defaultdict(lambda: defaultdict(list))
            for r in rows:
                if r["method"] not in METHODS or r["seed"] not in orc:
                    continue
                acc[(st, sc, r["seed"] // 10)][r["method"]].append(
                    (orc[r["seed"]]["return"] - r["return"], r["return"]))
                fp = r.get("final_factor_post")
                if fp:
                    c = list(r["combo"])
                    if sc == "shift":
                        c[0] = 1 - c[0]
                    W[r["method"]][(st, sc, r["seed"])].append(
                        any((fp[F] > 0.5) != bool(c[i]) for i, F in enumerate("ABC")))
            for k, v in acc.items():
                D[k] = {m: {"gap": float(np.mean([x[0] for x in xs])),
                            "ret": float(np.mean([x[1] for x in xs]))} for m, xs in v.items()}
    return D, W


def cluster_index(keys, rng):
    """Resample plants within each set; return the resampled list of keys."""
    by = defaultdict(lambda: defaultdict(list))
    for k in keys:
        by[k[0]][k[2]].append(k)
    out = []
    for st, plants in by.items():
        ps = list(plants)
        for p in rng.choice(len(ps), len(ps)):
            out += plants[ps[p]]
    return out


def cvar(g, q):
    g = np.sort(np.asarray(g))[::-1]
    return float(g[:max(1, int(round(q * len(g))))].mean())


def compare(D, a, b, rng):
    keys = [k for k, v in D.items() if a in v and b in v]
    d = lambda ks: np.array([D[k][a]["ret"] - D[k][b]["ret"] for k in ks])
    ga = lambda ks: np.array([D[k][a]["gap"] for k in ks])
    gb = lambda ks: np.array([D[k][b]["gap"] for k in ks])
    obs = {"n": len(keys), "mean": float(d(keys).mean()),
           "cvar10": (cvar(ga(keys), .1), cvar(gb(keys), .1)),
           "cvar5": (cvar(ga(keys), .05), cvar(gb(keys), .05)),
           "severe50": (float((ga(keys) > 50).mean()), float((gb(keys) > 50).mean())),
           "severe100": (float((ga(keys) > 100).mean()), float((gb(keys) > 100).mean()))}
    bs = defaultdict(list)
    for _ in range(B):
        ks = cluster_index(keys, rng)
        x, ya, yb = d(ks), ga(ks), gb(ks)
        bs["mean"].append(x.mean())
        bs["cvar10"].append(cvar(yb, .1) - cvar(ya, .1))
        bs["cvar5"].append(cvar(yb, .05) - cvar(ya, .05))
        bs["severe50"].append((yb > 50).mean() - (ya > 50).mean())
        bs["severe100"].append((yb > 100).mean() - (ya > 100).mean())
    ci = lambda v, lo=2.5, hi=97.5: (float(np.percentile(v, lo)), float(np.percentile(v, hi)))
    obs["mean_ci95"] = ci(bs["mean"])
    obs["mean_ci90"] = ci(bs["mean"], 5, 95)
    obs["tost_margin"] = float(max(abs(obs["mean_ci90"][0]), abs(obs["mean_ci90"][1])))
    for k in ("cvar10", "cvar5", "severe50", "severe100"):
        obs[f"{k}_reduction_ci95"] = ci(bs[k])
        obs[f"{k}_p_le0"] = float((np.asarray(bs[k]) <= 0).mean())
    return obs


def lockin(W, a, b):
    ka = {k: np.mean(v) > 0.5 for k, v in W[a].items()}
    kb = {k: np.mean(v) > 0.5 for k, v in W[b].items()}
    ks = sorted(set(ka) & set(kb))
    n10 = sum(1 for k in ks if not ka[k] and kb[k])
    n01 = sum(1 for k in ks if ka[k] and not kb[k])
    n = n10 + n01
    p = min(1.0, 2 * sum(comb(n, i) for i in range(min(n10, n01) + 1)) / 2 ** n) if n else 1.0
    return {"rate": (float(np.mean([ka[k] for k in ks])), float(np.mean([kb[k] for k in ks]))),
            "discordant": (n10, n01), "mcnemar_p": p, "n_runs": len(ks)}


def noninferiority(D, rng):
    """Non-inferiority of CauProbe-2 to expert-plus-greedy control with a margin
    fixed on independent data. M1, the effect of the comparator over greedy
    control, is estimated on the FIRST test set (never used for CauProbe-2),
    conservatively as the lower 95% bound of its pooled, plant-clustered
    bootstrap; the margin is M2 = M1 / 2 (the conventional 50%-preserved-effect
    rule). Non-inferiority holds if the lower bound of the two-sided 90% CI of
    CauProbe-2 minus expert-plus-greedy on the second and third sets exceeds -M2."""
    from cp_common import testrows
    T1 = ["main", "noisy", "lpwan", "noisy_lpwan", "poor_docs", "shift"]
    H = defaultdict(dict)
    for sc in T1:
        rows = testrows([json.loads(l) for l in
                         (CP_RESULTS / f"e1_test_{sc}.jsonl").read_text().splitlines() if l.strip()])
        acc = defaultdict(lambda: defaultdict(list))
        for r in rows:
            if r["method"] in ("cauprobe-noprobe", "greedy") and "error" not in r:
                acc[("test1", sc, r["seed"] // 10)][r["method"]].append(r["return"])
        for k, v in acc.items():
            if len(v) == 2:
                H[k] = {m: {"ret": float(np.mean(x))} for m, x in v.items()}
    keys = list(H)
    d = lambda ks: np.array([H[k]["cauprobe-noprobe"]["ret"] - H[k]["greedy"]["ret"] for k in ks])
    bs = [d(cluster_index(keys, rng)).mean() for _ in range(B)]
    m1_mean = float(d(keys).mean())
    m1 = float(np.percentile(bs, 2.5))
    m2 = m1 / 2.0
    c = compare(D, "cauprobe-v2", "cauprobe-noprobe", rng)
    lo90 = c["mean_ci90"][0]
    return {"M1_point": m1_mean, "M1_lower95": m1, "M2_margin": m2,
            "n_hist_plant_scenarios": len(keys), "diff": c["mean"],
            "ci90": c["mean_ci90"], "noninferior": bool(lo90 > -m2),
            "equivalent_at_margin": bool(lo90 > -m2 and c["mean_ci90"][1] < m2)}


def main():
    import sys
    rng = np.random.default_rng(0)
    D, W = load()
    if "--ni" in sys.argv:
        ni = noninferiority(D, rng)
        (CP_RESULTS / "noninferiority.json").write_text(json.dumps(ni, indent=1, default=float))
        print(json.dumps(ni, indent=1, default=float))
        return
    out = {}
    for a, b in (("cauprobe-v2", "cauprobe-noprobe"), ("cauprobe-v2", "greedy"),
                 ("cauprobe-noprobe", "greedy")):
        out[f"{a}|{b}"] = compare(D, a, b, rng)
        out[f"{a}|{b}"]["lockin"] = lockin(W, a, b)
        print(a, b, json.dumps(out[f"{a}|{b}"], default=float))
    (CP_RESULTS / "pooled.json").write_text(json.dumps(out, indent=1, default=float))
    table(out)


def table(o):
    nm = {"cauprobe-v2": r"\ours-2", "cauprobe-noprobe": "expert + greedy",
          "greedy": "greedy"}
    L = [r"\begin{table*}[t]", r"\centering",
         r"\caption{Pooled over the two test sets on which \ours-2 was never developed: "
         r"the second (six scenarios) and the third (four scenarios: nominal, poor "
         r"documentation, LPWAN, noisy), 48 plants each, ten scenario sets, 480 "
         r"plant-scenarios, all with the original (training-vocabulary) expert; "
         r"cluster bootstrap over plants. \emph{Mean}: "
         r"paired difference in 14-day return, first minus second, with its 95\% "
         r"and 90\% CIs (no equivalence margin is implied; see text). "
         r"\emph{CVaR}: mean gap to the known-plant "
         r"reference over the worst 10\% / 5\% of plant-scenarios. "
         r"\emph{Severe}: share of plant-scenarios more than 50 / 100 behind "
         r"the reference. \emph{Lock-in}: runs ending on a wrong network "
         r"mechanism; exact McNemar test on discordant runs. Reductions are "
         r"second minus first, with 95\% CIs.}",
         r"\label{tab:pooled}", r"\footnotesize", r"\setlength{\tabcolsep}{3pt}",
         r"\begin{tabular}{lccc}", r"\toprule",
         r" & " + " & ".join(f"{nm[k.split('|')[0]]} vs {nm[k.split('|')[1]]}" for k in o)
         + r" \\", r"\midrule"]

    def f(x, nd=1):
        return f"{x:+.{nd}f}"

    rows = [("mean difference", lambda v: f"{f(v['mean'])} [{f(v['mean_ci95'][0])}, {f(v['mean_ci95'][1])}]"),
            ("\\quad 90\\% CI", lambda v: f"[{f(v['mean_ci90'][0])}, {f(v['mean_ci90'][1])}]"),
            ("CVaR$_{10}$ (first; second)", lambda v: f"{v['cvar10'][0]:.0f}; {v['cvar10'][1]:.0f}"),
            ("\\quad reduction [95\\% CI]", lambda v: f"{f(v['cvar10'][1]-v['cvar10'][0],0)} [{f(v['cvar10_reduction_ci95'][0],0)}, {f(v['cvar10_reduction_ci95'][1],0)}]"),
            ("CVaR$_{5}$ (first; second)", lambda v: f"{v['cvar5'][0]:.0f}; {v['cvar5'][1]:.0f}"),
            ("\\quad reduction [95\\% CI]", lambda v: f"{f(v['cvar5'][1]-v['cvar5'][0],0)} [{f(v['cvar5_reduction_ci95'][0],0)}, {f(v['cvar5_reduction_ci95'][1],0)}]"),
            ("severe $>$50 (first; second)", lambda v: f"{100*v['severe50'][0]:.1f}\\%; {100*v['severe50'][1]:.1f}\\%"),
            ("severe $>$100 (first; second)", lambda v: f"{100*v['severe100'][0]:.1f}\\%; {100*v['severe100'][1]:.1f}\\%"),
            ("lock-in (first; second)", lambda v: f"{100*v['lockin']['rate'][0]:.1f}\\%; {100*v['lockin']['rate'][1]:.1f}\\%"),
            ("\\quad McNemar $p$", lambda v: f"{v['lockin']['mcnemar_p']:.2g}")]
    for lab, fn in rows:
        L.append(lab + " & " + " & ".join(fn(v) for v in o.values()) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""]
    (ROOT / "paper" / "tables_cp_pooled.tex").write_text("\n".join(L))


if __name__ == "__main__":
    main()

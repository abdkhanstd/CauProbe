"""What predicts whether a probe pays? Analysis of cp_audit2.py.

For the audited probe decisions of CauProbe-2 (second test set): the AUROC of
each candidate score for separating probes that paid (realised value > 0)
from those that did not, with bootstrap CIs over plants; Spearman correlations
with the realised value; a cross-validated logistic model on all decision-state
features (folds by plant); and the realised value by stratum (reconfiguration
costs, knob moved, day, how undecided the processing-path factor was).
Writes results/cauprobe/test2/probe_audit2_summary.json and
paper/tables_cp_audit2.tex.
"""
from __future__ import annotations

import json
from collections import defaultdict

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from cp_common import CP_RESULTS, ROOT

P = CP_RESULTS / "test2" / "probe_audit2.jsonl"

SCORES = [("pred_net", "\\ours-2 predicted net value", 1),
          ("direct", "rollout (direct) value of the same probe", 1),
          ("evpi", "EVPI (posterior regret)", 1),
          ("hevpi", "$H_{\\mathrm{eff}}\\cdot$EVPI (static value bound)", 1),
          ("uncert", "$1 - \\max_s \\beta(s)$", 1),
          ("entropy", "entropy of the structure posterior", 1),
          ("b_undecided", "processing factor undecided", 1),
          ("early", "earliness ($-$day)", 1),
          ("m", "probe length", 1)]


def auroc_ci(y, s, groups, rng, B=5000):
    ug = np.unique(groups)
    idx = {g: np.flatnonzero(groups == g) for g in ug}
    bs = []
    for _ in range(B):
        i = np.concatenate([idx[g] for g in rng.choice(ug, len(ug))])
        if 0 < y[i].sum() < len(i):
            bs.append(roc_auc_score(y[i], s[i]))
    return float(roc_auc_score(y, s)), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def main():
    rows = [json.loads(l) for l in P.read_text().splitlines() if l.strip()]
    for r in rows:
        r["hevpi"] = r["h_eff"] * r["evpi"]
        r["uncert"] = 1.0 - r["max_beta"]
        r["early"] = -r["day"]
    rows = [r for r in rows if np.isfinite(r["pred_net"]) and np.isfinite(r["direct"])
            and np.isfinite(r["evpi"])]
    y = np.array([r["realised"] > 0 for r in rows])
    real = np.array([r["realised"] for r in rows])
    groups = np.array([r["seed"] for r in rows])
    rng = np.random.default_rng(0)
    out = {"n": len(rows), "paid": float(y.mean()),
           "realised_mean": float(real.mean()),
           "realised_ci": None, "scores": {}}
    ug = np.unique(groups)
    bs = [np.mean(np.concatenate([real[groups == g] for g in rng.choice(ug, len(ug))]))
          for _ in range(5000)]
    out["realised_ci"] = (float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)))
    for k, lab, sign in SCORES:
        s = sign * np.array([r[k] for r in rows], float)
        a = auroc_ci(y, s, groups, rng)
        rho = spearmanr(s, real)
        out["scores"][k] = {"label": lab, "auroc": a, "spearman": float(rho.correlation),
                            "spearman_p": float(rho.pvalue)}
    # cross-validated logistic model on all features, folds by plant
    F = ["pred_net", "direct", "evpi", "hevpi", "uncert", "entropy", "b_undecided",
         "a_undecided", "c_undecided", "h_eff", "kappa", "day", "m"]
    X = np.array([[r[k] for k in F] for r in rows], float)
    X = np.column_stack([X, [r["switch"] == "lpwan" for r in rows],
                         [("o" in (r["knob"] or "")) for r in rows]])
    pred = np.zeros(len(rows))
    folds = np.array_split(rng.permutation(ug), 8)
    for fg in folds:
        te = np.isin(groups, fg)
        sc = StandardScaler().fit(X[~te])
        m = LogisticRegression(C=0.5, max_iter=2000).fit(sc.transform(X[~te]), y[~te])
        pred[te] = m.predict_proba(sc.transform(X[te]))[:, 1]
    out["logistic_cv_auroc"] = auroc_ci(y, pred, groups, rng)
    # strata
    strata = defaultdict(list)
    for r in rows:
        strata[f"switch:{r['switch']}"].append(r["realised"])
        strata[f"offload_moved:{'o' in (r['knob'] or '')}"].append(r["realised"])
        strata[f"first_day:{r['day'] == 0}"].append(r["realised"])
        strata[f"b_undecided>0.25:{r['b_undecided'] > 0.25}"].append(r["realised"])
        strata[f"plant_B:{r['combo'][1]}"].append(r["realised"])
    out["strata"] = {k: {"n": len(v), "mean": float(np.mean(v)), "paid": float(np.mean(np.array(v) > 0))}
                     for k, v in sorted(strata.items())}
    (CP_RESULTS / "test2" / "probe_audit2_summary.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1)[:5000])
    L = [r"\begin{table}[t]", r"\centering",
         r"\caption{What predicts whether a probe of \ours-2 pays: AUROC of each "
         r"score for separating the " + str(len(rows)) + r" audited probe decisions "
         r"that paid from those that did not (95\% bootstrap CI over plants), and "
         r"Spearman correlation with the realised value. Last row: a logistic "
         r"model on all decision-state features, cross-validated with folds by "
         r"plant.}",
         r"\label{tab:audit2}", r"\footnotesize", r"\setlength{\tabcolsep}{3pt}",
         r"\begin{tabular}{lrr}", r"\toprule",
         r"Score at the decision & AUROC [95\% CI] & Spearman $\rho$ \\", r"\midrule"]
    for k, lab, _ in SCORES:
        s = out["scores"][k]
        a = s["auroc"]
        L.append(f"{lab} & {a[0]:.2f} [{a[1]:.2f}, {a[2]:.2f}] & " + f"{s['spearman']:+.2f}".replace("-", "$-$") + r" \\")
    a = out["logistic_cv_auroc"]
    L += [r"\midrule", f"all features, logistic (cross-validated) & {a[0]:.2f} [{a[1]:.2f}, {a[2]:.2f}] & -- \\\\",
          r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    (ROOT / "paper" / "tables_cp_audit2.tex").write_text("\n".join(L))


if __name__ == "__main__":
    main()

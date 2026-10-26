"""How much of a probe's realised value could any decision-time score foresee?

Analysis of cp_audit3.py (probe decisions of CauProbe-2 on the second test
set, each replayed R times with independent future noise shared by both
branches). Per decision, the replicate mean estimates the decision's expected
value and the replicate spread is outcome noise. Reports
  - the share of the variance of a single realised draw that lies between
    decisions (intraclass correlation): the ceiling on how well any score of
    the decision state can track one draw;
  - the reliability of each decision's R-replicate mean (Spearman-Brown from
    the ICC, and split-half on ranks), i.e. how well the mean ranks the
    decisions by their true expected value;
  - how many decisions have an expected value distinguishable from zero;
  - AUROC / Spearman of the decision-time scores against the sign and size of
    the expected value (not of one draw), with bootstrap CIs over plants, and
    Spearman corrected for the reliability of the mean (an estimate of, and
    bound on, the score's correlation with the true expected value);
  - the agreement of the single draw of cp_audit2.py with the replicate mean.
--src picks the run: "powered" (default; every audited decision, 36
replicates, cp_audit3.py --n 999 --reps 36 --chunk 6) or "48x6" (the first
run). Writes <run>_summary.json next to the run and paper/tables_cp_audit3.tex.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
from scipy.stats import rankdata, spearmanr, t as tdist
from sklearn.metrics import roc_auc_score

from cp_common import CP_RESULTS, ROOT

SRC = {"48x6": CP_RESULTS / "test2" / "probe_audit3.jsonl",
       "powered": CP_RESULTS / "test2" / "probe_audit3_powered.jsonl"}
P2 = CP_RESULTS / "test2" / "probe_audit2.jsonl"

SCORES = [("pred_net", "\\ours-2 predicted net value"),
          ("direct", "rollout (direct) value of the same probe"),
          ("hevpi", "$H_{\\mathrm{eff}}\\cdot$EVPI (static value bound)"),
          ("uncert", "$1 - \\max_s \\beta(s)$"),
          ("realised2", "a second draw of the same decision$^{a}$")]


def mn(x):
    """Typeset minus signs in a numeric cell."""
    return x.replace("-", "$-$")


def boot(fn, groups, rng, B=5000):
    ug = np.unique(groups)
    idx = {g: np.flatnonzero(groups == g) for g in ug}
    out = []
    for _ in range(B):
        i = np.concatenate([idx[g] for g in rng.choice(ug, len(ug))])
        try:
            v = fn(i)
        except ValueError:
            continue
        if np.isfinite(v):
            out.append(v)
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def load(src):
    """Decisions in order of first appearance, with a balanced replicate set:
    the replicates 0..R-1 that every decision has (jobs may hold blocks)."""
    reps = {}
    for d in (json.loads(l) for l in src.read_text().splitlines() if l.strip()):
        assert all(d["probed"]), "a replayed decision did not take the audited probe"
        v = reps.setdefault((d["seed"], d["switch"], d["t"]), {})
        for j, x in enumerate(d["realised_reps"]):
            v[d.get("k0", 0) + j] = x
    R = min(next(k for k in range(len(v) + 1) if k not in v) for v in reps.values())
    return {key: np.array([v[k] for k in range(R)], float) for key, v in reps.items()}, R


def split_half(V, rng, n_split=500):
    """Spearman-Brown corrected split-half reliability of the R-replicate mean,
    on ranks (Spearman) and raw values (Pearson), averaged over random splits."""
    R = V.shape[1]
    rs, rp = [], []
    for _ in range(n_split):
        p = rng.permutation(R)
        a, b = V[:, p[: R // 2]].mean(1), V[:, p[R // 2: 2 * (R // 2)]].mean(1)
        rs.append(spearmanr(a, b).correlation)
        rp.append(np.corrcoef(a, b)[0, 1])
    sb = lambda r: 2 * r / (1 + r) if r > -1 else np.nan
    # the halves have R//2 replicates each; Spearman-Brown to the full R
    k = R / (2 * (R // 2))
    full = lambda r: k * sb(r) / (1 + (k - 1) * sb(r))
    return float(full(np.mean(rs))), float(full(np.mean(rp))), float(np.mean(rs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="powered", choices=list(SRC))
    ap.add_argument("--table", default=str(ROOT / "paper" / "tables_cp_audit3.tex"))
    a = ap.parse_args()
    src = SRC[a.src]
    reps, R = load(src)
    r2 = {(r["seed"], r["switch"], r["t"]): r for r in
          (json.loads(l) for l in P2.read_text().splitlines() if l.strip())}
    rows = []
    for key, v in reps.items():
        b = r2[key]
        rows.append({**b, "reps": v, "mean": v.mean(), "sd": v.std(ddof=1),
                     "realised2": b["realised"], "hevpi": b["h_eff"] * b["evpi"],
                     "uncert": 1.0 - b["max_beta"]})
    n = len(rows)
    M = np.array([r["mean"] for r in rows])
    V = np.array([r["reps"] for r in rows])
    groups = np.array([r["seed"] for r in rows])
    rng = np.random.default_rng(0)
    # one-way random-effects ANOVA: decision is the factor
    msw = float(np.mean(V.var(axis=1, ddof=1)))
    msb = float(R * M.var(ddof=1))
    s2b = max(0.0, (msb - msw) / R)
    icc = s2b / (s2b + msw)

    def icc_fn(i):
        Vi = V[i]
        w = np.mean(Vi.var(axis=1, ddof=1))
        b = max(0.0, (R * Vi.mean(axis=1).var(ddof=1) - w) / R)
        return b / (b + w)

    # robust version on ranks within the pooled draws (heavy tails)
    rk = np.argsort(np.argsort(V.ravel())).reshape(V.shape).astype(float)
    msw_r = float(np.mean(rk.var(axis=1, ddof=1)))
    s2b_r = max(0.0, (R * rk.mean(axis=1).var(ddof=1) - msw_r) / R)
    icc_rank = s2b_r / (s2b_r + msw_r)
    # decisions whose expected value is distinguishable from zero (t, 5%)
    se = np.array([r["sd"] for r in rows]) / np.sqrt(R)
    tq = tdist.ppf(0.975, R - 1)
    pos = int(np.sum(M - tq * se > 0))
    neg = int(np.sum(M + tq * se < 0))
    zero_spread = int(np.sum(V.std(axis=1) < 1e-9))
    # sign consistency: share of replicates that agree with the sign of the mean
    agree = float(np.mean([(np.sign(r["reps"]) == np.sign(r["mean"])).mean() for r in rows]))
    out = {"src": src.name, "n": n, "reps": R, "plants": int(len(np.unique(groups))),
           "icc": icc, "icc_ci": boot(icc_fn, groups, rng),
           "icc_rank": icc_rank, "within_sd_median": float(np.median([r["sd"] for r in rows])),
           "mean_abs_expected_median": float(np.median(np.abs(M))),
           "expected_positive_sig": pos, "expected_negative_sig": neg,
           "zero_spread": zero_spread, "sign_agreement": agree,
           "paid_expected": float(np.mean(M > 0)),
           "mean_expected": float(M.mean()),
           "mean_expected_ci": boot(lambda i: M[i].mean(), groups, rng),
           "scores": {}}
    # how well the R-replicate mean ranks decisions by their expected value
    rel_icc = R * icc / (1 + (R - 1) * icc)
    rel_rank_icc = R * icc_rank / (1 + (R - 1) * icc_rank)
    rel_sh_rank, rel_sh_raw, r_half = split_half(V, np.random.default_rng(1))
    out["reliability"] = {"spearman_brown_icc": rel_icc, "spearman_brown_icc_rank": rel_rank_icc,
                          "split_half_rank": rel_sh_rank, "split_half_raw": rel_sh_raw,
                          "half_vs_half_spearman": r_half,
                          "spearman_brown_icc_ci": [R * c / (1 + (R - 1) * c) for c in out["icc_ci"]]}
    rel = max(rel_sh_rank, 1e-6)
    # secondary, robust target (fixed before the powered run was analysed): a
    # decision's mean rank among all pooled draws (ties averaged). Unlike the
    # mean it is not dominated by a decision's rare large draws.
    Q = rankdata(V.ravel()).reshape(V.shape)
    MQ = Q.mean(1)
    rel_q = max(split_half(Q, np.random.default_rng(3))[0], 1e-6)
    out["reliability"]["split_half_mean_rank"] = rel_q
    rng2, rng3 = np.random.default_rng(2), np.random.default_rng(4)
    # expected values distinguishable from zero after Benjamini-Hochberg (5%)
    if R > 1:
        with np.errstate(divide="ignore", invalid="ignore"):
            tt = np.where(se > 0, M / se, 0.0)
        p = 2 * tdist.sf(np.abs(tt), R - 1)
        ps = np.sort(p)
        kk = np.flatnonzero(ps <= 0.05 * np.arange(1, n + 1) / n)
        out["expected_nonzero_bh"] = int(kk[-1] + 1) if len(kk) else 0
        sig = (M - tq * se > 0) | (M + tq * se < 0)
    y = M > 0
    for k, lab in SCORES:
        s = np.array([r[k] for r in rows], float)
        ok = np.isfinite(s)
        a_ = float(roc_auc_score(y[ok], s[ok]))
        ci = boot(lambda i: roc_auc_score(y[i][ok[i]], s[i][ok[i]]), groups, rng)
        rho = spearmanr(s[ok], M[ok])
        rci = boot(lambda i: spearmanr(s[i][ok[i]], M[i][ok[i]]).correlation, groups, rng)
        corr = [float(np.clip(x / np.sqrt(rel), -1, 1)) for x in (rho.correlation, *rci)]
        out["scores"][k] = {"label": lab, "auroc": [a_, *ci], "spearman": [float(rho.correlation), *rci],
                            "spearman_p": float(rho.pvalue), "n": int(ok.sum()),
                            "spearman_corrected": corr}
        rq = spearmanr(s[ok], MQ[ok]).correlation
        rqci = boot(lambda i: spearmanr(s[i][ok[i]], MQ[i][ok[i]]).correlation, groups, rng2)
        out["scores"][k]["spearman_mean_rank"] = [float(rq), *rqci]
        out["scores"][k]["spearman_mean_rank_corrected"] = [
            float(np.clip(x / np.sqrt(rel_q), -1, 1)) for x in (rq, *rqci)]
        # sign established (t, 5%) and by link cost
        if R > 1 and sig.sum() >= 4 and 0 < y[sig].sum() < sig.sum():
            m = sig & ok
            out["scores"][k]["auroc_sign_established"] = [
                float(roc_auc_score(y[m], s[m])),
                *boot(lambda i: roc_auc_score(y[i][m[i]], s[i][m[i]]), groups, rng3),
                int(m.sum())]
        for sw in ("wifi", "lpwan"):
            m = ok & np.array([r["switch"] == sw for r in rows])
            if m.sum() > 3:
                out["scores"][k][sw] = {"n": int(m.sum()),
                                        "auroc": float(roc_auc_score(y[m], s[m])),
                                        "spearman": float(spearmanr(s[m], M[m]).correlation)}
    # does the single draw land on the same side as the expected value?
    d2 = np.array([r["realised2"] for r in rows])
    out["single_draw_sign_agrees"] = float(np.mean(np.sign(d2) == np.sign(M)))
    out["mean_pred_net"] = float(np.mean([r["pred_net"] for r in rows]))
    # replicates 0-5 of the powered run must reproduce the 48 x 6 run exactly
    if src != SRC["48x6"] and SRC["48x6"].exists():
        old, R0 = load(SRC["48x6"])
        diffs = [np.abs(reps[key][:R0] - v).max() for key, v in old.items() if key in reps]
        out["reproduces_48x6"] = {"decisions": len(diffs), "replicates": int(min(R0, R)),
                                  "max_abs_diff": float(max(diffs)) if diffs else None}
    js = src.with_name(src.stem + "_summary.json")
    js.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    rl = out["reliability"]
    L = [r"\begin{table}[t]", r"\centering",
         r"\caption{Replicated audit: " + str(n) + r" probe decisions of \ours-2 on "
         + str(out["plants"]) + r" second-test plants, each replayed " + str(R)
         + r" times with independent future noise shared by the probe and the greedy "
         r"branch. Scores at the decision against the sign (AUROC) and size "
         r"(Spearman $\rho$) of each decision's \emph{expected} value, i.e.\ the mean "
         r"over replicates, and against a robust location, the mean rank of its "
         r"draws among all draws; 95\% bootstrap CIs over plants. Share of a single "
         r"draw's variance that lies between decisions: "
         + f"{icc:.2f} [{out['icc_ci'][0]:.2f}, {out['icc_ci'][1]:.2f}] ({icc_rank:.2f} on ranks). "
         r"Split-half reliability (on ranks) of the " + str(R) + r"-replicate mean "
         + f"{rl['split_half_rank']:.2f}" + r", of the mean rank "
         + f"{rl['split_half_mean_rank']:.2f}" + r". "
         r"$\rho_{\mathrm{c}}$: $\rho$ and its CI divided by the square root of that "
         r"reliability, an estimate of the correlation with the true value. "
         r"$^{a}$The draw of the first audit, made with other noise: not available at "
         r"the decision, it shows what a predictor carrying the decision's own signal "
         r"achieves.}",
         r"\label{tab:audit3}", r"\footnotesize", r"\setlength{\tabcolsep}{3pt}",
         r"\begin{tabular}{lrrr}", r"\toprule",
         r"Score at the decision & AUROC [95\% CI] & Spearman $\rho$ [95\% CI] & "
         r"$\rho_{\mathrm{c}}$ [95\% CI] \\"]
    for panel, kr, kc in (("expected value (mean over replicates)", "spearman",
                           "spearman_corrected"),
                          ("mean rank of the draws", "spearman_mean_rank",
                           "spearman_mean_rank_corrected")):
        L += [r"\midrule", r"\multicolumn{4}{l}{\emph{Target: " + panel + r"}} \\"]
        for k, lab in SCORES:
            s = out["scores"][k]
            a_, rh, rc = s["auroc"], s[kr], s[kc]
            au = f"{a_[0]:.2f} [{a_[1]:.2f}, {a_[2]:.2f}]" if kr == "spearman" else "--"
            L.append(f"{lab} & {au} & "
                     + mn(f"{rh[0]:+.2f} [{rh[1]:+.2f}, {rh[2]:+.2f}]") + " & "
                     + mn(f"{rc[0]:+.2f} [{rc[1]:+.2f}, {rc[2]:+.2f}]") + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    with open(a.table, "w", newline="\n") as f:
        f.write("\n".join(L))


if __name__ == "__main__":
    main()

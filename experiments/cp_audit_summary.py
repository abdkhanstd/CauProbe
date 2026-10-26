"""Summarise the paired probe audit (cp_audit_probes.py).

For each iteration and stratum (reconfiguration costs; processing path B, the
factor the over-probing concerns; probe length m): mean predicted net value of
the chosen probe, mean realised value (paired difference in return to the end
of the run, probe minus greedy at that step), the share of probes that paid
(realised > 0), and a bootstrap interval for the mean realised value. Writes
results/cauprobe/test2/probe_audit_summary.json and figure cp_fig11_audit.
"""
from __future__ import annotations

import json
from collections import defaultdict

import numpy as np

from cp_common import CP_RESULTS, ROOT

P = CP_RESULTS / "test2" / "probe_audit.jsonl"


def boot(x, n=10000, seed=0):
    x = np.asarray(x, float)
    if len(x) < 2:
        return (float(x.mean()) if len(x) else float("nan"),) * 3
    rng = np.random.default_rng(seed)
    b = rng.choice(x, (n, len(x))).mean(1)
    return float(x.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def stats(rr):
    pr = np.array([r["pred_net"] for r in rr])
    re = np.array([r["realised"] for r in rr])
    return {"n": len(rr), "pred_net": float(np.nanmean(pr)),
            "pred_voi": float(np.nanmean([r["pred_voi"] for r in rr])),
            "realised": boot(re), "paid": float((re > 0).mean()),
            "ratio_pred_real": float(np.nanmean(pr) / re.mean()) if re.mean() else None,
            "corr": float(np.corrcoef(pr, re)[0, 1]) if len(rr) > 2 else None}


def main():
    rows = [json.loads(l) for l in P.read_text().splitlines() if l.strip()]
    out = {}
    groups = defaultdict(list)
    for r in rows:
        it = "v2" if r["method"] == "cauprobe-v2" else "v1"
        b = "B%d" % r["combo"][1]
        mm = "m=1" if r["m"] == 1 else "m>1"
        for key in (f"{it}|all", f"{it}|{r['switch']}", f"{it}|{b}", f"{it}|{mm}",
                    f"{it}|{r['switch']}|{b}", f"{it}|first" if r["t"] == min(
                        x["t"] for x in rows if x["method"] == r["method"]
                        and x["seed"] == r["seed"] and x["switch"] == r["switch"]) else None):
            if key:
                groups[key].append(r)
    for k, v in sorted(groups.items()):
        out[k] = stats(v)
        s = out[k]
        print(f"{k:16s} n={s['n']:3d} pred_net {s['pred_net']:7.2f} realised "
              f"{s['realised'][0]:7.2f} [{s['realised'][1]:6.2f},{s['realised'][2]:6.2f}] "
              f"paid {s['paid']:.2f} corr {s['corr']}")
    # discrimination: does the predicted value rank the probes that paid above
    # those that did not? (AUROC of predicted net value for realised > 0)
    from sklearn.metrics import roc_auc_score
    for it, m in (("v1", "cauprobe"), ("v2", "cauprobe-v2")):
        rr = [r for r in rows if r["method"] == m]
        y = np.array([r["realised"] > 0 for r in rr])
        s = np.array([r["pred_net"] for r in rr])
        rng = np.random.default_rng(1)
        bs = []
        for _ in range(5000):
            i = rng.integers(0, len(rr), len(rr))
            if 0 < y[i].sum() < len(i):
                bs.append(roc_auc_score(y[i], s[i]))
        out[f"{it}|auroc"] = {"auroc": float(roc_auc_score(y, s)),
                              "ci": (float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)))}
        print(it, "AUROC", out[f"{it}|auroc"])
    (CP_RESULTS / "test2" / "probe_audit_summary.json").write_text(json.dumps(out, indent=1))
    # table of strata for the paper
    strata = [("all decisions", "all"), ("Wi-Fi costs", "wifi"), ("LPWAN costs", "lpwan"),
              ("edge-bound plants (B0)", "B0"), ("WAN-bound plants (B1)", "B1"),
              ("LPWAN, WAN-bound", "lpwan|B1"), ("one-step probes", "m=1"),
              ("multi-step probes", "m>1")]
    L = [r"\begin{table}[t]", r"\centering",
         r"\caption{Paired audit of individual probe decisions, by stratum: number "
         r"of decisions, mean predicted net value, mean realised value [95\% "
         r"bootstrap CI] and the share that paid. Last row: AUROC of the "
         r"predicted value for telling probes that paid from probes that did not.}",
         r"\label{tab:audit}", r"\footnotesize", r"\setlength{\tabcolsep}{2.5pt}",
         r"\begin{tabular}{lrrrrrr}", r"\toprule",
         r"& \multicolumn{3}{c}{\ours-1} & \multicolumn{3}{c}{\ours-2} \\",
         r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
         r"Stratum & $n$ & predicted & realised [95\% CI], paid & $n$ & predicted & realised [95\% CI], paid \\",
         r"\midrule"]
    for lab, key in strata:
        cells = []
        for it in ("v1", "v2"):
            s = out.get(f"{it}|{key}")
            if s is None:
                cells += ["--"] * 3
                continue
            m, lo, hi = s["realised"]
            cells += [str(s["n"]), f"{s['pred_net']:.1f}",
                      f"{m:+.1f} [{lo:+.1f}, {hi:+.1f}], {100 * s['paid']:.0f}\\%"]
        L.append(lab + " & " + " & ".join(cells) + r" \\")
    a1, a2 = out["v1|auroc"], out["v2|auroc"]
    L += [r"\midrule",
          f"AUROC (paid vs.\\ not) & \\multicolumn{{3}}{{r}}{{{a1['auroc']:.2f} "
          f"[{a1['ci'][0]:.2f}, {a1['ci'][1]:.2f}]}} & \\multicolumn{{3}}{{r}}"
          f"{{{a2['auroc']:.2f} [{a2['ci'][0]:.2f}, {a2['ci'][1]:.2f}]}} \\\\",
          r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    (ROOT / "paper" / "tables_cp_audit.tex").write_text("\n".join(L))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7), sharey=True)
        for ax, it, title in zip(axes, ("cauprobe", "cauprobe-v2"),
                                 ("CauProbe-1", "CauProbe-2")):
            for sw, mk in (("wifi", "o"), ("lpwan", "^")):
                for b, col in ((0, "#1f5fa8"), (1, "#d1781e")):
                    rr = [r for r in rows if r["method"] == it and r["switch"] == sw
                          and r["combo"][1] == b]
                    if rr:
                        ax.scatter([r["pred_net"] for r in rr], [r["realised"] for r in rr],
                                   marker=mk, s=16, color=col, alpha=0.75,
                                   label=f"{'edge' if b == 0 else 'WAN'}-bound, {sw}")
            lim = max(1.0, max(abs(r["pred_net"]) for r in rows))
            ax.axhline(0, color="k", lw=0.6, ls=":")
            ax.plot([0, lim], [0, lim], color="#888", lw=0.8, ls="--")
            ax.set_xscale("symlog", linthresh=1)
            ax.set_yscale("symlog", linthresh=1)
            ax.set_xlabel("predicted net value of the probe")
            ax.set_title(title, fontsize=8.5)
        axes[0].set_ylabel("realised value (paired, to day 14)")
        axes[0].legend(fontsize=6, frameon=False)
        figs = ROOT / "paper" / "figs"
        fig.savefig(figs / "cp_fig11_audit.pdf", bbox_inches="tight")
        fig.savefig(figs / "cp_fig11_audit.png", dpi=200, bbox_inches="tight")
        print("wrote cp_fig11_audit")
    except Exception as e:
        print("figure skipped:", e)


if __name__ == "__main__":
    main()

"""Every figure of the paper, from the raw JSON (missing inputs are skipped)."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

from cp_common import CP_RESULTS, ROOT, testrows  # noqa: E402

FIGS = ROOT / "paper" / "figs"
FIGS.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 9, "font.family": "serif",
                     "axes.spines.top": False, "axes.spines.right": False,
                     "pdf.fonttype": 42})
COL = {"cauprobe": "#1f5fa8", "cauprobe-noask": "#5b8fd6",
       "cauprobe-noprobe": "#8fb7e8", "greedy": "#7f7f7f",
       "thompson": "#d1781e", "ids": "#9c5bb5", "identify": "#2e9b5e",
       "log-naive": "#b5b5b5", "oracle": "#000000", "linucb": "#c23b3b",
       "rule": "#8c564b", "advice-plan": "#e377c2"}
NAME = {"cauprobe": "CauProbe", "cauprobe-noask": "CauProbe w/o expert",
        "cauprobe-noprobe": "expert + greedy", "greedy": "greedy",
        "thompson": "Thompson", "ids": "IDS", "identify": "identify-then-act",
        "log-naive": "greedy + logs", "linucb": "LinUCB", "rule": "rule-based",
        "advice-plan": "advice-as-plan"}


def load_jsonl(p):
    p = Path(p)
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def save(fig, name):
    fig.savefig(FIGS / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIGS / f"{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


# ------------------------------------------------------------------ fig 1
def fig_architecture():
    import cp_fig_arch
    cp_fig_arch.draw(save)


# ------------------------------------------------------------------ fig 2
def fig_learning_curve():
    rows = testrows(load_jsonl(CP_RESULTS / "e1_test_main.jsonl"))
    if not rows:
        return
    orc = {(r["plant"], r["seed"]): r for r in rows if r["method"] == "oracle"}
    methods = ["cauprobe", "cauprobe-noprobe", "cauprobe-noask", "greedy",
               "thompson", "ids", "identify"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7))
    cp = {(r["plant"], r["seed"]): r for r in rows if r["method"] == "cauprobe"}
    for m in methods:
        per, rel = defaultdict(list), defaultdict(list)
        for r in rows:
            if r["method"] != m:
                continue
            o = orc.get((r["plant"], r["seed"]))
            if o is None:
                continue
            g = np.array(o["return_per_day"]) - np.array(r["return_per_day"])
            per[r["seed"] // 10].append(g)
            c = cp.get((r["plant"], r["seed"]))
            if c is not None:
                rel[r["seed"] // 10].append(np.array(c["return_per_day"])
                                            - np.array(r["return_per_day"]))
        if not per:
            continue
        G = np.array([np.mean(v, 0) for v in per.values()])     # plants x days
        d = np.arange(1, G.shape[1] + 1)
        axes[0].plot(d, G.mean(0), color=COL[m], label=NAME[m], lw=1.3)
        # right: cumulative return lost relative to CauProbe, paired by plant
        R = np.cumsum(np.array([np.mean(v, 0) for v in rel.values()]), 1)
        mu = R.mean(0)
        se = R.std(0, ddof=1) / np.sqrt(len(R)) if m != "cauprobe" else 0 * mu
        axes[1].plot(d, mu, color=COL[m], lw=1.3)
        axes[1].fill_between(d, mu - 1.96 * se, mu + 1.96 * se, color=COL[m],
                             alpha=0.12, lw=0)
    for a in axes:
        a.axhline(0, color="k", lw=0.6, ls=":")
        a.set_xlabel("day of commissioning")
        a.set_xticks([1, 4, 7, 10, 14])
    axes[0].set_ylabel("daily gap to reference")
    axes[1].set_ylabel("cum. loss vs CauProbe")
    axes[0].legend(fontsize=6.5, frameon=False, ncol=1)
    save(fig, "cp_fig2_learning")


# ------------------------------------------------------------------ fig 3
def fig_scenarios():
    scen = ["nominal", "noisy", "lpwan", "noisy_lpwan", "poor_docs", "shift"]
    methods = ["cauprobe-noprobe", "cauprobe", "cauprobe-noask",
               "thompson", "ids", "identify"]
    data = {}
    for sc in scen:
        rows = testrows(load_jsonl(CP_RESULTS / f"e1_test_{'main' if sc == 'nominal' else sc}.jsonl"))
        if not rows:
            continue
        # return gained over greedy control on the same plant and noise stream
        base = {(r["plant"], r["seed"]): r for r in rows if r["method"] == "greedy"}
        for m in methods:
            per = defaultdict(list)
            for r in rows:
                b = base.get((r["plant"], r["seed"]))
                if r["method"] == m and b is not None:
                    per[r["seed"] // 10].append(r["return"] - b["return"])
            if per:
                v = np.array([np.mean(x) for x in per.values()])
                data[(sc, m)] = (v.mean(), 1.96 * v.std(ddof=1) / np.sqrt(len(v)))
    if not data:
        return
    present = [s for s in scen if any((s, m) in data for m in methods)]
    fig, ax = plt.subplots(figsize=(7.2, 2.8))
    w = 0.8 / len(methods)
    for j, m in enumerate(methods):
        xs, ys, es = [], [], []
        for i, sc in enumerate(present):
            if (sc, m) in data:
                xs.append(i + (j - len(methods) / 2 + 0.5) * w)
                ys.append(data[(sc, m)][0])
                es.append(data[(sc, m)][1])
        ax.errorbar(xs, ys, yerr=es, fmt="o", ms=3.2, color=COL[m], lw=0.8,
                    capsize=1.5, label=NAME[m])
    ax.set_xticks(range(len(present)))
    ax.set_xticklabels([s.replace("_", "+").replace("lpwan", "LPWAN")
                        .replace("poor+docs", "poor docs").replace("shift",
                        "struct. change") for s in present])
    ax.axhline(0, color="k", lw=0.6, ls=":")
    ax.set_ylabel("return gained over greedy")
    ax.legend(fontsize=6.3, frameon=False, ncol=3, loc="lower left")
    save(fig, "cp_fig3_scenarios")


if __name__ == "__main__":
    import cp_figs_extra as XF
    fig_architecture()
    fig_learning_curve()
    fig_scenarios()
    XF.fig_ablation(save)
    XF.fig_tail(save)
    XF.fig_probe_timing(save)
    XF.fig_price(save)
    XF.fig_boptest(save)
    import cp_iter2 as I2
    I2.fig_scenarios(save, I2.scen_summary())
    I2.fig_price(save, I2.price_rows(), I2.voi_ask())

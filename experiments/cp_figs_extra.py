"""Additional figures: ablations, probe timing, price sweep, BOPTEST, expert."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from cp_common import CP_RESULTS, testrows  # noqa: E402


def load_jsonl(p):
    p = Path(p)
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


ABL = [("cauprobe", "CauProbe (full)"),
       ("cauprobe-particle", "VOI over particles"),
       ("cauprobe-nodur", "no durability correction"),
       ("cauprobe-nokappa", "no self-calibration"),
       ("cauprobe-h1", "horizon 1 day"),
       ("cauprobe-h7", "horizon 7 days"),
       ("cauprobe-noask", "no expert"),
       ("cauprobe-noprobe", "no probing"),
       ("cauprobe-expert:slm-0.5b-lora-uncal", "expert uncalibrated"),
       ("cauprobe-expert:tfidf-lr", "expert: TF-IDF"),
       ("cauprobe-expert:zeroshot-1.5b", "expert: zero-shot 1.5B"),
       ("cauprobe-nosafety", "no safety filter")]


def fig_ablation(save):
    rows = testrows(load_jsonl(CP_RESULTS / "e1_test_main.jsonl"))
    if not rows:
        return
    by = defaultdict(dict)
    for r in rows:
        by[r["method"]][(r["plant"], r["seed"])] = r
    ref = by.get("cauprobe", {})
    names, means, los, his, probes = [], [], [], [], []
    rng = np.random.default_rng(0)
    for m, lab in ABL:
        if m not in by:
            continue
        per = defaultdict(list)
        for k, r in by[m].items():
            if k in ref:
                per[k[1] // 10].append(ref[k]["return"] - r["return"])
        if not per:
            continue
        d = np.array([np.mean(v) for v in per.values()])
        b = rng.choice(d, (5000, len(d))).mean(1)
        names.append(lab)
        means.append(d.mean())
        los.append(d.mean() - np.percentile(b, 2.5))
        his.append(np.percentile(b, 97.5) - d.mean())
        probes.append(np.mean([r["probe_steps"] for r in by[m].values()]))
    fig, ax = plt.subplots(figsize=(7.0, 3.0))
    y = np.arange(len(names))[::-1]
    ax.barh(y, means, xerr=[los, his], color=["#1f5fa8" if n.startswith("CauProbe")
                                               else "#9bb7d9" for n in names],
            height=0.6, error_kw=dict(lw=0.8, capsize=2))
    ax.set_yticks(y)
    ax.set_yticklabels([f"{n} ({p:.0f} probes)" for n, p in zip(names, probes)],
                       fontsize=7.5)
    ax.axvline(0, color="k", lw=0.6)
    ax.set_xlabel("14-day return lost relative to full CauProbe (95% CI)")
    save(fig, "cp_fig4_ablation")


def fig_tail(save):
    """Left: per-plant return lost relative to CauProbe, sorted (the tail is
    where methods differ).
    Right: CauProbe minus greedy return per network structure (A,B,C), with
    95% bootstrap CIs over plants -- where buying knowledge pays and where it
    costs."""
    rows = testrows(load_jsonl(CP_RESULTS / "e1_test_main.jsonl"))
    if not rows:
        return
    by = defaultdict(dict)
    for r in rows:
        by[r["method"]][(r["plant"], r["seed"])] = r
    orc = by.get("oracle", {})
    col = {"greedy": "#7f7f7f", "cauprobe-noprobe": "#8fb7e8",
           "thompson": "#d1781e"}
    lab = {"cauprobe-noprobe": "expert + greedy", "greedy": "greedy",
           "thompson": "Thompson"}
    ref = by.get("cauprobe", {})
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6),
                             gridspec_kw={"width_ratios": [1, 1.25]})
    for m in col:
        per = defaultdict(list)
        for k, r in by.get(m, {}).items():
            if k in ref:
                per[k[1] // 10].append(ref[k]["return"] - r["return"])
        if not per:
            continue
        g = np.sort([np.mean(v) for v in per.values()])
        q = (np.arange(len(g)) + 0.5) / len(g)
        axes[0].plot(q, g, color=col[m], lw=1.3, label=lab[m])
    axes[0].axhline(0, color="k", lw=0.6, ls=":")
    axes[0].set_xlabel("plant quantile")
    axes[0].set_ylabel("return lost vs CauProbe")
    axes[0].legend(fontsize=6.5, frameon=False, loc="upper left")
    st = defaultdict(lambda: defaultdict(list))
    for k, c in by.get("cauprobe", {}).items():
        g = by.get("greedy", {}).get(k)
        if g is not None:
            s = "A%dB%dC%d" % tuple(c["combo"][:3])
            st[s][k[1] // 10].append(c["return"] - g["return"])
    rng = np.random.default_rng(0)
    names = sorted(st)
    mu, lo, hi = [], [], []
    for s in names:
        d = np.array([np.mean(v) for v in st[s].values()])
        b = rng.choice(d, (5000, len(d))).mean(1)
        mu.append(d.mean())
        lo.append(d.mean() - np.percentile(b, 2.5))
        hi.append(np.percentile(b, 97.5) - d.mean())
    x = np.arange(len(names))
    axes[1].bar(x, mu, yerr=[lo, hi], color=["#1f5fa8" if v > 0 else "#c9c9c9"
                                             for v in mu],
                error_kw=dict(lw=0.8, capsize=2), width=0.65)
    axes[1].axhline(0, color="k", lw=0.6)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names, fontsize=6.8, rotation=30)
    axes[1].set_ylabel("CauProbe $-$ greedy return")
    save(fig, "cp_fig8_tail")


def fig_probe_timing(save):
    p = CP_RESULTS / "probe_timing.json"
    if not p.exists():
        return
    res = json.loads(p.read_text())
    P = np.array([r["probes_per_day"] for r in res])
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.4))
    d = np.arange(1, P.shape[1] + 1)
    axes[0].bar(d, P.mean(0), color="#1f5fa8")
    axes[0].set_xlabel("day")
    axes[0].set_ylabel("probe steps per plant")
    ks = [np.array(r["kappa_trace"]) for r in res if r["kappa_trace"]]
    for k in ks:
        axes[1].plot(k[:, 0] / 288.0, k[:, 3], color="#1f5fa8", alpha=0.35, lw=0.8)
    axes[1].set_xlabel("day")
    axes[1].set_ylabel(r"self-calibration $\kappa$")
    axes[1].set_ylim(0, 1.05)
    save(fig, "cp_fig5_timing")


def fig_price(save):
    rows = load_jsonl(CP_RESULTS / "e7_price.jsonl")
    if not rows:
        return
    import cp_tables_extra as X
    st = X.price_rows(CP_RESULTS / "e7_price.jsonl")
    prices = sorted({k[2] for k in st if k[1] == "cauprobe"})
    # per-call prices implied by Table edge at the IT-energy price (0.4/kJ):
    # measured GPU joules, roofline joules, and published J/token x answer tokens
    band = None
    ep = CP_RESULTS / "edge.json"
    if ep.exists():
        e = json.loads(ep.read_text())
        js = [e["host_gpu"]["j_per_call"]] if "j_per_call" in e.get("host_gpu", {}) else []
        js += [row["q4_k_m"]["j_per_call"] for row in e["roofline"].values()]
        js += [j * e["answer_tokens"] for _, _, _, j in X.PUBLISHED if j is not None]
        band = (0.4 * min(js) / 1000.0, 0.4 * max(js) / 1000.0)
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6), sharey=True)
    for ax, sw, title in zip(axes, ("wifi", "lpwan"),
                             ("Wi-Fi reconfiguration", "LPWAN reconfiguration")):
        for m, c, lab in (("cauprobe", "#1f5fa8", "CauProbe"),
                          ("ask-always", "#d1781e", "ask-always")):
            g = [st[(sw, m, p)]["gain"] for p in prices if (sw, m, p) in st]
            if not g:
                continue
            g = np.array(g)
            ax.plot(prices[:len(g)], g[:, 0], "o-", color=c, ms=3, lw=1.2,
                    label=lab)
            ax.fill_between(prices[:len(g)], g[:, 1], g[:, 2], color=c,
                            alpha=0.12, lw=0)
        po = st.get((sw, "probe-only", 0.0))
        if po:
            ax.axhline(po["gain"][0], color="#2e9b5e", ls="--", lw=1,
                       label="probe-only")
        ax.axhline(0, color="k", lw=0.6, ls=":")
        if band is not None:
            ax.axvspan(*band, color="#999999", alpha=0.18, lw=0,
                       label="device range" if ax is axes[0] else None)
        ax.set_xscale("log")
        ax.set_xlabel("price of one expert call (reward units)")
        ax.set_title(title, fontsize=8.5)
    axes[0].set_ylabel("gain over greedy (7 days)")
    axes[0].legend(fontsize=6.5, frameon=False)
    save(fig, "cp_fig6_price")


def fig_boptest(save):
    files = [("v1prior", "o"), ("v2prior", "s"), ("v3ctrl", "^"), ("v4thdet", "D")]
    col = {"boptest-baseline": "#000000", "incumbent": "#8c564b",
           "nominal": "#7f7f7f", "greedy": "#9bb7d9", "thompson": "#d1781e",
           "cauprobe": "#1f5fa8", "cauprobe+expert": "#2e9b5e"}
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    seen = set()
    for f, mk in files:
        rows = load_jsonl(CP_RESULTS / f"e6_boptest_{f}.jsonl")
        by = defaultdict(list)
        for r in rows:
            by[r["method"]].append(r)
        for m, v in by.items():
            e = np.mean([x["energy_kwh"] for x in v])
            t = np.mean([x["tdis_kh"] for x in v])
            lab = m if m not in seen else None
            seen.add(m)
            ax.scatter(e, t, marker=mk, color=col.get(m, "#555"), s=22, label=lab,
                       alpha=0.85)
    ax.set_xlabel("energy per week (kWh)")
    ax.set_ylabel("occupied discomfort per week (K h)")
    ax.legend(fontsize=6, frameon=False, loc="upper right")
    save(fig, "cp_fig7_boptest")

"""Second iteration of CauProbe: tables, figures and numbers.

The second iteration (method `cauprobe-v2`) was developed after the first test
runs, on the dev and validation plants only, then frozen and run once on a
second set of fresh plants (cp_common.TEST2_PLANTS, OOL_TEST2_PLANTS) together
with the frozen first iteration and its rivals. Everything here reads
results/cauprobe/test2/ (or, with --val, results/cauprobe/validation_v2/, which
is what the development decisions were made on).

Called by cp_make_tables.py (tables -> paper/tables_cp_iter2.tex, summary),
cp_numbers.py (numbers quoted in the text) and cp_make_figures.py (figures);
can also be run on its own:  cp_iter2.py [--val]
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from cp_common import CP_RESULTS, ROOT
from cp_summarize import boot_ci, holm, paired_p

T2 = CP_RESULTS / "test2"
VAL2 = CP_RESULTS / "validation_v2"
SCEN = ["nominal", "noisy", "lpwan", "noisy_lpwan", "poor_docs", "shift"]
SCEN_LABEL = {"nominal": "nominal", "noisy": "noisy", "lpwan": "LPWAN",
              "noisy_lpwan": "noisy+LPWAN", "poor_docs": "poor docs",
              "shift": r"struct.\ change"}
LEARNERS = ["cauprobe-v2", "cauprobe", "cauprobe-noprobe", "cauprobe-v2-noask",
            "thompson", "ids", "identify"]
LABEL = {"cauprobe-v2": r"\ours-2", "cauprobe": r"\ours-1 (first iteration)",
         "cauprobe-noprobe": "expert + greedy", "cauprobe-v2-noask":
         r"\ours-2 without expert", "greedy": "greedy (same belief)",
         "thompson": "Thompson sampling", "ids": "IDS",
         "identify": "identify-then-act", "rule": "rule-based (incumbent)",
         "oracle": "known-plant reference"}
# the comparisons the second iteration was built to win, Holm-corrected
# together across all six scenarios
KEY = [("cauprobe-v2", "greedy"), ("cauprobe-v2", "cauprobe-noprobe"),
       ("cauprobe-v2", "cauprobe")]


def _load(path, seeds=None):
    p = Path(path)
    if not p.exists():
        return []
    rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    rows = [r for r in rows if "error" not in r]
    if seeds is not None:
        rows = [r for r in rows if r["seed"] // 10 in seeds]
    return rows


def data_dir(val=False):
    return VAL2 if val else T2


def seeds(val=False):
    if val:
        from cp_common import HELDOUT_PLANTS
        return {s for s, _ in HELDOUT_PLANTS}
    from cp_common import TEST2_SEEDS
    return TEST2_SEEDS


def rows(sc, val=False):
    return _load(data_dir(val) / f"e1_{sc}.jsonl", seeds(val))


def by_method(rr):
    b = defaultdict(dict)
    for r in rr:
        b[r["method"]].setdefault(r["seed"], r)
    return b


def pdiff(b, m1, m2, key=lambda r: r["return"]):
    """Per-plant mean difference m1 - m2 (paired on plant and noise)."""
    if m1 not in b or m2 not in b:
        return None
    acc = defaultdict(list)
    for s, r in b[m1].items():
        if s in b[m2]:
            acc[s // 10].append(key(r) - key(b[m2][s]))
    d = np.array([np.mean(v) for v in acc.values()])
    if len(d) < 2:
        return None
    return {"ci": boot_ci(d), "p": paired_p(d), "n": len(d),
            "pp": {k: float(np.mean(v)) for k, v in acc.items()}}


def wrong(r):
    fp = r.get("final_factor_post")
    if not fp:
        return None
    c = list(r["combo"])
    if r.get("scenario") == "shift":
        c[0] = 1 - c[0]
    return any((fp[F] > 0.5) != bool(c[i]) for i, F in enumerate("ABC"))


def scen_summary(val=False):
    out = {}
    for sc in SCEN:
        b = by_method(rows(sc, val))
        if "greedy" not in b:
            continue
        st = {}
        for m in b:
            rr = list(b[m].values())
            d = {"n_runs": len(rr),
                 "probes": float(np.mean([r["probe_steps"] for r in rr])),
                 "asks": float(np.mean([len(r["asks"]) for r in rr])),
                 "switch_cost": float(np.mean([r["switch_cost"] for r in rr])),
                 "misses": float(np.mean([r["viol_net"] + r["viol_th"] for r in rr]))}
            w = [wrong(r) for r in rr]
            if all(x is not None for x in w):
                d["wrong"] = float(np.mean(w))
                d["n_wrong"] = int(np.sum(w))
            g = pdiff(b, m, "greedy")
            if g and m != "greedy":
                d["vs_greedy"] = {k: g[k] for k in ("ci", "p", "n")}
            o = pdiff(b, "oracle", m)
            if o and m != "oracle":
                d["gap"] = o["ci"]
            st[m] = d
        cmp = {}
        for a, c in KEY + [("cauprobe-noprobe", "greedy"), ("cauprobe", "greedy"),
                           ("cauprobe-v2", "cauprobe-v2-noask"),
                           ("cauprobe-v2-noask", "greedy"),
                           ("cauprobe", "cauprobe-noprobe")]:
            x = pdiff(b, a, c)
            if x:
                cmp[f"{a}|{c}"] = {k: x[k] for k in ("ci", "p", "n")}
        out[sc] = {"methods": st, "cmp": cmp}
    # Holm across scenarios for the key comparisons
    keys = [(sc, f"{a}|{c}") for sc in out for a, c in KEY
            if f"{a}|{c}" in out[sc]["cmp"]]
    if keys:
        adj = holm([out[sc]["cmp"][k]["p"] for sc, k in keys])
        for (sc, k), pa in zip(keys, adj):
            out[sc]["cmp"][k]["p_holm"] = float(pa)
    return out


def lockin(val=False):
    """Nominal scenario: where greedy control ends on a wrong mechanism, what
    do the two iterations do (by processing bottleneck B of the plant)."""
    b = by_method(rows("nominal", val))
    if "greedy" not in b:
        return {}
    out = {}
    for m in ("cauprobe", "cauprobe-v2", "cauprobe-noprobe"):
        if m not in b:
            continue
        for bb, name in ((0, "edge_bound"), (1, "wan_bound")):
            ss = [s for s, r in b["greedy"].items() if wrong(r) and r["combo"][1] == bb
                  and s in b[m]]
            out[f"{m}|{name}"] = {
                "n": len(ss),
                "gain": float(np.mean([b[m][s]["return"] - b["greedy"][s]["return"]
                                       for s in ss])) if ss else None,
                "probes": float(np.mean([b[m][s]["probe_steps"] for s in ss])) if ss else None,
                "still_wrong": int(sum(bool(wrong(b[m][s])) for s in ss))}
        # probing on WAN-bound plants overall
        for bb, name in ((0, "edge_bound"), (1, "wan_bound")):
            rr = [r for r in b[m].values() if r["combo"][1] == bb]
            out[f"{m}|probes_{name}"] = float(np.mean([r["probe_steps"] for r in rr]))
    return out


def ool_summary(val=False):
    """Out-of-library plants, per kind: return over greedy and over the
    incumbent, probe steps, deadline misses, detector alarms."""
    if val:
        from cp_common import OOL_VAL_PLANTS as OP
    else:
        from cp_common import OOL_TEST2_PLANTS as OP
    rr = _load(data_dir(val) / "e1_ool.jsonl", {s for s, _, _ in OP})
    b = by_method(rr)
    out = {}
    for kind in ("tsch", "congested_wan", "broker_fault"):
        bk = {m: {s: r for s, r in d.items() if r["ool"] == kind} for m, d in b.items()}
        if not bk.get("greedy"):
            continue
        e = {}
        for m in bk:
            if not bk[m]:
                continue
            rr_ = list(bk[m].values())
            x = {"probes": float(np.mean([r["probe_steps"] for r in rr_])),
                 "misses": float(np.mean([r["viol_net"] for r in rr_])),
                 "alarm": float(np.mean([r.get("misspec_alarms", 0) > 0 for r in rr_])),
                 "fallback": float(np.mean([r["fallback_steps"] for r in rr_]))}
            for ref in ("greedy", "rule"):
                d = pdiff(bk, m, ref) if m != ref else None
                if d:
                    x[f"vs_{ref}"] = {k: d[k] for k in ("ci", "p", "n")}
            e[m] = x
        out[kind] = e
    return out


def price_rows(val=False):
    rr = _load(data_dir(val) / "e7_price.jsonl")
    base = {(r["seed"], r["switch"]): r for r in rr if r["method"] == "greedy"}
    acc = defaultdict(list)
    for r in rr:
        g = base.get((r["seed"], r["switch"]))
        if g is None or r["method"] == "greedy":
            continue
        acc[(r["switch"], r["method"], r["price"])].append(
            (r["return"] - g["return"], len(r["asks"]), r["probe_steps"]))
    out = {}
    for k, v in acc.items():
        g = np.array([x[0] for x in v])
        out[k] = {"gain": boot_ci(g), "asks": float(np.mean([x[1] for x in v])),
                  "probes": float(np.mean([x[2] for x in v])), "n": len(v)}
    return out


def voi_ask(val=False):
    p = data_dir(val) / "voi_ask.json"
    return json.loads(p.read_text()) if p.exists() else None


# ------------------------------------------------------------------ tables
def _cell(x, nd=1):
    if x is None:
        return "--"
    return f"{x['ci'][0]:+.{nd}f}"


def table_scenarios(S):
    cols = [sc for sc in SCEN if sc in S]
    if not cols:
        return ""
    L = [r"\begin{table*}[t]", r"\centering",
         r"\caption{Second iteration on the second fresh test set (48 plants, "
         r"14 days; two runs per plant in the nominal scenario, one in the "
         r"others). Top: return gained over greedy control on the same plant "
         r"and noise stream (mean over plants; higher is better), with the 95\% "
         r"bootstrap interval for the three comparisons \ours-2 was built to "
         r"win; $^{*}$/$^{\dagger}$: \ours-2 better/worse than the named "
         r"controller at $p_{\mathrm{Holm}}<0.05$ (Holm over the 18 key tests). "
         r"Bottom: probe steps per run and share of runs ending on a wrong "
         r"network mechanism.}",
         r"\label{tab:iter2}", r"\footnotesize", r"\setlength{\tabcolsep}{3pt}",
         r"\begin{tabular}{l" + "r" * len(cols) + "}", r"\toprule",
         " & " + " & ".join(SCEN_LABEL[c] for c in cols) + r" \\", r"\midrule",
         r"\multicolumn{%d}{l}{\emph{Return gained over greedy control}} \\"
         % (len(cols) + 1)]
    for m in LEARNERS:
        if not any(m in S[c]["methods"] for c in cols):
            continue
        cells = [_cell(S[c]["methods"].get(m, {}).get("vs_greedy")) for c in cols]
        L.append(LABEL[m] + " & " + " & ".join(cells) + r" \\")
    L.append(r"\midrule")
    L.append(r"\multicolumn{%d}{l}{\emph{\ours-2 minus \ldots\ (95\%% CI)}} \\"
             % (len(cols) + 1))
    for a, c in KEY:
        cells = []
        for sc in cols:
            x = S[sc]["cmp"].get(f"{a}|{c}")
            if x is None:
                cells.append("--")
                continue
            m_, lo, hi = x["ci"]
            mark = ""
            if x.get("p_holm", 1.0) < 0.05:
                mark = r"$^{*}$" if m_ > 0 else r"$^{\dagger}$"
            cells.append(f"{m_:+.1f}{mark} [{lo:+.0f}, {hi:+.0f}]")
        L.append(LABEL[c] + " & " + " & ".join(cells) + r" \\")
    L.append(r"\midrule")
    L.append(r"\multicolumn{%d}{l}{\emph{Probe steps per run}} \\" % (len(cols) + 1))
    for m in ("cauprobe-v2", "cauprobe", "cauprobe-v2-noask"):
        if not any(m in S[c]["methods"] for c in cols):
            continue
        cells = [f"{S[c]['methods'][m]['probes']:.0f}" if m in S[c]["methods"]
                 else "--" for c in cols]
        L.append(LABEL[m] + " & " + " & ".join(cells) + r" \\")
    L.append(r"\multicolumn{%d}{l}{\emph{Runs ending on a wrong mechanism}} \\"
             % (len(cols) + 1))
    for m in ("cauprobe-v2", "cauprobe", "cauprobe-noprobe", "greedy"):
        cells = []
        for c in cols:
            x = S[c]["methods"].get(m, {})
            cells.append(f"{100 * x['wrong']:.0f}\\%" if "wrong" in x else "--")
        L.append(LABEL[m] + " & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""]
    return "\n".join(L)


def table_price(P, V=None):
    if not P:
        return ""
    prices = sorted({k[2] for k in P if k[1] == "cauprobe-v2"})
    L = [r"\begin{table*}[t]", r"\centering",
         r"\caption{Price sweep on 32 plants of the second test set (7 days): "
         r"gain over greedy control on the same plant and noise stream, mean "
         r"[95\% CI], and the share of runs in which the notes were read. "
         r"Ask-always reads whatever it costs and never probes.}",
         r"\label{tab:iter2price}", r"\footnotesize", r"\setlength{\tabcolsep}{2.5pt}",
         r"\begin{tabular}{lrrrrrrrrrr}", r"\toprule",
         r"& \multicolumn{5}{c}{Wi-Fi reconfiguration} & "
         r"\multicolumn{5}{c}{LPWAN reconfiguration} \\",
         r"\cmidrule(lr){2-6}\cmidrule(lr){7-11}",
         r"price/call & \ours-2 & reads & \ours-1 & reads & ask-always & "
         r"\ours-2 & reads & \ours-1 & reads & ask-always \\", r"\midrule"]
    for p in prices:
        cells = []
        for sw in ("wifi", "lpwan"):
            v2 = P.get((sw, "cauprobe-v2", p))
            v1 = P.get((sw, "cauprobe", p))
            aa = P.get((sw, "ask-always", p))
            cells += [f"{v2['gain'][0]:+.1f} [{v2['gain'][1]:+.0f}, {v2['gain'][2]:+.0f}]"
                      if v2 else "--", f"{v2['asks']:.2f}" if v2 else "--",
                      f"{v1['gain'][0]:+.1f}" if v1 else "--",
                      f"{v1['asks']:.2f}" if v1 else "--",
                      f"{aa['gain'][0]:+.1f}" if aa else "--"]
        L.append(f"{p:g} & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""]
    return "\n".join(L)


# ------------------------------------------------------------------ figures
def fig_scenarios(save, S):
    import matplotlib.pyplot as plt
    cols = [sc for sc in SCEN if sc in S]
    if not cols:
        return
    ms = [("cauprobe-noprobe", "#8fb7e8", "expert + greedy"),
          ("cauprobe", "#7f7f7f", "CauProbe-1"),
          ("cauprobe-v2", "#1f5fa8", "CauProbe-2"),
          ("cauprobe-v2-noask", "#5b8fd6", "CauProbe-2 w/o expert"),
          ("thompson", "#d1781e", "Thompson"), ("ids", "#9c5bb5", "IDS"),
          ("identify", "#2e9b5e", "identify-then-act")]
    ms = [x for x in ms if any(x[0] in S[c]["methods"] for c in cols)]
    fig, ax = plt.subplots(figsize=(7.2, 2.8))
    w = 0.8 / len(ms)
    for j, (m, col, lab) in enumerate(ms):
        xs, ys, lo, hi = [], [], [], []
        for i, sc in enumerate(cols):
            x = S[sc]["methods"].get(m, {}).get("vs_greedy")
            if x is None:
                continue
            xs.append(i + (j - len(ms) / 2 + 0.5) * w)
            ys.append(x["ci"][0])
            lo.append(x["ci"][0] - x["ci"][1])
            hi.append(x["ci"][2] - x["ci"][0])
        ax.errorbar(xs, ys, yerr=[lo, hi], fmt="o", ms=3.2, color=col, lw=0.8,
                    capsize=1.5, label=lab)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([SCEN_LABEL[c].replace(r"\ ", " ") for c in cols])
    ax.axhline(0, color="k", lw=0.6, ls=":")
    ax.set_ylabel("return gained over greedy")
    ax.legend(fontsize=6.3, frameon=False, ncol=4, loc="lower left")
    save(fig, "cp_fig9_iter2_scenarios")


def fig_price(save, P, V=None):
    import matplotlib.pyplot as plt
    if not P:
        return
    prices = sorted({k[2] for k in P if k[1] == "cauprobe-v2"})
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6), sharey=True)
    for ax, sw, title in zip(axes, ("wifi", "lpwan"),
                             ("Wi-Fi reconfiguration", "LPWAN reconfiguration")):
        for m, c, lab in (("cauprobe-v2", "#1f5fa8", "CauProbe-2"),
                          ("cauprobe", "#7f7f7f", "CauProbe-1"),
                          ("ask-always", "#d1781e", "ask-always")):
            g = np.array([P[(sw, m, p)]["gain"] for p in prices if (sw, m, p) in P])
            if not len(g):
                continue
            xs = [p for p in prices if (sw, m, p) in P]
            ax.plot(xs, g[:, 0], "o-", color=c, ms=3, lw=1.2, label=lab)
            ax.fill_between(xs, g[:, 1], g[:, 2], color=c, alpha=0.12, lw=0)
        ax.axhline(0, color="k", lw=0.6, ls=":")
        rv = (V.get(sw) or {}).get("real_mean") if V else None
        if rv is not None:
            # the realised value of a reading under this panel's costs
            ax.axvline(rv, color="#2e9b5e", lw=0.8, ls="--",
                       label="realised value of a reading" if ax is axes[0] else None)
        ax.set_xscale("log")
        ax.set_xlabel("price of one expert call (reward units)")
        ax.set_title(title, fontsize=8.5)
    axes[0].set_ylabel("gain over greedy (7 days)")
    axes[0].legend(fontsize=6.5, frameon=False)
    save(fig, "cp_fig10_iter2_price")


# ------------------------------------------------------------------ development
DEV = CP_RESULTS / "v2_dev"
# every variant run on the 16 development plants while the second iteration
# was developed, in order: (label, description, {scenario: (file, method)})
DEV_VARIANTS = [
    ("first iteration", r"\ours-1 as frozen",
     {"nominal": ("diag_v1_nominal", "cauprobe"), "lpwan": ("diag_v2a_lpwan", "cauprobe")}),
    ("expert + greedy", r"\ours-1 without probing",
     {"nominal": ("diag_v1_nominal", "cauprobe-noprobe"),
      "lpwan": ("diag_v2a_lpwan", "cauprobe-noprobe")}),
    ("a", r"free-learning horizon (2-day rollouts, regret held after them)",
     {"nominal": ("diag_v2a_nominal", "cauprobe-v2"), "lpwan": ("diag_v2a_lpwan", "cauprobe-v2")}),
    ("b", r"a + per-structure gains clipped at 0, stratified reading value, "
          r"8 truths, measured durability",
     {"nominal": ("diag_v2c_nominal", "cauprobe-v2"), "lpwan": ("diag_v2c_lpwan", "cauprobe-v2")}),
    ("c", r"b with 3-day rollouts and a geometric tail",
     {"nominal": ("diag_v2c_nominal", "cauprobe-v2@fl_tail=geometric+fl_days=3"),
      "lpwan": ("diag_v2c_lpwan", "cauprobe-v2@fl_tail=geometric+fl_days=3")}),
    ("d", r"c + return after a probe to the configuration it left",
     {"nominal": ("diag_v2d_nominal", "cauprobe-v2")}),
    ("e", r"c + return after a probe to the shadow greedy configuration "
          r"(frozen as \ours-2)",
     {"nominal": ("diag_v2e_nominal", "cauprobe-v2"), "lpwan": ("diag_v2e_lpwan", "cauprobe-v2")}),
]


def dev_rows():
    """Per dev variant and scenario: mean gain over greedy (per-plant, paired),
    probe steps, runs on a wrong mechanism."""
    from cp_common import DEV_EVAL_SEEDS
    cache = {}
    out = []
    for lab, desc, spec in DEV_VARIANTS:
        row = {"label": lab, "desc": desc}
        for sc, (f, m) in spec.items():
            if f not in cache:
                cache[f] = _load(DEV / f"{f}.jsonl", DEV_EVAL_SEEDS)
            rr = cache[f]
            g = {r["seed"]: r for r in rr if r["method"] == "greedy"}
            if not g:    # greedy rows of this scenario live in the v1 files
                gf = "diag_v1_nominal" if sc == "nominal" else "diag_v2a_lpwan"
                if gf not in cache:
                    cache[gf] = _load(DEV / f"{gf}.jsonl", DEV_EVAL_SEEDS)
                g = {r["seed"]: r for r in cache[gf] if r["method"] == "greedy"}
            mm = {r["seed"]: r for r in rr if r["method"] == m}
            acc = defaultdict(list)
            for s, r in mm.items():
                if s in g:
                    acc[s // 10].append(r["return"] - g[s]["return"])
            d = np.array([np.mean(v) for v in acc.values()])
            if len(d) > 1:
                row[sc] = {"ci": boot_ci(d), "probes": float(np.mean(
                    [r["probe_steps"] for r in mm.values()])), "n": len(mm),
                    "wrong": float(np.mean([bool(wrong(r)) for r in mm.values()]))}
        out.append(row)
    return out


def table_dev(D):
    L = [r"\begin{table}[h]", r"\centering",
         r"\caption{Development of the second iteration, on the 16 development "
         r"plants only (two runs each, 14 days): return gained over greedy "
         r"control on the same plant and noise stream, mean [95\% bootstrap CI "
         r"over plants], and probe steps per run. Every variant that was run is "
         r"listed, in order. Probe counts of the first iteration are lower "
         r"bounds (Section~\ref{sec:v2}).}",
         r"\label{tab:iter2dev}", r"\footnotesize", r"\setlength{\tabcolsep}{3pt}",
         r"\begin{tabular}{p{5.4cm}rrrr}", r"\toprule",
         r"Variant & nominal & probes & LPWAN & probes \\", r"\midrule"]
    for row in D:
        cells = []
        for sc in ("nominal", "lpwan"):
            x = row.get(sc)
            if x is None:
                cells += ["--", "--"]
            else:
                m_, lo, hi = x["ci"]
                cells += [f"{m_:+.1f} [{lo:+.0f}, {hi:+.0f}]", f"{x['probes']:.0f}"]
        name = row["label"] if row["label"] in ("first iteration", "expert + greedy") \
            else f"({row['label']}) {row['desc']}"
        L.append(name + " & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(L)


def table_val(S):
    """Validation plants: the decision to freeze was taken on these."""
    cols = [sc for sc in SCEN if sc in S]
    if not cols:
        return ""
    L = [r"\begin{table}[h]", r"\centering",
         r"\caption{Validation of the second iteration on the 48 validation "
         r"plants (run once, after development and before freezing; two runs "
         r"in the nominal scenario, one in the others): return gained over "
         r"greedy control, mean [95\% CI], and probe steps per run.}",
         r"\label{tab:iter2val}", r"\footnotesize", r"\setlength{\tabcolsep}{3pt}",
         r"\begin{tabular}{l" + "r" * len(cols) + "}", r"\toprule",
         " & " + " & ".join(SCEN_LABEL[c] for c in cols) + r" \\", r"\midrule"]
    for m in ("cauprobe-v2", "cauprobe", "cauprobe-noprobe"):
        cells = []
        for c in cols:
            x = S[c]["methods"].get(m, {}).get("vs_greedy")
            cells.append("--" if x is None else
                         f"{x['ci'][0]:+.1f} [{x['ci'][1]:+.0f}, {x['ci'][2]:+.0f}]")
        L.append(LABEL[m] + " & " + " & ".join(cells) + r" \\")
    for m in ("cauprobe-v2", "cauprobe"):
        cells = [f"{S[c]['methods'][m]['probes']:.0f}" if m in S[c]["methods"]
                 else "--" for c in cols]
        L.append(LABEL[m] + r", probe steps & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(L)


# ------------------------------------------------------------------ numbers
def numbers(val=False):
    S = scen_summary(val)
    N = {"scen": {}}
    for sc, d in S.items():
        e = {}
        for k, x in d["cmp"].items():
            e[k] = {"mean": x["ci"][0], "lo": x["ci"][1], "hi": x["ci"][2],
                    "p": x["p"], "p_holm": x.get("p_holm")}
        for m, x in d["methods"].items():
            e[f"probes:{m}"] = x["probes"]
            e[f"asks:{m}"] = x["asks"]
            e[f"switch:{m}"] = x["switch_cost"]
            if "wrong" in x:
                e[f"wrong:{m}"] = x["wrong"]
                e[f"n_wrong:{m}"] = x["n_wrong"]
            if "gap" in x:
                e[f"gap:{m}"] = x["gap"][0]
            if "vs_greedy" in x:
                e[f"vs_greedy:{m}"] = x["vs_greedy"]["ci"][0]
                e[f"vs_greedy_p:{m}"] = x["vs_greedy"]["p"]
            e[f"n_runs:{m}"] = x["n_runs"]
        N["scen"][sc] = e
    N["lockin"] = lockin(val)
    N["ool"] = ool_summary(val)
    # compute: mean wall-clock seconds per 14-day run (nominal scenario; 56
    # runs in parallel on one host, so comparable between methods only)
    b = by_method(rows("nominal", val))
    N["wall_s"] = {m: float(np.mean([r["wall_s"] for r in d.values()]))
                   for m, d in b.items()}
    P = price_rows(val)
    N["price"] = {"|".join(map(str, k)): v for k, v in P.items()}
    N["voi_ask"] = voi_ask(val)
    # how many scenarios v2 beats greedy / expert+greedy in (point estimate,
    # and significantly after Holm)
    for a, c in KEY:
        k = f"{a}|{c}"
        xs = [S[sc]["cmp"][k] for sc in S if k in S[sc]["cmp"]]
        N[f"count:{k}"] = {"n": len(xs),
                           "better": sum(x["ci"][0] > 0 for x in xs),
                           "sig_better": sum(x["ci"][0] > 0 and x.get("p_holm", 1) < 0.05
                                             for x in xs),
                           "sig_worse": sum(x["ci"][0] < 0 and x.get("p_holm", 1) < 0.05
                                            for x in xs)}
    return N


if __name__ == "__main__":
    val = "--val" in sys.argv
    N = numbers(val)
    print(json.dumps({k: v for k, v in N.items() if k != "price"}, indent=1,
                     default=float)[:20000])

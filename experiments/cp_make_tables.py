"""Every results table of the paper, generated from the raw JSON.

Writes paper/tables_cp.tex (one \\newcommand-free file of tables, each wrapped
in its own table environment) and results/cauprobe/summary.json with the
numbers the text quotes. Missing inputs are skipped with a note, so the script
can be run while experiments are still going.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from cp_common import CP_RESULTS, ROOT, heldout, testrows
from cp_summarize import boot_ci, holm, paired_p
import cp_tables_extra as X

PAPER = ROOT / "paper"
OUT_TEX = PAPER / "tables_cp.tex"
SUMMARY = CP_RESULTS / "summary.json"

LABEL = {
    "oracle": "known-plant reference",
    "rule": "rule-based (incumbent)",
    "fallback": "fallback configuration",
    "nominal": "nominal-model MPC",
    "linucb": "LinUCB",
    "advice-plan": "advice-as-plan (prior design)",
    "greedy": "greedy (same belief)",
    "log-naive": "greedy + confounded logs",
    "thompson": "Thompson sampling",
    "thompson-unsafe": "Thompson sampling, no filter",
    "ids": "IDS",
    "identify": "identify-then-act",
    "cauprobe": r"\ours",
    "cauprobe-noask": r"\ours\ without expert",
    "cauprobe-noprobe": r"\ours\ without probing",
    "cauprobe-nosafety": r"\ours\ without safety filter",
    "cauprobe+logs-naive": r"\ours\ + confounded logs",
    "cauprobe-particle": r"VOI over particles",
    "cauprobe-nodur": r"no durability correction",
    "cauprobe-nokappa": r"no self-calibration",
    "cauprobe-h1": r"horizon $H = 1$ day",
    "cauprobe-h7": r"horizon $H = 7$ days",
    "cauprobe-expert:tfidf-lr": r"expert: TF-IDF + LR",
    "cauprobe-expert:slm-0.5b-lora-uncal": r"expert: SLM, uncalibrated",
    "cauprobe-expert:zeroshot-1.5b": r"expert: zero-shot 1.5B",
}


def load_jsonl(p):
    p = Path(p)
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def per_plant(rows, ref_rows, key):
    """Per-plant mean of key(row, ref_row) over runs, paired on (plant, seed)."""
    ref = {(r["plant"], r["seed"]): r for r in ref_rows}
    acc = defaultdict(list)
    for r in rows:
        k = (r["plant"], r["seed"])
        if k in ref:
            v = key(r, ref[k])
            if v is not None:
                acc[r["seed"] // 10].append(v)
    return {p: float(np.mean(v)) for p, v in acc.items()}


def wrong_mech(r, o=None):
    """1 if the run ends with the posterior on the wrong side of 1/2 for any
    network factor (the thermal factor is never misidentified in our runs)."""
    fp = r.get("final_factor_post")
    if not fp:
        return None
    c = list(r["combo"])
    if r.get("scenario") == "shift":
        c[0] = 1 - c[0]          # the radio was swapped on day 3: A flipped
    return float(any((fp[F] > 0.5) != bool(c[i]) for i, F in enumerate("ABC")))


def cvar(vals, q=0.1):
    """Mean of the worst q-fraction of per-plant gaps."""
    a = np.sort(np.asarray(vals))[::-1]
    return float(a[:max(1, int(round(len(a) * q)))].mean())


def method_stats(rows, ref="cauprobe"):
    by = defaultdict(list)
    for r in rows:
        by[r["method"]].append(r)
    if "oracle" not in by:
        return {}
    orc = by["oracle"]
    out = {}
    metrics = {
        "gap": lambda r, o: o["return"] - r["return"],
        "gap_net": lambda r, o: o["r_net"] - r["r_net"],
        "gap_th": lambda r, o: o["r_th"] - r["r_th"],
        "day1": lambda r, o: o["return_per_day"][0] - r["return_per_day"][0],
        "misses": lambda r, o: r["viol_net"] + r["viol_th"],
        "excess_misses": lambda r, o: (r["viol_net"] + r["viol_th"])
        - (o["viol_net"] + o["viol_th"]),
        "probes": lambda r, o: r["probe_steps"],
        "switches": lambda r, o: r.get("n_switches", 0),
        "asks": lambda r, o: len(r["asks"]),
        "th_probes": lambda r, o: r["th_probes"],
        "fallback": lambda r, o: r["fallback_steps"],
        "alarms": lambda r, o: r.get("misspec_alarms", 0),
        "wrong": wrong_mech,
    }
    for m, rr in by.items():
        st = {}
        for name, f in metrics.items():
            pp = per_plant(rr, orc, f)
            if pp:
                st[name] = boot_ci(list(pp.values()))
                if name == "gap":
                    st["_pp"] = pp
        st["n_plants"] = len(st.get("_pp", {}))
        if st.get("_pp"):
            st["cvar10"] = cvar(list(st["_pp"].values()))
        out[m] = st
    if ref in out:
        refpp = out[ref]["_pp"]
        others = [m for m in out if m != ref]
        ds, ps = [], []
        for m in others:
            pp = out[m]["_pp"]
            common = sorted(set(pp) & set(refpp))
            d = np.array([pp[k] - refpp[k] for k in common])
            ds.append(boot_ci(d) if len(d) > 1 else (float("nan"),) * 3)
            ps.append(paired_p(d) if len(d) > 1 else 1.0)
        for m, d, p, pa in zip(others, ds, ps, holm(ps)):
            out[m]["vs_ref"] = {"diff": d, "p": p, "p_holm": float(pa)}
    return out


def fmt_ci(t, nd=1):
    m, lo, hi = t
    return f"{m:.{nd}f} [{lo:.{nd}f}, {hi:.{nd}f}]"


def fmt_p(p):
    if p < 1e-3:
        return r"$<$0.001"
    return f"{p:.3f}"


def main_table(st, order, caption, label, ref="cauprobe"):
    lines = [r"\begin{table*}[t]", r"\centering", rf"\caption{{{caption}}}",
             rf"\label{{{label}}}", r"\footnotesize", r"\setlength{\tabcolsep}{2.4pt}",
             r"\begin{tabular}{lrrrrrrrrrr}", r"\toprule",
             r"Method & gap [95\% CI] & network & thermal & day 1 & "
             r"CVaR$_{10}$ & misses & probes & reconf. & wrong & "
             r"$\Delta$ vs \ours\ ($p_{\mathrm{Holm}}$) \\",
             r"\midrule"]
    for m in order:
        if m not in st:
            continue
        s = st[m]
        vs = s.get("vs_ref")
        vs_s = "--" if m == ref or vs is None else \
            f"{vs['diff'][0]:+.1f} ({fmt_p(vs['p_holm'])})"
        name = LABEL.get(m, m)
        if m == ref:
            name = r"\textbf{" + name + "}"
        wr = f"{100 * s['wrong'][0]:.0f}\\%" if "wrong" in s else "--"
        gm, glo, ghi = s["gap"]
        lines.append(
            f"{name} & {gm:.1f} [{glo:.0f}, {ghi:.0f}] & {s['gap_net'][0]:.1f} & "
            f"{s['gap_th'][0]:.2f} & {s['day1'][0]:.1f} & {s['cvar10']:.0f} & "
            f"{s['misses'][0]:.1f} & {s['probes'][0]:.0f} & "
            f"{s['switches'][0]:.0f} & {wr} & {vs_s} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""]
    return "\n".join(lines)


MAIN_ORDER = ["cauprobe", "cauprobe-noask", "cauprobe-noprobe", "greedy",
              "thompson", "ids", "identify", "log-naive", "nominal",
              "advice-plan", "linucb", "rule", "fallback", "oracle"]
ABL_ORDER = ["cauprobe", "cauprobe-particle", "cauprobe-nodur",
             "cauprobe-nokappa", "cauprobe-h1", "cauprobe-h7",
             "cauprobe-nosafety", "thompson-unsafe", "cauprobe+logs-naive",
             "cauprobe-expert:slm-0.5b-lora-uncal", "cauprobe-expert:tfidf-lr",
             "cauprobe-expert:zeroshot-1.5b", "cauprobe-noask",
             "cauprobe-noprobe"]
CORE = ["cauprobe", "cauprobe-noask", "cauprobe-noprobe", "greedy", "thompson",
        "ids", "identify", "advice-plan", "linucb", "rule"]
SCEN = ["nominal", "noisy", "lpwan", "noisy_lpwan", "poor_docs", "shift"]
SCEN_LABEL = {"nominal": "nominal", "noisy": "noisy", "lpwan": "LPWAN",
              "noisy_lpwan": r"noisy+LPWAN", "poor_docs": "poor docs",
              "shift": "struct.\\ change"}


def scenario_table(stats_by_scen):
    cols = [s for s in SCEN if s in stats_by_scen]
    lines = [r"\begin{table*}[t]", r"\centering",
             r"\caption{Gap to the known-plant reference over 14 days, per "
             r"deployment scenario (mean over the 48 test plants; lower is better). "
             r"Bold: best learning method in the column. $^{*}$: worse than "
             r"\ours\ at $p_{\mathrm{Holm}}<0.05$; $^{\dagger}$: better than "
             r"\ours\ at $p_{\mathrm{Holm}}<0.05$. The last column is each "
             r"method's worst gap across scenarios minus the best method's gap "
             r"in that scenario (its worst-case shortfall).}",
             r"\label{tab:scenarios}", r"\footnotesize",
             r"\begin{tabular}{l" + "r" * len(cols) + "r}", r"\toprule",
             "Method & " + " & ".join(SCEN_LABEL[c] for c in cols)
             + r" & worst shortfall \\", r"\midrule"]
    learners = [m for m in CORE if m not in ("rule",)]
    best = {c: min((stats_by_scen[c][m]["gap"][0] for m in learners
                    if m in stats_by_scen[c]), default=np.nan) for c in cols}
    for m in CORE:
        cells = []
        short = []
        for c in cols:
            st = stats_by_scen[c].get(m)
            if st is None:
                cells.append("--")
                continue
            g = st["gap"][0]
            short.append(g - best[c])
            s = f"{g:.1f}"
            vs = st.get("vs_ref")
            if vs is not None and vs["p_holm"] < 0.05:
                s += r"$^{*}$" if vs["diff"][0] > 0 else r"$^{\dagger}$"
            if abs(g - best[c]) < 1e-9 and m in learners:
                s = r"\textbf{" + s + "}"
            cells.append(s)
        lines.append(LABEL.get(m, m) + " & " + " & ".join(cells)
                     + f" & {max(short) if short else float('nan'):.1f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""]
    return "\n".join(lines)


def main():
    tex = ["% generated by experiments/cp_make_tables.py -- do not edit", ""]
    summary = {}
    evp = CP_RESULTS / "expert_eval.json"
    if evp.exists():
        ev = json.loads(evp.read_text())
        tex.append(X.expert_table(ev))
        summary["expert"] = {k: {"tau": ev[k]["tau"],
                                 "val_acc": ev[k]["val"]["mean_acc_all"],
                                 "test_acc": ev[k]["test_cal"]["mean_acc_all"],
                                 "test_auroc": ev[k]["test_cal"]["mean_auroc"],
                                 "test_ece": ev[k]["test_cal"]["mean_ece"],
                                 "test_ece_raw": ev[k]["test_raw"]["mean_ece"],
                                 "val_ece": ev[k]["val"]["mean_ece"]}
                             for k in X.EXPERT_LABEL if k in ev}
    bfiles = [CP_RESULTS / f"e6_boptest_{t}.jsonl" for t in
              ("v1prior", "v2prior", "v3ctrl")]
    bfiles = [f for f in bfiles if f.exists()]
    v4 = CP_RESULTS / "e6_boptest_v4thdet.jsonl"
    if bfiles:
        tex.append(X.boptest_table(bfiles[-1]))
        summary["boptest"] = X.boptest_summary(bfiles + ([v4] if v4.exists() else []))
    # primary results: the 48 fresh test plants; appendix: validation plants
    main_rows = testrows(load_jsonl(CP_RESULTS / "e1_test_main.jsonl"))
    val_rows = heldout(load_jsonl(CP_RESULTS / "validation_v1" / "e1_main.jsonl"))
    appendix_tex = []
    if val_rows:
        dst = method_stats(val_rows)
        summary["main_devplants"] = {m: {k: v for k, v in s.items() if k != "_pp"}
                                     for m, s in dst.items()}
        appendix_tex.append(X.devplant_table(dst, MAIN_ORDER, LABEL))
    if main_rows:
        st = method_stats(main_rows)
        summary["main"] = {m: {k: v for k, v in s.items() if k != "_pp"}
                           for m, s in st.items()}
        tex.append(main_table(
            st, MAIN_ORDER,
            r"Commissioning the 48 test plants for 14 days (nominal scenario, "
            r"two runs per plant). \emph{Gap}: return of the known-plant "
            r"reference minus the method's, summed over 14 days (lower is "
            r"better; negative means better than the reference), with its "
            r"network and thermal parts and its value on day 1. "
            r"\emph{CVaR$_{10}$}: mean gap of the worst 10\% of plants. "
            r"\emph{Misses}: deadline violations. \emph{Probes}: steps "
            r"deviating from the greedy action for information. \emph{Reconf.}: "
            r"fleet reconfigurations. \emph{Wrong}: runs ending with the "
            r"posterior on the wrong side of $1/2$ for some network mechanism "
            r"(belief-based methods only). Last column: paired difference in "
            r"gap against \ours, bootstrap over plants, Holm-corrected.",
            "tab:main"))
        tex.append(main_table(
            st, ABL_ORDER,
            r"Ablations of \ours\ (nominal scenario, 48 test plants, two runs). "
            r"Each row removes or replaces one component; columns as in "
            r"Table~\ref{tab:main}.", "tab:ablation"))
    stats_by_scen = {}
    for sc in SCEN:
        rows = testrows(load_jsonl(CP_RESULTS / f"e1_test_{'main' if sc == 'nominal' else sc}.jsonl"))
        if rows:
            stats_by_scen[sc] = method_stats(rows)
            summary.setdefault("scenarios", {})[sc] = {
                m: {k: v for k, v in s.items() if k != "_pp"}
                for m, s in stats_by_scen[sc].items()}
    if len(stats_by_scen) > 1:
        tex.append(scenario_table(stats_by_scen))
    op = CP_RESULTS / "e1_test_ool.jsonl"
    if op.exists():
        tex.append(X.ool_table(op, LABEL))
        summary["ool"] = {"|".join(k): v for k, v in X.ool_rows(op).items()}
    pp = CP_RESULTS / "e7_price.jsonl"
    if pp.exists():
        tex.append(X.price_table(pp, [0.01, 0.1, 1.0, 3.0, 10.0, 30.0]))
        summary["price"] = {"|".join(map(str, k)): v
                            for k, v in X.price_rows(pp).items()}
    ep = CP_RESULTS / "edge.json"
    if ep.exists():
        tex.append(X.edge_table(ep))
    if bfiles:
        appendix_tex.append(X.boptest_iterations_table(
            bfiles + ([v4] if v4.exists() else [])))
    (PAPER / "tables_cp_appendix.tex").write_text("\n".join(appendix_tex))
    OUT_TEX.write_text("\n".join(tex))
    # second iteration: second fresh test set (main text) and its development
    # and validation (appendix); see cp_iter2.py
    import cp_iter2 as I2
    S2 = I2.scen_summary()
    S2v = I2.scen_summary(val=True)
    P2 = I2.price_rows()
    (PAPER / "tables_cp_iter2.tex").write_text(
        "% generated by experiments/cp_make_tables.py -- do not edit\n"
        + I2.table_scenarios(S2) + "\n" + I2.table_price(P2, I2.voi_ask()))
    (PAPER / "tables_cp_iter2_appendix.tex").write_text(
        "% generated by experiments/cp_make_tables.py -- do not edit\n"
        + I2.table_dev(I2.dev_rows()) + "\n" + I2.table_val(S2v))
    summary["iter2"] = S2
    summary["iter2_val"] = S2v
    SUMMARY.write_text(json.dumps(summary, indent=1, default=float))
    print("wrote", OUT_TEX, "scenarios:", list(stats_by_scen))


if __name__ == "__main__":
    main()

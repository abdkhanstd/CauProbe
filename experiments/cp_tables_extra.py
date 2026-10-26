"""Expert, BOPTEST, price-sweep and edge tables (used by cp_make_tables.py)."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from cp_summarize import boot_ci

EXPERT_LABEL = {"slm-0.5b-lora": r"Qwen2.5-0.5B + LoRA (ours)",
                "tfidf-lr": "TF-IDF + logistic regression",
                "zeroshot-1.5b": "Qwen2.5-1.5B, zero-shot",
                "zeroshot-0.5b": "Qwen2.5-0.5B, zero-shot"}

BOP_LABEL = {"boptest-baseline": "BOPTEST controller",
             "incumbent": r"fixed incumbent (24\,$^\circ$C)",
             "nominal": "nominal grey box", "greedy": "greedy",
             "thompson": "Thompson sampling", "cauprobe": r"\ours",
             "cauprobe+expert": r"\ours\ + expert",
             "cauprobe-v2": r"\ours-2", "cauprobe-v2+expert": r"\ours-2 + expert"}


def load_jsonl(p):
    p = Path(p)
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def fmt_ci(t, nd=1):
    m, lo, hi = t
    return f"{m:.{nd}f} [{lo:.{nd}f}, {hi:.{nd}f}]"


def _mean(m, key):
    xs = [m[F][key] for F in "ABCD" if m[F][key] is not None]
    return float(np.mean(xs))


def expert_table(ev):
    lines = [r"\begin{table*}[t]", r"\centering",
             r"\caption{The text expert. Mean over the four mechanism questions "
             r"of accuracy on validation notes written in the \emph{training} "
             r"vocabulary, and, on 2\,000 notes written in the held-out "
             r"vocabulary: accuracy overall, on factors the note mentions "
             r"correctly, and on factors about which it states stale "
             r"(opposite) information; AUROC of the logit; and expected "
             r"calibration error after the temperature fitted on "
             r"training-vocabulary validation notes. Chance accuracy is 0.5.}",
             r"\label{tab:expert}", r"\small",
             r"\begin{tabular}{lrrrrrr}", r"\toprule",
             r"& train vocab. & \multicolumn{5}{c}{held-out vocabulary} \\",
             r"\cmidrule(lr){2-2}\cmidrule(lr){3-7}",
             r"Expert & accuracy & accuracy & mentioned & stale & AUROC & ECE \\",
             r"\midrule"]
    for k in ("slm-0.5b-lora", "tfidf-lr", "zeroshot-1.5b", "zeroshot-0.5b"):
        if k not in ev:
            continue
        v, t = ev[k]["val"], ev[k]["test_cal"]
        lines.append(f"{EXPERT_LABEL[k]} & {v['mean_acc_all']:.3f} & "
                     f"{t['mean_acc_all']:.3f} & {_mean(t, 'acc_mentioned'):.3f} & "
                     f"{_mean(t, 'acc_stale'):.3f} & {t['mean_auroc']:.3f} & "
                     f"{t['mean_ece']:.3f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""]
    return "\n".join(lines)


def boptest_table(path):
    rows = load_jsonl(path)
    # the second iteration, run afterwards on the same weeks and seeds
    rows += load_jsonl(Path(path).with_name(Path(path).stem + "_v2.jsonl"))
    by = defaultdict(dict)
    for r in rows:
        by[r["method"]][r["start_day"]] = r
    inc = by.get("incumbent", {})
    lines = [r"\begin{table}[t]", r"\centering",
             r"\caption{BOPTEST \texttt{bestest\_air}, ten summer weeks, third library "
             r"specification (the controller's logged response in the likelihood), "
             r"before addition of the thermal misfit detector of the fourth "
             r"iteration (Supplementary Section~\ref{S-sec:supp-boptest}). "
             r"Reward per week is $-(0.25\times$tariff-weighted "
             r"kWh $+$ K\,h of occupied discomfort), mean [95\% CI]; "
             r"discomfort and energy as measured; $\Delta$: paired difference "
             r"in reward from the fixed incumbent.}",
             r"\label{tab:boptest}", r"\small", r"\setlength{\tabcolsep}{3pt}",
             r"\begin{tabular}{lrrrr}", r"\toprule",
             r"Controller & reward & K\,h & kWh & $\Delta$ \\", r"\midrule"]
    for m in ("boptest-baseline", "incumbent", "nominal", "greedy", "thompson",
              "cauprobe", "cauprobe+expert", "cauprobe-v2", "cauprobe-v2+expert"):
        if m not in by:
            continue
        days = sorted(set(by[m]) & set(inc))
        rew = [by[m][d]["reward"] for d in days]
        dis = float(np.mean([by[m][d]["tdis_kh"] for d in days]))
        en = float(np.mean([by[m][d]["energy_kwh"] for d in days]))
        dd = np.array([by[m][d]["reward"] - inc[d]["reward"] for d in days])
        dstr = "--" if m == "incumbent" else f"{dd.mean():+.2f}"
        lines.append(f"{BOP_LABEL[m]} & {fmt_ci(boot_ci(rew), 2)} & {dis:.2f} & "
                     f"{en:.1f} & {dstr} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def boptest_summary(paths):
    out = {}
    for p in paths:
        rows = load_jsonl(p)
        by = defaultdict(list)
        for r in rows:
            by[r["method"]].append(r)
        out[Path(p).stem] = {
            m: {"reward": float(np.mean([x["reward"] for x in v])),
                "tdis": float(np.mean([x["tdis_kh"] for x in v])),
                "energy": float(np.mean([x["energy_kwh"] for x in v])),
                "probes": float(np.mean([x["th_probes"] for x in v]))}
            for m, v in by.items()}
    return out


def price_rows(path):
    """Paired gain over greedy (same plant, same noise) per method/price/switch."""
    rows = load_jsonl(path)
    base = {(r["plant"], r["seed"], r["switch"]): r for r in rows
            if r["method"] == "greedy"}
    acc = defaultdict(list)
    for r in rows:
        b = base.get((r["plant"], r["seed"], r["switch"]))
        if b is None or r["method"] == "greedy":
            continue
        acc[(r["switch"], r["method"], r["price"])].append(
            (r["return"] - b["return"], len(r["asks"]), r["probe_steps"]))
    out = {}
    for k, v in acc.items():
        g = np.array([x[0] for x in v])
        out[k] = {"gain": boot_ci(g), "asks": float(np.mean([x[1] for x in v])),
                  "probes": float(np.mean([x[2] for x in v])), "n": len(v)}
    return out


def price_table(path, prices):
    st = price_rows(path)
    if not st:
        return ""
    lines = [r"\begin{table*}[t]", r"\centering",
             r"\caption{Who should pay for which knowledge. Gain in 7-day "
             r"return over greedy control on the same plant and noise stream "
             r"(32 plants, mean [95\% CI]; higher is better) as the price of "
             r"one expert invocation rises, under Wi-Fi-like and LPWAN-like "
             r"reconfiguration costs. \emph{Ask-always} reads the text whatever "
             r"it costs and never probes; \emph{probe-only} never reads it. "
             r"\ours's invocations and probe steps show which source it "
             r"buys.}", r"\label{tab:price}", r"\small",
             r"\setlength{\tabcolsep}{3pt}",
             r"\begin{tabular}{lrrrrrrrr}", r"\toprule",
             r"& \multicolumn{4}{c}{Wi-Fi reconfiguration} & "
             r"\multicolumn{4}{c}{LPWAN reconfiguration} \\",
             r"\cmidrule(lr){2-5}\cmidrule(lr){6-9}",
             r"price/call & \ours & ask-always & asks & probes & \ours & "
             r"ask-always & asks & probes \\", r"\midrule"]
    for p in prices:
        cells = []
        for sw in ("wifi", "lpwan"):
            c = st.get((sw, "cauprobe", p))
            a = st.get((sw, "ask-always", p))
            if c is None or a is None:
                cells += ["--"] * 4
                continue
            cells += [f"{c['gain'][0]:+.1f} [{c['gain'][1]:+.1f}, {c['gain'][2]:+.1f}]",
                      f"{a['gain'][0]:+.1f}", f"{c['asks']:.2f}", f"{c['probes']:.0f}"]
        lines.append(f"{p:g} & " + " & ".join(cells) + r" \\")
    po = [st.get((sw, "probe-only", 0.0)) for sw in ("wifi", "lpwan")]
    if all(po):
        lines.append(r"\midrule")
        lines.append(r"probe-only & \multicolumn{4}{l}{"
                     f"{po[0]['gain'][0]:+.1f} ({po[0]['probes']:.0f} probe steps)"
                     r"} & \multicolumn{4}{l}{"
                     f"{po[1]['gain'][0]:+.1f} ({po[1]['probes']:.0f} probe steps)"
                     r"} \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""]
    return "\n".join(lines)


# Published measurements of the same base model (Qwen2.5-0.5B, 4-bit, Ollama):
# (label, cite key, generation tok/s, J per generated token or None).
PUBLISHED = [
    ("Raspberry Pi 5 (CPU)", "renney2026cloudedge", 0.86, 10.83),
    ("Raspberry Pi 5 (CPU)", "dfrobot2024slm", 19.41, None),
    ("Jetson Orin Nano Super (CPU)", "renney2026cloudedge", 13.04, 0.92),
]
DEV_LABEL = {"raspberry-pi-4": r"Raspberry Pi 4 (4$\times$A72)",
             "raspberry-pi-5": r"Raspberry Pi 5 (4$\times$A76)",
             "intel-n100": "Intel N100 gateway",
             "jetson-orin-nano-cpu": "Jetson Orin Nano Super (CPU)"}


def edge_table(path):
    p = Path(path)
    if not p.exists():
        return ""
    e = json.loads(p.read_text())
    P, G = e["prompt_tokens"], e["answer_tokens"]
    lines = [r"\begin{table*}[t]", r"\centering",
             r"\caption{What one expert invocation costs (" + f"{P}" +
             r"-token prompt, " + f"{G}" + r" answer tokens). \emph{Measured}: "
             r"llama.cpp on four pinned cores of the host (Xeon E5-2698 v3) and "
             r"the fp16 model on an RTX 2080 Ti with NVML energy and the settled "
             r"idle draw subtracted. \emph{Emulated}: the AArch64 build under "
             r"QEMU (Cortex-A76): correctness and memory only. "
             r"\emph{Modelled}: roofline scaling of the measured host rates "
             r"(prefill by peak compute; decode by attainable memory bandwidth, "
             r"capped at the prefill rate, where $^{c}$ marks the cap binding); "
             r"energy is active power above idle times latency. "
             r"\emph{Published}: independent measurements of the same base "
             r"model with a 4-bit Ollama build; $^{*}$ decode of the "
             + f"{G}" + r" answer tokens only. The spread between the two "
             r"published Raspberry Pi~5 figures is why the agent takes the "
             r"price of a call as an input (Section~\ref{sec:res-price}).}",
             r"\label{tab:edge}", r"\small", r"\setlength{\tabcolsep}{3.5pt}",
             r"\begin{tabular}{llrrrr}", r"\toprule",
             r"Platform & build & prefill tok/s & decode tok/s & s/call & J/call \\",
             r"\midrule", r"\multicolumn{6}{l}{\emph{Measured}} \\"]
    for q, c in e.get("host_cpu", {}).items():
        if "pp_tps" in c:
            lines.append(f"host CPU, 4 threads & {q.upper().replace('_', '-')} & "
                         f"{c['pp_tps']:.0f} & {c['tg_tps']:.1f} & "
                         f"{c['call_ms']/1000:.2f} & -- \\\\")
    g = e.get("host_gpu", {})
    if "call_ms" in g:
        lines.append(f"RTX 2080 Ti (HF, fp16) & F16 & -- & -- & "
                     f"{g['call_ms']/1000:.2f} & {g['j_per_call']:.1f} \\\\")
    q = e.get("qemu", {})
    if q.get("ran"):
        lines += [r"\midrule", r"\multicolumn{6}{l}{\emph{Emulated (QEMU, "
                  r"AArch64): answers identical to x86 on "
                  + f"{q['answers_identical']}/{q['n']}"
                  + r" prompts; peak resident memory "
                  + f"{q['peak_rss_mib']:.0f}" + r"\,MiB incl.\ emulator}} \\"]
    lines += [r"\midrule", r"\multicolumn{6}{l}{\emph{Modelled (roofline)}} \\"]
    for dev, row in e.get("roofline", {}).items():
        r = row.get("q4_k_m")
        if r is None:
            continue
        cap = "$^{c}$" if r.get("decode_bound") == "compute" else ""
        lines.append(f"{DEV_LABEL.get(dev, dev)} & Q4-K-M & {r['pp_tps']:.0f} & "
                     f"{r['tg_tps']:.1f}{cap} & {r['call_s']:.2f} & "
                     f"{r['j_per_call']:.1f} \\\\")
    lines += [r"\midrule", r"\multicolumn{6}{l}{\emph{Published (Ollama, 4-bit)}} \\"]
    for lab, key, tps, jtok in PUBLISHED:
        j = f"{jtok * G:.0f}$^{{*}}$" if jtok is not None else "--"
        lines.append(f"{lab} \\cite{{{key}}} & Q4 & -- & {tps:.2f} & "
                     f"{G / tps:.1f}$^{{*}}$ & {j} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""]
    return "\n".join(lines)


def boptest_iterations_table(paths):
    """Mean weekly reward of each learner across the four library iterations."""
    names = {"e6_boptest_v1prior": "(1) fixed dead band",
             "e6_boptest_v2prior": "(2) dead band, gain in prior",
             "e6_boptest_v3ctrl": "(3) controller response in likelihood",
             "e6_boptest_v4thdet": "(4) + thermal out-of-library detector"}
    meth = ["greedy", "thompson", "cauprobe", "cauprobe+expert"]
    inc_rows = load_jsonl(paths[0])
    inc = {r["start_day"]: r["reward"] for r in inc_rows if r["method"] == "incumbent"}
    lines = [r"\begin{table}[h]", r"\centering",
             r"\caption{BOPTEST: mean weekly reward of the learners across the "
             r"four library iterations, minus the fixed incumbent's on the same "
             r"week (positive is better than the incumbent). Alarms: weeks in "
             r"which the thermal detector fired (iteration 4 only).}",
             r"\label{tab:boptest-iter}", r"\small", r"\setlength{\tabcolsep}{3pt}",
             r"\begin{tabular}{lrrrrr}", r"\toprule",
             r"Iteration & greedy & Thompson & \ours & \ours+expert & alarms \\",
             r"\midrule"]
    for p in paths:
        rows = load_jsonl(p)
        by = defaultdict(dict)
        for r in rows:
            by[r["method"]][r["start_day"]] = r
        cells = []
        for m in meth:
            if m not in by:
                cells.append("--")
                continue
            d = [by[m][k]["reward"] - inc[k] for k in by[m] if k in inc]
            cells.append(f"{np.mean(d):+.2f}")
        al = sum(int(r.get("th_alarms", 0) > 0) for r in rows if r["method"] == "cauprobe")
        al_s = f"{al}/{sum(1 for r in rows if r['method'] == 'cauprobe')}" \
            if Path(p).stem.endswith("v4thdet") else "--"
        lines.append(names.get(Path(p).stem, Path(p).stem) + " & " + " & ".join(cells)
                     + f" & {al_s} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def devplant_table(st, order, label_map):
    lines = [r"\begin{table}[h]", r"\centering",
             r"\caption{Validation plants (48, nominal scenario, two runs), "
             r"with \ours\ as first frozen (self-calibration judged at the end "
             r"of each probe): 14-day gap to the known-plant reference, mean "
             r"[95\% CI]. These runs led to the revision of the self-calibration "
             r"check described in Section~\ref{sec:setup}.}",
             r"\label{tab:devplants}", r"\small", r"\begin{tabular}{lr}",
             r"\toprule", r"Method & gap \\", r"\midrule"]
    for m in order:
        if m in st:
            g = st[m]["gap"]
            lines.append(f"{label_map.get(m, m)} & {g[0]:.1f} [{g[1]:.1f}, {g[2]:.1f}] \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


OOL_LABEL = {"tsch": "scheduled MAC (TSCH)", "congested_wan": "self-congesting backhaul",
             "broker_fault": "intermittent broker fault"}
OOL_ORDER = ["greedy", "thompson", "ids", "identify", "cauprobe", "cauprobe-noask",
             "cauprobe-noprobe", "advice-plan", "linucb"]


def ool_rows(path):
    """Per method and out-of-library type: return minus the rule-based
    incumbent's on the same plant and noise (bootstrap over plants), deadline
    misses, share of runs in which the detector fired, fallback steps."""
    rows = load_jsonl(path)
    inc = {(r["plant"], r["seed"]): r for r in rows if r["method"] == "rule"}
    out = {}
    for m in OOL_ORDER + ["rule"]:
        for k in OOL_LABEL:
            rr = [r for r in rows if r["method"] == m and r["ool"] == k
                  and (r["plant"], r["seed"]) in inc]
            if not rr:
                continue
            pp = defaultdict(list)
            for r in rr:
                pp[r["seed"] // 10].append(r["return"] - inc[(r["plant"], r["seed"])]["return"])
            out[(m, k)] = {"d_rule": boot_ci([float(np.mean(v)) for v in pp.values()]),
                           "misses": float(np.mean([r["viol_net"] for r in rr])),
                           "alarm": float(np.mean([r.get("misspec_alarms", 0) > 0 for r in rr])),
                           "fallback": float(np.mean([r["fallback_steps"] for r in rr])),
                           "n_plants": len(pp)}
    return out


def ool_table(path, label_map):
    o = ool_rows(path)
    if not o:
        return ""
    kinds = [k for k in OOL_LABEL if any(kk == k for _, kk in o)]
    lines = [r"\begin{table*}[t]", r"\centering",
             r"\caption{Out-of-library plants (eight per mechanism, two runs, 14 "
             r"days). $\Delta$: return minus the rule-based incumbent's on the same "
             r"plant and noise (higher is better; 95\% bootstrap CI over plants). "
             r"Misses: deadline violations of the method (incumbent's in the last "
             r"row). Alarm: share of runs in which the latency out-of-library "
             r"detector fired, after which the network side runs the verified "
             r"fallback.}",
             r"\label{tab:ool}", r"\footnotesize", r"\setlength{\tabcolsep}{3pt}",
             r"\begin{tabular}{l" + "rrr" * len(kinds) + "}", r"\toprule",
             "Method & " + " & ".join(r"\multicolumn{3}{c}{" + OOL_LABEL[k] + "}"
                                      for k in kinds) + r" \\",
             "".join(r"\cmidrule(lr){%d-%d}" % (2 + 3 * i, 4 + 3 * i)
                     for i in range(len(kinds))),
             " & " + " & ".join([r"$\Delta$ vs incumbent & misses & alarm"] * len(kinds))
             + r" \\", r"\midrule"]
    for m in OOL_ORDER + ["rule"]:
        cells = []
        for k in kinds:
            x = o.get((m, k))
            if x is None:
                cells += ["--"] * 3
            elif m == "rule":
                cells += ["0", f"{x['misses']:.0f}", "--"]
            else:
                cells += [fmt_ci(x["d_rule"], 0), f"{x['misses']:.0f}",
                          f"{100 * x['alarm']:.0f}" + r"\%"]
        lines.append(label_map.get(m, m) + " & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""]
    return "\n".join(lines)

"""Small-panel versions of the paper's data figures.

Every graph is drawn by the original plotting functions (cp_make_figures,
cp_figs_extra, cp_iter2, cp_audit_summary), but on axes laid out at final
print size: each panel has the same box (axes plus fixed margins), text is
7 pt, legends move to a panel of their own, and each panel is saved as its own
PDF (<figure>_a.pdf, <figure>_b.pdf, ..., <figure>_leg.pdf) in paper/figs.
The paper sets them three to a row at their natural size, 0.26 of the text
width (wide panels 0.52).

    python experiments/cp_figs_small.py
"""
from __future__ import annotations

import json
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.transforms import Bbox  # noqa: E402

import cp_make_figures as MF  # noqa: E402  (sets its own rcParams on import)
import cp_figs_extra as XF  # noqa: E402
import cp_iter2 as I2  # noqa: E402
from cp_common import CP_RESULTS, ROOT  # noqa: E402

FIGS = ROOT / "paper" / "figs"
plt.rcParams.update({
    "font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5,
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Nimbus Roman", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix", "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5, "lines.linewidth": 1.0,
    "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})

# panel geometry in inches: axes width/height and the margins around them.
# A panel is 1.68 in wide, 0.26 of the 6.48 in text width, so the paper sets it
# at its natural size (\pw = 0.26\textwidth); wide panels are 0.52.
PANEL = dict(aw=1.20, ah=0.95, L=0.42, R=0.06, B=0.36, T=0.18)
WIDE = 0.52 * 6.476 - 0.48  # axes width of a 0.52-textwidth panel
SPEC: dict = {}
# axis labels that are too long for a small panel
SHORT = {
    "realised value (paired, to day 14)": "realised value (to day 14)",
    "14-day return lost relative to full CauProbe (95% CI)":
        "return lost vs full CauProbe (95% CI)",
    "price of one expert call (reward units)": "price per expert call",
}
WRAP = {"noisy+LPWAN": "noisy+\nLPWAN", "poor docs": "poor\ndocs",
        "struct. change": "struct.\nchange"}


def _geometry():
    s = dict(PANEL)
    s.update(SPEC)
    return s, s["L"] + s["aw"] + s["R"], s["B"] + s["ah"] + s["T"]


def _subplots(nrows=1, ncols=1, figsize=None, sharex=False, sharey=False,
              gridspec_kw=None, **kw):
    s, pw, ph = _geometry()
    fig = plt.figure(figsize=(ncols * pw, ph))
    axes = []
    for i in range(ncols):
        share = axes[0] if (sharey and axes) else None
        axes.append(fig.add_axes([(i * pw + s["L"]) / (ncols * pw), s["B"] / ph,
                                  s["aw"] / (ncols * pw), s["ah"] / ph],
                                 sharey=share))
    fig._cp_panels = (ncols, pw, ph)
    return (fig, axes[0]) if ncols == 1 else (fig, np.array(axes))


# figures of the first test set, where "CauProbe" is the first iteration
TEST1 = {"cp_fig2_learning", "cp_fig3_scenarios", "cp_fig4_ablation",
         "cp_fig8_tail", "cp_fig6_price"}


def _v1(text, name):
    return re.sub(r"CauProbe(?!-)", "CauProbe-1", text) if name in TEST1 else text


def save(fig, name):
    n, pw, ph = fig._cp_panels
    handles, labels = [], []
    for ax in fig.axes:
        h, lab = ax.get_legend_handles_labels()
        for hh, ll in zip(h, lab):
            ll = _v1(ll, name)
            if ll not in labels and not ll.startswith("_"):
                handles.append(hh)
                labels.append(ll)
        if ax.get_legend() is not None:
            ax.get_legend().remove()
        ax.set_xlabel(_v1(SHORT.get(ax.get_xlabel(), ax.get_xlabel()), name))
        ax.set_ylabel(_v1(SHORT.get(ax.get_ylabel(), ax.get_ylabel()), name))
        if name in TEST1 and name == "cp_fig4_ablation":
            ax.set_yticks(ax.get_yticks())
            ax.set_yticklabels([_v1(t.get_text(), name) for t in ax.get_yticklabels()])
        ticks = [t.get_text() for t in ax.get_xticklabels()]
        if "scenarios" in name and any(t in WRAP for t in ticks):
            # scenario names on two lines, so that neighbours do not run together
            ax.set_xticks(ax.get_xticks())
            ax.set_xticklabels([WRAP.get(t, t) for t in ticks])
        if any(re.fullmatch(r"A\dB\dC\d", t) for t in ticks):
            # mechanism combinations: upright text does not fit eight to a panel
            ax.tick_params(axis="x", labelrotation=90, labelsize=6)
    for i in range(n):
        fig.savefig(FIGS / f"{name}_{'abcdef'[i]}.pdf",
                    bbox_inches=Bbox([[i * pw, 0], [(i + 1) * pw, ph]]))
    if handles:
        _, _, ph1 = _geometry()
        # legends take one ordinary panel's width, also beside a wide panel
        lf = plt.figure(figsize=(PANEL["L"] + PANEL["aw"] + PANEL["R"], ph1))
        lf.legend(handles, labels, loc="center", frameon=False, ncol=1,
                  handlelength=1.8, borderaxespad=0)
        lf.savefig(FIGS / f"{name}_leg.pdf")
        plt.close(lf)
    plt.close(fig)
    print("wrote", name, n, "panel(s)", "+ legend" if handles else "")


def fig_audit(save_fn):
    """Same content as the figure of cp_audit_summary.py."""
    p = CP_RESULTS / "test2" / "probe_audit.jsonl"
    rows = [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7), sharey=True)
    for ax, it, title in zip(axes, ("cauprobe", "cauprobe-v2"),
                             ("CauProbe-1", "CauProbe-2")):
        for sw, mk in (("wifi", "o"), ("lpwan", "^")):
            for b, col in ((0, "#1f5fa8"), (1, "#d1781e")):
                rr = [r for r in rows if r["method"] == it and r["switch"] == sw
                      and r["combo"][1] == b]
                if rr:
                    ax.scatter([r["pred_net"] for r in rr], [r["realised"] for r in rr],
                               marker=mk, s=10, color=col, alpha=0.75, lw=0,
                               label=f"{'edge' if b == 0 else 'WAN'}-bound, "
                                     f"{'Wi-Fi' if sw == 'wifi' else 'LPWAN'}")
        lim = max(1.0, max(abs(r["pred_net"]) for r in rows))
        ax.axhline(0, color="k", lw=0.6, ls=":")
        ax.plot([0, lim], [0, lim], color="#888", lw=0.8, ls="--",
                label="prediction = realisation")
        ax.set_xscale("symlog", linthresh=1)
        ax.set_yscale("symlog", linthresh=1)
        ax.set_xlabel("predicted net value of the probe")
        ax.set_title(title)
    axes[0].set_ylabel("realised value (paired, to day 14)")
    save_fn(fig, "cp_fig11_audit")


def fig_pooled(save_fn):
    """Pooled second and third test sets (Table 8, cp_pooled.py): the upper
    tail of the gap to the known-plant reference over the plant-scenarios, and
    the share of runs ending on a wrong network mechanism in each set."""
    import cp_pooled as PO
    D, W = PO.load()
    ms = [("cauprobe-v2", "#1f5fa8", "CauProbe-2"),
          ("cauprobe-noprobe", "#8fb7e8", "expert + greedy"),
          ("greedy", "#7f7f7f", "greedy (same belief)")]
    keys = [k for k, v in D.items() if all(m in v for m, _, _ in ms)]
    runs = set.intersection(*(set(W[m]) for m, _, _ in ms))
    fig, axes = plt.subplots(1, 2)
    q = (np.arange(len(keys)) + 0.5) / len(keys)
    tail = q >= 0.75
    for m, col, lab in ms:
        g = np.sort([D[k][m]["gap"] for k in keys])
        axes[0].plot(q[tail], g[tail], color=col, lw=1.2, label=lab)
    axes[0].axvspan(0.9, 1.0, color="k", alpha=0.07, lw=0)
    axes[0].set_xlim(0.75, 1.0)
    axes[0].set_xticks([0.8, 0.9, 1.0])
    axes[0].set_xticklabels(["0.8", "0.9", "1"])
    axes[0].set_xlabel("quantile of plant-scenarios")
    axes[0].set_ylabel("gap to reference (14 days)")
    sets = (("test2", "test 2"), ("test3", "test 3"))
    w = 0.27
    for j, (m, col, _) in enumerate(ms):
        for i, (st, _) in enumerate(sets):
            v = [np.mean(W[m][k]) > 0.5 for k in runs if k[0] == st]
            axes[1].bar(i + (j - 1) * w, 100 * np.mean(v), w, color=col)
    axes[1].set_xticks(range(len(sets)))
    axes[1].set_xticklabels([s for _, s in sets])
    axes[1].set_ylabel("runs on a wrong mechanism (%)")
    save_fn(fig, "cp_fig12_pooled")


def fig_fleetcal(save_fn):
    """Fleet calibration on the 48 second-test sites, nominal notes
    (cp_fleetcal.py): reliability of the expert with its training-vocabulary
    temperatures and with temperatures refitted, before each site, to the final
    posteriors of all earlier CauProbe-2 sites; and calibration error against
    the number of earlier sites, by the source of the labels."""
    import cp_fleetcal as FC
    from causlm.probe.expert import (FACTORS, ExpertModel, expected_calibration_error,
                                     fit_temperature)
    z, lab = FC.logits("nominal"), FC.labels("nominal")
    order = [s for s, _ in FC.HELDOUT_PLANTS + FC.TEST_PLANTS + FC.TEST2_PLANTS]
    order = [s for s in order if s in z and s in lab]
    evalset = [s for s, _ in FC.TEST2_PLANTS if s in z and s in lab]
    base = ExpertModel.from_json(CP_RESULTS / "experts" / f"{FC.EXP}.json")

    def probs(tau, s):
        return [1 / (1 + np.exp(-tau[F] * z[s][F])) for F in FACTORS]

    def truth(s):
        return [lab[s]["oracle"][F] for F in FACTORS]

    P0 = sum((probs(base.tau, s) for s in evalset), [])
    P1, Y1 = [], []
    for s in evalset:
        hist = [h for h in order[:order.index(s)] if "cauprobe-v2" in lab[h]]
        if len(hist) < 8:
            continue
        tau = {F: fit_temperature([z[h][F] for h in hist],
                                  [lab[h]["cauprobe-v2"][F] for h in hist]) for F in FACTORS}
        P1 += probs(tau, s)
        Y1 += truth(s)
    Y0 = sum((truth(s) for s in evalset), [])
    fig, axes = plt.subplots(1, 2)
    ax = axes[0]
    edges = np.linspace(0, 1, 11)
    for P, Y, col, lab_ in ((P0, Y0, "#d1781e", "training vocabulary"),
                            (P1, Y1, "#1f5fa8", "fleet, CauProbe-2 labels")):
        P, Y = np.asarray(P), np.asarray(Y, float)
        ece = expected_calibration_error(P, Y)
        idx = np.clip(np.digitize(P, edges) - 1, 0, 9)
        xs, ys, ns = [], [], []
        for b in range(10):
            if (idx == b).sum() >= 5:
                xs.append(P[idx == b].mean())
                ys.append(Y[idx == b].mean())
                ns.append((idx == b).sum())
        ax.plot(xs, ys, "-", color=col, lw=1.0, label=lab_)
        ax.scatter(xs, ys, s=4 + 40 * np.asarray(ns) / len(P), color=col, lw=0, zorder=3)
        print(f"  {lab_}: ECE {ece:.3f} on {len(P)} answers")
    ax.plot([0, 1], [0, 1], color="#888", lw=0.8, ls="--")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([0, 0.5, 1])
    ax.set_xticklabels(["0", "0.5", "1"])
    ax.set_xlabel("predicted probability")
    ax.set_ylabel("observed frequency")
    ax = axes[1]
    d = json.loads((CP_RESULTS / "fleetcal.json").read_text())["nominal"]
    for src, col, ls, mk, lab_ in (
            ("cauprobe", "#5b8fd6", "-", "o", "CauProbe-1 labels"),
            ("oracle", "k", ":", None, "true mechanisms"),
            ("greedy", "#7f7f7f", "-", "s", "greedy labels"),
            ("cauprobe-noprobe", "#8fb7e8", "-", "^", "expert + greedy labels")):
        pts = d["by_fleet_size"][src]
        ax.plot([p["n"] for p in pts], [p["ece"] for p in pts], ls=ls, marker=mk, ms=3,
                color=col, lw=1.0, label=lab_)
    ax.axhline(d["calibration"]["train-vocab"]["ece"], color="#d1781e", lw=0.8, ls="--")
    ax.set_xscale("log", base=2)
    ax.set_xticks([4, 8, 16, 32, 64])
    ax.set_xticklabels(["4", "8", "16", "32", "64"])
    ax.set_ylim(0, 0.2)
    ax.set_xlabel("earlier sites in the fleet")
    ax.set_ylabel("calibration error (ECE)")
    save_fn(fig, "cp_fig13_fleetcal")


JOBS = [
    (lambda: fig_pooled(save), {}),
    (lambda: fig_fleetcal(save), {}),
    (lambda: MF.fig_learning_curve(), {}),
    (lambda: MF.fig_scenarios(), {"aw": WIDE, "B": 0.46}),
    (lambda: XF.fig_ablation(save), {"aw": WIDE - 1.48, "ah": 1.75, "L": 1.90}),
    (lambda: XF.fig_tail(save), {"B": 0.50}),
    (lambda: XF.fig_probe_timing(save), {}),
    (lambda: XF.fig_price(save), {}),
    (lambda: I2.fig_scenarios(save, I2.scen_summary()), {"aw": WIDE, "B": 0.46}),
    (lambda: I2.fig_price(save, I2.price_rows(), I2.voi_ask()), {}),
    (lambda: fig_audit(save), {}),
]

if __name__ == "__main__":
    plt.subplots = _subplots
    MF.save = save
    for job, spec in JOBS:
        SPEC.clear()
        SPEC.update(spec)
        job()

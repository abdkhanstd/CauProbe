"""Small-panel versions of the paper's data figures.

Every graph is drawn by the original plotting functions (cp_make_figures,
cp_figs_extra, cp_iter2, cp_audit_summary), but on axes laid out at final
print size: each panel has the same box (axes plus fixed margins), text is
7 pt, legends move to a panel of their own, and each panel is saved as its own
PDF (<figure>_a.pdf, <figure>_b.pdf, ..., <figure>_leg.pdf) in paper/figs.
The paper sets them two or three to a row at 0.32 of the text width.

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

# panel geometry in inches: axes width/height and the margins around them
PANEL = dict(aw=1.62, ah=1.30, L=0.48, R=0.08, B=0.40, T=0.22)
SPEC: dict = {}
# axis labels that are too long for a small panel
SHORT = {
    "realised value (paired, to day 14)": "realised value (to day 14)",
    "14-day return lost relative to full CauProbe (95% CI)":
        "return lost vs full CauProbe (95% CI)",
    "price of one expert call (reward units)": "price per expert call",
}


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
    for i in range(n):
        fig.savefig(FIGS / f"{name}_{'abcdef'[i]}.pdf",
                    bbox_inches=Bbox([[i * pw, 0], [(i + 1) * pw, ph]]))
    if handles:
        _, pw1, ph1 = _geometry()
        lf = plt.figure(figsize=(2.18, ph1))
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


JOBS = [
    (lambda: MF.fig_learning_curve(), {}),
    (lambda: MF.fig_scenarios(), {"aw": 3.80}),
    (lambda: XF.fig_ablation(save), {"aw": 1.75, "ah": 2.05, "L": 1.95}),
    (lambda: XF.fig_tail(save), {"B": 0.52}),
    (lambda: XF.fig_probe_timing(save), {}),
    (lambda: XF.fig_price(save), {}),
    (lambda: I2.fig_scenarios(save, I2.scen_summary()), {"aw": 3.80}),
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

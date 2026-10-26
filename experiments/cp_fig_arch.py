"""Figure 1: the CauProbe controller (schematic, no data)."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402


def draw(save):
    fig, ax = plt.subplots(figsize=(7.4, 3.6))
    ax.set_xlim(0, 104)
    ax.set_ylim(0, 50)
    ax.axis("off")

    def box(x, y, w, h, text, fc="#eef3fa", ec="#1f5fa8", fs=7.6):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.5",
                                    fc=fc, ec=ec, lw=1.0))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fs, linespacing=1.25)

    def arrow(x0, y0, x1, y1, c="#333333"):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                                     mutation_scale=9, lw=0.9, color=c))

    def label(x, y, t, c="#333333"):
        ax.text(x, y, t, fontsize=6.2, ha="center", va="center", color=c,
                linespacing=1.15)

    # top row
    box(1, 29, 19, 13, "Plant\nunknown\nmechanisms A-D\n(hidden occupancy)",
        fc="#f4f4f4", ec="#555555")
    box(29, 29, 21, 13, "Posterior over\nstructural hypotheses\n"
        r"particles $(s,\theta,w)$" "\ntempered, fixed share")
    box(58, 31, 19, 9, "Structural value\n" r"$V(\beta)$, EVPI," "\n"
        r"$H_{\mathrm{eff}}$, $\kappa$")
    box(85, 31, 18, 9, "Choose: act,\n" r"probe $(a,m)$, ask," "\nnight test, fall back",
        fc="#fdf0e2", ec="#d1781e")
    # bottom row
    box(1, 10, 19, 9, "Commissioning text\n(notes, tickets)", fc="#f4f4f4",
        ec="#555555")
    box(29, 10, 21, 9, "On-device expert\n(0.5B SLM + LoRA)\ncalibrated "
        r"$p(z \mid s)$", fc="#eef7f0", ec="#2e9b5e")
    box(58, 10, 19, 9, "Relative chance\nconstraint +\nsurprise CUSUM",
        fc="#fbeaea", ec="#c23b3b")
    box(85, 10, 18, 9, "Verified\nfallback", fc="#f4f4f4", ec="#555555")
    # arrows and labels
    arrow(20.6, 35.5, 28.4, 35.5)
    label(24.6, 44.2, "own actions:\n" r"interventional $y$")
    arrow(50.6, 35.5, 57.4, 35.5)
    arrow(77.6, 35.5, 84.4, 35.5)
    arrow(20.6, 14.5, 28.4, 14.5)
    label(24.5, 5.8, "read only if\n"
          r"$H_{\mathrm{eff}}\,\mathrm{VOI}_{ask} > c_{ask}$")
    arrow(39.5, 19.6, 39.5, 28.4, c="#2e9b5e")
    label(45.0, 24.0, "likelihood\nof answer", c="#2e9b5e")
    arrow(67.5, 19.6, 67.5, 30.4)
    label(72.5, 24.0, "admissible\nset", c="#c23b3b")
    arrow(94, 30.4, 94, 19.6, c="#c23b3b")
    label(99.5, 24.0, "alarm", c="#c23b3b")
    ax.plot([94, 94, 10.5], [40.6, 47.0, 47.0], color="#d1781e", lw=0.9)
    ax.add_patch(FancyArrowPatch((10.5, 47.0), (10.5, 42.6), arrowstyle="-|>",
                                 mutation_scale=9, lw=0.9, color="#d1781e"))
    label(52, 48.7, "setting, probe or step test applied to the plant",
          c="#d1781e")
    ax.text(52, 1.0, "Every information action is priced in reward units: lost "
            "reward and reconfiguration (probe), Joules and contention (ask).",
            fontsize=6.8, ha="center", style="italic")
    save(fig, "cp_fig1_architecture")

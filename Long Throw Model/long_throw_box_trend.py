"""
Long throw-ins that land in the penalty area
==============================================
Eredivisie 2022/23 -> 2025/26. Same long-throw-in universe as
long_throw_box_shot_trend.py (qualifier 107, real length >=25m) and the
same box definition (x >= 83, 21.1 <= y <= 78.9), but isolates the first
funnel stage on its own: does the throw's landing point reach the box at
all, regardless of what happens after.

In Marc Lamberts' Meridian house style (housestyle/ package at repo root).

Usage: python3 long_throw_box_trend.py
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from housestyle import style, components  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from long_throw_box_shot_trend import season_stats, SEASONS, SEASON_LABELS  # noqa: E402

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
SOURCE = "Opta/StatsPerform - Eredivisie 2023-2026 Long Throw data"


def make_chart(stats, mode, out_path):
    palette, _ = style.apply(mode)
    box_pct = [s[1] for s in stats]
    box_delta = box_pct[-1] - box_pct[0]

    fig = plt.figure(figsize=(11.5, 8.2))

    # --- main panel: 4-season trend line -------------------------------
    ax = fig.add_axes([0.09, 0.14, 0.84, 0.56])
    xw = np.arange(len(SEASONS))
    ax.plot(xw, box_pct, marker="o", markersize=7, color=palette["accent"], linewidth=2.6, zorder=4)

    for i, v in enumerate(box_pct):
        ax.text(i, v + 0.35, f"{v:.1f}%", ha="center", fontsize=9.5, color=palette["accent"], fontweight="bold")

    ax.set_xticks(xw)
    ax.set_xticklabels(SEASON_LABELS, fontsize=10.5)
    ax.set_ylabel("Share of long throw-ins (%)", fontsize=10.5, color=palette["ink_secondary"])
    ax.set_xlim(-0.3, len(SEASONS) - 0.1)
    ax.set_ylim(0, max(box_pct) + 4)
    ax.tick_params(colors=palette["ink_muted"])
    ax.grid(True, axis="y", color=palette["grid"], linewidth=0.7, zorder=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title("Long throw-ins landing in the penalty area, by season",
                fontsize=11.5, color=palette["ink_secondary"], loc="left", pad=10)

    components.header(
        fig, kicker="Long throw-ins",
        title="Roughly 1 in 6 long throw-ins reaches the penalty area",
        dek=f"Eredivisie 2022/23-2025/26  ·  throw lands in the box: "
            f"{box_pct[0]:.1f}% -> {box_pct[-1]:.1f}%  ({box_delta:+.1f}pp)",
        palette=palette)
    components.footer(fig, source=SOURCE, palette=palette)

    fig.savefig(out_path, dpi=200, facecolor=palette["surface"])
    plt.close(fig)
    print("Saved:", out_path)


def main():
    stats = []
    for s in SEASONS:
        n, box_pct, combo_pct = season_stats(s)
        stats.append((n, box_pct, combo_pct))
        print(f"{s}: n={n}  box={box_pct:.2f}%")

    for mode in ("light", "dark"):
        d = os.path.join(OUT_DIR, "Visual - Dark" if mode == "dark" else "Visual - Light")
        os.makedirs(d, exist_ok=True)
        out_path = os.path.join(d, "long_throw_box_trend_2022_2026.png")
        make_chart(stats, mode, out_path)


if __name__ == "__main__":
    main()

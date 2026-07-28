"""
Where open-play crosses come from: left flank vs right flank
==============================================================
Eredivisie 2025/26 — uses the Ecuador-2026-trained cross models' scored
output (eredivisie_open_play_cross_events_scored.csv, produced by
score_eredivisie_crosses.py). wide_channel splits origin y < 50 (left) vs
y >= 50 (right); pred_cross_delivery_value / pred_cross_completion /
pred_cross_chance_creation are the model's per-cross predictions.

In Marc Lamberts' Meridian house style (housestyle/ package at repo root).

Usage: python3 cross_origin_left_vs_right.py
"""
import os
import sys

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
from mplsoccer import Pitch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from housestyle import style, components, colors  # noqa: E402

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(OUT_DIR, "eredivisie_open_play_cross_events_scored.csv")
SOURCE = ("Opta event data, Eredivisie 2025/26 · open-play crosses only (excludes corners/"
          "free kicks/throw-ins) · scored with the Ecuador 2026-trained cross models")


def visual_dir(theme):
    d = os.path.join(OUT_DIR, "Visual - Dark" if theme == "dark" else "Visual - Light")
    os.makedirs(d, exist_ok=True)
    return d


def seq_cmap(mode):
    seq = colors.SEQUENTIAL_BLUE_DARK if mode == "dark" else colors.SEQUENTIAL_BLUE_LIGHT
    hexes = [seq[k] for k in sorted(seq)]
    return LinearSegmentedColormap.from_list("meridian_blue", hexes)


def make_chart(df, mode, out_path):
    palette, _ = style.apply(mode)
    left = df[df["wide_channel"] == "left"]
    right = df[df["wide_channel"] == "right"]

    n_left, n_right = len(left), len(right)
    n_total = n_left + n_right
    share_left = n_left / n_total * 100
    share_right = n_right / n_total * 100
    dv_left = left["pred_cross_delivery_value"].mean()
    dv_right = right["pred_cross_delivery_value"].mean()
    comp_left = left["pred_cross_completion"].mean() * 100
    comp_right = right["pred_cross_completion"].mean() * 100
    chance_left = left["pred_cross_chance_creation"].mean() * 100
    chance_right = right["pred_cross_chance_creation"].mean() * 100

    fig = plt.figure(figsize=(14.5, 8.6))

    # --- pitch: origin density, full attacking third -----------------
    ax_pitch = fig.add_axes([0.055, 0.10, 0.55, 0.62])
    pitch = Pitch(pitch_type="opta", pitch_color=palette["surface"], line_color=palette["axis"],
                 linewidth=1.1, half=True, line_zorder=2)
    pitch.draw(ax=ax_pitch)
    stats = pitch.bin_statistic(df["x"], df["y"], statistic="count", bins=(20, 16))
    stats["statistic"] = np.where(stats["statistic"] == 0, np.nan, stats["statistic"])
    pitch.heatmap(stats, ax=ax_pitch, cmap=seq_cmap(mode), edgecolors=palette["surface"], linewidth=0.3, zorder=1)
    ax_pitch.axhline(50, color=palette["accent"], linewidth=1.3, linestyle=(0, (5, 4)), zorder=3)
    ax_pitch.text(52, 96, "LEFT FLANK", fontsize=10, fontweight="bold", color=palette["ink_primary"],
                 ha="left", va="top", family="sans-serif")
    ax_pitch.text(52, 4, "RIGHT FLANK", fontsize=10, fontweight="bold", color=palette["ink_primary"],
                 ha="left", va="bottom", family="sans-serif")
    ax_pitch.text(98, 96, f"{n_left:,} crosses  ·  {share_left:.1f}%", fontsize=9.5,
                 color=palette["ink_secondary"], ha="right", va="top", family="sans-serif")
    ax_pitch.text(98, 4, f"{n_right:,} crosses  ·  {share_right:.1f}%", fontsize=9.5,
                 color=palette["ink_secondary"], ha="right", va="bottom", family="sans-serif")
    ax_pitch.set_title("Origin density, all open-play crosses", fontsize=11.5,
                       color=palette["ink_secondary"], pad=10, loc="left")

    # --- bar panel: model-rated quality, left vs right ----------------
    ax_bar = fig.add_axes([0.665, 0.14, 0.30, 0.55])
    metrics = ["Share of\ncrosses (%)", "Pred. delivery\nvalue (x100)", "Pred.\ncompletion (%)", "Pred. chance\ncreation (%)"]
    left_vals = [share_left, dv_left * 100, comp_left, chance_left]
    right_vals = [share_right, dv_right * 100, comp_right, chance_right]

    yw = np.arange(len(metrics))
    h = 0.34
    ax_bar.barh(yw + h / 2, left_vals, height=h, color=palette["accent"], zorder=3, label="Left flank")
    ax_bar.barh(yw - h / 2, right_vals, height=h, color=palette["axis"], zorder=3, label="Right flank")
    for i, (lv, rv) in enumerate(zip(left_vals, right_vals)):
        ax_bar.text(lv + max(left_vals + right_vals) * 0.02, i + h / 2, f"{lv:.1f}", va="center",
                    fontsize=9.5, color=palette["ink_primary"], fontweight="bold")
        ax_bar.text(rv + max(left_vals + right_vals) * 0.02, i - h / 2, f"{rv:.1f}", va="center",
                    fontsize=9.5, color=palette["ink_secondary"])
    ax_bar.set_yticks(yw)
    ax_bar.set_yticklabels(metrics, fontsize=9.5)
    ax_bar.set_xlim(0, max(left_vals + right_vals) * 1.22)
    ax_bar.invert_yaxis()
    ax_bar.tick_params(axis="x", colors=palette["ink_muted"], labelsize=9)
    ax_bar.grid(True, axis="x", color=palette["grid"], linewidth=0.7, zorder=0)
    ax_bar.set_axisbelow(True)
    for spine in ax_bar.spines.values():
        spine.set_visible(False)
    ax_bar.legend(loc="upper center", bbox_to_anchor=(0.5, 1.16), ncol=2, frameon=False,
                 fontsize=9.5, labelcolor=palette["ink_secondary"])

    components.header(
        fig, kicker="Crosses",
        title="Left-flank crosses edge out the right on quality, despite an even split",
        dek=f"Eredivisie 2025/26  ·  {n_total:,} open-play crosses  ·  "
            f"{share_left:.1f}% left vs {share_right:.1f}% right  ·  "
            f"model-predicted delivery value {dv_left:.3f} (left) vs {dv_right:.3f} (right)",
        palette=palette)
    components.footer(fig, source=SOURCE, palette=palette)

    fig.savefig(out_path, dpi=200, facecolor=palette["surface"])
    plt.close(fig)
    print("Saved:", out_path)


def main():
    df = pd.read_csv(CSV_PATH)
    for mode in ("light", "dark"):
        out_path = os.path.join(visual_dir(mode), "cross_origin_left_vs_right_2025_26.png")
        make_chart(df, mode, out_path)


if __name__ == "__main__":
    main()

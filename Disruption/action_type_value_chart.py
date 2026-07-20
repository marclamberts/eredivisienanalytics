"""
One metric, one chart: mean disruption value per action, by action type.
Eredivisie 2025/26.

Answers "what kind of defending actually matters most" -- not a table of
every stat cut, just mean disruption_value per action for each of the six
linked action types, ranked.

Usage: python3 action_type_value_chart.py [out.png]
"""
import os
import sys

import matplotlib.pyplot as plt
import pandas as pd

import league_disruption_visuals as ldv
from league_disruption_visuals import add_logo

OUT_DIR = os.path.dirname(os.path.abspath(__file__))


def main():
    out_name = sys.argv[1] if len(sys.argv) > 1 else "action_type_value_chart.png"

    df = pd.read_csv(os.path.join(ldv.CSV_DIR, "all_eredivisie_disruption_values.csv"))
    g = df.groupby("action_type").agg(
        n=("disruption_value", "size"),
        mean_value=("disruption_value", "mean"),
    ).reset_index()
    g["mean_value_x1000"] = g["mean_value"] * 1000
    g = g.sort_values("mean_value_x1000", ascending=True)

    for theme in ("dark", "light"):
        ldv.set_theme(theme)
        out_path = os.path.join(ldv.visual_dir(theme), out_name)

        fig, ax = plt.subplots(figsize=(11, 6.5))
        fig.patch.set_facecolor(ldv.BG)
        ax.set_facecolor(ldv.BG)

        vmax = g["mean_value_x1000"].max()
        colors = [ldv.GOLD_RAMP(0.3 + 0.6 * (v / vmax)) for v in g["mean_value_x1000"]]
        ax.barh(g["action_type"], g["mean_value_x1000"], color=colors, height=0.6, zorder=3)

        for i, (val, n) in enumerate(zip(g["mean_value_x1000"], g["n"])):
            ax.text(val + vmax * 0.02, i, f"{val:.2f}", va="center", fontsize=12,
                   color=ldv.TEXT_MAIN, fontweight="bold")

        ax.tick_params(axis="y", colors=ldv.TEXT_MAIN, labelsize=12, length=0)
        ax.tick_params(axis="x", colors=ldv.TEXT_SUB, labelsize=9.5)
        for spine in ("top", "right", "left"):
            ax.spines[spine].set_visible(False)
        ax.spines["bottom"].set_color(ldv.PITCH_LINE)
        ax.grid(axis="x", color=ldv.PITCH_LINE, linewidth=0.6, alpha=0.5, zorder=0)
        ax.set_xlim(0, vmax * 1.18)
        ax.set_xlabel("Mean disruption value per action, x1000", fontsize=10.5, color=ldv.TEXT_SUB)

        fig.text(0.5, 0.965, "What Kind of Defending Actually Matters", fontsize=22,
                 fontweight="bold", ha="center", color=ldv.TEXT_MAIN)
        fig.text(0.5, 0.92,
                 "Eredivisie 2025/26  ·  Season  ·  mean disruption value per action, by action type",
                 fontsize=11, ha="center", color=ldv.TEXT_SUB)

        fig.text(0.98, 0.02, "Marc Lamberts", fontsize=9.5, ha="right", color=ldv.TEXT_FOOT, style="italic")
        fig.text(0.02, 0.02, "Data via Opta | disruption value = P(pass completes) x xT it would "
                 "have added; higher = this action type more often breaks up genuinely "
                 "threatening passes, not just any pass", fontsize=7.3, color=ldv.TEXT_FOOT)

        fig.subplots_adjust(left=0.18, right=0.93, top=0.86, bottom=0.14)
        add_logo(fig)
        fig.savefig(out_path, dpi=200, facecolor=ldv.BG)
        plt.close(fig)
        print("Saved:", out_path)


if __name__ == "__main__":
    main()

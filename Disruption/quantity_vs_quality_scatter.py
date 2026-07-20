"""
Disruption quantity vs quality - Eredivisie 2025/26
======================================================
Separates "makes a lot of tackles" from "actually stops the passes that
matter": x = number of linked defensive actions this season (volume),
y = mean disruption_value per action (quality -- see disruption_value_model.py).
A player can rank top-5 in raw action count and bottom-half in value per
action if most of what they stop has no forward threat (see: most of
T. Blokzijl's 55 actions in the earlier leaderboard).

Filters to players with >= MIN_ACTIONS linked disruptions so one lucky
early strike doesn't produce a false "100% quality" outlier.

Usage: python3 quantity_vs_quality_scatter.py [out.png]
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import league_disruption_visuals as ldv
from league_disruption_visuals import add_logo

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
MIN_ACTIONS = 5
N_LABELS = 14


def main():
    out_name = sys.argv[1] if len(sys.argv) > 1 else "quantity_vs_quality_scatter.png"

    df = pd.read_csv(os.path.join(ldv.CSV_DIR, "disruption_value_player_summary.csv"))
    df = df[df["actions_linked"] >= MIN_ACTIONS].copy()
    df["quality_x1000"] = df["mean_disruption_value"] * 1000

    x_med = df["actions_linked"].median()
    y_med = df["quality_x1000"].median()

    for theme in ("dark", "light"):
        ldv.set_theme(theme)
        out_path = os.path.join(ldv.visual_dir(theme), out_name)

        fig, ax = plt.subplots(figsize=(12.5, 10))
        fig.patch.set_facecolor(ldv.BG)
        ax.set_facecolor(ldv.BG)

        vmax = df["total_disruption_value_x1000"].max()
        sizes = 60 + 500 * (df["total_disruption_value_x1000"] / vmax)
        colors = [ldv.GOLD_RAMP(0.25 + 0.65 * (v / vmax)) for v in df["total_disruption_value_x1000"]]
        ax.scatter(df["actions_linked"], df["quality_x1000"], s=sizes, color=colors,
                  edgecolors="#ffe9b8", linewidth=0.5, alpha=0.88, zorder=3)

        ax.axvline(x_med, color=ldv.PITCH_LINE, lw=1.1, zorder=1)
        ax.axhline(y_med, color=ldv.PITCH_LINE, lw=1.1, zorder=1)

        quad_kw = dict(fontsize=10, color=ldv.TEXT_SUB, fontweight="bold", alpha=0.85)
        ax.text(df["actions_linked"].max() * 0.98, df["quality_x1000"].max() * 0.98,
               "HIGH VOLUME, HIGH QUALITY", ha="right", va="top", **quad_kw)
        ax.text(df["actions_linked"].min() + 0.3, df["quality_x1000"].max() * 0.98,
               "SPECIALIST\n(low volume, high quality)", ha="left", va="top", **quad_kw)
        ax.text(df["actions_linked"].max() * 0.98, df["quality_x1000"].min() * 1.15 - 0.15,
               "BUSY, LOW IMPACT", ha="right", va="bottom", **quad_kw)

        label_pool = pd.concat([
            df.sort_values("total_disruption_value", ascending=False).head(8),
            df[df["actions_linked"] <= x_med].sort_values("quality_x1000", ascending=False).head(6),
        ]).drop_duplicates(subset="player_id").head(N_LABELS)

        texts = []
        for _, r in label_pool.iterrows():
            texts.append(ax.annotate(
                f"{r['player_name']}", xy=(r["actions_linked"], r["quality_x1000"]),
                xytext=(6, 5), textcoords="offset points", fontsize=8.6, color=ldv.TEXT_MAIN, zorder=5))

        try:
            from adjustText import adjust_text
            adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle="-", color="#5a6070", lw=0.6))
        except ImportError:
            pass

        ax.set_xlabel("Linked defensive actions this season (volume)", fontsize=10.5, color=ldv.TEXT_SUB)
        ax.set_ylabel("Mean disruption value per action, x1000 (quality)", fontsize=10.5, color=ldv.TEXT_SUB)
        ax.tick_params(colors=ldv.TEXT_SUB, labelsize=9)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color(ldv.PITCH_LINE)
        ax.grid(color=ldv.PITCH_LINE, linewidth=0.5, alpha=0.4, zorder=0)
        ax.set_xlim(left=-3)
        ax.set_ylim(bottom=-0.05 * df["quality_x1000"].max())

        fig.text(0.5, 0.965, "Disruption: Volume vs. Quality", fontsize=24, fontweight="bold",
                 ha="center", color=ldv.TEXT_MAIN)
        fig.text(0.5, 0.935,
                 f"Eredivisie 2025/26  ·  Season  ·  players with ≥{MIN_ACTIONS} linked "
                 f"disruptions ({len(df)} players)  ·  bubble size = total disruption value",
                 fontsize=10.5, ha="center", color=ldv.TEXT_SUB)

        fig.text(0.98, 0.012, "Marc Lamberts", fontsize=9.5, ha="right", color=ldv.TEXT_FOOT, style="italic")
        fig.text(0.02, 0.012, "Data via Opta | quality = value denied per action, not just how "
                 "often a player makes a defensive action", fontsize=7.5, color=ldv.TEXT_FOOT)

        fig.subplots_adjust(left=0.11, right=0.96, top=0.90, bottom=0.08)
        add_logo(fig)
        fig.savefig(out_path, dpi=200, facecolor=ldv.BG)
        plt.close(fig)
        print("Saved:", out_path)


if __name__ == "__main__":
    main()

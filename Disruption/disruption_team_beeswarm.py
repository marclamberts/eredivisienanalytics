"""
Team beeswarm: player disruption value per90, z-scored - Eredivisie 2025/26
==============================================================================
Same idea as GDA/gda_team_beeswarm.png (one row per team, a bee-swarm of
that team's players, yellow tick = team median) but for disruption value
instead of GDA, and standardized to z-scores rather than the raw metric --
disruption_value_per90 is a tiny, awkwardly-scaled xT-derived number
(typically 0.0001-0.005), so z-scoring it ("how many standard deviations
above/below the league-average disruptor is this player") is far more
readable than the raw units, and it's what most recruitment-style beeswarm
charts actually plot.

z is computed only over players with >= MIN_ACTIONS linked disruptions
(same floor as quantity_vs_quality_scatter.py), so a player with one lucky
early denial doesn't get a wild z-score off a sample of one.

Usage: python3 disruption_team_beeswarm.py [out.png]
"""
import os
import sys

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
import seaborn as sns

import league_disruption_visuals as ldv
from league_disruption_visuals import add_logo

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
MIN_ACTIONS = 5


def main():
    out_name = sys.argv[1] if len(sys.argv) > 1 else "disruption_team_beeswarm.png"

    df = pd.read_csv(os.path.join(ldv.CSV_DIR, "disruption_value_player_summary.csv"))
    df = df[df["actions_linked"] >= MIN_ACTIONS].copy()

    mu, sigma = df["disruption_value_per90"].mean(), df["disruption_value_per90"].std()
    df["z_raw"] = (df["disruption_value_per90"] - mu) / sigma
    # A couple of low-sample players (e.g. 5 matches, one big denial) sit past
    # z=6 and would otherwise stretch the whole axis so hard that the bulk of
    # normal players compresses into a sliver on the left. Capping at +-3
    # (still >99th percentile of a normal distribution) keeps their outlier
    # status visible without distorting everyone else's position.
    df["z"] = df["z_raw"].clip(-3, 3)
    n_capped = int((df["z_raw"].abs() > 3).sum())

    team_order = (df.groupby("team_name")["z"].median()
                 .sort_values(ascending=False).index.tolist())

    for theme in ("dark", "light"):
        ldv.set_theme(theme)
        out_path = os.path.join(ldv.visual_dir(theme), out_name)
        diverging = LinearSegmentedColormap.from_list("div", [ldv.BLUE, "#5a6070", ldv.GOLD])

        sns.set_style("white")
        fig, ax = plt.subplots(figsize=(13, 11))
        fig.patch.set_facecolor(ldv.BG)
        ax.set_facecolor(ldv.BG)

        sns.swarmplot(data=df, x="z", y="team_name", order=team_order, hue="z",
                     palette=diverging, hue_norm=(-3, 3), size=7,
                     edgecolor=ldv.BG, linewidth=0.5, ax=ax, legend=False)

        if ax.collections:
            coll = ax.collections[-1]
            offsets = coll.get_offsets()
            # match each plotted point back to its row by (y, x) so marker size
            # can encode sample size (actions_linked) without affecting the
            # swarm's collision layout, which swarmplot already computed at a
            # uniform size
            sizes = []
            df_sorted = df.copy()
            for x, y in offsets:
                row_team = team_order[int(round(y))]
                cand = df_sorted[(df_sorted["team_name"] == row_team) &
                                 (np.isclose(df_sorted["z"], x, atol=1e-6))]
                sizes.append(40 + 9 * cand["actions_linked"].iloc[0] if len(cand) else 60)
            coll.set_sizes(sizes)

        for i, team in enumerate(team_order):
            med = df.loc[df["team_name"] == team, "z"].median()
            ax.plot([med, med], [i - 0.32, i + 0.32], color=ldv.GOLD, lw=2.4, zorder=5, solid_capstyle="round")

        ax.axvline(0, color=ldv.PITCH_LINE, lw=1.2, zorder=1)
        ax.set_xlabel("Disruption value per 90, z-score (league-standardized)", fontsize=11, color=ldv.TEXT_SUB)
        ax.set_ylabel("")
        ax.tick_params(colors=ldv.TEXT_SUB, labelsize=10)
        for label in ax.get_yticklabels():
            label.set_color(ldv.TEXT_MAIN)
        for spine in ("top", "right", "left"):
            ax.spines[spine].set_visible(False)
        ax.spines["bottom"].set_color(ldv.PITCH_LINE)
        ax.grid(axis="x", color=ldv.PITCH_LINE, linewidth=0.5, alpha=0.4, zorder=0)

        fig.text(0.5, 0.965, "Team Beeswarm: Disruption Value per 90", fontsize=23, fontweight="bold",
                 ha="center", color=ldv.TEXT_MAIN)
        fig.text(0.5, 0.938,
                 f"Eredivisie 2025/26  ·  Season  ·  players with ≥{MIN_ACTIONS} linked disruptions "
                 f"({len(df)} players)  ·  gold tick = team median  ·  dot size = actions linked",
                 fontsize=10.5, ha="center", color=ldv.TEXT_SUB)

        fig.text(0.98, 0.012, "Marc Lamberts", fontsize=9.5, ha="right", color=ldv.TEXT_FOOT, style="italic")
        fig.text(0.02, 0.012, f"Data via Opta | z-score = (player's value per90 - league mean) / "
                 f"league std dev, among qualifying players | {n_capped} extreme low-sample outliers "
                 f"capped at z=±3 for readability", fontsize=7.3, color=ldv.TEXT_FOOT)

        fig.subplots_adjust(left=0.20, right=0.97, top=0.90, bottom=0.06)
        add_logo(fig)
        fig.savefig(out_path, dpi=200, facecolor=ldv.BG)
        plt.close(fig)
        print("Saved:", out_path)


if __name__ == "__main__":
    main()

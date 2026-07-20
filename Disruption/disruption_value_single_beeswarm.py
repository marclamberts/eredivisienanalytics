"""
Disruption Value - single-line beeswarm. Eredivisie 2025/26.

One metric only: total_disruption_value (season sum of P(pass completes) x
xT it would have added, per disruption_value_model.py), x1000 for
readability. Every qualifying player on one swarm line, colour = the same
metric (sequential, darker/paler gold = lower, bright gold = higher), top
outliers labelled.

Usage: python3 disruption_value_single_beeswarm.py [out.png]
"""
import os
import sys

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from league_disruption_visuals import GOLD_RAMP, BG, PITCH_LINE, TEXT_SUB, TEXT_FOOT, add_logo

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
MIN_ACTIONS = 5
N_LABELS = 10


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        OUT_DIR, "disruption_value_single_beeswarm.png")

    df = pd.read_csv(os.path.join(OUT_DIR, "disruption_value_player_summary.csv"))
    df = df[df["actions_linked"] >= MIN_ACTIONS].copy()
    df["row"] = "Value"
    vmax = df["total_disruption_value_x1000"].max()

    # Deterministic staggered labels instead of adjustText: with everything
    # crammed onto one swarm line there's almost no vertical room, and
    # force-directed placement tends to collapse several labels onto the
    # same spot. Sort by x, bump to the next height tier whenever the next
    # label would sit within MIN_GAP of the previous one at the same tier --
    # uncapped, then size the figure to whatever tier depth that produces,
    # so a dense cluster of top values (which needs more tiers) never
    # collides instead of silently overflowing a fixed-height figure.
    top = df.sort_values("total_disruption_value_x1000", ascending=False).head(N_LABELS)
    top = top.sort_values("total_disruption_value_x1000", ascending=True)

    x_range = df["total_disruption_value_x1000"].max() - df["total_disruption_value_x1000"].min()
    min_gap = x_range * 0.09
    tiers_last_x = []
    placements = []
    for _, r in top.iterrows():
        x = r["total_disruption_value_x1000"]
        tier = 0
        while tier < len(tiers_last_x) and (x - tiers_last_x[tier]) < min_gap:
            tier += 1
        if tier == len(tiers_last_x):
            tiers_last_x.append(x)
        else:
            tiers_last_x[tier] = x
        placements.append((r["player_name"], x, tier))

    max_tier = max((t for _, _, t in placements), default=0)
    tier_step = 0.42
    y_top_data = 0.7 + max_tier * tier_step + 0.35

    fig, ax = plt.subplots(figsize=(14, 6.5 + max_tier * 0.55))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    sns.swarmplot(data=df, x="total_disruption_value_x1000", y="row", hue="total_disruption_value_x1000",
                 palette=GOLD_RAMP, hue_norm=(0, vmax), size=8, edgecolor=BG, linewidth=0.5,
                 ax=ax, legend=False)

    for name, x, tier in placements:
        y_label = 0.7 + tier * tier_step
        ax.plot([x, x], [0.42, y_label - 0.1], color="#5a6070", lw=0.6, zorder=5)
        ax.text(x, y_label, name, ha="center", va="bottom", fontsize=9.3, color="white",
               fontweight="bold", zorder=6)

    ax.set_yticks([])
    ax.set_ylabel("")
    ax.set_ylim(-1.1, y_top_data)
    ax.set_xlabel("Disruption value this season, x1000  (P(pass completes) x xT denied, summed)",
                 fontsize=10.5, color=TEXT_SUB)
    ax.tick_params(axis="x", colors=TEXT_SUB, labelsize=10)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(PITCH_LINE)
    ax.grid(axis="x", color=PITCH_LINE, linewidth=0.5, alpha=0.4, zorder=0)
    ax.set_xlim(left=-vmax * 0.02)

    fig.text(0.5, 0.955, "Disruption Value", fontsize=25, fontweight="bold", ha="center", color="white")
    fig.text(0.5, 0.90,
             f"Eredivisie 2025/26  ·  Season  ·  every player with ≥{MIN_ACTIONS} linked "
             f"disruptions ({len(df)} players)  ·  one dot = one player",
             fontsize=11, ha="center", color=TEXT_SUB)

    fig.text(0.98, 0.025, "Marc Lamberts", fontsize=9.5, ha="right", color=TEXT_FOOT, style="italic")
    fig.text(0.02, 0.025, "Data via Opta | disruption_value_model.py", fontsize=7.5, color=TEXT_FOOT)

    fig.subplots_adjust(left=0.03, right=0.98, top=0.83, bottom=0.13)
    add_logo(fig)
    fig.savefig(out_path, dpi=200, facecolor=BG)
    plt.close(fig)
    print("Saved:", out_path)


if __name__ == "__main__":
    main()

"""
Horizontal disruption-VALUE pitch map for one player - Eredivisie 2025/26
============================================================================
Every linked defensive action for a player, plotted on a horizontal
mplsoccer Pitch, sized and coloured by disruption_value (P(pass completes)
x threat it would have created -- see disruption_value_model.py). Each
action's x/y is normalized to "distance from this player's own goal" using
that team's attacking direction for the match/period, so actions from
different matches/periods/halves all sit on one consistent orientation
(own goal on the left, opponent goal on the right) instead of being mixed
across whichever way the team happened to be attacking that half.

Usage: python3 plot_player_disruption_value_pitch.py "<player name>" [out.png]
"""
import os
import sys

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from mplsoccer import Pitch

import build_disruption_model as bdm
import league_disruption_visuals as ldv
from league_disruption_visuals import compute_attack_directions, add_logo

OUT_DIR = os.path.dirname(os.path.abspath(__file__))



def main():
    args = [a for a in sys.argv[1:] if a]
    if not args:
        raise SystemExit("Usage: python3 plot_player_disruption_value_pitch.py \"<player name>\" [out.png]")
    player_name = args[0]
    safe_name = player_name.replace(" ", "_").replace(".", "")
    out_name = args[1] if len(args) > 1 else f"pitch_disruption_value_{safe_name}.png"

    values = pd.read_csv(os.path.join(ldv.CSV_DIR, "all_eredivisie_disruption_values.csv"))
    player = values[values["player_name"] == player_name].copy()
    if player.empty:
        raise SystemExit(f"No linked disruptions found for '{player_name}' in "
                         f"all_eredivisie_disruption_values.csv -- check spelling")

    print(f"{len(player)} linked disruptions for {player_name}. Normalizing to a common "
          f"attacking direction...")
    directions = compute_attack_directions()
    dir_vals = np.array([
        directions.get((mf, cid, p), 1)
        for mf, cid, p in zip(player["match_file"], player["contestant_id"], player["period_id"])
    ])
    player["x_own"] = np.where(dir_vals == 1, player["x"], bdm.GOAL_X - player["x"])
    player["y_own"] = np.where(dir_vals == 1, player["y"], 68.0 - player["y"])

    team_name = player["team_name"].iloc[0]
    n = len(player)
    total_value = player["disruption_value"].sum() * 1000
    nonzero = (player["disruption_value"] > 0).sum()

    for theme in ("dark", "light"):
        ldv.set_theme(theme)
        out_path = os.path.join(ldv.visual_dir(theme), out_name)

        fig = plt.figure(figsize=(13.5, 9.2))
        fig.patch.set_facecolor(ldv.BG)
        pitch = Pitch(pitch_type="uefa", pitch_color=ldv.BG, line_color=ldv.PITCH_LINE,
                     linewidth=1.2, half=False)
        ax = fig.add_axes([0.04, 0.17, 0.92, 0.62])
        pitch.draw(ax=ax)
        ax.annotate("", xy=(102, -4.5), xytext=(3, -4.5),
                   arrowprops=dict(arrowstyle="-|>", color=ldv.TEXT_SUB, lw=1.2))
        ax.text(52.5, -7.5, "direction of attack (own goal → opponent goal)",
               ha="center", fontsize=8.5, color=ldv.TEXT_SUB)

        fig.text(0.5, 0.965, player_name, fontsize=26, fontweight="bold", ha="center", color=ldv.TEXT_MAIN)
        fig.text(0.5, 0.93,
                 f"Disruption Value Map  ·  {team_name}  ·  Eredivisie 2025/26  ·  Season  ·  "
                 f"{n} linked actions, {nonzero} denied a threatening pass",
                 fontsize=11.5, ha="center", color=ldv.TEXT_SUB)

        vmax = max(player["disruption_value"].max(), 1e-6)
        sizes = 260 + 3600 * (player["disruption_value"] / vmax)
        colors = [ldv.GOLD_RAMP(0.3 + 0.65 * (v / vmax)) if v > 0 else "#3a3f4a"
                 for v in player["disruption_value"]]
        edge_colors = [ldv.GOLD if v > 0 else "#5a6070" for v in player["disruption_value"]]

        pitch.scatter(player["x_own"], player["y_own"], ax=ax, s=sizes, color=colors,
                     edgecolors=edge_colors, linewidth=1.3, zorder=3, alpha=0.92)

        labelled = player.sort_values("disruption_value", ascending=False).head(6)
        for _, r in labelled.iterrows():
            if r["disruption_value"] <= 0:
                continue
            pitch.annotate(f"{r['disruption_value']*1000:.1f}", xy=(r["x_own"], r["y_own"]),
                          ax=ax, ha="center", va="center", fontsize=8, fontweight="bold",
                          color=ldv.BG, zorder=4)

        legend_elems = [
            Line2D([0], [0], marker="o", color="none", markerfacecolor=ldv.GOLD, markeredgecolor=ldv.GOLD,
                  markersize=14, label="Denied a threatening pass (value > 0)"),
            Line2D([0], [0], marker="o", color="none", markerfacecolor="#3a3f4a",
                  markeredgecolor="#5a6070", markersize=8, label="Denied a pass with no forward threat"),
        ]
        fig.legend(handles=legend_elems, loc="lower center", bbox_to_anchor=(0.5, 0.815), ncol=2,
                  frameon=False, fontsize=9.3, labelcolor=ldv.LEGEND_TEXT)

        caption = (f"Total disruption value this season: {total_value:.1f} (x1000)  ·  marker size "
                   f"= value of that single denial  ·  numbers show value x1000 for the top denials")
        fig.text(0.5, 0.075, caption, fontsize=9.3, ha="center", color=ldv.LEGEND_TEXT)
        fig.text(0.98, 0.02, "Marc Lamberts", fontsize=9.5, ha="right", color=ldv.TEXT_FOOT, style="italic")
        fig.text(0.02, 0.02, "Data via Opta | disruption value = P(pass completes) x xT it would "
                 "have added (Disruption/disruption_value_model.py)", fontsize=7.2, color=ldv.TEXT_FOOT)

        add_logo(fig)
        fig.savefig(out_path, dpi=200, facecolor=ldv.BG)
        plt.close(fig)
        print("Saved:", out_path)


if __name__ == "__main__":
    main()

"""
One team's OWN disrupting -- as valued pitch bins, with who did it listed
alongside. Eredivisie 2025/26.

Companion/mirror to team_value_bins_pitch.py, which shows which of
TEAM_NAME's own passes got broken up (TEAM_NAME as the attacking victim).
This script asks the opposite question: where does TEAM_NAME do its OWN
disrupting, and which of TEAM_NAME's own players is doing it -- filtering
all_eredivisie_disruption_values.csv on team_name (the disruptor) rather
than linked_pass_team (the disrupted attacker). x/y normalized to
"distance from TEAM_NAME's own goal" via that team's own attacking
direction per match/period, same convention as central_defending_leaderboard.py.

Usage: python3 team_own_defending_pitch.py "<team name>" [out.png]
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mplsoccer import VerticalPitch

import build_disruption_model as bdm
import league_disruption_visuals as ldv
from league_disruption_visuals import compute_attack_directions, add_logo

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
N_PLAYERS_LISTED = 10


def main():
    args = [a for a in sys.argv[1:] if a]
    if not args:
        raise SystemExit("Usage: python3 team_own_defending_pitch.py \"<team name>\" [out.png]")
    team_name = args[0]
    safe_name = team_name.replace(" ", "_")
    out_name = args[1] if len(args) > 1 else f"own_defending_bins_{safe_name}.png"

    values = pd.read_csv(os.path.join(ldv.CSV_DIR, "all_eredivisie_disruption_values.csv"))
    team_vals = values[values["team_name"] == team_name].copy()
    if team_vals.empty:
        raise SystemExit(f"No linked disruptions found for '{team_name}' -- check spelling "
                         f"against all_eredivisie_disruption_values.csv's team_name")

    print(f"Computing attacking direction per period so every zone is relative to "
          f"{team_name}'s own goal...")
    directions = compute_attack_directions()
    dir_vals = np.array([
        directions.get((mf, cid, p), 1)
        for mf, cid, p in zip(team_vals["match_file"], team_vals["contestant_id"], team_vals["period_id"])
    ])
    team_vals["x_own"] = np.where(dir_vals == 1, team_vals["x"], bdm.GOAL_X - team_vals["x"])
    team_vals["y_own"] = np.where(dir_vals == 1, team_vals["y"], 68.0 - team_vals["y"])
    team_vals["value_x1000"] = team_vals["disruption_value"] * 1000

    for theme in ("dark", "light"):
        ldv.set_theme(theme)
        out_path = os.path.join(ldv.visual_dir(theme), out_name)

        fig = plt.figure(figsize=(16, 10.5))
        fig.patch.set_facecolor(ldv.BG)

        pitch = VerticalPitch(pitch_type="uefa", pitch_color=ldv.BG, line_color=ldv.PITCH_LINE,
                              linewidth=1.1, half=False, line_zorder=2)
        ax = fig.add_axes([0.03, 0.08, 0.62, 0.76])
        pitch.draw(ax=ax)

        stats = pitch.bin_statistic(team_vals["x_own"], team_vals["y_own"],
                                    values=team_vals["value_x1000"], statistic="sum", bins=(9, 7))
        hm = pitch.heatmap(stats, ax=ax, cmap=ldv.GOLD_RAMP, edgecolors=ldv.BG, linewidth=0.4, zorder=1)
        cax = fig.add_axes([0.665, 0.24, 0.014, 0.44])
        cb = fig.colorbar(hm, cax=cax)
        cb.set_label("Value denied, x1000 (summed per zone)", color=ldv.TEXT_SUB, fontsize=8)
        cb.ax.yaxis.set_tick_params(color=ldv.TEXT_SUB, labelcolor=ldv.TEXT_SUB, labelsize=7.5)

        n = len(team_vals)
        total_val = team_vals["value_x1000"].sum()
        fig.text(0.36, 0.965, team_name, fontsize=27, fontweight="bold", ha="center", color=ldv.TEXT_MAIN)
        fig.text(0.36, 0.925,
                 f"Own Disrupting, Valued Bins  ·  Eredivisie 2025/26  ·  Season  ·  {n} actions  ·  "
                 f"{total_val:.1f} total value denied",
                 fontsize=11.5, ha="center", color=ldv.TEXT_SUB)

        panel = fig.add_axes([0.71, 0.08, 0.27, 0.76])
        panel.set_facecolor(ldv.BG)
        panel.axis("off")
        panel.set_title("Who Did The Denying", fontsize=15, fontweight="bold", color=ldv.TEXT_MAIN,
                        loc="left", pad=10)

        leaderboard = (team_vals.groupby("player_name")
                      .agg(n=("value_x1000", "size"), total=("value_x1000", "sum"))
                      .reset_index().sort_values("total", ascending=False).head(N_PLAYERS_LISTED))
        vmax = max(leaderboard["total"].max(), 1e-6)

        y0, dy = 0.95, 1.0 / (N_PLAYERS_LISTED + 1.5)
        for i, (_, r) in enumerate(leaderboard.iterrows()):
            y = y0 - i * dy
            bar_w = 0.32 * (r["total"] / vmax)
            color = ldv.GOLD_RAMP(0.3 + 0.6 * (r["total"] / vmax))
            panel.add_patch(plt.Rectangle((0.0, y - 0.014), bar_w, 0.020, transform=panel.transAxes,
                                          facecolor=color, edgecolor="none", zorder=2))
            panel.text(0.0, y + 0.018, r["player_name"], transform=panel.transAxes, fontsize=9.5,
                      color=ldv.TEXT_MAIN, fontweight="bold", va="bottom")
            panel.text(0.98, y - 0.001, f"{r['n']}x  ·  {r['total']:.2f}", transform=panel.transAxes,
                      fontsize=8.6, color=ldv.LEGEND_TEXT, va="center", ha="right")

        fig.text(0.98, 0.02, "Marc Lamberts", fontsize=9.5, ha="right", color=ldv.TEXT_FOOT, style="italic")
        fig.text(0.02, 0.02, "Data via Opta | cell = sum of disruption_value in that zone "
                 "(P(pass completes) x xT denied); pitch runs bottom = own goal, top = opponent goal",
                 fontsize=7.3, color=ldv.TEXT_FOOT)

        add_logo(fig, width=0.08)
        fig.savefig(out_path, dpi=200, facecolor=ldv.BG)
        plt.close(fig)
        print("Saved:", out_path)


if __name__ == "__main__":
    main()

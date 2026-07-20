"""
Small multiples: where every team disrupts passes. Eredivisie 2025/26.

One mini VerticalPitch per team (18 panels), each showing that team's linked
defensive actions this season. x/y normalized per team per period to
"distance from own goal" (same convention as league_disruption_visuals.py's
heatmap) so a season spread across matches/halves where the team attacked
in opposite raw directions lands on one consistent orientation, and panels
are genuinely comparable to each other.

Usage: python3 team_pitch_grid.py [out.png]
"""
import glob
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
N_COLS, N_ROWS = 6, 3


def main():
    out_name = sys.argv[1] if len(sys.argv) > 1 else "team_pitch_grid.png"

    disr = pd.read_csv(os.path.join(ldv.CSV_DIR, "all_eredivisie_disruption_models.csv"))
    linked = disr[disr["linked"] == True].copy()  # noqa: E712

    print("Computing per-team attacking direction per period...")
    directions = compute_attack_directions()
    dir_vals = np.array([
        directions.get((mf, cid, p), 1)
        for mf, cid, p in zip(linked["match_file"], linked["contestant_id"], linked["period_id"])
    ])
    linked["x_own"] = np.where(dir_vals == 1, linked["x"], bdm.GOAL_X - linked["x"])
    linked["y_own"] = np.where(dir_vals == 1, linked["y"], 68.0 - linked["y"])

    team_summary = pd.read_csv(os.path.join(ldv.CSV_DIR, "disruption_value_team_summary.csv"))
    team_order = team_summary.sort_values("total_disruption_value", ascending=False)["team_name"].tolist()

    for theme in ("dark", "light"):
        ldv.set_theme(theme)
        out_path = os.path.join(ldv.visual_dir(theme), out_name)

        fig = plt.figure(figsize=(N_COLS * 3.6, N_ROWS * 4.6))
        fig.patch.set_facecolor(ldv.BG)
        fig.text(0.5, 0.975, "Where Every Team Disrupts Passes", fontsize=25, fontweight="bold",
                 ha="center", color=ldv.TEXT_MAIN)
        fig.text(0.5, 0.952,
                 "Eredivisie 2025/26  ·  Season  ·  every linked defensive action, normalized to "
                 "\"distance from own goal\"  ·  ranked by total disruption value",
                 fontsize=11.5, ha="center", color=ldv.TEXT_SUB)

        grid_top, grid_bottom = 0.90, 0.05
        grid_left, grid_right = 0.02, 0.98
        cell_w = (grid_right - grid_left) / N_COLS
        cell_h = (grid_top - grid_bottom) / N_ROWS

        for i, team in enumerate(team_order):
            row, col = divmod(i, N_COLS)
            left = grid_left + col * cell_w
            bottom = grid_top - (row + 1) * cell_h
            ax = fig.add_axes([left + cell_w * 0.06, bottom + cell_h * 0.03,
                               cell_w * 0.88, cell_h * 0.82])

            pitch = VerticalPitch(pitch_type="uefa", pitch_color=ldv.BG, line_color=ldv.PITCH_LINE,
                                  linewidth=0.8, half=False)
            pitch.draw(ax=ax)

            sub = linked[linked["team_name"] == team]
            pitch.scatter(sub["x_own"], sub["y_own"], ax=ax, s=22, color=ldv.GOLD, alpha=0.6,
                         edgecolors="none", zorder=3)

            info = team_summary[team_summary["team_name"] == team].iloc[0]
            ax.set_title(f"{team}", fontsize=10.5, fontweight="bold", color=ldv.TEXT_MAIN, pad=4)
            fig.text(left + cell_w / 2, bottom + cell_h * 0.02,
                     f"{int(info['actions_linked'])} actions  ·  {info['total_disruption_value_x1000']:.1f} value",
                     fontsize=7.8, ha="center", color=ldv.TEXT_SUB)

        fig.text(0.98, 0.012, "Marc Lamberts", fontsize=9.5, ha="right", color=ldv.TEXT_FOOT, style="italic")
        fig.text(0.02, 0.012, "Data via Opta | pitch runs bottom = own goal, top = opponent goal",
                 fontsize=8, color=ldv.TEXT_FOOT)

        add_logo(fig, width=0.05)
        fig.savefig(out_path, dpi=200, facecolor=ldv.BG)
        plt.close(fig)
        print("Saved:", out_path)


if __name__ == "__main__":
    main()

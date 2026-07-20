"""
Small multiples: where each action type disrupts passes. Eredivisie 2025/26.

Companion to action_type_value_chart.py -- that chart ranks the six linked
action types (Tackle/Interception/Clearance/Aerial Duel/Ball Recovery/
Dispossessed) by mean value per action; this shows WHERE each one happens,
normalized to "distance from the disrupting team's own goal" per team per
period so the six panels share one consistent orientation.

Usage: python3 action_type_pitch_grid.py [out.png]
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
ACTION_ORDER = ["Clearance", "Dispossessed", "Ball Recovery", "Aerial Duel", "Interception", "Tackle"]


def main():
    out_name = sys.argv[1] if len(sys.argv) > 1 else "action_type_pitch_grid.png"

    values = pd.read_csv(os.path.join(ldv.CSV_DIR, "all_eredivisie_disruption_values.csv"))

    print("Computing per-team attacking direction per period...")
    directions = compute_attack_directions()
    dir_vals = np.array([
        directions.get((mf, cid, p), 1)
        for mf, cid, p in zip(values["match_file"], values["contestant_id"], values["period_id"])
    ])
    values["x_own"] = np.where(dir_vals == 1, values["x"], bdm.GOAL_X - values["x"])
    values["y_own"] = np.where(dir_vals == 1, values["y"], 68.0 - values["y"])

    for theme in ("dark", "light"):
        ldv.set_theme(theme)
        out_path = os.path.join(ldv.visual_dir(theme), out_name)

        fig = plt.figure(figsize=(6 * 3.4, 5.6))
        fig.patch.set_facecolor(ldv.BG)
        fig.text(0.5, 0.955, "Where Each Action Type Disrupts Passes", fontsize=23, fontweight="bold",
                 ha="center", color=ldv.TEXT_MAIN)
        fig.text(0.5, 0.905,
                 "Eredivisie 2025/26  ·  Season  ·  every linked defensive action  ·  colour = mean "
                 "disruption value for that action type (see action_type_value_chart.py)",
                 fontsize=10.5, ha="center", color=ldv.TEXT_SUB)

        n = len(ACTION_ORDER)
        cell_w = 1.0 / n
        for i, action in enumerate(ACTION_ORDER):
            left = i * cell_w
            ax = fig.add_axes([left + cell_w * 0.08, 0.10, cell_w * 0.84, 0.72])
            pitch = VerticalPitch(pitch_type="uefa", pitch_color=ldv.BG, line_color=ldv.PITCH_LINE,
                                  linewidth=0.8, half=False)
            pitch.draw(ax=ax)

            sub = values[values["action_type"] == action]
            mean_val = sub["disruption_value"].mean() * 1000
            color = ldv.GOLD_RAMP(0.3 + 0.6 * min(mean_val / 0.6, 1.0))
            pitch.scatter(sub["x_own"], sub["y_own"], ax=ax, s=16, color=color, alpha=0.55,
                         edgecolors="none", zorder=3)

            ax.set_title(action, fontsize=12, fontweight="bold", color=ldv.TEXT_MAIN, pad=6)
            fig.text(left + cell_w / 2, 0.055, f"n={len(sub)}  ·  {mean_val:.2f} mean value",
                     fontsize=8.3, ha="center", color=ldv.TEXT_SUB)

        fig.text(0.98, 0.012, "Marc Lamberts", fontsize=9.5, ha="right", color=ldv.TEXT_FOOT, style="italic")
        fig.text(0.02, 0.012, "Data via Opta | pitch runs bottom = own goal, top = opponent goal",
                 fontsize=8, color=ldv.TEXT_FOOT)

        add_logo(fig, width=0.045)
        fig.savefig(out_path, dpi=200, facecolor=ldv.BG)
        plt.close(fig)
        print("Saved:", out_path)


if __name__ == "__main__":
    main()

"""
League-wide interception transition-threat visuals - Eredivisie 2025/26
===========================================================================
Two charts built on build_interception_transition_model.py's output:
  1. A pitch heatmap of WHERE interceptions that kick off the most
     transition threat happen (value-weighted bins, not just counts).
  2. A leaderboard of the top individual interceptors by total transition
     threat created.

Reuses Disruption/league_disruption_visuals.py's theming/plotting helpers
(same palette, same add_logo/make_leaderboard machinery) so this reads as
one visual family with the rest of the repo rather than a one-off style.

Usage: python3 interception_transition_visuals.py
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mplsoccer import VerticalPitch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Disruption"))
import build_disruption_model as bdm  # noqa: E402
import league_disruption_visuals as ldv  # noqa: E402
from league_disruption_visuals import compute_attack_directions, add_logo  # noqa: E402

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_DIR = os.path.join(OUT_DIR, "CSV")


def visual_dir(theme):
    d = os.path.join(OUT_DIR, "Visual - Dark" if theme == "dark" else "Visual - Light")
    os.makedirs(d, exist_ok=True)
    return d


def make_value_heatmap(interceptions, out_path):
    print("Computing per-team attacking direction per period to normalize "
          "interception x/y onto a common 'distance from own goal' axis...")
    directions = compute_attack_directions()
    dir_vals = np.array([
        directions.get((mf, cid, p), 1)
        for mf, cid, p in zip(interceptions["match_file"], interceptions["contestant_id"],
                              interceptions["period_id"])
    ])
    interceptions = interceptions.copy()
    interceptions["x_own"] = np.where(dir_vals == 1, interceptions["x"], bdm.GOAL_X - interceptions["x"])
    interceptions["y_own"] = np.where(dir_vals == 1, interceptions["y"], 68.0 - interceptions["y"])
    interceptions["value_x1000"] = interceptions["transition_threat"] * 1000

    fig = plt.figure(figsize=(11, 13.7))
    fig.patch.set_facecolor(ldv.BG)
    pitch = VerticalPitch(pitch_type="uefa", pitch_color=ldv.BG, line_color=ldv.PITCH_LINE,
                          linewidth=1.1, half=False, line_zorder=2)
    ax = fig.add_axes([0.04, 0.09, 0.92, 0.78])
    pitch.draw(ax=ax)

    stats = pitch.bin_statistic(interceptions["x_own"], interceptions["y_own"],
                                values=interceptions["value_x1000"], statistic="sum", bins=(16, 12))
    hm = pitch.heatmap(stats, ax=ax, cmap=ldv.GOLD_RAMP, edgecolors=ldv.BG, linewidth=0.15, zorder=1)
    cax = fig.add_axes([0.90, 0.24, 0.016, 0.45])
    cb = fig.colorbar(hm, cax=cax)
    cb.set_label("Transition threat, x1000 (summed per zone)", color=ldv.TEXT_SUB, fontsize=8.5)
    cb.ax.yaxis.set_tick_params(color=ldv.TEXT_SUB, labelcolor=ldv.TEXT_SUB, labelsize=7.5)

    fig.text(0.5, 0.965, "Where Interceptions Spark Transitions", fontsize=23, fontweight="bold",
             ha="center", color=ldv.TEXT_MAIN)
    fig.text(0.5, 0.935, f"Eredivisie 2025/26  ·  Season  ·  {len(interceptions)} interceptions",
             fontsize=11.5, ha="center", color=ldv.TEXT_SUB)
    fig.text(0.5, 0.915, "cell = total xT the intercepting team generated within "
             "10s of winning the ball back there", fontsize=11.5, ha="center", color=ldv.TEXT_SUB)

    own_third = (interceptions["x_own"] < 35).mean()
    mid_third = ((interceptions["x_own"] >= 35) & (interceptions["x_own"] < 70)).mean()
    att_third = (interceptions["x_own"] >= 70).mean()
    fig.text(0.5, 0.058,
             f"{own_third:.0%} of interceptions in the defensive third, {mid_third:.0%} in "
             f"midfield, {att_third:.0%} in the attacking third",
             fontsize=9.5, ha="center", color=ldv.LEGEND_TEXT)
    fig.text(0.5, 0.041, "x normalized per team per period to \"distance from own goal\" "
             "(bottom = own goal, top = opponent goal)", fontsize=9.5, ha="center", color=ldv.LEGEND_TEXT)
    fig.text(0.98, 0.006, "Marc Lamberts", fontsize=9.5, ha="right", color=ldv.TEXT_FOOT, style="italic")
    fig.text(0.02, 0.006, "Data via Opta | transition threat = positive xT added by the intercepting "
             "team's own actions in the 10s after the interception, before losing the ball back",
             fontsize=7.0, color=ldv.TEXT_FOOT)

    add_logo(fig)
    fig.savefig(out_path, dpi=200, facecolor=ldv.BG)
    plt.close(fig)
    print("Saved:", out_path)


def main():
    interceptions = pd.read_csv(os.path.join(CSV_DIR, "all_eredivisie_interception_transitions.csv"))
    player_summary = pd.read_csv(os.path.join(CSV_DIR, "interception_transition_player_summary.csv"))

    for theme in ("dark", "light"):
        ldv.set_theme(theme)
        out_dir = visual_dir(theme)
        make_value_heatmap(interceptions, os.path.join(out_dir, "league_interception_transition_heatmap.png"))
        ldv.make_leaderboard(
            player_summary, os.path.join(out_dir, "top_interceptors_by_transition_threat_leaderboard.png"),
            value_col="total_transition_threat_x1000",
            count_col="interceptions",
            title="Top Interceptors by Transition Threat",
            subtitle="Eredivisie 2025/26  ·  Season  ·  transition threat = xT the intercepting "
                     "team created in the 10s after the regain, before losing the ball back  ·  "
                     "x1000 for readability",
            footer="Data via Opta | rewards interceptions that spark a real counter, not just "
                   "winning the ball back",
            value_fmt="{:.1f}")


if __name__ == "__main__":
    main()

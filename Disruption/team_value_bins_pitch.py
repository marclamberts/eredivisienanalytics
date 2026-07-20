"""
One team's disrupted passes as VALUED bins, with who did it listed
alongside. Eredivisie 2025/26.

Same question as team_disrupted_passes_pitch.py -- which of TEAM_NAME's
passes got broken up, and by whom -- but the pitch itself is a binned
heatmap of disruption_value (sum per zone, from disruption_value_model.py)
instead of individual pass lines/markers: cell color = how much attacking
threat was actually taken away in that zone, not just how many times a
pass was touched there. x/y (point of disruption) normalized to "distance
from TEAM_NAME's own goal" per match/period so every zone reads the same
way regardless of which end they were shooting at that half.

Usage: python3 team_value_bins_pitch.py "<team name>" [out.png]
"""
import glob
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mplsoccer import VerticalPitch

import build_disruption_model as bdm
from league_disruption_visuals import compute_attack_directions, GOLD_RAMP, BG, PITCH_LINE, TEXT_SUB, TEXT_FOOT, add_logo

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
N_PLAYERS_LISTED = 10


def main():
    args = [a for a in sys.argv[1:] if a]
    if not args:
        raise SystemExit("Usage: python3 team_value_bins_pitch.py \"<team name>\" [out.png]")
    team_name = args[0]
    safe_name = team_name.replace(" ", "_")
    out_path = args[1] if len(args) > 1 else os.path.join(
        OUT_DIR, f"value_bins_{safe_name}.png")

    values = pd.read_csv(os.path.join(OUT_DIR, "all_eredivisie_disruption_values.csv"))
    team_vals = values[values["linked_pass_team"] == team_name].copy()
    if team_vals.empty:
        raise SystemExit(f"No disrupted passes found for '{team_name}' -- check spelling "
                         f"against all_eredivisie_disruption_values.csv's linked_pass_team")

    files = sorted(glob.glob(os.path.join(bdm.DATA_DIR, "*.json")))
    cid_to_team = bdm.build_global_cid_to_team(files)
    team_to_cid = {t: c for c, t in cid_to_team.items()}
    team_cid = team_to_cid.get(team_name)

    print("Computing attacking direction per period so every zone is relative to "
          f"{team_name}'s own goal...")
    directions = compute_attack_directions()
    dir_vals = np.array([
        directions.get((mf, team_cid, p), 1)
        for mf, p in zip(team_vals["match_file"], team_vals["period_id"])
    ])
    team_vals["x_own"] = np.where(dir_vals == 1, team_vals["x"], bdm.GOAL_X - team_vals["x"])
    team_vals["y_own"] = np.where(dir_vals == 1, team_vals["y"], 68.0 - team_vals["y"])
    team_vals["value_x1000"] = team_vals["disruption_value"] * 1000

    fig = plt.figure(figsize=(16, 10.5))
    fig.patch.set_facecolor(BG)

    # line_zorder must be above the heatmap's zorder (1): zero-value cells
    # render as exactly the background colour (the ramp starts at BG), and
    # with a grid this coarse/sparse those opaque cells otherwise paint
    # straight over the pitch markings underneath instead of just blending
    # into open grass (mplsoccer's own heatmap examples set this for the
    # same reason).
    pitch = VerticalPitch(pitch_type="uefa", pitch_color=BG, line_color=PITCH_LINE,
                          linewidth=1.1, half=False, line_zorder=2)
    ax = fig.add_axes([0.03, 0.08, 0.62, 0.76])
    pitch.draw(ax=ax)

    stats = pitch.bin_statistic(team_vals["x_own"], team_vals["y_own"],
                                values=team_vals["value_x1000"], statistic="sum", bins=(9, 7))
    hm = pitch.heatmap(stats, ax=ax, cmap=GOLD_RAMP, edgecolors=BG, linewidth=0.4, zorder=1)
    cax = fig.add_axes([0.665, 0.24, 0.014, 0.44])
    cb = fig.colorbar(hm, cax=cax)
    cb.set_label("Value denied, x1000 (summed per zone)", color=TEXT_SUB, fontsize=8)
    cb.ax.yaxis.set_tick_params(color=TEXT_SUB, labelcolor=TEXT_SUB, labelsize=7.5)

    n = len(team_vals)
    total_val = team_vals["value_x1000"].sum()
    fig.text(0.36, 0.965, team_name, fontsize=27, fontweight="bold", ha="center", color="white")
    fig.text(0.36, 0.925,
             f"Disrupted Passes, Valued Bins  ·  Eredivisie 2025/26  ·  Season  ·  {n} passes  ·  "
             f"{total_val:.1f} total value denied",
             fontsize=11.5, ha="center", color=TEXT_SUB)

    # side panel: who did it
    panel = fig.add_axes([0.71, 0.08, 0.27, 0.76])
    panel.set_facecolor(BG)
    panel.axis("off")
    panel.set_title("Who Denied That Value", fontsize=15, fontweight="bold", color="white",
                    loc="left", pad=10)

    leaderboard = (team_vals.groupby(["player_name", "team_name"])
                  .agg(n=("value_x1000", "size"), total=("value_x1000", "sum"))
                  .reset_index().sort_values("total", ascending=False).head(N_PLAYERS_LISTED))
    vmax = max(leaderboard["total"].max(), 1e-6)

    y0, dy = 0.95, 1.0 / (N_PLAYERS_LISTED + 1.5)
    for i, (_, r) in enumerate(leaderboard.iterrows()):
        y = y0 - i * dy
        bar_w = 0.32 * (r["total"] / vmax)
        color = GOLD_RAMP(0.3 + 0.6 * (r["total"] / vmax))
        panel.add_patch(plt.Rectangle((0.0, y - 0.014), bar_w, 0.020, transform=panel.transAxes,
                                      facecolor=color, edgecolor="none", zorder=2))
        panel.text(0.0, y + 0.018, r["player_name"], transform=panel.transAxes, fontsize=9.5,
                  color="white", fontweight="bold", va="bottom")
        panel.text(0.0, y - 0.020, r["team_name"], transform=panel.transAxes, fontsize=7.3,
                  color=TEXT_SUB, va="top")
        panel.text(0.98, y - 0.001, f"{r['n']}x  ·  {r['total']:.2f}", transform=panel.transAxes,
                  fontsize=8.6, color="#c7ccd4", va="center", ha="right")

    fig.text(0.98, 0.02, "Marc Lamberts", fontsize=9.5, ha="right", color=TEXT_FOOT, style="italic")
    fig.text(0.02, 0.02, "Data via Opta | cell = sum of disruption_value in that zone "
             "(P(pass completes) x xT denied); pitch runs bottom = own goal, top = opponent goal",
             fontsize=7.3, color=TEXT_FOOT)

    add_logo(fig, width=0.08)
    fig.savefig(out_path, dpi=200, facecolor=BG)
    plt.close(fig)
    print("Saved:", out_path)


if __name__ == "__main__":
    main()

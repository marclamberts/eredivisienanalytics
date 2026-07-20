"""
One team's disrupted passes, on a pitch, with who did it listed alongside.
Eredivisie 2025/26.

Every pass by TEAM_NAME that was broken up by a specific opponent defensive
action this season (from all_eredivisie_disruption_models.csv), drawn on a
horizontal pitch: gold line = the pass, X = where an opponent won it. Next
to the pitch, a ranked list of exactly which opposing players did it, with
counts and total value denied -- so "which players have done that" is
answered right where you're looking, not in a separate table.

Usage: python3 team_disrupted_passes_pitch.py "<team name>" [out.png]
"""
import os
import sys

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
from mplsoccer import Pitch

from league_disruption_visuals import GOLD, GOLD_RAMP, BG, PITCH_LINE, TEXT_SUB, TEXT_FOOT, add_logo

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
N_PLAYERS_LISTED = 10


def main():
    args = [a for a in sys.argv[1:] if a]
    if not args:
        raise SystemExit("Usage: python3 team_disrupted_passes_pitch.py \"<team name>\" [out.png]")
    team_name = args[0]
    safe_name = team_name.replace(" ", "_")
    out_path = args[1] if len(args) > 1 else os.path.join(
        OUT_DIR, f"disrupted_passes_{safe_name}.png")

    disr = pd.read_csv(os.path.join(OUT_DIR, "all_eredivisie_disruption_models.csv"))
    team_disr = disr[(disr["linked"] == True) &  # noqa: E712
                     (disr["linked_pass_team"] == team_name)].copy()
    if team_disr.empty:
        raise SystemExit(f"No disrupted passes found for '{team_name}' -- check spelling "
                         f"against all_eredivisie_disruption_models.csv's linked_pass_team")

    fig = plt.figure(figsize=(16, 10.5))
    fig.patch.set_facecolor(BG)

    pitch = Pitch(pitch_type="uefa", pitch_color=BG, line_color=PITCH_LINE,
                 linewidth=1.2, half=False)
    ax = fig.add_axes([0.03, 0.10, 0.66, 0.74])
    pitch.draw(ax=ax)

    for _, r in team_disr.iterrows():
        pitch.lines(r["linked_pass_start_x"], r["linked_pass_start_y"],
                   r["linked_pass_end_x"], r["linked_pass_end_y"], ax=ax,
                   color=GOLD, lw=1.6, alpha=0.4, zorder=2, comet=False)
        pitch.scatter(r["x"], r["y"], ax=ax, s=70, marker="X", color=GOLD,
                     edgecolors=BG, linewidth=0.8, alpha=0.85, zorder=3)

    n = len(team_disr)
    n_disruptors = team_disr["player_name"].nunique()
    n_teams = team_disr["team_name"].nunique()
    fig.text(0.5, 0.965, team_name, fontsize=27, fontweight="bold", ha="center", color="white")
    fig.text(0.5, 0.925,
             f"Passes Disrupted by Opponents  ·  Eredivisie 2025/26  ·  Season  ·  "
             f"{n} passes broken up by {n_disruptors} different players across {n_teams} teams",
             fontsize=12, ha="center", color=TEXT_SUB)

    legend_elems = [
        Line2D([0], [0], color=GOLD, linewidth=2, alpha=0.6, label="Disrupted pass"),
        Line2D([0], [0], marker="X", color="none", markerfacecolor=GOLD, markeredgecolor=BG,
              markersize=10, label="Point of disruption"),
    ]
    ax.legend(handles=legend_elems, loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=2,
             frameon=False, fontsize=9.5, labelcolor="#c7ccd4")

    # side panel: who did it
    panel = fig.add_axes([0.71, 0.10, 0.27, 0.74])
    panel.set_facecolor(BG)
    panel.axis("off")
    panel.set_title("Who Disrupted Them", fontsize=15, fontweight="bold", color="white",
                    loc="left", pad=10)

    leaderboard = (team_disr.groupby(["player_name", "team_name"])
                  .agg(n=("disruption_score", "size"), total=("disruption_score", "sum"))
                  .reset_index().sort_values("total", ascending=False).head(N_PLAYERS_LISTED))
    vmax = leaderboard["total"].max()

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
        panel.text(0.98, y - 0.001, f"{r['n']}x  ·  {r['total']:.1f}", transform=panel.transAxes,
                  fontsize=8.6, color="#c7ccd4", va="center", ha="right")

    fig.text(0.98, 0.02, "Marc Lamberts", fontsize=9.5, ha="right", color=TEXT_FOOT, style="italic")
    fig.text(0.02, 0.02, "Data via Opta | disruption value = P(pass completes) x xT it would "
             "have added (see disruption_value_model.py); list shows count and total value "
             "denied", fontsize=7.6, color=TEXT_FOOT)

    add_logo(fig, width=0.08)
    fig.savefig(out_path, dpi=200, facecolor=BG)
    plt.close(fig)
    print("Saved:", out_path)


if __name__ == "__main__":
    main()

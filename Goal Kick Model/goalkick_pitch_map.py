"""
One team's goal kicks on a pitch - Eredivisie 2025/26
=========================================================
Every goal kick taken by TEAM_NAME this season
(Goal Kick Model/CSV/all_eredivisie_goalkicks.csv), drawn as a line from
where it was taken to where it was aimed: terracotta = that goal kick led
to a shot within 5 actions (build_goalkick_shot_model.py's definition),
muted = it didn't. x/y normalized per team per period to "distance from
TEAM_NAME's own goal" (own goal on the left) so every goal kick reads the
same way regardless of which end they defended that half -- same
convention as Disruption/team_value_bins_pitch.py.

In Marc Lamberts' Meridian house style (housestyle/ package at the repo
root).

Usage: python3 goalkick_pitch_map.py "<team name>" [out.png]
"""
import glob
import os
import sys

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from mplsoccer import Pitch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from housestyle import style, components  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Disruption"))
import build_disruption_model as bdm  # noqa: E402
from league_disruption_visuals import compute_attack_directions  # noqa: E402

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_DIR = os.path.join(OUT_DIR, "CSV")
SOURCE = "Opta event data, Eredivisie 2025/26"


def visual_dir(theme):
    d = os.path.join(OUT_DIR, "Visual - Dark" if theme == "dark" else "Visual - Light")
    os.makedirs(d, exist_ok=True)
    return d


def make_pitch_map(team_gk, team_name, mode, out_path):
    palette, _ = style.apply(mode)

    directions = compute_attack_directions()
    dir_vals = np.array([
        directions.get((mf, cid, p), 1)
        for mf, cid, p in zip(team_gk["match_file"], team_gk["contestant_id"], team_gk["period_id"])
    ])
    team_gk = team_gk.copy()
    team_gk["x_own"] = np.where(dir_vals == 1, team_gk["x"], bdm.GOAL_X - team_gk["x"])
    team_gk["y_own"] = np.where(dir_vals == 1, team_gk["y"], 68.0 - team_gk["y"])
    team_gk["end_x_own"] = np.where(dir_vals == 1, team_gk["end_x"], bdm.GOAL_X - team_gk["end_x"])
    team_gk["end_y_own"] = np.where(dir_vals == 1, team_gk["end_y"], 68.0 - team_gk["end_y"])

    fig = plt.figure(figsize=(11, 8.6))
    pitch = Pitch(pitch_type="uefa", pitch_color=palette["surface"], line_color=palette["axis"],
                 linewidth=1.1, half=False, line_zorder=2)
    ax = fig.add_axes([0.05, 0.10, 0.90, 0.62])
    pitch.draw(ax=ax)

    hit = team_gk[team_gk["reached_shot_within_5"]]
    miss = team_gk[~team_gk["reached_shot_within_5"]]

    for _, r in miss.iterrows():
        pitch.lines(r["x_own"], r["y_own"], r["end_x_own"], r["end_y_own"], ax=ax,
                   color=palette["axis"], lw=1.3, alpha=0.55, zorder=2, comet=False)
    for _, r in hit.iterrows():
        pitch.lines(r["x_own"], r["y_own"], r["end_x_own"], r["end_y_own"], ax=ax,
                   color=palette["accent"], lw=2.4, alpha=0.95, zorder=3, comet=False)
        pitch.scatter(r["end_x_own"], r["end_y_own"], ax=ax, s=60, marker="o",
                     color=palette["accent"], edgecolors=palette["surface"], linewidth=0.8, zorder=4)

    legend_elems = [
        Line2D([0], [0], color=palette["accent"], linewidth=2.4, label="Led to a shot within 5 actions"),
        Line2D([0], [0], color=palette["axis"], linewidth=1.6, alpha=0.7, label="No shot within 5 actions"),
    ]
    ax.legend(handles=legend_elems, loc="upper center", bbox_to_anchor=(0.5, -0.03), ncol=2,
             frameon=False, fontsize=9.5, labelcolor=palette["ink_secondary"])

    n = len(team_gk)
    n_hit = len(hit)
    components.header(
        fig, kicker="Goal Kicks",
        title=f"{team_name}: {n_hit} of {n} goal kicks reach a shot within 5 actions",
        dek=f"Eredivisie 2025/26  ·  Season  ·  {n_hit / n:.1%} shot rate  ·  "
            f"pitch runs own goal (left) to opponent goal (right)",
        palette=palette)
    components.footer(fig, source=SOURCE, palette=palette)

    fig.savefig(out_path, dpi=200, facecolor=palette["surface"])
    plt.close(fig)
    print("Saved:", out_path)


def main():
    args = [a for a in sys.argv[1:] if a]
    team_name = args[0] if args else "Heracles Almelo"
    safe_name = team_name.replace(" ", "_")
    out_name = args[1] if len(args) > 1 else f"goalkick_pitch_map_{safe_name}.png"

    goalkicks = pd.read_csv(os.path.join(CSV_DIR, "all_eredivisie_goalkicks.csv"))
    team_gk = goalkicks[goalkicks["team_name"] == team_name].dropna(subset=["end_x", "end_y"])
    if team_gk.empty:
        raise SystemExit(f"No goal kicks found for '{team_name}' -- check spelling against "
                         f"all_eredivisie_goalkicks.csv's team_name")

    for mode in ("light", "dark"):
        out_path = os.path.join(visual_dir(mode), out_name)
        make_pitch_map(team_gk, team_name, mode, out_path)


if __name__ == "__main__":
    main()

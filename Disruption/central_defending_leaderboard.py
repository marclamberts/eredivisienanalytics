"""
Which teams defend best in central areas? Eredivisie 2025/26.

"Central" = the middle third of the pitch width (y between 22.7m and
45.3m out of 68m), excluding both flanks -- the central channel, not the
wide areas. Ranks all 18 teams by disruption value denied
per match in that channel (see disruption_value_model.py), using each
team's own attacking-direction normalization so left/right and which end
they were shooting at doesn't matter.

Usage: python3 central_defending_leaderboard.py [out.png]
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import build_disruption_model as bdm
import league_disruption_visuals as ldv
from league_disruption_visuals import compute_attack_directions, add_logo

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
CENTRAL_LOW, CENTRAL_HIGH = 68 / 3, 2 * 68 / 3


def main():
    out_name = sys.argv[1] if len(sys.argv) > 1 else "central_defending_leaderboard.png"

    values = pd.read_csv(os.path.join(ldv.CSV_DIR, "all_eredivisie_disruption_values.csv"))
    directions = compute_attack_directions()
    dir_vals = np.array([
        directions.get((mf, cid, p), 1)
        for mf, cid, p in zip(values["match_file"], values["contestant_id"], values["period_id"])
    ])
    values["y_own"] = np.where(dir_vals == 1, values["y"], 68.0 - values["y"])

    central = values[(values["y_own"] >= CENTRAL_LOW) & (values["y_own"] <= CENTRAL_HIGH)].copy()
    print(f"{len(central)} / {len(values)} linked disruptions ({len(central)/len(values):.0%}) "
          f"fall in the central third of the pitch width")

    matches = pd.read_csv(os.path.join(ldv.CSV_DIR, "disruption_value_team_summary.csv"))[
        ["team_name", "matches"]]

    g = central.groupby("team_name").agg(
        actions_linked=("disruption_value", "size"),
        total_disruption_value=("disruption_value", "sum"),
    ).reset_index().merge(matches, on="team_name")
    g["total_disruption_value_x1000"] = g["total_disruption_value"] * 1000
    g["per_match_x1000"] = g["total_disruption_value_x1000"] / g["matches"]

    g = g.sort_values("per_match_x1000", ascending=True)

    for theme in ("dark", "light"):
        ldv.set_theme(theme)
        out_path = os.path.join(ldv.visual_dir(theme), out_name)

        fig, ax = plt.subplots(figsize=(12.5, 9.5))
        fig.patch.set_facecolor(ldv.BG)
        ax.set_facecolor(ldv.BG)

        vmax = g["per_match_x1000"].max()
        colors = [ldv.GOLD_RAMP(0.3 + 0.6 * (v / vmax)) for v in g["per_match_x1000"]]
        ax.barh(g["team_name"], g["per_match_x1000"], color=colors, height=0.62, zorder=3)

        for i, (val, n) in enumerate(zip(g["per_match_x1000"], g["actions_linked"])):
            ax.text(val + vmax * 0.012, i, f"{val:.2f}  ({int(n)} actions)",
                   va="center", fontsize=9.5, color=ldv.LEGEND_TEXT)

        ax.tick_params(axis="y", colors=ldv.TEXT_MAIN, labelsize=10.5, length=0)
        ax.tick_params(axis="x", colors=ldv.TEXT_SUB, labelsize=9)
        for spine in ("top", "right", "left"):
            ax.spines[spine].set_visible(False)
        ax.spines["bottom"].set_color(ldv.PITCH_LINE)
        ax.grid(axis="x", color=ldv.PITCH_LINE, linewidth=0.6, alpha=0.5, zorder=0)
        ax.set_xlim(0, vmax * 1.2)

        fig.text(0.5, 0.965, "Best Central Defending", fontsize=24, fontweight="bold",
                 ha="center", color=ldv.TEXT_MAIN)
        fig.text(0.5, 0.935,
                 "Eredivisie 2025/26  ·  Season  ·  disruption value denied per match, central "
                 "third of the pitch width only (flanks excluded)",
                 fontsize=10.5, ha="center", color=ldv.TEXT_SUB)

        fig.text(0.98, 0.012, "Marc Lamberts", fontsize=9.5, ha="right", color=ldv.TEXT_FOOT, style="italic")
        fig.text(0.02, 0.012, "Data via Opta | central = y-coordinate normalized per team's own "
                 "attacking direction, middle third of the 68m width  ·  x1000 for readability",
                 fontsize=7.3, color=ldv.TEXT_FOOT)

        fig.subplots_adjust(left=0.20, right=0.95, top=0.90, bottom=0.07)
        add_logo(fig)
        fig.savefig(out_path, dpi=200, facecolor=ldv.BG)
        plt.close(fig)
        print("Saved:", out_path)

    print(g.sort_values("per_match_x1000", ascending=False)[
        ["team_name", "matches", "actions_linked", "per_match_x1000"]].to_string(index=False))


if __name__ == "__main__":
    main()

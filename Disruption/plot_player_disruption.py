"""
Pass map + disruption overlay for one player - Eredivisie 2025/26
====================================================================
Reads the CSV written by player_pressure_difficulty.py (all of a player's
passes with baseline/pressure-aware difficulty) and all_eredivisie_
disruption_models.csv (which defensive actions broke up which of the
player's passes), and draws a single mplsoccer VerticalPitch:
  - every pass as an arrow, coloured green (completed) or red (incomplete),
    line width scaled by pressure-aware difficulty (thicker = harder pass)
  - passes that were disrupted (linked to a specific opponent defensive
    action) highlighted in gold with the disruptor's name/action annotated

Usage: python3 plot_player_disruption.py "<player name>" [out.png]
If player_difficulty_<player>.csv doesn't exist yet, run
player_pressure_difficulty.py "<player name>" first.
"""
import os
import sys

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mplsoccer import VerticalPitch

import league_disruption_visuals as ldv
from league_disruption_visuals import add_logo

OUT_DIR = os.path.dirname(os.path.abspath(__file__))


def main():
    args = [a for a in sys.argv[1:] if a]
    if not args:
        raise SystemExit("Usage: python3 plot_player_disruption.py \"<player name>\"")
    player_name = args[0]
    safe_name = player_name.replace(" ", "_").replace(".", "")

    pass_path = os.path.join(ldv.CSV_DIR, f"player_difficulty_{safe_name}.csv")
    if not os.path.exists(pass_path):
        raise SystemExit(f"{pass_path} not found -- run player_pressure_difficulty.py "
                          f"\"{player_name}\" first")
    passes = pd.read_csv(pass_path)

    disr = pd.read_csv(os.path.join(ldv.CSV_DIR, "all_eredivisie_disruption_models.csv"))
    disrupted_all = disr[disr["linked_pass_player"] == player_name].copy()
    disrupted_keys = set(zip(disrupted_all["match_file"], disrupted_all["linked_pass_event_id"]))
    passes["is_disrupted"] = [
        (mf, eid) in disrupted_keys for mf, eid in zip(passes["match_file"], passes["event_id"])
    ]

    for theme in ("dark", "light"):
        ldv.set_theme(theme)
        out_path = os.path.join(ldv.visual_dir(theme), f"pitch_disruption_{safe_name}.png")
        disrupted = disrupted_all

        fig = plt.figure(figsize=(11, 13.7))
        fig.patch.set_facecolor(ldv.BG)
        pitch = VerticalPitch(pitch_type="uefa", pitch_color=ldv.BG, line_color=ldv.PITCH_LINE,
                              linewidth=1.1, half=False)
        ax = fig.add_axes([0.04, 0.08, 0.92, 0.80])
        pitch.draw(ax=ax)

        fig.text(0.5, 0.965, player_name, fontsize=26, fontweight="bold", ha="center", color=ldv.TEXT_MAIN)
        n = len(passes)
        completion = passes["outcome"].mean()
        fig.text(0.5, 0.935,
                 f"Pass Map & Disruption  ·  Eredivisie 2025/26  ·  Season  ·  "
                 f"{n} passes, {completion:.0%} completed",
                 fontsize=12, ha="center", color=ldv.TEXT_SUB)

        ordinary = passes[~passes["is_disrupted"]]
        for _, r in ordinary.iterrows():
            color = ldv.GREEN if r["outcome"] == 1 else ldv.RED
            lw = 0.4 + 1.6 * r["pressure_difficulty"]
            pitch.lines(r["start_x"], r["start_y"], r["end_x"], r["end_y"], ax=ax, color=color,
                       lw=lw, alpha=0.22, zorder=2, comet=False)
            pitch.scatter(r["start_x"], r["start_y"], ax=ax, s=7, color=color, alpha=0.4,
                         zorder=3, linewidths=0)

        disrupted = disrupted.sort_values("linked_pass_start_y").reset_index(drop=True)
        note_lines = []
        for i, r in disrupted.iterrows():
            n = i + 1
            sx, sy = r["linked_pass_start_x"], r["linked_pass_start_y"]
            ex, ey = r["linked_pass_end_x"], r["linked_pass_end_y"]
            pitch.lines(sx, sy, ex, ey, ax=ax, color=ldv.GOLD, lw=3.2, alpha=0.95, zorder=4, comet=False)
            pitch.scatter(sx, sy, ax=ax, s=60, color=ldv.GOLD, edgecolors=ldv.TEXT_MAIN, linewidth=1.0, zorder=5)
            pitch.scatter(r["x"], r["y"], ax=ax, s=280, marker="o", color=ldv.BG,
                         edgecolors=ldv.GOLD, linewidth=2.0, zorder=5)
            pitch.annotate(str(n), xy=(r["x"], r["y"]), ax=ax, ha="center", va="center",
                          fontsize=11, fontweight="bold", color=ldv.GOLD, zorder=6)
            note_lines.append(f"{n}.  {r['player_name']}  ({r['action_type']}, {r['team_name']})  "
                              f"—  denied a {r['disruption_score']:.0%} pass")

        legend_elems = [
            Line2D([0], [0], color=ldv.GREEN, linewidth=2.5, label=f"Completed pass ({(passes['outcome']==1).sum()})"),
            Line2D([0], [0], color=ldv.RED, linewidth=2.5, label=f"Incomplete pass ({(passes['outcome']==0).sum()})"),
            Line2D([0], [0], color=ldv.GOLD, linewidth=3, label=f"Disrupted by opponent ({len(disrupted)})"),
        ]
        fig.legend(handles=legend_elems, loc="lower center", bbox_to_anchor=(0.5, 0.882),
                  ncol=3, frameon=False, fontsize=9, labelcolor=ldv.LEGEND_TEXT)

        caption = ("Line width = pressure-aware pass difficulty (thicker = harder). "
                   "Numbered gold markers = where an opponent action broke up this "
                   "player's pass; % = the model's predicted completion probability "
                   "for that pass just before it was denied.")
        fig.text(0.5, 0.062, caption, fontsize=8.6, ha="center", color=ldv.LEGEND_TEXT, wrap=True)
        for j, line in enumerate(note_lines):
            fig.text(0.5, 0.046 - j * 0.011, line, fontsize=8.3, ha="center", color=ldv.GOLD)
        fig.text(0.98, 0.006, "Marc Lamberts", fontsize=9.5, ha="right", color=ldv.TEXT_FOOT, style="italic")
        fig.text(0.02, 0.006, "Data via Opta | Eredivisie 2025/26 · pass completion model + disruption linking "
                 "(see Disruption/build_disruption_model.py)", fontsize=7.0, color=ldv.TEXT_FOOT)

        add_logo(fig)
        fig.savefig(out_path, dpi=200, facecolor=ldv.BG)
        plt.close(fig)
        print("Saved:", out_path)


if __name__ == "__main__":
    main()

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

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

BG = "#0d1117"
PITCH_LINE = "#2c3a4d"
TEXT_MAIN = "#e6e9ee"
TEXT_SUB = "#9aa4b2"
GREEN = "#4a9e5c"
RED = "#e0765c"
GOLD = "#ffc247"


def add_logo(fig, width=0.11, margin=0.014):
    import matplotlib.image as mpimg
    logo_path = "/Users/marclamberts/Downloads/Waltzing Analytics Logo Type.png"
    try:
        img = mpimg.imread(logo_path)
    except FileNotFoundError:
        return
    fig_w, fig_h = fig.get_size_inches()
    img_h, img_w = img.shape[0], img.shape[1]
    width_in = width * fig_w
    height_in = width_in * (img_h / img_w)
    height = height_in / fig_h
    left = 1 - margin - width
    bottom = 1 - margin - height
    logo_ax = fig.add_axes([left, bottom, width, height], zorder=10)
    logo_ax.patch.set_alpha(0)
    logo_ax.set_xlim(0, img_w)
    logo_ax.set_ylim(img_h, 0)
    logo_ax.imshow(img)
    logo_ax.axis("off")


def main():
    args = [a for a in sys.argv[1:] if a]
    if not args:
        raise SystemExit("Usage: python3 plot_player_disruption.py \"<player name>\" [out.png]")
    player_name = args[0]
    safe_name = player_name.replace(" ", "_").replace(".", "")

    pass_path = os.path.join(OUT_DIR, f"player_difficulty_{safe_name}.csv")
    if not os.path.exists(pass_path):
        raise SystemExit(f"{pass_path} not found -- run player_pressure_difficulty.py "
                          f"\"{player_name}\" first")
    passes = pd.read_csv(pass_path)

    disr = pd.read_csv(os.path.join(OUT_DIR, "all_eredivisie_disruption_models.csv"))
    disrupted = disr[disr["linked_pass_player"] == player_name].copy()
    disrupted_keys = set(zip(disrupted["match_file"], disrupted["linked_pass_event_id"]))
    passes["is_disrupted"] = [
        (mf, eid) in disrupted_keys for mf, eid in zip(passes["match_file"], passes["event_id"])
    ]

    out_path = args[1] if len(args) > 1 else os.path.join(OUT_DIR, f"pitch_disruption_{safe_name}.png")

    fig = plt.figure(figsize=(11, 13.7))
    fig.patch.set_facecolor(BG)
    pitch = VerticalPitch(pitch_type="uefa", pitch_color=BG, line_color=PITCH_LINE,
                          linewidth=1.1, half=False)
    ax = fig.add_axes([0.04, 0.08, 0.92, 0.80])
    pitch.draw(ax=ax)

    fig.text(0.5, 0.965, player_name, fontsize=26, fontweight="bold", ha="center", color="white")
    n = len(passes)
    completion = passes["outcome"].mean()
    fig.text(0.5, 0.935,
             f"Pass Map & Disruption  ·  Eredivisie 2025/26  ·  Season  ·  "
             f"{n} passes, {completion:.0%} completed",
             fontsize=12, ha="center", color=TEXT_SUB)

    ordinary = passes[~passes["is_disrupted"]]
    for _, r in ordinary.iterrows():
        color = GREEN if r["outcome"] == 1 else RED
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
        pitch.lines(sx, sy, ex, ey, ax=ax, color=GOLD, lw=3.2, alpha=0.95, zorder=4, comet=False)
        pitch.scatter(sx, sy, ax=ax, s=60, color=GOLD, edgecolors="white", linewidth=1.0, zorder=5)
        pitch.scatter(r["x"], r["y"], ax=ax, s=280, marker="o", color=BG,
                     edgecolors=GOLD, linewidth=2.0, zorder=5)
        pitch.annotate(str(n), xy=(r["x"], r["y"]), ax=ax, ha="center", va="center",
                      fontsize=11, fontweight="bold", color=GOLD, zorder=6)
        note_lines.append(f"{n}.  {r['player_name']}  ({r['action_type']}, {r['team_name']})  "
                          f"—  denied a {r['disruption_score']:.0%} pass")

    legend_elems = [
        Line2D([0], [0], color=GREEN, linewidth=2.5, label=f"Completed pass ({(passes['outcome']==1).sum()})"),
        Line2D([0], [0], color=RED, linewidth=2.5, label=f"Incomplete pass ({(passes['outcome']==0).sum()})"),
        Line2D([0], [0], color=GOLD, linewidth=3, label=f"Disrupted by opponent ({len(disrupted)})"),
    ]
    fig.legend(handles=legend_elems, loc="lower center", bbox_to_anchor=(0.5, 0.882),
              ncol=3, frameon=False, fontsize=9, labelcolor="#c7ccd4")

    caption = ("Line width = pressure-aware pass difficulty (thicker = harder). "
               "Numbered gold markers = where an opponent action broke up this "
               "player's pass; % = the model's predicted completion probability "
               "for that pass just before it was denied.")
    fig.text(0.5, 0.062, caption, fontsize=8.6, ha="center", color="#c7ccd4", wrap=True)
    for j, line in enumerate(note_lines):
        fig.text(0.5, 0.046 - j * 0.011, line, fontsize=8.3, ha="center", color=GOLD)
    fig.text(0.98, 0.006, "Marc Lamberts", fontsize=9.5, ha="right", color="#6b7684", style="italic")
    fig.text(0.02, 0.006, "Data via Opta | Eredivisie 2025/26 · pass completion model + disruption linking "
             "(see Disruption/build_disruption_model.py)", fontsize=7.0, color="#6b7684")

    add_logo(fig)
    fig.savefig(out_path, dpi=200, facecolor=BG)
    plt.close(fig)
    print("Saved:", out_path)


if __name__ == "__main__":
    main()

"""
Passes into the shot hot zone: the small patch inside the box
(shot_hotspot_analysis.py) that produces 44.9% of the league's shot xG from
just 16.7% of the box's area. Filters the already-scored open-play
box-entry attempts (eredivisie_box_entry_attempts_scored.csv, from
score_eredivisie_box_entries.py) down to only those targeting that patch,
and ranks players/teams on getting the ball into the single most
dangerous part of the box, not just the box in general.

Requires shot_hotspot_analysis.py and score_eredivisie_box_entries.py to
have been run first (for shot_hot_zone.json and the scored attempts CSV).

Usage: python3 passes_to_hot_zone.py [out_dir]
"""
import sys
import json
import re
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from mplsoccer import VerticalPitch

BG = "#0d1117"
GRID_COLOR = "#232a35"
TEXT_MAIN = "#e6e9ee"
TEXT_SUB = "#9aa4b2"
BLUE = "#2f8fd1"
AMBER = "#ffc247"
PITCH_LINE = "#2c3a4d"

LOGO_PATH = "/Users/marclamberts/Downloads/Waltzing Analytics Logo Type.png"
PREFIX_RE = re.compile(r"^(CSD|CD|CS|SD)\s+")


def clean_name(name):
    return PREFIX_RE.sub("", name)


def add_logo(fig, width=0.11, margin=0.014):
    import matplotlib.image as mpimg
    try:
        img = mpimg.imread(LOGO_PATH)
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


def make_leaderboard(g, zone, out_path, min_attempts=8):
    g = g[g["attempts"] >= min_attempts].sort_values("attempts", ascending=False).head(15)
    g = g.iloc[::-1]

    fig, ax = plt.subplots(figsize=(15, 11))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    labels = [f"{row.player}  ({clean_name(row.team)})" for row in g.itertuples()]
    ax.barh(labels, g["attempts"], color=BLUE, height=0.65, zorder=3, label="Attempts into hot zone")
    ax.scatter(g["completed"], labels, color="white", edgecolors=BG, s=90, zorder=4, label="Completed")
    for i, (att, comp) in enumerate(zip(g["attempts"], g["completed"])):
        label_x = max(att, comp) + g["attempts"].max() * 0.02
        ax.text(label_x, i, f"{att:.0f}", va="center", ha="left", color=TEXT_MAIN, fontsize=10.5, fontweight="bold")

    ax.set_xlim(0, g["attempts"].max() * 1.15)
    ax.set_xlabel("Open-play passes attempted into the shot hot zone (season total)",
                  fontsize=11, color=TEXT_MAIN, fontweight="bold", labelpad=10)
    ax.tick_params(axis="x", colors=TEXT_SUB, labelsize=10)
    ax.tick_params(axis="y", colors=TEXT_MAIN, labelsize=11)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(True, axis="x", color=GRID_COLOR, linewidth=0.6, alpha=0.6, zorder=0)
    ax.legend(loc="lower right", frameon=False, fontsize=9.5, labelcolor=TEXT_MAIN)

    fig.text(0.06, 0.965, "Passes Into the Shot Hot Zone", fontsize=24, fontweight="bold", color="white")
    fig.text(0.06, 0.938, f"x in [{zone['x_lo']:.0f}, {zone['x_hi']:.0f}], y in [{zone['y_lo']:.0f}, {zone['y_hi']:.0f}]"
             f"  ·  min {min_attempts} attempts  ·  Eredivisie 2025/26  ·  Season", fontsize=12, color=TEXT_SUB)
    fig.text(0.06, 0.012, "Data via Opta | hot zone = the patch of the box responsible for 44.9% of the "
             "league's shot xG from 16.7% of its area (see shot_hotspot_analysis.py) · open play only",
             fontsize=7.6, color="#6b7684")
    fig.text(0.98, 0.012, "Marc Lamberts", fontsize=9, ha="right", color="#6b7684", style="italic")
    fig.subplots_adjust(left=0.24, right=0.96, top=0.90, bottom=0.07)
    add_logo(fig)
    fig.savefig(out_path, dpi=200, facecolor=BG)
    plt.close(fig)
    print("Saved:", out_path)


def make_zone_map(df, zone, out_path):
    completed = df[df["outcome"] == 1]

    fig = plt.figure(figsize=(11, 10.6))
    fig.patch.set_facecolor(BG)
    fig.text(0.5, 0.955, "Completed Passes Into the Hot Zone", fontsize=24, fontweight="bold", ha="center", color="white")
    fig.text(0.5, 0.918, "Every open-play pass this season landing in the league's highest-danger patch  ·  "
             "Eredivisie 2025/26", fontsize=11.5, ha="center", color=TEXT_SUB)

    pitch_ax = fig.add_axes([0.04, 0.10, 0.92, 0.72])
    pitch = VerticalPitch(pitch_type="opta", pitch_color=BG, line_color=PITCH_LINE, linewidth=1.1, half=True)
    pitch.draw(ax=pitch_ax)

    pitch.polygon([[[zone["x_lo"], zone["y_lo"]], [zone["x_lo"], zone["y_hi"]],
                    [zone["x_hi"], zone["y_hi"]], [zone["x_hi"], zone["y_lo"]]]],
                  ax=pitch_ax, facecolor=AMBER, edgecolor=AMBER, alpha=0.12, linewidth=2.0, zorder=1)

    for r in completed.itertuples():
        pitch.lines(r.x, r.y, r.end_x, r.end_y, ax=pitch_ax, color=BLUE, lw=1.0, alpha=0.35, zorder=2, comet=False)
    pitch.scatter(completed["end_x"], completed["end_y"], ax=pitch_ax, s=16, color=AMBER, alpha=0.75, zorder=3, linewidths=0)

    caption = f"{len(df)} attempts this season  ·  {len(completed)} completed ({len(completed) / len(df) * 100:.0f}%)"
    fig.text(0.5, 0.045, caption, fontsize=11.5, ha="center", color="#c7ccd4")

    fig.text(0.98, 0.006, "Marc Lamberts", fontsize=9.5, ha="right", color="#6b7684", style="italic")
    fig.text(0.02, 0.006, "Data via Opta | Eredivisie 2025/26 event data · open-play passes only, "
             "excludes corners/free kicks/throw-ins", fontsize=7.3, color="#6b7684")

    add_logo(fig)
    fig.savefig(out_path, dpi=200, facecolor=BG)
    plt.close(fig)
    print("Saved:", out_path)


def main():
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent

    with open(out_dir / "shot_hot_zone.json") as f:
        zone = json.load(f)["zone"]

    attempts = pd.read_csv(out_dir / "eredivisie_box_entry_attempts_scored.csv")
    hot = attempts[(attempts["end_x"] >= zone["x_lo"]) & (attempts["end_x"] <= zone["x_hi"]) &
                   (attempts["end_y"] >= zone["y_lo"]) & (attempts["end_y"] <= zone["y_hi"])].copy()
    print(f"{len(hot)} of {len(attempts)} open-play box-entry attempts ({len(hot) / len(attempts) * 100:.1f}%) "
          f"targeted the hot zone.")
    print(f"Completion rate in zone: {hot['outcome'].mean():.3f} vs. {attempts['outcome'].mean():.3f} for the box overall.")

    hot_csv = out_dir / "hot_zone_attempts_scored.csv"
    hot.to_csv(hot_csv, index=False)
    print("Saved:", hot_csv)

    player_summary = hot.groupby(["player", "team"]).agg(
        attempts=("outcome", "size"),
        completed=("outcome", "sum"),
        expected=("pred_xbe", "sum"),
    ).reset_index()
    player_summary["completion_rate"] = player_summary["completed"] / player_summary["attempts"]
    player_summary["added_value"] = player_summary["completed"] - player_summary["expected"]
    player_summary = player_summary.sort_values("attempts", ascending=False)
    player_csv = out_dir / "hot_zone_player_summary.csv"
    player_summary.to_csv(player_csv, index=False)
    print("Saved:", player_csv)

    team_summary = hot.groupby("team").agg(
        attempts=("outcome", "size"),
        completed=("outcome", "sum"),
        expected=("pred_xbe", "sum"),
    ).reset_index()
    team_summary["completion_rate"] = team_summary["completed"] / team_summary["attempts"]
    team_summary["added_value"] = team_summary["completed"] - team_summary["expected"]
    team_summary = team_summary.sort_values("attempts", ascending=False)
    team_csv = out_dir / "hot_zone_team_summary.csv"
    team_summary.to_csv(team_csv, index=False)
    print("Saved:", team_csv)

    make_leaderboard(player_summary, zone, out_dir / "04_top_hot_zone_passers_eredivisie.png")
    make_zone_map(hot, zone, out_dir / "05_hot_zone_completed_passes_eredivisie.png")

    print("Done.")


if __name__ == "__main__":
    main()

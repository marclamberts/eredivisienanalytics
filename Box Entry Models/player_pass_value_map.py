"""
One player's open-play box-entry passes this season, each drawn on the
pitch and colored by its xBE value (P(pass into the box succeeds), from
build_box_entry_model.py / score_eredivisie_box_entries.py). Completed
attempts get a filled end-marker, incomplete ones a hollow ring, so you
can see both how hard a pass was (color) and whether it landed (marker).

Usage: python3 player_pass_value_map.py "M. Godts" [out.png]
"""
import sys
import re
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection
from mplsoccer import Pitch

BG = "#0d1117"
PITCH_LINE = "#2c3a4d"
TEXT_MAIN = "#e6e9ee"
TEXT_SUB = "#9aa4b2"
CMAP = "RdYlGn"
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


def make_plot(df, player, out_path):
    team = df["team"].mode().iat[0]
    n = len(df)
    n_complete = int(df["outcome"].sum())
    total_xbe = df["pred_xbe"].sum()

    fig = plt.figure(figsize=(14, 10.5))
    fig.patch.set_facecolor(BG)
    fig.text(0.5, 0.955, clean_name(player), fontsize=26, fontweight="bold", ha="center", color="white")
    fig.text(0.5, 0.915, f"{clean_name(team)}  ·  Open-Play Box-Entry Passes, Colored by xBE  ·  "
             "Eredivisie 2025/26  ·  Season", fontsize=12, ha="center", color=TEXT_SUB)

    pitch_ax = fig.add_axes([0.04, 0.08, 0.80, 0.76])
    pitch = Pitch(pitch_type="opta", pitch_color=BG, line_color=PITCH_LINE, linewidth=1.1)
    pitch.draw(ax=pitch_ax)

    norm = plt.Normalize(vmin=0, vmax=1)
    cmap = plt.get_cmap(CMAP)

    segments = [[(r.x, r.y), (r.end_x, r.end_y)] for r in df.itertuples()]
    colors = [cmap(norm(r.pred_xbe)) for r in df.itertuples()]
    lc = LineCollection(segments, colors=colors, linewidths=2.0, alpha=0.8, zorder=2)
    pitch_ax.add_collection(lc)

    complete = df[df["outcome"] == 1]
    incomplete = df[df["outcome"] == 0]
    pitch.scatter(complete["end_x"], complete["end_y"], ax=pitch_ax,
                  c=[cmap(norm(v)) for v in complete["pred_xbe"]],
                  s=70, edgecolors="white", linewidths=1.0, zorder=3)
    pitch.scatter(incomplete["end_x"], incomplete["end_y"], ax=pitch_ax,
                  facecolors="none", edgecolors=[cmap(norm(v)) for v in incomplete["pred_xbe"]],
                  s=70, linewidths=1.4, zorder=3)

    cbar_ax = fig.add_axes([0.87, 0.15, 0.02, 0.62])
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label("xBE (P completed)", color=TEXT_MAIN, fontsize=10, fontweight="bold")
    cbar.ax.yaxis.set_tick_params(color=TEXT_SUB, labelcolor=TEXT_SUB)
    cbar.outline.set_edgecolor(TEXT_SUB)

    legend_elems = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#9aa4b2",
               markeredgecolor="white", markersize=10, label=f"Completed ({n_complete})"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="none",
               markeredgecolor="#9aa4b2", markersize=10, label=f"Incomplete ({n - n_complete})"),
    ]
    fig.legend(handles=legend_elems, loc="lower center", bbox_to_anchor=(0.44, 0.015),
               ncol=2, frameon=False, fontsize=10, labelcolor=TEXT_MAIN)

    caption = f"{n} attempts  ·  {n_complete} completed ({n_complete / n * 100:.0f}%)  ·  {total_xbe:.1f} total xBE"
    fig.text(0.44, 0.045, caption, fontsize=11.5, ha="center", color="#c7ccd4")

    fig.text(0.98, 0.006, "Marc Lamberts", fontsize=9.5, ha="right", color="#6b7684", style="italic")
    fig.text(0.02, 0.006, "Data via Opta | Eredivisie 2025/26 event data · open-play only, excludes "
             "corners/free kicks/throw-ins · xBE from build_box_entry_model.py", fontsize=7.3, color="#6b7684")

    add_logo(fig)
    fig.savefig(out_path, dpi=200, facecolor=BG)
    plt.close(fig)
    print("Saved:", out_path)
    print(f"{player}: n={n} completed={n_complete} ({n_complete / n * 100:.1f}%) total_xbe={total_xbe:.2f}")


def main():
    out_dir = Path(__file__).parent
    player = sys.argv[1] if len(sys.argv) > 1 else "M. Godts"
    out = sys.argv[2] if len(sys.argv) > 2 else out_dir / f"07_pass_value_map_{player.replace(' ', '_')}.png"

    df = pd.read_csv(out_dir / "eredivisie_box_entry_attempts_scored.csv")
    match = df[df["player"].str.lower() == player.lower()]
    if match.empty:
        options = sorted(df["player"].unique())
        candidates = [p for p in options if player.lower() in p.lower()]
        if not candidates:
            print(f"Player '{player}' not found.")
            sys.exit(1)
        match = df[df["player"] == candidates[0]]
        player = candidates[0]

    make_plot(match, player, out)


if __name__ == "__main__":
    main()

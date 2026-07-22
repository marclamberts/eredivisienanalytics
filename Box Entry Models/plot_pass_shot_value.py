"""
Leaderboard for Pass Shot Value (PSV), the regression companion to xBE
(see pass_shot_value_model.py). Requires pass_shot_value_model.py to have
been run first.

Usage: python3 plot_pass_shot_value.py [out_dir]
"""
import sys
import re
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

BG = "#0d1117"
GRID_COLOR = "#232a35"
TEXT_MAIN = "#e6e9ee"
TEXT_SUB = "#9aa4b2"
BLUE = "#2f8fd1"
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


def main():
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent
    g = pd.read_csv(out_dir / "pass_shot_value_player_summary.csv")
    g = g[g["attempts"] >= 15].sort_values("total_psv", ascending=False).head(15).iloc[::-1]

    fig, ax = plt.subplots(figsize=(15, 11))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    labels = [f"{r.player}  ({clean_name(r.team)})" for r in g.itertuples()]
    ax.barh(labels, g["total_psv"], color=BLUE, height=0.65, zorder=3)
    for i, v in enumerate(g["total_psv"]):
        ax.text(v + g["total_psv"].max() * 0.012, i, f"{v:.2f}", va="center", ha="left",
                color=TEXT_MAIN, fontsize=10.5, fontweight="bold")

    ax.set_xlabel("Total Pass Shot Value (sum of per-pass regression scores, 0-1 each)",
                  fontsize=11, color=TEXT_MAIN, fontweight="bold", labelpad=10)
    ax.tick_params(axis="x", colors=TEXT_SUB, labelsize=10)
    ax.tick_params(axis="y", colors=TEXT_MAIN, labelsize=11)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(True, axis="x", color=GRID_COLOR, linewidth=0.6, alpha=0.6, zorder=0)

    fig.text(0.06, 0.965, "Pass Shot Value (PSV)", fontsize=24, fontweight="bold", color="white")
    fig.text(0.06, 0.938, "Regression score per box-entry pass: predicted xG of the shot it produces "
             "· min 15 attempts · Eredivisie 2025/26", fontsize=12, color=TEXT_SUB)
    fig.text(0.06, 0.012, "Data via Opta | PSV regresses each pass against the xG of a same-team shot "
             "within the next 5 events (0 if none) -- held-out R² is low (~0.05): shot quality after a "
             "pass is mostly noise, vs. xBE's AUC 0.82 for simply arriving in the box",
             fontsize=7.4, color="#6b7684")
    fig.text(0.98, 0.045, "Marc Lamberts", fontsize=9, ha="right", color="#6b7684", style="italic")
    fig.subplots_adjust(left=0.24, right=0.96, top=0.90, bottom=0.1)
    add_logo(fig)
    fig.savefig(out_dir / "06_pass_shot_value_leaderboard_eredivisie.png", dpi=200, facecolor=BG)
    plt.close(fig)
    print("Saved:", out_dir / "06_pass_shot_value_leaderboard_eredivisie.png")


if __name__ == "__main__":
    main()

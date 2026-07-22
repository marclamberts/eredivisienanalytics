"""
Apply the trained Expected Box Entry (xBE) model to the Eredivisie 2025/26
event data and produce the season leaderboards.

xBE is an alternative to xA: it values a pass by the probability it
successfully delivers the ball into the penalty box, not by the quality of
whatever shot the receiver takes afterward. A player's total xBE ("expected
box entries provided") is directly comparable to xA as a season output, but
isolates the passer's own contribution from a teammate's finishing.

Usage: python3 score_eredivisie_box_entries.py [out_dir]
"""
import sys
import glob
import re
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from adjustText import adjust_text

from build_box_entry_model import DATA_DIR, FEATURE_COLS, extract_box_entry_attempts

BG = "#0d1117"
GRID_COLOR = "#232a35"
TEXT_MAIN = "#e6e9ee"
TEXT_SUB = "#9aa4b2"
GREEN = "#4a9e5c"
RED = "#e0765c"
BLUE = "#2f8fd1"
AMBER = "#e0b34a"

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


def make_top_box_entry_passers(df, out_path, min_attempts=15):
    g = df.groupby(["player", "team"]).agg(
        n=("pred_xbe", "size"),
        expected=("pred_xbe", "sum"),
        actual=("outcome", "sum"),
    ).reset_index()
    g = g[g["n"] >= min_attempts].sort_values("expected", ascending=False).head(15)
    g = g.iloc[::-1]

    fig, ax = plt.subplots(figsize=(15, 11))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    labels = [f"{row.player}  ({clean_name(row.team)})" for row in g.itertuples()]
    ax.barh(labels, g["expected"], color=BLUE, height=0.65, zorder=3, label="Expected (xBE)")
    ax.scatter(g["actual"], labels, color="white", edgecolors=BG, s=90, zorder=4, label="Actual box entries")
    for i, (exp, act) in enumerate(zip(g["expected"], g["actual"])):
        label_x = max(exp, act) + g["expected"].max() * 0.012
        ax.text(label_x, i, f"{exp:.1f}", va="center", ha="left", color=TEXT_MAIN, fontsize=10.5, fontweight="bold")

    ax.set_xlim(0, max(g["expected"].max(), g["actual"].max()) * 1.12)
    ax.set_xlabel("Expected box entries provided (sum of xBE)", fontsize=11, color=TEXT_MAIN, fontweight="bold", labelpad=10)
    ax.tick_params(axis="x", colors=TEXT_SUB, labelsize=10)
    ax.tick_params(axis="y", colors=TEXT_MAIN, labelsize=11)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(True, axis="x", color=GRID_COLOR, linewidth=0.6, alpha=0.6, zorder=0)
    ax.legend(loc="lower right", frameon=False, fontsize=9.5, labelcolor=TEXT_MAIN)

    fig.text(0.06, 0.965, "Top Box Entry Passers", fontsize=24, fontweight="bold", color="white")
    fig.text(0.06, 0.938, f"By expected box entries (xBE)  ·  min {min_attempts} open-play attempts  ·  "
             "Eredivisie 2025/26  ·  Season", fontsize=12, color=TEXT_SUB)
    fig.text(0.06, 0.012, "Data via Opta | xBE = P(pass into the box succeeds), trained on this season's "
             "open-play passes · alternative to xA -- values reaching the box, not the receiver's shot",
             fontsize=7.6, color="#6b7684")
    fig.text(0.98, 0.012, "Marc Lamberts", fontsize=9, ha="right", color="#6b7684", style="italic")
    fig.subplots_adjust(left=0.24, right=0.96, top=0.90, bottom=0.07)
    add_logo(fig)
    fig.savefig(out_path, dpi=200, facecolor=BG)
    plt.close(fig)
    print("Saved:", out_path)


def make_team_creativity_quadrant(df, out_path):
    g = df.groupby("team").agg(
        attempts=("pred_xbe", "size"),
        expected=("pred_xbe", "sum"),
        actual=("outcome", "sum"),
    ).reset_index()
    g["added_value"] = g["actual"] - g["expected"]

    fig, ax = plt.subplots(figsize=(15, 12.5))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    x_med = g["attempts"].median()
    y_med = g["added_value"].median()
    ax.axvline(x_med, color=TEXT_SUB, linewidth=1.0, linestyle=(0, (4, 3)), alpha=0.6, zorder=1)
    ax.axhline(y_med, color=TEXT_SUB, linewidth=1.0, linestyle=(0, (4, 3)), alpha=0.6, zorder=1)

    colors = [GREEN if (row.attempts >= x_med and row.added_value >= y_med) else
             (RED if (row.attempts < x_med and row.added_value < y_med) else AMBER)
             for row in g.itertuples()]
    ax.scatter(g["attempts"], g["added_value"], s=140, color=colors, edgecolors=BG, linewidths=1.2, zorder=3)

    texts = [ax.text(row.attempts, row.added_value, clean_name(row.team), fontsize=10.5, color=TEXT_MAIN, zorder=4)
             for row in g.itertuples()]
    adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle="-", color="#6b7684", lw=0.6))

    ax.set_xlabel("Box-entry attempts (season total)", fontsize=11.5, color=TEXT_MAIN, fontweight="bold", labelpad=10)
    ax.set_ylabel("Added value vs. xBE\n(box entries completed − expected)", fontsize=11.5, color=TEXT_MAIN, fontweight="bold", labelpad=10)
    ax.tick_params(colors=TEXT_SUB, labelsize=10)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(True, color=GRID_COLOR, linewidth=0.6, alpha=0.6, zorder=0)

    fig.text(0.08, 0.965, "Team Box-Entry Profile", fontsize=24, fontweight="bold", color="white")
    fig.text(0.08, 0.938, "Volume of attempts vs. execution above expectation  ·  Eredivisie 2025/26  ·  Season",
             fontsize=12, color=TEXT_SUB)
    fig.text(0.08, 0.030, "Data via Opta | Eredivisie 2025/26 event data · open-play box-entry passes only "
             "(excludes corners/free kicks/throw-ins) · xBE trained on this season's data", fontsize=7.4, color="#6b7684")
    fig.text(0.08, 0.012, "Dashed lines = league median. Top-right: attempt a lot and execute above expectation.",
             fontsize=7.4, color="#6b7684")
    fig.text(0.98, 0.030, "Marc Lamberts", fontsize=9, ha="right", color="#6b7684", style="italic")
    fig.subplots_adjust(left=0.11, right=0.96, top=0.90, bottom=0.09)
    add_logo(fig)
    fig.savefig(out_path, dpi=200, facecolor=BG)
    plt.close(fig)
    print("Saved:", out_path)


def main():
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading model...")
    model = joblib.load(out_dir / "models" / "model_box_entry.pkl")

    files = sorted(glob.glob(f"{DATA_DIR}/*.json"))
    print(f"Parsing {len(files)} Eredivisie matches for box-entry attempts...")
    df, _ = extract_box_entry_attempts(files)
    print(f"Extracted {len(df)} open-play box-entry attempts across {df['team'].nunique()} teams.")

    df["pred_xbe"] = model.predict_proba(df[FEATURE_COLS])[:, 1]

    csv_path = out_dir / "eredivisie_box_entry_attempts_scored.csv"
    df.to_csv(csv_path, index=False)
    print("Saved:", csv_path)

    print("\nSanity check vs. training positive rate:")
    print(f"  actual completion rate {df['outcome'].mean():.3f}, mean predicted xBE {df['pred_xbe'].mean():.3f}")

    player_summary = df.groupby(["player", "team"]).agg(
        attempts=("pred_xbe", "size"),
        expected_box_entries=("pred_xbe", "sum"),
        actual_box_entries=("outcome", "sum"),
    ).reset_index()
    player_summary["added_value"] = player_summary["actual_box_entries"] - player_summary["expected_box_entries"]
    player_summary = player_summary.sort_values("expected_box_entries", ascending=False)
    player_csv = out_dir / "box_entry_player_summary.csv"
    player_summary.to_csv(player_csv, index=False)
    print("Saved:", player_csv)

    team_summary = df.groupby("team").agg(
        attempts=("pred_xbe", "size"),
        expected_box_entries=("pred_xbe", "sum"),
        actual_box_entries=("outcome", "sum"),
    ).reset_index()
    team_summary["added_value"] = team_summary["actual_box_entries"] - team_summary["expected_box_entries"]
    team_summary = team_summary.sort_values("expected_box_entries", ascending=False)
    team_csv = out_dir / "box_entry_team_summary.csv"
    team_summary.to_csv(team_csv, index=False)
    print("Saved:", team_csv)

    make_top_box_entry_passers(df, out_dir / "01_top_box_entry_passers_eredivisie.png")
    make_team_creativity_quadrant(df, out_dir / "02_team_box_entry_profile_eredivisie.png")

    print("Done.")


if __name__ == "__main__":
    main()

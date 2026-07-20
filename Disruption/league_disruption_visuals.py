"""
League-wide disruption visuals - Eredivisie 2025/26
=====================================================
Two mplsoccer/matplotlib charts built on the season disruption output:
  1. A pitch heatmap of WHERE linked defensive actions break up passes.
  2. A leaderboard of the top individual disruptors by total disruption value.

Usage: python3 league_disruption_visuals.py [out_dir]
"""
import glob
import os
import sys

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
from mplsoccer import VerticalPitch

import build_disruption_model as bdm

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

BG = "#0d1117"
PITCH_LINE = "#2c3a4d"
TEXT_SUB = "#9aa4b2"
TEXT_FOOT = "#6b7684"
GOLD = "#ffc247"

GOLD_RAMP = LinearSegmentedColormap.from_list("gold_ramp", [BG, "#5a4415", GOLD, "#fff2cf"])


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


def compute_attack_directions():
    """(match_file, contestant_id, period_id) -> 1 if the team attacks toward
    higher x that period, else -1. Raw event x is absolute pitch position, not
    relative to either team's own goal, so aggregating disruptions across many
    matches without this normalization mixes teams attacking in opposite
    directions and produces a meaningless "third" breakdown (same convention
    as Scripts/coach_profiling.py's get_attack_direction: a team's average
    pass position sits mostly in their own build-up territory, so avg_x < 50
    implies they're attacking toward high x that period).
    """
    files = sorted(glob.glob(os.path.join(bdm.DATA_DIR, "*.json")))
    directions = {}
    for fn in files:
        basename, events, _ = bdm.load_match(fn)
        sums = {}
        for e in events:
            if e["typeId"] != bdm.T_PASS or e.get("x") is None:
                continue
            if e["x"] == 0 and e["y"] == 0:
                continue
            key = (e["contestantId"], e["periodId"])
            s = sums.setdefault(key, [0.0, 0])
            s[0] += e["x"]
            s[1] += 1
        for (cid, period), (total, n) in sums.items():
            avg_x = total / n if n else 50
            directions[(basename, cid, period)] = 1 if avg_x < 50 else -1
    return directions


def make_heatmap(disr, out_path):
    linked = disr[disr["linked"] == True].copy()  # noqa: E712

    print("Computing per-team attacking direction per period to normalize "
          "disruption x-coordinates onto a common 'distance from own goal' axis...")
    directions = compute_attack_directions()
    dir_vals = [directions.get((mf, cid, p), 1)
                for mf, cid, p in zip(linked["match_file"], linked["contestant_id"], linked["period_id"])]
    dir_arr = np.array(dir_vals)
    linked["x_own"] = np.where(dir_arr == 1, linked["x"], bdm.GOAL_X - linked["x"])
    linked["y_own"] = np.where(dir_arr == 1, linked["y"], 68.0 - linked["y"])

    fig = plt.figure(figsize=(11, 13.7))
    fig.patch.set_facecolor(BG)
    pitch = VerticalPitch(pitch_type="uefa", pitch_color=BG, line_color=PITCH_LINE,
                          linewidth=1.1, half=False)
    ax = fig.add_axes([0.04, 0.09, 0.92, 0.78])
    pitch.draw(ax=ax)

    fig.text(0.5, 0.965, "Where Passes Get Disrupted", fontsize=24, fontweight="bold",
             ha="center", color="white")
    fig.text(0.5, 0.935, f"Eredivisie 2025/26  ·  Season  ·  {len(linked)} passes directly",
             fontsize=11.5, ha="center", color=TEXT_SUB)
    fig.text(0.5, 0.915, "broken up by a tackle, interception, clearance, aerial duel, "
             "recovery or dispossession", fontsize=11.5, ha="center", color=TEXT_SUB)

    stats = pitch.bin_statistic(linked["x_own"], linked["y_own"], statistic="count", bins=(18, 14))
    hm = pitch.heatmap(stats, ax=ax, cmap=GOLD_RAMP, edgecolors=BG, linewidth=0.15, zorder=1)
    cax = fig.add_axes([0.90, 0.24, 0.016, 0.45])
    cb = fig.colorbar(hm, cax=cax)
    cb.set_label("Disruptions", color=TEXT_SUB, fontsize=8.5)
    cb.ax.yaxis.set_tick_params(color=TEXT_SUB, labelcolor=TEXT_SUB, labelsize=7.5)

    own_third = (linked["x_own"] < 35).mean()
    mid_third = ((linked["x_own"] >= 35) & (linked["x_own"] < 70)).mean()
    att_third = (linked["x_own"] >= 70).mean()
    caption_lines = [
        f"{own_third:.0%} in the disrupting team's defensive third, {mid_third:.0%} in "
        f"midfield, {att_third:.0%} in their attacking third",
        "x normalized per team per period to \"distance from own goal\" (bottom = own goal, "
        "top = opponent goal) so directions aren't mixed",
    ]
    for i, line in enumerate(caption_lines):
        fig.text(0.5, 0.058 - i * 0.017, line, fontsize=9.5, ha="center", color="#c7ccd4")
    fig.text(0.98, 0.006, "Marc Lamberts", fontsize=9.5, ha="right", color=TEXT_FOOT, style="italic")
    fig.text(0.02, 0.006, "Data via Opta | disruption = defensive action linked to the specific "
             "opponent pass it broke up (see Disruption/build_disruption_model.py)",
             fontsize=7.0, color=TEXT_FOOT)

    add_logo(fig)
    fig.savefig(out_path, dpi=200, facecolor=BG)
    plt.close(fig)
    print("Saved:", out_path)


def make_leaderboard(player_summary, out_path, top_n=15, value_col="total_disruption",
                     count_col="actions_linked", title="Top Disruptors",
                     subtitle="Eredivisie 2025/26  ·  Season  ·  total disruption value = sum "
                              "of the pass completion probability denied by each linked "
                              "defensive action",
                     footer="Data via Opta | higher bar = denied more, harder-to-stop passes, "
                            "not just more defensive actions",
                     value_fmt="{:.1f}"):
    top = player_summary.sort_values(value_col, ascending=False).head(top_n)
    top = top.iloc[::-1]  # smallest at bottom for a horizontal barh

    fig, ax = plt.subplots(figsize=(12.5, 9.5))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    vmax = top[value_col].max()
    colors = [GOLD_RAMP(0.35 + 0.55 * (v / vmax)) for v in top[value_col]]
    ax.barh(range(len(top)), top[value_col], color=colors, height=0.62, zorder=3)

    labels = [f"{name}  ·  {team}" for name, team in zip(top["player_name"], top["team_name"])]
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(labels, fontsize=9.3, color="white")
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", colors=TEXT_SUB, labelsize=9)

    for i, (val, n_actions) in enumerate(zip(top[value_col], top[count_col])):
        ax.text(val + vmax * 0.012, i, f"{value_fmt.format(val)}  ({int(n_actions)} actions)",
               va="center", fontsize=9.5, color="#c7ccd4")

    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(PITCH_LINE)
    ax.grid(axis="x", color=PITCH_LINE, linewidth=0.6, alpha=0.5, zorder=0)
    ax.set_xlim(0, vmax * 1.22)

    fig.text(0.5, 0.965, title, fontsize=24, fontweight="bold", ha="center", color="white")
    fig.text(0.5, 0.935, subtitle, fontsize=10.5, ha="center", color=TEXT_SUB)

    fig.text(0.98, 0.012, "Marc Lamberts", fontsize=9.5, ha="right", color=TEXT_FOOT, style="italic")
    fig.text(0.02, 0.012, footer, fontsize=7.5, color=TEXT_FOOT)

    fig.subplots_adjust(left=0.40, right=0.95, top=0.90, bottom=0.07)
    add_logo(fig)
    fig.savefig(out_path, dpi=200, facecolor=BG)
    plt.close(fig)
    print("Saved:", out_path)


def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else OUT_DIR
    os.makedirs(out_dir, exist_ok=True)

    disr = pd.read_csv(os.path.join(OUT_DIR, "all_eredivisie_disruption_models.csv"))
    player_summary = pd.read_csv(os.path.join(OUT_DIR, "disruption_player_summary.csv"))

    make_heatmap(disr, os.path.join(out_dir, "league_disruption_heatmap.png"))
    make_leaderboard(player_summary, os.path.join(out_dir, "top_disruptors_leaderboard.png"))


if __name__ == "__main__":
    main()

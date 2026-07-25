"""
League-wide interception transition-threat visuals - Eredivisie 2025/26
===========================================================================
Two charts built on build_interception_transition_model.py's output, in
Marc Lamberts' Meridian house style (housestyle/ package at the repo
root -- warm paper/ink-navy surface, one terracotta accent spent once,
serif kicker-headline-dek header, Waltzing Analytics logo + dated credit
line on every chart):
  1. A pitch heatmap of WHERE interceptions that kick off the most
     transition threat happen (value-weighted bins, not just counts).
  2. A leaderboard of the top individual interceptors by total transition
     threat created.

Usage: python3 interception_transition_visuals.py
"""
import os
import sys

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
from mplsoccer import VerticalPitch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from housestyle import style, components  # noqa: E402
from housestyle.colors import SEQUENTIAL_BLUE_LIGHT, SEQUENTIAL_BLUE_DARK  # noqa: E402

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


def sequential_cmap(mode):
    steps = SEQUENTIAL_BLUE_LIGHT if mode == "light" else SEQUENTIAL_BLUE_DARK
    hexes = [steps[k] for k in sorted(steps)]
    return LinearSegmentedColormap.from_list(f"meridian_blue_{mode}", hexes)


def make_value_heatmap(interceptions, mode, out_path):
    palette, _ = style.apply(mode)

    print("Computing per-team attacking direction per period to normalize "
          "interception x/y onto a common 'distance from own goal' axis...")
    directions = compute_attack_directions()
    dir_vals = np.array([
        directions.get((mf, cid, p), 1)
        for mf, cid, p in zip(interceptions["match_file"], interceptions["contestant_id"],
                              interceptions["period_id"])
    ])
    interceptions = interceptions.copy()
    interceptions["x_own"] = np.where(dir_vals == 1, interceptions["x"], bdm.GOAL_X - interceptions["x"])
    interceptions["y_own"] = np.where(dir_vals == 1, interceptions["y"], 68.0 - interceptions["y"])
    interceptions["value_x1000"] = interceptions["transition_threat"] * 1000

    own_third = (interceptions["x_own"] < 35).mean()
    mid_third = ((interceptions["x_own"] >= 35) & (interceptions["x_own"] < 70)).mean()
    att_third = (interceptions["x_own"] >= 70).mean()
    thirds = {"defensive": own_third, "midfield": mid_third, "attacking": att_third}
    lead_third = max(thirds, key=thirds.get)

    fig = plt.figure(figsize=(8.2, 10.4))
    fig.patch.set_facecolor(palette["surface"])
    pitch = VerticalPitch(pitch_type="uefa", pitch_color=palette["surface"], line_color=palette["axis"],
                          linewidth=1.1, half=False, line_zorder=2)
    ax = fig.add_axes([0.07, 0.10, 0.80, 0.66])
    pitch.draw(ax=ax)

    stats = pitch.bin_statistic(interceptions["x_own"], interceptions["y_own"],
                                values=interceptions["value_x1000"], statistic="sum", bins=(16, 12))
    hm = pitch.heatmap(stats, ax=ax, cmap=sequential_cmap(mode), edgecolors=palette["surface"],
                       linewidth=0.15, zorder=1)
    cax = fig.add_axes([0.90, 0.22, 0.022, 0.42])
    cb = fig.colorbar(hm, cax=cax)
    cb.set_label("Transition threat, x1000", color=palette["ink_secondary"], fontsize=8.5)
    cb.ax.yaxis.set_tick_params(color=palette["ink_muted"], labelcolor=palette["ink_muted"], labelsize=7.5)
    cb.outline.set_visible(False)

    components.header(
        fig, kicker="Interceptions",
        title=f"Most transition threat starts in the {lead_third} third",
        dek=f"{own_third:.0%} defensive, {mid_third:.0%} midfield, {att_third:.0%} attacking third  ·  "
            f"{len(interceptions):,} interceptions this season",
        palette=palette)
    components.footer(fig, source=SOURCE, palette=palette)

    fig.savefig(out_path, dpi=200, facecolor=palette["surface"])
    plt.close(fig)
    print("Saved:", out_path)


def make_leaderboard(player_summary, mode, out_path, top_n=15):
    palette, _ = style.apply(mode)
    top = player_summary.sort_values("total_transition_threat_x1000", ascending=False).head(top_n)
    top = top.iloc[::-1]  # smallest at bottom for a horizontal barh
    leader = top.iloc[-1]

    fig = plt.figure(figsize=(9.5, 9))
    ax = fig.add_axes([0.34, 0.10, 0.58, 0.66])
    ax.set_facecolor(palette["surface"])

    bars = ax.barh(range(len(top)), top["total_transition_threat_x1000"], height=0.62, zorder=3)
    components.highlight_bars(bars, accent_index=len(top) - 1, palette=palette)

    labels = [f"{name}  ·  {team}" for name, team in zip(top["player_name"], top["team_name"])]
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(labels, fontsize=9.3, color=palette["ink_primary"])
    ax.tick_params(axis="y", length=0)

    vmax = top["total_transition_threat_x1000"].max()
    for i, (val, n) in enumerate(zip(top["total_transition_threat_x1000"], top["interceptions"])):
        ax.text(val + vmax * 0.02, i, f"{val:.1f}  ({int(n)} interceptions)",
                va="center", fontsize=9, color=palette["ink_secondary"])
    ax.set_xlim(0, vmax * 1.28)
    ax.grid(axis="x", zorder=0)

    components.header(
        fig, kicker="Interceptions",
        title=f"{leader['player_name']} sparks the most dangerous transitions",
        dek=f"Top {top_n} interceptors by transition threat created, x1000 for readability",
        palette=palette)
    components.footer(fig, source=SOURCE, palette=palette)

    fig.savefig(out_path, dpi=200, facecolor=palette["surface"])
    plt.close(fig)
    print("Saved:", out_path)


def main():
    interceptions = pd.read_csv(os.path.join(CSV_DIR, "all_eredivisie_interception_transitions.csv"))
    player_summary = pd.read_csv(os.path.join(CSV_DIR, "interception_transition_player_summary.csv"))

    for mode in ("light", "dark"):
        out_dir = visual_dir(mode)
        make_value_heatmap(interceptions, mode,
                           os.path.join(out_dir, "league_interception_transition_heatmap.png"))
        make_leaderboard(player_summary, mode,
                         os.path.join(out_dir, "top_interceptors_by_transition_threat_leaderboard.png"))


if __name__ == "__main__":
    main()

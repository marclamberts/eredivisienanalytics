"""
Goal kick -> shot-within-5-actions leaderboard - Eredivisie 2025/26
=======================================================================
Team leaderboard built on build_goalkick_shot_model.py's output, in Marc
Lamberts' Meridian house style (housestyle/ package at the repo root).

Usage: python3 goalkick_visuals.py
"""
import os
import sys

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from housestyle import style, components  # noqa: E402

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_DIR = os.path.join(OUT_DIR, "CSV")
SOURCE = "Opta event data, Eredivisie 2025/26"
MIN_GOAL_KICKS = 15  # drop teams with too few goal kicks for the rate to be meaningful


def visual_dir(theme):
    d = os.path.join(OUT_DIR, "Visual - Dark" if theme == "dark" else "Visual - Light")
    os.makedirs(d, exist_ok=True)
    return d


def make_leaderboard(team_summary, mode, out_path):
    palette, _ = style.apply(mode)
    teams = team_summary[team_summary["goal_kicks"] >= MIN_GOAL_KICKS].copy()
    teams["shot_rate_pct"] = teams["shot_rate"] * 100
    teams = teams.sort_values("shot_rate_pct", ascending=False)
    teams = teams.iloc[::-1]  # smallest at bottom for a horizontal barh
    leader = teams.iloc[-1]

    fig = plt.figure(figsize=(9.5, 9.5))
    ax = fig.add_axes([0.36, 0.10, 0.56, 0.66])
    ax.set_facecolor(palette["surface"])

    bars = ax.barh(range(len(teams)), teams["shot_rate_pct"], height=0.62, zorder=3)
    components.highlight_bars(bars, accent_index=len(teams) - 1, palette=palette)

    ax.set_yticks(range(len(teams)))
    ax.set_yticklabels(teams["team_name"], fontsize=9.6, color=palette["ink_primary"])
    ax.tick_params(axis="y", length=0)

    vmax = teams["shot_rate_pct"].max()
    for i, (rate, n, shots) in enumerate(zip(teams["shot_rate_pct"], teams["goal_kicks"],
                                              teams["shots_within_5"])):
        ax.text(rate + vmax * 0.03, i, f"{rate:.1f}%  ({int(shots)}/{int(n)} goal kicks)",
                va="center", fontsize=8.8, color=palette["ink_secondary"])
    ax.set_xlim(0, vmax * 1.5)
    ax.set_xlabel("Goal kicks leading to a shot within 5 actions")
    ax.grid(axis="x", zorder=0)

    components.header(
        fig, kicker="Goal Kicks",
        title=f"{leader['team_name']} gets to a shot fastest from goal kicks",
        dek=f"Eredivisie 2025/26  ·  share of goal kicks reaching a shot within 5 actions  ·  "
            f"teams with {MIN_GOAL_KICKS}+ goal kicks",
        palette=palette)
    components.footer(fig, source=SOURCE, palette=palette)

    fig.savefig(out_path, dpi=200, facecolor=palette["surface"])
    plt.close(fig)
    print("Saved:", out_path)


def main():
    team_summary = pd.read_csv(os.path.join(CSV_DIR, "goalkick_team_summary.csv"))

    for mode in ("light", "dark"):
        out_dir = visual_dir(mode)
        make_leaderboard(team_summary, mode,
                         os.path.join(out_dir, "goalkick_shot_within_5_leaderboard.png"))


if __name__ == "__main__":
    main()

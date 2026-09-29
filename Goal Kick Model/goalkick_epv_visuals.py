"""
Goal kick Expected Possession Value visuals - Eredivisie 2025/26
===================================================================
Built on build_goalkick_epv_model.py's output, in Marc Lamberts' Meridian
house style (housestyle/ package at the repo root):
  1. Team leaderboard: peak possession value reached from goal kicks
     (goals per 100 goal kicks), with the value of the kick itself alongside.
  2. Pitch grid: where goal kicks land vs. the possession value they build.
  3. Scatter: share of short goal kicks vs. peak possession value per team.

Usage: python3 goalkick_epv_visuals.py
"""
import os
import sys

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
import pandas as pd
from mplsoccer import Pitch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from housestyle import style, components, colors  # noqa: E402

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_DIR = os.path.join(OUT_DIR, "CSV")
SOURCE = "Opta event data, Eredivisie 2025/26 · EPV model trained on 2022/23-2024/25"
MIN_ZONE_KICKS = 15  # hide landing zones with too few goal kicks to average
# hand-placed scatter labels where teams sit on top of each other:
# team -> (x offset pt, y offset pt, horizontal alignment)
LABEL_NUDGE = {
    "SC Heerenveen": (6, -12, "left"),
    "Go Ahead Eagles": (6, -12, "left"),
    "FC Groningen": (-8, 3, "right"),
    "Feyenoord Rotterdam": (-8, 3, "right"),
}


def visual_dir(theme):
    d = os.path.join(OUT_DIR, "Visual - Dark" if theme == "dark" else "Visual - Light")
    os.makedirs(d, exist_ok=True)
    return d


def make_leaderboard(teams, mode, out_path):
    palette, _ = style.apply(mode)
    teams = teams.sort_values("poss_epv_max_per100", ascending=True).reset_index(drop=True)
    leader = teams.iloc[-1]

    fig = plt.figure(figsize=(9.5, 9.5))
    ax = fig.add_axes([0.34, 0.10, 0.58, 0.66])

    bars = ax.barh(range(len(teams)), teams["poss_epv_max_per100"], height=0.62, zorder=3)
    components.highlight_bars(bars, accent_index=len(teams) - 1, palette=palette)
    ax.scatter(teams["gk_epv_per100"], range(len(teams)), s=26, marker="D", zorder=4,
               color=palette["ink_primary"], edgecolors=palette["surface"], linewidth=0.6)

    ax.set_yticks(range(len(teams)))
    ax.set_yticklabels(teams["team_name"], fontsize=9.6, color=palette["ink_primary"])
    ax.tick_params(axis="y", length=0)
    vmax = teams["poss_epv_max_per100"].max()
    for i, (v, n, short) in enumerate(zip(teams["poss_epv_max_per100"], teams["goal_kicks"],
                                          teams["share_short"])):
        ax.text(v + vmax * 0.02, i, f"{v:.2f}   ({int(n)} kicks, {short:.0%} short)",
                va="center", fontsize=8.6, color=palette["ink_secondary"])
    ax.set_xlim(0, vmax * 1.45)
    ax.set_xlabel("Expected goals per 100 goal kicks")
    ax.grid(axis="x", zorder=0)
    ax.grid(axis="y", visible=False)
    ax.legend(handles=[
        Line2D([0], [0], color=palette["axis"], lw=7, label="Peak possession value reached"),
        Line2D([0], [0], marker="D", color="none", markerfacecolor=palette["ink_primary"],
               markersize=6, label="Value of the goal kick itself"),
    ], loc="lower right", frameon=False, fontsize=8.8)

    components.header(
        fig, kicker="Goal Kick EPV",
        title=f"{leader['team_name']} build the most value from their goal kicks",
        dek="Eredivisie 2025/26  ·  mean peak EPV (P score − P concede, next 10 actions) "
            "before losing the ball",
        palette=palette)
    components.footer(fig, source=SOURCE, palette=palette)
    fig.savefig(out_path, dpi=200, facecolor=palette["surface"])
    plt.close(fig)
    print("Saved:", out_path)


def make_zone_pitch(zones, mode, out_path):
    palette, _ = style.apply(mode)
    seq = colors.SEQUENTIAL_BLUE_LIGHT if mode == "light" else colors.SEQUENTIAL_BLUE_DARK
    cmap = LinearSegmentedColormap.from_list("meridian_seq", [seq[k] for k in sorted(seq)])

    z = zones[zones["n"] >= MIN_ZONE_KICKS]
    vmin, vmax = z["poss_epv_max_per100"].min(), z["poss_epv_max_per100"].max()
    best = z.loc[z["poss_epv_max_per100"].idxmax()]

    fig = plt.figure(figsize=(11, 8.6))
    pitch = Pitch(pitch_type="uefa", pitch_color=palette["surface"], line_color=palette["axis"],
                  linewidth=1.1, line_zorder=3)
    ax = fig.add_axes([0.05, 0.10, 0.90, 0.62])
    pitch.draw(ax=ax)

    for _, r in z.iterrows():
        frac = (r["poss_epv_max_per100"] - vmin) / (vmax - vmin) if vmax > vmin else 0.5
        ax.add_patch(plt.Rectangle((r["x0"], r["y0"]), r["x1"] - r["x0"], r["y1"] - r["y0"],
                                   facecolor=cmap(frac), edgecolor=palette["surface"],
                                   linewidth=1.5, alpha=0.9, zorder=2))
        txt_color = seq[100] if frac > 0.55 else palette["ink_primary"]
        cx, cy = (r["x0"] + r["x1"]) / 2, (r["y0"] + r["y1"]) / 2
        ax.text(cx, cy + 2.2, f"{r['poss_epv_max_per100']:.2f}", ha="center", va="center",
                fontsize=12, fontweight="bold", color=txt_color, zorder=4)
        ax.text(cx, cy - 2.6, f"{int(r['n'])} kicks · {r['completion']:.0%} won",
                ha="center", va="center", fontsize=7.8, color=txt_color, zorder=4)
    ax.annotate("", xy=(60, -3.5), xytext=(40, -3.5), annotation_clip=False,
                arrowprops=dict(arrowstyle="->", color=palette["ink_muted"], lw=1.2))
    ax.text(50, -5.8, "Attacking direction", ha="center", va="top", fontsize=8.5,
            color=palette["ink_muted"])

    if best["x0"] >= 17.5 and best["x1"] <= 35:
        zone_name = "just outside their own box"
    elif best["x1"] <= 35:
        zone_name = "inside their own box"
    else:
        zone_name = "beyond their own third"
    components.header(
        fig, kicker="Goal Kick EPV",
        title=f"Goal kicks played {zone_name} build the most value",
        dek="Eredivisie 2025/26  ·  mean peak expected possession value per 100 goal kicks, "
            f"by landing zone  ·  zones with {MIN_ZONE_KICKS}+ kicks",
        palette=palette)
    components.footer(fig, source=SOURCE, palette=palette)
    fig.savefig(out_path, dpi=200, facecolor=palette["surface"])
    plt.close(fig)
    print("Saved:", out_path)


def make_short_share_scatter(teams, mode, out_path):
    palette, _ = style.apply(mode)
    fig = plt.figure(figsize=(9.5, 7.6))
    ax = fig.add_axes([0.10, 0.13, 0.84, 0.60])

    leader_idx = teams["poss_epv_max_per100"].idxmax()
    muted = teams.drop(index=leader_idx)
    ax.scatter(muted["share_short"] * 100, muted["poss_epv_max_per100"], s=70,
               color=palette["axis"], edgecolors=palette["surface"], linewidth=0.8, zorder=3)
    lead = teams.loc[leader_idx]
    ax.scatter(lead["share_short"] * 100, lead["poss_epv_max_per100"], s=110,
               color=palette["accent"], edgecolors=palette["surface"], linewidth=0.8, zorder=4)
    for _, r in teams.iterrows():
        dx, dy, ha = LABEL_NUDGE.get(r["team_name"], (6, 3, "left"))
        ax.annotate(r["team_name"], (r["share_short"] * 100, r["poss_epv_max_per100"]),
                    xytext=(dx, dy), textcoords="offset points", fontsize=8.2, ha=ha,
                    color=palette["accent"] if r.name == leader_idx else palette["ink_secondary"])
    corr = teams["share_short"].corr(teams["poss_epv_max_per100"])
    ax.set_xlim(teams["share_short"].min() * 100 - 5, teams["share_short"].max() * 100 + 10)
    ax.set_xlabel("Share of goal kicks played short (landing in own third), %")
    ax.set_ylabel("Peak possession value, goals per 100 goal kicks")
    ax.grid(axis="both")

    components.header(
        fig, kicker="Goal Kick EPV",
        title="Playing short from goal kicks goes with more possession value",
        dek=f"Eredivisie 2025/26  ·  one dot per team  ·  correlation r = {corr:.2f}",
        palette=palette)
    components.footer(fig, source=SOURCE, palette=palette)
    fig.savefig(out_path, dpi=200, facecolor=palette["surface"])
    plt.close(fig)
    print("Saved:", out_path)


def main():
    teams = pd.read_csv(os.path.join(CSV_DIR, "goalkick_epv_team_summary.csv"))
    zones = pd.read_csv(os.path.join(CSV_DIR, "goalkick_epv_zone_grid.csv"))
    for mode in ("light", "dark"):
        out_dir = visual_dir(mode)
        make_leaderboard(teams, mode, os.path.join(out_dir, "goalkick_epv_team_leaderboard.png"))
        make_zone_pitch(zones, mode, os.path.join(out_dir, "goalkick_epv_landing_zones.png"))
        make_short_share_scatter(teams, mode, os.path.join(out_dir, "goalkick_epv_short_vs_value.png"))


if __name__ == "__main__":
    main()

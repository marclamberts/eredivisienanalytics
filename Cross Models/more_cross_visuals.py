"""
Second batch of visuals from the Eredivisie cross-model scoring run,
mined from the rest of the Ecuador Cross Models visuals/ folder (the
parts that only need the 6 cross models already applied, not the
separate clearance-landing model): team cross maps, a league-wide cross
maps grid, cross origin heatmaps, delivery type breakdown, cross
outcome mix by team, and a per-cross efficiency leaderboard.

Usage: python3 more_cross_visuals.py [out_dir]
"""
import sys
import collections
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mplsoccer import Pitch, VerticalPitch

from score_eredivisie_crosses import BG, GRID_COLOR, TEXT_MAIN, TEXT_SUB, GREEN, RED, BLUE, clean_name, add_logo

CSV_PATH = "/Users/marclamberts/Event data/Eredivisie/Cross Models/eredivisie_open_play_cross_events_scored.csv"
PITCH_LINE = "#2c3a4d"
GOLD = "#ffc247"
PURPLE = "#7b7fd6"


def outcome_class(row):
    if row["goal_created"]:
        return "goal"
    if row["shot_created"]:
        return "shot_no_goal"
    if row["outcome"] == 1:
        return "complete_no_shot"
    return "incomplete"


def make_team_cross_map(df, team, out_path, figsize=(11, 11)):
    sub = df[(df["team"] == team) & (df["shot_created"] == 1)]
    goals = sub[sub["goal_created"] == 1]
    shots = sub[sub["goal_created"] == 0]

    fig = plt.figure(figsize=figsize, facecolor=BG)
    pitch = VerticalPitch(pitch_type="opta", pitch_color=BG, line_color=PITCH_LINE, linewidth=1.1, half=True)
    ax = fig.add_axes([0.05, 0.08, 0.90, 0.72])
    pitch.draw(ax=ax)

    for _, r in shots.iterrows():
        pitch.lines(r["x"], r["y"], r["end_x"], r["end_y"], ax=ax, color=GOLD, lw=1.1, alpha=0.65, zorder=2, comet=False)
        pitch.scatter(r["end_x"], r["end_y"], ax=ax, s=30, color=GOLD, alpha=0.85, zorder=3, linewidths=0)
    for _, r in goals.iterrows():
        pitch.lines(r["x"], r["y"], r["end_x"], r["end_y"], ax=ax, color=GREEN, lw=1.8, alpha=0.9, zorder=3, comet=False)
        pitch.scatter(r["end_x"], r["end_y"], ax=ax, s=60, color=GREEN, edgecolors="white", linewidth=0.8, zorder=4)

    handles = [
        Line2D([0], [0], color=GREEN, linewidth=2.5, label=f"Goal ({len(goals)})"),
        Line2D([0], [0], color=GOLD, linewidth=2, label=f"Shot, no goal ({len(shots)})"),
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.83), ncol=2,
              frameon=False, fontsize=10.5, labelcolor="#c7ccd4")

    n_total = len(sub)
    conv = len(goals) / n_total * 100 if n_total else 0
    providers = sub["player"].value_counts().head(4)
    top_str = "  ·  ".join(f"{name} ({n})" for name, n in providers.items())

    fig.text(0.5, 0.965, clean_name(team), fontsize=24, fontweight="bold", ha="center", color="white")
    fig.text(0.5, 0.935, "Crosses That Created a Shot  ·  Open Play Only  ·  Eredivisie 2025/26  ·  Season",
             fontsize=11.5, ha="center", color=TEXT_SUB)
    fig.text(0.5, 0.045, f"{n_total} open-play crosses led to a shot this season  ·  {len(goals)} goals  ·  "
             f"{conv:.0f}% conversion", fontsize=12, ha="center", color="#c7ccd4")
    fig.text(0.5, 0.022, f"Top providers: {top_str}" if top_str else "", fontsize=9, ha="center", color="#6b7684")
    fig.text(0.02, 0.006, "Data via Opta | Eredivisie 2025/26 event data · open-play crosses only · "
             "shot linked via qualifier 55 · model trained on Ecuador 2026, applied unchanged",
             fontsize=7.0, color="#6b7684")
    fig.text(0.98, 0.006, "Marc Lamberts", fontsize=9, ha="right", color="#6b7684", style="italic")
    add_logo(fig, width=0.13)
    fig.savefig(out_path, dpi=200, facecolor=BG)
    plt.close(fig)
    print("Saved:", out_path)


def make_league_cross_maps_grid(df, out_path):
    teams = sorted(df["team"].unique())
    teams_by_shots = df[df["shot_created"] == 1].groupby("team").size().sort_values(ascending=False)
    teams = [t for t in teams_by_shots.index] + [t for t in teams if t not in teams_by_shots.index]

    n = len(teams)
    cols, rows = 4, int(np.ceil(n / 4))
    fig = plt.figure(figsize=(4.6 * cols, 4.9 * rows), facecolor=BG)
    fig.text(0.02, 0.988, "Eredivisie 2025/26  ·  Crosses That Created a Shot — All 18 Teams",
             fontsize=20, fontweight="bold", color="white", va="top")
    fig.text(0.02, 0.965, "Open play only  ·  ranked by shot-creating crosses  ·  Season",
             fontsize=11, color=TEXT_SUB, va="top")

    for i, team in enumerate(teams):
        r, c = divmod(i, cols)
        ax = fig.add_axes([0.01 + c / cols, 0.94 - (r + 1) * (0.90 / rows), 1 / cols - 0.012, 0.90 / rows - 0.03])
        pitch = VerticalPitch(pitch_type="opta", pitch_color=BG, line_color=PITCH_LINE, linewidth=0.8, half=True)
        pitch.draw(ax=ax)
        sub = df[(df["team"] == team) & (df["shot_created"] == 1)]
        goals = sub[sub["goal_created"] == 1]
        shots = sub[sub["goal_created"] == 0]
        for _, r_ in shots.iterrows():
            pitch.lines(r_["x"], r_["y"], r_["end_x"], r_["end_y"], ax=ax, color=GOLD, lw=0.8, alpha=0.55, zorder=2, comet=False)
        for _, r_ in goals.iterrows():
            pitch.lines(r_["x"], r_["y"], r_["end_x"], r_["end_y"], ax=ax, color=GREEN, lw=1.3, alpha=0.9, zorder=3, comet=False)
        ax.set_title(f"{clean_name(team)}  ({len(sub)}, {len(goals)}g)", fontsize=10.5, fontweight="bold", color=TEXT_MAIN, pad=4)

    fig.text(0.02, 0.008, "Data via Opta | Eredivisie 2025/26 event data · gold = shot no goal, green = goal · "
             "open-play crosses only, shot linked via qualifier 55", fontsize=8, color="#6b7684")
    fig.text(0.98, 0.008, "Marc Lamberts · Waltzing Analytics", fontsize=8.5, ha="right", color="#6b7684", style="italic")
    fig.savefig(out_path, dpi=170, facecolor=BG)
    plt.close(fig)
    print("Saved:", out_path)


def make_origin_heatmaps(df, out_path):
    fig = plt.figure(figsize=(17, 9.5), facecolor=BG)
    fig.text(0.5, 0.955, "Where the Danger Comes From", fontsize=24, fontweight="bold", ha="center", color="white")
    fig.text(0.5, 0.918, "Open-play cross origin volume vs. average predicted delivery value  ·  Eredivisie 2025/26  ·  Season",
             fontsize=11.5, ha="center", color=TEXT_SUB)

    pitch = Pitch(pitch_type="opta", pitch_color=BG, line_color=PITCH_LINE, linewidth=1.0)

    ax1 = fig.add_axes([0.03, 0.08, 0.44, 0.76])
    pitch.draw(ax=ax1)
    ax1.set_title("Origin volume (all crosses)", fontsize=13, color=TEXT_MAIN, pad=8)
    stats_n = pitch.bin_statistic(df["x"], df["y"], statistic="count", bins=(24, 18))
    stats_n["statistic"] = np.where(stats_n["statistic"] == 0, np.nan, stats_n["statistic"])
    pitch.heatmap(stats_n, ax=ax1, cmap="YlOrRd", edgecolors=BG, linewidth=0.1)

    ax2 = fig.add_axes([0.52, 0.08, 0.44, 0.76])
    pitch.draw(ax=ax2)
    ax2.set_title("Avg. predicted delivery value (min 3 crosses/cell)", fontsize=13, color=TEXT_MAIN, pad=8)
    stats_val = pitch.bin_statistic(df["x"], df["y"], values=df["pred_cross_delivery_value"], statistic="mean", bins=(24, 18))
    stats_cnt = pitch.bin_statistic(df["x"], df["y"], statistic="count", bins=(24, 18))
    stats_val["statistic"] = np.where(stats_cnt["statistic"] < 3, np.nan, stats_val["statistic"])
    pitch.heatmap(stats_val, ax=ax2, cmap="YlOrRd", edgecolors=BG, linewidth=0.1)

    fig.text(0.03, 0.02, "Data via Opta | Eredivisie 2025/26 event data · open-play crosses only (excludes "
             "corners/free kicks/throw-ins)", fontsize=8, color="#6b7684")
    fig.text(0.98, 0.02, "Marc Lamberts", fontsize=9, ha="right", color="#6b7684", style="italic")
    add_logo(fig, width=0.09, margin=0.012)
    fig.savefig(out_path, dpi=200, facecolor=BG)
    plt.close(fig)
    print("Saved:", out_path)


def make_delivery_type_breakdown(df, out_path):
    order = ["right_foot", "left_foot", "other"]
    counts = df["body_part"].value_counts().reindex(order).fillna(0)
    comp = df.groupby("body_part")["outcome"].mean().reindex(order)
    chance = df.groupby("body_part")["shot_created"].mean().reindex(order)

    fig, axes = plt.subplots(1, 2, figsize=(18, 8.2))
    fig.patch.set_facecolor(BG)
    colors = [BLUE, PURPLE, GREEN]

    ax = axes[0]
    ax.set_facecolor(BG)
    bars = ax.bar(order, counts, color=colors, zorder=3)
    for b, v in zip(bars, counts):
        ax.text(b.get_x() + b.get_width() / 2, v + max(counts) * 0.01, f"{int(v):,}", ha="center", color=TEXT_MAIN, fontsize=11, fontweight="bold")
    ax.set_title("Crosses by delivery type", fontsize=14, color=TEXT_MAIN, loc="left", pad=10)
    ax.tick_params(colors=TEXT_SUB, labelsize=11)
    ax.grid(True, axis="y", color=GRID_COLOR, alpha=0.6, zorder=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    # "other" body_part is a handful of events in this feed (n<10) --
    # rates on that few samples are noise, so the rate panel only shows
    # the two body parts with a meaningful sample size.
    rate_order = [bp for bp in order if counts.get(bp, 0) >= 30]
    ax = axes[1]
    ax.set_facecolor(BG)
    xw = np.arange(len(rate_order))
    w = 0.35
    ax.bar(xw - w / 2, comp.reindex(rate_order), width=w, color=BLUE, label="Completion rate", zorder=3)
    ax.bar(xw + w / 2, chance.reindex(rate_order), width=w, color=GOLD, label="Chance-creation rate", zorder=3)
    ax.set_xticks(xw)
    ax.set_xticklabels(rate_order)
    ax.set_title("Outcome rate by delivery type", fontsize=14, color=TEXT_MAIN, loc="left", pad=10)
    ax.tick_params(colors=TEXT_SUB, labelsize=11)
    ax.legend(frameon=False, labelcolor="#c7ccd4", fontsize=10)
    ax.grid(True, axis="y", color=GRID_COLOR, alpha=0.6, zorder=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    if len(rate_order) < len(order):
        ax.text(0.98, 0.96, "\"other\" omitted (n<30, too few for a reliable rate)",
                transform=ax.transAxes, ha="right", va="top", fontsize=8.5, color=TEXT_SUB, style="italic")

    fig.suptitle("Delivery Type Breakdown", fontsize=24, fontweight="bold", color="white", x=0.5, y=0.99)
    fig.text(0.5, 0.925, "Open-play crosses  ·  Eredivisie 2025/26  ·  Season", fontsize=12, ha="center", color=TEXT_SUB)
    fig.text(0.02, 0.01, "Data via Opta | Eredivisie 2025/26 event data · open-play crosses only · shot linked via qualifier 55",
             fontsize=8, color="#6b7684")
    fig.text(0.98, 0.01, "Marc Lamberts", fontsize=9, ha="right", color="#6b7684", style="italic")
    add_logo(fig, width=0.08, margin=0.012)
    fig.subplots_adjust(left=0.05, right=0.97, top=0.86, bottom=0.10, wspace=0.18)
    fig.savefig(out_path, dpi=200, facecolor=BG)
    plt.close(fig)
    print("Saved:", out_path)


def make_outcome_by_team(df, out_path):
    df = df.copy()
    df["outcome_class"] = df.apply(outcome_class, axis=1)
    cats = ["goal", "shot_no_goal", "complete_no_shot", "incomplete"]
    colors = {"goal": GREEN, "shot_no_goal": GOLD, "complete_no_shot": BLUE, "incomplete": "#333a48"}

    counts_n = df.groupby("team").size()
    tab = pd.crosstab(df["team"], df["outcome_class"], normalize="index").reindex(columns=cats).fillna(0) * 100
    order = tab["goal"].sort_values(ascending=False).index
    tab = tab.loc[order]
    productive = tab["goal"] + tab["shot_no_goal"] + tab["complete_no_shot"]

    # The 4 outcomes are exhaustive (sum to 100%), but "incomplete" is
    # 60-90% of every bar and isn't the interesting part of this chart --
    # it's covered by the completion-rate panel in the delivery-type
    # breakdown chart instead. Crop the x-axis just past the productive
    # total so goal/shot/complete-no-shot -- what this chart is actually
    # ranking on -- fill most of the frame; incomplete still starts at
    # the right place and simply runs off the visible edge.
    x_max = productive.max() * 1.28

    fig, ax = plt.subplots(figsize=(15, 10.5))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    labels = [clean_name(t) for t in tab.index]
    left = np.zeros(len(tab))
    bar_h = 0.62
    for cat in cats:
        vals = tab[cat].values
        ax.barh(labels, vals, left=left, height=bar_h, color=colors[cat],
               label=cat.replace("_", " "), zorder=3, edgecolor=BG, linewidth=0.6)
        left += vals

    for i, team in enumerate(tab.index):
        g = tab.loc[team, "goal"]
        prod = productive.loc[team]
        n = counts_n.loc[team]
        ax.text(prod + x_max * 0.012, i, f"{prod:.0f}% productive  ·  {g:.1f}% goals  ·  n={n}",
               va="center", ha="left", fontsize=9, color=TEXT_MAIN, zorder=4)

    ax.set_xlabel("Share of open-play crosses (%) -- \"incomplete\" runs off-frame at right", fontsize=10.5,
                 color=TEXT_MAIN, fontweight="bold", labelpad=8)
    ax.tick_params(colors=TEXT_SUB, labelsize=10.5)
    ax.invert_yaxis()
    ax.set_xlim(0, x_max)
    ax.set_ylim(len(tab) - 0.3, -0.7)
    ax.legend(loc="upper center", bbox_to_anchor=(0.35, 1.13), ncol=4, frameon=False, fontsize=10, labelcolor="#c7ccd4")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(True, axis="x", color=GRID_COLOR, alpha=0.5, zorder=0)

    fig.text(0.06, 0.975, "Cross Outcome Mix by Team", fontsize=22, fontweight="bold", color="white")
    fig.text(0.06, 0.948, "Open-play crosses, ranked by goal share  ·  Eredivisie 2025/26  ·  Season", fontsize=11.5, color=TEXT_SUB)
    fig.text(0.06, 0.015, "Data via Opta | Eredivisie 2025/26 event data · open-play crosses only · shot linked via "
             "qualifier 55 · \"productive\" = goal + shot no goal + complete no shot", fontsize=7.6, color="#6b7684")
    fig.text(0.98, 0.015, "Marc Lamberts", fontsize=9, ha="right", color="#6b7684", style="italic")
    fig.subplots_adjust(left=0.21, right=0.96, top=0.85, bottom=0.07)
    add_logo(fig, width=0.09, margin=0.012)
    fig.savefig(out_path, dpi=200, facecolor=BG)
    plt.close(fig)
    print("Saved:", out_path)


def make_efficiency_leaderboard(df, out_path, min_crosses=15):
    g = df.groupby(["player", "team"]).agg(
        n=("pred_cross_delivery_value", "size"),
        avg_value=("pred_cross_delivery_value", "mean"),
    ).reset_index()
    g = g[g["n"] >= min_crosses].sort_values("avg_value", ascending=False).head(15).iloc[::-1]

    fig, ax = plt.subplots(figsize=(15, 11))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    labels = [f"{row.player}  ({clean_name(row.team)}, n={row.n})" for row in g.itertuples()]
    ax.barh(labels, g["avg_value"], color=GOLD, height=0.65, zorder=3)
    for i, v in enumerate(g["avg_value"]):
        ax.text(v + 0.001, i, f"{v:.3f}", va="center", ha="left", color=TEXT_MAIN, fontsize=10.5, fontweight="bold")

    ax.set_xlabel("Avg. predicted delivery value per cross", fontsize=11, color=TEXT_MAIN, fontweight="bold", labelpad=10)
    ax.tick_params(axis="x", colors=TEXT_SUB, labelsize=10)
    ax.tick_params(axis="y", colors=TEXT_MAIN, labelsize=10.5)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(True, axis="x", color=GRID_COLOR, linewidth=0.6, alpha=0.6, zorder=0)

    fig.text(0.06, 0.965, "Cross Efficiency Leaderboard", fontsize=22, fontweight="bold", color="white")
    fig.text(0.06, 0.938, f"By average delivery value per attempt  ·  min {min_crosses} open-play crosses  ·  "
             f"Eredivisie 2025/26  ·  Season", fontsize=11.5, color=TEXT_SUB)
    fig.text(0.06, 0.012, "Data via Opta | Eredivisie 2025/26 event data · open-play crosses only · quality, not "
             "volume -- compare with the Top Crossers (volume) chart", fontsize=7.6, color="#6b7684")
    fig.text(0.98, 0.012, "Marc Lamberts", fontsize=9, ha="right", color="#6b7684", style="italic")
    fig.subplots_adjust(left=0.30, right=0.96, top=0.90, bottom=0.07)
    add_logo(fig, width=0.09, margin=0.012)
    fig.savefig(out_path, dpi=200, facecolor=BG)
    plt.close(fig)
    print("Saved:", out_path)


def main():
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/Users/marclamberts/Event data/Eredivisie/Cross Models")
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(CSV_PATH)
    print(f"Loaded {len(df)} scored crosses.")

    make_team_cross_map(df, "PSV Eindhoven", out_dir / "03_cross_map_psv.png")
    make_league_cross_maps_grid(df, out_dir / "04_cross_maps_grid_eredivisie.png")
    make_origin_heatmaps(df, out_dir / "05_cross_origin_heatmaps_eredivisie.png")
    make_delivery_type_breakdown(df, out_dir / "06_delivery_type_breakdown_eredivisie.png")
    make_outcome_by_team(df, out_dir / "07_cross_outcome_by_team_eredivisie.png")
    make_efficiency_leaderboard(df, out_dir / "08_cross_efficiency_leaderboard_eredivisie.png")

    print("Done.")


if __name__ == "__main__":
    main()

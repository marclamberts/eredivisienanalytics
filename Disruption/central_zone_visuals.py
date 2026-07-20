"""
Ten visuals drilling into central defending. Eredivisie 2025/26.

Companion set to central_defending_leaderboard.py. All use the same
convention: y_own normalized per disrupting team's own attacking direction
(compute_attack_directions), central third = y in [22.67, 45.33] of the
68m width, flanks = everything else.

Usage: python3 central_zone_visuals.py [out_dir]
"""
import glob
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
from mplsoccer import VerticalPitch

import build_disruption_model as bdm
import league_disruption_visuals as ldv
from league_disruption_visuals import compute_attack_directions, add_logo

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
CENTRAL_LOW, CENTRAL_HIGH = 68 / 3, 2 * 68 / 3
MIN_ACTIONS = 5


def prep():
    files = sorted(glob.glob(os.path.join(bdm.DATA_DIR, "*.json")))
    values = pd.read_csv(os.path.join(ldv.CSV_DIR, "all_eredivisie_disruption_values.csv"))
    directions = compute_attack_directions()
    dir_vals = np.array([
        directions.get((mf, cid, p), 1)
        for mf, cid, p in zip(values["match_file"], values["contestant_id"], values["period_id"])
    ])
    values["x_own"] = np.where(dir_vals == 1, values["x"], bdm.GOAL_X - values["x"])
    values["y_own"] = np.where(dir_vals == 1, values["y"], 68.0 - values["y"])
    values["value_x1000"] = values["disruption_value"] * 1000
    values["is_central"] = (values["y_own"] >= CENTRAL_LOW) & (values["y_own"] <= CENTRAL_HIGH)
    team_matches = pd.read_csv(os.path.join(ldv.CSV_DIR, "disruption_value_team_summary.csv"))[
        ["team_name", "matches"]]
    return values, team_matches


def hbar(g, value_col, count_col, out_path, title, subtitle, footer, value_fmt="{:.2f}",
        label_col="team_name", figsize=(12.5, 9.5), left_margin=0.20):
    g = g.sort_values(value_col, ascending=True)
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(ldv.BG)
    ax.set_facecolor(ldv.BG)
    vmax = max(g[value_col].max(), 1e-9)
    colors = [ldv.GOLD_RAMP(0.3 + 0.6 * (v / vmax)) for v in g[value_col]]
    ax.barh(g[label_col], g[value_col], color=colors, height=0.62, zorder=3)
    for i, (val, n) in enumerate(zip(g[value_col], g[count_col])):
        ax.text(val + vmax * 0.012, i, f"{value_fmt.format(val)}  ({int(n)})",
               va="center", fontsize=9.3, color=ldv.LEGEND_TEXT)
    ax.tick_params(axis="y", colors=ldv.TEXT_MAIN, labelsize=10, length=0)
    ax.tick_params(axis="x", colors=ldv.TEXT_SUB, labelsize=9)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(ldv.PITCH_LINE)
    ax.grid(axis="x", color=ldv.PITCH_LINE, linewidth=0.6, alpha=0.5, zorder=0)
    ax.set_xlim(0, vmax * 1.2)
    fig.text(0.5, 0.965, title, fontsize=23, fontweight="bold", ha="center", color=ldv.TEXT_MAIN)
    fig.text(0.5, 0.935, subtitle, fontsize=10, ha="center", color=ldv.TEXT_SUB)
    fig.text(0.98, 0.012, "Marc Lamberts", fontsize=9.5, ha="right", color=ldv.TEXT_FOOT, style="italic")
    fig.text(0.02, 0.012, footer, fontsize=7.2, color=ldv.TEXT_FOOT)
    fig.subplots_adjust(left=left_margin, right=0.95, top=0.90, bottom=0.07)
    add_logo(fig)
    fig.savefig(out_path, dpi=200, facecolor=ldv.BG)
    plt.close(fig)
    print("Saved:", out_path)


# 1. Wide (flank) defending leaderboard -- mirror of central
def viz_wide_leaderboard(values, team_matches, out_dir):
    wide = values[~values["is_central"]]
    g = wide.groupby("team_name").agg(n=("value_x1000", "size"), total=("value_x1000", "sum")) \
        .reset_index().merge(team_matches, on="team_name")
    g["per_match"] = g["total"] / g["matches"]
    hbar(g, "per_match", "n", os.path.join(out_dir, "01_wide_defending_leaderboard.png"),
        "Best Wide Defending",
        "Eredivisie 2025/26 · Season · disruption value denied per match, flanks only (central third excluded)",
        "Data via Opta | wide = outside the middle third of the 68m width")


# 2. Central vs wide split, grouped bar per team
def viz_central_vs_wide(values, team_matches, out_dir):
    g = values.groupby(["team_name", "is_central"])["value_x1000"].sum().unstack(fill_value=0)
    g.columns = ["wide", "central"]
    g = g.reset_index().merge(team_matches, on="team_name")
    g["central_pm"] = g["central"] / g["matches"]
    g["wide_pm"] = g["wide"] / g["matches"]
    g = g.sort_values("central_pm", ascending=True)

    fig, ax = plt.subplots(figsize=(13, 9.5))
    fig.patch.set_facecolor(ldv.BG)
    ax.set_facecolor(ldv.BG)
    y = np.arange(len(g))
    h = 0.36
    ax.barh(y + h / 2, g["central_pm"], height=h, color=ldv.GOLD, alpha=0.9, label="Central third", zorder=3)
    ax.barh(y - h / 2, g["wide_pm"], height=h, color=ldv.BLUE, alpha=0.85, label="Flanks", zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(g["team_name"], fontsize=9.8, color=ldv.TEXT_MAIN)
    ax.tick_params(axis="x", colors=ldv.TEXT_SUB, labelsize=9)
    ax.tick_params(axis="y", length=0)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(ldv.PITCH_LINE)
    ax.grid(axis="x", color=ldv.PITCH_LINE, linewidth=0.5, alpha=0.4, zorder=0)
    ax.legend(loc="lower right", frameon=False, fontsize=9.5, labelcolor=ldv.LEGEND_TEXT)
    fig.text(0.5, 0.965, "Central vs. Wide Defending", fontsize=23, fontweight="bold", ha="center", color=ldv.TEXT_MAIN)
    fig.text(0.5, 0.935, "Eredivisie 2025/26 · Season · disruption value denied per match, by zone",
             fontsize=10.5, ha="center", color=ldv.TEXT_SUB)
    fig.text(0.98, 0.012, "Marc Lamberts", fontsize=9.5, ha="right", color=ldv.TEXT_FOOT, style="italic")
    fig.text(0.02, 0.012, "Data via Opta | central = middle third of pitch width, flanks = outer two thirds",
             fontsize=7.2, color=ldv.TEXT_FOOT)
    fig.subplots_adjust(left=0.19, right=0.96, top=0.90, bottom=0.07)
    add_logo(fig)
    out_path = os.path.join(out_dir, "02_central_vs_wide.png")
    fig.savefig(out_path, dpi=200, facecolor=ldv.BG)
    plt.close(fig)
    print("Saved:", out_path)


# 3. Central defending value by action type
def viz_central_by_action_type(values, out_dir):
    central = values[values["is_central"]]
    g = central.groupby("action_type").agg(n=("value_x1000", "size"), mean=("value_x1000", "mean")).reset_index()
    hbar(g, "mean", "n", os.path.join(out_dir, "03_central_by_action_type.png"),
        "Central Defending: What Works", "Eredivisie 2025/26 · Season · mean disruption value per action, central third only",
        "Data via Opta | mean value per action, by action type, central third of pitch width",
        label_col="action_type", figsize=(11, 6.5), left_margin=0.18)


# 4. Small multiples: central-only disruptions per team
def viz_central_pitch_grid(values, team_matches, out_dir):
    central = values[values["is_central"]]
    team_order = (central.groupby("team_name")["value_x1000"].sum()
                 .sort_values(ascending=False).index.tolist())
    n_cols, n_rows = 6, 3
    fig = plt.figure(figsize=(n_cols * 3.4, n_rows * 4.4))
    fig.patch.set_facecolor(ldv.BG)
    fig.text(0.5, 0.975, "Central-Zone Disruptions, Every Team", fontsize=23, fontweight="bold",
             ha="center", color=ldv.TEXT_MAIN)
    fig.text(0.5, 0.952, "Eredivisie 2025/26 · Season · linked disruptions inside the central third only",
             fontsize=10.5, ha="center", color=ldv.TEXT_SUB)
    grid_top, grid_bottom, grid_left, grid_right = 0.90, 0.05, 0.02, 0.98
    cell_w, cell_h = (grid_right - grid_left) / n_cols, (grid_top - grid_bottom) / n_rows
    for i, team in enumerate(team_order):
        row, col = divmod(i, n_cols)
        left = grid_left + col * cell_w
        bottom = grid_top - (row + 1) * cell_h
        ax = fig.add_axes([left + cell_w * 0.06, bottom + cell_h * 0.03, cell_w * 0.88, cell_h * 0.80])
        pitch = VerticalPitch(pitch_type="uefa", pitch_color=ldv.BG, line_color=ldv.PITCH_LINE, linewidth=0.8, half=False)
        pitch.draw(ax=ax)
        sub = central[central["team_name"] == team]
        pitch.scatter(sub["x_own"], sub["y_own"], ax=ax, s=20, color=ldv.GOLD, alpha=0.6, edgecolors="none", zorder=3)
        ax.set_title(team, fontsize=10, fontweight="bold", color=ldv.TEXT_MAIN, pad=4)
        fig.text(left + cell_w / 2, bottom + cell_h * 0.01, f"{len(sub)} actions · {sub['value_x1000'].sum():.1f} value",
                 fontsize=7.5, ha="center", color=ldv.TEXT_SUB)
    fig.text(0.98, 0.012, "Marc Lamberts", fontsize=9.5, ha="right", color=ldv.TEXT_FOOT, style="italic")
    fig.text(0.02, 0.012, "Data via Opta | pitch runs bottom = own goal, top = opponent goal",
             fontsize=8, color=ldv.TEXT_FOOT)
    add_logo(fig, width=0.045)
    out_path = os.path.join(out_dir, "04_central_pitch_grid.png")
    fig.savefig(out_path, dpi=200, facecolor=ldv.BG)
    plt.close(fig)
    print("Saved:", out_path)


# 5. Central defending volume leaderboard (raw actions/match, not value)
def viz_central_volume(values, team_matches, out_dir):
    central = values[values["is_central"]]
    g = central.groupby("team_name").size().rename("n").reset_index().merge(team_matches, on="team_name")
    g["per_match"] = g["n"] / g["matches"]
    hbar(g, "per_match", "n", os.path.join(out_dir, "05_central_volume_leaderboard.png"),
        "Central Defending Volume", "Eredivisie 2025/26 · Season · raw central-zone defensive actions per match (not value-weighted)",
        "Data via Opta | contrast with 01/central_defending_leaderboard.png -- volume alone, no quality weighting",
        value_fmt="{:.1f}")


# 6. Individual player central-defending leaderboard
def viz_central_player_leaderboard(values, out_dir):
    central = values[values["is_central"]]
    g = central.groupby(["player_name", "team_name"]).agg(
        n=("value_x1000", "size"), total=("value_x1000", "sum")).reset_index()
    g = g[g["n"] >= MIN_ACTIONS].sort_values("total", ascending=False).head(15)
    g["label"] = g["player_name"] + "  ·  " + g["team_name"]
    hbar(g, "total", "n", os.path.join(out_dir, "06_central_player_leaderboard.png"),
        "Top Central Defenders", f"Eredivisie 2025/26 · Season · players with ≥{MIN_ACTIONS} central-zone actions, ranked by total value",
        "Data via Opta | central third of pitch width only", label_col="label")


# 7. Central quality vs volume scatter (teams)
def viz_central_quality_volume(values, team_matches, out_dir):
    central = values[values["is_central"]]
    g = central.groupby("team_name").agg(n=("value_x1000", "size"), mean=("value_x1000", "mean")).reset_index()
    fig, ax = plt.subplots(figsize=(11, 8.5))
    fig.patch.set_facecolor(ldv.BG)
    ax.set_facecolor(ldv.BG)
    vmax = g["mean"].max()
    colors = [ldv.GOLD_RAMP(0.3 + 0.6 * (v / vmax)) for v in g["mean"]]
    ax.scatter(g["n"], g["mean"], s=220, color=colors, edgecolors="#ffe9b8", linewidth=0.6, zorder=3)
    for _, r in g.iterrows():
        ax.annotate(r["team_name"], xy=(r["n"], r["mean"]), xytext=(6, 4), textcoords="offset points",
                   fontsize=8.3, color=ldv.TEXT_MAIN)
    ax.axvline(g["n"].median(), color=ldv.PITCH_LINE, lw=1)
    ax.axhline(g["mean"].median(), color=ldv.PITCH_LINE, lw=1)
    ax.set_xlabel("Central-zone actions (volume)", fontsize=10.5, color=ldv.TEXT_SUB)
    ax.set_ylabel("Mean value per central action (quality)", fontsize=10.5, color=ldv.TEXT_SUB)
    ax.tick_params(colors=ldv.TEXT_SUB, labelsize=9)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(ldv.PITCH_LINE)
    ax.grid(color=ldv.PITCH_LINE, linewidth=0.4, alpha=0.4)
    fig.text(0.5, 0.965, "Central Defending: Volume vs. Quality", fontsize=21, fontweight="bold", ha="center", color=ldv.TEXT_MAIN)
    fig.text(0.5, 0.93, "Eredivisie 2025/26 · Season · one dot per team, central third only", fontsize=10.5, ha="center", color=ldv.TEXT_SUB)
    fig.text(0.98, 0.015, "Marc Lamberts", fontsize=9.5, ha="right", color=ldv.TEXT_FOOT, style="italic")
    fig.subplots_adjust(left=0.10, right=0.96, top=0.88, bottom=0.09)
    add_logo(fig)
    out_path = os.path.join(out_dir, "07_central_quality_vs_volume.png")
    fig.savefig(out_path, dpi=200, facecolor=ldv.BG)
    plt.close(fig)
    print("Saved:", out_path)


# 8. Best central defender's own pitch map (leader from viz 6)
def viz_top_central_defender_pitch(values, out_dir):
    central = values[values["is_central"]]
    g = central.groupby(["player_name", "team_name"]).agg(
        n=("value_x1000", "size"), total=("value_x1000", "sum")).reset_index()
    g = g[g["n"] >= MIN_ACTIONS].sort_values("total", ascending=False)
    top_player = g.iloc[0]
    sub = central[central["player_name"] == top_player["player_name"]]

    fig = plt.figure(figsize=(9, 11.5))
    fig.patch.set_facecolor(ldv.BG)
    pitch = VerticalPitch(pitch_type="uefa", pitch_color=ldv.BG, line_color=ldv.PITCH_LINE, linewidth=1.1,
                          half=False, line_zorder=2)
    ax = fig.add_axes([0.06, 0.08, 0.88, 0.76])
    pitch.draw(ax=ax)
    vmax = max(sub["value_x1000"].max(), 1e-9)
    sizes = 120 + 900 * (sub["value_x1000"] / vmax)
    colors = [ldv.GOLD_RAMP(0.3 + 0.6 * (v / vmax)) for v in sub["value_x1000"]]
    pitch.scatter(sub["x_own"], sub["y_own"], ax=ax, s=sizes, color=colors, edgecolors=ldv.GOLD,
                 linewidth=0.8, alpha=0.9, zorder=3)
    fig.text(0.5, 0.965, top_player["player_name"], fontsize=25, fontweight="bold", ha="center", color=ldv.TEXT_MAIN)
    fig.text(0.5, 0.925, f"{top_player['team_name']} · Best Central Defender · Eredivisie 2025/26 · "
             f"{int(top_player['n'])} central actions, {top_player['total']:.1f} total value",
             fontsize=11, ha="center", color=ldv.TEXT_SUB)
    fig.text(0.98, 0.02, "Marc Lamberts", fontsize=9.5, ha="right", color=ldv.TEXT_FOOT, style="italic")
    fig.text(0.02, 0.02, "Data via Opta | pitch runs bottom = own goal, top = opponent goal", fontsize=7.5, color=ldv.TEXT_FOOT)
    add_logo(fig, width=0.09)
    out_path = os.path.join(out_dir, "08_top_central_defender_pitch.png")
    fig.savefig(out_path, dpi=200, facecolor=ldv.BG)
    plt.close(fig)
    print("Saved:", out_path)


# 9. Central value share -- how central-reliant is each team's defending
def viz_central_share(values, out_dir):
    g = values.groupby(["team_name", "is_central"])["value_x1000"].sum().unstack(fill_value=0)
    g.columns = ["wide", "central"]
    g["share"] = g["central"] / (g["central"] + g["wide"]).replace(0, np.nan) * 100
    g = g.reset_index()
    hbar(g, "share", "share", os.path.join(out_dir, "09_central_value_share.png"),
        "How Central-Reliant Is Each Team's Defending?",
        "Eredivisie 2025/26 · Season · % of total disruption value that comes from the central third",
        "Data via Opta | high % = defends mostly through the middle, low % = relies more on the flanks",
        value_fmt="{:.0f}%")


# 10. Fine-grained central-corridor heatmap, league-wide
def viz_central_corridor_heatmap(values, out_dir):
    central = values[values["is_central"]].copy()
    fig = plt.figure(figsize=(9, 12))
    fig.patch.set_facecolor(ldv.BG)
    pitch = VerticalPitch(pitch_type="uefa", pitch_color=ldv.BG, line_color=ldv.PITCH_LINE, linewidth=1.1,
                          half=False, line_zorder=2)
    ax = fig.add_axes([0.06, 0.08, 0.80, 0.78])
    pitch.draw(ax=ax)
    stats = pitch.bin_statistic(central["x_own"], central["y_own"], statistic="count", bins=(14, 6))
    hm = pitch.heatmap(stats, ax=ax, cmap=ldv.GOLD_RAMP, edgecolors=ldv.BG, linewidth=0.3, zorder=1)
    cax = fig.add_axes([0.88, 0.24, 0.02, 0.44])
    cb = fig.colorbar(hm, cax=cax)
    cb.set_label("Disruptions", color=ldv.TEXT_SUB, fontsize=8)
    cb.ax.yaxis.set_tick_params(color=ldv.TEXT_SUB, labelcolor=ldv.TEXT_SUB, labelsize=7.5)
    fig.text(0.46, 0.965, "The Central Corridor, League-Wide", fontsize=21, fontweight="bold", ha="center", color=ldv.TEXT_MAIN)
    fig.text(0.46, 0.925, f"Eredivisie 2025/26 · Season · {len(central)} disruptions inside the central third only",
             fontsize=10.5, ha="center", color=ldv.TEXT_SUB)
    fig.text(0.98, 0.015, "Marc Lamberts", fontsize=9.5, ha="right", color=ldv.TEXT_FOOT, style="italic")
    fig.text(0.02, 0.015, "Data via Opta | zoomed into the central third, finer bins than the full-pitch heatmap",
             fontsize=7.3, color=ldv.TEXT_FOOT)
    add_logo(fig, width=0.10)
    out_path = os.path.join(out_dir, "10_central_corridor_heatmap.png")
    fig.savefig(out_path, dpi=200, facecolor=ldv.BG)
    plt.close(fig)
    print("Saved:", out_path)


def main():
    values, team_matches = prep()

    for theme in ("dark", "light"):
        ldv.set_theme(theme)
        out_dir = ldv.visual_dir(theme)

        viz_wide_leaderboard(values, team_matches, out_dir)
        viz_central_vs_wide(values, team_matches, out_dir)
        viz_central_by_action_type(values, out_dir)
        viz_central_pitch_grid(values, team_matches, out_dir)
        viz_central_volume(values, team_matches, out_dir)
        viz_central_player_leaderboard(values, out_dir)
        viz_central_quality_volume(values, team_matches, out_dir)
        viz_top_central_defender_pitch(values, out_dir)
        viz_central_share(values, out_dir)
        viz_central_corridor_heatmap(values, out_dir)


if __name__ == "__main__":
    main()

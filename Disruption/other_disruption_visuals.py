"""
Ten more disruption visuals, different angles from the central-zone set.
Eredivisie 2025/26.

1. First half vs second half, league-wide
2. Home vs away disruption value, per team
3. League-wide weekly trend of total disruption value
4. Pass difficulty distribution, league-wide (baseline completion model)
5. Top 5 teams, multi-axis radar
6. Disruptor vs disrupted quadrant (value dealt vs value conceded)
7. Value denied by pitch third (defensive/mid/attacking, length-wise)
8. Distribution of disruption value per team (box plot)
9. Hall of fame: 10 biggest single disruptions of the season
10. Cumulative disruption value race, top 6 teams over the season

Usage: python3 other_disruption_visuals.py [out_dir]
"""
import glob
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import build_disruption_model as bdm
import league_disruption_visuals as ldv
from league_disruption_visuals import add_logo

OUT_DIR = os.path.dirname(os.path.abspath(__file__))


def load_values():
    values = pd.read_csv(os.path.join(ldv.CSV_DIR, "all_eredivisie_disruption_values.csv"))
    parsed = values["match_file"].str.extract(r"(?P<date>\d{4}-\d{2}-\d{2})_(?P<home>.+) - (?P<away>.+)")
    values["date"] = pd.to_datetime(parsed["date"])
    values["home_team"] = parsed["home"]
    values["away_team"] = parsed["away"]
    values["is_home"] = values["team_name"] == values["home_team"]
    values["value_x1000"] = values["disruption_value"] * 1000
    return values


def footer(fig, text):
    fig.text(0.98, 0.012, "Marc Lamberts", fontsize=9.5, ha="right", color=ldv.TEXT_FOOT, style="italic")
    fig.text(0.02, 0.012, text, fontsize=7.3, color=ldv.TEXT_FOOT)


# 1. First half vs second half, league-wide
def viz_half_split(values, out_dir):
    g = values.groupby("period_id")["value_x1000"].agg(["size", "sum", "mean"])
    labels = ["1st half", "2nd half"] + (["Extra time"] if len(g) > 2 else [])
    fig, axes = plt.subplots(1, 2, figsize=(11, 6))
    fig.patch.set_facecolor(ldv.BG)
    for ax in axes:
        ax.set_facecolor(ldv.BG)
    totals = g["sum"].reindex([1, 2]).fillna(0)
    means = g["mean"].reindex([1, 2]).fillna(0)
    colors = [ldv.GOLD, ldv.BLUE]
    axes[0].bar(["1st half", "2nd half"], totals, color=colors, width=0.55, zorder=3)
    axes[0].set_title("Total value denied", color=ldv.TEXT_MAIN, fontsize=12)
    axes[1].bar(["1st half", "2nd half"], means, color=colors, width=0.55, zorder=3)
    axes[1].set_title("Mean value per action", color=ldv.TEXT_MAIN, fontsize=12)
    for ax in axes:
        ax.tick_params(colors=ldv.TEXT_SUB, labelsize=10)
        for label in ax.get_xticklabels():
            label.set_color(ldv.TEXT_MAIN)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color(ldv.PITCH_LINE)
        ax.grid(axis="y", color=ldv.PITCH_LINE, linewidth=0.5, alpha=0.4, zorder=0)
    fig.text(0.5, 0.965, "Does Defending Get Sharper After Half-Time?", fontsize=18, fontweight="bold",
             ha="center", color=ldv.TEXT_MAIN)
    fig.text(0.5, 0.925, "Eredivisie 2025/26 · Season · disruption value by half, league-wide",
             fontsize=10, ha="center", color=ldv.TEXT_SUB)
    footer(fig, "Data via Opta | period_id 1 = first half, 2 = second half")
    fig.subplots_adjust(left=0.08, right=0.96, top=0.84, bottom=0.10, wspace=0.28)
    add_logo(fig, width=0.09)
    out_path = os.path.join(out_dir, "b01_half_split.png")
    fig.savefig(out_path, dpi=200, facecolor=ldv.BG)
    plt.close(fig)
    print("Saved:", out_path)


# 2. Home vs away disruption value, per team
def viz_home_away(values, out_dir):
    g = values.groupby(["team_name", "is_home"])["value_x1000"].sum().unstack(fill_value=0)
    g.columns = ["away", "home"]
    g = g.reset_index()
    g["diff"] = g["home"] - g["away"]
    g = g.sort_values("diff", ascending=True)

    fig, ax = plt.subplots(figsize=(12.5, 9.5))
    fig.patch.set_facecolor(ldv.BG)
    ax.set_facecolor(ldv.BG)
    y = np.arange(len(g))
    h = 0.36
    ax.barh(y + h / 2, g["home"], height=h, color=ldv.GOLD, label="Home", zorder=3)
    ax.barh(y - h / 2, g["away"], height=h, color=ldv.BLUE, label="Away", zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(g["team_name"], fontsize=9.8, color=ldv.TEXT_MAIN)
    ax.tick_params(axis="x", colors=ldv.TEXT_SUB, labelsize=9)
    ax.tick_params(axis="y", length=0)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(ldv.PITCH_LINE)
    ax.grid(axis="x", color=ldv.PITCH_LINE, linewidth=0.5, alpha=0.4, zorder=0)
    ax.legend(loc="lower right", frameon=False, fontsize=9.5, labelcolor=ldv.LEGEND_TEXT)
    fig.text(0.5, 0.965, "Home vs. Away Disruption Value", fontsize=22, fontweight="bold", ha="center", color=ldv.TEXT_MAIN)
    fig.text(0.5, 0.935, "Eredivisie 2025/26 · Season · total disruption value denied, home and away matches",
             fontsize=10.5, ha="center", color=ldv.TEXT_SUB)
    footer(fig, "Data via Opta | home/away from each match's fixture listing")
    fig.subplots_adjust(left=0.19, right=0.96, top=0.90, bottom=0.07)
    add_logo(fig)
    out_path = os.path.join(out_dir, "b02_home_away.png")
    fig.savefig(out_path, dpi=200, facecolor=ldv.BG)
    plt.close(fig)
    print("Saved:", out_path)


# 3. League-wide weekly trend
def viz_weekly_trend(values, out_dir):
    weekly = values.set_index("date").resample("W")["value_x1000"].sum()
    weekly = weekly[weekly > 0]

    fig, ax = plt.subplots(figsize=(13, 6.5))
    fig.patch.set_facecolor(ldv.BG)
    ax.set_facecolor(ldv.BG)
    ax.plot(weekly.index, weekly.values, color=ldv.GOLD, lw=2, marker="o", markersize=4, zorder=3)
    ax.fill_between(weekly.index, weekly.values, color=ldv.GOLD, alpha=0.12, zorder=2)
    ax.tick_params(colors=ldv.TEXT_SUB, labelsize=9)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(ldv.PITCH_LINE)
    ax.grid(axis="y", color=ldv.PITCH_LINE, linewidth=0.5, alpha=0.4, zorder=0)
    ax.set_ylabel("Total disruption value, x1000", fontsize=10, color=ldv.TEXT_SUB)
    fig.text(0.5, 0.965, "Disruption Value Over the Season", fontsize=22, fontweight="bold", ha="center", color=ldv.TEXT_MAIN)
    fig.text(0.5, 0.93, "Eredivisie 2025/26 · league-wide total, weekly", fontsize=10.5, ha="center", color=ldv.TEXT_SUB)
    footer(fig, "Data via Opta | weekly sum of disruption_value across all matches played that week")
    fig.subplots_adjust(left=0.07, right=0.96, top=0.87, bottom=0.10)
    add_logo(fig, width=0.08)
    out_path = os.path.join(out_dir, "b03_weekly_trend.png")
    fig.savefig(out_path, dpi=200, facecolor=ldv.BG)
    plt.close(fig)
    print("Saved:", out_path)


# 4. Pass difficulty distribution, league-wide
def build_pass_difficulty_df():
    import joblib
    files = sorted(glob.glob(os.path.join(bdm.DATA_DIR, "*.json")))
    cid_to_team = bdm.build_global_cid_to_team(files)
    rows = []
    for fn in files:
        basename, events, team_map = bdm.load_match(fn, cid_to_team)
        rows.extend(bdm.extract_passes(basename, events, team_map))
    pass_df = pd.DataFrame(rows)
    model = joblib.load(os.path.join(bdm.MODEL_DIR, "model_pass_completion.pkl"))
    pass_df["pred"] = model.predict_proba(pass_df[bdm.FEATURE_COLS])[:, 1]
    pass_df["difficulty"] = 1 - pass_df["pred"]
    return pass_df


def viz_difficulty_distribution(out_dir, pass_df=None):
    if pass_df is None:
        pass_df = build_pass_difficulty_df()

    fig, ax = plt.subplots(figsize=(11, 6.5))
    fig.patch.set_facecolor(ldv.BG)
    ax.set_facecolor(ldv.BG)
    ax.hist(pass_df["difficulty"], bins=40, color=ldv.GOLD, alpha=0.85, zorder=3)
    ax.axvline(pass_df["difficulty"].median(), color=ldv.TEXT_MAIN, lw=1.2, linestyle="--", zorder=4)
    ax.text(pass_df["difficulty"].median() + 0.01, ax.get_ylim()[1] * 0.9,
           f"median {pass_df['difficulty'].median():.2f}", color=ldv.TEXT_MAIN, fontsize=9)
    ax.tick_params(colors=ldv.TEXT_SUB, labelsize=9)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(ldv.PITCH_LINE)
    ax.grid(axis="y", color=ldv.PITCH_LINE, linewidth=0.5, alpha=0.4, zorder=0)
    ax.set_xlabel("Pass difficulty (1 - predicted completion probability)", fontsize=10, color=ldv.TEXT_SUB)
    ax.set_ylabel("Passes", fontsize=10, color=ldv.TEXT_SUB)
    fig.text(0.5, 0.965, "How Hard Is the Average Pass?", fontsize=22, fontweight="bold", ha="center", color=ldv.TEXT_MAIN)
    fig.text(0.5, 0.93, f"Eredivisie 2025/26 · Season · {len(pass_df)} passes, baseline completion model",
             fontsize=10.5, ha="center", color=ldv.TEXT_SUB)
    footer(fig, "Data via Opta | most passes are easy (short, safe circulation); the long right tail is what disruption targets")
    fig.subplots_adjust(left=0.08, right=0.96, top=0.87, bottom=0.11)
    add_logo(fig, width=0.08)
    out_path = os.path.join(out_dir, "b04_difficulty_distribution.png")
    fig.savefig(out_path, dpi=200, facecolor=ldv.BG)
    plt.close(fig)
    print("Saved:", out_path)


# 5. Top 5 teams, multi-axis radar
def viz_team_radar(values, out_dir, directions=None):
    team_matches = pd.read_csv(os.path.join(ldv.CSV_DIR, "disruption_value_team_summary.csv"))
    central_low, central_high = 68 / 3, 2 * 68 / 3
    if directions is None:
        from league_disruption_visuals import compute_attack_directions
        directions = compute_attack_directions()
    dir_vals = np.array([
        directions.get((mf, cid, p), 1)
        for mf, cid, p in zip(values["match_file"], values["contestant_id"], values["period_id"])
    ])
    values = values.copy()
    values["y_own"] = np.where(dir_vals == 1, values["y"], 68.0 - values["y"])
    values["is_central"] = (values["y_own"] >= central_low) & (values["y_own"] <= central_high)

    per_team = values.groupby("team_name").agg(
        total=("value_x1000", "sum"), n=("value_x1000", "size"), mean=("value_x1000", "mean"))
    central_pm = values[values["is_central"]].groupby("team_name").size()
    per_team = per_team.join(team_matches.set_index("team_name")["matches"])
    per_team["total_pm"] = per_team["total"] / per_team["matches"]
    per_team["actions_pm"] = per_team["n"] / per_team["matches"]
    per_team["central_pm"] = central_pm.reindex(per_team.index).fillna(0) / per_team["matches"]

    metrics = ["total_pm", "actions_pm", "mean", "central_pm"]
    labels = ["Value/match", "Actions/match", "Value/action", "Central actions/match"]
    norm = (per_team[metrics] - per_team[metrics].min()) / (per_team[metrics].max() - per_team[metrics].min())

    top_teams = per_team.sort_values("total_pm", ascending=False).head(5).index.tolist()
    colors = [ldv.GOLD, ldv.BLUE, ldv.GREEN, ldv.RED, "#c084fc"]

    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]

    fig = plt.figure(figsize=(11, 9.5))
    fig.patch.set_facecolor(ldv.BG)
    ax = fig.add_axes([0.09, 0.08, 0.62, 0.78], polar=True)
    ax.set_facecolor(ldv.BG)
    for team, color in zip(top_teams, colors):
        vals = norm.loc[team, metrics].tolist()
        vals += vals[:1]
        ax.plot(angles, vals, color=color, linewidth=2, label=team, zorder=3)
        ax.fill(angles, vals, color=color, alpha=0.06, zorder=2)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=9.5, color=ldv.TEXT_MAIN)
    ax.tick_params(axis="x", pad=14)
    ax.set_yticklabels([])
    ax.spines["polar"].set_color(ldv.PITCH_LINE)
    ax.grid(color=ldv.PITCH_LINE, alpha=0.6)
    ax.legend(loc="center left", bbox_to_anchor=(1.22, 0.5), frameon=False, fontsize=10, labelcolor=ldv.TEXT_MAIN)
    fig.text(0.5, 0.97, "Top 5 Teams: Defensive Profile", fontsize=18, fontweight="bold", ha="center", color=ldv.TEXT_MAIN)
    fig.text(0.5, 0.935, "Eredivisie 2025/26 · min-max normalized across all 18 teams", fontsize=9.5,
             ha="center", color=ldv.TEXT_SUB)
    footer(fig, "Data via Opta | axes independently normalized 0-1 across the full league")
    out_path = os.path.join(out_dir, "b05_team_radar.png")
    fig.savefig(out_path, dpi=200, facecolor=ldv.BG)
    plt.close(fig)
    print("Saved:", out_path)


# 6. Disruptor vs disrupted quadrant
def viz_disruptor_vs_disrupted(values, out_dir):
    dealt = values.groupby("team_name")["value_x1000"].sum().rename("dealt")
    conceded = values.groupby("linked_pass_team")["value_x1000"].sum().rename("conceded")
    g = pd.concat([dealt, conceded], axis=1).fillna(0)
    matches = pd.read_csv(os.path.join(ldv.CSV_DIR, "disruption_value_team_summary.csv")).set_index("team_name")["matches"]
    g = g.join(matches)
    g["dealt_pm"] = g["dealt"] / g["matches"]
    g["conceded_pm"] = g["conceded"] / g["matches"]
    g = g.reset_index().rename(columns={"index": "team_name"})

    fig, ax = plt.subplots(figsize=(11, 9.5))
    fig.patch.set_facecolor(ldv.BG)
    ax.set_facecolor(ldv.BG)
    ax.scatter(g["conceded_pm"], g["dealt_pm"], s=180, color=ldv.GOLD, edgecolors="#ffe9b8",
              linewidth=0.6, zorder=3)
    x_cutoff = g["conceded_pm"].min() + 0.85 * (g["conceded_pm"].max() - g["conceded_pm"].min())
    for _, r in g.iterrows():
        near_right_edge = r["conceded_pm"] >= x_cutoff
        xytext = (-6, 4) if near_right_edge else (6, 4)
        ax.annotate(r["team_name"], xy=(r["conceded_pm"], r["dealt_pm"]), xytext=xytext,
                   textcoords="offset points", fontsize=8.3, color=ldv.TEXT_MAIN,
                   ha="right" if near_right_edge else "left")
    xmed, ymed = g["conceded_pm"].median(), g["dealt_pm"].median()
    ax.axvline(xmed, color=ldv.PITCH_LINE, lw=1)
    ax.axhline(ymed, color=ldv.PITCH_LINE, lw=1)
    ax.text(ax.get_xlim()[1] * 0.98, ax.get_ylim()[1] * 0.98, "SOLID BOTH WAYS", ha="right", va="top",
           fontsize=9, color=ldv.TEXT_SUB, fontweight="bold")
    ax.text(ax.get_xlim()[0] * 1.0 + 0.02, ax.get_ylim()[1] * 0.98, "VULNERABLE, BUT DISRUPTS", ha="left",
           va="top", fontsize=9, color=ldv.TEXT_SUB, fontweight="bold")
    ax.margins(x=0.10, y=0.08)
    ax.set_xlabel("Value conceded per match (passes disrupted BY opponents)", fontsize=10, color=ldv.TEXT_SUB)
    ax.set_ylabel("Value dealt per match (passes disrupted OF opponents)", fontsize=10, color=ldv.TEXT_SUB)
    ax.tick_params(colors=ldv.TEXT_SUB, labelsize=9)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(ldv.PITCH_LINE)
    ax.grid(color=ldv.PITCH_LINE, linewidth=0.4, alpha=0.4)
    fig.text(0.5, 0.965, "Disruptor vs. Disrupted", fontsize=22, fontweight="bold", ha="center", color=ldv.TEXT_MAIN)
    fig.text(0.5, 0.93, "Eredivisie 2025/26 · Season · value dealt vs. value conceded, per match",
             fontsize=10.5, ha="center", color=ldv.TEXT_SUB)
    footer(fig, "Data via Opta | top-left = disrupts opponents well AND rarely gets disrupted itself")
    fig.subplots_adjust(left=0.09, right=0.96, top=0.88, bottom=0.08)
    add_logo(fig)
    out_path = os.path.join(out_dir, "b06_disruptor_vs_disrupted.png")
    fig.savefig(out_path, dpi=200, facecolor=ldv.BG)
    plt.close(fig)
    print("Saved:", out_path)


# 7. Value denied by pitch third (length-wise)
def viz_by_third(values, out_dir, directions=None):
    if directions is None:
        from league_disruption_visuals import compute_attack_directions
        directions = compute_attack_directions()
    dir_vals = np.array([
        directions.get((mf, cid, p), 1)
        for mf, cid, p in zip(values["match_file"], values["contestant_id"], values["period_id"])
    ])
    values = values.copy()
    values["x_own"] = np.where(dir_vals == 1, values["x"], bdm.GOAL_X - values["x"])
    values["third"] = pd.cut(values["x_own"], bins=[-0.1, 35, 70, 105.1],
                             labels=["Defensive third", "Middle third", "Attacking third"])
    matches = pd.read_csv(os.path.join(ldv.CSV_DIR, "disruption_value_team_summary.csv")).set_index("team_name")["matches"]
    g = values.groupby(["team_name", "third"], observed=True)["value_x1000"].sum().unstack(fill_value=0)
    g = g.div(matches, axis=0).dropna()
    g = g.sort_values("Defensive third", ascending=True)

    fig, ax = plt.subplots(figsize=(13, 9.5))
    fig.patch.set_facecolor(ldv.BG)
    ax.set_facecolor(ldv.BG)
    left = np.zeros(len(g))
    colors = {"Defensive third": ldv.BLUE, "Middle third": ldv.GOLD, "Attacking third": ldv.GREEN}
    for col in ["Defensive third", "Middle third", "Attacking third"]:
        ax.barh(g.index, g[col], left=left, color=colors[col], label=col, zorder=3)
        left += g[col].values
    ax.tick_params(axis="y", colors=ldv.TEXT_MAIN, labelsize=9.8, length=0)
    ax.tick_params(axis="x", colors=ldv.TEXT_SUB, labelsize=9)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(ldv.PITCH_LINE)
    ax.grid(axis="x", color=ldv.PITCH_LINE, linewidth=0.5, alpha=0.4, zorder=0)
    ax.legend(loc="lower right", frameon=False, fontsize=9.5, labelcolor=ldv.LEGEND_TEXT)
    fig.text(0.5, 0.965, "Where Down the Pitch Does Each Team Defend?", fontsize=20, fontweight="bold",
             ha="center", color=ldv.TEXT_MAIN)
    fig.text(0.5, 0.935, "Eredivisie 2025/26 · Season · disruption value denied per match, by third of pitch length",
             fontsize=10, ha="center", color=ldv.TEXT_SUB)
    footer(fig, "Data via Opta | thirds normalized per team's own attacking direction")
    fig.subplots_adjust(left=0.19, right=0.96, top=0.90, bottom=0.07)
    add_logo(fig)
    out_path = os.path.join(out_dir, "b07_value_by_third.png")
    fig.savefig(out_path, dpi=200, facecolor=ldv.BG)
    plt.close(fig)
    print("Saved:", out_path)


# 8. Distribution of disruption value per team (box plot)
def viz_value_boxplot(values, out_dir):
    order = values.groupby("team_name")["value_x1000"].mean().sort_values(ascending=False).index.tolist()
    data = [values[values["team_name"] == t]["value_x1000"].values for t in order]

    fig, ax = plt.subplots(figsize=(14, 7.5))
    fig.patch.set_facecolor(ldv.BG)
    ax.set_facecolor(ldv.BG)
    bp = ax.boxplot(data, vert=True, patch_artist=True, showfliers=True,
                    flierprops=dict(marker="o", markersize=3, markerfacecolor=ldv.GOLD,
                                    markeredgecolor="none", alpha=0.5),
                    medianprops=dict(color=ldv.TEXT_MAIN, linewidth=1.5),
                    whiskerprops=dict(color=ldv.TEXT_SUB), capprops=dict(color=ldv.TEXT_SUB))
    for patch in bp["boxes"]:
        patch.set_facecolor(ldv.GOLD)
        patch.set_alpha(0.55)
        patch.set_edgecolor(ldv.GOLD)
    ax.set_xticks(range(1, len(order) + 1))
    ax.set_xticklabels(order, rotation=55, ha="right", fontsize=7.6, color=ldv.TEXT_MAIN)
    ax.tick_params(axis="y", colors=ldv.TEXT_SUB, labelsize=9)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(ldv.PITCH_LINE)
    ax.grid(axis="y", color=ldv.PITCH_LINE, linewidth=0.5, alpha=0.4, zorder=0)
    ax.set_ylabel("Disruption value per action, x1000", fontsize=10, color=ldv.TEXT_SUB)
    fig.text(0.5, 0.965, "Spread of Disruption Value, Every Team", fontsize=20, fontweight="bold",
             ha="center", color=ldv.TEXT_MAIN)
    fig.text(0.5, 0.93, "Eredivisie 2025/26 · Season · distribution of individual action values, not just the mean",
             fontsize=10, ha="center", color=ldv.TEXT_SUB)
    footer(fig, "Data via Opta | a high median with a long tail means a team occasionally makes a huge denial, not just steady ones")
    fig.subplots_adjust(left=0.06, right=0.97, top=0.87, bottom=0.24)
    add_logo(fig, width=0.07)
    out_path = os.path.join(out_dir, "b08_value_boxplot.png")
    fig.savefig(out_path, dpi=200, facecolor=ldv.BG)
    plt.close(fig)
    print("Saved:", out_path)


# 9. Hall of fame: 10 biggest single disruptions
def viz_hall_of_fame(values, out_dir):
    top = values.sort_values("value_x1000", ascending=False).head(10).iloc[::-1]
    labels = [f"{p}  ({t})  vs {opp}" for p, t, opp in
             zip(top["player_name"], top["team_name"], top["linked_pass_team"])]

    fig, ax = plt.subplots(figsize=(13, 8.5))
    fig.patch.set_facecolor(ldv.BG)
    ax.set_facecolor(ldv.BG)
    vmax = top["value_x1000"].max()
    colors = [ldv.GOLD_RAMP(0.3 + 0.6 * (v / vmax)) for v in top["value_x1000"]]
    ax.barh(range(len(top)), top["value_x1000"], color=colors, height=0.62, zorder=3)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(labels, fontsize=9.5, color=ldv.TEXT_MAIN)
    for i, val in enumerate(top["value_x1000"]):
        ax.text(val + vmax * 0.012, i, f"{val:.2f}", va="center", fontsize=9.5, color=ldv.LEGEND_TEXT)
    ax.tick_params(axis="x", colors=ldv.TEXT_SUB, labelsize=9)
    ax.tick_params(axis="y", length=0)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(ldv.PITCH_LINE)
    ax.grid(axis="x", color=ldv.PITCH_LINE, linewidth=0.5, alpha=0.4, zorder=0)
    ax.set_xlim(0, vmax * 1.15)
    fig.text(0.5, 0.965, "Hall of Fame: Biggest Single Denials", fontsize=21, fontweight="bold",
             ha="center", color=ldv.TEXT_MAIN)
    fig.text(0.5, 0.93, "Eredivisie 2025/26 · Season · the 10 highest-value individual disruptions",
             fontsize=10.5, ha="center", color=ldv.TEXT_SUB)
    footer(fig, "Data via Opta | value = P(pass completes) x xT it would have added")
    fig.subplots_adjust(left=0.34, right=0.95, top=0.89, bottom=0.07)
    add_logo(fig)
    out_path = os.path.join(out_dir, "b09_hall_of_fame.png")
    fig.savefig(out_path, dpi=200, facecolor=ldv.BG)
    plt.close(fig)
    print("Saved:", out_path)


# 10. Cumulative disruption value race, top 6 teams
def viz_cumulative_race(values, out_dir):
    top6 = values.groupby("team_name")["value_x1000"].sum().sort_values(ascending=False).head(6).index.tolist()
    colors = [ldv.GOLD, ldv.BLUE, ldv.GREEN, ldv.RED, "#c084fc", "#5eead4"]

    fig, ax = plt.subplots(figsize=(13, 7.5))
    fig.patch.set_facecolor(ldv.BG)
    ax.set_facecolor(ldv.BG)
    for team, color in zip(top6, colors):
        sub = values[values["team_name"] == team].sort_values("date")
        daily = sub.groupby("date")["value_x1000"].sum().cumsum()
        ax.plot(daily.index, daily.values, color=color, lw=2.2, label=team, zorder=3)
        ax.scatter([daily.index[-1]], [daily.values[-1]], color=color, s=30, zorder=4)
    ax.tick_params(colors=ldv.TEXT_SUB, labelsize=9)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(ldv.PITCH_LINE)
    ax.grid(axis="y", color=ldv.PITCH_LINE, linewidth=0.5, alpha=0.4, zorder=0)
    ax.set_ylabel("Cumulative disruption value, x1000", fontsize=10, color=ldv.TEXT_SUB)
    ax.legend(loc="upper left", frameon=False, fontsize=9.5, labelcolor=ldv.TEXT_MAIN)
    fig.text(0.5, 0.965, "Season-Long Disruption Race", fontsize=22, fontweight="bold", ha="center", color=ldv.TEXT_MAIN)
    fig.text(0.5, 0.93, "Eredivisie 2025/26 · top 6 teams by total disruption value, running total",
             fontsize=10.5, ha="center", color=ldv.TEXT_SUB)
    footer(fig, "Data via Opta | cumulative sum of disruption_value by match date")
    fig.subplots_adjust(left=0.07, right=0.96, top=0.87, bottom=0.10)
    add_logo(fig, width=0.08)
    out_path = os.path.join(out_dir, "b10_cumulative_race.png")
    fig.savefig(out_path, dpi=200, facecolor=ldv.BG)
    plt.close(fig)
    print("Saved:", out_path)


def main():
    values = load_values()
    from league_disruption_visuals import compute_attack_directions
    directions = compute_attack_directions()
    pass_df = build_pass_difficulty_df()

    for theme in ("dark", "light"):
        ldv.set_theme(theme)
        out_dir = ldv.visual_dir(theme)

        viz_half_split(values, out_dir)
        viz_home_away(values, out_dir)
        viz_weekly_trend(values, out_dir)
        viz_difficulty_distribution(out_dir, pass_df=pass_df)
        viz_team_radar(values, out_dir, directions=directions)
        viz_disruptor_vs_disrupted(values, out_dir)
        viz_by_third(values, out_dir, directions=directions)
        viz_value_boxplot(values, out_dir)
        viz_hall_of_fame(values, out_dir)
        viz_cumulative_race(values, out_dir)


if __name__ == "__main__":
    main()

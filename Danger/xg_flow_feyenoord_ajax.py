"""
xG flow map: cumulative xG over match time for Feyenoord vs. Ajax
(2026-03-22, Danger/all_eredivisie_danger_models.csv, one match_file).
Step chart, one line per team, goals marked and labelled.

Built in the Meridian house style (housestyle package).

Usage: python3 Danger/xg_flow_feyenoord_ajax.py [out_dir]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import matplotlib.pyplot as plt

from housestyle import style, components

REPO_ROOT = Path(__file__).resolve().parent.parent
DANGER_CSV = REPO_ROOT / "Danger" / "all_eredivisie_danger_models.csv"
MATCH_FILE = "2026-03-22_Feyenoord Rotterdam - AFC Ajax.json"

TEAMS = [("Feyenoord Rotterdam", "20vymiy7bo8wkyxai3ew494fz"),
         ("AFC Ajax", "d0zdg647gvgc95xdtk1vpbkys")]


def cumulative_series(shots, max_minute):
    shots = shots.sort_values("time_min")
    xs, ys = [0], [0]
    running = 0.0
    for _, r in shots.iterrows():
        xs.append(r["time_min"])
        ys.append(running)
        running += r["xg"]
        xs.append(r["time_min"])
        ys.append(running)
    xs.append(max_minute)
    ys.append(running)
    return xs, ys


def main():
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent

    df = pd.read_csv(DANGER_CSV)
    match = df[df["match_file"] == MATCH_FILE].copy()
    max_minute = max(96, match["time_min"].max() + 2)

    palette, cats = style.apply("light")
    ink_blue, teal = cats[0], cats[2]
    colors = {TEAMS[0][1]: ink_blue, TEAMS[1][1]: teal}
    short = {"Feyenoord Rotterdam": "Feyenoord", "AFC Ajax": "Ajax"}

    fig = plt.figure(figsize=(11, 7.6))
    fig.patch.set_facecolor(palette["surface"])
    ax = fig.add_axes([0.08, 0.15, 0.87, 0.60])

    totals = {}
    for team_name, cid in TEAMS:
        shots = match[match["contestant_id"] == cid]
        xs, ys = cumulative_series(shots, max_minute)
        color = colors[cid]
        ax.step(xs, ys, where="post", color=color, linewidth=2.4, zorder=3,
                 label=f"{short[team_name]} ({shots['xg'].sum():.2f} xG)")
        totals[team_name] = shots["xg"].sum()

        goals = shots[shots["is_goal"] == 1].sort_values("time_min")
        for _, g in goals.iterrows():
            cum_at_goal = shots[shots["time_min"] <= g["time_min"]]["xg"].sum()
            ax.scatter([g["time_min"]], [cum_at_goal], s=110, color=color,
                       edgecolors=palette["surface"], linewidths=1.4, zorder=5)
            ax.annotate(f"{g['player_name']} {int(g['time_min'])}'",
                        xy=(g["time_min"], cum_at_goal), xytext=(6, 8),
                        textcoords="offset points", fontsize=9, fontweight="bold",
                        color=palette["ink_primary"])

    ax.axvline(45, color=palette["axis"], linewidth=0.8, linestyle=(0, (3, 3)), zorder=1)

    ax.set_xlim(0, max_minute)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Minute", fontsize=10.5, color=palette["ink_secondary"])
    ax.set_ylabel("Cumulative xG", fontsize=10.5, color=palette["ink_secondary"])
    ax.grid(True, axis="y", color=palette["grid"], linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", frameon=False, fontsize=10, labelcolor=palette["ink_secondary"])

    fey_goals = int(match[(match["contestant_id"] == TEAMS[0][1]) & (match["is_goal"] == 1)].shape[0])
    aja_goals = int(match[(match["contestant_id"] == TEAMS[1][1]) & (match["is_goal"] == 1)].shape[0])

    fey_xg, aja_xg = totals[TEAMS[0][0]], totals[TEAMS[1][0]]
    leader = short[TEAMS[0][0]] if fey_xg > aja_xg else short[TEAMS[1][0]]
    if fey_goals == aja_goals:
        title = f"{leader} dominated xG, but the game still finished {fey_goals}-{aja_goals}"
    else:
        title = f"{leader} led on xG as Feyenoord and Ajax finished {fey_goals}-{aja_goals}"

    components.header(
        fig,
        kicker="Match report · xG flow",
        title=title,
        dek=f"Cumulative xG by minute  ·  Feyenoord {totals[TEAMS[0][0]]:.2f} xG, "
            f"Ajax {totals[TEAMS[1][0]]:.2f} xG  ·  Eredivisie 2025/26, 22 Mar 2026",
        palette=palette,
    )
    components.footer(
        fig,
        source="Opta/StatsPerform, Eredivisie 2025/26",
        palette=palette,
    )

    out_path = out_dir / "xg_flow_feyenoord_ajax.png"
    fig.savefig(out_path, dpi=200, facecolor=palette["surface"])
    plt.close(fig)
    print(f"Feyenoord {totals[TEAMS[0][0]]:.2f} xG, Ajax {totals[TEAMS[1][0]]:.2f} xG, "
          f"final {fey_goals}-{aja_goals}")
    print("Saved:", out_path)


if __name__ == "__main__":
    main()

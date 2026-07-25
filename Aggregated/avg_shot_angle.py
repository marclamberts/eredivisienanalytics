"""
League-wide average shot angle, Eredivisie, 2022-2023 through 2025-2026.

"Shot angle" here is the standard xG-model definition: the angle subtended
by the goal (the two posts) as seen from the shot location -- a shot taken
square-on from the penalty spot sees a wide angle; one from a tight
touchline cutback sees a narrow one, even at a similar distance. Not
qualifier 213 ("angle (rad)", the pass/shot's own direction of travel --
a different thing).

No direction normalisation applied. Checked directly before writing this:
every team's own goal kicks, in every period, across all four seasons,
average x in the 1-6 range (out of 1236+ team-period buckets checked, zero
exceptions) -- i.e. this feed already always shows a team attacking toward
x=100, in both halves, with no flip at half-time. That contradicts an
assumption made earlier in this session (that raw coordinates are a single
shared pitch frame needing a per-half flip, used by team_directions() in
wing_play_comparison.py / diagonal_vs_relationism.py / restart_analysis.py
/ ajax_coach_style*.py) -- see this repo's chat history / commit messages
around the second-ball-to-shot pitch map for the full account. Shots here
are computed straight from raw (x, y) against a fixed goal at (105, 34),
exactly like build_season_aggregate.py's own dist_to_goal_m() -- which
turns out to have been correct all along, since it never flipped either.

Usage: python3 avg_shot_angle.py
"""
import csv
import glob
import json
import math
import os
import sys

import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(ROOT)
SEASONS = ["2022-2023", "2023-2024", "2024-2025", "2025-2026"]

sys.path.insert(0, REPO_ROOT)
from housestyle import style, components  # noqa: E402

X_SCALE, Y_SCALE = 1.05, 0.68
GOAL_X, GOAL_Y = 105.0, 34.0
GOAL_WIDTH_M = 7.32
POST1 = (GOAL_X, GOAL_Y - GOAL_WIDTH_M / 2)
POST2 = (GOAL_X, GOAL_Y + GOAL_WIDTH_M / 2)
SHOT_TYPES = {13, 14, 15, 16}


def to_m(x, y):
    return x * X_SCALE, y * Y_SCALE


def shot_angle_deg(x, y):
    xm, ym = to_m(x, y)
    v1 = (POST1[0] - xm, POST1[1] - ym)
    v2 = (POST2[0] - xm, POST2[1] - ym)
    dot = v1[0] * v2[0] + v1[1] * v2[1]
    n1 = math.hypot(*v1)
    n2 = math.hypot(*v2)
    if n1 == 0 or n2 == 0:
        return None
    cos_a = max(-1.0, min(1.0, dot / (n1 * n2)))
    return math.degrees(math.acos(cos_a))


def season_avg(season):
    total_angle, n_shots = 0.0, 0
    for path in sorted(glob.glob(os.path.join(REPO_ROOT, "Events", season, "*.json"))):
        raw = json.load(open(path, encoding="utf-8"))
        for e in raw.get("event", []):
            if e.get("typeId") not in SHOT_TYPES:
                continue
            x, y = e.get("x"), e.get("y")
            if x is None or y is None:
                continue
            angle = shot_angle_deg(x, y)
            if angle is None:
                continue
            total_angle += angle
            n_shots += 1
    return (total_angle / n_shots if n_shots else None), n_shots


def main():
    rows = []
    for season in SEASONS:
        avg_deg, n_shots = season_avg(season)
        rows.append({"season": season, "avg_shot_angle_deg": round(avg_deg, 3), "n_shots": n_shots})
        print(f"{season}: n_shots={n_shots}  avg_shot_angle_deg={avg_deg:.3f}")

    csv_path = os.path.join(ROOT, "avg_shot_angle.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["season", "avg_shot_angle_deg", "n_shots"])
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {csv_path}")

    build_chart(rows)


def build_chart(rows):
    palette, cats = style.apply("light")
    fig = plt.figure(figsize=(9.5, 6.4))
    ax = fig.add_axes([0.11, 0.16, 0.82, 0.56])

    seasons = [r["season"] for r in rows]
    vals = [r["avg_shot_angle_deg"] for r in rows]
    bars = ax.bar(seasons, vals, width=0.55)
    components.highlight_bars(bars, accent_index=len(vals) - 1, palette=palette)
    for i, v in enumerate(vals):
        ax.annotate(f"{v:.1f}°", xy=(i, v), xytext=(0, 6), textcoords="offset points",
                    ha="center", fontsize=10.5, fontweight="bold", color=palette["ink_primary"])
    ax.set_ylabel("Average shot angle (degrees)")
    ax.set_ylim(0, max(vals) * 1.3)

    first, last = vals[0], vals[-1]
    pct_change = (last - first) / first * 100
    if abs(pct_change) < 2:
        title = "Eredivisie average shot angle has barely moved in four seasons"
    else:
        direction = "wider" if last > first else "tighter"
        title = f"Eredivisie shots are coming from a {direction} angle than four seasons ago"

    components.header(
        fig, kicker="Shot Angle",
        title=title,
        dek=f"League-wide average shot angle (angle to goal at the moment of the shot), "
            f"{seasons[0]} to {seasons[-1]} ({pct_change:+.1f}%)",
        palette=palette,
    )
    components.footer(fig, source="Opta/StatsPerform 2022-2026", palette=palette)

    out_path = os.path.join(ROOT, "avg_shot_angle.png")
    fig.savefig(out_path, facecolor=fig.get_facecolor(), dpi=150)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()

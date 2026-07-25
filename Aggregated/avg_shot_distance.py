"""
League-wide average shot distance, Eredivisie, across every season with
event data in this repo.

Note on scope: the user asked for 2022-2023 through 2025-2026, but this
repo's Events/ only has 2023-2024, 2024-2025, and 2025-2026 -- there is no
Events/2022-2023 (or any other 2022-2023 file) anywhere in the repository.
Rather than guess or skip the request, this covers the three seasons that
do exist and says so on the chart itself.

Reuses the already-computed, already-verified per-player shot_dist_total_m
and shots columns from Aggregated/<season>/player_season_aggregated.csv
(dist_to_goal_m at the moment of the shot, typeId 13/14/15/16 -- Miss,
Post, Attempt Saved, Goal -- summed in build_season_aggregate.py) rather
than recomputing from raw events: summing both columns across every player
and dividing gives the league-wide average directly, weighted by each
player's actual shot volume (not a naive average of players' own averages,
which would overweight low-volume shooters).

Usage: python3 avg_shot_distance.py
"""
import csv
import os
import sys

import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(ROOT)
SEASONS = ["2023-2024", "2024-2025", "2025-2026"]
MISSING_SEASON = "2022-2023"

sys.path.insert(0, REPO_ROOT)
from housestyle import style, components  # noqa: E402


def season_avg(season):
    path = os.path.join(ROOT, season, "player_season_aggregated.csv")
    total_dist, total_shots = 0.0, 0
    with open(path, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            try:
                shots = int(float(row.get("shots") or 0))
                dist = float(row.get("shot_dist_total_m") or 0)
            except ValueError:
                continue
            total_shots += shots
            total_dist += dist
    return total_dist / total_shots if total_shots else None, total_shots


def main():
    rows = []
    for season in SEASONS:
        avg_m, n_shots = season_avg(season)
        rows.append({"season": season, "avg_shot_distance_m": round(avg_m, 3), "n_shots": n_shots})
        print(f"{season}: n_shots={n_shots}  avg_shot_distance_m={avg_m:.3f}")

    csv_path = os.path.join(ROOT, "avg_shot_distance.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["season", "avg_shot_distance_m", "n_shots"])
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {csv_path}")

    build_chart(rows)


def build_chart(rows):
    palette, cats = style.apply("light")
    fig = plt.figure(figsize=(8.5, 6.4))
    ax = fig.add_axes([0.12, 0.16, 0.80, 0.56])

    seasons = [r["season"] for r in rows]
    vals = [r["avg_shot_distance_m"] for r in rows]
    bars = ax.bar(seasons, vals, width=0.5)
    components.highlight_bars(bars, accent_index=len(vals) - 1, palette=palette)
    for i, v in enumerate(vals):
        ax.annotate(f"{v:.1f}m", xy=(i, v), xytext=(0, 6), textcoords="offset points",
                    ha="center", fontsize=10.5, fontweight="bold", color=palette["ink_primary"])
    ax.set_ylabel("Average shot distance (metres)")
    ax.set_ylim(0, max(vals) * 1.3)

    first, last = vals[0], vals[-1]
    pct_change = (last - first) / first * 100
    if abs(pct_change) < 2:
        title = "Eredivisie average shot distance has barely moved in three seasons"
    else:
        direction = "closer" if last < first else "further out"
        title = f"Eredivisie shots are being taken from {direction} than three seasons ago"

    components.header(
        fig, kicker="Shot Distance",
        title=title,
        dek=f"League-wide average shot distance, {seasons[0]} to {seasons[-1]} ({pct_change:+.1f}%)",
        palette=palette,
    )
    components.footer(
        fig, source="Opta/StatsPerform 2023-2026",
        note=f"{MISSING_SEASON} not shown, no data",
        palette=palette,
    )

    out_path = os.path.join(ROOT, "avg_shot_distance.png")
    fig.savefig(out_path, facecolor=fig.get_facecolor(), dpi=150)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()

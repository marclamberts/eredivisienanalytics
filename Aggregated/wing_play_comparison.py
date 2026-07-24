"""
Wing play comparison, per team: what share of a team's open-play passing
goes through the wide corridors (y<25 or y>75 on Opta's 0-100 pitch), split
left vs. right from the ATTACKING team's own perspective (correcting for
the second-half end swap -- see note below).

This is NOT a re-read of Analysis/Coach Profiling/team_metrics_aggregated.csv's
existing `wing_pct` column. That script's own qualifier constants (checked
while building this) are wrong in several places for this feed -- e.g. its
Q_GOAL_KICK=72 is actually "left foot", and its Q_END_X/Q_END_Y (141/140)
are swapped versus the verified 140=end_x/141=end_y used everywhere else in
Aggregated/. So its "open play" filter (meant to exclude goal kicks) is
silently excluding something else instead. Rather than trust that, this
script recomputes wing play directly from Events/<season> using the
qualifiers already verified in Aggregated/build_season_aggregate.py's own
README (1=Long ball, 2=Cross, 3=Through ball, 5=Free kick, 6=Corner,
107=Throw-in).

Attack-direction correction: Opta pitch coordinates don't flip when a team
switches ends at half-time, so "high y" is only "that team's left wing" in
whichever half they're attacking towards higher x. Direction per team per
half is inferred the same way Coach Profiling does it (average x of the
team's own open-play passes; >=50 implies they're already playing into the
far half on average, so treated as attacking towards lower x that half).

Usage: python3 wing_play_comparison.py [season]   (default: 2025-2026)
"""
import csv
import glob
import json
import os
import statistics
import sys

import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEASON = sys.argv[1] if len(sys.argv) > 1 else "2025-2026"
EVENTS_DIR = os.path.join(ROOT, "Events", SEASON)
OUT_DIR = os.path.join(ROOT, "Aggregated", SEASON)

sys.path.insert(0, ROOT)
from housestyle import style, components  # noqa: E402
from housestyle.colors import DIVERGING_LIGHT  # noqa: E402

Q_FREE_KICK, Q_CORNER, Q_THROW_IN = 5, 6, 107


def qmap(e):
    return {q["qualifierId"] for q in e.get("qualifier", []) or []}


def is_restart(quals):
    return bool(quals & {Q_FREE_KICK, Q_CORNER, Q_THROW_IN})


def main():
    xt_team_rows = list(csv.DictReader(open(os.path.join(ROOT, "xT", "xt_team_summary.csv"),
                                              encoding="utf-8-sig")))
    team_name_by_cid = {r["contestant_id"]: r["team_name"] for r in xt_team_rows}

    teams = {}  # team_name -> {"op_passes": n, "left": n, "right": n}

    for path in sorted(glob.glob(os.path.join(EVENTS_DIR, "*.json"))):
        raw = json.load(open(path, encoding="utf-8"))
        events = raw.get("event", [])

        by_team_period = {}
        for e in events:
            if e["typeId"] != 1:
                continue
            quals = qmap(e)
            if is_restart(quals):
                continue
            cid = e.get("contestantId")
            period = e.get("periodId")
            if period not in (1, 2):
                continue
            by_team_period.setdefault((cid, period), []).append(e)

        direction = {}
        for (cid, period), plist in by_team_period.items():
            avg_x = statistics.fmean(e["x"] for e in plist)
            direction[(cid, period)] = 1 if avg_x < 50 else -1

        for (cid, period), plist in by_team_period.items():
            team = team_name_by_cid.get(cid, cid)
            d = teams.setdefault(team, {"op_passes": 0, "left": 0, "right": 0})
            dirn = direction[(cid, period)]
            for e in plist:
                d["op_passes"] += 1
                y = e.get("y", 50)
                if y <= 25:
                    raw_side = "low"
                elif y >= 75:
                    raw_side = "high"
                else:
                    continue
                if dirn == 1:
                    side = "left" if raw_side == "high" else "right"
                else:
                    side = "right" if raw_side == "high" else "left"
                d[side] += 1

    rows = []
    for team, d in teams.items():
        n = d["op_passes"]
        rows.append({
            "team": team,
            "left_pct": round(d["left"] / n * 100, 2) if n else 0.0,
            "right_pct": round(d["right"] / n * 100, 2) if n else 0.0,
            "wing_pct": round((d["left"] + d["right"]) / n * 100, 2) if n else 0.0,
        })
    rows.sort(key=lambda r: r["wing_pct"], reverse=True)

    with open(os.path.join(OUT_DIR, "wing_play_by_team.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["team", "left_pct", "right_pct", "wing_pct"])
        w.writeheader()
        w.writerows(rows)

    most_wing = rows[0]
    most_left_biased = max(rows, key=lambda r: r["left_pct"] - r["right_pct"])
    most_right_biased = max(rows, key=lambda r: r["right_pct"] - r["left_pct"])
    league_left = statistics.fmean(r["left_pct"] for r in rows)
    league_right = statistics.fmean(r["right_pct"] for r in rows)
    print(f"Most wing play overall: {most_wing['team']} ({most_wing['wing_pct']}%)")
    print(f"Most left-biased: {most_left_biased['team']} "
          f"(L {most_left_biased['left_pct']}% vs R {most_left_biased['right_pct']}%)")
    print(f"Most right-biased: {most_right_biased['team']} "
          f"(L {most_right_biased['left_pct']}% vs R {most_right_biased['right_pct']}%)")
    print(f"League average: left {league_left:.1f}%, right {league_right:.1f}%")

    # ---- chart: diverging left/right wing-play share per team ----
    plot_rows = sorted(rows, key=lambda r: r["wing_pct"])  # ascending, so barh reads top-to-bottom descending
    palette, _ = style.apply("light")
    left_color = DIVERGING_LIGHT["cool"][2]
    right_color = DIVERGING_LIGHT["warm"][2]

    fig = plt.figure(figsize=(9.5, 8.5))
    ax = fig.add_axes([0.30, 0.14, 0.64, 0.62])

    teams_sorted = [r["team"] for r in plot_rows]
    ax.barh(teams_sorted, [-r["left_pct"] for r in plot_rows], color=left_color, label="Left wing")
    ax.barh(teams_sorted, [r["right_pct"] for r in plot_rows], color=right_color, label="Right wing")
    ax.axvline(0, color=palette["axis"], linewidth=1.0)

    max_val = max(max(r["left_pct"], r["right_pct"]) for r in plot_rows)
    step = 10
    top = (int(max_val // step) + 1) * step
    ticks = list(range(-top, top + 1, step))
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{abs(t):.0f}%" for t in ticks])
    ax.set_xlabel("Share of open-play passes from the wide corridor (y<25 or y>75)")
    ax.legend(loc="lower right", frameon=False, fontsize=9)

    bias_note = (f"{most_right_biased['team']} leans right hardest" if
                 (most_right_biased["right_pct"] - most_right_biased["left_pct"]) >=
                 (most_left_biased["left_pct"] - most_left_biased["right_pct"]) else
                 f"{most_left_biased['team']} leans left hardest")

    components.header(
        fig, kicker="Wing Play",
        title=f"{most_wing['team']} plays through the wings more than any other Eredivisie side",
        dek=f"Left vs. right wide-corridor share of open-play passes, {SEASON} ({bias_note})",
        palette=palette,
    )
    components.footer(
        fig, source=f"Opta/StatsPerform {SEASON}",
        palette=palette,
    )
    fig.savefig(os.path.join(OUT_DIR, "wing_play_comparison.png"), facecolor=fig.get_facecolor(), dpi=150)
    print(f"Wrote {os.path.join(OUT_DIR, 'wing_play_comparison.png')}")


if __name__ == "__main__":
    main()

"""
Pitch map: where each team wins the SECOND ball off its own open-play long
balls -- i.e. the first contact was lost or unproductive, but the very next
touch belongs to the same team anyway (restart_analysis.py's own
"second_ball_recovered" flag, reused directly via import, not redefined).

This is a narrower, more literal cut than the long-ball-retention work in
ajax_coach_style/ (which also counts possession re-established a few
touches later as "indirect retention"). Here we only plot the exact
recovery event location for the strict second-ball case.

Attack direction is normalised per team per half using goal_kick_directions()
(pinned to each team's own goal-kick location, not average pass x -- see
that function's docstring in restart_analysis.py for why: the avg-pass-x
heuristic used elsewhere in Aggregated/ flips incorrectly for roughly a
quarter of team-periods when a team's territorial dominance in a given
half pulls their own average pass position past x=50) with a full
180-degree rotation of (x, y) -- not just an x-flip -- so every team's own
left/right sense is preserved while everyone attacks the same way on the
shared pitch grid, and the six panels are directly comparable.

Season: 2025-2026 only (xT/xt_team_summary.csv's team-name-by-contestantId
lookup, reused from wing_play_comparison.py etc., is only built for that
season -- see build_season_aggregate.py's own season gating for why).

Usage: python3 second_ball_pitch_map.py
"""
import csv
import glob
import json
import os
import sys

import matplotlib.pyplot as plt
from mplsoccer import Pitch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SEASON = "2025-2026"
EVENTS_DIR = os.path.join(ROOT, "Events", SEASON)
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "Aggregated"))
from housestyle import style, components  # noqa: E402
from restart_analysis import (  # noqa: E402
    goal_kick_directions, find_restarts, walk_forward, walk_backward, analyse_restart,
)

TEAMS = ["Feyenoord Rotterdam", "PSV Eindhoven", "AFC Ajax", "FC Twente",
         "Nijmegen Eendracht Combinatie", "Alkmaar Zaanstreek"]
SHORT_NAME = {
    "Feyenoord Rotterdam": "Feyenoord", "PSV Eindhoven": "PSV", "AFC Ajax": "Ajax",
    "FC Twente": "FC Twente", "Nijmegen Eendracht Combinatie": "NEC",
    "Alkmaar Zaanstreek": "AZ",
}


def rot(x, y, d):
    return (x, y) if d == 1 else (100 - x, 100 - y)


def main():
    with open(os.path.join(ROOT, "xT", "xt_team_summary.csv"), encoding="utf-8-sig") as f:
        team_name_by_cid = {r["contestant_id"]: r["team_name"] for r in csv.DictReader(f)}

    points = {t: [] for t in TEAMS}
    n_long_balls = {t: 0 for t in TEAMS}

    for path in sorted(glob.glob(os.path.join(EVENTS_DIR, "*.json"))):
        raw = json.load(open(path, encoding="utf-8"))
        events = [e for e in raw.get("event", []) if e.get("periodId") in (1, 2)]
        cids = {e.get("contestantId") for e in events if e.get("contestantId")}
        match_teams = {cid: team_name_by_cid.get(cid) for cid in cids}
        if not any(t in TEAMS for t in match_teams.values()):
            continue

        names = list(match_teams.values())
        if len(names) != 2 or None in names:
            continue
        opp_of_team = {names[0]: names[1], names[1]: names[0]}
        team_of_cid = match_teams

        directions = goal_kick_directions(events, team_of_cid)
        restarts = [r for r in find_restarts(events, team_of_cid, opp_of_team, directions)
                    if r.kind == "long_ball" and r.team in TEAMS]

        for r in restarts:
            n_long_balls[r.team] += 1
            seq = walk_forward(events, r.idx, r.t0)
            pre_seq = walk_backward(events, r.idx, r.t0)
            result = analyse_restart(r, seq, pre_seq, team_of_cid, directions, {}, path, None)
            if not result.get("second_ball_recovered"):
                continue
            ev = seq[1][2]
            x, y = ev.get("x"), ev.get("y")
            if x is None or y is None:
                continue
            d = directions.get((r.team, r.period), 1)
            nx, ny = rot(x, y, d)
            points[r.team].append((nx, ny))

    for t in TEAMS:
        print(f"{t:<32} long_balls={n_long_balls[t]:>4}  second_balls_won={len(points[t]):>4}")

    build_pitch_grid(points, n_long_balls)


def build_pitch_grid(points, n_long_balls):
    palette, cats = style.apply("light")
    fig = plt.figure(figsize=(15, 10.5))
    top, bottom = 0.80, 0.09
    n_rows, n_cols = 2, 3
    h_gap, v_gap = 0.03, 0.10
    panel_w = (0.94 - h_gap * (n_cols - 1)) / n_cols
    panel_h = (top - bottom - v_gap * (n_rows - 1)) / n_rows

    pitch = Pitch(pitch_type="opta", pitch_color=palette["surface"],
                  line_color=palette["axis"], linewidth=1.2, line_zorder=2)

    for i, team in enumerate(points):
        row_i, col_i = divmod(i, n_cols)
        left = 0.03 + col_i * (panel_w + h_gap)
        ax_bottom = top - (row_i + 1) * panel_h - row_i * v_gap
        ax = fig.add_axes([left, ax_bottom, panel_w, panel_h])
        pitch.draw(ax=ax)
        xs = [p[0] for p in points[team]]
        ys = [p[1] for p in points[team]]
        ax.scatter(xs, ys, s=42, color=cats[i % len(cats)], alpha=0.75,
                   edgecolor="white", linewidth=0.5, zorder=3)
        rate_pct = len(points[team]) / n_long_balls[team] * 100 if n_long_balls[team] else 0
        ax.set_title(f"{SHORT_NAME[team]}  ({len(points[team])} of {n_long_balls[team]} "
                      f"long balls, {rate_pct:.0f}%)",
                      fontsize=11, color=palette["ink_primary"], fontweight="bold", pad=8)

    components.header(
        fig, kicker="Second-Ball Recovery",
        title="Where the big six-plus rivals win the second ball off their own long balls",
        dek=f"Recovery location, own open-play long balls, {SEASON} -- direction normalised, "
            "every team attacks toward the right",
        palette=palette, top=0.94,
    )
    components.footer(fig, source=f"Opta/StatsPerform {SEASON}", palette=palette)

    out_path = os.path.join(OUT_DIR, "second_ball_pitch_map.png")
    fig.savefig(out_path, facecolor=fig.get_facecolor(), dpi=150)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()

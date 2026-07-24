"""
Same second-ball-recovery cut as second_ball_pitch_map.py, but drawn as the
actual long ball itself -- start point to landing spot -- rather than a
single dot at the recovery point. The landing spot (qualifier 140/141, the
pass's own end_x/end_y) and the second-ball recovery event are usually the
same patch of grass (the recovery is by definition the very next touch),
so it doubles as "roughly where the second ball was won."

400-670 individual arrows per team is unreadable overplotted (tried it --
solid spaghetti). Binned instead, mplsoccer's own tool for exactly this:
a heatmap of landing-zone density (bins=6x4) plus one averaged flow arrow
per zone showing the typical direction long balls into that zone came
from -- Pitch.flow(), the same technique as mplsoccer's own pass-map
example, not a from-scratch binning.

Same six teams, same season, same second_ball_recovered flag reused
directly from restart_analysis.py -- see second_ball_pitch_map.py's
docstring for the shared methodology notes (direction normalisation,
scope).

Usage: python3 second_ball_pass_map.py
"""
import csv
import glob
import json
import os
import sys

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from mplsoccer import Pitch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SEASON = "2025-2026"
EVENTS_DIR = os.path.join(ROOT, "Events", SEASON)
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "Aggregated"))
from housestyle import style, components  # noqa: E402
from restart_analysis import (  # noqa: E402
    team_directions, find_restarts, walk_forward, walk_backward, analyse_restart,
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

    arrows = {t: [] for t in TEAMS}  # (x0, y0, x1, y1) per completed long ball
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

        directions = team_directions(events, team_of_cid)
        restarts = [r for r in find_restarts(events, team_of_cid, opp_of_team, directions)
                    if r.kind == "long_ball" and r.team in TEAMS]

        for r in restarts:
            n_long_balls[r.team] += 1
            seq = walk_forward(events, r.idx, r.t0)
            pre_seq = walk_backward(events, r.idx, r.t0)
            result = analyse_restart(r, seq, pre_seq, team_of_cid, directions, {}, path, None)
            if not result.get("second_ball_recovered"):
                continue
            d = directions.get((r.team, r.period), 1)
            x0, y0 = rot(r.ox, r.oy, d)
            x1, y1 = rot(r.ex, r.ey, d)
            arrows[r.team].append((x0, y0, x1, y1))

    for t in TEAMS:
        print(f"{t:<32} long_balls={n_long_balls[t]:>4}  second_balls_won={len(arrows[t]):>4}")

    build_pitch_grid(arrows, n_long_balls)


def build_pitch_grid(arrows, n_long_balls):
    palette, cats = style.apply("light")
    fig = plt.figure(figsize=(15, 10.5))
    top, bottom = 0.80, 0.09
    n_rows, n_cols = 2, 3
    h_gap, v_gap = 0.03, 0.10
    panel_w = (0.94 - h_gap * (n_cols - 1)) / n_cols
    panel_h = (top - bottom - v_gap * (n_rows - 1)) / n_rows

    pitch = Pitch(pitch_type="opta", pitch_color=palette["surface"],
                  line_color=palette["axis"], linewidth=1.2, line_zorder=2)

    for i, team in enumerate(arrows):
        row_i, col_i = divmod(i, n_cols)
        left = 0.03 + col_i * (panel_w + h_gap)
        ax_bottom = top - (row_i + 1) * panel_h - row_i * v_gap
        ax = fig.add_axes([left, ax_bottom, panel_w, panel_h])
        pitch.draw(ax=ax)
        x0 = [a[0] for a in arrows[team]]
        y0 = [a[1] for a in arrows[team]]
        x1 = [a[2] for a in arrows[team]]
        y1 = [a[3] for a in arrows[team]]

        team_color = cats[i % len(cats)]
        cmap = LinearSegmentedColormap.from_list("team_density", [palette["surface"], team_color])
        bs = pitch.bin_statistic(x1, y1, statistic="count", bins=(6, 4))
        pitch.heatmap(bs, ax=ax, cmap=cmap, edgecolors=palette["surface"], zorder=1)
        pitch.flow(x0, y0, x1, y1, ax=ax, bins=(6, 4), arrow_type="scale", arrow_length=11,
                   color=palette["ink_primary"], alpha=0.85, width=1.6,
                   headwidth=3.5, headlength=3.5, headaxislength=3, zorder=3)
        rate_pct = len(arrows[team]) / n_long_balls[team] * 100 if n_long_balls[team] else 0
        ax.set_title(f"{SHORT_NAME[team]}  ({len(arrows[team])} of {n_long_balls[team]} "
                      f"long balls, {rate_pct:.0f}%)",
                      fontsize=11, color=palette["ink_primary"], fontweight="bold", pad=8)

    components.header(
        fig, kicker="Second-Ball Recovery",
        title="Where long balls that won the second ball actually landed",
        dek=f"Landing-zone density (shading) and average direction (arrows) of own open-play long "
            f"balls where the very next touch is theirs, {SEASON} -- every team attacks toward the right",
        palette=palette, top=0.94,
    )
    components.footer(fig, source=f"Opta/StatsPerform {SEASON}", palette=palette)

    out_path = os.path.join(OUT_DIR, "second_ball_pass_map.png")
    fig.savefig(out_path, facecolor=fig.get_facecolor(), dpi=150)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()

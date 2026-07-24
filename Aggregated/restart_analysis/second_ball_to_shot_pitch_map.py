"""
Long balls where the team won the second ball AND it led to a shot -- the
sharpest possible cut of the second-ball-recovery work: not just "did we
get the ball back", but "did getting it back actually produce a shot."

Reuses restart_analysis.py's own definitions directly, not redefined:
  - second_ball_recovered: first contact lost/unproductive, but the very
    next touch belongs to the restart team (same flag used in
    second_ball_pitch_map.py / second_ball_pass_map.py).
  - shot_within_20 (SCR20 in the league-wide metric set): a shot by the
    restart team within 20s of the restart itself -- reusing that same
    20s window here rather than inventing a different one, for
    consistency with the rest of Aggregated/restart_analysis/.

Only restarts where BOTH are true. For each, three points are plotted:
long-ball origin -> landing spot (thin, muted) -> shot location (bold,
team-colored arrow), with the shot itself marked as a filled star if it
was a goal, an open circle otherwise. Volume is much lower than the two
prior scripts (a subset of a subset), so individual paths are legible
without binning.

Usage: python3 second_ball_to_shot_pitch_map.py
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
SHOT_WINDOW_S = 20.0

sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "Aggregated"))
from housestyle import style, components  # noqa: E402
from restart_analysis import (  # noqa: E402
    goal_kick_directions, find_restarts, walk_forward, walk_backward, analyse_restart,
    SHOT_TYPES,
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


def first_shot(seq, team_of_cid, team, window_s):
    for elapsed, cid, e in seq:
        if elapsed > window_s:
            break
        if e.get("typeId") in SHOT_TYPES and team_of_cid.get(cid) == team:
            return e
    return None


def main():
    with open(os.path.join(ROOT, "xT", "xt_team_summary.csv"), encoding="utf-8-sig") as f:
        team_name_by_cid = {r["contestant_id"]: r["team_name"] for r in csv.DictReader(f)}

    chains = {t: [] for t in TEAMS}  # (x0,y0, x1,y1, xs,ys, is_goal)
    n_long_balls = {t: 0 for t in TEAMS}
    n_second_ball_won = {t: 0 for t in TEAMS}

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
            n_second_ball_won[r.team] += 1
            if not result.get("shot_within_20"):
                continue
            shot = first_shot(seq, team_of_cid, r.team, SHOT_WINDOW_S)
            if shot is None or shot.get("x") is None:
                continue
            d = directions.get((r.team, r.period), 1)
            x0, y0 = rot(r.ox, r.oy, d)
            x1, y1 = rot(r.ex, r.ey, d)
            xs, ys = rot(shot["x"], shot["y"], d)
            chains[r.team].append((x0, y0, x1, y1, xs, ys, shot.get("typeId") == 16))

    for t in TEAMS:
        n_shots = len(chains[t])
        print(f"{t:<32} long_balls={n_long_balls[t]:>4}  second_ball_won={n_second_ball_won[t]:>4}  "
              f"-> shot within {SHOT_WINDOW_S:.0f}s={n_shots:>3}  "
              f"({n_shots / n_second_ball_won[t] * 100 if n_second_ball_won[t] else 0:.1f}% of second-ball wins, "
              f"goals={sum(1 for c in chains[t] if c[6])})")

    build_pitch_grid(chains, n_long_balls, n_second_ball_won)
    build_single_team(chains, n_long_balls, n_second_ball_won, "FC Twente", team_index=3)


def build_pitch_grid(chains, n_long_balls, n_second_ball_won):
    palette, cats = style.apply("light")
    fig = plt.figure(figsize=(15, 10.5))
    top, bottom = 0.80, 0.09
    n_rows, n_cols = 2, 3
    h_gap, v_gap = 0.03, 0.10
    panel_w = (0.94 - h_gap * (n_cols - 1)) / n_cols
    panel_h = (top - bottom - v_gap * (n_rows - 1)) / n_rows

    pitch = Pitch(pitch_type="opta", pitch_color=palette["surface"],
                  line_color=palette["axis"], linewidth=1.2, line_zorder=2)

    for i, team in enumerate(chains):
        row_i, col_i = divmod(i, n_cols)
        left = 0.03 + col_i * (panel_w + h_gap)
        ax_bottom = top - (row_i + 1) * panel_h - row_i * v_gap
        ax = fig.add_axes([left, ax_bottom, panel_w, panel_h])
        pitch.draw(ax=ax)

        team_color = cats[i % len(cats)]
        rows = chains[team]
        if rows:
            x0 = [c[0] for c in rows]; y0 = [c[1] for c in rows]
            x1 = [c[2] for c in rows]; y1 = [c[3] for c in rows]
            xs = [c[4] for c in rows]; ys = [c[5] for c in rows]
            pitch.lines(x0, y0, x1, y1, ax=ax, color=palette["axis"], alpha=0.55,
                       linewidth=1.1, zorder=2.5)
            pitch.arrows(x1, y1, xs, ys, ax=ax, color=team_color, alpha=0.75,
                        width=1.6, headwidth=4.5, headlength=4.5, zorder=3)
            gx = [c[4] for c in rows if c[6]]
            gy = [c[5] for c in rows if c[6]]
            nx = [c[4] for c in rows if not c[6]]
            ny = [c[5] for c in rows if not c[6]]
            pitch.scatter(nx, ny, ax=ax, s=60, marker="o", color=palette["surface"],
                         edgecolor=team_color, linewidth=1.4, zorder=4)
            if gx:
                pitch.scatter(gx, gy, ax=ax, s=140, marker="*", color=palette["accent"],
                             edgecolor="white", linewidth=0.6, zorder=5)

        n_shots = len(rows)
        n_goals = sum(1 for c in rows if c[6])
        pct = n_shots / n_second_ball_won[team] * 100 if n_second_ball_won[team] else 0
        ax.set_title(f"{SHORT_NAME[team]}  ({n_shots} shots, {n_goals} goals, "
                      f"{pct:.0f}% of {n_second_ball_won[team]} second-ball wins)",
                      fontsize=10.5, color=palette["ink_primary"], fontweight="bold", pad=8)

    components.header(
        fig, kicker="Second-Ball To Shot",
        title="When winning the second ball off a long ball actually produced a shot",
        dek=f"Own open-play long balls where the second ball was won AND a shot followed within "
            f"{SHOT_WINDOW_S:.0f}s, {SEASON} (★ = goal) -- every team attacks toward the right",
        palette=palette, top=0.94,
    )
    components.footer(fig, source=f"Opta/StatsPerform {SEASON}", palette=palette)

    out_path = os.path.join(OUT_DIR, "second_ball_to_shot_pitch_map.png")
    fig.savefig(out_path, facecolor=fig.get_facecolor(), dpi=150)
    print(f"Wrote {out_path}")


def build_single_team(chains, n_long_balls, n_second_ball_won, team, team_index):
    """Same chain (long-ball origin -> landing -> shot) as one panel of the
    six-team grid, blown up to its own full-size chart -- same data, same
    colour, just easier to read the individual chains for one team."""
    palette, cats = style.apply("light")
    team_color = cats[team_index % len(cats)]
    rows = chains[team]

    fig = plt.figure(figsize=(11, 8.8))
    ax = fig.add_axes([0.06, 0.13, 0.88, 0.58])
    pitch = Pitch(pitch_type="opta", pitch_color=palette["surface"],
                  line_color=palette["axis"], linewidth=1.3, line_zorder=2)
    pitch.draw(ax=ax)

    if rows:
        x0 = [c[0] for c in rows]; y0 = [c[1] for c in rows]
        x1 = [c[2] for c in rows]; y1 = [c[3] for c in rows]
        xs = [c[4] for c in rows]; ys = [c[5] for c in rows]
        pitch.lines(x0, y0, x1, y1, ax=ax, color=palette["axis"], alpha=0.6,
                   linewidth=1.5, zorder=2.5)
        pitch.arrows(x1, y1, xs, ys, ax=ax, color=team_color, alpha=0.8,
                    width=2.2, headwidth=5, headlength=5, zorder=3)
        gx = [c[4] for c in rows if c[6]]
        gy = [c[5] for c in rows if c[6]]
        nx = [c[4] for c in rows if not c[6]]
        ny = [c[5] for c in rows if not c[6]]
        pitch.scatter(nx, ny, ax=ax, s=110, marker="o", color=palette["surface"],
                     edgecolor=team_color, linewidth=2, zorder=4)
        if gx:
            pitch.scatter(gx, gy, ax=ax, s=280, marker="*", color=palette["accent"],
                         edgecolor="white", linewidth=1, zorder=5)

    n_shots = len(rows)
    n_goals = sum(1 for c in rows if c[6])
    pct = n_shots / n_second_ball_won[team] * 100 if n_second_ball_won[team] else 0
    components.header(
        fig, kicker="Second-Ball To Shot",
        title=f"{team}: {n_shots} shots and {n_goals} goals off a won second ball",
        dek=f"Own long balls -> won second ball -> shot within {SHOT_WINDOW_S:.0f}s "
            f"({pct:.0f}% of {n_second_ball_won[team]} wins), {SEASON}",
        palette=palette, top=0.94,
    )
    components.footer(
        fig, source=f"Opta/StatsPerform {SEASON}",
        note="★ = goal · grey line = the long ball · arrow = second ball to shot",
        palette=palette,
    )

    safe_name = team.replace(" ", "_").lower()
    out_path = os.path.join(OUT_DIR, f"second_ball_to_shot_{safe_name}.png")
    fig.savefig(out_path, facecolor=fig.get_facecolor(), dpi=150)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()

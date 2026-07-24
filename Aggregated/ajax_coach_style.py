"""
Ajax playing style, per coach/manager, across 2023-2024 -> 2025-2026.

The Opta event feed has no manager/coach field anywhere (matchDetails only
has periodId/matchStatus/winner/scores) so tenure boundaries come from
outside the data -- sourced via web search and cross-checked against
publicly reported sack/appointment dates, then applied by match date:

  2023-2024  Maurice Steijn        2023-08-12 -> 2023-10-22  (sacked 23 Oct 2023)
             Hedwiges Maduro (c)   2023-10-29 -> 2023-10-29  (caretaker, 1 match)
             John van 't Schip     2023-11-02 -> 2024-05-19
  2024-2025  Francesco Farioli     2024-08-11 -> 2025-05-18  (full season)
  2025-2026  John Heitinga         2025-08-10 -> 2025-11-01  (sacked 6 Nov 2025)
             Fred Grim (interim)   2025-11-09 -> 2026-03-07  (Garcia appointed 8 Mar 2026)
             Oscar Garcia          2026-03-14 -> 2026-05-24

Maduro's single match is kept in the table for completeness but flagged
n=1 everywhere -- not a reliable regime average, shown for transparency
rather than folded silently into either neighbour.

Ajax's contestantId is stable across all three seasons' event files
(d0zdg647gvgc95xdtk1vpbkys) -- confirmed directly, not assumed, by
checking the recurring id across one Ajax home match per season.

Style metrics computed directly from Events/<season>/*Ajax*.json, with
attack-direction correction per team per half (average x of a team's own
open-play passes decides which way they're attacking that half -- same
method as wing_play_comparison.py / diagonal_vs_relationism.py). This
script does NOT reuse build_season_aggregate.py's progressive-pass /
forward-pass logic, which compares raw end_x > x with no such per-half
correction and so silently mislabels forward/backward for whichever team
is defending the x=100 end in a given half.

Usage: python3 ajax_coach_style.py
"""
import csv
import glob
import json
import os
import statistics
import sys

import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "Aggregated", "ajax_coach_style")
os.makedirs(OUT_DIR, exist_ok=True)

sys.path.insert(0, ROOT)
from housestyle import style, components  # noqa: E402

AJAX_CID = "d0zdg647gvgc95xdtk1vpbkys"

X_SCALE, Y_SCALE = 1.05, 0.68
GOAL_X, GOAL_Y = 105.0, 34.0
PROG_OWN_HALF_M = 27.432
PROG_TO_ATT_HALF_M = 13.716
PROG_ATT_HALF_M = 9.144
FINAL_THIRD_X = 200.0 / 3.0

Q_LONG_BALL, Q_CROSS, Q_THROUGH_BALL, Q_FREE_KICK, Q_CORNER = 1, 2, 3, 5, 6
Q_THROW_IN = 107
Q_END_X, Q_END_Y = 140, 141
Q_LENGTH_M = 212
RESTART_QUALIFIERS = {Q_FREE_KICK, Q_CORNER, Q_THROW_IN}
NON_TOUCH_TYPES = {17, 18, 19, 27, 28, 30, 32, 34, 37, 40, 43, 58, 65, 70, 71, 79, 84}

REGIMES = [
    ("2023-2024", "Maurice Steijn", "2023-08-12", "2023-10-22"),
    ("2023-2024", "Hedwiges Maduro (caretaker)", "2023-10-23", "2023-10-29"),
    ("2023-2024", "John van 't Schip", "2023-10-30", "2024-06-01"),
    ("2024-2025", "Francesco Farioli", "2024-08-01", "2025-06-01"),
    ("2025-2026", "John Heitinga", "2025-08-01", "2025-11-01"),
    ("2025-2026", "Fred Grim (interim)", "2025-11-02", "2026-03-07"),
    ("2025-2026", "Oscar Garcia", "2026-03-08", "2026-06-01"),
]


def qmap(e):
    return {q["qualifierId"]: q.get("value") for q in e.get("qualifier", []) or []}


def num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def to_m(x, y):
    return x * X_SCALE, y * Y_SCALE


def dist_to_goal_m(x, y):
    xm, ym = to_m(x, y)
    return ((GOAL_X - xm) ** 2 + (GOAL_Y - ym) ** 2) ** 0.5


def is_progressive(sx, sy, ex, ey):
    gain = dist_to_goal_m(sx, sy) - dist_to_goal_m(ex, ey)
    if sx < 50 and ex < 50:
        return gain >= PROG_OWN_HALF_M
    if sx < 50 <= ex:
        return gain >= PROG_TO_ATT_HALF_M
    return gain >= PROG_ATT_HALF_M


def match_date(path):
    return os.path.basename(path).split("_", 1)[0]


def find_regime(season, date_str):
    for s, coach, start, end in REGIMES:
        if s == season and start <= date_str <= end:
            return coach
    return None


def team_directions(events, cid_of_interest_only=None):
    """avg-x heuristic, per (contestantId, period), from that team's own
    completed-or-not open-play passes -- same method used elsewhere in
    Aggregated/ so results are directly comparable."""
    by_team_period = {}
    for e in events:
        if e.get("typeId") != 1 or e.get("periodId") not in (1, 2):
            continue
        q = qmap(e)
        if RESTART_QUALIFIERS & q.keys():
            continue
        cid = e.get("contestantId")
        by_team_period.setdefault((cid, e["periodId"]), []).append(e["x"])
    return {key: (1 if statistics.fmean(xs) < 50 else -1) for key, xs in by_team_period.items()}


def analyse_match(path):
    raw = json.load(open(path, encoding="utf-8"))
    events = [e for e in raw.get("event", []) if e.get("periodId") in (1, 2)]
    if not any(e.get("contestantId") == AJAX_CID for e in events):
        return None

    directions = team_directions(events)
    opp_cid = next((e["contestantId"] for e in events
                     if e.get("contestantId") and e["contestantId"] != AJAX_CID), None)

    def norm_x_as(x, cid, period):
        d = directions.get((cid, period))
        if d is None:
            return x
        return x if d == 1 else 100 - x

    minutes = None
    md = raw.get("matchDetails", {})
    if md.get("matchLengthMin") is not None:
        minutes = num(md["matchLengthMin"]) + num(md.get("matchLengthSec", 0)) / 60.0

    m = {
        "passes": 0, "long_balls": 0, "crosses": 0,
        "progressive": 0, "final_third_entries": 0,
        "pass_len_sum": 0.0, "pass_len_n": 0,
        "wing_passes": 0, "op_passes": 0,
        "shots": 0, "goals": 0,
        "touches": 0, "territory_x_sum": 0.0,
        "own_pass_share": None,
        "opp_press_zone_passes": 0, "own_def_actions_press_zone": 0,
    }

    team_pass_ct = {AJAX_CID: 0, opp_cid: 0}
    for e in events:
        if e.get("typeId") == 1:
            cid = e.get("contestantId")
            if cid in team_pass_ct:
                team_pass_ct[cid] += 1
    total_passes = team_pass_ct[AJAX_CID] + team_pass_ct.get(opp_cid, 0)
    if total_passes:
        m["own_pass_share"] = team_pass_ct[AJAX_CID] / total_passes * 100

    for e in events:
        cid = e.get("contestantId")
        t = e.get("typeId")
        period = e.get("periodId")
        q = qmap(e)
        x, y = e.get("x"), e.get("y")

        # --- PPDA zone bookkeeping: locate ANY event in Ajax's own frame ---
        if cid and x is not None:
            ajax_frame_x = norm_x_as(x, AJAX_CID, period)
            in_press_zone = ajax_frame_x < 60
            if in_press_zone:
                if cid == opp_cid and t == 1:
                    m["opp_press_zone_passes"] += 1
                elif cid == AJAX_CID and t in (4, 7, 8):  # foul, tackle, interception
                    m["own_def_actions_press_zone"] += 1

        if cid != AJAX_CID:
            continue

        nx = norm_x_as(x, AJAX_CID, period) if x is not None else None

        if t not in NON_TOUCH_TYPES and x is not None and not (x == 0 and y == 0):
            m["touches"] += 1
            m["territory_x_sum"] += nx

        if t in (13, 14, 15, 16):
            m["shots"] += 1
            if t == 16:
                m["goals"] += 1

        if t != 1:
            continue

        is_restart = bool(RESTART_QUALIFIERS & q.keys())
        completed = e.get("outcome") == 1
        m["passes"] += 1
        if Q_LONG_BALL in q:
            m["long_balls"] += 1
        if Q_CROSS in q:
            m["crosses"] += 1

        length_m = q.get(Q_LENGTH_M)
        if length_m is not None:
            m["pass_len_sum"] += num(length_m)
            m["pass_len_n"] += 1

        if not is_restart:
            m["op_passes"] += 1
            if y is not None and (y <= 25 or y >= 75):
                m["wing_passes"] += 1

        if completed and Q_END_X in q:
            ex, ey = num(q[Q_END_X]), num(q.get(Q_END_Y))
            nex = norm_x_as(ex, AJAX_CID, period)
            if is_progressive(nx, y, nex, ey):
                m["progressive"] += 1
            if nx < FINAL_THIRD_X <= nex:
                m["final_third_entries"] += 1

    m["minutes"] = minutes if minutes else 90.0
    return m


def main():
    match_paths = []
    for season in ("2023-2024", "2024-2025", "2025-2026"):
        for path in sorted(glob.glob(os.path.join(ROOT, "Events", season, "*Ajax*.json"))):
            date_str = match_date(path)
            coach = find_regime(season, date_str)
            if coach is None:
                print(f"WARNING: no regime match for {season} {date_str}", file=sys.stderr)
                continue
            match_paths.append((season, coach, path))

    agg = {}
    order = []
    for season, coach, path in match_paths:
        key = (season, coach)
        if key not in agg:
            agg[key] = {"n": 0, "minutes": 0.0, "passes": 0, "long_balls": 0, "crosses": 0,
                        "progressive": 0, "final_third_entries": 0, "pass_len_sum": 0.0,
                        "pass_len_n": 0, "wing_passes": 0, "op_passes": 0, "shots": 0,
                        "goals": 0, "touches": 0, "territory_x_sum": 0.0,
                        "own_pass_share_sum": 0.0, "own_pass_share_n": 0,
                        "opp_press_zone_passes": 0, "own_def_actions_press_zone": 0}
            order.append(key)
        m = analyse_match(path)
        if m is None:
            continue
        a = agg[key]
        a["n"] += 1
        a["minutes"] += m["minutes"]
        for f in ("passes", "long_balls", "crosses", "progressive", "final_third_entries",
                  "pass_len_sum", "pass_len_n", "wing_passes", "op_passes", "shots", "goals",
                  "touches", "territory_x_sum", "opp_press_zone_passes",
                  "own_def_actions_press_zone"):
            a[f] += m[f]
        if m["own_pass_share"] is not None:
            a["own_pass_share_sum"] += m["own_pass_share"]
            a["own_pass_share_n"] += 1

    rows = []
    for season, coach in order:
        a = agg[(season, coach)]
        n90 = a["minutes"] / 90.0 if a["minutes"] else a["n"]
        rows.append({
            "season": season,
            "coach": coach,
            "matches": a["n"],
            "possession_pct": round(a["own_pass_share_sum"] / a["own_pass_share_n"], 1)
                if a["own_pass_share_n"] else "",
            "ppda": round(a["opp_press_zone_passes"] / a["own_def_actions_press_zone"], 2)
                if a["own_def_actions_press_zone"] else "",
            "long_ball_pct": round(a["long_balls"] / a["passes"] * 100, 1) if a["passes"] else "",
            "cross_pct": round(a["crosses"] / a["passes"] * 100, 1) if a["passes"] else "",
            "wing_pct": round(a["wing_passes"] / a["op_passes"] * 100, 1) if a["op_passes"] else "",
            "avg_pass_length_m": round(a["pass_len_sum"] / a["pass_len_n"], 2)
                if a["pass_len_n"] else "",
            "progressive_passes_per90": round(a["progressive"] / n90, 2) if n90 else "",
            "final_third_entries_per90": round(a["final_third_entries"] / n90, 2) if n90 else "",
            "territory_index": round(a["territory_x_sum"] / a["touches"], 1) if a["touches"] else "",
            "shots_per90": round(a["shots"] / n90, 2) if n90 else "",
            "goals_per90": round(a["goals"] / n90, 2) if n90 else "",
        })

    csv_path = os.path.join(OUT_DIR, "ajax_coach_style.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {csv_path}")

    for r in rows:
        flag = "  (n=1, not reliable)" if r["matches"] == 1 else ""
        print(f"{r['season']} {r['coach']:<28} n={r['matches']:>2}  poss={r['possession_pct']}%  "
              f"PPDA={r['ppda']}  long_ball={r['long_ball_pct']}%  wing={r['wing_pct']}%  "
              f"prog/90={r['progressive_passes_per90']}  territory={r['territory_index']}{flag}")

    build_chart(rows)


def build_chart(rows):
    palette, cats = style.apply("light")
    labels = [f"{r['coach'].split(' (')[0]}\n{r['season']}" for r in rows]
    n = len(rows)
    colors = [cats[i % len(cats)] for i in range(n)]

    metrics = [
        ("possession_pct", "Possession share (%)"),
        ("ppda", "PPDA (lower = more pressing)"),
        ("long_ball_pct", "Long balls (% of passes)"),
        ("wing_pct", "Wide-corridor passes (%)"),
        ("progressive_passes_per90", "Progressive passes /90"),
        ("territory_index", "Territory index (0=own goal, 100=opp. goal)"),
    ]

    fig = plt.figure(figsize=(13.5, 9.5))
    grid_top, grid_bottom = 0.78, 0.17
    n_rows, n_cols = 2, 3
    h_gap, v_gap = 0.045, 0.14
    panel_w = (0.90 - h_gap * (n_cols - 1)) / n_cols
    panel_h = (grid_top - grid_bottom - v_gap * (n_rows - 1)) / n_rows

    for i, (field, panel_title) in enumerate(metrics):
        row_i, col_i = divmod(i, n_cols)
        left = 0.05 + col_i * (panel_w + h_gap)
        bottom = grid_top - (row_i + 1) * panel_h - row_i * v_gap
        ax = fig.add_axes([left, bottom, panel_w, panel_h])
        vals = [r[field] if r[field] != "" else 0 for r in rows]
        bars = ax.bar(range(n), vals, color=colors, width=0.65)
        for j, r in enumerate(rows):
            if r["matches"] == 1:
                bars.patches[j].set_hatch("///")
                bars.patches[j].set_edgecolor(palette["ink_muted"])
        ax.set_xticks(range(n))
        ax.set_xticklabels([lb.split("\n")[0] for lb in labels], fontsize=7.5, rotation=32, ha="right")
        ax.set_title(panel_title, fontsize=10, color=palette["ink_primary"], loc="left", pad=6)
        ax.tick_params(axis="y", labelsize=8)

    components.header(
        fig, kicker="Playing Style By Manager",
        title="How Ajax's identity shifted from Steijn to Garcia",
        dek="Ajax playing-style metrics per coaching regime, 2023-2024 to 2025-2026 "
            "(hatched bar = single-match caretaker spell, n=1)",
        palette=palette, top=0.94,
    )
    components.footer(fig, source="Opta/StatsPerform 2023-2026", palette=palette)

    out_path = os.path.join(OUT_DIR, "ajax_coach_style.png")
    fig.savefig(out_path, facecolor=fig.get_facecolor(), dpi=150)
    print(f"Wrote {out_path}")

    build_scatter(rows, palette, cats)


def build_scatter(rows, palette, cats):
    plot_rows = [r for r in rows if r["ppda"] != "" and r["possession_pct"] != ""]
    fig = plt.figure(figsize=(9.5, 7.8))
    ax = fig.add_axes([0.11, 0.14, 0.74, 0.58])
    for i, r in enumerate(plot_rows):
        marker = "^" if r["matches"] == 1 else "o"
        ax.scatter([r["possession_pct"]], [r["ppda"]], s=140, color=cats[i % len(cats)],
                   marker=marker, zorder=3, edgecolor="white", linewidth=0.8)
        label = r["coach"].split(" (")[0]
        ax.annotate(f"{label}\n{r['season']}", xy=(r["possession_pct"], r["ppda"]),
                    xytext=(7, 6), textcoords="offset points", fontsize=8.7,
                    color=palette["ink_primary"], fontweight="bold")
    xs = [r["possession_pct"] for r in plot_rows]
    ax.set_xlim(min(xs) - 2, max(xs) + 5)
    ax.invert_yaxis()
    ax.set_xlabel("Possession share (%)")
    ax.set_ylabel("PPDA (lower = higher-intensity press)")

    components.header(
        fig, kicker="Style Signature",
        title="Possession share vs. pressing intensity, by Ajax coach",
        dek="Each point is one coaching regime's full sample (▲ = single-match caretaker spell)",
        palette=palette,
    )
    components.footer(fig, source="Opta/StatsPerform 2023-2026", palette=palette)

    out_path = os.path.join(OUT_DIR, "ajax_coach_style_scatter.png")
    fig.savefig(out_path, facecolor=fig.get_facecolor(), dpi=150)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()

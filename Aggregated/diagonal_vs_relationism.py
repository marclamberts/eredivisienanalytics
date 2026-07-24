"""
Diagonal passing vs. the Relationism Index: do teams that play more
"relationist" football (per PSV Season Report/Scripts/relationism_index.py's
proxy score) also play more diagonal passes -- the passing-lane signature
relational/proximity-based combination play is supposed to produce, as
opposed to the vertical/horizontal lines of a fixed positional structure?

Relationism Index: ported directly from relationism_index.py's own method
(equal-weighted percentile blend of inverse pass distance, central-third
touch share, and passes-per-possession-sequence) rather than imported,
because that script's own pi_ratings_lib.py hardcodes a Mac-only DATA_DIR
that doesn't exist in this repo. Same formula, run against
Events/<season> directly.

Diagonal pass %: NEW metric, not previously in this repo. A completed
open-play pass (excluding free-kick/corner/throw-in, qualifiers 5/6/107)
of at least 5m is classed as "diagonal" if the angle of its direction from
the horizontal is between 25 and 65 degrees -- i.e. neither close to
straight upfield/backward (near 0 deg) nor a square ball across the pitch
(near 90 deg). Share of a team's completed open-play passes meeting that.

Usage: python3 diagonal_vs_relationism.py [season]   (default: 2025-2026)
"""
import csv
import glob
import json
import math
import os
import sys

import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEASON = sys.argv[1] if len(sys.argv) > 1 else "2025-2026"
EVENTS_DIR = os.path.join(ROOT, "Events", SEASON)
OUT_DIR = os.path.join(ROOT, "Aggregated", SEASON)

sys.path.insert(0, ROOT)
from housestyle import style, components  # noqa: E402

X_SCALE, Y_SCALE = 1.05, 0.68
Q_FREE_KICK, Q_CORNER, Q_THROW_IN = 5, 6, 107
Q_END_X, Q_END_Y = 140, 141

DIAG_MIN_DEG, DIAG_MAX_DEG = 25, 65
DIAG_MIN_LEN_M = 5.0

MIN_PASSES = 3            # relationism sequence filter, same as relationism_index.py
MIN_DURATION_S, MAX_DURATION_S = 1.0, 300.0
NON_TOUCH_TYPES = {17, 18, 19, 27, 28, 30, 32, 34, 37, 40, 43, 58, 65, 70, 71, 79, 84}


def qmap(e):
    return {q["qualifierId"]: q.get("value") for q in e.get("qualifier", []) or []}


def minute_value(e):
    return float(e.get("timeMin") or 0) + float(e.get("timeSec") or 0) / 60.0


def dist_m(x0, y0, x1, y1):
    dx, dy = (x1 - x0) * X_SCALE, (y1 - y0) * Y_SCALE
    return (dx ** 2 + dy ** 2) ** 0.5


def pass_end_xy(e, q):
    ex, ey = q.get(Q_END_X), q.get(Q_END_Y)
    return (float(ex) if ex is not None else e["x"]), (float(ey) if ey is not None else e["y"])


def percentile_rank(values):
    items = sorted(values.items(), key=lambda kv: kv[1])
    n = len(items)
    return {k: (i / (n - 1) * 100 if n > 1 else 50.0) for i, (k, v) in enumerate(items)}


def main():
    xt_team_rows = list(csv.DictReader(open(os.path.join(ROOT, "xT", "xt_team_summary.csv"),
                                              encoding="utf-8-sig")))
    team_name_by_cid = {r["contestant_id"]: r["team_name"] for r in xt_team_rows}

    pass_dist_sum, pass_dist_n = {}, {}
    central_n, touch_n = {}, {}
    seq_lengths = {}
    diag_n, op_pass_n = {}, {}

    for path in sorted(glob.glob(os.path.join(EVENTS_DIR, "*.json"))):
        raw = json.load(open(path, encoding="utf-8"))
        events = [e for e in raw.get("event", []) if e.get("periodId") in (1, 2, 3, 4) and e.get("contestantId")]
        events.sort(key=lambda e: (e.get("periodId", 0), minute_value(e), e.get("eventId", 0)))

        for e in events:
            team = team_name_by_cid.get(e["contestantId"])
            if team is None:
                continue
            q = qmap(e)

            if e.get("typeId") == 1:
                ex, ey = pass_end_xy(e, q)
                d = dist_m(e["x"], e["y"], ex, ey)
                pass_dist_sum[team] = pass_dist_sum.get(team, 0.0) + d
                pass_dist_n[team] = pass_dist_n.get(team, 0) + 1

                is_restart = bool({Q_FREE_KICK, Q_CORNER, Q_THROW_IN} & q.keys())
                if e.get("outcome") == 1 and not is_restart:
                    op_pass_n[team] = op_pass_n.get(team, 0) + 1
                    dx = (ex - e["x"]) * X_SCALE
                    dy = (ey - e["y"]) * Y_SCALE
                    length = (dx ** 2 + dy ** 2) ** 0.5
                    if length >= DIAG_MIN_LEN_M:
                        angle_deg = math.degrees(math.atan2(abs(dy), abs(dx))) if dx or dy else 0.0
                        if DIAG_MIN_DEG <= angle_deg <= DIAG_MAX_DEG:
                            diag_n[team] = diag_n.get(team, 0) + 1

            if e.get("typeId") in NON_TOUCH_TYPES:
                continue
            x, y = e.get("x"), e.get("y")
            if x is None or y is None or (x == 0 and y == 0):
                continue
            touch_n[team] = touch_n.get(team, 0) + 1
            if 33.33 <= y <= 66.67:
                central_n[team] = central_n.get(team, 0) + 1

        seq, cur_cid = [], None
        for e in events:
            cid = e["contestantId"]
            if cid != cur_cid:
                if seq:
                    _record_seq(seq, cur_cid, team_name_by_cid, seq_lengths)
                seq, cur_cid = [e], cid
            else:
                seq.append(e)
        if seq:
            _record_seq(seq, cur_cid, team_name_by_cid, seq_lengths)

    avg_pass_dist = {t: pass_dist_sum[t] / pass_dist_n[t] for t in pass_dist_n if pass_dist_n[t] > 0}
    central_share = {t: central_n[t] / touch_n[t] * 100 for t in touch_n if touch_n.get(t)}
    avg_seq_len = {t: sum(v) / len(v) for t, v in seq_lengths.items() if v}
    diagonal_pct = {t: diag_n.get(t, 0) / op_pass_n[t] * 100 for t in op_pass_n if op_pass_n[t] > 0}

    teams = [t for t in team_name_by_cid.values()
             if t in avg_pass_dist and t in central_share and t in avg_seq_len and t in diagonal_pct]

    short_combo_rank = percentile_rank({t: -avg_pass_dist[t] for t in teams})
    central_rank = percentile_rank({t: central_share[t] for t in teams})
    retention_rank = percentile_rank({t: avg_seq_len[t] for t in teams})
    relationism_index = {t: (short_combo_rank[t] + central_rank[t] + retention_rank[t]) / 3 for t in teams}

    rows = [{"team": t, "relationism_index": round(relationism_index[t], 1),
             "diagonal_pass_pct": round(diagonal_pct[t], 2)} for t in teams]
    rows.sort(key=lambda r: r["relationism_index"], reverse=True)

    with open(os.path.join(OUT_DIR, "diagonal_vs_relationism.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["team", "relationism_index", "diagonal_pass_pct"])
        w.writeheader()
        w.writerows(rows)

    xs = [r["relationism_index"] for r in rows]
    ys = [r["diagonal_pass_pct"] for r in rows]
    n = len(rows)
    mean_x, mean_y = sum(xs) / n, sum(ys) / n
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    sx = sum((x - mean_x) ** 2 for x in xs) ** 0.5
    sy = sum((y - mean_y) ** 2 for y in ys) ** 0.5
    corr = cov / (sx * sy) if sx and sy else 0.0
    print(f"Pearson r (Relationism Index vs. diagonal pass %) = {corr:.3f}")
    for r in rows:
        print(f"  {r['team']:<32} idx={r['relationism_index']:>5.1f}  diag%={r['diagonal_pass_pct']:>5.2f}")

    top_diag = max(rows, key=lambda r: r["diagonal_pass_pct"])

    palette, cats = style.apply("light")
    fig = plt.figure(figsize=(9.5, 7.5))
    ax = fig.add_axes([0.11, 0.15, 0.82, 0.58])
    ax.scatter(xs, ys, s=70, color=cats[0], zorder=3)
    ax.scatter([top_diag["relationism_index"]], [top_diag["diagonal_pass_pct"]],
               s=110, color=palette["accent"], zorder=4)
    ax.annotate(top_diag["team"], xy=(top_diag["relationism_index"], top_diag["diagonal_pass_pct"]),
                xytext=(8, 6), textcoords="offset points", fontsize=9.5, fontweight="bold",
                color=palette["accent"])

    # trend line (least squares), muted -- illustrative, not a claimed model
    if sx:
        slope = cov / sum((x - mean_x) ** 2 for x in xs)
        intercept = mean_y - slope * mean_x
        x0, x1 = min(xs), max(xs)
        ax.plot([x0, x1], [slope * x0 + intercept, slope * x1 + intercept],
                color=palette["axis"], linewidth=1.2, linestyle=(0, (4, 3)), zorder=2)

    ax.set_xlabel("Relationism Index (0 = most positional, 100 = most relationist)")
    ax.set_ylabel("Diagonal pass % (of completed open-play passes)")

    direction = "more" if corr > 0 else "less"
    strength = ("little to no" if abs(corr) < 0.3 else
                "a weak" if abs(corr) < 0.5 else
                "a moderate" if abs(corr) < 0.7 else "a strong")
    components.header(
        fig, kicker="Relationism",
        title=f"Relationism shows {strength} link to diagonal passing (r = {corr:.2f})",
        dek=f"Relationism Index vs. diagonal pass share, {SEASON} (18 teams, {direction} diagonal at higher index)",
        palette=palette,
    )
    components.footer(fig, source=f"Opta/StatsPerform {SEASON}", palette=palette)

    out_path = os.path.join(OUT_DIR, "diagonal_vs_relationism.png")
    fig.savefig(out_path, facecolor=fig.get_facecolor(), dpi=150)
    print(f"Wrote {out_path}")


def _record_seq(seq, cid, team_name_by_cid, seq_lengths):
    team = team_name_by_cid.get(cid)
    if team is None:
        return
    n_passes = sum(1 for e in seq if e["typeId"] == 1)
    if n_passes < MIN_PASSES:
        return
    first, last = seq[0], seq[-1]
    duration_s = (minute_value(last) - minute_value(first)) * 60
    if not (MIN_DURATION_S <= duration_s <= MAX_DURATION_S):
        return
    seq_lengths.setdefault(team, []).append(n_passes)


if __name__ == "__main__":
    main()

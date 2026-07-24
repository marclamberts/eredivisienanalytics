"""
Has xG per corner gone up? League-wide Eredivisie, 2023/24-2025/26.

No pre-scored xG exists for the two older seasons (the Danger/ CSVs with
a trained xg column only cover 2025/26, and Model/model_xg.pkl needs a
custom class that isn't in this repo), so this fits a small, transparent
xG model from scratch -- logistic regression on shot distance, shot
angle, and header -- pooled across all three seasons so every season is
scored on the exact same scale. Penalties get a fixed xG (historical
conversion rate) rather than a distance/angle estimate, since the
location is always the same and the outcome depends on the kicker, not
geometry.

A corner is credited with a shot as before (Cross Models/
score_eredivisie_crosses.py, corner_routines.py): shot's qualifier 55
(RELATED_EVENT_ID) points back to a corner event's eventId. "xG per
corner" = sum of xG over all shots a corner directly created, divided by
the number of corners taken that season -- the average danger produced
per corner, not just whether it produced a shot at all.

Usage: python3 xg_per_corner_trend.py [out.png]
"""
import sys
import glob
import json
import math
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from housestyle import style, components

DATA_ROOT = Path(__file__).resolve().parents[1] / "Events"
SEASONS = ["2023-2024", "2024-2025", "2025-2026"]
SEASON_LABELS = {"2023-2024": "2023/24", "2024-2025": "2024/25", "2025-2026": "2025/26"}

CORNER_QID = 6
RELATED_EVENT_QID = 55
HEAD_QID = 15
PENALTY_QID = 9
SHOT_TYPES = {13, 14, 15, 16}
GOAL_TYPE = 16

X_SCALE, Y_SCALE = 1.05, 0.68          # Opta 0-100 units -> metres (105x68m pitch)
GOAL_Y_LO, GOAL_Y_HALF = 50 - 7.32 / 2 / Y_SCALE, 7.32 / 2 / Y_SCALE  # goal mouth in Opta y-units
PENALTY_XG = 0.79                        # fixed, ~league-average penalty conversion


def qids(e):
    return {q["qualifierId"] for q in e.get("qualifier", []) or []}


def qmap(e):
    return {q["qualifierId"]: q.get("value") for q in e.get("qualifier", []) or []}


def shot_geometry(x, y):
    """Distance (m) to goal centre and shot angle (rad) subtended by the goal mouth."""
    gx = 100.0
    dx = (gx - x) * X_SCALE
    post1_y = 50 - 7.32 / 2 / Y_SCALE
    post2_y = 50 + 7.32 / 2 / Y_SCALE
    dy1 = (post1_y - y) * Y_SCALE
    dy2 = (post2_y - y) * Y_SCALE
    a1 = math.atan2(dy1, dx)
    a2 = math.atan2(dy2, dx)
    angle = abs(a1 - a2)
    dist = math.hypot(dx, (dy1 + dy2) / 2)
    return dist, angle


def load_season_events(season):
    files = sorted(glob.glob(str(DATA_ROOT / season / "*.json")))
    matches = []
    for fn in files:
        with open(fn) as f:
            data = json.load(f)
        events = [e for e in data.get("event", []) if e.get("periodId") in (1, 2)]
        if events:
            matches.append(events)
    return matches


def extract_shots(matches, season):
    """Non-penalty shots (features + goal label) and corner-created shots (raw, unlabeled)."""
    rows, corner_shot_rows = [], []
    for events in matches:
        corner_event_ids = set()
        for e in events:
            if e.get("typeId") == 1 and CORNER_QID in qids(e):
                corner_event_ids.add(e.get("eventId"))

        n_corners = len(corner_event_ids)

        for e in events:
            if e.get("typeId") not in SHOT_TYPES:
                continue
            q = qmap(e)
            x, y = e.get("x"), e.get("y")
            if x is None or y is None:
                continue
            is_penalty = PENALTY_QID in q
            is_goal = 1 if e.get("typeId") == GOAL_TYPE else 0
            is_header = 1 if HEAD_QID in q else 0
            dist, angle = shot_geometry(x, y)

            rel = q.get(RELATED_EVENT_QID)
            from_corner = False
            if rel is not None:
                try:
                    from_corner = int(rel) in corner_event_ids
                except (TypeError, ValueError):
                    from_corner = False

            row = {"season": season, "dist": dist, "angle": angle, "header": is_header,
                   "penalty": is_penalty, "goal": is_goal, "from_corner": from_corner}
            rows.append(row)
            if from_corner:
                corner_shot_rows.append(row)

        corner_shot_rows_meta = n_corners
    return rows, corner_shot_rows


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else str(
        Path(__file__).resolve().parent / "15_xg_per_corner_trend_eredivisie.png")

    all_rows = []
    season_data = {}
    for season in SEASONS:
        matches = load_season_events(season)
        rows, corner_rows = extract_shots(matches, season)
        n_corners = sum(1 for events in matches for e in events
                        if e.get("typeId") == 1 and CORNER_QID in qids(e))
        season_data[season] = {"rows": rows, "corner_rows": corner_rows,
                                "n_matches": len(matches), "n_corners": n_corners}
        all_rows.extend(rows)

    # Pooled xG model: non-penalty shots, all three seasons together, so every
    # season is scored on one consistent scale.
    train = [r for r in all_rows if not r["penalty"]]
    Xtr = np.array([[r["dist"], r["angle"], r["header"]] for r in train])
    ytr = np.array([r["goal"] for r in train])
    model = LogisticRegression(max_iter=1000)
    model.fit(Xtr, ytr)
    print(f"Pooled xG model fit on {len(train)} non-penalty shots "
          f"({ytr.sum()} goals, {ytr.mean():.3f} conversion rate)")

    results = []
    for season in SEASONS:
        d = season_data[season]
        corner_shots = d["corner_rows"]
        xg_sum = 0.0
        for r in corner_shots:
            if r["penalty"]:
                xg_sum += PENALTY_XG
            else:
                p = model.predict_proba([[r["dist"], r["angle"], r["header"]]])[0, 1]
                xg_sum += p
        n_corners = d["n_corners"]
        results.append({
            "season": season, "matches": d["n_matches"], "corners": n_corners,
            "corner_shots": len(corner_shots), "xg_per_corner": xg_sum / n_corners if n_corners else 0.0,
        })
        print(f"{season}: {d['n_matches']} matches, {n_corners} corners, "
              f"{len(corner_shots)} corner-shots, xG/corner={results[-1]['xg_per_corner']:.4f}")

    make_plot(results, out_path)


def make_plot(rows, out_path):
    palette, cats = style.apply("light")

    fig = plt.figure(figsize=(8.4, 5.6))
    ax = fig.add_axes([0.11, 0.22, 0.82, 0.54])

    x = list(range(len(rows)))
    labels = [SEASON_LABELS[r["season"]] for r in rows]
    vals = [r["xg_per_corner"] for r in rows]

    ax.plot(x, vals, marker="o", linewidth=2.6, color=cats[0], zorder=3)
    components.highlight_lines(ax, accent_index=0, palette=palette)

    for xi, yi in zip(x, vals):
        ax.annotate(f"{yi:.3f}", (xi, yi), textcoords="offset points", xytext=(0, 12),
                    ha="center", fontsize=10, color=palette["ink_primary"], fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlim(-0.3, len(x) - 0.7)
    ax.set_ylim(0, max(vals) * 1.35)
    ax.set_ylabel("xG per corner taken")

    direction = "risen" if vals[-1] > vals[0] else ("fallen" if vals[-1] < vals[0] else "held steady")
    pct = (vals[-1] / vals[0] - 1) * 100 if vals[0] else 0

    components.header(
        fig,
        kicker="Eredivisie",
        title=f"xG per corner has {direction} since 2023/24" + (f" ({pct:+.0f}%)" if vals[0] else ""),
        dek="League-wide average xG generated by shots created directly from a corner, per corner taken",
        palette=palette,
    )
    components.footer(
        fig,
        source="Opta event data, Eredivisie 2023/24-2025/26",
        palette=palette,
    )
    fig.text(0.06, 0.020, "xG: pooled logistic regression on shot distance, angle and header; "
             "penalties fixed at 0.79", fontsize=7.6, color=palette["ink_muted"],
             family="sans-serif", ha="left", va="top")

    fig.savefig(out_path, dpi=200, facecolor=fig.get_facecolor())
    print("Saved:", out_path)


if __name__ == "__main__":
    main()

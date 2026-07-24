"""
Has xG per delivery gone up? League-wide Eredivisie, 2023/24-2025/26,
for three delivery types: corners, attacking free kicks, and open-play
crosses launched from the attacking third.

No pre-scored xG exists for the two older seasons (the Danger/ CSVs with
a trained xg column only cover 2025/26, and Model/model_xg.pkl needs a
custom class that isn't in this repo), so this fits a small, transparent
xG model from scratch -- logistic regression on shot distance, shot
angle, and header -- pooled across all three seasons so every season and
every delivery type is scored on the exact same scale. Penalties get a
fixed xG (historical conversion rate) rather than a distance/angle
estimate, since the location is always the same and the outcome depends
on the kicker, not geometry.

Delivery definitions (qualifier scheme matches Cross Models/
score_eredivisie_crosses.py and PSV Season Report/Scripts/
corner_routines.py):
  - corner: typeId 1 (pass), qualifier 6 (CORNER_TAKEN) present.
  - attacking free kick: typeId 1 (pass), qualifier 5 (FREEKICK_TAKEN)
    present, taken from the attacking third (x >= 66.67) -- OR a direct
    free-kick shot itself (typeId 13-16, qualifier 26), also from the
    attacking third. qualifier 5 alone fires on every free-kick restart
    including deep defensive knock-ons, so the attacking-third filter is
    what keeps this comparable to corners (always taken from the byline).
  - open-play cross, attacking third: typeId 1 (pass), qualifier 2
    (CROSS) present, without qualifier 5/6/160 (not a free kick, corner,
    or throw-in), taken from the attacking third (x >= 66.67).

A delivery is credited with a shot when the shot's qualifier 55
(RELATED_EVENT_ID) points back to that delivery event's eventId (direct
free-kick shots are the exception -- there the shot IS the delivery, no
assist link needed). "xG per delivery" = sum of xG over every shot a
delivery type directly produced, divided by the number of deliveries of
that type taken that season.

Usage: python3 xg_per_delivery_trend.py [out.png]
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

CROSS_QID, FREEKICK_QID, CORNER_QID, THROWIN_QID = 2, 5, 6, 160
DIRECT_FREEKICK_QID = 26
RELATED_EVENT_QID = 55
HEAD_QID = 15
PENALTY_QID = 9
SHOT_TYPES = {13, 14, 15, 16}
GOAL_TYPE = 16
ATTACKING_THIRD_X = 66.67

X_SCALE, Y_SCALE = 1.05, 0.68          # Opta 0-100 units -> metres (105x68m pitch)
PENALTY_XG = 0.79                        # fixed, ~league-average penalty conversion

DELIVERIES = ["corner", "freekick", "cross"]
DELIVERY_LABELS = {"corner": "Corners", "freekick": "Attacking free kicks",
                    "cross": "Open-play crosses (attacking third)"}
# Fixed, distinct color per line (not the usual single-accent/muted-rest
# treatment) -- the point of this chart is telling all three series apart
# at a glance. Accent (terracotta) still marks corners, the series the
# analysis started from; the other two get their own categorical hues
# rather than collapsing to one shared gray.
DELIVERY_COLOR_SLOT = {"corner": "accent", "freekick": 0, "cross": 2}


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


def shot_row(e, season, extra):
    q = qmap(e)
    x, y = e.get("x"), e.get("y")
    if x is None or y is None:
        return None
    dist, angle = shot_geometry(x, y)
    return {"season": season, "dist": dist, "angle": angle,
            "header": 1 if HEAD_QID in q else 0,
            "penalty": PENALTY_QID in q,
            "goal": 1 if e.get("typeId") == GOAL_TYPE else 0,
            **extra}


def extract_match(events, season):
    """Per-match delivery counts and the shots each delivery type produced."""
    corner_ids, freekick_ids, cross_ids = set(), set(), set()
    for e in events:
        if e.get("typeId") != 1:
            continue
        q = qids(e)
        x = e.get("x")
        if CORNER_QID in q:
            corner_ids.add(e.get("eventId"))
        elif FREEKICK_QID in q:
            if x is not None and x >= ATTACKING_THIRD_X:
                freekick_ids.add(e.get("eventId"))
        elif CROSS_QID in q and THROWIN_QID not in q:
            if x is not None and x >= ATTACKING_THIRD_X:
                cross_ids.add(e.get("eventId"))

    n_direct_fk = 0
    all_shots = []
    delivery_shots = {"corner": [], "freekick": [], "cross": []}

    for e in events:
        if e.get("typeId") not in SHOT_TYPES:
            continue
        q = qmap(e)
        row = shot_row(e, season, {})
        if row is None:
            continue
        all_shots.append(row)

        if DIRECT_FREEKICK_QID in q:
            x = e.get("x")
            if x is not None and x >= ATTACKING_THIRD_X:
                n_direct_fk += 1
                delivery_shots["freekick"].append(row)
            continue

        rel = q.get(RELATED_EVENT_QID)
        if rel is None:
            continue
        try:
            rel = int(rel)
        except (TypeError, ValueError):
            continue
        if rel in corner_ids:
            delivery_shots["corner"].append(row)
        elif rel in freekick_ids:
            delivery_shots["freekick"].append(row)
        elif rel in cross_ids:
            delivery_shots["cross"].append(row)

    counts = {"corner": len(corner_ids), "freekick": len(freekick_ids) + n_direct_fk,
              "cross": len(cross_ids)}
    return all_shots, delivery_shots, counts


def xg_of(row, model):
    if row["penalty"]:
        return PENALTY_XG
    return model.predict_proba([[row["dist"], row["angle"], row["header"]]])[0, 1]


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else str(
        Path(__file__).resolve().parent / "15_xg_per_delivery_trend_eredivisie.png")

    all_train_rows = []
    season_data = {}
    for season in SEASONS:
        matches = load_season_events(season)
        totals = {"matches": 0, "counts": {k: 0 for k in DELIVERIES},
                  "delivery_shots": {k: [] for k in DELIVERIES}}
        for events in matches:
            shots, delivery_shots, counts = extract_match(events, season)
            all_train_rows.extend(shots)
            totals["matches"] += 1
            for k in DELIVERIES:
                totals["counts"][k] += counts[k]
                totals["delivery_shots"][k].extend(delivery_shots[k])
        season_data[season] = totals

    # Pooled xG model: non-penalty shots, all three seasons together, every
    # season and delivery type scored on one consistent scale.
    train = [r for r in all_train_rows if not r["penalty"]]
    Xtr = np.array([[r["dist"], r["angle"], r["header"]] for r in train])
    ytr = np.array([r["goal"] for r in train])
    model = LogisticRegression(max_iter=1000)
    model.fit(Xtr, ytr)
    print(f"Pooled xG model fit on {len(train)} non-penalty shots "
          f"({ytr.sum()} goals, {ytr.mean():.3f} conversion rate)")

    results = {k: [] for k in DELIVERIES}
    for season in SEASONS:
        d = season_data[season]
        print(f"{season}: {d['matches']} matches")
        for k in DELIVERIES:
            n = d["counts"][k]
            xg_sum = sum(xg_of(r, model) for r in d["delivery_shots"][k])
            val = xg_sum / n if n else 0.0
            results[k].append(val)
            print(f"  {k}: n={n}, shots={len(d['delivery_shots'][k])}, xG/delivery={val:.4f}")

    make_plot(results, out_path)


def make_plot(results, out_path):
    palette, cats = style.apply("light")

    fig = plt.figure(figsize=(9.4, 6.6))
    ax = fig.add_axes([0.10, 0.28, 0.84, 0.50])

    x = list(range(len(SEASONS)))
    labels = [SEASON_LABELS[s] for s in SEASONS]

    # Largest overall % change is the finding -> drives the headline (color
    # is fixed per delivery type below, independent of which one leads).
    changes = {k: (results[k][-1] / results[k][0] - 1) if results[k][0] else 0 for k in DELIVERIES}
    lead = max(changes, key=lambda k: abs(changes[k]))

    line_colors = {}
    for k in DELIVERIES:
        slot = DELIVERY_COLOR_SLOT[k]
        color = palette["accent"] if slot == "accent" else cats[slot]
        line_colors[k] = color
        ax.plot(x, results[k], marker="o", linewidth=2.4, color=color,
                label=DELIVERY_LABELS[k], zorder=3)

    for k in DELIVERIES:
        color = line_colors[k]
        for xi, yi in zip(x, results[k]):
            ax.annotate(f"{yi:.3f}", (xi, yi), textcoords="offset points", xytext=(0, 9),
                        ha="center", fontsize=8.5, color=color,
                        fontweight="bold" if k == lead else "normal")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlim(-0.3, len(x) - 0.7)
    ymax = max(v for vals in results.values() for v in vals)
    ax.set_ylim(0, ymax * 1.3)
    ax.set_ylabel("xG per delivery taken")
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, -0.14), ncol=1, frameon=False,
              fontsize=9.5, handlelength=1.6)

    lead_label = DELIVERY_LABELS[lead].lower()
    direction = "risen" if changes[lead] > 0 else "fallen"
    components.header(
        fig,
        kicker="Eredivisie",
        title=f"xG per delivery has {direction} most for {lead_label} ({changes[lead]*100:+.0f}%)",
        dek="League-wide average xG per corner, per attacking free kick, and per open-play "
            "cross from the attacking third",
        palette=palette,
    )
    components.footer(
        fig,
        source="Opta event data, Eredivisie 2023/24-2025/26",
        palette=palette,
    )
    fig.text(0.06, 0.020, "xG: pooled logistic regression on shot distance, angle and header; "
             "penalties fixed at 0.79. Free kicks/crosses restricted to attacking-third origin.",
             fontsize=7.6, color=palette["ink_muted"], family="sans-serif", ha="left", va="top")

    fig.savefig(out_path, dpi=200, facecolor=fig.get_facecolor())
    print("Saved:", out_path)


if __name__ == "__main__":
    main()

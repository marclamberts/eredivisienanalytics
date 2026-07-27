"""
xG For vs xG Against by match for a given Eerste Divisie 2025/26 team,
with a rolling-average trend line for each.

xG per shot comes from a distance/angle logistic regression fitted on
~34k shots across four Eredivisie seasons (2022/23-2025/26, see
train_xg_model() below) -- the Eerste Divisie itself has no pre-existing
shot-xg model in this repo, so this fits a simple, standard geometry-based
model and applies it to the team's Eerste Divisie shots. Penalties are
assigned the empirical penalty conversion rate instead of the geometry
model (all penalties share the same distance/angle, so the regression
would otherwise blend them into ordinary open-play shots). Own goals are
excluded from shot xG entirely, in line with standard practice.

Usage: python3 ado_den_haag_xg_trend.py ["HFC ADO Den Haag"] [out.png]
Team name must match the filename spelling, e.g. "VVV Venlo", "SV Roda JC".
"""
import glob
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from housestyle import style, components

REPO_ROOT = Path(__file__).resolve().parent.parent
EREDIVISIE_DIR = REPO_ROOT / "Eredivisie Events"
EERSTE_DIVISIE_DIR = REPO_ROOT / "Eerste Divisie Events" / "Eerste Divisie 2025-2026"
ROLL_WINDOW = 5

SHOT_TYPES = {13, 14, 15, 16}
PENALTY_QID = 9
OWN_GOAL_QID = 28
HEAD_QID = 15

PITCH_LENGTH = 105.0
PITCH_WIDTH = 68.0
GOAL_WIDTH = 7.32


def qmap(e):
    return {q["qualifierId"]: q.get("value") for q in e.get("qualifier", [])}


def shot_geometry(x, y):
    x_m = x / 100.0 * PITCH_LENGTH
    y_m = y / 100.0 * PITCH_WIDTH
    post1 = (PITCH_LENGTH, PITCH_WIDTH / 2 - GOAL_WIDTH / 2)
    post2 = (PITCH_LENGTH, PITCH_WIDTH / 2 + GOAL_WIDTH / 2)
    d1 = math.hypot(post1[0] - x_m, post1[1] - y_m)
    d2 = math.hypot(post2[0] - x_m, post2[1] - y_m)
    distance = math.hypot(PITCH_LENGTH - x_m, PITCH_WIDTH / 2 - y_m)
    cos_val = (d1 ** 2 + d2 ** 2 - GOAL_WIDTH ** 2) / (2 * d1 * d2)
    cos_val = max(-1.0, min(1.0, cos_val))
    angle = math.acos(cos_val)
    return distance, angle


def extract_shots(files):
    """All shots (own goals excluded) from a list of match json files,
    tagged with distance/angle/header/penalty/is_goal."""
    rows = []
    for fn in files:
        with open(fn) as f:
            data = json.load(f)
        for e in data["event"]:
            if e.get("typeId") not in SHOT_TYPES:
                continue
            q = qmap(e)
            if OWN_GOAL_QID in q:
                continue
            x, y = e.get("x"), e.get("y")
            if x is None or y is None:
                continue
            distance, angle = shot_geometry(x, y)
            rows.append({
                "match_file": fn, "contestant_id": e.get("contestantId"),
                "distance": distance, "angle": angle,
                "is_header": 1 if HEAD_QID in q else 0,
                "is_penalty": 1 if PENALTY_QID in q else 0,
                "is_goal": 1 if e.get("typeId") == 16 else 0,
            })
    return pd.DataFrame(rows)


def train_xg_model():
    files = sorted(glob.glob(f"{EREDIVISIE_DIR}/[0-9][0-9][0-9][0-9]-[0-9][0-9][0-9][0-9]/*.json"))
    print(f"Training xG model on {len(files)} Eredivisie matches...")
    df = extract_shots(files)
    open_play = df[df["is_penalty"] == 0]
    print(f"  {len(df)} shots total, {len(open_play)} non-penalty for model fit")

    clf = LogisticRegression()
    X = open_play[["distance", "angle", "is_header"]].values
    y = open_play["is_goal"].values
    clf.fit(X, y)

    penalties = df[df["is_penalty"] == 1]
    penalty_xg = float(penalties["is_goal"].mean()) if len(penalties) else 0.76
    print(f"  penalty conversion rate (n={len(penalties)}): {penalty_xg:.3f}")
    return clf, penalty_xg, len(df)


def score_shots(df, clf, penalty_xg):
    xg = np.zeros(len(df))
    is_pen = df["is_penalty"].values == 1
    if (~is_pen).any():
        xg[~is_pen] = clf.predict_proba(df.loc[~is_pen, ["distance", "angle", "is_header"]].values)[:, 1]
    xg[is_pen] = penalty_xg
    return xg


def build_team_map(files):
    import collections
    team_cid_sets = collections.defaultdict(list)
    for fn in files:
        m = re.match(r"\d{4}-\d{2}-\d{2}_(.+) - (.+)\.json$", fn.split("/")[-1])
        if not m:
            continue
        home, away = m.group(1), m.group(2)
        with open(fn) as f:
            data = json.load(f)
        cids = set(e["contestantId"] for e in data["event"] if "contestantId" in e)
        team_cid_sets[home].append(cids)
        team_cid_sets[away].append(cids)
    team_to_cid = {}
    for team, sets in team_cid_sets.items():
        inter = set.intersection(*sets)
        if len(inter) == 1:
            team_to_cid[team] = next(iter(inter))
    return team_to_cid


def opponent_name(fname, team_name):
    m = re.match(r"\d{4}-\d{2}-\d{2}_(.+) - (.+)\.json$", fname)
    home, away = m.group(1), m.group(2)
    return away if team_name in home else home


def rolling_mean(values, window):
    out = []
    for i in range(len(values)):
        lo = max(0, i - window + 1)
        chunk = values[lo:i + 1]
        out.append(sum(chunk) / len(chunk))
    return out


def collect_match_rows(clf, penalty_xg, team_name):
    files = sorted(glob.glob(f"{EERSTE_DIVISIE_DIR}/*.json"))
    team_to_cid = build_team_map(files)
    if team_name not in team_to_cid:
        print(f"Team '{team_name}' not found. Options: {sorted(team_to_cid)}")
        sys.exit(1)
    cid = team_to_cid[team_name]

    rows = []
    for fn in files:
        base = fn.split("/")[-1]
        m = re.match(r"(\d{4}-\d{2}-\d{2})_(.+) - (.+)\.json$", base)
        date, home, away = m.group(1), m.group(2), m.group(3)
        if team_name not in (home, away):
            continue
        shots = extract_shots([fn])
        if shots.empty:
            continue
        shots["xg"] = score_shots(shots, clf, penalty_xg)
        xg_for = shots.loc[shots["contestant_id"] == cid, "xg"].sum()
        xg_against = shots.loc[shots["contestant_id"] != cid, "xg"].sum()
        rows.append({
            "date": date, "opponent": opponent_name(base, team_name),
            "venue": "H" if home == team_name else "A",
            "xg_for": xg_for, "xg_against": xg_against,
        })
    rows.sort(key=lambda r: r["date"])
    return rows


def make_plot(rows, out_path, n_train_shots, team_name):
    n = len(rows)
    xs = list(range(1, n + 1))
    xgf = [r["xg_for"] for r in rows]
    xga = [r["xg_against"] for r in rows]
    roll_f = rolling_mean(xgf, ROLL_WINDOW)
    roll_a = rolling_mean(xga, ROLL_WINDOW)
    avg_f, avg_a = sum(xgf) / n, sum(xga) / n

    first_half_diff = sum(roll_f[:n // 2]) / (n // 2) - sum(roll_a[:n // 2]) / (n // 2)
    second_half_diff = sum(roll_f[n // 2:]) / (n - n // 2) - sum(roll_a[n // 2:]) / (n - n // 2)
    improving = second_half_diff > first_half_diff

    palette, cats = style.apply("dark")

    fig = plt.figure(figsize=(11, 6.8))
    ax = fig.add_axes([0.08, 0.16, 0.86, 0.58])

    ax.plot(xs, xgf, color=palette["axis"], alpha=0.5, linewidth=1, marker="o",
            markersize=3.5, zorder=2)
    ax.plot(xs, xga, color=palette["axis"], alpha=0.5, linewidth=1, marker="o",
            markersize=3.5, zorder=2, linestyle="--")
    ax.plot(xs, roll_f, linewidth=2.6, zorder=4, label=f"xG For ({ROLL_WINDOW}-match avg)")
    ax.plot(xs, roll_a, linewidth=2.6, zorder=3, label=f"xG Against ({ROLL_WINDOW}-match avg)")
    components.highlight_lines(ax, accent_index=2, palette=palette)
    ax.get_lines()[3].set_color(palette["ink_secondary"])
    ax.get_lines()[3].set_linewidth(1.8)

    ax.set_xlim(0.5, n + 0.5)
    ax.set_xticks(xs)
    ax.set_xticklabels([str(i) for i in xs], fontsize=7.5)
    ax.set_ylabel("xG per match", fontsize=10.5, color=palette["ink_secondary"])
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper left", frameon=False, fontsize=9.5, labelcolor=palette["ink_primary"])

    trend_word = "improved" if improving else "faded"
    components.header(
        fig,
        kicker=team_name,
        title=f"Attacking output {trend_word} relative to xG conceded over the season",
        dek=f"Eerste Divisie 2025/26 · rolling {ROLL_WINDOW}-match average · "
            f"season avg {avg_f:.2f} xG for / {avg_a:.2f} xG against per match",
        palette=palette,
    )
    components.footer(
        fig,
        source="Opta event data",
        note=f"xG: distance/angle model, {n_train_shots:,} Eredivisie shots 2022/23-2025/26",
        palette=palette,
    )

    fig.savefig(out_path, dpi=200, facecolor=fig.get_facecolor())
    print("Saved:", out_path)
    print(f"n_matches={n} avg_xgf={avg_f:.2f} avg_xga={avg_a:.2f}")


if __name__ == "__main__":
    team_name = sys.argv[1] if len(sys.argv) > 1 else "HFC ADO Den Haag"
    default_out = REPO_ROOT / "Eerste Divisie Events" / f"{team_name.lower().replace(' ', '_')}_xg_trend.png"
    out = sys.argv[2] if len(sys.argv) > 2 else str(default_out)
    clf, penalty_xg, n_train_shots = train_xg_model()
    rows = collect_match_rows(clf, penalty_xg, team_name)
    if not rows:
        print("No matches found for", team_name)
        sys.exit(1)
    make_plot(rows, out, n_train_shots, team_name)

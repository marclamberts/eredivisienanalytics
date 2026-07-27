"""
xG For vs xG Against by match for a given Eerste Divisie 2025/26 team,
with a rolling-average trend line for each.

xG per shot comes from the Waltzing Analytics xG Model
(Model/model_xg.pkl + Model/model_meta.pkl) -- a calibrated XGBoost
classifier (CalibratedXGB = XGBClassifier + IsotonicRegression calibrator)
fit on distance/angle/y_sym plus qualifier-derived shot context (body
part, phase of play, assist type). No model is trained or inferred here;
this script only rebuilds the exact 18 features the pickled model expects
(per model_meta.pkl's qualifier_mapping) and calls it. Penalties use the
model's own fixed pen_xg from the metadata rather than the classifier
(all penalties share the same distance/angle/y_sym, so the classifier
was not trained to discriminate among them). Own goals are excluded from
shot xG entirely, in line with standard practice.

Usage: python3 ado_den_haag_xg_trend.py ["HFC ADO Den Haag"] [out.png]
Team name must match the filename spelling, e.g. "VVV Venlo", "SV Roda JC".
"""
import glob
import json
import math
import re
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from housestyle import style, components

REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = REPO_ROOT / "Model"
EERSTE_DIVISIE_DIR = REPO_ROOT / "Eerste Divisie Events" / "Eerste Divisie 2025-2026"
ROLL_WINDOW = 5

SHOT_TYPES = {13, 14, 15, 16}
OWN_GOAL_QID = 28

PITCH_LENGTH = 105.0
PITCH_WIDTH = 68.0
GOAL_WIDTH = 7.32


class CalibratedXGB:
    """Stub matching the pickled class's attribute layout (clf, iso) so
    joblib.load can restore it -- no training logic, just the shape the
    original object was saved with."""

    def predict_proba(self, X):
        raw = self.clf.predict_proba(X)[:, 1]
        return self.iso.predict(raw)


def load_xg_model():
    import __main__
    __main__.CalibratedXGB = CalibratedXGB
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = joblib.load(MODEL_DIR / "model_xg.pkl")
    meta = joblib.load(MODEL_DIR / "model_meta.pkl")
    return model, meta


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
    y_sym = abs(y_m - PITCH_WIDTH / 2)
    return distance, angle, y_sym


def extract_shots(files, qm):
    """All shots (own goals excluded) from a list of match json files,
    with the exact feature set model_xg.pkl was trained on."""
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
            distance, angle, y_sym = shot_geometry(x, y)
            rows.append({
                "match_file": fn, "contestant_id": e.get("contestantId"),
                "distance": distance, "angle": angle, "y_sym": y_sym,
                "is_header": int(qm["HEADER"] in q),
                "is_right_foot": int(qm["RIGHT_FOOT"] in q),
                "is_left_foot": int(qm["LEFT_FOOT"] in q),
                "is_other_body_part": int(qm["OTHER_BODY_PART"] in q),
                "is_volley": int(qm["VOLLEY"] in q),
                "is_one_on_one": int(qm["ONE_ON_ONE"] in q),
                "is_fast_break": int(qm["FAST_BREAK"] in q),
                "is_from_corner": int(qm["FROM_CORNER"] in q),
                "is_free_kick": int(qm["DIRECT_FREE_KICK"] in q),
                "is_set_piece": int(qm["SET_PIECE"] in q),
                "is_throw_in_set_piece": int(qm["THROW_IN_SET_PIECE"] in q),
                "is_open_play": int(qm["REGULAR_PLAY"] in q),
                "is_assisted": int(qm["ASSISTED"] in q),
                "is_intentional_assist": int(qm["INTENTIONAL_ASSIST"] in q),
                "is_individual_play": int(qm["INDIVIDUAL_PLAY"] in q),
                "is_penalty": int(qm["PENALTY"] in q),
                "is_goal": 1 if e.get("typeId") == 16 else 0,
            })
    return pd.DataFrame(rows)


def score_shots(df, model, meta):
    feature_cols = meta["features"]
    xg = np.zeros(len(df))
    is_pen = df["is_penalty"].values == 1
    if (~is_pen).any():
        X = df.loc[~is_pen, feature_cols]
        xg[~is_pen] = model.predict_proba(X)
    xg[is_pen] = meta["pen_xg"]
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


def collect_match_rows(model, meta, qm, team_name):
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
        shots = extract_shots([fn], qm)
        if shots.empty:
            continue
        shots["xg"] = score_shots(shots, model, meta)
        xg_for = shots.loc[shots["contestant_id"] == cid, "xg"].sum()
        xg_against = shots.loc[shots["contestant_id"] != cid, "xg"].sum()
        rows.append({
            "date": date, "opponent": opponent_name(base, team_name),
            "venue": "H" if home == team_name else "A",
            "xg_for": xg_for, "xg_against": xg_against,
        })
    rows.sort(key=lambda r: r["date"])
    return rows


def make_plot(rows, out_path, team_name):
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
        note="xG: Waltzing Analytics xG Model (calibrated XGBoost)",
        palette=palette,
    )

    fig.savefig(out_path, dpi=200, facecolor=fig.get_facecolor())
    print("Saved:", out_path)
    print(f"n_matches={n} avg_xgf={avg_f:.2f} avg_xga={avg_a:.2f}")


if __name__ == "__main__":
    team_name = sys.argv[1] if len(sys.argv) > 1 else "HFC ADO Den Haag"
    default_out = REPO_ROOT / "Eerste Divisie Events" / f"{team_name.lower().replace(' ', '_')}_xg_trend.png"
    out = sys.argv[2] if len(sys.argv) > 2 else str(default_out)

    model, meta = load_xg_model()
    qm = meta["qualifier_mapping"]
    rows = collect_match_rows(model, meta, qm, team_name)
    if not rows:
        print("No matches found for", team_name)
        sys.exit(1)
    make_plot(rows, out, team_name)

"""
Expected Box Entry (xBE): an alternative to xA that values a pass by how
likely it was to deliver the ball into the opponent's penalty box, rather
than by the quality of a shot the receiver may or may not take. xA credits
the passer for the finisher's shot; xBE isolates the passer's own
contribution -- getting the ball into the danger zone -- independent of
what happens after.

This is a genuinely new model trained from scratch on this season's own
Eredivisie event data (not reverse-engineered from an external model, and
not a zone-transition/state-value framework like xT, VAEP or EPV): a
gradient-boosted classifier predicting P(pass into the box succeeds) from
pre-pass trajectory features (start/end location, length, angle, delivery
type, phase of play).

Universe: open-play passes (typeId 1) that start outside the penalty box
and are aimed inside it (end coordinates from qualifiers 140/141), i.e.
box-entry attempts, whether completed or not. Excludes free kicks, corners
and throw-ins (qualifiers 5, 6, 107, 160) -- set-piece delivery is a
different skill and already covered by the repo's Cross Models.

Usage: python3 build_box_entry_model.py [out_dir]
"""
import sys
import re
import glob
import json
import math
import collections
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss

DATA_DIR = "/home/user/eredivisienanalytics/Events"

BOX_X = 83.0
BOX_Y_LO, BOX_Y_HI = 21.1, 78.9
SET_PIECE_QIDS = {5, 6, 107, 160}

CROSS_QID, THROUGH_QID, PULL_BACK_QID = 2, 3, 195
HEAD_QID, RIGHT_QID, LEFT_QID = 15, 20, 72
FAST_BREAK_QID = 23
PASS_END_X_QID, PASS_END_Y_QID = 140, 141
LENGTH_QID, ANGLE_QID = 212, 213

X_SCALE, Y_SCALE = 1.05, 0.68

# five-lane pitch division (same convention as crosses_by_zone.py): each
# lane is 20% of the pitch width.
LANE = 100 / 5

NUMERIC_COLS = ["x", "y", "end_x", "end_y", "length", "angle", "minute",
                "dist_to_goal_start", "dist_to_goal_end", "byline_proximity",
                "lateral_shift", "is_cross", "is_through_ball", "is_cutback"]
CATEGORICAL_COLS = ["body_part", "phase", "origin_channel", "entry_channel", "period"]
FEATURE_COLS = NUMERIC_COLS + CATEGORICAL_COLS


def qmap(e):
    return {q["qualifierId"]: q.get("value") for q in e.get("qualifier", []) or []}


def in_box(x, y):
    return x >= BOX_X and BOX_Y_LO <= y <= BOX_Y_HI


def channel(y):
    if y < LANE or y > 100 - LANE:
        return "wide"
    if y < 2 * LANE or y > 100 - 2 * LANE:
        return "halfspace"
    return "central"


def build_team_map(files):
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


def extract_box_entry_attempts(files):
    """Open-play passes starting outside the box, aimed inside it."""
    team_to_cid = build_team_map(files)
    cid_to_team = {v: k for k, v in team_to_cid.items()}

    rows = []
    for fn in files:
        base = fn.split("/")[-1]
        with open(fn) as f:
            data = json.load(f)

        for e in data["event"]:
            if e.get("typeId") != 1:
                continue
            q = qmap(e)
            if SET_PIECE_QIDS & set(q.keys()):
                continue

            x, y = e.get("x"), e.get("y")
            if x is None or y is None or in_box(x, y):
                continue
            end_x, end_y = q.get(PASS_END_X_QID), q.get(PASS_END_Y_QID)
            try:
                end_x, end_y = float(end_x), float(end_y)
            except (TypeError, ValueError):
                continue
            if not in_box(end_x, end_y):
                continue

            dx_m = (end_x - x) * X_SCALE
            dy_m = (end_y - y) * Y_SCALE
            length = q.get(LENGTH_QID)
            angle = q.get(ANGLE_QID)
            try:
                length = float(length)
            except (TypeError, ValueError):
                length = math.hypot(dx_m, dy_m)
            try:
                angle = float(angle)
            except (TypeError, ValueError):
                angle = math.atan2(dy_m, dx_m)

            dist_to_goal_start = math.hypot(100 - x, 50 - y)
            dist_to_goal_end = math.hypot(100 - end_x, 50 - end_y)
            byline_proximity = 100 - end_x
            lateral_shift = abs(end_y - y)
            body_part = "head" if HEAD_QID in q else ("left_foot" if LEFT_QID in q else
                        ("right_foot" if RIGHT_QID in q else "other"))
            phase = "fast_break" if FAST_BREAK_QID in q else "regular"
            period = int(e.get("periodId"))
            minute = e.get("timeMin", 0) + e.get("timeSec", 0) / 60.0

            outcome = 1 if e.get("outcome") == 1 else 0
            cid = e.get("contestantId")
            team = cid_to_team.get(cid, cid)

            rows.append({
                "match_file": base, "team": team, "player": e.get("playerName", "?"),
                "period": period, "minute": minute,
                "x": x, "y": y, "end_x": end_x, "end_y": end_y,
                "length": length, "angle": angle,
                "dist_to_goal_start": dist_to_goal_start, "dist_to_goal_end": dist_to_goal_end,
                "byline_proximity": byline_proximity, "lateral_shift": lateral_shift,
                "is_cross": 1 if CROSS_QID in q else 0,
                "is_through_ball": 1 if THROUGH_QID in q else 0,
                "is_cutback": 1 if PULL_BACK_QID in q else 0,
                "body_part": body_part, "phase": phase,
                "origin_channel": channel(y), "entry_channel": channel(end_y),
                "outcome": outcome,
            })

    return pd.DataFrame(rows), team_to_cid


def build_pipeline():
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLS),
        ("num", "passthrough", NUMERIC_COLS),
    ])
    clf = HistGradientBoostingClassifier(
        max_depth=4, max_iter=300, learning_rate=0.05,
        l2_regularization=1.0, random_state=42,
    )
    return Pipeline([("pre", pre), ("clf", clf)])


def evaluate(pipe, df, test_idx):
    X_test = df.loc[test_idx, FEATURE_COLS]
    y_test = df.loc[test_idx, "outcome"]
    p = pipe.predict_proba(X_test)[:, 1]
    return {
        "n_test": int(len(test_idx)),
        "auc": float(roc_auc_score(y_test, p)),
        "brier": float(brier_score_loss(y_test, p)),
        "log_loss": float(log_loss(y_test, p)),
        "actual_rate": float(y_test.mean()),
        "mean_predicted": float(p.mean()),
    }


def main():
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent
    models_dir = out_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(glob.glob(f"{DATA_DIR}/*.json"))
    print(f"Parsing {len(files)} Eredivisie matches for box-entry attempts...")
    df, _ = extract_box_entry_attempts(files)
    print(f"Extracted {len(df)} open-play box-entry attempts across "
          f"{df['team'].nunique()} teams, {df['match_file'].nunique()} matches.")
    print(f"Overall completion rate: {df['outcome'].mean():.3f}")

    # split by match, not by row, so a match's attempts don't leak across
    # train/test (attempts within a match share correlated context).
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(splitter.split(df, groups=df["match_file"]))
    train_idx, test_idx = df.index[train_idx], df.index[test_idx]

    pipe = build_pipeline()
    pipe.fit(df.loc[train_idx, FEATURE_COLS], df.loc[train_idx, "outcome"])

    metrics = evaluate(pipe, df, test_idx)
    print("\nHeld-out match evaluation:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    # refit on the full season before shipping the model that will score it
    pipe.fit(df[FEATURE_COLS], df["outcome"])
    model_path = models_dir / "model_box_entry.pkl"
    joblib.dump(pipe, model_path)
    print("\nSaved:", model_path)

    meta = {
        "created_at": datetime.now().isoformat(),
        "source": DATA_DIR,
        "n_attempts": int(len(df)),
        "method": "Expected Box Entry (xBE): HistGradientBoostingClassifier predicting "
                  "P(open-play pass into the penalty box succeeds) from pre-pass "
                  "trajectory features. Alternative to xA -- values the pass itself "
                  "(reaching the box) rather than the receiver's shot quality. Not a "
                  "zone-transition (xT) or state-value (VAEP/EPV) framework.",
        "universe": "Open-play passes (typeId 1) starting outside the box (x<83 or "
                    "y outside [21.1, 78.9]) with intended end point (qualifiers "
                    "140/141) inside the box. Excludes free kicks/corners/throw-ins "
                    "(qualifiers 5, 6, 107, 160).",
        "feature_cols": FEATURE_COLS,
        "qualifier_mapping": {
            "2": "CROSS", "3": "THROUGH_BALL", "5": "FREEKICK_TAKEN", "6": "CORNER_TAKEN",
            "15": "HEAD", "20": "RIGHT_FOOT", "23": "FAST_BREAK", "72": "LEFT_FOOT",
            "107": "THROW_IN", "140": "PASS_END_X", "141": "PASS_END_Y",
            "160": "THROWIN_SETPIECE", "195": "PULL_BACK", "212": "LENGTH_M", "213": "ANGLE_RAD",
        },
        "held_out_evaluation": metrics,
    }
    meta_path = out_dir / "box_entry_model_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print("Saved:", meta_path)


if __name__ == "__main__":
    main()

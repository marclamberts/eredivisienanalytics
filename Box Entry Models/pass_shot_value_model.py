"""
Pass Shot Value (PSV): a regression companion to xBE. Where xBE is a
classifier's P(pass into the box succeeds), PSV regresses each open-play
box-entry attempt directly against a continuous target -- the xG of the
shot (if any) that the same team takes within the next few actions -- so
the output is a single continuous number in [0, 1] per pass: how much
scoring value this specific ball-into-the-box actually produced, not just
whether it arrived.

Target: for each attempt, look at the next LOOKAHEAD events in the match
(chronological order). If the same team takes a shot in that window, the
target is that shot's xG (from the Danger models, joined on the raw event
id); otherwise 0. This is a short lookahead heuristic, not a full
possession-chain/state-value model -- deliberately simpler than
VAEP/EPV -- so a shot a long buildup later than LOOKAHEAD events won't be
credited back to this pass.

Usage: python3 pass_shot_value_model.py [out_dir]
"""
import sys
import csv
import glob
import json
import math
import os
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import r2_score, mean_absolute_error

from build_box_entry_model import (
    BOX_X, BOX_Y_LO, BOX_Y_HI, SET_PIECE_QIDS, CROSS_QID, THROUGH_QID,
    PULL_BACK_QID, HEAD_QID, RIGHT_QID, LEFT_QID, FAST_BREAK_QID,
    PASS_END_X_QID, PASS_END_Y_QID, LENGTH_QID, ANGLE_QID, X_SCALE, Y_SCALE,
    NUMERIC_COLS, CATEGORICAL_COLS, FEATURE_COLS, qmap, in_box, channel, build_team_map,
)

DATA_DIR = "/home/user/eredivisienanalytics/Events"
DANGER_DIR = "/home/user/eredivisienanalytics/Danger"
SHOT_TYPES = {13, 14, 15, 16}
LOOKAHEAD = 5


def load_shot_xg_map(base_name):
    csv_path = os.path.join(DANGER_DIR, base_name[:-5] + "_danger_models.csv")
    m = {}
    if os.path.exists(csv_path):
        with open(csv_path, newline="") as f:
            for row in csv.DictReader(f):
                m[int(row["event_id"])] = float(row["xg"])
    return m


def extract_scored_pass_attempts(files):
    """Same open-play box-entry universe as build_box_entry_model, plus a
    continuous shot_xg regression target per attempt."""
    team_to_cid = build_team_map(files)
    cid_to_team = {v: k for k, v in team_to_cid.items()}

    rows = []
    for fn in files:
        base = fn.split("/")[-1]
        with open(fn) as f:
            data = json.load(f)
        events = [e for e in data["event"] if e.get("periodId") in (1, 2)]
        events.sort(key=lambda e: (e["periodId"], e.get("timeMin", 0), e.get("timeSec", 0), e.get("eventId", 0)))
        xg_map = load_shot_xg_map(base)

        for idx, e in enumerate(events):
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

            shot_xg = 0.0
            for k in range(idx + 1, min(idx + 1 + LOOKAHEAD, len(events))):
                nxt = events[k]
                if nxt.get("contestantId") != cid:
                    continue
                if nxt.get("typeId") in SHOT_TYPES:
                    shot_xg = xg_map.get(nxt.get("id"), 0.0)
                    break

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
                "outcome": outcome, "shot_xg": shot_xg,
            })

    return pd.DataFrame(rows)


def build_pipeline():
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLS),
        ("num", "passthrough", NUMERIC_COLS),
    ])
    reg = HistGradientBoostingRegressor(
        max_depth=4, max_iter=300, learning_rate=0.05,
        l2_regularization=1.0, random_state=42,
    )
    return Pipeline([("pre", pre), ("reg", reg)])


def main():
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent
    models_dir = out_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(glob.glob(f"{DATA_DIR}/*.json"))
    print(f"Parsing {len(files)} Eredivisie matches for box-entry attempts + shot-linked xG...")
    df = extract_scored_pass_attempts(files)
    print(f"Extracted {len(df)} attempts. Non-zero shot_xg target on "
          f"{(df['shot_xg'] > 0).mean() * 100:.1f}% of attempts. Mean target: {df['shot_xg'].mean():.4f}")

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(splitter.split(df, groups=df["match_file"]))
    train_idx, test_idx = df.index[train_idx], df.index[test_idx]

    pipe = build_pipeline()
    pipe.fit(df.loc[train_idx, FEATURE_COLS], df.loc[train_idx, "shot_xg"])

    pred_test = np.clip(pipe.predict(df.loc[test_idx, FEATURE_COLS]), 0, 1)
    y_test = df.loc[test_idx, "shot_xg"]
    metrics = {
        "n_test": int(len(test_idx)),
        "r2": float(r2_score(y_test, pred_test)),
        "mae": float(mean_absolute_error(y_test, pred_test)),
        "mean_actual": float(y_test.mean()),
        "mean_predicted": float(pred_test.mean()),
    }
    print("\nHeld-out match evaluation:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    pipe.fit(df[FEATURE_COLS], df["shot_xg"])
    model_path = models_dir / "model_pass_shot_value.pkl"
    joblib.dump(pipe, model_path)
    print("\nSaved:", model_path)

    df["pred_psv"] = np.clip(pipe.predict(df[FEATURE_COLS]), 0, 1)
    csv_path = out_dir / "eredivisie_pass_shot_value_scored.csv"
    df.to_csv(csv_path, index=False)
    print("Saved:", csv_path)

    player_summary = df.groupby(["player", "team"]).agg(
        attempts=("pred_psv", "size"),
        total_psv=("pred_psv", "sum"),
        actual_shot_xg=("shot_xg", "sum"),
    ).reset_index().sort_values("total_psv", ascending=False)
    player_csv = out_dir / "pass_shot_value_player_summary.csv"
    player_summary.to_csv(player_csv, index=False)
    print("Saved:", player_csv)

    meta = {
        "created_at": datetime.now().isoformat(),
        "source": DATA_DIR,
        "n_attempts": int(len(df)),
        "method": "Pass Shot Value (PSV): HistGradientBoostingRegressor predicting a continuous "
                  "target in [0, 1] per open-play box-entry attempt -- the xG of the shot the same "
                  "team takes within the next "
                  f"{LOOKAHEAD} events, or 0 if none. A regression companion to the xBE classifier: "
                  "xBE asks 'did it arrive', PSV asks 'how much shot value did it produce'.",
        "lookahead_events": LOOKAHEAD,
        "feature_cols": FEATURE_COLS,
        "held_out_evaluation": metrics,
    }
    meta_path = out_dir / "pass_shot_value_model_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print("Saved:", meta_path)


if __name__ == "__main__":
    main()

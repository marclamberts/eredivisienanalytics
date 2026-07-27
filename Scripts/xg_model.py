"""
Shared plumbing for scoring Opta shot events with the Waltzing Analytics
xG Model (Model/model_xg.pkl + Model/model_meta.pkl) -- a calibrated
XGBoost classifier (CalibratedXGB = XGBClassifier + IsotonicRegression
calibrator) fit on distance/angle/y_sym plus qualifier-derived shot
context (body part, phase of play, assist type). No model is trained or
inferred here; this module only rebuilds the exact 18 features the
pickled model expects (per model_meta.pkl's qualifier_mapping) and calls
it. Penalties use the model's own fixed pen_xg from the metadata rather
than the classifier (all penalties share the same distance/angle/y_sym,
so the classifier was not trained to discriminate among them). Own goals
are excluded from shot xG entirely, in line with standard practice.
"""
import json
import math
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = REPO_ROOT / "Model"

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
                "player": e.get("playerName", "?"), "player_id": e.get("playerId"),
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

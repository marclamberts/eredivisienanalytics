"""
Loader + scorer for the repo's real trained xG model (Model/model_xg.pkl),
to use in place of the simple own distance+angle geometric formula
duplicated across the CZ Events report folders' match_data.py files.

model_xg.pkl was written with joblib.dump (not plain pickle -- it embeds
raw numpy buffers via joblib.numpy_pickle.NumpyArrayWrapper, which is why
a bare pickle.load() fails with "invalid load key" errors; joblib.load()
handles it). It unpickles to a CalibratedXGB instance with two attributes:
  - clf: a fitted xgboost.sklearn.XGBClassifier (18 features, see below)
  - iso: a fitted sklearn.isotonic.IsotonicRegression that recalibrates
    clf's raw predict_proba output into the final xG probability.
The original CalibratedXGB class definition isn't present anywhere in
this repo (checked), so this module defines a minimal stand-in with the
same two attributes -- pickle/joblib restores instance state via
__dict__, not __init__, so the stand-in only needs to exist as a target
class; its predict() method below reconstructs the standard
calibrated-classifier pattern (raw clf.predict_proba, then isotonic
transform) rather than relying on a pickled method.

Feature order and semantics confirmed against Model/model_meta.pkl and
by inspecting the fitted trees' split-value ranges directly
(clf.get_booster().trees_to_dataframe()):
  - distance: metres to goal (range ~2.7-90 in training data) -- same
    definition as this repo's shot_xg()'s own `dist` term.
  - angle: degrees, goal-mouth subtended angle (range ~6.6-143.5) --
    same definition as this repo's shot_angle_deg().
  - y_sym: abs(y_metres - 34.0), metres from the pitch's central axis
    (range ~0.15-33.35, consistent with a 68m-wide pitch halved).
  - the remaining 15 features are all boolean 0/1 flags, one per Opta
    MA3 qualifierId per model_meta.pkl's qualifier_mapping (see
    QUALIFIER_FOR_FEATURE below) -- each is just "does this shot event
    carry that qualifierId".
  - Penalties are NOT run through the model: model_meta.pkl's
    'pen_xg': 0.79 is used as a fixed value instead (qualifier 9,
    PENALTY), matching this repo's convention of not re-deriving
    settled, well-known probabilities from a general shot model.

Usage:
    from Model.xg_model import XGModel
    model = XGModel.load()
    xg = model.score(distance_m=18.4, angle_deg=45.2, y_m=30.1,
                      qualifier_ids={15, 20, 22})  # header? no; right foot, open play
"""
import os

import joblib
import numpy as np

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(MODEL_DIR, "model_xg.pkl")

FEATURE_ORDER = [
    "distance", "angle", "y_sym", "is_header", "is_right_foot", "is_left_foot",
    "is_other_body_part", "is_volley", "is_one_on_one", "is_fast_break",
    "is_from_corner", "is_free_kick", "is_set_piece", "is_throw_in_set_piece",
    "is_open_play", "is_assisted", "is_intentional_assist", "is_individual_play",
]

# Opta MA3 qualifierId each boolean feature is derived from (per
# model_meta.pkl's qualifier_mapping). is_volley maps only to the
# dedicated VOLLEY qualifier (108), not OVERHEAD (109) or HALF_VOLLEY
# (110) -- meta only names "is_volley" among the core features, with no
# separate is_overhead/is_half_volley, so those two aren't folded in here.
QUALIFIER_FOR_FEATURE = {
    "is_header": 15,
    "is_right_foot": 20,
    "is_left_foot": 72,
    "is_other_body_part": 21,
    "is_volley": 108,
    "is_one_on_one": 89,
    "is_fast_break": 23,
    "is_from_corner": 25,
    "is_free_kick": 26,
    "is_set_piece": 24,
    "is_throw_in_set_piece": 160,
    "is_open_play": 22,
    "is_assisted": 29,
    "is_intentional_assist": 154,
    "is_individual_play": 215,
}
Q_PENALTY = 9
PENALTY_XG = 0.79  # model_meta.pkl's pen_xg -- penalties bypass the model


class CalibratedXGB:
    """Stand-in for the original (unavailable) training-time class --
    only needs to match the attribute names joblib.load restores state
    into (`clf`, `iso`); predict() below implements the standard
    raw-then-isotonic-calibrated pattern this naming implies."""

    def predict_proba_calibrated(self, X):
        raw = self.clf.predict_proba(X)[:, 1]
        return self.iso.predict(raw)


class XGModel:
    def __init__(self, calibrated):
        self._model = calibrated

    @classmethod
    def load(cls):
        # joblib.load needs CalibratedXGB resolvable as __main__.CalibratedXGB,
        # matching how the object was originally pickled.
        import __main__
        if not hasattr(__main__, "CalibratedXGB"):
            __main__.CalibratedXGB = CalibratedXGB
        obj = joblib.load(MODEL_PATH)
        return cls(obj)

    def score(self, distance_m, angle_deg, y_m, qualifier_ids):
        """distance_m/angle_deg: this shot's distance (m) and goal-mouth
        subtended angle (degrees) -- same convention as this repo's
        shot_xg(). y_m: shot location's y coordinate in metres (0-68).
        qualifier_ids: set/iterable of this event's Opta qualifierIds."""
        if Q_PENALTY in qualifier_ids:
            return PENALTY_XG
        qids = set(qualifier_ids)
        y_sym = abs(y_m - 34.0)
        row = [distance_m, angle_deg, y_sym] + [
            1.0 if QUALIFIER_FOR_FEATURE[feat] in qids else 0.0
            for feat in FEATURE_ORDER[3:]
        ]
        X = np.array([row], dtype=float)
        return float(self._model.predict_proba_calibrated(X)[0])

    def score_batch(self, rows):
        """rows: list of (distance_m, angle_deg, y_m, qualifier_ids) tuples.
        Returns a list of xG floats, penalties handled the same as score()."""
        out = [None] * len(rows)
        model_idx, model_rows = [], []
        for i, (distance_m, angle_deg, y_m, qualifier_ids) in enumerate(rows):
            if Q_PENALTY in qualifier_ids:
                out[i] = PENALTY_XG
                continue
            qids = set(qualifier_ids)
            y_sym = abs(y_m - 34.0)
            model_rows.append([distance_m, angle_deg, y_sym] + [
                1.0 if QUALIFIER_FOR_FEATURE[feat] in qids else 0.0
                for feat in FEATURE_ORDER[3:]
            ])
            model_idx.append(i)
        if model_rows:
            X = np.array(model_rows, dtype=float)
            preds = self._model.predict_proba_calibrated(X)
            for i, p in zip(model_idx, preds):
                out[i] = float(p)
        return out

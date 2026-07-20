"""
Pass difficulty factoring in disruption (pressure-aware) - Eredivisie 2025/26
==============================================================================
build_disruption_model.py trains a *pressure-free* baseline: P(pass completes)
from pass geometry alone, deliberately excluding defensive context so a
defensive action's disruption_score can be measured as a deviation from an
unpressured expectation.

This script asks the complementary question: what is the pass actually worth
accounting for the disruption around it? It adds a "pressure" feature -- how
much nearby opponent defensive activity preceded the pass -- and retrains a
second model, P(complete | geometry + pressure). difficulty = 1 - P(complete).
Comparing the two models' difficulty for the same pass shows how much of its
difficulty is attributable to being contested vs. simply a hard pass to hit.

Pressure feature (non-leaky: only looks *before* the pass, never at events
caused by the pass itself):
  - pressure_count: opponent Tackle/Interception/Clearance/Aerial/Recovery/
    Dispossessed events in the same period, in the 20s before the pass,
    within 25m of the pass's start location.
  - pressure_min_dist: distance (m) to the nearest such action (40m = none
    found in the window, i.e. an open, uncontested build-up situation).

Caveat: this is a coarse proxy. There's no tracking data, so "pressure" here
means *discrete defensive actions were recently recorded nearby*, not "an
opponent was standing next to the passer" -- most passes in a match happen
without any contest ever getting logged close by (build-up circulation among
unmarked teammates dominates raw pass counts), so pressure_count>0 fires for
a minority of passes. Report both the whole-sample average (diluted by the
uncontested majority) and the pressured-subset average, which isolates the
effect where the signal actually fires.

Usage: python3 player_pressure_difficulty.py "<player name>"
       python3 player_pressure_difficulty.py            # picks a random
                                                          # player (>=300 passes)
Outputs:
  Disruption/player_difficulty_<player>.csv
  Model/model_pass_difficulty_pressure.pkl (+ _meta.pkl)
"""
import glob
import os
import random
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss
from sklearn.model_selection import train_test_split

import build_disruption_model as bdm

ROOT = bdm.ROOT
MODEL_DIR = bdm.MODEL_DIR
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_DIR = os.path.join(OUT_DIR, "CSV")

PRESSURE_LOOKBACK_S = 20.0
PRESSURE_DIST_MAX = 25.0
NO_PRESSURE_DIST = 40.0

PRESSURE_FEATURES = ["pressure_count", "pressure_min_dist"]
ALL_FEATURES = bdm.FEATURE_COLS + PRESSURE_FEATURES


def add_pressure_features(pass_df, def_df):
    pass_df = pass_df.reset_index(drop=True)
    pcount = np.zeros(len(pass_df))
    pmind = np.full(len(pass_df), NO_PRESSURE_DIST)

    for match_file, pgroup in pass_df.groupby("match_file"):
        dgroup = def_df[def_df["match_file"] == match_file]
        if dgroup.empty:
            continue
        for period_id, pg in pgroup.groupby("period_id"):
            dg = dgroup[dgroup["period_id"] == period_id]
            if dg.empty:
                continue
            for team_id, pgt in pg.groupby("contestant_id"):
                opp = dg[dg["contestant_id"] != team_id]
                if opp.empty:
                    continue
                ot = opp["t"].to_numpy()
                ox = opp["x"].to_numpy()
                oy = opp["y"].to_numpy()
                pt = pgt["t"].to_numpy()
                px = pgt["start_x"].to_numpy()
                py = pgt["start_y"].to_numpy()
                dt = pt[:, None] - ot[None, :]
                dist = np.hypot(px[:, None] - ox[None, :], py[:, None] - oy[None, :])
                window = (dt > 0) & (dt <= PRESSURE_LOOKBACK_S) & (dist <= PRESSURE_DIST_MAX)
                cnt = window.sum(axis=1)
                dist_masked = np.where(window, dist, np.inf)
                mind = dist_masked.min(axis=1)
                mind[np.isinf(mind)] = NO_PRESSURE_DIST
                pcount[pgt.index.to_numpy()] = cnt
                pmind[pgt.index.to_numpy()] = mind

    pass_df["pressure_count"] = pcount
    pass_df["pressure_min_dist"] = pmind
    return pass_df


def train_pressure_model(pass_df):
    X = pass_df[ALL_FEATURES]
    y = pass_df["outcome"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    model = HistGradientBoostingClassifier(
        max_depth=6, learning_rate=0.06, max_iter=300,
        l2_regularization=1.0, random_state=42)
    model.fit(X_train, y_train)

    p_test = model.predict_proba(X_test)[:, 1]
    metrics = {
        "n_train": len(X_train), "n_test": len(X_test),
        "test_auc": roc_auc_score(y_test, p_test),
        "test_brier": brier_score_loss(y_test, p_test),
        "test_logloss": log_loss(y_test, p_test),
        "base_rate": float(y.mean()),
    }
    print("Pressure-aware completion model:", metrics)

    model.fit(X, y)
    return model, metrics


def main():
    args = [a for a in sys.argv[1:] if a]
    player_name = args[0] if args else None

    files = sorted(glob.glob(os.path.join(bdm.DATA_DIR, "*.json")))
    print(f"Found {len(files)} match files")

    cid_to_team = bdm.build_global_cid_to_team(files)
    all_pass_rows, all_def_rows = [], []
    for fn in files:
        basename, events, team_map = bdm.load_match(fn, cid_to_team)
        all_pass_rows.extend(bdm.extract_passes(basename, events, team_map))
        all_def_rows.extend(bdm.extract_defensive_actions(basename, events, team_map))
    pass_df = pd.DataFrame(all_pass_rows)
    def_df = pd.DataFrame(all_def_rows)
    print(f"Extracted {len(pass_df)} passes, {len(def_df)} defensive actions")

    if player_name is None:
        counts = pass_df.groupby("player_name").size()
        eligible = counts[counts >= 300]
        player_name = random.choice(list(eligible.index))
        print(f"No player given -> randomly picked: {player_name} ({eligible[player_name]} passes)")

    print("Computing pressure features (opponent defensive activity preceding each pass)...")
    pass_df = add_pressure_features(pass_df, def_df)

    baseline_model = joblib.load(os.path.join(MODEL_DIR, "model_pass_completion.pkl"))
    pass_df["baseline_completion_prob"] = baseline_model.predict_proba(pass_df[bdm.FEATURE_COLS])[:, 1]
    pass_df["baseline_difficulty"] = 1 - pass_df["baseline_completion_prob"]

    pressure_model, metrics = train_pressure_model(pass_df)
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(pressure_model, os.path.join(MODEL_DIR, "model_pass_difficulty_pressure.pkl"))
    joblib.dump({
        "features": ALL_FEATURES,
        "metrics": metrics,
        "pressure_lookback_s": PRESSURE_LOOKBACK_S,
        "pressure_dist_max": PRESSURE_DIST_MAX,
        "no_pressure_dist": NO_PRESSURE_DIST,
        "description": "P(pass completes) including a non-leaky pressure feature "
                        "(opponent defensive activity in the 8s before the pass, "
                        "within 15m). difficulty = 1 - predicted probability. "
                        "Compare to model_pass_completion.pkl (geometry-only) to "
                        "see how much of a pass's difficulty is attributable to "
                        "disruption/pressure vs. pure geometry.",
    }, os.path.join(MODEL_DIR, "model_pass_difficulty_pressure.pkl".replace(".pkl", "_meta.pkl")))

    pass_df["pressure_completion_prob"] = pressure_model.predict_proba(pass_df[ALL_FEATURES])[:, 1]
    pass_df["pressure_difficulty"] = 1 - pass_df["pressure_completion_prob"]
    pass_df["disruption_difficulty_delta"] = (
        pass_df["pressure_difficulty"] - pass_df["baseline_difficulty"])

    player_df = pass_df[pass_df["player_name"] == player_name].copy()
    if player_df.empty:
        print(f"No passes found for '{player_name}'. Check spelling / exact Opta name.")
        return
    player_df["time_sec"] = (player_df["t"] - player_df["time_min"] * 60).astype(int)

    out_cols = ["match_file", "event_id", "period_id", "time_min", "time_sec",
                "team_name", "start_x", "start_y", "end_x", "end_y", "length",
                "is_long_ball", "is_cross", "is_through_ball", "is_free_kick",
                "is_corner", "pressure_count", "pressure_min_dist", "outcome",
                "baseline_completion_prob", "baseline_difficulty",
                "pressure_completion_prob", "pressure_difficulty",
                "disruption_difficulty_delta"]
    safe_name = player_name.replace(" ", "_").replace(".", "")
    os.makedirs(CSV_DIR, exist_ok=True)
    out_path = os.path.join(CSV_DIR, f"player_difficulty_{safe_name}.csv")
    player_df[out_cols].to_csv(out_path, index=False)
    print(f"Wrote {out_path} ({len(player_df)} passes)")

    n = len(player_df)
    completed = player_df["outcome"].mean()
    pressured = player_df[player_df["pressure_count"] > 0]
    unpressured = player_df[player_df["pressure_count"] == 0]
    print(f"\n=== {player_name} -- {n} passes, {completed:.1%} completed ===")
    print(f"Mean baseline difficulty (geometry only):          {player_df['baseline_difficulty'].mean():.3f}")
    print(f"Mean pressure-aware difficulty (whole sample):     {player_df['pressure_difficulty'].mean():.3f}")
    print(f"Mean extra difficulty from disruption (whole sample, dilutes toward 0): "
          f"{player_df['disruption_difficulty_delta'].mean():+.3f}")
    print(f"Passes with recent nearby defensive activity:      "
          f"{len(pressured)} / {n} ({len(pressured)/n:.1%})")
    if len(pressured):
        print(f"  -> mean extra difficulty on THOSE passes:        "
              f"{pressured['disruption_difficulty_delta'].mean():+.3f}")
        print(f"  -> completion rate under pressure:               {pressured['outcome'].mean():.1%}")
    print(f"  -> completion rate with no nearby defensive activity: {unpressured['outcome'].mean():.1%}")

    print("\nTop 5 hardest passes attempted (by pressure-aware difficulty), with outcome:")
    hardest = player_df.sort_values("pressure_difficulty", ascending=False).head(5)
    print(hardest[["match_file", "time_min", "pressure_difficulty", "pressure_count",
                    "outcome"]].to_string(index=False))

    print("\nTop 5 completed passes that were hardest under pressure (best executions):")
    completed_hard = player_df[player_df["outcome"] == 1].sort_values(
        "pressure_difficulty", ascending=False).head(5)
    print(completed_hard[["match_file", "time_min", "pressure_difficulty",
                           "pressure_count", "length"]].to_string(index=False))


if __name__ == "__main__":
    main()

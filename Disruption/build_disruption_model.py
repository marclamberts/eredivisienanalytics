"""
Pass Disruption Model - Eredivisie 2025/26
===========================================
Answers: "how much would a given defensive action disrupt a pass?"

Approach (pass-completion-probability-drop):
  1. Train a baseline model P(pass completes) on pass-intrinsic features only
     (start/end location, length, angle, pass type, period, minute) using every
     pass attempt in the season -- this captures how likely a pass like this is
     to succeed in general, deliberately *excluding* any notion of defensive
     pressure so it represents an unpressured baseline.
  2. Link each defensive action (Tackle/Interception/Clearance/Aerial Duel/Ball
     Recovery/Dispossessed) to the specific failed opponent pass it broke up,
     using a chronological nearest-in-time-then-space greedy match within a
     tight window (same period, |dt| <= 4s, distance to the pass's start/end
     segment <= 15m). Each defensive action can claim at most one pass.
  3. disruption_score for a linked defensive action = the model's predicted
     completion probability for the pass it broke up. A defender who stops a
     pass the model rated 85% likely to succeed gets far more credit than one
     who mistimes a challenge on a Hail Mary ball the model already rated at 10%.

Data caveat: this Opta feed has no freeze-frame/opponent-tracking data, so
there is no direct "blocked pass" event flag to key off. "blocked" vs.
"intercepted" below is therefore a *derived* spatial label on the six
well-established action types (not a separate raw event type): a link is
called "blocked" when the defensive action lands within 5m of the passer's
release point, "intercepted" when it lands further along the pass's path,
and "recovery"/"dispossession" for ball-recovery/dispossessed events.

Usage: python3 build_disruption_model.py
Outputs:
  Disruption/<match>_disruption_models.csv   (one row per defensive action)
  Disruption/all_eredivisie_disruption_models.csv
  Disruption/disruption_player_summary.csv
  Disruption/disruption_team_summary.csv
  Model/model_pass_completion.pkl
  Model/model_pass_completion_meta.pkl
"""
import glob
import json
import math
import os
import re

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss
from sklearn.model_selection import train_test_split

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "Events")
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(ROOT, "Model")

X_SCALE, Y_SCALE = 1.05, 0.68           # Opta 0-100 units -> metres (105 x 68 pitch)
GOAL_X, GOAL_Y = 105.0, 34.0            # opponent goal, in metres

T_PASS = 1
DEFENSIVE_TYPES = {7: "Tackle", 8: "Interception", 12: "Clearance",
                   44: "Aerial Duel", 49: "Ball Recovery", 50: "Dispossessed"}

Q_HEAD, Q_CROSS, Q_THROUGH, Q_FREE_KICK, Q_CORNER = 1, 2, 3, 5, 6
Q_LONG_BALL = 107
Q_END_X, Q_END_Y = 140, 141
Q_LENGTH, Q_ANGLE = 212, 213

LINK_TIME_WINDOW = (-1.0, 4.0)   # seconds, defensive action relative to the failed pass
LINK_DIST_MAX = 15.0             # metres
BLOCK_DIST_MAX = 5.0             # metres from pass origin -> classed as "blocked" not "intercepted"

FEATURE_COLS = [
    "start_x", "start_y", "end_x", "end_y", "length", "angle",
    "progressive", "start_dist_to_goal", "end_dist_to_goal",
    "start_third", "end_third", "is_long_ball", "is_cross", "is_through_ball",
    "is_free_kick", "is_corner", "is_head", "period_id", "time_min",
]

PREFIX_RE = re.compile(r"^(CSD|CD|CS|SD)\s+")


def clean_name(name):
    return PREFIX_RE.sub("", name)


def qmap(e):
    return {q["qualifierId"]: q.get("value") for q in e.get("qualifier", []) or []}


def to_m(x, y):
    return x * X_SCALE, y * Y_SCALE


def event_time(e):
    return e["timeMin"] * 60 + e["timeSec"]


def build_global_cid_to_team(files):
    """Map contestantId -> team name using every match, not just one.

    A single match's event order does not reliably tell you which
    contestantId is home vs away (Opta does not always emit the home team's
    events first), so guessing per match mislabels some fixtures. Instead,
    collect the set of contestantIds seen in each match a team name appears
    in (home or away) and intersect across all of that team's matches: the
    team's own id is the one constant across every match, so a single-team-id
    intersection reliably resolves the true name for that id.
    """
    team_cid_sets = {}
    for fn in files:
        basename = os.path.basename(fn).replace(".json", "")
        m = re.match(r"\d{4}-\d{2}-\d{2}_(.+) - (.+)$", basename)
        if not m:
            continue
        home_name, away_name = m.group(1), m.group(2)
        with open(fn) as f:
            data = json.load(f)
        cids = set(e["contestantId"] for e in data["event"] if e.get("contestantId"))
        team_cid_sets.setdefault(home_name, []).append(cids)
        team_cid_sets.setdefault(away_name, []).append(cids)

    cid_to_team = {}
    for team, sets in team_cid_sets.items():
        inter = set.intersection(*sets) if sets else set()
        if len(inter) == 1:
            cid_to_team[next(iter(inter))] = team
    return cid_to_team


def pass_features(e, q):
    x, y = to_m(e["x"], e["y"])
    try:
        ex_raw, ey_raw = float(q[Q_END_X]), float(q[Q_END_Y])
    except (KeyError, TypeError, ValueError):
        return None
    ex, ey = to_m(ex_raw, ey_raw)
    try:
        length = float(q[Q_LENGTH])
    except (KeyError, TypeError, ValueError):
        length = math.hypot(ex - x, ey - y)
    try:
        angle = float(q[Q_ANGLE])
    except (KeyError, TypeError, ValueError):
        angle = math.atan2(ey - y, ex - x)

    start_dist_goal = math.hypot(GOAL_X - x, GOAL_Y - y)
    end_dist_goal = math.hypot(GOAL_X - ex, GOAL_Y - ey)

    def third(px):
        if px < 35:
            return 0
        if px < 70:
            return 1
        return 2

    return {
        "start_x": x, "start_y": y, "end_x": ex, "end_y": ey,
        "length": length, "angle": angle,
        "progressive": start_dist_goal - end_dist_goal,
        "start_dist_to_goal": start_dist_goal, "end_dist_to_goal": end_dist_goal,
        "start_third": third(x), "end_third": third(ex),
        "is_long_ball": int(Q_LONG_BALL in q), "is_cross": int(Q_CROSS in q),
        "is_through_ball": int(Q_THROUGH in q), "is_free_kick": int(Q_FREE_KICK in q),
        "is_corner": int(Q_CORNER in q), "is_head": int(Q_HEAD in q),
        "period_id": e["periodId"], "time_min": e["timeMin"],
    }


def load_match(fn, cid_to_team=None):
    with open(fn) as f:
        data = json.load(f)
    basename = os.path.basename(fn).replace(".json", "")
    events = data["event"]
    cid_to_team = cid_to_team or {}
    return basename, events, cid_to_team


def extract_passes(basename, events, team_map):
    rows = []
    for e in events:
        if e["typeId"] != T_PASS:
            continue
        q = qmap(e)
        feats = pass_features(e, q)
        if feats is None:
            continue
        row = dict(feats)
        row.update({
            "match_file": basename,
            "event_id": e["eventId"],
            "id": e["id"],
            "contestant_id": e["contestantId"],
            "team_name": team_map.get(e["contestantId"], e["contestantId"]),
            "player_id": e.get("playerId"),
            "player_name": e.get("playerName"),
            "outcome": int(e.get("outcome", 0)),
            "t": event_time(e),
        })
        rows.append(row)
    return rows


def extract_defensive_actions(basename, events, team_map):
    rows = []
    for e in events:
        if e["typeId"] not in DEFENSIVE_TYPES:
            continue
        if e.get("x") is None or e.get("y") is None:
            continue
        x, y = to_m(e["x"], e["y"])
        rows.append({
            "match_file": basename,
            "event_id": e["eventId"],
            "id": e["id"],
            "type_id": e["typeId"],
            "action_type": DEFENSIVE_TYPES[e["typeId"]],
            "contestant_id": e["contestantId"],
            "team_name": team_map.get(e["contestantId"], e["contestantId"]),
            "player_id": e.get("playerId"),
            "player_name": e.get("playerName"),
            "x": x, "y": y,
            "period_id": e["periodId"],
            "time_min": e["timeMin"], "time_sec": e["timeSec"],
            "t": event_time(e),
        })
    return rows


def point_to_segment_dist(px, py, ax, ay, bx, by):
    abx, aby = bx - ax, by - ay
    seg_len2 = abx * abx + aby * aby
    if seg_len2 == 0:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * abx + (py - ay) * aby) / seg_len2))
    cx, cy = ax + t * abx, ay + t * aby
    return math.hypot(px - cx, py - cy)


def link_disruptions(pass_df, def_df):
    """Greedy chronological nearest-in-time-then-space 1:1 matching per match/period."""
    pass_df = pass_df.reset_index(drop=True)
    def_df = def_df.reset_index(drop=True)
    def_df["used"] = False
    def_df["linked_pass_idx"] = -1

    for (match_file, period_id), pgroup in pass_df[pass_df["outcome"] == 0].groupby(
            ["match_file", "period_id"]):
        dgroup = def_df[(def_df["match_file"] == match_file) & (def_df["period_id"] == period_id)]
        if dgroup.empty:
            continue
        for pidx, prow in pgroup.sort_values("t").iterrows():
            cands = dgroup[(~def_df.loc[dgroup.index, "used"]) &
                           (def_df.loc[dgroup.index, "contestant_id"] != prow["contestant_id"]) &
                           (def_df.loc[dgroup.index, "t"] - prow["t"] >= LINK_TIME_WINDOW[0]) &
                           (def_df.loc[dgroup.index, "t"] - prow["t"] <= LINK_TIME_WINDOW[1])]
            if cands.empty:
                continue
            dists = cands.apply(
                lambda r: point_to_segment_dist(r["x"], r["y"], prow["start_x"], prow["start_y"],
                                                 prow["end_x"], prow["end_y"]), axis=1)
            cands = cands.assign(dist=dists)
            cands = cands[cands["dist"] <= LINK_DIST_MAX]
            if cands.empty:
                continue
            dt = (cands["t"] - prow["t"]).abs()
            cands = cands.assign(dt=dt)
            best = cands.sort_values(["dt", "dist"]).iloc[0]
            def_df.loc[best.name, "used"] = True
            def_df.loc[best.name, "linked_pass_idx"] = pidx

    return def_df


def train_completion_model(pass_df):
    X = pass_df[FEATURE_COLS]
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
    print("Pass completion model:", metrics)

    # refit on all data for the final artifact
    model.fit(X, y)
    return model, metrics


def main():
    import sys
    files = sorted(glob.glob(os.path.join(DATA_DIR, "*.json")))
    if len(sys.argv) > 1:
        files = files[: int(sys.argv[1])]
    print(f"Found {len(files)} match files in {DATA_DIR}")

    cid_to_team = build_global_cid_to_team(files)
    print(f"Resolved {len(cid_to_team)} team names from {len(files)} matches")

    all_pass_rows, all_def_rows = [], []
    for fn in files:
        basename, events, team_map = load_match(fn, cid_to_team)
        all_pass_rows.extend(extract_passes(basename, events, team_map))
        all_def_rows.extend(extract_defensive_actions(basename, events, team_map))

    pass_df = pd.DataFrame(all_pass_rows)
    def_df = pd.DataFrame(all_def_rows)
    print(f"Extracted {len(pass_df)} passes ({pass_df['outcome'].mean():.1%} completed), "
          f"{len(def_df)} defensive actions")

    model, metrics = train_completion_model(pass_df)
    pass_df["pred_completion_prob"] = model.predict_proba(pass_df[FEATURE_COLS])[:, 1]

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, os.path.join(MODEL_DIR, "model_pass_completion.pkl"))
    joblib.dump({
        "features": FEATURE_COLS,
        "metrics": metrics,
        "x_scale": X_SCALE, "y_scale": Y_SCALE,
        "goal": (GOAL_X, GOAL_Y),
        "link_time_window": LINK_TIME_WINDOW,
        "link_dist_max": LINK_DIST_MAX,
        "block_dist_max": BLOCK_DIST_MAX,
        "defensive_types": DEFENSIVE_TYPES,
        "description": "P(pass completes) baseline model; disruption_score for a "
                        "defensive action = this model's predicted completion "
                        "probability for the specific opponent pass it broke up.",
    }, os.path.join(MODEL_DIR, "model_pass_completion_meta.pkl"))
    print("Saved Model/model_pass_completion.pkl + model_pass_completion_meta.pkl")

    linked_def = link_disruptions(pass_df, def_df)

    linked_def["linked"] = linked_def["linked_pass_idx"] >= 0
    disruption_score = np.full(len(linked_def), np.nan)
    link_type = np.array([None] * len(linked_def), dtype=object)
    linked_pass_cols = {c: np.array([None] * len(linked_def), dtype=object)
                        for c in ["linked_pass_event_id", "linked_pass_player",
                                  "linked_pass_team", "linked_pass_start_x",
                                  "linked_pass_start_y", "linked_pass_end_x",
                                  "linked_pass_end_y"]}

    for i, row in linked_def[linked_def["linked"]].iterrows():
        prow = pass_df.loc[row["linked_pass_idx"]]
        disruption_score[i] = prow["pred_completion_prob"]
        dist_to_start = math.hypot(row["x"] - prow["start_x"], row["y"] - prow["start_y"])
        if row["action_type"] in ("Ball Recovery", "Dispossessed"):
            link_type[i] = "recovery" if row["action_type"] == "Ball Recovery" else "dispossession"
        elif dist_to_start <= BLOCK_DIST_MAX:
            link_type[i] = "blocked"
        else:
            link_type[i] = "intercepted"
        linked_pass_cols["linked_pass_event_id"][i] = prow["event_id"]
        linked_pass_cols["linked_pass_player"][i] = prow["player_name"]
        linked_pass_cols["linked_pass_team"][i] = prow["team_name"]
        linked_pass_cols["linked_pass_start_x"][i] = prow["start_x"]
        linked_pass_cols["linked_pass_start_y"][i] = prow["start_y"]
        linked_pass_cols["linked_pass_end_x"][i] = prow["end_x"]
        linked_pass_cols["linked_pass_end_y"][i] = prow["end_y"]

    linked_def["disruption_score"] = disruption_score
    linked_def["link_type"] = link_type
    for c, v in linked_pass_cols.items():
        linked_def[c] = v

    out_cols = ["match_file", "event_id", "period_id", "time_min", "time_sec",
                "contestant_id", "team_name", "player_id", "player_name",
                "type_id", "action_type", "x", "y", "linked", "link_type",
                "disruption_score", "linked_pass_event_id", "linked_pass_team",
                "linked_pass_player", "linked_pass_start_x", "linked_pass_start_y",
                "linked_pass_end_x", "linked_pass_end_y"]
    out_df = linked_def[out_cols].copy()

    for match_file, mgroup in out_df.groupby("match_file"):
        mgroup.to_csv(os.path.join(OUT_DIR, f"{match_file}_disruption_models.csv"), index=False)
    out_df.to_csv(os.path.join(OUT_DIR, "all_eredivisie_disruption_models.csv"), index=False)
    print(f"Wrote {out_df['match_file'].nunique()} per-match CSVs + aggregate "
          f"({len(out_df)} defensive actions, {int(out_df['linked'].sum())} linked "
          f"= {out_df['linked'].mean():.1%})")

    linked_only = out_df[out_df["linked"]]

    player_summary = linked_only.groupby(["player_id", "player_name", "team_name"]).agg(
        matches=("match_file", "nunique"),
        actions_linked=("disruption_score", "size"),
        total_disruption=("disruption_score", "sum"),
        mean_disruption=("disruption_score", "mean"),
        blocked=("link_type", lambda s: (s == "blocked").sum()),
        intercepted=("link_type", lambda s: (s == "intercepted").sum()),
        recoveries=("link_type", lambda s: (s == "recovery").sum()),
        dispossessions=("link_type", lambda s: (s == "dispossession").sum()),
    ).reset_index()
    player_summary["disruption_per90"] = (
        player_summary["total_disruption"] / player_summary["matches"])
    player_summary = player_summary.sort_values("total_disruption", ascending=False)
    player_summary.to_csv(os.path.join(OUT_DIR, "disruption_player_summary.csv"), index=False)

    team_summary = linked_only.groupby("team_name").agg(
        matches=("match_file", "nunique"),
        actions_linked=("disruption_score", "size"),
        total_disruption=("disruption_score", "sum"),
        mean_disruption=("disruption_score", "mean"),
    ).reset_index()
    team_summary["disruption_per_match"] = team_summary["total_disruption"] / team_summary["matches"]
    team_summary = team_summary.sort_values("disruption_per_match", ascending=False)
    team_summary.to_csv(os.path.join(OUT_DIR, "disruption_team_summary.csv"), index=False)

    print("Wrote disruption_player_summary.csv + disruption_team_summary.csv")
    print("\nTop 10 disruptors (total disruption value):")
    print(player_summary.head(10)[["player_name", "team_name", "matches", "actions_linked",
                                    "total_disruption", "disruption_per90"]].to_string(index=False))


if __name__ == "__main__":
    main()

"""
Goal Kick Expected Possession Value (EPV) Model - Eredivisie 2025/26
=======================================================================
Answers: "how much is a goal kick worth, in goals, once you account for
where it goes and whether the kicking team keeps the ball?" -- a value
question, where build_goalkick_shot_model.py only counts how often a goal
kick reaches a shot within 5 actions.

Approach (VAEP-style possession value, Decroos et al. 2019):
  1. Possession value model, trained on EVERY on-ball action (not just goal
     kicks) from the three previous Eredivisie seasons (2022/23-2024/25):
       P_score(s)   = P(acting team scores within the next K=10 actions)
       P_concede(s) = P(acting team concedes within the next K=10 actions)
       EPV(s)       = P_score(s) - P_concede(s)                  (in goals)
     State s = the action itself (type, outcome, start/end location,
     pass-type qualifiers) + the two previous actions (flipped into the
     acting team's frame when the opponent made them) + game context
     (period, minute, score difference). Two HistGradientBoosting
     classifiers, same library as the other models in this repo.
     Goal kicks are only ~1% of actions, so learning the value surface from
     all actions and then reading goal kicks off it is far more stable than
     fitting on goal kicks alone (a goal within 10 actions of a goal kick
     is a rare event).
  2. The model is evaluated out-of-sample on 2025/26 and 2025/26 is then
     scored with that same (never-seen-2025/26) model, so every goal kick
     value below is a genuine out-of-sample prediction.
  3. Per goal kick (Opta pass, typeId 1 + qualifier 124):
       gk_epv            EPV of the state the goal kick creates (where it
                         landed + whether it was completed)
       gk_epv_added      gk_epv minus the league-average goal kick EPV
       poss_epv_max      highest EPV the kicking team reached before the
                         opponent won the ball -- how far the build-up got
       poss_epv_gain     poss_epv_max - gk_epv
     plus realised outcomes (goal for/against within 10 actions, shot in the
     possession) so the expected numbers can be checked against reality.

Coordinates: this Opta feed stores every event in the acting team's own
frame (own goal at x=0, attacking toward x=100) -- both teams' goal kicks
sit at x~5 -- so no per-period direction flip is needed for the acting
team; only an opponent's action is mirrored into the acting team's frame.

Data caveat: no freeze frames, so the state knows nothing about where the
opponents stand (e.g. how high they press a short goal kick) -- that shows
up only indirectly, through whether the kick and the next actions succeed.

Usage: python3 build_goalkick_epv_model.py
Outputs:
  Goal Kick Model/CSV/all_eredivisie_goalkick_epv.csv       (one row per goal kick)
  Goal Kick Model/CSV/goalkick_epv_team_summary.csv
  Goal Kick Model/CSV/goalkick_epv_player_summary.csv       (by goalkeeper/kicker)
  Goal Kick Model/CSV/goalkick_epv_zone_grid.csv            (EPV by landing zone)
  Goal Kick Model/CSV/goalkick_epv_calibration.csv
  Model/model_epv_score.pkl, Model/model_epv_concede.pkl, Model/model_epv_meta.json
"""
import glob
import json
import math
import os
from multiprocessing import Pool

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVENTS_DIR = os.path.join(ROOT, "Eredivisie Events")
TRAIN_SEASONS = ["2022-2023", "2023-2024", "2024-2025"]
SCORE_SEASON = "2025-2026"
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_DIR = os.path.join(OUT_DIR, "CSV")
MODEL_DIR = os.path.join(ROOT, "Model")

X_SCALE, Y_SCALE = 1.05, 0.68  # Opta 0-100 units -> metres (105 x 68 pitch)
PITCH_X, PITCH_Y = 105.0, 68.0
GOAL_X, GOAL_Y = 105.0, 34.0

K_ACTIONS = 10  # label horizon: goal within the next 10 actions (VAEP default)

T_PASS, T_GOAL = 1, 16
SHOT_TYPES = {13, 14, 15, 16}  # Miss, Post, Attempt Saved, Goal
# On-ball actions only: drops admin/state events (deleted 43, ball out 5,
# corner awarded 6, cards, subs, period start/end, formation changes, ...).
ACTION_TYPES = {
    1,   # Pass
    2,   # Offside pass
    3,   # Take on
    7,   # Tackle
    8,   # Interception
    10,  # Save
    11,  # Claim
    12,  # Clearance
    13, 14, 15, 16,  # shots
    41,  # Punch
    42,  # Good skill
    44,  # Aerial
    45,  # Challenge
    49,  # Ball recovery
    50,  # Dispossessed
    51,  # Error
    52,  # Keeper pick-up
    54,  # Smother
    61,  # Ball touch
    74,  # Blocked pass
}
# An opponent event of these types with outcome 0 is them LOSING a duel,
# not them gaining the ball, so it does not end the kicking team's possession.
DUEL_TYPES = {3, 44, 45}

Q_OWN_GOAL = 28
Q_END_X, Q_END_Y = 140, 141
Q_GOAL_KICK = 124
FLAG_QUALIFIERS = {
    "long_ball": 1, "cross": 2, "head_pass": 3, "through_ball": 4,
    "free_kick": 5, "corner": 6, "head": 15, "throw_in": 107,
    "goal_kick": 124, "chipped": 155, "launch": 157, "switch": 196,
}

ACTION_FEATURES = [
    "type_id", "outcome", "x", "y", "end_x", "end_y", "dx", "dy", "move_len",
    "start_dist", "start_angle", "end_dist", "end_angle",
] + [f"q_{k}" for k in FLAG_QUALIFIERS]
PREV_FEATURES = ["type_id", "outcome", "x", "y", "end_x", "end_y", "same_team", "dt"]
CONTEXT_FEATURES = ["period_id", "minute", "score_diff"]
FEATURE_COLS = (ACTION_FEATURES
                + [f"p1_{c}" for c in PREV_FEATURES]
                + [f"p2_{c}" for c in PREV_FEATURES]
                + CONTEXT_FEATURES)
CATEGORICAL = ["type_id", "p1_type_id", "p2_type_id"]
ALL_TYPE_IDS = sorted(ACTION_TYPES)


def qmap(e):
    return {q["qualifierId"]: q.get("value") for q in e.get("qualifier", []) or []}


def goal_geometry(x, y):
    dist = math.hypot(GOAL_X - x, GOAL_Y - y)
    angle = abs(math.atan2(GOAL_Y - y, GOAL_X - x))
    return dist, angle


def goal_team(e, cids):
    """Team credited with a goal event. Opta files an own goal under the
    conceding team's contestantId with qualifier 28."""
    cid = e["contestantId"]
    if any(q["qualifierId"] == Q_OWN_GOAL for q in e.get("qualifier", []) or []):
        others = [c for c in cids if c != cid]
        return others[0] if others else None
    return cid


def match_actions(fn):
    """One row per on-ball action in a match, with state features in the
    acting team's frame and the VAEP score/concede labels."""
    with open(fn) as f:
        data = json.load(f)
    match_file = os.path.basename(fn).replace(".json", "")
    events = [e for e in data["event"]
              if e.get("contestantId") and e.get("x") is not None
              and e["typeId"] in ACTION_TYPES and e.get("periodId") in (1, 2, 3, 4)]
    events.sort(key=lambda e: (e["periodId"], e["timeMin"] * 60 + e["timeSec"], e["eventId"]))
    cids = sorted({e["contestantId"] for e in events})

    rows = []
    goals_by_team = {c: 0 for c in cids}
    for e in events:
        q = qmap(e)
        x, y = e["x"] * X_SCALE, e["y"] * Y_SCALE
        try:
            ex, ey = float(q[Q_END_X]) * X_SCALE, float(q[Q_END_Y]) * Y_SCALE
        except (KeyError, TypeError, ValueError):
            ex, ey = x, y
        sd, sa = goal_geometry(x, y)
        ed, ea = goal_geometry(ex, ey)
        cid = e["contestantId"]
        opp = [c for c in cids if c != cid]
        row = {
            "match_file": match_file,
            "id": e["id"], "event_id": e["eventId"],
            "contestant_id": cid,
            "player_id": e.get("playerId"), "player_name": e.get("playerName"),
            "period_id": e["periodId"], "time_min": e["timeMin"], "time_sec": e["timeSec"],
            "t": e["timeMin"] * 60 + e["timeSec"],
            "minute": e["timeMin"],
            "type_id": e["typeId"], "outcome": int(e.get("outcome", 0) or 0),
            "x": x, "y": y, "end_x": ex, "end_y": ey,
            "dx": ex - x, "dy": ey - y, "move_len": math.hypot(ex - x, ey - y),
            "start_dist": sd, "start_angle": sa, "end_dist": ed, "end_angle": ea,
            "score_diff": goals_by_team[cid] - (goals_by_team[opp[0]] if opp else 0),
            "goal_for_team": None,
        }
        for name, qid in FLAG_QUALIFIERS.items():
            row[f"q_{name}"] = int(qid in q)
        if e["typeId"] == T_GOAL:
            scorer = goal_team(e, cids)
            row["goal_for_team"] = scorer
            if scorer in goals_by_team:
                goals_by_team[scorer] += 1  # score_diff is pre-action, so update after
        rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # previous two actions, mirrored into the current actor's frame when the
    # opponent made them
    for lag in (1, 2):
        p = df.shift(lag)
        same = (p["contestant_id"] == df["contestant_id"]) & (p["period_id"] == df["period_id"])
        valid = p["period_id"] == df["period_id"]
        pre = f"p{lag}_"
        df[pre + "same_team"] = np.where(valid, same.astype(int), -1)
        df[pre + "type_id"] = np.where(valid, p["type_id"], -1).astype(int)
        df[pre + "outcome"] = np.where(valid, p["outcome"], -1)
        for c, size in (("x", PITCH_X), ("end_x", PITCH_X), ("y", PITCH_Y), ("end_y", PITCH_Y)):
            v = np.where(same, p[c], size - p[c])
            df[pre + c] = np.where(valid, v, np.nan)
        df[pre + "dt"] = np.where(valid, df["t"] - p["t"], np.nan)

    # labels: goal for / against the acting team within actions i..i+K-1,
    # same period only
    n = len(df)
    goal_team_arr = df["goal_for_team"].to_numpy(dtype=object)
    period = df["period_id"].to_numpy()
    actor = df["contestant_id"].to_numpy()
    scores = np.zeros(n, dtype=np.int8)
    concedes = np.zeros(n, dtype=np.int8)
    goal_idx = np.flatnonzero(df["goal_for_team"].notna().to_numpy())
    for g in goal_idx:
        for i in range(max(0, g - K_ACTIONS + 1), g + 1):
            if period[i] != period[g]:
                continue
            if actor[i] == goal_team_arr[g]:
                scores[i] = 1
            else:
                concedes[i] = 1
    df["scores"] = scores
    df["concedes"] = concedes
    df["season_row"] = np.arange(n)
    return df


def load_season(season, workers=4):
    files = sorted(glob.glob(os.path.join(EVENTS_DIR, season, "*.json")))
    if not files:
        raise SystemExit(f"No match files found in {os.path.join(EVENTS_DIR, season)}")
    with Pool(workers) as pool:
        frames = pool.map(match_actions, files)
    df = pd.concat([f for f in frames if not f.empty], ignore_index=True)
    df["season"] = season
    print(f"  {season}: {len(files)} matches, {len(df):,} actions, "
          f"{int(df['goal_for_team'].notna().sum())} goals")
    return df


def build_global_cid_to_team(season):
    """contestantId -> team name, via the intersection trick used by
    Disruption/build_disruption_model.py (the one id constant across all of a
    team's fixtures is theirs)."""
    import re
    team_cid_sets = {}
    for fn in sorted(glob.glob(os.path.join(EVENTS_DIR, season, "*.json"))):
        m = re.match(r"\d{4}-\d{2}-\d{2}_(.+) - (.+)$", os.path.basename(fn)[:-5])
        if not m:
            continue
        with open(fn) as f:
            cids = {e["contestantId"] for e in json.load(f)["event"] if e.get("contestantId")}
        for team in m.groups():
            team_cid_sets.setdefault(team, []).append(cids)
    out = {}
    for team, sets in team_cid_sets.items():
        inter = set.intersection(*sets)
        if len(inter) == 1:
            out[next(iter(inter))] = team
    return out


def to_matrix(df):
    X = df[FEATURE_COLS].astype("float32").copy()
    # map categorical type ids onto 0..n (unknown/missing -> n) for HGB
    lookup = {t: i for i, t in enumerate(ALL_TYPE_IDS)}
    for c in CATEGORICAL:
        X[c] = df[c].map(lookup).fillna(len(ALL_TYPE_IDS)).astype("float32")
    return X


def train_classifier(X, y):
    clf = HistGradientBoostingClassifier(
        max_iter=400, learning_rate=0.08, max_leaf_nodes=48, min_samples_leaf=200,
        l2_regularization=1.0, early_stopping=True, validation_fraction=0.1,
        n_iter_no_change=20, random_state=42,
        categorical_features=[FEATURE_COLS.index(c) for c in CATEGORICAL],
    )
    clf.fit(X, y)
    return clf


def evaluate(name, y, p):
    base = np.full_like(p, y.mean())
    out = {
        "target": name, "n": int(len(y)), "positive_rate": float(y.mean()),
        "auc": float(roc_auc_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "brier_baseline": float(brier_score_loss(y, base)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "log_loss_baseline": float(log_loss(y, base, labels=[0, 1])),
    }
    print(f"  {name:>9}: AUC {out['auc']:.3f} | Brier {out['brier']:.5f} "
          f"(base {out['brier_baseline']:.5f}) | log loss {out['log_loss']:.4f} "
          f"(base {out['log_loss_baseline']:.4f})")
    return out


def calibration_table(y, p, name, bins=10):
    q = pd.qcut(p, bins, labels=False, duplicates="drop")
    t = pd.DataFrame({"bin": q, "pred": p, "obs": y}).groupby("bin").agg(
        n=("obs", "size"), mean_pred=("pred", "mean"), mean_obs=("obs", "mean")).reset_index()
    t.insert(0, "target", name)
    return t


def goal_kick_type(end_x):
    if end_x < 35:
        return "short"
    if end_x < 52.5:
        return "mid"
    return "long"


def value_goal_kicks(df, team_map):
    """One row per 2025/26 goal kick with its EPV and the kicking team's
    possession that followed."""
    rows = []
    for match_file, m in df.groupby("match_file", sort=False):
        m = m.reset_index(drop=True)
        gk_idx = m.index[(m["type_id"] == T_PASS) & (m["q_goal_kick"] == 1)]
        cid_arr = m["contestant_id"].to_numpy()
        per_arr = m["period_id"].to_numpy()
        type_arr = m["type_id"].to_numpy()
        out_arr = m["outcome"].to_numpy()
        epv_arr = m["epv"].to_numpy()
        x_arr = m["x"].to_numpy()
        for i in gk_idx:
            cid, period = cid_arr[i], per_arr[i]
            chain = [i]
            for j in range(i + 1, len(m)):
                if per_arr[j] != period:
                    break
                if cid_arr[j] != cid:
                    if type_arr[j] in DUEL_TYPES and out_arr[j] == 0:
                        continue  # opponent lost a duel: we still have it
                    break
                chain.append(j)
            chain = np.array(chain)
            shot = bool(np.isin(type_arr[chain], list(SHOT_TYPES)).any())
            goal = bool((type_arr[chain] == T_GOAL).any())
            r = m.loc[i]
            rows.append({
                "match_file": match_file, "event_id": r["event_id"], "id": r["id"],
                "period_id": period, "time_min": r["time_min"], "time_sec": r["time_sec"],
                "contestant_id": cid, "team_name": team_map.get(cid, cid),
                "player_id": r["player_id"], "player_name": r["player_name"],
                "x": r["x"], "y": r["y"], "end_x": r["end_x"], "end_y": r["end_y"],
                "length": r["move_len"],
                "gk_type": goal_kick_type(r["end_x"]),
                "completed": int(r["outcome"]),
                "score_diff": r["score_diff"],
                "gk_p_score": r["p_score"], "gk_p_concede": r["p_concede"],
                "gk_epv": r["epv"],
                "poss_n_actions": len(chain) - 1,
                "poss_max_x": float(x_arr[chain].max()),
                "poss_epv_max": float(epv_arr[chain].max()),
                "poss_epv_end": float(epv_arr[chain[-1]]),
                "poss_shot": shot, "poss_goal": goal,
                "goal_for_10": int(r["scores"]), "goal_against_10": int(r["concedes"]),
            })
    gk = pd.DataFrame(rows)
    gk["gk_epv_added"] = gk["gk_epv"] - gk["gk_epv"].mean()
    gk["poss_epv_gain"] = gk["poss_epv_max"] - gk["gk_epv"]
    gk["realised_net_10"] = gk["goal_for_10"] - gk["goal_against_10"]
    return gk


def summarise(gk, keys):
    s = gk.groupby(keys).agg(
        matches=("match_file", "nunique"),
        goal_kicks=("gk_epv", "size"),
        share_short=("gk_type", lambda t: (t == "short").mean()),
        share_long=("gk_type", lambda t: (t == "long").mean()),
        completion=("completed", "mean"),
        mean_length=("length", "mean"),
        gk_p_score=("gk_p_score", "mean"),
        gk_p_concede=("gk_p_concede", "mean"),
        gk_epv=("gk_epv", "mean"),
        gk_epv_added=("gk_epv_added", "mean"),
        poss_epv_max=("poss_epv_max", "mean"),
        poss_epv_gain=("poss_epv_gain", "mean"),
        poss_shot_rate=("poss_shot", "mean"),
        goals_for_10=("goal_for_10", "sum"),
        goals_against_10=("goal_against_10", "sum"),
        total_epv=("gk_epv", "sum"),
    ).reset_index()
    for t in ("short", "long"):
        by_t = gk[gk["gk_type"] == t].groupby(keys)["gk_epv"].mean().rename(f"gk_epv_{t}")
        s = s.merge(by_t, on=keys, how="left")
    # headline numbers in goals per 100 goal kicks -- raw EPV per kick is ~1e-3
    for c in ("gk_epv", "gk_epv_added", "poss_epv_max", "poss_epv_gain",
              "gk_epv_short", "gk_epv_long"):
        s[f"{c}_per100"] = s[c] * 100
    s["realised_net_10_per100"] = (s["goals_for_10"] - s["goals_against_10"]) / s["goal_kicks"] * 100
    return s.sort_values("gk_epv_added", ascending=False)


def zone_grid(gk, x_edges=(0, 17.5, 35, 52.5, 70, 87.5, 105), y_edges=(0, 17, 34, 51, 68)):
    g = gk.copy()
    g["zx"] = pd.cut(g["end_x"].clip(0, 104.99), x_edges, labels=False, include_lowest=True)
    g["zy"] = pd.cut(g["end_y"].clip(0, 67.99), y_edges, labels=False, include_lowest=True)
    z = g.groupby(["zx", "zy"]).agg(
        n=("gk_epv", "size"), completion=("completed", "mean"),
        gk_epv=("gk_epv", "mean"), poss_epv_max=("poss_epv_max", "mean"),
        poss_shot_rate=("poss_shot", "mean")).reset_index()
    z["x0"] = z["zx"].map(lambda i: x_edges[int(i)])
    z["x1"] = z["zx"].map(lambda i: x_edges[int(i) + 1])
    z["y0"] = z["zy"].map(lambda i: y_edges[int(i)])
    z["y1"] = z["zy"].map(lambda i: y_edges[int(i) + 1])
    z["gk_epv_per100"] = z["gk_epv"] * 100
    z["poss_epv_max_per100"] = z["poss_epv_max"] * 100
    return z


def main():
    print("Loading training seasons...")
    train = pd.concat([load_season(s) for s in TRAIN_SEASONS], ignore_index=True)
    print("Loading scoring season...")
    test = load_season(SCORE_SEASON)

    X_train, X_test = to_matrix(train), to_matrix(test)
    print(f"Training on {len(train):,} actions ({', '.join(TRAIN_SEASONS)})...")
    clf_score = train_classifier(X_train, train["scores"])
    clf_concede = train_classifier(X_train, train["concedes"])

    test["p_score"] = clf_score.predict_proba(X_test)[:, 1]
    test["p_concede"] = clf_concede.predict_proba(X_test)[:, 1]
    test["epv"] = test["p_score"] - test["p_concede"]

    print(f"Out-of-sample evaluation on {SCORE_SEASON} (all actions):")
    metrics = [evaluate("scores", test["scores"].to_numpy(), test["p_score"].to_numpy()),
               evaluate("concedes", test["concedes"].to_numpy(), test["p_concede"].to_numpy())]
    calib = pd.concat([
        calibration_table(test["scores"].to_numpy(), test["p_score"].to_numpy(), "scores"),
        calibration_table(test["concedes"].to_numpy(), test["p_concede"].to_numpy(), "concedes"),
    ], ignore_index=True)

    team_map = build_global_cid_to_team(SCORE_SEASON)
    gk = value_goal_kicks(test, team_map)
    print(f"Valued {len(gk):,} goal kicks across {gk['match_file'].nunique()} matches")
    print(f"  mean goal kick EPV: {gk['gk_epv'].mean() * 100:+.3f} goals per 100 kicks "
          f"(P_score {gk['gk_p_score'].mean():.4f}, P_concede {gk['gk_p_concede'].mean():.4f})")
    print(f"  realised within 10 actions: {gk['goal_for_10'].mean():.4f} scored, "
          f"{gk['goal_against_10'].mean():.4f} conceded")
    by_type = gk.groupby("gk_type").agg(n=("gk_epv", "size"), completion=("completed", "mean"),
                                        epv_per100=("gk_epv", lambda v: v.mean() * 100),
                                        poss_max_per100=("poss_epv_max", lambda v: v.mean() * 100),
                                        realised_net_per100=("realised_net_10", lambda v: v.mean() * 100))
    print(by_type.to_string())

    os.makedirs(CSV_DIR, exist_ok=True)
    gk.to_csv(os.path.join(CSV_DIR, "all_eredivisie_goalkick_epv.csv"), index=False)
    team_summary = summarise(gk, ["team_name"])
    team_summary.to_csv(os.path.join(CSV_DIR, "goalkick_epv_team_summary.csv"), index=False)
    player_summary = summarise(gk, ["player_id", "player_name", "team_name"])
    player_summary.to_csv(os.path.join(CSV_DIR, "goalkick_epv_player_summary.csv"), index=False)
    zone_grid(gk).to_csv(os.path.join(CSV_DIR, "goalkick_epv_zone_grid.csv"), index=False)
    calib.to_csv(os.path.join(CSV_DIR, "goalkick_epv_calibration.csv"), index=False)

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(clf_score, os.path.join(MODEL_DIR, "model_epv_score.pkl"))
    joblib.dump(clf_concede, os.path.join(MODEL_DIR, "model_epv_concede.pkl"))
    with open(os.path.join(MODEL_DIR, "model_epv_meta.json"), "w") as f:
        json.dump({
            "method": "VAEP-style possession value: EPV = P(score in next K actions) - "
                      "P(concede in next K actions)",
            "k_actions": K_ACTIONS,
            "train_seasons": TRAIN_SEASONS, "test_season": SCORE_SEASON,
            "n_train_actions": int(len(train)), "n_test_actions": int(len(test)),
            "feature_cols": FEATURE_COLS, "categorical": CATEGORICAL,
            "type_id_order": ALL_TYPE_IDS,
            "out_of_sample_metrics": metrics,
        }, f, indent=2)

    print("\nTeams by goal kick EPV added (goals per 100 goal kicks vs league average):")
    print(team_summary[["team_name", "goal_kicks", "share_short", "completion",
                        "gk_epv_added_per100", "poss_epv_max_per100", "realised_net_10_per100"]]
          .round(3).to_string(index=False))


if __name__ == "__main__":
    main()

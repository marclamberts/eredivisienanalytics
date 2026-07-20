"""VAEP (Valuing Actions by Estimating Probabilities), Decroos et al. (2019),
implemented directly against this season's event data -- not the socceraction
library and not the paper's original trained weights (neither is used here).

Method:
1. Convert Events/*.json into an ordered action stream per match: one row per
   on-ball action (pass, cross, take-on, shot, tackle, interception,
   clearance, aerial, recovery, blocked pass, foul, card, goalkeeper action),
   with start/end x,y (end coords via qualifiers 140/141 where the type has
   them, else end==start for point actions).
2. Build a "gamestate" feature vector for the state ending at each action k,
   from actions k, k-1, k-2 (type, success, start/end location, polar
   distance/angle to goal, whether same team as action k).
3. Label each state: does the team acting at k score within the next 10
   actions (regardless of intervening possession changes)? Does it concede?
4. Train two gradient-boosted classifiers (sklearn HistGradientBoostingClassifier
   -- no xgboost/lightgbm available in this environment) on those state/label
   pairs, with a match-level train/test split to report held-out AUC.
5. Get out-of-fold probabilities for every action (5-fold, grouped by match,
   via cross_val_predict) so player values aren't inflated by in-sample fit.
6. offensive_value(a_k) = P(score | state_k) - P(score | state_{k-1})
   defensive_value(a_k) = P(concede | state_{k-1}) - P(concede | state_k)
   VAEP(a_k) = offensive_value + defensive_value, attributed to the action's
   player, summed to a season total.

Known simplification vs. the published method: no explicit "dribble/carry"
action is synthesized between consecutive touches (Decroos et al. insert one
whenever the ball moves >3m between actions); this run scores each raw event
as its own action instead. Documented here rather than silently assumed.
"""
import csv
import glob
import json
import math
import os
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, cross_val_predict

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EVENTS_DIR = os.path.join(ROOT, "Events")
CACHE_PATH = os.path.join(ROOT, "Season stats", "Scripts", "_vaep_cache.csv")

GOAL_TYPE = 16
LOOKAHEAD = 10
LOOKBACK = 3  # current + 2 previous actions in a gamestate

# type_id -> simplified action-type label
TYPE_MAP = {
    1: "pass", 3: "take_on", 4: "foul", 7: "tackle", 8: "interception",
    12: "clearance", 13: "shot", 14: "shot", 15: "shot", 16: "shot",
    17: "card", 44: "aerial", 49: "recovery", 74: "blocked_pass",
    10: "keeper", 11: "keeper", 41: "keeper", 42: "keeper", 52: "keeper", 54: "keeper", 59: "keeper",
}
ACTION_TYPES = sorted(set(TYPE_MAP.values())) + ["none"]
TYPE_TO_CODE = {t: i for i, t in enumerate(ACTION_TYPES)}

GOAL_X, GOAL_Y = 100.0, 50.0


def qmap(e):
    return {q.get("qualifierId"): q.get("value") for q in e.get("qualifier", []) or []}


def num(v, default=0.0):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def polar(x, y):
    dx, dy = GOAL_X - x, GOAL_Y - y
    dist = math.hypot(dx, dy)
    angle = math.atan2(dy, dx)
    return dist, angle


def load_shot_xg():
    xg = {}
    with open(os.path.join(ROOT, "Danger", "all_eredivisie_danger_models.csv"), encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            xg[(r["match_file"], r["event_id"])] = num(r["xg"])
    return xg


def build_action_stream(fn, data, shot_xg):
    """One dict per action, in chronological order."""
    events = [e for e in data.get("event", []) if e.get("typeId") in TYPE_MAP and e.get("playerId")]
    events.sort(key=lambda e: (e.get("periodId", 0), e.get("timeMin", 0), e.get("timeSec", 0), e.get("eventId", 0)))

    actions = []
    for e in events:
        tid = e.get("typeId")
        atype = TYPE_MAP[tid]
        q = qmap(e)
        x, y = num(e.get("x")), num(e.get("y"))
        if tid == 1:  # pass: has an end location
            ex, ey = num(q.get(140, x)), num(q.get(141, y))
        else:
            ex, ey = x, y
        success = 1 if e.get("outcome") == 1 else 0
        xg = shot_xg.get((fn, str(e.get("id"))), 0.0) if atype == "shot" else 0.0
        actions.append({
            "match": fn, "player_id": e.get("playerId"), "team": e.get("contestantId"),
            "type": atype, "success": success, "x": x, "y": y, "ex": ex, "ey": ey,
            "xg": xg, "is_goal": 1 if (tid == GOAL_TYPE and success) else 0,
            "t": e.get("periodId", 0) * 10000 + e.get("timeMin", 0) * 60 + e.get("timeSec", 0),
        })
    return actions


def build_state_features(actions):
    """Returns (X, meta) where X is a numpy feature matrix (one row per
    action, state ending at that action) and meta has match/player/team/type
    for attribution."""
    n = len(actions)
    n_types = len(ACTION_TYPES)
    # per-lookback-slot: type one-hot (n_types), success, start dist/angle,
    # end dist/angle, same_team_as_current
    feat_per_slot = n_types + 5
    X = np.zeros((n, feat_per_slot * LOOKBACK), dtype=np.float32)

    for i, a in enumerate(actions):
        for slot in range(LOOKBACK):
            j = i - slot
            base = slot * feat_per_slot
            if j < 0 or actions[j]["match"] != a["match"]:
                X[i, base + TYPE_TO_CODE["none"]] = 1.0
                continue
            aj = actions[j]
            X[i, base + TYPE_TO_CODE[aj["type"]]] = 1.0
            sd, sa = polar(aj["x"], aj["y"])
            ed, ea = polar(aj["ex"], aj["ey"])
            X[i, base + n_types] = aj["success"]
            X[i, base + n_types + 1] = sd / 100.0
            X[i, base + n_types + 2] = sa
            X[i, base + n_types + 3] = ed / 100.0
            X[i, base + n_types + 4] = 1.0 if aj["team"] == a["team"] else 0.0
    return X


def build_labels(actions):
    n = len(actions)
    label_score = np.zeros(n, dtype=np.int8)
    label_concede = np.zeros(n, dtype=np.int8)
    # process per match for correctness and speed
    by_match = defaultdict(list)
    for i, a in enumerate(actions):
        by_match[a["match"]].append(i)
    for fn, idxs in by_match.items():
        idxs.sort()
        m = len(idxs)
        for pos, i in enumerate(idxs):
            team = actions[i]["team"]
            end = min(pos + LOOKAHEAD, m - 1)
            for pos2 in range(pos + 1, end + 1):
                j = idxs[pos2]
                if actions[j]["is_goal"]:
                    if actions[j]["team"] == team:
                        label_score[i] = 1
                    else:
                        label_concede[i] = 1
    return label_score, label_concede


def main():
    print("Loading shot xG lookup...")
    shot_xg = load_shot_xg()

    print("Building action stream...")
    actions = []
    for path in sorted(glob.glob(os.path.join(EVENTS_DIR, "*.json"))):
        fn = os.path.basename(path)
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        actions.extend(build_action_stream(fn, data, shot_xg))
    print(f"{len(actions)} actions across {len({a['match'] for a in actions})} matches")

    print("Building gamestate features...")
    X = build_state_features(actions)

    print("Building labels (goal within next 10 actions)...")
    y_score, y_concede = build_labels(actions)
    print(f"  score rate {y_score.mean():.4f}, concede rate {y_concede.mean():.4f}")

    matches = np.array([a["match"] for a in actions])
    match_ids = {m: i for i, m in enumerate(sorted(set(matches)))}
    groups = np.array([match_ids[m] for m in matches])

    # held-out AUC on a single match-level split, for a documented quality number
    rng = np.random.RandomState(42)
    all_matches = sorted(match_ids)
    rng.shuffle(all_matches)
    n_test = max(1, len(all_matches) // 5)
    test_matches = set(all_matches[:n_test])
    test_mask = np.array([m in test_matches for m in matches])
    train_mask = ~test_mask

    print("Training held-out split for AUC reporting...")
    clf_score = HistGradientBoostingClassifier(max_iter=200, random_state=0)
    clf_score.fit(X[train_mask], y_score[train_mask])
    auc_score = roc_auc_score(y_score[test_mask], clf_score.predict_proba(X[test_mask])[:, 1])

    clf_concede = HistGradientBoostingClassifier(max_iter=200, random_state=0)
    clf_concede.fit(X[train_mask], y_concede[train_mask])
    auc_concede = roc_auc_score(y_concede[test_mask], clf_concede.predict_proba(X[test_mask])[:, 1])
    print(f"  held-out AUC: score={auc_score:.4f}, concede={auc_concede:.4f}")

    print("Computing out-of-fold probabilities (5-fold, grouped by match)...")
    gkf = GroupKFold(n_splits=5)
    prob_score = cross_val_predict(
        HistGradientBoostingClassifier(max_iter=200, random_state=0), X, y_score,
        cv=gkf.split(X, y_score, groups), method="predict_proba", n_jobs=-1,
    )[:, 1]
    prob_concede = cross_val_predict(
        HistGradientBoostingClassifier(max_iter=200, random_state=0), X, y_concede,
        cv=gkf.split(X, y_concede, groups), method="predict_proba", n_jobs=-1,
    )[:, 1]

    print("Computing per-action VAEP...")
    base_score = y_score.mean()
    base_concede = y_concede.mean()
    off_val = np.zeros(len(actions))
    def_val = np.zeros(len(actions))
    by_match_idx = defaultdict(list)
    for i, a in enumerate(actions):
        by_match_idx[a["match"]].append(i)
    for fn, idxs in by_match_idx.items():
        idxs.sort()
        prev_s, prev_c = base_score, base_concede
        for i in idxs:
            off_val[i] = prob_score[i] - prev_s
            def_val[i] = prev_c - prob_concede[i]
            prev_s, prev_c = prob_score[i], prob_concede[i]
    vaep = off_val + def_val

    print("Aggregating to player level...")
    player_vaep = defaultdict(lambda: {"off": 0.0, "def": 0.0, "vaep": 0.0, "actions": 0})
    for i, a in enumerate(actions):
        p = player_vaep[a["player_id"]]
        p["off"] += off_val[i]
        p["def"] += def_val[i]
        p["vaep"] += vaep[i]
        p["actions"] += 1

    out = pd.DataFrame([
        {"player_id": pid, "vaep_offensive": v["off"], "vaep_defensive": v["def"],
         "VAEP": v["vaep"], "vaep_actions": v["actions"]}
        for pid, v in player_vaep.items()
    ])
    out.sort_values("VAEP", ascending=False, inplace=True)
    out.to_csv(CACHE_PATH, index=False)
    print(f"Wrote {CACHE_PATH}: {len(out)} players")
    print(out.head(15).to_string())

    meta_path = os.path.join(ROOT, "Season stats", "Scripts", "_vaep_meta.txt")
    with open(meta_path, "w") as f:
        f.write(f"held_out_auc_score={auc_score:.4f}\nheld_out_auc_concede={auc_concede:.4f}\n"
                f"n_actions={len(actions)}\nn_matches={len(match_ids)}\n"
                f"base_score_rate={base_score:.4f}\nbase_concede_rate={base_concede:.4f}\n")
    print(f"Wrote {meta_path}")


if __name__ == "__main__":
    main()

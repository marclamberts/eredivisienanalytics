"""RAPM (Regularized Adjusted Plus-Minus) for the Eredivisie 2025/26 season.

Method: reconstruct exact on-pitch lineups over time from
GDA/gda_player_stints.csv (start_s/end_s are on one continuous match clock
that runs through both halves without resetting -- verified against raw
substitution events: R. Margaret's sub-off event has periodId 2,
timeMin=45, timeSec=0, i.e. continuous_s = timeMin*60+timeSec = 2700,
exactly matching his stint's end_s=2700). Split each match into segments at
every substitution/red-card boundary (either team), so each segment has a
constant 22-player lineup. For each segment, target = goal (or shot xG)
differential per 90 between the two teams; predictors = one column per
player, +1 if on the pitch for the "reference" side of that match, -1 for
the other side (arbitrary per-match reference; RAPM coefficients are still
each player's own marginal effect). Fit Ridge regression, weighted by
segment duration, with alpha chosen by cross-validation.

This is a real regression against the season's exact lineup data -- not a
reused/approximated number -- but it is naturally noisy: goal-based RAPM in
particular has very little signal per match (few goals per season), so an
xG-based version (xRAPM) is reported alongside as the more reliable read.
"""
import csv
import glob
import json
import os
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EVENTS_DIR = os.path.join(ROOT, "Events")


def rows(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def continuous_s(time_min, time_sec):
    return time_min * 60 + time_sec


def build_segments():
    """Returns (segments, match_ids) where segments is a list of dicts:
    {"match": fn, "players_for": set, "players_against": set,
     "goal_diff": float, "xg_diff": float, "minutes": float}."""
    stints_by_match = defaultdict(list)
    for r in rows(os.path.join(ROOT, "GDA", "gda_player_stints.csv")):
        stints_by_match[r["match_file"]].append(r)

    danger_by_match = defaultdict(list)
    for r in rows(os.path.join(ROOT, "Danger", "all_eredivisie_danger_models.csv")):
        danger_by_match[r["match_file"]].append(r)

    segments = []
    for fn, stints in stints_by_match.items():
        path = os.path.join(EVENTS_DIR, fn)
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        match_duration = num(stints[0]["match_duration_seconds"])

        cids = sorted({s["contestant_id"] for s in stints})
        if len(cids) != 2:
            continue
        ref_cid, opp_cid = cids[0], cids[1]

        goals = defaultdict(list)  # cid -> [continuous_s, ...]
        for e in data.get("event", []):
            if e.get("typeId") == 16 and e.get("outcome") == 1:
                goals[e.get("contestantId")].append(continuous_s(e.get("timeMin", 0), e.get("timeSec", 0)))

        shots_xg = defaultdict(list)  # cid -> [(t, xg), ...]
        for r in danger_by_match.get(fn, []):
            shots_xg[r["contestant_id"]].append((continuous_s(num(r["time_min"]), num(r["time_sec"])), num(r["xg"])))

        boundaries = {0.0, match_duration}
        for s in stints:
            boundaries.add(num(s["start_s"]))
            boundaries.add(num(s["end_s"]))
        boundaries = sorted(b for b in boundaries if 0 <= b <= match_duration)

        for t0, t1 in zip(boundaries[:-1], boundaries[1:]):
            if t1 - t0 < 1:
                continue
            mid = (t0 + t1) / 2
            for_players = {s["player_id"] for s in stints
                           if s["contestant_id"] == ref_cid and num(s["start_s"]) <= mid < num(s["end_s"])}
            against_players = {s["player_id"] for s in stints
                                if s["contestant_id"] == opp_cid and num(s["start_s"]) <= mid < num(s["end_s"])}
            if not for_players or not against_players:
                continue

            g_for = sum(1 for t in goals.get(ref_cid, []) if t0 <= t < t1)
            g_against = sum(1 for t in goals.get(opp_cid, []) if t0 <= t < t1)
            xg_for = sum(v for t, v in shots_xg.get(ref_cid, []) if t0 <= t < t1)
            xg_against = sum(v for t, v in shots_xg.get(opp_cid, []) if t0 <= t < t1)

            segments.append({
                "match": fn,
                "players_for": for_players,
                "players_against": against_players,
                "goal_diff": g_for - g_against,
                "xg_diff": xg_for - xg_against,
                "minutes": (t1 - t0) / 60.0,
            })
    return segments


def fit_rapm(segments, target_key, min_minutes=180):
    """Ridge regression of per-90 differential on which players were on
    the pitch. Players under min_minutes total are folded into the
    intercept-only baseline (dropped as regression columns) to keep the
    design matrix from being dominated by single-appearance noise, but
    every player who appears at all still gets a coefficient of 0 (league
    average) if excluded this way -- so instead we keep everyone in and let
    the ridge penalty shrink low-minute players toward 0 naturally."""
    player_minutes = defaultdict(float)
    for seg in segments:
        for pid in seg["players_for"] | seg["players_against"]:
            player_minutes[pid] += seg["minutes"]
    players = sorted(player_minutes)
    idx = {pid: i for i, pid in enumerate(players)}

    n, p = len(segments), len(players)
    row_ind, col_ind, data_val = [], [], []
    y = np.zeros(n)
    w = np.zeros(n)
    groups = np.zeros(n, dtype=int)
    match_ids = sorted({seg["match"] for seg in segments})
    match_idx = {m: i for i, m in enumerate(match_ids)}

    for i, seg in enumerate(segments):
        for pid in seg["players_for"]:
            row_ind.append(i); col_ind.append(idx[pid]); data_val.append(1.0)
        for pid in seg["players_against"]:
            row_ind.append(i); col_ind.append(idx[pid]); data_val.append(-1.0)
        minutes = seg["minutes"]
        y[i] = seg[target_key] / (minutes / 90.0)
        w[i] = minutes
        groups[i] = match_idx[seg["match"]]

    X = sparse.csr_matrix((data_val, (row_ind, col_ind)), shape=(n, p))

    alphas = np.logspace(1, 5, 25)
    cv = GroupKFold(n_splits=5)
    model = RidgeCV(alphas=alphas, cv=cv.split(X, y, groups))
    model.fit(X, y, sample_weight=w)

    return {pid: model.coef_[idx[pid]] for pid in players}, model.alpha_, player_minutes


if __name__ == "__main__":
    print("Building lineup segments...")
    segments = build_segments()
    print(f"{len(segments)} segments across {len({s['match'] for s in segments})} matches")

    print("Fitting goals-based RAPM...")
    rapm_goals, alpha_g, minutes = fit_rapm(segments, "goal_diff")
    print(f"  alpha={alpha_g}")

    print("Fitting xG-based xRAPM...")
    rapm_xg, alpha_x, _ = fit_rapm(segments, "xg_diff")
    print(f"  alpha={alpha_x}")

    out = pd.DataFrame({
        "player_id": list(minutes.keys()),
        "rapm_minutes": list(minutes.values()),
    })
    out["RAPM"] = out["player_id"].map(rapm_goals)
    out["xRAPM"] = out["player_id"].map(rapm_xg)
    out.sort_values("xRAPM", ascending=False, inplace=True)

    out_path = os.path.join(ROOT, "Season stats", "Scripts", "_rapm_cache.csv")
    out.to_csv(out_path, index=False)
    print(f"Wrote {out_path}: {len(out)} players")
    print(out.head(15).to_string())

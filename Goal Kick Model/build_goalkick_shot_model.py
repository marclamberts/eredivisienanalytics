"""
Goal Kick -> Shot Model - Eredivisie 2025/26
================================================
Answers: "how often does a goal kick lead to a shot within 5 actions?" --
a directness/build-up-speed question distinct from the Interception
Transition model (Interception Model/build_interception_transition_model.py),
which values transitions by xT rather than counting to a shot, and starts
from regains rather than restarts.

Approach:
  1. Find every goal kick: an Opta pass event (typeId 1) carrying
     qualifierId 124 ("Goal Kick"). Confirmed against this season's data --
     these all originate a couple of metres from the kicking team's own
     byline, central-ish y, almost always taken by the goalkeeper.
  2. Walk forward chronologically through that match/period's other
     ball-involvement events (anything with x/y + a contestantId),
     starting right after the goal kick, accumulating events by the SAME
     team (the kicking team keeping the ball). The chain ends at the first
     of:
       - a shot by the kicking team (typeId in {13 Miss, 14 Post,
         15 Attempt Saved, 16 Goal)) -- success, record how many actions it took
       - a touch by the opposing team (possession lost back)
       - 5 actions accumulated without a shot
       - the period ending
     "5 actions" is a hard count cap here, not a time window (contrast
     with the interception model's 10-second window) -- the question
     asked is literally "within 5 actions", not "within N seconds".

Data caveat: same Opta feed as the other models in this repo, no
freeze-frame data, so "possession lost back to the opponent" is inferred
purely from contestantId changing on the next ball-involvement event.

Usage: python3 build_goalkick_shot_model.py
Outputs:
  Goal Kick Model/CSV/<match>_goalkicks.csv
  Goal Kick Model/CSV/all_eredivisie_goalkicks.csv
  Goal Kick Model/CSV/goalkick_player_summary.csv   (by goalkeeper/kicker)
  Goal Kick Model/CSV/goalkick_team_summary.csv
"""
import glob
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Disruption"))
import build_disruption_model as bdm  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "Events", "2025-2026")
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_DIR = os.path.join(OUT_DIR, "CSV")

T_PASS = 1
Q_GOAL_KICK = 124
SHOT_TYPES = {13, 14, 15, 16}  # Miss, Post, Attempt Saved, Goal
ACTION_LIMIT = 5


def event_ball_involvements(events):
    """Every event with a location and a team on it, chronologically
    ordered -- same filter/ordering as
    Interception Model/build_interception_transition_model.py so the two
    models agree on what counts as a "touch"."""
    rows = [e for e in events if e.get("x") is not None and e.get("contestantId")]
    rows.sort(key=lambda e: (e["periodId"], e["timeMin"] * 60 + e["timeSec"], e["eventId"]))
    return rows


def is_goal_kick(e):
    if e["typeId"] != T_PASS:
        return False
    return any(q["qualifierId"] == Q_GOAL_KICK for q in e.get("qualifier", []) or [])


def goal_kick_end_xy(e):
    """Where the goal kick itself was aimed, for pitch maps -- separate from
    the chain's later actions. None if Opta didn't tag end coordinates."""
    q = bdm.qmap(e)
    try:
        return bdm.to_m(float(q[bdm.Q_END_X]), float(q[bdm.Q_END_Y]))
    except (KeyError, TypeError, ValueError):
        return None, None


def build_goalkick_chains(basename, events, team_map):
    touches = event_ball_involvements(events)

    rows = []
    for i, e in enumerate(touches):
        if not is_goal_kick(e):
            continue
        cid = e["contestantId"]
        period = e["periodId"]
        x, y = bdm.to_m(e["x"], e["y"])
        end_x, end_y = goal_kick_end_xy(e)

        chain_ids, reached_shot, end_reason = [], False, "period_end"
        for ev in touches[i + 1:]:
            if ev["periodId"] != period:
                end_reason = "period_end"
                break
            if ev["contestantId"] != cid:
                end_reason = "turnover"
                break
            chain_ids.append(ev["id"])
            if ev["typeId"] in SHOT_TYPES:
                reached_shot = True
                end_reason = "shot"
                break
            if len(chain_ids) >= ACTION_LIMIT:
                end_reason = "action_limit"
                break
        else:
            end_reason = "period_end"

        rows.append({
            "match_file": basename,
            "event_id": e["id"],
            "period_id": period,
            "time_min": e["timeMin"], "time_sec": e["timeSec"],
            "contestant_id": cid,
            "team_name": team_map.get(cid, cid),
            "player_id": e.get("playerId"),
            "player_name": e.get("playerName"),
            "x": x, "y": y,
            "end_x": end_x, "end_y": end_y,
            "chain_n_actions": len(chain_ids),
            "chain_end_reason": end_reason,
            "reached_shot_within_5": reached_shot,
            "n_actions_to_shot": len(chain_ids) if reached_shot else None,
        })
    return rows


def main():
    files = sorted(glob.glob(os.path.join(DATA_DIR, "*.json")))
    print(f"Found {len(files)} match files in {DATA_DIR}")
    if not files:
        raise SystemExit(f"No match files found in {DATA_DIR}")

    cid_to_team = bdm.build_global_cid_to_team(files)
    print(f"Resolved {len(cid_to_team)} team names from {len(files)} matches")

    all_rows = []
    for fn in files:
        basename, events, team_map = bdm.load_match(fn, cid_to_team)
        all_rows.extend(build_goalkick_chains(basename, events, team_map))

    goalkicks = pd.DataFrame(all_rows)
    print(f"Found {len(goalkicks)} goal kicks across {goalkicks['match_file'].nunique()} matches, "
          f"{goalkicks['reached_shot_within_5'].mean():.1%} led to a shot within "
          f"{ACTION_LIMIT} actions")

    os.makedirs(CSV_DIR, exist_ok=True)
    for match_file, mgroup in goalkicks.groupby("match_file"):
        mgroup.to_csv(os.path.join(CSV_DIR, f"{match_file}_goalkicks.csv"), index=False)
    goalkicks.to_csv(os.path.join(CSV_DIR, "all_eredivisie_goalkicks.csv"), index=False)
    print(f"Wrote {goalkicks['match_file'].nunique()} per-match CSVs + aggregate")

    player_summary = goalkicks.groupby(["player_id", "player_name", "team_name"]).agg(
        matches=("match_file", "nunique"),
        goal_kicks=("reached_shot_within_5", "size"),
        shots_within_5=("reached_shot_within_5", "sum"),
        mean_actions_to_shot=("n_actions_to_shot", "mean"),
    ).reset_index()
    player_summary["shot_rate"] = player_summary["shots_within_5"] / player_summary["goal_kicks"]
    player_summary = player_summary.sort_values("shot_rate", ascending=False)
    player_summary.to_csv(os.path.join(CSV_DIR, "goalkick_player_summary.csv"), index=False)

    team_summary = goalkicks.groupby("team_name").agg(
        matches=("match_file", "nunique"),
        goal_kicks=("reached_shot_within_5", "size"),
        shots_within_5=("reached_shot_within_5", "sum"),
        mean_actions_to_shot=("n_actions_to_shot", "mean"),
    ).reset_index()
    team_summary["shot_rate"] = team_summary["shots_within_5"] / team_summary["goal_kicks"]
    team_summary["goal_kicks_per_match"] = team_summary["goal_kicks"] / team_summary["matches"]
    team_summary = team_summary.sort_values("shot_rate", ascending=False)
    team_summary.to_csv(os.path.join(CSV_DIR, "goalkick_team_summary.csv"), index=False)

    print("Wrote goalkick_player_summary.csv + goalkick_team_summary.csv")
    print("\nTeams by goal-kick -> shot-within-5-actions rate (min not yet filtered):")
    print(team_summary[["team_name", "matches", "goal_kicks", "shots_within_5", "shot_rate"]]
          .to_string(index=False))


if __name__ == "__main__":
    main()

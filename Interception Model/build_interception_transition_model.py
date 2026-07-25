"""
Interception Transition-Threat Model - Eredivisie 2025/26
=============================================================
Answers a different question than Disruption/disruption_value_model.py.
That model asks "how much attacking threat did this defensive action DENY
the opponent" (P(pass completes) x xT the pass would have added). This
model asks the flip side: "how much attacking threat did THIS TEAM go on
to CREATE in the moments right after winning the ball back with an
interception" -- i.e. the value of the transition/counter-attack the
interception kick-started, not the danger it snuffed out.

Approach:
  1. Find every Interception event (Opta typeId 8).
  2. Walk forward chronologically through that match/period's other
     ball-involvement events (anything with x/y + a contestantId) starting
     right after the interception, accumulating events by the SAME team.
     The chain -- the "transition" -- ends at the first of:
       - a touch by the opposing team (possession lost back = the counter
         is over, whatever happens after that is a new phase of play)
       - TRANSITION_WINDOW_SECONDS elapsed since the interception
       - the period ending
     This mirrors how counter-attack/transition windows are usually
     defined in the literature: the immediate 5-15s burst after a regain,
     not the whole subsequent possession (a team that regains the ball and
     patiently builds for two minutes isn't "transitioning" any more, and
     crediting the interceptor for a goal 40 passes later would be absurd).
  3. Value that chain using the season's existing Expected Threat action
     values (xT/xt_action_values.csv, already built and used by
     Disruption/disruption_value_model.py): sum positive_xT_added over
     every pass/carry/take-on in the chain. That total -- transition_threat
     -- is credited to the interceptor.

Data caveat: same Opta feed as the Disruption model, no freeze-frame data,
so "possession lost back to the opponent" is inferred purely from
contestantId changing on the next ball-involvement event, not from a
tracked ball-out-of-play state.

Usage: python3 build_interception_transition_model.py
Outputs:
  Interception Model/CSV/<match>_interception_transitions.csv
  Interception Model/CSV/all_eredivisie_interception_transitions.csv
  Interception Model/CSV/interception_transition_player_summary.csv
  Interception Model/CSV/interception_transition_team_summary.csv
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Disruption"))
import build_disruption_model as bdm  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "Events", "2025-2026")
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_DIR = os.path.join(OUT_DIR, "CSV")
XT_ACTION_VALUES_PATH = os.path.join(ROOT, "xT", "xt_action_values.csv")

T_INTERCEPTION = 8
SHOT_TYPES = {13, 14, 15, 16}  # Miss, Post, Attempt Saved, Goal

TRANSITION_WINDOW_SECONDS = 10.0  # how long a "transition" stays a transition after the regain


def event_ball_involvements(events):
    """Every event with a location and a team on it, chronologically
    ordered, i.e. anything that can plausibly start, continue or end a
    transition chain. Non-ball admin events (subs, cards, formation
    changes, period start/end) carry no x/y and are dropped, same filter
    build_disruption_model.py's extract_defensive_actions already uses."""
    rows = [e for e in events if e.get("x") is not None and e.get("contestantId")]
    rows.sort(key=lambda e: (e["periodId"], e["timeMin"] * 60 + e["timeSec"], e["eventId"]))
    return rows


def build_transition_chains(basename, events, team_map):
    """One row per interception: the chain of same-team event ids that
    followed it, plus how/why the chain ended."""
    touches = event_ball_involvements(events)

    rows = []
    for i, e in enumerate(touches):
        if e["typeId"] != T_INTERCEPTION:
            continue
        cid = e["contestantId"]
        period = e["periodId"]
        t0 = e["timeMin"] * 60 + e["timeSec"]
        x, y = bdm.to_m(e["x"], e["y"])

        chain_ids, reached_shot, end_reason = [], False, "period_end"
        last_t = t0
        for ev in touches[i + 1:]:
            if ev["periodId"] != period:
                end_reason = "period_end"
                break
            t = ev["timeMin"] * 60 + ev["timeSec"]
            if t - t0 > TRANSITION_WINDOW_SECONDS:
                end_reason = "window_expired"
                break
            if ev["contestantId"] != cid:
                end_reason = "turnover"
                break
            chain_ids.append(ev["id"])
            last_t = t
            if ev["typeId"] in SHOT_TYPES:
                reached_shot = True
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
            "chain_event_ids": chain_ids,
            "chain_n_actions": len(chain_ids),
            "chain_duration_sec": round(last_t - t0, 1) if chain_ids else 0.0,
            "chain_end_reason": end_reason,
            "chain_reached_shot": reached_shot,
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
        all_rows.extend(build_transition_chains(basename, events, team_map))

    interceptions = pd.DataFrame(all_rows)
    print(f"Found {len(interceptions)} interceptions across {interceptions['match_file'].nunique()} matches")

    print("Loading xT/xt_action_values.csv to value each transition chain...")
    xt = pd.read_csv(XT_ACTION_VALUES_PATH, usecols=["match_name", "event_id", "positive_xT_added"],
                      low_memory=False)
    xt_by_match = {mf: g.set_index("event_id")["positive_xT_added"] for mf, g in xt.groupby("match_name")}

    def chain_value(row):
        lookup = xt_by_match.get(row["match_file"])
        if lookup is None or not row["chain_event_ids"]:
            return 0.0
        return float(lookup.reindex(row["chain_event_ids"]).fillna(0.0).sum())

    interceptions["transition_threat"] = interceptions.apply(chain_value, axis=1)
    interceptions = interceptions.drop(columns=["chain_event_ids"])

    os.makedirs(CSV_DIR, exist_ok=True)
    for match_file, mgroup in interceptions.groupby("match_file"):
        mgroup.to_csv(os.path.join(CSV_DIR, f"{match_file}_interception_transitions.csv"), index=False)
    interceptions.to_csv(os.path.join(CSV_DIR, "all_eredivisie_interception_transitions.csv"), index=False)
    print(f"Wrote {interceptions['match_file'].nunique()} per-match CSVs + aggregate "
          f"({len(interceptions)} interceptions, "
          f"{(interceptions['chain_end_reason'] == 'turnover').mean():.1%} chains ended by turnover, "
          f"{interceptions['chain_reached_shot'].mean():.1%} reached a shot within "
          f"{TRANSITION_WINDOW_SECONDS:.0f}s)")

    player_summary = interceptions.groupby(["player_id", "player_name", "team_name"]).agg(
        matches=("match_file", "nunique"),
        interceptions=("transition_threat", "size"),
        total_transition_threat=("transition_threat", "sum"),
        mean_transition_threat=("transition_threat", "mean"),
        chains_reaching_shot=("chain_reached_shot", "sum"),
    ).reset_index()
    player_summary["transition_threat_per90"] = (
        player_summary["total_transition_threat"] / player_summary["matches"])
    player_summary["shot_rate"] = player_summary["chains_reaching_shot"] / player_summary["interceptions"]
    # xT deltas are tiny by construction (typical single-action xT ~0.001-0.02, same
    # scale note as Disruption/disruption_value_model.py); x1000 columns exist purely
    # so the numbers are readable in a table/chart.
    player_summary["total_transition_threat_x1000"] = player_summary["total_transition_threat"] * 1000
    player_summary["transition_threat_per90_x1000"] = player_summary["transition_threat_per90"] * 1000
    player_summary = player_summary.sort_values("total_transition_threat", ascending=False)
    player_summary.to_csv(os.path.join(CSV_DIR, "interception_transition_player_summary.csv"), index=False)

    team_summary = interceptions.groupby("team_name").agg(
        matches=("match_file", "nunique"),
        interceptions=("transition_threat", "size"),
        total_transition_threat=("transition_threat", "sum"),
        chains_reaching_shot=("chain_reached_shot", "sum"),
    ).reset_index()
    team_summary["transition_threat_per_match"] = (
        team_summary["total_transition_threat"] / team_summary["matches"])
    team_summary["total_transition_threat_x1000"] = team_summary["total_transition_threat"] * 1000
    team_summary["transition_threat_per_match_x1000"] = team_summary["transition_threat_per_match"] * 1000
    team_summary = team_summary.sort_values("transition_threat_per_match", ascending=False)
    team_summary.to_csv(os.path.join(CSV_DIR, "interception_transition_team_summary.csv"), index=False)

    print("Wrote interception_transition_player_summary.csv + interception_transition_team_summary.csv")
    print("\nTop 10 interceptors by transition threat created:")
    print(player_summary.head(10)[["player_name", "team_name", "matches", "interceptions",
                                    "total_transition_threat", "transition_threat_per90"]]
          .to_string(index=False))


if __name__ == "__main__":
    main()

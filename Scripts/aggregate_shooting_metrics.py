"""
Aggregated shooting metrics for the Eredivisie season.

Reads shot events (Danger/all_eredivisie_danger_models.csv) and minutes
played (GDA/gda_player_match_on_pitch.csv), and produces season-level
(not per-match) shooting metrics for every player and every team:
totals, per-30-minutes rates, and per-90-minutes rates.

Outputs:
  Aggregated/shooting_player_aggregated.csv
  Aggregated/shooting_team_aggregated.csv
"""

import csv
import os
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DANGER_PATH = os.path.join(ROOT, "Danger", "all_eredivisie_danger_models.csv")
MINUTES_PATH = os.path.join(ROOT, "GDA", "gda_player_match_on_pitch.csv")
TEAM_NAMES_PATH = os.path.join(ROOT, "xT", "xt_team_summary.csv")
OUT_DIR = os.path.join(ROOT, "Aggregated")
os.makedirs(OUT_DIR, exist_ok=True)


def rows(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def per(value, minutes, per_minutes):
    if minutes <= 0:
        return 0.0
    return value / minutes * per_minutes


def pct(numerator, denominator):
    if denominator <= 0:
        return 0.0
    return numerator / denominator * 100.0


# ---------------------------------------------------------------- team names
team_names = {r["contestant_id"]: r["team_name"] for r in rows(TEAM_NAMES_PATH)}


# ---------------------------------------------------------------- minutes played
player_minutes = defaultdict(float)
player_matches = defaultdict(set)
player_team_minutes = defaultdict(lambda: defaultdict(float))
player_last_match = {}
team_match_minutes = {}  # (match_file, contestant_id) -> match minutes

for r in rows(MINUTES_PATH):
    pid = r["player_id"]
    cid = r["contestant_id"]
    mins = num(r["minutes"])
    match_file = r["match_file"]

    player_minutes[pid] += mins
    player_matches[pid].add(match_file)
    player_team_minutes[pid][cid] += mins
    if pid not in player_last_match or match_file > player_last_match[pid][0]:
        player_last_match[pid] = (match_file, cid, r["player_name"])

    # Team's minutes of exposure = sum of each match's duration (not summed
    # player minutes, which would multiply by ~11 players on the pitch).
    team_match_minutes[(match_file, cid)] = num(r["match_minutes"])

team_minutes = defaultdict(float)
team_matches = defaultdict(set)
for (match_file, cid), mins in team_match_minutes.items():
    team_minutes[cid] += mins
    team_matches[cid].add(match_file)


# ---------------------------------------------------------------- shot events
SHOT_TYPES = {"13", "14", "15", "16"}  # Miss, Post, Attempt Saved, Goal


def new_bucket():
    return {
        "shots": 0,
        "goals": 0,
        "on_target": 0,
        "off_target": 0,
        "blocked": 0,
        "woodwork": 0,
        "penalties_taken": 0,
        "penalty_goals": 0,
        "xg": 0.0,
        "npxg": 0.0,
        "psxg": 0.0,
        "xgot": 0.0,
        "danger_score": 0.0,
    }


player_stats = defaultdict(new_bucket)
player_name_of = {}
team_stats = defaultdict(new_bucket)

for r in rows(DANGER_PATH):
    if r["type_id"] not in SHOT_TYPES:
        continue

    pid = r["player_id"]
    cid = r["contestant_id"]
    is_goal = num(r["is_goal"]) == 1
    is_post = num(r["is_post"]) == 1
    is_on_target = num(r["is_on_target"]) == 1
    is_blocked = num(r["is_outfield_block"]) == 1
    is_penalty = num(r["is_penalty"]) == 1
    xg = num(r["xg"])
    psxg = num(r["psxg"])
    xgot = num(r["xgot"])
    danger_score = num(r["danger_score"])

    player_name_of[pid] = r["player_name"]

    for bucket in (player_stats[pid], team_stats[cid]):
        bucket["shots"] += 1
        bucket["goals"] += 1 if is_goal else 0
        bucket["on_target"] += 1 if is_on_target else 0
        bucket["blocked"] += 1 if is_blocked else 0
        bucket["woodwork"] += 1 if is_post else 0
        bucket["off_target"] += 1 if (not is_on_target and not is_blocked and not is_post) else 0
        bucket["penalties_taken"] += 1 if is_penalty else 0
        bucket["penalty_goals"] += 1 if (is_penalty and is_goal) else 0
        bucket["xg"] += xg
        bucket["npxg"] += 0.0 if is_penalty else xg
        bucket["psxg"] += psxg
        bucket["xgot"] += xgot
        bucket["danger_score"] += danger_score


# ---------------------------------------------------------------- build rows
def build_row(minutes, matches, s, extra):
    row = dict(extra)
    row["minutes"] = round(minutes, 1)
    row["matches"] = matches
    row["shots_total"] = s["shots"]
    row["goals_total"] = s["goals"]
    row["non_penalty_goals_total"] = s["goals"] - s["penalty_goals"]
    row["shots_on_target_total"] = s["on_target"]
    row["shots_off_target_total"] = s["off_target"]
    row["shots_blocked_total"] = s["blocked"]
    row["shots_woodwork_total"] = s["woodwork"]
    row["penalties_taken_total"] = s["penalties_taken"]
    row["penalty_goals_total"] = s["penalty_goals"]
    row["xg_total"] = round(s["xg"], 3)
    row["npxg_total"] = round(s["npxg"], 3)
    row["psxg_total"] = round(s["psxg"], 3)
    row["xgot_total"] = round(s["xgot"], 3)
    row["danger_score_total"] = round(s["danger_score"], 3)
    row["goals_minus_xg_total"] = round(s["goals"] - s["xg"], 3)
    row["psxg_minus_xgot_total"] = round(s["psxg"] - s["xgot"], 3)

    row["shot_accuracy_pct"] = round(pct(s["on_target"], s["shots"]), 2)
    row["conversion_rate_pct"] = round(pct(s["goals"], s["shots"]), 2)
    row["xg_per_shot"] = round(s["xg"] / s["shots"], 3) if s["shots"] else 0.0

    for label, per_minutes in (("per30", 30), ("per90", 90)):
        row[f"shots_{label}"] = round(per(s["shots"], minutes, per_minutes), 3)
        row[f"goals_{label}"] = round(per(s["goals"], minutes, per_minutes), 3)
        row[f"shots_on_target_{label}"] = round(per(s["on_target"], minutes, per_minutes), 3)
        row[f"xg_{label}"] = round(per(s["xg"], minutes, per_minutes), 3)
        row[f"npxg_{label}"] = round(per(s["npxg"], minutes, per_minutes), 3)
        row[f"psxg_{label}"] = round(per(s["psxg"], minutes, per_minutes), 3)
        row[f"xgot_{label}"] = round(per(s["xgot"], minutes, per_minutes), 3)
        row[f"danger_score_{label}"] = round(per(s["danger_score"], minutes, per_minutes), 3)

    return row


PLAYER_COLUMNS = [
    "player_id", "player_name", "team", "teams", "matches", "minutes",
    "shots_total", "goals_total", "non_penalty_goals_total",
    "shots_on_target_total", "shots_off_target_total", "shots_blocked_total",
    "shots_woodwork_total", "penalties_taken_total", "penalty_goals_total",
    "xg_total", "npxg_total", "psxg_total", "xgot_total", "danger_score_total",
    "goals_minus_xg_total", "psxg_minus_xgot_total",
    "shot_accuracy_pct", "conversion_rate_pct", "xg_per_shot",
    "shots_per30", "goals_per30", "shots_on_target_per30", "xg_per30",
    "npxg_per30", "psxg_per30", "xgot_per30", "danger_score_per30",
    "shots_per90", "goals_per90", "shots_on_target_per90", "xg_per90",
    "npxg_per90", "psxg_per90", "xgot_per90", "danger_score_per90",
]

TEAM_COLUMNS = [
    "contestant_id", "team", "matches", "minutes",
    "shots_total", "goals_total", "non_penalty_goals_total",
    "shots_on_target_total", "shots_off_target_total", "shots_blocked_total",
    "shots_woodwork_total", "penalties_taken_total", "penalty_goals_total",
    "xg_total", "npxg_total", "psxg_total", "xgot_total", "danger_score_total",
    "goals_minus_xg_total", "psxg_minus_xgot_total",
    "shot_accuracy_pct", "conversion_rate_pct", "xg_per_shot",
    "shots_per30", "goals_per30", "shots_on_target_per30", "xg_per30",
    "npxg_per30", "psxg_per30", "xgot_per30", "danger_score_per30",
    "shots_per90", "goals_per90", "shots_on_target_per90", "xg_per90",
    "npxg_per90", "psxg_per90", "xgot_per90", "danger_score_per90",
]


player_rows = []
for pid, s in player_stats.items():
    minutes = player_minutes.get(pid, 0.0)
    matches = len(player_matches.get(pid, ()))
    _, last_cid, last_name = player_last_match.get(pid, (None, None, player_name_of.get(pid, "")))
    team_ids = [cid for cid, m in player_team_minutes[pid].items() if m > 0]
    team_names_played = [team_names.get(cid, cid) for cid in team_ids]

    extra = {
        "player_id": pid,
        "player_name": last_name or player_name_of.get(pid, ""),
        "team": team_names.get(last_cid, last_cid or ""),
        "teams": "; ".join(sorted(set(team_names_played))),
    }
    player_rows.append(build_row(minutes, matches, s, extra))

player_rows.sort(key=lambda r: r["goals_total"], reverse=True)

with open(os.path.join(OUT_DIR, "shooting_player_aggregated.csv"), "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=PLAYER_COLUMNS)
    writer.writeheader()
    writer.writerows(player_rows)


team_rows = []
for cid, s in team_stats.items():
    minutes = team_minutes.get(cid, 0.0)
    matches = len(team_matches.get(cid, ()))
    extra = {
        "contestant_id": cid,
        "team": team_names.get(cid, cid),
    }
    team_rows.append(build_row(minutes, matches, s, extra))

team_rows.sort(key=lambda r: r["goals_total"], reverse=True)

with open(os.path.join(OUT_DIR, "shooting_team_aggregated.csv"), "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=TEAM_COLUMNS)
    writer.writeheader()
    writer.writerows(team_rows)

print(f"Wrote {len(player_rows)} players -> {os.path.join(OUT_DIR, 'shooting_player_aggregated.csv')}")
print(f"Wrote {len(team_rows)} teams -> {os.path.join(OUT_DIR, 'shooting_team_aggregated.csv')}")

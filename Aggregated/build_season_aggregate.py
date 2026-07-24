"""
Build one Wyscout-style player-season and team-season stat sheet for a given
Eredivisie season: the standard counting stats (passing, crossing, duels,
defensive actions, discipline, goalkeeping, shooting, assists/xA) computed
directly from the raw Opta event stream in Events/<season>/*.json, plus the
"new metrics" this repo already has models for (xT progression value, GDA
on-pitch + action impact, disruption value, expected box entries, pass-shot
value, hot-zone passing, expected-completion crossing value) joined on top.

Event-code reference used below (typeId / qualifierId), confirmed against
this repo's own Events/2025-2026 data -- see Aggregated/<season>/README.md
for how each was verified (frequency sanity checks, paired-event checks for
fouls/offsides, distance/touchline checks for qualifiers):
  typeId   1 Pass            2 Offside Pass      3 Take On        4 Foul
           6 Corner Awarded   7 Tackle            8 Interception   10 Save
           11 Claim           12 Clearance        13 Miss          14 Post
           15 Attempt Saved   16 Goal             17 Card          41 Punch
           44 Aerial Duel     49 Ball Recovery    50 Dispossessed  51 Error
           52 Keeper Pick-up  54 Smother          55 Offside Provoked
           58 Penalty Faced   59 Keeper Sweeper
  qualifierId  1 Long ball  2 Cross  3 Through ball  5 Free kick  6 Corner
               15 Head  20 Right foot  31 Yellow card  32 Second yellow
               33 Red card  72 Left foot  107 Throw-in  140/141 pass end x/y
               195 Pull back  212 length (m)  213 angle (rad)

This does NOT recompute the xT/GDA/disruption/box-entry/crossing models
themselves -- those already exist per metric folder (see their own meta.json
files) and, for SEASON = "2025-2026", already covered the full season before
this script existed. This script computes the Wyscout-style raw stats itself
(nothing to join for those -- no per-metric folder already has them), and
joins the advanced-metric folders on top.

No composite score is produced -- see Aggregated/<season>/README.md for why.

Usage: python3 build_season_aggregate.py [season]   (default: 2025-2026)
"""
import csv
import glob
import json
import os
import statistics
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEASON = sys.argv[1] if len(sys.argv) > 1 else "2025-2026"
OUT_DIR = os.path.join(ROOT, "Aggregated", SEASON)
MIN_MINUTES_RELIABLE = 450  # ~5 full matches; same cutoff netlify-app/generate_data.py uses for GDA

X_SCALE, Y_SCALE = 1.05, 0.68           # Opta 0-100 units -> metres (105x68 pitch), matches Disruption model
GOAL_X, GOAL_Y = 105.0, 34.0
PROG_OWN_HALF_M = 27.432    # 30 yards: both ends in own half
PROG_TO_ATT_HALF_M = 13.716  # 15 yards: crosses into attacking half
PROG_ATT_HALF_M = 9.144     # 10 yards: both ends in attacking half
FINAL_THIRD_X = 200.0 / 3.0

Q_LONG_BALL, Q_CROSS, Q_THROUGH_BALL = 1, 2, 3
Q_END_X, Q_END_Y = 140, 141
Q_YELLOW, Q_SECOND_YELLOW, Q_RED = 31, 32, 33

RAW_COUNT_FIELDS = [
    "passes_attempted", "passes_completed", "progressive_passes", "forward_passes",
    "passes_into_final_third", "long_balls_attempted", "long_balls_completed",
    "crosses_attempted", "crosses_completed", "through_balls_attempted", "through_balls_completed",
    "take_ons_attempted", "take_ons_successful",
    "tackles_attempted", "tackles_won", "interceptions", "clearances",
    "aerial_duels", "aerial_duels_won", "ball_recoveries", "dispossessed", "errors",
    "fouls_committed", "fouls_won", "offsides", "yellow_cards", "red_cards",
    "key_passes", "assists", "xa",
    "gk_saves", "gk_claims", "gk_punches", "gk_pickups", "gk_smothers",
    "gk_penalties_faced", "gk_sweeper_actions",
]


def load(*parts):
    path = os.path.join(ROOT, *parts)
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def num(v, default=0.0):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def pct(numerator, denominator):
    return round(numerator / denominator * 100, 2) if denominator else ""


def minutes_bucket(m):
    if m < 450:
        return "<450"
    if m < 900:
        return "450-900"
    if m < 1500:
        return "900-1500"
    if m < 2000:
        return "1500-2000"
    return "2000+"


def zscores(values):
    """Return {key: z} for a dict of key->value, using the sample's own mean/stdev."""
    xs = list(values.values())
    if len(xs) < 2:
        return {k: None for k in values}
    mean = statistics.fmean(xs)
    sd = statistics.pstdev(xs)
    if sd == 0:
        return {k: 0.0 for k in values}
    return {k: (v - mean) / sd for k, v in values.items()}


def dist_to_goal_m(x, y):
    xm, ym = x * X_SCALE, y * Y_SCALE
    return ((GOAL_X - xm) ** 2 + (GOAL_Y - ym) ** 2) ** 0.5


def is_progressive_pass(sx, sy, ex, ey):
    gain = dist_to_goal_m(sx, sy) - dist_to_goal_m(ex, ey)
    if sx < 50 and ex < 50:
        return gain >= PROG_OWN_HALF_M
    if sx < 50 <= ex:
        return gain >= PROG_TO_ATT_HALF_M
    return gain >= PROG_ATT_HALF_M


def qmap(e):
    return {q["qualifierId"]: q.get("value") for q in e.get("qualifier", []) or []}


def new_counter():
    return {f: 0 for f in RAW_COUNT_FIELDS}


def process_raw_events(events_dir, team_name_by_cid, danger_by_match):
    """One pass over every match file: league table + Wyscout-style player counts."""
    league = {}
    player_counts = {}

    def acc_for(pid, pname, team):
        row = player_counts.setdefault(pid, {"player_name": pname, "team_name": team, **new_counter()})
        return row

    for path in sorted(glob.glob(os.path.join(events_dir, "*.json"))):
        raw = json.load(open(path, encoding="utf-8"))
        fn = os.path.basename(path)
        stem = fn[:-5]
        _, fixture = stem.split("_", 1)
        home, away = fixture.split(" - ", 1)
        scores = raw.get("matchDetails", {}).get("scores", {})
        ft = scores.get("total", scores.get("ft", {}))
        h, a = ft.get("home"), ft.get("away")
        if h is not None and a is not None:
            h, a = int(h), int(a)
            for name, gf, ga in ((home, h, a), (away, a, h)):
                c = league.setdefault(name, {"team_name": name, "matches": 0, "wins": 0, "draws": 0,
                                              "losses": 0, "goals_for": 0, "goals_against": 0})
                c["matches"] += 1
                c["goals_for"] += gf
                c["goals_against"] += ga
                if gf > ga:
                    c["wins"] += 1
                elif gf == ga:
                    c["draws"] += 1
                else:
                    c["losses"] += 1

        events = raw.get("event", [])
        for e in events:
            pid = e.get("playerId")
            if not pid:
                continue
            team = team_name_by_cid.get(e.get("contestantId"), e.get("contestantId"))
            row = acc_for(pid, e.get("playerName", ""), team)
            t = e["typeId"]
            outcome = e.get("outcome")
            q = qmap(e)

            if t == 1:  # Pass
                row["passes_attempted"] += 1
                completed = outcome == 1
                if completed:
                    row["passes_completed"] += 1
                if Q_LONG_BALL in q:
                    row["long_balls_attempted"] += 1
                    if completed:
                        row["long_balls_completed"] += 1
                if Q_CROSS in q:
                    row["crosses_attempted"] += 1
                    if completed:
                        row["crosses_completed"] += 1
                if Q_THROUGH_BALL in q:
                    row["through_balls_attempted"] += 1
                    if completed:
                        row["through_balls_completed"] += 1
                if completed and Q_END_X in q:
                    ex, ey = num(q[Q_END_X]), num(q.get(Q_END_Y))
                    sx, sy = num(e.get("x")), num(e.get("y"))
                    if ex > sx:
                        row["forward_passes"] += 1
                    if sx < FINAL_THIRD_X <= ex:
                        row["passes_into_final_third"] += 1
                    if is_progressive_pass(sx, sy, ex, ey):
                        row["progressive_passes"] += 1
            elif t == 3:  # Take On
                row["take_ons_attempted"] += 1
                if outcome == 1:
                    row["take_ons_successful"] += 1
            elif t == 4:  # Foul
                if outcome == 1:
                    row["fouls_won"] += 1
                else:
                    row["fouls_committed"] += 1
            elif t == 7:  # Tackle
                row["tackles_attempted"] += 1
                if outcome == 1:
                    row["tackles_won"] += 1
            elif t == 8:
                row["interceptions"] += 1
            elif t == 12:
                row["clearances"] += 1
            elif t == 17:  # Card
                if Q_RED in q or Q_SECOND_YELLOW in q:
                    row["red_cards"] += 1
                elif Q_YELLOW in q:
                    row["yellow_cards"] += 1
            elif t == 44:  # Aerial duel
                row["aerial_duels"] += 1
                if outcome == 1:
                    row["aerial_duels_won"] += 1
            elif t == 49:
                row["ball_recoveries"] += 1
            elif t == 50:
                row["dispossessed"] += 1
            elif t == 51:
                row["errors"] += 1
            elif t == 55:  # Offside Provoked -> the attacker flagged offside
                row["offsides"] += 1
            elif t == 10:
                row["gk_saves"] += 1
            elif t == 11:
                row["gk_claims"] += 1
            elif t == 41:
                row["gk_punches"] += 1
            elif t == 52:
                row["gk_pickups"] += 1
            elif t == 54:
                row["gk_smothers"] += 1
            elif t == 58:
                row["gk_penalties_faced"] += 1
            elif t == 59:
                row["gk_sweeper_actions"] += 1

        # Key passes / assists / xA: link each shot to the nearest preceding
        # completed same-team pass within 4 events (same convention already
        # used by netlify-app/generate_data.py).
        for sr in danger_by_match.get(fn, []):
            idx = int(num(sr["event_index"]))
            cid = sr["contestant_id"]
            for e in reversed(events[max(0, idx - 4):idx]):
                if e.get("contestantId") == cid and e.get("typeId") == 1 and e.get("outcome") == 1:
                    pid = e.get("playerId")
                    if not pid:
                        break
                    team = team_name_by_cid.get(cid, cid)
                    row = acc_for(pid, e.get("playerName", ""), team)
                    row["key_passes"] += 1
                    row["xa"] += num(sr["xg"])
                    if num(sr["is_goal"]) == 1:
                        row["assists"] += 1
                    break

    for c in league.values():
        c["goal_diff"] = c["goals_for"] - c["goals_against"]
        c["points"] = c["wins"] * 3 + c["draws"]
    return league, player_counts


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    events_dir = os.path.join(ROOT, "Events", SEASON)

    # --- shared team-name <-> contestant_id map (from xT, which has both) ---
    xt_team_rows = load("xT", "xt_team_summary.csv")
    team_name_by_cid = {r["contestant_id"]: r["team_name"] for r in xt_team_rows}

    danger_rows = load("Danger", "all_eredivisie_danger_models.csv")
    danger_by_match = {}
    for r in danger_rows:
        danger_by_match.setdefault(r["match_file"], []).append(r)

    print("Parsing raw events for Wyscout-style counting stats...")
    league_table, raw_counts = process_raw_events(events_dir, team_name_by_cid, danger_by_match)

    # =========================== PLAYER TABLE ===========================
    players = {}  # key: player_id if known else ("__noid__", name, team)

    def get_row(player_id, player_name, team_name):
        key = player_id if player_id else ("__noid__", player_name, team_name)
        return players.setdefault(key, {
            "player_id": player_id or "", "player_name": player_name, "team_name": team_name,
        })

    # Wyscout-style raw counts, computed above straight from events
    for pid, r in raw_counts.items():
        row = get_row(pid, r["player_name"], r["team_name"])
        row.update({f: r[f] for f in RAW_COUNT_FIELDS})

    # GDA: canonical minutes/matches + on-pitch goal-difference-added
    for r in load("GDA", "gda_player_summary.csv"):
        team = team_name_by_cid.get(r["contestant_id"], r["contestant_id"])
        row = get_row(r["player_id"], r["player_name"], team)
        row.update({
            "matches": int(num(r["matches"])),
            "minutes": round(num(r["minutes"]), 1),
            "goal_difference_added": round(num(r["goal_difference_added"]), 4),
            "goal_difference_added_per90": round(num(r["goal_difference_added_per90"]), 4),
            "action_gda_actual": round(num(r["action_gda_actual"]), 4),
        })

    # xT: progression / creativity via possession value added
    for r in load("xT", "xt_player_summary.csv"):
        row = get_row(r["player_id"], r["player_name"], r["team_name"])
        row.update({
            "actions": int(num(r["actions"])),
            "xt_passes": int(num(r["passes"])),
            "carries": int(num(r["carries"])),
            "xt_take_ons": int(num(r["take_ons"])),
            "xT_added": round(num(r["xT_added"]), 4),
            "positive_xT_added": round(num(r["positive_xT_added"]), 4),
            "xT_per_100_actions": round(num(r["xT_per_100_actions"]), 4),
        })

    # Disruption: defensive disruption (volume + shot-value-denied version)
    disr_by_id = {r["player_id"]: r for r in load("Disruption", "CSV", "disruption_player_summary.csv")}
    disrv_by_id = {r["player_id"]: r for r in load("Disruption", "CSV", "disruption_value_player_summary.csv")}
    for pid, r in disr_by_id.items():
        row = get_row(pid, r["player_name"], r["team_name"])
        row.update({
            "disruption_actions_linked": int(num(r["actions_linked"])),
            "disruption_total": round(num(r["total_disruption"]), 4),
            "disruption_per90": round(num(r["disruption_per90"]), 4),
        })
    for pid, r in disrv_by_id.items():
        row = get_row(pid, r["player_name"], r["team_name"])
        row.update({
            "disruption_value_total": round(num(r["total_disruption_value"]), 6),
            "disruption_value_per90": round(num(r["disruption_value_per90"]), 6),
        })

    # Danger (shot-level): aggregate to player -- shots, on target, goals, xG, PSxG, danger score
    danger_agg = {}
    for r in danger_rows:
        pid = r["player_id"]
        team = team_name_by_cid.get(r["contestant_id"], r["contestant_id"])
        d = danger_agg.setdefault(pid, {"player_name": r["player_name"], "team_name": team,
                                          "shots": 0, "shots_on_target": 0, "goals": 0,
                                          "xg": 0.0, "psxg": 0.0, "danger_score": 0.0})
        d["shots"] += 1
        d["shots_on_target"] += int(num(r["is_on_target"]))
        d["goals"] += int(num(r["is_goal"]))
        d["xg"] += num(r["xg"])
        d["psxg"] += num(r["psxg"])
        d["danger_score"] += num(r["danger_score"])
    for pid, d in danger_agg.items():
        row = get_row(pid, d["player_name"], d["team_name"])
        row.update({
            "shots": d["shots"], "shots_on_target": d["shots_on_target"], "goals": d["goals"],
            "xg": round(d["xg"], 4), "psxg": round(d["psxg"], 4),
            "danger_score": round(d["danger_score"], 4),
        })

    # Name+team-keyed sources (no player_id available in these files)
    nameteam_index = {(r["player_name"], r["team_name"]): r["player_id"] for r in players.values()
                       if r["player_id"]}

    def get_row_nameteam(player_name, team_name):
        pid = nameteam_index.get((player_name, team_name))
        if pid:
            return players[pid]
        return get_row(None, player_name, team_name)

    for r in load("Box Entry Models", "box_entry_player_summary.csv"):
        row = get_row_nameteam(r["player"], r["team"])
        row.update({
            "box_entry_attempts": int(num(r["attempts"])),
            "box_entry_expected": round(num(r["expected_box_entries"]), 4),
            "box_entry_actual": round(num(r["actual_box_entries"]), 4),
            "box_entry_added_value": round(num(r["added_value"]), 4),
        })

    for r in load("Box Entry Models", "pass_shot_value_player_summary.csv"):
        row = get_row_nameteam(r["player"], r["team"])
        row.update({
            "psv_attempts": int(num(r["attempts"])),
            "psv_total": round(num(r["total_psv"]), 4),
            "psv_actual_shot_xg": round(num(r["actual_shot_xg"]), 4),
        })

    for r in load("Box Entry Models", "hot_zone_player_summary.csv"):
        row = get_row_nameteam(r["player"], r["team"])
        row.update({
            "hotzone_attempts": int(num(r["attempts"])),
            "hotzone_completed": int(num(r["completed"])),
            "hotzone_expected": round(num(r["expected"]), 4),
            "hotzone_completion_rate": round(num(r["completion_rate"]), 4),
            "hotzone_added_value": round(num(r["added_value"]), 4),
        })

    for r in load("Cross Models", "xp_player_leaderboard_eredivisie.csv"):
        row = get_row_nameteam(r["player"], r["team"])
        row.update({
            "xp_crosses": int(num(r["crosses"])),
            "xp_cross_completed": int(num(r["completed"])),
            "xp_cross_expected_completed": round(num(r["expected_completed"]), 4),
            "xp_cross_completion_rate": round(num(r["completion_rate"]), 4),
            "xp_cross_expected_completion_rate": round(num(r["expected_completion_rate"]), 4),
            "xp_cross_added_value": round(num(r["xP_added_value"]), 4),
        })

    player_rows = list(players.values())

    # Completion-rate percentages (ratio of two raw counts, not a per-90 rate)
    for row in player_rows:
        row["pass_completion_pct"] = pct(row.get("passes_completed", 0), row.get("passes_attempted", 0))
        row["long_ball_completion_pct"] = pct(row.get("long_balls_completed", 0), row.get("long_balls_attempted", 0))
        row["cross_completion_pct"] = pct(row.get("crosses_completed", 0), row.get("crosses_attempted", 0))
        row["through_ball_completion_pct"] = pct(row.get("through_balls_completed", 0), row.get("through_balls_attempted", 0))
        row["take_on_success_pct"] = pct(row.get("take_ons_successful", 0), row.get("take_ons_attempted", 0))
        row["tackle_win_pct"] = pct(row.get("tackles_won", 0), row.get("tackles_attempted", 0))
        row["aerial_duel_win_pct"] = pct(row.get("aerial_duels_won", 0), row.get("aerial_duels", 0))
        row["shot_on_target_pct"] = pct(row.get("shots_on_target", 0), row.get("shots", 0))

    # Reliability flags + per-90 rates derived from GDA minutes (the one minutes source)
    per90_bases = RAW_COUNT_FIELDS + [
        "xT_added", "box_entry_added_value", "psv_total", "xp_cross_added_value",
        "danger_score", "xg", "goals", "shots", "shots_on_target",
    ]
    for row in player_rows:
        minutes = row.get("minutes", 0.0)
        row["minutes_bucket"] = minutes_bucket(minutes)
        row["reliable_sample"] = minutes >= MIN_MINUTES_RELIABLE
        for base in per90_bases:
            if base in row and minutes > 0:
                row[f"{base}_per90"] = round(row[base] * 90.0 / minutes, 4)

    # z-scores among the reliable-sample subset, for a handful of core per-90 rates
    # (not a composite -- see README -- just standardised components for later use)
    z_bases = ["xT_added_per90", "box_entry_added_value_per90", "disruption_value_per90",
               "danger_score_per90", "progressive_passes_per90"]
    for base in z_bases:
        values = {i: row[base] for i, row in enumerate(player_rows)
                  if row.get("reliable_sample") and base in row}
        z = zscores(values)
        for i, row in enumerate(player_rows):
            if i in z:
                row[f"{base}_z"] = round(z[i], 3) if z[i] is not None else ""

    player_cols = [
        "player_id", "player_name", "team_name", "matches", "minutes", "minutes_bucket",
        "reliable_sample",

        # --- Wyscout-style: passing ---
        "passes_attempted", "passes_completed", "pass_completion_pct",
        "passes_attempted_per90", "passes_completed_per90",
        "forward_passes", "forward_passes_per90",
        "progressive_passes", "progressive_passes_per90", "progressive_passes_per90_z",
        "passes_into_final_third", "passes_into_final_third_per90",
        "long_balls_attempted", "long_balls_completed", "long_ball_completion_pct", "long_balls_attempted_per90",
        "crosses_attempted", "crosses_completed", "cross_completion_pct", "crosses_attempted_per90",
        "through_balls_attempted", "through_balls_completed", "through_ball_completion_pct",
        "through_balls_attempted_per90",
        "key_passes", "key_passes_per90", "assists", "assists_per90", "xa", "xa_per90",

        # --- Wyscout-style: dribbling / duels / defensive actions ---
        "take_ons_attempted", "take_ons_successful", "take_on_success_pct", "take_ons_attempted_per90",
        "tackles_attempted", "tackles_won", "tackle_win_pct", "tackles_attempted_per90",
        "interceptions", "interceptions_per90",
        "clearances", "clearances_per90",
        "aerial_duels", "aerial_duels_won", "aerial_duel_win_pct", "aerial_duels_per90",
        "ball_recoveries", "ball_recoveries_per90",
        "dispossessed", "dispossessed_per90",
        "errors", "errors_per90",

        # --- Wyscout-style: discipline ---
        "fouls_committed", "fouls_committed_per90", "fouls_won", "fouls_won_per90",
        "offsides", "offsides_per90", "yellow_cards", "red_cards",

        # --- Wyscout-style: shooting ---
        "shots", "shots_on_target", "shot_on_target_pct", "goals",
        "shots_per90", "shots_on_target_per90", "goals_per90",
        "xg", "xg_per90", "psxg",

        # --- Wyscout-style: goalkeeping ---
        "gk_saves", "gk_saves_per90", "gk_claims", "gk_punches", "gk_pickups",
        "gk_smothers", "gk_penalties_faced", "gk_sweeper_actions",

        # --- New metrics: progression / possession value ---
        "actions", "xt_passes", "carries", "xt_take_ons",
        "xT_added", "xT_added_per90", "positive_xT_added", "xT_per_100_actions", "xT_added_per90_z",
        "box_entry_attempts", "box_entry_expected", "box_entry_actual", "box_entry_added_value",
        "box_entry_added_value_per90", "box_entry_added_value_per90_z",
        "psv_attempts", "psv_total", "psv_total_per90", "psv_actual_shot_xg",
        "hotzone_attempts", "hotzone_completed", "hotzone_expected", "hotzone_completion_rate",
        "hotzone_added_value",
        "xp_crosses", "xp_cross_completed", "xp_cross_expected_completed", "xp_cross_completion_rate",
        "xp_cross_expected_completion_rate", "xp_cross_added_value", "xp_cross_added_value_per90",

        # --- New metrics: defensive disruption value / overall impact ---
        "disruption_actions_linked", "disruption_total", "disruption_per90",
        "disruption_value_total", "disruption_value_per90", "disruption_value_per90_z",
        "danger_score", "danger_score_per90", "danger_score_per90_z",
        "goal_difference_added", "goal_difference_added_per90", "action_gda_actual",
    ]

    player_rows.sort(key=lambda r: r.get("minutes", 0), reverse=True)
    with open(os.path.join(OUT_DIR, "player_season_aggregated.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=player_cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(player_rows)

    # ============================ TEAM TABLE =============================
    teams = league_table

    # Roll the Wyscout-style raw counts up from player rows (sum by team)
    team_raw_sums = {}
    for row in player_rows:
        t = team_raw_sums.setdefault(row["team_name"], new_counter())
        for f in RAW_COUNT_FIELDS:
            t[f] += row.get(f, 0) or 0
    for team_name, sums in team_raw_sums.items():
        t = teams.setdefault(team_name, {"team_name": team_name})
        t.update(sums)
        t["pass_completion_pct"] = pct(sums["passes_completed"], sums["passes_attempted"])
        t["cross_completion_pct"] = pct(sums["crosses_completed"], sums["crosses_attempted"])
        t["tackle_win_pct"] = pct(sums["tackles_won"], sums["tackles_attempted"])
        t["aerial_duel_win_pct"] = pct(sums["aerial_duels_won"], sums["aerial_duels"])

    for r in xt_team_rows:
        t = teams.setdefault(r["team_name"], {"team_name": r["team_name"]})
        t.update({
            "actions": int(num(r["actions"])),
            "carries": int(num(r["carries"])),
            "xT_added": round(num(r["xT_added"]), 4),
            "positive_xT_added": round(num(r["positive_xT_added"]), 4),
            "xT_per_action": round(num(r["xT_per_action"]), 5),
        })

    danger_team = {}
    for r in danger_rows:
        team = team_name_by_cid.get(r["contestant_id"], r["contestant_id"])
        d = danger_team.setdefault(team, {"shots": 0, "goals": 0, "xg": 0.0, "danger_score": 0.0})
        d["shots"] += 1
        d["goals"] += int(num(r["is_goal"]))
        d["xg"] += num(r["xg"])
        d["danger_score"] += num(r["danger_score"])
    for team, d in danger_team.items():
        t = teams.setdefault(team, {"team_name": team})
        t.update({"goals_from_shots": d["goals"],
                  "xg": round(d["xg"], 4), "danger_score": round(d["danger_score"], 4)})

    for r in load("Disruption", "CSV", "disruption_team_summary.csv"):
        t = teams.setdefault(r["team_name"], {"team_name": r["team_name"]})
        t.update({
            "disruption_total": round(num(r["total_disruption"]), 4),
            "disruption_per_match": round(num(r["disruption_per_match"]), 4),
        })
    for r in load("Disruption", "CSV", "disruption_value_team_summary.csv"):
        t = teams.setdefault(r["team_name"], {"team_name": r["team_name"]})
        t.update({
            "disruption_value_total": round(num(r["total_disruption_value"]), 6),
            "disruption_value_per_match": round(num(r["disruption_value_per_match"]), 6),
        })

    for r in load("Box Entry Models", "box_entry_team_summary.csv"):
        t = teams.setdefault(r["team"], {"team_name": r["team"]})
        t.update({
            "box_entry_attempts": int(num(r["attempts"])),
            "box_entry_expected": round(num(r["expected_box_entries"]), 4),
            "box_entry_actual": round(num(r["actual_box_entries"]), 4),
            "box_entry_added_value": round(num(r["added_value"]), 4),
        })

    for r in load("Cross Models", "xp_team_leaderboard_eredivisie.csv"):
        t = teams.setdefault(r["team"], {"team_name": r["team"]})
        t.update({
            "xp_crosses": int(num(r["crosses"])),
            "xp_cross_completed": int(num(r["completed"])),
            "xp_cross_expected_completed": round(num(r["expected_completed"]), 4),
            "xp_cross_completion_rate": round(num(r["completion_rate"]), 4),
            "xp_cross_expected_completion_rate": round(num(r["expected_completion_rate"]), 4),
            "xp_cross_added_value": round(num(r["xP_added_value"]), 4),
        })

    for r in load("Analysis", "Coach Profiling", "team_metrics_aggregated.csv"):
        t = teams.setdefault(r["team"], {"team_name": r["team"]})
        t.update({
            "style_long_ball_pct": round(num(r["long_ball_pct"]), 4),
            "style_deep_circulation_pct": round(num(r["deep_circ_pct"]), 4),
            "style_wing_pct": round(num(r["wing_pct"]), 4),
            "style_territory": round(num(r["territory"]), 3),
            "style_cross_pct": round(num(r["cross_pct"]), 4),
            "style_ppda": round(num(r["ppda"]), 3),
            "style_low_block_pct": round(num(r["low_block"]), 4),
            "style_counters_per90": round(num(r["counters_per90"]), 3),
        })

    for r in load("Analysis", "Formation", "formation_metrics_agg.csv"):
        t = teams.setdefault(r["team"], {"team_name": r["team"]})
        t.update({
            "dominant_back_line": r.get("dominant_back_line", ""),
            "top_formation": r.get("top_formation", ""),
        })

    team_rows = list(teams.values())
    team_cols = [
        "team_name", "matches", "wins", "draws", "losses", "goals_for", "goals_against",
        "goal_diff", "points",

        "passes_attempted", "passes_completed", "pass_completion_pct", "progressive_passes",
        "forward_passes", "passes_into_final_third",
        "long_balls_attempted", "long_balls_completed",
        "crosses_attempted", "crosses_completed", "cross_completion_pct",
        "through_balls_attempted", "through_balls_completed",
        "key_passes", "assists", "xa",
        "take_ons_attempted", "take_ons_successful",
        "tackles_attempted", "tackles_won", "tackle_win_pct", "interceptions", "clearances",
        "aerial_duels", "aerial_duels_won", "aerial_duel_win_pct",
        "ball_recoveries", "dispossessed", "errors",
        "fouls_committed", "fouls_won", "offsides", "yellow_cards", "red_cards",

        "actions", "carries", "xT_added", "positive_xT_added", "xT_per_action",
        "shots", "goals_from_shots", "xg", "danger_score",
        "disruption_total", "disruption_per_match",
        "disruption_value_total", "disruption_value_per_match",
        "box_entry_attempts", "box_entry_expected", "box_entry_actual", "box_entry_added_value",
        "xp_crosses", "xp_cross_completed", "xp_cross_expected_completed", "xp_cross_completion_rate",
        "xp_cross_expected_completion_rate", "xp_cross_added_value",
        "style_long_ball_pct", "style_deep_circulation_pct", "style_wing_pct", "style_territory",
        "style_cross_pct", "style_ppda", "style_low_block_pct", "style_counters_per90",
        "dominant_back_line", "top_formation",
    ]
    team_rows.sort(key=lambda r: r.get("points", 0), reverse=True)
    with open(os.path.join(OUT_DIR, "team_season_aggregated.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=team_cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(team_rows)

    print(f"Wrote {len(player_rows)} players and {len(team_rows)} teams to {OUT_DIR}")


if __name__ == "__main__":
    main()

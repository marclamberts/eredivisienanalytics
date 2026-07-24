"""
Consolidate the existing per-metric season outputs (xT, GDA, Danger, Disruption,
Box Entry, Cross Models, Coach Profiling, Formation) into one player-season and
one team-season table for a given Eredivisie season.

This does NOT recompute any underlying model (xG, xT, disruption value, ...).
Those already exist per metric folder and, for SEASON = "2025-2026", already
reflect the full season (309/309 matches, Aug 2025 - May 2026). This script only
joins them on shared keys (player_id where available, otherwise player_name +
team_name; contestant_id / team_name for teams) and derives the per-90 figures
and minutes-based reliability flags that are missing from the per-metric files.

Deliberately NOT included: a single weighted composite score. Combining these
components into one number is a modelling decision (which weights, validated
how) that hasn't been tested yet -- see Aggregated/<season>/README.md. What
this script produces is the clean, joined base a composite would be built on.

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


def load(*parts):
    path = os.path.join(ROOT, *parts)
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def num(v, default=0.0):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


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


def build_league_table(events_dir):
    """Points/GF/GA table computed straight from matchDetails scores."""
    clubs = {}
    for path in sorted(glob.glob(os.path.join(events_dir, "*.json"))):
        raw = json.load(open(path, encoding="utf-8"))
        stem = os.path.basename(path)[:-5]
        _, fixture = stem.split("_", 1)
        home, away = fixture.split(" - ", 1)
        scores = raw.get("matchDetails", {}).get("scores", {})
        ft = scores.get("total", scores.get("ft", {}))
        h, a = ft.get("home"), ft.get("away")
        if h is None or a is None:
            continue
        h, a = int(h), int(a)
        for name, gf, ga in ((home, h, a), (away, a, h)):
            c = clubs.setdefault(name, {"team_name": name, "matches": 0, "wins": 0, "draws": 0,
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
    for c in clubs.values():
        c["goal_diff"] = c["goals_for"] - c["goals_against"]
        c["points"] = c["wins"] * 3 + c["draws"]
    return clubs


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    events_dir = os.path.join(ROOT, "Events", SEASON)

    # --- shared team-name <-> contestant_id map (from xT, which has both) ---
    xt_team_rows = load("xT", "xt_team_summary.csv")
    team_name_by_cid = {r["contestant_id"]: r["team_name"] for r in xt_team_rows}

    # =========================== PLAYER TABLE ===========================
    players = {}  # key: player_id if known else ("name", "name","team")

    def get_row(player_id, player_name, team_name):
        key = player_id if player_id else ("__noid__", player_name, team_name)
        return players.setdefault(key, {
            "player_id": player_id or "", "player_name": player_name, "team_name": team_name,
        })

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
            "passes": int(num(r["passes"])),
            "carries": int(num(r["carries"])),
            "take_ons": int(num(r["take_ons"])),
            "xT_added": round(num(r["xT_added"]), 4),
            "positive_xT_added": round(num(r["positive_xT_added"]), 4),
            "xT_per_100_actions": round(num(r["xT_per_100_actions"]), 4),
        })

    # Disruption: defensive disruption (volume + shot-value-denied version)
    disr_by_id = {}
    for r in load("Disruption", "CSV", "disruption_player_summary.csv"):
        disr_by_id[r["player_id"]] = r
    disrv_by_id = {}
    for r in load("Disruption", "CSV", "disruption_value_player_summary.csv"):
        disrv_by_id[r["player_id"]] = r
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

    # Danger (shot-level): aggregate to player -- shots, goals, xG, PSxG, danger score
    danger_agg = {}
    for r in load("Danger", "all_eredivisie_danger_models.csv"):
        pid = r["player_id"]
        team = team_name_by_cid.get(r["contestant_id"], r["contestant_id"])
        d = danger_agg.setdefault(pid, {"player_name": r["player_name"], "team_name": team,
                                          "shots": 0, "goals": 0, "xg": 0.0, "psxg": 0.0, "danger_score": 0.0})
        d["shots"] += 1
        d["goals"] += int(num(r["is_goal"]))
        d["xg"] += num(r["xg"])
        d["psxg"] += num(r["psxg"])
        d["danger_score"] += num(r["danger_score"])
    for pid, d in danger_agg.items():
        row = get_row(pid, d["player_name"], d["team_name"])
        row.update({
            "shots": d["shots"], "goals": d["goals"],
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
            "crosses": int(num(r["crosses"])),
            "cross_completed": int(num(r["completed"])),
            "cross_expected_completed": round(num(r["expected_completed"]), 4),
            "cross_completion_rate": round(num(r["completion_rate"]), 4),
            "cross_expected_completion_rate": round(num(r["expected_completion_rate"]), 4),
            "cross_xP_added_value": round(num(r["xP_added_value"]), 4),
        })

    player_rows = list(players.values())

    # Reliability flags + per-90 rates derived from GDA minutes (the one minutes source)
    per90_bases = ["xT_added", "box_entry_added_value", "psv_total", "cross_xP_added_value",
                   "danger_score", "xg", "goals", "shots"]
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
               "danger_score_per90"]
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
        "actions", "passes", "carries", "take_ons",
        "xT_added", "xT_added_per90", "positive_xT_added", "xT_per_100_actions", "xT_added_per90_z",
        "box_entry_attempts", "box_entry_expected", "box_entry_actual", "box_entry_added_value",
        "box_entry_added_value_per90", "box_entry_added_value_per90_z",
        "psv_attempts", "psv_total", "psv_total_per90", "psv_actual_shot_xg",
        "hotzone_attempts", "hotzone_completed", "hotzone_expected", "hotzone_completion_rate",
        "hotzone_added_value",
        "crosses", "cross_completed", "cross_expected_completed", "cross_completion_rate",
        "cross_expected_completion_rate", "cross_xP_added_value", "cross_xP_added_value_per90",
        "disruption_actions_linked", "disruption_total", "disruption_per90",
        "disruption_value_total", "disruption_value_per90", "disruption_value_per90_z",
        "shots", "goals", "xg", "psxg", "danger_score",
        "shots_per90", "goals_per90", "xg_per90", "danger_score_per90", "danger_score_per90_z",
        "goal_difference_added", "goal_difference_added_per90", "action_gda_actual",
    ]

    player_rows.sort(key=lambda r: r.get("minutes", 0), reverse=True)
    with open(os.path.join(OUT_DIR, "player_season_aggregated.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=player_cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(player_rows)

    # ============================ TEAM TABLE =============================
    teams = build_league_table(events_dir)

    for r in xt_team_rows:
        t = teams.setdefault(r["team_name"], {"team_name": r["team_name"]})
        t.update({
            "actions": int(num(r["actions"])), "passes": int(num(r["passes"])),
            "carries": int(num(r["carries"])), "take_ons": int(num(r["take_ons"])),
            "xT_added": round(num(r["xT_added"]), 4),
            "positive_xT_added": round(num(r["positive_xT_added"]), 4),
            "xT_per_action": round(num(r["xT_per_action"]), 5),
        })

    danger_team = {}
    for r in load("Danger", "all_eredivisie_danger_models.csv"):
        team = team_name_by_cid.get(r["contestant_id"], r["contestant_id"])
        d = danger_team.setdefault(team, {"shots": 0, "goals": 0, "xg": 0.0, "danger_score": 0.0})
        d["shots"] += 1
        d["goals"] += int(num(r["is_goal"]))
        d["xg"] += num(r["xg"])
        d["danger_score"] += num(r["danger_score"])
    for team, d in danger_team.items():
        t = teams.setdefault(team, {"team_name": team})
        t.update({"shots": d["shots"], "goals_from_shots": d["goals"],
                  "xg": round(d["xg"], 4), "danger_score": round(d["danger_score"], 4)})

    for r in load("Disruption", "CSV", "disruption_team_summary.csv"):
        t = teams.setdefault(r["team_name"], {"team_name": r["team_name"]})
        t.update({
            "disruption_actions_linked": int(num(r["actions_linked"])),
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
            "crosses": int(num(r["crosses"])),
            "cross_completed": int(num(r["completed"])),
            "cross_expected_completed": round(num(r["expected_completed"]), 4),
            "cross_completion_rate": round(num(r["completion_rate"]), 4),
            "cross_expected_completion_rate": round(num(r["expected_completion_rate"]), 4),
            "cross_xP_added_value": round(num(r["xP_added_value"]), 4),
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
        "actions", "passes", "carries", "take_ons", "xT_added", "positive_xT_added", "xT_per_action",
        "shots", "goals_from_shots", "xg", "danger_score",
        "disruption_actions_linked", "disruption_total", "disruption_per_match",
        "disruption_value_total", "disruption_value_per_match",
        "box_entry_attempts", "box_entry_expected", "box_entry_actual", "box_entry_added_value",
        "crosses", "cross_completed", "cross_expected_completed", "cross_completion_rate",
        "cross_expected_completion_rate", "cross_xP_added_value",
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

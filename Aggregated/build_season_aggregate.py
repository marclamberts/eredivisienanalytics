"""
Build a Wyscout-style player-season and team-season stat sheet for a given
Eredivisie season straight from the raw Opta event stream in
Events/<season>/*.json, plus the "new metrics" this repo already has models
for (xT, GDA, disruption value, expected box entries, pass-shot value,
hot-zone passing, expected-completion crossing value) joined on top.

Outputs, per season:
  Aggregated/<season>/player_season_aggregated.csv   (flat, every column)
  Aggregated/<season>/team_season_aggregated.csv
  Aggregated/<season>/eredivisie_<season>_aggregated.xlsx  (same numbers,
    split across category tabs -- see build_workbook.py)

Event-code reference used below (typeId / qualifierId), verified against
this repo's Events/2025-2026 data itself (frequency sanity checks against
known per-match rates, paired-event checks for fouls/offsides/cards,
touchline/length checks for qualifiers) rather than trusted from the two
existing in-repo scripts, which disagree with each other on qualifiers
1, 15 and 107 -- see Aggregated/<season>/README.md for how each was checked.
  typeId   1 Pass            2 Offside Pass      3 Take On        4 Foul
           6 Corner Awarded   7 Tackle            8 Interception   10 Save
           11 Claim           12 Clearance        13 Miss          14 Post
           15 Attempt Saved   16 Goal             17 Card          41 Punch
           44 Aerial Duel     49 Ball Recovery    50 Dispossessed  51 Error
           52 Keeper Pick-up  54 Smother          55 Offside Provoked
           58 Penalty Faced   59 Keeper Sweeper   61 Ball Touch
  qualifierId  1 Long ball  2 Cross  3 Through ball  5 Free kick  6 Corner
               15 Head  20 Right foot  31 Yellow card  32 Second yellow
               33 Red card  72 Left foot  107 Throw-in  140/141 pass end x/y
               195 Pull back  212 length (m)  213 angle (rad)

No composite score is produced -- see Aggregated/<season>/README.md for why.

Usage: python3 build_season_aggregate.py [season]   (default: 2025-2026)
"""
import csv
import glob
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from column_layout import categorize, PLAYER_TAB_RULES, TEAM_TAB_RULES  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEASON = sys.argv[1] if len(sys.argv) > 1 else "2025-2026"
OUT_DIR = os.path.join(ROOT, "Aggregated", SEASON)
MIN_MINUTES_RELIABLE = 450  # ~5 full matches; same cutoff netlify-app/generate_data.py uses for GDA

X_SCALE, Y_SCALE = 1.05, 0.68           # Opta 0-100 units -> metres (105x68 pitch), matches Disruption model
GOAL_X, GOAL_Y = 105.0, 34.0
PROG_OWN_HALF_M = 27.432     # 30 yards: both ends in own half
PROG_TO_ATT_HALF_M = 13.716  # 15 yards: crosses into attacking half
PROG_ATT_HALF_M = 9.144      # 10 yards: both ends in attacking half
DEF_THIRD_MAX, MID_THIRD_MAX = 100.0 / 3.0, 200.0 / 3.0
BOX_X_MIN, BOX_Y_MIN, BOX_Y_MAX = 83.0, 21.1, 78.9   # same box definition as Box Entry Models
SHORT_MAX_M, MEDIUM_MAX_M = 15.0, 30.0
CARRY_MIN_M, CARRY_MAX_GAP_S = 3.0, 8.0

Q_LONG_BALL, Q_CROSS, Q_THROUGH_BALL, Q_FREE_KICK, Q_CORNER = 1, 2, 3, 5, 6
Q_HEAD, Q_RIGHT_FOOT, Q_LEFT_FOOT, Q_THROW_IN = 15, 20, 72, 107
Q_END_X, Q_END_Y = 140, 141
Q_YELLOW, Q_SECOND_YELLOW, Q_RED = 31, 32, 33
Q_PULL_BACK = 195

BALL_ACTION_TYPES = {1, 3, 13, 14, 15, 16, 61}   # pass/take-on/shot/ball-touch: carry chain can run through these
DEAD_BALL_QUALIFIERS = {Q_FREE_KICK, Q_CORNER, Q_THROW_IN}

RAW_COUNT_FIELDS = [
    # passing
    "passes_attempted", "passes_completed", "forward_passes", "backward_passes", "lateral_passes",
    "progressive_passes", "passes_into_final_third",
    "passes_def_third", "passes_completed_def_third", "passes_mid_third", "passes_completed_mid_third",
    "passes_att_third", "passes_completed_att_third",
    "short_passes", "short_passes_completed", "medium_passes", "medium_passes_completed",
    "long_passes_len", "long_passes_len_completed",
    "long_balls_attempted", "long_balls_completed",
    "crosses_attempted", "crosses_completed", "crosses_left_side", "crosses_left_completed",
    "crosses_right_side", "crosses_right_completed",
    "through_balls_attempted", "through_balls_completed",
    "passes_left_foot", "passes_left_foot_completed", "passes_right_foot", "passes_right_foot_completed",
    "passes_head", "passes_head_completed",
    "passes_received", "progressive_passes_received",
    # creativity
    "key_passes", "assists", "xa", "shot_creating_actions", "goal_creating_actions",
    "key_passes_cutback", "assists_cutback", "xa_cutback",
    "key_passes_cross", "assists_cross", "xa_cross",
    "key_passes_through_ball", "assists_through_ball", "xa_through_ball",
    "key_passes_set_piece", "assists_set_piece", "xa_set_piece",
    "key_passes_open_play", "assists_open_play", "xa_open_play",
    # carries
    "carries_computed", "carry_distance_m", "progressive_carries", "carries_into_final_third",
    "carries_into_box",
    # duels / dribbling
    "take_ons_attempted", "take_ons_successful",
    "tackles_attempted", "tackles_won", "tackles_def_third", "tackles_mid_third", "tackles_att_third",
    "interceptions", "interceptions_def_third", "interceptions_mid_third", "interceptions_att_third",
    "clearances", "aerial_duels", "aerial_duels_won", "aerial_duels_def_half", "aerial_duels_won_def_half",
    "aerial_duels_att_half", "aerial_duels_won_att_half",
    "ball_recoveries", "ball_recoveries_def_third", "ball_recoveries_mid_third", "ball_recoveries_att_third",
    "dispossessed", "errors",
    "touches", "touches_in_box",
    # discipline
    "fouls_committed", "fouls_won", "offsides", "yellow_cards", "red_cards_straight", "red_cards_second_yellow",
    # shooting
    "shots", "goals",
    "shots_left_foot", "goals_left_foot", "shots_right_foot", "goals_right_foot",
    "shots_head", "goals_head", "shots_inside_box", "goals_inside_box", "shots_outside_box",
    "goals_outside_box", "shot_dist_total_m",
    # set pieces
    "corners_taken", "corners_completed", "free_kicks_taken", "free_kick_shots", "free_kick_goals",
    "throw_ins_taken",
    # goalkeeping
    "gk_saves", "gk_claims", "gk_punches", "gk_pickups", "gk_smothers", "gk_penalties_faced",
    "gk_sweeper_actions",
    # half splits (core subset)
    "progressive_passes_1h", "progressive_passes_2h", "key_passes_1h", "key_passes_2h",
    "shots_1h", "shots_2h", "tackles_attempted_1h", "tackles_attempted_2h",
    "interceptions_1h", "interceptions_2h",
    # home/away splits (core subset)
    "goals_home", "goals_away", "shots_home", "shots_away",
    "key_passes_home", "key_passes_away", "tackles_attempted_home", "tackles_attempted_away",
]


MISSING_FILES = []  # populated by load() when a per-metric file isn't available for this season

# Every file load() reads (xT/, GDA/, Danger/, Disruption/CSV/, Box Entry
# Models/, Cross Models/, Analysis/Coach Profiling/, Analysis/Formation/) is a
# single unified path with NO season in it anywhere -- it was computed once,
# against whichever season's Events/ existed at the time (2025-2026). For any
# other season these files still *exist* on disk (so a plain try/except
# FileNotFoundError would silently load them anyway) but hold the wrong
# season's player/team ids entirely -- that's real cross-season
# contamination, not a graceful degradation. So: only read them at all when
# this run's SEASON is the one they were actually built from.
NEW_METRICS_SEASON = "2025-2026"


def load(*parts):
    """Load a CSV one of this repo's other pipelines produced -- all of them
    "new metric" sources that only exist for NEW_METRICS_SEASON. Returns []
    (and notes it in MISSING_FILES) for every other season, and also if the
    file is simply missing on disk."""
    if SEASON != NEW_METRICS_SEASON:
        rel = os.path.join(*parts)
        if rel not in MISSING_FILES:
            MISSING_FILES.append(rel)
        return []
    path = os.path.join(ROOT, *parts)
    try:
        with open(path, encoding="utf-8-sig", newline="") as f:
            return list(csv.DictReader(f))
    except FileNotFoundError:
        rel = os.path.join(*parts)
        MISSING_FILES.append(rel)
        return []


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
    xs = list(values.values())
    if len(xs) < 2:
        return {k: None for k in values}
    mean = statistics.fmean(xs)
    sd = statistics.pstdev(xs)
    if sd == 0:
        return {k: 0.0 for k in values}
    return {k: (v - mean) / sd for k, v in values.items()}


def to_m(x, y):
    return x * X_SCALE, y * Y_SCALE


def dist_m(x1, y1, x2, y2):
    x1m, y1m = to_m(x1, y1)
    x2m, y2m = to_m(x2, y2)
    return ((x1m - x2m) ** 2 + (y1m - y2m) ** 2) ** 0.5


def dist_to_goal_m(x, y):
    xm, ym = to_m(x, y)
    return ((GOAL_X - xm) ** 2 + (GOAL_Y - ym) ** 2) ** 0.5


def is_progressive(sx, sy, ex, ey):
    gain = dist_to_goal_m(sx, sy) - dist_to_goal_m(ex, ey)
    if sx < 50 and ex < 50:
        return gain >= PROG_OWN_HALF_M
    if sx < 50 <= ex:
        return gain >= PROG_TO_ATT_HALF_M
    return gain >= PROG_ATT_HALF_M


def zone_of(x):
    if x < DEF_THIRD_MAX:
        return "def"
    if x < MID_THIRD_MAX:
        return "mid"
    return "att"


def in_box(x, y):
    return x >= BOX_X_MIN and BOX_Y_MIN <= y <= BOX_Y_MAX


def qmap(e):
    return {q["qualifierId"]: q.get("value") for q in e.get("qualifier", []) or []}


def event_time(e):
    return e["timeMin"] * 60 + e["timeSec"]


def derive_team_name_by_cid(events_dir):
    """contestantId -> team name, built directly from Events/<season> (no
    dependency on xT/xt_team_summary.csv, which only exists for 2025-2026).

    contestantId is an opaque hash with no name attached anywhere in the
    feed; the only names are in each match filename ("Home - Away.json").
    Resolve which contestantId is home vs. away per match by comparing goal
    counts (typeId 16) against matchDetails' actual final score -- skipping
    draws and any match where goals don't reconcile -- then take each
    contestantId's majority-vote name across the whole season. Validated
    against 2025-2026's known xT mapping: 18/18 correct from ~1 dozen
    decisive matches per team; a full season gives far more than that.
    """
    votes = {}
    for path in sorted(glob.glob(os.path.join(events_dir, "*.json"))):
        raw = json.load(open(path, encoding="utf-8"))
        stem = os.path.basename(path)[:-5]
        _, fixture = stem.split("_", 1)
        home, away = fixture.split(" - ", 1)
        scores = raw.get("matchDetails", {}).get("scores", {})
        ft = scores.get("total", scores.get("ft", {}))
        h, a = ft.get("home"), ft.get("away")
        if h is None or a is None or h == a:
            continue
        h, a = int(h), int(a)
        events = raw.get("event", [])
        all_cids = {e.get("contestantId") for e in events if e.get("contestantId")}
        if len(all_cids) != 2:
            continue
        c1, c2 = tuple(all_cids)
        goals = {}
        for e in events:
            if e.get("typeId") == 16:
                goals[e["contestantId"]] = goals.get(e["contestantId"], 0) + 1
        g1, g2 = goals.get(c1, 0), goals.get(c2, 0)
        if g1 == h and g2 == a:
            pair = ((c1, home), (c2, away))
        elif g1 == a and g2 == h:
            pair = ((c1, away), (c2, home))
        else:
            continue  # goals don't reconcile with the scoreline (e.g. own goals); skip
        for cid, name in pair:
            votes.setdefault(cid, {}).setdefault(name, 0)
            votes[cid][name] += 1
    return {cid: max(names.items(), key=lambda kv: kv[1])[0] for cid, names in votes.items()}


def compute_match_minutes(events, match_length_min):
    """Per-player minutes played in one match, from substitution (typeId 18
    off / 19 on) and red-card (typeId 17, qualifier 32/33) events -- the
    same stint concept GDA/gda_model_meta.json describes, computed directly
    here so minutes don't depend on GDA's own (2025-2026-only) output."""
    subbed_on, subbed_off, sent_off, seen = {}, {}, {}, set()
    for e in events:
        pid = e.get("playerId")
        if not pid:
            continue
        t = e["typeId"]
        tmin = event_time(e) / 60.0
        if t in (1, 3, 4, 7, 8, 12, 13, 14, 15, 16, 17, 18, 19, 44, 49, 50, 61):
            seen.add(pid)
        if t == 19:
            subbed_on[pid] = tmin
        elif t == 18:
            subbed_off[pid] = tmin
        elif t == 17:
            q = qmap(e)
            if Q_SECOND_YELLOW in q or Q_RED in q:
                sent_off[pid] = tmin
    minutes = {}
    for pid in seen:
        start = subbed_on.get(pid, 0.0)
        end = subbed_off.get(pid, match_length_min)
        if pid in sent_off:
            end = min(end, sent_off[pid])
        if end > start:
            minutes[pid] = end - start
    return minutes


def new_counter():
    return {f: 0 for f in RAW_COUNT_FIELDS}


def process_raw_events(events_dir, team_name_by_cid, danger_lookup):
    """One pass over every match file: league table, per-match possession
    share, self-computed minutes/matches, and every Wyscout-style player
    count listed in RAW_COUNT_FIELDS."""
    league = {}
    player_counts = {}
    team_match_passes = {}  # team_name -> list of (own_pass_attempts, opp_pass_attempts) per match

    def acc_for(pid, pname, team):
        row = player_counts.setdefault(pid, {"player_name": pname, "team_name": team,
                                              "minutes": 0.0, "matches": 0, **new_counter()})
        if pname and not row["player_name"]:
            row["player_name"] = pname
        return row

    for path in sorted(glob.glob(os.path.join(events_dir, "*.json"))):
        raw = json.load(open(path, encoding="utf-8"))
        fn = os.path.basename(path)
        stem = fn[:-5]
        _, fixture = stem.split("_", 1)
        home, away = fixture.split(" - ", 1)
        match_length_min = raw.get("matchDetails", {}).get("matchLengthMin", 90) + \
            raw.get("matchDetails", {}).get("matchLengthSec", 0) / 60.0
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

        match_minutes = compute_match_minutes(events, match_length_min)
        player_info_this_match = {}
        for e in events:
            pid = e.get("playerId")
            if pid and pid not in player_info_this_match:
                team = team_name_by_cid.get(e.get("contestantId"), e.get("contestantId"))
                player_info_this_match[pid] = (e.get("playerName", ""), team)
        for pid, mins in match_minutes.items():
            pname, team = player_info_this_match.get(pid, ("", ""))
            row = acc_for(pid, pname, team)
            row["minutes"] += mins
            row["matches"] += 1

        # per-match pass counts by contestantId, for a season-average possession proxy
        match_pass_counts = {}
        for e in events:
            if e["typeId"] == 1:
                cid = e.get("contestantId")
                match_pass_counts[cid] = match_pass_counts.get(cid, 0) + 1
        cids = list(match_pass_counts.keys())
        if len(cids) == 2:
            for cid in cids:
                other = cids[1] if cid == cids[0] else cids[0]
                team = team_name_by_cid.get(cid, cid)
                team_match_passes.setdefault(team, []).append(
                    (match_pass_counts[cid], match_pass_counts[other]))

        last_action = {}  # pid -> (x, y, time_s) end point of their last ball-action, for carry detection

        for idx, e in enumerate(events):
            pid = e.get("playerId")
            if not pid:
                continue
            cid = e.get("contestantId")
            team = team_name_by_cid.get(cid, cid)
            row = acc_for(pid, e.get("playerName", ""), team)
            t = e["typeId"]
            outcome = e.get("outcome")
            q = qmap(e)
            x, y = num(e.get("x")), num(e.get("y"))
            is_home = (team == home)
            half = "1h" if e.get("periodId") == 1 else ("2h" if e.get("periodId") == 2 else None)

            if t == 1:  # Pass
                row["passes_attempted"] += 1
                completed = outcome == 1
                zone = zone_of(x)
                row[f"passes_{zone}_third"] += 1
                if completed:
                    row["passes_completed"] += 1
                    row[f"passes_completed_{zone}_third"] += 1
                if Q_LONG_BALL in q:
                    row["long_balls_attempted"] += 1
                    if completed:
                        row["long_balls_completed"] += 1
                if Q_CROSS in q:
                    row["crosses_attempted"] += 1
                    side = "left" if y >= 50 else "right"
                    row[f"crosses_{side}_side"] += 1
                    if completed:
                        row["crosses_completed"] += 1
                        row[f"crosses_{side}_completed"] += 1
                if Q_THROUGH_BALL in q:
                    row["through_balls_attempted"] += 1
                    if completed:
                        row["through_balls_completed"] += 1
                if Q_RIGHT_FOOT in q:
                    row["passes_right_foot"] += 1
                    if completed:
                        row["passes_right_foot_completed"] += 1
                if Q_LEFT_FOOT in q:
                    row["passes_left_foot"] += 1
                    if completed:
                        row["passes_left_foot_completed"] += 1
                if Q_HEAD in q:
                    row["passes_head"] += 1
                    if completed:
                        row["passes_head_completed"] += 1
                length_m = num(q.get(212)) if 212 in q else None
                if length_m is not None:
                    bucket = "short_passes" if length_m < SHORT_MAX_M else (
                        "medium_passes" if length_m < MEDIUM_MAX_M else "long_passes_len")
                    row[bucket] += 1
                    if completed:
                        row[f"{bucket}_completed"] += 1
                if Q_CORNER in q:
                    row["corners_taken"] += 1
                    if completed:
                        row["corners_completed"] += 1
                if Q_FREE_KICK in q:
                    row["free_kicks_taken"] += 1
                if Q_THROW_IN in q:
                    row["throw_ins_taken"] += 1

                if completed and Q_END_X in q:
                    ex, ey = num(q[Q_END_X]), num(q.get(Q_END_Y))
                    if ex > x + 1e-9:
                        row["forward_passes"] += 1
                    elif ex < x - 1e-9:
                        row["backward_passes"] += 1
                    else:
                        row["lateral_passes"] += 1
                    if x < MID_THIRD_MAX <= ex:
                        row["passes_into_final_third"] += 1
                    if is_progressive(x, y, ex, ey):
                        row["progressive_passes"] += 1
                        if half:
                            row[f"progressive_passes_{half}"] += 1
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
                zone = zone_of(x)
                row[f"tackles_{zone}_third"] += 1
                if outcome == 1:
                    row["tackles_won"] += 1
                if half:
                    row[f"tackles_attempted_{half}"] += 1
                row[f"tackles_attempted_{'home' if is_home else 'away'}"] += 1
            elif t == 8:
                row["interceptions"] += 1
                row[f"interceptions_{zone_of(x)}_third"] += 1
                if half:
                    row[f"interceptions_{half}"] += 1
            elif t == 12:
                row["clearances"] += 1
            elif t == 17:  # Card
                if Q_RED in q:
                    row["red_cards_straight"] += 1
                elif Q_SECOND_YELLOW in q:
                    row["red_cards_second_yellow"] += 1
                elif Q_YELLOW in q:
                    row["yellow_cards"] += 1
            elif t == 44:  # Aerial duel
                row["aerial_duels"] += 1
                half_pitch = "def_half" if x < 50 else "att_half"
                row[f"aerial_duels_{half_pitch}"] += 1
                if outcome == 1:
                    row["aerial_duels_won"] += 1
                    row[f"aerial_duels_won_{half_pitch}"] += 1
            elif t == 49:
                row["ball_recoveries"] += 1
                row[f"ball_recoveries_{zone_of(x)}_third"] += 1
            elif t == 50:
                row["dispossessed"] += 1
            elif t == 51:
                row["errors"] += 1
            elif t == 55:
                row["offsides"] += 1
            elif t in (13, 14, 15, 16):  # shots
                row["shots"] += 1
                if t == 16:
                    row["goals"] += 1
                if Q_HEAD in q:
                    row["shots_head"] += 1
                elif Q_LEFT_FOOT in q:
                    row["shots_left_foot"] += 1
                else:
                    row["shots_right_foot"] += 1
                if in_box(x, y):
                    row["shots_inside_box"] += 1
                else:
                    row["shots_outside_box"] += 1
                row["shot_dist_total_m"] += dist_to_goal_m(x, y)
                if Q_FREE_KICK in q:
                    row["free_kick_shots"] += 1
                if t == 16:
                    if Q_HEAD in q:
                        row["goals_head"] += 1
                    elif Q_LEFT_FOOT in q:
                        row["goals_left_foot"] += 1
                    else:
                        row["goals_right_foot"] += 1
                    if in_box(x, y):
                        row["goals_inside_box"] += 1
                    else:
                        row["goals_outside_box"] += 1
                    if Q_FREE_KICK in q:
                        row["free_kick_goals"] += 1
                row[f"shots_{'home' if is_home else 'away'}"] += 1
                if half:
                    row[f"shots_{half}"] += 1
                if t == 16:
                    row[f"goals_{'home' if is_home else 'away'}"] += 1

                # key pass / assist / xA / SCA / GCA: anchored on this shot
                # (detected directly from typeId, not from the Danger CSV --
                # only the xG value itself, when available, comes from there)
                is_goal = t == 16
                shot_xg = danger_lookup.get((fn, str(e.get("id"))), 0.0)
                for e2 in reversed(events[max(0, idx - 4):idx]):
                    if e2.get("contestantId") == cid and e2.get("typeId") == 1 and \
                            e2.get("outcome") == 1 and e2.get("playerId"):
                        row2 = acc_for(e2["playerId"], e2.get("playerName", ""), team)
                        row2["key_passes"] += 1
                        row2["xa"] += shot_xg
                        half2 = "1h" if e2.get("periodId") == 1 else ("2h" if e2.get("periodId") == 2 else None)
                        if half2:
                            row2[f"key_passes_{half2}"] += 1
                        row2[f"key_passes_{'home' if is_home else 'away'}"] += 1
                        if is_goal:
                            row2["assists"] += 1

                        eq = qmap(e2)
                        if Q_PULL_BACK in eq:
                            dtype = "cutback"
                        elif Q_CROSS in eq:
                            dtype = "cross"
                        elif Q_THROUGH_BALL in eq:
                            dtype = "through_ball"
                        elif Q_FREE_KICK in eq or Q_CORNER in eq:
                            dtype = "set_piece"
                        else:
                            dtype = "open_play"
                        row2[f"key_passes_{dtype}"] += 1
                        row2[f"xa_{dtype}"] += shot_xg
                        if is_goal:
                            row2[f"assists_{dtype}"] += 1
                        break

                credited = set()
                for e2 in reversed(events[max(0, idx - 6):idx]):
                    if len(credited) >= 2:
                        break
                    if e2.get("contestantId") != cid or not e2.get("playerId"):
                        continue
                    counts_as_action = (
                        (e2.get("typeId") == 1 and e2.get("outcome") == 1) or
                        (e2.get("typeId") == 3 and e2.get("outcome") == 1) or
                        (e2.get("typeId") == 4 and e2.get("outcome") == 1)  # foul won
                    )
                    if not counts_as_action or e2["playerId"] in credited:
                        continue
                    credited.add(e2["playerId"])
                    row2 = acc_for(e2["playerId"], e2.get("playerName", ""), team)
                    row2["shot_creating_actions"] += 1
                    if is_goal:
                        row2["goal_creating_actions"] += 1
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

            # touches (approximate: any on-ball involvement)
            if t in (1, 3, 12, 13, 14, 15, 16, 61):
                row["touches"] += 1
                if in_box(x, y):
                    row["touches_in_box"] += 1

            # --- carry detection: chain through pass/take-on/shot/ball-touch events ---
            if t in BALL_ACTION_TYPES:
                prev = last_action.get(pid)
                now_t = event_time(e)
                is_restart = bool(q.keys() & DEAD_BALL_QUALIFIERS) if q else False
                if prev and not is_restart:
                    px, py, pt = prev
                    if 0 < now_t - pt <= CARRY_MAX_GAP_S:
                        gained_m = dist_m(px, py, x, y)
                        if gained_m >= CARRY_MIN_M:
                            row["carries_computed"] += 1
                            row["carry_distance_m"] += gained_m
                            if is_progressive(px, py, x, y):
                                row["progressive_carries"] += 1
                            if px < MID_THIRD_MAX <= x:
                                row["carries_into_final_third"] += 1
                            if not in_box(px, py) and in_box(x, y):
                                row["carries_into_box"] += 1
                # update this player's last action end-point
                if t == 1 and outcome == 1 and Q_END_X in q:
                    last_action[pid] = (num(q[Q_END_X]), num(q.get(Q_END_Y)), now_t)
                elif t == 1 and outcome != 1:
                    last_action.pop(pid, None)  # failed pass: chain broken
                else:
                    last_action[pid] = (x, y, now_t)

        # passes received: the next on-ball event by a teammate within 5s of a completed pass
        for i, e in enumerate(events):
            if e.get("typeId") != 1 or e.get("outcome") != 1 or not e.get("playerId"):
                continue
            cid = e.get("contestantId")
            q = qmap(e)
            if Q_END_X not in q:
                continue
            ex, ey = num(q[Q_END_X]), num(q.get(Q_END_Y))
            sx, sy = num(e.get("x")), num(e.get("y"))
            t0 = event_time(e)
            for nxt in events[i + 1:i + 4]:
                if nxt.get("typeId") not in BALL_ACTION_TYPES or not nxt.get("playerId"):
                    continue
                if event_time(nxt) - t0 > 5:
                    break
                if nxt.get("contestantId") == cid and nxt.get("playerId") != e.get("playerId"):
                    recv = acc_for(nxt["playerId"], nxt.get("playerName", ""),
                                    team_name_by_cid.get(cid, cid))
                    recv["passes_received"] += 1
                    if is_progressive(sx, sy, ex, ey):
                        recv["progressive_passes_received"] += 1
                break

    for c in league.values():
        c["goal_diff"] = c["goals_for"] - c["goals_against"]
        c["points"] = c["wins"] * 3 + c["draws"]

    team_possession = {}
    for team, matches in team_match_passes.items():
        shares = [own / (own + opp) for own, opp in matches if (own + opp) > 0]
        team_possession[team] = round(statistics.fmean(shares) * 100, 2) if shares else None

    return league, player_counts, team_possession


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    events_dir = os.path.join(ROOT, "Events", SEASON)

    print(f"Resolving team identities for {SEASON} from Events (goal-count vs. scoreline vote)...")
    team_name_by_cid = derive_team_name_by_cid(events_dir)

    xt_team_rows = load("xT", "xt_team_summary.csv")     # 2025-2026 only; [] otherwise
    danger_rows = load("Danger", "all_eredivisie_danger_models.csv")  # 2025-2026 only; [] otherwise
    danger_lookup = {(r["match_file"], r["event_id"]): num(r["xg"]) for r in danger_rows}

    print("Parsing raw events for the full Wyscout-style stat sheet (this takes a minute)...")
    league_table, raw_counts, team_possession = process_raw_events(events_dir, team_name_by_cid, danger_lookup)

    # possession-adjusted defensive stats: teams that see less of the ball make
    # more raw defensive actions just by virtue of facing more opposition
    # possession, so scale each player's rate by their team's opponent-possession
    # share relative to the league average opponent-possession share.
    opp_poss = {team: 100 - p for team, p in team_possession.items() if p is not None}
    avg_opp_poss = statistics.fmean(opp_poss.values()) if opp_poss else 50.0

    # =========================== PLAYER TABLE ===========================
    players = {}

    def get_row(player_id, player_name, team_name):
        key = player_id if player_id else ("__noid__", player_name, team_name)
        return players.setdefault(key, {
            "player_id": player_id or "", "player_name": player_name, "team_name": team_name,
        })

    for pid, r in raw_counts.items():
        row = get_row(pid, r["player_name"], r["team_name"])
        row.update({f: r[f] for f in RAW_COUNT_FIELDS})
        row["minutes"] = round(r["minutes"], 1)
        row["matches"] = r["matches"]

    for r in load("GDA", "gda_player_summary.csv"):
        team = team_name_by_cid.get(r["contestant_id"], r["contestant_id"])
        row = get_row(r["player_id"], r["player_name"], team)
        row.update({
            "matches": int(num(r["matches"])),
            "minutes": round(num(r["minutes"]), 1),
            "goals_against_on": round(num(r["goals_against_on"]), 2),
            "goal_difference_added": round(num(r["goal_difference_added"]), 4),
            "goal_difference_added_per90": round(num(r["goal_difference_added_per90"]), 4),
            "action_gda_actual": round(num(r["action_gda_actual"]), 4),
        })

    for r in load("xT", "xt_player_summary.csv"):
        row = get_row(r["player_id"], r["player_name"], r["team_name"])
        row.update({
            "actions": int(num(r["actions"])),
            "xt_passes": int(num(r["passes"])),
            "xt_carries": int(num(r["carries"])),
            "xt_take_ons": int(num(r["take_ons"])),
            "xT_added": round(num(r["xT_added"]), 4),
            "positive_xT_added": round(num(r["positive_xT_added"]), 4),
            "xT_per_100_actions": round(num(r["xT_per_100_actions"]), 4),
        })

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

    danger_agg = {}
    for r in danger_rows:
        pid = r["player_id"]
        team = team_name_by_cid.get(r["contestant_id"], r["contestant_id"])
        d = danger_agg.setdefault(pid, {"player_name": r["player_name"], "team_name": team,
                                          "shots": 0, "shots_on_target": 0, "goals": 0, "np_goals": 0,
                                          "xg": 0.0, "np_xg": 0.0, "psxg": 0.0, "danger_score": 0.0})
        is_pen = num(r["is_penalty"]) == 1
        d["shots"] += 1
        d["shots_on_target"] += int(num(r["is_on_target"]))
        d["goals"] += int(num(r["is_goal"]))
        d["xg"] += num(r["xg"])
        d["psxg"] += num(r["psxg"])
        d["danger_score"] += num(r["danger_score"])
        if not is_pen:
            d["np_goals"] += int(num(r["is_goal"]))
            d["np_xg"] += num(r["xg"])
    for pid, d in danger_agg.items():
        row = get_row(pid, d["player_name"], d["team_name"])
        row.update({
            "shots": d["shots"], "shots_on_target": d["shots_on_target"], "goals": d["goals"],
            "np_goals": d["np_goals"], "np_xg": round(d["np_xg"], 4),
            "xg": round(d["xg"], 4), "psxg": round(d["psxg"], 4),
            "danger_score": round(d["danger_score"], 4),
        })

    penalties = {}
    for r in danger_rows:
        if num(r["is_penalty"]) == 1:
            pid = r["player_id"]
            p = penalties.setdefault(pid, {"player_name": r["player_name"],
                                            "team_name": team_name_by_cid.get(r["contestant_id"], r["contestant_id"]),
                                            "taken": 0, "scored": 0})
            p["taken"] += 1
            p["scored"] += int(num(r["is_goal"]))
    for pid, p in penalties.items():
        row = get_row(pid, p["player_name"], p["team_name"])
        row.update({"penalties_taken": p["taken"], "penalties_scored": p["scored"]})

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

    # clean sheets: matches with 0 goals conceded while on pitch (needs the
    # per-match on-pitch file GDA already produced)
    try:
        onpitch = load("GDA", "gda_player_match_on_pitch.csv")
        cs = {}
        for r in onpitch:
            pid = r["player_id"]
            conceded = num(r.get("goals_against_on", r.get("goals_against", 0)))
            d = cs.setdefault(pid, {"matches": 0, "clean": 0})
            d["matches"] += 1
            if conceded == 0:
                d["clean"] += 1
        for pid, d in cs.items():
            if pid in players:
                players[pid]["clean_sheets"] = d["clean"]
    except (FileNotFoundError, KeyError):
        pass  # column names in that file didn't match what we expected; skip rather than guess

    # completion / win percentages (ratio of two counts, not a per-90 rate)
    pct_pairs = [
        ("pass_completion_pct", "passes_completed", "passes_attempted"),
        ("long_ball_completion_pct", "long_balls_completed", "long_balls_attempted"),
        ("cross_completion_pct", "crosses_completed", "crosses_attempted"),
        ("through_ball_completion_pct", "through_balls_completed", "through_balls_attempted"),
        ("take_on_success_pct", "take_ons_successful", "take_ons_attempted"),
        ("tackle_win_pct", "tackles_won", "tackles_attempted"),
        ("aerial_duel_win_pct", "aerial_duels_won", "aerial_duels"),
        ("shot_on_target_pct", "shots_on_target", "shots"),
        ("passes_def_third_completion_pct", "passes_completed_def_third", "passes_def_third"),
        ("passes_mid_third_completion_pct", "passes_completed_mid_third", "passes_mid_third"),
        ("passes_att_third_completion_pct", "passes_completed_att_third", "passes_att_third"),
        ("short_pass_completion_pct", "short_passes_completed", "short_passes"),
        ("medium_pass_completion_pct", "medium_passes_completed", "medium_passes"),
        ("long_pass_len_completion_pct", "long_passes_len_completed", "long_passes_len"),
        ("left_foot_pass_completion_pct", "passes_left_foot_completed", "passes_left_foot"),
        ("right_foot_pass_completion_pct", "passes_right_foot_completed", "passes_right_foot"),
        ("head_pass_completion_pct", "passes_head_completed", "passes_head"),
        ("penalty_conversion_pct", "penalties_scored", "penalties_taken"),
        ("free_kick_shot_conversion_pct", "free_kick_goals", "free_kick_shots"),
        ("corner_completion_pct", "corners_completed", "corners_taken"),
        ("save_pct", "gk_saves", None),  # special-cased below (needs goals_against_on)
    ]
    for row in player_rows:
        for name, numer, denom in pct_pairs:
            if name == "save_pct":
                saves, conceded = row.get("gk_saves", 0), row.get("goals_against_on", 0)
                row[name] = pct(saves, saves + conceded) if (saves + conceded) else ""
                continue
            row[name] = pct(row.get(numer, 0), row.get(denom, 0)) if numer in row else ""
        row["goal_involvement"] = (row.get("goals", 0) or 0) + (row.get("assists", 0) or 0)
        row["avg_shot_distance_m"] = round(row["shot_dist_total_m"] / row["shots"], 2) if row.get("shots") else ""
        row["total_duels"] = row.get("tackles_attempted", 0) + row.get("aerial_duels", 0) + row.get("take_ons_attempted", 0)
        row["total_duels_won"] = row.get("tackles_won", 0) + row.get("aerial_duels_won", 0) + row.get("take_ons_successful", 0)
        row["total_duel_win_pct"] = pct(row["total_duels_won"], row["total_duels"])
        row["possession_losses"] = row.get("dispossessed", 0) + row.get("errors", 0)
        row["progressive_actions"] = row.get("progressive_passes", 0) + row.get("progressive_carries", 0)
        row["final_third_entries"] = row.get("passes_into_final_third", 0) + row.get("carries_into_final_third", 0)
        row["total_box_entries"] = round(num(row.get("box_entry_actual", 0)), 4) + row.get("carries_into_box", 0)
        defensive_zone_sum = 0
        for zone in ("def", "mid", "att"):
            v = row.get(f"tackles_{zone}_third", 0) + row.get(f"interceptions_{zone}_third", 0) + \
                row.get(f"ball_recoveries_{zone}_third", 0)
            row[f"defensive_actions_{zone}_third"] = v

        # possession-adjusted (PAdj) defensive volume stats
        team_opp_poss = opp_poss.get(row["team_name"])
        padj_factor = (avg_opp_poss / team_opp_poss) if team_opp_poss else 1.0
        for base in ("tackles_attempted", "interceptions", "clearances", "aerial_duels", "ball_recoveries"):
            row[f"padj_{base}"] = round(row.get(base, 0) * padj_factor, 3)

    # per-90 rates, derived from GDA minutes (the one minutes source)
    per90_bases = RAW_COUNT_FIELDS + [
        "xT_added", "box_entry_added_value", "psv_total", "xp_cross_added_value",
        "danger_score", "xg", "np_xg", "goals", "np_goals", "shots", "shots_on_target",
        "goal_involvement", "total_duels", "total_duels_won", "possession_losses",
        "progressive_actions", "final_third_entries", "total_box_entries",
        "padj_tackles_attempted", "padj_interceptions", "padj_clearances", "padj_aerial_duels",
        "padj_ball_recoveries",
        "defensive_actions_def_third", "defensive_actions_mid_third", "defensive_actions_att_third",
    ]
    for row in player_rows:
        minutes = row.get("minutes", 0.0)
        row["minutes_bucket"] = minutes_bucket(minutes)
        row["reliable_sample"] = minutes >= MIN_MINUTES_RELIABLE
        for base in per90_bases:
            if base in row and minutes > 0:
                row[f"{base}_per90"] = round(row[base] * 90.0 / minutes, 4)

    z_bases = ["xT_added_per90", "box_entry_added_value_per90", "disruption_value_per90",
               "danger_score_per90", "progressive_passes_per90", "progressive_actions_per90",
               "padj_tackles_attempted_per90", "padj_interceptions_per90"]
    for base in z_bases:
        values = {i: row[base] for i, row in enumerate(player_rows)
                  if row.get("reliable_sample") and base in row}
        z = zscores(values)
        for i, row in enumerate(player_rows):
            if i in z:
                row[f"{base}_z"] = round(z[i], 3) if z[i] is not None else ""

    all_player_keys = []
    seen = set()
    for row in player_rows:
        for k in row:
            if k not in seen:
                all_player_keys.append(k)
                seen.add(k)
    player_tabs = categorize(all_player_keys, PLAYER_TAB_RULES)
    player_cols = ["player_id", "player_name", "team_name"]
    for _, cols in player_tabs:
        player_cols.extend(cols)

    player_rows.sort(key=lambda r: r.get("minutes", 0), reverse=True)
    with open(os.path.join(OUT_DIR, "player_season_aggregated.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=player_cols, extrasaction="ignore", restval="")
        w.writeheader()
        w.writerows(player_rows)

    # ============================ TEAM TABLE =============================
    teams = league_table

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
        t["possession_pct"] = team_possession.get(team_name, "")

    for r in xt_team_rows:
        t = teams.setdefault(r["team_name"], {"team_name": r["team_name"]})
        t.update({
            "actions": int(num(r["actions"])),
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
        t.update({"goals_from_shots": d["goals"], "xg": round(d["xg"], 4),
                  "danger_score": round(d["danger_score"], 4)})

    for r in load("Disruption", "CSV", "disruption_team_summary.csv"):
        t = teams.setdefault(r["team_name"], {"team_name": r["team_name"]})
        t.update({"disruption_total": round(num(r["total_disruption"]), 4),
                  "disruption_per_match": round(num(r["disruption_per_match"]), 4)})
    for r in load("Disruption", "CSV", "disruption_value_team_summary.csv"):
        t = teams.setdefault(r["team_name"], {"team_name": r["team_name"]})
        t.update({"disruption_value_total": round(num(r["total_disruption_value"]), 6),
                  "disruption_value_per_match": round(num(r["disruption_value_per_match"]), 6)})

    for r in load("Box Entry Models", "box_entry_team_summary.csv"):
        t = teams.setdefault(r["team"], {"team_name": r["team"]})
        t.update({"box_entry_attempts": int(num(r["attempts"])),
                  "box_entry_expected": round(num(r["expected_box_entries"]), 4),
                  "box_entry_actual": round(num(r["actual_box_entries"]), 4),
                  "box_entry_added_value": round(num(r["added_value"]), 4)})

    for r in load("Cross Models", "xp_team_leaderboard_eredivisie.csv"):
        t = teams.setdefault(r["team"], {"team_name": r["team"]})
        t.update({"xp_crosses": int(num(r["crosses"])), "xp_cross_completed": int(num(r["completed"])),
                  "xp_cross_expected_completed": round(num(r["expected_completed"]), 4),
                  "xp_cross_completion_rate": round(num(r["completion_rate"]), 4),
                  "xp_cross_expected_completion_rate": round(num(r["expected_completion_rate"]), 4),
                  "xp_cross_added_value": round(num(r["xP_added_value"]), 4)})

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
        t.update({"dominant_back_line": r.get("dominant_back_line", ""),
                  "top_formation": r.get("top_formation", "")})

    team_rows = list(teams.values())
    team_rows.sort(key=lambda r: r.get("points", 0), reverse=True)

    all_team_keys = []
    seen_t = set()
    for row in team_rows:
        for k in row:
            if k not in seen_t:
                all_team_keys.append(k)
                seen_t.add(k)
    team_tabs = categorize(all_team_keys, TEAM_TAB_RULES, identity_cols=[])
    team_cols = []
    for _, cols in team_tabs:
        team_cols.extend(cols)

    with open(os.path.join(OUT_DIR, "team_season_aggregated.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=team_cols, extrasaction="ignore", restval="")
        w.writeheader()
        w.writerows(team_rows)

    metric_cols = [c for c in player_cols if c not in ("player_id", "player_name", "team_name")]
    print(f"Wrote {len(player_rows)} players ({len(metric_cols)} metric columns) "
          f"and {len(team_rows)} teams to {OUT_DIR}")
    if MISSING_FILES:
        print(f"Not available for {SEASON} (needs {NEW_METRICS_SEASON}'s pipeline; columns left blank): "
              + ", ".join(sorted(set(MISSING_FILES))))


if __name__ == "__main__":
    main()

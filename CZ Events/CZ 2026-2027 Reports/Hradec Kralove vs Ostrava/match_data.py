"""
Shared data loading + parsing for the FC Hradec Králové vs FC Baník Ostrava
pre-match preview (Chance liga, CZ 2026-2027, matchday 3, 2026-08-09).

Unlike the post-match reports in this repo, there is no event feed for
THIS fixture -- it hasn't been played yet. Instead this pulls each team's
SEASON-TO-DATE totals -- every match they have actually played so far,
pooled together, not a single matchday-1 snapshot -- as the early-season
form/style base. As of today that is 2 matches each: Hradec drew 0-0 away
at Bohemians 1905 after winning 2-1 at home over FK Pardubice; Ostrava
lost 0-4 at home to Slavia Praha after winning 1-0 away at FC Zlín. Same
Opta MA3 event feed conventions as the rest of this repo (see
Disruption/build_disruption_model.py, and the post-match report's own
match_data.py in the sibling "Hradec Kralove vs Pardubice" folder, which
this file's geometry/xG/PPDA functions are carried over from unchanged).
Structurally this file is the same template as the sibling "Ostrava vs
Slavia Praha" and "Bohemians vs Hradec Kralove" pre-match previews,
re-pointed at a different fixture -- the BOHEMIANS_ID/HRADEC_ID/
TEPLICE_ID/PARDUBICE_ID constants below are left untouched (not renamed)
because ALL_MATCHES further down still needs them correct as 4 of the 16
teams in the league-wide season sample; HRADEC_ID doubles as this
fixture's home team, and OSTRAVA_ID (this fixture's away team) is
likewise already correct from the sibling Slavia-fixture report.
ALL_MATCHES itself was expanded from 7 matchday-1-only files to all 16
matchday-1 + matchday-2 files once matchday 2 finished, which is also
where the previously-missing SK Artis Brno vs FK Mladá Boleslav game
(this league has 16 teams, not 14 -- that 8th matchday-1 fixture simply
hadn't been added to the repo yet when the earlier sibling reports were
built) came in.

Team IDs cross-checked two ways: (1) goal-scorer contestantId counts in
the Hradec-Pardubice feed against that match's 2-1 final score, and the
Zlín-Ostrava feed against its 0-1 score (the 1-goal side is Ostrava,
scorer V. Jurečka), and (2) both fixture teams' IDs also appear in "CZ
2026-2027 Matches.csv" as the fixture-3 contestants, which additionally
supplies this fixture's venue/date/kickoff (no matching event feed
exists for those, since the match is still days away of "today").

Usage: import this module from the chart scripts in this folder.
"""
import json
import math
import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
EVENTS_DIR = os.path.join(REPO_ROOT, "CZ Events", "CZ 2026-2027")

BOHEMIANS_ID = "bqcrqg0367eqzrt4vjb5apu6g"
HRADEC_ID = "1v75g4bk8vzrvu0jmaro6lila"
TEPLICE_ID = "41eivtin75c5fu33x3zfx956b"
PARDUBICE_ID = "4xbgquadoen1b303u4hi9nhg9"

OSTRAVA_ID = "dfvvrv84skv23rsn1k6kt4slc"
ZLIN_ID = "aj1nbeiatqrs6e47mnhjidn15"
ARTIS_BRNO_ID = "6onh4wiqdb8r50qa6oyxlbafg"
MLADA_BOLESLAV_ID = "2qui2adwsi022vu84b8gdw4z0"

TEAM_NAMES = {
    HRADEC_ID: "FC Hradec Králové",
    OSTRAVA_ID: "FC Baník Ostrava",
    PARDUBICE_ID: "FK Pardubice",
    ZLIN_ID: "FC Zlín",
}
TEAM_SHORT = {
    HRADEC_ID: "Hradec Kr.",
    OSTRAVA_ID: "Ostrava",
    PARDUBICE_ID: "Pardubice",
    ZLIN_ID: "Zlín",
}

# This fixture (not yet played)
FIXTURE_HOME_ID, FIXTURE_AWAY_ID = HRADEC_ID, OSTRAVA_ID
FIXTURE_HOME_NAME, FIXTURE_AWAY_NAME = TEAM_NAMES[HRADEC_ID], TEAM_NAMES[OSTRAVA_ID]
COMPETITION = "Chance Liga 2026/27, Matchday 3"
VENUE = "FINEP Arena, Hradec Králové"
MATCH_DATE = "2026-08-09"
KICKOFF_LOCAL = "17:00"
SOURCE = "Opta event data (every match each team has played this season) + own xG model"

X_SCALE, Y_SCALE = 1.05, 0.68     # Opta 0-100 units -> metres (105 x 68 pitch)
GOAL_X = 105.0
GOAL_Y = 34.0
GOAL_WIDTH = 7.32

T_PASS, T_TAKE_ON, T_FOUL, T_OUT = 1, 3, 4, 5
T_CORNER_AWARDED = 6
T_TACKLE, T_INTERCEPTION = 7, 8
T_CLEARANCE = 12
T_MISS, T_ATTEMPT_SAVED, T_GOAL, T_POST = 13, 15, 16, 14
T_CARD = 17
T_SUB_OFF, T_SUB_ON = 18, 19
T_CHALLENGE = 45
T_AERIAL = 44
T_BALL_RECOVERY, T_DISPOSSESSED = 49, 50
T_BLOCKED_PASS = 74

SHOT_TYPES = {T_MISS, T_POST, T_ATTEMPT_SAVED, T_GOAL}
DEFENSIVE_TYPES = {T_TACKLE: "Tackle", T_INTERCEPTION: "Interception", T_CLEARANCE: "Clearance"}
PRESSING_TYPES = {T_TACKLE: "Tackle", T_INTERCEPTION: "Interception", T_CHALLENGE: "Challenge"}

Q_LONG_BALL = 1
Q_CROSS, Q_THROUGH, Q_FREE_KICK, Q_CORNER = 2, 3, 5, 6
Q_HEAD = 15
Q_RIGHT_FOOT, Q_LEFT_FOOT = 20, 72
Q_END_X, Q_END_Y = 140, 141
Q_ZONE = 56
Q_REGULAR_PLAY, Q_FAST_BREAK, Q_SET_PIECE, Q_FROM_CORNER = 22, 23, 24, 25
Q_BIG_CHANCE = 80
Q_YELLOW_CARD, Q_SECOND_YELLOW, Q_RED_CARD = 31, 32, 33
Q_RELATED_EVENT = 233
Q_GOAL_KICK = 124
Q_CUTBACK = 195
Q_THROW_IN = 107

SET_PIECE_QIDS = {Q_FREE_KICK, Q_CORNER, Q_THROW_IN}
WIDE_THIRD = 100 / 3
SWITCH_MIN_DIST_M = 30
PITCH_X, PITCH_Y = 105.0, 68.0

ZONE14 = (70.0, 88.5, 27.2, 40.8)              # x0, x1, y0, y1
HALF_SPACES = [(52.5, 105.0, 13.6, 27.2), (52.5, 105.0, 40.8, 54.4)]
BOX_Y = (13.84, 54.16)

PPDA_ZONE_M = 63.0


def qmap(e):
    return {q["qualifierId"]: q.get("value") for q in e.get("qualifier", []) or []}


def has_q(e, qid):
    return any(q["qualifierId"] == qid for q in e.get("qualifier", []) or [])


def event_time(e):
    return e["timeMin"] * 60 + e["timeSec"]


def load_match(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    events = data["event"]
    events.sort(key=lambda e: (e["periodId"], event_time(e), e["eventId"]))
    return data["matchDetails"], events


def team_name(cid):
    # Falls back to ALL_TEAM_NAMES (all 16 teams, defined later in this module)
    # so TeamSnapshot.team_name resolves correctly for any team in the league
    # sample, not just this fixture's two sides in the small TEAM_NAMES dict.
    if cid in TEAM_NAMES:
        return TEAM_NAMES[cid]
    return ALL_TEAM_NAMES.get(cid, cid)


def team_short(cid):
    return TEAM_SHORT.get(cid, cid)


def to_m(x, y):
    return x * X_SCALE, y * Y_SCALE


def compute_attack_directions(events):
    sums = {}
    for e in events:
        if e["typeId"] != T_PASS or e.get("x") is None:
            continue
        if e["x"] == 0 and e["y"] == 0:
            continue
        key = (e["contestantId"], e["periodId"])
        s = sums.setdefault(key, [0.0, 0])
        s[0] += e["x"]
        s[1] += 1
    return {key: (1 if (total / n if n else 50) < 50 else -1) for key, (total, n) in sums.items()}


def norm_xy(e, directions):
    d = directions.get((e["contestantId"], e["periodId"]), 1)
    x, y = e["x"], e["y"]
    if d == 1:
        return x, y
    return 100.0 - x, 100.0 - y


def shot_angle_deg(x_m, y_m):
    dx = GOAL_X - x_m
    if dx <= 0:
        return 0.0
    y1 = y_m - (GOAL_Y - GOAL_WIDTH / 2)
    y2 = y_m - (GOAL_Y + GOAL_WIDTH / 2)
    denom = dx * dx + y1 * y2
    a = math.atan2(GOAL_WIDTH * dx, denom) if denom != 0 else math.pi / 2
    if a < 0:
        a += math.pi
    return math.degrees(a)


def shot_xg(x_m, y_m, is_header):
    dist = math.hypot(GOAL_X - x_m, GOAL_Y - y_m)
    angle = shot_angle_deg(x_m, y_m)
    z = -2.0 + 3.6 * math.radians(angle) - 0.085 * dist - (0.65 if is_header else 0.0)
    xg = 1.0 / (1.0 + math.exp(-z))
    return max(0.015, min(0.94, xg))


def xt_value(x100, y100):
    x_m, y_m = to_m(x100, y100)
    return shot_xg(x_m, y_m, is_header=False)


def build_shots(events, directions):
    rows = []
    for e in events:
        if e["typeId"] not in SHOT_TYPES or e.get("x") is None:
            continue
        x, y = norm_xy(e, directions)
        xm, ym = to_m(x, y)
        is_header = has_q(e, Q_HEAD)
        xg = shot_xg(xm, ym, is_header)
        outcome = {T_GOAL: "Goal", T_ATTEMPT_SAVED: "Saved", T_MISS: "Miss", T_POST: "Post"}[e["typeId"]]
        if has_q(e, Q_FROM_CORNER):
            situation = "Corner"
        elif has_q(e, Q_SET_PIECE):
            situation = "Set piece"
        elif has_q(e, Q_FAST_BREAK):
            situation = "Fast break"
        else:
            situation = "Open play"
        rows.append({
            "contestantId": e["contestantId"],
            "team": team_name(e["contestantId"]),
            "player": e.get("playerName", "Unknown"),
            "eventId": e["eventId"],
            "minute": e["timeMin"],
            "period": e["periodId"],
            "x": xm, "y": ym,
            "outcome": outcome,
            "on_target": e["typeId"] in (T_GOAL, T_ATTEMPT_SAVED),
            "is_goal": e["typeId"] == T_GOAL,
            "is_header": is_header,
            "big_chance": has_q(e, Q_BIG_CHANCE),
            "situation": situation,
            "xg": xg,
        })
    return rows


def build_passes(events, directions):
    rows = []
    for e in events:
        if e["typeId"] != T_PASS or e.get("x") is None:
            continue
        q = qmap(e)
        x, y = norm_xy(e, directions)
        xm, ym = to_m(x, y)
        completed = e["outcome"] == 1
        end_x = end_y = ex = ey = None
        if Q_END_X in q and Q_END_Y in q:
            d = directions.get((e["contestantId"], e["periodId"]), 1)
            ex, ey = float(q[Q_END_X]), float(q[Q_END_Y])
            if d == -1:
                ex, ey = 100.0 - ex, 100.0 - ey
            end_x, end_y = to_m(ex, ey)
        start_dist = math.hypot(GOAL_X - xm, GOAL_Y - ym)
        end_dist = math.hypot(GOAL_X - end_x, GOAL_Y - end_y) if end_x is not None else None
        progressive = (completed and end_dist is not None and
                       end_dist <= start_dist * 0.75 and end_x > xm)
        xt_start = xt_value(x, y)
        xt_end = xt_value(ex, ey) if ex is not None else None
        xt_added = (xt_end - xt_start) if (completed and xt_end is not None) else 0.0

        is_switch = False
        if not (SET_PIECE_QIDS & set(q.keys())) and ex is not None:
            raw_y0, raw_y1 = e["y"], float(q[Q_END_Y])
            left0, right0 = raw_y0 <= WIDE_THIRD, raw_y0 >= (100 - WIDE_THIRD)
            left1, right1 = raw_y1 <= WIDE_THIRD, raw_y1 >= (100 - WIDE_THIRD)
            if (left0 and right1) or (right0 and left1):
                dx = (float(q[Q_END_X]) - e["x"]) / 100 * PITCH_X
                dy = (raw_y1 - raw_y0) / 100 * PITCH_Y
                if math.hypot(dx, dy) >= SWITCH_MIN_DIST_M:
                    is_switch = True

        rows.append({
            "contestantId": e["contestantId"],
            "team": team_name(e["contestantId"]),
            "player": e.get("playerName", "Unknown"),
            "playerId": e.get("playerId"),
            "minute": e["timeMin"], "second": e["timeSec"],
            "period": e["periodId"],
            "eventId": e["eventId"],
            "x": xm, "y": ym,
            "end_x": end_x, "end_y": end_y,
            "completed": completed,
            "is_cross": has_q(e, Q_CROSS),
            "is_corner": has_q(e, Q_CORNER),
            "is_long_ball": has_q(e, Q_LONG_BALL),
            "is_cutback": has_q(e, Q_CUTBACK),
            "is_goal_kick": has_q(e, Q_GOAL_KICK),
            "is_switch": is_switch and completed,
            "progressive": progressive,
            "final_third_entry": completed and start_dist > 35.0 and end_dist is not None and end_dist <= 35.0,
            "box_entry": (completed and end_x is not None and end_x >= 88.5
                          and 13.84 <= end_y <= 54.16 and not (xm >= 88.5 and 13.84 <= ym <= 54.16)),
            "xt_added": xt_added,
        })
    return rows


def build_defensive_actions(events, directions):
    rows = []
    for e in events:
        if e["typeId"] not in DEFENSIVE_TYPES or e.get("x") is None:
            continue
        x, y = norm_xy(e, directions)
        xm, ym = to_m(x, y)
        rows.append({
            "contestantId": e["contestantId"],
            "team": team_name(e["contestantId"]),
            "player": e.get("playerName", "Unknown"),
            "minute": e["timeMin"], "period": e["periodId"],
            "x": xm, "y": ym,
            "action": DEFENSIVE_TYPES[e["typeId"]],
            "success": e.get("outcome", 1) == 1,
        })
    return rows


def build_pressing_actions(events, directions):
    rows = []
    for e in events:
        if e.get("x") is None or not e.get("contestantId"):
            continue
        t = e["typeId"]
        if t in PRESSING_TYPES:
            action = PRESSING_TYPES[t]
        elif t == T_FOUL and e.get("outcome") == 0:
            action = "Foul"
        else:
            continue
        x, y = norm_xy(e, directions)
        xm, ym = to_m(x, y)
        rows.append({
            "contestantId": e["contestantId"],
            "team": team_name(e["contestantId"]),
            "player": e.get("playerName", "Unknown"),
            "minute": e["timeMin"], "period": e["periodId"],
            "x": xm, "y": ym,
            "action": action,
        })
    return rows


def compute_ppda(passes, pressing_actions, contestant_id, lo=None, hi=None):
    """Opponent = "not contestant_id" rather than a single fixed id, so this
    still works when passes/pressing_actions are pooled across several
    matches (each match only ever has two sides, so "not us" == "whichever
    opponent we faced that game")."""
    def in_window(m):
        return (lo is None or m >= lo) and (hi is None or m < hi)

    opp_passes = sum(1 for p in passes if p["contestantId"] != contestant_id
                      and in_window(p["minute"]) and p["x"] <= PPDA_ZONE_M)
    def_actions = sum(1 for d in pressing_actions if d["contestantId"] == contestant_id
                       and in_window(d["minute"]) and d["x"] >= (105.0 - PPDA_ZONE_M))
    return opp_passes / def_actions if def_actions else float("nan")


def build_recoveries(events, directions):
    rows = []
    for e in events:
        if e["typeId"] != T_BALL_RECOVERY or e.get("x") is None:
            continue
        x, y = norm_xy(e, directions)
        xm, ym = to_m(x, y)
        rows.append({
            "contestantId": e["contestantId"],
            "team": team_name(e["contestantId"]),
            "player": e.get("playerName", "Unknown"),
            "minute": e["timeMin"], "period": e["periodId"],
            "x": xm, "y": ym,
        })
    return rows


def build_cards(events):
    rows = []
    for e in events:
        if e["typeId"] != T_CARD:
            continue
        if has_q(e, Q_RED_CARD) or has_q(e, Q_SECOND_YELLOW):
            kind = "Red" if has_q(e, Q_RED_CARD) else "2nd Yellow"
        else:
            kind = "Yellow"
        rows.append({
            "contestantId": e["contestantId"],
            "team": team_name(e["contestantId"]),
            "player": e.get("playerName", "Unknown"),
            "minute": e["timeMin"],
            "kind": kind,
        })
    return rows


def build_touches(events, directions):
    rows = []
    for e in events:
        if e.get("x") is None or not e.get("contestantId"):
            continue
        if e["typeId"] in (T_SUB_OFF, T_SUB_ON):
            continue
        x, y = norm_xy(e, directions)
        xm, ym = to_m(x, y)
        rows.append({
            "contestantId": e["contestantId"],
            "team": team_name(e["contestantId"]),
            "minute": e["timeMin"], "second": e["timeSec"], "period": e["periodId"],
            "x": xm, "y": ym,
            "typeId": e["typeId"],
        })
    return rows


def build_duels(events, directions):
    """Tackle + Aerial + Challenge, each contestantId's own outcome --
    same convention as the post-match report's _build_duels helper."""
    rows = []
    for e in events:
        if e.get("x") is None or not e.get("contestantId"):
            continue
        if e["typeId"] not in (T_TACKLE, T_AERIAL, T_CHALLENGE):
            continue
        x, y = norm_xy(e, directions)
        xm, ym = to_m(x, y)
        rows.append({
            "contestantId": e["contestantId"],
            "team": team_name(e["contestantId"]),
            "player": e.get("playerName", "Unknown"),
            "minute": e["timeMin"], "period": e["periodId"],
            "x": xm, "y": ym,
            "action": {T_TACKLE: "Tackle", T_AERIAL: "Aerial", T_CHALLENGE: "Challenge"}[e["typeId"]],
            "success": e.get("outcome", 1) == 1,
        })
    return rows


def build_substitutions(events):
    rows = []
    for e in events:
        if e["typeId"] != T_SUB_OFF:
            continue
        rows.append({"contestantId": e["contestantId"], "team": team_name(e["contestantId"]),
                      "player_off": e.get("playerName", "Unknown"), "minute": e["timeMin"]})
    return rows


def build_turnovers(events, directions):
    """A team's own failed pass or Dispossessed event in their own attacking
    half -- the moment they lost the ball already deep in the opponent's
    territory, handing it straight back in a dangerous area."""
    rows = []
    for e in events:
        if e.get("x") is None or not e.get("contestantId"):
            continue
        is_failed_pass = e["typeId"] == T_PASS and e.get("outcome") == 0
        is_dispossessed = e["typeId"] == T_DISPOSSESSED
        if not (is_failed_pass or is_dispossessed):
            continue
        x, y = norm_xy(e, directions)
        xm, ym = to_m(x, y)
        if xm < 52.5:
            continue
        rows.append({"contestantId": e["contestantId"], "team": team_name(e["contestantId"]),
                      "player": e.get("playerName", "Unknown"), "minute": e["timeMin"], "period": e["periodId"],
                      "x": xm, "y": ym, "kind": "Failed pass" if is_failed_pass else "Dispossessed"})
    return rows


def build_shot_assists(events, directions, shots):
    """For each shot, the most recent completed pass by the shooting team
    since the ball last changed teams -- "who set this shot up"."""
    ball_events = [e for e in events if e.get("x") is not None and e.get("contestantId")
                   and (e["typeId"] in (T_PASS, T_TAKE_ON, T_TACKLE, T_INTERCEPTION, T_CLEARANCE,
                                        T_AERIAL, T_BALL_RECOVERY, T_DISPOSSESSED, 61)
                        or e["typeId"] in SHOT_TYPES)]
    ball_events.sort(key=lambda e: (e["periodId"], event_time(e), e["eventId"]))

    assists = {}
    current_team, pending_pass = None, None
    for e in ball_events:
        cid = e["contestantId"]
        if cid != current_team:
            current_team, pending_pass = cid, None
        if e["typeId"] in SHOT_TYPES:
            if pending_pass is not None and pending_pass.get("playerName") != e.get("playerName"):
                assists[e["eventId"]] = pending_pass
        elif e["typeId"] == T_PASS and e.get("outcome") == 1:
            pending_pass = e

    xg_by_eventid = {s["eventId"]: s["xg"] for s in shots}
    rows = []
    for e in events:
        if e["typeId"] not in SHOT_TYPES or e.get("x") is None:
            continue
        ap = assists.get(e["eventId"])
        if ap is None:
            continue
        x, y = norm_xy(ap, directions)
        xm, ym = to_m(x, y)
        q = qmap(ap)
        end_x = end_y = None
        if Q_END_X in q and Q_END_Y in q:
            d = directions.get((ap["contestantId"], ap["periodId"]), 1)
            ex, ey = float(q[Q_END_X]), float(q[Q_END_Y])
            if d == -1:
                ex, ey = 100.0 - ex, 100.0 - ey
            end_x, end_y = to_m(ex, ey)
        sx, sy = norm_xy(e, directions)
        sxm, sym = to_m(sx, sy)
        rows.append({
            "contestantId": e["contestantId"], "team": team_name(e["contestantId"]),
            "shooter": e.get("playerName", "Unknown"), "assister": ap.get("playerName", "Unknown"),
            "minute": e["timeMin"], "x": xm, "y": ym,
            "end_x": end_x if end_x is not None else sxm, "end_y": end_y if end_y is not None else sym,
            "shot_x": sxm, "shot_y": sym, "shot_xg": xg_by_eventid.get(e["eventId"], 0.0),
            "is_goal": e["typeId"] == T_GOAL,
        })
    return rows


class TeamSnapshot:
    """Everything derived from ALL of a team's played matches so far this
    season (season-to-date totals, not a single matchday snapshot): shots,
    passes, defensive/pressing actions, touches, cards etc, pooled across
    every match in ALL_TEAM_MATCHES[team_id] and indexable by either that
    team's own contestantId (its own numbers) or "not that id" (numbers
    conceded, whichever opponent was on the other side of each match).

    Per-match info (opponent, venue, own result) is kept in
    self.matches for anything that needs a per-match breakdown (e.g. a
    match log), while every aggregate stat below is summed/pooled/averaged
    across the full list."""

    def __init__(self, team_id):
        records = ALL_TEAM_MATCHES[team_id]
        self.team_id = team_id
        self.team_name = team_name(team_id)
        self.matches = []
        self.shots, self.passes, self.defs, self.pressing = [], [], [], []
        self.recoveries, self.cards, self.touches, self.duels = [], [], [], []
        self.subs, self.turnovers, self.assists = [], [], []

        for idx, m in enumerate(records):
            is_home = m["home_id"] == team_id
            opponent_id = m["away_id"] if is_home else m["home_id"]
            opponent_name = m["away_name"] if is_home else m["home_name"]
            match_details, events = load_match(m["path"])
            directions = compute_attack_directions(events)
            shots = build_shots(events, directions)
            for s in shots:
                s["match_idx"] = idx  # which entry in self.matches this shot's own build-up chain lives in

            scores = match_details["scores"]["ft"]
            own_score = scores["home"] if is_home else scores["away"]
            opp_score = scores["away"] if is_home else scores["home"]
            outcome = "Won" if own_score > opp_score else ("Lost" if own_score < opp_score else "Drew")
            venue_desc = "at home" if is_home else "away"
            result = f"{outcome} {own_score}-{opp_score} {venue_desc}"

            self.matches.append(dict(
                matchday=m["matchday"], opponent_id=opponent_id, opponent_name=opponent_name,
                was_home=is_home, outcome=outcome, own_score=own_score, opp_score=opp_score,
                result=result, events=events, directions=directions,
            ))
            self.shots += shots
            self.passes += build_passes(events, directions)
            self.defs += build_defensive_actions(events, directions)
            self.pressing += build_pressing_actions(events, directions)
            self.recoveries += build_recoveries(events, directions)
            self.cards += build_cards(events)
            self.touches += build_touches(events, directions)
            self.duels += build_duels(events, directions)
            self.subs += build_substitutions(events)
            self.turnovers += build_turnovers(events, directions)
            self.assists += build_shot_assists(events, directions, shots)

        self.matches_played = len(self.matches)
        self.wins = sum(1 for m in self.matches if m["outcome"] == "Won")
        self.draws = sum(1 for m in self.matches if m["outcome"] == "Drew")
        self.losses = sum(1 for m in self.matches if m["outcome"] == "Lost")
        self.points = self.wins * 3 + self.draws
        self.record = f"{self.wins}W {self.draws}D {self.losses}L"

    def own(self, rows):
        return [r for r in rows if r["contestantId"] == self.team_id]

    def against(self, rows):
        return [r for r in rows if r["contestantId"] != self.team_id]

    @property
    def opponents_label(self):
        """A single opponent's name if only one match is on record, else a
        generic plural -- used in dek/caption text that used to say "vs
        {single opponent}" when every page was built from one match."""
        if self.matches_played == 1:
            return self.matches[0]["opponent_name"]
        return "their opponents"

    @property
    def xg_for(self):
        return sum(s["xg"] for s in self.own(self.shots))

    @property
    def xg_against(self):
        return sum(s["xg"] for s in self.against(self.shots))

    def ppda_for(self):
        return compute_ppda(self.passes, self.pressing, self.team_id)

    def verticality(self):
        fwd = [p["end_x"] - p["x"] for p in self.own(self.passes)
               if p["completed"] and p["end_x"] is not None and p["end_x"] > p["x"]]
        return sum(fwd) / len(fwd) if fwd else 0.0

    def def_line_height(self):
        d = self.own(self.defs) + [p for p in self.own(self.pressing) if p["action"] != "Foul"]
        return sum(p["x"] for p in d) / len(d) if d else float("nan")

    def halfspace_zone14_counts(self):
        passes = [p for p in self.own(self.passes) if p["completed"] and p["end_x"] is not None]
        zone14 = sum(1 for p in passes if ZONE14[0] <= p["end_x"] < ZONE14[1] and ZONE14[2] <= p["end_y"] < ZONE14[3])
        halfspace = sum(1 for p in passes if any(x0 <= p["end_x"] < x1 and y0 <= p["end_y"] < y1
                                                  for x0, x1, y0, y1 in HALF_SPACES))
        return halfspace, zone14

    def touch_share(self):
        h = len(self.own(self.touches))
        a = len(self.against(self.touches))
        return h / (h + a) if (h + a) else float("nan")

    def field_tilt(self):
        h = sum(1 for t in self.own(self.touches) if t["x"] >= 70)
        a = sum(1 for t in self.against(self.touches) if t["x"] >= 70)
        return h / (h + a) if (h + a) else float("nan")

    def pass_tempo_mps(self):
        """Avg metres/second of this team's completed passes -- pass
        distance divided by the time gap to the next event in that same
        match (any team). Gaps > 8s (stoppages, fouls, VAR checks) are
        excluded as noise, not genuine tempo. Computed per match (event
        order only means anything within one match) then pooled across
        every match played."""
        speeds = []
        for m in self.matches:
            events, directions = m["events"], m["directions"]
            ordered = sorted(events, key=lambda e: (e["periodId"], event_time(e), e["eventId"]))
            for i in range(len(ordered) - 1):
                e = ordered[i]
                if e["typeId"] != T_PASS or e.get("contestantId") != self.team_id or e.get("outcome") != 1:
                    continue
                nxt = ordered[i + 1]
                if nxt["periodId"] != e["periodId"]:
                    continue
                dt = event_time(nxt) - event_time(e)
                if dt <= 0 or dt > 8:
                    continue
                q = qmap(e)
                if Q_END_X not in q or Q_END_Y not in q:
                    continue
                x, y = norm_xy(e, directions)
                xm, ym = to_m(x, y)
                d = directions.get((e["contestantId"], e["periodId"]), 1)
                ex, ey = float(q[Q_END_X]), float(q[Q_END_Y])
                if d == -1:
                    ex, ey = 100.0 - ex, 100.0 - ey
                exm, eym = to_m(ex, ey)
                dist = math.hypot(exm - xm, eym - ym)
                speeds.append(dist / dt)
        return sum(speeds) / len(speeds) if speeds else 0.0

    def pass_volume(self):
        return len(self.own(self.passes))


def simulate_scorelines(home_shots, away_shots, n=20000, seed=42, cap=6):
    """Monte Carlo simulation cross-pairing each fixture side's own MW1
    shot-xG list (their shot volume + quality from the one match on
    record) as a form proxy for this unplayed fixture. Same mechanic as
    the post-match report's simulate_scorelines, but the two shot lists
    come from two different, unrelated matches -- an early-season style
    projection, not a model fit to this specific pairing."""
    import random
    rng = random.Random(seed)
    home_xgs = [s["xg"] for s in home_shots]
    away_xgs = [s["xg"] for s in away_shots]

    score_counts = {}
    home_goals, away_goals = [], []
    for _ in range(n):
        h = sum(1 for xg in home_xgs if rng.random() < xg)
        a = sum(1 for xg in away_xgs if rng.random() < xg)
        home_goals.append(h)
        away_goals.append(a)
        key = (min(h, cap), min(a, cap))
        score_counts[key] = score_counts.get(key, 0) + 1

    home_win = sum(1 for h, a in zip(home_goals, away_goals) if h > a) / n
    draw = sum(1 for h, a in zip(home_goals, away_goals) if h == a) / n
    away_win = sum(1 for h, a in zip(home_goals, away_goals) if h < a) / n
    return {
        "score_counts": score_counts, "n": n, "cap": cap,
        "home_win": home_win, "draw": draw, "away_win": away_win,
        "home_goal_dist": home_goals, "away_goal_dist": away_goals,
    }


# ---------------------------------------------------------------------------
# Full 16-team league sample -- every match feed this repo has for the
# season so far (matchdays 1-2, 16 matches, all 16 teams), used by the
# pace-vs-volume and verticality-ranking pages and by TeamSnapshot itself
# for this fixture's two sides. Season to date, not a single round -- once
# more matchdays are played, adding their event files to ALL_MATCHES is
# the only change needed for every page to pick them up automatically,
# since TeamSnapshot pools over however many matches ALL_TEAM_MATCHES
# lists for a given team.
# ---------------------------------------------------------------------------

ALL_MATCHES = [
    # Matchday 1 (2026-07-25 to 2026-07-27)
    dict(matchday=1, path=os.path.join(EVENTS_DIR, "2026-07-25_FC Viktoria Plzeň - FC Slovan Liberec.json"),
         home_id="c6fx1460nlkawjgh67sp7a1hd", home_name="Viktoria Plzeň",
         away_id="2c4rs2vp0tyjiqa7y3gfttf24", away_name="Slovan Liberec"),
    dict(matchday=1, path=os.path.join(EVENTS_DIR, "2026-07-25_FC Zbrojovka Brno - AC Sparta Praha.json"),
         home_id="6k350zwynsc23f0sxw9akgc6y", home_name="Zbrojovka Brno",
         away_id="5ocdn3a6s75u0d0dy0rbou0xc", away_name="Sparta Praha"),
    dict(matchday=1, path=os.path.join(EVENTS_DIR, "2026-07-25_FC Zlín - FC Baník Ostrava.json"),
         home_id=ZLIN_ID, home_name="Zlín", away_id=OSTRAVA_ID, away_name="Baník Ostrava"),
    dict(matchday=1, path=os.path.join(EVENTS_DIR, "2026-07-25_FK Teplice - Bohemians Praha 1905.json"),
         home_id=TEPLICE_ID, home_name="Teplice", away_id=BOHEMIANS_ID, away_name="Bohemians 1905"),
    dict(matchday=1, path=os.path.join(EVENTS_DIR, "2026-07-26_FC Hradec Králové - FK Pardubice.json"),
         home_id=HRADEC_ID, home_name="Hradec Králové", away_id=PARDUBICE_ID, away_name="Pardubice"),
    dict(matchday=1, path=os.path.join(EVENTS_DIR, "2026-07-26_FK Jablonec - SK Sigma Olomouc.json"),
         home_id="bdz8tx20ekj1ryi2e2u13jdl", home_name="Jablonec",
         away_id="dchxm00ei80l8ljbcfpdill8k", away_name="Sigma Olomouc"),
    dict(matchday=1, path=os.path.join(EVENTS_DIR, "2026-07-26_SK Slavia Praha - 1. FC Slovácko.json"),
         home_id="8kpapuorr6hf0vosnovbreqqd", home_name="Slavia Praha",
         away_id="bp5x8iw8pstucx6s4iqht6xqf", away_name="Slovácko"),
    dict(matchday=1, path=os.path.join(EVENTS_DIR, "2026-07-27_SK Artis Brno - FK Mladá Boleslav.json"),
         home_id=ARTIS_BRNO_ID, home_name="Artis Brno", away_id=MLADA_BOLESLAV_ID, away_name="Mladá Boleslav"),
    # Matchday 2 (2026-07-31 to 2026-08-02)
    dict(matchday=2, path=os.path.join(EVENTS_DIR, "2026-07-31_AC Sparta Praha - FC Zlín.json"),
         home_id="5ocdn3a6s75u0d0dy0rbou0xc", home_name="Sparta Praha", away_id=ZLIN_ID, away_name="Zlín"),
    dict(matchday=2, path=os.path.join(EVENTS_DIR, "2026-08-01_1. FC Slovácko - SK Artis Brno.json"),
         home_id="bp5x8iw8pstucx6s4iqht6xqf", home_name="Slovácko", away_id=ARTIS_BRNO_ID, away_name="Artis Brno"),
    dict(matchday=2, path=os.path.join(EVENTS_DIR, "2026-08-01_FC Baník Ostrava - SK Slavia Praha.json"),
         home_id=OSTRAVA_ID, home_name="Baník Ostrava",
         away_id="8kpapuorr6hf0vosnovbreqqd", away_name="Slavia Praha"),
    dict(matchday=2, path=os.path.join(EVENTS_DIR, "2026-08-01_FC Slovan Liberec - FK Teplice.json"),
         home_id="2c4rs2vp0tyjiqa7y3gfttf24", home_name="Slovan Liberec", away_id=TEPLICE_ID, away_name="Teplice"),
    dict(matchday=2, path=os.path.join(EVENTS_DIR, "2026-08-01_FC Viktoria Plzeň - FC Zbrojovka Brno.json"),
         home_id="c6fx1460nlkawjgh67sp7a1hd", home_name="Viktoria Plzeň",
         away_id="6k350zwynsc23f0sxw9akgc6y", away_name="Zbrojovka Brno"),
    dict(matchday=2, path=os.path.join(EVENTS_DIR, "2026-08-02_Bohemians Praha 1905 - FC Hradec Králové.json"),
         home_id=BOHEMIANS_ID, home_name="Bohemians 1905", away_id=HRADEC_ID, away_name="Hradec Králové"),
    dict(matchday=2, path=os.path.join(EVENTS_DIR, "2026-08-02_FK Pardubice - FK Jablonec.json"),
         home_id=PARDUBICE_ID, home_name="Pardubice", away_id="bdz8tx20ekj1ryi2e2u13jdl", away_name="Jablonec"),
    dict(matchday=2, path=os.path.join(EVENTS_DIR, "2026-08-02_SK Sigma Olomouc - FK Mladá Boleslav.json"),
         home_id="dchxm00ei80l8ljbcfpdill8k", home_name="Sigma Olomouc",
         away_id=MLADA_BOLESLAV_ID, away_name="Mladá Boleslav"),
]

ALL_TEAM_NAMES = {}
ALL_TEAM_MATCHES = {}
for m in ALL_MATCHES:
    ALL_TEAM_NAMES[m["home_id"]] = m["home_name"]
    ALL_TEAM_NAMES[m["away_id"]] = m["away_name"]
    ALL_TEAM_MATCHES.setdefault(m["home_id"], []).append(m)
    ALL_TEAM_MATCHES.setdefault(m["away_id"], []).append(m)


class LeagueSeason:
    """Every team with a match feed in this repo, one season-to-date
    TeamSnapshot each (pools however many matches ALL_TEAM_MATCHES lists
    for that team -- 2 each right now, matchdays 1-2)."""

    def __init__(self):
        self.teams = {tid: TeamSnapshot(tid) for tid in ALL_TEAM_NAMES}

    def ranking(self, metric_fn, reverse=True):
        vals = [(tid, metric_fn(tm)) for tid, tm in self.teams.items()]
        vals = [(tid, v) for tid, v in vals if v == v]
        return sorted(vals, key=lambda kv: kv[1], reverse=reverse)


if __name__ == "__main__":
    boh = TeamSnapshot(HRADEC_ID)
    hkr = TeamSnapshot(OSTRAVA_ID)
    for snap in (boh, hkr):
        print(team_name(snap.team_id), f"({snap.matches_played} matches played):", snap.record,
              f"({snap.points} pts)")
        for m in snap.matches:
            print(f"    MD{m['matchday']}: {m['result']} vs {m['opponent_name']}")
        print("  xG for/against:", round(snap.xg_for, 2), round(snap.xg_against, 2))
        print("  shots for/against:", len(snap.own(snap.shots)), len(snap.against(snap.shots)))
        print("  PPDA:", round(snap.ppda_for(), 2))

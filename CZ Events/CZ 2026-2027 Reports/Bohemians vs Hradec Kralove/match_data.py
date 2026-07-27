"""
Shared data loading + parsing for the Bohemians 1905 vs FC Hradec Králové
pre-match preview (Chance liga, CZ 2026-2027, matchday 2, 2026-08-02).

Unlike the post-match reports in this repo, there is no event feed for
THIS fixture -- it hasn't been played yet. Instead this pulls each team's
own matchday-1 event feed (Bohemians' 1-3 loss at FK Teplice; Hradec's
2-1 win over FK Pardubice) as the early-season form/style base, one file
per team rather than one shared file for both. Same Opta MA3 event feed
conventions as the rest of this repo (see Disruption/build_disruption_
model.py, and the post-match report's own match_data.py in the sibling
"Hradec Kralove vs Pardubice" folder, which this file's geometry/xG/PPDA
functions are carried over from unchanged).

Team IDs cross-checked two ways: (1) goal-scorer contestantId counts in
the Teplice-Bohemians feed against that match's 3-1 final score (the
3-goal contestantId is Teplice, the 1-goal side Bohemians), and (2) both
teams' IDs also appear in "CZ 2026-2027 Matches.csv" as the fixture-2
contestants, which additionally supplies this fixture's venue/date/
kickoff (no matching event feed exists for those, since the match is
still 6 days away of "today").

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

TEAM_NAMES = {
    BOHEMIANS_ID: "Bohemians 1905",
    HRADEC_ID: "FC Hradec Králové",
    TEPLICE_ID: "FK Teplice",
    PARDUBICE_ID: "FK Pardubice",
}
TEAM_SHORT = {
    BOHEMIANS_ID: "Bohemians",
    HRADEC_ID: "Hradec Kr.",
    TEPLICE_ID: "Teplice",
    PARDUBICE_ID: "Pardubice",
}

# This fixture (not yet played)
FIXTURE_HOME_ID, FIXTURE_AWAY_ID = BOHEMIANS_ID, HRADEC_ID
FIXTURE_HOME_NAME, FIXTURE_AWAY_NAME = TEAM_NAMES[BOHEMIANS_ID], TEAM_NAMES[HRADEC_ID]
COMPETITION = "Chance Liga 2026/27, Matchday 2"
VENUE = "Stadion Ďolíček, Prague"
MATCH_DATE = "2026-08-02"
KICKOFF_LOCAL = "17:30"
SOURCE = "Opta event data (each team's own matchday-1 fixture) + own xG model"

# Each team's own matchday-1 fixture: (event file, that team's opponent id,
# opponent name, venue, own final score as "W/D/L H-A")
TEAM_MW1 = {
    BOHEMIANS_ID: dict(
        path=os.path.join(EVENTS_DIR, "2026-07-25_FK Teplice - Bohemians Praha 1905.json"),
        opponent_id=TEPLICE_ID, opponent_name="FK Teplice", venue="Na Stínadlech, Teplice",
        was_home=False, result="Lost 1-3 away",
    ),
    HRADEC_ID: dict(
        path=os.path.join(EVENTS_DIR, "2026-07-26_FC Hradec Králové - FK Pardubice.json"),
        opponent_id=PARDUBICE_ID, opponent_name="FK Pardubice", venue="FINEP Arena, Hradec Králové",
        was_home=True, result="Won 2-1 at home",
    ),
}

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
    return TEAM_NAMES.get(cid, cid)


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


def compute_ppda(passes, pressing_actions, contestant_id, opponent_id, lo=None, hi=None):
    def in_window(m):
        return (lo is None or m >= lo) and (hi is None or m < hi)

    opp_passes = sum(1 for p in passes if p["contestantId"] == opponent_id
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


class TeamSnapshot:
    """Everything derived from one team's own matchday-1 fixture: its shots,
    passes, defensive/pressing actions, touches, cards etc, indexable by
    either that team's contestantId (its own numbers) or its MW1
    opponent's (numbers conceded)."""

    def __init__(self, team_id):
        info = TEAM_MW1[team_id]
        self.team_id = team_id
        self.opponent_id = info["opponent_id"]
        self.opponent_name = info["opponent_name"]
        self.venue = info["venue"]
        self.result = info["result"]
        self.match_details, self.events = load_match(info["path"])
        self.directions = compute_attack_directions(self.events)
        self.shots = build_shots(self.events, self.directions)
        self.passes = build_passes(self.events, self.directions)
        self.defs = build_defensive_actions(self.events, self.directions)
        self.pressing = build_pressing_actions(self.events, self.directions)
        self.recoveries = build_recoveries(self.events, self.directions)
        self.cards = build_cards(self.events)
        self.touches = build_touches(self.events, self.directions)
        self.duels = build_duels(self.events, self.directions)

    def own(self, rows):
        return [r for r in rows if r["contestantId"] == self.team_id]

    def against(self, rows):
        return [r for r in rows if r["contestantId"] == self.opponent_id]

    @property
    def xg_for(self):
        return sum(s["xg"] for s in self.own(self.shots))

    @property
    def xg_against(self):
        return sum(s["xg"] for s in self.against(self.shots))

    def ppda_for(self):
        return compute_ppda(self.passes, self.pressing, self.team_id, self.opponent_id)


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


if __name__ == "__main__":
    boh = TeamSnapshot(BOHEMIANS_ID)
    hkr = TeamSnapshot(HRADEC_ID)
    for snap in (boh, hkr):
        print(team_name(snap.team_id), "MW1:", snap.result, "vs", snap.opponent_name)
        print("  xG for/against:", round(snap.xg_for, 2), round(snap.xg_against, 2))
        print("  shots for/against:", len(snap.own(snap.shots)), len(snap.against(snap.shots)))
        print("  PPDA:", round(snap.ppda_for(), 2))

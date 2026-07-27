"""
Shared data loading + parsing for the FC Hradec Králové vs FK Pardubice
post-match report (Chance liga, CZ 2026-2027, matchday 29).

Opta MA3 event feed -- same typeId/qualifierId conventions as the rest of
this repo (see Disruption/build_disruption_model.py, Goal Kick Model/
build_goalkick_shot_model.py). No packaged xG feed for the Czech league
data, so shots are scored with a small distance+angle geometric model
(own model, not provider-supplied -- flagged as such in every chart's
source line, same convention as the Slavia template this report follows).

Usage: import this module from the chart scripts in this folder.
"""
import json
import math
import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
DATA_PATH = os.path.join(
    REPO_ROOT, "CZ Events", "CZ 2026-2027", "2026-07-26_FC Hradec Králové - FK Pardubice.json",
)

HOME_ID = "1v75g4bk8vzrvu0jmaro6lila"
AWAY_ID = "4xbgquadoen1b303u4hi9nhg9"
HOME_NAME = "FC Hradec Králové"
AWAY_NAME = "FK Pardubice"
COMPETITION = "Chance Liga 2026/27, Matchday 29"
VENUE = "FINEP Arena, Hradec Králové"
MATCH_DATE = "2026-07-26"
SOURCE = "Opta event data + own xG model"

X_SCALE, Y_SCALE = 1.05, 0.68     # Opta 0-100 units -> metres (105 x 68 pitch)
GOAL_X = 105.0
GOAL_Y = 34.0
GOAL_WIDTH = 7.32

T_PASS, T_TAKE_ON, T_FOUL, T_OUT = 1, 3, 4, 5
T_CORNER_AWARDED = 6
T_TACKLE, T_INTERCEPTION = 7, 8
T_CLEARANCE = 12
T_MISS, T_ATTEMPT_SAVED, T_GOAL, T_POST = 13, 15, 16, 14
T_SUB_OFF, T_SUB_ON = 18, 19
T_AERIAL = 44
T_BALL_RECOVERY, T_DISPOSSESSED = 49, 50
T_BLOCKED_PASS = 74

SHOT_TYPES = {T_MISS, T_POST, T_ATTEMPT_SAVED, T_GOAL}
DEFENSIVE_TYPES = {T_TACKLE: "Tackle", T_INTERCEPTION: "Interception", T_CLEARANCE: "Clearance"}

Q_HEAD, Q_CROSS, Q_THROUGH, Q_FREE_KICK, Q_CORNER = 1, 2, 3, 5, 6
Q_END_X, Q_END_Y = 140, 141
Q_ZONE = 56
Q_BIG_CHANCE = 80


def qmap(e):
    return {q["qualifierId"]: q.get("value") for q in e.get("qualifier", []) or []}


def has_q(e, qid):
    return any(q["qualifierId"] == qid for q in e.get("qualifier", []) or [])


def event_time(e):
    return e["timeMin"] * 60 + e["timeSec"]


def load_events():
    with open(DATA_PATH, encoding="utf-8") as f:
        data = json.load(f)
    events = data["event"]
    events.sort(key=lambda e: (e["periodId"], event_time(e), e["eventId"]))
    return data["matchDetails"], events


def team_name(cid):
    return HOME_NAME if cid == HOME_ID else AWAY_NAME


def to_m(x, y):
    return x * X_SCALE, y * Y_SCALE


def compute_attack_directions(events):
    """(contestantId, periodId) -> 1 if that team attacks toward higher x
    that period, else -1. Derived from each team's average pass-event x per
    period: a team building mostly in x<50 that period is attacking toward
    x=100 (same heuristic as Disruption/league_disruption_visuals.py's
    compute_attack_directions, scoped to this one match)."""
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
    """Event x/y (Opta 0-100) rotated so THIS event's team always attacks
    toward x=100, y unrotated-consistent -- i.e. "own goal on the left"."""
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
        rows.append({
            "contestantId": e["contestantId"],
            "team": team_name(e["contestantId"]),
            "player": e.get("playerName", "Unknown"),
            "minute": e["timeMin"],
            "period": e["periodId"],
            "x": xm, "y": ym,
            "outcome": outcome,
            "on_target": e["typeId"] in (T_GOAL, T_ATTEMPT_SAVED),
            "is_goal": e["typeId"] == T_GOAL,
            "is_header": is_header,
            "big_chance": has_q(e, Q_BIG_CHANCE),
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
        end_x = end_y = None
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
            "progressive": progressive,
            "final_third_entry": completed and start_dist > 35.0 and end_dist is not None and end_dist <= 35.0,
            "box_entry": (completed and end_x is not None and end_x >= 88.5
                          and 13.84 <= end_y <= 54.16 and not (xm >= 88.5 and 13.84 <= ym <= 54.16)),
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


def build_touches(events, directions):
    """Any ball-involvement event -- used for possession/touch-share and
    field-tilt/thirds proxies (this feed has no official live-possession
    clock, so touch share is the standard stand-in)."""
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


if __name__ == "__main__":
    md, events = load_events()
    directions = compute_attack_directions(events)
    shots = build_shots(events, directions)
    passes = build_passes(events, directions)
    defs = build_defensive_actions(events, directions)
    touches = build_touches(events, directions)
    print("scores", md["scores"])
    print("shots", len(shots), "passes", len(passes), "def actions", len(defs), "touches", len(touches))
    for s in sorted(shots, key=lambda r: r["minute"]):
        if s["is_goal"]:
            print("GOAL", s["minute"], s["team"], s["player"], round(s["xg"], 2))
    home_xg = sum(s["xg"] for s in shots if s["contestantId"] == HOME_ID)
    away_xg = sum(s["xg"] for s in shots if s["contestantId"] == AWAY_ID)
    print("xG", HOME_NAME, round(home_xg, 2), AWAY_NAME, round(away_xg, 2))

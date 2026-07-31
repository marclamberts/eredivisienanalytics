"""
Shared data loading + parsing for Besiktas (Besiktas Jimnastik Kulubu) vs
FC Midtjylland, UEFA Europa League 2026/27, 2nd Qualifying Round -- a
two-legged tie, both legs already played:

  Leg 1: Besiktas 1-0 Midtjylland, 2026-07-23, Tupras Stadyumu, Istanbul
  Leg 2: Midtjylland 0-2 Besiktas, 2026-07-30, MCH Arena, Herning

Besiktas win 3-0 on aggregate and advance to the 3rd Qualifying Round.
This is a team report (Besiktas' performance across the tie), not a
single fixture -- every chart in build_charts.py either shows one leg
at a time or pools Besiktas' own numbers across both legs; Midtjylland
is always "the opponent", never a second protagonist.

Opta MA1 event feed, one file per leg. Unlike this repo's CZ Events files
(matchDetails/event live at the JSON's top level), these feeds nest that
one level down under "liveData" -- see load_match(). Same typeId/
qualifierId conventions as the rest of this repo otherwise (Disruption/
build_disruption_model.py; the CZ post-match report's match_data.py,
whose geometry/xG/PPDA functions are carried over here unchanged).
Qualifiers spot-checked against this tie's own data: 127+169 of the two
legs' passes carry qualifier 1 at long-ball length -> Long ball; exactly
5 and 3 shots (matching the two legs' own headed-attempt counts) carry
qualifier 15 -> Head; both goalscorers (Kokcu leg 1, Rashica + Kokcu leg
2) resolve to Besiktas' own contestantId.

Usage: import this module from the chart scripts in this folder.
"""
import json
import math
import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
EVENTS_DIR = os.path.join(REPO_ROOT, "Europa League Events", "Europa League 2026-2027")

BESIKTAS_ID = "2ez9cvam9lp9jyhng3eh3znb4"
MIDTJYLLAND_ID = "59as3grjvj19voay31j3yfgni"

TEAM_NAMES = {BESIKTAS_ID: "Besiktas", MIDTJYLLAND_ID: "FC Midtjylland"}
TEAM_SHORT = {BESIKTAS_ID: "Besiktas", MIDTJYLLAND_ID: "Midtjylland"}

COMPETITION = "UEFA Europa League 2026/27, 2nd Qualifying Round"
SOURCE = "Opta event data + own xG model"

LEG1 = dict(
    path=os.path.join(EVENTS_DIR, "2026-07-23_Besiktas - Midtjylland.json"),
    leg=1, home_id=BESIKTAS_ID, away_id=MIDTJYLLAND_ID,
    venue="Tupras Stadyumu, Istanbul", date="2026-07-23",
)
LEG2 = dict(
    path=os.path.join(EVENTS_DIR, "2026-07-30_Midtjylland - Besiktas.json"),
    leg=2, home_id=MIDTJYLLAND_ID, away_id=BESIKTAS_ID,
    venue="MCH Arena, Herning", date="2026-07-30",
)

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

PPDA_ZONE_M = 63.0
BOX_Y = (13.84, 54.16)
ZONE14 = (70.0, 88.5, 27.2, 40.8)
HALF_SPACES = [(52.5, 105.0, 13.6, 27.2), (52.5, 105.0, 40.8, 54.4)]


def qmap(e):
    return {q["qualifierId"]: q.get("value") for q in e.get("qualifier", []) or []}


def has_q(e, qid):
    return any(q["qualifierId"] == qid for q in e.get("qualifier", []) or [])


def event_time(e):
    return e["timeMin"] * 60 + e["timeSec"]


def load_match(leg_info):
    with open(leg_info["path"], encoding="utf-8") as f:
        data = json.load(f)
    ld = data["liveData"]
    events = ld["event"]
    events.sort(key=lambda e: (e["periodId"], event_time(e), e["eventId"]))
    return ld["matchDetails"], events


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


class Leg:
    """One played leg of the tie -- its own events, shots, passes etc, plus
    convenience filters keyed to Besiktas vs "the opponent" rather than
    home/away (since Besiktas is home in leg 1, away in leg 2)."""

    def __init__(self, leg_info):
        self.leg_num = leg_info["leg"]
        self.home_id = leg_info["home_id"]
        self.away_id = leg_info["away_id"]
        self.venue = leg_info["venue"]
        self.date = leg_info["date"]
        self.match_details, self.events = load_match(leg_info)
        self.directions = compute_attack_directions(self.events)
        self.shots = build_shots(self.events, self.directions)
        self.passes = build_passes(self.events, self.directions)
        self.defs = build_defensive_actions(self.events, self.directions)
        self.pressing = build_pressing_actions(self.events, self.directions)
        self.recoveries = build_recoveries(self.events, self.directions)
        self.cards = build_cards(self.events)
        self.touches = build_touches(self.events, self.directions)
        self.duels = build_duels(self.events, self.directions)
        self.subs = build_substitutions(self.events)
        self.turnovers = build_turnovers(self.events, self.directions)
        self.assists = build_shot_assists(self.events, self.directions, self.shots)

    def besiktas(self, rows):
        return [r for r in rows if r["contestantId"] == BESIKTAS_ID]

    def opponent(self, rows):
        return [r for r in rows if r["contestantId"] == MIDTJYLLAND_ID]

    @property
    def besiktas_home(self):
        return self.home_id == BESIKTAS_ID

    @property
    def score_line(self):
        s = self.match_details["scores"]["ft"]
        return f"{team_short(self.home_id)} {s['home']}-{s['away']} {team_short(self.away_id)}"

    @property
    def besiktas_goals(self):
        s = self.match_details["scores"]["ft"]
        return s["home"] if self.besiktas_home else s["away"]

    @property
    def opponent_goals(self):
        s = self.match_details["scores"]["ft"]
        return s["away"] if self.besiktas_home else s["home"]

    def besiktas_xg(self):
        return sum(s["xg"] for s in self.besiktas(self.shots))

    def opponent_xg(self):
        return sum(s["xg"] for s in self.opponent(self.shots))

    def besiktas_ppda(self):
        return compute_ppda(self.passes, self.pressing, BESIKTAS_ID, MIDTJYLLAND_ID)

    def opponent_ppda(self):
        return compute_ppda(self.passes, self.pressing, MIDTJYLLAND_ID, BESIKTAS_ID)

    def touch_share(self):
        h = len(self.besiktas(self.touches))
        a = len(self.opponent(self.touches))
        return h / (h + a) if (h + a) else float("nan")


if __name__ == "__main__":
    leg1 = Leg(LEG1)
    leg2 = Leg(LEG2)
    for leg in (leg1, leg2):
        print(f"Leg {leg.leg_num}: {leg.score_line}  ({leg.venue}, {leg.date})")
        print("  Besiktas xG/opponent xG:", round(leg.besiktas_xg(), 2), round(leg.opponent_xg(), 2))
        print("  Besiktas PPDA:", round(leg.besiktas_ppda(), 2), "Opponent PPDA:", round(leg.opponent_ppda(), 2))
        print("  Besiktas touch share:", round(leg.touch_share(), 3))
    agg_h = leg1.besiktas_goals + leg2.besiktas_goals
    agg_a = leg1.opponent_goals + leg2.opponent_goals
    print(f"Aggregate: Besiktas {agg_h}-{agg_a} Midtjylland")

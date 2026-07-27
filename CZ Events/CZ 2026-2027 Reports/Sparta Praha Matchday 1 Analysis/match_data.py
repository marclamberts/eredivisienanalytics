"""
Shared data loading + parsing for the AC Sparta Praha matchday-1 analysis
(Chance liga, CZ 2026-2027, 2026-07-25, Zbrojovka Brno 3-1 Sparta Praha).

Rebuilds the page set of a season-long "SPL" template the user supplied
(yellow/green scheme, full CZ 2025/26 season + all-16-team percentiles)
in Marc Lamberts' Meridian house style (dark mode), but scoped to a single
round of data: this repo only has matchday-1 event feeds for the CZ
2026-2027 season (7 of the round's 8 fixtures, 14 of 16 teams). Every
"league ranking" style page in the source template is rebuilt here from
that 14-team, single-match sample -- explicitly labelled "Matchday 1"
rather than "season", never dressed up as more data than it is.

Opta MA3 event feed, same typeId/qualifierId conventions as the rest of
this repo. New metric definitions not already used elsewhere in this repo
are sourced from existing modules rather than invented fresh:
  - Goal kick: typeId 1 pass carrying qualifierId 124 (Goal Kick Model/
    build_goalkick_shot_model.py).
  - Cutback: typeId 1 pass carrying qualifierId 195 (Box Entry Models/
    build_box_entry_model.py's PULL_BACK_QID). Rare at this sample size
    (0-2 per match leaguewide in MW1) -- Sparta's own match has zero, so
    that page is swapped for a box-entry map instead (see build_charts.py).
  - Switch of play: open-play pass (typeId 1, no set-piece qualifier
    5/6/107) that starts in one wide channel (y <= 100/3 or >= 200/3) and
    ends in the opposite one, real pass distance >= 30m (PSV Season
    Report/Scripts/switches.py).
  - Zone 14 / half-spaces: same rectangles as the sibling post-match
    report's build_charts.py (ZONE14, HALF_SPACES below).
Formation codes (typeId 34 qualifier 130) are NOT decoded into names --
there is no verified lookup table for this feed's numeric formation IDs
in this repo, and guessing one would be worse than omitting it. The
"shape" page instead plots real average on-pitch positions.

Usage: import this module from the chart scripts in this folder.
"""
import json
import math
import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
EVENTS_DIR = os.path.join(REPO_ROOT, "CZ Events", "CZ 2026-2027")

SPARTA_ID = "5ocdn3a6s75u0d0dy0rbou0xc"
BRNO_ID = "6k350zwynsc23f0sxw9akgc6y"

COMPETITION = "Chance Liga 2026/27, Matchday 1"
VENUE = "Stadion Za Lužánkami, Brno"
MATCH_DATE = "2026-07-25"
SOURCE = "Opta event data (CZ 2026-27, matchday 1, 14 of 16 teams) + own xG model"

# All matchday-1 fixtures this repo has an event feed for -- (file, home_id,
# home_name, away_id, away_name). Doubles as the "league sample" for every
# ranking page: 14 teams, one match each, explicitly not a season.
MATCHES = [
    dict(path="2026-07-25_FC Viktoria Plzeň - FC Slovan Liberec.json",
         home_id="c6fx1460nlkawjgh67sp7a1hd", home_name="Viktoria Plzeň",
         away_id="2c4rs2vp0tyjiqa7y3gfttf24", away_name="Slovan Liberec"),
    dict(path="2026-07-25_FC Zbrojovka Brno - AC Sparta Praha.json",
         home_id=BRNO_ID, home_name="Zbrojovka Brno",
         away_id=SPARTA_ID, away_name="Sparta Praha"),
    dict(path="2026-07-25_FC Zlín - FC Baník Ostrava.json",
         home_id="aj1nbeiatqrs6e47mnhjidn15", home_name="Zlín",
         away_id="dfvvrv84skv23rsn1k6kt4slc", away_name="Baník Ostrava"),
    dict(path="2026-07-25_FK Teplice - Bohemians Praha 1905.json",
         home_id="41eivtin75c5fu33x3zfx956b", home_name="Teplice",
         away_id="bqcrqg0367eqzrt4vjb5apu6g", away_name="Bohemians 1905"),
    dict(path="2026-07-26_FC Hradec Králové - FK Pardubice.json",
         home_id="1v75g4bk8vzrvu0jmaro6lila", home_name="Hradec Králové",
         away_id="4xbgquadoen1b303u4hi9nhg9", away_name="Pardubice"),
    dict(path="2026-07-26_FK Jablonec - SK Sigma Olomouc.json",
         home_id="bdz8tx20ekj1ryi2e2u13jdl", home_name="Jablonec",
         away_id="dchxm00ei80l8ljbcfpdill8k", away_name="Sigma Olomouc"),
    dict(path="2026-07-26_SK Slavia Praha - 1. FC Slovácko.json",
         home_id="8kpapuorr6hf0vosnovbreqqd", home_name="Slavia Praha",
         away_id="bp5x8iw8pstucx6s4iqht6xqf", away_name="Slovácko"),
]

TEAM_NAMES = {}
TEAM_MATCH_FILE = {}
for m in MATCHES:
    TEAM_NAMES[m["home_id"]] = m["home_name"]
    TEAM_NAMES[m["away_id"]] = m["away_name"]
    TEAM_MATCH_FILE[m["home_id"]] = m
    TEAM_MATCH_FILE[m["away_id"]] = m

X_SCALE, Y_SCALE = 1.05, 0.68     # Opta 0-100 units -> metres (105 x 68 pitch)
GOAL_X = 105.0
GOAL_Y = 34.0
GOAL_WIDTH = 7.32
PITCH_X, PITCH_Y = 105.0, 68.0

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
Q_CROSS, Q_THROUGH, Q_FREE_KICK, Q_CORNER, Q_THROW_IN = 2, 3, 5, 6, 107
Q_HEAD = 15
Q_RIGHT_FOOT, Q_LEFT_FOOT = 20, 72
Q_END_X, Q_END_Y = 140, 141
Q_ZONE = 56
Q_REGULAR_PLAY, Q_FAST_BREAK, Q_SET_PIECE, Q_FROM_CORNER = 22, 23, 24, 25
Q_BIG_CHANCE = 80
Q_YELLOW_CARD, Q_SECOND_YELLOW, Q_RED_CARD = 31, 32, 33
Q_GOAL_KICK = 124
Q_CUTBACK = 195

SET_PIECE_QIDS = {Q_FREE_KICK, Q_CORNER, Q_THROW_IN}
WIDE_THIRD = 100 / 3
SWITCH_MIN_DIST_M = 30

PPDA_ZONE_M = 63.0

ZONE14 = (70.0, 88.5, 27.2, 40.8)              # x0, x1, y0, y1
HALF_SPACES = [(52.5, 105.0, 13.6, 27.2), (52.5, 105.0, 40.8, 54.4)]
BOX_Y = (13.84, 54.16)


def qmap(e):
    return {q["qualifierId"]: q.get("value") for q in e.get("qualifier", []) or []}


def has_q(e, qid):
    return any(q["qualifierId"] == qid for q in e.get("qualifier", []) or [])


def event_time(e):
    return e["timeMin"] * 60 + e["timeSec"]


def load_match(filename):
    with open(os.path.join(EVENTS_DIR, filename), encoding="utf-8") as f:
        data = json.load(f)
    events = data["event"]
    events.sort(key=lambda e: (e["periodId"], event_time(e), e["eventId"]))
    return data["matchDetails"], events


def team_name(cid):
    return TEAM_NAMES.get(cid, cid)


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
            "contestantId": e["contestantId"], "team": team_name(e["contestantId"]),
            "player": e.get("playerName", "Unknown"), "eventId": e["eventId"],
            "minute": e["timeMin"], "period": e["periodId"], "x": xm, "y": ym,
            "outcome": outcome, "on_target": e["typeId"] in (T_GOAL, T_ATTEMPT_SAVED),
            "is_goal": e["typeId"] == T_GOAL, "is_header": is_header,
            "big_chance": has_q(e, Q_BIG_CHANCE), "situation": situation, "xg": xg,
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
            "contestantId": e["contestantId"], "team": team_name(e["contestantId"]),
            "player": e.get("playerName", "Unknown"), "playerId": e.get("playerId"),
            "minute": e["timeMin"], "second": e["timeSec"], "period": e["periodId"],
            "eventId": e["eventId"], "x": xm, "y": ym, "end_x": end_x, "end_y": end_y,
            "completed": completed, "is_cross": has_q(e, Q_CROSS), "is_corner": has_q(e, Q_CORNER),
            "is_long_ball": has_q(e, Q_LONG_BALL), "is_cutback": has_q(e, Q_CUTBACK),
            "is_goal_kick": has_q(e, Q_GOAL_KICK), "is_switch": is_switch and completed,
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
            "contestantId": e["contestantId"], "team": team_name(e["contestantId"]),
            "player": e.get("playerName", "Unknown"), "minute": e["timeMin"], "period": e["periodId"],
            "x": xm, "y": ym, "action": DEFENSIVE_TYPES[e["typeId"]], "success": e.get("outcome", 1) == 1,
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
            "contestantId": e["contestantId"], "team": team_name(e["contestantId"]),
            "player": e.get("playerName", "Unknown"), "minute": e["timeMin"], "period": e["periodId"],
            "x": xm, "y": ym, "action": action,
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
            "contestantId": e["contestantId"], "team": team_name(e["contestantId"]),
            "minute": e["timeMin"], "second": e["timeSec"], "period": e["periodId"],
            "x": xm, "y": ym, "typeId": e["typeId"],
        })
    return rows


def build_cards(events):
    rows = []
    for e in events:
        if e["typeId"] != T_CARD:
            continue
        kind = "Red" if has_q(e, Q_RED_CARD) else ("2nd Yellow" if has_q(e, Q_SECOND_YELLOW) else "Yellow")
        rows.append({"contestantId": e["contestantId"], "team": team_name(e["contestantId"]),
                      "player": e.get("playerName", "Unknown"), "minute": e["timeMin"], "kind": kind})
    return rows


def build_substitutions(events):
    rows = []
    for e in events:
        if e["typeId"] != T_SUB_OFF:
            continue
        rows.append({"contestantId": e["contestantId"], "team": team_name(e["contestantId"]),
                      "player_off": e.get("playerName", "Unknown"), "minute": e["timeMin"]})
    return rows


def build_recoveries(events, directions):
    rows = []
    for e in events:
        if e["typeId"] != T_BALL_RECOVERY or e.get("x") is None:
            continue
        x, y = norm_xy(e, directions)
        xm, ym = to_m(x, y)
        rows.append({"contestantId": e["contestantId"], "team": team_name(e["contestantId"]),
                      "player": e.get("playerName", "Unknown"), "minute": e["timeMin"], "period": e["periodId"],
                      "x": xm, "y": ym})
    return rows


def build_duels(events, directions):
    """Tackle + Aerial + Challenge, each contestantId's own outcome."""
    rows = []
    for e in events:
        if e.get("x") is None or not e.get("contestantId"):
            continue
        if e["typeId"] not in (T_TACKLE, T_AERIAL, T_CHALLENGE):
            continue
        x, y = norm_xy(e, directions)
        xm, ym = to_m(x, y)
        rows.append({"contestantId": e["contestantId"], "team": team_name(e["contestantId"]),
                      "player": e.get("playerName", "Unknown"), "minute": e["timeMin"], "period": e["periodId"],
                      "x": xm, "y": ym,
                      "action": {T_TACKLE: "Tackle", T_AERIAL: "Aerial", T_CHALLENGE: "Challenge"}[e["typeId"]],
                      "success": e.get("outcome", 1) == 1})
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


def simulate_match_scorelines(home_shots, away_shots, n=20000, seed=42, cap=6):
    """Monte Carlo simulation from THIS match's own shots (not the league
    sample): each shot converts independently with probability = its own
    xG. Same mechanic as the sibling post-match reports' simulate_scorelines."""
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
    return {"score_counts": score_counts, "n": n, "cap": cap,
            "home_win": home_win, "draw": draw, "away_win": away_win}


class TeamMatch:
    """Everything derived from ONE team's matchday-1 event feed: shots,
    passes, defensive/pressing actions, touches, cards, indexable by that
    team's own contestantId or its MW1 opponent's (numbers conceded)."""

    def __init__(self, team_id):
        info = TEAM_MATCH_FILE[team_id]
        self.team_id = team_id
        self.team_name = team_name(team_id)
        self.opponent_id = info["away_id"] if team_id == info["home_id"] else info["home_id"]
        self.opponent_name = team_name(self.opponent_id)
        self.is_home = team_id == info["home_id"]
        self.match_details, self.events = load_match(info["path"])
        self.directions = compute_attack_directions(self.events)
        self.shots = build_shots(self.events, self.directions)
        self.passes = build_passes(self.events, self.directions)
        self.defs = build_defensive_actions(self.events, self.directions)
        self.pressing = build_pressing_actions(self.events, self.directions)
        self.cards = build_cards(self.events)
        self.touches = build_touches(self.events, self.directions)
        self.subs = build_substitutions(self.events)
        self.recoveries = build_recoveries(self.events, self.directions)
        self.duels = build_duels(self.events, self.directions)
        self.turnovers = build_turnovers(self.events, self.directions)
        self.assists = build_shot_assists(self.events, self.directions, self.shots)

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

    def verticality(self):
        """Avg forward (toward opponent goal) distance per completed forward pass, metres."""
        fwd = [p["end_x"] - p["x"] for p in self.own(self.passes)
               if p["completed"] and p["end_x"] is not None and p["end_x"] > p["x"]]
        return sum(fwd) / len(fwd) if fwd else 0.0

    def def_line_height(self):
        """Avg x of this team's defensive actions while defending -- estimated
        line height, own goal at x=0."""
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
        distance divided by the time gap to the next event in the match
        (any team), used as a proxy for how quickly the pass was played.
        Gaps > 8s (stoppages, fouls, VAR checks) are excluded as noise,
        not genuine tempo."""
        ordered = sorted(self.events, key=lambda e: (e["periodId"], event_time(e), e["eventId"]))
        speeds = []
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
            x, y = norm_xy(e, self.directions)
            xm, ym = to_m(x, y)
            d = directions_val = self.directions.get((e["contestantId"], e["periodId"]), 1)
            ex, ey = float(q[Q_END_X]), float(q[Q_END_Y])
            if d == -1:
                ex, ey = 100.0 - ex, 100.0 - ey
            exm, eym = to_m(ex, ey)
            dist = math.hypot(exm - xm, eym - ym)
            speeds.append(dist / dt)
        return sum(speeds) / len(speeds) if speeds else 0.0

    def pass_volume(self):
        return len(self.own(self.passes))


class LeagueMW1:
    """All 14 teams with a matchday-1 feed in this repo, one TeamMatch each.
    The "league ranking" pages' comparison set -- a single round, not a
    season, and every chart built from it says so."""

    def __init__(self):
        self.teams = {tid: TeamMatch(tid) for tid in TEAM_NAMES}

    def ranking(self, metric_fn, reverse=True):
        vals = [(tid, metric_fn(tm)) for tid, tm in self.teams.items()]
        vals = [(tid, v) for tid, v in vals if v == v]  # drop NaN
        return sorted(vals, key=lambda kv: kv[1], reverse=reverse)


if __name__ == "__main__":
    sparta = TeamMatch(SPARTA_ID)
    print(sparta.team_name, "MW1 vs", sparta.opponent_name, "(home)" if sparta.is_home else "(away)")
    print("  xG for/against:", round(sparta.xg_for, 2), round(sparta.xg_against, 2))
    print("  Verticality:", round(sparta.verticality(), 1), "m/pass")
    print("  Def line height:", round(sparta.def_line_height(), 1), "m")
    print("  Touch share:", round(sparta.touch_share(), 2), "Field tilt:", round(sparta.field_tilt(), 2))
    hs, z14 = sparta.halfspace_zone14_counts()
    print("  Halfspace passes:", hs, "Zone14 passes:", z14)
    switches = sum(1 for p in sparta.own(sparta.passes) if p["is_switch"])
    print("  Switches of play:", switches)
    print("  Goal kicks:", sum(1 for p in sparta.own(sparta.passes) if p["is_goal_kick"]))
    print("  Cutbacks:", sum(1 for p in sparta.own(sparta.passes) if p["is_cutback"]))

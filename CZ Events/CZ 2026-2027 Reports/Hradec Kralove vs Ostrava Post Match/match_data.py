"""
Shared data loading + parsing for the FC Hradec Králové vs FC Baník
Ostrava post-match report (Chance liga, CZ 2026-2027, matchday 3).

Opta MA3 event feed -- same typeId/qualifierId conventions as the rest of
this repo (see Disruption/build_disruption_model.py, Goal Kick Model/
build_goalkick_shot_model.py). No packaged xG feed for the Czech league
data, so shots are scored with a small distance+angle geometric model
(own model, not provider-supplied -- flagged as such in every chart's
source line, same convention as the sibling Hradec Kralove vs Pardubice
template this report follows). This own model is a known simplification
of a real provider xG model (checked against Opta's own published
figures for a different Hradec match: Opta had 0.99/2.33, this model
produced a more extreme 0.34/3.60 for the same shots) -- treat this
report's xG numbers as a consistent, comparable-within-this-repo proxy,
not a substitute for a calibrated provider model.

Team IDs cross-checked against the goal-scorer contestantId counts and
the 2-1 final score in matchDetails/scores/ft (Hradec won at home).
Both team IDs match the constants already used in the sibling "Hradec
Kralove vs Ostrava" pre-match preview for this same fixture.

Usage: import this module from the chart scripts in this folder.
"""
import json
import math
import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
DATA_PATH = os.path.join(
    REPO_ROOT, "CZ Events", "CZ 2026-2027", "2026-08-09_FC Hradec Králové - FC Baník Ostrava.json",
)

HOME_ID = "1v75g4bk8vzrvu0jmaro6lila"
AWAY_ID = "dfvvrv84skv23rsn1k6kt4slc"
HOME_NAME = "FC Hradec Králové"
AWAY_NAME = "FC Baník Ostrava"
COMPETITION = "Chance Liga 2026/27, Matchday 3"
VENUE = "FINEP Arena, Hradec Králové"
MATCH_DATE = "2026-08-09"
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
T_CARD = 17
T_SUB_OFF, T_SUB_ON = 18, 19
T_CHALLENGE = 45
T_AERIAL = 44
T_BALL_RECOVERY, T_DISPOSSESSED = 49, 50
T_BLOCKED_PASS = 74

SHOT_TYPES = {T_MISS, T_POST, T_ATTEMPT_SAVED, T_GOAL}
DEFENSIVE_TYPES = {T_TACKLE: "Tackle", T_INTERCEPTION: "Interception", T_CLEARANCE: "Clearance"}
# Standard PPDA defensive-action set (tackles + interceptions + fouls
# committed + challenges) -- deliberately excludes clearances and aerials,
# which most public PPDA definitions leave out.
PRESSING_TYPES = {T_TACKLE: "Tackle", T_INTERCEPTION: "Interception", T_CHALLENGE: "Challenge"}


# Verified empirically against this match's own data (see build_charts.py
# session notes) rather than trusted from the Disruption module's constants,
# which mislabel Q_HEAD as qualifier 1:
#   - qualifier 1: 0/21 shots carry it, but 143/841 passes do, at ~3x the
#     average length of untagged passes (44.2m vs 15.2m) -> Long ball.
#   - qualifier 15: present on exactly the match's 2 headed shot attempts,
#     including F. Čech's headed goal from a corner -> Head.
#   - qualifier 20 / 72: right-footed / left-footed (18 and 1 of 21 shots).
Q_LONG_BALL = 1
Q_CROSS, Q_THROUGH, Q_FREE_KICK, Q_CORNER = 2, 3, 5, 6
Q_HEAD = 15
Q_RIGHT_FOOT, Q_LEFT_FOOT = 20, 72
Q_END_X, Q_END_Y = 140, 141
Q_ZONE = 56
Q_REGULAR_PLAY, Q_FAST_BREAK, Q_SET_PIECE, Q_FROM_CORNER = 22, 23, 24, 25
Q_BIG_CHANCE = 80
Q_YELLOW_CARD, Q_SECOND_YELLOW, Q_RED_CARD = 31, 32, 33
# Shot->Save/Block link, NOT an assist link (verified: e.g. shot eventId 25's
# qualifier-233 target eventId 47 resolves to the goalkeeper's Save record,
# not a preceding pass) -- do not use this for shot-assist detection.
Q_RELATED_EVENT = 233

# Standard PPDA zone cutoff: only count opposition passes/pressing actions
# in the defending-from-possession team's own 60% of the pitch (105m pitch
# -> 63m from their own goal); teams rarely press inside the opponent's
# attacking-most 40%, so that band is excluded, same convention widely used
# for Opta/Wyscout-derived PPDA (understat glossary etc.).
PPDA_ZONE_M = 63.0


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


# ---------------------------------------------------------------------------
# Threat surface for "xT flow" -- NOT the repo's pre-trained Eredivisie xT
# grid (xT/xt_grid_values.csv). That grid was checked before use here: it is
# essentially flat (~0.12) across the whole pitch, own box included, with a
# small bump only in the last two columns near goal -- not a usable spatial
# gradient, so using it would dress up noise as insight. Instead this reuses
# the same, already-verified distance+angle geometry as shot_xg: the value
# of a location is "the xG of a shot taken from here" (is_header=False),
# smooth and monotonic own-goal-to-opponent-goal. This is a threat PROXY,
# not a possession-value model (no pass-completion or continuation term) --
# labelled as such on every chart that uses it.
# ---------------------------------------------------------------------------

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
    """Tackle + Interception + Challenge (all outcome-agnostic) plus fouls
    actually committed by this contestantId -- the PPDA denominator's
    action set. Foul events come in contestantId pairs sharing an eventId;
    verified against this match's four cards (each carded player's own
    Foul record has outcome==0, the opponent's outcome==1), so outcome==0
    is "this team committed the foul", not outcome==1."""
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
    """PPDA for `contestant_id` pressing `opponent_id`: opponent PASS
    ATTEMPTS (not just completions -- a pass broken up by pressure still
    counts) in the opponent's own 60% of the pitch, divided by
    contestant_id's tackles+interceptions+challenges+fouls-committed in
    that same physical zone. lo/hi optionally restrict to a minute window."""
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


def build_substitutions(events):
    rows = []
    for e in events:
        if e["typeId"] != T_SUB_OFF:
            continue
        rows.append({
            "contestantId": e["contestantId"],
            "team": team_name(e["contestantId"]),
            "player_off": e.get("playerName", "Unknown"),
            "minute": e["timeMin"],
        })
    return rows


def build_shot_assists(events, directions, shots):
    """For each shot, the most recent completed pass by the shooting team
    since the ball last changed teams -- i.e. "who set this shot up", not
    the shot's own qualifier-233 "related event", which links to its
    Save/Block counterpart, not an assist (verified: shot eventId 25's
    qualifier-233 target, eventId 47, resolves to the goalkeeper's Save
    record). A shot straight off a loose-ball duel or the shooter's own
    take-on with no intervening teammate pass gets no assist credited."""
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
            "contestantId": e["contestantId"],
            "team": team_name(e["contestantId"]),
            "shooter": e.get("playerName", "Unknown"),
            "assister": ap.get("playerName", "Unknown"),
            "minute": e["timeMin"],
            "x": xm, "y": ym, "end_x": end_x if end_x is not None else sxm,
            "end_y": end_y if end_y is not None else sym,
            "shot_x": sxm, "shot_y": sym,
            "shot_xg": xg_by_eventid.get(e["eventId"], 0.0),
            "is_goal": e["typeId"] == T_GOAL,
        })
    return rows


def build_turnovers(events, directions):
    """A team's own failed pass or Dispossessed event -- i.e. the moment
    they lost the ball -- located in their own attacking half. Losing it
    that far forward is what "dangerous area" means here: it hands the
    opponent the ball already deep in their own attacking third."""
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
        rows.append({
            "contestantId": e["contestantId"],
            "team": team_name(e["contestantId"]),
            "player": e.get("playerName", "Unknown"),
            "minute": e["timeMin"], "period": e["periodId"],
            "x": xm, "y": ym,
            "kind": "Failed pass" if is_failed_pass else "Dispossessed",
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


def simulate_scorelines(shots, n=20000, seed=42, cap=6):
    """Monte Carlo match simulation: each shot converts independently with
    probability = its own xG (Bernoulli), summed per team per draw. Returns
    (score_counts, home_goal_counts, away_goal_counts) where score_counts
    is a dict {(home_goals, away_goals): count} with goals capped at `cap`
    for a bounded scoreline grid (any game with more goals than that from
    a side folds into the `cap` row/column)."""
    import random
    rng = random.Random(seed)
    home_xgs = [s["xg"] for s in shots if s["contestantId"] == HOME_ID]
    away_xgs = [s["xg"] for s in shots if s["contestantId"] == AWAY_ID]

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

    pressing = build_pressing_actions(events, directions)
    home_ppda = compute_ppda(passes, pressing, HOME_ID, AWAY_ID)
    away_ppda = compute_ppda(passes, pressing, AWAY_ID, HOME_ID)
    print("PPDA", HOME_NAME, round(home_ppda, 2), AWAY_NAME, round(away_ppda, 2))
    print("pressing actions", len(pressing), "recoveries", len(build_recoveries(events, directions)),
          "cards", len(build_cards(events)), "subs", len(build_substitutions(events)))

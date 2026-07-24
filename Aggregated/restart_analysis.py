"""
Goal kicks and open-play long balls: usage, first-contact, second-ball,
possession-establishment, territorial, pressure-escape, territorial-dominance,
territorial-lock, progression/chance-creation, and risk metrics, computed
separately for the two restart types, across 2023-2024/2024-2025/2025-2026.

SCOPE -- read before trusting a number (also in Aggregated/restart_analysis/README.md):

Qualifier 124 = Goal Kick, verified empirically against this repo's own
Events data (100% of occurrences sit within 8m of a goal, central y, mixed
length/outcome/long-ball-qualifier co-occurrence -- exactly what a goal
kick should look like), NOT assumed from any existing script.

This feed has no tracking data -- only ball-involving events. That rules
out, honestly, rather than approximated:
  - Press Bypass Value (PBV): needs every opponent's x-position, not just
    the ones who touched the ball. NOT computed.
  - Possession Value Added (PVA) / Net Possession Value (NPV): needs a full
    possession-value model (a much larger build than this task, and this
    repo doesn't have one). NOT computed.
  - "Uncontested" reception (for excluding from First-Contact Win Rate):
    approximated only as "didn't go straight out of play" -- a genuinely
    uncontested vs. quietly-contested first touch can't be told apart
    without tracking.
  - "Pressure" at the moment of restart: approximated as an opponent
    defensive-type event (tackle/interception/challenge/aerial/foul) within
    5 seconds and 15m of the passer beforehand. Only applied to open-play
    long balls -- a goal kick is inherently unpressured in the on-ball
    sense, so goal-kick pressure-escape metrics are not computed.
  - xG/xT/PV-dependent metrics (xGPR20, xGCA20, NCV20, xTPR15): only
    computable for 2025-2026, the one season with a trained xG model
    (Danger/) and an xT grid (xT/) in this repo. Left blank for the other
    two seasons rather than guessed.

Everything else is computed directly from Events/<season> for all three
seasons. RDV's weights are exactly as specified in the brief -- not
validated here, per the brief's own caution that they should be tested,
not treated as correct.

Usage: python3 restart_analysis.py
"""
import csv
import glob
import json
import math
import os
import statistics
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEASONS = ["2023-2024", "2024-2025", "2025-2026"]
OUT_DIR = os.path.join(ROOT, "Aggregated", "restart_analysis")
XG_SEASON = "2025-2026"

X_SCALE, Y_SCALE = 1.05, 0.68
GOAL_X, GOAL_Y = 105.0, 34.0
MID_X, FINAL_THIRD_X = 200.0 / 3.0, 200.0 / 3.0
BOX_X_MIN, BOX_Y_MIN, BOX_Y_MAX = 83.0, 21.1, 78.9
LONG_GOAL_KICK_M = 40.0
WINDOW_MAX_S = 60.0

Q_GOAL_KICK = 124
Q_LONG_BALL, Q_CROSS, Q_THROUGH_BALL, Q_FREE_KICK, Q_CORNER, Q_THROW_IN = 1, 2, 3, 5, 6, 107
Q_END_X, Q_END_Y = 140, 141

BALL_EVENT_TYPES = {1, 3, 7, 8, 12, 13, 14, 15, 16, 44, 49, 50, 61}
SHOT_TYPES = {13, 14, 15, 16}
DEFENSIVE_TYPES = {7, 8, 12, 44, 45, 49}  # tackle/interception/clearance/aerial/challenge/recovery


def qmap(e):
    return {q["qualifierId"]: q.get("value") for q in e.get("qualifier", []) or []}


def num(v, default=0.0):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def event_time(e):
    return e["timeMin"] * 60 + e["timeSec"]


def dist_m(x1, y1, x2, y2):
    dx, dy = (x1 - x2) * X_SCALE, (y1 - y2) * Y_SCALE
    return (dx * dx + dy * dy) ** 0.5


def dist_to_goal_m(x, y, goal_x):
    xm, ym = x * X_SCALE, y * Y_SCALE
    gx = goal_x * X_SCALE
    return ((gx - xm) ** 2 + (GOAL_Y - ym) ** 2) ** 0.5


def in_box(x, y):
    return x >= BOX_X_MIN and BOX_Y_MIN <= y <= BOX_Y_MAX


def mean(xs):
    xs = [x for x in xs if x is not None]
    return statistics.fmean(xs) if xs else None


def rate(numer, denom):
    return numer / denom if denom else None


class Restart:
    __slots__ = ("kind", "team", "opp", "t0", "idx", "ox", "oy", "ex", "ey",
                 "is_long", "under_pressure", "period")

    def __init__(self, kind, team, opp, t0, idx, ox, oy, ex, ey, is_long, under_pressure, period):
        self.kind = kind  # "goal_kick" | "long_ball"
        self.team = team
        self.opp = opp
        self.t0 = t0
        self.idx = idx
        self.ox, self.oy = ox, oy
        self.ex, self.ey = ex, ey
        self.is_long = is_long
        self.under_pressure = under_pressure
        self.period = period


def team_directions(events, team_of_cid):
    """{(team, period): 1 or -1}, 1 = attacks toward higher x, same heuristic
    used elsewhere in Aggregated/ (avg x of a team's own open-play passes)."""
    buckets = {}
    for e in events:
        if e.get("typeId") != 1 or not e.get("contestantId"):
            continue
        q = qmap(e)
        if q.keys() & {Q_FREE_KICK, Q_CORNER, Q_THROW_IN, Q_GOAL_KICK}:
            continue
        period = e.get("periodId")
        if period not in (1, 2):
            continue
        team = team_of_cid.get(e["contestantId"])
        if team is None:
            continue
        buckets.setdefault((team, period), []).append(e["x"])
    return {k: (1 if statistics.fmean(v) < 50 else -1) for k, v in buckets.items()}


def norm_x(x, team, period, directions):
    d = directions.get((team, period), 1)
    return x if d == 1 else 100 - x


def find_restarts(events, team_of_cid, opp_of_team, directions):
    restarts = []
    for idx, e in enumerate(events):
        if e.get("typeId") != 1 or not e.get("contestantId"):
            continue
        q = qmap(e)
        team = team_of_cid.get(e["contestantId"])
        if team is None:
            continue
        opp = opp_of_team.get(team)
        period = e.get("periodId")
        if period not in (1, 2):
            continue
        ox, oy = e["x"], e["y"]
        ex, ey = (num(q[Q_END_X]), num(q.get(Q_END_Y))) if Q_END_X in q else (ox, oy)
        t0 = event_time(e)

        if Q_GOAL_KICK in q:
            own_goal_x = 0.0 if ox < 50 else 100.0
            is_long = dist_to_goal_m(ex, ey, own_goal_x) >= LONG_GOAL_KICK_M
            restarts.append(Restart("goal_kick", team, opp, t0, idx, ox, oy, ex, ey,
                                     is_long, False, period))
        elif Q_LONG_BALL in q and not (q.keys() & {Q_FREE_KICK, Q_CORNER, Q_THROW_IN, Q_GOAL_KICK}):
            under_pressure = False
            for e2 in reversed(events[max(0, idx - 12):idx]):
                if t0 - event_time(e2) > 5:
                    break
                if e2.get("typeId") in DEFENSIVE_TYPES and e2.get("contestantId") and \
                        team_of_cid.get(e2["contestantId"]) != team and \
                        dist_m(e2.get("x", ox), e2.get("y", oy), ox, oy) <= 15:
                    under_pressure = True
                    break
            restarts.append(Restart("long_ball", team, opp, t0, idx, ox, oy, ex, ey,
                                     True, under_pressure, period))
    return restarts


def walk_forward(events, start_idx, t0, window=WINDOW_MAX_S):
    """Ball-involving events (any team) after a restart, within `window` seconds,
    as (elapsed_seconds, contestantId, event)."""
    out = []
    for e in events[start_idx + 1:]:
        te = event_time(e)
        elapsed = te - t0
        if elapsed > window:
            break
        if e.get("typeId") in BALL_EVENT_TYPES and e.get("contestantId"):
            out.append((elapsed, e.get("contestantId"), e))
    return out


PRE_POST_WINDOW_S = 45.0  # brief allows 30 or 60; splitting the difference


def walk_backward(events, start_idx, t0, window=PRE_POST_WINDOW_S):
    """Same as walk_forward but looking before the restart -- elapsed is
    seconds *before* t0 (positive number, event earlier in time)."""
    out = []
    for e in reversed(events[max(0, start_idx - 200):start_idx]):
        te = event_time(e)
        elapsed = t0 - te
        if elapsed > window:
            break
        if elapsed < 0:
            continue
        if e.get("typeId") in BALL_EVENT_TYPES and e.get("contestantId"):
            out.append((elapsed, e.get("contestantId"), e))
    return out


def field_tilt_and_tps(pairs, r, team_of_cid, directions):
    """Share of final-third touches, and share of attacking-half touches
    (both measured in the RESTART TEAM's attacking direction, applied to
    both sides), that belong to the restart team vs. restart+opponent
    combined, over a set of (elapsed, contestantId, event) tuples."""
    ft_team = ft_opp = half_team = half_opp = 0
    for _, cid, e in pairs:
        t = team_of_cid.get(cid)
        if t not in (r.team, r.opp):
            continue
        x = norm_x(e.get("x", 50), r.team, r.period, directions)
        is_ft = x >= FINAL_THIRD_X
        is_half = x >= 50
        if t == r.team:
            ft_team += is_ft
            half_team += is_half
        else:
            ft_opp += is_ft
            half_opp += is_half
    ft = rate(ft_team, ft_team + ft_opp)
    tps = rate(half_team, half_team + half_opp)
    return ft, tps, ft_team, ft_opp


def load_xt_grid():
    """16x12 xT grid (xT/xt_grid_values.csv, 2025-2026 only) -> {(bin_x,bin_y): xT}."""
    path = os.path.join(ROOT, "xT", "xt_grid_values.csv")
    if not os.path.exists(path):
        return None
    grid = {}
    x_bins, y_bins = 16, 12
    with open(path, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            bx = min(x_bins - 1, int(float(row["zone_x"]) / (100.0 / x_bins)))
            by = min(y_bins - 1, int(float(row["zone_y"]) / (100.0 / y_bins)))
            grid[(bx, by)] = float(row["xT"])
    return grid


def xt_at(grid, x, y):
    if grid is None:
        return None
    bx = min(15, max(0, int(x / (100.0 / 16))))
    by = min(11, max(0, int(y / (100.0 / 12))))
    return grid.get((bx, by))


def analyse_restart(r, seq, pre_seq, team_of_cid, directions, danger_lookup, match_file, xt_grid=None):
    """One restart's full metric inputs, from its forward event sequence."""
    out = {"kind": r.kind, "team": r.team, "is_long": r.is_long, "under_pressure": r.under_pressure}

    # Section 7: territorial dominance (field tilt / TPS pre vs. post, both
    # windows PRE_POST_WINDOW_S long) -- computed regardless of whether the
    # restart itself was "contestable", since this is about broader control.
    if r.opp is not None:
        ft_post, tps_post, ft_team_n, ft_opp_n = field_tilt_and_tps(seq, r, team_of_cid, directions)
        ft_pre, tps_pre, _, _ = field_tilt_and_tps(pre_seq, r, team_of_cid, directions)
        out["ft_post"], out["ft_pre"] = ft_post, ft_pre
        if ft_post is not None and ft_pre is not None:
            out["fts"] = ft_post - ft_pre
        out["tps_post"], out["tps_pre"] = tps_post, tps_pre
        if tps_post is not None and tps_pre is not None:
            out["tds"] = tps_post - tps_pre
        out["ftad"] = ft_team_n - ft_opp_n

        first_ft_team = None
        for elapsed, cid, e in seq:
            if norm_x(e.get("x", 50), r.team, r.period, directions) >= FINAL_THIRD_X:
                first_ft_team = team_of_cid.get(cid)
                break
        if first_ft_team is not None:
            out["nep"] = first_ft_team == r.team

        first_shot_team = None
        for elapsed, cid, e in seq:
            if e.get("typeId") in SHOT_TYPES:
                first_shot_team = team_of_cid.get(cid)
                break
        if first_shot_team is not None:
            out["nsp"] = first_shot_team == r.team

    if not seq:
        out["contestable"] = False
        return out

    first = seq[0]
    fc_elapsed, fc_cid, fc_event = first
    fc_team = team_of_cid.get(fc_cid)
    went_out = fc_event.get("typeId") == 5
    out["contestable"] = not went_out
    if not out["contestable"]:
        return out

    out["first_contact_won"] = fc_team == r.team
    out["first_contact_is_aerial"] = fc_event.get("typeId") == 44
    if out["first_contact_is_aerial"]:
        out["aerial_won"] = fc_event.get("outcome") == 1

    fcx, fcy = fc_event.get("x", r.ex), fc_event.get("y", r.ey)
    out["fc_direction_gain"] = (norm_x(fcx, r.team, r.period, directions) -
                                 norm_x(r.ox, r.team, r.period, directions))

    # productive first contact: does a teammate (of whoever won it) touch the
    # ball within 3s after?
    productive = False
    for elapsed, cid, e in seq[1:]:
        if elapsed - fc_elapsed > 3:
            break
        if team_of_cid.get(cid) == fc_team:
            productive = True
            break
    out["fc_won_and_productive"] = out["first_contact_won"] and productive

    # --- possession-holding walk: classify every subsequent touch as
    # restart-team or opponent, tracking consecutive-same-team streaks
    # (time span + count) to find "established possession" for either side.
    def established_possession_times():
        """First (elapsed, team, x, y) where a side hits 3 controlled actions
        in a row, or 5s of continuous hold, for restart team and opponent."""
        streak_team, streak_start_elapsed, streak_count = None, 0.0, 0
        found = {r.team: None, r.opp: None}
        last_xy = {}
        for elapsed, cid, e in seq:
            team = team_of_cid.get(cid)
            if team is None:
                continue
            last_xy[team] = (e.get("x", r.ex), e.get("y", r.ey))
            if team == streak_team:
                streak_count += 1
            else:
                streak_team, streak_start_elapsed, streak_count = team, elapsed, 1
            duration = elapsed - streak_start_elapsed
            if found.get(streak_team) is None and (streak_count >= 3 or duration >= 5.0):
                x, y = last_xy.get(streak_team, (r.ex, r.ey))
                found[streak_team] = (elapsed, streak_team, x, y)
        return found

    established = established_possession_times()
    restart_established = established.get(r.team)
    opp_established = established.get(r.opp) if r.opp else None
    out["restart_established"] = restart_established
    out["opp_established"] = opp_established

    # counterattack conceded: opponent establishes possession and reaches the
    # final third (in the restart team's own attacking direction) within 15s
    # of establishing it
    if opp_established:
        est_t, _, _, _ = opp_established
        out["counterattack_conceded"] = any(
            team_of_cid.get(cid) == r.opp and elapsed - est_t <= 15 and
            norm_x(e.get("x", r.ex), r.team, r.period, directions) >= FINAL_THIRD_X
            for elapsed, cid, e in seq if elapsed >= est_t)

    # direct retention: first contact won AND a teammate controls within 3s
    out["direct_retention"] = out["fc_won_and_productive"]
    # indirect retention: restart team establishes possession despite losing
    # first contact (won it back after a duel/opponent touch)
    out["indirect_retention"] = (not out["first_contact_won"]) and restart_established is not None

    # second-ball recovery: after an UNCONTROLLED first contact (lost, or won
    # but not productive), does the restart team win the very next touch?
    out["uncontrolled_first_contact"] = not out["fc_won_and_productive"]
    if out["uncontrolled_first_contact"] and len(seq) > 1:
        nxt_team = team_of_cid.get(seq[1][1])
        out["second_ball_recovered"] = nxt_team == r.team
    if not out["first_contact_won"]:
        out["recovery_after_opp_first_contact"] = restart_established is not None

    out["duel_phase_controlled"] = restart_established is not None
    if restart_established:
        out["time_to_second_ball_control"] = restart_established[0] - fc_elapsed

    # territorial: positions at fixed checkpoints (10/15s) and at established possession
    def pos_at(window_s):
        last = None
        for elapsed, cid, e in seq:
            if elapsed > window_s:
                break
            last = (e.get("x", r.ex), e.get("y", r.ey))
        return last

    p10, p15 = pos_at(10), pos_at(15)
    ox_n = norm_x(r.ox, r.team, r.period, directions)
    if p10:
        out["rtg_10"] = norm_x(p10[0], r.team, r.period, directions) - ox_n
    if p15:
        out["rtg_15"] = norm_x(p15[0], r.team, r.period, directions) - ox_n
        if xt_grid is not None:
            xt_start = xt_at(xt_grid, r.ox, r.oy)
            xt_end = xt_at(xt_grid, *p15)
            if xt_start is not None and xt_end is not None:
                out["xt_gain_15"] = xt_end - xt_start

    ctg_ref = restart_established or opp_established
    if ctg_ref:
        out["ctg"] = norm_x(ctg_ref[2], r.team, r.period, directions) - ox_n
    if restart_established:
        out["ntg"] = norm_x(restart_established[2], r.team, r.period, directions) - ox_n
    elif opp_established:
        out["ntg"] = norm_x(opp_established[2], r.team, r.period, directions) - ox_n - 20  # negative-possession penalty
    else:
        out["ntg"] = 0.0

    survives_10 = restart_established is not None
    out["etg_basic"] = (out.get("ctg", 0.0) * (1.0 if survives_10 else 0.0))
    if restart_established and restart_established[0] <= 10:
        s_i = 1.0
    elif out.get("rtg_15", 0) is not None and out.get("rtg_15", -99) > 5 and restart_established is None and opp_established is None:
        s_i = 0.5
    elif opp_established is not None and norm_x(opp_established[2], r.team, r.period, directions) < 40:
        s_i = -1.0
    else:
        s_i = 0.0
    out["etg_event"] = (out.get("ctg", 0.0) or 0.0) * s_i

    reaches_halfway = any(norm_x(e.get("x", r.ex), r.team, r.period, directions) >= 50
                           for _, cid, e in seq if team_of_cid.get(cid) == r.team)
    out["reaches_halfway"] = reaches_halfway
    if reaches_halfway and restart_established:
        out["remains_beyond_halfway_15s"] = restart_established[0] <= 15 and \
            norm_x(restart_established[2], r.team, r.period, directions) >= 50

    if restart_established:
        out["final_third_establishment"] = norm_x(restart_established[2], r.team, r.period, directions) >= FINAL_THIRD_X

    def box_entry_within(window_s):
        for elapsed, cid, e in seq:
            if elapsed > window_s:
                break
            if team_of_cid.get(cid) != r.team:
                continue
            q = qmap(e)
            ex2, ey2 = (num(q[Q_END_X]), num(q.get(Q_END_Y))) if Q_END_X in q else (e.get("x"), e.get("y"))
            if in_box(ex2, ey2):
                return True
        return False

    out["box_entry_20"] = box_entry_within(20)

    for w in (5, 10, 15):
        out[f"possession_established_{w}"] = restart_established is not None and restart_established[0] <= w

    def controlled_actions_after(window_s):
        return sum(1 for elapsed, cid, e in seq if elapsed <= window_s and team_of_cid.get(cid) == r.team)

    out["actions_sustained"] = controlled_actions_after(15)

    if restart_established:
        t_est = restart_established[0]
        for w in (10, 15, 20):
            still_active = any(team_of_cid.get(cid) == r.team for elapsed, cid, e in seq
                                if t_est < elapsed <= t_est + w)
            out[f"survives_{w}"] = still_active

    # pressure escape (long balls only, per scope note). Both criteria are
    # evaluated at the established-possession location itself (not merely
    # "touched halfway at some transient point"), so a sequence classified
    # as an escape always has a non-negative escape distance by construction.
    if r.under_pressure:
        if restart_established:
            established_x = norm_x(restart_established[2], r.team, r.period, directions)
            gained_20 = (established_x - ox_n) >= 20
            beyond_halfway = established_x >= 50
        else:
            gained_20 = beyond_halfway = False
        out["pressure_escape"] = bool(restart_established and (gained_20 or beyond_halfway))
        if out["pressure_escape"]:
            out["escape_distance"] = norm_x(restart_established[2], r.team, r.period, directions) - ox_n
        out["failed_escape"] = opp_established is not None and opp_established[0] <= 5

    # territorial-lock / opponent containment (only meaningful when the
    # OPPONENT ends up with the ball -- i.e. restart team lost it)
    if opp_established:
        out["opp_recovery"] = True
        contained = True
        exit_time = None
        for elapsed, cid, e in seq:
            if elapsed <= opp_established[0]:
                continue
            if team_of_cid.get(cid) != r.opp:
                continue
            if norm_x(e.get("x", r.ex), r.team, r.period, directions) >= 100.0 / 3.0:
                exit_time = elapsed - opp_established[0]
                break
        out["opp_contained_10"] = exit_time is None or exit_time > 10
        out["opp_contained_15"] = exit_time is None or exit_time > 15
        out["opp_reaches_middle_15"] = exit_time is not None and exit_time <= 15
        long_return = False
        for elapsed, cid, e in seq:
            if elapsed <= opp_established[0] or elapsed - opp_established[0] > 10:
                continue
            if team_of_cid.get(cid) == r.opp and e.get("typeId") == 1:
                q2 = qmap(e)
                if Q_LONG_BALL in q2:
                    long_return = True
                    break
        out["opp_long_return_10"] = long_return

    if restart_established and restart_established[0] <= 15 and \
            norm_x(restart_established[2], r.team, r.period, directions) >= 50:
        out["high_regain_15"] = True
    else:
        out["high_regain_15"] = False

    # progression / chance creation / risk
    out["progresses_20_15s"] = out.get("rtg_15") is not None and out["rtg_15"] >= 20
    out["reaches_final_third_15"] = out.get("rtg_15") is not None and \
        (ox_n + out["rtg_15"]) >= FINAL_THIRD_X

    def shots_within(window_s, team_filter):
        found = []
        for elapsed, cid, e in seq:
            if elapsed > window_s:
                break
            if e.get("typeId") not in SHOT_TYPES:
                continue
            if team_of_cid.get(cid) != team_filter:
                continue
            xg = danger_lookup.get((match_file, str(e.get("id"))), 0.0)
            found.append(xg)
        return found

    team_shots_20 = shots_within(20, r.team)
    opp_shots_20 = shots_within(20, r.opp) if r.opp else []
    out["shot_within_20"] = len(team_shots_20) > 0
    out["xg_for_20"] = sum(team_shots_20)
    out["xg_against_20"] = sum(opp_shots_20)

    out["opp_progression_or_shot"] = bool(opp_established and (
        norm_x(opp_established[2], r.team, r.period, directions) - norm_x(r.ex, r.team, r.period, directions) <= -20
        or opp_shots_20))

    # APER: established possession lands in middle or attacking third (x>=33.3 normalized)
    out["aper"] = bool(restart_established and
                        norm_x(restart_established[2], r.team, r.period, directions) >= 100.0 / 3.0)

    return out


def count_possessions(events, team_of_cid):
    """Segment the whole match into possession sequences by contestantId
    change (same method relationism_index.py already uses), per team: total
    possessions, and how many carry an open-play long ball in their first
    3 pass events -- the denominator/numerator LBR and LBD need."""
    totals = {}

    def flush(cid, seq):
        team = team_of_cid.get(cid)
        if team is None or not seq:
            return
        d = totals.setdefault(team, {"possessions": 0, "long_ball_possessions": 0})
        d["possessions"] += 1
        passes_seen = 0
        for e in seq:
            if e.get("typeId") != 1:
                continue
            passes_seen += 1
            q = qmap(e)
            if Q_LONG_BALL in q and not (q.keys() & {Q_FREE_KICK, Q_CORNER, Q_THROW_IN, Q_GOAL_KICK}):
                d["long_ball_possessions"] += 1
                break
            if passes_seen >= 3:
                break

    seq, cur_cid = [], None
    for e in events:
        cid = e.get("contestantId")
        if not cid:
            continue
        if cid != cur_cid:
            flush(cur_cid, seq)
            seq, cur_cid = [e], cid
        else:
            seq.append(e)
    flush(cur_cid, seq)
    return totals


def zscore_map(values):
    xs = list(values.values())
    if len(xs) < 2:
        return {k: 0.0 for k in values}
    mu, sd = statistics.fmean(xs), statistics.pstdev(xs)
    if sd == 0:
        return {k: 0.0 for k in values}
    return {k: (v - mu) / sd for k, v in values.items()}


def normal_cdf(z):
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


def aggregate_team(results):
    """One (season, kind, team) group of per-restart result dicts -> the
    full metric set defined in the brief."""
    n = len(results)
    contestable = [r for r in results if r.get("contestable")]
    nc = len(contestable)
    m = {"n_restarts": n, "n_contestable": nc}

    m["long_rate"] = rate(sum(1 for r in results if r["is_long"]), n)

    fc_won = [r for r in contestable if r.get("first_contact_won")]
    m["fcwr"] = rate(len(fc_won), nc)
    aerial = [r for r in contestable if r.get("first_contact_is_aerial")]
    m["afcwr"] = rate(sum(1 for r in aerial if r.get("aerial_won")), len(aerial))
    m["fcdv"] = mean([r.get("fc_direction_gain") for r in contestable])
    m["pfcr"] = rate(sum(1 for r in fc_won if r.get("fc_won_and_productive")), len(fc_won))

    uncontrolled = [r for r in contestable if r.get("uncontrolled_first_contact")]
    m["sbrr"] = rate(sum(1 for r in uncontrolled if r.get("second_ball_recovered")), len(uncontrolled))
    opp_fc = [r for r in contestable if r.get("first_contact_won") is False]
    m["ralfc"] = rate(sum(1 for r in opp_fc if r.get("recovery_after_opp_first_contact")), len(opp_fc))
    m["dpcr"] = rate(sum(1 for r in contestable if r.get("duel_phase_controlled")), nc)
    m["tsbc"] = mean([r.get("time_to_second_ball_control") for r in contestable])

    for w in (5, 10, 15):
        m[f"per{w}"] = rate(sum(1 for r in results if r.get(f"possession_established_{w}")), n)
    m["dpr"] = rate(sum(1 for r in results if r.get("direct_retention")), n)
    m["ipr"] = rate(sum(1 for r in results if r.get("indirect_retention")), n)
    m["err"] = (m["dpr"] or 0) + (m["ipr"] or 0) if (m["dpr"] is not None or m["ipr"] is not None) else None
    for w in (10, 15, 20):
        established = [r for r in results if r.get("restart_established")]
        m[f"psr{w}"] = rate(sum(1 for r in established if r.get(f"survives_{w}")), len(established))
    m["actions_sustained"] = mean([r.get("actions_sustained") for r in results])

    m["rtg_15"] = mean([r.get("rtg_15") for r in results])
    m["ctg"] = mean([r.get("ctg") for r in results])
    m["ntg"] = mean([r.get("ntg") for r in results])
    m["etg_basic"] = mean([r.get("etg_basic") for r in results])
    m["etg_event"] = mean([r.get("etg_event") for r in results])
    reach_halfway = [r for r in results if r.get("reaches_halfway")]
    m["trr"] = rate(sum(1 for r in reach_halfway if r.get("remains_beyond_halfway_15s")), len(reach_halfway))
    m["fter"] = rate(sum(1 for r in results if r.get("final_third_establishment")), n)
    m["ber20"] = rate(sum(1 for r in results if r.get("box_entry_20")), n)

    pressured = [r for r in results if r.get("under_pressure")]
    escapes = [r for r in pressured if "pressure_escape" in r]
    m["pesr"] = rate(sum(1 for r in escapes if r["pressure_escape"]), len(escapes))
    m["sed"] = mean([r.get("escape_distance") for r in pressured if r.get("pressure_escape")])
    m["fer"] = rate(sum(1 for r in escapes if r.get("failed_escape")), len(escapes))

    opp_recoveries = [r for r in results if r.get("opp_recovery")]
    m["tlr"] = rate(sum(1 for r in opp_recoveries if r.get("opp_contained_10")), len(opp_recoveries))
    m["oep15"] = (1 - rate(sum(1 for r in opp_recoveries if r.get("opp_reaches_middle_15")),
                            len(opp_recoveries))) if opp_recoveries else None
    m["frr"] = rate(sum(1 for r in opp_recoveries if r.get("opp_long_return_10")), len(opp_recoveries))
    m["hrr15"] = rate(sum(1 for r in results if r.get("high_regain_15")), n)

    m["pr15"] = rate(sum(1 for r in results if r.get("progresses_20_15s")), n)
    m["far15"] = rate(sum(1 for r in results if r.get("reaches_final_third_15")), n)
    m["scr20"] = rate(sum(1 for r in results if r.get("shot_within_20")), n)
    m["xgpr20"] = rate(sum(r.get("xg_for_20", 0) for r in results), n)
    m["xgca20"] = rate(sum(r.get("xg_against_20", 0) for r in results), n)
    m["ncv20"] = (m["xgpr20"] - m["xgca20"]) if (m["xgpr20"] is not None and m["xgca20"] is not None) else None

    m["dtr"] = rate(sum(1 for r in results if r.get("opp_progression_or_shot")), n)
    m["aper"] = rate(sum(1 for r in results if r.get("aper")), n)
    m["xtpr15"] = mean([r.get("xt_gain_15") for r in results])

    m["ft_post"] = mean([r.get("ft_post") for r in results])
    m["ft_pre"] = mean([r.get("ft_pre") for r in results])
    m["fts"] = mean([r.get("fts") for r in results])
    m["tps_post"] = mean([r.get("tps_post") for r in results])
    m["tps_pre"] = mean([r.get("tps_pre") for r in results])
    m["tds"] = mean([r.get("tds") for r in results])
    m["ftad"] = mean([r.get("ftad") for r in results])
    nep_vals = [r["nep"] for r in results if "nep" in r]
    m["nep"] = rate(sum(1 for v in nep_vals if v), len(nep_vals))
    nsp_vals = [r["nsp"] for r in results if "nsp" in r]
    m["nsp"] = rate(sum(1 for v in nsp_vals if v), len(nsp_vals))
    m["ccr15"] = rate(sum(1 for r in results if r.get("counterattack_conceded")), n)

    return m


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_season_aggregate import derive_team_name_by_cid  # noqa: E402


RDV_COMPONENTS = [("aper", 0.25), ("etg_event", 0.20), ("fts", 0.15), ("hrr15", 0.15),
                  ("xtpr15", 0.15), ("dtr", -0.10)]


def compute_rdv(team_metrics_for_kind):
    """z-score each RDV component across teams (within this season+kind), then
    the weighted composite, then map to 0-100 via the normal CDF."""
    zmaps = {}
    for name, _ in RDV_COMPONENTS:
        vals = {t: m[name] for t, m in team_metrics_for_kind.items() if m.get(name) is not None}
        zmaps[name] = zscore_map(vals)
    rdv = {}
    rdv_core = {}  # excludes xtpr15, usable for all 3 seasons
    for t in team_metrics_for_kind:
        score = 0.0
        score_core = 0.0
        core_weight = 0.0
        any_missing = False
        for name, w in RDV_COMPONENTS:
            z = zmaps[name].get(t, 0.0)
            score += w * z
            if name != "xtpr15":
                score_core += w * z
                core_weight += abs(w)
            if t not in zmaps[name]:
                any_missing = True
        rdv[t] = None if any_missing else round(100 * normal_cdf(score), 1)
        rdv_core[t] = round(100 * normal_cdf(score_core / core_weight * sum(abs(w) for _, w in RDV_COMPONENTS)), 1)
    return rdv, rdv_core


NUMERIC_METRIC_KEYS = [
    "long_rate", "lrr", "lbr", "lbd", "fcwr", "afcwr", "fcdv", "pfcr", "sbrr", "ralfc", "dpcr",
    "tsbc", "per5", "per10", "per15", "dpr", "ipr", "err", "psr10", "psr15", "psr20",
    "actions_sustained", "rtg_15", "ctg", "ntg", "etg_basic", "etg_event", "trr", "fter", "ber20",
    "pesr", "sed", "fer", "tlr", "oep15", "frr", "hrr15", "pr15", "far15", "scr20", "xgpr20",
    "xgca20", "ncv20", "dtr", "aper", "xtpr15", "ft_post", "ft_pre", "fts", "tps_post", "tps_pre",
    "tds", "ftad", "nep", "nsp", "ccr15",
]


def write_csv(path, rows, fieldnames):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore", restval="")
        w.writeheader()
        for row in rows:
            w.writerow({k: ("" if v is None else (round(v, 4) if isinstance(v, float) else v))
                        for k, v in row.items()})


def league_aggregate(all_results_by_kind_season):
    """Pool every team's restarts together (all 18 teams) for a season+kind ->
    one league-wide aggregate row, using the same aggregate_team() function."""
    pooled = []
    for team_results in all_results_by_kind_season.values():
        pooled.extend(team_results)
    return aggregate_team(pooled) if pooled else {}


def linear_trend(values_by_season):
    """OLS slope of metric value on season index (1,2,3) -- average annual
    change, ignoring seasons where the value is missing."""
    pts = [(i + 1, v) for i, s in enumerate(SEASONS) for v in [values_by_season.get(s)] if v is not None]
    if len(pts) < 2:
        return None
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    denom = sum((x - mx) ** 2 for x in xs)
    if denom == 0:
        return None
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / denom
    return slope


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    season_team_metrics = {}
    season_raw_groups = {}

    for season in SEASONS:
        print(f"Processing {season}...")
        events_dir = os.path.join(ROOT, "Events", season)
        if not os.path.isdir(events_dir):
            print(f"  skipping, no Events/{season}")
            continue
        team_of_cid = derive_team_name_by_cid(events_dir)

        danger_lookup = {}
        xt_grid = None
        if season == XG_SEASON:
            danger_path = os.path.join(ROOT, "Danger", "all_eredivisie_danger_models.csv")
            if os.path.exists(danger_path):
                with open(danger_path, encoding="utf-8-sig") as f:
                    for row in csv.DictReader(f):
                        danger_lookup[(row["match_file"], row["event_id"])] = num(row["xg"])
            xt_grid = load_xt_grid()

        by_group = {}
        possession_totals = {}
        goal_kick_totals = {}

        match_files = sorted(glob.glob(os.path.join(events_dir, "*.json")))
        for path in match_files:
            raw = json.load(open(path, encoding="utf-8"))
            fn = os.path.basename(path)
            events = raw.get("event", [])
            teams_here = sorted({team_of_cid.get(e.get("contestantId")) for e in events
                                  if e.get("contestantId") and team_of_cid.get(e.get("contestantId"))})
            if len(teams_here) != 2:
                continue
            opp_of_team = {teams_here[0]: teams_here[1], teams_here[1]: teams_here[0]}
            directions = team_directions(events, team_of_cid)
            restarts = find_restarts(events, team_of_cid, opp_of_team, directions)

            for r in restarts:
                seq = walk_forward(events, r.idx, r.t0)
                pre_seq = walk_backward(events, r.idx, r.t0)
                result = analyse_restart(r, seq, pre_seq, team_of_cid, directions, danger_lookup, fn, xt_grid)
                by_group.setdefault((r.kind, r.team), []).append(result)
                if r.kind == "goal_kick":
                    d = goal_kick_totals.setdefault(r.team, {"n": 0, "n_long": 0})
                    d["n"] += 1
                    d["n_long"] += int(r.is_long)

            for team, d in count_possessions(events, team_of_cid).items():
                tot = possession_totals.setdefault(team, {"possessions": 0, "long_ball_possessions": 0})
                tot["possessions"] += d["possessions"]
                tot["long_ball_possessions"] += d["long_ball_possessions"]

        teams = sorted(set(team_of_cid.values()))
        team_metrics = {"goal_kick": {}, "long_ball": {}}
        for kind in ("goal_kick", "long_ball"):
            for team in teams:
                results = by_group.get((kind, team), [])
                if not results:
                    continue
                m = aggregate_team(results)
                if kind == "goal_kick":
                    gk = goal_kick_totals.get(team, {"n": 0, "n_long": 0})
                    m["lrr"] = rate(gk["n_long"], gk["n"])
                else:
                    pt = possession_totals.get(team, {"possessions": 0, "long_ball_possessions": 0})
                    m["lbr"] = (rate(m["n_restarts"], pt["possessions"]) * 100) if pt["possessions"] else None
                    m["lbd"] = rate(pt["long_ball_possessions"], pt["possessions"])
                team_metrics[kind][team] = m

        for kind in ("goal_kick", "long_ball"):
            rdv, rdv_core = compute_rdv(team_metrics[kind])
            for team, m in team_metrics[kind].items():
                m["rdv"] = rdv.get(team)
                m["rdv_core"] = rdv_core.get(team)

        season_team_metrics[season] = team_metrics
        season_raw_groups[season] = by_group

        for kind in ("goal_kick", "long_ball"):
            rows = []
            for team, m in sorted(team_metrics[kind].items()):
                rows.append({"team": team, **m})
            out_path = os.path.join(ROOT, "Aggregated", season,
                                     f"restart_{kind}_team.csv")
            fieldnames = ["team", "n_restarts", "n_contestable"] + NUMERIC_METRIC_KEYS + ["rdv", "rdv_core"]
            write_csv(out_path, rows, fieldnames)
            print(f"  wrote {out_path} ({len(rows)} teams)")

    # ---- league-wide (all teams pooled) per season, per kind ----
    league_rows = []
    for kind in ("goal_kick", "long_ball"):
        for season in SEASONS:
            groups = season_raw_groups.get(season, {})
            pooled = []
            for (k, team), results in groups.items():
                if k == kind:
                    pooled.extend(results)
            if not pooled:
                continue
            m = aggregate_team(pooled)
            m["kind"] = kind
            m["season"] = season
            league_rows.append(m)
    write_csv(os.path.join(OUT_DIR, "league_wide_by_season.csv"), league_rows,
              ["kind", "season", "n_restarts", "n_contestable"] + NUMERIC_METRIC_KEYS)

    # ---- three-season comparison (league-wide) ----
    comparison_rows = []
    for kind in ("goal_kick", "long_ball"):
        by_season = {row["season"]: row for row in league_rows if row["kind"] == kind}
        for key in NUMERIC_METRIC_KEYS:
            values = {s: by_season[s].get(key) for s in SEASONS if s in by_season}
            s1, s3 = values.get(SEASONS[0]), values.get(SEASONS[-1])
            delta = (s3 - s1) if (s1 is not None and s3 is not None) else None
            pct = (delta / s1 * 100) if (delta is not None and s1 not in (None, 0)) else None
            trend = linear_trend(values)
            comparison_rows.append({
                "kind": kind, "metric": key,
                SEASONS[0]: values.get(SEASONS[0]), SEASONS[1]: values.get(SEASONS[1]),
                SEASONS[2]: values.get(SEASONS[2]),
                "delta_s3_minus_s1": delta, "pct_change": pct, "annual_trend_beta1": trend,
            })
    write_csv(os.path.join(OUT_DIR, "three_season_comparison.csv"), comparison_rows,
              ["kind", "metric"] + SEASONS + ["delta_s3_minus_s1", "pct_change", "annual_trend_beta1"])
    print(f"Wrote {os.path.join(OUT_DIR, 'three_season_comparison.csv')}")
    print(f"Wrote {os.path.join(OUT_DIR, 'league_wide_by_season.csv')}")


if __name__ == "__main__":
    main()


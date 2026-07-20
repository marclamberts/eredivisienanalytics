""""Goals Added"-style possession-value model, inspired by American Soccer
Analysis's publicly described g+ approach: an Expected Possession Value
(EPV) surface over pitch zones, action-level value = EPV after minus EPV
before, split into ASA's category vocabulary (Dribbling, Passing,
Receiving, Shooting, Interrupting, Fouling, Goalkeeping).

This is NOT American Soccer Analysis's actual model -- their exact EPV
surface, category definitions, and passer/receiver value split are
proprietary and undocumented. This is an independent from-scratch EPV model
(a Markov value-iteration over a 16x12 pitch grid, same family of technique
as Karun Singh's xT and this repo's own xT/GDA models, but estimated fresh
here rather than reusing either), used to approximate the spirit of ASA's
category breakdown. Documented simplifications:
  - Dribbling = take-on actions only (typeId 3); carries are not
    synthesized as separate actions.
  - Receiving value = 50% of a completed pass's value, credited to the
    receiver (the next action's player, if same team within 8 seconds).
    ASA's real passer/receiver split ratio is not public; 50/50 is this
    script's own choice, not a discovered constant.
  - Fouling value uses the same EPV grid mirrored to the opponent's shot
    zone as a proxy for "value of a free kick given away", not a
    dedicated set-piece model.
  - Goalkeeping is scored separately (PSxG minus goals conceded while
    that player was in goal), since ASA also models goalkeeping
    separately from its outfield g+ categories.

Grid: 16 bins along x (0-100), 12 bins along y (0-100) = 192 zones.
"""
import csv
import glob
import json
import os
from collections import defaultdict

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EVENTS_DIR = os.path.join(ROOT, "Events")
CACHE_PATH = os.path.join(ROOT, "Season stats", "Scripts", "_goals_added_cache.csv")

NX, NY = 16, 12
RECEIVE_WINDOW_S = 8.0
RECEIVE_SHARE = 0.5

TACKLE, INTERCEPTION, CLEARANCE, BLOCKED_PASS = 7, 8, 12, 74
FOUL = 4
CROSS_QID = 2


def rows(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def num(v, default=0.0):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def qmap(e):
    return {q.get("qualifierId"): q.get("value") for q in e.get("qualifier", []) or []}


def zone_of(x, y):
    xi = min(NX - 1, max(0, int(x / 100.0 * NX)))
    yi = min(NY - 1, max(0, int(y / 100.0 * NY)))
    return xi * NY + yi


def zone_center(z):
    xi, yi = divmod(z, NY)
    return (xi + 0.5) / NX * 100.0, (yi + 0.5) / NY * 100.0


def load_shot_xg():
    xg = {}
    for r in rows(os.path.join(ROOT, "Danger", "all_eredivisie_danger_models.csv")):
        xg[(r["match_file"], r["event_id"])] = (num(r["xg"]), num(r["psxg"]), r["is_goal"] == "1")
    return xg


def load_positions_gk():
    """player_id -> True if their season position group is GK, reusing the
    same Team Set Up qualifiers as the other Season stats builders."""
    votes = defaultdict(lambda: defaultdict(int))
    for path in sorted(glob.glob(os.path.join(EVENTS_DIR, "*.json"))):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        for e in data.get("event", []):
            if e.get("typeId") != 34:
                continue
            q = qmap(e)
            pids = (q.get(30) or "").split(", ")
            pos = (q.get(44) or "").split(", ")
            slots = (q.get(131) or "").split(", ")
            if not (len(pids) == len(pos) == len(slots)):
                continue
            for pid, p, s in zip(pids, pos, slots):
                if s.strip() != "0" and p.strip() in ("1", "2", "3", "4"):
                    votes[pid][int(p)] += 1
    return {pid: (max(c, key=c.get) == 1) for pid, c in votes.items()}


def main():
    print("Loading shot xG/PSxG lookup...")
    shot_xg = load_shot_xg()
    is_gk = load_positions_gk()

    print("Building action stream + Markov transition counts...")
    n_from = np.zeros(NX * NY)
    n_shot = np.zeros(NX * NY)
    sum_xg = np.zeros(NX * NY)
    trans = defaultdict(float)  # (z_from, z_to) -> successful-move count

    actions_all = []  # kept for the second (valuation) pass

    for path in sorted(glob.glob(os.path.join(EVENTS_DIR, "*.json"))):
        fn = os.path.basename(path)
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        events = [e for e in data.get("event", [])
                  if e.get("typeId") in (1, 3, 13, 14, 15, 16) and e.get("playerId")]
        events.sort(key=lambda e: (e.get("periodId", 0), e.get("timeMin", 0), e.get("timeSec", 0), e.get("eventId", 0)))

        for e in events:
            tid = e.get("typeId")
            q = qmap(e)
            x, y = num(e.get("x")), num(e.get("y"))
            z0 = zone_of(x, y)
            success = e.get("outcome") == 1

            if tid == 1:  # pass
                ex, ey = num(q.get(140, x)), num(q.get(141, y))
                z1 = zone_of(ex, ey)
                n_from[z0] += 1
                if success:
                    trans[(z0, z1)] += 1
                actions_all.append({
                    "match": fn, "player_id": e.get("playerId"), "team": e.get("contestantId"),
                    "type": "pass", "success": int(success), "z0": z0, "z1": z1,
                    "t": e.get("periodId", 0) * 10000 + e.get("timeMin", 0) * 60 + e.get("timeSec", 0),
                    "id": e.get("id"), "is_cross": CROSS_QID in q,
                })
            elif tid == 3:  # take-on / dribble
                ex = min(100.0, x + 5.0) if success else x
                z1 = zone_of(ex, y)
                n_from[z0] += 1
                if success:
                    trans[(z0, z1)] += 1
                actions_all.append({
                    "match": fn, "player_id": e.get("playerId"), "team": e.get("contestantId"),
                    "type": "take_on", "success": int(success), "z0": z0, "z1": z1,
                    "t": e.get("periodId", 0) * 10000 + e.get("timeMin", 0) * 60 + e.get("timeSec", 0),
                    "id": e.get("id"), "is_cross": False,
                })
            else:  # shot
                xg, psxg, is_goal = shot_xg.get((fn, str(e.get("id"))), (0.05, 0.0, False))
                n_from[z0] += 1
                n_shot[z0] += 1
                sum_xg[z0] += xg
                actions_all.append({
                    "match": fn, "player_id": e.get("playerId"), "team": e.get("contestantId"),
                    "type": "shot", "success": int(tid == 16), "z0": z0, "z1": z0,
                    "t": e.get("periodId", 0) * 10000 + e.get("timeMin", 0) * 60 + e.get("timeSec", 0),
                    "id": e.get("id"), "is_cross": False, "xg": xg,
                })

        # defensive / foul actions for Interrupting & Fouling categories
        for e in data.get("event", []):
            tid = e.get("typeId")
            if tid not in (TACKLE, INTERCEPTION, CLEARANCE, BLOCKED_PASS, FOUL) or not e.get("playerId"):
                continue
            x, y = num(e.get("x")), num(e.get("y"))
            actions_all.append({
                "match": fn, "player_id": e.get("playerId"), "team": e.get("contestantId"),
                "type": {TACKLE: "tackle", INTERCEPTION: "interception", CLEARANCE: "clearance",
                         BLOCKED_PASS: "blocked_pass", FOUL: "foul"}[tid],
                "success": int(e.get("outcome") == 1), "z0": zone_of(x, y), "z1": zone_of(x, y),
                "t": e.get("periodId", 0) * 10000 + e.get("timeMin", 0) * 60 + e.get("timeSec", 0),
                "id": e.get("id"), "is_cross": False,
            })

    print(f"{len(actions_all)} actions collected")

    print("Solving EPV via value iteration...")
    avg_xg = np.divide(sum_xg, n_shot, out=np.zeros_like(sum_xg), where=n_shot > 0)
    p_shot = np.divide(n_shot, n_from, out=np.zeros_like(n_shot), where=n_from > 0)
    move_probs = defaultdict(list)  # z0 -> [(z1, prob), ...]
    for (z0, z1), c in trans.items():
        if n_from[z0] > 0:
            move_probs[z0].append((z1, c / n_from[z0]))

    EPV = np.zeros(NX * NY)
    for it in range(60):
        new_EPV = p_shot * avg_xg
        for z0, lst in move_probs.items():
            new_EPV[z0] += sum(p * EPV[z1] for z1, p in lst)
        delta = np.abs(new_EPV - EPV).max()
        EPV = new_EPV
        if delta < 1e-7:
            break
    print(f"  converged after {it + 1} iterations, EPV range [{EPV.min():.4f}, {EPV.max():.4f}]")

    print("Valuing actions and assigning receiving credit...")
    by_match = defaultdict(list)
    for i, a in enumerate(actions_all):
        by_match[a["match"]].append(i)

    category_value = defaultdict(lambda: defaultdict(float))  # player_id -> category -> value

    for fn, idxs in by_match.items():
        idxs.sort(key=lambda i: actions_all[i]["t"])
        for pos, i in enumerate(idxs):
            a = actions_all[i]
            pid, team = a["player_id"], a["team"]
            v_before = EPV[a["z0"]]

            if a["type"] == "shot":
                value = a["xg"] - v_before
                category_value[pid]["Shooting"] += value
                continue

            if a["type"] in ("pass", "take_on"):
                if a["success"]:
                    v_after = EPV[a["z1"]]
                    value = v_after - v_before
                else:
                    value = -v_before  # turnover cost
                cat = "Dribbling" if a["type"] == "take_on" else "Passing"
                if a["type"] == "pass" and a["success"]:
                    # Split every completed pass's value (positive or
                    # negative) between passer and receiver, not just
                    # progressive ones -- otherwise routine sideways/backward
                    # circulation (most passes in a match) hits the passer at
                    # full penalty with no offsetting share of the rarer
                    # progressive passes' credit, systematically dragging
                    # high-volume passers' totals negative.
                    passer_share = value * (1 - RECEIVE_SHARE)
                    receiver_share = value * RECEIVE_SHARE
                    category_value[pid][cat] += passer_share
                    if pos + 1 < len(idxs):
                        nxt = actions_all[idxs[pos + 1]]
                        if nxt["team"] == team and (nxt["t"] - a["t"]) <= RECEIVE_WINDOW_S:
                            category_value[nxt["player_id"]]["Receiving"] += receiver_share
                        else:
                            category_value[pid][cat] += receiver_share  # no clear receiver, keep with passer
                else:
                    category_value[pid][cat] += value
                continue

            if a["type"] in ("tackle", "interception", "clearance", "blocked_pass"):
                # Value of gaining possession here, in the defender's own
                # attacking frame -- deliberately NOT the opponent's mirrored
                # threat value: the attacking side's own failed-pass penalty
                # (-v_before, above) already prices in the turnover, and a
                # broken-down attack often produces several defensive touches
                # (tackle, then clearance, ...) in the same sequence, so
                # crediting each one the full mirrored opponent EPV would
                # multiply-count a single prevented attack. This rewards
                # winning the ball in advanced areas (high press) without
                # that blow-up.
                value = EPV[a["z0"]] if a["success"] else 0.0
                category_value[pid]["Interrupting"] += value
                continue

            if a["type"] == "foul":
                mirror_z = zone_of(*[100 - c if k == 0 else c for k, c in enumerate(zone_center(a["z0"]))])
                # outcome==1 on typeId 4 => this player committed the foul (see build_defensive_metrics.py)
                value = -EPV[mirror_z] * 0.3 if a["success"] else EPV[a["z0"]] * 0.3
                category_value[pid]["Fouling"] += value

    print("Scoring goalkeeping (PSxG minus goals conceded while in goal)...")
    gk_value = defaultdict(float)
    stints = rows(os.path.join(ROOT, "GDA", "gda_player_stints.csv"))
    gk_stints = [s for s in stints if is_gk.get(s["player_id"])]
    stints_by_match = defaultdict(list)
    for s in gk_stints:
        stints_by_match[s["match_file"]].append(s)
    for r in rows(os.path.join(ROOT, "Danger", "all_eredivisie_danger_models.csv")):
        fn = r["match_file"]
        t = num(r["time_min"]) * 60 + num(r["time_sec"])
        candidates = [s for s in stints_by_match.get(fn, [])
                      if s["contestant_id"] != r["contestant_id"] and num(s["start_s"]) <= t < num(s["end_s"])]
        if candidates:
            keeper = candidates[0]["player_id"]
            gk_value[keeper] += num(r["psxg"]) - (1.0 if r["is_goal"] == "1" else 0.0)

    print("Aggregating to player level...")
    all_categories = ["Dribbling", "Passing", "Receiving", "Shooting", "Interrupting", "Fouling"]
    records = []
    all_pids = set(category_value) | set(gk_value)
    for pid in all_pids:
        cats = category_value.get(pid, {})
        row = {"player_id": pid}
        for c in all_categories:
            row[c] = cats.get(c, 0.0)
        row["Goalkeeping"] = gk_value.get(pid, 0.0)
        row["Goals Added"] = sum(row[c] for c in all_categories) + row["Goalkeeping"]
        records.append(row)

    out = pd.DataFrame(records)
    out.sort_values("Goals Added", ascending=False, inplace=True)
    out.to_csv(CACHE_PATH, index=False)
    print(f"Wrote {CACHE_PATH}: {len(out)} players")
    print(out.head(15).to_string())


if __name__ == "__main__":
    main()

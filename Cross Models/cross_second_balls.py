"""
Second Balls After Open-Play Crosses -- Eredivisie 2025/26
================================================================
Where does the loose ball actually land after a cross is headed or
cleared away? Same "second contested action" method as
PSV Season Report/Scripts/set_piece_second_balls.py (built for a
team's own corners, free kicks and long throws), applied here to a
team's own OPEN-PLAY crosses -- the same population the cross model
(Cross Models/score_eredivisie_crosses.py) scores: an Opta pass event
carrying qualifier 2 (Cross), without qualifier 5 (free kick), 6
(corner) or 160 (throw-in set piece).

Why this matters tactically: the front line's job on a cross is to
win first contact, not necessarily to keep the ball -- a cleared
header routinely travels 30-40m before anyone gets a clean touch on
it. That second contest, not the six-yard box, is where a spare
defender (holding midfielder / covering centre-back) needs to stand
to sweep up. Plotting where second balls are actually won or lost
turns that into a concrete zone instead of a general instruction.

Method: for each open-play cross, look at the next events (~12s
window) for a contested action (aerial duel, tackle, interception,
clearance, ball recovery). The SECOND such contested action is "the
second ball" -- the first is usually just the target player's initial
header/contact -- allowing at most one clean pass in between (e.g. a
knockdown headed on to a teammate). Plotted wherever that second
contest happened, with x/y normalized per team per period to
"distance from this team's own goal" (own goal on the left) so every
cross reads the same regardless of which end they defended that half
-- same convention as Goal Kick Model/goalkick_pitch_map.py.

In Marc Lamberts' Meridian house style (housestyle/ package at the
repo root).

Usage: python3 cross_second_balls.py "<team name>" [out.png]
"""
import collections
import glob
import json
import os
import re
import sys

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mplsoccer import Pitch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from housestyle import style, components  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "Eredivisie Events", "2025-2026")
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
SOURCE = "Opta event data, Eredivisie 2025/26"

T_PASS = 1
CROSS_QID, FREEKICK_QID, CORNER_QID, THROWIN_QID = 2, 5, 6, 160
CONTESTED_TYPES = {44, 7, 8, 12, 49}  # Aerial, Tackle, Interception, Clearance, Ball Recovery
WINDOW_EVENTS = 5
WINDOW_MIN = 12 / 60
MAX_CLEAN_PASSES_BETWEEN = 1
X_SCALE, Y_SCALE = 1.05, 0.68
GOAL_X = 105.0

PREFIX_RE = re.compile(r"^(CSD|CD|CS|SD)\s+")


def clean_name(name):
    return PREFIX_RE.sub("", name)


def qmap(e):
    return {q["qualifierId"]: q.get("value") for q in e.get("qualifier", []) or []}


def to_m(x, y):
    return x * X_SCALE, y * Y_SCALE


def minute_value(e):
    return float(e.get("timeMin") or 0) + float(e.get("timeSec") or 0) / 60.0


def build_team_map(files):
    team_cid_sets = collections.defaultdict(list)
    for fn in files:
        m = re.match(r"\d{4}-\d{2}-\d{2}_(.+) - (.+)\.json$", os.path.basename(fn))
        if not m:
            continue
        home, away = m.group(1), m.group(2)
        with open(fn) as f:
            data = json.load(f)
        cids = set(e["contestantId"] for e in data["event"] if e.get("contestantId"))
        team_cid_sets[home].append(cids)
        team_cid_sets[away].append(cids)
    team_to_cid = {}
    for team, sets in team_cid_sets.items():
        inter = set.intersection(*sets)
        if len(inter) == 1:
            team_to_cid[team] = next(iter(inter))
    return team_to_cid


def compute_attack_directions(files):
    """(match_file, contestant_id, period_id) -> 1 if the team attacks
    toward higher x that period, else -1. Same average-pass-position
    heuristic as Disruption/league_disruption_visuals.py, recomputed
    here so this script isn't tied to that module's own DATA_DIR."""
    directions = {}
    for fn in files:
        basename = os.path.basename(fn)
        with open(fn) as f:
            data = json.load(f)
        sums = {}
        for e in data["event"]:
            if e.get("typeId") != T_PASS or e.get("x") is None:
                continue
            if e["x"] == 0 and e["y"] == 0:
                continue
            key = (e["contestantId"], e["periodId"])
            s = sums.setdefault(key, [0.0, 0])
            s[0] += e["x"]
            s[1] += 1
        for (cid, period), (total, n) in sums.items():
            avg_x = total / n if n else 50
            directions[(basename, cid, period)] = 1 if avg_x < 50 else -1
    return directions


def is_open_play_cross(e):
    if e.get("typeId") != T_PASS:
        return False
    q = qmap(e)
    if CROSS_QID not in q:
        return False
    return not (FREEKICK_QID in q or CORNER_QID in q or THROWIN_QID in q)


def collect(files, cid, directions):
    events_out = []
    for fn in files:
        basename = os.path.basename(fn)
        with open(fn) as f:
            data = json.load(f)
        evs = [e for e in data["event"] if e.get("periodId") in (1, 2)]
        evs.sort(key=lambda e: (e["periodId"], minute_value(e), e.get("eventId", 0)))

        for i, e in enumerate(evs):
            if e.get("contestantId") != cid or not is_open_play_cross(e):
                continue

            t0 = minute_value(e)
            window = [ev for ev in evs[i + 1:i + 1 + WINDOW_EVENTS]
                      if minute_value(ev) - t0 <= WINDOW_MIN]
            first_idx = next((j for j, ev in enumerate(window) if ev.get("typeId") in CONTESTED_TYPES), None)
            if first_idx is None:
                continue

            # second contested action, allowing at most one clean pass in
            # between -- if the ball strings together more than that,
            # possession has settled and any later contest is unrelated
            sb, clean_passes = None, 0
            for ev in window[first_idx + 1:]:
                if ev.get("typeId") in CONTESTED_TYPES:
                    sb = ev
                    break
                if ev.get("typeId") == T_PASS and ev.get("outcome") == 1:
                    clean_passes += 1
                    if clean_passes > MAX_CLEAN_PASSES_BETWEEN:
                        break
            if sb is None or sb.get("x") is None or sb.get("y") is None:
                continue

            period = e["periodId"]
            direction = directions.get((basename, cid, period), 1)
            x, y = to_m(float(sb["x"]), float(sb["y"]))
            if direction == -1:
                x, y = GOAL_X - x, 68.0 - y
            won = sb.get("contestantId") == cid
            events_out.append({"x": x, "y": y, "won": won})
    return events_out


def make_plot(team_name, events, mode, out_path):
    palette, _ = style.apply(mode)
    n_total = len(events)
    n_won = sum(1 for e in events if e["won"])
    pct = n_won / n_total * 100 if n_total else 0

    fig = plt.figure(figsize=(11, 8.6))
    pitch = Pitch(pitch_type="uefa", pitch_color=palette["surface"], line_color=palette["axis"],
                  linewidth=1.1, half=False, line_zorder=2)
    ax = fig.add_axes([0.05, 0.10, 0.90, 0.62])
    pitch.draw(ax=ax)

    for e in sorted(events, key=lambda e: e["won"]):
        color = palette["accent"] if e["won"] else palette["axis"]
        alpha = 0.9 if e["won"] else 0.45
        pitch.scatter(e["x"], e["y"], ax=ax, s=90, color=color,
                     edgecolors=palette["surface"], linewidths=1.2, alpha=alpha, zorder=3)

    legend_elems = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=palette["accent"], markersize=10,
              label="Second ball won by this team"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=palette["axis"], markersize=10,
              alpha=0.6, label="Second ball won by opposition"),
    ]
    ax.legend(handles=legend_elems, loc="upper center", bbox_to_anchor=(0.5, -0.03), ncol=2,
             frameon=False, fontsize=9.5, labelcolor=palette["ink_secondary"])

    components.header(
        fig, kicker="Crosses -- Second Balls",
        title=f"{clean_name(team_name)}: {n_won} of {n_total} second balls off open-play crosses ({pct:.0f}%)",
        dek="Eredivisie 2025/26  ·  Season  ·  open-play crosses only  ·  "
            "pitch runs own goal (left) to opponent goal (right)",
        palette=palette)
    components.footer(fig, source=SOURCE, palette=palette)

    fig.savefig(out_path, dpi=200, facecolor=palette["surface"])
    plt.close(fig)
    print("Saved:", out_path)


def visual_dir(theme):
    d = os.path.join(OUT_DIR, "Visual - Dark" if theme == "dark" else "Visual - Light")
    os.makedirs(d, exist_ok=True)
    return d


def main():
    args = [a for a in sys.argv[1:] if a]
    team_name = args[0] if args else "PSV Eindhoven"
    safe_name = team_name.replace(" ", "_")
    out_name = args[1] if len(args) > 1 else f"cross_second_balls_{safe_name}.png"

    files = sorted(glob.glob(os.path.join(DATA_DIR, "*.json")))
    if not files:
        raise SystemExit(f"No match files found in {DATA_DIR}")
    team_to_cid = build_team_map(files)
    match = next((full for full in team_to_cid if team_name.lower() in full.lower()), None)
    if match is None:
        raise SystemExit(f"Team '{team_name}' not found. Options: {sorted(team_to_cid)}")
    cid = team_to_cid[match]

    directions = compute_attack_directions(files)
    events = collect(files, cid, directions)
    if not events:
        raise SystemExit(f"No identifiable second-ball contests found for '{match}'")

    for mode in ("light", "dark"):
        out_path = os.path.join(visual_dir(mode), out_name)
        make_plot(match, events, mode, out_path)

    print(f"n={len(events)} won={sum(1 for e in events if e['won'])} "
          f"pct={sum(1 for e in events if e['won']) / len(events):.1%}")


if __name__ == "__main__":
    main()

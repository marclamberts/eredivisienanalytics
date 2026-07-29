"""
Second Balls After Open-Play Crosses -- Eredivisie 2025/26
================================================================
When this team crosses (attacking), do they win the second ball?
Same "second contested action" method as
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
knockdown headed on to a teammate). Restricted to second balls that
land in the attacking half (x >= 50): a contest that rebounds back
past halfway is a broken-down transition, not really the crossing
phase anymore.

No home/away or half-time flip is applied. Checked against this feed's
own shot locations (both teams, every period, several matches): raw x
already sits high (75-97) whenever a team shoots, in BOTH periods --
each event's x/y is relative to the ACTING team's own attacking
direction (0 = their own goal, 100 = the goal they attack), not a
shared physical pitch frame. A period-based direction flip (the
convention used by Goal Kick Model/goalkick_pitch_map.py, built for a
differently-behaved feed) would silently scatter events onto the
wrong end of the pitch here. The same per-team-relative convention
also means an opposing player's event needs converting into our frame
via (100-x, 100-y) before use -- confirmed exactly against aerial-duel
pairs, which this feed logs twice (once per player, cross-referenced
by qualifier 233): the two copies' coordinates are exact (100-x,
100-y) mirrors of each other. See dedupe_aerial_duels/to_own_frame.

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
from mplsoccer import VerticalPitch

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
ATTACKING_HALF_X = 50.0

PREFIX_RE = re.compile(r"^(CSD|CD|CS|SD)\s+")


def clean_name(name):
    return PREFIX_RE.sub("", name)


def qmap(e):
    return {q["qualifierId"]: q.get("value") for q in e.get("qualifier", []) or []}


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


def is_open_play_cross(e):
    if e.get("typeId") != T_PASS:
        return False
    q = qmap(e)
    if CROSS_QID not in q:
        return False
    return not (FREEKICK_QID in q or CORNER_QID in q or THROWIN_QID in q)


def dedupe_aerial_duels(evs, cid):
    """Aerial duels (typeId 44) are logged TWICE -- once per player, each in
    that player's own team's coordinate frame, cross-referenced by
    qualifier 233 (each copy's value is the other copy's eventId).
    Confirmed on this feed: the pair's (x, y) are exact (100-x, 100-y)
    mirrors of each other, and outcome is 1 for the winner's copy, 0 for
    the loser's -- so left undeduped, a single real duel is scanned as two
    separate "contested actions" back to back. Keep one merged copy per
    duel, preferring the side belonging to `cid` (already in our frame)."""
    by_event_id = {e.get("eventId"): e for e in evs if e.get("typeId") == 44}
    drop = set()
    for eid, e in by_event_id.items():
        if eid in drop:
            continue
        try:
            partner_id = int(qmap(e).get(233))
        except (TypeError, ValueError):
            continue
        partner = by_event_id.get(partner_id)
        if partner is None or partner.get("eventId") in drop:
            continue
        loser_id = partner.get("eventId") if e.get("contestantId") == cid else eid
        drop.add(loser_id)
    return [e for e in evs if not (e.get("typeId") == 44 and e.get("eventId") in drop)]


def to_own_frame(e):
    """x/y on this feed are relative to the ACTING player's own team's
    attacking direction (0 = their own goal, 100 = the goal they attack),
    not a shared physical pitch frame -- confirmed via the aerial-duel
    pairs above and via clearance/shot locations across periods and
    home/away sides. An event authored by the opposing team therefore
    needs converting into our own frame via the same (100-x, 100-y)
    mirror before it can be plotted or compared against our own events."""
    x, y = float(e["x"]), float(e["y"])
    if e.get("_own"):
        return x, y
    return 100.0 - x, 100.0 - y


def collect(files, cid):
    events_out = []
    for fn in files:
        with open(fn) as f:
            data = json.load(f)
        evs = [e for e in data["event"] if e.get("periodId") in (1, 2)]
        evs.sort(key=lambda e: (e["periodId"], minute_value(e), e.get("eventId", 0)))
        evs = dedupe_aerial_duels(evs, cid)
        for e in evs:
            e["_own"] = e.get("contestantId") == cid

        for i, e in enumerate(evs):
            if not e["_own"] or not is_open_play_cross(e):
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

            x, y = to_own_frame(sb)
            if x < ATTACKING_HALF_X:
                continue
            won = sb["_own"] and sb.get("outcome") == 1
            events_out.append({"x": x, "y": y, "won": won})
    return events_out


def make_plot(team_name, events, mode, out_path):
    palette, _ = style.apply(mode)
    n_total = len(events)
    n_won = sum(1 for e in events if e["won"])
    pct = n_won / n_total * 100 if n_total else 0

    fig = plt.figure(figsize=(9.5, 11))
    pitch = VerticalPitch(pitch_type="opta", pitch_color=palette["surface"], line_color=palette["axis"],
                          linewidth=1.1, half=True, line_zorder=2)
    ax = fig.add_axes([0.08, 0.10, 0.84, 0.65])
    pitch.draw(ax=ax)

    for e in sorted(events, key=lambda e: e["won"]):
        color = palette["accent"] if e["won"] else palette["axis"]
        alpha = 0.9 if e["won"] else 0.45
        pitch.scatter(e["x"], e["y"], ax=ax, s=100, color=color,
                     edgecolors=palette["surface"], linewidths=1.2, alpha=alpha, zorder=3)

    legend_elems = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=palette["accent"], markersize=10,
              label="Second ball won by this team"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=palette["axis"], markersize=10,
              alpha=0.6, label="Second ball won by opposition"),
    ]
    ax.legend(handles=legend_elems, loc="upper center", bbox_to_anchor=(0.5, -0.03), ncol=1,
             frameon=False, fontsize=9.5, labelcolor=palette["ink_secondary"])

    components.header(
        fig, kicker="Crosses -- Second Balls",
        title=f"{clean_name(team_name)}: {n_won} of {n_total} second balls off open-play crosses ({pct:.0f}%)",
        dek="Eredivisie 2025/26  ·  Season  ·  attacking half only  ·  "
            "second contested action after the cross",
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

    events = collect(files, cid)
    if not events:
        raise SystemExit(f"No identifiable second-ball contests found for '{match}'")

    for mode in ("light", "dark"):
        out_path = os.path.join(visual_dir(mode), out_name)
        make_plot(match, events, mode, out_path)

    print(f"n={len(events)} won={sum(1 for e in events if e['won'])} "
          f"pct={sum(1 for e in events if e['won']) / len(events):.1%}")


if __name__ == "__main__":
    main()

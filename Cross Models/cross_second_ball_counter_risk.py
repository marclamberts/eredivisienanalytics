"""
Counter Risk From Lost Cross Second Balls -- Eredivisie 2025/26
================================================================
Follow-up to cross_second_balls.py: it shows PSV win only ~17% of
second balls after their own open-play crosses. That alone doesn't
say whether it matters -- most lost second balls probably just reset
into a settled defensive phase. This asks the sharper question: of
the ones PSV lose, how many turn into an actual shot against them
within the transition window, i.e. is this rest-defence gap actually
costing chances, or is it noise?

Method: same "second contested action" identification as
cross_second_balls.py (open-play crosses only, aerial duels deduped
via qualifier 233, own-frame coordinates -- see that file's docstring
for why). For every one of THIS team's lost second balls, walk forward
through the match's ball-involvement events (same convention as
Interception Model/build_interception_transition_model.py) crediting
them to the opponent, ending the chain at the first of:
  - a shot by the opponent (Miss/Post/Attempt Saved/Goal)
  - the ball coming back to this team
  - TRANSITION_WINDOW_SECONDS elapsed (10s, same window as the
    Interception model, for comparability)
  - the period ending
Aerial duels inside the chain are handled outcome-aware (contestantId
alone doesn't tell you who kept the ball for a duel, since both
players get a record -- see cross_second_balls.dedupe_aerial_duels),
otherwise a duel the opponent actually won would look like a false
turnover and cut the chain short.

Shot locations are plotted as-is, with no coordinate flip: a shot is
always recorded in the SHOOTING team's own attacking-direction frame,
and every shot here is by the opponent attacking this team's goal, so
a plain half=True pitch (box at top) already reads correctly as
"danger against."

In Marc Lamberts' Meridian house style (housestyle/ package at the
repo root).

Usage: python3 cross_second_ball_counter_risk.py "<team name>" [out.png]
"""
import glob
import json
import os
import re
import sys

import matplotlib.pyplot as plt
from mplsoccer import VerticalPitch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from housestyle import style, components  # noqa: E402

from cross_second_balls import (  # noqa: E402
    DATA_DIR, build_team_map, collect, clean_name, dedupe_aerial_duels,
)

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
SOURCE = "Opta event data, Eredivisie 2025/26"

SHOT_TYPES = {13, 14, 15, 16}  # Miss, Post, Attempt Saved, Goal
GOAL_TYPE = 16
AERIAL_TYPE = 44
TRANSITION_WINDOW_SECONDS = 10.0


def event_ball_involvements(events):
    """Same filter/order as Interception Model/build_interception_transition_model.py
    so the two models agree on what counts as a "touch"."""
    rows = [e for e in events if e.get("x") is not None and e.get("contestantId")]
    rows.sort(key=lambda e: (e["periodId"], e["timeMin"] * 60 + e["timeSec"], e["eventId"]))
    return rows


def event_time(e):
    return e["timeMin"] * 60 + e["timeSec"]


def find_transition_shot(touches, start_idx, cid, opponent_cid):
    """Walk forward from a lost second ball, crediting the chain to
    opponent_cid, and report the first shot it produces (if any) within
    the transition window."""
    t0 = event_time(touches[start_idx])
    period = touches[start_idx]["periodId"]
    for ev in touches[start_idx + 1:]:
        if ev["periodId"] != period:
            return None
        if event_time(ev) - t0 > TRANSITION_WINDOW_SECONDS:
            return None
        if ev["typeId"] == AERIAL_TYPE:
            # both duelists get a record -- contestantId alone doesn't
            # tell you who kept the ball, outcome does
            opponent_keeps_it = (
                (ev["contestantId"] == opponent_cid and ev.get("outcome") == 1)
                or (ev["contestantId"] == cid and ev.get("outcome") == 0)
            )
            if not opponent_keeps_it:
                return None
        elif ev["contestantId"] != opponent_cid:
            return None
        if ev["typeId"] in SHOT_TYPES:
            return ev
    return None


def other_cid(basename, cid, team_to_cid):
    m = re.match(r"\d{4}-\d{2}-\d{2}_(.+) - (.+)\.json$", basename)
    if not m:
        return None
    home, away = m.group(1), m.group(2)
    home_cid, away_cid = team_to_cid.get(home), team_to_cid.get(away)
    if cid == home_cid:
        return away_cid
    if cid == away_cid:
        return home_cid
    return None


def collect_counter_shots(files, cid, team_to_cid):
    events = collect(files, cid)
    lost = [e for e in events if not e["won"]]

    by_match = {}
    for e in lost:
        by_match.setdefault(e["basename"], []).append(e["sb"])

    shots, n_led_to_shot = [], 0
    for basename, sb_events in by_match.items():
        fn = os.path.join(DATA_DIR, basename)
        with open(fn) as f:
            data = json.load(f)
        touches = event_ball_involvements(data["event"])
        touches = dedupe_aerial_duels(touches, cid)
        by_id = {t["id"]: i for i, t in enumerate(touches)}

        opponent_cid = other_cid(basename, cid, team_to_cid)
        if opponent_cid is None:
            continue

        for sb in sb_events:
            idx = by_id.get(sb["id"])
            if idx is None:
                continue
            shot = find_transition_shot(touches, idx, cid, opponent_cid)
            if shot is not None:
                n_led_to_shot += 1
                shots.append(shot)

    return len(lost), n_led_to_shot, shots


def make_plot(team_name, n_lost, n_led_to_shot, shots, mode, out_path):
    palette, _ = style.apply(mode)
    pct = n_led_to_shot / n_lost * 100 if n_lost else 0
    n_goals = sum(1 for s in shots if s["typeId"] == GOAL_TYPE)

    fig = plt.figure(figsize=(9.5, 11))
    pitch = VerticalPitch(pitch_type="opta", pitch_color=palette["surface"], line_color=palette["axis"],
                          linewidth=1.1, half=True, line_zorder=2)
    ax = fig.add_axes([0.08, 0.10, 0.84, 0.65])
    pitch.draw(ax=ax)

    for s in shots:
        is_goal = s["typeId"] == GOAL_TYPE
        pitch.scatter(s["x"], s["y"], ax=ax, s=170 if is_goal else 110,
                     color=palette["accent"], edgecolors=palette["surface"],
                     linewidths=1.4, alpha=0.95 if is_goal else 0.75,
                     marker="*" if is_goal else "o", zorder=3)

    stat = f"{n_led_to_shot} of {n_lost} lost second balls -> opponent shot within 10s"
    if n_goals:
        stat += f"  ({n_goals} goal{'s' if n_goals != 1 else ''} conceded)"
    ax.text(0.5, -0.06, stat, transform=ax.transAxes, ha="center", va="top",
           fontsize=10.5, color=palette["ink_secondary"])

    components.header(
        fig, kicker="Crosses -- Second-Ball Counter Risk",
        title=f"{clean_name(team_name)}: {pct:.0f}% of lost cross second balls concede a shot",
        dek="Eredivisie 2025/26  ·  Season  ·  shots shown are the opponent's, attacking this team's goal  ·  "
            "10s transition window",
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
    out_name = args[1] if len(args) > 1 else f"cross_second_ball_counter_risk_{safe_name}.png"

    files = sorted(glob.glob(os.path.join(DATA_DIR, "*.json")))
    if not files:
        raise SystemExit(f"No match files found in {DATA_DIR}")
    team_to_cid = build_team_map(files)
    match = next((full for full in team_to_cid if team_name.lower() in full.lower()), None)
    if match is None:
        raise SystemExit(f"Team '{team_name}' not found. Options: {sorted(team_to_cid)}")
    cid = team_to_cid[match]

    n_lost, n_led_to_shot, shots = collect_counter_shots(files, cid, team_to_cid)
    if n_lost == 0:
        raise SystemExit(f"No lost second-ball contests found for '{match}'")

    for mode in ("light", "dark"):
        out_path = os.path.join(visual_dir(mode), out_name)
        make_plot(match, n_lost, n_led_to_shot, shots, mode, out_path)

    print(f"n_lost={n_lost} led_to_shot={n_led_to_shot} pct={n_led_to_shot / n_lost:.1%}")


if __name__ == "__main__":
    main()

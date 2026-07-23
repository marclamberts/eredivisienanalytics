"""
Pass map: open-play passes switching from one half-space channel straight
into the other (left half-space y in [21, 33) to right half-space
y in (67, 79], or vice versa, on the Opta 100x100 grid) -- diagonal
switches of play that skip the central channel, using the same half-space
boundary already established in PSV Season Report/Scripts/
individual_metric_pitches.py.

Built in the Meridian house style (housestyle package).

Usage: python3 "Box Entry Models/halfspace_to_halfspace_passmap.py" ["Team Name"] [out_dir]
Defaults to Alkmaar Zaanstreek (AZ).
"""
import sys
import re
import glob
import json
import collections
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
from mplsoccer import VerticalPitch

from housestyle import style, components

EVENTS_DIR = Path(__file__).resolve().parent.parent / "Events"

LEFT_HS = {"y_lo": 21.0, "y_hi": 33.0}
RIGHT_HS = {"y_lo": 67.0, "y_hi": 79.0}

NICKNAMES = {
    "Feyenoord Rotterdam": "Feyenoord", "AFC Ajax": "Ajax", "PSV Eindhoven": "PSV",
    "FC Twente": "Twente", "FC Utrecht": "Utrecht", "FC Groningen": "Groningen",
    "FC Volendam": "Volendam", "SC Heerenveen": "Heerenveen", "SC Telstar": "Telstar",
    "SBV Excelsior": "Excelsior", "NAC Breda": "NAC Breda", "PEC Zwolle": "PEC Zwolle",
    "Sparta Rotterdam": "Sparta", "Fortuna Sittard": "Fortuna", "Heracles Almelo": "Heracles",
    "Alkmaar Zaanstreek": "AZ", "Go Ahead Eagles": "Go Ahead Eagles",
    "Nijmegen Eendracht Combinatie": "NEC",
}


def qmap(e):
    return {q["qualifierId"]: q.get("value") for q in e.get("qualifier", []) or []}


def end_xy(e, q):
    x, y = e.get("x"), e.get("y")
    try:
        ex = float(q.get(140, x))
    except (TypeError, ValueError):
        ex = x
    try:
        ey = float(q.get(141, y))
    except (TypeError, ValueError):
        ey = y
    return ex, ey


def build_team_map(files):
    team_cid_sets = collections.defaultdict(list)
    for fn in files:
        m = re.match(r"\d{4}-\d{2}-\d{2}_(.+) - (.+)\.json$", fn.split("/")[-1])
        if not m:
            continue
        home, away = m.group(1), m.group(2)
        with open(fn) as f:
            data = json.load(f)
        cids = set(e["contestantId"] for e in data["event"] if "contestantId" in e)
        team_cid_sets[home].append(cids)
        team_cid_sets[away].append(cids)
    team_to_cid = {}
    for team, sets in team_cid_sets.items():
        inter = set.intersection(*sets)
        if len(inter) == 1:
            team_to_cid[team] = next(iter(inter))
    return team_to_cid


def in_range(v, z):
    return z["y_lo"] <= v < z["y_hi"]


def load_halfspace_switches(cid):
    files = sorted(glob.glob(str(EVENTS_DIR / "*.json")))
    passes = []
    for fn in files:
        with open(fn) as f:
            data = json.load(f)
        for e in data["event"]:
            if e.get("contestantId") != cid or e.get("typeId") != 1:
                continue
            if e.get("periodId") not in (1, 2):
                continue
            x, y = e.get("x"), e.get("y")
            if x is None or y is None:
                continue
            q = qmap(e)
            ex, ey = end_xy(e, q)
            if ex is None or ey is None:
                continue
            left_to_right = in_range(y, LEFT_HS) and in_range(ey, RIGHT_HS)
            right_to_left = in_range(y, RIGHT_HS) and in_range(ey, LEFT_HS)
            if not (left_to_right or right_to_left):
                continue
            passes.append({
                "x0": x, "y0": y, "x1": ex, "y1": ey,
                "player": e.get("playerName", "?"),
                "completed": e.get("outcome") == 1,
            })
    return passes


def main():
    team_name = sys.argv[1] if len(sys.argv) > 1 else "Alkmaar Zaanstreek"
    player_filter = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] else None
    out_dir = Path(sys.argv[3]) if len(sys.argv) > 3 else Path(__file__).parent

    files = sorted(glob.glob(str(EVENTS_DIR / "*.json")))
    team_map = build_team_map(files)
    match = next((full for full in team_map if team_name.lower() in full.lower()), None)
    if match is None:
        print(f"Team '{team_name}' not found. Options: {list(team_map)}")
        sys.exit(1)
    cid = team_map[match]

    passes = load_halfspace_switches(cid)
    if player_filter:
        all_players = sorted({p["player"] for p in passes})
        matched_player = next((p for p in all_players if player_filter.lower() in p.lower()), None)
        if matched_player is None:
            print(f"Player '{player_filter}' not found among {match} half-space switchers. "
                  f"Options: {all_players}")
            sys.exit(1)
        passes = [p for p in passes if p["player"] == matched_player]
        player_filter = matched_player
    completed = [p for p in passes if p["completed"]]

    scope = f"{match}" + (f" / {player_filter}" if player_filter else "")
    print(f"{scope}: {len(passes)} half-space to half-space pass attempts, {len(completed)} completed "
          f"({len(completed) / max(len(passes), 1) * 100:.1f}%).")

    palette, cats = style.apply("light")
    ink_blue = cats[0]

    fig = plt.figure(figsize=(9.6, 9.8))
    fig.patch.set_facecolor(palette["surface"])

    pitch_ax = fig.add_axes([0.05, 0.08, 0.90, 0.68])
    pitch = VerticalPitch(pitch_type="opta", pitch_color=palette["surface"],
                           line_color=palette["axis"], linewidth=1.1, half=False)
    pitch.draw(ax=pitch_ax)

    for hs in (LEFT_HS, RIGHT_HS):
        pitch.polygon([[[0, hs["y_lo"]], [0, hs["y_hi"]], [100, hs["y_hi"]], [100, hs["y_lo"]]]],
                      ax=pitch_ax, facecolor=palette["accent"], edgecolor="none",
                      alpha=0.07, zorder=1)
    # mplsoccer renders low Opta-y on the right and high Opta-y on the left for
    # VerticalPitch, so the on-pitch label text is swapped relative to the
    # LEFT_HS/RIGHT_HS variable names (which are just the two y-bands, named
    # for the pass-direction logic, not for their rendered screen position).
    label_x = 6
    pitch.text(label_x, (LEFT_HS["y_lo"] + LEFT_HS["y_hi"]) / 2, "RIGHT HALF-SPACE",
               ax=pitch_ax, ha="center", va="center", fontsize=8, fontweight="bold",
               color=palette["accent"], family="sans-serif", rotation=90)
    pitch.text(label_x, (RIGHT_HS["y_lo"] + RIGHT_HS["y_hi"]) / 2, "LEFT HALF-SPACE",
               ax=pitch_ax, ha="center", va="center", fontsize=8, fontweight="bold",
               color=palette["accent"], family="sans-serif", rotation=90)

    for p in passes:
        pitch.lines(p["x0"], p["y0"], p["x1"], p["y1"], ax=pitch_ax,
                     color=ink_blue if p["completed"] else palette["ink_muted"],
                     lw=1.6 if p["completed"] else 1.0,
                     alpha=0.55 if p["completed"] else 0.28,
                     zorder=3 if p["completed"] else 2, comet=p["completed"])

    pitch.scatter([p["x1"] for p in completed], [p["y1"] for p in completed], ax=pitch_ax,
                   s=26, color=ink_blue, edgecolors=palette["surface"], linewidths=0.7, zorder=4)
    incomplete = [p for p in passes if not p["completed"]]
    pitch.scatter([p["x1"] for p in incomplete], [p["y1"] for p in incomplete], ax=pitch_ax,
                   s=16, facecolor="none", edgecolors=palette["ink_muted"], linewidths=0.8, zorder=3)

    team_short = NICKNAMES.get(match, match)

    if player_filter:
        kicker = f"{team_short} · {player_filter}"
        title = f"{player_filter} completes {len(completed)} half-space switches this season"
        note = None
        fname_suffix = f"{team_short}_{player_filter}".lower().replace(" ", "_").replace(".", "")
    else:
        top_player = None
        if completed:
            top_player = collections.Counter(p["player"] for p in completed).most_common(1)[0]
        kicker = team_short
        title = f"{team_short} switch play between half-spaces {len(completed)} times this season"
        note = f"Top passer: {top_player[0]} ({top_player[1]})" if top_player else None
        fname_suffix = team_short.lower().replace(" ", "_")

    components.header(
        fig,
        kicker=kicker,
        title=title,
        dek=f"Passes from one half-space straight into the other  ·  "
            f"{len(completed)}/{len(passes)} completed "
            f"({len(completed) / max(len(passes), 1) * 100:.0f}%)  ·  Eredivisie 2025/26",
        palette=palette,
    )
    components.footer(
        fig,
        source="Opta/StatsPerform, Eredivisie 2025/26",
        note=note,
        palette=palette,
    )

    out_path = out_dir / f"halfspace_to_halfspace_passmap_{fname_suffix}.png"
    fig.savefig(out_path, dpi=200, facecolor=palette["surface"])
    plt.close(fig)
    print("Saved:", out_path)


if __name__ == "__main__":
    main()

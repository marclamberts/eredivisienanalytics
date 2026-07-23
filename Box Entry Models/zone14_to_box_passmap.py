"""
Pass map: open-play passes starting in zone 14 (the central channel just
outside the box, x in [66.7, 83), y in [33, 67] on the Opta 100x100 grid)
that reach the penalty box (x >= 83, y in [21.1, 78.9] -- box bounds from
Box Entry Models/shot_hot_zone.json). Broader than
telstar_zone14_to_hotzone_passmap.py, which targets only the small
highest-danger patch inside the box; this one covers box entries from
zone 14 in general.

Built in the Meridian house style (housestyle package).

Usage: python3 "Box Entry Models/zone14_to_box_passmap.py" ["Team Name"] [out_dir]
Defaults to Feyenoord Rotterdam.
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

ZONE14 = {"x_lo": 66.7, "x_hi": 83.0, "y_lo": 33.0, "y_hi": 67.0}


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


def in_zone(x, y, z):
    return z["x_lo"] <= x < z["x_hi"] and z["y_lo"] <= y <= z["y_hi"]


def load_zone14_passes(cid):
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
            if x is None or y is None or not in_zone(x, y, ZONE14):
                continue
            q = qmap(e)
            ex, ey = end_xy(e, q)
            if ex is None or ey is None:
                continue
            passes.append({
                "x0": x, "y0": y, "x1": ex, "y1": ey,
                "player": e.get("playerName", "?"),
                "completed": e.get("outcome") == 1,
            })
    return passes


def main():
    team_name = sys.argv[1] if len(sys.argv) > 1 else "Feyenoord Rotterdam"
    out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(__file__).parent

    with open(Path(__file__).parent / "shot_hot_zone.json") as f:
        meta = json.load(f)
    box = {"x_lo": meta["box_x"], "x_hi": 100.0, "y_lo": meta["box_y_lo"], "y_hi": meta["box_y_hi"]}

    files = sorted(glob.glob(str(EVENTS_DIR / "*.json")))
    team_map = build_team_map(files)
    match = next((full for full in team_map if team_name.lower() in full.lower()), None)
    if match is None:
        print(f"Team '{team_name}' not found. Options: {list(team_map)}")
        sys.exit(1)
    cid = team_map[match]

    passes = load_zone14_passes(cid)
    targeted = [p for p in passes if in_zone(p["x1"], p["y1"], box)]
    landed = [p for p in targeted if p["completed"]]

    print(f"{len(passes)} zone 14 pass attempts, {len(targeted)} aimed at the box, "
          f"{len(landed)} of those completed ({len(landed) / max(len(targeted), 1) * 100:.1f}%).")

    palette, cats = style.apply("light")
    ink_blue = cats[0]

    fig = plt.figure(figsize=(8, 9.6))
    fig.patch.set_facecolor(palette["surface"])

    pitch_ax = fig.add_axes([0.05, 0.10, 0.90, 0.66])
    pitch = VerticalPitch(pitch_type="opta", pitch_color=palette["surface"],
                           line_color=palette["axis"], linewidth=1.1, half=True)
    pitch.draw(ax=pitch_ax)

    pitch.polygon([[[ZONE14["x_lo"], ZONE14["y_lo"]], [ZONE14["x_lo"], ZONE14["y_hi"]],
                    [ZONE14["x_hi"], ZONE14["y_hi"]], [ZONE14["x_hi"], ZONE14["y_lo"]]]],
                  ax=pitch_ax, facecolor="none", edgecolor=palette["ink_muted"],
                  linestyle=(0, (4, 3)), linewidth=1.3, zorder=1)
    pitch.text(ZONE14["x_lo"] - 2.2, (ZONE14["y_lo"] + ZONE14["y_hi"]) / 2,
               "ZONE 14", ax=pitch_ax, ha="center", va="center", fontsize=8.5,
               fontweight="bold", color=palette["ink_muted"], family="sans-serif",
               rotation=90)

    pitch.polygon([[[box["x_lo"], box["y_lo"]], [box["x_lo"], box["y_hi"]],
                    [box["x_hi"], box["y_hi"]], [box["x_hi"], box["y_lo"]]]],
                  ax=pitch_ax, facecolor=palette["accent"], edgecolor=palette["accent"],
                  alpha=0.10, linewidth=1.6, zorder=1)
    pitch.text(box["x_hi"] - 3.5, (box["y_lo"] + box["y_hi"]) / 2,
               "BOX", ax=pitch_ax, ha="center", va="center", fontsize=9,
               fontweight="bold", color=palette["accent"], family="sans-serif",
               rotation=90)

    for p in targeted:
        pitch.lines(p["x0"], p["y0"], p["x1"], p["y1"], ax=pitch_ax,
                     color=ink_blue if p["completed"] else palette["ink_muted"],
                     lw=1.6 if p["completed"] else 1.0,
                     alpha=0.55 if p["completed"] else 0.28,
                     zorder=3 if p["completed"] else 2, comet=p["completed"])

    pitch.scatter([p["x1"] for p in landed], [p["y1"] for p in landed], ax=pitch_ax,
                   s=28, color=ink_blue, edgecolors=palette["surface"],
                   linewidths=0.7, zorder=4)
    incomplete = [p for p in targeted if not p["completed"]]
    pitch.scatter([p["x1"] for p in incomplete], [p["y1"] for p in incomplete], ax=pitch_ax,
                   s=18, facecolor="none", edgecolors=palette["ink_muted"],
                   linewidths=0.8, zorder=3)

    top_player = None
    if landed:
        top_player = collections.Counter(p["player"] for p in landed).most_common(1)[0]

    NICKNAMES = {
        "Feyenoord Rotterdam": "Feyenoord", "AFC Ajax": "Ajax", "PSV Eindhoven": "PSV",
        "FC Twente": "Twente", "FC Utrecht": "Utrecht", "FC Groningen": "Groningen",
        "FC Volendam": "Volendam", "SC Heerenveen": "Heerenveen", "SC Telstar": "Telstar",
        "SBV Excelsior": "Excelsior", "NAC Breda": "NAC Breda", "PEC Zwolle": "PEC Zwolle",
        "Sparta Rotterdam": "Sparta", "Fortuna Sittard": "Fortuna", "Heracles Almelo": "Heracles",
        "Alkmaar Zaanstreek": "AZ", "Go Ahead Eagles": "Go Ahead Eagles",
        "Nijmegen Eendracht Combinatie": "NEC",
    }
    team_short = NICKNAMES.get(match, match)
    components.header(
        fig,
        kicker=team_short,
        title=f"{team_short} complete {len(landed)} of {len(targeted)} zone 14 passes into the box",
        dek=f"Zone 14 passes reaching the box  ·  "
            f"{len(landed) / max(len(targeted), 1) * 100:.0f}% completion  ·  Eredivisie 2025/26",
        palette=palette,
    )
    note = f"Top target: {top_player[0]} ({top_player[1]})" if top_player else None
    components.footer(
        fig,
        source="Opta/StatsPerform, Eredivisie 2025/26",
        note=note,
        palette=palette,
    )

    out_path = out_dir / f"zone14_to_box_passmap_{team_short.lower().replace(' ', '_')}.png"
    fig.savefig(out_path, dpi=200, facecolor=palette["surface"])
    plt.close(fig)
    print("Saved:", out_path)


if __name__ == "__main__":
    main()

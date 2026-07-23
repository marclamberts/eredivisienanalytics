"""
Cutbacks, PSV Eindhoven vs. Feyenoord Rotterdam, Eredivisie 2025/26.

A cutback here is an open-play pass that starts deep and wide (x >= 83,
y <= 21 or y >= 79 on the Opta 100x100 grid — level with or past the
byline) and is pulled back into the central channel of the box (end
x >= 83, end y in [33, 67]), tagged as a cross either directly (Opta
qualifier 2) or by the same wide/box-entry geometry used for the
progressive-pass families in individual_metric_pitches.py. Provider
qualifier 195 ("pull back") is present in the schema but never populated
in this feed (see Cross Models/score_eredivisie_crosses.py), so the
geometric definition is used instead, extended here to include
unsuccessful attempts alongside completions.

Built in the Meridian house style (housestyle package).

Usage: python3 "Cross Models/cutbacks_psv_vs_feyenoord.py" [out_dir]
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
CROSS_QID = 2

TEAMS = ["PSV Eindhoven", "Feyenoord Rotterdam"]


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


def load_cutbacks(files, cid):
    recs = []
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
            is_wide_start = y <= 21 or y >= 79
            is_box_end = ex >= 83 and 21 <= ey <= 79
            is_cross = CROSS_QID in q or (x >= 55 and is_wide_start and is_box_end and abs(ey - y) >= 12)
            if is_cross and x >= 83 and is_wide_start and ex >= 83 and 33 <= ey <= 67:
                recs.append({
                    "x0": x, "y0": y, "x1": ex, "y1": ey,
                    "player": e.get("playerName", "?"),
                    "completed": e.get("outcome") == 1,
                })
    return recs


def draw_panel(fig, left, cutbacks, team, palette, accent_team, ink_blue):
    is_accent = team == accent_team
    mark_color = palette["accent"] if is_accent else ink_blue
    completed = [c for c in cutbacks if c["completed"]]

    pitch_ax = fig.add_axes([left, 0.06, 0.42, 0.60])
    pitch = VerticalPitch(pitch_type="opta", pitch_color=palette["surface"],
                           line_color=palette["axis"], linewidth=1.1, half=True)
    pitch.draw(ax=pitch_ax)

    for c in completed:
        pitch.lines(c["x0"], c["y0"], c["x1"], c["y1"], ax=pitch_ax,
                     color=mark_color, lw=1.6, alpha=0.55, zorder=3, comet=True)

    pitch.scatter([c["x1"] for c in completed], [c["y1"] for c in completed],
                   ax=pitch_ax, s=26, color=mark_color, edgecolors=palette["surface"],
                   linewidths=0.7, zorder=4)

    rate = len(completed) / len(cutbacks) * 100 if cutbacks else 0
    fig.text(left + 0.21, 0.795, team, ha="center", fontsize=13.5, fontweight="bold",
              color=palette["ink_primary"], family="sans-serif")
    fig.text(left + 0.21, 0.755, f"{len(completed)} completed", ha="center", fontsize=20,
              fontweight="bold", color=mark_color, family="sans-serif")
    fig.text(left + 0.21, 0.725, f"of {len(cutbacks)} attempts  ·  {rate:.0f}% complete",
              ha="center", fontsize=9.5, color=palette["ink_secondary"], family="sans-serif")

    return len(completed), len(cutbacks)


def main():
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent

    files = sorted(glob.glob(str(EVENTS_DIR / "*.json")))
    team_map = build_team_map(files)

    data = {t: load_cutbacks(files, team_map[t]) for t in TEAMS}
    completed_counts = {t: sum(c["completed"] for c in data[t]) for t in TEAMS}
    leader = max(completed_counts, key=completed_counts.get)
    trailer = [t for t in TEAMS if t != leader][0]
    diff = completed_counts[leader] - completed_counts[trailer]
    pct = (completed_counts[leader] / completed_counts[trailer] - 1) * 100 if completed_counts[trailer] else 0

    print({t: (completed_counts[t], len(data[t])) for t in TEAMS})

    palette, cats = style.apply("light")
    ink_blue = cats[0]

    fig = plt.figure(figsize=(13, 9.2))
    fig.patch.set_facecolor(palette["surface"])

    draw_panel(fig, 0.03, data[leader], leader, palette, leader, ink_blue)
    draw_panel(fig, 0.53, data[trailer], trailer, palette, leader, ink_blue)

    components.header(
        fig,
        kicker="Cross Models",
        title=f"{leader} complete {diff} more cutbacks than {trailer.split()[0]} this season",
        dek=f"Open-play cutbacks from the byline into the central box, all fixtures  ·  "
            f"{leader} +{pct:.0f}% on completions  ·  Eredivisie 2025/26",
        palette=palette,
    )
    components.footer(
        fig,
        source="Opta/StatsPerform, Eredivisie 2025/26",
        palette=palette,
    )

    out_path = out_dir / "cutbacks_psv_vs_feyenoord_eredivisie.png"
    fig.savefig(out_path, dpi=200, facecolor=palette["surface"])
    plt.close(fig)
    print("Saved:", out_path)


if __name__ == "__main__":
    main()

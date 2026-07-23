"""
Pass sonar ("pass patterns"), Feyenoord vs. Ajax (2026-03-22, single
match). The pitch is split into a 6x4 grid; each cell shows a small wind
rose of that team's completed passes originating there, direction binned
into 8 compass sectors (0 deg = towards the attacking goal, since x is
already normalized so a team always attacks increasing x), petal length
= share of that cell's passes going that way.

Built in the Meridian house style (housestyle package).

Usage: python3 Danger/pass_sonar_feyenoord_ajax.py [out_dir]
"""
import sys
import json
import collections
import math
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
from mplsoccer import VerticalPitch

from housestyle import style, components

EVENTS_DIR = Path(__file__).resolve().parent.parent / "Events"
MATCH_FILE = EVENTS_DIR / "2026-03-22_Feyenoord Rotterdam - AFC Ajax.json"

TEAMS = [("Feyenoord Rotterdam", "20vymiy7bo8wkyxai3ew494fz"),
         ("AFC Ajax", "d0zdg647gvgc95xdtk1vpbkys")]

X_BINS, Y_BINS = 6, 4
N_DIRECTIONS = 8
MIN_PASSES_PER_CELL = 3


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


def load_completed_passes(data, cid):
    recs = []
    for e in data["event"]:
        if e.get("typeId") != 1 or e.get("contestantId") != cid:
            continue
        if e.get("periodId") not in (1, 2) or e.get("outcome") != 1:
            continue
        x, y = e.get("x"), e.get("y")
        if x is None or y is None:
            continue
        q = qmap(e)
        ex, ey = end_xy(e, q)
        if ex is None or ey is None:
            continue
        recs.append((x, y, ex, ey))
    return recs


def cell_of(x, y):
    xi = min(int(x / 100 * X_BINS), X_BINS - 1)
    yi = min(int(y / 100 * Y_BINS), Y_BINS - 1)
    return xi, yi


def wedge_polygon(cx, cy, r, theta1, theta2, n=6):
    if r <= 0:
        return None
    pts = [(cx, cy)]
    for i in range(n + 1):
        a = theta1 + (theta2 - theta1) * i / n
        pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    pts.append((cx, cy))
    return pts


def draw_sonar_panel(fig, left, passes, team_name, palette, mark_color):
    ax = fig.add_axes([left, 0.06, 0.44, 0.62])
    pitch = VerticalPitch(pitch_type="opta", pitch_color=palette["surface"],
                           line_color=palette["axis"], linewidth=1.1, half=False)
    pitch.draw(ax=ax)
    ax.set_title(f"{team_name}  ·  {len(passes)} completed passes", fontsize=12,
                 fontweight="bold", color=palette["ink_primary"], pad=8, family="sans-serif")

    cell_dirs = collections.defaultdict(lambda: collections.Counter())
    for x, y, ex, ey in passes:
        xi, yi = cell_of(x, y)
        angle = math.atan2(ey - y, ex - x)
        bin_idx = int(((angle + math.pi) / (2 * math.pi) * N_DIRECTIONS) + 0.5) % N_DIRECTIONS
        cell_dirs[(xi, yi)][bin_idx] += 1

    max_count = max((c for counts in cell_dirs.values() for c in counts.values()), default=1)
    cell_w, cell_h = 100 / X_BINS, 100 / Y_BINS
    max_r = min(cell_w, cell_h) * 0.44
    sector = 2 * math.pi / N_DIRECTIONS

    for (xi, yi), counts in cell_dirs.items():
        total = sum(counts.values())
        if total < MIN_PASSES_PER_CELL:
            continue
        cx, cy = (xi + 0.5) * cell_w, (yi + 0.5) * cell_h
        for bin_idx, count in counts.items():
            theta_center = bin_idx / N_DIRECTIONS * 2 * math.pi - math.pi
            r = max_r * (count / max_count) ** 0.5
            poly = wedge_polygon(cx, cy, r, theta_center - sector / 2, theta_center + sector / 2)
            if poly is None:
                continue
            pitch.polygon([poly], ax=ax, facecolor=mark_color, edgecolor=palette["surface"],
                          linewidth=0.5, alpha=0.75, zorder=3)
        pitch.scatter([cx], [cy], ax=ax, s=8, color=palette["ink_primary"], zorder=4)


def main():
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent

    with open(MATCH_FILE) as f:
        data = json.load(f)

    palette, cats = style.apply("light")
    ink_blue, teal = cats[0], cats[2]
    short = {"Feyenoord Rotterdam": "Feyenoord", "AFC Ajax": "Ajax"}

    fig = plt.figure(figsize=(13, 8.8))
    fig.patch.set_facecolor(palette["surface"])

    counts = {}
    for i, ((team_name, cid), color) in enumerate(zip(TEAMS, [ink_blue, teal])):
        passes = load_completed_passes(data, cid)
        counts[team_name] = len(passes)
        draw_sonar_panel(fig, 0.03 + i * 0.50, passes, short[team_name], palette, color)

    components.header(
        fig,
        kicker="Match report · Pass patterns",
        title="Where Feyenoord and Ajax directed their passing from every zone",
        dek=f"Pass sonar: petal length = share of a cell's completed passes in that direction, "
            f"{X_BINS}x{Y_BINS} grid  ·  Eredivisie 2025/26, 22 Mar 2026",
        palette=palette,
    )
    components.footer(
        fig,
        source="Opta/StatsPerform, Eredivisie 2025/26",
        note=f"cells with fewer than {MIN_PASSES_PER_CELL} passes omitted",
        palette=palette,
    )

    out_path = out_dir / "pass_sonar_feyenoord_ajax.png"
    fig.savefig(out_path, dpi=200, facecolor=palette["surface"])
    plt.close(fig)
    print(counts)
    print("Saved:", out_path)


if __name__ == "__main__":
    main()

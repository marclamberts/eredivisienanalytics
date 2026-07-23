"""
Progressive passes, Feyenoord vs. Ajax (2026-03-22, single match). A pass
is progressive if it gains >=10m towards goal, or starts in the defensive
half and reaches the middle third, or starts in the middle third and
reaches the final third -- same definition as PSV Season Report/Scripts/
individual_metric_pitches.py, applied to just this one match's events.

Built in the Meridian house style (housestyle package).

Usage: python3 Danger/progressive_passes_feyenoord_ajax.py [out_dir]
"""
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
from mplsoccer import VerticalPitch

from housestyle import style, components

EVENTS_DIR = Path(__file__).resolve().parent.parent / "Events"
MATCH_FILE = EVENTS_DIR / "2026-03-22_Feyenoord Rotterdam - AFC Ajax.json"

TEAMS = [("Feyenoord Rotterdam", "20vymiy7bo8wkyxai3ew494fz"),
         ("AFC Ajax", "d0zdg647gvgc95xdtk1vpbkys")]


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


def load_progressive(data, cid):
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
        dx = ex - x
        if dx >= 10 or (x < 50 and ex >= 66.7) or (x < 66.7 and ex >= 83):
            recs.append({"x0": x, "y0": y, "x1": ex, "y1": ey,
                         "player": e.get("playerName", "?")})
    return recs


def draw_panel(fig, left, recs, team_name, palette, mark_color):
    ax = fig.add_axes([left, 0.06, 0.44, 0.62])
    pitch = VerticalPitch(pitch_type="opta", pitch_color=palette["surface"],
                           line_color=palette["axis"], linewidth=1.1, half=False)
    pitch.draw(ax=ax)
    ax.set_title(f"{team_name}  ·  {len(recs)} progressive passes", fontsize=12,
                 fontweight="bold", color=palette["ink_primary"], pad=8, family="sans-serif")

    for r in recs:
        pitch.lines(r["x0"], r["y0"], r["x1"], r["y1"], ax=ax, color=mark_color,
                     lw=1.4, alpha=0.55, zorder=2, comet=True)
    pitch.scatter([r["x1"] for r in recs], [r["y1"] for r in recs], ax=ax,
                   s=20, color=mark_color, edgecolors=palette["surface"], linewidths=0.6, zorder=3)


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
        recs = load_progressive(data, cid)
        counts[team_name] = len(recs)
        draw_panel(fig, 0.03 + i * 0.50, recs, short[team_name], palette, color)

    fey_n, aja_n = counts[TEAMS[0][0]], counts[TEAMS[1][0]]
    leader = short[TEAMS[0][0]] if fey_n > aja_n else short[TEAMS[1][0]]
    diff = abs(fey_n - aja_n)

    components.header(
        fig,
        kicker="Match report · Progressive passes",
        title=f"{leader} completed {diff} more progressive passes than their opponent",
        dek=f">=10m gained, or defensive to middle third, or middle to final third  ·  "
            f"Feyenoord {fey_n}, Ajax {aja_n}  ·  Eredivisie 2025/26, 22 Mar 2026",
        palette=palette,
    )
    components.footer(
        fig,
        source="Opta/StatsPerform, Eredivisie 2025/26",
        palette=palette,
    )

    out_path = out_dir / "progressive_passes_feyenoord_ajax.png"
    fig.savefig(out_path, dpi=200, facecolor=palette["surface"])
    plt.close(fig)
    print(f"Feyenoord {fey_n}, Ajax {aja_n}")
    print("Saved:", out_path)


if __name__ == "__main__":
    main()

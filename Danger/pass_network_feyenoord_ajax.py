"""
Single-match pass network: Feyenoord Rotterdam vs. AFC Ajax.
Nodes = average touch position (pass origin + receiving end) for each
starting-XI player, sized by pass volume. Edges = combined completed-pass
count between two starters, next-touch heuristic (same convention as
PSV Season Report/Scripts/pass_network.py, scoped to one match instead of
a full season). Substitutes are excluded, matching standard pass-network
practice; a starter's own data simply stops once Opta records their
substitution, so no explicit time cutoff is needed.

Built in the Meridian house style (housestyle package).

Usage: python3 Danger/pass_network_feyenoord_ajax.py [out_dir]
"""
import sys
import glob
import json
import collections
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from mplsoccer import VerticalPitch
from adjustText import adjust_text

from housestyle import style, components

EVENTS_DIR = Path(__file__).resolve().parent.parent / "Events"
MATCH_FILE = EVENTS_DIR / "2026-03-22_Feyenoord Rotterdam - AFC Ajax.json"
MIN_EDGE_PASSES = 3

TEAMS = [
    ("Feyenoord Rotterdam", "20vymiy7bo8wkyxai3ew494fz"),
    ("AFC Ajax", "d0zdg647gvgc95xdtk1vpbkys"),
]


def qmap(e):
    return {q["qualifierId"]: q.get("value") for q in e.get("qualifier", []) or []}


def starting_xi(data, cid):
    for e in data["event"]:
        if e.get("typeId") == 34 and e.get("contestantId") == cid:
            q = qmap(e)
            ids = (q.get(30) or "").split(", ")
            codes = (q.get(44) or "").split(", ")
            return {pid for pid, code in zip(ids, codes) if code in ("1", "2", "3", "4")}
    return set()


def formation(data, cid):
    for e in data["event"]:
        if e.get("typeId") == 34 and e.get("contestantId") == cid:
            q = qmap(e)
            codes = (q.get(44) or "").split(", ")
            on = [c for c in codes if c in ("1", "2", "3", "4")]
            d, m, f = on.count("2"), on.count("3"), on.count("4")
            return f"{d}-{m}-{f}"
    return None


def collect_network(data, cid, xi):
    positions = collections.defaultdict(list)
    pass_counts = collections.Counter()
    edges = collections.Counter()

    evs = [e for e in data["event"] if e.get("periodId") in (1, 2)]
    evs.sort(key=lambda e: (e["periodId"], e.get("timeMin", 0), e.get("timeSec", 0), e.get("eventId", 0)))

    for i, e in enumerate(evs):
        if e.get("typeId") != 1 or e.get("contestantId") != cid:
            continue
        pid = e.get("playerId")
        if pid not in xi:
            continue
        positions[pid].append((e["x"], e["y"]))
        pass_counts[pid] += 1
        if e.get("outcome") != 1:
            continue
        if i + 1 >= len(evs):
            continue
        nxt = evs[i + 1]
        if nxt.get("contestantId") != cid:
            continue
        rpid = nxt.get("playerId")
        if rpid is None or rpid == pid or rpid not in xi:
            continue
        q = qmap(e)
        ex = float(q.get(140, e["x"]))
        ey = float(q.get(141, e["y"]))
        positions[rpid].append((ex, ey))
        pair = tuple(sorted((pid, rpid)))
        edges[pair] += 1

    return positions, edges, pass_counts


def draw_panel(fig, left, data, team_name, cid, names, palette, mark_color):
    xi = starting_xi(data, cid)
    positions, edges, pass_counts = collect_network(data, cid, xi)
    shape = formation(data, cid)

    ax = fig.add_axes([left, 0.06, 0.44, 0.72])
    pitch = VerticalPitch(pitch_type="opta", pitch_color=palette["surface"],
                           line_color=palette["axis"], linewidth=1.1, half=False)
    pitch.draw(ax=ax)

    label = team_name if not shape else f"{team_name}  ·  {shape}"
    ax.set_title(label, fontsize=12.5, fontweight="bold", color=palette["ink_primary"], pad=8,
                 family="sans-serif")

    avg_pos = {}
    for pid in xi:
        pts = positions.get(pid, [])
        if not pts:
            continue
        avg_pos[pid] = (sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts))

    max_passes = max(pass_counts.values(), default=1)
    max_edge = max(edges.values(), default=1)

    for (a, b), cnt in edges.items():
        if cnt < MIN_EDGE_PASSES or a not in avg_pos or b not in avg_pos:
            continue
        x0, y0 = avg_pos[a]
        x1, y1 = avg_pos[b]
        lw = 0.7 + (cnt / max_edge) * 5.0
        alpha = 0.22 + (cnt / max_edge) * 0.5
        pitch.lines(x0, y0, x1, y1, ax=ax, color=mark_color, lw=lw, alpha=alpha, zorder=2, comet=False)

    texts = []
    for pid, (x, y) in avg_pos.items():
        n_passes = pass_counts.get(pid, 0)
        size = 180 + (n_passes / max_passes) * 950
        pitch.scatter(x, y, ax=ax, s=size, color=mark_color, edgecolors=palette["surface"],
                       linewidths=1.6, zorder=3, alpha=0.95)
        surname = names.get(pid, "?").split(" ")[-1]
        ann = pitch.annotate(
            surname, xy=(x, y), ax=ax, ha="center", va="center",
            fontsize=8.2, fontweight="bold", color=palette["ink_primary"], zorder=5, clip_on=False,
            path_effects=[pe.withStroke(linewidth=2.6, foreground=palette["surface"])],
            arrowprops=dict(arrowstyle="-", color=palette["ink_muted"], lw=0.7, alpha=0.85,
                             shrinkA=0, shrinkB=4),
        )
        texts.append(ann)

    adjust_text(texts, ax=ax, expand=(1.2, 1.3),
                force_text=(0.4, 0.6), force_static=(0.3, 0.4),
                only_move={"text": "xy", "static": "xy", "explode": "xy", "pull": "xy"})

    completed = sum(1 for e in data["event"] if e.get("typeId") == 1 and e.get("contestantId") == cid
                     and e.get("playerId") in xi and e.get("outcome") == 1)
    return completed


def main():
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent

    with open(MATCH_FILE) as f:
        data = json.load(f)
    scores = data["matchDetails"]["scores"]["ft"]

    names = {}
    for e in data["event"]:
        if e.get("playerId") and e.get("playerName"):
            names[e["playerId"]] = e["playerName"]

    palette, cats = style.apply("light")
    ink_blue = cats[0]

    fig = plt.figure(figsize=(13, 9.4))
    fig.patch.set_facecolor(palette["surface"])

    completed = {}
    completed[TEAMS[0][0]] = draw_panel(fig, 0.03, data, TEAMS[0][0], TEAMS[0][1], names,
                                          palette, palette["accent"])
    completed[TEAMS[1][0]] = draw_panel(fig, 0.53, data, TEAMS[1][0], TEAMS[1][1], names,
                                          palette, ink_blue)

    short = {"Feyenoord Rotterdam": "Feyenoord", "AFC Ajax": "Ajax"}
    diff = completed[TEAMS[0][0]] - completed[TEAMS[1][0]]
    leader, trailer = (TEAMS[0][0], TEAMS[1][0]) if diff >= 0 else (TEAMS[1][0], TEAMS[0][0])
    diff = abs(diff)

    components.header(
        fig,
        kicker="Match report",
        title=f"{short[leader]} completed {diff} more passes than {short[trailer]} in a "
              f"{scores['home']}-{scores['away']} draw",
        dek="Starting-XI pass network, full match  ·  node size = passes, line width = combinations  ·  "
            "Eredivisie 2025/26, 22 Mar 2026",
        palette=palette,
    )
    components.footer(
        fig,
        source="Opta/StatsPerform, Eredivisie 2025/26",
        note="edge = completed pass between two starters, next-touch heuristic, min 3 combined",
        palette=palette,
    )

    out_path = out_dir / "pass_network_feyenoord_ajax.png"
    fig.savefig(out_path, dpi=200, facecolor=palette["surface"])
    plt.close(fig)
    print(completed)
    print("Saved:", out_path)


if __name__ == "__main__":
    main()

"""
Pass-network clusters for the Feyenoord vs. Ajax match: same starting-XI
pass network as pass_network_feyenoord_ajax.py, with players grouped into
their most common pass-combination clusters via weighted greedy-modularity
community detection (same method as PSV Season Report/Scripts/
pass_network_clusters.py). Shaded hulls and matching node/edge colour mark
each recurring passing unit (e.g. back line + keeper vs. midfield/attack)
so the clusters stand out instead of one undifferentiated web of lines.

Built in the Meridian house style (housestyle package).

Usage: python3 Danger/pass_network_clusters_feyenoord_ajax.py [out_dir]
"""
import sys
import json
import collections
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import networkx as nx
from mplsoccer import VerticalPitch
from adjustText import adjust_text

from housestyle import style, components
from pass_network_feyenoord_ajax import (
    MATCH_FILE, TEAMS, MIN_EDGE_PASSES,
    starting_xi, formation, collect_network,
)


def detect_clusters(edges, node_ids):
    graph = nx.Graph()
    graph.add_nodes_from(node_ids)
    for (a, b), cnt in edges.items():
        if cnt >= MIN_EDGE_PASSES and a in node_ids and b in node_ids:
            graph.add_edge(a, b, weight=cnt)
    communities = nx.algorithms.community.greedy_modularity_communities(graph, weight="weight")
    cluster_of = {}
    for idx, comm in enumerate(communities):
        for pid in comm:
            cluster_of[pid] = idx
    return cluster_of, len(communities)


def draw_panel(fig, left, data, team_name, cid, names, palette, cluster_colors):
    xi = starting_xi(data, cid)
    positions, edges, pass_counts = collect_network(data, cid, xi)
    shape = formation(data, cid)

    ax = fig.add_axes([left, 0.06, 0.44, 0.72])
    pitch = VerticalPitch(pitch_type="opta", pitch_color=palette["surface"],
                           line_color=palette["axis"], linewidth=1.1, half=False)
    pitch.draw(ax=ax)

    avg_pos = {}
    for pid in xi:
        pts = positions.get(pid, [])
        if not pts:
            continue
        avg_pos[pid] = (sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts))

    cluster_of, n_clusters = detect_clusters(edges, set(avg_pos))
    cluster_members = collections.defaultdict(list)
    for pid in avg_pos:
        cluster_members[cluster_of.get(pid, -1)].append(pid)

    label = team_name if not shape else f"{team_name}  ·  {shape}  ·  {n_clusters} clusters"
    ax.set_title(label, fontsize=12.5, fontweight="bold", color=palette["ink_primary"], pad=8,
                 family="sans-serif")

    for cl_idx, members in cluster_members.items():
        if len(members) < 3:
            continue
        color = cluster_colors[cl_idx % len(cluster_colors)]
        xs = [avg_pos[p][0] for p in members]
        ys = [avg_pos[p][1] for p in members]
        hull = pitch.convexhull(xs, ys)
        pitch.polygon(hull, ax=ax, facecolor=color, edgecolor=color, alpha=0.14, lw=1.2, zorder=1)

    max_passes = max(pass_counts.values(), default=1)
    max_edge = max(edges.values(), default=1)

    for (a, b), cnt in edges.items():
        if cnt < MIN_EDGE_PASSES or a not in avg_pos or b not in avg_pos:
            continue
        x0, y0 = avg_pos[a]
        x1, y1 = avg_pos[b]
        lw = 0.7 + (cnt / max_edge) * 5.0
        same_cluster = cluster_of.get(a) == cluster_of.get(b)
        color = cluster_colors[cluster_of.get(a, 0) % len(cluster_colors)] if same_cluster else palette["ink_muted"]
        alpha = (0.3 + (cnt / max_edge) * 0.55) if same_cluster else (0.14 + (cnt / max_edge) * 0.16)
        pitch.lines(x0, y0, x1, y1, ax=ax, color=color, lw=lw, alpha=alpha, zorder=2, comet=False)

    texts = []
    for pid, (x, y) in avg_pos.items():
        n_passes = pass_counts.get(pid, 0)
        size = 180 + (n_passes / max_passes) * 950
        node_color = cluster_colors[cluster_of.get(pid, 0) % len(cluster_colors)]
        pitch.scatter(x, y, ax=ax, s=size, color=node_color, edgecolors=palette["surface"],
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

    return n_clusters, cluster_members, names


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
    # exclude cats[1] (terracotta) -- reserved as the fixed-chrome accent, not a cluster colour
    cluster_colors = [cats[0], cats[2], cats[3], cats[4], cats[5], cats[6], cats[7]]

    fig = plt.figure(figsize=(13, 9.4))
    fig.patch.set_facecolor(palette["surface"])

    short = {"Feyenoord Rotterdam": "Feyenoord", "AFC Ajax": "Ajax"}
    results = {}
    for i, (team_name, cid) in enumerate(TEAMS):
        left = 0.03 + i * 0.50
        n_clusters, cluster_members, _ = draw_panel(fig, left, data, team_name, cid, names,
                                                      palette, cluster_colors)
        results[team_name] = n_clusters

    fey_n, aja_n = results["Feyenoord Rotterdam"], results["AFC Ajax"]
    if fey_n == aja_n:
        title = f"Feyenoord and Ajax each split into {fey_n} passing clusters"
    else:
        title = f"Feyenoord's passing split into {fey_n} clusters, Ajax's into {aja_n}"
    components.header(
        fig,
        kicker="Match report",
        title=title,
        dek=f"Starting-XI pass network with greedy-modularity clusters, full match  ·  "
            f"shaded hull = recurring passing unit  ·  Eredivisie 2025/26, 22 Mar 2026  ·  "
            f"{scores['home']}-{scores['away']}",
        palette=palette,
    )
    components.footer(
        fig,
        source="Opta/StatsPerform, Eredivisie 2025/26",
        note="clusters = greedy-modularity community detection on completed-pass counts, min 3 combined",
        palette=palette,
    )

    out_path = out_dir / "pass_network_clusters_feyenoord_ajax.png"
    fig.savefig(out_path, dpi=200, facecolor=palette["surface"])
    plt.close(fig)
    print(results)
    print("Saved:", out_path)


if __name__ == "__main__":
    main()

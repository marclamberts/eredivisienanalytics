"""
Positionist vs Relationist build-up style contrast.

"Positional play" (juego de posicion) and "relationism" are opposing
tactical philosophies: positionism holds players in separated zones/lanes
and progresses the ball between them (structured, wider passing distances);
relationism clusters players near the ball for close combination play
(one-twos, third-man runs, tight passing distances), prioritising proximity
over fixed zonal occupation.

This is NOT a validated tactical classifier - there's no ground-truth label
for "style" in the event data, only a transparent proxy: for every
qualifying build-up sequence (>=3 passes, any starting third, ending in a
shot - same pool as buildup_patterns.py, just not restricted to one third),
take the mean distance between consecutive touches (passes + carries).
Sequences below that team's own median distance are labelled Relationist
(tight, proximity-based combinations); sequences above are labelled
Positionist (longer, zone-to-zone progression). The most representative
example of each half (median pass-count, preferring a goal) is drawn, same
selection rule as buildup_patterns.py's pick_representative().

Usage: python3 style_patterns.py "Nijmegen Eendracht Combinatie" [out.png]
"""
import glob
import os
import statistics
import sys

import matplotlib.pyplot as plt
from mplsoccer import VerticalPitch

from buildup_patterns import (
    DATA_DIR, BG, PITCH_LINE, C_AMBER, C_GREEN, SHOT_NAMES, MIN_PASSES,
    build_team_map, collect_sequences, sequence_route, pick_representative,
    clean_name, add_logo,
)

STYLE_COLORS = {"Positionist": "#2f8fd1", "Relationist": "#f06fa3"}
STYLE_BLURB = {
    "Positionist": "Longer action distances: ball moved between players holding separated zones",
    "Relationist": "Shorter action distances: ball moved between players clustered close together",
}


def action_distances(item):
    points, _ = sequence_route(item["seq"])
    dists = []
    for i in range(len(points) - 1):
        (x0, y0), (x1, y1) = points[i], points[i + 1]
        dists.append(((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5)
    return dists


def classify_by_style(qualifying):
    for item in qualifying:
        dists = action_distances(item)
        item["_mean_dist"] = sum(dists) / len(dists) if dists else 0.0

    median = statistics.median(i["_mean_dist"] for i in qualifying)
    relationist = [i for i in qualifying if i["_mean_dist"] <= median]
    positionist = [i for i in qualifying if i["_mean_dist"] > median]
    return relationist, positionist, median


def make_plot(team_name, categories, total_n, median_dist, out_path):
    fig = plt.figure(figsize=(13, 12.5))
    fig.patch.set_facecolor(BG)

    fig.text(0.03, 0.955, clean_name(team_name), fontsize=27, fontweight="bold", color="white")
    fig.text(0.03, 0.918, "Positionist vs Relationist  ·  Build-Up Style Contrast  ·  Eredivisie 2025/26  ·  Season",
             fontsize=13.5, fontweight="bold", color="#c7ccd4")
    fig.text(0.03, 0.892, f"{total_n} qualifying sequences this season (uninterrupted possession, "
             f"≥{MIN_PASSES} passes, any starting third, ending in a shot)  ·  "
             f"team median action distance: {median_dist:.1f} (Opta units)",
             fontsize=10, color="#9aa4b2")

    n_panels = len(categories)
    axes = [fig.add_axes([0.02 + i * (0.96 / n_panels), 0.05, 0.96 / n_panels - 0.02, 0.72])
            for i in range(n_panels)]

    for ax, cat in zip(axes, categories):
        pitch = VerticalPitch(pitch_type="opta", pitch_color=BG, line_color=PITCH_LINE,
                               linewidth=1.1, half=False)
        pitch.draw(ax=ax)

        item = cat["example"]
        points, players = sequence_route(item["seq"])
        n = len(points)
        for i in range(n - 1):
            x0, y0 = points[i]
            x1, y1 = points[i + 1]
            is_shot_leg = (i == n - 2)
            color = C_GREEN if (is_shot_leg and item["shot"]["typeId"] == 16) else C_AMBER
            pitch.lines(x0, y0, x1, y1, ax=ax, color=color, lw=2.6 if is_shot_leg else 2.0,
                       alpha=0.95, zorder=2, comet=True,
                       transparent=True if not is_shot_leg else False)

        for i, (x, y) in enumerate(points):
            is_last = i == n - 1
            marker_color = C_GREEN if (is_last and item["shot"]["typeId"] == 16) else C_AMBER
            size = 340 if is_last else 200
            marker = "*" if is_last else "o"
            pitch.scatter(x, y, ax=ax, s=size if marker == "o" else size * 2.2, color=marker_color,
                          edgecolors=BG, linewidths=1.6, zorder=4, marker=marker)
            if not is_last:
                pitch.annotate(str(i + 1), xy=(x, y), ax=ax, ha="center", va="center",
                               fontsize=8.5, fontweight="bold", color=BG, zorder=5)

        shot_label = SHOT_NAMES[item["shot"]["typeId"]]
        shooter = item["shot"].get("playerName", "?")
        x_center = ax.get_position().x0 + ax.get_position().width / 2
        fig.text(x_center, 0.825, cat["name"], fontsize=17, fontweight="bold", ha="center",
                 color=STYLE_COLORS[cat["name"]], transform=fig.transFigure)
        fig.text(x_center, 0.798, STYLE_BLURB[cat["name"]], fontsize=8.7, ha="center",
                 color="#9aa4b2", transform=fig.transFigure, style="italic")
        fig.text(x_center, 0.768, f"{item['n_passes']} passes · avg action distance {item['_mean_dist']:.1f} "
                 f"· shot by {shooter} · {shot_label}",
                 fontsize=10, ha="center", color="#e6e9ee", transform=fig.transFigure)

    fig.text(0.03, 0.014, "Data via Opta | Eredivisie 2025/26 event data · numbers = touch sequence order · "
             "★ = shot location · style is a distance-based proxy, not a validated tactical label",
             fontsize=8, color="#6b7684")
    fig.text(0.98, 0.014, "Marc Lamberts · Waltzing Analytics", fontsize=9.5, ha="right",
             color="#6b7684", style="italic")

    add_logo(fig)
    fig.savefig(out_path, dpi=200, facecolor=BG)
    print("Saved:", out_path)


if __name__ == "__main__":
    team_name = sys.argv[1] if len(sys.argv) > 1 else "Nijmegen Eendracht Combinatie"
    out = sys.argv[2] if len(sys.argv) > 2 else f"Visuals/BuildUp/{team_name.replace(' ', '_')}_style_patterns.png"

    os.makedirs(os.path.dirname(out), exist_ok=True)

    files = sorted(glob.glob(f"{DATA_DIR}/*.json"))
    team_to_cid = build_team_map(files)
    match = next((full for full in team_to_cid if team_name.lower() in full.lower()), None)
    if match is None:
        print(f"Team '{team_name}' not found. Options: {list(team_to_cid)}")
        sys.exit(1)
    cid = team_to_cid[match]

    qualifying = collect_sequences(files, cid, zone_lo=0, zone_hi=100)
    relationist, positionist, median_dist = classify_by_style(qualifying)

    categories = [
        {"name": "Positionist", "count": len(positionist), "example": pick_representative(positionist)},
        {"name": "Relationist", "count": len(relationist), "example": pick_representative(relationist)},
    ]

    print(f"Total qualifying sequences: {len(qualifying)}  |  median action distance: {median_dist:.2f}")
    for c in categories:
        print(c["name"], c["count"], "example n_passes=", c["example"]["n_passes"],
              "mean_dist=", round(c["example"]["_mean_dist"], 2), "match=", c["example"]["match"])

    make_plot(match, categories, len(qualifying), median_dist, out)

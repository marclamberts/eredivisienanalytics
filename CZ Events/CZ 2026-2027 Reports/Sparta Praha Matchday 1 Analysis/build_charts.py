"""
AC Sparta Praha - Matchday 1 Analysis
==========================================================
Chance Liga 2026/27, matchday 1 (2026-07-25, Stadion Za Lužánkami, Brno).
Zbrojovka Brno won 3-1; Sparta's goal came from J. Mercado.

Rebuilds the page set of a season-long "SPL" template the user supplied
(yellow/green scheme, full CZ 2025/26 season, all-16-team percentiles) in
Marc Lamberts' Meridian house style (dark mode), scoped down to what this
repo actually has: matchday-1 event feeds for 14 of the CZ 2026-2027
season's 16 teams. Every "league ranking" page here is built from that
14-team, single-match sample and says so in its dek -- never presented as
season-level or full-league data. Two pages that don't survive the
scope-down honestly: the source deck's 5-match formation-frequency panel
and 3-date passing-network sequence become one match's average positions
and one passing network; its cutback-reception-zone page is swapped for a
box-entries map, since Sparta's own matchday-1 fixture has zero cutback
events (rare leaguewide at this sample size -- see match_data.py).

Data: Opta MA3 event feed (CZ Events/CZ 2026-2027), parsed in
match_data.py; shots scored with this repo's own distance+angle xG model
(no provider xG in this feed).

Usage: python3 "build_charts.py"
Outputs PNGs into ./Visuals, then run build_pdf.py to compile the deck.
"""
import math
import os
import sys

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from mplsoccer import Pitch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from housestyle import style, components  # noqa: E402
from housestyle.colors import CATEGORICAL_DARK  # noqa: E402

import match_data as md  # noqa: E402

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Visuals")
os.makedirs(OUT_DIR, exist_ok=True)

FIGSIZE = (13.33, 7.5)
OPP_C = CATEGORICAL_DARK[0]   # ink blue -- Zbrojovka Brno (this match's opponent)
LEAGUE_MUTED = "#5A6672"      # league-context gray, distinct from house axis gray


def save(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=170, facecolor=fig.get_facecolor())
    plt.close(fig)
    print("Saved:", path)


def new_fig():
    palette, cats = style.apply("dark")
    fig = plt.figure(figsize=FIGSIZE)
    return fig, palette


def new_pitch(palette):
    return Pitch(pitch_type="uefa", pitch_color=palette["surface"], line_color=palette["axis"],
                 linewidth=1.0, half=False, line_zorder=2, pad_left=2, pad_right=2)


# ---------------------------------------------------------------------------
# 01. Cover
# ---------------------------------------------------------------------------

def cover(sparta):
    palette, _ = style.apply("dark")
    fig = plt.figure(figsize=FIGSIZE)
    fig.patch.set_facecolor(palette["surface"])

    fig.text(0.5, 0.66, "AC SPARTA PRAHA", fontsize=36, fontweight="bold",
              color=palette["accent"], family="serif", ha="center", va="center")
    fig.text(0.5, 0.57, "MATCHDAY 1 ANALYSIS", fontsize=14, fontweight="bold",
              color=palette["ink_secondary"], family="sans-serif", ha="center", va="center", )

    scores = sparta.match_details["scores"]["ft"]
    h_score, a_score = (scores["home"], scores["away"]) if sparta.is_home else (scores["away"], scores["home"])
    fig.text(0.5, 0.44, f"{md.team_name(md.BRNO_ID)}  {scores['home']} – {scores['away']}  Sparta Praha",
              fontsize=17, color=palette["ink_primary"], family="sans-serif", ha="center", va="center")
    fig.text(0.5, 0.385, f"{md.COMPETITION}  ·  {md.VENUE}  ·  {md.MATCH_DATE}", fontsize=11.5,
              color=palette["ink_secondary"], family="sans-serif", ha="center", va="center")

    fig.text(0.5, 0.24, f"{components.MARK} MATCH ANALYSIS  ·  20 PAGES", fontsize=13, fontweight="bold",
              color=palette["accent"], family="sans-serif", ha="center", va="center")
    fig.text(0.5, 0.195, "Rebuilt in Meridian house style from a season-long template -- scoped here to\n"
                         "matchday 1, the only round with an event feed in this repo (14 of 16 teams)",
              fontsize=9.5, color=palette["ink_muted"], family="sans-serif", ha="center", va="center")

    components.brand_mark(fig, palette=palette, right=0.94, y=0.93)
    components.footer(fig, source=md.SOURCE, palette=palette)
    save(fig, "01_cover.png")


# ---------------------------------------------------------------------------
# 02. Shape: average on-pitch positions
# ---------------------------------------------------------------------------

def shape_average_positions(sparta):
    fig, palette = new_fig()
    pitch = new_pitch(palette)
    ax = fig.add_axes([0.05, 0.10, 0.55, 0.62])
    pitch.draw(ax=ax)

    by_player = {}
    for p in sparta.own(sparta.passes):
        by_player.setdefault(p["player"], []).append((p["x"], p["y"]))
    avg = {pl: (np.mean([v[0] for v in vs]), np.mean([v[1] for v in vs]), len(vs))
           for pl, vs in by_player.items() if len(vs) >= 5}
    max_n = max(v[2] for v in avg.values()) if avg else 1
    for pl, (x, y, n) in avg.items():
        size = 300 + 700 * (n / max_n)
        pitch.scatter(x, y, ax=ax, s=size, color=palette["surface"], edgecolors=palette["accent"],
                      linewidth=2.0, zorder=4)
        pitch.annotate(pl.split(" ")[-1], (x, y), ax=ax, ha="center", va="center", fontsize=8.2,
                       color=palette["ink_primary"], fontweight="bold", zorder=5)

    ax2 = fig.add_axes([0.66, 0.18, 0.30, 0.50])
    ax2.axis("off")
    lines = [
        ("Result", "Lost 1-3 away"),
        ("xG created / conceded", f"{sparta.xg_for:.2f} / {sparta.xg_against:.2f}"),
        ("Touch share", f"{sparta.touch_share():.0%}"),
        ("Field tilt (final 3rd)", f"{sparta.field_tilt():.0%}"),
        ("PPDA", f"{sparta.ppda_for():.1f}"),
        ("Verticality", f"{sparta.verticality():.1f} m/pass"),
    ]
    for i, (label, val) in enumerate(lines):
        y = 1.0 - i * 0.16
        ax2.text(0.0, y, label.upper(), fontsize=9, fontweight="bold", color=palette["accent"], va="top")
        ax2.text(0.0, y - 0.06, val, fontsize=13.5, color=palette["ink_primary"], va="top", fontweight="bold")
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1.05)

    components.header(fig, kicker="Shape",
                       title="Sparta Praha: average on-pitch position, matchday 1",
                       dek="Node = avg. pass location (≥5 passes), size = passes played  ·  shown from real "
                           "positions, not a guessed formation label (no verified code lookup for this feed)",
                       palette=palette)
    components.footer(fig, source=md.SOURCE, palette=palette)
    save(fig, "02_shape_average_positions.png")


# ---------------------------------------------------------------------------
# 03. Passing network
# ---------------------------------------------------------------------------

def _average_positions(team_passes, min_passes=6):
    completed = [p for p in team_passes if p["completed"] and p["end_x"] is not None]
    by_player = {}
    for p in completed:
        by_player.setdefault(p["player"], []).append((p["x"], p["y"]))
    return {pl: (np.mean([v[0] for v in vs]), np.mean([v[1] for v in vs]), len(vs))
            for pl, vs in by_player.items() if len(vs) >= min_passes}


def _combinations(team_passes, avg_pos):
    combos = {}
    ordered = sorted(team_passes, key=lambda p: (p["period"], p["minute"] * 60 + p["second"]))
    for i in range(len(ordered) - 1):
        p, nxt = ordered[i], ordered[i + 1]
        if not p["completed"]:
            continue
        if p["player"] not in avg_pos or nxt["player"] not in avg_pos or p["player"] == nxt["player"]:
            continue
        key = tuple(sorted((p["player"], nxt["player"])))
        combos[key] = combos.get(key, 0) + 1
    return combos


def passing_network(sparta):
    fig, palette = new_fig()
    pitch = new_pitch(palette)
    color = palette["accent"]
    team_passes = sparta.own(sparta.passes)

    ax = fig.add_axes([0.02, 0.10, 0.54, 0.62])
    avg_pos = _average_positions(team_passes)
    combos = _combinations(team_passes, avg_pos)
    pitch.draw(ax=ax)
    max_c = max(combos.values()) if combos else 1
    for (p1, p2), c in combos.items():
        if c < 2:
            continue
        x1, y1, _ = avg_pos[p1]
        x2, y2, _ = avg_pos[p2]
        pitch.lines(x1, y1, x2, y2, ax=ax, color=color, alpha=0.25 + 0.5 * (c / max_c),
                    lw=0.6 + 3.0 * (c / max_c), zorder=2)
    max_n = max(v[2] for v in avg_pos.values()) if avg_pos else 1
    for pl, (x, y, n) in avg_pos.items():
        size = 260 + 900 * (n / max_n)
        pitch.scatter(x, y, ax=ax, s=size, color=palette["surface"], edgecolors=color,
                      linewidth=2.0, zorder=4)
        pitch.annotate(pl.split(" ")[-1], (x, y), ax=ax, ha="center", va="center", fontsize=8.6,
                       color=palette["ink_primary"], fontweight="bold", zorder=5)

    by_player = {}
    for p in team_passes:
        by_player.setdefault(p["player"], {"att": 0, "comp": 0, "prog": 0, "box": 0})
        d = by_player[p["player"]]
        d["att"] += 1
        d["comp"] += int(p["completed"])
        d["prog"] += int(p["progressive"])
        d["box"] += int(p["box_entry"])

    ax2 = fig.add_axes([0.60, 0.14, 0.37, 0.58])
    ax2.axis("off")
    rows = sorted(by_player.items(), key=lambda kv: -kv[1]["att"])[:14]
    headers = ["Player", "Pass", "Acc%", "Prog", "Box"]
    col_x = [0.0, 0.50, 0.64, 0.80, 0.94]
    for x, h in zip(col_x, headers):
        ax2.text(x, 1.0, h, fontsize=9.5, fontweight="bold", color=palette["ink_primary"], va="top",
                 ha="left" if x == 0 else "center")
    ax2.axhline(0.975, color=palette["axis"], linewidth=0.9)
    row_h = 0.94 / max(len(rows), 1)
    for i, (pl, d) in enumerate(rows):
        y = 0.94 - i * row_h
        acc = d["comp"] / d["att"] if d["att"] else 0
        vals = [pl, str(d["att"]), f"{acc:.0%}", str(d["prog"]), str(d["box"])]
        for x, v in zip(col_x, vals):
            ax2.text(x, y, v, fontsize=8.8, color=palette["ink_secondary"], va="top",
                     ha="left" if x == 0 else "center")
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1.03)

    components.header(fig, kicker="Passing Network",
                       title="Sparta Praha: matchday-1 build-up shape",
                       dek="Average completed-pass position (≥6 passes), full match, attacking right",
                       palette=palette)
    components.footer(fig, source=md.SOURCE, palette=palette)
    save(fig, "03_passing_network.png")


# ---------------------------------------------------------------------------
# 04. Ball progression by pitch third
# ---------------------------------------------------------------------------

def ball_progression(sparta):
    fig, palette = new_fig()
    pitch = new_pitch(palette)
    prog = [p for p in sparta.own(sparta.passes) if p["progressive"]]
    thirds = [("Defensive third", 0, 35), ("Middle third", 35, 70), ("Offensive third", 70, 105.1)]
    axes = [fig.add_axes([0.03 + i * 0.325, 0.36, 0.30, 0.44]) for i in range(3)]
    for ax, (label, lo, hi) in zip(axes, thirds):
        pitch.draw(ax=ax)
        pts = [p for p in prog if lo <= p["x"] < hi]
        xs = [p["x"] for p in pts]
        ys = [p["y"] for p in pts]
        if xs:
            stats = pitch.bin_statistic(xs, ys, statistic="count", bins=(6, 4))
            pitch.heatmap(stats, ax=ax, cmap="Oranges", edgecolors=palette["surface"], alpha=0.9, zorder=1)
        ax.set_title(f"{label} ({len(pts)})", color=palette["ink_primary"], fontsize=11, fontweight="bold",
                     family="sans-serif")

    by_player = {}
    for p in prog:
        by_player[p["player"]] = by_player.get(p["player"], 0) + 1
    top = sorted(by_player.items(), key=lambda kv: -kv[1])[:10]
    ax_tab = fig.add_axes([0.10, 0.10, 0.80, 0.20])
    ax_tab.axis("off")
    n = len(top)
    for i, (pl, c) in enumerate(top):
        x = (i % 5) * 0.2
        y = 1.0 - (i // 5) * 0.5
        ax_tab.text(x, y, f"{pl}", fontsize=9.5, color=palette["ink_primary"], va="top", ha="left", fontweight="bold")
        ax_tab.text(x, y - 0.22, f"{c} progressive passes", fontsize=8.5, color=palette["ink_muted"], va="top", ha="left")
    ax_tab.set_xlim(0, 1)
    ax_tab.set_ylim(0, 1.1)

    components.header(fig, kicker="Progression",
                       title=f"Sparta Praha: {len(prog)} progressive passes, where they started",
                       dek="Origin location of completed progressive passes, by pitch third (own-goal-left convention)",
                       palette=palette)
    components.footer(fig, source=md.SOURCE, palette=palette)
    save(fig, "04_ball_progression.png")


# ---------------------------------------------------------------------------
# 05. Possession vs field tilt (league scatter)
# ---------------------------------------------------------------------------

def possession_vs_field_tilt(league):
    fig, palette = new_fig()
    ax = fig.add_axes([0.10, 0.16, 0.82, 0.58])

    for tid, tm in league.teams.items():
        x, y = tm.touch_share() * 100, tm.field_tilt() * 100
        is_sparta = tid == md.SPARTA_ID
        ax.scatter([x], [y], s=170 if is_sparta else 60,
                   color=palette["accent"] if is_sparta else LEAGUE_MUTED,
                   edgecolors=palette["ink_primary"] if is_sparta else "none", linewidth=1.6, zorder=5 if is_sparta else 3)
        ax.annotate(tm.team_name, xy=(x, y), xytext=(7, 5), textcoords="offset points",
                    fontsize=9.5 if is_sparta else 8, color=palette["ink_primary"] if is_sparta else palette["ink_muted"],
                    fontweight="bold" if is_sparta else "normal")

    ax.axhline(50, color=palette["axis"], linewidth=0.8, linestyle=":")
    ax.axvline(50, color=palette["axis"], linewidth=0.8, linestyle=":")
    ax.set_xlabel("Touch share this match (%)")
    ax.set_ylabel("Final-third touch share (%)")

    components.header(fig, kicker="Possession",
                       title="Sparta had the ball and the territory, matchday 1",
                       dek="Touch share vs share of final-third touches, all 14 teams with a matchday-1 feed",
                       palette=palette)
    components.footer(fig, source=md.SOURCE, palette=palette)
    save(fig, "05_possession_vs_field_tilt.png")


# ---------------------------------------------------------------------------
# 06. Progressive pass density (origin vs reception)
# ---------------------------------------------------------------------------

def progressive_pass_density(sparta):
    fig, palette = new_fig()
    pitch = new_pitch(palette)
    ax1 = fig.add_axes([0.03, 0.12, 0.44, 0.62])
    ax2 = fig.add_axes([0.53, 0.12, 0.44, 0.62])
    pitch.draw(ax=ax1)
    pitch.draw(ax=ax2)

    prog = [p for p in sparta.own(sparta.passes) if p["progressive"]]
    ox = [p["x"] for p in prog]; oy = [p["y"] for p in prog]
    rx = [p["end_x"] for p in prog]; ry = [p["end_y"] for p in prog]
    stats1 = pitch.bin_statistic(ox, oy, statistic="count", bins=(7, 5))
    stats2 = pitch.bin_statistic(rx, ry, statistic="count", bins=(7, 5))
    pitch.heatmap(stats1, ax=ax1, cmap="Oranges", edgecolors=palette["surface"], alpha=0.92, zorder=1)
    pitch.heatmap(stats2, ax=ax2, cmap="Oranges", edgecolors=palette["surface"], alpha=0.92, zorder=1)
    ax1.set_title("Origin locations", color=palette["ink_primary"], fontsize=12, fontweight="bold", family="sans-serif")
    ax2.set_title("Reception locations", color=palette["ink_primary"], fontsize=12, fontweight="bold", family="sans-serif")

    components.header(fig, kicker="Progressive Passing",
                       title="Sparta Praha: where progressive passes start and land",
                       dek=f"{len(prog)} progressive passes, matchday 1, attacking right",
                       palette=palette)
    components.footer(fig, source=md.SOURCE, palette=palette)
    save(fig, "06_progressive_pass_density.png")


# ---------------------------------------------------------------------------
# 07. Switches of play (league ranking + spatial map)
# ---------------------------------------------------------------------------

def switches_of_play(sparta, league):
    fig, palette = new_fig()
    pitch = new_pitch(palette)
    ax1 = fig.add_axes([0.05, 0.14, 0.42, 0.58])
    pitch.draw(ax=ax1)
    sw = [p for p in sparta.own(sparta.passes) if p["is_switch"]]
    for p in sw:
        pitch.arrows(p["x"], p["y"], p["end_x"], p["end_y"], ax=ax1, color=palette["accent"],
                    alpha=0.8, width=2.0, headwidth=6, headlength=6, zorder=4)
    ax1.set_title(f"{len(sw)} switches of play (this match)", color=palette["ink_primary"], fontsize=11.5,
                  fontweight="bold", family="sans-serif")

    ax2 = fig.add_axes([0.55, 0.42, 0.40, 0.30])
    ranking = league.ranking(lambda tm: sum(1 for p in tm.own(tm.passes) if p["is_switch"]))
    vals = [v for _, v in ranking]
    ypos_sparta = next(i for i, (tid, _) in enumerate(ranking) if tid == md.SPARTA_ID)
    xs = np.arange(len(ranking))
    colors = [palette["accent"] if tid == md.SPARTA_ID else LEAGUE_MUTED for tid, _ in ranking]
    ax2.bar(xs, vals, color=colors)
    ax2.set_xticks([])
    ax2.set_ylabel("Switches")
    ax2.set_title(f"League rank: #{ypos_sparta + 1} of {len(ranking)} (matchday 1)", color=palette["ink_primary"],
                  fontsize=10.5, fontweight="bold", family="sans-serif")

    by_player = {}
    for p in sw:
        by_player[p["player"]] = by_player.get(p["player"], 0) + 1
    ax3 = fig.add_axes([0.55, 0.14, 0.40, 0.22])
    top = sorted(by_player.items(), key=lambda kv: -kv[1])
    ypos = np.arange(len(top))[::-1]
    ax3.barh(ypos, [v for _, v in top], color=palette["accent"])
    ax3.set_yticks(ypos)
    ax3.set_yticklabels([k for k, _ in top], fontsize=9.5)
    ax3.set_xlabel("Switches completed")

    components.header(fig, kicker="Switches Of Play",
                       title="Sparta Praha's flank-to-flank passing, matchday 1",
                       dek="Completed open-play pass, ≥30m, wide channel to wide channel  ·  ranked among the "
                           "14 teams with a matchday-1 feed, not a season table",
                       palette=palette)
    components.footer(fig, source=md.SOURCE, palette=palette)
    save(fig, "07_switches_of_play.png")


# ---------------------------------------------------------------------------
# 08. Creative zone analysis (halfspace / zone 14)
# ---------------------------------------------------------------------------

def creative_zone_analysis(sparta, league):
    fig, palette = new_fig()
    ax1 = fig.add_axes([0.06, 0.16, 0.40, 0.30])
    ax2 = fig.add_axes([0.56, 0.16, 0.40, 0.30])

    hs_rank = league.ranking(lambda tm: tm.halfspace_zone14_counts()[0])
    z14_rank = league.ranking(lambda tm: tm.halfspace_zone14_counts()[1])
    for ax, rank, label in ((ax1, hs_rank, "Half-space passes"), (ax2, z14_rank, "Zone-14 passes")):
        xs = np.arange(len(rank))
        colors = [palette["accent"] if tid == md.SPARTA_ID else LEAGUE_MUTED for tid, _ in rank]
        ax.bar(xs, [v for _, v in rank], color=colors)
        ax.set_xticks([])
        ax.set_title(label, color=palette["ink_primary"], fontsize=11.5, fontweight="bold", family="sans-serif")
        sparta_rank = next(i for i, (tid, _) in enumerate(rank) if tid == md.SPARTA_ID)
        ax.set_xlabel(f"Sparta: #{sparta_rank + 1} of {len(rank)}")

    def top_players(zone_check):
        counts = {}
        for p in sparta.own(sparta.passes):
            if p["completed"] and p["end_x"] is not None and zone_check(p["end_x"], p["end_y"]):
                counts[p["player"]] = counts.get(p["player"], 0) + 1
        return sorted(counts.items(), key=lambda kv: -kv[1])[:8]

    hs_top = top_players(lambda x, y: any(x0 <= x < x1 and y0 <= y < y1 for x0, x1, y0, y1 in md.HALF_SPACES))
    z14_top = top_players(lambda x, y: md.ZONE14[0] <= x < md.ZONE14[1] and md.ZONE14[2] <= y < md.ZONE14[3])

    ax3 = fig.add_axes([0.14, 0.52, 0.32, 0.24])
    ax4 = fig.add_axes([0.64, 0.52, 0.32, 0.24])
    for ax, top in ((ax3, hs_top), (ax4, z14_top)):
        ypos = np.arange(len(top))[::-1]
        ax.barh(ypos, [v for _, v in top], color=palette["accent"])
        ax.set_yticks(ypos)
        ax.set_yticklabels([k for k, _ in top], fontsize=9)

    components.header(fig, kicker="Creative Zones",
                       title="Sparta Praha's half-space and zone-14 receptions, matchday 1",
                       dek="Completed-pass reception counts  ·  league bars = 14-team matchday-1 sample",
                       palette=palette)
    components.footer(fig, source=md.SOURCE, palette=palette)
    save(fig, "08_creative_zone_analysis.png")


# ---------------------------------------------------------------------------
# 09. Box entries (swapped in for the source deck's cutback page -- Sparta's
# own matchday-1 fixture has zero cutback events; see match_data.py)
# ---------------------------------------------------------------------------

def box_entries_map(sparta):
    fig, palette = new_fig()
    pitch = new_pitch(palette)
    ax = fig.add_axes([0.05, 0.10, 0.90, 0.62])
    pitch.draw(ax=ax)

    entries = [p for p in sparta.own(sparta.passes) if p["box_entry"]]
    for p in entries:
        is_cross = p["is_cross"]
        pitch.arrows(p["x"], p["y"], p["end_x"], p["end_y"], ax=ax,
                    color=palette["accent"] if is_cross else CATEGORICAL_DARK[2],
                    alpha=0.85, width=2.0, headwidth=6, headlength=6, zorder=4)

    legend_elems = [Line2D([0], [0], color=palette["accent"], lw=2.2, label="Cross into box"),
                    Line2D([0], [0], color=CATEGORICAL_DARK[2], lw=2.2, label="Other pass into box")]
    fig.legend(handles=legend_elems, loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 0.05), fontsize=10.5, labelcolor=palette["ink_secondary"])

    components.header(fig, kicker="Box Entries",
                       title=f"Sparta Praha: {len(entries)} completed passes into the box, matchday 1",
                       dek="Swapped in for the source template's cutback map -- Sparta's own matchday-1 fixture "
                           "recorded zero cutbacks (rare leaguewide at this sample size, see match_data.py)",
                       palette=palette)
    components.footer(fig, source=md.SOURCE, palette=palette)
    save(fig, "09_box_entries_map.png")


# ---------------------------------------------------------------------------
# 10. Long ball targets
# ---------------------------------------------------------------------------

def long_ball_targets(sparta):
    fig, palette = new_fig()
    pitch = new_pitch(palette)
    ax = fig.add_axes([0.03, 0.10, 0.55, 0.62])
    pitch.draw(ax=ax)

    lb = [p for p in sparta.own(sparta.passes) if p["is_long_ball"] and p["completed"] and p["end_x"] is not None]
    by_player = {}
    for p in lb:
        by_player.setdefault(p["player"], []).append(p)
    for pl, pts in by_player.items():
        xs = [p["end_x"] for p in pts]; ys = [p["end_y"] for p in pts]
        pitch.scatter(xs, ys, ax=ax, s=90 + 30 * len(pts), color=palette["accent"], alpha=0.7,
                      edgecolors=palette["surface"], linewidth=0.6, zorder=4)

    top = sorted(by_player.items(), key=lambda kv: -len(kv[1]))[:8]
    ax2 = fig.add_axes([0.63, 0.16, 0.33, 0.50])
    ypos = np.arange(len(top))[::-1]
    ax2.barh(ypos, [len(v) for _, v in top], color=palette["accent"])
    ax2.set_yticks(ypos)
    ax2.set_yticklabels([k for k, _ in top], fontsize=10)
    ax2.set_xlabel("Completed long balls received")

    components.header(fig, kicker="Long Balls",
                       title=f"Sparta Praha: {len(lb)} completed long balls, matchday 1",
                       dek="Reception locations, own goal on the left, attacking right",
                       palette=palette)
    components.footer(fig, source=md.SOURCE, palette=palette)
    save(fig, "10_long_ball_targets.png")


# ---------------------------------------------------------------------------
# 11. Goal kick end locations
# ---------------------------------------------------------------------------

def goal_kick_end_locations(sparta):
    fig, palette = new_fig()
    pitch = new_pitch(palette)
    ax = fig.add_axes([0.05, 0.10, 0.90, 0.62])
    pitch.draw(ax=ax)

    gks = [p for p in sparta.own(sparta.passes) if p["is_goal_kick"] and p["end_x"] is not None]
    short = [p for p in gks if math.hypot(p["end_x"] - p["x"], p["end_y"] - p["y"]) < 30]
    long_ = [p for p in gks if math.hypot(p["end_x"] - p["x"], p["end_y"] - p["y"]) >= 30]
    if short:
        pitch.scatter([p["end_x"] for p in short], [p["end_y"] for p in short], ax=ax, s=140,
                      color=CATEGORICAL_DARK[2], edgecolors=palette["surface"], linewidth=0.8,
                      alpha=0.85, zorder=4, label=f"Short (<30m), {len(short)}")
    if long_:
        pitch.scatter([p["end_x"] for p in long_], [p["end_y"] for p in long_], ax=ax, s=140,
                      color=palette["accent"], edgecolors=palette["surface"], linewidth=0.8,
                      alpha=0.85, zorder=4, label=f"Long (≥30m), {len(long_)}")
    fig.legend(loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.05),
               fontsize=10.5, labelcolor=palette["ink_secondary"])

    components.header(fig, kicker="Goal Kicks",
                       title=f"Sparta Praha: {len(gks)} goal kicks, matchday 1",
                       dek="End location of each goal kick, own goal on the left, attacking right",
                       palette=palette)
    components.footer(fig, source=md.SOURCE, palette=palette)
    save(fig, "11_goal_kick_end_locations.png")


# ---------------------------------------------------------------------------
# 12-13. League ranking bars: halfspace receptions (middle third / final third)
# ---------------------------------------------------------------------------

def _league_rank_bar(league, metric_fn, page_num, kicker, title, dek, fname):
    fig, palette = new_fig()
    ax = fig.add_axes([0.22, 0.12, 0.70, 0.62])
    ranking = league.ranking(metric_fn)
    ypos = np.arange(len(ranking))[::-1]
    colors = [palette["accent"] if tid == md.SPARTA_ID else palette["axis"] for tid, _ in ranking]
    ax.barh(ypos, [v for _, v in ranking], color=colors)
    ax.set_yticks(ypos)
    ax.set_yticklabels([f"#{i+1}  {md.team_name(tid)}" for i, (tid, _) in enumerate(ranking)], fontsize=10)
    for y, (tid, v) in zip(ypos, ranking):
        weight = "bold" if tid == md.SPARTA_ID else "normal"
        ax.text(v + max(v2 for _, v2 in ranking) * 0.015, y, f"{v:.1f}" if isinstance(v, float) else str(v),
                va="center", fontsize=9.5, color=palette["ink_primary"], fontweight=weight)
    avg = sum(v for _, v in ranking) / len(ranking)
    ax.axvline(avg, color=palette["ink_muted"], linewidth=1.0, linestyle="--")
    ax.text(avg, len(ranking) - 0.3, " sample avg", fontsize=8, color=palette["ink_muted"], va="bottom")

    components.header(fig, kicker=kicker, title=title, dek=dek, palette=palette)
    components.footer(fig, source=md.SOURCE, palette=palette)
    save(fig, f"{page_num}_{fname}.png")


def middle_third_halfspace_ranking(league):
    def metric(tm):
        passes = [p for p in tm.own(tm.passes) if p["completed"] and p["end_x"] is not None and 35 <= p["end_x"] < 70]
        return sum(1 for p in passes if any(x0 <= p["end_x"] < x1 and y0 <= p["end_y"] < y1 for x0, x1, y0, y1 in md.HALF_SPACES))
    _league_rank_bar(league, metric, "12", "League Ranking",
                      "Middle-third half-space receptions, matchday 1",
                      "Completed-pass receptions in the middle-third half-spaces  ·  14-team matchday-1 sample",
                      "middle_third_halfspace")


def final_third_halfspace_ranking(league):
    def metric(tm):
        passes = [p for p in tm.own(tm.passes) if p["completed"] and p["end_x"] is not None and p["end_x"] >= 70]
        return sum(1 for p in passes if any(x0 <= p["end_x"] < x1 and y0 <= p["end_y"] < y1 for x0, x1, y0, y1 in md.HALF_SPACES))
    _league_rank_bar(league, metric, "13", "League Ranking",
                      "Final-third half-space receptions, matchday 1",
                      "Completed-pass receptions in the final-third half-spaces  ·  14-team matchday-1 sample",
                      "final_third_halfspace")


def verticality_ranking(league):
    _league_rank_bar(league, lambda tm: tm.verticality(), "14", "League Ranking",
                      "Team verticality, matchday 1",
                      "Avg forward distance (m) per completed forward pass  ·  14-team matchday-1 sample",
                      "verticality")


def def_line_height_ranking(league):
    _league_rank_bar(league, lambda tm: tm.def_line_height(), "17", "League Ranking",
                      "Average defensive line height, matchday 1",
                      "Estimated from x-location of defensive/pressing actions  ·  14-team matchday-1 sample",
                      "def_line_height")


# ---------------------------------------------------------------------------
# 15. Attacking sequence involvements
# ---------------------------------------------------------------------------

def attacking_sequence_involvements(sparta):
    fig, palette = new_fig()
    ax = fig.add_axes([0.20, 0.16, 0.72, 0.58])

    shots_by = {}
    for s in sparta.own(sparta.shots):
        shots_by[s["player"]] = shots_by.get(s["player"], 0) + 1
    prog_by = {}
    for p in sparta.own(sparta.passes):
        if p["progressive"] or p["box_entry"]:
            prog_by[p["player"]] = prog_by.get(p["player"], 0) + 1

    players = sorted(set(shots_by) | set(prog_by), key=lambda pl: -(shots_by.get(pl, 0) * 3 + prog_by.get(pl, 0)))[:12]
    ypos = np.arange(len(players))[::-1]
    shot_vals = [shots_by.get(pl, 0) for pl in players]
    prog_vals = [prog_by.get(pl, 0) for pl in players]
    ax.barh(ypos, shot_vals, color=palette["accent"], label="Shots")
    ax.barh(ypos, prog_vals, left=shot_vals, color=CATEGORICAL_DARK[2], label="Progressive pass / box entry")
    ax.set_yticks(ypos)
    ax.set_yticklabels(players, fontsize=10.5)
    ax.set_xlabel("Involvements, matchday 1")
    ax.legend(loc="lower right", frameon=False, fontsize=9.5, labelcolor=palette["ink_secondary"])

    components.header(fig, kicker="Attacking Involvement",
                       title="Sparta Praha: who drove the attack, matchday 1",
                       dek="Shots plus progressive passes / box entries per player, this match only",
                       palette=palette)
    components.footer(fig, source=md.SOURCE, palette=palette)
    save(fig, "15_attacking_sequence_involvements.png")


# ---------------------------------------------------------------------------
# 16. Shot conversion vs xG per shot (league scatter)
# ---------------------------------------------------------------------------

def shot_conversion_vs_xg(league):
    fig, palette = new_fig()
    ax = fig.add_axes([0.10, 0.16, 0.82, 0.58])

    for tid, tm in league.teams.items():
        shots = tm.own(tm.shots)
        if not shots:
            continue
        conv = sum(1 for s in shots if s["is_goal"]) / len(shots) * 100
        xg_shot = sum(s["xg"] for s in shots) / len(shots)
        is_sparta = tid == md.SPARTA_ID
        ax.scatter([xg_shot], [conv], s=170 if is_sparta else 60,
                   color=palette["accent"] if is_sparta else LEAGUE_MUTED,
                   edgecolors=palette["ink_primary"] if is_sparta else "none", linewidth=1.6,
                   zorder=5 if is_sparta else 3)
        ax.annotate(tm.team_name, xy=(xg_shot, conv), xytext=(7, 5), textcoords="offset points",
                    fontsize=9.5 if is_sparta else 8, color=palette["ink_primary"] if is_sparta else palette["ink_muted"],
                    fontweight="bold" if is_sparta else "normal")

    ax.set_xlabel("xG per shot (own model)")
    ax.set_ylabel("Shot conversion (%)")

    components.header(fig, kicker="Finishing",
                       title="Shot conversion vs chance quality, matchday 1",
                       dek="Non-penalty shots, all 14 teams with a matchday-1 feed  ·  own xG model, not a season sample",
                       palette=palette)
    components.footer(fig, source=md.SOURCE, palette=palette)
    save(fig, "16_shot_conversion_vs_xg.png")


# ---------------------------------------------------------------------------
# 18. Shot map
# ---------------------------------------------------------------------------

def shot_map(sparta):
    fig, palette = new_fig()
    pitch = new_pitch(palette)
    ax = fig.add_axes([0.05, 0.10, 0.90, 0.62])
    pitch.draw(ax=ax)

    for s in sparta.own(sparta.shots):
        size = 90 + s["xg"] * 900
        if s["is_goal"]:
            pitch.scatter(s["x"], s["y"], ax=ax, s=size, marker="o", color=palette["accent"],
                          edgecolors=palette["ink_primary"], linewidth=1.4, zorder=5)
        else:
            pitch.scatter(s["x"], s["y"], ax=ax, s=size, marker="o", facecolors="none",
                          edgecolors=palette["accent"], linewidth=1.6, alpha=0.85, zorder=4)
    ax.text(0.02, -0.06, "Hollow = shot   ● Filled = goal   Size = xG", transform=ax.transAxes,
            fontsize=8.5, color=palette["ink_muted"])

    xg = sparta.xg_for
    shots = sparta.own(sparta.shots)
    components.header(fig, kicker="Shot Map",
                       title=f"Sparta Praha: {len(shots)} shots, {xg:.2f} xG, matchday 1",
                       dek="Own xG model: distance + angle to goal, header penalty applied",
                       palette=palette)
    components.footer(fig, source=md.SOURCE, palette=palette)
    save(fig, "18_shot_map.png")


# ---------------------------------------------------------------------------
# 19. Defending: pressing + defensive actions
# ---------------------------------------------------------------------------

def defending_overview(sparta):
    fig, palette = new_fig()
    pitch = new_pitch(palette)
    ax = fig.add_axes([0.03, 0.10, 0.55, 0.62])
    pitch.draw(ax=ax)
    markers = {"Tackle": "o", "Interception": "D", "Clearance": "s"}
    action_colors = {"Tackle": CATEGORICAL_DARK[2], "Interception": CATEGORICAL_DARK[3],
                      "Clearance": palette["ink_muted"]}
    team_defs = sparta.own(sparta.defs)
    for action, marker in markers.items():
        pts = [d for d in team_defs if d["action"] == action]
        if not pts:
            continue
        pitch.scatter([p["x"] for p in pts], [p["y"] for p in pts], ax=ax, s=80, marker=marker,
                      color=action_colors[action], edgecolors=palette["surface"], linewidth=0.6,
                      alpha=0.9, zorder=4)
    legend_elems = [Line2D([0], [0], marker=markers[a], color=palette["surface"], markerfacecolor=action_colors[a],
                            markersize=10, label=a, linewidth=0) for a in markers]
    fig.legend(handles=legend_elems, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.30, 0.05), fontsize=10, labelcolor=palette["ink_secondary"])

    ax2 = fig.add_axes([0.64, 0.16, 0.32, 0.52])
    ax2.axis("off")
    counts = {a: sum(1 for d in team_defs if d["action"] == a) for a in markers}
    lines = [
        ("PPDA", f"{sparta.ppda_for():.1f}"),
        ("Tackles", str(counts["Tackle"])),
        ("Interceptions", str(counts["Interception"])),
        ("Clearances", str(counts["Clearance"])),
        ("Def. line height", f"{sparta.def_line_height():.1f} m"),
        ("xG conceded", f"{sparta.xg_against:.2f}"),
    ]
    for i, (label, val) in enumerate(lines):
        y = 1.0 - i * 0.16
        ax2.text(0.0, y, label.upper(), fontsize=9, fontweight="bold", color=palette["accent"], va="top")
        ax2.text(0.0, y - 0.06, val, fontsize=14, color=palette["ink_primary"], va="top", fontweight="bold")
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1.05)

    components.header(fig, kicker="Defending",
                       title="Sparta Praha: defensive actions, matchday 1",
                       dek="Own goal on the left, attacking right",
                       palette=palette)
    components.footer(fig, source=md.SOURCE, palette=palette)
    save(fig, "19_defending_overview.png")


# ---------------------------------------------------------------------------
# 20. Report card (closing summary)
# ---------------------------------------------------------------------------

def report_card(sparta):
    palette, _ = style.apply("dark")
    fig = plt.figure(figsize=FIGSIZE)
    fig.patch.set_facecolor(palette["surface"])

    scores = sparta.match_details["scores"]["ft"]
    fig.text(0.5, 0.90, f"{md.team_name(md.BRNO_ID)}  {scores['home']}-{scores['away']}  Sparta Praha",
              fontsize=18, fontweight="bold", color=palette["ink_primary"], family="serif",
              ha="center", va="center")
    fig.text(0.5, 0.855, f"{md.COMPETITION}  ·  {md.VENUE}  ·  {md.MATCH_DATE}",
              fontsize=10.5, color=palette["ink_secondary"], ha="center", va="center")

    passes = sparta.own(sparta.passes)
    rows = [
        ("xG created", f"{sparta.xg_for:.2f}"),
        ("xG conceded", f"{sparta.xg_against:.2f}"),
        ("Touch share", f"{sparta.touch_share():.0%}"),
        ("Pass accuracy", f"{sum(1 for p in passes if p['completed']) / len(passes):.0%}"),
        ("PPDA", f"{sparta.ppda_for():.1f}"),
        ("Verticality", f"{sparta.verticality():.1f} m/pass"),
    ]

    ax = fig.add_axes([0.20, 0.20, 0.60, 0.55])
    ax.axis("off")
    n = len(rows)
    for i, (label, val) in enumerate(rows):
        y = 0.85 - i * (0.85 / n)
        ax.text(0.0, y, label, fontsize=12, color=palette["ink_muted"], ha="left", va="top")
        ax.text(1.0, y, val, fontsize=15, fontweight="bold", color=palette["accent"], ha="right", va="top")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    components.brand_mark(fig, palette=palette, right=0.94, y=0.965)
    components.footer(fig, source=md.SOURCE, palette=palette)
    save(fig, "20_report_card.png")


def main():
    sparta = md.TeamMatch(md.SPARTA_ID)
    league = md.LeagueMW1()

    cover(sparta)
    shape_average_positions(sparta)
    passing_network(sparta)
    ball_progression(sparta)
    possession_vs_field_tilt(league)
    progressive_pass_density(sparta)
    switches_of_play(sparta, league)
    creative_zone_analysis(sparta, league)
    box_entries_map(sparta)
    long_ball_targets(sparta)
    goal_kick_end_locations(sparta)
    middle_third_halfspace_ranking(league)
    final_third_halfspace_ranking(league)
    verticality_ranking(league)
    attacking_sequence_involvements(sparta)
    shot_conversion_vs_xg(league)
    def_line_height_ranking(league)
    shot_map(sparta)
    defending_overview(sparta)
    report_card(sparta)
    print("Done.")


if __name__ == "__main__":
    main()

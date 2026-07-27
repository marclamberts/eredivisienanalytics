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

Expanded from the original 20-page scope-down to a full 50-page deck,
folding in the sibling post-match reports' full repertoire (xG flow, shot
log, goal build-ups, pass networks/heatmaps/crossing/zone-14 maps for
both teams, PPDA/field-tilt/possession over time, duels, discipline,
momentum timeline, impact leaderboard, Monte Carlo win probability + xG
scoreline matrix from this match's own shots, an xT flow + leaderboard,
shot assists/key passes, set pieces, a head-to-head radar) plus one page
the source template didn't have at all: pass tempo (m/s) vs pass volume,
across the same 14-team matchday-1 sample.

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
from housestyle.colors import CATEGORICAL_DARK, DARK, STATUS_DARK  # noqa: E402

import match_data as md  # noqa: E402

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Visuals")
os.makedirs(OUT_DIR, exist_ok=True)

FIGSIZE = (13.33, 7.5)
SPARTA_C = DARK["accent"]     # terracotta -- Sparta Praha, matches the dark palette's own accent
OPP_C = CATEGORICAL_DARK[0]   # ink blue -- Zbrojovka Brno (this match's opponent)
GOOD_C = STATUS_DARK["good"]
WARN_C = STATUS_DARK["warning"]
CRIT_C = STATUS_DARK["critical"]
LEAGUE_MUTED = "#5A6672"      # league-context gray, distinct from house axis gray


def team_color(cid):
    return SPARTA_C if cid == md.SPARTA_ID else OPP_C


def team_short(cid):
    return "Sparta" if cid == md.SPARTA_ID else "Zbrojovka"


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

    fig.text(0.5, 0.24, f"{components.MARK} MATCH ANALYSIS  ·  50 PAGES", fontsize=13, fontweight="bold",
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
    save(fig, "12_shape_average_positions.png")


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
    save(fig, "07_pass_network_sparta.png")


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
    save(fig, "13_ball_progression.png")


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
    save(fig, "41_possession_vs_field_tilt.png")


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
    save(fig, "09_progressive_pass_density.png")


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
    save(fig, "42_switches_of_play.png")


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
    save(fig, "43_creative_zone_analysis.png")


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
    save(fig, "37_box_entries_map.png")


# ---------------------------------------------------------------------------
# 10. Long ball targets
# ---------------------------------------------------------------------------

def long_ball_targets(team, page_num, slug, color):
    fig, palette = new_fig()
    pitch = new_pitch(palette)
    ax = fig.add_axes([0.03, 0.10, 0.55, 0.62])
    pitch.draw(ax=ax)

    lb = [p for p in team.own(team.passes) if p["is_long_ball"] and p["completed"] and p["end_x"] is not None]
    by_player = {}
    for p in lb:
        by_player.setdefault(p["player"], []).append(p)
    for pl, pts in by_player.items():
        xs = [p["end_x"] for p in pts]; ys = [p["end_y"] for p in pts]
        pitch.scatter(xs, ys, ax=ax, s=90 + 30 * len(pts), color=color, alpha=0.7,
                      edgecolors=palette["surface"], linewidth=0.6, zorder=4)

    top = sorted(by_player.items(), key=lambda kv: -len(kv[1]))[:8]
    ax2 = fig.add_axes([0.63, 0.16, 0.33, 0.50])
    ypos = np.arange(len(top))[::-1]
    ax2.barh(ypos, [len(v) for _, v in top], color=color)
    ax2.set_yticks(ypos)
    ax2.set_yticklabels([k for k, _ in top], fontsize=10)
    ax2.set_xlabel("Completed long balls received")

    components.header(fig, kicker="Long Balls",
                       title=f"{team.team_name}: {len(lb)} completed long balls, matchday 1",
                       dek="Reception locations, own goal on the left, attacking right",
                       palette=palette)
    components.footer(fig, source=md.SOURCE, palette=palette)
    save(fig, f"{page_num}_long_ball_targets_{slug}.png")


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
    save(fig, "36_goal_kick_end_locations.png")


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
    _league_rank_bar(league, metric, "44", "League Ranking",
                      "Middle-third half-space receptions, matchday 1",
                      "Completed-pass receptions in the middle-third half-spaces  ·  14-team matchday-1 sample",
                      "middle_third_halfspace")


def final_third_halfspace_ranking(league):
    def metric(tm):
        passes = [p for p in tm.own(tm.passes) if p["completed"] and p["end_x"] is not None and p["end_x"] >= 70]
        return sum(1 for p in passes if any(x0 <= p["end_x"] < x1 and y0 <= p["end_y"] < y1 for x0, x1, y0, y1 in md.HALF_SPACES))
    _league_rank_bar(league, metric, "45", "League Ranking",
                      "Final-third half-space receptions, matchday 1",
                      "Completed-pass receptions in the final-third half-spaces  ·  14-team matchday-1 sample",
                      "final_third_halfspace")


def verticality_ranking(league):
    _league_rank_bar(league, lambda tm: tm.verticality(), "46", "League Ranking",
                      "Team verticality, matchday 1",
                      "Avg forward distance (m) per completed forward pass  ·  14-team matchday-1 sample",
                      "verticality")


def def_line_height_ranking(league):
    _league_rank_bar(league, lambda tm: tm.def_line_height(), "49", "League Ranking",
                      "Average defensive line height, matchday 1",
                      "Estimated from x-location of defensive/pressing actions  ·  14-team matchday-1 sample",
                      "def_line_height")


def pace_vs_volume_ranking(league):
    """The user's requested "m/s vs amount of passes" chart -- pass tempo
    (metres/second) vs total pass volume, all 14 teams. A team can be
    high-volume-low-tempo (patient possession) or low-volume-high-tempo
    (direct/counter-attacking); this scatter is where that split shows up."""
    fig, palette = new_fig()
    ax = fig.add_axes([0.10, 0.16, 0.82, 0.58])

    for tid, tm in league.teams.items():
        x, y = tm.pass_volume(), tm.pass_tempo_mps()
        is_sparta = tid == md.SPARTA_ID
        ax.scatter([x], [y], s=170 if is_sparta else 60,
                   color=palette["accent"] if is_sparta else LEAGUE_MUTED,
                   edgecolors=palette["ink_primary"] if is_sparta else "none", linewidth=1.6,
                   zorder=5 if is_sparta else 3)
        ax.annotate(tm.team_name, xy=(x, y), xytext=(7, 5), textcoords="offset points",
                    fontsize=9.5 if is_sparta else 8, color=palette["ink_primary"] if is_sparta else palette["ink_muted"],
                    fontweight="bold" if is_sparta else "normal")

    avg_x = sum(tm.pass_volume() for tm in league.teams.values()) / len(league.teams)
    avg_y = sum(tm.pass_tempo_mps() for tm in league.teams.values()) / len(league.teams)
    ax.axvline(avg_x, color=palette["axis"], linewidth=0.8, linestyle=":")
    ax.axhline(avg_y, color=palette["axis"], linewidth=0.8, linestyle=":")
    ax.set_xlabel("Passes completed, matchday 1")
    ax.set_ylabel("Pass tempo (m/s)")

    components.header(fig, kicker="Tempo",
                       title="Pace of play vs pass volume, matchday 1",
                       dek="Pass tempo = pass distance ÷ time to the next event (gaps >8s excluded)  ·  "
                           "14-team matchday-1 sample, not a season",
                       palette=palette)
    components.footer(fig, source=md.SOURCE, palette=palette)
    save(fig, "40_pace_vs_volume.png")


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
    save(fig, "47_attacking_sequence_involvements.png")


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
    save(fig, "48_shot_conversion_vs_xg.png")


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
    save(fig, "32_shot_map.png")


# ---------------------------------------------------------------------------
# Report card (closing summary)
# ---------------------------------------------------------------------------

def report_card(sparta, brno):
    palette, _ = style.apply("dark")
    fig = plt.figure(figsize=FIGSIZE)
    fig.patch.set_facecolor(palette["surface"])

    scores = sparta.match_details["scores"]["ft"]
    fig.text(0.5, 0.90, f"{md.team_name(md.BRNO_ID)}  {scores['home']}-{scores['away']}  Sparta Praha",
              fontsize=18, fontweight="bold", color=palette["ink_primary"], family="serif",
              ha="center", va="center")
    fig.text(0.5, 0.855, f"{md.COMPETITION}  ·  {md.VENUE}  ·  {md.MATCH_DATE}",
              fontsize=10.5, color=palette["ink_secondary"], ha="center", va="center")

    sp = sparta.own(sparta.passes)
    bp = sparta.against(sparta.passes)
    rows = [
        ("xG", f"{sparta.xg_against:.2f}", f"{sparta.xg_for:.2f}"),
        ("Shots", str(len(sparta.against(sparta.shots))), str(len(sparta.own(sparta.shots)))),
        ("Touch share", f"{1 - sparta.touch_share():.0%}", f"{sparta.touch_share():.0%}"),
        ("Pass accuracy", f"{sum(1 for p in bp if p['completed']) / len(bp):.0%}",
         f"{sum(1 for p in sp if p['completed']) / len(sp):.0%}"),
        ("PPDA", f"{brno.ppda_for():.1f}", f"{sparta.ppda_for():.1f}"),
        ("Verticality (m/pass)", f"{brno.verticality():.1f}", f"{sparta.verticality():.1f}"),
    ]

    ax = fig.add_axes([0.14, 0.20, 0.72, 0.55])
    ax.axis("off")
    ax.text(0.0, 1.0, md.team_name(md.BRNO_ID), fontsize=12.5, fontweight="bold", color=OPP_C, ha="left", va="top")
    ax.text(1.0, 1.0, "Sparta Praha", fontsize=12.5, fontweight="bold", color=SPARTA_C, ha="right", va="top")
    n = len(rows)
    for i, (label, bval, sval) in enumerate(rows):
        y = 0.85 - i * (0.85 / n)
        ax.text(0.0, y, bval, fontsize=13, fontweight="bold", color=palette["ink_primary"], ha="left", va="top")
        ax.text(0.5, y, label, fontsize=10.5, color=palette["ink_muted"], ha="center", va="top")
        ax.text(1.0, y, sval, fontsize=13, fontweight="bold", color=palette["ink_primary"], ha="right", va="top")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    components.brand_mark(fig, palette=palette, right=0.94, y=0.965)
    components.footer(fig, source=md.SOURCE, palette=palette)
    save(fig, "50_report_card.png")


# ---------------------------------------------------------------------------
# 02. Match summary: shot map + KPI bars, both teams
# ---------------------------------------------------------------------------

def match_summary(sparta):
    fig, palette = new_fig()
    pitch = new_pitch(palette)
    ax = fig.add_axes([0.02, 0.10, 0.55, 0.62])
    pitch.draw(ax=ax)

    for s in sparta.shots:
        color = team_color(s["contestantId"])
        x = s["x"] if s["contestantId"] == md.BRNO_ID else 105 - s["x"]
        y = s["y"] if s["contestantId"] == md.BRNO_ID else 68 - s["y"]
        size = 90 + s["xg"] * 900
        if s["is_goal"]:
            pitch.scatter(x, y, ax=ax, s=size, marker="o", color=color,
                          edgecolors=palette["ink_primary"], linewidth=1.4, zorder=5)
        else:
            pitch.scatter(x, y, ax=ax, s=size, marker="o", facecolors="none",
                          edgecolors=color, linewidth=1.6, alpha=0.85, zorder=4)
    ax.text(0.02, -0.06, "Hollow = shot   ● Filled = goal   Size = xG", transform=ax.transAxes,
            fontsize=8.5, color=palette["ink_muted"])

    brno_xg = sparta.against(sparta.shots)
    sparta_shots = sparta.own(sparta.shots)
    brno_pass = sparta.against(sparta.passes)
    sparta_pass = sparta.own(sparta.passes)

    rows = [
        ("Expected goals", f"{sum(s['xg'] for s in brno_xg):.2f}", f"{sparta.xg_for:.2f}",
         sum(s["xg"] for s in brno_xg), sparta.xg_for),
        ("Shots (on target)",
         f"{len(brno_xg)} ({sum(1 for s in brno_xg if s['on_target'])})",
         f"{len(sparta_shots)} ({sum(1 for s in sparta_shots if s['on_target'])})",
         len(brno_xg), len(sparta_shots)),
        ("Touch share", f"{1 - sparta.touch_share():.0%}", f"{sparta.touch_share():.0%}",
         1 - sparta.touch_share(), sparta.touch_share()),
        ("Pass accuracy",
         f"{sum(1 for p in brno_pass if p['completed']) / len(brno_pass):.0%}",
         f"{sum(1 for p in sparta_pass if p['completed']) / len(sparta_pass):.0%}",
         sum(1 for p in brno_pass if p["completed"]) / len(brno_pass),
         sum(1 for p in sparta_pass if p["completed"]) / len(sparta_pass)),
    ]

    ax2 = fig.add_axes([0.60, 0.14, 0.36, 0.56])
    ax2.axis("off")
    n = len(rows)
    for i, (label, bval, sval, bnum, snum) in enumerate(rows):
        y = 1 - (i + 0.5) / n
        ax2.text(0.5, y + 0.075, label, ha="center", va="bottom", fontsize=11.5,
                 fontweight="bold", color=palette["ink_primary"])
        total = bnum + snum if (bnum + snum) > 0 else 1
        frac = bnum / total
        bar_y = y - 0.01
        h = 0.05
        ax2.add_patch(plt.Rectangle((0.0, bar_y), frac, h, color=OPP_C))
        ax2.add_patch(plt.Rectangle((frac, bar_y), 1 - frac, h, color=SPARTA_C))
        ax2.text(0.02, bar_y + h / 2, bval, ha="left", va="center", fontsize=10.5,
                 fontweight="bold", color=palette["surface"])
        ax2.text(0.98, bar_y + h / 2, sval, ha="right", va="center", fontsize=10.5,
                 fontweight="bold", color=palette["surface"])
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)

    legend_elems = [Line2D([0], [0], marker="o", color=palette["surface"], markerfacecolor=OPP_C,
                            markersize=10, label=md.team_name(md.BRNO_ID), linewidth=0),
                    Line2D([0], [0], marker="o", color=palette["surface"], markerfacecolor=SPARTA_C,
                            markersize=10, label="Sparta Praha", linewidth=0)]
    fig.legend(handles=legend_elems, loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 0.015), fontsize=10.5, labelcolor=palette["ink_secondary"])

    scores = sparta.match_details["scores"]["ft"]
    components.header(fig, kicker="Match Summary",
                       title=f"{md.team_name(md.BRNO_ID)} {scores['home']}-{scores['away']} Sparta Praha",
                       dek="Shot map and headline numbers, both ends attacking right",
                       palette=palette)
    components.footer(fig, source=md.SOURCE, palette=palette)
    save(fig, "02_match_summary.png")


# ---------------------------------------------------------------------------
# 03. xG flow
# ---------------------------------------------------------------------------

def xg_flow(sparta):
    fig, palette = new_fig()
    ax = fig.add_axes([0.08, 0.16, 0.78, 0.60])

    def series(cid):
        team_shots = sorted([s for s in sparta.shots if s["contestantId"] == cid], key=lambda s: s["minute"])
        mins, cum, total = [0.0], [0.0], 0.0
        for s in team_shots:
            mins.append(s["minute"]); cum.append(total)
            total += s["xg"]
            mins.append(s["minute"]); cum.append(total)
        mins.append(96); cum.append(total)
        return mins, cum

    for cid, color, name in ((md.BRNO_ID, OPP_C, "Zbrojovka Brno"), (md.SPARTA_ID, SPARTA_C, "Sparta")):
        mins, cum = series(cid)
        ax.plot(mins, cum, color=color, linewidth=2.4, zorder=4)
        ax.fill_between(mins, cum, step=None, color=color, alpha=0.10, zorder=1)
        ax.annotate(f"{name}\n{cum[-1]:.2f} xG", xy=(1, cum[-1]), xycoords=("axes fraction", "data"),
                    xytext=(10, 0), textcoords="offset points", color=color, fontsize=10,
                    fontweight="bold", va="center", ha="left", annotation_clip=False)

    for cid, color in ((md.BRNO_ID, OPP_C), (md.SPARTA_ID, SPARTA_C)):
        team_shots = sorted([s for s in sparta.shots if s["contestantId"] == cid], key=lambda s: s["minute"])
        running = 0.0
        for s in team_shots:
            if s["is_goal"]:
                ax.scatter([s["minute"]], [running], marker="*", s=220, color=palette["ink_primary"],
                           edgecolors=color, linewidth=1.6, zorder=6)
                ax.annotate(f"{s['player']} {s['minute']}'", xy=(s["minute"], running),
                            xytext=(0, 14), textcoords="offset points", ha="center",
                            fontsize=8.5, color=palette["ink_secondary"])
            running += s["xg"]

    ax.axvline(45, color=palette["axis"], linewidth=0.8, linestyle=":")
    ax.set_xlim(0, 100)
    ax.set_xlabel("Minute")
    ax.set_ylabel("Cumulative xG")

    brno_xg = sum(s["xg"] for s in sparta.against(sparta.shots))
    components.header(fig, kicker="xG Flow",
                       title=f"Zbrojovka Brno out-created Sparta {brno_xg:.2f} to {sparta.xg_for:.2f} xG",
                       dek="Zbrojovka Brno 3-1 Sparta Praha  ·  cumulative expected goals by minute",
                       palette=palette)
    components.footer(fig, source=md.SOURCE, palette=palette)
    save(fig, "03_xg_flow.png")


# ---------------------------------------------------------------------------
# 04. Shot quality table
# ---------------------------------------------------------------------------

def shot_quality_table(sparta):
    fig, palette = new_fig()
    ax = fig.add_axes([0.05, 0.12, 0.90, 0.62])
    ax.axis("off")

    ordered = sorted(sparta.shots, key=lambda s: s["minute"])
    cols = ["Min", "Team", "Player", "Situation", "Body", "Outcome", "xG"]
    widths = [0.06, 0.20, 0.24, 0.16, 0.12, 0.12, 0.10]
    x0 = [sum(widths[:i]) for i in range(len(widths))]

    header_y = 1.0
    for x, w, label in zip(x0, widths, cols):
        ax.text(x, header_y, label, fontsize=10.5, fontweight="bold", color=palette["ink_primary"],
                va="top", ha="left")
    ax.axhline(header_y - 0.025, xmin=0, xmax=1, color=palette["axis"], linewidth=1.0)

    row_h = 0.95 / max(len(ordered), 1)
    for i, s in enumerate(ordered):
        y = header_y - 0.05 - i * row_h
        color = team_color(s["contestantId"])
        weight = "bold" if s["is_goal"] else "normal"
        vals = [f"{s['minute']}'", team_short(s["contestantId"]), s["player"], s["situation"],
                "Head" if s["is_header"] else "Foot", s["outcome"], f"{s['xg']:.2f}"]
        for x, w, v in zip(x0, widths, vals):
            ax.text(x, y, v, fontsize=9.5, color=color if x == x0[1] else palette["ink_primary"],
                    fontweight=weight, va="top", ha="left")
        if s["is_goal"]:
            ax.text(0.985, y, "★", fontsize=11, color=GOOD_C, va="top", ha="right")
    ax.set_xlim(0, 1)
    ax.set_ylim(header_y - 0.05 - len(ordered) * row_h, 1.03)

    components.header(fig, kicker="Shot Log",
                       title=f"All {len(ordered)} shots of the match, ranked by kickoff time",
                       dek="Own xG model: distance + angle to goal, header penalty applied",
                       palette=palette)
    components.footer(fig, source=md.SOURCE, palette=palette)
    save(fig, "04_shot_quality_table.png")


# ---------------------------------------------------------------------------
# 05. Goal build-ups
# ---------------------------------------------------------------------------

def goal_buildups(sparta):
    fig, palette = new_fig()
    pitch = new_pitch(palette)

    goals = sorted([s for s in sparta.shots if s["is_goal"]], key=lambda s: s["minute"])
    n = len(goals)
    axes = [fig.add_axes([0.02 + i * (0.96 / n), 0.10, 0.96 / n - 0.02, 0.62]) for i in range(n)]

    for ax, g in zip(axes, goals):
        pitch.draw(ax=ax)
        color = team_color(g["contestantId"])
        team_events = [e for e in sparta.events if e["contestantId"] == g["contestantId"]
                       and e.get("x") is not None and e["typeId"] in (1, 3, 61)
                       and md.event_time(e) <= g["minute"] * 60 + 59]
        team_events.sort(key=lambda e: (e["periodId"], md.event_time(e), e["eventId"]))
        chain = team_events[-4:]
        pts = []
        for e in chain:
            x, y = md.norm_xy(e, sparta.directions)
            xm, ym = md.to_m(x, y)
            pts.append((xm, ym))
        pts.append((g["x"], g["y"]))

        for j in range(len(pts) - 1):
            x1, y1 = pts[j]
            x2, y2 = pts[j + 1]
            alpha = 0.45 + 0.55 * (j / (len(pts) - 1))
            pitch.arrows(x1, y1, x2, y2, ax=ax, color=color, alpha=alpha, width=2.2,
                        headwidth=6, headlength=6, zorder=3)
        pitch.scatter(g["x"], g["y"], ax=ax, s=260, marker="*", color=palette["ink_primary"],
                      edgecolors=color, linewidth=1.6, zorder=6)
        ax.set_title(f"{g['minute']}'  {g['player']}\n{md.team_name(g['contestantId'])}", color=color,
                     fontsize=11, fontweight="bold", family="sans-serif")

    fig.text(0.5, 0.085, "Last 4 touches before each goal (passes, take-ons and ball touches by the scoring team)",
              ha="center", fontsize=9, color=palette["ink_muted"])

    components.header(fig, kicker="Goal Build-Ups",
                       title=f"How all {n} goals were made",
                       dek="Zbrojovka Brno 3-1 Sparta Praha  ·  possession chain leading to each goal",
                       palette=palette)
    components.footer(fig, source=md.SOURCE, palette=palette)
    save(fig, "05_goal_buildups.png")


# ---------------------------------------------------------------------------
# 06. Pass network (combined overview)
# ---------------------------------------------------------------------------

def _draw_pass_network(ax, team_passes, color, palette, pitch, min_passes=8, node_scale=1.0, label_size=7.6):
    avg_pos = _average_positions(team_passes, min_passes)
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
        size = (260 + 900 * (n / max_n)) * node_scale
        pitch.scatter(x, y, ax=ax, s=size, color=palette["surface"], edgecolors=color,
                      linewidth=2.0, zorder=4)
        last = pl.split(" ")[-1]
        pitch.annotate(last, (x, y), ax=ax, ha="center", va="center", fontsize=label_size,
                       color=palette["ink_primary"], fontweight="bold", zorder=5)
    return avg_pos, combos


def pass_network_combined(sparta):
    fig, palette = new_fig()
    pitch = new_pitch(palette)
    ax1 = fig.add_axes([0.02, 0.10, 0.47, 0.62])
    ax2 = fig.add_axes([0.51, 0.10, 0.47, 0.62])

    brno_passes = sparta.against(sparta.passes)
    sparta_passes = sparta.own(sparta.passes)
    _draw_pass_network(ax1, brno_passes, OPP_C, palette, pitch)
    _draw_pass_network(ax2, sparta_passes, SPARTA_C, palette, pitch)
    ax1.set_title(md.team_name(md.BRNO_ID), color=OPP_C, fontsize=13, fontweight="bold", family="sans-serif")
    ax2.set_title("Sparta Praha", color=SPARTA_C, fontsize=13, fontweight="bold", family="sans-serif")

    fig.text(0.5, 0.09, "Node position = average completed-pass location (≥ 8 passes)  ·  "
                         "Node size = passes played  ·  Line width = pass combinations (≥ 2)",
              ha="center", fontsize=9, color=palette["ink_muted"])

    components.header(fig, kicker="Pass Network",
                       title="How each side built play, matchday 1",
                       dek="Average completed-pass position, full match, both attacking right",
                       palette=palette)
    components.footer(fig, source=md.SOURCE, palette=palette)
    save(fig, "06_pass_network_combined.png")


def pass_network_team(team, page_num, slug, color):
    fig, palette = new_fig()
    pitch = new_pitch(palette)
    name = team.team_name
    team_passes = team.own(team.passes)

    ax = fig.add_axes([0.02, 0.10, 0.54, 0.62])
    _draw_pass_network(ax, team_passes, color, palette, pitch, min_passes=6, node_scale=1.15, label_size=8.6)

    by_player = {}
    for p in team_passes:
        by_player.setdefault(p["player"], {"att": 0, "comp": 0, "prog": 0, "box": 0, "cross": 0})
        d = by_player[p["player"]]
        d["att"] += 1
        d["comp"] += int(p["completed"])
        d["prog"] += int(p["progressive"])
        d["box"] += int(p["box_entry"])
        d["cross"] += int(p["is_cross"] and p["completed"])

    ax2 = fig.add_axes([0.60, 0.14, 0.37, 0.58])
    ax2.axis("off")
    rows = sorted(by_player.items(), key=lambda kv: -kv[1]["att"])[:14]
    headers = ["Player", "Pass", "Acc%", "Prog", "Box", "Cross"]
    col_x = [0.0, 0.46, 0.58, 0.72, 0.84, 0.94]
    for x, h in zip(col_x, headers):
        ax2.text(x, 1.0, h, fontsize=9.5, fontweight="bold", color=palette["ink_primary"], va="top",
                 ha="left" if x == 0 else "center")
    ax2.axhline(0.975, color=palette["axis"], linewidth=0.9)
    row_h = 0.94 / max(len(rows), 1)
    for i, (pl, d) in enumerate(rows):
        y = 0.94 - i * row_h
        acc = d["comp"] / d["att"] if d["att"] else 0
        vals = [pl, str(d["att"]), f"{acc:.0%}", str(d["prog"]), str(d["box"]), str(d["cross"])]
        for x, v in zip(col_x, vals):
            ax2.text(x, y, v, fontsize=8.8, color=palette["ink_secondary"], va="top",
                     ha="left" if x == 0 else "center")
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1.03)

    components.header(fig, kicker="Pass Network",
                       title=f"{name}: passing volume and progression, player by player",
                       dek="Average completed-pass position (≥ 6 passes) and full pass log, attacking right",
                       palette=palette)
    components.footer(fig, source=md.SOURCE, palette=palette)
    save(fig, f"{page_num}_pass_network_{slug}.png")


# ---------------------------------------------------------------------------
# 10-11. Touch heatmap
# ---------------------------------------------------------------------------

def touch_heatmap(team, page_num, slug, color):
    fig, palette = new_fig()
    pitch = new_pitch(palette)
    name = team.team_name
    ax = fig.add_axes([0.06, 0.10, 0.88, 0.62])
    pitch.draw(ax=ax)

    own = team.own(team.touches)
    xs = [t["x"] for t in own]
    ys = [t["y"] for t in own]
    cmap = "Oranges" if color == SPARTA_C else "Blues"
    stats = pitch.bin_statistic(xs, ys, statistic="count", bins=(9, 6))
    pitch.heatmap(stats, ax=ax, cmap=cmap, edgecolors=palette["surface"], alpha=0.92, zorder=1)

    components.header(fig, kicker="Territory",
                       title=f"{name}: where the team spent its {len(xs)} touches",
                       dek="Touch density by pitch zone, matchday 1, attacking right",
                       palette=palette)
    components.footer(fig, source=md.SOURCE, palette=palette)
    save(fig, f"{page_num}_touch_heatmap_{slug}.png")


# ---------------------------------------------------------------------------
# 14. Field tilt over time
# ---------------------------------------------------------------------------

def field_tilt_over_time(sparta):
    fig, palette = new_fig()
    ax = fig.add_axes([0.16, 0.16, 0.78, 0.58])

    bucket = 5
    max_min = 95
    edges = list(range(0, max_min + bucket, bucket))
    tilt, centers = [], []
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        b = sum(1 for t in sparta.touches if lo <= t["minute"] < hi and t["contestantId"] == md.BRNO_ID and t["x"] >= 70)
        s = sum(1 for t in sparta.touches if lo <= t["minute"] < hi and t["contestantId"] == md.SPARTA_ID and t["x"] >= 70)
        total = b + s
        tilt.append((b / total - 0.5) * 100 if total else 0.0)
        centers.append((lo + hi) / 2)

    tilt = np.array(tilt)
    centers = np.array(centers)
    ax.fill_between(centers, tilt, 0, where=(tilt >= 0), color=OPP_C, alpha=0.75, step="mid")
    ax.fill_between(centers, tilt, 0, where=(tilt < 0), color=SPARTA_C, alpha=0.75, step="mid")
    ax.axhline(0, color=palette["axis"], linewidth=1.0)
    ax.axvline(45, color=palette["axis"], linewidth=0.8, linestyle=":")

    for s in sparta.shots:
        if s["is_goal"]:
            y = 46 if s["contestantId"] == md.BRNO_ID else -46
            ax.scatter([s["minute"]], [y], marker="*", s=200, color=palette["ink_primary"],
                       edgecolors=team_color(s["contestantId"]), linewidth=1.4, zorder=6)

    ax.set_ylim(-55, 55)
    ax.set_xlim(0, max_min)
    ax.set_xlabel("Minute")
    ax.set_ylabel("Field tilt (final-third touch share)")
    ax.set_yticks([-50, -25, 0, 25, 50])
    ax.set_yticklabels(["Sparta 100%", "75%", "Even", "75%", "Zbrojovka 100%"], fontsize=9.5)

    overall = 1 - sparta.field_tilt()
    components.header(fig, kicker="Field Tilt",
                       title=f"Zbrojovka Brno controlled the final third, {overall:.0%} of touches to {1 - overall:.0%}",
                       dek="Share of final-third touches, 5-minute buckets  ·  ★ marks a goal",
                       palette=palette)
    components.footer(fig, source=md.SOURCE, palette=palette)
    save(fig, "14_field_tilt_over_time.png")


# ---------------------------------------------------------------------------
# 15. Possession by thirds
# ---------------------------------------------------------------------------

def possession_thirds(sparta):
    fig, palette = new_fig()
    ax = fig.add_axes([0.16, 0.24, 0.68, 0.40])

    zone_colors = [CATEGORICAL_DARK[0], CATEGORICAL_DARK[3], CATEGORICAL_DARK[1]]
    zone_labels = ["Defensive", "Middle", "Attacking"]

    def thirds(cid):
        t = [x for x in sparta.touches if x["contestantId"] == cid]
        d = sum(1 for x in t if x["x"] < 35)
        m = sum(1 for x in t if 35 <= x["x"] < 70)
        a = sum(1 for x in t if x["x"] >= 70)
        total = d + m + a
        return [d / total, m / total, a / total], total

    for i, (cid, name, color) in enumerate(((md.BRNO_ID, md.team_name(md.BRNO_ID), OPP_C), (md.SPARTA_ID, "Sparta Praha", SPARTA_C))):
        fracs, total = thirds(cid)
        y = 1 - i
        left = 0
        for frac, zc, zl in zip(fracs, zone_colors, zone_labels):
            ax.barh(y, frac, left=left, height=0.6, color=zc)
            if frac > 0.06:
                ax.text(left + frac / 2, y, f"{frac:.0%}", ha="center", va="center",
                        fontsize=10.5, fontweight="bold", color=palette["surface"])
            left += frac
        ax.text(-0.02, y, f"{name}\n({total} touches)", ha="right", va="center", fontsize=10.5,
                fontweight="bold", color=color)

    ax.set_xlim(0, 1)
    ax.set_ylim(-0.7, 1.7)
    ax.set_yticks([])
    ax.set_xlabel("Share of touches")
    ax.set_axisbelow(True)

    legend_elems = [Line2D([0], [0], marker="s", color=palette["surface"], markerfacecolor=zc,
                            markersize=12, label=zl, linewidth=0) for zc, zl in zip(zone_colors, zone_labels)]
    fig.legend(handles=legend_elems, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 0.06), fontsize=10.5, labelcolor=palette["ink_secondary"])

    components.header(fig, kicker="Possession",
                       title="Sparta had the ball, but spent far less time in the final third",
                       dek="Distribution of touches across pitch thirds, both teams' own attacking direction",
                       palette=palette)
    components.footer(fig, source=md.SOURCE, palette=palette)
    save(fig, "15_possession_thirds.png")


# ---------------------------------------------------------------------------
# 16. Progression comparison bars
# ---------------------------------------------------------------------------

def progression_bars(sparta):
    fig, palette = new_fig()
    ax = fig.add_axes([0.24, 0.16, 0.66, 0.56])

    brno_passes = sparta.against(sparta.passes)
    sparta_passes = sparta.own(sparta.passes)

    metrics = [
        ("Progressive passes", sum(1 for p in brno_passes if p["progressive"]),
         sum(1 for p in sparta_passes if p["progressive"])),
        ("Final-third entries", sum(1 for p in brno_passes if p["final_third_entry"]),
         sum(1 for p in sparta_passes if p["final_third_entry"])),
        ("Passes into the box", sum(1 for p in brno_passes if p["box_entry"]),
         sum(1 for p in sparta_passes if p["box_entry"])),
        ("Completed crosses", sum(1 for p in brno_passes if p["is_cross"] and p["completed"]),
         sum(1 for p in sparta_passes if p["is_cross"] and p["completed"])),
    ]

    n = len(metrics)
    ypos = np.arange(n)[::-1]
    maxval = max(max(b, s) for _, b, s in metrics) * 1.15
    for y, (label, b, s) in zip(ypos, metrics):
        ax.barh(y + 0.18, b, height=0.32, color=OPP_C)
        ax.barh(y - 0.18, s, height=0.32, color=SPARTA_C)
        ax.text(b + maxval * 0.015, y + 0.18, str(b), va="center", fontsize=10, color=palette["ink_primary"])
        ax.text(s + maxval * 0.015, y - 0.18, str(s), va="center", fontsize=10, color=palette["ink_primary"])
    ax.set_yticks(ypos)
    ax.set_yticklabels([m[0] for m in metrics], fontsize=11.5, color=palette["ink_primary"])
    ax.set_xlim(0, maxval)
    ax.set_xlabel("Count")
    ax.grid(axis="x")
    ax.set_axisbelow(True)

    legend_elems = [Line2D([0], [0], marker="s", color=palette["surface"], markerfacecolor=OPP_C,
                            markersize=12, label=md.team_name(md.BRNO_ID), linewidth=0),
                    Line2D([0], [0], marker="s", color=palette["surface"], markerfacecolor=SPARTA_C,
                            markersize=12, label="Sparta Praha", linewidth=0)]
    fig.legend(handles=legend_elems, loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 0.02), fontsize=10.5, labelcolor=palette["ink_secondary"])

    components.header(fig, kicker="Progression",
                       title="How each side moved the ball forward, matchday 1",
                       dek="Progressive pass = completed pass cutting ≥25% off the distance to goal",
                       palette=palette)
    components.footer(fig, source=md.SOURCE, palette=palette)
    save(fig, "16_progression_bars.png")


# ---------------------------------------------------------------------------
# 17. PPDA by time window
# ---------------------------------------------------------------------------

def ppda_by_window(sparta):
    fig, palette = new_fig()
    ax1 = fig.add_axes([0.08, 0.16, 0.40, 0.58])
    ax2 = fig.add_axes([0.56, 0.16, 0.40, 0.58])

    buckets = [(0, 15), (15, 30), (30, 45), (45, 60), (60, 75), (75, 96)]
    labels = ["0-15", "15-30", "30-45", "45-60", "60-75", "75-90+"]

    def bucketed(cid, opp_id):
        return [md.compute_ppda(sparta.passes, sparta.pressing, cid, opp_id, lo, hi) for lo, hi in buckets]

    brno_vals = bucketed(md.BRNO_ID, md.SPARTA_ID)
    sparta_vals = bucketed(md.SPARTA_ID, md.BRNO_ID)
    finite = [v for v in brno_vals + sparta_vals if not math.isnan(v)]
    shared_max = max(finite) * 1.15 if finite else 1.0

    for ax, vals, color, name, cid, opp in ((ax1, brno_vals, OPP_C, md.team_name(md.BRNO_ID), md.BRNO_ID, md.SPARTA_ID),
                                             (ax2, sparta_vals, SPARTA_C, "Sparta Praha", md.SPARTA_ID, md.BRNO_ID)):
        xs = np.arange(len(labels))
        clean = [v if not math.isnan(v) else 0 for v in vals]
        ax.bar(xs, clean, color=color)
        for x, v in zip(xs, vals):
            if not math.isnan(v):
                ax.text(x, v + shared_max * 0.02, f"{v:.1f}", ha="center", fontsize=9.5,
                        color=palette["ink_primary"], fontweight="bold")
        overall = md.compute_ppda(sparta.passes, sparta.pressing, cid, opp)
        ax.axhline(overall, color=palette["ink_muted"], linestyle="--", linewidth=1.0)
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, fontsize=8.5)
        ax.set_ylim(0, shared_max)
        ax.set_title(f"{name}\nOverall PPDA: {overall:.1f}", color=color, fontsize=11.5,
                     fontweight="bold", family="sans-serif")
        ax.set_ylabel("PPDA")

    components.header(fig, kicker="Pressing",
                       title="Pressing intensity across the match, by 15-minute window",
                       dek="Passes per defensive action in the opponent's own 60%  ·  lower = more intense press",
                       palette=palette)
    components.footer(fig, source=md.SOURCE, palette=palette)
    save(fig, "17_ppda_by_window.png")


# ---------------------------------------------------------------------------
# 18. Defensive actions (both teams)
# ---------------------------------------------------------------------------

def defensive_actions(sparta):
    fig, palette = new_fig()
    pitch = new_pitch(palette)
    ax1 = fig.add_axes([0.02, 0.10, 0.47, 0.62])
    ax2 = fig.add_axes([0.51, 0.10, 0.47, 0.62])
    pitch.draw(ax=ax1)
    pitch.draw(ax=ax2)

    markers = {"Tackle": "o", "Interception": "D", "Clearance": "s"}
    action_colors = {"Tackle": CATEGORICAL_DARK[2], "Interception": CATEGORICAL_DARK[3],
                      "Clearance": palette["ink_muted"]}

    for ax, cid in ((ax1, md.BRNO_ID), (ax2, md.SPARTA_ID)):
        team_defs = [d for d in sparta.defs if d["contestantId"] == cid]
        for action, marker in markers.items():
            pts = [d for d in team_defs if d["action"] == action]
            if not pts:
                continue
            pitch.scatter([p["x"] for p in pts], [p["y"] for p in pts], ax=ax, s=80, marker=marker,
                          color=action_colors[action], edgecolors=palette["surface"], linewidth=0.6,
                          alpha=0.9, zorder=4)
        counts = {a: sum(1 for d in team_defs if d["action"] == a) for a in markers}
        title = f"{team_short(cid)}\nTkl {counts['Tackle']}  ·  Int {counts['Interception']}  ·  Clr {counts['Clearance']}"
        ax.set_title(title, color=team_color(cid), fontsize=12, fontweight="bold", family="sans-serif")

    legend_elems = [Line2D([0], [0], marker=markers[a], color=palette["surface"], markerfacecolor=action_colors[a],
                            markersize=10, label=a, linewidth=0) for a in markers]
    fig.legend(handles=legend_elems, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 0.02), fontsize=10.5, labelcolor=palette["ink_secondary"])

    components.header(fig, kicker="Defending",
                       title="Where each side won the ball back, matchday 1",
                       dek="Tackles, interceptions and clearances, own goal on the left, attacking right",
                       palette=palette)
    components.footer(fig, source=md.SOURCE, palette=palette)
    save(fig, "18_defensive_actions.png")


# ---------------------------------------------------------------------------
# 19. Duels summary
# ---------------------------------------------------------------------------

def duels_summary(sparta):
    fig, palette = new_fig()
    ax = fig.add_axes([0.24, 0.16, 0.66, 0.56])

    kinds = ["Tackle", "Aerial", "Challenge"]
    n = len(kinds)
    ypos = np.arange(n)[::-1]
    maxval = 0
    rows = []
    for kind in kinds:
        b = [d for d in sparta.duels if d["contestantId"] == md.BRNO_ID and d["action"] == kind]
        s = [d for d in sparta.duels if d["contestantId"] == md.SPARTA_ID and d["action"] == kind]
        b_won = sum(1 for d in b if d["success"])
        s_won = sum(1 for d in s if d["success"])
        rows.append((kind, len(b), b_won, len(s), s_won))
        maxval = max(maxval, len(b), len(s))
    maxval *= 1.2

    for y, (kind, b_n, b_won, s_n, s_won) in zip(ypos, rows):
        b_rate = b_won / b_n if b_n else 0
        s_rate = s_won / s_n if s_n else 0
        ax.barh(y + 0.18, b_n, height=0.32, color=palette["axis"])
        ax.barh(y + 0.18, b_won, height=0.32, color=OPP_C)
        ax.barh(y - 0.18, s_n, height=0.32, color=palette["axis"])
        ax.barh(y - 0.18, s_won, height=0.32, color=SPARTA_C)
        ax.text(b_n + maxval * 0.015, y + 0.18, f"{b_won}/{b_n} ({b_rate:.0%})", va="center",
                fontsize=9.5, color=palette["ink_primary"])
        ax.text(s_n + maxval * 0.015, y - 0.18, f"{s_won}/{s_n} ({s_rate:.0%})", va="center",
                fontsize=9.5, color=palette["ink_primary"])
    ax.set_yticks(ypos)
    ax.set_yticklabels([f"{k} duels" for k in kinds], fontsize=11.5, color=palette["ink_primary"])
    ax.set_xlim(0, maxval)
    ax.set_xlabel("Contested (solid = won)")
    ax.grid(axis="x")
    ax.set_axisbelow(True)

    legend_elems = [Line2D([0], [0], marker="s", color=palette["surface"], markerfacecolor=OPP_C,
                            markersize=12, label=f"{md.team_name(md.BRNO_ID)} won", linewidth=0),
                    Line2D([0], [0], marker="s", color=palette["surface"], markerfacecolor=SPARTA_C,
                            markersize=12, label="Sparta Praha won", linewidth=0)]
    fig.legend(handles=legend_elems, loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 0.02), fontsize=10.5, labelcolor=palette["ink_secondary"])

    components.header(fig, kicker="Duels",
                       title="Tackle, aerial and loose-ball duel win rates",
                       dek="Bar length = duels contested, solid fill = duels won",
                       palette=palette)
    components.footer(fig, source=md.SOURCE, palette=palette)
    save(fig, "19_duels_summary.png")


# ---------------------------------------------------------------------------
# 20. Discipline
# ---------------------------------------------------------------------------

def discipline(sparta):
    fig, palette = new_fig()
    ax = fig.add_axes([0.24, 0.36, 0.66, 0.36])

    brno_fouls = sum(1 for d in sparta.pressing if d["contestantId"] == md.BRNO_ID and d["action"] == "Foul")
    sparta_fouls = sum(1 for d in sparta.pressing if d["contestantId"] == md.SPARTA_ID and d["action"] == "Foul")
    brno_yellow = sum(1 for c in sparta.cards if c["contestantId"] == md.BRNO_ID and c["kind"] == "Yellow")
    sparta_yellow = sum(1 for c in sparta.cards if c["contestantId"] == md.SPARTA_ID and c["kind"] == "Yellow")
    brno_red = sum(1 for c in sparta.cards if c["contestantId"] == md.BRNO_ID and c["kind"] in ("Red", "2nd Yellow"))
    sparta_red = sum(1 for c in sparta.cards if c["contestantId"] == md.SPARTA_ID and c["kind"] in ("Red", "2nd Yellow"))

    metrics = [("Fouls committed", brno_fouls, sparta_fouls), ("Yellow cards", brno_yellow, sparta_yellow),
               ("Red cards", brno_red, sparta_red)]
    n = len(metrics)
    ypos = np.arange(n)[::-1]
    maxval = max(max(b, s) for _, b, s in metrics) * 1.3 or 1
    for y, (label, b, s) in zip(ypos, metrics):
        ax.barh(y + 0.18, b, height=0.32, color=OPP_C)
        ax.barh(y - 0.18, s, height=0.32, color=SPARTA_C)
        ax.text(b + maxval * 0.02, y + 0.18, str(b), va="center", fontsize=10, color=palette["ink_primary"])
        ax.text(s + maxval * 0.02, y - 0.18, str(s), va="center", fontsize=10, color=palette["ink_primary"])
    ax.set_yticks(ypos)
    ax.set_yticklabels([m[0] for m in metrics], fontsize=11.5, color=palette["ink_primary"])
    ax.set_xlim(0, maxval)
    ax.grid(axis="x")
    ax.set_axisbelow(True)

    legend_elems = [Line2D([0], [0], marker="s", color=palette["surface"], markerfacecolor=OPP_C,
                            markersize=12, label=md.team_name(md.BRNO_ID), linewidth=0),
                    Line2D([0], [0], marker="s", color=palette["surface"], markerfacecolor=SPARTA_C,
                            markersize=12, label="Sparta Praha", linewidth=0)]
    fig.legend(handles=legend_elems, loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 0.20), fontsize=10.5, labelcolor=palette["ink_secondary"])

    sorted_cards = sorted(sparta.cards, key=lambda c: c["minute"])
    rows_per_col = max(1, math.ceil(len(sorted_cards) / 2)) if sorted_cards else 1
    for i, c in enumerate(sorted_cards):
        col, row = divmod(i, rows_per_col)
        x = 0.30 + col * 0.42
        y = 0.13 - row * 0.032
        fig.text(x, y, f"{c['minute']}' {c['player']} ({c['kind']})", ha="left", va="top",
                  fontsize=8.8, color=team_color(c["contestantId"]))

    components.header(fig, kicker="Discipline",
                       title="Fouls and cards, matchday 1",
                       dek="Foul committed = this team's own Foul record with outcome 0 (see match_data.py)",
                       palette=palette)
    components.footer(fig, source=md.SOURCE, palette=palette)
    save(fig, "20_discipline.png")


# ---------------------------------------------------------------------------
# 21. Momentum timeline
# ---------------------------------------------------------------------------

def momentum_timeline(sparta):
    fig, palette = new_fig()
    ax_rect = [0.16, 0.30, 0.78, 0.32]
    ax = fig.add_axes(ax_rect)

    ax.axhline(1, color=OPP_C, linewidth=6, alpha=0.25, solid_capstyle="round")
    ax.axhline(0, color=SPARTA_C, linewidth=6, alpha=0.25, solid_capstyle="round")
    ylim = (-0.6, 1.6)
    for data_y, name, color in ((1, "Zbrojovka", OPP_C), (0, "Sparta", SPARTA_C)):
        fig_y = ax_rect[1] + ax_rect[3] * (data_y - ylim[0]) / (ylim[1] - ylim[0])
        fig.text(ax_rect[0] - 0.01, fig_y, name, ha="right", va="center", fontsize=11,
                  fontweight="bold", color=color)

    def row(cid):
        return 1 if cid == md.BRNO_ID else 0

    for s in sparta.shots:
        if s["is_goal"]:
            ax.scatter([s["minute"]], [row(s["contestantId"])], marker="*", s=320,
                       color=palette["ink_primary"], edgecolors=team_color(s["contestantId"]),
                       linewidth=1.8, zorder=5)
            ax.annotate(f"{s['player']} {s['minute']}'", xy=(s["minute"], row(s["contestantId"])),
                        xytext=(0, 16 if s["contestantId"] == md.BRNO_ID else -20),
                        textcoords="offset points", ha="center", fontsize=8.5, color=palette["ink_secondary"])

    for c in sparta.cards:
        marker_color = CRIT_C if c["kind"] != "Yellow" else WARN_C
        ax.scatter([c["minute"]], [row(c["contestantId"])], marker="s", s=110, color=marker_color,
                   edgecolors=palette["surface"], linewidth=1.0, zorder=4)

    for s in sparta.subs:
        ax.scatter([s["minute"]], [row(s["contestantId"])], marker="^", s=70, color=palette["ink_muted"],
                   alpha=0.85, zorder=3)

    ax.set_xlim(0, 96)
    ax.set_ylim(-0.6, 1.6)
    ax.set_yticks([])
    ax.set_xlabel("Minute")
    ax.axvline(45, color=palette["axis"], linewidth=0.8, linestyle=":")

    legend_elems = [Line2D([0], [0], marker="*", color=palette["surface"], markerfacecolor=palette["ink_primary"],
                            markeredgecolor=palette["ink_primary"], markersize=14, label="Goal", linewidth=0),
                    Line2D([0], [0], marker="s", color=palette["surface"], markerfacecolor=WARN_C,
                            markersize=10, label="Yellow card", linewidth=0),
                    Line2D([0], [0], marker="^", color=palette["surface"], markerfacecolor=palette["ink_muted"],
                            markersize=10, label="Substitution", linewidth=0)]
    fig.legend(handles=legend_elems, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 0.14), fontsize=10.5, labelcolor=palette["ink_secondary"])

    scores = sparta.match_details["scores"]["ft"]
    components.header(fig, kicker="Match Narrative",
                       title="Goals, cards and changes across the 90 minutes",
                       dek=f"Zbrojovka Brno {scores['home']}-{scores['away']} Sparta Praha  ·  key moments timeline",
                       palette=palette)
    components.footer(fig, source=md.SOURCE, palette=palette)
    save(fig, "21_momentum_timeline.png")


# ---------------------------------------------------------------------------
# 22. Impact leaderboard
# ---------------------------------------------------------------------------

def impact_leaderboard(sparta):
    fig, palette = new_fig()
    ax1 = fig.add_axes([0.08, 0.20, 0.40, 0.54])
    ax2 = fig.add_axes([0.56, 0.20, 0.40, 0.54])

    def score_players(cid):
        scores = {}
        for s in sparta.shots:
            if s["contestantId"] != cid:
                continue
            scores[s["player"]] = scores.get(s["player"], 0) + s["xg"] + (3.0 if s["is_goal"] else 0)
        for p in sparta.passes:
            if p["contestantId"] != cid:
                continue
            scores[p["player"]] = scores.get(p["player"], 0) + 0.15 * p["progressive"] + 0.35 * p["box_entry"]
        for d in sparta.defs:
            if d["contestantId"] != cid:
                continue
            scores[d["player"]] = scores.get(d["player"], 0) + 0.3
        return sorted(scores.items(), key=lambda kv: -kv[1])[:6]

    for ax, cid, color, name in ((ax1, md.BRNO_ID, OPP_C, md.team_name(md.BRNO_ID)), (ax2, md.SPARTA_ID, SPARTA_C, "Sparta Praha")):
        top = score_players(cid)[::-1]
        ypos = np.arange(len(top))
        vals = [v for _, v in top]
        ax.barh(ypos, vals, color=color)
        ax.set_yticks(ypos)
        ax.set_yticklabels([p for p, _ in top], fontsize=10, color=palette["ink_primary"])
        for y, v in zip(ypos, vals):
            ax.text(v + max(vals, default=1) * 0.02, y, f"{v:.1f}", va="center", fontsize=9,
                    color=palette["ink_secondary"])
        ax.set_title(name, color=color, fontsize=12.5, fontweight="bold", family="sans-serif")
        ax.set_xlabel("Impact score")

    fig.text(0.5, 0.09, "Simple composite: xG + 3×goals + 0.15×progressive pass + 0.35×box entry + "
                         "0.3×defensive action  ·  not an official rating",
              ha="center", fontsize=8.8, color=palette["ink_muted"])

    components.header(fig, kicker="Impact",
                       title="Who did the most, by a simple composite score",
                       dek="Shooting, progression and defending combined into one rough index",
                       palette=palette)
    components.footer(fig, source=md.SOURCE, palette=palette)
    save(fig, "22_impact_leaderboard.png")


# ---------------------------------------------------------------------------
# 23. Win probability & xG by situation
# ---------------------------------------------------------------------------

def _donut(ax, frac, color, palette, label, sublabel):
    ax.pie([frac, 1 - frac], radius=1.0, startangle=90, counterclock=False,
           colors=[color, palette["axis"]], wedgeprops=dict(width=0.32, edgecolor=palette["surface"], linewidth=1.5))
    ax.text(0, 0.12, f"{frac:.0%}", ha="center", va="center", fontsize=20, fontweight="bold", color=palette["ink_primary"])
    ax.text(0, -0.12, sublabel, ha="center", va="center", fontsize=8.5, color=palette["ink_muted"])
    ax.set_title(label, color=color, fontsize=11.5, fontweight="bold", family="sans-serif", pad=2)


def win_probability(sparta, sim):
    fig, palette = new_fig()
    ax1 = fig.add_axes([0.06, 0.34, 0.19, 0.32])
    ax2 = fig.add_axes([0.28, 0.34, 0.19, 0.32])
    _donut(ax1, sim["home_win"], OPP_C, palette, "Zbrojovka", "WIN PROBABILITY")
    _donut(ax2, sim["away_win"], SPARTA_C, palette, "Sparta", "WIN PROBABILITY")
    fig.text(0.275, 0.30, f"Draw: {sim['draw']:.0%}", ha="center", fontsize=10.5,
              color=palette["ink_secondary"], fontweight="bold")

    ax3 = fig.add_axes([0.56, 0.20, 0.38, 0.52])
    situations = ["Open play", "Fast break", "Set piece", "Corner"]
    brno_vals = [sum(s["xg"] for s in sparta.shots if s["contestantId"] == md.BRNO_ID and s["situation"] == sit)
                 for sit in situations]
    sparta_vals = [sum(s["xg"] for s in sparta.shots if s["contestantId"] == md.SPARTA_ID and s["situation"] == sit)
                 for sit in situations]
    n = len(situations)
    ypos = np.arange(n)[::-1]
    maxval = max(brno_vals + sparta_vals) * 1.25 or 1
    for y, b, s in zip(ypos, brno_vals, sparta_vals):
        ax3.barh(y + 0.18, b, height=0.32, color=OPP_C)
        ax3.barh(y - 0.18, s, height=0.32, color=SPARTA_C)
        ax3.text(b + maxval * 0.02, y + 0.18, f"{b:.2f}", va="center", fontsize=9.5, color=palette["ink_primary"])
        ax3.text(s + maxval * 0.02, y - 0.18, f"{s:.2f}", va="center", fontsize=9.5, color=palette["ink_primary"])
    ax3.set_yticks(ypos)
    ax3.set_yticklabels(situations, fontsize=10.5, color=palette["ink_primary"])
    ax3.set_xlim(0, maxval)
    ax3.set_xlabel("xG")
    ax3.grid(axis="x")
    ax3.set_axisbelow(True)
    ax3.set_title("xG by situation", color=palette["ink_primary"], fontsize=11.5, fontweight="bold", family="sans-serif")

    legend_elems = [Line2D([0], [0], marker="s", color=palette["surface"], markerfacecolor=OPP_C,
                            markersize=12, label=md.team_name(md.BRNO_ID), linewidth=0),
                    Line2D([0], [0], marker="s", color=palette["surface"], markerfacecolor=SPARTA_C,
                            markersize=12, label="Sparta Praha", linewidth=0)]
    fig.legend(handles=legend_elems, loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 0.10), fontsize=10.5, labelcolor=palette["ink_secondary"])

    scores = sparta.match_details["scores"]["ft"]
    components.header(fig, kicker="Match Odds",
                       title=f"Zbrojovka Brno were the heavy favourite on chances created ({sim['home_win']:.0%} win probability)",
                       dek=f"{sim['n']:,}-simulation Monte Carlo from this match's own shot xG  ·  "
                           f"actual result {scores['home']}-{scores['away']}",
                       palette=palette)
    components.footer(fig, source=md.SOURCE, palette=palette)
    save(fig, "23_win_probability.png")


# ---------------------------------------------------------------------------
# 24. xG scoreline matrix
# ---------------------------------------------------------------------------

def xg_scoreline_matrix(sparta, sim):
    fig, palette = new_fig()
    ax = fig.add_axes([0.14, 0.14, 0.66, 0.58])

    cap = sim["cap"]
    n = sim["n"]
    grid = np.zeros((cap + 1, cap + 1))
    for (h, a), c in sim["score_counts"].items():
        grid[h, a] = c / n

    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("wa_seq", [palette["surface"], palette["accent"]])
    ax.imshow(grid, cmap=cmap, origin="lower", vmin=0, aspect="equal")
    for h in range(cap + 1):
        for a in range(cap + 1):
            v = grid[h, a]
            if v < 0.001:
                continue
            txt_color = palette["surface"] if v > grid.max() * 0.5 else palette["ink_primary"]
            ax.text(a, h, f"{v:.1%}", ha="center", va="center", fontsize=8.2, color=txt_color)

    labels = [str(i) for i in range(cap)] + [f"{cap}+"]
    ax.set_xticks(range(cap + 1))
    ax.set_xticklabels(labels)
    ax.set_yticks(range(cap + 1))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Sparta goals")
    ax.set_ylabel("Zbrojovka Brno goals")
    ax.grid(False)

    scores = sparta.match_details["scores"]["ft"]
    fth, fta = scores["home"], scores["away"]
    rect = plt.Rectangle((min(fta, cap) - 0.5, min(fth, cap) - 0.5), 1, 1, fill=False,
                          edgecolor=palette["ink_primary"], linewidth=2.6, zorder=5)
    ax.add_patch(rect)
    ax.annotate("Actual result", xy=(min(fta, cap), min(fth, cap)), xytext=(cap * 0.55, cap * 0.15),
                fontsize=9.5, color=palette["ink_primary"], fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=palette["ink_primary"]))

    components.header(fig, kicker="Scoreline Probability",
                       title=(f"{fth}-{fta} was the single most likely scoreline from these chances"
                              if grid[min(fth, cap), min(fta, cap)] == grid.max()
                              else f"{fth}-{fta} was a plausible outcome from these chances, but not the likeliest"),
                       dek=f"Simulated scoreline probabilities from this match's own shot xG, {n:,} runs  ·  own xG model",
                       palette=palette)
    components.footer(fig, source=md.SOURCE, palette=palette)
    save(fig, "24_xg_scoreline_matrix.png")


# ---------------------------------------------------------------------------
# 25. xT flow  (expected threat, requested)
# ---------------------------------------------------------------------------

def xt_flow(sparta):
    fig, palette = new_fig()
    ax = fig.add_axes([0.08, 0.16, 0.78, 0.60])

    def series(cid):
        team_passes = sorted([p for p in sparta.passes if p["contestantId"] == cid and p["completed"]],
                              key=lambda p: p["minute"])
        mins, cum, total = [0.0], [0.0], 0.0
        for p in team_passes:
            mins.append(p["minute"]); cum.append(total)
            total += max(0.0, p["xt_added"])
            mins.append(p["minute"]); cum.append(total)
        mins.append(96); cum.append(total)
        return mins, cum

    for cid, color, name in ((md.BRNO_ID, OPP_C, "Zbrojovka Brno"), (md.SPARTA_ID, SPARTA_C, "Sparta")):
        mins, cum = series(cid)
        ax.plot(mins, cum, color=color, linewidth=2.4, zorder=4)
        ax.fill_between(mins, cum, step=None, color=color, alpha=0.10, zorder=1)
        ax.annotate(f"{name}\n{cum[-1]:.2f} xT", xy=(1, cum[-1]), xycoords=("axes fraction", "data"),
                    xytext=(10, 0), textcoords="offset points", color=color, fontsize=10,
                    fontweight="bold", va="center", ha="left", annotation_clip=False)

    for s in sparta.shots:
        if s["is_goal"]:
            ax.axvline(s["minute"], color=palette["axis"], linewidth=0.7, linestyle=":", zorder=1)

    ax.axvline(45, color=palette["axis"], linewidth=0.8, linestyle=":")
    ax.set_xlim(0, 100)
    ax.set_xlabel("Minute")
    ax.set_ylabel("Cumulative xT added (completed passes)")

    brno_xt = sum(max(0.0, p["xt_added"]) for p in sparta.against(sparta.passes) if p["completed"])
    sparta_xt = sum(max(0.0, p["xt_added"]) for p in sparta.own(sparta.passes) if p["completed"])
    leader = "Zbrojovka Brno" if brno_xt > sparta_xt else "Sparta"
    components.header(fig, kicker="xT Flow",
                       title=f"{leader} generated more threat through their passing, matchday 1",
                       dek="Cumulative expected threat (xT) added by completed passes  ·  own xT proxy model "
                           "(distance+angle geometry, not a possession-value model -- see match_data.py)",
                       palette=palette)
    components.footer(fig, source=md.SOURCE, palette=palette)
    save(fig, "25_xt_flow.png")


# ---------------------------------------------------------------------------
# 26. xT leaderboard
# ---------------------------------------------------------------------------

def xt_leaderboard(sparta):
    fig, palette = new_fig()
    ax1 = fig.add_axes([0.08, 0.20, 0.40, 0.54])
    ax2 = fig.add_axes([0.56, 0.20, 0.40, 0.54])

    def top_players(cid, color, ax, name):
        by_player = {}
        for p in sparta.passes:
            if p["contestantId"] != cid or not p["completed"]:
                continue
            by_player[p["player"]] = by_player.get(p["player"], 0) + max(0.0, p["xt_added"])
        top = sorted(by_player.items(), key=lambda kv: -kv[1])[:6][::-1]
        ypos = np.arange(len(top))
        vals = [v for _, v in top]
        ax.barh(ypos, vals, color=color)
        ax.set_yticks(ypos)
        ax.set_yticklabels([p for p, _ in top], fontsize=10, color=palette["ink_primary"])
        for y, v in zip(ypos, vals):
            ax.text(v + max(vals, default=1) * 0.02, y, f"{v:.2f}", va="center", fontsize=9, color=palette["ink_secondary"])
        ax.set_title(name, color=color, fontsize=12.5, fontweight="bold", family="sans-serif")
        ax.set_xlabel("xT added")

    top_players(md.BRNO_ID, OPP_C, ax1, md.team_name(md.BRNO_ID))
    top_players(md.SPARTA_ID, SPARTA_C, ax2, "Sparta Praha")

    components.header(fig, kicker="Threat Creation",
                       title="Who generated the most expected threat through passing",
                       dek="Sum of positive xT added by completed passes, own xT proxy model, matchday 1",
                       palette=palette)
    components.footer(fig, source=md.SOURCE, palette=palette)
    save(fig, "26_xt_leaderboard.png")


# ---------------------------------------------------------------------------
# 27. Shot assists map
# ---------------------------------------------------------------------------

def shot_assists_map(sparta):
    fig, palette = new_fig()
    pitch = new_pitch(palette)
    ax1 = fig.add_axes([0.02, 0.10, 0.47, 0.62])
    ax2 = fig.add_axes([0.51, 0.10, 0.47, 0.62])
    pitch.draw(ax=ax1)
    pitch.draw(ax=ax2)

    for ax, cid, color, name in ((ax1, md.BRNO_ID, OPP_C, md.team_name(md.BRNO_ID)), (ax2, md.SPARTA_ID, SPARTA_C, "Sparta Praha")):
        assists = [a for a in sparta.assists if a["contestantId"] == cid]
        for a in assists:
            marker_c = GOOD_C if a["is_goal"] else color
            pitch.arrows(a["x"], a["y"], a["end_x"], a["end_y"], ax=ax, color=marker_c,
                        alpha=0.85, width=2.0, headwidth=6, headlength=6, zorder=4)
            pitch.scatter(a["shot_x"], a["shot_y"], ax=ax, s=60 + a["shot_xg"] * 500,
                         color=marker_c, edgecolors=palette["surface"], linewidth=0.8, zorder=5)
        ax.set_title(f"{name} ({len(assists)} shot assists)", color=color, fontsize=12, fontweight="bold",
                     family="sans-serif")

    legend_elems = [Line2D([0], [0], color=GOOD_C, lw=2.2, label="Assist -> goal"),
                    Line2D([0], [0], color=palette["ink_muted"], lw=2.2, label="Assist -> other shot")]
    fig.legend(handles=legend_elems, loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 0.03), fontsize=10.5, labelcolor=palette["ink_secondary"])

    components.header(fig, kicker="Chance Creation",
                       title="Shot assists: the last completed pass before each shot",
                       dek="Own goal on the left, attacking right  ·  a shot with no intervening teammate pass "
                           "gets no assist credited",
                       palette=palette)
    components.footer(fig, source=md.SOURCE, palette=palette)
    save(fig, "27_shot_assists_map.png")


# ---------------------------------------------------------------------------
# 28. Key passes leaderboard
# ---------------------------------------------------------------------------

def key_passes_leaderboard(sparta):
    fig, palette = new_fig()
    ax = fig.add_axes([0.20, 0.16, 0.72, 0.58])

    by_player = {}
    for a in sparta.assists:
        by_player.setdefault(a["assister"], {"n": 0, "xg": 0.0, "cid": a["contestantId"]})
        by_player[a["assister"]]["n"] += 1
        by_player[a["assister"]]["xg"] += a["shot_xg"]

    top = sorted(by_player.items(), key=lambda kv: -kv[1]["xg"])[:12][::-1]
    ypos = np.arange(len(top))
    vals = [d["xg"] for _, d in top]
    colors = [team_color(d["cid"]) for _, d in top]
    ax.barh(ypos, vals, color=colors)
    ax.set_yticks(ypos)
    ax.set_yticklabels([f"{pl} ({d['n']})" for pl, d in top], fontsize=10.5)
    for y, (pl, d) in zip(ypos, top):
        ax.text(d["xg"] + max(vals, default=1) * 0.015, y, f"{d['xg']:.2f} xG", va="center", fontsize=9,
                color=palette["ink_secondary"])
    ax.set_xlabel("xG of shots assisted")

    components.header(fig, kicker="Chance Creation",
                       title="Key passes: who created the most dangerous chances",
                       dek="Sum of xG on shots each player assisted  ·  count in brackets = shot assists",
                       palette=palette)
    components.footer(fig, source=md.SOURCE, palette=palette)
    save(fig, "28_key_passes_leaderboard.png")


# ---------------------------------------------------------------------------
# 29. Crossing map
# ---------------------------------------------------------------------------

def crossing_map(team, page_num, slug, color):
    fig, palette = new_fig()
    pitch = new_pitch(palette)
    name = team.team_name
    ax = fig.add_axes([0.05, 0.10, 0.90, 0.62])
    pitch.draw(ax=ax)

    crosses = [p for p in team.own(team.passes) if p["is_cross"] and p["end_x"] is not None]
    completed = [p for p in crosses if p["completed"]]
    incomplete = [p for p in crosses if not p["completed"]]
    for p in incomplete:
        pitch.arrows(p["x"], p["y"], p["end_x"], p["end_y"], ax=ax, color=palette["ink_muted"],
                    alpha=0.35, width=1.4, headwidth=5, headlength=5, zorder=2)
    for p in completed:
        pitch.arrows(p["x"], p["y"], p["end_x"], p["end_y"], ax=ax, color=color,
                    alpha=0.85, width=2.0, headwidth=6, headlength=6, zorder=4)

    legend_elems = [Line2D([0], [0], color=color, lw=2.2, label=f"Completed ({len(completed)})"),
                    Line2D([0], [0], color=palette["ink_muted"], lw=1.8, alpha=0.6, label=f"Incomplete ({len(incomplete)})")]
    fig.legend(handles=legend_elems, loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 0.05), fontsize=10.5, labelcolor=palette["ink_secondary"])

    components.header(fig, kicker="Crossing",
                       title=f"{name}: {len(crosses)} crosses, matchday 1",
                       dek="Own goal on the left, attacking right",
                       palette=palette)
    components.footer(fig, source=md.SOURCE, palette=palette)
    save(fig, f"{page_num}_crossing_map_{slug}.png")


# ---------------------------------------------------------------------------
# 30. Final third entries
# ---------------------------------------------------------------------------

def final_third_entries(sparta):
    fig, palette = new_fig()
    pitch = new_pitch(palette)
    ax1 = fig.add_axes([0.02, 0.10, 0.47, 0.62])
    ax2 = fig.add_axes([0.51, 0.10, 0.47, 0.62])
    pitch.draw(ax=ax1)
    pitch.draw(ax=ax2)

    for ax, cid, color, name in ((ax1, md.BRNO_ID, OPP_C, md.team_name(md.BRNO_ID)), (ax2, md.SPARTA_ID, SPARTA_C, "Sparta Praha")):
        entries = [p for p in sparta.passes if p["contestantId"] == cid and p["final_third_entry"]
                   and p["end_x"] is not None]
        for p in entries:
            pitch.arrows(p["x"], p["y"], p["end_x"], p["end_y"], ax=ax, color=color,
                        alpha=0.6, width=1.6, headwidth=5, headlength=5, zorder=3)
        ax.set_title(f"{name} ({len(entries)})", color=color, fontsize=12, fontweight="bold", family="sans-serif")

    components.header(fig, kicker="Progression",
                       title="Completed passes into the final third, matchday 1",
                       dek="Own goal on the left, attacking right",
                       palette=palette)
    components.footer(fig, source=md.SOURCE, palette=palette)
    save(fig, "30_final_third_entries.png")


# ---------------------------------------------------------------------------
# 31. Zone 14 & half-space map
# ---------------------------------------------------------------------------

def zone14_halfspace_map(team, page_num, slug, color):
    fig, palette = new_fig()
    pitch = new_pitch(palette)
    name = team.team_name
    ax = fig.add_axes([0.05, 0.10, 0.90, 0.62])
    pitch.draw(ax=ax)

    z0, z1, zy0, zy1 = md.ZONE14
    ax.add_patch(plt.Rectangle((z0, zy0), z1 - z0, zy1 - zy0, facecolor=CATEGORICAL_DARK[3],
                                alpha=0.18, edgecolor=CATEGORICAL_DARK[3], linewidth=1.0, zorder=1))
    for x0, x1, y0, y1 in md.HALF_SPACES:
        ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor=color, alpha=0.10,
                                    edgecolor=color, linewidth=0.8, zorder=1))

    passes = [p for p in team.own(team.passes) if p["completed"] and p["end_x"] is not None]
    z14 = [p for p in passes if z0 <= p["end_x"] < z1 and zy0 <= p["end_y"] < zy1]
    hs = [p for p in passes if any(x0 <= p["end_x"] < x1 and y0 <= p["end_y"] < y1 for x0, x1, y0, y1 in md.HALF_SPACES)]
    for p in hs:
        pitch.scatter(p["end_x"], p["end_y"], ax=ax, s=50, color=color, alpha=0.7, zorder=3)
    for p in z14:
        pitch.scatter(p["end_x"], p["end_y"], ax=ax, s=70, color=CATEGORICAL_DARK[3], alpha=0.8, zorder=4)

    legend_elems = [Line2D([0], [0], marker="o", color=palette["surface"], markerfacecolor=color,
                            markersize=9, label=f"Half-space reception ({len(hs)})", linewidth=0),
                    Line2D([0], [0], marker="o", color=palette["surface"], markerfacecolor=CATEGORICAL_DARK[3],
                            markersize=9, label=f"Zone-14 reception ({len(z14)})", linewidth=0)]
    fig.legend(handles=legend_elems, loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 0.05), fontsize=10.5, labelcolor=palette["ink_secondary"])

    components.header(fig, kicker="Creative Zones",
                       title=f"{name}: half-space and zone-14 receptions, matchday 1",
                       dek="Completed-pass receptions in the two most dangerous central-lane zones, attacking right",
                       palette=palette)
    components.footer(fig, source=md.SOURCE, palette=palette)
    save(fig, f"{page_num}_zone14_halfspace_{slug}.png")


# ---------------------------------------------------------------------------
# 33. Set-piece analysis
# ---------------------------------------------------------------------------

def set_piece_analysis(sparta):
    fig, palette = new_fig()
    ax = fig.add_axes([0.24, 0.16, 0.66, 0.56])

    situations = ["Corner", "Set piece"]
    n = len(situations)
    ypos = np.arange(n)[::-1]
    rows = []
    maxval = 0
    for sit in situations:
        b = [s for s in sparta.shots if s["contestantId"] == md.BRNO_ID and s["situation"] == sit]
        s = [s for s in sparta.shots if s["contestantId"] == md.SPARTA_ID and s["situation"] == sit]
        rows.append((sit, len(b), sum(x["xg"] for x in b), len(s), sum(x["xg"] for x in s)))
        maxval = max(maxval, len(b), len(s))
    maxval = maxval * 1.3 or 1

    for y, (label, bn, bxg, sn, sxg) in zip(ypos, rows):
        ax.barh(y + 0.18, bn, height=0.32, color=OPP_C)
        ax.barh(y - 0.18, sn, height=0.32, color=SPARTA_C)
        ax.text(bn + maxval * 0.02, y + 0.18, f"{bn} shots, {bxg:.2f} xG", va="center", fontsize=9.5,
                color=palette["ink_primary"])
        ax.text(sn + maxval * 0.02, y - 0.18, f"{sn} shots, {sxg:.2f} xG", va="center", fontsize=9.5,
                color=palette["ink_primary"])
    ax.set_yticks(ypos)
    ax.set_yticklabels([f"From {s.lower()}" for s in situations], fontsize=11.5, color=palette["ink_primary"])
    ax.set_xlim(0, maxval)
    ax.grid(axis="x")
    ax.set_axisbelow(True)

    legend_elems = [Line2D([0], [0], marker="s", color=palette["surface"], markerfacecolor=OPP_C,
                            markersize=12, label=md.team_name(md.BRNO_ID), linewidth=0),
                    Line2D([0], [0], marker="s", color=palette["surface"], markerfacecolor=SPARTA_C,
                            markersize=12, label="Sparta Praha", linewidth=0)]
    fig.legend(handles=legend_elems, loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 0.20), fontsize=10.5, labelcolor=palette["ink_secondary"])

    components.header(fig, kicker="Set Pieces",
                       title="Shots and chance quality from dead-ball situations",
                       dek="'Set piece' = direct free kick shot; 'Corner' includes second-phase shots tagged from a corner",
                       palette=palette)
    components.footer(fig, source=md.SOURCE, palette=palette)
    save(fig, "33_set_piece_analysis.png")


# ---------------------------------------------------------------------------
# 38. Team radar
# ---------------------------------------------------------------------------

def team_radar(sparta):
    fig, palette = new_fig()
    ax = fig.add_axes([0.26, 0.18, 0.48, 0.58], polar=True)

    brno_pass = sparta.against(sparta.passes)
    sparta_pass = sparta.own(sparta.passes)
    brno_xg = sum(s["xg"] for s in sparta.against(sparta.shots))
    b_touch = sum(1 for t in sparta.touches if t["contestantId"] == md.BRNO_ID and t["x"] >= 70)
    s_touch = sum(1 for t in sparta.touches if t["contestantId"] == md.SPARTA_ID and t["x"] >= 70)

    metrics = [
        ("xG", brno_xg, sparta.xg_for),
        ("Shots", len(sparta.against(sparta.shots)), len(sparta.own(sparta.shots))),
        ("Progressive passes", sum(1 for p in brno_pass if p["progressive"]), sum(1 for p in sparta_pass if p["progressive"])),
        ("Box entries", sum(1 for p in brno_pass if p["box_entry"]), sum(1 for p in sparta_pass if p["box_entry"])),
        ("Final-third touches", b_touch, s_touch),
        ("Pass accuracy", sum(1 for p in brno_pass if p["completed"]) / len(brno_pass),
         sum(1 for p in sparta_pass if p["completed"]) / len(sparta_pass)),
        ("Pressing (inv. PPDA)", 1 / md.compute_ppda(sparta.passes, sparta.pressing, md.BRNO_ID, md.SPARTA_ID),
         1 / sparta.ppda_for()),
    ]
    labels = [m[0] for m in metrics]
    n = len(labels)
    brno_norm = [m[1] / max(m[1], m[2], 1e-9) for m in metrics]
    sparta_norm = [m[2] / max(m[1], m[2], 1e-9) for m in metrics]

    angles = [i / n * 2 * math.pi for i in range(n)] + [0]
    for vals, color, name in ((brno_norm, OPP_C, md.team_name(md.BRNO_ID)), (sparta_norm, SPARTA_C, "Sparta Praha")):
        pts = vals + [vals[0]]
        ax.plot(angles, pts, color=color, linewidth=2.2, marker="o", markersize=4, label=name, zorder=3)
        ax.fill(angles, pts, color=color, alpha=0.15, zorder=2)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=9.5, color=palette["ink_primary"])
    ax.set_yticks([])
    ax.set_ylim(0, 1.15)
    ax.spines["polar"].set_color(palette["axis"])
    ax.grid(color=palette["grid"])

    fig.legend(loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.02),
               fontsize=10.5, labelcolor=palette["ink_secondary"])

    components.header(fig, kicker="Head To Head",
                       title="A shape comparison across the match's key numbers",
                       dek="Each axis normalized to the better of the two teams that match (=1.0)",
                       palette=palette)
    components.footer(fig, source=md.SOURCE, palette=palette)
    save(fig, "38_team_radar.png")


# ---------------------------------------------------------------------------
# 39. Shot zones heatmap
# ---------------------------------------------------------------------------

def shot_zones_heatmap(sparta):
    fig, palette = new_fig()
    pitch = new_pitch(palette)
    ax1 = fig.add_axes([0.02, 0.10, 0.47, 0.62])
    ax2 = fig.add_axes([0.51, 0.10, 0.47, 0.62])
    pitch.draw(ax=ax1)
    pitch.draw(ax=ax2)

    for ax, cid, color, name in ((ax1, md.BRNO_ID, OPP_C, md.team_name(md.BRNO_ID)), (ax2, md.SPARTA_ID, SPARTA_C, "Sparta Praha")):
        shots = [s for s in sparta.shots if s["contestantId"] == cid]
        xs = [s["x"] for s in shots]; ys = [s["y"] for s in shots]
        cmap = "Oranges" if color == SPARTA_C else "Blues"
        if xs:
            stats = pitch.bin_statistic(xs, ys, statistic="count", bins=(6, 4))
            pitch.heatmap(stats, ax=ax, cmap=cmap, edgecolors=palette["surface"], alpha=0.9, zorder=1)
        ax.set_title(f"{name} ({len(shots)} shots)", color=color, fontsize=12, fontweight="bold", family="sans-serif")

    components.header(fig, kicker="Shot Origin",
                       title="Where each side's shots came from, matchday 1",
                       dek="Shot count density by pitch zone, attacking right",
                       palette=palette)
    components.footer(fig, source=md.SOURCE, palette=palette)
    save(fig, "39_shot_zones_heatmap.png")


def main():
    sparta = md.TeamMatch(md.SPARTA_ID)
    brno = md.TeamMatch(md.BRNO_ID)
    league = md.LeagueMW1()
    sim = md.simulate_match_scorelines(brno.own(brno.shots), sparta.own(sparta.shots))

    cover(sparta)
    match_summary(sparta)
    xg_flow(sparta)
    shot_quality_table(sparta)
    goal_buildups(sparta)
    pass_network_combined(sparta)
    passing_network(sparta)
    pass_network_team(brno, "08", "brno", OPP_C)
    progressive_pass_density(sparta)
    touch_heatmap(sparta, "10", "sparta", SPARTA_C)
    touch_heatmap(brno, "11", "brno", OPP_C)
    shape_average_positions(sparta)
    ball_progression(sparta)
    field_tilt_over_time(sparta)
    possession_thirds(sparta)
    progression_bars(sparta)
    ppda_by_window(sparta)
    defensive_actions(sparta)
    duels_summary(sparta)
    discipline(sparta)
    momentum_timeline(sparta)
    impact_leaderboard(sparta)
    win_probability(sparta, sim)
    xg_scoreline_matrix(sparta, sim)
    xt_flow(sparta)
    xt_leaderboard(sparta)
    shot_assists_map(sparta)
    key_passes_leaderboard(sparta)
    crossing_map(sparta, "29", "sparta", SPARTA_C)
    final_third_entries(sparta)
    zone14_halfspace_map(sparta, "31", "sparta", SPARTA_C)
    pace_vs_volume_ranking(league)
    long_ball_targets(sparta, "34", "sparta", SPARTA_C)
    long_ball_targets(brno, "35", "brno", OPP_C)
    goal_kick_end_locations(sparta)
    box_entries_map(sparta)
    shot_map(sparta)
    set_piece_analysis(sparta)
    team_radar(sparta)
    possession_vs_field_tilt(league)
    switches_of_play(sparta, league)
    creative_zone_analysis(sparta, league)
    middle_third_halfspace_ranking(league)
    final_third_halfspace_ranking(league)
    verticality_ranking(league)
    attacking_sequence_involvements(sparta)
    shot_zones_heatmap(sparta)
    shot_conversion_vs_xg(league)
    def_line_height_ranking(league)
    report_card(sparta, brno)
    print("Done.")


if __name__ == "__main__":
    main()

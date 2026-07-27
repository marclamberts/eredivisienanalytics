"""
Bohemians 1905 vs FC Hradec Králové - Pre-Match Preview
==========================================================
Chance Liga 2026/27, matchday 2 (2026-08-02, 17:30 local, Stadion
Ďolíček, Prague). Fixture not yet played -- built 6 days out from the
match, on 2026-07-27.

Follows the visual grammar of this repo's post-match reports (Meridian
house style, housestyle/ package at the repo root) but a different data
shape: there is no shared event feed for this fixture yet, so every page
is built from each team's OWN matchday-1 fixture --
  Bohemians 1905: lost 1-3 away at FK Teplice
  FC Hradec Králové: won 2-1 at home over FK Pardubice
-- as an early-season form/style snapshot, not a head-to-head. That is
one match of data per team; every page is scoped to what a single match
can honestly support (team-level shape and volume, not multi-game trend
lines), and is labelled as such.

Data: Opta MA3 event feed (CZ Events/CZ 2026-2027), parsed in
match_data.py; shots scored with the same own distance+angle xG model as
the sibling post-match report (no provider xG in this feed).

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
from housestyle.colors import CATEGORICAL_DARK, STATUS_DARK  # noqa: E402

import match_data as md  # noqa: E402

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Visuals")
os.makedirs(OUT_DIR, exist_ok=True)

FIGSIZE = (13.33, 7.5)
BOH_C = CATEGORICAL_DARK[2]    # teal -- Bohemians 1905
HKR_C = CATEGORICAL_DARK[0]    # ink blue -- FC Hradec Kralove (same slot as the sibling post-match report)
GOOD_C = STATUS_DARK["good"]
WARN_C = STATUS_DARK["warning"]
BOH_SHORT, HKR_SHORT = "Bohemians", "Hradec Kr."

BOX_Y = (13.84, 54.16)


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


def team_color(cid):
    if cid == md.BOHEMIANS_ID:
        return BOH_C
    if cid == md.HRADEC_ID:
        return HKR_C
    return None  # opponent from a team's own MW1 game -- caller supplies a muted color


def team_short(cid):
    return md.TEAM_SHORT.get(cid, cid)


# ---------------------------------------------------------------------------
# 01. Cover
# ---------------------------------------------------------------------------

def cover():
    palette, _ = style.apply("dark")
    fig = plt.figure(figsize=FIGSIZE)
    fig.patch.set_facecolor(palette["surface"])

    fig.text(0.5, 0.64, md.FIXTURE_HOME_NAME.upper(), fontsize=34, fontweight="bold",
              color=BOH_C, family="serif", ha="center", va="center")
    fig.text(0.5, 0.555, "vs", fontsize=16, color=palette["ink_muted"],
              family="sans-serif", ha="center", va="center")
    fig.text(0.5, 0.47, md.FIXTURE_AWAY_NAME.upper(), fontsize=34, fontweight="bold",
              color=HKR_C, family="serif", ha="center", va="center")

    fig.text(0.5, 0.375, f"Kickoff {md.KICKOFF_LOCAL}, {md.MATCH_DATE}", fontsize=14,
              fontweight="bold", color=palette["ink_primary"], family="sans-serif",
              ha="center", va="center")

    fig.text(0.5, 0.30, f"{md.COMPETITION}  ·  {md.VENUE}", fontsize=12,
              color=palette["ink_secondary"], family="sans-serif", ha="center", va="center")

    fig.text(0.5, 0.19, f"{components.MARK} PRE-MATCH PREVIEW  ·  19 PAGES", fontsize=13, fontweight="bold",
              color=palette["accent"], family="sans-serif", ha="center", va="center")
    fig.text(0.5, 0.145, "Built from each team's matchday-1 fixture -- early-season form, not head-to-head",
              fontsize=9.5, color=palette["ink_muted"], family="sans-serif", ha="center", va="center")

    components.brand_mark(fig, palette=palette, right=0.94, y=0.93)
    components.footer(fig, source=md.SOURCE, palette=palette)
    save(fig, "01_cover.png")


# ---------------------------------------------------------------------------
# 02. Fixture context
# ---------------------------------------------------------------------------

def fixture_context(boh, hkr):
    fig, palette = new_fig()
    ax = fig.add_axes([0.06, 0.14, 0.88, 0.58])
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    facts = [
        ("Competition", md.COMPETITION),
        ("Venue", f"{md.VENUE} (Bohemians home)"),
        ("Kickoff", f"{md.MATCH_DATE}, {md.KICKOFF_LOCAL} local"),
        ("Round", "Both sides' 2nd league match of 2026/27"),
    ]
    y0 = 1.0
    for i, (label, val) in enumerate(facts):
        y = y0 - i * 0.075
        ax.text(0.0, y, label.upper(), fontsize=9.5, fontweight="bold", color=palette["accent"], va="top")
        ax.text(0.30, y, val, fontsize=11.5, color=palette["ink_primary"], va="top")

    ax.axhline(0.66, xmin=0, xmax=1, color=palette["axis"], linewidth=1.0)

    col_x = [0.0, 0.52]
    for x, snap, name, color in zip(col_x, (boh, hkr), (md.FIXTURE_HOME_NAME, md.FIXTURE_AWAY_NAME), (BOH_C, HKR_C)):
        y = 0.58
        ax.text(x, y, name, fontsize=14, fontweight="bold", color=color, va="top", family="serif")
        y -= 0.09
        ax.text(x, y, f"Matchday 1: {snap.result} vs {snap.opponent_name}", fontsize=10.5,
                color=palette["ink_primary"], va="top")
        y -= 0.075
        ax.text(x, y, f"xG created / conceded: {snap.xg_for:.2f} / {snap.xg_against:.2f}", fontsize=10.5,
                color=palette["ink_secondary"], va="top")
        y -= 0.075
        ax.text(x, y, f"Shots for / against: {len(snap.own(snap.shots))} / {len(snap.against(snap.shots))}",
                fontsize=10.5, color=palette["ink_secondary"], va="top")
        y -= 0.075
        ax.text(x, y, f"PPDA (pressing intensity): {snap.ppda_for():.1f}", fontsize=10.5,
                color=palette["ink_secondary"], va="top")

    fig.text(0.06, 0.155, "Every page in this preview is built from that one matchday-1 fixture per team -- "
                          "an early-season style snapshot, not a multi-game trend or a head-to-head "
                          "(these two sides have not yet met this season).",
              fontsize=9, color=palette["ink_muted"], ha="left", va="top", wrap=True)

    components.header(fig, kicker="Fixture Preview",
                       title=f"{md.FIXTURE_HOME_NAME} host {md.FIXTURE_AWAY_NAME}",
                       dek="What each side showed in their opening match of the season",
                       palette=palette)
    components.footer(fig, source=md.SOURCE, palette=palette)
    save(fig, "02_fixture_context.png")


# ---------------------------------------------------------------------------
# 03. Matchday-1 shot maps, one pitch per team (for + against)
# ---------------------------------------------------------------------------

def shot_maps_mw1(boh, hkr):
    fig, palette = new_fig()
    pitch = new_pitch(palette)
    ax1 = fig.add_axes([0.02, 0.10, 0.47, 0.62])
    ax2 = fig.add_axes([0.51, 0.10, 0.47, 0.62])
    pitch.draw(ax=ax1)
    pitch.draw(ax=ax2)

    for ax, snap, color, name in ((ax1, boh, BOH_C, md.FIXTURE_HOME_NAME), (ax2, hkr, HKR_C, md.FIXTURE_AWAY_NAME)):
        for s in snap.shots:
            is_own = s["contestantId"] == snap.team_id
            c = color if is_own else palette["ink_muted"]
            size = 70 + s["xg"] * 800
            if s["is_goal"]:
                pitch.scatter(s["x"], s["y"], ax=ax, s=size, marker="o", color=c,
                              edgecolors=palette["ink_primary"], linewidth=1.4, zorder=5)
            else:
                pitch.scatter(s["x"], s["y"], ax=ax, s=size, marker="o", facecolors="none",
                              edgecolors=c, linewidth=1.4, alpha=0.8, zorder=4)
        ax.set_title(f"{name}\n{snap.result} vs {snap.opponent_name}", color=color, fontsize=12,
                     fontweight="bold", family="sans-serif")

    legend_elems = [Line2D([0], [0], marker="o", color=palette["surface"], markerfacecolor=palette["ink_muted"],
                            markeredgecolor=palette["ink_muted"], markersize=10, label="Shot conceded", linewidth=0),
                    Line2D([0], [0], marker="o", color=palette["surface"], markerfacecolor=palette["surface"],
                            markeredgecolor=palette["ink_primary"], markersize=10, label="Own shot / goal", linewidth=0)]
    fig.legend(handles=legend_elems, loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 0.03), fontsize=10.5, labelcolor=palette["ink_secondary"])

    components.header(fig, kicker="Matchday 1",
                       title="Shot maps from each side's opening match, own goal on the left",
                       dek="Filled = goal  ·  Size = xG  ·  Own team's own colour, muted = conceded",
                       palette=palette)
    components.footer(fig, source=md.SOURCE, palette=palette)
    save(fig, "03_shot_maps_mw1.png")


# ---------------------------------------------------------------------------
# 04. xG snapshot comparison
# ---------------------------------------------------------------------------

def xg_snapshot(boh, hkr):
    fig, palette = new_fig()
    ax = fig.add_axes([0.24, 0.16, 0.66, 0.56])

    metrics = [
        ("xG created", boh.xg_for, hkr.xg_for),
        ("xG conceded", boh.xg_against, hkr.xg_against),
        ("Shots", len(boh.own(boh.shots)), len(hkr.own(hkr.shots))),
        ("Big chances", sum(1 for s in boh.own(boh.shots) if s["big_chance"]),
         sum(1 for s in hkr.own(hkr.shots) if s["big_chance"])),
    ]
    n = len(metrics)
    ypos = np.arange(n)[::-1]
    maxval = max(max(b, h) for _, b, h in metrics) * 1.25 or 1
    for y, (label, b, h) in zip(ypos, metrics):
        ax.barh(y + 0.18, b, height=0.32, color=BOH_C)
        ax.barh(y - 0.18, h, height=0.32, color=HKR_C)
        fb = f"{b:.2f}" if isinstance(b, float) else str(b)
        fh = f"{h:.2f}" if isinstance(h, float) else str(h)
        ax.text(b + maxval * 0.02, y + 0.18, fb, va="center", fontsize=10, color=palette["ink_primary"])
        ax.text(h + maxval * 0.02, y - 0.18, fh, va="center", fontsize=10, color=palette["ink_primary"])
    ax.set_yticks(ypos)
    ax.set_yticklabels([m[0] for m in metrics], fontsize=11.5, color=palette["ink_primary"])
    ax.set_xlim(0, maxval)
    ax.grid(axis="x")
    ax.set_axisbelow(True)

    legend_elems = [Line2D([0], [0], marker="s", color=palette["surface"], markerfacecolor=BOH_C,
                            markersize=12, label=md.FIXTURE_HOME_NAME, linewidth=0),
                    Line2D([0], [0], marker="s", color=palette["surface"], markerfacecolor=HKR_C,
                            markersize=12, label=md.FIXTURE_AWAY_NAME, linewidth=0)]
    fig.legend(handles=legend_elems, loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 0.02), fontsize=10.5, labelcolor=palette["ink_secondary"])

    components.header(fig, kicker="Chance Quality",
                       title="Hradec created more, and gave up far less, on opening day",
                       dek="Matchday-1 shot numbers, own xG model  ·  one match per side",
                       palette=palette)
    components.footer(fig, source=md.SOURCE, palette=palette)
    save(fig, "04_xg_snapshot.png")


# ---------------------------------------------------------------------------
# 05-06. Pass network -- one page per team
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


def _draw_pass_network(ax, team_passes, color, palette, pitch, min_passes=6, node_scale=1.15, label_size=8.6):
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


def pass_network_team_page(snap, color, name, page_num, slug):
    fig, palette = new_fig()
    pitch = new_pitch(palette)
    team_passes = snap.own(snap.passes)

    ax = fig.add_axes([0.02, 0.10, 0.54, 0.62])
    _draw_pass_network(ax, team_passes, color, palette, pitch)

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
                       title=f"{name}: how they built play at matchday 1",
                       dek=f"Average completed-pass position (≥ 6 passes) vs {snap.opponent_name}, attacking right",
                       palette=palette)
    components.footer(fig, source=md.SOURCE, palette=palette)
    save(fig, f"{page_num}_pass_network_{slug}.png")


# ---------------------------------------------------------------------------
# 07-08. Touch heatmap -- one page per team
# ---------------------------------------------------------------------------

def touch_heatmap_team_page(snap, color, name, page_num, slug, cmap):
    fig, palette = new_fig()
    pitch = new_pitch(palette)
    ax = fig.add_axes([0.06, 0.10, 0.88, 0.62])
    pitch.draw(ax=ax)

    own = snap.own(snap.touches)
    xs = [t["x"] for t in own]
    ys = [t["y"] for t in own]
    stats = pitch.bin_statistic(xs, ys, statistic="count", bins=(9, 6))
    pitch.heatmap(stats, ax=ax, cmap=cmap, edgecolors=palette["surface"], alpha=0.92, zorder=1)

    components.header(fig, kicker="Territory",
                       title=f"{name}: where they spent their {len(xs)} touches at matchday 1",
                       dek=f"Touch density by pitch zone vs {snap.opponent_name}, attacking right",
                       palette=palette)
    components.footer(fig, source=md.SOURCE, palette=palette)
    save(fig, f"{page_num}_touch_heatmap_{slug}.png")


# ---------------------------------------------------------------------------
# 09. Possession thirds
# ---------------------------------------------------------------------------

def possession_thirds(boh, hkr):
    fig, palette = new_fig()
    ax = fig.add_axes([0.16, 0.24, 0.68, 0.40])

    zone_colors = [CATEGORICAL_DARK[0], CATEGORICAL_DARK[3], CATEGORICAL_DARK[1]]
    zone_labels = ["Defensive", "Middle", "Attacking"]

    def thirds(snap):
        t = snap.own(snap.touches)
        d = sum(1 for x in t if x["x"] < 35)
        m = sum(1 for x in t if 35 <= x["x"] < 70)
        a = sum(1 for x in t if x["x"] >= 70)
        total = d + m + a
        return [d / total, m / total, a / total], total

    for i, (snap, name, color) in enumerate(((boh, md.FIXTURE_HOME_NAME, BOH_C), (hkr, md.FIXTURE_AWAY_NAME, HKR_C))):
        fracs, total = thirds(snap)
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
                       title="Each side's territory split at matchday 1",
                       dek="Distribution of touches across pitch thirds, own attacking direction",
                       palette=palette)
    components.footer(fig, source=md.SOURCE, palette=palette)
    save(fig, "09_possession_thirds.png")


# ---------------------------------------------------------------------------
# 10. Progression comparison bars
# ---------------------------------------------------------------------------

def progression_bars(boh, hkr):
    fig, palette = new_fig()
    ax = fig.add_axes([0.24, 0.16, 0.66, 0.56])

    bp = boh.own(boh.passes)
    hp = hkr.own(hkr.passes)
    metrics = [
        ("Progressive passes", sum(1 for p in bp if p["progressive"]), sum(1 for p in hp if p["progressive"])),
        ("Final-third entries", sum(1 for p in bp if p["final_third_entry"]),
         sum(1 for p in hp if p["final_third_entry"])),
        ("Passes into the box", sum(1 for p in bp if p["box_entry"]), sum(1 for p in hp if p["box_entry"])),
        ("Completed crosses", sum(1 for p in bp if p["is_cross"] and p["completed"]),
         sum(1 for p in hp if p["is_cross"] and p["completed"])),
    ]
    n = len(metrics)
    ypos = np.arange(n)[::-1]
    maxval = max(max(b, h) for _, b, h in metrics) * 1.15
    for y, (label, b, h) in zip(ypos, metrics):
        ax.barh(y + 0.18, b, height=0.32, color=BOH_C)
        ax.barh(y - 0.18, h, height=0.32, color=HKR_C)
        ax.text(b + maxval * 0.015, y + 0.18, str(b), va="center", fontsize=10, color=palette["ink_primary"])
        ax.text(h + maxval * 0.015, y - 0.18, str(h), va="center", fontsize=10, color=palette["ink_primary"])
    ax.set_yticks(ypos)
    ax.set_yticklabels([m[0] for m in metrics], fontsize=11.5, color=palette["ink_primary"])
    ax.set_xlim(0, maxval)
    ax.set_xlabel("Count")
    ax.grid(axis="x")
    ax.set_axisbelow(True)

    legend_elems = [Line2D([0], [0], marker="s", color=palette["surface"], markerfacecolor=BOH_C,
                            markersize=12, label=md.FIXTURE_HOME_NAME, linewidth=0),
                    Line2D([0], [0], marker="s", color=palette["surface"], markerfacecolor=HKR_C,
                            markersize=12, label=md.FIXTURE_AWAY_NAME, linewidth=0)]
    fig.legend(handles=legend_elems, loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 0.02), fontsize=10.5, labelcolor=palette["ink_secondary"])

    components.header(fig, kicker="Progression",
                       title="How each side moved the ball forward at matchday 1",
                       dek="Progressive pass = completed pass cutting ≥25% off the distance to goal",
                       palette=palette)
    components.footer(fig, source=md.SOURCE, palette=palette)
    save(fig, "10_progression_bars.png")


# ---------------------------------------------------------------------------
# 11. PPDA / pressing comparison
# ---------------------------------------------------------------------------

def ppda_pressing(boh, hkr):
    fig, palette = new_fig()
    ax1 = fig.add_axes([0.08, 0.16, 0.40, 0.58])
    ax2 = fig.add_axes([0.56, 0.16, 0.40, 0.58])

    buckets = [(0, 15), (15, 30), (30, 45), (45, 60), (60, 75), (75, 96)]
    labels = ["0-15", "15-30", "30-45", "45-60", "60-75", "75-90+"]

    def bucketed(snap):
        return [md.compute_ppda(snap.passes, snap.pressing, snap.team_id, snap.opponent_id, lo, hi)
                for lo, hi in buckets]

    for ax, snap, color, name in ((ax1, boh, BOH_C, md.FIXTURE_HOME_NAME), (ax2, hkr, HKR_C, md.FIXTURE_AWAY_NAME)):
        vals = bucketed(snap)
        finite = [v for v in vals if not math.isnan(v)]
        top = max(finite) * 1.15 if finite else 1.0
        xs = np.arange(len(labels))
        clean = [v if not math.isnan(v) else 0 for v in vals]
        ax.bar(xs, clean, color=color)
        for x, v in zip(xs, vals):
            if not math.isnan(v):
                ax.text(x, v + top * 0.02, f"{v:.1f}", ha="center", fontsize=9.5,
                        color=palette["ink_primary"], fontweight="bold")
        overall = snap.ppda_for()
        ax.axhline(overall, color=palette["ink_muted"], linestyle="--", linewidth=1.0)
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, fontsize=8.5)
        ax.set_ylim(0, top)
        ax.set_title(f"{name}\nOverall PPDA: {overall:.1f}", color=color, fontsize=11.5,
                     fontweight="bold", family="sans-serif")
        ax.set_ylabel("PPDA")

    components.header(fig, kicker="Pressing",
                       title="Pressing intensity at matchday 1, by 15-minute window",
                       dek="Passes per defensive action in the opponent's own 60%  ·  lower = more intense press",
                       palette=palette)
    components.footer(fig, source=md.SOURCE, palette=palette)
    save(fig, "11_ppda_pressing.png")


# ---------------------------------------------------------------------------
# 12. Defensive actions
# ---------------------------------------------------------------------------

def defensive_actions(boh, hkr):
    fig, palette = new_fig()
    pitch = new_pitch(palette)
    ax1 = fig.add_axes([0.02, 0.10, 0.47, 0.62])
    ax2 = fig.add_axes([0.51, 0.10, 0.47, 0.62])
    pitch.draw(ax=ax1)
    pitch.draw(ax=ax2)

    markers = {"Tackle": "o", "Interception": "D", "Clearance": "s"}
    action_colors = {"Tackle": CATEGORICAL_DARK[2], "Interception": CATEGORICAL_DARK[3],
                      "Clearance": palette["ink_muted"]}

    for ax, snap, color, name in ((ax1, boh, BOH_C, md.FIXTURE_HOME_NAME), (ax2, hkr, HKR_C, md.FIXTURE_AWAY_NAME)):
        team_defs = snap.own(snap.defs)
        for action, marker in markers.items():
            pts = [d for d in team_defs if d["action"] == action]
            if not pts:
                continue
            xs = [p["x"] for p in pts]
            ys = [p["y"] for p in pts]
            pitch.scatter(xs, ys, ax=ax, s=80, marker=marker, color=action_colors[action],
                          edgecolors=palette["surface"], linewidth=0.6, alpha=0.9, zorder=4)
        counts = {a: sum(1 for d in team_defs if d["action"] == a) for a in markers}
        title = f"{name}\nTkl {counts['Tackle']}  ·  Int {counts['Interception']}  ·  Clr {counts['Clearance']}"
        ax.set_title(title, color=color, fontsize=12, fontweight="bold", family="sans-serif")

    legend_elems = [Line2D([0], [0], marker=markers[a], color=palette["surface"], markerfacecolor=action_colors[a],
                            markersize=10, label=a, linewidth=0) for a in markers]
    fig.legend(handles=legend_elems, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 0.02), fontsize=10.5, labelcolor=palette["ink_secondary"])

    components.header(fig, kicker="Defending",
                       title="Where each side won the ball back at matchday 1",
                       dek="Tackles, interceptions and clearances, own goal on the left, attacking right",
                       palette=palette)
    components.footer(fig, source=md.SOURCE, palette=palette)
    save(fig, "12_defensive_actions.png")


# ---------------------------------------------------------------------------
# 13. Duels & discipline
# ---------------------------------------------------------------------------

def duels_discipline(boh, hkr):
    fig, palette = new_fig()
    ax1 = fig.add_axes([0.08, 0.16, 0.55, 0.58])
    ax2 = fig.add_axes([0.72, 0.16, 0.24, 0.58])

    kinds = ["Tackle", "Aerial", "Challenge"]
    n = len(kinds)
    ypos = np.arange(n)[::-1]
    maxval = 0
    rows = []
    for kind in kinds:
        b = [d for d in boh.own(boh.duels) if d["action"] == kind]
        h = [d for d in hkr.own(hkr.duels) if d["action"] == kind]
        b_won = sum(1 for d in b if d["success"])
        h_won = sum(1 for d in h if d["success"])
        rows.append((kind, len(b), b_won, len(h), h_won))
        maxval = max(maxval, len(b), len(h))
    maxval *= 1.25

    for y, (kind, b_n, b_won, h_n, h_won) in zip(ypos, rows):
        b_rate = b_won / b_n if b_n else 0
        h_rate = h_won / h_n if h_n else 0
        ax1.barh(y + 0.18, b_n, height=0.32, color=palette["axis"])
        ax1.barh(y + 0.18, b_won, height=0.32, color=BOH_C)
        ax1.barh(y - 0.18, h_n, height=0.32, color=palette["axis"])
        ax1.barh(y - 0.18, h_won, height=0.32, color=HKR_C)
        ax1.text(b_n + maxval * 0.015, y + 0.18, f"{b_won}/{b_n} ({b_rate:.0%})", va="center",
                fontsize=9, color=palette["ink_primary"])
        ax1.text(h_n + maxval * 0.015, y - 0.18, f"{h_won}/{h_n} ({h_rate:.0%})", va="center",
                fontsize=9, color=palette["ink_primary"])
    ax1.set_yticks(ypos)
    ax1.set_yticklabels([f"{k} duels" for k in kinds], fontsize=11, color=palette["ink_primary"])
    ax1.set_xlim(0, maxval)
    ax1.set_xlabel("Contested (solid = won)")
    ax1.grid(axis="x")
    ax1.set_axisbelow(True)
    ax1.set_title("Duel win rates", color=palette["ink_primary"], fontsize=11.5, fontweight="bold",
                   family="sans-serif")

    def fouls_cards(snap):
        fouls = sum(1 for d in snap.own(snap.pressing) if d["action"] == "Foul")
        yellow = sum(1 for c in snap.own(snap.cards) if c["kind"] == "Yellow")
        red = sum(1 for c in snap.own(snap.cards) if c["kind"] in ("Red", "2nd Yellow"))
        return fouls, yellow, red

    b_fouls, b_yellow, b_red = fouls_cards(boh)
    h_fouls, h_yellow, h_red = fouls_cards(hkr)
    disc_metrics = [("Fouls", b_fouls, h_fouls), ("Yellows", b_yellow, h_yellow), ("Reds", b_red, h_red)]
    n2 = len(disc_metrics)
    ypos2 = np.arange(n2)[::-1]
    maxval2 = max(max(b, h) for _, b, h in disc_metrics) * 1.4 or 1
    for y, (label, b, h) in zip(ypos2, disc_metrics):
        ax2.barh(y + 0.18, b, height=0.32, color=BOH_C)
        ax2.barh(y - 0.18, h, height=0.32, color=HKR_C)
        ax2.text(b + maxval2 * 0.03, y + 0.18, str(b), va="center", fontsize=9, color=palette["ink_primary"])
        ax2.text(h + maxval2 * 0.03, y - 0.18, str(h), va="center", fontsize=9, color=palette["ink_primary"])
    ax2.set_yticks(ypos2)
    ax2.set_yticklabels([m[0] for m in disc_metrics], fontsize=10, color=palette["ink_primary"])
    ax2.set_xlim(0, maxval2)
    ax2.set_title("Discipline", color=palette["ink_primary"], fontsize=11.5, fontweight="bold", family="sans-serif")

    legend_elems = [Line2D([0], [0], marker="s", color=palette["surface"], markerfacecolor=BOH_C,
                            markersize=12, label=md.FIXTURE_HOME_NAME, linewidth=0),
                    Line2D([0], [0], marker="s", color=palette["surface"], markerfacecolor=HKR_C,
                            markersize=12, label=md.FIXTURE_AWAY_NAME, linewidth=0)]
    fig.legend(handles=legend_elems, loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 0.03), fontsize=10.5, labelcolor=palette["ink_secondary"])

    components.header(fig, kicker="Physicality",
                       title="Duels contested and discipline at matchday 1",
                       dek="Tackle, aerial and loose-ball duels, plus fouls and cards",
                       palette=palette)
    components.footer(fig, source=md.SOURCE, palette=palette)
    save(fig, "13_duels_discipline.png")


# ---------------------------------------------------------------------------
# 14-15. Key players to watch -- one page per team
# ---------------------------------------------------------------------------

def key_players_team_page(snap, color, name, page_num, slug):
    fig, palette = new_fig()
    ax = fig.add_axes([0.10, 0.16, 0.80, 0.56])

    def score_players():
        scores = {}
        for s in snap.own(snap.shots):
            scores[s["player"]] = scores.get(s["player"], 0) + s["xg"] + (3.0 if s["is_goal"] else 0)
        for p in snap.own(snap.passes):
            scores[p["player"]] = scores.get(p["player"], 0) + 0.15 * p["progressive"] + 0.35 * p["box_entry"]
        for d in snap.own(snap.defs):
            scores[d["player"]] = scores.get(d["player"], 0) + 0.3
        return sorted(scores.items(), key=lambda kv: -kv[1])[:8]

    top = score_players()[::-1]
    ypos = np.arange(len(top))
    vals = [v for _, v in top]
    ax.barh(ypos, vals, color=color)
    ax.set_yticks(ypos)
    ax.set_yticklabels([p for p, _ in top], fontsize=11, color=palette["ink_primary"])
    for y, v in zip(ypos, vals):
        ax.text(v + max(vals, default=1) * 0.02, y, f"{v:.1f}", va="center", fontsize=9.5,
                color=palette["ink_secondary"])
    ax.set_xlabel("Impact score (matchday 1)")

    fig.text(0.5, 0.10, "Simple composite: xG + 3×goals + 0.15×progressive pass + 0.35×box entry + "
                         "0.3×defensive action  ·  not an official rating, one match of evidence",
              ha="center", fontsize=8.8, color=palette["ink_muted"])

    components.header(fig, kicker="Players To Watch",
                       title=f"{name}: who stood out at matchday 1",
                       dek=f"vs {snap.opponent_name}  ·  shooting, progression and defending combined",
                       palette=palette)
    components.footer(fig, source=md.SOURCE, palette=palette)
    save(fig, f"{page_num}_key_players_{slug}.png")


# ---------------------------------------------------------------------------
# 16. Team radar
# ---------------------------------------------------------------------------

def team_radar(boh, hkr):
    fig, palette = new_fig()
    ax = fig.add_axes([0.26, 0.18, 0.48, 0.58], polar=True)

    bp = boh.own(boh.passes)
    hp = hkr.own(hkr.passes)
    b_touch = sum(1 for t in boh.own(boh.touches) if t["x"] >= 70)
    h_touch = sum(1 for t in hkr.own(hkr.touches) if t["x"] >= 70)

    metrics = [
        ("xG created", boh.xg_for, hkr.xg_for),
        ("xG conceded (inv.)", 1 / max(boh.xg_against, 0.05), 1 / max(hkr.xg_against, 0.05)),
        ("Shots", len(boh.own(boh.shots)), len(hkr.own(hkr.shots))),
        ("Progressive passes", sum(1 for p in bp if p["progressive"]), sum(1 for p in hp if p["progressive"])),
        ("Box entries", sum(1 for p in bp if p["box_entry"]), sum(1 for p in hp if p["box_entry"])),
        ("Final-third touches", b_touch, h_touch),
        ("Pass accuracy", sum(1 for p in bp if p["completed"]) / len(bp), sum(1 for p in hp if p["completed"]) / len(hp)),
        ("Pressing (inv. PPDA)", 1 / boh.ppda_for(), 1 / hkr.ppda_for()),
    ]
    labels = [m[0] for m in metrics]
    n = len(labels)
    boh_norm = [m[1] / max(m[1], m[2], 1e-9) for m in metrics]
    hkr_norm = [m[2] / max(m[1], m[2], 1e-9) for m in metrics]

    angles = [i / n * 2 * math.pi for i in range(n)] + [0]
    for vals, color, name in ((boh_norm, BOH_C, md.FIXTURE_HOME_NAME), (hkr_norm, HKR_C, md.FIXTURE_AWAY_NAME)):
        pts = vals + [vals[0]]
        ax.plot(angles, pts, color=color, linewidth=2.2, marker="o", markersize=4, label=name, zorder=3)
        ax.fill(angles, pts, color=color, alpha=0.15, zorder=2)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=8.8, color=palette["ink_primary"])
    ax.set_yticks([])
    ax.set_ylim(0, 1.15)
    ax.spines["polar"].set_color(palette["axis"])
    ax.grid(color=palette["grid"])

    fig.legend(loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.02),
               fontsize=10.5, labelcolor=palette["ink_secondary"])

    components.header(fig, kicker="Shape Comparison",
                       title="Matchday-1 numbers side by side",
                       dek="Each axis normalized to the better of the two teams that match (=1.0)  ·  different opponents",
                       palette=palette)
    components.footer(fig, source=md.SOURCE, palette=palette)
    save(fig, "16_team_radar.png")


# ---------------------------------------------------------------------------
# 17. Keys to the game (narrative)
# ---------------------------------------------------------------------------

def keys_to_the_game(boh, hkr):
    palette, _ = style.apply("dark")
    fig = plt.figure(figsize=FIGSIZE)
    fig.patch.set_facecolor(palette["surface"])

    points = []
    if hkr.xg_against < boh.xg_against:
        points.append(f"Hradec conceded far less ({hkr.xg_against:.2f} xG) than Bohemians did "
                       f"({boh.xg_against:.2f} xG) -- Bohemians' back line will need a cleaner day than "
                       f"the one at Teplice.")
    if boh.ppda_for() < hkr.ppda_for():
        points.append(f"Bohemians pressed higher on opening day (PPDA {boh.ppda_for():.1f} vs "
                       f"{hkr.ppda_for():.1f}) -- expect them to try to disrupt Hradec's build-up early "
                       f"rather than sit off.")
    else:
        points.append(f"Hradec pressed higher on opening day (PPDA {hkr.ppda_for():.1f} vs "
                       f"{boh.ppda_for():.1f}) -- if that intensity travels, Bohemians' buildup play will "
                       f"be under pressure from the first whistle.")
    b_touch = sum(1 for t in boh.own(boh.touches) if t["x"] >= 70)
    h_touch = sum(1 for t in hkr.own(hkr.touches) if t["x"] >= 70)
    if h_touch > b_touch:
        points.append(f"Hradec spent more of their opener in the final third ({h_touch} touches there vs "
                       f"Bohemians' {b_touch}) -- if that territorial edge travels away from home, Bohemians "
                       "will need to defend deep for longer spells.")
    else:
        points.append(f"Bohemians actually had more final-third touches in their opener ({b_touch} vs "
                       f"Hradec's {h_touch}) despite losing it -- territory alone did not convert into "
                       "chances or points, and won't be enough here either without better shot quality.")
    points.append(f"Bohemians play at home at {md.VENUE.split(',')[0]}, where their season starts against "
                   f"a Hradec side that has won its opener; the double edge of first home game plus facing "
                   "the form team is the clearest storyline in.")

    fig.text(0.5, 0.045, "★ Small-sample caveat: every number above is drawn from one match per team "
                          "(matchday 1). Treat as early style signal, not settled form.",
              ha="center", fontsize=9, color=palette["ink_muted"], style="italic")

    ax = fig.add_axes([0.08, 0.16, 0.84, 0.58])
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    row_h = 1.0 / len(points)
    for i, txt in enumerate(points):
        y = 1.0 - (i + 0.15) * row_h
        ax.text(0.0, y, f"{i + 1}.", fontsize=15, fontweight="bold", color=palette["accent"], va="top")
        ax.text(0.06, y, txt, fontsize=12, color=palette["ink_primary"], va="top", wrap=True,
                linespacing=1.5)

    components.header(fig, kicker="Keys To The Game",
                       title="Four things worth watching for on 2026-08-02",
                       dek="Reasoned from each side's matchday-1 numbers",
                       palette=palette)
    components.footer(fig, source=md.SOURCE, palette=palette)
    save(fig, "17_keys_to_the_game.png")


# ---------------------------------------------------------------------------
# 18. Win probability (Monte Carlo, cross-paired MW1 shot lists)
# ---------------------------------------------------------------------------

def _donut(ax, frac, color, palette, label, sublabel):
    ax.pie([frac, 1 - frac], radius=1.0, startangle=90, counterclock=False,
           colors=[color, palette["axis"]], wedgeprops=dict(width=0.32, edgecolor=palette["surface"], linewidth=1.5))
    ax.text(0, 0.12, f"{frac:.0%}", ha="center", va="center", fontsize=20, fontweight="bold", color=palette["ink_primary"])
    ax.text(0, -0.12, sublabel, ha="center", va="center", fontsize=8.5, color=palette["ink_muted"])
    ax.set_title(label, color=color, fontsize=11.5, fontweight="bold", family="sans-serif", pad=2)


def win_probability(boh, hkr, sim):
    fig, palette = new_fig()
    ax1 = fig.add_axes([0.06, 0.34, 0.19, 0.32])
    ax2 = fig.add_axes([0.28, 0.34, 0.19, 0.32])
    _donut(ax1, sim["home_win"], BOH_C, palette, BOH_SHORT, "WIN PROBABILITY")
    _donut(ax2, sim["away_win"], HKR_C, palette, HKR_SHORT, "WIN PROBABILITY")
    fig.text(0.275, 0.30, f"Draw: {sim['draw']:.0%}", ha="center", fontsize=10.5,
              color=palette["ink_secondary"], fontweight="bold")

    ax3 = fig.add_axes([0.56, 0.20, 0.38, 0.52])
    cap = sim["cap"]
    n = sim["n"]
    grid = {}
    for (h, a), c in sim["score_counts"].items():
        grid[(h, a)] = c / n
    top_scores = sorted(grid.items(), key=lambda kv: -kv[1])[:6]
    labels = [f"{h}-{a}" if h < cap and a < cap else f"{h}+{'-' if h>=cap else ''}{a}{'+' if a>=cap else ''}"
              for (h, a), _ in top_scores]
    vals = [v for _, v in top_scores]
    ypos = np.arange(len(vals))[::-1]
    ax3.barh(ypos, vals, color=palette["accent"])
    ax3.set_yticks(ypos)
    ax3.set_yticklabels(labels, fontsize=10)
    for y, v in zip(ypos, vals):
        ax3.text(v + max(vals) * 0.02, y, f"{v:.1%}", va="center", fontsize=9.5, color=palette["ink_primary"])
    ax3.set_xlim(0, max(vals) * 1.25)
    ax3.set_xlabel("Simulated probability")
    ax3.set_title(f"Most likely scorelines ({BOH_SHORT}–{HKR_SHORT})", color=palette["ink_primary"],
                   fontsize=11.5, fontweight="bold", family="sans-serif")
    ax3.grid(axis="x")
    ax3.set_axisbelow(True)

    components.header(fig, kicker="Match Projection",
                       title=(f"{md.FIXTURE_HOME_NAME} rate as favourites ({sim['home_win']:.0%})"
                              if sim["home_win"] >= sim["away_win"]
                              else f"{md.FIXTURE_AWAY_NAME} rate as favourites ({sim['away_win']:.0%})"),
                       dek=f"{n:,}-simulation Monte Carlo cross-pairing each side's matchday-1 shot xG list "
                           "-- an early-season projection, not a fitted model",
                       palette=palette)
    components.footer(fig, source=md.SOURCE, palette=palette)
    save(fig, "18_win_probability.png")


# ---------------------------------------------------------------------------
# 19. Report card (closing summary)
# ---------------------------------------------------------------------------

def report_card(boh, hkr, sim):
    palette, _ = style.apply("dark")
    fig = plt.figure(figsize=FIGSIZE)
    fig.patch.set_facecolor(palette["surface"])

    fig.text(0.5, 0.90, f"{md.FIXTURE_HOME_NAME}  vs  {md.FIXTURE_AWAY_NAME}",
              fontsize=18, fontweight="bold", color=palette["ink_primary"], family="serif",
              ha="center", va="center")
    fig.text(0.5, 0.855, f"{md.COMPETITION}  ·  {md.VENUE}  ·  {md.MATCH_DATE}, {md.KICKOFF_LOCAL} local",
              fontsize=10.5, color=palette["ink_secondary"], ha="center", va="center")

    bp = boh.own(boh.passes)
    hp = hkr.own(hkr.passes)
    rows = [
        ("Matchday-1 result", boh.result, hkr.result),
        ("xG created", f"{boh.xg_for:.2f}", f"{hkr.xg_for:.2f}"),
        ("xG conceded", f"{boh.xg_against:.2f}", f"{hkr.xg_against:.2f}"),
        ("Pass accuracy", f"{sum(1 for p in bp if p['completed']) / len(bp):.0%}",
         f"{sum(1 for p in hp if p['completed']) / len(hp):.0%}"),
        ("PPDA", f"{boh.ppda_for():.1f}", f"{hkr.ppda_for():.1f}"),
        ("Projected win prob.", f"{sim['home_win']:.0%}", f"{sim['away_win']:.0%}"),
    ]

    ax = fig.add_axes([0.14, 0.20, 0.72, 0.55])
    ax.axis("off")
    ax.text(0.0, 1.0, md.FIXTURE_HOME_NAME, fontsize=12.5, fontweight="bold", color=BOH_C, ha="left", va="top")
    ax.text(1.0, 1.0, md.FIXTURE_AWAY_NAME, fontsize=12.5, fontweight="bold", color=HKR_C, ha="right", va="top")
    n = len(rows)
    for i, (label, bval, hval) in enumerate(rows):
        y = 0.85 - i * (0.85 / n)
        ax.text(0.0, y, bval, fontsize=13, fontweight="bold", color=palette["ink_primary"], ha="left", va="top")
        ax.text(0.5, y, label, fontsize=10.5, color=palette["ink_muted"], ha="center", va="top")
        ax.text(1.0, y, hval, fontsize=13, fontweight="bold", color=palette["ink_primary"], ha="right", va="top")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    components.brand_mark(fig, palette=palette, right=0.94, y=0.965)
    components.footer(fig, source=md.SOURCE, palette=palette)
    save(fig, "19_report_card.png")


def main():
    boh = md.TeamSnapshot(md.BOHEMIANS_ID)
    hkr = md.TeamSnapshot(md.HRADEC_ID)
    sim = md.simulate_scorelines(boh.own(boh.shots), hkr.own(hkr.shots))

    cover()
    fixture_context(boh, hkr)
    shot_maps_mw1(boh, hkr)
    xg_snapshot(boh, hkr)
    pass_network_team_page(boh, BOH_C, md.FIXTURE_HOME_NAME, "05", "bohemians")
    pass_network_team_page(hkr, HKR_C, md.FIXTURE_AWAY_NAME, "06", "hradec")
    touch_heatmap_team_page(boh, BOH_C, md.FIXTURE_HOME_NAME, "07", "bohemians", "Greens")
    touch_heatmap_team_page(hkr, HKR_C, md.FIXTURE_AWAY_NAME, "08", "hradec", "Blues")
    possession_thirds(boh, hkr)
    progression_bars(boh, hkr)
    ppda_pressing(boh, hkr)
    defensive_actions(boh, hkr)
    duels_discipline(boh, hkr)
    key_players_team_page(boh, BOH_C, md.FIXTURE_HOME_NAME, "14", "bohemians")
    key_players_team_page(hkr, HKR_C, md.FIXTURE_AWAY_NAME, "15", "hradec")
    team_radar(boh, hkr)
    keys_to_the_game(boh, hkr)
    win_probability(boh, hkr, sim)
    report_card(boh, hkr, sim)
    print("Done.")


if __name__ == "__main__":
    main()

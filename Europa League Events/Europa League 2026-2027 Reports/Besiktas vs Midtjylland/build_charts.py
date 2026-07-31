"""
Besiktas vs FC Midtjylland - Two-Legged Tie Report
====================================================
UEFA Europa League 2026/27, 2nd Qualifying Round. Besiktas beat Midtjylland
1-0 at home (Kokcu 25', 2026-07-23, Tupras Stadyumu -- Midtjylland's Etim
sent off in the 14th minute) then 2-0 away (Rashica 69', Kokcu 75',
2026-07-30, MCH Arena -- Midtjylland's Erlic sent off in the 52nd minute)
to advance 3-0 on aggregate despite conceding the larger share of the
tie's underlying chances (2.75 xG against vs 2.03 for -- see match_data.py
Leg.besiktas_xg/opponent_xg).

Built in Marc Lamberts' Meridian house style (dark mode) -- housestyle/
package at the repo root -- following the page conventions of this repo's
CZ Events post-match reports, but reframed as a TEAM report rather than a
single-fixture one: Besiktas is "us" throughout, Midtjylland is always
"the opponent", and every page either shows one leg at a time or pools
Besiktas' own numbers across both legs, rather than a single match's two
contestants. Besiktas attack toward x=100 in every pitch chart in both
legs (match_data.norm_xy rotates each leg's own two periods independently
of home/away venue), so no per-leg orientation flips are needed here.

Data: Opta MA1 event feed, one file per leg (Europa League Events/Europa
League 2026-2027), parsed in match_data.py; shots scored with a small own
distance+angle xG model (no provider xG in this feed).

Usage: python3 build_charts.py
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
BES_C = CATEGORICAL_DARK[0]     # ink blue -- Besiktas, "us", both legs
OPP_C = CATEGORICAL_DARK[1]     # terracotta -- Midtjylland, "the opponent"
GOOD_C = STATUS_DARK["good"]
WARN_C = STATUS_DARK["warning"]
CRIT_C = STATUS_DARK["critical"]
BES_SHORT = "Besiktas"
OPP_SHORT = "Midtjylland"


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
    return BES_C if cid == md.BESIKTAS_ID else OPP_C


def team_short(cid):
    return BES_SHORT if cid == md.BESIKTAS_ID else OPP_SHORT


# ---------------------------------------------------------------------------
# 01. Cover
# ---------------------------------------------------------------------------

def cover(leg1, leg2):
    palette, _ = style.apply("dark")
    fig = plt.figure(figsize=FIGSIZE)
    fig.patch.set_facecolor(palette["surface"])

    agg_h = leg1.besiktas_goals + leg2.besiktas_goals
    agg_a = leg1.opponent_goals + leg2.opponent_goals

    fig.text(0.5, 0.665, "BESIKTAS", fontsize=36, fontweight="bold", color=BES_C,
              family="serif", ha="center", va="center")
    fig.text(0.5, 0.575, "vs", fontsize=16, color=palette["ink_muted"], family="sans-serif",
              ha="center", va="center")
    fig.text(0.5, 0.485, "FC MIDTJYLLAND", fontsize=30, fontweight="bold", color=OPP_C,
              family="serif", ha="center", va="center")

    fig.text(0.5, 0.375, f"{agg_h} – {agg_a}", fontsize=34, fontweight="bold",
              color=palette["ink_primary"], family="sans-serif", ha="center", va="center")
    fig.text(0.5, 0.315, "AGGREGATE ACROSS TWO LEGS", fontsize=10.5, color=palette["ink_muted"],
              family="sans-serif", ha="center", va="center")

    fig.text(0.5, 0.245, f"Leg 1: {leg1.score_line}   ·   {leg1.venue}   ·   {leg1.date}",
              fontsize=10.5, color=palette["ink_secondary"], family="sans-serif", ha="center", va="center")
    fig.text(0.5, 0.205, f"Leg 2: {leg2.score_line}   ·   {leg2.venue}   ·   {leg2.date}",
              fontsize=10.5, color=palette["ink_secondary"], family="sans-serif", ha="center", va="center")

    fig.text(0.5, 0.125, f"{components.MARK} BESIKTAS ADVANCE TO THE 3RD QUALIFYING ROUND", fontsize=12.5,
              fontweight="bold", color=palette["accent"], family="sans-serif", ha="center", va="center")
    fig.text(0.5, 0.088, md.COMPETITION, fontsize=10, color=palette["ink_muted"],
              family="sans-serif", ha="center", va="center")

    components.brand_mark(fig, palette=palette, right=0.94, y=0.93)
    components.footer(fig, source=md.SOURCE, palette=palette)
    save(fig, "01_cover.png")


# ---------------------------------------------------------------------------
# 02. Tie summary
# ---------------------------------------------------------------------------

def tie_summary(leg1, leg2):
    fig, palette = new_fig()

    bes_shots = leg1.besiktas(leg1.shots) + leg2.besiktas(leg2.shots)
    opp_shots = leg1.opponent(leg1.shots) + leg2.opponent(leg2.shots)
    bes_pass = leg1.besiktas(leg1.passes) + leg2.besiktas(leg2.passes)
    opp_pass = leg1.opponent(leg1.passes) + leg2.opponent(leg2.passes)
    bes_touch = len(leg1.besiktas(leg1.touches)) + len(leg2.besiktas(leg2.touches))
    opp_touch = len(leg1.opponent(leg1.touches)) + len(leg2.opponent(leg2.touches))
    bes_xg = sum(s["xg"] for s in bes_shots)
    opp_xg = sum(s["xg"] for s in opp_shots)

    ax0 = fig.add_axes([0.06, 0.665, 0.88, 0.13])
    ax0.axis("off")
    ax0.set_xlim(0, 1)
    ax0.set_ylim(-0.5, 1.5)
    for i, leg in enumerate((leg1, leg2)):
        y = 1 - i
        goals_txt = ", ".join(f"{s['player']} {s['minute']}′" for s in leg.besiktas(leg.shots)
                               if s["is_goal"]) or "–"
        ax0.text(0.0, y, f"LEG {leg.leg_num}", fontsize=10, fontweight="bold", color=palette["ink_muted"], va="center")
        ax0.text(0.13, y, leg.score_line, fontsize=13.5, fontweight="bold", color=palette["ink_primary"], va="center")
        ax0.text(0.42, y, leg.venue, fontsize=9.5, color=palette["ink_secondary"], va="center")
        ax0.text(0.70, y, goals_txt, fontsize=9.5, color=BES_C, va="center")

    ax2 = fig.add_axes([0.28, 0.12, 0.44, 0.50])
    ax2.axis("off")
    rows = [
        ("Expected goals", f"{bes_xg:.2f}", f"{opp_xg:.2f}", bes_xg, opp_xg),
        ("Shots (on target)",
         f"{len(bes_shots)} ({sum(1 for s in bes_shots if s['on_target'])})",
         f"{len(opp_shots)} ({sum(1 for s in opp_shots if s['on_target'])})",
         len(bes_shots), len(opp_shots)),
        ("Goals scored", str(sum(1 for s in bes_shots if s["is_goal"])), str(sum(1 for s in opp_shots if s["is_goal"])),
         sum(1 for s in bes_shots if s["is_goal"]) or 0.01, sum(1 for s in opp_shots if s["is_goal"]) or 0.01),
        ("Touch share", f"{bes_touch / (bes_touch + opp_touch):.0%}", f"{opp_touch / (bes_touch + opp_touch):.0%}",
         bes_touch, opp_touch),
        ("Pass accuracy",
         f"{sum(1 for p in bes_pass if p['completed']) / len(bes_pass):.0%}",
         f"{sum(1 for p in opp_pass if p['completed']) / len(opp_pass):.0%}",
         sum(1 for p in bes_pass if p["completed"]) / len(bes_pass),
         sum(1 for p in opp_pass if p["completed"]) / len(opp_pass)),
    ]
    n = len(rows)
    for i, (label, bval, oval, bnum, onum) in enumerate(rows):
        y = 1 - (i + 0.5) / n
        ax2.text(0.5, y + 0.075, label, ha="center", va="bottom", fontsize=11.5,
                  fontweight="bold", color=palette["ink_primary"])
        total = bnum + onum if (bnum + onum) > 0 else 1
        frac = bnum / total
        bar_y = y - 0.01
        h = 0.05
        ax2.add_patch(plt.Rectangle((0.0, bar_y), frac, h, color=BES_C))
        ax2.add_patch(plt.Rectangle((frac, bar_y), 1 - frac, h, color=OPP_C))
        ax2.text(0.02, bar_y + h / 2, bval, ha="left", va="center", fontsize=10.5,
                  fontweight="bold", color=palette["surface"])
        ax2.text(0.98, bar_y + h / 2, oval, ha="right", va="center", fontsize=10.5,
                  fontweight="bold", color=palette["surface"])
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)

    legend_elems = [Line2D([0], [0], marker="o", color=palette["surface"], markerfacecolor=BES_C,
                            markersize=10, label="Besiktas", linewidth=0),
                    Line2D([0], [0], marker="o", color=palette["surface"], markerfacecolor=OPP_C,
                            markersize=10, label="Midtjylland", linewidth=0)]
    fig.legend(handles=legend_elems, loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 0.04), fontsize=10.5, labelcolor=palette["ink_secondary"])

    components.header(fig, kicker="Tie Summary",
                       title="Besiktas advance 3-0 despite conceding the larger share of chances",
                       dek=f"Combined totals, leg 1 + leg 2  ·  {bes_xg:.2f} xG for vs {opp_xg:.2f} xG against",
                       palette=palette)
    components.footer(fig, source=md.SOURCE, palette=palette)
    save(fig, "02_tie_summary.png")


# ---------------------------------------------------------------------------
# 03-04. Shot map -- one page per leg
# ---------------------------------------------------------------------------

def shot_map_leg(leg, page_num, headline, dek):
    fig, palette = new_fig()
    pitch = new_pitch(palette)
    ax = fig.add_axes([0.05, 0.10, 0.90, 0.62])
    pitch.draw(ax=ax)

    for s in leg.shots:
        color = team_color(s["contestantId"])
        size = 90 + s["xg"] * 900
        if s["is_goal"]:
            pitch.scatter(s["x"], s["y"], ax=ax, s=size, marker="o", color=color,
                          edgecolors=palette["ink_primary"], linewidth=1.4, zorder=5)
            ax.annotate(f"{s['player']} {s['minute']}′", xy=(s["x"], s["y"]), xytext=(0, 12),
                        textcoords="offset points", ha="center", fontsize=9, fontweight="bold",
                        color=color)
        else:
            pitch.scatter(s["x"], s["y"], ax=ax, s=size, marker="o", facecolors="none",
                          edgecolors=color, linewidth=1.6, alpha=0.85, zorder=4)
    ax.text(0.02, -0.06, "Hollow = shot   ● Filled = goal   Size = xG   Besiktas attack right",
            transform=ax.transAxes, fontsize=8.5, color=palette["ink_muted"])

    legend_elems = [Line2D([0], [0], marker="o", color=palette["surface"], markerfacecolor=BES_C,
                            markersize=10, label="Besiktas", linewidth=0),
                    Line2D([0], [0], marker="o", color=palette["surface"], markerfacecolor=OPP_C,
                            markersize=10, label="Midtjylland", linewidth=0)]
    fig.legend(handles=legend_elems, loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 0.03), fontsize=10.5, labelcolor=palette["ink_secondary"])

    components.header(fig, kicker=f"Leg {leg.leg_num} Shot Map", title=headline, dek=dek, palette=palette)
    components.footer(fig, source=md.SOURCE, palette=palette)
    save(fig, f"{page_num}_shot_map_leg{leg.leg_num}.png")


# ---------------------------------------------------------------------------
# 05. xG flow, both legs
# ---------------------------------------------------------------------------

def xg_flow_both(leg1, leg2):
    fig, palette = new_fig()
    ax1 = fig.add_axes([0.07, 0.16, 0.40, 0.58])
    ax2 = fig.add_axes([0.56, 0.16, 0.40, 0.58])

    def series(shots):
        ordered = sorted(shots, key=lambda s: s["minute"])
        mins, cum, total = [0.0], [0.0], 0.0
        for s in ordered:
            mins.append(s["minute"]); cum.append(total)
            total += s["xg"]
            mins.append(s["minute"]); cum.append(total)
        mins.append(96); cum.append(total)
        return mins, cum

    for ax, leg in ((ax1, leg1), (ax2, leg2)):
        for i, (rows_fn, color, name) in enumerate(
                ((leg.besiktas, BES_C, BES_SHORT), (leg.opponent, OPP_C, OPP_SHORT))):
            team_shots = rows_fn(leg.shots)
            mins, cum = series(team_shots)
            ax.plot(mins, cum, color=color, linewidth=2.4, zorder=4)
            ax.fill_between(mins, cum, color=color, alpha=0.10, zorder=1)
            ax.text(0.03, 0.95 - i * 0.09, f"{name}: {cum[-1]:.2f} xG", transform=ax.transAxes,
                    color=color, fontsize=9.5, fontweight="bold", va="top", ha="left")
            running = 0.0
            for s in sorted(team_shots, key=lambda s: s["minute"]):
                if s["is_goal"]:
                    ax.scatter([s["minute"]], [running], marker="*", s=180, color=palette["ink_primary"],
                               edgecolors=color, linewidth=1.4, zorder=6)
                    ax.annotate(f"{s['player']} {s['minute']}′", xy=(s["minute"], running),
                                xytext=(0, 12), textcoords="offset points", ha="center",
                                fontsize=7.8, color=palette["ink_secondary"])
                running += s["xg"]
        ax.axvline(45, color=palette["axis"], linewidth=0.8, linestyle=":")
        ax.set_xlim(0, 100)
        ax.set_xlabel("Minute")
        ax.set_title(f"Leg {leg.leg_num}: {leg.score_line}", fontsize=11.5, fontweight="bold",
                     family="sans-serif", color=palette["ink_primary"])
    ax1.set_ylabel("Cumulative xG")

    components.header(fig, kicker="xG Flow",
                       title="Midtjylland's chances arrived in bursts; Besiktas' came from volume",
                       dek="Cumulative expected goals by minute, each leg",
                       palette=palette)
    components.footer(fig, source=md.SOURCE, palette=palette)
    save(fig, "05_xg_flow.png")


# ---------------------------------------------------------------------------
# 06. Shot log -- Besiktas' shots, both legs
# ---------------------------------------------------------------------------

def shot_log(leg1, leg2):
    fig, palette = new_fig()
    cols = ["Min", "Player", "Situation", "Body", "Outcome", "xG"]
    widths = [0.12, 0.32, 0.20, 0.14, 0.14, 0.08]
    x0 = [sum(widths[:i]) for i in range(len(widths))]

    axes = [fig.add_axes([0.04, 0.10, 0.44, 0.64]), fig.add_axes([0.53, 0.10, 0.44, 0.64])]
    for ax, leg in zip(axes, (leg1, leg2)):
        ax.axis("off")
        shots = sorted(leg.besiktas(leg.shots), key=lambda s: s["minute"])
        header_y = 1.0
        for x, label in zip(x0, cols):
            ax.text(x, header_y, label, fontsize=10, fontweight="bold", color=palette["ink_primary"],
                    va="top", ha="left")
        ax.axhline(header_y - 0.02, xmin=0, xmax=1, color=palette["axis"], linewidth=1.0)
        row_h = 0.92 / 29
        for i, s in enumerate(shots):
            y = header_y - 0.045 - i * row_h
            weight = "bold" if s["is_goal"] else "normal"
            vals = [f"{s['minute']}′", s["player"], s["situation"], "Head" if s["is_header"] else "Foot",
                    s["outcome"], f"{s['xg']:.2f}"]
            for x, v in zip(x0, vals):
                ax.text(x, y, v, fontsize=8.6, color=palette["ink_primary"], fontweight=weight, va="top", ha="left")
            if s["is_goal"]:
                ax.text(0.995, y, "★", fontsize=10, color=GOOD_C, va="top", ha="right")
        ax.set_xlim(0, 1)
        ax.set_ylim(header_y - 0.045 - 29 * row_h, 1.03)
        ax.set_title(f"Leg {leg.leg_num}  ·  {len(shots)} shots  ·  {sum(s['xg'] for s in shots):.2f} xG",
                     fontsize=11, fontweight="bold", family="sans-serif", color=palette["ink_primary"])

    components.header(fig, kicker="Shot Log",
                       title="Every Besiktas shot across both legs, ranked by kickoff time",
                       dek="Own xG model: distance + angle to goal, header penalty applied",
                       palette=palette)
    components.footer(fig, source=md.SOURCE, palette=palette)
    save(fig, "06_shot_log.png")


# ---------------------------------------------------------------------------
# 07. Goal build-ups
# ---------------------------------------------------------------------------

def goal_buildups(leg1, leg2):
    fig, palette = new_fig()
    pitch = new_pitch(palette)

    goals = []
    for leg in (leg1, leg2):
        for s in sorted(leg.besiktas(leg.shots), key=lambda s: s["minute"]):
            if s["is_goal"]:
                goals.append((leg, s))

    n = len(goals)
    axes = [fig.add_axes([0.02 + i * (0.96 / n), 0.10, 0.96 / n - 0.02, 0.60]) for i in range(n)]

    for ax, (leg, g) in zip(axes, goals):
        pitch.draw(ax=ax)
        team_events = [e for e in leg.events if e["contestantId"] == md.BESIKTAS_ID
                       and e.get("x") is not None and e["typeId"] in (1, 3, 61)
                       and md.event_time(e) <= g["minute"] * 60 + 59]
        team_events.sort(key=lambda e: (e["periodId"], md.event_time(e), e["eventId"]))
        chain = team_events[-4:]
        pts = []
        for e in chain:
            x, y = md.norm_xy(e, leg.directions)
            xm, ym = md.to_m(x, y)
            pts.append((xm, ym))
        pts.append((g["x"], g["y"]))

        for j in range(len(pts) - 1):
            x1, y1 = pts[j]
            x2, y2 = pts[j + 1]
            alpha = 0.45 + 0.55 * (j / (len(pts) - 1))
            pitch.arrows(x1, y1, x2, y2, ax=ax, color=BES_C, alpha=alpha, width=2.2,
                        headwidth=6, headlength=6, zorder=3)
        pitch.scatter(g["x"], g["y"], ax=ax, s=260, marker="*", color=palette["ink_primary"],
                      edgecolors=BES_C, linewidth=1.6, zorder=6)
        ax.set_title(f"Leg {leg.leg_num}, {g['minute']}′  {g['player']}\n{g['xg']:.2f} xG",
                     color=BES_C, fontsize=10.5, fontweight="bold", family="sans-serif")

    fig.text(0.5, 0.085, "Last 4 touches before each goal (passes, take-ons and ball touches by Besiktas)",
              ha="center", fontsize=9, color=palette["ink_muted"])

    components.header(fig, kicker="Goal Build-Ups",
                       title="Kokcu's brace either side of Rashica settled a tie built on fine margins",
                       dek="3-0 on aggregate  ·  possession chain leading to each Besiktas goal",
                       palette=palette)
    components.footer(fig, source=md.SOURCE, palette=palette)
    save(fig, "07_goal_buildups.png")


# ---------------------------------------------------------------------------
# 08. Pass network -- Besiktas, both legs side by side
# ---------------------------------------------------------------------------

def _average_positions(team_passes, min_passes=8):
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


def _draw_pass_network(ax, team_passes, color, palette, pitch, min_passes=8):
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
    for pl, (x, y, cnt) in avg_pos.items():
        size = 260 + 900 * (cnt / max_n)
        pitch.scatter(x, y, ax=ax, s=size, color=palette["surface"], edgecolors=color,
                      linewidth=2.0, zorder=4)
        last = pl.split(" ")[-1]
        pitch.annotate(last, (x, y), ax=ax, ha="center", va="center", fontsize=7.4,
                       color=palette["ink_primary"], fontweight="bold", zorder=5)


def pass_network_both_legs(leg1, leg2):
    fig, palette = new_fig()
    pitch = new_pitch(palette)
    ax1 = fig.add_axes([0.02, 0.10, 0.47, 0.62])
    ax2 = fig.add_axes([0.51, 0.10, 0.47, 0.62])

    _draw_pass_network(ax1, leg1.besiktas(leg1.passes), BES_C, palette, pitch)
    _draw_pass_network(ax2, leg2.besiktas(leg2.passes), BES_C, palette, pitch)
    ax1.set_title(f"Leg 1 ({leg1.score_line})", color=BES_C, fontsize=12.5, fontweight="bold", family="sans-serif")
    ax2.set_title(f"Leg 2 ({leg2.score_line})", color=BES_C, fontsize=12.5, fontweight="bold", family="sans-serif")

    fig.text(0.5, 0.09, "Node position = average completed-pass location (≥ 8 passes)  ·  "
                         "Node size = passes played  ·  Line width = pass combinations (≥ 2)",
              ha="center", fontsize=9, color=palette["ink_muted"])

    components.header(fig, kicker="Pass Network",
                       title="Besiktas' shape held from Istanbul to Herning",
                       dek="Besiktas' average completed-pass position, both legs attacking right",
                       palette=palette)
    components.footer(fig, source=md.SOURCE, palette=palette)
    save(fig, "08_pass_network.png")


# ---------------------------------------------------------------------------
# 09. Touch heatmap -- Besiktas, both legs
# ---------------------------------------------------------------------------

def touch_heatmap_both_legs(leg1, leg2):
    fig, palette = new_fig()
    pitch = new_pitch(palette)
    ax1 = fig.add_axes([0.02, 0.10, 0.47, 0.62])
    ax2 = fig.add_axes([0.51, 0.10, 0.47, 0.62])

    cmap = plt.cm.colors.LinearSegmentedColormap.from_list(
        "bes", [palette["surface"], BES_C])

    for ax, leg in ((ax1, leg1), (ax2, leg2)):
        pitch.draw(ax=ax)
        touches = leg.besiktas(leg.touches)
        xs = [t["x"] for t in touches]
        ys = [t["y"] for t in touches]
        pitch.kdeplot(xs, ys, ax=ax, cmap=cmap, fill=True, levels=60, alpha=0.85, zorder=1)
        ax.set_title(f"Leg {leg.leg_num}  ·  {leg.touch_share():.0%} touch share",
                     color=BES_C, fontsize=12.5, fontweight="bold", family="sans-serif")

    components.header(fig, kicker="Territory",
                       title="Besiktas camped further forward at home than they did away",
                       dek="Besiktas touch density, both legs attacking right",
                       palette=palette)
    components.footer(fig, source=md.SOURCE, palette=palette)
    save(fig, "09_touch_heatmap.png")


# ---------------------------------------------------------------------------
# 10. PPDA across both legs
# ---------------------------------------------------------------------------

def ppda_both_legs(leg1, leg2):
    fig, palette = new_fig()
    ax1 = fig.add_axes([0.08, 0.16, 0.40, 0.58])
    ax2 = fig.add_axes([0.56, 0.16, 0.40, 0.58])

    buckets = [(0, 15), (15, 30), (30, 45), (45, 60), (60, 75), (75, 96)]
    labels = ["0-15", "15-30", "30-45", "45-60", "60-75", "75-90+"]

    def bucketed(leg, contestant_id, opp_id):
        return [md.compute_ppda(leg.passes, leg.pressing, contestant_id, opp_id, lo, hi) for lo, hi in buckets]

    for ax, leg in ((ax1, leg1), (ax2, leg2)):
        bes_vals = bucketed(leg, md.BESIKTAS_ID, md.MIDTJYLLAND_ID)
        opp_vals = bucketed(leg, md.MIDTJYLLAND_ID, md.BESIKTAS_ID)
        finite = [v for v in bes_vals + opp_vals if not math.isnan(v)]
        shared_max = max(finite) * 1.15 if finite else 1.0
        xs = np.arange(len(labels))
        width = 0.36
        bes_clean = [v if not math.isnan(v) else 0 for v in bes_vals]
        opp_clean = [v if not math.isnan(v) else 0 for v in opp_vals]
        ax.bar(xs - width / 2, bes_clean, width=width, color=BES_C, label="Besiktas")
        ax.bar(xs + width / 2, opp_clean, width=width, color=OPP_C, label="Midtjylland")
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, fontsize=8.2)
        ax.set_ylim(0, shared_max)
        ax.set_ylabel("PPDA")
        ax.set_title(f"Leg {leg.leg_num}  ·  Besiktas {leg.besiktas_ppda():.1f}  vs  "
                     f"Midtjylland {leg.opponent_ppda():.1f}", fontsize=10.5, fontweight="bold",
                     family="sans-serif", color=palette["ink_primary"])

    fig.legend(*ax1.get_legend_handles_labels(), loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 0.03), fontsize=10.5, labelcolor=palette["ink_secondary"])

    components.header(fig, kicker="Pressing",
                       title="Besiktas pressed hardest early in Istanbul, then let the game come to them",
                       dek="Passes per defensive action in the opponent's own 60%  ·  lower = more intense press",
                       palette=palette)
    components.footer(fig, source=md.SOURCE, palette=palette)
    save(fig, "10_ppda.png")


# ---------------------------------------------------------------------------
# 11. Progression -- box entries, both legs
# ---------------------------------------------------------------------------

def box_entries_both_legs(leg1, leg2):
    fig, palette = new_fig()
    pitch = new_pitch(palette)
    ax1 = fig.add_axes([0.02, 0.10, 0.47, 0.62])
    ax2 = fig.add_axes([0.51, 0.10, 0.47, 0.62])

    for ax, leg in ((ax1, leg1), (ax2, leg2)):
        pitch.draw(ax=ax)
        entries = [p for p in leg.besiktas(leg.passes) if p["box_entry"]]
        prog = sum(1 for p in leg.besiktas(leg.passes) if p["progressive"])
        for p in entries:
            pitch.arrows(p["x"], p["y"], p["end_x"], p["end_y"], ax=ax, color=BES_C,
                        alpha=0.85, width=2.0, headwidth=6, headlength=6, zorder=3)
        ax.set_title(f"Leg {leg.leg_num}  ·  {len(entries)} box entries  ·  {prog} progressive passes",
                     color=BES_C, fontsize=11.5, fontweight="bold", family="sans-serif")

    components.header(fig, kicker="Progression",
                       title="Besiktas found the box far more often at home than away",
                       dek="Besiktas' completed passes ending inside the penalty area, both legs attacking right",
                       palette=palette)
    components.footer(fig, source=md.SOURCE, palette=palette)
    save(fig, "11_box_entries.png")


# ---------------------------------------------------------------------------
# 12. Defensive actions -- Besiktas, both legs
# ---------------------------------------------------------------------------

def defensive_actions_both_legs(leg1, leg2):
    fig, palette = new_fig()
    pitch = new_pitch(palette)
    ax1 = fig.add_axes([0.02, 0.10, 0.47, 0.62])
    ax2 = fig.add_axes([0.51, 0.10, 0.47, 0.62])
    markers = {"Tackle": "o", "Interception": "^", "Clearance": "s"}

    for ax, leg in ((ax1, leg1), (ax2, leg2)):
        pitch.draw(ax=ax)
        defs = leg.besiktas(leg.defs)
        for action, marker in markers.items():
            pts = [d for d in defs if d["action"] == action]
            if not pts:
                continue
            pitch.scatter([d["x"] for d in pts], [d["y"] for d in pts], ax=ax, marker=marker,
                          s=70, color=BES_C, edgecolors=palette["surface"], linewidth=0.6, alpha=0.85, zorder=3)
        ax.set_title(f"Leg {leg.leg_num}  ·  {len(defs)} defensive actions",
                     color=BES_C, fontsize=12, fontweight="bold", family="sans-serif")

    legend_elems = [Line2D([0], [0], marker=m, color=palette["surface"], markerfacecolor=BES_C,
                            markersize=10, label=k, linewidth=0) for k, m in markers.items()]
    fig.legend(handles=legend_elems, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 0.03), fontsize=10.5, labelcolor=palette["ink_secondary"])

    components.header(fig, kicker="Defending",
                       title="Besiktas defended deeper and busier in Herning than in Istanbul",
                       dek="Besiktas tackles, interceptions and clearances, both legs attacking right",
                       palette=palette)
    components.footer(fig, source=md.SOURCE, palette=palette)
    save(fig, "12_defensive_actions.png")


# ---------------------------------------------------------------------------
# 13. Duels
# ---------------------------------------------------------------------------

def duels_both_legs(leg1, leg2):
    fig, palette = new_fig()
    ax = fig.add_axes([0.22, 0.16, 0.68, 0.56])

    kinds = ["Tackle", "Aerial", "Challenge"]
    rows = []
    for kind in kinds:
        for leg in (leg1, leg2):
            bes = [d for d in leg.besiktas(leg.duels) if d["action"] == kind]
            bes_won = sum(1 for d in bes if d["success"])
            rows.append((f"{kind} · Leg {leg.leg_num}", len(bes), bes_won))

    ypos = np.arange(len(rows))[::-1]
    maxval = max(n for _, n, _ in rows) * 1.25 or 1
    for y, (label, n, won) in zip(ypos, rows):
        rate = won / n if n else 0
        ax.barh(y, n, height=0.55, color=palette["axis"])
        ax.barh(y, won, height=0.55, color=BES_C)
        ax.text(n + maxval * 0.015, y, f"{won}/{n} ({rate:.0%})", va="center", fontsize=9.5,
                color=palette["ink_primary"])
    ax.set_yticks(ypos)
    ax.set_yticklabels([r[0] for r in rows], fontsize=10.5, color=palette["ink_primary"])
    ax.set_xlim(0, maxval)
    ax.set_xlabel("Besiktas duels contested (solid = won)")
    ax.grid(axis="x")
    ax.set_axisbelow(True)

    components.header(fig, kicker="Duels",
                       title="Besiktas' duel win rate climbed leg to leg, aerials still a weak point",
                       dek="Tackle, aerial and loose-ball duel outcomes, leg by leg",
                       palette=palette)
    components.footer(fig, source=md.SOURCE, palette=palette)
    save(fig, "13_duels.png")


# ---------------------------------------------------------------------------
# 14. Discipline
# ---------------------------------------------------------------------------

def discipline_both_legs(leg1, leg2):
    fig, palette = new_fig()
    ax = fig.add_axes([0.16, 0.42, 0.68, 0.28])

    all_cards = [(leg, c) for leg in (leg1, leg2) for c in leg.cards]
    ax.axhline(1, color=BES_C, linewidth=6, alpha=0.20, solid_capstyle="round")
    ax.axhline(0, color=OPP_C, linewidth=6, alpha=0.20, solid_capstyle="round")

    for row in (0, 1):
        row_cards = sorted([(leg, c) for leg, c in all_cards
                             if (c["contestantId"] == md.BESIKTAS_ID) == (row == 1)],
                            key=lambda lc: lc[1]["minute"])
        rung = 0
        prev_minute = None
        for leg, c in row_cards:
            if prev_minute is not None and c["minute"] - prev_minute < 6:
                rung += 1
            else:
                rung = 0
            prev_minute = c["minute"]
            color = CRIT_C if c["kind"] != "Yellow" else WARN_C
            marker = "s" if c["kind"] != "Yellow" else "o"
            ax.scatter([c["minute"]], [row], marker=marker, s=140, color=color,
                       edgecolors=palette["surface"], linewidth=1.0, zorder=4)
            base = 14 + rung * 13
            ax.annotate(f"L{leg.leg_num} {c['minute']}′ {c['player']}", xy=(c["minute"], row),
                        xytext=(0, base if row == 1 else -base - 4), textcoords="offset points", ha="center",
                        fontsize=7.6, color=palette["ink_secondary"], rotation=0)

    ax.set_yticks([0, 1])
    ax.set_yticklabels([OPP_SHORT, BES_SHORT], fontsize=11, fontweight="bold")
    ax.set_ylim(-0.6, 1.6)
    ax.set_xlim(0, 96)
    ax.set_xlabel("Minute (both legs overlaid on a single 0-90 axis)")
    ax.axvline(45, color=palette["axis"], linewidth=0.8, linestyle=":")

    legend_elems = [Line2D([0], [0], marker="o", color=palette["surface"], markerfacecolor=WARN_C,
                            markersize=10, label="Yellow", linewidth=0),
                    Line2D([0], [0], marker="s", color=palette["surface"], markerfacecolor=CRIT_C,
                            markersize=10, label="Red / 2nd yellow", linewidth=0)]
    fig.legend(handles=legend_elems, loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 0.12), fontsize=10.5, labelcolor=palette["ink_secondary"])

    yellow_bes = sum(1 for _, c in all_cards if c["contestantId"] == md.BESIKTAS_ID and c["kind"] == "Yellow")
    yellow_opp = sum(1 for _, c in all_cards if c["contestantId"] == md.MIDTJYLLAND_ID and c["kind"] == "Yellow")
    red_opp = sum(1 for _, c in all_cards if c["contestantId"] == md.MIDTJYLLAND_ID and c["kind"] != "Yellow")

    fig.text(0.5, 0.20, f"Midtjylland played the closing stages of BOTH legs a man down "
                        f"({red_opp} red cards)  ·  Besiktas {yellow_bes} yellows, Midtjylland {yellow_opp} yellows",
              ha="center", fontsize=9.5, color=palette["ink_muted"])

    components.header(fig, kicker="Discipline",
                       title="Midtjylland finished both legs with ten men",
                       dek="Etim sent off 14′ in leg 1, Erlic sent off 52′ in leg 2",
                       palette=palette)
    components.footer(fig, source=md.SOURCE, palette=palette)
    save(fig, "14_discipline.png")


# ---------------------------------------------------------------------------
# 15. Impact leaderboard
# ---------------------------------------------------------------------------

def impact_leaderboard(leg1, leg2):
    fig, palette = new_fig()
    ax = fig.add_axes([0.22, 0.20, 0.68, 0.54])

    scores = {}
    shot_n, goal_n = {}, {}
    for leg in (leg1, leg2):
        for s in leg.besiktas(leg.shots):
            scores[s["player"]] = scores.get(s["player"], 0) + s["xg"] + (3.0 if s["is_goal"] else 0)
            shot_n[s["player"]] = shot_n.get(s["player"], 0) + 1
            if s["is_goal"]:
                goal_n[s["player"]] = goal_n.get(s["player"], 0) + 1
        for p in leg.besiktas(leg.passes):
            scores[p["player"]] = scores.get(p["player"], 0) + 0.15 * p["progressive"] + 0.35 * p["box_entry"]
        for d in leg.besiktas(leg.defs):
            scores[d["player"]] = scores.get(d["player"], 0) + 0.3

    top = sorted(scores.items(), key=lambda kv: -kv[1])[:8][::-1]
    ypos = np.arange(len(top))
    vals = [v for _, v in top]
    ax.barh(ypos, vals, color=BES_C)
    ax.set_yticks(ypos)
    labels = [f"{p}  ({goal_n.get(p, 0)}g)" if goal_n.get(p) else p for p, _ in top]
    ax.set_yticklabels(labels, fontsize=10.5, color=palette["ink_primary"])
    for y, v in zip(ypos, vals):
        ax.text(v + max(vals, default=1) * 0.02, y, f"{v:.1f}", va="center", fontsize=9.5,
                color=palette["ink_secondary"])
    ax.set_xlabel("Impact score, both legs combined")

    fig.text(0.5, 0.07, "Simple composite: xG + 3×goals + 0.15×progressive pass + 0.35×box entry + "
                         "0.3×defensive action  ·  not an official rating",
              ha="center", fontsize=8.8, color=palette["ink_muted"])

    components.header(fig, kicker="Impact",
                       title="Kokcu's brace put him clear at the top across both legs",
                       dek="Besiktas' shooting, progression and defending combined into one rough index",
                       palette=palette)
    components.footer(fig, source=md.SOURCE, palette=palette)
    save(fig, "15_impact_leaderboard.png")


# ---------------------------------------------------------------------------
# 16. Goalkeeping / chances denied
# ---------------------------------------------------------------------------

def chances_denied(leg1, leg2):
    fig, palette = new_fig()
    pitch = new_pitch(palette)
    ax = fig.add_axes([0.05, 0.10, 0.62, 0.62])
    pitch.draw(ax=ax)

    opp_shots = leg1.opponent(leg1.shots) + leg2.opponent(leg2.shots)
    outcome_marker = {"Saved": "o", "Post": "D", "Miss": "x", "Goal": "*"}
    outcome_color = {"Saved": WARN_C, "Post": OPP_C, "Miss": palette["axis"], "Goal": CRIT_C}
    for s in opp_shots:
        m = outcome_marker[s["outcome"]]
        c = outcome_color[s["outcome"]]
        size = 90 + s["xg"] * 900
        pitch.scatter(s["x"], s["y"], ax=ax, s=size, marker=m, color=c,
                      edgecolors=palette["ink_primary"], linewidth=1.0, alpha=0.9, zorder=4)
    ax.text(0.02, -0.06, "● Saved   ◆ Post   ✕ Miss   Size = xG   Midtjylland attack right",
            transform=ax.transAxes, fontsize=8.5, color=palette["ink_muted"])

    saved = [s for s in opp_shots if s["outcome"] == "Saved"]
    posts = [s for s in opp_shots if s["outcome"] == "Post"]
    xg_denied = sum(s["xg"] for s in saved) + sum(s["xg"] for s in posts)
    total_xg = sum(s["xg"] for s in opp_shots)

    ax2 = fig.add_axes([0.70, 0.16, 0.26, 0.54])
    ax2.axis("off")
    metrics = [
        ("Midtjylland shots faced", str(len(opp_shots))),
        ("On target", str(sum(1 for s in opp_shots if s["on_target"]))),
        ("Saved by Besiktas' keeper", str(len(saved))),
        ("Off the woodwork", str(len(posts))),
        ("Goals conceded", "0"),
        ("xG denied (saved + post)", f"{xg_denied:.2f}"),
        ("Midtjylland's total xG", f"{total_xg:.2f}"),
    ]
    for i, (label, val) in enumerate(metrics):
        y = 1 - i / (len(metrics) - 1) * 0.92
        ax2.text(0.0, y, label, fontsize=9.8, color=palette["ink_secondary"], va="center")
        ax2.text(1.0, y, val, fontsize=11.5, fontweight="bold", color=BES_C, va="center", ha="right")
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)

    components.header(fig, kicker="Chances Denied",
                       title=f"A clean sheet on aggregate, built from {xg_denied:.2f} xG saved or off the frame",
                       dek="Every shot Midtjylland took against Besiktas, both legs combined",
                       palette=palette)
    components.footer(fig, source=md.SOURCE, palette=palette)
    save(fig, "16_chances_denied.png")


# ---------------------------------------------------------------------------
# 17. Momentum timeline, both legs
# ---------------------------------------------------------------------------

def momentum_timeline_both_legs(leg1, leg2):
    fig, palette = new_fig()
    axes = [fig.add_axes([0.16, 0.56, 0.78, 0.20]), fig.add_axes([0.16, 0.26, 0.78, 0.20])]

    for ax, leg in zip(axes, (leg1, leg2)):
        ax.axhline(1, color=BES_C, linewidth=6, alpha=0.22, solid_capstyle="round")
        ax.axhline(0, color=OPP_C, linewidth=6, alpha=0.22, solid_capstyle="round")

        def row(cid):
            return 1 if cid == md.BESIKTAS_ID else 0

        goals = sorted([s for s in leg.shots if s["is_goal"]], key=lambda s: s["minute"])
        prev_minute = None
        rung = 0
        for s in goals:
            r = row(s["contestantId"])
            ax.scatter([s["minute"]], [r], marker="*", s=280,
                       color=palette["ink_primary"], edgecolors=team_color(s["contestantId"]),
                       linewidth=1.6, zorder=5)
            if prev_minute is not None and s["minute"] - prev_minute < 8:
                rung += 1
            else:
                rung = 0
            prev_minute = s["minute"]
            base = 14 + rung * 15
            ax.annotate(f"{s['player']} {s['minute']}′", xy=(s["minute"], r),
                        xytext=(0, base if r == 1 else -base - 4),
                        textcoords="offset points", ha="center", fontsize=8, color=palette["ink_secondary"])
        for c in leg.cards:
            marker_color = CRIT_C if c["kind"] != "Yellow" else WARN_C
            ax.scatter([c["minute"]], [row(c["contestantId"])], marker="s", s=100, color=marker_color,
                       edgecolors=palette["surface"], linewidth=1.0, zorder=4)
        for sub in leg.subs:
            ax.scatter([sub["minute"]], [row(sub["contestantId"])], marker="^", s=55,
                       color=palette["ink_muted"], alpha=0.85, zorder=3)

        ax.set_xlim(0, 96)
        ax.set_ylim(-0.6, 1.6)
        ax.set_yticks([0, 1])
        ax.set_yticklabels([OPP_SHORT, BES_SHORT], fontsize=9.5)
        ax.axvline(45, color=palette["axis"], linewidth=0.8, linestyle=":")
        ax.set_title(f"Leg {leg.leg_num}: {leg.score_line}", fontsize=10.5, fontweight="bold",
                     family="sans-serif", color=palette["ink_primary"])
    axes[1].set_xlabel("Minute")

    legend_elems = [Line2D([0], [0], marker="*", color=palette["surface"], markerfacecolor=palette["ink_primary"],
                            markeredgecolor=palette["ink_primary"], markersize=14, label="Goal", linewidth=0),
                    Line2D([0], [0], marker="s", color=palette["surface"], markerfacecolor=WARN_C,
                            markersize=10, label="Yellow card", linewidth=0),
                    Line2D([0], [0], marker="s", color=palette["surface"], markerfacecolor=CRIT_C,
                            markersize=10, label="Red / 2nd yellow", linewidth=0),
                    Line2D([0], [0], marker="^", color=palette["surface"], markerfacecolor=palette["ink_muted"],
                            markersize=10, label="Substitution", linewidth=0)]
    fig.legend(handles=legend_elems, loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, 0.12), fontsize=9.8, labelcolor=palette["ink_secondary"])

    components.header(fig, kicker="Match Narrative",
                       title="Two red cards, five bookings, three Besiktas goals",
                       dek="Goals, cards and changes across both legs' 90 minutes",
                       palette=palette)
    components.footer(fig, source=md.SOURCE, palette=palette)
    save(fig, "17_momentum_timeline.png")


# ---------------------------------------------------------------------------
# 18. Closing summary
# ---------------------------------------------------------------------------

def closing_summary(leg1, leg2):
    palette, _ = style.apply("dark")
    fig = plt.figure(figsize=FIGSIZE)
    fig.patch.set_facecolor(palette["surface"])

    fig.text(0.06, 0.80, f"{components.MARK} ROAD AHEAD", fontsize=11, fontweight="bold",
              color=palette["accent"], family="sans-serif", ha="left", va="top")
    fig.text(0.06, 0.735, "Besiktas advance to the 3rd Qualifying Round", fontsize=19,
              color=palette["ink_primary"], family="serif", ha="left", va="top")
    fig.text(0.06, 0.685, "UEFA Europa League 2026/27, having won this tie 3-0 on aggregate", fontsize=11,
              color=palette["ink_secondary"], family="sans-serif", ha="left", va="top")

    bullets = [
        "Won both legs (1-0 home, 2-0 away) without conceding a goal in either -- Midtjylland's 2.75 combined "
        "xG (vs Besiktas' 2.03) never turned into a goal.",
        "Both legs were shaped by a Midtjylland red card: Etim in the 14th minute of leg 1, Erlic's second "
        "yellow in the 52nd minute of leg 2 -- Midtjylland played over two-thirds of the tie a man light.",
        "Orkun Kokcu scored both of Besiktas' goals from open play at combined 0.30 xG (25′ leg 1, 75′ leg 2), "
        "either side of Milot Rashica's 69th-minute strike in leg 2 -- three low-probability finishes decided the tie.",
        f"Besiktas' pressing intensity (PPDA) eased from leg 1 to leg 2 as territory swung the other way -- "
        f"{leg1.touch_share():.0%} Besiktas touch share at home, roughly even ({leg2.touch_share():.0%}) away.",
        "Besiktas' keeper faced 12 shots on target across both legs and conceded none, backed by two Midtjylland "
        "efforts off the woodwork in leg 2.",
    ]
    y = 0.60
    for b in bullets:
        fig.text(0.06, y, "•", fontsize=13, color=palette["accent"], family="sans-serif", va="top")
        fig.text(0.085, y, b, fontsize=10.3, color=palette["ink_primary"], family="sans-serif",
                  va="top", wrap=True, linespacing=1.5)
        y -= 0.115

    components.brand_mark(fig, palette=palette, right=0.94, y=0.93)
    components.footer(fig, source=md.SOURCE, palette=palette)
    save(fig, "18_closing_summary.png")


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def main():
    leg1 = md.Leg(md.LEG1)
    leg2 = md.Leg(md.LEG2)

    cover(leg1, leg2)
    tie_summary(leg1, leg2)
    shot_map_leg(leg1, "03",
                 "Besiktas peppered a ten-man Midtjylland for 90 minutes, but needed only one moment",
                 f"Leg 1: {leg1.score_line}  ·  Midtjylland's Etim sent off, 14′")
    shot_map_leg(leg2, "04",
                 "Midtjylland created the better chances away from home, and still lost 0-2",
                 f"Leg 2: {leg2.score_line}  ·  Midtjylland's Erlic sent off, 52′")
    xg_flow_both(leg1, leg2)
    shot_log(leg1, leg2)
    goal_buildups(leg1, leg2)
    pass_network_both_legs(leg1, leg2)
    touch_heatmap_both_legs(leg1, leg2)
    ppda_both_legs(leg1, leg2)
    box_entries_both_legs(leg1, leg2)
    defensive_actions_both_legs(leg1, leg2)
    duels_both_legs(leg1, leg2)
    discipline_both_legs(leg1, leg2)
    impact_leaderboard(leg1, leg2)
    chances_denied(leg1, leg2)
    momentum_timeline_both_legs(leg1, leg2)
    closing_summary(leg1, leg2)

    print("Done.")


if __name__ == "__main__":
    main()

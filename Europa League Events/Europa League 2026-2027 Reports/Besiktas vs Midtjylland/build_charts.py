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
    save(fig, "54_closing_summary.png")


# ---------------------------------------------------------------------------
# 18-19. Progressive passes -- one page per leg
# ---------------------------------------------------------------------------

def progressive_passes_leg(leg, page_num):
    fig, palette = new_fig()
    pitch = new_pitch(palette)
    prog = [p for p in leg.besiktas(leg.passes) if p["progressive"]]

    ax = fig.add_axes([0.02, 0.10, 0.96, 0.62])
    pitch.draw(ax=ax)
    for p in prog:
        is_box = p["box_entry"]
        pitch.arrows(p["x"], p["y"], p["end_x"], p["end_y"], ax=ax,
                    color=GOOD_C if is_box else BES_C, alpha=0.9 if is_box else 0.55,
                    width=2.4 if is_box else 1.4, headwidth=6, headlength=6,
                    zorder=4 if is_box else 3)

    legend_elems = [Line2D([0], [0], color=BES_C, lw=2.0, label="Progressive pass"),
                    Line2D([0], [0], color=GOOD_C, lw=2.4, label="...into the box")]
    fig.legend(handles=legend_elems, loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 0.06), fontsize=10.5, labelcolor=palette["ink_secondary"])

    n_box = sum(1 for p in prog if p["box_entry"])
    components.header(fig, kicker="Progression",
                       title=f"Leg {leg.leg_num}: {len(prog)} progressive passes, {n_box} of them straight into the box",
                       dek=f"{leg.score_line}  ·  progressive pass = completed pass cutting ≥25% off the "
                           "distance to goal, Besiktas attacking right",
                       palette=palette)
    components.footer(fig, source=md.SOURCE, palette=palette)
    save(fig, f"{page_num}_progressive_passes_leg{leg.leg_num}.png")


# ---------------------------------------------------------------------------
# 20-21. Passing volume & directness scatter -- one page per leg
# ---------------------------------------------------------------------------

def passing_directness_leg(leg, page_num):
    fig, palette = new_fig()
    ax = fig.add_axes([0.10, 0.16, 0.82, 0.58])

    by_player = {}
    for p in leg.besiktas(leg.passes):
        if not p["completed"] or p["end_x"] is None:
            continue
        d = by_player.setdefault(p["player"], {"gain": 0.0, "att": 0, "comp": 0})
        d["gain"] += p["end_x"] - p["x"]
    for p in leg.besiktas(leg.passes):
        d = by_player.setdefault(p["player"], {"gain": 0.0, "att": 0, "comp": 0})
        d["att"] += 1
        d["comp"] += int(p["completed"])

    items = [(pl, d) for pl, d in by_player.items() if d["att"] >= 5]
    xs = [d["gain"] for _, d in items]
    ys = [d["comp"] / d["att"] for _, d in items]
    sizes = [40 + d["att"] * 6 for _, d in items]
    ax.scatter(xs, ys, s=sizes, color=BES_C, alpha=0.85, edgecolors=palette["surface"], linewidth=0.8, zorder=3)
    for (pl, d), x, y in zip(items, xs, ys):
        ax.annotate(pl.split(" ")[-1], xy=(x, y), xytext=(6, 4), textcoords="offset points",
                    fontsize=8.5, color=palette["ink_secondary"])
    ax.axhline(np.mean(ys) if ys else 0, color=palette["axis"], linewidth=0.8, linestyle="--")
    ax.set_xlabel("Net metres gained by completed passes (forward - backward)")
    ax.set_ylabel("Pass completion %")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))

    components.header(fig, kicker="Passing Profile",
                       title=f"Leg {leg.leg_num}: who progressed the ball for Besiktas, and how safely",
                       dek=f"{leg.score_line}  ·  players with ≥ 5 pass attempts  ·  bubble size = passes attempted",
                       palette=palette)
    components.footer(fig, source=md.SOURCE, palette=palette)
    save(fig, f"{page_num}_passing_directness_leg{leg.leg_num}.png")


# ---------------------------------------------------------------------------
# 22. Possession by thirds, leg by leg
# ---------------------------------------------------------------------------

def possession_thirds_both_legs(leg1, leg2):
    fig, palette = new_fig()
    ax = fig.add_axes([0.16, 0.24, 0.68, 0.40])

    zone_colors = [CATEGORICAL_DARK[0], CATEGORICAL_DARK[3], CATEGORICAL_DARK[1]]
    zone_labels = ["Defensive", "Middle", "Attacking"]

    def thirds(leg):
        t = leg.besiktas(leg.touches)
        d = sum(1 for x in t if x["x"] < 35)
        m = sum(1 for x in t if 35 <= x["x"] < 70)
        a = sum(1 for x in t if x["x"] >= 70)
        total = d + m + a
        return [d / total, m / total, a / total], total

    for i, leg in enumerate((leg1, leg2)):
        fracs, total = thirds(leg)
        y = 1 - i
        left = 0
        for frac, zc, zl in zip(fracs, zone_colors, zone_labels):
            ax.barh(y, frac, left=left, height=0.6, color=zc)
            if frac > 0.06:
                ax.text(left + frac / 2, y, f"{frac:.0%}", ha="center", va="center",
                        fontsize=10.5, fontweight="bold", color=palette["surface"])
            left += frac
        ax.text(-0.02, y, f"Leg {leg.leg_num}\n({total} touches)", ha="right", va="center", fontsize=10.5,
                fontweight="bold", color=BES_C)

    ax.set_xlim(0, 1)
    ax.set_ylim(-0.7, 1.7)
    ax.set_yticks([])
    ax.set_xlabel("Share of Besiktas' touches")
    ax.set_axisbelow(True)

    legend_elems = [Line2D([0], [0], marker="s", color=palette["surface"], markerfacecolor=zc,
                            markersize=12, label=zl, linewidth=0) for zc, zl in zip(zone_colors, zone_labels)]
    fig.legend(handles=legend_elems, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 0.06), fontsize=10.5, labelcolor=palette["ink_secondary"])

    components.header(fig, kicker="Possession",
                       title="Besiktas' touches skewed far more attacking at home than away",
                       dek="Distribution of Besiktas' own touches across pitch thirds, each leg",
                       palette=palette)
    components.footer(fig, source=md.SOURCE, palette=palette)
    save(fig, "22_possession_thirds.png")


# ---------------------------------------------------------------------------
# 23. Progression comparison, leg by leg
# ---------------------------------------------------------------------------

def progression_bars_both_legs(leg1, leg2):
    fig, palette = new_fig()
    ax = fig.add_axes([0.24, 0.16, 0.66, 0.56])

    p1 = leg1.besiktas(leg1.passes)
    p2 = leg2.besiktas(leg2.passes)
    metrics = [
        ("Progressive passes", sum(1 for p in p1 if p["progressive"]), sum(1 for p in p2 if p["progressive"])),
        ("Final-third entries", sum(1 for p in p1 if p["final_third_entry"]),
         sum(1 for p in p2 if p["final_third_entry"])),
        ("Passes into the box", sum(1 for p in p1 if p["box_entry"]), sum(1 for p in p2 if p["box_entry"])),
        ("Completed crosses", sum(1 for p in p1 if p["is_cross"] and p["completed"]),
         sum(1 for p in p2 if p["is_cross"] and p["completed"])),
    ]
    n = len(metrics)
    ypos = np.arange(n)[::-1]
    maxval = max(max(h, a) for _, h, a in metrics) * 1.15
    for y, (label, h, a) in zip(ypos, metrics):
        ax.barh(y + 0.18, h, height=0.32, color=BES_C)
        ax.barh(y - 0.18, a, height=0.32, color=BES_C, alpha=0.45)
        ax.text(h + maxval * 0.015, y + 0.18, str(h), va="center", fontsize=10, color=palette["ink_primary"])
        ax.text(a + maxval * 0.015, y - 0.18, str(a), va="center", fontsize=10, color=palette["ink_primary"])
    ax.set_yticks(ypos)
    ax.set_yticklabels([m[0] for m in metrics], fontsize=11.5, color=palette["ink_primary"])
    ax.set_xlim(0, maxval)
    ax.set_xlabel("Count")
    ax.grid(axis="x")
    ax.set_axisbelow(True)

    legend_elems = [Line2D([0], [0], marker="s", color=palette["surface"], markerfacecolor=BES_C,
                            markersize=12, label="Leg 1 (home)", linewidth=0),
                    Line2D([0], [0], marker="s", color=palette["surface"], markerfacecolor=BES_C, alpha=0.45,
                            markersize=12, label="Leg 2 (away)", linewidth=0)]
    fig.legend(handles=legend_elems, loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 0.02), fontsize=10.5, labelcolor=palette["ink_secondary"])

    components.header(fig, kicker="Progression",
                       title="Besiktas progressed the ball at a similar rate in both legs, but found the box far less away",
                       dek="Progressive pass = completed pass cutting ≥25% off the distance to goal",
                       palette=palette)
    components.footer(fig, source=md.SOURCE, palette=palette)
    save(fig, "23_progression.png")


# ---------------------------------------------------------------------------
# 24. Ball recoveries by third, leg by leg
# ---------------------------------------------------------------------------

def recoveries_by_third_both_legs(leg1, leg2):
    fig, palette = new_fig()
    ax = fig.add_axes([0.24, 0.16, 0.66, 0.56])

    def by_third(leg):
        r = leg.besiktas(leg.recoveries)
        return [sum(1 for x in r if x["x"] < 35), sum(1 for x in r if 35 <= x["x"] < 70),
                sum(1 for x in r if x["x"] >= 70)]

    leg1_counts = by_third(leg1)
    leg2_counts = by_third(leg2)
    zone_labels = ["Defensive third", "Middle third", "Attacking third"]
    n = len(zone_labels)
    ypos = np.arange(n)[::-1]
    maxval = max(leg1_counts + leg2_counts) * 1.2
    for y, label, h, a in zip(ypos, zone_labels, leg1_counts, leg2_counts):
        ax.barh(y + 0.18, h, height=0.32, color=BES_C)
        ax.barh(y - 0.18, a, height=0.32, color=BES_C, alpha=0.45)
        ax.text(h + maxval * 0.015, y + 0.18, str(h), va="center", fontsize=10, color=palette["ink_primary"])
        ax.text(a + maxval * 0.015, y - 0.18, str(a), va="center", fontsize=10, color=palette["ink_primary"])
    ax.set_yticks(ypos)
    ax.set_yticklabels(zone_labels, fontsize=11.5, color=palette["ink_primary"])
    ax.set_xlim(0, maxval)
    ax.set_xlabel("Besiktas ball recoveries")
    ax.grid(axis="x")
    ax.set_axisbelow(True)

    legend_elems = [Line2D([0], [0], marker="s", color=palette["surface"], markerfacecolor=BES_C,
                            markersize=12, label="Leg 1 (home)", linewidth=0),
                    Line2D([0], [0], marker="s", color=palette["surface"], markerfacecolor=BES_C, alpha=0.45,
                            markersize=12, label="Leg 2 (away)", linewidth=0)]
    fig.legend(handles=legend_elems, loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 0.02), fontsize=10.5, labelcolor=palette["ink_secondary"])

    components.header(fig, kicker="Recoveries",
                       title="Besiktas won the ball back much higher up the pitch at home",
                       dek="Besiktas' ball recoveries by pitch third, each leg",
                       palette=palette)
    components.footer(fig, source=md.SOURCE, palette=palette)
    save(fig, "24_recoveries.png")


# ---------------------------------------------------------------------------
# 25-26. Crossing map -- one page per leg
# ---------------------------------------------------------------------------

def crossing_map_leg(leg, page_num):
    fig, palette = new_fig()
    pitch = new_pitch(palette)
    ax = fig.add_axes([0.05, 0.10, 0.90, 0.62])
    pitch.draw(ax=ax)

    crosses = [p for p in leg.besiktas(leg.passes) if p["is_cross"] and p["end_x"] is not None]
    completed = [p for p in crosses if p["completed"]]
    incomplete = [p for p in crosses if not p["completed"]]
    for p in incomplete:
        pitch.arrows(p["x"], p["y"], p["end_x"], p["end_y"], ax=ax, color=palette["axis"],
                    alpha=0.55, width=1.4, headwidth=5, headlength=5, zorder=2)
    for p in completed:
        pitch.arrows(p["x"], p["y"], p["end_x"], p["end_y"], ax=ax, color=BES_C,
                    alpha=0.9, width=2.2, headwidth=6, headlength=6, zorder=3)

    acc = len(completed) / len(crosses) if crosses else 0
    legend_elems = [Line2D([0], [0], color=BES_C, lw=2.2, label="Completed"),
                    Line2D([0], [0], color=palette["axis"], lw=1.4, label="Incomplete")]
    fig.legend(handles=legend_elems, loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 0.06), fontsize=10.5, labelcolor=palette["ink_secondary"])

    components.header(fig, kicker="Crossing",
                       title=f"Leg {leg.leg_num}: Besiktas found a teammate with {len(completed)} of "
                             f"{len(crosses)} crosses ({acc:.0%})",
                       dek=f"{leg.score_line}  ·  all open-play and set-piece crosses, attacking right",
                       palette=palette)
    components.footer(fig, source=md.SOURCE, palette=palette)
    save(fig, f"{page_num}_crossing_map_leg{leg.leg_num}.png")


# ---------------------------------------------------------------------------
# 27-28. Long balls -- one page per leg
# ---------------------------------------------------------------------------

def long_balls_leg(leg, page_num):
    fig, palette = new_fig()
    pitch = new_pitch(palette)
    ax = fig.add_axes([0.05, 0.10, 0.90, 0.62])
    pitch.draw(ax=ax)

    long_balls = [p for p in leg.besiktas(leg.passes) if p["is_long_ball"] and p["end_x"] is not None]
    completed = [p for p in long_balls if p["completed"]]
    incomplete = [p for p in long_balls if not p["completed"]]
    for p in incomplete:
        pitch.arrows(p["x"], p["y"], p["end_x"], p["end_y"], ax=ax, color=palette["axis"],
                    alpha=0.55, width=1.4, headwidth=5, headlength=5, zorder=2)
    for p in completed:
        pitch.arrows(p["x"], p["y"], p["end_x"], p["end_y"], ax=ax, color=BES_C,
                    alpha=0.9, width=2.2, headwidth=6, headlength=6, zorder=3)

    acc = len(completed) / len(long_balls) if long_balls else 0
    legend_elems = [Line2D([0], [0], color=BES_C, lw=2.2, label="Completed"),
                    Line2D([0], [0], color=palette["axis"], lw=1.4, label="Incomplete")]
    fig.legend(handles=legend_elems, loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 0.06), fontsize=10.5, labelcolor=palette["ink_secondary"])

    components.header(fig, kicker="Direct Play",
                       title=f"Leg {leg.leg_num}: {len(completed)} of {len(long_balls)} Besiktas long balls "
                             f"found a teammate ({acc:.0%})",
                       dek=f"{leg.score_line}  ·  passes tagged long ball by Opta, attacking right",
                       palette=palette)
    components.footer(fig, source=md.SOURCE, palette=palette)
    save(fig, f"{page_num}_long_balls_leg{leg.leg_num}.png")


# ---------------------------------------------------------------------------
# 29-30. Aerial duels -- one page per leg
# ---------------------------------------------------------------------------

def aerial_duels_leg(leg, page_num):
    fig, palette = new_fig()
    pitch = new_pitch(palette)
    ax = fig.add_axes([0.05, 0.10, 0.90, 0.62])
    pitch.draw(ax=ax)

    team_duels = [d for d in leg.besiktas(leg.duels) if d["action"] == "Aerial"]
    won = [d for d in team_duels if d["success"]]
    lost = [d for d in team_duels if not d["success"]]
    if lost:
        pitch.scatter([d["x"] for d in lost], [d["y"] for d in lost], ax=ax, s=90, marker="x",
                      color=palette["ink_muted"], linewidth=1.6, alpha=0.85, zorder=3)
    if won:
        pitch.scatter([d["x"] for d in won], [d["y"] for d in won], ax=ax, s=110, marker="o",
                      color=BES_C, edgecolors=palette["surface"], linewidth=1.0, zorder=4)

    win_rate = len(won) / len(team_duels) if team_duels else 0
    legend_elems = [Line2D([0], [0], marker="o", color=palette["surface"], markerfacecolor=BES_C,
                            markersize=10, label="Won", linewidth=0),
                    Line2D([0], [0], marker="x", color=palette["ink_muted"], markersize=10, label="Lost", linewidth=0)]
    fig.legend(handles=legend_elems, loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 0.06), fontsize=10.5, labelcolor=palette["ink_secondary"])

    components.header(fig, kicker="Aerial Duels",
                       title=f"Leg {leg.leg_num}: Besiktas won {len(won)} of {len(team_duels)} aerial duels ({win_rate:.0%})",
                       dek=f"{leg.score_line}  ·  Besiktas' own goal on the left, attacking right",
                       palette=palette)
    components.footer(fig, source=md.SOURCE, palette=palette)
    save(fig, f"{page_num}_aerial_duels_leg{leg.leg_num}.png")


# ---------------------------------------------------------------------------
# 31-32. Zone 14 & half-space passes (destination) -- one page per leg
# ---------------------------------------------------------------------------

def zone14_leg(leg, page_num):
    fig, palette = new_fig()
    pitch = new_pitch(palette)
    ax = fig.add_axes([0.05, 0.10, 0.90, 0.62])
    pitch.draw(ax=ax)

    z14x0, z14x1, z14y0, z14y1 = md.ZONE14
    pitch.polygon([np.array([(z14x0, z14y0), (z14x1, z14y0), (z14x1, z14y1), (z14x0, z14y1)])],
                  ax=ax, color=CATEGORICAL_DARK[6], alpha=0.18, zorder=1)
    for hx0, hx1, hy0, hy1 in md.HALF_SPACES:
        pitch.polygon([np.array([(hx0, hy0), (hx1, hy0), (hx1, hy1), (hx0, hy1)])],
                      ax=ax, color=BES_C, alpha=0.10, zorder=1)

    team_passes = [p for p in leg.besiktas(leg.passes) if p["completed"] and p["end_x"] is not None]
    z14 = [p for p in team_passes if z14x0 <= p["end_x"] <= z14x1 and z14y0 <= p["end_y"] <= z14y1]
    hs = [p for p in team_passes
          if any(hx0 <= p["end_x"] <= hx1 and hy0 <= p["end_y"] <= hy1 for hx0, hx1, hy0, hy1 in md.HALF_SPACES)
          and p not in z14]

    for p in hs:
        pitch.arrows(p["x"], p["y"], p["end_x"], p["end_y"], ax=ax, color=BES_C, alpha=0.55,
                    width=1.4, headwidth=5, headlength=5, zorder=3)
    for p in z14:
        pitch.arrows(p["x"], p["y"], p["end_x"], p["end_y"], ax=ax, color=palette["ink_primary"], alpha=0.9,
                    width=2.0, headwidth=6, headlength=6, zorder=4)

    legend_elems = [Line2D([0], [0], color=palette["ink_primary"], lw=2.0, label=f"Into Zone 14 ({len(z14)})"),
                    Line2D([0], [0], color=BES_C, lw=1.6, label=f"Into a half-space ({len(hs)})")]
    fig.legend(handles=legend_elems, loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 0.06), fontsize=10.5, labelcolor=palette["ink_secondary"])

    components.header(fig, kicker="Central Access",
                       title=f"Leg {leg.leg_num}: how often did Besiktas find Zone 14 and the half-spaces?",
                       dek=f"{leg.score_line}  ·  completed passes ending in the central Zone 14 box or "
                           "either half-space channel",
                       palette=palette)
    components.footer(fig, source=md.SOURCE, palette=palette)
    save(fig, f"{page_num}_zone14_halfspace_leg{leg.leg_num}.png")


# ---------------------------------------------------------------------------
# 33-34. Passes ORIGINATING from Zone 14 / half-spaces -- one page per leg
# ---------------------------------------------------------------------------

def origin_zone14_leg(leg, page_num):
    fig, palette = new_fig()
    pitch = new_pitch(palette)
    ax = fig.add_axes([0.05, 0.10, 0.90, 0.62])
    pitch.draw(ax=ax)

    z14x0, z14x1, z14y0, z14y1 = md.ZONE14
    pitch.polygon([np.array([(z14x0, z14y0), (z14x1, z14y0), (z14x1, z14y1), (z14x0, z14y1)])],
                  ax=ax, color=CATEGORICAL_DARK[6], alpha=0.18, zorder=1)
    for hx0, hx1, hy0, hy1 in md.HALF_SPACES:
        pitch.polygon([np.array([(hx0, hy0), (hx1, hy0), (hx1, hy1), (hx0, hy1)])],
                      ax=ax, color=BES_C, alpha=0.10, zorder=1)

    team_passes = [p for p in leg.besiktas(leg.passes) if p["completed"] and p["end_x"] is not None]
    z14 = [p for p in team_passes if z14x0 <= p["x"] <= z14x1 and z14y0 <= p["y"] <= z14y1]
    hs = [p for p in team_passes
          if any(hx0 <= p["x"] <= hx1 and hy0 <= p["y"] <= hy1 for hx0, hx1, hy0, hy1 in md.HALF_SPACES)
          and p not in z14]

    for p in hs:
        pitch.arrows(p["x"], p["y"], p["end_x"], p["end_y"], ax=ax, color=BES_C, alpha=0.55,
                    width=1.4, headwidth=5, headlength=5, zorder=3)
    for p in z14:
        pitch.arrows(p["x"], p["y"], p["end_x"], p["end_y"], ax=ax, color=palette["ink_primary"], alpha=0.9,
                    width=2.0, headwidth=6, headlength=6, zorder=4)

    legend_elems = [Line2D([0], [0], color=palette["ink_primary"], lw=2.0, label=f"From Zone 14 ({len(z14)})"),
                    Line2D([0], [0], color=BES_C, lw=1.6, label=f"From a half-space ({len(hs)})")]
    fig.legend(handles=legend_elems, loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 0.06), fontsize=10.5, labelcolor=palette["ink_secondary"])

    components.header(fig, kicker="Central Progression",
                       title=f"Leg {leg.leg_num}: what did Besiktas do once the ball reached Zone 14 or a half-space?",
                       dek=f"{leg.score_line}  ·  completed passes originating in the central Zone 14 box or "
                           "either half-space channel",
                       palette=palette)
    components.footer(fig, source=md.SOURCE, palette=palette)
    save(fig, f"{page_num}_origin_zone14_halfspace_leg{leg.leg_num}.png")


# ---------------------------------------------------------------------------
# 35. Final third entries, both legs
# ---------------------------------------------------------------------------

def final_third_entries_both_legs(leg1, leg2):
    fig, palette = new_fig()
    pitch = new_pitch(palette)
    ax1 = fig.add_axes([0.02, 0.10, 0.47, 0.62])
    ax2 = fig.add_axes([0.51, 0.10, 0.47, 0.62])
    pitch.draw(ax=ax1)
    pitch.draw(ax=ax2)

    zone_colors = {"Left": CATEGORICAL_DARK[0], "Central": CATEGORICAL_DARK[2], "Right": CATEGORICAL_DARK[1]}

    def zone_of(y):
        if y >= 45.33:
            return "Left"
        if y <= 22.67:
            return "Right"
        return "Central"

    for ax, leg in ((ax1, leg1), (ax2, leg2)):
        entries = [p for p in leg.besiktas(leg.passes) if p["final_third_entry"]]
        counts = {"Left": 0, "Central": 0, "Right": 0}
        for p in entries:
            zone = zone_of(p["y"])
            counts[zone] += 1
            pitch.arrows(p["x"], p["y"], p["end_x"], p["end_y"], ax=ax, color=zone_colors[zone],
                        width=1.6, headwidth=5, headlength=5, alpha=0.85, zorder=3)
        title = (f"Leg {leg.leg_num}  ({len(entries)} entries)\n"
                 f"L: {counts['Left']}  ·  C: {counts['Central']}  ·  R: {counts['Right']}")
        ax.set_title(title, color=BES_C, fontsize=11.5, fontweight="bold", family="sans-serif")

    legend_elems = [Line2D([0], [0], color=zone_colors[z], lw=2.4, label=z) for z in ("Left", "Central", "Right")]
    fig.legend(handles=legend_elems, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 0.02), fontsize=10.5, labelcolor=palette["ink_secondary"])

    components.header(fig, kicker="Final Third",
                       title="Besiktas found the final third down the flanks in both legs",
                       dek="Besiktas' completed passes into the final third, by origin lane",
                       palette=palette)
    components.footer(fig, source=md.SOURCE, palette=palette)
    save(fig, "35_final_third_entries.png")


# ---------------------------------------------------------------------------
# 36. Opponent touch heatmap, both legs
# ---------------------------------------------------------------------------

def opponent_touch_heatmap_both_legs(leg1, leg2):
    fig, palette = new_fig()
    pitch = new_pitch(palette)
    ax1 = fig.add_axes([0.02, 0.10, 0.47, 0.62])
    ax2 = fig.add_axes([0.51, 0.10, 0.47, 0.62])

    cmap = plt.cm.colors.LinearSegmentedColormap.from_list("opp", [palette["surface"], OPP_C])

    for ax, leg in ((ax1, leg1), (ax2, leg2)):
        pitch.draw(ax=ax)
        touches = leg.opponent(leg.touches)
        xs = [t["x"] for t in touches]
        ys = [t["y"] for t in touches]
        pitch.kdeplot(xs, ys, ax=ax, cmap=cmap, fill=True, levels=60, alpha=0.85, zorder=1)
        ax.set_title(f"Leg {leg.leg_num}  ·  {1 - leg.touch_share():.0%} touch share",
                     color=OPP_C, fontsize=12.5, fontweight="bold", family="sans-serif")

    components.header(fig, kicker="Territory",
                       title="Midtjylland's threat came from central areas, not the flanks",
                       dek="Midtjylland touch density, both legs (own goal on the left)",
                       palette=palette)
    components.footer(fig, source=md.SOURCE, palette=palette)
    save(fig, "36_opponent_touch_heatmap.png")


# ---------------------------------------------------------------------------
# 37. Build-up to shot, leg by leg
# ---------------------------------------------------------------------------

def _possession_chain_lengths(events, shots, team_id):
    ball_events = [e for e in events if e.get("x") is not None and e.get("contestantId")
                   and (e["typeId"] in (1, 3, 7, 8, 12, 44, 49, 50, 61) or e["typeId"] in md.SHOT_TYPES)]
    ball_events.sort(key=lambda e: (e["periodId"], md.event_time(e), e["eventId"]))

    lengths = []
    run_team, run_len = None, 0
    for e in ball_events:
        cid = e["contestantId"]
        if cid == run_team:
            if e["typeId"] == 1 and e.get("outcome") == 1:
                run_len += 1
        else:
            run_team, run_len = cid, (1 if e["typeId"] == 1 and e.get("outcome") == 1 else 0)
        if e["typeId"] in md.SHOT_TYPES and cid == team_id:
            lengths.append(run_len)
    return lengths


def buildup_to_shot_both_legs(leg1, leg2):
    fig, palette = new_fig()
    ax1 = fig.add_axes([0.08, 0.16, 0.40, 0.58])
    ax2 = fig.add_axes([0.56, 0.16, 0.40, 0.58])

    bins = [(0, 3, "Direct (0-3)"), (4, 6, "Build-up (4-6)"), (7, 99, "Elaborate (7+)")]

    for ax, leg in ((ax1, leg1), (ax2, leg2)):
        lengths = _possession_chain_lengths(leg.events, leg.shots, md.BESIKTAS_ID)
        counts = [sum(1 for n in lengths if lo <= n <= hi) for lo, hi, _ in bins]
        xs = np.arange(len(bins))
        ax.bar(xs, counts, color=BES_C)
        for x, c in zip(xs, counts):
            ax.text(x, c + max(counts, default=0) * 0.03 + 0.05, str(c), ha="center", fontsize=10.5,
                    color=palette["ink_primary"], fontweight="bold")
        ax.set_xticks(xs)
        ax.set_xticklabels([b[2] for b in bins], fontsize=9)
        ax.set_title(f"Leg {leg.leg_num}\n{len(lengths)} shot sequences", color=BES_C, fontsize=11.5,
                     fontweight="bold", family="sans-serif")
        ax.set_ylabel("Sequences")

    components.header(fig, kicker="Build-Up",
                       title="Besiktas' shots mostly came from short, direct sequences in both legs",
                       dek="Besiktas' possession sequences ending in a shot, grouped by completed-pass count",
                       palette=palette)
    components.footer(fig, source=md.SOURCE, palette=palette)
    save(fig, "37_buildup_to_shot.png")


# ---------------------------------------------------------------------------
# 38. Set piece analysis (corners), leg by leg
# ---------------------------------------------------------------------------

def set_piece_analysis_both_legs(leg1, leg2):
    fig, palette = new_fig()
    pitch = new_pitch(palette)
    ax1 = fig.add_axes([0.02, 0.10, 0.47, 0.62])
    ax2 = fig.add_axes([0.51, 0.10, 0.47, 0.62])
    pitch.draw(ax=ax1)
    pitch.draw(ax=ax2)

    for ax, leg in ((ax1, leg1), (ax2, leg2)):
        corners = [p for p in leg.besiktas(leg.passes) if p["is_corner"] and p["end_x"] is not None]
        completed = [c for c in corners if c["completed"]]
        for c in corners:
            pitch.arrows(c["x"], c["y"], c["end_x"], c["end_y"], ax=ax,
                        color=BES_C if c["completed"] else palette["axis"],
                        alpha=0.85 if c["completed"] else 0.5, width=1.8, headwidth=5, headlength=5,
                        zorder=3 if c["completed"] else 2)
        shots_from_corner = sum(1 for s in leg.besiktas(leg.shots) if s["situation"] == "Corner")
        title = (f"Leg {leg.leg_num}  ({len(corners)} corners, {len(completed)} found a teammate)\n"
                 f"{shots_from_corner} shot(s) from a corner")
        ax.set_title(title, color=BES_C, fontsize=11, fontweight="bold", family="sans-serif")

    components.header(fig, kicker="Set Pieces",
                       title="Besiktas' corner delivery and what it produced",
                       dek="Every Besiktas corner kick, completed (solid) vs not, attacking right",
                       palette=palette)
    components.footer(fig, source=md.SOURCE, palette=palette)
    save(fig, "38_set_piece_analysis.png")


# ---------------------------------------------------------------------------
# 39. Defensive line height, both legs
# ---------------------------------------------------------------------------

def defensive_line_height_both_legs(leg1, leg2):
    fig, palette = new_fig()
    ax = fig.add_axes([0.10, 0.16, 0.80, 0.58])

    bucket = 15
    edges = list(range(0, 96, bucket)) + [96]
    centers = [(edges[i] + min(edges[i + 1], 96)) / 2 for i in range(len(edges) - 1)]

    for leg, style_kw in ((leg1, dict(linestyle="-", marker="o")), (leg2, dict(linestyle="--", marker="s"))):
        pressing = leg.besiktas(leg.pressing)
        heights = []
        for i in range(len(edges) - 1):
            lo, hi = edges[i], edges[i + 1]
            xs = [d["x"] for d in pressing if lo <= d["minute"] < hi]
            heights.append(np.mean(xs) if xs else np.nan)
        ax.plot(centers, heights, color=BES_C, linewidth=2.2, markersize=6, zorder=3,
                label=f"Leg {leg.leg_num}", **style_kw)

    ax.set_xlim(0, 100)
    ax.set_ylim(0, 70)
    ax.set_xlabel("Minute")
    ax.set_ylabel("Average distance from own goal (m)")
    ax.axvline(45, color=palette["axis"], linewidth=0.8, linestyle=":")
    ax.legend(loc="lower right", frameon=False, fontsize=10, labelcolor=palette["ink_secondary"])

    components.header(fig, kicker="Defensive Line",
                       title="Besiktas defended from a higher line at home than they did away",
                       dek="Average location of Besiktas' tackles, interceptions and challenges, 15-minute buckets",
                       palette=palette)
    components.footer(fig, source=md.SOURCE, palette=palette)
    save(fig, "39_defensive_line_height.png")


# ---------------------------------------------------------------------------
# 40. Shot zones heatmap, both legs
# ---------------------------------------------------------------------------

def shot_zones_heatmap_both_legs(leg1, leg2):
    fig, palette = new_fig()
    pitch = new_pitch(palette)
    ax1 = fig.add_axes([0.02, 0.10, 0.47, 0.62])
    ax2 = fig.add_axes([0.51, 0.10, 0.47, 0.62])

    for ax, leg in ((ax1, leg1), (ax2, leg2)):
        pitch.draw(ax=ax)
        team_shots = leg.besiktas(leg.shots)
        xs = [s["x"] for s in team_shots]
        ys = [s["y"] for s in team_shots]
        if xs:
            stats = pitch.bin_statistic(xs, ys, statistic="count", bins=(6, 5))
            stats["statistic"] = np.where(stats["statistic"] == 0, np.nan, stats["statistic"])
            cmap_obj = plt.cm.colors.LinearSegmentedColormap.from_list("wa_shots", [palette["surface"], BES_C])
            cmap_obj.set_bad(alpha=0)
            pitch.heatmap(stats, ax=ax, cmap=cmap_obj, vmin=0, edgecolors="none", alpha=0.9, zorder=1)
        pitch.scatter(xs, ys, ax=ax, s=40, color=palette["ink_primary"], edgecolors=palette["surface"],
                      linewidth=0.6, zorder=3)
        ax.set_title(f"Leg {leg.leg_num}  ({len(team_shots)} shots)", color=BES_C, fontsize=12,
                     fontweight="bold", family="sans-serif")

    components.header(fig, kicker="Shot Locations",
                       title="Besiktas' shots came from much closer range at home",
                       dek="Besiktas' shot density with individual attempts marked, attacking right",
                       palette=palette)
    components.footer(fig, source=md.SOURCE, palette=palette)
    save(fig, "40_shot_zones_heatmap.png")


# ---------------------------------------------------------------------------
# 41. Passing direction breakdown, leg by leg
# ---------------------------------------------------------------------------

def passing_direction_both_legs(leg1, leg2):
    fig, palette = new_fig()
    ax = fig.add_axes([0.16, 0.24, 0.68, 0.40])

    dir_colors = [CATEGORICAL_DARK[0], CATEGORICAL_DARK[3], CATEGORICAL_DARK[1]]
    dir_labels = ["Forward", "Sideways", "Backward"]

    def classify(p):
        dx = p["end_x"] - p["x"]
        dy = abs(p["end_y"] - p["y"])
        if dx > 5 and dx > dy:
            return "Forward"
        if dx < -5 and abs(dx) > dy:
            return "Backward"
        return "Sideways"

    def fracs(leg):
        team_passes = [p for p in leg.besiktas(leg.passes) if p["completed"] and p["end_x"] is not None]
        counts = {k: 0 for k in dir_labels}
        for p in team_passes:
            counts[classify(p)] += 1
        total = sum(counts.values()) or 1
        return [counts[k] / total for k in dir_labels], total

    for i, leg in enumerate((leg1, leg2)):
        vals, total = fracs(leg)
        y = 1 - i
        left = 0
        for frac, dc, dl in zip(vals, dir_colors, dir_labels):
            ax.barh(y, frac, left=left, height=0.6, color=dc)
            if frac > 0.06:
                ax.text(left + frac / 2, y, f"{frac:.0%}", ha="center", va="center",
                        fontsize=10.5, fontweight="bold", color=palette["surface"])
            left += frac
        ax.text(-0.02, y, f"Leg {leg.leg_num}\n({total} passes)", ha="right", va="center", fontsize=10.5,
                fontweight="bold", color=BES_C)

    ax.set_xlim(0, 1)
    ax.set_ylim(-0.7, 1.7)
    ax.set_yticks([])
    ax.set_xlabel("Share of Besiktas' completed passes")
    ax.set_axisbelow(True)

    legend_elems = [Line2D([0], [0], marker="s", color=palette["surface"], markerfacecolor=dc,
                            markersize=12, label=dl, linewidth=0) for dc, dl in zip(dir_colors, dir_labels)]
    fig.legend(handles=legend_elems, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 0.06), fontsize=10.5, labelcolor=palette["ink_secondary"])

    components.header(fig, kicker="Passing Style",
                       title="Besiktas' passing was no less direct on the road",
                       dek="Forward/sideways/backward classified from each pass's start/end location (≥5m threshold)",
                       palette=palette)
    components.footer(fig, source=md.SOURCE, palette=palette)
    save(fig, "41_passing_direction.png")


# ---------------------------------------------------------------------------
# 42. xT flow, both legs
# ---------------------------------------------------------------------------

def xt_flow_both_legs(leg1, leg2):
    fig, palette = new_fig()
    ax1 = fig.add_axes([0.07, 0.16, 0.40, 0.58])
    ax2 = fig.add_axes([0.56, 0.16, 0.40, 0.58])

    def series(passes):
        ordered = sorted(passes, key=lambda p: (p["period"], p["minute"] * 60 + p["second"]))
        mins, cum, total = [0.0], [0.0], 0.0
        for p in ordered:
            total += p["xt_added"]
            mins.append(p["minute"])
            cum.append(total)
        mins.append(96)
        cum.append(total)
        return mins, cum

    for ax, leg in ((ax1, leg1), (ax2, leg2)):
        for rows_fn, color, name in ((leg.besiktas, BES_C, BES_SHORT), (leg.opponent, OPP_C, OPP_SHORT)):
            passes = [p for p in rows_fn(leg.passes) if p["completed"]]
            mins, cum = series(passes)
            ax.plot(mins, cum, color=color, linewidth=2.0, zorder=4)
            ax.fill_between(mins, cum, color=color, alpha=0.10, zorder=1)
        ax.axhline(0, color=palette["axis"], linewidth=0.8)
        ax.axvline(45, color=palette["axis"], linewidth=0.8, linestyle=":")
        ax.set_xlim(0, 100)
        ax.set_xlabel("Minute")
        ax.set_title(f"Leg {leg.leg_num}: {leg.score_line}", fontsize=11, fontweight="bold",
                     family="sans-serif", color=palette["ink_primary"])
    ax1.set_ylabel("Cumulative threat added")

    legend_elems = [Line2D([0], [0], color=BES_C, lw=2.2, label="Besiktas"),
                    Line2D([0], [0], color=OPP_C, lw=2.2, label="Midtjylland")]
    fig.legend(handles=legend_elems, loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 0.02), fontsize=10.5, labelcolor=palette["ink_secondary"])

    components.header(fig, kicker="xT Flow",
                       title="Threat generated from passing, minute by minute",
                       dek="Cumulative threat added by completed passes  ·  own simplified threat surface "
                           "(shot-xG geometry, not a possession-value model)",
                       palette=palette)
    components.footer(fig, source=md.SOURCE, palette=palette)
    save(fig, "42_xt_flow.png")


# ---------------------------------------------------------------------------
# 43. xT leaderboard, combined
# ---------------------------------------------------------------------------

def xt_leaderboard(leg1, leg2):
    fig, palette = new_fig()
    ax = fig.add_axes([0.22, 0.16, 0.68, 0.58])

    scores = {}
    for leg in (leg1, leg2):
        for p in leg.besiktas(leg.passes):
            if not p["completed"]:
                continue
            scores[p["player"]] = scores.get(p["player"], 0.0) + p["xt_added"]

    top = sorted(scores.items(), key=lambda kv: -kv[1])[:10][::-1]
    ypos = np.arange(len(top))
    vals = [v for _, v in top]
    ax.barh(ypos, vals, color=BES_C)
    ax.set_yticks(ypos)
    ax.set_yticklabels([p for p, _ in top], fontsize=10.5, color=palette["ink_primary"])
    for y, v in zip(ypos, vals):
        ax.text(v + max(vals, default=0.01) * 0.02, y, f"{v:.2f}", va="center", fontsize=9.5,
                color=palette["ink_secondary"])
    ax.set_xlabel("Threat added by completed passes, both legs combined")
    ax.axvline(0, color=palette["axis"], linewidth=0.8)

    components.header(fig, kicker="xT Leaderboard",
                       title="Besiktas' most threatening passers across the tie",
                       dek="Total threat added by completed passes, own simplified threat surface",
                       palette=palette)
    components.footer(fig, source=md.SOURCE, palette=palette)
    save(fig, "43_xt_leaderboard.png")


# ---------------------------------------------------------------------------
# 44. Shot assists map, both legs
# ---------------------------------------------------------------------------

def shot_assists_map_both_legs(leg1, leg2):
    fig, palette = new_fig()
    pitch = new_pitch(palette)
    ax1 = fig.add_axes([0.02, 0.10, 0.47, 0.62])
    ax2 = fig.add_axes([0.51, 0.10, 0.47, 0.62])
    pitch.draw(ax=ax1)
    pitch.draw(ax=ax2)

    for ax, leg in ((ax1, leg1), (ax2, leg2)):
        team_assists = [a for a in leg.assists if a["contestantId"] == md.BESIKTAS_ID]
        for a in team_assists:
            is_goal = a["is_goal"]
            pitch.arrows(a["x"], a["y"], a["end_x"], a["end_y"], ax=ax,
                        color=GOOD_C if is_goal else BES_C, alpha=0.95 if is_goal else 0.6,
                        width=2.6 if is_goal else 1.5, headwidth=6, headlength=6,
                        zorder=4 if is_goal else 3)
        ax.set_title(f"Leg {leg.leg_num}  ({len(team_assists)} shot assists)", color=BES_C,
                     fontsize=12, fontweight="bold", family="sans-serif")

    legend_elems = [Line2D([0], [0], color=palette["ink_muted"], lw=1.8, label="Led to a shot"),
                    Line2D([0], [0], color=GOOD_C, lw=2.6, label="Led to a goal")]
    fig.legend(handles=legend_elems, loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 0.02), fontsize=10.5, labelcolor=palette["ink_secondary"])

    components.header(fig, kicker="Shot Assists",
                       title="The pass that unlocked each Besiktas shot",
                       dek="Most recent completed pass by Besiktas before the shot, attacking right",
                       palette=palette)
    components.footer(fig, source=md.SOURCE, palette=palette)
    save(fig, "44_shot_assists_map.png")


# ---------------------------------------------------------------------------
# 45. Key passes / xA leaderboard, combined
# ---------------------------------------------------------------------------

def key_passes_leaderboard(leg1, leg2):
    fig, palette = new_fig()
    ax = fig.add_axes([0.05, 0.12, 0.90, 0.62])
    ax.axis("off")

    by_player = {}
    for leg in (leg1, leg2):
        for a in leg.assists:
            if a["contestantId"] != md.BESIKTAS_ID:
                continue
            d = by_player.setdefault(a["assister"], {"n": 0, "xa": 0.0, "goals": 0})
            d["n"] += 1
            d["xa"] += a["shot_xg"]
            d["goals"] += int(a["is_goal"])
    rows = sorted(by_player.items(), key=lambda kv: -kv[1]["xa"])

    cols = ["Player", "Shot assists", "Goal assists", "xA (shot xG created)"]
    widths = [0.36, 0.20, 0.20, 0.24]
    x0 = [sum(widths[:i]) for i in range(len(widths))]
    header_y = 1.0
    for x, label in zip(x0, cols):
        ax.text(x, header_y, label, fontsize=10.5, fontweight="bold", color=palette["ink_primary"], va="top")
    ax.axhline(header_y - 0.03, xmin=0, xmax=1, color=palette["axis"], linewidth=1.0)

    row_h = 0.9 / max(len(rows), 1)
    for i, (player, d) in enumerate(rows):
        y = header_y - 0.06 - i * row_h
        vals = [player, str(d["n"]), str(d["goals"]), f"{d['xa']:.2f}"]
        for x, v in zip(x0, vals):
            ax.text(x, y, v, fontsize=9.5, color=BES_C if x == x0[0] else palette["ink_primary"], va="top")
    ax.set_xlim(0, 1)
    ax.set_ylim(header_y - 0.06 - len(rows) * row_h, 1.03)

    components.header(fig, kicker="Key Passes",
                       title="Olaitan created both of Besiktas' second-leg goals",
                       dek="xA proxy = sum of xG on shots created by that player's pass, both legs combined",
                       palette=palette)
    components.footer(fig, source=md.SOURCE, palette=palette)
    save(fig, "45_key_passes_leaderboard.png")


# ---------------------------------------------------------------------------
# 46. Turnovers in dangerous areas, both legs
# ---------------------------------------------------------------------------

def turnovers_dangerous_both_legs(leg1, leg2):
    fig, palette = new_fig()
    pitch = new_pitch(palette)
    ax1 = fig.add_axes([0.02, 0.10, 0.47, 0.62])
    ax2 = fig.add_axes([0.51, 0.10, 0.47, 0.62])
    pitch.draw(ax=ax1)
    pitch.draw(ax=ax2)

    markers = {"Failed pass": "o", "Dispossessed": "X"}
    for ax, leg in ((ax1, leg1), (ax2, leg2)):
        team_t = leg.besiktas(leg.turnovers)
        for kind, marker in markers.items():
            pts = [t for t in team_t if t["kind"] == kind]
            if not pts:
                continue
            pitch.scatter([p["x"] for p in pts], [p["y"] for p in pts], ax=ax, s=80, marker=marker,
                          color=BES_C, edgecolors=palette["surface"], linewidth=0.8, alpha=0.85, zorder=3)
        ax.set_title(f"Leg {leg.leg_num}  ({len(team_t)} lost in Besiktas' own attacking half)", color=BES_C,
                     fontsize=11, fontweight="bold", family="sans-serif")

    legend_elems = [Line2D([0], [0], marker=markers[k], color=palette["surface"], markerfacecolor=palette["ink_secondary"],
                            markeredgecolor=palette["ink_secondary"], markersize=10, label=k, linewidth=0) for k in markers]
    fig.legend(handles=legend_elems, loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 0.02), fontsize=10.5, labelcolor=palette["ink_secondary"])

    components.header(fig, kicker="Turnovers",
                       title="Where Besiktas gave the ball away going forward",
                       dek="Failed passes and dispossessions in Besiktas' own attacking half, own goal on the left",
                       palette=palette)
    components.footer(fig, source=md.SOURCE, palette=palette)
    save(fig, "46_turnovers_dangerous.png")


# ---------------------------------------------------------------------------
# 47. Team radar comparison, aggregate
# ---------------------------------------------------------------------------

def team_radar(leg1, leg2):
    fig, palette = new_fig()
    ax = fig.add_axes([0.26, 0.18, 0.48, 0.58], polar=True)

    bes_pass = leg1.besiktas(leg1.passes) + leg2.besiktas(leg2.passes)
    opp_pass = leg1.opponent(leg1.passes) + leg2.opponent(leg2.passes)
    bes_shots = leg1.besiktas(leg1.shots) + leg2.besiktas(leg2.shots)
    opp_shots = leg1.opponent(leg1.shots) + leg2.opponent(leg2.shots)
    bes_touch = leg1.besiktas(leg1.touches) + leg2.besiktas(leg2.touches)
    opp_touch = leg1.opponent(leg1.touches) + leg2.opponent(leg2.touches)
    bes_ft = sum(1 for t in bes_touch if t["x"] >= 70)
    opp_ft = sum(1 for t in opp_touch if t["x"] >= 70)

    pooled_passes = leg1.passes + leg2.passes
    pooled_pressing = leg1.pressing + leg2.pressing
    bes_ppda = md.compute_ppda(pooled_passes, pooled_pressing, md.BESIKTAS_ID, md.MIDTJYLLAND_ID)
    opp_ppda = md.compute_ppda(pooled_passes, pooled_pressing, md.MIDTJYLLAND_ID, md.BESIKTAS_ID)

    metrics = [
        ("xG", sum(s["xg"] for s in bes_shots), sum(s["xg"] for s in opp_shots)),
        ("Shots", len(bes_shots), len(opp_shots)),
        ("Progressive passes", sum(1 for p in bes_pass if p["progressive"]), sum(1 for p in opp_pass if p["progressive"])),
        ("Box entries", sum(1 for p in bes_pass if p["box_entry"]), sum(1 for p in opp_pass if p["box_entry"])),
        ("Final-third touches", bes_ft, opp_ft),
        ("Pass accuracy", sum(1 for p in bes_pass if p["completed"]) / len(bes_pass),
         sum(1 for p in opp_pass if p["completed"]) / len(opp_pass)),
        ("Pressing (inv. PPDA)", 1 / bes_ppda, 1 / opp_ppda),
    ]
    labels = [m[0] for m in metrics]
    n = len(labels)
    bes_norm = [m[1] / max(m[1], m[2], 1e-9) for m in metrics]
    opp_norm = [m[2] / max(m[1], m[2], 1e-9) for m in metrics]

    angles = [i / n * 2 * math.pi for i in range(n)] + [0]
    for vals, color, name in ((bes_norm, BES_C, BES_SHORT), (opp_norm, OPP_C, OPP_SHORT)):
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

    components.header(fig, kicker="Head to Head",
                       title="A shape comparison across the tie's key numbers",
                       dek="Each axis normalized to the better of the two sides across both legs combined (=1.0)",
                       palette=palette)
    components.footer(fig, source=md.SOURCE, palette=palette)
    save(fig, "47_team_radar.png")


# ---------------------------------------------------------------------------
# 48. Win probability & xG by situation, aggregate simulation
# ---------------------------------------------------------------------------

def _donut(ax, frac, color, palette, label, sublabel):
    ax.pie([frac, 1 - frac], radius=1.0, startangle=90, counterclock=False,
           colors=[color, palette["axis"]], wedgeprops=dict(width=0.32, edgecolor=palette["surface"], linewidth=1.5))
    ax.text(0, 0.12, f"{frac:.0%}", ha="center", va="center", fontsize=20, fontweight="bold", color=palette["ink_primary"])
    ax.text(0, -0.12, sublabel, ha="center", va="center", fontsize=8.5, color=palette["ink_muted"])
    ax.set_title(label, color=color, fontsize=11.5, fontweight="bold", family="sans-serif", pad=2)


def win_probability_situation(leg1, leg2, sim):
    fig, palette = new_fig()
    ax1 = fig.add_axes([0.06, 0.34, 0.19, 0.32])
    ax2 = fig.add_axes([0.28, 0.34, 0.19, 0.32])
    _donut(ax1, sim["besiktas_win"], BES_C, palette, BES_SHORT, "TIE WIN PROBABILITY")
    _donut(ax2, sim["opponent_win"], OPP_C, palette, OPP_SHORT, "TIE WIN PROBABILITY")
    fig.text(0.275, 0.30, f"Draw: {sim['draw']:.0%}", ha="center", fontsize=10.5,
              color=palette["ink_secondary"], fontweight="bold")

    ax3 = fig.add_axes([0.56, 0.20, 0.38, 0.52])
    bes_shots = leg1.besiktas(leg1.shots) + leg2.besiktas(leg2.shots)
    opp_shots = leg1.opponent(leg1.shots) + leg2.opponent(leg2.shots)
    situations = ["Open play", "Fast break", "Set piece", "Corner"]
    bes_vals = [sum(s["xg"] for s in bes_shots if s["situation"] == sit) for sit in situations]
    opp_vals = [sum(s["xg"] for s in opp_shots if s["situation"] == sit) for sit in situations]
    n = len(situations)
    ypos = np.arange(n)[::-1]
    maxval = max(bes_vals + opp_vals) * 1.25 or 1
    for y, h, a in zip(ypos, bes_vals, opp_vals):
        ax3.barh(y + 0.18, h, height=0.32, color=BES_C)
        ax3.barh(y - 0.18, a, height=0.32, color=OPP_C)
        ax3.text(h + maxval * 0.02, y + 0.18, f"{h:.2f}", va="center", fontsize=9.5, color=palette["ink_primary"])
        ax3.text(a + maxval * 0.02, y - 0.18, f"{a:.2f}", va="center", fontsize=9.5, color=palette["ink_primary"])
    ax3.set_yticks(ypos)
    ax3.set_yticklabels(situations, fontsize=10.5, color=palette["ink_primary"])
    ax3.set_xlim(0, maxval)
    ax3.set_xlabel("xG, both legs combined")
    ax3.grid(axis="x")
    ax3.set_axisbelow(True)
    ax3.set_title("xG by situation", color=palette["ink_primary"], fontsize=11.5, fontweight="bold", family="sans-serif")

    legend_elems = [Line2D([0], [0], marker="s", color=palette["surface"], markerfacecolor=BES_C,
                            markersize=12, label="Besiktas", linewidth=0),
                    Line2D([0], [0], marker="s", color=palette["surface"], markerfacecolor=OPP_C,
                            markersize=12, label="Midtjylland", linewidth=0)]
    fig.legend(handles=legend_elems, loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 0.10), fontsize=10.5, labelcolor=palette["ink_secondary"])

    components.header(fig, kicker="Tie Odds",
                       title=f"On the balance of chances, this was closer to a coin flip than a 3-0 rout ({sim['besiktas_win']:.0%} Besiktas)",
                       dek=f"{sim['n']:,}-simulation Monte Carlo from both legs' pooled shot xG  ·  actual aggregate 3-0",
                       palette=palette)
    components.footer(fig, source=md.SOURCE, palette=palette)
    save(fig, "48_win_probability.png")


# ---------------------------------------------------------------------------
# 49. xG scoreline matrix, aggregate simulation
# ---------------------------------------------------------------------------

def xg_scoreline_matrix(sim):
    fig, palette = new_fig()
    ax = fig.add_axes([0.14, 0.22, 0.66, 0.52])

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
    ax.set_xlabel("Midtjylland goals")
    ax.set_ylabel("Besiktas goals")
    ax.grid(False)

    fth, fta = 3, 0
    rect = plt.Rectangle((min(fta, cap) - 0.5, min(fth, cap) - 0.5), 1, 1, fill=False,
                          edgecolor=palette["ink_primary"], linewidth=2.6, zorder=5)
    ax.add_patch(rect)
    fig.text(0.14, 0.11, "▢ White outline marks the actual aggregate scoreline (Besiktas 3, Midtjylland 0)",
              fontsize=9, color=palette["ink_muted"])

    components.header(fig, kicker="Scoreline Probability",
                       title=("3-0 was the single most likely aggregate scoreline from these chances"
                              if grid[min(fth, cap), min(fta, cap)] == grid.max()
                              else "3-0 was a plausible aggregate outcome from these chances, but not the likeliest"),
                       dek=f"Simulated aggregate scoreline probabilities from pooled shot xG, {n:,} runs  ·  own xG model",
                       palette=palette)
    components.footer(fig, source=md.SOURCE, palette=palette)
    save(fig, "49_xg_scoreline_matrix.png")


# ---------------------------------------------------------------------------
# 50. Substitution impact
# ---------------------------------------------------------------------------

def substitution_impact(leg1, leg2):
    fig, palette = new_fig()
    ax = fig.add_axes([0.10, 0.18, 0.84, 0.56])

    rows = []
    for leg in (leg1, leg2):
        bes_shots = leg.besiktas(leg.shots)
        subs = sorted({s["minute"] for s in leg.subs if s["contestantId"] == md.BESIKTAS_ID})
        for m in subs:
            window_xg = sum(s["xg"] for s in bes_shots if m <= s["minute"] < m + 10)
            window_goals = sum(1 for s in bes_shots if s["is_goal"] and m <= s["minute"] < m + 10)
            rows.append((f"L{leg.leg_num} {m}′", window_xg, window_goals))

    ypos = np.arange(len(rows))[::-1]
    vals = [r[1] for r in rows]
    maxval = max(vals) * 1.25 if vals else 1
    bars = ax.barh(ypos, vals, color=BES_C)
    for y, (label, v, g) in zip(ypos, rows):
        txt = f"{v:.2f} xG" + (f"  ·  {g} goal{'s' if g != 1 else ''}" if g else "")
        ax.text(v + maxval * 0.02, y, txt, va="center", fontsize=9.5, color=palette["ink_primary"])
    ax.set_yticks(ypos)
    ax.set_yticklabels([r[0] for r in rows], fontsize=10, color=palette["ink_primary"])
    ax.set_xlim(0, maxval)
    ax.set_xlabel("Besiktas xG created in the 10 minutes after each substitution")

    components.header(fig, kicker="Substitutions",
                       title="Besiktas' first change in leg 2 preceded both of that night's goals",
                       dek="Besiktas' own xG in the 10 minutes following each Besiktas substitution, both legs",
                       palette=palette)
    components.footer(fig, source=md.SOURCE, palette=palette)
    save(fig, "50_substitution_impact.png")


# ---------------------------------------------------------------------------
# 51. Player workload / involvement leaderboard
# ---------------------------------------------------------------------------

def player_workload(leg1, leg2):
    fig, palette = new_fig()
    ax = fig.add_axes([0.22, 0.14, 0.68, 0.60])

    involvement = {}
    for leg in (leg1, leg2):
        for p in leg.besiktas(leg.passes):
            involvement[p["player"]] = involvement.get(p["player"], 0) + 1
        for s in leg.besiktas(leg.shots):
            involvement[s["player"]] = involvement.get(s["player"], 0) + 1
        for d in leg.besiktas(leg.defs):
            involvement[d["player"]] = involvement.get(d["player"], 0) + 1

    top = sorted(involvement.items(), key=lambda kv: -kv[1])[:10][::-1]
    ypos = np.arange(len(top))
    vals = [v for _, v in top]
    ax.barh(ypos, vals, color=BES_C)
    ax.set_yticks(ypos)
    ax.set_yticklabels([p for p, _ in top], fontsize=10.5, color=palette["ink_primary"])
    for y, v in zip(ypos, vals):
        ax.text(v + max(vals) * 0.015, y, str(v), va="center", fontsize=9.5, color=palette["ink_secondary"])
    ax.set_xlabel("Passes + shots + defensive actions, both legs combined")

    components.header(fig, kicker="Workload",
                       title="Besiktas' busiest players across the tie",
                       dek="Involvement proxy: pass attempts + shots + defensive actions, both legs combined",
                       palette=palette)
    components.footer(fig, source=md.SOURCE, palette=palette)
    save(fig, "51_player_workload.png")


# ---------------------------------------------------------------------------
# 52. Home vs away split
# ---------------------------------------------------------------------------

def home_away_split(leg1, leg2):
    fig, palette = new_fig()
    ax = fig.add_axes([0.28, 0.16, 0.62, 0.58])

    p1, p2 = leg1.besiktas(leg1.passes), leg2.besiktas(leg2.passes)
    s1, s2 = leg1.besiktas(leg1.shots), leg2.besiktas(leg2.shots)
    d1, d2 = leg1.besiktas(leg1.defs), leg2.besiktas(leg2.defs)

    metrics = [
        ("xG", sum(s["xg"] for s in s1), sum(s["xg"] for s in s2), "{:.2f}"),
        ("Shots", len(s1), len(s2), "{:.0f}"),
        ("Pass accuracy", sum(1 for p in p1 if p["completed"]) / len(p1),
         sum(1 for p in p2 if p["completed"]) / len(p2), "{:.0%}"),
        ("Touch share", leg1.touch_share(), leg2.touch_share(), "{:.0%}"),
        ("PPDA", leg1.besiktas_ppda(), leg2.besiktas_ppda(), "{:.1f}"),
        ("Progressive passes", sum(1 for p in p1 if p["progressive"]), sum(1 for p in p2 if p["progressive"]), "{:.0f}"),
        ("Defensive actions", len(d1), len(d2), "{:.0f}"),
    ]
    n = len(metrics)
    ypos = np.arange(n)[::-1]
    for y, (label, h, a, fmt) in zip(ypos, metrics):
        maxval = max(h, a, 1e-9)
        ax.barh(y + 0.18, h / maxval, height=0.32, color=BES_C)
        ax.barh(y - 0.18, a / maxval, height=0.32, color=BES_C, alpha=0.45)
        ax.text(h / maxval + 0.02, y + 0.18, fmt.format(h), va="center", fontsize=9.5, color=palette["ink_primary"])
        ax.text(a / maxval + 0.02, y - 0.18, fmt.format(a), va="center", fontsize=9.5, color=palette["ink_primary"])
    ax.set_yticks(ypos)
    ax.set_yticklabels([m[0] for m in metrics], fontsize=11, color=palette["ink_primary"])
    ax.set_xlim(0, 1.35)
    ax.set_xticks([])
    ax.grid(False)

    legend_elems = [Line2D([0], [0], marker="s", color=palette["surface"], markerfacecolor=BES_C,
                            markersize=12, label="Leg 1 (home)", linewidth=0),
                    Line2D([0], [0], marker="s", color=palette["surface"], markerfacecolor=BES_C, alpha=0.45,
                            markersize=12, label="Leg 2 (away)", linewidth=0)]
    fig.legend(handles=legend_elems, loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 0.03), fontsize=10.5, labelcolor=palette["ink_secondary"])

    components.header(fig, kicker="Home vs Away",
                       title="Besiktas were a different, more expansive team at home",
                       dek="Besiktas' own numbers, leg 1 (home) vs leg 2 (away)  ·  bars scaled to the larger of the two",
                       palette=palette)
    components.footer(fig, source=md.SOURCE, palette=palette)
    save(fig, "52_home_away_split.png")


# ---------------------------------------------------------------------------
# 53. Report card (aggregate)
# ---------------------------------------------------------------------------

def report_card(leg1, leg2):
    palette, _ = style.apply("dark")
    fig = plt.figure(figsize=FIGSIZE)
    fig.patch.set_facecolor(palette["surface"])

    fig.text(0.5, 0.90, "Besiktas  3-0  Midtjylland  (aggregate)", fontsize=18, fontweight="bold",
              color=palette["ink_primary"], family="serif", ha="center", va="center")
    fig.text(0.5, 0.855, f"{md.COMPETITION}", fontsize=10.5, color=palette["ink_secondary"],
              ha="center", va="center")

    bes_shots = leg1.besiktas(leg1.shots) + leg2.besiktas(leg2.shots)
    opp_shots = leg1.opponent(leg1.shots) + leg2.opponent(leg2.shots)
    bes_pass = leg1.besiktas(leg1.passes) + leg2.besiktas(leg2.passes)
    opp_pass = leg1.opponent(leg1.passes) + leg2.opponent(leg2.passes)
    bes_touch = leg1.besiktas(leg1.touches) + leg2.besiktas(leg2.touches)
    opp_touch = leg1.opponent(leg1.touches) + leg2.opponent(leg2.touches)
    bes_defs = leg1.besiktas(leg1.defs) + leg2.besiktas(leg2.defs)
    opp_defs = leg1.opponent(leg1.defs) + leg2.opponent(leg2.defs)
    pooled_passes = leg1.passes + leg2.passes
    pooled_pressing = leg1.pressing + leg2.pressing
    bes_ppda = md.compute_ppda(pooled_passes, pooled_pressing, md.BESIKTAS_ID, md.MIDTJYLLAND_ID)
    opp_ppda = md.compute_ppda(pooled_passes, pooled_pressing, md.MIDTJYLLAND_ID, md.BESIKTAS_ID)

    rows = [
        ("Expected goals", f"{sum(s['xg'] for s in bes_shots):.2f}", f"{sum(s['xg'] for s in opp_shots):.2f}"),
        ("Shots", str(len(bes_shots)), str(len(opp_shots))),
        ("Touch share", f"{len(bes_touch) / (len(bes_touch) + len(opp_touch)):.0%}",
         f"{len(opp_touch) / (len(bes_touch) + len(opp_touch)):.0%}"),
        ("Pass accuracy", f"{sum(1 for p in bes_pass if p['completed']) / len(bes_pass):.0%}",
         f"{sum(1 for p in opp_pass if p['completed']) / len(opp_pass):.0%}"),
        ("PPDA", f"{bes_ppda:.1f}", f"{opp_ppda:.1f}"),
        ("Tackles + interceptions",
         str(sum(1 for d in bes_defs if d["action"] in ("Tackle", "Interception"))),
         str(sum(1 for d in opp_defs if d["action"] in ("Tackle", "Interception")))),
    ]

    ax = fig.add_axes([0.14, 0.20, 0.72, 0.55])
    ax.axis("off")
    ax.text(0.0, 1.0, "Besiktas", fontsize=12.5, fontweight="bold", color=BES_C, ha="left", va="top")
    ax.text(1.0, 1.0, "Midtjylland", fontsize=12.5, fontweight="bold", color=OPP_C, ha="right", va="top")
    n = len(rows)
    for i, (label, hval, aval) in enumerate(rows):
        y = 0.85 - i * (0.85 / n)
        ax.text(0.0, y, hval, fontsize=13, fontweight="bold", color=palette["ink_primary"], ha="left", va="top")
        ax.text(0.5, y, label, fontsize=10.5, color=palette["ink_muted"], ha="center", va="top")
        ax.text(1.0, y, aval, fontsize=13, fontweight="bold", color=palette["ink_primary"], ha="right", va="top")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    components.brand_mark(fig, palette=palette, right=0.94, y=0.965)
    components.footer(fig, source=md.SOURCE, palette=palette)
    save(fig, "53_report_card.png")


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

    progressive_passes_leg(leg1, "18")
    progressive_passes_leg(leg2, "19")
    passing_directness_leg(leg1, "20")
    passing_directness_leg(leg2, "21")
    possession_thirds_both_legs(leg1, leg2)
    progression_bars_both_legs(leg1, leg2)
    recoveries_by_third_both_legs(leg1, leg2)
    crossing_map_leg(leg1, "25")
    crossing_map_leg(leg2, "26")
    long_balls_leg(leg1, "27")
    long_balls_leg(leg2, "28")
    aerial_duels_leg(leg1, "29")
    aerial_duels_leg(leg2, "30")
    zone14_leg(leg1, "31")
    zone14_leg(leg2, "32")
    origin_zone14_leg(leg1, "33")
    origin_zone14_leg(leg2, "34")
    final_third_entries_both_legs(leg1, leg2)
    opponent_touch_heatmap_both_legs(leg1, leg2)
    buildup_to_shot_both_legs(leg1, leg2)
    set_piece_analysis_both_legs(leg1, leg2)
    defensive_line_height_both_legs(leg1, leg2)
    shot_zones_heatmap_both_legs(leg1, leg2)
    passing_direction_both_legs(leg1, leg2)
    xt_flow_both_legs(leg1, leg2)
    xt_leaderboard(leg1, leg2)
    shot_assists_map_both_legs(leg1, leg2)
    key_passes_leaderboard(leg1, leg2)
    turnovers_dangerous_both_legs(leg1, leg2)
    team_radar(leg1, leg2)

    bes_shots = leg1.besiktas(leg1.shots) + leg2.besiktas(leg2.shots)
    opp_shots = leg1.opponent(leg1.shots) + leg2.opponent(leg2.shots)
    sim = md.simulate_scorelines(bes_shots, opp_shots)
    win_probability_situation(leg1, leg2, sim)
    xg_scoreline_matrix(sim)

    substitution_impact(leg1, leg2)
    player_workload(leg1, leg2)
    home_away_split(leg1, leg2)
    report_card(leg1, leg2)
    closing_summary(leg1, leg2)

    print("Done.")


if __name__ == "__main__":
    main()

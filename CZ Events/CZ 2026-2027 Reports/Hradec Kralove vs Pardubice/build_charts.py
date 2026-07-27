"""
FC Hradec Králové vs FK Pardubice - Post-Match Analysis
==========================================================
Chance Liga 2026/27, matchday 29 (2026-07-26, FINEP Arena). Hradec Králové
won 2-1 at home (Van Buren 43', Čech 55'; Boledovič 59' for Pardubice).

Follows the page set of the Slavia Praha post-match template the user
supplied, rebuilt in Marc Lamberts' Meridian house style (dark mode) --
housestyle/ package at the repo root -- instead of the template's own
navy/gold/green look, and considerably expanded: every team-vs-team page
in the template gets a same-team-both-sides page here too, plus new pages
(heatmaps, zone 14 / half-space passing, recoveries, duels, discipline,
a momentum timeline, a simple impact leaderboard) not in the source deck.

Data: Opta MA3 event feed (CZ Events/CZ 2026-2027), parsed in
match_data.py; shots scored with a small own distance+angle xG model
(no provider xG in this feed). PPDA uses the standard tackles +
interceptions + challenges + fouls-committed action set in the opponent's
own 60% of the pitch -- see match_data.compute_ppda for the Foul
outcome-convention check against this match's carded players.

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
HOME_C = CATEGORICAL_DARK[0]   # ink blue -- FC Hradec Králové
AWAY_C = CATEGORICAL_DARK[1]   # terracotta -- FK Pardubice
GOOD_C = STATUS_DARK["good"]
WARN_C = STATUS_DARK["warning"]
HOME_SHORT = "Hradec Kr."
AWAY_SHORT = "Pardubice"

BOX_Y = (13.84, 54.16)
ZONE14 = (70.0, 88.5, 27.2, 40.8)              # x0, x1, y0, y1
HALF_SPACES = [(52.5, 105.0, 13.6, 27.2), (52.5, 105.0, 40.8, 54.4)]


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
    return HOME_C if cid == md.HOME_ID else AWAY_C


def team_short(cid):
    return HOME_SHORT if cid == md.HOME_ID else AWAY_SHORT


# ---------------------------------------------------------------------------
# 01. Cover
# ---------------------------------------------------------------------------

def cover(scores):
    palette, _ = style.apply("dark")
    fig = plt.figure(figsize=FIGSIZE)
    fig.patch.set_facecolor(palette["surface"])

    fig.text(0.5, 0.62, md.HOME_NAME.upper(), fontsize=34, fontweight="bold",
              color=HOME_C, family="serif", ha="center", va="center")
    fig.text(0.5, 0.535, "vs", fontsize=16, color=palette["ink_muted"],
              family="sans-serif", ha="center", va="center")
    fig.text(0.5, 0.45, md.AWAY_NAME.upper(), fontsize=34, fontweight="bold",
              color=AWAY_C, family="serif", ha="center", va="center")

    fig.text(0.5, 0.345, f"{scores['ft']['home']} – {scores['ft']['away']}",
              fontsize=30, fontweight="bold", color=palette["ink_primary"],
              family="sans-serif", ha="center", va="center")
    fig.text(0.5, 0.29, f"(HT {scores['ht']['home']}-{scores['ht']['away']})",
              fontsize=11, color=palette["ink_muted"], family="sans-serif",
              ha="center", va="center")

    fig.text(0.5, 0.215, f"{md.COMPETITION}  ·  {md.VENUE}", fontsize=12,
              color=palette["ink_secondary"], family="sans-serif", ha="center", va="center")

    fig.text(0.5, 0.13, f"{components.MARK} POST-MATCH ANALYSIS  ·  30 PAGES", fontsize=13, fontweight="bold",
              color=palette["accent"], family="sans-serif", ha="center", va="center")

    components.brand_mark(fig, palette=palette, right=0.94, y=0.93)
    components.footer(fig, source=md.SOURCE, palette=palette)
    save(fig, "01_cover.png")


# ---------------------------------------------------------------------------
# 02. Match summary: shot map + KPI bars
# ---------------------------------------------------------------------------

def match_summary(shots, passes, touches):
    fig, palette = new_fig()
    pitch = new_pitch(palette)
    ax = fig.add_axes([0.02, 0.10, 0.55, 0.62])
    pitch.draw(ax=ax)

    for s in shots:
        color = team_color(s["contestantId"])
        x = s["x"] if s["contestantId"] == md.HOME_ID else 105 - s["x"]
        y = s["y"] if s["contestantId"] == md.HOME_ID else 68 - s["y"]
        size = 90 + s["xg"] * 900
        if s["is_goal"]:
            pitch.scatter(x, y, ax=ax, s=size, marker="o", color=color,
                          edgecolors=palette["ink_primary"], linewidth=1.4, zorder=5)
        else:
            pitch.scatter(x, y, ax=ax, s=size, marker="o", facecolors="none",
                          edgecolors=color, linewidth=1.6, alpha=0.85, zorder=4)
    ax.text(0.02, -0.06, "Hollow = shot   ● Filled = goal   Size = xG", transform=ax.transAxes,
            fontsize=8.5, color=palette["ink_muted"])

    def touch_share():
        h = sum(1 for t in touches if t["contestantId"] == md.HOME_ID)
        a = sum(1 for t in touches if t["contestantId"] == md.AWAY_ID)
        return h / (h + a)

    home_xg = sum(s["xg"] for s in shots if s["contestantId"] == md.HOME_ID)
    away_xg = sum(s["xg"] for s in shots if s["contestantId"] == md.AWAY_ID)
    home_shots = [s for s in shots if s["contestantId"] == md.HOME_ID]
    away_shots = [s for s in shots if s["contestantId"] == md.AWAY_ID]
    home_pass = [p for p in passes if p["contestantId"] == md.HOME_ID]
    away_pass = [p for p in passes if p["contestantId"] == md.AWAY_ID]

    rows = [
        ("Expected goals", f"{home_xg:.2f}", f"{away_xg:.2f}", home_xg, away_xg),
        ("Shots (on target)",
         f"{len(home_shots)} ({sum(1 for s in home_shots if s['on_target'])})",
         f"{len(away_shots)} ({sum(1 for s in away_shots if s['on_target'])})",
         len(home_shots), len(away_shots)),
        ("Big chances", str(sum(1 for s in home_shots if s["big_chance"])),
         str(sum(1 for s in away_shots if s["big_chance"])),
         sum(1 for s in home_shots if s["big_chance"]), sum(1 for s in away_shots if s["big_chance"])),
        ("Touch share", f"{touch_share():.0%}", f"{1 - touch_share():.0%}", touch_share(), 1 - touch_share()),
        ("Pass accuracy",
         f"{sum(1 for p in home_pass if p['completed']) / len(home_pass):.0%}",
         f"{sum(1 for p in away_pass if p['completed']) / len(away_pass):.0%}",
         sum(1 for p in home_pass if p["completed"]) / len(home_pass),
         sum(1 for p in away_pass if p["completed"]) / len(away_pass)),
    ]

    ax2 = fig.add_axes([0.60, 0.14, 0.36, 0.56])
    ax2.axis("off")
    n = len(rows)
    for i, (label, hval, aval, hnum, anum) in enumerate(rows):
        y = 1 - (i + 0.5) / n
        ax2.text(0.5, y + 0.075, label, ha="center", va="bottom", fontsize=11.5,
                 fontweight="bold", color=palette["ink_primary"])
        total = hnum + anum if (hnum + anum) > 0 else 1
        frac = hnum / total
        bar_y = y - 0.01
        h = 0.05
        ax2.add_patch(plt.Rectangle((0.0, bar_y), frac, h, color=HOME_C))
        ax2.add_patch(plt.Rectangle((frac, bar_y), 1 - frac, h, color=AWAY_C))
        ax2.text(0.02, bar_y + h / 2, hval, ha="left", va="center", fontsize=10.5,
                 fontweight="bold", color=palette["surface"])
        ax2.text(0.98, bar_y + h / 2, aval, ha="right", va="center", fontsize=10.5,
                 fontweight="bold", color=palette["surface"])
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)

    legend_elems = [Line2D([0], [0], marker="o", color=palette["surface"], markerfacecolor=HOME_C,
                            markersize=10, label=md.HOME_NAME, linewidth=0),
                    Line2D([0], [0], marker="o", color=palette["surface"], markerfacecolor=AWAY_C,
                            markersize=10, label=md.AWAY_NAME, linewidth=0)]
    fig.legend(handles=legend_elems, loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 0.015), fontsize=10.5, labelcolor=palette["ink_secondary"])

    components.header(fig, kicker="Match Summary",
                       title=f"{md.HOME_NAME} 2-1 {md.AWAY_NAME}",
                       dek="Shot map and headline numbers, both ends attacking right",
                       palette=palette)
    components.footer(fig, source=md.SOURCE, palette=palette)
    save(fig, "02_match_summary.png")


# ---------------------------------------------------------------------------
# 03. xG flow
# ---------------------------------------------------------------------------

def xg_flow(shots):
    fig, palette = new_fig()
    ax = fig.add_axes([0.08, 0.16, 0.78, 0.60])

    def series(cid):
        team_shots = sorted([s for s in shots if s["contestantId"] == cid], key=lambda s: s["minute"])
        mins, cum, total = [0.0], [0.0], 0.0
        for s in team_shots:
            mins.append(s["minute"]); cum.append(total)
            total += s["xg"]
            mins.append(s["minute"]); cum.append(total)
        mins.append(96); cum.append(total)
        return mins, cum, team_shots

    for cid, color, name in ((md.HOME_ID, HOME_C, HOME_SHORT), (md.AWAY_ID, AWAY_C, AWAY_SHORT)):
        mins, cum, team_shots = series(cid)
        ax.plot(mins, cum, color=color, linewidth=2.4, zorder=4)
        ax.fill_between(mins, cum, step=None, color=color, alpha=0.10, zorder=1)
        ax.annotate(f"{name}\n{cum[-1]:.2f} xG", xy=(1, cum[-1]), xycoords=("axes fraction", "data"),
                    xytext=(10, 0), textcoords="offset points", color=color, fontsize=10,
                    fontweight="bold", va="center", ha="left", annotation_clip=False)

    for cid, color in ((md.HOME_ID, HOME_C), (md.AWAY_ID, AWAY_C)):
        team_shots = sorted([s for s in shots if s["contestantId"] == cid], key=lambda s: s["minute"])
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

    home_xg = sum(s["xg"] for s in shots if s["contestantId"] == md.HOME_ID)
    away_xg = sum(s["xg"] for s in shots if s["contestantId"] == md.AWAY_ID)
    components.header(fig, kicker="xG Flow",
                       title=f"{md.HOME_NAME} out-created {md.AWAY_NAME} {home_xg:.2f} to {away_xg:.2f} xG",
                       dek=f"{md.HOME_NAME} 2-1 {md.AWAY_NAME}  ·  cumulative expected goals by minute",
                       palette=palette)
    components.footer(fig, source=md.SOURCE, palette=palette)
    save(fig, "03_xg_flow.png")


# ---------------------------------------------------------------------------
# 04. Shot quality table
# ---------------------------------------------------------------------------

def shot_quality_table(shots):
    fig, palette = new_fig()
    ax = fig.add_axes([0.05, 0.12, 0.90, 0.62])
    ax.axis("off")

    ordered = sorted(shots, key=lambda s: s["minute"])
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
                       title=f"All {len(shots)} shots of the match, ranked by kickoff time",
                       dek="Own xG model: distance + angle to goal, header penalty applied",
                       palette=palette)
    components.footer(fig, source=md.SOURCE, palette=palette)
    save(fig, "04_shot_quality_table.png")


# ---------------------------------------------------------------------------
# 05. Goal build-ups
# ---------------------------------------------------------------------------

def goal_buildups(events, directions, shots):
    fig, palette = new_fig()
    pitch = new_pitch(palette)

    goals = sorted([s for s in shots if s["is_goal"]], key=lambda s: s["minute"])
    n = len(goals)
    axes = [fig.add_axes([0.02 + i * (0.96 / n), 0.10, 0.96 / n - 0.02, 0.62]) for i in range(n)]

    for ax, g in zip(axes, goals):
        pitch.draw(ax=ax)
        color = team_color(g["contestantId"])
        team_events = [e for e in events if e["contestantId"] == g["contestantId"]
                       and e.get("x") is not None and e["typeId"] in (1, 3, 61)
                       and md.event_time(e) <= g["minute"] * 60 + 59]
        team_events.sort(key=lambda e: (e["periodId"], md.event_time(e), e["eventId"]))
        chain = team_events[-4:]
        pts = []
        for e in chain:
            x, y = md.norm_xy(e, directions)
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
                       title="How all three goals were made",
                       dek=f"{md.HOME_NAME} 2-1 {md.AWAY_NAME}  ·  possession chain leading to each goal",
                       palette=palette)
    components.footer(fig, source=md.SOURCE, palette=palette)
    save(fig, "05_goal_buildups.png")


# ---------------------------------------------------------------------------
# 06. Pass network (combined overview)
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


def pass_network_combined(passes):
    fig, palette = new_fig()
    pitch = new_pitch(palette)
    ax1 = fig.add_axes([0.02, 0.10, 0.47, 0.62])
    ax2 = fig.add_axes([0.51, 0.10, 0.47, 0.62])

    home_passes = [p for p in passes if p["contestantId"] == md.HOME_ID]
    away_passes = [p for p in passes if p["contestantId"] == md.AWAY_ID]
    _draw_pass_network(ax1, home_passes, HOME_C, palette, pitch)
    _draw_pass_network(ax2, away_passes, AWAY_C, palette, pitch)
    ax1.set_title(md.HOME_NAME, color=HOME_C, fontsize=13, fontweight="bold", family="sans-serif")
    ax2.set_title(md.AWAY_NAME, color=AWAY_C, fontsize=13, fontweight="bold", family="sans-serif")

    fig.text(0.5, 0.09, "Node position = average completed-pass location (≥ 8 passes)  ·  "
                         "Node size = passes played  ·  Line width = pass combinations (≥ 2)",
              ha="center", fontsize=9, color=palette["ink_muted"])

    components.header(fig, kicker="Pass Network",
                       title="Hradec built through the middle; Pardubice leant on their flanks",
                       dek="Average completed-pass position, full match, both attacking right",
                       palette=palette)
    components.footer(fig, source=md.SOURCE, palette=palette)
    save(fig, "06_pass_network.png")


# ---------------------------------------------------------------------------
# 07-08. Pass network -- one full page per team, with a stats table
# ---------------------------------------------------------------------------

def pass_network_team_page(passes, cid, page_num):
    fig, palette = new_fig()
    pitch = new_pitch(palette)
    color = team_color(cid)
    name = md.team_name(cid)
    team_passes = [p for p in passes if p["contestantId"] == cid]

    ax = fig.add_axes([0.02, 0.10, 0.54, 0.62])
    avg_pos, combos = _draw_pass_network(ax, team_passes, color, palette, pitch, min_passes=6,
                                          node_scale=1.15, label_size=8.6)

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
    save(fig, f"{page_num}_pass_network_{'home' if cid == md.HOME_ID else 'away'}.png")


# ---------------------------------------------------------------------------
# 09-10. Progressive passes -- one page per team
# ---------------------------------------------------------------------------

def progressive_passes_team_page(passes, cid, page_num):
    fig, palette = new_fig()
    pitch = new_pitch(palette)
    color = team_color(cid)
    name = md.team_name(cid)
    team_passes = [p for p in passes if p["contestantId"] == cid]
    prog = [p for p in team_passes if p["progressive"]]

    ax = fig.add_axes([0.02, 0.10, 0.96, 0.62])
    pitch.draw(ax=ax)
    for p in prog:
        is_box = p["box_entry"]
        pitch.arrows(p["x"], p["y"], p["end_x"], p["end_y"], ax=ax,
                    color=GOOD_C if is_box else color, alpha=0.9 if is_box else 0.55,
                    width=2.4 if is_box else 1.4, headwidth=6, headlength=6,
                    zorder=4 if is_box else 3)

    legend_elems = [Line2D([0], [0], color=color, lw=2.0, label="Progressive pass"),
                    Line2D([0], [0], color=GOOD_C, lw=2.4, label="...into the box")]
    fig.legend(handles=legend_elems, loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 0.06), fontsize=10.5, labelcolor=palette["ink_secondary"])

    n_box = sum(1 for p in prog if p["box_entry"])
    components.header(fig, kicker="Progression",
                       title=f"{name}: {len(prog)} progressive passes, {n_box} of them straight into the box",
                       dek="Progressive pass = completed pass cutting ≥25% off the distance to goal, attacking right",
                       palette=palette)
    components.footer(fig, source=md.SOURCE, palette=palette)
    save(fig, f"{page_num}_progressive_passes_{'home' if cid == md.HOME_ID else 'away'}.png")


# ---------------------------------------------------------------------------
# 11-12. Passing volume & directness scatter -- one page per team
# ---------------------------------------------------------------------------

def passing_directness_team_page(passes, cid, page_num):
    fig, palette = new_fig()
    color = team_color(cid)
    name = md.team_name(cid)
    ax = fig.add_axes([0.10, 0.16, 0.82, 0.58])

    by_player = {}
    for p in passes:
        if p["contestantId"] != cid or not p["completed"] or p["end_x"] is None:
            continue
        d = by_player.setdefault(p["player"], {"gain": 0.0, "att": 0, "comp": 0})
        d["gain"] += max(0.0, p["x"] - p["end_x"]) * -1 + max(0.0, p["end_x"] - p["x"])
    for p in passes:
        if p["contestantId"] != cid:
            continue
        d = by_player.setdefault(p["player"], {"gain": 0.0, "att": 0, "comp": 0})
        d["att"] += 1
        d["comp"] += int(p["completed"])

    items = [(pl, d) for pl, d in by_player.items() if d["att"] >= 5]
    xs = [d["gain"] for _, d in items]
    ys = [d["comp"] / d["att"] for _, d in items]
    sizes = [40 + d["att"] * 6 for _, d in items]
    ax.scatter(xs, ys, s=sizes, color=color, alpha=0.85, edgecolors=palette["surface"], linewidth=0.8, zorder=3)
    for (pl, d), x, y in zip(items, xs, ys):
        ax.annotate(pl.split(" ")[-1], xy=(x, y), xytext=(6, 4), textcoords="offset points",
                    fontsize=8.5, color=palette["ink_secondary"])
    ax.axhline(np.mean(ys) if ys else 0, color=palette["axis"], linewidth=0.8, linestyle="--")
    ax.set_xlabel("Net metres gained by completed passes (forward - backward)")
    ax.set_ylabel("Pass completion %")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))

    components.header(fig, kicker="Passing Profile",
                       title=f"{name}: who progressed the ball, and how safely",
                       dek="Players with ≥ 5 pass attempts  ·  bubble size = passes attempted",
                       palette=palette)
    components.footer(fig, source=md.SOURCE, palette=palette)
    save(fig, f"{page_num}_passing_directness_{'home' if cid == md.HOME_ID else 'away'}.png")


# ---------------------------------------------------------------------------
# 13-14. Touch heatmap -- one page per team
# ---------------------------------------------------------------------------

def touch_heatmap_team_page(touches, cid, page_num):
    fig, palette = new_fig()
    pitch = new_pitch(palette)
    color = team_color(cid)
    name = md.team_name(cid)
    ax = fig.add_axes([0.06, 0.10, 0.88, 0.62])
    pitch.draw(ax=ax)

    xs = [t["x"] for t in touches if t["contestantId"] == cid]
    ys = [t["y"] for t in touches if t["contestantId"] == cid]
    stats = pitch.bin_statistic(xs, ys, statistic="count", bins=(9, 6))
    cmap = "Blues" if cid == md.HOME_ID else "Oranges"
    pitch.heatmap(stats, ax=ax, cmap=cmap, edgecolors=palette["surface"], alpha=0.92, zorder=1)

    components.header(fig, kicker="Territory",
                       title=f"{name}: where the team spent its {len(xs)} touches",
                       dek="Touch density by pitch zone, attacking right",
                       palette=palette)
    components.footer(fig, source=md.SOURCE, palette=palette)
    save(fig, f"{page_num}_touch_heatmap_{'home' if cid == md.HOME_ID else 'away'}.png")


# ---------------------------------------------------------------------------
# 15. Field tilt
# ---------------------------------------------------------------------------

def field_tilt(touches, shots):
    fig, palette = new_fig()
    ax = fig.add_axes([0.16, 0.16, 0.78, 0.58])

    bucket = 5
    max_min = 95
    edges = list(range(0, max_min + bucket, bucket))
    tilt, centers = [], []
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        h = sum(1 for t in touches if lo <= t["minute"] < hi and t["contestantId"] == md.HOME_ID and t["x"] >= 70)
        a = sum(1 for t in touches if lo <= t["minute"] < hi and t["contestantId"] == md.AWAY_ID and t["x"] >= 70)
        total = h + a
        tilt.append((h / total - 0.5) * 100 if total else 0.0)
        centers.append((lo + hi) / 2)

    tilt = np.array(tilt)
    centers = np.array(centers)
    ax.fill_between(centers, tilt, 0, where=(tilt >= 0), color=HOME_C, alpha=0.75, step="mid")
    ax.fill_between(centers, tilt, 0, where=(tilt < 0), color=AWAY_C, alpha=0.75, step="mid")
    ax.axhline(0, color=palette["axis"], linewidth=1.0)
    ax.axvline(45, color=palette["axis"], linewidth=0.8, linestyle=":")

    for s in shots:
        if s["is_goal"]:
            y = 46 if s["contestantId"] == md.HOME_ID else -46
            ax.scatter([s["minute"]], [y], marker="*", s=200, color=palette["ink_primary"],
                       edgecolors=team_color(s["contestantId"]), linewidth=1.4, zorder=6)

    ax.set_ylim(-55, 55)
    ax.set_xlim(0, max_min)
    ax.set_xlabel("Minute")
    ax.set_ylabel("Field tilt (final-third touch share)")
    ax.set_yticks([-50, -25, 0, 25, 50])
    ax.set_yticklabels([f"{AWAY_SHORT} 100%", "75%", "Even", "75%", f"{HOME_SHORT} 100%"], fontsize=9.5)

    h_touches = sum(1 for t in touches if t["contestantId"] == md.HOME_ID and t["x"] >= 70)
    a_touches = sum(1 for t in touches if t["contestantId"] == md.AWAY_ID and t["x"] >= 70)
    overall = h_touches / (h_touches + a_touches)
    components.header(fig, kicker="Field Tilt",
                       title=f"{md.HOME_NAME} controlled the final third, {overall:.0%} of touches to {1 - overall:.0%}",
                       dek="Share of final-third touches, 5-minute buckets  ·  ★ marks a goal",
                       palette=palette)
    components.footer(fig, source=md.SOURCE, palette=palette)
    save(fig, "15_field_tilt.png")


# ---------------------------------------------------------------------------
# 16. Possession by thirds
# ---------------------------------------------------------------------------

def possession_thirds(touches):
    fig, palette = new_fig()
    ax = fig.add_axes([0.16, 0.24, 0.68, 0.40])

    zone_colors = [CATEGORICAL_DARK[0], CATEGORICAL_DARK[3], CATEGORICAL_DARK[1]]
    zone_labels = ["Defensive", "Middle", "Attacking"]

    def thirds(cid):
        t = [x for x in touches if x["contestantId"] == cid]
        d = sum(1 for x in t if x["x"] < 35)
        m = sum(1 for x in t if 35 <= x["x"] < 70)
        a = sum(1 for x in t if x["x"] >= 70)
        total = d + m + a
        return [d / total, m / total, a / total], total

    for i, (cid, name, color) in enumerate(((md.HOME_ID, md.HOME_NAME, HOME_C), (md.AWAY_ID, md.AWAY_NAME, AWAY_C))):
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
                       title="Hradec spent far more of the match in the final third",
                       dek="Distribution of touches across pitch thirds, both teams' own attacking direction",
                       palette=palette)
    components.footer(fig, source=md.SOURCE, palette=palette)
    save(fig, "16_possession_thirds.png")


# ---------------------------------------------------------------------------
# 17. Progression comparison bars
# ---------------------------------------------------------------------------

def progression_bars(passes):
    fig, palette = new_fig()
    ax = fig.add_axes([0.24, 0.16, 0.66, 0.56])

    home_passes = [p for p in passes if p["contestantId"] == md.HOME_ID]
    away_passes = [p for p in passes if p["contestantId"] == md.AWAY_ID]

    metrics = [
        ("Progressive passes", sum(1 for p in home_passes if p["progressive"]),
         sum(1 for p in away_passes if p["progressive"])),
        ("Final-third entries", sum(1 for p in home_passes if p["final_third_entry"]),
         sum(1 for p in away_passes if p["final_third_entry"])),
        ("Passes into the box", sum(1 for p in home_passes if p["box_entry"]),
         sum(1 for p in away_passes if p["box_entry"])),
        ("Completed crosses", sum(1 for p in home_passes if p["is_cross"] and p["completed"]),
         sum(1 for p in away_passes if p["is_cross"] and p["completed"])),
    ]

    n = len(metrics)
    ypos = np.arange(n)[::-1]
    maxval = max(max(h, a) for _, h, a in metrics) * 1.15
    for y, (label, h, a) in zip(ypos, metrics):
        ax.barh(y + 0.18, h, height=0.32, color=HOME_C)
        ax.barh(y - 0.18, a, height=0.32, color=AWAY_C)
        ax.text(h + maxval * 0.015, y + 0.18, str(h), va="center", fontsize=10, color=palette["ink_primary"])
        ax.text(a + maxval * 0.015, y - 0.18, str(a), va="center", fontsize=10, color=palette["ink_primary"])
    ax.set_yticks(ypos)
    ax.set_yticklabels([m[0] for m in metrics], fontsize=11.5, color=palette["ink_primary"])
    ax.set_xlim(0, maxval)
    ax.set_xlabel("Count")
    ax.grid(axis="x")
    ax.set_axisbelow(True)

    legend_elems = [Line2D([0], [0], marker="s", color=palette["surface"], markerfacecolor=HOME_C,
                            markersize=12, label=md.HOME_NAME, linewidth=0),
                    Line2D([0], [0], marker="s", color=palette["surface"], markerfacecolor=AWAY_C,
                            markersize=12, label=md.AWAY_NAME, linewidth=0)]
    fig.legend(handles=legend_elems, loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 0.02), fontsize=10.5, labelcolor=palette["ink_secondary"])

    components.header(fig, kicker="Progression",
                       title=f"{md.HOME_NAME} progressed the ball far more often than {md.AWAY_NAME}",
                       dek="Progressive pass = completed pass cutting ≥25% off the distance to goal",
                       palette=palette)
    components.footer(fig, source=md.SOURCE, palette=palette)
    save(fig, "17_progression.png")


# ---------------------------------------------------------------------------
# 18. Ball recoveries by third
# ---------------------------------------------------------------------------

def recoveries_by_third(recoveries):
    fig, palette = new_fig()
    ax = fig.add_axes([0.24, 0.16, 0.66, 0.56])

    zone_colors = [CATEGORICAL_DARK[0], CATEGORICAL_DARK[3], CATEGORICAL_DARK[1]]
    zone_labels = ["Defensive third", "Middle third", "Attacking third"]

    def by_third(cid):
        r = [x for x in recoveries if x["contestantId"] == cid]
        return [sum(1 for x in r if x["x"] < 35), sum(1 for x in r if 35 <= x["x"] < 70),
                sum(1 for x in r if x["x"] >= 70)]

    home_counts = by_third(md.HOME_ID)
    away_counts = by_third(md.AWAY_ID)
    n = len(zone_labels)
    ypos = np.arange(n)[::-1]
    maxval = max(home_counts + away_counts) * 1.2
    for y, label, h, a in zip(ypos, zone_labels, home_counts, away_counts):
        ax.barh(y + 0.18, h, height=0.32, color=HOME_C)
        ax.barh(y - 0.18, a, height=0.32, color=AWAY_C)
        ax.text(h + maxval * 0.015, y + 0.18, str(h), va="center", fontsize=10, color=palette["ink_primary"])
        ax.text(a + maxval * 0.015, y - 0.18, str(a), va="center", fontsize=10, color=palette["ink_primary"])
    ax.set_yticks(ypos)
    ax.set_yticklabels(zone_labels, fontsize=11.5, color=palette["ink_primary"])
    ax.set_xlim(0, maxval)
    ax.set_xlabel("Ball recoveries")
    ax.grid(axis="x")
    ax.set_axisbelow(True)

    legend_elems = [Line2D([0], [0], marker="s", color=palette["surface"], markerfacecolor=HOME_C,
                            markersize=12, label=md.HOME_NAME, linewidth=0),
                    Line2D([0], [0], marker="s", color=palette["surface"], markerfacecolor=AWAY_C,
                            markersize=12, label=md.AWAY_NAME, linewidth=0)]
    fig.legend(handles=legend_elems, loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 0.02), fontsize=10.5, labelcolor=palette["ink_secondary"])

    components.header(fig, kicker="Recoveries",
                       title="Where each side won the ball back",
                       dek="Ball recoveries by pitch third, each team's own attacking direction",
                       palette=palette)
    components.footer(fig, source=md.SOURCE, palette=palette)
    save(fig, "18_recoveries.png")


# ---------------------------------------------------------------------------
# 19. Defensive actions map (combined)
# ---------------------------------------------------------------------------

def defensive_actions(defs):
    fig, palette = new_fig()
    pitch = new_pitch(palette)
    ax1 = fig.add_axes([0.02, 0.10, 0.47, 0.62])
    ax2 = fig.add_axes([0.51, 0.10, 0.47, 0.62])
    pitch.draw(ax=ax1)
    pitch.draw(ax=ax2)

    markers = {"Tackle": "o", "Interception": "D", "Clearance": "s"}
    action_colors = {"Tackle": CATEGORICAL_DARK[2], "Interception": CATEGORICAL_DARK[3],
                      "Clearance": palette["ink_muted"]}

    for ax, cid in ((ax1, md.HOME_ID), (ax2, md.AWAY_ID)):
        team_defs = [d for d in defs if d["contestantId"] == cid]
        for action, marker in markers.items():
            pts = [d for d in team_defs if d["action"] == action]
            if not pts:
                continue
            # p["x"]/p["y"] are already normalized (norm_xy) so this team's
            # own goal sits at x=0 -- plot as-is, own goal left, attacking
            # right, same convention as every other pitch page in this deck.
            xs = [p["x"] for p in pts]
            ys = [p["y"] for p in pts]
            pitch.scatter(xs, ys, ax=ax, s=80, marker=marker, color=action_colors[action],
                          edgecolors=palette["surface"], linewidth=0.6, alpha=0.9, zorder=4)
        counts = {a: sum(1 for d in team_defs if d["action"] == a) for a in markers}
        title = f"{md.team_name(cid)}\nTkl {counts['Tackle']}  ·  Int {counts['Interception']}  ·  Clr {counts['Clearance']}"
        ax.set_title(title, color=team_color(cid), fontsize=12, fontweight="bold", family="sans-serif")

    legend_elems = [Line2D([0], [0], marker=markers[a], color=palette["surface"], markerfacecolor=action_colors[a],
                            markersize=10, label=a, linewidth=0) for a in markers]
    fig.legend(handles=legend_elems, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 0.02), fontsize=10.5, labelcolor=palette["ink_secondary"])

    components.header(fig, kicker="Defending",
                       title="Both sides defended at similar volume; Pardubice cleared more under pressure",
                       dek="Tackles, interceptions and clearances, own goal on the left, attacking right",
                       palette=palette)
    components.footer(fig, source=md.SOURCE, palette=palette)
    save(fig, "19_defensive_actions.png")


# ---------------------------------------------------------------------------
# 20. PPDA (fixed formula)
# ---------------------------------------------------------------------------

def ppda(passes, pressing_actions):
    fig, palette = new_fig()
    ax1 = fig.add_axes([0.08, 0.16, 0.40, 0.58])
    ax2 = fig.add_axes([0.56, 0.16, 0.40, 0.58])

    buckets = [(0, 15), (15, 30), (30, 45), (45, 60), (60, 75), (75, 96)]
    labels = ["0-15", "15-30", "30-45", "45-60", "60-75", "75-90+"]

    def bucketed(cid, opp_id):
        return [md.compute_ppda(passes, pressing_actions, cid, opp_id, lo, hi) for lo, hi in buckets]

    home_vals = bucketed(md.HOME_ID, md.AWAY_ID)
    away_vals = bucketed(md.AWAY_ID, md.HOME_ID)
    finite = [v for v in home_vals + away_vals if not math.isnan(v)]
    shared_max = max(finite) * 1.15 if finite else 1.0

    for ax, vals, color, name, cid, opp in ((ax1, home_vals, HOME_C, md.HOME_NAME, md.HOME_ID, md.AWAY_ID),
                                             (ax2, away_vals, AWAY_C, md.AWAY_NAME, md.AWAY_ID, md.HOME_ID)):
        xs = np.arange(len(labels))
        clean = [v if not math.isnan(v) else 0 for v in vals]
        ax.bar(xs, clean, color=color)
        for x, v in zip(xs, vals):
            if not math.isnan(v):
                ax.text(x, v + shared_max * 0.02, f"{v:.1f}", ha="center", fontsize=9.5,
                        color=palette["ink_primary"], fontweight="bold")
        overall = md.compute_ppda(passes, pressing_actions, cid, opp)
        ax.axhline(overall, color=palette["ink_muted"], linestyle="--", linewidth=1.0)
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, fontsize=8.5)
        ax.set_ylim(0, shared_max)
        ax.set_title(f"{name}\nOverall PPDA: {overall:.1f}", color=color, fontsize=11.5,
                     fontweight="bold", family="sans-serif")
        ax.set_ylabel("PPDA")

    components.header(fig, kicker="Pressing",
                       title="Pardubice pressed harder in the moments that mattered, despite less territory",
                       dek="Passes per defensive action (tackle+interception+challenge+foul committed) in the "
                           "opponent's own 60%  ·  lower = more intense press",
                       palette=palette)
    components.footer(fig, source=md.SOURCE, palette=palette)
    save(fig, "20_ppda.png")


# ---------------------------------------------------------------------------
# 21. Duels summary
# ---------------------------------------------------------------------------

def duels_summary(duels):
    fig, palette = new_fig()
    ax = fig.add_axes([0.24, 0.16, 0.66, 0.56])

    kinds = ["Tackle", "Aerial", "Challenge"]
    n = len(kinds)
    ypos = np.arange(n)[::-1]
    maxval = 0
    rows = []
    for kind in kinds:
        h = [d for d in duels if d["contestantId"] == md.HOME_ID and d["action"] == kind]
        a = [d for d in duels if d["contestantId"] == md.AWAY_ID and d["action"] == kind]
        h_won = sum(1 for d in h if d["success"])
        a_won = sum(1 for d in a if d["success"])
        rows.append((kind, len(h), h_won, len(a), a_won))
        maxval = max(maxval, len(h), len(a))
    maxval *= 1.2

    for y, (kind, h_n, h_won, a_n, a_won) in zip(ypos, rows):
        h_rate = h_won / h_n if h_n else 0
        a_rate = a_won / a_n if a_n else 0
        ax.barh(y + 0.18, h_n, height=0.32, color=palette["axis"])
        ax.barh(y + 0.18, h_won, height=0.32, color=HOME_C)
        ax.barh(y - 0.18, a_n, height=0.32, color=palette["axis"])
        ax.barh(y - 0.18, a_won, height=0.32, color=AWAY_C)
        ax.text(h_n + maxval * 0.015, y + 0.18, f"{h_won}/{h_n} ({h_rate:.0%})", va="center",
                fontsize=9.5, color=palette["ink_primary"])
        ax.text(a_n + maxval * 0.015, y - 0.18, f"{a_won}/{a_n} ({a_rate:.0%})", va="center",
                fontsize=9.5, color=palette["ink_primary"])
    ax.set_yticks(ypos)
    ax.set_yticklabels([f"{k} duels" for k in kinds], fontsize=11.5, color=palette["ink_primary"])
    ax.set_xlim(0, maxval)
    ax.set_xlabel("Contested (solid = won)")
    ax.grid(axis="x")
    ax.set_axisbelow(True)

    legend_elems = [Line2D([0], [0], marker="s", color=palette["surface"], markerfacecolor=HOME_C,
                            markersize=12, label=f"{md.HOME_NAME} won", linewidth=0),
                    Line2D([0], [0], marker="s", color=palette["surface"], markerfacecolor=AWAY_C,
                            markersize=12, label=f"{md.AWAY_NAME} won", linewidth=0)]
    fig.legend(handles=legend_elems, loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 0.02), fontsize=10.5, labelcolor=palette["ink_secondary"])

    components.header(fig, kicker="Duels",
                       title="Tackle, aerial and loose-ball duel win rates",
                       dek="Bar length = duels contested, solid fill = duels won",
                       palette=palette)
    components.footer(fig, source=md.SOURCE, palette=palette)
    save(fig, "21_duels.png")


# ---------------------------------------------------------------------------
# 22-23. Crossing map -- one page per team
# ---------------------------------------------------------------------------

def crossing_map_team_page(passes, cid, page_num):
    fig, palette = new_fig()
    pitch = new_pitch(palette)
    color = team_color(cid)
    name = md.team_name(cid)
    ax = fig.add_axes([0.05, 0.10, 0.90, 0.62])
    pitch.draw(ax=ax)

    crosses = [p for p in passes if p["contestantId"] == cid and p["is_cross"] and p["end_x"] is not None]
    completed = [p for p in crosses if p["completed"]]
    incomplete = [p for p in crosses if not p["completed"]]
    for p in incomplete:
        pitch.arrows(p["x"], p["y"], p["end_x"], p["end_y"], ax=ax, color=palette["axis"],
                    alpha=0.55, width=1.4, headwidth=5, headlength=5, zorder=2)
    for p in completed:
        pitch.arrows(p["x"], p["y"], p["end_x"], p["end_y"], ax=ax, color=color,
                    alpha=0.9, width=2.2, headwidth=6, headlength=6, zorder=3)

    acc = len(completed) / len(crosses) if crosses else 0
    legend_elems = [Line2D([0], [0], color=color, lw=2.2, label="Completed"),
                    Line2D([0], [0], color=palette["axis"], lw=1.4, label="Incomplete")]
    fig.legend(handles=legend_elems, loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 0.06), fontsize=10.5, labelcolor=palette["ink_secondary"])

    components.header(fig, kicker="Crossing",
                       title=f"{name}: {len(completed)} of {len(crosses)} crosses found a teammate ({acc:.0%})",
                       dek="All open-play and set-piece crosses, attacking right",
                       palette=palette)
    components.footer(fig, source=md.SOURCE, palette=palette)
    save(fig, f"{page_num}_crossing_map_{'home' if cid == md.HOME_ID else 'away'}.png")


# ---------------------------------------------------------------------------
# 24. Final third entries (combined)
# ---------------------------------------------------------------------------

def final_third_entries(passes):
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

    for ax, cid in ((ax1, md.HOME_ID), (ax2, md.AWAY_ID)):
        entries = [p for p in passes if p["contestantId"] == cid and p["final_third_entry"]]
        counts = {"Left": 0, "Central": 0, "Right": 0}
        for p in entries:
            zone = zone_of(p["y"])
            counts[zone] += 1
            pitch.arrows(p["x"], p["y"], p["end_x"], p["end_y"], ax=ax, color=zone_colors[zone],
                        width=1.6, headwidth=5, headlength=5, alpha=0.85, zorder=3)
        title = (f"{md.team_name(cid)}  ({len(entries)} entries)\n"
                 f"L: {counts['Left']}  ·  C: {counts['Central']}  ·  R: {counts['Right']}")
        ax.set_title(title, color=team_color(cid), fontsize=11.5, fontweight="bold", family="sans-serif")

    legend_elems = [Line2D([0], [0], color=zone_colors[z], lw=2.4, label=z) for z in ("Left", "Central", "Right")]
    fig.legend(handles=legend_elems, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 0.02), fontsize=10.5, labelcolor=palette["ink_secondary"])

    components.header(fig, kicker="Final Third",
                       title="Both teams found their way in mostly down the flanks",
                       dek="Completed passes into the final third, by origin lane",
                       palette=palette)
    components.footer(fig, source=md.SOURCE, palette=palette)
    save(fig, "24_final_third_entries.png")


# ---------------------------------------------------------------------------
# 25-26. Zone 14 & half-space passes -- one page per team
# ---------------------------------------------------------------------------

def zone14_team_page(passes, cid, page_num):
    fig, palette = new_fig()
    pitch = new_pitch(palette)
    color = team_color(cid)
    name = md.team_name(cid)
    ax = fig.add_axes([0.05, 0.10, 0.90, 0.62])
    pitch.draw(ax=ax)

    z14x0, z14x1, z14y0, z14y1 = ZONE14
    pitch.polygon([np.array([(z14x0, z14y0), (z14x1, z14y0), (z14x1, z14y1), (z14x0, z14y1)])],
                  ax=ax, color=CATEGORICAL_DARK[6], alpha=0.18, zorder=1)
    for hx0, hx1, hy0, hy1 in HALF_SPACES:
        pitch.polygon([np.array([(hx0, hy0), (hx1, hy0), (hx1, hy1), (hx0, hy1)])],
                      ax=ax, color=color, alpha=0.10, zorder=1)

    team_passes = [p for p in passes if p["contestantId"] == cid and p["completed"] and p["end_x"] is not None]
    z14 = [p for p in team_passes if z14x0 <= p["end_x"] <= z14x1 and z14y0 <= p["end_y"] <= z14y1]
    hs = [p for p in team_passes
          if any(hx0 <= p["end_x"] <= hx1 and hy0 <= p["end_y"] <= hy1 for hx0, hx1, hy0, hy1 in HALF_SPACES)
          and p not in z14]

    for p in hs:
        pitch.arrows(p["x"], p["y"], p["end_x"], p["end_y"], ax=ax, color=color, alpha=0.55,
                    width=1.4, headwidth=5, headlength=5, zorder=3)
    for p in z14:
        pitch.arrows(p["x"], p["y"], p["end_x"], p["end_y"], ax=ax, color=palette["ink_primary"], alpha=0.9,
                    width=2.0, headwidth=6, headlength=6, zorder=4)

    legend_elems = [Line2D([0], [0], color=palette["ink_primary"], lw=2.0, label=f"Into Zone 14 ({len(z14)})"),
                    Line2D([0], [0], color=color, lw=1.6, label=f"Into a half-space ({len(hs)})")]
    fig.legend(handles=legend_elems, loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 0.06), fontsize=10.5, labelcolor=palette["ink_secondary"])

    components.header(fig, kicker="Central Access",
                       title=f"{name}: how often did they find Zone 14 and the half-spaces?",
                       dek="Completed passes ending in the central Zone 14 box or either half-space channel",
                       palette=palette)
    components.footer(fig, source=md.SOURCE, palette=palette)
    save(fig, f"{page_num}_zone14_halfspace_{'home' if cid == md.HOME_ID else 'away'}.png")


# ---------------------------------------------------------------------------
# 27. Build-up to shot
# ---------------------------------------------------------------------------

def _possession_chain_lengths(events, shots):
    """For each shot, the number of consecutive completed passes by the
    shooting team immediately before it (reset whenever the ball changes
    team). Same 'possession sequence' idea as build_goalkick_shot_model.py's
    action-chain walk, just counted backwards from a shot instead of
    forwards from a goal kick. Shot typeIds must be in this filter too --
    they carry x/y and a contestantId like any other ball event, so leaving
    them out (as an earlier version of this function did) means the "is
    this event a shot" check below never fires."""
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
        if e["typeId"] in md.SHOT_TYPES:
            lengths.append((cid, run_len))
    return lengths


def buildup_to_shot(events, shots):
    fig, palette = new_fig()
    ax1 = fig.add_axes([0.08, 0.16, 0.40, 0.58])
    ax2 = fig.add_axes([0.56, 0.16, 0.40, 0.58])

    lengths = _possession_chain_lengths(events, shots)
    bins = [(0, 3, "Direct (0-3)"), (4, 6, "Build-up (4-6)"), (7, 99, "Elaborate (7+)")]

    for ax, cid, color, name in ((ax1, md.HOME_ID, HOME_C, md.HOME_NAME), (ax2, md.AWAY_ID, AWAY_C, md.AWAY_NAME)):
        team_lengths = [n for c, n in lengths if c == cid]
        counts = [sum(1 for n in team_lengths if lo <= n <= hi) for lo, hi, _ in bins]
        xs = np.arange(len(bins))
        ax.bar(xs, counts, color=color)
        for x, c in zip(xs, counts):
            ax.text(x, c + max(counts, default=0) * 0.03 + 0.05, str(c), ha="center", fontsize=10.5,
                    color=palette["ink_primary"], fontweight="bold")
        ax.set_xticks(xs)
        ax.set_xticklabels([b[2] for b in bins], fontsize=9)
        ax.set_title(f"{name}\n{len(team_lengths)} shot sequences", color=color, fontsize=11.5,
                     fontweight="bold", family="sans-serif")
        ax.set_ylabel("Sequences")

    components.header(fig, kicker="Build-Up",
                       title="Both sides mostly shot from short, direct sequences",
                       dek="Possession sequences ending in a shot, grouped by completed-pass count",
                       palette=palette)
    components.footer(fig, source=md.SOURCE, palette=palette)
    save(fig, "27_buildup_to_shot.png")


# ---------------------------------------------------------------------------
# 28. Discipline
# ---------------------------------------------------------------------------

def discipline(pressing_actions, cards):
    fig, palette = new_fig()
    ax = fig.add_axes([0.24, 0.36, 0.66, 0.36])

    home_fouls = sum(1 for d in pressing_actions if d["contestantId"] == md.HOME_ID and d["action"] == "Foul")
    away_fouls = sum(1 for d in pressing_actions if d["contestantId"] == md.AWAY_ID and d["action"] == "Foul")
    home_yellow = sum(1 for c in cards if c["contestantId"] == md.HOME_ID and c["kind"] == "Yellow")
    away_yellow = sum(1 for c in cards if c["contestantId"] == md.AWAY_ID and c["kind"] == "Yellow")
    home_red = sum(1 for c in cards if c["contestantId"] == md.HOME_ID and c["kind"] in ("Red", "2nd Yellow"))
    away_red = sum(1 for c in cards if c["contestantId"] == md.AWAY_ID and c["kind"] in ("Red", "2nd Yellow"))

    metrics = [("Fouls committed", home_fouls, away_fouls), ("Yellow cards", home_yellow, away_yellow),
               ("Red cards", home_red, away_red)]
    n = len(metrics)
    ypos = np.arange(n)[::-1]
    maxval = max(max(h, a) for _, h, a in metrics) * 1.3 or 1
    for y, (label, h, a) in zip(ypos, metrics):
        ax.barh(y + 0.18, h, height=0.32, color=HOME_C)
        ax.barh(y - 0.18, a, height=0.32, color=AWAY_C)
        ax.text(h + maxval * 0.02, y + 0.18, str(h), va="center", fontsize=10, color=palette["ink_primary"])
        ax.text(a + maxval * 0.02, y - 0.18, str(a), va="center", fontsize=10, color=palette["ink_primary"])
    ax.set_yticks(ypos)
    ax.set_yticklabels([m[0] for m in metrics], fontsize=11.5, color=palette["ink_primary"])
    ax.set_xlim(0, maxval)
    ax.grid(axis="x")
    ax.set_axisbelow(True)

    legend_elems = [Line2D([0], [0], marker="s", color=palette["surface"], markerfacecolor=HOME_C,
                            markersize=12, label=md.HOME_NAME, linewidth=0),
                    Line2D([0], [0], marker="s", color=palette["surface"], markerfacecolor=AWAY_C,
                            markersize=12, label=md.AWAY_NAME, linewidth=0)]
    fig.legend(handles=legend_elems, loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 0.20), fontsize=10.5, labelcolor=palette["ink_secondary"])

    sorted_cards = sorted(cards, key=lambda c: c["minute"])
    rows_per_col = max(1, math.ceil(len(sorted_cards) / 2))
    for i, c in enumerate(sorted_cards):
        col, row = divmod(i, rows_per_col)
        x = 0.30 + col * 0.42
        y = 0.13 - row * 0.032
        fig.text(x, y, f"{c['minute']}' {c['player']} ({c['kind']})", ha="left", va="top",
                  fontsize=8.8, color=team_color(c["contestantId"]))

    components.header(fig, kicker="Discipline",
                       title="Fouls and cards",
                       dek="Foul committed = verified against this match's carded players (see match_data.py)",
                       palette=palette)
    components.footer(fig, source=md.SOURCE, palette=palette)
    save(fig, "28_discipline.png")


# ---------------------------------------------------------------------------
# 29. Momentum timeline
# ---------------------------------------------------------------------------

def momentum_timeline(shots, cards, subs):
    fig, palette = new_fig()
    ax_rect = [0.16, 0.30, 0.78, 0.32]
    ax = fig.add_axes(ax_rect)

    ax.axhline(1, color=HOME_C, linewidth=6, alpha=0.25, solid_capstyle="round")
    ax.axhline(0, color=AWAY_C, linewidth=6, alpha=0.25, solid_capstyle="round")
    ylim = (-0.6, 1.6)
    for data_y, name, color in ((1, HOME_SHORT, HOME_C), (0, AWAY_SHORT, AWAY_C)):
        fig_y = ax_rect[1] + ax_rect[3] * (data_y - ylim[0]) / (ylim[1] - ylim[0])
        fig.text(ax_rect[0] - 0.01, fig_y, name, ha="right", va="center", fontsize=11,
                  fontweight="bold", color=color)

    def row(cid):
        return 1 if cid == md.HOME_ID else 0

    for s in shots:
        if s["is_goal"]:
            ax.scatter([s["minute"]], [row(s["contestantId"])], marker="*", s=320,
                       color=palette["ink_primary"], edgecolors=team_color(s["contestantId"]),
                       linewidth=1.8, zorder=5)
            ax.annotate(f"{s['player']} {s['minute']}'", xy=(s["minute"], row(s["contestantId"])),
                        xytext=(0, 16 if s["contestantId"] == md.HOME_ID else -20),
                        textcoords="offset points", ha="center", fontsize=8.5, color=palette["ink_secondary"])

    for c in cards:
        marker_color = STATUS_DARK["critical"] if c["kind"] != "Yellow" else STATUS_DARK["warning"]
        ax.scatter([c["minute"]], [row(c["contestantId"])], marker="s", s=110, color=marker_color,
                   edgecolors=palette["surface"], linewidth=1.0, zorder=4)

    for s in subs:
        ax.scatter([s["minute"]], [row(s["contestantId"])], marker="^", s=70, color=palette["ink_muted"],
                   alpha=0.85, zorder=3)

    ax.set_xlim(0, 96)
    ax.set_ylim(-0.6, 1.6)
    ax.set_yticks([])
    ax.set_xlabel("Minute")
    ax.axvline(45, color=palette["axis"], linewidth=0.8, linestyle=":")

    legend_elems = [Line2D([0], [0], marker="*", color=palette["surface"], markerfacecolor=palette["ink_primary"],
                            markeredgecolor=palette["ink_primary"], markersize=14, label="Goal", linewidth=0),
                    Line2D([0], [0], marker="s", color=palette["surface"], markerfacecolor=STATUS_DARK["warning"],
                            markersize=10, label="Yellow card", linewidth=0),
                    Line2D([0], [0], marker="^", color=palette["surface"], markerfacecolor=palette["ink_muted"],
                            markersize=10, label="Substitution", linewidth=0)]
    fig.legend(handles=legend_elems, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 0.14), fontsize=10.5, labelcolor=palette["ink_secondary"])

    components.header(fig, kicker="Match Narrative",
                       title="Goals, cards and changes across the 90 minutes",
                       dek=f"{md.HOME_NAME} 2-1 {md.AWAY_NAME}  ·  key moments timeline",
                       palette=palette)
    components.footer(fig, source=md.SOURCE, palette=palette)
    save(fig, "29_momentum_timeline.png")


# ---------------------------------------------------------------------------
# 30. Impact leaderboard
# ---------------------------------------------------------------------------

def impact_leaderboard(shots, passes, defs):
    fig, palette = new_fig()
    ax1 = fig.add_axes([0.08, 0.20, 0.40, 0.54])
    ax2 = fig.add_axes([0.56, 0.20, 0.40, 0.54])

    def score_players(cid):
        scores = {}
        for s in shots:
            if s["contestantId"] != cid:
                continue
            scores[s["player"]] = scores.get(s["player"], 0) + s["xg"] + (3.0 if s["is_goal"] else 0)
        for p in passes:
            if p["contestantId"] != cid:
                continue
            scores[p["player"]] = scores.get(p["player"], 0) + 0.15 * p["progressive"] + 0.35 * p["box_entry"]
        for d in defs:
            if d["contestantId"] != cid:
                continue
            scores[d["player"]] = scores.get(d["player"], 0) + 0.3
        return sorted(scores.items(), key=lambda kv: -kv[1])[:6]

    for ax, cid, color, name in ((ax1, md.HOME_ID, HOME_C, md.HOME_NAME), (ax2, md.AWAY_ID, AWAY_C, md.AWAY_NAME)):
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
    save(fig, "30_impact_leaderboard.png")


def main():
    match_details, events = md.load_events()
    directions = md.compute_attack_directions(events)
    shots = md.build_shots(events, directions)
    passes = md.build_passes(events, directions)
    defs = md.build_defensive_actions(events, directions)
    pressing_actions = md.build_pressing_actions(events, directions)
    recoveries = md.build_recoveries(events, directions)
    cards = md.build_cards(events)
    subs = md.build_substitutions(events)
    touches = md.build_touches(events, directions)
    # Tackle + Aerial + Challenge, each with a win/loss flag -- a dedicated
    # duels set for the duels-summary page (separate from pressing_actions,
    # which excludes Aerial since PPDA's standard action set doesn't use it).
    duels = _build_duels(events, directions)

    cover(match_details["scores"])
    match_summary(shots, passes, touches)
    xg_flow(shots)
    shot_quality_table(shots)
    goal_buildups(events, directions, shots)
    pass_network_combined(passes)
    pass_network_team_page(passes, md.HOME_ID, "07")
    pass_network_team_page(passes, md.AWAY_ID, "08")
    progressive_passes_team_page(passes, md.HOME_ID, "09")
    progressive_passes_team_page(passes, md.AWAY_ID, "10")
    passing_directness_team_page(passes, md.HOME_ID, "11")
    passing_directness_team_page(passes, md.AWAY_ID, "12")
    touch_heatmap_team_page(touches, md.HOME_ID, "13")
    touch_heatmap_team_page(touches, md.AWAY_ID, "14")
    field_tilt(touches, shots)
    possession_thirds(touches)
    progression_bars(passes)
    recoveries_by_third(recoveries)
    defensive_actions(defs)
    ppda(passes, pressing_actions)
    duels_summary(duels)
    crossing_map_team_page(passes, md.HOME_ID, "22")
    crossing_map_team_page(passes, md.AWAY_ID, "23")
    final_third_entries(passes)
    zone14_team_page(passes, md.HOME_ID, "25")
    zone14_team_page(passes, md.AWAY_ID, "26")
    buildup_to_shot(events, shots)
    discipline(pressing_actions, cards)
    momentum_timeline(shots, cards, subs)
    impact_leaderboard(shots, passes, defs)


def _build_duels(events, directions):
    rows = []
    for e in events:
        if e.get("x") is None or not e.get("contestantId"):
            continue
        if e["typeId"] not in (md.T_TACKLE, md.T_AERIAL, md.T_CHALLENGE):
            continue
        x, y = md.norm_xy(e, directions)
        xm, ym = md.to_m(x, y)
        rows.append({
            "contestantId": e["contestantId"],
            "player": e.get("playerName", "Unknown"),
            "minute": e["timeMin"],
            "x": xm, "y": ym,
            "action": {md.T_TACKLE: "Tackle", md.T_AERIAL: "Aerial", md.T_CHALLENGE: "Challenge"}[e["typeId"]],
            "success": e.get("outcome", 0) == 1,
        })
    return rows


if __name__ == "__main__":
    main()

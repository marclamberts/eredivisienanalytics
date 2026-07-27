"""
FC Hradec Králové vs FK Pardubice - Post-Match Analysis
==========================================================
Chance Liga 2026/27, matchday 29 (2026-07-26, FINEP Arena). Hradec Králové
won 2-1 at home (Van Buren 43', Čech 55'; Boledovič 59' for Pardubice).

Follows the page set of the Slavia Praha post-match template the user
supplied, rebuilt in Marc Lamberts' Meridian house style (dark mode) --
housestyle/ package at the repo root -- instead of the template's own
navy/gold/green look. Data: Opta MA3 event feed (CZ Events/CZ 2026-2027),
parsed in match_data.py; shots scored with a small own distance+angle xG
model (no provider xG in this feed).

Usage: python3 "build_charts.py"
Outputs PNGs into ./Visuals, then run build_pdf.py to compile the deck.
"""
import math
import os
import sys

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
import numpy as np
from mplsoccer import Pitch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from housestyle import style, components  # noqa: E402
from housestyle.colors import DARK, CATEGORICAL_DARK, STATUS_DARK  # noqa: E402

import match_data as md  # noqa: E402

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Visuals")
os.makedirs(OUT_DIR, exist_ok=True)

FIGSIZE = (13.33, 7.5)
HOME_C = CATEGORICAL_DARK[0]   # ink blue -- FC Hradec Králové
AWAY_C = CATEGORICAL_DARK[1]   # terracotta -- FK Pardubice
GOAL_C = STATUS_DARK["good"]
HOME_SHORT = "Hradec Kr."
AWAY_SHORT = "Pardubice"

md.HOME_ID, md.AWAY_ID  # noqa


def save(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=170, facecolor=fig.get_facecolor())
    plt.close(fig)
    print("Saved:", path)


def new_fig():
    palette, cats = style.apply("dark")
    fig = plt.figure(figsize=FIGSIZE)
    return fig, palette


# ---------------------------------------------------------------------------
# 1. Cover
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

    fig.text(0.5, 0.13, f"{components.MARK} POST-MATCH ANALYSIS", fontsize=13, fontweight="bold",
              color=palette["accent"], family="sans-serif", ha="center", va="center")

    components.brand_mark(fig, palette=palette, right=0.94, y=0.93)
    components.footer(fig, source=md.SOURCE, palette=palette)
    save(fig, "01_cover.png")


# ---------------------------------------------------------------------------
# 2. xG flow
# ---------------------------------------------------------------------------

def xg_flow(shots):
    fig, palette = new_fig()
    ax = fig.add_axes([0.08, 0.16, 0.78, 0.60])

    def series(cid):
        team_shots = sorted([s for s in shots if s["contestantId"] == cid], key=lambda s: s["minute"])
        mins = [0.0]
        cum = [0.0]
        total = 0.0
        for s in team_shots:
            mins.append(s["minute"])
            cum.append(total)
            total += s["xg"]
            mins.append(s["minute"])
            cum.append(total)
        mins.append(96)
        cum.append(total)
        return mins, cum, team_shots

    for cid, color, name in ((md.HOME_ID, HOME_C, HOME_SHORT), (md.AWAY_ID, AWAY_C, AWAY_SHORT)):
        mins, cum, team_shots = series(cid)
        ax.plot(mins, cum, color=color, linewidth=2.4, zorder=4)
        ax.fill_between(mins, cum, step=None, color=color, alpha=0.10, zorder=1)
        ax.annotate(f"{name}\n{cum[-1]:.2f} xG", xy=(1, cum[-1]), xycoords=("axes fraction", "data"),
                    xytext=(10, 0), textcoords="offset points", color=color, fontsize=10,
                    fontweight="bold", va="center", ha="left", annotation_clip=False)

    # goal markers drawn precisely from cumulative totals at each minute
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
    save(fig, "02_xg_flow.png")


# ---------------------------------------------------------------------------
# 3. Match summary: shot map + KPI bars
# ---------------------------------------------------------------------------

def match_summary(shots, passes, touches):
    fig, palette = new_fig()
    pitch = Pitch(pitch_type="uefa", pitch_color=palette["surface"], line_color=palette["axis"],
                  linewidth=1.0, half=False, line_zorder=2, pad_left=2, pad_right=2)
    ax = fig.add_axes([0.02, 0.10, 0.55, 0.62])
    pitch.draw(ax=ax)

    for s in shots:
        color = HOME_C if s["contestantId"] == md.HOME_ID else AWAY_C
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
    save(fig, "03_match_summary.png")


# ---------------------------------------------------------------------------
# 4. Pass network
# ---------------------------------------------------------------------------

def _pass_network_side(ax, team_passes, color, palette, pitch):
    completed = [p for p in team_passes if p["completed"] and p["end_x"] is not None]
    by_player = {}
    for p in completed:
        by_player.setdefault(p["player"], []).append((p["x"], p["y"]))
    avg_pos = {pl: (np.mean([v[0] for v in vs]), np.mean([v[1] for v in vs]), len(vs))
               for pl, vs in by_player.items() if len(vs) >= 8}

    # naive receiver = next same-team ball-touch player, used only to
    # weight combination edges (no receiver field in this feed)
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
        last = pl.split(" ")[-1]
        pitch.annotate(last, (x, y), ax=ax, ha="center", va="center", fontsize=7.6,
                       color=palette["ink_primary"], fontweight="bold", zorder=5)
    return avg_pos


def pass_network(passes):
    fig, palette = new_fig()
    pitch = Pitch(pitch_type="uefa", pitch_color=palette["surface"], line_color=palette["axis"],
                  linewidth=1.0, half=False, line_zorder=2, pad_left=2, pad_right=2)
    ax1 = fig.add_axes([0.02, 0.10, 0.47, 0.62])
    ax2 = fig.add_axes([0.51, 0.10, 0.47, 0.62])

    home_passes = [p for p in passes if p["contestantId"] == md.HOME_ID]
    away_passes = [p for p in passes if p["contestantId"] == md.AWAY_ID]
    _pass_network_side(ax1, home_passes, HOME_C, palette, pitch)
    _pass_network_side(ax2, away_passes, AWAY_C, palette, pitch)
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
    save(fig, "04_pass_network.png")


# ---------------------------------------------------------------------------
# 5. Field tilt
# ---------------------------------------------------------------------------

def field_tilt(touches, shots):
    fig, palette = new_fig()
    ax = fig.add_axes([0.16, 0.16, 0.78, 0.58])

    bucket = 5
    max_min = 95
    edges = list(range(0, max_min + bucket, bucket))
    tilt = []
    centers = []
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
                       edgecolors=HOME_C if s["contestantId"] == md.HOME_ID else AWAY_C, linewidth=1.4, zorder=6)

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
    save(fig, "05_field_tilt.png")


# ---------------------------------------------------------------------------
# 6. Progression comparison
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
    save(fig, "06_progression.png")


# ---------------------------------------------------------------------------
# 7. Defensive actions
# ---------------------------------------------------------------------------

def defensive_actions(defs):
    fig, palette = new_fig()
    pitch = Pitch(pitch_type="uefa", pitch_color=palette["surface"], line_color=palette["axis"],
                  linewidth=1.0, half=False, line_zorder=2, pad_left=2, pad_right=2)
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
            xs = [105 - p["x"] for p in pts]   # defending own goal on the left
            ys = [68 - p["y"] for p in pts]
            pitch.scatter(xs, ys, ax=ax, s=80, marker=marker, color=action_colors[action],
                          edgecolors=palette["surface"], linewidth=0.6, alpha=0.9, zorder=4)
        counts = {a: sum(1 for d in team_defs if d["action"] == a) for a in markers}
        title = f"{md.team_name(cid)}\nTkl {counts['Tackle']}  ·  Int {counts['Interception']}  ·  Clr {counts['Clearance']}"
        ax.set_title(title, color=HOME_C if cid == md.HOME_ID else AWAY_C, fontsize=12,
                     fontweight="bold", family="sans-serif")

    legend_elems = [Line2D([0], [0], marker=markers[a], color=palette["surface"], markerfacecolor=action_colors[a],
                            markersize=10, label=a, linewidth=0) for a in markers]
    fig.legend(handles=legend_elems, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 0.02), fontsize=10.5, labelcolor=palette["ink_secondary"])

    components.header(fig, kicker="Defending",
                       title="Both sides defended at similar volume; Pardubice cleared more under pressure",
                       dek="Tackles, interceptions and clearances, own goal on the left",
                       palette=palette)
    components.footer(fig, source=md.SOURCE, palette=palette)
    save(fig, "07_defensive_actions.png")


# ---------------------------------------------------------------------------
# 8. PPDA
# ---------------------------------------------------------------------------

def ppda(passes, defs):
    fig, palette = new_fig()
    ax1 = fig.add_axes([0.08, 0.16, 0.40, 0.58])
    ax2 = fig.add_axes([0.56, 0.16, 0.40, 0.58])

    buckets = [(0, 15), (15, 30), (30, 45), (45, 60), (60, 75), (75, 96)]
    labels = ["0-15", "15-30", "30-45", "45-60", "60-75", "75-90+"]

    def ppda_for(defending_id, attacking_id):
        vals = []
        for lo, hi in buckets:
            opp_passes = sum(1 for p in passes if p["contestantId"] == attacking_id
                              and lo <= p["minute"] < hi and p["x"] <= 70 and p["completed"])
            def_actions = sum(1 for d in defs if d["contestantId"] == defending_id
                               and lo <= d["minute"] < hi and d["x"] >= 35)
            vals.append(opp_passes / def_actions if def_actions else np.nan)
        return vals

    home_vals = ppda_for(md.HOME_ID, md.AWAY_ID)
    away_vals = ppda_for(md.AWAY_ID, md.HOME_ID)
    shared_max = max(v for v in home_vals + away_vals if not math.isnan(v)) * 1.15

    for ax, vals, color, name in ((ax1, home_vals, HOME_C, md.HOME_NAME), (ax2, away_vals, AWAY_C, md.AWAY_NAME)):
        xs = np.arange(len(labels))
        clean = [v if not math.isnan(v) else 0 for v in vals]
        ax.bar(xs, clean, color=color)
        for x, v in zip(xs, vals):
            if not math.isnan(v):
                ax.text(x, v + shared_max * 0.02, f"{v:.1f}", ha="center", fontsize=9.5,
                        color=palette["ink_primary"], fontweight="bold")
        overall = sum(v for v in vals if not math.isnan(v)) / sum(1 for v in vals if not math.isnan(v))
        ax.axhline(overall, color=palette["ink_muted"], linestyle="--", linewidth=1.0)
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, fontsize=8.5)
        ax.set_ylim(0, shared_max)
        ax.set_title(f"{name}\nOverall PPDA: {overall:.1f}", color=color, fontsize=11.5,
                     fontweight="bold", family="sans-serif")
        ax.set_ylabel("PPDA")

    components.header(fig, kicker="Pressing",
                       title="Hradec pressed higher and harder than Pardubice all match",
                       dek="Passes per defensive action in the opponent's build-up two-thirds  ·  lower = more intense press",
                       palette=palette)
    components.footer(fig, source=md.SOURCE, palette=palette)
    save(fig, "08_ppda.png")


# ---------------------------------------------------------------------------
# 9. Final third entries
# ---------------------------------------------------------------------------

def final_third_entries(passes):
    fig, palette = new_fig()
    pitch = Pitch(pitch_type="uefa", pitch_color=palette["surface"], line_color=palette["axis"],
                  linewidth=1.0, half=False, line_zorder=2, pad_left=2, pad_right=2)
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
        ax.set_title(title, color=HOME_C if cid == md.HOME_ID else AWAY_C, fontsize=11.5,
                     fontweight="bold", family="sans-serif")

    legend_elems = [Line2D([0], [0], color=zone_colors[z], lw=2.4, label=z) for z in ("Left", "Central", "Right")]
    fig.legend(handles=legend_elems, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 0.02), fontsize=10.5, labelcolor=palette["ink_secondary"])

    components.header(fig, kicker="Final Third",
                       title="Both teams found their way in mostly down the flanks",
                       dek="Completed passes into the final third, by origin lane",
                       palette=palette)
    components.footer(fig, source=md.SOURCE, palette=palette)
    save(fig, "09_final_third_entries.png")


# ---------------------------------------------------------------------------
# 10. Possession by thirds
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
    ax.grid(False)
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
    save(fig, "10_possession_thirds.png")


# ---------------------------------------------------------------------------
# 11. BONUS: goal build-ups
# ---------------------------------------------------------------------------

def goal_buildups(events, directions, shots):
    fig, palette = new_fig()
    pitch = Pitch(pitch_type="uefa", pitch_color=palette["surface"], line_color=palette["axis"],
                  linewidth=1.0, half=False, line_zorder=2, pad_left=1, pad_right=1)

    goals = sorted([s for s in shots if s["is_goal"]], key=lambda s: s["minute"])
    n = len(goals)
    axes = [fig.add_axes([0.02 + i * (0.96 / n), 0.10, 0.96 / n - 0.02, 0.62]) for i in range(n)]

    for ax, g in zip(axes, goals):
        pitch.draw(ax=ax)
        color = HOME_C if g["contestantId"] == md.HOME_ID else AWAY_C
        team_events = [e for e in events if e["contestantId"] == g["contestantId"]
                       and e.get("x") is not None and e["typeId"] in (1, 3, 61)
                       and md.event_time(e) <= g["minute"] * 60 + 59]
        team_events.sort(key=lambda e: (e["periodId"], md.event_time(e), e["eventId"]))
        chain = team_events[-4:]
        pts = []
        for e in chain:
            x, y = md.norm_xy(e, directions)
            xm, ym = md.to_m(x, y)
            pts.append((xm, ym, e.get("playerName", "?")))
        pts.append((g["x"], g["y"], g["player"]))

        for j in range(len(pts) - 1):
            x1, y1, _ = pts[j]
            x2, y2, _ = pts[j + 1]
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
    save(fig, "11_goal_buildups.png")


def main():
    match_details, events = md.load_events()
    directions = md.compute_attack_directions(events)
    shots = md.build_shots(events, directions)
    passes = md.build_passes(events, directions)
    defs = md.build_defensive_actions(events, directions)
    touches = md.build_touches(events, directions)

    cover(match_details["scores"])
    xg_flow(shots)
    match_summary(shots, passes, touches)
    pass_network(passes)
    field_tilt(touches, shots)
    progression_bars(passes)
    defensive_actions(defs)
    ppda(passes, defs)
    final_third_entries(passes)
    possession_thirds(touches)
    goal_buildups(events, directions, shots)


if __name__ == "__main__":
    main()

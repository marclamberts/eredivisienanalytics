"""
Third batch of season-level pitch visuals for PSV Eindhoven, mined from
the Netherlands-Sweden match-report style (WC 2026/reports) and rebuilt
as season aggregates: goal-mouth placement, pass-completion & forward-
ratio zone matrices, width of play, defensive line height trend, xG
conceded / danger against, individual player heatmaps, ball losses,
press map, duels map, touches in box, long balls, free kicks, carries,
box defending, and shot types by situation.

Usage: python3 season_expansion_pitches.py [out_dir]
"""
import sys
import glob
import json
import math
import collections
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap
from mplsoccer import VerticalPitch, Pitch

from individual_metric_pitches import (
    DATA_DIR, TEAM_NAME, C_NAVY, C_INDIGO, C_PURPLE, C_PINK, C_CORAL, C_AMBER,
    C_GREEN, BG, PITCH_LINE, TEXT_SUB, TEXT_FOOT, ZONE_SHADE,
    clean_name, add_logo, build_team_map, qmap, end_xy, load_danger,
    _new_pitch_fig, _title_block, _footer, _save, _density_cmap,
)

SHOT_TYPES = {13, 14, 15, 16}
HEAD_QID, RIGHT_QID, LEFT_QID, PENALTY_QID = 15, 20, 72, 9
CORNER_QID, FREEKICK_QID, THROWIN_QID = 6, 5, 107
DEF_TYPES = {7, 8, 12, 44, 49, 50, 52, 54, 55, 61}
TOUCH_EXCLUDE = {18, 19, 30, 32, 34, 37, 40, 70, 71, 90, 91}

C_RED = "#ff6b6b"
C_BLUE = "#4da3ff"


def load_all(files, cid):
    """Single pass over PSV's 34 matches gathering everything this batch needs."""
    all_passes = []
    touches = []
    player_touches = collections.Counter()
    player_names = {}
    carries = []
    losses = []
    duels = []
    press = []
    box_defending = []
    free_kicks = []
    shots = []
    match_lines = []  # per-match avg def line / avg touch line, keyed by date

    for fn in files:
        base = fn.split("/")[-1]
        date = base[:10]
        with open(fn) as f:
            data = json.load(f)
        events = [e for e in data["event"] if e.get("periodId") in (1, 2)]
        events.sort(key=lambda e: (e["periodId"], e.get("timeMin", 0), e.get("timeSec", 0), e.get("eventId", 0)))

        prev_touch_by_player = {}
        def_xs, touch_xs = [], []
        last_corner_t, last_freekick_t = None, None

        for idx, e in enumerate(events):
            tid = e.get("typeId")
            ecid = e.get("contestantId")
            if ecid != cid:
                continue
            x, y = e.get("x"), e.get("y")
            pid = e.get("playerId")
            pname = e.get("playerName")
            if pname:
                player_names[pid] = pname
            q = qmap(e)

            if x is not None and tid not in TOUCH_EXCLUDE:
                touches.append({"x": x, "y": y, "player": pname})
                touch_xs.append(x)
                if pid:
                    player_touches[pid] += 1

            t_now = e.get("timeMin", 0) + e.get("timeSec", 0) / 60.0

            if tid == 1 and x is not None:
                success = e.get("outcome") == 1
                ex, ey = end_xy(e, q)
                if ex is not None and ey is not None:
                    all_passes.append({"x0": x, "y0": y, "x1": ex, "y1": ey, "success": success})
                if CORNER_QID in q:
                    last_corner_t = t_now
                if FREEKICK_QID in q:
                    last_freekick_t = t_now

            if tid in DEF_TYPES and x is not None:
                def_xs.append(x)
                if x <= 17 and 21 <= (y or 0) <= 79:
                    box_defending.append({"x": x, "y": y, "won": e.get("outcome") == 1})
                if x >= 50:
                    press.append({"x": x, "y": y, "won": e.get("outcome") == 1})

            if tid in (7, 44, 45, 83) and x is not None:
                duels.append({"x": x, "y": y, "won": e.get("outcome") == 1,
                              "aerial": tid == 44})

            if x is not None and (
                (tid == 1 and e.get("outcome") == 0) or tid == 50 or (tid == 3 and e.get("outcome") == 0)
            ):
                losses.append({"x": x, "y": y})

            if pid and x is not None and y is not None:
                prev = prev_touch_by_player.get(pid)
                if prev is not None:
                    dt = (e.get("timeMin", 0) + e.get("timeSec", 0) / 60.0) - prev["t"]
                    cd = math.sqrt((x - prev["x"]) ** 2 + (y - prev["y"]) ** 2)
                    if 0.02 <= dt <= 0.35 and 3 <= cd <= 45:
                        carries.append({"x0": prev["x"], "y0": prev["y"], "x1": x, "y1": y})
                prev_touch_by_player[pid] = {"x": x, "y": y, "t": e.get("timeMin", 0) + e.get("timeSec", 0) / 60.0}

            if tid in SHOT_TYPES and x is not None:
                gm_y, gm_z = q.get(102), q.get(103)
                try:
                    gm_y = float(gm_y) if gm_y is not None else None
                    gm_z = float(gm_z) if gm_z is not None else None
                except (TypeError, ValueError):
                    gm_y, gm_z = None, None
                situation = "open_play"
                if PENALTY_QID in q:
                    situation = "penalty"
                elif last_corner_t is not None and 0 <= t_now - last_corner_t <= 0.75:
                    situation = "corner"
                elif last_freekick_t is not None and 0 <= t_now - last_freekick_t <= 0.25:
                    situation = "free_kick"
                shots.append({
                    "event_id": str(e.get("id")), "match_file": base,
                    "x": x, "y": y, "gm_y": gm_y, "gm_z": gm_z,
                    "is_goal": tid == 16, "is_on_target": tid in (15, 16),
                    "situation": situation,
                })
                if situation == "free_kick":
                    free_kicks.append({"x": x, "y": y, "is_goal": tid == 16, "is_on_target": tid in (15, 16)})

        if def_xs:
            match_lines.append({"date": date, "def_line": np.mean(def_xs),
                                "touch_line": np.mean(touch_xs) if touch_xs else np.nan})

    shots_df = pd.DataFrame(shots)
    danger = load_danger()
    shots_df = shots_df.merge(danger[["match_file", "event_id", "xg", "danger_score"]],
                               on=["match_file", "event_id"], how="left")
    shots_df["xg"] = shots_df["xg"].fillna(0.05)
    shots_df["danger_score"] = shots_df["danger_score"].fillna(0.05)

    opp_shots = danger[(danger["match_file"].isin(shots_df["match_file"].unique())) & (danger["contestant_id"] != cid)].copy()

    lines_df = pd.DataFrame(match_lines).sort_values("date").reset_index(drop=True)

    top_players = [pid for pid, _ in player_touches.most_common(6)]
    player_touch_map = collections.defaultdict(list)
    for t in touches:
        pass

    return {
        "all_passes": all_passes, "touches": touches, "carries": carries,
        "losses": losses, "duels": duels, "press": press,
        "box_defending": box_defending, "free_kicks": free_kicks,
        "shots": shots_df, "opp_shots": opp_shots, "lines": lines_df,
        "player_names": player_names, "player_touches": player_touches,
        "top_players": top_players,
    }


# ---------------------------------------------------------------------------
# A) Goal-mouth shot placement
# ---------------------------------------------------------------------------
def make_goal_mouth(shots, out_path):
    fig = plt.figure(figsize=(16, 9), facecolor=BG)
    fig.text(0.5, 0.94, clean_name(TEAM_NAME), fontsize=24, fontweight="bold", ha="center", color="white")
    fig.text(0.5, 0.90, "Goal Mouth Placement · Eredivisie 2025/26 · Season", fontsize=12, ha="center", color=TEXT_SUB)

    ax = fig.add_axes([0.20, 0.18, 0.60, 0.55])
    ax.set_facecolor(BG)
    GY0, GY1, GZ0, GZ1 = 44.7, 55.3, 0, 38.0
    margin_y, margin_z = 6, 10
    ax.set_xlim(GY0 - margin_y, GY1 + margin_y)
    ax.set_ylim(GZ0 - margin_z * 0.3, GZ1 + margin_z)
    ax.set_box_aspect(2.44 / 7.32)
    ax.axis("off")
    ax.add_patch(plt.Rectangle((GY0, GZ0), GY1 - GY0, GZ1 - GZ0, fill=False, edgecolor="white", lw=3))
    for gz in np.linspace(GZ0, GZ1, 5):
        ax.plot([GY0, GY1], [gz, gz], color=PITCH_LINE, lw=0.6)
    for gy in np.linspace(GY0, GY1, 6):
        ax.plot([gy, gy], [GZ0, GZ1], color=PITCH_LINE, lw=0.6)
    ax.axhline(GZ0, color="#3a4658", lw=1.2)

    on = shots.dropna(subset=["gm_y", "gm_z"])
    goals = on[on["is_goal"]]
    saved = on[(~on["is_goal"]) & (on["is_on_target"])]
    missed = on[~on["is_on_target"]]

    ax.scatter(missed["gm_y"], missed["gm_z"], s=45, facecolors="none", edgecolors="#8a93a1",
              alpha=0.55, linewidths=1.0, zorder=2, label=f"Missed ({len(missed)})")
    ax.scatter(saved["gm_y"], saved["gm_z"], s=70, color=C_CORAL, alpha=0.75, edgecolors="white",
              linewidths=0.8, zorder=3, label=f"On target ({len(saved)})")
    ax.scatter(goals["gm_y"], goals["gm_z"], s=140, color=C_AMBER, alpha=0.95, edgecolors=C_AMBER,
              marker="*", linewidths=1.2, zorder=4, label=f"Goal ({len(goals)})")

    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.14), ncol=3, frameon=False,
             fontsize=10, labelcolor="#c7ccd4")
    _footer(fig, f"{len(shots)} shots, {len(on)} plotted with goal-mouth coordinates (on/near target only)",
            "gm_y/gm_z = Opta qualifiers 102/103, goal-mouth y/height; view from the shooter's perspective")
    _save(fig, out_path)


# ---------------------------------------------------------------------------
# B) Zone matrices: pass completion % and forward-pass ratio
# ---------------------------------------------------------------------------
def _zone_matrix(all_passes, out_path, title, value_fn, cmap, str_format, methodology, subtitle):
    x = np.array([p["x0"] for p in all_passes])
    y = np.array([p["y0"] for p in all_passes])
    vals = np.array([value_fn(p) for p in all_passes], dtype=float)

    fig, pitch = _new_pitch_fig(figsize=(11, 15), half=False)
    ax = fig.add_axes([0.04, 0.08, 0.92, 0.78])
    pitch.draw(ax=ax)
    _title_block(fig, title, subtitle)

    stats_val = pitch.bin_statistic_positional(x, y, values=vals, statistic="mean", positional="full")
    stats_cnt = pitch.bin_statistic_positional(x, y, statistic="count", positional="full")
    for sv in stats_val:
        pitch.heatmap(sv, ax=ax, cmap=cmap, edgecolors=BG, linewidth=1.2, zorder=1, alpha=0.92,
                     vmin=0, vmax=1)
    for sv, sc in zip(stats_val, stats_cnt):
        vflat = sv["statistic"].ravel()
        nflat = sc["statistic"].ravel()
        cxflat = np.asarray(sv["cx"]).ravel()
        cyflat = np.asarray(sv["cy"]).ravel()
        for v, n, cx, cy in zip(vflat, nflat, cxflat, cyflat):
            if np.isnan(v) or n == 0:
                continue
            ax.text(cx, cy, f"{v*100:.0f}%\n({int(n)})", ha="center", va="center",
                   fontsize=8.5, fontweight="bold", color="white", zorder=3)

    _footer(fig, f"{len(all_passes)} passes this season", methodology)
    _save(fig, out_path)


def make_pass_completion_matrix(all_passes, out_path):
    cmap = LinearSegmentedColormap.from_list("comp", ["#5c1a1a", "#8a5a1f", "#3a6b3a"])
    _zone_matrix(all_passes, out_path, "Pass Completion Matrix · Eredivisie 2025/26 · Season",
                lambda p: 1.0 if p["success"] else 0.0, cmap, "{:.0f}%",
                "Cell = pass completion % (count) by starting zone, Juego de Posición grid · green = higher accuracy",
                None)


def make_forward_pass_matrix(all_passes, out_path):
    cmap = LinearSegmentedColormap.from_list("fwd", ["#1f2937", "#8a5a1f", "#ffc247"])
    completed = [p for p in all_passes if p["success"]]
    _zone_matrix(completed, out_path, "Forward Pass Ratio · Eredivisie 2025/26 · Season",
                lambda p: 1.0 if (p["x1"] - p["x0"]) > 5 else 0.0, cmap, "{:.0f}%",
                "Cell = % of completed passes going forward (end_x > start_x + 5) by starting zone · brighter = more direct",
                None)


# ---------------------------------------------------------------------------
# C) Width of play (KDE)
# ---------------------------------------------------------------------------
def make_width_of_play(touches, out_path):
    fig = plt.figure(figsize=(16, 9), facecolor=BG)
    fig.text(0.5, 0.94, clean_name(TEAM_NAME), fontsize=24, fontweight="bold", ha="center", color="white")
    fig.text(0.5, 0.90, "Width of Play · Eredivisie 2025/26 · Season", fontsize=12, ha="center", color=TEXT_SUB)

    ax = fig.add_axes([0.05, 0.10, 0.90, 0.76])
    pitch = Pitch(pitch_type="opta", pitch_color=BG, line_color=PITCH_LINE, linewidth=1.1)
    pitch.draw(ax=ax)
    x = np.array([t["x"] for t in touches])
    y = np.array([t["y"] for t in touches])
    pitch.kdeplot(x, y, ax=ax, cmap="magma", fill=True, levels=100, alpha=0.85, zorder=1, thresh=0.02)
    pitch.draw(ax=ax)

    left = np.mean(y >= 67) * 100
    right = np.mean(y <= 33) * 100
    centre = 100 - left - right
    _footer(fig, f"Left {left:.0f}%  ·  Centre {centre:.0f}%  ·  Right {right:.0f}%  ({len(touches)} touches)",
            "Left/right from the attacking team's perspective (y ≥ 67 wide-left, y ≤ 33 wide-right)")
    _save(fig, out_path)


# ---------------------------------------------------------------------------
# D) Line height trend
# ---------------------------------------------------------------------------
def make_line_height_trend(lines_df, out_path):
    fig, ax = plt.subplots(figsize=(15, 8))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    n = len(lines_df)
    xs = np.arange(1, n + 1)
    ax.plot(xs, lines_df["touch_line"], color=C_AMBER, lw=2.2, marker="o", markersize=4, label="Avg. touch height (whole team)")
    ax.plot(xs, lines_df["def_line"], color=C_CORAL, lw=2.2, marker="o", markersize=4, label="Avg. defensive-action height")
    ax.axhline(lines_df["def_line"].mean(), color=C_CORAL, lw=1.0, ls=(0, (4, 3)), alpha=0.5)
    ax.axhline(lines_df["touch_line"].mean(), color=C_AMBER, lw=1.0, ls=(0, (4, 3)), alpha=0.5)

    ax.set_xlim(0.5, n + 0.5)
    ax.set_xlabel("Matchday", color=TEXT_SUB, fontsize=10.5)
    ax.set_ylabel("Pitch x-position (0=own goal, 100=opp. goal)", color=TEXT_SUB, fontsize=10.5)
    ax.tick_params(colors=TEXT_SUB)
    for spine in ax.spines.values():
        spine.set_color(PITCH_LINE)
    ax.grid(True, axis="y", color=PITCH_LINE, lw=0.6, alpha=0.6)
    ax.legend(loc="upper left", frameon=False, fontsize=10, labelcolor="#c7ccd4")

    fig.text(0.06, 0.955, clean_name(TEAM_NAME), fontsize=22, fontweight="bold", color="white")
    fig.text(0.06, 0.925, "Team Line Height by Match · Eredivisie 2025/26 · Season", fontsize=11.5, color=TEXT_SUB)
    fig.text(0.06, 0.02, f"Data via Opta | Eredivisie 2025/26 event data · def. line = avg x of tackles/interceptions/"
             f"clearances/aerials/recoveries; touch line = avg x of all touches",
             fontsize=7.6, color=TEXT_FOOT)
    fig.text(0.98, 0.02, "Marc Lamberts", fontsize=9, ha="right", color=TEXT_FOOT, style="italic")
    fig.subplots_adjust(left=0.07, right=0.97, top=0.87, bottom=0.10)
    add_logo(fig)
    fig.savefig(out_path, dpi=200, facecolor=BG)
    plt.close(fig)
    print("Saved:", out_path)


# ---------------------------------------------------------------------------
# E) xG conceded shot map + F) danger-against heatmap
# ---------------------------------------------------------------------------
def make_xg_conceded(opp_shots, out_path):
    fig, pitch = _new_pitch_fig(half=True)
    ax = fig.add_axes([0.04, 0.06, 0.92, 0.82])
    pitch.draw(ax=ax)
    _title_block(fig, "Shots Conceded · xG Against · Eredivisie 2025/26 · Season",
                 None)

    goals = opp_shots[opp_shots["is_goal"] == 1]
    ontarget = opp_shots[(opp_shots["is_goal"] == 0) & (opp_shots["is_on_target"] == 1)]
    off = opp_shots[opp_shots["is_on_target"] == 0]
    for sub, color, alpha in [(off, "#8a93a1", 0.5), (ontarget, C_BLUE, 0.78), (goals, C_RED, 0.95)]:
        if len(sub) == 0:
            continue
        sizes = 1500 * sub["xg"].clip(0.02, 0.85) + 40
        pitch.scatter(sub["x"], sub["y"], ax=ax, s=sizes, color=color, alpha=alpha,
                     edgecolors="white", linewidth=1.0, zorder=3 if color == C_RED else 2)
    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=C_RED, markeredgecolor=C_RED, markersize=11, label=f"Goal conceded ({len(goals)})"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=C_BLUE, markeredgecolor="white", markersize=11, label=f"Saved ({len(ontarget)})"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#8a93a1", markeredgecolor="white", markersize=9, label=f"Off target ({len(off)})"),
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.885), ncol=3,
              frameon=False, fontsize=9.5, labelcolor="#c7ccd4")
    xg_total = opp_shots["xg"].sum()
    _footer(fig, f"{len(opp_shots)} shots faced · {int(goals.shape[0])} conceded · {xg_total:.1f} xG against",
            "Opponent shots in all 34 PSV matches, plotted from PSV's defensive perspective (attacking toward top)")
    _save(fig, out_path)


def make_danger_against_heatmap(opp_shots, out_path):
    fig, pitch = _new_pitch_fig(half=True)
    ax = fig.add_axes([0.07, 0.06, 0.72, 0.82])
    pitch.draw(ax=ax)
    _title_block(fig, "Danger Against · Eredivisie 2025/26 · Season", None)
    x = opp_shots["x"].to_numpy()
    y = opp_shots["y"].to_numpy()
    w = opp_shots["danger_score"].fillna(0).to_numpy()
    stats = pitch.bin_statistic(x, y, values=w, statistic="sum", bins=(24, 20))
    pitch.heatmap(stats, ax=ax, cmap=_density_cmap(C_RED), edgecolors=BG, linewidth=0.15, zorder=1, alpha=0.9)
    pitch.draw(ax=ax)
    cax = fig.add_axes([0.83, 0.20, 0.014, 0.55])
    sm = plt.cm.ScalarMappable(cmap=_density_cmap(C_RED))
    sm.set_array(stats["statistic"])
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label("Summed danger against", color=TEXT_SUB, fontsize=8.5)
    cb.ax.yaxis.set_tick_params(color=TEXT_SUB, labelcolor=TEXT_SUB, labelsize=7.5)
    _footer(fig, f"{len(opp_shots)} shots faced this season", "Where opponents generated their danger against PSV, all 34 matches")
    _save(fig, out_path)


# ---------------------------------------------------------------------------
# G) Player heatmaps (top 6 by touches)
# ---------------------------------------------------------------------------
def make_player_heatmaps(touches, player_names, top_players, out_path):
    fig = plt.figure(figsize=(17, 12))
    fig.patch.set_facecolor(BG)
    fig.text(0.5, 0.965, clean_name(TEAM_NAME), fontsize=24, fontweight="bold", ha="center", color="white")
    fig.text(0.5, 0.935, "Player Touch Heatmaps · Top 6 by Touches · Eredivisie 2025/26 · Season",
             fontsize=12, ha="center", color=TEXT_SUB)

    by_player = collections.defaultdict(lambda: {"x": [], "y": []})
    for t in touches:
        by_player[t["player"]]["x"].append(t["x"])
        by_player[t["player"]]["y"].append(t["y"])

    counts = collections.Counter({name: len(v["x"]) for name, v in by_player.items()})
    top6 = [name for name, _ in counts.most_common(6)]

    axes = [fig.add_axes([0.03 + (i % 3) * 0.325, 0.50 - (i // 3) * 0.40, 0.30, 0.32]) for i in range(6)]
    for ax, name in zip(axes, top6):
        pitch = VerticalPitch(pitch_type="opta", pitch_color=BG, line_color=PITCH_LINE, linewidth=0.9, half=False)
        pitch.draw(ax=ax)
        xs = np.array(by_player[name]["x"])
        ys = np.array(by_player[name]["y"])
        if len(xs) >= 15:
            pitch.kdeplot(xs, ys, ax=ax, cmap="magma", fill=True, levels=60, alpha=0.85, thresh=0.03, zorder=1)
            pitch.draw(ax=ax)
        else:
            pitch.scatter(xs, ys, ax=ax, s=20, color=C_AMBER, alpha=0.6, zorder=2)
        ax.set_title(f"{name}  ({counts[name]} touches)", fontsize=11, fontweight="bold", color="white", pad=8)

    _footer(fig, "Touch density per player, all competitive minutes this season", None)
    _save(fig, out_path)


# ---------------------------------------------------------------------------
# H-M) generic scatter / arrow map builder
# ---------------------------------------------------------------------------
def make_scatter_map(records, out_path, title, subtitle, color_won, color_lost, label,
                     methodology, key="won", half=False):
    fig, pitch = _new_pitch_fig(half=half)
    ax = fig.add_axes([0.04, 0.08, 0.92, 0.78])
    pitch.draw(ax=ax)
    _title_block(fig, title, subtitle)

    won = [r for r in records if r.get(key)]
    lost = [r for r in records if not r.get(key)]
    if lost:
        pitch.scatter([r["x"] for r in lost], [r["y"] for r in lost], ax=ax, s=55, color=color_lost,
                     alpha=0.6, edgecolors="white", linewidth=0.6, zorder=2)
    if won:
        pitch.scatter([r["x"] for r in won], [r["y"] for r in won], ax=ax, s=55, color=color_won,
                     alpha=0.8, edgecolors="white", linewidth=0.6, zorder=3)

    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=color_won, markeredgecolor="white", markersize=10, label=f"Won ({len(won)})"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=color_lost, markeredgecolor="white", markersize=10, label=f"Lost ({len(lost)})"),
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.882), ncol=2,
              frameon=False, fontsize=9.5, labelcolor="#c7ccd4")
    _footer(fig, f"{len(records)} {label} this season", methodology)
    _save(fig, out_path)


def make_location_map(records, out_path, title, color, label, methodology, half=False):
    fig, pitch = _new_pitch_fig(half=half)
    ax = fig.add_axes([0.04, 0.08, 0.92, 0.78])
    pitch.draw(ax=ax)
    _title_block(fig, title, None)
    pitch.scatter([r["x"] for r in records], [r["y"] for r in records], ax=ax, s=50, color=color,
                 alpha=0.65, edgecolors="white", linewidth=0.5, zorder=2)
    legend_elems = [Line2D([0], [0], marker="o", color="none", markerfacecolor=color, markeredgecolor="white", markersize=10, label=label)]
    fig.legend(handles=legend_elems, loc="lower center", bbox_to_anchor=(0.5, 0.882), ncol=1,
              frameon=False, fontsize=9.5, labelcolor="#c7ccd4")
    _footer(fig, f"{len(records)} {label.lower()}s this season", methodology)
    _save(fig, out_path)


def make_arrow_map(records, out_path, title, color, label, methodology, max_arrows=300):
    fig, pitch = _new_pitch_fig(half=False)
    ax = fig.add_axes([0.04, 0.08, 0.92, 0.78])
    pitch.draw(ax=ax)
    _title_block(fig, title, None)
    sample = records
    if len(records) > max_arrows:
        rng = np.random.default_rng(7)
        idx = rng.choice(len(records), size=max_arrows, replace=False)
        sample = [records[i] for i in idx]
    for r in sample:
        pitch.lines(r["x0"], r["y0"], r["x1"], r["y1"], ax=ax, color=color, lw=1.1, alpha=0.55, zorder=2, comet=False)
        pitch.scatter(r["x0"], r["y0"], ax=ax, s=12, color=color, alpha=0.8, zorder=3, linewidths=0)
        pitch.scatter(r["x1"], r["y1"], ax=ax, s=22, facecolor="none", edgecolors=color, linewidths=1.0, alpha=0.8, zorder=3)
    note = "" if len(records) <= max_arrows else f" (random sample of {max_arrows} shown)"
    legend_elems = [Line2D([0], [0], color=color, linewidth=2, label=label + note)]
    fig.legend(handles=legend_elems, loc="lower center", bbox_to_anchor=(0.5, 0.882), ncol=1,
              frameon=False, fontsize=9.5, labelcolor="#c7ccd4")
    _footer(fig, f"{len(records)} {label.lower()}s this season", methodology)
    _save(fig, out_path)


# ---------------------------------------------------------------------------
# N) Shot types by situation
# ---------------------------------------------------------------------------
def make_shot_situation(shots, out_path):
    fig = plt.figure(figsize=(17, 11))
    fig.patch.set_facecolor(BG)
    fig.text(0.5, 0.955, clean_name(TEAM_NAME), fontsize=26, fontweight="bold", ha="center", color="white")
    fig.text(0.5, 0.918, "Shot Map · By Situation · Eredivisie 2025/26 · Season",
             fontsize=12, ha="center", color=TEXT_SUB)

    panels = [
        ("Open Play", shots[shots["situation"] == "open_play"], C_NAVY),
        ("Corner", shots[shots["situation"] == "corner"], C_INDIGO),
        ("Free Kick", shots[shots["situation"] == "free_kick"], C_PINK),
        ("Penalty", shots[shots["situation"] == "penalty"], C_AMBER),
    ]
    axes = [fig.add_axes([0.015 + i * 0.245, 0.06, 0.225, 0.80]) for i in range(4)]
    for ax, (label, sub, color) in zip(axes, panels):
        pitch = VerticalPitch(pitch_type="opta", pitch_color=BG, line_color=PITCH_LINE, linewidth=1.0, half=True)
        pitch.draw(ax=ax)
        goals = sub[sub["is_goal"]]
        others = sub[~sub["is_goal"]]
        if len(others):
            pitch.scatter(others["x"], others["y"], ax=ax, s=1000 * others["xg"].clip(0.02, 0.85) + 25,
                         color=color, alpha=0.55, edgecolors="white", linewidth=0.7, zorder=2)
        if len(goals):
            pitch.scatter(goals["x"], goals["y"], ax=ax, s=1000 * goals["xg"].clip(0.02, 0.85) + 25,
                         color=C_AMBER, alpha=0.95, edgecolors=C_AMBER, linewidth=1.1, zorder=3)
        ax.set_title(f"{label}\n{len(sub)} shots, {len(goals)} goals", fontsize=10.5, fontweight="bold",
                    color=color, pad=8)

    _footer(fig, f"{len(shots)} total shots this season", "Situation identified from Opta set-piece qualifiers (5=free kick, 6=corner, 9=penalty)")
    _save(fig, out_path)


def main():
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/tmp/psv_expansion")
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(glob.glob(f"{DATA_DIR}/*.json"))
    team_to_cid = build_team_map(files)
    match_key = next((full for full in team_to_cid if TEAM_NAME.lower() in full.lower()), None)
    cid = team_to_cid[match_key]

    print("Parsing all PSV matches (expansion pass)...")
    d = load_all(files, cid)
    print(f"passes={len(d['all_passes'])} touches={len(d['touches'])} carries={len(d['carries'])} "
          f"losses={len(d['losses'])} duels={len(d['duels'])} press={len(d['press'])} "
          f"box_def={len(d['box_defending'])} free_kicks={len(d['free_kicks'])} "
          f"shots={len(d['shots'])} opp_shots={len(d['opp_shots'])} matches_with_lines={len(d['lines'])}")

    make_goal_mouth(d["shots"], out_dir / "01_goal_mouth_psv.png")
    make_pass_completion_matrix(d["all_passes"], out_dir / "02_pass_completion_matrix_psv.png")
    make_forward_pass_matrix(d["all_passes"], out_dir / "03_forward_pass_matrix_psv.png")
    make_width_of_play(d["touches"], out_dir / "04_width_of_play_psv.png")
    make_line_height_trend(d["lines"], out_dir / "05_line_height_trend_psv.png")
    make_xg_conceded(d["opp_shots"], out_dir / "06_xg_conceded_psv.png")
    make_danger_against_heatmap(d["opp_shots"], out_dir / "07_danger_against_psv.png")
    make_player_heatmaps(d["touches"], d["player_names"], d["top_players"], out_dir / "08_player_heatmaps_psv.png")

    long_balls = [p for p in d["all_passes"] if p["success"] and
                  math.hypot(p["x1"] - p["x0"], p["y1"] - p["y0"]) >= 30]
    make_arrow_map(long_balls, out_dir / "09_long_ball_map_psv.png", "Long Balls (≥30m) · Eredivisie 2025/26 · Season",
                  C_GREEN, "Long ball", "Completed pass ≥30m real distance, any direction")

    make_arrow_map(d["carries"], out_dir / "10_carry_map_psv.png", "Carries · Eredivisie 2025/26 · Season",
                  C_CORAL, "Carry", "Same-player consecutive touches, 3-45m apart, within 21 seconds")

    make_scatter_map(d["duels"], out_dir / "11_duels_map_psv.png", "Duels · Aerial & Ground · Eredivisie 2025/26 · Season",
                     None, C_AMBER, "#4a5568", "duel", "Aerial duels (qualifier 44) + tackles/challenges, won vs lost")

    make_location_map(d["press"], out_dir / "12_press_map_psv.png", "Press Map · High Defensive Actions · Eredivisie 2025/26 · Season",
                      C_PINK, "Press event", "Defensive action (tackle/interception/clearance/aerial/recovery) at x≥50 (opposition half)")

    make_location_map(d["losses"], out_dir / "13_ball_losses_psv.png", "Ball Losses · Eredivisie 2025/26 · Season",
                      C_CORAL, "Loss", "Failed pass, failed take-on, or dispossession")

    box_touches = [{"x": t["x"], "y": t["y"]} for t in d["touches"] if t["x"] >= 83 and 21 <= (t["y"] or 0) <= 79]
    make_location_map(box_touches, out_dir / "14_touches_in_box_psv.png", "Touches in the Box · Eredivisie 2025/26 · Season",
                      C_AMBER, "Touch", "All touches inside the opposition penalty area (x≥83, y 21-79)")

    if d["free_kicks"]:
        make_location_map(d["free_kicks"], out_dir / "15_free_kick_map_psv.png", "Direct Free Kicks · Eredivisie 2025/26 · Season",
                          C_PURPLE, "Free kick", "Direct free-kick shot attempts (Opta qualifier 5, excluding penalties)")
    else:
        print("Skipping free-kick map: this event feed does not tag direct free-kick shots (qualifier 5 never appears on a shot event).")

    make_location_map(d["box_defending"], out_dir / "16_box_defending_psv.png", "Box Defending · Eredivisie 2025/26 · Season",
                      C_BLUE, "Action", "Defensive actions inside PSV's own box (x≤17, y 21-79)")

    make_shot_situation(d["shots"], out_dir / "17_shot_types_situation_psv.png")

    print("All season expansion charts done.")


if __name__ == "__main__":
    main()

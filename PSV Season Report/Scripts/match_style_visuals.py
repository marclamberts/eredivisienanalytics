"""
Netherlands-Sweden match-report style visuals, aggregated across PSV
Eindhoven's full 2025/26 Eredivisie season (34 games) instead of a
single match: season shot map, danger heatmap, a PSV-vs-league-average
radar, a points trajectory story, a home/away scout card, and a
results strip. Grammar (pitch, palette, header/footer) mirrors
match_report_addons.py's single-match report style.

Usage: python3 match_style_visuals.py [out_dir]
"""
import sys
import re
import glob
import json
import math
import collections
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, Arc, Wedge
from matplotlib.colors import LinearSegmentedColormap
from mplsoccer import VerticalPitch

DATA_DIR = "/Users/marclamberts/Event data/Eredivisie"
DANGER_DIR = "/Users/marclamberts/Event data/Eredivisie/Danger"
LOGO_PATH = "/Users/marclamberts/Downloads/Waltzing Analytics Logo Type.png"
TEAM_NAME = "PSV Eindhoven"

BG = "#0d1117"
PANEL = "#161b22"
GRID = "#26384d"
PITCH_LINE = "#2c3a4d"
TEXT = "#f5f7fb"
MUTED = "#8ca0b3"
PSV_COLOR = "#ff8a3d"
LEAGUE_COLOR = "#4da3ff"
GOLD = "#ffd166"
GREEN = "#55d68b"
RED = "#ff6b6b"
PURPLE = "#b691ff"

DANGER_CMAP = LinearSegmentedColormap.from_list("danger", ["#0d1117", "#1f4e79", "#2f8fd1", "#ffc247", "#ff6b6b"])


def add_logo(fig, width=0.13, margin=0.016):
    import matplotlib.image as mpimg
    try:
        img = mpimg.imread(LOGO_PATH)
    except FileNotFoundError:
        return
    fig_w, fig_h = fig.get_size_inches()
    img_h, img_w = img.shape[0], img.shape[1]
    width_in = width * fig_w
    height_in = width_in * (img_h / img_w)
    height = height_in / fig_h
    left = 1 - margin - width
    bottom = 1 - margin - height
    logo_ax = fig.add_axes([left, bottom, width, height], zorder=10)
    logo_ax.patch.set_alpha(0)
    logo_ax.set_xlim(0, img_w)
    logo_ax.set_ylim(img_h, 0)
    logo_ax.imshow(img)
    logo_ax.axis("off")




def _header(fig, eyebrow, title, subtitle=""):
    fig.text(0.045, 0.965, eyebrow.upper(), color=MUTED, fontsize=9, weight="bold")
    fig.text(0.045, 0.928, title, color=TEXT, fontsize=21, weight="bold")
    if subtitle:
        fig.text(0.045, 0.900, subtitle, color=MUTED, fontsize=10)
    fig.text(0.955, 0.865, "PSV EINDHOVEN · 2025/26", color=MUTED, fontsize=8.5, ha="right")
    fig.text(0.955, 0.025, "Data: Opta | Season aggregate, Netherlands-Sweden report style", color=MUTED, fontsize=7, ha="right")


def _save(fig, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=190, facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", path)


PREFIX_RE = re.compile(r"^(CSD|CD|CS|SD)\s+")


def clean_name(name):
    return PREFIX_RE.sub("", name)


def build_team_map(files):
    team_cid_sets = collections.defaultdict(list)
    for fn in files:
        m = re.match(r"\d{4}-\d{2}-\d{2}_(.+) - (.+)\.json$", fn.split("/")[-1])
        if not m:
            continue
        home, away = m.group(1), m.group(2)
        with open(fn) as f:
            data = json.load(f)
        cids = set(e["contestantId"] for e in data["event"] if "contestantId" in e)
        team_cid_sets[home].append(cids)
        team_cid_sets[away].append(cids)
    team_to_cid = {}
    for team, sets in team_cid_sets.items():
        inter = set.intersection(*sets)
        if len(inter) == 1:
            team_to_cid[team] = next(iter(inter))
    return team_to_cid


def load_season_matches(files, team_name, cid):
    """Chronological list of PSV match dicts with result info."""
    matches = []
    for fn in files:
        base = fn.split("/")[-1]
        m = re.match(r"(\d{4}-\d{2}-\d{2})_(.+) - (.+)\.json$", base)
        if not m:
            continue
        date, home_name, away_name = m.group(1), m.group(2), m.group(3)
        if team_name not in (home_name, away_name):
            continue
        with open(fn) as f:
            data = json.load(f)
        scores = data.get("matchDetails", {}).get("scores", {}).get("total")
        if not scores:
            continue
        is_home = home_name == team_name
        gf = scores["home"] if is_home else scores["away"]
        ga = scores["away"] if is_home else scores["home"]
        opp = away_name if is_home else home_name
        result = "W" if gf > ga else ("L" if gf < ga else "D")
        pts = 3 if result == "W" else (1 if result == "D" else 0)
        matches.append({
            "date": date, "file": fn, "is_home": is_home, "opponent": opp,
            "gf": gf, "ga": ga, "result": result, "points": pts,
        })
    matches.sort(key=lambda r: r["date"])
    return matches


def load_danger(files_glob=None):
    paths = sorted(glob.glob(f"{DANGER_DIR}/[0-9][0-9][0-9][0-9]-*_danger_models.csv"))
    frames = []
    for p in paths:
        try:
            frames.append(pd.read_csv(p))
        except Exception:
            continue
    df = pd.concat(frames, ignore_index=True)
    df["contestant_id"] = df["contestant_id"].astype(str)
    return df


# ---------------------------------------------------------------------------
# 1) Season shot map
# ---------------------------------------------------------------------------
def make_shot_map(danger, cid, matches, out_path):
    shots = danger[danger["contestant_id"] == cid].copy()
    fig = plt.figure(figsize=(16, 9), facecolor=BG)
    _header(fig, "Netherlands-Sweden report style · xG shot map",
            "PSV Eindhoven — Season Shot Map", f"{len(shots)} shots · {len(matches)} matches · 2025/26 Eredivisie")

    ax = fig.add_axes([0.045, 0.06, 0.60, 0.84])
    pitch = VerticalPitch(pitch_type="opta", pitch_color=BG, line_color=PITCH_LINE,
                          linewidth=1.1, half=True)
    pitch.draw(ax=ax)
    goals = shots[shots["is_goal"] == 1]
    ontarget = shots[(shots["is_goal"] == 0) & (shots["is_on_target"] == 1)]
    off = shots[(shots["is_on_target"] == 0) & (shots["is_goal"] == 0)]

    for sub, color, alpha in [(off, MUTED, 0.55), (ontarget, PSV_COLOR, 0.78), (goals, GOLD, 0.95)]:
        if len(sub) == 0:
            continue
        sizes = 1300 * sub["xg"].fillna(0.03).clip(0.02, 0.8) + 45
        edge = GOLD if color == GOLD else TEXT
        pitch.scatter(sub["x"], sub["y"], ax=ax, s=sizes, color=color, alpha=alpha,
                      edgecolors=edge, linewidth=1.1, zorder=3 if color == GOLD else 2)

    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=GOLD, markeredgecolor=GOLD, markersize=11, label=f"Goal ({len(goals)})"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=PSV_COLOR, markeredgecolor=TEXT, markersize=11, label=f"On target ({len(ontarget)})"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=MUTED, markeredgecolor=TEXT, markersize=9, label=f"Off target / blocked ({len(off)})"),
    ]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.03), ncol=3,
              facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT, fontsize=9)

    side = fig.add_axes([0.685, 0.10, 0.28, 0.72])
    side.set_facecolor(PANEL)
    side.axis("off")
    side.add_patch(Rectangle((0, 0), 1, 1, transform=side.transAxes, facecolor=PANEL, edgecolor=GRID, lw=0.9))
    side.text(0.08, 0.93, "Shot Quality — Season", color=TEXT, fontsize=15, weight="bold", transform=side.transAxes)

    n_shots = len(shots)
    xg_total = shots["xg"].sum()
    rows = [
        ("Shots", f"{n_shots}"),
        ("Goals", f"{int(goals.shape[0])}"),
        ("Total xG", f"{xg_total:.2f}"),
        ("xG / shot", f"{(xg_total / n_shots if n_shots else 0):.2f}"),
        ("On target %", f"{(shots['is_on_target'].mean() * 100 if n_shots else 0):.0f}%"),
        ("Conversion", f"{(goals.shape[0] / n_shots * 100 if n_shots else 0):.1f}%"),
        ("Big chances (xG≥0.2)", f"{int((shots['xg'] >= 0.2).sum())}"),
        ("Goals / 90", f"{(goals.shape[0] / len(matches) if matches else 0):.2f}"),
        ("xG / 90", f"{(xg_total / len(matches) if matches else 0):.2f}"),
        ("Shots / 90", f"{(n_shots / len(matches) if matches else 0):.1f}"),
    ]
    y0 = 0.82
    for i, (lab, val) in enumerate(rows):
        y = y0 - i * 0.078
        side.text(0.08, y, lab, color=MUTED, fontsize=10.5, transform=side.transAxes)
        side.text(0.92, y, val, color=TEXT, fontsize=12, weight="bold", ha="right", transform=side.transAxes)

    add_logo(fig)
    _save(fig, out_path)


# ---------------------------------------------------------------------------
# 2) Season danger heatmap
# ---------------------------------------------------------------------------
def make_danger_heatmap(danger, cid, matches, out_path):
    shots = danger[danger["contestant_id"] == cid].copy()
    fig = plt.figure(figsize=(16, 9), facecolor=BG)
    _header(fig, "Netherlands-Sweden report style · danger density",
            "PSV Eindhoven — Season Danger Heatmap",
            "Weighted by danger_score (mean of xG, PSxG, situation danger, P(on target), xGOT)")
    ax = fig.add_axes([0.07, 0.06, 0.72, 0.84])
    pitch = VerticalPitch(pitch_type="opta", pitch_color=BG, line_color=PITCH_LINE,
                          linewidth=1.1, half=True)
    pitch.draw(ax=ax)
    x = shots["x"].to_numpy()
    y = shots["y"].to_numpy()
    w = shots["danger_score"].fillna(0).to_numpy()
    stats = pitch.bin_statistic(x, y, values=w, statistic="sum", bins=(24, 20))
    pitch.heatmap(stats, ax=ax, cmap=DANGER_CMAP, edgecolors=BG, linewidth=0.15, zorder=1, alpha=0.88)
    pitch.draw(ax=ax)

    cax = fig.add_axes([0.83, 0.14, 0.014, 0.60])
    sm = plt.cm.ScalarMappable(cmap=DANGER_CMAP)
    sm.set_array(stats["statistic"])
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label("Summed danger score", color=MUTED, fontsize=8.5)
    cb.ax.yaxis.set_tick_params(color=MUTED, labelcolor=MUTED, labelsize=7.5)
    add_logo(fig)
    _save(fig, out_path)


# ---------------------------------------------------------------------------
# 3) League metrics pass (one sweep over all matches) -> radar
# ---------------------------------------------------------------------------
def build_league_metrics(files, danger):
    team_minutes = collections.Counter()
    team_passes = collections.Counter()
    team_passes_ok = collections.Counter()
    team_box_entries = collections.Counter()
    team_progressive = collections.Counter()
    team_ppda_opp_passes = collections.Counter()
    team_ppda_def_actions = collections.Counter()
    team_high_regains = collections.Counter()

    for fn in files:
        with open(fn) as f:
            data = json.load(f)
        events = [e for e in data["event"] if e.get("periodId") in (1, 2)]
        cids = sorted(set(e["contestantId"] for e in events if "contestantId" in e))
        if len(cids) != 2:
            continue
        for cid in cids:
            team_minutes[cid] += 90.0
        for e in events:
            tid = e.get("typeId")
            cid = e.get("contestantId")
            if cid not in cids:
                continue
            other = cids[0] if cid == cids[1] else cids[1]
            x = e.get("x")
            if tid == 1:  # pass
                team_passes[cid] += 1
                if e.get("outcome") == 1:
                    team_passes_ok[cid] += 1
                    qmap = {q["qualifierId"]: q.get("value") for q in e.get("qualifier", [])}
                    try:
                        end_x = float(qmap.get(140, x))
                    except (TypeError, ValueError):
                        end_x = x
                    end_y_raw = qmap.get(141, e.get("y"))
                    try:
                        end_y = float(end_y_raw)
                    except (TypeError, ValueError):
                        end_y = e.get("y")
                    if x is not None and end_x is not None:
                        if (end_x - x) >= 10 or (x < 66.7 and end_x >= 83):
                            team_progressive[cid] += 1
                        if x < 83 and end_x >= 83 and end_y is not None and 21 <= end_y <= 79:
                            team_box_entries[cid] += 1
                # PPDA: opposition (=other team) passes in their own <60% of pitch
                if x is not None and x <= 60:
                    team_ppda_opp_passes[other] += 1
            if tid in (7, 8, 44) and x is not None and x >= 40:  # tackle/interception/aerial in opp build-up zone
                team_ppda_def_actions[other] += 1
            if tid in (49, 52) and x is not None and x >= 66.7:
                team_high_regains[cid] += 1

    teams = list(team_minutes.keys())
    out = {}
    for cid in teams:
        mins = max(team_minutes[cid], 1.0)
        games = mins / 90.0
        out[cid] = {
            "pass_pct": (team_passes_ok[cid] / team_passes[cid] * 100) if team_passes[cid] else 0.0,
            "box_entries_90": team_box_entries[cid] / games,
            "progressive_90": team_progressive[cid] / games,
            "ppda": (team_ppda_opp_passes[cid] / team_ppda_def_actions[cid]) if team_ppda_def_actions[cid] else 20.0,
            "high_regains_90": team_high_regains[cid] / games,
        }

    dshots = danger.copy()
    dshots["match_id"] = dshots["match_file"]
    xg_for = dshots.groupby("contestant_id")["xg"].sum()
    shots_for = dshots.groupby("contestant_id").size()
    match_teams = dshots.groupby("match_file")["contestant_id"].agg(lambda s: sorted(set(s)))
    xg_against = collections.Counter()
    for match_file, cids in match_teams.items():
        if len(cids) != 2:
            continue
        sub = dshots[dshots["match_file"] == match_file]
        for cid in cids:
            other_sum = sub[sub["contestant_id"] != cid]["xg"].sum()
            xg_against[cid] += other_sum

    for cid in teams:
        games = max(team_minutes[cid] / 90.0, 1.0)
        out.setdefault(cid, {})
        out[cid]["xg_for_90"] = xg_for.get(cid, 0.0) / games
        out[cid]["xga_90"] = xg_against.get(cid, 0.0) / games
        out[cid]["shots_90"] = shots_for.get(cid, 0) / games
    return out


def pct_rank_value(values, value):
    arr = np.array(sorted(values))
    if len(arr) == 0:
        return 50.0
    return float((arr < value).sum() / len(arr) * 100)


def make_radar(league_metrics, cid, out_path):
    axes_def = [
        ("xG For /90", "xg_for_90", False),
        ("xG Against /90", "xga_90", True),
        ("Shots /90", "shots_90", False),
        ("Box Entries /90", "box_entries_90", False),
        ("Progressive /90", "progressive_90", False),
        ("Pass Completion %", "pass_pct", False),
        ("High Regains /90", "high_regains_90", False),
        ("PPDA (pressing)", "ppda", True),
    ]
    labels = [a[0] for a in axes_def]
    psv_pct = []
    for _, key, invert in axes_def:
        vals = [m[key] for m in league_metrics.values() if key in m]
        v = league_metrics[cid].get(key, np.nan)
        p = pct_rank_value(vals, v)
        if invert:
            p = 100 - p
        psv_pct.append(p)

    n = len(labels)
    angles = [i / n * 2 * math.pi for i in range(n)]
    angles += angles[:1]
    values = psv_pct + psv_pct[:1]

    fig = plt.figure(figsize=(11, 11), facecolor=BG)
    _header(fig, "Netherlands-Sweden report style · radar",
            "PSV Eindhoven — Season Radar vs Eredivisie Average",
            "Percentile rank across all 18 teams · 50 = league average")
    ax = fig.add_axes([0.10, 0.06, 0.80, 0.78], polar=True)
    ax.set_facecolor(BG)
    ax.set_theta_offset(math.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, color=TEXT, fontsize=10.5)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(["20", "40", "60", "80", "100"], color=MUTED, fontsize=7.5)
    ax.set_ylim(0, 100)
    ax.spines["polar"].set_color(GRID)
    ax.grid(color=GRID, alpha=0.6)

    ax.plot(angles, [50] * (n + 1), color=MUTED, lw=1.0, ls="--", alpha=0.6)
    ax.plot(angles, values, color=PSV_COLOR, lw=2.4, zorder=3)
    ax.fill(angles, values, color=PSV_COLOR, alpha=0.28, zorder=2)
    ax.scatter(angles[:-1], values[:-1], color=PSV_COLOR, edgecolors=TEXT, s=45, zorder=4)

    add_logo(fig)
    _save(fig, out_path)


# ---------------------------------------------------------------------------
# 4) Points trajectory
# ---------------------------------------------------------------------------
def make_points_trajectory(matches, out_path):
    fig = plt.figure(figsize=(16, 8.5), facecolor=BG)
    _header(fig, "Netherlands-Sweden report style · season story",
            "PSV Eindhoven — Points Trajectory", "Cumulative points across the 34-game 2025/26 season")
    ax = fig.add_axes([0.06, 0.14, 0.90, 0.66])
    ax.set_facecolor(PANEL)

    cum = np.cumsum([m["points"] for m in matches])
    x = np.arange(1, len(matches) + 1)
    league_pace_ppg = 1.35
    ax.plot(x, league_pace_ppg * x, color=MUTED, lw=1.2, ls="--", alpha=0.7, label="League-average pace")
    ax.plot(x, cum, color=PSV_COLOR, lw=2.6, zorder=3, label="PSV cumulative points")
    colors = {"W": GREEN, "D": MUTED, "L": RED}
    ax.scatter(x, cum, color=[colors[m["result"]] for m in matches], edgecolors=TEXT, s=55, zorder=4)

    ax.set_xlim(0.5, len(matches) + 0.5)
    ax.set_xlabel("Matchday", color=MUTED, fontsize=10)
    ax.set_ylabel("Cumulative points", color=MUTED, fontsize=10)
    ax.tick_params(colors=MUTED)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.grid(True, axis="y", color=GRID, alpha=0.5, lw=0.6)
    ax.legend(loc="upper left", facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT, fontsize=9.5)

    final_pts = cum[-1]
    w = sum(1 for m in matches if m["result"] == "W")
    d = sum(1 for m in matches if m["result"] == "D")
    l = sum(1 for m in matches if m["result"] == "L")
    gf = sum(m["gf"] for m in matches)
    ga = sum(m["ga"] for m in matches)
    fig.text(0.06, 0.855, f"Final: {final_pts} pts  ·  {w}W {d}D {l}L  ·  GF {gf} – GA {ga}  ·  GD {gf-ga:+d}",
             color=TEXT, fontsize=12.5, weight="bold")

    add_logo(fig)
    _save(fig, out_path)


# ---------------------------------------------------------------------------
# 5) Home / away scout card
# ---------------------------------------------------------------------------
def make_home_away_card(matches, danger, cid, out_path):
    home = [m for m in matches if m["is_home"]]
    away = [m for m in matches if not m["is_home"]]
    home_files = {m["file"].split("/")[-1] for m in home}
    away_files = {m["file"].split("/")[-1] for m in away}
    d = danger[danger["contestant_id"] == cid]
    xg_home = d[d["match_file"].isin(home_files)]["xg"].sum()
    xg_away = d[d["match_file"].isin(away_files)]["xg"].sum()

    fig = plt.figure(figsize=(16, 9), facecolor=BG)
    _header(fig, "Netherlands-Sweden report style · scout card",
            "PSV Eindhoven — Home vs Away", "Full 2025/26 Eredivisie season split")
    ax = fig.add_axes([0.04, 0.08, 0.92, 0.76])
    ax.axis("off")

    def record(games):
        w = sum(1 for m in games if m["result"] == "W")
        d_ = sum(1 for m in games if m["result"] == "D")
        l = sum(1 for m in games if m["result"] == "L")
        gf = sum(m["gf"] for m in games)
        ga = sum(m["ga"] for m in games)
        pts = sum(m["points"] for m in games)
        return w, d_, l, gf, ga, pts

    for col, (label, games, xg, color) in enumerate([
        ("HOME", home, xg_home, PSV_COLOR),
        ("AWAY", away, xg_away, LEAGUE_COLOR),
    ]):
        x0 = 0.03 + col * 0.50
        w, d_, l, gf, ga, pts = record(games)
        ax.add_patch(Rectangle((x0, 0.02), 0.45, 0.90, transform=ax.transAxes, facecolor=PANEL, edgecolor=GRID, lw=0.9))
        ax.text(x0 + 0.03, 0.86, label, color=color, fontsize=20, weight="bold", transform=ax.transAxes)
        ax.text(x0 + 0.03, 0.80, f"{len(games)} matches", color=MUTED, fontsize=10, transform=ax.transAxes)
        rows = [
            ("Record (W-D-L)", f"{w}-{d_}-{l}"),
            ("Points", f"{pts}  ({pts/max(len(games),1):.2f} ppg)"),
            ("Goals For", f"{gf}  ({gf/max(len(games),1):.2f}/gm)"),
            ("Goals Against", f"{ga}  ({ga/max(len(games),1):.2f}/gm)"),
            ("Goal Difference", f"{gf-ga:+d}"),
            ("Total xG", f"{xg:.2f}  ({xg/max(len(games),1):.2f}/gm)"),
        ]
        for i, (lab, val) in enumerate(rows):
            y = 0.66 - i * 0.10
            ax.text(x0 + 0.03, y, lab, color=MUTED, fontsize=11.5, transform=ax.transAxes)
            ax.text(x0 + 0.42, y, val, color=TEXT, fontsize=13, weight="bold", ha="right", transform=ax.transAxes)

    add_logo(fig)
    _save(fig, out_path)


# ---------------------------------------------------------------------------
# 6) Results strip
# ---------------------------------------------------------------------------
def make_results_strip(matches, out_path):
    n = len(matches)
    cols = 17
    rows = math.ceil(n / cols)
    fig = plt.figure(figsize=(17, 2.1 * rows + 2), facecolor=BG)
    _header(fig, "Netherlands-Sweden report style · form guide",
            "PSV Eindhoven — Season Results Strip", "Chronological, matchday 1 to 34")
    ax = fig.add_axes([0.02, 0.05, 0.96, 0.72])
    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)
    ax.invert_yaxis()
    ax.axis("off")
    colors = {"W": GREEN, "D": "#5b6b7a", "L": RED}
    for i, m in enumerate(matches):
        r, c = divmod(i, cols)
        color = colors[m["result"]]
        ax.add_patch(Rectangle((c + 0.05, r + 0.05), 0.9, 0.9, facecolor=color, edgecolor=BG, lw=1.5))
        venue = "H" if m["is_home"] else "A"
        opp_short = clean_name(m["opponent"])[:11]
        ax.text(c + 0.5, r + 0.32, f"{m['result']} {m['gf']}-{m['ga']}", ha="center", va="center",
                color="white", fontsize=8.5, weight="bold")
        ax.text(c + 0.5, r + 0.58, f"{opp_short} ({venue})", ha="center", va="center",
                color="#0d1117", fontsize=6.3)
        ax.text(c + 0.5, r + 0.80, f"MD{i+1}", ha="center", va="center", color="#0d1117", fontsize=5.6, alpha=0.75)

    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], marker="s", color="none", markerfacecolor=GREEN, markersize=13, label="Win"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor="#5b6b7a", markersize=13, label="Draw"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor=RED, markersize=13, label="Loss"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, facecolor=PANEL, edgecolor=GRID,
               labelcolor=TEXT, fontsize=10, bbox_to_anchor=(0.5, 0.01))
    add_logo(fig)
    _save(fig, out_path)


def main():
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/tmp/psv_match_style")
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(glob.glob(f"{DATA_DIR}/*.json"))
    team_to_cid = build_team_map(files)
    match_key = next((full for full in team_to_cid if TEAM_NAME.lower() in full.lower()), None)
    if match_key is None:
        raise SystemExit(f"Team '{TEAM_NAME}' not found")
    cid = team_to_cid[match_key]

    matches = load_season_matches(files, TEAM_NAME, cid)
    print(f"PSV matches found: {len(matches)}")
    danger = load_danger()
    danger["contestant_id"] = danger["contestant_id"].astype(str)

    make_shot_map(danger, cid, matches, out_dir / "01_shot_map_season_psv.png")
    make_danger_heatmap(danger, cid, matches, out_dir / "02_danger_heatmap_season_psv.png")

    print("Building league-wide metrics (single pass over all matches)...")
    league_metrics = build_league_metrics(files, danger)
    make_radar(league_metrics, cid, out_dir / "03_radar_vs_league_psv.png")

    make_points_trajectory(matches, out_dir / "04_points_trajectory_psv.png")
    make_home_away_card(matches, danger, cid, out_dir / "05_home_away_scoutcard_psv.png")
    make_results_strip(matches, out_dir / "06_results_strip_psv.png")

    print("All match-style visuals done.")


if __name__ == "__main__":
    main()

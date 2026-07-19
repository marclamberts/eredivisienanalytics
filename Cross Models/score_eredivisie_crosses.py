"""
Apply the six Ecuador-2026-trained cross models (model_cross_completion,
model_cross_chance_creation, model_cross_goal_contribution,
model_cross_defended, model_cross_delivery_value,
model_cross_outcome_multiclass) to the Eredivisie 2025/26 event data.

Feature engineering matches the training pipeline as reverse-engineered
from model_meta.pkl's qualifier_mapping and open_play_cross_events_scored.csv:
  - length/angle read directly from Opta qualifiers 212/213 (matches the
    scored CSV exactly: length = sqrt((dx*1.05)^2 + (dy*0.68)^2) m,
    angle = atan2(dy*0.68, dx*1.05) -- used as a fallback if the
    qualifiers are absent on a given event).
  - all other derived distance features (dist_to_goal_start/end,
    byline_proximity, delivery_depth, lateral_shift,
    end_zone_y_dist_center) use the same 1.05/0.68 metre scaling for
    consistency with the confirmed length formula.
  - pull_back = Opta qualifier 195; phase = qualifier 23 (FAST_BREAK)
    vs regular; body_part = qualifiers 72/20/other; wide_channel =
    left/right by origin y; period = raw periodId (int), matching the
    fitted OneHotEncoder categories exactly.
  - "open play cross" = qualifier 2 (CROSS) present, without qualifier
    5 (FREEKICK_TAKEN), 6 (CORNER_TAKEN), or 160 (THROWIN_SETPIECE).

Usage: python3 score_eredivisie_crosses.py [out_dir]
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
import joblib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from adjustText import adjust_text

DATA_DIR = "/Users/marclamberts/Event data/Eredivisie"
MODELS_DIR = "/Users/marclamberts/Event data/Ecuador 2026/Cross Models/models"
LOGO_PATH = "/Users/marclamberts/Downloads/Waltzing Analytics Logo Type.png"

BG = "#0d1117"
GRID_COLOR = "#232a35"
TEXT_MAIN = "#e6e9ee"
TEXT_SUB = "#9aa4b2"
GREEN = "#4a9e5c"
RED = "#e0765c"
BLUE = "#2f8fd1"

CROSS_QID, FREEKICK_QID, CORNER_QID, THROWIN_QID = 2, 5, 6, 160
HEAD_QID, RIGHT_QID, LEFT_QID = 15, 20, 72
FAST_BREAK_QID, PULL_BACK_QID, RELATED_EVENT_QID = 23, 195, 55
LENGTH_QID, ANGLE_QID = 212, 213
PASS_END_X_QID, PASS_END_Y_QID = 140, 141

SHOT_TYPES = {13, 14, 15, 16}
CLEARANCE_TYPES = {8, 12}  # interception, clearance

FEATURE_COLS = ['x', 'y', 'end_x', 'end_y', 'length', 'angle', 'minute',
                'dist_to_goal_start', 'dist_to_goal_end', 'byline_proximity',
                'delivery_depth', 'lateral_shift', 'end_zone_y_dist_center',
                'pull_back', 'body_part', 'phase', 'wide_channel', 'period']

X_SCALE, Y_SCALE = 1.05, 0.68
PREFIX_RE = re.compile(r"^(CSD|CD|CS|SD)\s+")


def clean_name(name):
    return PREFIX_RE.sub("", name)


def add_logo(fig, width=0.11, margin=0.014):
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


def qmap(e):
    return {q["qualifierId"]: q.get("value") for q in e.get("qualifier", []) or []}


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


def extract_open_play_crosses(files):
    team_to_cid = build_team_map(files)
    cid_to_team = {v: k for k, v in team_to_cid.items()}

    rows = []
    for fn in files:
        base = fn.split("/")[-1]
        with open(fn) as f:
            data = json.load(f)
        events = [e for e in data["event"] if e.get("periodId") in (1, 2)]
        events.sort(key=lambda e: (e["periodId"], e.get("timeMin", 0), e.get("timeSec", 0), e.get("eventId", 0)))

        # RELATED_EVENT_ID (qualifier 55) lives on the SHOT, pointing back
        # to the assisting event's eventId (not its "id"). Build the
        # reverse index once per match: assist eventId -> (shot, goal).
        led_to_shot, led_to_goal = set(), set()
        for e in events:
            if e.get("typeId") not in SHOT_TYPES:
                continue
            sq = qmap(e)
            rel = sq.get(RELATED_EVENT_QID)
            if rel is None:
                continue
            try:
                rel = int(rel)
            except (TypeError, ValueError):
                continue
            led_to_shot.add(rel)
            if e.get("typeId") == 16:
                led_to_goal.add(rel)

        for idx, e in enumerate(events):
            if e.get("typeId") != 1:
                continue
            q = qmap(e)
            if CROSS_QID not in q:
                continue
            if FREEKICK_QID in q or CORNER_QID in q or THROWIN_QID in q:
                continue

            x, y = e.get("x"), e.get("y")
            if x is None or y is None:
                continue
            end_x = q.get(PASS_END_X_QID)
            end_y = q.get(PASS_END_Y_QID)
            try:
                end_x, end_y = float(end_x), float(end_y)
            except (TypeError, ValueError):
                continue

            dx_m = (end_x - x) * X_SCALE
            dy_m = (end_y - y) * Y_SCALE
            length = q.get(LENGTH_QID)
            angle = q.get(ANGLE_QID)
            try:
                length = float(length)
            except (TypeError, ValueError):
                length = math.hypot(dx_m, dy_m)
            try:
                angle = float(angle)
            except (TypeError, ValueError):
                angle = math.atan2(dy_m, dx_m)

            # Raw 0-100 Opta units, unscaled -- must match the Ecuador training
            # pipeline's feature engineering exactly (it never applies the
            # 1.05/0.68 metre scaling to these derived features, only x/y/end_x/
            # end_y themselves are in raw units there too). The X_SCALE/Y_SCALE
            # metre conversion is kept only for the length/angle fallback below,
            # since that mirrors qualifiers 212/213 which are genuinely metric.
            dist_to_goal_start = math.hypot(100 - x, 50 - y)
            dist_to_goal_end = math.hypot(100 - end_x, 50 - end_y)
            byline_proximity = 100 - x
            delivery_depth = end_x - x
            lateral_shift = abs(end_y - y)
            end_zone_y_dist_center = abs(end_y - 50)
            pull_back = 1 if PULL_BACK_QID in q else 0
            body_part = "left_foot" if LEFT_QID in q else ("right_foot" if RIGHT_QID in q else "other")
            phase = "fast_break" if FAST_BREAK_QID in q else "regular"
            wide_channel = "left" if y < 50 else "right"
            period = int(e.get("periodId"))
            minute = e.get("timeMin", 0) + e.get("timeSec", 0) / 60.0

            outcome = 1 if e.get("outcome") == 1 else 0
            cid = e.get("contestantId")
            team = cid_to_team.get(cid, cid)

            cross_event_id = e.get("eventId")
            shot_created = 1 if cross_event_id in led_to_shot else 0
            goal_created = 1 if cross_event_id in led_to_goal else 0

            cleared = 0
            if idx + 1 < len(events):
                nxt = events[idx + 1]
                if nxt.get("contestantId") != cid and nxt.get("typeId") in CLEARANCE_TYPES:
                    cleared = 1

            rows.append({
                "match_file": base, "team": team, "player": e.get("playerName", "?"),
                "period": period, "minute": minute,
                "x": x, "y": y, "end_x": end_x, "end_y": end_y,
                "length": length, "angle": angle,
                "dist_to_goal_start": dist_to_goal_start, "dist_to_goal_end": dist_to_goal_end,
                "byline_proximity": byline_proximity, "delivery_depth": delivery_depth,
                "lateral_shift": lateral_shift, "end_zone_y_dist_center": end_zone_y_dist_center,
                "pull_back": pull_back, "body_part": body_part, "phase": phase,
                "wide_channel": wide_channel, "outcome": outcome,
                "shot_created": shot_created, "goal_created": goal_created, "cleared": cleared,
            })

    return pd.DataFrame(rows), team_to_cid


def score(df, models):
    X = df[FEATURE_COLS].copy()
    df["pred_cross_completion"] = models["cross_completion"].predict_proba(X)[:, 1]
    df["pred_cross_chance_creation"] = models["cross_chance_creation"].predict_proba(X)[:, 1]
    df["pred_cross_goal_contribution"] = models["cross_goal_contribution"].predict_proba(X)[:, 1]
    df["pred_cross_defended"] = models["cross_defended"].predict_proba(X)[:, 1]
    df["pred_cross_delivery_value"] = models["cross_delivery_value"].predict(X)
    mc = models["cross_outcome_multiclass"]
    df["pred_cross_outcome_multiclass"] = mc.predict(X)
    return df


def make_top_crossers(df, out_path, min_crosses=15):
    g = df.groupby(["player", "team"]).agg(
        n=("pred_cross_delivery_value", "size"),
        threat=("pred_cross_delivery_value", "sum"),
    ).reset_index()
    g = g[g["n"] >= min_crosses].sort_values("threat", ascending=False).head(15)
    g = g.iloc[::-1]

    fig, ax = plt.subplots(figsize=(15, 11))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    labels = [f"{row.player}  ({clean_name(row.team)})" for row in g.itertuples()]
    ax.barh(labels, g["threat"], color=BLUE, height=0.65, zorder=3)
    for i, v in enumerate(g["threat"]):
        ax.text(v + 0.05, i, f"{v:.1f}", va="center", ha="left", color=TEXT_MAIN, fontsize=10.5, fontweight="bold")

    ax.set_xlabel("Cross threat (sum of predicted delivery value)", fontsize=11, color=TEXT_MAIN, fontweight="bold", labelpad=10)
    ax.tick_params(axis="x", colors=TEXT_SUB, labelsize=10)
    ax.tick_params(axis="y", colors=TEXT_MAIN, labelsize=11)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(True, axis="x", color=GRID_COLOR, linewidth=0.6, alpha=0.6, zorder=0)

    fig.text(0.06, 0.965, "Top Crossers", fontsize=24, fontweight="bold", color="white")
    fig.text(0.06, 0.938, f"By cross threat  ·  min {min_crosses} open-play crosses  ·  Eredivisie 2025/26  ·  Season",
             fontsize=12, color=TEXT_SUB)
    fig.text(0.06, 0.012, "Data via Opta | Eredivisie 2025/26 event data · open-play crosses only "
             "(excludes corners/free kicks/throw-ins) · models trained on Ecuador 2026 data, applied unchanged",
             fontsize=7.6, color="#6b7684")
    fig.text(0.98, 0.012, "Marc Lamberts", fontsize=9, ha="right", color="#6b7684", style="italic")
    fig.subplots_adjust(left=0.24, right=0.96, top=0.90, bottom=0.07)
    add_logo(fig)
    fig.savefig(out_path, dpi=200, facecolor=BG)
    plt.close(fig)
    print("Saved:", out_path)


def make_team_attack_vs_defend(df, out_path):
    attack = df.groupby("team")["pred_cross_delivery_value"].sum()

    faced = []
    for fn, sub in df.groupby("match_file"):
        m = re.match(r"\d{4}-\d{2}-\d{2}_(.+) - (.+)\.json$", fn)
        if not m:
            continue
        home, away = m.groups()
        for attacking_team, defending_team in [(home, away), (away, home)]:
            s = sub[sub["team"] == attacking_team]
            if len(s):
                faced.append({"defending_team": defending_team, "cleared": s["cleared"].sum(),
                              "expected_cleared": s["pred_cross_defended"].sum()})
    faced_df = pd.DataFrame(faced)
    defend = faced_df.groupby("defending_team").agg(
        cleared=("cleared", "sum"), expected=("expected_cleared", "sum")).reset_index()
    defend["added_value"] = defend["cleared"] - defend["expected"]

    plot_df = pd.DataFrame({"team": attack.index, "attack": attack.values}).merge(
        defend.rename(columns={"defending_team": "team"}), on="team", how="inner")

    fig, ax = plt.subplots(figsize=(15, 12.5))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    # The model's clearance probability is trained on Ecuador 2026 and
    # does not transfer with a well-calibrated absolute zero point to
    # the Eredivisie (every team came out negative vs. a raw 0 line --
    # predicted clear rate 0.502 vs. actual 0.418 league-wide). Center
    # the reference line on the Eredivisie league median instead, so
    # the chart reads as "better/worse than this league's peers" rather
    # than a mis-calibrated absolute claim.
    x_med = plot_df["attack"].median()
    y_med = plot_df["added_value"].median()
    ax.axvline(x_med, color=TEXT_SUB, linewidth=1.0, linestyle=(0, (4, 3)), alpha=0.6, zorder=1)
    ax.axhline(y_med, color=TEXT_SUB, linewidth=1.0, linestyle=(0, (4, 3)), alpha=0.6, zorder=1)

    colors = [GREEN if (row.attack >= x_med and row.added_value >= y_med) else
             (RED if (row.attack < x_med and row.added_value < y_med) else "#e0b34a")
             for row in plot_df.itertuples()]
    ax.scatter(plot_df["attack"], plot_df["added_value"], s=140, color=colors,
              edgecolors=BG, linewidths=1.2, zorder=3)

    texts = []
    for row in plot_df.itertuples():
        texts.append(ax.text(row.attack, row.added_value, clean_name(row.team),
                             fontsize=10.5, color=TEXT_MAIN, zorder=4))
    adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle="-", color="#6b7684", lw=0.6))

    ax.set_xlabel("Attacking cross threat (season total)", fontsize=11.5, color=TEXT_MAIN, fontweight="bold", labelpad=10)
    ax.set_ylabel("Defensive clearance added value, vs. league median\n(crosses cleared − expected)", fontsize=11.5, color=TEXT_MAIN, fontweight="bold", labelpad=10)
    ax.tick_params(colors=TEXT_SUB, labelsize=10)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(True, color=GRID_COLOR, linewidth=0.6, alpha=0.6, zorder=0)

    fig.text(0.08, 0.965, "Team Cross Profile", fontsize=24, fontweight="bold", color="white")
    fig.text(0.08, 0.938, "Creating danger vs. defending it  ·  Eredivisie 2025/26  ·  Season", fontsize=12, color=TEXT_SUB)
    fig.text(0.08, 0.030, "Data via Opta | Eredivisie 2025/26 event data · open-play crosses only (excludes "
             "corners/free kicks/throw-ins) · shot linked via qualifier 55 · models trained on Ecuador 2026 "
             "data, applied unchanged", fontsize=7.4, color="#6b7684")
    fig.text(0.08, 0.012, "Dashed lines = Eredivisie league median, not zero (model's clearance probability "
             "runs high vs. actual on this league: 0.50 predicted vs. 0.42 actual)",
             fontsize=7.4, color="#6b7684")
    fig.text(0.98, 0.030, "Marc Lamberts", fontsize=9, ha="right", color="#6b7684", style="italic")
    fig.subplots_adjust(left=0.11, right=0.96, top=0.90, bottom=0.09)
    add_logo(fig)
    fig.savefig(out_path, dpi=200, facecolor=BG)
    plt.close(fig)
    print("Saved:", out_path)


def main():
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/Users/marclamberts/Event data/Eredivisie/Cross Models")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading models...")
    models = {}
    for name in ["cross_completion", "cross_chance_creation", "cross_goal_contribution",
                "cross_defended", "cross_delivery_value", "cross_outcome_multiclass"]:
        models[name] = joblib.load(f"{MODELS_DIR}/model_{name}.pkl")

    files = sorted(glob.glob(f"{DATA_DIR}/*.json"))
    print(f"Parsing {len(files)} Eredivisie matches for open-play crosses...")
    df, team_to_cid = extract_open_play_crosses(files)
    print(f"Extracted {len(df)} open-play cross events across {df['team'].nunique()} teams.")

    df = score(df, models)

    csv_path = out_dir / "eredivisie_open_play_cross_events_scored.csv"
    df.to_csv(csv_path, index=False)
    print("Saved:", csv_path)

    print("\nSanity check vs. training positive rates:")
    print(f"  cross_completion: actual outcome=1 rate {df['outcome'].mean():.3f}, "
          f"mean predicted {df['pred_cross_completion'].mean():.3f}")
    print(f"  cross_chance_creation: actual shot_created rate {df['shot_created'].mean():.3f}, "
          f"mean predicted {df['pred_cross_chance_creation'].mean():.3f}")
    print(f"  cross_goal_contribution: actual goal_created rate {df['goal_created'].mean():.3f}, "
          f"mean predicted {df['pred_cross_goal_contribution'].mean():.3f}")
    print(f"  cross_defended: actual cleared rate {df['cleared'].mean():.3f}, "
          f"mean predicted {df['pred_cross_defended'].mean():.3f}")

    make_top_crossers(df, out_dir / "01_top_crossers_eredivisie.png")
    make_team_attack_vs_defend(df, out_dir / "02_team_attack_vs_defend_eredivisie.png")

    print("Done.")


if __name__ == "__main__":
    main()

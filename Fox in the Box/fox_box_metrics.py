"""
Fox in the Box -- Step 1: build the 16 raw metrics.

A "fox in the box" is the classic penalty-area poacher: a player whose game
is concentrated in and around the six-yard box, who needs few touches to get
a shot away, and who converts presence into goals at a clinical rate. This
script turns the season's Opta-style event data (Events/*.json), the shot
model output (Danger/all_eredivisie_danger_models.csv) and the already-computed
player minutes (GDA/gda_player_summary.csv) into 16 per-player metrics split
across four pillars:

  A. Box presence      -- how much of their game happens in the box
  B. Shot volume & instinct -- how eager they are to shoot once there
  C. Finishing output   -- raw goal production
  D. Clinical efficiency -- quality and conversion of the chances they get

Only players with >= MIN_MINUTES minutes are kept (same qualification bar
used elsewhere in this repo, e.g. netlify-app's scouting comparison group).

Usage: python3 fox_box_metrics.py [out_dir]
"""
import sys
import glob
import json
import collections
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
EVENTS_DIR = ROOT / "Events"
DANGER_PATH = ROOT / "Danger" / "all_eredivisie_danger_models.csv"
GDA_PATH = ROOT / "GDA" / "gda_player_summary.csv"
XT_TEAM_PATH = ROOT / "xT" / "xt_team_summary.csv"

MIN_MINUTES = 450

BOX_X = 83.0
BOX_Y_LO, BOX_Y_HI = 21.1, 78.9
BIG_CHANCE_XG = 0.2

# Event typeIds that never represent an on-ball touch (formation/admin/card/
# substitution events etc.) -- same set used in relationism_index.py.
NON_TOUCH_TYPES = {17, 18, 19, 27, 28, 30, 32, 34, 37, 40, 43, 58, 65, 70, 71, 79, 84}
AERIAL_TYPE = 44


def in_box(x, y):
    return x is not None and y is not None and x >= BOX_X and BOX_Y_LO <= y <= BOX_Y_HI


def dist_to_goal_m(x, y):
    dx = (100.0 - x) / 100 * 105.0
    dy = (50.0 - y) / 100 * 68.0
    return (dx ** 2 + dy ** 2) ** 0.5


def load_team_names():
    df = pd.read_csv(XT_TEAM_PATH)
    return dict(zip(df["contestant_id"], df["team_name"]))


def load_minutes():
    """One row per player, minutes summed across any mid-season club stints;
    name/team taken from whichever stint carries the most minutes."""
    df = pd.read_csv(GDA_PATH)
    minutes = {}
    for row in df.itertuples():
        cur = minutes.get(row.player_id)
        if cur is None:
            minutes[row.player_id] = {
                "player_name": row.player_name,
                "contestant_id": row.contestant_id,
                "minutes": float(row.minutes),
                "_main_stint_minutes": float(row.minutes),
            }
        else:
            cur["minutes"] += float(row.minutes)
            if float(row.minutes) > cur["_main_stint_minutes"]:
                cur["_main_stint_minutes"] = float(row.minutes)
                cur["player_name"] = row.player_name
                cur["contestant_id"] = row.contestant_id
    return minutes


def extract_touches_and_aerials(files):
    touches_total = collections.defaultdict(int)
    touches_box = collections.defaultdict(int)
    aerial_box_won = collections.defaultdict(int)
    names = {}

    for fn in files:
        with open(fn) as f:
            data = json.load(f)
        for e in data.get("event", []):
            pid = e.get("playerId")
            if not pid:
                continue
            type_id = e.get("typeId")
            x, y = e.get("x"), e.get("y")

            if type_id == AERIAL_TYPE:
                if e.get("outcome") == 1 and in_box(x, y):
                    aerial_box_won[pid] += 1

            if type_id in NON_TOUCH_TYPES:
                continue
            if x is None or y is None or (x == 0 and y == 0):
                continue

            touches_total[pid] += 1
            if in_box(x, y):
                touches_box[pid] += 1
            names[pid] = e.get("playerName", names.get(pid))

    return touches_total, touches_box, aerial_box_won, names


def load_shot_stats():
    df = pd.read_csv(DANGER_PATH)
    df = df[df["is_penalty"] == 0].copy()
    df["is_box"] = df.apply(lambda r: in_box(r["x"], r["y"]), axis=1)
    df["dist_m"] = df.apply(lambda r: dist_to_goal_m(r["x"], r["y"]), axis=1)
    df["is_big_chance"] = df["xg"] >= BIG_CHANCE_XG

    g = df.groupby("player_id").agg(
        player_name=("player_name", "first"),
        contestant_id=("contestant_id", "first"),
        shots=("xg", "size"),
        shots_box=("is_box", "sum"),
        shots_on_target=("is_on_target", "sum"),
        goals=("is_goal", "sum"),
        xg_sum=("xg", "sum"),
        big_chances=("is_big_chance", "sum"),
        avg_shot_distance_m=("dist_m", "mean"),
    ).reset_index()
    return g


def build_metrics(out_dir):
    files = sorted(glob.glob(str(EVENTS_DIR / "*.json")))
    print(f"Parsing {len(files)} Eredivisie matches for touches and aerials...")
    touches_total, touches_box, aerial_box_won, event_names = extract_touches_and_aerials(files)

    print("Loading shot model output...")
    shot_stats = load_shot_stats()

    print("Loading player minutes...")
    minutes = load_minutes()
    team_names = load_team_names()

    rows = []
    for player_id, info in minutes.items():
        mins = info["minutes"]
        if mins < MIN_MINUTES:
            continue

        shot_row = shot_stats[shot_stats["player_id"] == player_id]
        shots = int(shot_row["shots"].iloc[0]) if len(shot_row) else 0
        shots_box = int(shot_row["shots_box"].iloc[0]) if len(shot_row) else 0
        shots_on_target = int(shot_row["shots_on_target"].iloc[0]) if len(shot_row) else 0
        np_goals = int(shot_row["goals"].iloc[0]) if len(shot_row) else 0
        xg_sum = float(shot_row["xg_sum"].iloc[0]) if len(shot_row) else 0.0
        big_chances = int(shot_row["big_chances"].iloc[0]) if len(shot_row) else 0
        avg_shot_distance_m = float(shot_row["avg_shot_distance_m"].iloc[0]) if len(shot_row) and shots > 0 else None

        t_total = touches_total.get(player_id, 0)
        t_box = touches_box.get(player_id, 0)
        aerial_box = aerial_box_won.get(player_id, 0)

        p90 = 90.0 / mins

        row = {
            "player_id": player_id,
            "player_name": event_names.get(player_id, info["player_name"]),
            "team": team_names.get(info["contestant_id"], "Unknown"),
            "minutes": round(mins, 1),
            # --- raw inputs, kept for transparency / debugging ---
            "touches_total": t_total,
            "touches_box": t_box,
            "aerial_box_won": aerial_box,
            "shots": shots,
            "shots_box": shots_box,
            "shots_on_target": shots_on_target,
            "np_goals": np_goals,
            "xg_sum": round(xg_sum, 3),
            "big_chances": big_chances,
            # --- Pillar A: Box presence ---
            "touches_in_box_p90": round(t_box * p90, 3),
            "box_touch_share_pct": round(t_box / t_total * 100, 2) if t_total else 0.0,
            "shots_in_box_share_pct": round(shots_box / shots * 100, 2) if shots else 0.0,
            "aerial_win_box_p90": round(aerial_box * p90, 3),
            # --- Pillar B: Shot volume & instinct ---
            "shots_p90": round(shots * p90, 3),
            "shots_on_target_pct": round(shots_on_target / shots * 100, 2) if shots else 0.0,
            "shots_per_box_touch_pct": round(shots / t_box * 100, 2) if t_box else 0.0,
            "big_chances_p90": round(big_chances * p90, 3),
            # --- Pillar C: Finishing output ---
            "np_goals_p90": round(np_goals * p90, 3),
            "goals_per_shot_pct": round(np_goals / shots * 100, 2) if shots else 0.0,
            "goals_minus_xg_p90": round((np_goals - xg_sum) * p90, 3),
            "goals_per_box_touch_pct": round(np_goals / t_box * 100, 2) if t_box else 0.0,
            # --- Pillar D: Clinical efficiency ---
            "xg_per_shot": round(xg_sum / shots, 4) if shots else 0.0,
            "goals_per_xg_ratio": round(np_goals / xg_sum, 3) if xg_sum > 0 else 0.0,
            "goals_per_shot_on_target_pct": round(np_goals / shots_on_target * 100, 2) if shots_on_target else 0.0,
            "avg_shot_distance_m": round(avg_shot_distance_m, 2) if avg_shot_distance_m is not None else None,
        }
        rows.append(row)

    df = pd.DataFrame(rows).sort_values("np_goals_p90", ascending=False)

    # avg_shot_distance_m is undefined (no shots) for a handful of eligible
    # players -- backfill with the pool average so downstream z-scoring
    # doesn't have to special-case missing values.
    pool_avg_dist = df["avg_shot_distance_m"].mean()
    df["avg_shot_distance_m"] = df["avg_shot_distance_m"].fillna(round(pool_avg_dist, 2))

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "fox_box_metrics.csv"
    df.to_csv(csv_path, index=False)
    print("Saved:", csv_path)

    json_path = out_dir / "fox_box_metrics.json"
    with open(json_path, "w") as f:
        json.dump(df.to_dict(orient="records"), f, indent=2)
    print("Saved:", json_path)

    print(f"\n{len(df)} players with >= {MIN_MINUTES} minutes qualified.")
    return df


if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent
    build_metrics(out)

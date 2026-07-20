"""
Disruption VALUE model - Eredivisie 2025/26
=============================================
build_disruption_model.py's disruption_score is a probability only: the
baseline model's predicted completion probability for the pass that got
broken up. That rewards stopping *likely* passes, but treats a denied
sideways square ball the same as a denied through-ball into the box, as
long as both had the same completion odds. It answers "how surprising was
this denial", not "how much did it matter."

This script adds the missing half using the season's existing Expected
Threat grid (xT/xt_grid_values.csv, already built and used elsewhere in
this repo): for each linked defensive action, look up how much attacking
threat the passer's team stood to gain had the pass reached its intended
end location (positive_xT_added, clipped at 0 like xT/xt_action_values.csv
already does), then:

    disruption_value = disruption_score * positive_xT_added

i.e. expected threat denied = P(pass would have succeeded) x
(attacking value it would have created). A defender who stops a raspy
through-ball into the box scores far higher here than one who mistimes a
challenge on a backwards pass, even if both passes had similar completion
odds -- disruption_score alone can't tell those apart.

xT zone lookup requires the PASSING team's attacking direction (xT is
defined relative to "which way is this team's goal"), computed the same
way as league_disruption_visuals.py's per-team-per-period normalization.

Usage: python3 disruption_value_model.py
Outputs:
  Disruption/all_eredivisie_disruption_values.csv   (every linked action + xT/value columns)
  Disruption/disruption_value_player_summary.csv
  Disruption/disruption_value_team_summary.csv
"""
import glob
import os

import numpy as np
import pandas as pd

import build_disruption_model as bdm
from league_disruption_visuals import compute_attack_directions, make_leaderboard

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
XT_GRID_PATH = os.path.join(bdm.ROOT, "xT", "xt_grid_values.csv")

X_BINS, Y_BINS = 16, 12
X_BIN_W, Y_BIN_W = 100.0 / X_BINS, 100.0 / Y_BINS


def load_xt_grid():
    grid = pd.read_csv(XT_GRID_PATH)
    return grid.set_index("zone")["xT"].to_dict()


def zone_index(x_opta, y_opta):
    xb = np.clip((x_opta // X_BIN_W).astype(int), 0, X_BINS - 1)
    yb = np.clip((y_opta // Y_BIN_W).astype(int), 0, Y_BINS - 1)
    return yb * X_BINS + xb


def main():
    files = sorted(glob.glob(os.path.join(bdm.DATA_DIR, "*.json")))
    xt_by_zone = load_xt_grid()
    print(f"Loaded xT grid: {len(xt_by_zone)} zones ({X_BINS}x{Y_BINS})")

    cid_to_team = bdm.build_global_cid_to_team(files)
    team_to_cid = {team: cid for cid, team in cid_to_team.items()}

    disr = pd.read_csv(os.path.join(OUT_DIR, "all_eredivisie_disruption_models.csv"))
    linked = disr[disr["linked"] == True].copy()  # noqa: E712
    print(f"{len(linked)} linked defensive actions to value")

    print("Computing per-team attacking direction per period "
          "(xT is defined relative to the PASSING team's own goal)...")
    directions = compute_attack_directions()

    passer_cid = linked["linked_pass_team"].map(team_to_cid)
    dir_vals = np.array([
        directions.get((mf, cid, p), 1)
        for mf, cid, p in zip(linked["match_file"], passer_cid, linked["period_id"])
    ])

    # meters -> Opta 0-100 scale, then flip x onto a common "this team attacks toward x=100" frame
    start_x_opta = linked["linked_pass_start_x"] / bdm.X_SCALE
    start_y_opta = linked["linked_pass_start_y"] / bdm.Y_SCALE
    end_x_opta = linked["linked_pass_end_x"] / bdm.X_SCALE
    end_y_opta = linked["linked_pass_end_y"] / bdm.Y_SCALE

    start_x_norm = np.where(dir_vals == 1, start_x_opta, 100 - start_x_opta)
    end_x_norm = np.where(dir_vals == 1, end_x_opta, 100 - end_x_opta)

    linked["start_zone"] = zone_index(start_x_norm, start_y_opta.to_numpy())
    linked["end_zone"] = zone_index(end_x_norm, end_y_opta.to_numpy())
    linked["start_xT"] = linked["start_zone"].map(xt_by_zone)
    linked["end_xT"] = linked["end_zone"].map(xt_by_zone)
    linked["xT_added"] = linked["end_xT"] - linked["start_xT"]
    linked["positive_xT_added"] = linked["xT_added"].clip(lower=0)
    linked["disruption_value"] = linked["disruption_score"] * linked["positive_xT_added"]

    out_cols = ["match_file", "event_id", "period_id", "time_min", "time_sec",
                "contestant_id", "team_name", "player_id", "player_name",
                "action_type", "x", "y", "link_type", "disruption_score",
                "linked_pass_team", "linked_pass_player", "start_xT", "end_xT",
                "positive_xT_added", "disruption_value"]
    linked[out_cols].to_csv(os.path.join(OUT_DIR, "all_eredivisie_disruption_values.csv"),
                            index=False)
    print("Wrote all_eredivisie_disruption_values.csv")

    player_summary = linked.groupby(["player_id", "player_name", "team_name"]).agg(
        matches=("match_file", "nunique"),
        actions_linked=("disruption_value", "size"),
        total_disruption_score=("disruption_score", "sum"),
        total_positive_xT_denied=("positive_xT_added", "sum"),
        total_disruption_value=("disruption_value", "sum"),
        mean_disruption_value=("disruption_value", "mean"),
    ).reset_index()
    player_summary["disruption_value_per90"] = (
        player_summary["total_disruption_value"] / player_summary["matches"])
    # xT deltas are tiny by construction (typical single-action xT ~0.001-0.02);
    # x1000 columns exist purely so the numbers are readable in a table/chart,
    # same idea as xT/xt_player_summary.csv's xT_per_100_actions columns.
    player_summary["total_disruption_value_x1000"] = player_summary["total_disruption_value"] * 1000
    player_summary["disruption_value_per90_x1000"] = player_summary["disruption_value_per90"] * 1000
    player_summary = player_summary.sort_values("total_disruption_value", ascending=False)
    player_summary.to_csv(os.path.join(OUT_DIR, "disruption_value_player_summary.csv"), index=False)

    team_summary = linked.groupby("team_name").agg(
        matches=("match_file", "nunique"),
        actions_linked=("disruption_value", "size"),
        total_disruption_value=("disruption_value", "sum"),
    ).reset_index()
    team_summary["disruption_value_per_match"] = (
        team_summary["total_disruption_value"] / team_summary["matches"])
    team_summary["total_disruption_value_x1000"] = team_summary["total_disruption_value"] * 1000
    team_summary["disruption_value_per_match_x1000"] = team_summary["disruption_value_per_match"] * 1000
    team_summary = team_summary.sort_values("disruption_value_per_match", ascending=False)
    team_summary.to_csv(os.path.join(OUT_DIR, "disruption_value_team_summary.csv"), index=False)

    print("Wrote disruption_value_player_summary.csv + disruption_value_team_summary.csv")
    print("\nTop 10 by disruption VALUE (probability denied x threat it would have created):")
    print(player_summary.head(10)[["player_name", "team_name", "matches", "actions_linked",
                                    "total_disruption_value", "disruption_value_per90"]]
          .to_string(index=False))

    make_leaderboard(
        player_summary, os.path.join(OUT_DIR, "top_disruptors_by_value_leaderboard.png"),
        value_col="total_disruption_value_x1000",
        title="Top Disruptors by Value",
        subtitle="Eredivisie 2025/26  ·  Season  ·  disruption value = P(pass completes) x "
                 "threat (xT) it would have added had it succeeded  ·  x1000 for readability",
        footer="Data via Opta | rewards stopping progressive, threatening passes -- not just "
               "any pass -- using the season's existing xT grid (xT/xt_grid_values.csv)",
        value_fmt="{:.1f}")


if __name__ == "__main__":
    main()

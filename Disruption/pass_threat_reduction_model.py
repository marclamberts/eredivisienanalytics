"""
Pass Threat Reduction metric - Eredivisie 2025/26
====================================================
disruption_value_model.py's disruption_value answers "how much threat did
this defensive action deny" in RAW terms (sum of xT across a season). That
rewards teams/players who simply face more (or more dangerous) opponent
possession, the same way raw tackle counts reward busy defenders over good
ones. This script turns it into a RATE: what share of the threat a team's
opponents actually generated did that team's defense manage to take away?

    pass_threat_reduction_pct = total_threat_denied / total_pass_threat_faced

  - total_pass_threat_faced (the denominator) = sum of positive_xT_added
    over EVERY opponent pass attempt in a match, completed or not, using
    each pass's intended end location (qualifiers 140/141, exactly as
    build_disruption_model.py's pass_features already extracts). This is
    the total attacking value the opponent's passing tried to create
    against that team, not just the passes that got broken up.
  - total_threat_denied (the numerator) = disruption_value_model.py's
    disruption_value, summed -- the threat actually removed via a linked
    defensive action.

Both sides are valued with the SAME season xT grid
(xT/xt_grid_values.csv) and the SAME zone-lookup method
disruption_value_model.py already uses, so the ratio is apples-to-apples
(mixing that grid method with the separate ML action-value model in
xT/xt_action_values.csv -- which only covers completed passes anyway --
would make the ratio meaningless).

Player-level, "how much of the team's total threat faced did I personally
deny" isn't directly comparable across players with very different playing
time, so two versions are reported:
  - pass_threat_reduction_share_pct: this player's share of the TEAM's full
    season total_pass_threat_faced (rewards nothing but total denied value;
    useful for "how much of this team's defending is this one player").
  - pass_threat_reduction_rate_pct: the same share, but with both sides put
    on a per-team-match basis first, so a squad player and a nailed-on
    starter are compared on the same footing.

Usage: python3 pass_threat_reduction_model.py
Requires Disruption/CSV/all_eredivisie_disruption_values.csv and
disruption_value_player_summary.csv to already exist (run
disruption_value_model.py first).
Outputs:
  Disruption/CSV/pass_threat_reduction_team_summary.csv
  Disruption/CSV/pass_threat_reduction_player_summary.csv
  Disruption/Visual - Dark|Light/pass_threat_reduction_team_leaderboard.png
  Disruption/Visual - Dark|Light/pass_threat_reduction_player_leaderboard.png
"""
import glob
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import build_disruption_model as bdm
import disruption_value_model as dvm
import league_disruption_visuals as ldv
from league_disruption_visuals import add_logo, compute_attack_directions, make_leaderboard

CSV_DIR = ldv.CSV_DIR

# build_disruption_model.py's DATA_DIR ("Events/") and interception model's
# ("Events/2025-2026/") are both stale machine-local paths left over from
# development on Marc's laptop; the raw match JSON actually checked into
# this repo lives under "Eredivisie Events/2025-2026". Point bdm at the
# real directory so every bdm.* helper (extract_passes, load_match,
# build_global_cid_to_team) and league_disruption_visuals.compute_attack_directions
# -- which reads bdm.DATA_DIR itself -- resolve to real files.
bdm.DATA_DIR = os.path.join(bdm.ROOT, "Eredivisie Events", "2025-2026")

MIN_ACTIONS = 5  # same qualification floor as disruption_value_single_beeswarm.py
MATCH_RE = re.compile(r"\d{4}-\d{2}-\d{2}_(.+) - (.+)$")


def load_all_passes():
    files = sorted(glob.glob(os.path.join(bdm.DATA_DIR, "*.json")))
    if not files:
        raise SystemExit(f"No match files found in {bdm.DATA_DIR}")
    cid_to_team = bdm.build_global_cid_to_team(files)
    rows = []
    for fn in files:
        basename, events, team_map = bdm.load_match(fn, cid_to_team)
        rows.extend(bdm.extract_passes(basename, events, team_map))
    return pd.DataFrame(rows)


def opponent_map(match_files):
    """basename -> {team_name: opponent_team_name} from the "<date>_<home> -
    <away>" filename convention (same regex build_global_cid_to_team uses)."""
    m = {}
    for basename in match_files:
        mm = MATCH_RE.match(basename)
        if not mm:
            continue
        home, away = mm.group(1), mm.group(2)
        m[(basename, home)] = away
        m[(basename, away)] = home
    return m


def value_all_passes(pass_df):
    """positive_xT_added for every pass attempt, completed or not, using
    the pass's intended end location -- same zone-grid method
    disruption_value_model.py applies to linked defensive actions, just run
    over every pass instead of only the ones that got broken up."""
    xt_by_zone = dvm.load_xt_grid()
    directions = compute_attack_directions()

    dir_vals = np.array([
        directions.get((mf, cid, p), 1)
        for mf, cid, p in zip(pass_df["match_file"], pass_df["contestant_id"], pass_df["period_id"])
    ])

    start_x_opta = pass_df["start_x"] / bdm.X_SCALE
    start_y_opta = pass_df["start_y"] / bdm.Y_SCALE
    end_x_opta = pass_df["end_x"] / bdm.X_SCALE
    end_y_opta = pass_df["end_y"] / bdm.Y_SCALE

    start_x_norm = np.where(dir_vals == 1, start_x_opta, 100 - start_x_opta)
    end_x_norm = np.where(dir_vals == 1, end_x_opta, 100 - end_x_opta)

    start_zone = dvm.zone_index(start_x_norm, start_y_opta.to_numpy())
    end_zone = dvm.zone_index(end_x_norm, end_y_opta.to_numpy())
    start_xT = pd.Series(start_zone).map(xt_by_zone).to_numpy()
    end_xT = pd.Series(end_zone).map(xt_by_zone).to_numpy()
    pass_df = pass_df.copy()
    pass_df["positive_xT_added"] = np.clip(end_xT - start_xT, 0, None)
    return pass_df


def make_team_leaderboard(team_summary, out_path, top_n=18):
    top = team_summary.sort_values("pass_threat_reduction_pct", ascending=False).head(top_n)
    top = top.iloc[::-1]

    fig, ax = plt.subplots(figsize=(12.5, 9.5))
    fig.patch.set_facecolor(ldv.BG)
    ax.set_facecolor(ldv.BG)

    vmax = top["pass_threat_reduction_pct"].max()
    colors = [ldv.GOLD_RAMP(0.35 + 0.55 * (v / vmax)) for v in top["pass_threat_reduction_pct"]]
    ax.barh(range(len(top)), top["pass_threat_reduction_pct"], color=colors, height=0.62, zorder=3)

    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top["team_name"], fontsize=10.5, color=ldv.TEXT_MAIN)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", colors=ldv.TEXT_SUB, labelsize=9)

    for i, (pct, faced) in enumerate(zip(top["pass_threat_reduction_pct"], top["total_pass_threat_faced"])):
        ax.text(pct + vmax * 0.012, i, f"{pct:.2f}%", va="center", fontsize=9.8, color=ldv.LEGEND_TEXT)

    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(ldv.PITCH_LINE)
    ax.grid(axis="x", color=ldv.PITCH_LINE, linewidth=0.6, alpha=0.5, zorder=0)
    ax.set_xlim(0, vmax * 1.18)

    fig.text(0.5, 0.965, "Pass Threat Reduction", fontsize=24, fontweight="bold",
              ha="center", color=ldv.TEXT_MAIN)
    fig.text(0.5, 0.935, "Eredivisie 2025/26  ·  Season  ·  share of opponents' total pass threat "
                          "(xT) that this team's defense denied",
              fontsize=10.5, ha="center", color=ldv.TEXT_SUB)

    fig.text(0.98, 0.012, "Marc Lamberts", fontsize=9.5, ha="right", color=ldv.TEXT_FOOT, style="italic")
    fig.text(0.02, 0.012, "Data via Opta | threat faced = xT every opponent pass attempt (completed "
                          "or not) tried to create; threat denied = xT removed by a linked defensive "
                          "action (Disruption/disruption_value_model.py)",
              fontsize=7.0, color=ldv.TEXT_FOOT)

    fig.subplots_adjust(left=0.24, right=0.95, top=0.90, bottom=0.07)
    add_logo(fig)
    fig.savefig(out_path, dpi=200, facecolor=ldv.BG)
    plt.close(fig)
    print("Saved:", out_path)


def main():
    print(f"Loading all passes from {bdm.DATA_DIR} ...")
    pass_df = load_all_passes()
    print(f"Extracted {len(pass_df)} pass attempts ({pass_df['outcome'].mean():.1%} completed)")

    pass_df = value_all_passes(pass_df)
    opp_map = opponent_map(pass_df["match_file"].unique())
    pass_df["opponent_team"] = [opp_map.get((mf, tn)) for mf, tn in
                                 zip(pass_df["match_file"], pass_df["team_name"])]
    pass_df = pass_df[pass_df["opponent_team"].notna()]

    threat_faced = pass_df.groupby("opponent_team").agg(
        matches=("match_file", "nunique"),
        passes_faced=("positive_xT_added", "size"),
        total_pass_threat_faced=("positive_xT_added", "sum"),
    ).reset_index().rename(columns={"opponent_team": "team_name"})

    dv_team = pd.read_csv(os.path.join(CSV_DIR, "disruption_value_team_summary.csv"))
    team_summary = threat_faced.merge(
        dv_team[["team_name", "total_disruption_value"]], on="team_name", how="left")
    team_summary["total_disruption_value"] = team_summary["total_disruption_value"].fillna(0.0)
    team_summary = team_summary.rename(columns={"total_disruption_value": "total_threat_denied"})
    team_summary["pass_threat_reduction_pct"] = (
        100 * team_summary["total_threat_denied"] / team_summary["total_pass_threat_faced"])
    team_summary["pass_threat_faced_per_match"] = (
        team_summary["total_pass_threat_faced"] / team_summary["matches"])
    team_summary["pass_threat_denied_per_match"] = (
        team_summary["total_threat_denied"] / team_summary["matches"])
    team_summary = team_summary.sort_values("pass_threat_reduction_pct", ascending=False)
    team_summary.to_csv(os.path.join(CSV_DIR, "pass_threat_reduction_team_summary.csv"), index=False)
    print("Wrote pass_threat_reduction_team_summary.csv")
    print("\nPass Threat Reduction % by team:")
    print(team_summary[["team_name", "matches", "total_pass_threat_faced", "total_threat_denied",
                         "pass_threat_reduction_pct"]].to_string(index=False))

    dv_player = pd.read_csv(os.path.join(CSV_DIR, "disruption_value_player_summary.csv"))
    player_summary = dv_player.merge(
        team_summary[["team_name", "matches", "total_pass_threat_faced"]]
        .rename(columns={"matches": "team_matches"}),
        on="team_name", how="left")
    player_summary["pass_threat_reduction_share_pct"] = (
        100 * player_summary["total_disruption_value"] / player_summary["total_pass_threat_faced"])
    player_per_match = player_summary["total_disruption_value"] / player_summary["matches"]
    team_faced_per_match = player_summary["total_pass_threat_faced"] / player_summary["team_matches"]
    player_summary["pass_threat_reduction_rate_pct"] = 100 * player_per_match / team_faced_per_match
    player_summary = player_summary.rename(columns={"total_pass_threat_faced": "team_total_pass_threat_faced"})
    player_summary = player_summary.sort_values("pass_threat_reduction_rate_pct", ascending=False)
    player_summary.to_csv(os.path.join(CSV_DIR, "pass_threat_reduction_player_summary.csv"), index=False)
    print("Wrote pass_threat_reduction_player_summary.csv")

    qualified = player_summary[player_summary["actions_linked"] >= MIN_ACTIONS]
    print(f"\nTop 10 by Pass Threat Reduction rate (>= {MIN_ACTIONS} linked actions):")
    print(qualified.head(10)[["player_name", "team_name", "matches", "actions_linked",
                               "pass_threat_reduction_rate_pct"]].to_string(index=False))

    for theme in ("dark", "light"):
        ldv.set_theme(theme)
        out_dir = ldv.visual_dir(theme)
        make_team_leaderboard(team_summary, os.path.join(out_dir, "pass_threat_reduction_team_leaderboard.png"))
        make_leaderboard(
            qualified, os.path.join(out_dir, "pass_threat_reduction_player_leaderboard.png"),
            value_col="pass_threat_reduction_rate_pct",
            title="Top Pass Threat Reducers",
            subtitle=f"Eredivisie 2025/26  ·  Season  ·  players with ≥{MIN_ACTIONS} linked "
                     "disruptions  ·  threat denied per team-match, as a % of the team's own "
                     "threat-faced-per-match rate",
            footer="Data via Opta | rate = (this player's threat denied / their matches) as a % of "
                   "(their team's total threat faced / team matches) -- comparable across squad "
                   "players and nailed-on starters alike",
            value_fmt="{:.2f}%")


if __name__ == "__main__":
    main()

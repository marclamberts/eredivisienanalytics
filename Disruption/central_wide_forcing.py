"""
Which teams defend the central areas best -- by disrupting passes AND
forcing opponents into the wide areas? Eredivisie 2025/26.

Two signals, combined:
  1. Central disruption value/match (from all_eredivisie_disruption_values.csv,
     same convention as central_defending_leaderboard.py) -- how much threat
     a team denies specifically inside the central third.
  2. Wide-forcing index -- reprocesses every match's full pass log (not just
     disrupted passes) and asks: when team X is defending, what share of the
     OPPONENT's passes start in the central third, vs the league-average
     share? A team that forces opponents wide pushes that share down.
     y_own is normalized to the passing team's own attacking direction
     (compute_attack_directions), so which end a team shoots at doesn't
     matter, exactly as elsewhere in this repo.

A team that both denies central threat AND suppresses opponents' central
passing share is genuinely squeezing the middle of the pitch, not just
racking up actions there.

Usage: python3 central_wide_forcing.py [out_dir]
"""
import glob
import os
import re
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import build_disruption_model as bdm
import league_disruption_visuals as ldv
from league_disruption_visuals import compute_attack_directions, add_logo

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
CENTRAL_LOW, CENTRAL_HIGH = 68 / 3, 2 * 68 / 3
MATCH_RE = re.compile(r"\d{4}-\d{2}-\d{2}_(.+) - (.+)$")


def central_disruption_value_per_match():
    values = pd.read_csv(os.path.join(ldv.CSV_DIR, "all_eredivisie_disruption_values.csv"))
    directions = compute_attack_directions()
    dir_vals = np.array([
        directions.get((mf, cid, p), 1)
        for mf, cid, p in zip(values["match_file"], values["contestant_id"], values["period_id"])
    ])
    values["y_own"] = np.where(dir_vals == 1, values["y"], 68.0 - values["y"])
    central = values[(values["y_own"] >= CENTRAL_LOW) & (values["y_own"] <= CENTRAL_HIGH)].copy()

    matches = pd.read_csv(os.path.join(ldv.CSV_DIR, "disruption_value_team_summary.csv"))[
        ["team_name", "matches"]]
    g = central.groupby("team_name").agg(
        central_actions=("disruption_value", "size"),
        central_value=("disruption_value", "sum"),
    ).reset_index().merge(matches, on="team_name")
    g["central_value_pm_x1000"] = g["central_value"] * 1000 / g["matches"]
    return g[["team_name", "matches", "central_actions", "central_value_pm_x1000"]]


def opponent_central_pass_share():
    """For every match, extract ALL passes (not just disrupted ones), tag
    each with the OPPONENT (the team defending against it), and compute what
    share of the opponent-attempted passes start in the central third."""
    files = sorted(glob.glob(os.path.join(bdm.DATA_DIR, "*.json")))
    cid_to_team = bdm.build_global_cid_to_team(files)
    directions = compute_attack_directions()

    rows = []
    for fn in files:
        basename = os.path.basename(fn).replace(".json", "")
        m = MATCH_RE.match(basename)
        if not m:
            continue
        home_name, away_name = m.group(1), m.group(2)
        _, events, _ = bdm.load_match(fn, cid_to_team)
        passes = bdm.extract_passes(basename, events, cid_to_team)
        for p in passes:
            team = p["team_name"]
            opponent = away_name if team == home_name else (home_name if team == away_name else None)
            if opponent is None:
                continue
            d = directions.get((basename, p["contestant_id"], p["period_id"]), 1)
            y_own = p["start_y"] if d == 1 else 68.0 - p["start_y"]
            rows.append((opponent, CENTRAL_LOW <= y_own <= CENTRAL_HIGH))

    df = pd.DataFrame(rows, columns=["defending_team", "is_central"])
    league_share = df["is_central"].mean()

    g = df.groupby("defending_team").agg(
        opp_passes_faced=("is_central", "size"),
        opp_central_passes=("is_central", "sum"),
    ).reset_index()
    g["opp_central_share"] = g["opp_central_passes"] / g["opp_passes_faced"]
    g["wide_forcing_pp"] = (league_share - g["opp_central_share"]) * 100
    g = g.rename(columns={"defending_team": "team_name"})
    print(f"League-average central passing share: {league_share:.1%} "
          f"({len(df)} passes across {len(files)} matches)")
    return g[["team_name", "opp_passes_faced", "opp_central_share", "wide_forcing_pp"]]


def zscore(s):
    return (s - s.mean()) / s.std(ddof=0)


def main():
    out_dir_arg = sys.argv[1] if len(sys.argv) > 1 else None

    cv = central_disruption_value_per_match()
    wf = opponent_central_pass_share()
    g = cv.merge(wf, on="team_name")
    g["composite"] = zscore(g["central_value_pm_x1000"]) + zscore(g["wide_forcing_pp"])
    g = g.sort_values("composite", ascending=False)

    print("\nCentral defending + wide-forcing, ranked by composite z-score:")
    print(g[["team_name", "matches", "central_value_pm_x1000", "wide_forcing_pp",
             "composite"]].to_string(index=False))

    g.to_csv(os.path.join(ldv.CSV_DIR, "central_wide_forcing.csv"), index=False)

    for theme in ("dark", "light"):
        ldv.set_theme(theme)
        out_dir = out_dir_arg or ldv.visual_dir(theme)

        # 1. Composite leaderboard
        gg = g.sort_values("composite", ascending=True)
        fig, ax = plt.subplots(figsize=(12.5, 9.5))
        fig.patch.set_facecolor(ldv.BG)
        ax.set_facecolor(ldv.BG)
        vmax, vmin = gg["composite"].max(), gg["composite"].min()
        span = max(vmax - vmin, 1e-9)
        colors = [ldv.GOLD_RAMP(0.3 + 0.6 * ((v - vmin) / span)) for v in gg["composite"]]
        ax.barh(gg["team_name"], gg["composite"], color=colors, height=0.62, zorder=3)
        for i, (val, cvpm, wfpp) in enumerate(zip(gg["composite"], gg["central_value_pm_x1000"], gg["wide_forcing_pp"])):
            label = f"{val:+.2f}  (central {cvpm:.2f}, wide {wfpp:+.1f}pp)"
            ax.text(val + span * 0.03 * (1 if val >= 0 else -1), i, label,
                    va="center", ha="left" if val >= 0 else "right", fontsize=8.6, color=ldv.LEGEND_TEXT)
        ax.axvline(0, color=ldv.PITCH_LINE, lw=1)
        ax.tick_params(axis="y", colors=ldv.TEXT_MAIN, labelsize=10.3, length=0)
        ax.tick_params(axis="x", colors=ldv.TEXT_SUB, labelsize=9)
        for spine in ("top", "right", "left"):
            ax.spines[spine].set_visible(False)
        ax.spines["bottom"].set_color(ldv.PITCH_LINE)
        ax.grid(axis="x", color=ldv.PITCH_LINE, linewidth=0.6, alpha=0.5, zorder=0)
        ax.set_xlim(vmin - span * 0.55, vmax + span * 0.55)
        fig.text(0.5, 0.965, "Best Central Defenders: Disrupt + Force Wide", fontsize=22, fontweight="bold",
                 ha="center", color=ldv.TEXT_MAIN)
        fig.text(0.5, 0.933,
                 "Eredivisie 2025/26 · Season · composite z-score of central disruption value/match "
                 "and opponent central-passing suppression", fontsize=10, ha="center", color=ldv.TEXT_SUB)
        fig.text(0.98, 0.012, "Marc Lamberts", fontsize=9.5, ha="right", color=ldv.TEXT_FOOT, style="italic")
        fig.text(0.02, 0.012, "Data via Opta | wide-forcing pp = league-average opponent central-passing "
                 "share minus each team's own opponents' central share, in percentage points",
                 fontsize=7.2, color=ldv.TEXT_FOOT)
        fig.subplots_adjust(left=0.20, right=0.95, top=0.90, bottom=0.07)
        add_logo(fig)
        out_path = os.path.join(out_dir, "central_wide_forcing_leaderboard.png")
        fig.savefig(out_path, dpi=200, facecolor=ldv.BG)
        plt.close(fig)
        print("Saved:", out_path)

        # 2. Quadrant scatter: central disruption value vs wide-forcing
        fig, ax = plt.subplots(figsize=(11.5, 9))
        fig.patch.set_facecolor(ldv.BG)
        ax.set_facecolor(ldv.BG)
        xmed, ymed = g["central_value_pm_x1000"].median(), g["wide_forcing_pp"].median()
        vmax = g["composite"].max()
        colors = [ldv.GOLD_RAMP(0.3 + 0.6 * max(0, v) / max(vmax, 1e-9)) for v in g["composite"]]
        ax.scatter(g["central_value_pm_x1000"], g["wide_forcing_pp"], s=230, color=colors,
                  edgecolors="#ffe9b8", linewidth=0.6, zorder=3)
        for _, r in g.iterrows():
            ax.annotate(r["team_name"], xy=(r["central_value_pm_x1000"], r["wide_forcing_pp"]),
                       xytext=(6, 4), textcoords="offset points", fontsize=8.4, color=ldv.TEXT_MAIN)
        ax.axvline(xmed, color=ldv.PITCH_LINE, lw=1)
        ax.axhline(ymed, color=ldv.PITCH_LINE, lw=1)
        ax.set_xlabel("Central disruption value per match (x1000)", fontsize=10.5, color=ldv.TEXT_SUB)
        ax.set_ylabel("Wide-forcing index (pp, vs league-average opponent central share)",
                      fontsize=10.5, color=ldv.TEXT_SUB)
        ax.tick_params(colors=ldv.TEXT_SUB, labelsize=9)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color(ldv.PITCH_LINE)
        ax.grid(color=ldv.PITCH_LINE, linewidth=0.4, alpha=0.4)
        fig.text(0.5, 0.965, "Central Disruption vs. Forcing Play Wide", fontsize=21, fontweight="bold",
                 ha="center", color=ldv.TEXT_MAIN)
        fig.text(0.5, 0.93, "Eredivisie 2025/26 · Season · top-right = disrupts centrally AND pushes "
                 "opponents to the flanks", fontsize=10, ha="center", color=ldv.TEXT_SUB)
        fig.text(0.98, 0.015, "Marc Lamberts", fontsize=9.5, ha="right", color=ldv.TEXT_FOOT, style="italic")
        fig.subplots_adjust(left=0.11, right=0.96, top=0.88, bottom=0.09)
        add_logo(fig)
        out_path = os.path.join(out_dir, "central_wide_forcing_quadrant.png")
        fig.savefig(out_path, dpi=200, facecolor=ldv.BG)
        plt.close(fig)
        print("Saved:", out_path)


if __name__ == "__main__":
    main()

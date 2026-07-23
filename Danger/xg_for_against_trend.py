"""
xG for vs. xG against by match, with a rolling average, for a single team.
Uses the shot-level xg column already computed in Danger/*_danger_models.csv
(mean of 5 calibrated shot models — see danger_score methodology). Team
name -> contestantId comes from the raw Events/*.json fixtures, same
mapping used across the other Danger/ scripts.

Built in the Meridian house style (housestyle package).

Usage: python3 Danger/xg_for_against_trend.py ["Team Name"] [out_dir]
Defaults to Nijmegen Eendracht Combinatie.
"""
import sys
import re
import glob
import json
import collections
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import matplotlib.pyplot as plt

from housestyle import style, components

REPO_ROOT = Path(__file__).resolve().parent.parent
EVENTS_DIR = REPO_ROOT / "Events"
DANGER_DIR = REPO_ROOT / "Danger"
ROLL_WINDOW = 5


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


def opponent_name(matchup, team_name):
    m = re.match(r"(.+) - (.+)", matchup)
    if not m:
        return "?"
    home, away = m.group(1), m.group(2)
    return away if team_name.lower() in home.lower() else home


def rolling_mean(values, window):
    out = []
    for i in range(len(values)):
        lo = max(0, i - window + 1)
        chunk = values[lo:i + 1]
        out.append(sum(chunk) / len(chunk))
    return out


def collect(team_name, cid):
    files = [f for f in sorted(glob.glob(str(DANGER_DIR / "[0-9][0-9][0-9][0-9]-*_danger_models.csv")))
             if team_name.lower() in f.lower()]
    rows = []
    for fn in files:
        df = pd.read_csv(fn)
        base = Path(fn).name.replace("_danger_models.csv", "")
        m = re.match(r"(\d{4}-\d{2}-\d{2})_(.+)", base)
        date, matchup = m.group(1), m.group(2)
        cids_in_file = df["contestant_id"].unique()
        opp = next((c for c in cids_in_file if c != cid), None)
        xg_for = df.loc[df["contestant_id"] == cid, "xg"].sum()
        xg_against = df.loc[df["contestant_id"] == opp, "xg"].sum() if opp is not None else 0.0
        rows.append({"date": date, "opponent": opponent_name(matchup, team_name),
                      "xg_for": xg_for, "xg_against": xg_against})
    rows.sort(key=lambda r: r["date"])
    return rows


def main():
    team_name = sys.argv[1] if len(sys.argv) > 1 else "Nijmegen Eendracht Combinatie"
    out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(__file__).parent

    files = sorted(glob.glob(str(EVENTS_DIR / "*.json")))
    team_map = build_team_map(files)
    match = next((full for full in team_map if team_name.lower() in full.lower()), None)
    if match is None:
        print(f"Team '{team_name}' not found. Options: {list(team_map)}")
        sys.exit(1)
    cid = team_map[match]

    rows = collect(match, cid)
    if not rows:
        print("No danger_models.csv files found for this team.")
        sys.exit(1)

    n = len(rows)
    xs = list(range(1, n + 1))
    xgf = [r["xg_for"] for r in rows]
    xga = [r["xg_against"] for r in rows]
    roll_f = rolling_mean(xgf, ROLL_WINDOW)
    roll_a = rolling_mean(xga, ROLL_WINDOW)
    avg_f, avg_a = sum(xgf) / n, sum(xga) / n
    matches_ahead = sum(1 for f, a in zip(xgf, xga) if f > a)

    palette, cats = style.apply("light")
    ink_blue = cats[0]
    teal = cats[2]

    fig = plt.figure(figsize=(12.5, 7.6))
    fig.patch.set_facecolor(palette["surface"])
    ax = fig.add_axes([0.07, 0.15, 0.88, 0.60])

    ax.plot(xs, xgf, color=ink_blue, alpha=0.25, linewidth=1.1, marker="o", markersize=3.5, zorder=2)
    ax.plot(xs, xga, color=teal, alpha=0.25, linewidth=1.1, marker="o", markersize=3.5, zorder=2)
    ax.plot(xs, roll_f, color=ink_blue, linewidth=2.6, zorder=3,
            label=f"xG for ({ROLL_WINDOW}-match avg)")
    ax.plot(xs, roll_a, color=teal, linewidth=2.6, zorder=3,
            label=f"xG against ({ROLL_WINDOW}-match avg)")

    ax.set_xticks(xs)
    ax.set_xticklabels([str(i) for i in xs], fontsize=8.5)
    ax.set_xlabel("Matchday", fontsize=10.5, color=palette["ink_secondary"])
    ax.set_ylabel("xG", fontsize=10.5, color=palette["ink_secondary"])
    ax.grid(True, axis="y", color=palette["grid"], linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper right", frameon=False, fontsize=9.5, labelcolor=palette["ink_secondary"])

    team_short = match.replace("Nijmegen Eendracht Combinatie", "NEC")
    components.header(
        fig,
        kicker=team_short,
        title=f"{team_short} have out-created opponents in {matches_ahead} of {n} matches this season",
        dek=f"Rolling {ROLL_WINDOW}-match xG for vs. against  ·  season avg {avg_f:.2f} for, "
            f"{avg_a:.2f} against  ·  Eredivisie 2025/26",
        palette=palette,
    )
    components.footer(
        fig,
        source="Opta/StatsPerform, Eredivisie 2025/26",
        palette=palette,
    )

    out_path = out_dir / f"xg_for_against_trend_{team_short.lower().replace(' ', '_')}.png"
    fig.savefig(out_path, dpi=200, facecolor=palette["surface"])
    plt.close(fig)
    print(f"n_matches={n} avg_xgf={avg_f:.2f} avg_xga={avg_a:.2f} matches_ahead={matches_ahead}")
    print("Saved:", out_path)


if __name__ == "__main__":
    main()

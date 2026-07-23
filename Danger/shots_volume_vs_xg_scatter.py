"""
Team scatter: season shot volume vs. total xG, Eredivisie 2025/26.
Reads the league-wide scored shots (Danger/all_eredivisie_danger_models.csv,
one row per shot with contestant_id and xg) and maps contestant_id -> team
name from the raw Events/*.json fixtures. One point per team; the diagonal
reads as shot quality (xG per shot) — teams above the trend create better
chances, not just more of them.

Built in the Meridian house style (housestyle package).

Usage: python3 Danger/shots_volume_vs_xg_scatter.py [out_dir]
"""
import sys
import re
import glob
import json
import collections
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from adjustText import adjust_text

from housestyle import style, components

REPO_ROOT = Path(__file__).resolve().parent.parent
EVENTS_DIR = REPO_ROOT / "Events"
DANGER_CSV = REPO_ROOT / "Danger" / "all_eredivisie_danger_models.csv"


def build_team_map():
    files = sorted(glob.glob(str(EVENTS_DIR / "*.json")))
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


def clean(name):
    return (name.replace("AFC ", "").replace("SC ", "").replace("SBV ", "")
                .replace("FC ", "").replace("PSV ", "PSV ").strip())


def main():
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent

    df = pd.read_csv(DANGER_CSV)
    team_map = build_team_map()
    cid_to_team = {v: k for k, v in team_map.items()}
    df["team"] = df["contestant_id"].map(cid_to_team)

    g = df.groupby("team").agg(shots=("xg", "size"), xg=("xg", "sum")).reset_index()
    g["xg_per_shot"] = g["xg"] / g["shots"]
    g = g.sort_values("xg", ascending=False).reset_index(drop=True)

    palette, cats = style.apply("light")
    ink_blue = cats[0]

    fig = plt.figure(figsize=(9, 7.4))
    fig.patch.set_facecolor(palette["surface"])
    ax = fig.add_axes([0.10, 0.14, 0.84, 0.62])

    slope = g["xg"].sum() / g["shots"].sum()
    xs = np.array([g["shots"].min() * 0.9, g["shots"].max() * 1.06])
    ax.plot(xs, xs * slope, color=palette["axis"], linewidth=1.1,
            linestyle=(0, (4, 3)), zorder=1)

    top_team = g.iloc[0]["team"]
    texts = []
    for _, row in g.iterrows():
        is_top = row["team"] == top_team
        ax.scatter(row["shots"], row["xg"],
                   s=150 if is_top else 100,
                   color=palette["accent"] if is_top else ink_blue,
                   edgecolors=palette["surface"], linewidths=1.2,
                   zorder=4 if is_top else 3)
        texts.append(ax.text(row["shots"], row["xg"], clean(row["team"]),
                              ha="center", va="center", fontsize=8.7,
                              color=palette["ink_primary"] if is_top else palette["ink_secondary"],
                              fontweight="bold" if is_top else "normal"))

    adjust_text(texts, ax=ax,
                arrowprops=dict(arrowstyle="-", color=palette["axis"], lw=0.7),
                expand=(1.3, 1.6))

    ax.set_xlabel("Shots (season total)", fontsize=10.5, color=palette["ink_secondary"])
    ax.set_ylabel("Total xG (season)", fontsize=10.5, color=palette["ink_secondary"])
    ax.grid(True, axis="both", color=palette["grid"], linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)

    components.header(
        fig,
        kicker="Eredivisie 2025/26",
        title=f"{clean(top_team)} lead the league in xG without leading in shots",
        dek="Season shot volume vs. total xG, all 18 teams  ·  "
            "dashed line = league-average xG per shot",
        palette=palette,
    )
    components.footer(
        fig,
        source="Opta event data via xG model, Eredivisie 2025/26",
        palette=palette,
    )

    out_path = out_dir / "shots_volume_vs_xg_eredivisie.png"
    fig.savefig(out_path, dpi=200, facecolor=palette["surface"])
    plt.close(fig)
    print("Saved:", out_path)


if __name__ == "__main__":
    main()

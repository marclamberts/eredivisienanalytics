"""
Bar chart: total xDanger (danger_score) by team, Eredivisie 2025/26.
danger_score is the shot-danger composite already computed per shot in
Danger/*.csv (see Danger models) — blends xG, shot placement probability
and post-shot quality into one 0-1 score per attempt. Summed across the
season it reads as "total danger created," a broader signal than raw xG
since it also rewards well-placed shots that a keeper has to work to stop.

Built in the Meridian house style (housestyle package).

Usage: python3 Danger/xdanger_bar_eredivisie.py [out_dir]
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
                .replace("FC ", "").strip())


def main():
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent

    df = pd.read_csv(DANGER_CSV)
    team_map = build_team_map()
    cid_to_team = {v: k for k, v in team_map.items()}
    df["team"] = df["contestant_id"].map(cid_to_team)

    g = df.groupby("team")["danger_score"].sum().reset_index()
    g = g.sort_values("danger_score", ascending=True).reset_index(drop=True)

    palette, cats = style.apply("light")
    ink_blue = cats[0]

    fig = plt.figure(figsize=(9.5, 9.2))
    fig.patch.set_facecolor(palette["surface"])
    ax = fig.add_axes([0.32, 0.13, 0.60, 0.62])

    labels = [clean(t) for t in g["team"]]
    bars = ax.barh(labels, g["danger_score"], color=ink_blue, height=0.66, zorder=3)
    top_idx = len(g) - 1
    components.highlight_bars(bars, accent_index=top_idx, palette=palette)

    for i, v in enumerate(g["danger_score"]):
        ax.text(v + 1.5, i, f"{v:.0f}", va="center", ha="left", fontsize=9,
                 color=palette["ink_secondary"])

    ax.set_xlabel("Total xDanger (season)", fontsize=10.5, color=palette["ink_secondary"])
    ax.grid(True, axis="x", color=palette["grid"], linewidth=0.7, zorder=0)
    ax.grid(False, axis="y")
    ax.set_axisbelow(True)
    ax.set_xlim(0, g["danger_score"].max() * 1.12)

    top_team = g.iloc[-1]["team"]
    components.header(
        fig,
        kicker="Danger models",
        title=f"{clean(top_team)} generate more danger than anyone else in the league",
        dek="Total xDanger (danger_score) from all shots, all 18 teams  ·  Eredivisie 2025/26",
        palette=palette,
    )
    components.footer(
        fig,
        source="Opta event data via danger model, Eredivisie 2025/26",
        palette=palette,
    )

    out_path = out_dir / "xdanger_bar_eredivisie.png"
    fig.savefig(out_path, dpi=200, facecolor=palette["surface"])
    plt.close(fig)
    print("Saved:", out_path)


if __name__ == "__main__":
    main()

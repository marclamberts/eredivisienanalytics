"""
xG shot map: every shot a team has taken this season, plotted on the
attacking half, sized by xG and coloured by outcome. Same shot-quality
data as the rest of the Danger/ scripts (Danger/all_eredivisie_danger_models.csv).

Built in the Meridian house style (housestyle package).

Usage: python3 Danger/xg_shotmap.py ["Team Name"] [out_dir]
Defaults to Sparta Rotterdam.
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
from matplotlib.lines import Line2D
from mplsoccer import VerticalPitch

from housestyle import style, components

REPO_ROOT = Path(__file__).resolve().parent.parent
EVENTS_DIR = REPO_ROOT / "Events"
DANGER_CSV = REPO_ROOT / "Danger" / "all_eredivisie_danger_models.csv"

NICKNAMES = {
    "Feyenoord Rotterdam": "Feyenoord", "AFC Ajax": "Ajax", "PSV Eindhoven": "PSV",
    "FC Twente": "Twente", "FC Utrecht": "Utrecht", "FC Groningen": "Groningen",
    "FC Volendam": "Volendam", "SC Heerenveen": "Heerenveen", "SC Telstar": "Telstar",
    "SBV Excelsior": "Excelsior", "NAC Breda": "NAC Breda", "PEC Zwolle": "PEC Zwolle",
    "Sparta Rotterdam": "Sparta", "Fortuna Sittard": "Fortuna", "Heracles Almelo": "Heracles",
    "Alkmaar Zaanstreek": "AZ", "Go Ahead Eagles": "Go Ahead Eagles",
    "Nijmegen Eendracht Combinatie": "NEC",
}


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


def main():
    team_name = sys.argv[1] if len(sys.argv) > 1 else "Sparta Rotterdam"
    out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(__file__).parent

    files = sorted(glob.glob(str(EVENTS_DIR / "*.json")))
    team_map = build_team_map(files)
    match = next((full for full in team_map if team_name.lower() in full.lower()), None)
    if match is None:
        print(f"Team '{team_name}' not found. Options: {list(team_map)}")
        sys.exit(1)
    cid = team_map[match]

    df = pd.read_csv(DANGER_CSV)
    shots = df[df["contestant_id"] == cid].copy()
    shots["is_goal"] = shots["is_goal"].astype(bool)
    shots["is_on_target"] = shots["is_on_target"].astype(bool)

    goals = shots[shots["is_goal"]]
    on_target = shots[(~shots["is_goal"]) & (shots["is_on_target"])]
    off_target = shots[~shots["is_on_target"]]

    palette, cats = style.apply("light")
    ink_blue = cats[0]
    off_color = palette["axis"]

    fig = plt.figure(figsize=(8.6, 10.4))
    fig.patch.set_facecolor(palette["surface"])

    pitch_ax = fig.add_axes([0.05, 0.09, 0.90, 0.68])
    pitch = VerticalPitch(pitch_type="opta", pitch_color=palette["surface"],
                           line_color=palette["axis"], linewidth=1.1, half=True)
    pitch.draw(ax=pitch_ax)

    for sub, color, alpha, zorder in [
        (off_target, off_color, 0.55, 2),
        (on_target, ink_blue, 0.80, 3),
        (goals, palette["accent"], 0.95, 4),
    ]:
        if len(sub) == 0:
            continue
        sizes = 1200 * sub["xg"].clip(0.02, 0.85) + 35
        pitch.scatter(sub["x"], sub["y"], ax=pitch_ax, s=sizes, color=color, alpha=alpha,
                       edgecolors=palette["surface"], linewidth=1.0, zorder=zorder)

    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=palette["accent"],
               markeredgecolor=palette["accent"], markersize=11, label=f"Goal ({len(goals)})"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=ink_blue,
               markeredgecolor=ink_blue, markersize=11, label=f"On target ({len(on_target)})"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=off_color,
               markeredgecolor=off_color, markersize=9, label=f"Off target/blocked ({len(off_target)})"),
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.775), ncol=3,
               frameon=False, fontsize=9, labelcolor=palette["ink_secondary"])

    total_xg = shots["xg"].sum()
    team_short = NICKNAMES.get(match, match)
    components.header(
        fig,
        kicker=team_short,
        title=f"{team_short} have scored {len(goals)} goals from {total_xg:.1f} xG this season",
        dek=f"Every shot, sized by xG  ·  {len(shots)} shots, {total_xg / max(len(shots), 1):.2f} xG/shot  ·  "
            f"Eredivisie 2025/26",
        palette=palette,
    )
    components.footer(
        fig,
        source="Opta/StatsPerform, Eredivisie 2025/26",
        palette=palette,
    )

    out_path = out_dir / f"xg_shotmap_{team_short.lower().replace(' ', '_')}.png"
    fig.savefig(out_path, dpi=200, facecolor=palette["surface"])
    plt.close(fig)
    print(f"{match}: {len(shots)} shots, {len(goals)} goals, {total_xg:.2f} total xG, "
          f"{total_xg / max(len(shots), 1):.3f} xG/shot")
    print("Saved:", out_path)


if __name__ == "__main__":
    main()

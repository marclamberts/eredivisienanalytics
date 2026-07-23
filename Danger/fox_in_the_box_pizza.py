"""
"Fox in the box" pizza plot: percentile ranks on twelve poacher-profile
metrics across three categories, one player against an Eredivisie
box-forward peer group (minutes >= 900, shots >= 15, and >= 50% of shots
taken from inside the penalty box -- narrower and more defensible than a
flat "attacker" filter, since it excludes wide/deep shot-takers who
wouldn't be called foxes in the box in the first place).

Categories:
  Volume & Profile  -- how much of their shooting is box-based, and how
  Shot Quality       -- how good the chances are once they get there
  Finishing Output   -- what they actually do with them

Sources: shot-level data (Danger/all_eredivisie_danger_models.csv),
enriched with body-part (header) info recovered from the raw Events/*.json
qualifiers (qualifierId 15 = head; not present in the Danger CSV itself),
and season minutes (GDA/gda_player_summary.csv).

Built in the Meridian house style (housestyle package).

Usage: python3 Danger/fox_in_the_box_pizza.py ["Player Name"] [out_dir]
Defaults to A. Ueda.
"""
import sys
import glob
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from housestyle import style, components

REPO_ROOT = Path(__file__).resolve().parent.parent
EVENTS_DIR = REPO_ROOT / "Events"
DANGER_CSV = REPO_ROOT / "Danger" / "all_eredivisie_danger_models.csv"
GDA_CSV = REPO_ROOT / "GDA" / "gda_player_summary.csv"

BOX_X = 83.0
SIX_YARD_X = 94.0          # six-yard-box depth on the Opta 100x100 grid
BIG_CHANCE_XG = 0.20       # conventional big-chance threshold (provider-independent rule of thumb)
HEAD_QID = 15
SHOT_TYPE_IDS = {13, 14, 15, 16}

MIN_MINUTES = 900
MIN_SHOTS = 15
MIN_BOX_SHARE = 0.50

# (key, label, format, category_index)
METRICS = [
    ("shots90", "Shots /90", "{:.1f}", 0),
    ("box_share_pct", "Shots in box %", "{:.0f}%", 0),
    ("six_yard_share_pct", "Six-yard-box shots %", "{:.0f}%", 0),
    ("header_share_pct", "Headed shots %", "{:.0f}%", 0),
    ("xg_per_shot", "xG per shot", "{:.2f}", 1),
    ("danger_per_shot", "Danger score /shot", "{:.2f}", 1),
    ("big_chance_share_pct", "Big-chance shots %", "{:.0f}%", 1),
    ("sot_pct", "Shots on target %", "{:.0f}%", 1),
    ("npg90", "Non-pen goals /90", "{:.2f}", 2),
    ("npxg90", "Non-pen xG /90", "{:.2f}", 2),
    ("conversion", "Conversion %", "{:.0f}%", 2),
    ("finishing", "Goals minus xG", "{:+.1f}", 2),
]
CATEGORY_NAMES = ["Volume & Profile", "Shot Quality", "Finishing Output"]


def build_header_flags():
    """event_id -> True/False for whether a shot was headed. Scans the raw
    Events files once, since body_part isn't carried in the Danger CSV."""
    flags = {}
    for fn in sorted(glob.glob(str(EVENTS_DIR / "*.json"))):
        with open(fn) as f:
            data = json.load(f)
        for e in data["event"]:
            if e.get("typeId") not in SHOT_TYPE_IDS:
                continue
            qids = {q["qualifierId"] for q in e.get("qualifier", []) or []}
            flags[e["id"]] = HEAD_QID in qids
    return flags


def build_peer_table():
    d = pd.read_csv(DANGER_CSV)
    d["pen_goal"] = (d["is_penalty"] == 1) & (d["is_goal"] == 1)
    d["in_box"] = d["x"] >= BOX_X
    d["six_yard"] = d["x"] >= SIX_YARD_X
    d["big_chance"] = d["xg"] >= BIG_CHANCE_XG

    header_flags = build_header_flags()
    d["is_header"] = d["event_id"].map(header_flags).fillna(False)

    g = d.groupby("player_name").agg(
        shots=("xg", "size"), goals=("is_goal", "sum"), xg=("xg", "sum"),
        on_target=("is_on_target", "sum"), pen_shots=("is_penalty", "sum"),
        pen_goals=("pen_goal", "sum"), box_share=("in_box", "mean"),
        six_yard_share=("six_yard", "mean"), header_share=("is_header", "mean"),
        big_chance_share=("big_chance", "mean"), danger=("danger_score", "sum"),
        danger_per_shot=("danger_score", "mean"),
    ).reset_index()
    pen_xg = d[d["is_penalty"] == 1].groupby("player_name")["xg"].sum()
    g["pen_xg"] = g["player_name"].map(pen_xg).fillna(0.0)

    g["npg"] = g["goals"] - g["pen_goals"]
    g["npxg"] = g["xg"] - g["pen_xg"]

    gda = pd.read_csv(GDA_CSV)
    mins = gda.groupby("player_name")["minutes"].sum().reset_index()
    m = g.merge(mins, on="player_name", how="left").dropna(subset=["minutes"])
    m = m[(m["minutes"] >= MIN_MINUTES) & (m["shots"] >= MIN_SHOTS)
          & (m["box_share"] >= MIN_BOX_SHARE)].copy()

    m["npg90"] = m["npg"] / m["minutes"] * 90
    m["npxg90"] = m["npxg"] / m["minutes"] * 90
    m["shots90"] = m["shots"] / m["minutes"] * 90
    m["sot_pct"] = m["on_target"] / m["shots"] * 100
    m["xg_per_shot"] = m["xg"] / m["shots"]
    m["conversion"] = m["goals"] / m["shots"] * 100
    m["finishing"] = m["npg"] - m["npxg"]
    m["box_share_pct"] = m["box_share"] * 100
    m["six_yard_share_pct"] = m["six_yard_share"] * 100
    m["header_share_pct"] = m["header_share"] * 100
    m["big_chance_share_pct"] = m["big_chance_share"] * 100
    return m


def main():
    player_name = sys.argv[1] if len(sys.argv) > 1 else "A. Ueda"
    out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(__file__).parent

    peers = build_peer_table()
    match = next((p for p in peers["player_name"] if player_name.lower() in p.lower()), None)
    if match is None:
        print(f"Player '{player_name}' not found among qualifying box forwards "
              f"(min {MIN_MINUTES} minutes, {MIN_SHOTS} shots, {MIN_BOX_SHARE:.0%} box share). "
              f"Options: {sorted(peers['player_name'])}")
        sys.exit(1)

    row = peers[peers["player_name"] == match].iloc[0]
    percentiles, raw_values = [], []
    for key, _, _, _ in METRICS:
        pct = (peers[key] < row[key]).mean() * 100
        percentiles.append(pct)
        raw_values.append(row[key])

    palette, cats = style.apply("light")
    cat_colors = [cats[0], cats[2], cats[6]]  # ink blue / teal / violet -- categorical, accent stays unused here

    n = len(METRICS)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    width = 2 * np.pi / n * 0.88

    fig = plt.figure(figsize=(10.8, 12.2))
    fig.patch.set_facecolor(palette["surface"])
    ax = fig.add_axes([0.17, 0.145, 0.66, 0.585], projection="polar")
    ax.set_facecolor(palette["surface"])
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    ax.bar(angles, [100] * n, width=width, color=palette["grid"], zorder=1, edgecolor="none")
    for ring in (20, 40, 60, 80, 100):
        ax.plot(np.linspace(0, 2 * np.pi, 200), [ring] * 200, color=palette["axis"],
                 linewidth=0.6, alpha=0.6, zorder=2)

    bar_colors = [cat_colors[cat_idx] for *_, cat_idx in METRICS]
    ax.bar(angles, percentiles, width=width, color=bar_colors, alpha=0.88,
           zorder=3, edgecolor=palette["surface"], linewidth=1.5)

    ax.set_ylim(0, 100)
    ax.set_xticks(angles)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.spines["polar"].set_visible(False)
    ax.grid(False)

    for angle, pct, raw, (key, label, fmt, cat_idx) in zip(angles, percentiles, raw_values, METRICS):
        ax.text(angle, 122, label, ha="center", va="center", fontsize=9.0,
                 fontweight="bold", color=palette["ink_primary"], family="sans-serif")
        ax.text(angle, pct + 8 if pct < 88 else pct - 8,
                 fmt.format(raw), ha="center", va="center", fontsize=8.3,
                 fontweight="bold",
                 color=palette["ink_primary"] if pct < 88 else palette["surface"],
                 family="sans-serif")

    # category legend, under the dek
    legend_y = 0.775
    swatch_xs = [0.30, 0.50, 0.70]
    for x, name, color in zip(swatch_xs, CATEGORY_NAMES, cat_colors):
        fig.add_artist(mpatches.Rectangle((x, legend_y - 0.006), 0.014, 0.014,
                                            transform=fig.transFigure, color=color))
        fig.text(x + 0.022, legend_y + 0.001, name, fontsize=9.5, color=palette["ink_secondary"],
                  family="sans-serif", ha="left", va="center")

    components.header(
        fig,
        kicker=f"Fox in the box · {match}",
        title=f"{match} ranks among the league's best box forwards on nearly every metric",
        dek=f"Percentile vs. Eredivisie box forwards (min {MIN_MINUTES} min, {MIN_SHOTS}+ shots, "
            f"{MIN_BOX_SHARE:.0%}+ shots in box, n={len(peers)})  ·  Eredivisie 2025/26",
        palette=palette,
    )
    components.footer(
        fig,
        source="Opta/StatsPerform, Eredivisie 2025/26",
        note=f"big chance = shot with xG >= {BIG_CHANCE_XG:.2f}",
        palette=palette,
    )

    out_path = out_dir / f"fox_in_the_box_pizza_{match.lower().replace(' ', '_').replace('.', '')}.png"
    fig.savefig(out_path, dpi=200, facecolor=palette["surface"])
    plt.close(fig)
    print(f"{match} (n_peers={len(peers)}): " +
          ", ".join(f"{lbl}={fmt.format(v)} (p{p:.0f})"
                     for (k, lbl, fmt, c), v, p in zip(METRICS, raw_values, percentiles)))
    print("Saved:", out_path)


if __name__ == "__main__":
    main()

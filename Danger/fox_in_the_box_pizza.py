"""
"Fox in the box" pizza plot: percentile ranks on eight poacher-profile
shooting metrics, one player against an Eredivisie attacking peer group
(minutes >= 900, shots >= 15 -- enough attacking involvement to be a fair
comparison without requiring a position field the underlying data doesn't
carry). Combines shot-level data (Danger/all_eredivisie_danger_models.csv)
with season minutes (GDA/gda_player_summary.csv).

Built in the Meridian house style (housestyle package).

Usage: python3 Danger/fox_in_the_box_pizza.py ["Player Name"] [out_dir]
Defaults to A. Ueda.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from housestyle import style, components

REPO_ROOT = Path(__file__).resolve().parent.parent
DANGER_CSV = REPO_ROOT / "Danger" / "all_eredivisie_danger_models.csv"
GDA_CSV = REPO_ROOT / "GDA" / "gda_player_summary.csv"

BOX_X = 83.0
MIN_MINUTES = 900
MIN_SHOTS = 15

METRICS = [
    ("npg90", "Non-pen goals /90", "{:.2f}"),
    ("npxg90", "Non-pen xG /90", "{:.2f}"),
    ("shots90", "Shots /90", "{:.1f}"),
    ("sot90", "Shots on target /90", "{:.1f}"),
    ("xg_per_shot", "xG per shot", "{:.2f}"),
    ("conversion", "Conversion %", "{:.0f}%"),
    ("finishing", "Goals minus xG", "{:+.1f}"),
    ("box_share_pct", "Shots in box %", "{:.0f}%"),
]


def build_peer_table():
    d = pd.read_csv(DANGER_CSV)
    d["pen_goal"] = (d["is_penalty"] == 1) & (d["is_goal"] == 1)
    d["in_box"] = d["x"] >= BOX_X

    g = d.groupby("player_name").agg(
        shots=("xg", "size"), goals=("is_goal", "sum"), xg=("xg", "sum"),
        on_target=("is_on_target", "sum"), pen_shots=("is_penalty", "sum"),
        pen_goals=("pen_goal", "sum"), box_share=("in_box", "mean"),
    ).reset_index()
    pen_xg = d[d["is_penalty"] == 1].groupby("player_name")["xg"].sum()
    g["pen_xg"] = g["player_name"].map(pen_xg).fillna(0.0)

    g["npg"] = g["goals"] - g["pen_goals"]
    g["npxg"] = g["xg"] - g["pen_xg"]

    gda = pd.read_csv(GDA_CSV)
    mins = gda.groupby("player_name")["minutes"].sum().reset_index()
    m = g.merge(mins, on="player_name", how="left").dropna(subset=["minutes"])
    m = m[(m["minutes"] >= MIN_MINUTES) & (m["shots"] >= MIN_SHOTS)].copy()

    m["npg90"] = m["npg"] / m["minutes"] * 90
    m["npxg90"] = m["npxg"] / m["minutes"] * 90
    m["shots90"] = m["shots"] / m["minutes"] * 90
    m["sot90"] = m["on_target"] / m["minutes"] * 90
    m["xg_per_shot"] = m["xg"] / m["shots"]
    m["conversion"] = m["goals"] / m["shots"] * 100
    m["finishing"] = m["npg"] - m["npxg"]
    m["box_share_pct"] = m["box_share"] * 100
    return m


def main():
    player_name = sys.argv[1] if len(sys.argv) > 1 else "A. Ueda"
    out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(__file__).parent

    peers = build_peer_table()
    match = next((p for p in peers["player_name"] if player_name.lower() in p.lower()), None)
    if match is None:
        print(f"Player '{player_name}' not found among qualifying attackers "
              f"(min {MIN_MINUTES} minutes, {MIN_SHOTS} shots). "
              f"Options: {sorted(peers['player_name'])}")
        sys.exit(1)

    row = peers[peers["player_name"] == match].iloc[0]
    percentiles, raw_values = [], []
    for key, _, _ in METRICS:
        pct = (peers[key] < row[key]).mean() * 100
        percentiles.append(pct)
        raw_values.append(row[key])

    palette, cats = style.apply("light")
    ink_blue = cats[0]

    n = len(METRICS)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    width = 2 * np.pi / n * 0.88

    fig = plt.figure(figsize=(10.5, 11.6))
    fig.patch.set_facecolor(palette["surface"])
    ax = fig.add_axes([0.17, 0.16, 0.66, 0.60], projection="polar")
    ax.set_facecolor(palette["surface"])
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    ax.bar(angles, [100] * n, width=width, color=palette["grid"], zorder=1, edgecolor="none")
    for ring in (20, 40, 60, 80, 100):
        ax.plot(np.linspace(0, 2 * np.pi, 200), [ring] * 200, color=palette["axis"],
                 linewidth=0.6, alpha=0.6, zorder=2)

    bars = ax.bar(angles, percentiles, width=width, color=ink_blue, alpha=0.88,
                   zorder=3, edgecolor=palette["surface"], linewidth=1.5)
    for a, pct in zip(angles, percentiles):
        if pct >= 90:
            bars[list(angles).index(a)].set_color(palette["accent"])

    ax.set_ylim(0, 100)
    ax.set_xticks(angles)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.spines["polar"].set_visible(False)
    ax.grid(False)

    for angle, pct, raw, (key, label, fmt) in zip(angles, percentiles, raw_values, METRICS):
        ax.text(angle, 122, label, ha="center", va="center", fontsize=9.3,
                 fontweight="bold", color=palette["ink_primary"], family="sans-serif")
        ax.text(angle, pct + 8 if pct < 88 else pct - 8,
                 fmt.format(raw), ha="center", va="center", fontsize=8.5,
                 fontweight="bold",
                 color=palette["ink_primary"] if pct < 88 else palette["surface"],
                 family="sans-serif")

    team = "Feyenoord Rotterdam" if match == "A. Ueda" else ""
    components.header(
        fig,
        kicker=f"Fox in the box · {match}",
        title=f"{match} ranks in the league's top tier on every poacher metric",
        dek=f"Percentile vs. Eredivisie attackers (min {MIN_MINUTES} min, {MIN_SHOTS}+ shots, "
            f"n={len(peers)})  ·  Eredivisie 2025/26",
        palette=palette,
    )
    components.footer(
        fig,
        source="Opta/StatsPerform, Eredivisie 2025/26",
        note="terracotta = 90th percentile or above",
        palette=palette,
    )

    out_path = out_dir / f"fox_in_the_box_pizza_{match.lower().replace(' ', '_').replace('.', '')}.png"
    fig.savefig(out_path, dpi=200, facecolor=palette["surface"])
    plt.close(fig)
    print(f"{match}: " + ", ".join(f"{lbl}={fmt.format(v)} (p{p:.0f})"
                                     for (k, lbl, fmt), v, p in zip(METRICS, raw_values, percentiles)))
    print("Saved:", out_path)


if __name__ == "__main__":
    main()

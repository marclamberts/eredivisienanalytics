"""
Fox in the Box -- Step 2: turn the 16 metrics into a profile.

Method:
  1. z-score every metric across the qualified player pool (mean 0, std 1).
     avg_shot_distance_m is inverted first (shorter range = more fox-like).
  2. Blend the 16 z-scores into a composite score three different ways:
       - arithmetic mean  -- plain average across all 16 traits. Rewards
         well-rounded production, treats every metric equally.
       - weighted mean    -- same z-scores, but Finishing Output and Clinical
         Efficiency (the pillars that most define "fox in the box") count
         for more than Box Presence and Shot Volume & Instinct.
       - harmonic mean     -- computed on each metric rescaled to a positive
         0-100 scale. The harmonic mean is dragged down hard by any single
         weak metric, so it flags "one-trick" players who post a huge number
         in one metric but are anonymous everywhere else -- the opposite of
         a genuine all-around fox in the box.
  3. Each of the three composites is min-max rescaled to 0-100 and averaged
     into one final Fox-in-the-Box Index.

Usage: python3 fox_box_profile.py [out_dir]
"""
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

METRICS_JSON = Path(__file__).parent / "fox_box_metrics.json"
LOGO_PATH = "/Users/marclamberts/Downloads/Waltzing Analytics Logo Type.png"

BG = "#0d1117"
GRID_COLOR = "#232a35"
TEXT_MAIN = "#e6e9ee"
TEXT_SUB = "#9aa4b2"
AMBER = "#e0b34a"
BLUE = "#2f8fd1"
GREEN = "#4a9e5c"
RED = "#e0765c"

HARMONIC_FLOOR = 5.0  # keeps one rock-bottom metric from collapsing the harmonic mean to ~0

# (key, label, pillar, weight, invert)
# invert=True means a *lower* raw value is more "fox in the box" -- its
# z-score is flipped before it's combined with the rest.
METRIC_DEFS = [
    ("touches_in_box_p90",           "Touches in box /90",       "Box Presence",         0.9, False),
    ("box_touch_share_pct",          "Box touch share %",        "Box Presence",         0.9, False),
    ("shots_in_box_share_pct",       "Shots in box share %",     "Box Presence",         0.9, False),
    ("aerial_win_box_p90",           "Aerial wins in box /90",   "Box Presence",         0.9, False),

    ("shots_p90",                    "Shots /90",                "Shot Volume & Instinct", 1.0, False),
    ("shots_on_target_pct",          "Shots on target %",        "Shot Volume & Instinct", 1.0, False),
    ("shots_per_box_touch_pct",      "Shots per box touch %",    "Shot Volume & Instinct", 1.0, False),
    ("big_chances_p90",              "Big chances /90",          "Shot Volume & Instinct", 1.0, False),

    ("np_goals_p90",                 "NP goals /90",             "Finishing Output",      1.3, False),
    ("goals_per_shot_pct",           "Goals per shot %",         "Finishing Output",      1.3, False),
    ("goals_minus_xg_p90",           "Goals minus xG /90",       "Finishing Output",      1.3, False),
    ("goals_per_box_touch_pct",      "Goals per box touch %",    "Finishing Output",      1.3, False),

    ("xg_per_shot",                  "xG per shot",              "Clinical Efficiency",   1.2, False),
    ("goals_per_xg_ratio",           "Goals / xG ratio",         "Clinical Efficiency",   1.2, False),
    ("goals_per_shot_on_target_pct", "Goals per SoT %",          "Clinical Efficiency",   1.2, False),
    ("avg_shot_distance_m",          "Avg shot distance (m)",    "Clinical Efficiency",   1.2, True),
]

PILLARS = ["Box Presence", "Shot Volume & Instinct", "Finishing Output", "Clinical Efficiency"]


def add_logo(fig, width=0.11, margin=0.014):
    import matplotlib.image as mpimg
    try:
        img = mpimg.imread(LOGO_PATH)
    except FileNotFoundError:
        return
    fig_w, fig_h = fig.get_size_inches()
    img_h, img_w = img.shape[0], img.shape[1]
    width_in = width * fig_w
    height_in = width_in * (img_h / img_w)
    height = height_in / fig_h
    left = 1 - margin - width
    bottom = 1 - margin - height
    logo_ax = fig.add_axes([left, bottom, width, height], zorder=10)
    logo_ax.patch.set_alpha(0)
    logo_ax.set_xlim(0, img_w)
    logo_ax.set_ylim(img_h, 0)
    logo_ax.imshow(img)
    logo_ax.axis("off")


def min_max_scale(s, lo=0.0, hi=100.0):
    span = s.max() - s.min()
    if span == 0:
        return pd.Series(50.0, index=s.index)
    return (s - s.min()) / span * (hi - lo) + lo


def build_profile(df):
    z = pd.DataFrame(index=df.index)
    for key, _, _, _, invert in METRIC_DEFS:
        col = df[key].astype(float)
        std = col.std()
        col_z = (col - col.mean()) / std if std > 0 else pd.Series(0.0, index=col.index)
        z[key] = -col_z if invert else col_z

    # --- arithmetic mean composite: equal weight across all 16 z-scores ---
    arithmetic = z[[k for k, *_ in METRIC_DEFS]].mean(axis=1)

    # --- weighted mean composite: pillar weights from METRIC_DEFS ---
    weights = np.array([w for *_, w, _ in METRIC_DEFS])
    weighted = (z[[k for k, *_ in METRIC_DEFS]].values * weights).sum(axis=1) / weights.sum()
    weighted = pd.Series(weighted, index=df.index)

    # --- harmonic mean composite: needs positive inputs, so rescale each
    # z-score to 0-100 first, then floor it so no metric can hit exactly 0 ---
    scaled = z[[k for k, *_ in METRIC_DEFS]].apply(min_max_scale)
    scaled = scaled.clip(lower=HARMONIC_FLOOR)
    harmonic = len(METRIC_DEFS) / (1.0 / scaled).sum(axis=1)

    profile = df[["player_id", "player_name", "team", "minutes"]].copy()
    profile["arithmetic_mean_z"] = arithmetic
    profile["weighted_mean_z"] = weighted
    profile["harmonic_mean_score"] = harmonic

    profile["arithmetic_scaled"] = min_max_scale(arithmetic)
    profile["weighted_scaled"] = min_max_scale(weighted)
    profile["harmonic_scaled"] = min_max_scale(harmonic)

    profile["fox_in_the_box_index"] = (
        profile["arithmetic_scaled"] + profile["weighted_scaled"] + profile["harmonic_scaled"]
    ) / 3

    # pillar-level scaled scores (0-100), for the radar breakdown chart
    for pillar in PILLARS:
        keys = [k for k, _, p, _, _ in METRIC_DEFS if p == pillar]
        profile[f"pillar_{pillar}"] = min_max_scale(z[keys].mean(axis=1))

    profile = profile.sort_values("fox_in_the_box_index", ascending=False).reset_index(drop=True)
    profile.insert(0, "rank", range(1, len(profile) + 1))
    return profile


def make_leaderboard(profile, out_path, top_n=15):
    top = profile.head(top_n).iloc[::-1]

    fig, ax = plt.subplots(figsize=(13.5, 0.62 * top_n + 2.4))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    y_pos = list(range(len(top)))
    ax.barh(y_pos, top["fox_in_the_box_index"], color=AMBER, height=0.62, zorder=3)
    for y, val in zip(y_pos, top["fox_in_the_box_index"]):
        ax.text(val + 1.0, y, f"{val:.1f}", va="center", ha="left", fontsize=10.5,
                color=TEXT_MAIN, fontweight="bold")

    labels = [f"#{r}  {n}  ({t})" for r, n, t in zip(top["rank"], top["player_name"], top["team"])]
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=10.2)
    ax.tick_params(axis="x", colors=TEXT_SUB, labelsize=10)
    ax.tick_params(axis="y", colors=TEXT_MAIN, length=0)
    ax.set_xlabel("Fox-in-the-Box Index  (0-100, blend of arithmetic / weighted / harmonic mean composites)",
                 fontsize=10.5, color=TEXT_MAIN, fontweight="bold", labelpad=10)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(True, axis="x", color=GRID_COLOR, linewidth=0.6, alpha=0.6, zorder=0)
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.7, len(top) - 0.2)

    fig.text(0.05, 0.975, "Fox in the Box", fontsize=22, fontweight="bold", color="white")
    fig.text(0.05, 0.951, f"Top {top_n} penalty-area poachers  ·  Eredivisie 2025/26  ·  min. 450 minutes",
             fontsize=11, color=TEXT_SUB)
    fig.text(0.05, 0.020, "Data via Opta | 16 metrics across Box Presence, Shot Volume & Instinct, Finishing "
             "Output and Clinical Efficiency, z-scored across the qualified pool then blended via arithmetic, "
             "weighted and harmonic mean composites", fontsize=7.6, color="#6b7684")
    fig.text(0.98, 0.014, "Marc Lamberts · Waltzing Analytics", fontsize=9, ha="right",
             color="#6b7684", style="italic")

    fig.subplots_adjust(left=0.30, right=0.94, top=0.905, bottom=0.09)
    add_logo(fig)
    fig.savefig(out_path, dpi=200, facecolor=BG)
    plt.close(fig)
    print("Saved:", out_path)


def make_composite_comparison(profile, out_path, top_n=10):
    """Shows how the three composite methods rank the same top players --
    the harmonic mean is the one most likely to demote a one-dimensional
    poacher relative to the arithmetic/weighted means."""
    top = profile.head(top_n)
    x = np.arange(len(top))
    width = 0.26

    fig, ax = plt.subplots(figsize=(14, 7.5))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    ax.bar(x - width, top["arithmetic_scaled"], width, color=BLUE, label="Arithmetic mean", zorder=3)
    ax.bar(x, top["weighted_scaled"], width, color=AMBER, label="Weighted mean", zorder=3)
    ax.bar(x + width, top["harmonic_scaled"], width, color=GREEN, label="Harmonic mean", zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{n} ({t})" for n, t in zip(top["player_name"], top["team"])],
                       fontsize=9, color=TEXT_MAIN, rotation=28, ha="right")
    ax.tick_params(axis="y", colors=TEXT_SUB, labelsize=10)
    ax.set_ylabel("Composite score (0-100, min-max scaled)", fontsize=11, color=TEXT_MAIN,
                 fontweight="bold", labelpad=10)
    ax.set_ylim(0, 105)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(True, axis="y", color=GRID_COLOR, linewidth=0.6, alpha=0.6, zorder=0)
    ax.legend(loc="upper right", frameon=False, fontsize=10, labelcolor=TEXT_MAIN)

    fig.text(0.06, 0.965, "Three Ways to Score a Fox in the Box", fontsize=20, fontweight="bold", color="white")
    fig.text(0.06, 0.935, f"Top {top_n} by final index  ·  arithmetic vs. weighted vs. harmonic mean composite",
             fontsize=11, color=TEXT_SUB)
    fig.text(0.06, 0.015, "Data via Opta | Harmonic mean is pulled down hardest by any single weak metric -- a "
             "big gap below the other two bars flags a one-dimensional scorer rather than an all-around poacher",
             fontsize=7.6, color="#6b7684")
    fig.text(0.98, 0.015, "Marc Lamberts · Waltzing Analytics", fontsize=9, ha="right",
             color="#6b7684", style="italic")

    fig.subplots_adjust(left=0.07, right=0.97, top=0.895, bottom=0.22)
    add_logo(fig)
    fig.savefig(out_path, dpi=200, facecolor=BG)
    plt.close(fig)
    print("Saved:", out_path)


def make_pillar_radar(profile, out_path, n=4):
    """Small pillar radar for the top n players -- shows *why* each one
    scores well (all-around vs. concentrated in one or two pillars)."""
    top = profile.head(n)
    angles = np.linspace(0, 2 * np.pi, len(PILLARS), endpoint=False).tolist()
    angles += angles[:1]

    fig, axes = plt.subplots(1, n, figsize=(4.6 * n, 5.2), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor(BG)
    if n == 1:
        axes = [axes]

    colors = [AMBER, BLUE, GREEN, RED]
    for i, (ax, row) in enumerate(zip(axes, top.to_dict("records"))):
        ax.set_facecolor(BG)
        values = [row[f"pillar_{p}"] for p in PILLARS]
        values += values[:1]
        ax.plot(angles, values, color=colors[i % len(colors)], linewidth=2, zorder=3)
        ax.fill(angles, values, color=colors[i % len(colors)], alpha=0.25, zorder=2)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([p.replace(" & ", "\n& ").replace(" ", "\n", 1) for p in PILLARS],
                           fontsize=8, color=TEXT_MAIN)
        ax.set_ylim(0, 100)
        ax.set_yticks([25, 50, 75])
        ax.set_yticklabels(["25", "50", "75"], fontsize=6.5, color=TEXT_SUB)
        ax.spines["polar"].set_color(GRID_COLOR)
        ax.grid(color=GRID_COLOR, linewidth=0.7)
        ax.set_title(f"#{row['rank']}  {row['player_name']}\n{row['team']}", fontsize=10.5,
                    color="white", fontweight="bold", pad=14)

    fig.text(0.5, 0.965, "Pillar Breakdown -- Fox in the Box Leaders", fontsize=17, fontweight="bold",
             ha="center", color="white")
    fig.text(0.98, 0.015, "Marc Lamberts · Waltzing Analytics", fontsize=8.5, ha="right",
             color="#6b7684", style="italic")
    fig.subplots_adjust(left=0.03, right=0.97, top=0.76, bottom=0.06, wspace=0.5)
    fig.savefig(out_path, dpi=200, facecolor=BG)
    plt.close(fig)
    print("Saved:", out_path)


def main():
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(METRICS_JSON) as f:
        df = pd.DataFrame(json.load(f))
    print(f"Loaded {len(df)} qualified players and their 16 metrics.")

    profile = build_profile(df)

    csv_path = out_dir / "fox_box_profile.csv"
    profile.to_csv(csv_path, index=False)
    print("Saved:", csv_path)

    json_path = out_dir / "fox_box_profile.json"
    with open(json_path, "w") as f:
        json.dump(profile.to_dict(orient="records"), f, indent=2)
    print("Saved:", json_path)

    print("\nTop 10 Fox-in-the-Box Index:")
    print(profile[["rank", "player_name", "team", "fox_in_the_box_index"]].head(10).to_string(index=False))

    make_leaderboard(profile, out_dir / "01_fox_in_the_box_leaderboard.png")
    make_composite_comparison(profile, out_dir / "02_composite_method_comparison.png")
    make_pillar_radar(profile, out_dir / "03_pillar_breakdown_top4.png")

    print("\nDone.")


if __name__ == "__main__":
    main()

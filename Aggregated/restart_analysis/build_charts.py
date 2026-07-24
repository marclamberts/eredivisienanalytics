"""
Charts for the goal-kick / open-play long-ball restart analysis, in Marc
Lamberts' Meridian house style (housestyle/ at the repo root).

Usage: python3 build_charts.py
"""
import csv
import os
import sys

import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
SEASON = "2025-2026"
SEASONS = ["2023-2024", "2024-2025", "2025-2026"]

sys.path.insert(0, ROOT)
from housestyle import style, components  # noqa: E402


def read_csv(path):
    with open(path, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def f(v):
    return float(v) if v not in (None, "") else None


# ---------------------------------------------------------------------
# Chart 1: RDV leaderboard, goal kicks, 2025-2026
# ---------------------------------------------------------------------
def chart_rdv_leaderboard(kind, label):
    rows = read_csv(os.path.join(ROOT, "Aggregated", SEASON, f"restart_{kind}_team.csv"))
    rows = [r for r in rows if f(r["rdv_core"]) is not None]
    rows.sort(key=lambda r: f(r["rdv_core"]))

    palette, _ = style.apply("light")
    fig = plt.figure(figsize=(9, 8.5))
    ax = fig.add_axes([0.30, 0.12, 0.64, 0.62])
    bars = ax.barh([r["team"] for r in rows], [f(r["rdv_core"]) for r in rows])
    components.highlight_bars(bars, accent_index=len(rows) - 1, palette=palette)
    ax.set_xlabel("Restart Dominance Value -- core (0-100)")
    ax.set_xlim(0, 100)

    leader = rows[-1]
    components.header(
        fig, kicker="Restart Dominance",
        title=f"{leader['team']} controls {label} restarts better than anyone else",
        dek=f"RDV-core (APER, ETG, FTS, HRR15, DTR weighted per the brief), {label}, {SEASON}",
        palette=palette,
    )
    components.footer(fig, source=f"Opta/StatsPerform {SEASON}", palette=palette)
    out = os.path.join(OUT_DIR, f"rdv_leaderboard_{kind}.png")
    fig.savefig(out, facecolor=fig.get_facecolor(), dpi=150)
    print("Wrote", out)


# ---------------------------------------------------------------------
# Chart 2: First-contact win rate, goal kick vs long ball, 3 seasons
# ---------------------------------------------------------------------
def chart_fcwr_comparison():
    comp = read_csv(os.path.join(OUT_DIR, "three_season_comparison.csv"))
    gk = next(r for r in comp if r["kind"] == "goal_kick" and r["metric"] == "fcwr")
    lb = next(r for r in comp if r["kind"] == "long_ball" and r["metric"] == "fcwr")

    palette, cats = style.apply("light")
    fig = plt.figure(figsize=(8.5, 6.2))
    ax = fig.add_axes([0.12, 0.16, 0.82, 0.56])

    x = range(len(SEASONS))
    w = 0.32
    gk_vals = [f(gk[s]) * 100 for s in SEASONS]
    lb_vals = [f(lb[s]) * 100 for s in SEASONS]
    bars_gk = ax.bar([i - w / 2 for i in x], gk_vals, width=w, color=cats[0], label="Goal kicks")
    bars_lb = ax.bar([i + w / 2 for i in x], lb_vals, width=w, color=palette["accent"], label="Open-play long balls")
    ax.set_xticks(list(x))
    ax.set_xticklabels(SEASONS)
    ax.set_ylabel("First-contact win rate (%)")
    ax.set_ylim(0, 100)
    ax.legend(loc="upper left", frameon=False, fontsize=9.5, bbox_to_anchor=(0.0, 1.12))

    components.header(
        fig, kicker="First Contact",
        title="Goal kicks are a far easier first ball to win than open-play long balls",
        dek=f"League-wide first-contact win rate by restart type, {SEASONS[0]} to {SEASONS[-1]}",
        palette=palette,
    )
    components.footer(fig, source="Opta/StatsPerform 2023-2026", palette=palette)
    out = os.path.join(OUT_DIR, "fcwr_goal_kick_vs_long_ball.png")
    fig.savefig(out, facecolor=fig.get_facecolor(), dpi=150)
    print("Wrote", out)


# ---------------------------------------------------------------------
# Chart 3: goal-kick territorial gain (CTG), 3-season decline
# ---------------------------------------------------------------------
def chart_ctg_trend():
    comp = read_csv(os.path.join(OUT_DIR, "three_season_comparison.csv"))
    row = next(r for r in comp if r["kind"] == "goal_kick" and r["metric"] == "ctg")
    vals = [f(row[s]) for s in SEASONS]

    palette, _ = style.apply("light")
    fig = plt.figure(figsize=(8, 6.2))
    ax = fig.add_axes([0.13, 0.16, 0.80, 0.56])
    bars = ax.bar(SEASONS, vals, width=0.55)
    components.highlight_bars(bars, accent_index=len(vals) - 1, palette=palette)
    for i, v in enumerate(vals):
        ax.annotate(f"{v:.1f}m", xy=(i, v), xytext=(0, 6), textcoords="offset points",
                    ha="center", fontsize=10, fontweight="bold", color=palette["ink_primary"])
    ax.set_ylabel("Controlled territorial gain (metres)")
    ax.set_ylim(0, max(vals) * 1.25)

    pct = float(row["pct_change"])
    direction = "less" if vals[-1] < vals[0] else "more"
    components.header(
        fig, kicker="Territorial Gain",
        title=f"Goal kicks are gaining {direction} ground than three seasons ago",
        dek=f"League-wide controlled territorial gain (CTG) per goal kick ({pct:.0f}% change, {SEASONS[0]} to {SEASONS[-1]})",
        palette=palette,
    )
    components.footer(fig, source="Opta/StatsPerform 2023-2026", palette=palette)
    out = os.path.join(OUT_DIR, "ctg_trend_goal_kick.png")
    fig.savefig(out, facecolor=fig.get_facecolor(), dpi=150)
    print("Wrote", out)


# ---------------------------------------------------------------------
# Chart 4: long-ball pressure-escape rate, 3-season trend
# ---------------------------------------------------------------------
def chart_pesr_trend():
    comp = read_csv(os.path.join(OUT_DIR, "three_season_comparison.csv"))
    row = next(r for r in comp if r["kind"] == "long_ball" and r["metric"] == "pesr")
    vals = [f(row[s]) * 100 for s in SEASONS]

    palette, cats = style.apply("light")
    fig = plt.figure(figsize=(9.3, 6.2))
    ax = fig.add_axes([0.11, 0.16, 0.82, 0.56])
    ax.plot(SEASONS, vals, marker="o", markersize=7, linewidth=2.2, color=cats[0])
    components.label_endpoint(ax, 2, vals[-1], f"{vals[-1]:.0f}%", palette["accent"], palette=palette)
    for i, v in enumerate(vals[:-1]):
        ax.annotate(f"{v:.0f}%", xy=(i, v), xytext=(0, -16), textcoords="offset points",
                    ha="center", fontsize=9.5, color=palette["ink_secondary"])
    ax.set_ylabel("Pressure-escape rate (%)")
    ax.set_ylim(0, 100)

    direction = "more" if vals[-1] > vals[0] else "less"
    components.header(
        fig, kicker="Pressure Escape",
        title=f"Long balls are escaping pressure {direction} often than three seasons ago",
        dek=f"League-wide long-ball pressure-escape rate, {SEASONS[0]} to {SEASONS[-1]}",
        palette=palette,
    )
    components.footer(fig, source="Opta/StatsPerform 2023-2026", palette=palette)
    out = os.path.join(OUT_DIR, "pesr_trend_long_ball.png")
    fig.savefig(out, facecolor=fig.get_facecolor(), dpi=150)
    print("Wrote", out)


if __name__ == "__main__":
    chart_rdv_leaderboard("goal_kick", "goal-kick")
    chart_rdv_leaderboard("long_ball", "long-ball")
    chart_fcwr_comparison()
    chart_ctg_trend()
    chart_pesr_trend()

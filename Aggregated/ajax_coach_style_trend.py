"""
Ajax playing style, match-by-match with a rolling average -- the continuous
companion to ajax_coach_style.py's per-regime bar comparison. Regime bars
show the level each coach settled at; this shows how abruptly (or not)
Ajax's identity actually moved when the touchline changed, since a coaching
change rarely rewrites a team's style from the very next kickoff.

Reuses ajax_coach_style.py's per-match event analysis directly (same
AJAX_CID, direction-correction, regime dates) rather than recomputing it --
see that file's docstring for the sourcing and methodology notes.

Usage: python3 ajax_coach_style_trend.py   (window size via ROLL_WINDOW below)
"""
import glob
import os
import sys

import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "Aggregated", "ajax_coach_style")

sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from housestyle import style, components  # noqa: E402
from ajax_coach_style import (  # noqa: E402
    AJAX_CID, REGIMES, analyse_match, find_regime, match_date,
)

ROLL_WINDOW = 5


def rolling_mean(values, window):
    out = []
    for i in range(len(values)):
        lo = max(0, i - window + 1)
        chunk = [v for v in values[lo:i + 1] if v is not None]
        out.append(sum(chunk) / len(chunk) if chunk else None)
    return out


def main():
    matches = []
    for season in ("2023-2024", "2024-2025", "2025-2026"):
        for path in sorted(glob.glob(os.path.join(ROOT, "Events", season, "*Ajax*.json"))):
            date_str = match_date(path)
            coach = find_regime(season, date_str)
            if coach is None:
                continue
            m = analyse_match(path)
            if m is None:
                continue
            matches.append({"season": season, "coach": coach, "date": date_str, "m": m})

    possession, ppda, long_ball = [], [], []
    for row in matches:
        m = row["m"]
        possession.append(m["own_pass_share"])
        ppda.append(m["opp_press_zone_passes"] / m["own_def_actions_press_zone"]
                     if m["own_def_actions_press_zone"] else None)
        long_ball.append(m["long_balls"] / m["passes"] * 100 if m["passes"] else None)

    poss_roll = rolling_mean(possession, ROLL_WINDOW)
    ppda_roll = rolling_mean(ppda, ROLL_WINDOW)
    lb_roll = rolling_mean(long_ball, ROLL_WINDOW)

    # transitions: index of the first match of each new regime (after the first)
    transitions = []
    prev_coach = None
    for i, row in enumerate(matches):
        if row["coach"] != prev_coach:
            if prev_coach is not None:
                transitions.append((i, row["coach"]))
            prev_coach = row["coach"]

    x = list(range(len(matches)))
    palette, cats = style.apply("light")

    fig = plt.figure(figsize=(13, 10.5))
    panels = [
        (poss_roll, possession, "Possession share (%)", cats[0]),
        (ppda_roll, ppda, "PPDA (lower = more pressing)", cats[2]),
        (lb_roll, long_ball, "Long balls (% of passes)", cats[4 % len(cats)]),
    ]
    top, bottom = 0.80, 0.10
    gap = 0.05
    panel_h = (top - bottom - gap * (len(panels) - 1)) / len(panels)

    for i, (roll, raw, ylabel, color) in enumerate(panels):
        ax_bottom = top - (i + 1) * panel_h - i * gap
        ax = fig.add_axes([0.08, ax_bottom, 0.88, panel_h])
        ax.scatter(x, raw, s=14, color=palette["axis"], alpha=0.5, zorder=2)
        ax.plot(x, roll, color=color, linewidth=2.4, zorder=4)
        for tx, coach in transitions:
            ax.axvline(tx - 0.5, color=palette["ink_muted"], linewidth=0.9,
                       linestyle=(0, (3, 3)), zorder=1)
        ax.set_ylabel(ylabel, fontsize=9.5)
        ax.set_xlim(-1, len(matches))
        if i < len(panels) - 1:
            ax.set_xticklabels([])
        ax.tick_params(axis="both", labelsize=8.5)

    # coach-name labels along the top panel only
    ax_top = fig.axes[0]
    prev_x = 0
    boundaries = transitions + [(len(matches), None)]
    for i, (tx, coach) in enumerate(boundaries):
        mid = (prev_x + tx) / 2
        label = matches[int(mid)]["coach"].split(" (")[0]
        width = tx - prev_x
        # stagger narrow segments onto a higher row so adjacent short spells
        # (e.g. Maduro's single match) don't collide with their neighbours
        y = 1.10 if width <= 3 else 1.04
        ax_top.annotate(label, xy=(mid, y), xycoords=("data", "axes fraction"),
                         fontsize=8.3, ha="center", color=palette["ink_secondary"],
                         fontweight="bold")
        prev_x = tx

    fig.axes[-1].set_xlabel("Match number (chronological, 2023-2024 -> 2025-2026)", fontsize=9.5)

    components.header(
        fig, kicker="Style Over Time",
        title="Ajax's style moved gradually, not overnight, with each coaching change",
        dek=f"{ROLL_WINDOW}-match rolling average per metric; dashed lines mark a coaching change; "
            "dots are individual match values",
        palette=palette, top=0.94,
    )
    components.footer(fig, source="Opta/StatsPerform 2023-2026", palette=palette)

    out_path = os.path.join(OUT_DIR, "ajax_coach_style_trend.png")
    fig.savefig(out_path, facecolor=fig.get_facecolor(), dpi=150)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()

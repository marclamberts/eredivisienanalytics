"""
League-wide Eredivisie season trends: open-play crosses per 90, corners
per 90, and shots created directly from a corner per 90 -- one point per
season (2023-24, 2024-25, 2025-26), one line graph.

Qualifier scheme matches Cross Models/score_eredivisie_crosses.py and
PSV Season Report/Scripts/corner_routines.py:
  - open play cross: typeId 1 (pass), qualifier 2 (CROSS) present,
    without qualifier 5 (FREEKICK_TAKEN), 6 (CORNER_TAKEN), or
    160 (THROWIN_SETPIECE)
  - corner: typeId 1 (pass), qualifier 6 (CORNER_TAKEN) present
  - shot: typeId in {13, 14, 15, 16} (miss, post, saved, goal)
  - shot created by a corner: shot's qualifier 55 (RELATED_EVENT_ID)
    points back to the corner event's eventId

Per-90 here is a team-match rate: totals / (matches * 2), since every
league match is a fixed 90(+) minutes for both teams.

Usage: python3 season_trends_crosses_corners.py [out.png]
"""
import sys
import glob
import json
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from housestyle import style, components

DATA_ROOT = Path(__file__).resolve().parents[1] / "Events"
SEASONS = ["2023-2024", "2024-2025", "2025-2026"]
SEASON_LABELS = {"2023-2024": "2023/24", "2024-2025": "2024/25", "2025-2026": "2025/26"}

CROSS_QID, FREEKICK_QID, CORNER_QID, THROWIN_QID = 2, 5, 6, 160
RELATED_EVENT_QID = 55
SHOT_TYPES = {13, 14, 15, 16}


def qids(e):
    return {q["qualifierId"] for q in e.get("qualifier", []) or []}


def qmap(e):
    return {q["qualifierId"]: q.get("value") for q in e.get("qualifier", []) or []}


def season_rates(season):
    files = sorted(glob.glob(str(DATA_ROOT / season / "*.json")))
    n_matches = 0
    open_play_crosses = 0
    corners = 0
    corner_shots = 0

    for fn in files:
        with open(fn) as f:
            data = json.load(f)
        events = [e for e in data.get("event", []) if e.get("periodId") in (1, 2)]
        if not events:
            continue
        n_matches += 1

        corner_event_ids = set()
        for e in events:
            if e.get("typeId") != 1:
                continue
            q = qids(e)
            if CORNER_QID in q:
                corners += 1
                corner_event_ids.add(e.get("eventId"))
            elif CROSS_QID in q and not (FREEKICK_QID in q or THROWIN_QID in q):
                open_play_crosses += 1

        for e in events:
            if e.get("typeId") not in SHOT_TYPES:
                continue
            rel = qmap(e).get(RELATED_EVENT_QID)
            if rel is None:
                continue
            try:
                rel = int(rel)
            except (TypeError, ValueError):
                continue
            if rel in corner_event_ids:
                corner_shots += 1

    team_matches = n_matches * 2
    return {
        "season": season,
        "matches": n_matches,
        "crosses_p90": open_play_crosses / team_matches,
        "corners_p90": corners / team_matches,
        "corner_shots_p90": corner_shots / team_matches,
    }


def make_plot(rows, out_path):
    palette, cats = style.apply("light")

    fig = plt.figure(figsize=(8.4, 5.8))
    ax = fig.add_axes([0.10, 0.24, 0.82, 0.54])

    x = list(range(len(rows)))
    labels = [SEASON_LABELS[r["season"]] for r in rows]
    crosses = [r["crosses_p90"] for r in rows]
    corners = [r["corners_p90"] for r in rows]
    corner_shots = [r["corner_shots_p90"] for r in rows]

    ax.plot(x, corners, marker="o", linewidth=2.0, color=cats[1], label="Corners / 90", zorder=2)
    ax.plot(x, corner_shots, marker="o", linewidth=2.0, color=cats[2], label="Shots from a corner / 90", zorder=2)
    lines_cross = ax.plot(x, crosses, marker="o", linewidth=2.6, label="Open-play crosses / 90", zorder=3)

    components.highlight_lines(ax, accent_index=2, palette=palette)

    for xi, yi in zip(x, crosses):
        ax.annotate(f"{yi:.1f}", (xi, yi), textcoords="offset points", xytext=(0, 10),
                    ha="center", fontsize=9, color=palette["ink_primary"], fontweight="bold")
    for xi, yi in zip(x, corners):
        ax.annotate(f"{yi:.1f}", (xi, yi), textcoords="offset points", xytext=(0, -16),
                    ha="center", fontsize=8.5, color=palette["ink_muted"])
    for xi, yi in zip(x, corner_shots):
        ax.annotate(f"{yi:.1f}", (xi, yi), textcoords="offset points", xytext=(0, 8),
                    ha="center", fontsize=8.5, color=palette["ink_muted"])

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlim(-0.3, len(x) - 0.7)
    ax.set_ylim(0, max(crosses) * 1.25)
    ax.set_ylabel("Per 90 (team-match average)")
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, -0.22), ncol=3, frameon=False,
              fontsize=9, handlelength=1.6, columnspacing=1.3)

    components.header(
        fig,
        kicker="Eredivisie",
        title="Open-play crosses per 90 have held steady across three seasons",
        dek="League-wide team-match average · open-play crosses, corners, and shots created from a corner",
        palette=palette,
    )
    components.footer(fig, source="Opta event data, Eredivisie 2023/24-2025/26 (all matches)", palette=palette)

    fig.savefig(out_path, dpi=200, facecolor=fig.get_facecolor())
    print("Saved:", out_path)


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else str(
        Path(__file__).resolve().parent / "14_crosses_corners_season_trend_eredivisie.png")
    rows = [season_rates(s) for s in SEASONS]
    for r in rows:
        print(f"{r['season']}: {r['matches']} matches | crosses/90={r['crosses_p90']:.2f} "
              f"corners/90={r['corners_p90']:.2f} corner_shots/90={r['corner_shots_p90']:.2f}")
    make_plot(rows, out_path)


if __name__ == "__main__":
    main()

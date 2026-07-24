"""
Wing play, overall: one bar per team, no left/right split -- just total
share of open-play passes from the wide corridor (Opta y<25 or y>75).

Reuses wing_play_by_team.csv (built by wing_play_comparison.py); run that
first if it doesn't exist yet for this season.

Usage: python3 wing_play_overall.py [season]   (default: 2025-2026)
"""
import csv
import os
import sys

import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEASON = sys.argv[1] if len(sys.argv) > 1 else "2025-2026"
OUT_DIR = os.path.join(ROOT, "Aggregated", SEASON)

sys.path.insert(0, ROOT)
from housestyle import style, components  # noqa: E402


def main():
    csv_path = os.path.join(OUT_DIR, "wing_play_by_team.csv")
    with open(csv_path, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    rows.sort(key=lambda r: float(r["wing_pct"]))  # ascending -> barh reads top-to-bottom descending

    palette, _ = style.apply("light")
    fig = plt.figure(figsize=(11, 8.5))
    ax = fig.add_axes([0.24, 0.12, 0.70, 0.64])

    bars = ax.barh([r["team"] for r in rows], [float(r["wing_pct"]) for r in rows])
    components.highlight_bars(bars, accent_index=len(rows) - 1, palette=palette)
    ax.set_xlabel("Share of open-play passes from the wide corridor (y<25 or y>75)")
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")

    leader = rows[-1]
    components.header(
        fig, kicker="Wing Play",
        title=f"{leader['team']} plays through the wings more than any other Eredivisie side",
        dek=f"Wide-corridor share of open-play passes (left + right combined), {SEASON}",
        palette=palette,
    )
    components.footer(fig, source=f"Opta/StatsPerform {SEASON}", palette=palette)

    out_path = os.path.join(OUT_DIR, "wing_play_overall.png")
    fig.savefig(out_path, facecolor=fig.get_facecolor(), dpi=150)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()

"""
Left vs right cross origin split, 2022/23 -> 2025/26
=====================================================
Where open-play crosses originate (left flank, y<50, vs right flank,
y>=50) across four Eredivisie seasons, and the average difference between
the first (2022/23) and most recent (2025/26) season. Built from the raw
Opta event data in "Eredivisie Events/<season>/*.json", using the same
open-play-cross definition as score_eredivisie_crosses.py's
extract_open_play_crosses (qualifier 2 = CROSS, excluding qualifiers 5/6/
160 = free kick/corner/throw-in). This trend uses cross volume/origin only
-- the six Ecuador-2026-trained cross models (delivery value, completion,
etc.) were fit and validated only on 2025/26, so they aren't reapplied to
the three earlier seasons here.

In Marc Lamberts' Meridian house style (housestyle/ package at repo root).

Usage: python3 cross_origin_left_vs_right_trend.py
"""
import glob
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from housestyle import style, components  # noqa: E402

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
EVENTS_DIR = os.path.join(os.path.dirname(OUT_DIR), "Eredivisie Events")
SEASONS = ["2022-2023", "2023-2024", "2024-2025", "2025-2026"]
SEASON_LABELS = ["2022/23", "2023/24", "2024/25", "2025/26"]
SOURCE = ("Opta event data, Eredivisie 2022/23-2025/26 · open-play crosses only · "
          "origin/volume only, not the 2025/26-only cross models")

CROSS_QID, FREEKICK_QID, CORNER_QID, THROWIN_QID = 2, 5, 6, 160


def qmap(e):
    return {q["qualifierId"]: q.get("value") for q in e.get("qualifier", []) or []}


def season_shares(season):
    n_left = n_right = 0
    for fn in glob.glob(os.path.join(EVENTS_DIR, season, "*.json")):
        with open(fn) as f:
            data = json.load(f)
        for e in data["event"]:
            if e.get("typeId") != 1 or e.get("periodId") not in (1, 2):
                continue
            q = qmap(e)
            if CROSS_QID not in q:
                continue
            if FREEKICK_QID in q or CORNER_QID in q or THROWIN_QID in q:
                continue
            y = e.get("y")
            if y is None:
                continue
            if y < 50:
                n_left += 1
            else:
                n_right += 1
    total = n_left + n_right
    return n_left / total * 100, n_right / total * 100, total


def make_chart(shares, mode, out_path):
    palette, _ = style.apply(mode)
    left_pct = [s[0] for s in shares]
    right_pct = [s[1] for s in shares]
    left_delta = left_pct[-1] - left_pct[0]
    right_delta = right_pct[-1] - right_pct[0]

    fig = plt.figure(figsize=(14.5, 8.2))

    # --- main panel: 4-season trend line -------------------------------
    ax = fig.add_axes([0.065, 0.14, 0.58, 0.56])
    xw = np.arange(len(SEASONS))
    ax.plot(xw, left_pct, marker="o", markersize=7, color=palette["accent"], linewidth=2.6, zorder=4)
    ax.plot(xw, right_pct, marker="o", markersize=7, color=palette["axis"], linewidth=2.2, zorder=3)
    ax.axhline(50, color=palette["grid"], linewidth=1.0, linestyle=(0, (3, 3)), zorder=1)

    for i, v in enumerate(left_pct):
        ax.text(i, v + 0.28, f"{v:.1f}%", ha="center", fontsize=9.5, color=palette["accent"], fontweight="bold")
    for i, v in enumerate(right_pct):
        ax.text(i, v - 0.42, f"{v:.1f}%", ha="center", fontsize=9.5, color=palette["ink_secondary"])

    components.label_endpoint(ax, xw[-1], left_pct[-1], "  Left flank", palette["accent"], palette=palette)
    components.label_endpoint(ax, xw[-1], right_pct[-1], "  Right flank", palette["ink_secondary"], palette=palette)

    ax.set_xticks(xw)
    ax.set_xticklabels(SEASON_LABELS, fontsize=10.5)
    ax.set_ylabel("Share of open-play crosses (%)", fontsize=10.5, color=palette["ink_secondary"])
    ax.set_xlim(-0.3, len(SEASONS) - 0.15)
    ax.set_ylim(min(right_pct) - 1.6, max(left_pct) + 1.6)
    ax.tick_params(colors=palette["ink_muted"])
    ax.grid(True, axis="y", color=palette["grid"], linewidth=0.7, zorder=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title("Origin share by season", fontsize=11.5, color=palette["ink_secondary"], loc="left", pad=10)

    # --- side panel: the 2022/23 -> 2025/26 difference -----------------
    ax2 = fig.add_axes([0.705, 0.20, 0.245, 0.44])
    cats = ["Left flank", "Right flank"]
    deltas = [left_delta, right_delta]
    bar_colors = [palette["accent"], palette["ink_secondary"]]
    yb = np.arange(len(cats))
    ax2.barh(yb, deltas, height=0.42, color=bar_colors, zorder=3)
    ax2.axvline(0, color=palette["axis"], linewidth=1.0, zorder=2)
    for i, d in enumerate(deltas):
        ax2.text(d + (0.03 if d >= 0 else -0.03), i, f"{d:+.1f}pp", va="center",
                 ha="left" if d >= 0 else "right", fontsize=10.5, fontweight="bold",
                 color=palette["ink_primary"])
    ax2.set_yticks(yb)
    ax2.set_yticklabels(cats, fontsize=10)
    lim = max(abs(min(deltas)), abs(max(deltas))) * 2.4
    ax2.set_xlim(-lim, lim)
    ax2.invert_yaxis()
    ax2.tick_params(axis="x", colors=palette["ink_muted"], labelsize=9)
    ax2.set_xlabel("Change in share, 2022/23 -> 2025/26 (pp)", fontsize=9.5, color=palette["ink_secondary"])
    for spine in ax2.spines.values():
        spine.set_visible(False)
    ax2.grid(False)
    ax2.set_title("Net change", fontsize=11.5, color=palette["ink_secondary"], loc="left", pad=10)

    components.header(
        fig, kicker="Crosses",
        title="The left-flank cross bias has barely moved in four seasons",
        dek=f"Eredivisie 2022/23-2025/26  ·  left flank's share: {left_pct[0]:.1f}% -> {left_pct[-1]:.1f}% "
            f"({left_delta:+.1f}pp)  ·  right: {right_pct[0]:.1f}% -> {right_pct[-1]:.1f}% ({right_delta:+.1f}pp)",
        palette=palette)
    components.footer(fig, source=SOURCE, palette=palette)

    fig.savefig(out_path, dpi=200, facecolor=palette["surface"])
    plt.close(fig)
    print("Saved:", out_path)


def main():
    shares = [season_shares(s) for s in SEASONS]
    for season, (l, r, n) in zip(SEASONS, shares):
        print(f"{season}: left {l:.2f}%  right {r:.2f}%  n={n}")

    for mode in ("light", "dark"):
        d = os.path.join(OUT_DIR, "Visual - Dark" if mode == "dark" else "Visual - Light")
        os.makedirs(d, exist_ok=True)
        out_path = os.path.join(d, "cross_origin_left_vs_right_trend_2022_2026.png")
        make_chart(shares, mode, out_path)


if __name__ == "__main__":
    main()

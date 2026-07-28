"""
Long throw-ins: how many reach the box and lead to a shot within 5 seconds
============================================================================
Eredivisie 2022/23 -> 2025/26, built directly from the raw Opta event data
in "Eredivisie Events/<season>/*.json".

Definitions:
  - Long throw-in: a pass event carrying qualifier 107 (throw-in) whose
    real-world length (using pitch dims 105x68m) is >= 25m -- the same
    threshold used elsewhere in this repo (PSV Season Report/Scripts/
    long_throwins.py).
  - Reaches the box: the throw's landing point (qualifier 140/141, end x/y)
    falls inside the penalty area (x >= 83.0, 21.1 <= y <= 78.9 -- the
    box definition from Box Entry Models/build_box_entry_model.py).
  - Leads to a shot within 5 seconds: any shot (typeId 13/14/15/16) by the
    throw-taking team occurs within 5 real seconds of the throw event
    (using the event timeStamp, not the truncated timeMin/timeSec), within
    the same match period.

The headline metric is the intersection of both: reaches the box AND a
shot follows within 5 seconds.

In Marc Lamberts' Meridian house style (housestyle/ package at repo root).

Usage: python3 long_throw_box_shot_trend.py
"""
import glob
import json
import os
import sys
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from housestyle import style, components  # noqa: E402

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
EVENTS_DIR = os.path.join(os.path.dirname(OUT_DIR), "Eredivisie Events")
SEASONS = ["2022-2023", "2023-2024", "2024-2025", "2025-2026"]
SEASON_LABELS = ["2022/23", "2023/24", "2024/25", "2025/26"]
SOURCE = ("Opta event data, Eredivisie 2022/23-2025/26 · long throw-ins (qualifier 107, >=25m) · "
          "box = x>=83, 21.1<=y<=78.9 · shot window = 5s")

THROW_IN_QID = 107
PASS_END_X_QID, PASS_END_Y_QID = 140, 141
MIN_DIST_M = 25.0
SHOT_TYPES = {13, 14, 15, 16}
WINDOW_SEC = 5.0
BOX_X = 83.0
BOX_Y_LO, BOX_Y_HI = 21.1, 78.9


def qmap(e):
    return {q["qualifierId"]: q.get("value") for q in e.get("qualifier", []) or []}


def dist_m(x0, y0, x1, y1):
    dx = (x1 - x0) / 100 * 105.0
    dy = (y1 - y0) / 100 * 68.0
    return (dx ** 2 + dy ** 2) ** 0.5


def ts(e):
    return datetime.fromisoformat(e["timeStamp"])


def in_box(x, y):
    return x >= BOX_X and BOX_Y_LO <= y <= BOX_Y_HI


def season_stats(season):
    n = n_box = n_box_shot = 0
    for fn in glob.glob(os.path.join(EVENTS_DIR, season, "*.json")):
        with open(fn) as f:
            data = json.load(f)
        events = [e for e in data["event"] if e.get("periodId") in (1, 2)]
        events.sort(key=lambda e: (e["periodId"], e.get("timeMin", 0), e.get("timeSec", 0), e.get("eventId", 0)))

        for idx, e in enumerate(events):
            if e.get("typeId") != 1:
                continue
            q = qmap(e)
            if THROW_IN_QID not in q:
                continue
            x0, y0 = float(e["x"]), float(e["y"])
            x1 = float(q.get(PASS_END_X_QID, x0))
            y1 = float(q.get(PASS_END_Y_QID, y0))
            if dist_m(x0, y0, x1, y1) < MIN_DIST_M:
                continue

            n += 1
            box = in_box(x1, y1)
            if box:
                n_box += 1

            cid, period, t0 = e.get("contestantId"), e.get("periodId"), ts(e)
            shot5 = False
            for nxt in events[idx + 1: idx + 40]:
                if nxt.get("periodId") != period:
                    break
                dt = (ts(nxt) - t0).total_seconds()
                if dt > WINDOW_SEC:
                    break
                if dt >= 0 and nxt.get("typeId") in SHOT_TYPES and nxt.get("contestantId") == cid:
                    shot5 = True
                    break
            if box and shot5:
                n_box_shot += 1

    return n, n_box / n * 100, n_box_shot / n * 100


def make_chart(stats, mode, out_path):
    palette, _ = style.apply(mode)
    box_pct = [s[1] for s in stats]
    combo_pct = [s[2] for s in stats]
    box_delta = box_pct[-1] - box_pct[0]
    combo_delta = combo_pct[-1] - combo_pct[0]

    fig = plt.figure(figsize=(14.5, 8.2))

    # --- main panel: 4-season trend line -------------------------------
    ax = fig.add_axes([0.065, 0.14, 0.58, 0.56])
    xw = np.arange(len(SEASONS))
    ax.plot(xw, box_pct, marker="o", markersize=7, color=palette["axis"], linewidth=2.2, zorder=3)
    ax.plot(xw, combo_pct, marker="o", markersize=7, color=palette["accent"], linewidth=2.6, zorder=4)

    for i, v in enumerate(box_pct):
        ax.text(i, v + 0.45, f"{v:.1f}%", ha="center", fontsize=9.5, color=palette["ink_secondary"])
    for i, v in enumerate(combo_pct):
        ax.text(i, v - 0.75, f"{v:.1f}%", ha="center", fontsize=9.5, color=palette["accent"], fontweight="bold")

    components.label_endpoint(ax, xw[-1], box_pct[-1], "  Reaches the box", palette["ink_secondary"], palette=palette)
    components.label_endpoint(ax, xw[-1], combo_pct[-1], "  ...and a shot within 5s", palette["accent"], palette=palette)

    ax.set_xticks(xw)
    ax.set_xticklabels(SEASON_LABELS, fontsize=10.5)
    ax.set_ylabel("Share of long throw-ins (%)", fontsize=10.5, color=palette["ink_secondary"])
    ax.set_xlim(-0.3, len(SEASONS) - 0.1)
    ax.set_ylim(0, max(box_pct) + 4)
    ax.tick_params(colors=palette["ink_muted"])
    ax.grid(True, axis="y", color=palette["grid"], linewidth=0.7, zorder=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title("Long throw-in outcomes by season", fontsize=11.5, color=palette["ink_secondary"],
                loc="left", pad=10)

    # --- side panel: the 2022/23 -> 2025/26 difference -----------------
    ax2 = fig.add_axes([0.705, 0.20, 0.245, 0.44])
    cats = ["Reaches\nthe box", "...+ shot\nwithin 5s"]
    deltas = [box_delta, combo_delta]
    bar_colors = [palette["ink_secondary"], palette["accent"]]
    yb = np.arange(len(cats))
    ax2.barh(yb, deltas, height=0.42, color=bar_colors, zorder=3)
    ax2.axvline(0, color=palette["axis"], linewidth=1.0, zorder=2)
    for i, d in enumerate(deltas):
        ax2.text(d + (0.03 if d >= 0 else -0.03), i, f"{d:+.1f}pp", va="center",
                 ha="left" if d >= 0 else "right", fontsize=10.5, fontweight="bold",
                 color=palette["ink_primary"])
    ax2.set_yticks(yb)
    ax2.set_yticklabels(cats, fontsize=10)
    lim = max(abs(d) for d in deltas) * 2.6
    ax2.set_xlim(-lim, lim)
    ax2.invert_yaxis()
    ax2.tick_params(axis="x", colors=palette["ink_muted"], labelsize=9)
    ax2.set_xlabel("Change, 2022/23 -> 2025/26 (pp)", fontsize=9.5, color=palette["ink_secondary"])
    for spine in ax2.spines.values():
        spine.set_visible(False)
    ax2.set_title("Net change", fontsize=11.5, color=palette["ink_secondary"], loc="left", pad=10)

    components.header(
        fig, kicker="Long throw-ins",
        title="Barely 2% of long throw-ins turn into a shot within 5 seconds",
        dek=f"Eredivisie 2022/23-2025/26  ·  reaches the box: {box_pct[0]:.1f}% -> {box_pct[-1]:.1f}%  ·  "
            f"box + shot within 5s: {combo_pct[0]:.1f}% -> {combo_pct[-1]:.1f}%  ({combo_delta:+.1f}pp)",
        palette=palette)
    components.footer(fig, source=SOURCE, palette=palette)

    fig.savefig(out_path, dpi=200, facecolor=palette["surface"])
    plt.close(fig)
    print("Saved:", out_path)


def main():
    stats = []
    for s in SEASONS:
        n, box_pct, combo_pct = season_stats(s)
        stats.append((n, box_pct, combo_pct))
        print(f"{s}: n={n}  box={box_pct:.2f}%  box+shot<=5s={combo_pct:.2f}%")

    for mode in ("light", "dark"):
        d = os.path.join(OUT_DIR, "Visual - Dark" if mode == "dark" else "Visual - Light")
        os.makedirs(d, exist_ok=True)
        out_path = os.path.join(d, "long_throw_box_shot_trend_2022_2026.png")
        make_chart(stats, mode, out_path)


if __name__ == "__main__":
    main()

"""
Where do most Eredivisie shots actually come from? Grids all non-penalty
shot locations from the season's scored Danger models (Danger/*.csv, one
row per shot with x, y, xg, is_goal), finds the highest-danger contiguous
patch inside the penalty box (peak-xG cell grown to its neighbours), and
saves that patch as a reusable "hot zone" for passes_to_hot_zone.py.

Usage: python3 shot_hotspot_analysis.py [out_dir]
"""
import sys
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mplsoccer import VerticalPitch

DANGER_DIR = "/home/user/eredivisienanalytics/Danger"
LOGO_PATH = "/Users/marclamberts/Downloads/Waltzing Analytics Logo Type.png"

BG = "#0d1117"
PITCH_LINE = "#2c3a4d"
TEXT_MAIN = "#e6e9ee"
TEXT_SUB = "#9aa4b2"
AMBER = "#ffc247"

BOX_X = 83.0
BOX_Y_LO, BOX_Y_HI = 21.1, 78.9
GRID_N = 6           # box divided into a GRID_N x GRID_N grid
GROW_THRESHOLD = 0.65  # neighbour cells within 65% of the peak cell's xg join the hot zone


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


def load_shots():
    files = sorted(glob.glob(f"{DANGER_DIR}/*.csv"))
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    return df[df["is_penalty"] == 0].copy()


def find_hot_zone(box_shots):
    """Bin the box into a GRID_N x GRID_N grid of total xG, then grow the
    peak cell into its 4-connected neighbours while they stay within
    GROW_THRESHOLD of the peak, and return the bounding box of that
    contiguous patch."""
    x_edges = np.linspace(BOX_X, 100, GRID_N + 1)
    y_edges = np.linspace(BOX_Y_LO, BOX_Y_HI, GRID_N + 1)
    xi = np.clip(np.digitize(box_shots["x"], x_edges) - 1, 0, GRID_N - 1)
    yi = np.clip(np.digitize(box_shots["y"], y_edges) - 1, 0, GRID_N - 1)

    xg_grid = np.zeros((GRID_N, GRID_N))
    for i, j, xg in zip(xi, yi, box_shots["xg"]):
        xg_grid[i, j] += xg

    peak = np.unravel_index(np.argmax(xg_grid), xg_grid.shape)
    threshold = xg_grid[peak] * GROW_THRESHOLD

    visited = {peak}
    frontier = [peak]
    while frontier:
        i, j = frontier.pop()
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ni, nj = i + di, j + dj
            if 0 <= ni < GRID_N and 0 <= nj < GRID_N and (ni, nj) not in visited:
                if xg_grid[ni, nj] >= threshold:
                    visited.add((ni, nj))
                    frontier.append((ni, nj))

    is_ = [c[0] for c in visited]
    js_ = [c[1] for c in visited]
    zone = {
        "x_lo": float(x_edges[min(is_)]), "x_hi": float(x_edges[max(is_) + 1]),
        "y_lo": float(y_edges[min(js_)]), "y_hi": float(y_edges[max(js_) + 1]),
    }
    return zone, xg_grid, x_edges, y_edges


def zone_stats(df, zone):
    hot = df[(df["x"] >= zone["x_lo"]) & (df["x"] <= zone["x_hi"]) &
             (df["y"] >= zone["y_lo"]) & (df["y"] <= zone["y_hi"])]
    area_share = ((zone["x_hi"] - zone["x_lo"]) * (zone["y_hi"] - zone["y_lo"])) / \
                 ((100 - BOX_X) * (BOX_Y_HI - BOX_Y_LO))
    return {
        "n_shots": int(len(hot)),
        "shot_share": float(len(hot) / len(df)),
        "xg_share": float(hot["xg"].sum() / df["xg"].sum()),
        "goal_share": float(hot["is_goal"].sum() / df["is_goal"].sum()),
        "avg_xg_per_shot": float(hot["xg"].mean()),
        "avg_xg_per_shot_league": float(df["xg"].mean()),
        "box_area_share": float(area_share),
    }


def make_heatmap(df, zone, out_path):
    fig = plt.figure(figsize=(11, 12.6))
    fig.patch.set_facecolor(BG)

    fig.text(0.5, 0.965, "Where Eredivisie Shots Come From", fontsize=24, fontweight="bold", ha="center", color="white")
    fig.text(0.5, 0.935, "Non-penalty shot locations, season total  ·  Eredivisie 2025/26",
             fontsize=12, ha="center", color=TEXT_SUB)

    pitch_ax = fig.add_axes([0.04, 0.08, 0.92, 0.80])
    pitch = VerticalPitch(pitch_type="opta", pitch_color=BG, line_color=PITCH_LINE, linewidth=1.1, half=True)
    pitch.draw(ax=pitch_ax)

    bs = pitch.bin_statistic(df["x"], df["y"], values=df["xg"], statistic="sum", bins=(24, 18))
    pitch.heatmap(bs, ax=pitch_ax, cmap="inferno", alpha=0.85, zorder=1)

    pitch.polygon([[[zone["x_lo"], zone["y_lo"]], [zone["x_lo"], zone["y_hi"]],
                    [zone["x_hi"], zone["y_hi"]], [zone["x_hi"], zone["y_lo"]]]],
                  ax=pitch_ax, facecolor="none", edgecolor=AMBER, linewidth=2.4, zorder=3)
    pitch.annotate("HOT ZONE", xy=((zone["y_lo"] + zone["y_hi"]) / 2, zone["x_hi"] + 1.5),
                   ax=pitch_ax, ha="center", va="bottom", fontsize=11, fontweight="bold", color=AMBER, zorder=4)

    fig.text(0.98, 0.02, "Marc Lamberts", fontsize=9.5, ha="right", color="#6b7684", style="italic")
    fig.text(0.02, 0.02, "Data via Opta | Eredivisie 2025/26 shots via the Danger models, excludes penalties · "
             "heatmap = total xG per cell", fontsize=7.3, color="#6b7684")

    add_logo(fig)
    fig.savefig(out_path, dpi=200, facecolor=BG)
    plt.close(fig)
    print("Saved:", out_path)


def main():
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_shots()
    print(f"Loaded {len(df)} non-penalty shots.")

    box_shots = df[(df["x"] >= BOX_X) & (df["y"] >= BOX_Y_LO) & (df["y"] <= BOX_Y_HI)]
    print(f"{len(box_shots)} shots ({len(box_shots) / len(df) * 100:.1f}%) come from inside the box, "
          f"carrying {box_shots['xg'].sum() / df['xg'].sum() * 100:.1f}% of total xG.")

    zone, xg_grid, x_edges, y_edges = find_hot_zone(box_shots)
    stats = zone_stats(df, zone)

    print(f"\nHot zone: x in [{zone['x_lo']:.1f}, {zone['x_hi']:.1f}], "
          f"y in [{zone['y_lo']:.1f}, {zone['y_hi']:.1f}]")
    print(f"  {stats['box_area_share'] * 100:.1f}% of the box's area produces "
          f"{stats['shot_share'] * 100:.1f}% of shots, {stats['xg_share'] * 100:.1f}% of xG, "
          f"{stats['goal_share'] * 100:.1f}% of goals")
    print(f"  avg xG/shot in zone: {stats['avg_xg_per_shot']:.3f} vs. league avg {stats['avg_xg_per_shot_league']:.3f}")

    meta = {"zone": zone, "stats": stats, "grid_n": GRID_N, "grow_threshold": GROW_THRESHOLD,
            "box_x": BOX_X, "box_y_lo": BOX_Y_LO, "box_y_hi": BOX_Y_HI}
    meta_path = out_dir / "shot_hot_zone.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print("\nSaved:", meta_path)

    make_heatmap(df, zone, out_dir / "03_shot_hotspot_eredivisie.png")


if __name__ == "__main__":
    main()

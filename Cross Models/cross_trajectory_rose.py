"""
Cross trajectory rose: what angle crosses are actually hit at
================================================================
Eredivisie 2025/26 — a polar "wind rose" of open-play cross delivery
angle, built from the cross models' scored output
(eredivisie_open_play_cross_events_scored.csv). angle is Opta's raw pass
angle (qualifier 213, radians, standard math convention: 0 deg = straight
toward the byline, 90 deg = square across the pitch, 180 deg = pulled
back). Right-flank angles are mirrored (theta -> -theta) onto the same
compass as the left flank so the two roses are directly comparable --
almost all crosses fall within a ~0-140 deg arc (forward-and-across),
so both roses are drawn as a half-circle fan rather than a full 360 deg
rose to avoid two-thirds of dead space.

Bar length = share of that flank's crosses in the 10 deg bin (frequency).
Bar color = mean pred_cross_delivery_value in the bin (the cross model's
composite quality score) -- so the same chart shows how often a bin is
hit AND how much it's worth. The single highest-value bin per flank is
outlined in the house accent and called out directly.

In Marc Lamberts' Meridian house style (housestyle/ package at repo root).

Usage: python3 cross_trajectory_rose.py
"""
import os
import sys

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from housestyle import style, components, colors  # noqa: E402

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(OUT_DIR, "eredivisie_open_play_cross_events_scored.csv")
SOURCE = ("Opta event data, Eredivisie 2025/26 · open-play crosses only · scored with the "
          "Ecuador 2026-trained cross models · right flank mirrored onto the left's compass")

BIN_WIDTH_DEG = 10
N_BINS = 180 // BIN_WIDTH_DEG
MIN_N = 15  # bins below this are too sparse to trust a mean on


def seq_cmap(mode):
    seq = colors.SEQUENTIAL_BLUE_DARK if mode == "dark" else colors.SEQUENTIAL_BLUE_LIGHT
    hexes = [seq[k] for k in sorted(seq)]
    return LinearSegmentedColormap.from_list("meridian_blue", hexes)


def prep(df):
    deg = np.degrees(df["angle"]) % 360
    df = df.copy()
    df["mdeg"] = np.where(df["wide_channel"] == "right", (-deg) % 360, deg)
    edges = np.arange(0, 181, BIN_WIDTH_DEG)
    df["bin"] = pd.cut(df["mdeg"], bins=edges, right=False, include_lowest=True, labels=False)
    return df, edges


def bin_stats(df, side, edges):
    sub = df[df["wide_channel"] == side]
    total = len(sub)
    g = sub.groupby("bin").agg(n=("x", "size"), dv=("pred_cross_delivery_value", "mean"))
    g = g.reindex(range(len(edges) - 1), fill_value=0)
    g["dv"] = g["dv"].fillna(0)
    g["share"] = g["n"] / total * 100
    return g, total


def draw_rose(ax, g, edges, cmap, norm, palette, title):
    theta = np.radians(edges[:-1] + BIN_WIDTH_DEG / 2)
    width = np.radians(BIN_WIDTH_DEG * 0.92)

    peak_idx = g[g["n"] >= MIN_N]["dv"].idxmax()
    modal_idx = g["n"].idxmax()

    for i, row in g.iterrows():
        if row["n"] < MIN_N:
            color, alpha, ec, lw = palette["grid"], 0.55, palette["axis"], 0.6
        else:
            color, alpha, ec, lw = cmap(norm(row["dv"])), 0.98, palette["surface"], 0.6
        if i == peak_idx:
            ec, lw = palette["accent"], 2.6
        ax.bar(theta[i], row["share"], width=width, bottom=0, color=color, alpha=alpha,
               edgecolor=ec, linewidth=lw, zorder=3)

    ax.set_thetamin(0)
    ax.set_thetamax(180)
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(1)
    ax.set_thetagrids([0, 45, 90, 135, 180],
                      labels=["Toward\nbyline", "45°", "Square\nball", "135°", "Pulled\nback"],
                      fontsize=9.5, color=palette["ink_secondary"])
    rmax = g["share"].max() * 1.35
    ax.set_rlim(0, rmax)
    ax.set_rgrids([rmax / 3, rmax * 2 / 3, rmax],
                 labels=[f"{rmax / 3:.0f}%", f"{rmax * 2 / 3:.0f}%", f"{rmax:.0f}%"], angle=135)
    ax.tick_params(axis="y", colors=palette["ink_muted"], labelsize=8)
    ax.grid(color=palette["grid"], linewidth=0.7)
    ax.spines["polar"].set_color(palette["axis"])
    ax.set_facecolor(palette["surface"])
    ax.set_title(title, fontsize=12, color=palette["ink_primary"], pad=18, fontweight="bold")
    return peak_idx, modal_idx


def caption_for(g, edges, peak_idx, modal_idx):
    peak_lo, peak_hi = edges[peak_idx], edges[peak_idx] + BIN_WIDTH_DEG
    modal_lo, modal_hi = edges[modal_idx], edges[modal_idx] + BIN_WIDTH_DEG
    line1 = f"Most common: {modal_lo:.0f}-{modal_hi:.0f}°  ({g.loc[modal_idx, 'share']:.1f}% of crosses)"
    line2 = (f"Most valuable: {peak_lo:.0f}-{peak_hi:.0f}°  ({g.loc[peak_idx, 'dv']:.3f}, "
            f"+{(g.loc[peak_idx, 'dv'] / g.loc[modal_idx, 'dv'] - 1) * 100:.0f}% vs. the common angle)")
    return line1, line2


def make_chart(df, mode, out_path):
    palette, _ = style.apply(mode)
    df, edges = prep(df)
    g_left, n_left = bin_stats(df, "left", edges)
    g_right, n_right = bin_stats(df, "right", edges)

    vmax = max(g_left["dv"].max(), g_right["dv"].max())
    cmap = seq_cmap(mode)
    norm = Normalize(vmin=0, vmax=vmax)

    fig = plt.figure(figsize=(14.5, 10.0))
    ax_left = fig.add_axes([0.045, 0.335, 0.44, 0.46], projection="polar")
    ax_right = fig.add_axes([0.515, 0.335, 0.44, 0.46], projection="polar")

    peak_l, modal_l = draw_rose(ax_left, g_left, edges, cmap, norm, palette, f"Left flank  ·  {n_left:,} crosses")
    peak_r, modal_r = draw_rose(ax_right, g_right, edges, cmap, norm, palette, f"Right flank  ·  {n_right:,} crosses")

    for x0, g, pk, md in ((0.045, g_left, peak_l, modal_l), (0.515, g_right, peak_r, modal_r)):
        line1, line2 = caption_for(g, edges, pk, md)
        fig.text(x0 + 0.22, 0.258, line1, ha="center", va="top", fontsize=9.5,
                 color=palette["ink_secondary"], family="sans-serif")
        fig.text(x0 + 0.22, 0.232, line2, ha="center", va="top", fontsize=9.5,
                 color=palette["ink_secondary"], family="sans-serif")

    cax = fig.add_axes([0.40, 0.155, 0.20, 0.016])
    sm = ScalarMappable(norm=norm, cmap=cmap)
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cb.set_label("Mean predicted cross delivery value", fontsize=9, color=palette["ink_secondary"])
    cb.ax.tick_params(labelsize=8.5, colors=palette["ink_muted"])
    cb.outline.set_visible(False)

    components.header(
        fig, kicker="Crosses",
        title="The most common cross angle isn't the most dangerous one",
        dek="Eredivisie 2025/26  ·  delivery angle into the box (bar length = share of crosses, "
            "color = predicted value)  ·  right flank mirrored onto the left's compass",
        palette=palette)
    components.footer(fig, source=SOURCE, palette=palette)

    fig.savefig(out_path, dpi=200, facecolor=palette["surface"])
    plt.close(fig)
    print("Saved:", out_path)


def main():
    df = pd.read_csv(CSV_PATH)
    for mode in ("light", "dark"):
        d = os.path.join(OUT_DIR, "Visual - Dark" if mode == "dark" else "Visual - Light")
        os.makedirs(d, exist_ok=True)
        out_path = os.path.join(d, "cross_trajectory_rose_2025_26.png")
        make_chart(df, mode, out_path)


if __name__ == "__main__":
    main()

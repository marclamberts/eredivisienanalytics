"""Eredivisie power rating chart, Meridian house style.

Reads Opta's global club power rankings, filters to Eredivisie clubs, and
renders a horizontal bar chart (light + dark) highlighting the league leader.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from housestyle import style, components

DATA_PATH = Path(__file__).resolve().parent / "opta_club_rankings.xlsx"
OUT_DIR = Path(__file__).resolve().parent


def load_eredivisie_ratings() -> pd.DataFrame:
    df = pd.read_excel(DATA_PATH)
    ered = df[df["league"] == "Eredivisie"].copy()
    # Descending by rank (worst first) so the best (lowest) rank lands last,
    # i.e. at the top of the horizontal bar chart.
    ered = ered.sort_values("rank", ascending=False).reset_index(drop=True)
    return ered


def build_chart(df: pd.DataFrame, mode: str) -> None:
    palette, cats = style.apply(mode)

    fig = plt.figure(figsize=(8.4, 8.6))
    ax = fig.add_axes([0.30, 0.10, 0.60, 0.66])

    leader = df["team"].iloc[-1]
    worst_rank = df["rank"].max()
    # Invert rank into bar length so the best (lowest-numbered) global rank
    # reads as the longest, most prominent bar; the real rank is annotated.
    bar_values = worst_rank - df["rank"] + 10
    bars = ax.barh(df["team"], bar_values, height=0.62)
    accent_index = len(df) - 1  # leader sits last after sort (top of chart)
    components.highlight_bars(bars, accent_index=accent_index, palette=palette)

    ax.set_xlim(0, bar_values.max() * 1.12)
    ax.set_xticks([])
    ax.set_xlabel("Global club rank (Opta, all leagues worldwide) — lower is better")
    ax.set_ylabel("")
    ax.grid(axis="x", visible=False)
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", labelsize=10)

    for i, (team, rank, value) in enumerate(zip(df["team"], df["rank"], bar_values)):
        color = palette["accent"] if i == accent_index else palette["ink_secondary"]
        weight = "bold" if i == accent_index else "normal"
        ax.text(value + bar_values.max() * 0.012, i, f"#{rank:,}", va="center", ha="left",
                 fontsize=9.5, color=color, fontweight=weight)

    components.header(
        fig,
        kicker="Power Rating",
        title=f"{leader} is the only Eredivisie club inside the world's top 50",
        dek="Opta global club power ranking (#1 = best of ~13,800 rated clubs), current Eredivisie clubs",
        palette=palette,
    )
    components.footer(fig, source="Opta club power rankings", palette=palette)

    suffix = "dark" if mode == "dark" else "light"
    out_path = OUT_DIR / f"eredivisie_power_rating_{suffix}.png"
    fig.savefig(out_path, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    df = load_eredivisie_ratings()
    build_chart(df, "light")
    build_chart(df, "dark")


if __name__ == "__main__":
    main()

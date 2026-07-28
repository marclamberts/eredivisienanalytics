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
    ered = ered.sort_values("rating", ascending=True).reset_index(drop=True)
    return ered


def build_chart(df: pd.DataFrame, mode: str) -> None:
    palette, cats = style.apply(mode)

    fig = plt.figure(figsize=(8.4, 8.6))
    ax = fig.add_axes([0.30, 0.10, 0.60, 0.66])

    leader = df["team"].iloc[-1]
    bars = ax.barh(df["team"], df["rating"], height=0.62)
    accent_index = len(df) - 1  # leader sits last after ascending sort (top of chart)
    components.highlight_bars(bars, accent_index=accent_index, palette=palette)

    ax.set_xlim(70, 92)
    ax.set_xlabel("Opta power rating (0-100 global scale)")
    ax.set_ylabel("")
    ax.grid(axis="x", visible=True)
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", labelsize=10)

    for i, (team, rating) in enumerate(zip(df["team"], df["rating"])):
        color = palette["accent"] if i == accent_index else palette["ink_secondary"]
        weight = "bold" if i == accent_index else "normal"
        ax.text(rating + 0.25, i, f"{rating:.1f}", va="center", ha="left",
                 fontsize=9.5, color=color, fontweight=weight)

    components.header(
        fig,
        kicker="Power Rating",
        title=f"{leader} stands alone atop the Eredivisie",
        dek="Opta club power ratings (0-100 global scale), current Eredivisie clubs",
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

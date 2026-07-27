"""
Top 5 Eerste Divisie 2025/26 players by total xG (Waltzing Analytics xG
Model), each shown as cumulative goals vs cumulative xG across the
matches in which they registered a shot -- the classic "is this player
over- or under-performing their shot quality" read.

Usage: python3 eerste_divisie_top_scorers_xg.py [out.png] [top_n]
"""
import glob
import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from housestyle import style, components
from xg_model import load_xg_model, extract_shots, score_shots

REPO_ROOT = Path(__file__).resolve().parent.parent
EERSTE_DIVISIE_DIR = REPO_ROOT / "Eerste Divisie Events" / "Eerste Divisie 2025-2026"


def top_scorers(df, top_n):
    totals = (df.groupby(["player", "player_id", "contestant_id"])
              .agg(xg=("xg", "sum"), goals=("is_goal", "sum"), shots=("is_goal", "size"))
              .reset_index()
              .sort_values("xg", ascending=False))
    return totals.head(top_n)


def cumulative_series(df, player_id):
    sub = df[df["player_id"] == player_id].sort_values(["date", "match_file"])
    match_order = {d: i for i, d in enumerate(sorted(sub["date"].unique()), start=1)}
    sub = sub.assign(match_idx=sub["date"].map(match_order))
    per_match = sub.groupby("match_idx").agg(xg=("xg", "sum"), goals=("is_goal", "sum")).reset_index()
    per_match = per_match.sort_values("match_idx")
    cum_xg = per_match["xg"].cumsum().tolist()
    cum_goals = per_match["goals"].cumsum().tolist()
    return per_match["match_idx"].tolist(), cum_xg, cum_goals


def make_plot(df, top, out_path):
    palette, cats = style.apply("dark")

    fig = plt.figure(figsize=(13, 9.2))
    grid_top, grid_bottom = 0.58, 0.11
    n = len(top)
    ncols = 3
    nrows = -(-n // ncols)
    gs = fig.add_gridspec(nrows, ncols, left=0.07, right=0.96,
                          top=grid_top, bottom=grid_bottom, hspace=1.05, wspace=0.28)

    for i, (_, row) in enumerate(top.iterrows()):
        ax = fig.add_subplot(gs[i // ncols, i % ncols])
        xs, cum_xg, cum_goals = cumulative_series(df, row["player_id"])
        ax.plot(xs, cum_xg, linewidth=1.8, color=palette["axis"], zorder=2)
        ax.plot(xs, cum_goals, linewidth=2.4, color=palette["accent"], zorder=3)

        delta = row["goals"] - row["xg"]
        sign = "+" if delta >= 0 else ""
        ax.set_title(row["player"], fontsize=11.5, color=palette["ink_primary"],
                     fontweight="bold", loc="left", pad=28)
        ax.text(0.0, 1.06, f"{int(row['goals'])} goals from {row['xg']:.1f} xG ({sign}{delta:.1f})",
                transform=ax.transAxes, fontsize=8.5, color=palette["ink_secondary"],
                ha="left", va="bottom")

        ax.set_xlabel("Matches with a shot", fontsize=8, color=palette["ink_secondary"])
        ax.tick_params(labelsize=7.5)
        ax.grid(True, axis="y", alpha=0.5)

    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], color=palette["accent"], linewidth=2.4, label="Cumulative goals"),
        Line2D([0], [0], color=palette["axis"], linewidth=1.8, label="Cumulative xG"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, grid_top + 0.14),
               ncol=2, frameon=False, fontsize=10, labelcolor=palette["ink_primary"])

    components.header(
        fig,
        kicker="Eerste Divisie 2025/26",
        title="The league's top 5 scorers are all beating their xG",
        dek="Cumulative goals vs cumulative xG, matches with a shot recorded",
        palette=palette,
        top=0.965,
    )
    components.footer(
        fig,
        source="Opta/StatsPerform",
        note="xG: Waltzing Analytics xG Model · ranked by total xG across the season",
        palette=palette,
    )

    fig.savefig(out_path, dpi=200, facecolor=fig.get_facecolor())
    print("Saved:", out_path)


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else str(
        REPO_ROOT / "Eerste Divisie Events" / "top5_scorers_cumulative_xg.png")
    top_n = int(sys.argv[2]) if len(sys.argv) > 2 else 5

    model, meta = load_xg_model()
    qm = meta["qualifier_mapping"]

    files = sorted(glob.glob(f"{EERSTE_DIVISIE_DIR}/*.json"))
    print(f"Scoring shots across {len(files)} Eerste Divisie 2025/26 matches...")
    df = extract_shots(files, qm)
    df["xg"] = score_shots(df, model, meta)
    df["date"] = df["match_file"].apply(lambda p: p.split("/")[-1][:10])

    top = top_scorers(df, top_n)
    print(top[["player", "goals", "xg", "shots"]].to_string(index=False))

    make_plot(df, top, out)

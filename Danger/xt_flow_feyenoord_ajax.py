"""
xT flow map: cumulative expected threat (xT) over match time for
Feyenoord vs. Ajax (2026-03-22), from the pre-computed action-level xT
model (xT/xt_action_values.csv -- Karun Singh-style xT over passes,
carries and take-ons; see xT/xt_model_meta.json for methodology).

Built in the Meridian house style (housestyle package).

Usage: python3 Danger/xt_flow_feyenoord_ajax.py [out_dir]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import matplotlib.pyplot as plt

from housestyle import style, components

REPO_ROOT = Path(__file__).resolve().parent.parent
XT_CSV = REPO_ROOT / "xT" / "xt_action_values.csv"
MATCH_FILE = "2026-03-22_Feyenoord Rotterdam - AFC Ajax.json"

TEAM_ORDER = ["Feyenoord Rotterdam", "AFC Ajax"]
SHORT = {"Feyenoord Rotterdam": "Feyenoord", "AFC Ajax": "Ajax"}


def cumulative_series(actions, max_minute):
    actions = actions.sort_values("time_seconds")
    xs, ys = [0], [0]
    running = 0.0
    for _, r in actions.iterrows():
        minute = r["time_seconds"] / 60.0
        xs.append(minute)
        ys.append(running)
        running += max(r["xT_added"], 0.0)
        xs.append(minute)
        ys.append(running)
    xs.append(max_minute)
    ys.append(running)
    return xs, ys


def main():
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent

    df = pd.read_csv(XT_CSV, low_memory=False)
    match = df[df["match_file"] == MATCH_FILE].copy()
    match["time_seconds"] = pd.to_numeric(match["time_seconds"], errors="coerce")
    match["xT_added"] = pd.to_numeric(match["xT_added"], errors="coerce")
    match = match.dropna(subset=["time_seconds", "xT_added"])
    max_minute = max(96, match["time_seconds"].max() / 60.0 + 2)

    palette, cats = style.apply("light")
    ink_blue, teal = cats[0], cats[2]
    colors = {TEAM_ORDER[0]: ink_blue, TEAM_ORDER[1]: teal}

    fig = plt.figure(figsize=(11, 7.6))
    fig.patch.set_facecolor(palette["surface"])
    ax = fig.add_axes([0.08, 0.15, 0.87, 0.60])

    totals = {}
    for team in TEAM_ORDER:
        actions = match[match["team_name"] == team]
        xs, ys = cumulative_series(actions, max_minute)
        color = colors[team]
        total_pos = actions.loc[actions["xT_added"] > 0, "xT_added"].sum()
        totals[team] = total_pos
        ax.step(xs, ys, where="post", color=color, linewidth=2.4, zorder=3,
                 label=f"{SHORT[team]} ({total_pos:.2f} xT)")

    ax.axvline(45, color=palette["axis"], linewidth=0.8, linestyle=(0, (3, 3)), zorder=1)

    ax.set_xlim(0, max_minute)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Minute", fontsize=10.5, color=palette["ink_secondary"])
    ax.set_ylabel("Cumulative xT", fontsize=10.5, color=palette["ink_secondary"])
    ax.grid(True, axis="y", color=palette["grid"], linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", frameon=False, fontsize=10, labelcolor=palette["ink_secondary"])

    fey_xt, aja_xt = totals[TEAM_ORDER[0]], totals[TEAM_ORDER[1]]
    leader = SHORT[TEAM_ORDER[0]] if fey_xt > aja_xt else SHORT[TEAM_ORDER[1]]
    ratio = max(fey_xt, aja_xt) / max(min(fey_xt, aja_xt), 0.01)
    if ratio < 1.15:
        title = f"{leader} narrowly edged the open-play threat battle"
    else:
        title = f"{leader} generated {ratio:.1f}x more threat through open play"

    components.header(
        fig,
        kicker="Match report · xT flow",
        title=title,
        dek=f"Cumulative expected threat (xT) by minute, passes/carries/take-ons  ·  "
            f"Feyenoord {fey_xt:.2f} xT, Ajax {aja_xt:.2f} xT  ·  Eredivisie 2025/26, 22 Mar 2026",
        palette=palette,
    )
    components.footer(
        fig,
        source="Opta/StatsPerform, Eredivisie 2025/26",
        note="xT = Karun Singh-style expected threat model",
        palette=palette,
    )

    out_path = out_dir / "xt_flow_feyenoord_ajax.png"
    fig.savefig(out_path, dpi=200, facecolor=palette["surface"])
    plt.close(fig)
    print(f"Feyenoord {fey_xt:.3f} xT, Ajax {aja_xt:.3f} xT")
    print("Saved:", out_path)


if __name__ == "__main__":
    main()

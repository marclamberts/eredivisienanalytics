"""
Besiktas aerial duel ratings -- UEFA Europa League 2026/27, 2nd Qualifying
Round, both legs vs Midtjylland combined.

Rating = simple win rate (aerial duels won / aerial duels contested), NOT
a proprietary skill model like StatsBomb's HOPS -- this feed carries no
such rating, so the honest label here is "aerial duel win rate", sourced
straight from Opta's own Aerial (typeId 44) event outcomes.

Zone cuts: the pitch-box-specific splits (defensive box / attacking box)
have too few duels in this data to be meaningful (0 players clear a
min-3-duels bar in either box across 68 total aerial duels), so this
uses cuts with real sample size instead -- overall, defensive half,
middle third, attacking half.

Usage: python3 aerial_duel_ratings.py
Outputs PNGs into ./Aerial Duel Ratings.
"""
import os
import sys
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from housestyle import style, components  # noqa: E402

import match_data as md  # noqa: E402

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Aerial Duel Ratings")
os.makedirs(OUT_DIR, exist_ok=True)
FIGSIZE = (13.33, 7.5)
MIN_DUELS = 3


def save(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=170, facecolor=fig.get_facecolor())
    plt.close(fig)
    print("Saved:", path)


def rating_chart(aerials, zone_label, zone_slug, filt):
    by_player = defaultdict(lambda: [0, 0])
    for d in aerials:
        if not filt(d):
            continue
        by_player[d["player"]][1] += 1
        if d["success"]:
            by_player[d["player"]][0] += 1
    rows = [(p, w / n, n) for p, (w, n) in by_player.items() if n >= MIN_DUELS]
    rows.sort(key=lambda r: -r[1])
    rows = rows[::-1]  # ascending for barh (top rank at top of chart)

    palette, _ = style.apply("light")
    fig = plt.figure(figsize=FIGSIZE)
    ax = fig.add_axes([0.24, 0.16, 0.62, 0.56])

    ypos = np.arange(len(rows))
    vals = [r[1] for r in rows]
    colors = [palette["accent"] if i == len(rows) - 1 else palette["axis"] for i in range(len(rows))]
    bars = ax.barh(ypos, vals, color=colors, height=0.6)
    for y, (p, rate, n) in zip(ypos, rows):
        ax.text(rate + 0.015, y, f"{rate:.2f}  (n={n})", va="center", fontsize=10, color=palette["ink_secondary"])
    ax.set_yticks(ypos)
    ax.set_yticklabels([r[0] for r in rows], fontsize=11, color=palette["ink_primary"])
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("Aerial duel win rate")
    ax.grid(axis="x")
    ax.set_axisbelow(True)

    top_player = rows[-1][0] if rows else "n/a"
    where = "overall" if zone_slug == "overall" else f"in the {zone_label.lower()}"
    components.header(fig, kicker=f"Besiktas · {zone_label}",
                       title=f"{top_player} rates highest for Besiktas {where}",
                       dek=f"Aerial duel win rate (wins / duels contested)  ·  min {MIN_DUELS} duels  ·  "
                           f"{len(rows)} qualifying players",
                       palette=palette)
    components.footer(fig, source="Opta event data, both legs vs Midtjylland (UEL 2026/27, 2nd QR)",
                       note="Win rate, not a modelled skill rating -- no HOPS-equivalent in this feed",
                       palette=palette)
    save(fig, f"besiktas_aerial_ratings_{zone_slug}.png")


def main():
    leg1 = md.Leg(md.LEG1)
    leg2 = md.Leg(md.LEG2)
    all_duels = leg1.besiktas(leg1.duels) + leg2.besiktas(leg2.duels)
    aerials = [d for d in all_duels if d["action"] == "Aerial"]
    print(f"Total Besiktas aerial duels, both legs: {len(aerials)}")

    rating_chart(aerials, "Overall", "overall", lambda d: True)
    rating_chart(aerials, "Defensive Half", "defensive_half", lambda d: d["x"] < 52.5)
    rating_chart(aerials, "Middle Third", "middle_third", lambda d: 35.0 <= d["x"] < 70.0)
    rating_chart(aerials, "Attacking Half", "attacking_half", lambda d: d["x"] >= 52.5)

    print("Done.")


if __name__ == "__main__":
    main()

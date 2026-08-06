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

Set pieces: the Aerial event itself carries no corner/free-kick flag
(checked -- its only qualifiers are 56 zone, 233 related-event-id, and
an unlabelled 285/286 pair that doesn't track outcome or situation), so
"from a set piece" is inferred instead: an aerial duel counts if the most
recent pass by EITHER side in the preceding 6 seconds (same period) was
tagged corner or free-kick. That flags 14 of leg 1's 66 aerials and 8 of
leg 2's 70 as set-piece duels (both sides combined -- Besiktas' own share
is 7 and 4). Sample is small enough that this chart drops the qualifying
bar to 2 duels rather than the 3 used elsewhere, flagged in its own dek.

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


def rating_chart(aerials, zone_label, zone_slug, filt, min_duels=MIN_DUELS, dek_extra=None):
    by_player = defaultdict(lambda: [0, 0])
    for d in aerials:
        if not filt(d):
            continue
        by_player[d["player"]][1] += 1
        if d["success"]:
            by_player[d["player"]][0] += 1
    rows = [(p, w / n, n) for p, (w, n) in by_player.items() if n >= min_duels]
    rows.sort(key=lambda r: -r[1])
    rows = rows[::-1]  # ascending for barh (top rank at top of chart)

    palette, _ = style.apply("light")
    fig = plt.figure(figsize=FIGSIZE)
    ax = fig.add_axes([0.24, 0.16, 0.62, 0.56])

    ypos = np.arange(max(len(rows), 1))
    vals = [r[1] for r in rows]
    top_rate = rows[-1][1] if rows else None
    n_tied_top = sum(1 for _, r, _ in rows if abs(r - top_rate) < 1e-9) if rows else 0
    # a genuine tie for first isn't "the finding" -- accent is reserved for
    # a single standout, so a tie leaves every bar the same muted colour
    colors = [palette["accent"] if (n_tied_top == 1 and i == len(rows) - 1) else palette["axis"]
              for i in range(len(rows))]
    if rows:
        ax.barh(ypos, vals, color=colors, height=0.6)
        for y, (p, rate, n) in zip(ypos, rows):
            ax.text(rate + 0.015, y, f"{rate:.2f}  (n={n})", va="center", fontsize=10, color=palette["ink_secondary"])
        ax.set_yticks(ypos)
        ax.set_yticklabels([r[0] for r in rows], fontsize=11, color=palette["ink_primary"])
    else:
        ax.set_yticks([])
        ax.text(0.5, 0.5, "No player clears the qualifying threshold", transform=ax.transAxes,
                ha="center", va="center", fontsize=11, color=palette["ink_muted"])
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("Aerial duel win rate")
    ax.grid(axis="x")
    ax.set_axisbelow(True)

    where = "overall" if zone_slug == "overall" else f"in {zone_label.lower()}"
    if rows:
        top_rate = rows[-1][1]
        tied = [p for p, r, n in rows if abs(r - top_rate) < 1e-9]
        if len(tied) > 1:
            names = " and ".join([", ".join(tied[:-1]), tied[-1]]) if len(tied) > 2 else " and ".join(tied)
            title = f"{names} shared the best aerial record for Besiktas {where}, at {top_rate:.2f}"
        else:
            title = f"{tied[0]} rates highest for Besiktas {where}"
    else:
        title = f"No Besiktas player clears {min_duels} duels {where}"
    dek = f"Aerial duel win rate (wins / duels contested)  ·  min {min_duels} duels  ·  {len(rows)} qualifying players"
    if dek_extra:
        dek += f"  ·  {dek_extra}"
    components.header(fig, kicker=f"Besiktas · {zone_label}",
                       title=title,
                       dek=dek,
                       palette=palette)
    components.footer(fig, source="Opta event data, both legs vs Midtjylland (UEL 2026/27, 2nd QR)",
                       note="Win rate, not a modelled skill rating -- no HOPS-equivalent in this feed",
                       palette=palette)
    save(fig, f"besiktas_aerial_ratings_{zone_slug}.png")


def besiktas_set_piece_aerials(leg):
    """Besiktas aerial duels (typeId 44) whose most recent preceding pass
    (either side, same period, within 6 seconds) was tagged corner or
    free-kick -- see module docstring for why this proxy is used instead
    of a direct flag on the Aerial event."""
    events = sorted(leg.events, key=lambda e: (e["periodId"], md.event_time(e), e["eventId"]))
    rows = []
    for a in events:
        if a["typeId"] != 44 or a["contestantId"] != md.BESIKTAS_ID:
            continue
        t, period = md.event_time(a), a["periodId"]
        best = None
        for e in events:
            if e["periodId"] != period:
                continue
            et = md.event_time(e)
            if et > t or et < t - 6:
                continue
            if e["typeId"] != 1:
                continue
            best = e
        if best is None:
            continue
        q = md.qmap(best)
        if md.Q_CORNER in q or md.Q_FREE_KICK in q:
            rows.append({"player": a.get("playerName", "Unknown"), "success": a["outcome"] == 1})
    return rows


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

    set_piece_rows = besiktas_set_piece_aerials(leg1) + besiktas_set_piece_aerials(leg2)
    print(f"Besiktas set-piece aerial duels, both legs: {len(set_piece_rows)}")
    rating_chart(set_piece_rows, "Set Pieces", "set_pieces", lambda d: True, min_duels=2,
                 dek_extra="duel occurred within 6s of a corner or free-kick delivery")

    print("Done.")


if __name__ == "__main__":
    main()

"""
Attacking-third throw-ins that turn into a shot: pitch map of where the
throw-in was taken, for every throw-in that (a) starts in the attacking
third (x >= 66.67) and (b) the throwing team keeps the ball -- no
opposition touch -- for at least 2 more actions before a shot goes in.
League-wide, all three Eredivisie seasons (2023/24-2025/26) combined.

Throw-in: typeId 1 (pass), qualifier 107 (THROW_IN) present. Retention
is walked forward event-by-event from the throw-in: administrative
events (cards, subs, formation changes, period markers -- typeIds in
ADMIN_SKIP) are skipped over without counting or breaking the sequence;
any event belonging to the other team ends it (turnover, doesn't
qualify); a shot (typeId 13-16) by the throwing team ends it and
qualifies only if at least 2 of the throwing team's own actions
happened in between. This is the same "consecutive same-contestantId
= one possession" heuristic implicit in the rest of this repo's event
parsing, just made explicit and walked forward instead of only looking
one event ahead (as score_eredivisie_crosses.py's "cleared" flag does).

Usage: python3 throwin_retention_shot_map.py [out.png]
"""
import sys
import glob
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mplsoccer import VerticalPitch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from housestyle import style, components

DATA_ROOT = Path(__file__).resolve().parents[1] / "Events"
SEASONS = ["2023-2024", "2024-2025", "2025-2026"]

THROWIN_QID = 107
SHOT_TYPES = {13, 14, 15, 16}
GOAL_TYPE = 16
ATTACKING_THIRD_X = 66.67
MIN_RETAINED_ACTIONS = 2

# Cards, subs, formation changes, period/delay markers -- skipped over
# when walking a sequence forward, neither counted as an action nor
# treated as a turnover.
ADMIN_SKIP = {17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 30, 32, 34, 36, 40, 43}


def qids(e):
    return {q["qualifierId"] for q in e.get("qualifier", []) or []}


def load_season_events(season):
    files = sorted(glob.glob(str(DATA_ROOT / season / "*.json")))
    matches = []
    for fn in files:
        with open(fn) as f:
            data = json.load(f)
        events = [e for e in data.get("event", []) if e.get("periodId") in (1, 2)]
        if events:
            matches.append(events)
    return matches


def find_qualifying_throwins(events):
    """Every attacking-third throw-in, tagged with whether it qualifies
    (>=2 retained actions before a shot) and the shot event if so."""
    rows = []
    n_att_third = 0
    for idx, e in enumerate(events):
        if e.get("typeId") != 1 or THROWIN_QID not in qids(e):
            continue
        x, y = e.get("x"), e.get("y")
        if x is None or y is None or x < ATTACKING_THIRD_X:
            continue
        n_att_third += 1

        cid = e.get("contestantId")
        retained = 0
        qualifies = False
        shot_e = None
        for e2 in events[idx + 1:]:
            if e2.get("typeId") in ADMIN_SKIP:
                continue
            if e2.get("contestantId") != cid:
                break
            if e2.get("typeId") in SHOT_TYPES:
                shot_e = e2
                qualifies = retained >= MIN_RETAINED_ACTIONS
                break
            retained += 1

        if qualifies:
            rows.append({
                "x": x, "y": y, "shot_x": shot_e.get("x"), "shot_y": shot_e.get("y"),
                "goal": shot_e.get("typeId") == GOAL_TYPE, "retained": retained,
            })
    return rows, n_att_third


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else str(
        Path(__file__).resolve().parent / "16_throwin_retention_shot_map_eredivisie.png")

    all_rows = []
    total_att_third = 0
    for season in SEASONS:
        matches = load_season_events(season)
        for events in matches:
            rows, n_att = find_qualifying_throwins(events)
            all_rows.extend(rows)
            total_att_third += n_att

    n_goals = sum(1 for r in all_rows if r["goal"])
    print(f"Attacking-third throw-ins: {total_att_third}")
    print(f"Qualifying (>= {MIN_RETAINED_ACTIONS} retained actions, ending in a shot): {len(all_rows)} "
          f"({len(all_rows) / total_att_third * 100:.1f}% of attacking-third throw-ins)")
    print(f"Of which goals: {n_goals}")

    make_plot(all_rows, total_att_third, out_path)


def make_plot(rows, total_att_third, out_path):
    palette, cats = style.apply("light")
    goal_color = palette["accent"]
    shot_color = cats[0]

    fig = plt.figure(figsize=(9.4, 6.4))
    pitch = VerticalPitch(pitch_type="opta", pitch_color=palette["surface"],
                          line_color=palette["ink_muted"], linewidth=1.2, half=True)
    ax = fig.add_axes([0.07, 0.20, 0.86, 0.50])
    pitch.draw(ax=ax)
    ax.set_xlim(-2, 102)
    ax.set_ylim(63, 101)

    goals = [r for r in rows if r["goal"]]
    shots = [r for r in rows if not r["goal"]]

    # Individual throw-in -> shot lines get unreadable at this volume (400+
    # events pooled over 3 seasons); plot origins only, jittered slightly
    # off the touchline so overlapping throw-ins are still visible as a
    # cluster rather than one solid dot.
    import numpy as np
    rng = np.random.default_rng(0)

    def jitter_y(y):
        return min(100, max(0, y + rng.uniform(-1.6, 1.6)))

    for r in shots:
        pitch.scatter(r["x"], jitter_y(r["y"]), ax=ax, s=50, color=shot_color, alpha=0.55,
                     edgecolors="none", zorder=3)
    for r in goals:
        pitch.scatter(r["x"], jitter_y(r["y"]), ax=ax, s=110, color=goal_color, alpha=0.95,
                     edgecolors=palette["surface"], linewidth=0.9, zorder=4)

    handles = [
        Line2D([0], [0], color=goal_color, linewidth=0, marker="o", markersize=10,
              label=f"Goal ({len(goals)})"),
        Line2D([0], [0], color=shot_color, linewidth=0, marker="o", markersize=8,
              label=f"Shot, no goal ({len(shots)})"),
    ]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.06), ncol=2,
             frameon=False, fontsize=10, labelcolor=palette["ink_primary"])

    pct = len(rows) / total_att_third * 100 if total_att_third else 0
    components.header(
        fig,
        kicker="Eredivisie",
        title=f"1 in {round(100 / pct) if pct else 0} attacking-third throw-ins that stick leads to a shot",
        dek=f"Retaining possession for {MIN_RETAINED_ACTIONS}+ actions before a shot "
            f"· {len(rows)} of {total_att_third} attacking-third throw-ins ({pct:.0f}%)",
        palette=palette,
    )
    components.footer(
        fig,
        source="Opta event data, Eredivisie 2023/24-2025/26 (all matches, all 18 teams)",
        palette=palette,
    )
    fig.text(0.06, 0.020, "Dot = throw-in origin (jittered off the touchline to separate overlaps). "
             "Retention: no opposition touch between throw-in and shot, admin events (cards/subs) ignored.",
             fontsize=7.6, color=palette["ink_muted"], family="sans-serif",
             ha="left", va="top")

    fig.savefig(out_path, dpi=200, facecolor=fig.get_facecolor())
    print("Saved:", out_path)


if __name__ == "__main__":
    main()

"""
Ajax's open-play long-ball retention, per coach/manager, 2023-2024 -> 2025-2026.

"Retention" here means the restart_analysis.py brief's own definitions,
reused directly (not redefined) via import so the numbers stay consistent
with the league-wide restart analysis already in Aggregated/restart_analysis/:

  - DPR (direct possession retention): first contact won AND a teammate
    touches the ball again within 3s of that first contact.
  - IPR (indirect possession retention): first contact LOST, but Ajax still
    reaches "established possession" (3 consecutive Ajax touches, or 5
    continuous seconds of Ajax touches) via the second ball.
  - ERR (established retention rate) = DPR + IPR -- the headline number:
    how often a long ball ends with Ajax actually holding the ball, whether
    that came from winning the first header/control or scrapping back the
    second ball.
  - PER5/10/15: possession-established rate within 5/10/15s of the restart
    (a looser, time-boxed cut of the same idea).
  - PSR10/15/20: of the long balls where Ajax DID establish possession, how
    often that possession survives another 10/15/20s (retention that holds
    up, vs. immediately giving it back).

Only Ajax's own long balls count (not long balls played against Ajax).
Coach regimes and Ajax's contestantId are exactly as defined in
ajax_coach_style.py -- imported from there, not re-typed.

Usage: python3 ajax_long_ball_retention.py
"""
import glob
import json
import os
import sys

import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "Aggregated", "ajax_coach_style")

sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from housestyle import style, components  # noqa: E402
from ajax_coach_style import AJAX_CID, REGIMES, find_regime, match_date  # noqa: E402
from restart_analysis import (  # noqa: E402
    team_directions, find_restarts, walk_forward, walk_backward,
    analyse_restart, rate, mean,
)

AJAX, OPP = "Ajax", "Opponent"


def analyse_match(path):
    raw = json.load(open(path, encoding="utf-8"))
    events = [e for e in raw.get("event", []) if e.get("periodId") in (1, 2)]
    cids = {e.get("contestantId") for e in events if e.get("contestantId")}
    if AJAX_CID not in cids:
        return []
    opp_cid = next((c for c in cids if c != AJAX_CID), None)
    team_of_cid = {AJAX_CID: AJAX, opp_cid: OPP}
    opp_of_team = {AJAX: OPP, OPP: AJAX}

    directions = team_directions(events, team_of_cid)
    restarts = [r for r in find_restarts(events, team_of_cid, opp_of_team, directions)
                if r.kind == "long_ball" and r.team == AJAX]

    results = []
    for r in restarts:
        seq = walk_forward(events, r.idx, r.t0)
        pre_seq = walk_backward(events, r.idx, r.t0)
        results.append(analyse_restart(r, seq, pre_seq, team_of_cid, directions, {}, path, None))
    return results


def aggregate(results):
    n = len(results)
    if n == 0:
        return None
    contestable = [r for r in results if r.get("contestable")]
    nc = len(contestable)
    fc_won = [r for r in contestable if r.get("first_contact_won")]
    m = {
        "n_long_balls": n,
        "n_contestable": nc,
        "fcwr": rate(len(fc_won), nc),
        "dpr": rate(sum(1 for r in results if r.get("direct_retention")), n),
        "ipr": rate(sum(1 for r in results if r.get("indirect_retention")), n),
    }
    m["err"] = round((m["dpr"] or 0) + (m["ipr"] or 0), 4) if n else None
    for w in (5, 10, 15):
        m[f"per{w}"] = rate(sum(1 for r in results if r.get(f"possession_established_{w}")), n)
    for w in (10, 15, 20):
        established = [r for r in results if r.get("restart_established")]
        m[f"psr{w}"] = rate(sum(1 for r in established if r.get(f"survives_{w}")), len(established))
    return m


def main():
    match_results = []  # (season, coach, results)
    for season in ("2023-2024", "2024-2025", "2025-2026"):
        for path in sorted(glob.glob(os.path.join(ROOT, "Events", season, "*Ajax*.json"))):
            date_str = match_date(path)
            coach = find_regime(season, date_str)
            if coach is None:
                continue
            match_results.append((season, coach, analyse_match(path)))

    by_regime = {}
    order = []
    n_matches = {}
    for season, coach, results in match_results:
        key = (season, coach)
        if key not in by_regime:
            by_regime[key] = []
            order.append(key)
            n_matches[key] = 0
        by_regime[key].extend(results)
        n_matches[key] += 1

    rows = []
    for season, coach in order:
        m = aggregate(by_regime[(season, coach)])
        if m is None:
            continue
        rows.append({"season": season, "coach": coach, "matches": n_matches[(season, coach)], **m})

    csv_path = os.path.join(OUT_DIR, "ajax_long_ball_retention.csv")
    import csv as csv_mod
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv_mod.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {csv_path}")

    for r in rows:
        flag = "  (n=1 match, not reliable)" if r["matches"] == 1 else ""
        print(f"{r['season']} {r['coach']:<28} matches={r['matches']:>2}  long_balls={r['n_long_balls']:>4}  "
              f"FCWR={r['fcwr']:.1%}  DPR={r['dpr']:.1%}  IPR={r['ipr']:.1%}  ERR={r['err']:.1%}{flag}")

    build_chart(rows)


def build_chart(rows):
    palette, cats = style.apply("light")
    n = len(rows)
    colors = [cats[i % len(cats)] for i in range(n)]
    labels = [r["coach"].split(" (")[0] for r in rows]

    fig = plt.figure(figsize=(11.5, 8.2))
    ax = fig.add_axes([0.09, 0.16, 0.85, 0.58])

    x = range(n)
    w = 0.38
    dpr_vals = [r["dpr"] * 100 for r in rows]
    ipr_vals = [r["ipr"] * 100 for r in rows]
    err_vals = [r["err"] * 100 for r in rows]

    bars_dpr = ax.bar([i - w / 2 for i in x], dpr_vals, width=w, color=cats[0],
                       label="Direct retention (won first contact)")
    bars_ipr = ax.bar([i + w / 2 for i in x], ipr_vals, width=w, color=palette["accent"],
                       label="Indirect retention (won the second ball)")
    for i, r in enumerate(rows):
        ax.annotate(f"{err_vals[i]:.0f}%", xy=(i, max(dpr_vals[i], ipr_vals[i]) + 1.5),
                    ha="center", fontsize=9, fontweight="bold", color=palette["ink_primary"])
        if r["matches"] == 1:
            for bar in (bars_dpr.patches[i], bars_ipr.patches[i]):
                bar.set_hatch("///")
                bar.set_edgecolor(palette["ink_muted"])

    ax.set_xticks(list(x))
    tick_labels = [f"{lb} (n=1)" if r["matches"] == 1 else lb for lb, r in zip(labels, rows)]
    ax.set_xticklabels(tick_labels, fontsize=9.5, rotation=20, ha="right")
    ax.set_ylabel("Share of Ajax's own open-play long balls (%)")
    ax.legend(loc="upper right", frameon=False, fontsize=9, bbox_to_anchor=(1.0, 1.14))

    best = max(rows, key=lambda r: r["err"])
    components.header(
        fig, kicker="Long-Ball Retention",
        title=f"Ajax held onto the most long balls under {best['coach'].split(' (')[0]}",
        dek="Direct + indirect possession retention rate (ERR, labelled above each pair) "
            "on Ajax's own open-play long balls, by coach",
        palette=palette,
    )
    components.footer(fig, source="Opta/StatsPerform 2023-2026", palette=palette)

    out_path = os.path.join(OUT_DIR, "ajax_long_ball_retention.png")
    fig.savefig(out_path, facecolor=fig.get_facecolor(), dpi=150)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()

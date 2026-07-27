"""
xG For vs xG Against by match for a given Eerste Divisie 2025/26 team,
with a rolling-average trend line for each.

xG per shot comes from the Waltzing Analytics xG Model
(Model/model_xg.pkl + Model/model_meta.pkl) -- a calibrated XGBoost
classifier (CalibratedXGB = XGBClassifier + IsotonicRegression calibrator)
fit on distance/angle/y_sym plus qualifier-derived shot context (body
part, phase of play, assist type). No model is trained or inferred here;
this script only rebuilds the exact 18 features the pickled model expects
(per model_meta.pkl's qualifier_mapping) and calls it. Penalties use the
model's own fixed pen_xg from the metadata rather than the classifier
(all penalties share the same distance/angle/y_sym, so the classifier
was not trained to discriminate among them). Own goals are excluded from
shot xG entirely, in line with standard practice.

Usage: python3 ado_den_haag_xg_trend.py ["HFC ADO Den Haag"] [out.png]
Team name must match the filename spelling, e.g. "VVV Venlo", "SV Roda JC".
"""
import glob
import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from housestyle import style, components
from xg_model import load_xg_model, extract_shots, score_shots

REPO_ROOT = Path(__file__).resolve().parent.parent
EERSTE_DIVISIE_DIR = REPO_ROOT / "Eerste Divisie Events" / "Eerste Divisie 2025-2026"
ROLL_WINDOW = 5


def build_team_map(files):
    import collections
    team_cid_sets = collections.defaultdict(list)
    for fn in files:
        m = re.match(r"\d{4}-\d{2}-\d{2}_(.+) - (.+)\.json$", fn.split("/")[-1])
        if not m:
            continue
        home, away = m.group(1), m.group(2)
        with open(fn) as f:
            data = json.load(f)
        cids = set(e["contestantId"] for e in data["event"] if "contestantId" in e)
        team_cid_sets[home].append(cids)
        team_cid_sets[away].append(cids)
    team_to_cid = {}
    for team, sets in team_cid_sets.items():
        inter = set.intersection(*sets)
        if len(inter) == 1:
            team_to_cid[team] = next(iter(inter))
    return team_to_cid


def opponent_name(fname, team_name):
    m = re.match(r"\d{4}-\d{2}-\d{2}_(.+) - (.+)\.json$", fname)
    home, away = m.group(1), m.group(2)
    return away if team_name in home else home


def rolling_mean(values, window):
    out = []
    for i in range(len(values)):
        lo = max(0, i - window + 1)
        chunk = values[lo:i + 1]
        out.append(sum(chunk) / len(chunk))
    return out


def collect_match_rows(model, meta, qm, team_name):
    files = sorted(glob.glob(f"{EERSTE_DIVISIE_DIR}/*.json"))
    team_to_cid = build_team_map(files)
    if team_name not in team_to_cid:
        print(f"Team '{team_name}' not found. Options: {sorted(team_to_cid)}")
        sys.exit(1)
    cid = team_to_cid[team_name]

    rows = []
    for fn in files:
        base = fn.split("/")[-1]
        m = re.match(r"(\d{4}-\d{2}-\d{2})_(.+) - (.+)\.json$", base)
        date, home, away = m.group(1), m.group(2), m.group(3)
        if team_name not in (home, away):
            continue
        shots = extract_shots([fn], qm)
        if shots.empty:
            continue
        shots["xg"] = score_shots(shots, model, meta)
        xg_for = shots.loc[shots["contestant_id"] == cid, "xg"].sum()
        xg_against = shots.loc[shots["contestant_id"] != cid, "xg"].sum()
        rows.append({
            "date": date, "opponent": opponent_name(base, team_name),
            "venue": "H" if home == team_name else "A",
            "xg_for": xg_for, "xg_against": xg_against,
        })
    rows.sort(key=lambda r: r["date"])
    return rows


def make_plot(rows, out_path, team_name):
    n = len(rows)
    xs = list(range(1, n + 1))
    xgf = [r["xg_for"] for r in rows]
    xga = [r["xg_against"] for r in rows]
    roll_f = rolling_mean(xgf, ROLL_WINDOW)
    roll_a = rolling_mean(xga, ROLL_WINDOW)
    avg_f, avg_a = sum(xgf) / n, sum(xga) / n

    first_half_diff = sum(roll_f[:n // 2]) / (n // 2) - sum(roll_a[:n // 2]) / (n // 2)
    second_half_diff = sum(roll_f[n // 2:]) / (n - n // 2) - sum(roll_a[n // 2:]) / (n - n // 2)
    improving = second_half_diff > first_half_diff

    palette, cats = style.apply("dark")

    fig = plt.figure(figsize=(11, 6.8))
    ax = fig.add_axes([0.08, 0.16, 0.86, 0.58])

    ax.plot(xs, xgf, color=palette["axis"], alpha=0.5, linewidth=1, marker="o",
            markersize=3.5, zorder=2)
    ax.plot(xs, xga, color=palette["axis"], alpha=0.5, linewidth=1, marker="o",
            markersize=3.5, zorder=2, linestyle="--")
    ax.plot(xs, roll_f, linewidth=2.6, zorder=4, label=f"xG For ({ROLL_WINDOW}-match avg)")
    ax.plot(xs, roll_a, linewidth=2.6, zorder=3, label=f"xG Against ({ROLL_WINDOW}-match avg)")
    components.highlight_lines(ax, accent_index=2, palette=palette)
    ax.get_lines()[3].set_color(palette["ink_secondary"])
    ax.get_lines()[3].set_linewidth(1.8)

    ax.set_xlim(0.5, n + 0.5)
    ax.set_xticks(xs)
    ax.set_xticklabels([str(i) for i in xs], fontsize=7.5)
    ax.set_ylabel("xG per match", fontsize=10.5, color=palette["ink_secondary"])
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper left", frameon=False, fontsize=9.5, labelcolor=palette["ink_primary"])

    trend_word = "improved" if improving else "faded"
    components.header(
        fig,
        kicker=team_name,
        title=f"Attacking output {trend_word} relative to xG conceded over the season",
        dek=f"Eerste Divisie 2025/26 · rolling {ROLL_WINDOW}-match average · "
            f"season avg {avg_f:.2f} xG for / {avg_a:.2f} xG against per match",
        palette=palette,
    )
    components.footer(
        fig,
        source="Opta/StatsPerform",
        note="xG: Waltzing Analytics xG Model",
        palette=palette,
    )

    fig.savefig(out_path, dpi=200, facecolor=fig.get_facecolor())
    print("Saved:", out_path)
    print(f"n_matches={n} avg_xgf={avg_f:.2f} avg_xga={avg_a:.2f}")


if __name__ == "__main__":
    team_name = sys.argv[1] if len(sys.argv) > 1 else "HFC ADO Den Haag"
    default_out = REPO_ROOT / "Eerste Divisie Events" / f"{team_name.lower().replace(' ', '_')}_xg_trend.png"
    out = sys.argv[2] if len(sys.argv) > 2 else str(default_out)

    model, meta = load_xg_model()
    qm = meta["qualifier_mapping"]
    rows = collect_match_rows(model, meta, qm, team_name)
    if not rows:
        print("No matches found for", team_name)
        sys.exit(1)
    make_plot(rows, out, team_name)

"""
Individual per-metric pitch charts for PSV Eindhoven, 2025/26 season.
Eleven standalone mplsoccer VerticalPitch visuals instead of a couple of
combined ones: three shot maps (xG, danger, body-part types) plus eight
pass-type maps (key passes, shot assists, second assists, progressive
passes, through passes, half-space passes, zone 14 passes, cutbacks).
Season-aggregated across all 34 PSV Eredivisie fixtures. Grammar (pitch
type, palette, header/footer) matches the rest of the report's
mplsoccer-based team-viz scripts.

Usage: python3 individual_metric_pitches.py [out_dir]
"""
import sys
import re
import glob
import json
import collections
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from mplsoccer import VerticalPitch

DATA_DIR = "/Users/marclamberts/Event data/Eredivisie"
DANGER_DIR = "/Users/marclamberts/Event data/Eredivisie/Danger"
LOGO_PATH = "/Users/marclamberts/Downloads/Waltzing Analytics Logo Type.png"
TEAM_NAME = "PSV Eindhoven"

C_NAVY = "#2f8fd1"
C_INDIGO = "#7b7fd6"
C_PURPLE = "#c179d1"
C_PINK = "#f06fa3"
C_CORAL = "#ff8a75"
C_AMBER = "#ffc247"
C_GREEN = "#4ade80"
BG = "#0d1117"
PITCH_LINE = "#2c3a4d"
TEXT_SUB = "#9aa4b2"
TEXT_FOOT = "#6b7684"
ZONE_SHADE = "#1e3a5f"

SHOT_TYPES = {13, 14, 15, 16}
CROSS_QID = 2
HEAD_QID = 15
RIGHT_QID = 20
LEFT_QID = 72
PENALTY_QID = 9

PREFIX_RE = re.compile(r"^(CSD|CD|CS|SD)\s+")


def clean_name(name):
    return PREFIX_RE.sub("", name)


def add_logo(fig, width=0.13, margin=0.016):
    import matplotlib.image as mpimg
    try:
        img = mpimg.imread(LOGO_PATH)
    except FileNotFoundError:
        return
    fig_w, fig_h = fig.get_size_inches()
    img_h, img_w = img.shape[0], img.shape[1]
    width_in = width * fig_w
    height_in = width_in * (img_h / img_w)
    height = height_in / fig_h
    left = 1 - margin - width
    bottom = 1 - margin - height
    logo_ax = fig.add_axes([left, bottom, width, height], zorder=10)
    logo_ax.patch.set_alpha(0)
    logo_ax.set_xlim(0, img_w)
    logo_ax.set_ylim(img_h, 0)
    logo_ax.imshow(img)
    logo_ax.axis("off")


def build_team_map(files):
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


def qmap(e):
    return {q["qualifierId"]: q.get("value") for q in e.get("qualifier", []) or []}


def end_xy(e, q):
    x, y = e.get("x"), e.get("y")
    try:
        ex = float(q.get(140, x))
    except (TypeError, ValueError):
        ex = x
    try:
        ey = float(q.get(141, y))
    except (TypeError, ValueError):
        ey = y
    return ex, ey


def load_danger():
    paths = sorted(glob.glob(f"{DANGER_DIR}/[0-9][0-9][0-9][0-9]-*_danger_models.csv"))
    frames = [pd.read_csv(p) for p in paths]
    df = pd.concat(frames, ignore_index=True)
    df["contestant_id"] = df["contestant_id"].astype(str)
    df["event_id"] = df["event_id"].astype(str)
    return df


def load_events(files, cid):
    out = {k: [] for k in ["progressive", "through", "halfspace", "zone14", "cutback",
                            "key_passes", "shot_assists", "second_assists"]}
    shots = []

    for fn in files:
        base = fn.split("/")[-1]
        with open(fn) as f:
            data = json.load(f)
        events = [e for e in data["event"] if e.get("periodId") in (1, 2)]
        events.sort(key=lambda e: (e["periodId"], e.get("timeMin", 0), e.get("timeSec", 0), e.get("eventId", 0)))

        for idx, e in enumerate(events):
            tid = e.get("typeId")
            ecid = e.get("contestantId")
            x, y = e.get("x"), e.get("y")

            if ecid == cid and tid == 1 and x is not None and y is not None:
                q = qmap(e)
                success = e.get("outcome") == 1
                ex, ey = end_xy(e, q)
                if ex is None or ey is None:
                    continue
                dx = ex - x
                rec = {"x0": x, "y0": y, "x1": ex, "y1": ey, "player": e.get("playerName", "?")}
                if success:
                    is_wide_start = y <= 21 or y >= 79
                    is_box_end = ex >= 83 and 21 <= ey <= 79
                    is_cross = CROSS_QID in q or (x >= 55 and is_wide_start and is_box_end and abs(ey - y) >= 12)
                    if dx >= 10 or (x < 50 and ex >= 66.7) or (x < 66.7 and ex >= 83):
                        out["progressive"].append(rec)
                    if ex >= 66.7 and dx >= 12 and 33 <= ey <= 67:
                        out["through"].append(rec)
                    if (21 <= y < 33 or 67 < y <= 79) or (21 <= ey < 33 or 67 < ey <= 79):
                        out["halfspace"].append(rec)
                    if 66.7 <= ex < 83 and 33 <= ey <= 67:
                        out["zone14"].append(rec)
                    if is_cross and x >= 83 and is_wide_start and ex >= 83 and 33 <= ey <= 67:
                        out["cutback"].append(rec)

            if ecid == cid and tid in SHOT_TYPES and x is not None:
                q_shot = qmap(e)
                is_goal = tid == 16
                shots.append({
                    "event_id": str(e.get("id")), "match_file": base,
                    "x": x, "y": y, "is_goal": is_goal,
                    "is_on_target": tid in (15, 16),
                    "head": HEAD_QID in q_shot, "right": RIGHT_QID in q_shot, "left": LEFT_QID in q_shot,
                    "penalty": PENALTY_QID in q_shot,
                })

                if idx > 0:
                    prev = events[idx - 1]
                    if (prev.get("contestantId") == cid and prev.get("typeId") == 1
                            and prev.get("outcome") == 1 and prev.get("playerId") != e.get("playerId")
                            and prev.get("x") is not None):
                        pq = qmap(prev)
                        pex, pey = end_xy(prev, pq)
                        kp_rec = {"x0": prev["x"], "y0": prev["y"], "x1": pex, "y1": pey,
                                  "player": prev.get("playerName", "?"), "is_goal": is_goal,
                                  "is_on_target": tid in (15, 16)}
                        out["key_passes"].append(kp_rec)
                        if tid in (15, 16):
                            out["shot_assists"].append(kp_rec)
                        if is_goal and idx > 1:
                            prev2 = events[idx - 2]
                            if (prev2.get("contestantId") == cid and prev2.get("typeId") == 1
                                    and prev2.get("outcome") == 1
                                    and prev2.get("playerId") not in (prev.get("playerId"), e.get("playerId"))
                                    and prev2.get("x") is not None):
                                p2q = qmap(prev2)
                                p2ex, p2ey = end_xy(prev2, p2q)
                                out["second_assists"].append({
                                    "x0": prev2["x"], "y0": prev2["y"], "x1": p2ex, "y1": p2ey,
                                    "player": prev2.get("playerName", "?"),
                                })

    shots_df = pd.DataFrame(shots)
    danger = load_danger()
    shots_df = shots_df.merge(danger[["match_file", "event_id", "xg", "danger_score"]],
                               on=["match_file", "event_id"], how="left")
    shots_df["xg"] = shots_df["xg"].fillna(0.05)
    shots_df["danger_score"] = shots_df["danger_score"].fillna(0.05)
    return out, shots_df


def _new_pitch_fig(figsize=(11, 13.7), half=False):
    fig = plt.figure(figsize=figsize)
    fig.patch.set_facecolor(BG)
    pitch = VerticalPitch(pitch_type="opta", pitch_color=BG, line_color=PITCH_LINE,
                          linewidth=1.1, half=half)
    return fig, pitch


def _title_block(fig, title, subtitle):
    fig.text(0.5, 0.965, clean_name(TEAM_NAME), fontsize=26, fontweight="bold", ha="center", color="white")
    fig.text(0.5, 0.933, title, fontsize=12, ha="center", color=TEXT_SUB)


def _footer(fig, caption, methodology):
    import textwrap
    fig_w_in = fig.get_size_inches()[0]
    wrap_width = max(40, int(fig_w_in * 8.6))
    lines = textwrap.wrap(caption, width=wrap_width, max_lines=2, placeholder="…")
    positions = [0.062, 0.043]
    for line, y in zip(lines, positions):
        fig.text(0.5, y, line, fontsize=11.5, ha="center", color="#c7ccd4")
    if methodology:
        meth_lines = textwrap.wrap(methodology, width=wrap_width + 10, max_lines=1, placeholder="…")
        fig.text(0.5, 0.021, meth_lines[0] if meth_lines else methodology,
                 fontsize=8.2, ha="center", color=TEXT_FOOT)
    fig.text(0.98, 0.006, "Marc Lamberts", fontsize=9.5, ha="right", color=TEXT_FOOT, style="italic")
    fig.text(0.02, 0.006, "Data via Opta | Eredivisie 2025/26 event data · season aggregate, all 34 PSV fixtures",
             fontsize=7.3, color=TEXT_FOOT)


def _save(fig, out_path):
    add_logo(fig)
    fig.savefig(out_path, dpi=200, facecolor=BG)
    plt.close(fig)
    print("Saved:", out_path)


# ---------------------------------------------------------------------------
# 1) Shot map — xG
# ---------------------------------------------------------------------------
def make_shotmap_xg(shots, out_path):
    fig, pitch = _new_pitch_fig(half=True)
    ax = fig.add_axes([0.04, 0.06, 0.92, 0.82])
    pitch.draw(ax=ax)
    _title_block(fig, "Shot Map · Expected Goals (xG) · Eredivisie 2025/26 · Season",
                 "Size = xG · gold = goal · orange = on target · grey = off target/blocked")

    goals = shots[shots["is_goal"]]
    ontarget = shots[(~shots["is_goal"]) & (shots["is_on_target"])]
    off = shots[~shots["is_on_target"]]
    for sub, color, alpha, edge in [(off, "#8a93a1", 0.55, "white"), (ontarget, C_CORAL, 0.8, "white"), (goals, C_AMBER, 0.95, C_AMBER)]:
        if len(sub) == 0:
            continue
        sizes = 1500 * sub["xg"].clip(0.02, 0.85) + 40
        pitch.scatter(sub["x"], sub["y"], ax=ax, s=sizes, color=color, alpha=alpha,
                     edgecolors=edge, linewidth=1.1, zorder=3 if color == C_AMBER else 2)

    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=C_AMBER, markeredgecolor=C_AMBER, markersize=11, label=f"Goal ({len(goals)})"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=C_CORAL, markeredgecolor="white", markersize=11, label=f"On target ({len(ontarget)})"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#8a93a1", markeredgecolor="white", markersize=9, label=f"Off target ({len(off)})"),
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.885), ncol=3,
              frameon=False, fontsize=9.5, labelcolor="#c7ccd4")

    xg_total = shots["xg"].sum()
    _footer(fig, f"{len(shots)} shots · {int(goals.shape[0])} goals · {xg_total:.1f} total xG · {xg_total/max(len(shots),1):.2f} xG/shot",
            "xg = mean of 5 calibrated shot models (Danger/*_danger_models.csv)")
    _save(fig, out_path)


# ---------------------------------------------------------------------------
# 2) Shot map — Danger
# ---------------------------------------------------------------------------
def make_shotmap_danger(shots, out_path):
    fig, pitch = _new_pitch_fig(half=True)
    ax = fig.add_axes([0.04, 0.06, 0.92, 0.82])
    pitch.draw(ax=ax)
    _title_block(fig, "Shot Map · Danger Score · Eredivisie 2025/26 · Season",
                 "Size & color = danger_score (xG, PSxG, situation danger, P(on target), xGOT blend)")

    order = shots.sort_values("danger_score")
    sizes = 1500 * (order["danger_score"] / max(order["danger_score"].max(), 0.01)).clip(0.03, 1.0) + 35
    sc = pitch.scatter(order["x"], order["y"], ax=ax, s=sizes, c=order["danger_score"],
                       cmap="inferno", alpha=0.85, edgecolors="white", linewidth=0.6, zorder=2)
    cax = fig.add_axes([0.90, 0.20, 0.016, 0.55])
    cb = fig.colorbar(sc, cax=cax)
    cb.set_label("Danger score", color=TEXT_SUB, fontsize=8.5)
    cb.ax.yaxis.set_tick_params(color=TEXT_SUB, labelcolor=TEXT_SUB, labelsize=7.5)

    top = order.tail(1)
    _footer(fig, f"{len(shots)} shots · mean danger {shots['danger_score'].mean():.2f} · "
                 f"peak danger {shots['danger_score'].max():.2f}",
            "danger_score blends five calibrated shot-quality models into one 0-1 index")
    _save(fig, out_path)


# ---------------------------------------------------------------------------
# 3) Shot map — Types (body part, 3-panel)
# ---------------------------------------------------------------------------
def make_shotmap_types(shots, out_path):
    fig = plt.figure(figsize=(17, 11))
    fig.patch.set_facecolor(BG)
    fig.text(0.5, 0.955, clean_name(TEAM_NAME), fontsize=26, fontweight="bold", ha="center", color="white")
    fig.text(0.5, 0.918, "Shot Map · By Body Part · Eredivisie 2025/26 · Season",
             fontsize=12, ha="center", color=TEXT_SUB)

    panels = [
        ("Right Foot", shots[shots["right"]], C_NAVY),
        ("Left Foot", shots[shots["left"]], C_INDIGO),
        ("Head", shots[shots["head"]], C_PINK),
    ]
    axes = [fig.add_axes([0.03 + i * 0.325, 0.06, 0.30, 0.80]) for i in range(3)]
    for ax, (label, sub, color) in zip(axes, panels):
        pitch = VerticalPitch(pitch_type="opta", pitch_color=BG, line_color=PITCH_LINE, linewidth=1.0, half=True)
        pitch.draw(ax=ax)
        goals = sub[sub["is_goal"]]
        others = sub[~sub["is_goal"]]
        sizes_o = 1100 * others["xg"].clip(0.02, 0.85) + 30
        sizes_g = 1100 * goals["xg"].clip(0.02, 0.85) + 30
        pitch.scatter(others["x"], others["y"], ax=ax, s=sizes_o, color=color, alpha=0.55,
                     edgecolors="white", linewidth=0.8, zorder=2)
        pitch.scatter(goals["x"], goals["y"], ax=ax, s=sizes_g, color=C_AMBER, alpha=0.95,
                     edgecolors=C_AMBER, linewidth=1.2, zorder=3)
        n = len(sub)
        ng = len(goals)
        xg = sub["xg"].sum()
        ax.set_title(f"{label}   ·   {n} shots, {ng} goals, {xg:.1f} xG", fontsize=12, fontweight="bold",
                    color=color, pad=10)

    _footer(fig, f"{len(shots)} total shots · body part from Opta shot qualifiers (15=head, 20=right foot, 72=left foot)",
            "Gold = goal, in each panel's own colour = other shot outcomes")
    _save(fig, out_path)


# ---------------------------------------------------------------------------
# Generic pass-arrow chart builder
# ---------------------------------------------------------------------------
DENSITY_CMAP = None  # set lazily below to avoid import order issues


def _density_cmap(color):
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("density", [BG, color])


def make_pass_arrow_chart(records, out_path, title, subtitle, color, caption_label,
                          methodology, half=False, show_zone=None, top_players=True,
                          mode="arrows", max_arrows=350):
    fig, pitch = _new_pitch_fig(half=half)
    ax = fig.add_axes([0.04, 0.08, 0.92, 0.78])
    pitch.draw(ax=ax)
    _title_block(fig, title, subtitle)

    if show_zone == "halfspace":
        for y0 in (21, 67):
            pitch.polygon([[[0, y0], [100, y0], [100, y0 + 12], [0, y0 + 12]]], ax=ax,
                         facecolor=ZONE_SHADE, edgecolor="none", alpha=0.35, zorder=0.5)
    elif show_zone == "zone14":
        pitch.polygon([[[66.7, 33], [83, 33], [83, 67], [66.7, 67]]], ax=ax,
                     facecolor=ZONE_SHADE, edgecolor=C_PINK, linewidth=1.0, alpha=0.4, zorder=0.5)

    player_counts = collections.Counter()
    for r in records:
        if "player" in r:
            player_counts[r["player"]] += 1

    if mode == "density":
        xs = np.array([r["x1"] for r in records])
        ys = np.array([r["y1"] for r in records])
        stats = pitch.bin_statistic(xs, ys, statistic="count", bins=(24, 20))
        pitch.heatmap(stats, ax=ax, cmap=_density_cmap(color), edgecolors=BG, linewidth=0.12,
                      zorder=1, alpha=0.92)
        pitch.draw(ax=ax)
        if show_zone == "zone14":
            pitch.annotate("ZONE 14", xy=(74.8, 50), ax=ax, ha="center", va="center", fontsize=10,
                          fontweight="bold", color="white", alpha=0.9, zorder=3)
        legend_elems = [Line2D([0], [0], marker="s", color="none", markerfacecolor=color,
                               markersize=13, label=f"{caption_label} density (end location)")]
    else:
        sample = records
        if len(records) > max_arrows:
            rng = np.random.default_rng(42)
            idx = rng.choice(len(records), size=max_arrows, replace=False)
            sample = [records[i] for i in idx]
        for r in sample:
            pitch.lines(r["x0"], r["y0"], r["x1"], r["y1"], ax=ax, color=color, lw=1.1, alpha=0.55,
                       zorder=2, comet=False)
            pitch.scatter(r["x0"], r["y0"], ax=ax, s=12, color=color, alpha=0.8, zorder=3, linewidths=0)
            pitch.scatter(r["x1"], r["y1"], ax=ax, s=22, facecolor="none", edgecolors=color,
                         linewidths=1.0, alpha=0.8, zorder=3)
        if show_zone == "zone14":
            pitch.annotate("ZONE 14", xy=(74.8, 50), ax=ax, ha="center", va="center", fontsize=10,
                          fontweight="bold", color=C_PINK, alpha=0.85, zorder=1)
        note = "" if len(records) <= max_arrows else f" (random sample of {max_arrows} shown)"
        legend_elems = [Line2D([0], [0], color=color, linewidth=2, label=caption_label + note)]

    fig.legend(handles=legend_elems, loc="lower center", bbox_to_anchor=(0.5, 0.882),
              ncol=1, frameon=False, fontsize=9.5, labelcolor="#c7ccd4")

    label_lower = caption_label.lower()
    plural = label_lower + "es" if label_lower.endswith("pass") else label_lower + "s"
    caption = f"{len(records)} {plural} this season"
    if top_players and player_counts:
        top = player_counts.most_common(4)
        top_str = "  ·  ".join(f"{name} ({n})" for name, n in top)
        caption = caption + "  ·  Top: " + top_str
    _footer(fig, caption, methodology)
    _save(fig, out_path)


def main():
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/tmp/psv_metric_pitches")
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(glob.glob(f"{DATA_DIR}/*.json"))
    team_to_cid = build_team_map(files)
    match_key = next((full for full in team_to_cid if TEAM_NAME.lower() in full.lower()), None)
    if match_key is None:
        raise SystemExit(f"Team '{TEAM_NAME}' not found")
    cid = team_to_cid[match_key]

    print("Parsing all PSV matches (single pass)...")
    out, shots = load_events(files, cid)
    print(f"shots={len(shots)} progressive={len(out['progressive'])} through={len(out['through'])} "
          f"halfspace={len(out['halfspace'])} zone14={len(out['zone14'])} cutback={len(out['cutback'])} "
          f"key_passes={len(out['key_passes'])} shot_assists={len(out['shot_assists'])} "
          f"second_assists={len(out['second_assists'])}")

    make_shotmap_xg(shots, out_dir / "01_shotmap_xg_psv.png")
    make_shotmap_danger(shots, out_dir / "02_shotmap_danger_psv.png")
    make_shotmap_types(shots, out_dir / "03_shotmap_types_psv.png")

    make_pass_arrow_chart(out["key_passes"], out_dir / "04_key_passes_psv.png",
                          "Key Passes · Any Shot Created · Eredivisie 2025/26 · Season",
                          "Every completed pass whose very next event is a shot attempt by PSV",
                          C_NAVY, "Key pass",
                          "Key pass = completed pass immediately followed by a PSV shot attempt (any outcome)")

    make_pass_arrow_chart(out["shot_assists"], out_dir / "05_shot_assists_psv.png",
                          "Shot Assists · On-Target Chances Created · Eredivisie 2025/26 · Season",
                          "Passes immediately preceding a shot that was on target or a goal",
                          C_AMBER, "Shot assist",
                          "Shot assist = key pass whose resulting shot was saved, on target, or a goal")

    make_pass_arrow_chart(out["second_assists"], out_dir / "06_second_assists_psv.png",
                          "Second Assists · The Pass Before The Assist · Eredivisie 2025/26 · Season",
                          "Completed pass immediately preceding the assist pass, for PSV goals only",
                          C_PURPLE, "Second assist",
                          "Second assist = completed pass immediately before the assist in a goal's build-up chain")

    make_pass_arrow_chart(out["progressive"], out_dir / "07_progressive_passes_psv.png",
                          "Progressive Passes · Eredivisie 2025/26 · Season",
                          "Completed passes gaining 10+ pitch units, or into the final third/box from deeper",
                          C_GREEN, "Progressive pass",
                          "Progressive = dx ≥ 10, or starts <66.7m and ends in the box, or starts <50m and ends in the final third",
                          mode="density")

    make_pass_arrow_chart(out["through"], out_dir / "08_through_passes_psv.png",
                          "Through Passes · Eredivisie 2025/26 · Season",
                          "Completed passes splitting the defense into the central attacking channel",
                          C_PINK, "Through pass",
                          "Through pass = completed pass ending ≥66.7m with ≥12m forward progress into the central lane (y 33-67)")

    make_pass_arrow_chart(out["halfspace"], out_dir / "09_halfspace_passes_psv.png",
                          "Half-Space Passes · Eredivisie 2025/26 · Season",
                          "Completed passes starting or ending in the half-space channels",
                          C_CORAL, "Half-space pass",
                          "Half-space = pitch bands y 21-33 and y 67-79 (between the touchline lane and the central channel)",
                          show_zone="halfspace", mode="density")

    make_pass_arrow_chart(out["zone14"], out_dir / "10_zone14_passes_psv.png",
                          "Zone 14 Passes · Eredivisie 2025/26 · Season",
                          "Completed passes ending in the central zone just outside the box",
                          C_PINK, "Zone 14 pass",
                          "Zone 14 = x 66.7-83, y 33-67 (central channel between the final third line and the box)",
                          show_zone="zone14", max_arrows=250)

    make_pass_arrow_chart(out["cutback"], out_dir / "11_cutbacks_psv.png",
                          "Cutbacks · Eredivisie 2025/26 · Season",
                          "Byline crosses pulled back into the central channel at the edge of the six-yard box",
                          C_AMBER, "Cutback",
                          "Cutback = cross from the byline (x≥83, wide start) pulled back to x≥83, central lane (y 33-67)")

    print("All individual metric pitch charts done.")


if __name__ == "__main__":
    main()

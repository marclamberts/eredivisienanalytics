"""
Category rules for splitting the wide player/team CSVs into workbook tabs.
Both build_season_aggregate.py (for a sensible CSV column order) and
build_workbook.py (for the actual xlsx tabs) import `categorize()` from here
so the two stay in sync automatically -- neither hardcodes a column list that
could drift from what the build script actually produces.

Order matters: `categorize()` assigns every column to the FIRST rule it
matches, so more specific rules (e.g. home/away splits, the model-based "new
metrics") are listed before the broad ones (e.g. general passing) they would
otherwise be swallowed by. Add new stat name patterns here, not in the two
build scripts.
"""

IDENTITY_COLS = ["player_id", "player_name", "team_name"]

# (tab name, predicate on column name) -- first match wins
def _has_split_token(c):
    return bool({"1h", "2h", "home", "away"} & set(c.split("_")))


PLAYER_TAB_RULES = [
    ("Playing Time", lambda c: c in ("matches", "minutes", "minutes_bucket", "reliable_sample")),

    ("Splits - Half & Home-Away", _has_split_token),

    ("Goalkeeping", lambda c: c.startswith("gk_") or c in ("save_pct", "clean_sheets")),

    ("New Metrics - Progression Value", lambda c: (
        c.startswith(("xT_", "box_entry_", "psv_", "hotzone_", "xp_cross"))
        or "xT_" in c
        or c in ("actions", "xt_passes", "xt_carries", "xt_take_ons")
    )),

    ("New Metrics - Defensive & Overall Value", lambda c: (
        c.startswith(("disruption_", "danger_score", "padj_"))
        or c in ("goal_difference_added", "goal_difference_added_per90",
                 "action_gda_actual", "goals_against_on")
    )),

    ("Set Pieces", lambda c: c.startswith(("corners_", "corner_", "free_kick", "penalt", "throw_in"))),

    ("Discipline", lambda c: c.startswith(("fouls_", "yellow_card", "red_card")) or c.startswith("offside")),

    ("Creativity", lambda c: (
        c.startswith(("key_passes", "assists", "xa_", "xa_per90", "shot_creating_actions",
                       "goal_creating_actions", "passes_received", "progressive_passes_received"))
        or c in ("xa", "xa_per90")
    )),

    ("Progression (raw)", lambda c: (
        c.startswith(("progressive_passes", "progressive_carries", "carries_", "carry_",
                       "progressive_actions", "final_third_entries", "total_box_entries",
                       "passes_into_final_third"))
    )),

    ("Shooting", lambda c: (
        c.startswith(("shots", "shot_", "goals", "xg", "np_xg", "np_goals", "psxg",
                       "avg_shot_distance", "goal_involvement"))
    )),

    ("Crossing", lambda c: c.startswith("crosses_") or c.startswith("cross_completion")),

    ("Passing", lambda c: (
        c.startswith(("passes_", "pass_", "forward_pass", "backward_pass", "lateral_pass",
                       "long_ball", "through_ball", "short_pass", "medium_pass", "long_pass"))
        or "_pass_" in c or c.endswith("_pass_completion_pct")
    )),

    ("Duels & Defensive Actions", lambda c: (
        c.startswith(("take_ons_", "take_on_", "tackle", "interception", "clearance",
                       "aerial_duel", "ball_recover", "dispossessed", "error",
                       "total_duel", "defensive_actions_", "touches", "possession_losses"))
    )),
]

TEAM_TAB_RULES = [
    ("League Table", lambda c: c in (
        "team_name", "matches", "wins", "draws", "losses", "goals_for", "goals_against",
        "goal_diff", "points", "possession_pct",
    )),
    ("New Metrics", lambda c: (
        c.startswith(("xT_", "box_entry_", "xp_cross_", "disruption_", "danger_score", "xg"))
    )),
    ("Style & Formation", lambda c: c.startswith("style_") or c in ("dominant_back_line", "top_formation")),
    ("Wyscout-style Totals", lambda c: True),  # fallback: everything else (raw counting stats)
]


def categorize(columns, rules, identity_cols=IDENTITY_COLS):
    """Split `columns` into (tab_name, [cols]) groups per `rules`, preserving
    the original column order within each tab. Identity columns are excluded
    (callers add them to every tab themselves). Anything matching no rule
    lands in a final "Other" tab so nothing is ever silently dropped."""
    tabs = []
    tab_index = {}
    for col in columns:
        if col in identity_cols:
            continue
        tab_name = None
        for name, pred in rules:
            if pred(col):
                tab_name = name
                break
        if tab_name is None:
            tab_name = "Other"
        if tab_name not in tab_index:
            tab_index[tab_name] = len(tabs)
            tabs.append((tab_name, []))
        tabs[tab_index[tab_name]][1].append(col)
    return tabs

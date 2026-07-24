"""
Turn a snake_case metric column (e.g. "padj_tackles_attempted_per90_z") into a
short, human-readable label for the xlsx (e.g. "PAdj Tackles /90 (Z)").

The underlying CSV columns keep their exact snake_case names -- this module
only affects how they're *labelled* in the workbook, so nothing that reads
the CSVs (column_layout.py's rules, other scripts) needs to change.
"""
import re

# Whole-column overrides for names a token-by-token pass would mangle.
EXACT_OVERRIDES = {
    "player_id": "Player ID",
    "player_name": "Player",
    "team_name": "Team",
    "psv_attempts": "Pass-Shot Value Attempts",
    "psv_total": "Pass-Shot Value",
    "psv_actual_shot_xg": "Pass-Shot Value: Actual Shot xG",
    "psv_total_per90": "Pass-Shot Value /90",
    "xp_crosses": "xCmp Crosses",
    "xp_cross_completed": "xCmp Crosses Cmp",
    "xp_cross_expected_completed": "xCmp Crosses Expected Cmp",
    "xp_cross_completion_rate": "xCmp Crosses Cmp %",
    "xp_cross_expected_completion_rate": "xCmp Crosses Expected Cmp %",
    "xp_cross_added_value": "xCmp Crosses Added Value",
    "xp_cross_added_value_per90": "xCmp Crosses Added Value /90",
    "long_passes_len": "Long Passes (by length)",
    "long_passes_len_completed": "Long Passes (by length) Cmp",
    "long_pass_len_completion_pct": "Long Pass (by length) Cmp %",
    "long_passes_len_per90": "Long Passes (by length) /90",
    "long_passes_len_completed_per90": "Long Passes (by length) Cmp /90",
    "goal_difference_added": "GDA",
    "goal_difference_added_per90": "GDA /90",
    "action_gda_actual": "On-Ball GDA Value",
    "goals_against_on": "Goals Conceded (On Pitch)",
}

# Token -> display form. Looked up case-insensitively; anything not listed
# is just Title-cased.
TOKEN_MAP = {
    "xt": "xT", "xg": "xG", "xa": "xA", "psxg": "PSxG", "np": "NP", "xp": "xCmp",
    "gk": "GK", "padj": "PAdj", "sca": "SCA", "gca": "GCA", "id": "ID",
    "def": "Def", "mid": "Mid", "att": "Att", "third": "3rd", "half": "Half",
    "ppda": "PPDA", "gda": "GDA", "psv": "PSV", "hotzone": "Hot Zone",
    "completed": "Cmp",
}

# Tokens dropped outright -- they don't add information once the rest of the
# name is legible (e.g. "passes_attempted" -> "Passes", not "Passes Attempted").
DROP_TOKENS = {"attempted", "taken", "computed", "linked", "total"}

ZONE_SUFFIXES = {
    ("def", "third"): "Def 3rd", ("mid", "third"): "Mid 3rd", ("att", "third"): "Att 3rd",
    ("def", "half"): "Def Half", ("att", "half"): "Att Half",
}
SPLIT_SUFFIXES = {"1h": "1H", "2h": "2H", "home": "Home", "away": "Away"}


def _title(tok):
    low = tok.lower()
    if low in TOKEN_MAP:
        return TOKEN_MAP[low]
    if tok.isdigit():
        return tok
    return tok.capitalize()


def display_name(col):
    if col in EXACT_OVERRIDES:
        return EXACT_OVERRIDES[col]

    tokens = col.split("_")
    zscore_suffix = None
    suffixes = []  # collected trailing markers, in the order they should render

    if tokens and tokens[-1] == "z":
        zscore_suffix = "(Z)"   # rendered last, regardless of strip order below
        tokens = tokens[:-1]
    if tokens and tokens[-1] == "per90":
        suffixes.append("/90")
        tokens = tokens[:-1]
    if tokens and tokens[-1] == "pct":
        suffixes.append("%")
        tokens = tokens[:-1]
    if tokens and tokens[-1] in SPLIT_SUFFIXES:
        suffixes.append(f"({SPLIT_SUFFIXES[tokens[-1]]})")
        tokens = tokens[:-1]
    if len(tokens) >= 2 and tuple(tokens[-2:]) in ZONE_SUFFIXES:
        suffixes.append(f"({ZONE_SUFFIXES[tuple(tokens[-2:])]})")
        tokens = tokens[:-2]
    units_suffix = None
    if tokens and tokens[-1] == "m":
        units_suffix = "(m)"
        tokens = tokens[:-1]

    tokens = [t for t in tokens if t.lower() not in DROP_TOKENS]
    base = " ".join(_title(t) for t in tokens) if tokens else col

    all_suffixes = ([units_suffix] if units_suffix else []) + suffixes + ([zscore_suffix] if zscore_suffix else [])
    label = base
    for s in all_suffixes:
        label = f"{label} {s}"
    return label


if __name__ == "__main__":
    import csv
    import sys
    with open(sys.argv[1], encoding="utf-8-sig") as f:
        header = next(csv.reader(f))
    seen = {}
    for col in header:
        name = display_name(col)
        print(f"{col:<45} -> {name}")
        seen.setdefault(name, []).append(col)
    dupes = {k: v for k, v in seen.items() if len(v) > 1}
    if dupes:
        print("\nCOLLISIONS:")
        for name, cols in dupes.items():
            print(f"  {name!r}: {cols}")

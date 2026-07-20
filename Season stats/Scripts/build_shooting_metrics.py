"""Build Season stats/Shooting Metrics.xlsx from the season's event-derived shot
and minutes data (Danger/all_eredivisie_danger_models.csv, GDA/gda_player_summary.csv,
xT/xt_team_summary.csv). Two tabs: Per 90 and Total.
"""
import csv
import os
from collections import defaultdict, Counter

import pandas as pd
from openpyxl.styles import Font, Alignment, PatternFill
from openpyxl.utils import get_column_letter

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT_DIR = os.path.join(ROOT, "Season stats")
OUT_PATH = os.path.join(OUT_DIR, "Shooting Metrics.xlsx")


def rows(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


# --- team names -------------------------------------------------------------
team_names = {}
for r in rows(os.path.join(ROOT, "xT", "xt_team_summary.csv")):
    team_names[r["contestant_id"]] = r["team_name"]

# --- minutes / appearances per player (season totals from per-match rows) ---
minutes = defaultdict(lambda: {"name": "", "contestant_ids": Counter(), "minutes": 0.0, "matches": 0})
for r in rows(os.path.join(ROOT, "GDA", "gda_player_summary.csv")):
    pid = r["player_id"]
    m = minutes[pid]
    m["name"] = r["player_name"]
    m["contestant_ids"][r["contestant_id"]] += 1
    m["minutes"] += num(r["minutes"])
    m["matches"] += int(num(r["matches"]))

# --- shot-level shooting metrics ---------------------------------------------
shot_rows = rows(os.path.join(ROOT, "Danger", "all_eredivisie_danger_models.csv"))

agg = defaultdict(lambda: {
    "name": "", "contestant_ids": Counter(),
    "shots": 0, "sot": 0, "post": 0, "blocked": 0,
    "pen_att": 0, "pen_goals": 0, "goals": 0,
    "xg": 0.0, "npxg": 0.0, "psxg": 0.0, "xgot": 0.0, "danger": 0.0,
})
for r in shot_rows:
    pid = r["player_id"]
    a = agg[pid]
    a["name"] = r["player_name"] or a["name"] or "Unknown"
    a["contestant_ids"][r["contestant_id"]] += 1
    is_pen = r["is_penalty"] == "1"
    is_goal = r["is_goal"] == "1"

    a["shots"] += 1
    a["sot"] += int(r["is_on_target"] == "1")
    a["post"] += int(r["is_post"] == "1")
    a["blocked"] += int(r["is_outfield_block"] == "1")
    a["goals"] += int(is_goal)
    a["pen_att"] += int(is_pen)
    a["pen_goals"] += int(is_pen and is_goal)
    a["xg"] += num(r["xg"])
    a["npxg"] += 0.0 if is_pen else num(r["xg"])
    a["psxg"] += num(r["psxg"])
    a["xgot"] += num(r["xgot"])
    a["danger"] += num(r["danger_score"])

# --- assemble one row per shooter -------------------------------------------
records = []
for pid, a in agg.items():
    if a["shots"] == 0:
        continue
    m = minutes.get(pid)
    mins = m["minutes"] if m else 0.0
    matches = m["matches"] if m else 0
    cid_counter = m["contestant_ids"] if m and m["contestant_ids"] else a["contestant_ids"]
    contestant_id = cid_counter.most_common(1)[0][0] if cid_counter else None
    team = team_names.get(contestant_id, "Unknown")
    name = (m["name"] if m else "") or a["name"]

    goals, xg, npxg, psxg, xgot, danger = a["goals"], a["xg"], a["npxg"], a["psxg"], a["xgot"], a["danger"]
    np_goals = goals - a["pen_goals"]

    records.append({
        "Player": name,
        "Team": team,
        "Matches": matches,
        "Minutes": round(mins, 1),
        "Shots": a["shots"],
        "SoT": a["sot"],
        "SoT %": round(a["sot"] / a["shots"] * 100, 1) if a["shots"] else 0.0,
        "Goals": goals,
        "Pen Goals": a["pen_goals"],
        "Pen Att": a["pen_att"],
        "xG": round(xg, 3),
        "npxG": round(npxg, 3),
        "PSxG": round(psxg, 3),
        "xGOT": round(xgot, 3),
        "Danger": round(danger, 3),
        "xG/Shot": round(xg / a["shots"], 3) if a["shots"] else 0.0,
        "G-xG": round(goals - xg, 3),
        "npG-npxG": round(np_goals - npxg, 3),
        "G-PSxG": round(goals - psxg, 3),
    })

total_df = pd.DataFrame.from_records(records)
total_df.sort_values("xG", ascending=False, inplace=True)
total_df.reset_index(drop=True, inplace=True)

# --- per-90 tab ---------------------------------------------------------------
per90_cols = ["Shots", "SoT", "Goals", "Pen Goals", "Pen Att", "xG", "npxG", "PSxG", "xGOT", "Danger"]
per90_df = total_df.copy()
factor = per90_df["Minutes"].replace(0, pd.NA) / 90.0
for col in per90_cols:
    per90_df[f"{col}/90"] = (per90_df[col] / factor).round(3)
per90_df["G-xG/90"] = (per90_df["G-xG"] / factor).round(3)
per90_df["npG-npxG/90"] = (per90_df["npG-npxG"] / factor).round(3)
per90_df["G-PSxG/90"] = (per90_df["G-PSxG"] / factor).round(3)

per90_keep = (
    ["Player", "Team", "Matches", "Minutes"]
    + [f"{c}/90" for c in per90_cols]
    + ["SoT %", "xG/Shot", "G-xG/90", "npG-npxG/90", "G-PSxG/90"]
)
per90_df = per90_df[per90_keep].fillna(0.0)
per90_df.sort_values("xG/90", ascending=False, inplace=True)
per90_df.reset_index(drop=True, inplace=True)

# --- write workbook -------------------------------------------------------
os.makedirs(OUT_DIR, exist_ok=True)
with pd.ExcelWriter(OUT_PATH, engine="openpyxl") as writer:
    per90_df.to_excel(writer, sheet_name="Per 90", index=False)
    total_df.to_excel(writer, sheet_name="Total", index=False)

# --- formatting -------------------------------------------------------------
from openpyxl import load_workbook

wb = load_workbook(OUT_PATH)
header_font = Font(name="Arial", bold=True, color="FFFFFF")
header_fill = PatternFill("solid", fgColor="1F4E78")
body_font = Font(name="Arial")

for sheet_name in ("Per 90", "Total"):
    ws = wb[sheet_name]
    ws.freeze_panes = "C2"
    for col_idx, cell in enumerate(ws[1], start=1):
        cell.font = header_font
        cell.fill = header_fill
        cell.alignment = Alignment(horizontal="center")
        col_letter = get_column_letter(col_idx)
        max_len = max(len(str(cell.value)), 8)
        for row_cell in ws[col_letter][1:]:
            row_cell.font = body_font
            if isinstance(row_cell.value, (int, float)) and col_idx > 2:
                row_cell.number_format = "0.0" if abs(row_cell.value) >= 10 or row_cell.value == int(row_cell.value) else "0.000"
            max_len = max(max_len, len(str(row_cell.value)))
        ws.column_dimensions[col_letter].width = min(max_len + 2, 30)
    ws.auto_filter.ref = ws.dimensions

wb.save(OUT_PATH)
print(f"Wrote {OUT_PATH}: {len(total_df)} shooters")

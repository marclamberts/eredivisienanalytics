"""
Build one self-contained HTML stats site covering every season this repo has
aggregated (2023-2024, 2024-2025, 2025-2026): every metric, split into
FBref-style category tabs, sortable, searchable -- "Opta meets FBref".

All data is embedded inline as JSON (no fetch() of local files, which most
browsers block on file:// -- this has to open by double-click, not a server).

Usage: python3 build_html_site.py
"""
import csv
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_PATH = os.path.join(ROOT, "Aggregated", "site", "index.html")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from column_layout import categorize, PLAYER_TAB_RULES, TEAM_TAB_RULES
from display_names import display_name

SEASONS = ["2023-2024", "2024-2025", "2025-2026"]
PLAYER_PREFIX = ["player_name", "team_name", "minutes", "matches"]
TEAM_PREFIX = ["team_name", "matches", "points"]


def read_csv(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        r = csv.reader(f)
        header = next(r)
        rows = list(r)
    return header, rows


def is_number(s):
    if s in (None, ""):
        return False
    try:
        float(s)
        return True
    except ValueError:
        return False


def build_tabs(csv_path, rules, prefix, identity_cols):
    header, rows = read_csv(csv_path)
    col_index = {name: i for i, name in enumerate(header)}
    raw_tabs = categorize(header, rules, identity_cols=identity_cols)
    prefix_idx = [col_index[c] for c in prefix if c in col_index]
    prefix_present = [c for c in prefix if c in col_index]

    reliable_idx = col_index.get("reliable_sample")

    tabs = []
    for tab_name, cols in raw_tabs:
        cols = [c for c in cols if c not in prefix]
        if not cols:
            continue
        idx = [col_index[c] for c in cols]
        # skip tabs with no data at all this season (e.g. "new metrics" for
        # 2023-2024/2024-2025, which this repo's model pipeline never ran on)
        has_any_data = any(any(row[i] not in ("", None) for i in idx) for row in rows)
        if not has_any_data:
            continue
        display_cols = [display_name(c) for c in prefix_present] + [display_name(c) for c in cols]
        raw_cols = prefix_present + cols
        table_rows = []
        for row in rows:
            vals = [row[i] for i in prefix_idx] + [row[i] for i in idx]
            table_rows.append(vals)
        tabs.append({"name": tab_name, "cols": display_cols, "rawCols": raw_cols, "rows": table_rows})

    reliable = [row[reliable_idx] == "True" for row in rows] if reliable_idx is not None else [True] * len(rows)
    return tabs, reliable


def main():
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    payload = {}
    for season in SEASONS:
        season_dir = os.path.join(ROOT, "Aggregated", season)
        player_csv = os.path.join(season_dir, "player_season_aggregated.csv")
        team_csv = os.path.join(season_dir, "team_season_aggregated.csv")
        if not os.path.exists(player_csv):
            print(f"Skipping {season}: no aggregated data yet")
            continue
        player_tabs, reliable = build_tabs(player_csv, PLAYER_TAB_RULES, PLAYER_PREFIX,
                                            identity_cols=["player_id", "player_name", "team_name"])
        team_tabs, _ = build_tabs(team_csv, TEAM_TAB_RULES, TEAM_PREFIX, identity_cols=["team_name"])
        for tab in player_tabs:
            tab["reliable"] = reliable
        payload[season] = {"playerTabs": player_tabs, "teamTabs": team_tabs}
        n_players = len(player_tabs[0]["rows"]) if player_tabs else 0
        n_teams = len(team_tabs[0]["rows"]) if team_tabs else 0
        print(f"{season}: {len(player_tabs)} player tabs ({n_players} players), "
              f"{len(team_tabs)} team tabs ({n_teams} teams)")

    data_json = json.dumps(payload, separators=(",", ":"))
    html = TEMPLATE.replace("__SEASONS__", json.dumps(SEASONS)).replace("__DATA__", data_json)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.write(html)
    size_mb = os.path.getsize(OUT_PATH) / 1_000_000
    print(f"Wrote {OUT_PATH} ({size_mb:.1f} MB)")


TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Eredivisie Stats -- 2023-24 to 2025-26</title>
<style>
:root{
  --navy:#0b1e33; --navy-2:#122a45; --accent:#1c7ed6; --accent-2:#0ca678;
  --paper:#ffffff; --row-alt:#f4f7fa; --border:#dde3ea; --text:#1b2733;
  --text-muted:#5c6b7a; --header-text:#eaf1f8;
}
*{box-sizing:border-box}
body{margin:0;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
  color:var(--text);background:var(--paper);font-size:14px}
header{background:linear-gradient(135deg,var(--navy) 0%,var(--navy-2) 100%);color:var(--header-text);
  padding:18px 24px 0}
.brand{display:flex;align-items:baseline;gap:10px;margin-bottom:14px}
.brand .mark{background:var(--accent);color:#fff;font-weight:800;border-radius:6px;padding:3px 8px;font-size:15px;
  letter-spacing:.02em}
.brand h1{font-size:19px;margin:0;font-weight:700;letter-spacing:.01em}
.brand .sub{color:#9fb3c8;font-size:12.5px}
.seasons{display:flex;gap:6px;padding-bottom:0}
.season-btn{background:transparent;border:none;color:#a9bdd1;padding:9px 16px;font-size:13.5px;font-weight:600;
  cursor:pointer;border-bottom:3px solid transparent;border-radius:6px 6px 0 0}
.season-btn:hover{color:#fff;background:rgba(255,255,255,.06)}
.season-btn.active{color:#fff;border-bottom-color:var(--accent);background:rgba(255,255,255,.08)}
.view-toggle{display:flex;gap:6px;padding:10px 24px;background:var(--navy-2);border-bottom:1px solid rgba(255,255,255,.08)}
.view-btn{background:rgba(255,255,255,.06);border:1px solid rgba(255,255,255,.12);color:#cfe0f0;padding:6px 14px;
  font-size:12.5px;font-weight:600;border-radius:16px;cursor:pointer}
.view-btn.active{background:var(--accent);border-color:var(--accent);color:#fff}
.tabbar{display:flex;flex-wrap:wrap;gap:4px;padding:10px 24px;background:#eef2f6;border-bottom:1px solid var(--border);
  position:sticky;top:0;z-index:5}
.tab-btn{background:#fff;border:1px solid var(--border);color:var(--text-muted);padding:6px 13px;font-size:12.5px;
  font-weight:600;border-radius:14px;cursor:pointer;white-space:nowrap}
.tab-btn:hover{border-color:var(--accent);color:var(--accent)}
.tab-btn.active{background:var(--navy);border-color:var(--navy);color:#fff}
.controls{display:flex;flex-wrap:wrap;align-items:center;gap:14px;padding:12px 24px;background:#fff;
  border-bottom:1px solid var(--border)}
.controls input[type=text]{padding:7px 11px;border:1px solid var(--border);border-radius:6px;font-size:13px;
  min-width:220px}
.controls select{padding:7px 10px;border:1px solid var(--border);border-radius:6px;font-size:13px;background:#fff}
.controls label{font-size:12.5px;color:var(--text-muted);display:flex;align-items:center;gap:6px;cursor:pointer}
.controls .count{margin-left:auto;font-size:12px;color:var(--text-muted)}
.table-wrap{overflow:auto;max-height:calc(100vh - 210px)}
table{border-collapse:collapse;width:100%;font-size:12.5px}
thead th{position:sticky;top:0;background:var(--navy);color:var(--header-text);text-align:right;padding:8px 10px;
  font-weight:600;cursor:pointer;white-space:nowrap;border-right:1px solid rgba(255,255,255,.08);z-index:2}
thead th:nth-child(1),thead th:nth-child(2){text-align:left}
thead th:nth-child(1){position:sticky;left:0;z-index:4;width:40px}
thead th:nth-child(2){position:sticky;left:40px;z-index:4;width:170px}
thead th .raw{display:block;font-weight:400;font-size:9.5px;color:#9fb3c8;text-transform:none}
thead th.sorted-asc:after{content:" \25B2";color:var(--accent-2)}
thead th.sorted-desc:after{content:" \25BC";color:var(--accent-2)}
tbody td{padding:5px 10px;text-align:right;white-space:nowrap;border-bottom:1px solid #eef1f4;background:var(--paper)}
tbody td:nth-child(1),tbody td:nth-child(2){text-align:left;position:sticky;z-index:1}
tbody td:nth-child(1){left:0;font-weight:600;width:40px}
tbody td:nth-child(2){left:40px;width:170px}
tbody tr:nth-child(even) td{background:var(--row-alt)}
tbody tr:hover td{background:#e7f1fb}
tbody tr.dim{opacity:.4}
.rank{color:var(--text-muted);font-variant-numeric:tabular-nums}
footer{padding:16px 24px 28px;color:var(--text-muted);font-size:11.5px;border-top:1px solid var(--border)}
.pill{display:inline-block;background:#eef2f6;border-radius:10px;padding:1px 8px;font-size:10.5px;color:var(--text-muted)}
</style>
</head>
<body>
<header>
  <div class="brand">
    <span class="mark">ED</span>
    <h1>Eredivisie Stats</h1>
    <span class="sub">Every metric, every season &middot; built from Opta event data in this repo</span>
  </div>
  <div class="seasons" id="seasonBar"></div>
</header>
<div class="view-toggle">
  <button class="view-btn active" data-view="players">Players</button>
  <button class="view-btn" data-view="teams">Teams</button>
</div>
<div class="tabbar" id="tabBar"></div>
<div class="controls">
  <input type="text" id="search" placeholder="Search player or team...">
  <select id="teamFilter"><option value="">All teams</option></select>
  <label id="reliableLabel"><input type="checkbox" id="reliableOnly" checked> 450+ minutes only</label>
  <span class="count" id="rowCount"></span>
</div>
<div class="table-wrap"><table id="dataTable"><thead></thead><tbody></tbody></table></div>
<footer>
  Data: Opta event data via this repo's own aggregation pipeline (<span class="pill">Aggregated/build_season_aggregate.py</span>).
  2023-24 and 2024-25 show the counting-stat categories only -- the "New Metrics" tabs (xT, GDA, disruption value, expected box
  entries, crossing xP) were only computed for 2025-26 so far and are hidden here rather than shown empty.
  Click a column header to sort; click again to reverse. Blank cells sort to the bottom in either direction.
</footer>
<script>
const SEASONS = __SEASONS__;
const DATA = __DATA__;

let state = { season: SEASONS[SEASONS.length-1], view: "players", tabIndex: 0, sortCol: null, sortDir: 1,
              search: "", team: "", reliableOnly: true };

function seasonData(){ return DATA[state.season]; }
function tabs(){ return state.view === "players" ? seasonData().playerTabs : seasonData().teamTabs; }
function currentTab(){ return tabs()[Math.min(state.tabIndex, tabs().length-1)]; }

function renderSeasonBar(){
  const bar = document.getElementById("seasonBar");
  bar.innerHTML = "";
  SEASONS.forEach(s => {
    if(!DATA[s]) return;
    const b = document.createElement("button");
    b.className = "season-btn" + (s===state.season ? " active" : "");
    b.textContent = s;
    b.onclick = () => { state.season = s; state.tabIndex = 0; state.sortCol = null; render(); };
    bar.appendChild(b);
  });
}

function renderTabBar(){
  const bar = document.getElementById("tabBar");
  bar.innerHTML = "";
  tabs().forEach((t, i) => {
    const b = document.createElement("button");
    b.className = "tab-btn" + (i===state.tabIndex ? " active" : "");
    b.textContent = t.name;
    b.onclick = () => { state.tabIndex = i; state.sortCol = null; render(); };
    bar.appendChild(b);
  });
}

function renderTeamFilter(){
  const sel = document.getElementById("teamFilter");
  const teams = new Set();
  seasonData().playerTabs[0].rows.forEach(r => teams.add(r[1]));
  const prev = sel.value;
  sel.innerHTML = '<option value="">All teams</option>';
  Array.from(teams).sort().forEach(t => {
    const o = document.createElement("option"); o.value = t; o.textContent = t; sel.appendChild(o);
  });
  sel.value = teams.has(prev) ? prev : "";
  state.team = sel.value;
}

function fmt(v){
  if(v === "" || v === null || v === undefined) return "";
  const n = Number(v);
  if(Number.isNaN(n)) return v;
  if(Number.isInteger(n)) return n.toLocaleString();
  return n.toFixed(2);
}

function cmp(a, b, dir){
  const ea = a === "", eb = b === "";
  if(ea && eb) return 0;
  if(ea) return 1;   // blanks always last, regardless of direction
  if(eb) return -1;
  const na = Number(a), nb = Number(b);
  if(!Number.isNaN(na) && !Number.isNaN(nb)) return (na - nb) * dir;
  return String(a).localeCompare(String(b)) * dir;
}

function render(){
  renderSeasonBar();
  renderTabBar();
  if(state.view === "players") renderTeamFilter();
  const tab = currentTab();
  const thead = document.querySelector("#dataTable thead");
  const tbody = document.querySelector("#dataTable tbody");

  if(state.sortCol === null){ state.sortCol = tab.cols.length > 2 ? 2 : 0; state.sortDir = -1; }

  let rows = tab.rows.map((r, i) => ({ r, reliable: tab.reliable ? tab.reliable[i] : true }));
  if(state.search){
    const q = state.search.toLowerCase();
    rows = rows.filter(x => x.r[0].toLowerCase().includes(q) || x.r[1].toLowerCase().includes(q));
  }
  if(state.view === "players" && state.team){
    rows = rows.filter(x => x.r[1] === state.team);
  }
  const showDimmed = state.view === "players" && !state.reliableOnly;
  if(state.view === "players" && state.reliableOnly){
    rows = rows.filter(x => x.reliable);
  }
  rows.sort((x, y) => cmp(x.r[state.sortCol], y.r[state.sortCol], state.sortDir));

  thead.innerHTML = "";
  const trh = document.createElement("tr");
  const rankTh = document.createElement("th"); rankTh.textContent = "#"; rankTh.style.textAlign="right";
  trh.appendChild(rankTh);
  tab.cols.forEach((c, i) => {
    const th = document.createElement("th");
    th.innerHTML = c + '<span class="raw">' + tab.rawCols[i] + "</span>";
    if(i === state.sortCol) th.classList.add(state.sortDir === 1 ? "sorted-asc" : "sorted-desc");
    th.onclick = () => {
      if(state.sortCol === i) state.sortDir *= -1; else { state.sortCol = i; state.sortDir = i < 2 ? 1 : -1; }
      render();
    };
    trh.appendChild(th);
  });
  thead.appendChild(trh);

  tbody.innerHTML = "";
  rows.forEach((x, rank) => {
    const tr = document.createElement("tr");
    if(showDimmed && !x.reliable) tr.classList.add("dim");
    const rankTd = document.createElement("td"); rankTd.textContent = rank+1; rankTd.className = "rank";
    tr.appendChild(rankTd);
    x.r.forEach(v => {
      const td = document.createElement("td");
      td.textContent = fmt(v);
      tr.appendChild(td);
    });
    tbody.appendChild(tr);
  });
  document.getElementById("rowCount").textContent = rows.length + " rows";
}

document.querySelectorAll(".view-btn").forEach(b => b.onclick = () => {
  document.querySelectorAll(".view-btn").forEach(x => x.classList.remove("active"));
  b.classList.add("active");
  state.view = b.dataset.view; state.tabIndex = 0; state.sortCol = null;
  document.getElementById("teamFilter").style.display = state.view === "players" ? "" : "none";
  document.getElementById("reliableLabel").style.display = state.view === "players" ? "" : "none";
  render();
});
document.getElementById("search").addEventListener("input", e => { state.search = e.target.value; render(); });
document.getElementById("teamFilter").addEventListener("change", e => { state.team = e.target.value; render(); });
document.getElementById("reliableOnly").addEventListener("change", e => { state.reliableOnly = e.target.checked; render(); });

render();
</script>
</body>
</html>
"""

if __name__ == "__main__":
    main()

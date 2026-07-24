# Eredivisie Stats site

`index.html` -- one self-contained file (no server, no external requests;
just open it) covering every metric this repo has aggregated for
2023-2024, 2024-2025 and 2025-2026: an "Opta meets FBref" stats database --
FBref's category-tabbed, sortable, dense tables, in a cleaner navy/white
shell.

Built by `../build_html_site.py`, which reads each season's
`player_season_aggregated.csv` / `team_season_aggregated.csv`, splits them
into tabs with `../column_layout.py` (the same rules the xlsx workbook
uses), relabels columns with `../display_names.py`, and embeds everything
as inline JSON -- deliberately not `fetch()`-ed from separate files, since
browsers block that against `file://` and this has to open by double-click.

## What's there per season

- **2025-2026**: all 13 player tabs + 4 team tabs, including the two "New
  Metrics" tabs (xT, GDA, disruption value, expected box entries, crossing
  xP) -- those models have only been run for this season.
- **2023-2024 / 2024-2025**: 12 player tabs + 2 team tabs -- every
  Wyscout-style counting-stat category (passing, crossing, duels, defensive
  actions, discipline, shooting, set pieces, goalkeeping, creativity
  including the delivery-type xA/key-pass split, progression, splits, and
  the possession-adjusted PAdj defensive stats, all of which this repo's
  own event parser computes directly and needed no external model). Tabs
  that would be entirely empty for a season (the two "New Metrics" tabs)
  are hidden rather than shown blank.

## UI

- Season tabs, then Players/Teams toggle, then category tabs (FBref-style).
- Click a column header to sort; click again to reverse. Blanks always
  sort last, in either direction, rather than as 0.
- Search box filters by player or team name; team dropdown (players view
  only); "450+ minutes only" checkbox (the same `reliable_sample` cutoff
  used throughout `Aggregated/`) toggles between the reliable-sample subset
  and everyone, dimming the excluded rows rather than hiding them so a
  sub-450-minute breakout player is still discoverable.
- The small grey text under each header is the exact CSV column name, for
  anyone cross-referencing a formula against the raw data.

## Regenerating

```
python3 ../build_season_aggregate.py <season>   # for each season first
python3 ../build_html_site.py                   # rebuilds index.html from all 3
```

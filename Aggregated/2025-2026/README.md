# Season 2025-2026 aggregated data

Five files, all built from the same numbers:

- `player_season_aggregated.csv` -- 553 players x 393 metric columns, flat,
  exact snake_case names. Built by `../build_season_aggregate.py`.
- `team_season_aggregated.csv` -- 18 teams, flat. Same script.
- `eredivisie_2025-2026_aggregated.xlsx` -- the same numbers split across 13
  player tabs + 4 team tabs + a Glossary tab (below), so nobody has to scroll
  a 393-column sheet to find "tackles." Built by `../build_workbook.py`.
- `eredivisie_2025-2026_explore.ipynb` -- a short exploratory notebook
  (leaderboards + 4 Meridian-house-style charts, including the xA-by-
  delivery-type breakdown). Built by `../build_notebook.py`, executed with
  `jupyter nbconvert --execute`.

## Concise names vs. exact names

The CSVs keep exact, unambiguous snake_case column names (`padj_tackles_
attempted_per90_z`) -- that's what any script, formula, or the
`column_layout.py` category rules key off. The **xlsx and notebook** relabel
every column through `../display_names.py` for reading (`PAdj Tackles /90
(Z)`): row 1 of every xlsx tab is the concise label, row 2 (small, grey) is
the exact CSV column name it maps to, so a formula referencing a specific
field always has an unambiguous answer without leaving the tab. Run
`python3 display_names.py <csv path>` to print every column's exact name
next to its display label (also flags any accidental collisions).

Full season: 309 matches, 2025-08-08 to 2026-05-24. A few teams show 35-36
matches instead of 34 because of the Eredivisie's end-of-season European-
qualification play-offs, not a data error.

This reads like a Wyscout player/team statistics export -- the standard
counting stats (passing, crossing, duels, defensive actions, discipline,
goalkeeping, shooting, set pieces, key passes/assists/xA, progression),
mostly per-90, plus this repo's own "new metrics" (xT progression value,
GDA, disruption value, expected box entries, pass-shot value, hot-zone
passing, expected-completion crossing value, possession-adjusted defensive
volume) joined on.

## Two different origins, one row

- **Wyscout-style counting stats** are computed **directly from the raw
  Opta event stream** in `Events/2025-2026/*.json` by
  `build_season_aggregate.py` itself -- nothing to join here, no per-metric
  folder in this repo already had them.
- **New metrics** (xT, GDA, disruption value, box entries, pass-shot value,
  hot-zone passing, crossing xP) are **joined in** from the metric folders
  that already existed (`xT/`, `GDA/`, `Disruption/CSV/`, `Box Entry
  Models/`, `Cross Models/`) -- not recomputed.

Join keys: `player_id` (shared Opta id across xT/GDA/Danger/Disruption) where
available; `(player_name, team_name)` for the files that don't carry an id
(Box Entry, Pass-Shot-Value, hot-zone, and Cross Models player leaderboards)
-- verified to overlap 100% with the id-joined player set before relying on
it. Team name strings were checked identical across all sources.

## Event-code reference (verified against this season's data, not assumed)

Two of this repo's own scripts (`Disruption/build_disruption_model.py` and
`Box Entry Models/build_box_entry_model.py`) disagree with each other on what
qualifiers 1, 15 and 107 mean. Every code used below was checked directly
against `Events/2025-2026`: qualifier 107 sits on events whose start point is
on the touchline 100% of the time (throw-in, not "long ball"); qualifier 1
sits mostly on ~43m passes (long ball, not "head"); qualifier 15 sits mostly
on clearances/shots, not passes (body part = head). Fouls, offsides and cards
were checked by finding the paired event each one generates.

```
typeId   1 Pass            2 Offside Pass      3 Take On        4 Foul
         6 Corner Awarded  7 Tackle            8 Interception   10 Save
         11 Claim          12 Clearance        13 Miss          14 Post
         15 Attempt Saved  16 Goal             17 Card          41 Punch
         44 Aerial Duel    49 Ball Recovery    50 Dispossessed  51 Error
         52 Keeper Pick-up 54 Smother          55 Offside Provoked
         58 Penalty Faced  59 Keeper Sweeper   61 Ball Touch

qualifierId  1 Long ball   2 Cross   3 Through ball   5 Free kick   6 Corner
             15 Head   20 Right foot   31 Yellow card   32 Second yellow
             33 Red card   72 Left foot   107 Throw-in   140/141 pass end x/y
             195 Pull back   212 length (m)   213 angle (rad)
```

`fouls_committed` / `fouls_won`: every Foul (typeId 4) fires twice at the same
instant, one team at outcome=0 (committed it), the other at outcome=1 (won the
free kick). Same pattern for offsides: typeId 2 ("Offside Pass") is credited
to the passer, typeId 55 ("Offside Provoked") to the attacker actually in the
offside position -- `offsides` uses typeId 55, matching "times caught
offside" in a normal stats sheet.

## Definitions worth reading before you trust a column

- **Progressive pass/carry**: cuts the distance to the opponent's goal by at
  least 30 yards if both ends are in the player's own half, 15 yards if it
  crosses into the attacking half, or 10 yards if both ends are already in
  the attacking half (a common public heuristic, not this repo's invention).
  Set pieces are not excluded. **Known face-validity problem**: goalkeepers
  can top the progressive-passes leaderboard because long distribution
  satisfies this distance formula the same way a real line-breaking pass
  does -- filter to outfield players before using this for scouting.
- **Carries**: this feed has no tracking data, so "carries" are *inferred*:
  chain a player's own consecutive ball-touching events (pass/take-on/shot/
  ball-touch) when the time gap is <=8 seconds and the ball moved >=3m,
  resetting the chain after a dead-ball restart or a failed pass. This is a
  heuristic, the same kind of inference `xT/xt_model_meta.json` already used
  for its own (separate, not reused here) carry count -- expect it to
  misjudge individual sequences even where the season totals look sane.
- **Key pass / Assist / xA**: for every shot, walk back up to 4 events for
  the nearest completed same-team pass (same convention already used by
  `netlify-app/generate_data.py`). `xa` sums the shot's own xG onto the
  passer regardless of outcome; `assists` only counts shots that scored.
- **By delivery type** (`_cutback`, `_cross`, `_through_ball`, `_set_piece`,
  `_open_play` suffixes on `key_passes`/`assists`/`xa`): the same qualifying
  pass is classified by what it was -- pull-back (qualifier 195), cross
  (qualifier 2), through ball (qualifier 3), free-kick/corner (qualifier 5
  or 6), else open play -- so "who creates from cutbacks" and "who creates
  from open play" don't get blended into one number. The five buckets sum
  exactly to the unbroken `key_passes`/`assists`/`xa` total (checked, not
  assumed).
- **Shot-Creating Actions (SCA) / Goal-Creating Actions (GCA)**: up to the 2
  most recent successful actions (completed pass, successful take-on, or a
  won foul) by the shooting team before a shot/goal, credited to up to 2
  different players -- modelled after the FBref concept, computed here from
  this feed's own events, not sourced from FBref.
- **Passes received**: approximated as the very next on-ball action by a
  teammate within 5 seconds of a completed pass. No explicit "intended
  receiver" tag exists in this feed, so a deflected pass or a receiver who
  took a 6th second will be missed.
- **Crosses (raw)** (`crosses_attempted`, etc.) count every pass carrying the
  cross qualifier, including set pieces -- a *different universe* from
  `xp_cross_*` (from `Cross Models/xp_player_leaderboard_eredivisie.csv`,
  which may be scoped to open play only). The two will not match by design;
  that's not a data error.
- **PAdj (possession-adjusted)** tackles/interceptions/clearances/aerial
  duels/ball recoveries: raw per-90 rate x (league-average opponent-
  possession% / this team's own opponent-possession%), where team
  possession% is this team's average share of total match pass attempts
  across its own matches (computed here, not sourced elsewhere). Teams that
  see less of the ball face more raw defensive opportunities, so this scales
  a low-possession team's defender up and a high-possession team's down for
  a fairer read across team styles.
- **goal_difference_added (GDA)** mixes on-pitch goal differential with an
  action-value model (see `GDA/gda_model_meta.json`) -- a player on a poor
  team can show a large negative value even with good individual
  performances, because it partly reflects team goals conceded while they
  were on the pitch.

## What's still deliberately NOT in here

**No composite score.** Every component is per-90 and, for the core "new
metrics", z-scored among `reliable_sample` players -- enough to build a
weighted composite on top, but combining components into one number is a
modelling choice (which weights, validated how) that hasn't been through face
validity (video review), construct validity (does it just track possession
share / position / minutes?), or weight-sensitivity testing.

**No shrinkage / confidence intervals.** `reliable_sample` (minutes >= 450,
~5 full matches) is a blunt cutoff, not empirical-Bayes shrinkage.

**No cross-season or predictive validity check.** This is one season's
descriptive aggregate.

## The 13 player tabs (and 4 team tabs) in the xlsx

Playing Time, Passing, Progression (raw), Crossing, Creativity, Duels &
Defensive Actions, Discipline, Shooting, Set Pieces, Goalkeeping, Splits -
Half & Home-Away, New Metrics - Progression Value, New Metrics - Defensive &
Overall Value // League Table, Wyscout-style Totals, New Metrics, Style &
Formation. The split rules live in `../column_layout.py` and are shared by
both build scripts, so the CSV column order and the xlsx tabs can't drift
apart.

## Column notes

- `minutes` / `matches` come from `GDA/gda_player_summary.csv` (the only
  source that tracks stint-level on/off pitch time); every `_per90` column
  divides by that figure.
- `xg` / `shots` / `shots_on_target` / `danger_score` are summed from
  `Danger/all_eredivisie_danger_models.csv` shot events; `np_goals`/`np_xg`
  exclude penalties using that file's own `is_penalty` flag.
- Team `matches` differ (34 vs 35 vs 36) because of end-of-season play-off
  matches; don't compare team per-match rates without checking `matches`
  first.
- **`style_*` columns (team table) inherit a bug from their source.** They're
  joined straight from `Analysis/Coach Profiling/team_metrics_aggregated.csv`,
  which is *not* recomputed here. While building `wing_play_comparison.py`
  (below) its source, `Scripts/coach_profiling.py`, turned out to have
  several wrong qualifier constants for this feed -- e.g. its
  `Q_GOAL_KICK = 72` is actually "left foot" (qualifier 72, verified earlier
  in this README), and its `Q_END_X`/`Q_END_Y` (141/140) are swapped versus
  the verified 140=end_x/141=end_y used everywhere else in `Aggregated/`.
  That means its "open play" filter (meant to exclude goal kicks) is
  silently excluding something else instead, which flows into
  `style_wing_pct`, `style_long_ball_pct`, `style_deep_circulation_pct`, and
  `style_territory`. Not fixed here -- fixing `coach_profiling.py` itself is
  a separate task from this season's aggregate. `wing_play_comparison.py`
  recomputes wing play independently with the verified qualifiers rather
  than trusting `style_wing_pct`.

## Wing play comparison

`wing_play_comparison.py` builds `wing_play_by_team.csv` and
`wing_play_comparison.png`: each team's share of open-play passes from the
wide corridors (Opta y<25 or y>75), split left vs. right *from the attacking
team's own perspective* -- which requires correcting for the second-half end
swap, since Opta coordinates don't flip when a team switches ends (direction
per team per half is inferred from their average pass x, same heuristic
Coach Profiling uses). Built fresh from `Events/2025-2026`, not a re-read of
`style_wing_pct` -- see the caveat above for why.

`wing_play_overall.py` builds `wing_play_overall.png`: the same underlying
numbers with no left/right split -- one bar per team, combined wide-corridor
share, standard leaderboard-with-one-highlight treatment.

## Diagonal passing vs. the Relationism Index

`diagonal_vs_relationism.py` builds `diagonal_vs_relationism.csv` and
`diagonal_vs_relationism.png`: a scatter of each team's Relationism Index
(the proxy score from `PSV Season Report/Scripts/relationism_index.py` --
equal-weighted percentile blend of inverse average pass distance, central-
third touch share, and passes per possession sequence) against a new
**diagonal pass %** metric -- the share of a team's completed open-play
passes (excluding free-kick/corner/throw-in) whose direction sits between
25 deg and 65 deg off the horizontal, for passes of at least 5m. Neither
near-straight upfield/backward nor a square ball across the pitch --
theoretically the passing-lane signature relational, proximity-based
combination play is supposed to produce, versus the fixed lines of a more
positional structure.

The Relationism Index is **recomputed here**, not read from
`relationism_index.py`'s output -- that script's `pi_ratings_lib.py`
hardcodes a Mac-only path that doesn't exist in this repo, so its method
(same formula, same weights) is ported to run directly against
`Events/2025-2026` instead. Both are proxy scores built from event data,
not claims about a club's actual coaching philosophy.

Correlation this season is weak (Pearson r = 0.34, stated in the chart
title rather than implied) -- read as "a weak signal, not a strong
relationship," not as evidence relationism causes diagonal passing.

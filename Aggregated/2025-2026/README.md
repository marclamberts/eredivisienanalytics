# Season 2025-2026 aggregated data

`player_season_aggregated.csv` (549 rows) and `team_season_aggregated.csv` (18 rows),
built by `../build_season_aggregate.py`. Full season: 309 matches, 2025-08-08 to
2026-05-24. A few teams show 35-36 matches instead of 34 because of the Eredivisie's
end-of-season European-qualification play-offs, not a data error.

## What this is

A join of the season's existing per-metric outputs onto one row per player and one
row per team, plus the per-90 rates and minutes-based reliability flag that were
missing. It does not recompute xG, xT, disruption value, pass-completion probability,
or any other model — those already live in their own folders (`xT/`, `GDA/`,
`Danger/`, `Disruption/CSV/`, `Box Entry Models/`, `Cross Models/`,
`Analysis/Coach Profiling/`, `Analysis/Formation/`) and, for this season, already
covered the full 309 matches before this file existed.

Join keys: `player_id` (shared Opta id across xT/GDA/Danger/Disruption) where
available; `(player_name, team_name)` for the three files that don't carry an id
(Box Entry, Pass-Shot-Value, hot-zone, and Cross Models player leaderboards) —
verified to overlap 100% with the id-joined player set before relying on it.
Team name strings were checked identical across all six team-level sources, so no
fuzzy matching was needed there.

## What's deliberately NOT in here

**No composite score.** Progression/creativity/disruption components are included
side by side, per-90 and z-scored (among `reliable_sample` players only) so a
weighted composite could be built on top -- but combining them into one number is
a modelling choice that hasn't been through face validity (video review), construct
validity (does it just track possession share / position / minutes?), or weight
sensitivity testing yet. Publishing a single "Progression Score" number without
that work would be exactly the mistake the brief warns against: a mathematically
fine but unvalidated metric.

**No shrinkage / confidence intervals.** `reliable_sample` (minutes >= 450, ~5 full
matches) is a blunt reliability cutoff, not empirical-Bayes shrinkage. Small-sample
players' raw and per-90 numbers are included but should be read as noisy, especially
`_per90` columns for anyone near the low end of `minutes`.

**No cross-season or predictive validity check.** This is one season's descriptive
aggregate. Testing whether any of these components predict next season's output,
or are stable for the same player across seasons, needs 2024-2025 and 2023-2026
run through the same join and is a separate follow-up, not attempted here.

## Column notes

- `minutes` / `matches` come from `GDA/gda_player_summary.csv` (the only source
  that tracks stint-level on/off pitch time); every per-90 column in this file is
  divided by that `minutes` figure.
- `goal_difference_added` mixes on-pitch goal differential with an action-value
  model (see `GDA/gda_model_meta.json`) -- a goalkeeper or defender on a poor team
  can show a large negative value here even with good individual performances,
  because it partly reflects team goals conceded while on the pitch, not solely
  the player's own actions.
- `xg` / `shots` / `danger_score` are summed from `Danger/all_eredivisie_danger_models.csv`
  shot events, joined by `player_id` + `contestant_id`.
- Team `matches` differ (34 vs 35 vs 36) because of end-of-season play-off matches;
  don't compare team per-match rates without checking `matches` first.

# Season 2025-2026 aggregated data

`player_season_aggregated.csv` (553 rows) and `team_season_aggregated.csv` (18 rows),
built by `../build_season_aggregate.py`. Full season: 309 matches, 2025-08-08 to
2026-05-24. A few teams show 35-36 matches instead of 34 because of the Eredivisie's
end-of-season European-qualification play-offs, not a data error.

This is meant to read like a Wyscout player/team statistics export -- the standard
counting stats (passing, crossing, duels, defensive actions, discipline,
goalkeeping, shooting, assists/xA), all per-90 -- plus this repo's own "new metrics"
(xT progression value, GDA, disruption value, expected box entries, pass-shot
value, hot-zone passing, expected-completion crossing value) joined on at the end
of each row.

## Two different origins, one row

- **Wyscout-style counting stats** (passing / crossing / duels / defensive
  actions / discipline / goalkeeping / shooting / key passes / assists / xA) are
  computed **directly from the raw Opta event stream** in `Events/2025-2026/*.json`
  by `build_season_aggregate.py` itself -- nothing to join here, no per-metric
  folder in this repo already had them.
- **New metrics** (xT, GDA, disruption value, box entries, pass-shot value,
  hot-zone passing, crossing xP) are **joined in** from the metric folders that
  already existed (`xT/`, `GDA/`, `Disruption/CSV/`, `Box Entry Models/`,
  `Cross Models/`) -- not recomputed.

Join keys: `player_id` (shared Opta id across xT/GDA/Danger/Disruption) where
available; `(player_name, team_name)` for the four files that don't carry an id
(Box Entry, Pass-Shot-Value, hot-zone, and Cross Models player leaderboards) --
verified to overlap 100% with the id-joined player set before relying on it.
Team name strings were checked identical across all sources.

## Event-code reference (verified against this season's data, not assumed)

Two of this repo's own scripts (`Disruption/build_disruption_model.py` and
`Box Entry Models/build_box_entry_model.py`) disagree with each other on what
qualifiers 1, 15 and 107 mean. Rather than pick one, every code used below was
checked against `Events/2025-2026` directly: qualifier 107 sits on events whose
start point is on the touchline 100% of the time (throw-in, not "long ball" as
the disruption script has it); qualifier 1 sits mostly on ~43m passes (long
ball, not "head" as the disruption script has it); qualifier 15 sits mostly on
clearances/shots, not passes (body part = head, confirming the box-entry
script). Fouls and offsides were checked by finding the paired event each one
generates (same second, opposite team, opposite outcome) to work out which
side of the pair is "committed by" vs "won by"/"suffered by".

```
typeId   1 Pass            2 Offside Pass      3 Take On        4 Foul
         6 Corner Awarded  7 Tackle            8 Interception   10 Save
         11 Claim          12 Clearance        13 Miss          14 Post
         15 Attempt Saved  16 Goal             17 Card          41 Punch
         44 Aerial Duel    49 Ball Recovery    50 Dispossessed  51 Error
         52 Keeper Pick-up 54 Smother          55 Offside Provoked
         58 Penalty Faced  59 Keeper Sweeper

qualifierId  1 Long ball   2 Cross   3 Through ball   5 Free kick   6 Corner
             15 Head   20 Right foot   31 Yellow card   32 Second yellow
             33 Red card   72 Left foot   107 Throw-in   140/141 pass end x/y
             195 Pull back   212 length (m)   213 angle (rad)
```

`fouls_committed` / `fouls_won`: every Foul (typeId 4) fires twice at the same
instant, one team at outcome=0 (committed it) and the other at outcome=1 (won
the free kick) -- confirmed by inspecting a sample directly. Same pattern for
offsides: typeId 2 ("Offside Pass") is credited to the passer, typeId 55
("Offside Provoked") to the attacker who was actually in the offside position
-- `offsides` below uses typeId 55, matching what "times caught offside" means
in a normal stats sheet.

## Definitions worth stating explicitly

- **Progressive pass**: a completed pass that cuts the distance to the
  opponent's goal by at least 30 yards if both ends are in the passer's own
  half, 15 yards if it crosses into the attacking half, or 10 yards if both
  ends are already in the attacking half (a common public heuristic, not this
  repo's invention). Set pieces (corners, free kicks, throw-ins) are **not**
  excluded, so a team's regular corner-taker or long-throw specialist will
  look inflated here relative to a stricter open-play-only version.
- **Key pass / assist / xA**: for every shot, walk back up to 4 events and
  credit the nearest completed same-team pass -- same convention already used
  by `netlify-app/generate_data.py`, not something new introduced here.
- `xa` (expected assists) sums the *shot's* xG onto the passer, whether or not
  the shot was scored; `assists` only counts the shots that actually went in.

## Known face-validity problem -- read before using `progressive_passes`

Sorting `progressive_passes_per90` puts several **goalkeepers** at the very
top (long defensive distribution covers 40+ metres just like a genuine
line-breaking pass does, and the formula above can't tell them apart). This is
exactly the kind of face-validity failure the brief this was built against
warns about: a metric can be arithmetically correct and still misrepresent
the football concept it's named after. Two honest options, neither applied
here without a decision from you: (a) filter to outfield players before using
this column, or (b) exclude the deepest zone of the pitch from the "own half"
progressive threshold so keeper distribution stops qualifying. Left as-is so
the column stays an unfiltered, auditable count.

## What's still deliberately NOT in here

**No composite score.** Every component above is per-90 and, for the core
"new metrics", z-scored among `reliable_sample` players -- enough to build a
weighted composite on top, but combining them into one number is a modelling
choice (which weights, validated how) that hasn't been through face validity
(video review), construct validity (does it just track possession share /
position / minutes?), or weight-sensitivity testing yet.

**No shrinkage / confidence intervals.** `reliable_sample` (minutes >= 450,
~5 full matches) is a blunt cutoff, not empirical-Bayes shrinkage. Per-90
figures for players below that line should be read as noisy.

**No cross-season or predictive validity check.** This is one season's
descriptive aggregate. Testing whether any component predicts next season's
output, or is stable for the same player across seasons, needs 2024-2025 and
2023-2024 run through the same script and is a separate follow-up.

## Column notes

- `minutes` / `matches` come from `GDA/gda_player_summary.csv` (the only
  source that tracks stint-level on/off pitch time); every `_per90` column is
  divided by that figure.
- `goal_difference_added` mixes on-pitch goal differential with an action
  value model (see `GDA/gda_model_meta.json`) -- a goalkeeper or defender on a
  poor team can show a large negative value even with good individual
  performances, because it partly reflects team goals conceded while on the
  pitch.
- `xg` / `shots` / `shots_on_target` / `danger_score` are summed from
  `Danger/all_eredivisie_danger_models.csv` shot events.
- Team `matches` differ (34 vs 35 vs 36) because of end-of-season play-off
  matches; don't compare team per-match rates without checking `matches`
  first.

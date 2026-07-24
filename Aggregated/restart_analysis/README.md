# Goal kicks and open-play long balls: full metric framework, 2023-24 to 2025-26

Built by `../restart_analysis.py`. Goal kicks and open-play long balls are
computed **separately** throughout, per the brief -- they share the same
sequence-tracking engine but never share a denominator.

## Outputs

- `Aggregated/<season>/restart_goal_kick_team.csv` -- one row per team, every
  metric below, for goal-kick restarts that season.
- `Aggregated/<season>/restart_long_ball_team.csv` -- same, for open-play
  long balls.
- `league_wide_by_season.csv` -- all 18 teams pooled, one row per
  season x kind (the same aggregation function, run on every team's restarts
  together instead of one team's).
- `three_season_comparison.csv` -- every metric's 2023-24 / 2024-25 / 2025-26
  league-wide value, `delta` (season 3 minus season 1), `%change`, and the
  linear-trend slope (`β1`, average annual change via OLS on season index
  1/2/3). No competition/score-state/opponent-strength/venue controls are
  applied -- the brief asks for them "for fair comparisons"; this is the
  descriptive layer underneath that, not a substitute for it.

## Foundational definition, verified rather than assumed

**Goal kick = qualifier 124** on a pass event. Checked directly against
`Events/2025-2026` before trusting it (see `restart_analysis.py`'s
docstring): every occurrence sits within a few metres of a goal at central
y, the taker is a known goalkeeper, outcome and length are both mixed (goal
kicks aren't always long or always successful), and per-match frequency
(~11-23 combined) is exactly what a goal-kick count should look like. This
matters because two of this repo's *own* existing scripts get qualifiers
wrong for this feed (see `Aggregated/2025-2026/README.md`), so nothing here
was assumed from precedent.

**Long goal kick**: end location at least 40m from the *taking* team's own
goal (the brief's threshold, applied to whichever goal the kick is actually
taken from -- not always "the same side of the pitch" once you're deep into
extra time or a flipped second half).

**Open-play long ball**: a pass carrying the long-ball qualifier, excluding
free-kicks/corners/throw-ins/goal-kicks (the same set-piece exclusions used
throughout `Aggregated/`).

## What genuinely can't be computed from this feed (and isn't faked)

- **Press Bypass Value (PBV)**: needs every opponent's position, not just
  the ones who touched the ball. This is event data, not tracking data.
  Not computed, anywhere, for any season.
- **Possession Value Added / Net Possession Value (PVA, NPV)**: these need a
  full possession-value model (every touch valued, not just shots) --  a
  separate, much larger build this repo doesn't have. Not computed.
- **xG/xT-dependent metrics** (`xgpr20`, `xgca20`, `ncv20`, `xtpr15`): only
  computable for 2025-2026, the one season with a trained xG model
  (`Danger/`) and xT grid (`xT/`) in this repo. Genuinely blank for
  2023-2024 and 2024-2025 -- not zero, not interpolated.
- **"Uncontested" reception** (for excluding from First-Contact Win Rate):
  approximated only as "didn't go straight out of play." A quietly
  uncontested first touch and a genuinely (if unsuccessfully) contested one
  look identical without tracking data.
- **Pressure at the restart**: approximated as an opponent defensive-type
  event within 5s and 15m of the passer beforehand -- applied to open-play
  long balls only. A goal kick is inherently unpressured in the on-ball
  sense (the keeper isn't being tackled), so goal-kick pressure-escape
  metrics (`pesr`/`sed`/`fer`) are not computed at all for that kind.

## Key operational choices (all documented in the module's docstrings, summarised here)

- **First contact** = the next ball-involving event (either team) after the
  restart, within 60s.
- **Established possession** = 3 consecutive same-team touches, or 5
  continuous seconds of one team's touches, matching the brief's definition
  exactly.
- **Territorial checkpoints** at 10s/15s; **field-tilt/TPS pre-post window**
  = 45s each side (the brief allows 30 or 60; this splits the difference).
- **ETG** is computed both ways the brief gives: the simple
  `CTG x survives-10s` version (`etg_basic`) and the event-level `S_i`
  version (`etg_event`) used in RDV.
- **RDV**: computed exactly to the brief's weights and z-score/Φ
  construction for 2025-2026 (the only season with `xtpr15`). For
  2023-2024/2024-2025, `rdv_core` substitutes the same weighted formula
  *excluding* the `xtpr15` term (rescaled so the remaining weights still sum
  correctly) -- report `rdv_core` when comparing across all three seasons,
  `rdv` only within 2025-2026. Per the brief's own caution, these weights
  are the brief's specification, not validated against an outcome here.

## Sanity checks performed before trusting this

Every rate-type metric (bounded [0,1] by construction: FCWR, AFCWR, PFCR,
SBRR, TLR, PESR, PSR, PER, ...) was scanned across all three seasons, both
kinds, all 18 teams -- zero out-of-range values. A first implementation of
Successful Escape Distance (SED) produced a *negative* league-wide average
for 2025-2026, which is impossible by the metric's own definition (only
sequences classified as escapes should count, and an escape can't have
negative distance) -- traced to a bug (a "reached halfway at any transient
point during the sequence" check standing in for "the actual established-
possession location is beyond halfway"), fixed, and re-verified positive
(~22-25m) across all three seasons before use.

## Reading the three-season comparison

A few examples from `three_season_comparison.csv` (2025-26 league averages
in brackets):

- Goal-kick territorial gain has trended down over the three seasons
  (`ctg`: 22.2m -> 21.6m -> 18.6m; `rtg_15` similarly), while the long-kick
  rate itself hasn't moved in a straight line (53% -> 47% -> 51%) -- teams
  aren't just playing goal kicks shorter, the ones they do play (long or
  short) are gaining less ground on average.
- Long-ball pressure-escape rate (PESR) has fallen (33% -> 35% -> 27%),
  alongside a stable first-contact win rate (~47-49%) -- teams are winning
  the first ball off a pressured long ball about as often as before, but
  converting fewer of those into a genuine escape (20m+ gained, or beyond
  halfway) once they do.
- 2025-2026 PSV Eindhoven and Feyenoord Rotterdam lead both goal-kick and
  long-ball RDV -- consistent with them finishing 1st and 2nd -- with
  `rdv_core` spanning from the low 80s down to the low teens across the
  league, a wide spread for what is a fairly specific phase of play.

None of this is causal. It describes what happened, not why -- the brief's
own required next step (controlling for competition, score state, opponent
strength, game location) is a separate, further analysis this doesn't
attempt.

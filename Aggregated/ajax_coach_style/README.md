# Ajax playing style, per coach/manager (2023-2024 -> 2025-2026)

## Why this needed outside sourcing

The Opta event feed carries no manager/coach/lineup/official data anywhere --
`matchDetails` is limited to `periodId`, `matchStatus`, `winner`,
`matchLengthMin/Sec`, `period`, and `scores`. There is no field to read a
coach from. Tenure boundaries below come from public reporting, not the
dataset, and are applied to matches purely by date.

## Coaching regimes used

| Season | Coach | Dates | Matches |
|---|---|---|---|
| 2023-2024 | Maurice Steijn | 2023-08-12 -> 2023-10-22 | 7 |
| 2023-2024 | Hedwiges Maduro (caretaker) | 2023-10-29 | 1 |
| 2023-2024 | John van 't Schip | 2023-11-02 -> 2024-05-19 | 26 |
| 2024-2025 | Francesco Farioli | full season | 34 |
| 2025-2026 | John Heitinga | 2025-08-10 -> 2025-11-01 | 11 |
| 2025-2026 | Fred Grim (interim) | 2025-11-09 -> 2026-03-07 | 15 |
| 2025-2026 | Oscar Garcia | 2026-03-14 -> 2026-05-24 | 10 |

Sourced facts, cross-checked across multiple reports: Steijn sacked 23 Oct
2023 after a 5-0 loss to PSV; Maduro took the single match before van 't
Schip was installed on 30 Oct 2023. Heitinga was sacked 6 Nov 2025 (after 3
Eredivisie defeats and a 3-0 Champions League loss to Galatasaray, 11
league games into the season); Grim ran the side as interim until Oscar
Garcia's appointment on 8 March 2026; Garcia's contract was ended by mutual
agreement on 21 June 2026, after the 2025-2026 season had finished.

**Maduro's single match (n=1) is kept in the table and chart for
completeness, marked with a hatched bar / triangle marker throughout --
it is not a reliable regime average and shouldn't be read as "his style,"
just the one data point that exists.**

## Metrics (all computed directly from `Events/<season>/*Ajax*.json`)

Ajax's `contestantId` (`d0zdg647gvgc95xdtk1vpbkys`) is stable across all
three seasons -- confirmed directly by checking the recurring id across one
Ajax home match per season, not assumed.

- **possession_pct** -- Ajax's share of the two teams' total pass attempts, averaged per match.
- **ppda** -- passes allowed, per defensive action: opponent pass attempts in Ajax's own defensive 60% of the pitch, divided by Ajax's own tackles + interceptions + fouls committed in that same zone (lower = higher-intensity press). Summed across all matches in the regime, not averaged per match, so busier matches carry proportionally more weight.
- **long_ball_pct** / **cross_pct** -- share of all Ajax pass attempts (qualifiers 1 and 2).
- **wing_pct** -- share of Ajax's completed-or-not open-play passes (excludes free kicks/corners/throw-ins) originating from the wide corridor, `y<25` or `y>75`.
- **avg_pass_length_m** -- mean of qualifier 212 (`length`) across all Ajax pass attempts.
- **progressive_passes_per90**, **final_third_entries_per90** -- same distance-to-goal thresholds as `build_season_aggregate.py`'s `is_progressive()` (30y/15y/10y depending on which half start and end fall in), applied to direction-corrected coordinates (see below). Per-90 uses each match's actual `matchLengthMin/Sec`, summed across the regime and divided by 90.
- **territory_index** -- mean direction-corrected x of every Ajax touch (0 = own goal line, 100 = opponent's), i.e. a "field tilt" for Ajax alone regardless of opponent.
- **shots_per90**, **goals_per90** -- typeId 13/14/15/16.

### Attack-direction correction

Opta's raw x/y don't flip when a team switches ends at half-time, so "which
way is forward" has to be inferred per team per half: the average x of that
team's own open-play passes decides it (same heuristic used in
`wing_play_comparison.py` and `diagonal_vs_relationism.py`). This script
does **not** reuse `build_season_aggregate.py`'s progressive-pass logic
directly, because that function compares raw `end_x > x` with no such
per-half correction -- it silently mislabels forward/backward for whichever
team is defending the x=100 end in a given half. The progressive-pass/
final-third thresholds themselves (the metres-gained cutoffs) are reused
unchanged; only the coordinates fed into them are corrected first.

## Reading the results

- Everyone's PPDA sits in a fairly narrow 7.4-9.6 band -- Ajax has never
  been a genuinely low-press side under any of these seven regimes, but
  Heitinga (7.41) and Steijn (7.45) pressed the highest, and van 't Schip
  (9.20) and Farioli (9.04) sat the deepest.
- Possession is highest under Steijn (61.4%) and van 't Schip (59.8%),
  lowest under Garcia (54.6%, excluding Maduro's single match at 46.1%).
- Long-ball share is lowest under Heitinga (9.4%) and Farioli (9.8%),
  highest under Garcia (12.1%) among full regimes.
- Progressive passing peaks under van 't Schip (54.3/90); every other
  regime sits within a tighter 46-48/90 band.

No composite "identity score" is produced -- these are read as a profile
across six metrics, not collapsed into one number.

## Files

- `ajax_coach_style.csv` -- one row per coaching regime.
- `ajax_coach_style.png` -- six-panel small-multiples comparing every regime on the metrics above.
- `ajax_coach_style_scatter.png` -- possession share vs. PPDA, one point per regime.

Source: Opta/StatsPerform 2023-2026 (event data); coaching dates from public
reporting, not the Opta feed.

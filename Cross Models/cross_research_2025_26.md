# Crossing in the 2025/26 Eredivisie

## Volume is not value

Crosses are often assessed with two blunt measures: how many a team attempts and how many reach a teammate. Both are useful, but neither tells us whether the delivery created danger, whether the target was ambitious, or whether the player performed better than the difficulty of the attempt implied.

This study analyses **7,714 open-play crosses from 309 matches** in the 2025/26 Eredivisie dataset, covering 8 August 2025 to 24 May 2026. Set pieces are excluded. Each cross is evaluated using the repository's six cross models, originally trained on Ecuador 2026 data, alongside its observed completion, shot and goal outcomes.

The central conclusion is simple:

> Crossing should be evaluated as a chain of decisions, not as a count of balls played into the box.

A high-volume crossing team is not necessarily a dangerous one. A completed cross is not necessarily a valuable one. Conversely, a failed delivery can still be a sensible attempt if it attacks a high-value area.

## Executive findings

- The league produced **25.0 open-play crosses per match**, or approximately **12.5 per team per match**.
- **20.96%** of crosses were completed, slightly above the model expectation of **19.93%**.
- **12.91%** created a shot and **1.65%** were directly associated with a goal.
- A cross into the six-yard zone was completed less often than one into the main penalty-box zone, **17.1% versus 25.0%**, but produced a goal more often, **2.11% versus 1.33%**.
- Deliveries from the byline had the lowest completion rate, **18.8%**, but the highest average predicted delivery value among the four origin bands.
- Go Ahead Eagles crossed most frequently at **16.8 per match**, while PEC Zwolle attempted only **6.8 per match**.
- PSV converted crossing into goals most effectively at team level: **16 goals from 478 crosses**, or **0.47 cross-assisted goals per match**.
- FC Utrecht were the strongest completion overperformers: **26.6% actual versus 20.6% expected**, a margin of **+6.1 percentage points**.
- Ajax were the clearest underperformers: **16.3% actual versus 20.4% expected**, a margin of **−4.1 percentage points**.
- M. van Bergen generated the greatest total modelled delivery value, but his lead was driven partly by an exceptional **183-cross** workload. S. Dest had the highest average delivery value among the listed high-volume players.

## 1. The league baseline

| Measure | Eredivisie 2025/26 |
|---|---:|
| Matches | 309 |
| Open-play crosses | 7,714 |
| Crosses per match | 24.96 |
| Completion rate | 20.96% |
| Expected completion rate | 19.93% |
| Shot creation rate | 12.91% |
| Goal creation rate | 1.65% |
| Immediate clearance rate | 41.76% |

Roughly one in five crosses reached a teammate, one in eight created a shot, and one in sixty-one was linked directly to a goal. These three rates describe different parts of the same process. Completion reflects access to a receiver; shot creation reflects whether the possession became an attempt; goal creation captures the rare final outcome.

The difference matters because **42.5% of completed crosses did not create a shot**. Completion is therefore an intermediate event, not the attacking objective.

## 2. Where a cross ends matters more than whether it is completed

The destination of the ball changes the trade-off between security and threat.

| End zone | Crosses | Completion | Shot creation | Goal creation | Avg. predicted delivery value |
|---|---:|---:|---:|---:|---:|
| Outside/deep, x < 83 | 338 | 13.9% | 3.0% | 0.6% | 0.009 |
| Main box, x = 83–92 | 3,908 | 25.0% | 14.6% | 1.3% | 0.051 |
| Six-yard zone, x > 92 | 3,468 | 17.1% | 12.0% | 2.1% | 0.057 |

The main-box band is the easiest productive target: it returns the best completion and shot-creation rates. The six-yard zone is different. It is harder to access, but the reward is larger when the ball does arrive. Its goal rate is approximately **58% higher** than that of crosses ending in the main-box band.

This is why raw completion can punish ambition. A player repeatedly attacking the goalkeeper-defender corridor may complete fewer crosses while producing more goal threat per delivery.

## 3. Crossing from the byline is high-risk, high-value

| Origin band | Crosses | Completion | Shot creation | Goal creation | Avg. predicted delivery value |
|---|---:|---:|---:|---:|---:|
| Deep/wide, x < 70 | 270 | 24.1% | 10.4% | 1.1% | 0.043 |
| Advanced, x = 70–85 | 2,971 | 21.8% | 13.2% | 1.6% | 0.048 |
| Final, x = 85–95 | 3,380 | 20.7% | 12.8% | 1.7% | 0.054 |
| Byline, x > 95 | 1,093 | 18.8% | 13.2% | 1.5% | 0.058 |

Completion declines as the origin moves closer to the byline, yet predicted delivery value rises. The pattern is not contradictory. Later deliveries are played into denser defensive areas, but they are also made closer to goal and can force defenders and goalkeepers to defend facing their own net.

Deep crosses show the opposite profile: easier to complete, but less valuable on average. That makes tactical sense. The defence has more time to adjust and the receiver is often further from goal.

## 4. Team styles: volume, efficiency and outcome

### Crossing attack

| Team | Crosses per match | Completion | Expected completion | Shot-creating crosses per match | Cross-created goals per match |
|---|---:|---:|---:|---:|---:|
| Go Ahead Eagles | 16.76 | 18.9% | 19.3% | 1.97 | 0.24 |
| NAC Breda | 15.56 | 21.2% | 20.4% | 2.24 | 0.24 |
| NEC | 15.15 | 19.8% | 19.8% | 1.71 | 0.18 |
| PSV | 14.06 | 23.4% | 21.6% | 2.21 | 0.47 |
| FC Utrecht | 13.14 | 26.6% | 20.6% | 2.28 | 0.28 |
| Feyenoord | 12.74 | 22.4% | 19.4% | 1.85 | 0.35 |
| Ajax | 11.42 | 16.3% | 20.4% | 1.14 | 0.14 |
| AZ | 9.62 | 22.3% | 20.4% | 1.47 | 0.09 |
| PEC Zwolle | 6.76 | 18.7% | 18.2% | 0.71 | 0.12 |

Three distinct profiles emerge:

1. **Volume crossing:** Go Ahead and NAC use crosses frequently, but Go Ahead's completion is close to expectation rather than exceptional.
2. **Efficient crossing:** Utrecht and Feyenoord complete materially more crosses than the model expects.
3. **Outcome crossing:** PSV combine above-expected completion with the league's strongest goal return.

Ajax are the notable negative outlier. Their crossing volume is not unusually high, so the issue is not simply indiscriminate crossing. They underperform expected completion by 4.1 percentage points and generate only 1.14 shots per match from crosses.

### Completion above expectation

Expected completion controls for the modelled difficulty of each delivery. It gives a better description of execution than raw completion alone.

| Rank | Team | Crosses | Actual | Expected | Difference |
|---:|---|---:|---:|---:|---:|
| 1 | FC Utrecht | 473 | 26.6% | 20.6% | +6.1 pp |
| 2 | Fortuna Sittard | 398 | 23.4% | 18.7% | +4.6 pp |
| 3 | FC Twente | 413 | 24.7% | 20.3% | +4.4 pp |
| 4 | Feyenoord | 433 | 22.4% | 19.4% | +3.1 pp |
| 5 | AZ | 327 | 22.3% | 20.4% | +1.9 pp |
| 16 | FC Volendam | 366 | 18.6% | 19.6% | −1.0 pp |
| 17 | Sparta Rotterdam | 460 | 17.8% | 20.1% | −2.2 pp |
| 18 | Ajax | 411 | 16.3% | 20.4% | −4.1 pp |

The interpretation should remain careful. Overperformance may capture delivery skill, receiver quality, movement, tactical spacing or model misspecification. It should not automatically be assigned to the crosser.

## 5. The players creating the greatest crossing threat

The following table uses total predicted delivery value and applies a minimum of 30 crosses. Total value rewards both quality and repeatability.

| Rank | Player | Team | Crosses | Total delivery value | Value per cross | Observed shots created |
|---:|---|---|---:|---:|---:|---:|
| 1 | M. van Bergen | Sparta Rotterdam | 183 | 11.60 | 0.063 | 24 |
| 2 | I. Perišić | PSV | 139 | 8.07 | 0.058 | 19 |
| 3 | Y. Taha | FC Groningen | 122 | 6.50 | 0.053 | 17 |
| 4 | B. Kemper | NAC Breda | 114 | 6.34 | 0.056 | 18 |
| 5 | S. Ouaissa | NEC | 87 | 6.11 | 0.070 | 8 |
| 6 | S. El Karouani | FC Utrecht | 142 | 5.99 | 0.042 | 22 |
| 7 | A. Hadj Moussa | Feyenoord | 89 | 5.79 | 0.065 | 12 |
| 8 | V. Zagaritis | SC Heerenveen | 97 | 4.75 | 0.049 | 13 |
| 9 | B. Önal | NEC | 77 | 4.68 | 0.061 | 12 |
| 10 | G. de Regt | Excelsior | 95 | 4.54 | 0.048 | 17 |

Van Bergen is the leading source of accumulated cross threat because no other player approaches his volume. Perišić combines high volume with strong value. Ouaissa and Hadj Moussa are especially interesting because they generate more value per delivery than several players above them.

For recruitment, total and average value should be used together:

- **Total value** identifies players who can repeatedly carry a crossing workload.
- **Value per cross** identifies selective or high-quality delivery.
- **Completion above expectation** adds an execution layer.
- **Shot and goal outcomes** should be heavily regressed because they depend on receivers and small samples.

## 6. Crossing is also a defensive statistic

The number of crosses conceded says something about whether an opponent can enter and remain in wide advanced zones.

| Team | Opponent crosses faced per match | Opponent cross-created shots per match | Opponent cross-created goals per match |
|---|---:|---:|---:|
| NEC | 8.76 | 1.35 | 0.15 |
| Feyenoord | 8.91 | 1.09 | 0.24 |
| FC Groningen | 10.20 | 1.54 | 0.23 |
| PSV | 10.41 | 1.12 | 0.06 |
| FC Utrecht | 10.50 | 1.14 | 0.17 |
| PEC Zwolle | 15.18 | 1.91 | 0.44 |
| Go Ahead Eagles | 15.59 | 2.65 | 0.29 |
| SC Telstar | 16.24 | 2.06 | 0.21 |

NEC and Feyenoord suppress crossing access best by volume. PSV do not concede the fewest deliveries, but allow only 0.06 cross-created goals per match. PEC concede the largest goal return.

These are descriptive outcomes, not isolated measures of full-back or centre-back quality. Opponent strength, game state, pressing, block height and goalkeeper behaviour all influence the numbers.

## 7. What the models add

The repository contains six cross models:

- completion probability
- chance-creation probability
- goal-contribution probability
- probability of being defended
- continuous delivery value
- multiclass cross outcome

Together, they separate four questions that conventional crossing statistics collapse:

1. **Was the decision sensible?** Delivery value and chance probability assess the location and type of attempt.
2. **Was the execution good?** Actual completion relative to expected completion measures over- or underperformance.
3. **Did the team exploit the delivery?** Shot and goal creation include receiver movement and finishing.
4. **How did the defence respond?** Defended and clearance outcomes describe disruption.

This is a more useful framework than “crosses completed”, particularly for comparing players in different tactical roles.

## 8. Important validation warning

The models were trained on Ecuador 2026 data and applied to the Eredivisie without a documented Eredivisie recalibration stage. That makes this a **cross-competition transfer exercise**.

At league level:

- observed completion: **20.96%**
- predicted completion: **19.93%**
- observed shot creation: **12.91%**
- predicted chance creation: **12.25%**
- observed goal creation: **1.65%**
- predicted goal contribution: **1.32%**

The averages are reasonably close, but average calibration does not guarantee good calibration within teams, zones or player roles. Ranking stability, Brier score, log loss, calibration slope and reliability curves should be tested before the model is used for recruitment decisions.

Two event flags also require investigation:

- `pull_back` is zero for all 7,714 events.
- only six crosses are labelled `fast_break`, and all six are associated with a goal.

Those distributions are implausibly sparse and suggest a qualifier-mapping or event-definition issue. Pull-back and fast-break conclusions should therefore not be published until the raw Opta qualifiers are audited.

## 9. Practical implications

### For coaching

- Do not target a generic increase in crossing volume. Specify the origin, target zone and box occupation required.
- Separate deliveries into deep crosses, conventional wide crosses, byline balls and cutbacks.
- Judge the process using expected chance creation and target-zone access before judging goals.
- When analysing defending, distinguish preventing the cross from defending the box after the cross.

### For recruitment

- Compare players within delivery roles rather than across all crossers.
- Use per-cross value and total value together.
- Adjust for team possession, territorial dominance, game state and the number of available targets.
- Regress observed goal contribution strongly toward expectation.
- Review video of high model-overperformers to determine whether the edge belongs to delivery technique, receiver movement or tactical structure.

### For future research

1. Recalibrate the models on Eredivisie outcomes.
2. Add receiver locations and number of attackers in the box.
3. Separate inswinging, outswinging, driven, chipped and cutback deliveries.
4. Control for match state and defensive block.
5. Compare open-play crossing rates across multiple seasons to test whether crosses are genuinely declining.
6. Measure the possession value after second balls, not only the first receiver or shot.

## Methodology

An event qualifies as an open-play cross when it:

- is an Opta pass event (`typeId = 1`);
- contains cross qualifier `2`;
- does not contain free-kick qualifier `5`, corner qualifier `6`, or throw-in set-piece qualifier `160`;
- occurs in period 1 or 2; and
- has valid start and end coordinates.

Shot and goal creation are linked through qualifier `55`, where the shot references the assisting event ID. An immediate clearance is recorded when the next event belongs to the opponent and is an interception or clearance.

Origin bands use the cross start x-coordinate. Destination bands use the end x-coordinate. Coordinates are Opta's 0–100 system. Team matches are inferred from filenames, and the 309-match sample includes matches after the 306-match regular-season schedule.

## Reproducibility

This report uses:

- `Cross Models/eredivisie_open_play_cross_events_scored.csv`
- `Cross Models/xp_player_leaderboard_eredivisie.csv`
- `Cross Models/xp_team_leaderboard_eredivisie.csv`
- `Cross Models/score_eredivisie_crosses.py`

All percentages in the report are calculated from the event-level scored file. Player totals use a minimum of 30 crosses unless otherwise stated.

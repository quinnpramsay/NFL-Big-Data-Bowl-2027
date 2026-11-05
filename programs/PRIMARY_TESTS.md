# Primary tests

**Status: DRAFT, written November 4, 2026. Confirm or edit before the frozen run (week 5).**

These five tests were drafted from the leads in the November 3 audit (`juice_plus_improvement_plan.md`). They were
written down before the outcome tables in `juice_transfer.py` existed, but after the audit had seen early versions of
P1 and P2. So P1 and P2 are confirmations of earlier leads, not fresh predictions. Say so in the writeup.

Everything else in `transfer_report.txt` is **exploratory**. Label it that way in the text.

| ID | Claim | Statistic | Pass rule |
|---|---|---|---|
| P1 | Tracked agility adds to Game Juice+ beyond the official 40 and RAS-style | Partial correlation, within-group ranks, pooled over groups with an agility score | 95% bootstrap interval above 0 |
| P2 | The new tracked burst (yard 2 to 10 split) predicts NFL get-off for DL and EDGE | Partial correlation with alignment-adjusted get-off, group dummies | Interval above 0. The audit's lead-story rule: about 0.35 or more means lead with pass rush |
| P3 | WR tracked agility transfers to separation at the throw | Mixed-model transfer (beta / player-effect SD), play level, route-adjusted separation | Interval above 0 |
| P4 | A Combine is worth N snaps of top speed for rookies | Smallest k where first-k game data alone beats the Combine alone, out of class | Report k and the Combine's added R² at k = 100 |
| P5 | Combine and game share the same physics: acceleration-speed top gear (v0) | Spearman of Combine v0 (all drills) with game v0, pooled within-group ranks | Interval above 0 |

Rules for the frozen run:

- Change no setting after seeing a primary result. If a setting has to change (a bug), rerun all five and report both.
- One run produces every number in the writeup. The run ID is in every CSV and in `run_manifest.json`.
- Single-drill scans, blend weights and new score versions are out of scope (audit, "Stop doing").

## Setting change log

- **November 5, 2026: burst redefined on both sides** (before the frozen run). Combine E = yard 2-10 split averaged with
  mean acceleration over yards 2-10. Game burst = the same two numbers, measured clock-free from the snap spot (the old
  game burst was mostly reaction time). Old behavior: `BURST_METHOD=split GAME_BURST=snap`. All five tests rerun:

  | Test | Before (run 20261105T005112Z-69e030) | After (run 20261105T161250Z-641f77) |
  |---|---|---|
  | P1 | +0.169 [+0.036, +0.294], supported | +0.173 [+0.019, +0.304], supported |
  | P2 | +0.168 [-0.086, +0.416], not shown | +0.144 [-0.093, +0.383], not shown |
  | P3 | -0.054 [-0.321, +0.213], not shown | unchanged (does not use burst) |
  | P4 | k* = 3 | k* = 3 |
  | P5 | +0.224 [+0.128, +0.316], supported | unchanged (does not use burst) |
- **November 5, 2026: robustness fixes** (run 20261105T163235Z-ffa63e). Game agility per play uses the second-highest frame (one noisy
  frame can set a max); E shrinkage uses the retest reliability of the E piece itself (0.67), not the mean of its parts
  (0.57); the held-out formula uses the same plain least-squares size model as the main score (it still used ridge).
  P1 +0.172 [+0.030, +0.296] supported; P2 +0.147 [-0.091, +0.383] not shown; P3, P4 (k* = 3), P5 unchanged.

## Secondary checks (written November 5, 2026, before `juice_casework.py` was run)

Follow-ups on the two leads that survived: WR/DB agility (P1) and the OL burst chain (exploratory, chain table).
These were written after seeing the point estimates in `transfer_report.txt` (sections A, B, D), so they are
checks of known leads, not fresh predictions. They do not change P1-P5.

| ID | Claim | Statistic | Pass rule |
|---|---|---|---|
| S1 | WR + DB: Combine agility predicts the same trait in games, beyond the 40 | Partial Spearman, Combine A vs game agility, holding the 40, within-group ranks + group dummy | 95% bootstrap interval above 0, and A minus the 40 (each holding the other) above 0 |
| S2 | S1 is not a selection artifact | S1 re-estimated (a) weighted by inverse probability of reaching 50 plays, (b) on mean game agility at floors 10, 25, 50, 100, 200 plays | (a) interval above 0; (b) every point estimate above 0 |
| S3 | OL: game burst links to lower peak pressure for reasons other than play mix | Link 2 with the outcome rebuilt from dropbacks that are NOT burst plays, holding the player's screen share and number of burst plays | 95% bootstrap interval above 0 |
| S4 | OL: the chain carries through | Indirect effect a x b (Combine E -> game burst -> peak pressure, b holding E), percentile bootstrap | Interval above 0 |

Late-round case studies are descriptive (no pass rule). Everything else in `casework_report.txt` is exploratory.

## Exploratory rescue of the OL chain (written November 5, 2026, after S3/S4 failed)

S3 and S4 failed because the OL game burst score (95th percentile) rises with the number of burst plays. Two
alternative measures, chosen AFTER seeing that result, so anything they show is exploratory, not a pass of S3/S4:
(a) mean game burst per player (floors 10 and 20 plays), (b) mixed-model player effect (random intercept, every
player with 1+ burst play, shrunk toward the group mean in proportion to his play count). Both size-free within OL.
"Improves" means: link 2 on non-burst dropbacks, holding screen share and number of burst plays, has a 95%
interval above 0 AND link 1 (Combine burst -> game burst, holding the 40) has a 95% interval above 0.
If neither measure does both, OL stays a side note in the writeup.

## Route-running link and 2026 forward test (written November 5, 2026, before `juice_routes.py` or `juice_forward.py` existed)

**Setting change (bug fix, before the frozen run): game agility for WR, TE and DB pass plays stops at the throw.**
`GAME_AGIL_WINDOW=prethrow` (default) measures the per-play agility frame only between the snap and `pass_forward`
for WR, TE and DB on plays with a forward pass (1.0 s or more of pre-throw frames; shorter plays have no agility value).
Run plays, sacks, scrambles and the other positions keep every frame after the snap. Old behavior:
`GAME_AGIL_WINDOW=all`. Per the rules above, P1-P5 and S1-S4 are all rerun and both versions reported.

The route-running checks below were written AFTER exploratory scratchpad versions of them had been run on run
ffa63e (WR cut speed vs separation +0.34, chain about +0.03 yd per Combine SD). They are checks of known leads,
not fresh predictions. Everything else in `routes_report.txt` is exploratory.

| ID | Claim | Statistic | Pass rule |
|---|---|---|---|
| R1 | WR separation at the throw is a stable player trait | Split-half (random halves of route plays, Spearman-Brown) of route-adjusted separation, WRs with 50+ routes | 0.5 or more |
| R2 | WR pre-throw cut speed buys separation | (a) player level: Spearman of mean cut speed with mean separation, WRs with 50+ routes; (b) play level within player: yards of separation per SD of cut speed, cluster bootstrap over players | Both 95% intervals above 0 |
| R3 | The Combine-to-separation chain carries through for WR | Indirect effect a x b in yards: a = Combine piece -> game piece (SD per SD), b = game piece -> separation (yd per SD), joint percentile bootstrap over players. Two chains: 40-yard dash -> pre-throw top speed, Combine drill cut speed -> game cut speed | Interval above 0 for each chain, reported in yards and EPA per 100 targets |
| R4 | Pre-throw game agility links to separation better than the all-frames version did | Spearman of player game agility (Game Juice+ agility piece, this run) with mean separation, WRs with 50+ routes, compared with the same statistic on run ffa63e (+0.139 from `juice_routes.py`; the scratchpad version gave +0.13; comparator corrected before the new run's value was seen) | Point estimate higher than +0.139 (descriptive, no interval rule) |

**F1. 2026 forward test (outcome not yet seen).** Predictions are frozen by `juice_forward.py --freeze`, which writes
`forward_predictions_2026.csv` with a timestamp and refuses to overwrite it. Predictors use 2023-25 data only:
(a) route-adjusted separation 2023-25 (baseline), (b) pre-throw movement score (mean of z cut speed and z pre-throw top
speed, route-adjusted), (c) Game Juice+, (d) Combine Juice+. Outcome: 2026 regular-season Next Gen Stats separation
over expected for WRs in the cohort with 30+ 2026 targets, where expected = linear fit of `avg_separation` on
`avg_intended_air_yards` and `avg_cushion` over every 2026 NGS receiver (season-to-date row, week 0). Scored by
`juice_forward.py --score` on the latest nflverse release before January 1, 2027.
Pass rule: Spearman of (b) with the outcome has a 95% interval above 0. Also reported, no pass rule: (a), (c), (d),
and (b) holding (a).

## Setting change log (continued)

- **November 5, 2026: bug-fix pass** (verification run 20261105T192954Z-947468, not yet written to `juice_nb_output`).
  Changes: game agility for WR/TE/DB pass plays stops at the throw (`GAME_AGIL_WINDOW=prethrow`); `size_free` leaves
  position groups under 10 players blank; each report section has its own random seed (`section_rng`), so every
  bootstrap interval moved slightly even where the estimate did not.

  | Test | Before (run ffa63e) | After (run 947468) |
  |---|---|---|
  | P1 | +0.172 [+0.030, +0.296], supported | +0.147 [+0.015, +0.274], supported |
  | P2 | +0.147 [-0.091, +0.383], not shown | +0.147 [-0.112, +0.378], not shown (new seed only) |
  | P3 | -0.054 [-0.321, +0.213], not shown | unchanged |
  | P4 | k* = 10, n = 169 | k* = 10, n = 163 (6 players in groups too small to size-adjust left out) |
  | P5 | +0.224 [+0.128, +0.316], supported | unchanged |
  | S1 | +0.328 [+0.147, +0.479]; A minus 40 +0.187 [-0.092, +0.429]: does not pass | +0.274 [+0.092, +0.444]; A minus 40 +0.257 [-0.021, +0.512]: does not pass |
  | S2 | passes | passes (+0.253 [+0.029, +0.445]; floors +0.24 to +0.28) |
  | S3 | +0.212 [-0.003, +0.412], does not pass | +0.212 [+0.004, +0.414], passes |
  | S4 | +0.086 [-0.000, +0.209], does not pass | +0.086 [+0.004, +0.199], passes |
  | Year-2 agility (exploratory) | +0.256 [+0.103, +0.389] | +0.243 [+0.053, +0.418] |
  | R4 | (new) | +0.114 vs +0.139: not shown. Pre-throw agility does not link to separation better at the Game Juice+ level |

  S3 and S4 have the same point estimates as before. Only the new seed moved their lower bounds from just below 0 to
  just above it, so they sit on the boundary under either seed. Treat the OL chain as unchanged (borderline), not
  as newly confirmed.

- **November 5, 2026: pre-throw game agility reverted.** After run 947468, `GAME_AGIL_WINDOW` defaults back to `all`
  (every frame after the snap), the setting behind P1-P5 and S1-S4 since run ffa63e. Reason: R4 was not shown
  (+0.114 vs +0.139) and P1 weakened (+0.147 vs +0.172). The option stays in `juice.py` as `GAME_AGIL_WINDOW=prethrow`.
  R4 is not scored on runs without the pre-throw window. The other fixes in the bug-fix pass stay.

- **November 5, 2026: R2's within-player link is explained by cushion (exploratory, found after R2 was scored).**
  Run 20261105T200039Z-a2540c, same WR plays as `juice_routes.py` section C (68 WRs with 50+ routes). Found while
  testing a model of expected movement from play context (gradient boosting, folds by player). That test and its
  scripts were on a branch that has since been deleted, so these numbers cannot be rerun from this repo.
  Within player, yards of separation per SD of cut speed (cluster bootstrap over players):

  | Adjustment of cut speed | Outcome: route-adjusted separation | Outcome: separation over expected |
  |---|---|---|
  | Cells: route x man/zone x time to throw (R2 as scored) | +0.097 [+0.072, +0.120] | +0.004 [-0.019, +0.030] |
  | Cell features plus cushion and lined-up position | +0.010 [-0.019, +0.036] | +0.005 [-0.021, +0.028] |
  | Full play context (14 features) | -0.003 [-0.029, +0.023] | +0.004 [-0.023, +0.029] |
  | Full play context minus cushion | +0.063 [+0.037, +0.087] | -0.012 [-0.035, +0.012] |

  Dropping any other single feature leaves the link at about 0. Cushion is the one that matters: when the defender
  plays off, the same receiver both cuts faster and is more open at the throw, so the play-by-play link measures the
  coverage, not the receiver. R2 (b) stays scored as passed, but it should not be described as cut speed buying
  separation play by play.
  The player-level link survives, smaller: Spearman of player mean cut speed with separation is +0.33 with the cell
  adjustment and +0.13 to +0.21 with full context (any one feature left out); the two-piece movement score (cut speed
  and top speed) is +0.22 [-0.01, +0.44] with full context vs +0.34 with cells. Cushion is partly the defense's response to a
  receiver's speed, so adjusting player means for it may remove some real skill; the within-player test is the
  cleaner one. Claims about cut speed and separation are player-level only. R3 is unaffected (its b link is
  estimated on player means).

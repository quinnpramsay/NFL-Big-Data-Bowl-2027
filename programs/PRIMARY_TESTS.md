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

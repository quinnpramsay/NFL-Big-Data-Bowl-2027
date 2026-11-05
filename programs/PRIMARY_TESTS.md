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

# R18-X verdict: "why does it never descend" (2026-09-07)

The question: why does the model never go downstairs, and is it really just a lack of training? This file answers only that question.
Zero training and zero protocol changes; two **diagnostic replays** (not exams) run on the consumed paired pool 2_133 (2133000–2133047),
worker 7e31dc54, coach-v03, sustain-loot-v1, completion-l2-r18c, retreat-v1, hunting fully on, sampled decoding seeded per game,
i.e. the retreat control arm of R18-C/G/F. Driver `r10-staging/run_r18x_l2geom.py` (a new file) and outputs `r10-staging/r18x-l2geom/` (not published);
summary scripts `geom_agg.py` and `geom_agg2.py` (local, not published).
Replay identity: the micro_steps / deaths / kills / clvl of the 48 games are bit-identical to `r18f-probe/r18f-retreat.json` (only 6 games differ in the
"depth" column, because the replay records the level at game end and the probe the deepest level; `source_changed_during_run` is empty).

## 1. Per-game telemetry of the ten earlier arms (`depth_diag.py`, local, not published)

Taking the retreat control arm (48 games) as the example; the other arms have the same structure:

| Quantity | Value |
|---|---|
| Level-2 arrivals / level-3 arrivals | 37 / 2 |
| 90 level-2 stays: retreat 57, death 25, descent 4, clock 4 | median 417 ticks and 5 kills per stay; median level-2 roster 140, median 128 left on leaving |
| 121 level-2 DIVE windows | **opened on the arrival tick** (median delay of the first window 0); 59% of the level-2 time; endings: retreat trigger 61, cap 30, stall 22, descent 2, death 2 |
| The coach's DIVE triggers on level 2 | rule-ready 47, rule-ready but depth ruler <1 (const_dive) 74 |
| 452 level-1 DIVE windows | 37 descents (after clearing down to 6 monsters); the rest fuse/cap/stall |
| 25 level-2 deaths | median 352 ticks on the level; mostly clvl 3–5 and AC 7–9 at death; 20 died during a retreat |

**Conclusion 1: the manager is not the bottleneck.** The coach opens a DIVE window as soon as the character reaches level 2, and 60% of the level-2 time is spent in DIVE windows.

## 2. Stairs-geometry replay (natural policy, 48 games, 36 first arrivals on level 2)

| Quantity | Value |
|---|---|
| Chebyshev distance to the down stairs on reaching level 2 | median 22 (quartiles 12 / 36; within 5 tiles in 5 games, 40+ tiles in 6 games) |
| Distance between the up and down stairs | median 22.5 |
| Monsters inside the corridor box / within 8 tiles of the stairs | median 3 / 0 |
| A monster-avoiding BFS sampled every 10 decisions (1333 samples) | **1299 (97%) have a path to the stairs that touches no monster**; only 9 trapped |
| 87 level-2 stays | 26 got within 5 tiles of the stairs, 13 within 2 tiles; 4 descents (2 games) |
| 14 stays that arrived within 6 tiles of the stairs | 2 descents, 3 deaths, 8 retreats |
| 8236 worker decisions in level-2 DIVE windows | **a11 (the descend macro) was available every time but pressed only 192 times (2.3%)**; a9 attack 2225, a10 hunt 1183, direction keys 1–8 4396 in total |
| Of these, the 695 decisions within 5 tiles of the stairs | a11 available 695 times, pressed 20 |

**Conclusion 2: the stairs are reachable; the worker will not go.** Not a distant map, not blocking monsters, not the mask.

## 3. Counterfactual: forcing a11 in DIVE windows (the same 48 games)

| | Natural | Forced a11 |
|---|---|---|
| Deepest-level histogram | L1 11 / L2 35 / L3 2 | L1 11 / **L2 16 / L3 20 / L4 1** |
| Level-3 arrivals | 2 / 36 | **21 / 36** |
| First arrival on level 2 → down to level 3 | median 4794 ticks | **median 108 ticks** |
| Survival | 15 / 48 | **5 / 48** |
| Survival of level-3 arrivals | 1 / 2 | **0 / 21** |
| Death level | L1 7 / L2 25 / L3 1 | L1 7 / L2 23 / **L3 12 / L4 1** |
| Total level-2 kills / games ending at clvl 4+ | 888 / 21 games | 213 / 12 games |
| Per-game pairing | — | depth +19 / −0; survival −10 / +0 |

**Conclusion 3: depth is only one action away, but a level-3 warrior with AC 9 dies on level 3.**
The natural policy's "reckless fighting" on level 2 at least buys 21 games reaching level 4+ and 15 survivors; forcing the descent throws all of that away.

## 4. Why it will not press a11 (chain of evidence)

1. The worker's wage (the env.py v4 reward table): +1.0 per kill, +0.01 per XP point; the descend bonus **goes to the manager** (the wage-stripping rule,
   `worker_descend_bonus_fraction` defaults to 0), and the worker only has the "readiness escrow", which vests after a ready descent if it does not die in the next window.
2. R16 training (the last run of worker 7e31dc54, 449 games, 220 reaching level 2 and 12 reaching level 3): **244 descent escrows, 0 vested**
   (R16 verdict §4; finding 2 of the R17.0 verdict: still 0 after switching to the v0.2 ruler, because AC 9 needs dropped armor and 100 gold cannot buy it).
3. So what it learned is "DIVE window = keep fighting"; it goes down from level 1 only because there is nothing left to fight after clearing (452 level-1 DIVE windows gave 37 descents,
   with on average only 6 monsters left on the level at the time). Level 2 always has something to fight, so it never goes down.

## 5. Answer

**It is not a matter of training volume.** Across a lineage of four million steps this action never earned anything, and another ten million steps would change nothing. The real two walls:

- **Wall 1 (pay)**: descending was never paid. The training arm of this round is the **first** world in the lineage where escrow can vest: coach-v03's six rules judge the character ready
  most of the time on level 2 (all 121 of 121 DIVE windows were opened voluntarily by the coach), the loot-selling economy makes AC 9+ affordable, and escrow vests if the next window ends without death.
- **Wall 2 (strength)**: the snowball on level 2 does not roll: only 25 kills per 9000-tick game (roster 140), 60% of the time spent on retreat round trips, hazard 0.37 per 1000 ticks.
  This is the "positioning/aggro/target selection" lesson raised in the design review, and it can only be trained. The zero-training interfaces (retreat, portal, whole-map hunting off) push the hazard down to 0.23–0.27,
  but they do not change "cannot win".

This also answers "what to train": learning windows only on level 2 (earned-dive-suffix), escrow with a vesting path in the reward for the first time, together with the three design decisions
(loot selling in training, whole-map hunting off on level 2, carrying scrolls) that turn level 2 into a world where the snowball can roll. If after training the worker learns "snowball first, then descend",
level-3 arrivals and survival will move together; if it only learns to descend, the counterfactual repeats itself (depth up, survival down); the four gates and the paired re-test can tell these outcomes apart.

## 6. Seeds and files

Replays on the consumed pool 2_133; the virgin pools are untouched; no frozen artifact changed; new files: this verdict, and (not published) `run_r18x_l2geom.py`, `r18x-l2geom/`,
`geom_agg.py`, `geom_agg2.py`, `depth_diag.py`.

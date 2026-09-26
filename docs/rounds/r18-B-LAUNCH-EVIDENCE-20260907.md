# R18-B launch evidence (draft, 2026-09-07)

## 1. Evidence from the zero-training probes (same pool 2_133, same worker 7e31dc54, same coach coach-v03)

| Gate | Result |
|---|---|
| T0'' (retreat, 1800-tick window) | PASS: survivors 13 -> 31, L2 deaths 27 -> 10 |
| R18-C (retreat, 9000-tick window) | The loop turns (25 games went back down; second retreats succeed 84% of the time); hazard 0.72 -> 0.37; **no progress** (depth distribution and clvl unchanged) |
| R18-D/E (aggro cap, target choice) | **Null result**: hazard, survival and pull count unchanged; target choice also lowered kill efficiency |
| R18-G (level 2 closed, full-map hunt) | **Effective but not significant**: hazard 0.37 -> 0.23 (-38%), survivors 15 -> 19 (net +4, UCB95 +0.035), monsters near the character at trigger 6.1 -> 5.5, survivors staying on level 2 4 -> 9; kill efficiency 13 -> 9; depth unchanged |
| R18-F (town portal scroll) | **Effective but not significant**: hazard 0.37 -> 0.27 (-28%), survivors 15 -> 17 (net +2, UCB95 +0.056), L2+ occupancy after returning 0.29 -> 0.46, 21 town trips and 21 returns; the 200-gold scroll cut end-of-game gold from 216 to 71; depth unchanged (L3 4 vs 2) |

Common conclusion of the zero-training probes: **the interface is there, but the frozen worker does not learn to
use it by itself**. Retreat is the only word that works without learning, because it is a scripted hand. The
other words (hold, target choice, gate) are either ineffective or need training. So the next step is training:
let the worker learn in a world that has retreat (and perhaps the gate).

## 2. Two small gates before launch (P0 and P1, about 30 minutes each)

- **P0, retreat without selling gear (done)**: sustain-v6 + retreat vs sustain-v6, r18c clock: survivors 7 vs 4
  (3 saved, 0 lost, UCB95 -0.004); L2 hazard 0.537 -> 0.347 (-35%); 84 retreats, 64 arrivals, 20 deaths on the
  way. **The hazard effect does not depend on selling gear, but the survival gain does**: with gear selling,
  retreat saves a net 10 games (R18-C); without it, only 3. A character that retreats upstairs needs gold to
  recover.
  Conclusion: training under the R21 contract (training allows only sustain-v6) means training in a harsher world
  than any of these probes. Proposal: revise the R21 contract so that sustain-loot-v1 enters training (R18-B3: the
  clock check in train_ppo and the clock literal of the sustain-loot-v1 recipe in eval_contract need a new
  version), so that the training world equals the measured world. This is the first decision needed.
- **P1, final byte certification**: rebuild the bridge after the R18-F integration -> full test suite -> probe
  regression -> two-way re-bake 4/4 + 4/4 -> recast the control group of the six-exam set.

## 3. Proposed arm (R18-B training arm "retreat-arm-a")

| Item | Value | Basis |
|---|---|---|
| Start | worker 7e31dc54 (earned-dive-suffix-v1) | current worker |
| Environment | l2-town-v1 / full / **sustain-v6** / coach-v03 / **resource_retreat=retreat-v1** | R18-B wiring; P0 decides whether loot changes |
| Clock | **completion-l2-r18c** (arrival 12000 / observation 6000 / 9000 after the first descent) | R18-B2; the 1800 ticks of v1 would be used up by retreats |
| Other words | aggro_cap=off, engagement_priority=off, hunt_scope per the R18-G result, resource_portal per the R18-F result | words that failed without training stay out of arm v1 |
| Escrow rule | windows triggered by a retreat pay no escrow (R18-B) | prevents "bleed to 50% to collect escrow" |
| Steps / parallelism | same as the R17.1 draft (to fill in: environment count, total steps, evaluation interval) | |
| Gates | the four gates A-D as before + six-exam control + T0'' same-pool paired retest (trained worker vs 7e31dc54, same interface) | standing protocol |

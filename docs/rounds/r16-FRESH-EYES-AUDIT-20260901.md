# AlphaDiablo: synthesis of seven audits (2026-09-01)

## 1. Deduplicated and merged results

The seven reports raised 38 candidate causes in total, merged into 12 independent cause clusters (C1-C12).
Merging rule: the same mechanism, the same code location or the same data source form one cluster; items
that differ only in "lens" are not split.

| Cluster | Merged from | In one sentence |
|---|---|---|
| C1 the L2 gate of the readiness table is impossible by specification | audit 1-3/4, audit 2-1, audit 4-1, audit 6-1 | HP90 ⇔ clvl4 ⇔ 8040 XP > the XP pool of a full L1 clear; AC15 needs a +8 drop lottery |
| C2 the clear-the-level escape hatch is dead code | audit 3-5, audit 4-2 | raw['monsters'] contains only live monsters, alive/total≡1, and 4 golem slots always have hp>0 |
| C3 argmax in exams / deployment is disconnected from training sampling | audit 3-3/4, audit 5-6, audit 7-1, audit 3 (clock inheritance) | wall-bump loops, fake 140 exhaustion, DIVE limit cycles; kills halved, DIVE success 60%→10% |
| C4 measurement artefact from death stripping gear | audit 1 (exclusion item), audit 2-2, audit 4-4, audit 5-2, audit 6-2, audit 7-2 | the death rows' AC=4/dmg=1/gold=50 are post-mortem values; "zero gear growth / gold with nowhere to spend" is distorted |
| C5 frontier saturation of the a10 exploration macro at radius 12 | audit 1-1, audit 4-3, audit 6-5; **conflicts with an exclusion item of audit 3** | local BFS finds no candidate → wait → fake exhaustion; the reachable roster is only 40-65% |
| C6 the clock trio: 3000-step episode / FARM_SCENE_CAP=1800 / a definition of progress that is too cheap | audit 1-2, audit 3-1/2, audit 6-4 | 3000 steps physically cap at ≈clvl2; the cap chases the worker downstairs at 31% cleared; pacing counts as "progress" |
| C7 HP economy priced at zero + a12 mask + no prior for a13 + wrong sign of the depth multiplier | audit 5-1/5, audit 7-5, audit 4 (counter-evidence of an exclusion item) | 100% of deaths at belt=0 with potions on the floor not picked up; on L2 each kill costs 2.4-5× the HP, while the wage is only +25% |
| C8 end-of-episode accounting runs backwards: timeout≡death, anti-idling at half pay, the in-leg death rate rises monotonically | audit 3-7, audit 5-3/4, audit 7-3 | zero survival premium; trajectories that grind levels on L1 are punished |
| C9 the gear channel: drop rates, durability, no shops / no gold pick-up, E1 price ratio | audit 2-3/4/5, audit 4-5, audit 6-3 | the AC dimension is a lottery and transient; gold is always 100 and never picked up |
| C10 approach shaping is below the noise floor | audit 7-4 | a1-a8 ≈ uniform noise, movement never learned; the root of C3's wall-bump loops |
| C11 statistical diseases of the protocol: death censoring, 37 lethal seeds, the gate B anchor works against survival | audit 6-6/7/8 | end-of-episode means are censored quantities |
| C12 an explosive barrel seals the spawn pocket | audit 3-6 | 1/128 seeds, certain death |

---

## 2. Top 8 ranking (strength of evidence × impact if true × cost of verification)

### 1 · C1: the first rung of the readiness table is unreachable by construction
- **Strongest evidence**: at `worker_env.py:148-154` the L2 row = (clvl3, HP90, AC15, dmg8), taking the min of
  four ratios; warrior HP = 70 + 8×(clvl−1) (attributes.tsv adjLife18/lvlLife2/chrLife2 +
  `diablogym.cpp:2069-2079` 3 vitality : 2 strength), measured clvl3→86, clvl4→94, so **HP≥90 ⇔ clvl≥4 ⇔
  8040 XP**; while the XP pool of a full L1 clear (including the level-difference discount at
  `player.cpp:2437`) measured 5875-9124 in four independent runs, and on half the seeds even clearing the
  whole level does not reach clvl4. AC: 7 at spawn (dex20/5 + Buckler 3); at mlvl≤2 the only armour drops
  are Cap/Cape/Rags/Cloak/Buckler, at an armour drop rate of ≈1.65% per kill. Four audits reached the same
  conclusion independently: `readiness_power_ratio(raw,2)` is still 0.467 after injecting 8040 XP (AC is the
  weak link). Also: by the table's own definition a freshly spawned character scores only 0.58 on L1; the
  table itself is mis-scaled.
- **Impact**: decisive. Escrow refusing everything (vested=0/denied=15459) is a constant, with zero
  correlation to worker ability; the criterion behind R15's "growth engine" root cause is invalid.
- **Discriminating probe (≤5 minutes)**: `python -c 'from diablogym.worker_env import readiness_power_ratio
  as f; print(f({"char_level":3,"max_hp":86,"armor_class":15,"item_max_damage":8,"damage_mod":0},2))'` →
  expected 0.956<1; with max_hp=94 → 1.0. Together with the two rows in farm12000.json where 122-128 kills
  give <8040 XP, this closes the case.
- **Direction of the fix**: work the thresholds back from engine arithmetic: lower the L2 row to values such
  as (clvl2, HP78, AC9, dmg6) that are reachable within 3000 steps, or switch to a "share of available XP +
  survival" gate; remove the redundancy between the HP and clvl columns (from L4 on HP systematically
  dominates clvl).

### 2 · C2: the clear-the-level escape hatch is dead code, so the exhaustion mask is the only way down
- **Strongest evidence**: the comment at `worker_env.py:133-144` assumes dead monsters stay with hp≤0, but
  measured: reset with 107 monsters → probe_kill of 90 → len=17, alive=17, dead_listed=0
  (`diablogym.cpp:1362-1370` continues past `hasNoLife()` at export); in addition, 4 golem placeholder slots
  (mlvl12 / xp0 / invisible) always have hp>0. Across the two regimes, 408+182 window ends had
  alive==roster without exception; 225 DIVE windows had cleared=0 and forced_dive=100%.
- **Impact**: high. The "25% escape hatch" of readiness-v2 never opened, and "free-riding the exhaustion
  flag" was not opportunism but the only legal way down; any curriculum that uses the raw roster share as a
  gate or a wage condition will fail silently.
- **Discriminating probe (2 minutes)**: reset → probe_kill_monster one by one → step(20) → print
  `len(monsters)`, `sum(hp>0)`, `readiness_floor_cleared(raw)`.
- **Direction of the fix**: cleared share = `monster_kill_total / (monster_kill_total + alive − golem)`, with
  the base recorded at the start of the episode; switch the manager observation to the same definition.

### 3 · C3: argmax in exams / deployment is disconnected from the training sampling regime
- **Strongest evidence**: `eval_assembled.py:1600` and `probe_r15_deployment.py` both use
  `deterministic=True`; training entropy 0.96-1.07 nat, top-1 only 0.51-0.65. Same zip, same seeds: argmax
  kills 23.6-30.5 vs sampled 39.3-51.0 (×1.6-2.2), DIVE descend 10% vs 60% (the latter matches the training
  r13_dive_audit 636/1083), a2/a4/a8 wall bumps take 45-50% of decisions with no movement in 73-91% of
  them, fuse window closures 52% vs 0.9%; 13/32 seeds burn ≥500 steps on L1 (26% of the total budget).
  Side mechanism: after a DIVE stall, layer_clock is not reset in `_win_begin` (`options_env.py:750-801`),
  so the next FARM window is exhausted again after a single decision (16/49).
- **Impact**: high (at the evaluation layer). Half of the readings "3000-step growth ceiling clvl
  1.5-1.7" and "cannot dive" are decoding artefacts; three audits agree; small effect on training itself
  (2.6%).
- **Discriminating probe (≤10 minutes)**: copy probe_r15_deployment.py with `deterministic=False` (np
  seeded, `torch.set_num_threads(1)`), compare kills/clvl/depth/the spectrum of DIVE reasons on 32 paired
  seeds.
- **Direction of the fix**: exams in the same regime as training (sampling or low temperature); reset
  layer_clock in `_win_begin`; make the fuse recovery action "switch action" instead of repeating the same
  macro.

### 4 · C4: the death-stripping artefact contaminates the "decisive data"
- **Strongest evidence**: `player.cpp:2691-2694`: in single player a death to a monster sets
  dropItems=true → InvBody drops item by item as DeadItem, DropHalfPlayersGold. In the four deploy JSONs:
  died ⇔ (AC=4 ∧ dmg=1 ∧ gold=50), 315/315 without exception; survivors average AC 8.6-8.8 (30-43% at ≥9,
  highest 17), dmg 6.3-7.0, gold always 100; the long-horizon 8 episodes' AC of 5.9 = a mix of 4 corpses at
  AC4 and survivors at 7/7/7/10. Six audits agree.
- **Impact**: decisive at the diagnostic level. About half of "almost zero gear growth" is an artefact;
  "60-80 gold with nowhere to spend" is actually gold that was never picked up (there is no pick-up-gold
  action). The readiness gate reads live raw and is unaffected.
- **Discriminating probe (1 minute)**: recompute AC/hit_damage/gold in r15-deploy3-*.json stratified by
  died; or change the probe to record the step before dead flips.
- **Direction of the fix**: sample every end-of-episode metric at the step before death; report stratified
  by died + normalised per thousand steps.

### 5 · C5: frontier saturation of the a10 exploration macro at radius 12 → fake exhaustion → forced descent to death
- **Strongest evidence**: `env.py:3232-3276, 3364-3369`: BFS is limited to a 25×25 window, candidates must be
  ≥5 cells away and not within ±1 of the footprint, and no candidate returns None → wait. Three independent
  reproductions with a greedy script: 11 of 16 seeds stall (8-79% of monsters left); seed 2114000 stuck at
  step 2487 with 61 live monsters, 58 reachable by whole-map BFS, the nearest closed door 59 steps away;
  kills at 12000 steps = kills at 3000 steps (42=42). Positive control with a global BFS fallback: 4 of 6
  stalled seeds rise to 82-101 kills and clvl3. Under OptionsEnv, w4 FARM closed as exhausted with 63% of
  monsters alive, and w5 was forced into DIVE and died at hp 15/78.
- **⚠ Conflict to resolve**: audit 3, driving with the trained worker, measured that when exhaustion
  triggered the a10 planner had a command 38/39 times, with a median distance of 2 cells to the nearest
  frontier, and concluded that "vision starvation does not hold". Both can be true at once: a greedy script
  pressing a10 continuously uses up the in-window candidates and saturates; the trained worker does not
  press a10 under argmax (only 12%) and resets the clock by pacing under sampling (C6: progress is too
  cheap). **Common ground: the exhaustion signal is unrelated to the real cleared share.**
- **Impact**: in affected episodes (about 1/4 to 2/3 of seeds, depending on the regime) farm income is
  halved, and it directly triggers unpaid dives and the wave of deaths on L2.
- **Discriminating probe (≤10 minutes)**: for the trained worker (sampling regime) on 16 seeds, record the
  rate at which `_plan_explore_step` returns None, the number of live monsters at that time and the number
  of live monsters reachable by whole-map BFS; also record the median Chebyshev distance from each "new
  cell" to the existing footprint (≤2 means "progress = pacing").
- **Direction of the fix**: when the window has no candidate, fall back to a radius-112 global BFS and take
  a waypoint on the window edge toward the nearest unvisited cell / live-monster area (~40 lines of
  Python); base the exhaustion criterion on kills / damage / distant new cells, and do not count new cells
  next to the footprint.

### 6 · C6: the clock trio: 3000-step episode / FARM_SCENE_CAP=1800 / the definition of progress
- **Strongest evidence**: L1 averages 65-73 XP per kill, and the best greedy rate is 15-18 kills per
  thousand steps → a 3000-step ceiling of ≈41-58 kills ≈2600-3500 XP, exactly between clvl2 and clvl3
  (4620); clvl3 first appears at ≈6000 steps; the worker's survivors at 42-45 kills already reach 85% of the
  greedy ceiling. Under sampling, 23 of 27 exhaustions are the cap type, with a median cleared share of
  0.31 at trigger time and 70-100 live monsters still around; the worker then descends after a median of
  193 steps, and 24/32 die. Counterfactual cap=6000: deaths 24→8, but kills/clvl unchanged (39.3→41.5,
  clvl3 3/32→3/32). A 30000-step long horizon is still chased off L1 after 1800 steps → "no growth with ten
  times the budget" is invalid as a finding.
- **Impact**: high. Any "farm first, then dive" curriculum is mathematically impossible within 3000 steps;
  the cap sets the share of "descend and die"; lifting the cap alone produces no growth (it has to go
  together with C5/C7).
- **Discriminating probe (done + 15 minutes)**: the three-column comparison with AUDIT_FARM_SCENE_CAP=6000
  has been run (outputs not published); add 8 seeds at 30000 steps with the same override and see whether
  clvl still stops at 2.
- **Direction of the fix**: episode length ≥6000 steps or staged episodes; trigger the cap on "cleared share
  / decay of kill throughput" rather than a fixed number of steps; progress = kills / damage / pick-ups /
  distant new cells.

### 7 · C7: the HP economy is priced at zero: HP / potions have no price, a12 is masked, a13 has no prior, and the depth multiplier has the wrong sign
- **Strongest evidence**: `env.py:4566-4711` _reward has no HP or potion term; `--no-drink-sovereignty`
  makes m[12]=False, and the reflex only drains below 50% HP; priors exist only for a14 (2.5) / a11 (2.0).
  In three probe groups, 41/41 deaths happened at belt=0, and in 39/41 there were still potions on the
  floor of the level; exam deaths 117/117 and 106/106 at belt=0; a13 is legal in only 8% of decisions,
  adopted 11-20%, converted 6-33%. HP lost per kill: L1 1.6-2.8 vs L2 8.9, while the wage is ×1.25; the
  break-even death risk is 11.7% per kill vs a measured 1.5% → "keep fighting" is always worth it. Audit 4's
  counter-evidence: the greedy script had 0 deaths on 4 seeds, with potions left at the end → deaths are
  behavioural, not a matter of engine sustain. The in-leg deaths per episode rise monotonically 0.45→0.79.
- **Impact**: high. 67-74% of deaths cut the growth window at ≈1400 steps; alive episodes have xp 4035 vs
  died 2793; after descending, each potion on L2 is worth only 3 kills.
- **Discriminating probe (≤15 minutes)**: wrap the probe_r15_deployment callback with a "hoard potions
  override" (mask[13] ∧ hp≥50% → force 13); arm C on 24 seeds against the baseline's 17/24 deaths; a
  significant drop in deaths proves that unpriced potions are a binding clause.
- **Direction of the fix**: price HP differences / potions held; add a logit prior for a13; open a12
  autonomy; apply the depth multiplier to net income (after the HP cost).

### 8 · C8: end-of-episode accounting runs backwards: timeout≡death, anti-idling at half pay, zero survival premium
- **Strongest evidence**: `worker_env.py:1262-1265, 1377-1430`: surviving episodes with no progress in the
  last 140 steps are charged as terminal_death (L1 −26 > an L2 death −18.8); sentinel 83/883 episodes,
  −2143.6; 6 of 7 deterministic survivors are hit, including the best trajectory (81 kills, clvl3, L4);
  after the cap hits, layer_clock is pinned at ≥140 and cannot be released, so surviving to the end after
  the cap ≡ death. `env.py:4693-4705`: anti-idling counts decision steps, is not reset by kills, and halves
  farm after >300, which hits exactly the episodes that grind L1 without descending (clvl3 needs ≈70 kills,
  squarely in this class).
- **Impact**: medium. 29% of surviving episodes lose ≈11 kills' worth of income; PPO has no gradient pushing
  toward survival; together with C7 it explains the rising in-leg death rate.
- **Discriminating probe (≤20 minutes)**: audit_window_probe.py already has the fields
  `timeout_wo_progress`/`exhausted_end`; cross-check the share of timed-out episodes that hit the cap
  (expected ≈100%); or run a 9-minute smoke leg with the timeout penalty set to 0 and look at the slope of
  deaths per episode.
- **Direction of the fix**: decouple the timeout penalty from the death penalty (surviving to the end ≥0);
  count anti-idling in micro-steps and reset it on kills; price a survival premium explicitly.

**Not in the top 8 but registered**: C9 the gear channel (drop rate 1-3.6% per kill, Rags break in 300-700
steps, no shops / no gold pick-up, E1 AC:dmg ≈1:21): high-confidence facts, but for the L1→L2 gate only a
sub-item of C1 (AC15 is a lottery); C10 approach shaping 3-4 orders of magnitude below the noise floor
(a1-a8 ≈ uniform noise): the root of C3's wall-bump loops, needs regression verification; C11 statistical
diseases of the protocol (death censoring with median episode length 455-1440 steps, 37/128 lethal seeds,
the gate B anchor works against survival): fixed by changing the reporting definitions; C12 an explosive
barrel seals the spawn pocket in 1/128 seeds: drop the seed from the pool.

---

## 3. Suspects ruled out (cross-confirmed by the seven audits)

**Engine and bridge layer**
- XP award chain: per-kill XP = monstdat × (1+0.1Δlvl), booked exactly; zero_xp_kills=0; the 27th kill at
  2007 XP levelled up on time
- Multiplayer XP cap / difficulty multiplier / whoHit gap / Experience.tsv tampering / ValidatePlayer
  clamping
- Stat-point black hole: AutoSpendStatPoints runs at the end of every Step, statpts is always 0, and HP
  matches the formula bit for bit
- Lighting (IsTileLit) works headless; drops are not disabled (41%×26%); the a9/a10/a11/a12/a13/a14 macros
  themselves all execute, native receipt accepts=1
- a11 pathfinding: radius-112 whole-map BFS, stairs loosely reachable on 128/128 seeds, 16344 requests /
  16341 executions
- Weapon/shield durability does not break within 3000 steps (only light chest/head armour breaks, see C9)
- Engine sustain: the greedy script had 0 deaths on 4 seeds with potions left at the end → deaths are
  behavioural

**Window rules and mask layer**
- The a10 quest mask locking doors: progression_targets L1-L4 always empty, zero handoffs in 5065 windows
- Footprint contamination across levels: everything is cleared on a scene change
- DIVE mask illegal because the stairs are unexplored: the trigger export ignores lighting
- TAU_CAP truncating credit in 37.8% of cases: window boundaries are not terminal for PPO (done=False +
  bootstrap)
- The protected_walk barrier: it only blocks cells next to stairs
- Level-up window closures interrupting fights: live-window wages are continuous, no monetary effect
- Effect of RESUPPLY windows: 0 windows chosen
- The worker cannot see the clock / scene budget: wrapper_scalars include layer_clock/exhausted/
  scene_fraction

**Reward and training layer**
- E1 gear priced too low (actually too large per event, 12 ≈ 12 kills)
- Approach shaping dominating decisions in magnitude (±1-6 per episode, cannot dominate; but see C10 for the
  opposite)
- The xp ratio of 0.01 per point out of balance
- ent_coef/target_kl choking exploration: entropy 0.96-1.07 nat, KL early stop 17/160
- lr / training too short to learn: parameter displacement 19-30%
- Side effects of separate-root-context-critic-v2: EV 0.93-0.95
- The actor cannot see gear opportunities: p(a14|legal) already 0.46
- The a14 prior blocked by the mask: 55-60% hit rate when legal
- Value function collapse / Adam state contamination / distillation anchoring pulling back
- A deformed action histogram inherited from the previous generation's doctrine: not deformed under
  sampling; the deformation appears under argmax

**Protocol layer**
- Seed pool / map bias: training uses the whole uint31 range, exams use a reserved range of the same
  distribution
- Different episode lengths in training and exams: both sides use 3000
- clvl/max_hp reset to zero after death: not contaminated (only AC/dmg/gold are)
- raw monsters reachable=0 meaning unreachable: it actually means invisible

---

## 4. Causal coupling graph

**Main chain (XP does not grow)**:
C1 impossible threshold ← C6 3000-step ceiling at clvl2 ← {C5 a10 saturation halves the reachable roster,
C6 the cap chases the worker away at 31% cleared, C7 deaths cut episodes at 1400 steps}
→ even if C5/C6/C7 are fixed, C1 remains a constant 0; conversely, changing only C1 without C6 still does not
reach clvl3 within 3000 steps. **C1 and C6 must be changed together.**

**Main chain (forced descent → death)**:
C2 escape hatch is dead code → the exhaustion mask is the only way → the exhaustion trigger differs by
regime {sampling: C6 cap 1800; argmax: C3 fake 140 clock + clock inheritance; greedy / some seeds: C5 a10
None} → all three are unrelated to the real cleared share → a clvl2 worker loses 8.9 HP per kill on L2 (C7
wrong sign) → death → C4 post-mortem readings → "zero gear growth" misdiagnosis → R15 placed the root cause
in the growth engine.

**The apparent contradiction between C5 and C6's definition of progress**: audits 1/4/6 say a10 saturates
and returns None (exhaustion too early), audit 3 says pacing resets the clock (exhaustion never comes). They
are two sides of one defect: both "progress" and "frontier" are defined by the footprint ±1, so pacing can
free-ride on it and monsters behind walls can starve it. The unified fix is to switch both to a kills /
distance / global-BFS definition.

**C3 → C8**: argmax limit cycles (a11↔a13 oscillation, BFS None→wait→fuse) burn the budget and then fall into
"no progress in the last 140 steps" → charged as death; the sampling regime hardly ever triggers it. C10
(movement keys never learned) is the root of C3's wall-bump loops.

**C7 ↔ C8**: unpriced HP + timeout≡death + anti-idling at half pay all point the same way, telling the
worker that "staying alive is worth nothing and grinding L1 does not pay", which correlates with the in-leg
death rate rising monotonically 0.45→0.79 (causation needs an intervention experiment).

**C4 and everything**: not a cause, but it underlies the "decisive data" of the R15 verdict and must be
corrected before the other items are re-judged.

---

## 5. Plain-language conclusion (≤300 words)

What is most likely missing beyond the five layers: **the problem is not the worker but the ruler and the
gate**. First, under the warrior's stat allocation, HP90 in the L2 row of the readiness table is equivalent
to level 4 (8040 XP), while clearing every monster on L1 yields only 6-9 thousand XP: the threshold is
arithmetically closed to any policy, and the 636 refusals are an inevitable constant. Second, the "release
at 25% cleared" escape hatch reads a roster that contains only live monsters and is always false; so the
worker can go down only through the exhaustion flag, and the exhaustion flag is unrelated to the real
cleared share. Third, the decisive data "AC 5-7, zero gear growth, gold with nowhere to spend" are corpse
readings (on death the engine strips the gear and halves the gold), and exams use argmax while training
samples, so kills are underestimated by half and the descent success rate by a factor of six.

Verify three things first:
1. The readiness-table arithmetic (5 minutes): call readiness_power_ratio with clvl3/HP86 and check that it
   is <1; compare the L1 XP pool with 8040.
2. Recompute the four deployment JSONs stratified by survival + a 32-seed sampling vs argmax pairing (15
   minutes).
3. Reproduce the dead-code escape hatch + the rate at which a10 returns None and progress comes from pacing
   under the trained worker (15 minutes).

If all three are confirmed, R16's first priority is not to retrain the worker but to change the threshold
definitions, the exhaustion/clearing criteria and the evaluation regime.

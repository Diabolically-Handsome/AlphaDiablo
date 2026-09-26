# R17 direction review (2026-09-02)

Dossier: the R16 verdict [r16-VERDICT-20260902.md](r16-VERDICT-20260902.md) (ledger row VERDICT_R16); four
R17 proposals (1 resource channel / 2 L2 logistics and survival course / 3 readiness table v0.3 / 4
readiness-teacher manager), each with three adversarial critiques (feasibility / science / goal alignment,
12 in total); four surveys (town economy, gear and readiness, RESUPPLY and the manager path, L2 death
forensics). Review rules: a fatal defect outweighs strengths; structural progress toward "breaking
through" outweighs polishing L2; the readiness principle (descend only once the power value is reached)
must not be satisfied by definition; no more endless turning of wage knobs; engineering and calendar costs
are reported honestly.
This is a new file and changes no existing file; all references are file:line, relative to the repository.

## 1. Rationale

### 1. The R16 conclusion (only the numbers are repeated)
Four gates passed for the first time ever (A/B/C/D against the new-rule anchor); deployment on 16 seeds:
arm survival 5/16 vs certified worker 10/16, L2 reached 10/16 vs 1/16, clvl 3.06 vs 2.25; the arm died on
L2 in 8/10 cases (AC at death 7,7,7,7,7,8,9,16; L2 roster 114-162 monsters; median step of death 3885);
training classroom 244 descents, readiness escrow vested 0 / unready_denied 6,441. No crown, as
pre-registered. The R16 verdict named the next layer of root cause "the dead end of cleared but unready":
the coach opens DIVE when the level is cleared regardless of readiness, while AC 9 can come only from an
armour drop and gold stays at 100 with nowhere to spend it.

### 2. Three corrections to the rationale (each confirmed independently by ≥9 of the 12 critiques)
**Correction 1: the real name of the dead end is "the exhaustion flag forces the descent", not "dive when
cleared".**
- The coach rule is indeed `want = DIVE if (ready or cleared) else FARM`
  (python/diablogym/worker_env.py:1128-1130; mirrored in the deployment probe
  train/runs/r10-staging/probe_r15_deployment.py:215-224).
- But in deployment L1 was never cleared within 6000 steps: the three L1 survivor seeds still had 18/13/35
  monsters at the end (r16-deploy-arm.json rows 2114001/002/007), and the anchor's campers had 9-45 left.
  `cleared` (kill ratio 1.0, worker_env.py:169-180) almost never triggers in deployment.
- What really decides the descent is the frozen mask rule: `forced_dive = _farm_handoff(...) or
  (self.exhausted and m[DIVE]); m[FARM] = not forced_dive` (python/diablogym/options_env.py:739-741).
  `exhausted` is set when a FARM window has no progress for KILL_PATIENCE=140 micro-steps **or**
  farm_scene_steps ≥ farm_scene_cap (3600) (options_env.py:63, :1022-1027; `_mark_exhausted` :542-549), and
  `reset_layer_clock_on_window` clears only layer_clock, not the flag (:770-774).
- Evidence: in R16 training, window closures for exhausted 284 ≈ descend 244
  (r16-arm-a-constitution/sentinel.jsonl, last row); 1614 live DIVE windows gave only 244 descents
  (r13_dive_audit.jsonl, last row); L2 death steps in deployment 3768/3869/3885 ≈ the scene budget of 3600
  + travel.
- Corollary: any option that changes only the coach's want (3, 4) delays the descent by at most one window
  on this path (≤140-600 steps); the R15 verdict already said that "every want of the coach is overridden by
  the mask fallback" (r15-VERDICT-20260901.md §2.1). To make the readiness principle **satisfiable** on
  cleared / exhausted levels, both of these are needed: (a) a growth channel other than diving, so that
  "ready" is reachable; (b) the exhaustion escape hatch made visible to readiness and accounted for (a
  forced unready descent = no escrow payment, counted separately).

**Correction 2: three latent defects that must be fixed whatever the direction.**
1. The escrow readiness gate uses the v0.1 ruler: `_descend_escrow_settlement` at worker_env.py:1866-1877
   calls `readiness_power_ratio` (v0.1 table :197-205, L2 needs clvl3/HP90/AC15/dmg8), while the coach uses
   `readiness_power_ratio_v2` (:1128 → v0.2 table :139-141, L2 = clvl2/AC9/dmg6). vested 0 / denied 6,441 is
   a structural necessity; the R16 verdict's "none of 244 descents met the bar" is partly a ruler artefact
   (in 4 of the 10 L2 arrivals in deployment AC ≥ 9, which under v0.2 should vest).
2. A landmine in the leashed_ppo receipt: the zero-timeout branch requires `float(reward) == worker_wage`
   (train/leashed_ppo.py:4746), and the death-equivalent branch requires `== worker_wage + timeout_total`
   (:4785); but worker_env's end-of-episode policy_reward also adds `_escrow_vest + _hp_econ + depth
   shaping` (worker_env.py:2113-2121, :2207-2214). The 78 timeouts per leg in R16 did not trigger it only
   because vest was always 0; the first time escrow really vests in the same window as a timeout, a
   RuntimeError blows up the leg.
3. Missing instrumentation: no archive records the descent step, belt/AC/HP at descent, or whether the
   descent was forced or voluntary (the probe panel probe_r15_deployment.py:102-114 has no hp/belt/descent
   step). "AC at descent vs at death" and "trip too long vs a11 stall" can only be inferred.

**Correction 3: several pieces of "evidence" cited by all four proposals do not hold.**
- "Every death row has belt=0" is a tautology of the brainstem reflex: `_drain` pours potions every step
  when 2*hp<max_hp ∧ belt>0 (options_env.py:1144-1157; predicate env.py:956-962), and the largest single hit
  on L2, 15 HP, is not enough to skip the reflex, so under any policy the belt must be 0 at death. The anchor
  shows the same 41/41 and 47/47 (r16-anchor-xdevil-a / -xm29-a.json). It does not prove that "the worker
  cannot use potions", only that "it dies after the potions are gone", i.e. **supply** is the constraint
  (which actually supports option 1).
- Gold arithmetic: the engine's RndItemForMonsterLevel drop rate = P(GenerateRnd(100) ≤ 40) × P(second roll
  > 25) ≈ 0.41 × 0.74 ≈ 30% per kill (engine Source/items.cpp:3257-3266), not the proposal's 45%; a 100-kill
  L1 clear ≈ 285 gold, not 420. And the engine's AutoPickup scans the 8 neighbouring cells only when a step
  completes (Source/player.cpp:442; Source/qol/autopickup.cpp:93-110), so gold under a monster killed in
  place with a9 is picked up only by later movement. Coverage was **not measured**.
- "The DEVIL manager keeps 29/128 more alive" (the core argument of option 4) is an arrival artefact:
  DEVIL×7e31dc54 reaches L2 in only 4/128 and 0/128 on the 3000-step exam sheets
  (eval-assembled/r16-arm-a-constitution-xdevil-{a,b}.json), while M29 reaches it in 45/128 and dies on L2
  27 times: the difference is who reaches L2, not the quality of decisions.
- "α2 proves this policy family can learn to survive" (option 2): α2's devil-sheet depth histograms are
  {1:128}/{1:127,2:1} (r13-arm-a2-shaping-xdevil-{a,b}.json): α2 survived by refusing to dive and never
  survived on L2.

## 2. Four routes: summaries and panel scores

### Summaries
| Route | In one sentence | Change surface | Self-reported cost |
|---|---|---|---|
| 1 Resource channel | automatic gold pick-up + going upstairs back to town + UI-free buying of armour / potions / free healing at Pepin + a readiness-v4 coach (cleared/exhausted and not ready → one town trip, no dive); the RESUPPLY window (Discrete(3) unchanged) carries the town script | bridge ~280 lines of C++ (gold pick-up / going upstairs / shop observation / buy-sell transactions) + ~630 lines of Python in env/options/worker/train/eval + new probes/tests/drivers | 95-110 h, 3 weeks; the first C++ round |
| 2 L2 logistics and survival course | worker-side deep-start curriculum (p=0.5 to L2) + red-zone HP-loss pricing ×3 + an a12 drinking prior + escrow ruler v0.2; no engine/env/options change | ~300 lines in worker_env/leashed_ppo/train_ppo + new probes/drivers | 30-36 h, 3-4 days |
| 3 Readiness table v0.3 | table v0.3 (L2/L3 AC changed to 7, a new belt column belt≥2) + DIVE only when cleared∧ready + the escrow gate switched to the v0.3 ruler + a belt-floor RESUPPLY coach rule + a fix for the receipt landmine | ~200 lines in worker_env/leashed_ppo/train_ppo + new probes/tests | 14-20 h, 3-4 days |
| 4 Readiness-teacher manager | initialise the Discrete(3) manager by BC from the readiness-v3 script, then fine-tune with MaskablePPO (worker 7e31dc54 drives FARM+DIVE); training-side objective r' = R − 48·[voluntary unready descent] + a time integral of surviving depth | ~240 lines in the options path of train_ppo + ~1000 lines of new files for BC/probes/drivers | 48-64 h, 2 weeks; 12-20 h of training |

### Panel scores (feasibility / science / alignment; out of 10; verdict = recommend_with_fixes / reject)
| Route | Feas. | Sci. | Align. | Total | Verdicts | Fatal defects (a reject verdict or "fatal as pre-registered") | Main fixable defects |
|---|---|---|---|---|---|---|---|
| 1 Resource channel | 6 | 5 | 5 | **16** | 3× recommend_with_fixes | No reject. The science critique found three "fatal as pre-registered but fixable" items: (F1) the main metric alive/16@6000 can be gamed for free by trips truncating exposure time (a 600-step town trip pushes 3 of 8 L2 deaths past the horizon); (F2) 16 seeds have no power (only 5/16 vs 11/16 gives p<0.05); (F3) a trip trigger that includes `self.oe.exhausted` reopens R15's free-riding of the exhaustion flag and stacks it with the 24-unit v0.2 escrow bounty | Gold arithmetic overestimated ×1.5, pick-up coverage not measured; going up and down stairs pays +40 per level repeatedly through `_reward` (env.py:4784-4801); the worker observation contains gold/1000 (env.py:5012,5110) → OOD confound in T0; TOWN_CAP 1000 > TAU_CAP 600 conflicts with the window rules; counting gold as positive_progress (options_env.py:912) changes the FARM window rules; nav.walk_to bypasses env.step and does not count steps (nav.py:36-49); the threshold numbers differ from analyze_arm_gates.py:53-55 while claiming to be "verbatim"; the rebuild pipeline has not run since 27 July; `_town_trip_legal` lacks (cleared∨cap)∧¬ready, so the M29 anchor could go to town for 1000 steps from its starting 100 gold; "crowning the T0 configuration" bypasses the four gates; the Lazarus argument was already voided by AutoTurnInBetrayerStaffForMonotonicTask (src/diablogym.cpp:2085-2098) |
| 2 Survival course | 6 | 3 | 5 | 14 | 1× reject (science) + 2× with_fixes (the alignment critique judged it "fatal as a direction") | Science F1: the belt=0 tautology, and the anchor never drinks actively yet survives better; F2: the wage arithmetic is off by ~4× (measured HP loss 2.09 HP per kill → −0.21 per kill; even ×3 only breaks even), and the red-zone penalty accumulates only in the belt=0 state = a second death penalty, while R14 already showed the price axis has no solution; F3: escrow vests on any non-death window closure (worker_env.py:1826-1848, including 'end'/'stall'/'cap'), so the optimal policy is to refuse to dive until ~5800 steps and then descend to collect 24; F4: the kill-switch mixes curriculum and non-curriculum episodes; F5: it does not touch the dead end. Alignment: it trains the worker to "survive descents that violate the readiness principle", and even full success is only L2 polishing | worker_env is a protocol file, so all six anchors must be re-baked (eval_contract.py:55-63); under autonomy the a12 band is already enforced by options_env.py:1776-1800 (once per window, forbidden after the reflex), so the premise of an a12 prior is wrong; "63% of steps on L1" confuses window type with level; a 12-window curriculum can exceed 6000 steps; continued-training candidate zips are not actually forbidden (R-6 overestimated in the other direction) |
| 3 Table v0.3 | 7 | 5 | 3 | 15 | 1× reject (alignment) + 2× with_fixes (the science critique judged it "fatal to the causal claim") | Alignment: an **inverted ruler**. Changing the L2/L3 AC to 7 = the starting AC, the third downward revision in a row (the human guide table has L2 AC 15-25, L3 30-40, r15-READINESS-TABLE-draft.md:15-17), so "ready" is satisfied by definition; there is no mechanism that lowers h(2) (R14: h comes from skill, not from the ruler); the cleared∧ready gate is a no-op on the exhaustion-flag path; it again delays the engineering on the critical path. Science: belt≥2 = the starting equipment, and clvl≥3 is already held by 206/220 arrivals, so the mechanism cannot produce 0.80→0.55-0.65; the exhaustion exit is misdescribed as "opening only at 3600"; RESUPPLY-first hands control to a script that does not attack, and it also triggers on L2 with monsters around | The three code facts and the two landmines (escrow ruler, receipt) are all true and were first found by this proposal; the self-reported size of 100-200 lines is really ~520-590; an exact reproduction of G0-A(2) needs an explicit far_tiles=0; the escrow gate is evaluated at window close while the coach evaluates at window open, so the belt column would be misaligned by reflex drinking |
| 4 Teacher manager | 6 | 3 | 3 | 12 | 2× reject (science, alignment) + 1× with_fixes | Science F1: the mechanism contradicts its own cited data (DEVIL never leaves L1); F2: the objective does not count survival (−48 exactly cancels the +48 descent bonus, the survival term is ≤24 per episode, the J difference O(10-30) ≈ 0.2-0.4 σ, and the premise of R9's collapse is unchanged); F3: G5 (died ≤ DEVIL+5) and G2 (L2 reached ≥0.8×) are mutually exclusive, and the only way to satisfy both is "FARM until the scene cap forces the descent" = const-FARM → hits G4. Alignment: it turns the hard rule of "give the manager a power value" into a soft penalty that 140 idle steps can bypass (free-riding at manager level); the init-source checkpoint's full state_dict exact match (train_ppo.py:7241-7255) cannot seed a Discrete(4) TOWN manager; not on the list of candidates for decision; falsified twice, in R9 and R12 | "No re-bake needed" is false (train_ppo is under the five-file re-bake rule, r13_ledger.jsonl:4); the coach's cleared branch cannot be represented in the 303-dim observation (visible ∧ reachable monsters vs the whole roster; kills on the current level /50 saturate); the BC wiring is 40-60 lines heavier than stated; the throughput estimate of 6-8 h with 8 envs is unmeasured |

### Transplantable parts (already grafted into the decision)
- From 3: the escrow ruler fix v0.1→v0.2; the leashed_ppo receipt fix; a **belt column** in the readiness
  table (but AC not lowered); per-descent telemetry; "a zero-training ablation of the previous worker in the
  new form" as a decision point.
- From 4: the G0-M1 scripted counterfactual managers (readiness-v3-strict / const-FARM / const-DIVE) that
  measure the free-margin share and the forced-descent share: the most valuable missing measurement since
  R16, and it needs no change to train_ppo (~20 lines of probe extension); the culture of behavioural
  identity pre-checks.
- From 2: the worker-side deep-start curriculum (handed to R18); per-step hp/belt/block trajectories in the
  probe; red-zone pricing and the a12 prior are **not adopted**.

## 3. Decision

### Direction: 1, the resource channel (pick up gold → town trip to buy armour and potions → descend only when ready), as the main line of R17, with R17.0 "ruler and instrumentation" as a mandatory prelude;
### Second choice: 3 (its parts only); 2 handed to R18; 4 closed.

**Reasons (rule by rule):**
1. A fatal defect outweighs strengths: 1 is the only direction without a reject verdict; 2, 3 and 4 each have
   at least one reject, and the rejects are all at the level of mechanism (2: wage arithmetic and vest
   timing; 3: inverted ruler and a no-op gate; 4: arrival artefact and an objective that ignores
   survival), not wording. The three "fatal as pre-registered" items of 1 are all about **metrics and how
   the pre-registration is written** (exposure normalisation, seed count, trigger clause), and the revisions
   in this decision remove them.
2. Structural progress toward "breaking through": only 1 builds a growth channel. The human table needs AC
   30-110 and resistances from L3 on, and the v0.2 table also requires AC to rise per level
   (worker_env.py:140); without a non-lottery source of gear and potions, the same dead end repeats on L3
   (h(3)=0.67), L4 and so on. The bridge primitives that 1 lands (going upstairs, town navigation, NPC
   transactions, shop observation) are required for the town portals at L5/9/13, town portal scrolls and
   resupplying potions at depth; this review **does not accept** the Lazarus→Cain argument (the bridge
   already bypasses it, src/diablogym.cpp:2085-2098).
3. The readiness principle: on cleared / exhausted levels, "ready" today is an armour lottery of ~2% per kill,
   and the armour breaks in 3-12 hits (audit C9). Option 1 makes "reach the power value" **reachable** on
   every seed, instead of **true by definition** (3) or **a soft penalty that can be bypassed** (4). The
   exhaustion escape hatch stays as an escape hatch, but becomes **visible in the accounts**: a forced
   unready descent pays no escrow, is counted separately and serves as a stop-loss metric.
4. Against knob-turning: 1 is a change to environment affordances, not to prices; the bulk of 2, 3 and 4 are
   wage / ruler / coach constants. Option 1 adds only three new constants (TOWN_CAP, TOWN_MIN_GOLD, the belt
   column threshold), each grounded in engine arithmetic, with no grid.
5. Honest costs: 1 is the first change to the bridge since the binary of 2026-07-27, and the rebuild
   pipeline (build/CMakeCache.txt points at a temporary directory that no longer exists) has never been
   run: **that is the calendar risk**, not the line count. So R17.0 puts "a bit-level proof of rebuilding the
   unchanged bridge" before any C++, and makes the zero-training T0 factorial probe the only criterion for
   launching a training arm, so that three weeks are not burned on a broken channel.

### Sequence
**R17.0 ruler and instrumentation (2-3 days, zero training, zero C++ changes; can run in parallel with the
bridge construction of option 1)**
- (a) New file train/runs/r10-staging/probe_r17_deployment.py (a clone of probe_r15, whose bytes stay
  unchanged): for every descent record {beat, belt_heals, hp, AC, clvl, ratio_v2, forced = ¬mask[FARM],
  trigger reason cleared/cap/idle-clock}; at death, the belt and the visible floor potions on the level;
  steps spent on L2; kills by level; managers: readiness-v3 (its sixteen rows 2114000-015 must be
  bit-identical to r16-deploy-arm.json), readiness-v3-strict, const-FARM, const-DIVE; workers: 7e31dc54 and
  the certified worker; 48 seeds 2114000-2114047 × 6000 steps (≈ 4.5 min per combination, at the ledger's
  87 s per 16 seeds). Output = the R17 baseline row A0′ and the **forced-descent share** (which decides
  whether the exhaustion exit needs a constitutional change).
- (b) Amendment 1 (off by default; worker_env/leashed_ppo are protocol files → the old-rule 4×128 + all six
  r16-anchor sheets re-baked bit for bit): `--worker-descend-escrow-readiness-table {v1,v2}`
  (worker_env.py:1872 switched to `readiness_power_ratio_v2`); the leashed_ppo receipt checks against
  `worker_wage + vest + hp_econ + shaping` (a missing key = 0, byte-equivalent under the old rules). G0 row:
  an 8192-step smoke test proving vested > 0 and zero RuntimeError.
- (c) G0-0a: rebuild the **unchanged** src/diablogym.cpp from a local DevilutionX checkout (sha 34c4cfc2,
  -DDEVILUTIONX_SRC) and prove the 4×128 old-rule anchors bit for bit, separating toolchain drift from code
  drift. If this step fails, not a single line of C++ is changed for option 1.

**R17.1 resource channel construction (weeks 1-3): the original option 1 + the mandatory revisions of this
review**
- Trip legality lives in OptionsEnv `_town_trip_legal` and is **manager-independent**: dlvl == 1 (R17 starts
  trips only from L1) ∧ the WM_DIABPREVLVL trigger exists ∧ (cleared ∨ farm_scene_steps ≥ farm_scene_cap) ∧
  ¬ready ∧ trips_this_floor == 0 ∧ gold ≥ 50. **Never use `self.oe.exhausted`** (the 140-step idle gate);
  the trigger reason is recorded per trip, and an idle-clock trigger = a hard stop-loss.
- TOWN_CAP = 600 (≤ TAU_CAP), with the town window rule placed before the general cap rule; the RESUPPLY
  script still picks up potions first when a13 is executable, and such a window does not count as an "empty
  run".
- Town macro phase 0 = **scripted gold sweep** (act_pickup_gold_at: CMD_GOTOAGETITEM → AutoGetItem →
  GoldAutoPlace, inv.cpp:1739-1758) that clears the visible gold piles without relying on incidental
  movement; autoGoldPickup is switched on as well.
- Gold is **not** counted as positive_progress (an exclusion added at options_env.py:912), so the FARM
  window rules do not change.
- Inside town windows, `_reward` pays the dl term only when cur > `_econ_episode_max_depth` (round trips net
  to zero), and `_econ_steps_on_level` is frozen (env.py:4784-4801, :4923-4929); bit-level proof with the
  default off.
- The SHOP/DOWN phases use **a walker that accounts per step** (the a11 trigger walker with a parameterised
  msg), not nav.walk_to (which bypasses env.step / the reflex / the fuse).
- Purchase planner: armour must have durability ≥ 15 (Cloak 40 gold / durability 18, Quilted 200/30;
  **never buy Rags**), with the AC target from the table; spend the remaining money on potions up to the belt
  limit; Pepin heals to full for free.
- Readiness table **v0.3 = v0.2 + a BELT column** (L2 ≥ 4; basis: RestorePartialLife averages ≈ 40 HP at 86,
  one potion ≈ +19 hits absorbed, four potions ≈ ×2.9; economically 100 + ≥140 gold picked up − 40 for a
  Cloak ≥ 4 potions); **AC stays at 9**. readiness-v4 coach: ready_v03 → DIVE; (cleared ∨ cap) ∧ ¬ready ∧ trip
  legal → RESUPPLY (town); otherwise FARM; once the trips are used up the exhaustion exit works as before,
  but is recorded as forced-unready and pays no escrow.
- Worker observation gold/1000 (env.py:5012, :5110): T0 gets a "gold pick-up on / town trips off" control
  row; if that row shows |Δalive| > 2/48, the training arm adds a default-off flag that clamps gold in the
  worker view to 0.1.
- Thresholds exactly as in analyze_arm_gates.py:53-55 (A: UCB95 < 0 ∧ died ≤ 110; B: LCB > −0.10; C: ≥
  0.95×); no more claiming "verbatim" while changing the numbers.
- **T0 crowns nothing**: crowning an environment change would need its own rule; T0 only proves the
  mechanism and serves as the launch criterion for a training arm.
- The fallback of a "virtual shop at the stairs" can be adopted **only by a separate decision**, not
  substituted by an engineer.

**Gate T0, a four-row zero-training factorial probe** (R16 worker 7e31dc54, 48 seeds): (a) gold pick-up on /
town trips off; (b) potions only; (c) armour only; (d) the full channel. Criteria in §4.

**R17.1 training arm** (only if T0 passes): continue training 7e31dc54, readiness-v4, escrow ruler v2, T =
round2048(266240/(1−s)); four gates against the recast r17-anchor; deployment against A1 = T0(d) and A2 =
certified worker / R17 environment.

**R18**: the usable parts of option 2 (the worker-side deep-start curriculum + a retreat macro) open as a
course once "potions and armour are available" (R17 telemetry decides whether the wall is logistics or
tactics); a sell / repair / town portal economy for L3+; roadmap amendment (R17 shop core → R18 sell /
repair / portal + L2 survival course → R19 L3-L5 curriculum).
Option 4 reopens only once TOWN/RESUPPLY become options worth switching to (Mark-I).

## 4. Suggested skeleton of the R17 pre-registration

### 1. Main metrics (deployment probe in r17 form: readiness-v4 / autonomy on / v4 / hunt / cap 3600 / clock reset / 6000 steps / sampled; ≥48 seeds, 128 preferred)
- **Main-1 survival (exposure-normalised)**: L2 hazard per thousand steps = L2 deaths / Σ steps spent on L2;
  plus "survival at first descent step + 1800" (fixed post-descent exposure). The arm/T0 compared with
  A1 and A2; paired McNemar (the estimator of analyze_arm_gates.py:23-31).
- **Main-2 growth retention**: L2 reached ≥ 0.8 × A0′; mean clvl ≥ A0′ − 0.1; L1 kills per thousand
  **dungeon steps** (town steps excluded) ≥ 0.85 × A0′.
- Pareto count alive ∧ L2.
- Mechanism sub-metrics (informational, tautologies forbidden): distribution of trip trigger reasons
  (cleared / cap / idle-clock, the last must be 0); median trip steps; AC and durability bought; potions
  bought; empty runs; belt/AC at descent; gold picked up per L1 clear; forced-unready descent share (T0(d) ≤
  25%).
- Crowning rule: only through the four gates (against r17-anchor) + deployment "survival not worse than A2
  and growth significantly better than A2" (the R16 wording, anchored on A2 rather than a constant).

### 2. Gate T0 (zero training; the only criterion for launching a training arm)
(d) against A0′: paired saved − lost ≥ +6/48 and UCB95 < 0; L2 reached ≥ 0.8 × A0′; L2 hazard per thousand
steps ≤ 0.7 × A0′; (d) − (a) survival ≥ +4/48 (a channel effect rather than observation drift); (b) and (c)
reported separately, and the verdict must name the load-bearing lever. Any unmet → no training arm; R17
verdict = channel not confirmed / not attributable.

### 3. Four gates and the deployment gate
- A/B against r17-anchor-devil-{a,b} (certified worker, recast with the R17 flags): A UCB95 < 0 ∧ died ≤ 110;
  B LCB > −0.10.
- C against r17-anchor-m29-{a,b}: ret and kills ≥ 0.95× (the file rule); ≥ 1.0 reported for information.
- D: D1 FARM argmax agreement < 0.99 against 7e31dc54 before continued training, reported **both** on R16
  environment observations and on R17 environment observations; D2 DIVE against the script < 0.99.
- Deployment gate (arm vs T0(d)): survival ≥ T0(d) − 2/48 ∧ hazard ≤ T0(d) ∧ (clvl or kills per thousand
  dungeon steps) ≥ 1.05×; otherwise no crown, and no "crowning of T0".
- Training-leg gate (last row of r13_dive_audit): descends per 10 live DIVE windows ≥ 1.0; stall share ≤
  25%; vested > 0; zero receipt RuntimeErrors.

### 4. Anchoring method
- Default off, zero drift: old-rule r10-cand/r10-anchor 4×128 bit for bit (run_r13_rebake.py); the six
  r16-anchor-* sheets bit for bit with all flags off; probe_r15_deployment with all flags off, 16 rows
  bit-identical to r16-deploy-arm/anchor.json (under the new bundle).
- G0-0a: rebuild the unchanged bridge → 4×128 bit for bit (before any C++).
- Recast: r17-anchor-{devil,m29,m29full}-{a,b}, certified worker, R17 flags, 128 seeds × 3000 steps (new
  archives; gold pick-up changes every trajectory, and the manager-independent `_town_trip_legal` guarantees
  that M29 will not go to town from spawn).
- Deployment anchors: A0′ (R16 worker / R16 environment, re-probed on 48 seeds under the new bundle, the first
  16 rows bit-identical to the R16 rows); A1 = T0(d); A2 = certified worker / R17 environment.
- Freeze: implementation bundle + bridge .so + engine .so + pre-registration sha256 into r13_ledger.jsonl;
  after freezing, any recipe change needs an amendment in a separate file.

### 5. G0 acceptance
| Item | Content | Pass line |
|---|---|---|
| G0-0a | rebuild the unchanged bridge, 4×128 old rules | bit for bit (hard stop-loss) |
| G0-0 | rebuild with the new C++ + suite (876 + new) | all green |
| G0-1 | headless town round-trip smoke test, 16 seeds (starting from L1) | 0 crashes/hangs; guards do not throw inside windows and must throw outside; return survival ≥ 15/16; median trip ≤ 400 steps, maximum ≤ 600; L1 roster / floor items identical before and after the trip |
| G0-2 | two runs on the same seeds (including shop stock / purchases / gold) | bit for bit (hard stop-loss) |
| G0-3 | economy, 16 seeds | gold pick-up coverage (piles generated vs piles picked up) ≥ 60% and ≥ 150 gold per L1 clear; ≥ 12/16 with AC ≥ 9 and durability ≥ 15 after the trip; ≥ 12/16 with belt ≥ 4; empty runs ≤ 2/16 |
| G0-4 | anti-free-riding | trips triggered by idle-clock = 0; ≤ 1 per level; L1 kills per thousand dungeon steps under readiness-v4 ≥ 0.85 × readiness-v3 on the same seeds |
| G0-5 | calibration (run_r17_g0.py, the R16 recipe + amendment 1) | descends per 10 windows ≥ 1.0; stall ≤ 25%; vested > 0; zero RuntimeError; s → T |

### 6. Numeric stop-losses (pre-registered; triggering means acting, and thresholds may not be changed afterwards)
| Trigger | Action |
|---|---|
| G0-0a not bit-identical | stop; retry pinning the compiler / flags from the old CMakeCache; still failing → escalate for a decision (bridge sha rule) |
| G0-1 crash/hang or return survival < 15/16 | stop; a fallback needs a separate decision |
| G0-2 not bit-identical | hard stop-loss |
| G0-3 coverage < 60% or gold < 150 | condition the potion sub-metrics on the measured income; still < 150 → L1 economics falsified, reported as is |
| G0-4 idle-clock trips > 0 or kills < 0.85× | drop the cap clause (cleared only) and rerun; still triggered → stop |
| T0 (d) − (a) < +4/48 | observation drift rather than the channel → no training arm |
| T0 L2 reached < 0.8 × A0′ | use the descent panel to tell "trip too long" from "a11 stall"; do not adjust thresholds |
| Gate C < 0.95× on both sheets | anti-forgetting failed; not relaxed |
| Calendar: G0-0a…G0-2 not passed by the end of week 2 | escalate for review; total > 4 weeks → freeze the scope and move the rest to R18 |

## 5. Engineering and calendar costs (honest)
| Item | Estimate | Basis |
|---|---|---|
| R17.0 probe (new file) | ~150 lines, 4-6 h; running 6 combinations × 4.5 min ≈ 30 min | ledger: 87 s per 16 seeds |
| R17.0 amendment 1 (escrow ruler + receipt) | ~20 lines + tests, 4-6 h; re-bake 4×128 ≈ 17 min + six sheets ≈ 50 min | ledger timings of the R16 re-bake and the six sheets |
| G0-0a rebuild of the unchanged bridge | 4-8 h (pipeline unverified; CMakeCache points at a path that no longer exists) | build/CMakeCache.txt:380; .so mtime 2026-07-27 |
| Option 1 bridge | 280-340 lines of C++, 32-40 h | proposal ~280 + the gold-sweep macro |
| Option 1 env.py (town macro + per-step walker + planner + round trips netting to zero in the reward) | 28-34 h | proposal 20-24 h + critiques +80-150 lines |
| Option 1 wiring in options_env/worker_env/train_ppo/eval | 16-18 h | proposal |
| Option 1 tests + G0 probes | 16-20 h | proposal 14-16 + coverage/reward tests |
| Pre-registration / critic panel / verdict | 12-14 h | same as R16 |
| Buffer (headless loading of town mid-game unverified) | 8-12 h | survey gap |
| **Total** | **~125-150 h; 3-4 weeks of calendar time** (R17.0 done within week 1; T0 reachable at the end of week 2 = the earliest go/no-go) | |
| Compute | < 12 h: probe 48 seeds ≈ 4.5 min per combination; six anchors ≈ 50 min; re-bake ≈ 1.5 h; training leg ≈ 1 h (R16 325,632 steps in 2,709 s); six sheets ≈ 45 min | ledger |

For comparison: R16 took ≈ 9 h from approval of the constitutional repair to the verdict, because it was all
Python; R17 is the first change to the bridge, and the constraints are the two unknowns of rebuilding and
headless town, not the line count. The 3-4 days of option 2 and the 3-4 days of option 3 are cheap, but
neither changes the dead end that the next layer has to face.

## 6. Risks and stop-losses (supplementing §4.6)
- **Scope creep**: sell / repair / town portals, starting from L2, and manager learning all go to R18+; R17
  does only one round trip starting from L1.
- **Contract drift**: changes to env/options/worker rotate the implementation bundle and the protocol
  bundle → all anchors re-baked (already in the calendar); frozen artefacts only get new files, never edits.
- **The temptation of "crowning T0"**: T0 is a proof of mechanism; crowning goes only through the four
  gates + the deployment gate; a separate crowning procedure for environment changes would first need its
  own rule.
- **A hollow fallback**: a virtual shop at the stairs would remove every primitive aimed at breaking
  through, leaving only L2 polishing; it can be adopted only by a separate decision.
- **Durability erosion**: bought Rags break in 3-5 hits; the planner's durability ≥ 15 filter is written into
  the base recipe rather than kept as a contingency; stop-loss: if in T0 more than 50% of L2 deaths have AC
  back at 7 at death → the verdict marks the AC lever as failed, and R18 evaluates repair transactions.
- **Exposure confounding**: steps spent on trips mechanically raise survival at 6000 steps, so the main
  metric is now exposure-normalised; any contingency that "extends TOWN_CAP" is legitimate only under the
  normalised metric.
- **Anchor compression**: after gold pick-up, RESUPPLY becomes legal more often for the M29 anchor, so gate C
  may compress toward 1.0; enforced as written, not relaxed afterwards.

## 7. In one sentence
R17 takes the resource channel: first spend two or three days fixing the ruler (escrow v0.2), the fuse (the
receipt landmine) and the instruments (descent telemetry, forced share) and measuring the baseline, then
make the first C++ cut, so that "descend only once the power value is reached" has a way forward on every
seed, with no more wage-knob turning; the zero-training T0 factorial probe decides whether a training arm
launches, and whether the channel is confirmed will be known by the end of week 2.

## Appendix: index of key references
- Coach rule worker_env.py:1128-1130, fallback order :1141-1150; probe mirror probe_r15_deployment.py:215-235.
- Mask escape hatch options_env.py:739-741; exhaustion set at :63, :542-549, :1022-1027; clock reset does not
  clear the flag :770-774; positive_progress :902-912.
- Escrow ruler mismatch worker_env.py:1866-1877 (v0.1 table :197-205) vs :1128 (v0.2 table :139-141);
  vest/forfeit :1826-1848.
- Receipt landmine leashed_ppo.py:4746, :4785 vs worker_env.py:2113-2121, :2207-2214.
- Reflex drinking options_env.py:1144-1157, env.py:956-962; a12 autonomy band options_env.py:1776-1800.
- Gold: obs src/diablogym.cpp:1258; autoGoldPickup=false :1825; floor items have no value field :1477-1499;
  worker view env.py:5012, :5110; engine drop rate items.cpp:3257-3266; AutoPickup player.cpp:442 /
  autopickup.cpp:93-110.
- Town transition: SyncLoad WM_DIABPREVLVL src/diablogym.cpp:1132-1138; DisableLevelBacktracking :1878;
  patches/0005; env guard env.py:2190-2205; StartNewLvl precedent :3356-3361; shops SetupTownStores
  stores.cpp:2162-2181; buy/sell kernel stores.cpp:2861-2864 / 2658-2671 / 2102-2120; HealPlayer :1018-1029.
- Repeated payment for going up and down env.py:4784-4801, :4923-4929; a14 atomic gear swap
  src/diablogym.cpp:2795-2872.
- Thresholds analyze_arm_gates.py:53-55, :66-68; protocol file table eval_contract.py:54-64; implementation
  bundle train_ppo.py:798-809.
- Deployment rows r16-deploy-arm.json / r16-deploy-anchor.json; last training rows sentinel.jsonl /
  r13_dive_audit.jsonl; ledger r13_ledger.jsonl (VERDICT_R16, probe timing, re-bake timing).

# R16 pre-registration (frozen): constitutional repair of the ruler, the gate, the exam rule, the accounting and the field of view
Status: FROZEN 2026-09-02. Approved on 2026-09-01 for immediate repair; finalised at G0-6.
Basis: [r16-FRESH-EYES-AUDIT-20260901.md](r16-FRESH-EYES-AUDIT-20260901.md) (seven auditors + a judge; 38
candidate causes → 12 clusters; three items of the top 8 confirmed in review).

## 1. Rationale (verdict)
> The problem is not the worker but the ruler and the gate. R16's first priority is not to retrain the
> worker but to repair the constitution.

The last two layers of the five-layer root-cause chain R12→R15 (qualification, growth engine) rest on three
distorted readings: the readiness-table thresholds are unreachable by engine arithmetic (a constant 0), the
deployment probe reads post-mortem values (fake zero growth), and exam argmax is disconnected from
training sampling (fake refusal to dive: same worker, same seeds, kills 35.1 vs 47.4).
The real constraints are: an exploration macro with a 12-cell field of view (reachable roster 40-65%), a
scene budget that chases the worker downstairs after 1800 steps, and accounting that tells the worker
"staying alive is worth nothing and grinding levels does not pay".

## 2. Constitutional clauses (all off by default; the old path staying bit-identical is a launch precondition)
| Clause | Cluster | Rule | Where |
|---|---|---|---|
| 1 Ruler | C1 | readiness table v0.2: worked back from engine arithmetic, L2=(clvl2/AC9/dmg6), HP column removed (a deterministic function of clvl); the `readiness-v3` coach | worker_env |
| 2 Gate | C2 | cleared share switched to a kill-based definition kills/(kills+alive−golem), golem slots removed; escape threshold **1.0** (G0-3/4/5 ruling: at 0.75 the monsters left on the level block the a11 main line, 39 stalls in 57 windows; dive at once when no live monster is left, and unreachable leftovers are covered by the exhaustion flag) | worker_env / probe v3 |
| 3 Exam rule | C3 | `--worker-decoding sample` (seeded per episode, single-threaded, reproducible) in exam sheets and the deployment probe; reset layer_clock when a FARM window opens (`reset_layer_clock_on_window`) | eval_assembled / options_env |
| 4 Accounting | C7/C8 | `worker_hp_loss_price` (asymmetric pricing of HP loss), `worker_potion_pickup_bonus`, an a13 logit prior, potion autonomy opened; `worker_no_progress_timeout_credit=zero` (timeout ≠ death); economy v4 (anti-idling counted in micro-steps and reset on kills) | worker_env / leashed_ppo / env |
| 4″ Passing the zero-timeout domain down | C8 | `timeout_credit=zero` tripped two frozen training rules (twice in G0): the worker_env audit passes the domain down through info under the key `no_progress_timeout_credit`; leashed_ppo `_update_info_buffer` gets a zero-penalty branch (a timeout still ends the window, the split accounts stay zero, reward == wage, the three death accounts are zero), and the summary condition of `validate_worker_onpolicy_pg_receipt` is relaxed from strictly negative to non-positive; the old branches / key sets are unchanged | worker_env / leashed_ppo |
| 5 Field of view | C5/C6 | `explore_global_fallback` (whole-map BFS fallback when a10 has no candidate in the window; team A measured that it almost never triggers on 8 seeds, 375→375 kills); **`explore_global_hunt` (when no monster is visible in the window, advance toward the nearest live monster by whole-map BFS; the audit's positive control reproduced 375→670 kills, clvl 2→3): this clause is the main force of R16**; `progress_far_tiles` (anchor-set semantics: only new cells at Chebyshev distance ≥ far from the start and from every earlier progress anchor count as progress); `farm_scene_cap` configurable | env / options_env |
| 5′ Declaration of extra authority | — | hunt/fallback both call `bridge.local_map(radius=112)` a second time on the same observation to run a whole-map BFS; the worker's observation (a 25×25 window) cannot see the whole map, so this is environment-side privileged information of the same kind as a11; the documented clause of macro a10, "no second re-plan on the same observation", is explicitly waived under these two switches | written into the pre-registration |
| 4′ Economy v4 | C8 | v4 = all fields of v2 from the same source + `idle_counts_micro_beats` + `idle_reset_on_kill`; the anti-idling threshold stays at 300 but its unit becomes micro-steps (≈60 seconds of game time without a kill → farm×0.5); six runs G0-3…6 showed no anomaly, so the magnitude stays | env |
| Protocol | C4/C6/C11 | end-of-episode metrics sampled at the step before death + reports stratified by survival; episode length ≥6000 steps (contract max_steps whitelist); lethal seeds registered, not removed (reports stratified) | probe / train_ppo |

## 3. Anchoring method (R16 = a new world)
- Under the default flags the old-rule anchors (r10-cand/r10-anchor) keep **reproducing bit for bit**:
  that is how "off by default, zero drift" is enforced, running run_r13_rebake.py as before;
- The new-rule anchors are **recast**: the certified worker in the R16 exam configuration (sampled decoding,
  hunt, scene 3600, clock reset, economy v4; the 3000-step protocol unchanged) on six sheets of 128 seeds
  each, filed separately as r16-anchor-* (with meta.protocol.r16_environment as the marker); their numbers
  are **not directly comparable** with the old world.

## 4′. G0 calibration record (2026-09-02)
| Run | Variable | live DIVE windows | descend | stall | per 10 windows | Result |
|---|---|---|---|---|---|---|
| G0-1/2 | everything on | — | — | — | — | tripped two frozen training rules (zero timeout credit); rules fixed |
| G0-3 | everything on, cleared 0.75, far 3 | 57 | 4 | 39 | 0.70 | fail |
| G0-4 | far 0 | 41 | 4 | 21 | 0.98 | fail (stalls halved) |
| G0-5 | a13 0 | 60 | 2 | 33 | 0.33 | fail (a13 is not the cause) |
| G0-6 | cleared **1.0**, far 0 | 26 | 3 | 4 | **1.15** | **pass**; s=0.181 → **T=325,632** |
Ruling: clearing escape threshold 1.0 (mechanism of the stalls caused by leftover monsters blocking the
path: triggers/monsters are visible across the whole level, so the stall is not about unknown stairs); C6
distant progress shelved (far=0; the v4 micro-step anti-idling already closes the pacing loophole); a13
prior 2.0 kept; HP pricing 0.1 kept (−1.9 per window, within 30% of the wage); potion pick-up bonus 2.0
kept.

## 4. Recipe (v0.3, calibrated at G0-6, T=325,632)
Arm r16-arm-a (the same recipe as run_r16_g0.py except total-steps): continued training of the certified
worker; coach readiness-v3; economy v4; scope farm-dive-v1; max_steps 6000; escrow 0.5×d^1.6 +
readiness-conditional escrow (kept); a11 prior 2.0; a13 prior 2.0 (the same magnitude as a11);
hp_loss_price 0.1 (losing ~80 HP from full ≈ 8 units ≈ 30% of the L1 death penalty of 26: losing HP hurts
before dying does); potion_pickup_bonus 2.0; timeout_credit=zero; explore_global_hunt (main force,
fallback off); progress_far_tiles 0 (C6 shelved); farm_scene_cap 3600 (= the old 1800 × the episode-length
factor 2); reset_layer_clock_on_window; drink_sovereignty open; readiness-v3 clearing threshold 1.0;
training decoding samples as usual.
G0-6 calibration: s=0.181, T=round2048(266240/(1−0.181))=**325,632**; HP pricing −1.9 per window.
Exam sheets: the four gates keep the 3000-step eval_assembled protocol; the new-rule anchors are filed
separately with `--worker-decoding sample` + the R16 environment flags (meta.protocol.r16_environment);
deployment probe v3 in the sixth-parameter JSON form (readiness-v3 / autonomy on / v4 / hunt / cap 3600 /
clock reset), max_steps 6000, sampled decoding, 16 seeds 2114000-2114015; the arm is compared against the
certified worker (the new-rule deployment anchor) in the same form.
Launch chain (driver scripts, not published): run_r16_anchors.py (six new-rule anchor sheets) →
run_r16_arm_a.py 325632 (training → gate D → six sheets → analyze_arm_gates.py <arm> r16-anchor →
deployment probe v3 for arm and anchor).

## 5. Criteria (revised)
- Main metric: the deployment probe's (sampled decoding, sampled before death, stratified by survival)
  clvl/AC/kills/depth normalised per thousand steps + survival rate, against the R16 new-rule anchor;
- The four gates stay but are **against the new-rule anchor**: A superior survival, B depth retention (the
  anchor becomes the R16 scripted worker), C anti-forgetting, D behavioural identity; the old anchor of gate
  B (reckless depth) is abolished per audit C11.

## 6. Work split
Team A: env.py (field-of-view fallback / distant progress / economy v4); team B: eval_assembled + probes
(sampled decoding / sampling before death); team C: leashed_ppo + train_ppo (a13 prior / wiring every CLI
contract whitelist); core team: worker_env + options_env (v0.2 table / kill-based clearing / HP accounting
/ timeout rule / scene budget / clock reset), already landed, enforcement file 12/12 green.

## Freeze statement
The four teams merged (the deliveries of A/B/C and the core team are all in the ledger) + the suite all
green (876 passed, the version before the re-bake; the final-byte suite and the bit-for-bit re-bake of the
4×128 old-rule anchors are enforced by r16_gauntlet.sh, and PASS is a launch precondition) + G0-6
calibration (T=325,632). This file is frozen as r16-PREREG-FROZEN-20260902.md with its sha256 in the
ledger (not published); after freezing, any change to the recipe needs an amendment in a separate file.

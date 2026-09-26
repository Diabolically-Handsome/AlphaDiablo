# R18-A pre-registration: the retreat interface retreat-v1 and Gate T0″ (2026-09-06)

## 0. Background

T0′ (`r17-T0-PRIME-VERDICT-20260906.md`) established that after the funding chain was fixed and the ready share went from 50% to 18%, the L2 hazard rate did not fall; the six readiness rules do not separate life from death.
Design diagnosis: **the problem is not the worker; the manager does not tell the worker clearly what to do**.
On L2 the manager has only one word (FARM), and the worker has no hands (it cannot retreat, cannot select targets and has no drinking rhythm).
Design decision: the interface is incomplete, so a town-return interface is added now.

This pre-registration was frozen after construction was complete and the gates had passed, before the probe was launched. R22/R23 stay parked; the readiness ruler is not recalibrated; no training arm is launched.

## 1. Interface (R18-A retreat-v1; default off = unchanged bit for bit)

1. **Engine gate** (`src/resource_protocol.hpp`): a new branch for main levels, `source ≥ 2 → target = source − 1` with message WM_DIABPREVLVL,
   `accepted = retreatEnabled && retreatAuthorized`, receipts `retreat_ascent` / `retreat_not_authorized`; when off it is still the old default
   `unauthorized_transition`. `configure_retreat(authorized)` can only authorize on main L2+; reaching the upper level consumes the authorization and does `retreats_started += 1`;
   at most 3 per game; **each completed retreat gives the loot-selling economy one more town trip** (`MaxLootServiceTrips + retreats_started`; the old cap of 2 when off).
   The observation adds `retreat_enabled` (always present) and `retreat_authorized / retreats_started / max_retreats` (only when on).
   On the Python side a failed identity check turns it off (`validate_native_retreat`).
2. **Manager**: the action space is unchanged (Discrete(3)). When on, `action_masks()` fires the retreat rule on L2+ and narrows the mask to RESUPPLY only
   (the same mechanism as the town trip); any window closes on the tick the retreat rule fires (`retreat_trigger`).
   **Retreat rule v1** (HP is used only for drinking and the town-return decision, readiness rule 3): `hp ≤ 0.5·max_hp`, or no potions and `hp ≤ 0.75·max_hp`; the pressure trigger is off by default.
3. **Worker**: the scripted `RetreatService` (`python/diablogym/resource_retreat.py`) walks to the up stairs (monster-avoiding pathing first, plain pathing if that fails),
   drinks on the way when `hp ≤ 0.4·max_hp` and potions are available (at least 20 ticks apart), and on the stairs waits for the engine trigger; on reaching the upper level it hands control back with `("complete",)`;
   failures (stairs missing/unreachable/refused 8 times/900-tick cap/unexpected scene) also hand back with `("complete",)` and cool down for 300 ticks: **a failed retreat never ends the game**.
4. **After the upper level**: the existing coach and town trip continue as before (HP < 80% counts toward the native shortfall → if town trips remain, return to town to heal/buy potions; once ready, DIVE back to L2).
5. `WorkerWindowEnv` explicitly rejects retreat-v1 (training windows, escrow and receipts are wired after the probe).

## 2. Gate T0″ (zero-training mechanism probe)

Driver `r10-staging/run_t0_double_prime.py` (not published); pool 2_133 (2133000–2133047, a consumed pool, same-seed pairs); worker 7e31dc54;
clock completion-l2-v1 (12000 real ticks to arrive / 6000 observation denominator / 1800 of observation after the first descent); decoding: sample; manager = the resource coach coach-v03.

| Group | Configuration |
|---|---|
| (e) t0pp-loot-retreat | (d′) + `resource_retreat=retreat-v1` |
| (d′) t0pp-loot-noretreat | exactly the configuration of T0′ t0p-full-loot, re-run on the R18-A bytes |

### Criteria (frozen before launch)

1. Paired survival (e) − (d′): saved − lost ≥ +6/48, and the one-sided UCB95 of the death-rate difference < 0;
2. L2 deaths of (e) ≤ 0.7 × (d′);
3. Mechanism (reported, not judged): share of retreat attempts reaching the upper level ≥ 0.6, share dying during the retreat ≤ 0.25; with the hp0 distribution, trigger types, median duration and the number of games with a second retreat;
4. Regression: the (d′) re-run rows (minus the two new None keys) are bit-identical to the T0′ t0p-full-loot rows; **otherwise the whole probe is void (VOID)**, and "off = unchanged bit for bit" is investigated first.

Verdict: 1, 2 and 4 all met → T0″_PASS (allows drafting R18-B: wiring retreat into training windows and receipts); 4 not met → VOID; otherwise → FAIL.
On FAIL, triage by the mechanism items: if the arrival share is high but survival does not rise → the lever is in "what to do after retreating" (recovery on the upper level/descending again) and the trigger timing; if the arrival share is low →
the lever is in the retreat execution (route choice, being chased). Both go into the verdict, but **the thresholds must not be changed and re-run for this on the same day** (a threshold change = a new pre-registration).

### Known limits of the time and observation window

Only 1800 ticks are observed after the first descent, while one "retreat → town → descend again" takes about 1000–2000 ticks, so this probe mainly tests "does retreating turn deaths on L2 into survival",
not "can the agent make progress after retreating and descending again". The latter needs a new clock recipe (an R18-B topic, outside this pre-registration).

## 3. Certification chain (all must pass before launch)

- the full pytest suite (including the new `tests/test_resource_retreat.py`) with 0 failures;
- probe regression: the `readiness-v3` deployment row sha equals 33023de1… (probe version `r17-deployment-v3-r18a-retreat`);
- the two-way re-bake (new bridge 4/4, July bridge 4/4) passes on the final R18-A bytes;
- the sha256 of this file is recorded in the ledger; after launch the protocol source files may not change (a change voids the re-bakes/controls/probes in progress).

## 4. Seeds and files

2_133 consumes 2 more groups (8 in total); the virgin pools 2_116–119 and 2_126–128 are untouched. New files: this pre-registration, `run_t0_double_prime.py` (not published),
`python/diablogym/resource_retreat.py`, `tests/test_resource_retreat.py`, `t0-double-prime/` (not published).
Changed protocol files: `src/resource_protocol.hpp`, `src/diablogym.cpp`, `python/diablogym/{resource_protocol,env,options_env,worker_env,resource_sustain_loot}.py`,
and the probe `probe_r17_deployment.py` (row keys: only `retreats_started` and `retreat` added under `resource`, None when off).
Pre-change copies were kept in a local work directory (not published).

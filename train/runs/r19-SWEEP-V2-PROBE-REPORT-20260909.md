# R19 — sweep-v2 (chests, barrels **and sarcophagi**, main L1 + main L2)

**Date** 2026-09-08 (night) · **Implementer** r19-sweep-v2-implementer
**Tree** `/home/laure/r17_work/r19/sweep-tree` (Python only; `build -> /home/laure/r17_work/r17-1/build-res`)
**Patch** `/home/laure/r17_work/r19/sweep2.patch` (`diff -ru` against `/home/laure/r17_work/r19/sweep-base`, a pristine rsync of the main tree)
**Main tree `/home/laure/AlphaDiablo/diablogym` was never written to** except the append-only ledger.

Chairman's ruling 2026-09-08: *chests and barrels are a key income mechanic and must be
swept properly on L1 **and** L2.*

---

## 0. Headline, in one paragraph

`sweep-v2` is registered, byte-identical to `sweep-v1` when off, and the OFF-arm
regression reproduces the certified control row **exactly**
(`rows_sha_v3 = 0e5a1acd…`, 48/48 rows, 0 runtime errors). On the 48-seed pool the ON arm
opens **408 chests (+61%), 438 barrels (+90%) and 392 sarcophagi (0 before)**, banks
**+15% gold collected** and **+67% median gold at the end**, and is not measurably more
dangerous (paired McNemar saved 9 / lost 6, net **+3 alive**, one-sided 95% UCB on net
harm **0.0694**). **But every one of the 268 sweep windows opened on L1 and not one opened
on L2** — §6 measures exactly why, and it is the trigger law, not the scope law. The L2
half of the ruling is therefore *implemented and inert*; the fix belongs to the window law,
which tonight's brief forbade me to move.

> **REVIEW ROUND, 2026-09-09.** A reviewer returned this report **fix-first**
> (2 high, 3 medium, 2 low). The code was fixed, the tests re-run and all three
> probe arms re-measured on the fixed tree. **§10 is the authoritative record of
> what changed and what the numbers are now.** Sections 1-9 below are kept
> unedited as the first-pass record except where a line is explicitly struck;
> where §5 and §10 disagree, **§10 governs** — §5's sweep counters were taken
> before the booking defect in finding 2b was repaired.

---

## 1. The law

`resource_sweep ∈ ("off", "sweep-v1", "sweep-v2")`. One flag, three values. Every version
difference lives in **one table**, `SWEEP_LAWS` in `python/diablogym/resource_sweep.py`;
nothing else in the module is version-aware.

| law row | sweep-v1 (frozen) | sweep-v2 (R19) |
|---|---|---|
| `floors` | `(1,)` | `(1, 2)` |
| `taken_kinds` | chest, barrel | chest, barrel, **sarcophagus** |
| `skipped_kinds` | chest_trapped, **sarcophagus**, barrel_explosive | chest_trapped, barrel_explosive |
| `microstep_budget` (per episode) | 900 | **2400** |
| `max_targets` (per episode) | 16 | **48** |
| `floor_keys` (memory keys carry the floor) | `False` | **`True`** — see §4 |

**Unchanged in both**, because no threshold of another law moves tonight:
`SWEEP_MONSTER_ABORT_RADIUS = 3`, `SWEEP_HP_ABORT_FRACTION = 0.5`,
`SWEEP_WINDOW_COOLDOWN = 60`, `SWEEP_EMPTY_WINDOWS = 3`, `SWEEP_OPEN_ATTEMPTS = 6`,
`SWEEP_WALK_REJECTIONS = 8`, the window-boundary trigger (`cleared` or the 3600-beat FARM
cap, plus a remaining town-trip slot), the lit-object memory, the receipt booking law, and
the mask collapse to RESUPPLY.

**The per-window cap.** There is no independent per-window number in either version: a
window is capped at whatever remains of the *episode* budget at the moment it opens
(`start()` / `command()` freeze `_window_cap = budget_remaining`). It therefore cannot bind
before the episode budget does, and it is **not** binding in the data: 0 of 48 ON-arm
episodes exhausted the 2400-microstep budget (2 of 48 exhausted 900 under v1).

**Sarcophagus.** Taken by v2. It may release a skeleton; the monster-near abort is
**unchanged**, and the release is measured rather than assumed.

> ~~`monsters_near_after_sarcophagus` counts the increase in alive monsters within 4
> tiles of the sarcophagus tile between the beat we issue the first `open` on it and the
> beat the engine reports it opened. Across 392 opened sarcophagi on 48 seeds it fired
> **1**.~~ **STRUCK (review round, finding 2).** The delta was read inside
> `_book_operated`, which `_command` reaches only *after* the `sweep_monster_near` early
> return — so a skeleton released inside the 3-tile abort ring ended the window one rung
> above the measurement, and the object was instead booked on the first command of a
> *later* window, a beat that by construction had already passed the radius-3 check. The
> only spawn the metric could ever see was one landing in the single-tile ring at
> distance exactly 4. **"1 across 392" was an artefact, not a safety result.** The
> measurement now happens on the first beat after an *accepted* operate, before the abort
> ladder, and reports its own denominator — see **§10.4**.

Trapped chests and explosive barrels stay skipped in both versions (melee suicide) and are
still counted in `targets_skipped_by_kind`.

---

## 2. Every changed site (11 source files + 1 new test file)

| file | change |
|---|---|
| `python/diablogym/resource_sweep.py` | `SweepLaw` / `SWEEP_LAWS`, `SweepService(protocol=…)`, floor gating, floor-scoped keys under v2, the R19 telemetry |
| `python/diablogym/resource_protocol.py` | `RESOURCE_SWEEP_PROTOCOLS` gains `"sweep-v2"`; the two `validate_sweep_protocol` messages become f-strings so they name the value actually asked for |
| `python/diablogym/options_env.py` | build the service with `protocol=self.resource_sweep`; **add `and not (sweep is not None and sweep.active)` to the portal and retreat start guards** — see §3 |
| `python/diablogym/worker_env.py` | escrow settlement: a sweep close on main L2 that is **death-equivalent** forfeits, exactly like a portal close — see §3 |
| `python/diablogym/resource_retreat.py` | `danger_reason()` **extracted** from `trigger_reason()` (no threshold, clause or order changed) so the settlement can ask the retreat law its verdict without copying a number — the house law `test_the_danger_clauses_have_exactly_one_home` |
| `train/eval_contract.py` | the archive-identity table accepts `sweep-v1` **or** `sweep-v2` for `resource_sweep` (the other two laws' vocabularies untouched) |
| `train/train_ppo.py` | `_validate_args` and the CLI `choices=` now read `RESOURCE_SWEEP_PROTOCOLS` instead of a copied literal |
| `train/eval_assembled.py` | same, and the law-pinning loop names the value actually passed |
| `train/migrate_loot_candidate.py` | `SWEEPS` gains `"sweep-v2"` |
| `tests/test_resource_sweep.py`, `tests/test_r18b6_training_wiring.py`, `tests/test_r18b3b_loot_warm_start.py` | the vocabulary assertions and the three "unregistered name" probes moved from `sweep-v2` to `sweep-v3`; the B6 escrow ruling amended (§3) |
| `tests/test_r19_sweep_v2.py` | **new**, 41 tests |

`python/diablogym/env.py`, `python/diablogym/worker_env.py`'s validator, and
`train/runs/r10-staging/probe_r17_deployment.py` needed **no** vocabulary edit: all three
pass the value through the single shared `validate_sweep_protocol`, and
`_R16_ENV_KEYS` already carried `resource_sweep`. `PROBE_VERSION` is **unchanged**
(`r17-deployment-v3-r18m2`) — deliberately, because the certified control row names it.

> **REVIEW ROUND (2026-09-09) — the same list, after the fixes.** The patch
> `/home/laure/r17_work/r19/sweep2.patch` is now **13 files**: `options_env.py`,
> `resource_protocol.py`, `resource_retreat.py`, `resource_sweep.py`, `worker_env.py`;
> `tests/{test_r19_sweep_v2,test_resource_sweep,test_r18b6_training_wiring,test_r18b3b_loot_warm_start}.py`;
> `train/{eval_assembled,eval_contract,migrate_loot_candidate,train_ppo}.py`.
> **`src/` is byte-identical to the baseline** (`diff -rq`: no differences) — the engine
> and the C++ were never touched, before or after the review. The four files changed by
> the review round itself are `resource_sweep.py`, `options_env.py`, `worker_env.py`
> (a comment and a documented gap only) and `tests/test_r19_sweep_v2.py`. §10.1.

---

## 3. Two invariants sweep-v2 breaks, and the fences

Both were written down by R18 as *provable* facts. They were provable **because the sweep
was main-L1 only**. sweep-v2 is not, so each needed an explicit fence.

### 3.1 The portal / retreat start guards (`options_env.py`)

`options_env.py:1094-1120` carried an R18-M review-round note explaining why the portal
start guard does **not** mention the sweep — "the depths are disjoint by law … Widen the
sweep's depth … and that stops being true — add `and not (sweep is not None and
sweep.active)` here and below." That is exactly what I did, in both guards. Without it a
portal or retreat leg could be *started* on a beat a sweep window already owns the body,
and the RESUPPLY owner chain (`options_env.py:2383`, sweep first) would then never drive
it. Under `sweep-v1` the new term is provably inert (`sweep.active` implies
`dungeon_level == 1`, where both `trigger_reason`s already return `None`), which is one of
the reasons the OFF arm is byte-identical.

### 3.2 The escrow forfeiture (`worker_env.py`)

`worker_env.py:2451-2461` argued that a sweep close can neither *be* death-equivalent nor
*steal* a dangerous close, because the sweep is main L1 and both forfeiting clauses are
main L2+. Under sweep-v2 that is false, and the failure mode is concrete: a v2 window open
on L2 hands back with `sweep_low_hp` at exactly `hp <= 0.5 * max` — which is **one of the
retreat law's three clauses, not "the retreat law's own threshold"** as this sentence
originally claimed (review round, finding 1: `empty_belt` and `pressure` were invisible to
the sweep; §10.1 row 1 is the repair) — and because the **sweep rung of `_win_term` sits above the retreat and portal
rungs**, the close is reported as `sweep_complete`, not `retreat_trigger`. The pending
escrow would then **vest**, re-opening precisely the "dive hurt to half hp, then hand the
body back and cash out" channel the retreat clause was written to close.

Fence: `_sweep_close_is_death_equivalent()`, modelled line for line on
`_portal_close_is_death_equivalent()` — main L2+, not a set level, and the retreat
service's own `danger_reason(raw)` is not None. No threshold is copied into `worker_env`;
`danger_reason` was **extracted** from `RetreatService.trigger_reason` so the numbers keep
exactly one home (the R18-B5 pattern, pinned by
`tests/test_r18b5_hunt_portal_training.py::PortalErrandCloseTests::test_the_danger_clauses_have_exactly_one_home`,
which caught me copying them and is the reason the extraction exists).

The B6 test that asserted the settlement reads none of the three new flags is **amended,
not deleted**: it now asserts the settlement reads `resource_sweep` and the two sweep
reasons and **not** `resource_identify` / `resource_weapon_upgrade`, plus two new tests for
the forfeiting and vesting branches and one for the flag-off short circuit.

Honest caveat: the probe arm runs `descend_escrow_fraction = 0`, so this clause is
**exercised only by unit tests tonight**, never by the probe. It matters the moment a
training arm turns escrow on.

---

## 4. The one place where "byte-identical v1" cost something real

Two dungeon levels share one coordinate space. `sweep-v1`'s per-episode memory
(`_seen_objects`, `_booked`, `_done_targets`, `_operated_tiles`, `_snapshot_tiles`) is keyed
`(kind, x, y)` with **no floor** — and `OptionsEnv.action_masks` calls `sweep.observe(raw)`
on **every** manager beat, at every depth. So a v1 episode that descends records L2 objects
into an L1-only sweep's memory, and an L2 chest at `(66,74)` marks the L1 chest at
`(66,74)` as "already seen" (and, symmetrically, an object booked on one floor retires the
other floor's object at that tile).

**This is a real sweep-v1 defect.** I found it by writing the correct key and then failing
the regression: with the floor in the v1 key, **4 of 48 OFF-arm rows moved**
(seeds 2133003, 2133008, 2133031, 2133034 — e.g. 2133031 went from alive-at-L1 to dead-on-L2,
and `objects_seen_lit` rose 43 → 78 on that seed because the cross-floor collisions
stopped merging distinct objects).

Byte-identity is a hard rule tonight, so the floor lives in the key **only under a law row
that says so** (`SweepLaw.floor_keys`: `False` for v1, `True` for v2). sweep-v1 keeps the
defect and keeps its certified rows; sweep-v2 does not have it. **I am reporting the v1
defect rather than silently repairing it** — repairing it re-certifies every v1 row and is
the chairman's call, not mine.

---

## 5. The evidence  *(first pass, 2026-09-08 — superseded by §10 where they differ)*

Probe `r17-deployment-v3-r18m2`, worker `7e31dc54`
(`/home/laure/AlphaDiablo/diablogym/train/runs/r16-arm-a-constitution/model_candidate.zip`),
pool 2_133 seeds 2133000-2133047 in 3 shards (a/b/c), `max_steps 6000`, decoding `sample`.
Driver `/home/laure/r17_work/r19/sweep2_probe_driver.py` (`paired()` and `arm_stats()`
copied verbatim from `/home/laure/r17_work/r18/m2_probe_driver.py`).
Rows: `/home/laure/r17_work/r19/sweep-probe/{sweep2-off-rows.json,sweep2-on-rows.json}`;
aggregates `sweep2-analysis.json`; logs `sweep2-{off,on}-{a,b,c}.log`.

### 5.1 Regression (the gate)

| arm | overrides | `rows_sha_v3` | equal to control? |
|---|---|---|---|
| **OFF** | the 16 certified v1-world keys, **unchanged** (`resource_sweep=sweep-v1`) | `0e5a1acd2fb2c07cd27eb72f9ecc574322e72f362ef0376b692376d9a99be886` | **YES** |
| ON | the same 16 with `resource_sweep="sweep-v2"` | `b9cd9634498e02220e8b16dda6d7b0a1f79c5274ac63d4cf3f1660f46b182885` | (arm) |

48 rows each, **0 runtime errors** in either arm. Paired OFF vs control: 0 saved, 0 lost, 0
discordant — the OFF arm is the control row, row for row.

*Provenance note.* Before the §4 fix the OFF arm produced `066bf39d…`. I did **not** assume
that was my delta: I built a pristine baseline tree (`/home/laure/r17_work/r19/sweep-base`,
the same rsync + the same `build -> r17-1/build-res` symlink, no R19 code) and ran the same
arm on it. It reproduced `0e5a1acd…` exactly, which proved the mismatch **was** mine and
sent me to §4. (For the record, the certified control row was minted in
`/home/laure/r17_work/r18/merge-tree`, whose `build` points at `merge-build` — a different
`.so` (`c7efb32f…` vs `f5ad9916…`) — and whose `worker_env.py` predates the R18-B7 prefix
guard. Neither difference moves this row: the main tree + `r17-1/build-res` reproduces it.)

### 5.2 The sweep itself, ON vs control (48 seeds)

| metric | control (= OFF) | ON (sweep-v2) |
|---|---|---|
| chests opened | 253 | **408** (+61%) |
| barrels smashed | 231 | **438** (+90%) |
| sarcophagi opened | 0 | **392** |
| sweep windows | 104 | **268** |
| targets attempted | 541 | 1310 |
| accepted `open` / rejected | 491 / 0 | 1250 / 0 |
| distinct drop identities seen | 243 | **553** (+128%) |
| objects seen lit | 2803 | 4363 |
| sweep microsteps (total / median per episode) | 19356 / 418.5 | 49019 / **1267** |
| **share of all beats spent sweeping** | **3.60 %** | **8.55 %** |
| episodes that hit the target cap | 27 of 48 | **5 of 48** |
| episodes that exhausted the budget | 2 of 48 | **0 of 48** |
| skipped by kind | sarcophagus 487, barrel_explosive 285 | barrel_explosive 285 |
| window end reasons | monster_near 68, target_cap 27, no_reachable 7, budget 2 | **monster_near 234**, no_reachable 29, target_cap 5 |

**Where the gain came from.** *(Confirmed by a dedicated arm in the review round: see
**§10.3**, which runs `sweep-v1` with the cap raised to 48 and nothing else.)* Not the
microstep budget — that was never binding. It came from the **target cap**: v1 hit 16 targets in 27 of 48 episodes, and once a v1 episode is
at the cap `trigger_reason` returns `None` forever, so the sweep is finished for the
episode. Raising it to 48 let the law keep re-opening windows (104 → 268) and cut cap-bound
episodes to 5. The sarcophagus is pure addition.

**By floor.**

| floor | ON windows | chests | barrels | sarcophagi | microsteps |
|---|---|---|---|---|---|
| L1 | **268** | 408 | 438 | 392 | 49019 |
| **L2** | **0** | 0 | 0 | 0 | 0 |

**Lit but unswept at end of episode** (objects this episode saw lit, still openable, never
booked by us), summed over 48 seeds.

> **NOT COMPARABLE ACROSS ARMS (review round, finding 3).** The memory key drops the floor
> whenever `law.floor_keys` is False — i.e. for the whole OFF arm — while `observe()` runs
> at every depth. OFF's counts therefore merge L1/L2/L3 objects that share a tile and are
> systematically depressed; ON's are floor-scoped and are not. The two columns below are
> two *measurement definitions*, not two behaviours. Each arm's own numbers stay valid
> within that arm. **§10.5** gives the comparison that is real.

| arm | chest | barrel | sarcophagus | barrel_explosive (law-skipped) | episode-level reason (episodes) |
|---|---|---|---|---|---|
| OFF (v1) | 464 | 655 | 658 | 346 | cap 27, never_idle 10, monster_near 9, budget 2 |
| ON (v2) | 481 | 652 | 416 | 541 | never_idle 28, monster_near 15, cap 5 |

~~Read that carefully: v2 opens far more, yet leaves *about as many* chests and barrels
standing. The population of lit objects grew (more windows ⇒ more of the floor lit)~~
**— the "more of the floor lit" reading is WITHDRAWN (finding 3): part of that growth is
the key change, not more light.** What survives the review is the reason ladder, which is
within-arm and therefore sound: the binding constraint moved from `cap` (27 episodes) to
`never_idle` (28 episodes) — the sweep simply never got a legal, idle, monster-free beat
with those objects reachable.

~~`monsters_near_after_sarcophagus` across 392 opened sarcophagi: **1**.~~
**STRUCK — see §10.4 for the measured figure.**

### 5.3 Safety, depth and money

| metric | control (= OFF) | ON (sweep-v2) |
|---|---|---|
| alive at end | 20 / 48 | **23 / 48** |
| deaths by floor | L1 7, L2 19, L3 2 | L1 7, **L2 17**, L3 1 |
| reached L2 / L3 | 34 / **4** | 33 / **1** |
| L2 beats | 85578 | 78834 |
| **L2 hazard per 1k beats** | 0.2220 | **0.2156** |
| L2 kills | 894 | 758 |
| L1 kills | 4780 | 4943 |
| first-descent beat (median, n) | 5535 (41) | 5443 (41) |
| micro_steps (median) | 12000 | 12747 |
| gold collected (total / median) | 12178 / 282.5 | **14023 / 324.0** |
| gold sold (total) | 7591 | 9064 |
| gold spent (total) | 15442 | 16572 |
| **gold at end (median / total)** | 78 / 6855 | **130 / 8918** |
| purchases (potions + armour + weapons) | 161 | 169 |
| loot collected (items) | 156 | 184 |
| loot trips | 105 | 109 |
| retreats | 71 | 64 |

**Paired McNemar on alive-at-end, ON vs the certified control** (formula from
`m2_probe_driver.paired()`): **saved 9, lost 6, net +3, discordant 15, one-sided 95% UCB on
net harm 0.0694.** ON vs OFF is the same pair of arms and gives the same numbers. Read
plainly: three more survivors, well inside noise — the honest claim is *"not measurably
more dangerous"*, not *"safer"*.

**The one clear cost is depth.** L3 reach falls 4 → 1 and L2 beats fall 85578 → 78834: the
pair spends 8.55% of its beats sweeping L1 instead of 3.60%, and gets deeper less often. If
depth is the thing the campaign is buying, that is the price on the ticket.

**No purchase-side blow-up.** Potions and other purchases move 161 → 169 while gold
collected moves +15%, i.e. the extra income is mostly *banked* (median wallet 78 → 130),
not converted. Whether that is good depends on the spend law, which is another fleet's
tonight.

### 5.4 Seed 2133010, as instructed

| | control | OFF (v1, mine) | ON (v2) |
|---|---|---|---|
| chests / barrels / sarcophagi | 7 / 2 / — | 7 / 2 / 0 | **7 / 1 / 5** |
| targets attempted | 10 | 10 | **14** |
| sweep microsteps / budget left | 756 / 144 | 756 / 144 | 707 / **1693** |
| windows | 2 (both L1, both `sweep_monster_near`) | 2 (both L1) | 2 (both L1, both `sweep_monster_near`) |
| skipped by kind | sarcophagus 10, barrel_explosive 14 | same | **barrel_explosive 14** |
| lit but unswept (takeable) | — | 36 (12 chest, 24 barrel) | 72 (20 chest, 16 sarc, 36 barrel) |
| first descent / L2 beats | 5842 / 1292 | 5842 / 1292 | 5418 / 1191 |
| died / depth | yes / L2 | yes / L2 | yes / L2 |
| gold collected / final | 453 / 22 | 453 / 22 | **573 / 2** |

The named symptom is fixed: the 10 sarcophagi this episode walked past are no longer
law-skipped, and 5 of them were opened. The 14 explosive barrels are still skipped, as
ruled. The episode still dies on L2 — the sweep did not save it, and its budget was
**nowhere near** spent (1693 of 2400 left): on this seed the binding law is the
monster-near abort, twice, not any cap.

---

## 6. The finding that matters most: **not one L2 window ever opened**

The scope law works — `SweepService(protocol="sweep-v2")` fires on L2 in the unit tests,
`start()` accepts L2, and the abort law holds there. But across 48 seeds, **0 of 268
windows opened on L2**. That is a *trigger* failure, and it is measurable:

| fact (ON arm, 48 seeds) | value |
|---|---|
| episodes that reached L2 | 33 |
| distinct L2 floor visits | 84 |
| **median beats per L2 visit** | **503** |
| L2 visits reaching the 3600-beat FARM cap | **2 of 84** |
| **L2 visits that ended with the floor cleared** | **0 of 84** |
| median beats on floor at an L2 death | 549 |
| — for contrast, L1 visits | median 1223 beats; 52 of 206 reach the FARM cap; **57 of 206 cleared** |

The sweep's window-boundary trigger consumes exactly two facts — *floor cleared* or
*`farm_scene_steps >= 3600`* — and on L2 **neither ever becomes true within a visit**. The
pair is on an L2 floor for a median of 503 beats and never clears it; with
`reset_layer_clock_on_window=true` the FARM clock restarts on entry, so 3600 is unreachable
in 82 of 84 visits.

**I did not change the window law** — the brief says to report it with numbers instead, and
these are the numbers. Three candidate remedies, for the chairman, in increasing order of
how much law they move:

1. **A depth-aware trigger fact.** Let the v2 law also fire on a *bounded* L2 fact the pair
   actually reaches — e.g. "N beats on this floor with no alive monster within R", or the
   loot economy's own arrival beat. Smallest change; still a new trigger clause.
2. **In-window preemption** (explicitly deferred tonight): let the sweep interrupt a live
   FARM window on L2 when a lit object is close and the floor is momentarily quiet. This is
   the mechanism `gold-grab-v1` uses in the neighbouring tree, and it is why that law
   reaches L2 at all (1 window) where this one reaches none.
3. **Accept L1-only in practice** and retire the L2 clause until the pair survives long
   enough on L2 for it to be reachable. Cheapest, and honest: today the L2 half is inert.

> **RECOMMENDATION ADDED IN THE REVIEW ROUND (finding 5).** All three options above are
> ways to make the L2 trigger *fire*. None of them should be taken yet, and the reason is
> in this section already: **as the law stands, an L2 window is negative-value on every
> axis the moment it becomes reachable.** (a) Its income is exactly **zero** — the collect
> phase is gated to `dungeon_level == 1`, so nothing it drops is ever picked up; (b) it
> spends episode microsteps on the floor where the pair already dies 17-19 times in 48
> episodes, and the L1-only version already costs L3 reach 4 → 1; (c) until the review
> round it also outranked two of the retreat law's three danger clauses there (finding 1;
> now fixed, but the body-ownership cost of *any* L2 window remains). My recommendation
> to the chairman: **do not enable the L2 trigger by any mechanism until
> `SustainLootService`'s collect phase covers L2, and adjudicate the depth-aware trigger
> and the collect gate as ONE change, not two.**

There is a **second, independent** reason to be careful about L2 sweeping, which I found
while reading rather than measuring: `SustainLootService`'s collect phase is gated to
`dungeon_level == 1` (`resource_sustain_loot.py:323-325`), and `maybe_start` refuses any
floor but L1. So loot dropped by an L2 sweep has **no collect stage that will ever pick it
up**; only what the worker itself walks over would be banked. I kept the v1 trip-slot gate
unchanged (it is the frozen law, and no other law's threshold moves tonight) and left the
by-floor telemetry in place so that the day an L2 window does open, the orphaned-drop cost
is visible instead of assumed.

---

## 7. Tests

New file `tests/test_r19_sweep_v2.py` — **41 tests, all passing**, in seven suites:

* `SweepV2RegistrationTests` (4) — the three-value vocabulary; exactly one `SWEEP_LAWS` row
  per non-off value (a validator that accepts a value nobody can build a service for is a
  validator that lies); the deployment validator on v2's terms; an unregistered name refused
  by **both** the validator and the constructor; the fail-closed native identity both ways.
* `SweepV2LawRowTests` (4) — the frozen v1 row, the R19 v2 row, the **shared, unmoved**
  abort thresholds, and that the default service is v1.
* `SweepV1ByteIdentityTests` (5) — a default service and an explicit `sweep-v1` service
  produce the same commands, the same bookings and the same serialised telemetry on the same
  synthetic raws; v1 still refuses the sarcophagus and still books it as skipped; every R18
  telemetry key survives (additive only); the two law rows are distinct objects.
* `SweepV2ScopeTests` (5) — L1 yes, L2 yes, L3 no, L4 no, town no, set level no; v1 still
  refuses L2; `start()` refuses a floor outside its row; **a window whose floor changes
  under it hands back** (`sweep_unexpected_scene`) — "still legal" is not "still my floor".
* `SweepV2KindTests` (5) — the sarcophagus is targeted, booked and attributed to its floor;
  a released skeleton is counted; trapped chests and explosive barrels are never commanded
  in either version and are counted once each.
* `SweepV2CapTests` (5) — 16/900 vs 48/2400 in both the trigger and the in-window rungs;
  the per-window cap *is* the remaining episode budget; `record_steps` charges the episode
  budget **and** the floor.
* `SweepV2FloorKeyTests` (3) — an L1 object never books the L2 object at the same tile; the
  lit memory is per floor; a target retired on one floor is still live on the other.
* `SweepV2TelemetryTests` (7) — `by_floor` for both floors, the floor on every window
  record, `lit_but_unswept_at_end` counting only what is still openable, an opened object
  never reported unswept, the reason ladder, `gold_from_sweep` null **on purpose** (§8), and
  the reported constants being the law row.
* `SweepV2AbortTests` (4) — the 3-tile monster ring and the 50% hp share still end a window
  **on L2**; the law refuses to open one while hurt or crowded; a hand-back keeps the budget,
  the untouched objects and the 60-beat cooldown.

**Targeted neighbourhood** (`test_r19_sweep_v2`, `test_resource_sweep`,
`test_r18b6_training_wiring`, `test_r18b3b_loot_warm_start`, `test_r18b5_hunt_portal_training`,
`test_resource_retreat`): **312 passed, 677 subtests passed, 0 failed** — including the six
native-bridge tests in `test_resource_sweep.py` that open a real chest and smash a real
barrel against the isolated build.

**Wide regression** — every test file importing `options_env` / `worker_env` /
`resource_sweep` / `resource_sustain_loot` / `resource_retreat` / `resource_portal` /
`eval_contract` / `migrate_loot` (56 files), run in **my tree** and in the **pristine
baseline tree** with the same selection:

| tree | result |
|---|---|
| baseline `/home/laure/r17_work/r19/sweep-base` | **59 failed**, 1767 passed, 3 errors, 1275 subtests passed |
| sweep-v2 `/home/laure/r17_work/r19/sweep-tree` | **59 failed**, 1811 passed, 3 errors, 1293 subtests passed |

The failing set is **identical** (`diff` of the sorted `FAILED`/`ERROR` lines: empty). Every
one of the 44 extra passes is one of my new tests. The 59 pre-existing failures and 3 errors
are stale-checkout artefacts of a partial tree (missing `train/runs` assets), not files I
touched; four further files (`_a14_fuse_receipt_probe.py`, `_progression_probe.py`,
`test_content_case_aux.py`, `test_worker_env.py`) fail at *collection* on both trees for the
same reason and were excluded from both runs.

Logs: `/home/laure/r17_work/r19/wide2-{base,sweep2}.log`,
`/home/laure/r17_work/r19/wide2-{base,sweep2}-fails.txt`.

---

## 8. Telemetry added (additive only; `rows_sha_v3` restricts to the 17 v3 columns, so none
of this can move a certified row)

* `by_floor: {1: {...}, 2: {...}}` — per floor: `chests`, `barrels`, `sarcophagi`,
  `windows`, `microsteps`, `targets`, `items_dropped_seen`, `open_accepted`,
  `skipped_by_kind`.
* `lit_but_unswept_at_end` — `by_kind`, `takeable`, `by_law_skipped`, `total`,
  `takeable_total`, and one episode-level `reason` ∈ `cap` / `budget` / `monster_near` /
  `never_idle`. **Stated honestly in the code and here: that reason is an episode-level
  attribution, read in the order the law itself would have stopped us — it is not
  per-object causation.**
* `sarcophagi_opened`, `monsters_near_after_sarcophagus`.
* `floor` and `sarcophagi0/1` on every window record.
* `floors`, `taken_kinds`, `skipped_kinds`, `floor_scoped_keys` — the law row, reported so a
  row can say which law produced it.
* `gold_from_sweep` — **`null`, on purpose, not by omission.** A chest's gold lands on the
  *floor*; only the loot service's collect stage banks it, and that number is already
  reported as `resource.loot_cumulative.gold_collected`. Any figure booked here would double
  count it. `gold_delta_in_windows` (the wallet movement while the script owned the body,
  summed over windows) is reported instead and is **not** an income measure.

Every R18 telemetry key is unchanged and still present; a reader written against `sweep-v1`
keeps working.

---

## 9. Known gaps

1. **The L2 half of the ruling is inert.** 0 of 268 windows opened on L2 (§6). The law is
   right; the trigger cannot reach it. This is the one thing I would put in front of the
   chairman first — **and the review round's recommendation is that it stay inert until
   the collect gate moves (§6, §10.6): firing it first is a strict loss, not a partial
   win.**
2. **L2 drops would be orphaned.** `SustainLootService`'s collect phase is L1-only
   (`resource_sustain_loot.py:323-325`), so the day an L2 window does open, its loot has no
   collect stage. Not fixed — it is another law's threshold.
3. **sweep-v1 carries a cross-floor memory defect** (§4): its object memory is keyed without
   the floor while `observe()` runs at every depth. Reported, not repaired, because
   repairing it moves 4 of 48 certified v1 rows.
4. **The escrow fence is unit-tested only.** The probe arm runs
   `descend_escrow_fraction = 0`, so `_sweep_close_is_death_equivalent` never fired in the
   48-seed run. It matters the first time a training arm turns escrow on. **And it is
   CONDITIONAL: with `resource_retreat` and `resource_portal` both off it reaches no
   danger law and vests — a cash-out channel sweep-v2 opens. Left failing open on purpose;
   see §10.7.**
5. **No training smoke.** Tonight was Python-only and probe-only; `train_ppo` /
   `eval_assembled` / `migrate_loot_candidate` were registered and unit-tested but no arm was
   launched under `--resource-sweep sweep-v2`.
6. **Depth cost not adjudicated.** L3 reach 4 → 1 and L2 beats −8% are real and consistent
   with spending 8.55% of beats sweeping L1. Whether the +15% gold and +3 survivors are worth
   it is a campaign judgement, not a probe result.
7. **`gold_from_sweep` is unanswered, not answered as zero.** Attributing banked gold to the
   sweep needs the collect stage to tag a pickup with the object that dropped it; the channel
   carries no such tag today.
8. **The `sweep_danger_*` rung has never fired in a probe.** It is unreachable on this
   pool for the same reason the whole L2 clause is: 0 of the ON arm's windows open on L2.
   It is unit-tested against the real `RetreatService` and is, on this evidence,
   *implemented and unexercised* — exactly like the L2 scope itself.
9. **Merge risk with `gold-grab-v1`** (`/home/laure/r17_work/r19/gold-tree`, untouched by me):
   we both edit `options_env.py`'s service chain and `worker_env.py`. In particular that fleet
   also reasons about the RESUPPLY owner chain and the escrow close reasons — the two changes
   in §3 will need to be reconciled by hand tomorrow, not auto-merged.

---

## 10. Review round (2026-09-09)

Reviewer verdict: **fix-first** — 2 high, 3 medium, 2 low. Everything in this section
was re-measured after the fixes; no figure here is carried over from the first pass.
Driver `/home/laure/r17_work/r19/sweep2_review_driver.py`; rows, logs and aggregates
under `/home/laure/r17_work/r19/sweep-probe-rr/`.

### 10.1 What was fixed in the tree

| # | sev | the finding | the fix |
|---|---|---|---|
| 1 | high | A live sweep window blocked `RetreatService.trigger_reason` (`options_env.py:1126-1131`) while the sweep itself mirrored only ONE of the retreat law's three danger clauses (`low_hp`, `SWEEP_HP_ABORT_FRACTION == RetreatPolicy.hp_fraction`). `empty_belt` (belt 0 and hp ≤ 0.75·max) and `pressure` could hold the body on main L2 for a whole budget. The escrow fence already asked the retreat law for **all three**, so the money side and the body-ownership side of one fence disagreed. | The sweep asks the **law itself**. `SweepService.set_danger_law()` binds `RetreatService` (portal as fallback — the exact order `_sweep_close_is_death_equivalent` uses); `OptionsEnv` binds it where the service is built. Two rungs consume it: `trigger_reason` refuses to **open** a window on main L2+ while a verdict stands, and `_command` hands back at once as `sweep_danger_<clause>`. **No threshold is copied into `resource_sweep.py`.** Provably inert under sweep-v1 (`floors == (1,)` vs `SWEEP_DANGER_FLOOR = 2`). |
| 2 | high | `monsters_near_after_sarcophagus` could not observe what it existed to observe: the delta was read inside `_book_operated`, **after** the radius-3 abort return, so the only frame it could ever be read on was monster-free. It reported 1 across 392 opens. | The watch is armed by an **accepted** receipt (a refused operate opened nothing) and resolved by `_resolve_sarcophagus_watches()` at the **top of the next beat, before the abort ladder**, and from `observe()` once the window has closed. The telemetry now carries its own denominator: `sarcophagus_watches_measured` / `sarcophagus_watches_unmeasured`. §10.4. |
| 2b | high | Second order: every `*_opened` count was a **floor, not a count** — an object operated on a window's last beat was booked only by a *later* window's first command, i.e. never if there was no later window. | New law-row field `SweepLaw.book_on_close` (**v2 True, v1 False**): under v2 `_book_operated` also runs at the top of `_end` and on every `observe()`. Deliberately **off for v1** — turning it on moves booked counts and `_done_targets` on certified rows. |
| 3 | med | The OFF-vs-ON `lit_but_unswept_at_end` and `objects_seen_lit` comparison was two *measurement definitions*, not two behaviours. | Struck in §5.2; replaced by the paired L1-only subset in §10.5. |
| 4 | med | The ON arm moved five things at once, so no headline delta was attributable. | Ran the reviewer's decisive arm. §10.3. |
| 5 | med | The report never said that firing the L2 trigger before the collect gate moves would be a strict loss. | Said, in §6 and §10.6. |
| 6 | low | R19 moved `_key(obj, floor)` in front of the visibility test in `_observe_objects` — which runs every beat at every depth — widening a frozen path's crash surface for any native row without integer x/y. | Malformed rows are skipped, not raised, in `_observe_objects` **and** `_book_operated` (which `book_on_close` now reaches from `observe()`, inheriting the same exposure). Pinned by `ObserveRobustnessTests`. |
| 7a | low | The RESUPPLY owner-chain comment still argued depth-disjointness and that "the order carries no behaviour". | Rewritten: the order is now **load-bearing** and ranks the sweep above the retreat on L2; the comment states what makes that safe (the finding-1 rungs plus the escrow fence) and what would break it. |
| 7b | low | The escrow fence returns `False` when neither `resource_retreat` nor `resource_portal` has a policy. | **Not changed** — §10.7. Documented at the `return False` and pinned by a test. |

### 10.2 The gate, re-run end to end on the fixed tree

| arm | tree | overrides | `rows_sha_v3` | verdict |
|---|---|---|---|---|
| **OFF** | `sweep-tree` | the 16 certified v1-world keys, **unchanged** | `0e5a1acd2fb2c07cd27eb72f9ecc574322e72f362ef0376b692376d9a99be886` | **EQUAL to the certified control row** |
| **ON** | `sweep-tree` | the same 16 with `resource_sweep="sweep-v2"` | `b9cd9634498e02220e8b16dda6d7b0a1f79c5274ac63d4cf3f1660f46b182885` | **unchanged** from the pre-review ON arm (`b9cd9634…`) — the fixes moved telemetry and law rungs, not a single command, on this pool |
| **CAP** | `attrib-tree` | the same 16 with `sweep-v1`, and `SWEEP_MAX_TARGETS = 48` in the tree | `18cf062fd1a301ff7366b3184e296d98ddfbe43fa7e2249a5bdc99e29ab20403` | the attribution arm (§10.3) |

48 rows per arm; **0 runtime errors** (OFF 0 / ON 0 / CAP 0). Paired OFF vs the
certified control: saved 0, lost 0, discordant 0 — the OFF arm **is** the control
row, row for row.

The tree was fingerprinted before and after the run and did not change during it
(`rr2-tree-fingerprint-{before,after}.txt`), which is also what the probe's own
`source_changed_during_run` guard reports.

### 10.3 Finding 4 — the decisive attribution arm

The reviewer's objection: `sweep-v2` changes **five** things against the control
(floors 1→(1,2); + sarcophagus; `max_targets` 16→48; `microstep_budget` 900→2400;
`floor_keys` False→True), and the first pass already showed two of them did no work
(0 windows on L2, 0 episodes budget-bound). So run the arm that moves **only the cap**.

`CAP` = the 16 certified overrides with `resource_sweep=sweep-v1`, run in
`/home/laure/r17_work/r19/attrib-tree` — a pristine rsync of the main tree whose only
difference from the certified baseline is one integer, `SWEEP_MAX_TARGETS = 48`
(verified by `diff -r`: one line; no R19 code anywhere in that tree).

| metric | control (= OFF) | **CAP** (v1, cap 48) | ON (sweep-v2) |
|---|---|---|---|
| chests opened | 253 | 331 | 408 |
| barrels smashed | 231 | 329 | 444 |
| sarcophagi opened | 0 | 0 | 396 |
| sweep windows | 104 | 139 | 268 |
| targets attempted | 541 | 731 | 1310 |
| episodes at the target cap | 27 of 48 | 0 of 48 | 5 of 48 |
| episodes that exhausted the budget | 2 of 48 | 25 of 48 | 0 of 48 |
| distinct drop identities | 243 | 342 | 522 |
| sweep microsteps (total) | 19356 | 29423 | 49019 |
| share of all beats spent sweeping | 3.60 % | 5.32 % | 8.54 % |
| gold collected (total) | 12178 | 13226 | 14023 |
| gold collected (median) | 282.5 | 300.5 | 324.0 |
| gold sold (total) | 7591 | 6754 | 9064 |
| gold at end (median) | 78.0 | 91.5 | 130.0 |
| loot items collected | 156 | 152 | 184 |
| purchases | 161 | 162 | 169 |
| loot trips | 105 | 108 | 109 |
| retreats | 71 | 67 | 64 |
| alive at end | 20 of 48 | 23 of 48 | 23 of 48 |
| reached L2 | 34 | 31 | 33 |
| **reached L3** | 4 | 4 | 1 |
| L2 beats | 85578 | 85069 | 78834 |
| L2 hazard per 1k beats | 0.2220 | 0.1881 | 0.2156 |
| L1 kills | 4780 | 4806 | 4943 |
| first-descent beat (median) | 5535 | 5567 | 5443 |
| micro_steps (median) | 12000.0 | 12000.0 | 12747.0 |

**Paired McNemar on alive-at-end vs the certified control** (`m2_probe_driver.paired`):

| arm | saved | lost | net | discordant | one-sided 95% UCB on net harm |
|---|---|---|---|---|---|
| CAP (v1, cap 48) | 6 | 3 | +3 | 9 | 0.0392 |
| ON (sweep-v2) | 9 | 6 | +3 | 15 | 0.0694 |

**How much of the ON arm's move the cap alone buys** — CAP's distance from the control
as a share of ON's distance from the control:

| metric | control | CAP | ON | CAP's share of ON's move |
|---|---|---|---|---|
| chests | 253 | 331 | 408 | 50 % |
| barrels | 231 | 329 | 444 | 46 % |
| gold collected | 12178 | 13226 | 14023 | 57 % |
| sweep windows | 104 | 139 | 268 | 21 % |
| distinct drop identities | 243 | 342 | 522 | 35 % |

**Reading — and the reviewer's hypothesis is only half right.** The cap alone buys about
**42%** of ON's move away from the control on the five headline quantities (chests
50%, barrels 46%, gold collected 57%). It is not the whole story, and the arm says
exactly why: **raising the cap makes the MICROSTEP BUDGET the binding law instead.**

| law | control (v1: cap 16, budget 900) | CAP (v1: cap 48, budget 900) | ON (v2: cap 48, budget 2400) |
|---|---|---|---|
| episodes stopped by the **target cap** | 27 of 48 | **0 of 48** | 5 of 48 |
| episodes stopped by the **microstep budget** | 2 of 48 | **25 of 48** | 0 of 48 |

Under v1 the cap binds in 27 of 48 episodes and the 900-microstep budget in 2. Raise the
cap alone and the cap stops binding entirely (0 of 48) while the budget starts binding in
**25 of 48**. Only when the budget is raised too (ON) does neither bind. So the first
pass's claim that "the 2400 budget was never binding" is true of the ON arm and
**misleading as an attribution**: the budget raise is doing real work — it is what stops
the raised cap from immediately hitting the old budget. The remaining difference is the
sarcophagus kind (396 opened, a kind neither v1 arm can touch) and the floor-scoped keys.
The **L2 scope contributed nothing**: 0 windows opened there.

**The cost, and the part that matters for the campaign.** ON buys its extra half of the
gain with depth: **L3 reach 4 → 1, while the CAP arm keeps it at 4** — and the two arms
end with the *same* number of survivors (23 of 48 each, against 20 for the control). The
sweep's share of all beats goes 3.60% → 8.54% under ON but only 5.32% under CAP, and
CAP's paired UCB on net harm is the tighter of the two (0.0392 vs 0.0694). On this pool the
cap-only arm is the better trade on every campaign axis except raw loot volume; that is a
chairman's call, but it is now a call with an arm behind it.

### 10.4 Finding 2 — the sarcophagus metric now has a beat and a denominator

| ON arm, 48 seeds | first pass | **review round** |
|---|---|---|
| sarcophagi opened | 392 | **396** |
| `monsters_near_after_sarcophagus` | 1 | **76** |
| opens the delta was actually measured on | *not reported; structurally near zero* | **396** |
| watches still unmeasured at episode end | *not reported* | **0** |

The opened count moved because of finding 2b as well as finding 2: an object operated
on the closing beat of a window used to be dropped entirely. Read the danger figure
**with its denominator**: 76 released monsters observed across 396 measured opens
(0.192 per open). The first pass's "1 across 392" is withdrawn as an artefact; this number is a
measurement, and it is the first one this law has. **It is also not a small number:**
at 0.192 released monsters per opened sarcophagus it is a real cost of the new kind, and
the first pass reported the opposite.

**One other figure moved for the same reason, and it moved DOWN.** `items_dropped_seen`
(distinct drop identities) reads 522 in this run against 553 in the first pass. Nothing
about the commands changed — `rows_sha_v3` is identical — but `_count_drops` fires the
beat an object is booked, and `book_on_close` books it on the first beat the engine
reports it opened instead of up to a whole window later. The later reading could bank any
new floor identity that appeared near that tile in the meantime (a monster drop from the
fight that closed the window, for instance). **522 is the tighter figure; 553 was measured
on a frame that had had time to collect other things.** Neither is a clean per-object
attribution, and §8's `gold_from_sweep = null` reasoning applies to it unchanged.

### 10.5 Finding 3 — the comparison that is actually valid

`_lit_state` / `_seen_objects` are keyed by `_key(obj, floor)`, which **drops the**
**floor** whenever `law.floor_keys` is False — the control, OFF and CAP arms — while
`observe()` runs at every depth. Those arms therefore merge objects that share a tile
across floors; the ON arm's keys do not. Cross-arm lit totals are two definitions.

The definition-safe subset is episodes that never left main L1 — and it is only a
*paired* subset if the same seeds qualify in **every** arm, because whether an episode
descends is itself a behavioural outcome. That intersection is **12 of 48 seeds**:

| arm | `objects_seen_lit` | chests opened | barrels smashed | sarcophagi opened |
|---|---|---|---|---|
| control (= OFF) | 425 | 17 | 6 | 0 |
| CAP (v1, cap 48) | 425 | 17 | 6 | 0 |
| ON (sweep-v2) | 565 | 22 | 15 | 23 |

Seeds: `2133002, 2133006, 2133014, 2133021, 2133022, 2133028, 2133031, 2133032, 2133038, 2133039, 2133044, 2133046`.

Two things to read off that table. **(a) `lit_but_unswept_at_end` is absent from the
control and CAP columns entirely** — both run pre-R19 code, which has no such key — so
that metric has no cross-arm comparison at all, valid or otherwise, and the §5.2 table's
OFF column is the *only* place it ever existed for a v1 arm. **(b) CAP is byte-for-byte
the control on this subset** (425 / 17 / 6): on the twelve episodes that never leave L1,
raising the target cap changes nothing, because those episodes never reached 16 targets in
the first place. The cap's entire effect lives in the episodes that descend — which is
also where the L3 cost is paid.

**Caveat, stated rather than assumed.** The native channel is whitelisted to five
dungeon object kinds (`src/resource_sweep.hpp`), so for a seed that never descends the
only other scene visited is town. I did not measure town's object rows, so this is the
**tightest available** comparison, not a proof of zero collision. Full data:
`sweep-probe-rr/rr-l1-paired.json`.

### 10.6 Finding 5 — the recommendation on the L2 clause

§6 lists three ways to make the L2 trigger fire. **None of them should be taken yet.**
As the law stands, an L2 window is negative-value on every axis the moment it becomes
reachable:

1. **Its income is exactly zero.** `SustainLootService`'s collect phase is gated to
   `dungeon_level == 1` (`resource_sustain_loot.py:323-325`) and `maybe_start` refuses
   any floor but L1, so nothing an L2 window drops is ever picked up.
2. **It spends episode microsteps on the floor the pair already dies on** — 19 and 17
   of 48 deaths are on L2 in the control and ON arms — and the L1-only version already
   costs L3 reach 4 → 1.
3. **It contends for the body with the retreat law.** Finding 1 is fixed, so the sweep
   now hands back on any danger verdict; but the rung order still ranks the sweep above
   the retreat on L2, and every beat an L2 window holds is a beat the retreat does not.

**Recommendation to the chairman: do not enable the L2 trigger by any of the three
mechanisms until `SustainLootService`'s collect phase covers L2, and adjudicate the
depth-aware trigger and the collect gate as ONE change, not two.** On tonight's
evidence the shippable half of sweep-v2 is the **cap plus the microstep budget**
(§10.3) — both L1 work, neither needing the L2 clause, the sarcophagus kind or the key
rescoping to deliver what they deliver.

### 10.7 Finding 7b — what I did **not** change, and why

`worker_env._sweep_close_is_death_equivalent` returns `False` when neither
`resource_retreat` nor `resource_portal` has a policy. In that configuration — legal,
never certified, and not the v1-world the probe runs — a sweep-v2 window on main L2 is
the only **non-terminating** close available at `hp <= 50%`, so the pending descend
escrow vests there: a cash-out channel that sweep-v2 opens and that did not exist
before it. Making the fence fail **closed** would reverse the sibling fence's stated
house doctrine (*rather under-forfeit than forfeit an errand*) for a case the chairman
has not ruled on, so I left it failing open, wrote the exposure into the code at the
`return False`, and pinned it with `EscrowFenceScopeTests`. **It is a ruling, not a
fix.**

Also **not** repaired, for the same reason: the sweep-v1 cross-floor key defect (§4)
and `book_on_close` for sweep-v1. Both move certified v1 rows.

### 10.8 Tests after the fixes

* `tests/test_r19_sweep_v2.py` — **63 passed, 16 subtests passed in 0.05s** (41 before the review round). The five new
  suites are `SweepDangerLawTests` (10), `SarcophagusWatchTests` (6),
  `BookOnCloseTests` (3), `ObserveRobustnessTests` (2), `EscrowFenceScopeTests` (1).
  The danger tests drive a **real `RetreatService`**, never a stub — a test that
  re-mirrored the three clauses would pin the defect, not the repair.
* Targeted neighbourhood (`test_r19_sweep_v2`, `test_resource_sweep`,
  `test_r18b6_training_wiring`, `test_r18b3b_loot_warm_start`,
  `test_r18b5_hunt_portal_training`, `test_resource_retreat`): **334 passed, 683 subtests passed in 5.52s**
* Wide regression, the **same 55 test files** run in my tree and in the pristine
  baseline tree `/home/laure/r17_work/r19/sweep-base`:

| tree | result |
|---|---|
| baseline `sweep-base` | 59 failed, 1767 passed, 3 errors, 1275 subtests passed in 269.12s (0:04:29) |
| fixed `sweep-tree` | 59 failed, 1770 passed, 3 errors, 1283 subtests passed in 269.28s (0:04:29) |

  **New failures on my tree vs the baseline: none. Failures that disappeared: none.**
  The 59 pre-existing failures and 3 collection errors are stale-checkout artefacts of
  a partial tree (missing `train/runs` assets) on **both** trees; the failing id sets
  are identical. `tests/test_r19_sweep_v2.py` is not in this list because it does not
  exist in the baseline tree — it is run separately above.
  Logs: `rr2-wide-{sweep2,base}-raw.txt`, `rr2-wide-{sweep2,base}-ids.txt`.


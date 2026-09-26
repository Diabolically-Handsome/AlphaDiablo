# R19-M2 report: merging the four design rulings, probes and attribution

- Date: 2026-09-09 (`by` = `r19-m2-integrator`)
- Where things lived: the work trees, patch, scripts, logs and probe outputs named in this report were kept in a
  local R19 work directory and are not published. They are shown by their paths relative to that directory;
  "main tree" means the main development tree.
- Merge tree: `m2-merge/` (`rsync -a --exclude __pycache__` from the R19-M1 merge tree; the `build` symlink to
  the R17.1 resource build kept as is)
- Patch: `m2.patch` (`diff -ruN` against the **main tree**, **25 files**, **regenerated after the review round:
  +10028 / -215 lines**, all LF, sha256 `315133a531bbc13d`; before the review it was +9208 / -222). 25 = the 21
  entries from the M1 round + 4 new ones in this round (`python/diablogym/resource_emergency_stop.py`,
  `tests/test_r19_m2_emergency_stop.py`, `tests/test_r19_m2_rulings.py`, `tests/test_resource_readiness_law.py`)
- Inputs: implementer A's `m2-tree/` (rulings 1/2/3 + coach-v05) and implementer B's `stop-tree/` (ruling 4's
  emergency stop, stop-v1)
- Read-only check of the other trees: the main tree **only had its ledger appended** (a `find -newermt` from the
  start of this round over the whole main tree matched only `train/runs/r10-staging/r13_ledger.jsonl`; `src/`
  and the merge tree are file-by-file identical under `diff -rq` = `SRC_IDENTICAL`); the M1 merge tree
  `merge-tree/` got **zero writes in this round** (`grep -c coach-v05` = 0, no hit for
  `resource_emergency_stop`)
- Probe outputs: `m2-probe/` (5 arms x 48 seeds x 3 shards = 15 shard jobs, plus the inherited certified control
  rows and the M1 ALL-ON rows)
- **Review round (2026-09-09, `by` = `r19-m2-fixer`)**: a reviewer raised 4 high + 5 medium findings, and **all
  nine were fixed**; the code, tests, probes, patch and this report were all re-run or rewritten. **Where a
  paragraph in sections 2, 8, 10, 11.1, 11.3 or 13 carries a `> Review-round correction` quote block, the review
  round prevails.** The full account is in **section 16**; the review-round probe outputs are in `m2rr-probe/`.
  In one sentence: **the OFF regression still equals the certified control rows on both reruns; M2-SET is
  bit-for-bit unchanged; the two parts of ruling 4 still do not beat M2-SET after their defects were fixed.**

---

## 0. One-page summary

1. **The regression passes.** With every new rule off (the 16-key certified control rows), the merge tree's
   `rows_sha_v3` = `0e5a1acd2fb2c07cd27eb72f9ecc574322e72f362ef0376b692376d9a99be886`, **bit-identical to the
   certified control rows**. 48 rows, runtime errors 0.
2. **Rulings 1/2/3 (M2-SET) are the best arm of the campaign so far**: survival **28/48** (control 20, M1 ALL-ON
   24), L2 hazard **0.1374** (control 0.2220, M1 0.1573), paired saved 14 / lost 6 / net **+8**, one-sided 95%
   UCB = **-0.0186 < 0**. **This is the first time in the campaign that "worse" is excluded at 95%.**
3. **The price is depth.** In the same arm, L2 arrivals drop from 34 to **30**, L3 arrivals from 4 to **2**, and
   clvl >= 4 from 25 to **18**. The three rules buy survival by **staying shallow**. The campaign's north star is
   depth, so this **needs a design decision**; it is not a trade-off the integrator can make.
4. **For the two parts of ruling 4 I added attribution arms, and they point in opposite directions** (same 48
   seeds, same worker):
   - `coach-v05` added alone to M2-SET: survival 28 -> **25**, paired vs M2-SET **saved 0 / lost 3**, hazard
     0.1374 -> 0.1794. **It only loses.**
   - `stop-v1` added alone to M2-SET: survival **28 (unchanged)**, paired vs M2-SET saved 5 / lost 5 / **net 0**,
     but L2 hazard 0.1374 -> **0.1270** (lowest of all), L2 beats +8.1%, and UCB95 against the certified control
     rows = **-0.01087 < 0**. **It changes who survives, not how many: 5 saved, 5 lost.**
   - Both together (M2-FULL): survival 26, UCB95 0.01734, **worse than rulings 1/2/3 alone**.
5. **The counterexample implementer B predicted is confirmed by the probe.** The two named seeds, 2133002 and
   2133047, both survive in M2-SET and **both die once `stop-v1` is added**, and `M2-SET+stop` and `M2-FULL` are
   **field-for-field identical** on those two seeds. The deaths belong to the `stop-v1` half, not to coach-v05.
6. **Ruling 4b (the stall lock) is nearly silent in the tested world**: over 48 games it suppresses only 9-13
   windows, while `forced_through` ("let through because FARM is already masked") happens **70-72** times. DIVE
   windows closed by stall (control 99) are **102** in the coach arm, **no decrease**.
7. **A diagnostic premise needs correcting**: the task brief said seed 2133047 had "12 const_dive windows, all
   closed by stall". On the **certified control rows** I count **11 DIVE windows, 5 of them const_dive, and 0
   closed by stall** (per-window table in section 11.3). Across all 48 control games, `const_dive ∩ stall` is
   only **6 windows** (267 const_dive, 99 stall). **Ruling 4b was designed around a picture that the probe rows
   themselves do not support.**

---

## 1. How the merge tree was built

```
m2-merge  = rsync -a --exclude __pycache__  merge-tree/           (M1 certified probe output)
          + A's 7 exclusive files (taken directly from m2-tree)
          + B's 3 exclusive files (taken directly from stop-tree, including the new rule module and new tests)
          + 6 files changed by both: git merge-file three-way merge (base = merge-tree)
```

| Owner | Files |
|---|---|
| **A only** | `python/diablogym/env.py`, `resource_sustain.py`, `resource_sweep.py`, `tests/test_r18b3b_loot_warm_start.py`, `tests/test_r19_m2_rulings.py` (new), `tests/test_resource_readiness_law.py`, `train/migrate_loot_candidate.py` |
| **B only** | `python/diablogym/resource_emergency_stop.py` (new), `tests/test_r19_gold_grab.py`, `tests/test_r19_m2_emergency_stop.py` (new) |
| **Both** | `options_env.py`, `resource_protocol.py`, `worker_env.py`, `train/eval_contract.py`, `train/runs/r10-staging/probe_r17_deployment.py`, `train/train_ppo.py` |

### 1.1 Only one conflict

`git merge-file` reported only **1 conflict** across the 6 files, at the **same tail** of the chain of
"new services per game" constructors in `OptionsEnv._reset_services`: A builds coach-v05's stall bookkeeping
there, and B builds the emergency stop service. Neither block reads the other's fields, so it was **resolved as
purely additive: keep both, A first, then B** (in the order they are consulted at run time: the coach is asked at
the option boundary in `resource_option_choice`, the stop rule at the tail of each worker tick inside the
window). Resolution script: `m2/resolve.py` (the reason is written into a code comment).

The other 5 files merged cleanly, but **a clean automatic merge does not prove that both sides landed**, so:

### 1.2 Union check (not "looks right")

`m2/verify_merge.py`: for each of the 6 files changed by both, it computes, against the M1 merge tree as base,
the multiset of added and removed lines for A, for B and for the **merge result**, and requires

```
added(merged)   == added(A) | added(B)
removed(merged) == removed(A) | removed(B)
```

**All 12 items (6 files x added/removed) are equal, MISMATCHES: 0**:

| File | Added lines | Removed lines |
|---|---|---|
| `options_env.py` | 218 | 4 |
| `resource_protocol.py` | 299 | 9 |
| `worker_env.py` | 18 | 3 |
| `train/eval_contract.py` | 41 | 5 |
| `probe_r17_deployment.py` | 26 | 1 |
| `train/train_ppo.py` | 25 | 10 |

Both rounds touched the tail of `_R16_ENV_KEYS` (B appends `resource_emergency_stop`); after the merge it is the
union, and B had already changed the old assertion that pinned the closing bracket into a direct membership
check (`test_r19_gold_grab.py`).

### 1.3 The certified rules are untouched, rule by rule (not a grep: the rule rows are read out and compared)

`m2/lawdump.py` imports the deployment-side modules in **a separate process per tree** and dumps to JSON the
two rows of `SWEEP_LAWS`, every field of `RetreatPolicy()`, all upper-case constants of both modules, the coach
vocabulary, and all upper-case constants of `resource_sustain` expanded as dataclasses. The `diff` of the M1
merge tree against the M2 merge tree has **only `>` lines (additions) and only two `<` lines**: the formerly
empty `coach_v03_constants: {}`, and the last element line of the vocabulary tuple (a comma now follows
`"coach-v03"`). **No existing value changed**:

- the `sweep-v1` row only gains a `max_targets_per_floor: null` field, **inert by construction**;
- the `sweep-v2` row has `max_targets_per_floor: 24`;
- `RetreatPolicy` (`hp_fraction 0.5` / `drink_hp_fraction 0.4` / `empty_belt_hp_fraction 0.75` /
  `pressure_radius 6` / `pressure_count 0`): **not a single field is in the diff**;
- purchase mode `"full"`: the new branches are all in the `v2_potions` / `v2_armor` stages, and `"full"` returns
  at `minimum_potions`; the bit-identical OFF arm of the probe is the empirical proof.

---

## 2. Tests: the **union** of both implementers' neighbourhoods

Command: `pytest -p no:randomly -q <21 files>`, `DIABLOGYM_ROOT=<tree>`; the control is **a clean copy of the
merge base made with the same rsync recipe**, `m2a-pristine/` (checked with `diff -rq` =
`PRISTINE_IS_MERGE_TREE`).

| Tree | Result |
|---|---|
| clean control (19 shared files) | 12 failed / **918 passed** / 20 skipped / 726 subtests |
| **m2-merge** (19 shared + 2 new test files) | 12 failed / **1060 passed** / 20 skipped / 740 subtests |

> **Review-round correction (2026-09-09, see section 16.3)**: this section originally compared only **half** of
> the failures. Both sides reported `12 failed`, but `t-{pristine,merged}-fails.txt` each held only 6 ids; the
> other 6 were printed by pytest-subtests as `SUBFAILED(key=...) <id>` and were missed by the anchor
> `grep -E "^(FAILED|ERROR) "`, and they belong to **two other test ids that never appeared in this section's
> list**. Rerun with `^(FAILED|ERROR|SUBFAILED)` and normalised subtest keys: **both sides have 12 failure lines
> and 8 distinct test ids, identical line by line**, and there are still 0 new failures. The correct failure set
> is in section 16.3.

**New failures 0, fixes 0** (`comm` empty in both directions). The extra **142 passed** are exactly the two new
test files (A's 52 + B's 90). All failure ids shared by both sides come from the **tree shape**: by recipe the
work trees do not carry `train/runs/...model_candidate.zip` or `train/models/v22-h-manager/policy.npz`:

```
tests/test_r13_live_dive.py::test_eval_registration_param_fail_closed
tests/test_r18b6_training_wiring.py::MakeEnvTests::test_each_law_still_needs_a_worker_or_options_env
tests/test_r18b6_training_wiring.py::MakeEnvTests::test_smith_v1_can_never_run_outside_the_full_purchase_mode
tests/test_r18b6_training_wiring.py::MakeEnvTests::test_the_factory_forwards_each_law_and_omits_the_defaults
tests/test_r18b6_training_wiring.py::MakeEnvTests::test_the_factory_forwards_the_whole_b6_world
tests/test_r18b6_training_wiring.py::TrainingCliTests::test_validate_args_accepts_the_whole_b6_world
```

Logs: `m2/t-{pristine,merged}.log`; failure sets `t-{pristine,merged}-fails.txt`. **The full `tests/` suite was
still not run** (an inherited gap).

---

## 3. Probe design

- worker: `7e31dc54` = the `model_candidate.zip` of the `r16-arm-a-constitution` run in the main tree
- seeds: **only** pool 2_133, 2133000-2133047, three shards a/b/c (16 + 16 + 16), `max_steps 6000`,
  `decoding sample`
- driver: `m2/m2_probe_driver.py` (`arm_stats` / `paired` are **verbatim** from the R18 `m2_probe_driver.py`,
  passed down through M1's `m1_probe_driver_rr.py`; `m2_stats` is the new, **add-only** half in this round)
- the income / spending / first-descent rows come from `m2/agg2.py`, which extracts M1's
  `m1rr/agg_rr.py::stats` with `ast` and **executes it unchanged** (not retyped), and **asserts** that it
  reproduces on the M1 ALL-ON rows the 12 numbers printed in section 11.3 of the M1 report (not published);
  otherwise the script stops. So this report and the M1 report use the same ruler.
- sha256 fingerprints of `python/ train/ tests/` before and after the run: **the tree did not change during the
  run** (`m2/tree-fingerprint-{before,after}.txt`).

### 3.1 The seven arms

| Arm | Override keys | `rows_sha_v3` | runtime err |
|---|---|---|---|
| certified control rows (inherited) | 16 keys | `0e5a1acd…be886` | - |
| **M2 OFF** | 16 keys, unchanged | **`0e5a1acd…be886`** (match) | 0 |
| M1 ALL-ON (inherited) | +full-v2 +sweep-v2 +gold-grab-v1 | `579816824729d33b…` | - |
| **M2-SET** | same as above (the **code** is after rulings 1/2/3) | `f33f7af5f236f1fc273293160910b5606f7d03f8a08e91c499ed1a45ded75da4` | 0 |
| **M2-SET+coach** (attribution) | M2-SET + `resource_readiness_law=coach-v05` | `27b722fb8d8a769a225ac375ad76e395e96b80719e5ee0621675ec41d3fb1ada` | 0 |
| **M2-SET+stop** (attribution) | M2-SET + `resource_emergency_stop=stop-v1` | `9842d5f809f086f3dfc9516dd75b63919c35f2b89bba2b85c6a4ba976b4cb4e8` | 0 |
| **M2-FULL** | M2-SET + coach-v05 + stop-v1 | `68f41f5ce5bf230319829edae2004393005835ef278d1d6e92935b084922b4ab` | 0 |

> **I added the attribution arms; the task brief did not name them.** Reason: M2-FULL turns on two rules at once,
> and if it is worse than M2-SET the report could only say "we don't know which half". Two 48-seed arms (6 shards,
> about 4 minutes of wall time) turned that sentence into a number. The seed pool was not widened; it is still
> only 2133000-2133047.

### 3.2 Regression

**M2 OFF's `rows_sha_v3` is bit-identical to the certified control rows**; paired saved 0 / lost 0 /
discordant 0. In other words, the four new paths (ruling 1's per-floor cap, `None` on the `sweep-v1` row; ruling
3's ranking, which lives only in the `v2_*` stages; coach-v05, whose object is `None` under coach-v03; stop-v1,
whose service is `None` when the flag is `off`) **are bit-for-bit silent in the control world**.

---

## 4. Main table (same 48 seeds, same worker, same control rows)

| Metric | control / OFF | M1 ALL-ON | **M2-SET** | +coach | +stop | M2-FULL |
|---|---|---|---|---|---|---|
| **Survival /48** | 20 | 24 | **28** | 25 | **28** | 26 |
| Reached L2 | 34 | 33 | **30** | 30 | 30 | 30 |
| Reached L3 | 4 | 5 | **2** | 2 | 2 | 2 |
| L2 deaths | 19 | 14 | **12** | 14 | **12** | 13 |
| L2 beats | 85578 | 89015 | 87358 | 78026 | **94473** | 86267 |
| **L2 hazard /1k** | 0.2220 | 0.1573 | **0.1374** | 0.1794 | **0.1270** | 0.1507 |
| L2 kills | 894 | 835 | 826 | 773 | 813 | 766 |
| clvl >= 4 | 25 | 24 | **18** | 16 | 23 | 21 |
| median clvl | 4.0 | 3.5 | 3.0 | 3.0 | 3.0 | 3.0 |
| deaths_by_floor | {1:7, 2:19, 3:2} | {1:8, 2:14, 3:2} | **{1:8, 2:12}** | {1:8, 2:14, 3:1} | {1:8, 2:12} | {1:8, 2:13, 3:1} |
| total micro-steps | 537999 | 578079 | 522130 | 512179 | 549965 | 542031 |

Paired McNemar (against the **certified control rows**; formula verbatim from the R18-M2 driver):

| Arm | saved | lost | net | discordant | **UCB95 (one-sided)** |
|---|---|---|---|---|---|
| M2 OFF | 0 | 0 | 0 | 0 | 0.00000 |
| M1 ALL-ON | 12 | 8 | +4 | 20 | 0.06865 |
| **M2-SET** | 14 | 6 | **+8** | 20 | **-0.01860** |
| M2-SET+coach | 11 | 6 | +5 | 17 | 0.03495 |
| **M2-SET+stop** | 15 | 7 | **+8** | 22 | **-0.01087** |
| M2-FULL | 12 | 6 | +6 | 18 | 0.01734 |

Paired (against **M1 ALL-ON**, i.e. "what rulings 1/2/3 actually added"):

| Arm | saved | lost | net | discordant | UCB95 |
|---|---|---|---|---|---|
| **M2-SET** | 8 | 4 | **+4** | 12 | 0.03372 |
| M2-SET+coach | 7 | 6 | +1 | 13 | 0.10263 |
| M2-SET+stop | 10 | 6 | +4 | 16 | 0.05231 |
| M2-FULL | 10 | 8 | +2 | 18 | 0.10340 |

Paired (against **M2-SET**, i.e. the net effect of each part of ruling 4):

| Arm | saved | lost | net | discordant | UCB95 |
|---|---|---|---|---|---|
| M2-SET + coach-v05 | **0** | **3** | **-3** | 3 | 0.11997 |
| M2-SET + stop-v1 | 5 | 5 | **0** | 10 | 0.10837 |

**Reading (as is)**: M2-SET and M2-SET+stop are the only two arms with UCB95 < 0, **the first time we can say at
one-sided 95% "not worse than the control rows"**; but UCB95 is only -0.019 / -0.011, still far from "proven
better", and with 48 seeds there are only 20 / 22 discordant pairs. **This is still a promising reading, not a
certification.**

---

## 5. The cost in depth (the point a decision must see first)

The three arms (M2-SET and its two derivatives) **all** show:

- reached L2: 34 -> **30** (4 fewer games ever went downstairs)
- reached L3: 4 -> **2**
- clvl >= 4: 25 -> **18** (+stop back to 23)
- L2 kills: 894 -> 826

while the first descent is **earlier** (median beat 7280 -> 6798) and **stronger** (median AC 10 -> 17, belt
4 -> 6). So it is not that "the bar for descending went up", but that **fewer games descend, and those that do go
less deep**. The most likely mechanism is in section 8: the per-floor cap invites sweep onto L2, and the extra
6.6k micro-steps of chest sweeping and 345 targets on L2 are taken from "keep going down".

**I do not make this trade-off.** The campaign's goal is depth; the survival bought with depth in this round is
+8 / 48. Whether to accept it is a design decision.

---

## 6. Ruling 1 acceptance: per-floor target cap (24 + 24)

**Done, and it is the strongest effect of this round.**

| | control/OFF (sweep-v1) | M1 ALL-ON | **M2-SET** |
|---|---|---|---|
| L1 windows | 104 | 1012 | 560 |
| **L2 windows** | 0 | **15** | **192** |
| L1 targets | 541 | 1942 | 1067 |
| **L2 targets** | 0 | **33** | **345** |
| L2 chests / barrels / sarcophagi | 0 / 0 / 0 | 9 / 8 / 6 | **87 / 79 / 81** |
| L2 micro-steps | 0 | 462 | **6618** |
| L2 share of all sweep windows | 0% | **1.4%** | **25.5%** |

The veto histogram shifts from `target_cap` to `floor_target_cap`:

- M1 ALL-ON: `{target_cap: 38143, cooldown: 12255, monster_near_pair: 4825, …}`
- **M2-SET**: `{floor_target_cap: 33737, cooldown: 8685, monster_near_pair: 5658, target_cap: 2919, monster_near_pair_unseen: 1127, low_hp: 404, monster_near_target: 210, danger_law: 147}`

The rule's own ledger: `max_targets_per_floor = 24`, `targets_by_floor {1: 1067, 2: 345}`,
`targets_remaining_by_floor {1: 85, 2: 807}`, `floor_target_cap_denials 33737`, **41 / 48 games hit the
per-floor cap**.

Reading: **the L1 quota is essentially used up** (cap 48 x 24 = 1152, 1067 used, 85 left), while **the L2 quota
is far from used** (345 used, 807 left). So after this change **what limits L2 is no longer the quota but
something else** (`cooldown` 8685, `monster_near_pair` 5658, and the 18 games that never reached L2). If a next
change wants deeper L2 sweeping, **it should not touch this cap again**.

**The micro-step budget was not split per floor, following A's judgement** (`microstep_budget_split_by_floor:
False`, readable per arm in telemetry). This round's data support that: the veto histogram has **no
`episode_budget` bucket** (the OFF arm's sweep-v1 has 104, from v1's narrow budget), and M2-SET's sweep
micro-steps over all 48 games total 25363, not even a fraction of the 2400-per-game budget.

---

## 7. Ruling 2: `require_trip_slot=False` approved; no code change in this round

Per the ruling, zero code change; the outputs are a comment on the rule row in `resource_sweep.py` (recording
"R19-M2 (2026-09-09) ruling 2: ratified") and 4 regression tests (the v2 row is still `False`, the v1 row still
`True`; v2 still opens a window at `loot_trip_slots_left=0`; under the same condition v1 still records
`no_trip_slot`). All 4 pass in this round's union.

**Newly measured cost in this round** (M1 measured only the 3 items on the L2 side):

| Orphaned drops (not gold; still on the floor in the last frame / caused by this sweep) | M1 ALL-ON | **M2-SET** |
|---|---|---|
| L1 | 115 / 203 | 60 / 113 |
| **L2** | 3 / 3 | **35 / 41** |

On L2, **35 of 41** items are never picked up, because L2 sweep windows went from 15 to 192 and
`require_trip_slot=False` means a window opens without asking "is there still a backpack slot on this trip".
**This cost was already approved; this round updates its size from 3 to 35.** It can be reverted in one line.
(The gold half is recorded separately and not added: L2 `gold {dropped 68, still_on_floor 21}`.)

---

## 8. Ruling 3 acceptance: rank by "native projected gain / gold"

**The ranking really runs and the defect is partly fixed, but not up to the number named in the ruling.**

| | control/OFF | M1 ALL-ON | **M2-SET** | +stop |
|---|---|---|---|---|
| weapon upgrades | **29** | 8 | **12** | 16 |
| weapon spend (upgrade rule ledger) | 6620 | 1780 | 2520 | 3390 |
| potion spend (full-v2 ledger) | 0 | 8900 | **3600** | 3450 |
| armour spend (full-v2 ledger) | 0 | 7770 | **9055** | 8480 |
| total spend | 15442 | 25486 | 23048 | 23012 |
| median AC at first descent | 10.0 | 15 | **17.0** | 17.0 |
| median belt at first descent | 4.0 | 8 | 6.0 | 6.0 |

The ranking's own ledger (M2-SET, 48 games, 97 trips, 85 of which ranked at least once):

- **235 decisions** (`v2_potions` 129 / `v2_armor` 106)
- **winners: armour 157 / potions 77 / weapon 1**
- potion leg stop reasons: `outranked 52` / `belt_full 31` / `unaffordable 9`
- armour leg stop reasons: `no_upgrade 53` / `over_budget 38` / **`weapon_outranks 1`**
- money left in the wallet when the weapon won: **1 trip, 215 gold** (median 215)
- mean native ratio per leg: **armour 0.07077 / potions 0.02000 / weapon 0.01070**

**One point must be stated plainly: this ranking is dominated by units.** Armour's native unit is AC points,
the weapon's is maximum damage points, and potions' is belt slots; with these three rulers **armour's gain per
gold is 6.6 times the weapon's**, so the weapon leg won only 1 of 90 candidates. The defect named in the ruling
was "weapon upgrades 29 -> 4 under a normal wallet"; this round raises it from 8 to **12** (+50%), **not back to
29**, and the increase comes mainly from "stop when armour is unaffordable" and "stop when potions are
outranked", not from the weapon winning the ranking.

**A next step for the decision (no new arm needed)**: the rule also records `ratio_normalised` (each leg divided
by **the readiness rule's own** requirement for that leg), **recorded but not ranked on**.

> **Review-round correction (2026-09-09, see section 16.5)**: the first version of this table was **the product of
> a bug**. Armour was divided by `_READINESS_V2_AC[depth]` (9 at the first descent) and the weapon by
> `_READINESS_V2_DMG[depth]` (6), **but the potion leg was divided by nothing**: for the 213 potion candidates
> `ratio_normalised` and `ratio` were **bit-identical**. So "potions win 183 after normalisation" was just the
> arithmetic of "two legs shrunk 6-9 times, the third not". The potion leg is now divided by its own readiness
> requirement `required_belt_heals` (constantly 4 in this round), and re-ranked on **the same 235 decisions** the
> conclusion **reverses**:

| Ranking key | armour wins | potions win | weapon wins |
|---|---|---|---|
| **native ratio (current rule)** | **157** | 77 | **1** |
| normalised ratio (after the fix; recorded only, not used) | **131** | 102 | 2 |
| ~~normalised ratio (before the fix, potion leg not normalised)~~ | ~~50~~ | ~~183~~ | ~~2~~ |

Mean normalised ratio per leg (measured in the review round, M2-SET arm): **armour 0.010110 / potions 0.005000 /
weapon 0.002141**, with denominators **7.0 / 4.0 / 5.0** respectively (= `ratio` / `ratio_normalised`).

**Corrected conclusion**: changing the ruler **does not** move money to potions; armour still wins (131/235) and
the weapon still almost never wins (1 -> 2). **Both rulers point to the same thing: ratio ranking does not buy the
weapon back.** Buying the weapon back needs a **rule independent of the ratio** (for example "give the weapon leg
at least one chance per trip" or "a separate floor on the damage axis"); that is a new ruling, and I did not add
it myself.

**Integration judgement (proposed by A, re-checked and kept by me)**: weapon **purchases** are still executed only
by the certified smith-v1 leg; spend-v2 only ranks and does not move purchases. When the weapon wins, the armour
leg stops and leaves the money in the wallet, and a few ticks later smith-v1 spends it on **the same trip**.
Empirical check in this round: this path was actually taken (`weapon_outranks` once, leaving 215 gold).
**Sample size 1.**

---

## 9. The two parts of ruling 4: attribution

**This is the step I added in this round, and the most important table of the round.**

| | M2-SET | +coach-v05 | +stop-v1 | both on |
|---|---|---|---|---|
| survival | **28** | 25 | **28** | 26 |
| L2 hazard /1k | 0.1374 | 0.1794 | **0.1270** | 0.1507 |
| L2 beats | 87358 | 78026 | **94473** | 86267 |
| clvl >= 4 | 18 | 16 | **23** | 21 |
| paired vs M2-SET | - | **saved 0 / lost 3** | saved 5 / lost 5 | - |
| paired vs control rows, UCB95 | -0.0186 | 0.03495 | **-0.01087** | 0.01734 |

**coach-v05: only loses.** It kills three seeds and saves none; L2 hazard rises from 0.1374 to 0.1794 and L2
beats drop 10.7%.

**stop-v1: changes who, not how many.** Net survival 0 (5 saved, 5 lost), but **L2 hazard is the lowest of all at
0.1270**, L2 beats **+8.1%** (longer on L2 with fewer deaths), and clvl >= 4 goes from 18 back to 23. It buys back
"time alive on L2", just not "number of games that survive to the end".

---

## 10. Ruling 4, stop-v1's own ledger

M2-SET+stop (48 games; M2-FULL numbers in brackets):

- worker ticks **76913**, of which in scope (main line L1/L2, worker-owned windows) **76706** (99.7%)
- **50 activations** (52), distributed `{L1: 42, L2: 8}` (`{42, 10}`); **25 / 48 games activated at least
  once** (26)
- trigger conditions: `crowd 34` / `adjacent 16` (`36 / 16`)
- vetoes: `cooldown 234` (242); **`activation_cap` never**; nobody hit the 6-per-game cap
- aborts: `door_on_route 3`; the other four buckets (`no_tile` / `no_route` / `off_grid_step` /
  `protected_tile`) are **all 0**, so **the frozen planner picks a tile and reaches it in the real engine**
- drinking: `drinks_attempted 14 / executed 14` (15 / 15), **0 failures**
- disengagement: **508 steps** (532), `distance_gain_total` **50** (50), so **each activation widens the minimum
  distance to the nearest monster by only 1.0 tile on average** (0.96)
- 30-tick follow-up: **alive 33 / dead 17 / unsettled 0** (34 / 18 / 0)
- `unseen_crowd_beats 198` (208): the frozen "ignore visibility" criterion, recorded but not gated

**The two most glaring numbers**: **508 steps for 50 tiles**, and **34% dead at follow-up**. The vehicle does move,
drink and not get stuck (aborts are almost all 0), but it **hardly disengages at all**: the 12-tick cap, the
radius-6 square and the "must strictly increase the minimum distance" rule together buy on average one tile on a
real map. That explains why it saves 5 and loses 5.

> **Review-round addition (2026-09-09, see section 16.4): this section missed the column that mattered most.**
> It said nothing about **how** each of the 50 activations ended. Counting `windows[].outcome`:
> **`steps_spent 33` / `arrived 9` / `closed:death 5` / `door_on_route 3`**. So **in 5 activations the character
> walked to death while the rule held the body**, and 33 used up the 12-tick budget without reaching the target.
> Three more numbers this section lacked: in **49 / 50 activations the nearest monster was already adjacent**
> (`min_distance0 == 1`), in **36 / 50 the belt was empty** (the "drink first" leg was a no-op), and in **13 / 50
> hit points were already <= 20% of maximum** (lowest 2/78). The recommendation "fix the vehicle first, then
> discuss thresholds" was put up for decision without knowing about the 5 deaths inside the vehicle.

### 10.1 The three named seeds

| Seed | control / OFF | M1 ALL-ON | M2-SET | +coach | **+stop** | M2-FULL |
|---|---|---|---|---|---|---|
| **2133002** | died L1@559 (clvl 1, AC 7, 21 kills) | alive, depth 1 | **alive**, depth 1 (clvl 3, 132 kills, 12000 micro-steps) | alive (same as M2-SET) | **died L1@2208** (clvl 2, 62 kills; stop rule activated 4 times, drank 1, walked 43 steps, follow-up 3 alive 1 dead) | **died**, field-for-field the same as +stop |
| **2133010** | died L2@12001 (clvl 4, AC 7) | died L2@12321 (AC 20) | **alive**, depth 2 (clvl 4, AC 20, 171 kills, 17264 micro-steps) | alive | alive (0 stop activations) | alive |
| **2133047** | **alive**, depth **3** (AC 8, 144 kills) | alive, depth 2 | alive, depth 2 (150 kills, AC 12) | alive | **died**, depth 2 (127 kills; stop rule activated 2 times, **both follow-ups dead**) | **died**, field-for-field the same as +stop |

**The attribution is clean**: these two counterexamples are **exactly the same** as M2-SET in the `+coach` arm and
**exactly the same** as M2-FULL in the `+stop` arm. **The stop-v1 half kills them.**

This matches what implementer B wrote in advance in section 3 of B's notes (not published): replaying these
thresholds on the certified trajectory of 2133002, condition A could open at the earliest at tick 78, while the
ruling read tick 58. B did not change the ruled numbers on its own but put the measurement on the table, and
**now the probe gives it a result: with the ruled numbers this rule not only fails to save 2133002, in the merged
world it turns it from "alive" into "dead".**

---

## 11. Ruling 4, coach-v05's own ledger, and a correction to the diagnostic premise

### 11.1 The rule hardly speaks

M2-SET+coach (48 games; M2-FULL in brackets):

- suppressed windows **9** (13), distributed `{L2: 5, L3: 4}` (`{L2: 9, L3: 4}`), **never on L1**
- suppression reasons: `const_dive_unready 8` / `stall_lock 1` (`10 / 3`)
- **`forced_through` 72 (70)**, distributed `{L1: 71, L2: 1}` (`{L1: 66, L2: 4}`)
- stall windows seen, `stall_windows_seen` 102 (100), `{L1: 90, L2: 12}`
- stall lock activations 13 (12), `{L1: 11, L2: 2}`; **7 / 48 games still locked at the end**
- ~~unlocks `{farm_levelup 85, town_trip 91, descended 74}`~~

> **Review-round correction (2026-09-09, see section 16.6): the line above is false.** `_unlock` wrote its counter
> **outside** the `if floor in self._locked:` guard, and the town-trip branch incremented it whether or not
> anything was locked. So `{85, 91, 74}` are **call counts**, not unlocks. Real unlocks are in `lock_events`:
> **`locked` 13 times, `unlocked` 6 times, and all 6 are `by: "descended"`**. In other words **the only thing that
> ever released a lock was exactly the forced dive the lock was meant to stop**; `town_trip` / `farm_levelup` /
> `farm_scene` / `farm_cleared` **never released one**, and 7 more games were still locked at the end. Anyone
> reading `{85, 91, 74}` would think the lock clears itself; the opposite is true. The counter is now split into
> `unlocks` (real unlocks) and `unlock_calls` (calls); review-round measurements in section 16.6.

**Mechanism**: the 4.3 guard, "never fight the frozen mask", **almost always applies** on L1: in 71 cases on L1
that should have been refused, FARM was already masked, so the window was let through and recorded as
`forced_through`. So **coach-v05 is essentially inert on L1 by construction**; it really speaks only on L2 and L3,
9 windows in total. **9 windows cost 3 survivors.**

### 11.2 Acceptance: DIVE windows closed by stall did not decrease

| Arm | DIVE windows | closed by stall | const_dive windows | **const_dive ∩ stall** |
|---|---|---|---|---|
| control / OFF | 710 | **99** | 267 | **6** |
| M1 ALL-ON | 800 | 87 | 208 | 1 |
| M2-SET | 763 | 108 | 193 | 1 |
| **M2-SET+coach** | 743 | **102** | 186 | 1 |
| M2-SET+stop | 785 | 112 | 200 | 1 |
| M2-FULL | 763 | 100 | 191 | 1 |

The windows ruling 4b set out to break dropped only from 108 to 102 in the coach arm. **This change cut almost
nothing.**

### 11.3 Correction of the diagnostic premise (must be recorded)

The task brief's picture was: seed 2133047 "circles for 400 ticks, 12 const_dive DIVE windows, readiness 0.91,
all closed by stall". On the **certified control rows** I printed that game's DIVE windows one by one (script
`m2/extra.py`):

```
beat0=5909  tau=63   coach_ready  end=sweep_trigger   ratio_v2=1.1111
beat0=6060  tau=68   coach_ready  end=sweep_trigger   ratio_v2=1.1111
beat0=6191  tau=611  coach_ready  end=cap             ratio_v2=1.1111
beat0=6802  tau=182  coach_ready  end=descend         ratio_v2=1.1111  (descended)
beat0=6984  tau=268  const_dive   end=retreat_trigger ratio_v2=0.9091
beat0=9151  tau=211  coach_ready  end=scene           ratio_v2=1.1111  (descended)
beat0=9362  tau=381  const_dive   end=retreat_trigger ratio_v2=0.9091
beat0=11499 tau=245  coach_ready  end=scene           ratio_v2=1.1111  (descended)
beat0=11744 tau=442  const_dive   end=fuse            ratio_v2=0.9091
beat0=12186 tau=61   const_dive   end=descend         ratio_v2=0.9091  (descended)
beat0=12247 tau=601  const_dive   end=cap             ratio_v2=0.75
```

**11 windows, 5 const_dive, 0 closed by stall.** `ratio_v2` is indeed 0.91 (matching "readiness 0.91"), but the
close reasons are `retreat_trigger` / `fuse` / `cap` / `descend`, **not `stall`**.

Across all 48 control games: 710 DIVE windows, **267** const_dive, **99** closed by stall, and **only 6 in the
intersection**. The median `ratio_v2` of const_dive windows is **0.500**, and **100% are below 1.0**, so the
threshold of rule (a) holds on **every** const_dive window; it suppresses only 8-10 of them entirely because of
that mask guard.

**Conclusion**: ruling 4b (two consecutive stall windows without descending -> lock the floor) **was designed
around a picture that the probe rows themselves do not support**; what the probe really shows is "many
const_dive windows (267/710), median readiness 0.5, all kinds of close reasons". To treat that, rule (a) is the
right direction, but on L1 it is currently disabled entirely by the "never fight the mask" guard.

> **Review-round addition (2026-09-09): "rule (a) is the right direction" needs a number.** I classified every
> DIVE window by `(trigger, forced)`:

| Arm | const_dive windows | of which `forced=True` (FARM already masked) | **`forced=False` (where rule (a) can actually act)** |
|---|---|---|---|
| certified control rows / OFF | 267 | 198 | **69** |
| **M2-SET (the world that would actually ship)** | 193 | 186 | **7** |
| M2-SET+coach | 187 | 186 | **1** |

> **Under the M2-SET code, rule (a) can act on only about 7 windows per 48 games.** That is exactly why it
> suppresses only 8 (4 after the review), and it is the real value of the recommendation "keep only rule (a)" in
> section 13.3: **unless the "never fight the frozen mask" guard is re-decided at the same time, "keep only rule
> (a)" keeps almost nothing.** That would be a **new ruling**, not a deletion.

---

## 12. Income / spending / first descent (the same ruler as M1 section 11.3)

| Metric | control/OFF | M1 ALL-ON | **M2-SET** | +coach | +stop | M2-FULL |
|---|---|---|---|---|---|---|
| games with a first descent | 34 | 33 | 30 | 30 | 30 | 30 |
| median first-descent beat | 7280.0 | 7486 | **6798.0** | 6798.0 | 7235.0 | 7235.0 |
| median first-descent belt | 4.0 | 8 | 6.0 | 6.0 | 6.0 | 6.0 |
| median first-descent **AC** | 10.0 | 15 | **17.0** | 17.0 | 17.0 | 17.0 |
| median first-descent damage | 9.0 | 9 | 8.0 | 8.0 | 7.0 | 7.0 |
| town trips | 105 | 105 | 97 | 97 | 108 | 109 |
| town micro-steps | 150434 | 161004 | 143703 | 142454 | 157810 | 157261 |
| income: trip collect | 12178 | 10841 | 8625 | 8574 | 9673 | 9622 |
| income: selling | 7591 | 8786 | **9066** | 8700 | 7388 | 7064 |
| income: selling (identified) | 3769 | 4280 | **5326** | 4824 | 3526 | 3024 |
| income: **gold pickup** | 0 | 8229 | **8695** | 8382 | 8765 | 8478 |
| spend: total | 15442 | 25486 | 23048 | 22836 | 23012 | 23101 |
| spend: repair | 767 | 1111 | 613 | 616 | 682 | 736 |
| spend: identify | 1600 | 1900 | 2100 | 2000 | 1600 | 1500 |
| spend: weapon (upgrade rule) | 6620 | 1780 | 2520 | 2520 | 3390 | 3390 |
| spend: potions (full-v2 ledger) | 0 | 8900 | 3600 | 3600 | 3450 | 3450 |
| spend: armour (full-v2 ledger) | 0 | 7770 | **9055** | 8940 | 8480 | 8565 |
| items identified | 16 | 19 | **21** | 20 | 16 | 15 |
| retreats | 55 | 63 | **49** | 48 | 61 | 61 |
| median gold_final | 78.0 | 89.0 | **92.0** | 87.0 | 92.0 | 85.0 |

> `sweep gold` is still **reported as empty, not as 0** (the M1 section 11.3 criterion carries over: no channel
> tags a coin with "which chest dropped it"; `gold_dropped_seen` counts **piles**, not amounts).

Time shares: sweeping **4.86%** (M1 6.93%), gold pickup **2.44%** (2.12%), town **27.52%** (27.85%). Per-floor
gold pickup ledger (M2-SET): `{L1: {windows 402, piles 545, gold 5205}, L2: {windows 169, piles 184, gold
3490}}`; L2 gold is 14% above M1 (3064).

---

## 13. Reading and recommendations

1. **Rulings 1/2/3 are worth keeping.** M2-SET is the campaign's first arm with UCB95 < 0; L2 hazard falls from
   0.2220 to 0.1374, and paired against M1 ALL-ON it is also net +4. Ruling 1 has the largest effect (L2 sweep
   windows 15 -> 192).
2. **But decide the depth trade-off first.** In the same arm L3 falls from 4 to 2, L2 arrivals from 34 to 30, and
   clvl >= 4 from 25 to 18. If depth is a hard goal, "trade shallowness for survival" has to be paid back sooner
   or later, and paying now is cheaper.
3. **Recommendation: do not adopt coach-v05 (as currently written).** Over 48 games it suppresses only 9 windows,
   loses 3 survivors and raises L2 hazard by 31%, while the problem it targets (DIVE windows closed by stall)
   **does not shrink by a single window**. To keep it, I would keep only rule (a), drop rule (b), and **re-decide
   the "never fight the mask" guard**, which is exactly what disables rule (a) on L1.
   > **Review-round qualification**: (i) "keep only rule (a)" without re-deciding the guard acts on only **about
   > 7 windows / 48 games** (section 11.3 review addition), which keeps almost nothing; (ii) the review round did
   > **not** remove rule (b), which is part of the ruling and not the reviewer's call, but gave it a release path
   > **that actually fires in the measured world** (section 16.6); (iii) after the review the coach arm is alive
   > 25 -> **26**, hazard 0.1794 -> **0.1737**; the conclusion ("only loses") **does not change**, and paired
   > against M2-SET it is still saved 0 / lost 2.
4. **stop-v1 is a real fork; it needs its own decision.** Net survival 0, but it buys back exposure time on L2
   (beats +8.1%, lowest hazard of all at 0.1270), and clvl >= 4 goes from 18 back to 23. Its mechanism ledger is
   **healthy** (aborts almost all 0, drinking 100% successful, cap never hit); what is wrong is that **the
   disengagement barely works**: 508 steps for 50 tiles. My recommendation: **fix the vehicle first, then discuss
   thresholds**: loosen `disengage_microsteps` 12 and "must strictly increase the minimum distance" together, or
   rerun an arm with the crowd condition B measured in section 3 of B's notes ("adjacent >= 5, regardless of
   hit points"). Both are new rulings.
   > **Review-round correction**: the vehicle **has been fixed** (hit points / trigger condition / separation
   > re-measured every tick, section 16.4), but in the **opposite** direction to this recommendation: tightened,
   > not loosened. Result: **deaths inside the vehicle 5 -> 0**, and the named seed 2133002 moves from "L1 tick
   > 2208, clvl 2" to "L2 tick 14509, clvl 4, AC 25"; **but survival 28 -> 25**, and disengagement becomes nearly
   > zero (508 steps -> 43 steps, 50 tiles -> 12 tiles). **The fixed vehicle is safe, and also nearly inert.** See
   > sections 16.4 and 16.8.
5. **Ruling 3 is only half fixed.** Weapon upgrades 8 -> 12, not back to 29; under the native ruler armour's gain
   per gold is 6.6 times the weapon's, and the weapon wins only 1 of 90 candidates. The normalised ruler gives
   **potions over armour** (183/235), still not the weapon. Buying the weapon back needs a rule independent of the
   ratio.

---

## 14. Known gaps (not hidden)

1. **The depth regression is not explained at the mechanism level.** The mechanism I give (the per-floor cap
   invites micro-steps onto L2) is **an argument**, not a measurement: I did not run the counterfactual arm
   "sweep-v2 + per-floor cap but sweeping off on L2".
2. **The attribution arms are a single 48-game run each, with only 3 (coach) and 10 (stop) discordant pairs.**
   "coach-v05 only loses" rests on **3 discordant seeds**, and `UCB95 0.11997` is far above 0. **Do not read it as
   a certification.**
3. **The same holds for stop-v1's "5 saved, 5 lost"** (discordant 10, UCB95 0.10837). The only strong statement of
   this round is the field-by-field attribution of the two **named seeds**, which are individual cases, not a
   distribution.
4. **coach-v05 has no "consulted but let through" telemetry.** It counts only suppressions and `forced_through`,
   not the number of times `dive_verdict` was entered and returned None, so the rest of "186 const_dive windows,
   only 8 suppressed" in section 11.1 can only be explained by `forced_through` and the mask guard, **not checked
   window by window**. Next round: add a `consulted` counter.
5. **Training is still not possible.** Both new rules have only fingerprint binding and a deployment-side
   validator; there is **no** `make_env` forwarding of `--resource-emergency-stop` /
   `--resource-readiness-law=coach-v05` (the same trade-off as gold-grab-v1 / R19-A1). This round is Python-only
   and probe-only; **a training smoke run was not done (not verified)**.
6. **The full `tests/` suite was not run**; the 3 collection failures + 3 Biteq ERRORs recorded by M1 are
   tree-shape issues, not touched this round. The 6 failures in the union fail identically on the clean control.
7. **The L2 cost of `require_trip_slot=False` rose from 3 to 35** (section 7). It was approved, but the approval
   saw 3.
8. **The `ratio_normalised` re-ranking was computed offline** (the table in section 8 comes from `m2/extra.py`
   re-running the ranking key on the same 235 decisions), **not from an arm that was run**. It answers "who wins
   with the other ruler", not "how many survive with the other ruler".
9. **`spend_v2`'s `weapon_outranks` appears only once**: the path "armour stops and leaves the money for
   smith-v1" **was taken only once** (215 gold). Sample size 1.
10. **`_spend_v2_back_edges` counts per game and is not reset per trip** (A's notes, section 5.7, not published);
    this round's telemetry did not read it out separately.
11. **Seed-pool discipline**: only pool 2_133, 2133000-2133047, was used; the fresh pools 2_116-119 /
    2_126-128 were not touched. But **the same pool has been used repeatedly in this campaign**, and the 5 new
    arms consumed it once more; the overfitting risk is accumulating, a methodological gap. **The review round
    ran 5 more (OFF/FULL/STOP twice), so this gap is heavier than before.**

**Gaps added by the review round (2026-09-09)**

12. **The fixed vehicle is nearly inert.** stop-v1 now walks only 43 steps in 44 activations and buys 12 tiles;
    `arrived` and `steps_spent` are **0 each**, and 19/44 are hit on the very first step (`hp_fell`). What remains
    is basically **those 15 drinks**. The discipline "give the body back" is right, but `abort_on_hp_loss` is the
    **strictest, parameterless version**; "how much hit-point loss means the path is blocked" (e.g. allow one
    tick of damage, or a percentage of max_hp) is a **new ruling** that I did not set myself.
13. **`gain_denominator` exists only in the ranking function, not in the probe rows.** The three denominators in
    the section 8 table (7.0 / 4.0 / 5.0) were back-solved from `ratio / ratio_normalised`, not read from the
    rows. Adding the key means rerunning four arms just for telemetry, which this round did not do.
14. **coach-v05's `farm_windows` release path is too sparse.** It fired once in 48 games, because locks mostly
    close at the tail of a game (`farm_windows_while_locked` is only 4 windows in total). **The 7 / 48 games still
    locked at the end are not solved**, only no longer hidden by false telemetry.
15. **None of the review-round differences between arms is significant.** stop vs M2-SET has only 7 discordant
    pairs, coach only 2, FULL only 10; all three UCB95 > 0. **Every "better/worse" in this section must be read
    as a direction, not a conclusion.**
16. **The new vocabulary in `eval_assembled.py` is pinned only at source level and at the
    `validate_r16_environment` level; no evaluation was actually run with `--resource-readiness-law coach-v05`**
    (evaluation must go through the certified archive path, and this round is probe-only). **Not verified.**

---

## 15. Artifacts (local R19 work directory, not published)

| Purpose | Path |
|---|---|
| merge tree | `m2-merge/` |
| patch against the **main tree** (25 files) | `m2.patch` |
| A's / B's raw diffs | `m2/{a,b}.diff` |
| three-way merge results | `m2/mf/*.py` |
| conflict resolution script | `m2/resolve.py` |
| **union check** (merge == A ∪ B) | `m2/verify_merge.py` |
| **certified rule row comparison** | `m2/lawdump.py` + `law-{m1,m2}.json` |
| test logs / failure sets | `m2/t-{pristine,merged}{.log,-fails.txt}` |
| probe driver (3 main arms) | `m2/m2_probe_driver.py` |
| probe driver (2 attribution arms) | `m2/attrib.py` |
| arm summaries | `m2-probe/m2-summary-{all,attrib}.json` |
| per-arm rows | `m2-probe/m2-{off,set,full,coach,stop}-rows.json` |
| income / spending / first descent (the M1 ruler) | `m2/agg2.py` + `agg2.json` |
| 2133047 per-window table + normalised re-ranking | `m2/extra.py` |
| **all review-round outputs** | see section 16.9 |
| rendered tables | `m2/render{,2}.txt` |
| tree fingerprints (before/after) | `m2/tree-fingerprint-{before,after}.txt` |
| smoke run (2 seeds, FULL arm) | `m2/smoke-full.{json,log}` |
| ledger events | `R19_M2_INTEGRATED`, `R19_M2_PROBE_RESULT` |

---

## 16. Review round (2026-09-09)

A reviewer raised **4 high + 5 medium** findings on m2-merge. All were fixed, and the code, tests, probes, patch
and this report were re-run or rewritten. This section is the full account of that round, **including three
places where I did not follow the reviewer's suggestion, and one change I made and then reverted**.

## 16.1 One-page summary

| | Result |
|---|---|
| **OFF regression (every new rule OFF)** | `rows_sha_v3` = `0e5a1acd2fb2c07cd27eb72f9ecc574322e72f362ef0376b692376d9a99be886` = certified control rows, **equal on two independent reruns** (match) |
| **M2-SET** | `f33f7af5…`, **bit-identical to before the review** (paired saved 0 / lost 0); the H1 and H4 changes **change no behaviour** |
| **M2-FULL'** | `72a86601…`, alive **22** (26 before the review), hazard **0.1948** (0.1507) |
| **M2-SET+coach'** | `615f1281…`, alive **26** (25), hazard **0.1737** (0.1794) |
| **M2-SET+stop'** | `c79526a8…`, alive **25** (28), hazard **0.1528** (0.1270) |
| test union | clean control and m2-merge **each 12 failure lines / 8 test ids, identical line by line**; 0 new failures |
| main tree / M1 merge tree | main tree unchanged except the ledger; M1 merge tree zero writes; `src/` byte-identical to the main tree; `build` symlink untouched |

**What must be said first**: this round made stop-v1's vehicle **safe** (deaths inside the vehicle **5 -> 0**) and
also **inert** (disengagement 508 steps -> 43 steps, 50 tiles -> 12 tiles), and **alive 28 -> 25**. The coach half
barely moves (25 -> 26). **M2-SET (rulings 1/2/3) is still the campaign's only arm with UCB95 < 0, and it is
bit-for-bit unchanged in this round.** After their defects were fixed, **neither part of ruling 4 beats M2-SET**.

## 16.2 Nine findings, one by one

| # | Level | Location | Action |
|---|---|---|---|
| H1 | high | `train/eval_assembled.py` | **changed**: the vocabulary is **imported** from `diablogym.resource_protocol` (`choices=RESOURCE_READINESS_LAWS`), and the two `!= "coach-v03"` gates became `not _is_coach(...)`; new `EvalAssembledRegistrationTests` (5 cases, including a neighbourhood check that scans `train/*.py` and `python/diablogym/*.py`) |
| H2 | high | `options_env._emergency_stop_tail` | **changed**: three things (hit points / trigger condition / separation) are re-measured **before every tick** of the scripted walk, see 16.4. **The suggested "separation floor" was not added**; reasons in 16.7 |
| H3 | high | `resource_protocol.ReadinessCoachV05` | **changed**: added a release path that actually fires in the measured world (close 2 FARM windows on a locked floor) plus a census of window close reasons, see 16.6. **The suggestion "remove rule (b)" was not followed**; reasons in 16.7 |
| H4 | high | `resource_sustain.spend_v2_rank_legs` | **changed**: the potion leg is normalised by its own readiness requirement `required_belt_heals`; the re-ranking table in section 8 was recomputed, and the conclusion reversed |
| M5 | medium | `m2/tests.sh` | **changed**: `grep` includes `SUBFAILED` and normalises subtest keys; rerun, see 16.3 |
| M6 | medium | `_unlock` telemetry | **changed**: `unlocks` counts only real unlocks, `unlock_calls` counts calls; section 11.1 corrected |
| M7 | medium | coach-v05 has no floor scope | **changed**: `COACH_V05_FLOORS = (1, 2)`, the same set as stop-v1's `floors`; on L3 it only records and does not refuse |
| M8 | medium | report sections 11.1/11.3/13.3 | **changed**: the 186/186 forced overlap and the "about 7 windows it can act on" are in section 11.3; section 13.3 is qualified |
| M9 | medium | attribution (wage / kills / fuse) | **changed**: the stop rule's W increment, kills and fuses are booked separately and subtracted from the worker's books, see 16.5 |

## 16.3 M5: the test union's failure set (only half was compared before)

`pytest-subtests` prints its failures as `SUBFAILED(key='...') <id>`, which the original
`grep -E "^(FAILED|ERROR) "` did not catch, so **only 6 of each side's 12 failures were compared**; the missing 6
belong to **two other test ids**:

```
tests/test_r18b6_training_wiring.py::MakeEnvTests::test_the_clock_and_retreat_rings_shadow_that_gate_in_the_real_fixture
tests/test_r18b6_training_wiring.py::TrainingCliTests::test_validate_args_pins_each_law_to_the_loot_itinerary
```

Rerun with the corrected anchor (script `m2rr/rr_tests.sh`):

| Tree | Result | failure lines | distinct test ids |
|---|---|---|---|
| clean control `m2a-pristine` (19 shared files) | 12 failed / **918 passed** / 20 skipped / 726 subtests | 12 | **8** |
| **m2-merge** (19 shared + 2 new test files) | 12 failed / **1103 passed** / 20 skipped / 742 subtests | 12 | **8** |

**`NEW FAILURES IN MERGED` and `NEW FAILING IDS IN MERGED` are both empty**, and the 12 failure lines on both
sides are **identical line by line**. The extra **185 passed** are the two new test files (the review round added
43 new cases, including `ReviewRound*`, `EvalAssembledRegistrationTests`, `CoachV05ReviewRoundTests` and
`SpendV2NormalisedRatioTests`). Logs `m2rr/rr-{pristine,merged}.log`, sets `rr-{pristine,merged}-{fails,ids}.txt`.

## 16.4 H2: the vehicle no longer holds on for thirteen ticks

**Defect (the reviewer's measurement, which I recomputed item by item)**: v1's loop re-measured **nothing** apart
from "arrived or not" and the four planner refusals. Of 50 activations, **49** fired when the nearest monster was
**already adjacent**, **36** had an empty belt ("drink first" was a no-op), **13** had hit points <= 20% of
maximum, **33** used up all 12 ticks without arriving, and **5** ended with `closed:death`: **the character died
while the rule held the body**.

**Fix**: re-measure three things **before every scripted step**, and give the body back as soon as any of them
says no. All three thresholds are written **on the rule row** (so the second rule row is still a dict record,
not a code change):

| New rule-row field | Value | One-line reason |
|---|---|---|
| `abort_on_hp_loss` | `True` | give the body back when hit points fall below those **at the start of the walk**. No parameter. 5 of 50 died inside the vehicle; a "disengagement" bought with blood is not one |
| `abort_on_trigger_cleared` | `True` | give the body back when the trigger condition no longer holds. The vehicle answers only that one fact |
| `no_progress_steps` | **4** | give the body back after 4 consecutive steps that do not widen the minimum distance to the nearest visible monster. Measured: 508 steps bought only 50 tiles; 4 is **a third** of the ruled 12-tick budget, enough to go round one corner and enough to show that the path is blocked |

**Measured (M2-SET+stop, 48 games; before the review in brackets)**:

| | before review | **after review** |
|---|---|---|
| activations | 50 | **44** |
| disengagement steps | **508** | **43** |
| `distance_gain_total` | 50 tiles | **12 tiles** |
| tiles bought per activation | 1.0 | **0.273** |
| **how they ended** | `steps_spent 33 / arrived 9 / closed:death 5 / door_on_route 3` | **`trigger_cleared 25 / hp_fell 19`** |
| **deaths inside the vehicle** | **5** | **0** |
| drinks (attempted / done) | 14 / 14 | 15 / 15 |
| empty belt at activation | 36 | 29 |
| hit points <= 20% of max at activation | 13 | **3** |
| `min_distance0` histogram | `{1: 49, 2: 1}` | `{1: 44}` |
| 30-tick follow-up | alive 33 / dead 17 | alive 30 / dead 13 / unsettled 1 |
| planner aborts (`door_on_route` etc.) | 3 | **0** |

**Mechanism in one sentence**: `hp_fell` is 19/44, **hit on the very first step of the walk**. This is what the
reviewer's diagnosis ("walking away for twelve ticks while adjacent") looks like in the data. Now it gives the
body back after one step, so the rule effectively degrades to "**drink once when hurt and surrounded, then walk
about one step**": `arrived` never, `steps_spent` never. **Safe, but nearly inert.**

**Named seed 2133002 (before -> after the review, same seed, same arm)**:

| | `+stop` before review | **`+stop` after review** |
|---|---|---|
| outcome | died **L1 @ tick 2208** | died **L2 @ tick 14509** |
| clvl / AC / max_hp | 2 / 7 / 78 | **4 / 25 / 94** |
| kills | 62 | **161** |

**With the vehicle fixed this seed still dies, but one level deeper, after 6.6 times as long and 2.6 times as many
kills.** 2133047 is essentially the same as before the review (died L2 @ 13792 vs 13883, clvl 4, AC 13, belt 0).

## 16.5 M9: scripted ticks no longer count as the worker's wage

`_emergency_stop_tail()` used to sit **inside** the `worker_wage` / `worker_kills` difference: up to 13 scripted
ticks (one a12 + twelve direction keys, none chosen by the policy) were booked as the PPO wage and kills of **the
network's single proposal**. Worse,
`fuse = primary if primary.fuse_tripped else (ending if ending.fuse_tripped else None)` would report **a fuse
tripped by a scripted direction key** as **the worker's fuse**, while `worker_env`'s current meaning of
`fuse_tripped` is "**reject the proposal**", and `leashed_ppo` uses it to filter BC.

**Fix**: take `W` and `_ep_kills` before and after the call, book the difference into the window's new
`stop_wage` / `stop_kills`, and **subtract** it from `worker_wage` / `worker_kills`; when `ending is stop_ending`
and a fuse tripped, record `stop_fuses` and do **not** count it as the worker's. The probes in this round do not
train, so this has **no numeric effect**; it is a **reward-hacking surface** that must be closed before
`--resource-emergency-stop` is wired into `make_env` (the gap in section 14.5). New
`ReviewRoundCreditAssignmentTests` (6 cases).

## 16.6 H3 / M6 / M7: coach-v05's lock, telemetry and scope

**M7 scope.** coach-v05 originally had **no floor scope**, and **4 of its 9 suppressions fell on L3**, outside the
evidence of ruling 4b (the L1/L2 diagnosis) and a direct tax on the campaign's single north star (depth). Now
`COACH_V05_FLOORS = (1, 2)`, **the same set** as stop-v1's `floors`; outside it only `out_of_scope` is recorded,
nothing is refused. Measured: `out_of_scope = 2` (all on L3), and suppressions go from `{L2: 5, L3: 4}` to
**`{L2: 5}`** (`const_dive_unready 4` + `stall_lock 1`).

**M6 telemetry.** `unlocks` now increments only when a floor is actually released; `unlock_calls` keeps the call
count. Measured (coach arm, 48 games):

```
lock_events   locked 14 / unlocked 7   (unlocked: descended 6, farm_windows 1)
unlocks       {descended: 6, farm_windows: 1}          <- 7 real unlocks
unlock_calls  {farm_levelup 85, town_trip 91, descended 73, farm_windows 1}  <- 250 calls
games still locked at the end   7 / 48
```

**H3 release path.** The reviewer asked why `COACH_V05_FARM_UNLOCK_REASONS` never fired. This round gives a
**measurement** instead of a guess, from the new `window_end_reasons` census (coach arm, 48 games):

```
FARM      sweep_trigger 524, grab_trigger 420, cap 122, fuse 93, levelup 85,
          exhausted 58, retreat_trigger 18, death 11, end 4
DIVE      fuse 213, cap 107, stall 105, sweep_trigger 103, grab_trigger 87,
          scene 41, retreat_trigger 39, descend 32, end 16, death 1
RESUPPLY  sweep_complete 743, grab_complete 565, scene 231,
          resource_complete 91, retreat_complete 48, death 10, end 6
```

Three readings:
1. **In this world FARM windows end with `scene` and `descend` 0 times each**; `levelup` happens 85 times but
   never on "a floor that is currently locked", so the set `("scene","descend","levelup")` really is empty in the
   measured world;
2. the question the reviewer asked me to check has an answer: **retreat-driven RESUPPLY windows publish
   `retreat_complete` (48 times), not `resource_complete` (91 times)**. I did **not** add `retreat_complete` to
   the town-trip unlock set: it is the retreat vehicle **giving the body back**, not "a town trip completed", and
   treating it as a town trip would let the lock be released by something that did not happen. **Recorded, not
   changed.**
3. new release path **`farm_windows`**: the lock is released when **2 FARM windows close on the locked floor**.
   `2` is not a new number but the ruled `COACH_V05_STALL_WINDOWS` read in reverse (two windows of evidence to
   lock, two windows of evidence to unlock). Measured: **fired once**; `farm_windows_while_locked` is only **4**
   windows over 48 games (`exhausted 2 / end 1 / grab_trigger 1`); **locks mostly close at the tail of a game**,
   after which few FARM windows remain to close. So this path **is alive but sparse**, and the 7 / 48 games still
   locked at the end **are not solved**.

## 16.7 Three places where I did not follow the suggestion (must be recorded)

1. **No "separation >= 2" floor for condition A.** The reviewer's reason: "49/50 fire when adjacent; this rule
   never acts pre-emptively". A floor would **silence the rule on the ruling's own example**: at tick 78 of
   2133002, hp is 33/70 with nine adjacent monsters, separation 1, while condition B needs hp <= 0.3 x 70 = 21, so
   **neither condition holds**. Instead I widened the crowd ring from 2 to 3 to try to get pre-emptive samples and
   ran 48 games: the `min_distance0` histogram moved only from `{1: 49, 2: 1}` to `{1: 44, 2: 1}`, **so no
   pre-emption was bought**. The ring therefore **went back to the ruled 2**, and the radius-3 arm is kept as a
   counterfactual record (`m2rr-probe/radius3/`, `rows_sha_v3 13e4f69c…`, alive 26 / hazard 0.1385 / clvl >= 4
   26). **Making this rule act pre-emptively needs a different mechanism (a much larger ring, a lower count, or an
   "approach speed" criterion); that would be a new ruling.**
2. **Rule (b) was not removed.** The reviewer offered the option "or adopt the report's own 13.3 and delete rule
   (b)". Rule (b) is **part of the ruling**, and deleting it is not something a review can do. I gave it a release
   path that fires. I also record **a change I got wrong and reverted**: I first added "unlock when readiness
   recovers" (release on `ratio >= 1.0 or cleared`). Rule (a) refuses exactly when `ratio < 1.0 and not cleared`,
   so the two are **complements**: that release would make `stall_lock` **unreachable forever**, deleting rule (b)
   under the name of a repair. `CoachV05StallLockTests` caught it at once; it was reverted, and the episode is
   kept as a comment in `resource_protocol.py`.
3. **`gain_denominator` was not written into the spend ledger page.** The ranking function now carries
   `gain_denominator` on every leg (pinned by a test), but **the candidate records in the probe rows do not have
   this key**; adding it would consume the seed pool once more to rerun four arms purely for telemetry. The three
   legs' denominators can be recovered exactly from `ratio / ratio_normalised` (7.0 / 4.0 / 5.0 in this round),
   so it is **recorded as a gap** (section 14.13), not rerun.

## 16.8 The five review-round arms (48 seeds per arm, pool 2_133, same worker, same control rows)

| Arm | `rows_sha_v3` | alive | L2 | L3 | L2 hazard/1k | L2 beats | L2 deaths | clvl >= 4 | paired vs control UCB95 | paired vs M2-SET |
|---|---|---|---|---|---|---|---|---|---|---|
| control rows / OFF | `0e5a1acd…` (match) | 20 | 34 | **4** | 0.2220 | 85578 | 19 | 25 | 0 | - |
| **M2-SET** (rulings 1/2/3) | `f33f7af5…` | **28** | 30 | 2 | **0.1374** | 87358 | **12** | 18 | **-0.01860** | - |
| M2-FULL' | `72a86601…` | 22 | 30 | 3 | 0.1948 | 82150 | 16 | 22 | +0.09506 | saved 2 / lost 8 |
| M2-SET+coach' | `615f1281…` | 26 | 30 | 2 | 0.1737 | 80607 | 14 | 16 | +0.01734 | saved 0 / lost 2 |
| M2-SET+stop' | `c79526a8…` | 25 | 30 | 2 | 0.1528 | 91604 | 14 | 23 | +0.04315 | saved 2 / lost 5 |

The same-named arms from **before** the review, side by side (same seeds, same worker):

| Arm | `rows_sha_v3` | alive | hazard/1k | L2 beats | clvl >= 4 |
|---|---|---|---|---|---|
| M2-FULL | `68f41f5c…` | 26 | 0.1507 | 86267 | 21 |
| M2-SET+coach | `27b722fb…` | 25 | 0.1794 | 78026 | 16 |
| M2-SET+stop | `9842d5f8…` | 28 | 0.1270 | 94473 | 23 |

**Reading (without embellishment)**:

1. **Rulings 1/2/3 are unaffected and the conclusion stands.** M2-SET is bit-for-bit unchanged (H1 is the
   evaluator CLI and H4 is recorded-only telemetry; neither is on the decision path) and is still the campaign's
   only arm with UCB95 < 0.
2. **Fixing the vehicle** turns stop-v1 from "5 saved, 5 lost, 5 deaths inside the vehicle" into "**0 deaths
   inside the vehicle**, alive 25". **Net survival goes from 0 to -3** (against M2-SET). This is not significant
   (only 7 discordant, UCB95 0.15195), but the direction is not good, and what it buys back is **depth**:
   clvl >= 4 18 -> 23, L2 beats +4.9%.
3. **With a scope and a release path, coach-v05 still only loses** (saved 0 / lost 2). The recommendation in
   section 13.3 stands.
4. **The two new rules together (M2-FULL') are worse than either alone** (alive 22), in the same direction as
   before the review.
5. **A methodological sentence that must be recorded**: this round ran 5 arms again on the same 48-seed pool (OFF/
   FULL/STOP twice). **The overfitting risk on this pool is accumulating** (section 14.11).

## 16.9 Review-round artifacts (local R19 work directory, not published)

| Purpose | Path |
|---|---|
| all change scripts (anchor-by-anchor replacements, reproducible) | `m2rr/p1_protocol.py` … `p16_section16.py` |
| corrected test-union script | `m2rr/rr_tests.sh` |
| test logs / failure sets | `m2rr/rr-{pristine,merged}{.log,-fails.txt,-ids.txt}` |
| probe driver (5 arms, reusing M2's statistics code) | `m2rr/rr_driver.py` |
| probe scripts (two passes) | `m2rr/rr_probe.sh`, `rr2_probe.sh` |
| per-arm rows | `m2rr-probe/rr-{off,set,full,coach,stop}-rows.json` |
| arm summaries | `m2rr-probe/rr-summary-{all,"off,full,stop"}.json` |
| **radius-3 counterfactual arm (reverted change)** | `m2rr-probe/radius3/` |
| review-round reading script / output | `m2rr/rr_read.py`, `rr-read-final.txt` |
| tree fingerprints (before / after, two passes) | `m2rr/rr{,2}-tree-{before,after}.txt` |
| identity check + patch regeneration | `m2rr/rr_patch.sh` |
| **patch against the main tree (25 files, +10028 / -215 content lines, all LF, sha256 `315133a531bbc13d…`)** | `m2.patch` (regenerated; the line-count convention is in the corresponding ledger entry) |
| backup of this report from before the review | `m2rr/M2-REPORT.md.prereview.bak` |

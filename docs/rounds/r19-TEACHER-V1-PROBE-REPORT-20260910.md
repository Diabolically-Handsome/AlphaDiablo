# R19 oracle teacher, step one (teacher-v1 probe): report, filed version (2026-09-10)

**Verdict: the teacher (teacher-v1) did not beat the student (7e31dc54). After the review round fixed every
blocking finding and the three arms were rerun, the conclusion is unchanged.**

| Arm (M2-SET world, pool 2_133, 48 seeds) | alive at game end | L1 deaths | L1 hazard per 1,000 ticks | reached L3 | row hash rows_sha_v3 |
|---|---|---|---|---|---|
| PARENT 7e31dc54 (regression, = M2-SET bit for bit) | 28 | 8 | 0.0213 | 2 | f33f7af5… |
| ORACLE (frozen dispatch inner loop, no instincts) | 12 | 3 | 0.0100 | 26 | 1507c3be… (bit-identical in both rounds) |
| TEACHER-V1 (ORACLE + instinct layer, with the terrain fix) | 7 | 10 | 0.0366 | 24 | 26d5c8f5… |

- Paired (event criterion): TEACHER-V1 vs PARENT, death by game end saved 2 / lost 23 (net -21, UCB95 +0.574);
  L1 death saved 7 / lost 9 (net -2, UCB95 +0.178).
- ORACLE vs PARENT, L1 death saved 7 / lost 2 (net +5, UCB95 -0.004, **lost if a single seed flips**); not
  reaching L3 net +24 (UCB95 -0.372).
- Instinct layer isolated (TEACHER-V1 vs ORACLE): L1 death net -7, survival at game end net -5 (the +4 reported
  in round one had saved/lost swapped; corrected in the review round).
- The disengage clause's "1 tile of separation gain per tick" is a tautology; measured, in round one 85% of
  disengage ticks did not move (0.060 tiles/tick), and after the terrain fix 63.5% still did not move (0.180
  tiles/tick).
- The explosion of resupply windows (10244 vs the parent's 3815) did **not** go away when disengage ticks were
  halved; a13 pickups rose from 857 to 2088; the pickup clause is now the prime suspect, and two L1 deaths
  (2133027, 2133029) had zero disengage ticks.
- The potion clause took effect 159 times and was refused 395 times by the potion-autonomy mask (autonomy opens
  only the [0.50, 0.75) hit-point band, once per window).

**Archive**: the full report, the module `teacher_v1.py` (+69 unit tests), the drivers, all row files and
receipts of the three arms, the round-one originals and `teacher-v1.patch` (the diff against m2-merge) were
filed in `train/runs/r19-reports/teacher-v1/` (not published). The main-tree fingerprint was bit-identical before
and after the work (`main-tree-fingerprint-{before,after}-teacher.txt`, sha 2781e74bbd3276cf); zero writes to
the M2 merge tree; the fresh pools were not touched.

**Next step (approved 2026-09-10)**: a second round of zero-training probes. A: the oracle takes only FARM windows
(DIVE windows go back to the student); B = A + the potion clause; C = B + contact-only disengagement (hp <= 35%
and a monster within 1 tile, with a walkability filter); D = ORACLE + the pickup clause only (ablation, to check
the resupply-window explosion). The criteria remain L1 deaths and paired survival.

---

The implementer's report follows (all numbers come from its own output files).

# R19 TEACHER-V1 report (the oracle as teacher, step one: first prove that the teacher beats the student)

- Date: 2026-09-10 (**round two = review round**; round one on the same day)
- Role: teacher-v1 implementer
- Basis: the design ruling of 2026-09-10 (recorded in the ledger, not published): oracle thinking, i.e. scripts
  become teachers, not runtime rules; first prove that the teacher beats the student.
- Where things lived: the work tree, scripts and outputs named below were kept in a local R19 work directory and
  are not published; paths are relative to that directory.
- Work tree: `oracle-tree/` (`rsync -a --exclude __pycache__` from the read-only `m2-merge`; the `build` symlink
  points to the R17.1 resource build)
- The main tree and the M2 merge tree `m2-merge`: **read-only, zero writes** (re-verified file by file with
  sha256 in this round, see section 1).
- Seeds: only pool 2_133, 2133000-2133047, in 3 shards (a 2133000-2133015 / b 2133016-2133031 /
  c 2133032-2133047), plus a second run of shard a as a **terrain-neutral control**. Fresh pools 2_116-119 and
  2_126-128: **not touched** (this round enumerated every seed in every output to check).
- All round-one outputs are kept unchanged in `round1/`; anything this report says "round one said" can be
  checked there.
- Every number in this report comes from files produced in this round, with the source marked at each place;
  anything without a supporting file is marked "not verified".

---

## 0. One-sentence conclusion (VERDICT; still holds after the review round)

**The teacher still did not beat the student. After fixing the problems found in the review round, the
conclusion did not flip.**

- **TEACHER-V1 (base + instinct layer) vs PARENT: neither line passed.** Alive at game end **7 / 48 vs 28 / 48**
  (paired saved 2 / lost 23, net **-21**, one-sided UCB95 **0.57378**); **L1 deaths 10 vs 8** (paired saved 7 /
  lost 9, net **-2**, UCB95 **0.17839**). The terrain fix brought L1 deaths down from 12 in round one to 10 and
  disengage ticks from 1748 to 859, but **neither line passed**.
- **ORACLE (only the frozen inner loop dispatch, no instincts) vs PARENT: one win and one loss, bit-identical to
  round one.** `rows_sha_v3 = 1507c3be…8748acbe`, **exactly equal** to round one, which by itself proves that
  "this round's changes have zero effect on the base". **L1 deaths 3 vs 8** (saved 7 / lost 2, net **+5**, UCB95
  **-0.00437**); games reaching L3 **26 vs 2**; but alive at game end 12 vs 28.
- **The instinct layer is still negative on the same base**: TEACHER-V1 vs ORACLE, L1 death saved 1 / lost 8
  (net **-7**), alive at game end saved 3 / lost 8 (net **-5**).
- "Did it wreck killing or depth?" Still **no**: depth rose sharply (L3 from 2 to 24, depth_mean 1.667 -> 2.562),
  and total kills fell 15% (5550 -> 4734). **What collapsed is survival at game end, and the cause is still
  depth.**
- **The most important new fact of the review round**: round one reported "disengagement buys 1.000 tile per
  tick" as a measurement. This round measured for the first time "did this step actually move", and the answer
  is: **85.3% of round one's disengage ticks did not move** (643 of 754 settled disengage ticks in shard a), and
  the measured separation gain was only **45 tiles** in total, not 754. With the terrain filter this improves to
  **63.5% not moving**, with a measured gain of 153 tiles / 850 ticks: **still "neither fighting nor really
  getting away"**.

---

## 1. Review round: the 12 findings, one by one

The external review returned `fix-first`, with 4 high, 4 medium and 4 low findings. One by one below. **"Fixed" =
code or report changed in this round; "escalated" = needs a design decision, not something the implementer can
settle.**

| # | Level | Finding | Action |
|---|---|---|---|
| 1 | high | The report said twice that "the 400-tick budget used at most 133 ticks", but 133 is the **arm-level total** `disengage_no_improving_move`; the real per-game maximum is **272 ticks** (seed 2133005, 272/808 = 33.7%), **68%** of the fence, not 33% | **Fixed**: numbers corrected; the **distribution** of "disengage ticks per game" is now a table (`tables.md` §T6) with min/median/mean/max, the number of games above 100 ticks, max/budget, and the top 10 games, so it cannot be confused with `no_improving_move` again |
| 2 | high | The constants block called 400 "about 2.4% of a game's 16879 worker ticks"; `m2-summary-all.json` contains no 16879 at all, and the denominator is not worker ticks | **Fixed**: recomputed from round one's own `rows[].teacher.beats`: 48 games, **51631** worker ticks in total, mean **1075.6**, median **514.0**, min **86**, max **11166**; so 400 ticks is **37.2% of the mean game and 77.8% of the median game**, and **13/48 games have fewer than 400 ticks in total**, where the fence cannot possibly apply. **The value 400 is the literal ruled value and is not changed**; only the reasoning changed. Whether the fence should take another shape (a share of the game's ticks / a cap on consecutive ticks) is **escalated** |
| 3 | high | `_disengage_move` looked only at the worker mask, not at walkability: the mask in `env.py` (then lines 2248-2252) masks only "progress-protection trigger tiles", while walls, closed doors and soft walls **are never masked**, so disengaging into a wall re-sends the same action every tick | **Fixed** (behaviour change): the disengage target now reads `bridge.local_map(radius=1)` and excludes non-walkable / door / hazard / explosive_softwall, the same exclusions as the certified `resource_emergency_stop.choose_disengage_tile`; **the only channel not read is `local_map["monster"]`**, which contains unlit monsters behind walls (`env.py`, then lines 6089-6091), i.e. coordinates in darkness, and the teacher's charter is "visible monsters only". A **terrain-neutral control** shows that the new engine query does not disturb the world (section 3) |
| 4 | high | The potion-pickup clause (Chebyshev 1 block) sits above the disengage clause (Chebyshev 2 trigger); the two radii are inconsistent, and **no counter** could separate these cases from the 856 pickups | **Counter fixed**: new `pickup_while_disengage_trigger`; this round measured **9 of 2086 pickups (0.4%) overriding a live disengage trigger**. **Radii and clause order escalated**: both are literal ruled values ("no visible monster within 1 tile -> a13", pickup before disengage); changing them is a ruling, not implementation |
| 5 | medium | `disengage_separation_gain` can structurally only ever be +1 (one Chebyshev step + a strict-increase requirement), yet it was reported in the telemetry table as a measurement, and it recorded the **intended target**, counting +1 even for moves that did not execute | **Fixed**: renamed `disengage_intended_gain` and marked as an identity in the constants block and the tables; a real measurement was added: on the next tick, compare the player tile and `min_visible_monster_distance` from the real raw state, producing `disengage_moved` / `disengage_not_moved` / `disengage_achieved_gain`. Results in sections 0 and 7.1 |
| 6 | medium | The report said three times that ORACLE's L1 bound -0.00437 was "the only comparison this round with a one-sided 95% upper bound below 0", while its own table had "did not reach L3" below 0 too | **Fixed**: changed to "**the only survival / L1-death comparison below 0**" (which is true). `teacher_extra2.py` now prints all 12 paired comparisons and counts how many are below 0: **4 in this round** (see section 6) |
| 7 | medium | The only clean win (ORACLE L1 3 vs 8, UCB95 -0.00437) clears the line by only 0.004, on 9 discordant pairs; flipping one seed reverses the sign | **Fixed**: the driver now computes single-seed sensitivity into `teacher-summary.json`, reported in section 6.1: flip one pair -> **+0.03924**; turn one into a concordant pair -> **+0.01156**. Both are above 0. **Whether to recheck ORACLE's L1 result on a fresh pool is escalated** |
| 8 | high | `diablogym_worker_action12_mode` was hard-coded as `"permanently-masked"`, while M2-SET has `drink_sovereignty=True`, so `options_env` would raise `RuntimeError`; the three arms ran only because `teacher_shard.py` patched it from outside | **Fixed, but differently from the review's suggestion**: the review suggested "derive it from the env in `bind()`". **Tried; it does not work**: `OptionsEnv.__init__` uses exactly this worker attribute to resolve the global `drink_sovereignty` (`options_env.py`, then lines 125-161, called at line 590), before any `bind()`, and a wrong value raises `ValueError` at env construction (this exception was observed in this round). So instead `make_teacher()` **requires** an explicit `drink_sovereignty` (no default): the teacher declares its own contract, and `bind()` cross-checks it against the env. The unit tests check it with `options_env`'s two real validation functions |
| 9 | low | Section 6.1 said "10 games had clvl only 1-2" while its own table had 11; it also said all 12 were ground down while backing off, yet 2133027 / 2133029 had 0 disengage ticks | **Fixed**: round one's numbers recomputed as **11/12**; 2133027 / 2133029 confirmed at 0 disengage ticks, and in this round they **still** die on L1 with 0 disengage ticks (section 7.4). This is the only direct evidence in the whole round that "the disengage clause is not the only killer", listed separately in section 7.4 and gap 8 of section 10 |
| 10 | low | `_base` called `env._worker_masks_and_distance()` a second time each tick; `env.py` (then lines 2259-2268) accumulates `_aggro_cap_fired` while building masks, which would double that telemetry in a world with aggro_cap on | **Fixed**: with `strict=True`, `bind()` **refuses** an env with aggro_cap on, and the module docstring records the limitation. aggro_cap is off in M2-SET, so the three arms of this round are unaffected |
| 11 | low | Three paths have no production evidence: `grace_decisions = 0`, `base_fallbacks = 0` (the first rung of `BASE_FALLBACK_LADDER`, a10, is a behaviour choice the ruling did not name), `illegal_guard` unreachable under `strict=True`, and the adjacent-monster tiebreak `adj` is always 0 | **Marked, not changed**: all four places state in the code "never executed in 96 real games", and they are named in the telemetry table in section 7 and the gaps in section 10. In this round `grace_decisions` and `base_fallbacks` are **still 0** |
| 12 | low | The regression receipt cited the pre-review `m2-probe/m2-set-rows.json`, not the review-round output `m2rr-probe/rr-set-rows.json` | **Fixed**: the driver now **cites both files** and on every run recomputes both files' `rows_sha_v3` with its own formula into `teacher-probe/regression-receipt.json`. Both measure `f33f7af5…ded75da4` in this round |

### 1.1 One more finding from the review round itself (the review missed it)

**Round one's section 5 entry "TEACHER-V1 vs ORACLE, alive at game end = saved 8 / lost 4 / net +4" had the
wrong sign.**

`teacher_extra.py`'s `paired(arm, ctl, pred)` scores `pred` as a **bad event** (`saved` = the arm does not have
the event while the control does), but round one passed `alive`, a **good event**, for the survival comparison,
so saved and lost were swapped wholesale. Recomputed from round one's own rows: the correct value is **saved 4 /
lost 8 / net -4, UCB95 +0.20039**. A quick check: in round one teacher-v1 had only 8 survivors and ORACLE 12, and
**the arm with fewer survivors cannot have a positive net**. Round one's reading 2 in section 5, "net +4 on
survival at game end", was therefore wrong: the instinct layer is negative against the base on survival at game
end as well. This round's `teacher_extra2.py` uses `died` as the predicate; the section 6 table is produced by
the driver's own `paired()` (which was right from the start).

### 1.2 Read-only tree re-verification (redone this round)

| Check | Result |
|---|---|
| sha256 of `env.py` / `options_env.py` / `worker_env.py` / `resource_emergency_stop.py` / `probe_r17_deployment.py` against `m2-merge` | **all five SAME** |
| files in `m2-merge` changed today | **0** |
| latest mtime of main-tree `python/diablogym/*.py` | still **2026-09-07** |
| main-tree files changed today | **only the ledger `r13_ledger.jsonl`** |
| seeds appearing anywhere in this round's outputs | 2133000-2133047, **48, no exception**; fresh pools 2_116-119 / 2_126-128 **not touched** |
| `source_changed_during_run` of the three arms | all `[]` |

---

## 2. Deliverables (local work directory, not published)

| Item | Path | first 16 of sha256 |
|---|---|---|
| teacher module | `oracle-tree/train/runs/r10-staging/teacher_v1.py` | `65c1a14f029e7311` |
| unit tests | `oracle-tree/train/runs/r10-staging/test_teacher_v1.py` | `f12d94d98f911e24` |
| shard runner | `teacher_shard.py` | `1bf9a8ad45696e46` |
| probe driver | `teacher_probe_driver.py` | `49448d8e91f262e9` |
| review-round diagnostics | `teacher_extra2.py` | `bdd292b2f5aef90d` |
| terrain-neutral control | `observe_control.py` | `d6f45d51ac9b15bd` |
| table generator | `gen_tables.py` | `e51d6d3806e72c95` |
| rows / statistics / pairing | `teacher-probe/{parent,oracle,teacher-v1}-rows.json`, `teacher-summary.json` | - |
| regression receipt | `teacher-probe/regression-receipt.json` | - |
| neutral-control receipt | `teacher-probe/terrain-neutrality-receipt.json` | - |
| tables and diagnostics | `teacher-probe/{tables.md,extra2.txt}` | - |
| unit-test log | `teacher-tests-rr.log` | - |
| all round-one outputs | `round1/` | - |

**Rule impact: none.** `teacher_v1.py` stays in `train/runs/r10-staging/` (probe side), is not in the
`probe_r17_deployment._SOURCE_FILES` fingerprint bundle **or in any other fingerprint bundle**; it registers no
rule, changes no threshold and is not imported by any runtime path.

---

## 3. What the teacher is (after the review round)

`make_teacher(variant, drink_sovereignty, ...) -> callable`, satisfying the attribute contract of
`load_zip_policy`, and it can be placed directly into `OptionsEnv(workers={FARM: cb, DIVE: cb})`; `cb.bind(env)`
binds the env, and the teacher reads `env.env._raw` exactly as `probe_r17_deployment.run_episode` does.

Decision order on each worker tick (**clause order and thresholds unchanged, all literal ruled values**):

0. **gear grace**: `a14 if legal else a0`.
1. **Instinct (1), potion discipline**: `belt > 0 and hp < 0.60*max` -> `a12` (respecting the mask).
2. **Instinct (2), potion pickup**: `a13 legal` and **0 visible monsters within 1 tile** -> `a13`.
3. **Instinct (3), DISENGAGE**: triggers on `(hp <= 0.60*max and >= 2 visible live monsters within 2 tiles)` or
   `(hp <= 0.35*max and >= 1 visible live monster within 1 tile)`; among **legal and walkable** `a1..a8`, choose the
   target that most strictly increases "minimum distance to visible monsters" (ties: first closer to an up
   staircase within 12 tiles, then fewer adjacent monsters, then the lowest action number); if there is none, fall
   back to the base; budget of 400 ticks per game.
4. **base = the frozen oracle inner loop** `options_env.dispatch` (verbatim, not one line changed).

**What the review round changed (only two items change behaviour or interface):**

1. **Clause 3 gained a terrain filter** (review high 3). This is the only change in this round that changes the
   action output.
2. **`drink_sovereignty` became a required parameter of `make_teacher`** (review high 8). An interface change, not
   an action change.

Everything else is **telemetry and documentation**: the rename to `disengage_intended_gain`, the new
`disengage_moved / _not_moved / _achieved_gain / _settled`, `disengage_terrain_blocked_dirs /
_terrain_unavailable / _no_walkable_improving_move`, `pickup_while_disengage_trigger`, and the aggro_cap refusal
gate in `bind()`.

### 3.1 Terrain-neutral control (proves that the new engine query does not disturb the world)

The module gained a third mode, `terrain="observe"`: **it still queries `bridge.local_map` on every disengage
tick and still counts which directions it would filter out, but does not filter**. Shard a (2133000-2133015) was
rerun with it:

```
round2 observe  rows_sha_v3 = ff09b3b763fb48acde653668b0bce650ce3853a44eee3e86928a537b29ede50d
round1 teacher-v1 shard a   = ff09b3b763fb48acde653668b0bce650ce3853a44eee3e86928a537b29ede50d
equal = True   row_field_diffs = []   teacher_telemetry_diffs = []   runtime_errors = 0
```
(`teacher-probe/terrain-neutrality-receipt.json`)

**Conclusion: the new `bridge.local_map` query has zero effect on the engine, and this round's refactoring has
zero effect on decisions.** Every difference between round 2 and round 1 can only come from the terrain filter
itself.

**And this control incidentally measured round one's most important number** (see section 7.1).

### 3.2 Constants block (the only one) and where each number comes from

| Constant | Value | One-line reason (source) |
|---|---|---|
| `POTION_HP_FRACTION` | 0.60 | in `a13-trace-2133002.json`, ticks 57-72, hp is stuck at 49/70 = 0.70 and a12 is legal throughout, yet the belt waits until 46/70 and 36/70 to drink; `drink_sovereignty` opens the worker's a12 exactly in `[0.50, 0.75)`, and 0.60 is the midpoint of this only worker-owned band |
| `PICKUP_BLOCK_MONSTER_RADIUS` | 1 | same trace, tick 66: walking to a floor item with a monster already adjacent is exactly how this game died. **[review] this 1 is inconsistent with the disengage clause's 2**; it is a literal ruled value; this round only adds a counter, and the radii and clause order are escalated |
| `DISENGAGE_CROWD_*` | 0.60 / 2 / 2 | ledger `R19_DIAG_2133002_PER_BEAT_AUDIT`: "beats 57-65: hp 49/70, 1-3 adjacent monsters … chose a9 every beat" |
| `DISENGAGE_CONTACT_*` | 0.35 / 1 / 1 | `M2-REPORT.md` section 10 review round: in 13 of stop-v1's 50 activations hit points were already <= 20% of maximum; 0.35 sits above the 0.50 reflex drain band |
| `DISENGAGE_EPISODE_BUDGET_BEATS` | 400 | **[corrected in review]** round one's "about 2.4% of a game's 16879 worker ticks" was wrong. From round one's own `rows[].teacher.beats`: 48 games, 51631 worker ticks, mean 1075.6 / median 514.0 / min 86 / max 11166 -> 400 ticks is **37.2% of the mean game and 77.8% of the median game**, and **13/48 games have fewer than 400 ticks in total**. The value stays (literal ruled value); a different shape is **escalated** |
| `STAIRS_TIEBREAK_RADIUS` | 12 | the same search as in `resource_retreat.py` (then line 221); 12 is the engine's own candidate radius |
| `DISENGAGE_TERRAIN_RADIUS` | 1 | **[new in review]** disengagement is a single Chebyshev step, so only the 8 neighbouring tiles can be targets; 1 is the smallest `local_map` radius covering them. Not a threshold, a geometric fact of a1..a8 |

---

## 4. Regression (the first gate)

PARENT arm = worker `7e31dc5402caed73…`, M2-SET world (17 keys), 6000 steps, `sample`, 3 shards merged and sorted
by seed. **Rerun in this round, round one's shards not reused**:

```
rows_sha_v3 = f33f7af5f236f1fc273293160910b5606f7d03f8a08e91c499ed1a45ded75da4
expected    = f33f7af5f236f1fc273293160910b5606f7d03f8a08e91c499ed1a45ded75da4
equal       = True
```

**[review 12 fixed]** Both certified row files are cited, and **recomputed in this round with the driver's own
formula**:

| Certified row file | `rows_sha_v3` recomputed in this round |
|---|---|
| `m2rr-probe/rr-set-rows.json` (review-round output) | `f33f7af5…ded75da4` |
| `m2-probe/m2-set-rows.json` (before the review round) | `f33f7af5…ded75da4` |

`all_references_agree = True` (`teacher-probe/regression-receipt.json`). `alive 28 / L1 deaths 8 / L2 hazard
0.1374 / L2 kills 826 / L3 2` also match the M2-SET row of `M2-REPORT.md`. **The regression passes.**

| Arm | rows_sha_v3 | vs round one | runtime_errors | wall time |
|---|---|---|---|---|
| PARENT | `f33f7af5…ded75da4` (**= expected**) | **bit-identical** | 0 | 602.1 s |
| ORACLE | `1507c3be68a0da2991802392605fee9026952329697397fa6f36ba6f8748acbe` | **bit-identical** | 0 | 489.3 s |
| TEACHER-V1 | `26d5c8f52ed5e566f420961e32a3b97139132c7ecb705d75be2f45d01ce70e0a` | changed (terrain filter) | 0 | 634.4 s |

> ORACLE being bit-identical is **independent evidence of correctness**: this round changed the module in many
> places (required parameter, telemetry rewrite, terrain layer, aggro_cap gate), and the base arm without
> instincts did not change by a single byte, which shows those changes really only touch the instinct layer.

`source_changed_during_run` is empty for all three arms.

---

## 5. Full table (arm_stats)

| Column | PARENT (7e31dc54) | ORACLE | TEACHER-V1 (round 2) |
|---|---|---|---|
| alive (48 games) | **28** | 12 | 7 |
| deaths | 20 | 36 | 41 |
| **L1 deaths** | **8** | **3** | 10 |
| L2 deaths | 12 | 14 | 12 |
| deaths by floor | {1:8, 2:12} | {1:3, 2:14, 3:11, 4:7, 5:1} | {1:10, 2:12, 3:11, 4:6, 5:1, 6:1} |
| L2 beats | 87358 | 58735 | 43289 |
| L2 hazard /1k | 0.1374 | 0.2384 | 0.2772 |
| L2 kills | 826 | 763 | 625 |
| L1 kills | 4723 | 4534 | 3994 |
| kills total | 5550 | 5469 | 4734 |
| clvl >= 4 | 18 | **20** | 16 |
| clvl mean | 3.229 | 3.438 | 3.062 |
| reached L2 | 30 | **41** | 35 |
| **reached L3** | 2 | **26** | 24 |
| depth mean | 1.667 | **2.646** | 2.562 |
| micro_steps median | 12000.0 | 9302.0 | 8164.0 |
| median tick of L1->L2 | 6798.0 | 6064 | 6059 |
| belt at L1->L2, median / mean | 6.0 / 5.967 | 5 / 5.146 | 5 / 5.143 |
| mean AC at L1->L2 | 17.067 | 16.293 | 15.029 |
| alive at first descent + 1800 / observable denominator | 34 / 37 | 27 / 40 | 24 / 36 |
| gold collected / sold / spent | 8625 / 9066 / 23048 | 8638 / 7274 / 19738 | 6270 / 4917 / 15684 |
| sweep windows / chests / barrels / sarcophagi | 752 / 339 / 376 / 334 | 786 / 363 / 417 / 339 | 701 / 332 / 360 / 300 |
| grab windows / piles / gold | 571 / 729 / 8695 | 547 / 727 / 8701 | 468 / 617 / 6992 |
| retreats | 58 | 87 | 71 |
| belt 0 at death | 11/20 | 21/36 | 23/41 |

All 48 columns are in `teacher-probe/tables.md` §T1.
> `sweep_gold` is 0 in all three arms: a definition issue of the sweep-v2 telemetry itself (gold goes through
> `gold_delta` and the gold-grab channel), not a difference between arms; copied as is, not explained.

### 5.1 Exposure and hazard per floor (`teacher-probe/extra2.txt`)

| Arm | L1 ticks / deaths / hazard per mille | L2 ticks / deaths / hazard per mille | L3 ticks / deaths / hazard per mille | L4 ticks / deaths / hazard per mille | L5 ticks / deaths / hazard per mille |
|---|---|---|---|---|---|
| PARENT | 374770 / 8 / **0.0213** | 87358 / 12 / 0.1374 | 935 / 0 / 0.0 | 0 / 0 / - | 0 / 0 / - |
| ORACLE | 300406 / 3 / **0.0100** | 58735 / 14 / 0.2384 | 9809 / 11 / 1.1214 | 5372 / 7 / 1.3031 | 1008 / 1 / 0.9921 |
| TEACHER-V1 | 273193 / 10 / **0.0366** | 43289 / 12 / 0.2772 | 6674 / 11 / 1.6482 | 12252 / 6 / 0.4897 | 526 / 1 / 1.9011 |

**This table is still the pivot of the whole report.** Replace "alive at game end" with "death rate per 1,000
ticks on the same floor":

- **ORACLE is more than twice as safe as the parent on L1** (0.0100 vs 0.0213).
- **TEACHER-V1 is still more dangerous than the parent on L1** (0.0366 vs 0.0213); the terrain fix brought it
  down 22% from round one's 0.0469, but it is still 1.7 times the parent.
- On L2 the parent is the safest of the three arms; **the hazard on L3/L4/L5 is 5-10 times that of L2**, and only
  the two teacher arms actually went there (the parent spent only 935 ticks on L3 and never reached L4/L5).
  **"Alive at game end" is not a fair question between two arms whose depth differs this much.**

### 5.2 Depth-reached ladder

| Arm | reached L1 | L2 | L3 | L4 | L5 |
|---|---|---|---|---|---|
| PARENT | 48 | 30 | 2 | 0 | 0 |
| ORACLE | 48 | 41 | **26** | 11 | 1 |
| TEACHER-V1 | 48 | 35 | 24 | 12 | 3 |

---

## 6. Paired tests

**[review 6 fixed]** Below are **all 12** paired comparisons, not a selection; `teacher_extra2.py` itself counts
how many one-sided 95% upper bounds are below 0.

| Pair (event = bad, more negative is better) | saved | lost | net | one-sided UCB95 |
|---|---|---|---|---|
| ORACLE vs PARENT, death by game end | 2 | 18 | -16 | 0.46458 |
| **ORACLE vs PARENT, death on L1** | **7** | **2** | **+5** | **-0.00437** |
| ORACLE vs PARENT, did not reach L3 | 25 | 1 | +24 | **-0.37177** |
| ORACLE vs PARENT, did not reach L2 | 14 | 3 | +11 | **-0.09876** |
| TEACHER-V1 vs PARENT, death by game end | 2 | 23 | -21 | 0.57378 |
| TEACHER-V1 vs PARENT, death on L1 | 7 | 9 | -2 | 0.17839 |
| TEACHER-V1 vs PARENT, did not reach L3 | 22 | 0 | +22 | **-0.34003** |
| TEACHER-V1 vs PARENT, did not reach L2 | 12 | 7 | +5 | 0.04315 |
| **TEACHER-V1 vs ORACLE, death by game end (instinct layer isolated)** | 3 | 8 | **-5** | 0.21511 |
| **TEACHER-V1 vs ORACLE, death on L1 (instinct layer isolated)** | **1** | **8** | **-7** | 0.24264 |
| TEACHER-V1 vs ORACLE, did not reach L2 | 2 | 8 | -6 | 0.22923 |
| TEACHER-V1 vs ORACLE, did not reach L3 | 5 | 7 | -2 | 0.15997 |

**4 of the 12 comparisons have a one-sided 95% upper bound below 0**, all of them depth comparisons or ORACLE's
L1. So the correct statement is: **-0.00437 is the only "survival / L1 death" comparison of this round below 0**,
not "the only one".

Three readings:

1. **The base (frozen dispatch) beats the student on L1**, and this is the only thing in this round that clears
   the line in the **survival / L1 class** (-0.00437).
2. **On the same base the instinct layer pushes L1 deaths from 3 to 10** (saved 1 / lost 8). **It is also negative
   against the base on survival at game end** (saved 3 / lost 8, net -5); round one's reported net +4 was a sign
   error, see section 1.1.
3. Both teacher arms beat the student decisively on depth, **and that is the direct cause of their collapse in
   survival at game end**.

### 6.1 **[review 7]** Single-seed sensitivity of ORACLE's L1 win

| Case | saved | lost | net | UCB95 |
|---|---|---|---|---|
| measured | 7 | 2 | +5 | **-0.00437** |
| any saved pair flipped to lost | 6 | 3 | +3 | **+0.03924** |
| any saved pair turned concordant | 6 | 2 | +4 | **+0.01156** |

**This 95% result is only 0.004 below the line, and flipping one seed removes it**; it also rests on pool 2_133,
which this report itself admits is accumulating overfitting (section 10, gap 7). Anyone reading "ORACLE is the
teacher available now" must read this row with it.

---

## 7. The teacher's own telemetry

| Telemetry | ORACLE | TEACHER-V1 (round 2) | round-one TEACHER-V1 |
|---|---|---|---|
| games / worker ticks | 48 / 51381 | 48 / 53570 | 48 / 51631 |
| instinct (1) drinks | 0 | **159** | 154 |
| instinct (2) potion pickups | 0 | **2086** | 856 |
| instinct (3) disengage ticks | 0 | **859** | 1748 |
| of which crowd / contact condition held | 0 / 0 | 788 / 353 | 1655 / 784 |
| disengage miss (no legal walkable improving target) | 0 | 252 | 133 |
| disengage miss (no legal move at all) | 0 | 0 | 0 |
| **intended separation gain (tiles, always equal to the tick count)** | 0 | **859** | 1748 |
| **[new] ticks that actually moved** | 0 | **310 (36.5%)** | not measured |
| **[new] ticks that did not move** | 0 | **540 (63.5%)** | not measured |
| **[new] measured separation gain (tiles)** | 0 | **153** (0.180 / settled tick) | not measured |
| [new] directions filtered by terrain | 0 | 1602 | - |
| [new] ticks without a terrain source | 0 | **0** | - |
| [new] pickup overriding a live disengage trigger | 0 | **9** (0.4% of 2086 pickups) | not measured |
| stairs tiebreak applied | 0 | 240 | 219 |
| ticks stopped by the 400-tick budget / games that exhausted it | 0 / 0 | **0 / 0** | 0 / 0 |
| wanted to drink but refused by the env mask | 0 | **395** | 475 |
| gear grace decisions | **0** | **0** | 0 |
| base illegal fallback | **0** | **0** | 0 |
| games with at least one instinct trigger | 0 | 47 / 48 | 47 / 48 |
| action histogram | `{9:14386, 10:19456, 11:16294, 13:909, 14:336}` | `{1:79, 2:187, 3:16, 4:310, 5:11, 6:142, 7:15, 8:99, 9:11914, 10:13789, 11:24041, 12:159, 13:2088, 14:720}` | - |

### 7.1 **[review 3+5] The most important table of this round: did disengagement actually move**

Round one reported "`disengage_separation_gain / disengage_beats = 1748 / 1748 = 1.000` tile per tick". The review
pointed out that this is an **identity** (one Chebyshev step changes the distance by at most 1, and the ranking
requires a strict increase), and that it records the **intended target**, counting +1 even when the move did not
execute. This round measured the real effect for the first time:

| Criterion | disengage ticks | settled | actually moved | did not move | intended gain | **measured gain** |
|---|---|---|---|---|---|---|
| **round-one behaviour** (shard a, neutral control rerun) | 761 | 754 | 111 (**14.7%**) | 643 (**85.3%**) | 761 tiles | **45 tiles** (0.060/tick) |
| **this round + terrain filter** (all 48 games) | 859 | 850 | 310 (**36.5%**) | 540 (**63.5%**) | 859 tiles | **153 tiles** (0.180/tick) |

Two things must be read out:

1. **In round one, 85% of the disengage clause's ticks did not move at all.** The reported "1.000 tile bought per
   tick" is really **0.060 tiles**, off by a factor of 17. This is not a misinterpretation in round one's report;
   it **never measured this** (round one's own gap 4 in its section 9 admits "how many are empty moves: not verified").
2. **The terrain filter is the right direction, but far from enough.** Move rate 14.7% -> 36.5% and measured gain
   0.060 -> 0.180 tiles/tick are both threefold improvements; yet **63.5% of disengage ticks still do not move**.
   Walls are not the only cause; the most likely remaining cause is **a monster standing on the target tile** (we
   **deliberately do not read** the `local_map["monster"]` channel, because it contains monsters in darkness and
   would break the teacher's "visible monsters only" charter). **Escalated**: whether the teacher may read the
   physical occupancy channel is a ruling.

### 7.2 The real distribution against the 400-tick budget (`tables.md` §T6)

| Distribution | ORACLE | TEACHER-V1 (round 2) | round one |
|---|---|---|---|
| disengage ticks per game, min / median / mean / **max** | 0 / 0 / 0 / 0 | 0 / 5.0 / 17.9 / **149** | 0 / 14.5 / 36.4 / **272** |
| max / 400-tick budget | 0 | **0.372** | **0.68** |
| games above 100 ticks | 0 | 2 | 6 |
| games with fewer than 400 worker ticks in total (fence cannot apply) | 3 / 48 | **11 / 48** | **13 / 48** |
| worker ticks per game: total / min / median / mean / max | 51381 / 75 / 594.0 / 1070.4 / 11166 | 53570 / 86 / 523.5 / 1116.0 / 11166 | 51631 / 86 / 514.0 / 1075.6 / 11166 |

- Round one said "the heaviest game used 133 ticks, a threefold margin under the fence"; **the true value is 272
  ticks, a 1.5-fold margin**.
- This round's terrain filter brings the maximum down to 149 ticks (a 2.7-fold margin).
- But **in both rounds about a quarter of the games have fewer than 400 worker ticks in total**, where the fence
  **cannot apply mathematically**. **The fence is still inert**, and this round again did **not** test whether 400
  is the right number.

### 7.3 A structural side effect: the window count explodes (round one's explanation is overturned)

| Arm | manager windows total | farm | dive | resupply | mask_forced | descents total |
|---|---|---|---|---|---|---|
| PARENT | 3815 | 1341 | 763 | 1711 | 2208 | 167 |
| ORACLE | 4396 | 1688 | 973 | 1735 | 2453 | 219 |
| TEACHER-V1 (round 2) | **10244** | 1391 | 1248 | **7605** | **8293** | 183 |
| TEACHER-V1 (round one) | 10007 | 1337 | 1205 | 7465 | 8108 | 174 |

Round one attributed this to the **disengage actions** ("a1-a8 produce no positive progress inside the window, so
the window stalls and closes"). **This round halved the disengage ticks (1748 -> 859) and the window total rose
slightly (10007 -> 10244), so that attribution does not hold.** What moved in the same direction is potion
pickup: a13 rose from 857 to 2088, and RESUPPLY windows 7465 -> 7605. This report **states only the correlation
and draws no causal conclusion**; establishing causality needs clause-by-clause ablation (section 10, gap 8),
which this round did not do.

### 7.4 Ten L1 deaths, and what each instinct did

| Seed | death tick | last hp/max | belt | kills | clvl | drink / pickup / disengage | disengage moved / did not move |
|---|---|---|---|---|---|---|---|
| 2133001 | 764 | 4/70 | 0 | 19 | 1 | 2 / 0 / **149** | 10 / **138** |
| 2133008 | 866 | 2/78 | 0 | 42 | 2 | 2 / 0 / 22 | 20 / 1 |
| 2133011 | 2855 | 6/78 | 0 | 64 | 2 | 2 / 10 / 49 | 24 / 25 |
| 2133014 | 2165 | 4/78 | 0 | 51 | 2 | 2 / 1 / 62 | 11 / **50** |
| 2133018 | 2221 | 1/78 | 0 | 49 | 2 | 0 / 0 / **103** | 4 / **98** |
| 2133021 | 665 | 5/70 | 0 | 19 | 1 | 1 / 0 / 8 | 7 / 1 |
| 2133026 | 665 | 3/70 | 0 | 25 | 1 | 1 / 0 / 12 | 11 / 1 |
| **2133027** | 883 | 4/70 | 0 | 20 | 1 | 1 / 0 / **0** | 0 / 0 |
| **2133029** | 2821 | 11/78 | 0 | 68 | 2 | 3 / 1 / **0** | 0 / 0 |
| 2133042 | 3365 | 1/86 | 0 | 93 | 3 | 1 / 0 / 63 | 25 / **38** |

**Pathology: all 10 died with an empty belt, and 9 had clvl only 1-2** (the correct round-one number for the same
cell is 11/12; the report text said 10, see review 9). The three heaviest games (2133001 / 2133018 / 2133014) are
still **dozens to a hundred and fifty ticks of backing off, most of them not moving**.

**[review 9, listed separately] `2133027` and `2133029` have 0 disengage ticks**, 0 in both rounds, and both die
on L1 in both rounds. They are the only direct evidence in the whole round that **the disengage clause is not the
only killer**: in those two games only the potion clause and the pickup clause could act. This is exactly why
gap 8 in section 10 asks for clause-by-clause ablation.

Control: ORACLE's 3 L1 deaths are 2133017@852 / 2133021@2563 / 2133026@551; the parent's 8 are 2133008@746 /
2133009@2466 / 2133015@3011 / 2133017@1039 / 2133023@4724 / 2133024@2104 / 2133039@2047 / 2133044@833.

---

## 8. Named seeds

**First a correction of the criterion (made in round one, restated here).** The seven named L1-death seeds
(2133002, 2133021, 2133022, 2133038, 2133006, 2133039, 2133014) are the L1-death set of the **v1-world CONTROL
arm**, **not of M2-SET**. In the certified M2-SET PARENT, only **2133039** of these seven still dies on L1; M2-SET's
own 8 L1-death seeds are **2133008, 2133009, 2133015, 2133017, 2133023, 2133024, 2133039, 2133044**.

| Seed | PARENT | ORACLE | TEACHER-V1 (round 2) |
|---|---|---|---|
| 2133002 | alive depth 1 (clvl 3, 132 kills) | alive depth 2 (clvl 3, 126 kills) | died L4@7321 (clvl 4, 128 kills, belt 0); drink 4 / pickup 10 / disengage 5 |
| 2133021 | died L2@7387 (clvl 3, 103 kills) | died L1@2563 (clvl 2, 68 kills) | died L1@665 (clvl 1, 19 kills); drink 1 / pickup 0 / disengage 8 |
| 2133022 | alive depth 2 (clvl 4, 142 kills) | alive depth 1 (clvl 3, 92 kills) | died L6@8724 (depth 6, clvl 3, 109 kills, belt 6) |
| 2133038 | alive depth 2 (clvl 4, 162 kills) | **alive depth 3** (clvl 5, 178 kills) | died L5@12454 (depth 5, clvl 5, 170 kills, belt 5) |
| 2133006 | alive depth 2 (clvl 4, 148 kills) | died L2@7942 (clvl 3, 107 kills) | **alive depth 3** (clvl 5, AC 25, 161 kills) |
| 2133039 | **died L1@2047** (clvl 2, 69 kills) | died L2@6698 (clvl 3, 81 kills) | died L2@9721 (clvl 2, 54 kills) |
| 2133014 | alive depth 2 (clvl 5, 168 kills) | died L2@9799 (depth 3, 116 kills) | died L1@2165 (clvl 2, 51 kills); disengage 62 |
| 2133010 | alive depth 2 (clvl 4, 171 kills) | died L3@9285 (depth 4, 135 kills) | died L3@12633 (clvl 5, 170 kills, belt 7) |

- **2133039** (the only M2-SET seed that is both named and really dies on L1): round one's teacher-v1 saved it;
  **after this round's terrain fix it dies on L2@9721**. In other words, round one's "most beautiful rescue" **did
  not reproduce after the fix**, which shows how fragile these single-seed stories are and why they should not be
  used as evidence.
- **2133038 / 2133022**: the parent survives; teacher-v1 dies in both rounds, just deeper this round (L5 / L6).
- **2133014**: killed on L1 by teacher-v1 in both rounds; this round 50 of its 62 disengage ticks did not move.

The per-seed 48-row comparison table is in `teacher-probe/tables.md` §T5.

---

## 9. Unit tests (`teacher-tests-rr.log`, all 69 pass)

```
==== 69 passed, 0 failed ====
```

37 in round one, 69 in this round; all 32 new ones are review-round fixes:

1. **Contract (review 8)**: `make_teacher` **refuses** to give `drink_sovereignty` a default; the mode declared
   for each of the two values is checked with `options_env`'s **real** validation functions
   (`_resolve_worker_drink_sovereignty`, `_validate_worker_action12_contract`); **round one's hard-coded
   `"permanently-masked"` is rejected by the real resolver under M2-SET** (pinned by a unit test); the
   cross-check between `bind()` and the env raises.
2. **Terrain (review 3)**: when the best target is a wall, the next-best walkable target is chosen (a6 -> a8); the
   door / hazard / explosive channels can each exclude; when every improving target is a wall, it falls back to
   the base and counts separately; without a terrain source it falls back to the mask alone and counts.
   **Regression pin**: from 8 identical wall-facing states, round one would send the same wall-blocked move 8 times
   `[6,6,6,6,6,6,6,6]` (this round **pins that defect in a unit test**), and with the terrain filter on it becomes
   `[9,9,9,9,9,9,9,9]` (fallback to base).
3. **Movement telemetry (review 5)**: one case each for did not move / actually moved / measured gain 0 because a
   monster followed; **on 4000 random states it proves that `disengage_intended_gain` is always +1** (an identity,
   not a measurement).
4. **Pickup counter (review 4)**: with hp 21/70, three visible monsters at Chebyshev 2 and a13 legal, the teacher
   **still** returns a13 as ruled (clause order unchanged), but this time **it is counted**; pickups without a
   live trigger are not counted.
5. **aggro_cap gate (review 10)**: `bind()` raises under `strict=True` and lets it through under `strict=False`.
6. **8000 random synthetic states x (with random terrain) x two variants**: 100% of emitted actions are inside the
   mask.
7. **Determinism**: seed 2133002 run twice gives a bit-identical 555-tick action sequence, field-identical
   telemetry and the same `rows_sha_v3` (`bc4fd514…48718300`); it also asserts that this game **really took the
   terrain path** (`disengage_terrain_unavailable = 0`).

---

## 10. Methodological gaps (self-reported, updated in the review round)

1. **"Alive at game end" is not a fair question between two arms whose depth differs by about 0.9 levels.** The
   per-floor hazard table (section 5.1) and the "did not reach L3" pairing (section 6) are given as remedies, but
   a matched-depth arm that pins the teacher to L1/L2 was **not** run; it needs a new world configuration, outside
   this round's authorisation.
2. **The 400-tick budget**: round one's margin number was wrong (133 -> 272), and so was its reasoning (2.4% ->
   37%); both are corrected. But **the fence was never reached in either round**, and **about a quarter of the
   games have fewer than 400 worker ticks in total**, where it cannot apply mathematically. **Whether the number
   is right is still untested.** A different shape (share of the game / cap on consecutive ticks) is
   **escalated**.
3. **The disengage clause still does not move on 63.5% of its ticks.** The terrain filter improved this from 85.3%
   to 63.5%, the right direction, but not a solution. The most likely remaining cause is **a monster standing on
   the target tile**, and the `local_map["monster"]` channel contains monsters in darkness; reading it would break
   the teacher's "visible monsters only" charter. **Whether to allow it is escalated.**
4. **About 70% of the potion clause's decisions cannot be carried out**: 395 ticks wanted to drink and were
   refused by the mask, 159 ticks drank (395 / 554 = 71.3%; round one 475 / 629 = 75.5%). Making 0.60 actually take
   effect would require changing the safety envelope of `drink_sovereignty` or its "once per window" latch, a
   **rule change** this round was explicitly not allowed to make, so it is recorded, not changed.
5. **The adjacent-monster tiebreak `adj` is unreachable in practice** (strict increase => target minimum distance
   >= 2 => `adj` always 0). It is written as ruled, but it is dead code; marked in the code.
6. **Three paths have zero production evidence in both rounds** (review 11): `grace_decisions = 0`,
   `base_fallbacks = 0` (the first rung of `BASE_FALLBACK_LADDER`, a10, is a behaviour choice the ruling did not
   name), and `illegal_guard` is unreachable under `strict=True`. Only unit tests cover them, no games.
7. **Seed-pool overfitting keeps accumulating**: pool 2_133 was consumed **4 more times** in this round (three arms
   + the shard-a neutral control). **ORACLE's single L1 win is only 0.004 below the line and disappears if one seed
   flips** (section 6.1); nothing should be built on it before it is confirmed on a fresh pool. **Escalated.**
8. **The instinct layer still has no clause-by-clause ablation** (drink only / pickup only / disengage only).
   "TEACHER-V1 vs ORACLE" in section 6 can only show that the three together are negative. **2133027 and 2133029
   die on L1 with 0 disengage ticks** (section 7.4), direct evidence that "it is not only the disengage clause";
   without the ablation we cannot say which clause is negative.
9. **Round one's attribution of the window explosion was wrong** (section 7.3): halving the disengage ticks did
   not reduce the window total. a13 moved in the same direction (857 -> 2088). **Correlation only, no causality**;
   this also needs the ablation.
10. `sweep_gold` is 0 in all three arms (a telemetry definition issue); not pursued in this round.

---

## 11. Recommendations (not rulings)

1. **Step one still did not pass.** Per the ruling, "teacher-v1 beats the parent" is the precondition gate for
   BC/DAgger; after fixing everything the review round found, teacher-v1 **still did not win** (alive 7 vs 28, L1
   deaths 10 vs 8, neither line passed). **Going straight to step two is not recommended.**
2. **Do not discard the oracle inner loop along with it, but do not treat it as a proven win either.** Without the
   instinct layer, **ORACLE** is still the only thing in this round that clears the line in the survival / L1
   class (L1 3 vs 8, UCB95 -0.00437), and it raises games reaching L3 from 2 to 26 while total kills drop only
   1.5%. **But -0.00437 is only 0.004 below the line, and flipping one seed makes it +0.039**, on a pool that has
   been consumed repeatedly. To use ORACLE as BC demonstration data, **first confirm this L1 result on a fresh
   pool**.
3. **What the instinct layer needs to change is still the disengage mechanism, not its thresholds.** This round
   proved two things: (a) round one's "1 tile bought per tick" is an identity, not a measurement, and the real
   value is 0.060 tiles/tick; (b) with walkability it only reaches 0.180 tiles/tick, and 63.5% of ticks still do
   not move. A next version needs three things, **each of them a ruling**: (i) whether the physical occupancy
   channel may be read (a monster standing on the target tile); (ii) a give-up discipline when stuck (abort the
   clause after N consecutive ticks without moving); (iii) a **floor scope**: on L1 a clvl 1 character should fight,
   not back off (9 of this round's 10 L1 deaths are clvl 1-2).
4. **The pickup clause is now the most suspicious, and nobody has watched it.** a13 rose from 857 to 2088 in this
   round, while the frozen base asked for it only 2 times; the two L1 deaths with zero disengage ticks can only be
   blamed on the potion or pickup clause; the window explosion moves with it. **Recommendation: do the
   clause-by-clause ablation first, then tune.**
5. For the potion clause, either change the rule (the `drink_sovereignty` envelope; needs a ruling) or reduce it to
   "drink as early as possible within the `[0.50,0.75)` window the env allows"; the latter changes no rule.

---

## 12. Constraints for a teacher-to-student step (from an unpublished design draft)

A later, unpublished design draft for turning a teacher into training signal for the student recorded the hard
constraints that any such step must satisfy in this codebase:

- **Drinking and equipping cannot be distilled.** The exam contract excludes actions 12 and 14 from distillation
  (`LEGACY_DISTILLATION_EXCLUDED_ACTIONS = (12, 14)` in `train/leashed_ppo.py`; the `excluded_actions` check in
  `train/eval_assembled.py`), so a teacher's a12 (drink) and a14 (equip) decisions never reach the student through
  distillation.
- **Distillation rewrites only the 298-wide root.** A scheduled distillation must use the `legacy-root-logits`
  scope, i.e. the legacy 298-wide observation root (`ASYMMETRIC_WORKER_LEGACY_DIM = 298`); the asymmetric context
  branch is not a distillation target.
- **The anneal must close within the leg.** The distillation anneal (`distill_anneal_actor_rollouts`) must finish
  within the leg's actor rollouts (512 rollouts in a 1,048,576-step leg); otherwise `anneal_closed` in
  `_validate_asymmetric_worker_runtime_state` never holds and the exam rejects the checkpoint.
- **Resume legs inherit the critic warm-up receipt.** A leg continued with `--resume-from` carries the previous
  leg's critic warm-up evidence, and `train_ppo.py` rejects combining `--resume-from` with a fresh `--bc-init`
  warm start.
- **Attaching a BC teacher needs new flags.** The training CLI has no way to attach a behaviour-cloning teacher to
  such a leg today; the draft estimated about 40 lines of new flags and validation.
- **The teacher must be a pure function of (observation, mask).** Its labels must be reproducible from what the
  student sees; teacher-v1 clauses that depend on other state (the stairs tiebreak and the per-game 400-tick
  budget) would have to be dropped, and the instinct clauses judged from the 298-wide observation only.

---

*This report was written by the teacher-v1 implementer on 2026-09-10 (review round). All numbers come from files
this round produced under `teacher-probe/`, and round-one numbers from the unchanged `round1/` (both in the local
work directory, not published); every place without a source is marked "not verified".*

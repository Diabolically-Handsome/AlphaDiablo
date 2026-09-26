# R19 oracle teacher, step one, round two (per-window split and clause-by-clause ablation): report, filed version (2026-09-11)

**Verdict: one arm beat the student on level 1 without giving up survival or kills: ORACLE-FARM (the oracle inner
loop takes only FARM windows; DIVE windows go back to the student 7e31dc54).** But stated as "what was proven":
what won is "the oracle's FARM inner loop + the student's own DIVE behaviour". The level-1 statistic is
bit-identical to round one's ORACLE arm (the same three level-1 death seeds 2133017/2133021/2133026, the same
saved 7 / lost 2 / net +5 / UCB95 -0.00437), so round two adds no independent evidence for the level-1 claim.
What it adds is: "once the DIVE windows go back to the student, the oracle's level-1 discipline no longer costs
survival and kills".

| Arm (M2-SET, pool 2_133, 48 seeds; row hashes identical over three reruns) | alive at game end | L1 deaths | L1 hazard per 1,000 ticks | L2 hazard per 1,000 ticks | total kills | clvl >= 4 | reached L3 | rows_sha_v3 |
|---|---|---|---|---|---|---|---|---|
| PARENT 7e31dc54 (regression = M2-SET) | 28 | 8 | 0.0213 | 0.1374 | 5550 | 18 | 2 | f33f7af5… |
| ORACLE (regression = round one) | 12 | 3 | 0.0100 | 0.2384 | 5469 | 20 | 26 | 1507c3be… |
| **ORACLE-FARM** | **28** | **3** | **0.0072** | **0.1059** | **6353** | **31** | 5 | e47d5a58… |
| OF-POTION (+ potion clause) | 20 | 8 | 0.0214 | 0.1494 | 5938 | - | 4 | 17d3ea46… |
| OF-POTION-CONTACT (+ contact disengagement) | 20 | 8 | 0.0218 | - | 5917 | - | 4 | 525d18ef… |
| ORACLE-PICKUP (oracle + pickup clause only) | 9 | 4 | 0.0133 | 0.2551 | 5295 | - | 29 | 3f02ce54… |

Paired (event criterion, one-sided UCB95 from m2_probe_driver; 36 comparisons without multiplicity correction,
1.8 expected to cross the line under the null, 7 did, reducing to 3 facts):
- ORACLE-FARM vs PARENT: death by game end saved 9 / lost 9 (net 0); L1 death saved 7 / lost 2 (net +5, UCB95
  -0.004, **lost if a single seed flips**); not reaching L2 net +5.
- ORACLE-FARM vs ORACLE: death by game end saved 18 / lost 2 (net +16, UCB95 -0.202): once the DIVE windows go back
  to the student, the oracle no longer dies deep.
- Potion clause (OF-POTION vs ORACLE-FARM): L1 death saved 2 / lost 7 (net -5), death by game end net -8:
  **harmful**, and 53% of the wanted drinks are refused by the potion-autonomy mask.
- Contact disengagement (vs OF-POTION): 0/0 everywhere, **no effect**; it triggered on only 132 ticks, and 68% of
  those did not move (round one's defect is not cured, only smaller).
- Pickup clause (ORACLE-PICKUP vs ORACLE): L1 net -1 (harmless), but resupply windows 7761 vs 1735: **the cause of
  the resupply-window explosion is settled**. Pickup is a multi-tick walking macro; once it leads the character
  away from monsters, `_farm_handoff` masks FARM, the manager has only RESUPPLY left, opens a 9.7-tick short window,
  closes it, and loops. It is not that a13 is pressed more often (698 vs 909).

Routing uses the engine's own mechanism (`self._workers.get(int(option))` in `options_env.py`); all 85 unit tests
pass; in real games the tick counts served by the two callables match the environment ledger window by window;
oracle slot 38794 FARM ticks, student slot 51624 DIVE ticks (33901 of them in mask_forced windows). The main-tree
fingerprint was bit-identical before and after the work (a7a4c75c93b2cd1f); zero writes to m2-merge and
oracle-tree; teacher_v1.py untouched; fresh pools not touched.

**Archive**: the full report (including review-round section 16), the pass-1 report, `diagnostics.md`,
`teacher_v2.py` + 85 unit tests, the drivers, all row files and receipts of the six arms, and
`teacher-round2.patch` were filed in `train/runs/r19-reports/teacher-round2/` (not published).

**Two decisions requested**:
1. Replicate the level-1 claim: on a fresh pool (suggested 2_116, 48 seeds) run only the PARENT and ORACLE-FARM
   arms, zero training; this consumes one fresh pool.
2. Make ORACLE-FARM the teacher for step two: its FARM inner loop is essentially the generation-1 teacher of
   bc_worker (`dispatch("farm")`), so step two can reuse the existing gen-1 BC pipeline (M2-SET world + a new
   demonstration pool 2_144 + attachment flags) without any instinct clause, and the "teacher-v1-obs" and a12/a14
   exclusion problems of the second revision of the R19-A draft (not published) go away. Decision requested on
   whether to write a third revision on that basis.

---

The implementer's report follows (all numbers come from its own output files).

# R19 TEACHER round-two report (composite arm ORACLE-FARM and clause-by-clause ablation)

- Date: 2026-09-10 to 2026-09-11 (round two; round one and its review round were both on 2026-09-10)
- Role: teacher round-two implementer
- Basis: the round-two arms approved on 2026-09-10 (ledger event `R19_TEACHER_ROUND2_PLAN`; ledger not
  published), with authorisation to execute this round, **and only this round**, on its own.
- Where things lived: the work tree, scripts and outputs named below were kept in a local R19 work directory and
  are not published; paths are relative to that directory.
- Work tree: `oracle2-tree/` (`rsync -a --exclude __pycache__ --exclude .pytest_cache` from round one's
  **read-only** `oracle-tree`; the `build` symlink points to the R17.1 resource build)
- Read-only, zero writes: the main tree, `m2-merge`, `oracle-tree` (see section 12)
- The only new code is one module, `train/runs/r10-staging/teacher_v2.py`, and its own `test_teacher_v2.py`.
  **Round one's `teacher_v1.py` / `test_teacher_v1.py` are untouched** (sha256 bit-identical to `oracle-tree`,
  section 12). **No C++ or engine change, and no registered rule was touched.**
- Seeds: only pool 2_133, 2133000-2133047, in 3 shards (a/b/c). Fresh pools 2_116-119 and 2_126-128: **not
  touched** (section 12 enumerates all 48 seeds found in every output of this round).
- All numbers in this report come from this round's own files in `round2/probe/`; round-one numbers are marked
  "round one" and come from the unchanged `teacher-probe/`; anything without a supporting file is marked "not
  verified".
- **Review round** (2026-09-11): an independent review gave 7 findings (`verdict: ship`, no blocker or major),
  **all fixed**; two of them touched telemetry, so **all six arms were rerun and both regression gates were passed
  again**, and **`rows_sha_v3` is bit-identical for all six arms**. The whole account is in **section 16**, the
  final verdict in **section 16.4**.

---

## 0. One-sentence conclusion (VERDICT)

**One arm clears the line on L1 deaths: `ORACLE-FARM`. But this sentence must be written out in full, or it will
be read as something it cannot support (review-round finding 3, section 16):**

> **What won is "the oracle's FARM inner loop", not "a scripted teacher beat the student".**
> In `ORACLE-FARM`, **57.1% of worker ticks (51624 / 90418) are still played by the student itself** (33901 of
> them in `mask_forced` windows); the whole DIVE window goes back unchanged to 7e31dc54, and the oracle served only
> 38794 FARM ticks.
> **And the L1 statistic that carries this conclusion is the same statistic as round one's `ORACLE`**: the two
> arms' three L1 death seeds are **exactly the same** (2133017 / 2133021 / 2133026), the paired numbers are
> bit-identical (saved 7 / lost 2 / net **+5** / UCB95 **-0.00437**), and the L1 pairing of `oracle-farm` against
> `oracle` is **0 / 0 / 0**.
> **So round two provides no independent confirmation of that L1 result.**
> What round two really adds is something else: **once the DIVE windows go back to the student, the oracle's L1
> discipline no longer comes at the price of "charging down and dying"**: survival at game end stays at 28 vs 28,
> kills rise 14.5%, while round one's `ORACLE` had only 12 survivors.

`ORACLE-FARM` = **FARM windows go to the frozen oracle inner loop, DIVE windows go back to the student 7e31dc54**.

| vs PARENT | ORACLE-FARM | PARENT | paired (event = bad) |
|---|---|---|---|
| **death on L1** | **3** | 8 | saved 7 / lost 2 / net **+5** / UCB95 **-0.00437** |
| alive at game end | **28** | 28 | saved 9 / lost 9 / net 0 / UCB95 +0.14540 |
| total kills | **6353** | 5550 | +14.5% |
| games with clvl >= 4 | **31** | 18 | - |
| L1 hazard per 1,000 ticks | **0.0072** | 0.0213 | one third |
| L2 hazard per 1,000 ticks | **0.1059** | 0.1374 | - |
| alive 1800 ticks after the first descent | **42 / 44** | 34 / 37 | - |

**The question as posed, answered literally:** "Did any arm beat the student on L1 deaths (paired net > 0 and
one-sided UCB95 < 0) **while** not wrecking survival at game end and kills?" **Yes, and only ORACLE-FARM**
(`ORACLE`'s L1 pairing also clears the line, but it drops survival at game end from 28 to 12, so it fails the
second half). It brings L1 deaths from 8 down to 3 (UCB95 -0.00437), **survival at game end stays at 28 vs 28**,
**kills even rise 14.5%**, and depth only rises slightly from 1.667 to 1.833 (not the oracle's charge downstairs).

**The three ablations partly contradict round one's guesses:**

1. **The drink clause, not the pickup clause, is the L1 killer.** `OF-POTION` (= ORACLE-FARM + the drink clause
   only) pushes L1 deaths **straight back from 3 to 8** (paired against ORACLE-FARM: L1 saved 2 / lost 7 / net
   **-5**; death by game end net **-8**, survival 28 -> 20). **On its own it wipes out the whole L1 win.**
2. **The contact disengage clause does almost nothing.** `OF-POTION-CONTACT` vs `OF-POTION`: L1 death saved 0 /
   lost 0, death by game end saved 1 / lost 1, and the eight L1 death seeds are **exactly the same**. Over all 48
   games it triggers on only 193 ticks and really fires on 132 (round one: 859), and **68.0% (87/128) still do not
   move**: the old defect is not cured, just too small to see.
3. **The pickup clause really causes the RESUPPLY window explosion; round one guessed this one right.**
   `ORACLE-PICKUP` (= the full oracle base on both window types + the pickup clause only) blows RESUPPLY windows up
   from ORACLE's **1735 to 7761** and total windows from 4396 to 10560, almost exactly reproducing round one's
   TEACHER-V1 figures of 7605 / 10244. **But it is nearly harmless on L1** (4 L1 deaths vs the oracle's 3, paired
   net -1). So item 4 of section 11 of round one's report, "the pickup clause is now the most suspicious", **was
   right about the window explosion and wrong about L1 harm**.

**Why the composite arm works has mechanistic evidence, not luck:** the engine's own audit (in `options_env.py`,
aggregated window by window in this round by a read-only env subclass) shows that **77% of the oracle's ticks in
DIVE windows are idle**: of 18514 ticks only 4298 are actually executed (`no_effect_requests` 14216,
`fuse_trips` 567). Once the DIVE windows go back to the student, the execution rate in those windows becomes
**37939 / 51676 = 73% executed**. The oracle can fight but cannot go downstairs; the student can go downstairs
but cannot fight on L1. **The composite arm gives each job to the one that can do it.**

**A sentence that must be read together with the one above:** `ORACLE-FARM`'s L1 win **is the same statistic as
round one's ORACLE**: the three L1 death seeds are exactly the same (2133017 / 2133021 / 2133026), so
**-0.00437 is only 0.004 below the line and becomes +0.039 if one seed flips** (section 6.1), and it rests on pool
2_133, which has been consumed repeatedly. **Nothing should be built on this 95% result before it is replicated
on a fresh pool.**

---

## 1. What this round asks, and the six arms it ran

Round one concluded that the frozen oracle base alone halves the L1 per-tick hazard (0.0100 vs the student's
0.0213) but **charges down and dies** (26 games reach L3, survival at game end 12 vs 28), and that the instinct
layer (drink at 0.60 / pickup / crowd + contact disengagement) is **net negative** on the same base (L1 deaths
3 -> 10). Round one reported two gaps itself: "no clause-by-clause ablation of the instinct layer" and "the window
explosion is attributed by correlation only, not causally".

Six arms in this round, in the same M2-SET world (17 keys, **copied verbatim** from `teacher_probe_driver.py`),
on the same 48 seeds of pool 2_133 and the same three shards a/b/c:

| Arm | FARM windows | DIVE windows | Purpose |
|---|---|---|---|
| `parent` | student 7e31dc54 | student | regression gate, must equal `f33f7af5…` |
| `oracle` | teacher_v1 "oracle" | the same object | regression gate, must equal `1507c3be…` |
| `oracle-farm` | teacher_v1 "oracle" | student 7e31dc54 | **composite arm** |
| `of-potion` | oracle base + drink clause | student | drink clause only |
| `of-potion-contact` | oracle base + drink + contact disengagement | student | adds contact disengagement |
| `oracle-pickup` | oracle base + pickup clause | **the same object** (both window types) | pickup clause only |

`RESUPPLY` windows (and the sweep / gold-grab / portal / retreat stages under them) are **scripted rules
throughout and never go through a worker**, so `workers` has no RESUPPLY entry at all, and the `resupply` tick
count is **0 in all six arms** (section 7). This was measured in this round, not assumed.

---

## 2. Deliverables (local work directory, not published)

| File | What it is |
|---|---|
| `oracle2-tree/train/runs/r10-staging/teacher_v2.py` | the only new code of this round: `make_teacher2(variant, env=None, drink_sovereignty=True, parent_zip=…)` |
| `oracle2-tree/train/runs/r10-staging/test_teacher_v2.py` | this round's unit tests: 70 pass in the first pass (`round2/tests-full.log`), 85 pass after the review round tightened and added tests (`round2/tests-full-review.log`) |
| `round2/teacher2_shard.py` | one shard, one arm; rows go through `probe_r17_deployment.run_episode` unchanged |
| `round2/teacher2_driver.py` | six-arm driver, regression gates, `arm_stats`, paired tests |
| `round2/teacher2_extra.py` | generator of every table in this report (reads only this round's outputs) |
| `round2/probe/*-rows.json` | all rows and telemetry of the six arms |
| `round2/probe/regression-{parent,oracle}.json` | receipts of the two regression gates |
| `round2/probe/round2-summary.json` | summary (including all 36 pairings) |
| `round2/diagnostics.md` | machine-generated raw tables |
| `round2/tests-full.log` / `tests-nonepisode.log` | unit tests |
| `round2/driver-*.log`, `round2/probe/*.cmd` | a receipt for every command line |
| `round2/gate_then_arms.sh`, `run_regressions.sh`, `finalize.sh`, `smoke.sh`, `run_tests_*.sh`, `deploy.sh` | scripts actually run in the first pass |
| `round2/rr_full.sh`, `rr_regressions.sh`, `rr_arms.sh`, `prelaunch_review.sh`, `compare_passes.py`, `ledger_review.py` | scripts actually run in the **review round** (section 16) |
| `round2/probe-pass1/`, `round2/probe-pass2/` | all outputs of the first pass and the review round's first rerun, kept unchanged |
| `round2/review-round-compare.txt` | per-arm SHA / `arm_stats` comparison across the three passes |
| `round2/review-round-manifest-before.txt` | sha256 + timestamp of every script about to run, taken **before launch** (finding 4) |
| `round2/tests-full-review.log` | the review round's full unit-test log |
| `round2/TEACHER-ROUND2-REPORT.pass1.md`, `driver-*.pass1.log` | report and logs from before the review, kept unchanged |
| `round2/mk_report.py`, `round2/verdict.json` | this report's generator and the verdict Q&A |
| `round2/main-tree-fingerprint-{mid,after}.txt`, `…-review-{before,after}.txt` | proof that the main tree stayed read-only (first pass + review round) |

---

## 3. What the teacher is, and why the "routing" is verified rather than guessed

`teacher_v2.py`, like `teacher_v1.py`, is a **probe-side policy**: it registers no rule, changes no threshold, is
not imported by any runtime path, lives next to the probe in `train/runs/r10-staging/`, and **is not part of the
`probe_r17_deployment._SOURCE_FILES` fingerprint bundle or any other fingerprint bundle**.

**Routing uses the engine's own mechanism; code locations as of that work tree (then `options_env.py` line
numbers):**

- `options_env.py:2815`: `worker = self._workers.get(int(option))`: each option window **resolves its worker
  once**, and then the whole window (`:3112-3137`) goes through that one callable.
- `options_env.py:67`: `FARM, DIVE, RESUPPLY = 0, 1, 2`; `:1465`: `_win["mode"] = ("farm","dive","resupply")[option]`:
  **the option is the window type**.
- So `workers={FARM: oracle_cb, DIVE: parent_cb}` is the engine's native per-window dispatch, and no callable has
  to peek at the window type.
- A `RESUPPLY` window runs entirely through the scripted service chain in `:2822-3110`; sweep / gold-grab /
  portal / retreat are stages of the same chain (`:2846-2851`) and also never touch a worker.
- `forced_dive` / `mask_forced` are **manager-side** facts (`m[FARM] = not forced_dive`, `:1089` and `:1371`),
  recorded per window by the probe (`run_episode:600-621`). This round records the tick count of every DIVE window
  by `_win["window_id"]` and **joins** it with the probe row's own `dive_windows[].forced` to get the
  "forced-dive ticks" column in section 7.

**Hooks**: `run_episode` hands `on_beat` / `episode_reseed` to **one** `cb` only (`:564` and `:567`) and cannot
see `workers`. So `TeacherV2` is a **routing object**: `.workers` goes into the env, the object itself goes into
`run_episode` as `cb`, and its `on_beat` setter / `episode_reseed` / `bind` **fan out to every inner callable**.
Since only one callable is called per tick, `on_beat` **still fires exactly once per worker tick** (counted
directly in the unit tests: 3 ticks -> 3 calls). The student side's seeding discipline
(`model.set_random_seed(seed)` once per game, `load_zip_policy:201-203`) is **bit-identical** to the certified
control rows.

**Constants**: this round **introduces no new numeric threshold**. The single constants block of
`teacher_v2.py` **imports the values unchanged** from round one's frozen constants block, each with its source
(drink at 0.60, pickup Chebyshev 1, contact 0.35 / radius 1 / count 1, the 400-tick fence, terrain radius 1).
**The crowd trigger and `STAIRS_TIEBREAK_RADIUS` are deliberately not imported, per this round's plan** (no crowd
trigger, no stairs tiebreak).

**Two differences from round one's shard, both read-only and both shown by the regression gates to have zero
effect:** (1) each worker callable is wrapped in a `WorkerProbe` (count first, then forward unchanged); (2) the env
uses `teacher_v2.observing_env_class()`, an `OptionsEnv` subclass that only **reads the info dict already
returned** after `super().step()`, to aggregate the `worker_no_effect_requests` audit that the probe rows do not
carry. Both regression SHAs reproduce exactly (section 4), **which is the proof, not a claim, that these two
things do not disturb the world**.

---

## 4. Regression (the first gate)

  parent: got f33f7af5f236f1fc... expected f33f7af5f236f1fc... equal=True all_refs_agree=True
      ref m2rr-probe/rr-set-rows.json: f33f7af5f236f1fc
      ref m2-probe/m2-set-rows.json: f33f7af5f236f1fc
      ref teacher-probe/parent-rows.json: f33f7af5f236f1fc
  oracle: got 1507c3be68a0da29... expected 1507c3be68a0da29... equal=True all_refs_agree=True
      ref teacher-probe/oracle-rows.json: 1507c3be68a0da29

**Both gates pass**, and the `parent` SHA agrees at the same time with three independent existing outputs
(`m2rr-probe/rr-set-rows.json`, `m2-probe/m2-set-rows.json`, and round one's `teacher-probe/parent-rows.json`),
while `oracle` agrees with round one's `teacher-probe/oracle-rows.json`. **Only after the gates passed were the
four new arms started** (`round2/rr_full.sh` hard-codes this discipline as a conditional: the four new arms start
only if both receipts have `equal=true`, both have `all_references_agree`, and both are `"pass": "run"`;
otherwise `exit 2`).

**This discipline can now be checked on disk (fix for review-round finding 4)**: in the first pass the `--reuse`
summarising leg rewrote both receipts, so their mtimes ended up later than the four arms they were supposed to
gate, and mtimes alone could not prove the order. Now the `--reuse` leg writes sibling files
`probe/regression-*-reuse.json` (`"pass": "reuse-summarize"`); **the gate receipts `probe/regression-parent.json`
/ `regression-oracle.json` are written only by the leg that actually runs (`"pass": "run"`), and the gate itself
now requires `pass == "run"`**. File times of the review round's final pass, relative to the pre-launch manifest:

| File | time after the pre-launch manifest | Meaning |
|---|---|---|
| `round2/review-round-manifest-before.txt` | 0 | sha256 of every script about to run, **before launch** |
| `probe/regression-oracle.json` | +8 min 32 s | oracle regression receipt (the one that finished first) |
| `probe/regression-parent.json` | +9 min 53 s | student regression receipt |
| `probe/r2-oracle-farm-a.cmd`, `r2-of-potion-a.cmd` | +9 min 53 s | command-line receipts of the four new arms, **written in the same second, after the gate decision** |
| `probe/regression-*-reuse.json` | +21 min 20 s | sibling files of the `--reuse` summarising leg, **no longer overwriting the gate receipts** |

Neither gate receipt is later than the four new arms' `.cmd` files (the oracle's is 81 s earlier, the student's
in the same second: the gate decision `GATE result: YES` was printed in that second, and the four arms started in
the same second); the `REGRESSION` lines (`equal` true) in `driver-parent.log` / `driver-oracle.log` also come
before the launch; `round2/review-round-manifest-after.txt` is the same table taken at the end.

### 4.1 Run receipts

| arm | rows_sha_v3 | n | runtime_errors | elapsed_s | source_changed_during_run |
|---|---|---|---|---|---|
| parent | `f33f7af5f236f1fc…` | 48 | 0 | 592.5 | [] |
| oracle | `1507c3be68a0da29…` | 48 | 0 | 511.2 | [] |
| oracle-farm | `e47d5a5850ae1c46…` | 48 | 0 | 686.7 | [] |
| of-potion | `17d3ea46f63f88a6…` | 48 | 0 | 640.4 | [] |
| of-potion-contact | `525d18ef98ddbe8d…` | 48 | 0 | 648.5 | [] |
| oracle-pickup | `3f02ce54b8922626…` | 48 | 0 | 681.7 | [] |

`runtime_errors` is 0 for all six arms; `source_changed_during_run` is empty for all six (sha256 of 10 source
files + the worker zip before and after each run).

---

## 5. Full table

| stat | parent | oracle | oracle-farm | of-potion | of-potion-contact | oracle-pickup |
|---|---|---|---|---|---|---|
| n | 48 | 48 | 48 | 48 | 48 | 48 |
| alive | 28 | 12 | 28 | 20 | 20 | 9 |
| deaths | 20 | 36 | 20 | 28 | 28 | 39 |
| l1_deaths | 8 | 3 | 3 | 8 | 8 | 4 |
| l2_deaths | 12 | 14 | 14 | 18 | 18 | 13 |
| l2_beats | 87358 | 58735 | 132166 | 120480 | 112884 | 50962 |
| l2_hazard_per_1k | 0.1374 | 0.2384 | 0.1059 | 0.1494 | 0.1595 | 0.2551 |
| l1_kills | 4723 | 4534 | 4969 | 4639 | 4647 | 4374 |
| l2_kills | 826 | 763 | 1331 | 1298 | 1263 | 771 |
| kills_total | 5550 | 5469 | 6353 | 5938 | 5917 | 5295 |
| l2_reach | 30 | 41 | 35 | 35 | 35 | 41 |
| l3 | 2 | 26 | 5 | 4 | 4 | 29 |
| clvl_ge_4 | 18 | 20 | 31 | 27 | 26 | 24 |
| clvl_mean | 3.229 | 3.438 | 3.667 | 3.521 | 3.5 | 3.396 |
| depth_mean | 1.667 | 2.646 | 1.833 | 1.833 | 1.833 | 2.812 |
| micro_steps_median | 12000.0 | 9302.0 | 13420.5 | 12433.0 | 12083.5 | 9268.5 |
| fd2_n | 30 | 41 | 35 | 35 | 35 | 41 |
| fd2_beat_median | 6798.0 | 6064 | 6300 | 6390 | 6390 | 6065 |
| fd2_belt_median | 6.0 | 5 | 5 | 5 | 5 | 5 |
| fd2_ac_mean | 17.067 | 16.293 | 17.257 | 16.943 | 17.0 | 15.78 |
| fd2_clvl_mean | 3.0 | 2.927 | 3.0 | 3.086 | 3.086 | 2.927 |
| alive_at_fd_plus_1800 | 34 | 27 | 42 | 36 | 35 | 30 |
| fd_plus_1800_observed_n | 37 | 40 | 44 | 40 | 40 | 41 |
| gold_final_median | 92.0 | 87.0 | 126.0 | 92.5 | 96.0 | 102.0 |
| gold_collected | 8625 | 8638 | 10961 | 9225 | 8939 | 8478 |
| gold_sold | 9066 | 7274 | 10889 | 9672 | 9685 | 8123 |
| gold_spent | 23048 | 19738 | 26307 | 24303 | 23661 | 19862 |
| purchases | 276 | 241 | 302 | 269 | 264 | 251 |
| sales | 218 | 231 | 296 | 258 | 253 | 253 |
| loot_collected | 123 | 110 | 148 | 118 | 117 | 119 |
| identified | 21 | 12 | 19 | 17 | 17 | 9 |
| weapon_upgrades | 12 | 4 | 6 | 7 | 6 | 3 |
| sweep_windows | 752 | 786 | 886 | 875 | 867 | 793 |
| chests_opened | 339 | 363 | 390 | 375 | 376 | 364 |
| barrels_smashed | 376 | 417 | 462 | 449 | 445 | 417 |
| sweep_gold | 0 | 0 | 0 | 0 | 0 | 0 |
| grab_windows | 571 | 547 | 614 | 619 | 619 | 551 |
| grab_piles | 729 | 727 | 809 | 796 | 786 | 729 |
| grab_gold | 8695 | 8701 | 10155 | 9892 | 9732 | 8881 |
| retreats | 58 | 87 | 80 | 70 | 68 | 94 |
| belt0_at_death | 11/20 | 21/36 | 11/20 | 16/28 | 16/28 | 22/39 |
| windows_total | 3815 | 4396 | 4934 | 4539 | 4438 | 10560 |
| windows_farm | 1341 | 1688 | 1921 | 1730 | 1687 | 1467 |
| windows_dive | 763 | 973 | 1012 | 878 | 838 | 1332 |
| windows_resupply | 1711 | 1735 | 2001 | 1931 | 1913 | 7761 |
| windows_forced_dive | 497 | 718 | 552 | 483 | 473 | 665 |
| windows_mask_forced | 2208 | 2453 | 2553 | 2414 | 2386 | 8426 |
| deaths_by_floor | {"1": 8, "2": 12} | {"1": 3, "2": 14, "3": 11, "4": 7, "5": 1} | {"1": 3, "2": 14, "3": 3} | {"1": 8, "2": 18, "3": 1, "4": 1} | {"1": 8, "2": 18, "3": 1, "4": 1} | {"1": 4, "2": 13, "3": 12, "4": 8, "5": 1, "6": 1} |

### 5.1 Exposure and hazard per floor

| arm | L1 beats/deaths/hazard per 1k | L2 beats/deaths/hazard per 1k | L3 beats/deaths/hazard per 1k | L4 beats/deaths/hazard per 1k | L5 beats/deaths/hazard per 1k |
|---|---|---|---|---|---|
| parent | 374770 / 8 / 0.0213 | 87358 / 12 / 0.1374 | 935 / 0 / 0.0 | 0 / 0 / 0.0 | 0 / 0 / 0.0 |
| oracle | 300406 / 3 / 0.01 | 58735 / 14 / 0.2384 | 9809 / 11 / 1.1214 | 5372 / 7 / 1.3031 | 1008 / 1 / 0.9921 |
| oracle-farm | 414998 / 3 / 0.0072 | 132166 / 14 / 0.1059 | 4742 / 3 / 0.6326 | 0 / 0 / 0.0 | 0 / 0 / 0.0 |
| of-potion | 373201 / 8 / 0.0214 | 120480 / 18 / 0.1494 | 2185 / 1 / 0.4577 | 113 / 1 / 8.8496 | 0 / 0 / 0.0 |
| of-potion-contact | 366841 / 8 / 0.0218 | 112884 / 18 / 0.1595 | 2472 / 1 / 0.4045 | 113 / 1 / 8.8496 | 0 / 0 / 0.0 |
| oracle-pickup | 300145 / 4 / 0.0133 | 50962 / 13 / 0.2551 | 8632 / 12 / 1.3902 | 14758 / 8 / 0.5421 | 1029 / 1 / 0.9718 |

**This is the key table of this round.** `ORACLE-FARM` **brings the L1 hazard down to 0.0072 per 1,000 ticks**
(student 0.0213, oracle 0.0100) **while** not sending itself to L3/L4 as the oracle does (oracle L3 hazard 1.12,
L4 1.30; the composite arm has only 3 deaths on L3 and zero exposure on L4). `OF-POTION` pushes the L1 hazard back
to 0.0214, **exactly the student's**: the drink clause cancels the base's L1 discipline entirely.

### 5.2 Depth-reached ladder

| arm | >=L2 | >=L3 | >=L4 | >=L5 |
|---|---|---|---|---|
| parent | 30 | 2 | 0 | 0 |
| oracle | 41 | 26 | 11 | 1 |
| oracle-farm | 35 | 5 | 0 | 0 |
| of-potion | 35 | 4 | 1 | 0 |
| of-potion-contact | 35 | 4 | 1 | 0 |
| oracle-pickup | 41 | 29 | 14 | 2 |

The oracle and `ORACLE-PICKUP` reach L3 in 26 / 29 games, the composite arm in only 5; **this is exactly why the
composite arm survives**: it does not use DIVE windows to charge.

---

## 6. Paired tests (all 36, not a selection)

**Multiple comparisons first (review-round finding 5)**: below are **36 one-sided 95% tests** with **no
multiplicity correction**. Under the null hypothesis, **on average 36 x 0.05 = 1.8** of them cross the 0 line by
luck alone; in this round **7** actually cross. But these 7 are **not 7 independent things**; they are **3 facts**
counted 7 times:

| Underlying fact | Comparisons that cross the line | Count |
|---|---|---|
| the frozen oracle base halves the L1 hazard (**the same three seeds** 2133017/2133021/2133026) | `oracle_vs_parent · died_on_L1`, `oracle-farm_vs_parent · died_on_L1` | 2 |
| the oracle base on both window types **charges down** (`oracle` and `oracle-pickup` use the same base on both window types) | `oracle_vs_parent · not_reached_L2`, `· not_reached_L3`, `oracle-pickup_vs_parent · not_reached_L2`, `· not_reached_L3` | 4 |
| once the DIVE windows go back to the student, the oracle no longer charges down and dies | `oracle-farm_vs_oracle · died` | 1 |

**The one comparison that carries this round's VERDICT is only 0.004 below the 0 line** (-0.00437) and belongs to
the first row above: **it is the same statistic as round one's**. Add section 14.2 (pool 2_133 was consumed four
more times in this round), and **this "95%" label is weaker than it reads**. The real remedy is replication on a
fresh pool (2_116-119 / 2_126-128 untouched in this round, section 12), already written into section 15.2.

| pair (event = the BAD thing) | saved | lost | net | UCB95 one-sided |
|---|---|---|---|---|
| oracle vs parent · died | 2 | 18 | -16 | +0.46458 |
| oracle vs parent · died_on_L1 | 7 | 2 | +5 | -0.00437 **<0** |
| oracle vs parent · not_reached_L2 | 14 | 3 | +11 | -0.09876 **<0** |
| oracle vs parent · not_reached_L3 | 25 | 1 | +24 | -0.37177 **<0** |
| oracle-farm vs parent · died | 9 | 9 | +0 | +0.14540 |
| oracle-farm vs parent · died_on_L1 | 7 | 2 | +5 | -0.00437 **<0** |
| oracle-farm vs parent · not_reached_L2 | 9 | 4 | +5 | +0.01690 |
| oracle-farm vs parent · not_reached_L3 | 4 | 1 | +3 | +0.01268 |
| of-potion vs parent · died | 4 | 12 | -8 | +0.29791 |
| of-potion vs parent · died_on_L1 | 6 | 6 | +0 | +0.11872 |
| of-potion vs parent · not_reached_L2 | 10 | 5 | +5 | +0.02624 |
| of-potion vs parent · not_reached_L3 | 3 | 1 | +2 | +0.02616 |
| of-potion-contact vs parent · died | 3 | 11 | -8 | +0.28864 |
| of-potion-contact vs parent · died_on_L1 | 6 | 6 | +0 | +0.11872 |
| of-potion-contact vs parent · not_reached_L2 | 10 | 5 | +5 | +0.02624 |
| of-potion-contact vs parent · not_reached_L3 | 3 | 1 | +2 | +0.02616 |
| oracle-pickup vs parent · died | 2 | 21 | -19 | +0.53067 |
| oracle-pickup vs parent · died_on_L1 | 8 | 4 | +4 | +0.03372 |
| oracle-pickup vs parent · not_reached_L2 | 14 | 3 | +11 | -0.09876 **<0** |
| oracle-pickup vs parent · not_reached_L3 | 27 | 0 | +27 | -0.44471 **<0** |
| oracle-farm vs oracle · died | 18 | 2 | +16 | -0.20209 **<0** |
| oracle-farm vs oracle · died_on_L1 | 0 | 0 | +0 | +0.00000 |
| oracle-farm vs oracle · not_reached_L2 | 1 | 7 | -6 | +0.21728 |
| oracle-farm vs oracle · not_reached_L3 | 0 | 21 | -21 | +0.55529 |
| of-potion vs oracle-farm · died | 6 | 14 | -8 | +0.31473 |
| of-potion vs oracle-farm · died_on_L1 | 2 | 7 | -5 | +0.20396 |
| of-potion vs oracle-farm · not_reached_L2 | 4 | 4 | +0 | +0.09693 |
| of-potion vs oracle-farm · not_reached_L3 | 2 | 3 | -1 | +0.09731 |
| of-potion-contact vs of-potion · died | 1 | 1 | +0 | +0.04847 |
| of-potion-contact vs of-potion · died_on_L1 | 0 | 0 | +0 | +0.00000 |
| of-potion-contact vs of-potion · not_reached_L2 | 0 | 0 | +0 | +0.00000 |
| of-potion-contact vs of-potion · not_reached_L3 | 1 | 1 | +0 | +0.04847 |
| oracle-pickup vs oracle · died | 3 | 6 | -3 | +0.16424 |
| oracle-pickup vs oracle · died_on_L1 | 1 | 2 | -1 | +0.07999 |
| oracle-pickup vs oracle · not_reached_L2 | 2 | 2 | +0 | +0.06854 |
| oracle-pickup vs oracle · not_reached_L3 | 7 | 4 | +3 | +0.05019 |

  => 7 of 36 comparisons have a one-sided 95% UCB below zero:
     - oracle vs parent | died_on_L1
     - oracle vs parent | not_reached_L2
     - oracle vs parent | not_reached_L3
     - oracle-farm vs parent | died_on_L1
     - oracle-pickup vs parent | not_reached_L2
     - oracle-pickup vs parent | not_reached_L3
     - oracle-farm vs oracle | died

### 6.1 Single-seed sensitivity (computed for every L1 comparison that crosses the line)

  oracle vs parent · died_on_L1: measured -0.00437; one saved->lost +0.03924; one saved->concordant +0.01156
  oracle-farm vs parent · died_on_L1: measured -0.00437; one saved->lost +0.03924; one saved->concordant +0.01156

`ORACLE-FARM`'s and `ORACLE`'s L1 wins **are the same statistic**: the two arms' L1 death seed sets are exactly the
same (2133017 / 2133021 / 2133026, section 10.2). So this 95% result is exactly as fragile as in round one:
**flip one seed and it is gone**.

---

## 7. Routing telemetry: which callable served which windows

| arm | slot | role | farm beats | dive beats | resupply beats | forced-dive beats | unforced-dive beats | a12 | a13 |
|---|---|---|---|---|---|---|---|---|---|
| parent | farm_slot | parent | 34968 | 41260 | 0 | 31801 | 9459 | 55 | 584 |
| oracle | farm_slot | oracle | 32906 | 18475 | 0 | 16007 | 2468 | 0 | 909 |
| oracle-farm | farm_slot | oracle | 38794 | 0 | 0 | 0 | 0 | 0 | 492 |
| oracle-farm | dive_slot | parent | 0 | 51624 | 0 | 33901 | 17723 | 32 | 137 |
| of-potion | farm_slot | oracle | 33366 | 0 | 0 | 0 | 0 | 117 | 462 |
| of-potion | dive_slot | parent | 0 | 46515 | 0 | 31831 | 14684 | 29 | 180 |
| of-potion-contact | farm_slot | oracle | 31983 | 0 | 0 | 0 | 0 | 113 | 473 |
| of-potion-contact | dive_slot | parent | 0 | 43980 | 0 | 30983 | 12997 | 24 | 170 |
| oracle-pickup | farm_slot | oracle | 27176 | 27097 | 0 | 15095 | 12002 | 0 | 698 |

This table settles three things:

1. **The composite arms' routing is real**: in `oracle-farm` / `of-potion` / `of-potion-contact`, the oracle
   slot's `dive beats` is always **0** and the student slot's `farm beats` is always **0**.
2. **RESUPPLY windows never go through a worker**: `resupply beats` is **0** in all six arms.
3. **Ticks in forced dives (forced_dive) can be attributed**: for example, of the composite arm's 51624 DIVE ticks,
   33901 fall in `mask_forced` windows, all served by the student.

---

## 8. Clause-by-clause ablation: which helps, which hurts

| telemetry | oracle | oracle-farm | of-potion | of-potion-contact | oracle-pickup |
|---|---|---|---|---|---|
| policy_episodes | 48 | 48 | 48 | 48 | 48 |
| worker_beats_served_by_teacher | 51381 | 38794 | 33366 | 31983 | 54273 |
| grace_decisions | 0 | 0 | 0 | 0 | 0 |
| base_fallbacks | 0 | 0 | 0 | 0 | 0 |
| potion_wanted | 0 | 0 | 249 | 229 | 0 |
| instinct_potion | 0 | 0 | 117 | 113 | 0 |
| potion_blocked_by_mask | 0 | 0 | 132 | 116 | 0 |
| pickup_wanted | 0 | 0 | 0 | 0 | 1431 |
| instinct_pickup | 0 | 0 | 0 | 0 | 695 |
| pickup_blocked_by_contact | 0 | 0 | 0 | 0 | 736 |
| contact_trigger_beats | 0 | 0 | 0 | 193 | 0 |
| instinct_disengage | 0 | 0 | 0 | 132 | 0 |
| disengage_beats | 0 | 0 | 0 | 132 | 0 |
| disengage_moved | 0 | 0 | 0 | 41 | 0 |
| disengage_not_moved | 0 | 0 | 0 | 87 | 0 |
| disengage_achieved_gain | 0 | 0 | 0 | 31 | 0 |
| disengage_settled | 0 | 0 | 0 | 128 | 0 |
| disengage_settled_same_window | 0 | 0 | 0 | 127 | 0 |
| disengage_settled_cross_window | 0 | 0 | 0 | 1 | 0 |
| disengage_no_improving_move | 0 | 0 | 0 | 61 | 0 |
| disengage_no_legal_move | 0 | 0 | 0 | 0 | 0 |
| disengage_terrain_blocked_dirs | 0 | 0 | 0 | 269 | 0 |
| disengage_terrain_unavailable | 0 | 0 | 0 | 0 | 0 |
| disengage_budget_blocked | 0 | 0 | 0 | 0 | 0 |
| episodes_budget_exhausted | 0 | 0 | 0 | 0 | 0 |
  oracle: disengage beats per episode {"n": 48, "total": 0, "min": 0, "max": 0, "median": 0.0, "mean": 0.0, "episodes_over_100": 0, "max_fraction_of_budget": 0.0, "episodes_shorter_than_budget": 3}
  oracle-farm: disengage beats per episode {"n": 48, "total": 0, "min": 0, "max": 0, "median": 0.0, "mean": 0.0, "episodes_over_100": 0, "max_fraction_of_budget": 0.0, "episodes_shorter_than_budget": 3}
  of-potion: potion refusal rate 132/249 = 0.5301
  of-potion: disengage beats per episode {"n": 48, "total": 0, "min": 0, "max": 0, "median": 0.0, "mean": 0.0, "episodes_over_100": 0, "max_fraction_of_budget": 0.0, "episodes_shorter_than_budget": 8}
  of-potion-contact: potion refusal rate 116/229 = 0.5066
  of-potion-contact: disengage NOT-moved share 87/128 = 0.6797; achieved gain 31 tiles over 132 beats = 0.2348 tiles/beat
  of-potion-contact: disengage beats per episode {"n": 48, "total": 132, "min": 0, "max": 37, "median": 0.0, "mean": 2.8, "episodes_over_100": 0, "max_fraction_of_budget": 0.092, "episodes_shorter_than_budget": 8}
  oracle-pickup: disengage beats per episode {"n": 48, "total": 0, "min": 0, "max": 0, "median": 0.0, "mean": 0.0, "episodes_over_100": 0, "max_fraction_of_budget": 0.0, "episodes_shorter_than_budget": 5}

**Drink clause (harmful)**: 249 wanted drinks, **132 refused by the `drink_sovereignty` mask (53.0%)**, 117
actually drunk. Round one's refusal rate was 71.3%; lower this round, but still more than half. The cost: L1
deaths 3 -> 8, survival at game end 28 -> 20, kills 6353 -> 5938. The named seeds in section 10 show it most
directly: `2133027` dies in the composite arm on L2 at tick 13774 (159 kills), but with the drink clause **dies on
L1 at tick 883, clvl 1, 20 kills, belt 0**; `2133029` survives in the composite arm (189 kills) but with the drink
clause **dies on L1 at tick 2821, belt 0**; `2133038` likewise (alive -> L1 at tick 1459). **It burns the belt
early and then dies empty-handed early on L1.** (Round one's gap 8 in section 10 named `2133027` / `2133029` as
the two games that "died on L1 with zero disengage ticks"; **this round's answer is the drink clause**: in
`of-potion` these two games also have 0 disengage ticks and still die on L1.)

**Contact disengage clause (nearly ineffective)**: 193 trigger ticks, 132 firing ticks, **87/128 = 68.0% did not
move**, measured separation gain 31 tiles / 132 ticks = **0.235 tiles/tick**. Round one had 0.180 tiles/tick and
63.5% not moving; **removing the crowd condition only reduced the volume from 859 ticks to 132, without curing the
defect**. The 400-tick fence was **never reached** in this round (at most 37 ticks in one game = 9.2% of the
fence), and 8 games had fewer than 400 worker ticks in total, so **the fence is again untested in this round**.
Effect on results: the L1 death seed set is **exactly the same** as `of-potion`, and survival at game end is one
saved, one lost.

**Pickup clause (harmless on L1, but blows up the window count)**: see section 9.

---

## 9. RESUPPLY window explosion: attribution settled

| arm | windows total | farm | dive | resupply | forced_dive | a13 total (both slots) | a12 total | env beats (all windows) | farm+dive beats | worker calls | worker_no_effect_requests (farm+dive) | no-effect share (farm+dive beats) | no-effect share (per worker call) | no-effect share (OLD, all windows -- NOT comparable) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| parent | 3815 | 1341 | 763 | 1711 | 497 | 584 | 55 | 142594 | 76347 | 76228 | 14326 | 0.1876 | 0.1879 | 0.1005 |
| oracle | 4396 | 1688 | 973 | 1735 | 718 | 909 | 0 | 114213 | 51597 | 51381 | 22828 | 0.4424 | 0.4443 | 0.1999 |
| oracle-farm | 4934 | 1921 | 1012 | 2001 | 552 | 629 | 32 | 171010 | 90582 | 90418 | 27469 | 0.3033 | 0.3038 | 0.1606 |
| of-potion | 4539 | 1730 | 878 | 1931 | 483 | 642 | 146 | 150699 | 79934 | 79881 | 21904 | 0.274 | 0.2742 | 0.1453 |
| of-potion-contact | 4438 | 1687 | 838 | 1913 | 473 | 643 | 137 | 145037 | 76014 | 75963 | 20234 | 0.2662 | 0.2664 | 0.1395 |
| oracle-pickup | 10560 | 1467 | 1332 | 7761 | 665 | 698 | 0 | 129984 | 54483 | 54273 | 27358 | 0.5021 | 0.5041 | 0.2105 |

  denominator (RR2 finding 1): worker_no_effect_share = farm+dive worker_no_effect_requests / farm+dive env beats  [RR2 finding 1]
  worker calls = farm+dive (beats + fuse_trips - drain_attempts - recovery_actions), and it equals the two WorkerProbes' own measured count on every arm (env_audit.worker_calls_ledger_closes); the identity is pinned window by window on a real episode by test_teacher_v2.routing_real.

  per-mode no-effect audit (env's own ledger):
  parent: {"dive": {"beats": 41278, "drain_attempts": 18, "drains": 18, "executed_requests": 28941, "fuse_trips": 213, "no_effect_requests": 10708, "overrides": 213, "recovery_actions": 213, "windows": 763, "worker_no_effect_requests": 10520}, "farm": {"beats": 35069, "drain_attempts": 101, "drains": 101, "executed_requests": 30648, "fuse_trips": 93, "no_effect_requests": 3899, "overrides": 93, "recovery_actions": 93, "windows": 1341, "worker_no_effect_requests": 3806}, "resupply": {"beats": 66247, "drain_attempts": 0, "drains": 0, "executed_requests": 0, "fuse_trips": 0, "no_effect_requests": 0, "overrides": 0, "recovery_actions": 0, "windows": 1711, "worker_no_effect_requests": 0}}
  oracle: {"dive": {"beats": 18514, "drain_attempts": 39, "drains": 39, "executed_requests": 4298, "fuse_trips": 567, "no_effect_requests": 14216, "overrides": 567, "recovery_actions": 567, "windows": 973, "worker_no_effect_requests": 13649}, "farm": {"beats": 33083, "drain_attempts": 177, "drains": 177, "executed_requests": 23526, "fuse_trips": 378, "no_effect_requests": 9557, "overrides": 378, "recovery_actions": 378, "windows": 1688, "worker_no_effect_requests": 9179}, "resupply": {"beats": 62616, "drain_attempts": 0, "drains": 0, "executed_requests": 2, "fuse_trips": 0, "no_effect_requests": 0, "overrides": 0, "recovery_actions": 0, "windows": 1735, "worker_no_effect_requests": 0}}
  oracle-farm: {"dive": {"beats": 51676, "drain_attempts": 52, "drains": 52, "executed_requests": 37939, "fuse_trips": 200, "no_effect_requests": 11748, "overrides": 200, "recovery_actions": 200, "windows": 1012, "worker_no_effect_requests": 11586}, "farm": {"beats": 38906, "drain_attempts": 112, "drains": 112, "executed_requests": 22367, "fuse_trips": 656, "no_effect_requests": 16539, "overrides": 656, "recovery_actions": 656, "windows": 1921, "worker_no_effect_requests": 15883}, "resupply": {"beats": 80428, "drain_attempts": 0, "drains": 0, "executed_requests": 4, "fuse_trips": 0, "no_effect_requests": 0, "overrides": 0, "recovery_actions": 0, "windows": 2001, "worker_no_effect_requests": 0}}
  of-potion: {"dive": {"beats": 46540, "drain_attempts": 25, "drains": 25, "executed_requests": 33600, "fuse_trips": 197, "no_effect_requests": 11267, "overrides": 197, "recovery_actions": 197, "windows": 878, "worker_no_effect_requests": 11105}, "farm": {"beats": 33394, "drain_attempts": 28, "drains": 28, "executed_requests": 22150, "fuse_trips": 445, "no_effect_requests": 11244, "overrides": 445, "recovery_actions": 445, "windows": 1730, "worker_no_effect_requests": 10799}, "resupply": {"beats": 70765, "drain_attempts": 0, "drains": 0, "executed_requests": 28, "fuse_trips": 0, "no_effect_requests": 0, "overrides": 0, "recovery_actions": 0, "windows": 1931, "worker_no_effect_requests": 0}}
  of-potion-contact: {"dive": {"beats": 44001, "drain_attempts": 21, "drains": 21, "executed_requests": 31236, "fuse_trips": 196, "no_effect_requests": 11169, "overrides": 196, "recovery_actions": 196, "windows": 838, "worker_no_effect_requests": 11007}, "farm": {"beats": 32013, "drain_attempts": 30, "drains": 30, "executed_requests": 22409, "fuse_trips": 378, "no_effect_requests": 9604, "overrides": 378, "recovery_actions": 378, "windows": 1687, "worker_no_effect_requests": 9227}, "resupply": {"beats": 69023, "drain_attempts": 0, "drains": 0, "executed_requests": 28, "fuse_trips": 0, "no_effect_requests": 0, "overrides": 0, "recovery_actions": 0, "windows": 1913, "worker_no_effect_requests": 0}}
  oracle-pickup: {"dive": {"beats": 27167, "drain_attempts": 71, "drains": 71, "executed_requests": 4629, "fuse_trips": 899, "no_effect_requests": 22538, "overrides": 899, "recovery_actions": 898, "windows": 1332, "worker_no_effect_requests": 21641}, "farm": {"beats": 27316, "drain_attempts": 140, "drains": 140, "executed_requests": 21369, "fuse_trips": 230, "no_effect_requests": 5947, "overrides": 230, "recovery_actions": 230, "windows": 1467, "worker_no_effect_requests": 5717}, "resupply": {"beats": 75501, "drain_attempts": 0, "drains": 0, "executed_requests": 1, "fuse_trips": 427, "no_effect_requests": 10693, "overrides": 427, "recovery_actions": 428, "windows": 7761, "worker_no_effect_requests": 0}}

  round1 parent: windows total 3815, resupply 1711, a13 n/a, a12 n/a, alive 28, l1_deaths 8
  round1 oracle: windows total 4396, resupply 1735, a13 909, a12 n/a, alive 12, l1_deaths 3
  round1 teacher-v1: windows total 10244, resupply 7605, a13 2088, a12 159, alive 7, l1_deaths 10

**Conclusion: `ORACLE-PICKUP` alone reproduces round one's entire window explosion.** The oracle base on both
window types + **the pickup clause only** -> RESUPPLY windows 1735 -> **7761**, total windows 4396 -> **10560**,
`mask_forced` 2453 -> **8426**; round one, with all three instinct clauses, had 7605 / 10244 / 8293. **Almost an
exact reproduction.**

**Mechanism** (engine code + this round's audit, and the two agree; line numbers as of that work tree):

- `options_env.py:1082`: `m[RESUPPLY] = bool(controller_mask[13])`: **a RESUPPLY window is legal if and only if
  a13 is legal** (a visible, reachable potion within radius 12).
- `options_env.py:411-422`, `_farm_handoff`: **a story target exists and no fightable monster within radius 6 =>
  `m[FARM] = False`**.
- `resource_option_choice` (`:1422`), when not descending, takes the first legal item in the order
  `FARM -> RESUPPLY -> DIVE`.
- The pickup clause is a **multi-tick walking macro**: it takes the character away from the monsters to pick up a
  potion from the floor. Once away, `_farm_handoff` holds -> FARM is masked -> the potion is still on the floor
  (a13 still legal) -> **the manager has only RESUPPLY left** -> the scripted service runs a very short window ->
  the window closes -> repeat.
- The numbers agree: `ORACLE-PICKUP`'s RESUPPLY is 75501 ticks / 7761 windows = **9.7 ticks per window**, the
  oracle's 62616 / 1735 = **36.1 ticks per window**: **many more windows, each shorter**, exactly the shape of
  this loop. And only this arm's RESUPPLY windows show `fuse_trips 427` and `no_effect_requests 10693` (both are 0
  in the RESUPPLY windows of the other five arms): **the scripted service is spinning idle**.

**A counter-intuitive fact that must be stated**: the explosion is **not** caused by more a13 presses.
`ORACLE-PICKUP`'s a13 total is **698**, **fewer** than the **909** of `ORACLE` without the pickup clause (the pickup
clause fires 695 times and is blocked by contact 736 times; the frozen base itself asks for a13 only 3 more times).
**So round one's chain of reasoning, "a13 rose from 857 to 2088, so it is the pickup clause", was wrong, and its
conclusion happened to be right**: the real cause is that **this macro leads the character away from the
monsters**, not how many times it presses.

**The oracle idling in DIVE windows** (same audit table, another and more important reading):

| Arm | DIVE window ticks | executed | empty requests | fuse | execution rate |
|---|---|---|---|---|---|
| parent | 41278 | 28941 | 10708 | 213 | **70.1%** |
| **oracle** | 18514 | **4298** | 14216 | 567 | **23.2%** |
| **oracle-farm** (DIVE goes to the student) | 51676 | 37939 | 11748 | 200 | **73.4%** |
| oracle-pickup | 27167 | 4629 | 22538 | 899 | **17.0%** |

**77% of the oracle's DIVE-window ticks are requests with no effect.** This is the quantitative shape of "the
oracle can fight but cannot go downstairs", and the mechanistic explanation of why the composite arm works. (This
small table always used the **per-window-type** denominator, so it is correct; what the review round flagged was
the **total** column of the large table above, see below.)

**Denominator correction (review-round finding 1)**: the large table in section 9 originally had a single
"no-effect share" column whose denominator was the env's ticks over **all three window types**, **including
RESUPPLY, and RESUPPLY windows never call a worker** (`options_env.py:2815` has no entry for option 2; in the
per-window audit the resupply `worker_no_effect_requests` is 0 in all six arms). The RESUPPLY share of all ticks
differs a lot between arms (parent 46%, oracle-pickup 58%), **so that column is not comparable exactly between the
arms it was printed to compare**. The table now prints four denominators; the old column is kept but renamed and
marked as not comparable:

- **env beats (all windows)**: the old criterion, archive only;
- **farm+dive beats**: all ticks in worker-owned windows (including scripted ticks);
- **worker calls**: how many times the worker callable **was actually asked**, counted directly by the two
  `WorkerProbe`s and recomputed window by window from the env's own ledger
  `beats + fuse_trips - drain_attempts - recovery_actions`
  (`teacher_v2.observing_env_class()._AUDIT_IDENTITY`, pinned window by window on a whole real game by
  `test_teacher_v2.routing_real`);
- the two corresponding share columns.

**With the honest denominator, the ranking and ratios change**: parent 0.1876, oracle **0.4424**, oracle-farm
0.3033, of-potion 0.2740, of-potion-contact 0.2662, oracle-pickup **0.5021**. Under the old criterion oracle-farm
0.1606 vs oracle-pickup 0.2105 looked like 1.3 times; **the real worker-side gap is 1.7 times**. (The conclusion
does not change: the mechanism argument in section 0 always used the per-window-type denominator 18514 / 4298.)

---

## 10. Named seeds

| seed | parent | oracle | oracle-farm | of-potion | of-potion-contact | oracle-pickup |
|---|---|---|---|---|---|---|
| 2133002 | alive, depth 1, kills 132, clvl 3 | alive, depth 2, kills 126, clvl 3 | alive, depth 1, kills 136, clvl 4 | alive, depth 1, kills 134, clvl 3 | alive, depth 1, kills 134, clvl 3 | DIED L3@10293 belt0, depth 4, kills 154, clvl 4 |
| 2133021 | DIED L2@7387 belt4, depth 2, kills 103, clvl 3 | DIED L1@2563 belt0, depth 1, kills 68, clvl 2 | DIED L1@2563 belt0, depth 1, kills 68, clvl 2 | DIED L1@608 belt0, depth 1, kills 19, clvl 1 | DIED L1@645 belt0, depth 1, kills 19, clvl 1 | DIED L1@2563 belt0, depth 1, kills 68, clvl 2 |
| 2133022 | alive, depth 2, kills 142, clvl 4 | alive, depth 1, kills 92, clvl 3 | alive, depth 2, kills 136, clvl 4 | DIED L4@14278 belt8, depth 4, kills 134, clvl 4 | DIED L4@14278 belt8, depth 4, kills 134, clvl 4 | DIED L6@10063 belt0, depth 6, kills 116, clvl 3 |
| 2133038 | alive, depth 2, kills 162, clvl 4 | alive, depth 3, kills 178, clvl 5 | alive, depth 1, kills 127, clvl 3 | DIED L1@1459 belt0, depth 1, kills 44, clvl 2 | DIED L1@1496 belt0, depth 1, kills 47, clvl 2 | alive, depth 3, kills 178, clvl 5 |
| 2133006 | alive, depth 2, kills 148, clvl 4 | DIED L2@7942 belt0, depth 2, kills 107, clvl 3 | DIED L2@10532 belt7, depth 2, kills 111, clvl 3 | DIED L2@12120 belt3, depth 2, kills 146, clvl 4 | DIED L2@12120 belt3, depth 2, kills 146, clvl 4 | DIED L3@9995 belt8, depth 3, kills 142, clvl 4 |
| 2133039 | DIED L1@2047 belt0, depth 1, kills 69, clvl 2 | DIED L2@6698 belt4, depth 2, kills 81, clvl 3 | alive, depth 1, kills 72, clvl 3 | DIED L1@3941 belt0, depth 1, kills 48, clvl 2 | DIED L1@3941 belt0, depth 1, kills 48, clvl 2 | DIED L2@7732 belt5, depth 2, kills 45, clvl 2 |
| 2133014 | alive, depth 2, kills 168, clvl 5 | DIED L2@9799 belt0, depth 3, kills 116, clvl 3 | DIED L3@13890 belt0, depth 3, kills 187, clvl 6 | DIED L2@14672 belt0, depth 3, kills 174, clvl 5 | alive, depth 2, kills 184, clvl 5 | DIED L2@10113 belt0, depth 3, kills 116, clvl 3 |
| 2133010 | alive, depth 2, kills 171, clvl 4 | DIED L3@9285 belt0, depth 4, kills 135, clvl 4 | DIED L2@13190 belt8, depth 2, kills 146, clvl 4 | alive, depth 2, kills 169, clvl 4 | alive, depth 2, kills 169, clvl 4 | DIED L3@9196 belt5, depth 4, kills 136, clvl 4 |
| 2133027 | DIED L2@12214 belt0, depth 2, kills 125, clvl 3 | DIED L4@7254 belt4, depth 4, kills 119, clvl 3 | DIED L2@13774 belt3, depth 2, kills 159, clvl 4 | DIED L1@883 belt0, depth 1, kills 20, clvl 1 | DIED L1@883 belt0, depth 1, kills 20, clvl 1 | DIED L4@7341 belt4, depth 4, kills 118, clvl 3 |
| 2133029 | DIED L2@9762 belt8, depth 2, kills 121, clvl 3 | DIED L2@12293 belt0, depth 2, kills 141, clvl 4 | alive, depth 2, kills 189, clvl 5 | DIED L1@2821 belt0, depth 1, kills 68, clvl 2 | DIED L1@2821 belt0, depth 1, kills 68, clvl 2 | DIED L2@12714 belt0, depth 2, kills 142, clvl 4 |

### 10.1 Clause activity on the named seeds

  2133002 oracle: potion 0(refused 0) pickup 0 disengage 0 moved 0/0
  2133002 oracle-farm: potion 0(refused 0) pickup 0 disengage 0 moved 0/0
  2133002 of-potion: potion 1(refused 0) pickup 0 disengage 0 moved 0/0
  2133002 of-potion-contact: potion 1(refused 0) pickup 0 disengage 0 moved 0/0
  2133002 oracle-pickup: potion 0(refused 0) pickup 45 disengage 0 moved 0/0
  2133021 oracle: potion 0(refused 0) pickup 0 disengage 0 moved 0/0
  2133021 oracle-farm: potion 0(refused 0) pickup 0 disengage 0 moved 0/0
  2133021 of-potion: potion 1(refused 2) pickup 0 disengage 0 moved 0/0
  2133021 of-potion-contact: potion 1(refused 2) pickup 0 disengage 2 moved 1/2
  2133021 oracle-pickup: potion 0(refused 0) pickup 0 disengage 0 moved 0/0
  2133022 oracle: potion 0(refused 0) pickup 0 disengage 0 moved 0/0
  2133022 oracle-farm: potion 0(refused 0) pickup 0 disengage 0 moved 0/0
  2133022 of-potion: potion 3(refused 14) pickup 0 disengage 0 moved 0/0
  2133022 of-potion-contact: potion 3(refused 14) pickup 0 disengage 0 moved 0/0
  2133022 oracle-pickup: potion 0(refused 0) pickup 15 disengage 0 moved 0/0
  2133038 oracle: potion 0(refused 0) pickup 0 disengage 0 moved 0/0
  2133038 oracle-farm: potion 0(refused 0) pickup 0 disengage 0 moved 0/0
  2133038 of-potion: potion 2(refused 0) pickup 0 disengage 0 moved 0/0
  2133038 of-potion-contact: potion 2(refused 0) pickup 0 disengage 10 moved 5/10
  2133038 oracle-pickup: potion 0(refused 0) pickup 9 disengage 0 moved 0/0
  2133006 oracle: potion 0(refused 0) pickup 0 disengage 0 moved 0/0
  2133006 oracle-farm: potion 0(refused 0) pickup 0 disengage 0 moved 0/0
  2133006 of-potion: potion 3(refused 0) pickup 0 disengage 0 moved 0/0
  2133006 of-potion-contact: potion 3(refused 0) pickup 0 disengage 0 moved 0/0
  2133006 oracle-pickup: potion 0(refused 0) pickup 16 disengage 0 moved 0/0
  2133039 oracle: potion 0(refused 0) pickup 0 disengage 0 moved 0/0
  2133039 oracle-farm: potion 0(refused 0) pickup 0 disengage 0 moved 0/0
  2133039 of-potion: potion 1(refused 0) pickup 0 disengage 0 moved 0/0
  2133039 of-potion-contact: potion 1(refused 0) pickup 0 disengage 0 moved 0/0
  2133039 oracle-pickup: potion 0(refused 0) pickup 1 disengage 0 moved 0/0
  2133014 oracle: potion 0(refused 0) pickup 0 disengage 0 moved 0/0
  2133014 oracle-farm: potion 0(refused 0) pickup 0 disengage 0 moved 0/0
  2133014 of-potion: potion 3(refused 0) pickup 0 disengage 0 moved 0/0
  2133014 of-potion-contact: potion 2(refused 0) pickup 0 disengage 3 moved 2/3
  2133014 oracle-pickup: potion 0(refused 0) pickup 10 disengage 0 moved 0/0
  2133010 oracle: potion 0(refused 0) pickup 0 disengage 0 moved 0/0
  2133010 oracle-farm: potion 0(refused 0) pickup 0 disengage 0 moved 0/0
  2133010 of-potion: potion 2(refused 3) pickup 0 disengage 0 moved 0/0
  2133010 of-potion-contact: potion 2(refused 3) pickup 0 disengage 0 moved 0/0
  2133010 oracle-pickup: potion 0(refused 0) pickup 19 disengage 0 moved 0/0
  2133027 oracle: potion 0(refused 0) pickup 0 disengage 0 moved 0/0
  2133027 oracle-farm: potion 0(refused 0) pickup 0 disengage 0 moved 0/0
  2133027 of-potion: potion 1(refused 0) pickup 0 disengage 0 moved 0/0
  2133027 of-potion-contact: potion 1(refused 0) pickup 0 disengage 0 moved 0/0
  2133027 oracle-pickup: potion 0(refused 0) pickup 10 disengage 0 moved 0/0
  2133029 oracle: potion 0(refused 0) pickup 0 disengage 0 moved 0/0
  2133029 oracle-farm: potion 0(refused 0) pickup 0 disengage 0 moved 0/0
  2133029 of-potion: potion 3(refused 0) pickup 0 disengage 0 moved 0/0
  2133029 of-potion-contact: potion 3(refused 0) pickup 0 disengage 0 moved 0/0
  2133029 oracle-pickup: potion 0(refused 0) pickup 18 disengage 0 moved 0/0

### 10.2 L1 death seeds and the character level at death

  parent: 8 L1 deaths -> [(2133008, 2, 0), (2133009, 2, 0), (2133015, 3, 0), (2133017, 1, 0), (2133023, 3, 0), (2133024, 2, 0), (2133039, 2, 0), (2133044, 1, 0)]
  oracle: 3 L1 deaths -> [(2133017, 1, 0), (2133021, 2, 0), (2133026, 1, 0)]
  oracle-farm: 3 L1 deaths -> [(2133017, 1, 0), (2133021, 2, 0), (2133026, 1, 0)]
  of-potion: 8 L1 deaths -> [(2133009, 2, 0), (2133021, 1, 0), (2133027, 1, 0), (2133029, 2, 0), (2133033, 3, 0), (2133038, 2, 0), (2133039, 2, 0), (2133041, 1, 0)]
  of-potion-contact: 8 L1 deaths -> [(2133009, 2, 0), (2133021, 1, 0), (2133027, 1, 0), (2133029, 2, 0), (2133033, 3, 0), (2133038, 2, 0), (2133039, 2, 0), (2133041, 1, 0)]
  oracle-pickup: 4 L1 deaths -> [(2133011, 2, 0), (2133021, 2, 0), (2133026, 1, 0), (2133036, 2, 0)]

`ORACLE`'s and `ORACLE-FARM`'s three L1 death seeds are **exactly the same**; so are the eight of `OF-POTION` and
`OF-POTION-CONTACT`. Every L1 death (34 games across the six arms) has **belt = 0, without exception**.

---

## 11. Were kills and depth wrecked?

| arm | alive | kills_total | depth_mean | L3 reach | clvl_ge_4 | median micro_steps |
|---|---|---|---|---|---|---|
| parent | 28 | 5550 | 1.667 | 2 | 18 | 12000.0 |
| oracle | 12 | 5469 | 2.646 | 26 | 20 | 9302.0 |
| oracle-farm | 28 | 6353 | 1.833 | 5 | 31 | 13420.5 |
| of-potion | 20 | 5938 | 1.833 | 4 | 27 | 12433.0 |
| of-potion-contact | 20 | 5917 | 1.833 | 4 | 26 | 12083.5 |
| oracle-pickup | 9 | 5295 | 2.812 | 29 | 24 | 9268.5 |

`ORACLE-FARM`: alive 28 (= student), kills 6353 (> student's 5550), 31 games with clvl >= 4 (> student's 18),
median micro-steps 13420 (> student's 12000). **Nothing is wrecked, and most items are better.**
`OF-POTION` / `OF-POTION-CONTACT`: survival drops to 20, kills still above the student.
`ORACLE-PICKUP`: alive 9, kills 5295, **the same "charge down and die" shape as round one's oracle**.

---

## 12. Read-only proof and seed enumeration

  distinct seeds across every round-2 artefact: 48, min 2133000, max 2133047
  contiguous 2133000-2133047: True
  any seed in virgin pools 2_116-119 / 2_126-128: False
  seeds per arm: {'parent': 48, 'oracle': 48, 'oracle-farm': 48, 'of-potion': 48, 'of-potion-contact': 48, 'oracle-pickup': 48}

- The main-tree `tree_fp.sh` fingerprint (283 entries + the build link + `_diablogym*.so`) is **byte-identical**
  before and after this round: `diff` of `round2/main-tree-fingerprint-{mid,after}.txt` against
  `main-tree-fingerprint-before-round2.txt` is empty.
- In `m2-merge` and `oracle-tree`, **no file was modified** during this round (`find -newermt` is empty).
- The sha256 of `teacher_v1.py` / `test_teacher_v1.py` in `oracle2-tree` is **bit-identical** to `oracle-tree`
  (`65c1a14f…` / `f12d94d9…`).
- Seeds appearing in all outputs of this round: **exactly 48, contiguous 2133000-2133047**, with an **empty**
  intersection with the fresh pools 2_116-119 / 2_126-128.

---

## 13. Unit tests (`round2/tests-full-review.log`, 85 pass, 0 fail)

| Group | Coverage |
|---|---|
| routing (synthetic) | `workers` has only the FARM/DIVE entries, the two slots are different callables, FARM is the oracle and DIVE the student, RESUPPLY has no worker; a FakeEnv driver checks window by window "which callable served which window"; `on_beat` fans out and fires exactly once per tick; `episode_reseed` fans out to the student; the action-12 contract; `bind` rejects an env with inconsistent `drink_sovereignty`; the routing object refuses to be called as a worker |
| routing (real game) **tightened in the review round** | one whole real `OptionsEnv` game: the oracle slot has only farm ticks, the student slot only dive ticks, `resupply` ticks are 0, the observing env's window ledger agrees with the probe's, `forced_dive` can be joined by `window_id`; **and** (before review finding 2 these were three weak assertions): each slot's tick count **equals** the env's own worker-call ledger `beats + fuse_trips - drain_attempts - recovery_actions`; the set of window ids served by the student slot is **set-equal** to the env's dive windows, and the oracle slot's to the farm windows; every farm/dive window satisfies the identity **window by window**; windows counted by the probe + zero-call windows = the probe's window count; RESUPPLY windows have ticks but the env's own worker counters are all 0 |
| ranking equivalence (new in the review round) | `_disengage_move_contact` is a hand copy of the frozen `TeacherV1._disengage_move` without the stairs tiebreak: **on 1200 random states both give identical actions and gains** (when there is no up staircase within `STAIRS_TIEBREAK_RADIUS`), and with stairs present they **must differ** (otherwise the test is vacuous), so if anyone changes the frozen ranking in a later round, this test turns red |
| arm-name telemetry (new in the review round) | the three clause arms' `telemetry["variant"]` is the **arm name**, not the base name `oracle` (finding 7); `oracle` / `oracle-farm` still report `oracle` |
| clause truth tables | drink (belt / strictly below 0.60 / mask-refusal count), pickup (Chebyshev 1 boundary, radius 2 does not block, monsters in darkness do not block), contact disengagement (both 0.35 boundaries, **the crowd does not trigger**, **stairs do not change the choice**, terrain filter including hazard/door, clause order, 400-tick safety cap), gear grace still answered by the frozen layer 0, each arm runs only its own clauses |
| legality | 3 clause sets x 900 random states = **2700** states; emitted actions are **always inside the mask** |
| determinism | each of the four new arms run twice on the same seed: **bit-identical action sequences, identical `rows_sha_v3`** |

**A recording error that must be disclosed (already disclosed in the first pass, kept here)**: in the ledger
entry `R19_TEACHER_ROUND2_BUILT`, the `tests` field says "Full run (incl. two real-episode legs): ==== 54 passed
====", but 54 is the count of the `--no-episode` leg. The first pass's full unit-test run is **70 passing**
(`round2/tests-full.log`), and after the review round tightened and added tests it is **85 passing**
(`round2/tests-full-review.log`). The ledger is append-only, so the correction is in the `corrections` field of
`R19_TEACHER_ROUND2_RESULT`, and this round's count is in `R19_TEACHER_ROUND2_REVIEW_ROUND` as well as here.
**Both test counts in this report are read from the logs, not typed by hand** (`mk_report._tests`).

---

## 14. Methodological gaps (self-reported)

1. **The composite arm contains the student.** `ORACLE-FARM` is not a pure scripted teacher: its DIVE windows are
   the student itself. As a BC/DAgger teacher it is legitimate (it is a worker callable), but **it can teach the
   student only its FARM-window behaviour**; in DIVE windows it is bit-identical to the student and has nothing to
   teach. The accurate version of "the teacher beat the student" is: **"the oracle's FARM inner loop + the
   student's DIVE behaviour" beat "the student's FARM + the student's DIVE".**
2. **That L1 win is only 0.004 below the line, and it is the same statistic as round one** (the same three seeds).
   Pool 2_133 was consumed **4 more times** in this round (the four new arms), on top of many times in round one
   and its review. **Build nothing on it before it is replicated on a fresh pool.**
3. **"Alive at game end" is still not a fair question between arms of different depth**; the per-floor hazard
   table in section 5.1 and the "did not reach L2/L3" pairings are remedies, but no matched arm "pinned to the same
   depth" was run.
4. **The contact disengage clause still does not move on 68% of its ticks**; this round only reduced its volume.
   The most likely remaining cause is still **a monster standing on the target tile**, and reading
   `local_map["monster"]` would break the "visible monsters only" charter; **whether to allow it is still
   escalated**.
5. **The 400-tick fence was not reached in the third round either** (at most 37 ticks in one game in this round),
   and 8 games had fewer than 400 worker ticks in total. **Whether the number is right is still untested.**
6. **Cross-window settlement**: in the composite arm the oracle serves only FARM windows, so a disengagement fired
   on the last tick of a FARM window can only be settled ("did it actually move?") in the next FARM window. This
   round added a counter: of 128 settlements, **127 were in the same window and 1 crossed windows**, so the bias is
   negligible, but it exists and is recorded here.
7. **`grace_decisions = 0` and `base_fallbacks = 0`: zero game evidence in both rounds**, covered only by unit
   tests.
8. ~~**The engine window ledger and the probe tick ledger differ by 1**~~: **settled in the review round, no longer
   a gap.** The difference now has an exact name and an exact account: in worker-owned windows `_win_beat` is
   called from only three places (the brainstem drain `drain_attempts`, the worker's own tick, and the scripted
   recovery tick when a window opens after the previous fuse, `recovery_actions`), and a tripped tick returns before
   `w["beats"] += 1`, so **worker_calls + drain_attempts + recovery_actions == beats + fuse_trips**. This identity
   closes on every arm (`env_audit.worker_calls_ledger_closes`) and window by window on a real game (section 13).
   The table in section 9 now prints both the env criterion and the worker criterion and says explicitly that the
   old column is not comparable (finding 1). (In this round's world `resource_emergency_stop` is `off`, so the
   stop-v1 scripted tail in `options_env.py` (then line 2044) never ran a tick; if it is ever turned on, this
   identity needs a fourth term.)
9. `sweep_gold` is 0 in all six arms (a telemetry definition issue, as in round one); not pursued in this round.
10. **Only one world (M2-SET), one pool, 48 seeds and `sample` decoding were tested.** Nothing else.
11. **The 36 paired tests have no multiplicity correction**; the 7 that cross the line are really only 3 facts
    (section 6).
12. **`_disengage_move_contact` is still a second copy of the frozen ranking**: the review round added a
    1200-state equivalence test pinning the two copies together (section 13), but **did not merge them into one**;
    merging would mean touching the frozen `teacher_v1.py`, which this round was explicitly not allowed to do. The
    old defect that is really still unfixed is the **68% not moving** (item 4 above).

---

## 15. Recommendations (not rulings)

1. **Step one passed, but through the composite arm, not a pure scripted teacher.** Per the ruling, "the teacher
   beats the parent" is the precondition gate for BC/DAgger. `ORACLE-FARM` clears the line on L1 deaths (3 vs 8,
   UCB95 -0.00437) **without sacrificing survival at game end (28 vs 28) or kills (+14.5%)**. But its DIVE windows
   are the student (section 14.1). **Recommendation: step two can proceed, but the demonstration data should take
   only FARM-window decisions**, which is what the teacher really adds.
2. **First replicate the L1 result on a fresh pool.** It is only 0.004 below the line, one seed flip removes it,
   and it is the same statistic as round one. **This is what I think should be done first.**
3. **Recommendation: drop the drink clause outright, do not tune it.** This round shows that on its own it wipes
   out the whole L1 win (3 -> 8), and 53% of the wanted drinks are refused by the mask. **What it really needs is a
   change to the safety envelope of `drink_sovereignty`, a rule change this round was explicitly not allowed to
   make.** Until then the `0.60` rule does only one thing in the M2-SET world: **burn the belt early**.
4. **Recommendation: drop the contact disengage clause too.** On the same base it changes nothing (the L1 death
   seed set is exactly the same), and the old defect of 68% of ticks not moving is still there. Trying again
   still needs the ruling in section 14.4 (whether the physical occupancy channel may be read).
5. **Pickup clause: do not do it in the "instinct layer".** It is harmless on L1, but together with
   `m[RESUPPLY] = controller_mask[13]` + `_farm_handoff` it forms a **RESUPPLY idle loop of 9.7 ticks per window**
   (7761 windows). Picking up potions is already the scripted service's job; letting the worker do it too only
   makes the manager open short windows over and over. **If it is still wanted, what should change is the
   manager-side window-opening condition, which is a rule, not the teacher.**
6. **If the next round can do only one thing**: move `ORACLE-FARM` unchanged to a fresh pool and replicate L1 (no
   code change, only a different pool). This round's `teacher_v2.py` and driver can be reused directly.

---

## 16. Review round (REVIEW ROUND, 2026-09-11)

The independent review gave **7 findings, all about report / test / evidence hygiene, `verdict: ship`, none
changing any number or conclusion**. The review round fixed **all** 7, and because two of them touched telemetry,
**all six arms were rerun**.

### 16.1 One by one

| # | Level | Issue | Action |
|---|---|---|---|
| 1 | medium | the denominator of `no-effect share` in section 9 was the tick count over **all three window types**, including RESUPPLY, which never calls a worker; RESUPPLY's share ranges from 46% to 58% across arms, **so the column is not comparable between the arms it compares** | **Fixed**: the table now prints `env beats (all windows)` (archive, marked not comparable), `farm+dive beats`, `worker calls` (counted directly by the probes and recomputed window by window from the env's own ledger), and the shares for the two honest denominators. Values: parent 0.1876 / oracle 0.4424 / oracle-farm 0.3033 / of-potion 0.2740 / of-potion-contact 0.2662 / oracle-pickup 0.5021 |
| 2 | medium | three assertions in `test_teacher_v2.routing_real` were **weaker than their own names** (`> 0`, `<=`, `>=`); the `<=` one would PASS even if only 1 of 12 dive windows went to the student | **Fixed**: changed to set equality and exact identities (see section 13). To write exact identities, the observing env gained three per-window counters, `drain_attempts` / `drains` / `recovery_actions` |
| 3 | medium | the one-sentence VERDICT in section 0, "the teacher beat the student for the first time", reads as more than this arm can support: 57.1% of this arm's ticks are played by the student, and the L1 statistic is the same as round one's | **Fixed**: the first sentence of section 0 was rewritten as what was actually proven, and "57.1% is the student", "the same three seeds" and "the L1 pairing of `oracle-farm vs oracle` is 0/0/0" were put into that sentence itself; the ledger is append-only, so the correction is in the `corrections` field of `R19_TEACHER_ROUND2_REVIEW_ROUND` |
| 4 | low | the two regression receipts' mtimes were 6-10 minutes **later** than the four arms they were supposed to gate, because the `--reuse` summarising leg rewrote them; the gate script itself was archived only at the end | **Fixed**: `--reuse` now writes sibling files `regression-*-reuse.json` (`"pass": "reuse-summarize"`), and the gate receipts are written only by the leg that actually runs (`"pass": "run"`); `round2/review-round-manifest-before.txt` records the sha256 + timestamp of every script about to run **before launch** (section 4) |
| 5 | low | no multiplicity statement for the 36 one-sided 95% tests; most of the 7 that cross the line are not independent | **Fixed**: section 6 now opens with the expected number of crossings under the null (1.8) and a table showing that the 7 are really 3 facts |
| 6 | low | `_disengage_move_contact` is a hand copy of the frozen ranking; if the frozen copy is fixed later, this one will not follow, and no test would turn red | **Fixed (pinned, not merged)**: a new 1200-state pure-Python equivalence test + a non-vacuity test ("must differ when stairs are present"). Merging into one copy would touch the frozen `teacher_v1.py`, which this round was not allowed to do |
| 7 | low | `ClauseTeacher` passed `variant="oracle"` to the base, so every row of the three clause arms had `telemetry["variant"]` = `oracle` | **Fixed**: `ClauseTeacher` sets `self.variant` to the arm name after `super().__init__` and writes it again in `_blank_telemetry`. In the frozen `teacher_v1`, `self.variant` is read in only two places (telemetry `:282`, the strict-assertion message `:350`) and **is never a decision input**, which the unchanged SHAs of all six arms prove empirically |

### 16.2 Reruns (findings 2 and 7 touched telemetry, so all six arms were run)

**Both regression gates pass, and the four new arms' SHAs are also bit-identical**, i.e. every change above is
**pure telemetry / pure reporting** and none of them changed a single decision:

| arm | pass1 (first run, 2026-09-10) | pass2 (review round, 2026-09-11) | pass3 (review round final, 2026-09-11) | all equal | expected (regression) |
|---|---|---|---|---|---|
| parent | f33f7af5f236f1fc | f33f7af5f236f1fc | f33f7af5f236f1fc | True | f33f7af5f236f1fc == expected |
| oracle | 1507c3be68a0da29 | 1507c3be68a0da29 | 1507c3be68a0da29 | True | 1507c3be68a0da29 == expected |
| oracle-farm | e47d5a5850ae1c46 | e47d5a5850ae1c46 | e47d5a5850ae1c46 | True | (new arm, no reference constant) |
| of-potion | 17d3ea46f63f88a6 | 17d3ea46f63f88a6 | 17d3ea46f63f88a6 | True | (new arm, no reference constant) |
| of-potion-contact | 525d18ef98ddbe8d | 525d18ef98ddbe8d | 525d18ef98ddbe8d | True | (new arm, no reference constant) |
| oracle-pickup | 3f02ce54b8922626 | 3f02ce54b8922626 | 3f02ce54b8922626 | True | (new arm, no reference constant) |

  every arm identical across all three passes: True

  arm_stats identical across the three passes (the whole dict, not just the SHA):
  parent: True
  oracle: True
  oracle-farm: True
  of-potion: True
  of-potion-contact: True
  oracle-pickup: True

  per-arm receipts of the FINAL pass:
| arm | elapsed_s | runtime_errors | source_changed_during_run | worker calls (probe) | worker calls (env ledger) | ledger closes |
|---|---|---|---|---|---|---|
| parent | 592.5 | 0 | [] | 76228 | 76228 | True |
| oracle | 511.2 | 0 | [] | 51381 | 51381 | True |
| oracle-farm | 686.7 | 0 | [] | 90418 | 90418 | True |
| of-potion | 640.4 | 0 | [] | 79881 | 79881 | True |
| of-potion-contact | 648.5 | 0 | [] | 75963 | 75963 | True |
| oracle-pickup | 681.7 | 0 | [] | 54273 | 54273 | True |

  the two GATE receipts of the final pass (written by the RUN leg, never by the --reuse summarize leg):
  parent: equal=True pass='run' all_references_agree=True sha=f33f7af5f236f1fc
  oracle: equal=True pass='run' all_references_agree=True sha=1507c3be68a0da29

  telemetry variant label per arm (RR2 finding 7), row 0 of each:
  parent: farm_policy.telemetry['variant'] = None, v2_label = None
  oracle: farm_policy.telemetry['variant'] = 'oracle', v2_label = None
  oracle-farm: farm_policy.telemetry['variant'] = 'oracle', v2_label = None
  of-potion: farm_policy.telemetry['variant'] = 'of-potion', v2_label = 'of-potion'
  of-potion-contact: farm_policy.telemetry['variant'] = 'of-potion-contact', v2_label = 'of-potion-contact'
  oracle-pickup: farm_policy.telemetry['variant'] = 'oracle-pickup', v2_label = 'oracle-pickup'

  L1 death seed sets (RR2 finding 3: oracle and oracle-farm are the SAME statistic):
  parent: [2133008, 2133009, 2133015, 2133017, 2133023, 2133024, 2133039, 2133044]
  oracle: [2133017, 2133021, 2133026]
  oracle-farm: [2133017, 2133021, 2133026]
  of-potion: [2133009, 2133021, 2133027, 2133029, 2133033, 2133038, 2133039, 2133041]
  of-potion-contact: [2133009, 2133021, 2133027, 2133029, 2133033, 2133038, 2133039, 2133041]
  oracle-pickup: [2133011, 2133021, 2133026, 2133036]

  who actually played (RR2 finding 3), worker calls by slot:
  parent: total 76228 -- farm_slot(parent) 76228 = 100.0% [farm 34968 / dive 41260 / resupply 0; forced_dive beats 31801]
  oracle: total 51381 -- farm_slot(oracle) 51381 = 100.0% [farm 32906 / dive 18475 / resupply 0; forced_dive beats 16007]
  oracle-farm: total 90418 -- dive_slot(parent) 51624 = 57.1% [farm 0 / dive 51624 / resupply 0; forced_dive beats 33901], farm_slot(oracle) 38794 = 42.9% [farm 38794 / dive 0 / resupply 0; forced_dive beats 0]
  of-potion: total 79881 -- dive_slot(parent) 46515 = 58.2% [farm 0 / dive 46515 / resupply 0; forced_dive beats 31831], farm_slot(oracle) 33366 = 41.8% [farm 33366 / dive 0 / resupply 0; forced_dive beats 0]
  of-potion-contact: total 75963 -- dive_slot(parent) 43980 = 57.9% [farm 0 / dive 43980 / resupply 0; forced_dive beats 30983], farm_slot(oracle) 31983 = 42.1% [farm 31983 / dive 0 / resupply 0; forced_dive beats 0]
  oracle-pickup: total 54273 -- farm_slot(oracle) 54273 = 100.0% [farm 27176 / dive 27097 / resupply 0; forced_dive beats 15095]

### 16.3 Read/write discipline of the review round

- The main tree: 283 fingerprint entries + the `build` symlink + `_diablogym*.so`, **diff empty** against
  `main-tree-fingerprint-before-round2.txt` (`round2/main-tree-fingerprint-review-before.txt` and
  `…-review-after.txt`); the only main-tree file newer than the start of the round is still `r13_ledger.jsonl`.
- `m2-merge` and `oracle-tree`: `find … -newermt <start of the round>` is **empty**.
- `teacher_v1.py` (`65c1a14f…`) and `test_teacher_v1.py` (`f12d94d9…`) in `oracle2-tree` are **bit-identical** to
  the read-only `oracle-tree`.
- All outputs of the first and second passes are kept unchanged: `round2/probe-pass1/`, `round2/probe-pass2/`,
  `round2/TEACHER-ROUND2-REPORT.pass1.md`, `round2/driver-*.pass1.log`; the per-arm comparison of the first and
  final passes (SHA, the whole `arm_stats` dict, the pairing block, named seeds) is in
  `round2/review-round-compare.txt`; the SHA comparison of the three passes is in section 16.2.
- Seeds: still only 2133000-2133047, 48 per arm, the same set in every arm; fresh pools 2_116-119 / 2_126-128
  **not touched**.

### 16.4 Final VERDICT after the review round

**Unchanged.** The seven findings changed no number, and `rows_sha_v3` is unchanged for all six arms.

> **One arm clears the line on L1 deaths: `ORACLE-FARM`** (L1 deaths 3 vs 8, paired saved 7 / lost 2 / net +5 /
> one-sided UCB95 -0.00437), **while not wrecking survival at game end (28 vs 28) or kills (5550 -> 6353,
> +14.5%)**. This is the literal answer to the question as posed: "yes, and only this one".
>
> **But what won is "the oracle's FARM inner loop + the student's DIVE behaviour"**; 57.1% of this arm's worker
> ticks are still the student itself; **and its L1 statistic is the same statistic as round one's `ORACLE`** (the
> same three seeds; the L1 pairing of `oracle-farm vs oracle` is 0/0/0), only 0.004 below the 0 line, with 1.8 of
> 36 uncorrected tests expected to cross by luck alone. **Nothing should be built on this 95% result before it is
> replicated on a fresh pool.**
>
> **Which helps, which hurts**: the drink clause **hurts** (on its own it pushes L1 from 3 back to 8, survival
> 28 -> 20); the contact disengage clause **does nothing** (its L1 death seed set is exactly that of of-potion, and
> 68.0% of its disengage ticks still do not move); the pickup clause **is harmless on L1 but blows up the window
> count**. **Explanation of the RESUPPLY window explosion**: `m[RESUPPLY] = controller_mask[13]` +
> `_farm_handoff` + the `FARM -> RESUPPLY -> DIVE` fallback ladder, triggered by **the pickup walking macro leading
> the character away from the monsters**, form an idle loop of short windows at 9.7 ticks each (1735 -> 7761
> windows), **not** more a13 presses (`ORACLE-PICKUP`'s a13 is 698, fewer than `ORACLE`'s 909).

---

*This report was written by the teacher round-two implementer on 2026-09-10 and 2026-09-11. All numbers come from
files this round produced under `round2/probe/`, and the tables are machine-generated by
`round2/teacher2_extra.py`; round-one numbers come from the unchanged `teacher-probe/` (both in the local work
directory, not published); every place without a source is marked "not verified". The review round (section 16)
ran on 2026-09-11 on the same tree: all six arms rerun, both regression gates passed again, 85 unit tests pass.*

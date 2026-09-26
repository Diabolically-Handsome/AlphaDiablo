# PREREG-v31: new-world re-education (opening campaign of the v3 season; final version)

Status: **final version** (critic panel: one BLOCK vote + two AMEND votes; all 4 blockers / 12 major /
15 minor findings addressed; the two freeze prerequisites raised by the BLOCK, RESUME_SMOKE and G-R31,
have passed live and are in the ledger). Freeze discipline follows R1/R2: the commit is the notarisation,
and the text contains no self-referential sha. Campaign number v31 is assigned here under the era-switch
note of the [roadmap](../design/ROADMAP-course-plan.md); **the provisional v2-era numbers of courses 3, 4
and 5 (v31/v32/v33) are all void and will be assigned separately** (the roadmap note is updated in the
freeze commit).

## Rationale and authorisation

Course campaigns are unlocked from the recording of the R2 verdict (4618ee2). The v3 world differs
materially from the v2 training world: every manager in the incumbent lineage was taught in the old
world. **The succession question**: can a manager re-educated in the v3 world beat the incumbent
assembled agent? Two arms = "v3 from scratch" vs "continued training of the v2 stock", which also answers
"is old knowledge an asset or a burden in the new world" (the manager-side sequel to v25 "naive
fine-tuning" and v28 "a long tether binds").

**Scoping decision on record (panel major)**: this case is not among courses 2-5 of the roadmap, nor one
of the two candidates in R2's follow-up section (course 5 / the A rematch right); it is a re-run of course 1
(v29 manager re-education) in the v3 world. Applying the roadmap-level approval to this case is the
drafter's reasoning: every course campaign of the v3 era presupposes that "the incumbent assembled agent
still deserves its place in the new world", so the succession question is a gate that comes before the
course sequence. That reasoning is recorded as is; construction-level authorisation still rests on this
pre-registration passing its panel plus the freeze commit.

**Authorisation chain (panel major)**: the restart of training rests on the same go-ahead of 2026-07-12
that the R2 case already cited for its gold-seed re-measurement. This second use of one authorisation is
recorded as such (following R2's transcription discipline).

**Gold seeds (carried over from R1, panel major)**: if this case launches, it uses the gold seeds
9000-9031 once (v31-golden), under the campaign-level gold-evaluation discipline (precedent v24-v30, with
no second per-case sign-off; that choice is recorded in the R2 rationale). The GOLDEN_AUTHORIZED event
records that its authorisation comes from the campaign-level discipline, without a case-specific
approval.

**A rematch right**: not consumed. A is the "manager entropy-balanced rematch" (wording of the direction
review, not published); both arms of this case use the same ent 0.02, so there is no entropy variable
between the arms. The right survives, and its deferral clause ("usable only after the new v3 anchor is in
place") is unaffected by this case.

## Line-up (all sha values are frozen driver constants, asserted in preflight)

- Worker (unchanged): v28-worker-leg1 npz (976b6c05…). **Instrument definition fix**: every worker
  evaluation in v31 uses the **npz representation** (v29 mixed zip and npz; npz≡zip bit-level equivalence
  has not been re-proven under v3, and a single representation removes instrument asymmetry. This is a
  definitional choice, not a proof of equivalence; residual #3).
- Incumbent assembled agent (the challenged): v22-H × v28-leg1 (v3 gold-pool anchor 133.9 = r2-launch).
  **How v3-launch-anchor is used**: the gold-pool archive and the probe pairing line are on different
  pools, so under v29's same-pool per-seed pairing rule it cannot serve as the pairing baseline; this is
  the whole reason for spending a separate v31-ref-launch. The use of 133.9 is limited to cross-pool
  observation, the D3-0 circuit-breaker input and the title table; this is not a downgrade of the anchor.
- Initialisation of the stock arm: the M29-fresh checkpoint (9d5820bf…, nt=160000); its npz archive is
  89441388….
- **Use of the R2 anchors (full form, completed at panel major)**: beyond the full-text sha assertions of
  the four anchors, preflight re-checks comparability with the current runtime: freeze_eval_identity (each
  archive with its original line-up: launch=W×default H / science=W×M29 / throne=v24-leg7×default H /
  script=script×default H) → expected_eval_identity(tag, 9000-9031) → read_eval_archive →
  verify_eval_identity. Any failure → halt and report "R2 anchors are not comparable with the current
  runtime; a separate re-anchoring case is needed" (the scientific readings are final; re-spending is
  forbidden).
- **Incumbent identity pinned (panel major)**: the reference run's manager == the manager of r2-launch,
  guaranteed jointly by the built-in DEFAULT_MANAGER_SHA256 assertion on the default manager path of
  freeze_eval_identity and by the full-form re-check of r2-launch; both sha values are recorded with
  preflight_ok.

## D2 recipe and budget

**Construction items (panel BLOCK 1, in the freeze commit)**: train_ppo.py gets a manager-side options
resume path: 1. the admission gate `(worker ∨ options) ∧ algo==mppo`; 2. checkpoints on the options path
go through the common gate (CRC / key members / step count / finite weights), without the distill_beta
assertion (that marker belongs to the Leashed worker only); 3. class-faithful loading via
`MaskablePPO.load` (resume the class that was saved; no teacher/β involved, no G-KL-B obligation), with the
v24 seal assertions (lr/ent/gamma/n_steps/batch/target_kl/seed reset) unchanged; 4. `--teacher-override`
is restricted to the worker side. Construction is limited to the training side; the eval protocol bundle
is not touched. **Freeze prerequisites (done, in the ledger)**: RESUME_SMOKE (rc=0, 160000→160256,
continuous step by step) + G-R31 (zero-training load and export == the archived npz, 6/6 arrays equal bit
for bit, parity 0/1000; the driver re-proves it at every launch).

**Probe-pool references (two opening runs, 7000-7031; gates use no gold seeds)**:
- `v31-ref-launch` = W-npz × default H: the same-pool reference for the launch pairing; its mean R is the
  base from which the decision lines are derived on site. `v31-ref-science` = W-npz × M29-npz: the v3
  probe reading of the previous lineage, the pairing base for the net change of the cont arm. The refs
  event records the full set of depth gauges for both runs (died/depth2/DIVE/bonus/the three sentinel
  rates).
- **D3-0 reference sanity gate (panel BLOCK 3; an operations guard rail, not a scientific decision)**:
  v31-ref-launch must have died ≤ 8 ∧ R ∈ [100, 170] (endpoints registered by the panel, a wide band
  centred on the gold-pool anchor 133.9; here 133.9 is only a circuit-breaker input, consistent with "record
  across pools, do not decide"). Out of range → CASE_HALT_REF_CORRIDOR, halt and report: no arm is
  trained, no gold seed is spent, and no archive on record may stand in as the baseline; a re-test needs a
  separate case and a new freeze. This gate is the equivalent, after the world change, of v29's G-A0
  instrument gate.
- **Reference finality clause (panel blocker/major combined)**: once both references are validly filed
  they are frozen at case level (their sha values go into the ledger with the ref_archive event); on a
  resumed run, the "target archive does not exist" preflight does not apply to them, the first exam is
  accepted by its ledger sha, and re-spending via .void is forbidden. A damaged archive or a failed
  re-check → REF_INVALID halts the whole case; only after a correction commit may the whole case be reset
  (CASE_RESET, at most once per case; old arm archives are voided together, and mixing new references
  with old arms is forbidden). A reference run may be retaken automatically once only for a crash or an
  invalid archive (exam_retry semantics); a high or low score is no reason to retake. Two failures → halt;
  r2-launch may not stand in across pools. Both reference runs must be filed before any arm is trained;
  the absolute values of the decision lines are those recorded in the refs event and may not be
  recomputed during the case.
- **Case-level runtime reconciliation (panel major; the in-case version of R2 W10)**: preflight records
  CASE_RUNTIME (five sha values: bridge/engine/game_data/assets/python_protocol); before every later exam
  launch, the current snapshot is asserted equal to it; a mismatch → halt and report
  CASE_HALT_RUNTIME_DRIFT, not retaken as a crash and no .void.

**Two arms in series** (shared: `--options --algo mppo --gamma 1.0 --max-steps 3000 --n-steps 64
--num-envs 4 --worker-npz <W-npz>`):
- **v31-mfresh**: the v29-fresh recipe verbatim (lr 3e-4, ent 0.02, seed 22), `--total-steps 160000`;
  nt gate == 160000.
- **v31-mcont**: `--resume-from <M29 zip> --allow-legacy-resume`, lr 3e-4, ent 0.02, seed 26,
  **`--total-steps 160000` (incremental semantics; panel BLOCK 2 correction: train_ppo's --total-steps
  means "new samples", resume legs use reset_num_timesteps=False, and SB3 continues with +=; the v24 audit
  comment is on record)**: 160000 of re-education on top of the 160000 stock; **nt gate == 320000**
  (= 1250×256 as a book note; the new segment of 160000 = 625×256 satisfies the divisibility gate). The
  driver's ARMS entry has two fields: cli_steps (goes into the command line) and nt_target (the
  completion gate).
- **R31.2 wording + dual-definition duty (panel major)**: the arm comparison = an overall comparison of
  "the v3 from-scratch path vs the v2 stock continued-training path" (init / step accounting / seed are
  bundled into the path definition); single-factor attribution is forbidden. Any verdict or appendix that
  cites the arm comparison must **show both** definitions side by side, "cont total steps 2× (320k
  cumulative)" and "equal new-world steps (160k v3 steps each)"; single-definition citation is forbidden.
  If cont loses → the verdict must carry the qualification "hyperparameters were not adapted for continued
  training (no lr schedule, inherited optimiser state, entropy follows the fresh recipe); the collapse is a
  path-level result and must not be read as old knowledge necessarily being a burden (v25 naive
  fine-tuning precedent)"; if cont wins → attribution to any single factor (total steps / initialisation /
  seed) is forbidden. r31_2 is a single paired descriptive quantity and must not be read for significance.
- Clock: 10-30 minutes expected per arm (measured after the post-R2 evaluation speed-up); timeout still
  4 h per arm (margin discipline); evaluation 30 minutes.

## D3 decision ladder (decision lines derived from R on site; **canonical = the fractions 75/112.4 and 85/92**; 0.6673/0.9239 are display roundings, not the computing definition; the verdict text prints the fractions)

0. **Reference sanity gate** (see D2; before any arm training).
1. **Early abandonment gate**: both arms' s16 < (75/112.4)×R → "training failed; the succession question
   was not examined".
2. **Eligibility per arm** (v29 verbatim): died ≤6 ∧ not voided ∧ sentinel gate (descend ≤2.04% /
   cap <5% / override <3%, ≥8% voids; the dual-attribution clause unchanged); winner = the eligible arm
   with the highest full-32 mean (within ±0.05 → fewer deaths → take mfresh); substitution and no-winner
   tiers unchanged.
3. **Reproduction floor**: winner < (85/92)×R → "retraining did not reproduce the reference level; the
   question was not examined".
4. **Launch line**: paired against v31-ref-launch (join on seed key + set assertion): mean difference
   ≥+4 ∧ wins ≥18/32 ∧ died ≤6 ∧ sentinel; the dual-attribution "decide first, then spend" rule unchanged.
5. **Exhaustive no-launch tiers** (v29 verbatim): point-estimate gain / probe level [+2,+4) / re-elected
   <+2; wins ≥14 carries a width note.
6. **Gold evaluation (if launched; one arm, once, started manually by the operator)**: assembled agent =
   winner npz × W-npz, gold pool, tag v31-golden. **The P line is read against the throne's incumbent
   anchor value 97.7 (r2-throne, the R2 re-measurement; re-measurement does not change the title, and the
   throne title still belongs to the incumbent assembled agent) and the script reference 94.2
   (r2-script); values are read from the archives on site**: died >6 → regression; gold ≥ throne+4 ∧
   died ≤4 → P31 taking the throne; **∈(throne, throne+4)** ∧ died ≤4 → point-estimate gain, throne
   unchanged (open interval, following v29's definition; panel correction); > throne ∧ died 5-6 → level
   (safety); ∈[script, throne] → level; < script → regression. **Gold-pool opening history**: the gold
   evaluation was actually opened three times, for v22/v23/v24 (v29/v30 did not touch it); the five R2
   anchoring runs were measurement exposure, not openings; if this case launches, it is the 4th real
   gold opening and the 9th cumulative gold-pool exposure, and the fixed-pool bias is noted in the
   verdict.
   **Title transfer clause (panel blocker; two gates, two titles; the anchor-compliance rule of
   PREREG-v30 D3 written out)**: the throne and the incumbent assembled agent are two independent titles,
   decided by the gold P line and by the launch pairing line respectively; the verdict announces them on
   separate lines, and merging them into one narrative is forbidden. (a) **Succession criterion**: the
   winner passes the D3-4 launch line ∧ the gold evaluation is not in a regression tier (gold ≥ 94.2 ∧
   gold died ≤6) → the incumbent assembled agent changes: new incumbent = winner manager ×
   v28-worker-leg1; the v3 launch anchor moves with it := the v31-golden archive (anchor follows the
   throne); r2-launch 133.9 is downgraded to a historical reading of the predecessor and no longer serves
   as a pairing/table baseline, and the verdict appendix notes the end of its v3-launch-anchor role.
   (b) Gold evaluation in a regression tier (<94.2 or died >6) → succession failed: the incumbent and the
   launch anchor stay; the verdict records "gold evaluation regressed after launch, succession failed",
   and the launch itself is recorded as usual. (c) **The P line moves only the throne**: P31 taking the
   throne decides only the throne title; succession without a change of throne is a legal combination, and
   the verdict must note "the incumbent assembled agent and the throne belong to different assembled
   agents (the arrangement since v28)" and list both gold-pool readings. (d) No launch → neither title
   moves, and the verdict may use no title verb other than "re-elected".
7. **Depth secondary verdict + on-site baseline clause (panel major)**: the boundaries of the three tiers
   (≥12 learned / ≤7 not unlocked / out of band) are numbers inherited from v2; the refs event records the
   measured depth2 of ref-launch as the **on-site baseline**, which the verdict must cite alongside; if the
   on-site baseline is ≥12, the "learned" tier must be restated as "reaches the v2 inherited line, but the
   on-site baseline is already above it, so it does not discriminate"; no tier on its own is a Mark
   argument (the footnote discipline of v29 D3-7).
8. **Scientific observation (recorded, not decided; panel correction)**: **always compute** the per-seed
   paired difference of the cont arm's full 32 against v31-ref-science ("the net change of the stock
   continued training relative to its predecessor", independent of which arm wins); if the winner is not
   cont, add the winner comparison as cross-lineage side evidence; on the abandonment-gate path this
   observation is absent, as noted in the verdict.
9. **Multiple-comparison ledger (recorded, not decided; from v30 D3-7)**: this is the 5th challenger draw
   on the same-pool 18/32 line (11→16→17→v30 did not reach the launch check→this case); the note
   P(wins≥18|p=.5)≈43% goes with the verdict; the note also records that this case's baseline is the R
   derived on site rather than a historical archive.
10. **Limitation: the lines have not been recalibrated (panel major)**: every decision line in this case
    is a formal inheritance of v2 operating characteristics; type I / type II error rates under the v3
    score scale and variance structure have not been recalibrated. Near-line verdicts (mean difference
    within 1.0 of the line or wins within 1 of the line) must carry the note "near the line + line not
    recalibrated"; the death gate / void line being systematically too tight for deep-diving behaviour in
    v3 is a known bias, and any verdict decided by them must also cite the depth distribution and per-seed
    died; recalibration is a separate case, and adjusting lines within this case is forbidden.

## R lines (falsifiable predictions; point values registered by the drafter before freezing)

| # | Prediction | Band, point |
|---|---|---|
| R31.1 | Mean R of v31-ref-launch | ∈[100,145], point 125 |
| R31.2 | Paired (cont−fresh, full 32, same seeds) | ∈[−15,+15], point 0 (open in both directions) |
| R31.3 | Paired mean difference of the winner vs R | ∈[−10,+15], point +2 |
| R31.4 | Winner's number of depth≥2 seeds | ∈[8,28], point 16 (v3 depth shifted upward) |
| R31.5 | Gold (if launched) | ∈[95,140], point 115 |

## Residual uncertainty

1. The winner is a max-of-2 order statistic, which inflates the type I error of the launch line (v29
   residual #6 carried over).
2. The two arms bundle init / step accounting / seed into the path definition; attribution is limited to
   the overall comparison.
3. The decision lines are derived from R on site: R itself is one 32-seed sample, and the proportional
   lines shift with its variance. Registered as is, with no re-spending (the D3-0 corridor guards against
   collapse, not against sampling noise).
4. npz≡zip bit-level equivalence has not been re-proven under v3; using npz throughout is a definitional
   choice, not a proof.
5. The family-wise error rate of repeated same-pool challenger draws on the probe pool accumulates;
   disclosed in D3-9.
6. The cont arm's --allow-legacy-resume is a registered use of a one-off migration path; the numeric
   semantics of an old checkpoint under the v3 engine are not proven bit for bit, and are backstopped by
   the nt gate / seal assertions / G-R31.
7. cont's hyperparameters were not adapted for continued training (no lr schedule, inherited optimiser
   state).

## D4 P line (all v29 operations guard rails carried over, plus additions for this case)

Top-level exception → DRIVER_EXCEPTION + NEEDS_ATTENTION; timeouts of 4 h for training / 30 minutes for
evaluation (killpg the whole group + a guard against the ProcessLookupError race); exams refuse to
overwrite + .void rotation + timestamped logs; nt_zip is the only step-count source (asserted exactly
against nt_target per arm); preflight (full-form re-check of the four R2 anchors / artefact sha /
RESUME_SMOKE on record / G-R31 re-proof / target archive does not exist (except the two reference runs,
which use finality acceptance) / no leftovers in the ARMS directories / PROTOCOL_VERSION==3 self-refusal
gate); CASE_RUNTIME reconciliation at every exam; driver exclusive_lock; training failure / target not
reached → halt and report, and this version adds no retraining (v25 clause).
**Exit codes (panel minor; one preflight channel: every driver require raises OperationalFailure)**:
0 = verdict recorded (including no-launch tiers) / GOLDEN_AUTHORIZED waiting for a manual start;
2 = operational failure (preflight failed / reference sanity gate / training target not reached / repeated
exam failures / runtime drift / ledger unparseable); any other non-zero or abnormal death = P1
(DRIVER_EXCEPTION).

## D5 ledger and verdict discipline

`train/runs/v31/gate_ledger.jsonl` (not published). Full-32 events carry the depth gauges; arm_done contains the
steps_status diagnostic field (nt_zip stays the only step-count source; clone differences are declared as
they are); a repeated-failure path records a STOP event + attention (aligned with the v29 event
vocabulary). **Verdict discipline (panel major)**: GOLDEN_AUTHORIZED and every VERDICT_PATH event must
carry three numbers: both arms' full-32 means, r31_2 and R. The verdict appendix is written at the end of
this document.

---

## Verdict appendix (case closed 2026-07-13; the commit of this appendix is the notarisation, and it may not be changed once recorded)

Freeze commit `8c153b5`; launched on 2026-07-12, closed on 2026-07-13 (2 h 56 min in total: two reference
runs 4 min, fresh arm 93.4 min, cont arm 76.0 min, exams and decision 4 min). Driver exit code 0, with
zero OPERATIONAL events and zero retakes.

**Case verdict: no launch. The winner's 88.7 < the reproduction floor of 104.4 (=(85/92)×113.0),
"retraining did not reproduce the reference level; the succession question was not examined". No title
changes (D3-6(d): incumbent assembled agent = v22-H×v28-leg1, throne = the v24-golden lineage, both in
place; no launch, so no title verb is used). The gold seeds were not touched (the launch criterion was
never reached and launch_check was not recorded; the 5th draw on the 18/32 line was not consumed).**

**Raw numbers of the two arms (the three numbers of the verdict discipline + the dual-definition duty)**:
fresh full32 = **115.4, died 7/32**; cont full32 = **88.7, died 0/32**; r31_2 (cont−fresh paired) =
**−26.74**; R = 113.0. Step accounting in both definitions: cont total steps 2× (320k cumulative) ∧
equal new-world steps (160k v3 steps each); both definitions side by side, never one alone.

### Three scientific conclusions (each with its registered qualification)

1. **Continued training of the stock collapsed (cont): a manager-side replication of v25's "naive
   fine-tuning collapses"**: cont's paired net change against the v3 probe reading of its predecessor M29
   (140.9) is **−52.19, winning 1/32** (science_observation on record). Mandatory qualification: cont's
   hyperparameters were not adapted for continued training (lr 3e-4 without a schedule, inherited
   optimiser state, entropy follows the fresh recipe); the collapse is a **path-level result** and must
   not be read as the general conclusion "old knowledge is necessarily a burden". 160k new-world steps
   ground a 140.9 manager down to 88.7, but it did not lose a single life: the damage went toward
   retreating to shallow levels (depth2 15→2, DIVE 0.69→0.09, bonus 4.25→0.5 per episode), not toward
   dying.
2. **The from-scratch arm missed by one life (fresh): a depth learner stopped by the death gate**: 115.4
   (+2.4 against R, probe level), depth2 = **12** (reaches the v2 inherited "learned" line; the on-site
   baseline ref depth2 = 7), DIVE 0.56 per episode, bonus 3.5 per episode: the whole set of depth-economy
   gauges lit up; but **died 7 > 6, and the eligibility gate stopped it by one life** (substitution event
   on record; D3-2 substituted cont). **Near-line note (mandatory under D3-10)**: exactly one life from the
   line, "near the line + line not recalibrated". The death gate being systematically too tight for
   deep-diving behaviour in v3 is a known bias registered in the pre-registration, and this case is the
   first time it bit live; per-seed died and the depth distribution are in the full32 event archive
   (cb1e0b9f). fresh is a reincarnation of v29-explore (which was voided at died 8; this time died 7 blocks
   eligibility).
3. **Depth secondary verdict (for the substitute winner cont)**: "depth not unlocked (depth2=2 ≤ baseline
   7; on-site baseline ref depth2=7)". The secondary verdict is recorded as is and is not a Mark argument.

### R-line reconciliation (honest opening)

| # | Predicted band, point | Measured | Result |
|---|---|---|---|
| R31.1 | R ∈[100,145], point 125 | 113.0 | **in band** |
| R31.2 | cont−fresh ∈[−15,+15], point 0 | **−26.74** | **out of band** (the collapse was larger than the registration imagined) |
| R31.3 | winner vs R ∈[−10,+15], point +2 | −24.3 (substitute winner) | **out of band** (fresh's +2.4 was in band, but the winner was the substitute) |
| R31.4 | winner depth2 ∈[8,28], point 16 | 2 (substitute winner; fresh=12 in band) | **out of band** |
| R31.5 | gold | no launch | not opened |

All three out-of-band lines come from the "substitute winner" path: the prediction bands were written for
"the winner" and did not foresee the mean winner being blocked by eligibility and replaced by the
collapsed arm. Recorded as is; no bands are changed after the fact.

### Follow-ups (proposed; not executed in this case)

- fresh's one-life miss nails down an existing conclusion of the
  [roadmap](../design/ROADMAP-course-plan.md) even harder: **the depth learner lacks survival tools**
  (the same source as v30's "FARM retraining worsens deep deaths"). The design memo for courses 3 (gear)
  and 4 (potions) is on the critical path and is next on the agenda
  ([DESIGN-gear-and-potion-autonomy](../design/DESIGN-gear-and-potion-autonomy.md)).
- Recalibrating the death gate for v3 is a separate case (D3-10: no line adjustment within this case); if
  it is recalibrated, fresh's archive (cb1e0b9f) is the first material for review. **All fresh weights and
  archives are kept.**
- A protocol for continued training of stock (lr schedule / adaptation) is a separate case; the A rematch
  right is untouched.

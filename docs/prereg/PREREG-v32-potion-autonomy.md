# PREREG-v32: potion autonomy (4C: unmask with the safety net kept; draft, under panel review)

Status: **draft** (finalised after the panel review; freeze discipline follows R1/R2/v31: the commit is
the notarisation, and the text contains no self-referential sha). **Campaign number v32 is assigned here
under the era-switch note of the [roadmap](../design/ROADMAP-course-plan.md) (course 4C, potion
autonomy); sharing the number with the provisional v2 table entry 4=v32 is a coincidence, not an
inheritance, and provisional numbers are historical only; the roadmap note is updated in the freeze
commit.**

## Rationale and authorisation

- **Decision**: on 2026-07-15 the proposals of the design memo
  ([DESIGN-gear-and-potion-autonomy](../design/DESIGN-gear-and-potion-autonomy.md)) were decided: 4C
  (unmask with the safety net kept) and decoupling first were approved, 3A/3C were approved together,
  and 3B′/3B and 4B/4A were deferred and kept on record. **From this point the environment-side change
  is approved**, and this case is its registered construction.
- **Motivation**: the v31 verdict. The fresh arm was stopped by one life at died 7/32 (depth can be
  learned; lives cannot be kept); the death gate being systematically too tight for deep-diving behaviour
  in v3 is a registered bias. 4C answers directly: give the worker the right to drink on its own before
  half HP.
- **Decoupling clause (an obligation of the decision; three-way split per panel correction)**: the wage
  formula is not touched (DESCEND_UNIT=8.0 single source of truth; the wage identity unchanged). Transfer
  paths on failure: (a) main verdict in **tier 3 (used but ineffective / harmful) → as the decision
  provides, move to the 4 × 2 bundled evaluation case with its own pre-registration**; (b) main verdict in
  tier 2 (autonomy given but not used) → reopen the 4B agenda under item 5 of the decision (proposed
  separately, not bundled automatically); (c) the case is not examined at all (abandonment gate / floor /
  both arms ineligible / G0 failure / quota exhausted) → halt and report for a decision; the decision
  stays valid and nothing is transferred automatically.
- **Gold seeds (panel correction: the draft's citation was wrong, since the design memo never mentions
  gold evaluation; a drafting error, recorded as such)**: if this case launches, it uses 9000-9031 once
  (v32-golden), under the campaign-level gold-evaluation discipline (R2 rationale: campaign-level notice +
  a frozen pre-registration authorise the launch, with no second per-case sign-off) and the roadmap clause
  "each case's winner challenges the throne under gold-standard discipline" (roadmap of 2026-07-11). The
  GOLDEN_AUTHORIZED event records that its authorisation comes from the campaign-level discipline,
  without a case-specific approval (v31 clause verbatim).
- **Prior evidence (pre-case probe, on record)**: on 166,383 real worker observations, unmasking 12 flips
  the argmax **0 times** for both the incumbent worker v28-leg1 and the throne worker v24-leg7 (the a12
  logit never wins outright): direct evidence that 4C is inert on the trajectories of the deployed
  policies, and the foundation of the anchor-bridge clause (D3).

## One campaign, one prescription

Prescription = "**worker potion autonomy** (4C: the 0.5 reflex stays as the floor, m[12] unmasked) +
**relay retraining from the king anchor** (the v30 'worker relay' king-leg lineage, resuming the
incumbent worker v28-leg1, not the throne worker v24-leg7; the two titles must not be merged in the
narrative): can it be turned into survival and return?". Two legs on the same seed (sov/ctrl) exist
specifically to separate "the autonomy effect vs the retraining effect". No gold seeds (except the
registered gold evaluation), the probe pool only for gates, the wage formula untouched, 3B′ not adopted
(G0-ledger exemption registered).

## Construction list (environment side, approved, in the freeze commit)

- **E1 parameterised unmasking (with a belt precondition)**: `OptionsEnv`/`WorkerWindowEnv` get the
  constructor parameter `drink_sovereignty: bool = True` (**default True = the normal state of the new
  protocol**, with the semantics of 4C; it is parameterised rather than a bare line deletion so that it
  can serve the control leg: within the intent of the decision, with its form registered here). With
  autonomy on, `m[12] = base mask ∧ belt_heals>0` (**belt precondition**, aligned with the key-14
  precedent "legal only when there is gear"; the G0-ghost measurement showed the base mask does not look
  at the belt, and an empty-belt no-op step is a churn key that the fuse forcibly rewrites, pushing up
  overrides into the R4 sentinel; this precondition is a necessary closure found during construction and
  is registered as such). With autonomy off, `m[12] = False` (the old protocol verbatim). The reflex
  `_drain()` and the reflex inlined in dispatch are **not touched** (the 0.5 floor is always there).
  **Scope clause (panel major)**: the belt precondition adds semantics beyond the approved change (basis:
  the intent of 4C + alignment with the key-14 precedent + the necessity shown by the ghost measurement).
  It is listed separately for ratification at freeze; until it is ratified, the ledger event
  E1_SCOPE_NOTE records it as a discretionary addition without a specific approval. If ratification is
  refused → E1 reverts to plain unmasking, the empty-step semantics become a separate case, and G0 is
  rerun with a correction and a re-freeze.
- **E3 test sync**: the mask assertions in tests/test_worker_env.py are rewritten for the new protocol
  (the belt precondition of the unmasked key 12: m[12] = base mask ∧ belt>0, the same formula as E1; a
  dual assertion that the old-protocol knob always masks; the illegal belt=0 branch is covered by
  G0-ghost). train_ppo's training contract contract_revision 3→4 (+ the drink_sovereignty key); future
  non-legacy resumes of v30/v31 revision-3 contracts will be refused (residual registered).
- **E2 training-side knob**: train_ppo gets `--no-drink-sovereignty` (used only by the ctrl leg; not
  passing it = autonomy on). Construction is limited to the training/environment side; no hand edits to
  eval protocol files (the protocol semantics change naturally with the E1 default).
- **E4 documents carried along (registered as is)**: the design memo gets Appendix A (the verdict of the
  3A diagnosis case, a notice item authorised under item 3 of the decision) and its date is corrected
  from 07-13 to 07-15 (an after-the-fact inline correction to a document already notarised at db008d9,
  aligning the timeline with the decision, registered here); the root-cause addendum of FORENSICS #3 and
  the roadmap numbering addendum all go into the freeze commit of this volume.
- **Chain of consequences (registered as is)**: E1 changes the python_protocol/impl bundle sha →
  1. R1's BC receipt becomes invalid with the world → this case's preflight sequence has a built-in
  **bc_worker regeneration** (R1 clause reused: G1 data-side gate + receipt gate + the `--skip-dry`
  consumption chain; manager/flat are not regenerated, since their R1 FAIL verdict belongs to the sealed
  old world and is unaffected); 2. the old R2/v31 archives would fail a full-form re-check under the new
  runtime → this case uses the R2 anchors only through "full-text sha + the anchor-bridge clause" (D3),
  and all pairings are carried by in-case refs (v31 mode).

## D1 world identity (full v31 set carried over + items specific to this case)

W1 freeze notarisation (porcelain exemption lines as in R2) / W2 version / W3 engine REF / W4 diabdat /
W5 impl stage sampling / W6 runtime / W8 idle + driver mutual exclusion / W9 first-burn prerequisites /
W10 resume reconciliation / CASE_RUNTIME five-sha case-level reconciliation (a prerequisite of every exam)
/ reference finality clause: all carried over verbatim from the v31 driver. Specific to this case:

- **W7 artefacts**: king zip/npz (v28-leg1, sha as frozen constants), KING_SD (regenerated in-case by
  export_manager_sd + check_teacher_parity 0/1000, sha into the ledger), M29 npz (science ref), H npz
  (=DEFAULT_MANAGER_SHA256), BC_SD (regenerated in-case; its sha is the value recorded in the ledger).
- **W-G0, three forms (freeze prerequisites of E1; any failure → no freeze, no launch)**:
  - **G0-identity (upgraded to a live bit-level test)**: capture a baseline before the change: king npz
    × H, seeds 7000-7015, per-seed hashes (action sequence, per-step wage, final summary) + real
    per-window / per-episode assertions of the wage identity W≡R−bonus; after the change, replay the same
    runs, **equal bit for bit**. **Baseline notarisation (panel major)**: the baseline is captured with
    the same probe version on clean stashed code and filed as
    [`docs/assets/g0v32_traj_baseline.json`](../assets/g0v32_traj_baseline.json) in the freeze commit;
    its full-text sha is a frozen driver constant, asserted in preflight, with a G0_BASELINE event in the
    ledger (an accidental re-recording mismatches). In addition, the in-case refs of D2 must be **equal
    per seed and per field** to the v31 reference archives of the same name (ref-launch ≡ the full 113.0
    table, ref-science ≡ the full 140.9 table): a case-level proof of 32-seed bit-level G0; any
    inequality is a G0 failure and halts.
  - **G0-ghost (v12 lesion regression)**: 12 is still illegal when belt=0; an active 12 at hp≥0.5 drinks
    normally, with the step accounting / clock / fuse semantics unchanged; at hp<0.5∧belt>0, `_drain`
    still drains every step (the floor does not fail because of the unmask); the termination ladder on a
    death step is unchanged; no drinking after death (the anti-revival ghost). The probe script goes into
    the freeze commit.
  - **G0-ledger**: 3B′ not adopted, exemption registered; the wage-identity assertion is merged into
    G0-identity.

## D2 recipe

**Preflight sequence (strictly serial)**: E1/E2 construction commit → replay and compare against the
G0-identity baseline (captured before the change) → G0-ghost → bc_worker regeneration (R1 gate) →
KING_SD regeneration + parity → two in-case reference runs (below) → sanity gate → training of both legs.

**In-case refs (7000-7031, gate use only)**:
- `v32-ref-launch` = H × king npz: the pairing base; **bit-level assertion ≡ v31-ref-launch (the full
  113.0 table)**.
- `v32-ref-science` = M29 × king npz: the base for the 4C main verdict (deep-dive configuration);
  **bit-level assertion ≡ v31-ref-science (the full 140.9 table, died 3, depth2 15)**.
- Sanity gate: the two bit-level assertions are themselves the sanity gate (replacing the R corridor; the
  corridor clause is kept as a diagnostic reference if a bit-level assertion fails).

**Two legs (same-seed control; the v30 relay lineage verbatim + this case's prescription; the v30 driver
was checked against reality)**:
- Shared: `--worker --algo mppo --gamma 1.0 --max-steps 3000 --n-steps 512 --num-envs 4 --lr 3e-4
  --ent-coef 0.005 --seed 303000 --total-steps 499712 --distill-beta 0.015625 --teacher-sd <BC_SD>
  --teacher-override <KING_SD> --skip-dry --manager-npz <M29 npz> --resume-from <king zip>
  --allow-legacy-resume --calib-probes 3747984,3947984 --calib-record-only`
  (quantum 2048; lr/ent verbatim from run_v30_relay.py:429-437; calib are global-step probes = king
  nt+250k/+450k, the same form as v30 reachable_probes, recorded only; **the manager during training =
  M29**: the learning signal for autonomy is in deep windows, as in the v30 precedent; the king zip has no
  contract, the second registered use of the legacy path; both legs resume directly from the king zip,
  not in a chain).
- **leg-sov**: autonomy on (default). **leg-ctrl**: `--no-drink-sovereignty`. The **only variable between
  the legs = the autonomy knob** (same seed, same budget, same anchor, same β, same manager).
- nt gate: each leg's nt_zip == **3,997,696** (= king's measured 3,497,984 + 499,712; incremental
  semantics); timeout 6 h per leg; export with export_worker_npz (parity built in).

**Exam finality clause (panel blocker fix)**: every eval archive of this case (refs and leg exams) is
frozen at case level once filed and recorded with exam_ok (carrying the archive sha16); a resumed run
accepts it after re-checking the expected identity + the ledger sha (exam_or_adopt), and re-spending via
.void is forbidden; an archive without a ledger entry, or a ledger entry without an archive, halts and
reports.

**Exams**: each leg's s16 (7000-7015) → full32 (H pairing, the launch definition) → **full32-M29**
(fixed, one run per leg: for sov it is the 4C main-verdict reading, for ctrl a fixed control; changed from
the draft's "conditional extra run" to fixed so the verdict is unambiguous, with the budget of +1 probe run
registered as is).
**Exam protocol note**: every exam of both legs runs with autonomy on (the E1 default, the deployment
definition), with no hand edits to eval protocol files; in ctrl's exams a12 may be non-zero (always masked
in training, legal in the exam, and its bit-12 logit was never trained), so the a12 difference in
R32-split is read in the deployment definition.

## D3 decision ladder

The decision lines are derived on site from the mean R of `v32-ref-launch` (under the bit-level assertion
R ≡ 113.0; the fractions are canonical): abandonment gate (75/112.4)×R (triggered verdict = "training
failed **or transfer across managers failed** (training under M29, screening under H); the autonomy
question was not examined"); reproduction floor (85/92)×R; eligibility per leg (died ≤6 ∧ sentinel ∧ not
voided; the dual-attribution clause unchanged); launch line = paired mean difference vs ref-launch ≥+4 ∧
wins ≥18/32 ∧ died ≤6; exhaustive no-launch tiers + width note; multiple-comparison ledger (5th draw on the
18/32 line, **P(≥18|p=.5)≈30%, exactly 0.2983, with the verdict; the ≈43% in the v29-v31 volumes is a
slip for P(≥17): the frozen volumes are not changed, and the correction is noted with this case's
verdict**).
**Limitation: the lines have not been recalibrated (v31 D3-10 verbatim + this case's main-verdict lines
added)**: none of this case's decision lines nor the R32 main-verdict lines (died<3, mean difference ≥−2,
a12≥0.1) have been recalibrated under the autonomy protocol; near-line verdicts (died within 1 of the
line, mean difference within 1.0, wins within 1) must carry the note "near the line + line not
recalibrated"; any verdict decided by the death gate / void line must also cite the depth distribution
and per-seed died; recalibration is a separate case, and adjusting lines within this case is forbidden.
**Eligibility gate note**: autonomous drinking prolongs survival and can mechanically raise window length
and the cap rate; if the sov leg is ineligible only because cap_rate ≥5%, the decision stands, and the
verdict must carry the note "candidate drift in gate semantics" + the τ̄ difference against ref and the
per-window distribution.
**D3-2 winner clause (from v31 + tie-break order)**: winner = the eligible leg with the highest full-32
mean; a difference ≤0.05 → fewer deaths; still tied → take sov (the prescription leg first, registered
here); if the highest mean is blocked by eligibility → substitution (substitution recorded). The floor and
the launch line apply to the winner.

- **R32 main verdict (the scientific main verdict of 4C, independent of launch; always decided for the
  sov leg, win or lose)**: sov×M29 paired per seed against ref-science (140.9, died 3):
  - **"Autonomy turned into survival"**: died < 3 ∧ paired mean difference ≥ −2 (allowing a ±2 noise
    band) ∧ **a12 per episode ≥ 0.1** (a single constant A12_USE_LINE shared with tier 2; discrete meaning
    = at least 4 real drinks in 32 episodes): the success line of the decision is met. 0 < a12 < 0.1 ∧
    died < 3 ∧ mean difference ≥ −2 falls into tier 2 by tier order, and the tier-2 verdict must add "low
    usage appeared together with improved survival; attribution not proven". **Tier 1 has an extra guard
    against retraining effects**: if ctrl×M29 also has died < 3, the causal verb "turned into" is
    forbidden and the wording becomes "survival improved alongside autonomy; retraining effect is a
    candidate";
  - **"Autonomy given but not used"**: a12 per episode < 0.1 ∧ died ≤ 3 ∧ paired mean difference ≥ −2:
    4C is harmless and ineffective; move to the 4B agenda (reopening the deferred item 5 of the decision
    for discussion);
  - **Tier 3 (the complement, which guarantees exhaustiveness)**: every other case triggers the
    decoupling clause and moves to the bundled evaluation case; decision order tier 1 → tier 2 → tier 3.
    **The verdict is worded by sub-case; the tier name alone may not be used**: (a) a12≥0.1 ∧ died≤3 ∧
    mean difference≥−2 → "used, without turning into survival, without degradation" ("harmful" is
    forbidden; a positive mean difference adds "return improved while survival is unproven");
    (b) a12≥0.1 ∧ (died>3 ∨ mean difference<−2) → "used and degraded"; (c) a12<0.1 ∧ (died>3 ∨ mean
    difference<−2) → "degraded at low usage; retraining / protocol side effects are candidates" (stating
    the a12 level).
  - The verdict must carry (a12 per episode, the died pair, the paired mean difference, the per-seed
    depth vector; the R32_MAIN event also carries the sov-m29 and sciref columns). **Three further
    mandatory qualifications**: (i) the binomial note P(died≤2|n=32,p=3/32)≈0.41: the died tiers are a
    directional reading, not evidence of significance (the same discipline as the 18/32 note); (ii) the
    fixed ctrl×M29 control reading alongside; (iii) the set difference of death seeds (the seeds that died
    in ref and were rescued by sov, and the seeds newly dying under sov, listed by name).
- **R32-split (scientific observation of the control leg, recorded, not decided)**: sov−ctrl per-seed
  pairs (full32 H paired difference, died difference, a12 difference). **Reading rules (from R30.6)**:
  the paired win count always goes with it; |mean difference| < 2 reads "direction undetermined"; the
  verdict must note "the same seed only guarantees equivalent initial weights and env streams; the
  autonomy mask changes sampling from the first renormalised distribution, and trajectories fork at once:
  'the only variable' is a statement about configuration, not a trajectory pairing". **Triangle reading**:
  the three pairings sov−ctrl / ctrl−ref-launch / sov−ref-launch are shown in one table; the split
  narrative follows the triangle reconciliation, and no single reading may speak for "the autonomy
  effect". Both legs' step accounting is identical (+499,712 each), so no dual definition is needed.
- **Anchor-bridge clause (prerequisite of the gold P line)**: if launched, the gold evaluation is read
  against the throne's incumbent anchor value 97.7 (r2-throne, **the R2 re-measurement; re-measurement does
  not change the title, and the throne title still belongs to the incumbent assembled agent**) and the
  script reference 94.2 (r2-script). Bridge proofs: 1. the **filed** zero-flip probe
  ([`train/runs/probe-zeroflip/report.json`](../../train/runs/probe-zeroflip/report.json), sha a frozen
  driver constant; **coverage registered: observation corpus = the R1 BC demos, the distribution of
  scripted demonstration windows, neither the throne worker's deployment trajectories nor the gold pool**)
  + **the upgrade: a trajectory-closure probe on the throne worker** (v24-leg7×H, 7000-7015, baseline
  [`docs/assets/g0v32_traj_baseline_throne.json`](../assets/g0v32_traj_baseline_throne.json) in the freeze
  commit, replayed bit for bit); 2. G0-identity bit level (king) + two REF_BITEQ runs; 3. the scripted
  worker is a pure dispatch function: **G0-ghost assertion E, live** (autonomy on/off in script mode,
  equal bit for bit per window over whole episodes + a dynamic proof that dispatch never returns 12).
  **Qualification: the throne anchor 97.7 is a gold-pool reading, while the bridge proofs cover the probe
  pool, so the table verdict must carry "the throne-anchor bridge is a distributional extrapolation".** If
  any bridge proof fails → the gold evaluation reports numbers only without the table, and the title
  verdict is suspended and reported.
  **Title transfer clause (as in the written version of v31 pre-registration D3-6, adapted to the worker
  side)**: (a) succession criterion: the winning leg passes the launch line ∧ the gold evaluation is not in
  a regression tier (gold ≥ 94.2 ∧ gold died ≤6) → the incumbent assembled agent changes: new incumbent =
  v22-H × the winning worker; the v3 launch anchor moves with it := the v32-golden archive; r2-launch
  133.9 is downgraded to a historical reading of the predecessor, and the verdict appendix notes the end
  of its v3-launch-anchor role. (b) Gold evaluation in a regression tier → succession failed; the
  incumbent and the launch anchor stay, and the launch itself is recorded as usual. (c) The P line moves
  only the throne title; succession without a change of throne is a legal combination, and the verdict
  lists both assembled agents' gold-pool readings. (d) No launch → neither title moves, and the verdict may
  use no title verb other than "re-elected".
Gold-pool opening history: the gold evaluation was actually opened three times (v22/v23/v24); the five R2
anchoring runs were measurement exposure; v31 did not open it. If this case launches, it is the 4th real
gold opening and the 9th cumulative gold-pool exposure, and the fixed-pool bias is noted in the verdict.

## D4 P line

**P1 leg quota (written specifically for this case)**: each leg has a launch quota of 2, counted by
leg_start events in the ledger; if the first launch misses its target (rc≠0 or nt≠3,997,696), it halts
and reports as an operational failure, and after a restart a second launch is allowed
(train_ppo._prepare_run_dir automatically archives the previous run's leftover artefacts, and the resume
source is always the king zip, which does not conflict with "no additional retraining": retraining means
a new question, a relaunch continues the same question); quota exhausted → OPERATIONAL_FAILURE halt, and
editing the ledger by hand to extend it is forbidden. An npz export failure is an operational failure;
after a restart it goes through the re-export path (no leg_start recorded, no quota used); a relaunch
applies only to a training run that missed its target (rc≠0 or nt≠target).

The full v31 set is carried over (P1 ledger-based quota / P2 closed enumeration / P3 lock and idleness /
P4 preflight / P5 case-A halt / P6 chained re-freeze / exit-code table) + specific to this case: BC
regeneration R-W FAIL → the whole case halts (R1 clause; a missing teacher is season-level); any G0 form
fails → no freeze, no launch (the E1 construction is reverted with a correction commit; the decision stays
valid, redesign and try again); a refs bit-level assertion fails → CASE_HALT_G0 (exit code 7), and
downgrading to a pass because "the new reading is also reasonable" is forbidden.

## D5 ledger and R lines

The case ledger (not published) uses the v31 event vocabulary + the events G0_BASELINE / G0_REPLAY /
G0_GHOST / BC_REGEN / REF_BITEQ / E1_SCOPE_NOTE; REF_BITEQ is only for the PASS state, and a failure
records CASE_HALT_G0{via:REF_BITEQ}, exit code 7; at most one exam_ok per tag, and resumed acceptance
records exam_adopted without re-recording. The verdict appendix is written at the end of this document.

**R lines**: R32.1 both refs bit-level ≡ v31 (a point, not a band); R32.2 sov full32 ∈ [0.85R, 1.25R],
point 1.05R; R32.3 sov a12 per episode ∈[0, 8], point 1.5; R32.4 sov×M29 died ∈[0, 6], point 2 (a shift
down from ref-science's 3); R32.5 sov−ctrl paired mean difference ∈[−10,+15], point +3; R32.6 gold (if
launched) ∈[95,140]; R32.7 sov full32 a13 per episode minus ref-launch ∈[−2,+4], point +1 (belt-economy
gauge); R32.8 sov×M29 paired mean difference against ref-science (the main-verdict quantity) ∈[−15,+15],
point +2; R32.9 ctrl×M29 died ∈[0,8], point 5 (expected retraining side effect, the same source as v30's
deep deaths 3→7). **R32 gauges (belt economy, recorded, not decided)**: every full32/m29 exam_ok event
also carries a13 per episode and R windows per episode (RESUPPLY pressure), next to the same fields of the
refs: the "potion-logistics inflation" risk of the design memo turned into registered measurements.

**Residual uncertainty**: 1. same-seed legs remove only one source of randomness (the env stream); SGD /
rollout noise remains, and the split reading is a single pair; 2. the learning signal for a12 relies
entirely on compounding survival (no shaping), and 499,712 steps may not be enough: the "not used" tier is
a real possibility, and the 4B transfer path is in place; 3. the M29 paired exam adds one run on the sov
leg, and the family-wise error rate accumulates in the multiple-comparison ledger; 4. the regenerated BC
artefact differs in bytes from the R1 artefact (the impl sha changed), with the same G1 gate
specification: teacher generations are accounted separately; 5. the bit-level assertions depend on zero
rebuilds of engine and bridge (guarded by CASE_RUNTIME); a breach is a G0 failure, not a scientific
conclusion; 6. after renormalisation by the autonomy mask, the sov leg's distillation teacher distribution
contains the mass of the untrained bit-12 logit (limited in size at β=0.015625): a mechanical consequence
of the knob rather than an extra degree of freedom, registered as is; 7. **manager mismatch between
training and deployment (new in this case)**: both legs train under M29, while the launch pairing and the
gold evaluation are under H. In the v30 precedent, training, exams and gold evaluation were all under M29,
so it does not cover transfer across managers; deep-window exposure is low under H (ref depth2 7 vs 15),
so the deployed expression of a12 is compressed, and any citation of the verdict must show a12/died for
both sov full32 (H) and sov-m29 (M29); 8. G0 coverage: identity / REF_BITEQ only prove that the
deployment stack's trajectories are inert; the training distribution (exploratory rollouts) is not
covered; old-protocol equivalence of the ctrl leg rests on code-path identity + mask assertions, not on a
bit-level training replay; step-level bit identity on death steps was not sampled directly within the 16
seeds (the 16 baseline seeds have zero deaths, and bit-level coverage of the death path is carried by the
died seed rows of REF_BITEQ); 9. contract revision 3→4 means future non-legacy resumes of v30/v31
checkpoints will be refused: a known cost in the conservative direction; 10. the winner is a max-of-2
order statistic, which inflates the type I error of the launch line (v29 residual #6 carried over);
11. the dry-level anchor sentinel (sentinel.jsonl) keeps the old v28-v30 mask definition (11/12 always
masked) so that the cross-leg mismatch telemetry keeps the same scale: recorded, not decided; autonomy
behaviour is measured separately by the a12/a13/R-window gauges of the evaluation; this definitional choice
is registered here.

---

## Verdict appendix (2026-07-16, recorded by the operator; the commit is the notarisation)

### Operations incident (recorded before the verdict)

In the first launch on 2026-07-15, the sov leg was killed silently from outside after 44 minutes (the
process was tied to a host session that was replaced; last telemetry at step 3,984,272/3,997,696, no
leg_done, .run.lock not released, and the driver disappeared without an exception event). The aborted
directory was sealed as `train/runs/v32-sov-abort1` (not published), and an OPS_INCIDENT was recorded in the ledger.
Second launch on 2026-07-16: the launcher was hardened to run detached directly under launchd (PPID=1)
with a caffeinate assertion; the ledger-based launch quota stood at sov 2/2 and ctrl 1/2. Monitoring
lesson: the gate ledger is silent during a leg by design, so leg health must be checked through the
mtime of the progress heartbeat; a monitoring report with zero findings is not evidence of health (the
standing review rule of the [roadmap](../design/ROADMAP-course-plan.md) applies). After the second launch,
the W10 five-sha preflight reconciliation passed, the G0 replay / ghost checks turned green again under the
"re-verify at every launch" clause, and the references were carried over through exam_adopted + REF_BITEQ
(equal×2): the bit-level world did not move.

### Campaign record

sov leg (45.3 min, rc=0, nt_zip=3,997,696, npz 04636e1a…f94f59); ctrl leg (43.2 min, rc=0, same step
count, npz e26cda87…087a13). Six exams archived: sov-s16 86.4/0, ctrl-s16 104.0/0, sov-full32 79.2/died2,
ctrl-full32 92.3/died2, sov-m29 108.3/died8, ctrl-m29 125.2/died8 (sha values in the ledger's exam_ok
rows; the exam finality clause applies).

### Main verdict: tier 3, moved to the bundled evaluation case

R32 main verdict (ledger text): **tier 3 (degraded at low usage (retraining / protocol side effects are
candidates, a12=0.0)): a12 per episode 0.0, died 8 (line 3), mean difference −32.59 (band −2.0); the
decoupling clause is triggered and the case moves to the bundled evaluation case (as the decision
provides for failure)**. Launch check: the winner ctrl 92.3 < floor 104.4 (=85/92×113.0), **no launch;
the succession question was not examined**; the gold evaluation was not opened, and r2-launch 133.9 stays
in place. The binomial note and the death-seed sets (new_deaths 7005/7020/7024/7026/7028, rescued empty),
the depth vectors and belt_econ (sov_m29 a13 242.38 vs sciref 374.03) are all in the R32_MAIN event row.

### Three scientific conclusions

**F1 autonomy never exercised (hard result)**: a12 per episode = 0.0, for sov under both H and M29. The
zero-flip probe's prediction (0/166,383) is reproduced in vivo: unmasking alone does not change the
policy's ranking, and the belt precondition gate was never even reached. Together with the 3A verdict of
teacher priority starvation: **plain unmasking is ineffective, and 4B (positive-sample injection / a
shaping review) or a 3D-level intervention is a prerequisite for reviving a12**. The intent of tier 2's
"move to 4B" path is carried into the bundled case by tier 3, on the same grounds.
**F2 regression that comes with retraining (the case's main unexpected finding)**: the ctrl leg is the
old-protocol-equivalent recipe, and it is still −20.74 against ref (full32 92.3 vs 113.0; m29 125.2 vs
140.9, died 8 vs 3). That is: **continuing training for 500k steps on the throne checkpoint with the v30
leg recipe in the new world (the impl after the R1 re-anchoring) by itself causes a slide of about −20
points from the incumbent level**. Candidate attributions: inflation of deep deaths (the upper edge of the
R32.9 expectation, following the v30 precedent), the change of teacher generation (distillation from the
new BC artefact), or king sitting on a local peak for this recipe. This finding feeds directly into the
bundled evaluation case and the dead-gate recalibration case.
**F3 the directional reading for autonomy is negative, not a conclusion**: sov−ctrl paired −13.07 (wins
17/32, noise band), with 5 extra death seeds for sov; a13 sov_m29 242 vs ctrl_m29 201 (more belt use).
Under the R32-split definition clause, the same seed only guarantees initial equivalence, and no single
reading may speak for the autonomy effect; the autonomy-specific component waits for the bundled case's
design to separate it.

### R-line scorecard (prediction vs measurement)

R32.1 bit-level carry-over: **hit**; R32.2 sov full32 79.2 against band [96.1,141.3]: **out of band
(low)**; R32.3 a12 0.0 against band [0,8]: **in band (degenerate lower edge; residual 2's "not used" tier
came true)**; R32.4 sov-m29 died 8 against band [0,6]: **out of band (high)**; R32.5 split −13.07 against
band [−10,+15]: **out of band (low)**; R32.6 no launch, N/A; R32.7 the ref rows have no a13 field (a
structural gap from carrying over the v31-generation archives bit for bit): **cannot be decided, registered
as is**; R32.8 main-verdict quantity −32.59 against band [−15,+15]: **out of band (low)**; R32.9 ctrl died 8
against band [0,8]: **in band (upper edge; the expected retraining side effect materialised)**. Four lines
out of band, mainly because F2 was not in the prior model: the point estimates at pre-registration time
assumed by default that "the leg recipe ≈ reproduces the throne", and that default was falsified. This is
the case's biggest lesson about the procedure itself.

### Path decision and open items

1. Under the decoupling clause of the decision, the **4 × 2 bundled evaluation case** is eligible to start:
   it gets its own pre-registration with a critic panel first, and its launch needs its own approval
   (suggested: review it together with F2, with the ctrl arm made standard);
2. one more piece of evidence for the dead-gate recalibration case (ctrl died 8 / line 3);
3. the gold seeds were not burned; the two open items (gold-evaluation notice and ratification of the belt
   gate) stay open (the belt gate was never reached by behaviour in this case, so ratification has no
   material effect);
4. both legs' npz/zip and the six exam archives are frozen under the exam finality clause; any later case
   that cites them goes through resume adoption and does not burn them again.

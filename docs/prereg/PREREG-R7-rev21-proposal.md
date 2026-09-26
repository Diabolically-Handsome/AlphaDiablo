# Proposal: R7 campaign revision (rev20 → rev21)

Status: **draft awaiting approval**. This document is the implementation proposal for the ruling of the F3 final review
(docs/forensics/FORENSICS-F3-why-no-progress.md); it has not been approved and **constitutes no launch authorization**;
no code was changed, and run_r7_combat_recovery.py (rev20) and r7_statistics.py are untouched.
Drafted on 2026-07-27.

## Background

F3 final ruling: the only possible real sign of progress in the whole R6 case is **per-step combat efficiency** (step definition ~+6 wage per seed,
same sign in both pools, 64 pairs combined t=+2.26, bootstrap CI95=[+0.48,+6.22]; but the per-pool sign tests are not significant,
p=0.11~0.19, so it is a hypothesis to test, not an established fact). The four current R7 gate metrics (farm_worker_wage/
farm_worker_kills/ret/kills) are all **totals**, and each mixes an efficiency component with a survival-time
component -- and F3 has shown that the survival-time component is a symmetric seed lottery (death flips 6/14 favourable, p=0.79,
micro_steps differences +151/−152 mirrored across the two pools). If the R7 final exam again shows "totals back to zero", the current accounting
cannot answer "is the efficiency residual real" -- which is exactly the knowledge most worth bringing back after burning a virgin pool of 256 pairs.

## Revision 1 (new): pre-registered per-step efficiency scoring limb (record only)

- **Metric definition (pinned)**: per seed `rate(s) := farm_worker_wage(s) / micro_steps(s)`
  (direct division of schema-5 row fields, no new machinery); paired difference Δrate(s) = rate_cand(s) − rate_base(s).
- **Recording**: the development decision and the final analysis each carry a RATE_REPORT (mean/median/sign count
  /exact binomial p/deleveraged mean, statistical discipline per B1 D3 verbatim).
- **Nature**: **a scoring limb, record only** -- not in METRIC_RULES, uses no familywise α,
  and is not an input to any pass/fail. Reasons: (1) keep the familywise guarantee frozen in rev20 intact;
  (2) rate alone can be Goodharted by an "efficient early death" policy (the death non-inferiority gate is alongside, but making the scoring limb its own gate
  would need a separate case); (3) the mission of this limb is to **measure** the efficiency hypothesis, not to decide on candidates.
- **Pre-registered reference band (for reading, not a gate)**: point +0.004 per micro-step (≈ R6 measured +0.0041/+0.0048 in the two pools),
  band [0, +0.010]; out of band on either side is a registered finding. With 256 pairs the SEM of this limb is small enough to tell 0 from the point estimate
  (with 32 pairs it is not -- the F3 noise-floor review).
- **Definition rule (mandatory with every citation)**: an efficiency/exposure decomposition must state **whether it uses the window definition or the step definition**
  (for the same net difference the two differ by 3.3×, per the F3 internal-consistency review ruling); this limb always uses the step definition;
  the window definition (wage/farm_fresh_n) may only be a secondary diagnostic with a window-length confound note.

## Revision 2 (confirmation, no change needed): the seed-majority clause is already built in

F3 had suggested "keep the seed-majority clause". Check result: r7_statistics.MetricRule defaults to
`require_sign_test=True`, and all four rev20 gate metrics already carry an exact sign test -- **this is already satisfied,
zero changes**; recorded here so a later case does not raise it again. (Note: with 256 pairs the sign test needs a win rate ≥0.578,
which, from the de-meaned empirical distribution of the two R6 pools, corresponds to a shift of ≈+5.3 wage -- indeed more sensitive to small effects than the mean gate.)

## Revision 3 (new, lightweight): survival decomposition diagnostic with the final analysis

Besides RATE_REPORT, the final analysis records one symmetric rate×time decomposition (total paired difference = efficiency component
+ time component, with the per-seed list of death flips). No criterion, pure recording -- if the final exam totals go back to zero, this diagnostic
immediately answers "did the efficiency vanish, or is this the losing side of the lottery", without another forensics team.

## Implementation (if approved)

All changes are limited to the analysis/report sections of run_r7_combat_recovery.py (derived-metric computation and recording) +
matching unit tests; environment/evaluation-protocol files (python/diablogym/*, eval_assembled.py,
eval_contract.py) are **untouched**, and the eval archive schema does not change (rate is a direct division of row fields and is not stored in archives).
campaign_revision 20→21, following the rev20 CAMPAIGN_REVISION constant and the state-schema version convention.

## Relation to launch

This document does not change the R7 launch preconditions (engine available + recorded approval). Launch environment: WSL2 (from 2026-07-27). This runtime is a new numeric world: Mac-era archives (REF_BITEQ 113.0/140.9) are not expected to re-verify bit for bit; baselines are re-evaluated on the same runtime and never paired with Mac archives.


## Amendment 2: BC-v1 demos-only gate (A2, approved 2026-07-27)

Approved as plan A2, with the model-upgrade route as the fallback.

- **Origin**: R7 prepare-bc failed twice -- pool 2_102 was burned by a crash from a missing a14 fuse receipt
  (fixed); pool 2_104 was burned by a failed candidate quality gate (top1 0.833 vs gate 0.95, rare-key
  recall 0.34/0.46 vs 0.85; class-weight retries did not help). A zero-pool cost-ceiling measurement (burned-2_104
  demos, 200 epochs + class weights): top1≈0.88-0.90, rare keys ≤0.43 -- the gate is structurally unreachable under the unfiltered v3
  view of 07-25 (the Mac-era 1.0 was a product of the filtered view). Moreover, the R7 training command does not consume
  the BC policy (teacher=KING_SD, no --teacher-sd/--bc-aux-demos), so the 0.95 gate protects a relic of the
  R5/R6 era; what R7 actually consumes = demos (dry-anchor anchor + identity chain).
- **Changes**: (1) the bc_worker v1 candidate/final quality lines become record-only (fields archived as is);
  the release gate = demos validity (exact pool coverage / game discipline / a14 coverage / hard assertion of zero label conflicts,
  new _require_zero_exact_label_conflicts); (2) the two lines 0.95/0.85 in the data_gate branch of train_ppo._validate_bc_report
  become [0,1] range checks (identity chain / marker / bit-level recomputation consistency unchanged); (3) the v1 registration range advances
  to 2_106_000..127 (2_104 already burned, append-only).
- **Pool-economy warning**: only two runs, 2_106/2_108, remain in the even v1 range before it hits the 2_110_000 evaluation bank;
  if this attempt fails, the approved fallback, the model-upgrade route (plan C), is taken, with no more pools burned on trial and error.

## Amendment 3: A3 invariant correction + 2_140 legislation block (approved 2026-07-27)

- **Origin**: the first leg crashed at step 119,776 on the leashed_ppo termination-exclusivity ledger invariant --
  the fast-forward chain bridged "worker-window no-progress timeout penalty (-8/-32)" and "the whole-game 3000-step budget running out in an unsettled state"
  into the same transition; the accounting balanced (reward=wage+timeout ✓, the three death fields 0 ✓)
  but the checker was too strict. An isolation diagnostic tree reproduced it deterministically at the same step, with the full accounts of the offending frame on record.
- **A3 change**: that invariant has one limb that allows the legitimate co-occurrence "unsettled_budget_terminal=True, closed consistently with
  budget_exhausted=True"; the other eight limbs are unchanged.
- **Legislation block**: leashed_ppo is part of the implementation fingerprint bundle, so the change voids the BC artifacts of 2_108/2_103;
  the registry legislates, append-only, a BC block 2_140_000-2_158_xxx (avoiding the evaluation bank),
  with active pair v1=2_140_000..127 / v2=2_141_000..383, extended as each pair is consumed.
  Final account of pools burned that day: v1 2_102 (crash) / 2_104 (gate failure) / 2_106 (second hole) / 2_108 (voided by A3);
  v2 2_103 (voided by A3).


## Amendment 4: A4 reconciles the optimizer-step floor with target_kl early stopping (2026-07-27, covered by the A2/A3 approvals)

- **Origin**: leg 3 crashed at step 182,248 on the hard floor "≥8 actor optimizer steps per joint rollout"
  -- the frozen recipe itself has target_kl=0.01 (KL early stopping), and a rollout with a KL spike stopped early at step
  7 and was killed by the floor. The floor (a full epoch) contradicts the recipe's own knob, so it is judged a bug.
- **A4 changes**: (1) an early-stopped rollout records a kl_early_stopped flag, and the producer/validator floor exempts
  it down to ≥1 (liveness guarantee kept), while non-early-stopped rollouts still need ≥8; (2) the qualifies formula is unchanged (an early-stopped short
  rollout is recorded as False as is); (3) the R7 aggregate floor is converted by the flag; (4) audit schema /9→/10.
- **Registered ranges**: 2_140/2_141 are voided by the bundle change; the table extends to 2_142/2_143 (inside the legislation block).

## Amendment 5 (2026-07-28, draft awaiting approval): power recalibration of the death non-inferiority gate

### Facts (development-stage decision, 12 frozen analyses on record)
The R7 development stage ran 14 evaluations to completion with zero faults, decision DEVELOPMENT_SCIENTIFIC_FAIL;
the final exam had not been opened. Item by item:

- **Primary endpoint farm_worker_wage: 12/12 legs pass**. Improvements +14.5~+41.7,
  all LCB >0 after family-wise correction (strongest leg b-risk64-s2130100: +41.7, LCB +29.3).
  This is the first out-of-sample positive signal since v28 that replicates across 6 training RNGs × 2 pools.
- **First showing of the rev21 per-step efficiency limb (record_only): 12/12 legs significantly positive**.
  Deleveraged means +0.0089~+0.0209 wage per step, sign-test p down to 2.2e-14.
  This confirms that the efficiency residual F3 suspected is real and does not depend on survival-time leverage.
- **Observed deaths: all six risk64 legs are below the baseline** (−2.3~−5.5pp, mean −3.1pp);
  risk32 is higher in 3/6 legs (its s2130200 replicate is consistently the weakest).
- **The only item failing everywhere: deaths.noninferiority_upper_bound (12/12)**.

### Diagnosis: a pre-registered design defect (item 9, exposed by the campaign itself)
With α=0.005 (family-wise split) and ~30/128 discordant pairs, the half-width of the death non-inferiority CI is ≈0.110;
a margin of 0.05 means a candidate must be **observed ≥6pp safer** to show "not worse than 5pp" --
the bar was mathematically unreachable before any candidate arrived. The final-exam layer (n=256, margin 0.025, half-width
≈0.078, needs observed ≤−5.3pp) has the same problem. The gate was in effect a demand for being significantly safer, not for non-inferiority.

### Proposed amendment (three options, for decision)
- **Plan B (recommended)**: the development death gate becomes observed_not_higher (per-leg point estimate not higher,
  ≥2/3 seeds across both pools). On the frozen data: risk64 3/3 pass, risk32 1/3 fail →
  risk64 is the only selection (consistent with the wage/efficiency limbs). Non-inferiority inference moves to the final exam, with final margin
  0.05 (n=256 half-width 0.078, needs observed ≤−2.8pp; risk64 development mean −3.1pp,
  so it can be expected to pass, and fails honestly if the true difference is worse -- the final exam still has real decision power).
- **Plan A**: two-level margins converted by power (development 0.125 / final 0.05), structure unchanged.
  On the frozen development data risk64 still passes wage 12/12 but its NI upper bounds are 0.106~0.147;
  under 0.125, 4/6 legs pass → 2/3 met, and risk64 is again selected.
- **Plan C**: declare R7 dead and switch to the model-upgrade route (earlier this was the contingency for a BC gate failure;
  with wage now passing 12/12, not recommended).

### Honesty statement
- This amendment is proposed **after** seeing the development data; any re-decision based on the already-read 2_110/2_111 pools
  is labelled post-hoc; its justification is that the power defect **existed before the data**
  (pure α/n/margin algebra; the table above can be re-checked offline).
- The final-exam pool 2_120_000-2_120_256 is still unconsumed and unread, so the final decision is not contaminated.
- Implementation must touch run_r7_combat_recovery.py (gate logic + margin constant) → the launcher sha changes
  → the six legs' training receipts are voided. Two implementation paths to choose from at approval:
  (1) full retraining and re-evaluation (clean, but ~6h and burns two more 128 pools);
  (2) adoption: a separate post-hoc analysis file consumes the frozen archives to make the selection, and production/final
  reference the frozen sha from a new campaign file (no retraining, labelled as adoption).

### Amendment 5 approval and implementation record (2026-07-28)
- **Approval**: approved on 2026-07-28 -- plan B, 256 games, with adoption
  (straight to the final exam, no retraining of the development legs). Resource allocation: 24 cores reserved for training.
- **Implementation**: rev22. (1) FINAL_DEATH_MARGIN 0.025→0.05; (2) a new command
  adopt-development: checks the rev21 DEVELOPMENT_SCIENTIFIC_FAIL final state + a byte-exact inventory of all
  frozen artifacts (12 analyses + decision + 14×3 eval files + 6×2 training files),
  re-derives the selection under gate B (the pre-registered check set minus the single item deaths.noninferiority_upper_bound),
  writes amendment5-adoption.json (post_hoc: true) and then migrates the state atomically;
  (3) _validate_development_decision is rerouted in the adopted state: per-analysis recomputation is replaced by a byte-exact
  frozen re-check + an equality check of the gate-B re-derivation (used throughout production/final).
- **Correction of the exact reading**: under gate B risk64 is 2/3 (dev-a s2130200 has a flank failure on
  kills/ret exact_sign, not 3/3; the single death item observed_not_higher
  is still 6/6), and risk32 is 1/3; risk64 is selected, consistent with the draft's conclusion.
- **Old identity snapshot**: rev21 launcher c28623a0…, recipe af939783… (recorded in full in the pre section of the adoption document).

### Amendment 5-2: power correction of the final-exam margin (2026-07-28, adversarial review item 2, blocker)
- **Erratum**: the plan B draft argued with a McNemar-type half-width (z·√60/256≈0.078) that a 0.05 margin
  "can be expected to pass" -- but the statistic actually implemented is the bonferroni-one-sided-clopper-pearson
  risk difference (a joint bound spending α=0.005 on each of the two mutually exclusive death proportions), which is far more conservative. Recomputed with the repo's
  own CP function: with the true effect = the pooled risk64 development observation (−3.1pp), the pass rates at margins 0.025/0.05
  are only ~3%/~16%; scaling the discordance structure of the six risk64 legs ×2 to n=256, all fail
  at 0.05. On the final-exam gate the draft reproduced the very item-9 defect it had diagnosed.
- **Re-approval**: after the power table was presented (0.05→16%, 0.07→38%, 0.08→52%, 0.09→65%,
  0.10→77%, 0.11→86%), **0.10** (the recommended option) was chosen.
  Certification meaning: "extra deaths ≤10pp"; the observed death rate and the NI upper bound are published in full and not hidden behind the gate width.
- **Review fix bundle landed with the same approval** (all before the adoption ran): (1) anchoring of the decision moment
  (mandatory comparison of decision.analysis_sha256s + attestation chain check + pinned set of pre-registered
  check keys); (2) sealing guard for the adopted state (eval/train-development refuse re-entry,
  before any phase write); (3) an idempotent repair branch for the two-write crash window of the adoption; (4) fallback for the reroute key
  AMENDMENT5_PATH; (5) seven fixture-campaign tests (adoption / tamper blocking / crash window / sealing).

## Amendment 6 (2026-07-28, approved: switch to the reserve pool): final-exam incident, pool switch
- **Incident**: the final-exam baseline evaluation (pool 2_120) was interrupted at 214/256 by a machine reboot -- the launch
  marker had been written but the archive was not committed, and the protocol, as pre-registered, refused a retry of that run (the per-game
  numbers of the 214 partial games were already in the log, a partial observation). The candidate evaluation was never launched; the candidate model, the 0.10 margin and
  all criteria were frozen and committed before the incident, so no choice was made after the incident.
- **Ruling**: pool 2_120 is abandoned (permanently registered in the final registry, not deleted), and the next
  reserved range **2_121_000-2_121_256** is used (fresh and unread; MDE unchanged).
- **Implementation**: rev23. The adopt-final-incident command: (1) verifies the rev22 incident state
  (amendment 5 chain, production legs frozen, eval_final stopped at opened, sha of the burn marker, candidate
  never launched, sha of the incident log copy); (2) writes amendment6-final-incident.json
  (post_hoc, the whole incident + old and new identities + frozen sha of the production artifacts); (3) the burn marker and the opened document of the burned pool
  are archived into final-incident-20260728-reboot/; (4) state migration.
  Re-checking the production receipts is rerouted to frozen sha (the receipts are bound to the rev22 identity); the amendment 5 document check adds a chained
  identity; train-production is sealed; the crash window resumes idempotently; five fixture tests.
- **Old identity snapshot**: rev22 launcher 01a3e212…, recipe 9f842ca5…; burn
  marker 34f93496…; partial log 05edbb26… (last seed 2120213).

## R7 final-exam verdict and case closure (2026-07-28, approved: accept the verdict and start R8)
- **Verdict: FAIL (final state FINAL_SCIENTIFIC_FAIL)**. On the fresh 2_121 pool of 256 pairs:
  farm_worker_wage 46.4→84.8 (+38.41, family-wise LCB +29.80, pass);
  farm_worker_kills +16.50 (LCB +12.90, pass); kills +14.32 (pass);
  ret +31.28 (LCB +21.83, pass); gear gate passes; rev21 per-step efficiency limb +0.01867 per step,
  sign test p=1.51e-18, 197 wins / 59 losses; deaths 196/256 = 196/256, exactly equal,
  observed_not_higher passes, flips 31↔31 symmetric.
  **The only failure: deaths.noninferiority_upper_bound = 0.1086 > margin 0.10**
  (with 62 discordant pairs the CP-Bonferroni certification width is 10.86pp; at a true null effect this margin
  would pass only ~50% of the time).
- **Scientific conclusion**: the question "why the model does not improve" is formally overturned -- the improvement is solid out-of-sample evidence,
  and it is real per-step efficiency; the formal FAIL is a certification-width gap of 0.86pp, not any observed signal of harm.
- **Closure**: the verdict was accepted; the model is not released; the archives are frozen; the R8 certification campaign is opened
  (24 cores in parallel + the death yardstick recalibrated to the measured discordance density + a fresh pool).

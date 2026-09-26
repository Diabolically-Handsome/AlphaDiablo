# R12 pre-registration draft v0.1: in-place worker re-education (route A)
Status: DRAFT, under review (2026-08-30).

## 1. Rationale and argument (chain of evidence)
1. R9: the manager-side tools (retraining / curriculum) are exhausted, and both arms collapsed to constant
   FARM: depth is not a manager problem;
2. R10: economy v2 was legislated, and the manager dived to L6 at once, at the price of a 97% death rate
   (an unequipped speed run);
3. R11, two trials: any forfeiting bonus lock (K=1/3) sends the manager back to constant FARM; there is no
   stable intermediate state between the two attractors;
4. Trial 3: doubling training did not move the constant-DIVE attractor by a single bit (the "not taught
   enough" hypothesis is dead);
5. Structural mechanism: under this structure a three-action manager always collapses to the best constant
   command, because the frozen worker cannot survive at depth, state-dependent switching beats the best
   constant only by a thin +5.6 margin, and PPO has no incentive to learn switching.
Conclusion: the constraint is in the worker. The teaching order must be: first teach the worker to survive
at depth; only then does switching have value and the manager has something to learn (manager retraining
= R13+).

## 2. Proposition
Re-educate the certified worker in place (PPO fine-tuning) under the economy v2 wage so that it gains the
ability to survive at depth without forgetting its shallow combat strength.

## 3. Design
### 3.1 The training itself
- Mechanism: in-place training in WorkerWindowEnv (an existing mechanism, the birthplace of the R8
  certified worker); a frozen manager drives window selection, and the worker learns step by step inside
  its windows;
- Starting point: fine-tune the certified worker zip (sha 2837288d), keeping 20 generations of combat
  heritage, with an anti-forgetting gate as the backstop (see 4.3). A brand-new worker is not chosen (it
  would throw away all existing capability);
- No change to the observation/action contract: the dual-v4-asymmetric-v3 view, 15 actions,
  drink_sovereignty=False (the reflex backstop as before), to minimise the blast radius; potion autonomy
  (the 4B pair) is explicitly excluded and listed for R13;
- Economy: the full v2 set applies to the worker's wage: B depth multiplier (more money for fighting
  deep), C the revised decreasing death penalty (less fear of dying deep), E1 gear repricing (a14 is worth
  learning for the first time), D anti-idling. The worker wage-stripping clause (the descent bonus is
  stripped) stays: the descent decision and its bonus belong to the manager, and the worker earns only
  from survival and combat.

### 3.2 Two-arm design (coaching)
- Arm 1 "regular coach": manager = M29 (the frozen production artefact). Window composition is mature and
  stable, with little deep exposure: the conservative control;
- Arm 2 "devil coach": manager = r10-econ-mgr (the constant-DIVE speed runner). The worker is dragged
  deep continuously: maximum deep exposure, with the risks of shallow-skill starvation and learning
  collapse;
- Both arms share the starting point, economy, step count and seed discipline. The R9 two-arm machinery
  is reused.

### 3.3 Training parameters (★ = calibrated in G0)
- num-envs 32; n-steps 64; seed 22; lr 3e-4; ent 0.02 (the production recipe);
- total-steps ★ (worker steps are micro-steps, with different throughput from manager training; frozen
  after a 30-minute G0 smoke-test calibration; draft anchor 262,144);
- Fuses: a 24 h anti-zombie floor per training arm (R9 / trial-3 precedent); 2 h per evaluation.

## 4. Exams and criteria
### 4.1 Pools
- Paired baselines (already exist, no new consumption): four v2 exam sheets on 2_114/2_115, M29 × old
  worker (the anchor) and speed-runner × old worker (the R10 candidate sheets);
- Candidate exams: same pools and seeds, four new sheets per arm, M29 × new worker and speed-runner × new
  worker;
- Confirmation pool 2_116: spent only after all gates pass, on a manual command. The fresh pools
  2_117-119/2_126-129 are not touched.

### 4.2 Main gates (decided independently per arm; any arm passing all of them is a winner)
- Gate A, superior survival at depth: speed-runner × new worker vs speed-runner × old worker, paired
  reduction in deaths (McNemar UCB < 0; absolute target: died ≤ 110/128);
- Gate B, depth retention: speed-runner × new worker vs speed-runner × old worker, paired depth
  LCB > -0.10 (surviving must not come at the price of not diving; if depth rises, it is recorded as extra
  merit);
- Gate C, anti-forgetting (shallow non-inferiority): M29 × new worker vs M29 × old worker, ret_mean ≥
  0.95× and kills_mean ≥ 0.95×;
- Gate D, behavioural identity pre-check: argmax agreement between the new and old workers < 99% (the new
  R9 rule, run before the exam).

### 4.3 Recorded items (not gates)
- a14 gear: the opportunity / request / success triple (direct evidence of E1 taking effect);
- Survival curves by depth stratum (per-level death rates on L3/L4/L5);
- Change in reflex drinking frequency; distribution of micro-step length per episode.

## 5. Infrastructure and rules
- 32 environments as before; asynchronous collection stays an R13+ infrastructure item (not touched here);
- The worker product is a new lineage (a rev26 fine-tuned descendant) and is only an experimental artefact
  within this case; any release goes through the certification process separately (the full R8 gate set),
  and this case includes no release;
- NEW FILES ONLY; the embargoed seed ranges are not touched; launch once the ledger and the launch order
  are in place.

## 6. Risks and contingencies
- Risk 1, catastrophic forgetting: gate C is the backstop; if both arms fail gate C, rerun with lr lowered
  to 1e-4 (the only parameter remedy the pre-registration allows, to be recorded in the ledger);
- Risk 2, learning collapse in the devil arm (all deep windows, shallow skills starve): arm 1 is the
  fallback;
- Risk 3, unknown worker throughput: freeze the step count only after G0 calibration, to avoid an R10-style
  schedule misjudgement;
- Risk 4, once the new worker gives switching value, the manager is still a constant policy: expected, and
  exactly the evidence for opening R13 (manager retraining), not a failure of this case.

## 7. Process
G0 throughput calibration (30 minutes) → this case's parameters settled → final review → freeze (sha) →
launch order → arm 1 and arm 2 launched in sequence → eight exam sheets → paired verdict → final
decision.

## 8. Four decision points for the review
1. Approve the two-arm coaching design (regular + devil), or cut to one arm to save time?
2. Postpone potion autonomy to R13?
3. Fine-tuning start (not a brand-new worker) with the anti-forgetting gate as backstop?
4. Pool plan (reuse the 2_114/115 pairing, keep 2_116 for confirmation)?

---
## Revision v0.2 (2026-08-30, recipe finalised after all four points were approved)

Decision: both arms approved (a control arm is needed for the comparison to say anything); points 2-4
approved as proposed.

### Recipe base: the R8 certified worker's birth command (the release receipt text) inherited verbatim, with three kinds of change:
1. [Replace] --resume-from → the certified worker zip (sha 2837288d, in-place fine-tuning);
   --manager-npz → arm 1 M29 / arm 2 r10-econ-mgr (the speed-runner coach);
   --reward-economy v2 (the only scientific variable in the whole case: the wage system); seed 22;
2. [Remove] the whole teacher-distillation set (--teacher-override/--distill-beta/
   --distill-anneal-actor-rollouts): the king anchor carries the old shallow doctrine, and distilling
   toward it would fight the deep learning goal. This is re-education, not replication; the deviation is
   registered as such;
3. [Zero] --worker-additional-terminal-death-cost 64.0 → 0.0: v2 already contains the revised death
   pricing (2+24×0.7^(d-1)); an additional constant penalty would drown the depth gradient and work
   against the purpose of the reform.

### Kept decisions (and reasons)
- num-envs 4 × n-steps 512 × total-steps 266,240: the original R8 geometry kept verbatim, so that
  dry-curriculum (a schedule over all 130 rollouts), critic warm-up and every other rollout-measured
  semantics keep their original meaning; no unit reform, minimal blast radius (the 32-environment speed-up
  is left to R13 infrastructure);
- --reset-optimizer --reset-worker-critic --critic-warmup-steps 16384: the new economy abruptly changes
  the return scale, which is exactly what critic reset + warm-up are for;
- --worker-action14-logit-bonus 2.5, the observation view, autonomy off: unchanged.

### G0 calibration (in progress)
A 4,096-step smoke test measures worker throughput → sets the arm duration and fuse (24 h anti-zombie
floor, following precedent).

### Compliance record during G0 calibration (2026-08-30)
1. Warm-up > total steps was blocked (an adjustment for the smoke test only; the official arms keep
   16384);
2. Resuming across environments needs an explicit --allow-environment-restart-resume (the rule matches
   this case's scenario exactly; adopted);
3. [Third recipe deviation] --dry-curriculum-schedule removed: the BC PASS receipt of the dry-window
   demonstration set is bound by design to the code fingerprint of its birth world, so after the economy
   change the old certificate fails closed (the behaviour the rule explicitly intends). Two ways out:
   recast the BC certification (a separate campaign, not merged here) or drop the scaffold. Decision: drop
   it. The worker being re-educated has already internalised dry-window skills; the scaffold serves the
   learning-from-scratch stage and is not needed here.

---
## Final chapter v0.3 (2026-08-30, G0 calibration and freeze)

### G0 smoke-test receipt (r12-g0-smoke, 4,096 steps, full chain)
- Throughput 107 steps/s (worker micro-step stepping, no window barrier) → an official arm of 266,240
  steps needs about 42 minutes; training fuse set at 14,400 s (4 h anti-zombie, following precedent);
- Final sentinel report confirms: additional death_cost 0.0 in effect, distillation beta 0.0,
  fast-forward rewards at the normal v2 scale, model_candidate.zip written;
- Rule changes landed: _validate_resume_contract gains a whitelist exemption for environment-restart
  (nine fields; everything outside the whitelist stays strict); the training contract gains a
  reward_economy field (later resumes can see the identity of the wage system); two worker_env death
  reconstruction calls become economy-aware; one test contract revised to match (the G0'.d3 delegate
  signature).
- Environment verification: two runs of the 864-test suite (one run 863 + 1 timing-related flake, passed
  on an isolated rerun; the final run on an idle machine is in progress, and green is a precondition for
  launch).

### Freeze statement
Nothing below this line changes; the master copy is r12-PREREG-FROZEN-20260830.md, and its sha256 is
authoritative (a sha256 recorded for this file refers to the pre-translation text).
Driver: `train/runs/r10-staging/run_r12_campaign.py` (not published). Launch record:
[r12-LAUNCH-RECORD-20260830.md](r12-LAUNCH-RECORD-20260830.md).

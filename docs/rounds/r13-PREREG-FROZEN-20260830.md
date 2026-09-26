# R13 pre-registration draft v0.1: classroom reform, handing DIVE windows to the worker (live-DIVE)
Status: DRAFT, under review (2026-08-30).
Design process: a nine-member survey-design-critique workflow (5 lanes surveying code / logs / tests → 3
independent designs → adversarial review) → this synthesis.

## 1. Rationale and argument (chain of evidence)
1. R12 final verdict: in-place worker re-education failed in both arms; on the devil sheets, all three
   (old worker / arm 1 / arm 2) were bit-identical over 128 seeds.
2. **Mechanism correction (addendum 1 of the R12 verdict, found in the opening survey)**: the steering
   wheel of a DIVE window was never in the worker's hands.
   - Training: WorkerWindowEnv never registers workers when it builds OptionsEnv
     (worker_env.py:688-694), so non-FARM windows are run end to end by the frozen script's dispatch
     (options_env.py:1704→1731+), without even a forward pass of the learned policy;
   - Exams: eval_assembled registers only {FARM: callback} (eval_assembled.py:1179/1190), so the DIVE
     windows on the exam sheet are driven by the script too. The devil always chooses DIVE ⇒ the worker
     takes zero actions in the whole exam ⇒ bit identity is a trivial necessity.
3. There are three more mode-independent barriers of authority: the worker always masks a11 (main-line
   progress belongs to the manager, options_env.py:1618); steps 1-8 are forbidden from stepping on trigger
   cells (a hard ValueError, options_env.py:1107-1112); and the observation view
   dual-v4-asymmetric-v3 has no window-mode feature at all (mixing windows must alias states).
4. Conclusion: the constraint is not the coach but the classroom, and not only the classroom: **training,
   exams, authority and perception all have to be handed over**. Without any one of them the reform cannot
   be expressed: without training, R12 repeats itself; without exams, whatever is learned cannot show;
   without authority, the worker in the classroom still cannot press the descend key; without perception,
   the worker cannot tell which lesson it is in.

## 2. Proposition
Promote DIVE windows to first-class live learning windows (the worker drives each micro-step, they enter
the gradient, and deaths go into the wage on the spot), together with the handover of authority and an
extended perception, so that the certified worker learns to survive at depth under the v2 economy without
forgetting its shallow combat strength.

## 3. Design (skeleton = the accounting-constitution design, grafted with the anchoring method of the smallest-cut design and the acceptance gates of the deep-water-school design)

### 3.1 The reform itself (in-place rule change, flag off by default)
- New flag `--worker-learning-window-scope {farm-only, farm-dive-v1}`, default farm-only, with fail-closed
  named validation; under farm-dive-v1 the DIVE branch of `_advance_to_learning_window` opens a live
  window in the same way as FARM (cloning the ready template at worker_env.py:1318-1338:
  _win_begin(DIVE) + _consume_fuse_recovery (already supports dive) + _drain); RESUPPLY and p_skip dry
  FARM windows stay scripted fast-forwards (survey conclusion: RESUPPLY is a near-single-action,
  zero-gradient waste window, and live-DIVE already hands the resupply decision on deep dives to the
  worker).
- **Route decision (following the adversarial review): drop the subclass / injection route and change
  worker_env.py / options_env.py in place.** Cost = the protocol bundle sha rotates and the frozen anchors
  need re-baking; in return there is no legal ambiguity (no open question of cross-identity pairing) and no
  copy drift. The re-bake doubles as the ultimate certification that "off by default = bit-identical" (see
  4.1), turning the loss of the anchor economy into a free regression test.
- Discipline for the flag-off path: every new instance attribute is read with a getattr default (protecting
  9 __new__ shell tests); the default path consumes no extra RNG (protecting the bit-level parallel of
  G0'.a); the old WorkerSentinelCallback is not touched (the frozen R7 schema), and the new R13 audit uses
  a new sentinel class.

### 3.2 Rules for handing over authority (minimum delegation)
- Only when scope=farm-dive-v1 and self._win['opt']==DIVE: unmask m[11] and exempt the strong mask and the
  hard guard on stepping onto trigger cells in steps 1-8; **the a10 quest mask is not touched** (a safety
  clause, to be revisited on stall forensics); FARM/RESUPPLY window masks stay bit-identical (enforced by
  a new test file).
- The same rule on the exam side: eval_assembled gets the flag `--worker-window-registration {farm-only,
  farm-dive-v1}`, default farm-only; with the flag on it registers {FARM: cb, DIVE: cb}, activates the
  handover of authority and relaxes the FARM-only check; the default path is bit-identical.

### 3.3 Extended perception (new view + migration)
- New view `dual-v5-window-mode-v1`: the full v4 order of 635 + the snapshot prefix unchanged byte for
  byte, with 12 dimensions appended: [0:3] window-mode one-hot; [3] dive_best_d/15; [4] stall clock
  (tau−dive_last_progress_tau)/140; [5] tau/TAU_CAP; [6] dungeon_level/15; [7] (char_level−dungeon_level)
  normalised; [8] cumulative kills in the window, normalised; [9] HP difference in the window, normalised;
  [10:12] reserved zeros. New name / new layout constant / new sha registered side by side; the frozen old
  view is not touched.
- Migration tool migrate_worker_obs_v4_to_v5.py: certified worker zip → R13 entry zip, with the first layer
  zero-filled for the 12 new columns ⇒ at load time the argmax is bit-identical to the certified worker
  (gate D0, hard assertion ==100%); the contract bakes in worker_policy_observation_view=v5,
  worker_learning_window_scope=farm-dive-v1 and a parent_worker_zip_sha256 lineage field ⇒ key-by-key
  equality on resume holds naturally, **and the nine-field whitelist _ENVIRONMENT_RESTART_ALLOWED_DRIFT is
  not extended by a single field (open the door without opening a hole)**.

### 3.4 Accounting constitution
- Redefine "step": an SB3 timestep = one micro-step in any live window (FARM or DIVE).
- Step calibration formula (pre-registered, frozen after the G0 measurement): T = round2048(266240 /
  (1−s_G0)), s_G0 = dive_live_share in the G0 smoke test; hard cap 393,216 (192 rollouts); s_G0>0.32
  stops and reports. Prediction (from R12 arm 2, s≈0.27): T≈364,544, about 55-70 minutes of wall clock
  (the measured 107 steps/s already includes the ff forward passes). Reason: in R12, 266,240 were pure FARM
  lesson steps; keeping the number literally would silently cut FARM lesson time by about 27%, and since
  gate C was R12's real cause of death, that would repeat the same death: FARM lesson time must not become
  a variable that nobody legislated.
- sentinel_every 500,000→63,488 (aligned with ckpt, about 5-6 time points per run, fixing R12's
  accounting incident of having only two points, in the middle and at the end); new sentinel fields (new
  R13 class): dive_live_windows/steps/deaths/descends, a per-point dive_share series, a histogram of death
  depth × remaining steps, FARM window kills bucketed by dungeon_level, a per-window decomposition of the
  value loss, a11 usage rate.
- Accounting identity test: farm_live_steps + dive_live_steps == the increment of num_timesteps.

### 3.5 Training recipe (arm α, annotated flag by flag against the R12 recipe)
[19 flags unchanged] --worker --algo mppo --gamma 1.0 --max-steps 3000 --num-envs 4 --n-steps 512
--seed 22 (kept at 22 for longitudinal comparability with R12 on the same seed) --device cpu
--allow-environment-restart-resume --allow-manager-change --gradient-clip-mode
separate-root-context-critic-v2 --manager-heuristic level-margin-1 (a self-paced curriculum: DIVE opens
only when over-levelled; R12 measured a natural 73/27 ratio and the deepest classroom, with no new
curriculum machinery) --worker-action14-logit-bonus 2.5 --lr 0.0001 --ent-coef 0.005 --target-kl 0.01
--no-drink-sovereignty --worker-fast-forward-reward-credit terminal-death-only (still covers deaths in
the remaining scripted segments; DIVE deaths move automatically to the direct path and go into the wage on
the spot; the two paths are mutually exclusive, so there is no double penalty)
--worker-additional-terminal-death-cost 0.0 --reward-economy v2 (no rule change: the wage-stripping
identity W+=r−bonus applies automatically at micro-step level, the descent bonus still belongs to the
manager; the B depth multiplier / D anti-idling / decreasing death penalty apply identically under
live-DIVE through the shared window kernel).
[3 flags changed] --total-steps calibrated by the formula in 3.4; --resume-from the R13 entry zip (3.3);
--worker-policy-observation-view dual-v5-window-mode-v1.
[1 flag added] --worker-learning-window-scope farm-dive-v1.
[Warm-up changed] --critic-warmup-steps 32768 (16 rollouts; R12 used 8: the critic is blind to the
internal states of DIVE, out of distribution, so the actor is frozen and only the critic trains first to
cover them; a full reset is unnecessary).
[Explicitly not added] a11 logit guidance (see 5.3: a first-level lever if G0 acceptance fails, not part of
the first configuration), a pdive ratio curriculum (reserve for R13.2), and any positive descent shaping on
the worker side (it would violate the rule "the descent bonus belongs to the manager"; the survey showed
that a reward conditioned on a descend window closure = returning the descent bonus in disguise).
[Arm β (contingency, not launched in parallel)] the only difference: --worker-additional-terminal-death-cost
2.0. Activation criterion (pre-registered): in arm α's exam death histogram, "d≥7 and remaining steps <500"
makes up >20% of dive deaths, or the per-depth audit shows an end-game suicide pattern (a known loophole of
economy v2: at d≥9 a single kill earns 3.0+, which already exceeds the death penalty of ≤3.4).

## 4. Anchoring method and exams

### 4.1 Re-bake as certification (the core of the anchoring method)
The in-place rule change rotates the protocol sha, so the identities of the frozen anchors on 2_114/2_115
are void and must be re-baked: the same seed ranges rerun, with zero consumption of fresh seeds:
- Staged stop-loss: under the default flag (farm-only), first an 8-seed mini diff → 32 → the full 128;
- Criterion: the re-baked rows-sha must equal the old frozen anchors bit for bit; any difference stops
  everything fail-closed for investigation;
- Four re-baked sheets: devil × old worker a/b and M29 × old worker a/b, about 4 h, doubling as the
  ultimate certification that "off by default = bit-identical".

### 4.2 Exam sheets (all reuse the same seed ranges of 2_114/2_115, zero fresh seeds)
- Gate A/B sheets: devil × new worker with --worker-window-registration farm-dive-v1, 128×2 pools;
  baseline = the re-baked devil × old worker (the scripted incumbent = the deployed form; the adversarial
  review ruled that this is the correct control for the reform's existence claim, "the learned policy
  survives better at depth than the canonical script"); at the start of the exam assert rows-sha ≠ the
  anchor + a DIVE-window driver audit (turning R12's cause of death into a fail-closed pre-check).
- Gate C sheets: M29 × new worker, farm-only registration (the same protocol as the anchor, isolating the
  FARM forgetting variable), against the re-baked M29 anchor; behavioural confirmation: a small M29 sheet
  (16 seeds) of the entry zip at training step 0 ≡ the certified worker (protecting the attribution of the
  dimension-extension surgery).
- Informational exam (no gate; requested in review): M29 × new worker with full registration = a first
  look at the deployed form, same seed ranges, zero consumption.
- Pools: confirmation pool 2_116 is spent only on a manual command; the fresh pools 2_117-119/2_126-129 are
  not touched; embargoed ranges are never spent.

### 4.3 Criteria for the four gates
- Gate A, superior survival at depth [kept]: paired died on the devil sheets, McNemar UCB<0 and
  died≤110/128. Forensics planted in advance: a death decomposition bucketed by
  (char_level−dungeon_level) (the classroom only teaches "dive once you are strong", while the exam
  contains under-levelled forced dives; this prevents a curriculum/exam distribution mismatch from being
  misread as a failure of the reform).
- Gate B, depth retention [kept]: paired Δdepth on the devil sheets, LCB>−0.10. It doubles as an
  anti-cowardice gate: a worker might learn to "not press a11 and grind the stall clock to survive" and
  fool gate A, but it cannot fool gate B.
- Gate C, anti-forgetting [kept, not loosened]: on the M29 sheets ret≥0.95× and kills≥0.95× the anchor.
  R12's real cause of death; this case answers it in three ways: zero-fill migration + a mode one-hot to
  remove aliasing + doubled warm-up + a guaranteed amount of FARM lesson time (the T formula). Advance
  position (for approval): if A/B/D all pass and C alone fails, R13.2 takes the pdive ratio curriculum
  without lowering the gate.
- Gate D, behavioural identity [amended, D-v2]:
  D0 (new) immediate invariant of the migration: the entry zip vs the certified zip, probe argmax ==100%
  (protects the attribution of the surgery);
  D1 (kept) argmax agreement on FARM probe states <99% (proves learning happened);
  D2 (new) on ≥500 states inside DIVE windows, agreement between the new worker's argmax and the canonical
  script's dispatch action <99% (a direct counter-probe to R12's structural law; =100% is a structural
  failure, and burning an exam is forbidden).
  D0-D2 consume no seeds and run before any exam.

## 5. G0 calibration and acceptance gates (launch only when all are green)
1. Throughput calibration: the smoke test (≥8192 steps) measures s (dive_live_share) and steps/s → T is
   frozen by the formula in 3.4;
2. **Hard acceptance that the classroom actually happens**: in the smoke test, at least 1 descend window
   closure per 10 live DIVE windows; if not met = stop and tune, no forced launch (so that "the learning
   side idles" does not repeat R12's "the collection side idles");
3. Bare probe: with deterministic argmax and no bonus of any kind, publish the a11 press rate and the
   descend rate;
4. Inspect what logit bonuses act on: confirm whether such knobs act only on training sampling or also on
   evaluation argmax; the finding decides whether, if acceptance fails,
   --worker-dive-action11-logit-bonus (default 0) may be introduced as a first-level lever (introducing it
   reruns the full G0 acceptance);
5. The 864-test legacy suite all green + the new test file all green (bit-level parallel / authority rules /
   the wage identity Σw≡R−bonus holds in DIVE windows / deaths booked once / v5 prefix bit-identical /
   migration 100% / the accounting identity / sentinel schema closure);
6. Training fuse 14,400 s (4 h anti-zombie, following precedent); 2 h per evaluation.

## 6. Risks and contingencies (revised after the adversarial review)
- Risk 1, incentive vacuum × a11 cold start (the main scientific risk): after wage stripping, pressing a11
  earns nothing directly, and its logit was never trained; defences = the G0 hard acceptance (5.2) + the
  bare probe + a pre-registered first-level lever (a11 bonus) + a second-level lever (potential-based depth
  shaping, a constitutional slot in the policy_reward layer, activated on command).
- Risk 2, catastrophic FARM forgetting (gate C, R12's real cause of death): the three-way answer + the
  guaranteed T; if it fails, forensics must be able to tell "FARM gradient diluted" from "FARM input
  distribution drift" (live-DIVE rewrites the state distribution of later FARM windows in the same
  episode); the bucketed sentinel fields are planted in advance (3.4).
- Risk 3, end-game suicide arbitrage: the arm β contingency + a numeric activation threshold (3.5).
- Risk 4, out-of-distribution backlash from the critic: doubled warm-up + target-kl as a second gate + an
  early warning from the per-window value loss sentinel; warm-up covers only the DIVE distribution under
  zero-filled behaviour, and a second drift after the worker learns to dive is handled by forensics on the
  sentinel series.
- Risk 5, dive_share drift during training: T is frozen and not changed mid-run; any sentinel with
  dive_share>0.40 is recorded as an anomaly in the ledger.
- Risk 6, re-bake mismatch: the default path has drifted; stop fail-closed and investigate (the 8-seed
  mini diff is an up-front stop-loss).
- Risk 7, missing co-adaptation of the assembled agent: the manager has never worked with "a worker that
  can drive DIVE", so a 2_116 confirmation run in assembled form would test a combination that has never
  existed; the informational exam gives a first look, and manager retraining is listed for R14.
- Risk 8, a break between rounds: gates A/B change the exam protocol, so R13 numbers cannot be compared
  directly with R12/R10; the launch order records this permanently.

## 7. Process
Survey dossier filed → review of this case (eight decision points) → rule changes + new test file → 864 +
new file all green → anchor re-bake as certification (4.1) → D0-D2 probes → G0 calibration + acceptance
gates → freeze (sha) → launch order → arm α launch (~1 h) → six exam sheets (~6 h) → verdict on the four
gates → final decision → (if all green) the 2_116 confirmation run on command.

## 8. Eight decision points for the review
1. **Route**: in-place rule change + re-bake as certification (recommended; no legal ambiguity)? Or the
   subclass / injection route that saves the 4 h re-bake (ruled by the adversarial review to be fragile in
   engineering and legally leaky; not recommended)?
2. **Step rule**: approve the formula T=round2048(266240/(1−s_G0)) + the hard cap of 393,216 + the stop line
   at s>0.32? Or keep 266,240 literally (silently cutting FARM lesson time by about 27%; not
   recommended)?
3. **Number of arms**: a single arm α with β registered as a contingency (recommended)? Or two arms in
   parallel (+1 h of training and twice the exam sheets)?
4. **Gate D amendment D-v2** (the three probes D0/D1/D2) and the change to gate A's exam protocol,
   formally approved?
5. **Authority over the a11 lever**: if G0 acceptance fails, pre-authorise introducing the a11 logit bonus
   and rerunning G0 (recommended)? Or always stop and ask?
6. **Suicide-arbitrage lever**: pre-authorise the activation threshold of arm β (the numeric criterion in
   3.5; recommended)? Or ask case by case?
7. **Advance position on gate C**: if A/B/D pass and C alone fails, R13.2 takes the ratio curriculum
   without lowering the gate; take that position now (recommended)?
8. **Informational exam** (M29 × new worker with full registration, a first look at deployment, no gate,
   no consumption; recommended)?

---
## Revision v0.2 (2026-08-30: all eight points approved + a special step rule)
Decision: all points approved as recommended; more training steps may be added if they turn out to be
insufficient, since the step count may itself be a major factor.

1. **All eight decision points pass as recommended**: 1. in-place rule change + re-bake as certification;
   2. the step formula (as amended by item 2 below); 3. single arm α + β registered as a contingency;
   4. gate D amendment D-v2 + the change to gate A's exam protocol; 5. the a11 lever pre-authorised
   (introduced and G0 rerun if G0 fails); 6. the numeric activation threshold of arm β pre-authorised;
   7. the advance position on gate C (a lone C failure goes to the R13.2 ratio curriculum without lowering
   the gate); 8. the informational deployment run goes ahead.
2. **Step rule amended (special approval: more steps may be added if needed)**:
   - The hard cap of 393,216 in §3.4 becomes a soft cap: s_G0>0.32 no longer stops the run; recompute by
     the formula, record an anomaly in the ledger, and continue;
   - New "continued-training pre-authorisation" clause: if after the final verdict the forensics show
     under-training (pre-registered criterion: the dive descend rate or the a11 usage rate is still rising
     monotonically over the last two sentinel points, or the per-window value loss is still falling
     significantly), one continued-training leg of +1×T is pre-authorised (same seed, same recipe, resume;
     a ledger addendum is enough, no new command needed).
3. **Scientific note (recorded honestly)**: R11 trial 3 already showed that the manager-side "not taught
   enough" hypothesis is dead (2× steps, bit-identical), and R12 failed for structural reasons (zero DIVE
   gradient), so steps were not the constraint then; but the a11 cold start and the brand-new state
   distribution of R13 make the step count a real candidate constraint for the first time. The intuition
   about steps applies exactly here, so item 2 is written into the rules as decided.

---
## Revision v0.3 (2026-08-30, design revision after the implementation survey: the v4 view route)

### Three survey findings (each overturns one premise of §3.3 in v0.1)
1. **The freeze chain of the v5 view is far longer than estimated**: the 12-dim appended block must become a
   new segment of the controller_wire layout (rotating DUAL_WORKER_LAYOUT_FROZEN_SHA256), must extend
   leashed_ppo's frozen parameter count (+384, three constants), must perform matching surgery on the Adam
   moments (skipping it corrupts silently), and the asymmetric policy topology is locked by module-level
   constants (13012/13024 cannot coexist in one process): the migration tool grows from "one zero-fill
   step" into architectural surgery across four files;
2. **Critic warm-up is structurally unusable**: --critic-warmup-steps is bound to --reset-worker-critic,
   while a checkpoint with a contract forbids a repeated reset (an iron rule of the dual-v4 contract), and
   configure_critic_migration also refuses to rerun: the warm-up of 32768 in the v0.2 recipe cannot be
   executed;
3. **The mask features are the mode signal**: after the handover of authority, the worker-mask segment of
   the v4 observation (617-632) faithfully reflects the new rules: m[11] can be 1 only inside a live DIVE
   window. "Window-mode perception" is already built into v4, and the aliasing blind spot (states where
   both sides have m[11]=0) is exactly where the mode makes no behavioural difference (with no reachable
   stairs, the optimal DIVE behaviour ≈ FARM).

### Decision (under the delegation "the rest as proposed")
- **The first R13 launch takes the v4 view route**: no migration tool, no layout rotation, no leashed_ppo
  surgery; resume directly from the certified worker zip (the same resume path as R12 arm 2);
- The full v5 set (explicit mode one-hot + DIVE progress features + parameter-extension surgery) is
  downgraded to an **R13.2 reserve**: the environment-side observation code is implemented and registered
  (unreachable by default), and the training CLI explicitly blocks it; if the v4 route fails for lack of
  perception (revealed by the G0 classroom acceptance or the D2 probe), it becomes its own case;
- Replacement for warm-up: no warm-up, relying on the two gates target-kl 0.01 + lr 1e-4 (the same live
  precedent as R12 arm 2: 266k steps without collapse); the critic's out-of-distribution shock on DIVE
  states is watched through the value loss and guarded by gate C;
- The contract route becomes **a single whitelist key**: _ENVIRONMENT_RESTART_ALLOWED_DRIFT +=
  worker_learning_window_scope (the same kind of rule change as R12's reward_economy; there is no
  migration zip to bake into, and the "open the door without opening a hole" zip route goes into the
  reserve together with v5);
- Gate D revised: D0 (migration identity) is dropped together with the migration tool; the starting point is
  the certified worker itself, so identity holds by definition; D1/D2 unchanged.

### Implementation list (all landed, 2026-08-30)
- python/diablogym/worker_env.py: _LEARNING_WINDOW_SCOPES + fail-closed validation; a new __init__
  parameter learning_window_scope (an explicit parameter to prevent **env_kwargs leaks);
  _advance_to_learning_window gets a new branch that opens live DIVE windows (the same shape as the FARM
  template); step() does per-window accounting when the flag is on (farm/dive_live_steps, a11
  requests/executions, window-closure reasons bucketed, all read with .get to tolerate missing values and
  protect __new__ shells); v5 observation constants imported and a dimension branch (reserve);
- python/diablogym/options_env.py: a new OptionsEnv parameter dive_live_sovereignty (default False); the
  "inside a live DIVE window" exemption in _worker_masks_and_distance and in the hard guard of
  _win_step_worker (the a10 quest mask is not touched); v5 view registration + construction of the 12-dim
  appended block (reserve); the controller snapshot gate extended to the dual family;
- train/train_ppo.py: the --worker-learning-window-scope CLI + validation + a cross gate (farm-dive-v1 ⇒
  the dual-v4 view); make_env/partial/contract (farm-only is always None, rev26 unchanged) / a single
  whitelist key; set-based decisions for the dual family (policy binding assertion + resume classifier);
  R13DiveAuditCallback (an independent new class, r13_dive_audit.jsonl, without touching the frozen
  sentinel surface); v5 constants registered but blocked in the CLI;
- train/eval_assembled.py: the --worker-window-registration CLI (default farm-only); evaluate() registers
  both {FARM,DIVE} + passes dive_live_sovereignty through (with the flag off, not even the keyword
  appears); three instrumentation fixes (all keys replaced consistently to keep the action14
  reconciliation closed / the divergence reference taken dynamically by window mode / parameters
  fail-closed); DIVE constants + a drift check;
- tests/test_r13_live_dive.py: a new file with four tests (named validation / flag-off barriers and zero
  leakage / flag-on authority + the accounting identity / fail-closed on the exam side).

### Gates passed (as of this revision)
- 27 core enforcement tests (four files, including the G0'.a bit-level parallel) all green; the new file
  4/4 green;
- In-process smoke test: live DIVE windows open, a11 unmasked and actually executed 71 times, 1 real
  descent, 1 deep death booked into the wage on the spot, accounting identity 4000/4000;
- **8-seed mini diff PASS**: with the default flag, every exam-sheet row is bit-identical to the frozen
  anchor r10-cand-a (0/8 mismatches): "off by default = bit-identical" holds at trajectory level;
- In progress: the full 864-test suite and the G0 smoke test (flag on, 8192 steps, full chain).

### Known limitations (registered honestly)
- No extension of the exam-sheet row/agg architecture (required for anchor re-bake equality): the worker's
  wage/kills in DIVE windows are merged into nonfarm_r/nonfarm_kills in the archives and not listed
  separately; forensics on deep behaviour rely on the training-side r13_dive_audit.jsonl and the exam
  sheets' depth/died/mode_seq;
- Under farm-dive-v1 registration, override_rate/cap_rate include DIVE windows in their denominators and
  are not on the same definition as the old sheets (take care when reading boards across rounds);
- Under double registration, worker_calls include DIVE steps (the engage counter is shared; the
  reconciliation stays closed).

---
## Final chapter v0.4 (2026-08-30, G0 calibration and freeze)

### G0 smoke test, receipts of two runs (8192 steps each, full chain, resuming the actual certified worker)
- **G0-1 (bare)**: dive_share 0.508; 52 live DIVE windows produced only 2 real descents (0.38/10 < 1.0):
  the classroom acceptance failed, so tuning stops under §5.2; but a11 was pressed and executed 255 times,
  with 1 descent and 1 deep death booked into the wage on the spot, so the handover mechanism itself
  works;
- **Lever activated (pre-authorised by decision point 5; ledger lever_activated)**:
  --worker-dive-action11-logit-bonus 2.0, the same mechanism as the a14 prior (added only to legal rows,
  gradients flow as usual, the same distribution in rollouts and evaluation); in the old rule domain a11 is
  always masked, so the prior is automatically inert; a double write after load (live attribute +
  policy_kwargs) guarantees the prior is baked into the product zip and does not evaporate at exam time;
  contract None-off key + whitelist entry + the full set of binding assertions;
- **G0-2 (lever)**: **acceptance passed**: descends_per_10_windows 1.72 (≥1.0), 10 real descents in 58
  windows, a11 pressed 689 times, dive_share 0.383.

### Step count frozen (the special step rule applies)
s_G0 = 0.383 > the 0.32 soft cap (v0.2 revision: no stop, record an anomaly and continue) →
**T = round2048(266240/(1−0.383)) = 432,128 (211 rollouts, about 67 minutes)**, above the former soft cap
of 393,216, allowed under the approved step rule; the continued-training pre-authorisation (+1×T) stays on
standby as in v0.2.

### Gate status
- 866+4 suite green (2 files that read half-written files during editing passed on isolated reruns; the
  final run on a clean machine is a launch precondition); tests/test_r13_live_dive.py 4/4 green;
- 8-seed mini diff PASS (0/8 mismatches); the full re-bake of the four anchors as certification is in
  progress;
- Launch artefacts (not published): run_r13_rebake.py (anchoring method), probe_r13_identity.py (gate D),
  run_r13_arm_a.py (T=432,128 frozen, gate D as a fail-closed precondition, six exam sheets, a mandatory
  rows-sha≠anchor assertion at the start of the devil sheets).

### Freeze statement
Nothing below this line changes; the master copy is r13-PREREG-FROZEN-20260830.md, and its sha256 is
authoritative (a sha256 recorded for this file refers to the pre-translation text).
Driver: `train/runs/r10-staging/run_r13_arm_a.py` (not published).

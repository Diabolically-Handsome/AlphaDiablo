# v24-KL "leash": pre-registration (fixed annealing + two trip lines)

**Frozen: 2026-07-10 (frozen on synthesis; the commit timestamp is the notarization). Any deviation after training
starts must cite a pre-registration clause; otherwise it is recorded as a violation.**

> Source: a design panel (2 design proposals x 4 critiques x 1 SB3 source audit x 1 synthesis).
> Prescription in one line: give the operating brain a crutch first, then let it drop the crutch slowly.
> Five hard rules: (1) **zero changes to environment/wage/observation/mask/manager** (the precondition for not
> re-running G0/G0'/G0''; eval_assembled.py on the evaluation side is also unchanged); (2) **a single prescription**:
> only the train-side leash moves; a dry-level mix intervention is a different prescription, and its appearance in this
> version is a violation; (3) the gold seeds are opened only once, at the final review (see "gold-standard discipline"
> in the gates chapter, the only place in the text that lists the range); (4) the probe seeds 7000-7031 serve only as
> gates; (5) lessons 14/15/16 (EV audit)/17 and the power discipline (n=32 SE~4, an n=16 verdict does not feed on
> noise) apply as before.

## 0. The question

v23 proved "it can be learned" (1M peak 105.7 on the first 16 probe seeds, +12.6% over the script on the same
subset, divergence rate 29%), but not "it can hold" (1.5M=59.2, 4M=42.6 broke through the collapse line; gold 77.0 =
0.82x, no P1 tier reached). Autopsy: 95.8% of the training distribution is dry-level windows (wage ~0); in a reward
desert entropy is a free wind, and the BC anchor drifts away for free. **The single v24 prescription: add a leash to
the teacher to the PPO loss (a distillation term), anneal it on a fixed schedule, and let go only once the gait is
steady.** The exam is one falsifiable sentence: **after beta anneals to 0, the score holds the peak.**

## 1. Final decisions (D1-D5)

### D1 Mathematical form of the leash: forward distillation cross-entropy CE(pi_T, pi_theta), teacher renormalized per sample under the mask

**Decision**: L_distill = E_s[ -sum_a pi_T(a|s,mask) * log pi_theta(a|s,mask) ],
total loss = policy_loss + ent_coef * entropy_loss + vf_coef * value_loss + beta * L_distill.
Temperature tau=1, no temperature knob. CE and the forward KL(pi_T || pi_theta) differ by the constant H(pi_T), so the
gradients are identical point by point: this is not a fork in the road; CE avoids computing the teacher entropy, and tb
logs train/distill_ce.

**Audit facts take precedence over designer assumptions (two corrections; the conclusion stands, with measured
reasons)**:

1. **Reverse KL is ruled out, but not because it "explodes".** Both designs claimed that the log ratios on actions
   where the teacher has near-zero probability (delta ~10-40 nats / 16.1 nats) make reverse KL explode; the audit
   falsified this by measurement: the teacher (policy_sd.pt, 1000 obs x demos.npz) has a median top-1 probability of
   0.99971 and a median entropy of 0.0034 nats, but its logits are finite (-3.9..+7.6) and the non-top-1 log-probs
   **have a lower bound ~ -11.5** (median -10.89): reverse KL is numerically finite and does not explode. The real
   reason for ruling it out: every unit of probability mass off the anchor carries a ~10-11.5 nats penalty, and the
   optimum of reverse KL forces the student's entropy down to the teacher's ~0.0034 nats, fighting head-on with the
   entropy bonus of ent_coef=0.005: at any effective beta it is equivalent to welding the policy back in place, sealing
   off the 29% divergence room the v23 peak depended on. The ruling stands; the charge is corrected.
2. **The initial CE ~ the teacher entropy ~ 0.0034 nats** (not the 0.05-0.15 design A guessed): the student is
   warm-started from the teacher weights, so calibrating beta loss value against loss value carries **zero
   information**; it must be calibrated by gradient/behavior (see D2).

**A leash, not a welding torch (verified by source and by measurement)**: dCE/dz = pi_theta - pi_T, each component in
[-1,1], L1<=2, independent of the teacher's sharpness; the restoring force is proportional to the drift: a spring, not
a weld. When the student drifts away completely, CE only grows linearly with the logit distance, and the gradient does
not explode.

**Numerical safety of the mask (audit BLOCKER 1, mandatory)**: the teacher has measured mass on the always-masked
keys 11/12 (median 3.8e-5, P99 4.8e-4); the student's masked log-probs are ~HUGE_NEG=-1e8 (sb3_contrib
distributions.py, a finite value, not -inf). **A bare CE would contain garbage terms of ~3.8e3..4.8e4 and drown the
whole PPO loss at any beta.** The teacher logits must first be set per **sample** by the rollout mask with
`th.where(mask.bool(), t_logits, th.full_like(t_logits, -1e8))` (full_like keeps device/dtype, a critic's correction)
and then softmaxed: in float32 the masked positions underflow exactly to 0.0, 0 x (-1e8) = 0, no NaN. G-KL-A pins this
property with an assertion (if upstream ever changes HUGE_NEG to -inf, 0 x (-inf) = NaN shows up at the gate).

**Teacher carrier**: the frozen BC net train/runs/bc-worker/policy_sd.pt (a training artifact, not published; PiHead 298->64->64->15). The original
dispatch function eats the raw big dict, while train() only has the 298-dim obs of the buffer, so it is unreachable;
G1 already proved the BC net has zero divergence from the script over 42,048 calls. **The G1 verdict qualification is
carried verbatim**: the teacher was only examined on its own trajectory distribution, and its labels on the new states
the learner drifts into are unverified; the leash decaying on the annealing schedule is exactly the stop-loss for this.
Teacher fidelity is guarded by G-KL-C (torch == numpy).

### D2 beta0 calibration: analytic beta0=0.5; all empirical corridors abolished, recast as a beta-monotonic behavior gate

**Decision**: beta0 = 0.5, from three simultaneous analytic constraints (design B's framework, kept):
(a) **desert constraint**: in dry-level windows (95.8% of the training distribution) the only persistent force is the
entropy wind ent_coef=0.005; beta0=0.5 gives leash:wind ~ 100:1, sealing off "free drift"; (b) **no-weld
constraint**: at a drift of d=10% the leash's logit gradient ~ 2 beta d = 0.1 << the O(1) of PG under normalized
advantages, so divergences with a real advantage in fresh-level windows remain affordable (the +12.6% of the 1M peak
came through exactly this door); (c) **mixing scale**: both gradient terms share an O(1) upper bound
(normalize_advantage as implemented, ppo_mask.py:348-349), so beta is a dimensionless mixing ratio, transferable
across wage scales.

**The empirical gradient corridors of both designs are abolished** (two independent fatal flaws raised by four
critics, all adopted): design A's 50k/100k probes fall inside the 200k **policy-head** freeze window (as implemented
in train_ppo.py:283-304: what is frozen is mlp_extractor.policy_net + action_net, while the value head trains; design
B's "value head frozen" wording was a slip, the source is authoritative), so the CE gradient path runs entirely through
frozen parameters, measuring 0/0 or making autograd raise directly, and the only correction quota would be burned by a
degenerate measurement; design B's corridor rho=||beta grad CE||/||grad PG|| in [0.3,3] rests on a wrong scale
analysis (both terms are small early on while close to the anchor, so a healthy region may have rho~0.05-0.1; near
equilibrium rho is approximately invariant to beta, so the x4 prescription turns a knob that does not respond); a
first measurement outside the corridor would very likely be followed by a second -> "out of the corridor twice means
dead" = the experiment kills itself before it runs. **Lesson: a statistic that is not monotonic in beta, or that
degenerates in reachable states, must not serve as a gate.**

**Recast G-CAL (see the gates chapter)**: the probes move after the unfreeze (global steps 300k/600k of leg 1); the
gradient norms are only booked, not gated; the verdict statistic becomes **teacher_diverge (the student-teacher argmax
mismatch rate on rollout states)**, which decreases monotonically in beta and is defensible. Anchoring fact: without a
leash, v23 had only 1.8% divergence at 500k, so with a live leash at beta=0.5 a diverge >20% can only mean the scale or
the wiring is broken. One bounded recalibration: beta0 <- 2.0 (x4), restart leg 1 from the BC init, and the burned
steps are deducted from leg 8; only once in the whole run; a second trigger = the design is judged dead, write the
verdict, do not change the code.

**Booked, not gated**: the CE trajectory is expected to rise from ~0.003 nats with divergence to ~0.1-1 nats (10-30%
divergence); CE persistently >3 nats = an early warning that the leash was snapped by the advantage flow, recorded only
and not judged, made visible by the leg exams. Inside leg 1's 0-200k freeze window distill_ce "has values but no
gradient"; the ledger notes this, and an autopsy must not read it as leash tension.

### D3 Gating rule: fixed-schedule annealing + two trip lines (the exam is a fuse, not a steering wheel)

**Decision: adopt design B's skeleton; drop design A's performance-gated halving + rollback.** Three reasons (each
confirmed by two critics): (1) **noise economics**: full gating bets every "lower beta or not" on a ~3-point margin at
the 0.97x threshold, while inter-leg training randomness (v23 trajectory swings >40 points) far exceeds it, so the beta
trajectory degenerates into a noise-driven random walk; the main-path beta decisions of a fixed schedule make zero
noisy comparisons, and the exam only carries a one-sided safety verdict (62.8 is 3-5 SE from the expected band, it
cannot be broken through by noise). (2) **Falsifiability guarantee**: A's rollback arithmetic makes beta=0 unreachable
within 8 legs after any single drop past leg 3 (a critic worked out the arithmetic): budget spent, proposition
unexamined; fixed halving reaches 0 **by construction** at 6M, leaving 2M of crutch-free road, and those 2M are the
experiment itself. (3) **Mechanism economics**: peak definition/rollback semantics/beta rising again/retraining quota
are each an attack surface (A's leg-0 rollback cannot be executed on a reachable path, a fatal flaw confirmed by two
critics); this design compresses everything into six closed-form clauses.

**Three corrections (critics' fatal flaws/violations, all adopted)**:
- **Removed "3 consecutive soft trips count as a hard trip"**: a self-reinforcing loop (a soft trip freezes beta ->
  the policy is pulled toward the teacher's level ~93.9 -> soft trips become easier) that, by the design's own
  predictions, would execute a healthy run with ~0.8 probability. A soft trip never stops and never rolls back; it only
  freezes beta.
- **De-noised peak definition P\***: P\* := the **second largest value** of the multiset {93.9} union {completed leg
  exam scores} (93.9 = the known closed-form constant of the script/BC on 7000-7015; with a single element the second
  largest = the larger of that element and 93.9, i.e. 93.9). A single stroke of subset luck (v23 lesson: the same ckpt
  scored 105.7 on the first 16 vs 76.0 on the full 32) cannot ratchet the line up; a real peak must be reproduced by two
  legs to raise the line. A peak can only be established by the **official exam of an end-of-leg ckpt**; mid-leg 500k
  checkpoints have no peak eligibility.
- **Step quantization registered as is**: SB3 advances by whole rollouts (512x4=2048), so "1M steps/leg" is really
  489x2048 = **1,001,472 steps/leg**, 8 legs = 8,011,776 steps; the hard budget cap is booked on this real number, the
  ledger records the actual num_timesteps, and the final reconciliation does not count it as a literal breach.

**Clause text (the driver script run_v24_legs.py is the sole executor, with no human discretion; gate_ledger.jsonl
keeps a trace of every clause)**:

- **[leg-1]** Total budget 8M (nominal) = 8 legs x 1,001,472 steps, num_envs=4, beta constant within a leg. The
  nominal beta of leg k is beta_k = beta0 * 2^{-(k-1)}, k=1..6 (0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625); anything
  below 0.01 after halving is pinned to 0 -> nominal legs 7 and 8 have beta=0 throughout. beta is monotonically
  non-increasing over the whole run and never rises under any circumstances. Pre-registered training seed formula per
  leg: seed_k = 100000 + 1000 x k (the assertion rejecting neighbourhoods of the 7000/9000 ranges stays; explicit
  reseeding after resume), sealing off the "rollback/re-run = redraw the lottery" ambiguity (a critic's violation item,
  adopted).
- **[exam-2]** After each leg closes and before the next leg launches, the exam is mandatory:
  `eval_assembled.py --worker <end-of-leg model_final.zip> --seeds 7000-7015 --tag v24-leg{k}` (argmax assembled
  replay, deterministic for a given ckpt, zero evaluation noise across legs; --tag is forced unique, otherwise the eight
  legs overwrite the same JSON, a critic's violation item, adopted). Without the previous leg's exam JSON the driver
  refuses to launch (mechanical interlock); it must also check that the previous leg's subprocess exit code == 0
  **and** that the global step count in status.json >= the leg target (the finally block would also write a half leg
  as model_final.zip, a critic's technical risk, adopted). The exam sha256 goes into the ledger.
- **[hard-3] Collapse trip line**: any leg exam < **62.8** -> **training of this run stops permanently**. No resume,
  no retraining with a changed beta, no "train a bit longer": **the rollback-retrain limit = 0**, so infinite rollback
  does not exist by definition; "rollback" only refers to the evaluation target (final-review candidates are chosen
  from existing end-of-leg ckpts by the D5 rule). The definition made explicit (critics' demand, not left for readers
  to discover): 62.8 = 0.8 x G1 is derived from the full-32 pool (script 78.5 on that pool); applied to the 16-seed exam
  (script 93.9 on the same subset) it is effectively a looser ~0.67x line, exactly the same ruler as the 2M/4M checks
  of v23 appendices B/C, so "kept unrelaxed" holds literally.
- **[soft-4] Peak-holding trip line**: leg k exam < 0.97 x P\* -> the beta of leg k+1 is not lowered
  (beta_{k+1}=beta_k), the schedule shifts right as a whole, the total stays 8M and no legs are added: a soft trip eats
  the beta=0 road at the tail; if the beta=0 legs are eaten entirely, that fact itself triggers P-crutch-no, **without
  stopping or executing the run**. The comparison uses the exam score as is, to 1 decimal; exactly equal to the threshold
  counts as held.
- **[seal-5] Knob seal**: apart from the one beta0 x4 recalibration pre-registered in G-CAL, beta0, leg length, trip
  line numbers, exam seeds, ent_coef=0.005, gamma=1.0, freeze 200k and n_steps=512 cannot move for the whole run; after
  loading, a resumed leg **asserts** that ent_coef/gamma/n_steps match the frozen recipe (load's
  `__dict__.update(data)` carries all hyperparameters back from the ckpt while config only records CLI values; a
  critic's violation item: beta is not the only hyperparameter that can silently live on, so this is sealed by
  assertion). A dry-level mix intervention is a violation on sight.
- **[final-6]** Close when the budget is exhausted or [hard-3] triggers; there is no extension clause of any kind. Leg
  failure/machine crash -> re-run with the leg's original config (including seed_k); burned steps count toward the
  budget: a crash is not an excuse to redraw the lottery.

### D4 Integration mechanics (every line number verified against the source by the audit)

Take the intersection of the two designs, corrected per audit BLOCKERs 1-5. Key decisions:

- **The subclass overrides train() as a whole** (sb3_contrib 2.9.0 has no loss hook; honestly copy
  ppo_mask.py:309-426; monkeypatching and forking are both worse, per the audit). Insertion point: after entropy_loss
  (line 384), changing one line at `loss = ...` (line 386).
- **The student distribution must come from a second forward pass** (audit BLOCKER 3): evaluate_actions only returns
  (values, log_prob, entropy) and does not expose the 15-dim distribution; use
  `self.policy.get_distribution(obs, action_masks=rollout_data.action_masks)` (policies.py:355-368), where
  `dist.distribution.logits` are the normalized log-probs. A second forward pass through a 64-dim MLP costs next to
  nothing (the audit measured the teacher forward at 182us/batch, <2% overall; no budget may go to caching/AMP).
- **load() construction contract** (audit BLOCKER 4): the subclass's own parameters (distill_beta=0.0,
  teacher_path=None) must have defaults (load constructs with only the four parameters policy/env/device/
  _init_setup_model); the teacher module goes into `_excluded_save_params()` and is rebuilt in `_setup_model()` from
  self.teacher_path (the order holds on both the fresh and the load path); otherwise the teacher is cloudpickled into
  every 500k ckpt.
- **Leg-style resume** (audit BLOCKERs 2/5): train_ppo.py has **no** resume path today, so one must be added;
  `reset_num_timesteps=False` (the default True resets the counter -> ckpt file names restart and overwrite each other,
  and the beta schedule and budget bookkeeping all break), `learn(total_timesteps=leg increment)` (False means **train
  N more steps**; passing the cumulative value would over-train); global steps are continuous -> CheckpointCallback file
  names are globally unique and the tb x-axis is continuous. On resume, --bc-init (it would overwrite the trained policy
  with the raw BC weights) and --freeze-policy-steps are forbidden. **beta injection = an explicit
  `model.distill_beta = args.distill_beta` + assert hasattr after load** (load's kwargs write __dict__ directly without
  validation, so a typo is silently swallowed; the explicit override seals off "the previous leg's beta silently living
  on"). Adam momentum carries over with load (confirmed by the audit), so there is no lr shock between legs.
  **Recorded warning**: anyone later adding a schedule to lr/clip, or setting target_kl (currently None, early stopping
  inactive; once active, beta would silently change the effective number of epochs), or switching the worker to --arch
  attn (a shared trainable extractor would let CE secretly train the extractor during the "freeze period") would
  silently break the semantics of this section.
- **tb path fix** (a critic's violation item): load carries the old tensorboard_log path back from the ckpt, so the
  curves of legs 2-8 would all be written into leg 1's directory; after resume, set `model.tensorboard_log = this leg's
  run directory` explicitly.
- **sps instrument fix** (a critic's violation item): EpisodeJsonlCallback's sps = num_timesteps/elapsed is inflated
  dozens of times on a resumed leg (the counter starts at kM), so the <1.8M/h downgrade gate would never fire; change it
  to (num_timesteps - leg start)/elapsed, and the driver script times it independently to cross-check.
- **Sentinel fix** (audit + critics): WorkerSentinelCallback.next_at is hard-coded to 500k, so a resumed leg would
  spray empty stats lines at the start; in `_on_training_start` set next_at = ((num_timesteps//every)+1) x every; the
  line dict adds beta and distill_ce fields (read from model attributes), so sentinel.jsonl carries leash readings every
  500k and is reconciled leg by leg against gate_ledger (two-book audit).
- **Fail-loud promoted to a specification clause** (a critic's violation item): inside train(), beta>0 with teacher is
  None must assert and crash: a missing teacher must not degrade silently.

**Code change list (file level; zero files on the environment side)**:

| File | Change |
|---|---|
| `train/leashed_ppo.py` (new) | `LeashedMaskablePPO(MaskablePPO)`: __init__ adds distill_beta=0.0/teacher_path=None; _setup_model rebuilds the teacher (assembles 298->64->64->15 by the SB3 key names of bc_worker.py:112-121, .eval().requires_grad_(False).to(device)); _excluded_save_params()+["teacher"]; train() overridden (when beta>0: assert the teacher exists; get_distribution second forward pass; teacher no_grad forward -> per-sample mask full_like(-1e8) -> softmax; ce=-(t_probs*logp).sum(-1).mean(); loss += beta*ce; log train/distill_ce (mean across minibatches, not just the last batch)/train/distill_beta/train/teacher_diverge/train/teacher_top1_conf (off-distribution early warning, observation only); with beta=0 the whole block is skipped by an if); an embedded calibration probe (--calib-probes list of global steps: at each point measure g_ce/g_pg/diverge with th.autograd.grad (retain_graph, does not pollute .grad) and write calib.jsonl; diverge>20% sets the _calib_tripped flag) |
| `train/train_ppo.py` | argparse adds --distill-beta (0.0)/--teacher-sd (default runs/bc-worker/policy_sd.pt)/--resume-from/--calib-probes; assertions: resume forbids bc-init/freeze, after resume assert the ent_coef/gamma/n_steps contract; config records the new fields; the resume path of the mppo branch: LeashedMaskablePPO.load(env=...) -> explicit beta override + tensorboard_log override + set_random_seed(seed_k); learn(leg increment, reset_num_timesteps=False); sps computed as a difference from the leg start; WorkerSentinelCallback: next_at initialization fix + beta/distill_ce fields; the sentinel callback returns False on _calib_tripped to end the leg |
| `train/run_v24_legs.py` (new driver) | serial train->exam->verdict->ledger; the beta schedule and the soft/hard trip logic **live only in this file**; mechanical interlocks (no exam, no launch; exit code + step count checks; --tag forced; sha256 booked); the G-CAL verdict is executed from calib.jsonl |
| `train/eval_assembled.py` | **no change** (the audit confirmed: MaskablePPO.load constructs with the calling class and reads the subclass zip; the teacher is excluded, so there is no reference to a custom class; distill_beta is an inert attribute) |
| environment/wage/observation/mask/manager | **not a single line changes** (not re-running G0/G0'/G0'' holds) |

### D5 Gates, R lines, P lines: see the next two chapters (key decisions: G-KL-A must start from a random initialization, a critic's fatal flaw; the G3 candidate set is fixed as the top-2 end-of-leg ckpts, sealing the "screen one more" back door)

## 2. Gates

- **G-KL-A (weld end, before launch)**: **start from a random initialization** (a critic's fatal-flaw fix: starting from
  the BC init, divergence stays <2% at 50k even without a leash, so a dead leash would still be all green and the gate
  would have zero power), beta=100, ~50k steps: CE must fall from ~2.71 (uniform, log 15) to <0.05, and the student and
  teacher argmax must agree on >=99% of 2000 rollout states. Side assertions: distill_ce finite throughout, the
  contribution of masked keys to CE exactly 0 (pinning the HUGE_NEG=-1e8 semantics), and beta>0 without a teacher
  asserts and crashes correctly. Failure = no launch.
- **G-KL-B (zero end, before launch)**: beta=0, a fixed injected buffer + controlled RNG (np/torch both reset; the
  error must distinguish "the gradient path really changed" from "RNG phase difference", a critic's technical risk); the
  policy/value/entropy losses of a single train() update differ from stock MaskablePPO by <1e-6, and the distill
  block is skipped entirely by its if guard (the code walkthrough is recorded): legs 7-8 are exactly the
  v23 recipe, guaranteed by construction.
- **G-KL-C (teacher fidelity, before launch)**: the torch teacher vs eval_assembled's np_policy_from_sd: after key
  remapping the 6 tensors are allclose one by one (atol 1e-6) **and** the maximum absolute logit difference over a
  1000-obs forward pass is <1e-4 **and** the argmax agreement is 100% (critics: comparing only argmax is insensitive to
  misplaced weights).
- **G-CAL (calibration gate, leg 1 @300k and @600k, both after the 200k unfreeze)**: (a) distill_ce finite and >0;
  (b) g_ce>0 at both probe points (the leash is alive); (c) teacher_diverge >20% at either probe point triggers the
  action. A failure of (a)/(b) = a wiring bug -> fix the code and re-run leg 1 under the [final-6] crash clause (burned
  steps count toward the budget). (c) triggered with (a)(b) green and G-KL-A passed = a scale failure -> **the one and
  only** beta0 <- 2.0, restart leg 1 from the BC init, burned steps deducted from leg 8 (leg 8 = 1,001,472 - burned
  steps; if leg 8 is cut below 500k, P-crutch-true is automatically downgraded to the half tier); a second trigger =
  the design is judged dead: stop and write the verdict, no code change. The gradient-norm ratio g_ce/g_pg is only
  booked throughout (calib.jsonl + tb), with no corridor; see D2 for why it was abolished.
- **G-LEG (leg gate)**: the D3 clauses [exam-2] [hard-3] [soft-4], numbers and definitions as above, not repeated.
- **G3 (the only trigger of the gold standard)**: the candidate set is **fixed** = the top-2 end-of-leg ckpts by leg
  exam score (mid-leg 500k ckpts are for autopsy only and never have candidacy, sealing off an on-site rescreen in the
  style of v23's "switch ckpt knob"). Each candidate runs the full 32 (7000-7031): winner = the higher full-32 mean
  (within +/-0.05 a tie goes to the leg with the lower beta); gold-evaluation eligibility line: full-32 mean >= **74.6**
  (=0.95 x H7=78.5) and deaths <=6/32 and all four R4 sentinels pass (FARM level-change rate >2.04% loses
  gold-evaluation eligibility, >6% with deaths >6/32 voids the run; override sentinel <3%, >=8% voids the data; cap
  <5%; farm tau-bar within script +/-25%: the numbers follow the interpreted text of v23 appendix B). Below the
  eligibility line -> no gold seeds spent, the verdict stated as is (spirit of P2).
- **Gold-standard discipline**: the gold seeds **9000-9031** are opened only once in the whole case, here, for the G3
  winner only, single arm, no re-education, no retry knob. The gold verdict must carry that candidate's divergence rate
  from the script (guarding against hollow-victory rhetoric; a critic's clause, adopted).

## 3. R lines (falsifiable predictions, with numbers) and P lines (verdicts)

| # | Prediction | Number |
|---|---|---|
| R-v24.1 peak | the maximum of the leg exams (7000-7015 definition) | in [95,115], point prediction 106; if <93.9 (the script value on the same subset), the leash brings no gain over v23 and the prescription is recorded as doubtful |
| R-v24.2 anti-collapse | every leg exam >=62.8 until the budget is spent; **in particular leg 4 (cumulative ~4M) >=62.8, a direct bet against v23's 4M=42.6** | point prediction: min leg exam >=75; any leg breaking through falsifies this line (the hard trip line executes the stop-loss at the same time) |
| R-v24.3 the crutch can be dropped (strongest falsifiable form) | at least one full beta=0 leg exists, and **the leg exam of the last beta=0 leg >= 0.95 x P\*** (P\* by the D3 second-largest definition, machine-decidable) | holds -> "the crutch can really be dropped"; soft trips eat all of the beta=0 road or the hard trip fires first -> judged negative, verdict **"cannot let go within 8M"**, no switching to "the trend looks good"; registered bias: comparing a single final exam against a rolling peak has an asymmetric winner's curse; the second-largest definition of P\* is the de-noising means, and the residual bias is recorded |
| R-v24.4 innovation-conservation sentinel | the divergence rate from the script of the last beta=0 leg exam | in [10%,45%]; <2% with a high score = a teacher parrot, and the verdict may only say "cruising on the anchor", never "independent of the anchor" |
| R-v24.5 sentinels | all four R4 sentinels green throughout; the beta/distill_ce of sentinel.jsonl reconcile leg by leg with gate_ledger | a mismatch = a mechanical violation, recorded |
| R-v24.6 gold (if triggered) | gold-pool mean | in [85,110], point prediction 92 (anchors: v23 gold 77.0 as the lower-bound lesson, the 1M peak's full 32 = 76.0 as the half-pool discount lesson) |

**P1 three tiers (numbers carried over from v23 without a single change)**: strong win gold mean >=93.9 and deaths
<=4/32; main win >=89.2 and deaths <=4/32; weak pass [84.5,89.2) and deaths <=4/32, where the verdict must not say
"reaches H level". Power discipline: n=32, SE~4; the n=16 leg exams are deterministic replays with zero evaluation
noise, but the subset noise of extrapolating to the full pool/gold pool is guarded by the G3 full 32: the two levels are
kept separate in the verdict.

**P-crutch (a new dedicated verdict tier)**:
- **True**: R-v24.3 holds and the final beta=0 leg exam >=0.97 x P\* and its divergence rate >=2% and gold >=89.2 and
  deaths <=4/32 -> "the leash taught it to walk alone; the operating brain is independent of the anchor" (the verdict
  carries the divergence number).
- **Half**: R-v24.3 holds but the final leg exam is in [0.95,0.97) x P\*, or the beta=0 road is cut to <2 full legs by
  the sps downgrade/recalibration deduction -> "peak holding exists, at a discount"; no promotion. The [0.95,0.97) grey
  zone is absorbed and named by this tier and is no longer a conflict between two readings (a critic's fatal-flaw fix:
  the R-line 0.95 and the soft-trip 0.97 have no operational conflict in a regime without rollback, since beta is
  already 0 and freezing 0 is still 0; only the verdict tiering remained, and it is fixed here).
- **False**: gold passes some P1 tier but the G3 winner comes from a beta>0 leg -> the verdict must say **"still on
  crutches; the independence claim does not hold"**; no borrowing credit.
- **No**: R-v24.3 judged negative -> "cannot drop the crutch, or cannot hold the peak", with a **dual attribution** from
  the ledger: a sequence of soft trips eating the schedule = falsification of the annealing schedule; a drop after beta=0
  = falsification of the capability; cause and effect must not be reversed, and a negative schedule verdict must not be
  written as a negative capability verdict (a critic's clause, adopted).
- **P2/P3 carried over from v23**: breaking the collapse line stops without embellishment; below the eligibility line,
  no gold seeds are spent.

## 4. Leg schedule (total budget 8M nominal = 8 x 1,001,472 real steps)

4-env SubprocVecEnv @900-950 sps, 1M steps ~18 minutes, a 16-seed exam ~7 minutes:

| Leg | Nominal step range | beta (nominal, shifted right by soft trips) | Events |
|---|---|---|---|
| before launch | - | - | G-KL-A/B/C ~15 minutes; any failure = no launch |
| 1 | 0-1.0M | 0.5 | bc-init + freeze-policy-steps 200k (freezes the **policy head**, value-head warm-up, the v23 recipe unchanged); G-CAL @300k/600k; leg exam |
| 2 | 1-2M | 0.25 | resume; leg exam |
| 3 | 2-3M | 0.125 | leg exam |
| 4 | 3-4M | 0.0625 | leg exam = **the main battlefield of R-v24.2** |
| 5 | 4-5M | 0.03125 | leg exam |
| 6 | 5-6M | 0.015625 | leg exam (the next step 0.0078<0.01 -> pinned to 0) |
| 7 | 6-7M | **0** | leg exam (peak-holding proof) |
| 8 | 7-8M (minus burned steps if recalibrated) | **0** | final exam -> R-v24.3 (confirmation proof) |

Wall clock: pre-launch gates ~15 min; training 8x18 ~2.4 h; leg exams 8x7 ~56 min; G3 full 32 x 2 ~28 min; one gold
evaluation ~14 min; total ~4.5 h (the teacher forward overhead is <2% and does not move the sps check line).
**sps downgrade clause**: the driver times itself; cumulative <1.8M/h -> legs 7-8 each cut to 500k (cut the tail, not
the head: the high-beta legs are safety-critical); pre-registered consequence: P-crutch-true is automatically
downgraded to the half tier (a critic's violation fix: a downgrade must downgrade the verdict at the same time; 1M of
road must not claim the credit of 2M).

## 5. Decision points (only real forks, with defaults)

1. **beta0 = 0.5 vs 0.25** (default 0.5). For 0.5: 100:1 against the entropy wind in the desert, drift prevention
   first; for 0.25: the v23 peak relied on 29% divergence, fear of suppressing the peak. The R-v24.1 lower bound 95 is
   the only detector; peak suppression can only be recorded after the fact.
2. **P\* peak definition = second largest vs simple max** (default second largest). Cost: a real peak must be
   reproduced by two legs to raise the soft-trip line; one leg of delay buys "half-pool luck cannot ratchet".
3. **The one beta0 x4 recalibration of G-CAL(c): keep vs delete** (default keep). For deleting: v23 showed 0-1M is safe
   even without a leash, so a recalibration leg is almost wasted; for keeping: bounded to once, the statistic is
   monotonic in beta, and burned steps come out of leg 8 without adding budget.
4. **beta=0 road of 2 legs vs 1 leg (leg 7 runs 0.0078 instead of pinning to zero)** (default 2 legs). R-v24.3 needs
   two proofs, "peak holding + confirmation"; the 1-leg option buys one more gentle annealing step with leg 7, but the
   final exam becomes a lone proof.
5. **G3 candidates top-2 vs top-1** (default top-2). One more full 32 ~14 minutes buys insurance against a 16-seed
   mirage (the 105.7->76.0 lesson).

## 6. Residual uncertainty (recorded as is)

1. **Extrapolating from the 16-seed half pool**: the peak, peak holding and soft trips all rest on the 7000-7015 half
   pool; the G3 full 32 is the only line of defence. If the peak ckpt drops a tier on the full 32, the "peak holding" of
   the whole annealing trajectory is peak holding on the half pool; the verdict must carry this qualification verbatim.
2. **The teacher's off-distribution labels are unverified** (a direct descendant of the G1 verdict qualification):
   during the high-beta first 2M the policy is anchored with 100:1 force to labels that were never examined; the
   teacher_top1_conf sentinel only observes and does not judge, and the damage is concentrated and hard to attribute.
3. **The opportunity cost of the leash suppressing the peak is invisible**: beta0=0.5 may kill the 29% creative
   divergence that produced the +12.6% peak, pinning S\* to the teacher line ~94 while the annealing goes "smoothly" and
   the score is mediocre; a low-end miss of R-v24.1 + the R-v24.4 divergence note are the only two eyes.
4. **The root cause of the reward desert is untreated** (single-prescription discipline): the leash only makes drift
   no longer free; the 95.8% dry-level mix is untouched. A blunt-knife failure (low CE, slowly falling scores, every leg
   just touching the 0.97 line) is only visible in the ledger's leg-exam sequence; an autopsy of a peak-holding failure
   must first check the fresh-level action shares before talking about beta.
5. **Soft-trip noise economics**: the 0.97 x P\* band is ~3 points wide, while inter-leg training randomness (v23
   swings >40 points) will very likely exceed it; false soft trips eating the beta=0 road risk mixing up "schedule
   falsified" and "capability falsified"; the dual-attribution clause of P-crutch-no is the only line of defence.

The audit's source facts took precedence over designer assumptions throughout.

## Decision record (2026-07-10, before any implementation code)

All five decision points take the defaults: (1) beta0=0.5 (100:1 against the entropy wind in the desert first; the
peak-suppression risk is detected by the R-v24.1 lower bound 95, and a miss is recorded as is); (2) P\* = second
largest (the half-pool mirage lesson of 105.7->76.0; one leg of delay buys "luck cannot ratchet"); (3) keep one bounded
beta0 x4 recalibration (burned steps deducted from leg 8, no added budget); (4) beta=0 road of 2 legs (peak-holding
proof + confirmation proof; a lone proof does not decide "can be dropped"); (5) G3 candidates top-2 (14 minutes buy the
full-32 insurance).

**Decision addendum (2026-07-10, before launch; the review panel exposed a tension inside the pre-registration)**:
the schedule semantics of the G-CAL recalibration beta0 <- 2.0 = **the whole beta>0 prefix is reordered by the closed
form beta_k = beta0 * 2^{-(k-1)}** ([2.0, 1.0, 0.5, 0.25, 0.125, 0.0625]), with legs 7/8 pinned at 0 (the two-leg beta=0
road of decision point 4 is unaffected by the recalibration and only affected by the burned-step deduction). Also: all
22 items confirmed by the review panel are implemented (crash interlock before G-CAL, P\* excludes the leg under
review, G3 override sentinel line 3% + a +/-0.05 tie band, crash-burned steps inside the 8M hard budget, per-attempt
autopsy archived, dual-probe wiring criterion, sps numerator and denominator on the same ledger); the driver's "stop
after 4 crashes in a row" is an operational self-protection clause, not a pre-registered gate; when it triggers, a
manual autopsy follows, recorded as is.

**Decision addendum 2 (2026-07-10, before the leg-8 exam score)**: the leg-6 soft trip (86.0<91.1) shrank the beta=0
road from 2 legs to 1 (leg 8 only). The downgrade list of P-crutch-half ("cut to <2 full legs by the sps
downgrade/recalibration deduction") did not list soft trips as a cause; following the original intent of decision
point 4 ("a lone proof does not decide 'can be dropped'") the interpretation is added: **whatever the cause, with <2
full beta=0 legs P-crutch-true is always downgraded to the half tier**. However high leg 8 scores, the verdict ceiling
for this case is "peak holding exists (with the lone-proof qualification)"; the full "can be dropped" claim is left to
a workstation rematch (a budget of two full beta=0 legs).

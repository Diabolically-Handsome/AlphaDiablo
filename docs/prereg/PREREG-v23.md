# v23-IS "single seam in place": pre-registration

**Frozen: 2026-07-10 (the commit timestamp of this file is the notarization). Any deviation during the run must cite a pre-registration clause; otherwise it is recorded as a violation.**

> Source: a design panel (3 design proposals x 6 critiques x 1 feasibility audit x 1 synthesis). The four open
> decisions were settled before launch; the record is in section 5.
> Two hard rules: (1) the 303-dim manager observation contract of the frozen v22-H does not change by a single byte;
> (2) the gold seeds 9000-9031 appear exactly once in the whole run: the final verdict.

## 0. The question

Under the frozen v22-H manager, replace the FARM operating brain with a learnable worker. This run proves one thing
only: **the operating brain can be learned**. It is a substitutability claim, not a claim of surpassing: a tie is a
win (the line is in P1).

## 1. Architecture decisions (D1-D6)

- **D1 Worker scope: FARM only.** DIVE/RESUPPLY/brainstem reflexes/the B4 fuse stay frozen as they are. In-place
  DIVE throughput is 40-100k steps/hour, not enough for a decent PPO within one run; the bridge has no snapshot API,
  so deep-level spawns are unavailable; RESUPPLY is a single line, `return 13`. A DIVE worker is a separate v24
  prescription.
- **D2 Training regime: in-place training, frozen H manager in the loop (numpy forward, argmax + manager mask),
  window = episode.** No curriculum manager (its coverage claim points the wrong way) and no skill gym (the snapshot
  API does not exist). The per-beat bookkeeping of the window loop (reflex/fuse/termination ladder) is extracted into
  a shared method, so OptionsEnv and WorkerWindowEnv run the same code. **After the refactor all six G0 items must be
  green before any baseline is measured.** Training seeds: an explicit sampler that rejects
  [7000,7032) union [9000,9032).
- **D3 Wage: the raw environment reward inside the window, with exactly one item stripped = the level-change bonus
  (the 8 x sum-of-levels term of delta-dlvl+); gamma_worker=1.0; a natural window end is terminated, and only the
  3000-step cut of the base episode is truncated.**
  - Dive-arbitrage fix (the only fatal flaw confirmed twice across the whole panel): masking key 11 cannot seal it
    (the 1-8 moves and the 10 explore macro can both step on stair tiles), so the wage must be stripped. The ledger
    identity **sum w_t == window R - 8 x sum delta-dlvl+** is written into G0' as a per-window assertion. The
    manager's ledger does not change by a single character; the difference goes to the manager alone.
  - EV after the fix: taking the stairs on a drained level ~ returning the clock (neutral, no betrayal of the BC
    anchor); farming monsters where there are monsters strictly dominates. The death penalty -8 x dlvl is kept.
  - Known boundary incentive (recorded, no gate): a levelup terminates the window and forfeits the remaining wage in
    it, which gives an ordering incentive to delay the level-up kill / hoard damage; magnitude audit: the lost
    immediate-kill EV ~2.35 > the hoardable gain; the tau-bar sentinel monitors it.
  - Standby patches (**at most one**, failure signatures pre-registered, no swapping on site): if the belt had
    potions at death -> **B1** death repricing (-8 x dlvl becomes a fixed -40); if the belt was empty at death ->
    **B2** potion-pickup shaping (+0.3 x delta-belt+, capped at 1.5/level < 2.35 for a single monster, stock is
    conserved and cannot be farmed).
- **D4 Worker: 298 dims = 295 + tau/TAU_CAP + layer_clock/KILL_PATIENCE + the exhausted flag; Discrete(15) with 11
  always masked (descending belongs to the manager) and 12 (drinking belongs to the brainstem), 14 keeps the base
  mask.** 298 is the direct implementation of the v22-R5 autopsy (the drained flag cannot be recovered from the stall
  clock) and the object of re-review (R2).
  - Reflex ownership: the worker never observes a "reflex pending" state; at window start and after every step the
    wrapper drains it (beat by beat through the fuse/clock/termination ladder; a drinking beat may cross
    exhausted/CAP/death, checked beat by beat). logged == executed, so the PPO buffer is not polluted.
  - BC: clone the bc_flat pipeline, collect in place (seeds 100-227, record FARM windows only, drop whole beats forced
    by the fuse), PiHead (298->64->64->15), held-out top-1 >=0.95, per-class recall >=0.85 for classes with >=300
    samples (a class below the bar gets one weighted-CE retrain, the only BC retry).
  - Guarding against an HBC collapse: --freeze-policy-steps 200000 (value-head warm-up; the HBC autopsy: no freeze +
    value from scratch + a 3247:1 class imbalance), --ent-coef 0.005 (the F precedent), a collapse alarm: every 500k
    steps, if the top-1 action share >90% and the return < the BC line -> stop and go to P2; "train a bit longer" is
    not allowed.
- **D5 Tie line 0.95x** (0.90 was too loose a verdict; the critics' view was adopted).
- **D6 Unit discipline: "steps" always means SB3 worker steps** (the status sps definition); the downgrade line uses
  the same unit.

## 2. Gates

- **G0**: after the refactor, all six items of `pytest tests/test_options_env.py` are green (before any baseline).
- **G0'**: `pytest tests/test_worker_env.py`: (a) the script worker driving WorkerWindowEnv == OptionsEnv + frozen H
  run directly, with the window sequence/tau/per-window R/mode_seq bit-identical on seeds 7000-7007; (b) mask units;
  (c) the wage identity asserted per window + override in info; (d) numpy manager == SB3 predict (1000 obs);
  (e) truncated really takes the SB3 bootstrap branch. Any failure = no launch.
- **G0'' (added locally, stricter than the panel)**: after the refactor, the script path reproduces the 2026-07-09
  probe archive (docs/assets/window_econ_v23_probe.json, argmax, seeds 7000-7031) **seed by seed in return, depth and
  death**: bit-level evidence that the refactor did not change v22 behavior.
- **G-H7**: `eval_assembled --seeds 7000-7031 --worker script` -> the measured H7 baseline (the gold-pool 93.9 must
  not serve as the probe baseline).
- **G1**: BC gate: held-out + recall meet the bar; `eval_assembled --worker bc` >= 0.85 x H7. One retry; two
  failures go to P2.
- **G2 (~4M steps)**: the latest ckpt, assembled, full-episode return on 16 seeds >= the BC value of G1; no alarm.
- **G3 (the only trigger of the gold seeds)**: the last 2-3 ckpts each screened on 16 seeds -> the winner on the full
  32: mean >=0.95 x H7 and deaths <=H7+2 and all R4 sentinels pass. At most 2 retry rounds; each round changes
  exactly one pre-registered knob (continue training 3M / B1 or B2 / switch ckpt).

## 3. R lines (falsifiable predictions) and P lines (verdicts)

| # | Prediction | Number |
|---|---|---|
| R1 | H7 lands in the same band as the gold pool | in [85,105], point prediction 95. **Appendix A warning: measured 78.3, R1 will very likely miss; reconcile when the gold is opened** |
| R2 | 298-dim BC in-place replay ratio | in [0.85,1.05] x H7, point 0.92; <0.85 kills the hypothesis "the observation defect is the root cause of R5" |
| R3 main | 8M-step assembled agent (7000 range) | in [0.95,1.10] x H7, subjective hit rate ~55% |
| R4 sentinels | FARM level-change rate <= script + 2pp; override <3%; cap <5%; tau-bar within script +/-25%; deaths <= H7+2 | Corridor sealed: level-change rate in (baseline+2pp, 3 x baseline] -> that ckpt loses gold-evaluation eligibility; >3 x baseline and deaths >6/32 -> run void (P3); override >=8% -> data void |
| R5 | Gold seeds (if triggered) | in [85,105], point 92 |

- **P1-strong win**: gold mean >=93.9 and deaths <=4/32 -> "the learned worker reaches H level".
- **P1-win (main line)**: gold mean >=89.2 and deaths <=4/32 and all G lines pass -> "the FARM operating brain can be
  learned and can replace the script's inner loop".
- **P1-weak pass**: [84.5,89.2) and deaths <=4/32 -> "an existence proof, replacement at a 5-10% discount"; it must
  not be written as "reaches H level".
- Verdict discipline: n=32, SE~4, a tie verdict carries +/-1 SE of noise; the power limitation is written into the
  verdict.
- **P2-surrender**: BC fails twice -> a from-scratch PPO arm (ent 0.02); go/no-go <0.8 x H7 -> no gold seeds spent,
  verdict "in-place worker learning insufficient within the budget of one run". **The gold-standard evaluation has a
  latest start point in the schedule (section 4); past the cut-off only the verdict is written, with no further
  training.**
- **P3-hard stop**: the R4 void line -> fall back to the previous ckpt without embellishment; collapse alarm -> stop
  and go to P2.

## 4. Schedule (as launched)

| Phase | Budget | Content |
|---|---|---|
| 1 | ~1.5 h | write the six code pieces (including the refactor of options_env into a shared window core) |
| 2 | ~0.75 h | G0 -> G0' -> G0'' -> H7 baseline |
| 3 | ~1 h | BC collection + training + G1 (including one retry slot) |
| 4 | ~3.5 h | PPO main training, 8M worker steps @4env (gamma 1.0, ent 0.005, freeze 200k, 500k/ckpt + alarm); **sps check ~1.5 h in: <1.8M/h cuts to 6M** |
| 5 | ~0.75 h | G3 screen + full 32 -> go/no-go |
| 6 | ~0.5 h | one-time opening of the gold seeds (only if G3 passes; single arm, no re-education) + R-line reconciliation |
| 7 | ~1.75 h | retry slack / verdict / the v23 chapter of DESIGN.md |

## 5. Decision record (2026-07-10)

1. **A natural window end = terminated** (the synthesis's position, over the auditor's objection): the SMDP contract
   itself says "the window is the worker's world; credit across windows belongs to the manager"; bootstrapping V(s')
   would leak the manager's authority back into the worker's ledger. "The worker learns the end of a window as the
   end of the world" is the contract, not a bug.
2. **P1 main win line = 89.2** (0.95 x 93.9); the weak-pass tier is kept, with a downgraded verdict.
3. **Mid-run patches: allowed** (B1/B2 chosen by failure signature, at most one, recorded when enabled).
4. **Main training 8M steps** (slack is worth more than steps).

## 6. Residual uncertainty (after the synthesis nobody could vouch for these; recorded as is)

1. Tolerance of the manager's observation-distribution drift (tau/stall-clock distributions drift with the learned
   worker, putting the frozen H off-distribution; a large G3 drop means a dual attribution in the verdict, and
   excessive drift = evidence for opening a v24 joint fine-tuning case).
2. The real contamination rate of fuse overrides (<3% is an estimate; the 8% void line is a backstop, not a
   guarantee).
3. Value-head quality with gamma=1 + 200k frozen steps (small window returns should be easy to learn; no precedent).
4. The measured fast-forward tax (~15% is an estimate; the sps check in phase 4 is the only line of defence).
5. Style drift at the end of drained levels after wage stripping (monitored by the tau-bar +/-25% sentinel; the root
   fix is left to v24).

## Appendix A: window-economics probe (measured 2026-07-10, before this pre-registration)

Frozen H + script worker, 7000-7031, archived as docs/assets/window_econ_v23_probe.json:

- **argmax** (the final-evaluation distribution): 80.7 windows/episode, DIVE 0.12/episode (28/32 episodes with zero
  DIVE), mean episode R **78.3**, deaths 4/32, depth histogram {L1:26, L2:4, L0:2}. FARM window composition: 2205
  exhausted dry-level windows (median tau 26, median wage -0.03) vs 22 levelup windows (mean wage 38.7): **dry-level
  junk windows make up ~97.6% (by window count)**. A training-data mix sentinel is recorded: the training log records
  the fresh-level/dry-level window shares every 500k steps.
- **sampled** (reference): 35.6 windows/episode, DIVE 3.0/episode, mean episode 147.1 but deaths 21/32; D/stall make
  up 44% of DIVE windows (= the meat for a v24 DIVE worker, archived for later).
- Direct inferences: (1) the R1 point prediction 95 conflicts with the measured 78.3; reconcile at the opening;
  (2) the assembled agent's gold evaluation is insensitive to a DIVE worker (0.12/episode), so the FARM worker alone
  carries the composite score: the two scoreboards are judged separately and credit must not be borrowed (merged into
  the P-line verdict discipline).

## Appendix B: interpretation of the R4 corridor (2026-07-10, before any training numbers)

The measured H7 (78.5, deaths 4/32; G0'' 32/32 regression PASS) gives a script level-change-rate baseline of
**0.04%**. The "3 x baseline" clause degenerates when the baseline is ~0 (3 x 0.04% = 0.12% < baseline+2pp = 2.04%,
so the corridor is empty). Recorded interpretation: **loss of gold-evaluation eligibility = level-change rate >2.04%
(baseline+2pp, original text unchanged); run void line = >6% (= 3 x (baseline+2pp), keeping the spirit of "x3") and
deaths >6/32**. The measured override baseline is 2.63%; the sentinel line <3% and the void line >=8% are unchanged.
Derived gate numbers anchored on H7: G1 >=66.7, G3 >=74.6 with deaths <=6/32, P2 surrender line <62.8. The R1
prediction band [85,105] misses the measured 78.5; this goes into the R-line scorecard at the opening.

**Interpretation of the collapse alarm (2026-07-10, before the main training launch)**: the measured BC demonstration
class distribution is {9: 97.1%, 10: 2.9%} (the locked room means even dry levels "always have monsters", so the
first dispatch branch is always true); the "top-1 >90%" criterion is inherently useless for the worker (the teacher
itself is over the line). New criterion: **at two checkpoints, ~2M and ~4M (G2), run an assembled 16-seed replay;
< 0.8 x G1 (= 62.8) means collapse, stop and go to P2**; a sustained decline of ep_rew_mean (wage definition, BC
expectation ~1.0/window) is an early-warning signal. Measured G1: BC assembled 78.5 = 1.00 x H7 (seed by seed identical
to the script). R2 hit ([0.85,1.05], point 0.92).

## Appendix C: pre-launch review panel record (2026-07-10)

The review panel (4 lenses x item-by-item adversarial falsification) confirmed 15 items and rejected 4. Actions:

- **Sentinel wiring (blocker/major family)**: WorkerSentinelCallback implemented (every 500k steps it aggregates the
  subprocess stats via get_attr: dry/fresh-level window mix, termination-reason spectrum, reroll count, cumulative
  action shares -> sentinel.jsonl). **The first launch (~0.3M steps) was voided and restarted because of this**: no
  vehicle goes on the road without instruments; the schedule's slack absorbed the 45-minute cost.
- **Hyperparameter guard rails**: --worker asserts mppo/gamma 1.0/3000 micro-steps; --seed rejects neighbourhoods that
  collide with the 7000/9000 ranges. (The first launch command was itself compliant; the guard rails protect the
  future.)
- **Seed discipline**: fallback rerolls in reset are counted in stats["reseeds"]; bc_worker asserts episodes==128 and
  reseeds==0. The reroll risk of the demonstrations already collected in this run is ~0 (the first argmax H decision
  is always FARM), and the reroll sampler already rejects the 7000/9000 ranges: **BC is not re-collected**; the
  assertion protects the future.
- **Test additions**: option_extra adds dlvl0/dlvl_end/dry; G0'(c) adds an independent reconciliation of the
  wage-stripping formula (bonus == DESCEND_UNIT x sum range(dlvl0, dlvl_end)); green again.
- **BC recall gate definition fix**: the classes under the bar are filtered on the full set >=300 (review panel:
  filtering inside held-out dilutes it tenfold). This run's results are unaffected (both class recalls are 1.0).
- **Note on the PPO return definition (recorded, no code change)**: the wage of the opening drain (reflex beats
  before the worker's first action) does not enter the PPO episode return: the worker's uncontrollable prefix is not
  booked to the worker, which is the correct semantics and not a defect; the ledger identity (G0'.c) and the
  manager's ledger are unaffected. The tail drain (reflexes triggered by a worker action) is booked to that step as
  usual.
- **G1 verdict qualification (2026-07-10, adopting the conclusion of an independent forensic review)**: the
  bit-level agreement of G1 with H7 was confirmed by the independent review as a real result (the worker branch
  actually took over: 42,048 calls over 32 seeds, the BC net had zero divergence from dispatch on every visited state,
  and the rebuilt JSON is byte-identical). Qualification: **G1 only proves that BC can copy the teacher on the
  teacher's own trajectory distribution; it was never forced to decide outside the script's trajectories. The real
  exam of "a learned worker can replace the script" falls on the PPO checkpoints of G2/G3**, and the verdict must carry
  this text. Follow-up: eval_assembled now records worker engagement evidence (call count/action histogram/divergence
  rate from the script), so from G2/G3 on the evaluation files carry their own provenance.

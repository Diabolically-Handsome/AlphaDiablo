# v30 "worker relay": pre-registration (final text, frozen after the panel's three lenses and 33 items were implemented; the verdict appendix is at the end)

**The question**: can the champion worker's weakness at depth be fixed? Autopsy: under M29-fresh the three death maps
(7006/7011/7027) all died in FARM windows staffed by the worker at depth>=2, while the same worker under v22-H had 0
deaths over 32 maps; a13 potion hoarding collapsed 39.7%->26.6%: an out-of-distribution degradation. **The weak spot
of the 140.3 system has moved from the manager to the worker.** **The single prescription: retrain v28-leg1 under
M29-fresh (frozen); the single variable between the two arms = the leash teacher** (king static self-anchor = the
first live test of "the anchor follows the king" / the old bc teacher as control).
**Decision record**: the direction review (not published) ranked A>B>C; on 2026-07-12 B was chosen to go first; the
one-time A-rematch clause moved to [docs/design/ROADMAP-course-plan.md](../design/ROADMAP-course-plan.md).

## D1 Cast, identity chain and gates

- Manager = M29-fresh npz (sha **894413884d04adfd**); worker start = the v28-leg1 zip (sha **2f7bc9dd810956c3**,
  num_timesteps **=3,497,984** asserted at launch).
- **Three identity-chain latches (panel blocker)**: (1) the king anchor sd is exported explicitly by the driver with
  two arguments to a stable path runs/v30/king_anchor_sd.pt, and its sha is recorded on export; (2) train_ppo
  `--teacher-override` is injected via **load kwargs** (after data and before _setup_model, built right the first
  time); after loading, assert that teacher_path matches + the teacher shape is 298->15, and the config receipt
  carries both the teacher and manager shas; (3) the driver asserts that the sd sha has not drifted after passing
  G-KL-W and before every leg launch.
- **G-KL-W (king arm only)**: check_teacher_parity.py: the teacher forward vs the v28-leg1 policy.npz (sha
  **976b6c05edaa0a32**), 0 argmax mismatches over 1000 obs; the obs definition is fixed as
  `np.random.default_rng(0).standard_normal((1000,298)).float32`. **The bc arm does not run this gate** (the BC
  teacher's ~30-40% divergence from the king is scientific background); instead, the BC sd is asserted in place + its
  launch-day sha is booked (panel major: gates are defined per arm). build_teacher picks by key name and silently
  ignores the value head: verified live by the panel, 0/1000, max|delta logit| ~1.5e-6.
- **G-A0W**: a live 32-seed run on launch day (tag v30-GA0W, refuses to overwrite), per seed ret+/-0.01/died/mode_seq
  == v29-mfresh-full32.json (sha **08633101c010a297**); "free" only means the baseline side needs no evaluation. The
  launch anchor v28-G3-leg1.json (sha **6fc6a44c7862424a**) is also asserted in preflight.
- teacher-override is passed again explicitly on every leg (idempotent; the copy carried inside the zip is a redundant
  channel that is not relied on); the config receipt carries manager_npz + sha (a train-side change, recorded together
  with --teacher-override).
- Leg seeds: **king = 301_000+1_000x(k-1); bc = 305_000+1_000x(k-1)** (single point of definition; zero overlap with
  the 101xxx/281xxx/22/24/7000/9000 ranges).
- The label source of the dry-level anchor sentinel is always the BC demos (L1/old-manager distribution): the king
  arm's readings are not interpreted as anchor drift and are not compared against the v28 band (verdict discipline).

## D2 Recipe, clock and guard rails

- Each arm **2 legs x 499,712** (= 244x2048, quantum = n_steps x num_envs, dropping the 256 wording); end-of-leg gate:
  zip num_timesteps == nt_chain+leg_steps (a dynamic expectation, burned steps deducted from leg 2).
- The training cmd frozen verbatim: `--worker --algo mppo --gamma 1.0 --max-steps 3000 --num-envs 4 --n-steps 512
  --lr 3e-4 --ent-coef 0.005 --seed <seed_k> --total-steps <leg> --distill-beta 0.015625 --teacher-sd <BC_SD>
  --skip-dry --manager-npz <M29 npz> --resume-from <prev> --calib-probes nt_chain+250k,nt_chain+450k
  --calib-record-only [king: --teacher-override]` (the opponent of seal-5 is clone drift, not a new manager; G-CAL is
  recorded only, not judged, as in v28, with the probes_ok gate on every leg).
- **Every evaluation uses --manager-npz (panel blocker)**: the leg exam/full 32/gold-evaluation commands are
  registered verbatim; eval_assembled falling back to v22-H by default is a known trap, so the argument is passed
  explicitly every time.
- **Trip line (pinned by a panel blocker)**: the same-position baseline = **147.0** of v29-mfresh-s16.json (16 seeds,
  M29 assembly, the same starting worker; asserted in preflight); **a single leg with score16 < 0.85x147.0 = 124.95 ->
  that arm stops training** (applies from the first leg; arms are independent; a tripped arm takes the exam with its
  last clean-close leg as the final leg). The hard trip 62.8 (the historical disaster floor) is kept.
- **Clock recalibration (panel major)**: sanity close line 4h/leg; timeout kill 4.5h/leg (a >4.4h slow-machine
  record exists); wall clock >2h records SLOW_MACHINE + NEEDS_ATTENTION without a verdict; a worst case of ~18h in
  total keeps running as usual; the gold standard is launched manually anyway and is not bound by a window.
- **The v30 G-oasis definition (panel major)**: dry>0 -> STOP (the true skip_dry invariant); **ff_dry==0 -> recorded +
  NEEDS_ATTENTION, not judged** (a legal leg-opening form under the M29 distribution, with a note that it originates
  from the v22-H calibration).
- Crash interlock/4-retry self-protection/clearing the table on entry/timestamped evaluation logs/process-group
  kill/top-level exception catch-all/restart protocol as in v28 residual #0 (.void rotation + archiving the runs/v30-*
  directories + a restart event).
- **Training-signal dilution registered as is**: M29 DIVE 0.69/episode, L2 windows estimated at 15-25%; "can't learn"
  is a legitimate outcome (the named tier of D3-2, a panel major implemented).

## D3 Verdict (in order: eligibility -> scientific main verdict -> floor -> prerequisite -> launch)

1. **Eligibility/winner/substitution**: sentinel gate = **v29 qual_of verbatim** (descend<=2.04% and cap<5% and
   override<3%, >=8% voids, DIVE>1 and deaths>6 voids, dual attribution lets through + dual_attr_ruling decide first,
   then spend; **tau-bar recorded only**: the v28 tau-bar band would wrongly kill the 140.3 baseline distribution
   itself, a panel major); winner = the highest full-32 mean among eligible arms (+/-0.05 -> fewer deaths -> king);
   substitution/no-winner tiers as in v29.
2. **Scientific main verdict (anchor = 140.3; four tiers in order, exhaustive)**:
   (1) **relay valid** = seeds with depth2 >=12 (exposure guard, against passing by skipping class; a panel major)
   and **deaths at depth>=2 <=1** and paired mean difference against 140.3 >=+2;
   (2) **relay invalid** = deaths at depth>=2 >=3 and paired <0;
   (3) **signal diluted / can't-learn tier (proposition undetermined)** = deaths at depth>=2 in {2,3} and |paired|<2;
   the verdict carries the L2 exposure count/a13 share (closing the loop of the D2 dilution clause);
   (4) out of band (labelled "exposure collapse" when depth2<12): the verdict must carry the triple (deaths at
   depth>=2, mean diff against 140.3, deaths) and the reason.
   **Machine definition**: deaths at depth>=2 = the number of rows in the full-32 rows with died and depth>=2 (a death
   ends the episode => the final depth = the depth of death; baseline 3 = the same definition read from the 140.3
   archive: 7006/7011/7027). The scientific main verdict and launch/throne/Mark-I **never rewrite each other**
   (footnote against narrative merging, a panel major).
3. **Reproduction floor (an independent gate)**: winner full 32 < **129.1 (=0.92x140.3, the lineage-ratio
   precedent)** -> "retraining did not reproduce the starting level", the launch flow ends, and the scientific main
   verdict is recorded as usual (both directions count as answers).
4. **Launch (anchor = 112.4, the incumbent assembled agent, per the anchor-compliance rule (D3))**: paired mean
   difference >=+4 and wins >=18/32 and eligibility (already guaranteed by D3-1) and **prerequisite: paired mean
   difference against 140.3 >=+2** (panel blocker: the existing stock's +27.86/17 wins nearly saturates the launch line,
   so the launch probability with zero learning is ~30-40%; raising the prerequisite to the same line as scientific
   "valid" cuts it to ~10-15%, and the launch narrative and the scientific narrative align automatically) and **the
   deep-level casualty conjunction: deaths at depth>=2 <=3 (no worse than the baseline)**.
5. **Exhaustive no-launch table (in order; each tier carries the quadruple (mean diff vs 112.4, wins, mean diff vs
   140.3, d2 deaths); wins >=14 add the width-shift note)**: (1) vs 140.3 <+2 -> the insufficient-new-evidence /
   stock-padding block tier; (2) d2 deaths >3 -> the deep-level-casualties-not-improved block tier; (3) >=+4 and wins
   <18 -> the point-estimate gain tier; (4) in [+2,+4) -> probe level; (5) <+2 -> the incumbent assembled agent stays.
6. **Gold standard (if launched)**: winner worker x M29-fresh npz, the 4th actual opening in the gold pool's history,
   the command registered verbatim (--manager-npz mandatory); the P lines against 97.2 follow v29 D3-6 in order + the
   quick reference goes with the GOLDEN event. **Mark-I = scientific main verdict "valid" and the three conditions of
   v29 D3-7 re-judged as "learned" on the winner's full 32 (7000-7031) and P30 takes the throne**; the depth
   instrumentation of the gold archive is recorded separately and does not enter the determination.
7. **Multiple-comparison ledger (recorded only, not judged; a disclosure-style correction)**: the 4th challenger draw
   on the same-pool 18/32 line (11->16->17->this case); the P(wins>=18|p=.5)~43% note + the amount of new evidence added
   to this case's launch line (prerequisite + d2 conjunction) go with the verdict.

## R lines

| # | Prediction | Number |
|---|---|---|
| R30.1 | winner full 32 | in [115,170], point 142 |
| R30.2 | the winner's deaths at depth>=2 (the main scientific metric) | baseline 3; in [0,4], point 1 |
| R30.3 | paired mean difference against 140.3 | in [-10,+15], point +3 |
| R30.4 | paired wins against 112.4 | in [14,24], point 19 |
| R30.5 | the winner's a13 share (recorded; **reconciliation zero = 0.266, same-manager definition**, 0.397 is a historical reference only; point 0.36 = predicted rebound in potion hoarding) | in [0.24,0.44] |
| R30.6 | king-bc full-32 paired (self-anchor vs old anchor) | in [-8,+15], point +5; **reading rule: the paired win count must accompany it, \|mean diff\|<2 is read as "direction undetermined"; if the two arms have different leg counts, carry a same-leg-position comparison and note the budget confound** |
| R30.7 | gold (if launched) | in [100,130], point 112 |

Verdict discipline: carry the depth distribution, a per-seed death table at depth>=2, the a13/a9/a10 mix, DIVE per
episode, the descent bonus paid (reconstructed from the final depth, a lower bound), per-seed pairing against both
anchors, the ledger, the direction-review record, the note on the dry-level anchor label source, and a mode_seq
summary.

## Residual uncertainty

1. The direction of the self-anchor's pull in OOD states (direction-review major finding): R30.6 is the first
   measurement.
2. Width is double-edged (retraining unfreezes 17 winning maps + 8 ties; the v28-leg3 corpse is on record): the trip
   line + the prerequisite are the backstop.
3. Training-signal dilution (closing the loop of D2/D3-2 tier 3); training under explore is not used because of its
   void status; the trade-off is recorded.
4. Manager coupling backfiring (drift in depth2 exposure): the scientific "valid" tier is already bound to the >=12
   exposure guard.
5. Attribution of the causes of death, n=11 (worker attribution Fisher p~0.5; depth x death p~0.002): this campaign is
   the interventional test.
6. The winner is a max-of-2; the two arms differ in teacher and seed range, so attribution is limited to "a whole
   comparison of the anchor mechanisms".
7. The BC teacher sd has no historical sha anchor; its launch-day sha is booked (an honest gap).

*Final text frozen: 2026-07-12. All 33 items of the panel's three lenses (5 blocker/15 major/13 minor) implemented,
including two live verifications (build_teacher picking by key name 0/1000, the timing of SB3 load kwargs); the commit
timestamp is the notarization.*

## Appendix: verdict record (booked 2026-07-12; the driver autopsy and restart are on the ledger)

king 132.1/7 deaths/7 d2 deaths/a13 49.2% (blocked by eligibility) -> bc substitutes at 65.9/1 death/a13 4.7% ->
below the floor 129.1, **no launch, the throne stays for the sixth time**; scientific main verdict (winner definition)
out of band/exposure collapse, with both arms' instrumentation carried by the verdict. **R30.6 = +66.2/22 wins: the
first live test of "the anchor follows the king" wins overwhelmingly; the BC anchor erased the winning moves within two
legs (39.7%->4.7%), the third case of "a long tether binds".** Deep-level casualties worsened 3->7: retraining FARM
windows is not a prescription for deep-level survival, and courses 2/3/4 are promoted to the critical path. Zero
contact with the gold pool. Details in the v30 chapter of DESIGN.md; the driver hotfix is in train/run_v30_relay.py
(metrics) and the resume runner is train/run_v30_verdict.py.

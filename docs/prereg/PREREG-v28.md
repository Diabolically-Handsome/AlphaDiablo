# v28 "oasis endurance": pre-registration (final text v2, frozen after all 24 decisions of the critic panel were implemented)

**The question**: is v26's width problem (paired wins 11/32) **under-training** or **intrinsic to the recipe**?
**The single prescription: endurance: the v26-leg6 checkpoint + beta constant at 0.015625 (never let go) + skip-dry
unchanged, trained for 8 more legs.** Zero environment changes; pathology basis:
[AUTOPSY-v26-width](../forensics/AUTOPSY-v26-width.md). Controls: the throne v24-golden 97.2; the launch anchor
v24-G3-leg7 (92.0 per seed, the same anchor as in earlier versions, **sha256_16=22d9442257d3a3c7 pinned and asserted
by the driver at launch**); width baseline leg6 = 5/16 (against the first 16 of the anchor), 11/32 (full 32); the
anchor's first-16 mean 110.1 / back-16 mean 73.8 (the two half pools are not exchangeable; four half-pool phantom cases
on record: -29.7/-12.8/-6.3/-13.4).

## D1 Mechanism (zero environment changes; all train-side changes recorded)

- Start = `train/runs/v26-leg6/model_final.zip` (a training artifact, not published; full 32 = 108.2, divergence 41.5%), via the `--resume-from` path
  (built in v24, seal-5 assertions), --bc-init/--freeze forbidden; **START = 2,998,272** (zip num_timesteps,
  = 6x499,712; asserted equal at launch; the status.json counter measured 268 steps behind, covered by the +/-2048
  slack of the crash interlock; step counting is unified with the real SB3 chain nt_chain as the source).
- **beta constant at 0.015625**, the teacher = the original BC teacher sd, unchanged (the anchor does not follow the
  king: a self-anchor is a new mechanism, left for the next version). Magnitude argument for the pull: dCE/dz =
  pi_theta - pi_T is bounded per component, so the upper bound of CE's pull on each logit = beta = 0.0156,
  **independent of the divergence size**; direct evidence = v26 leg 6 went 112.2->114.5 at the same operating point
  of beta=0.015625 with divergence rising to 41.5%; monotonicity corroboration = legs 4-5 went 98.6->112.2 at the
  tighter 0.0625/0.03125 steps (the panel's minor citation fix implemented).
- skip-dry unchanged; the dry-level anchor sentinel is recorded only, not judged, **expected band [55,90]%, point
  70** (the launch smoke test measured 63% at the start already: the 32% in the v26 verdict was the leg-6 opening
  reading, and it kept drifting during leg 6's training; the band was re-registered on the measurement before launch,
  and the correction is recorded).
  **Sentinel definition registered as is**: sampling at round 500k points is out of phase with the 499,712 leg length,
  so each leg gets only 1 point, ~2k steps after the leg start (~ the previous leg's end state); the final leg's end
  state has no in-training reading, so the verdict wording is "an 8-point per-leg curve (leg-start definition)".
  **Recorded incident**: v26's G-oasis gate was silently skipped by `if lines:` because leg 1 had no sentinel row, so it
  never actually ran; v28's leg 1 crosses the round 3.0M point and is guaranteed a row, and the gate is promoted to "no
  row = STOP".
- **G-CAL redefined (fix for two panel blockers)**: v26's 20% divergence trip line was calibrated near the anchor
  (measured 0.0 on leg 1), and its post-trip action (beta0 x4 recalibration back to a BC cold start) contradicts the
  endurance premise. **Definition clarified (launch smoke test)**: the probe's teacher_diverge on the training
  distribution starts at only 3-6% (the 41.5% is the script divergence of the evaluation definition, a different
  ruler); the 20% trip line would not necessarily fire at once, but its threshold was never calibrated for endurance
  from a settled point and the post-trip action is undefined, so for the whole of v28 it is downgraded to **recorded
  only, not judged** (a conservative choice, the correction recorded): `--calib-record-only` (a new flag of
  train_ppo/leashed_ppo; the tripped bit is still written to calib.jsonl, the flag is not armed, and the sentinel
  callback does not kill the leg); the recalibrate branch is physically removed. Probes switch to an **absolute-step
  formula** nt_chain+250k / +450k (computed by the driver per leg and passed as arguments; parts unreachable on a short
  leg are trimmed and booked); the probes_ok wiring gate (g_ce>0 and distill_ce>0) **runs every leg**; calib rotation
  to .void on crashed attempts runs every leg.
- Leg seeds **seed_k = 281_000 + 1_000 x (k-1)** (leg 1=281000 ... leg 8=288000; zero overlap with the earlier
  training seed range 101000-108000 and with the probe/gold ranges; cmd and log use the same variable, ruling out drift
  between two definition points).

## D2 Recipe and budget + clone difference table

- **8 legs x 499,712 (244x2048)**, hard budget of new steps 8x499,712 ~ 4.0M (the "8M" wording is abolished);
  full-chain rate anchor ~111 sps: ~75 minutes/leg, 8 legs + exams + G3 ~ 11h, a long run window. Crash interlock and
  burned steps follow the v26 semantics (burned steps come out of leg 8, the last leg = min(cap, LEG-burned)).
  **Operational guard rails (not verdict inputs, registered as is)**: the retry limit of 4/leg is operational
  self-protection, not a pre-registered gate; when it triggers, training stops and completed legs enter G3 as usual;
  training subprocess hang guard timeout 3h (> the 2h sanity line), evaluation 30 minutes, timeouts booked through the
  crash interlock; evaluation stdout archived to runs/v28/*.eval.*.log; a crash retry first rotates stale
  calib/sentinel files on entry (a gate must not be fake-passed by a previous attempt's records); the driver's
  top-level exception catch-all is recorded (DRIVER_EXCEPTION event + NEEDS_ATTENTION).
- **Trip lines (redefined, pseudo-code recorded)**: hard trip score<62.8 -> training permanently stopped, **completed
  legs enter the G3 candidate pool as usual** (verbatim aligned with v26's break semantics);
  `consec = consec+1 if score<103.1 else 0` (only clean-close legs count; crashed attempts neither count nor reset; the
  first leg can count as the first), consec>=2 -> early close into G3 (budget protection, not punishment); sanity
  2h/leg -> **close into G3** (the fix for v26's bare STOP, unifying the budget-protection philosophy); the hard trip
  takes precedence over the close decision. 103.1=round(0.9x114.5,1), **the 16-seed leg-exam definition** (strictly
  distinct from the verdict lines 103/108.2 of the full-32 definition, see D3). Accepted residual risks, recorded: a
  single leg landing in [62.8,103.1) triggers no action and the next leg chain-resumes (a design choice); alternating
  slow erosion (104/102/104...) is backstopped by the 8-leg budget cap.
- **Leg exam** (7000-7015) + **width probe**: the leg-exam JSON rows are paired with the first 16 of the anchor by
  **seed key** to count wins (`ret_leg(seed)>ret_anchor(seed)`, ties/missing rows do not count as wins), pure
  post-processing with zero extra evaluation, booked in the ledger (baseline 5/16).
- **Clone difference table (run_v26_legs.py -> run_v28_legs.py, panel blocker implemented)**: (1) all 8 legs
  resume, leg 1 starts from v26-leg6; the bc-init/freeze branches are physically removed; (2)
  BETA_SCHED/sched_idx/soft trip 0.97xP*/SCRIPT_SUBSET/recalibrate/tail_cut all removed, beta constant 0.015625; (3)
  ledger runs/v28/gate_ledger.jsonl, prefix v28-leg; (4) leg exam tag v28-leg{k}, G3 tag v28-G3-leg{kk} (**the v26/v27
  drivers hard-coded the leg exam tag v24-leg{k} and overwrote the per-seed leg-exam archives of v24 and v26 over two
  rounds, including the per-seed rows of 114.5, which are lost permanently; incident recorded**); (5) exam() refuses to
  overwrite an existing archive (archive immutability promoted to a case-wide clause), half-written archives are
  rotated to .void; (6) launch assertions: START, anchor sha 22d9442257d3a3c7, width baseline archive v26-G3-leg6.json
  sha 24a905a7baf0f70a, target archives absent; (7) pairing joins on the seed key and asserts the seed set =
  {7000..7031}; (8) every non-routine path writes runs/v28/NEEDS_ATTENTION; (9) the GOLDEN_AUTHORIZED event adds
  model_sha and the full-32 archive _sha, and the gold-evaluation command is pre-registered verbatim (see D3); (10) the
  eval-assembled anchor/gold/baseline archives are checked into git with this pre-registration (stopping the bleeding
  of the loss incident).
- Train-side code changes: train_ppo.py (+ the --calib-record-only flag, wired on both the resume and fresh paths),
  leashed_ppo.py (arming of the _calib_probe flag guarded by calib_record_only; calib_record_only in
  _excluded_save_params against smuggling), run_v28_legs.py (new).

## D3 Verdict and launch

- **G3 candidates**: the top-2 by end-of-leg 16-seed mean union the top-1 by width-probe wins (deduplicated, <=3),
  all on the full 32. Tie rules: a mean tie takes the earlier leg; a width tie first compares the 16-seed mean, then
  takes the earlier leg. **The starting v26-leg6 itself is not in the pool** (its full-32 archive is a control only,
  guarding against the under-training hypothesis feeding on itself). Clock note: a full-32 exam ~96 seconds/candidate,
  <10 minutes for 3.
- **Launch line (the same anchor as earlier versions)**: full-32 paired mean difference against the anchor >=+4 and
  wins >=18/32 and deaths <=6 and the sentinels (level change <=2.04% / cap <5% / override <3% are gates, >=8% voids
  the data; tau-bar band [27.8,46.4]; full-32 mean >=74.6). **Out-of-sample side line (the panel's
  selection-inflation fix)**: a candidate that entered the pool only through the width channel must also have paired
  wins >=8/16 on the back 16 of the full 32 (7016-7031) to launch (exported for free from the same G3 exam; baseline
  6/16, null hypothesis P(>=8)~16%); the verdict discipline carries a quantitative note: the width top-1 is a max-of-8
  order statistic, with a selection inflation of about +2~3 wins under the null.
- **Winner decision**: launch-eligible candidates sorted by full-32 mean, descending; within the +/-0.05 tie band
  first compare paired wins, then take the earlier leg (v26's min(beta) rule loses its meaning under constant beta and
  is abolished).
- **P lines (against the throne 97.2, an exhaustive partition)**: deaths >6 -> revert; gold >=101.2 and deaths <=4 ->
  **P28 takes the throne**; in (97.2,101.2) and deaths <=4 -> point-estimate gain (the throne does not move); >97.2
  and deaths in {5,6} -> tie (safety qualified); in [93.9,97.2] -> tie; <93.9 -> revert. **Gold-pool opening history
  (the panel's ordinal correction)**: actually opened three times, v22/v23/v24; v25/v26/v27 were all intercepted by the
  launch line and never opened; **if this case launches, it is the 4th actual opening** (not counting the 3rd by the
  v22 definition), and the fixed-pool selection bias goes with the verdict. The gold-evaluation command, pre-registered
  verbatim: `.venv/bin/python train/eval_assembled.py --worker <winner path without .zip> --seeds 9000-9031 --tag
  v28-golden --board`; after the opening a golden_result{mean,died,_sha,model_sha} event must be written back to
  gate_ledger.
- **No-launch tiers, an exhaustive three-key dispatch (winner = the highest full-32 mean among the non-void
  candidates; the numbers of a void leg may not serve as the winner, and all void falls into the "no winner" tier)**:
  (1) eligibility block (deaths>6/sentinels/mean eligibility) -> "eligibility failed, width question unanswered
  (outside statistical power)", with intrinsic/degradation wording forbidden (arithmetic note: a winner full 32 >=103
  => a mean difference against the anchor >=+11 => the mean-difference clause cannot be the blocking reason); (2) wins
  >=18 and mean diff >=4 but the side line not passed -> "the guard against selection inflation worked, does not spend
  the gold run"; (3) wins >=18 and mean diff <4 -> "width reached but magnitude not, point-estimate width improvement,
  does not spend the gold run"; (4) wins <18 and full 32 >=108.2 (the starting checkpoint itself) -> "**width problem
  confirmed intrinsic (within statistical power)**, the under-training hypothesis is rejected, the mechanism
  prescriptions are promoted to the workstation line"; (5) wins <18 and in [103,108.2) -> "endurance without mean
  gain, width question unanswered" (neither intrinsic nor degradation is judged); (6) <103 -> "endurance degraded,
  leg 6 is a local peak for this recipe".

## R lines

| # | Prediction | Number |
|---|---|---|
| R28.1 | peak leg exam (7000-7015) | in [105,130], point 118 |
| R28.2 | anti-collapse (made falsifiable) | all legs >=62.8; number of legs <103.1 in [0,2]; point prediction: the early close does not trigger |
| R28.3 | width probe (**median over all 8 legs**, guarding against max-of-8 inflation; the single value of the final selected leg is recorded separately) | in [6,12]/16, point 9 (baseline 5; under the null the median has a low probability of landing in the band, so it is falsifiable) |
| R28.4 | paired wins of the G3 winner on the full 32 | in [12,22]/32, point 17 (line = 18, the prediction band straddles the line and is registered as is: with a binomial sd~2.7, P(>=18)~43%, a coin flip) |
| R28.5 | gold (if launched) | in [100,118], point 108 |
| R28.6 | depth distribution (recorded) | unchanged from the baseline {0:2,1:26,2:4}; if it moves, recorded without narrative |

The "final selected leg" has one definition = the width top-1 leg that enters G3 (also when there is no launch).
Verdict discipline: carry the DIVE share, a mode_seq summary, the farm tau-bar, the 8-point dry-level anchor mismatch
curve (leg-start definition), the full width-probe curve, the selection-inflation note, the entry-distribution
confound note (inherited from PREREG-v26 recorded deviation 2) and the comparison with the four half-pool phantom
cases.

## Residual uncertainty

0. **Driver restart protocol**: the archive-absent assertion of preflight means a mid-campaign driver restart is
   necessarily refused, by design (archive immutability first). A mid-campaign restart must be done by hand: rotate
   each produced archive to .void + **archive and remove the runs/v28-leg\* run directories as a whole (preflight
   asserts the directories are absent)** + append a restart event with the reason to the ledger, then relaunch.
1. The erosion pattern of 4M of continued training over 8 legs under constant beta has no precedent; the two blind
   spots of the trip lines (a single badly hurt leg propagating down the chain, alternating slow erosion) are recorded
   as accepted residual risks.
2. The starting checkpoint is a "leg 6 survivor": if the v26 leg-6 peak is itself partly a 16-seed phantom (full 32
   proved 108.2, phantom -6.3), the endurance ceiling is reduced accordingly.
3. The per-seed leg-exam archives of v24/v26 were lost (the tag collision incident), so comparisons with the leg
   curves of earlier versions only have the agg definition left.
4. The fixed-pool bias of the 4th actual gold-pool opening (if launched) is carried by the verdict.

*Final text v2 frozen: 2026-07-11. The critic panel's first review (3 lenses, 24 items: 5 blocker/8 major/11 minor)
+ the pre-launch acceptance second review (clause reconciliation/adversarial code review/continued-training
semantics, 13 items: 1 blocker/5 major/7 minor, including two definition fixes from the smoke-test measurements) are
all implemented; the commit timestamp is the notarization.*

## Appendix: verdict record (booked 2026-07-11)

Leg curve: 105.5 (0 deaths, divergence 54.8%, width 6) -> 85.4 (0 deaths, 29.3%, width 4) -> 90.4 (3 deaths,
16.1%, width 5) -> **the first use of the early-close clause** (two consecutive legs <103.1), 0 burned steps, 5 legs of
budget saved. G3 candidates: leg 1 + leg 3 (width top-1 = leg 1, 2 after deduplication).
**Leg 1 full 32 = 112.4 (a project record), 0 deaths, paired +20.42, 16/32 wins (line 18, 2 short), back 16 wins
10/16, R4 all green: after four half-pool phantom cases, the first reversal ever** (first 16 = 105.5, back 16 = 119.3);
leg 3 full 32 = 72.5, 4 deaths (eligibility failed). **The gold run not spent; the throne v24-golden 97.2 stays for the
fourth time.**

Driver verdict (tier 4, machine-correct under the frozen clauses): "width problem confirmed intrinsic (within
statistical power) ... the under-training hypothesis is rejected, the mechanism prescriptions are promoted to the
workstation line". **A partition blind spot, recorded as is**: tier 4 lumps "width did not reach the line" together
with "width did not move", while the reality was a real move from 11 to 16 wins (+5); the strength of the verdict's
"intrinsic confirmed" is to be read discounted accordingly (verdict discipline precedent: a lone proof is downgraded
half a tier); the next version's P lines must add a "point-estimate width improvement" tier for wins in (baseline, 18).
This note does not change the verdict, only the narrative weight.

R reconciliation: R28.1 hit, at the lower edge of the band (peak 105.5, off the point 118); R28.2 band hit (all legs
>=62.8; number of legs <103.1 = 2, in the band) / point prediction missed (the early close triggered); R28.3 missed
(median of 3 legs = 5 < band; the campaign closed after 3 legs, the sample halved, recorded); **R28.4 hit (actual 16
vs point 17, almost on target)**; R28.5 not triggered; R28.6 moved, recorded without narrative: depth {0:2,1:23,2:7}
(baseline {0:2,1:26,2:4}).

Mechanism observation (a post-hoc narrative label, read together with v27): a start far from the anchor + a long run
at a constant small beta = **one leg of harvest (+4.2 on the full 32), then erosion back toward the anchor** (divergence
54.8->29.3->16.1, shrinking monotonically, scores collapsing toward the teacher's level): the teacher anchor is a slow
poison for a strong model that is already far from it. "The anchor follows the king" is promoted from a candidate
prescription to the first choice of the workstation line (double evidence: v27's "cannot settle" + this case's "cannot
stay tethered for long"). New base: v28-worker-leg1 archived (train/models/, full 32 = 112.4/0 deaths, the strongest
active wide-pool model); all later training starts from it.

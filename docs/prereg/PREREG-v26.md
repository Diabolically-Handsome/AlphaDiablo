# v26 "oasis": pre-registration (final text, frozen after all 3 critic lenses were implemented)

**The question**: the reward desert is the common source of wind behind three generations of anchor erosion (v23
running bare / v24 letting go / v25 fine-tuning). **The single prescription: skip-dry: dry-level revisit windows are
run by the script (they do not enter the PPO buffer), and the worker only takes lessons in fresh-level windows.** The
leash schedule, two trip lines, BC warm start, wage, observation, mask and manager are all v24 unchanged. Controls: the
throne v24-golden 97.2; the probe archive v24-G3-leg7 (92.0, per seed).

## D1 Mechanism and semantics (verified by the critics: zero races, zero new code paths; two semantic deviations recorded)

- WorkerWindowEnv gains `skip_dry: bool = False` (off by default = bit-identical, G0' green again). The skip branch =
  the script inner loop `oe.step(FARM)` (same path and same bookkeeping source as DIVE/RESUPPLY), counted in
  stats["ff_dry"]. Measured over 90 seconds: learning windows fresh 79 / dry 0, ff_dry 2277 (ok).
- **Recorded deviation 1**: dry is a flag set when the window opens; if the script kills leftover monsters on a dry
  level, the flag clears to fresh and the wage of that window is collected by the script; it is counted honestly as
  ff_dry and the verdict does not change ("dry level ~ zero wage" holds for most windows, not all).
- **Recorded deviation 2**: because the script run picks up potions/gear per dispatch, **the entry-state distribution
  of fresh-level learning windows shifts toward the demonstration distribution**: the A/B against v24 carries this
  confound note, and the verdict template must include it.
- **Anchor coverage collapse (critic major)**: the leash only anchors buffer states, so after skip_dry dry-level
  behavior has no anchor at all. Line of defence = **a dry-level anchor sentinel (recorded only, not judged)**: a fixed
  sample of 2000 teacher states from demos.npz with the exhausted flag = 1; every 500k steps measure the student's
  argmax mismatch rate against the teacher labels and write it to sentinel.jsonl; R26.6 registers an expectation of
  <=15%. The cross-window belt/gear resource channel is not guarded; recorded as residual uncertainty.
- **The BC anchor bytes do not change** (re-sampling = a second prescription; the dry-level demonstrations are exactly
  the only source of dry-level behavior in the anchor). The desert constraint (a) of beta0=0.5 loses its object under
  skip_dry, leaving only (b)(c) in support; the value is not changed, for the sake of the A/B; the peak-suppression risk
  is upgraded and recorded: **if R26.1 misses, the autopsy's first cut is beta, not skip_dry**.

## D2 Recipe and budget (recalibrated on a measurement: 463 sps@4env, 1M steps/leg ~ 36 minutes)

- **Leg system: 8 legs x 499,712 steps (244x2048)**, beta schedule [0.5,...,0.015625,0,0] unchanged (updates per beta
  step halved, registered as is); hard trip 62.8 / soft 0.97 x P\* (excluding the leg under review) unchanged; freeze
  200k (leg 1's active leash stretch is 300k, registered as is).
- **G-CAL probes recalibrated: (250k, 450k)** (600k is mechanically unreachable under the 500k leg system, a critic
  blocker); 250k has only 50k of drift after the unfreeze, so the diagnostic power reduction is recorded; the
  fresh-level diverge baseline has no anchoring measurement; the 20% trip line is kept, but the consequence of a
  recalibration is registered on the new clock: restarting leg 1 ~ 18 minutes, affordable.
- **Total window arithmetic**: training 8x499,712/463 ~ 144 minutes + gates/leg exams/G3/gold ~ 35 minutes ~
  **3 hours**, more than 1 h of slack before the same-day deadline. Window-protection clause: any leg wall clock >30
  minutes -> STOP "the same-day window is infeasible, left for the workstation line" (the only downgrade mechanism in
  the case; v24's SPS_FLOOR/tail_cut are disabled in the v26 driver: only one downgrade mechanism per case, a critics'
  clause).
- Code change list: worker_env.py (+skip_dry, already checked in); train_ppo.py (the --skip-dry flag, sentinel
  aggregation key +ff_dry, the DryAnchorSentinel callback); run_v26_legs.py (a parameterized clone of the v24 driver:
  LEG/PROBES/prefix/skip-dry/window protection; mechanism clauses unchanged).

## D3 Verdict and launch (v25 paired-criterion precedent)

- G lines: G0' green again (ok, already run) + G-oasis (the driver asserts on the first leg that stats has learning
  windows dry==0 and ff_dry>0) + the leg-system clauses as in v24.
- **Launch line**: the G3 winner (end-of-leg top-2 -> full 32) against v24-G3-leg7, per-seed paired mean difference
  >=+4 and paired wins >=18/32 and deaths <=6 and the sentinels (level-change rate <=2.04%/cap <5%/override <3% are
  gates, >=8% voids; tau-bar recorded only). Middle tier [+2,+4): does not spend the gold run, recorded.
- **P lines (against the throne 97.2, an exhaustive partition, v25 precedent)**: deaths>6 -> revert; gold >=101.2 and
  deaths <=4 -> **P26-surprise (significant, takes the throne)**; (97.2,101.2) and deaths <=4 -> point-estimate gain
  (throne does not move); >97.2 and deaths 5-6 -> tie (safety qualified); [93.9,97.2] -> tie; <93.9 -> revert. The
  crutch verdict discipline (beta=0 road, a lone proof downgraded to half) is inherited verbatim from v24. The
  fixed-pool selection bias of the fourth gold-pool opening is carried by the verdict.

## R lines

| # | Prediction | Number |
|---|---|---|
| R26.1 | peak leg exam (7000-7015) | in [100,125], point 112 |
| R26.2 | anti-collapse | every leg >=62.8; at most 1 soft trip (the desert wind source is removed) |
| R26.3 | dry-level fast-forward share **ff_dry/(ff_dry+fresh)** | in [0.90,0.97] (denominator definition fixed; DIVE/RESUPPLY/drain end windows excluded) |
| R26.4 | gold (if launched) | in [95,110], point 102 |
| R26.5 | final beta=0 leg (if the schedule gets there) | whether letting go still collapses with the desert removed = a direct re-review of lesson 19; neutral wording, no presumed conclusion |
| R26.6 | dry-level anchor sentinel (recorded only, not judged) | mismatch rate <=15%; >15% is not judged, the verdict carries it + dry-level behavior drift goes into the autopsy |

Verdict discipline: carry the DIVE share, a mode_seq summary, the farm tau-bar, the dry-level anchor mismatch rate and
the entry-distribution confound note.

*Final text frozen: 2026-07-10. The commit timestamp is the notarization.*

## Appendix: window redraw (2026-07-10, restart)

The driver stopped under the D2 window-protection clause (measured full-chain throughput 111 sps; the 463 sps of the
90 s probe was a wrong measurement of the environment loop without the learner in lockstep, the critics' blind
estimate of 110-160 was right, and the lesson is recorded: **"throughput must be measured on the full chain"**). The
run was then restarted, with the window constraint redrawn from a same-day deadline to a longer run window. Leg system/
beta schedule/gates/launch line unchanged, not a single word; the driver guard rail became a sanity limit of 2h/leg;
the half-finished leg 1 (177k steps) was voided and cleared, and training restarted from the BC init, with the burned
steps not counted (the void reason was the window redraw, not a training failure, so it does not use the [final-6]
crash budget).

## Appendix 2: v27 branch execution record (executed 2026-07-11)

The pre-registered branch rule: "v26 beats v24 on every dimension -> train the v26 recipe for 7M; otherwise -> train
the v24 recipe for 7M". The decision rule registered before the result: every dimension = gold launched and gold mean
>97.2 and gold deaths <=4. What happened: the v26 gold run was not launched (paired wins 11/32 < 18, correctly
intercepted by the launch line) -> **the Otherwise branch**: v27 = the v24 recipe (desert + leash), 8 legs x 874,496
steps ~ 7M, with the launch criterion upgraded to the paired standard (v25/v26 precedent) and the gold-standard
discipline as before. The v27 launch came 5.8 hours after the v26 verdict was booked; the gap is recorded as is. Also
corrected: a wording bug in the v26 driver's verdict text: the "[+2,+4) tier" wording in VERDICT_PATH was a branch
text error (in fact the mean difference +16.2 met the line and it failed on paired wins 11/32; the launch decision
itself followed the clauses correctly).

## Appendix 3: v27 verdict record (booked 2026-07-11)

The eight legs as recorded: 93.9 x4 (welded) -> 96.0 (beta=0.03125) -> 94.0 (beta=0.015625, divergence only 9.1%;
at the same beta step v24 had already unwelded to 29% and v26 to 28.6%: the timing of unwelding varies hugely when the
same recipe is redrawn) -> **leg 7 (beta=0, tail-cut 500k) = 106.9, 0 deaths (divergence 40.9%, the first leg in the
whole case that let go without collapsing and instead peaked, model_sha 68470b3c2652100b)** -> leg 8 (beta=0) = 58.4,
3 deaths (divergence 86.6%), hard trip stop, 6,245,124 real training steps.

G3 full 32: leg 7 = **93.5, 1 death** (R4 all green; **the fourth half-pool phantom case, -13.4**: first 16 pool
106.9, the back 16 pool inferred ~80.1) / leg 5 = 92.6, 3 deaths (divergence 0.4%, essentially the teacher itself).
**Paired launch criterion: leg 7 mean diff +1.51 / 12 wins, leg 5 +0.62 / 16 wins -> the +4/18 line not reached, the
gold run not spent, the throne v24-golden 97.2 stays for the third time** (the tail-cut downgrade note goes with the
verdict).

Control conclusion (the main answer of this branch): oasis 4M (108.2 full 32, +16.2) > desert 6.25M (93.5,
+1.51): **1.56x the compute does not buy back the recipe gap; the v26 advantage is a property of the recipe** (the
entry-distribution confound is carried per recorded deviation 2). Reproducibility note: the v24 throne path (a leg-6
soft trip froze beta -> 110.1 -> gold 97.2) was not reproduced; the desert recipe is luck-dependent. The leg-7 counter
peak was an unregistered surprise; "letting go close to the anchor = harvest, letting go far from the anchor = falling
off a cliff; letting go can harvest, but cannot settle" is recorded as a post-hoc hypothesis (leg 8's -48.5 was its
first predictive test, passed); a candidate amendment to lesson 19 is pending review. Two wall-clock anomalies (leg 2
triggered the sps downgrade, leg 6 crawled for 4.4h, machine cause unknown) and the verdict arriving about 5 hours
later than forecast are recorded as is. Details in the v27 chapter of DESIGN.md; the run artifacts (not published)
were under train/runs/v27/.

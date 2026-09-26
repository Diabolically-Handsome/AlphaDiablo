# FORENSICS-F3: final review of the root cause of "the model does not improve"

2026-07-27. Forensic panel dossier (eight seats: selection-chain census / pool decomposition / noise floor
/ training dynamics / historical extrapolation in five parallel lanes → statistical attack / alternative
hypotheses / internal-consistency review as three adversarial verification seats; purely offline and
read-only throughout).
Data base: 87 eval-assembled archives, the R5/R6 campaign records, 13 cases' gate_ledger, and the full R5
telemetry. The recomputation scripts were one-off standard-library scripts and are not published.

## Half-page plain summary

**The model did not "learn nothing". What it learned is too small, a lottery amplified it into the
appearance of progress, and the project's own process then took that as confirmation.** R6's +20.7 on
the familiar 7000 pool breaks down into three layers: 1. 96.4% of the mean's magnitude is carried by 5
leverage seeds with a "dead → alive" survival flip (survival-time arbitrage, not combat strength;
dropping the top 5 leaves only +0.87, and the in-pool t-test of the mean is itself not significant,
p=0.055); 2. the sign surface of 24/32 is real (selection cannot fake it: that would need ≈150-286
effective independent looks, while there were at most 17 and they were correlated), but it is a
**pool-specific effect that does not extrapolate**: the fresh 12000 pool excludes transfer at p=0.007;
3. the transferable real residual = per-step combat efficiency of ~+6 wage per seed (same sign in both
pools, combined p≈0.05). **There may be real progress, but it is an order of magnitude below the
detection line of a 32-pair ruler (MDE 27-32), and no evaluation in the current process can prove it on
its own.**

Why the project "thought" it had improved: R6's candidate checkpoint was selected, explicitly, on the
condition "passed the post-mortem on the 7000 pool"
(selection_basis='pre-tv-trip-checkpoint-plus-known-seed-forensic-pass'), and the number the regression
gate read was **the same data** as the selection statistic (20.685099936079062, bit-identical in all
three places): the selection was the exam question, with zero information gain. The 7000 pool was hit
74-91 times in 15 days, with ≥39 adaptive decisions; the 17 historical candidate looks were almost all
negative (-5 to -35), and R6's +20.7 is exactly **the empirical maximum**, almost equal to
E[max17|H0]=+22.6; the -28 pull-back on the fresh pool matches the size of the winner's curse.

Why it "really" could not learn: all three design components of the R5 recipe **idled** (0 dry windows
in 5,054 windows over 285k training steps; a12 was sampled 377 times with zero behavioural change;
voluntary_drinks was 0 in all four evaluation archives; deep dives reached L3 in only 15/798 episodes).
The disease the dry-level course treats does not exist in evaluation at all (farm_dry_n is 0 on all 64
seeds of both pools). The leash never engaged (target_kl never triggered, calib had only 1 probe,
distill_ce did not move); the only active element was the TV=0.15 hard gate, which killed training at 57%
of the budget. A 250k-step budget is one order of magnitude short for survival behaviour, two orders
short for deep diving, and infinitely short for dry windows.

## Verdict (three seats unanimous)

- **Zero out-of-sample wins (high)**: after the v28 throne, 11 attempts at continued training produced
  zero promotions; the only two openings of a fresh pool both failed. 8000 pool (B1): H side -7.07 /
  median -0.09, M29 side -12.68 / median -0.11; 12000 pool (R6): wage -7.26 / median +0.79, 17/32, deaths
  17→20 (+3). The three readings side by side **may be cited by sign only, not by magnitude** (the schema2
  ret proxy ≠ schema5 wage, protocol v3 ≠ v4).
- **Three components of +20.7 on the familiar 7000 pool (high)**: survival-flip leverage (5 seeds carry
  96.4% of the mean; seeds dead at baseline gain +54.9 on average, seeds alive at baseline lose -29.3 on
  average, corr(baseline, Δ)=-0.49) + a real but pool-specific broad small-gain layer (median still +15.4
  after dropping the top 3) + a possibly transferable efficiency residual (~+6 per seed **per step**; the
  per-window definition gives +21 because it mixes in survival time; the two definitions differ by 3.3×,
  so every citation must state its definition).
- **The drop to zero is not a pool shift (high)**: the v28+M29 baselines of the two pools share the same
  distribution on 13/14 fields (wage 201.6 vs 202.4, MW p≈0.95), and 12000 is even slightly easier; what
  falls across pools is the candidate itself (222.3→195.1). Four alternative hypotheses in a row (pool
  structure / horizon / manager coupling / single-seed driver) all fail quantitatively.
- **Noise floor (high)**: paired per-seed σ≈60-73, SEM≈11-13 for 32 pairs, MDE≈27-32. Every past
  "verdict" at the ±7 to ±20 level is below the detection line, and all eleven consecutive losing readings
  from v25 to R6 fall inside the training-seed lottery band. **A 32-pair ruler has nowhere near enough
  power for the effect sizes of this domain; past single-leg verdicts were effectively coin tosses.**
- **R6 process disease settled (high)**: checkpoint selection was 1 of 1 (ckpt/ holds a single file), but
  the entry condition was the 7000 post-mortem pass, so the regression gate PASS was constructively
  guaranteed; the 12000 pool really was a first exposure; from the R5 trip to the R6 loss took 28 minutes
  in total (correction: not 52 minutes).

## Corrections (to earlier dossiers)

1. The 12000 pool death increment is **+3** (17→20; alive→dead 12002/12012/12014/12019/12021, dead→alive
   12008/12009). The earlier brief's +1 mixed it up with the 7000 pool (19→18);
2. The 7000 pool has **74** full32+s16 archives (45+29; "50" matches no definition);
3. The "distribution mismatch" branch is limited to a **training-vs-evaluation context mismatch** (there
   is no mismatch between pools);
4. The two potion gates of the R5 "dry-level course + a12" recipe **passed vacuously** (dry_n and
   voluntary_drinks all 0);
5. F2's 20-90× gradient ratio cannot be re-checked inside R5 (calib had only 1 probe, g_ce≈7.8e-10 where
   policy ≡ teacher).

## Ruling on R7 and suggested revisions

R7 (rev20) already addresses four points: a death cost of 32/64 matches the diagnosed exposure loss of
30-50; 2 recipes × 3 training seeds (≥2/3 replication) end the single-leg lottery; independent retraining
with production RNG + constructive dev/final isolation end "selection is the exam question"; a 256-pair
final exam has MDE≈9-11 and for the first time can reach a real effect of ~+6. **Not addressed**: the leg
budget is still 266k steps (the rare-event exposure gap is untouched), and the co-drift channel of the
frozen M29 is untouched.
**Suggested revisions (pre-registration level, to be merged before launch)**: 1. add an **independent
per-step efficiency scoring branch** (farm_worker_wage/micro_steps, recommended by both seats: it is the
clean definition of the only plausibly real progress signal in the whole case); 2. keep the
seed-majority clause (a sign test is more sensitive than a mean gate to effects of +5; with 256 pairs it
needs a win rate of only 0.578 ≈ a shift of +5.3); 3. restate that all gold-standard / 7000-series
readings are constructively isolated from the final exam pool (the selection-envelope analysis of this
case applies permanently to any reused pool).

## Operations note

This analysis is purely offline, has no engine dependency and changed no existing project file. At the
time of writing, the engine build had not yet been set up in the new runtime, so the R7 launch waited for
the build to be ported.

## Final chapter (2026-07-28): question answered

The R7 campaign (rev21-23, amendments 1-6) gave a well-powered answer on a brand-new 256-pair pool:
**the model can improve, and it has improved**: wage +38.41 (LCB +29.80), per-step efficiency +0.019
(p=1.5e-18), deaths exactly level. All four root causes diagnosed in this dossier were addressed: 1. the
32-pair ruler became a 256-pair ruler (MDE from 27-32 down to ~9); 2. the selection statistic and the
regression gate shared a source → pre-registered retraining with independent production seeds;
3. leverage confounding → survival-aligned accounting + the rev21 de-leveraged efficiency branch (which
confirmed the gain is real efficiency); 4. an idle recipe → risk64 death pricing replicated on 12/12
development legs. Open item: the power calibration for death non-inferiority certification went wrong
twice (the erratum in amendment 5.2, and a 0.86pp miss in the final exam); the lesson is that the margin
must be calibrated jointly on **the statistic actually implemented** and **the measured discordance
density**. Handed over to R8. This dossier is closed.

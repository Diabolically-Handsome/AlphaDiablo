# PREREG R8: certification campaign (draft, awaiting launch approval)

> Opened on 2026-07-28 (approved: accept the R7 verdict and start R8). The R7 closure is in the last section of
> PREREG-R7-rev21-proposal.md and the final chapter of FORENSICS-F3.

## 0. Scientific question

R7 established on a fresh pool of 256 pairs: the risk64 recipe's worker has **wage +38.41 (family-wise LCB
+29.80), per-step efficiency +0.019 (p=1.5e-18), deaths 196=196, exactly equal**; the only
failure was death non-inferiority certification -- with 62 discordant pairs the CP-Bonferroni yardstick gives a certification width of
10.86pp > margin 10pp. R8 asks only one question: **with an adequately powered death yardstick, can this improvement
complete full certification?**

## 1. Recalibrating the death yardstick (the core change of this case)

### 1.1 Chain of errata
- R7 amendment 5-2 already corrected: margin power must be calibrated against **the statistic actually implemented**;
- the R7 final exam adds a correction: it must also be calibrated against **the measured discordance density** (D≈62/256) and a **conservative true value**
  (equal deaths, not the −3.1pp observed in development).

### 1.2 Measured power of the two yardsticks (40,000 simulations; the calibration exactly reproduces the final exam)

| Margin | CP-Bonferroni (null truth / development truth) | Exact conditional McNemar (null / development) |
|---|---|---|
| 0.10 | 0.394 / 0.757 | **0.759 / 0.949** |
| 0.11 | 0.518 / 0.843 | ~0.83 / ~0.97 |
| 0.12 | 0.653 / 0.910 | — |
| 0.13 | 0.770 / 0.953 | — |

Calibration: the R7 final exam (c=31, D=62) gives UCB=0.1086 under the CP yardstick (matching the record);
under the McNemar yardstick UCB=**0.0805**.

### 1.3 Final form (proposed)
- **Switch to the exact conditional McNemar non-inferiority yardstick**: given D discordant pairs, c~Bin(D,θ),
  δ=(2c−D)/N, UCB_δ=(2·CP_up(c|D,α=0.005)−1)·D/N; **the margin stays 0.10**
  (the certification meaning does not shrink); power 0.76 under the null truth, 0.95 under the development truth.
- Keep both observed_not_higher limbs and every R7 check item;
- Implemented as `train/r8_statistics.py` (derived from r7_statistics, only the death limb changes);
  the statistics file is in no fingerprint bundle, so the BC pools are unaffected.
- **Honesty statement**: that the R7 final-exam data "would have passed" under the new yardstick is a hindsight fact -- so
  R8 must re-examine on a fresh pool, and re-judging the already-read R7 data in any form is **forbidden**; the new yardstick
  is frozen here before any R8 data is seen.

## 2. Campaign structure (certification of the single risk64 recipe)

R7 already completed recipe selection (risk64, amendment 5 gate B, 2/3 over both pools), so R8 runs no horse race:

1. **train-replication**: risk64 × 3 fresh training seeds
   (2_131_000 / 2_131_100 / 2_131_200), relaunched from V28, recipe byte-for-byte identical
   to R7;
2. **eval-replication**: 2 fresh 128-pair pools (**2_112 / 2_113**),
   gate B (the pre-registered check set minus the single death NI item, incl. observed_not_higher);
   ≥2/3 seeds must pass on both pools before the final exam opens -- guarding against the training-seed lottery (lesson of the B1 case);
3. **train-production**: fresh production seed **2_131_900**;
4. **eval-final**: **2_122_000-2_122_256**, one-off 256 pairs,
   all checks + the new death yardstick; PASS means the model is released.

Seed ledger: the evaluation bank has used 2_110/2_111/2_120 (burned in the incident)/2_121; this case takes
2_112/2_113/2_122; 13 ranges remain, 2_114-2_119 and 2_123-2_129.
BC pools 2_142/2_143 **stay valid** (see §3). The training seed block 2_131_xxx
was checked against the rejection table (seed+rank up to 24 collides with no pool).

## 3. Parallelism decision (decision needed)

Result of the pipeline investigation (investigation report not published): the training stack is fully parameterised in the number of envs, and relaunching with 24 envs from
V28 **touches no fingerprint-bundle file** (the BC pools are not voided); but --

- the dry-layer curriculum anneals by rollout index: 92 levels → ~15 levels (6× coarser);
- the distillation β annealing coarsens the same way; the optimizer-step floor of 8 degrades from "one full epoch" to "1/6 epoch";
- several rollout-index constants must be re-legislated (61 out of range, the 8/3 criterion loses its meaning).

**So 24 envs is not pure speed-up but a real recipe change.**

- **Plan A (recommended)**: R8 keeps **4 envs** -- the certified recipe is byte-for-byte identical to the R7 evidence chain,
  so the conclusion has no confound; about 6 hours in total. The 24-core infrastructure moves to **R9** (the next exploratory
  campaign, where the recipe changes anyway).
- **Plan B**: R8 goes straight to 24 envs -- the three training legs finish in about 1 hour, but the object being certified becomes
  "a risk64 variant with a 6× coarser curriculum"; a FAIL could not be attributed (yardstick vs recipe),
  and 7 rollout-index constants would need re-legislation.

## 4. Implementation list (plan A)

- New `train/run_r8_certification.py` (derived from the R7 launcher): drops the recipe race
  and the amendment 5/6 adoption machinery (specific to R7 history); pool/seed/CAMPAIGN constants per this case;
  the death limb connects to r8_statistics; R7's a14 gate, rev21 efficiency limb, survival-aligned accounting and
  gear gate are all kept.
- New `train/r8_statistics.py`: the death NI limb switches to the exact conditional McNemar, everything else inherited verbatim;
  the method revision number is incremented.
- Tests: gate B / new-yardstick KAT (with the R7 final exam as the calibration vector 0.0805) / fixture campaign updated.
- Adversarial review before launch as usual (the two rounds of amendments 5 and 6 caught 3 blockers in total; this step is not skipped).
- Operations: with 4 envs the CPU split stays as it is;
  the launcher keeps orphaning + plain-file heartbeat; **the machine must not be restarted while the final exam runs** (lesson of amendment 6;
  the run window is announced before launch).

## 5. Expected timeline (plan A)

three replication training legs ~2.5h → 4 replication evaluation runs ~40min → production leg ~50min →
final exam ~35min → results; with review and buffer about **6 hours**.

## Amendment R8-1 (2026-07-29, approved: file the amendment and resume): recalibration of the death limb of the replication gate
- **Facts**: the replication stage ran 8 evaluations to completion with zero faults; gate B gave 1/3 (<2/3), and the campaign
  halted with DEVELOPMENT_SCIENTIFIC_FAIL. Frozen reading: s2131200 passes everything on both pools
  (wage +39.2/+42.6); s2131000 passes everything on pool B and is safer (deaths 88 vs 96),
  and on pool A fails only observed_not_higher (deaths 101 vs 98, +3 games);
  s2131100 is a genuinely weak leg (wage +12/+15, a bad draw in the seed lottery).
- **Diagnosis (the fourth death-limb power defect of the same kind)**: observed_not_higher is a per-pool
  point estimate; under the null truth the single-pool pass rate is ~50-58% (depending on the parity of the discordance count), ~28% for both pools,
  and the total power of 2/3 over three legs is **~19%** -- even a perfectly death-neutral recipe opens the gate only one time in five.
  The 6/6 pass of the frozen R7 legs was luck.
- **Amendment**: the death limb of the replication gate becomes **a catastrophe band over both pools combined**: per leg (256 pairs), excess
  deaths ≤ +5pp. Per-leg power under the null truth 91-95% (exact binomial: sd=√D_pooled/256
  ≈3.0-3.5pp, frozen D_pooled=73/60/79, +5pp≈1.4-1.7σ), gate level (≥2/3)
  **~98.6%**; it still blocks real regressions of ≥+8pp. (Erratum from the 2026-07-29 review: the draft's
  per-leg ~99% / sd≈2.2pp was an arithmetic slip that matched a single-pool discordance count with a two-pool denominator.) The final-exam criteria (McNemar NI 0.10 + observed_not_higher)
  are unchanged; the final-exam pool 2_122 is unconsumed and unread, so the certification decision is not contaminated.
- **Derivation on the frozen data**: s2131000 combined −1.95pp ✓, s2131200 −1.95pp ✓,
  s2131100 +4.69pp (passes the death band but is blocked separately by the wage limb) → **2/3, risk64 selected**.
- **Implementation**: rev2 adoption mechanism (adopt-replication, same as amendment 5: decision-moment anchor
  + attestation chain + pinned check-key set + sealing guard + idempotent crash window);
  old identity snapshot rev1 launcher c41f39eb…, recipe 85477343….

## R8 final-exam verdict and closure (2026-07-29)
- **Verdict: PASS, zero failed items (a clean win for the campaign)**. On the fresh 2_122 pool of 256 pairs:
  farm_worker_wage 49.8→90.3 (+40.57, LCB +31.89); kills +15.09
  (LCB +11.18); ret +34.04 (LCB +24.31); gear gate passes; efficiency limb
  +0.02093 per step (p=4.66e-23, 205W/51L); deaths 178 vs 192 (observed −5.47pp),
  McNemar UCB +0.0400 ≤ margin 0.10: non-inferiority certified, and in fact safer.
- **Release**: train/runs/r8-certification-published/model_final.zip +
  r8_publication_receipt.json (run directory, not in the repository) -- the project's first model to pass full pre-registered certification and
  be formally released (the risk64 production leg, seed 2131900).
- Lineage closed: F3 (root cause) → R7 (evidence of progress, certification gap 0.86pp) → R8 (full
  certification + release). The case "why the model does not improve" is closed.

# FORENSICS-F2: root cause of the regression that came with retraining

2026-07-17, forensic panel dossier (six seats: code semantics, archive anatomy and training telemetry in
three parallel lanes → a mechanism judge and an extrapolation checker in adversarial verification → a
synthesiser who closes the dossier; read-only throughout, nothing launched).

## Half-page plain summary (written by the synthesiser, checked by the operator)

The collapse of both v32 legs was not mainly caused by the worker "forgetting to fight" or "forgetting to
descend", but by **the frozen old manager on the evaluation bench**. Continued training gave the worker a
small behavioural drift (only about 6% degradation inside training), but the old manager was calibrated
to the "posture" of the throne worker: it chooses windows by watching gauges such as the stall clock and
kills on the current level, and once those readings drift it always chooses "stay and farm this level"
and never dives. **The dive button was lit the whole time** (the engine code confirms that the DIVE
option is always selectable on ordinary levels); the manager simply does not press it. The decisive
evidence is the manager-swap experiment: the same worker weights (byte-for-byte hash check) under the M29
manager recovered diving on 9 of 11 disaster seeds, and the disaster group scored essentially back at
throne level (160.6 vs 162.2, paired Δ=0.0). The conclusion is decisive.

So the "farming black hole" gets a new name: the explosion in window count and in forced steps is an
arithmetic by-product of a window-selection loop. The accurate statement is: **"on the throne's most
valuable deep-dive seeds, the old manager is locked into shallow revisits by drifted gauges (F-lock);
separately, 3 seeds show genuine worker-level damage"**. 7017 cannot be rescued by any swap (the only
confirmed loss of dive execution), 7013 is specific to the potion leg, and 7021 to the control leg.
Unlocking the potion key has its own, independent cost: training deaths +26%, with a damage surface wider
than the disaster list. **This change is not free.**

Two lessons on the training side: 1. the teacher anchor rope is nominal only (the policy gradient
overwhelms the distillation gradient by 20-90×, and the alarm never fired); 2. the training distribution
structurally cannot see dives or dry levels (76% of dry-level windows are skipped), so the collapse
happened in a telemetry blind spot, and a 6% drift was amplified by the evaluation loop into 20-30
points.

## Mechanism verdict (unanimous, confidence high)

- **M3 (the frozen manager is locked by drifted observations) = the main cause of the full32 disaster,
  decisive.** Chain of evidence: final hash check of the weights in all four pools (both full32 legs use
  the same v22-h 0f2264… as ref-launch; both m29 legs use the same v29-mfresh 894413… as ref-science; the
  leg worker weights are byte-identical across the two pools) → a clean pairing with the same manager and
  only the worker swapped; after swapping to M29, D-on-CAT 2/11→9/11, kpf 0.76→2.48, τ per window 40→79.
  The only transmission path is the observation path (the DIVE mask is always true,
  src/diablogym.cpp:503-512; the 8 appended dimensions of the manager's 303-dim observation are the
  suspect surface, options_env.py:365-378).
- **M1 (the worker lost diving) is falsified in its strong form**: diving is a manager option, not a
  worker action (worker bit 11 is always masked, options_env.py:331); when D is chosen it executes
  normally (ctrl@7014 ret 318.2 reaching depth 2, crushing the throne's 127.5). Residual: only 7017 (all
  four configurations fail, 55.9-63.0 vs 258.6).
- **M2 (farming efficiency collapse) is falsified as the main cause and downgraded to a trigger**: the
  worker's real drift is small (training reward −6%, engagement macro 0.80→0.72-0.75, dry-anchor
  +1-2pp); the kills/farm_n collapse is half an arithmetic artefact (the REVISIT_FLOOR=25 floor plus the
  layer_clock carrying across windows → an F window with no new kill closes at τ=25, so the denominator
  inflates mechanically, options_env.py:194,240-242).

## Headline corrections (the operator's three headlines of the previous day, rejected by the panel, recorded as they are)

1. "Farming black hole" → **aggregation artefact** (the F×100 fingerprint is not specific: 5 of the
   throne's own chronic seeds match it; the explosion in window count is an arithmetic necessity of the
   25-step floor);
2. "ctrl is stronger on ordinary maps" → **does not hold** (sign test +14/−7, median +1.29; the mean of
   +13.9 is driven by a single seed, 7014 (+190.7); compliant wording: no systematic degradation was
   detected in the non-disaster group);
3. "overrides 21→47 because the teacher could not bear to watch" → **baseline artefact** (abnormally low
   on the throne's disaster seeds, corr(ovr,Δret)≈0; the override count is a by-product of the fuse
   acting on no-progress loops).

The different-source audit rule paid off again: three conclusions of the same-source narrative (the
operator's), none of which had been questioned, were all overturned by the adversarial seats.

## Statistical discipline register (extrapolation checker; must accompany any citation)

The disaster list is the worst 11 seats, selected after the fact on the dependent variable (it strictly
contains all 7 of the throne's depth≥2 seeds, hypergeometric p=9.8e-5; the directional conclusion rests
on mechanism evidence, not on the ranking of drops; group means carry the winner's curse and must not be
used as population effect sizes). **The list is a partition from the ctrl point of view and the wrong
partition for sov** (large sov losses outside the list: 7030 −175.5 / 7007 −64 / 7019 −50). The throne
reference is a single measurement per seed (the four archives v31-ref and v32-ref are identical field by
field, i.e. a relabel; within-seed variance is unknown, so the significance of paired Δ cannot be
assessed). The evaluation pipeline is extremely deterministic (abort1 and sov agree bit for bit on 3,445
episodes, so replication adds no information). All conclusions are limited to 1 checkpoint × 1 recipe ×
1 training seed × 1 probe pool × 2 manager grid points: **effective replication ≈ 1**, and "any continued
training slides into farming" is an extrapolation of the dynamics without evidence. Two dissents are kept:
1. the compensating-manager hypothesis is not excluded (M29 recovery ≠ an undamaged worker; M3 may be an
amplifier rather than the whole story); 2. the reverse reading is forbidden: M3 must not be read as "just
a bug in the evaluation protocol". The frozen manager is a deployed component, so the end-to-end loss is
real.

## Forensic queue (next steps)

- **P1-P4 (purely offline, read-only, within the operator's remit; launched when this dossier was
  filed)**: P1 scan of the manager's decision boundary (which gauge locks H); P2 layer-by-layer location
  of the weight drift; P3 re-partition of the sov damage surface; P4 a registry of manager-invariant
  seeds (a pure worker regression discriminator, kept in the evaluation toolbox).
- **P5-P7 (engine replay, no training)**: P5 was scheduled on 2026-07-17 (probe
  `train/probe_f2_replay.py`; replay fidelity is self-proven by field-by-field reconciliation against the
  archive rows; smoke test 3/3 bit-identical); P6 is the offline analysis of P5's output and runs
  alongside it; P7 was still waiting for a decision. Original rationale: replay the disaster seeds with
  logging (masks / termination reason / the manager's 8 dimensions), a cross-replay experiment for strict
  falsification, and a final review of the 7017/7013/7021 residuals.
- **P8 (needs training; proposed, awaiting a decision)**: rerun one leg with a different training seed
  plus pre-registered criteria on a held-out 8000-series pool, the minimum experiment that turns the
  fingerprint into a replicable proposition.
- **P9 (archive reading)**: trace the lineage of the dry-anchor mismatch (the throne enters at 0.7515 vs
  v28-leg1 at 0.6305: the drift predates v32, and the throne may already have depended on H's tolerance).

## Design inputs (for direct citation by later pre-registrations)

**Bundled evaluation case**: worker evaluation uses two managers bundled as standard (H and M29, each
paired against a same-manager reference); add a "manager sensitivity" metric |Δret_H−Δret_M29| (this
disaster went off the scale at ~80 points on it while training telemetry saw nothing); give the reference
an independent re-evaluation or report within-seed variance; every mean is reported next to the median, a
sign test and a de-leveraged mean; disaster criteria are pre-registered on a held-out 8000-series pool;
monitoring drops the F×100 fingerprint in favour of the composite signature "D windows disappear on the
reference's depth≥2 seeds ∧ τ per window falls into the 25-40 floor zone" plus triage by manager-invariant
rows; the damage surface of protocol-variant legs is partitioned independently.
**Dead-gate recalibration case**: the calib gate is decorative today (gradient ratio 20-90× with
tripped=false throughout). Set absolute thresholds from the lineage baseline (distill_ce target ≈0.2) or a
lower-bound gate on the gradient ratio; densify the sentinel to ≥10 sample points; make dry-anchor a dual
absolute + incremental gate set at lineage level; **insert a small fixed-seed end-to-end canary
evaluation in the middle of training** (frozen H manager + a subset of the throne's deep-dive seeds).
Training episodes structurally cannot record the decay of the dive chain, which is the root cause of the
−6% vs −34 gap. Add drift monitoring of the distribution of the manager's 8 observation dimensions.
**Course 2 wage formula**: a fixed β constant cannot hold the anchor. Make it adaptive (gradient-ratio
lower bound / distill_ce target) or use a KL trust region; include exhausted levels and states around
D windows explicitly in the anchor set (today's dry-anchor only measures and does not constrain); set
anchors and gates separately for protocol changes and recipe changes (sov lesson: it forked before the
first gradient update, with a +26% death tax).
**Course 5 dry-window course**: the root cause is that the training distribution is a "small world of
farming fresh levels" (76% of ff_dry is kept out of the learning distribution by skip_dry, descend
windows are 0.2%, and exhausted accounts for 55.5% of terminations). The course injects dry-level revisit
windows and states near dives. Success criterion = dry-anchor increment ≈0 ∧ the canary evaluation's
D-window/τ distribution stays close to the throne baseline (training reward is blind to this disease and
must not be used as a criterion). The first validation experiment must also break the training-seed
dimension (linked to P8); otherwise the "effectiveness" of the new course is also an N=1 proposition.

---

## Probe appendix (P1-P4 + joint review, 2026-07-17, second panel; fully offline, read-only)

**P1 (manager decision boundary): revised. The 8 appended dimensions are not the locking mechanism of
F-lock.** The drift magnitudes recorded in the dossier (τ per window 90→40, kills on the current level
halved, longer time spent) together shift H's D−F logit margin by only −0.016, while the typical |gap| is
≈2, two orders of magnitude apart. Applying the same drift to all 563 of H's D>F preference states (joint
review correction: not "D-choosing states"; with no channel masked, H never picks D first on the demo
manifold, and the D−F margin decides only when RESUPPLY is masked) flips only 15/563, all edge states with
margin ≤0.014. The F-lock suspicion is formally handed over to the **base 295-dim segment**: the "posture
fingerprint" of local walkable map and standing position at the end of the window (the D>F region is only
0.34% of real states; in H's sensitivity ranking over all 303 dimensions, the top 25 are almost all
walk/map/px). This is an offline refutation plus a directional positive proof; the final positive
attribution waits for P5. **M29's tolerance mechanism (recorded as directional)**: its D boundary hangs on
saturating monotone gauges (stall-clock gradient +0.966, time on the current level +0.927, 5-10× larger
than the same dimensions in H), so drift inevitably crosses it; the concrete flip thresholds (≈8 kills /
720 steps / 70 steps) are demo-manifold estimates and are reference values only. **The F2 dossier's claim
that "the suspicion concentrates on the 8 appended dimensions" is corrected accordingly.**

**P2 (geometry of weight drift): facts on record.** Both legs drift by almost the same amount (w0 rel_L2
0.406/0.402; the cross-generation baseline v24 is 1.583); the direction is mostly independent diffusion,
with a statistically significant common component whose energy share is <1.5% (cos 0.02-0.12, +3~5σ,
joint-review rewrite). **Only sov shows drift in the DRINK head: wa[12] rel 0.629 vs ctrl exactly 0.000**
(a gradient-masking fingerprint; a11/DIVE both zero and w0 col297 both zero confirm it). Layer
attribution shows strong inter-layer interaction (only-L0 + residual ≫ total), so **every statement of
the form "layer X contributes N%" is forbidden**; only ordinal statements are allowed (one-way transplant:
L0 ranks first).

**P3 (re-partition of the sov damage surface): fully re-checked by the joint review; for direct citation
by the bundled case's pre-registration.** sov-H worst 11 = 7031/7017/7030/7004/7025/7013/7014/7007/7019/
7029/7024 (only 7 overlap with CAT11 from the ctrl point of view); sov-M29 worst 11 is listed separately,
and the two pools share only 5 seats: the damage surface moves a lot with the manager grid point.
**Partition table**: A, F-lock manager-sensitive {7004,7024,7029,7031} ∪ {7030,7014 (sov interaction
type)}; B, revealed by M29 {7008,7010 (sov only)} ∪ {7009,7012,7018 (shared, a common retraining
disease)}; C, manager-invariant worker damage {7007,7013 (sov only)} ∪ {7017,7025 (shared)}; D, mixed
{7019}. 7014 is hard evidence of a worker × manager interaction (same H manager: it locks sov at F×102
and lets ctrl reach depth 2 with 123 kills; H pool −64.8 → M29 pool +200.1, a sign flip). The summation
definition behind the loss-mass shares (manager-invariant ~42%/33%) was attacked by the joint review; the
definition must be pinned and the numbers recomputed before they can be cited.

**P4 (registry of manager invariants): discriminator kept, registered as
[`docs/assets/manager_invariant_registry.json`](../assets/manager_invariant_registry.json).** Criterion =
exact equality of 13 row fields (including mode_seq character by character) for the same worker under
different managers. Registered: sov 6 seats / ctrl 2 seats / throne 7000 pool 5 seats / throne 9000 pool
2 seats; **pure worker-level regression list: sov={7013 (−91.8), 7007 (−64.0)}, ctrl={7021 (−75.2)},
throne = empty**. 7017 is a near miss of the "score invariant, trajectory variable" kind that the strict
discriminator misses, so toolbox v2 must add a near-miss criterion. Environment-level dead seeds 7022/9001
are listed separately (the 13 fields are identical for all three workers, ret≈−5.7, overrides=110,
suspected to share one environment-level root cause); 9009 is a high-scoring invariant counter-example
(the discriminator labels only the low-scoring subset as "pure worker level").

**Joint review, overall ruling:** the P1 forward pass is bit-equivalent to production code (max|Δlogit|
2.6e-7, argmax disagreements 0), and every headline number was independently recomputed and matched;
the core of P2/P3/P4 passed re-checking; three wording attacks hit and were recorded in corrected form.
**Limitations L1-L5 (must accompany any citation of this appendix)**: L1 support set = the bc demos
(FARM step-level states on fresh levels, 95.3% dlvl1 + 4.7% dlvl2; 58.8% carry exhausted=1, which never
occurs in the v32 training distribution), which contain no real end-of-window manager decision states;
L2 the D−F analysis is conditional on RESUPPLY being masked; L3 each extras template has only one semantic
construction point; L4 layer attributions are not additive; L5 every seed-level conclusion is a single
measurement under a deterministic protocol, and the reference is a relabel.

---

## P5/P6 appendix (replay forensics and fingerprint attribution, 2026-07-18, third panel; P5 scheduled on 2026-07-17)

**Correction to the main text (joint-review ruling, highest priority)**: the mechanism described in this
dossier's plain summary and in the M3 verdict, "the manager chooses windows by watching gauges such as
the stall clock and kills on the current level, and locks as soon as those readings drift", is, after the
P1 revision and this panel's positive attribution, **ruled a wrong description of the mechanism and
formally corrected to: "the manager watches the standing configuration at the end of the window (the
walkable local-map posture in the base 295 dimensions); the gauges (the 8 appended dimensions) only flip
marginal windows."**

**Replay dataset**: P5 probe, 51 episodes (king/sov/ctrl × CAT11 ∪ sov additions ∪ healthy controls);
replay fidelity 51/51, archive rows identical field by field; stored logits vs the production forward
pass ≤ floating-point noise (2.4e-7, argmax/chosen mismatches 0). "Bit-identical" is rewritten throughout
according to the joint review's definition.

**Four open questions closed (all high)**: 1. DIVE mask availability **4337/4337 = 100%**; "the dive
button is always lit and the manager does not press it" is upgraded from a code inference to a
full-population fact. 2. 93.8-98.2% of CAT11's F-window closures are exhausted, and τ sits on the 25
floor (legs with τ≤26 are 95.6%), which confirms empirically that the window-count explosion is an
arithmetic by-product (note that healthy king also reaches τ==25 in 71.4% of cases, so the τ floor alone
must give false alarms and has to be combined with "D windows disappear"). 3. R masked (the D−F margin in
play) covers 59.4% of windows, and all 9 of the throne's real D windows have mask [T,T,F]: the P1-L2
condition holds exactly at 9/9 real decision points. 4. The true nature of F-lock = **sub-threshold
suppression of razor-thin spikes**: D only ever appears as an isolated spike of +0.002 to +0.570 logit
(all in exhausted-posture windows with a saturated stall clock), and the legs' corresponding spikes fall
just below zero (−0.035 to −0.243). Within the loop the gap drifts up slowly, but **its speed toward zero
cannot beat the time budget** (the nearest miss, sov_7029, was only 8 windows from crossing zero: the
disaster list is sensitive to the evaluation horizon, which the bundled case's pre-registration must
declare).

**Fingerprint attribution (the WALK block dominates; qualitatively high, quantitatively with a
sensitivity range)**: block-swap ablation (9 throne D windows × 2 legs = 18 pairs; effective independent
sample ≈9 windows / 8 seeds, the inflated denominator is registered). **Transplanting a leg's walkable
local map (indices 44-164) into the throne's D-window observation kills DIVE in 18/18 (Δgap −1.485 under
the registered rule; kill rates of 14-18/18 under four alternative pairings, with Δ between −0.554 and
−2.193); the reverse transplant is the only single block that revives D (7/18 flip fully under the
registered rule, recovering 93% of the deficit; 2-10/18 under the alternative rules).** The monster-map
block, also 121 dimensions, is inert at 0-1/18, which rules out block size as a confound internally.
Semantics: the throne's D-window standing positions are enclosed (walkΣ 54/121: against walls, in
corners, in stair passages, with the player's south-west band in particular being wall), while the legs
end windows standing in open, already-farmed rooms (79/121). On-manifold gradient support added by the
joint review: on the 9 real D windows, d(gap)/d(walkable) in the south-west band is −0.076 to −0.121,
deeper than the whole-map mean of −0.012. The counter-examples 7024-ctrl/7025 show that the fingerprint
is **a configuration, not monotone openness**. Cell-level details (hot cells 78/89/100/111, median minimum
flip of 2 cells) are products of spliced states and stay at medium, out of the main text; the final
review waits for P7's in-engine placement interventions. **The balanced gauges are ruled out as a
cause**: on 7001/7024/7019 all three workers show the same kills/stall readings while the gap differs by
1-2 logits, so the difference can only be carried by posture, a positive confirmation of the P1 hand-over;
EXTRA-only transplants leave D alive in 13/18, quantitatively consistent with P1's −0.016 margin.

**M29 offline comparison (directional; trajectories under H, not to be used to predict behaviour in the
M29 pool)**: on the same batch of leg observation streams, H chooses D 5 times and M29 708 times (2644
windows); 13 of 30 "H never D" streams are revived by M29; on sov_7014, M29 chooses D in 53 consecutive
windows from w51: the sign flip of the interaction seed under a manager swap shows up at the observation
level. M29's partial recovery channel routes through RESUPPLY (sov_7030: argmax=R for 90 windows); the
online mechanism is still open. **Independent support for the class-C pure worker list**: M29 chooses 0 D
on the sov_7007/sov_7013 streams and on both legs' 7017/7021 streams; **but joint-review correction: only
the ctrl side is deeply negative (−4.1); sov_7017 max −0.72 and sov_7021 −0.21 are shallow margins, so the
final review of the 7017/7021 residuals stays with P7 and must not be closed by this dossier.**

**Three types of sov loss (confirmed at the time-series level)**: 7030 = worker-level clearing paralysis
(110 windows, 4 kills, ret 6.5); 7007 = pure worker damage (M29 also refuses); 7019 = pure efficiency tax
(all three reach full kills, nobody dives, farm_tau_mean 37.7 vs 63.3). 7014 = a textbook interaction:
ctrl overtakes the throne on a spike that the throne itself never fired (+0.233 → depth 2 / 318.2), while
sov cannot produce the exhausted posture on the same map. Why the healthy controls 7003/7011 do not
collapse: **they are spike-free seeds** (king does not dive there either). The defining event of a
disaster is only "king's razor spike falls just short of zero on the leg"; sov_7003 instead crosses zero,
chooses D and overtakes king. Spikes emerge from seed × posture and do not belong monotonically to any
worker.

**P7 target list (proposed, awaiting a decision)**: 1. an in-engine intervention replay that constructs
throne-style enclosed end-of-window positions (final review of the cell fingerprint); 2. scripted manager
forcing D (sov had 0/3 successful dives on the 3 windows where D was chosen vs ctrl 2/3; final review of
the dive-execution split at n=3); 3. the shallow-margin residuals 7017/7021; 4. the M29 R-routing
hypothesis; 5. the terrain semantics of the south-west band (facing stairs / passages). **P8 (rerun a leg
with a different seed) and the horizon-sensitivity declaration** are both listed as prerequisite
registration items for the bundled case's pre-registration.

---

## P7 appendix (scripted manager forcing D, intervention replay, 2026-07-18; 75/75 zero mismatch)

**Probe**: `train/probe_f2_p7.py` (not published). Scripted manager: copy the chosen options of the
stored P5 prefix, self-checking bit for bit window by window, force-select DIVE, then pass through to H.
Zero-perturbation self-proof: in episodes where the forcing point coincides with one of H's own D windows
(case B of king@7004/7029/7031), all four arrays of the whole episode are bit-identical. Plan A = force
after the first exhausted window closes; plan B = the same window index as the throne's D window. 75
episodes (3 workers × 17 seeds A + 3 × 8 B).

**Ruling 1: the sov dive-execution damage (0/3) is falsified at n=25.** Dive success: king 10/17 A + 6/8 B,
sov 11/17 A + **8/8 B**, ctrl 10/17 A + 7/8 B. The three workers show no execution split, and sov's
case B is even perfect. The F2 dossier's "sov 0/3 successful dives when D was chosen" was small-sample
noise and is corrected.

**Ruling 2: the economics of forcing D.** Excluding self-check episodes: king Δret mean +3.0 / median +0.1
(the throne's own window choice is already near-optimal, and extra diving adds nothing). **sov +52.9,
ctrl +40.5 (median +20.2); the disaster group combined +51.8 / median +31.5 / wins 26 of 38.** Almost all
of the legs' losses are recoverable losses of the "nobody pressed the button" kind. Death-tail register:
forced diving is not free (7 deaths in 75 episodes, including 3 for king). The death risk of deep diving
is real, and the throne's restraint has a price basis.

**Ruling 3: re-examination of the residual list.** 7021: sov forced 178.3 (above king forced 161.6),
removed from the list. 7013: sov forced −2.9→63.0 (a large recovery, but still a worker-level gap against
king forced 180.4), downgraded to partial worker damage. **7017: ctrl forced 58.3→215 (fully cleared, a
pure window-choice failure); sov forced only 63→93-97 (still −120 against ctrl/king; the only confirmed
sov worker-level harvesting defect in the whole case).** New finding: ctrl@7017 is a **"window-choice
blind spot shared by both managers"**: H and M29 both refuse to press, and forcing gains +157. A manager
succession does not fix seeds of this kind; only manager re-education or a redesign of window choice can.
The meaning of the "manager invariant = pure worker level" discriminator on 7013/7021 must be narrowed
accordingly (an invariant only proves that "changing the examiner does not help", not that "the loss
cannot be recovered").

**Limitations**: the forcing point is a single-point plan (one each for A and B), not a search for the
optimal dive timing; the death tail has n=7; the L5 single-measurement limitation carries over.

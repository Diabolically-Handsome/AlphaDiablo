# Zero-launch archive audit — prerequisite for design decision 13 (course 5 dry-window curriculum × 4b positive examples)

**Audit date**: 2026-07-18
**Discipline**: read-only on existing archives throughout: zero engine runs, zero evaluation runs, zero repository writes (the analysis scripts are not published). Archives audited: the sentinel.jsonl of the three legs, the P5 replay dataset (probe-f2-replay, 51 games per window plus per-window manager observations), BC demos.npz (166,383 pairs) and its three `_previous` generations, the G0 trajectory baselines (king/throne, skip_dry=False), window_econ_v23_probe.json, the eval-assembled full32 row-level archives and probe-zeroflip/report.json.

**Method pins** (for citation by the pre-registration): X in demos.npz is the 298-dim worker observation, with feature positions verified on data -- hp=X[0]∈(0,1], belt=X[286]×8 (strict 1/8 steps), τ clock=X[295], stagnation clock=X[296], exhausted flag=X[297]∈{0,1}; the replay npz holds the 303-dim manager observation (one row per window decision), stagnation clock at 296. Replay dry-window detection = intersection of a chained reconstruction of the exhausted flag (exhausted→set; descend/scene/levelup→clear) ∧ stagnation clock==1.0 at window open; among the three workers' 3,531 FARM windows only 7 (0.2%) are ambiguous, so the detection is reliable.

---

## 1. Dry-window signal surface (Q1)

### 1.1 Number and share of dry windows

**Training side** (final sentinel.jsonl records, +499,712 steps per leg; note that the training manager = v29-mfresh, as recorded in status.json):

| Leg | Learning windows (all fresh) | ff_windows | ff_dry | ff_dry/ff_windows | Windows met dry:fresh | Steps/learning window |
|---|---|---|---|---|---|---|
| v32-sov | 3,548 | 63,971 | 48,499 | **75.8%** | 13.67:1 | 140.8 |
| v32-ctrl | 3,588 | 63,386 | 48,251 | **76.1%** | 13.45:1 | 139.3 |
| b1-p8 | 3,553 | 61,896 | 46,386 | **74.9%** | 13.06:1 | 140.6 |

Item 1 of the evidence chain, "ff_dry 76%", is confirmed (it is the share of dry windows among fast-forwarded windows). The training learning distribution has dry=0 (skip_dry on), recorded as is.

**Evaluation side** (P5 replay, H manager, 17 seeds -- enriched with catastrophe seeds; the selection-bias note applies):

| Worker | Dry windows | Dry windows/game | Share of FARM windows | tau==25 | tau≤27 | Ended by exhausted | levelup | death |
|---|---|---|---|---|---|---|---|---|
| king | 805 | 47.4 | 93.8% | 56.0% | 91.2% | 97.1% | 11 (1.4%) | 0 |
| sov | 1,356 | 79.8 | 97.8% | 48.9% | 98.1% | 98.5% | 4 (0.3%) | 0 |
| ctrl | 1,252 | 73.6 | 97.3% | 78.4% | 95.4% | 97.8% | 9 (0.7%) | 1 |

**Unbiased full32 cross-check** (eval-assembled row-level farm_tau_sum/farm_n, back-solved as a two-point mixture with tau_dry=25.7 and tau_fresh from each leg's measured replay): v32-ref-launch (king) ≈95.0%, v32-sov ≈97.1%, v32-ctrl ≈96.1%, p8 ≈96.0% of FARM windows are dry -- consistent with the enriched-seed replay, so selection bias does not distort the magnitude. Third source, the G0 trajectory baselines (skip_dry=False, 7000-7015): king 96.8%, throne 96.4%. **Conclusion: on the evaluation side 94-98% of FARM windows are revisits of dry levels, about 47-80 dry windows per game.**

### 1.2 Window length distribution

Dry windows: median tau 25 (exactly the REVISIT_FLOOR), tau≤27 covers 91-98%, thin right tail (king max 569, ctrl max 600, one each); median beats 13-24 (the a9 engage macro takes several micro-steps per beat, so beats≠tau). Demos side (scripted teacher): dry-window decision pairs mean 16.0 / median 13. Fresh windows: median tau 337-364, median beats 152-167, consistent with training-side steps/learning window ≈140. **The measured band for s_dry (SB3 steps per dry window) is ≈[13,25]; the midpoint 20 is used for the extrapolation in section 4.**

### 1.3 Termination reasons

- Dry windows (replay, 3,413 windows over the three workers): exhausted 97-98.5%, end 1-1.3%, levelup 0.3-1.4%, cap ≤0.1%, death 1/3,413 (0.03%).
- Training learning windows (= fresh windows, consistent over the three legs): exhausted 54.9-55.6%, levelup 24.9-26.1%, cap 9.5-10.5%, death 4.9-6.7%, end 3.1-3.9%, descend 0.2%. The evidence-chain values "exhausted 55.5%, descend 0.2%" are confirmed.

### 1.4 Wage distribution -- **per-window wages are not available; three proxies recorded**

**Pinned unavailability**: at collection time probe_f2_replay.py:102-106 stored only opt/reason/tau/beats/overrides and **dropped** the five fields R/W/bonus/drains/dry that option_extra had already computed; the G0 trajectory baselines store only per-game wage totals (window_log is used only for assertions and never written); eval-assembled rows have no window-level fields; demos.npz has no reward. Recorded as gap a (see section 5).

**Proxy 1 (closest, strongest): window_econ_v23_probe.json** (2026-07-10, frozen H + scripted teacher, 7000-7031 argmax; the raw archive behind PREREG-v23 appendix A; F windows have no descend, so W≡R):
- **F/exhausted (n=2,205): median R −0.03, 93.6% of windows negative, R∈[−0.14, +62.59], mean +0.69** -- dry-level window wages = a dense slightly negative body (wall-facing penalty of −0.002 per beat) + a thin fat tail (when a kill really happens in the window);
- for comparison F/levelup (n=22): median R 35.07, none negative; F/death (n=3): mean −5.72.

**Proxy 2: per-game wages x dry share in the G0 trajectory baselines** (skip_dry=False, dry windows are learning windows): king per-game wage/step vs dry-window share corr = **−0.871** (throne −0.862); games with dry share ≥0.8 have median wage/step 0.0430 vs 0.1611 for games with dry share <0.8 (throne: 0.0709 vs 0.2113) -- as dry-window density rises, wage density falls almost proportionally.

**Proxy 3: mechanism decomposition of the reward function** (frozen code, env.py:1126-1161): inside a dry window without kills the only possibly non-zero terms are the wall-facing penalty −0.002 per idle beat, approach shaping ±0.005×Δd, ΔAC⁺×0.5 (a14 never occurs in the demos, see section 2) and death −8×dlvl (only 1 case in 3,413 replayed dry windows). **⇒ the theoretical wage band of a kill-free dry window is ≈[−0.05, ~0], matching the proxy-1 median of −0.03.**

**Decisive answer on whether survival wages are non-zero: not always zero, but structured as "93-94% slightly negative + a 2-9% fat tail".** Evidence of the fat tail: replayed dry windows with tau>27 ("eventful windows") are 1.9-8.8%, of which 4-11 windows per worker convert into a levelup; 3.4% of demo dry windows (269/8,011) clear the flag inside the window (= a real kill). This is the archive answer to item (b) of the pre-registration, "identifiable signal inside dry windows": **death penalties are extremely rare, clearing remaining monsters is a 0.3-3.4% tail event, and the dense body is the −0.002 wall-facing penalty**.

---

## 2. Dry-window action distribution (Q2) -- demos.npz

**Core finding (overturning an assumption on record in the design memo (not published)): the current BC-v1 demos are a degenerate single-class set -- 100% of the 166,383 pairs are a9 (engage), and the action distribution is the same in the exhausted state (97,916 pairs, 58.85%) and the non-exhausted state (68,467 pairs).**

- **a12 = 0, a11 = 0**, checked pair by pair (expected 0 ✓; consistent across three sources: the bc_worker.py:99-101 collection-ban assertion and the zeroflip re-check a12_wins_all=0/166,383 (obs_count and demos_sha256 match exactly)).
- **The claim "the 3A verdict says dispatch mostly presses 10/13 in dry windows" has no support in any existing archive**: the statement appears only in the design memo (not published) itself, and docs/design/DESIGN-gear-and-potion-autonomy.md contains no such sentence. Data: in the current generation a10=a13=0. Mechanism (code + observation data): in all 166,383 demo states the monsters list is non-empty (X[8]>0 always, median count capped at 50; PREREG-v23 appendix B records that "locked rooms mean dry levels always have monsters too"), so dispatch's a9 branch always cuts off a13/a14; a10 exists only as fuse-forced beats, and bc_worker.py:80-83 **removes overridden beats entirely** -- so a10 is removed from the demo set by construction of the collection rule. Replay confirmation: overrides mean 0.43-0.51 per dry window, and 39.6-49.3% of dry windows contain a forced beat (the teacher's real behaviour in dry windows = idle a9 presses + the fuse forcing a10 about once every ~25 identical-signature beats).
- **Generation drift recorded**: `_previous/1783858471…` (generation of 2026-07-11, 167,182 pairs) contains 4,885 a10 pairs (2.92%) and passed the two-class recall gate; from 07-14 on (protocol 3) all three regenerations have 166,383 pairs, pure a9. The a10 class died out between generations; the mechanism is that the progression_targets∧near>6 branch no longer fires -- when the pre-registration cites "the teacher's dry-window demo content" it must freeze the wording to the current generation (pure a9).
- **Making "monsters nearly cleared" precise**: in the exhausted state the nearest monster has median distance 6, only 0.6% within 1 tile (adjacent) and 17.4% within 3 tiles; in the non-exhausted state 15.7% are within 1 tile. Almost all a9 demos in the exhausted state are idle presses at unreachable or distant monsters -- **if λ_bc is not limited to the class-12 subset, it will amplify exactly these idle presses** (the data support the R6 mitigation direction of the design memo (not published)).
- Inside dry windows (exhausted at window open) there are 128,096 pairs (77.0%), of which 76.4% are in the exhausted state and 23.6% in states after the flag was cleared inside the window (window split by the tau fall-back method; leakage noise 62 pairs = 0.06%, which does not affect the magnitude).

---

## 3. Demo yield estimate (Q3, calibration of the 4b n₁₂ gate)

Trigger condition hp∈[0.5,0.65)∧belt>0 (teacher-v2 preventive drinking branch):

- **State level**: 5,672/166,383 = **3.409%** (2,356 = 2.406% within the exhausted state; 3,316 = 4.843% within the non-exhausted state).
- **Window level (main plan: hysteresis ≤1 per window ⇒ direct estimate of n₁₂)**: of 8,309 learning windows, **244 windows (2.94%) contain ≥1 trigger state** (216 dry / 28 fresh) ⇒ **n₁₂ ≈ 244 pairs per collection**. Trajectories before the first trigger are identical to the current archive, so this is a faithful point estimate under the hysteresis design; divergence after the trigger is unavailable in principle (gap f). The static upper bound without hysteresis is 5,672 (in reality far lower, because hp>0.65 after drinking leaves the band; interval [244, 5,672), the lower end is credible).
- **Meaning for gate calibration (confirms panel M1 and supplies the number)**: 244 < 300 ⇒ **the current ≥300 per-class recall gate at bc_worker.py:163 will certainly not engage for a12, so a separate recall gate is required**; the class-12 share is expected at ≈0.15% of pairs.
- **held-out coverage** (split fixed by the current rng(23), held_out_episodes in bc_report verbatim): the 13 held-out games contain 1,975 trigger states, 107 trigger windows and 5 trigger games (games 178/187 are the heavy contributors, 880/596 states) -- **a12 held-out recall is measurable under the current split; the denominator is non-zero**. Training side, 115 games: 3,697 states / 137 windows. Trigger games in total: 41/128.
- **Joint hp×belt histogram** (all pairs, share %):

| hp\belt | 0 | 1-2 | 3-5 | 6-8 |
|---|---|---|---|---|
| <0.35 | 0.05 | 0 | 0 | 0 |
| 0.35-0.5 | 0.06 | 0 | 0 | 0 |
| **0.5-0.65** | 0.04 | **0.99** | **2.08** | **0.34** |
| 0.65-0.8 | 0.10 | 4.98 | 3.68 | 1.30 |
| 0.8-0.95 | 0.09 | 15.52 | 15.23 | 2.16 |
| 0.95-1 | 0.11 | 32.36 | 18.50 | 2.41 |

- Completeness self-check: hp<0.5∧belt>0 = **0 pairs** (the drain-ownership assertion holds on the data); hp<0.5∧belt=0 = 195 pairs; belt=0 inside the band (cannot drink) only 72 pairs. There is no sign of an existing clustering distortion between the 0.65 threshold and the drain exit at 0.5 (the in-band distribution is smooth along belt), and the R12 bimodality risk does not show in the current archive -- but these are teacher-v1 trajectories; the knock-on effects of v2 remain an emergent unknown.
- Side evidence at window open (replay): dry windows opened with hp∈[0.5,0.65) are 10.0% for sov, 8.2% for ctrl and 1.2% for king -- the autonomy legs open dry windows inside the band an order of magnitude more often than the throne, so the 4b demos land exactly in the state region the new legs cross most often.

---

## 4. Empirical basis of the annealing table (Q4)

All extrapolation parameters come from archives: N_dry=ff_dry, N_fresh=windows (final sentinel record of each leg), s_fresh=140.8/139.3/140.6 (499,712÷windows; leg length checked via status.json start_steps=3,497,984), s_dry∈[16,25] with midpoint 20 (replay beats mean 18.6-23.9, demos decision pairs mean 16.0). The three legs give almost identical numbers; the v32-sov column is used as the main table:

| p_skip | Dry learning windows per leg | Dry share of episodes | Fresh-window gradient share (s_dry=16/20/25) |
|---|---|---|---|
| 1.00 (status quo) | 0 | 0% | 100% |
| 0.90 | 4,850 | 57.8% | 86.6 / 83.7 / 80.5% |
| 0.75 | 12,125 | 77.4% | 72.0 / 67.3 / 62.2% |
| **0.50 (main-table end value)** | **24,250** | **87.2%** | **56.3 / 50.7 / 45.2%** |
| 0.25 | 36,374 | 91.1% | 46.2 / 40.7 / 35.5% |
| 0.00 (OC) | 48,499 | 93.2% | 39.2 / 34.0 / 29.2% |

(v32-ctrl: p=0.5 → 87.1%, 50.9%; b1-p8 → 86.7%, 51.9%. The three legs agree within ±1pp.)

- **Reading for the main-table p_end=0.5: dry windows make up ~87% of learning episodes, but by gradient (steps) fresh windows keep ~45-56% (midpoint 50.7%)** -- "keep half of the oasis guard rail" holds exactly in the step-share sense.
- **Reading for p_end=0 (OC row): episode share ~93%, fresh-window gradient share collapses to ~29-40% (midpoint 34%)** -- deeper than "collapses by nearly half": it collapses to about one third.
- **One correction (for rewording in the pre-registration)**: the 76% in section 3 of the design memo (not published), "about 76% of learning episodes are dry windows at p=0", is actually **ff_dry/ff_windows (share of dry windows among fast-forwarded windows, 75.8%)**, not the share of learning episodes; the latter is **92.9-93.2%** at p=0. The conservative conclusion is unaffected (the real number is more extreme, which strengthens the p_end=0.5 argument), but the frozen pre-registration text should use the correct definition.
- Quantum check: 301,056+198,656=499,712 ✓ exactly the leg length; 147+97=244 rollouts of 2048 ✓.
- Extrapolation assumptions (to be stated with the frozen text): (1) the counts of windows met come from scripted runs under skip_dry=True; once the worker takes over dry windows, the manager loop dynamics may drift (gap d); (2) the training manager is v29-mfresh while replay/evaluation use H, so s_dry is borrowed across managers; (3) s_dry does not account for drains separately (gap e).

---

## 5. Data gaps (the honest-boundary section "archive audit appendix" of the pre-registration)

a. **Per-window wage distribution (dry/fresh separately)**: none of the three archive types has it. probe_f2_replay.py had R/W/bonus/drains/dry from option_extra in hand at collection time but did not store them (train/probe_f2_replay.py:102-106); the G0 baselines store only per-game totals. Closest proxies = window_econ_v23 (F/exhausted median −0.03 / 93.6% negative) + G0 per-game corr −0.87 + reward-decomposition band [−0.05,~0]. **If the pre-registration needs window-level wage points or bands, the replay probe must be extended to store those fields and rerun -- that is a new launch and needs separate approval.**
b. **Per-beat (action, wage) pairs in dry windows**: demos have no reward, and no archive has per-beat wages.
c. **Training-side termination reasons of dry windows**: sentinel reasons only count learning windows (= fresh windows); ff windows have no reason breakdown. The existing dry-window reason distribution comes entirely from the H-manager evaluation side; carrying it over to the v29-mfresh training manager is an unproven extrapolation.
d. **Non-stationarity of the ff_dry count**: with p<1 the worker takes over dry windows, which changes the sequence of windows the manager meets; the annealing table is a static extrapolation.
e. **Distribution of drains (reflex drinking) inside dry windows**: extras were dropped; only indirect traces exist (sov dry windows with tau∈{26,27} are 49.2% vs 17.0% for ctrl).
f. **Trajectory divergence after a teacher-v2 trigger**: n₁₂=244 is a faithful first-trigger estimate; after drinking, hp leaves the band and later windows drift, which a zero-launch audit cannot observe in principle.
g. **Manager mismatch confound**: demo collection manager = v22-H, training manager = v29-mfresh -- the 4b demo distribution and the learning distribution are one manager generation apart; this confound is recorded for the honest-boundary table.
h. **Cross-generation caveat for window_econ**: an archive from the 2026-07-10/v23 period, whose five runtime sha values belong to a different generation from the current code; any citation of the wage proxy must carry the generation-gap caveat.
i. **Canary dry-window behaviour instruments (item (c) of the design memo (not published)) are entirely missing from the current archives**: dry-state action entropy, dry/fresh distill_ce split and similar instruments were never recorded, so the baseline can only be built from this case's first launch.

**Three suggested revisions to the assumptions of the design memo (not published)**: (1) "the teacher mostly presses 10/13 in dry windows" becomes "pure idle a9 presses + fuse-forced a10 (already removed from the demo set)"; (2) "76% at p=0" becomes 93% (episode definition), or is relabelled as the ff definition; (3) "monsters nearly cleared inside the window, wages ≈0" is made precise as "monster list always full / 0.6% adjacent, median wage −0.03 with 94% negative and a 2-9% fat tail" -- the mechanism gap is real, but the non-zero signal surface (death penalty, remaining-monster tail events, wall-facing penalty gradient) is now quantified, so item (b) of the pre-registration can be pinned from the tables above.

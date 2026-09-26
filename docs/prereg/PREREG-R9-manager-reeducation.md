# PREREG R9: re-educating the manager of the certified team (the dive frontier)

> Opened on 2026-07-31. Approval chain (all 2026-07-31): R9 first, with follow-up problems handled afterwards →
> "repair the legs first, then retrain the manager" and "add a curriculum arm" (both the recommended options) → three switches (A: keep the old
> endpoints / anchor folded into R9 / full-set definition) → launch authorised directly once the design was finalised.
> Prerequisite: PREREG-EFIX is closed (the execution repair is committed).

## 0. Scientific question and root cause (conclusion of the foundation review)

The certified assembled agent (frozen M29 manager + the certified risk64 worker) reaches L3+ in only 7.8% of games (v28 baseline
20.7%) -- **the stronger the worker, the shallower the assembled agent**. Two root causes:
- **T-lock (time lock)**: with the dive bonus stripped from wages, the worker's optimum is to exhaust the current level; the median first forced
  hand-over comes at 1495 micro-steps (v28: 312), so a second dive is squeezed out by the 3000-step limit;
- **E-fail (execution failure)** (already fixed by E-fix): a limit cycle in the descend macro + a stagnation clock that only counts elapsed time;
  after the fix the stagnation rate went 75%→0%, depth 3 deeper / 5 same / 0 shallower.
F2's F-lock does not hold for the new team (under the certified worker M29 chooses D voluntarily, 12 vs 1).
**R9's question: if the only component that never received v4-world education -- the manager -- is replaced, can the assembled agent
learn "exhausted means dive"?**

## 1. Campaign structure (v29 election machinery skeleton, two arms)

- **Team**: worker = the certified release model_final.zip (rev26,
  dual-v4-asymmetric-v3, sha 2837288d…), frozen, connected through the staging path without a receipt
  (avoiding the mandatory release-receipt gate, the same precedent as the official R8 evaluation); baseline manager = M29 npz
  (sha 89441388…, legacy-v3 view).
- **Two arms** (the only variable = the deep-level curriculum):
  - r9-mfresh: MaskablePPO MlpPolicy(64,64) fresh, mppo 160k steps,
    lr 3e-4, ent 0.02, seed 22, raw-v4 manager view (fed the real clock of the new protocol);
  - r9-mcurr: same recipe + a deep-start curriculum (with probability p=0.5 a prologue plays to
    dlvl≥2, cap 8 windows, death re-draws capped at 8; distribution-assertion telemetry guards against an F3-style idle run).
  The curriculum is implemented as a training-side wrapper (the train_ppo._SeedDiscipline prologue);
  the protocol bundle is untouched.
- **Sequence**: preflight → the obligation left over from E-fix, G0-6 full-table REF_BITEQ (replay of the old endpoints on the
  512 games of the R8 final exam, reconciled bit for bit) → anchor burn under the new protocol → two arms trained (4h timeout per arm) →
  npz export + parity → arm exams → paired verdict → final exam of the winner → final verdict.

## 2. Pool allocation and anchor (the only new-pool consumption of this case; approved: anchor folded into R9)

Virgin ranges of the evaluation bank, checked: 2_114-2_119 ∪ 2_123-2_129. This case takes:
- **2_123_000-127 / 2_124_000-127**: two 128-pair replication pools (arm exams);
- **2_125_000-255**: a one-off 256-pair final exam.
Anchor = the new-protocol archives of the baseline assembled agent (M29 × certified worker) on the three ranges above; the gold pool 9000 and the
7000/8000/12000 held-out pools are untouched; training seeds automatically avoid every reserved range through the _SeedDiscipline
rejection domain.

## 3. Verdict (R8-level statistical infrastructure + where the v31-D3-10 leftover obligation lands)

- Paired statistics = a re-implementation of the same primitives as r8_statistics (pinned by per-formula numeric-equivalence regression;
  the archive cross-check contract is specific to the R8 geometry, so the dual geometry of R9 needs its own verdict layer), family-wise α=0.05;
- **Gate limbs tailored to a depth campaign**: depth superiority limb (primary metric, per-seed paired
  difference exact_sign+mean_lcb) / wage non-inferiority limb (LCB ≥ −10%×anchor mean; a manager trading farm-level
  time for dives is the expected behaviour, so it may not be failed for wage not being superior; a collapse is still blocked) / death limb
  (exact conditional McNemar non-inferiority 0.10 + an observed line derived from the anchor); kills/ret/worker_kills
  are demoted to record-only diagnostics;
- **The death line is derived on the spot from the new anchor; the absolute 6/32 line may not be inherited** (v31-D3-10: the death gate is
  systematically tight on deep-dive behaviour, with evidence from three cases; the R8 final exam measured the certified assembled agent at died −5.5pp);
- **Pre-registered primary metrics**: L3+ game count, depth histogram, DIVE success/stall breakdown, dive
  bonus conversion; all wage limbs kept from the full R8 set (manager re-education may not come at the cost of a wage collapse;
  the baseline wage is also a non-inferiority anchor);
- **Three fool-proofing instruments** ("depth unlocked" requires all of them to move; baseline values = foundation-review measurements):
  median first forced hand-over 1495 micro-steps / DIVE window success rate 25% (re-read against the new anchor after the fix) /
  dlvl1:dlvl2 residence ratio 16058:3512;
- The derivation formulas of the decision lines (ABANDON/FLOOR with anchor readings plugged in) and their values are written into the campaign ledger;
  a borderline verdict must carry the "borderline" note (verdict discipline inherited from v31).

## 4. Honesty and red lines

- The implementation/protocol-bundle changes of E-fix and this case have voided every old anchor; the new anchor is burned before any candidate manager
  exists, and candidates never touch the anchor pools;
- The five protocol-bundle files + eval_assembled/eval_contract/leashed_ppo/r8_statistics
  are untouched during implementation (touching any of them = the new anchor is voided again);
- The curriculum arm's prologue reward does not enter the manager's return (it completes inside reset); manager return =
  net gain after take-over;
- Gold pool / held-out pools untouched; the G0-6 obligation is settled with this case.

## 5. Timeline (converted from the measured anchor)

G0-6 ~35min → anchor burn (128×2+256) ~35min → two arms trained 3.5-5h →
arm exams 4×128 ~40min → final exam 256 ~15min → verdict in minutes;
**about 5.5-8 hours in total**.

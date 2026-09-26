# DESIGN: courses 3 and 4, reviving the equip key and potion autonomy (design memo)

Status: **desk document** (2026-07-15; date corrected in the v32 frozen volume). This memo gives
blueprints only, with no construction. Any environment-side change needs a decision first, then its own
pre-registration + critic panel + G0 parallel proof before work starts (construction-level discipline
of the [roadmap](ROADMAP-course-plan.md)).

## 1. Why now (chain of evidence)

- **v31 opening campaign**: the fresh arm scored 115.4, depth 2 on 12 maps, DIVE in the repertoire. The
  whole depth economy lit up, yet the arm was stopped at the eligibility gate by **died 7/32, one death
  too many**. Depth can be learned; lives cannot be kept.
- **v30 verdict**: FARM retraining raised deep deaths from 3 to 7 (the opposite answer is on record).
  Retraining alone does not solve survival.
- **World v3**: the back-jump confinement legitimately removes the "run back to a shallow level to heal"
  path; staying deep becomes rigid, and the survival toolbox is down to **potions** (a passive reflex)
  and **armour** (nobody wears it).
- **Conclusion**: the survival courses are on the critical path of the depth route; course 3 (armour)
  and course 4 (potions) are the two legs a deep diver like the fresh arm needs to survive.

## 2. Course 3: reviving the equip key

### Diagnosis of the current state (pinned to code)

- a14 is legal only when wearable gear is present (mask); the scripted teacher's farm branch presses it
  ([options_env.py:84](../../python/diablogym/options_env.py#L84)); shaping price = **+0.5×ΔAC on
  equipping** ([env.py](../../python/diablogym/env.py), v15 "lesson 13" bootstrap shaping: AC is a
  conserved stock, ΔAC>0 ⟺ gear was actually put on, it cannot be farmed, and the negative Δ from gear
  lost on death is not penalised).
- **Extinction fact**: in v28, a14 was called 0 times in 50,000 calls
  ([FORENSICS-winning-moves](../forensics/FORENSICS-winning-moves.md) #3).
- **Mechanism hypothesis H3 (unit-price mismatch)**: +0.5×ΔAC is a small one-off amount (one piece of
  armour has ΔAC≈2-5 → +1 to 2.5, about half a monster), while the real value of armour is
  **compounding survival at depth**. On shallow levels (0-2) damage is covered by potions and the reflex,
  so the learner cannot feel the marginal value of armour within its experience distribution. Control
  case: why did a13 potion hoarding survive? It feeds the 0.5 reflex, so its return is felt immediately.

### Options (smallest cut first)

- **3A diagnosis first (no environment change)**: counterfactual probe. A scripted worker (which equips
  armour) vs the same script deprived of a14, paired seed by seed on the 7000 pool, measuring "the
  causal value curve of armour V_AC(depth)" per depth stratum, plus a14 opportunity frequencies. If V_AC
  is significantly positive from level 3 on and ≈0 on shallow levels → H3 confirmed. **No protocol
  change; can be queued directly.**
- **3C BC positive-sample injection (training side)**: a14 samples exist naturally in the demos (the
  script presses it), but with fewer than 300 pairs they miss the class threshold; class weighting or
  oversampling lets the BC teacher learn 14 firmly, protected by the β leash during PPO. **No environment
  change, no protocol change.**
- **3B′ training-side shaping (small protocol decision)**: add an immediate bonus for a14 in the training
  wrapper only, **leaving the evaluation ledger untouched**, which avoids re-anchoring. Cost: a new
  train/eval definition mismatch (the very problem course 5 is removing), which must be registered
  explicitly.
- **3B evaluation price-table review (large protocol decision; not recommended in principle)**: changing
  the unit price in env.py changes the definition of the evaluation return → **protocol version bump →
  every R2 anchor is set again**. Huge cost; kept on record only.

**Recommended order: 3A diagnosis → if confirmed, try 3C first (cheapest) → if nothing moves, reconsider
3B′; 3B stays shelved.**

## 3. Course 4: potion autonomy

### Diagnosis of the current state (pinned to code)

- a12 is always masked ([options_env.py:327](../../python/diablogym/options_env.py#L327), "drinking
  belongs to the brainstem"). The reflex **hp<0.5 ∧ belt>0 → 12** has two implementations: inline in the
  scripted dispatch (:61-63, embedded in every mode), and the learned worker's wrapper `_drain()`, which
  drains every step (it can run across exhaustion / CAP / death; one worker step = one action step + a
  reflex drain at the tail).
- **The v12 ghost lesson**: making potion drinking an option once caused the ghost (anti-revival) bug.
  Turning it into a reflex was the historical fix, and any unlock must carry a dedicated ghost
  regression.
- **The learner has long been stocking ammunition for autonomy**: in-combat potion hoarding with a13 is
  the only divergence that really pays (peak 49.2%, divergence content ≈ a13+a10). Ammunition logistics
  are already self-taught; all that is missing is the right to pull the trigger.
- **Reading the fresh arm's deaths in v31**: the 0.5 threshold is too late for burst damage at depth
  (half HP on L3+ can be lost in a single step); drinking early on its own initiative is a survival
  lever.

### Options (three tiers, all environment-side, all need a decision)

| Tier | Content | Cut | Risk |
|---|---|---|---|
| **4C unmask with the safety net kept (recommended first step)** | Reflex unchanged (0.5 floor); only `m[12]` is unmasked | One line, options_env.py:327 | Smallest: behavioural lower bound = status quo (a frozen policy that never presses 12 is step-for-step equivalent, provable bit for bit in G0); upside = drink earlier to survive, at the cost of more potions used (which closes the loop with a13 hoarding) |
| 4B soften | Threshold 0.5→0.35 + unmask; the reflex becomes a fallback | One mask line + the threshold constant | Death-rate rebound, medium risk |
| 4A remove the net | Delete the reflex; 12 is fully under policy control | Two places: dispatch and `_drain` | The roadmap predicts "the death rate will rebound"; largest |

**Coupling decision**: the roadmap requires course 4 to be evaluated together with course 2 (wage
renegotiation). This memo proposes **4C first, decoupled from the wage renegotiation**: 4C keeps the 0.5
floor, so the lower bound of death exposure is unchanged, the wage formula (w_t = r_t − 8×ΣΔd⁺, single
source of truth DESCEND_UNIT) is not touched, and the coupling premise is not triggered; the wage
renegotiation moves to the worker-retraining case. This is a deliberate deviation from the roadmap clause
and was decided separately (see the roadmap's numbering addendum of 2026-07-15).

## 4. G0 parallel proof plan (common to both courses, required before freezing)

1. **G0-identity**: a frozen policy (never presses the newly unlocked key / never equips) on the old and
   new env, seed by seed on the 7000 pool, **bit-level reconciliation step by step** (trajectory, return,
   the ledger identity Σw ≡ window R − DESCEND_UNIT×ΣΔdlvl⁺). The lower-bound equivalence of 4C makes
   this proof especially clean.
2. **G0-ghost regression**: v12 lesion replay probe + an assertion that `_drain` semantics are unchanged
   (the reflex still backstops; drinking steps still pass through the fuse / clock / termination ladder
   as usual).
3. **G0-ledger isolation** (if 3B′ is adopted): training shaping must not enter option_extra or the
   evaluation ledger; asserted by a probe.

## 5. Acceptance criteria and risk register (frozen in each case's pre-registration; draft marks here)

- **4C success line (draft)**: in the deep-dive configuration, paired died falls significantly ∧ the mean
  does not fall. A failure also has value: "given autonomy but not using it" is itself a recordable
  conclusion.
- **Course 3 success line (draft)**: a14 usage leaves 0 ∧ the paired gain at depth is positive.
- Risk register: death-rate rebound (4B/4A), potion-logistics inflation (RESUPPLY pressure; shares a
  probe with the opportunity-frequency statistics of 3A), width regression (v26 lesson; keep watching the
  τ̄/depth distribution), a new definition mismatch (3B′).
- **Suggested sequence**: v32 = 4C (smallest cut, a direct answer to the fresh arm's deaths in v31) →
  v33 = 3A+3C combined → wage renegotiation (the rest of course 2) → course 5.

---

## Appendix A: verdict of the 3A diagnosis case (2026-07-15, probe-gear-value, no protocol contact)

**H3 is superseded by the deeper H3′**: the paired counterfactual (armed vs gear_available removed,
H × script, seeds 7000-7031) gives paired differences of 0 everywhere and bit-identical trajectories for
both variants, because **the armed variant actually executed a14 0 times**, while equipment
opportunities (steps with gear_available=True) ran as high as **464.7 steps per episode**. Root cause:
the priority order of the dispatch farm branch (story > clearing monsters > picking up potions >
equipping) starves a14; inside a FARM window there is always something of higher priority to do.

**Verdict**: the extinction of a14 is **teacher priority starvation** (the teacher never demonstrates
it), which comes before any unit-price mismatch. The premise of 3C (BC positive-sample injection), "the
demos naturally contain a14 samples", is **falsified**, so 3C is shelved automatically as planned. The
viable routes for course 3 change: 3D, renegotiating teacher priority (dispatch is a frozen pure
function, so changing it is a protocol-level action that needs a dedicated case), or 3E, synthetic
demonstration injection (training side, to be designed separately). Data:
`train/runs/probe-gear-value/probe_report.json`; probe: `train/probe_gear_value.py` (a monkeypatch
research probe that writes nothing to evaluation archives). Neither is published.

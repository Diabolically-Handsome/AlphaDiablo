# Course plan (roadmap, 2026-07-11)

Basis: the list of untaught skills in
[FORENSICS-winning-moves.md](../forensics/FORENSICS-winning-moves.md). The principles are unchanged:
**one campaign, one prescription**. Approval at roadmap level does **not** replace per-case
pre-registration plus a critic panel at construction level. Every environment-side change needs its own
G0 parallel proof. "Anchor follows the throne" is a training mechanism that cuts across courses 2-5
(double evidence: v27 "cannot settle" and v28 "cannot stay tethered for long"; a teacher anchor is a slow
poison for a strong policy that sits far from that anchor).

| Course | Campaign | Content | Prerequisites / risks | Status |
|---|---|---|---|---|
| 1 Depth economy | **v29** | Retrain the manager against v28-leg1 with a 4x budget in two arms (normal entropy / high entropy); primary metrics = number of seeds reaching depth >= 2, DIVE per episode, realised 8xN bonus | All infrastructure is in the repository (v25 election machinery); no environment change | Draft pre-registration out, under panel review |
| 2 DIVE for the worker | v30 | Let the worker learn inside the DIVE window (today a three-line script: belt <= 2 pick up potions / fewer than 3 monsters fight / otherwise descend) | **Prerequisite: renegotiate the wage-stripping formula.** The worker currently bears the 8xN depth-death penalty but gets no descent bonus; unless that asymmetry is removed, shirking is the optimum at depth. Environment-side change, registered separately. Worker anchor = v28-leg1 self-anchor (first use of "anchor follows the throne") | Queued |
| 3 Revive the equip key | v31 | Bring a14 back from extinction (v28: 0 presses in 50,000 steps); BC positive-sample injection or a review of the ΔAC shaping unit price | Diagnose first (cost-effectiveness on level 1 vs compounding at depth), then prescribe; changing the shaping price is an environment-side change | Queued |
| 4 Potion autonomy | v32 | Unmask key 12 + remove or soften the brainstem's 0.5-threshold safety net | Removing the safety net will push the death rate back up; evaluate together with course 2; environment-side change | Queued |
| 5 Dry-window course | v33 | Turn skip_dry into a curriculum (annealed ratio) or align the evaluation definition, to remove the train/eval distribution mismatch | Mostly a training-side change; it reverses the v26 oasis, so width regression must be guarded against | Queued |
| No course | — | RESUPPLY (the script is near-optimal); town economy (mana / spells / shops / gold have no action interface) | v3x+ environment engineering, a separate project | — |

Target configuration (staged): **learning manager (understands depth) x learning worker (understands
logistics)** = Mark-I.
**Mark-I recognition line (coupled to the throne, PREREG-v29 D3-7) = depth secondary verdict "learned"
∧ P29 taking the throne.** The secondary verdict on its own counts only as "mechanism unlocked", not as
Mark-I (this guards against over-claiming a result that does not launch).
Courses 2-5 stack case by case into Mark-II and beyond; each case's winner challenges the throne under
gold-standard evaluation discipline.
**Rematch right for A (manager entropy-balanced rematch): one time only.** This clause comes from the
direction review (not published) and is kept here so that it survives across cases. It may be used at a
suitable point after v30; anchor = the incumbent assembled agent, under the anchor-compliance rule of
PREREG-v30 D3.

## Era switch note (2026-07-12, external audit dossier registered at 86c7270)

The external audit ruled that the protocol-v2 era is **sealed as internally valid history** (every
same-engine paired comparison survives, including the throne's six consecutive terms at 97.2 and all
v25-v30 verdicts); the absolute thresholds and baselines of v24-v30 are **deliberately voided**.
Prerequisite sequence for the v3 season: **the re-anchoring case (BC base regeneration -> teacher recast
-> anchor re-measurement)**; no course campaign may launch before all of it is complete. The
re-anchoring series has its own numbering, **R1, R2, ...** (it does not use course campaign numbers). In
the table above, 3 = v31, 4 = v32 and 5 = v33 were provisional v2-era assignments; the actual campaign
number is fixed when each case's pre-registration is frozen. The A rematch right is deferred until the
new v3 anchor is in place (the clause is unchanged; anchor = the incumbent assembled agent as determined
by the compliance rule at that time).

**Numbering correction (2026-07-12):** v31 was assigned to "new-world re-education (the succession
question)" when [PREREG-v31-new-world-reeducation](../prereg/PREREG-v31-new-world-reeducation.md) was
frozen. The provisional campaign numbers of course 3 (revive the equip key) and courses 4 and 5
(v31/v32/v33) are all void; actual numbers follow each case's freeze, and the v31/v32/v33 entries in the
"Campaign" column above are historical only.

## Standing review rule (2026-07-12, N = 3)

**If a component or a conclusion passes three consecutive reviews with zero findings, an audit by an
agent from a different source becomes mandatory.**
Rationale: when nothing wrong can be found from any angle, that fact may itself be the problem.
Same-source detectors (self-review, self-built panels) share blind spots with the object under review,
so zero findings are weak evidence; anything that looks perfect gets questioned. Execution details:
1. Count review events (a panel pass, an acceptance check and a re-run each count once).
2. Once triggered, hands off: the operator does not touch the repository while the audit agent works.
3. Close through the acceptance panel (read the full diff -> re-run the regressions -> semantic replay
   -> classify).

Precedents: two triggered audits so far (one of them in AlphaDiablo on 2026-07-12), and both found real
issues.

## Numbering addendum (2026-07-15)

v32 was assigned to "potion autonomy (4C: unmask with the safety net kept)" when
[PREREG-v32-potion-autonomy](../prereg/PREREG-v32-potion-autonomy.md) was frozen. For the course-4 row,
the clause "evaluate together with course 2" was settled by the design decision of 2026-07-15: 4C goes
first, decoupled; if it fails, the decision turns it into a combined 4 x 2 evaluation case (the
three-way decoupling clause of PREREG-v32). The diagnosis for the course-3 row is out: H3′ teacher
priority starvation, and the premise of 3C is falsified; see the root-cause determination in
[FORENSICS-winning-moves](../forensics/FORENSICS-winning-moves.md) #3 and Appendix A of [DESIGN-gear-and-potion-autonomy](DESIGN-gear-and-potion-autonomy.md).

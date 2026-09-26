# R20: continuous progress after real resupply

The current goal is for the candidate worker to learn to descend and develop reliably, aiming to reach L7 continuously from a normal start; the long-term goal remains L16 and Diablo. A successful town trip, compatibility tests or a single arrival on L2 do not complete this goal.

This round starts from the saved R19 source, the R16 worker, the certified worker and the native library recovery snapshot. The new native build and the experiments live in a separate local work directory (not published); the `build/` in use and the certified model are not replaced.

## Explicit candidate recipe

`resource_service_policy=sustain-v2` is added; the default remains `legacy-v1`, which keeps the old 600-micro-tick flow. The new policy only accepts `l2-town-v1` and `full`, and the service cap in normal evaluation is 1500 micro ticks; this is a new parameter proposed from the R19 route timings, still to be tested in practice, and it does not pose as the old protocol. The FARM trigger stays at 3600, and the manager actions and worker observation size stay as they were; the diagnostic `ResourceCalibration` still cannot enter training.

1. Use real gold pickup and the existing L1→town→original L1 permission. First observe the Smith's stock and repair quotes, then talk to Pepin normally, close the first story dialogue explicitly and heal for free, and read the potion prices.
2. With the current native readiness and the observed prices, compare: keep and repair, buy a chest armor, put on an owned qualifying chest armor, or normally take off a helmet/shield/chest armor whose durability fails the gate; each plan includes the necessary repairs of the other equipment and the minimum amount of instant belt potions. The comparison covers one equipment change plus repair, then recomputes from the actual result; it does not claim a global optimum over all item combinations.
3. Unequipping uses the engine's ordinary backpack capacity and equipment recalculation and keeps the original item. It is chosen only if the native projection still meets the armor, damage, weapon and life requirements. The preview is only for planning; the actual descent always re-reads the current native state.
4. When a complete plan is affordable, execute the next item of the cheapest plan first and then observe again; when it is not, free improvements that lose no required attribute are allowed, and the remaining money buys the minimum potions. Every purchase, repair and swap keeps its real receipt. If not everything can be bought, the shortfall is still reported, and no descent is forced.
5. Every move re-plans around the real monster occupancy; when the way is blocked the character may melee normally, drink or wait. Each blockage gets at most 180 real micro ticks and stops after 48 consecutive micro ticks without progress; one attack and its animation settlement take at most 12 micro ticks and still obey the service and whole-game deadlines. No monster is removed and no collision is cleared.

## Verification and later learning

First complete the unit tests and the rebuilt native tests, then re-check the earlier low-maximum-durability chest armor and the dynamic path-blocking samples. The formal verification reuses the per-native-micro-tick level-change ledger: first reaching L2 within 6000 micro ticks, then surviving the full following 1800 micro ticks. The first town trip is no longer a stopping point; non-arrivals, resource shortfalls and deaths stay in the denominator, and insufficient observation cannot count as success.

The new policy gets independent engineering smoke runs with two models and exact replays, followed by a paired comparison with fixed models. On an engineering error, the effect trial stops immediately and no further runs are dispatched. The identities of the models, source files, native library and configuration are all sealed. Whether 1500 is enough for the added detours and combat has to be measured again.

Before training, real development reachability and DIVE learning samples are checked first: the current worker has no dedicated gold-pickup action and resupply is executed by a script; scripted purchasing cannot be called PPO learning economic management. The R16 weights and contract are migrated explicitly first, and candidate training is considered only after a finite engineering training leg passes; the general resumed-training drift allowlist is not widened, and the certified model is not replaced automatically.

The current native course still refuses L3 and above; under this protocol no amount of training will reach L7. Once real L2 ability improves, the course is extended level by level with calibrated entry conditions, counting continuous L1→L2→L3→L4→L7 within the same game while keeping the earlier levels' costs, monsters, drops and de-duplicated bonuses. A deep pre-equipped spawn or someone else playing the levels cannot count as a continuous clear.

The training interface, the old model's effective teacher state, gold and potion-consumption supervision and the later stage specifications are in a training-path note in the local work directory (not published). This file is the implementation and acceptance agreement; live results are recorded in the experiment reports.

## sustain-v6: gear swaps preserve the equipment readiness already reached

`sustain-v6` is a separate, explicit experimental version. It keeps `sustain-v5`'s ordinary helmet, shield and chest-armor resupply plans, real stock and repair, the cheapest affordable complete plan that keeps effective blocking, and the new 450-micro-tick gold-pickup command window and 1500-micro-tick service cap. The old `legacy-v1` and `sustain-v2/v3/v4/v5` keep their original semantics; v6 does not change the default service policy.

A new native configuration `configure_resource_protocol(..., preserve_equipment_readiness=True)` is added; it is off by default, can only be configured between games and requires `l2-town-v1` to be enabled. v6 also explicitly enables the ordinary armor scope. Only when it is on does the native `resource_state` contain `preserve_equipment_readiness: true`; Python verifies the marker at start-up and at the existing later native observations, without extra observation, sampling or advancing micro ticks. A native library without this capability should fail explicitly.

The fix targets a conflict already confirmed in practice: a14's whole-set score could accept low-durability equipment because of an AC increase, while the actual descent also requires the equipped finite durability to be at least 15. All five real replay cases were commits where the score rose, HP was unchanged and, after the swap, only durability failed; the evidence is in a replay review in the local work directory (not published).

Candidate visibility and the actual a14 commit share these native constraints:

- When the current complete equipment set already meets the four canonical readiness items `armor`, `damage`, `weapon` and `durability`, the complete set after the replacement must still meet all four. Whether they are met comes directly from the existing native readiness check; thresholds are not copied and no other score is invented.
- Life, belt potions or level being temporarily below standard does not cancel the preservation of equipment readiness already reached. While the current equipment does not yet meet all four items, the old upgrade rules still apply, keeping the early development path.
- The existing survival, life ratio, strict whole-set score growth and curse restrictions still apply. Equipment attribute dependencies, two-handed slot use and the actual swap result use the engine's whole-body recalculation; empty slots and indestructible items follow the canonical rules.
- Before a native commit everything is recomputed from the live character and target item. Having seen a candidate earlier is no permanent permission; a refusal must not change items, gold, random numbers, network operations or the character's equipment.

This internal boolean is not added to the worker input, and it does not change the 15 actions, the 13,012-dim observation, the reward, the FARM budget, the 6000+1800 real-micro-tick end points, the native descent hard gate or the L2 course boundary. v6's service recipe and migration identity must be distinguishable from the old versions; only an explicit migration of the fixed R16 weights may enter this new experiment, and old G0/v4c or v5 checkpoints may not be silently continued through the general drift switches.

The verification order is: isolated compilation, native checks of the real counterexamples and normal upgrades, Python contract regression, full-result and trajectory compatibility of the old protocols, then finite case verification with the fixed R16. Engineering fixtures, preventing one bad swap or replaying consumed cases are not independent effects or model learning results. Long training, certification replacement and course extension are decided separately; the current verification status follows the corresponding candidates and reports.

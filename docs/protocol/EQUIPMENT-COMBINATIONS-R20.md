# R20 real equipment combinations and an optional potion reserve

## Purpose and wiring

`SustainCombinationService` is a separate experimental resupply class that inherits from `SustainCompletionService`. The existing training CLI and the old resupply factory still select the old service. This fixed-model evaluation explicitly replaces the completion service class before the environment is created, and saves an `experimental-runtime.json` per game recording the new code, bridge, engine, combination groups and potion reserve target. The new groups must not be read as the old protocol on the strength of the old `sustain-v6` factory parameters, and it must not be claimed that the fixed model was ever trained under the new service.

The service uses the original gold pickup, walking, talking, closing story dialogue, real buying, equipping, unequipping, repair and level-return actions. Every real command still goes through the unified micro-tick settlement, and the round-trip rewards and the live descent guard are unchanged. Better combination planning is resupply engineering; it is not the neural network learning to descend.

## Native preview and normal trades

`preview_resource_equipment_combinations(sequences, max_gold_cost)` only accepts ordinary Smith items actually seen during the current legal town resupply trip and the identities of items currently owned. At most 256 plans with at most 8 intents each; unknown identities, out-of-range input, stale stock and invalid price budgets are rejected. The available intents are buy-and-equip, equip an owned item, unequip, and repair of the finally kept equipment.

The preview copies the character and simulates in order with the real storage, payment, swap and attribute-cascade rules; it also checks the actual backpack cells, where the old equipment is stored, fixed-point life, weapon/armor durability and the minimum cash reserve for four instant potions. `Source/stores.cpp` factors out the normal storage and payment functions for an explicit Player; the GUI and normal buying keep their original entry points. A failed plan provides no executable first command or full projection and produces no gold, item, RNG or network operation. Re-computing the preview can set the existing life/mana redraw flags, so it is not claimed that every GUI flag stays bit-identical.

The returned `valid` only means the simulated trade is legal; the service must still check the final native readiness shortfalls. After one real command it re-previews the remaining combination, re-locates backpack indices by item identity after payment and keeps the committed remainder of the plan, so it does not keep buying back the bad equipment it just took off. If the remainder becomes invalid after real spending, the resupply ends explicitly and reports why.

The Python candidate catalogue is bounded: at most 16 observed shop items and 12 owned ordinary armor pieces, enumerating single items, pairs, a single item plus at most two unequips, and a final repair; more than 256 plans is reported as truncation. Finding no plan only shows that this observed, finite catalogue has none; it is not proof that the game economy has no solution.

## Potion reserve comparison

`reserve_belt_target=0` turns extra potion buying off by default; 6 or 8 are separate experimental settings. The minimum readiness threshold is still at least 4 instant potions in the belt. Only after every real native threshold (equipment, repair, free healing, the minimum four potions) is met does it top up to the reserve target with the remaining gold, the real free belt slots and Pepin's current stock. Extra buying ends when money is short or the belt is full; no resources are injected, the minimum equipment budget is not spent early and no threshold is changed.

## What the time limits mean

This candidate keeps `completion-l2-v1`: a budget of 12000 micro ticks to first reach main L2, then a full 1800-micro-tick observation after the actual arrival; a gold-pickup command window of 900, a total resupply budget of 3000 and a cumulative FARM trigger of 3600. The total budget is the cut-off of the current bounded experiment, not the time limit of a final 16-level clear, and it does not require speed-running. The primary metric is kept for comparison, while deaths, alive-but-out-of-budget, resupply failures and external interruptions are listed separately. Whether to extend the observation time is decided later from whether real growth occurs; running out of time is not equated with insufficient combat power.

The frozen model's input size 13012 and 15 actions are unchanged; the old time observation is still normalized by 6000 and is zero after 6000, a compatibility limit to be handled by a later migration. Learning currently uses gamma=1 and no uniform fixed per-micro-tick penalty, but the old reward's semantics such as the decay of positive FARM reward after more than 300 micro ticks without a kill are kept, so it cannot be claimed that time has no effect on the model. Long-horizon growth, stall detection and deeper courses need separate evaluation.

## Checks in this batch

First the new combination code and the related old native trade, armor and equipment-preservation tests are verified. Then 2 fixed-model control games with the old service are run, which must match in full rows and action traces apart from the frozen identity labels. Once the controls pass, the combination service and the combination plus an eight-potion target are compared on the same batch of 8 development seeds: fixed model, the same hard gates and time protocol, at most 18 games, two processes, 360 seconds in total, zero training, no automatic retries. An engineering error in the effect run stops it immediately; the results are kept in a local report (not published), and freezing the source files is not proof that the effect passed.

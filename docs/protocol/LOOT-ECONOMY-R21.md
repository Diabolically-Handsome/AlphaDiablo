# R21: selling loot and a second resupply trip

This batch covers code implementation and engineering verification. Training, fixed-model effect trials, certification replacement and extending the course beyond L3 remain paused. Item creation, gold setting or durability setting in engineering fixtures is only used to check boundaries and cannot count as natural starting resources or as a clear.

## Separate identity and recovery

The work was done in a separate local work directory (not published); engine changes live only in its `engine/` subdirectory and isolated builds in `build-r*/`. The original working source, its uncommitted changes, the library in use, the frozen v7/v8 and the models were not overwritten.

A `snapshot/` directory next to it keeps the working source before the change, the frozen v8 source and the original engine's uncommitted changes; `preparation.json` and `reports/preserved-artifacts.json` record the identities. The new implementation uses the completed v8 source as an isolated carrier, but the purchasing algorithm keeps v7's `SustainCompletionService`; the paused v9 combat-ordering draft was not adopted, and the earlier combination resupply experiment was not made the default.

## Explicitly enabled, old protocol kept

The new service is called `sustain-loot-v1` and requires `l2-town-v1/full` and `worker_time_protocol="completion-l2-v1"`. OptionsEnv automatically matches native `loot_economy=True`, the ordinary armor scope and equipment readiness preservation. The new native flag is off by default, and the old service keeps the old behaviour. The new environment identity cannot be mixed into old checkpoints through the generic resumed-training drift options; this batch opens no new training recipe.

The manager still has 3 actions and the worker 15 actions with the original dual-v4 13,012-dim input. Full inventory, sellable floor items, quotes and receipts only enter the resource state and audits; no features are silently appended for the frozen worker. Picking up gear, selling and returning to town remain disclosed scripted helpers and do not mean the neural network has learned economic management.

The time rules keep the completed completion-l2-v1: a deadline of 12000 real micro ticks for the first L2, then 1800 of complete observation; the observation denominator is still 6000. Each resupply trip is capped at 3000; gold pickup and gear pickup share a new 900-micro-tick command window, and the original cumulative FARM budget of 3600 is not reset. The second town trip also consumes the same game's remaining time and does not extend the game out of nowhere.

## Item and selling rules

- With the new flag on, an a14 gear swap must put every replaced old item into the backpack normally; if the rectangular space is insufficient, both the candidate and the actual commit are refused. Old items are never destroyed, gold is never added and items are never re-issued to force a swap through.
- The first trip additionally picks up observed, currently non-upgrade, non-quest equipment that the Smith would buy. On arrival the real floor identity, position, visibility and capacity are checked again; the item is put into the backpack normally and not worn automatically.
- Selling only covers safe idle equipment currently in the backpack. Currently worn items, quest items, items natively judged to be upgrades and items bought but not yet worn are protected. Quotes are only obtained in a real Smith interaction, each bound to the current index, full identity and price; a sale compacts the inventory, and the next one reads it again.
- The GUI and Gym share the normal selling price and transaction core. The native rule of a quarter of the value, truncated to an integer with a minimum of 1 gold, is used; no paid repair or identification is done automatically for a sale. Successful income goes into the personal gold inventory; stash wealth is not used.
- Failures such as no money, no space, wrong identity or a stale quote do not change trade resources, random numbers or the network queue; the wait after a real refusal in the script counts separately as real micro ticks. With no time left or after the terminal state, new commands are not executed and no wait is counted.
- Sale income `gold_sold`, floor gold pickup `gold_collected`, purchase/repair spending and per-trip/cumulative ledgers are kept separately. Potential sale prices do not count toward the wallet. Cash changes from real deaths and incomplete observations are recorded separately, and no income is fabricated.

## The second resupply trip

The first trip keeps the original trigger: a cleared level or a cumulative FARM of 3600, with a native readiness shortfall. The second trip is limited to main dungeon L1: the first trip has really returned and settled, the character is alive and able to decide, a native shortfall remains, and after returning to the level there was actual resource consumption, growth or new sellable loot; a second haul is also allowed for known remaining resources that the current native observation reconfirms are still on the floor and actually fit. A real HP drop that causes a health shortfall can create a healing need; mere idling, a change of position or an already saturated FARM count cannot trigger it again.

At most two trips are started per game, and native code also limits the number of actually authorized town exits. The shop and inventory are read again on each trip; the dungeon's monsters, drops, exploration, collected gold and reward ledger are kept, and the first-arrival reward is not paid again. No readiness gate is lowered (level, AC, damage, 80% life, 4 belt potions and durability 15); a failed second trip or exhausted trips must not force a release either.

## Verification boundary

The engineering checks cover real keeping/pickup/selling/spendable gold, backpack capacity, two-handed swaps, identity and quote expiry, failure atomicity, default-off compatibility, second-trip conditions and third-trip refusal, terminal and micro-tick boundaries, per-trip cash ledgers and native refusal of under-ready level changes. The build and test receipts of the local work directory (not published) are authoritative for the results.

The known historical case 2129887, "old sword 30 + club 5 can cover a 30-gold shortfall", is used only as an engineering amount counterexample. This batch does not replay that game and does not claim that selling items or a second resupply has improved the arrival rate, survival rate or clearing ability from natural starts.

## Record of this batch

Engineering verification passed: 192 unique cases, 137 subtests. The final native build is `build-r4`; the detailed groups, failure records and boundaries are in the final report and validation file of the local work directory (not published). No training or model effect trial was started.

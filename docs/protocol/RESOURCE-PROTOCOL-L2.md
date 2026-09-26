# L2 readiness gate and native town resupply protocol

Version: `l2-town-v1`. Date: 2026-09-04. This protocol is the L1→L2 engineering and mechanism verification stage of the full-clear goal; it does not show that the model can clear 16 levels in a row, and it provides no calibrated readiness standard for L3–16. Every effect conclusion must come from zero-training probes on a frozen implementation; this file does not pre-fill win rates or training recommendations.

## Interface and default behaviour

`OptionsEnv`, `WorkerWindowEnv` and the training `make_env` accept:

```python
resource_protocol="off"          # "off" | "l2-town-v1"
resource_purchase_mode="full"    # "none" | "heal" | "potions" | "armor" | "full"
```

`train/train_ppo.py` and `train/eval_assembled.py` provide CLI flags of the same names: `--resource-protocol` and `--resource-purchase-mode`. The CLI defaults to `off/full`; with the protocol off, the purchase mode may not be changed on its own. The training entry point only allows enabling it in `--worker` or `--options` environments; exposing the flags does not mean training was started in this round. This round adds 0 training steps.

Enabling the protocol changes the native level-change permission, the resupply service and its bookkeeping; the worker still uses the existing 15 actions and the original observation view. The native `resource_state` is used by the manager, the service controller and audits, and no observation dimensions are quietly appended for the frozen worker. The protocol and purchase mode enter the experiment identity. The current service recipe id is `l2-town-paid-repair-v2`; evaluation metadata, the training contract and the probe configuration record each arm's stage order explicitly, so that `full` before and after repair is never called the same treatment; the source bundle includes `python/diablogym/resource_protocol.py`.

The key bridge interfaces are `configure_resource_protocol`, `configure_town_service`, `act_pickup_gold_at`, `act_talk_towner`, `act_dismiss_dialog`, `act_buy_store_item`, `act_equip_inventory_item` and `act_repair_equipped_item`. Buy and equip requests bind both the index and `seed_hi/seed_lo/create_info/base_id` and reject stale or misplaced targets. `probe_resource_*` are native test fixtures; zero-training game probes do not call these state-changing interfaces.

## The hard gate is a hypothesis to be tested

The final L1→L2 permission is decided by the C++ `EvaluateResourceReadiness` and the player's native level-change entry point; Python does not reassemble an approximate ruler. All current conditions must hold at once:

| Condition | L2 entry threshold |
|---|---:|
| Character level | ≥ 2 |
| Native armor class | ≥ 9 |
| Current maximum physical weapon damage plus the character's damage modifier | ≥ 6 |
| Current life | ≥ 80% of maximum life, using native fixed-point integer cross-multiplication |
| Instant healing items in the belt | ≥ 4 |
| Minimum finite durability of the equipped head, hand and chest items | ≥ 15 |
| A usable, unbroken equipped weapon | at least one |

These values are a first falsifiable hypothesis, not a safety guarantee. Life, hit chance, resistances, being surrounded and future resource consumption can still kill a character that meets them. The first actual smoke run exposed insufficient durability of the old sword and shield, so ordinary paid repair was added to the `full` arm as an approved change. Repair is still constrained by actual gold, native prices, equipment identity and the durability cap; when the standard cannot be met the concrete economic shortfall must be reported; it cannot be taken as the RL not wanting to advance, and the thresholds must not be lowered silently to get more descents.

The manager, the worker permissions and the native entry point enforce the same permission result. `cleared`, the FARM scene cap or `exhausted` no longer exempt an under-ready descent. `DIVE` includes story operations and returning from quest set-levels, so insufficient readiness cannot seal off the whole DIVE meaning; a new main-dungeon descent is checked separately by the native target type and depth. This round forbids entering L3+, town returns not authorized by the protocol and unsupported teleports/shortcuts.

## The resupply loop and its failure boundaries

The service only supports **main dungeon L1 → town → the same L1**, at most once per game. Resource handling starts when readiness is not met and the level is cleared or the cumulative FARM scene micro ticks reach 3600; a 140-tick idle/exhausted state is not a town-trip permission. The flow is: sweep the gold piles already seen, walk the native stairs, buy/repair according to the arm, heal at Pepin and return to the original L1. `armor/full` go to the Smith first; `full` buys and puts on armor as needed, repairs the items still equipped that block the durability gate, then heals at Pepin and buys potions with the remaining money. `heal/potions` only go to Pepin. This routing avoids walking back and forth between two NPCs; the 600-micro-tick cap is unchanged.

The service has its own total budget of 600 micro ticks, covering walking, dialogue, buying, swapping gear and action settlement. Every real `bridge.step` counts toward the game's time; `finish/complete` are zero-micro-tick service events that do not pose as wait actions of the learning worker and are not charged extra stall costs. Going to town and back does not pay the reward for levels already reached again, and returning to the original L1 keeps that scene's exploration/combat bookkeeping, so old monsters or old tiles cannot be used to collect first-time rewards again.

When the service returns without readiness being met, FARM may continue if there are still enemies to clear or executable potion-pickup/gear-swap opportunities; the one-trip allowance does not reopen. No executable improvement path, an unavailable shop, a path failure or a service timeout all keep a concrete resource failure reason; nothing is released automatically, and "survival" cannot be obtained by waiting indefinitely.

## What was actually fixed in the headless shop

This round shares the commit core of buying/repair in the standard text shop with Gym; the separate `visual_store` interface was not refactored in this batch. The headless paths for approaching an NPC, opening the shop, explicitly ending quest dialogue, buying and equipping armor from the backpack were completed. Pepin may first give the poisoned-water quest dialogue; the script must close the dialogue explicitly and then heal for real; it cannot set life directly.

Shop quotes are only available to the service while the corresponding NPC shop is open. Quote and capacity queries stay read-only; refusals such as a stale item, insufficient gold or no space consume no gold, do not change the stock and consume no RNG. A successful purchase goes through the native payment and placement logic; if armor goes into the backpack, it is then put on by its full identity, and the old armor stays in the backpack. When the armor gate is already met, unnecessary purchases are skipped. The potion arms top up to at most 8 healing items when the budget and free belt slots allow; the hard gate's minimum is still 4.

Repair shares the native price and payment core with the normal shop; it does not use the warrior's repair skill and does not change durability for free. Only items in the `full` arm that are still equipped, have current durability below 15 and can reach 15 by a normal repair are repaired; old armor that was swapped out is kept but not repaired. Quotes are only read at the Smith, tried in price and equipment-slot order, and lack of money or a native refusal is recorded as a failure reason. Repair counts/costs are reported separately from purchases, and the total gold spent includes both; Pepin's prices are not read remotely, and money for buying up to eight potions is not guaranteed to be reserved.

The scope does not yet include selling items, a town-portal economy, buying spells/weapons, other merchants or round trips beyond L2. The service is an explicitly disclosed scripted helper, and its behaviour cannot be counted as a learning result of the frozen worker.

## Five arms and a fixed-exposure evaluation

`train/probe_resource_protocol.py` uses the existing R16 worker and the consumed seeds `2114000–2114047`. The five arms share the hard gate, sampling method, per-game seeds, single-threaded Torch, worker view and manager rules:

| Mode | Behaviour |
|---|---|
| `none` | sweeps gold; no town trip, no buying |
| `heal` | sweeps gold and returns to town on the same route; free healing, no item purchases |
| `potions` | returns to town to heal and buys potions |
| `armor` | Smith first to buy/equip armor as needed, then Pepin to heal; no potions, no repair |
| `full` | Smith to buy/equip armor as needed and normal paid repair, then Pepin to heal and buy potions with the remainder |

Timed from the normal dungeon start, the budget for first reaching L2 is **6000 micro ticks**. After the first actual L1→L2 level change, a full **1800 micro ticks** of follow-up observation is given, so the absolute cap is **7800 micro ticks**. The clock is updated by every real native micro-tick callback; the town trip does not pause it, quest set-levels are recorded as separate scenes, and entering L1 at the initial start does not count as reaching L2.

The primary result is the joint success of "actually reaching L2 within the budget and still being alive 1800 micro ticks after the first arrival". Not arriving, resource failures and deaths before arrival are all failures; alive but with insufficient observation counts as unknown/censored, never as a success. If a complete probe shows such censoring, an engineering failure is reported. Actual L2 exposure, arrival time, readiness shortfalls, service costs, resource failures and the paired-seed success difference are reported as well.

The hard gate makes the samples entering L2 differ between arms. So each arm's "survival rate of those who descended" is only a post-selection description and cannot be read directly as the pure physiological effect of potions or armor. The five arms first compare the overall effect of the resupply strategies. Whether the pure potion/armor effect can be separated further needs a separately designed trial in which all branches reach a real, common state that meets the same hard gate.

The survival-truncation report of the old `probe_r17_deployment.py` has been corrected: a survival that terminates before 1800 micro ticks after the first descent returns `None`; the summary reports the censored count separately and excludes unknown values from the observation denominator. The existing archives were not overwritten and do not pose as re-run results.

## Isolated candidates and how to run them

The original repository and native library were snapshotted first. The work and verification ran in a separate local work directory (not published): a `snapshot/` directory keeps the rollback evidence, the final `build-v4/` is the isolated native library and `candidate-v4/` is the frozen code mirror. Earlier candidates and their failure states are kept for traceability. The candidate directory does not follow the source automatically; existing evaluation directories must not be updated, and later changes should create new candidate and output directories.

Run inside the candidate mirror (a frozen copy of the repository), without replacing the repository's `build/`:

```sh
PYTHONPATH="$PWD/python" .venv/bin/python train/probe_resource_protocol.py \
  --phase smoke \
  --output-dir reports/smoke-v4-reproduction
```

The smoke run covers 16 R16 game seeds, a game-by-game repeat of the same 16 seeds and an independent re-check by the certified worker on 16 games. Besides no engineering anomalies and identical game-by-game repeats, each of the two models must complete at least one actual L1→town→L1 round trip, with the service ending in `complete` with canonical ready. 16 games that never trigger the shop, or that all die, cannot falsely pass the coverage gate.

Only after a smoke run with the same implementation/native library/model identity is PASS may this run:

```sh
PYTHONPATH="$PWD/python" .venv/bin/python train/probe_resource_protocol.py \
  --phase pilot \
  --engineering-receipt reports/smoke-v4-reproduction/summary.json \
  --output-dir reports/pilot-v4
```

The output directory must not exist. The driver saves the configuration, native and model identities, per-game JSONL, repeated trajectories, the certified re-check, failure states and the summary. It records read-only the existing raw snapshot on entering town, changes in shop stock contents, and the micro tick/position at service stage changes; when raw does not export the RNG it is recorded explicitly as empty, and no extra observe is called. PASS means the corresponding execution/coverage gate passed; it is not a certification of model ability, let alone an automatic authorization to resume training.

## Verification status and registration of results

The full V3 test run had 932 passes, 1 skip and 443 passing subtests; ten groups of old-rule/environment anchors with 128 seeds each gave 1,280 result rows identical row by row to the original archives. V4 only restores the automatic equip sound effect of the standard text shop and keeps the original line endings of unchanged C++ lines; Python, the service recipe, the readiness gate and the training path are unchanged. V4's 17 native tests and 42 shop transaction tests were all re-verified.

The native tests cover actually stepping onto stairs in a given direction, immediate health changes, refusal atomicity when touching a town portal and town-return messages, Pepin's first story dialogue, buying/swapping/repair, failures without resource or random-number side effects, normal gold pickup, preserving state on returning to the level, and cross-game shop replay. The scope is the Gym-reachable entry points in Release, single player with Lua disabled; the unreachable debug teleport and multiplayer arena menus were not changed.

Whether the final model smoke run, the spot check of the old rules on the delivered library and the zero-training effect trial were cleared to start, and the raw evidence paths, are recorded in the implementation results report (a local file, not published). If the real loop coverage in the smoke run does not pass, the five-arm effect trial is not run, the thresholds are not lowered, and no training is started and the certified model is not replaced.

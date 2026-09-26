# completion-l2-v1: a time contract that puts reliable completion first

This version aligns the real time boundaries in training with the wide-time evaluation that has already been completed. The goal is still normal character growth and reliable progress; micro ticks are used for honest bookkeeping and a finite compute budget. It adds no time-based reward penalty, and it does not treat budget exhaustion as death or as a clear.

## Activation and identity

Pass `--worker-time-protocol completion-l2-v1` explicitly; the default `legacy` keeps the historical behaviour and contract fields. The first batch supports only `earned-dive-suffix-v1`, `l2-town-v1/full/sustain-v6`, the original 15-action space and the `dual-v4-asymmetric-v3` observation (13,012 dims). The new version does not replace the original certified model.

| Item | completion-l2-v1 |
|---|---:|
| Original remaining-time observation denominator / `--max-steps` | 6000 |
| Physical deadline for first reaching an ordinary L2 | 12000 real micro ticks |
| Cumulative observation time after the first actual arrival | 1800 real micro ticks |
| Gold-pickup command window | 900 real micro ticks |
| Total resupply trip | 3000 real micro ticks |
| FARM scene cumulative budget | 3600 real micro ticks |

The single source of all values is the immutable recipe in `python/diablogym/completion_clock.py`. The training contract additionally binds `worker_time_protocol`, `worker_time_recipe` and the implementation identity; an ordinary resume cannot cross time protocols, even when an environment restart or optimizer reset is allowed. Crossing protocols needs a separate, verifiable initialization receipt.

## Actual clock and boundaries

Every game runs the original normal start; formal micro ticks count from 0 once L1 is entered, and the inherited initial town navigation still counts toward wall time. Only the per-tick `engine_level/is_set_level` the engine already produces is consumed; the engine is not queried, moved or reset in addition. Quest scenes do not count as an ordinary L2.

The first actual arrival on an ordinary L2 must carry the native `accepted/pretransition_ready` receipt; the physical deadline is set to `first arrival tick + 1800`. An early arrival can shorten the old 12000 limit, and an arrival exactly at tick 12000 extends it to 13800. Entering/leaving quests or returning to L2 again cannot extend the time again. Actual death, a real resupply terminal and the existing terminal rules still take precedence; the game is not forced to continue to fill the window.

The dynamic deadline is published before the external observation callback and this step's terminal check. A level change/walk animation already committed by an ordinary action settles against the updated physical deadline, and resource commands keep their original tighter local budgets; there are no extra policy decisions or unbilled actions. At the deadline the existing idle safe truncation and busy terminal rules stay as they are.

The R16 prefix still has its own per-environment lifetime attempt/real micro-tick limits. An active prefix's physical deadline cannot be extended by the arrival callback; reaching L2 unexpectedly before the hand-over is an engineering error, which stops immediately and records the native ticks already spent and the incompletely settled window; it cannot pose as a complete prefix or as a success. After a normal hand-over releases the prefix deadline, the current full game clock continues, without resetting gold, character, exploration or the FARM budget.

## Compatibility and limits of the conclusions

The observation denominator stays at 6000, so the remaining-time feature saturates at zero beyond 6000; this limitation is recorded explicitly in the contract. New data may include that segment, but it cannot be claimed that the model input gained new long-horizon planning information. Currently the action meanings are not changed, a11 is not forced, no target-progress reward is added and the GAE parameters are not adjusted.

`sustain-v6` still describes the original real shopping/repair algorithm; the new time recipe separately overrides its historical 450/1500 budgets, and the service telemetry reports the actual 900/3000 and the time protocol name. The "prefer higher AC" protection experiment candidate was not adopted.

Engineering replay, identical initialization parameters and training quality checks each provide their own evidence; none of them equals stable descending or a clear. Effects still need reporting of all seeds' real arrivals, complete follow-up, deaths, unmet criteria and unfinished results. L2 is currently the course boundary; the goal of going level by level to L7 and finally L16 stands.

# R19: calibrating resupply time and economic reachability

This batch follows the next step of the 2026-09-04 implementation report: first measure the complete duration and resource shortfalls of a normal resupply, then form parameter recommendations. It is a zero-training diagnostic batch, not the five-arm effect trial, and it does not replace the L2 survival check.

## Fixed boundaries

- The frozen R16 and the certified worker are kept, with unchanged action and observation sizes; the native library directly reuses R18's final V4 and is not rebuilt, and the library in use is not replaced.
- The native readiness gate is completely unchanged; the production protocol still defaults to 600 micro ticks for resupply and 3600 for FARM.
- The diagnostic object is `ResourceCalibration`, which only `OptionsEnv` accepts and the training Worker rejects. The diagnostic records explicitly carry `formal_metric_eligible=False`.
- The extended resupply observation lasts at most 2400 micro ticks and the whole diagnostic at most 10000; it stops at the first actual return from town to L1. The real round-trip completion time is recorded whether readiness was met or not. Deaths, services never triggered, path errors and reaching the observation cap are recorded separately, and unfinished samples are never treated as finished.
- At the start the worker still sees the original budget of 6000 micro ticks. Only after the service has started and is about to execute purely scripted actions may the underlying environment's stop limit be raised to 10000; the worker's observation clock is unchanged, and any policy call after the service has started is an engineering error. If the character meets readiness without a town trip or reaches L2 early, this is recorded separately and not mixed into the resupply duration samples.
- Both trigger timings keep the production denominator 3600 of the FARM progress in the worker observation; only the control threshold changes, and the shared first 1800 micro ticks of trajectory must be identical. Deaths before and after departure are counted separately.

## Comparison design

1. Using the diagnostic interface's 600/3600 configuration, check the trajectories and key results of the original V4 on the 16 R16 seeds and the 16 certified seeds, verifying default compatibility.
2. Extend R16 on the same 16 seeds to 2400, replay the same seeds across games, and independently re-check the certified model on 16 games; compare item by item with the prefix of the old trajectories up to the original termination time.
3. Once the engineering checks pass, run a paired calibration of FARM triggers at 3600 and 1800 on the frozen R16 for 2114000–2114047. Both groups use the same diagnostic service cap and worker observation clock; the 3600-group records already produced under exactly the same identity and configuration can be reused. The certified model re-checks 16 early-trigger samples separately.

Independent games may use at most three isolated processes, each with one PyTorch thread. The model re-seeds per game and writes results in the fixed seed order; the process count goes into the configuration. The 16-game replay runs serially and is compared row by row, to check whether parallel processes affect determinism. An engineering error stops the rest of the queue.

These are all consumed development seeds. They are used to understand the mechanism and pick the next version's parameters to be verified, not as unseen-seed generalization or clearing certification. On an engineering error stop immediately and locate it; if the diagnostic time limit is too short the samples are right-censored, and the maximum of the "finished" ones must not be passed off as the cap the whole population needs.

## Records and interpretation

Real time, level changes, native gate receipts and trajectories are checked per micro tick. Economic snapshots are saved at the start of the service, on entering town, on the first visit to a merchant, on quotes or equipment/gold/belt changes, before and after trades and at the end of the service; already obtained raw data is reused, and the shop is not queried or refreshed in addition. Duplicate snapshots may be de-duplicated, and trade records reference snapshots and real receipts.

The economic analysis only uses quotes and character states actually observed, distinguishing unknown quotes, item/attribute restrictions, limited stock, no capacity, insufficient maximum durability and insufficient cash. The cost of normally buyable/repairable candidates is not the optimum of the whole game; without a merchant visit its stock or prices cannot be guessed. The cost of buying potions beyond the hard gate's minimum of four is listed separately and cannot be used directly to assert that armor would have been affordable earlier.

The report should give, for each trigger timing: the service trigger rate, the real start time, the numbers of completions/deaths/right-censored samples, per-stage durations, return-to-level coverage within the original 600 and the candidate observation budget, the readiness rate on return, the number returning within the original total budget of 6000, and explainable resource shortfalls. Meeting readiness on return is never passed off as reaching L2 or surviving 1800 micro ticks.

The work ran in a separate local work directory (not published), with a start snapshot, a frozen diagnostic mirror and the raw records kept separately. The final results are registered in this batch's delivery report.

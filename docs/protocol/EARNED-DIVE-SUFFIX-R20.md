# R20: a DIVE learning suffix after real preparation

This protocol adds an optional training mode, `earned-dive-suffix-v1`; the default behaviour stays the historical mode.
It changes where training samples start; it does not change the model network, the number of actions, the observation size, the native readiness thresholds or how a normal deployment starts a game.
The code and engineering checks do not mean the model has learned to descend, and they do not authorize long training, extending the course or replacing the certified model.

## The question to test

The completed G0 v4c training contained 8,192 learning decisions, of which 539 (about 6.6%) were DIVE.
So this is not the first time DIVE is enabled, and it does not add the existing a11 prior; what is tested is whether more learning opportunities after genuinely meeting readiness help.
Readiness preservation across gear swaps uses the verified `sustain-v6`. The earlier engineering seeds were chosen by outcome and cannot serve as independent samples of the new mode's effect.

## Preparation, hand-over and learning

Every attempt runs the original normal reset and initialization navigation, after which the fixed original R16 makes the actual FARM decisions.
The manager keeps using the existing resource protocol; gold pickup, town return, healing, buying, repair, gear swaps and returning to the level use the original real service flows.
The parent model identity is fixed by SHA256:

`7e31dc5402caed733443abb9fef383c877d93b3623b199ba4b5c6293092592f0`

The hand-over is checked only in a new DIVE window chosen by the original manager, after the original window opening, fuse recovery and animation settling.
The character must still be on main L1, alive and in a decidable state, the live native readiness must pass, and the original mask must allow the opportunity.
The hand-over keeps the same already opened window, native state, manager history, exploration and clocks; it does not reopen the window and adds no observation or native step.
If readiness is not met after the opening settles, the parent model plays that window to the end; it cannot quietly swap players mid-window when readiness is restored.

The hand-over is a one-time change in that game. Afterwards both FARM and DIVE are executed by the model being trained, including L2 combat;
RESUPPLY keeps the original service flow. The parent model does not play FARM after the hand-over.
Rewards, deaths and failures of the preparation phase only go into a separate ledger; they cannot produce learner transitions or enter the PPO buffer.
Real rewards, terminations, bootstrapping and delayed settlement after the hand-over keep using the original Worker rules.

## Finite budgets and random numbers

The caller must specify `--worker-prefix-model`, `--worker-prefix-max-attempts` and
`--worker-prefix-max-microsteps` explicitly. The two budgets belong to the whole running lifetime of each environment instance and accumulate across resets;
they are not allowances refilled at every reset. On exhaustion a sampling-budget failure is reported and the instance stops; successful starts cannot be screened indefinitely.

The micro-tick budget starts at the formal native clock after the original normal reset has completed:
`budget_start=after_normal_env_reset`, `bootstrap_navigation_in_budget=false`.
The original environment's initialization town navigation happens before the formal clock is zeroed, and this mode keeps that behaviour; it still runs every time,
and the wall-clock time includes the complete reset. The prefix micro-tick limit cannot be claimed to cover the whole engine cost including initialization.

During preparation only the underlying physical deadline is narrowed to the remaining allowance, and the boundary is reached with the original macro actions, animations and terminal settlement.
`OptionsEnv.max_steps` keeps its original value, so the time features the model and the manager see do not change.
After the hand-over the original physical deadline is restored; an external change to that deadline is rejected and never silently restored to a guessed value.
Even if the public statistics are cleared, the budgets and receipts follow the private lifetime counters.

The fixed parent model uses private Python, NumPy and Torch CPU random streams.
After loading, per-game seeding and sampled prediction, the caller's random state is restored; the learner's global thread settings are not changed.
In normal training the parent model and the learner use independent random streams. An engineering fidelity control can explicitly let the fixed parent model continue into the suffix
for step-by-step comparison; such a diagnostic does not count as model learning or deployment effect.

## Recipe and model identity

The new mode only applies to CPU Worker MPPO, `dual-v4-asymmetric-v3`, 15 actions, the 13,012-dim observation,
`l2-town-v1/full/sustain-v6` and the existing `adjacent-v1` recovery.
The R16/G0 priors and rewards are kept: a14=2.5, DIVE a11=2, a13=2, reward v4,
HP loss price=0.1, potion pickup bonus=2, no-progress timeout credit=zero,
descend escrow fraction=0.5, power=1.6, the live native gate.
The 6000 formal micro ticks, FARM 3600, the window clock settings and the existing separate gradient-clipping groups are kept.
No skip-dry, deep pre-equipment, calibration injection or external teacher loss is used.

The `worker_prefix` contract records the parent model SHA, the input and mask protocols, the private random numbers, both budgets, and the clock and hand-over semantics.
The old mode adds no such field and does not load the parent model.
An ordinary resume cannot cross the learning scope or change the prefix contract through lenient drift options.
Entering the new scope from the original R16 must go through an explicit resource warm start: keep all original weights, clear Adam,
count the new training steps from zero, and keep the original R16's historical steps as provenance information.
The optimizer and results of the earlier G0 must not be silently treated as a continuation in the new mode.

## Verification and next boundaries

The engineering checks cover old-mode compatibility, construction and contract rejection, private random-number isolation, injuries and scene changes at the hand-over,
exact exhaustion inside macro actions, budgets across resets, the failure ledger, same-window hand-over and second-level FARM after the hand-over.
A synthetic sampling check uses the real buffer to verify that only suffix data enters and GAE is correct, without running an optimizer update.
A real run first binds the frozen source, native library, parent model, seeds, budgets, scripts and comparison fields;
on any error further dispatch stops immediately and the failure record is kept.

Effect verification still needs a deployment evaluation with normal starts and complete lifetimes, checking whether early development is harmed.
The formal primary metric is still first reaching L2 within 6000 micro ticks and then surviving the full following 1800 micro ticks;
verifying only 64 suffix decisions does not satisfy that metric. The current course boundary is still L2 and has not been extended to L7 or L16.
Only after the code, the contract and a zero-update engineering check are complete is a concrete finite training budget with stopping conditions put forward for a decision.

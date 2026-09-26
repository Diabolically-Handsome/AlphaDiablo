# R20 single-batch update diagnostic protocol

## Purpose and scope

The previous full-run learning group completed 8,192 learning steps, and none of its four batches passed the original PG update-direction condition. The existing receipts are not enough to tell whether the full GAE objective got worse as well, and the end-point weights were not saved. This protocol prepares a new diagnostic batch that answers this question with verifiable before/after states; it does not treat local metrics as the ability to clear the game, and it does not change the original candidate release conditions.

This is a proposal for a new budget; preparing the code does not mean the run has started. It allows only one full-run FARM+DIVE learning group, 4 environments of 512 steps each, 2,048 learning steps in total. It still initializes independently from the original R16 weights, with an empty Adam and new counters at zero; it uses the original sustain-v6, adjacent-v1, seeds 2168000 to 2168003 and the original learning/reward recipe. There is no second group, no automatic rerun and no certification replacement.

The stage limits are 120 seconds for initialization, 300 seconds for training and 120 seconds for offline analysis; a timeout is handled as an engineering stop, and the processes created in that stage are cleaned up. The uncompressed members of each evidence package total at most 512 MiB. A different number of learning steps changes the total-progress fields, so this is a first-batch diagnostic of the same recipe, and it is not claimed in advance to reproduce the previous run's first batch step by step.

## How the evidence is kept

The training CLI uses `--diagnostic-rollout first-update-v1` explicitly; it is off by default. It only accepts CPU Worker MaskablePPO, candidate, a fresh resource initialization and exactly one complete rollout, excluding resumed training and calibration interruptions.

In `on_rollout_end`, after GAE is complete and before the optimizer updates, the callback copies the original sealed arrays and per-row environment receipts and saves the policy and Adam state. The arrays keep the time-major order `[time, env, ...]` and must not be mixed with SB3's later env-major order or random minibatch order. The capture logic adds no policy inference, gradient computation, optimizer update or random sampling.

Only when one complete update and its original receipts close is the corresponding end point saved in `on_training_end`. The before and after packages bind the implementation, initialization, step count and starting actor identity. Without complete sampling or a complete update no comparable after package is produced; a capture error stops the trial. The original final release acceptance then runs as usual, and failure states and non-zero exits are kept.

The outputs are `first_update_before.zip`, `first_update_after.zip` and `rollout_diagnostic.json`. The packages are explicitly marked `DIAGNOSTIC_ONLY_NOT_PUBLISHABLE` and hold plain tensor state_dicts, arrays and JSON; they are not ordinary SB3 checkpoints and cannot serve as a certified model. Existing files of the same name must not be overwritten.

## Offline comparison and interpretation

The offline process loads only the initialization explicitly bound for this run to construct the same policy, loads the before and after state_dicts separately, and runs inference with the same threads and the original per-time-step environment batch shape. It uses the original actions, masks and old_log_probs and first verifies that the starting probabilities close with the collected records; records are never replaced to force ratio=1.

Two fixed objectives are compared, in both cases counting a decreasing loss as an improvement:

1. **Clipped surrogate on the full sealed GAE**: uses the clip formula and normalizes the whole batch's GAE only once. It includes the effect of later rewards and the value, but differs from the loss in actual training, where each minibatch is normalized separately.
2. **Original immediate combat-reward reference**: `q = combat_effect * transition_reward`, using the batch-centred `-mean((q-mean(q))*log_pi)`. The buffer reward, which includes the TimeLimit bootstrap, must not stand in for the real transition_reward.

Entropy, value MSE and probability changes are recorded as well. The current per-row receipts have no FARM/DIVE label, and groups must not be guessed from actions or masks; this version provides no grouped statistics without supporting evidence.

| Observation | Next step it can support |
|---|---|
| GAE objective improves, combat reference gets worse | check whether the reference metric agrees with the long-term objective; do not directly judge training a failure or relax the thresholds |
| both get worse | examine credit assignment and the update process further; this alone cannot blame Adam, entropy or clipping |
| both improve but the direction condition still rejects | check the difference between the local direction at the starting point and the finite-step objective change |
| data, probabilities or update receipts do not close | stop interpreting the effect and fix the engineering error first |

Whatever the result, an objective on the same data cannot prove better independent game performance. This version does not save the actual minibatch order or per-step gradients, cannot decompose the optimizer's causal effects item by item, and does not promise exact resumed training. Stable descending, a complete L2 survival window and the later level 7 still need independent real-game verification.

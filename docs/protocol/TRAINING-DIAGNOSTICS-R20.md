# R20: keeping a diagnostic when an update completes but fails release acceptance

Applies to `candidate` training with a resource warm start. This change keeps the original candidate acceptance conditions and does not change the reward, the readiness gate, the model input, the learning steps or the ordinary resumed-training rules.

`train_ppo.py` now records the five original acceptance results in one go: the complete update boundary, the exact target step count, the migration/inheritance evidence, the original on-policy PG release condition, and the model structure and behaviour evidence. Error messages distinguish "the update did not complete" from "the update completed but failed release acceptance".

A diagnostic can be kept only when `learn()` returns normally, the target is exact, the update has been consumed and the inheritance evidence is valid. The diagnostic saving logic also checks the original per-batch receipts, consecutive step counts, the actual update count, an empty pending-receipt queue, object/metadata consistency before and after saving, and the inner checkpoint's CRC, finite parameters and optimizer data. Sampling in progress, a half-batch update, identity drift or bad receipts never yield a diagnostic model.

A state that meets these integrity requirements but fails the PG or model-structure acceptance can be saved as an exclusive `training_diagnostic.zip`. Its only outer members are:

```text
manifest.json
completion_failure.json
receipts.json
checkpoint/model.sb3.zip
```

The outer archive lacks the top-level `data` and `.pth` an ordinary SB3 checkpoint needs, so the current ordinary resume and evaluation readers reject the whole diagnostic package. The package is explicitly marked `DIAGNOSTIC_ONLY_NOT_PUBLISHABLE` and binds the implementation, counters, failure report and inner model hash. It is not a certified model and does not mean exact resumed training is possible; the inner part, once explicitly unpacked, is still model data, and later analysis should use a separately registered diagnostic process.

Serialization forms the inner model only in memory, and the temporary file on disk also uses the outer format. The final file is published without overwriting; an existing file of the same name is not replaced. A saving failure is written separately to `training_diagnostic` in `status.json`; the original acceptance exception and the non-zero exit are kept, and the callback, environments and run lock are still cleaned up. Normal candidates still take the original saving path, and `model_sha256` is never impersonated by the diagnostic hash.

72 targeted checks passed, covering consistency of the old and new acceptance booleans, main-flow success/rejection/exception cleanup, the real SB3 serialization path, rejection by ordinary readers, non-finite data, metadata mismatches, exclusive files and failure rollback. The tests use synthetic objects or data; no new Diablo training, game run or formal initialization was added.

This fix cannot recover the 8,192-step weights already lost when the earlier process exited. The previous round's first learning budget has been used up and the second group was not started; this file does not authorize a rerun or a larger budget. In future qualifying rejected runs, the diagnostic format keeps the end-point weights, optimizer state and receipts; it does not contain the complete rollout, the order of each minibatch or the per-step gradient trajectory, so it cannot be used to decompose the contributions of entropy, clipping and Adam to the update direction step by step.

The PG reference direction is a local audit quantity, not the direction of long-term game return; failing acceptance does not prove that game performance got worse.

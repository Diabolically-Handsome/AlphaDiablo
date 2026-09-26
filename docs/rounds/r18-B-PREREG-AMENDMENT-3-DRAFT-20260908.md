# R18-B pre-registration amendment 3 (draft, 2026-09-08; **not frozen, to be decided**): an admission branch in the exam tool for the weights-only warm-start lineage

The frozen file `r18-B-PREREG-FROZEN-20260907.md` (sha 7f948dd6…, digest of the pre-translation text) and amendments 1 and 2 are unchanged; this is a new file and a draft.

## 1. Cause

Run `r18-arm-a-loot-4` finished normally (1 048 576 steps, 1052 games, published `model_candidate.zip` as PRODUCTION_CANDIDATE, sha 20adb0df…).
When gates A–D were run per §3 of the frozen file, **the exam tool `train/eval_assembled.py` refused to load this worker**, and all six volumes returned rc=1:

```
eval_contract.EvalContractError: asymmetric Worker checkpoint has not completed deployable actor training, or is not at a complete PPO update boundary:
{'num_timesteps': 1048576, 'last_completed_rollout': 1048576, 'ppo_optimizer_steps': 27023,
 'warmup_start': None, 'warmup_until': None, 'warmup_expected_rollouts': 0, 'warmup_completed_rollouts': 0,
 'warmup_optimizer_steps': 0, 'actor_optimizer_steps': 27023}
```

(The message is shown in English translation.)

The same refusal holds for **any** artifact of this lineage: the mid-run checkpoint of run 4 (888 832 steps) and the formal published file of the B6 smoke (4096 steps) were both re-checked and refused.

## 2. Root cause

`state_closed` in `eval_assembled._validate_asymmetric_worker_runtime_state` requires critic warm-up receipts
(`warmup_start ≥ 0`, `warmup_until > warmup_start`, `warmup_expected_rollouts > 0`, `_critic_warmup_completed is True` and so on).
The R18-B worker is a schema/2 **weights-only** warm start (`r16-to-sustain-loot-v1-…-weights-only-v2`): it inherits the actor and critic of the parent 7e31dc54,
this world has no critic warm-up period, and by design these fields are always `None/0/False` (the inherited-lineage branch of `LeashedMaskablePPO._assert_critic_migration_contract`
requires exactly this: it raises "resource warm-start contains current-world warmup history").

The training side's publication predicate (`train_ppo` `publication_eligible`) has a separate branch for the inherited lineage: a valid inheritance receipt + actor optimizer steps > 0; the exam tool has no corresponding branch.
The R16 arm went through a critic migration (with a warm-up period) and therefore passed; R18-B is the first weights-only lineage to enter the exam room, so the gap only showed up now.

## 3. Proposal (exam tool only; training/bridge/engine unchanged)

After the `state_closed` check in `_validate_asymmetric_worker_runtime_state` and before raising, add an inherited-lineage branch (the same criterion as the inherited branch of
`_assert_critic_migration_contract` and the publication predicate; the exam tool loads a plain `MaskablePPO` on the `require_published=False` path, so it is inlined):

```python
    inherited_resource = getattr(model, "_resource_warm_start_receipt", None)
    if not state_closed and isinstance(inherited_resource, dict):
        warmup_tuple = (
            getattr(model, "_critic_warmup_start_timesteps", None),
            getattr(model, "_critic_warmup_until_timesteps", None),
            getattr(model, "_critic_warmup_expected_rollouts", None),
            getattr(model, "_critic_warmup_rollouts_completed", None),
            getattr(model, "_critic_warmup_optimizer_steps_completed", None),
            getattr(model, "_critic_warmup_completed", None),
            getattr(model, "_critic_warmup_actor_sha256", None))
        try:
            from migrate_resource_candidate import validate_inherited_runtime
            validate_inherited_runtime(model)
            receipt_valid = True
        except (AttributeError, RuntimeError, TypeError, ValueError, ImportError):
            receipt_valid = False
        state_closed = (
            receipt_valid
            and warmup_tuple == (None, None, 0, 0, 0, False, None)
            and _is_plain_int(fields["num_timesteps"]) and fields["num_timesteps"] > 0
            and fields["last_completed_rollout"] == fields["num_timesteps"]
            and _is_plain_int(fields["ppo_optimizer_steps"]) and fields["ppo_optimizer_steps"] > 0
            and _is_plain_int(fields["actor_optimizer_steps"]) and fields["actor_optimizer_steps"] > 0
            and fields["ppo_optimizer_steps"] == fields["actor_optimizer_steps"])
```

This branch only decides "can it be loaded" and does not touch how row data is produced; non-inherited lineages (including the r9 certified worker and the R16 arm) take the original check, unchanged bit for bit.

## 4. Impact and re-certification

`train/eval_assembled.py` is in `PROTOCOL_SOURCE_FILES` (not in `_IMPLEMENTATION_SOURCE_FILES`), so the protocol bundle sha changes.
After merging, the following must be re-run: the full suite; the probe regression (16 seeds, rows sha 33023de1…); the re-cast of the six `r17-anchor-*` volumes (the r9 worker takes the original check, and its rows must be bit-identical to the existing six volumes);
the two-way re-bake 4/4 + 4/4. The training contract, warm-start receipts, bridge and engine are unaffected (the bundle does not contain this file).

## 5. Status of the provisional results

The six volumes were projected in a mirror root in a local work directory (not published; copies of `python/` + `train/*.py`, with `build` pointing at the incumbent bridge build-res) after applying the §3 branch to the copies (ledger `R18_B_GATES_ABC_PROVISIONAL`; the provisional results file is not published). In the first attempt the branch was written as a call to
`model._assert_critic_migration_contract()`, which raised AttributeError on a plain `MaskablePPO`; the error was swallowed and all six volumes were still refused; after switching to the inlined criterion they passed.
Not a byte of the main tree was changed; the provisional results are **not a verdict**, and whether to accept them is still to be decided. If §3 is merged and the §4 re-certification is complete, the formal re-cast of the six volumes should be bit-identical to the provisional rows (usable as a self-check after merging).

## 6. To be decided

1. whether to accept the provisional A–C results as R18-B's four-gate readings;
2. whether to merge the §3 branch into the exam tool (freeze this amendment → merge → §4 re-certification → formal re-cast);
3. a separate item (not part of this amendment): the **deployment form** of the earned-dive-suffix worker: the learner was never trained on level 1, yet in deployment it plays level 1 alone (today's E: level-1 deaths 25/48 vs 7/48).
   The results of the diagnostic composite form v0 (the parent plays until the first arrival on level 2, then the learner takes over; `gates/probe_composite.py` (not published), probe version string with `-composite-v0` appended) are in §7 of the [gates report](r18-B-GATES-REPORT-20260908.md);
   making it formal needs a new pre-registration (the hand-over rule should match training's "seven-condition qualifying DIVE window", not "first arrival on level 2").

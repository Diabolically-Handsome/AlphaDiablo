# R18-B pre-registration amendment 2 (2026-09-08): compact prefix audit

The frozen file and amendment 1 are unchanged; this is a new file.

## Cause
The relaunched arm `r18-arm-a-loot-2` (launched on 2026-09-07) crashed on 2026-09-08 with `OSError: [Errno 28] No space left on device`,
after 289 games and 253 952 steps (4 checkpoints: 63488 / 126976 / 190464 / 253952). Root cause: `PrefixAuditCallback` appended each environment's
**complete** prefix ledger (the whole attempt history + a full state snapshot at every DIVE window opening) to `worker_prefix_audit.jsonl` at the start and end of every rollout, so the file grew quadratically;
at 254k steps it had reached 6.2 GB and filled the disk, which is also why the final status write failed. The training maths does not depend on this file (pure audit IO).

## Fix (R18-B8, training side only: `train/train_ppo.py`)
`PrefixAuditCallback.compact_ledger`: at the start and end of a rollout only a compact receipt is written (counts + the last 2 attempts, the last 3 window openings of each attempt without state snapshots,
`ledger_form = compact-v1`); the complete ledger is written once at the end of training (`ledger_form = complete`). Tests: `tests/test_r18b8_prefix_audit_compact.py` (3 cases),
243 passed in the neighbourhood. The audit file was deleted (the ledger keeps a record of it); the checkpoints are kept.

## Identity
train_ppo.py sha256 20d1d362481315f2a2b1601c69aad721185e365a97a977d46325e899b97c1b3d
implementation_bundle_sha256 36188fda8553f1b49db331659df0027f9be7fee7083b33712c0322c24e4bd5fc
All other identities are the same as in amendment 1.

## Relaunch
First try an ordinary resume from `r18-arm-a-loot-2/ckpt/model_253952_steps.zip` (run name `r18-arm-a-loot-3`; the contract check decides on the spot);
if the check refuses (implementation fingerprint drift or the warm-start lineage rules), warm-start again instead (`r18-arm-a-loot-4`).

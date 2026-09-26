# R18-B pre-registration amendment 1 (2026-09-07): prefix guard

The frozen file `r18-B-PREREG-FROZEN-20260907.md` (sha 7f948dd6…, digest of the pre-translation text) is unchanged; this is a new file.

## Cause
The first arm, `r18-arm-a-loot`, crashed shortly after launch with `RuntimeError: completion L2 arrival before learner handoff`.
Root cause: coach-v03's six rules (without HP) let the manager open DIVE windows on level 1, while the "earned descent" hand-over requires the native seven-condition verdict (including HP ≥ 80%);
under the prefix rule a DIVE window that does not meet the hand-over conditions is played to the end by the frozen parent, which pressed the descend macro (or stepped on the stairs tile) and reached level 2 during the prefix,
and the completion clock failed closed as specified. None of the three 4096-step smoke runs (R18-B3b / M2 / B6) hit this gap; the first real run hit it in game 39.

## Fix (R18-B7, training side only: `python/diablogym/worker_env.py`)
`WorkerWindowEnv._prefix_guard_masks`: during the prefix, on main level 1, in a DIVE window that does not meet the hand-over conditions, the parent's action mask drops a11 and the direction keys that step onto trigger tiles
(the same function `_protected_walk_actions` as the frozen walking rule), with a0 as the fallback; other levels/scenes are unchanged. The hand-over definition ("ready idle after normal drain") is unchanged;
the parent's observation is unchanged; the deployment form (OptionsEnv, probes, exams) does not go through this path.
Tests: `tests/test_r18b7_prefix_guard.py` (5 cases) + 240 passed in the earned-suffix/r18c/B3b/B6 neighbourhood; the full suite and the 4096-step smoke are recorded in the ledger.

## Identity
worker_env.py sha256 dafa6d5bbe92c986edfee9ccae8a3f28aa5f716b668eecb357853fbe22a41dde
implementation_bundle_sha256 a13ae0dff4847cb9cd0783e99991a4e800c9f6e7a1687d33232acfabc23d8662
All other identities (bridge, engine, patch 0014, parent, probe) are the same as in the frozen file.

## Relaunch
New run name `r18-arm-a-loot-2`, new candidate directory, same budget as the frozen file (1 048 576 steps / 4 environments / checkpoint every 63 488 / 10-hour fuse).

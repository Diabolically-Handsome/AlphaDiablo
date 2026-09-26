# R18-G pre-registration: whole-map hunting off on level 2 and below (2026-09-07)

## 0. Background

The R18-D/E verdict: the aggro cap did not reduce the monsters nearby at retreat time (6.18 vs 6.14), because the crowd had been pulled before the cap took effect;
the pulling mechanism is a10's **whole-map hunt** (when the window has no visible monster, a whole-map BFS walks toward the nearest live monster, through closed doors, toward unlit monsters).
The suspicion raised in the design review, "it aggroes a whole level at once", points at exactly this mechanism. The T0′ verdict (2026-09-06) had already listed "hunting off on L2" as a cheap probe; it is added now.

## 1. Interface (default all = unchanged bit for bit)

`DiabloGymEnv(hunt_scope="l1-only")`: on main-line L2+ a10 no longer hunts over the whole map and only does local frontier exploration (the same as the old rule before R16);
L1 and quest set-levels keep the frozen behaviour (the L1 prefix is bit-identical to the retreat arm, so the pairing isolates only the L2 effect). Implementation: a single gate, `env._hunt_allowed_here(raw)`.

## 2. Probe

Driver `r10-staging/run_r18g_probe.py` (not published); pool 2_133 with same-seed pairs; worker 7e31dc54; clock completion-l2-r18c; manager coach-v03;
economy sustain-loot-v1; retreat retreat-v1; two arms: ctl `r18g-retreat` (a re-run of the R18-C/R18-D/E retreat-arm configuration) and G `r18g-retreat-hunt-l1` (+ hunt_scope=l1-only).

Criteria: regression (hard): the ctl rows are bit-identical to the R18-D/E control rows and the prefix up to the first L2 arrival is identical game by game in both arms, otherwise VOID.
Main hypotheses (reported, not judged; directions fixed in advance): H1 G's L2 hazard per 1000 ticks is lower than ctl; H2 paired saved−lost > 0 with UCB95 < 0;
H3 the mean number of live monsters nearby at the retreat trigger is lower than ctl; H4 G's L2 kills per 1000 ticks are not higher than ctl (pulling fewer = killing fewer, the expected cost); H5 L3+ arrivals and the level of the survivors.

## 3. Certification chain

Full suite with 0 failures (including `tests/test_hunt_scope.py`); probe regression equal; two-way re-bake (final bytes, done together after merging with R18-F); the sha256 of this file recorded in the ledger.

## 4. Seeds and files

2_133 consumes 2 more groups (16 in total); the virgin pools are untouched. New files: this pre-registration, `run_r18g_probe.py` and `r18g-probe/` (not published), `tests/test_hunt_scope.py`.
Changed protocol files: `env.py` (kwarg, gate), the probe (key, telemetry, version `r17-deployment-v3-r18g`).

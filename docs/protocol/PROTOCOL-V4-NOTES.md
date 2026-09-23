# Protocol-v4 notes from the R7/R8 period

These notes were moved out of the top-level README on 2026-09-23. They describe
the training/evaluation launcher, checkpoint, BC-gate and protocol-v4 rules as
they stood during the R7 combat-recovery and R8 certification campaigns
(July-August 2026). Later rounds (R9-R19) are documented in
[`../rounds/`](../rounds/) and in the other files of this folder. Paths are
relative to the repository root.

For ordinary research runs, `train/train_ppo.py` remains available. Do not use
it directly for the combat-recovery candidate: the R7 launcher freezes the BC
and implementation identities, compares the pre-registered recipes over the
fixed multi-seed development cohort, independently retrains the selected recipe
with the production RNG, and opens the 256-pair final gate exactly once. Only a
final PASS atomically publishes `model_final.zip` with its receipt; a direct
trainer/evaluator invocation is deliberately ineligible for that chain.

`train/run_v4_combat_recovery.py` is the archived rev12 launcher. Keep it only
for forensic replay of its historical evidence: it is not compatible with the
current rev22 training contract and is not a current publication entry point.

Training checkpoints are published atomically and only after a complete PPO
rollout update. `--resume-from` restores the policy, optimizer and global step,
then starts a new environment trajectory; it is parameter-state continuation,
not a bit-for-bit crash snapshot of engine state, wrappers, or Python/NumPy/
Torch RNG state. Contracted Worker checkpoints therefore require the explicit
`--allow-environment-restart-resume` acknowledgement and persist an immediate
parent/generation receipt. A checkpoint whose latest collected rollout has not
been consumed by an optimizer step is rejected. Checkpoints without the current
training contract instead require `--allow-legacy-resume` for an explicit
one-time migration; neither operation is an exact-trajectory claim.
`--total-steps` must be an exact multiple of `--n-steps × --num-envs`; the
trainer rejects any remainder instead of letting SB3 silently overshoot it.
Checkpoint-derived warm starts and teacher overrides retain a manifest, but the
manifest is not trusted on its own: the source checkpoint must still exist, its
hash and archive must validate, and every exported policy tensor must exactly
match the source. Keep the source ZIP beside any long-lived export.

BC `PASS` JSON is likewise treated as a claim, not evidence. The worker gate
requires exactly the 128 registered demonstration seeds, rejects actions that
are permanently masked for workers, and recomputes held-out metrics from the
hashed demonstrations and policy. Manager and flat gates deterministically
re-run their registered demo and 7000–7031 replay pools against the frozen
policy before training; the verified result is cached only for that exact
policy/runtime identity within the process.

Evaluation protocol v4 makes the native action and worker-window semantics
explicit. Action 0 is a real wait barrier (it cannot inherit a pending attack);
every non-terminal policy boundary is settled to `PM_STAND` with
`future == tile` and empty path/destination state, and the engine beats needed
to finish a committed animation are charged to the action's step budget,
reward delta and option duration. Entity masks use actual
visibility/reachability; damage reward is paid only for a new per-monster HP
low; and an ordinary FARM-window boundary is a non-terminal transition in the
same underlying game. FARM exploration may open an ordinary closed door when
it leads to new floor, but it cannot consume stairs or mandatory-story
interactions owned by the progression manager. Its 140-microstep no-progress
clock is independent of a scene-local 1,800-microstep cumulative FARM budget:
real combat/exploration resets the former, while the latter guarantees that
the progression manager still receives time to dive. Protocol v4 keeps the
298/303-wide shapes, but shape equality alone is not treated as semantic
compatibility. The frozen V28/KING/root actor and critic receive a canonical
protocol-v3 view: feature 286 is unpacked back to `belt_heals / 8`, feature 296
continues to carry the legacy no-kill clock, and feature 297 is decoded back to
the old `exhausted` bit. The frozen M29 manager likewise receives the legacy
feature-286 heal count, feature-296 no-kill clock and feature-298 layer-time
value. Current no-progress and scene-FARM budgets still govern window handoff,
but are not silently substituted into those frozen input slots.

For the current worker, feature 297 also carries a reversible drink latch.
Before any successful drink it remains the old `[0,1]` exhausted value; after a
successful voluntary drink or reflex drain it moves to the disjoint `[-2,-1]`
domain. Applying `-v-1` recovers the legacy bit exactly. The latch closes
worker-owned action 12 for the rest of that FARM window, while the emergency
brainstem drain remains available independently.
The existing base-observation belt scalar likewise carries both integer
preconditions without adding a column:
`belt_heals / 8 + belt_free_slots / 128`. The free-slot subscale perturbs an old
input by at most 0.0625 and cannot overlap adjacent heal-count buckets. Worker
drink capability depends only on visible HP/belt state in the `[0.5, 0.75)`
envelope and the visible latch. Repeated worker-owned drinking is therefore
hard-disabled after either kind of successful consumption, rather than left to
a hidden teacher history.

Potion execution is native-certified, not inferred from a key press. The
native action accepts only instant healing-potion item IDs (a Healing Scroll is
not a potion) and reports success only when HP rises or the matching belt count
falls. Worker receipts distinguish the requested action from the action that
actually executed, so rejected action-12 attempts cannot satisfy an exploration
or publication gate. The action-9 explore macro also yields immediately when
HP falls below one half and an instant potion remains, allowing the emergency
drain to act instead of monopolizing another macro beat.

The engage/explore buttons are resumable option controllers. Their failed-target
rotation, sticky frontier and visited-floor tables never change which policy
actions are legal; they only implement the chosen engage/explore action, and a
failed-target table must cycle rather than permanently blacklist a reachable
target. Consequently the 295-vector is not advertised as the full native game
state (unseen dungeon geometry is inherently partial). Encoding those controller
tables as a strict flat-MDP state would require new target/map channels and would
invalidate every existing 295/298/303 checkpoint; v4 instead exposes all wrapper
clocks/counters that directly gate masks or FARM termination while keeping macro
scheduling (including first-visit novelty) inside the explicitly partial option.

Protocol v4 retains v3's monotonic dungeon-depth and Lazarus bridge rules.
Because the task cannot naturally carry the Staff of Lazarus back to Cain, the
bridge performs one narrow, observable equivalent turn-in after the agent has
genuinely operated the stand and picked up the staff. Progression actions route
only mandatory interactions; they do not auto-clear combat or optional content.
The full-game resource probe exercises this chain against a real DIABDAT.MPQ.
The training/evaluation identity includes the actual main MPQ, complete
Resources tree, exact numerical package versions and native binaries mapped
into the process; the bridge rejects a main archive found through cwd/system
fallback instead of the explicit `data_dir`.

These semantic breaks intentionally invalidate all pre-v4 BC reports,
demonstrations, baselines and calibrated experiment thresholds. Re-run
`train/bc_worker.py`, regenerate protocol-v4 baselines, and recalibrate any
experiment driver before enabling it. Historical drivers remain fail-closed;
their old archives are immutable forensic records, not valid comparison rows
for new training or model selection.

Evaluation archives use schema v5 for exact contribution, potion-economy, and
curriculum-stratum accounting. Per-seed
returns are stored at full floating-point precision, and every episode records
its actual micro-step count and one mutually exclusive terminal kind
(`death`, `victory`, `game_over`, `time_limit_idle`, or
`time_limit_unsettled`). The manager ledger is partitioned into FARM and
non-FARM returns/kills; FARM additionally records `R`, worker wage `W`, stripped
descent bonus, and the wage/kills attributable specifically to
`_win_step_worker`. Opening brainstem drains and fuse recovery remain in the
whole FARM window ledger but cannot masquerade as reward delivered to the
learned policy. Each row also records voluntary drinks, reflex drains,
multi-drink windows, the maximum voluntary drinks in any FARM window, and the
ending healing-potion stock. FARM window counts, worker wage, and worker kills
are additionally partitioned into dry and fresh strata, so a curriculum/eval
distribution mismatch can be diagnosed without weakening the total paired
combat gate. Archive validation recomputes all aggregates and rejects any
violation of return, wage, kill, stratum, potion, step-budget, or terminal-kind
conservation.
Consequently an increase in total assembled return can be distinguished from
an actual increase in learned FARM-worker combat contribution.

### Protocol-v4 R4 audit outcome

The latest causal audit found no evidence that a fixed preventive-drink rule
improves combat strength. In native short training, the contextual action-12
mixture started at probability 0.05 and moved slightly downward; earlier
deterministic preventive-drink grafts also reduced return without reducing
deaths. R4 therefore treats the rev10 contextual mixture as an optional
on-policy exploration route, with `bc_aux_lambda=0.0`, rather than a behavior
that imitation learning or the publication gate must force into deterministic
argmax deployment.

This does not weaken the evidence chain. Before a candidate can publish, its
receipt must still prove at least 20 expected action-12 samples and at least 10
native-certified executions; requested-but-rejected key presses do not count.
The paired 7000–7031 gate remains the known-seed regression screening test:
worker wage and worker
kills must improve under the registered mean/seed-majority rules, total return
and total kills may not fall, deaths may not rise, and repeated worker-owned
drinking must remain absent. Fresh evaluation is not opened unless that gate
passes; the one-shot paired 12000–12031 pool is the independent final efficacy
evidence.

R4 is isolated from the earlier R1–R3 directories and artifacts. At the rev10
snapshot its interfaces were `bc-worker-v2-demos/4`, `bc-worker-v2/5` and
`bc-aux-behavior/7`, under training contract revision 12 and auxiliary
objective revision 10. That identity remains a forensic record; the rev13
data-firewall identity below supersedes it for all new artifacts. Older
receipts or campaign state cannot be relabelled as current evidence.

### Protocol-v4 rev13 final-heldout firewall

The rev13 audit invalidated the 1000–1383 BC-v2 replacement pool. A
preselection post-drink coverage diagnostic had traversed its final domain
before candidate selection, so the final split was already observed even
though the old failure receipt said that it had not been read. That pool is
burned. The fresh, disjoint registries are now exactly 2000–2127 for BC-v1 and
3000–3383 for BC-v2; consumers require exact episode coverage and reject the
previous pools.

Preselection coverage is restricted to fit and validation, and final coverage
and model scoring cannot start until candidate selection has passed. Collection
itself nevertheless traverses the whole registered pool, so both v1 and v2 now
create the immutable marker **before the first episode reset**, rather than
waiting for final scoring. It uses exclusive creation plus file/directory
synchronization in the stable sibling registry
`train/runs/_bc_final_holdout_registry/`; renaming or archiving either artifact
bundle cannot make the pool appear unused. The final PASS report binds both the
pool hash and exact marker-byte hash, and every consumer recomputes both. A
same-generator terminal receipt is also searched in the canonical directory
and every `_previous` archive. Pool identity uses its own immutable
`bc-final-holdout-pool/1` schema; changing the marker file format cannot create
a new identity for the same episodes.

The fresh BC ranges are part of the ordinary-training seed exclusion table, so
PPO cannot accidentally replay a BC final episode. The historical 0.70
fallback is disabled because it had no independent fresh registry and would
have reused the 0.65 final pool; any future fallback must register a separate
training-reserved pool first.

The registry is append-only across later campaigns. After the
`2_100_000..2_100_127` v1 and `2_101_000..2_101_383` v2 pools were opened,
they remained permanently burned and training-reserved. After the `2_102_000..2_102_127` v1 pool was opened by a collection run that crashed mid-traversal (2026-07-27, WSL port: the action-14 fuse path lacked its native gear receipt), that range too became permanently burned and training-reserved — the one-shot marker held, exactly as designed. The current active producer/consumer ranges are `2_104_000..2_104_127` for v1 and `2_103_000..2_103_383` for v2. Both are disjoint from the R7 evaluation bank
`2_110_000..2_129_999`; ordinary training rejects all four old/new BC ranges.
Changing the active registration does not itself collect an episode—the
one-shot marker is still created only by the explicit BC producer immediately
before its first reset.

These repairs first ran as the isolated R5 campaign. R4 is retained as forensic
evidence of its failed pre-repair BC attempt; it never opened either the
7000–7031 regression archive or the 12000–12031 one-shot fresh archive. R5
therefore used new control, candidate, and evaluation namespaces, while the
BC one-shot registry remains stable and global to the pool rather than to an
artifact bundle or campaign directory.

R5 later stopped exactly as designed when nested-validation root-anchor TV
reached `0.155435 > 0.15` at global step 3,782,656. It published no candidate
and did not open the fresh pool. Its last periodic safe checkpoint, at
3,747,840, had TV `0.074768`; a known-seed forensic replay passed every
regression component (worker wage `+20.685`, worker kills `+7.5`, return
`+21.132`, total kills `+7.813`, deaths `-1/32`). R6 therefore preregisters an
exact replay of the same first 122 rollouts from v28: 249,856 continuation
steps, unchanged constant optimizer/curriculum prefix, and an exact expected
policy-head SHA. It does not relax the TV gate or resume the tripped model.
The 7000 replay is adaptive development evidence; only the still-unopened
12000–12031 paired fresh pool can provide the independent final verdict.

The action-12 boundary circuit was corrected at the same time. Its open upper
edge is centered at the exact registered threshold 0.65 with slope 100; the
old `+0.002` offset had moved that center to 0.652 and classified real
0.651163 negatives on the positive side. The fit-only recall target is now
0.75. Calibration also fails closed unless legal-negative action-12
probability has mean at most `1e-4` and maximum at most `1e-3`. The contextual
adapter preflight separately binds the fit and validation positive-probability
minimum, mean and maximum to the registered 0.05 target, so an average cannot
hide statewise drift. Its liveness receipt now also performs one real isolated
policy-gradient optimizer step on nested-validation positives, proves that
eligible `p(a12)` and gate bias increase under a favorable advantage, and
restores policy plus optimizer state before training. Final checkpoint
publication additionally requires a persisted receipt that the last full
rollout actually completed at least one PPO optimizer step.

Current BC-v2 identities are `bc-worker-v2-demos/5`,
`bc-worker-v2/7`, and `a12-teacher-boundary/3`; the behavior receipt remains
`bc-aux-behavior/7`, while liveness is
`bc-aux-liveness-preflight/4`. The earlier `/4`, `/5`, and calibration `/2`
artifacts remain historical only and cannot be upgraded by editing metadata.

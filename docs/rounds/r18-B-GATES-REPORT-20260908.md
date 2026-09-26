# R18-B training-arm gates report (2026-09-08; run `r18-arm-a-loot-4`)

## 0. Conclusion (read this first; second version, with the provisional exams, the composite form and the a11 replay)

- **Training**: `r18-arm-a-loot-4` ran 1 048 576 steps and 1052 games, finished normally and published `model_candidate.zip` (sha 20adb0df…); the fuse did not trigger. Depth histogram of the training games {'0': 16, '1': 221, '2': 594, '3': 163, '4': 56, '5': 1, '6': 1} (221 games reached level 3+); escrow gate passed 1145/1270, vested 20377 / forfeited 26248 (one R16 run: 244/244 never vested; in this run escrow **vested at scale for the first time**).
- **Gate D** (identity) passes: against the r9 certified worker D1 0.650924 / D2 0.211894; against the parent D1 0.837778 / D2 0.511321 (threshold <0.99).
- **Gates A–C** (exams): the formal tool **structurally refuses** this lineage (a weights-only warm start has no critic warm-up receipts; the main tree is unchanged, see §3 and the draft of amendment 3). Provisional results in the mirror root (corrected branch, §7): A fails / B passes / C fails (demon volumes a/b deaths 53/44 vs anchor 41/31, UCB95 24.53/24.75; M29 volumes return ratio 0.947/0.8622, kill ratio 0.7844/0.7509); **whether to accept them is still to be decided**.
- **Gate E, the learner deployed alone** (§4): survival 17 vs 20 (saved 7/lost 10, UCB95 +0.203); level-1 deaths **25 vs 7**, level-2 arrivals 10 vs 34; the prefix differs in 42/48 (reported, not judged). Reading: learning windows are only on level 2 and above and level 1 is played by the frozen parent, so the learner never got level-1 data, yet the shared weights change: **level-1 ability regresses**, and deployed alone it collapses on level 1. The "pass" of Primary-1 (hazard 0.141) is an exposure artefact (level-2 ticks 21k vs 86k); Primary-2 and Primary-3 fail.
- **Composite form v0: the parent plays the prefix, the learner takes over from the first arrival on level 2** (§8; the prefix is bit-identical to the control in 48/48, so the pairing is clean): survival 19 vs 20 (saved 3/lost 4, UCB95 +0.111); level-2 hazard **0.1358 vs 0.222** (level-2 deaths 11 vs 19, level-2 ticks 81003 vs 85578, similar exposure: Primary-1 passes, and genuinely so); level-3 arrivals **15 vs 4**, 2→3 descents 23 vs 5, level 4 1 vs 0; level-3 deaths 10 vs 2, survivors among level-3 arrivals 4/15 vs 2/4; level-2 kills 418 vs 894, clvl≥4 17 vs 25: Primary-2 and Primary-3 fail.
- **F mechanism metrics** (§9; the replay is identical to the recorded rows in 45/48): a11 press rate in level-2 DIVE windows **13.9% vs 2.4%**, within 5 tiles of the stairs **24.1% vs 2.9%**; level-2 visits ending in a descent 7/17 vs 5/85.
- **In one sentence**: R18-B taught "descending" (escrow vested, a11 rate ×6, level-3 arrivals ×4) but not "staying alive after descending": deaths moved from level 2 to level 3 and total survival stayed flat; this is exactly one of the distinguishable outcomes announced in §1 of the frozen file (in the direction of counter-hypothesis one: depth up, level-3 survival rate 27% vs 50%). A separate finding: the earned-dive-suffix worker cannot be deployed alone (level-1 regression); the deployment form must be composite, or training must add level-1 retention.

## 1. Training side (run directory audit)

```
{
 "status": {
  "total_steps": 1048576,
  "target_steps": 1048576,
  "episodes": 1052,
  "sps": 29,
  "elapsed_sec": 36073,
  "training_ended": true,
  "publication_status": "PRODUCTION_CANDIDATE"
 },
 "dive_audit_last": {
  "r13_dive_audit": "v1",
  "step": 1048576,
  "dive_live_windows": 15403,
  "dive_live_steps": 871327,
  "farm_live_steps": 177249,
  "dive_live_descends": 1270,
  "dive_live_stalls": 6011,
  "dive_live_deaths": 104,
  "dive_live_a11_requests": 95608,
  "dive_live_a11_executed": 92386,
  "descend_escrow_ready_vested_count": 1145,
  "depth_shaping_credited": 0.0,
  "depth_shaping_refunded": 0.0,
  "descend_bonus_kept": 0.0,
  "descend_escrow_vested": 20376.93,
  "descend_escrow_forfeited": 26247.615,
  "descend_escrow_unready_denied": 8419.776,
  "hp_loss_charged": -15926.0,
  "potion_pickup_credited": 2410.0,
  "descend_escrow_gate_rows": 1270,
  "dive_share": 0.8309621810913086,
  "descends_per_10_windows": 0.8245147049276115,
  "final": true
 },
 "descend_gate": {
  "rows": 1270,
  "passed": 1145,
  "escrow_sum": 55140.2,
  "depth_hist": {
   "2": 898,
   "3": 284,
   "4": 82,
   "5": 4,
   "6": 2
  }
 },
 "episodes_all": {
  "n": 1052,
  "died_share": 0.562,
  "depth_hist": {
   "0": 16,
   "1": 221,
   "2": 594,
   "3": 163,
   "4": 56,
   "5": 1,
   "6": 1
  },
  "clvl_hist": {
   "2": 4,
   "3": 543,
   "4": 458,
   "5": 47
  },
  "kills_mean": 126.0,
  "reward_mean": 63.58,
  "len_mean": 993.6,
  "gold_mean": 166.1
 },
 "episodes_first200": {
  "n": 200,
  "died_share": 0.605,
  "depth_hist": {
   "0": 5,
   "1": 30,
   "2": 157,
   "3": 8
  },
  "clvl_hist": {
   "3": 65,
   "4": 114,
   "5": 21
  },
  "kills_mean": 137.3,
  "reward_mean": 73.65,
  "len_mean": 722.2,
  "gold_mean": 165.1
 },
 "episodes_last200": {
  "n": 200,
  "died_share": 0.53,
  "depth_hist": {
   "0": 1,
   "1": 48,
   "2": 123,
   "3": 21,
   "4": 7
  },
  "clvl_hist": {
   "3": 111,
   "4": 82,
   "5": 7
  },
  "kills_mean": 123.6,
  "reward_mean": 54.88,
  "len_mean": 1004.3,
  "gold_mean": 159.9
 },
 "sentinel_last": {
  "step": 1048576,
  "windows": 17203,
  "episodes": 1293,
  "reseeds": 237,
  "direct_terminal_deaths": 224,
  "transition_ff_terminal_deaths": 367,
  "direct_no_progress_timeouts": 270
 }
}
```

## 2. Gate D identity probe (r13-gateD-v1, no seeds consumed)

```
{
 "vs_r9_cert_worker": {
  "rc": 0,
  "probe": "r13-gateD-v1",
  "beats": 4000,
  "farm_states": 1461,
  "dive_states": 2539,
  "D1_farm_argmax_agreement_old_vs_new": 0.650924,
  "D2_dive_new_vs_script_agreement": 0.211894,
  "threshold": 0.99,
  "D1_pass": true,
  "D2_pass": true
 },
 "vs_parent_7e31dc54": {
  "rc": 0,
  "probe": "r13-gateD-v1",
  "beats": 4000,
  "farm_states": 1350,
  "dive_states": 2650,
  "D1_farm_argmax_agreement_old_vs_new": 0.837778,
  "D2_dive_new_vs_script_agreement": 0.511321,
  "threshold": 0.99,
  "D1_pass": true,
  "D2_pass": true
 }
}
```

## 3. Gates A–C exams (six r17-anchor control volumes, exam protocol in the old 3000-tick form, protocol off)

Formal tool attempt (`train/eval_assembled.py`, certified bytes): all six volumes refused.
```
{
 "xdevil-a": "eval_contract.EvalContractError: asymmetric Worker checkpoint has not completed deployable actor training, or is not at a complete PPO update boundary:{'num_timesteps': 1048576, 'last_completed_rollout': 1048576, 'ppo_optimizer_steps': 27023, 'warmup_start': None, 'warmup_until': None, 'warmup_expected_rollouts': 0, 'warmup_completed_rollouts': 0, 'warmup_optimizer_steps': 0, 'actor_optimizer_steps': 27023}",
 "xdevil-b": "eval_contract.EvalContractError: asymmetric Worker checkpoint has not completed deployable actor training, or is not at a complete PPO update boundary:{'num_timesteps': 1048576, 'last_completed_rollout': 1048576, 'ppo_optimizer_steps': 27023, 'warmup_start': None, 'warmup_until': None, 'warmup_expected_rollouts': 0, 'warmup_completed_rollouts': 0, 'warmup_optimizer_steps': 0, 'actor_optimizer_steps': 27023}",
 "xm29-a": "eval_contract.EvalContractError: asymmetric Worker checkpoint has not completed deployable actor training, or is not at a complete PPO update boundary:{'num_timesteps': 1048576, 'last_completed_rollout': 1048576, 'ppo_optimizer_steps': 27023, 'warmup_start': None, 'warmup_until': None, 'warmup_expected_rollouts': 0, 'warmup_completed_rollouts': 0, 'warmup_optimizer_steps': 0, 'actor_optimizer_steps': 27023}",
 "xm29-b": "eval_contract.EvalContractError: asymmetric Worker checkpoint has not completed deployable actor training, or is not at a complete PPO update boundary:{'num_timesteps': 1048576, 'last_completed_rollout': 1048576, 'ppo_optimizer_steps': 27023, 'warmup_start': None, 'warmup_until': None, 'warmup_expected_rollouts': 0, 'warmup_completed_rollouts': 0, 'warmup_optimizer_steps': 0, 'actor_optimizer_steps': 27023}",
 "xm29full-a": "eval_contract.EvalContractError: asymmetric Worker checkpoint has not completed deployable actor training, or is not at a complete PPO update boundary:{'num_timesteps': 1048576, 'last_completed_rollout': 1048576, 'ppo_optimizer_steps': 27023, 'warmup_start': None, 'warmup_until': None, 'warmup_expected_rollouts': 0, 'warmup_completed_rollouts': 0, 'warmup_optimizer_steps': 0, 'actor_optimizer_steps': 27023}",
 "xm29full-b": "eval_contract.EvalContractError: asymmetric Worker checkpoint has not completed deployable actor training, or is not at a complete PPO update boundary:{'num_timesteps': 1048576, 'last_completed_rollout': 1048576, 'ppo_optimizer_steps': 27023, 'warmup_start': None, 'warmup_until': None, 'warmup_expected_rollouts': 0, 'warmup_completed_rollouts': 0, 'warmup_optimizer_steps': 0, 'actor_optimizer_steps': 27023}"
}
```

**Root cause**: `_validate_asymmetric_worker_runtime_state` requires critic warm-up receipts (warmup_start/until, the expected rollout count, `_critic_warmup_completed`);
this lineage is a schema/2 weights-only warm start (inheriting the parent's actor+critic, no warm-up period), so these fields are always None/0; the formal published file of the B6 smoke was refused as well (re-checked).
The training side's publication predicate (`train_ppo` `publication_eligible`) has a separate branch for this lineage (valid inheritance receipt + actor steps > 0); the exam tool has no corresponding branch.
**No file** of the main tree was changed. Provisional approach: a mirror root in a local work directory (not published; copies of python/ + train/*.py, with build pointing at the incumbent bridge),
with the inherited-lineage branch added only to that function in the copies (the same criterion as the publication predicate), and the six volumes projected on the copies; the row-data generation path is bit-identical to the formal tool (the check only decides whether to load).
**Whether to accept this, and whether to merge the branch into the exam tool formally through pre-registration amendment 3 (which needs re-certification), is still to be decided.**

```
{
 "xdevil-a": {
  "rc": 1,
  "out": "$GATES_DIR/exam-root/train/runs/eval-assembled/r18-arm-a-loot-4-xdevil-a.json",
  "last_line": "eval_contract.EvalContractError: asymmetric Worker checkpoint has not completed deployable actor training, or is not at a complete PPO update boundary:{'num_timesteps': 1048576, 'last_completed_rollout': 1048576, 'ppo_optimizer_steps': 27023, 'warmup_start': None, 'warmup_until': None, 'warmup_expected_rollouts': 0, 'warmup_completed_rollouts': 0, 'warmup_optimizer_steps': 0, 'actor_optimizer_steps': 27023}"
 },
 "xdevil-b": {
  "rc": 1,
  "out": "$GATES_DIR/exam-root/train/runs/eval-assembled/r18-arm-a-loot-4-xdevil-b.json",
  "last_line": "eval_contract.EvalContractError: asymmetric Worker checkpoint has not completed deployable actor training, or is not at a complete PPO update boundary:{'num_timesteps': 1048576, 'last_completed_rollout': 1048576, 'ppo_optimizer_steps': 27023, 'warmup_start': None, 'warmup_until': None, 'warmup_expected_rollouts': 0, 'warmup_completed_rollouts': 0, 'warmup_optimizer_steps': 0, 'actor_optimizer_steps': 27023}"
 },
 "xm29-a": {
  "rc": 1,
  "out": "$GATES_DIR/exam-root/train/runs/eval-assembled/r18-arm-a-loot-4-xm29-a.json",
  "last_line": "eval_contract.EvalContractError: asymmetric Worker checkpoint has not completed deployable actor training, or is not at a complete PPO update boundary:{'num_timesteps': 1048576, 'last_completed_rollout': 1048576, 'ppo_optimizer_steps': 27023, 'warmup_start': None, 'warmup_until': None, 'warmup_expected_rollouts': 0, 'warmup_completed_rollouts': 0, 'warmup_optimizer_steps': 0, 'actor_optimizer_steps': 27023}"
 },
 "xm29-b": {
  "rc": 1,
  "out": "$GATES_DIR/exam-root/train/runs/eval-assembled/r18-arm-a-loot-4-xm29-b.json",
  "last_line": "eval_contract.EvalContractError: asymmetric Worker checkpoint has not completed deployable actor training, or is not at a complete PPO update boundary:{'num_timesteps': 1048576, 'last_completed_rollout': 1048576, 'ppo_optimizer_steps': 27023, 'warmup_start': None, 'warmup_until': None, 'warmup_expected_rollouts': 0, 'warmup_completed_rollouts': 0, 'warmup_optimizer_steps': 0, 'actor_optimizer_steps': 27023}"
 },
 "xm29full-a": {
  "rc": 1,
  "out": "$GATES_DIR/exam-root/train/runs/eval-assembled/r18-arm-a-loot-4-xm29full-a.json",
  "last_line": "eval_contract.EvalContractError: asymmetric Worker checkpoint has not completed deployable actor training, or is not at a complete PPO update boundary:{'num_timesteps': 1048576, 'last_completed_rollout': 1048576, 'ppo_optimizer_steps': 27023, 'warmup_start': None, 'warmup_until': None, 'warmup_expected_rollouts': 0, 'warmup_completed_rollouts': 0, 'warmup_optimizer_steps': 0, 'actor_optimizer_steps': 27023}"
 },
 "xm29full-b": {
  "rc": 1,
  "out": "$GATES_DIR/exam-root/train/runs/eval-assembled/r18-arm-a-loot-4-xm29full-b.json",
  "last_line": "eval_contract.EvalContractError: asymmetric Worker checkpoint has not completed deployable actor training, or is not at a complete PPO update boundary:{'num_timesteps': 1048576, 'last_completed_rollout': 1048576, 'ppo_optimizer_steps': 27023, 'warmup_start': None, 'warmup_until': None, 'warmup_expected_rollouts': 0, 'warmup_completed_rollouts': 0, 'warmup_optimizer_steps': 0, 'actor_optimizer_steps': 27023}"
 }
}
```

```
null
```

## 4. Gate E paired deployment re-test (48 seeds of 2_133, v1-world coverage, probe r17-deployment-v3-r18m2)

Control row: `ctl-v1-world-193557.json` (not published) rows_sha_v3 0e5a1acd2fb2c07c… (matches the ledger); this arm rows_sha_v3 34c0a4c349235d00…; RuntimeError 0.

| Metric | Training arm | Control (7e31dc54) |
|---|---|---|
| alive | 17 | 20 |
| l2 | 10 | 34 |
| l3 | 5 | 4 |
| l3_alive | 2 | 2 |
| l2_deaths | 3 | 19 |
| l2_beats | 21310 | 85578 |
| hazard_per_1k | 0.1408 | 0.222 |
| l2_kills | 119 | 894 |
| kills_mean | 69.5 | 118.8 |
| xp_mean | 4566.6 | 7969.2 |
| clvl4_plus | 5 | 25 |
| clvl_hist | {1: 10, 2: 17, 3: 16, 4: 5} | {1: 4, 2: 5, 3: 14, 4: 23, 5: 2} |
| death_dlvl_hist | {'1': 25, '2': 3, '3': 3} | {'1': 7, '2': 19, '3': 2} |
| gold_final_median | 50.0 | 78.0 |
| gold_final_total | 4171 | 6855 |
| chests_opened | 113 | 253 |
| barrels_smashed | 105 | 231 |
| sweep_windows | 65 | 104 |
| identified | 3 | 16 |
| identify_gold_spent | 300 | 1600 |
| sale_income_identified | 990 | 3769 |
| weapon_upgrades | 12 | 29 |
| weapon_gold_spent | 3030 | 6620 |
| retreats | 18 | 71 |
| deaths_with_retreat_active | 4 | 15 |
| dive_windows | 611 | 710 |
| dive_windows_descended | 27 | 90 |
| dive_window_end_reason_hist | {'cap': 75, 'death': 11, 'descend': 17, 'end': 14, 'fuse': 2, 'retreat_trigger': 11, 'scene': 10, 'stall': 436, 'sweep_trigger': 35} | {'cap': 159, 'death': 2, 'descend': 38, 'end': 13, 'fuse': 249, 'retreat_trigger': 57, 'scene': 52, 'stall': 99, 'sweep_trigger': 41} |
| descents_total | 65 | 191 |
| descent_trigger_hist | {'fallback': 38, 'coach_ready': 18, 'const_dive': 9} | {'fallback': 101, 'coach_ready': 86, 'const_dive': 4} |
| first_descent_beat_median | 5921.5 | 5535 |
| episodes_with_l2_descent | 10 | 34 |
| windows_total | 1143 | 1739 |
| windows_dive | 611 | 710 |
| windows_forced_dive | 533 | 593 |

Criteria:
```
{
 "main1_hazard_le_0.7x": {
  "arm": 0.1408,
  "control": 0.222,
  "threshold": 0.1554,
  "pass": true
 },
 "main2_paired_alive": {
  "saved": 7,
  "lost": 10,
  "net": -3,
  "discordant": 17,
  "ucb95_one_sided": 0.20302,
  "pass": false
 },
 "main3_growth": {
  "l2_kills": [
   119,
   894
  ],
  "l2_kills_needed": 1341,
  "clvl4_plus": [
   5,
   25
  ],
  "clvl4_needed": 38,
  "pass": false
 },
 "depth_reading": {
  "l3": [
   5,
   4
  ],
  "l3_alive": [
   2,
   2
  ],
  "reading": "anti-hypothesis-1 (reach up, survival not)"
 },
 "prefix_identity": {
  "seeds_identical_through_first_l2": 6,
  "seeds_different": 42,
  "different_seeds": [
   2133000,
   2133001,
   2133002,
   2133003,
   2133004,
   2133005,
   2133007,
   2133008,
   2133009,
   2133010,
   2133011,
   2133012,
   2133013,
   2133015,
   2133016,
   2133017,
   2133018,
   2133019,
   2133020,
   2133023,
   2133024,
   2133025,
   2133026,
   2133027,
   2133028,
   2133029,
   2133030,
   2133031,
   2133032,
   2133033,
   2133034,
   2133035,
   2133036,
   2133037,
   2133040,
   2133041,
   2133042,
   2133043,
   2133044,
   2133045,
   2133046,
   2133047
  ],
  "reading": "NOT identical -> report only (the trained weights also drive L1 in deployment form)"
 }
}
```

## 5. Mechanism sub-metrics (F)

- Deployment form (table above): total DIVE windows/descending windows, descent-trigger histogram, retreat count and deaths during retreats, sweep/identify/weapon-purchase counts, gold;
- Training form (§1): dive_live_descends / a11 requests and executions / escrow vested and forfeited amounts in the last `r13_dive_audit` row, and the gate passes and escrow total of `r17_descend_gate`;
- Not done: the a11 press rate in DIVE windows and the a11 rate within 5 tiles of the stairs need the replay driver (`run_r18x_l2geom.py`, not published, currently hard-codes the worker as the parent); left for the next step.

## 6. Identities

```
{
 "worker": "train/runs/r18-arm-a-loot-4/model_candidate.zip",
 "worker_sha256": "20adb0dfcdaed4cf5e10cf71f8fe5a387daf1073a43e1559b765ca759926ae47",
 "worker_kind": "model_candidate.zip (published)",
 "prereg": "r18-B-PREREG-FROZEN-20260907.md + amendments 1-2",
 "control_row": "0e5a1acd2fb2c07cd27eb72f9ecc574322e72f362ef0376b692376d9a99be886",
 "probe": "r17-deployment-v3-r18m2",
 "gates_dir": "$GATES_DIR (a local work directory, not published)",
 "eval_assembled_main_sha256": "73c2bae7a8edcaf428c0400b805a556ec80156353a95c9ce7e90cb474b23b8ee"
}
```

## 7. Provisional results of gates A–C (mirror root, corrected inherited-lineage branch; not a verdict)

The first provisional projection had the branch call `model._assert_critic_migration_contract()`; the exam tool loads a plain `MaskablePPO` on the `require_published=False` path, so the AttributeError was swallowed and all six volumes were still refused; after switching to the inlined criterion (all warm-up fields empty + `validate_inherited_runtime` + actor steps > 0 + complete PPO boundary) they passed. The full patch is in §3 of [`r18-B-PREREG-AMENDMENT-3-DRAFT-20260908.md`](r18-B-PREREG-AMENDMENT-3-DRAFT-20260908.md).

| Volume | Training arm | r17-anchor (r9 certified worker) | Gate |
|---|---|---|---|
| xdevil-a | deaths 53/128, return 108.55, L3 0 | deaths 41/128, return 140.47, L3 1 | A: saved 23/lost 35, UCB95 24.53 → fail; B: depth Δ 0.148, LCB 0.08 → pass |
| xm29-a | return 147.3, kills 49.7, deaths 96 | return 155.54, kills 63.3, deaths 47 | C: return ratio 0.947, kill ratio 0.7844 → fail |
| xm29full-a (info) | return 115.8, kills 51.5, deaths 49, L3 0 | return 147.24, kills 65.6, deaths 31, L3 0 | — |
| xdevil-b | deaths 44/128, return 120.4, L3 1 | deaths 31/128, return 150.15, L3 0 | A: saved 19/lost 32, UCB95 24.75 → fail; B: depth Δ 0.164, LCB 0.104 → pass |
| xm29-b | return 144.28, kills 49.0, deaths 99 | return 167.33, kills 65.3, deaths 45 | C: return ratio 0.8622, kill ratio 0.7509 → fail |
| xm29full-b (info) | return 110.07, kills 49.6, deaths 54, L3 0 | return 157.57, kills 69.6, deaths 26, L3 0 | — |

Row sha16 of the six volumes: {'xdevil-a': '353f5633d62f5e7e', 'xm29-a': 'eb29f94e19a7fde7', 'xm29full-a': '6362cfbf3e88abbe', 'xdevil-b': '92f08599130435ec', 'xm29-b': 'b2493bd80d8ec388', 'xm29full-b': 'a7b6d2fef68f848c'}; results file `ABC-PROVISIONAL.json` (local, not published); ledger `R18_B_GATES_ABC_PROVISIONAL`.

## 8. Composite form v0: the parent plays the level-1 prefix, the learner takes over from the first arrival on main level 2 (diagnostic; `gates/probe_composite.py`, not published; probe version string with `-composite-v0` appended)

Hand-over rule v0 = first standing on main level 2 (in training the hand-over happens when a "seven-condition qualifying DIVE window" opens on level 1, slightly earlier; the two should be aligned when this is formalized). The prefix is bit-identical to the control in 48/48; 33 games handed over, median hand-over tick 7430; RuntimeError 0; rows_sha_v3 ebc12b350247e980….

| Metric | Composite v0 | Control (7e31dc54) |
|---|---|---|
| alive | 19 | 20 |
| l2 | 34 | 34 |
| l3 | 15 | 4 |
| l3_alive | 4 | 2 |
| l4 | 1 | 0 |
| l2_deaths | 11 | 19 |
| l2_beats | 81003 | 85578 |
| hazard_per_1k | 0.1358 | 0.222 |
| l2_kills | 418 | 894 |
| l3_kills | 56 | 26 |
| kills_mean | 110.0 | 118.8 |
| xp_mean | 7166.2 | 7969.2 |
| clvl4_plus | 17 | 25 |
| clvl_hist | {'1': 4, '2': 5, '3': 22, '4': 16, '5': 1} | {'1': 4, '2': 5, '3': 14, '4': 23, '5': 2} |
| death_dlvl_hist | {'1': 7, '2': 11, '3': 10, '4': 1} | {'1': 7, '2': 19, '3': 2} |
| gold_final_median | 79.5 | 78.0 |
| identified | 15 | 16 |
| sale_income_identified | 2708 | 3769 |
| weapon_upgrades | 27 | 29 |
| retreats | 76 | 71 |
| deaths_with_retreat_active | 16 | 15 |
| dive_windows | 847 | 710 |
| dive_windows_descended | 103 | 90 |
| dive_window_end_reason_hist | {'cap': 133, 'death': 2, 'descend': 50, 'end': 14, 'fuse': 246, 'retreat_trigger': 53, 'scene': 53, 'stall': 255, 'sweep_trigger': 41} | {'cap': 159, 'death': 2, 'descend': 38, 'end': 13, 'fuse': 249, 'retreat_trigger': 57, 'scene': 52, 'stall': 99, 'sweep_trigger': 41} |
| descents_total | 199 | 191 |
| descents_2_to_3 | 23 | 5 |
| descent_trigger_hist | {'fallback': 96, 'coach_ready': 89, 'const_dive': 14} | {'fallback': 101, 'coach_ready': 86, 'const_dive': 4} |

Post-hand-over subset (the same 33 seeds, control on the same seeds): survival 11 vs 12; level-3 arrivals 15 vs 4; level-2 deaths 11 vs 19; death-level histogram {'3': 10, '2': 11, '4': 1} vs {'2': 19, '3': 2}; median ticks after the hand-over 5048.

Criteria:
```
{
 "main1_hazard_le_0.7x": {
  "arm": 0.1358,
  "control": 0.222,
  "threshold": 0.1554,
  "pass": true
 },
 "main2_paired_alive": {
  "saved": 3,
  "lost": 4,
  "net": -1,
  "discordant": 7,
  "ucb95_one_sided": 0.11137,
  "pass": false
 },
 "main3_growth": {
  "l2_kills": [
   418,
   894
  ],
  "clvl4_plus": [
   17,
   25
  ],
  "pass": false
 },
 "depth_reading": {
  "l3": [
   15,
   4
  ],
  "l3_alive": [
   4,
   2
  ],
  "l4": [
   1,
   0
  ],
  "reading": "both up"
 },
 "prefix_identity": {
  "seeds_identical_through_first_l2": 48,
  "different_seeds": []
 }
}
```

## 9. F replay: a11 mechanism metrics (`gates/run_l2geom_worker.py`, not published: a worker-parameterized copy of the replay driver; diagnostic)

Identity check of the replay against the probe's recorded rows: training arm 45/48 identical (different: [2133018, 2133025, 2133031]), control 45/48 identical (different: [2133005, 2133013, 2133047]); for R18-X the world without the new rules gave 48/48; in the world with the three new rules the replay driver diverges on about 6% of seeds, so the numbers are indicative.

| Metric (main level 2) | Training arm (deployed alone) | Control |
|---|---|---|
| l2_visits | 17 | 85 |
| l2_visit_exit_hist | {'descend': 7, 'death': 3, 'ascend': 5, 'end': 2} | {'ascend': 52, 'death': 19, 'end': 9, 'descend': 5} |
| l2_beats | 26463 | 147370 |
| l2_kills | 119 | 895 |
| dive_decisions_l2 | 2532 | 11012 |
| a11_in_dive | 352 | 261 |
| a11_rate_in_dive | 0.139 | 0.0237 |
| dive_beats_stairs_le5 | 622 | 595 |
| a11_when_le5 | 150 | 17 |
| a11_rate_when_le5 | 0.2412 | 0.0286 |
| episodes_descending_from_l2 | 5 | 4 |
| dive_action_hist | {'0': 22, '1': 59, '2': 74, '3': 131, '4': 287, '5': 234, '6': 23, '7': 570, '8': 141, '9': 179, '10': 447, '11': 352, '12': 2, '13': 10, '14': 1} | {'0': 177, '1': 684, '2': 1307, '3': 1631, '4': 1423, '5': 253, '6': 408, '7': 398, '8': 291, '9': 2160, '10': 1897, '11': 261, '12': 17, '13': 77, '14': 28} |

## 10. Reading and items to be decided

1. **Verdict on hypothesis H-B**: under the clean pairing (composite v0) Primary-1 passes, Primary-2 fails and Primary-3 fails; the depth metrics show "more level-3 arrivals, a lower survival rate among them, flat total survival": not "only learned to hang on" and not "learned nothing", but **learned to descend without learning to survive on level 3**.
2. **Level-1 regression** is an engineering finding independent of H-B: earned-dive-suffix gives the learner zero level-1 data, and once the shared weights drift, level 1 collapses (deployed alone: 25/48 deaths on level 1). Two ways out: make the composite deployment form formal (hand-over rule aligned with training; needs a new pre-registration and probe version), or add "level-1 retention" in training (mixing in level-1 windows / a KL anchor to the parent).
3. **Exam tool** admission of the weights-only lineage (draft amendment 3): whether to freeze, merge and re-certify; whether to accept the provisional A–C numbers.
4. **The direction of the next arm is not more steps**: a readiness gate at the level-3 entrance (a level-3 readiness table, a level-3 retreat rule, an escrow ruler for "stay alive after descending") and a catalogue of level-3 causes of death (damage attribution from replays).
5. The virgin pools 2_116–119 and 2_126–128 were untouched in this round; 2_133 is the paired pool (one row each for this arm and composite v0, registered).

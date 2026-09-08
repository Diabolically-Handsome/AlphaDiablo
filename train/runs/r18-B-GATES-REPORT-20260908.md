# R18-B 训练臂关门报告(2026-09-08,主刀;运行 `r18-arm-a-loot-4`)

## 〇、结论(先看这里;13:1x 第二版,含临时考卷、复合形态与 a11 回放)

- **训练**:`r18-arm-a-loot-4` 1 048 576 步、1052 局,12:41 正常收尾并发布 `model_candidate.zip`(sha 20adb0df…),保险丝未触发。训练局深度直方 {'0': 16, '1': 221, '2': 594, '3': 163, '4': 56, '5': 1, '6': 1}(三层+ 221 局);托管过门 1145/1270,兑现 20377 / 罚没 26248(R16 一程 244/244 未兑现;这一程托管**第一次大规模兑现**)。
- **闸 D**(身份)过:对 r9 认证工人 D1 0.650924 / D2 0.211894;对父代 D1 0.837778 / D2 0.511321(阈 <0.99)。
- **闸 A–C**(考卷):正式工具**结构性拒绝**本谱系(weights-only 热启动无 critic 预热回执;主树未改,见 §三 与修正案三草案)。镜像根临时结果(修正分支,§七):A 不过 / B 过 / C 不过(魔鬼卷 a/b 死亡 53/44 对锚 41/31,UCB95 24.53/24.75;M29 卷回报比 0.947/0.8622,击杀比 0.7844/0.7509)——**待主席裁定是否采信**。
- **闸 E,学习者独自部署**(§四):存活 17 vs 20(救 7/丢 10,UCB95 +0.203);一层死亡 **25 vs 7**,二层到达 10 vs 34;前缀 42/48 不同(报告不裁)。读法:学习窗只在二层以上、一层由冻结父代代打,学习者从未拿到一层数据,但共享权重在变——**一层能力退化**,独自部署在一层就垮。主-1 之「过」(风险率 0.141)是曝光假象(二层拍数 21k 对 86k),主-2、主-3 不过。
- **复合形态 v0,父代打前缀、学习者自首次到达二层接手**(§八;前缀 48/48 与对照逐位相同,故配对干净):存活 19 vs 20(救 3/丢 4,UCB95 +0.111);二层风险率 **0.1358 vs 0.222**(二层死亡 11 vs 19,二层拍数 81003 vs 85578,曝光相近——主-1 过,且是真的);三层到达 **15 vs 4**,2→3 下楼 23 vs 5,四层 1 vs 0;三层死亡 10 vs 2,三层到达者存活 4/15 对 2/4;二层击杀 418 vs 894,clvl≥4 17 vs 25——主-2、主-3 不过。
- **F 机制指标**(§九,回放 45/48 与记录行逐位相同):二层 DIVE 窗内 a11 按下率 **13.9% vs 2.4%**,离楼梯 ≤5 格时 **24.1% vs 2.9%**;二层访问以下楼结束 7/17 对 5/85。
- **一句话**:R18-B 把「下楼」教会了(托管兑现、a11 率 ×6、三层到达 ×4),没教会「下楼后活着」——死亡从二层搬到三层,总存活持平;这正是冻结件 §一 预告的可辨结局之一(反假设一的方向:深度 +、三层存活率 27% 对 50%)。另一个独立发现:earned-dive-suffix 工人不能独自部署(一层退化),部署形态必须是复合的,或训练里加一层保持。

## 一、训练侧(运行目录审计)

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

## 二、闸 D 身份探针(r13-gateD-v1,零种子消耗)

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

## 三、闸 A–C 考卷(六卷 r17-anchor 对照,考卷协议 3000 拍旧形态、协议关)

正式工具尝试(`train/eval_assembled.py`,已认证字节):六卷全部被拒。
```
{
 "xdevil-a": "eval_contract.EvalContractError: asymmetric Worker checkpoint 尚未完成可部署 actor 训练，或不在完整 PPO 更新边界:{'num_timesteps': 1048576, 'last_completed_rollout': 1048576, 'ppo_optimizer_steps': 27023, 'warmup_start': None, 'warmup_until': None, 'warmup_expected_rollouts': 0, 'warmup_completed_rollouts': 0, 'warmup_optimizer_steps': 0, 'actor_optimizer_steps': 27023}",
 "xdevil-b": "eval_contract.EvalContractError: asymmetric Worker checkpoint 尚未完成可部署 actor 训练，或不在完整 PPO 更新边界:{'num_timesteps': 1048576, 'last_completed_rollout': 1048576, 'ppo_optimizer_steps': 27023, 'warmup_start': None, 'warmup_until': None, 'warmup_expected_rollouts': 0, 'warmup_completed_rollouts': 0, 'warmup_optimizer_steps': 0, 'actor_optimizer_steps': 27023}",
 "xm29-a": "eval_contract.EvalContractError: asymmetric Worker checkpoint 尚未完成可部署 actor 训练，或不在完整 PPO 更新边界:{'num_timesteps': 1048576, 'last_completed_rollout': 1048576, 'ppo_optimizer_steps': 27023, 'warmup_start': None, 'warmup_until': None, 'warmup_expected_rollouts': 0, 'warmup_completed_rollouts': 0, 'warmup_optimizer_steps': 0, 'actor_optimizer_steps': 27023}",
 "xm29-b": "eval_contract.EvalContractError: asymmetric Worker checkpoint 尚未完成可部署 actor 训练，或不在完整 PPO 更新边界:{'num_timesteps': 1048576, 'last_completed_rollout': 1048576, 'ppo_optimizer_steps': 27023, 'warmup_start': None, 'warmup_until': None, 'warmup_expected_rollouts': 0, 'warmup_completed_rollouts': 0, 'warmup_optimizer_steps': 0, 'actor_optimizer_steps': 27023}",
 "xm29full-a": "eval_contract.EvalContractError: asymmetric Worker checkpoint 尚未完成可部署 actor 训练，或不在完整 PPO 更新边界:{'num_timesteps': 1048576, 'last_completed_rollout': 1048576, 'ppo_optimizer_steps': 27023, 'warmup_start': None, 'warmup_until': None, 'warmup_expected_rollouts': 0, 'warmup_completed_rollouts': 0, 'warmup_optimizer_steps': 0, 'actor_optimizer_steps': 27023}",
 "xm29full-b": "eval_contract.EvalContractError: asymmetric Worker checkpoint 尚未完成可部署 actor 训练，或不在完整 PPO 更新边界:{'num_timesteps': 1048576, 'last_completed_rollout': 1048576, 'ppo_optimizer_steps': 27023, 'warmup_start': None, 'warmup_until': None, 'warmup_expected_rollouts': 0, 'warmup_completed_rollouts': 0, 'warmup_optimizer_steps': 0, 'actor_optimizer_steps': 27023}"
}
```

**根因**:`_validate_asymmetric_worker_runtime_state` 要求 critic 预热回执(warmup_start/until、预期 rollout 数、`_critic_warmup_completed`);
本谱系是 schema/2 weights-only 热启动(继承父代 actor+critic,无预热期),这些字段恒为 None/0——B6 冒烟的正式发布件同样被拒(11:3x 复核)。
训练侧的发布谓词(`train_ppo` `publication_eligible`)对该谱系另有分支(继承回执有效 + actor 步数 > 0),考卷没有对应分支。
**未改动**主树任何文件。临时做法:镜像根 `~/r17_work/r18/gates/exam-root`(python/ + train/*.py 副本,build → 现役桥),
只在副本的该函数加继承谱系分支(与发布谓词同判据),六卷在副本上投射;行数据生成路径与正式工具逐位相同(校验只决定加载与否)。
**采信与否、以及是否以预注册修正案三把该分支正式并入考卷工具(需重认证),由主席裁定。**

```
{
 "xdevil-a": {
  "rc": 1,
  "out": "/home/laure/r17_work/r18/gates/exam-root/train/runs/eval-assembled/r18-arm-a-loot-4-xdevil-a.json",
  "last_line": "eval_contract.EvalContractError: asymmetric Worker checkpoint 尚未完成可部署 actor 训练，或不在完整 PPO 更新边界:{'num_timesteps': 1048576, 'last_completed_rollout': 1048576, 'ppo_optimizer_steps': 27023, 'warmup_start': None, 'warmup_until': None, 'warmup_expected_rollouts': 0, 'warmup_completed_rollouts': 0, 'warmup_optimizer_steps': 0, 'actor_optimizer_steps': 27023}"
 },
 "xdevil-b": {
  "rc": 1,
  "out": "/home/laure/r17_work/r18/gates/exam-root/train/runs/eval-assembled/r18-arm-a-loot-4-xdevil-b.json",
  "last_line": "eval_contract.EvalContractError: asymmetric Worker checkpoint 尚未完成可部署 actor 训练，或不在完整 PPO 更新边界:{'num_timesteps': 1048576, 'last_completed_rollout': 1048576, 'ppo_optimizer_steps': 27023, 'warmup_start': None, 'warmup_until': None, 'warmup_expected_rollouts': 0, 'warmup_completed_rollouts': 0, 'warmup_optimizer_steps': 0, 'actor_optimizer_steps': 27023}"
 },
 "xm29-a": {
  "rc": 1,
  "out": "/home/laure/r17_work/r18/gates/exam-root/train/runs/eval-assembled/r18-arm-a-loot-4-xm29-a.json",
  "last_line": "eval_contract.EvalContractError: asymmetric Worker checkpoint 尚未完成可部署 actor 训练，或不在完整 PPO 更新边界:{'num_timesteps': 1048576, 'last_completed_rollout': 1048576, 'ppo_optimizer_steps': 27023, 'warmup_start': None, 'warmup_until': None, 'warmup_expected_rollouts': 0, 'warmup_completed_rollouts': 0, 'warmup_optimizer_steps': 0, 'actor_optimizer_steps': 27023}"
 },
 "xm29-b": {
  "rc": 1,
  "out": "/home/laure/r17_work/r18/gates/exam-root/train/runs/eval-assembled/r18-arm-a-loot-4-xm29-b.json",
  "last_line": "eval_contract.EvalContractError: asymmetric Worker checkpoint 尚未完成可部署 actor 训练，或不在完整 PPO 更新边界:{'num_timesteps': 1048576, 'last_completed_rollout': 1048576, 'ppo_optimizer_steps': 27023, 'warmup_start': None, 'warmup_until': None, 'warmup_expected_rollouts': 0, 'warmup_completed_rollouts': 0, 'warmup_optimizer_steps': 0, 'actor_optimizer_steps': 27023}"
 },
 "xm29full-a": {
  "rc": 1,
  "out": "/home/laure/r17_work/r18/gates/exam-root/train/runs/eval-assembled/r18-arm-a-loot-4-xm29full-a.json",
  "last_line": "eval_contract.EvalContractError: asymmetric Worker checkpoint 尚未完成可部署 actor 训练，或不在完整 PPO 更新边界:{'num_timesteps': 1048576, 'last_completed_rollout': 1048576, 'ppo_optimizer_steps': 27023, 'warmup_start': None, 'warmup_until': None, 'warmup_expected_rollouts': 0, 'warmup_completed_rollouts': 0, 'warmup_optimizer_steps': 0, 'actor_optimizer_steps': 27023}"
 },
 "xm29full-b": {
  "rc": 1,
  "out": "/home/laure/r17_work/r18/gates/exam-root/train/runs/eval-assembled/r18-arm-a-loot-4-xm29full-b.json",
  "last_line": "eval_contract.EvalContractError: asymmetric Worker checkpoint 尚未完成可部署 actor 训练，或不在完整 PPO 更新边界:{'num_timesteps': 1048576, 'last_completed_rollout': 1048576, 'ppo_optimizer_steps': 27023, 'warmup_start': None, 'warmup_until': None, 'warmup_expected_rollouts': 0, 'warmup_completed_rollouts': 0, 'warmup_optimizer_steps': 0, 'actor_optimizer_steps': 27023}"
 }
}
```

```
null
```

## 四、闸 E 配对部署复测(48 种子 2_133,v1 世界覆盖,探针 r17-deployment-v3-r18m2)

对照行:`ctl-v1-world-193557.json` rows_sha_v3 0e5a1acd2fb2c07c…(与台账一致);本臂 rows_sha_v3 34c0a4c349235d00…;RuntimeError 0。

| 指标 | 训练臂 | 对照(7e31dc54) |
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

判据:
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

## 五、机制子指标(F)

- 部署形态(上表):DIVE 窗总数/下楼窗数、下楼触发直方、撤退次数与撤退中死亡、清扫/鉴定/买武器计数、金币;
- 训练形态(§一):`r13_dive_audit` 末行的 dive_live_descends / a11 请求与执行 / 托管兑现与罚没金额,`r17_descend_gate` 的过门数与托管总额;
- 未做:DIVE 窗内 a11 按下率与离楼梯 ≤5 格时的 a11 率需回放驱动(`run_r18x_l2geom.py` 目前把工人写死为父代),留待下一步。

## 六、身份

```
{
 "worker": "/home/laure/AlphaDiablo/diablogym/train/runs/r18-arm-a-loot-4/model_candidate.zip",
 "worker_sha256": "20adb0dfcdaed4cf5e10cf71f8fe5a387daf1073a43e1559b765ca759926ae47",
 "worker_kind": "model_candidate.zip (published)",
 "prereg": "r18-B-PREREG-FROZEN-20260907.md + amendments 1-2",
 "control_row": "0e5a1acd2fb2c07cd27eb72f9ecc574322e72f362ef0376b692376d9a99be886",
 "probe": "r17-deployment-v3-r18m2",
 "gates_dir": "/home/laure/r17_work/r18/gates",
 "eval_assembled_main_sha256": "73c2bae7a8edcaf428c0400b805a556ec80156353a95c9ce7e90cb474b23b8ee"
}
```

## 七、闸 A–C 临时结果(镜像根,修正后的继承谱系分支;不是裁定)

第一次临时投射(12:43)分支调用 `model._assert_critic_migration_contract()`,考卷在 `require_published=False` 路径加载的是纯 `MaskablePPO`,AttributeError 被吞、六卷仍拒;12:5x 改为内联判据(预热字段全空 + `validate_inherited_runtime` + actor 步数 > 0 + 完整 PPO 边界)后通过。补丁全文见 `r18-B-PREREG-AMENDMENT-3-DRAFT-20260908.md` §三。

| 卷 | 训练臂 | r17-anchor(r9 认证工人) | 闸 |
|---|---|---|---|
| xdevil-a | 死亡 53/128,回报 108.55,L3 0 | 死亡 41/128,回报 140.47,L3 1 | A:救 23/丢 35,UCB95 24.53 → 不过;B:深度Δ 0.148,LCB 0.08 → 过 |
| xm29-a | 回报 147.3,击杀 49.7,死亡 96 | 回报 155.54,击杀 63.3,死亡 47 | C:回报比 0.947,击杀比 0.7844 → 不过 |
| xm29full-a(信息) | 回报 115.8,击杀 51.5,死亡 49,L3 0 | 回报 147.24,击杀 65.6,死亡 31,L3 0 | — |
| xdevil-b | 死亡 44/128,回报 120.4,L3 1 | 死亡 31/128,回报 150.15,L3 0 | A:救 19/丢 32,UCB95 24.75 → 不过;B:深度Δ 0.164,LCB 0.104 → 过 |
| xm29-b | 回报 144.28,击杀 49.0,死亡 99 | 回报 167.33,击杀 65.3,死亡 45 | C:回报比 0.8622,击杀比 0.7509 → 不过 |
| xm29full-b(信息) | 回报 110.07,击杀 49.6,死亡 54,L3 0 | 回报 157.57,击杀 69.6,死亡 26,L3 0 | — |

六卷行 sha16:{'xdevil-a': '353f5633d62f5e7e', 'xm29-a': 'eb29f94e19a7fde7', 'xm29full-a': '6362cfbf3e88abbe', 'xdevil-b': '92f08599130435ec', 'xm29-b': 'b2493bd80d8ec388', 'xm29full-b': 'a7b6d2fef68f848c'};结果文件 `~/r17_work/r18/gates/ABC-PROVISIONAL.json`;台账 `R18_B_GATES_ABC_PROVISIONAL`。

## 八、复合形态 v0:父代打一层前缀,学习者自首次到达主二层接手(诊断,`gates/probe_composite.py`,探针版本串加 `-composite-v0`)

交接规则 v0 = 首次站上主二层(训练里的交接是一层上「七条件合格 DIVE 窗」开窗时,略早于此;正式化时应对齐)。前缀与对照逐位相同 48/48;交接 33 局,交接拍中位 7430;RuntimeError 0;rows_sha_v3 ebc12b350247e980…。

| 指标 | 复合 v0 | 对照(7e31dc54) |
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

交接后子集(同 33 粒种子,对照取同种子):存活 11 vs 12;三层到达 15 vs 4;二层死亡 11 vs 19;死亡层直方 {'3': 10, '2': 11, '4': 1} vs {'2': 19, '3': 2};交接后拍数中位 5048。

判据:
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

## 九、F 回放:a11 机制指标(`gates/run_l2geom_worker.py`,回放驱动的工人参数化副本;诊断)

回放与探针记录行的身份核对:训练臂 45/48 相同(不同:[2133018, 2133025, 2133031]),对照 45/48 相同(不同:[2133005, 2133013, 2133047])——R18-X 当夜对无新法世界是 48/48;新三法世界下回放驱动有约 6% 种子分岔,数字按「指示性」读。

| 指标(主二层) | 训练臂(独自部署) | 对照 |
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

## 十、主刀读法与待裁定事项

1. **假设 H-B 的判决**:三条主判据在干净配对(复合 v0)下 主-1 过、主-2 不过、主-3 不过;深度指标「三层到达升、三层到达者存活率降、总存活持平」——不是「只学会苟活」,也不是「什么都没学」,而是**学会下楼、未学会三层生存**。
2. **一层退化**是独立于 H-B 的工程发现:earned-dive-suffix 让学习者对一层零数据,共享权重漂移后一层崩(独自部署一层死亡 25/48)。两条路:部署形态正式复合化(交接规则与训练对齐,需新预注册与探针版本),或训练侧加「一层保持」(混入一层窗 / 对父代的 KL 锚)。
3. **考卷工具**对 weights-only 谱系的准入(修正案三草案):是否冻结并入并重认证;临时 A–C 数字是否采信。
4. **下一臂的方向不是更多步数**:三层入口的战备门(三层战备表、三层撤退法、「下楼后先活着」的托管尺)与三层死因图鉴(可用回放的伤害归因)。
5. 处女池 2_116–119、2_126–128 今日零接触;2_133 为配对池(本臂 + 复合 v0 各一行,已登记)。

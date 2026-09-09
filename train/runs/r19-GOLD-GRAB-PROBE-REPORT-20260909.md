# R19 报告：gold-grab-v1（`resource_gold_grab = gold-grab-v1`）

- 日期：2026-09-08
- 实现者标签：`r19-gold-grab-implementer`
- 工作树：`/home/laure/r17_work/r19/gold-tree`（Python only；`build -> /home/laure/r17_work/r17-1/build-res`，与主树同一条活桥）
- 主树 `/home/laure/AlphaDiablo/diablogym` 未被修改（唯一例外：账本 append）；`src/*.cpp|hpp`、引擎、C++ 一律未动，今晚无新引擎补丁
- 补丁：`/home/laure/r17_work/r19/gold.patch`（8 个文件：1 个新法模块、1 个新测试文件、6 处登记）
- 探针原始产物：`/home/laure/r17_work/r19/gold-probe/`（`*.cmd` / `*.log` / `*.json`），驱动脚本 `/home/laure/r17_work/r19/gs/gold_probe_driver.py`

> **2026-09-08 夜 · 复核轮已完成，见[第七节](#七复核轮2026-09-08-夜review-round)。**
> 复核者给了 2 high + 3 medium + 1 low，**全部已修**；两臂**已按修好的树全部重跑**
> （OFF 回归 sha 仍逐位相同；ON 臂产物在 `/home/laure/r17_work/r19/gold-probe-rr/`，
> 驱动脚本 `/home/laure/r17_work/r19/rr/gold_probe_driver_rr.py`）。
> **第一至六节是第一遍的原始记录，未删改**；其中被复核证伪的三处已在原地标注
> 〔复核轮更正〕。凡与第七节冲突之处，**以第七节为准**。

---

## 一、法条

主席 2026-09-08 裁定：**随手捡看得见的金币**。今天的金币只在一个地方进钱包——
loot 经济进城行程前的 `collect` 阶段，而且只在主线 L1
（`resource_sustain_loot.py:323-325` 把该阶段钉死在 `dungeon_level == 1`）。
本法把「捡」从行程里拿出来，做成一个和 sweep 同形的小脚本窗：

1. **旗**：`resource_gold_grab ∈ ("off", "gold-grab-v1")`，默认 `off`；
   只有 `resource_protocol=l2-town-v1` **且** `resource_service_policy=sustain-loot-v1`
   时才合法（`validate_gold_grab_protocol`，fail closed）。没有任何既有法条的阈值被改动。
2. **记忆**：每层一份、每局一份，只记**本局真正被照亮过**的金堆
   （native `resource_state["gold_items"]` 本身就是 `IsTileLit` 过滤过的，
   `src/resource_protocol.hpp:951-964`）。身份是四个 item identity 字段
   （`seed_hi`/`seed_lo`/`create_info`/`base_id`），**永远不是 active slot**。
   实现是把冻结的 `ObservedGoldMemory` 原样继承下来
   （`_FloorGoldMemory`，只覆盖场景谓词 `_main_l1`），每层一个实例
   （`GrabGoldMemory`）——所以两层的同一身份永不相撞，冻结类本身一个字节没改。
   作用域：**主线 L1 与主线 L2**；城、set level、L3+ 一律不在内。
3. **窗**：在其它服务开窗的同一处（`OptionsEnv.action_masks`）触发，四个条件全中才开：
   记忆里至少有一堆在 **R=8** 切比雪夫格内；玩家与目标堆 **3 格内没有活着的可见怪**；
   `hp > 0.5*max_hp`；玩家 idle（`player_mode == 0`）。
   窗内用 **sweep 用的那条冻结安全路径规划器**（先避怪 BFS、再普通 BFS）走到最近的记忆堆，
   用 **collect 阶段本来就在用的原生命令** `("gold", *gold_key)` →
   `bridge.act_pickup_gold_at` 捡起来（该 native 动作自己就要求玩家站在被照亮的堆上，
   `src/resource_protocol.hpp:1068-1081`），然后继续走 R 内的下一堆。
4. **中止**：可见/不可见的活怪进 3 格、`hp <= 0.5*max`、路线失败、连续 8 次走路被拒、
   每窗 **300** 微步、每局 **900** 微步预算、每局 **24** 堆上限——以及
   **撤退法想要身体的那一拍**（`retreat.trigger_reason` 是纯谓词，
   `resource_retreat.py:130-160`；主线 L1 上它按自身深度条款恒为 None，
   所以这一项只在 L2 说话）。
5. **入账**：每次捡金按**钱包差额**入账，在 native 命令之后的第一次观测上结算——
   与 `SustainGoldMemoryService.receipt` 的 `gold_collected += delta` 同一口径，
   因为 `act_pickup_gold_at` 只是排一个请求、不回报金额。观测到的 `value` **从不**被当作收入。
   窗永远不会与进城行程同时开（触发法里就禁止了），所以 loot 经济自己的
   `gold_collected` 与本法的 `gold_taken` 不可能重复记同一枚硬币；
   每次行程的 `_start_gold` 快照本来就已经包含本法在它之前捡到的钱，
   `_seal_trip` 的现金对账（residual 必须为 0）因此一字未动。

### 权限链（主席的裁定：撤退与传送保持优先）

- **触发**顺序：本法在 `action_masks` 里被**最后**问——在 sweep、
  `service.maybe_start`（进城行程）、portal 到达腿、portal 触发、retreat 触发**全部之后**。
  只有当这五者都不活动时才开窗。这样「想走的行程/撤退/传送」永远不会被捡金窗拖延。
- **占有链**（`OptionsEnv.step` 的 RESUPPLY owner chain）里按主席的话放在 sweep 之后、
  town service 之前。这一格**不带行为**：触发守卫保证同一拍不可能有两条腿活着；
  它是那条不变量的 fail-closed 声明。
- R18-M 在 `options_env.py` 留的明文告诫（「把 sweep 的深度放宽、或复制这个块做一条新腿，
  那条不变量就不成立了——请在这里和下面加上 `.active` 项」）**正是本法的情形**：
  本法是第一条开到主线 L2 的脚本窗。所以 portal 与 retreat 两个触发守卫都加上了
  `not gold_grab_service.active`。代价至多是关窗的那一拍。
  〔复核轮更正 2026-09-08〕**「代价至多是关窗的那一拍」在第一遍里是假的**：中止阶梯
  当时只有 retreat 项、没有 portal 项，而传送法不是撤退法的谓词（撤退条款 **加** 卷轴，
  外加根本不看血量的 `portal_standing`），所以一条中途变合法的传送腿会被饿死至多
  `GRAB_WINDOW_MICROSTEP_CAP = 300` 微步。复核轮已补 `grab_portal_wanted` 项与
  `bind_portal`，这句话现在才是代码提供的界。见 7.1（4）。
- `_win_term` 里加了与 sweep 逐行同构的一格：`grab_window` 时窗由**脚本**收
  （`grab_complete`），非 RESUPPLY 窗遇到 `grab.active` 立刻收（`grab_trigger`）。
  没有这一格，冻结 worker 会在一个已经塌成 RESUPPLY 的掩码下继续行动——
  这是 R18-H 复核轮踩过的同一个坑。

### 一处**比裁定更严**的地方（请复核者点名看）

裁定说「3 格内没有活着的**可见**怪」。本法两条都执行：可见口径（裁定）**和**
`SweepService` 的冻结口径（一切活怪，不问可见）。理由：这个窗和 sweep 一样把经理掩码
塌成 RESUPPLY，两者对「现在能不能占着身体」不该有两种答案。这一项**只会更窄**，
不会更宽；遥测把两种否决分开记（`monster_near_pair` / `monster_near_pair_unseen`）。

### 第 4 条（L2 行程期捡金）：**没做，并说明为什么**

主席要求「只有在既有服务里既便宜又安全时才做」。它不便宜，结构上做不到：
`ResourceService._command`（`resource_protocol.py:362-364`）对
`depth not in (0, 1)` 直接 `_finish("resource_unexpected_scene")`，
也就是**进城行程根本不能从 L2 出发**；从 L2 回城的载具是 R18-A 撤退 / R18-F 传送，
而主席明令不改撤退的走法。要让 collect 阶段在 L2 跑，就得改 town service 的相位机
（outbound 的 `_stairs` 腿假定 depth 1 → 城）。故**跳过**。
本法的第 2、3 条已经覆盖了这条裁定想要的实质：L2 的金币现在在**平时**就被捡，
不必等一次不存在的 L2 行程。

---

## 二、改了什么（词汇表登记，逐条与既有法条同位）

| 位置 | 内容 |
|---|---|
| `python/diablogym/resource_gold_grab.py`（新） | `GoldGrabService` / `GrabGoldMemory` / `_FloorGoldMemory` / `validate_gold_grab_protocol` 转出；常数 R=8、窗 300、局 900、堆 24、怪 3、hp 0.5、冷却 60 |
| `python/diablogym/resource_protocol.py` | `RESOURCE_GOLD_GRAB_PROTOCOLS = ("off", "gold-grab-v1")`；`validate_gold_grab_protocol(protocol, service_policy, gold_grab)` |
| `python/diablogym/options_env.py` | 构造器 kwarg `resource_gold_grab`（**不下传 DiabloGymEnv**，与 `resource_calibration` / R19-A1 `resource_retreat_scope` 同理：native 侧没有这个词）；每局一个 `gold_grab_service` 并 `bind_retreat`；`action_masks` 里的 observe + 触发块（最后一格）+ 掩码塌缩；portal/retreat 两个触发守卫加 `.active` 项；`resource_option_choice` 一格；RESUPPLY owner chain 一格；`_win_term` 一格 + `_win["grab_window"]`；两个遥测出口 |
| `python/diablogym/worker_env.py` | 训练侧同一个 validator 直通；默认 `off` 时 kwargs 逐字节不变；同样不下传 native |
| `train/runs/r10-staging/probe_r17_deployment.py` | `_R16_ENV_KEYS` 增 `"resource_gold_grab"`；每行 resource 遥测增 `gold_grab_flag` / `gold_grab`（完整 telemetry）/ `gold_grab_windows` / `gold_grab_piles` / `gold_grab_gold` / `gold_grab_microsteps`（关时 None/0，不是 v3 行键） |
| `train/eval_contract.py` | `_PYTHON_PROTOCOL_FILES` 增新模块；`R16_ENVIRONMENT_DEFAULTS["resource_gold_grab"] = "off"`；`validate_r16_environment` 把它并入 sweep/identify/weapon 那一支（身份必须同时写下 `l2-town-v1` 与 `sustain-loot-v1`） |
| `train/train_ppo.py` | 指纹清单增新模块（两份指纹都绑住每一个存在的法模块，R18-B6 的规矩） |

遥测（只增不改，关时严格是超集的空侧）：
`gold_grab {windows, piles_taken, gold_taken, aborts_by_reason, by_floor{1,2}, denials, ...}`；
另有每窗账本 `window_ledger`（trigger/floor/beat0/gold0/candidates0/piles_taken/gold_taken/outcome/reason/beat1/gold1/hp_min）
与 `failures`、`memory`（每层记忆的 telemetry）。旗关时 `gold_grab` 为 `None`、四个标量为 0。

---

## 三、测试

- 新增 `tests/test_r19_gold_grab.py`：**76 passed / 24 subtests passed（0 失败）**（`/home/laure/r17_work/r19/gold-tree`，
  `PYTHONPATH=python:train`）。覆盖：词汇表与 fail-closed validator（没有 loot 经济就拒）；
  四个触发条件逐条与边界（R=8 边界、切比雪夫对角、hp 51/50/49、idle、行程活动中、
  怪贴人 / 怪贴目标堆 / 死怪 / 109 无效标记 / 不可见活怪）；作用域（L0/L3/L4/L16/set level 全拒，
  L1 与 L2 都开）；记忆身份跨 active slot 变化不变、两层不相撞、光走开后仍记得、
  没有 gold 通道不等于空地板；三个上限（窗 300 / 局 900 / 24 堆）与冷却；
  中止规则（死、hp≤50%、可见怪、不可见活怪、场景切换、进城、撤退法要身体、走路连拒）；
  入账（钱包差额、观测 value 永不当收入、没动钱包的请求不是收入、被拒请求不等待）；
  默认关臂身份（两个 wrapper 的默认值、**永不下传 native**、探针行键、档案身份、两份指纹）。
- 邻域回归（本文件 + sweep / retreat / retreat-training / gold-memory / sustain-contract /
  loot-protocol-integration / probe-sustain / options_env / r18b6-training-wiring / hunt_scope）：
  **354 passed / 4 skipped / 527 subtests passed / 11 failed**。
  这 11 条失败**与本改动无关**：在同一台机、同样只 rsync 了 python/tests/train 的**纯净副本**
  `/home/laure/r17_work/r19/base-tree` 上跑同样的文件（去掉本文件），失败集合**逐条相同**
  （`/home/laure/r17_work/r19/gold-tests-base-full.log`：11 failed, 278 passed；本树 11 failed, 354 passed）。
  它们全部来自 `test_r18b6_training_wiring.py`，读的是本副本没有拷贝的
  `train/runs/r16-arm-a-constitution/model_candidate.zip`。
- 完整 10 分钟全量套件**未跑**（not verified）。
- 〔复核轮更正 2026-09-08〕上面两行是第一遍的计数。复核轮补测之后：新测试文件
  **112 passed / 37 subtests passed / 0 failed**，邻域
  **554 passed / 4 skipped / 711 subtests passed / 11 failed**（11 条仍与纯净副本逐条相同）。
  完整全量套件**仍未跑**。见 7.2 / 7.3。

---

## 四、探针（pool 2_133，48 种子 2133000-2133047，3 分片并行，worker 7e31dc54）

探针版本 `r17-deployment-v3-r18m2`，`max_steps 6000`，decoding `sample`，
覆盖 = `/home/laure/r17_work/r18/gates/v1world-overrides.json` 的 16 键。

### 4.1 OFF / 默认臂（回归）——**通过**

- `rows_sha_v3` = `0e5a1acd2fb2c07cd27eb72f9ecc574322e72f362ef0376b692376d9a99be886`
- 认证控制行 = `0e5a1acd2fb2c07cd27eb72f9ecc574322e72f362ef0376b692376d9a99be886`（`/home/laure/r17_work/r18/merge-probe/ctl-v1-world-193557.json`，48 行）
- **相等：是**；48 行；RuntimeError 计数 0；耗时 594.9s
- 合并行：`/home/laure/r17_work/r19/gold-probe/gold-off-merged-rows.json`

### 4.2 ON 臂 = 16 键 + `{"resource_gold_grab": "gold-grab-v1"}`

- 原始行：`/home/laure/r17_work/r19/gold-probe/gold-on-235012.json`；RuntimeError 计数 0；耗时 580.6s
- 配对 McNemar（公式取自 `/home/laure/r17_work/r18/m2_probe_driver.py` 的 `paired()`）：
  **saved 11、lost 9、net 2、discordant 20、
  ucb95_one_sided 0.11128**

#### 主表（control = 认证控制行；两臂同一 `arm_stats`）

| 指标 | control (OFF) | ON (gold-grab-v1) |
|---|---|---|
| 存活 / 48 | 20 | 22 |
| 到过 L2 | 34 | 29 |
| 到过 L3 | 4 | 4 |
| L1 死 | 7 | 8 |
| L2 死 | 19 | 17 |
| L2 拍数 | 85578 | 64618 |
| L2 危险度 /1k | 0.222 | 0.2631 |
| 首降拍 中位 | 5535 | 5851.5 |
| 有首降的局 | 41 | 38 |
| 首降时带药 中位 | 4 | 4.0 |
| 临死带药 中位 | 2.5 | 4.0 |
| gold_final 中位 | 78.0 | 119.5 |
| gold_final 合计 | 6855 | 7439 |
| 行程 collect 捡金 合计 | 12178 | 4678 |
| 卖货收入 合计 | 7591 | 6250 |
| 花掉 合计 | 15442 | 14724 |
| 采购件数 合计 | 161 | 139 |
| 进城行程 合计 | 105 | 88 |
| 采购/行程 | 1.533 | 1.58 |
| 花费/行程 | 147.07 | 167.32 |

#### 捡金本身（只有 ON 臂有）

- 窗 **858**、拿 **993** 堆、进钱包 **9240** 金、
  花 **18329** 微步（全臂 48 局），被拒请求 0、
  未结算 0、有窗的局 48/48
- 按层：`{'1': {'windows': 857, 'piles_taken': 992, 'gold_taken': 9215}, '2': {'windows': 1, 'piles_taken': 1, 'gold_taken': 25}}`
- 中止原因：`{'grab_monster_near': 403, 'grab_no_target_in_radius': 417, 'grab_pile_cap': 38}`
- 否决原因（只在「R 内确实有记忆金堆」的拍上计数）：`{'monster_near_pair': 2961, 'cooldown': 5002, 'monster_near_pair_unseen': 230, 'monster_near_pile': 89, 'pile_cap': 22671, 'low_hp': 431}`

#### 点名的两个种子

**2133002**（主席点名：对照里在第一次进城前就死；认证控制行给出的是 died=True、0 次行程）

- **control**:died=True depth=1 micro_steps=559 gold_final=50 clvl=1 kills=21 first_descent_beat=None trips=0 gold_collected=0 gold_spent=0 purchases=0 死于 L1 @beat 559(临死带药 0)
- **ON**:died=False depth=1 micro_steps=12000 gold_final=77 clvl=4 kills=136 first_descent_beat=6000 trips=2 gold_collected=93 gold_spent=412 purchases=6
  - grab:窗 20、拿 24 堆、231 金、耗 457 微步、按层 {'1': {'windows': 20, 'piles_taken': 24, 'gold_taken': 231, 'piles_seen_lit': 22}, '2': {'windows': 0, 'piles_taken': 0, 'gold_taken': 0, 'piles_seen_lit': 0}}、中止 {'grab_no_target_in_radius': 13, 'grab_monster_near': 6, 'grab_pile_cap': 1}

**2133010**（对照里 L1 上 46 堆、行程拿 33 堆；L2 死时还有 14 堆没捡）

- **control**:died=True depth=2 micro_steps=12001 gold_final=22 clvl=4 kills=148 first_descent_beat=5842 trips=5 gold_collected=453 gold_spent=599 purchases=6 死于 L2 @beat 12001(临死带药 0)
- **ON**:died=False depth=1 micro_steps=12000 gold_final=106 clvl=3 kills=124 first_descent_beat=5610 trips=2 gold_collected=125 gold_spent=401 purchases=2
  - grab:窗 20、拿 24 堆、252 金、耗 387 微步、按层 {'1': {'windows': 20, 'piles_taken': 24, 'gold_taken': 252, 'piles_seen_lit': 24}, '2': {'windows': 0, 'piles_taken': 0, 'gold_taken': 0, 'piles_seen_lit': 0}}、中止 {'grab_no_target_in_radius': 11, 'grab_monster_near': 8, 'grab_pile_cap': 1}

---

## 五、读数与建议

### 5.1 法条本身是好用的（机械层面全绿）

48 局里开了 **858** 个窗、捡起 **993** 堆、**9240** 金进钱包，
**0** 次原生请求被拒、**0** 次入账悬空、**0** 次 RuntimeError。
时间代价 18329 微步 / 48 局 = **每局约 382 微步**（本轮存活局的 micro_steps 观测值为 12000,即约 3.2%）。
默认关臂逐位复现认证控制行，`rows_sha_v3` 一字不差。

### 5.2 钱确实早到了，但**总收入几乎没变**

- `gold_final` 中位 **78 → 119.5**；合计 6855 → 7439。
- 但行程 collect 的捡金从 **12178 掉到 4678**：本法拿走的正是那些堆。
  三条收入加起来（grab + collect + 卖货）：control **19769** vs ON **20168**，只多 **2%**。
- 也就是说：**这不是「多赚钱」，这是「早赚钱、且不必活到进城才赚」**。
  这恰恰是主席点名的那条漏：2133002 在对照里第 559 拍就死在 L1，
  0 次行程、钱包 50、临死带药 0；ON 臂里它活到本轮观测的 12000 微步、clvl 4、
  跑了 2 次行程、买了 6 件东西——本法在它死之前先把 24 堆 231 金放进了它的钱包。

### 5.3 代价：**下楼变少了**

- 到过 L2：**34 → 29**；L2 拍数 **85578 → 64618（-24%）**；
  首降拍中位 5535 → 5851.5，有首降的局 41 → 38。
- 2133010 是这条代价的样板：对照里它 5 次行程、死在 L2；ON 臂里它**活着，但一整局没离开 L1**。
  这与账本第 687 行「rich start 反事实」的读数**完全同形**：
  钱多 → 进城更多 → L2 暴露更少 → 活得久但不下楼。本法把那份反事实变成了一条合法的法。

### 5.4 存活：**方向对，但不够证**

- 存活 20 → 22；配对 saved **11** / lost **9** / net **+2**、discordant 20、
  **ucb95_one_sided = 0.11128 > 0**。按 R18 的闸口口径，这**不算被证明无害**，
  更谈不上被证明有益：20 个不一致种子上的 ±2 与噪声无法区分。
- L2 死 19 → 17、L1 死 7 → 8；L2 危险度 0.222 → 0.2631——但那是在少了 24% 的 L2 拍数上算的，
  分母变了，这一格不能单独读。
- 临死带药中位 **2.5 → 4.0**：死的时候带的药更多了，说明死因已经不是「空带」，
  也说明钱变成药这条链路是通的。

### 5.5 主席第 2 条的 **L2 半边今晚基本没被测到**（重要缺口）

- 按层：L1 **857 窗 / 992 堆 / 9215 金**，L2 **1 窗 / 1 堆 / 25 金**。
- 原因写在否决表里：`pile_cap` 否决 **22671** 次——每局 24 堆的上限
  **在 L1 上就被花光了**，等走到 L2 已经没有额度。
- 所以「L2 也随手捡」这条今晚**没有被真正检验**（not verified）。

### 5.6 建议（给主席与复核者）

1. **不要以现在这个形状进闸**：ucb95 为正，收益主要是「早到的钱」，
   代价是「更少的 L2」，两者都没有被 48 个种子分辨清楚。
2. **先改额度的形状再测一轮**：把每局 24 堆改成**按层各一份**
   （例如 L1 12 堆 + L2 12 堆，或 L2 单独一份不与 L1 共享），
   这样第 2 条的 L2 半边才会真的被执行；现在的数据说不了 L2 的话。
3. **同时看「下楼」这条线**：本法与「更多钱 → 更多进城 → 更少下楼」的既有反事实同形。
   若主席要的是深度而不是寿命，本法需要和一条「进城预算」法配对，
   否则它会把命换成层。
4. **复现一轮再谈**：48 个种子、一轮 `sample` 解码，discordant 20。
   建议同一臂再跑一轮不同分片顺序，或扩到 96 个种子，再决定。

---

## 六、已知缺口（诚实清单）

- **第 4 条没做**：L2 行程期 collect。理由在第一节末，是结构性的（town service 拒绝 depth≥2），
  不是偷懒；要做必须动 town service 的相位机，属另一部法。
- **训练侧只做了直通**：`worker_env` 有 validator，但 `train_ppo` 只加了指纹绑定，
  **没有**加 CLI 旗 / `make_env` 转发（与 R19-A1 同一取舍）。今晚不训练，
  所以这只是「还不能训」，不是「训了会错」。要训必须先补
  `_validate_args` / `make_env` / `r16_environment` 三处，和 R18-B6 同款。
- **档案身份链当前一定 fail closed**：`validate_r16_environment` 里
  `resource_service_policy` 那一支还拒绝一切 `sustain-loot-v1` 档案（R18-B3：档案 schema
  还没有时钟键）。本法的身份分支和 sweep/identify/weapon 一样挂在它后面，现在铸不出档案。
  这是继承来的、刻意的 fail closed，不是本法新增的洞。
- **入账靠钱包差额**：`act_pickup_gold_at` 只排请求、不回金额，所以「捡到多少」只能用
  钱包差额。窗内没有别的东西能动钱包（触发法禁止与行程同开、不在商店），
  但如果一次捡金正好落在**局末的死/截断**上，差额可能永远观测不到——
  这类事件记在 `unresolved_pickups`，**从不**被编造成收入（本轮 ON 臂：0 次）。
- 〔复核轮更正 2026-09-08〕上面一条「入账靠钱包差额」还漏了一半：跨界的不是窗口而是
  **pending**，而且结算超时在第一遍里只在钱包没动时才生效。两处都已修，见 7.1（2）（3）。
  另外第一遍的 24 堆是**每局**上限，L1 先走到就把它花光，主席裁定的 L2 半边因此是
  **结构性不可达**（不只是没测到）；复核轮加了每层 12 的上限，L2 这一轮才真的有数据（7.5）。
- **`_win_term` 的 `grab_trigger` 会切断 FARM/DIVE 窗**：这与 sweep 的做法逐行同构，
  但本法开窗比 sweep 频繁得多（sweep 一局最多 16 个目标、只在 cleared/farm_cap 开）。
  窗数与被切断的窗数的代价体现在上表的 `first_descent_beat` 与 `l2` 两行，请复核者按那两行读。
- 全量测试套件未跑；训练未跑；只有这 48 个种子的一轮 `sample` 解码探针。


---

## 七、复核轮（2026-09-08 夜，review round）

复核者给了 6 条（2 条 high、3 条 medium、1 条 low）。**5 条全部修了**（含两条 high 与三条
medium），low 那条（缺集成测试）也补了。改动仍然只在副本树
`/home/laure/r17_work/r19/gold-tree`，仍然是 Python-only，仍然没有改任何既有法条的阈值；
主树 `/home/laure/AlphaDiablo/diablogym` 复核后仍与 `base-tree` 的 `python/`、`tests/`、
`train/*.py` 逐字节相同（只有 append-only 台账除外）。补丁重新生成：
`/home/laure/r17_work/r19/gold.patch`（同样 8 个文件）。

### 7.1 修了什么

**（1，high）托管罚没作用域 —— `worker_env.py` 新增 `_grab_close_is_death_equivalent`。**
复核者是对的：R18-B6 那段"深度构造性不交"的证明只对 sweep 成立
（sweep 只在 main L1，两条罚没条款只在 main L2+）。`GRAB_FLOORS = (1, 2)`，本法是**第一部**
把脚本窗口开到 main L2 的法，两个集合正好相交；而 `_win_term` 的 grab 档又在 portal/retreat
两档**之上**。于是一扇本该在 hp<=50% 以 `retreat_trigger` **全额罚没**收窗的 DIVE 窗，
会被一扇先开起来的抓金窗改写成 `grab_trigger` 而 vest，接着在抓金窗内部掉到半血、
以 `grab_complete` 再 vest 一次 —— 正是 R18-B/R18-B5 要关掉的"带伤潜到半血再套现"通道。
现在 `_forfeit_reason` 多一支：旗开 + `close_reason in ("grab_trigger","grab_complete")`
+ `_grab_close_is_death_equivalent()`。该谓词与传送侧 `_portal_close_is_death_equivalent`
同形：main L2+、非定场，且（抓金窗自己以 `grab_low_hp` / `grab_retreat_wanted` /
`grab_portal_wanted` / `grab_died` 交还身体）或（这一拍 `RetreatService.trigger_reason`
非 None）或（`PortalService.danger_reason` 非 None）。阈值仍只存在一处：这里只问服务自己的
纯谓词。`worker_env.py` 里 R18-B6 那段证明**保留**（对 sweep 仍成立），前面新增一段说明它
对本法为何失效。旗关时该支恒不可达。

**（2，high）排队中的捡金请求可以活过它的窗口。**复核者是对的，而且指出了要害：跨界的
不是**窗口**，是**pending**。修法两处：
- `GoldGrabService._end` 不再在 `self._pending` 未结算时返回。它改为在**受审计的、脚本自有的**
  `("wait",)` 命令上挂起（R17.1 ruling 3 在 RESUPPLY 循环里的 settle-through-wait 先例），
  直到硬币落袋或 `GRAB_SETTLE_MICROSTEPS` 超时把它退役；`_closing` 记住法条决定的收窗理由，
  下一拍 `_command` 顶部结算完再真正关窗。挂起可证有界：`step - pending["beat"]` 每拍单调增。
- `OptionsEnv.action_masks` 的 `service.maybe_start` 多了一道闩：
  `not (grab.active or grab.pending_pickup)`。这同时把模块 docstring 一直断言、
  但代码此前并没有保证的那条不变式（"抓金窗从不与进城行程同开"）真正锁上了。

**（3，medium）结算超时只在钱包没动时才生效。**原顺序是 `if delta > 0: 入账; return` 在前，
所以只要钱包动过就**完全没有**时限。现在超时先判；并且 pending 携带**发出它的那扇窗的下标**，
入账只会落到那扇窗上，绝不会盖到 `self.windows[-1]`（那时可能已经是另一层的另一扇窗）。
窗口已关的 pending 一律不得入账（记 `pickup_unresolved_stale`），永不编造金额。

**（4，medium）开着的抓金窗会饿死传送法，而且注释里那句"至多一拍"是假的。**
`_command` 的中止阶梯只有 retreat 项、没有 portal 项，而传送法**不是**撤退法的谓词
（它是撤退条款 **加** 身上有卷轴，另有 `portal_standing` 根本不看血量）。现在加了
`grab_portal_wanted` 项（`PortalService.trigger_reason` 是纯谓词，正是 `_win_term` 自己轮询的
那一个），`OptionsEnv` 构造时 `bind_portal`。`options_env.py` 里那句错误的注释已按代码事实改写。

**（5，medium）主席裁定的 L2 半边是结构性不可达，不只是没测到。**24 堆是**每局**上限，
L1 永远先被走到，预算在下楼前就花光了（前一轮：858 扇窗里 857 扇、993 堆里 992 堆在 L1，
`pile_cap` 拒绝 22671 次）。新增 `GRAB_MAX_PILES_PER_FLOOR = 12` 的**每层**上限；
主席的**每局 24 不变**，只是把其中一半留给这部法被扩宽到的那层。触发法多一条
`floor_pile_cap` 拒绝，中止阶梯多一个 `grab_floor_pile_cap` 理由。

**（6，low）补集成测试。**新增 `WinTermRungTests`（直接驱动出厂的
`OptionsEnv._win_term`：DIVE 窗内开抓金窗收 `grab_trigger`、抓金窗自收 `grab_complete`、
grab 档确实压在 portal/retreat 之上、旗关时该档惰性）、`GrabEscrowSettlementTests`
（直接驱动出厂的 `WorkerWindowEnv._descend_escrow_settlement`，L1/L2、四个危险理由、
七个无害理由、定场、旗关、death/retreat_trigger 不变，按 R18-B5 的同款 fake-self 先例）、
`GrabWiringTests`（真 `OptionsEnv` 构造：`_retreat`/`_portal` 确实绑上了，默认臂仍不造服务），
以及 `ReviewRoundPendingTests` / `ReviewRoundPortalDeferenceTests` / `ReviewRoundFloorCapTests`。

### 7.2 新测试确实咬人（反向验证）

把同一个测试文件放到**修复前**的三份源文件上（`/home/laure/r17_work/r19/rr/pre-tree`，
只把 `resource_gold_grab.py` / `options_env.py` / `worker_env.py` 换回改动前的版本）：
**24 failed / 92 passed**，失败正是上面 6 条对应的用例；在修复后的树上
**112 passed / 37 subtests passed / 0 failed**。

### 7.3 回归与两臂重跑（全部重跑，不是沿用）

- **OFF 回归（旗关）**：`rows_sha_v3 = 0e5a1acd2fb2c07cd27eb72f9ecc574322e72f362ef0376b692376d9a99be886`，
  期望 `0e5a1acd2fb2c07cd27eb72f9ecc574322e72f362ef0376b692376d9a99be886`，**equal = true**，48 行，
  RuntimeError 0，621.5s。
  行文件 `/home/laure/r17_work/r19/gold-probe-rr/gold2-off-merged-rows.json`。
- **ON 臂重跑**（16 键 + `resource_gold_grab=gold-grab-v1`，pool 2_133 48 种子 3 分片，
  worker 7e31dc54，max_steps 6000，decoding sample，探针 `r17-deployment-v3-r18m2`）：
  RuntimeError 0，630.8s，
  行文件 `/home/laure/r17_work/r19/gold-probe-rr/gold2-on-004806.json`。
- 邻域回归：**11 failed, 554 passed, 4 skipped, 711 subtests passed**；11 条失败与纯净副本 `base-tree` 的失败集合**逐条相同**
  （`/home/laure/r17_work/r19/rr/rr-fails.txt` 与 `base-fails.txt` diff 为空），
  全部来自 `test_r18b6_training_wiring.py` 读不到本副本没拷贝的
  `train/runs/r16-arm-a-constitution/model_candidate.zip`。完整 10 分钟全量套件仍**未跑**。

### 7.4 复核轮 ON 臂读数（control = 认证控制行；prefix = 复核前那一轮 ON）

| 指标 | control | ON（复核后） | ON（复核前） |
|---|---|---|---|
| alive | 20 | 21 | 22 |
| L2 到达 | 34 | 30 | 29 |
| L3 到达 | 4 | 1 | 4 |
| L1 死 | 7 | 7 | 8 |
| L2 死 | 19 | 20 | 17 |
| L2 beats | 85578 | 79203 | 64618 |
| L2 hazard/1k | 0.222 | 0.2525 | 0.2631 |
| 首次下楼 beat 中位 | 5535 | 5670.0 | 5851.5 |
| 有下楼的局数 | 41 | 40 | 38 |
| 下楼时腰带中位 | 4 | 5.0 | 4.0 |
| 死时腰带中位 | 2.5 | 0 | 4.0 |
| gold_final 中位 | 78.0 | 132.5 | 119.5 |
| gold_final 合计 | 6855 | 9178 | 7439 |
| collect 阶段进金 | 12178 | 7826 | 4678 |
| 卖货进金 | 7591 | 7523 | 6250 |
| 花掉的金 | 15442 | 15084 | 14724 |
| 采购次数 | 161 | 150 | 139 |
| loot 行程 | 105 | 103 | 88 |
| 每趟采购 | 1.533 | 1.456 | 1.58 |
| 每趟花费 | 147.07 | 146.45 | 167.32 |

配对 McNemar（vs 认证控制行）：saved 11、lost 10、net 1、
discordant 21、`ucb95_one_sided` **0.13614**。

### 7.5 捡金本身（复核后 vs 复核前）

| | 复核后 | 复核前 |
|---|---|---|
| 窗数 | 656 | 858 |
| 堆数 | 669 | 993 |
| 进钱包 | 7640 | 9240 |
| 微步 | 12402 | 18329 |
| 被拒的 native 请求 | 0 | 0 |
| 未结算 pickup | 0 | 0 |
| L1 窗/堆/金 | 509 / 523 / 4887 | 857 / 992 / 9215 |
| **L2 窗/堆/金** | **147 / 146 / 2753** | 1 / 1 / 25 |
| 有 L2 抓取的局数 | 21 | 1 |
| settle 挂起次数 / 最长结算微步 | 0 / 1 | 无此计数 |
| 窗末仍挂着 pending 的局 | 0 | 无此计数 |
| 窗账不平（gold1-gold0 != gold_taken）的窗 | 0 | 0 |

中止理由：`{"grab_floor_pile_cap": 41, "grab_monster_near": 319, "grab_no_target_in_radius": 290, "grab_pile_cap": 6}`

拒绝理由：`{"cooldown": 4172, "floor_pile_cap": 24236, "low_hp": 401, "monster_near_pair": 3014, "monster_near_pair_unseen": 187, "monster_near_pile": 132, "pile_cap": 2364}`

### 7.6 三条进金渠道（复核后 / 复核前 / control）

| 渠道 | control | ON（复核后） | ON（复核前） |
|---|---|---|---|
| 抓金 | 0 | 7640 | 9240 |
| 行程 collect | 12178 | 7826 | 4678 |
| 卖货 | 7591 | 7523 | 6250 |
| **合计** | **19769** | **22989** | **20168** |

### 7.7 点名的两个种子（复核后）

- **2133002** — control：died=True, depth=1, clvl=1, gold_final=50, trips=0, grab=None堆/None金；ON：died=False, depth=1, clvl=4, gold_final=42, trips=2, grab=12堆/100金
- **2133010** — control：died=True, depth=2, clvl=4, gold_final=22, trips=5, grab=None堆/None金；ON：died=False, depth=1, clvl=3, gold_final=56, trips=2, grab=12堆/143金

### 7.8 复核轮之后仍然成立的话（不要读成"已认证"）

- OFF 回归通过，`sha` 与认证控制行逐位相同 —— 旗关臂仍是逐位同构的。
- 机械层面：0 个被拒 native 请求、0 局在窗末还挂着 pending、窗账全部平。
- **仍未认证非有害**：配对 `ucb95_one_sided` 见 7.4，只要它 > 0 就不是"证明无害"。
- L2 半边现在**有**了数据（见 7.5 的 L2 行），但样本仍然只有这 48 个 `sample` 解码种子一轮；
  复核者建议"把 L2 数据作为过闸前置条件"，这一轮只是把它从**结构性不可达**变成**可测**，
  离"够证"还差得远。
- 训练侧仍然只有直通（没有 `train_ppo` CLI 旗 / `make_env` 转发），所以这部法今晚**训不了**；
  但（1）修好之后，第一次同时打开 `worker_descend_escrow_fraction > 0` 和本旗时，
  托管通道已经不再是洞。


### 7.9 复核轮这一臂怎么读（含**变坏**的那几行）

**好的方面。**
- 每层上限把主席裁定的 L2 半边从"结构性不可达"变成了**真的被测到**：
  L2 147 扇窗 / 146 堆 / 2753 金，分布在 48 局里的
  21 局（复核前：1 扇窗、1 堆、25 金、1 局）。
- 结算通道全绿，而且这一次是**可量化的**全绿：pickup_commands 669
  = accepted 669 = piles_taken 669，
  rejected 0，unresolved 0，
  窗末仍挂 pending 的局 0，窗账不平的窗
  0，**最长一次结算 1 微步**
  （上限 8）。也就是说复核者担心的那条"pending 跨界"路径在这 48 局里从未走到过——
  这正是他说的"未测到，不是已证安全"；现在它有单元测试（`ReviewRoundPendingTests`），
  但**田野证据仍然是 0**（`settle 挂起 0 次`）。
- 总收入这次真的多了：三渠道合计 control 19769 → ON 22989
  （+16.3%），
  不再是复核前那种"只提早、不增量"（+2%）。`gold_final` 中位
  78.0 → 132.5，合计 6855 → 9178。

**变坏 / 需要点名的方面（请复核者按这几行读）。**
- **仍未认证非有害**：saved 11 / lost 10 / net 1，
  `ucb95_one_sided` **0.13614** > 0。比复核前的 0.11128 **还大**，
  因为 lost 从 9 涨到 10。alive 只从 20 涨到 21。
- **深度还是掉**：L2 到达 34 → 30，**L3 到达 4 → 1**，
  L2 beats 85578 → 79203，L2 死 19 → 20，
  hazard/1k 0.222 → 0.2525。现在窗口开到了 L2，所以
  "分母变小"这个复核前的托词不再能全部解释它：L2 beats 只小了 7.4%，L2 死却多了一个。
- **死时腰带中位 2.5 → 0**。这是本轮最难看的一行，
  而且方向和复核前那一轮（4.0）相反。它可能只是"死在 L2 的样本换了一批"
  的噪声，也可能是抓金窗在 L2 多占了身体、把喝药/撤退的时机推后了——
  **本轮没有做因果归因，不要当结论读**。
- **撤退让位条款仍然 0 次点火**：中止理由里没有 `grab_retreat_wanted`、也没有 `grab_low_hp`，
  尽管这次 L2 上开了 147 扇窗。复核者说得对，这是widen 到 L2 的整个安全论证所依赖的
  那一条，它**仍然没有田野证据**（只有单元测试）。`grab_portal_wanted` 同理，而且本臂
  `resource_portal` 根本是关的（16 键覆盖里没有它），所以传送让位这一条**今晚完全不在场**。
- 收入结构变了：行程 collect 从 control 的 12178 掉到 7826，
  抓金拿走 7640——同一批堆，只是被更早地拿走了。

**建议（不变，且更强了）。**先不要为这部法开闸。要开，至少需要：
(1) 一轮把 `resource_portal` 一起打开的 ON 臂，让 `grab_portal_wanted` 有机会点火；
(2) 一个能让 `grab_retreat_wanted` / `grab_low_hp` 真的点火的场景（或明确承认这两条只有单元测试）；
(3) 更多种子 / 更多解码轮次，把 `ucb95_one_sided` 推到 0 以下；
(4) 对"死时腰带中位掉到 0"做一次逐局归因。

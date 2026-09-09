# R19-A1 报告：l1-opening retreat（`resource_retreat_scope = l1-plus`）

- 日期：2026-09-08
- 实现者标签：`r19-a1-implementer`
- 工作树：`/home/laure/r17_work/r19/a1-tree`（Python only；`build -> /home/laure/r17_work/r17-1/build-res`，即主树同一条活桥）
- 主树 `/home/laure/AlphaDiablo/diablogym` 未被修改（唯一例外：账本 append）
- 补丁：`/home/laure/r17_work/r19/a1.patch`（5 个文件，+228 行，无删除逻辑）
- 引擎 `/home/laure/alphadiablo-dev/devilutionX`、`src/*.cpp|hpp` 一律未改；今晚无新引擎补丁

---

## 一、法条与它的载具（先说结论：楼梯载具在 L1 结构上不可用）

`retreat-v1` 的 R18-A 撤退是「授权一次上楼收据」：

- `src/resource_protocol.hpp:204-209`：只有 `WM_DIABPREVLVL && source >= 2 && target == source-1`
  这一支才会给出 `retreat_ascent` 收据；
- `src/resource_protocol.hpp:391-400` `ConfigureRetreat`：`currlevel < 2` 直接 throw
  “retreat can depart only from main L2 or deeper”；
- 而 L1 → 城 的那道上楼梯，走的是**另一支**：
  `src/resource_protocol.hpp:216-217`，`source == 1 && target == 0` → `town_service_departure`，
  由 `gTownServiceAuthorized` 与战利品经济的行程预算把关。

所以 **「用 R18-A 的楼梯载具从 L1 撤退」在当前服务里结构上不可能**，除非改 C++（今晚禁止）。
本实现采用章程允许的「最小诚实变体」：

> **同一条触发法（阈值一字未改）在主线 L1 上开口；载具换成普通的
> loot-economy 进城行程**——脚本走上楼梯、升城、跑完整套进城行程
> （Pepin 治疗 / 装备 / 药水 / 卖货），再由行程自己的 `return` 阶段
> 走楼梯下回 L1，与任何一次进城行程之后完全一样。

进城腿的入口逐字沿用 R18-F `portal_arrival` 的先例
（`options_env.py` 中把 `service.phase` 直接置为 `"outbound"` 并显式
`bridge.configure_town_service(True)`），因此**撤退行程跳过 collect（捡金）阶段**：
这趟行程的目的是**离开**这层，不是搜刮这层。这一点是刻意的，且与 portal 先例同款。

### 三个组合问题的答案（都来自读到的代码，不是推测）

1. **L1 的上楼是否已经算一次进城行程？**
   **是。** 收据是 `town_service_departure`；`ResourceAfterLoad`（hpp:378-381）在 depth==0
   且该收据成立时置 `gTownServiceTrip=true` 并 `++gResourceServiceTripsStarted`。
   也就是说一次 L1 撤退**花掉**一个行程额度（额度 = `2 + retreats_started`，
   `resource_sustain_loot.py:182-186`）。
2. **每局 3 次的撤退上限还成立吗？**
   引擎侧不成立：`gResourceRetreatsStarted` 只在 `retreat_ascent` 抵达时自增（hpp:340-343），
   L1 撤退永远不会被它数到。**故在 Python 侧把上限按并集执行**：
   `trigger_reason` 现在判 `retreats_started(raw) + self.l1_retreats >= 3`。
   默认 scope 下 `l1_retreats` 恒为 0，判词与冻结法逐字等价。
3. **gold 100 的行程走得通吗？**
   走得通。行程链路上没有任何金币门槛，Pepin 的治疗本身免费，采购计划器按现金买它买得起的东西。
   实测：42 次 L1 撤退里 37 次抵达城内并跑完行程；全臂进城行程数 105 → 134。

---

## 二、改了什么（词汇表登记，逐条与既有法条同位）

| 位置 | 内容 |
|---|---|
| `python/diablogym/resource_protocol.py` | `RESOURCE_RETREAT_SCOPES = ("l2-plus", "l1-plus")`；`validate_retreat_scope(retreat, scope)`——非默认值必须 `resource_retreat == "retreat-v1"`，否则 fail closed |
| `python/diablogym/resource_retreat.py` | `RetreatService.scope` 字段 + `min_depth` 属性；`trigger_reason` 的深度判词改用 `min_depth`；上限改并集；新增 `start_l1` / `note_l1_denial` / `observe_l1` / `_close_l1`；遥测新增键 |
| `python/diablogym/options_env.py` | 构造器 kwarg `resource_retreat_scope`（**不下传 DiabloGymEnv**，与 `resource_calibration` 同理：native 侧没有 scope 这个词）；`RetreatService(scope=...)`；`action_masks` 里新增 **l1 腿**；两处遥测出口前调用 `observe_l1` |
| `python/diablogym/worker_env.py` | 训练侧同一个 validator 直通，默认 `l2-plus` 时 kwargs 逐字节不变 |
| `train/runs/r10-staging/probe_r17_deployment.py` | `_R16_ENV_KEYS` 增 `"resource_retreat_scope"`；每行 resource 遥测增 `retreat_scope` / `l1_retreats` / `l1_retreat_deaths_walking` |

两处必须点名的守卫（复核者请重点看）：

- 冻结的 L2 撤退块现在带上显式 `dungeon_level >= 2` 项。默认 scope 下
  `trigger_reason` 本来就在 depth<2 返回 None，故该项**不带任何行为**；它的作用是
  在 scope 放宽后仍把那一块钉死为 L2+ 的块。（不加会直接炸：
  `RetreatService.start` 在 L1 抛 “retreat can start only on main L2 or deeper”——
  这是本次实现中真实踩到并修好的第一个缺陷。）
- l1 腿带 `not sweep.active` 项。R18-M 在 `options_env.py` 留了明文告诫：
  “Widen the sweep's depth (or copy this block for a new leg) and that stops being
  true — add `and not (sweep is not None and sweep.active)`”。本腿正是把撤退搬到
  sweep 自己的楼层（主线 L1）的那条腿，所以这一项在这里、且只在这里出现。

### 遥测（只增不改）

- `attempts[]` 每条增 `"floor"`（触发所在层）与 `"vehicle"`（`stairs_ascent` / `town_trip`）；
  L1 条目另有 `"trip"`（它占用的行程序号）。R18-A 原有 12 个键与取值一字未动
  （`tests/test_r19_retreat_scope.py::DefaultArmIsByteIdenticalTests` 逐键钉死）。
- 计数器：`l1_retreats`、`l1_retreat_deaths_walking`（在 L1 走路途中死）、
  `l1_ascended`、`l1_denials[]`（法条开口但无载具，记 `trip_limit` 等原因并罚
  `RETREAT_FAILURE_COOLDOWN=300` 微步，防止窗口空转）。

---

## 三、测试

- 新增 `tests/test_r19_retreat_scope.py`：**30 tests / 92 subtests，全绿**
  （`/home/laure/r17_work/r19/a1-tree`，`PYTHONPATH=python:train`）。
  覆盖：L2 上两个 scope 判词逐例相同（13 个案例 × 深度 2/3 + 压力子句 + 冷却/active 子句）；
  L1 只在 `l1-plus` 开口且阈值同款（50%、51%、空带 75%、76% 边界）；
  城（depth 0）在任何 scope 下都不是撤退层；
  `start()` 在 L1 仍然拒绝（楼梯载具的 fail-closed）；
  3 次上限按 engine+L1 并集；denial 冷却；`observe_l1` 的四种结局
  （town_reached / died_walking / trip_ended / abandoned）与幂等；
  validator 在没有 `retreat-v1` 时拒绝该词；默认臂遥测是 R18-A 载荷的严格超集。
- 邻域回归（retreat / portal / protocol / sustain-loot / sweep / identify /
  loot-protocol-integration / probe-sustain）：**448 passed, 410 subtests passed**。
- 全部 import `options_env|worker_env|retreat|resource_sustain` 的测试文件
  （`/home/laure/r17_work/r19/tests-wide.log`）：**1756 passed / 39 skipped / 24 failed**。
  这 24 条**与本改动无关**：在同一台机、同样只 rsync 了 python/tests/train/*.py 的
  **纯净副本** `/home/laure/r17_work/r19/base-tree` 上跑同样三个文件，失败集合
  **逐条相同**（`comm` 双向差集为空，见 `/home/laure/r17_work/r19/tests-baseline.log`）。
  它们是 `test_content_case_infra.py` / `test_r18b6_training_wiring.py` /
  `test_r9_machinery.py` 里读取本副本没有拷贝的启动脚本与 CLI 资产的用例。
  完整 10 分钟全量套件未跑（**not verified**）。

---

## 四、探针（pool 2_133，48 种子，3 分片并行，worker 7e31dc54）

命令与原始产物：`/home/laure/r17_work/r19/probe/`（`a1_probe_driver.py`，
`*.cmd` / `*.log` / `*.json`），驱动脚本 `/home/laure/r17_work/r19/a1_probe_driver.py`。
探针版本 `r17-deployment-v3-r18m2`，`max_steps 6000`，decoding `sample`，
覆盖 = `/home/laure/r17_work/r18/gates/v1world-overrides.json` 的 16 键。

### 4.1 OFF / 默认臂（回归）——**通过**

```
rows_sha_v3 = 0e5a1acd2fb2c07cd27eb72f9ecc574322e72f362ef0376b692376d9a99be886
expected    = 0e5a1acd2fb2c07cd27eb72f9ecc574322e72f362ef0376b692376d9a99be886
equal = true   n_rows = 48   runtime_errors = 0   elapsed = 653.1 s
```

与认证控制行 `/home/laure/r17_work/r18/merge-probe/ctl-v1-world-193557.json` **逐位相等**。
派生统计亦逐项复现（L2 触达 34、L2 hazard 0.2220、首降中位 5535、L1 死亡 7、
retreats 71、chests 253）。默认 `l1_retreats = 0`。

### 4.2 ON 臂（控制覆盖 + `{"resource_retreat_scope": "l1-plus"}`）

`rows_sha_v3 = 1954e35f4d0100da1886b4c50295539ea41e4093bda66f00bec03b5532f2257c`
（`/home/laure/r17_work/r19/probe/a1-on-204811.json`），`runtime_errors = 0`，elapsed 635.4 s。

**配对 McNemar（存活，公式同 `m2_probe_driver.py:paired`）**

| | |
|---|---|
| saved | 8（2133010, 2133019, 2133020, **2133021**, **2133022**, 2133025, **2133038**, **2133039**） |
| lost | 8（2133011, 2133016, 2133023, 2133029, 2133031, 2133034, 2133035, 2133046） |
| net | **0** |
| discordant | 16 |
| ucb95_one_sided | **0.13708** |

**arm_stats（ON vs 控制行）**

| 指标 | ON | 控制 |
|---|---|---|
| alive | 20 | 20 |
| **L1 死亡** | **9** | **7** |
| L2 死亡 | 16 | 19 |
| L3 死亡 | 3 | 2 |
| L2 触达 | **29** | **34** |
| L3 触达 | 4 | 4 |
| L2 停留拍 | 81 599 | 85 578 |
| **L2 hazard /1k** | **0.1961** | **0.2220** |
| **首降拍中位**（有降的局） | **1696**（43 局） | **5535**（41 局） |
| L2 击杀 | 771 | 894 |
| gold（末次存活列）中位 / 合计 | 111.5 / 7478 | 150.0 / 9127 |
| `resource.gold_final` 中位 / 合计 | 82.5 / 5252 | 78.0 / 6855 |
| 进城行程数 | **134** | 105 |
| 开箱 / 砸桶 / sweep 窗 | **36 / 23 / 11** | **253 / 231 / 104** |
| 鉴定件数 / 花费 / 鉴定后售价收入 | 12 / 1200 / 2606 | 16 / 1600 / 3769 |
| 武器升级 / 花费 | 24 / 5560 | 29 / 6620 |
| 撤退 attempts 合计 | 98 | 71 |
| 其中 L2+ 楼梯成功 | 50 | 55 |
| **L1 撤退 / 抵城 / 走路途中死 / 被拒** | **42 / 37 / 5 / 19** | 0 / 0 / 0 / 0 |

**L1 撤退普查（42 次，分布在 42 个不同种子；每局至多 1 次）**

- 触发词全部 `low_hp`（`empty_belt` 一次也没触发：触发时 `belt0 == 0` 的次数为 **0**）；
- 触发时 hp 比例中位 **0.486**，触发拍中位 **650**，邻近怪中位 **6**；
- 从触发到结算中位 **158 拍**；
- 结局：`retreat_l1_town_reached` **37**，`retreat_l1_died_walking` **5**；
- 19 次法条开口后被拒，原因**全部**是 `trip_limit`。

### 4.3 预登记点名的七个种子（控制行中在 L1 死亡）

| 种子 | 控制 | ON | L1 撤退（beat0 hp0/max 邻怪 → 结局 @beat1） |
|---|---|---|---|
| 2133002 | L1 死 @559，clvl1，21 杀 | **L2 死 @16687**，clvl4，174 杀 | 508，29/70，10 → 抵城 @823 |
| 2133021 | L1 死 @601，clvl1，22 杀 | **存活**，clvl4，127 杀 | 310，34/70，1 → 抵城 @373 |
| 2133022 | L1 死 @1017，clvl1，29 杀 | **存活**，clvl4，142 杀 | 830，34/70，10 → 抵城 @1185 |
| 2133038 | L1 死 @1524，clvl2，67 杀 | **存活**，clvl3，128 杀 | 652，34/70，7 → 抵城 @879 |
| 2133006 | L1 死 @1531，clvl2，61 杀 | **L2 死 @7590**，clvl3，101 杀 | 404，32/70，7 → 抵城 @606 |
| 2133039 | L1 死 @2409，clvl2，74 杀 | **存活**，clvl3，76 杀 | 2053，33/78，11 → 抵城 @2382 |
| 2133014 | L1 死 @2433，clvl2，71 杀 | **仍 L1 死 @1691**，clvl1，30 杀 | 578，35/70，7 → 抵城 @704 |

**7/7 都开了口，7/7 都抵达了城**：4 局不再死，2 局把死亡从 L1 推到 L2（推后 6 000–16 000 拍），
1 局（2133014）撤退成功、回到 L1 后仍然死在 L1（比控制早，因为该局的整条轨迹已经不同）。
这一段就是预登记 §二 所诊断的机制本身——它确实存在、确实起作用。

---

## 五、判读（诚实版）

**机制成立，臂级不成立。**

- 被点名的病灶（L1 首次进城前的 7 次死亡）确实被这条法条命中：全部触发、全部抵城、
  4 例转为存活。
- 但全臂 **net 0**（saved 8 / lost 8，ucb95 = 0.137，远不能声称改善），
  **L1 死亡反而 7 → 9**，**L2 触达 34 → 29**。唯一稳健的正向读数是
  L2 hazard 0.2220 → 0.1961（分母也小了 4 000 拍，须谨慎）。
- 代价的机制是清楚的、可测的：**一次 L1 撤退花掉一个进城行程额度，而不是挣来一个**
  （§一.1/§一.2）。撤退在 farm cap 之前就把第一趟行程用掉，于是
  - `sweep-v1` 的 `loot_trip_slots_left` 守卫关门：开箱 253 → 36、砸桶 231 → 23、
    sweep 窗 104 → 11；
  - 鉴定腿与铁匠腿随之缩水（12/16 件、24/29 次），全臂 gold（末次存活列）9127 → 7478；
  - 首降中位 5535 → 1696：进城行程把 readiness 提前喂饱，于是 pair 早早下楼，
    带着更薄的身家进 L2 —— 这正是新增 L2/L3 死亡与 lost 的 8 局的来处。
- 因此我**不建议**按现状把 `l1-plus` 收进训练世界。让它变成净收益的那一步，
  是让 L1 撤退**挣来**它的行程额度（引擎 `gResourceRetreatsStarted` 或
  `MaxLootServiceTrips` 侧的一行），而那是 C++ 改动，今晚被明令禁止。

---

## 六、已知缺口（未做 / 未验证，逐条点名）

1. **训练侧词汇表未登记**：`train/train_ppo.py`（`_training_contract` / `_require` /
   CLI）、`train/eval_contract.py`、`train/migrate_loot_candidate.py` **都没有**加
   `resource_retreat_scope`。今晚的任务是部署侧探针，故只登记了
   options_env / worker_env / probe / validators 四处。**任何用 `l1-plus` 起训的臂
   都会被现有训练契约以“环境契约漂移”拒绝**——这是 R18-M2 §5.1 对 `hunt_scope`
   记录过的同一种拒绝。补齐前不要下达训练命令。
2. **L1 腿没有自己的原生收据**：它的收据就是普通的 `town_service_departure`，
   所以事后无法只凭引擎收据把「撤退行程」与「普通行程」分开；分辨只能靠
   Python 遥测（`attempts[].vehicle == "town_trip"`）。
3. **L1 撤退途中没有脚本喝药**：R18-A 的 `drink_hp_fraction=0.4` 只在楼梯载具的
   `_command` 里生效；L1 上身体归进城行程所有，本腿不注入喝药。5 次走路途中死亡
   （`retreat_l1_died_walking`）里有多少可以靠喝药避免，**未验证**。
4. **撤退行程跳过 collect（捡金）阶段**（沿用 portal_arrival 先例）。是否应改为
   「先捡再走」，**未做对照**。
5. **`_win_term` 的 `retreat_trigger` 提前收窗现在也会在 L1 触发**，
   因而 `worker_env.py:2466` 的托管罚没在 `l1-plus` 下会落到 L1 窗上。
   探针路径不走那条结算，**今晚未被验证**。
6. **denial 冷却是共享的**：一次 L1 `trip_limit` 拒绝会把 `_cooldown_until` 推后 300 微步，
   期间 L2 的撤退法条也被一并压住。影响量级**未测**。
7. 全量 10 分钟测试套件未跑；只跑了 §三 所述邻域与 43 个相关文件。
8. 只有一个 pool（2_133，48 种子）、一个 worker、一次运行；处女池
   2_116-119 / 2_126-128 **零接触**（本次未使用、未读取）。

---

## 七、复现命令

```bash
# 树
/home/laure/r17_work/r19/a1-tree            # build -> /home/laure/r17_work/r17-1/build-res
# 测试
bash /home/laure/r17_work/r19/run_tests_new.sh      # 新增用例
bash /home/laure/r17_work/r19/run_tests_wide.sh     # 邻域 + 相关文件
bash /home/laure/r17_work/r19/baseline.sh           # 纯净副本对照（失败集合相同）
# 探针（两臂并行）
bash /home/laure/r17_work/r19/run_both.sh
# 分析
bash /home/laure/r17_work/r19/run_an.sh             # -> probe/a1-analysis.json
```

---

## 八、复核轮（2026-09-08 夜，fix-and-verify）

复核意见给出 3 条 blocker + 4 条 major + 1 条 minor。本节逐条说明**做了什么**、
**在哪一行**、以及**哪个数字变了**。所有数字来自本轮亲自跑出来的文件
（`/home/laure/r17_work/r19/probe-rv/`、`tests-wide-*-full.log`），不是估计。

> **重要：§4.2 的 ON 臂数字已被本轮的重跑取代。**旧的
> `rows_sha_v3 = 1954e35f…` 是在两条 blocker 仍然存在的树上跑出来的，
> 它测的不是这条法条本身（见 B2）。新的 ON 臂在 §8.3。

### 8.1 blocker（3 条，全部已修）

**B1. 托管罚没（`worker_env.py`）在 `l1-plus` 下会落到主线 L1。**
R18-B 的罚没条款只写了 `close_reason == "retreat_trigger"`，没有深度项；
R19-A1 让触发法在 L1 也能说话，于是一条为 main L2+ 裁定的冻结奖励法被无旗守卫地
扩张了一层。而且紧邻的 R18-B6 注释块里那句**书面证明**
（"RetreatService.trigger_reason 在 dungeon_level < 2 直接返回 None"）被 A1 证伪了，
却没人改它——文件在为自己守卫的代码说假话。

修法（照传送孪生的先例，不是加一个裸的深度常量）：新增
`WorkerWindowEnv._retreat_close_is_main_l2_plus()`，罚没条款改为
`... and close_reason == "retreat_trigger" and self._retreat_close_is_main_l2_plus()`。
该方法在 **`scope == "l2-plus"` 时不读任何东西直接返回 True**，所以冻结世界
是*按构造*逐位不变的，而不是靠论证；只有 `l1-plus` 才去问活的 raw，且问不到
（拿不到 raw / 任务定场）时选择**不罚**——沿用传送条款"宁可少罚，不可罚跑腿"。
R18-B6 的证明段落已重写：两个罚没集合与 sweep 窗仍然构造性不交，但现在靠的是
**两个显式深度项**（`_win_term` 的 rung + 本方法），而不是触发法自己的深度条款。

**B2. `_win_term` 的 retreat rung 是第二个未加作用域的消费者。**
这是本轮影响最大的一条。在第一版里，`l1-plus` 下主线 L1 上任何非 RESUPPLY 窗
只要 hp ≤ 50%（或空腰带 ≤ 75%）就被强行以 `"retreat_trigger"` 收窗——**包括随后
被拒绝的那些拍**（trip_limit、服务/sweep/portal 在跑）。于是 ON 臂的 L1 窗结构
与控制臂的差异有一部分根本不是撤退机制造成的，旧报告的因果结论
（"L1 loot 经济崩塌"）是断言而非论证。

修法：rung 加显式 `int(raw.get("dungeon_level", 0)) >= 2`，且**刻意不配一条 L1 rung**。
L1 腿本来就不需要 rung：它跟 `SustainLootService.maybe_start` 一样从
`action_masks` 出发，被它打断的那个窗按普通的 scene 条款在走到城里的那一拍收掉。

**B3. 训练/档案词汇表未登记（这条法条根本不可被训练或认证）。**
已按 `resource_portal` / `hunt_scope` 的先例补齐：

| 文件 | 登记点 |
| --- | --- |
| `train/eval_contract.py` | `R16_ENVIRONMENT_DEFAULTS["resource_retreat_scope"]="l2-plus"`；`validate_r16_environment` 新分支：`l1-plus` 必须同时写下 `resource_retreat=retreat-v1` |
| `train/eval_assembled.py` | `--resource-retreat-scope`（choices l2-plus/l1-plus）、对应 `ap.error`、进 `requested_r16` |
| `train/train_ppo.py` | `_training_contract`（默认写 None）、`_ENVIRONMENT_RESTART_ALLOWED_DRIFT`、`_validate_resource_resume_identity` 的恒等键组、`_validate_args` 的三条 `_require`、`make_env` 形参 + 三条 `_require` + `resource_kwargs`、argparse、第二处契约写入、`make_env` 调用点 |
| `train/migrate_loot_candidate.py` | `target.setdefault("resource_retreat_scope", None)`（照 R18-M 对 `resource_portal` 的裁定：只有冻结缺省可铸，`l1-plus` 的热启动是另一次刻意行为） |

**未验证**：登记本身让 `_training_contract` 多一个无条件键，按 R18-B6 记录过的
同一种 key-presence 会计，**每一个 schema/2 臂的 `target_contract_sha256` 都会移动**，
旧收据需重铸。今晚没有热启动运行，故**这个新 sha 没有被测量**，我不给数字。

### 8.2 major（4 条：3 条已修，1 条一半代码修一半明写为局限）

**M1. 三次的额度被 L1 腿吃掉（把一条冻结法条悄悄收窄了）。**
第一版把 cap 写成 `retreats_started(raw) + self.l1_retreats >= 3`，而且
`l1_retreats` 在 **start**（尝试）时自增，`retreats_started` 只数引擎确认的**到达**。
修法：`RETREAT_L1_MAX_PER_EPISODE = 3`，`trigger_reason` **按楼层选预算**——
`depth >= 2` 走 `retreats_started(raw) >= RETREAT_MAX_PER_EPISODE`（R18-A 原文），
`depth == 1` 走 `self.l1_retreats >= RETREAT_L1_MAX_PER_EPISODE`。
证据（本轮 ON 臂）：**L2 楼梯尝试 77 次 vs 控制 71 次**，L2 楼梯到达 65 vs 55——
第一版是 56 vs 71（−21%）。冻结的 L2 法条不再被挤掉，反而因为身体活得久而多用了。

**M2. denial / 失败腿写的是共享冷却，会把冻结的 L2 法条静音 300 微步。**
修法：新增 `_l1_cooldown_until`，只有 L1 分支读写它；`_cooldown_until` 重新变成
R18-A 楼梯法的独占物。`trigger_reason` 同样按楼层选钟。测试两个方向都钉住
（L1 拒绝后 L2 仍说话；L2 冷却中 L1 仍可走）。

**M3. 30 个用例全是对手写替身的单元测试，没有一个碰到臂真正跑的那段代码。**
修法两步：(a) 把 `action_masks` 里的订票法抽成
`OptionsEnv._maybe_start_l1_retreat()` 与 `OptionsEnv._l1_retreat_vehicle_ready()`；
(b) 新文件 `tests/test_r19_retreat_scope_integration.py`，用**真的 `SustainLootService`**
驱动这两个生产方法，并且直接钉住复核点名的那条 off-by-one：
`start_l1(trip_count+1)` → `_begin_trip` 自增 → `_seal_trip` 写 `_trips[-1]["trip"]`
→ `observe_l1` 结算为 `ascended`（`test_the_trip_index_handed_to_start_l1_is_the_trip_the_service_opens`，
对 0 次和 1 次已有行程各跑一遍）。同一文件还钉住 `_win_term` 的作用域与托管罚没
的深度作用域。

**M4. 载具没被问过就订票；而且 L1 腿没有喝药。**
- **代码已修**：`_l1_retreat_vehicle_ready()` 在订票前问与
  `ResourceService._stairs` 一模一样的两个问题（视野里有没有 `WM_DIABPREVLVL` 触发器、
  `_plan_descend_path(..., avoid_monsters=True)` 有没有路），没有就记
  `no_vehicle` 拒绝、**不花任何额度**。本轮 ON 臂里这条拒绝**真的发生了 4 次**
  （`l1_denial_reasons = {"no_vehicle": 4, "trip_limit": 2}`）——即第一版会有 4 次
  白白烧掉一个 loot 行程槽 + 一次撤退额度、并把半血的角色留在原地。
- **未修，明写为局限**：L1 腿仍然**没有**脚本喝药。R18-A 的 `drink_hp_fraction`
  只活在楼梯载具的 `_command` 里；L1 上身体归进城行程所有，本轮没有往行程的
  命令流里注入喝药（那是改 loot 服务，风险与今晚的授权不相称）。本轮 30 次 L1 腿
  里 **2 次死在路上**（第一版 42 次里 5 次）。**这 2 次里有多少能靠喝药避免，未验证**；
  必须把它读成"本腿是在缺了 retreat-v1 一半（喝药那一半）的条件下被考核的"。

### 8.3 修好之后的 ON 臂（同池、同 worker、同探针，3 分片并行）

OFF/默认回归（先跑）：`rows_sha_v3 = 0e5a1acd2fb2c07cd27eb72f9ecc574322e72f362ef0376b692376d9a99be886`，
**与认证控制行逐字相等**，48 行，0 RuntimeError，650.4 s。

ON = 16 条控制覆盖 + `{"resource_retreat_scope": "l1-plus"}`，
`rows_sha_v3 = 0e388a102698d70116e2426337266a8da553c6da34f54d00176334c9064c9389`，
0 RuntimeError，679.5 s，产物 `/home/laure/r17_work/r19/probe-rv/a1-on-221543.json`。

配对 McNemar vs `ctl-v1-world-193557`：
**saved 7 / lost 3 / net +4 / discordant 10 / ucb95_one_sided 0.02322**。
（第一版：saved 8 / lost 8 / net 0 / ucb95 0.13708。）
`ucb95_one_sided` 仍 > 0，**按预登记的判读规则这仍不构成可主张的改进**；
但它从 0.137 掉到 0.023，方向与幅度都变了。

| 指标 | 控制 | ON（复核后） | ON（第一版，作废） |
| --- | --- | --- | --- |
| 存活 | 20 | **24** | 20 |
| L1 死亡 / L2 死亡 / L3 死亡 | 7 / 19 / 2 | 5 / 17 / 2 | 9 / 16 / 3 |
| L2 抵达 / L3 抵达 | 34 / 4 | 33 / **5** | 29 / 4 |
| L2 拍数 / hazard per 1k | 85578 / 0.2220 | 83127 / **0.2045** | 81599 / 0.1961 |
| 首次下潜拍数中位数（行数） | 5535 (41) | **2848.5 (44)** | 1696 (43) |
| loot 行程 | 105 | 134 | 134 |
| 箱子 / 木桶 / sweep 窗 | 253 / 231 / 104 | 111 / 96 / 53 | 36 / 23 / 11 |
| 鉴定件数 / 鉴定金 / 鉴定件售金 | 16 / 1600 / 3769 | 17 / 1700 / 3394 | 12 / 1200 / 2606 |
| 武器升级次数 / 花金 | 29 / 6620 | 25 / 5910 | 24 / 5560 |
| `gold_final` 中位数 / 合计 | 78.0 / 6855 | 64.5 / 6496 | 78.0 / 5252 |
| **L2 楼梯尝试 / 其中到达** | **71 / 55** | **77 / 65** | 56* / 50 |
| **L1 腿尝试 / 到城 / 途中死** | 0 / 0 / 0 | **30 / 28 / 2** | 42 / 37 / 5 |
| L1 拒绝（按理由） | — | `no_vehicle` 4, `trip_limit` 2 | `trip_limit` 19 |
| L1 触发词 | — | `low_hp` 20, `empty_belt` 10 | `low_hp` 42, `empty_belt` 0 |
| L1 触发时血量比中位数 / 触发拍中位数 / 结算耗拍中位数 | — | 0.500 / 1163.5 / 334.5 | 0.486 / 650 / 158 |

\* 第一版报告写的是"撤退尝试 98 vs 71"这一个合并数——复核意见点名批评过这种写法。
本轮的驱动器（`a1_probe_driver_review.py`）已经把两个楼层拆成
`retreats_l2_attempts` 与 `l1_attempt_records` 两个键，**不再产生合并数**。

**判读（诚实版，本轮）**：
1. 机制本身工作，而且比第一版干净得多：30 次 L1 腿里 28 次真的到城，只有 2 次死在路上；
   6 次拒绝全部有明确理由，其中 4 次是新加的载具前置检查挡下来的。
2. **冻结的 L2 撤退法被修复后不再被挤压**（77/65 vs 71/55），这是 M1+M2 的直接证据。
3. **仍然要花 L1 的 loot 经济**：箱子 111 vs 253、sweep 窗 53 vs 104。这一次这个代价
   是**真的**——它不再混着 B2 的窗截断假象，因为一次 L1 撤退确实占掉一个行程槽
   而不挣回一个（`gResourceRetreatsStarted` 只数 `retreat_ascent` 到达）。
   让 L1 撤退挣回自己的行程槽是一处 C++ 计数器改动，**今晚被明令禁止**。
4. 首次下潜从 5535 提前到 2848.5 拍，且 44/48 行有下潜（控制 41/48）；L2 hazard 略降
   （0.2045 vs 0.2220），L3 抵达 5 vs 4，存活 24 vs 20。
5. **结论不变但更强**：`ucb95_one_sided = 0.02322 > 0`，**不予采纳**（still not a
   claimable improvement）。差一点。要跨过去，最该先做的两件事按代价排序是：
   (a) 给 L1 腿补上 retreat-v1 的喝药那一半；(b) 让 L1 撤退挣回一个行程槽（C++）。

### 8.4 minor（1 条：接受，未改代码，改为明写）

复核意见指出"默认 OFF 逐位不变"这句话**在紧要处为真、在字面上为假**：
`rows_sha_v3` 只取 17 个固定键，动不了（本轮 OFF 臂再次证明了这一点）；
但默认作用域下 `RetreatService.telemetry()` 确实多了几个键
（`scope` / `l1_retreats` / `l1_retreat_deaths_walking` / `l1_ascended` / `l1_denials`，
本轮又多了 `l1_budget` / `l1_cooldown_until`），且每条 L2 attempt 记录多了
`floor` / `vehicle` 两个键。**这是有意的、只增不改**，测试
`DefaultArmIsByteIdenticalTests` 逐键钉住"新键是这几个、旧键值逐字不变"。
正确的说法是：**行数据与法条行为逐位不变；遥测是既有载荷的严格超集。**
本报告前文若有更强的措辞，以本段为准。

### 8.5 测试

* 新增 `tests/test_r19_retreat_scope_integration.py`（19 用例 / 14 subtests，真 `SustainLootService`）；
  `tests/test_r19_retreat_scope.py` 改写预算/冷却相关用例并加 4 条分离性用例。
  两个文件合计 **53 passed, 116 subtests passed**（34 + 19）。
* `tests/test_resource_retreat_training.py` 的 R18-B 托管替身补绑了新的生产方法，
  并新增 3 条用例把"冻结作用域每层都罚 / `l1-plus` 只在 main L2+ 罚 / 任务定场不罚"钉死
  （26 passed / 39 subtests，全绿）。`test_r18b5_hunt_portal_training.py` 与 `test_r18b6_training_wiring.py`
  的同款替身同样补绑。
* 邻域全跑（74 个相关文件，含 `eval_contract` / `train_ppo` / `migrate_loot` 的消费者）：
  patched **56 failed, 2238 passed, 99 skipped, 3 errors**；
  纯净副本 `base-tree` 同一份选择 **57 failed, 2181 passed, 99 skipped, 3 errors**。
  失败集合逐条对比：**patched 是 base 的真子集**，唯一的差是
  `test_resource_identify.py::…test_the_engine_price_constant_matches_the_python_one`
  在 base-tree 上失败（那份纯净副本没有 `patches/` 目录）而在本树通过。
  **本轮没有引入任何新失败。**
  日志：`tests-wide-base-full.log` / `tests-wide-review-full.log`。
* 全量 10 分钟套件仍未跑（同 §6.7）。

### 8.6 本轮产物

```
/home/laure/r17_work/r19/a1-tree                     # 仍然 build -> r17-1/build-res（未改 C++）
/home/laure/r17_work/r19/a1-review.patch             # 本轮相对 base-tree 的完整 diff（13 文件）
/home/laure/r17_work/r19/a1_probe_driver_review.py   # 拆分 L1/L2 计数的驱动器
/home/laure/r17_work/r19/probe-rv/a1-summary-off.json
/home/laure/r17_work/r19/probe-rv/a1-summary-on.json
/home/laure/r17_work/r19/probe-rv/a1-on-221543.json  # 48 行 + 双臂统计 + 配对
/home/laure/r17_work/r19/tests-wide-base-full.log
/home/laure/r17_work/r19/tests-wide-review-full.log
```

主树复核：`python/`、`tests/`、`src/`、`train/*.py`、`train/runs/r10-staging/probe_r17_deployment.py`
与纯净副本逐字节相同（唯一差异是主树自带的 `python/diablogym.egg-info`，rsync 排除项）。
今晚对主树唯一的写入仍然只有 append-only 账本。

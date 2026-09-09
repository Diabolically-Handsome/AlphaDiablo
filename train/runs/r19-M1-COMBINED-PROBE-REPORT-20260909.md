# R19-M1 报告：三法合并（spend-v2 + sweep-v2 + gold-grab-v1）与合并探针

- 日期：2026-09-09（`by` = `r19-m1-integrator`）
- 合并树：`/home/laure/r17_work/r19/merge-tree`（Python only；`build -> /home/laure/r17_work/r17-1/build-res`，与主树同一条活桥）
- 补丁：`/home/laure/r17_work/r19/m1.patch`（`diff -ruN` 对主树；21 个文件，+5769 / −174；`src/` 与主树**逐字节相同**，C++/引擎一字未动）
- 探针原始产物：`/home/laure/r17_work/r19/m1-probe-final/`（最终树；`m1-probe/` 是同一份代码在文档字符串修订前的首轮，两轮 sha 见 §6.1）
- 主树 `/home/laure/AlphaDiablo/diablogym` 未被修改（唯一例外：账本 append）

> ⚠️ **本报告有两轮。§一～§十是首轮（2026-09-09 中午）。复核轮（§十一，2026-09-09 下午）修了 7 条发现，其中一条（半径漏进窗内）改变了行为，所以三臂重跑过。**凡首轮与复核轮数字不一致处，以 §十一 为准**；每一处都在原地打了标记，没有删掉任何一个首轮数字。
>
> **一句话（首轮）**：三法按 spend-v2 → sweep-v2 → gold-grab-v1 合并，8 处冲突全部手工判定，另加 1 道整合围栏；主席 2026-09-09 上午第 2 条裁定（把 sweep-v2 的开窗条件换成 gold-grab 的那组事实、怪物半径 3→2）已实现并被测试钉住。**回归 sha 逐位相同**；ALL-ON 臂 alive 20→26（net +6，UCB95 0.02536），L2 hazard 0.2220→0.1235，L3 4→6；**sweep-v2 的 L2 半边不再是死的**：认证前 0/268 窗开在 L2，现在 51 窗 / 80 个目标。
>
> **一句话（复核轮，权威）**：7 条发现全部处理（2 high / 4 medium / 1 low），6 条改了代码。**回归 sha 仍逐位相同**（`0e5a1acd…`）；修好「半径漏进窗内」之后 ALL-ON 臂 alive 20→**24**（net **+4**，UCB95 **0.06865**），L2 hazard 0.2220→**0.1573**，L3 4→**5**；L2 半边仍然活着但更薄（**15 窗 / 33 目标**，卡住它的现在是全局 48 目标上限）；主席第 4 条要的孤儿掉落**去掉金币污染后 L2 只有 3 件**（首轮报的 21 里 66% 是硬币）。

---

## 一、合并树是怎么造的

按宪法配方新建 `/home/laure/r17_work/r19/merge-tree`：`rsync -a --exclude __pycache__` 主树的 `python/ src/ tests/ patches/` 与 `train/*.py`；`train/runs/eval-assembled` 与 `train/runs/r10-staging`（后者带 `probe_r17_deployment.py`）；`build -> /home/laure/r17_work/r17-1/build-res` 软链。

**一处必须记录的补充**：探针 `source_identity()` 的 `_SOURCE_FILES` 还绑住 `train/runs/r10-staging/probe_r15_deployment.py`，配方没提它，缺了它探针在启动时 `FileNotFoundError`。已从主树原样复制（未修改）。

三棵源树是**只读输入**，改动集用 `diff -ru` 对主树逐棵取得（**树是真相，不是 .patch 文件**）：

| 树 | python/ | train/ | tests/ |
|---|---|---|---|
| `spend-tree`（full-v2） | env, options_env, resource_protocol, resource_sustain, resource_sustain_loot, resource_weapon_upgrade, worker_env | eval_contract, train_ppo, eval_assembled, migrate_loot_candidate, probe | +`test_r19_spend_v2.py` |
| `sweep-tree`（sweep-v2） | options_env, resource_protocol, resource_retreat, resource_sweep, worker_env | eval_contract, train_ppo, eval_assembled, migrate_loot_candidate | +`test_r19_sweep_v2.py`，改 3 个既有测试文件 |
| `gold-tree`（gold-grab-v1） | +`resource_gold_grab.py`, options_env, resource_protocol, worker_env | eval_contract, train_ppo, probe | +`test_r19_gold_grab.py` |

合并方式：对三棵树都碰的 8 个文件做**三方合并**（`git merge-file <merged> <main> <tree>`，基线始终是主树，顺序 spend → sweep → gold）；只有一棵树碰的文件直接取该树的版本。`resource_sweep.py`/`resource_retreat.py` 取 sweep 树，`env.py`/`resource_sustain*.py`/`resource_weapon_upgrade.py` 取 spend 树，`resource_gold_grab.py` 取 gold 树。三棵树的**每一个测试文件都保留**。

## 二、8 处冲突，逐处怎么判的

自动合并出 8 处冲突（spend 与另两法零冲突——它碰的是 `resource_sustain*` 一线；冲突全部在 sweep×gold 之间，两法都在改同一条权限链）。

| # | 文件 | 冲突 | 判定 |
|---|---|---|---|
| 1 | `resource_protocol.py` | 两法在同一行插入词汇表 | 两条都留：`RESOURCE_SWEEP_PROTOCOLS = ("off","sweep-v1","sweep-v2")` 与 `RESOURCE_GOLD_GRAB_PROTOCOLS = ("off","gold-grab-v1")` |
| 2 | `train/eval_contract.py` | `validate_r16_environment` 的 `expected` 表：sweep 把值改成元组，gold 加一行 | 并成一张元组表，四把法全在：`{"resource_sweep": ("sweep-v1","sweep-v2"), "resource_identify": ("cain-v1",), "resource_weapon_upgrade": ("smith-v1",), "resource_gold_grab": ("gold-grab-v1",)}`；`_require(item in expected)` |
| 3 | `options_env.py` portal 触发守卫 | 两法各自补 R18-M 要求的 `.active` 项 | **两项都留**（`not (sweep is not None and sweep.active)` **且** `not gold_grab_service.active`）；各自在本法关时可证恒真 |
| 4 | `options_env.py` retreat 触发守卫 | 同上 | 同上；两法各自的「这只是延后、不是取消」论证都保留在注释里 |
| 5 | `options_env.py` RESUPPLY 占有链头 | sweep 重写了「次序是否承载行为」那段注释，gold 在同处插入 `grab = ...` | 两段注释都留（sweep 的「次序现在**承载行为**」是更强的话，放前面），链体是主席的次序：`sweep → grab → portal → retreat → resource_service` |
| 6 | `worker_env.py` 罚没谓词 | sweep 的 `_sweep_close_is_death_equivalent` 与 gold 的 `_grab_close_is_death_equivalent` 被自动合并器缝成了一个方法 | 拆回**两个完整方法**，逐字取自各自源树（sweep 版问撤退/传送法的 `danger_reason`；gold 版先看抓取窗自己的收窗理由，再问 `trigger_reason`/`danger_reason`） |
| 7 | `worker_env.py` `_descend_escrow_settlement` 的罚没梯 | 两法各加一档 | **两档都留**：`sweep_trigger/sweep_complete → _sweep_close_...` **或** `grab_trigger/grab_complete → _grab_close_...`；两法各自的旗关时短路 |
| 8 | `options_env.py` | （5 的同一 hunk 的尾部） | 见 5 |

### 一道**新加的整合围栏**（两棵单法树都不需要，合并后才需要）

`options_env.action_masks` 的 sweep 开窗条件补了 `and not (grab.active or grab.pending_pickup)`。理由写在代码里：

- gold-grab 自己的触发已经拒绝在 `sweep.active` 时开窗，**反向没有守卫**——单法世界里靠的是深度不交，而 sweep-v2 在 L1 与 L2 都跑，和 `GRAB_FLOORS=(1,2)` 完全重叠；
- 抓金窗在**窗口边界**上仍可能 `.active`（一扇非 RESUPPLY 窗以 `cap`/`scene` 收掉，而抓金腿还占着身体），而 RESUPPLY 占有链**优先 sweep**——没有这一项，sweep 会开在一条再也不会被驱动的抓金腿之上；
- `pending_pickup` 那一半是同一把钱包结算锁（gold 树在 `maybe_start` 处已经取过）。
- `resource_gold_grab` 关时 `gold_grab_service is None`，该项**可证恒真**，所以 sweep-v1 / sweep-v2 单独跑逐位不变。

### 三处登记，三个词都在

- `eval_contract.py`：`_PYTHON_PROTOCOL_FILES` 有 `resource_gold_grab.py`；`R16_ENVIRONMENT_DEFAULTS` 有 `resource_gold_grab: "off"`；`resource_purchase_mode` 收 `full-v2`；`_RESOURCE_SERVICE_STAGES` 与两个 recipe 分支都有 `full-v2`（`"full"` 一字未动）。
- `train_ppo.py`：两份指纹清单都列 `python/diablogym/resource_gold_grab.py`；`--resource-sweep` 的 choices 从部署侧 `RESOURCE_SWEEP_PROTOCOLS` **导入**（A1 的教训：抄下来的词汇表就是下一次漏注册的地方）；采购模式六个值含 `full-v2`。
- `probe_r17_deployment.py`：`_R16_ENV_KEYS` 含 `resource_gold_grab`；每行遥测含 `gold_grab*`（gold）、`spend_v2`（spend）、`sweep`（含本轮新增键）。

## 三、主席第 2 条裁定：sweep-v2 的开窗条件换了

### 3.1 问题（认证过的事实）

sweep-v2 认证时的触发仍是 loot 经济的**首程事实**：`cleared`（本层清空）或 `farm_scene_steps >= 3600`。两者在 6000 微步的一局里**在主线 L2 都不可达**：48 种子里 **0 / 268 扇窗开在 L2**，33 个到过 L2 的局中位停留 503 拍（SWEEP2-REPORT.md §6）。法条是对的，触发够不着它。

### 3.2 改成了什么（`SweepLaw` 的新字段，按法条行分叉）

```
trigger_mode        : "window-boundary"（v1，冻结）/ "in-radius"（v2）
trigger_radius      : None（v1）/ SWEEP_V2_TRIGGER_RADIUS = 8（v2）
monster_abort_radius: SWEEP_MONSTER_ABORT_RADIUS = 3（v1）/ 
                      SWEEP_V2_MONSTER_ABORT_RADIUS = 2（v2）
require_trip_slot   : True（v1）/ False（v2）
```

v2 的开窗条件现在**逐条对齐 gold-grab-v1**：

1. 记忆里有一个**本局被照亮过**、可交互、未做过的可取对象，在 **R=8** 切比雪夫格内；
2. **可见**活怪不在 **2** 格内（玩家侧）——并且沿用冻结口径再问一次「一切活怪（不问可见）」，与 gold-grab 同一个做法：只会更窄，不会更宽，遥测把 `monster_near_pair` 与 `monster_near_pair_unseen` 分开记；
3. **可见**活怪不在目标堆 2 格内（`monster_near_target`）；
4. `hp > 0.5*max_hp`；5. 玩家 idle；6. 冷却/预算/目标上限/空窗计数/危险法照旧。

在**其它服务开窗的同一处**（`OptionsEnv.action_masks`）评估，位置没动。
`cleared` / `farm_cap` 两个事实**仍被读**，但只用来给这扇窗**命名**（主席要的触发直方图），不再当闸。

### 3.3 一处需要主席点头的**整合判断**：`require_trip_slot`

v1 的开窗还有一条「必须还剩一次进城行程」的闸，理由是「不掉没人来捡的战利品」。**在合并后的世界里这句话对金币不再成立**：gold-grab-v1 在 L1 与 L2 都能不进城就把金堆装进钱包。主席给的替换条件里也没有这一条。故 **v2 不再问这条闸**（v1 原样保留），并把否决计数 `no_trip_slot` 留在遥测里。非金币掉落在 L2 仍然没人收——那是主席第 4 条裁定明确接受的已知代价，本轮**计了数**（§5、§6.4）。

### 3.4 没有动的东西

sweep 的预算（v2 2400 微步 / 48 目标）、收据入账、`set_danger_law` 绑定与 `sweep_danger_*` 交还档、`book_on_close`、`floor_keys`、跳过的危险种类——**一律未改**。gold-grab-v1 的每层 12 堆上限（12+12）按主席第 3 条**保留**。**没有任何其它法条的阈值被改动**；每一部缺省关的法仍缺省关；sweep-v1 与采购模式 `"full"` 逐字节不变。

## 四、遥测（只增不改）

`SweepService.telemetry()` 新增（关时/ v1 下也都成立，是严格超集）：

- `trigger_mode` / `trigger_radius` / `require_trip_slot`：这扇法条行的身份；
- `windows_by_floor`、`objects_by_floor`（chests/barrels/sarcophagi/targets/items_dropped_seen 按层）；
- `trigger_reasons`、`trigger_reasons_by_floor`：**触发理由直方图**；
- `trigger_denials`：**否决理由直方图**（只在「法条确实有东西可拒」时计数，gold-grab 的口径）；
- `orphaned_drops`：**按层**的 `{dropped, still_on_floor, collected_or_gone}`——本次 sweep 掉在该层的 item 身份，与「最后一次能看见该层地板时仍躺着的身份」求交。~~`floor_items` 只装非金币（金堆走 `gold_items` 另一条通道），所以这正是主席第 4 条要的那个数。~~

> ❌ **这句话是假的，复核轮发现 2 已订正（§11.1）**。`floor_items` 由 `src/diablogym.cpp` 无类型过滤地遍历 `ActiveItems` 建成，**装金币**；`gold_items` 是它的**被照亮子集**，不是另一条通道。实测：ALL-ON 两局里 61% 的 `floor_items` 观测是硬币。现在 `orphaned_drops` 的形状是 **`{floor: {"items": {...}, "gold": {...}}}`**，主席第 4 条要的是 `items` 那一半；新增 `gold_dropped_seen` 单记金堆。权威数字见 §11.3。

## 五、测试

三棵树的新测试文件**全部保留**，并集通过：

```
tests/test_r19_gold_grab.py + tests/test_r19_sweep_v2.py + tests/test_r19_spend_v2.py
  + tests/test_resource_sweep.py   ->  301 passed, 166 subtests passed
```

合并让两个既有断言过时，两处都按裁定改了（不是删）：

1. `test_r19_gold_grab.py::test_the_archive_identity_registers_the_flag`——`eval_contract` 的期望表因 sweep-v2 变成元组表，断言改成 `'"resource_gold_grab": ("gold-grab-v1",)'`；要求本身没变。
2. `test_r19_sweep_v2.py::test_the_abort_thresholds_are_shared_and_unchanged`——怪物环从模块常量变成法条行字段，断言改成「常量仍是 3、v1 行仍读 3、v2 行是 2、其余阈值一个没动」。

新增 **3 个类 / 20 个测试**把新触发钉死（`tests/test_r19_sweep_v2.py`，R19-M1 段）：

- `SweepV2InRadiusTriggerTests`：**在 L2 开窗**（`cleared=False`、FARM 钟为 0、`loot_trip_slots_left=0` —— 旧触发要的每一个事实都为假）；旧事实为真时仍用它们命名窗；R=8 边界（8 开、9 不开）；没被照亮过的对象不是候选；**2 格内可见怪不开窗**、**3 格怪对 v2 不再是阻碍（对 v1 仍是）**、**目标 2 格内可见怪不开窗**、不可见怪仍然拦；窗内交还环用的是法条行的环；hp/idle/危险法三档仍在，且危险法排在新触发**之前**；L3/城/定场仍在范围外。
- `SweepV1TriggerIsUnchangedTests`：v1 仍要 `cleared`/`farm_cap`、仍要行程名额、仍在 3 格拒绝、没有半径闸，法条行逐字段冻结。
- `SweepV2OrphanedDropTests`：掉在 L2 且最后一帧仍躺着 → 计入；被捡走 → 不计；v1 的形状一致且永不出现 L2 行。

邻域与全量（合并树，`DIABLOGYM_ROOT=<merge tree>`）：

| 套件 | 结果 |
|---|---|
| 三个新文件 + `test_resource_sweep.py` | **301 passed / 166 subtests，0 失败** |
| loot/sustain/weapon/identify/portal/retreat/sweep 邻域（glob） | **942 passed, 17 skipped, 546 subtests，0 失败** |
| 点名邻域（含 `test_options_env.py`、`test_r18b6_training_wiring.py` 等 16 个文件） | 15 failed / 707 passed —— **15 个失败逐条出现在纯净副本基线里**（`/home/laure/r17_work/r19/pristine-fails.txt`） |
| 全量 `tests/`（3 个收集失败文件除外） | 81 failed / 2642 passed / 99 skipped / 1545 subtests。与纯净基线逐条比对：**新增失败 0 个** |

被排除/新增错误的说明（都不是代码回归）：

- `test_build_contracts.py` / `test_content_case_aux.py` / `test_worker_env.py` 在**纯净副本上同样收集失败**（工作树缺 build/ 与 runs/ 下的固定产物），spend 复审轮已记录同样的 3 个文件；
- `test_content_case_driver.py::BiteqTests` 的 3 个 ERROR 是`W_PIN["v32-ref-launch"]` 指向 `train/runs/v32…` 的固定产物，合并树按配方只带 `eval-assembled` 与 `r10-staging` 两个 runs 目录 —— **树形，不是代码**。

## 六、探针

pool **2_133**，种子 2133000–2133047，**3 个并行分片**（a 2133000-2133015 / b 2133016-2133031 / c 2133032-2133047），worker `7e31dc54` = 主树 `train/runs/r16-arm-a-constitution/model_candidate.zip`，`max_steps 6000`、`decoding sample`、探针 `r17-deployment-v3-r18m2`。配对 McNemar 与单侧 95% UCB 的公式逐字取自 `/home/laure/r17_work/r18/m2_probe_driver.py::paired`；`arm_stats` 的口径同源（本轮补了收入/支出/按层/时间占比几列）。**处女池 2_116-119、2_126-128 零接触。**

### 6.1 三臂身份（**回归通过**）

> ⚠️ **§6.1～§6.5 是首轮数字。复核轮修好「半径漏进窗内」（发现 3）后三臂重跑，ALL-ON 与 RICH 的 `rows_sha_v3` 都变了。权威数字见 §11.2～§11.4。OFF 臂的回归 sha 两轮相同。**

| 臂 | overrides | `rows_sha_v3` | 判定 |
|---|---|---|---|
| **OFF** | 认证的 16 键，**一字未改** | `0e5a1acd2fb2c07cd27eb72f9ecc574322e72f362ef0376b692376d9a99be886` | **与认证控制行逐位相同** |
| **ALL-ON** | 同上 + `resource_purchase_mode=full-v2`、`resource_sweep=sweep-v2`、`resource_gold_grab=gold-grab-v1` | `fc9c01d9c6ec1bbde56f5087b84739078822ffa2e8c2c52758ccd82ef1fabb2e` | 新行 |
| **ALL-ON-RICH** | ALL-ON + reset 后 +3000 金 | `8f2b05076a6f4ee27c8470f0f92e601b4d0df2b27e6f260e9e78e4cf439dd76c` | **诊断，不是法条** |

48 行 / 臂，**runtime errors 0 / 0 / 0**。OFF 对认证控制行配对：saved 0 / lost 0 / discordant 0 —— OFF 臂**就是**那一行，逐行相同。
三臂**跑了两轮**：12:26 的首轮（`m1-probe/`）与 13:03 在最终树上的重跑（`m1-probe-final/`；两次之间只改过 `resource_sweep.py` 的两段文档字符串）。**三个 `rows_sha_v3` 两轮逐位相同** —— 这就是「那两处编辑没有碰行为」的证据，而不是我的断言。本报告的每个数字来自 `m1-probe-final/`。

最终树在跑之前/之后都取了指纹，运行期间未变（`m1/final-tree-fingerprint-{before,after}.txt`）。

### 6.2 三法单独 vs 合并（同 48 种子、同 worker、同控制行）

单法数字是从三棵树**自己的最终产物**重算的（不是抄它们的报告）：
`gold-probe-rr/gold2-on-004806.json`、`sweep-probe-rr/rr-on-rows.json`、`spend-probe-rr/spend2-on-merged.json`。

| 指标 | control（认证行） | M1 OFF | gold only | sweep only | spend only | **M1 ALL-ON** | ALL-ON-RICH（诊断） |
|---|---|---|---|---|---|---|---|
| 存活 alive /48 | 20 | 20 | 21 | 23 | 17 | 26 | 28 |
| 到达 L2 | 34 | 34 | 30 | 33 | 35 | 35 | 39 |
| 到达 L3 | 4 | 4 | 1 | 1 | 4 | 6 | 8 |
| L2 死亡 | 19 | 19 | 20 | 17 | 22 | 15 | 14 |
| L2 beats | 85578 | 85578 | 79203 | 78834 | 78301 | 121496 | 145012 |
| **L2 hazard /1k** | 0.222 | 0.222 | 0.2525 | 0.2156 | 0.281 | 0.1235 | 0.0965 |
| L2 kills | 894 | 894 | 679 | 758 | 738 | 1146 | 1286 |
| clvl>=4 | 25 | 25 | 19 | 21 | 24 | 26 | 33 |
| clvl 中位 | 4.0 | 4.0 | 3.0 | 3.0 | 3.5 | 4.0 | 4.0 |
| gold_final 中位 | 78.0 | 78.0 | 132.5 | 130.0 | 28.5 | 80.0 | 2448.5 |

按层死亡：

| 臂 | control（认证行） | M1 OFF | gold only | sweep only | spend only | **M1 ALL-ON** | ALL-ON-RICH（诊断） |
|---|---|---|---|---|---|---|---|
| deaths_by_floor | {"2": 19, "1": 7, "3": 2} | {"2": 19, "1": 7, "3": 2} | {"2": 20, "1": 7} | {"2": 17, "1": 7, "3": 1} | {"2": 22, "1": 7, "3": 2} | {"2": 15, "1": 7} | {"2": 14, "1": 3, "3": 3} |

配对 McNemar（vs 认证控制行）：

| 臂 | saved | lost | net | discordant | UCB95(单侧) |
|---|---|---|---|---|---|
| M1 OFF | 0 | 0 | +0 | 0 | 0.00000 |
| gold-grab-v1 单独 | 11 | 10 | +1 | 21 | 0.13614 |
| sweep-v2 单独 | 9 | 6 | +3 | 15 | 0.06940 |
| spend-v2 单独 | 3 | 6 | -3 | 9 | 0.16424 |
| **M1 ALL-ON** | 13 | 7 | +6 | 20 | 0.02536 |
| ALL-ON-RICH（诊断） | 15 | 7 | +8 | 22 | -0.01087 |

### 6.3 首降（第一次 L1→L2）、收入与支出

| 指标 | control（认证行） | M1 OFF | gold only | sweep only | spend only | **M1 ALL-ON** | ALL-ON-RICH（诊断） |
|---|---|---|---|---|---|---|---|
| 首降局数 | 34 | 34 | 30 | 33 | 35 | 35 | 39 |
| 首降 beat 中位 | 7280.0 | 7280.0 | 6883.5 | 7921 | 7135 | 7270 | 7766 |
| 首降 belt 中位 | 4.0 | 4.0 | 5.0 | 5 | 8 | 8 | 8 |
| 首降 **AC** 中位 | 10.0 | 10.0 | 10.0 | 11 | 12 | 15 | 20 |
| 首降 damage 中位 | 9.0 | 9.0 | 9.0 | 9 | 7 | 9 | 9 |
| 首降 clvl 中位 | 3.0 | 3.0 | 3.0 | 3 | 3 | 3 | 3 |
| 进城行程数 | 105 | 105 | 103 | 109 | 103 | 103 | 114 |
| 收入·行程 collect | 12178 | 12178 | 7826 | 14023 | 12362 | 10793 | 12411 |
| 收入·卖货 | 7591 | 7591 | 7523 | 9064 | 7111 | 9324 | 11396 |
| 收入·**捡金 grab** | 0 | 0 | 7640 | 0 | 0 | 9606 | 10717 |
| 支出·合计 | 15442 | 15442 | 15084 | 16572 | 20848 | 26214 | 49496 |
| 支出·修理 | 767 | 767 | 689 | 757 | 783 | 554 | 1051 |
| 支出·鉴定 | 1600 | 1600 | 1400 | 2500 | 1900 | 2300 | 2700 |
| 支出·武器（升级法） | 6620 | 6620 | 6430 | 5760 | 710 | 1440 | 13360 |
| 支出·药水（full-v2 账） | 0 | 0 | 0 | 0 | 7900 | 9000 | 10850 |
| 支出·护甲（full-v2 账） | 0 | 0 | 0 | 0 | 3950 | 8145 | 15960 |
| 支出·武器（full-v2 账） | 0 | 0 | 0 | 0 | 710 | 1440 | 13360 |

> `sweep gold`（扫箱直接带来的金）**报为空，不报为 0**：箱子的金掉在**地板**上，本轮由 gold-grab 或行程 collect 入账，没有任何通道给一枚硬币打上「这是哪个箱子掉的」标签。把它算成任何一个数都会与 `gold_collected` / `gold_taken` 重复计数（SWEEP2-REPORT.md 同一结论）。

### 6.4 扫箱 / 捡金 按层，以及主席第 4 条要的孤儿掉落

| 指标 | control（认证行） | M1 OFF | gold only | sweep only | spend only | **M1 ALL-ON** | ALL-ON-RICH（诊断） |
|---|---|---|---|---|---|---|---|
| 扫箱 chests | 253 | 253 | 236 | 408 | 251 | 454 | 489 |
| 扫箱 barrels | 231 | 231 | 220 | 444 | 238 | 605 | 687 |
| 扫箱 sarcophagi | 0 | 0 | 0 | 396 | 0 | 454 | 486 |
| 扫箱窗数 | 104 | 104 | 113 | 268 | 108 | 1265 | 1368 |
| 扫箱微步 | 19356 | 19356 | 18309 | 49019 | 19561 | 27630 | 30243 |
| 捡金窗数 | 0 | 0 | 656 | 0 | 0 | 577 | 683 |
| 捡金堆数 | 0 | 0 | 669 | 0 | 0 | 768 | 838 |
| 捡金微步 | 0 | 0 | 12402 | 0 | 0 | 12767 | 14787 |
| 时间占比·扫箱 | 0.036 | 0.036 | 0.0336 | 0.0854 | 0.0367 | 0.0466 | 0.0461 |
| 时间占比·捡金 | 0.0 | 0.0 | 0.0227 | 0.0 | 0.0 | 0.0215 | 0.0226 |
| 时间占比·进城 | 0.2796 | 0.2796 | 0.2714 | 0.2937 | 0.2944 | 0.2705 | 0.2675 |

**ALL-ON · sweep by floor**：`{"1": {"windows": 1214, "chests": 435, "barrels": 584, "sarcophagi": 437, "targets": 1793, "microsteps": 26497, "items_dropped_seen": 616}, "2": {"windows": 51, "chests": 19, "barrels": 21, "sarcophagi": 17, "targets": 80, "microsteps": 1133, "items_dropped_seen": 21}}`

**ALL-ON · sweep 触发理由**：`{"object_in_radius": 1165, "farm_cap": 96, "cleared": 4}`

**ALL-ON · sweep 否决理由**：`{"cooldown": 14647, "monster_near_pair": 8019, "monster_near_pair_unseen": 1227, "monster_near_target": 314, "target_cap": 34003, "low_hp": 74, "empty_windows": 4302}`

**ALL-ON · sweep 孤儿掉落**：`{"1": {"dropped": 616, "still_on_floor": 175, "collected_or_gone": 441}, "2": {"dropped": 21, "still_on_floor": 5, "collected_or_gone": 16}}`

**ALL-ON · grab by floor**：`{"1": {"windows": 395, "piles_taken": 522, "gold_taken": 4869, "piles_seen_lit": 1529}, "2": {"windows": 182, "piles_taken": 246, "gold_taken": 4737, "piles_seen_lit": 126}}`

**ALL-ON-RICH · sweep by floor**：`{"1": {"windows": 1288, "chests": 462, "barrels": 644, "sarcophagi": 464, "targets": 1937, "microsteps": 28530, "items_dropped_seen": 651}, "2": {"windows": 80, "chests": 27, "barrels": 43, "sarcophagi": 22, "targets": 122, "microsteps": 1713, "items_dropped_seen": 39}}`

**ALL-ON-RICH · sweep 触发理由**：`{"object_in_radius": 1243, "cleared": 3, "farm_cap": 122}`

**ALL-ON-RICH · sweep 否决理由**：`{"cooldown": 16304, "monster_near_pair": 5849, "monster_near_pair_unseen": 1611, "monster_near_target": 441, "target_cap": 37697, "low_hp": 354, "empty_windows": 5895}`

**ALL-ON-RICH · sweep 孤儿掉落**：`{"1": {"dropped": 651, "still_on_floor": 152, "collected_or_gone": 499}, "2": {"dropped": 39, "still_on_floor": 18, "collected_or_gone": 21}}`

**ALL-ON-RICH · grab by floor**：`{"1": {"windows": 444, "piles_taken": 548, "gold_taken": 5185, "piles_seen_lit": 1674}, "2": {"windows": 239, "piles_taken": 290, "gold_taken": 5532, "piles_seen_lit": 154}}`

对照：control / OFF 臂的 sweep 是 sweep-v1，`by_floor` 只有 L1，`orphaned_drops` 为空（v1 不记这两项——它是本轮新增的遥测，v1 的 `by_floor` 由 R19 sweep 树引入且只可能有 L1 行）。

### 6.5 点名的两个种子

| 种子 | control/OFF | ALL-ON | ALL-ON-RICH |
|---|---|---|---|
| 2133002 | 死于 L1 @ 559 拍（clvl 1, AC 7, kills 21） | **存活**，depth 1（clvl 4, AC 14, kills 136, 12000 微步） | 死于 L1 @ 2385 拍（clvl 2, AC 8, kills 62） |
| 2133010 | 死于 L2 @ 12001 拍（clvl 4, AC 7, kills 148） | 死于 L2 @ 17491 拍（clvl 4, AC 9, kills 162） | 死于 L2 @ 8785 拍（clvl 4, AC 20, kills 154） |

- **2133002（ALL-ON）**：sweep 窗按层 `{"1": 37}`，触发 `{"object_in_radius": 33, "farm_cap": 4}`，对象按层 `{"1": {"chests": 13, "barrels": 11, "sarcophagi": 11, "targets": 48, "items_dropped_seen": 22}}`，孤儿掉落 `{"1": {"dropped": 22, "still_on_floor": 10, "collected_or_gone": 12}}`；捡金按层 `{"1": {"windows": 9, "piles_taken": 12, "gold_taken": 113, "piles_seen_lit": 48}, "2": {"windows": 0, "piles_taken": 0, "gold_taken": 0, "piles_seen_lit": 0}}`（共 113 金）。
- **2133010（ALL-ON）**：sweep 窗按层 `{"1": 32}`，触发 `{"object_in_radius": 28, "farm_cap": 4}`，对象按层 `{"1": {"chests": 13, "barrels": 15, "sarcophagi": 10, "targets": 48, "items_dropped_seen": 16}}`，孤儿掉落 `{"1": {"dropped": 16, "still_on_floor": 2, "collected_or_gone": 14}}`；捡金按层 `{"1": {"windows": 10, "piles_taken": 12, "gold_taken": 142, "piles_seen_lit": 49}, "2": {"windows": 8, "piles_taken": 9, "gold_taken": 188, "piles_seen_lit": 1}}`（共 330 金）。

## 七、读数：合并是**加性**还是**互相挤**？

> ⚠️ **首轮读数。复核轮的权威读数见 §11.5：结论同向（仍是超加性），但强度更弱——net +6 → +4，UCB95 0.02536 → 0.06865。**

三法单独的净存活是 +1（gold）、+3（sweep）、-3（spend），和为 +1；**合并臂是 +6**。

**读数：超加性，不是互相挤。** 机制是看得见的，而且三条链都对得上：

1. **sweep 造货 → grab 变现 → spend 花掉。** 单独跑时这条链是断的：sweep-v2 单独把箱子打开，金掉在地板上等一次进城行程；gold-grab 单独有钱可捡但没人多开箱子；spend-v2 单独**有货架没钱**（它单法是唯一一个 net 为负的臂，−3）。合并后：扫箱数 253→454，捡金收入 7640→9606，总支出 15442→26214，而**首降 AC 中位 10.0→15**。
2. **护甲变成了活命。** L2 hazard 0.2220→0.1235（−44%），而 L2 beats 85578→121496（+42.0%）——**不是靠少下楼换来的**，是待得更久还死得更少。
3. **深度没有被脚本吃掉。** 两条单法臂各自把 L3 从 4 打到 1（GOLD/SWEEP 报告都记过这个代价），合并臂 L3 = 6，**比控制行还高**。时间占比：扫箱 4.66%、捡金 2.15%、进城 27.05%。

**但请按证据的强度读**：ALL-ON 的单侧 95% UCB 是 **0.02536**——**大于 0**，也就是说这一臂**仍不能在 95% 水平上排除变坏**（saved 13 / lost 7，discordant 只有 20）。它比三条单法臂都好得多（0.136 / 0.069 / 0.164），但 48 个种子不够把它推过零。**这是一次很有希望的读数，不是一次认证。**

ALL-ON-RICH（**诊断**）：net +8、UCB95 -0.01087（**小于 0**）、L3 8、hazard 0.0965。它说明这条链**目前仍然被钱卡着**：白送 3000 金能把支出从 26214 抬到 49496、把首降 AC 中位抬到 20。它不是法条，也不是公平装备，只回答「花钱这条腿是不是饿着」。

## 八、主席第 2 条裁定的验收：L2 半边活过来了吗？

> ⚠️ **首轮验收。复核轮的权威验收见 §11.6：L2 半边仍然活着，但从 51 窗 / 80 目标降到 15 窗 / 33 目标——首轮那个更大的数字是发现 3 缺陷行为的副产品（8 格一跳的短窗开得多）。**

**活了。** 认证前：0 / 268 扇窗开在 L2。现在（ALL-ON，同 48 种子）：

| | L1 | L2 |
|---|---|---|
| 窗 | 1214 | **51** |
| 目标 | 1793 | **80** |
| chests / barrels / sarcophagi | 435 / 584 / 437 | **19 / 21 / 17** |
| 微步 | 26497 | 1133 |
| 掉落（本 sweep 造成） | 616 | 21 |
| **最后一帧仍躺在地上**（主席第 4 条） | 175 / 616 | **5 / 21** |

**但 L2 只拿到了 4.0% 的窗**，原因在否决直方图里读得出来：`target_cap` 是最大的一档。sweep-v2 的 48 目标/局上限是**全局**的，新触发让 L1 把它基本吃光，等到了 L2 已经没有目标名额。gold-grab 早就有 12+12 的**按层**上限（主席第 3 条保留的那条），sweep 没有。**我没有自己加**：那会是一个主席今晚没有授权的新阈值。这是给下一轮的第一条建议（§九.1）。

## 九、已知缺口（不掩盖）

> ⚠️ **复核轮把这 10 条逐条重核过一遍，结论表在 §11.7；其中第 1 条更严重、第 4 条更远，第 2 条的度量已校正。**

1. **sweep-v2 的 48 目标上限是全局的，L1 会吃光它。** 见 §八。建议下一轮把它拆成按层（例如 24+24，与 gold-grab 的 12+12 同形），**这是一个阈值裁定，需要主席点头**。
2. **`require_trip_slot=False` 是我做的整合判断**（§3.3），不是主席逐字裁定的。它在合并世界里有理由（金币不再需要行程），但 L2 的**非金币**掉落确实变多了（21 件掉落中 5 件最后一帧仍躺着）。若主席不同意，改回 True 是一行。
3. **`collected_or_gone` 不等于「被收走了」。** 它是「最后一次看得见该层地板时已不在 `floor_items` 里」，可能是冻结 worker 自己捡了、也可能是通道范围问题。L2 上没有 collect 阶段，所以那一列**不能读成收入**。
4. **ALL-ON 仍不能在 95% 水平排除变坏**（UCB95 0.02536 > 0）。48 个种子、discordant 20。要认证需要更多种子或更强的效应。
5. **训练 smoke 没做**（not verified）。今晚是 Python-only + 探针-only：`train_ppo` / `eval_assembled` / `migrate_loot_candidate` 三处登记做了、单测过了，但**没有任何一条训练臂在 `--resource-sweep sweep-v2 --resource-gold-grab gold-grab-v1 --resource-purchase-mode full-v2` 下被启动过**。
6. **`migrate_loot_candidate` 会拒绝 full-v2 源 run**（spend 复审轮刻意 fail closed）。下一轮要么给它注册 full-v2，要么明确说「full-v2 只评测不训练」。**继承的缺口，未解。**
7. **escrow 罚没围栏两条都只有单测**：探针臂 `descend_escrow_fraction = 0`，`_sweep_close_is_death_equivalent` 与 `_grab_close_is_death_equivalent` 在 48 局里**一次都没触发过**。它们第一次说话会是在训练臂上。sweep 侧那条「撤退与传送都关时无法可问、于是 vest」的开口（SWEEP2-REPORT.md 10.7）**原样继承**，本轮没有改。
8. **sweep-v1 的跨层记忆缺陷原样继承**（`floor_keys=False`，`observe()` 在所有深度都跑），修它会移动 4/48 条认证行，仍然是主席的裁定，不是本轮的修理。
9. **全量套件里 3 个文件收集失败 + 3 个 Biteq ERROR 都是树形**（§五），在合并树里没跑过；纯净副本上同样跑不了。
10. **spend-v2 单法的负号没有被解释掉**：它单独跑 net −3 / UCB95 0.164。合并臂变正**不是**证明它单独无害，只是证明它在有钱的世界里有用。

## 十、产物清单

| 用途 | 路径 |
|---|---|
| 合并树 | `/home/laure/r17_work/r19/merge-tree` |
| 合并补丁（对主树 `diff -ruN`） | `/home/laure/r17_work/r19/m1.patch` |
| 三方合并 / 冲突判定 / 围栏脚本 | `/home/laure/r17_work/r19/m1/{merge.sh,resolve.py,fence.py}` |
| sweep-v2 触发改动脚本 | `/home/laure/r17_work/r19/m1/{sweepv2.py,sweepv2b.py,doc2.py,doc3.py}` |
| 测试改动脚本 | `/home/laure/r17_work/r19/m1/{tests1.py,tests2.py,tests3.py}` |
| 探针驱动 | `/home/laure/r17_work/r19/m1/m1_probe_driver.py`（最终树版 `…_final.py`） |
| rich 臂驱动 | `/home/laure/r17_work/r19/m1/m1_rich_probe.py` |
| 三臂原始行（首轮） | `/home/laure/r17_work/r19/m1-probe-final/m1-{off,on,rich}-rows.json` |
| 三臂汇总（首轮，**有错，见发现 5**） | `/home/laure/r17_work/r19/m1-probe-final/m1-summary-all.json` |
| 三臂原始行（**复核轮，权威**） | `/home/laure/r17_work/r19/m1-probe-rr/m1-{off,on,rich}-rows.json` |
| 三臂汇总（**复核轮，权威**） | `/home/laure/r17_work/r19/m1-probe-rr/m1-summary-all.json` |
| 首轮（文档字符串修订前）产物 | `/home/laure/r17_work/r19/m1-probe/` |
| 重算聚合（本报告每个数字的来源） | `/home/laure/r17_work/r19/m1/aggregates.json` + `agg.py` |
| 测试日志 | `/home/laure/r17_work/r19/m1/{t1.log,t-nbhd.log,t-nbhd2.log,t-full.log,t-full-fails.txt}` |
| 树指纹（跑前/跑后） | `/home/laure/r17_work/r19/m1/final-tree-fingerprint-{before,after}.txt` |

凡本报告未由上述文件直接支撑的说法，一律标注 not verified。

---

## 十一、复核轮（2026-09-09 下午，`by` = `r19-m1-fixer`）

复核给了 **7 条**：2 条 high、4 条 medium、1 条 low。**7 条全部处理**，其中 6 条改了代码，第 6 条按复核自己给的两条出路之一处理（改度量、不改法条，把校正后的数字交给主席）。**本节的数字是权威数字**；§六～§八的首轮数字已被本节取代，逐处都打了标记。

产物在 `/home/laure/r17_work/r19/m1rr/`（脚本、日志、证据）与 `/home/laure/r17_work/r19/m1-probe-rr/`（三臂原始行与汇总）。

### 11.1 逐条

#### 发现 1（HIGH，已修）：`OptionsEnv.action_masks` 的 `sweep` 是**未绑定局部变量**——这是合并对主树的回归

`sweep` 是 `action_masks()` 的函数局部量。**唯一的绑定**原先坐在 `if resource_service_policy == "sustain-loot-v1":` 分支里（还有一次多余的重绑定坐在**三个读取之后**），而**三个守卫在缩进 12 处读它**：portal 触发守卫、retreat 触发守卫、grab 触发守卫。只要服务策略不是 `sustain-loot-v1`——默认值 `legacy-v1` 就不是，而 `eval_contract.py` 与 `train_ppo.py` **都允许 `retreat-v1` 与它并存**——第一拍走到 retreat 守卫就 `UnboundLocalError`。主树**零处**这样的读取：sweep 树加了两处，gold 树加了一处，三处都活到了合并树里。今晚每条认证臂都用 `sustain-loot-v1`，所以三臂全都没碰到它。

**两个方向都实测过**（`m1rr/f1check.log`，配置 = `l2-town-v1` + `coach-v03` + `retreat-v1` + `sustain-v6`，真引擎）：

| 树 | 结果 |
|---|---|
| 合并树（**修前**，`m1rr/prefix-tree`） | `action_masks()` **第 1 次调用**即死于 `UnboundLocalError`（`options_env.py:1226`） |
| 合并树（**修后**） | 6 个种子共 **53 次** `action_masks()` 调用，无异常 |

**修法**：把 `sweep = getattr(self, "sweep_service", None)` 提到 `service = self.resource_service` 之后、缩进 12 处，绑定**一次**；删掉后面那次多余重绑定。在 `sustain-loot-v1` 下取值与旧代码逐位相同（回归 sha 印证了这一点）。

**钉子**：新增 `tests/test_r19_sweep_v2.py::ActionMasksLocalBindingTests`——一条不需要引擎的**支配性检查**：`action_masks()` 里每一个被守卫读到的局部量（`sweep`/`grab`/`portal`/`retreat`/`service`），都必须存在一次**先于它、且不在它之外的分支里**的绑定。这条测试在**修前的树上会失败**（`m1rr/ast_negcheck.sh`，报 `sweep` at line 1174），在修后通过——不是我的断言，是跑出来的。

#### 发现 2（HIGH，已修）：`floor_items` **装金币**，主席第 4 条要的数被污染了

首轮的 `_observe_floor_items` 断言「`floor_items` 只装 ITEM，金币走 `gold_items` 另一条通道」，§四原样重复了这句话。**这句话是假的**：

* `src/diablogym.cpp:1504-1526` 用 `for (i < ActiveItemCount)` 建 `obs["floor_items"]`，**没有任何类型过滤**；
* `src/resource_protocol.hpp:956-965` 从**同一个数组**建 `resource_state["gold_items"]`，条件是 `item._itype != ItemType::Gold || !IsTileLit(...)` 就跳过。

也就是说 `gold_items` 是 `floor_items` 的**被照亮的子集**，不是另一条通道。

**实测**（`m1rr/goldchan.py` → `m1rr/goldchan.json`，ALL-ON 两局、3759 个经理拍、126774 条 `floor_items` 观测）：

| 事实 | 数 |
|---|---|
| `gold_items` 观测总数 | 4165 |
| 其中**也在** `floor_items` 里 | **4165**（缺失 0） |
| 这 4165 条在 `floor_items` 里的 `item_type` | **全部是 11** |
| `floor_items` 里 `visible` 且 `item_type==11` 的观测 | **4165**（与上一行严格相等） |
| 其中**不在** `gold_items` 里的 | **0** |
| `floor_items` 里 `visible` 且 `item_type!=11` 却在 `gold_items` 里的 | **0** |
| `floor_items` 里 `item_type==11` 的观测（含未照亮） | **77512 / 126774 = 61%** |

两个方向都对上：`item_type == 11` **就是** `ItemType::Gold`（引擎源码不在盘上，所以这是**测出来的**，不是抄来的）。而 `gold_items` 只看得见被照亮的那 4165 条，`item_type` 连没照亮的金堆也认得——所以新代码用 `item_type`，不用 `gold_items` 求差。

**修法**：`resource_sweep.ITEM_TYPE_GOLD = 11` + `_is_gold_item()`；`_count_drops` 在计数处分流，`items_dropped_seen` 只数非金币，新增 `gold_dropped_seen`；`_observe_floor_items` 记两套「最后一帧」身份；`orphaned_drops()` 返回 **`{floor: {"items": {...}, "gold": {...}}}`**。主席第 4 条要的是 **`items` 那一半**。

**污染有多大**（同一批 OFF 臂行，新旧口径可直接对账）：

| OFF 臂 L1 | 首轮报的一个数 | 复核轮拆开 |
|---|---|---|
| dropped | 243 | **items 83** + gold 160 |
| still_on_floor | 93 | **items 48** + gold 45 |
| collected_or_gone | 150 | **items 35** + gold 115 |

**66% 的「掉落」是硬币**。首轮 §6.4 给主席的 L2 那行 `{dropped 21, still 5, gone 16}` 同样是混的，而且 `collected_or_gone` 被 gold-grab 的入袋压低了——那正是主席**排除在第 4 条之外**的通道。

`collected_or_gone` 的读法也在 docstring 里改窄了：它只表示「这个身份不在我们最后一次看见该层地板的那一帧里」，冻结 worker 自己捡走、gold-grab 入袋、引擎 despawn 全都落在这一档，**它不是收入**。

#### 发现 3（MEDIUM，已修）：半径**漏进了窗内**，把一扇窗变成了 8 格一跳的链——**这是本轮唯一移动了存活数的修理**

主席第 2 条改的是**触发**。首轮把 `SweepLaw.trigger_radius` 读在 `_reachable_targets` **内部**，而这个 helper 有**两个调用方**：经理侧触发，以及**窗内选目标**。于是一扇已经开着的 sweep-v2 窗只能选当前站位 8 格内的目标，最近的记忆对象一到 9 格就以 `sweep_no_reachable_target` 收窗，还要付 60 微步冷却，并且喂 `SWEEP_EMPTY_WINDOWS`（3 次连续空窗 = 本局退役）。认证过的 sweep-v2 是**一扇窗扫全层**。

签名是对得上的：

| | 窗数 | 微步 | 微步/窗 | `empty_windows` 否决 |
|---|---|---|---|---|
| 认证 sweep-v2 单法 | 268 | 49019 | **182.9** | — |
| 合并 ALL-ON（**首轮**） | 1265 | 27630 | **21.8** | **4302** |
| 合并 ALL-ON（**复核轮**） | 1027 | 40037 | **39.0** | **0（该档消失）** |

**修法**：`_reachable_targets(self, raw, env=None, *, radius=None)`；`radius` **只由 `trigger_reason` 传**（`radius=self.law.trigger_radius`），窗内调用 `self._reachable_targets(raw, env)` 不带半径，恢复认证行为。窗内够不够窄是**第二条法条改动**，不在主席第 2 条里，需要单独裁定。三条测试钉住（窗内越界仍可达 / 触发的 8 格边界不变 / v1 窗内无界），外加一条结构测试禁止窗内调用再带 `radius=`。

#### 发现 4（MEDIUM，已修）：sweep 会把传送法**饿一整扇窗**——gold-grab 在自己的复核轮修过同一个洞，合并把 sweep 那一半漏了

合并后的 portal 守卫在 `sweep.active` 时拒绝起传送腿，而 `_win_term` 的 sweep 档**排在 portal 档之前**，经理窗也不会收。唯一的上界是 sweep 自己的中止梯，而它的危险档问的是 `retreat.danger_reason`（`low_hp` / `empty_belt` / `pressure` 三条）。`PortalService.trigger_reason` 还会因 **`portal_standing`**（本层已有一扇门）点火，而 `resource_portal.py` 的 R18-B5 注释说得很清楚：**那不是危险条款**。于是一条中途变得合法的传送腿最多被推迟一整扇窗（`SWEEP_V2_MICROSTEP_BUDGET = 2400` 微步）。`options_env.py` 里那段**背诵 gold-grab 修理过程**的注释，就压在一个同样推迟 sweep 的守卫上面。

**修法**（逐条对齐 gold-grab）：`SweepService.bind_portal()` + `_portal_wanted()`；`_command` 的中止梯在**危险档之下**加一档 `sweep_portal_wanted`；`OptionsEnv` 在 `set_danger_law` 旁边绑；`worker_env._sweep_close_is_death_equivalent` 加上 sweep 自己的危险收窗理由档（与 `_grab_close_is_death_equivalent` 处理 `grab_portal_wanted` 同形），免得这条收窗把托管 **vest** 掉、重新打开 R18-B/B5 刚关上的套现通道；`_unswept_reason` 把它归到 `danger`；`telemetry()` 新增 `portal_law_bound`（False 就是「这一档永远不会说话」的原因）。

**今晚三臂里这一条一次都没说话**：16 键控制行里**没有** `resource_portal`，所以 `portal_service is None`，该档可证恒真地惰性。它是给**下一次把 portal-v1 与 sweep-v2 并排点亮**的那一轮准备的。五条测试钉住（含「不绑就会拿住身体」的反向钉子，以及「危险法仍排在传送档之前」）。

#### 发现 5（MEDIUM，已修）：三臂汇总把 gold-grab 和进城**报成了 0% 时间**

`m1_probe_driver_final.py` 累加了 `microsteps_used`（grab 遥测里没有这个键，真名是 `grab_microsteps`）和 `service_microsteps`（**任何遥测都没有这个键**，进城开销是 `loot_cumulative.steps`），两处都被 `or 0` 吞掉，于是 `m1-summary-all.json` 说 gold-grab 花 0%、进城花 0%——同一份文件里却记着 577 扇抓金窗、768 个金堆。**报告是对的，产物是错的**，而 §十的产物表把审计员指向了错的那个文件。

**修法**：新驱动 `m1/m1_probe_driver_rr.py`，两处读对键，并加一个 `must()` 取代 `or 0`——**缺键就抛，不再默默报零**。逐指标与 `agg.py` 对账时又抓出**第三处同类错误**：`trip_count`（真名 `loot_trip_count`），首轮把每条臂的进城次数都报成了 0。

新增 `m1rr/resummarize.py`：从**已经落盘的行**重建汇总（不重跑探针），并把驱动汇总与 `agg.py` 的独立聚合**逐指标对账**，任何一条不一致就非零退出。现在的结果：**23 个共享指标 × 3 臂全部一致**。

#### 发现 6（MEDIUM，**法条未改，度量改了**）：`require_trip_slot=False`

复核给了两条出路，我走第二条：**保留 `require_trip_slot=False`，但把发现 2 的金币污染修掉，用校正后的数字请主席裁定**。理由：主席第 2 条替换的是**触发事实清单**，清单里没有行程名额；第 4 条又明说接受 L2 非金币掉落留在地上、只要计数。但复核指出的问题是真的——首轮拿来支撑这个判断的 `orphaned_drops` 恰好是**分不开金币与物品**的那个数。现在分得开了。

**请主席看的是这两个数（ALL-ON，48 种子，只算 items）**：

| | L1 | **L2** |
|---|---|---|
| 本次 sweep 掉的**非金币** | 203 | **3** |
| 最后一帧**仍躺在地上** | 115 | **3** |
| 不在最后一帧里（**不等于被收走**） | 88 | 0 |
| （同臂的金币账，仅供参照） | 431 / 78 / 353 | 5 / 1 / 4 |

**L2 上今晚一共只有 3 件非金币掉落无人收**（首轮报的是混了金币的 21）。`no_trip_slot` 这一档在 OFF 臂（sweep-v1）被记了 **7388** 次，在 ALL-ON 臂**一次都没有**——闸门确实在咬，代价确实很小。

**若主席不同意，改回 `True` 是一行**（`SWEEP_LAWS["sweep-v2"]` 那一行；v1 两种情况都不动）。代码注释里已经写明这是**整合判断而非裁定**。

#### 发现 7（LOW，已修）：§五的测试表来自旧树

**全部在最终树上重跑**，日志在 `m1rr/`：

| 套件 | 结果 | 日志 |
|---|---|---|
| `test_r19_{sweep_v2,gold_grab,spend_v2}.py` + `test_resource_sweep.py` | **314 passed / 166 subtests，0 失败** | `m1rr/t-new.log` |
| loot/sustain/weapon/identify/portal/retreat/sweep/gold 邻域（glob） | **1075 passed, 17 skipped, 583 subtests，0 失败** | `m1rr/t-nbhd.log` |
| 全量 `tests/`（3 个树形收集失败文件除外） | 81 failed / **2655 passed** / 99 skipped / 1545 subtests | `m1rr/t-full.log` |
| 全量 vs 纯净副本基线（`m1/base.txt`） | **新增失败 0 个**（多出的 3 行是 `test_content_case_driver.py::BiteqTests` 的 ERROR，§五已记为树形；另有 1 条基线失败我们反而通过了） | `m1rr/t-full-new.txt` |

首轮全量是 2642 passed，本轮 2655——差 13 就是本轮新增的 13 条测试。

### 11.2 回归：**逐位相同**

| 臂 | `rows_sha_v3` | 判定 |
|---|---|---|
| **OFF**（16 键认证控制行，一字未改） | `0e5a1acd2fb2c07cd27eb72f9ecc574322e72f362ef0376b692376d9a99be886` | **与认证控制行逐位相同** ✅ |
| ALL-ON | `579816824729d33b4199fb93e500fad05dc3ca70523b871825e4b2aaa9ff5dfb` | 新行（首轮是 `fc9c01d9…`——发现 3 的修理移动了它） |
| ALL-ON-RICH（诊断） | `8a63af8e9b48f4754eccd66a58002ee8f0d7f26b3a88173a4a2a335f45dd1e33` | 诊断，不是法条 |

48 行/臂，**runtime errors 0 / 0 / 0**。OFF 对认证控制行配对：saved 0 / lost 0 / discordant 0。树在跑前跑后取了指纹，运行期间未变（`m1rr/tree-fingerprint-{before,after}.txt`）。

探针跑完之后**只动过一处**：`resource_sweep.py` 里 `require_trip_slot` 那段**纯注释**（发现 6 的措辞）。没有再花一轮探针去证明它无害——`m1rr/bytecheck.py` 把改前改后两份源码都编译掉，逐个 code object 比较**指令流、常量与名字表**（忽略注释必然移动的行号）：**5621 项逐项相同**（`m1rr/bytecheck.log`）。

### 11.3 三臂（**权威数字**，取代 §6.2/6.3/6.4）

单法臂仍取三棵源树自己的最终产物，未重跑。

| 指标 | control / M1 OFF | gold only | sweep only | spend only | **M1 ALL-ON** | ALL-ON-RICH（诊断） |
|---|---|---|---|---|---|---|
| 存活 alive /48 | 20 | 21 | 23 | 17 | **24** | 29 |
| 到达 L2 | 34 | 30 | 33 | 35 | 33 | 31 |
| 到达 L3 | 4 | 1 | 1 | 4 | **5** | 5 |
| L2 死亡 | 19 | 20 | 17 | 22 | **14** | 15 |
| L2 beats | 85578 | 79203 | 78834 | 78301 | **89015** | 126698 |
| **L2 hazard /1k** | 0.2220 | 0.2525 | 0.2156 | 0.2810 | **0.1573** | 0.1184 |
| L2 kills | 894 | 679 | 758 | 738 | 835 | 1136 |
| clvl>=4 | 25 | 19 | 21 | 24 | 24 | 27 |
| clvl 中位 | 4.0 | 3.0 | 3.0 | 3.5 | 3.5 | 4.0 |
| gold_final 中位 | 78.0 | 132.5 | 130.0 | 28.5 | 89.0 | 2513.0 |
| deaths_by_floor | {2:19, 1:7, 3:2} | {2:20, 1:7} | {2:17, 1:7, 3:1} | {2:22, 1:7, 3:2} | **{2:14, 1:8, 3:2}** | {2:15, 1:4} |

配对 McNemar（对认证控制行）：

| 臂 | saved | lost | net | discordant | UCB95(单侧) |
|---|---|---|---|---|---|
| M1 OFF | 0 | 0 | +0 | 0 | 0.00000 |
| gold-grab-v1 单独 | 11 | 10 | +1 | 21 | 0.13614 |
| sweep-v2 单独 | 9 | 6 | +3 | 15 | 0.06940 |
| spend-v2 单独 | 3 | 6 | −3 | 9 | 0.16424 |
| **M1 ALL-ON** | 12 | 8 | **+4** | 20 | **0.06865** |
| ALL-ON-RICH（诊断） | 14 | 5 | +9 | 19 | −0.04490 |

首降 / 收入 / 支出：

| 指标 | control / OFF | gold only | sweep only | spend only | **ALL-ON** | RICH（诊断） |
|---|---|---|---|---|---|---|
| 首降局数 | 34 | 30 | 33 | 35 | 33 | 31 |
| 首降 beat 中位 | 7280.0 | 6883.5 | 7921 | 7135 | 7486 | 7600 |
| 首降 belt 中位 | 4.0 | 5.0 | 5 | 8 | 8 | 8 |
| 首降 **AC** 中位 | 10.0 | 10.0 | 11 | 12 | **15** | 19 |
| 首降 damage 中位 | 9.0 | 9.0 | 9 | 7 | 9 | 9 |
| 进城行程数 | 105 | 103 | 109 | 103 | 105 | 98 |
| 收入·行程 collect | 12178 | 7826 | 14023 | 12362 | 10841 | 10499 |
| 收入·卖货 | 7591 | 7523 | 9064 | 7111 | 8786 | 9511 |
| 收入·**捡金 grab** | 0 | 7640 | 0 | 0 | **8229** | 9215 |
| 支出·合计 | 15442 | 15084 | 16572 | 20848 | **25486** | 44396 |
| 支出·修理 | 767 | 689 | 757 | 783 | 1111 | 1236 |
| 支出·鉴定 | 1600 | 1400 | 2500 | 1900 | 1900 | 2100 |
| 支出·武器（升级法） | 6620 | 6430 | 5760 | 710 | 1780 | 11300 |
| 支出·药水（full-v2 账） | 0 | 0 | 0 | 7900 | **8900** | 10400 |
| 支出·护甲（full-v2 账） | 0 | 0 | 0 | 3950 | **7770** | 14675 |
| 支出·武器（full-v2 账） | 0 | 0 | 0 | 710 | 1780 | 11300 |

> `sweep gold` 仍**报为空、不报为 0**（原因同 §6.3：没有通道给一枚硬币打上「哪个箱子掉的」标签；本轮新增的 `gold_dropped_seen` 数的是**堆数**，不是金额，不能当收入读）。

扫箱 / 捡金 / 时间占比：

| 指标 | control / OFF | gold only | sweep only | spend only | **ALL-ON** | RICH（诊断） |
|---|---|---|---|---|---|---|
| 扫箱 chests | 253 | 236 | 408 | 251 | **467** | 475 |
| 扫箱 barrels | 231 | 220 | 444 | 238 | **615** | 649 |
| 扫箱 sarcophagi | 0 | 0 | 396 | 0 | **442** | 456 |
| 扫箱窗数 | 104 | 113 | 268 | 108 | 1027 | 1062 |
| 扫箱微步 | 19356 | 18309 | 49019 | 19561 | 40037 | 44928 |
| 捡金窗数 | 0 | 656 | 0 | 0 | 569 | 601 |
| 捡金堆数 | 0 | 669 | 0 | 0 | 703 | 764 |
| 捡金微步 | 0 | 12402 | 0 | 0 | 12279 | 12752 |
| 时间占比·扫箱 | 0.0360 | 0.0336 | 0.0854 | 0.0367 | **0.0693** | 0.0698 |
| 时间占比·捡金 | 0.0 | 0.0227 | 0.0 | 0.0 | **0.0212** | 0.0198 |
| 时间占比·进城 | 0.2796 | 0.2714 | 0.2937 | 0.2944 | **0.2785** | 0.2373 |

**ALL-ON · sweep by floor**：`{"1": {"windows": 1012, "chests": 458, "barrels": 607, "sarcophagi": 436, "targets": 1942, "microsteps": 39575, "items_dropped_seen": 203, "gold_dropped_seen": 431}, "2": {"windows": 15, "chests": 9, "barrels": 8, "sarcophagi": 6, "targets": 33, "microsteps": 462, "items_dropped_seen": 3, "gold_dropped_seen": 5}}`

**ALL-ON · sweep 触发理由**：`{"object_in_radius": 983, "farm_cap": 42, "cleared": 2}`

**ALL-ON · sweep 否决理由**：`{"target_cap": 38143, "cooldown": 12255, "monster_near_pair": 4825, "monster_near_pair_unseen": 1173, "monster_near_target": 346, "low_hp": 110}`

**ALL-ON · sweep 孤儿掉落（拆开）**：`{"1": {"items": {"dropped": 203, "still_on_floor": 115, "collected_or_gone": 88}, "gold": {"dropped": 431, "still_on_floor": 78, "collected_or_gone": 353}}, "2": {"items": {"dropped": 3, "still_on_floor": 3, "collected_or_gone": 0}, "gold": {"dropped": 5, "still_on_floor": 1, "collected_or_gone": 4}}}`

**ALL-ON · grab by floor**：`{"1": {"windows": 405, "piles_taken": 542, "gold_taken": 5165, "piles_seen_lit": 1495}, "2": {"windows": 164, "piles_taken": 161, "gold_taken": 3064, "piles_seen_lit": 115}}`

**OFF（sweep-v1）· sweep 孤儿掉落（拆开）**：`{"1": {"items": {"dropped": 83, "still_on_floor": 48, "collected_or_gone": 35}, "gold": {"dropped": 160, "still_on_floor": 45, "collected_or_gone": 115}}}` —— 与首轮那一个混合数 `{243, 93, 150}` 逐项对得上。

**ALL-ON-RICH · sweep by floor**：`{"1": {"windows": 1042, "chests": 465, "barrels": 638, "sarcophagi": 448, "targets": 2020, "microsteps": 44344, "items_dropped_seen": 201, "gold_dropped_seen": 447}, "2": {"windows": 20, "chests": 10, "barrels": 11, "sarcophagi": 8, "targets": 39, "microsteps": 584, "items_dropped_seen": 5, "gold_dropped_seen": 6}}`；触发 `{"object_in_radius": 1016, "farm_cap": 46}`；孤儿 `{"1": {"items": {"dropped": 201, "still_on_floor": 113, "collected_or_gone": 88}, "gold": {"dropped": 447, "still_on_floor": 102, "collected_or_gone": 345}}, "2": {"items": {"dropped": 5, "still_on_floor": 3, "collected_or_gone": 2}, "gold": {"dropped": 6, "still_on_floor": 0, "collected_or_gone": 6}}}`；grab `{"1": {"windows": 431, "piles_taken": 554, "gold_taken": 5296, "piles_seen_lit": 1582}, "2": {"windows": 170, "piles_taken": 210, "gold_taken": 3919, "piles_seen_lit": 151}}`

### 11.4 点名的两个种子（取代 §6.5）

| 种子 | control/OFF | ALL-ON | ALL-ON-RICH |
|---|---|---|---|
| 2133002 | 死于 L1 @ 559 拍（clvl 1, AC 7, kills 21） | **存活**，depth 1（clvl 3, AC 9, kills 132, 12000 微步） | **存活**，depth 2（clvl 4, AC 23, kills 172, 17408 微步） |
| 2133010 | 死于 L2 @ 12001 拍（clvl 4, AC 7, kills 148） | 死于 L2 @ 12321 拍（clvl 4, **AC 20**, kills 156, belt 8） | **存活**，depth 1（clvl 3, AC 20, kills 124） |

- **2133002（ALL-ON）**：sweep 窗 `{"1": 22}`，触发全是 `object_in_radius`，对象 `{"1": {"chests": 11, "barrels": 16, "sarcophagi": 10, "targets": 48}}`，孤儿 items `{dropped 2, still 2, gone 0}` / gold `{19, 10, 9}`；捡金 `{"1": {"windows": 9, "piles_taken": 12, "gold_taken": 118}}`（共 118 金）。
- **2133010（ALL-ON）**：sweep 窗 `{"1": 24}`，孤儿 items `{3, 1, 2}` / gold `{14, 0, 14}`；捡金 `{"1": {"windows": 10, "piles": 12, "gold": 142}, "2": {"windows": 5, "piles": 5, "gold": 98}}`（共 240 金）。首降 AC 从 12 抬到 20，仍然死在 L2。

### 11.5 加性读数（取代 §七）

三法单独的净存活 +1（gold）、+3（sweep）、−3（spend），**和为 +1**；**合并臂是 +4**。

**仍然是超加性，但比首轮弱**（首轮 +6，那个数字建立在发现 3 的缺陷行为之上）。机制没变，而且三条链依旧对得上：

1. **sweep 造货 → grab 变现 → spend 花掉。** 扫箱 253 → 467、桶 231 → 615、石棺 0 → 442；捡金收入 0 → 8229；总支出 15442 → 25486（其中 full-v2 的护甲账 0 → 7770、药水账 0 → 8900）；**首降 AC 中位 10.0 → 15**。
2. **护甲变成了活命。** L2 hazard 0.2220 → **0.1573（−29%）**，而 L2 beats 85578 → 89015（**+4.0%**）——不是靠少下楼换来的。
3. **深度没有被脚本吃掉。** 两条单法臂各自把 L3 从 4 打到 1，合并臂 **L3 = 5**，比控制行高。时间占比：扫箱 6.93%、捡金 2.12%、进城 27.85%。

**证据强度（比首轮更弱，必须照实说）**：ALL-ON 的单侧 95% UCB 是 **0.06865**——**远大于 0**，saved 12 / lost 8 / discordant 20。首轮报的 0.02536 已经不能排除变坏，本轮修好窗内可达之后**更不能**。**这是一次有希望的读数，不是一次认证，而且比首轮更不接近认证。**

ALL-ON-RICH（**诊断**）：net +9、UCB95 **−0.0449（小于 0）**、L3 5、hazard 0.1184、支出 44396。结论与首轮同向且更强：**这条链目前仍然被钱卡着**。

### 11.6 主席第 2 条的验收（取代 §八）：L2 半边**活着，但更薄**

认证前 **0 / 268** 扇窗开在 L2。修好发现 3 之后（ALL-ON，同 48 种子）：

| | L1 | L2 |
|---|---|---|
| 窗 | 1012 | **15** |
| 目标 | 1942 | **33** |
| chests / barrels / sarcophagi | 458 / 607 / 436 | **9 / 8 / 6** |
| 微步 | 39575 | 462 |
| 非金币掉落（本 sweep 造成） | 203 | **3** |
| 最后一帧仍躺在地上（主席第 4 条） | 115 / 203 | **3 / 3** |

首轮报的是 51 窗 / 80 目标——那个数字**是发现 3 的副产品**：8 格一跳的短窗开得多、每扇短，L1 更早撞上 48 目标上限之外还留下了名额。窗内可达恢复认证行为后，L1 的一扇窗就能吃掉更多目标，**全局 48 目标上限被 L1 吃得更干净**：`target_cap` 否决 38143 次，是最大的一档，L2 只剩 **1.4%** 的窗（首轮 4.0%）。

**结论没变，只是更尖锐**：主席第 2 条要的触发**做到了**（`object_in_radius` 983 次点火，L2 上从 0 变成 15 扇窗），但**真正卡住 L2 的现在是 48 目标/局这个全局上限**。把它拆成按层（例如 24+24，与 gold-grab 的 12+12 同形）是**下一轮的第一条建议**，也是一个**必须由主席裁定的阈值**——我没有自己加。

### 11.7 本轮之后仍然成立的缺口

§九的 10 条缺口逐条复核，结果如下：

| § 九 | 状态 |
|---|---|
| 1. 48 目标上限是全局的，L1 吃光它 | **仍然成立，而且更严重**（见 11.6：L2 窗占比 4.0% → 1.4%） |
| 2. `require_trip_slot=False` 是整合判断 | **仍然成立**；度量已校正（发现 6），数字已放到主席面前，一行可回退 |
| 3. `collected_or_gone` 不等于「被收走了」 | **仍然成立**；docstring 与本节都把读法写窄了 |
| 4. ALL-ON 仍不能在 95% 排除变坏 | **仍然成立，且更远**（UCB95 0.02536 → **0.06865**） |
| 5. 训练 smoke 没做 | **仍然未做（not verified）**；今晚仍是 Python-only + 探针-only |
| 6. `migrate_loot_candidate` 拒绝 full-v2 源 run | **继承的缺口，未解** |
| 7. escrow 罚没围栏只有单测 | **仍然成立**；本轮又加了 sweep 侧的传送档，同样只有单测（`descend_escrow_fraction = 0`，48 局里一次未触发）。SWEEP2 §10.7 那道「撤退与传送都关时无法可问、于是 vest」的开口**原样继承** |
| 8. sweep-v1 跨层记忆缺陷 | **原样继承**（`floor_keys=False`），仍是主席的裁定 |
| 9. 全量套件 3 个收集失败 + 3 个 Biteq ERROR | **仍是树形**，纯净副本上同样跑不了 |
| 10. spend-v2 单法的负号未被解释掉 | **仍然成立**（net −3 / UCB95 0.164） |

**本轮新增的一条**：发现 4 修好的传送档、以及发现 1 修好的绑定，**都是在今晚三臂里不会说话的路径上**（`resource_portal` 不在 16 键里；每条认证臂都用 `sustain-loot-v1`）。它们的证据是单元测试与两次真引擎复现，**不是探针行**。

### 11.8 本轮产物

| 用途 | 路径 |
|---|---|
| 修理脚本（逐条对应发现） | `/home/laure/r17_work/r19/m1rr/fix{1..10}_*.py` |
| 金币通道实证（发现 2） | `/home/laure/r17_work/r19/m1rr/goldchan.py` + `goldchan.json` |
| 发现 1 的真引擎双向复现 | `/home/laure/r17_work/r19/m1rr/{f1check.py,runf1.sh,f1check.log}` |
| 发现 1 的钉子在旧树上确实会红 | `/home/laure/r17_work/r19/m1rr/ast_negcheck.sh` |
| 探针后唯一一处编辑无害的字节码证明 | `/home/laure/r17_work/r19/m1rr/{bytecheck.py,bytecheck.log}` |
| 复核轮探针驱动 | `/home/laure/r17_work/r19/m1/m1_probe_driver_rr.py` |
| 三臂原始行（**权威**） | `/home/laure/r17_work/r19/m1-probe-rr/m1-{off,on,rich}-rows.json` |
| 三臂汇总（**权威**，已与 agg 对账） | `/home/laure/r17_work/r19/m1-probe-rr/m1-summary-all.json` |
| 重算聚合（本节每个数字的来源） | `/home/laure/r17_work/r19/m1rr/aggregates-rr.json` + `agg_rr.py` |
| 汇总/聚合对账脚本 | `/home/laure/r17_work/r19/m1rr/resummarize.py` |
| 测试日志 | `/home/laure/r17_work/r19/m1rr/{t-new.log,t-nbhd.log,t-named.log,t-full.log,t-full-fails.txt,t-full-new.txt}` |
| 树指纹（跑前/跑后） | `/home/laure/r17_work/r19/m1rr/tree-fingerprint-{before,after}.txt` |
| 合并补丁（重新生成） | `/home/laure/r17_work/r19/m1.patch`（21 个文件，+6260 / −180；`src/` 仍与主树逐字节相同） |

凡本节未由上述文件直接支撑的说法，一律标注 not verified。**主树 `/home/laure/AlphaDiablo/diablogym` 本轮未被修改**（唯一例外：账本 append；用 `find -newermt` 验证过）。

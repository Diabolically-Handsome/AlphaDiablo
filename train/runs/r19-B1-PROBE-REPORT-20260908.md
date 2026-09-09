# R19-B1 — butcher-room-v1（按布局识别的屠夫房禁区 v2）

实现者 `r19-b1-implementer`，2026-09-08。工作副本 `/home/laure/r17_work/r19/b1-tree`，
自建桥 `/home/laure/r17_work/r19/b1-build`（`b1-tree/build` 指向它）。
主树 `/home/laure/AlphaDiablo/diablogym` 全程只读（唯一例外是追加式台账）。
补丁 `/home/laure/r17_work/r19/b1.patch`（`patch -p1`，**15 个文件**，复核轮后重新生成
并验证过：干净副本上 `patch -p1` 之后 15 个文件与 `b1-tree` 逐字节相同，
`verify_patch.sh`）。

---

> **2026-09-08 复核轮**：本报告已按 B1 代码复核的 11 条发现修订过一遍，
> 全部改动、重跑的两条臂与仍未做的事都写在末尾的 **§10 复核轮**。
> §2-§9 的正文里凡是被复核改掉的地方，都就地标了「复核轮」。

## 1. 一段话结论

法条建成、默认关闭逐位同构、**回归闸门通过**：把新旗关掉，我的树 + 我自己重建的桥
在 pool 2_133 的 48 个种子上跑出 `rows_sha_v3 =
0e5a1acd2fb2c07cd27eb72f9ecc574322e72f362ef0376b692376d9a99be886`，与认证的 v1-world
控制行 `ctl-v1-world-193557.json` **逐位相等**（`b1-probe/b1-summary-off.json`，
48 行，0 RuntimeError，646.5 s；**复核轮重跑，仍然相等**）。
开启臂的读数见 §6（**已是复核轮重跑后的数字**；复核修掉的 a9 缺陷
改变了开启臂，配对结果因此比初版弱，见 §10）；主席裁决的两条红线都守住了：房间只由**亮着看见过**的血腥装饰物
（新的原生 `kind="gore"` 通道）反推，绝不用怪物坐标，也绝不做隔黑暗的墙体模板匹配；
`boss_near` 撤退触发器只认 `visible=True` 的屠夫。

---

> 本节所有行号都是**复核轮之后**的 `/home/laure/r17_work/r19/b1-tree` 的行号；
> 复核修正挪动过 `resource_retreat.trigger_reason` 与 `env.py` 里若干函数的位置。

## 2. 设计

### 2.1 房间是可推的：一件亮着的装饰物 = 整间房

`devilutionX/Source/levels/drlg_l1.cpp:381-397` —— 大教堂 L2 若 `Q_BUTCHER` 可用，
`InitSetPiece()` 载入 `levels\l1data\rnd6.dun`（6×6 大砖 = **12×12 世界格**），
落点由 `SelectChamber()`（`drlg_l1.cpp:1267-1293`）在五个 chamber 中选：dungeon 坐标
`{16,2} {2,16} {16,30} {30,16} {16,16}`；世界坐标 = `2*dungeon + 16`，即
**`LEGAL_ORIGINS = {(48,20), (20,48), (48,76), (76,48), (48,48)}`**
（`boss_room.py:107`）。

`AddTortures()`（`devilutionX/Source/objects.cpp:341-363`）扫到 `dPiece == 366` 的锚点
`(ox, oy)` 后，把 12 件血腥物按**固定偏移**盖下去。这 12 个 id 在枚举里连续
（`tables/objdat.h:49-60`：`OBJ_TNUDEM1 = 29 … OBJ_TORTURE5 = 40`），偏移逐字抄进
`boss_room.py:86-100`。侦察给出的锚点 = `W + (3,3)`（`ANCHOR_OFFSET`,
`boss_room.py:101`）。

于是：**位置 = W + ANCHOR_OFFSET + GORE_OFFSETS[id]**，一件看见的血腥物就唯一确定 W。
反解后再与五个合法原点交叉校验；不在其中就是**拒绝**（记 log，本局不设禁区），
而不是围一个凭空发明的矩形。

**锚点是被实测证实的，不是被信任的。** 冒烟局 seed 2133001（`b1-smoke/on.json`）
看见 `OBJ_TORTURE3`(38) 与 `OBJ_TORTURE4`(39) 两件，各自独立反解出
`origin = (48,48)` —— 正好是一个合法 chamber 原点，`refusals = []`。若锚点错了
（小幅偏移），反解值会落在合法集合之外并被拒绝；它没有。

### 2.2 眼睛：`kind="gore"` 骑在 sweep-v1 的同一条通道上

C++ 侧只加**一个新的、独立的、默认 false 的桥全局** `gGoreObservation`
（`src/resource_sweep.hpp:24`, `ConfigureGoreObservation` 于 `:36-44`,
`IsGoreObject` 于 `:49-52`, 导出于 `src/diablogym.cpp:3035`）。
`ObserveSweepObjects()` 只在该旗打开时追加 gore 行（`resource_sweep.hpp:85-92`），
字段与箱子/桶完全同口径（`x, y, kind, visible=IsTileLit, interactable, solid`），
**外加一个整数 `object_id`，且只加在 gore 行上**（`:108-111`）——
给箱子行加键会改变 `SweepService` 已经在读的字典。

**sweep-v1 的行输出逐位不变，两道保险：**
1. 旗关时 C++ 一行 gore 都不发，物体列表与 H1 冻结的完全相同；
2. Python 侧 `SweepService._observe_objects` 现在跳过任何不属于
   `SWEEP_KNOWN_KINDS` 的 kind（`resource_sweep.py:76, 307-309`），
   所以即使通道被别的法条撑宽，`objects_seen_lit` 与每一条窗账也不会动。
   `_candidates` / `_record_skips` 本来就先按 kind 过滤，`gore` 既不在
   `SWEEP_TAKEN_KINDS` 也不在 `SWEEP_SKIPPED_KINDS`，天然被忽略。

`validate_native_sweep` 的"objects 存在 ⟺ sweep 开"身份检查在
「sweep 关 + boss_room 开」这一种组合下会把借用的通道读成泄漏，因此在那一种组合下
改由 `validate_native_boss_room`（`boss_room.py:140-163`）接管 fail-closed 身份，
其余情况一字不改（`env.py:1022-1027`）。gore 通道自己也按 R18-M 复盘的教训记在**类**上
（`DiabloGymEnv._gore_object_channel`, `env.py:645`, `:2356-2362`），
所以同进程里后一个关旗的 env 会负责把它关回去。

### 2.3 记忆与判据

`BossRoomMemory`（`boss_room.py:262-355`）：每局记住**在大教堂主 L2 上亮着看见过**的
gore 物体集合（`visible=True`；看过之后光移走也仍然算数，与
`SweepService._observe_objects` 同口径）。判据：

* **≥1 个决定性 id**（除 30 外的任一个），或
* **≥2 件不同的 gore 物体**（含 30）。

`OBJ_TNUDEM2`(30) 是唯一的弱证据：大教堂上 `THEME_TORTURE` 房间可以单独放它，
所以单看见一个 30 不算数（`GORE_WEAK_IDS`, `boss_room.py:105`）。

作用域 `boss_room_here`（`:165-173`）= 旗开 ∧ `dungeon_level == 2` ∧ 非 set level。
比 R18-J 的"L2+"更窄，因为这个 set piece 只可能在 L2；其它每一层与今天逐位相同。

### 2.4 禁区

`zone_tiles`（`:208-221`）= 12×12 足迹 `W..W+11` **加一格光环** = `W-1..W+12`，共 196 格。
**门口不豁免**：东墙那唯一的开口（大砖 (5,3) = 世界 x `W+10..W+11`, y `W+6..W+7`）
是陷阱的口，豁免它等于允许规划器走到门槛上把屠夫拉出来。

### 2.5 三个规划器 + a9，全部覆盖（R18-J 的第 6 号缺陷的一般化）

R18-J 被否的两个理由之一是"光环只进了 a10"。这一版把禁区送进**全部**决策口：

| 口 | 站点 | 形状 |
|---|---|---|
| a10 探索/hunt 目标 | `env.py` 的 `_plan_explore_step`、`_plan_explore_global_waypoint`（`:4498`） | 禁区并入 `protected` 硬禁集 |
| a10 实际走的路 | `env.py` 的 `_macro_explore` | 先用加宽 `protected_tiles` 的**规划专用**快照，失败退回冻结计划 |
| a9 目标筛选 | `env.py` 的 `controller_action_context`（mask[9] + nearest）、`_engage_candidate_for_action9`（`:3662`）、`_boss_room_filtered_candidates`（`:3570`） | **人在禁区外时**，站在禁区里的怪物永不是 a9 目标；全被筛掉则 mask[9]=False。**人已经站在禁区里时恒等**（复核轮 B1-R1） |
| a9 接敌走位 | `env.py` 的 `_macro_engage` | 同样的"偏好 + 回退" |
| **a11 下楼路径** | `env.py:4847`（`_plan_descend_path`） | **优先**走一条禁区当墙的 BFS；禁区版找不到"比脚下更接近楼梯"的格子时**就从房间里穿过去**（复核轮 B1-R8：这不是"禁区当墙"，是"免费时才绕"） |
| **撤退走路** | 同一个 `_plan_descend_path`，由 `resource_retreat.py` 的 `_walk` 以 `zone_tag="retreat_walk"` 调用 | 同上 |

`_plan_descend_path` 正是 a11/DIVE 与撤退脚本共用的那一个规划器
（a11 的 `_macro_descend` 与 `resource_retreat._walk`），所以一处改动同时补上
R18-J §8.6 明说"仍未覆盖"的那两个洞。

**三条防死锁的性质，逐条对着 R18-J 的复盘缺陷写：**

1. **禁区禁止"进入"，不禁止"站着"**（对 J 的 #2 阻断缺陷）。人已经在禁区里时，
   `_plan_explore_step` 改调 `_plan_boss_room_escape_step`（`env.py:3628`）——
   与局部可达 BFS 同窗、同通行判据的广搜，返回最近的**禁区外**格子，包装成普通的
   `("frontier", x, y)` 由冻结的 `_macro_explore` 执行；整个可达小口袋都在禁区里时返回
   `None`，调用方退回未施加禁区的冻结计划，**绝不会凭空造出一个 wait**。
2. **每一处都是"偏好 + 回退"**：`_boss_room_route_snapshot`（`env.py:3604`）与
   `_plan_descend_path` 的双 BFS 都先试禁区版、失败即用原版。
3. **`nearest` 与 `mask[9]` 同源**（对 J 的 #3）：被拒绝的怪物同时退出
   `controller_action_context` 的第二个返回值，否则 `_farm_handoff` 会在一场 mask[9]
   刚刚拒绝的战斗上看见"贴脸的敌人"，FARM 永不交窗。

wire 的 `protected_mask` 与**原始观测通道**不动：禁区是规划器侧的危险，不是新的引擎职权边界。

**复核轮 B1-R9 更正**：原文写成"与观测一律不动"是错的。`mask[9]` 与
`controller_action_context` 的第二个返回值 `nearest` **按设计会变**（这正是 a9 筛选的
全部意义），而 dual-view 工人的观测向量里**嵌着**动作掩码
（`options_env.py:2343-2347` `_validate_worker_action12_contract`），所以在双视角工人上
观测**会随之变**。变的是掩码那一段，不是 `protected_mask`，也不是任何原始通道。

### 2.6 `boss_near` 撤退触发器

`resource_retreat.py:169-199`（复核轮 B1-R6 之后的位置）：`resource_retreat="retreat-v1"` 且 `boss_room` 开时，
**看得见的**屠夫（`type == 51`，`visible=True`，`hp>0`，非 invalid）在切比雪夫 6 格内
=> 触发原因 `boss_near`。`butcher_visible_within`（`boss_room.py` 的 `butcher_visible_within`）只读
`visible` 标签（桥侧 `IsTileLit && !MFLAG_HIDDEN`），
黑暗里的屠夫对这个触发器不存在。

顺序与预算沿用 R18-J 复盘（J-REPORT §8.5）：**复核轮 B1-R6 后它排在最后一位**——
不只在 `low_hp` / `empty_belt` 之后，也在 `pressure` 之后（初版坐在 `pressure` 上面，
一旦策略带 `pressure_count > 0`，一个看得见的屠夫会抢走本该记在 `pressure` 名下的那一拍
并花掉 boss 预算；v1-world 的 `pressure_count = 0`，所以今晚两条臂都不受影响），
自带 `BOSS_RETREAT_MAX_PER_EPISODE = 1`，且永不动用本局最后
`BOSS_RETREAT_RESERVE = 1` 次撤退。`RETREAT_MAX_PER_EPISODE`、冷却、主 L2+ 作用域一字未改。

窗以**自己的**收窗理由 `boss_retreat_trigger` 关闭（`options_env.py:1515`，`_win_term`），
因为 R18-B 的 descend-escrow 罚没之所以罚 `retreat_trigger`，是因为那个触发器坐在
hp≤50%（等价死亡风险）；boss 触发可以在满血开火。`worker_env.py:2472-2476` 只加了一条
注释说明这个理由**故意**不在罚没集合里 —— 罚没法条一字未动。

### 2.7 词表登记（与 hunt_scope / sweep-v1 同样的地方）

`env.py`（env kwarg + 校验 + 每局记忆 + reset 清零）、
`options_env.py`（OptionsEnv kwarg + 直通 + 撤退策略）、
`worker_env.py`（训练侧直通，与部署侧同一个 validator）、
`probe_r17_deployment.py`（`_R16_ENV_KEYS`）、
`train/train_ppo.py` 与 `train/eval_contract.py` 的两个指纹 bundle ——
`tests/test_r18b6_training_wiring.py` 要求每个 `python/diablogym/*.py`
法条模块都在两个 bundle 里。

**复核轮 B1-R7 补齐的第五、第六处**：`train/train_ppo.py` 的 `--boss-room`
（`ap.add_argument` + `_validate_args` + `make_env` 形参与 `resource_kwargs` 直通 +
契约世界字典 + `_ENVIRONMENT_RESTART_ALLOWED_DRIFT`）与 `train/eval_assembled.py`
的 `--boss-room`（`ap.add_argument` + `requested_r16`），加上档案身份词表
`train/eval_contract.py` 的 `R16_ENVIRONMENT_DEFAULTS["boss_room"] = "off"`
与它自己的 validator 分支。初版这两处是空的，等于**组装认证跑根本打不开这条法**。

**落地代价（复核轮 B1-R11，必须说在前面）**：把 `python/diablogym/boss_room.py`
加进 `train/eval_contract.py` 的 `PROTOCOL_SOURCE_FILES` 之后，任何**早于今晚**的
eval 档案在这棵树上都会被 `validate_eval_archive` 以"协议源码 bundle 文件集合不完整"
拒掉。每一个新法条模块（`aggro_cap.py`、`engagement.py`……）都付过同一笔代价，
所以这是**落地后果**而不是缺陷；但 b1.patch 落到主树的那一刻起，既有 R16/R18 档案
在该树上复验不了，这件事要主席知情后再落。

**没有任何既有法条的阈值被改动。**

---

## 3. 遥测（只做加法）

`probe_r17_deployment.py:761-773`，落在行的 `resource` 字典里：

* `boss_room`（旗字面量）
* `boss_room_telemetry` = `BossRoomMemory.telemetry()`：
  `detected` / `detect_beat` / `origin` / `gore_seen_ids` / `gore_seen_objects` /
  `zone_blocks`（按规划器分桶）/ `boss_near_triggers` / `refusals` /
  `anchor_offset` / `room_size` / `zone_halo`
* `boss_near_triggers`（撤退 attempts 里 `trigger == "boss_near"` 的条数）
* 另加一条**与旗无关**的归因事实：`death.butcher_near` 与
  `last_live.butcher_near`（`probe:173-184, 574, 705`）—— 该活体快照上 6 格内有没有
  **看得见的**屠夫。

`V3_ROW_KEYS` 一个键都没加，所以 `rows_sha_v3` 不受影响 —— 这正是 §5 的回归能成立的原因。

**`zone_blocks` 的口径要说清楚，免得被读大：**
`a9_engage` 是**真正被拒绝的候选数**；`a10_explore` / `a10_global` / `a10_route` /
`a11_descend` / `retreat_walk` 是**"禁区在这次规划调用里被施加"的次数**，
不是"某一格被拒绝"的次数，**更不是"真的绕了路"的次数**（复核轮 B1-R8：
`a11_descend` / `retreat_walk` 计的是"禁区版 BFS 被跑了一次"，那一次也可能返回
`None`、最终仍旧从房间里穿过去）。它是"法条在多少次规划里真的出手了"的下界式代理量。

**复核轮的两处口径修正：**

1. `a10_global` 以前在"人站在禁区里"的那些拍上也记一次 block，而下一行恰恰**不施加**
   禁区——计数在替一道没有升起的墙作证。现在与 `_plan_descend_path` 同款：只有真的
   施加了禁区才记（`env.py` 的 `_plan_explore_global_waypoint`）。
2. `boss_room_telemetry.boss_near_triggers` 这个**记忆里的**键以前从头到尾没有任何地方
   给它加过 1（复核 B1-R5），连真的触发了的那三局也是 0；现在由 `OptionsEnv` 在
   `retreat.start(...)`（唯一真正开始一次 boss 撤退的地方）调
   `BossRoomMemory.note_boss_near_trigger()`。探针行里那个**独立**的
   `resource.boss_near_triggers`（数 `attempts` 里 `trigger == "boss_near"` 的条数）
   一直是对的；两者现在应当一致，§10 给出实测值。

---

## 4. 测试

`tests/test_r19_boss_room.py`（新文件）：初版 **65 tests / 97 subtests**；
**复核轮后 103 tests / 97 subtests passed**（新增 38 条，全部因为一条具名的复核发现，
见 §10）。覆盖：

* 协议元组、校验器（接受/拒绝/报错）、作用域（L1/L3/set level/None 全部关闭）；
* fail-closed 原生身份（缺通道、缺 `visible`、gore 行缺 `object_id` 都 raise；
  箱子行不需要 `object_id`）；
* **五个合法原点 × 12 个 id 的反解全覆盖**（60 个 subtest），
  偏移逐条对着 `objects.cpp:341-363` 断言；
* **非法原点的拒绝**（五个原点 × 五种一格/两格偏移，全部 `None`），未知 id 的拒绝；
* **弱 id 规则**：单个 30 不判定；不亮的 gore 永不进记忆；光移走后记忆仍在；
  非法原点被记进 `refusals` 且禁区为空；两件物体反解不一致 => `origin_disagreement`；
* **禁区几何**：14×14=196 格、原点角/远角/光环在内、外一格在外、门口四格在内、
  屠夫出生点 `W+(4,4)` 在内、`in_zone` 与 `zone_tiles` 全等；
* **旗关什么都不变**：`_boss_room_zone` 空集、a9 过滤返回 `None`（复用冻结元组）、
  `_boss_room_route_snapshot` 返回**同一个对象**（`assertIs`）、
  且 `_plan_explore_step` 在 25×25 全可走合成快照上 **旗关计划 == 旗开未侦测计划**；
* a10 规划：目标落在禁区外；**站在禁区里返回逃离命令而不是 wait**；
  逃离是最近的外部格；封死的口袋返回 `None`（调用方回退）；
* 路由快照只加宽 `protected_tiles`，不动 `protected_mask` 与 `candidates`；
* 撤退触发器：旗关时静默、满血触发、`low_hp` 压过它、黑暗中的屠夫不触发、
  自有预算与 reserve、主 L2 作用域不变、策略键只在开时出现且冻结的五键不变；
* sweep 隔离：`SweepService.observe` 看见 gore 行时 `objects_seen_lit` 不动，
  看见箱子时照常加一；
* probe 文件契约（旗在白名单里、行带三个遥测键）；
* **复核轮新增**：a9 在"人已站在禁区里"时恒等（B1-R1）；refusals 去重
  （同一条非法原点跑 39 拍仍只记 1 条，B1-R2）；决定性 id 投票
  （`Theme_Torture` 的 id-30 诱饵不再否决真原点，B1-R3）；
  `_plan_descend_path` 的禁区分支（禁区版优先 / 禁区版返回 `None` 时回退冻结版并
  真的穿过房间 / 站在里面时恒等 / `zone_tag` 进对桶，B1-R4a）；
  `_plan_explore_global_waypoint` 的禁区分支（房里的怪不是 hunt 目标、房外的还是、
  旗关恒等、站在里面恒等，B1-R4b）；`_macro_engage` / `_macro_explore` 用的那条
  路由偏好（冻结快照会穿过房间、路由快照不会，B1-R4c）；
  `_win_term` 的收窗理由（`boss_near` → `boss_retreat_trigger`，`low_hp` →
  `retreat_trigger`，B1-R4c）；`boss_near` 排在 `pressure` 之后（B1-R6）；
  `boss_near_triggers` 计数器真的会动（B1-R5）；两个 CLI 与档案身份词表
  （B1-R7）。

初版邻域回归（`/home/laure/r17_work/r19/b1-tests.log`）：
`test_r19_boss_room` + `test_resource_sweep` + `test_resource_retreat` +
`test_resource_retreat_training` + `test_hunt_scope` + `test_options_env` +
`test_aggro_engagement` + `test_resource_portal` + `test_resource_protocol` +
`test_r18b6_training_wiring` = **457 passed, 705 subtests passed**（105.9 s）。

过程中出现过、已经处理干净的两类失败：

1. `tests/test_r18b6_training_wiring.py::FingerprintTests` 报
   `python/diablogym/boss_room.py` 不在两个 bundle 里 —— **真缺陷**，已登记
   （`train_ppo.py:851`, `eval_contract.py:97`）。
2. `tests/test_resource_retreat.py::RetreatCommandTests` 的 `FakeEnv._plan_descend_path`
   不接受新的 `zone_tag` 关键字 —— 桩子问题。真规划器上这个关键字是**带默认值的**、
   纯加法的（`env.py:4822`），八个既有调用点一字未改；桩子加了 `**_zone_kw` 并写明原因
   （`tests/test_resource_retreat.py:71-75`）。
3. `tests/test_worker_env.py` 在副本里**收集就失败**（缺
   `train/models/v22-h-manager/policy.npz`），这是"副本不带模型工件"的限制，不是本改动：
   同一批测试在主树上跑是 `124 passed, 334 subtests passed`。
   为让 `test_r18b6_training_wiring` / `test_resource_retreat` 能真正跑起来，副本里对
   `train/runs/r16-arm-a-constitution` 建了一个**指向主树的只读软链**（不是拷贝）。

**复核轮的邻域回归**跑得比初版宽：凡是
`grep -l 'options_env|worker_env|retreat|resource_sustain|train_ppo|eval_assembled|
eval_contract|boss_room|engagement|explore'` 命中、且干净副本里也存在的
**74 个**测试文件，在**本树**与**未改动的干净副本
`/home/laure/r17_work/r19/base-tree`** 上各跑一遍，逐条比对失败清单
（`run_wide_both.sh`，把 `FAILED` / `SUBFAILED` / `ERROR` 三种行都收进来）：

| | 本树 (`b1-wide-raw.txt`) | 干净副本基线 (`baseline-wide-raw.txt`) |
|---|---|---|
| passed / subtests | **2 377 / 1 510** | 2 273 / 1 391 |
| failed / errors | **45 / 3** | 57 / 3 |
| 不同的失败结点 | 48 | 60 |
| **相对基线新增的失败** | **0 条** | — |
| 基线失败、本树反而过的 | **12 条** | — |

剩下的失败都属于"副本不是完整仓库"这一类（`run_r7` / `run_v4` 恢复机制、
`content_case` 驱动等），在基线上一样失败。

**这一步真的抓到了一条我自己的回归**（记作 **B1-R13**，详见 §10.1）：
B1-R7 给 `train_ppo._training_contract` 加的 `boss_room` 键，让 loot 暖启动的
`migrate_loot_candidate.target_contract` 与实时训练契约的**键集合**不再相等，
于是 `LiveWarmStartIdentityTests` / `LiveTrainingContractIdentityTests` 的 10 个
subtest 全挂。第一次比对因为只 `grep '^FAILED'`、漏掉了 `SUBFAILED` 行而**没看见**它；
把三种行都收进来重比才暴露出来。修法照抄同一文件里 R18-M 对 `resource_portal` 的
处置（`target.setdefault("boss_room", None)`，写在 `world` 之外、fail-closed）。
为让 `test_worker_env.py` / `test_content_case_aux.py` 能被收集，副本里对
`train/models` 建了一条**指向主树的只读软链**（不是拷贝）。

**完整 10 分钟测试套件仍然没有跑**（见 §8 已知缺口）。

**复核轮 B1-R10（主树卫生事故，已处理）**：复核指出主树里除台账外还有一个文件的
mtime 落在今晚——`.pytest_cache/v/cache/nodeids`。那是 pytest 默认把缓存写在
rootdir 下造成的，**没有任何源码文件被动过**（主树 `python/` `src/` `train/` 下
每个文件的 mtime 都早于 20:26）。复核轮起，本任务的每一次 pytest 都带
`-p no:cacheprovider`，并且**没有再在主树上跑过任何测试**；事故本身记进了台账事件
`R19_B1_REVIEW_ROUND` 的 `main_tree_hygiene` 字段。

---

## 5. 回归闸门（关旗 = 逐位同构）

`b1-probe/b1-summary-off.json`：

| 项 | 值 |
|---|---|
| probe | `r17-deployment-v3-r18m2` |
| worker | `7e31dc54` = `train/runs/r16-arm-a-constitution/model_candidate.zip` |
| overrides | `gates/v1world-overrides.json` 的 16 键，**不含** `boss_room` |
| seeds | 2133000-2133047（3 分片并行 a/b/c），max_steps 6000，decoding sample |
| `rows_sha_v3` | `0e5a1acd2fb2c07cd27eb72f9ecc574322e72f362ef0376b692376d9a99be886` |
| 期望（认证控制行） | `0e5a1acd2fb2c07cd27eb72f9ecc574322e72f362ef0376b692376d9a99be886` |
| **相等** | **true** |
| 行数 / RuntimeError | 48 / 0 |
| 用时 | 646.5 s（复核轮重跑） |

这条闸门同时证明了两件事：Python 侧默认关闭逐位同构，**以及**我自己从
`/home/laure/alphadiablo-dev/devilutionX`（未改动，patch 0014 已在）重建的桥
`b1-build/_diablogym…so`（sha256 `c95da114…0877`）在 gore 旗关闭时与线上
`build-res` 行为等价（`build-b1.log` 另有 `RESOURCES_IDENTICAL`）。

---

## 6. 开启臂 vs 认证控制行

> **这一节的全部数字来自复核轮的重跑**（`b1-probe/b1-analysis-review.json`）。
> 初版的数字（`b1-analysis.json`）已被 a9 缺陷 B1-R1 污染，两者的差在 §10.3。

臂 = 16 键控制 overrides + `{"boss_room": "butcher-room-v1"}`，同 48 个种子、同 worker、
同 probe、max_steps 6000、decoding sample、3 分片并行。
数据：`b1-probe/b1-on-merged.json`、`b1-probe/b1-analysis-review.json`。
`on rows_sha_v3 = c3e37785c367a2619efbb5a9e466322c0e62e331683b8b4415294e4f67ea6a88`，
0 RuntimeError，643.2 s。

### 6.1 arm_stats（公式逐字取自 `~/r17_work/r18/m2_probe_driver.py:arm_stats`）

| | 控制行（v1-world） | 开启臂（复核轮） |
|---|---|---|
| n | 48 | 48 |
| **存活** | **20** | **22** |
| 到 L2 / 到 L3 | 34 / 4 | 34 / 4 |
| **L2 死亡** | **19** | **17** |
| L2 停留拍 | 85 578 | 86 572 |
| **L2 hazard / 1k** | **0.2220** | **0.1964** |
| L2 击杀 | 894 | 911 |
| 终局金币 中位 / 合计 | 78.0 / 6 855 | 93.5 / 7 144 |
| 开箱 / 碎桶 / 扫窗 | 253 / 231 / 104 | 253 / 231 / 104 |
| 鉴定件数 / 花费 / 鉴定后售价 | 16 / 1 600 / 3 769 | 16 / 1 600 / 3 769 |
| 武器升级 / 花费 | 29 / 6 620 | 29 / 6 750 |
| 撤退次数 | 71 | 70 |

### 6.2 配对 McNemar（公式逐字取自同一文件的 `paired()`）

| saved | lost | net | discordant | **UCB95 单侧** |
|---|---|---|---|---|
| **2** | **0** | **+2** | 2 | **+0.00578** |

**UCB95 > 0：这条臂没有越过单侧 95% 门槛。**
初版报告写的 −0.00503（越线）是在 a9 缺陷下跑出来的，缺陷修好后它退回到 +0.00578。
方向仍然是对的（**0 对反向**），但"越过显著性门槛"这句话**必须收回**。
被救下的两局是 2133018 与 2133020，两局都是"控制行死在 L2、开启臂活到最后"。
n = 48、零训练、单一 worker、单次运行、**2 对不一致** —— 这是"方向对且没有反例"，
连"效应量的方向被钉住"都还谈不上。

### 6.3 侦测（每局侦测数 / 侦测拍）

| 项 | 值 |
|---|---|
| 有遥测的局 | 48 |
| 到过 L2 的局 | 34 |
| **亮着看见过 gore 的局 / 侦测成功的局** | **9 / 9** |
| 侦测率（对到过 L2 的局） | **0.265** |
| **拒绝（非法原点 / 原点不一致）** | **0** |
| 侦测拍 中位 / 最小 / 最大 | 11 261 / 6 352 / 18 786 |
| 观测到的原点 | (48,48)×4, (48,76)×2, (76,48)×1, (20,48)×1, (48,20)×1 —— **五个合法原点全部出现过** |
| 看见过的 gore id | 29-40 全部 12 种 |
| 看见的 gore 物体总数 | 42 |

侦测侧与初版**逐条相同**（同样 9 局、同样的侦测拍、同样的原点、同样 0 拒绝）——
这是应该的：复核修掉的三条都不改变"看见什么、推出哪间房"，只改变"认出之后怎么做"。
看见的 gore 物体从 39 涨到 42（id 31 首次出现），是因为修好 a9 之后，人在房里能动手，
于是在房里多待了一会儿、多点亮了几件。

每局至多侦测一次（房间只有一间），所以"每局侦测数" ∈ {0,1}；**9 局各 1 次**。

**决定性发现（复核轮后依然成立）：侦测集合 == 与控制行 V3 列有差异的种子集合，
一个不多一个不少。**

```
detected  seeds = [2133001, 2133010, 2133018, 2133020, 2133026, 2133027, 2133037, 2133045, 2133047]
v3-diff   seeds = [2133001, 2133010, 2133018, 2133020, 2133026, 2133027, 2133037, 2133045, 2133047]
```

法条**只在它真的认出房间的那 9 局里改变了轨迹**，其余 39 局与控制行逐位相同 ——
这是"默认关闭 + 作用域收窄"之外，第三重可审计的收敛证据。

逐局：

| seed | 原点 | 侦测拍 | 证据 id | 控制行 | 开启臂 | 记忆里的 `boss_near` |
|---|---|---|---|---|---|---|
| 2133001 | (48,48) | 11 261 | 38,39 | 死 @L2 | 死 @L2 | 0 |
| 2133010 | (48,48) | 8 732 | 29-40（12 种） | 死 @L2 | 死 @L2 | 1 |
| 2133018 | (48,76) | 12 941 | 38,39 | 死 @L2 | **活** | 0 |
| 2133020 | (48,48) | 18 786 | 38,39 | 死 @L2 | **活** | 0 |
| 2133026 | (76,48) | 11 278 | 38,39 | 死 @L2 | 死 @L2 | 0 |
| 2133027 | (20,48) | 7 147 | 29,30,32,33,34,35,36,37,40 | 死 @L2 | 死 @L2 | 1 |
| 2133037 | (48,48) | 6 352 | 29,30,32,33,34,35,36,37,40 | 死 @L2 | 死 @L2 | 1 |
| 2133045 | (48,20) | 16 925 | 38,39 | 活 | 活 | 0 |
| 2133047 | (48,76) | 7 235 | 38,39 | 活 @L3 | 活 @L3 | 0 |

两种证据形态仍然很干净：**6 局只看见 `{38,39}`**（`OBJ_TORTURE3`/`TORTURE4`，
偏移 `(2,-1)`/`(4,-1)`，即 y 最小的那两件，落在房间北侧的墙线 `W+2` 上；
"它们是从房外最先被点亮的两件"是对这个分布的**推测**，未单独验证），
**3 局看见九件以上的那一簇**（已经走到房口/房内），而恰好就是这 3 局触发了 `boss_near`。

### 6.4 zone_blocks 与 `boss_near`

| 规划器桶 | 复核轮计数（全部落在那 9 局里） | 初版 |
|---|---|---|
| `a9_engage`（**真被拒绝的候选数**） | **407** | 939 |
| `retreat_walk`（撤退走路施加禁区的规划次数） | 1 175 | 1 369 |
| `a10_route`（a10/a9 走位路由快照） | 557 | 611 |
| `a10_explore`（a10 目标规划） | 221 | 290 |
| `a11_descend`（a11/DIVE 下楼规划） | 27 | 33 |
| `a10_global`（全图 waypoint） | 0 | 0 |

`a9_engage` 从 939 掉到 407，**正是复核修正 B1-R1 的直接读数**：那 532 次里有一大批
是"人已经站在房里，法条却把每一只怪都拒了"——包括贴脸的那只。修好之后，
只有"人在房外、目标在房里"才算拒绝。

`a10_global` 为 0 是可解释的而不是可疑的：控制世界里 `hunt_scope="l1-only"`，
L2 上全图 hunt 变体本就极少被走到。

`boss_near` 撤退触发：**开启臂 3 次（3 局各 1 次，正好等于 per-episode 上限），
控制/关闭臂 0 次**。撤退触发直方图：关闭臂 `{low_hp: 71}`，
开启臂 `{low_hp: 67, boss_near: 3}` —— 与 R18-J §8.5 的教训一致，
`boss_near` **没有**去挤占 HP 急救的预算（`low_hp` 从 71 变成 67 的差异来自轨迹改变，
不是预算被吃掉：boss 触发自带 1 次上限且永不动用最后 1 次储备）。
**复核轮新增的一致性检查**：记忆里的 `boss_room_telemetry.boss_near_triggers`
（B1-R5 修好之后）合计 **3**，与探针那个独立计数的 3 **相等**，且正好落在
2133010 / 2133027 / 2133037 三局上。

### 6.5 屠夫归因的 L2 死亡

**用的是便宜口径，不是 R18-X 的重放归因**（重放不便宜，故按简报允许的退路走）：
一条 L2 死亡，若它**最后一次活体快照**上 6 格内有一个**看得见的**屠夫，就记在屠夫头上。
这条事实与旗无关地写进两臂的行里（`death.butcher_near`）。

| | 关闭臂（= 认证控制行，sha 相等） | 开启臂（复核轮） |
|---|---|---|
| L2 死亡总数 | 19 | 17 |
| **屠夫归因的 L2 死亡** | **6** | **3** |
| 归因不了的 L2 死亡（缺快照） | 0 | 0 |
| 归因种子 | 2133001, 2133010, 2133018, 2133026, 2133027, 2133037 | 2133010, 2133027, 2133037 |

被救下的 2 局（2133018、2133020）里 2133018 在控制行是**屠夫归因**的死亡；
2133001 与 2133026 的屠夫归因死亡也消失了（它们在开启臂仍死在 L2，但最后一刻身边
没有屠夫）。剩下 3 局（2133010、2133027、2133037）**都是那 3 局"走到房口才看见九件以上
血腥物"的局** —— 法条认出房间的时候，屠夫已经贴上来了；这正是 §8 的头号缺口。

### 6.6 L3 触达

4 → 4，**没动**。这与法条的意图一致（它是一条降低 L2 危险的法条，不是一条加深度的法条），
也说明禁区没有把下楼路堵死：`a11_descend` 桶只有 27 次，而 `_plan_descend_path`
的"禁区版失败即回退冻结版"从未导致任何一局卡住（0 RuntimeError，L2 停留拍
85 578 → 86 572，只多了 1.2%）。

---

## 7. 老实的读法

**能说的：** 机制在真实运行里成立且自洽 —— 9 局侦测、0 拒绝、五个合法原点全中、
侦测集合恰好等于轨迹变动集合；方向对（存活 +2、L2 死亡 −2、hazard −11.5%、
屠夫归因死亡 6→3），且 **0 对反向**。

**不能说的：** 这不是认证，而且**比初版报告更弱**。整个结论压在 **2 对不一致**上，
n = 48、零训练、单一 worker、每臂只跑了一次。按预注册公式 **UCB95 = +0.00578 > 0，
没有越过单侧 95% 门槛**——初版报告说"越线"，那是一条 a9 缺陷（B1-R1）撑起来的结论，
缺陷修好后它就退了回去。这条更正是本轮最重要的一句话。
"屠夫归因"用的是最后活体快照的便宜代理，不是伤害账本；它会漏掉"屠夫把你打残、
你走两步被别的怪补刀"的死亡，也会误收"屠夫恰好在场"的死亡。
`zone_blocks` 里除 `a9_engage` 外都是"规划调用次数"而不是"格子被拒次数"，
不能当成"避开了多少步"来读。

**最该看的一行仍然是 6.5 的最后一段**：法条救下的都是"远远看见 38/39 就绕开"的局，
而它没能救下的三局，都是"看见九件以上血腥物"——也就是人已经站在房口的那一刻才认出房间。
下一版的价值不在于把禁区做得更大，而在于**更早看见**。

---

## 8. 已知缺口 / 未验证

1. **侦测太晚**（最大的一条）。侦测拍中位 11 261，且 3/9 局是走到房口才认出来。
   `{38,39}` 那一对贴在北墙外沿，是从房外最先会被点亮的证据；如果 34 局里有更多局
   曾经"路过却没点亮"，我没有量它 —— **未验证**：我没有统计"到过 L2 但从未看见任何
   gore"的 25 局里，有多少局其实走到过房间附近。
2. **锚点 `W+(3,3)` 是被实测支持、不是被证明的。** 支持它的是：9 局侦测全部反解到
   五个合法原点之一、0 次拒绝、且两种独立的证据形态（`{38,39}` 与九件簇）给出一致结论。
   我**没有**去引擎里直接读 `dPiece == 366` 的坐标来做一次形式验证（那需要新的桥通道）。
   万一锚点错，失效模式是安全的（反解落到非法原点 → 记 log 拒绝 → 法条不生效）。
3. **弱 id 规则未被真实数据检验**：42 件被看见的 gore 里 id 30 只随那一簇一起出现，
   从来没有出现过"只看见一个 30"的局面，所以 `GORE_WEAK_IDS` 分支只有单元测试覆盖。
   复核轮 B1-R3 把这条规则收得更严（弱 id 不再参与投票、更不能否决），
   但**收严之后仍然只有单元测试作证**——探针里根本没有出现过诱饵。
4. **`a10_global` 桶仍为 0**：全图 hunt 路径上的禁区逻辑在两次探针里都**没有被执行过**
   （`hunt_scope="l1-only"`）。初版报告说它"有单元覆盖"是**假的**（复核 B1-R4b 实锤：
   `grep -rl _plan_explore_global_waypoint tests/` 当时什么都搜不到）。
   复核轮补上了 5 条真的单元测试，所以现在这句话成立了；但**真实运行覆盖仍然是零**。
5. ~~训练侧只登记了指纹 bundle，没有 CLI 旗~~ —— **复核轮 B1-R7 已补齐**。
   初版的这一条还漏说了一半：`train/eval_assembled.py` 里连提都没提 `boss_room`，
   所以**一次组装认证跑根本没法把这条法打开**。现在两个 CLI 都有 `--boss-room`：
   `train_ppo.py`（`ap.add_argument` + `_validate_args` + `make_env` 签名与
   `resource_kwargs` 直通 + 契约世界字典 + `_ENVIRONMENT_RESTART_ALLOWED_DRIFT`）与
   `eval_assembled.py`（`ap.add_argument` + `requested_r16`），
   档案身份词表 `eval_contract.R16_ENVIRONMENT_DEFAULTS` 加了 `"boss_room": "off"`
   并有自己的 validator 分支。与 `hunt_scope` 不同，这条法**不需要**再钉一个伴随开关：
   它自己的消费者（三个规划器 + a9）在 L2 侦测到房间那一拍就是活的，
   所以一份写着 `butcher-room-v1` 的身份不会为一部没跑过的法作证。
   **仍未验证**：没有真的跑过任何训练或组装认证（今晚是零训练探针），
   新加的 CLI 只有单元/契约测试作证。
6. **`_plan_descend_path` 多了一个带默认值的关键字 `zone_tag`**。对真实调用者是纯加法，
   但任何**镜像该签名的测试桩**都需要吃掉它（本次只有 `test_resource_retreat.py` 一个）。
7. **完整 10 分钟测试套件仍然没有跑**。复核轮把邻域从 457 项扩到 74 个文件 /
   2 377 passed，并用**未改动的干净副本**做了同清单基线对照，证明**新增失败 0 条**；
   但那仍然不是全量套件。副本里剩下的 55 条失败都属于"副本不是完整仓库"
   （`run_r7` / `run_v4` 恢复机制、`content_case` 驱动等），基线上也失败。
8. **每局只侦测一间房**：`BossRoomMemory` 认定一个原点后就不再更新。L2 只有一个 set piece，
   所以这在 v1 是对的；若将来把法条推广到别的层，这条假设要重写。
9. **`zone_blocks` 的口径**（§3 末）不是"被拒绝的格子数"，别当效应量读。
10. **归因口径**：见 §7；R18-X 的重放归因**没有**跑。
11. 遥测里 `boss_room_telemetry.refusals` 两次探针都全部为空，所以**拒绝路径只有单元
    测试覆盖，没有真实运行的证据**。复核轮 B1-R2 顺手把它的**代价**堵上了：
    以前只要证据够、原点又推不出唯一解，`_try_detect` 就**每一拍**往 `refusals` 里塞一个
    字典（6000 拍的一局约 1.2 万条），而 `telemetry()` 会把整份列表原样写进探针行。
    现在同一条拒绝只记一次。这条路径**依旧没有被真实数据走过**。
12. **复核轮的两条臂各只跑了一次**，与初版一样。初版与复核轮的开启臂差异
    （§10.3）本身就说明：这个体量的臂，一处行为改动就能把结论从"越线"翻成"没越线"。

---

## 9. 交付物

| | 路径 |
|---|---|
| 工作树 | `/home/laure/r17_work/r19/b1-tree` |
| 桥构建目录 | `/home/laure/r17_work/r19/b1-build`（`_diablogym…so` sha256 `c95da114fcf40ecc470925eea6cbe461da976684f82aa64ff0fe20b7d1100877`） |
| 构建日志 | `/home/laure/r17_work/r19/build-b1.log`（`RESOURCES_IDENTICAL`） |
| 补丁 | `/home/laure/r17_work/r19/b1.patch`（`patch -p1`，**15 文件**，复核轮后重生成并验证） |
| 测试日志 | 初版 `/home/laure/r17_work/r19/b1-tests.log`；复核轮 `b1-wide.txt`（本树）与 `baseline-wide.txt`（干净副本基线） |
| 关闭臂 | `/home/laure/r17_work/r19/b1-probe/b1-summary-off.json`、`b1-off-merged.json` |
| 开启臂 | `/home/laure/r17_work/r19/b1-probe/b1-on-merged.json`、`b1-summary-on.json` |
| 分析 | 初版 `/home/laure/r17_work/r19/b1-probe/b1-analysis.json`；**复核轮 `b1-analysis-review.json`（正文用的就是这一份）** |
| 冒烟（锚点实测） | `/home/laure/r17_work/r19/b1-smoke/on.json` |
| 探针驱动 | `/home/laure/r17_work/r19/b1_probe_driver.py` |
| 本报告 | `/home/laure/r17_work/r19/B1-REPORT.md` |
| 复核轮代码补丁脚本 | `patch_review.py`、`patch_review2.py`、`patch_review4.py`、`patch_tests.py`（同目录） |
| 复核轮报告补丁脚本 | `patch_report1.py` … `patch_report6.py` |
| 复核轮分析 / 台账脚本 | `scripts/analyze_b1_review.py`、`scripts/diff_seeds.py`、`scripts/ledger_b1_review.py` |
| 补丁自验 | `/home/laure/r17_work/r19/verify_patch.sh`（干净副本 + `patch -p1` 后 15 文件逐字节相同） |

台账：`R19_B1_BUTCHER_ROOM_V1_IMPLEMENTED`、`R19_B1_PROBE_RESULT` 与
**`R19_B1_REVIEW_ROUND`** 已追加到
`/home/laure/AlphaDiablo/diablogym/train/runs/r10-staging/r13_ledger.jsonl`。

主树未被修改（台账除外），`/home/laure/alphadiablo-dev/devilutionX` 未被修改，
虚种子池 2_116-119 / 2_126-128 零接触。


---

## 10. 复核轮（2026-09-08，`r19-b1-review-fixer`）

B1 的代码复核提了 11 条（3 条 major 判定为"阻断"级、8 条 minor）；
我自己在写测试与加宽回归时又抓到 2 条（B1-R12 遥测计数、**B1-R13 我自己造成的回归**）。
本节逐条写清楚**做了什么**、**没做什么**，以及重跑的两条臂。
所有代码改动都落在同一个副本 `/home/laure/r17_work/r19/b1-tree`，主树仍然只读。

### 10.1 逐条

| # | 级别 | 位置 | 复核说的 | 我做了什么 |
|---|---|---|---|---|
| **B1-R1** | major | `env.py` `_boss_room_filtered_candidates` | a9 筛选器是**唯一**一个没有"人站在里面就放行"豁免的禁区消费者。人一旦进了房（法条自己的回退路径允许这件事，而且那 3 局真的发生了），就**不许打任何一只怪，包括贴脸的那只**，直到走出去为止；`nearest` 也被清零，`_farm_handoff` 看见一片空地板 | **已修**：与另外三个消费者同款——`(player_x, player_y) in zone` 时**直接返回 `None`**（恒等，复用冻结候选元组），逃离计划独占那一拍。新增 4 条单元测试（含"贴脸屠夫在房内、人也在房内 ⇒ a9 照常绑定它"） |
| **B1-R2** | major | `boss_room.py` `_try_detect` | 证据够而原点推不出唯一解时，`refusals` **每一拍**加一个字典（6000 拍 ≈ 1.2 万条），`telemetry()` 原样写进探针行，无界增长 | **已修**：`_refusal_keys` 集合去重，同一条拒绝（`("illegal_origin", id, x, y)` / `("origin_disagreement", 原点元组)`）只记一次；`reset()` 一并清空。新增 4 条单元测试（跑 39 拍仍只有 1 条） |
| **B1-R3** | major | `boss_room.py` 原点投票 | 投票把**弱 id 30** 也算进去。`Theme_Torture` 会在普通主教堂主题房里按 `FlipCoin(6)` 撒 `OBJ_TNUDEM2`，所以主 L2 上出现游离的 id 30 是**常态**；一个恰好坐在另一个合法房原点上的诱饵就能让 `len(origins) != 1`，法条整局静默失效 | **已修**：**只用决定性 sighting 投票**（`decisive_sightings()`）。弱 id 仍然算进"≥2 件"的证据门槛，但**永不否决**；只有在一件决定性证据都没有时才让弱 id 推原点，且那时要求它们全体一致。新增 5 条单元测试（复核给的复现例现在正常侦测到 (48,48)、0 拒绝） |
| **B1-R4** | major | `tests/test_r19_boss_room.py` | 报告声称覆盖的 6 个执行点里有 3 个**零覆盖**：(a) `_plan_descend_path` 的禁区分支与"围栏失败即回退"（a11 + 撤退走路，占观测到的 zone_blocks 的一大半）；(b) `_plan_explore_global_waypoint`（报告说"有单元覆盖"，`grep` 什么都搜不到）；(c) `_engage_candidate_for_action9` / 两个 macro 的路由偏好 / `boss_retreat_trigger` 收窗理由 | **已修**：新增 **38 条**测试（65 → **103 tests / 97 subtests，全绿**）。(a) 6 条：围栏版优先且不进禁区、走廊几何下围栏版返回 `None` 时回退冻结版并**真的穿过房间**、站在禁区里恒等、旗关恒等、`zone_tag` 进对桶、撤退脚本确实带 `zone_tag="retreat_walk"`；(b) 5 条：房内怪不是 hunt 目标、旗关时它是、房外怪仍然是、站在里面恒等、旗关 == 旗开未侦测；(c) 7 条：a9 三态 + 冻结快照会穿过房间而路由快照不会 + 两个 macro 的"偏好+回退"源码契约 + `_win_term` 的三种收窗理由 |
| **B1-R5** | minor | `boss_room.py:278/349` | `boss_near_triggers` 只被初始化、从没被加过 1，连真触发的三局也是 0，报告却把它列成遥测字段 | **已修**：新增 `BossRoomMemory.note_boss_near_trigger()`，由 `OptionsEnv` 在 `retreat.start(...)`（**唯一**真正开始一次 boss 撤退的地方，不是 `trigger_reason`——那个收窗阶梯也会调，会重复计数）调用。**实测验证**：重跑的开启臂里记忆计数合计 = 3，与探针那个独立计数的 3 相等，且正好落在 2133010/2133027/2133037 三局 |
| **B1-R6** | minor | `resource_retreat.py` | `boss_near` 排在 `pressure` **上面**；报告只说了"在 HP 触发之后"。带 `pressure_count > 0` 的策略下，一个看得见的屠夫会抢走本该记给 `pressure` 的那一拍 | **已修**：移到**最后一位**（HP 两条 → `pressure` → `boss_near`），docstring 与报告 §2.6 同步改写。v1-world 的 `pressure_count = 0`，所以两条臂的数字不受影响；新增 3 条测试钉住新顺序 |
| **B1-R7** | minor | `train/eval_assembled.py`、`train/train_ppo.py` | 按 `hunt_scope` 的先例，这条法该出现在**五个**地方；实际只在两个指纹 bundle 里。`eval_assembled.py` 里连字面量都没有 ⇒ **组装认证跑根本打不开这条法** | **已修**：`train_ppo.py` 加 `--boss-room`、`_validate_args` 校验（词表 + 只经 worker/options 直通）、`make_env` 形参与 `resource_kwargs` 直通、契约世界字典、`_ENVIRONMENT_RESTART_ALLOWED_DRIFT`；`eval_assembled.py` 加 `--boss-room` 与 `requested_r16`；`eval_contract.R16_ENVIRONMENT_DEFAULTS` 加 `"boss_room": "off"` 并有自己的 validator 分支。**与 `hunt_scope` 不同不再钉伴随开关**，理由写在代码注释与 §8 缺口 5 里。3 条契约测试 |
| **B1-R8** | minor | `env.py` `_plan_descend_path` | 围栏 BFS 只在"绕路免费"时生效；房间正好卡在人与楼梯之间时它返回 `None`，代码就从房间里穿过去。报告 §2.5 写成"禁区当墙的 BFS"，读起来比代码强 | **改了措辞，没改代码**：§2.5 的那一行现在写"优先走围栏版；围栏版找不到更近的格子时**就从房间里穿过去**"，§3 也写清楚 `a11_descend` / `retreat_walk` 计的是"围栏版被跑了一次"而不是"真的绕了路"。复核提的"限长绕路规则（围栏路径不超过冻结路径的 k 倍就接受）"**没有实现**——那是新的设计取舍，不该在复核轮里偷偷加进去 |
| **B1-R9** | minor | 报告 §2.5 末句 | "wire 的 `protected_mask` 与观测一律不动"是错的：`mask[9]` 与 `nearest` 按设计会变，而 dual-view 工人的观测向量里嵌着动作掩码 | **已改措辞**：§2.5 现在明写"原始观测通道不动；`mask[9]` 与 `nearest` 按设计变；双视角工人的嵌入掩码因此也随之变" |
| **B1-R10** | minor | 主树 `.pytest_cache/v/cache/nodeids` | 主树今晚除台账外还有一个文件 mtime 被动过——pytest 默认把缓存写在 rootdir | **已处理，但那个文件没有被删**（删它等于再写一次主树）。复核轮起：所有 pytest 都带 `-p no:cacheprovider`，且**没有再在主树上跑过任何测试**；事故写进台账事件 `R19_B1_REVIEW_ROUND` 的 `main_tree_hygiene` 字段与本节。主树的 `python/` `src/` `train/` 下没有任何源码文件被改动 |
| **B1-R11** | minor | `train/eval_contract.py:97` | 把 `boss_room.py` 加进 `PROTOCOL_SOURCE_FILES` 之后，任何**早于今晚**的 eval 档案在这棵树上都会因"协议源码 bundle 文件集合不完整"而无法复验 | **已在报告里点名**（本条 + §10.2）。这是每一个新法条模块都付过的落地代价（`aggro_cap.py`、`engagement.py` 同款），不是缺陷；但 b1.patch 落到主树的**那一刻**起，所有既有 R16/R18 档案在该树上都复验不了，**这件事必须由主席知情后再落地** |
| **B1-R13** | 自查 | `train/migrate_loot_candidate.py` | （B1-R7 之后跑加宽邻域时自己发现的**我造成的回归**）给 `train_ppo._training_contract` 加 `boss_room` 键之后，loot 暖启动的 `target_contract` 与实时训练契约的**键集合**不再相等；`json_sha256` 对键的有无敏感，所以每一次 schema/2 的 loot 暖启动都会死在 `validate_inherited_receipt`。10 个 subtest 挂掉 | **已修**：照抄同一文件里 R18-M 对 `resource_portal` 的处置——`target.setdefault("boss_room", None)`，**写在 `world` 之外**，所以 B3b 注册的世界词表（`WORLD_KEYS`、收据的 `target_world`、`ALLOWED_CONTRACT_KEYS`）逐字节不变，而一次真的带 `--boss-room butcher-room-v1` 的 loot 暖启动仍然被 `validate_target_contract` 的"漂移出已注册世界"闸**拒掉**（fail-closed）。真要注册它得单独裁定，不是复核修正该干的事。修完两个 Live 测试 **5 passed / 16 subtests** |
| **B1-R12** | 自查 | `env.py` `_plan_explore_global_waypoint` | （写 B1-R4b 的测试时自己发现的）它在"人站在禁区里"的拍上也记一次 `a10_global` block，而下一行恰恰**不施加**禁区——计数在替一道没升起的墙作证 | **已修**：改成与 `_plan_descend_path` 同款，只有真的施加了禁区才 `note_block`。只影响遥测计数（该桶两次探针都是 0） |

### 10.2 没做的事（明说）

1. **限长绕路规则**（B1-R8 的可选建议）没做——那是设计改动，不是复核修正。
2. **两个 macro（`_macro_engage` / `_macro_explore`）本身没有被端到端测试**：它们需要真引擎。
   复核轮测的是它们用的那条机制（路由快照 → `_plan_controller_path`）加一条源码契约
   （两处都是"路由快照 → 失败回退冻结快照"）。
3. **新加的两个 CLI 旗没有真的跑过一次训练或组装认证**（今晚是零训练探针），
   只有单元/契约测试作证。
4. **主树上那个 `.pytest_cache` 文件没有被删除**（见 B1-R10）。
5. **完整 10 分钟测试套件仍然没有跑**（见 §8 缺口 7）。
6. **拒绝路径与弱 id 路径仍然只有单元测试**（探针里 0 拒绝、没有游离的 id 30）。
7. **`a10_global` 的真实运行覆盖仍是零**。
8. **R18-X 的重放归因仍然没有跑**；屠夫归因用的还是那条便宜代理。

### 10.3 重跑：关闭臂仍然逐位相等，开启臂**变弱了**

两条臂都用同一套配方重跑（pool 2_133 的 48 个种子，3 分片并行 a/b/c，worker 7e31dc54，
probe `r17-deployment-v3-r18m2`，max_steps 6000，decoding sample）。

**关闭臂（回归闸门）**：
`rows_sha_v3 = 0e5a1acd2fb2c07cd27eb72f9ecc574322e72f362ef0376b692376d9a99be886`，
与认证控制行**逐位相等**，48 行，0 RuntimeError，646.5 s。**闸门通过。**

**开启臂**：`rows_sha_v3` 从 `d87ccdef…a7c` 变成 `c3e37785…a88`。

| | 控制行 | 初版开启臂 | **复核轮开启臂** |
|---|---|---|---|
| 存活 | 20 | 23 | **22** |
| L2 死亡 | 19 | 16 | **17** |
| L2 hazard / 1k | 0.2220 | 0.1828 | **0.1964** |
| L2 击杀 | 894 | 927 | **911** |
| 终局金币合计 | 6 855 | 7 414 | **7 144** |
| saved / lost | — | 3 / 0 | **2 / 0** |
| **UCB95 单侧** | — | **−0.00503（越线）** | **+0.00578（没越线）** |
| 屠夫归因 L2 死亡 | 6 | 3 | **3** |
| `a9_engage` 拒绝数 | — | 939 | **407** |
| 侦测局 / 拒绝 | — | 9 / 0 | **9 / 0**（逐条相同） |
| 侦测集合 == V3 差异集合 | — | 是 | **是**（同样那 9 个种子） |

**这一格必须被读见：初版报告说的"今晚第一条越过显著性门槛的臂"，
在 a9 缺陷修好之后不成立了。** UCB95 从 −0.005 变成 +0.006，
被救下的局从 3 局（2133001、2133018、2133020）变成 2 局（2133018、2133020）。
换句话说，初版那一局额外的存活，是靠"人站在屠夫房里却不许还手"这条**缺陷**换来的
——那不是一条法，那是一个 bug 恰好在 n=48 上帮了一次忙。

（初版臂的**逐局**行数据已被这次重跑覆盖写掉，所以我**说不出**那第三局究竟是 2133001
还是 2133026；只能确定它是这两个之一，因为它们是控制行死、初版活、复核轮又死的候选，
而屠夫归因的三局 2133010/2133027/2133037 两版一致。这条我不去猜。）

侦测侧**逐条不变**（9 局、同样的侦测拍与原点、0 拒绝、侦测集合 == V3 差异集合），
这正是应该的：三条修正都不改"看见什么、推出哪间房"，只改"认出之后怎么做"。

### 10.4 复核轮的测试与回归

* `tests/test_r19_boss_room.py`：**103 tests / 97 subtests passed**（0.6 s）。
* 邻域：74 个测试文件，本树与干净副本基线各跑一遍（`run_wide_both.sh`）——
  本树 **2 377 passed / 1 510 subtests / 45 failed / 3 errors**，
  基线 **2 273 passed / 1 391 subtests / 57 failed / 3 errors**；
  把 `FAILED` / `SUBFAILED` / `ERROR` 三种行都收进来逐条比对：
  **新增失败 0 条**，另有 12 条基线失败在本树上反而是过的。
  第一次比对只收了 `FAILED` 行，漏掉 `SUBFAILED`，因此**漏看了 B1-R13**；
  这条教训写在这里，别人重跑时请用 `run_wide_both.sh` 而不是那个旧脚本。
* 补丁重生成为 **15 个文件**（相对初版新增 `train/eval_assembled.py` 与
  `train/migrate_loot_candidate.py`），并自验过：干净副本 + `patch -p1` 之后
  15 个文件与 `b1-tree` **逐字节相同**（`verify_patch.sh`）。
* **两条臂不需要因 B1-R13 重跑**：`train/migrate_loot_candidate.py` 既不被
  `probe_r17_deployment.py` 也不被 `python/diablogym/` 里的任何模块 import
  （`grep` 为空），也不在 `PROTOCOL_SOURCE_FILES` 里，更不碰 `V3_ROW_KEYS`。
  它是 §10.3 的两条臂跑完之后**唯一**被改过的文件。
* 桥**没有重建**（本轮没有 C++ 改动）：`b1-build/_diablogym…so` 仍是
  `c95da114fcf40ecc470925eea6cbe461da976684f82aa64ff0fe20b7d1100877`。

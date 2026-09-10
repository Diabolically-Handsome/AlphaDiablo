# R19-M2 报告：主席四条裁定的合并、探针与归因

- 日期：2026-09-09（`by` = `r19-m2-integrator`）
- 合并树：`/home/laure/r17_work/r19/m2-merge`（`rsync -a --exclude __pycache__`
  自 R19-M1 合并树；`build -> /home/laure/r17_work/r17-1/build-res` 原样保留）
- 补丁：`/home/laure/r17_work/r19/m2.patch`（`diff -ruN` 对**主树**，**25 个文件**，
  **复核轮后重生成：+10028 / −215 行**，全 LF，sha256 `315133a531bbc13d`；复核前是 +9208 / −222）。25 = M1 那一轮已有的 21 个条目 + 本轮新增的 4 个
  （`python/diablogym/resource_emergency_stop.py`、
  `tests/test_r19_m2_emergency_stop.py`、`tests/test_r19_m2_rulings.py`、
  `tests/test_resource_readiness_law.py`）
- 输入：实施者 A 的 `/home/laure/r17_work/r19/m2-tree`（裁定 1/2/3 + coach-v05）、
  实施者 B 的 `/home/laure/r17_work/r19/stop-tree`（裁定 4 的紧急停止 stop-v1）
- 只读树复核：主树 `/home/laure/AlphaDiablo/diablogym` **只被 append 了账本**
  （`find /home/laure/AlphaDiablo/diablogym -newermt "2026-09-09 17:30"`
  ——本轮起点——在整棵主树上只命中
  `train/runs/r10-staging/r13_ledger.jsonl` 一个文件；`src/` 与合并树
  `diff -rq` 逐文件相同 = `SRC_IDENTICAL`）；M1 合并树 `/home/laure/r17_work/r19/merge-tree`
  **本轮零写入**（`grep -c coach-v05` = 0，`resource_emergency_stop` 零命中）
- 探针产物：`/home/laure/r17_work/r19/m2-probe/`（5 臂 × 48 种子 × 3 分片 = 15 个分片作业，
  外加继承的认证控制行与 M1 ALL-ON 行）
- **复核轮（2026-09-09 晚，`by` = `r19-m2-fixer`）**：一位复核者提了 4 high + 5 medium，
  **九条全部改掉**，代码 / 测试 / 探针 / 补丁 / 本报告都重跑重写过。
  **§二、§八、§十、§11.1、§11.3、§十三 里带 `> 复核轮更正` 引用块的段落，
  以复核轮为准**；完整交代见 **§十六**，复核轮探针产物在
  `/home/laure/r17_work/r19/m2rr-probe/`。
  一句话：**OFF 回归两次重跑都仍等于认证控制行；M2-SET 逐比特未变；
  裁定 4 的两条在缺陷修好之后依然赢不过 M2-SET。**

---

## 〇、一页结论

1. **回归通过。** 新法全关（16 键认证控制行）时，合并树的
   `rows_sha_v3` = `0e5a1acd2fb2c07cd27eb72f9ecc574322e72f362ef0376b692376d9a99be886`，
   **与认证控制行逐位相同**。48 行，runtime errors 0。
2. **裁定 1/2/3（M2-SET）是本战役到目前为止最好的一臂**：存活 **28/48**
   （控制 20，M1 ALL-ON 24），L2 hazard **0.1374**（控制 0.2220，M1 0.1573），
   配对 saved 14 / lost 6 / net **+8**，单侧 95% UCB = **−0.0186 < 0**——
   **这是本战役第一次在 95% 下排除“变坏”**。
3. **代价是深度。** 同一臂到达 L2 从 34 掉到 **30**，到达 L3 从 4 掉到 **2**，
   clvl≥4 从 25 掉到 **18**。三条法用**留在浅层**换来了存活。战役的北极星是
   55 层深度，所以这条**必须由主席裁定**，不是我能替他判的取舍。
4. **裁定 4 的两条我加跑了归因臂，结论是它们方向相反**（同 48 种子、同 worker）：
   - `coach-v05` 单独加进 M2-SET：存活 28 → **25**，配对 vs M2-SET
     **saved 0 / lost 3**，hazard 0.1374 → 0.1794。**它只赔不赚。**
   - `stop-v1` 单独加进 M2-SET：存活 **28（持平）**，配对 vs M2-SET
     saved 5 / lost 5 / **net 0**，但 L2 hazard 0.1374 → **0.1270**（全场最低）、
     L2 beats +8.1%，对认证控制行的 UCB95 = **−0.01087 < 0**。
     **它换人不换数：救下 5 个、赔掉 5 个。**
   - 两条同开（M2-FULL）：存活 26，UCB95 0.01734——**比只上裁定 1/2/3 更差**。
5. **实施者 B 预告的反例被探针证实了。** 点名的两个种子 2133002 与 2133047，
   在 M2-SET 里都活着，加上 `stop-v1` 后**都死了**，而且
   `M2-SET+stop` 与 `M2-FULL` 在这两个种子上**逐字段相同**——
   死因归到 `stop-v1` 这一半，不是 coach-v05。
6. **裁定 4b（停滞锁）在被测世界里几乎不说话**：48 局里只压掉 9~13 扇窗，
   而“因为 FARM 已被掩掉所以放行”的 `forced_through` 有 **70~72** 次。
   以 stall 收窗的 DIVE 窗数（控制 99）在 coach 臂是 **102**，**没有下降**。
7. **诊断口径必须校正**：任务书说种子 2133047 有「12 扇 const_dive 窗、全部以
   stall 收窗」。在**认证控制行**上我数到的是 **11 扇 DIVE 窗、其中 5 扇 const_dive、
   以 stall 收窗的 0 扇**（逐窗表见 §11.2）。全 48 局控制行上
   `const_dive ∩ stall` 只有 **6 扇**（const_dive 267 扇、stall 99 扇）。
   **裁定 4b 是照着一个探针行本身不支持的画像设计的。**

---

## 一、合并树是怎么造的

```
m2-merge  = rsync -a --exclude __pycache__  merge-tree/           (M1 认证探针产物)
          + A 的 7 个独占文件（直接取 m2-tree）
          + B 的 3 个独占文件（直接取 stop-tree，含新法模块与新测试卷）
          + 6 个双方都改的文件：git merge-file 三方合并（base = merge-tree）
```

| 归属 | 文件 |
|---|---|
| **仅 A** | `python/diablogym/env.py`、`resource_sustain.py`、`resource_sweep.py`、`tests/test_r18b3b_loot_warm_start.py`、`tests/test_r19_m2_rulings.py`（新）、`tests/test_resource_readiness_law.py`、`train/migrate_loot_candidate.py` |
| **仅 B** | `python/diablogym/resource_emergency_stop.py`（新）、`tests/test_r19_gold_grab.py`、`tests/test_r19_m2_emergency_stop.py`（新） |
| **双方** | `options_env.py`、`resource_protocol.py`、`worker_env.py`、`train/eval_contract.py`、`train/runs/r10-staging/probe_r17_deployment.py`、`train/train_ppo.py` |

### 1.1 冲突只有一处

`git merge-file` 在 6 个文件里只报了 **1 处冲突**，在
`OptionsEnv._reset_services` 的“每局新建服务”那一串构造器的**同一个尾部**：
A 在那里建 coach-v05 的停滞簿记，B 在那里建紧急停止服务。两段互不读对方的字段，
**判为纯加性，两段都要，A 先 B 后**（顺序按运行期被问的先后：教练在选项边界
`resource_option_choice` 被问，停止法在窗内每个工人拍的尾部被问）。
决议脚本：`/home/laure/r17_work/r19/m2/resolve.py`（把这句理由写进了代码注释）。

其余 5 个文件自动合并干净——但**自动干净不等于两边都落了地**，所以：

### 1.2 并集校验（不是“看起来对”）

`/home/laure/r17_work/r19/m2/verify_merge.py`：对 6 个双改文件，各自
以 M1 合并树为基线算出 A 的增删行多重集、B 的增删行多重集与**合并结果**的增删行多重集，
要求

```
added(merged)   == added(A) | added(B)
removed(merged) == removed(A) | removed(B)
```

**12 项（6 文件 × 增/删）全部相等，MISMATCHES: 0**：

| 文件 | 新增行 | 删除行 |
|---|---|---|
| `options_env.py` | 218 | 4 |
| `resource_protocol.py` | 299 | 9 |
| `worker_env.py` | 18 | 3 |
| `train/eval_contract.py` | 41 | 5 |
| `probe_r17_deployment.py` | 26 | 1 |
| `train/train_ppo.py` | 25 | 10 |

`_R16_ENV_KEYS` 的尾部两轮都动过（B 追加 `resource_emergency_stop`），合并后是并集，
且 B 已经把那条钉右括号的旧断言改成直接问成员关系（`test_r19_gold_grab.py`）。

### 1.3 认证法条逐条未被碰（不是 grep，是把法条行读出来比）

`/home/laure/r17_work/r19/m2/lawdump.py` 在**每棵树各一个进程**里 `import` 部署侧模块，
把 `SWEEP_LAWS` 的两行、`RetreatPolicy()` 的每个字段、两个模块的全部大写常量、
coach 词汇表、`resource_sustain` 的全部大写常量 dataclass-展开成 JSON。
M1 合并树 vs M2 合并树的 `diff`，**只有 `>` 行（新增），只有两处 `<` 行**——
一处是原本为空的 `coach_v03_constants: {}`，一处是词汇表元组的最后一个元素行
（`"coach-v03"` 后面多了逗号）。**没有任何既有数值被改动**：

- `sweep-v1` 行只多出一个 `max_targets_per_floor: null` 字段——**按构造惰性**；
- `sweep-v2` 行 `max_targets_per_floor: 24`；
- `RetreatPolicy`（`hp_fraction 0.5` / `drink_hp_fraction 0.4` /
  `empty_belt_hp_fraction 0.75` / `pressure_radius 6` / `pressure_count 0`）
  **一个字段都没进 diff**；
- 采购模式 `"full"`：新分支全在 `v2_potions` / `v2_armor` 两个阶段里，
  `"full"` 在 `minimum_potions` 就返回；探针 OFF 臂的逐位相同是它的实证。

---

## 二、测试：两位实施者邻域的**并集**

命令：`pytest -p no:randomly -q <21 个文件>`，`DIABLOGYM_ROOT=<树>`，
对照是**同一条 rsync 配方做出的净合并树副本** `/home/laure/r17_work/r19/m2a-pristine`
（`diff -rq` 复核 = `PRISTINE_IS_MERGE_TREE`）。

| 树 | 结果 |
|---|---|
| 净对照（19 个共有文件） | 12 failed / **918 passed** / 20 skipped / 726 subtests |
| **m2-merge**（19 共有 + 2 个新卷） | 12 failed / **1060 passed** / 20 skipped / 740 subtests |

> **复核轮更正（2026-09-09 晚，见 §十六.3）**：本节原来只比对了**一半**的失败。
> 两边都报 `12 failed`，但 `t-{pristine,merged}-fails.txt` 各只有 6 个 id——
> 另外 6 条是 pytest-subtests 打成 `SUBFAILED(key=...) <id>`，被
> `grep -E "^(FAILED|ERROR) "` 这个锚漏掉了，而它们属于**另外两个从未出现在本节
> 列表里的 test id**。改用 `^(FAILED|ERROR|SUBFAILED)` 并把 subtest key 归一化后
> 重跑：**两边都是 12 条失败行、8 个互不相同的 test id，逐条完全相同**，
> 新增失败仍然为 0。正确的失败面见 §十六.3。

**新增失败 0，修复 0**（`comm` 双向皆空）。多出的 **142 passed** 正好是两个新卷
（A 的 52 + B 的 90）。两边共同的失败 id 全是**树形**——工作树按配方不带
`train/runs/...model_candidate.zip` 与 `train/models/v22-h-manager/policy.npz`：

```
tests/test_r13_live_dive.py::test_eval_registration_param_fail_closed
tests/test_r18b6_training_wiring.py::MakeEnvTests::test_each_law_still_needs_a_worker_or_options_env
tests/test_r18b6_training_wiring.py::MakeEnvTests::test_smith_v1_can_never_run_outside_the_full_purchase_mode
tests/test_r18b6_training_wiring.py::MakeEnvTests::test_the_factory_forwards_each_law_and_omits_the_defaults
tests/test_r18b6_training_wiring.py::MakeEnvTests::test_the_factory_forwards_the_whole_b6_world
tests/test_r18b6_training_wiring.py::TrainingCliTests::test_validate_args_accepts_the_whole_b6_world
```

日志：`/home/laure/r17_work/r19/m2/t-{pristine,merged}.log`，
失败集合 `t-{pristine,merged}-fails.txt`。**全量 `tests/` 仍未跑**（继承缺口）。

---

## 三、探针设计

- worker：`7e31dc54` = `/home/laure/AlphaDiablo/diablogym/train/runs/r16-arm-a-constitution/model_candidate.zip`
- 种子：**只有** 2_133 池 2133000–2133047，三分片 a/b/c（16 + 16 + 16），
  `max_steps 6000`，`decoding sample`
- 驱动：`/home/laure/r17_work/r19/m2/m2_probe_driver.py`
  （`arm_stats` / `paired` **逐字**来自
  `/home/laure/r17_work/r18/m2_probe_driver.py`，经 M1 的
  `m1_probe_driver_rr.py` 传下来；`m2_stats` 是本轮新增的**只增不改**的一半）
- 收入 / 支出 / 首降那几行来自
  `/home/laure/r17_work/r19/m2/agg2.py`，它把 M1 的
  `m1rr/agg_rr.py::stats` 用 `ast` **原样取出来执行**（不是重打一遍），
  并**断言**它在 M1 ALL-ON 行上复现 M1-REPORT §11.3 印的 12 个数字，
  否则脚本直接停——所以本报告与 M1 报告是同一把尺。
- 跑前跑后对 `python/ train/ tests/` 取 sha256 指纹：**运行期间树未变**
  （`m2/tree-fingerprint-{before,after}.txt`）。

### 3.1 七条臂

| 臂 | 覆盖键 | `rows_sha_v3` | runtime err |
|---|---|---|---|
| 认证控制行（继承） | 16 键 | `0e5a1acd…be886` | — |
| **M2 OFF** | 16 键，一字未改 | **`0e5a1acd…be886`** ✅ | 0 |
| M1 ALL-ON（继承） | +full-v2 +sweep-v2 +gold-grab-v1 | `579816824729d33b…` | — |
| **M2-SET** | 同上（**代码**是裁定 1/2/3 之后的） | `f33f7af5f236f1fc273293160910b5606f7d03f8a08e91c499ed1a45ded75da4` | 0 |
| **M2-SET+coach**（归因） | M2-SET + `resource_readiness_law=coach-v05` | `27b722fb8d8a769a225ac375ad76e395e96b80719e5ee0621675ec41d3fb1ada` | 0 |
| **M2-SET+stop**（归因） | M2-SET + `resource_emergency_stop=stop-v1` | `9842d5f809f086f3dfc9516dd75b63919c35f2b89bba2b85c6a4ba976b4cb4e8` | 0 |
| **M2-FULL** | M2-SET + coach-v05 + stop-v1 | `68f41f5ce5bf230319829edae2004393005835ef278d1d6e92935b084922b4ab` | 0 |

> **归因臂是我加的，任务书没点名。** 理由：M2-FULL 一次开两条法，
> 一旦它比 M2-SET 差，报告只能写“不知道是哪一半”。两条 48 种子臂
> （6 个分片，墙钟约 4 分钟）把这句话换成了一个数。种子池未扩，
> 仍然只有 2133000–2133047。

### 3.2 回归

**M2 OFF 的 `rows_sha_v3` 与认证控制行逐位相同**，配对 saved 0 / lost 0 /
discordant 0。也就是说：ruling 1 的按层上限（`sweep-v1` 行为 `None`）、
ruling 3 的排序（只活在 `v2_*` 阶段）、coach-v05（coach-v03 下对象为 `None`）、
stop-v1（旗为 `off` 时服务为 `None`）**四条新路径在控制世界里逐位不说话**。

---

## 四、主表（同 48 种子、同 worker、同控制行）

| 指标 | control / OFF | M1 ALL-ON | **M2-SET** | +coach | +stop | M2-FULL |
|---|---|---|---|---|---|---|
| **存活 /48** | 20 | 24 | **28** | 25 | **28** | 26 |
| 到达 L2 | 34 | 33 | **30** | 30 | 30 | 30 |
| 到达 L3 | 4 | 5 | **2** | 2 | 2 | 2 |
| L2 死亡 | 19 | 14 | **12** | 14 | **12** | 13 |
| L2 beats | 85578 | 89015 | 87358 | 78026 | **94473** | 86267 |
| **L2 hazard /1k** | 0.2220 | 0.1573 | **0.1374** | 0.1794 | **0.1270** | 0.1507 |
| L2 kills | 894 | 835 | 826 | 773 | 813 | 766 |
| clvl ≥ 4 | 25 | 24 | **18** | 16 | 23 | 21 |
| clvl 中位 | 4.0 | 3.5 | 3.0 | 3.0 | 3.0 | 3.0 |
| deaths_by_floor | {1:7, 2:19, 3:2} | {1:8, 2:14, 3:2} | **{1:8, 2:12}** | {1:8, 2:14, 3:1} | {1:8, 2:12} | {1:8, 2:13, 3:1} |
| 微步合计 | 537999 | 578079 | 522130 | 512179 | 549965 | 542031 |

配对 McNemar（对**认证控制行**，公式逐字取自 R18-M2 驱动）：

| 臂 | saved | lost | net | discordant | **UCB95（单侧）** |
|---|---|---|---|---|---|
| M2 OFF | 0 | 0 | 0 | 0 | 0.00000 |
| M1 ALL-ON | 12 | 8 | +4 | 20 | 0.06865 |
| **M2-SET** | 14 | 6 | **+8** | 20 | **−0.01860** |
| M2-SET+coach | 11 | 6 | +5 | 17 | 0.03495 |
| **M2-SET+stop** | 15 | 7 | **+8** | 22 | **−0.01087** |
| M2-FULL | 12 | 6 | +6 | 18 | 0.01734 |

配对（对 **M1 ALL-ON**，即“裁定 1/2/3 到底带来了什么”）：

| 臂 | saved | lost | net | discordant | UCB95 |
|---|---|---|---|---|---|
| **M2-SET** | 8 | 4 | **+4** | 12 | 0.03372 |
| M2-SET+coach | 7 | 6 | +1 | 13 | 0.10263 |
| M2-SET+stop | 10 | 6 | +4 | 16 | 0.05231 |
| M2-FULL | 10 | 8 | +2 | 18 | 0.10340 |

配对（对 **M2-SET**，即裁定 4 两条各自的净效果）：

| 臂 | saved | lost | net | discordant | UCB95 |
|---|---|---|---|---|---|
| M2-SET + coach-v05 | **0** | **3** | **−3** | 3 | 0.11997 |
| M2-SET + stop-v1 | 5 | 5 | **0** | 10 | 0.10837 |

**读法（照实）**：M2-SET 与 M2-SET+stop 是仅有的两条 UCB95 < 0 的臂，
**第一次可以在单侧 95% 下说“不比控制行差”**；但 UCB95 只有 −0.019 / −0.011，
离“证明更好”还很远，48 个种子上 discordant 只有 20 / 22。
**这仍然是一次有希望的读数，不是一次认证。**

---

## 五、深度的代价（必须让主席先看见的一条）

三条臂（M2-SET 及其两条派生）**同时**出现：

- 到达 L2：34 → **30**（少 4 局根本没下过楼）
- 到达 L3：4 → **2**
- clvl ≥ 4：25 → **18**（+stop 回到 23）
- L2 kills：894 → 826

而首降**更早**（beat 中位 7280 → 6798）、**更壮**（AC 中位 10 → 17，belt 4 → 6）。
所以不是“下楼门槛提高了”，而是**下楼的局数变少、下去以后走得更浅**。
最可能的机制在 §8：按层上限把 sweep 请到了 L2，L2 上多出 6.6k 微步的扫箱与
345 个目标，这些微步是从“继续往下走”里拿的。

**这条我不替主席判。** 战役的目标是 55 层深度；本轮用深度换来的存活是
+8 / 48。要不要，是裁定。

---

## 六、裁定 1 验收：按层目标上限（24 + 24）

**做到了，而且是本轮效果最大的一条。**

| | control/OFF (sweep-v1) | M1 ALL-ON | **M2-SET** |
|---|---|---|---|
| L1 窗 | 104 | 1012 | 560 |
| **L2 窗** | 0 | **15** | **192** |
| L1 目标 | 541 | 1942 | 1067 |
| **L2 目标** | 0 | **33** | **345** |
| L2 chests / barrels / sarcophagi | 0 / 0 / 0 | 9 / 8 / 6 | **87 / 79 / 81** |
| L2 微步 | 0 | 462 | **6618** |
| L2 占全部 sweep 窗 | 0% | **1.4%** | **25.5%** |

否决直方图从 `target_cap` 换档到 `floor_target_cap`：

- M1 ALL-ON：`{target_cap: 38143, cooldown: 12255, monster_near_pair: 4825, …}`
- **M2-SET**：`{floor_target_cap: 33737, cooldown: 8685, monster_near_pair: 5658, target_cap: 2919, monster_near_pair_unseen: 1127, low_hp: 404, monster_near_target: 210, danger_law: 147}`

法条自己的账：`max_targets_per_floor = 24`，
`targets_by_floor {1: 1067, 2: 345}`，
`targets_remaining_by_floor {1: 85, 2: 807}`，
`floor_target_cap_denials 33737`，**41 / 48 局撞上了按层上限**。

读法：**L1 的名额基本被用光**（48 × 24 = 1152 的上限，用掉 1067，只剩 85），
**L2 的名额远远没用完**（用掉 345，剩 807）。也就是说这一刀之后，
**卡住 L2 的不再是名额，而是别的东西**（`cooldown` 8685、`monster_near_pair` 5658、
以及根本没到 L2 的那 18 局）。下一刀若还想加深 L2 的扫箱，**不应该再动这个上限**。

**微步预算按 A 的判断没有拆**（`microstep_budget_split_by_floor: False`，
遥测里逐臂可读）。本轮数据支持这个判断：否决直方图里**没有 `episode_budget` 这一档**
（OFF 臂 sweep-v1 有 104 次，那是 v1 的窄预算），M2-SET 全 48 局 sweep 微步
25363，对 2400/局 的预算连零头都没用到。

---

## 七、裁定 2：`require_trip_slot=False` 已批准，本轮无代码改动

按裁定，代码零改动；产物是法条行注释（`resource_sweep.py:243-252`，写明
“R19-M2 (2026-09-09) CHAIRMAN RULING 2: RATIFIED”）与 4 条回归测试
（v2 行仍 `False`、v1 行仍 `True`、v2 在 `loot_trip_slots_left=0` 下仍开窗、
v1 在同条件下仍记 `no_trip_slot`）。这 4 条在本轮并集里通过。

本轮**新量到的代价**（M1 只量到 L2 一侧的 3 件）：

| 孤儿掉落（非金币，最后一帧仍在地上 / 本 sweep 造成） | M1 ALL-ON | **M2-SET** |
|---|---|---|
| L1 | 115 / 203 | 60 / 113 |
| **L2** | 3 / 3 | **35 / 41** |

L2 上 41 件里有 **35 件**没人捡——因为 L2 的扫箱窗从 15 扇涨到 192 扇，
而 `require_trip_slot=False` 意味着开窗时不问“这一趟还有没有背包位”。
**这是主席已经批准的代价，本轮把它的量级从 3 更新到 35。** 一行可回退。
（金币那一半另记，不与之相加：L2 `gold {dropped 68, still_on_floor 21}`。）

---

## 八、裁定 3 验收：按“原生投影收益 / 金币”排序

**排序确实在跑，缺陷部分修好，但没有修到主席点名的那个数。**

| | control/OFF | M1 ALL-ON | **M2-SET** | +stop |
|---|---|---|---|---|
| 武器升级次数 | **29** | 8 | **12** | 16 |
| 武器支出（升级法账） | 6620 | 1780 | 2520 | 3390 |
| 药水支出（full-v2 账） | 0 | 8900 | **3600** | 3450 |
| 护甲支出（full-v2 账） | 0 | 7770 | **9055** | 8480 |
| 总支出 | 15442 | 25486 | 23048 | 23012 |
| 首降 AC 中位 | 10.0 | 15 | **17.0** | 17.0 |
| 首降 belt 中位 | 4.0 | 8 | 6.0 | 6.0 |

排序自己的账（M2-SET，48 局 97 趟，其中 85 趟至少排了一次序）：

- 决策 **235 次**（`v2_potions` 129 / `v2_armor` 106）
- **赢家：护甲 157 / 药水 77 / 武器 1**
- 药水腿停手理由：`outranked 52` / `belt_full 31` / `unaffordable 9`
- 护甲腿停手理由：`no_upgrade 53` / `over_budget 38` / **`weapon_outranks 1`**
- 武器赢下时留在钱包里的钱：**1 趟、215 金**（中位 215）
- 各腿的原生比值均值：**护甲 0.07077 / 药水 0.02000 / 武器 0.01070**

**必须说清楚的一条：这个排序被“单位”支配。** 护甲的原生单位是 AC 点、
武器是最大伤害点、药水是腰带格；按这三把尺，**每金币的护甲收益是武器的 6.6 倍**，
所以武器腿 90 次候选里只赢了 1 次。主席点名的缺陷是
「正常钱包下武器升级 29 → 4」；本轮把它从 8 抬到 **12**（+50%），
**没有抬回 29**，而且抬起来的那部分主要来自“护甲买不起时停手”与
“药水被压过时停手”，不是来自武器赢下排序。

**给主席的下一刀（不用再跑一条臂）**：法条同时记了
`ratio_normalised`（每条腿各除以**战备法自己**对该腿的要求），**只记不排**。

> **复核轮更正（2026-09-09 晚，见 §十六.5）**：本节初版的这张表是**一个 bug 的
> 产物**。当时护甲除以 `_READINESS_V2_AC[depth]`（首降时 9）、武器除以
> `_READINESS_V2_DMG[depth]`（6），**药水腿却什么都没除**——
> 213 次药水候选的 `ratio_normalised` 与 `ratio` **逐比特相同**。
> 于是“归一化后药水赢 183”只是“两条腿被缩小 6~9 倍、第三条没缩”的算术。
> 药水腿现在除以它自己的战备要求 `required_belt_heals`（本轮实测恒为 4），
> 在**同一批 235 次决策**上重排，结论**反过来**：

| 排序键 | 护甲赢 | 药水赢 | 武器赢 |
|---|---|---|---|
| **原生比值（现行法）** | **157** | 77 | **1** |
| 归一化比值（修好后，只记录、未使用） | **131** | 102 | 2 |
| ~~归一化比值（修好前，药水腿未归一）~~ | ~~50~~ | ~~183~~ | ~~2~~ |

各腿归一化后的比值均值（复核轮实测，M2-SET 臂）：
**护甲 0.010110 / 药水 0.005000 / 武器 0.002141**，
对应的分母分别是 **7.0 / 4.0 / 5.0**（= `ratio` ÷ `ratio_normalised`）。

**修正后的结论**：换尺子**并不会**把钱搬到药水——护甲仍然赢（131/235）；
武器仍然几乎不赢（1 → 2 次）。**两把尺子都指向同一件事：
比值排序买不回武器。** 要真的把武器买回来，需要一条**与比值无关的规则**
（例如“每趟至少给武器腿留一次机会”或“伤害轴单独设下限”），
那是一次新的裁定，我没有自己加。

**整合判断（A 提出、我复核并沿用）**：武器的**购买**仍只由认证过的 smith-v1
腿执行；spend-v2 只排序、不搬购买，武器赢下时护甲腿停手把钱留在钱包里，
几拍后 smith-v1 在**同一趟**花掉。本轮实证：这条路径确实走到了
（`weapon_outranks 1` 次，留下 215 金）。**样本量 1，请主席知悉。**

---

## 九、裁定 4 的两条：归因

**这是本轮我加的一步，也是本轮最重要的一张表。**

| | M2-SET | +coach-v05 | +stop-v1 | 两条同开 |
|---|---|---|---|---|
| 存活 | **28** | 25 | **28** | 26 |
| L2 hazard /1k | 0.1374 | 0.1794 | **0.1270** | 0.1507 |
| L2 beats | 87358 | 78026 | **94473** | 86267 |
| clvl ≥ 4 | 18 | 16 | **23** | 21 |
| 配对 vs M2-SET | — | **saved 0 / lost 3** | saved 5 / lost 5 | — |
| 配对 vs 控制行 UCB95 | −0.0186 | 0.03495 | **−0.01087** | 0.01734 |

**coach-v05：只赔不赚。** 三个种子被它弄死，一个都没救回来，
L2 hazard 反而从 0.1374 涨到 0.1794，L2 beats 掉 10.7%。

**stop-v1：换人不换数。** 净存活 0（救 5 赔 5），但
**L2 hazard 是全场最低的 0.1270**，L2 beats **+8.1%**（在 L2 上待得更久
还死得更少），clvl≥4 从 18 回到 23。它把“在 L2 上活着的时间”买回来了，
只是没把“最后活下来的局数”买回来。

---

## 十、裁定 4 · stop-v1 的自己的账

M2-SET+stop（48 局；M2-FULL 的数字在括号里）：

- 工人拍 **76913**，其中在范围内（主线 L1/L2、工人自有窗）**76706**（99.7%）
- **激活 50 次**（52），分布 `{L1: 42, L2: 8}`（`{42, 10}`），
  **25 / 48 局至少激活过一次**（26）
- 触发条：`crowd 34` / `adjacent 16`（`36 / 16`）
- 否决：`cooldown 234`（242）；**`activation_cap` 一次都没有**——6 次/局的上限没人撞到
- 中止：`door_on_route 3`，其余四档（`no_tile` / `no_route` / `off_grid_step` /
  `protected_tile`）**全为 0**——**冻结规划器在真引擎里选得出格子、走得到**
- 喝药：`drinks_attempted 14 / executed 14`（15 / 15），**0 次失败**
- 脱离：**508 步**（532），`distance_gain_total` **50**（50）
  → **平均每次激活只把与最近怪的最小距离拉开 1.0 格**（0.96）
- 30 拍随访：**alive 33 / dead 17 / unsettled 0**（34 / 18 / 0）
- `unseen_crowd_beats 198`（208）——冻结的“不问可见”口径，只记不闸

**最刺眼的两个数**：**508 步换 50 格**，以及**随访 34% 死亡**。
载具确实会动、会喝药、不会卡住（中止几乎全 0），但它**几乎没有真的脱离**：
12 拍上限、半径 6 的方阵、以及“必须严格增大最小距离”的细则合起来，
在真地图上平均只买到一格。这解释了为什么它救 5 个又赔 5 个。

> **复核轮补数（2026-09-09 晚，见 §十六.4）——本节漏掉了最该报的那一栏。**
> 50 次激活各自是**怎么结束**的，本节一个字都没说。把 `windows[].outcome`
> 数出来：**`steps_spent 33` / `arrived 9` / `closed:death 5` / `door_on_route 3`**。
> 也就是说 **5 次激活是在法条握着身体的时候把人走死的**，
> 而 33 次把 12 拍预算走完也没走到目的地。再加三个本节没有的数：
> **49 / 50 次激活时最近的怪已经贴身**（`min_distance0 == 1`）、
> **36 / 50 次腰带是空的**（“先喝药”那条腿是空操作）、
> **13 / 50 次血量已经 ≤ 20% 上限**（最低 2/78）。
> 主席被要求在不知道“5 次死在载具里”的情况下裁“先修载具、再谈阈值”。

### 10.1 点名的三个种子

| 种子 | control / OFF | M1 ALL-ON | M2-SET | +coach | **+stop** | M2-FULL |
|---|---|---|---|---|---|---|
| **2133002** | 死 L1@559（clvl 1, AC 7, 21 kills） | 活，depth 1 | **活**，depth 1（clvl 3, 132 kills, 12000 微步） | 活（同 M2-SET） | **死 L1@2208**（clvl 2, 62 kills；停止法激活 4 次，喝药 1，走 43 步，随访 3 活 1 死） | **死**，与 +stop 逐字段相同 |
| **2133010** | 死 L2@12001（clvl 4, AC 7） | 死 L2@12321（AC 20） | **活**，depth 2（clvl 4, AC 20, 171 kills, 17264 微步） | 活 | 活（停止法 0 次激活） | 活 |
| **2133047** | **活**，depth **3**（AC 8, 144 kills） | 活，depth 2 | 活，depth 2（150 kills, AC 12） | 活 | **死**，depth 2（127 kills；停止法激活 2 次，**随访 2 次全死**） | **死**，与 +stop 逐字段相同 |

**归因是干净的**：这两个反例在 `+coach` 臂里与 M2-SET **一模一样**，
在 `+stop` 臂里与 M2-FULL **一模一样**。**杀死它们的是 stop-v1 这一半。**

这与实施者 B 在 `M2B-NOTES.md` §3 里提前写下的那句话对上了：
把这套阈值放回 2133002 的认证轨迹上重放，A 条最早第 78 拍才可能开口，
而主席读的是第 58 拍。B 没有擅自改主席的数字，而是把测量摆上桌——
**现在探针给了它一个结果：按裁定的数字，这部法不但救不了 2133002，
在合并世界里还把它从“活”变成了“死”。**

---

## 十一、裁定 4 · coach-v05 的自己的账，以及一处诊断口径的更正

### 11.1 法条几乎没说话

M2-SET+coach（48 局；M2-FULL 括号内）：

- 压掉的窗 **9**（13），分布 `{L2: 5, L3: 4}`（`{L2: 9, L3: 4}`）——**L1 上一次都没有**
- 压制理由：`const_dive_unready 8` / `stall_lock 1`（`10 / 3`）
- **`forced_through` 72（70）**，分布 `{L1: 71, L2: 1}`（`{L1: 66, L2: 4}`）
- 见到的 stall 窗 `stall_windows_seen` 102（100），`{L1: 90, L2: 12}`
- 停滞锁激活 13（12），`{L1: 11, L2: 2}`；**7 / 48 局在收尾时仍锁着**
- ~~解锁 `{farm_levelup 85, town_trip 91, descended 74}`~~

> **复核轮更正（2026-09-09 晚，见 §十六.6）：上面这一行是假的。**
> `_unlock` 把计数器写在 `if floor in self._locked:` 这道闸的**外面**，
> 进城那一支更是不管有没有东西被锁都加一。所以 `{85, 91, 74}` 是**调用次数**，
> 不是解锁次数。真正的解锁要看 `lock_events`：
> **`locked` 13 次、`unlocked` 6 次、而且 6 次全是 `by: "descended"`**——
> 也就是说**唯一真正解开过锁的，正是这条锁想拦住的那次强制下潜**；
> `town_trip` / `farm_levelup` / `farm_scene` / `farm_cleared` **一次都没解开过**，
> 另外 7 局到收尾还锁着。读到 `{85, 91, 74}` 的人会以为这把锁自清自洁，
> 事实完全相反。计数器已分成 `unlocks`（真解锁）与 `unlock_calls`（调用），
> 复核轮实测见 §十六.6。

**机制读法**：4.3 那道“永不与冻结掩码相争”的护栏在 L1 上**几乎总是生效**——
L1 上 71 次本该拒绝的场合，FARM 已经被掩掉，于是放行并记 `forced_through`。
所以 **coach-v05 在 L1 上按构造基本是惰性的**，它真正说话的地方是 L2 与 L3，
一共 9 扇窗。**9 扇窗赔掉 3 个存活。**

### 11.2 acceptance：以 stall 收窗的 DIVE 窗，没有下降

| 臂 | DIVE 窗 | 以 stall 收窗 | const_dive 窗 | **const_dive ∩ stall** |
|---|---|---|---|---|
| control / OFF | 710 | **99** | 267 | **6** |
| M1 ALL-ON | 800 | 87 | 208 | 1 |
| M2-SET | 763 | 108 | 193 | 1 |
| **M2-SET+coach** | 743 | **102** | 186 | 1 |
| M2-SET+stop | 785 | 112 | 200 | 1 |
| M2-FULL | 763 | 100 | 191 | 1 |

裁定 4b 要打断的那类窗，在 coach 臂里从 108 只降到 102。**这一刀基本没切到东西。**

### 11.3 诊断口径的更正（必须写下来）

任务书给的画像是：种子 2133047「400 拍绕圈、12 扇 const_dive DIVE 窗、
readiness 0.91、全部以 stall 收窗」。我在**认证控制行**上把那一局的 DIVE 窗
逐扇打出来（脚本 `/home/laure/r17_work/r19/m2/extra.py`）：

```
beat0=5909  tau=63   coach_ready  end=sweep_trigger   ratio_v2=1.1111
beat0=6060  tau=68   coach_ready  end=sweep_trigger   ratio_v2=1.1111
beat0=6191  tau=611  coach_ready  end=cap             ratio_v2=1.1111
beat0=6802  tau=182  coach_ready  end=descend         ratio_v2=1.1111  (descended)
beat0=6984  tau=268  const_dive   end=retreat_trigger ratio_v2=0.9091
beat0=9151  tau=211  coach_ready  end=scene           ratio_v2=1.1111  (descended)
beat0=9362  tau=381  const_dive   end=retreat_trigger ratio_v2=0.9091
beat0=11499 tau=245  coach_ready  end=scene           ratio_v2=1.1111  (descended)
beat0=11744 tau=442  const_dive   end=fuse            ratio_v2=0.9091
beat0=12186 tau=61   const_dive   end=descend         ratio_v2=0.9091  (descended)
beat0=12247 tau=601  const_dive   end=cap             ratio_v2=0.75
```

**11 扇窗、5 扇 const_dive、以 stall 收窗的 0 扇。**
`ratio_v2` 确实是 0.91（与“readiness 0.91”对得上），但收窗理由是
`retreat_trigger` / `fuse` / `cap` / `descend`，**不是 `stall`**。

全 48 局控制行上：DIVE 窗 710 扇，const_dive **267** 扇，以 stall 收窗 **99** 扇，
**两者的交集只有 6 扇**。而 const_dive 窗的 `ratio_v2` 中位是 **0.500**、
**100% 低于 1.0**——所以规则 (a) 的门槛在**每一扇** const_dive 窗上都成立，
它之所以只压掉 8~10 扇，全是那道掩码护栏挡下来的。

**结论**：裁定 4b（两扇连续 stall 且未下楼 → 锁本层）**照着一个探针行本身
不支持的画像设计**；探针里真正存在的是「const_dive 窗数量大（267/710）、
readiness 中位 0.5、收窗理由五花八门」这件事。要治它，规则 (a) 才是对的方向，
但它现在被“不与掩码相争”这条护栏在 L1 上整个架空了。

> **复核轮补数（2026-09-09 晚）——“规则 (a) 是对的方向”这句话必须带上一个数。**
> 我把每扇 DIVE 窗按 `(trigger, forced)` 分了类：

| 臂 | const_dive 窗 | 其中 `forced=True`（FARM 已被掩掉） | **`forced=False`（规则 (a) 真能说话的）** |
|---|---|---|---|
| 认证控制行 / OFF | 267 | 198 | **69** |
| **M2-SET（会实际上线的那个世界）** | 193 | 186 | **7** |
| M2-SET+coach | 187 | 186 | **1** |

> **在 M2-SET 的代码下，规则 (a) 每 48 局只有约 7 扇窗可以作用**——
> 这正是它只压掉 8 扇（复核后 4 扇）的原因，也是 §十三.3“只留规则 (a)”这条建议
> 的真实含金量：**除非主席同时重裁那道“永不与冻结掩码相争”的护栏，
> 否则“只留规则 (a)”几乎等于什么都不留。** 那是一次**新的裁定**，不是一次删减。

---

## 十二、收入 / 支出 / 首降（与 M1 §11.3 同一把尺）

| 指标 | control/OFF | M1 ALL-ON | **M2-SET** | +coach | +stop | M2-FULL |
|---|---|---|---|---|---|---|
| 首降局数 | 34 | 33 | 30 | 30 | 30 | 30 |
| 首降 beat 中位 | 7280.0 | 7486 | **6798.0** | 6798.0 | 7235.0 | 7235.0 |
| 首降 belt 中位 | 4.0 | 8 | 6.0 | 6.0 | 6.0 | 6.0 |
| 首降 **AC** 中位 | 10.0 | 15 | **17.0** | 17.0 | 17.0 | 17.0 |
| 首降 damage 中位 | 9.0 | 9 | 8.0 | 8.0 | 7.0 | 7.0 |
| 进城行程 | 105 | 105 | 97 | 97 | 108 | 109 |
| 进城微步 | 150434 | 161004 | 143703 | 142454 | 157810 | 157261 |
| 收入·行程 collect | 12178 | 10841 | 8625 | 8574 | 9673 | 9622 |
| 收入·卖货 | 7591 | 8786 | **9066** | 8700 | 7388 | 7064 |
| 收入·卖货（已鉴定） | 3769 | 4280 | **5326** | 4824 | 3526 | 3024 |
| 收入·**捡金** | 0 | 8229 | **8695** | 8382 | 8765 | 8478 |
| 支出·合计 | 15442 | 25486 | 23048 | 22836 | 23012 | 23101 |
| 支出·修理 | 767 | 1111 | 613 | 616 | 682 | 736 |
| 支出·鉴定 | 1600 | 1900 | 2100 | 2000 | 1600 | 1500 |
| 支出·武器（升级法） | 6620 | 1780 | 2520 | 2520 | 3390 | 3390 |
| 支出·药水（full-v2 账） | 0 | 8900 | 3600 | 3600 | 3450 | 3450 |
| 支出·护甲（full-v2 账） | 0 | 7770 | **9055** | 8940 | 8480 | 8565 |
| 鉴定件数 | 16 | 19 | **21** | 20 | 16 | 15 |
| 撤退次数 | 55 | 63 | **49** | 48 | 61 | 61 |
| gold_final 中位 | 78.0 | 89.0 | **92.0** | 87.0 | 92.0 | 85.0 |

> `sweep gold` 仍**报为空、不报为 0**（继承 M1 §11.3 的口径：没有通道给一枚硬币
> 打上“哪个箱子掉的”标签；`gold_dropped_seen` 数的是**堆数**不是金额）。

时间占比：扫箱 **4.86%**（M1 6.93%）、捡金 **2.44%**（2.12%）、进城 **27.52%**（27.85%）。
按层的捡金账（M2-SET）：`{L1: {windows 402, piles 545, gold 5205},
L2: {windows 169, piles 184, gold 3490}}`——L2 的金比 M1（3064）多 14%。

---

## 十三、读数与建议

1. **裁定 1/2/3 值得留下。** M2-SET 是本战役第一条 UCB95 < 0 的臂，
   L2 hazard 从 0.2220 一路降到 0.1374，配对对 M1 ALL-ON 也是 net +4。
   裁定 1 是其中效果最大的一条（L2 扫箱窗 15 → 192）。
2. **但请主席先裁深度这一刀。** 同一臂 L3 从 4 掉到 2、L2 到达从 34 掉到 30、
   clvl≥4 从 25 掉到 18。如果 55 层是硬目标，那么“用浅层换存活”这条路
   迟早要还，而现在还得越早越便宜。
3. **coach-v05 建议不予采纳（按现在的写法）。** 它 48 局只压 9 扇窗、
   赔掉 3 个存活、把 L2 hazard 抬高 31%，而它要治的病（以 stall 收窗的
   DIVE 窗）**一扇都没少**。若要留，我建议只留规则 (a)、去掉规则 (b)，
   并且**重新裁定那道“不与掩码相争”的护栏**——正是它让规则 (a) 在 L1 上失效。
   > **复核轮限定**：(i)“只留规则 (a)”若不同时重裁那道护栏，可作用面只有
   > **约 7 扇窗 / 48 局**（§11.3 复核补数），基本等于什么都不留；
   > (ii) 复核轮**没有**去掉规则 (b)——那是主席的裁定，不是我的——
   > 而是给了它一条**在实测世界里真的会触发**的释放路径（§十六.6）；
   > (iii) 复核后 coach 臂 alive 25 → **26**、hazard 0.1794 → **0.1737**，
   > 结论（“只赔不赚”）**没有改变**，配对对 M2-SET 仍是 saved 0 / lost 2。
4. **stop-v1 是一次真正的分叉，请主席自己选。** 它净存活 0，
   但把 L2 上的暴露时间买回来了（beats +8.1%，hazard 全场最低 0.1270），
   clvl≥4 从 18 回到 23。它的机制账**很健康**（中止几乎全 0、喝药 100% 成功、
   上限没撞到），坏在**脱离几乎无效**：508 步只换 50 格。
   我的建议是：**先修载具，再谈阈值**——把 `disengage_microsteps` 12
   与“必须严格增大最小距离”这两条一起放宽，或者按 B 在 §3 里量出的
   「相邻 ≥ 5、与血量无关」的人群条重跑一条臂。两件都是新的裁定。
   > **复核轮更正**：载具**已经修了**（每一拍重测血量 / 触发条 / 分离度，
   > §十六.4），但方向与本条的建议**相反**——不是放宽，而是收紧。
   > 结果：**载具内死亡 5 → 0**，点名种子 2133002 从「L1 第 2208 拍、clvl 2」
   > 推到「L2 第 14509 拍、clvl 4、AC 25」；**但存活数 28 → 25**，
   > 脱离几乎变成 0（508 步 → 43 步，50 格 → 12 格）。
   > **修好的载具是安全的，也几乎是惰性的。** 见 §十六.4 与 §十六.8。
5. **裁定 3 只修好了一半。** 武器升级 8 → 12，没有回到 29；
   护甲在原生尺下每金币收益是武器的 6.6 倍，武器 90 次候选只赢 1 次。
   归一化尺换来的是**药水压倒护甲**（183/235），仍然不是武器。
   要买回武器需要一条与比值无关的规则。

---

## 十四、已知缺口（不掩盖）

1. **深度回归未被解释到机制层。** 我给出的机制（按层上限把微步请到 L2）
   是**论证**，不是测量：我没有跑“sweep-v2 + 按层上限但 L2 关掉扫箱”这条反事实臂。
2. **归因臂只有一次 48 局，discordant 只有 3（coach）与 10（stop）。**
   “coach-v05 只赔不赚”这句话建立在 **3 个不一致种子**上，
   `UCB95 0.11997` 远大于 0。**不能当认证读。**
3. **stop-v1 的“救 5 赔 5”同理**（discordant 10，UCB95 0.10837）。
   本轮唯一强的说法是那两个**点名种子**的逐字段归因，那是个案，不是分布。
4. **coach-v05 没有“被咨询但放行”的遥测。** 它只数压制与 `forced_through`，
   不数“进了 `dive_verdict` 又返回 None”的次数，所以 §11.1 里
   「186 扇 const_dive 窗只压了 8 扇」的其余部分我只能用 `forced_through`
   与掩码护栏解释，**不能逐扇核对**。下一轮建议加一个 `consulted` 计数器。
5. **训练侧仍然训不了。** 两条新法都只有指纹绑定与部署侧 validator，
   **没有** `--resource-emergency-stop` / `--resource-readiness-law=coach-v05`
   的 `make_env` 转发（继承 gold-grab-v1 / R19-A1 的同一取舍）。
   本轮是 Python-only + 探针-only，**训练 smoke 未做（not verified）**。
6. **全量 `tests/` 未跑**；M1 记录的 3 个收集失败 + 3 个 Biteq ERROR 是树形问题，
   本轮未触及。并集里那 6 个失败在净对照上逐条同样失败。
7. **`require_trip_slot=False` 的 L2 代价从 3 涨到 35**（§7）。主席已批准，
   但批准时看到的是 3。
8. **`ratio_normalised` 的重排是离线算的**（§8 的表由
   `m2/extra.py` 在同一批 235 次决策上重跑排序键得到），
   **不是一条跑过的臂**。它回答“换尺子谁赢”，不回答“换尺子存活多少”。
9. **`spend_v2` 的 `weapon_outranks` 只出现 1 次**：那条“护甲停手把钱留给
   smith-v1”的缝**只被走过一次**（215 金）。样本量 1。
10. **`_spend_v2_back_edges` 是整局计数、不按趟清零**（A 的 §五.7），
    本轮遥测里没有单独把它拉出来读。
11. **种子池纪律**：只用了 2_133 池 2133000–2133047；
    处女池 2_116-119 / 2_126-128 零接触。但**同一池已经被本战役反复使用**，
    5 条新臂又消费了它一次——过拟合风险在累积，这是一条方法论缺口。
    **复核轮又跑了 5 条（OFF/FULL/STOP 跑了两遍），这条缺口比原来更重。**

**复核轮新增的缺口（2026-09-09 晚）**

12. **修好的载具几乎是惰性的。** stop-v1 现在 44 次激活总共只走 43 步、
    买到 12 格，`arrived` 与 `steps_spent` **各 0 次**，19/44 是第一步就挨打
    （`hp_fell`）。它剩下的作用基本只是**那 15 次喝药**。
    “还身体”这条纪律是对的，但 `abort_on_hp_loss` 是**无参数的最严版本**；
    “掉多少血才算走不通”（例如允许挨 1 拍、或按 max_hp 的百分比）是一次
    **新的裁定**，我没有自己定。
13. **`gain_denominator` 只在排序函数里，没进探针行。** §八 那张表的三条分母
    （7.0 / 4.0 / 5.0）是用 `ratio ÷ ratio_normalised` 反解出来的，不是行里读到的。
    加这个键要为纯遥测重跑四条臂，本轮没做。
14. **coach-v05 的 `farm_windows` 释放路径太稀。** 48 局只触发 1 次，
    因为锁大多在一局尾巴上才扣上（`farm_windows_while_locked` 全场只有 4 扇）。
    **7 / 48 局收尾仍锁着这件事没有解决**，只是不再被假遥测掩盖。
15. **复核轮的臂间差异都不显著。** stop 对 M2-SET 的 discordant 只有 7、
    coach 只有 2、FULL 只有 10；三条 UCB95 全 > 0。
    **本节所有“变好/变坏”都必须当作方向而不是结论读。**
16. **`eval_assembled.py` 的新词表只做了源码级与 `validate_r16_environment`
    级的钉定，没有真的用 `--resource-readiness-law coach-v05` 跑一次评测**
    （评测要走认证档案路径，本轮是 probe-only）。**not verified。**

---

## 十五、产物清单

| 用途 | 路径 |
|---|---|
| 合并树 | `/home/laure/r17_work/r19/m2-merge` |
| 对**主树**的补丁（25 文件） | `/home/laure/r17_work/r19/m2.patch` |
| A / B 的原始差异 | `/home/laure/r17_work/r19/m2/{a,b}.diff` |
| 三方合并结果 | `/home/laure/r17_work/r19/m2/mf/*.py` |
| 冲突决议脚本 | `/home/laure/r17_work/r19/m2/resolve.py` |
| **并集校验**（合并 == A ∪ B） | `/home/laure/r17_work/r19/m2/verify_merge.py` |
| **认证法条行比对** | `/home/laure/r17_work/r19/m2/lawdump.py` + `law-{m1,m2}.json` |
| 测试日志 / 失败集合 | `/home/laure/r17_work/r19/m2/t-{pristine,merged}{.log,-fails.txt}` |
| 探针驱动（3 主臂） | `/home/laure/r17_work/r19/m2/m2_probe_driver.py` |
| 探针驱动（2 归因臂） | `/home/laure/r17_work/r19/m2/attrib.py` |
| 臂汇总 | `/home/laure/r17_work/r19/m2-probe/m2-summary-{all,attrib}.json` |
| 逐臂行 | `/home/laure/r17_work/r19/m2-probe/m2-{off,set,full,coach,stop}-rows.json` |
| 收入/支出/首降（同 M1 的尺） | `/home/laure/r17_work/r19/m2/agg2.py` + `agg2.json` |
| 2133047 逐窗 + 归一化重排 | `/home/laure/r17_work/r19/m2/extra.py` |
| **复核轮的全部产物** | 见 §16.9 |
| 渲染表 | `/home/laure/r17_work/r19/m2/render{,2}.txt` |
| 树指纹（跑前/跑后） | `/home/laure/r17_work/r19/m2/tree-fingerprint-{before,after}.txt` |
| 冒烟（2 种子，FULL 臂） | `/home/laure/r17_work/r19/m2/smoke-full.{json,log}` |
| 账本 | `R19_M2_INTEGRATED`、`R19_M2_PROBE_RESULT` |

---

# 十六、复核轮（Review round，2026-09-09 晚）

一位复核者在 m2-merge 上提了 **4 条 high + 5 条 medium**。全部改掉了，
代码、测试、探针、补丁、本报告都重跑/重写了一遍。本节是那一轮的完整交代，
**包括三处我没有照复核者的建议做的地方，以及一处我做了又退回去的改动**。

## 16.1 一页结论

| | 结果 |
|---|---|
| **OFF 回归（每条新法 OFF）** | `rows_sha_v3` = `0e5a1acd2fb2c07cd27eb72f9ecc574322e72f362ef0376b692376d9a99be886` = 认证控制行，**两次独立重跑都相等** ✅ |
| **M2-SET** | `f33f7af5…`，**与复核前逐比特相同**（配对 saved 0 / lost 0）——H1 与 H4 两条修改**不改变任何行为** |
| **M2-FULL′** | `72a86601…`，alive **22**（复核前 26），hazard **0.1948**（0.1507） |
| **M2-SET+coach′** | `615f1281…`，alive **26**（25），hazard **0.1737**（0.1794） |
| **M2-SET+stop′** | `c79526a8…`，alive **25**（28），hazard **0.1528**（0.1270） |
| 测试并集 | 净对照与 m2-merge **各 12 条失败行 / 8 个 test id，逐条相同**；新增失败 0 |
| 主树 / M1 合并树 | 主树除 ledger 外零改动；M1 合并树零写入；`src/` 与主树逐字节相同；`build` 软链未动 |

**必须先说的一句**：本轮把 stop-v1 的载具修**安全**了（载具内死亡 **5 → 0**），
也把它修**惰**了（脱离 508 步 → 43 步、50 格 → 12 格），
而 **alive 28 → 25**。coach 半边基本不动（25 → 26）。
**M2-SET（裁定 1/2/3）仍然是本战役唯一 UCB95 < 0 的臂，且本轮逐比特未变。**
裁定 4 的两条，在修好缺陷之后，**依然没有一条能赢过 M2-SET**。

## 16.2 九条发现，逐条处置

| # | 级别 | 位置 | 处置 |
|---|---|---|---|
| H1 | high | `train/eval_assembled.py` | **改**：词表从 `diablogym.resource_protocol` **导入**（`choices=RESOURCE_READINESS_LAWS`），两处 `!= "coach-v03"` 闸换成 `not _is_coach(...)`；新增 `EvalAssembledRegistrationTests`（5 个用例，含一条扫遍 `train/*.py` 与 `python/diablogym/*.py` 的邻域检查） |
| H2 | high | `options_env._emergency_stop_tail` | **改**：脚本走路的**每一拍之前**重测三件事（血量 / 触发条 / 分离度），见 16.4。**未照建议加“分离度下限”**，理由见 16.7 |
| H3 | high | `resource_protocol.ReadinessCoachV05` | **改**：加了一条在实测世界里真会触发的释放路径（锁着的层上关掉 2 扇 FARM 窗）＋窗口收窗理由普查，见 16.6。**未照建议“去掉规则 (b)”**，理由见 16.7 |
| H4 | high | `resource_sustain.spend_v2_rank_legs` | **改**：药水腿按它自己的战备要求 `required_belt_heals` 归一；§八 的重排表整张重算，结论反转 |
| M5 | medium | `m2/tests.sh` | **改**：`grep` 加 `SUBFAILED` 并归一化 subtest key；重跑，见 16.3 |
| M6 | medium | `_unlock` 遥测 | **改**：`unlocks` 只数真解锁，`unlock_calls` 数调用；§11.1 已更正 |
| M7 | medium | coach-v05 无层作用域 | **改**：`COACH_V05_FLOORS = (1, 2)`，与 stop-v1 的 `floors` 同一组；L3 上只记录不拒绝 |
| M8 | medium | 报告 §11.1/11.3/13.3 | **改**：186/186 forced 重叠与“可作用面约 7 扇窗”写进 §11.3；§13.3 加限定 |
| M9 | medium | 归因（wage / kills / fuse） | **改**：stop 的 W 增量、击杀与 fuse 单独记账并从 worker 账里扣掉，见 16.5 |

## 16.3 M5：测试并集的失败面（原来只比了一半）

`pytest-subtests` 把它的失败打成 `SUBFAILED(key='...') <id>`，
原来的 `grep -E "^(FAILED|ERROR) "` 抓不到，于是**每边 12 条失败只有 6 条进了比对**，
漏掉的 6 条属于**另外两个 test id**：

```
tests/test_r18b6_training_wiring.py::MakeEnvTests::test_the_clock_and_retreat_rings_shadow_that_gate_in_the_real_fixture
tests/test_r18b6_training_wiring.py::TrainingCliTests::test_validate_args_pins_each_law_to_the_loot_itinerary
```

改锚重跑（脚本 `/home/laure/r17_work/r19/m2rr/rr_tests.sh`）：

| 树 | 结果 | 失败行 | 互不相同的 test id |
|---|---|---|---|
| 净对照 `m2a-pristine`（19 共有文件） | 12 failed / **918 passed** / 20 skipped / 726 subtests | 12 | **8** |
| **m2-merge**（19 共有 + 2 新卷） | 12 failed / **1103 passed** / 20 skipped / 742 subtests | 12 | **8** |

**`NEW FAILURES IN MERGED` 与 `NEW FAILING IDS IN MERGED` 双双为空**，
两边的 12 条失败行**逐条相同**。多出的 **185 passed** = 两个新卷（复核轮又加了
`ReviewRound*`、`EvalAssembledRegistrationTests`、`CoachV05ReviewRoundTests`、
`SpendV2NormalisedRatioTests` 等共 43 个新用例）。
日志 `m2rr/rr-{pristine,merged}.log`，集合 `rr-{pristine,merged}-{fails,ids}.txt`。

## 16.4 H2：载具不再一握十三拍

**缺陷（复核者的实测，我逐条复算过）**：v1 的循环里除了“到没到”与四档规划器拒绝之外
**什么都不重测**。50 次激活中 **49 次**在最近的怪**已经贴身**时开火、
**36 次**腰带是空的（“先喝药”是空操作）、**13 次**血量已 ≤ 20% 上限、
**33 次**把 12 拍走完也没到、**5 次**以 `closed:death` 结束——
**人是在法条握着身体的时候死的**。

**改法**：在**每一次脚本步之前**重测三件事，任一说不就立刻还身体。三个阈值都写在
**法条行**上（所以第二行法条仍然是一条 dict 记录，不是改代码）：

| 法条行新字段 | 值 | 一行理由 |
|---|---|---|
| `abort_on_hp_loss` | `True` | 血量低于**开始走路时**的血量就还身体。无参数。50 次里 5 次死在载具内；用血买来的“脱离”不是脱离 |
| `abort_on_trigger_cleared` | `True` | 触发条不再成立就还身体。载具只回答那一个事实 |
| `no_progress_steps` | **4** | 连续 4 步没有把“与最近可见怪的最小距离”拉大就还身体。实测 508 步只买到 50 格；4 是主席 12 拍预算的**三分之一**——够绕一个弯，也够证明这次走不通 |

**实测（M2-SET+stop，48 局；括号是复核前）**：

| | 复核前 | **复核后** |
|---|---|---|
| 激活 | 50 | **44** |
| 脱离步数 | **508** | **43** |
| `distance_gain_total` | 50 格 | **12 格** |
| 每次激活买到的格子 | 1.0 | **0.273** |
| **结束方式** | `steps_spent 33 / arrived 9 / closed:death 5 / door_on_route 3` | **`trigger_cleared 25 / hp_fell 19`** |
| **载具内死亡** | **5** | **0** |
| 喝药（尝试 / 成功） | 14 / 14 | 15 / 15 |
| 激活时腰带为空 | 36 | 29 |
| 激活时血量 ≤ 20% 上限 | 13 | **3** |
| `min_distance0` 直方图 | `{1: 49, 2: 1}` | `{1: 44}` |
| 30 拍随访 | alive 33 / dead 17 | alive 30 / dead 13 / unsettled 1 |
| 规划器中止（`door_on_route` 等） | 3 | **0** |

**机制读法，一句话**：`hp_fell` 占 19/44——**开始走的第一步就挨打**。
这正是复核者的诊断（“贴身时走开十二拍”）在数据上的样子。现在它走一步就还手，
于是这部法实际退化成「**受伤且被围时喝一口药，再走大约一步**」：
`arrived` 一次都没有，`steps_spent` 一次都没有。**安全，但几乎是惰性的。**

**点名种子 2133002（复核前 → 复核后，同一颗种子、同一条臂）**：

| | 复核前 `+stop` | **复核后 `+stop`** |
|---|---|---|
| 结局 | 死 **L1 @ 2208 拍** | 死 **L2 @ 14509 拍** |
| clvl / AC / max_hp | 2 / 7 / 78 | **4 / 25 / 94** |
| 击杀 | 62 | **161** |

**载具修好之后，这颗种子仍然死，但它是在下一层、6.6 倍时长、2.6 倍击杀之后死的。**
2133047 与复核前基本相同（死 L2 @ 13792 vs 13883，clvl 4，AC 13，belt 0）。

## 16.5 M9：脚本的拍不再算工人的工资

`_emergency_stop_tail()` 原本坐在 `worker_wage` / `worker_kills` 的差分**里面**：
最多 13 拍脚本动作（一次 a12 + 十二个方向键，没有一个是策略选的）被记成
**网络那一次提案**的 PPO 工资与伴随击杀；更糟的是
`fuse = primary if primary.fuse_tripped else (ending if ending.fuse_tripped else None)`
会把**脚本方向键踩掉的保险丝**报成 **worker 的保险丝**——
而 `worker_env` 对 `fuse_tripped` 的现语义是「**拒绝提案**」，`leashed_ppo` 用它筛 BC。

**改法**：在调用前后各取一次 `W` 与 `_ep_kills`，差额记进窗口新增的
`stop_wage` / `stop_kills`，并从 `worker_wage` / `worker_kills` 里**减掉**；
`ending is stop_ending` 且踩了保险丝时记 `stop_fuses` 且**不**算工人的。
本轮探针不训练，所以这条**没有数字上的影响**；它是在
`--resource-emergency-stop` 接进 `make_env`（§十四.5 的缺口）之前必须先堵的
**reward-hacking 面**。新增 `ReviewRoundCreditAssignmentTests`（6 个用例）。

## 16.6 H3 / M6 / M7：coach-v05 的锁、遥测与作用域

**M7 作用域**。coach-v05 原来**没有层作用域**，9 次压制里 **4 次落在 L3**——
既在裁定 4b 的证据范围（L1/L2 的诊断）之外，又直接对战役唯一的北极星（深度）征税。
现在 `COACH_V05_FLOORS = (1, 2)`，与 stop-v1 的 `floors` **同一组数**；
层外只记 `out_of_scope`，不拒绝。实测：`out_of_scope = 2`（全在 L3），
压制从 `{L2: 5, L3: 4}` 变成 **`{L2: 5}`**（`const_dive_unready 4` + `stall_lock 1`）。

**M6 遥测**。`unlocks` 现在只在真的释放了某一层时才加，`unlock_calls` 保留调用数。
实测（coach 臂 48 局）：

```
lock_events   locked 14 / unlocked 7   （unlocked: descended 6, farm_windows 1）
unlocks       {descended: 6, farm_windows: 1}          ← 真解锁 7 次
unlock_calls  {farm_levelup 85, town_trip 91, descended 73, farm_windows 1}  ← 调用 250 次
收尾仍锁着的局   7 / 48
```

**H3 释放路径**。复核者问“为什么 `COACH_V05_FARM_UNLOCK_REASONS` 一次都没触发”。
本轮给出了**测量**而不是猜测——新加的 `window_end_reasons` 普查（coach 臂 48 局）：

```
FARM      sweep_trigger 524, grab_trigger 420, cap 122, fuse 93, levelup 85,
          exhausted 58, retreat_trigger 18, death 11, end 4
DIVE      fuse 213, cap 107, stall 105, sweep_trigger 103, grab_trigger 87,
          scene 41, retreat_trigger 39, descend 32, end 16, death 1
RESUPPLY  sweep_complete 743, grab_complete 565, scene 231,
          resource_complete 91, retreat_complete 48, death 10, end 6
```

三条读数：
1. **FARM 窗在这个世界里 `scene` 与 `descend` 各 0 次**，`levelup` 有 85 次
   但从来没有落在“正锁着的层”上——所以 `("scene","descend","levelup")` 这组
   在实测世界里确实是空的；
2. 复核者要我查的那件事有答案了：**撤退驱动的 RESUPPLY 窗发布的是
   `retreat_complete`（48 次），不是 `resource_complete`（91 次）**。
   我**没有**把 `retreat_complete` 加进进城解锁集：它是撤退载具**交还身体**，
   不是“一趟进城完成了”，把它当成进城会让锁被一件没发生的事解开。**记录，不动手。**
3. 新释放路径 **`farm_windows`**：**锁着的层上关掉 2 扇 FARM 窗**就释放。
   `2` 不是新数，是主席的 `COACH_V05_STALL_WINDOWS` 反过来读
   （两扇窗的证据上锁，两扇窗的证据解锁）。
   实测：**触发了 1 次**；`farm_windows_while_locked` 48 局只有 **4** 扇
   （`exhausted 2 / end 1 / grab_trigger 1`）——
   **锁大多是在一局的尾巴上才扣上的**，此后没有几扇 FARM 窗可关。
   所以这条路径**是活的但很稀**，7 / 48 局收尾仍然锁着这件事**没有解决**。

## 16.7 三处我没有照建议做（必须写下来）

1. **没有给 A 条加“分离度 ≥ 2”的下限。** 复核者的理由是“49/50 次贴身开火，
   这条法从来没有先发制人过”。加下限会**让这条法在主席自己的例子上闭嘴**：
   2133002 第 78 拍是 hp 33/70、九个贴身——分离度 1，而 B 条要 hp ≤ 0.3×70 = 21，
   **两条都不成立**。我改成把人群环从 2 放宽到 3 去争取先发制人的样本，
   跑了 48 局：`min_distance0` 直方图从 `{1: 49, 2: 1}` 只挪到 `{1: 44, 2: 1}`，
   **等于没买到先发制人**。于是**把环退回主席的 2**，
   半径 3 的那条臂作为反事实留档（`m2rr-probe/radius3/`，
   `rows_sha_v3 13e4f69c…`，alive 26 / hazard 0.1385 / clvl≥4 26）。
   **要让这条法真的先发制人，需要另一种机制（大得多的环、更低的计数、
   或者一条“接近速度”的判据），那是一次新的裁定。**
2. **没有去掉规则 (b)。** 复核者给了“或者采纳报告自己的 13.3、删掉规则 (b)”这个选项。
   规则 (b) 是**主席的裁定**，删它不是复核能做的事。我给了它一条会触发的释放路径。
   顺带记一处**我自己写错又退回去的改动**：我最初还加了一条
   “战备恢复即解锁”（`ratio ≥ 1.0 或 cleared` 就释放）。
   规则 (a) 拒绝的条件正好是 `ratio < 1.0 且 not cleared`，
   两者**互为补集**——那条释放会让 `stall_lock` **永远不可达**，
   等于借修理之名删掉规则 (b)。被 `CoachV05StallLockTests` 当场抓到，已退回，
   并在 `resource_protocol.py` 里把这段写成注释留证。
3. **没有把 `gain_denominator` 写进 spend 账页。** 排序函数现在每条腿都带
   `gain_denominator`（有测试钉住），但**探针行里的候选记录没有这个键**——
   加了就得为纯遥测再消费一次种子池重跑四条臂。三条腿的分母可以从
   `ratio ÷ ratio_normalised` 精确还原（本轮 = 7.0 / 4.0 / 5.0），所以**记为缺口**
   （§十四.13），不重跑。

## 16.8 复核轮五条臂（48 种子 / 臂，池 2_133，同 worker、同控制行）

| 臂 | `rows_sha_v3` | alive | L2 | L3 | L2 hazard/1k | L2 beats | L2 死 | clvl≥4 | 配对 vs 控制行 UCB95 | 配对 vs M2-SET |
|---|---|---|---|---|---|---|---|---|---|---|
| 控制行 / OFF | `0e5a1acd…` ✅ | 20 | 34 | **4** | 0.2220 | 85578 | 19 | 25 | 0 | — |
| **M2-SET**（裁定 1/2/3） | `f33f7af5…` | **28** | 30 | 2 | **0.1374** | 87358 | **12** | 18 | **−0.01860** | — |
| M2-FULL′ | `72a86601…` | 22 | 30 | 3 | 0.1948 | 82150 | 16 | 22 | +0.09506 | saved 2 / lost 8 |
| M2-SET+coach′ | `615f1281…` | 26 | 30 | 2 | 0.1737 | 80607 | 14 | 16 | +0.01734 | saved 0 / lost 2 |
| M2-SET+stop′ | `c79526a8…` | 25 | 30 | 2 | 0.1528 | 91604 | 14 | 23 | +0.04315 | saved 2 / lost 5 |

复核**前**的同名臂，摆在一起对照（同种子、同 worker）：

| 臂 | `rows_sha_v3` | alive | hazard/1k | L2 beats | clvl≥4 |
|---|---|---|---|---|---|
| M2-FULL | `68f41f5c…` | 26 | 0.1507 | 86267 | 21 |
| M2-SET+coach | `27b722fb…` | 25 | 0.1794 | 78026 | 16 |
| M2-SET+stop | `9842d5f8…` | 28 | 0.1270 | 94473 | 23 |

**读数（不加修饰）**：

1. **裁定 1/2/3 不受影响，结论不变。** M2-SET 逐比特未变
   （H1 是评测器 CLI、H4 是只记不排的遥测，两者都不进决策路径），
   仍是本战役唯一 UCB95 < 0 的臂。
2. **修好载具**把 stop-v1 从「救 5 赔 5、5 次死在载具里」变成
   「**0 次死在载具里**、alive 25」。**净存活从 0 变成 −3**（对 M2-SET）。
   这不显著（discordant 只有 7，UCB95 0.15195），但方向不好看，
   而且它买回来的东西是**深度**：clvl≥4 18 → 23、L2 beats +4.9%。
3. **coach-v05 加了作用域与释放路径之后依旧只赔不赚**（saved 0 / lost 2）。
   §十三.3 的建议不变。
4. **两条新法合起来（M2-FULL′）比任一单条都差**（alive 22），与复核前同向。
5. **一句必须写下来的方法论**：本轮把 5 条臂在同一个 48 种子池上又跑了一遍
   （其中 OFF/FULL/STOP 跑了两遍）。**同一池的过拟合风险在累积**（§十四.11）。

## 16.9 复核轮产物

| 用途 | 路径 |
|---|---|
| 全部改动脚本（逐条锚点替换，可复算） | `/home/laure/r17_work/r19/m2rr/p1_protocol.py` … `p16_section16.py` |
| 修正后的测试并集脚本 | `/home/laure/r17_work/r19/m2rr/rr_tests.sh` |
| 测试日志 / 失败集合 | `/home/laure/r17_work/r19/m2rr/rr-{pristine,merged}{.log,-fails.txt,-ids.txt}` |
| 探针驱动（5 臂，复用 M2 的统计代码） | `/home/laure/r17_work/r19/m2rr/rr_driver.py` |
| 探针脚本（两趟） | `/home/laure/r17_work/r19/m2rr/rr_probe.sh`、`rr2_probe.sh` |
| 逐臂行 | `/home/laure/r17_work/r19/m2rr-probe/rr-{off,set,full,coach,stop}-rows.json` |
| 臂汇总 | `/home/laure/r17_work/r19/m2rr-probe/rr-summary-{all,"off,full,stop"}.json` |
| **半径 3 反事实臂（已退回的改动）** | `/home/laure/r17_work/r19/m2rr-probe/radius3/` |
| 复核轮读数脚本 / 输出 | `/home/laure/r17_work/r19/m2rr/rr_read.py`、`rr-read-final.txt` |
| 树指纹（跑前 / 跑后，两趟） | `/home/laure/r17_work/r19/m2rr/rr{,2}-tree-{before,after}.txt` |
| 身份检查 + 补丁重生成 | `/home/laure/r17_work/r19/m2rr/rr_patch.sh` |
| **对主树的补丁（25 文件，+10028 / −215 内容行，全 LF，sha256 `315133a531bbc13d…`）** | `/home/laure/r17_work/r19/m2.patch`（已重生成；行数口径见账本 21:00 那条） |
| 本报告复核前的备份 | `/home/laure/r17_work/r19/m2rr/M2-REPORT.md.prereview.bak` |

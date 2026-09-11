# R19 · 神谕老师第一步·第二轮(按窗分工与逐条消融)报告 — 登记版(2026-09-11 凌晨,主刀归档)

**判决:有一条臂在不牺牲存活与杀伤的前提下赢了学生的一层:ORACLE-FARM(神谕内环只接 FARM 窗,DIVE 窗交回学生 7e31dc54)。**
但要按「证明了什么」来说:赢的是「神谕的 FARM 内环 + 学生自己的 DIVE 行为」;一层那条统计与第一轮 ORACLE 臂逐位相同(同样三个一层死亡种子 2133017/2133021/2133026,同样 saved 7 / lost 2 / net +5 / UCB95 −0.00437),第二轮没有给一层主张增加独立证据;它增加的是「把 DIVE 窗还给学生之后,神谕的一层纪律不再以存活和杀伤为代价」。

| 臂(M2-SET,2_133 池 48 种子;三次复跑行哈希全同) | 局末存活 | 一层死亡 | 一层每千拍风险 | 二层每千拍风险 | 总击杀 | clvl≥4 | 到达三层 | rows_sha_v3 |
|---|---|---|---|---|---|---|---|---|
| PARENT 7e31dc54(回归 = M2-SET) | 28 | 8 | 0.0213 | 0.1374 | 5550 | 18 | 2 | f33f7af5… |
| ORACLE(回归 = 第一轮) | 12 | 3 | 0.0100 | 0.2384 | 5469 | 20 | 26 | 1507c3be… |
| **ORACLE-FARM** | **28** | **3** | **0.0072** | **0.1059** | **6353** | **31** | 5 | e47d5a58… |
| OF-POTION(+ 药水子句) | 20 | 8 | 0.0214 | 0.1494 | 5938 | — | 4 | 17d3ea46… |
| OF-POTION-CONTACT(+ 贴身脱离) | 20 | 8 | 0.0218 | — | 5917 | — | 4 | 525d18ef… |
| ORACLE-PICKUP(神谕 + 仅捡拾子句) | 9 | 4 | 0.0133 | 0.2551 | 5295 | — | 29 | 3f02ce54… |

配对(事件口径,m2_probe_driver 单侧 UCB95;36 个比较无多重校正,零假设下预期 1.8 个越线,实得 7 个,归结为 3 个事实):
- ORACLE-FARM 对 PARENT:局末死亡 saved 9 / lost 9(net 0);一层死亡 saved 7 / lost 2(net +5,UCB95 −0.004,**一个种子翻转即失守**);未到二层 net +5。
- ORACLE-FARM 对 ORACLE:局末死亡 saved 18 / lost 2(net +16,UCB95 −0.202)——把 DIVE 窗还给学生,神谕就不再死在深层。
- 药水子句(OF-POTION 对 ORACLE-FARM):一层死亡 saved 2 / lost 7(net −5),局末死亡 net −8 —— **有害**,且 53% 的想喝被饮药主权掩码拒绝。
- 贴身脱离(对 OF-POTION):全部 0/0 —— **无作用**;只触发 132 拍,其中 68% 原地未动(第一轮缺陷未根治,只是变小)。
- 捡拾子句(ORACLE-PICKUP 对 ORACLE):一层 net −1(无害),但补给窗 7761 对 1735 —— **补给窗爆炸归因定案**:捡拾是多拍行走宏,把人带离怪物后 `_farm_handoff` 把 FARM 掩掉,经理只剩 RESUPPLY 可选,开一个 9.7 拍的短窗又关,循环;不是 a13 按得多(698 对 909)。

路由是引擎自己的机制(`options_env.py:2815 self._workers.get(int(option))`),单元卷 85 项全过,真实局里两个 callable 服务的拍数与环境台账逐窗相等;神谕槽 38794 FARM 拍,学生槽 51624 DIVE 拍(其中 33901 在 mask_forced 窗)。
主树指纹编队前后逐位相同(a7a4c75c93b2cd1f);m2-merge、oracle-tree 零写入;teacher_v1.py 未动;处女池零接触。

**归档**:完整报告(含复核轮 §十六)、pass1 报告、`diagnostics.md`、`teacher_v2.py` + 85 项单元卷、驱动、六臂全部行文件与收据、`teacher-round2.patch` 在 `train/runs/r19-reports/teacher-round2/`。

**给主席的两项裁定请求(凌晨未叫醒您,等您裁)**:
1. 一层主张的复现:在一个处女池(建议 2_116,48 种子)上只跑 PARENT 与 ORACLE-FARM 两臂,零训练;这会消耗一个处女池。
2. 第二步的老师改为 ORACLE-FARM:它的 FARM 内环本质上就是 bc_worker 第 1 世代的教师(`dispatch("farm")`),第二步可直接沿用既有 gen-1 BC 管线(M2-SET 世界 + 新示范池 2_144 + 挂载旗标),不含任何本能子句,修订二稿里的「teacher-v1-obs」与 a12/a14 排除问题随之消失;请裁是否据此出修订三稿。

---

以下为编队报告原文(实施者 + 两名对抗审稿 + 修复员,Opus;数字均出自其自产文件):

# R19 · TEACHER 第二轮报告（合成臂 ORACLE-FARM 与逐条消融）

- 日期：2026-09-10 深夜 → 2026-09-11 凌晨（第二轮；第一轮与复核轮同为 2026-09-10）
- 角色：teacher 第二轮实施者
- 裁定依据：主席 2026-09-10 ~23:00 批准的第二轮臂（ledger `R19_TEACHER_ROUND2_PLAN`）；
  主席已入睡，授权自主执行**且仅执行**本轮。
- 工作树：`/home/laure/r17_work/r19/oracle2-tree`
  （`rsync -a --exclude __pycache__ --exclude .pytest_cache` 自**只读**的第一轮
  `oracle-tree`；`build -> /home/laure/r17_work/r17-1/build-res`）
- 只读且零写入：主树 `/home/laure/AlphaDiablo/diablogym`、`m2-merge`、`oracle-tree`（见 §十二）
- 新代码只有一个模块 `train/runs/r10-staging/teacher_v2.py` 与它自己的
  `test_teacher_v2.py`。**第一轮的 `teacher_v1.py` / `test_teacher_v1.py` 一字未动**
  （sha256 与 `oracle-tree` 逐位相同，§十二）。**没有任何 C++／引擎改动，没有动任何已注册法条。**
- 种子：只用 2_133 池 2133000–2133047，3 片（a/b/c）。
  处女池 2_116-119、2_126-128：**零接触**（§十二 枚举了本轮全部产物里的全部 48 个种子）。
- 本报告所有数字来自本轮自产文件 `round2/probe/`；第一轮数字标明「第一轮」并取自原样保留的
  `r19/teacher-probe/`；没有文件支持的一律写「未验证」。
- **复核轮**（2026-09-11 凌晨）：独立复核给了 7 条 finding（`verdict: ship`，无 blocker／major），
  **全部已修**；其中两条动到遥测，因此**六条臂全部重跑、两道回归闸重过**，
  而 **`rows_sha_v3` 六条臂逐位不变**。整段见 **§十六**，最终裁定见 **§16.4**。

---

## 零、一句话结论（VERDICT）

**有一条臂在 L1 死亡上过线：`ORACLE-FARM`。但这句话必须逐字写全，
否则它会被读成一句它撑不住的话（复核轮 finding 3，§十六）：**

> **赢的是「神谕的 FARM 内环」，不是「一个脚本老师赢了学生」。**
> `ORACLE-FARM` 里 **57.1% 的工人拍（51624 / 90418）仍然是学生本人在打**
> （其中 33901 拍落在 `mask_forced` 窗里），整个 DIVE 窗原样交还给 7e31dc54；
> 神谕只服务了 38794 拍 FARM。
> **而且承载这条结论的 L1 统计，和第一轮 `ORACLE` 是同一个统计**：
> 两条臂的三个 L1 死亡种子**完全相同**（2133017 / 2133021 / 2133026），
> 配对数字逐位相同（saved 7 / lost 2 / net **+5** / UCB95 **−0.00437**），
> 而 `oracle-farm` 对 `oracle` 的 L1 配对是 **0 / 0 / 0**。
> **所以第二轮没有为那条 L1 结论提供任何独立确认。**
> 第二轮真正新增的事实是另一件：**把 DIVE 窗还给学生之后，
> 神谕那套 L1 纪律不再以「冲下去送死」为代价**——
> 局末存活 28 对 28 没掉，杀伤反涨 14.5%，而第一轮的 `ORACLE` 存活只有 12。

`ORACLE-FARM` = **FARM 窗交给冻结的神谕内环，DIVE 窗交还给学生 7e31dc54**。

| 对 PARENT | ORACLE-FARM | PARENT | 配对（事件=坏事） |
|---|---|---|---|
| **死于 L1** | **3** | 8 | saved 7 / lost 2 / net **+5** / UCB95 **−0.00437** |
| 局末存活 | **28** | 28 | saved 9 / lost 9 / net 0 / UCB95 +0.14540 |
| 总杀伤 | **6353** | 5550 | ＋14.5% |
| clvl≥4 的局 | **31** | 18 | — |
| L1 每千拍危害 | **0.0072** | 0.0213 | 三分之一 |
| L2 每千拍危害 | **0.1059** | 0.1374 | — |
| 首降后 1800 拍仍活 | **42 / 44** | 34 / 37 | — |

**主席问的那句话逐字回答：**
「有没有任何一条臂在 L1 死亡上赢过学生（配对 net > 0 且单侧 UCB95 < 0），
**同时**没有把局末存活和杀伤打崩？」——**有，且只有 ORACLE-FARM 一条**
（`ORACLE` 的 L1 配对同样过线，但它把局末存活从 28 打到 12，所以不满足后半句）。
它把 L1 死亡从 8 压到 3（UCB95 −0.00437），**局末存活 28 对 28 没有掉**，
**杀伤反而涨了 14.5%**，深度只从 1.667 轻涨到 1.833（不是神谕那种冲下去送死）。

**三条消融的结论，和第一轮的猜测部分相反：**

1. **喝药条是 L1 的凶手，不是拾药条。**
   `OF-POTION`（= ORACLE-FARM ＋ 只加喝药条）把 L1 死亡从 **3 直接推回 8**
   （对 ORACLE-FARM 配对：L1 saved 2 / lost 7 / net **−5**；局末死亡 net **−8**，
   存活 28 → 20）。**它一条就把整个 L1 胜利吃光了。**
2. **接触脱离条几乎什么都没做。** `OF-POTION-CONTACT` 对 `OF-POTION`：
   L1 死亡 saved 0 / lost 0，局末死亡 saved 1 / lost 1，八个 L1 死亡种子**一模一样**。
   全 48 局只触发 193 拍、真开火 132 拍（第一轮是 859 拍），
   而且**仍有 68.0%（87/128）原地没动**——老毛病没治好，只是量小到看不见。
3. **拾药条确实是 RESUPPLY 窗口爆炸的原因，这一条第一轮猜对了。**
   `ORACLE-PICKUP`（= 完整神谕 base 两边窗 ＋ 只加拾药条）
   把 RESUPPLY 窗从 ORACLE 的 **1735 炸到 7761**、总窗口从 4396 炸到 10560——
   几乎完全复现了第一轮 TEACHER-V1 的 7605 / 10244。
   **但它对 L1 几乎无害**（L1 死亡 4 对神谕的 3，配对 net −1）。
   所以第一轮报告 §十一 第 4 条「拾药条现在最可疑」——
   **对窗口爆炸是对的，对 L1 伤害是错的**。

**为什么合成臂有效，有机制证据，不是运气：**
引擎自己的审计（`options_env.py:2014/2294`，本轮由只读的 env 子类逐窗汇总）显示，
**神谕在 DIVE 窗里 77% 的拍是空转**：18514 拍里只有 4298 拍真的被执行
（`no_effect_requests` 14216、`fuse_trips` 567）。
把 DIVE 窗还给学生之后，同样的窗口执行率变成 **37939 / 51676 = 73% 真执行**。
神谕会打架，不会下楼；学生会下楼，不会打 L1。**合成臂就是把这两件事各归其主。**

**一句必须和上面那句一起看的话：**
`ORACLE-FARM` 的 L1 胜利，**和第一轮 ORACLE 的那条是同一个统计**——
三个 L1 死亡种子完全相同（2133017 / 2133021 / 2133026），
所以 **−0.00437 只比线低 0.004，翻一个种子就变成 +0.039**（§6.1），
而且它建在 2_133 这个已经被反复消费的池子上。
**在处女池上复验之前，不该有任何东西建在这条 95% 结论上。**

---

## 一、本轮问什么、跑了哪六条臂

第一轮的结论是：冻结的神谕 base 单独把 L1 每拍危害砍半（0.0100 对学生 0.0213），
但**冲下去送死**（26 局到 L3，局末存活 12 对 28）；而本能层（喝药 0.60 / 拾药 / 人群＋接触脱离）
在同一个 base 上是**净负**（L1 死亡 3 → 10）。第一轮自曝的两个缺口是
「本能层没有逐条消融」和「窗口爆炸的归因只有相关没有因果」。

本轮六条臂，同一个 M2-SET 世界（17 键，从 `r19/teacher_probe_driver.py` **逐字复制**），
同一个 2_133 池 48 个种子、同样的 a/b/c 三片：

| 臂 | FARM 窗 | DIVE 窗 | 作用 |
|---|---|---|---|
| `parent` | 学生 7e31dc54 | 学生 | 回归闸，必须等于 `f33f7af5…` |
| `oracle` | teacher_v1 "oracle" | 同一个对象 | 回归闸，必须等于 `1507c3be…` |
| `oracle-farm` | teacher_v1 "oracle" | 学生 7e31dc54 | **合成臂** |
| `of-potion` | 神谕 base ＋ 喝药条 | 学生 | 只加喝药条 |
| `of-potion-contact` | 神谕 base ＋ 喝药 ＋ 接触脱离 | 学生 | 再加接触脱离 |
| `oracle-pickup` | 神谕 base ＋ 拾药条 | **同一个对象**（两边窗） | 只加拾药条 |

`RESUPPLY` 窗（以及它名下的 sweep / gold-grab / portal / retreat 各级）
**全程是法条脚本，不经过任何 worker**——所以 `workers` 里根本没有 RESUPPLY 项，
六条臂的 `resupply` 拍数**全部是 0**（§七），这是本轮测出来的，不是假设的。

---

## 二、交付物

| 文件 | 是什么 |
|---|---|
| `oracle2-tree/train/runs/r10-staging/teacher_v2.py` | 本轮唯一的新代码：`make_teacher2(variant, env=None, drink_sovereignty=True, parent_zip=…)` |
| `oracle2-tree/train/runs/r10-staging/test_teacher_v2.py` | 本轮单元卷：第一遍 70 项全过（`round2/tests-full.log`），复核轮加严并新增后 85 项全过（`round2/tests-full-review.log`） |
| `round2/teacher2_shard.py` | 一片一臂，行经 `probe_r17_deployment.run_episode` 逐字不动 |
| `round2/teacher2_driver.py` | 六臂驱动、回归闸、`arm_stats`、配对检定 |
| `round2/teacher2_extra.py` | 本报告全部表格的生成器（只读本轮产物） |
| `round2/probe/*-rows.json` | 六条臂的全部行与遥测 |
| `round2/probe/regression-{parent,oracle}.json` | 两道回归闸的收据 |
| `round2/probe/round2-summary.json` | 汇总（含全部 36 个配对） |
| `round2/diagnostics.md` | 机器生成的原始表格 |
| `round2/tests-full.log` / `tests-nonepisode.log` | 单元卷 |
| `round2/driver-*.log`、`round2/probe/*.cmd` | 每一条命令行的收据 |
| `round2/gate_then_arms.sh`、`run_regressions.sh`、`finalize.sh`、`smoke.sh`、`run_tests_*.sh`、`deploy.sh` | 第一遍实际跑过的脚本 |
| `round2/rr_full.sh`、`rr_regressions.sh`、`rr_arms.sh`、`prelaunch_review.sh`、`compare_passes.py`、`ledger_review.py` | **复核轮**实际跑过的脚本（§十六） |
| `round2/probe-pass1/`、`round2/probe-pass2/` | 第一遍与复核轮第一次重跑的全部产物，原样保留 |
| `round2/review-round-compare.txt` | 三遍之间的逐臂 SHA／`arm_stats` 对照 |
| `round2/review-round-manifest-before.txt` | **启动前**对每个将要运行的脚本的 sha256 ＋ 时间戳（finding 4） |
| `round2/tests-full-review.log` | 复核轮的完整单元卷 |
| `round2/TEACHER-ROUND2-REPORT.pass1.md`、`driver-*.pass1.log` | 复核前的报告与日志，原样保留 |
| `round2/mk_report.py`、`round2/verdict.json` | 本报告的生成器与裁定问答 |
| `round2/main-tree-fingerprint-{mid,after}.txt`、`…-review-{before,after}.txt` | 主树只读证明（第一遍 ＋ 复核轮） |

---

## 三、老师是什么，以及「路由」是验过的不是猜的

`teacher_v2.py` 和 `teacher_v1.py` 一样是**探针侧策略**：不注册任何法条、不改任何阈值、
不被任何运行时路径 import，住在探针旁边的 `train/runs/r10-staging/`，
**不属于 `probe_r17_deployment._SOURCE_FILES` 指纹包，也不属于任何其它指纹包**。

**路由用的是引擎自己的机制，代码位置逐处标注：**

- `options_env.py:2815`：`worker = self._workers.get(int(option))`——
  每个选项窗**只解析一次** worker，然后 `:3112-3137` 整扇窗都走这一个 callable。
- `options_env.py:67`：`FARM, DIVE, RESUPPLY = 0, 1, 2`；
  `:1465`：`_win["mode"] = ("farm","dive","resupply")[option]`——**选项就是窗口种类**。
- 所以 `workers={FARM: oracle_cb, DIVE: parent_cb}` 就是引擎原生的按窗分派，
  不需要在一个 callable 里偷看窗口种类。
- `RESUPPLY` 窗整扇由 `:2822-3110` 的脚本服务链跑完；
  sweep / gold-grab / portal / retreat 是同一条链上的级（`:2846-2851`），同样不碰 worker。
- `forced_dive` / `mask_forced` 是**经理侧**的事实（`m[FARM] = not forced_dive`，
  `:1089` 与 `:1371`），探针逐窗记录（`run_episode:600-621`）。
  本轮把每扇 DIVE 窗的拍数按 `_win["window_id"]` 记下来，
  再和探针行自己的 `dive_windows[].forced` 做**连接**，才得出 §七 的「强制下楼拍」一列。

**钩子**：`run_episode` 只把 `on_beat` / `episode_reseed` 交给**一个** `cb`（`:564` 与 `:567`），
它看不见 `workers`。所以 `TeacherV2` 是一个**路由对象**：
`.workers` 进 env，它本身作为 `cb` 进 `run_episode`，
它的 `on_beat` setter / `episode_reseed` / `bind` **扇出到每一个内层 callable**。
因为一拍只会有一个 callable 被调用，`on_beat` **每个工人拍仍然恰好触发一次**
（单元卷里直接数了：3 拍 → 3 次）。学生那一侧的定种纪律
（`model.set_random_seed(seed)` 每局一次，`load_zip_policy:201-203`）
与认证对照行**逐位相同**。

**常数**：本轮**没有引入任何新的数值阈值**。
`teacher_v2.py` 的那一个常数块把第一轮冻结常数块里的值**原样 import**，
逐条写明出处（0.60 喝药、拾药 Chebyshev 1、接触 0.35 / 半径 1 / 计数 1、400 拍围栏、地形半径 1）。
**人群触发条与 `STAIRS_TIEBREAK_RADIUS` 按主席本轮的措辞被刻意不 import**
（「NO crowd trigger、NO stairs tie-break」）。

**两处与第一轮 shard 的差别，都是只读的，而且被回归闸证明为零影响：**
(1) 每个 worker callable 外面套了一层 `WorkerProbe`（先记数，再原样转调）；
(2) env 用 `teacher_v2.observing_env_class()`——一个只 `super().step()` 之后
**读已经返回的 info 字典**的 `OptionsEnv` 子类，用来汇总探针行不带的
`worker_no_effect_requests` 审计。两条回归 SHA 完全复现（§四），
**这就是这两样东西没有扰动世界的证明**，不是断言。

---

## 四、回归（第一道闸）

  parent: got f33f7af5f236f1fc... expected f33f7af5f236f1fc... equal=True all_refs_agree=True
      ref /home/laure/r17_work/r19/m2rr-probe/rr-set-rows.json: f33f7af5f236f1fc
      ref /home/laure/r17_work/r19/m2-probe/m2-set-rows.json: f33f7af5f236f1fc
      ref /home/laure/r17_work/r19/teacher-probe/parent-rows.json: f33f7af5f236f1fc
  oracle: got 1507c3be68a0da29... expected 1507c3be68a0da29... equal=True all_refs_agree=True
      ref /home/laure/r17_work/r19/teacher-probe/oracle-rows.json: 1507c3be68a0da29

两道闸**都过**，而且 `parent` 的 SHA 同时与三个独立的既有产物一致
（`m2rr-probe/rr-set-rows.json`、`m2-probe/m2-set-rows.json`、
第一轮的 `teacher-probe/parent-rows.json`），
`oracle` 与第一轮 `teacher-probe/oracle-rows.json` 一致。
**只有闸过了，四条新臂才被启动**（`round2/rr_full.sh` 把这条纪律写死成条件分支：
两张收据都 `equal=true`、两张都 `all_references_agree`、且两张都是 `"pass": "run"`
才启动四条新臂，否则 `exit 2`）。

**这条纪律现在在磁盘上可查（复核轮 finding 4 的修复）**：第一遍时 `--reuse` 汇总腿
会把两张收据覆盖重写，于是收据的 mtime 反而晚于它本该把关的四条臂，
光看 mtime 无法证明先后。现在 `--reuse` 腿写的是兄弟文件
`probe/regression-*-reuse.json`（`"pass": "reuse-summarize"`），
**闸收据 `probe/regression-parent.json` / `regression-oracle.json` 只由真正跑的那一腿写
（`"pass": "run"`），闸本身现在也要求 `pass == "run"`**。
复核轮最终那一遍的磁盘时间戳（`ls --time-style=+%H:%M:%S`）：

| 文件 | mtime | 含义 |
|---|---|---|
| `probe/regression-oracle.json` | 01:09:27 | 神谕回归收据（先完成的那条） |
| `probe/regression-parent.json` | 01:10:48 | 学生回归收据 |
| `probe/r2-oracle-farm-a.cmd`、`r2-of-potion-a.cmd` | 01:10:48 | 四条新臂的命令行收据，**闸判定之后同一秒才写出** |
| `probe/regression-*-reuse.json` | 01:22:15 | `--reuse` 汇总腿的兄弟文件，**不再覆盖闸收据** |

两张闸收据**都不晚于**四条新臂的 `.cmd`（神谕早 81 秒，学生同一秒——
闸的判定 `GATE result: YES` 打在 01:10:48，四条臂在同一秒被启动）；
`driver-parent.log` / `driver-oracle.log` 里的 `REGRESSION` 行（`equal` 为 true）同样在启动之前；
`round2/review-round-manifest-before.txt` 在 **01:00:55、也就是启动前**
把每个将要运行的脚本逐个 sha256 并打了时间戳，
`round2/review-round-manifest-after.txt` 是收尾时的同一张表。

### 4.1 运行收据

| arm | rows_sha_v3 | n | runtime_errors | elapsed_s | source_changed_during_run |
|---|---|---|---|---|---|
| parent | `f33f7af5f236f1fc…` | 48 | 0 | 592.5 | [] |
| oracle | `1507c3be68a0da29…` | 48 | 0 | 511.2 | [] |
| oracle-farm | `e47d5a5850ae1c46…` | 48 | 0 | 686.7 | [] |
| of-potion | `17d3ea46f63f88a6…` | 48 | 0 | 640.4 | [] |
| of-potion-contact | `525d18ef98ddbe8d…` | 48 | 0 | 648.5 | [] |
| oracle-pickup | `3f02ce54b8922626…` | 48 | 0 | 681.7 | [] |

`runtime_errors` 六条臂全 0；`source_changed_during_run` 六条臂全空
（跑之前和跑之后对 10 个源文件 ＋ worker zip 逐个 sha256）。

---

## 五、全表

| stat | parent | oracle | oracle-farm | of-potion | of-potion-contact | oracle-pickup |
|---|---|---|---|---|---|---|
| n | 48 | 48 | 48 | 48 | 48 | 48 |
| alive | 28 | 12 | 28 | 20 | 20 | 9 |
| deaths | 20 | 36 | 20 | 28 | 28 | 39 |
| l1_deaths | 8 | 3 | 3 | 8 | 8 | 4 |
| l2_deaths | 12 | 14 | 14 | 18 | 18 | 13 |
| l2_beats | 87358 | 58735 | 132166 | 120480 | 112884 | 50962 |
| l2_hazard_per_1k | 0.1374 | 0.2384 | 0.1059 | 0.1494 | 0.1595 | 0.2551 |
| l1_kills | 4723 | 4534 | 4969 | 4639 | 4647 | 4374 |
| l2_kills | 826 | 763 | 1331 | 1298 | 1263 | 771 |
| kills_total | 5550 | 5469 | 6353 | 5938 | 5917 | 5295 |
| l2_reach | 30 | 41 | 35 | 35 | 35 | 41 |
| l3 | 2 | 26 | 5 | 4 | 4 | 29 |
| clvl_ge_4 | 18 | 20 | 31 | 27 | 26 | 24 |
| clvl_mean | 3.229 | 3.438 | 3.667 | 3.521 | 3.5 | 3.396 |
| depth_mean | 1.667 | 2.646 | 1.833 | 1.833 | 1.833 | 2.812 |
| micro_steps_median | 12000.0 | 9302.0 | 13420.5 | 12433.0 | 12083.5 | 9268.5 |
| fd2_n | 30 | 41 | 35 | 35 | 35 | 41 |
| fd2_beat_median | 6798.0 | 6064 | 6300 | 6390 | 6390 | 6065 |
| fd2_belt_median | 6.0 | 5 | 5 | 5 | 5 | 5 |
| fd2_ac_mean | 17.067 | 16.293 | 17.257 | 16.943 | 17.0 | 15.78 |
| fd2_clvl_mean | 3.0 | 2.927 | 3.0 | 3.086 | 3.086 | 2.927 |
| alive_at_fd_plus_1800 | 34 | 27 | 42 | 36 | 35 | 30 |
| fd_plus_1800_observed_n | 37 | 40 | 44 | 40 | 40 | 41 |
| gold_final_median | 92.0 | 87.0 | 126.0 | 92.5 | 96.0 | 102.0 |
| gold_collected | 8625 | 8638 | 10961 | 9225 | 8939 | 8478 |
| gold_sold | 9066 | 7274 | 10889 | 9672 | 9685 | 8123 |
| gold_spent | 23048 | 19738 | 26307 | 24303 | 23661 | 19862 |
| purchases | 276 | 241 | 302 | 269 | 264 | 251 |
| sales | 218 | 231 | 296 | 258 | 253 | 253 |
| loot_collected | 123 | 110 | 148 | 118 | 117 | 119 |
| identified | 21 | 12 | 19 | 17 | 17 | 9 |
| weapon_upgrades | 12 | 4 | 6 | 7 | 6 | 3 |
| sweep_windows | 752 | 786 | 886 | 875 | 867 | 793 |
| chests_opened | 339 | 363 | 390 | 375 | 376 | 364 |
| barrels_smashed | 376 | 417 | 462 | 449 | 445 | 417 |
| sweep_gold | 0 | 0 | 0 | 0 | 0 | 0 |
| grab_windows | 571 | 547 | 614 | 619 | 619 | 551 |
| grab_piles | 729 | 727 | 809 | 796 | 786 | 729 |
| grab_gold | 8695 | 8701 | 10155 | 9892 | 9732 | 8881 |
| retreats | 58 | 87 | 80 | 70 | 68 | 94 |
| belt0_at_death | 11/20 | 21/36 | 11/20 | 16/28 | 16/28 | 22/39 |
| windows_total | 3815 | 4396 | 4934 | 4539 | 4438 | 10560 |
| windows_farm | 1341 | 1688 | 1921 | 1730 | 1687 | 1467 |
| windows_dive | 763 | 973 | 1012 | 878 | 838 | 1332 |
| windows_resupply | 1711 | 1735 | 2001 | 1931 | 1913 | 7761 |
| windows_forced_dive | 497 | 718 | 552 | 483 | 473 | 665 |
| windows_mask_forced | 2208 | 2453 | 2553 | 2414 | 2386 | 8426 |
| deaths_by_floor | {"1": 8, "2": 12} | {"1": 3, "2": 14, "3": 11, "4": 7, "5": 1} | {"1": 3, "2": 14, "3": 3} | {"1": 8, "2": 18, "3": 1, "4": 1} | {"1": 8, "2": 18, "3": 1, "4": 1} | {"1": 4, "2": 13, "3": 12, "4": 8, "5": 1, "6": 1} |

### 5.1 逐层曝光与危害

| arm | L1 beats/deaths/hazard per 1k | L2 beats/deaths/hazard per 1k | L3 beats/deaths/hazard per 1k | L4 beats/deaths/hazard per 1k | L5 beats/deaths/hazard per 1k |
|---|---|---|---|---|---|
| parent | 374770 / 8 / 0.0213 | 87358 / 12 / 0.1374 | 935 / 0 / 0.0 | 0 / 0 / 0.0 | 0 / 0 / 0.0 |
| oracle | 300406 / 3 / 0.01 | 58735 / 14 / 0.2384 | 9809 / 11 / 1.1214 | 5372 / 7 / 1.3031 | 1008 / 1 / 0.9921 |
| oracle-farm | 414998 / 3 / 0.0072 | 132166 / 14 / 0.1059 | 4742 / 3 / 0.6326 | 0 / 0 / 0.0 | 0 / 0 / 0.0 |
| of-potion | 373201 / 8 / 0.0214 | 120480 / 18 / 0.1494 | 2185 / 1 / 0.4577 | 113 / 1 / 8.8496 | 0 / 0 / 0.0 |
| of-potion-contact | 366841 / 8 / 0.0218 | 112884 / 18 / 0.1595 | 2472 / 1 / 0.4045 | 113 / 1 / 8.8496 | 0 / 0 / 0.0 |
| oracle-pickup | 300145 / 4 / 0.0133 | 50962 / 13 / 0.2551 | 8632 / 12 / 1.3902 | 14758 / 8 / 0.5421 | 1029 / 1 / 0.9718 |

**这张表是本轮最关键的一张。**
`ORACLE-FARM` 在 **L1 上把每千拍危害压到 0.0072**（学生 0.0213、神谕 0.0100），
**同时**没有像神谕那样把自己送到 L3/L4（神谕 L3 危害 1.12、L4 1.30；
合成臂 L3 只有 3 次死亡、L4 零曝光）。
`OF-POTION` 把 L1 危害打回 0.0214——**和学生一模一样**，等于喝药条把 base 的 L1 纪律整个抵消了。

### 5.2 到达深度阶梯

| arm | >=L2 | >=L3 | >=L4 | >=L5 |
|---|---|---|---|---|
| parent | 30 | 2 | 0 | 0 |
| oracle | 41 | 26 | 11 | 1 |
| oracle-farm | 35 | 5 | 0 | 0 |
| of-potion | 35 | 4 | 1 | 0 |
| of-potion-contact | 35 | 4 | 1 | 0 |
| oracle-pickup | 41 | 29 | 14 | 2 |

神谕与 `ORACLE-PICKUP` 到 L3 的局是 26 / 29，合成臂只有 5——
**这正是合成臂活下来的原因**：它没有拿 DIVE 窗去冲。

---

## 六、配对检定（全部 36 个，不是挑出来的）

**先说多重比较（复核轮 finding 5）**：下面是 **36 个单侧 95% 检定**，
**没有做任何多重性校正**。在零假设下，36 个单侧 95% 检定里
**平均会有 36 × 0.05 ≈ 1.8 个**纯靠运气越过 0 线；本轮实际越线 **7 个**。
但这 7 个**不是 7 件独立的事**，它们只是 **3 件事**被数了 7 次：

| 底层事实 | 越线的比较 | 次数 |
|---|---|---|
| 冻结神谕 base 把 L1 危害砍半（**同样三个种子** 2133017/2133021/2133026） | `oracle_vs_parent · died_on_L1`、`oracle-farm_vs_parent · died_on_L1` | 2 |
| 神谕 base 两边窗就会**往下冲**（`oracle` 与 `oracle-pickup` 的两边窗是同一个 base） | `oracle_vs_parent · not_reached_L2`、`· not_reached_L3`、`oracle-pickup_vs_parent · not_reached_L2`、`· not_reached_L3` | 4 |
| 把 DIVE 窗还给学生，神谕就不再冲下去送死 | `oracle-farm_vs_oracle · died` | 1 |

**承载本轮 VERDICT 的那一个只比 0 线低 0.004**（−0.00437），
且属于上表第一行——**它和第一轮是同一个统计**。
再加上 §十四.2（2_133 池今晚又被消费了四次），
**这个「95%」的标签比字面上要弱**。真正的解药是处女池复验
（2_116-119 / 2_126-128 本轮零接触，§十二），已经写进 §十五.2。

| pair (event = the BAD thing) | saved | lost | net | UCB95 one-sided |
|---|---|---|---|---|
| oracle vs parent · died | 2 | 18 | -16 | +0.46458 |
| oracle vs parent · died_on_L1 | 7 | 2 | +5 | -0.00437 **<0** |
| oracle vs parent · not_reached_L2 | 14 | 3 | +11 | -0.09876 **<0** |
| oracle vs parent · not_reached_L3 | 25 | 1 | +24 | -0.37177 **<0** |
| oracle-farm vs parent · died | 9 | 9 | +0 | +0.14540 |
| oracle-farm vs parent · died_on_L1 | 7 | 2 | +5 | -0.00437 **<0** |
| oracle-farm vs parent · not_reached_L2 | 9 | 4 | +5 | +0.01690 |
| oracle-farm vs parent · not_reached_L3 | 4 | 1 | +3 | +0.01268 |
| of-potion vs parent · died | 4 | 12 | -8 | +0.29791 |
| of-potion vs parent · died_on_L1 | 6 | 6 | +0 | +0.11872 |
| of-potion vs parent · not_reached_L2 | 10 | 5 | +5 | +0.02624 |
| of-potion vs parent · not_reached_L3 | 3 | 1 | +2 | +0.02616 |
| of-potion-contact vs parent · died | 3 | 11 | -8 | +0.28864 |
| of-potion-contact vs parent · died_on_L1 | 6 | 6 | +0 | +0.11872 |
| of-potion-contact vs parent · not_reached_L2 | 10 | 5 | +5 | +0.02624 |
| of-potion-contact vs parent · not_reached_L3 | 3 | 1 | +2 | +0.02616 |
| oracle-pickup vs parent · died | 2 | 21 | -19 | +0.53067 |
| oracle-pickup vs parent · died_on_L1 | 8 | 4 | +4 | +0.03372 |
| oracle-pickup vs parent · not_reached_L2 | 14 | 3 | +11 | -0.09876 **<0** |
| oracle-pickup vs parent · not_reached_L3 | 27 | 0 | +27 | -0.44471 **<0** |
| oracle-farm vs oracle · died | 18 | 2 | +16 | -0.20209 **<0** |
| oracle-farm vs oracle · died_on_L1 | 0 | 0 | +0 | +0.00000 |
| oracle-farm vs oracle · not_reached_L2 | 1 | 7 | -6 | +0.21728 |
| oracle-farm vs oracle · not_reached_L3 | 0 | 21 | -21 | +0.55529 |
| of-potion vs oracle-farm · died | 6 | 14 | -8 | +0.31473 |
| of-potion vs oracle-farm · died_on_L1 | 2 | 7 | -5 | +0.20396 |
| of-potion vs oracle-farm · not_reached_L2 | 4 | 4 | +0 | +0.09693 |
| of-potion vs oracle-farm · not_reached_L3 | 2 | 3 | -1 | +0.09731 |
| of-potion-contact vs of-potion · died | 1 | 1 | +0 | +0.04847 |
| of-potion-contact vs of-potion · died_on_L1 | 0 | 0 | +0 | +0.00000 |
| of-potion-contact vs of-potion · not_reached_L2 | 0 | 0 | +0 | +0.00000 |
| of-potion-contact vs of-potion · not_reached_L3 | 1 | 1 | +0 | +0.04847 |
| oracle-pickup vs oracle · died | 3 | 6 | -3 | +0.16424 |
| oracle-pickup vs oracle · died_on_L1 | 1 | 2 | -1 | +0.07999 |
| oracle-pickup vs oracle · not_reached_L2 | 2 | 2 | +0 | +0.06854 |
| oracle-pickup vs oracle · not_reached_L3 | 7 | 4 | +3 | +0.05019 |

  => 7 of 36 comparisons have a one-sided 95% UCB below zero:
     - oracle vs parent | died_on_L1
     - oracle vs parent | not_reached_L2
     - oracle vs parent | not_reached_L3
     - oracle-farm vs parent | died_on_L1
     - oracle-pickup vs parent | not_reached_L2
     - oracle-pickup vs parent | not_reached_L3
     - oracle-farm vs oracle | died

### 6.1 单种子敏感度（凡是 L1 类过线的都算一遍）

  oracle vs parent · died_on_L1: measured -0.00437; one saved->lost +0.03924; one saved->concordant +0.01156
  oracle-farm vs parent · died_on_L1: measured -0.00437; one saved->lost +0.03924; one saved->concordant +0.01156

`ORACLE-FARM` 和 `ORACLE` 的 L1 胜利**是同一个统计**：
两条臂的 L1 死亡种子集合完全相同（2133017 / 2133021 / 2133026，§10.2）。
所以这条 95% 结论的脆弱程度和第一轮完全一样：**翻一个种子就没了**。

---

## 七、路由遥测：哪个 callable 服务了哪种窗

| arm | slot | role | farm beats | dive beats | resupply beats | forced-dive beats | unforced-dive beats | a12 | a13 |
|---|---|---|---|---|---|---|---|---|---|
| parent | farm_slot | parent | 34968 | 41260 | 0 | 31801 | 9459 | 55 | 584 |
| oracle | farm_slot | oracle | 32906 | 18475 | 0 | 16007 | 2468 | 0 | 909 |
| oracle-farm | farm_slot | oracle | 38794 | 0 | 0 | 0 | 0 | 0 | 492 |
| oracle-farm | dive_slot | parent | 0 | 51624 | 0 | 33901 | 17723 | 32 | 137 |
| of-potion | farm_slot | oracle | 33366 | 0 | 0 | 0 | 0 | 117 | 462 |
| of-potion | dive_slot | parent | 0 | 46515 | 0 | 31831 | 14684 | 29 | 180 |
| of-potion-contact | farm_slot | oracle | 31983 | 0 | 0 | 0 | 0 | 113 | 473 |
| of-potion-contact | dive_slot | parent | 0 | 43980 | 0 | 30983 | 12997 | 24 | 170 |
| oracle-pickup | farm_slot | oracle | 27176 | 27097 | 0 | 15095 | 12002 | 0 | 698 |

三件事被这张表证死：

1. **合成臂的路由是真的**：`oracle-farm` / `of-potion` / `of-potion-contact`
   三条臂里，神谕那一格的 `dive beats` 恒为 **0**，学生那一格的 `farm beats` 恒为 **0**。
2. **RESUPPLY 窗从不经过 worker**：六条臂的 `resupply beats` 全部是 **0**。
3. **强制下楼（forced_dive）的拍归属可以查**：例如合成臂 51624 个 DIVE 拍里
   33901 拍落在 `mask_forced` 的窗里，全部由学生服务。

---

## 八、逐条消融：哪条帮、哪条害

| telemetry | oracle | oracle-farm | of-potion | of-potion-contact | oracle-pickup |
|---|---|---|---|---|---|
| policy_episodes | 48 | 48 | 48 | 48 | 48 |
| worker_beats_served_by_teacher | 51381 | 38794 | 33366 | 31983 | 54273 |
| grace_decisions | 0 | 0 | 0 | 0 | 0 |
| base_fallbacks | 0 | 0 | 0 | 0 | 0 |
| potion_wanted | 0 | 0 | 249 | 229 | 0 |
| instinct_potion | 0 | 0 | 117 | 113 | 0 |
| potion_blocked_by_mask | 0 | 0 | 132 | 116 | 0 |
| pickup_wanted | 0 | 0 | 0 | 0 | 1431 |
| instinct_pickup | 0 | 0 | 0 | 0 | 695 |
| pickup_blocked_by_contact | 0 | 0 | 0 | 0 | 736 |
| contact_trigger_beats | 0 | 0 | 0 | 193 | 0 |
| instinct_disengage | 0 | 0 | 0 | 132 | 0 |
| disengage_beats | 0 | 0 | 0 | 132 | 0 |
| disengage_moved | 0 | 0 | 0 | 41 | 0 |
| disengage_not_moved | 0 | 0 | 0 | 87 | 0 |
| disengage_achieved_gain | 0 | 0 | 0 | 31 | 0 |
| disengage_settled | 0 | 0 | 0 | 128 | 0 |
| disengage_settled_same_window | 0 | 0 | 0 | 127 | 0 |
| disengage_settled_cross_window | 0 | 0 | 0 | 1 | 0 |
| disengage_no_improving_move | 0 | 0 | 0 | 61 | 0 |
| disengage_no_legal_move | 0 | 0 | 0 | 0 | 0 |
| disengage_terrain_blocked_dirs | 0 | 0 | 0 | 269 | 0 |
| disengage_terrain_unavailable | 0 | 0 | 0 | 0 | 0 |
| disengage_budget_blocked | 0 | 0 | 0 | 0 | 0 |
| episodes_budget_exhausted | 0 | 0 | 0 | 0 | 0 |
  oracle: disengage beats per episode {"n": 48, "total": 0, "min": 0, "max": 0, "median": 0.0, "mean": 0.0, "episodes_over_100": 0, "max_fraction_of_budget": 0.0, "episodes_shorter_than_budget": 3}
  oracle-farm: disengage beats per episode {"n": 48, "total": 0, "min": 0, "max": 0, "median": 0.0, "mean": 0.0, "episodes_over_100": 0, "max_fraction_of_budget": 0.0, "episodes_shorter_than_budget": 3}
  of-potion: potion refusal rate 132/249 = 0.5301
  of-potion: disengage beats per episode {"n": 48, "total": 0, "min": 0, "max": 0, "median": 0.0, "mean": 0.0, "episodes_over_100": 0, "max_fraction_of_budget": 0.0, "episodes_shorter_than_budget": 8}
  of-potion-contact: potion refusal rate 116/229 = 0.5066
  of-potion-contact: disengage NOT-moved share 87/128 = 0.6797; achieved gain 31 tiles over 132 beats = 0.2348 tiles/beat
  of-potion-contact: disengage beats per episode {"n": 48, "total": 132, "min": 0, "max": 37, "median": 0.0, "mean": 2.8, "episodes_over_100": 0, "max_fraction_of_budget": 0.092, "episodes_shorter_than_budget": 8}
  oracle-pickup: disengage beats per episode {"n": 48, "total": 0, "min": 0, "max": 0, "median": 0.0, "mean": 0.0, "episodes_over_100": 0, "max_fraction_of_budget": 0.0, "episodes_shorter_than_budget": 5}

**喝药条（害）**：249 次想喝，**132 次被 `drink_sovereignty` 掩码拒绝（53.0%）**，
真喝到 117 次。第一轮的拒绝率是 71.3%，本轮低一些但仍然过半。
代价是 L1 死亡 3 → 8、局末存活 28 → 20、杀伤 6353 → 5938。
看 §十 的点名种子最直观：`2133027` 合成臂死在 L2 第 13774 拍（159 杀），
加了喝药条之后**死在 L1 第 883 拍、clvl 1、20 杀、belt 0**；
`2133029` 合成臂活着（189 杀），加喝药条后**死在 L1 第 2821 拍、belt 0**；
`2133038` 同理（活着 → L1 第 1459 拍）。
**它把 belt 提前烧掉，然后在 L1 早期空手送命。**
（第一轮 §十 缺口 8 点名的 `2133027` / `2133029` 两局「零脱离拍却死在 L1」，
**本轮的答案就是喝药条**：这两局在 `of-potion` 里脱离拍也是 0，照样死在 L1。）

**接触脱离条（几乎无效）**：193 拍触发、132 拍开火、
**87/128 = 68.0% 原地没动**，实测分离增益 31 格 / 132 拍 = **0.235 格/拍**。
第一轮是 0.180 格/拍、63.5% 不动；**去掉人群条只是把量从 859 拍降到 132 拍，
毛病一点没治**。400 拍围栏本轮**一次都没够到**（单局最多 37 拍 = 围栏的 9.2%），
8 局整局不到 400 个工人拍——**围栏这一轮仍然没有被测到**。
对结果的影响：L1 死亡种子集合与 `of-potion` **完全相同**，局末存活 1 救 1 亏。

**拾药条（对 L1 无害，但炸窗口）**：见 §九。

---

## 九、RESUPPLY 窗口爆炸：归因定案

| arm | windows total | farm | dive | resupply | forced_dive | a13 total (both slots) | a12 total | env beats (all windows) | farm+dive beats | worker calls | worker_no_effect_requests (farm+dive) | no-effect share (farm+dive beats) | no-effect share (per worker call) | no-effect share (OLD, all windows -- NOT comparable) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| parent | 3815 | 1341 | 763 | 1711 | 497 | 584 | 55 | 142594 | 76347 | 76228 | 14326 | 0.1876 | 0.1879 | 0.1005 |
| oracle | 4396 | 1688 | 973 | 1735 | 718 | 909 | 0 | 114213 | 51597 | 51381 | 22828 | 0.4424 | 0.4443 | 0.1999 |
| oracle-farm | 4934 | 1921 | 1012 | 2001 | 552 | 629 | 32 | 171010 | 90582 | 90418 | 27469 | 0.3033 | 0.3038 | 0.1606 |
| of-potion | 4539 | 1730 | 878 | 1931 | 483 | 642 | 146 | 150699 | 79934 | 79881 | 21904 | 0.274 | 0.2742 | 0.1453 |
| of-potion-contact | 4438 | 1687 | 838 | 1913 | 473 | 643 | 137 | 145037 | 76014 | 75963 | 20234 | 0.2662 | 0.2664 | 0.1395 |
| oracle-pickup | 10560 | 1467 | 1332 | 7761 | 665 | 698 | 0 | 129984 | 54483 | 54273 | 27358 | 0.5021 | 0.5041 | 0.2105 |

  denominator (RR2 finding 1): worker_no_effect_share = farm+dive worker_no_effect_requests / farm+dive env beats  [RR2 finding 1]
  worker calls = farm+dive (beats + fuse_trips - drain_attempts - recovery_actions), and it equals the two WorkerProbes' own measured count on every arm (env_audit.worker_calls_ledger_closes); the identity is pinned window by window on a real episode by test_teacher_v2.routing_real.

  per-mode no-effect audit (env's own ledger):
  parent: {"dive": {"beats": 41278, "drain_attempts": 18, "drains": 18, "executed_requests": 28941, "fuse_trips": 213, "no_effect_requests": 10708, "overrides": 213, "recovery_actions": 213, "windows": 763, "worker_no_effect_requests": 10520}, "farm": {"beats": 35069, "drain_attempts": 101, "drains": 101, "executed_requests": 30648, "fuse_trips": 93, "no_effect_requests": 3899, "overrides": 93, "recovery_actions": 93, "windows": 1341, "worker_no_effect_requests": 3806}, "resupply": {"beats": 66247, "drain_attempts": 0, "drains": 0, "executed_requests": 0, "fuse_trips": 0, "no_effect_requests": 0, "overrides": 0, "recovery_actions": 0, "windows": 1711, "worker_no_effect_requests": 0}}
  oracle: {"dive": {"beats": 18514, "drain_attempts": 39, "drains": 39, "executed_requests": 4298, "fuse_trips": 567, "no_effect_requests": 14216, "overrides": 567, "recovery_actions": 567, "windows": 973, "worker_no_effect_requests": 13649}, "farm": {"beats": 33083, "drain_attempts": 177, "drains": 177, "executed_requests": 23526, "fuse_trips": 378, "no_effect_requests": 9557, "overrides": 378, "recovery_actions": 378, "windows": 1688, "worker_no_effect_requests": 9179}, "resupply": {"beats": 62616, "drain_attempts": 0, "drains": 0, "executed_requests": 2, "fuse_trips": 0, "no_effect_requests": 0, "overrides": 0, "recovery_actions": 0, "windows": 1735, "worker_no_effect_requests": 0}}
  oracle-farm: {"dive": {"beats": 51676, "drain_attempts": 52, "drains": 52, "executed_requests": 37939, "fuse_trips": 200, "no_effect_requests": 11748, "overrides": 200, "recovery_actions": 200, "windows": 1012, "worker_no_effect_requests": 11586}, "farm": {"beats": 38906, "drain_attempts": 112, "drains": 112, "executed_requests": 22367, "fuse_trips": 656, "no_effect_requests": 16539, "overrides": 656, "recovery_actions": 656, "windows": 1921, "worker_no_effect_requests": 15883}, "resupply": {"beats": 80428, "drain_attempts": 0, "drains": 0, "executed_requests": 4, "fuse_trips": 0, "no_effect_requests": 0, "overrides": 0, "recovery_actions": 0, "windows": 2001, "worker_no_effect_requests": 0}}
  of-potion: {"dive": {"beats": 46540, "drain_attempts": 25, "drains": 25, "executed_requests": 33600, "fuse_trips": 197, "no_effect_requests": 11267, "overrides": 197, "recovery_actions": 197, "windows": 878, "worker_no_effect_requests": 11105}, "farm": {"beats": 33394, "drain_attempts": 28, "drains": 28, "executed_requests": 22150, "fuse_trips": 445, "no_effect_requests": 11244, "overrides": 445, "recovery_actions": 445, "windows": 1730, "worker_no_effect_requests": 10799}, "resupply": {"beats": 70765, "drain_attempts": 0, "drains": 0, "executed_requests": 28, "fuse_trips": 0, "no_effect_requests": 0, "overrides": 0, "recovery_actions": 0, "windows": 1931, "worker_no_effect_requests": 0}}
  of-potion-contact: {"dive": {"beats": 44001, "drain_attempts": 21, "drains": 21, "executed_requests": 31236, "fuse_trips": 196, "no_effect_requests": 11169, "overrides": 196, "recovery_actions": 196, "windows": 838, "worker_no_effect_requests": 11007}, "farm": {"beats": 32013, "drain_attempts": 30, "drains": 30, "executed_requests": 22409, "fuse_trips": 378, "no_effect_requests": 9604, "overrides": 378, "recovery_actions": 378, "windows": 1687, "worker_no_effect_requests": 9227}, "resupply": {"beats": 69023, "drain_attempts": 0, "drains": 0, "executed_requests": 28, "fuse_trips": 0, "no_effect_requests": 0, "overrides": 0, "recovery_actions": 0, "windows": 1913, "worker_no_effect_requests": 0}}
  oracle-pickup: {"dive": {"beats": 27167, "drain_attempts": 71, "drains": 71, "executed_requests": 4629, "fuse_trips": 899, "no_effect_requests": 22538, "overrides": 899, "recovery_actions": 898, "windows": 1332, "worker_no_effect_requests": 21641}, "farm": {"beats": 27316, "drain_attempts": 140, "drains": 140, "executed_requests": 21369, "fuse_trips": 230, "no_effect_requests": 5947, "overrides": 230, "recovery_actions": 230, "windows": 1467, "worker_no_effect_requests": 5717}, "resupply": {"beats": 75501, "drain_attempts": 0, "drains": 0, "executed_requests": 1, "fuse_trips": 427, "no_effect_requests": 10693, "overrides": 427, "recovery_actions": 428, "windows": 7761, "worker_no_effect_requests": 0}}

  round1 parent: windows total 3815, resupply 1711, a13 n/a, a12 n/a, alive 28, l1_deaths 8
  round1 oracle: windows total 4396, resupply 1735, a13 909, a12 n/a, alive 12, l1_deaths 3
  round1 teacher-v1: windows total 10244, resupply 7605, a13 2088, a12 159, alive 7, l1_deaths 10

**结论：`ORACLE-PICKUP` 一条就复现了第一轮的整个窗口爆炸。**
神谕 base 两边窗 ＋ **只加拾药条** → RESUPPLY 窗 1735 → **7761**、
总窗 4396 → **10560**、`mask_forced` 2453 → **8426**；
第一轮三条本能条一起上是 7605 / 10244 / 8293。**几乎逐位重现。**

**机制**（引擎代码 ＋ 本轮审计，两头对得上）：

- `options_env.py:1082`：`m[RESUPPLY] = bool(controller_mask[13])`——
  **RESUPPLY 窗合法 ⟺ a13 合法**（半径 12 内有可见可达的药）。
- `options_env.py:411-422` `_farm_handoff`：**有剧情目标且半径 6 内无可战怪 ⇒ `m[FARM] = False`**。
- `resource_option_choice`（`:1422`）在不下楼时按 `FARM → RESUPPLY → DIVE` 取第一个合法项。
- 拾药条是个**多拍走路宏**：它把角色从怪身边带走去捡地上的药。
  一旦走开，`_farm_handoff` 成立 → FARM 被掩码 → 药还在地上（a13 仍合法）→
  **经理只剩 RESUPPLY 可选** → 脚本服务跑一扇极短的窗 → 收窗 → 重复。
- 数字对得上：`ORACLE-PICKUP` 的 RESUPPLY 是 75501 拍 / 7761 扇 = **9.7 拍一扇**，
  神谕是 62616 / 1735 = **36.1 拍一扇**——**扇数炸了，每扇变短**，正是这个循环的形状。
  而且只有这条臂的 RESUPPLY 窗里出现了 `fuse_trips 427` 与 `no_effect_requests 10693`
  （其余五条臂的 RESUPPLY 这两项都是 0）——**脚本服务在空转**。

**一个必须说出来的反直觉事实**：爆炸**不是** a13 变多造成的。
`ORACLE-PICKUP` 的 a13 总数是 **698**，比不带拾药条的 `ORACLE` 的 **909 还少**
（拾药条开火 695 次、被接触挡掉 736 次；冻结 base 自己只剩 3 次想要 a13）。
**所以第一轮「a13 从 857 涨到 2088，所以是拾药条」的推理链条是错的，结论碰巧是对的**：
真正的原因是**这个宏把角色带离怪群**，不是它按了多少次。

**神谕在 DIVE 窗里的空转**（同一张审计表，另一条更重要的读数）：

| 臂 | DIVE 窗拍数 | 真执行 | 空请求 | fuse | 真执行率 |
|---|---|---|---|---|---|
| parent | 41278 | 28941 | 10708 | 213 | **70.1%** |
| **oracle** | 18514 | **4298** | 14216 | 567 | **23.2%** |
| **oracle-farm**（DIVE 归学生） | 51676 | 37939 | 11748 | 200 | **73.4%** |
| oracle-pickup | 27167 | 4629 | 22538 | 899 | **17.0%** |

**神谕的 DIVE 窗有 77% 的拍是没有效果的请求。**
这是「神谕会打架、不会下楼」的定量形状，也是合成臂为什么有效的机制解释。
（这张小表一直用的是**逐窗种**的分母，所以它是对的；被复核轮挑出来的是上面那张大表的
**总计**那一列，见下。）

**分母更正（复核轮 finding 1）**：§九 大表原先只有一列「no-effect share」，
分母是 env 在**全部三种窗**上的拍数——**RESUPPLY 也算在内，而 RESUPPLY 窗根本不叫 worker**
（`options_env.py:2815` 对 option 2 没有条目；逐窗审计里六条臂的 resupply
`worker_no_effect_requests` 全是 0）。RESUPPLY 占全部拍数的比例在各臂之间差得很远
（parent 46%、oracle-pickup 58%），**所以那一列恰恰在它被印出来做比较的那些臂之间不可比**。
现在表里印四个分母，旧的那一列保留但改名并标注不可比：

- **env beats (all windows)**：旧口径，仅作存档；
- **farm+dive beats**：worker 拥有的窗里的全部拍（含脚本拍）；
- **worker calls**：worker callable**真正被问了多少次**——由两个 `WorkerProbe` 直接数出来，
  并由 env 自己的账 `beats + fuse_trips − drain_attempts − recovery_actions` 逐窗复算对上
  （`teacher_v2.observing_env_class()._AUDIT_IDENTITY`，由 `test_teacher_v2.routing_real`
  在一整局真实局上逐窗钉死）；
- 相应的两列 share。

**换成诚实分母之后，名次和倍数都变了**：
parent 0.1876、oracle **0.4424**、oracle-farm 0.3033、of-potion 0.2740、
of-potion-contact 0.2662、oracle-pickup **0.5021**。
旧口径下 oracle-farm 0.1606 对 oracle-pickup 0.2105 看起来是 1.3 倍，
**真实的 worker 侧差距是 1.7 倍**。
（结论不因此改变：§零 的机制论证用的一直是逐窗种分母 18514 / 4298。）

---

## 十、点名的种子

| seed | parent | oracle | oracle-farm | of-potion | of-potion-contact | oracle-pickup |
|---|---|---|---|---|---|---|
| 2133002 | alive, depth 1, kills 132, clvl 3 | alive, depth 2, kills 126, clvl 3 | alive, depth 1, kills 136, clvl 4 | alive, depth 1, kills 134, clvl 3 | alive, depth 1, kills 134, clvl 3 | DIED L3@10293 belt0, depth 4, kills 154, clvl 4 |
| 2133021 | DIED L2@7387 belt4, depth 2, kills 103, clvl 3 | DIED L1@2563 belt0, depth 1, kills 68, clvl 2 | DIED L1@2563 belt0, depth 1, kills 68, clvl 2 | DIED L1@608 belt0, depth 1, kills 19, clvl 1 | DIED L1@645 belt0, depth 1, kills 19, clvl 1 | DIED L1@2563 belt0, depth 1, kills 68, clvl 2 |
| 2133022 | alive, depth 2, kills 142, clvl 4 | alive, depth 1, kills 92, clvl 3 | alive, depth 2, kills 136, clvl 4 | DIED L4@14278 belt8, depth 4, kills 134, clvl 4 | DIED L4@14278 belt8, depth 4, kills 134, clvl 4 | DIED L6@10063 belt0, depth 6, kills 116, clvl 3 |
| 2133038 | alive, depth 2, kills 162, clvl 4 | alive, depth 3, kills 178, clvl 5 | alive, depth 1, kills 127, clvl 3 | DIED L1@1459 belt0, depth 1, kills 44, clvl 2 | DIED L1@1496 belt0, depth 1, kills 47, clvl 2 | alive, depth 3, kills 178, clvl 5 |
| 2133006 | alive, depth 2, kills 148, clvl 4 | DIED L2@7942 belt0, depth 2, kills 107, clvl 3 | DIED L2@10532 belt7, depth 2, kills 111, clvl 3 | DIED L2@12120 belt3, depth 2, kills 146, clvl 4 | DIED L2@12120 belt3, depth 2, kills 146, clvl 4 | DIED L3@9995 belt8, depth 3, kills 142, clvl 4 |
| 2133039 | DIED L1@2047 belt0, depth 1, kills 69, clvl 2 | DIED L2@6698 belt4, depth 2, kills 81, clvl 3 | alive, depth 1, kills 72, clvl 3 | DIED L1@3941 belt0, depth 1, kills 48, clvl 2 | DIED L1@3941 belt0, depth 1, kills 48, clvl 2 | DIED L2@7732 belt5, depth 2, kills 45, clvl 2 |
| 2133014 | alive, depth 2, kills 168, clvl 5 | DIED L2@9799 belt0, depth 3, kills 116, clvl 3 | DIED L3@13890 belt0, depth 3, kills 187, clvl 6 | DIED L2@14672 belt0, depth 3, kills 174, clvl 5 | alive, depth 2, kills 184, clvl 5 | DIED L2@10113 belt0, depth 3, kills 116, clvl 3 |
| 2133010 | alive, depth 2, kills 171, clvl 4 | DIED L3@9285 belt0, depth 4, kills 135, clvl 4 | DIED L2@13190 belt8, depth 2, kills 146, clvl 4 | alive, depth 2, kills 169, clvl 4 | alive, depth 2, kills 169, clvl 4 | DIED L3@9196 belt5, depth 4, kills 136, clvl 4 |
| 2133027 | DIED L2@12214 belt0, depth 2, kills 125, clvl 3 | DIED L4@7254 belt4, depth 4, kills 119, clvl 3 | DIED L2@13774 belt3, depth 2, kills 159, clvl 4 | DIED L1@883 belt0, depth 1, kills 20, clvl 1 | DIED L1@883 belt0, depth 1, kills 20, clvl 1 | DIED L4@7341 belt4, depth 4, kills 118, clvl 3 |
| 2133029 | DIED L2@9762 belt8, depth 2, kills 121, clvl 3 | DIED L2@12293 belt0, depth 2, kills 141, clvl 4 | alive, depth 2, kills 189, clvl 5 | DIED L1@2821 belt0, depth 1, kills 68, clvl 2 | DIED L1@2821 belt0, depth 1, kills 68, clvl 2 | DIED L2@12714 belt0, depth 2, kills 142, clvl 4 |

### 10.1 点名种子上的逐条本能活动

  2133002 oracle: potion 0(refused 0) pickup 0 disengage 0 moved 0/0
  2133002 oracle-farm: potion 0(refused 0) pickup 0 disengage 0 moved 0/0
  2133002 of-potion: potion 1(refused 0) pickup 0 disengage 0 moved 0/0
  2133002 of-potion-contact: potion 1(refused 0) pickup 0 disengage 0 moved 0/0
  2133002 oracle-pickup: potion 0(refused 0) pickup 45 disengage 0 moved 0/0
  2133021 oracle: potion 0(refused 0) pickup 0 disengage 0 moved 0/0
  2133021 oracle-farm: potion 0(refused 0) pickup 0 disengage 0 moved 0/0
  2133021 of-potion: potion 1(refused 2) pickup 0 disengage 0 moved 0/0
  2133021 of-potion-contact: potion 1(refused 2) pickup 0 disengage 2 moved 1/2
  2133021 oracle-pickup: potion 0(refused 0) pickup 0 disengage 0 moved 0/0
  2133022 oracle: potion 0(refused 0) pickup 0 disengage 0 moved 0/0
  2133022 oracle-farm: potion 0(refused 0) pickup 0 disengage 0 moved 0/0
  2133022 of-potion: potion 3(refused 14) pickup 0 disengage 0 moved 0/0
  2133022 of-potion-contact: potion 3(refused 14) pickup 0 disengage 0 moved 0/0
  2133022 oracle-pickup: potion 0(refused 0) pickup 15 disengage 0 moved 0/0
  2133038 oracle: potion 0(refused 0) pickup 0 disengage 0 moved 0/0
  2133038 oracle-farm: potion 0(refused 0) pickup 0 disengage 0 moved 0/0
  2133038 of-potion: potion 2(refused 0) pickup 0 disengage 0 moved 0/0
  2133038 of-potion-contact: potion 2(refused 0) pickup 0 disengage 10 moved 5/10
  2133038 oracle-pickup: potion 0(refused 0) pickup 9 disengage 0 moved 0/0
  2133006 oracle: potion 0(refused 0) pickup 0 disengage 0 moved 0/0
  2133006 oracle-farm: potion 0(refused 0) pickup 0 disengage 0 moved 0/0
  2133006 of-potion: potion 3(refused 0) pickup 0 disengage 0 moved 0/0
  2133006 of-potion-contact: potion 3(refused 0) pickup 0 disengage 0 moved 0/0
  2133006 oracle-pickup: potion 0(refused 0) pickup 16 disengage 0 moved 0/0
  2133039 oracle: potion 0(refused 0) pickup 0 disengage 0 moved 0/0
  2133039 oracle-farm: potion 0(refused 0) pickup 0 disengage 0 moved 0/0
  2133039 of-potion: potion 1(refused 0) pickup 0 disengage 0 moved 0/0
  2133039 of-potion-contact: potion 1(refused 0) pickup 0 disengage 0 moved 0/0
  2133039 oracle-pickup: potion 0(refused 0) pickup 1 disengage 0 moved 0/0
  2133014 oracle: potion 0(refused 0) pickup 0 disengage 0 moved 0/0
  2133014 oracle-farm: potion 0(refused 0) pickup 0 disengage 0 moved 0/0
  2133014 of-potion: potion 3(refused 0) pickup 0 disengage 0 moved 0/0
  2133014 of-potion-contact: potion 2(refused 0) pickup 0 disengage 3 moved 2/3
  2133014 oracle-pickup: potion 0(refused 0) pickup 10 disengage 0 moved 0/0
  2133010 oracle: potion 0(refused 0) pickup 0 disengage 0 moved 0/0
  2133010 oracle-farm: potion 0(refused 0) pickup 0 disengage 0 moved 0/0
  2133010 of-potion: potion 2(refused 3) pickup 0 disengage 0 moved 0/0
  2133010 of-potion-contact: potion 2(refused 3) pickup 0 disengage 0 moved 0/0
  2133010 oracle-pickup: potion 0(refused 0) pickup 19 disengage 0 moved 0/0
  2133027 oracle: potion 0(refused 0) pickup 0 disengage 0 moved 0/0
  2133027 oracle-farm: potion 0(refused 0) pickup 0 disengage 0 moved 0/0
  2133027 of-potion: potion 1(refused 0) pickup 0 disengage 0 moved 0/0
  2133027 of-potion-contact: potion 1(refused 0) pickup 0 disengage 0 moved 0/0
  2133027 oracle-pickup: potion 0(refused 0) pickup 10 disengage 0 moved 0/0
  2133029 oracle: potion 0(refused 0) pickup 0 disengage 0 moved 0/0
  2133029 oracle-farm: potion 0(refused 0) pickup 0 disengage 0 moved 0/0
  2133029 of-potion: potion 3(refused 0) pickup 0 disengage 0 moved 0/0
  2133029 of-potion-contact: potion 3(refused 0) pickup 0 disengage 0 moved 0/0
  2133029 oracle-pickup: potion 0(refused 0) pickup 18 disengage 0 moved 0/0

### 10.2 L1 死亡的种子与当时的等级

  parent: 8 L1 deaths -> [(2133008, 2, 0), (2133009, 2, 0), (2133015, 3, 0), (2133017, 1, 0), (2133023, 3, 0), (2133024, 2, 0), (2133039, 2, 0), (2133044, 1, 0)]
  oracle: 3 L1 deaths -> [(2133017, 1, 0), (2133021, 2, 0), (2133026, 1, 0)]
  oracle-farm: 3 L1 deaths -> [(2133017, 1, 0), (2133021, 2, 0), (2133026, 1, 0)]
  of-potion: 8 L1 deaths -> [(2133009, 2, 0), (2133021, 1, 0), (2133027, 1, 0), (2133029, 2, 0), (2133033, 3, 0), (2133038, 2, 0), (2133039, 2, 0), (2133041, 1, 0)]
  of-potion-contact: 8 L1 deaths -> [(2133009, 2, 0), (2133021, 1, 0), (2133027, 1, 0), (2133029, 2, 0), (2133033, 3, 0), (2133038, 2, 0), (2133039, 2, 0), (2133041, 1, 0)]
  oracle-pickup: 4 L1 deaths -> [(2133011, 2, 0), (2133021, 2, 0), (2133026, 1, 0), (2133036, 2, 0)]

`ORACLE` 与 `ORACLE-FARM` 的三个 L1 死亡种子**完全相同**；
`OF-POTION` 与 `OF-POTION-CONTACT` 的八个也**完全相同**。
所有 L1 死亡（六条臂共 34 局）**无一例外 belt = 0**。

---

## 十一、杀伤与深度有没有被打崩

| arm | alive | kills_total | depth_mean | L3 reach | clvl_ge_4 | median micro_steps |
|---|---|---|---|---|---|---|
| parent | 28 | 5550 | 1.667 | 2 | 18 | 12000.0 |
| oracle | 12 | 5469 | 2.646 | 26 | 20 | 9302.0 |
| oracle-farm | 28 | 6353 | 1.833 | 5 | 31 | 13420.5 |
| of-potion | 20 | 5938 | 1.833 | 4 | 27 | 12433.0 |
| of-potion-contact | 20 | 5917 | 1.833 | 4 | 26 | 12083.5 |
| oracle-pickup | 9 | 5295 | 2.812 | 29 | 24 | 9268.5 |

`ORACLE-FARM`：存活 28（＝学生）、杀伤 6353（＞学生 5550）、clvl≥4 31 局（＞学生 18）、
中位微拍 13420（＞学生 12000）。**没有任何一项被打崩，多数项还更好。**
`OF-POTION` / `OF-POTION-CONTACT`：存活掉到 20，杀伤仍高于学生。
`ORACLE-PICKUP`：存活 9、杀伤 5295，**和第一轮神谕一样是「冲下去送死」的形状**。

---

## 十二、只读证明与种子枚举

  distinct seeds across every round-2 artefact: 48, min 2133000, max 2133047
  contiguous 2133000-2133047: True
  any seed in virgin pools 2_116-119 / 2_126-128: False
  seeds per arm: {'parent': 48, 'oracle': 48, 'oracle-farm': 48, 'of-potion': 48, 'of-potion-contact': 48, 'oracle-pickup': 48}

- 主树 `tree_fp.sh` 指纹（283 项 ＋ build 链接 ＋ `_diablogym*.so`）
  在本轮开工前与收工后**逐字节相同**：`round2/main-tree-fingerprint-{mid,after}.txt`
  与 `r19/main-tree-fingerprint-before-round2.txt` `diff` 为空。
- `m2-merge`、`oracle-tree` 在本轮时间窗内**没有任何文件被修改**（`find -newermt` 为空）。
- `oracle2-tree` 里的 `teacher_v1.py` / `test_teacher_v1.py` 的 sha256
  与 `oracle-tree` 的**逐位相同**（`65c1a14f…` / `f12d94d9…`）。
- 本轮全部产物里出现的种子：**恰好 48 个，2133000–2133047 连续**，
  与处女池 2_116-119 / 2_126-128 的交集为**空**。

---

## 十三、单元卷（`round2/tests-full-review.log`，85 项全过、0 项失败）

| 组 | 覆盖 |
|---|---|
| 路由（合成） | `workers` 只有 FARM/DIVE 两项、两格是不同 callable、FARM 是神谕/DIVE 是学生、RESUPPLY 无 worker；驱动 FakeEnv 逐窗验证「哪个 callable 服务了哪种窗」；`on_beat` 扇出且每拍恰好一次；`episode_reseed` 扇出到学生；action-12 契约；`bind` 拒绝 `drink_sovereignty` 不一致的 env；路由对象拒绝被当 worker 调用 |
| 路由（真实局）**复核轮加严** | 一整局真 `OptionsEnv`：神谕格只有 farm 拍、学生格只有 dive 拍、`resupply` 拍为 0、观测 env 的窗口账与探针窗口账一致、`forced_dive` 可按 `window_id` 连接；**并且**（复核轮 finding 2 之前是三条弱断言）：两格的拍数**等于** env 自己的 worker 调用账 `beats + fuse_trips − drain_attempts − recovery_actions`；学生格服务的窗口 id 集合与 env 的 dive 窗口集合**集合相等**、神谕格与 farm 窗口集合**集合相等**；每一扇 farm/dive 窗**逐窗**满足那条恒等式；探针计到的窗 ＋ 零调用窗 ＝ 探针窗口数；RESUPPLY 窗有拍但 env 自己的 worker 计数器全 0 |
| 排名等价（复核轮新增） | `_disengage_move_contact` 是冻结 `TeacherV1._disengage_move` 去掉楼梯 tie-break 的手抄本：**1200 个随机状态上两者动作与增益逐个相同**（在 `STAIRS_TIEBREAK_RADIUS` 内没有上楼梯时），且在有楼梯时两者**必须不同**（否则该测试是空的）——所以下一轮谁改了冻结的排名，这条会红 |
| 臂名遥测（复核轮新增） | 三条条款臂的 `telemetry["variant"]` 是**臂名**而不是底座名 `oracle`（finding 7），`oracle` / `oracle-farm` 仍报 `oracle` |
| 条款真值表 | 喝药（belt/0.60 严格小于/掩码拒绝计数）、拾药（Chebyshev 1 边界、半径 2 不挡、暗处怪不挡）、接触脱离（0.35 双边界、**人群不触发**、**楼梯不改变选择**、地形筛选含 hazard/door、条款次序、400 拍安全帽）、gear-grace 仍由冻结的第 0 层回答、每条臂只跑自己的条款 |
| 合法性 | 3 套条款 × 900 个随机状态 = **2700** 个状态，emit 的动作**永远在掩码内** |
| 确定性 | 四条新臂，每条同一种子跑两遍，**动作序列逐位相同、`rows_sha_v3` 相同** |

**一处必须自曝的记录错误（第一遍就自曝过，这里保留）**：ledger 的
`R19_TEACHER_ROUND2_BUILT` 那条里，`tests` 字段写的是
「Full run (incl. two real-episode legs): ==== 54 passed ====」，
那个 54 其实是 `--no-episode` 那一腿的数。
第一遍的完整单元卷是 **70 项全过**（`round2/tests-full.log`），
复核轮加严并新增之后是 **85 项全过**（`round2/tests-full-review.log`）。
ledger 只能追加不能改，所以更正写在 `R19_TEACHER_ROUND2_RESULT` 的 `corrections` 字段里，
本轮的数写在 `R19_TEACHER_ROUND2_REVIEW_ROUND` 里，也写在这里。
**本报告里的两个测试计数都是从日志里读出来的，不是手打的**（`mk_report._tests`）。

---

## 十四、方法学缺口（自曝）

1. **合成臂里装着学生。** `ORACLE-FARM` 不是一个纯脚本老师：它的 DIVE 窗就是学生本人。
   作为 BC/DAgger 的老师它是合法的（它是一个 worker callable），
   但**它能教给学生的只有 FARM 窗里的行为**；DIVE 窗上它和学生逐位相同，没有信息可教。
   「老师赢了学生」这句话的准确版本是：
   **「神谕的 FARM 内环 ＋ 学生的 DIVE 行为」赢了「学生的 FARM ＋ 学生的 DIVE」。**
2. **那条 L1 胜利只比线低 0.004，且与第一轮是同一个统计**（同样三个种子）。
   2_133 池今晚又被消费了 **4 次**（四条新臂），加上第一轮/复核轮已经很多次。
   **在处女池上复验之前不要建任何东西在它上面。**
3. **「局末存活」在深度不同的两条臂之间仍然不是公平的问句**，
   §5.1 的逐层危害表与「未到达 L2/L3」配对是补救，但没有跑「钉在同一深度」的匹配臂。
4. **接触脱离条仍有 68% 的拍原地不动**，本轮只是把量降下来；
   最可能的剩余原因仍是**落点上站着怪**，而读 `local_map["monster"]` 会突破
   「只看可见怪」的宪章——**要不要允许，仍然上呈**。
5. **400 拍围栏第三轮仍然没有被够到**（本轮单局最多 37 拍），
   8 局整局不到 400 个工人拍。**这个数合不合适仍然没有测到。**
6. **跨窗结算**：合成臂里神谕只服务 FARM 窗，所以在一扇 FARM 窗最后一拍开火的脱离，
   要等下一扇 FARM 窗才能结算「到底走动了没有」。本轮加了计数：
   128 次结算里 **127 次同窗、1 次跨窗**，所以这个偏差可以忽略——但它是存在的，记在这里。
7. **`grace_decisions = 0`、`base_fallbacks = 0` 两轮零对局证据**，只有单元卷覆盖。
8. ~~**引擎窗口账与探针拍账差 1**~~ —— **复核轮已结清，不再是缺口**。
   那个差额现在有精确的名字和精确的账：worker 拥有的窗里 `_win_beat` 只从三处被调用
   （脑干排水 `drain_attempts`、worker 自己的一拍、上一次 fuse 之后开窗那一拍脚本恢复
   `recovery_actions`），而跳闸的一拍在 `w["beats"] += 1` 之前就返回了，于是
   **worker_calls + drain_attempts + recovery_actions == beats + fuse_trips**。
   六条臂上这条恒等式逐臂闭合（`env_audit.worker_calls_ledger_closes`），
   真实局上逐窗闭合（§十三）。§九 的表现在同时印 env 口径与 worker 口径，
   并明说旧那一列不可比（finding 1）。
   （本轮世界里 `resource_emergency_stop` 是 `off`，所以 `options_env.py:2044`
   的 stop-v1 脚本尾巴一拍都没跑；若哪天打开，这条恒等式需要第四项。）
9. `sweep_gold` 六条臂皆 0（遥测口径问题，与第一轮相同），本轮不追。
10. **只测了一个世界（M2-SET）、一个池、48 个种子、`sample` 解码。** 没有别的。
11. **36 个配对检定没有做多重性校正**，越线的 7 个其实只是 3 件事（§六）。
12. **`_disengage_move_contact` 仍然是冻结排名的第二份拷贝**：复核轮加了一条
    1200 状态的等价测试把两份钉在一起（§十三），但**没有合并成一份**——
    合并要动冻结的 `teacher_v1.py`，本轮明令不许动。
    真正没修的老毛病仍然是那 **68% 原地不动**（见上面第 4 条）。

---

## 十五、给主席的建议（不是裁定）

1. **第一步过了，但过的是合成臂，不是纯脚本老师。**
   按裁定，「老师打赢父代」是 BC/DAgger 的前置闸。
   `ORACLE-FARM` 在 L1 死亡上过线（3 对 8，UCB95 −0.00437），
   **且没有牺牲局末存活（28 对 28）或杀伤（＋14.5%）**。
   但它的 DIVE 窗就是学生（§十四.1）。
   **建议：可以走第二步，但演示数据只取 FARM 窗的决策**——
   那才是老师真正多出来的东西。
2. **先在处女池上复验那条 L1 结论。** 它只比线低 0.004，翻一个种子就没了，
   而且和第一轮是同一个统计。**这是我认为最该先做的一件事。**
3. **喝药条建议直接砍掉，不要调参。**
   本轮证明它一条就把整个 L1 胜利吃光（3 → 8），
   而且 53% 的想喝被掩码拒绝——**它真正需要的是动 `drink_sovereignty` 的安全包络，
   那是法条改动，本轮明令不许碰**。在那之前，`0.60` 这条规则在 M2-SET 世界里
   只会做一件事：**把 belt 提前烧光**。
4. **接触脱离条建议也砍掉。** 它在同一个 base 上什么都没改变
   （L1 死亡种子集合一模一样），而 68% 的拍原地不动这个老毛病还在。
   要再试，仍然需要 §十四.4 的那条裁定（允不允许读物理占位通道）。
5. **拾药条：不要在「本能层」里做。** 它对 L1 无害，
   但它和 `m[RESUPPLY] = controller_mask[13]` ＋ `_farm_handoff` 一起
   形成了一个**每扇 9.7 拍的 RESUPPLY 空转循环**（7761 扇）。
   捡药这件事本来就是脚本服务的职权；让 worker 也去捡，只会让经理反复开短窗。
   **如果还想要它，该改的是经理侧的开窗条件，那是法条，不是老师。**
6. **下一轮如果只能做一件事**：把 `ORACLE-FARM` 原样搬到一个处女池上复验 L1
   （不改任何代码，只换池）。本轮的 `teacher_v2.py` 与驱动可以直接复用。

---

## 十六、复核轮（REVIEW ROUND，2026-09-11 00:22–01:27）

独立复核给出 **7 条 finding，全部为报告／测试／取证卫生，`verdict: ship`，
没有一条改变任何数字或结论**。复核轮把 7 条**全部**修了，
并且因为其中两条动到了遥测，**六条臂全部重跑**。

### 16.1 逐条

| # | 级别 | 问题 | 处置 |
|---|---|---|---|
| 1 | medium | §九 的 `no-effect share` 分母是**全部三种窗**的拍数，含从不叫 worker 的 RESUPPLY；RESUPPLY 占比在各臂间从 46% 到 58%，**所以这一列在它要比较的臂之间不可比** | **已修**：表里改印 `env beats (all windows)`（存档、标注不可比）、`farm+dive beats`、`worker calls`（探针直接数出、并由 env 自己的账逐窗复算）以及两个诚实分母的 share。数值：parent 0.1876 / oracle 0.4424 / oracle-farm 0.3033 / of-potion 0.2740 / of-potion-contact 0.2662 / oracle-pickup 0.5021 |
| 2 | medium | `test_teacher_v2.routing_real` 里三条断言**弱于它们自己的名字**（`> 0`、`<=`、`>=`），其中 `<=` 那条即使 12 扇 dive 窗只有 1 扇走了学生也会 PASS | **已修**：改成集合相等与精确恒等式（见 §十三）。为了写得出精确恒等式，观测 env 新增了 `drain_attempts` / `drains` / `recovery_actions` 三个逐窗计数 |
| 3 | medium | §零 的一句话 VERDICT「老师第一次赢过了学生」读起来比这条臂能撑的多：这条臂 57.1% 的拍是学生打的，且那条 L1 统计与第一轮是同一个统计 | **已修**：§零 的头一句改写成实际被证明的那句，并把「57.1% 是学生」「同样三个种子」「`oracle-farm vs oracle` 的 L1 配对是 0/0/0」放进头一句本身；ledger 只能追加，更正写进 `R19_TEACHER_ROUND2_REVIEW_ROUND` 的 `corrections` |
| 4 | low | 两张回归收据的 mtime（23:59）**晚于**它们本该把关的四条臂（23:49–23:53），因为 `--reuse` 汇总腿把它们覆盖重写了；闸脚本自己也是收尾时才归档的 | **已修**：`--reuse` 改写兄弟文件 `regression-*-reuse.json`（`"pass": "reuse-summarize"`），闸收据只由真跑那一腿写（`"pass": "run"`）；`round2/review-round-manifest-before.txt` 在**启动前**把每个将要运行的脚本 sha256 ＋ 时间戳存档（§四） |
| 5 | low | 36 个单侧 95% 检定没有多重性说明；越线的 7 个里多数不独立 | **已修**：§六 开头给出零假设下的期望越线数 1.8，并列表指出那 7 个其实是 3 件事 |
| 6 | low | `_disengage_move_contact` 是冻结排名的手抄本，冻结那份将来被修时这份不会跟着修，且没有任何测试会红 | **已修（钉住，未合并）**：新增 1200 状态的纯 Python 等价测试 ＋ 一条「有楼梯时必须不同」的非空性测试。合并成一份要动冻结的 `teacher_v1.py`，本轮不许 |
| 7 | low | `ClauseTeacher` 把 `variant="oracle"` 传给底座，于是三条条款臂每一行的 `telemetry["variant"]` 都写着 `oracle` | **已修**：`ClauseTeacher` 在 `super().__init__` 之后把 `self.variant` 改成臂名，并在 `_blank_telemetry` 里再写一次。`self.variant` 在冻结的 `teacher_v1` 里只被两处读到（遥测 `:282`、strict 断言文案 `:350`），**从不作为决策输入**——这一点由「六条臂 SHA 全部不变」实测证明 |

### 16.2 重跑（finding 2 与 finding 7 动了遥测，所以六条臂全跑）

**两道回归闸都过，四条新臂的 SHA 也逐位不变**——即上面所有改动都是**纯遥测／纯报告**，
没有一条改变过任何一次决策：

| arm | pass1 (first run, 2026-09-10 23:42-23:53) | pass2 (review round, 2026-09-11 00:31-00:53) | pass3 (review round final, 2026-09-11 01:00-) | all equal | expected (regression) |
|---|---|---|---|---|---|
| parent | f33f7af5f236f1fc | f33f7af5f236f1fc | f33f7af5f236f1fc | True | f33f7af5f236f1fc == expected |
| oracle | 1507c3be68a0da29 | 1507c3be68a0da29 | 1507c3be68a0da29 | True | 1507c3be68a0da29 == expected |
| oracle-farm | e47d5a5850ae1c46 | e47d5a5850ae1c46 | e47d5a5850ae1c46 | True | (new arm, no chairman constant) |
| of-potion | 17d3ea46f63f88a6 | 17d3ea46f63f88a6 | 17d3ea46f63f88a6 | True | (new arm, no chairman constant) |
| of-potion-contact | 525d18ef98ddbe8d | 525d18ef98ddbe8d | 525d18ef98ddbe8d | True | (new arm, no chairman constant) |
| oracle-pickup | 3f02ce54b8922626 | 3f02ce54b8922626 | 3f02ce54b8922626 | True | (new arm, no chairman constant) |

  every arm identical across all three passes: True

  arm_stats identical across the three passes (the whole dict, not just the SHA):
  parent: True
  oracle: True
  oracle-farm: True
  of-potion: True
  of-potion-contact: True
  oracle-pickup: True

  per-arm receipts of the FINAL pass:
| arm | elapsed_s | runtime_errors | source_changed_during_run | worker calls (probe) | worker calls (env ledger) | ledger closes |
|---|---|---|---|---|---|---|
| parent | 592.5 | 0 | [] | 76228 | 76228 | True |
| oracle | 511.2 | 0 | [] | 51381 | 51381 | True |
| oracle-farm | 686.7 | 0 | [] | 90418 | 90418 | True |
| of-potion | 640.4 | 0 | [] | 79881 | 79881 | True |
| of-potion-contact | 648.5 | 0 | [] | 75963 | 75963 | True |
| oracle-pickup | 681.7 | 0 | [] | 54273 | 54273 | True |

  the two GATE receipts of the final pass (written by the RUN leg, never by the --reuse summarize leg):
  parent: equal=True pass='run' all_references_agree=True sha=f33f7af5f236f1fc
  oracle: equal=True pass='run' all_references_agree=True sha=1507c3be68a0da29

  telemetry variant label per arm (RR2 finding 7), row 0 of each:
  parent: farm_policy.telemetry['variant'] = None, v2_label = None
  oracle: farm_policy.telemetry['variant'] = 'oracle', v2_label = None
  oracle-farm: farm_policy.telemetry['variant'] = 'oracle', v2_label = None
  of-potion: farm_policy.telemetry['variant'] = 'of-potion', v2_label = 'of-potion'
  of-potion-contact: farm_policy.telemetry['variant'] = 'of-potion-contact', v2_label = 'of-potion-contact'
  oracle-pickup: farm_policy.telemetry['variant'] = 'oracle-pickup', v2_label = 'oracle-pickup'

  L1 death seed sets (RR2 finding 3: oracle and oracle-farm are the SAME statistic):
  parent: [2133008, 2133009, 2133015, 2133017, 2133023, 2133024, 2133039, 2133044]
  oracle: [2133017, 2133021, 2133026]
  oracle-farm: [2133017, 2133021, 2133026]
  of-potion: [2133009, 2133021, 2133027, 2133029, 2133033, 2133038, 2133039, 2133041]
  of-potion-contact: [2133009, 2133021, 2133027, 2133029, 2133033, 2133038, 2133039, 2133041]
  oracle-pickup: [2133011, 2133021, 2133026, 2133036]

  who actually played (RR2 finding 3), worker calls by slot:
  parent: total 76228 -- farm_slot(parent) 76228 = 100.0% [farm 34968 / dive 41260 / resupply 0; forced_dive beats 31801]
  oracle: total 51381 -- farm_slot(oracle) 51381 = 100.0% [farm 32906 / dive 18475 / resupply 0; forced_dive beats 16007]
  oracle-farm: total 90418 -- dive_slot(parent) 51624 = 57.1% [farm 0 / dive 51624 / resupply 0; forced_dive beats 33901], farm_slot(oracle) 38794 = 42.9% [farm 38794 / dive 0 / resupply 0; forced_dive beats 0]
  of-potion: total 79881 -- dive_slot(parent) 46515 = 58.2% [farm 0 / dive 46515 / resupply 0; forced_dive beats 31831], farm_slot(oracle) 33366 = 41.8% [farm 33366 / dive 0 / resupply 0; forced_dive beats 0]
  of-potion-contact: total 75963 -- dive_slot(parent) 43980 = 57.9% [farm 0 / dive 43980 / resupply 0; forced_dive beats 30983], farm_slot(oracle) 31983 = 42.1% [farm 31983 / dive 0 / resupply 0; forced_dive beats 0]
  oracle-pickup: total 54273 -- farm_slot(oracle) 54273 = 100.0% [farm 27176 / dive 27097 / resupply 0; forced_dive beats 15095]

### 16.3 复核轮的读写纪律

- 主树 `/home/laure/AlphaDiablo/diablogym`：283 条指纹 ＋ `build` 符号链接 ＋ `_diablogym*.so`
  与 `r19/main-tree-fingerprint-before-round2.txt` **diff 为空**
  （`round2/main-tree-fingerprint-review-before.txt` 与 `…-review-after.txt`）；
  唯一新于 23:14 的主树文件仍然只有 `r13_ledger.jsonl`。
- `m2-merge` 与 `oracle-tree`：`find … -newermt '2026-09-10 23:14'` **为空**。
- `oracle2-tree` 里 `teacher_v1.py`（`65c1a14f…`）与 `test_teacher_v1.py`（`f12d94d9…`）
  与只读的 `oracle-tree` **逐位相同**。
- 第一遍与第二遍的全部产物都原样留着：`round2/probe-pass1/`、`round2/probe-pass2/`、
  `round2/TEACHER-ROUND2-REPORT.pass1.md`、`round2/driver-*.pass1.log`；
  第一遍与最终那一遍的逐臂对照（SHA、整个 `arm_stats` 字典、配对块、点名种子）
  在 `round2/review-round-compare.txt`；三遍的 SHA 对照见 §16.2。
- 种子：仍然只有 2133000–2133047，六条臂各 48 个、逐臂集合相同；
  处女池 2_116-119 / 2_126-128 **零接触**。

### 16.4 复核轮之后的最终 VERDICT

**不变。** 七条 finding 没有改动任何一个数字，六条臂的 `rows_sha_v3` 全部不变。

> **有一条臂在 L1 死亡上过线：`ORACLE-FARM`**
> （L1 死亡 3 对 8，配对 saved 7 / lost 2 / net +5 / 单侧 UCB95 −0.00437），
> **同时没有把局末存活（28 对 28）和杀伤（5550 → 6353，＋14.5%）打崩**——
> 这是主席那句话的字面答案，答案是「有，且只有这一条」。
>
> **但它赢的是「神谕的 FARM 内环 ＋ 学生的 DIVE 行为」**，
> 这条臂 57.1% 的工人拍仍然是学生本人；
> **而且它的 L1 统计与第一轮的 `ORACLE` 是同一个统计**（同样三个种子，
> `oracle-farm vs oracle` 的 L1 配对是 0/0/0），只比 0 线低 0.004，
> 36 个未校正的检定里期望有 1.8 个纯靠运气越线。
> **在处女池上复验之前，不该有任何东西建在这条 95% 结论上。**
>
> **哪条帮、哪条害**：喝药条 **害**（一条就把 L1 从 3 推回 8，存活 28 → 20）；
> 接触脱离条 **什么都没做**（L1 死亡种子集合与 of-potion 完全相同，
> 且仍有 68.0% 的脱离拍原地不动）；拾药条 **对 L1 无害但炸窗口**。
> **RESUPPLY 窗口爆炸的解释**：`m[RESUPPLY] = controller_mask[13]` ＋ `_farm_handoff`
> ＋ `FARM → RESUPPLY → DIVE` 的回退阶梯，被**拾药这个多拍走路宏把角色带离怪群**触发，
> 形成每扇 9.7 拍的空转短窗循环（1735 → 7761 扇）——
> **不是**因为 a13 按得多（`ORACLE-PICKUP` 的 a13 是 698，比 `ORACLE` 的 909 还少）。

---

*本报告由 teacher 第二轮实施者写于 2026-09-10 深夜至 2026-09-11 凌晨。
所有数字来自 `/home/laure/r17_work/r19/round2/probe/` 下本轮自产的文件，
表格由 `round2/teacher2_extra.py` 机器生成；
第一轮的数字来自原样保留的 `/home/laure/r17_work/r19/teacher-probe/`；
没有出处的地方一律写了「未验证」。
复核轮（§十六）于 2026-09-11 凌晨在同一棵树上执行，
六条臂全部重跑、两道回归闸重过，单元卷 85 项全过。*

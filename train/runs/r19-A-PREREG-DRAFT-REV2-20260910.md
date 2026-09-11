<!-- 登记说明(主刀,2026-09-10 夜):本文件是第二步(老师→学生管线)设计席的合成稿,**草案、未冻结、未发射**。
第一步的老师(teacher-v1)没有赢过学生,故第二步暂不启动;保留本稿是因为它查清了任何老师都绕不开的硬事实
(考卷合同要求蒸馏排除动作 = [12,14],喝药与穿装备行为在现行合同下不可蒸馏;蒸馏只改写 298 词根;退火须在 512 个 rollout 内闭合;
resume 腿继承 critic 预热回执、--bc-init 全新暖启动在结构上不可受理;挂 BC 老师需约 40 行新旗标;老师须是 (观测, 掩码) 的纯函数)。
面板记录见 r19-STEP2-DESIGN-PANEL-20260910.md;三份设计原稿在 train/runs/r19-reports/step2-design/。步骤一数字占位【待填:步骤一】保留原样,以 r19-TEACHER-V1-PROBE-REPORT-20260910.md 为准。 -->

# R19-A 训练臂预注册 **修订二稿**(2026-09-10,合成席)— 一层毕业臂 + 老师→学生管线

> **状态:草案 / 未冻结 / 未入台账 / 不得据此发射。**
> 本文件是 `train/runs/r19-A-PREREG-DRAFT-20260909.md` 的修订稿,落在设计目录
> `/home/laure/r17_work/r19/step2-design/`,**不是主树文件**。冻结时须以**新文件**落到
> `train/runs/`,sha 入台账后方可发射(宪法:冻结件新文件、先 sha 后发射)。
> 上位文件:`r19-PREREG-DRAFT-20260908.md`(战役草案)。
> 步骤一(teacher-v1 vs 学生 7e31dc54)正在跑;**凡需步骤一数字处一律写
> 【待填:步骤一】**,不得以任何默认值代替。

---

## 〇、主席裁定(2026-09-08 晚 — 2026-09-10)

1. 收入端:金币边打边捡(gold-grab-v1)、清扫 v2(箱/桶/石棺,一二层,目标预算按层 24+24,不占进城名额)、
   购物 v2(full-v2,按「预计收益/金价」排序)——合并探针 R19-M1/M2。
2. 生存端:一层窗内急停(stop-v1)、教练 coach-v05(战备比 < 1 不无条件开 DIVE;连续两个停滞 DIVE 窗后本层锁 DIVE)。
3. 部署形态终点是**一张网**;earned-dive-suffix 的「一层退化」不能再出现。
4. 阈值变更须新预注册;处女池零接触;冻结件 sha 入台账后才发射。
5. **(2026-09-10 新裁)脚本变成老师,不变成运行时规则。** 步骤一检验 teacher-v1 是否胜过学生
   7e31dc54;**步骤二(本修订稿第八节)把老师的行为搬进学生**,终点仍是一张网、**运行时无脚本本能**。

---

## 一、假设与可辨结局

**H-A(不变)**:在「收入进得来、钱花得对、被围能脱身、教练不乱开窗」的世界里,从父代 7e31dc54 热启动、
所有窗都由学习者实况打(含一层)的工人,会把一层打稳(一层死亡 ≤ 5%、一层清空 ≥ 50%),
并在首次下楼时带着更好的装备(腰带 8、护甲 ≥ 15),二层风险率不升。

**H-B(新增,步骤二)**:teacher-v1 在步骤一取得的一层生存优势,其**可由学生观测判定的那一部分**
(= teacher-v1-obs,第八节 §2),能经由 v24 橡皮筋蒸馏迁入 7e31dc54,使训练后候选
(甲)对 7e31dc54 成对净胜,(乙)**逼近或胜过 teacher-v1**,(丙)β 退火归零后**不带任何运行时脚本**。

**反假设**:
(甲)一层稳了但从不下楼(首降率降、深度不动);
(乙)收入法条挤占战斗(二层击杀、clvl ≥ 4 降);
(丙)急停被学成「一挨打就跑」(击杀/局降、一层清空率降);
(丁)**(新)老师优势落在两条不可观测子句上**(上楼梯平局打破、400 拍预算)——
则本管线搬不动它,由闸 B0 在花任何训练预算之前判定;
(戊)**(新)E 过而 F 不过**——候选靠别的路子买到了存活,而不是内化了老师的本能;
此时只报告,不得宣称「教师已内化」。

步骤一实测:teacher-v1 一层死亡 【待填:步骤一】/48(父代对照 7/48);成对 saved/lost/net
【待填:步骤一】;单侧 UCB95 【待填:步骤一】;二层击杀 / 首降 【待填:步骤一】;
`disengage_stairs_tiebreak` 触发率 【待填:步骤一】;`disengage_budget_exhausted`
【待填:步骤一】(`teacher_v1.py:176-179`)。

---

## 二、世界(训练 = 测量;以 M2 合并树的认证字节为准)

| 项 | 值 | 来源 |
|---|---|---|
| 起点权重 | 7e31dc54 | 现役工人 `train/runs/r16-arm-a-constitution/model_candidate.zip` |
| 热启动 | **不重置 critic**:普通 `--resume-from`,**不得**传 `--reset-worker-critic` | critic 预热回执随 resume 继承(`leashed_ppo.py:3261-3267`);考卷九项由继承计数器闭合(`eval_assembled.py:1013`,调用 `:1581`) |
| 学习窗范围 | farm-dive-v1(全部窗实况,含一层) | 主席裁定 3 |
| 协议 / 购买 / 经济 | l2-town-v1 / full-v2 / sustain-loot-v1 | M1/M2 |
| 收入法 | gold-grab-v1(12+12)、sweep-v2(24+24,触发半径 8,怪物半径 2)、cain-v1、smith-v1 | M1/M2 |
| 生存法 | retreat-v1、stop-v1、hunt_scope=l1-only、portal 关、boss_room 关 | M2 |
| 教练 | coach-v05 | M2 |
| 时钟 / 托管 | completion-l2-r18c;托管 0.5 / 幂 1.6 / 战备门 / 尺 v2;兑现条件不变 | R18-B |
| 工资项 | 与 R18-B 固定字典相同 | R16 |
| PPO | mppo、CPU、lr 1e-4、ent 0.005、target-kl 0.01、n-steps 512、4 环境(父代合同钉死) | R16 |
| **蒸馏(新)** | `--distill-beta 0.05`,`--distill-anneal-actor-rollouts 256`,scope 恒为 `legacy-root-logits`,`excluded_actions` 恒为 `[12, 14]` | `eval_assembled.py:1485-1533` 对 rev ≥ 23 要求该字面量;合同书写方硬编码 `list(LEGACY_DISTILLATION_EXCLUDED_ACTIONS)`(`train_ppo.py:2671-2674`) |

**世界一致性要求(新)**:示范采集、蒸馏训练、配对复测**三者同一世界**——
同一 M2-SET 法条集、同一 `learning_window_scope='farm-dive-v1'`、同一 legacy-v3 观测面。
现役 `bc_worker.collect()` 三项全不满足(FARM-only、无 `resource_*` 入参、
`worker_env.py:815-823` 的默认关协议世界),故必须重采(第八节 §4)。

---

## 三、闸门

沿用一稿的 A–F,并按第八节补入 B0 / B1 / G0 / G5 / G6。

- **A–D 四闸**:六卷 `r17-anchor-*` 重铸对照(考卷协议 3000 拍旧形态、协议关);
  额外**显式复跑** `_validate_asymmetric_worker_runtime_state`(`eval_assembled.py:1013`)与
  rev22+ 的 `_asymmetric_worker_deployment_evidence_complete`(`:1067-1080`),
  确认继承回执完好;并确认 rev21+ 蒸馏合同闭合与
  `anneal_closed`(`eval_assembled.py:1100-1121`:`anneal_completed >= anneal_rollouts`
  且 `last_effective_beta == 0.0`)。**阈值一律不动。**
- **E 配对复测(主判据)**:同 M2-FULL 世界、同池 2_133(48 种子 2133000-2133047),
  对照行 = M2-FULL 臂的父代行(rows sha 待 M2 出);
  - 主-1 一层死亡 ≤ 5%(对照 7/48 = 15%);
  - 主-2 一层清空率(剩 ≤ 3 只)≥ 50%(对照 10%);
  - 主-3 成对存活 saved − lost ≥ +6/48 且单侧 UCB95 < 0;
  - 守门(不得退化):二层风险率 ≤ 对照 × 1.1、二层击杀 ≥ 0.9 × 对照、首降率 ≥ 对照 − 3;
  - 反假设(丙)检查:急停触发后 30 拍内存活率、每局击杀。
- **E′(新,第三臂)**:候选 vs **teacher-v1**,同 48 种子成对:
  net ≥ −3/48(「逼近或胜过老师」)。是闸还是报告行,待主席裁定(第七节)。
- **F 机制子指标**:急停触发次数与结局;coach-v05 锁窗次数;DIVE 窗停滞数;
  捡金/清扫/购物按层账本;腰带与护甲首降分布;托管兑现/罚没;
  **(新)本能三指标**:a12 按压的 hp 分布(应集中在 [0.50, 0.60))、
  a13 仅在无接触怪时按下的比例、被围态(hp ≤ 0.60 且 ≥2 邻怪)的分离度增益。

---

## 四、身份(冻结时填)

协议 bundle sha、桥 sha(现役 build-res)、引擎 sha、补丁栈 0001–0014、工人 zip sha、探针版本、
对照行 rows sha、**教师行 rows sha(步骤一)**、M2 合并补丁 sha、
**新增(步骤二)**:`_implementation_bundle_sha256()`(M2 后)、`generator_sha256 = sha256(train/bc_worker.py)`(改动 D 后)、
`demos.npz` sha、`bc_report.json` sha、`policy_sd.pt` 的 `policy_sha256`、
一次性 marker 的 `pool_sha256`、观测性探针 json sha、本文件 sha → 台账。

---

## 五、预算与时间

- 训练腿:步数 1 048 576(= 512 rollout × 2048 步/rollout,4 环境 × n-steps 512);
  **行动者 rollout 总数 = 512**,故 `--distill-anneal-actor-rollouts 256`
  (必须 ≥ 2 且远小于 512,否则 `anneal_closed` 永不闭合 → 考卷 FAIL)。
- 训练腿墙钟:按 R16 实测 `sps 120`(`train/runs/r16-arm-a-constitution/status.json`)约 **2.4 h**;
  M2 世界更慢,**不确定**,保险丝仍取 10 小时。
- 其余:M2 安装与认证 【待填:M2 报告】;改动 B/C/D/E 的实现 + 单元测试约 2–4 h;
  示范采集 384 局 × 3000 拍 **未验证**(按 v1 的 128 局墙钟外推);三次 48 种子成对扫 【待填】;
  考卷 + 配对 约 3 h。
- 存档/哨兵 63 488;发射前泡机 ≥ 30 min;磁盘 ≥ 20 GB;Windows 更新暂停。
- 评估池 2_133;**处女池 2_116–119、2_126–128 零接触**(本稿任何步骤均不触及)。

---

## 六、发射前顺序(固定)

1. **M2 合并树通过审核 → 安装到主树**(新版本串,旧默认路径逐位不变)→ 全套件 + 探针回归(16 种子 33023de1…)
   + 六卷考场重铸身份 + 双向重烤。**M2 必须把每一个新法条模块写进 `_IMPLEMENTATION_SOURCE_FILES`**
   (`train_ppo.py:801-849`;该元组自带「不存在的路径会在 `_implementation_bundle_sha256` 失败关闭」的注释);
   `python/diablogym/resource_emergency_stop.py` 目前**主树根本没有**,teacher-v1 在主树连 import 都不成立。
2. **步骤二的全部代码改动(第八节 §5 的 A–E)与 M2 骑同一次版本跃迁**——一次套件、一次回归、
   一次六卷身份、一次重烤;**不得二次跃迁**。
3. 对照行:7e31dc54 × M2-FULL 世界 × 2_133(48 种子)在主树上重跑并登记 rows sha;
   教师行沿用步骤一的 rows sha(同 48 种子、同世界,否则重跑)。
4. **闸 B0**(零训练)→ 示范采集与克隆 → **克隆逐类闸** → **闸 B1**(零训练)→ 铸教师工件。
5. 4096 步冒烟 + 30 分钟泡机(确认 `distill_ce` 有限且下降、合同 `excluded_actions == [12,14]`)。
6. 冻结本文件的**主树新文件版**(sha 入台账)→ 发射 → 台账。

任一步失败即停;**不得为救闸门改阈值**(改阈值 = 新预注册)。

---

## 七、待主席裁定

1. **附着路线**:路线 A(参数化 `data_gate` 池身份 + 新旗标 `--teacher-bc-sd`,约 40 行指纹内代码,
   出处绑定最硬)还是路线 B(把克隆权重写进 7e31dc54 副本、用未改动的 `export_manager_sd.py`
   铸导出清单、走既有 `--teacher-override`,指纹内 **0 行**,但铸出一张血统含混的嵌合检查点)?
   合成席推荐 **A**,B 列为回退 F2 并在台账标注 `teacher_only_not_deployable`。
2. **接受 a12 不迁移**:考卷合同本身要求 `excluded_actions == [12, 14]`
   (`eval_assembled.py:1485-1533`),故 potion 子句**不可能**由 CE 承载;
   本稿不碰该常量,如实在闸 F 里测它,并把「蒸馏通道扩到 a12」列入 R20 议程(须自带合同版本跃迁与预注册)。
3. **接受保真上限**(上楼梯平局打破 + 400 拍预算不可观测,见第八节 §2),而不是新铸观测视图版本
   ——后者会重建行动者输入、作废 7e31dc54 血统与六卷考场身份,并强制重新 critic 预热。
4. **标签世代 = 1**(禁 11/12)与**新示范池 2_144_000..2_144_383** 是否照准。
5. β = 0.05 / 退火 256 rollout 直接打,还是先做 1/8 预算的 β ∈ {0.02, 0.05, 0.10} 三点扫?
6. E′(对老师的成对复测)是**闸**还是**报告行**?
7. DAgger(学生自采、老师贴标)是否**预先授权**为回退 F4,还是必须另立预注册?

---

## 八、老师 → 学生管线(步骤二,新增)

### §1 结论一句话

在 M2-SET 世界里,用**既有** BC 机器把 **teacher-v1-obs**(= teacher-v1 去掉两条不可观测子句)克隆成一张
298→64→64→15 的网,以**新旗标 + 既有 `data_gate` 校验器**挂到 `--resume-from 7e31dc54` 的腿上,
经 v24 橡皮筋(scope `legacy-root-logits`、`excluded_actions [12,14]` 原封)把**a13 与 a1..a8**
搬进学生的 298 词根,β 线性退火到 0;终点是一张网、运行时无脚本。
**不新增观测视图、不改任何阈值、不动蒸馏排除常量、不改 `eval_assembled.py` 与 `leashed_ppo.py`。**

### §2 观测性裁定(本设计的枢纽)

两个表示同时在场:
**学生的行动者**吃的是 dual 13012 行(`leashed_ppo.py:2302-2316`;7e31dc54 的
`mlp_extractor._context_enabled = True`,预热结束时原子置位 `:4597-4601`),
掩码在 `617:632`、怪物行带 `visible` 字段;
**蒸馏教师网**吃的只有 `obs[:, :298]`(`leashed_ppo.py:5099-5108`),
且 CE 只改写 `policy_net(features[:, :298]) + action_net`,语境残差**故意留给奖励**
(`leashed_ppo.py:5126-5154`)。**不对称在教师侧,不在学生侧。**

298 行布局(据 `env.py:5904-6037` 读出):
idx 0 = hp/max_hp;8,9 = 怪数/50、最近切比雪夫/30(**全部**怪物,未按可见性过滤);
10,11 = **下楼 / 进度目标** 的 dx,dy;12..43 = 8 个最近怪槽位;44..164 = 11×11 可走格
(`_MAP_RADIUS=5`);165..285 = 11×11 怪物占位面;286 = `legacy_belt_heals/8`;
287..289 = 最近治疗品 dx,dy,present;295..297 = τ、层时钟、drink 闩。
瓦片 (dx,dy) 的怪物面下标 = `165 + (dy+5)*11 + (dx+5)`,玩家中心 = 225。

| teacher-v1 子句 | 298(教师网可见?) | 裁定 |
|---|---|---|
| 层 0 装备宽限 a14 | 由环境把掩码收窄到 {a0,a14};a14 logit 先验 2.5 已在位 | **等价可得,无须动作**;但 a14 被 CE 排除,不由蒸馏承载 |
| I potion:`belt>0 ∧ hp<0.60·max` | idx 0 与 idx 286 的锐函数 | **完全可观测,但被考卷合同排除出 CE(裁定 2)** |
| II pickup:`mask[13] ∧ 半径1 内无怪` | 225 周围 3×3 块 | **可观测**(至可见性差) |
| III 脱离触发:`hp≤0.60 ∧ ≥2 在 Cheb 2` 或 `hp≤0.35 ∧ ≥1 在 Cheb 1` | idx 0 + 5×5 / 3×3 块 | **可观测**(至可见性差) |
| III 脱离走位排序:合法 a1..a8 最大化最小分离 | 触发保证怪在 ≤2,能改变排序的怪都在 Cheb ≤3,半径 5 面内;12..43 为第二源 | **可观测** |
| III 上楼梯平局打破 | idx 10/11 是**下楼**;dual 行里也没有 trigger 通道 | **不可观测** |
| III 400 拍/局脱离预算 | 无对应特征(非马尔可夫) | **不可观测** |
| IV 基座 dispatch | 读 raw;R16 BC-v1 对同一基座的克隆 top-1 = **0.6951**(`train/runs/bc-worker/bc_report.json`,`data_gate: PASS`) | **约 0.70 保真上限,接受** |

一条残余单侧缺口:老师数**可见且存活**的怪(`resource_emergency_stop.py:191-208`),
而 298 的怪物面(`dMonster!=0`)与怪槽位都是**未过滤的超集**。

**裁定:限制老师,不扩观测。** 定义
**teacher-v1-obs = teacher-v1 − 上楼梯平局打破 − 400 拍预算,且三条本能子句一律从 298 行判**。
如此,被贴的标签就是 `(obs, mask)` 的**精确无记忆函数**,本能类召回可趋近 1.0,
不可约损失只剩基座的约 0.30。
新铸观测视图版本**当场否决**:会重建行动者输入、作废血统与六卷身份、强制重新 critic 预热。
因为 teacher-v1-obs ≠ teacher-v1,**闸 B0 在花任何预算之前先把二者在 2_133 上对测一遍**。

### §3 附着路线(老师怎么挂上 resume 腿)

事实(已复核):
- `--teacher-override` 只在 worker + resume 腿合法(`train_ppo.py:6811-6812`),
  且**只**由 `_validate_export_manifest` 校验(`:6931-6934`,`:10382-10385`),
  该校验要求工件 `.pt` 与某真检查点的 `policy.pth` **键集/张量逐位相同**(`:7745-7770`)——
  BC 头只有六个 SB3 键,**永远过不了**。
- resume 腿的第三分支(检查点自带 `teacher_path`)走的是
  `_validate_bc_report(saved_teacher, "data_gate")`(`train_ppo.py:10874-10890`),
  这是一张**会对着 `demos.npz` 重算**的真回执(`:7560-7568`),并绑定 manager NPZ、实现束 sha
  与 `generator_sha256 = sha256(train/bc_worker.py)`(`:7513-7520`)。**缺的只是一个 CLI 入口去设这个路径。**
- **(合成席新查,三席均未发现)** `data_gate` 分支把 v1 面写死了:
  `_validate_bc_final_holdout_marker(p.parent, **1**, **_WORKER_BC_DEMO_SEEDS**, rec)`(`:7519-7520`),
  且 `_validate_worker_bc_evidence` 要求 demos 的 episode 唯一值**逐位等于**
  `_WORKER_BC_DEMO_SEEDS = 2_142_000..2_142_127`(`:195`,`:7218-7220`),并禁止 `y ∈ (11,12)`(`:201`,`:7212-7213`)。
  而该池**已被 marker `752f5150` 一次性烧毁**(注册表 8 枚 marker 复核),
  `_assert_final_holdout_unused`(`bc_worker.py:170-182`)拒绝同池再采。

**路线 A(推荐)**:把 `data_gate` 的**池身份参数化**(默认值 = 现有 v1 元组,默认路径逐位不变),
再加新旗标 `--teacher-bc-sd`,守卫与 `--teacher-override` 完全一致、二者互斥,
校验仍调**同一个**既有 `_validate_bc_report(..., "data_gate")`,然后写进
`_load_kw["teacher_path"] / ["teacher_sha256"]`。**不新增任何校验语义,只是让一条已认证的路可达。**
连带推论:**标签世代取 1**(禁 11 与 12)。既然 a12 本就被 CE 排除,丢弃 a12 标签**零迁移成本**,
还让 v1 面的禁采守卫**一点不弱化**——这是宪法上最便宜的一步。

**路线 B(回退 F2)**:把克隆的六个 SB3 键写进 7e31dc54 副本(形状同构:
`policy_net.0.weight (64,298)`、`action_net.weight (15,64)`,共 94 个策略张量),`model.save()`,
再用**未改动**的 `train/export_manager_sd.py` 铸出 `.pt` + 清单,走既有 `--teacher-override`。
指纹内 0 行,但铸的是一张血统含混、看上去像可部署的嵌合体。**宪法上不推荐**,仅在改动 B/C 被否时启用,
且必须在台账标注 `teacher_only_not_deployable`。

**真 DAgger:不值这份代码。** v24 的 CE 本来就算在 `rollout_data.observations`——
**学生自己走到的状态**上(`leashed_ppo.py:6405-6419`),分布校正已经在了;
真 DAgger 只是把克隆的标签换成脚本的现场标签,需要活教师进向量环境、env-info 标签通道、
回放缓冲区 schema 扩展与第二条 CE 路径,横跨三个指纹文件并全量重认证。
**改为测量它本可买到的东西**:观测性探针 G0 直接报告「克隆 vs 脚本在老师访问状态上的分歧率」;
分歧率小 → DAgger 不值钱;大 → 走回退 F4。

### §4 示范:池、窗、世界、标签

**采集器**:改 `train/bc_worker.py` 本身(改动 D),**不另起新文件**——
因为 `data_gate` 的 `generator_sha256` 硬绑 `sha256(train/bc_worker.py)`(`train_ppo.py:7513-7520`),
新文件产出的报告**永远进不了那个校验器**(这是 minimal-code 方案在此处的致命缺陷)。
`bc_worker.py` **不在** `_IMPLEMENTATION_SOURCE_FILES`(`train_ppo.py:801-849`),
所以改它**不动实现束**,只按设计改 `generator_sha256`——这正是它的回执。

- **观测面**:`legacy_policy_observation_view=True`(同 `collect()`,`bc_worker.py:306-313`)。
  **不得**用 `collect_v2` 的 a12 叠加面(`:875-882`),那与蒸馏输入在 idx 286 / 297 两处不同,
  会造成静默的 train/serve 偏斜。
- **窗范围**:`learning_window_scope='farm-dive-v1'`(`worker_env.py:125-133`)。
  `collect()` 从不传该参数,即 FARM-only;而 CE 是在学生的 rollout 上算的,里面两种窗都有,
  且 298 前缀**不含窗模式位**,FARM-only 的克隆会把 FARM 行为导出到长得一样的 DIVE 状态里。
- **世界**:传 M2-SET 的 `resource_*` 入参(purchase full-v2、sweep-v2、gold-grab-v1、coach、retreat、hunt_scope l1-only …);
  `collect()` 一个都不传,即默认关协议世界(`worker_env.py:815-823`)。
- **标签**:**世代 1**,禁 a11 与 a12(`bc_worker.forbidden_actions_for_generation`);
  a13 与 a1..a8 不受限;a14 仍是强制召回类,≥ 64 个标签、≥ 16 局(`train_ppo.py:206-207`),
  由装备宽限层自然产出。老师从不提 a11(`teacher_v1.py:101`)。
  按 `collect()` 的既有规则丢弃 `overridden` 与 `executed_action is None` 的拍(`bc_worker.py:325-335`)。
- **规模**:384 局(= 已登记的 3× 采集因子)。脱离子句稀疏,需要的是局数而不是拍数。

**示范种子池(须立法登记)**:**2_144_000 .. 2_144_383**(384 个)。
- 在 BC 立法块内(2_140_000–2_158_xxx),远离评测银行 2_11x;
- 与 `_BURNED_BC_EPISODES`(`train_ppo.py:196-200`)、注册表 8 枚 marker、2_133、
  以及**处女池 2_116–119 / 2_126–128** 全部不交集;
- 为何不用旧池:v1 的 2_142_000..127 **已烧**(marker `752f5150`);
  v2 的 2_143_000..383 **已登记未开封**,但它是 v2(a12 叠加面 / bc-aux)**保留池**,
  吃掉它等于毁掉那条线的保留池;
- 一次性 marker 仍走**既有**注册表助手烧;世代 1 被旧 marker 复验规则接受
  (`train_ppo.py:683-684` 要求 `teacher_generation in (1,2)`;世代 3 会让此后**每一次** BC 采集失败关闭)。
- **2_133(48 种子 2133000-2133047)只用于成对评测**(B0 / B1 / E / E′),绝不用于采集或训练。

### §5 代码改动清单(逐条:文件 / 位置 / 做什么 / 是否指纹内 / 如何认证)

| # | 文件(函数/位置) | 做什么 | 指纹内? | 认证 |
|---|---|---|---|---|
| **A** | `train/train_ppo.py`(常量区 `:195-207`、`:280-285`) | 新增 `_WORKER_BC_V3_DEMO_SEEDS = tuple(range(2_144_000, 2_144_384))`,并把它并入模块载入期的不交集断言(`:281-285`) | **是**(实现束 + 生成器消费方) | 纯加常量;既有不交集断言即是测试;骑 M2 同一次版本跃迁 |
| **B** | `train/train_ppo.py`(`_validate_bc_report` `:7519-7520`;`_validate_worker_bc_evidence` `:7218-7220`) | 把 `data_gate` 的**池身份参数化**:新增关键字 `demo_seeds`,默认 `_WORKER_BC_DEMO_SEEDS`;逐层透传到 marker 校验与 demos 的 episode 逐位比对。**世代仍写死 1,禁采集合 `(11,12)` 原封不弱化** | **是** | 默认路径逐位等价测试(不传新参数时字节不变,与 v24 在 β=0 上用的同一论证);两条新单测:新池报告通过 / 旧池报告仍按原样通过;错池报告必须抛错 |
| **C** | `train/train_ppo.py`(CLI 旁 `--teacher-override` `:6811-6812`;`_load_kw` 链 `:10864-10890`;出处记录 `:10627-10629`) | 新增 `--teacher-bc-sd`:守卫与 `--teacher-override` 一致(仅 worker + resume),二者互斥;用**既有** `_validate_bc_report(p, "data_gate", demo_seeds=_WORKER_BC_V3_DEMO_SEEDS)` 校验,取 `policy_sha256`,写入 `_load_kw["teacher_path"]/["teacher_sha256"]`;出处串写进 status.json | **是** | 负测:非 worker / 非 resume / 与 `--teacher-override` 同用 → 抛错;报告非 PASS → 抛错;腿后 `model.teacher_path` 等于旗标值(比照 `:11089-11093`);出处串必须能在 status.json 里被预注册钉住 |
| **D** | `train/bc_worker.py`(教师钩子 `:292-304`;`collect()` `:306-364`) | 新增 **v3 面**:`teacher_action_v3`(把 teacher-v1-obs 的阶梯**原样内联**——装备宽限 / potion / pickup / 脱离 / 冻结 dispatch,三条本能一律**从 298 行**判)与 `collect_v3()`(farm-dive-v1、M2-SET 入参、v3 池、世代 1)。**v1/v2 两个面一字不改** | 生成器 sha 内(**非**实现束) | `generator_sha256` 变更**是有意的**,并且是回执本身;`_reject_same_generator_terminal`(`:242-266`)强制新终态;**等价性测试**:在 G0 的定点回放上,`teacher_action_v3` 与 `teacher_v1.TeacherV1('teacher-v1')` 去掉两条不可观测子句后**逐拍相等**;确定性重放(同种子 → 同标签序列) |
| **E** | `python/diablogym/worker_env.py`(`BC_RESERVED_SEED_RANGES` `:91-112`) | 追加一条 `(2_144_000, 2_144_384)`,使普通训练排除新池 | **是** | 纯追加常量;既有种子表单测 + 注册表自带的 `_assert_bc_final_holdout_pool_disjoint` 扫描;骑同一次版本跃迁,**不得二次跃迁** |
| **M2** | `python/diablogym/resource_emergency_stop.py` 等新法条模块 → `_IMPLEMENTATION_SOURCE_FILES`(`train_ppo.py:801-849`) | M2 安装的一部分,但在此点名:示范报告绑的就是这个束,且该元组**失败关闭** | **是**(属 M2 的认证包) | M2 自己的套件 + 回归 33023de1 + 六卷身份 + 重烤 |
| 新 | `train/runs/r19-step2/`(观测性探针、三臂配对驱动、`teacher_v1.py` 副本、台账行) | G0 探针;2_133 三臂成对(rows_sha_v3、arm_stats、成对 McNemar saved/lost/net + 单侧 UCB95,沿用 `r18/m2_probe_driver.py` 与 `probe_r17_deployment.run_episode/load_zip_policy`) | **否**(不属任何指纹束,须在文件头明写) | 对照臂行哈希与步骤一驱动在同 48 种子上逐位一致;两个种子的确定性重放 |

**一律不动**:`train/eval_assembled.py`、`train/leashed_ppo.py`、`train/export_manager_sd.py`、
`python/diablogym/options_env.py`、`env.py`、`controller_wire.py`;
不新增观测视图版本;不改任何阈值;不动 `LEGACY_DISTILLATION_EXCLUDED_ACTIONS`。

### §6 管线(每步一张回执、每步可回滚)

| # | 步骤 | 产物 / 回执 | 回滚 |
|---|---|---|---|
| P0 | M2-SET 装入主树(含 `_IMPLEMENTATION_SOURCE_FILES` 绑定);改动 A–E 骑同一次版本跃迁 | 新树指纹、新 `implementation_sha256` | 指纹 diff;回退补丁 |
| P1 | 对照行:7e31dc54 × M2-SET × 2_133 | 对照 `rows_sha` 入台账 | 确定性重跑 |
| P2 | 教师行:`oracle` 与 `teacher-v1`(沿用步骤一) | 教师 `rows_sha` | — |
| P3 | **G0 观测性探针(零训练)**:回放 P2 的局,逐拍导出 `(obs_dual, obs298, mask, teacher_action, clause_source)`;离线各拟合一枚克隆(298 / dual),报告 top-1、逐子句召回、上楼梯平局打破分歧率 | `r19-step2-observability.json` | 只读 |
| P4 | 登记新池 2_144_000..383(世代 1),烧一次性 marker | `pool_sha256` marker | marker 本身即记录:第二次必被拒 |
| P5 | `collect_v3()` 采 384 局 → 训练克隆(既有 `PiHead`/`train_bc`/`export_sb3_sd`,`bc_worker.py:367-377, 524-582`) | `demos.npz`、`policy_sd.pt`、`bc_report.json`(`data_gate: PASS`) | 删产物;marker 依设计保持已烧 |
| P6 | **B0 / 克隆逐类闸 / B1**(全部零训练,见 §7) | 三份闸门读数 | 只读 |
| P7 | **冻结**:主树新文件版预注册,sha 与上列全部身份入台账 | 台账行 | — |
| P8 | 4096 步冒烟 + 30 分钟泡机 | 冒烟日志 | kill |
| P9 | 训练腿(§8)→ 考卷 A–D → 2_133 三臂成对 E / E′ → 闸 F | `model_candidate.zip`(继承并延长回执) | 父代 zip 原封未动 |

顺序不可交换:P0 与改动 A–E 必须在 P5 **之前**落地,因为 `bc_report.implementation_sha256`
绑的是整棵树的束(`train_ppo.py:851-891`),且 `_reject_same_generator_terminal` 拒绝同
实现 + 生成器 sha 的重试;现役 v1 报告的 sha 本来就已过期(`b40b154c` vs r16 的 `9ff658d7`),
无论如何都要重采。

### §7 闸门与停机规则(步骤二新增部分)

- **G0(P3,零训练)**:298 克隆在老师访问状态上的 top-1 < 0.85,**或**脱离子句召回 < 0.70 → 不发射,转回退。
  同时报告克隆-脚本分歧率(DAgger 的价值上界)。
- **B0(零训练,花任何预算之前)**:teacher-v1-obs vs teacher-v1,2_133 成对。
  *闸*:teacher-v1-obs 保住 teacher-v1 对 7e31dc54 的一层死亡降幅的 **≥ 80%**,
  且其对 7e31dc54 的成对净胜为正、单侧 UCB95 < 0。
  同时报告 `disengage_stairs_tiebreak` 与 `disengage_budget_exhausted`(步骤一遥测)。
  **若优势落在被砍掉的两条子句上,§2 的限制就是错的:停机**(反假设丁)。
- **克隆逐类闸(不是看总分——总分恰好盖住要迁移的东西)**:
  `data_gate` PASS;**recall(a13) ≥ 0.80**;脱离走位块(a1..a8)在本能拍上的 top-1 ≥ 0.85;
  基座 top-1 ≥ 0.65(v1 实测 0.6951 减去更宽窗范围的余量);a14 满足 `train_ppo.py:206-207`。
  **注意**:v1 报告里 `class_recalls['13'] = 0.45`(`train/runs/bc-worker/bc_report.json`)——
  那是**基座 dispatch 的捡拾**,不是本能子句;teacher-v1-obs 的 a13 触发是 `(obs, mask)` 的锐函数,
  理应远高于 0.45。**这仍是全流程最可能停机的一道闸,保留原值不放松。**
  只许一次类加权重试(「只记不裁」的既有规矩)。
- **B1(零训练)**:把克隆网本身当作工人 callable 在 2_133 上跑,对 7e31dc54 与 teacher-v1-obs 成对。
  *闸*:克隆保住 teacher-v1-obs 成对净胜的 **≥ 70%**。
  这是蒸馏能交付的**诚实天花板**,在花第一个 CPU 小时之前测。
  **读数口径**:该臂跑的是纯 298 网,而腿只能改写 root、语境残差仍由奖励驱动
  (`leashed_ppo.py:5126-5154`),故 B1 是一个**宽上界**,不是对候选的预测。
- **G2(冒烟)**:`distill_ce` 有限、为正、下降;合同里 `scope == legacy-root-logits`、
  `excluded_actions == [12,14]`、`schedule == linear-inclusive-zero-v1`、`anneal_actor_rollouts == 256`。
- **G3(考卷 A–D)**:必须以**原阈值**通过。
  任何「需要动阈值」= 放弃这条腿,而不是放弃阈值。
- **G4 = E**、**G5 = E′**(见第三节)。
- **G6(终点纯净)**:`last_effective_beta == 0.0` 且 `anneal_closed` 为真;
  部署 callable 只是那个 zip;`teacher_v1.py` 不被部署路径上的任何东西 import(一条 grep 测试);
  status.json 把这层脚手架登记为 **scripted-teacher,expiry = 本腿结束**。
- **停机规则**:B0 / 克隆逐类闸 / B1 / G0 任一失败;冒烟的 `distill_ce` 不降;
  考卷校验器拒收候选回执;主树与探针舰队冲突;`target_kl` 连续两次跳闸;
  50% 检查点处 8 种子哨兵的一层死亡率劣于对照;墙钟超 10 小时保险丝;
  **任何处女池接触**(2_116–119 / 2_126–128)→ 立即停机并记台账。

### §8 训练腿(确切配方)

```
--resume-from <M2 世界下的 7e31dc54 对照 zip> --worker --device cpu --seed <显式>
--teacher-bc-sd <.../r19-step2/policy_sd.pt>          # 路线 A;路线 B 则为 --teacher-override
--distill-beta 0.05 --distill-anneal-actor-rollouts 256
# 明确不传 --reset-worker-critic:父代 critic 预热回执必须原样继承
# PPO 由父代合同钉死:lr 1e-4 / ent 0.005 / target-kl 0.01 / n-steps 512 / 4 envs / 1 048 576 步
```

β 取小且退火,是因为 CE 只落在 legacy 词根上;β 太大会压平那条专为修战斗而建的语境残差
(`leashed_ppo.py:5129-5134`)。β = 0.05 是**未经先例验证的猜测**;
若冒烟显示 `distill_kl` 近乎平坦(太弱)或 PPO 损失被压住(太强),改走 1/8 预算的三点扫
{0.02, 0.05, 0.10}(须在冻结前完成,否则即为改配方)。

### §9 回退(按代价从低到高)

- **F1 · 限制老师**:已内建为 teacher-v1-obs。若 G0 卡在可见性差上,再把「可见存活怪」换成
  半径 ≤ 2 的**未过滤**存活集(298 面的函数),重跑步骤一的成对探针确认受限老师仍胜父代,再克隆它。
- **F2 · 权重手术 + `--teacher-override`**:若改动 B/C 被否。指纹内 0 行,血统含混;
  台账标注 `teacher_only_not_deployable`。
- **F3 · 残差老师**:若逐类召回高但闸 E 平。只在本能子句**确实触发**的拍上给老师标签,
  其余拍给**学生自己的贪心动作**——CE 在约 97% 的拍上 ≈ 0,不再被基座的 0.30 噪声拖着走。
  实现只在 `collect_v3` 里加载 7e31dc54 作标注器,约 40 行,**仍在 `bc_worker.py` 的 v3 面内**。
  若步骤一的优势集中在本能拍上,这是首选修法(优于加大 β)。
- **F4 · DAgger 一轮(数据决定,不是代码决定)**:同一采集器,行为策略换成学生,标签仍是 teacher-v1-obs,
  两者在同一拍取。需要**第二个**示范池与第二枚 marker,**须主席预先授权**(第七节问题 7)。
- **F5 · 如实报告负结果**:若 B0 / B1 结构性失败,步骤二的诚实产出就是
  「teacher-v1 的优势不是工人观测的函数」——这本身是真发现;
  下一问是「把蒸馏通道搬到 dual 行」(一次真正的 `leashed_ppo.py` 改动 + 合同版本跃迁),
  须自带预注册与自己的 critic 预热,**不得夹带进本腿**。

### §10 诚实声明(必须原样进台账)

1. **potion 子句(a12)不由本管线迁移。** 考卷合同本身要求 `excluded_actions == [12, 14]`;
   `drink_sovereignty` 只开掩码窗,不使学生去按。步骤一若把多数救命归因于 potion
   (`S_potion` = 【待填:步骤一】),那么闸 E 即便过了,**也不能说「老师被内化了」**——闸 F 会指出这一点。
2. **a14 同理不由 CE 承载**,沿用 7e31dc54 既有的 logit 先验 2.5。
3. **基座保真上限约 0.70**(v1 实测 0.6951),是可控但真实的噪声源。
4. **B1 是宽上界**:纯 298 网的成对表现不可能被这条只改 root 的腿完全复现。
5. 上楼梯平局打破与 400 拍预算**被明确放弃**,其代价由闸 B0 量化,不由猜测掩盖。

<!-- 登记说明(主刀,2026-09-10 夜):第二步设计席(3 名设计员 × 3 个角度 → 3 名评委 × 3 个镜头,逐条对着代码核实 → 1 名合成员;全部 Opus)的面板记录。非预注册。
关键主张已由主刀在主树代码中复核:leashed_ppo.py:170 LEGACY_DISTILLATION_EXCLUDED_ACTIONS=(12,14);eval_assembled.py:1485-1533 合同要求 excluded_actions==[12,14](rev≥23);
eval_assembled.py:1100-1121 anneal_closed;leashed_ppo.py:5126-5154 蒸馏 CE 只作用于 298 词根;leashed_ppo.py:2302-2316 行动者 = 词根 + 语境适配器。 -->

# R19 步骤二 · 神谕教师面板评审总结(2026-09-10,合成席)

本文件为面板记录,**非预注册**。预注册修订草案见
`/home/laure/r17_work/r19/step2-design/R19-A-PREREG-DRAFT-REV2-20260910.md`(草案/未冻结)。

---

## 1. 评分表(评审席:考卷可受理性 + 宪法合规)

| 设计 | 分数 | 一句话判词 |
|---|---|---|
| **certification-first**(胜出) | **7.5 / 10** | 对机器读得最准、考卷姿态最安全;唯一读出「橡皮筋只改写 298 词根、语境残差留给奖励」(`leashed_ppo.py:5126-5154`)的设计;附教师走**新增旗标 + 既有校验器**,而不是放宽旧校验器 |
| minimal-code | 6.5 / 10 | 工程最省、招牌最错:其核心补丁(`--distill-excluded-actions`)与考卷的蒸馏合同直接冲突(见 §2.1);但 B0/B1/闸 F 与 298 布局表是全场最好的 |
| fidelity-first | 5.5 / 10 | 科学最好、法律最弱:唯一注意到 M2 必须把新法条模块绑进实现束;但全文未提 a12/a14 蒸馏排除,标签政策建立在错误前提上;其补丁**放宽既有已认证闸门**,是宪法上最差的一手 |

无并列。合成稿采 certification-first 的骨架,并移植另两席的可用件(§4)。

---

## 2. 被判驳的主张(评审席已核,合成席复核)

### minimal-code

1. **(决定性)**「考卷可受理性是免费的……eval_assembled 一行不改」——错。
   `train/eval_assembled.py:1485-1533`:contract_revision ≥ 21 且 β>0 时要求
   `distillation.scope == "legacy-root-logits"` 且 `excluded_actions == [12, 14]`(rev ≥ 23 为该字面量);
   合同书写方硬编码 `list(LEGACY_DISTILLATION_EXCLUDED_ACTIONS)`(`train_ppo.py:2671-2674`),
   `_CONTRACT_REVISION = 26`。所以 `--distill-excluded-actions worker-full-v1` 的腿要么在考卷上抛
   `EvalContractError`,要么发一张与实际 CE 支撑集不符的假回执。
   **该 22 行补丁不是局部改动:它需要合同版本跃迁 + 新考卷分支 + 自己的预注册。**
2. 「行动者看不到掩码,只吃 `features[:, :298]`」——只在 `actor_context_enabled=False` 时成立
   (`leashed_ppo.py:2302-2316`);7e31dc54 的 zip 里 `mlp_extractor._context_enabled = True`。
3. 「v2 池 2_143_000..383 也已烧毁」——错。注册表 8 枚 marker 中**无**任何 2_143 段(合成席复核:
   `1e50974c`=2140000..127/gen1、`62161b13`=2104、`62642217`=2102、`628f6325`=3000..3383/gen2、
   `752f5150`=2142000..127/gen1、`83ed6551`=2108、`adedee94`=2000..2127、`cf023af0`=2106)。
   2_143 是**已登记未开封**。新池仍然要开,但正确理由是「2_143 是 v2(a12 叠加面 / bc-aux)的保留池」,不是「已烧」。
4. 「`train_ppo.py:9490` 属 KING 锚,不动」——错,那是 E5 蒸馏-CE 探针;只改 `:6409` 会让探针与损失量错位,
   正好把它自己依赖的冒烟闸门弄瞎。

### certification-first(胜出方,仍须修正)

5. `--distill-anneal-actor-rollouts 1024` 不可能:1 048 576 步 ÷ (512 × 4) = **512** 个行动者 rollout;
   `eval_assembled.py:1100-1121` 要求 `anneal_completed >= anneal_rollouts` 且 `last_effective_beta == 0.0`,
   退火永远闭不上 → 考卷 FAIL。修订稿取 **256**。
6. 「世代 3」既多余又危险:`train_ppo.py:683-684` 在不交集扫描时**复验每一枚旧 marker**,要求
   `teacher_generation in (1, 2)`;一旦落下世代 3 的 marker,此后**每一次** BC 采集都会失败关闭。
   且其世代 3 拟禁 `(11,)`,正是 `_WORKER_BC_V2_FORBIDDEN_ACTIONS`(`train_ppo.py:284`),即世代 2。
7. 「a12 靠奖励迁移」是希望,不是通道:`drink_sovereignty` 只**开掩码窗**,不使学生去按。
   诚实说法:**本管线根本不迁移 potion 子句**;闸 F 必须如实测它。
8. 预算「10-13 小时」不是 R16 先例:`train/runs/r16-arm-a-constitution/status.json` 记 `sps 120`,
   1 048 576 步约 **2.4 小时**(M2 世界可能更慢,标注为不确定)。

### fidelity-first

9. 「升到世代 2 就能把 potion 放回橡皮筋」——错:`leashed_ppo.py:170` 与未覆写的调用点 `:6408-6409`
   在**两个分布**里都删掉 a12,与标签世代无关。该设计全文未提此排除,其标签政策前提失效。
10. 「世代 2 的代价是 `train_ppo.py:3536-3560` 的第二个补丁」——那是 ④乙 bc-aux 示范库加载器,
    钉在 `_BC_V2_COLLECTION_EPISODES`,不在教师附着路径上。
11. **宪法质疑**:其补丁让 `--teacher-override` 改为「导出清单**或**同级 bc_report 皆可」,
    即**放宽一个既有已认证闸门的含义**。certification-first 的新旗标到达**同一个**已认证校验器,
    能力相同而校验漂移严格更少。

---

## 3. 合成席新查(三席与评审席均未发现,对胜出设计有实质代价)

**`data_gate` 校验器写死的是 v1 面,不是一个通用面。**

- `_validate_bc_report(..., "data_gate")` 调
  `_validate_bc_final_holdout_marker(p.parent, **1**, **_WORKER_BC_DEMO_SEEDS**, rec)`
  (`train/train_ppo.py:7519-7520`),即**世代 1 + 池 2_142_000..2_142_127**。
- `_validate_worker_bc_evidence` 要求 demos 的 `episode_id` 唯一值**逐位等于** `_WORKER_BC_DEMO_SEEDS`
  (`train_ppo.py:7218-7220`),并禁止 `y ∈ (11, 12)`(`train_ppo.py:201, 7212-7213`)。
- 而 `_WORKER_BC_DEMO_SEEDS = tuple(range(2_142_000, 2_142_128))`(`:195`)**已被 marker `752f5150`
  一次性烧毁**;`_assert_final_holdout_unused`(`bc_worker.py:170-182`)拒绝同池再采。

推论:certification-first 的 `--teacher-bc-sd`「约 18 行、不新增校验语义」**低估了成本**——
新池的报告根本进不了 `data_gate`,除非同时把**池身份**参数化。修订稿据此把改动拆成
「池参数化(默认逐位不变)」+「新旗标」两笔,并因此**把标签世代定为 1**(禁 11/12):
既然 a12 本就被考卷合同排除在 CE 之外,丢弃 a12 标签**零迁移成本**,还能让 v1 面的禁采守卫**一点不弱化**。
这一手同时兑现了 fidelity-first 的「默认世代 1」直觉——但理由与它给出的不同。

---

## 4. 移植清单(合成稿采纳)

**来自 minimal-code**

- **B0 / B1 两道零训练闸**:先测 teacher-v1-obs 对 teacher-v1,再测克隆网本身对 7e31dc54;
  **花任何训练预算之前**先知道天花板。
- **闸 F 机制子指标**:a12 按压分布、a13 的无接触约束、被围态的分离度增益——
  「E 过而 F 不过」不得宣称「教师已内化」。
- **teacher-v1-obs**:砍掉上楼梯平局打破与 400 拍预算,三条本能子句一律**从 298 行**判,
  使标签成为 `(obs, mask)` 的精确无记忆函数。
- 298 布局表(idx 0 / 8,9 / 10,11 / 12..43 / 44..164 / 165..285 / 286 / 295..297)。
- 示范必须走 `learning_window_scope='farm-dive-v1'` + M2-SET 的 `resource_*` 入参;
  **用 v1 面的 legacy 视图,不用 v2 的 a12 叠加视图**(否则 idx 286/297 两特征静默偏斜)。
- 残差教师回退(本能子句处给教师标签,其余处给学生自己的贪心动作)。

**来自 fidelity-first**

- **M2 安装必须把新法条模块写进 `_IMPLEMENTATION_SOURCE_FILES`**(`train_ppo.py:801-849`;
  该元组自带「boss_avoidance.py 故意缺席,不存在的路径会失败关闭」的注释),
  否则示范绑定的实现束是错的。
- 120 sps 的落地计时(全场唯一有据的估算)。
- 「DAgger 在这里是**数据**决定,不是代码决定」(`bc_worker.py` 不在实现束内)——作为回退保留,不进主线。
- 双行(13012)与 298 的不对称表:**学生行动者**几乎看得见全部子句,**教师网**看不见;不对称在教师侧。

**来自 certification-first(胜出骨架)**

- 不动 `LEGACY_DISTILLATION_EXCLUDED_ACTIONS`,不新增观测视图,不改任何阈值。
- 只蒸馏 CE 能承载的子句(a13 + a1..a8 + 基座);a12/a14 如实记为**不迁移**。
- P0..P9 每步一张回执、每步可回滚;G0 观测性探针;
  「需要动阈值 = 放弃这条腿,而不是放弃阈值」。
- 新旗标到达既有校验器,而不是放宽旧旗标的含义。

# R13 预注册草案 v0.1:教室改革——DIVE 窗口主权移交(live-DIVE)
状态:DRAFT,待总设计师审阅。起草:Claude,2026-08-30 晚。
设计流程:9 员勘察-设计-批判工作流(5 路代码/日志/测试勘察 → 3 独立设计 → 对抗审查)→ 本合成案。
主席立案批文:「照您这么说的话 是时候打破天花板了 哈哈哈」。

## 一、案由与立论(证据链)
1. R12 终判:工人在位再教育双臂皆败;魔鬼卷三方(旧工/臂一/臂二)128 种子逐位一致。
2. **机理更正(R12 终判书补记一,立案勘察发现)**:DIVE 窗口的方向盘从来不在工人手里——
   - 训练态:WorkerWindowEnv 构造 OptionsEnv 从不注册 workers(worker_env.py:688-694),
     非 FARM 窗由冻结脚本 dispatch 整窗代跑(options_env.py:1704→1731+),连学习策略的前向都没有;
   - 考试态:eval_assembled 只注册 {FARM: callback}(eval_assembled.py:1179/1190),
     考卷上的 DIVE 窗同样脚本开车。魔鬼恒 DIVE ⇒ 整场考试工人零动作 ⇒ 逐位一致是平凡必然。
3. 另有三道与模式无关的职权铁栅:工人恒掩 a11(主线推进归经理,options_env.py:1618)、
   1-8 步禁踏 trigger 格(硬 ValueError,options_env.py:1107-1112)、观测视图
   dual-v4-asymmetric-v3 无任何窗口模式特征(混窗必状态混叠)。
4. 结论:约束不在教练在教室,且不止教室——**训练、考试、职权、感知四个面都要移交**。
   缺任何一面改革都不可表达:不移训练面=R12 原地重演;不移考试面=学了也考不出来;
   不移职权面=放进课堂也按不了下楼键;不移感知面=工人分不清自己在上哪门课。

## 二、命题
把 DIVE 窗口升格为一等 live 学习窗(工人逐微拍驱动、进梯度、死亡就地入 wage),
配套主权移交与感知扩容,使认证工人在 v2 经济下学会深层生存,且不遗忘浅层战斗力。

## 三、设计(骨架=计量立宪案,嫁接最小刀口的锚法与深水学堂的验收闸)

### 3.1 改革本体(在位修法,默认关旗)
- 新旗 `--worker-learning-window-scope {farm-only, farm-dive-v1}` 默认 farm-only,
  fail-closed 具名校验;farm-dive-v1 时 `_advance_to_learning_window` 的 DIVE 分支
  改走 FARM 同构 live 开窗(克隆 worker_env.py:1318-1338 现成模板:
  _win_begin(DIVE)+_consume_fuse_recovery(已支持 dive)+_drain);
  RESUPPLY 与 p_skip 干层 FARM 维持脚本快进(勘察结论:RESUPPLY 近单动作零梯度废窗,
  且 live-DIVE 已把深潜补给决策交给工人)。
- **路线裁定(采对抗审查建议):弃子类/注入路线,直接在位修改 worker_env.py/options_env.py**。
  代价=协议 bundle sha 旋转、冻结锚需重烤;换来法理零争议(无跨身份配对悬案)与零复制漂移。
  重烤兼任「默认关=逐位不变」的终极认证(见 4.1),把锚经济损失反转为免费回归测试。
- 旗关路径纪律:新实例属性一律 getattr 默认值读取(护 9 处 __new__ 壳测试)、
  默认路径零额外 RNG 消耗(护 G0'.a 逐位平行)、旧 WorkerSentinelCallback 零改动
  (R7 冻结 schema),R13 新审计走新哨兵类。

### 3.2 主权移交法条(最小放权)
- 仅当 scope=farm-dive-v1 且 self._win['opt']==DIVE:解禁 m[11]、豁免 1-8 踏 trigger
  格的强掩与硬守卫;**a10 剧情掩码不动**(安全条款,凭 stall 法证再议);
  FARM/RESUPPLY 窗掩码逐位不变(新测试档位判执法)。
- 考试面同法:eval_assembled 新旗 `--worker-window-registration {farm-only, farm-dive-v1}`
  默认 farm-only;旗开时注册 {FARM: cb, DIVE: cb} 并激活主权移交,放宽 FARM-only 校验;
  默认路径逐位不变。

### 3.3 感知扩容(新视图+迁移)
- 新视图 `dual-v5-window-mode-v1`:v4 全序 635+snapshot 前缀逐字节不变,尾部追加 12 维:
  [0:3] 窗口模式 one-hot;[3] dive_best_d/15;[4] stall 钟 (tau−dive_last_progress_tau)/140;
  [5] tau/TAU_CAP;[6] dungeon_level/15;[7] (char_level−dungeon_level) 归一;
  [8] 窗内累计击杀归一;[9] 窗内 hp 差分归一;[10:12] 保留零位。
  新名/新布局常量/新 sha 并列注册,冻结旧视图零接触。
- 迁移工具 migrate_worker_obs_v4_to_v5.py:认证工人 zip → R13 入学 zip,首层对新 12 列
  零填充 ⇒ 载入时刻 argmax 与认证工人逐位一致(D0 门,硬断言 ==100%);
  契约烘焙 worker_policy_observation_view=v5、worker_learning_window_scope=farm-dive-v1、
  parent_worker_zip_sha256 血统字段 ⇒ resume 逐键相等自然成立,
  **_ENVIRONMENT_RESTART_ALLOWED_DRIFT 九字段白名单一字不扩(开门不开洞)**。

### 3.4 计量立宪
- 「步」修宪:SB3 timestep = 任意 live 窗(FARM 或 DIVE)一个微拍。
- 步数定标公式(预注册,G0 实测后冻结):T = round2048(266240 / (1−s_G0)),
  s_G0 = G0 烟测 dive_live_share;硬顶 393,216(192 rollouts);s_G0>0.32 中止上报。
  预测(R12 臂二底数 s≈0.27):T≈364,544,墙钟约 55-70 分钟(107 步/秒实测已含 ff 前向)。
  立法理由:266,240 在 R12 是纯 FARM 课时,字面沿用=默许 FARM 课时暗降约 27%,
  在 C 门已是 R12 真死因的前提下等于复刻死法——FARM 课时不得成为无人立法的暗变量。
- sentinel_every 500,000→63,488(与 ckpt 对齐,run 内约 5-6 个时序点,修复 R12 只有
  中尾两点的计量事故);新哨兵字段(R13 新类):dive_live_windows/steps/deaths/descends、
  dive_share 逐点序列、死亡深度×剩余步数直方、按 dungeon_level 分桶的 FARM 窗 kills、
  value loss 分窗分解、a11 使用率。
- 计量恒等式测试:farm_live_steps + dive_live_steps == num_timesteps 增量。

### 3.5 训练配方(臂 α,基于 R12 配方逐旗标注)
【不动 19 旗】--worker --algo mppo --gamma 1.0 --max-steps 3000 --num-envs 4 --n-steps 512
--seed 22(留 22,保 R12 同种子纵向可比)--device cpu --allow-environment-restart-resume
--allow-manager-change --gradient-clip-mode separate-root-context-critic-v2
--manager-heuristic level-margin-1(自定步速课程:超等级才开 DIVE,R12 实测天然 73/27 配比
且教室最深,零新课程机器)--worker-action14-logit-bonus 2.5 --lr 0.0001 --ent-coef 0.005
--target-kl 0.01 --no-drink-sovereignty --worker-fast-forward-reward-credit terminal-death-only
(仍覆盖残余脚本段死亡;DIVE 死亡自动迁入 direct 路径就地入 wage,两路结构互斥无重复计罚)
--worker-additional-terminal-death-cost 0.0 --reward-economy v2(零修法:剥薪恒等式
W+=r−bonus 微拍级自动生效,下楼奖金依旧归经理;B 深度乘数/D 反躺平/递减死亡罚金
经共享窗口核在 live-DIVE 下自动同构生效)。
【改 3 旗】--total-steps 按 3.4 公式定标;--resume-from R13 入学 zip(3.3);
--worker-policy-observation-view dual-v5-window-mode-v1。
【增 1 旗】--worker-learning-window-scope farm-dive-v1。
【改暖机】--critic-warmup-steps 32768(16 rollouts,R12 为 8:critic 对 DIVE 内部状态是
分布外盲区,冻 actor 只训 critic 先行覆盖;全量重置不必要)。
【明确不加】a11 logit 引导(见 5.3,G0 验收不达标时的一级杠杆,非首发配置)、
pdive 配比课程(R13.2 备用)、任何工人侧正向下楼塑形(违「下楼奖金归经理」法条,
勘察已证:以 descend 收窗为条件的奖励=变相返还下楼奖金)。
【臂 β(contingency,不并行发射)】唯一差异 --worker-additional-terminal-death-cost 2.0。
激活判据(预注册):臂 α 考卷死亡直方中「d≥7 且剩余步数<500」占 dive 死亡比例>20%,
或分深度审计呈现残局送死模式(经济 v2 既有漏洞:d≥9 单杀收入 3.0+ 已超死亡罚金 ≤3.4)。

## 四、锚法与考试

### 4.1 重烤即认证(锚法核心)
在位修法旋转协议 sha,2_114/2_115 冻结锚身份作废,须重烤——同段种子复跑,零处女消耗:
- 分段止损:默认旗(farm-only)下先 8 种子迷你 diff → 32 → 128 全量;
- 判据:重烤 rows-sha 必须与旧冻结锚逐位相等,不等即 fail-closed 中止排查;
- 重烤四卷:魔鬼×旧工 a/b、M29×旧工 a/b,约 4h,兼任「默认关=逐位不变」终极认证。

### 4.2 考卷(全部复用 2_114/2_115 同段种子,零处女消耗)
- 门 A/B 卷:魔鬼×新工,--worker-window-registration farm-dive-v1,128×2 池;
  基线=重烤魔鬼×旧工(脚本在位者=部署形态,对抗审查裁定:这才是改革存在性主张的
  正确对照——「学习策略在深层活得比正典脚本好」);
  开卷强制断言 rows-sha ≠ 锚 + DIVE 窗司机审计(把 R12 死因变成 fail-closed 前置检查)。
- 门 C 卷:M29×新工,farm-only 注册(与锚协议同构,隔离 FARM 遗忘变量),对重烤 M29 锚;
  行为级确认:训练第 0 步入学 zip 的 M29 小卷(16 种子)≡ 认证工人(护扩维手术归因)。
- 信息性考场(不设门,主席点菜):M29×新工全注册=部署形态摸底,同段零消耗。
- 池籍:确认池 2_116 仅凭主席令消耗;处女池 2_117-119/2_126-129 零接触;禁运段永不消耗。

### 4.3 四门判据
- 门 A 深层生存优越【沿用】:魔鬼卷 died 配对 McNemar UCB<0 且 died≤110/128。
  法证预埋:按 (char_level−dungeon_level) 分桶的死亡分解(课堂只教「强了再潜」而考卷
  含欠等级强潜,防把课程-考卷分布错配误读为改革失败)。
- 门 B 深度保持【沿用】:魔鬼卷配对 Δ深度 LCB>−0.10。兼任反怯懦闸:工人可能学会
  「不按 a11 磨 stall 保命」骗过门 A,骗不过门 B。
- 门 C 反遗忘【沿用,不放水】:M29 卷 ret≥0.95× 且 kills≥0.95× 锚。R12 真死因,
  本案三重答卷:零填充迁移+模式 one-hot 去混叠+暖机翻倍+FARM 课时保底(T 公式)。
  预表态(留主席核准):若 A/B/D 全过而 C 独败,R13.2 走 pdive 配比课程,不降门槛。
- 门 D 行为同一性【修宪 D-v2】:
  D0(新)迁移即时不变量:入学 zip 对认证 zip 探针 argmax ==100%(护手术归因);
  D1(沿用)FARM 探针态 argmax 一致率 <99%(证明学习发生);
  D2(新)≥500 个 DIVE 窗内状态上,新工 argmax vs 正典脚本 dispatch 动作一致率 <99%
  (对 R12 结构定律的直接反证探针,=100% 即结构性失败,禁烧考试场)。
  D0-D2 全部零种子消耗、先于一切考场执行。

## 五、G0 定标与验收闸(全绿方可发射)
1. 吞吐定标:烟测(≥8192 步)实测 s(dive_live_share)与步/秒 → 按 3.4 公式冻结 T;
2. **课堂发生性硬验收**:烟测中每 10 个 live DIVE 窗 ≥1 次 descend 收窗;不达标=中止调参,
   不硬发射(防「学习面空转」重演 R12 的「采集面空转」);
3. 素颜探针:无任何 bonus 的确定性 argmax 下,a11 按压率与 descend 率公示;
4. logit-bonus 作用面勘验:确认该类旋钮只作用于训练采样还是同时作用于评测 argmax,
   勘验结论决定验收不达标时可否引入 --worker-dive-action11-logit-bonus(默认 0)作为
   一级杠杆(引入即重跑 G0 全套验收);
5. 864 旧套件全绿 + 新测试档全绿(位判平行/主权法/工资恒等式 Σw≡R−bonus 在 DIVE 窗
   成立/死亡单次计账/v5 前缀逐位/迁移 100%/计量恒等式/哨兵 schema 闭合);
6. 训练熔断 14,400s(4h 防僵尸,主席判例);评测单场 2h。

## 六、风险与预案(按对抗审查修订)
- 风险1 激励真空×a11 冷启动(头号科学风险):剥薪后按 a11 直接报酬为零,logit 从未受训;
  防线=G0 硬验收(5.2)+素颜探针+预注册一级杠杆(a11 bonus)+二级杠杆(势函数深度塑形,
  policy_reward 层合宪槽位,凭令激活)。
- 风险2 FARM 灾难性遗忘(C 门,R12 真死因):三重答卷+T 公式保底;若败,法证须能区分
  「FARM 梯度被稀释」vs「FARM 输入分布漂移」(live-DIVE 改写同局后续 FARM 窗的
  状态分布)——分桶哨兵字段已预埋(3.4)。
- 风险3 残局送死套利:臂 β 预案+数值激活门槛(3.5)。
- 风险4 critic 分布外反噬:暖机翻倍+target-kl 第二道闸+value loss 分窗哨兵预警;
  暖机只覆盖零填充行为下的 DIVE 分布,学会下潜后的二次漂移凭哨兵序列法证。
- 风险5 dive_share 训练内漂移:冻结 T 不中途改,逐哨兵 dive_share>0.40 记异常台账。
- 风险6 重烤不等:默认路径存在漂移,fail-closed 中止排查(8 种子迷你 diff 前置止损)。
- 风险7 组装体协同适应缺口:经理从未与「会开 DIVE 车的工人」协同,2_116 确认场若用
  组装体形态测的是从未存在过的组合——信息性考场先行摸底,经理再训列 R14。
- 风险8 跨轮断代:A/B 门换考试协议,R13 与 R12/R10 数字不可直接对比,LAUNCH_ORDER 立碑。

## 七、流程
勘察卷宗归档 → 主席审阅本案(八个裁决点)→ 修法+新测试档 → 864+新档全绿 →
锚重烤即认证(4.1)→ D0-D2 探针 → G0 定标+验收闸 → 冻结(sha)→ LAUNCH_ORDER →
臂 α 发射(~1h)→ 六卷考试(~6h)→ 四门对判 → 呈主席终裁 →(全绿)凭令 2_116 确认场。

## 八、待主席裁决的八个决策点
1. **路线**:在位修法+重烤即认证(本案推荐,法理零争议)?或子类/注入省 4h 重烤
   (对抗审查裁定工程脆弱+法理有隙,不推荐)?
2. **步数法**:批准 T=round2048(266240/(1−s_G0)) 公式+硬顶 393,216+s>0.32 中止线?
   或字面沿用 266,240(默许 FARM 课时暗降约 27%,不推荐)?
3. **臂数**:单臂 α + β 登记为 contingency(推荐)?或双臂并行(+1h 训练+考卷翻倍)?
4. **门 D 修宪 D-v2**(D0/D1/D2 三探针)与门 A 考试协议变更,正式批准?
5. **a11 杠杆权限**:G0 验收不达标时,预授权引入 a11 logit bonus 并重跑 G0(推荐)?
   或一律停下请令?
6. **送死套利杠杆**:臂 β 激活门槛(3.5 数值判据)预授权(推荐)?或个案请令?
7. **C 门预表态**:若 A/B/D 过而 C 独败,R13.2 走配比课程不降门槛——现在表态(推荐)?
8. **信息性考场**(M29×新工全注册部署摸底,零门零消耗,推荐跑)?

---
## 修订 v0.2(2026-08-30 晚,主席八点全批 + 步数法特批)
主席批文:「好 步数不够的话可以继续加 我在想步数可能也是一个主要的因素 别的按您说得来」。

1. **八个裁决点全部按推荐通过**:①在位修法+重烤即认证;②步数公式(经第 2 条修正);
   ③单臂 α + β 登记 contingency;④门 D 修宪 D-v2 + 门 A 考试协议变更;⑤a11 杠杆预授权
   (G0 不达标时引入并重跑 G0);⑥β 臂数值激活门槛预授权;⑦C 门预表态(独败走 R13.2
   配比课程,不降门槛);⑧信息性部署摸底场照跑。
2. **步数法修正(主席特批「步数不够可以继续加」)**:
   - §3.4 硬顶 393,216 降格为软顶:s_G0>0.32 不再中止,按公式重算、台账记异常、继续;
   - 新增「续训预授权」条款:终判后若法证显示训练不足——预注册判据:最后两个哨兵点的
     dive descend 率或 a11 使用率仍单调上升,或 value loss 分窗分解仍显著下降——
     预授权一条 +1×T 的续训腿(同种子同配方 resume,台账补记即可,无需新令)。
3. **科学告知(诚实记录)**:R11 试跑三已证明经理侧「没教够」假说死亡(2× 步数逐位不动),
   R12 败因是结构性(DIVE 零梯度),步数非当时约束;但 R13 的 a11 冷启动+全新状态分布
   使步数首次成为真实候选约束——主席直觉恰好用在对的地方,故第 2 条按批文立法。

---
## 修订 v0.3(2026-08-30 深夜,实施勘察后的设计修订:v4 视图路线)

### 三项勘察发现(每项都推翻 v0.1 §3.3 的一块前提)
1. **v5 视图的冻结连锁远超预估**:12 维追加块必须成为 controller_wire 布局新段
   (旋转 DUAL_WORKER_LAYOUT_FROZEN_SHA256)、必须扩 leashed_ppo 冻结参数计数
   (+384,三常量)、必须对 Adam 动量做同位手术(漏做即静默腐蚀)、且 asymmetric
   policy 拓扑由模块级常量锁死(13012/13024 不能共存一进程)——迁移工具从
   「零填充一步」膨胀为跨四文件的架构手术;
2. **critic 暖机结构性不可用**:--critic-warmup-steps 与 --reset-worker-critic
   绑定,而带契约的 checkpoint 禁止重复 reset(dual-v4 契约铁律),
   configure_critic_migration 亦拒绝重跑——v0.2 配方中的暖机 32768 不可执行;
3. **掩码特征即模式信号**:主权移交后,v4 观测的工人掩码段(617-632)语义如实
   反映新法——m[11] 仅在 live DIVE 窗内可为 1。「窗口模式感知」已内生于 v4,
   混叠盲区(双方 m[11]=0 的状态)恰是模式差异行为无关的状态(无可达楼梯时
   DIVE 最优行为≈FARM)。

### 裁定(依主席「别的按您说得来」授权)
- **R13 首发走 v4 视图路线**:零迁移工具、零布局旋转、零 leashed_ppo 手术,
  直接从认证工人 zip 续训(与 R12 臂二同款 resume 通道);
- v5 全套(显式模式 one-hot + DIVE 进展特征 + 参数扩容手术)降级为 **R13.2
  预备役**:环境侧观测代码已实现并注册(默认不可达),训练 CLI 明文封锁,
  若 v4 路线因感知不足失败(G0 课堂发生性验收或 D2 探针揭示),再行立案;
- 暖机替代:无暖机,靠 target-kl 0.01 + lr 1e-4 双闸(R12 臂二同款实跑先例,
  266k 步无塌缩);critic 对 DIVE 态的分布外冲击由 value loss 观察 + C 门守;
- 契约路线改为**白名单一键**:_ENVIRONMENT_RESTART_ALLOWED_DRIFT +=
  worker_learning_window_scope(R12 reward_economy 同款修法;无迁移 zip 可烘焙,
  「开门不开洞」的 zip 路线随 v5 一并列预备役);
- 门 D 修订:D0(迁移同一性)随迁移工具取消——起点即认证工人本体,同一性
  定义性成立;D1/D2 照旧。

### 实施清单(全部落地,2026-08-30 夜)
- python/diablogym/worker_env.py:_LEARNING_WINDOW_SCOPES + fail-closed 校验;
  __init__ 新参 learning_window_scope(显式形参防 **env_kwargs 泄漏);
  _advance_to_learning_window 新增 DIVE live 开窗分支(FARM 模板同构);
  step() 旗开分窗计量(farm/dive_live_steps、a11 请求/执行、收窗 reason 分桶,
  全部 .get 容缺护 __new__ 壳);v5 观测常量导入与维度分支(预备役);
- python/diablogym/options_env.py:OptionsEnv 新参 dive_live_sovereignty
  (默认 False);_worker_masks_and_distance 与 _win_step_worker 硬守卫的
  「live DIVE 窗内」条件豁免(a10 剧情掩码不动);v5 视图注册+12 维追加块
  构造(预备役);controller snapshot 门扩 dual 家族;
- train/train_ppo.py:--worker-learning-window-scope CLI+校验+交叉门
  (farm-dive-v1 ⇒ dual-v4 视图);make_env/partial/契约(farm-only 恒 None,
  rev26 不动)/白名单一键;dual 家族集合判定(policy 绑定断言+续训分类器);
  R13DiveAuditCallback(独立新类,r13_dive_audit.jsonl,不触冻结哨兵面);
  v5 常量注册但 CLI 封锁;
- train/eval_assembled.py:--worker-window-registration CLI(默认 farm-only);
  evaluate() 双注册 {FARM,DIVE}+dive_live_sovereignty 透传(旗关连关键字都
  不出现);instrumentation 三修(全键统一替换保 action14 对账闭合/分歧参照系
  按窗口模式动态取/参数 fail-closed);DIVE 常量+漂移检查;
- tests/test_r13_live_dive.py:四测新档(具名校验/旗关铁栅与零泄漏/旗开
  主权+计量恒等式/考试面 fail-closed)。

### 已过闸门(截至本修订)
- 27 项核心执法测试(G0'.a 逐位平行等四档)全绿;新档 4/4 全绿;
- 进程内烟测:DIVE live 窗开出、a11 解禁且 71 次真实执行、1 次真实下楼、
  1 次深层死亡就地入 wage、计量恒等式 4000/4000;
- **8 种子迷你 diff PASS**:默认旗全考卷行与冻结锚 r10-cand-a 逐位相等
  (0/8 失配)——「默认关=逐位不变」在轨迹级成立;
- 进行中:864 全量套件、G0 烟测(旗开 8192 步全链路)。

### 已知限制(诚实登记)
- 考卷 row/agg 架构零扩(锚重烤等式所需):DIVE 窗工人工资/击杀在档案中
  并入 nonfarm_r/nonfarm_kills,不单列;深层行为法证以训练侧
  r13_dive_audit.jsonl 与考卷 depth/died/mode_seq 为准;
- override_rate/cap_rate 在 farm-dive-v1 注册下含 DIVE 窗分母,与旧卷非同
  口径(跨轮读榜须注意);
- worker_calls 在双注册下含 DIVE 拍(engage 计数器共享,对账闭合)。

---
## 终章 v0.4(2026-08-30 深夜,G0 定标与冻结)

### G0 烟测两轮收据(各 8192 步全链路,resume 认证工人真身)
- **G0-1(素颜)**:dive_share 0.508;52 个 live DIVE 窗仅 2 次真实下楼
  (0.38/10 < 1.0)——课堂发生性验收未达标,按 §5.2 中止调参;
  但 a11 被按压且执行 255 次、1 次下楼 1 次深层死亡就地入 wage,
  主权移交机制本身工作正常;
- **杠杆启用(主席裁决点 5 预授权,台账 lever_activated)**:
  --worker-dive-action11-logit-bonus 2.0,a14 先验同款机制(合法行才加、
  梯度照流、rollout 与评测同分布);a11 在旧法域恒被掩,先验自动惰性;
  post-load 双写(活体属性 + policy_kwargs)保证先验烘焙进产物 zip,
  考卷不蒸发;契约 None-off 键 + 白名单入册 + 绑定断言全套;
- **G0-2(杠杆)**:**验收通过**——descends_per_10_windows 1.72(≥1.0),
  58 窗 10 次真实下楼,a11 按压 689 次,dive_share 0.383。

### 步数冻结(主席步数特批适用)
s_G0 = 0.383 > 0.32 软顶(v0.2 修订:不中止,记异常继续)→
**T = round2048(266240/(1−0.383)) = 432,128(211 rollouts,约 67 分钟)**,
超原软顶 393,216,依主席「步数不够可以继续加」批文放行;续训预授权条款
(+1×T)照 v0.2 待命。

### 闸门状态
- 866+4 套件绿(2 个中途编辑读到半成品的档隔离复跑通过;清机终审轮
  发射前置);tests/test_r13_live_dive.py 4/4 绿;
- 8 种子迷你 diff PASS(0/8 失配);四锚全量重烤即认证进行中;
- 发射件:run_r13_rebake.py(锚法)、probe_r13_identity.py(门 D)、
  run_r13_arm_a.py(T=432,128 冻结、门 D fail-closed 前置、六卷考试、
  魔鬼卷开卷 rows-sha≠锚 强制断言)。

### 冻结声明
本文件自本行以下不再修改;正本复制为 r13-PREREG-FROZEN-20260830.md,
以该文件 sha256 为准。驱动:train/runs/r10-staging/run_r13_arm_a.py。

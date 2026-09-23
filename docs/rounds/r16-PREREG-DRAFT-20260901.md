# R16 预注册草案 v0.1:修宪——尺子、门、考试法、记账、视野
状态:DRAFT(施工中)。主席批文 2026-09-01:「没问题 立刻进行修复!」
依据:r16-FRESH-EYES-AUDIT-20260901.md(七审计员+法官,190 万 token,
38 候选致因→12 簇,Top 8 当庭验证三项坐实)。

## 一、案由(判词)
> 问题不在工人,在尺子和门。R16 首要不是再训工人,而是修宪。

R12→R15 五层根因链的后两层(资质、成长引擎)建立在三个失真读数上:
战备表门槛按引擎算术不可达(常数 0)、部署探针读尸检值(假零成长)、
考卷 argmax 与训练采样脱节(假不下潜:同工人同种子 kills 35.1 vs 47.4)。
真实约束是:探索宏视野 12 格(可达花名册 40-65%)、1800 拍就赶人下楼的
场景预算、告诉工人"活着不值钱、磨等级不划算"的记账。

## 二、修宪条款(全部默认关;旧法路径逐位不变为发射前置)
| 条 | 簇 | 法条 | 落点 |
|---|---|---|---|
| 1 尺子 | C1 | 战备表 v0.2:按引擎算术反推,L2=(clvl2/AC9/dmg6),删 HP 列(clvl 的确定函数);`readiness-v3` 教练 | worker_env |
| 2 门 | C2 | 清场比改击杀口径 kills/(kills+alive−golem),golem 槽剔除;逃生阈值 **1.0**(G0-3/4/5 裁决:0.75 时楼层残怪堵住 a11 主线,57 窗 39 stall;无活怪即刻下潜,不可达残怪由榨干旗兜底) | worker_env / probe v3 |
| 3 考试法 | C3 | `--worker-decoding sample`(逐局定种、单线程可复现)进考卷与部署探针;FARM 开窗清零 layer_clock(`reset_layer_clock_on_window`) | eval_assembled / options_env |
| 4 记账 | C7/C8 | `worker_hp_loss_price`(非对称失血计价)、`worker_potion_pickup_bonus`、a13 logit 先验、开放喝药主权;`worker_no_progress_timeout_credit=zero`(超时≠死亡);经济 v4(反躺平按微拍计、击杀清零) | worker_env / leashed_ppo / env |
| 4″ 超时零记的法域下传 | C8 | `timeout_credit=zero` 触发两条冻结训练法条(G0 两次触雷):worker_env 审计以 `no_progress_timeout_credit` 键把法域随 info 下传;leashed_ppo `_update_info_buffer` 新增零罚金分支(超时仍终止窗、分账恒零、reward==工资、死亡三账为零),`validate_worker_onpolicy_pg_receipt` 汇总条件由严格负放宽为非正;旧法分支/键集一字不动 | worker_env / leashed_ppo |
| 5 视野 | C5/C6 | `explore_global_fallback`(a10 窗内无候选时全图 BFS 回退——A 队实测 8 种子几乎不触发,375→375 杀);**`explore_global_hunt`(窗内无可见怪时全图 BFS 朝最近存活怪推进;复现审计正对照 375→670 杀,clvl 2→3)——本条为 R16 主力**;`progress_far_tiles`(锚点集语义:与起点/历次进展锚切比雪夫距离 ≥ far 的新格才计进展);`farm_scene_cap` 可配置 | env / options_env |
| 5′ 越权声明 | — | hunt/fallback 均在同一观测后二次调 `bridge.local_map(radius=112)` 做全图 BFS——工人观测(25×25 窗)看不到全图,属 a11 同款的环境侧越权信息;宏 a10 的"禁止同观测二次重规划"文档条款在此二开关下明文豁免 | 预注册明写 |
| 4′ 经济 v4 | C8 | v4 = v2 全字段同源 + `idle_counts_micro_beats` + `idle_reset_on_kill`;反躺平阈值沿用 300 但单位改微拍(≈60 秒游戏时间无击杀即 farm×0.5);G0-3…6 六跑未见异常,量级维持 | env |
| 协议 | C4/C6/C11 | 局末指标死亡前一拍采样 + 存活分层报告;局长 ≥6000 拍(契约 max_steps 白名单);致死种子登记不剔除(报告分层) | probe / train_ppo |

## 三、锚法(R16 = 新世界)
- 旧法锚(r10-cand/r10-anchor)在默认旗下继续**逐位重现**——这是"默认关
  零漂移"的执法,照旧跑 run_r13_rebake.py;
- 新法锚**重铸**:认证工人在 R16 考卷配置(采样解码、hunt、场景 3600、
  时钟清零、v4 经济;协议 3000 拍不动)下六卷各 128 种子另立档案 r16-anchor-*
  (meta.protocol.r16_environment 立碑),与旧世界数字**不可直接比较**。

## 四′、G0 定标记录(2026-09-02 01:50–02:40Z 本地)
| 跑批 | 变量 | live DIVE 窗 | descend | stall | /10 窗 | 结论 |
|---|---|---|---|---|---|---|
| G0-1/2 | 全开 | — | — | — | — | 触雷两条冻结训练法条(超时零记),修法 |
| G0-3 | 全开,清场 0.75,far 3 | 57 | 4 | 39 | 0.70 | 未过 |
| G0-4 | far 0 | 41 | 4 | 21 | 0.98 | 未过(stall 减半) |
| G0-5 | a13 0 | 60 | 2 | 33 | 0.33 | 未过(a13 非病因) |
| G0-6 | 清场 **1.0**,far 0 | 26 | 3 | 4 | **1.15** | **通过**;s=0.181 → **T=325,632** |
裁决:清场逃生阈值 1.0(残怪堵路致失速的机理:triggers/monsters 全层可见,失速非楼梯未知);
C6 远距进展搁置(far=0;v4 微拍反躺平已封踱步漏洞);a13 先验 2.0 维持;hp 计价 0.1 维持
(−1.9/窗,工资三成以内);拾药记账 2.0 维持。

## 四、配方(v0.3,G0-6 定标,T=325,632)
臂 r16-arm-a(run_r16_g0.py 同配方,只差 total-steps):认证工人续训;
教练 readiness-v3;economy v4;scope farm-dive-v1;max_steps 6000;
托管 0.5×d^1.6 + 战备条件托管(沿用);a11 先验 2.0;a13 先验 2.0(a11 同款量级);
hp_loss_price 0.1(满血 ~80 HP 流失 ≈ 8 单位 ≈ L1 死亡罚金 26 的三成:失血先于死亡就疼);
potion_pickup_bonus 2.0;timeout_credit=zero;explore_global_hunt(主力,fallback 不开);
progress_far_tiles 0(C6 搁置);farm_scene_cap 3600(= 旧 1800 × 局长倍率 2);
reset_layer_clock_on_window;drink_sovereignty 开放;readiness-v3 清场阈值 1.0;
训练解码采样如常。
G0-6 定标:s=0.181,T=round2048(266240/(1−0.181))=**325,632**;hp 计价 −1.9/窗。
考卷:四门沿用 eval_assembled 3000 拍协议,新法锚以 `--worker-decoding sample`
+ R16 环境旗(meta.protocol.r16_environment)另立档案;部署探针 v3 第 6 参数
JSON 形态(readiness-v3 / 主权开 / v4 / hunt / cap 3600 / clock reset),
max_steps 6000,sample 解码,16 种子 2114000-2114015;臂与认证工人(新法部署锚)同形态对照。
发射链:run_r16_anchors.py(六卷新法锚)→ run_r16_arm_a.py 325632(训练 → 门 D →
六卷 → analyze_arm_gates.py <arm> r16-anchor → 部署探针 v3 臂/锚)。

## 五、判据(修订)
- 主指标:部署探针(采样解码、死亡前采样、存活分层)的 clvl/AC/kills/
  depth 每千拍归一 + 存活率,对照 R16 新法锚;
- 四门沿用但**对新法锚**:A 存活优越、B 深度保持(锚改为 R16 脚本工人)、
  C 反遗忘、D 行为同一性;门 B 的旧锚(莽夫深度)按审计 C11 废止。

## 六、施工分区
A 队 env.py(视野回退/远距进展/经济 v4);B 队 eval_assembled+探针(采样
解码/死亡前采样);C 队 leashed_ppo+train_ppo(a13 先验/全部 CLI 契约白名单
接线);主刀 worker_env+options_env(v0.2 表/击杀清场/血量记账/超时法/
场景预算/时钟清零)——已落刀,执法档 12/12 绿。

## 冻结声明
四队合龙(A/B/C/主刀 交付均入台账)+ 套件全绿(876 passed,重烤前一版;最终字节
套件与 4×128 旧法锚逐位重烤由 r16_gauntlet.sh 执法,PASS 为发射前置)+ G0-6 定标
(T=325,632)。本文件冻结为 r16-PREREG-FROZEN-20260902.md 并以 sha256 入台账;
冻结后配方任何改动须以修正案另立文件。

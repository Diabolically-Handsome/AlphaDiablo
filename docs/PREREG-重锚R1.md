# PREREG-重锚R1:protocol-v3 开元 · BC 基底重生成(终稿)

状态:**终稿**(批评者面板 wf_ea94119c-207 过审:4 批评者、3 blocker /
10 major / 12 minor 全部落地)。冻结纪律循 PREREG-v30 先例:**commit 即
公证**——冻结 commit = 收录本文档终稿与驱动器 train/run_reanchor_r1.py
的那次 commit;正文不回填任何自指 sha,冻结 sha 仅由驱动器写入台账
FREEZE_SHA 事件,判决附录引用之。

## 案由

外源审计(卷宗 86c7270)裁定 protocol-v2 封卷;三份存量 bc_report.json
均无 v3 身份回执,而 train_ppo._validate_bc_report 拒收无回执 BC 产物。
重生成理由**三分而非一概**(面板修正:审计结论转述须忠实):

- **worker**——审计当日重验 held-out top1=1.000(质量再证),重生成纯系
  身份要求。**前科来历限定**:盘上 bc-worker/bc_report.json(mtime
  2026-07-12 12:48)之键集既非 v23 生成器亦非现行 v3 生成器所能产出,
  生成器未入库、且已就地覆写 v23 原始报告(无 _previous 存档);经复核其
  policy_sha256/demos_sha256 与盘上 2026-07-10 00:42 工件逐字一致,认定为
  外源审计重验之物证残留,证据等级 = **来历不明确的旁证**:top1=1.000
  仅作 R-W 预测之先验,不得充当任何闸门证据。
- **manager**——v2 报告同池比值 0.6498 FAIL(2026-07-09,同池口径
  50.379/77.531,可比)本身成立,重生成兼具身份与假设复核双重理由。
- **flat**——旧比值 0.5974(2026-07-09)之口径被 v3 脚本注记认定有偏
  ("把种子难度差当成策略损失",分母系示范池教师均值),与本案同池闸
  **不可直接换算**;重生成兼具身份与方法矫正双重理由。

外源审计关于 BC 三件套之结论(worker 重验 1.000;manager/flat 不可信)
迄未见于任何在册档案,本段即为其入册转录。三份存量报告世代判定以
**字段集为准**(均无 schema_version/protocol_version 键 = pre-v3),
mtime 仅作旁证。

## 一战一处方

本案处方 = "在 v3 世界重铸 BC 基底并验收回执"。本案**不烧金种子
9000-9031**(定锚是 R2),**不动环境侧一行代码**,**不训练任何 PPO**。
探针池 7000-7031 仅作 manager/flat 脚本内建重放基准(闸门用途,合规)。
面板查证:金种子/环境侧/PPO 三项无蠕变。

## D1 世界身份(先决,全部机器可执行,驱动器预检逐条断言)

- **W1 冻结公证三步**(循 PREREG-v30"commit 即公证"):① 面板意见落地
  后定稿,单次 commit 收录本文档终稿 + 驱动器 + ROADMAP 注记(此后 D1-D5
  正文不可改);② 发车预检断言 `git status --porcelain` 为空,并把
  `git rev-parse HEAD` 逐字写入台账 FREEZE_SHA 事件;③ 断言
  `git log -1 --format=%H -- docs/PREREG-重锚R1.md` == HEAD 同一 commit
  (判决附录系判决后新 commit,不受本预检约束)。驱动器**不内嵌**冻结
  sha,禁止任何"sha 抄录 commit"。
- **W2** `eval_contract.PROTOCOL_VERSION == 3` 且评测契约
  `SCHEMA_VERSION == 2`(注意与 R-S 的 BC 报告 schema_version==1 系
  **两个不同常数**,分别断言,不得混用)。
- **W3** `engine_binary_path(ROOT)` 唯一定位成功;断言环境变量
  `DEVILUTIONX_REF` 未设置,且 bootstrap.sh 逐字含
  `ENGINE_REF="${DEVILUTIONX_REF:-34c4cfc2e733240ac717f23bba2def887c793008}"`
  (该行本身含 env 覆盖语法,默认值即钉死 SHA;二进制字节身份另由 W5
  bundle 绑定)。
- **W4** 游戏数据目录含 `diabdat.mpq`(全内容世界;sha256 记入台账)。
- **W5** `train_ppo._implementation_bundle_sha256()`(私有函数,居
  train/train_ppo.py;eval_contract 无同名 API)取样为**阶段粒度**:
  每阶段发车前、收车后各取一次并记台账(三阶段共六次);任一阶段前后
  两值不等、或任一取样 ≠ 案级首取值、或 ≠ 该阶段 bc_report.json 内
  implementation_sha256 → 该阶段产物**包括科学 FAIL 报告**一并作废,
  记 OPERATIONAL。注:脚本内建漂移断言仅覆盖 PASS 写权重路径,科学
  FAIL 报告的世界身份由本条驱动器前后取样兜底。
- **W6** 台账完整记录 `eval_contract.runtime_versions_identity()`(钉死
  numpy/gymnasium/torch/stable-baselines3/sb3-contrib/tensorboard 六件套
  及 Python 实现/版本/cache_tag;任一漂移该函数自身 fail-loud)。
- **W7 采集夹具(v2 冻结经理)**:`train/models/v22-h-manager/policy.npz`
  在位,且 `sha256 == eval_contract.DEFAULT_MANAGER_SHA256`
  (`0f2264860b0960e7951efd424836b90c09c002cebca7bf8109fd669b13be63d7`),
  记入台账。**合法性登记**:该文件系 v2 时代冻结产物,本案仅作 S1 示范
  采集夹具(窗口调度器,固定函数输入)使用,不是本案任何科学结论的
  被试、不构成 v2 基线复用;其身份已被 v3 评测契约钦定为默认经理
  (DEFAULT_MANAGER_RELATIVE 同一路径、同一常量),故与 v3 开元不冲突。
  bc_worker.py 构造 WorkerWindowEnv 未传 manager_sha256 钉死参数,本预检
  是该依赖唯一的先验闸;train_ppo 加载期将以报告 manager_npz_sha256 对
  当时盘上文件复验。该夹具的 v3 世代更替属 R2 及以后案。
- **W8 前科封存**:发车前对 train/runs/bc-{worker,manager,flat}/ 现存
  bc_report.json、policy_sd.pt、demos.npz 逐一 sha256+mtime 入台账
  (PREV_ARTIFACT 事件;含 bc-worker 报告 mtime 晚于权重的错位事实,
  原样记录、不解释、不处置)。声明:其将被各脚本 begin_output_attempt
  迁往同目录 _previous/<time_ns>/,该迁移属**封卷保全**(位置变更、字节
  不变),不违反运行档案纪律;判决附录须含"前科 sha ↔ _previous 落位
  路径"对照表,任何字节差 → OPERATIONAL。

## D2 三件套配方(脚本逐字执行,不改任何一行;顺序 = 关键路径优先)

执行顺序:**worker → manager → flat**(worker 是 v3 赛季教师,最先出分)。

| 阶段 | 命令 | 内建闸门(脚本自带,fail-loud) | 预计时长(估算,非依据) |
|---|---|---|---|
| S1 | `.venv/bin/python train/bc_worker.py` | **G1-数据侧**(= PREREG-v23 G1 之数据半闸):held-out top1≥0.95 ∧ 门槛类(全集≥300 样本)召回≥0.85;类加权重训 = 唯一重试;示范种子恰 100-227 零兜底;恒掩 11/12 零混入 | ~1-2h |
| S2 | `.venv/bin/python train/bc_manager.py` | 无记忆比值闸:BC(7000池)/教师(同池) ≥ 0.85 | ~0.5-1h |
| S3 | `.venv/bin/python train/bc_flat.py` | 同型比值闸 ≥ 0.85 | ~1-2h |

**运维纪律**(驱动器负责,对表条款:驱动器 = train/run_reanchor_r1.py,
随冻结 commit 入库,仅作编排、不改 train/ 任何脚本;正文与驱动器实况
一致):

- **并发纪律**:三阶段严格串行;每阶段发车前断言机器空闲——
  `pgrep -f 'train/(bc_|train_ppo|eval_assembled|run_v[0-9]+)'` 零命中,
  否则不发车(P6 型呈报)。案执行期间禁止本机启动其他训练/评测任务。
  理由:比值闸为回报制不受负载影响,但 4h wall-clock 直接受负载影响,
  负载假超时会白耗每阶段仅一次的重跑额度。
- **超时**:每阶段 wall-clock 上限 4h;发车一律
  `Popen(..., start_new_session=True)`,超时 `os.killpg(SIGKILL)` 后必须
  `proc.wait()` 回收(照 run_v30_relay 先例)。经查三脚本为单进程(引擎
  进程内加载、无 DataLoader worker),孤儿引擎风险为零,本条系防御未来
  变更的纪律。SIGKILL 遗留的引擎 scratch 由 env 注册表锁下次初始化自动
  回收,驱动器**不得**手动删 scratch;\*.tmp.\* 半成品由下次运行覆写
  自愈,同样不手动清理。
- **时长实证**:每阶段实际起止 wall-clock 记台账(前科无时长记录,表中
  预计时长视为估算);S1 超时重跑前核对日志"采集 n/128 局"进度行以区分
  卡死与慢速并记台账,但 4h 硬线不因进度豁免;实测时长为 R2 及后续案的
  预算校准依据。
- **日志**:stdout+stderr 合流 `train/runs/reanchor-r1/logs/S{n}-{role}-{ts}.log`。

## D3 R 线(可证伪预测 + 判词,先写后跑)

**判词来源条款**:阶段判词唯一来源 = bc_report.json 的判词键
(data_gate/hypothesis/memoryless_hypothesis)+ R-S 回执闸;**脚本退出码
不作判词依据**。判词键 ∈ {PASS, FAIL} 且回执过闸 → 科学判词;报告缺失、
判词键 RUNNING/缺失、回执缺失/失配 → OPERATIONAL。三分
{PASS, FAIL, OPERATIONAL} 穷尽全部出口(W 线不过 = P6 不发车,驱动器
崩溃 = P1)。

- **R-W(工人教师)**:预测 **PASS**(先验:审计重验 top1=1.000,来历
  限定见案由)。
  - PASS → 登记"v3 教师(数据侧)就位";判词强制携带
    (pairs, held_out_top1, 逐类召回表, class_weighted_retry) 与
    policy_sha256/demos_sha256。**G1 判词限定(v23 附录 C 义务承继)**:
    G1-数据侧只证明 BC 能在教师自己的轨迹分布上复刻教师;v23-G1 之组装
    重放半闸(eval_assembled ≥ 0.85×基线)因基线属定锚案职权,**顺延至
    R2 实测**;R1 判决书不得宣称"学习工人可平替脚本"。
  - FAIL(含唯一重试后)→ **全案停机呈报**。法理:同配方前世界满分,
    FAIL 优先解释为世界级异常,S2/S3 读数在该背景下不可解释,worker
    先行使停机零沉没成本。停机后 S2/S3 之补跑须经总设计师书面裁定,以
    补充登记入册(不改本案 R 线),其结论须标注"教师缺位背景下采集"。
- **R-M(经理无记忆假设)**:预测 **FAIL**(前科 0.6498,同池口径可比;
  机制候选 = env.exhausted 包装器状态不可观测)。
  - FAIL → 登记:"在注册配方(64×64 MLP/10 epoch/示范池 100-227)下,
    经理教师不可被 303 维无记忆 BC 以 ≥0.85 比值克隆(v3 世界);机制
    候选 = env.exhausted 包装器状态不可观测。除名依据为**工程不可用性**,
    非'非无记忆函数'的存在性证明。"M-BC 臂 v3 时代除名,未来选举
    fresh-only;复活需教师改造案(另立预注册;涉环境侧 → 总设计师亲批)。
  - PASS → 登记:"v3 世界下,该教师可被注册配方的 303 维无记忆 BC 以
    ratio=<数值> ≥0.85 克隆";与 v2 前科之差异**不作单一归因**,候选解释
    并列入册:(a) v2 回跳/换层结算缺陷;(b) v3 语义变更致教师状态依赖
    减弱(前科或为 v2 世界真性质);(c) 统计波动。ratio ∈ [0.85, 0.90)
    加注"边缘通过"。臂保留资格仅以 v3 读数为据,不背书归因。
  - 判词强制携带 (pairs, bc_replay_7000, teacher_7000, ratio)。
- **R-F(平面无记忆假设)**:预测 **FAIL**,但**置信度低于 R-M**(前科
  口径限定见案由:旧 0.5974 与同池闸不可换算,仅方向性证据;开牌无论
  何向均如实入册、不作事后追认)。判词模板同 R-M 型(296 维;机制候选 =
  停滞钟 _clock 包装器状态不可观测);携带
  (pairs, bc_replay_mean_7000s, teacher_replay_mean_7000s, ratio)。
- **R-S(回执闸,三份报告逐一;以 train_ppo 实码为准,不另立弱化子集)**:
  - **PASS 报告**:驱动器调
    `train_ppo._validate_bc_report(policy_sd.pt 路径, 对应闸名,
    verify_replay=False)` 全部断言通过即 R-S PASS——含键集**精确相等**于
    `_BC_PASS_KEYS[闸名]`(多键与缺键同罪)、schema_version==1(此系 BC
    报告 schema `train_ppo._BC_REPORT_SCHEMA_VERSION`,非 W2 之评测契约
    SCHEMA_VERSION==2)、protocol_version==3、implementation_sha256==
    当前 bundle、**generator_sha256 == 当前对应 train/bc_*.py 文件 sha
    (等值,非仅在场)**、policy/demos 字节级绑定、worker 证据重算、
    worker 报告绑定当前冻结 manager NPZ。重放重算(verify_replay=True
    全量口径)留给消费闸:R2 与赛季训练加载期由同一验证器全量执行。
  - **FAIL 报告**(验证器设计上拒收 FAIL,由驱动器按同一标准手检):
    键集必须恰等于 `_BC_PASS_KEYS[闸名]` 去除 policy_sha256(worker 另
    去除 demos_sha256;**manager_npz_sha256 系 worker 恒有 provenance,
    FAIL 亦必须在场且等于 W7 台账值**);回执四键(schema_version/
    protocol_version/implementation_sha256/generator_sha256)俱全且逐一
    等于当前世界值(generator 同样取等值口径)。
  - 任一条不满足 → OPERATIONAL(产物作废;归档由下次发车时脚本
    begin_output_attempt 保证)。**科学 FAIL 不豁免回执闸。**

**案级判词(穷尽)**:R-W PASS ∧ R-S 全过 → **案结**(R-M/R-F 各自
PASS/FAIL 均为合法结局);R-W FAIL → 停机呈报;任一阶段重跑额度耗尽后
仍 OPERATIONAL → 停机呈报;ANOMALY(见 D4)→ 停机呈报。案级判词强制
携带:三阶段判词 + 全部回执 sha + 台账路径 + 冻结 sha。

## D4 P 线(操作性意外,穷尽)

- **P1 驱动器顶层异常/人为中断** → DRIVER_EXCEPTION 台账 +
  NEEDS_ATTENTION。**续跑幂等判定**(按阶段,唯一事实源 = canonical
  bc_report.json;脚本无内建跳过,bc_*.py 每次启动即把 canonical 产物
  归档 _previous 后从零重跑,故跳过必须由驱动器判定,禁止误发车):
  - (a) 终局-PASS:R-S PASS 分支全过 → 跳过该阶段;
  - (b) 终局-科学 FAIL:R-S FAIL 分支全过 → **S1:无论首跑续跑,一律
    即时触发 R-W 停机呈报,不得发车 S2/S3**;S2/S3:不重跑(禁止加试),
    台账登 STAGE_SKIP_FAIL 后继续;
  - (c) 其余一切(RUNNING 残留/缺键/多键/不可解析/缺文件)→ 视为未完成,
    重跑;此重跑计入该阶段 OPERATIONAL 额度(**每阶段累计 1 次,不分
    P1-P7 成因;第二次任何 OPERATIONAL → 停机呈报**)。RUNNING 残留不
    手工清理(重跑时脚本自动归档;停机则原样封存,train_ppo 验证器天然
    拒收,无误吃风险)。
- **P2 exclusive_lock 冲突** → OPERATIONAL。锁系 fcntl.flock 建议锁,
  内核随进程死亡(含 kill -9)自动释放;残留 .bc.lock 文件不持锁、无害,
  **禁止以删除锁文件作为清理手段**(删除持有中的锁文件会使互斥失效)。
  冲突必意味着存在存活持有者:`lsof` 定位(锁文件内 pid= 行仅系上一
  持有者线索,不可为据),确认为非本案进程并终止后重试一次;再冲突 →
  停机呈报。锁在脚本入口、begin_output_attempt 之前获取,P2 不伤盘上产物。
- **P4 provenance 漂移**(脚本内建断言或 W5 阶段粒度取样触发)→
  OPERATIONAL,重跑一次;再犯 → 停机呈报。
- **P5 超时(>4h)** → killpg + wait 清场,OPERATIONAL,重跑一次;再犯 →
  停机呈报。
- **P6 预检不过**(W 线任一断言失败/机器不空闲)→ 不发车,呈报。
- **P7 非零退出但报告无 FAIL 判词**(RUNNING 占位/缺失/键残缺;典型:
  种子纪律断言)→ OPERATIONAL,重跑一次;再犯 → 停机呈报。科学 FAIL 的
  判定标准(穷尽)= 报告判词键=="FAIL" ∧ R-S FAIL 分支全过;缺一律按
  OPERATIONAL,不得登记为科学结论。
- **ANOMALY(科学异常,不占重跑额度)**:阶段日志出现"比值闸无定义"
  (同池教师均回报 ≤0)→ 不重跑(固定种子下重跑零信息增量,且"教师在
  v3 世界均回报 ≤0"系应即呈报的科学异常),直接停机呈报总设计师。

## D5 台账、日志与纪律出处

台账 = `train/runs/reanchor-r1/gate_ledger.jsonl`(JSONL 逐行事件,只
追加;续跑先读旧台账并断言逐行可解析,不可解析 → 停机呈报)。每阶段
入册字段至少含:阶段名、发车/收车 wall-clock、退出码、日志路径、前后
两次 implementation_bundle_sha256、判词、回执 sha(policy/demos/
manager_npz,如适用)、科学数字(D3 判词纪律)。NEEDS_ATTENTION 同目录。

**两条独立纪律及出处**(面板修正:不再悬空外引):① **文档冻结纪律**
——面板过审冻结后 D1-D5 正文不可改;判决与更正一律以文末附录追加,
附录一经落账同样不可改,后续更正以新增更正条目进行(先例:PREREG-v30
附录,commit 时间戳即公证)。② **运行档案纪律**——pre-v3 运行档案为
不可改法证记录(出处:README "Pre-v3 archives remain useful as
immutable forensic records"),本案对其唯一允许的操作是 W8 登记后的
_previous 保全迁移。

## 后续案预告(不在本案执行)

**与 ROADMAP 时代切换注记之序列对表**:"BC 基底重生成 → 教师重铸 →
定锚重测"三步折叠为两案——**R1 兼营前两步**(worker BC 即 v3 赛季教师
之重铸;经理 npz 教师系冻结工件不重铸,身份由回执 sha 锁定;脚本教师
无需重铸、只需 R2 重测),**R2 营第三步**(定锚重测)。三步之外不存在
隐案;"全部完成前任何课程战役不得发车"以 R2 判决落账为完成标志。

**R2 定锚案** = 在 v3 世界金种子重测四方:① 科学锚系 = 140.3 阵容
(M29-fresh × v28-worker-leg1,v29 史高);② launch 锚 = **现任组装体
112.4 阵容(法庭合规裁定,v30 判决⑤连任在案)**;③ 王座 v24-golden;
④ 脚本教师们——以此建立 v3 launch/science 双锚,新旧头衔对应关系照
v30 D3 法庭口径逐字承继,不因重测改变任何在位名分。金种子每案一烧、
手动点火、须新预注册 + 总设计师知会。**A 复赛权**(一次性,法庭条款,
存续于 ROADMAP)解锁条件同步顺延:须待 R2 之 v3 新锚落地后方可动用,
锚 = 届时合规裁定的现任组装体;本案与 R2 均不消耗该权利。R2 之后课
③④⑤设计案(环境侧,总设计师亲批)方可排队。

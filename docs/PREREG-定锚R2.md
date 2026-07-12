# PREREG-定锚R2:protocol-v3 开元 · 金种子定锚重测(终稿)

状态:**终稿**(批评者面板 wf_a2e6d05d-7c4 过审:三席、4 blocker / 13
major / 15 minor 全部落地)。冻结纪律循 R1/v30:**commit 即公证**,正文
零自指 sha,冻结 sha 仅入台账 FREEZE_SHA 事件。

## 案由与授权

R1 案结(v3 教师就位——系**序列先决**而非本案技术依赖:五发中唯 A5
加载 BC 产物,R1 与本案的实质联结为『组装考核顺延 R2』条款,见 A5/R-G1);
v2 绝对基线经外源审计故意作废;ROADMAP 序列第三步"定锚重测"即本案,
**判决附录落账后**课程战役解锁。

**金种子授权(知会事实链如实转录,面板修正)**:金种子 9000-9031 每案
一烧、手动点火、须预注册 + 总设计师知会(R1 条款承继)。事实链:①
ROADMAP 时代切换注记与 R1 判决附录已预告 R2 系金种子重测且须新预注册 +
知会;② 本席 2026-07-12 夜两次明告总设计师"R2 要烧金种子…您点头,R2
今晚就能开注册";③ 总设计师原话『把这两个PR趁今天抓紧报了吧 谢谢您
然后我们开始训练吧』系对②之直接应答。裁定:该应答构成**知情知会**
(informed nod)——本案零训练、系训练重启之序列先决,原话未逐字提及
金种子,但②已先行明告,推理链如实入册。面板曾提"冻结 commit 另经
总设计师过目后方点火"之更严方案;驾驶席按 v29/v30 先例(战役级知会 +
预注册冻结即点火,无逐案金种子二次签批)裁定不采,裁量与理由在此入册。
点火纪律:冻结 commit 落账后,值守手动启动驱动器一次即"手动点火",
五发系同一次点火内之预注册串行。

## 一战一处方

处方 = "四方定锚测量 + 一发闸门考核,金种子各烧一次,数字即锚"。
**定锚是测量,不是竞赛**:无胜负判词、不烧任何金牌资格、不改王座/现任
名分(头衔承继照 v30 D3 法庭口径逐字)。零训练、零环境侧改动、探针池
7000-7031 不触碰。

## D1 世界身份(先决,机器可执行)

- **W1 冻结公证**:porcelain 输出剔除唯一豁免行
  `?? train/leaderboard-assembled-v3.md`(金评 --board 之必然产物,
  protocol-v3 榜首建于本案;豁免行原文入台账,榜文件随判决附录 commit
  收录)后须为空;断言本文档与驱动器 train/run_reanchor_r2.py 的
  `git log -1 --format=%H` 均 == HEAD。**FREEZE_SHA 仅在全部 W 线断言
  通过后与 PREFLIGHT_OK 同批落账,预检失败不留 FREEZE_SHA**。台账允许
  多条 FREEZE_SHA(链式重冻结,P6):续跑断言 HEAD == 最后一条;新
  FREEZE_SHA 须携 {prev_sha, reason},reason 取自值守写入的
  `train/runs/reanchor-r2/REFREEZE_REASON` 文件(非空)。
- **W2** `PROTOCOL_VERSION==3 ∧ SCHEMA_VERSION==2`。
- **W3** `engine_binary_path` 唯一;`DEVILUTIONX_REF` 未设;bootstrap.sh
  逐字含 `ENGINE_REF="${DEVILUTIONX_REF:-34c4cfc2e733240ac717f23bba2def887c793008}"`。
- **W4** diabdat.mpq 在位(sha 入台账)。
- **W5 实现束取样**:`train_ppo._implementation_bundle_sha256()` 每发
  前后各取一次。**发车前取样 ≠ 案级首取值 → 不点火,全案停机呈报**
  (漂移在场,重烧无意义,不占 P2 额度;CASE_HALT_IMPL_DRIFT,退出码 5);
  收车后取样 ≠ 案级首取值 → 该发 OPERATIONAL(P2;-b 点火前仍须通过
  发车前取样,即漂移已复原方可重烧)。
- **W6** `runtime_versions_identity()` 完整入台账。
- **W7 工件钉死**(全文 sha 以驱动器冻结常量为准,预检失配即 **P4**
  不发车):
  - worker v28-leg1 npz = `976b6c05…f27f8a`;worker v24-leg7 npz =
    `a31fa7c6…fbc2a`;manager M29(**已按家底纪律归档**
    `train/models/v29-manager-mfresh/policy.npz`,与 runs 原件逐字节同,
    随冻结 commit 入库)= `89441388…7b66d6`;manager v22-H ==
    `DEFAULT_MANAGER_SHA256`;A5 之 v3 BC 教师
    `train/runs/bc-worker/policy_sd.pt` = `f052067a…ff98e6`(R1 回执)。
  - **前科档案钉死**(PRIORS,法证参照先封存后引用,v30 先例):
    v29-mfresh-full32.json / v28-G3-leg1.json / v24-golden.json 全文 sha
    驱动器常量断言,PRIOR_REFERENCE 事件入台账。
- **W8 机器空闲 + 驱动器互斥**:每发前 `pgrep -f
  'train/(bc_|train_ppo|eval_assembled|run_v[0-9]+|run_reanchor)'` 命中集
  剔除驱动器自身 pid 后判空;驱动器全程持
  `train/runs/reanchor-r2/.driver.lock` 排它锁(exclusive_lock),锁冲突
  → 不发车、不占额度、呈报(P3 型)。
- **W9 首燃先决**:台账无某发 FIRING_START 时,断言 eval-assembled 下
  不存在该发 `<tag>.json` 与 `<tag>-b.json`;已存在 → 停机呈报(严防
  案前手工档被静默采信或挤占首发 tag)。
- **W10 续跑对账**:台账已有 PREFLIGHT_OK 时,断言本次
  implementation/engine_binary/diabdat 三 sha 与首条 PREFLIGHT_OK 逐字
  相等(build/ 不受 porcelain 管辖,须以台账对账设防);不等 → 停机呈报。
- **W11 每发身份重申**:每发发车前 require 该发 worker/manager 工件 sha
  == W7 冻结常量(v30 逐腿 require_sha256 先例),并重申 W1 口径 git
  断言(覆盖 eval_assembled.py 等 W5 bundle 外协议文件之案中漂移)。

## D2 五发配方(每发额度 = 台账制 2 次点火;协议 = eval_assembled 逐字)

| 发 | tag | worker | manager(逐次显式 --manager-npz,承 v30 面板 blocker 纪律,禁用默认回落) | v2 前科(池别如实标注) |
|---|---|---|---|---|
| A1 科学锚 | `r2-science` | v28-worker-leg1 npz | v29-manager-mfresh npz | 140.3(**7000-7031 探针池**,sha16 08633101;金种子无前科) |
| A2 发射锚·现任 | `r2-launch` | v28-worker-leg1 npz | v22-h-manager npz | 112.4(**7000-7031 探针池**,sha16 6fc6a44c;金种子无前科) |
| A3 王座 | `r2-throne` | v24-worker-leg7 npz | v22-h-manager npz | **97.2(9000-9031 金池**,sha16 d9387dcb) |
| A4 脚本参照 | `r2-script` | `script` | v22-h-manager npz | **93.9(9000-9031 金池**,leaderboard-hier ppo-hier-v22-h 在册行,中位 103.45、死 2/32;v22 评测器读数、无档案 sha 可引,证据等级 = 在册旁证;7000 池对照 = H7 78.5 仅作参照不充前科) |
| A5 教师组装考核(闸门发,**非锚**) | `r2-bcworker` | `bc`(v3 教师 policy_sd.pt) | v22-h-manager npz | 无前科(v3 新造物) |

命令形:`.venv/bin/python train/eval_assembled.py --worker <spec>
--manager-npz <npz> --seeds 9000-9031 --board --tag <tag>`。串行,每发
wall-clock 上限 1h(超时 killpg+wait,击杀竞态 ProcessLookupError 不改
P2 定性);日志 `train/runs/reanchor-r2/logs/<tag>-{ts}.log`。
GOLDEN_AUTHORIZED 台账事件(五发命令原文 + 知会出处 + freeze_sha)于
预检通过后、首发前落账(v28 D2⑨/v29 先例)。

## D3 R 线

- **R-V(有效性,每发,机器可执行)**:发车时刻的输入冻结由评测器内部
  身份冻结、reserve_output 与出档自验承担;驱动器于**复验时**(续跑幂等
  在发车前、新发在收车后即时)自取 `freeze_eval_identity(ROOT,
  worker_spec, manager)` 快照作第三方复核,以 `expected_eval_identity(
  快照, tag, seeds=9000-9031)` 执行 `read_eval_archive(档案, **expected)`
  且 `verify_eval_identity(快照, ROOT)` 全过,**并断言快照之
  worker/manager sha == W7 冻结常量**(script/bc 之 worker sha 系协议
  bundle/闸报告派生,按快照自洽即可)→ 该发有效,agg.ret_mean 即锚值;
  任何一步失败 → OPERATIONAL(P2)。**科学读数一经有效落档即终局,
  禁止重烧**;台账已有 FIRING_VALID 而现快照 R-V 不过 → 判 runtime
  案中漂移,全案停机(CASE_HALT_RUNTIME_DRIFT,退出码 6),禁止转 -b。
- **R-O(弱序预测,登记不裁)**:预测 `r2-science ≥ r2-launch ≥
  r2-script`;王座与发射锚相对序不预测。序成立与否由驱动器计入
  R_O_OBSERVATION 台账事件,判决附录照抄。**破序不构成任何重测、加烧或
  复议理由;金种子不因 R-O 结果发生任何再暴露。**
- **R-Δ(跨世界位移,只记不裁)**:**唯 A3(97.2)与 A4(93.9)为金池
  同池跨世界差**;A1/A2 前科系 7000 探针池,其差为『跨世界 × 跨池』
  复合量,只入册、禁止以位移名义解读;归因候选(回跳禁闭 / 换层结算
  修复 / 引擎补丁 / 种子池更换)不可分割。判决附录对照表须分列
  「同池跨世界」与「跨池跨世界」两栏。驱动器落 R_DELTA 事件。
- **R-G1(承 R1 判决附录『组装考核顺延 R2』冻结义务)**:
  `r2-bcworker.agg.ret_mean ≥ 0.85 × r2-script 读数` → 登记"学习工人
  组装能力达标(金池口径)";< 0.85 → 如实入册,不停机、不授锚、不改
  名分。**该发读数永不称锚**。本条同时化解 R1"四方预告"与"半闸顺延"
  之内部张力,裁决在此入册。驱动器落 R_G1 事件。
- **锚之授予(案结判词)**:五发全部有效 → ANCHOR_GRANT 台账事件
  (全案至多一条,案结后驱动器幂等退出):`v3-science-anchor := 该发
  有效档案`(tag 为 r2-science 或其 P2 重烧 r2-science-b,以
  FIRING_VALID.tag 为准),`v3-launch-anchor` 同理;每锚对象须含
  {firing, tag, archive_path(仓库相对路径), archive_sha256,
  seeds, worker{kind,path,sha256}, manager{path,sha256}, ret_mean,
  v2_prior, freeze_sha},供后续战役驱动器照 read_comparable_reference
  先例消费(读前断言 sha,**禁止按 tag 猜路径**——残档与真锚可能同目录
  并存)。名分零变动;**王座在位锚值 := r2-throne 读数**(后续金牌 P 线
  对王座一律对表此值);**脚本参照值 := r2-script 读数**(后续资格线/
  比例线基线重铸之源);"入册备查"仅指不授 anchor 头衔,不免除对表义务。
  **课程战役解锁与 A 复赛权可动用之完成标志 = 本案判决附录落账之
  commit**(照 R1 冻结条款逐字;ANCHOR_GRANT 系机器先决与判决输入,
  不单独构成解锁;附录未落账期间任何课程战役不得发车)。

## D4 P 线(穷尽;退出码对照表见末)

- **P1 顶层异常/中断** → DRIVER_EXCEPTION + NEEDS_ATTENTION。续跑幂等
  采信**双条件**:该 tag 档案 R-V 全过 ∧ 台账中该 tag 有 FIRING_START
  且无 FIRING_INVALID/RESIDUE_SEALED 记录;缺台账记录的档案不得采信
  (W9)。
- **P2 发射额度(台账制)**:某发点火次数 = 台账该发 FIRING_START 计数
  (含未收车之发车,跨续跑累计);计数已达 2 而无有效档案 → 不再点火,
  CASE_HALT_OPERATIONAL 停机。磁盘残档在位与否不得作为额度依据。
  **可 -b 重烧(封闭枚举,每发限一次)**:(a) 子进程非零退出/崩溃且无
  档案;(b) 超时击杀;(c) 档案在而 R-V 失败且该发收车 W5 取样 ≠ 案级
  首取值(收车后漂移,-b 前须漂移已复原);(d) reserve_output 拒覆写
  残档。**必须停机、不得 -b**:发车前任何断言失败(W5 前漂移/W11 失配/
  git 不净/机器不空闲)、台账不可解析、R-V 失败而该发前后 W5 取样均等于
  案级首取值且档案结构自洽(成因不明,重烧零信息增量,呈报)。凡产生
  而未过 R-V 之档案,无论首跑续跑,一律 RESIDUE_SEALED{tag, sha, why}
  (去重),不删不改。
- **P3 机器不空闲 / 驱动器锁冲突** → 不发车、不占额度,呈报。
- **P4 预检 W 线不过** → PREFLIGHT_FAIL 事件,不发车,呈报(退出码 3,
  不走 DRIVER_EXCEPTION 通道)。
- **P5 停机取舍(甲案,面板两案择一之裁定)**:某发额度耗尽 → 全案
  停机,后续发不发车——失败优先解释为世界级异常,宁欠采不冒烧;已有效
  之发保留,循 P6 链式条款在续案承继(重验仍过即承继,禁止重烧——科学
  读数终局条款优先于冻结形式)。
- **P6 链式重冻结**:停机后如须修驱动器/文书,以勘误 + 新 commit 接续;
  台账允许多条 FREEZE_SHA(携 prev_sha + REFREEZE_REASON 文件之 reason),
  续跑断言 HEAD == 最后一条。

**退出码对照**:0 = 案结(ANCHOR_GRANT)/ 案已结幂等退出;2 = P2/P5
额度耗尽停机;3 = P4 预检不过;4 = P3 不空闲/锁冲突;5 = W5 发车前漂移
停机;6 = R-V 案中 runtime 漂移停机;其余非零/异常死亡 = P1。

## D5 台账、榜面与判决

台账 `train/runs/reanchor-r2/gate_ledger.jsonl`(JSONL 追加;续跑断言
逐行可解析)。每发入册:FIRING_START{firing, tag, impl_before}、
FIRING_EXIT{exit_code, log, impl_after}、FIRING_VALID(含 agg 全表、
archive_path/sha、worker/manager 身份、impl_after)或
FIRING_INVALID{why=异常原文首行}、RESIDUE_SEALED(去重,含 why)。
案级:PREFLIGHT_OK(首条为对账基准)、PRIOR_REFERENCE、
GOLDEN_AUTHORIZED、R_O_OBSERVATION、R_DELTA、R_G1、ANCHOR_GRANT(唯一)。

**榜面处置**:leaderboard-assembled-v3.md 每次成功出档各留一行(行键
tag@sha16 绑定档案,契约拒改写);被作废之首发行与残档行一律保留备查,
不删不改;科学判据以台账为唯一来源,榜面行数不作判词依据。判决附录
锚值对照表按 A1-A5 固定顺序呈现,**禁止按数值排序**;附录注明:本案
榜行系定锚测量之档案登记,非名次、非挑战、不改任何在位名分。

**档案入库**(v28 D2⑩ 止血先例):判决附录 commit 须以 `git add -f`
收录五发全部有效档案与被封存残档(train/runs/eval-assembled/r2-*.json)
及榜文件;ANCHOR_GRANT 所引 sha256 必须能在该 commit 内逐字复验,复验
不过 → 附录不得落账。

## 后续(不在本案执行)

判决附录落账 → 课程战役解锁、A 复赛权可动用。下一训练案候选(另案
预注册):课⑤ 干窗课程(训练侧)或 A 复赛权;课③④设计文书(环境侧)
另呈总设计师案头亲批。

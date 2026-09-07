# R17 方向合议庭裁决书(2026-09-02)

呈:主席。案卷:R16 判决书 r16-VERDICT-20260902.md(台账 VERDICT_R16 行);四份 R17 提案
(① 资源通道 / ② L2 后勤生存课 / ③ 战备表 v0.3 / ④ 战备教师经理),每份各附三份对抗性批评
(可行性 / 科学性 / 目标对齐,共 12 份);四份勘察(城镇经济、装备与战备、RESUPPLY 与经理路径、
L2 死亡法医)。本庭裁决规则:致命缺陷重于优点;结构性推进「打穿」重于打磨 L2;主席原则
「只有刷到数值以上才可以前往下一层」不可被定义式满足;不再无休止拧工资旋钮;工程与日历成本如实报。
本文件为新立档案,不改动任何既有文件;引用一律 file:line(仓库 /home/laure/AlphaDiablo/diablogym)。

## 一、案由

### 1. R16 结论(不再复述数字以外的部分)
史上首次四门全过(A/B/C/D 对新法锚);部署 16 种子:臂存活 5/16 vs 认证工人 10/16,L2 到达 10/16 vs 1/16,
clvl 3.06 vs 2.25;臂在 L2 阵亡 8/10(死时 AC 7,7,7,7,7,8,9,16;L2 花名册 114–162 只;死亡中位 3885 拍);
训练课堂 244 次下楼,战备托管 vested 0 / unready_denied 6,441。依预注册不加冕。R16 判决书将下一层根因命名为
「清空却未达标的死局」(cleared-but-unready):教练在楼层清空时不论战备即开 DIVE,而 AC 9 只能靠护甲掉落,
金币恒 100 无处可花。

### 2. 合议庭对案由的三处更正(12 份批评中 ≥9 份独立坐实)
**更正一:死局的真名是「榨干旗强制下楼」,不是「清空即下潜」。**
- 教练规则确为 `want = DIVE if (ready or cleared) else FARM`(python/diablogym/worker_env.py:1128-1130;
  部署探针镜像 train/runs/r10-staging/probe_r15_deployment.py:215-224)。
- 但部署中 L1 从未在 6000 拍内清空:三个 L1 存活种子终局仍剩 18/13/35 只怪(r16-deploy-arm.json 行 2114001/002/007),
  锚的露营者剩 9–45 只。`cleared`(击杀比 1.0,worker_env.py:169-180)在部署里几乎从不触发。
- 真正拍板下楼的是冻结掩码法:`forced_dive = _farm_handoff(...) or (self.exhausted and m[DIVE]); m[FARM] = not forced_dive`
  (python/diablogym/options_env.py:739-741)。`exhausted` 在 FARM 窗无进展满 KILL_PATIENCE=140 微拍**或**
  farm_scene_steps ≥ farm_scene_cap(3600)时置位(options_env.py:63, :1022-1027;`_mark_exhausted` :542-549),
  `reset_layer_clock_on_window` 只清 layer_clock 不清旗(:770-774)。
- 证据:R16 训练收窗原因 exhausted 284 ≈ descend 244(r16-arm-a-constitution/sentinel.jsonl 终行);
  1614 个 live DIVE 窗仅 244 次下楼(r13_dive_audit.jsonl 终行);部署 L2 死亡拍 3768/3869/3885 ≈ 场景预算 3600 + 路程。
- 推论:任何只改教练 want 的方案(③、④)在这条路径上至多推迟一个窗(≤140–600 拍),
  R15 判决书早已写明「教练的任何 want 都被掩码回退覆盖」(r15-VERDICT-20260901.md §二.1)。
  要让主席原则在清空/榨干楼层上**可满足**,必须同时具备:(a)一条非下潜的成长通道,使「达标」可达;
  (b)榨干逃生口对战备可见并记账(强制未达标下楼 = 不付托管、单独统计)。

**更正二:三处潜伏缺陷,任何方向都必须先修。**
1. 托管战备门用的是 v0.1 尺:`_descend_escrow_settlement` 在 worker_env.py:1866-1877 调 `readiness_power_ratio`
   (v0.1 表 :197-205,L2 需 clvl3/HP90/AC15/dmg8),而教练用 `readiness_power_ratio_v2`(:1128 → v0.2 表 :139-141,
   L2 = clvl2/AC9/dmg6)。vested 0 / denied 6,441 是结构必然;R16 判决书「244 次下楼无一达标」一句部分是尺子伪影
   (部署 10 次 L2 到达中 4 次 AC ≥ 9,按 v0.2 应当 vest)。
2. leashed_ppo 收据地雷:超时零记分支要求 `float(reward) == worker_wage`(train/leashed_ppo.py:4746),
   死亡等价分支要求 `== worker_wage + timeout_total`(:4785);而 worker_env 终局 policy_reward 另加
   `_escrow_vest + _hp_econ + 深度塑形`(worker_env.py:2113-2121, :2207-2214)。R16 每腿 78 次超时未触雷
   只因 vest 恒 0;托管一旦真的 vest,第一次与超时同窗即 RuntimeError 炸腿。
3. 仪表缺失:没有任何档案记录下楼拍、下楼时腰带/AC/HP、强制还是自愿(探针面板 probe_r15_deployment.py:102-114
   无 hp/belt/下楼拍)。「AC 在下楼时 vs 死亡时」「行程太长 vs a11 失速」全部只能推断。

**更正三:几条被四份提案共同引用的"证据"不成立。**
- 「所有死亡行 belt=0」是脑干反射的同义反复:`_drain` 在 2*hp<max_hp ∧ belt>0 时逐拍灌药
  (options_env.py:1144-1157;谓词 env.py:956-962),L2 单击最高 15 HP 不足以跳过反射,任何策略死时 belt 必为 0。
  锚同样 41/41、47/47(r16-anchor-xdevil-a / -xm29-a.json)。它不证明「工人不会用药」,
  只证明「药被喝光后才死」——即**供给**是约束(这一点反而支持 ①)。
- 金币算术:引擎 RndItemForMonsterLevel 掉落率 = P(GenerateRnd(100) ≤ 40) × P(第二次 > 25) ≈ 0.41 × 0.74 ≈ 30%/杀
  (engine Source/items.cpp:3257-3266),不是提案的 45%;100 杀 L1 清场 ≈ 285 金而非 420。
  且引擎 AutoPickup 只在一步走完时扫 8 邻格(Source/player.cpp:442;Source/qol/autopickup.cpp:93-110),
  原地 a9 砍死的怪脚下的金堆需要后续走位才会被捡。覆盖率**未量**。
- 「DEVIL 经理让存活多 29/128」(④ 的核心论据)是到达伪影:DEVIL×7e31dc54 在 3000 拍考卷里 L2 到达仅 4/128、0/128
  (eval-assembled/r16-arm-a-constitution-xdevil-{a,b}.json),M29 到达 45/128 且在 L2 死 27——差的是谁到了 L2,不是决策质量。
- 「α2 证明本策略族能学会生存」(②):α2 的 devil 卷深度直方 {1:128}/{1:127,2:1}
  (r13-arm-a2-shaping-xdevil-{a,b}.json)——α2 靠罢潜活下来,从未在 L2 生存过。

## 二、四条路:方案摘要与陪审团评分

### 方案摘要
| 路 | 一句话 | 改动面 | 自报成本 |
|---|---|---|---|
| ① 资源通道 | 自动拾金 + 上楼回城 + UI-free 买甲/买药/Pepin 免费治疗 + readiness-v4 教练(清空/榨干且未达标 → 回城一次,不下潜);RESUPPLY 窗口(Discrete(3) 不变)承载回城脚本 | 桥 ~280 行 C++(拾金/上楼/商店观测/买卖事务)+ env/options/worker/train/eval ~630 行 Python + 新探针/测试/驱动 | 95–110 h,3 周;首次 C++ 轮 |
| ② L2 后勤生存课 | 工人侧深起点课程(p=0.5 到 L2)+ 红区失血计价 ×3 + a12 喝药先验 + 托管尺 v0.2;零引擎/env/options 改动 | worker_env/leashed_ppo/train_ppo ~300 行 + 新探针/驱动 | 30–36 h,3–4 天 |
| ③ 战备表 v0.3 | 表 v0.3(L2/L3 AC 改 7,新增腰带列 belt≥2)+ cleared∧ready 才开 DIVE + 托管门换 v0.3 尺 + 腰带下限 RESUPPLY 教练规则 + 收据地雷修补 | worker_env/leashed_ppo/train_ppo ~200 行 + 新探针/测试 | 14–20 h,3–4 天 |
| ④ 战备教师经理 | 用 readiness-v3 脚本 BC 初始化 Discrete(3) 经理,再 MaskablePPO 微调(工人 7e31dc54 驱动 FARM+DIVE);训练侧目标 r' = R − 48·[自愿未达标下楼] + 存活深度时间积分 | train_ppo options 路径 ~240 行 + BC/探针/驱动 ~1000 行新文件 | 48–64 h,2 周;训练 12–20 h |

### 陪审团评分表(可行性 / 科学性 / 目标对齐;10 分制;裁定 = recommend_with_fixes / reject)
| 路 | 可行 | 科学 | 对齐 | 合计 | 裁定分布 | 致命缺陷(裁定 reject 或"按预注册即致命") | 可修缺陷要点 |
|---|---|---|---|---|---|---|---|
| ① 资源通道 | 6 | 5 | 5 | **16** | 3× recommend_with_fixes | 无 reject。科学性判三条"按预注册即致命但可修":(F1)主指标 alive/16@6000 可被行程截断曝光时间白得(回城 600 拍即可把 3/8 个 L2 死亡推出地平线);(F2)16 种子无功效(5/16 vs 11/16 才 p<0.05);(F3)行程触发含 `self.oe.exhausted` 即重开 R15 白嫖榨干旗漏洞并叠加 v0.2 托管 24 单位赏金 | 金币算术 ×1.5 高估、拾取覆盖未量;来回上下楼被 `_reward` 重复付 +40/层(env.py:4784-4801);工人观测含 gold/1000(env.py:5012,5110)→ T0 的 OOD 混杂;TOWN_CAP 1000 > TAU_CAP 600 窗法冲突;金币计入 positive_progress(options_env.py:912)改变 FARM 窗法;nav.walk_to 绕过 env.step 不计拍(nav.py:36-49);门槛数值与 analyze_arm_gates.py:53-55 不符却自称"verbatim";重建管线自 7 月 27 日后从未跑过;`_town_trip_legal` 未含 (cleared∨cap)∧¬ready,M29 锚从出生金 100 起就能回城 1000 拍;'T0 配置加冕'越过四门;Lazarus 论据已被 AutoTurnInBetrayerStaffForMonotonicTask(src/diablogym.cpp:2085-2098)废掉 |
| ② 生存课 | 6 | 3 | 5 | 14 | 1× reject(科学)+ 2× with_fixes(对齐一份判"作为方向致命") | 科学 F1 belt=0 同义反复、锚零主动喝药却更能活;F2 工资算术差 ~4×(实测失血 2.09 HP/杀 → −0.21/杀,×3 也只到收支平衡),红区罚只在 belt=0 状态累积 = 第二份死亡罚金,R14 已证价格轴无解;F3 托管在任何非死亡收窗即 vest(worker_env.py:1826-1848,含 'end'/'stall'/'cap'),最优策略是罢潜到 ~5800 拍再下楼领 24;F4 课程/非课程回合混算 kill-switch;F5 不碰死局。对齐:训练工人去"活过违反主席原则的下楼",且完全成功也只是 L2 打磨 | worker_env 属协议文件须六锚全重烤(eval_contract.py:55-63);a12 波段在主权下已由 options_env.py:1776-1800 强制(每窗一次、反射后禁)故 a12 先验前提错;"63% 步在 L1"混淆窗类型与楼层;12 窗课程可超 6000 拍;续训候选 zip 并无禁令(R-6 反向高估) |
| ③ 表 v0.3 | 7 | 5 | 3 | 15 | 1× reject(对齐)+ 2× with_fixes(科学判"对因果主张致命") | 对齐:**尺子倒挂**——L2/L3 AC 改 7 = 出生 AC,连续第三次下调(人类攻略表 L2 AC 15–25、L3 30–40,r15-READINESS-TABLE-draft.md:15-17),「达标」被定义式满足;无任何降低 h(2) 的机理(R14: h 靠技能不靠尺子);cleared∧ready 门在榨干旗路径上是空操作;再度推迟关键路径工程。科学:belt≥2 = 起始装备,clvl≥3 已被 206/220 到达者持有,机制不可能产出 0.80→0.55–0.65;榨干口被误述为"3600 才开";RESUPPLY-first 把控制交给不攻击的脚本且在 L2 带怪时也触发 | 三处代码事实与两处地雷(托管尺、收据)全部属实且最先由本案发现;规模自报 100–200 行实为 ~520–590 行;G0-A(2) 精确复现须显式 far_tiles=0;托管门在收窗评估而教练在开窗评估,腰带列会因反射灌药错位 |
| ④ 教师经理 | 6 | 3 | 3 | 12 | 2× reject(科学、对齐)+ 1× with_fixes | 科学 F1 机理与自引数据矛盾(DEVIL 从不离开 L1);F2 目标不计生存(−48 恰抵消 +48 下楼奖,存活项 ≤24/局,J 差 O(10–30) ≈ 0.2–0.4 σ,R9 塌缩前提未变);F3 G5(died ≤ DEVIL+5)与 G2(L2 到达 ≥0.8×)互斥,唯一同时满足者是"FARM 到场景上限被迫下楼" = const-FARM → 撞 G4。对齐:把主席"告诉经理一个数值"的硬规则改成可用 140 拍闲置绕过的软罚(经理级白嫖);init-source checkpoint 全 state_dict 精确匹配(train_ppo.py:7241-7255)不能种到 Discrete(4) TOWN 经理;不在呈主席候选名单;R9/R12 两次证伪 | "无需重烤"不实(train_ppo 属五文件重烤法,r13_ledger.jsonl:4);教练的 cleared 分支在 303 维观测中不可表(可见∧可达怪 vs 全花名册;本层击杀 /50 饱和);BC 接线比自述重 40–60 行;吞吐估计 8 envs 6–8 h 无实测 |

### 可移植的零件(裁决中已嫁接)
- 自 ③:托管尺 v0.1→v0.2 修正;leashed_ppo 收据修补;战备表加**腰带列**(但 AC 不下调);逐次下楼遥测;
  「前任工人在新形态零训练消融」作为决策点。
- 自 ④:G0-M1 脚本反事实经理(readiness-v3-strict / const-FARM / const-DIVE)量出自由边缘份额与强制下楼份额
  ——这是 R16 之后最值钱的一项缺失测量,且不需要动 train_ppo(~20 行探针扩展);行为同一性前检文化。
- 自 ②:工人侧深起点课程(移交 R18);探针逐拍 hp/belt/block 轨迹;红区计价与 a12 先验**不采纳**。

## 三、裁决

### 方向:① 资源通道(拾金 → 回城买甲买药 → 达标才下楼),作为 R17 主线;以 R17.0「尺子与仪表」为强制前奏;
### 次选:③(仅其零件);② 移交 R18;④ 关闭。

**理由(按本庭规则逐条):**
1. 致命缺陷重于优点:①是唯一没有 reject 裁定的方向;②③④各至少一份 reject,且 reject 的都是机理层
   (②工资算术与 vest 时机、③尺子倒挂与空操作门、④到达伪影与目标不计生存),不是措辞层。
   ①的三条"按预注册即致命"全是**度量与预注册写法**问题(曝光归一化、种子数、触发子句),用本裁决的修订即可消除。
2. 结构性推进「打穿」:只有 ① 建立成长通道。人类表 L3+ 需 AC 30–110 与抗性,v0.2 表也要求 AC 每层递增
   (worker_env.py:140),没有非彩票的装备/药品来源,同一死局会在 L3(h(3)=0.67)、L4… 逐层重演。
   ① 落地的桥原语(上楼转场、城镇导航、NPC 事务、商店观测)是 L5/9/13 城镇传送、传送门卷轴、深层补药必需的;
   本庭**不采信** Lazarus→Cain 论据(桥已绕过,src/diablogym.cpp:2085-2098)。
3. 主席原则:在清空/榨干楼层上,今天「达标」是 ~2%/杀的护甲彩票且 3–12 击即碎(审计 C9)。
   ① 让「刷到数值以上」在每个种子上**可达**而不是**定义式成立**(③)或**软罚可绕**(④)。
   榨干逃生口保留为逃生口,但改为**记账可见**:强制未达标下楼不付托管、单独统计、作为止损指标。
4. 反旋钮:① 是环境可供性(affordance)改动,不是价格;②③④ 的主体都是工资/尺子/教练常数。
   ① 中新增常数只保留三个(TOWN_CAP、TOWN_MIN_GOLD、腰带列阈值),各有引擎算术依据,不设网格。
5. 成本如实:① 是自 2026-07-27 二进制以来第一次动桥,重建管线(build/CMakeCache.txt 指向已消失的
   /tmp/alphadiablo-dev)从未跑过——**这才是日历风险**,不是行数。所以 R17.0 把"未改动桥重建逐位证明"
   排在任何 C++ 之前,并把零训练的 T0 因子探针作为发射训练臂的唯一判据,避免在坏通道上烧三周。

### 序列
**R17.0 尺子与仪表(2–3 天,零训练,零 C++ 改动;可与 ① 桥施工并行)**
- (a) 新档案 train/runs/r10-staging/probe_r17_deployment.py(克隆 probe_r15,后者字节不动):每次下楼记
  {beat, belt_heals, hp, AC, clvl, ratio_v2, forced = ¬mask[FARM], 触发原因 cleared/cap/idle-clock},
  死亡时 belt 与本层可见地面药,L2 停留拍,分层击杀;经理:readiness-v3(其 2114000-015 十六行须与
  r16-deploy-arm.json 逐位相等)、readiness-v3-strict、const-FARM、const-DIVE;工人:7e31dc54 与认证工人;
  48 种子 2114000-2114047 × 6000 拍(≈ 4.5 min/组合,按台账 87 s/16 种子)。产出 = R17 基线行 A0′ 与
  **强制下楼份额**(决定榨干口是否需要修宪)。
- (b) 修正案一(默认关;worker_env/leashed_ppo 属协议文件 → 旧法 4×128 + 六卷 r16-anchor 全部逐位重烤):
  `--worker-descend-escrow-readiness-table {v1,v2}`(worker_env.py:1872 换 `readiness_power_ratio_v2`);
  leashed_ppo 收据改为对照 `worker_wage + vest + hp_econ + shaping`(缺键 = 0,旧法字节等价)。
  G0 行:8192 步冒烟证明 vested > 0 且零 RuntimeError。
- (c) G0-0a:以**未改动**的 src/diablogym.cpp 从 ~/alphadiablo-dev/devilutionX(sha 34c4cfc2,-DDEVILUTIONX_SRC)
  重建,证明 4×128 旧法锚逐位——把工具链漂移与代码漂移分开。此步不过,① 不动一行 C++。

**R17.1 资源通道施工(第 1–3 周)——① 原案 + 本庭强制修订**
- 行程合法性写在 OptionsEnv `_town_trip_legal`,**经理无关**:dlvl == 1(R17 仅 L1 起程)∧ WM_DIABPREVLVL 触发点存在
  ∧ (cleared ∨ farm_scene_steps ≥ farm_scene_cap) ∧ ¬ready ∧ trips_this_floor == 0 ∧ gold ≥ 50。
  **绝不使用 `self.oe.exhausted`**(140 拍闲置门);触发原因逐次记账,idle-clock 触发 = 硬止损。
- TOWN_CAP = 600(≤ TAU_CAP),城镇窗规置于通用 cap 规之前;RESUPPLY 脚本在 a13 可执行时仍先拾药,该窗不计"空跑"。
- 城镇宏阶段 0 = **脚本扫金**(act_pickup_gold_at:CMD_GOTOAGETITEM → AutoGetItem → GoldAutoPlace,inv.cpp:1739-1758)
  扫清可见金堆,不依赖偶然走位;autoGoldPickup 同时开。
- 金币**不计** positive_progress(options_env.py:912 增加排除)——不改 FARM 窗法。
- `_reward` 在城镇窗内仅当 cur > `_econ_episode_max_depth` 才付 dl 项(来回净零),`_econ_steps_on_level` 冻结
  (env.py:4784-4801, :4923-4929);默认关逐位证明。
- SHOP/DOWN 阶段用**按拍记账的步行者**(a11 触发点步行者参数化 msg),不用 nav.walk_to(绕过 env.step/反射/保险丝)。
- 购买规划器:护甲要求耐久 ≥ 15(Cloak 40 金/耐久 18,Quilted 200/30;**永不买 Rags**),AC 目标按表;
  余额尽数买药至腰带上限;Pepin 免费满血。
- 战备表 **v0.3 = v0.2 + BELT 列**(L2 ≥ 4;依据:RestorePartialLife 均 ≈ 40 HP@86,一瓶 ≈ +19 击承受,
  四瓶 ≈ ×2.9;经济上 100 + ≥140 拾金 − 40 Cloak ≥ 4 瓶);**AC 保持 9**。readiness-v4 教练:
  ready_v03 → DIVE;(cleared ∨ cap) ∧ ¬ready ∧ 行程合法 → RESUPPLY(城镇);否则 FARM;行程用尽后榨干口照旧,
  但记为 forced-unready、不付托管。
- 工人观测 gold/1000(env.py:5012, :5110):T0 设"拾金开/回城关"对照行;若该行 |Δalive| > 2/48,
  训练臂加默认关旗把工人视图 gold 钳制为 0.1。
- 门槛按 analyze_arm_gates.py:53-55 原文(A: UCB95 < 0 ∧ died ≤ 110;B: LCB > −0.10;C: ≥ 0.95×),不再自称 verbatim 却改数。
- **T0 不加冕**:环境改动加冕须主席另立法;T0 只做机理证明与训练臂发射判据。
- "楼梯口虚拟商店"回退**只能由主席裁定**,不得由工程师替换。

**Gate T0 四行零训练因子探针**(R16 工人 7e31dc54,48 种子):(a)拾金开/回城关;(b)只买药;(c)只买甲;(d)全通道。
判据见 §四。

**R17.1 训练臂**(仅当 T0 过):续训 7e31dc54,readiness-v4,托管尺 v2,T = round2048(266240/(1−s));
四门对重铸 r17-anchor;部署对 A1 = T0(d) 与 A2 = 认证工人/R17 环境。

**R18**:② 的可用零件——工人侧深起点课程 + 撤退宏——在"有药有甲"的前提下开课(R17 遥测决定是后勤墙还是战术墙);
卖/修/传送门经济供 L3+;路线图修正案(R17 商店内核 → R18 卖修/传送门 + L2 生存课 → R19 L3–L5 课程)。
④ 仅在 TOWN/RESUPPLY 成为值得切换的选项后重开(Mark-I)。

## 四、建议的 R17 预注册骨架

### 1. 主指标(部署探针 r17 形态:readiness-v4 / 主权开 / v4 / hunt / cap 3600 / clock reset / 6000 拍 / 采样;≥48 种子,优先 128)
- **主-1 存活(曝光归一化)**:L2 每千拍风险率 = L2 死亡 / Σ L2 停留拍;以及「首次下楼拍 + 1800 时存活」
  (固定下楼后曝光)。臂/T0 对 A1、A2 比较;成对 McNemar(analyze_arm_gates.py:23-31 估计量)。
- **主-2 成长保持**:L2 到达 ≥ 0.8 × A0′;clvl 均值 ≥ A0′ − 0.1;L1 每千**地牢拍**(扣除城镇拍)击杀 ≥ 0.85 × A0′。
- 帕累托计数 alive ∧ L2。
- 机制子指标(信息量,禁用同义反复):行程触发原因分布(cleared / cap / idle-clock,后者须为 0);行程拍中位;
  购入 AC 与耐久;购入药数;空跑次数;下楼时 belt/AC;每次 L1 清场拾金;forced-unready 下楼份额(T0(d) ≤ 25%)。
- 加冕规则:仅经四门(对 r17-anchor)+ 部署「存活不劣于 A2 且成长显著优于 A2」(R16 措辞,锚定 A2 而非常数)。

### 2. Gate T0(零训练;发射训练臂的唯一判据)
(d) 对 A0′:成对 saved − lost ≥ +6/48 且 UCB95 < 0;L2 到达 ≥ 0.8 × A0′;L2 每千拍风险率 ≤ 0.7 × A0′;
(d) − (a) 存活 ≥ +4/48(通道效应而非观测漂移);(b) 与 (c) 各自报告,判决书须点名承重杠杆。
任一不满足 → 不发射训练臂;R17 判决 = 通道未坐实/不可归因。

### 3. 四门与部署门
- A/B 对 r17-anchor-devil-{a,b}(认证工人,R17 旗重铸):A UCB95 < 0 ∧ died ≤ 110;B LCB > −0.10。
- C 对 r17-anchor-m29-{a,b}:ret 与 kills ≥ 0.95×(文件规则);≥ 1.0 作信息报告。
- D:D1 FARM argmax 一致 < 0.99 对续训前 7e31dc54,**同时**在 R16 环境观测与 R17 环境观测上报告;D2 DIVE 对脚本 < 0.99。
- 部署门(臂 vs T0(d)):存活 ≥ T0(d) − 2/48 ∧ 风险率 ≤ T0(d) ∧ (clvl 或每千地牢拍击杀) ≥ 1.05×;否则不加冕且不"加冕 T0"。
- 训练腿门(r13_dive_audit 终行):descends/10 live DIVE 窗 ≥ 1.0;stall 份额 ≤ 25%;vested > 0;零收据 RuntimeError。

### 4. 锚法
- 默认关零漂移:旧法 r10-cand/r10-anchor 4×128 逐位(run_r13_rebake.py);六卷 r16-anchor-* 旗全关逐位;
  probe_r15_deployment 旗全关 16 行逐位等于 r16-deploy-arm/anchor.json(新 bundle 下)。
- G0-0a:未改动桥重建 → 4×128 逐位(任何 C++ 之前)。
- 重铸:r17-anchor-{devil,m29,m29full}-{a,b},认证工人,R17 旗,128 种子 × 3000 拍(新档案;拾金改变每条轨迹,
  且 `_town_trip_legal` 经理无关保证 M29 不会从出生就回城)。
- 部署锚:A0′(R16 工人/R16 环境,新 bundle 下 48 种子重探,前 16 行逐位等于 R16 行);A1 = T0(d);A2 = 认证工人/R17 环境。
- 冻结:实现 bundle + 桥 .so + 引擎 .so + 预注册 sha256 入 r13_ledger.jsonl;冻结后改配方一律修正案另立文件。

### 5. G0 验收
| 项 | 内容 | 通过线 |
|---|---|---|
| G0-0a | 未改动桥重建,4×128 旧法 | 逐位(硬止损) |
| G0-0 | 新 C++ 重建 + 套件(876 + 新) | 全绿 |
| G0-1 | 无头城镇往返冒烟 16 种子(L1 起程) | 0 崩溃/挂起;窗内守卫不抛、窗外必抛;返程存活 ≥ 15/16;行程中位 ≤ 400 拍、最大 ≤ 600;L1 花名册/地面物往返前后一致 |
| G0-2 | 同种子两跑(含商店库存/购买/金币) | 逐位(硬止损) |
| G0-3 | 经济 16 种子 | 拾金覆盖率(生成堆 vs 拾取堆)≥ 60% 且每次 L1 清场 ≥ 150 金;行程后 AC ≥ 9 且耐久 ≥ 15 者 ≥ 12/16;belt ≥ 4 者 ≥ 12/16;空跑 ≤ 2/16 |
| G0-4 | 反白嫖 | idle-clock 触发行程 = 0;每层 ≤ 1 次;readiness-v4 下 L1 每千地牢拍击杀 ≥ 0.85 × readiness-v3 同种子 |
| G0-5 | 定标(run_r17_g0.py,R16 配方 + 修正案一) | descends/10 窗 ≥ 1.0;stall ≤ 25%;vested > 0;零 RuntimeError;s → T |

### 6. 数值止损(预注册,触发即动作,不得事后改阈)
| 触发 | 动作 |
|---|---|
| G0-0a 不逐位 | 停;按旧 CMakeCache 钉编译器/旗标重试;仍不过 → 呈主席(桥 sha 法) |
| G0-1 崩溃/挂起或返程存活 < 15/16 | 停;回退方案须主席裁定 |
| G0-2 不逐位 | 硬止损 |
| G0-3 覆盖 < 60% 或拾金 < 150 | 药品子指标改为按实测收益条件化;仍 < 150 → L1 经济学证伪,如实报 |
| G0-4 idle-clock 行程 > 0 或击杀 < 0.85× | 去掉 cap 子句(仅 cleared)重跑;仍触 → 停 |
| T0 (d) − (a) < +4/48 | 观测漂移而非通道 → 不发射训练臂 |
| T0 L2 到达 < 0.8 × A0′ | 用下楼面板区分"行程太长"与"a11 失速",不调阈值 |
| 门 C 两卷均 < 0.95× | 反遗忘失败,不放宽 |
| 日历:第 2 周末未过 G0-0a…G0-2 | 呈主席复核;总期 > 4 周 → 冻结范围,剩余转 R18 |

## 五、工程与日历成本(如实)
| 项 | 估计 | 依据 |
|---|---|---|
| R17.0 探针(新文件) | ~150 行,4–6 h;运行 6 组合 × 4.5 min ≈ 30 min | 台账 87 s/16 种子 |
| R17.0 修正案一(托管尺 + 收据) | ~20 行 + 测试 4–6 h;重烤 4×128 ≈ 17 min + 六卷 ≈ 50 min | r13_ledger 02:31-02:48;02:22-03:11 |
| G0-0a 未改动桥重建 | 4–8 h(管线未验证;CMakeCache 指向已消失路径) | build/CMakeCache.txt:380;.so mtime 2026-07-27 |
| ① 桥 | 280–340 行 C++,32–40 h | 提案 ~280 + 拾金扫金宏 |
| ① env.py(城镇宏 + 按拍步行者 + 规划器 + 奖励来回净零) | 28–34 h | 提案 20–24 h + 批评 +80–150 行 |
| ① options_env/worker_env/train_ppo/eval 接线 | 16–18 h | 提案 |
| ① 测试 + G0 探针 | 16–20 h | 提案 14–16 + 覆盖率/奖励测试 |
| 预注册/批评面板/判决 | 12–14 h | R16 同款 |
| 缓冲(无头城镇中途加载未验证) | 8–12 h | 勘察缺口 |
| **合计** | **~125–150 h;日历 3–4 周**(R17.0 在第 1 周内完成;T0 在第 2 周末可达 = 最早 go/no-go) | |
| 算力 | < 12 h:探针 48 种子 ≈ 4.5 min/组合;六锚 ≈ 50 min;重烤 ≈ 1.5 h;训练腿 ≈ 1 h(R16 325,632 步 2,709 s);六卷 ≈ 45 min | 台账 |

对照:R16 从修宪批准到判决 ≈ 9 h,因为全在 Python;R17 是第一次动桥,约束在重建与无头城镇两处未知,不在行数。
② 的 3–4 天与 ③ 的 3–4 天便宜,但两者都不改变下一层要面对的同一死局。

## 六、风险与止损(补充 §四.6)
- **范围蔓延**:卖/修/传送门、L2 起程、经理学习一律 R18+;R17 只做 L1 起程一次往返。
- **契约漂移**:env/options/worker 改动旋转实现 bundle 与协议 bundle → 全锚重烤(已计入日历);冻结物新档案只增不改。
- **"T0 加冕"诱惑**:T0 是机理证明;加冕只走四门 + 部署门;若主席欲为环境改动另设加冕程序,须先立法。
- **回退方案空心化**:楼梯口虚拟商店会删除所有面向打穿的原语,只剩 L2 打磨——只能由主席裁定采用。
- **耐久侵蚀**:购入 Rags 会在 3–5 击碎裂;规划器耐久 ≥ 15 过滤写进基础配方而非应急项;止损:T0 中 L2 死者
  死时 AC 回落到 7 者 > 50% → 判决书标注 AC 杠杆失效,R18 评估修理事务。
- **曝光混杂**:行程消耗拍数会机械抬高 6000 拍存活——主指标已改为曝光归一化;任何"延长 TOWN_CAP"的应急项
  只有在归一化指标下才合法。
- **锚压缩**:M29 锚在拾金后 RESUPPLY 更常合法,门 C 可能向 1.0 压缩——按法执行,不事后放宽。

## 七、对主席的一句话
R17 走资源通道:先用两三天把尺子(托管 v0.2)、保险丝(收据地雷)和仪表(下楼遥测、强制份额)修好并量出基线,
再动第一刀 C++,让「只有刷到数值以上才下楼」在每个种子上有路可走——不再拧工资旋钮;
零训练的 T0 因子探针决定是否发射训练臂,通道坐实与否两周末即见分晓。

## 附:关键引用索引
- 教练规则 worker_env.py:1128-1130,回退顺序 :1141-1150;探针镜像 probe_r15_deployment.py:215-235。
- 掩码逃生口 options_env.py:739-741;榨干置位 :63, :542-549, :1022-1027;时钟清零不清旗 :770-774;positive_progress :902-912。
- 托管尺错位 worker_env.py:1866-1877(v0.1 表 :197-205)vs :1128(v0.2 表 :139-141);vest/forfeit :1826-1848。
- 收据地雷 leashed_ppo.py:4746, :4785 vs worker_env.py:2113-2121, :2207-2214。
- 反射灌药 options_env.py:1144-1157,env.py:956-962;a12 主权波段 options_env.py:1776-1800。
- 金币:obs src/diablogym.cpp:1258;autoGoldPickup=false :1825;地面物无价值字段 :1477-1499;工人视图 env.py:5012, :5110;
  引擎掉率 items.cpp:3257-3266;AutoPickup player.cpp:442 / autopickup.cpp:93-110。
- 城镇转场:SyncLoad WM_DIABPREVLVL src/diablogym.cpp:1132-1138;DisableLevelBacktracking :1878;patches/0005;
  env 守卫 env.py:2190-2205;StartNewLvl 先例 :3356-3361;商店 SetupTownStores stores.cpp:2162-2181;
  买卖内核 stores.cpp:2861-2864 / 2658-2671 / 2102-2120;HealPlayer :1018-1029。
- 来回下楼重复付酬 env.py:4784-4801, :4923-4929;a14 原子换装 src/diablogym.cpp:2795-2872。
- 门槛 analyze_arm_gates.py:53-55, :66-68;协议文件表 eval_contract.py:54-64;实现 bundle train_ppo.py:798-809。
- 部署行 r16-deploy-arm.json / r16-deploy-anchor.json;训练终行 sentinel.jsonl / r13_dive_audit.jsonl;
  台账 r13_ledger.jsonl(VERDICT_R16、探针计时 04:16-04:19、重烤 02:31-02:48)。

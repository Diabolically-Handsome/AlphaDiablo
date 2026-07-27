# 呈报件:R7 战役修订提案(rev20 → rev21)

状态:**草案待批**。本件系 F3 终审卷(docs/FORENSICS-F3-why-no-progress.md)裁定
之落实提案,未经总设计师亲批,**不构成任何发车授权**;未改动任何代码,
run_r7_combat_recovery.py(rev20)与 r7_statistics.py 原封未动。
2026-07-27,值守起草。

## 案由

F3 终审定谳:R6 全案唯一疑似真进步信号是**每步战斗效率**(步数口径 ~+6 wage/种子,
两池同号,合并 64 对 t=+2.26、bootstrap CI95=[+0.48,+6.22];但逐池符号检验不显著,
p=0.11~0.19,系待验假说非既证事实)。现行 R7 四条门指标(farm_worker_wage/
farm_worker_kills/ret/kills)全部是**总量指标**,每一条都混合了效率分量与存活时长
分量——而 F3 已证明存活时长分量是对称种子彩票(死亡翻转 6/14 有利,p=0.79,
micro_steps 差 +151/−152 两池镜像)。若 R7 终考再次出现"总量归零",现行记账
无法回答"效率残余是否真实存在"——那正是烧掉 256 对处女池后最该带回来的知识。

## 修订一(新增):每步效率预注册记分肢(只记不裁)

- **指标定义(钉死)**:逐种子 `rate(s) := farm_worker_wage(s) / micro_steps(s)`
  (schema-5 行字段直除,零新机械);配对差 Δrate(s) = rate_cand(s) − rate_base(s)。
- **落账**:development 决策与 final 分析各随卷落 RATE_REPORT(均值/中位/符号计数
  /精确二项 p/去杠杆均值,统计纪律照 B1 D3 逐字)。
- **性质**:**记分肢,只记不裁**——不进 METRIC_RULES、不占 familywise α、
  不作任何 pass/fail 输入。理由:①保持 rev20 已冻结的 familywise 保证原封;
  ②rate 单独可被"高效早死"策略 Goodhart(死亡非劣性门在侧,但记分肢独立成门
  需另案论证);③本肢的使命是**测量**效率假说,非裁决候选。
- **预注册参考带(供判读,非门)**:点 +0.004/微步(≈R6 两池实测 +0.0041/+0.0048),
  带 [0, +0.010];带外任一侧均系登记级发现。256 对下该肢 SEM 足以分辨 0 与点估
  (32 对下不能——F3 噪声地板卷)。
- **口径立法(强制随一切引用)**:效率/暴露分解必须声明**窗口口径还是步数口径**
  (同一净差下两口径相差 3.3 倍,F3 内洽审查裁定);本肢一律步数口径;
  窗口口径(wage/farm_fresh_n)只许作带窗长混杂注记的次级诊断。

## 修订二(确认,无需改动):seed-majority 条款已内建

F3 曾建议"保留 seed-majority 条款"。核验结果:r7_statistics.MetricRule 默认
`require_sign_test=True`,rev20 四条门指标全部已带精确符号检验——**此条已满足,
零改动**,在此如实入册以免后案重提。(附系数:256 对下符号检验需胜率 ≥0.578,
按 R6 两池去均值经验分布折算≈平移 +5.3 wage,对小效应确比均值门灵敏。)

## 修订三(新增,轻量):final 分析随卷落存活分解诊断

final 分析除 RATE_REPORT 外,随卷落一次 rate×time 对称分解(总配对差 = 效率分量
+ 时长分量,逐种子死亡翻转清单)。零判据、纯落账——若终考总量再归零,本诊断
即时回答"归零是效率消失还是彩票反面签",不必再开一次法证编队。

## 实现面(若批)

全部改动限 run_r7_combat_recovery.py 分析/报告段(derived metric 计算与落账)+
配套单测;环境/评测协议文件(python/diablogym/*、eval_assembled.py、
eval_contract.py)**零触碰**,eval 档案 schema 不动(rate 系行字段直除,不入档案)。
campaign_revision 20→21,依 rev20 之 CAMPAIGN_REVISION 常量与状态 schema 升版惯例。

## 与发车的关系

本件不改变 R7 发车前置(引擎可用 + 总设计师亲批)。发车环境选项与约束见
docs/OPS-windows-feasibility.md。


## 修正案二:BC-v1 demos-only 门(A2,总设计师 2026-07-27 批)

批文原文:「可以 就听您的 A2方案为主 不行就只能给模型升级了」(回退路线=模型升级)。

- **缘起**:R7 prepare-bc 两次失守——2_102 池被 a14 保险丝回执缺失崩溃烧毁
  (已修);2_104 池被候选质量门 FAIL 烧毁(top1 0.833 vs 门 0.95,稀有键
  recall 0.34/0.46 vs 0.85,类权重试无效)。零池成本天花板测量(burned-2_104
  demos,200ep+类权):top1≈0.88-0.90、稀有键≤0.43——门在 07-25 未过滤 v3
  视图下结构性不可达(Mac 时代 1.0 系过滤视图产物)。且 R7 训练命令不消费
  BC 策略(teacher=KING_SD,无 --teacher-sd/--bc-aux-demos),0.95 门保护的是
  R5/R6 时代遗物;R7 实际消费物 = demos(dry-anchor 锚 + 身份链)。
- **改动**:①bc_worker v1 候选/终评质量线降为只记不裁(字段原样落档);
  发布门 = demos-validity(池覆盖精确/局纪律/a14 覆盖/零标签冲突硬断言,
  新增 _require_zero_exact_label_conflicts);②train_ppo._validate_bc_report
  data_gate 分支的 0.95/0.85 两行降为 [0,1] 范围检查(身份链/marker/逐位
  复算一致性原封);③v1 登记段推进 2_106_000..127(2_104 已烧,append-only)。
- **池经济警示**:v1 偶数段仅剩 2_106/2_108 两发即撞 2_110_000 评测银行;
  本次失败则按批文转模型升级路线(方案C),不再烧池试错。

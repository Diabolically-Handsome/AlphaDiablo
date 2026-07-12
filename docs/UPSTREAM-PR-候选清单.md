# DevilutionX 上游 PR 候选清单(呈报稿,2026-07-12 补丁庭 wf_ea94119c-207)

**性质:仅呈报,不提交。**任何 PR 的提交/修改均须总设计师亲批后另行执行;
本次鉴定全程零 GitHub 写操作。上游 = diasurgical/devilutionX,本项目钉死
commit 34c4cfc2,七补丁逐一独立鉴定(各配一名审查官,均实取上游 master
原文 + 检索 issue/PR 为证)。

## 总裁定一览

| 补丁 | 分类 | 上游现状 | 建议 |
|---|---|---|---|
| 0001 城镇资源回退判空 | **可上游** | 未修,master 三处代码未动 | 新开单 PR(头号候选) |
| 0002 怪物源导弹动画判空 | **可上游** | **已在途:PR #8606,glebm 已 APPROVED** | 等合并,勿动 |
| 0003 SFX 长度判空 | **可上游** | **同上,#8606 同卷** | 等合并,勿动 |
| 0004 清空聊天史 | 纯本地 | 上游视跨局保留为特性("New Game" 分隔条目) | 留在本地补丁集 |
| 0005 回跳禁闭开关 | 纯本地 | 任务空间设计策略,非引擎缺陷 | 留在本地;远期走 Lua 钩子路线 |
| 0006 MPQ path() 访问器 | 需改造后可上游 | 未修;零树内调用者是硬伤 | 捆树内消费者再提(形态 B) |
| 0007 headless 跳过过场 | **可上游** | 未修;上游同款守卫惯用法现成 | 新开单 PR(建议改为 play_movie 层通用守卫) |

## 关键事实

1. **HeadlessMode 是上游一等特性,不是我们的私货**——Source/headless_mode
   .{hpp,cpp} 系上游代码,上游自己的 timedemo 测试与 CI 就在 headless 下
   跑。所以 0001/0002/0003/0007 修的都是**上游自身形态下可复现的真 bug**,
   上游化名正言顺;0004/0005 依赖的则是"单进程反复 reset 的嵌入形态",
   那才是我们的私货,故纯本地。
2. **渠道已经温热**:0002+0003 早于 2026-07-07 以 PR #8606("Fix two
   crashes when running in HeadlessMode",Diabolically-Handsome 账号)
   提交,核心维护者 glebm 已 APPROVED、CI 全绿、待第二位维护者合并;
   RFC #7974(rouming 的 AI/RL 集成提案)中维护者明确欢迎此类修复。
3. **0007 附带发现**:上游结局动画(diabvic/diabend.smk)同样无 headless
   守卫,且本地也未打对应补丁——智能体真杀掉 Diablo 会撞同类崩溃。故
   0007 的 PR 首选形态是 play_movie() 层加一行 `if (HeadlessMode) return;`
   (一行覆盖全部 10+ 调用点,含结局动画),顺手把我们自己通关之日的
   崩溃也提前修掉。**建议无论是否上游化,先在本地补一版通用守卫**
   (环境侧改动,届时按家法单独立案亲批)。
4. **0001 是头号新候选**:headless + spawn.mpq(上游 CI 同款数据形态)
   进城即崩,PR 正文可完全用上游自己的复现路径立论,照 #8606 模板写,
   预计零摩擦。0006 单发大概率被问"谁用它"而搁置,须捆一个树内消费者
   (如 packed 模式加载诊断报路径)再提。
5. **工艺提醒**:上游仓库 CRLF(.gitattributes `* -text`),本地补丁文件
   系 LF——提交时在上游 checkout 内直接改动生成 diff,勿直投本地补丁。

## 若您画圈,建议的提交次序

① 0001(单 PR,标题 "Fix town level loading in HeadlessMode when
Hellfire data is absent");② 0007(单 PR,play_movie 层通用守卫,标题
"Skip video playback in HeadlessMode");③ 0006(攒到有树内消费者再说);
0002/0003 只需盯 #8606 合并,合并后下次 bump 引擎基线时删本地补丁。

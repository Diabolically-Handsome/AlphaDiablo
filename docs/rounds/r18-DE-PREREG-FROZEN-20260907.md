# R18-D/E 预注册:仇恨上限 hold-v1 与择敌 threat-v1 的零训练探针(2026-09-07 凌晨)

## 〇、案由

R18-C 判决:撤退让循环转起来,但角色在 L2 上打不赢——瓶颈从"不会撤"移到"打不赢"。主席 01:05 令:
「怀疑模型一次招惹一整层,设仇恨上限 3–5 只,原地处理完再打别的;择敌,优先血少、威胁大的怪」。
证据:T0″ 撤退触发时身边 6 格内平均 7.5 只活怪;L2 离开时花名册中位 134 只。

## 一、接口(默认关 = 逐位不变)

- **R18-D `aggro_cap="hold-v1"`**(`python/diablogym/aggro_cap.py`,`env.controller_action_context`):在主线 L2+,当 a9 可执行
  (有可接敌目标)且 **6 格内可见活怪 ≥ 5** 时,屏蔽 a10(探索/全图求战 = 唯一的拉怪动作)。`mask[9]` 守卫保证永远留有可打的动作,
  不会死锁进 exhausted。计数 `_aggro_cap_fired`(掩码求值次数)。
- **R18-E `engagement_priority="threat-v1"`**(`python/diablogym/engagement.py`,`env._engage_candidate_for_action9`):在主线 L2+,
  a9 的目标从"wire 首个可接敌行"改为同一冻结候选集上的排序:远程 AI 优先 → 威胁权重(按运行时 `ai`,不按类型;唯一怪覆盖 ai)
  → 最大伤害 → **当前血量最低** → 距离 → id。观测、wire 顺序、奖励键不变;规则是快照的纯函数,奖励的接近项与宏一致。
  计数 `_engagement_decisions` / `_engagement_reordered`(与 canonical 选择不同的次数)。
- **v1 作用域 = 主线 L2+**:L1 前缀与撤退臂逐位相同,配对比较只隔离 L2 效应(冒烟已验:seed 2133001 首降 5181、L2 到达 8021 与撤退臂相同)。
  冒烟里曾把规则放到全楼层:L1 农场动力学被大幅改变(cap 在 L1 触发数百次,一局提前以服务终止收场),故收窄作用域。
- 阈值(5 只 / 6 格)与撤退法本夜不再改;改即新预注册。

## 二、探针

驱动 `r10-staging/run_r18de_probe.py`;池 2_133 同种子配对;工人 7e31dc54;时钟 completion-l2-r18c(9000 拍观察窗);
经理 = coach-v03;经济 sustain-loot-v1;撤退 retreat-v1 全开。四臂并行:

| 臂 | 配置 |
|---|---|
| ctl `r18de-retreat` | 撤退(= R18-C 撤退臂配置,在 R18-D/E 字节上重跑) |
| A `r18de-cap` | 撤退 + hold-v1 |
| B `r18de-threat` | 撤退 + threat-v1 |
| AB `r18de-cap-threat` | 撤退 + 两者 |

### 判据

- **回归(硬)**:ctl 行(剔除新增 None 键)与 R18-C 撤退臂行逐位相同;四臂到首次 L2 到达为止的 descents 前缀逐局相同。不同则 VOID。
- **主假设(只报不判,方向事先写死)**:H1 AB 与 A 的 L2 每千拍风险率低于 ctl;H2 AB 的配对存活 saved−lost > 0 且 UCB95 < 0;
  H3 撤退触发时身边活怪均值(monsters_near0)在 A/AB 低于 ctl(仇恨上限确实限制了拉怪);
  H4 L2 每千拍击杀数(打赢效率)在 B/AB 不低于 ctl;H5 最大深度分布(L3+ 到达数)。
- 机理报告:cap 触发次数、择敌改选份额、撤退次数/成功/途中死亡、上楼后 L2+ 占用、再下局数、幸存者钟响所在层。

本探针不是训练发射闸门。它决定 R18-B 训练臂用哪套接口(撤退 / 撤退+上限 / 撤退+择敌 / 全开)。

## 三、认证链(发射前)

全套件 0 失败(含 `tests/test_aggro_engagement.py`、`tests/test_resource_retreat_training.py`);探针回归 33023de1… 相等;
双向重烤 4/4 + 4/4(镜像根刷新);本文件 sha256 记入台账。

## 四、种子与文件

2_133 再消耗 4 组(累计 14 组);处女池零接触。新文件:本预注册、`run_r18de_probe.py`、`aggro_cap.py`、`engagement.py`、
测试文件、`r18de-probe/`。改动协议文件:`env.py`(两个 kwarg、三个惰性行字段、掩码规则、选择器包装)、`options_env.py`(工人窗也带撤退遥测)、
探针(两个键、遥测字段、版本 `r17-deployment-v3-r18de`)。改前副本 `~/r17_work/r18/*.pre-r18de`。

# R18-A 预注册:撤退接口 retreat-v1 与 Gate T0″(2026-09-06 深夜)

## 〇、案由与主席令

T0′(`r17-T0-PRIME-VERDICT-20260906.md`)坐实:资金链打通、达标份额 50% → 18% 之后,L2 风险率不降;
六条战备法不区分生死。主席诊断(22:20):**问题不出在工人,出在经理没清楚地告诉工人该怎么做**——
经理在 L2 上只有一个词(FARM),工人没有手(不能撤、不能择敌、没有喝药节奏)。
主席令(22:27):「接口不全,再做一个回城的接口,现在就开干」。

本预注册在施工完成、闸门通过之后、探针发射之前冻结。R22/R23 继续停放;不重标战备尺子;不发射任何训练臂。

## 一、接口(R18-A retreat-v1,默认关 = 逐位不变)

1. **引擎门**(`src/resource_protocol.hpp`):新增分支——主层 `source ≥ 2 → target = source − 1`、消息 WM_DIABPREVLVL,
   `accepted = retreatEnabled && retreatAuthorized`,回执 `retreat_ascent` / `retreat_not_authorized`;关时仍为旧默认
   `unauthorized_transition`。`configure_retreat(authorized)` 只能在主层 L2+ 授权;到达上层即消耗授权并 `retreats_started += 1`;
   每局至多 3 次;**每完成一次撤退,卖装备经济多换一次回城**(`MaxLootServiceTrips + retreats_started`,关时为旧上限 2)。
   观测新增 `retreat_enabled`(恒有)与 `retreat_authorized / retreats_started / max_retreats`(仅开时)。
   Python 侧身份校验失败即关闭(`validate_native_retreat`)。
2. **经理**:动作空间不变(Discrete(3))。开时,`action_masks()` 在 L2+ 上按撤退法开火并把掩码收成只剩 RESUPPLY
   (与回城之旅同一机制);任何窗在撤退法开火那一拍收窗(`retreat_trigger`)。
   **撤退法 v1**(HP 只用于喝药与回城决定,裁定 3):`hp ≤ 0.5·max_hp`,或 药空且 `hp ≤ 0.75·max_hp`;压力触发默认关。
3. **工人**:脚本 `RetreatService`(`python/diablogym/resource_retreat.py`):走向上楼梯(先避怪寻路,不通则不避),
   途中 `hp ≤ 0.4·max_hp` 且有药则喝(间隔 ≥ 20 拍),站上楼梯即等待引擎触发;到达上层 → `("complete",)` 交还控制;
   失败(楼梯缺失/不可达/被拒 8 次/900 拍上限/意外场景)也 `("complete",)` 交还并冷却 300 拍——**撤退失败从不终止本局**。
4. **上层之后**:原有教练与回城之旅照旧(HP < 80% 计入原生缺口 → 若还有回城额度则回城治疗/买药;达标即 DIVE 回 L2)。
5. `WorkerWindowEnv` 明确拒绝 retreat-v1(训练窗、托管与回执在探针之后再接)。

## 二、Gate T0″(零训练机理探针)

驱动 `r10-staging/run_t0_double_prime.py`;池 2_133(2133000–2133047,已消耗池,同种子配对);工人 7e31dc54;
时钟 completion-l2-v1(12000 实际拍到达 / 6000 观测分母 / 首降后 1800 观察);解码 sample;经理 = 资源教练 coach-v03。

| 组 | 配置 |
|---|---|
| (e) t0pp-loot-retreat | (d′) + `resource_retreat=retreat-v1` |
| (d′) t0pp-loot-noretreat | 与 T0′ t0p-full-loot 完全同配置,在 R18-A 字节上重跑 |

### 判据(发射前冻结)

1. 成对存活 (e) − (d′):saved − lost ≥ +6/48,且死亡率差单侧 UCB95 < 0;
2. (e) 的 L2 死亡数 ≤ 0.7 × (d′);
3. 机理(只报不判):撤退尝试中到达上层份额 ≥ 0.6,撤退途中死亡份额 ≤ 0.25;附 hp0 分布、触发类型、用时中位数、二次撤退局数;
4. 回归:(d′) 重跑行(剔除两个新增 None 键)与 T0′ t0p-full-loot 行逐位相同——**不同则整个探针作废(VOID)**,先查"关=逐位不变"。

裁决:1、2、4 全满足 → T0″_PASS(允许起草 R18-B:把撤退接入训练窗与回执);4 不满足 → VOID;其余 → FAIL。
FAIL 时按机理项分诊:若到达份额高而存活不升 → 杠杆在"撤了以后干什么"(上层恢复/再下)与触发时机;若到达份额低 →
杠杆在撤退执行(择路、被追打);两者都要写进判决书,但**不得在同一夜为此改阈值再跑**(阈值改动 = 新预注册)。

### 时间与观察窗的已知局限

首降后只观察 1800 拍,一次"撤 → 回城 → 再下"约 1000–2000 拍,故本探针主要检验"撤退是否把 L2 上的死改成活",
不检验"撤完再下能否推进"。后者需要新的时钟配方(R18-B 议题,不在本预注册内)。

## 三、认证链(发射前必须全部通过)

- 全套件 pytest(含新增 `tests/test_resource_retreat.py`)0 失败;
- 探针回归:`readiness-v3` 部署行 sha 与 33023de1… 相等(探针版本 `r17-deployment-v3-r18a-retreat`);
- 双向重烤(新桥 4/4、七月桥 4/4)在 R18-A 最终字节上 PASS;
- 本文件 sha256 记入台账;发射后不得再改协议源文件(改即作废在跑的重烤/对照/探针)。

## 四、种子与文件

2_133 再消耗 2 组(累计 8 组);处女池 2_116–119、2_126–128 零接触。新文件:本预注册、`run_t0_double_prime.py`、
`python/diablogym/resource_retreat.py`、`tests/test_resource_retreat.py`、`t0-double-prime/`。
修改的协议文件:`src/resource_protocol.hpp`、`src/diablogym.cpp`、`python/diablogym/{resource_protocol,env,options_env,worker_env,resource_sustain_loot}.py`、
探针 `probe_r17_deployment.py`(行键:仅 `resource` 下新增 `retreats_started`、`retreat`,关时为 None)。
改前副本:`~/r17_work/r18/*.pre-r18`。

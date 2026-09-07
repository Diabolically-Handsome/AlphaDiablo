# R18-F 预注册:回城卷轴 portal-v1 的零训练探针(2026-09-07 凌晨)

## 〇、案由

主席 01:05:「回城卷轴肯定是要做的」。R18-C 显示撤退循环能转但每次撤退要走回楼梯,25–28% 的撤退死在路上;
卷轴把"走到楼梯"换成"原地开门",把回城的代价压到最低,代价是 200 金一张、回来落在施法原地。

## 一、接口(默认关 = 逐位不变;Opus R18-F 工程师实装,主刀集成)

- 引擎:女巫 Adria 进入无头商店(`patches/0013-headless-witch-vendor.patch`,漂移门 PASS);桥:传送门的门分成两支
  (`portal_to_town` L2+→城、`portal_return` 城→L2+,均需 Python 授权;其余传送消息照旧拒绝);`configure_resource_protocol(..., portal=True)`;
  `configure_resource_portal`;`act_cast_town_portal`;每局至多 2 次;观测 `portal_enabled` 恒有,开时另有授权/次数/门状态/卷轴数。
  跨桥同一性:关时四种传送消息在城内与地牢的回执、整个 resource_state 与旧桥逐位相同(仅多一个 `portal_enabled` 键)。
- Python:`resource_portal="portal-v1"` 要求 l2-town-v1 + coach-v03 + retreat-v1;`PortalService`:回城时到 Adria 补卷轴
  (留足 4 格药带才买)、在 L2+ 撤退法开火且身上有卷轴时优先开门(否则走楼梯撤退)、进门、城内流程照旧、从城里的门回来;
  失败从不终止本局。三级优先链:门 → 撤退 → 回城服务。
- 已知未解:回来落在施法原地、面对幸存的怪群,血量不变;卷轴 200 金;放置失败照样耗卷。

## 二、探针

驱动 `r10-staging/run_r18f_probe.py`;池 2_133 同种子配对;工人 7e31dc54;时钟 completion-l2-r18c;经理 coach-v03;
经济 sustain-loot-v1;撤退 retreat-v1;两臂:ctl `r18f-retreat`(重跑)与 P `r18f-retreat-portal`(+ portal-v1)。

判据:回归(硬)——ctl 行与 R18-G 对照行逐位相同;两臂首次 L2 到达前缀逐局相同;否则 VOID。
主假设(只报不判,方向事先写死):H1 P 的 L2 每千拍风险率低于 ctl;H2 配对 saved−lost > 0 且 UCB95 < 0;
H3 P 的"撤退途中死亡"份额(含门与楼梯两种)低于 ctl 的 25%;H4 P 的再下局数与 L2+ 占用份额不低于 ctl(门保住了深度);
H5 卷轴经济:买卷数、门次数、失败原因分布、末期金币。

## 三、认证链

新桥(build-res 从同一源码重建)→ 全套件 0 失败(含 `tests/test_resource_portal.py`)→ 探针回归相等 → 双向重烤在新桥/最终字节上 4/4 + 4/4 →
本文件 sha256 记台账。

## 四、种子与文件

2_133 再消耗 2 组(累计 18 组);处女池零接触。新文件:本预注册、`run_r18f_probe.py`、`resource_portal.py`、`tests/test_resource_portal.py`、
`patches/0013-*.patch`、`r18f-probe/`。改动协议文件:`src/resource_protocol.hpp`、`src/diablogym.cpp`、`resource_protocol.py`、`env.py`、`options_env.py`、探针。

# R18-B 预注册修正案一(2026-09-07 23:4x)— 前缀护栏

冻结件 `r18-B-PREREG-FROZEN-20260907.md`(sha 7f948dd6…)不动;本文件为新文件。

## 事由
首发臂 `r18-arm-a-loot`(23:03 发射)于 23:25 崩溃:`RuntimeError: completion L2 arrival before learner handoff`。
根因:coach-v03 的六条法(不含血量)让经理在一层开 DIVE 窗,而"挣得的下楼"交接要求原生七条判定(含血量 ≥80%);
不满足交接条件的 DIVE 窗按前缀法由冻结父代整窗打完,父代按了下楼宏(或踏楼梯格)在前缀期内到达二层,
完成时钟按法 fail-closed。R18-B3b / M2 / B6 三次 4096 步冒烟均未命中该缝;首发跑到第 39 局命中。

## 修正(R18-B7,仅训练侧 `python/diablogym/worker_env.py`)
`WorkerWindowEnv._prefix_guard_masks`:在前缀期、主一层、未达交接条件的 DIVE 窗内,父代动作掩码去掉 a11 与踏 trigger 格的方向键
(与冻结的行走法同一函数 `_protected_walk_actions`),a0 保底;其余楼层/场景不变。交接定义("ready idle after normal drain")不变;
父代的观测不变;部署形态(OptionsEnv、探针、考卷)不经此路径。
测试 `tests/test_r18b7_prefix_guard.py`(5 例)+ earned-suffix/r18c/B3b/B6 邻域 240 passed;全套件与 4096 步冒烟另记台账。

## 身份
worker_env.py sha256 dafa6d5bbe92c986edfee9ccae8a3f28aa5f716b668eecb357853fbe22a41dde
implementation_bundle_sha256 a13ae0dff4847cb9cd0783e99991a4e800c9f6e7a1687d33232acfabc23d8662
其余身份(桥、引擎、补丁 0014、父代、探针)与冻结件相同。

## 重发
新运行名 `r18-arm-a-loot-2`,新候选目录,预算与冻结件相同(1 048 576 步 / 4 环境 / 存档 63 488 / 10 小时保险丝)。

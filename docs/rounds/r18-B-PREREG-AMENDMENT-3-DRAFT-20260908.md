# R18-B 预注册修正案三(草案,2026-09-08 午;**未冻结,待主席裁定**)— 考卷工具对 weights-only 热启动谱系的准入分支

冻结件 `r18-B-PREREG-FROZEN-20260907.md`(sha 7f948dd6…)与修正案一、二不动;本文件为新文件、草案。

## 一、事由

运行 `r18-arm-a-loot-4` 于 12:41 正常收尾(1 048 576 步,1052 局,发布 `model_candidate.zip` PRODUCTION_CANDIDATE,sha 20adb0df…)。
按冻结件 §三 跑 A–D 四闸时,**考卷工具 `train/eval_assembled.py` 拒绝加载该工人**,六卷全部 rc=1:

```
eval_contract.EvalContractError: asymmetric Worker checkpoint 尚未完成可部署 actor 训练,或不在完整 PPO 更新边界:
{'num_timesteps': 1048576, 'last_completed_rollout': 1048576, 'ppo_optimizer_steps': 27023,
 'warmup_start': None, 'warmup_until': None, 'warmup_expected_rollouts': 0, 'warmup_completed_rollouts': 0,
 'warmup_optimizer_steps': 0, 'actor_optimizer_steps': 27023}
```

同一拒绝对本谱系**任何**产物成立:运行 4 的中途存档(888 832 步)、B6 冒烟(4096 步)的正式发布件,均于 11:2x–11:3x 复核被拒。

## 二、根因

`eval_assembled._validate_asymmetric_worker_runtime_state` 的 `state_closed` 要求 critic 预热回执
(`warmup_start ≥ 0`、`warmup_until > warmup_start`、`warmup_expected_rollouts > 0`、`_critic_warmup_completed is True` 等)。
R18-B 的工人是 schema/2 **weights-only** 热启动(`r16-to-sustain-loot-v1-…-weights-only-v2`):继承父代 7e31dc54 的 actor + critic,
本世界没有 critic 预热期,这些字段按设计恒为 `None/0/False`(`LeashedMaskablePPO._assert_critic_migration_contract` 的继承谱系分支
正是这样要求的:"resource warm-start contains current-world warmup history" 即报错)。

训练侧的发布谓词(`train_ppo` `publication_eligible`)对继承谱系另有分支:继承回执有效 + actor 优化步 > 0;考卷没有对应分支。
R16 的臂经 critic 迁移(有预热期)所以通过;R18-B 是第一条 weights-only 谱系进考场,缺口今天才暴露。

## 三、提案(仅考卷工具;训练/桥/引擎不动)

在 `_validate_asymmetric_worker_runtime_state` 的 `state_closed` 判定之后、抛错之前,加继承谱系分支(与
`_assert_critic_migration_contract` 的继承分支和发布谓词同判据;考卷在 `require_published=False` 路径加载的是纯 `MaskablePPO`,故内联):

```python
    inherited_resource = getattr(model, "_resource_warm_start_receipt", None)
    if not state_closed and isinstance(inherited_resource, dict):
        warmup_tuple = (
            getattr(model, "_critic_warmup_start_timesteps", None),
            getattr(model, "_critic_warmup_until_timesteps", None),
            getattr(model, "_critic_warmup_expected_rollouts", None),
            getattr(model, "_critic_warmup_rollouts_completed", None),
            getattr(model, "_critic_warmup_optimizer_steps_completed", None),
            getattr(model, "_critic_warmup_completed", None),
            getattr(model, "_critic_warmup_actor_sha256", None))
        try:
            from migrate_resource_candidate import validate_inherited_runtime
            validate_inherited_runtime(model)
            receipt_valid = True
        except (AttributeError, RuntimeError, TypeError, ValueError, ImportError):
            receipt_valid = False
        state_closed = (
            receipt_valid
            and warmup_tuple == (None, None, 0, 0, 0, False, None)
            and _is_plain_int(fields["num_timesteps"]) and fields["num_timesteps"] > 0
            and fields["last_completed_rollout"] == fields["num_timesteps"]
            and _is_plain_int(fields["ppo_optimizer_steps"]) and fields["ppo_optimizer_steps"] > 0
            and _is_plain_int(fields["actor_optimizer_steps"]) and fields["actor_optimizer_steps"] > 0
            and fields["ppo_optimizer_steps"] == fields["actor_optimizer_steps"])
```

该分支只决定"能否加载",不触碰行数据的生成路径;非继承谱系(含 r9 认证工人、R16 臂)走原判定,逐位不变。

## 四、影响与重认证

`train/eval_assembled.py` 在 `PROTOCOL_SOURCE_FILES` 内(不在 `_IMPLEMENTATION_SOURCE_FILES` 内),协议 bundle sha 会变。
并入后须重跑:全套件;探针回归(16 种子 rows sha 33023de1…);六卷 `r17-anchor-*` 重铸(r9 工人走原判定,行须与既有六卷逐位相同);
双向重烤 4/4 + 4/4。训练合同、热启动回执、桥与引擎均不受影响(bundle 不含此文件)。

## 五、今日临时结果的地位

12:43 起在镜像根 `~/r17_work/r18/gates/exam-root`(`python/` + `train/*.py` 副本,`build` → 现役桥 build-res)对副本施加 §三 分支后
投射六卷(见 `~/r17_work/r18/gates/ABC-PROVISIONAL.json`、台账 `R18_B_GATES_ABC_PROVISIONAL`)。第一次尝试(12:43)分支写成调用
`model._assert_critic_migration_contract()`,在纯 `MaskablePPO` 上抛 AttributeError 被吞、六卷仍拒;12:5x 改为内联判据后通过。
主树未改任何字节;临时结果**不是裁定**,采信与否由主席决定。若并入 §三 并完成 §四 重认证,正式重铸六卷应与临时行逐位相同(可作为并入后的自检)。

## 六、待裁定

1. 是否采信临时 A–C 结果作为 R18-B 的四闸读数;
2. 是否把 §三 分支并入考卷工具(冻结本修正案 → 并入 → §四 重认证 → 正式重铸);
3. 另立事项(不属本修正案):earned-dive-suffix 工人的**部署形态**——学习者从未在一层训练、部署时却独自打一层(今日 E:一层死亡 25/48 对 7/48)。
   诊断用复合形态 v0(父代打到首次到达二层、学习者接手;`gates/probe_composite.py`,探针版本串加 `-composite-v0`)的结果见关门报告 §七;
   正式化需新预注册(交接规则应与训练的"七条件合格 DIVE 窗"一致,而非"首次到达二层")。

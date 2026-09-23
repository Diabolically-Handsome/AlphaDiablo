# R18-B 预注册修正案二(2026-09-08 02:4x)— 前缀审计压缩

冻结件与修正案一不动;本文件为新文件。

## 事由
重发臂 `r18-arm-a-loot-2`(2026-09-07 23:40 发射)于 02:19 崩溃:`OSError: [Errno 28] No space left on device`。
289 局、253 952 步(4 份存档:63488 / 126976 / 190464 / 253952)。根因:`PrefixAuditCallback` 在每个 rollout 首尾把每个环境的
**完整**前缀账本(全部尝试历史 + 每次 DIVE 开窗的完整状态快照)整体追加写入 `worker_prefix_audit.jsonl`,文件按二次方增长,
254k 步时已 6.2 GB,把只剩 6 GB 的盘写满;`最终状态写入失败` 也因此。训练数学与该文件无关(纯审计 IO)。

## 修正(R18-B8,仅训练侧 `train/train_ppo.py`)
`PrefixAuditCallback.compact_ledger`:rollout 首尾只写紧凑回执(计数 + 最近 2 次尝试、每次尝试最近 3 次开窗且去掉状态快照,
`ledger_form = compact-v1`);训练结束写一次完整账本(`ledger_form = complete`)。测试 `tests/test_r18b8_prefix_audit_compact.py`(3 例),
邻域 243 passed。审计文件已删除(保留台账记载);存档保留。

## 身份
train_ppo.py sha256 20d1d362481315f2a2b1601c69aad721185e365a97a977d46325e899b97c1b3d
implementation_bundle_sha256 36188fda8553f1b49db331659df0027f9be7fee7083b33712c0322c24e4bd5fc
其余身份与修正案一相同。

## 重发
先尝试从 `r18-arm-a-loot-2/ckpt/model_253952_steps.zip` 普通续训(运行名 `r18-arm-a-loot-3`,合同校验当场裁定);
若校验拒绝(实现指纹漂移或热启动谱系规则),改为重新热启动(`r18-arm-a-loot-4`)。磁盘:`alphaxiang` 890 GB 待主席处理。

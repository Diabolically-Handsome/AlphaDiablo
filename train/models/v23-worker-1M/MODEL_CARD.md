# v23-worker-1M — 第一个可学习操作脑(FARM 工人,1M 步峰值检查点)

判决主角,虽败犹史:金种子 **77.0**(中位 91.4,死 3/32),P1 三档未达标(0.82×
对 v22-H 脚本工人的 93.9)——**可替换性主张不成立**,判决书见 docs/DESIGN.md
v23 章与 docs/PREREG-v23.md。入库理由:它是本仓库第一个在脚本轨迹之外做决策
并局部赢过教师的操作脑(前 16 探针种子 105.7 对脚本 93.9,+12.6%,分歧率 29%;
自发使用教师从未按过的捡药/穿装键),也是"锚在报酬荒漠中漂移"(教训十八草案)
的物证——训练轨迹 500k=94.0 → **1M=105.7(峰)** → 1.5M=59.2 → 4M=42.6(塌缩停机)。

| 项 | 值 |
|---|---|
| 架构 | MaskablePPO MlpPolicy(64,64),γ=1.0,观测 298 维,Discrete(15) 掩 11/12 |
| 训练 | WorkerWindowEnv 在位(冻结 v22-H 经理),BC 热启动 + freeze 200k,ent 0.005,run `ppo-worker-v23`(2026-07-10 夜) |
| 世界 | v20 规则;工资 = 原始奖励 − 换层奖金(剥薪套利修复,全程换层率 0.0) |
| SHA-256 前缀 | `b6e1cbdd0137feca` |
| 随附 | sentinel.jsonl(500k 步粒度哨兵:干/鲜配比、动作份额、终止原因谱) |

## 复现

```bash
.venv/bin/python train/eval_assembled.py --worker train/models/v23-worker-1M/model --seeds 9000-9031
```

观测契约由 `python/diablogym/options_env.py` 的 `_worker_obs` 定义;经理必须是
`train/models/v22-h-manager`(numpy 前向,G0' 位级对账)。

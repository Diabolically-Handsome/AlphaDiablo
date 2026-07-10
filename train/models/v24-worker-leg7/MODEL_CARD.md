# v24-worker-leg7 — 皮筋工人(首个金牌超越脚本系统的可学习操作脑)

**金种子 97.2(中位 93.8,死 2/32)对 v22-H 脚本系统的 93.9 —— P1-强胜,
可替换性主张获证。** 同案双判决:P-拐杖-伪("仍拄拐":胜者带 β=0.0156 轻锚训成;
β=0 的腿 8 撒手即崩 55.6/79% 分歧)。金牌分歧率 0.66% → 判词"贴锚续航"。
全案:docs/PREREG-v24.md + docs/DESIGN.md v24 章 + 随附 gate_ledger.jsonl(八腿全录)。

| 项 | 值 |
|---|---|
| 架构 | LeashedMaskablePPO MlpPolicy(64,64),γ=1.0,观测 298 维,Discrete(15) 掩 11/12 |
| 训练 | 7 腿累计 ~7.01M 步在位(冻结 v22-H 经理),β 日程 0.5→0.015625(腿 6 软绊冻结),CE 皮筋对冻结 BC 教师 |
| 世界 | v20 规则;工资 = 原始奖励 − 换层奖金(全程换层率 ≤0.05%) |
| SHA-256 前缀 | `ac65d4eb91fdb678` |
| 总设计师处方 | "先给操作脑一个拐杖,再让它慢慢扔掉"(2026-07-10)——前半句获证,后半句被诚实证伪:退火终点不是零,是一羽之重(教训十九草案) |

## 复现

```bash
.venv/bin/python train/eval_assembled.py --worker train/models/v24-worker-leg7/model --seeds 9000-9031
```

经理必须是 `train/models/v22-h-manager`(numpy 前向);观测契约见
`python/diablogym/options_env.py::_worker_obs`;皮筋实现 `train/leashed_ppo.py`。

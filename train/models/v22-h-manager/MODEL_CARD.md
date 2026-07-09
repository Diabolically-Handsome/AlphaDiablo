# v22-H — 策略脑 v1(分层经理,首个入库模型)

第一个赢下预注册实弹对抗(P7)的模型,原样封存。

## 身份

| 项 | 值 |
|---|---|
| 架构 | MaskablePPO `MlpPolicy` (64,64),γ=1.0,gae_lambda=0.95 |
| 观测 | 303 维(295 基础 + 剩余时间/停滞钟/本层击杀/本层步数/上次选项 one-hot(3)/上次 τ) |
| 动作 | Discrete(3):FARM / DIVE / RESUPPLY(选项级,SMDP;内环为神谕逐字冻结宏) |
| 训练 | 3M 微步 @ M1 Max,`train_ppo.py --options`,run `ppo-hier-v22-h`(2026-07-09) |
| 引擎 | DevilutionX 钉死 `34c4cfc2e733`(bootstrap.sh ENGINE_REF)+ diablogym 桥(v20 世界规则:下楼阶梯 + 死亡阶梯 + 自动加点) |
| SHA-256 | `f3b579d2b0c9b613045692435a46702d1a9e8de8fc62e155c651f565d8bd6f1a` |

## 战绩(金种子 9000-9031,终评协议)

均回报 **93.9**(中位 103.45),死亡 **2/32**,对恶魔臂 F 成对胜 24/32。
教师脚本 101.5 但死 25/32——本模型用 7.5% 的回报换掉了 92% 的死亡。
完整对局记录见 `train/leaderboard-hier.md`,判决书见 `docs/DESIGN.md` v22 章。

## 复现 / 加载

```bash
.venv/bin/python train/evaluate_options.py train/models/v22-h-manager/model_final --options
```

```python
from sb3_contrib import MaskablePPO
model = MaskablePPO.load("train/models/v22-h-manager/model_final", device="cpu")
```

注意:观测契约(303 维布局)由 `python/diablogym/options_env.py` 定义,
改动该文件的 `_mgr_obs` 即作废此模型——只能新训,不能兼容。

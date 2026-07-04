"""DiabloGym v1 训练:PPO 学清地牢 1 层。

用法:
  ../.venv/bin/python train/train_ppo.py --total-steps 2000000 --num-envs 4
  (指标落盘到 runs/<run>/progress.jsonl + status.json,dashboard.py 实时读取)
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "python"))

from sb3_contrib import RecurrentPPO
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv


def make_env():
    from diablogym import DiabloGymEnv

    env = DiabloGymEnv(
        ticks_per_step=4,      # 每个决策 = 0.2 秒游戏时间
        max_steps=1500,        # v8:回到 run6 已验证配方,LSTM 是唯一新变量
        start_in_dungeon=True, # 跳过城镇,直接站在地牢 1 层入口
        include_raw=False,     # 训练不传 raw 大字典(多进程 IPC 减负)
    )
    return Monitor(env)


class EpisodeJsonlCallback(BaseCallback):
    """逐局把战绩写进 progress.jsonl;周期性刷新 status.json(供 dashboard 轮询)。"""

    def __init__(self, run_dir: pathlib.Path, config: dict):
        super().__init__()
        self.run_dir = run_dir
        self.config = config
        self.ep_count = 0
        self.t0 = time.time()
        self._progress = open(run_dir / "progress.jsonl", "a", buffering=1)
        self._last_status = 0.0

    def _on_step(self) -> bool:
        for info in self.locals["infos"]:
            ep = info.get("episode")
            if ep is None:
                continue
            self.ep_count += 1
            extra = info.get("episode_extra", {})
            line = {
                "ep": self.ep_count,
                "t": round(time.time() - self.t0, 1),
                "reward": round(float(ep["r"]), 3),
                "len": int(ep["l"]),
                **extra,
            }
            self._progress.write(json.dumps(line, ensure_ascii=False) + "\n")

        now = time.time()
        if now - self._last_status > 1.0:
            self._last_status = now
            elapsed = now - self.t0
            status = {
                "run": self.run_dir.name,
                "total_steps": int(self.num_timesteps),
                "target_steps": self.config["total_steps"],
                "episodes": self.ep_count,
                "sps": round(self.num_timesteps / max(1e-9, elapsed)),
                "elapsed_sec": round(elapsed),
                "updated_at": now,
                "config": self.config,
            }
            (self.run_dir / "status.json").write_text(json.dumps(status, ensure_ascii=False))
        return True

    def _on_training_end(self) -> None:
        self._progress.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--total-steps", type=int, default=2_000_000)
    ap.add_argument("--num-envs", type=int, default=4)
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--device", default="cpu", help="cpu / mps(小 MLP 通常 cpu 更快)")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--n-steps", type=int, default=512, help="每个 env 每轮采样步数")
    ap.add_argument("--algo", default="ppo", choices=["ppo", "rppo"],
                    help="rppo = RecurrentPPO/LSTM(B 计划:学习记忆替代手写宏状态机)")
    ap.add_argument("--arch", default="mlp", choices=["mlp", "attn"],
                    help="attn = 实体注意力感知(v9:AlphaStar 式 entity encoder + 地图 CNN)")
    args = ap.parse_args()

    run_name = args.run_name or time.strftime("ppo-l1-%m%d-%H%M%S")
    run_dir = pathlib.Path(__file__).resolve().parent / "runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "total_steps": args.total_steps,
        "num_envs": args.num_envs,
        "device": args.device,
        "lr": args.lr,
        "n_steps": args.n_steps,
        "algo": ("RecurrentPPO/MlpLstmPolicy" if args.algo == "rppo" else "PPO/MlpPolicy")
                + ("+EntityAttention" if args.arch == "attn" else ""),
        "goal": "地牢 1 层:杀怪拿 XP,找楼梯下 2 层",
    }
    print(f"== DiabloGym PPO 训练 == run={run_name}")
    print(f"   {config}")

    if args.num_envs == 1:
        vec_env = DummyVecEnv([make_env])
    else:
        vec_env = SubprocVecEnv([make_env] * args.num_envs, start_method="spawn")

    common = dict(
        learning_rate=args.lr,
        gamma=0.99,
        ent_coef=0.02,  # 首训 0.01 时策略塌缩成单方向面壁,提熵防锁死
        device=args.device,
        verbose=1,
        tensorboard_log=str(run_dir / "tb"),
    )
    policy_kwargs = {}
    if args.arch == "attn":
        from models import EntityAttentionExtractor
        policy_kwargs = dict(
            features_extractor_class=EntityAttentionExtractor,
            features_extractor_kwargs=dict(features_dim=256),
            net_arch=dict(pi=[128], vf=[128]),
        )
    if args.algo == "rppo":
        model = RecurrentPPO(
            "MlpLstmPolicy", vec_env,
            n_steps=256, batch_size=256,
            policy_kwargs=dict(lstm_hidden_size=128, n_lstm_layers=1, **policy_kwargs),
            **common,
        )
    else:
        model = PPO("MlpPolicy", vec_env, n_steps=args.n_steps, batch_size=256,
                    policy_kwargs=policy_kwargs or None, **common)

    callback = EpisodeJsonlCallback(run_dir, config)
    try:
        model.learn(total_timesteps=args.total_steps, callback=callback)
    finally:
        model.save(str(run_dir / "model_final"))
        vec_env.close()
        print(f"模型已保存: {run_dir}/model_final.zip")


if __name__ == "__main__":
    main()

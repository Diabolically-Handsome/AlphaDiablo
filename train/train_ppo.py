"""DiabloGym v1 训练:PPO 学清地牢 1 层。

用法(仓库根目录):
  .venv/bin/python train/train_ppo.py --total-steps 3000000 --num-envs 4
  (指标落盘到 runs/<run>/progress.jsonl + status.json,dashboard.py 实时读取)
"""

from __future__ import annotations

import argparse
import functools
import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "python"))

from sb3_contrib import RecurrentPPO
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv


def make_env(max_steps: int = 1500, deep: bool = False, death_ladder: bool = False,
             options: bool = False, flat_clock: bool = False):
    from diablogym import DiabloGymEnv

    if options:
        # v22:策略脑/操作脑——OptionsEnv 自带 deep+death_ladder 默认
        from diablogym import OptionsEnv
        return Monitor(OptionsEnv(max_steps=max_steps))
    if flat_clock:
        # v22 恶魔臂 F:296 维平面(停滞钟与策略脑同一块表)
        from diablogym import StagnationClockWrapper
        return Monitor(StagnationClockWrapper(DiabloGymEnv(
            ticks_per_step=4, max_steps=max_steps, start_in_dungeon=True,
            include_raw=False, descend_ladder=True, death_ladder=True)))
    env = DiabloGymEnv(
        ticks_per_step=4,      # 每个决策 = 0.2 秒游戏时间
        max_steps=max_steps,   # 1500 = 冠军(v6)配方;3000 = v10 长局实验 + v17 深水区。
                               # 32 种子排行榜评估固定 1500 步(可比性);深水区章另立新表
        start_in_dungeon=True, # 跳过城镇,直接站在地牢 1 层入口
        include_raw=False,     # 训练不传 raw 大字典(多进程 IPC 减负)
        descend_ladder=deep,   # v17:下楼奖金层数递进(8×N),给"往下活着"一个未来
        death_ladder=death_ladder,  # v18:死在 N 层罚 8×N——"活着抵达"要赢过"摸到深度"
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
    ap.add_argument("--algo", default="ppo", choices=["ppo", "rppo", "mppo"],
                    help="rppo = RecurrentPPO/LSTM(B 计划:学习记忆替代手写宏状态机);"
                         "mppo = MaskablePPO(v16:无效动作掩码,env.action_masks)")
    ap.add_argument("--arch", default="mlp", choices=["mlp", "attn"],
                    help="attn = 实体注意力感知(v9:AlphaStar 式 entity encoder + 地图 CNN)")
    ap.add_argument("--max-steps", type=int, default=1500,
                    help="episode 步数上限;1500 = 冠军(v6)配方,3000 = v10 长局实验")
    ap.add_argument("--seed", type=int, default=None,
                    help="训练种子(SB3 全局种子 + 环境 reset 种子;多进程采样时序仍会引入少量不确定性,只保证近似复现)")
    ap.add_argument("--deep", action="store_true",
                    help="v17 深水区:下楼奖金层数递进(N→N+1 付 8×N);配合 --max-steps 3000")
    ap.add_argument("--death-ladder", action="store_true",
                    help="v18:死亡成本随层数定价(死在 N 层罚 8×N,替代恒 -2)")
    ap.add_argument("--options", action="store_true",
                    help="v22:策略脑/操作脑(OptionsEnv,Discrete(3);须配 --algo mppo --gamma 1.0)")
    ap.add_argument("--flat-clock", action="store_true",
                    help="v22 恶魔臂:296 维平面(停滞钟入观测),配 --bc-init 用")
    ap.add_argument("--bc-init", default=None,
                    help="行为克隆热启动:载入策略头 state_dict 路径")
    ap.add_argument("--freeze-policy-steps", type=int, default=0,
                    help="BC 热启动后冻结策略头只训价值头的步数")
    ap.add_argument("--gamma", type=float, default=0.99,
                    help="折扣因子。0.99 半衰期 69 步(1500 步旧章口径);"
                         "v20 深水区用 0.997;--options(v22)应为 1.0")
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
        "max_steps": args.max_steps,
        "seed": args.seed,
        "algo": ({"rppo": "RecurrentPPO/MlpLstmPolicy",
                  "mppo": "MaskablePPO/MlpPolicy(gear-key mask)"}.get(args.algo, "PPO/MlpPolicy")
                 + ("+EntityAttention" if args.arch == "attn" else "")),
        "goal": ("深水区:层数递进奖金,活着往下潜(L3/L4)" if args.deep
                 else "地牢 1 层:杀怪拿 XP,找楼梯下 2 层"),
        "deep": args.deep,
        "death_ladder": args.death_ladder,
        "gamma": args.gamma,
        "options": args.options,      # v22:True 时 Monitor ep_len 口径=策略脑决策数
        "flat_clock": args.flat_clock,
        "bc_init": args.bc_init,
    }
    print(f"== DiabloGym PPO 训练 == run={run_name}")
    print(f"   {config}")

    env_fn = functools.partial(make_env, args.max_steps, args.deep, args.death_ladder,
                               args.options, args.flat_clock)
    if args.num_envs == 1:
        vec_env = DummyVecEnv([env_fn])
    else:
        vec_env = SubprocVecEnv([env_fn] * args.num_envs, start_method="spawn")

    common = dict(
        learning_rate=args.lr,
        gamma=args.gamma,
        ent_coef=0.02,  # 首训 0.01 时策略塌缩成单方向面壁,提熵防锁死
        device=args.device,
        verbose=1,
        tensorboard_log=str(run_dir / "tb"),
        seed=args.seed,
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
    elif args.algo == "mppo":
        # v16:掩码采样与掩码更新都由 MaskablePPO 处理;掩码本身来自
        # env.action_masks()(经 VecEnv.env_method 收集)。注意这是算法实现的
        # 整体更换,开牌异常时首要嫌疑人(诚实账本已记)。
        from sb3_contrib import MaskablePPO
        model = MaskablePPO("MlpPolicy", vec_env, n_steps=args.n_steps, batch_size=256,
                            policy_kwargs=policy_kwargs or None, **common)
    else:
        model = PPO("MlpPolicy", vec_env, n_steps=args.n_steps, batch_size=256,
                    policy_kwargs=policy_kwargs or None, **common)

    if args.bc_init:
        # v22 恶魔臂:BC 热启动策略头;冻结期只训价值头(经典雷:新价值头的
        # 首次 PPO 更新会摧毁 BC 策略,先冻结抗住)
        import torch

        sd = torch.load(args.bc_init, map_location="cpu")
        missing, unexpected = model.policy.load_state_dict(sd, strict=False)
        print(f"   BC 热启动:loaded(missing={len(missing)}, unexpected={len(unexpected)})")
        if args.freeze_policy_steps > 0:
            from stable_baselines3.common.callbacks import BaseCallback

            pi_params = (list(model.policy.mlp_extractor.policy_net.parameters())
                         + list(model.policy.action_net.parameters()))
            for p in pi_params:
                p.requires_grad = False

            class _Unfreeze(BaseCallback):
                def __init__(self, when):
                    super().__init__()
                    self.when, self.done_ = when, False

                def _on_step(self):
                    if not self.done_ and self.num_timesteps >= self.when:
                        for p in pi_params:
                            p.requires_grad = True
                        self.done_ = True
                        print(f"   策略头解冻 @ {self.num_timesteps}")
                    return True

            unfreeze_cb = _Unfreeze(args.freeze_policy_steps)
        else:
            unfreeze_cb = None
    else:
        unfreeze_cb = None

    # 每 ~50 万步存一次检查点(v9 在 40% 被终止时权重全丢的教训)
    ckpt = CheckpointCallback(
        save_freq=max(1, 500_000 // args.num_envs),
        save_path=str(run_dir / "ckpt"), name_prefix="model",
    )
    callback = EpisodeJsonlCallback(run_dir, config)
    try:
        cbs = [callback, ckpt] + ([unfreeze_cb] if unfreeze_cb else [])
        model.learn(total_timesteps=args.total_steps, callback=cbs)
    finally:
        model.save(str(run_dir / "model_final"))
        # 工作进程崩溃(如引擎段错误)后,SubprocVecEnv.close() 会在断管上永久阻塞,
        # 把"响亮的异常"变成"无声的挂死"(v11 连续三次的死状)。给 close 上闹钟:
        # 超时就放弃清理,让异常正常冒出、进程以非零码退出,监控才有尸体可验
        import signal

        def _close_timeout(*_):
            raise TimeoutError("vec_env.close() 超时(疑似 worker 已死)")

        signal.signal(signal.SIGALRM, _close_timeout)
        signal.alarm(20)
        try:
            vec_env.close()
        except Exception as e:
            print(f"vec_env.close 异常(忽略,不影响已保存的模型): {e}")
        finally:
            signal.alarm(0)
        print(f"模型已保存: {run_dir}/model_final.zip")


if __name__ == "__main__":
    main()

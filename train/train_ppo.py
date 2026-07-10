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
             options: bool = False, flat_clock: bool = False,
             worker: bool = False, manager_npz: str | None = None,
             worker_npz: str | None = None, skip_dry: bool = False):
    from diablogym import DiabloGymEnv

    if worker:
        # v23:FARM 操作脑在位训练——episode = 冻结 H 经理选中的一个 FARM 窗口
        # (rng_seed=None → 各子进程独立熵源,种子采样器拒采 7000/9000 段)
        # v26:skip_dry=True 时干层复访窗由脚本代跑,不进学习分布(绿洲处方)
        from diablogym import WorkerWindowEnv
        return Monitor(WorkerWindowEnv(manager_npz=manager_npz, max_steps=max_steps,
                                       skip_dry=skip_dry))
    if options:
        # v22:策略脑/操作脑——OptionsEnv 自带 deep+death_ladder 默认
        # v25:worker_npz 非空时挂 npz 工人(NumpyManager 在本函数体内构造——
        # spawn 子进程免 torch,PREREG-v25 D1 条款),并套种子纪律薄包装
        import gymnasium as _gym
        import numpy as _np

        from diablogym import NumpyManager, OptionsEnv
        from diablogym.worker_env import sample_train_seed

        if worker_npz:
            # 条款要点:工人以 npz+numpy 前向进子进程(不 pickle 网络、不 load SB3
            # 模型、不逐拍 torch 前向)。torch 模块本身随 train_ppo 顶层 import 进入
            # 子进程(v23 先例同),"无 torch"断言不可实现,预注册已如实修正。
            net = NumpyManager(worker_npz)
            env = OptionsEnv(max_steps=max_steps,
                             workers={0: lambda obs, mask: net.choose(obs, mask)})

            class _SeedDiscipline(_gym.Wrapper):
                """v25:reset(seed=None) → 拒采 7000/9000 段(逐局种子入 info)。"""

                def __init__(self, e):
                    super().__init__(e)
                    self._rng = _np.random.default_rng()

                def reset(self, *, seed=None, options=None):
                    if seed is None:
                        seed = sample_train_seed(self._rng)
                    obs, info = self.env.reset(seed=seed)
                    info["episode_seed"] = seed
                    return obs, info

                def action_masks(self):
                    return self.env.action_masks()

            return Monitor(_SeedDiscipline(env))
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


class WorkerSentinelCallback(BaseCallback):
    """v23 哨兵(PREREG 附录A/C):每 500k 步汇总子进程 WorkerWindowEnv.stats
    (干/鲜层窗配比、终止原因谱、兜底滚局数)+ 累计动作份额 → sentinel.jsonl。
    塌缩裁决本身走 2M/4M 检查点组装重放(附录C),此处只供遥测与验尸。"""

    def __init__(self, run_dir: pathlib.Path, every: int = 500_000):
        super().__init__()
        self.run_dir = run_dir
        self.every = every
        self.next_at = every
        self.action_counts = None

    def _on_training_start(self) -> None:
        # v24 修正:resume 腿的全局步不从 0 起——对齐到下一个 500k 边界,防空喷
        self.next_at = ((self.num_timesteps // self.every) + 1) * self.every

    def _on_step(self) -> bool:
        import numpy as np
        # v24 G-CAL:标定探针置旗即终止本腿(驱动裁决重标定,预注册条款)
        if getattr(self.model, "_calib_tripped", False):
            print("   [G-CAL] teacher_diverge>20% —— 终止本腿,交驱动裁决")
            return False
        acts = self.locals.get("actions")
        if acts is not None:
            if self.action_counts is None:
                self.action_counts = np.zeros(15, dtype=np.int64)
            for a in np.asarray(acts).ravel():
                self.action_counts[int(a)] += 1
        if self.num_timesteps >= self.next_at:
            self.next_at += self.every
            per_env = self.model.get_env().get_attr("stats")   # 经 Monitor.__getattr__ 透传
            agg = {"windows": 0, "dry": 0, "fresh": 0, "ff_windows": 0,
                   "ff_dry": 0, "episodes": 0, "reseeds": 0}    # ff_dry: v26 绿洲口径
            reasons = {}
            for s in per_env:
                for k in agg:
                    agg[k] += s.get(k, 0)
                for k, v in s.get("reasons", {}).items():
                    reasons[k] = reasons.get(k, 0) + v
            top1 = int(self.action_counts.argmax()) if self.action_counts is not None else -1
            share = (float(self.action_counts[top1] / max(1, self.action_counts.sum()))
                     if top1 >= 0 else 0.0)
            line = {"sentinel": "v23", "step": int(self.num_timesteps), **agg,
                    "dry_share": round(agg["dry"] / max(1, agg["dry"] + agg["fresh"]), 4),
                    "reasons": reasons, "top1_action": top1, "top1_share": round(share, 4),
                    # v24 皮筋读数(与 gate_ledger 双簿对账)
                    "beta": getattr(self.model, "distill_beta", None),
                    "distill_ce": getattr(self.model, "_last_distill_ce", None),
                    "teacher_diverge": getattr(self.model, "_last_diverge", None)}
            with open(self.run_dir / "sentinel.jsonl", "a") as f:
                f.write(json.dumps(line, ensure_ascii=False) + "\n")
            print(f"   [哨兵] {line}")
        return True


class DryAnchorSentinel(BaseCallback):
    """v26 干层锚哨兵(只记不裁,PREREG-v26 R26.6):demos.npz 中榨干旗=1 的教师态
    固定抽 2000,每 500k 步测学生 argmax 对教师标签的失配率——skip_dry 下干层行为
    无锚裸奔,这只表是它唯一的观察者。"""

    def __init__(self, run_dir: pathlib.Path, demos_npz: str, every: int = 500_000):
        super().__init__()
        import numpy as np
        self.run_dir = run_dir
        self.every = every
        self.next_at = every
        z = np.load(demos_npz)
        X, Y = z["X"], z["Y"]
        m = X[:, 297] == 1.0          # 观测第 298 维 = 榨干旗(工人观测契约)
        idx = np.random.default_rng(26).choice(
            np.flatnonzero(m), size=min(2000, int(m.sum())), replace=False)
        self.X, self.Y = X[idx], Y[idx]

    def _on_training_start(self) -> None:
        self.next_at = ((self.num_timesteps // self.every) + 1) * self.every

    def _on_step(self) -> bool:
        if self.num_timesteps >= self.next_at:
            self.next_at += self.every
            import numpy as np
            import torch as th
            with th.no_grad():
                obs = th.as_tensor(self.X, device=self.model.device)
                dist = self.model.policy.get_distribution(obs)
                pred = dist.distribution.logits.argmax(-1).cpu().numpy()
            mis = float((pred != self.Y).mean())
            line = {"sentinel": "dry-anchor", "step": int(self.num_timesteps),
                    "mismatch": round(mis, 4), "n": int(len(self.Y))}
            with open(self.run_dir / "sentinel.jsonl", "a") as f:
                f.write(json.dumps(line) + "\n")
            print(f"   [干层锚] {line}")
        return True


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
        self._steps0 = 0

    def _on_training_start(self) -> None:
        # v24 修正:sps 按本腿增量计(resume 腿否则虚高几十倍,降档闸门失明)
        self._steps0 = self.num_timesteps
        self.t0 = time.time()

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
                "sps": round((self.num_timesteps - self._steps0) / max(1e-9, elapsed)),
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
    ap.add_argument("--worker", action="store_true",
                    help="v23:FARM 操作脑在位训练(WorkerWindowEnv,Discrete(15) 掩 11/12;"
                         "须配 --algo mppo --gamma 1.0,见 docs/PREREG-v23.md)")
    ap.add_argument("--manager-npz",
                    default=str(pathlib.Path(__file__).resolve().parent
                                / "models" / "v22-h-manager" / "policy.npz"),
                    help="冻结经理权重 npz(export_manager_npz.py 产出)")
    ap.add_argument("--worker-npz", default=None,
                    help="v25:经理训练时挂 npz 工人(OptionsEnv workers 组装口)")
    ap.add_argument("--skip-dry", action="store_true",
                    help="v26 绿洲:干层复访窗脚本代跑,工人只在鲜层窗上课")
    ap.add_argument("--ent-coef", type=float, default=0.02,
                    help="熵系数(v22 恶魔臂微调用 0.005 防 BC 漂移)")
    ap.add_argument("--bc-init", default=None,
                    help="行为克隆热启动:载入策略头 state_dict 路径")
    ap.add_argument("--freeze-policy-steps", type=int, default=0,
                    help="BC 热启动后冻结策略头只训价值头的步数")
    ap.add_argument("--gamma", type=float, default=0.99,
                    help="折扣因子。0.99 半衰期 69 步(1500 步旧章口径);"
                         "v20 深水区用 0.997;--options(v22)应为 1.0")
    ap.add_argument("--distill-beta", type=float, default=0.0,
                    help="v24 皮筋系数 β(CE 对冻结 BC 教师;0=纯 v23 配方,G-KL-B 证逐位等价)")
    ap.add_argument("--teacher-sd",
                    default=str(pathlib.Path(__file__).resolve().parent
                                / "runs" / "bc-worker" / "policy_sd.pt"),
                    help="v24 教师 state_dict(SB3 键名)")
    ap.add_argument("--resume-from", default=None,
                    help="v24 分腿续训:上一腿 model_final.zip 路径(禁与 --bc-init/--freeze 同用)")
    ap.add_argument("--calib-probes", default="",
                    help="v24 G-CAL 探针全局步(逗号分隔,只在腿 1 传 300000,600000)")
    args = ap.parse_args()

    if args.resume_from:
        # PREREG-v24 D4:resume 禁 bc-init(覆写已训策略)与 freeze(语义只属腿 1)
        assert not args.bc_init and args.freeze_policy_steps == 0, (
            "PREREG-v24:--resume-from 禁与 --bc-init/--freeze-policy-steps 同用")

    if args.worker:
        # PREREG-v23 D3/契约6 护栏(审查团:默认值静默违约,--options 时代靠人肉记忆)
        assert args.algo == "mppo" and args.gamma == 1.0 and args.max_steps == 3000, (
            "PREREG-v23:--worker 须配 --algo mppo --gamma 1.0 --max-steps 3000")
    if args.options:
        # PREREG-v25:对称守护断言——终结"--options 靠人肉记忆"时代
        assert args.algo == "mppo" and args.gamma == 1.0 and args.max_steps == 3000, (
            "PREREG-v25:--options 须配 --algo mppo --gamma 1.0 --max-steps 3000")
        if args.worker_npz:
            assert args.n_steps == 64 and args.seed is not None, (
                "PREREG-v25 D2:换届选举须 --n-steps 64(v22-H 原配方)且显式 --seed")
    if (args.worker or args.options) and args.seed is not None:
        bad = set(range(7000, 7032)) | set(range(9000, 9032))
        assert not any(args.seed + r in bad for r in range(64)), (
            "种子纪律:--seed 邻域(+rank)撞探针/金种子段")

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
        "worker": args.worker,        # v23:True 时 ep 口径=FARM 窗口,reward=工资 w
        "bc_init": args.bc_init,
        "ent_coef": args.ent_coef,
        "freeze_policy_steps": args.freeze_policy_steps,
        "distill_beta": args.distill_beta,    # v24 皮筋
        "resume_from": args.resume_from,
        "worker_npz": args.worker_npz,        # v25 换届:经理训练挂 npz 工人
    }
    print(f"== DiabloGym PPO 训练 == run={run_name}")
    print(f"   {config}")

    env_fn = functools.partial(make_env, args.max_steps, args.deep, args.death_ladder,
                               args.options, args.flat_clock,
                               args.worker, args.manager_npz, args.worker_npz,
                               args.skip_dry)
    if args.num_envs == 1:
        vec_env = DummyVecEnv([env_fn])
    else:
        vec_env = SubprocVecEnv([env_fn] * args.num_envs, start_method="spawn")

    common = dict(
        learning_rate=args.lr,
        gamma=args.gamma,
        ent_coef=args.ent_coef,  # 默认 0.02(首训 0.01 曾面壁塌缩);v22 恶魔臂 0.005
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
        # v24:worker 路一律走 LeashedMaskablePPO(β=0 时 G-KL-B 证与原版逐位等价)
        calib = [int(x) for x in args.calib_probes.split(",") if x.strip()]
        if args.resume_from:
            from leashed_ppo import LeashedMaskablePPO
            model = LeashedMaskablePPO.load(args.resume_from, env=vec_env,
                                            device=args.device)
            # PREREG-v24 D4:β 显式覆盖(load 直写 __dict__ 无校验,不许静默续命);
            # tb 路径同理(否则腿 2-8 曲线全写进腿 1 目录);旋钮封条断言。
            assert hasattr(model, "distill_beta"), "resume 对象不是 LeashedMaskablePPO"
            model.distill_beta = args.distill_beta
            model.calib_probes, model.calib_out = calib, (
                str(run_dir / "calib.jsonl") if calib else None)
            model.tensorboard_log = str(run_dir / "tb")
            assert (model.ent_coef == args.ent_coef and model.gamma == args.gamma
                    and model.n_steps == args.n_steps), (
                "PREREG-v24 封-5:resume 腿超参与冻结配方不符")
            assert model.target_kl is None, "PREREG-v24 D4:target_kl 必须为 None"
            if args.distill_beta > 0:
                assert model.teacher is not None, "β>0 但教师未随 teacher_path 重建"
            if args.seed is not None:
                model.set_random_seed(args.seed)
            print(f"   [v24] resume @ {model.num_timesteps} 步,β={model.distill_beta}")
        elif args.worker:
            from leashed_ppo import LeashedMaskablePPO
            model = LeashedMaskablePPO(
                "MlpPolicy", vec_env, n_steps=args.n_steps, batch_size=256,
                policy_kwargs=policy_kwargs or None,
                distill_beta=args.distill_beta,
                teacher_path=args.teacher_sd if args.distill_beta > 0 else None,
                calib_probes=calib,
                calib_out=str(run_dir / "calib.jsonl") if calib else None,
                **common)
        else:
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
    sentinel_cb = WorkerSentinelCallback(run_dir) if args.worker else None
    dry_cb = (DryAnchorSentinel(run_dir, str(pathlib.Path(__file__).resolve().parent
                                             / "runs" / "bc-worker" / "demos.npz"))
              if (args.worker and args.skip_dry) else None)
    try:
        cbs = ([callback, ckpt] + ([unfreeze_cb] if unfreeze_cb else [])
               + ([sentinel_cb] if sentinel_cb else [])
               + ([dry_cb] if dry_cb else []))
        # v24:resume 腿 reset_num_timesteps=False(False 语义 = 再训 N 步,全局步连续
        # → ckpt 文件名全局唯一、β 日程与预算记账不断;审计 BLOCKER 2)
        model.learn(total_timesteps=args.total_steps, callback=cbs,
                    reset_num_timesteps=not args.resume_from)
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

"""DiabloGym v1 训练:PPO 学清地牢 1 层。

用法(仓库根目录):
  .venv/bin/python train/train_ppo.py --total-steps 2998272 --num-envs 4
  (指标落盘到 runs/<run>/progress.jsonl + status.json,dashboard.py 实时读取)
"""

from __future__ import annotations

import argparse
import fcntl
import functools
import hashlib
import io
import json
import math
import os
import pathlib
import shutil
import sys
import time
import zipfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "python"))

from eval_contract import RUNTIME_PACKAGE_VERSIONS
from sb3_contrib import RecurrentPPO
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv


_POLICY_HEAD_KEYS = (
    "mlp_extractor.policy_net.0.weight", "mlp_extractor.policy_net.0.bias",
    "mlp_extractor.policy_net.2.weight", "mlp_extractor.policy_net.2.bias",
    "action_net.weight", "action_net.bias",
)
_RUN_ARTIFACTS = (
    "progress.jsonl", "status.json", "status.tmp.json", "sentinel.jsonl", "calib.jsonl",
    # 发射夜审计 A 修:三件新仪表档系追加写("a" 模式),不入列则同名重跑
    # 残留堆积,课程腿第二发腿终全表复核必假判 CASE_HALT_G0(G0-2a 16:55:51 同因)。
    "dry_curriculum.jsonl", "distill_ce_probe.jsonl", "drywin_metrics.jsonl",
    "model_final.zip", "policy.npz", "policy_sd.pt", "tb", "ckpt",
)
_GEAR_PRESENT_INDEX = 293  # base obs zero-based; 文档中的“第 294 维”
# E1 连带改(:404 legacy 打印):契约修订号单一真源——值记录与打印同取此常量。
# E4(PREREG-内容案 圈 7,契约 4→5):+dry_curriculum/+bc_aux 两键,三腿统一
# rev5——L-base 双键均 "disabled"(同案零双版本);skip_dry 键仍 CLI 旗字面值
# (rev3 勘正,跨案取证锚禁被谓词覆写);旧检查点(rev4)非 legacy 续训将拒
# 系已知代价照录(R8/残余⑨)。
_CONTRACT_REVISION = 5
_RUNTIME_VERSIONS = dict(RUNTIME_PACKAGE_VERSIONS)
_ALGORITHM_RECIPE = {
    "gae_lambda": 0.95,
    "n_epochs": 10,
    "clip_range": 0.2,
    "clip_range_vf": None,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "normalize_advantage": True,
    "target_kl": None,
    "use_sde": False,
    "sde_sample_freq": -1,
}
_SCHEDULE_PROBES = (1.0, 0.5, 0.0)
_BC_REPORT_SCHEMA_VERSION = 1
_EXPORT_MANIFEST_SCHEMA_VERSION = 1
_WORKER_BC_DEMO_SEEDS = tuple(range(100, 228))
_WORKER_BC_FORBIDDEN_ACTIONS = (11, 12)
# E3 ④乙:禁采断言世代条件化(图纸 E2 共同真源)——v1 禁 (11,12) 原封(上行),
# v2 禁 11 允 12(守卫面不弱化)。
_WORKER_BC_V2_FORBIDDEN_ACTIONS = (11,)
# E3 ④乙:λ_bc 主案冻结常量(D7 注册裁量,与蒸馏锚 β=0.015625 同量级);
# 系 L-full CLI 显传值之文档/测试锚,非 --bc-aux-lambda 默认值(默认 0.0=不在位)。
_BC_AUX_MAIN_LAMBDA = 0.015625
# E6 探针集钉死(PREREG-内容案 E6):三腿仪表探针一律钉 BC-v1 demos【字节】——
# 双参考 0.7515/0.6305 跨腿跨案可比之担保物系 demos 数组同一性,由此字节断言
# 承载(np.savez_compressed 字节确定,承工程 B2 实测);加载处断言,伪字节必炸。
# BC-v2 仅经 --bc-aux-demos 进辅助损失,canonical bc-worker 路径不动。
_BC_V1_DEMOS_SHA256 = (
    "3bf892d611e41853eca8fce0cb146753af41ad2c3a21b6c581df1041fb1d9363")
# E5 探针专用 rng 种子(承 DryAnchorSentinel rng(26) 先例,孪生件同形;
# 只读探针自有流,不碰训练 RNG)。
_E5_PROBE_RNG_SEED = 26
# E5 探针示范态每组抽样上限(承 DryAnchorSentinel 固定抽 2000 先例)。
_E5_PROBE_GROUP_CAP = 2000
_BC_REPLAY_SEEDS = tuple(range(7000, 7032))
_BC_REPLAY_CACHE: dict[tuple[str, str, str, str], dict] = {}
_BC_PASS_KEYS = {
    "data_gate": {
        "schema_version", "pairs", "held_out_top1", "held_out_pairs",
        "held_out_episodes", "class_recalls", "class_weighted_retry",
        "data_gate", "protocol_version", "implementation_sha256",
        "generator_sha256", "manager_npz_sha256", "policy_sha256",
        "demos_sha256",
    },
    "hypothesis": {
        "schema_version", "pairs", "teacher_demo_mean", "bc_replay_7000",
        "teacher_7000", "ratio", "hypothesis", "protocol_version",
        "implementation_sha256", "generator_sha256", "policy_sha256",
    },
    "memoryless_hypothesis": {
        "schema_version", "pairs", "teacher_mean_demo",
        "bc_replay_mean_7000s", "teacher_replay_mean_7000s", "ratio",
        "memoryless_hypothesis", "protocol_version",
        "implementation_sha256", "generator_sha256", "policy_sha256",
    },
}


def _require(condition: bool, message: str) -> None:
    """训练契约不能用 assert：`python -O` 会把 assert 整段删掉。"""
    if not condition:
        raise ValueError(message)


def _is_plain_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_sha256(value) -> bool:
    return (isinstance(value, str) and len(value) == 64
            and all(character in "0123456789abcdef" for character in value))


def _finite_number(value, label: str) -> float:
    _require(isinstance(value, (int, float)) and not isinstance(value, bool),
             f"{label} 必须是数值")
    result = float(value)
    _require(math.isfinite(result), f"{label} 必须有限")
    return result


def _checkpoint_path(path: str | pathlib.Path) -> pathlib.Path:
    """按 SB3 规则容忍命令行省略 `.zip`。"""
    p = pathlib.Path(path)
    return p if p.exists() or p.suffix == ".zip" else pathlib.Path(f"{p}.zip")


def _capture_file_sha256(path: str | pathlib.Path, label: str) -> str:
    p = pathlib.Path(path)
    try:
        payload = p.read_bytes()
    except OSError as exc:
        raise ValueError(f"{label} 不可读: {p}: {exc}") from exc
    return hashlib.sha256(payload).hexdigest()


def _implementation_bundle_sha256() -> str:
    """Bind resume to code, native binaries and the actual game content."""
    import sysconfig

    from eval_contract import (content_identity, engine_binary_path,
                               runtime_versions_identity)

    root = pathlib.Path(__file__).resolve().parents[1]
    rel_paths = [
        pathlib.Path("train/train_ppo.py"),
        pathlib.Path("train/leashed_ppo.py"),
        pathlib.Path("train/models.py"),
        pathlib.Path("train/eval_contract.py"),
        pathlib.Path("python/diablogym/__init__.py"),
        pathlib.Path("python/diablogym/env.py"),
        pathlib.Path("python/diablogym/nav.py"),
        pathlib.Path("python/diablogym/options_env.py"),
        pathlib.Path("python/diablogym/worker_env.py"),
    ]
    suffix = sysconfig.get_config_var("EXT_SUFFIX")
    _require(bool(suffix), "当前 Python 没有 EXT_SUFFIX，无法绑定原生桥")
    rel_paths.append(pathlib.Path("build") / f"_diablogym{suffix}")

    digest = hashlib.sha256()
    for rel in rel_paths:
        p = root / rel
        try:
            payload = p.read_bytes()
        except OSError as exc:
            raise ValueError(f"实现绑定文件不可读: {p}: {exc}") from exc
        name = rel.as_posix().encode()
        digest.update(len(name).to_bytes(4, "big"))
        digest.update(name)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)

    # The bridge dynamically links the engine, so hashing only the extension
    # is insufficient. Locate the one actual engine binary fail-closed rather
    # than silently omitting it on another build layout.
    engine = engine_binary_path(root)
    try:
        engine_payload = engine.read_bytes()
    except OSError as exc:
        raise ValueError(f"实现绑定 engine 不可读: {engine}: {exc}") from exc
    engine_label = b"native-engine"
    digest.update(len(engine_label).to_bytes(4, "big"))
    digest.update(engine_label)
    digest.update(len(engine_payload).to_bytes(8, "big"))
    digest.update(engine_payload)

    # MPQ and Resources change world generation, collision, monsters and
    # rewards without changing source or binary bytes. Bind content rather
    # than host-specific absolute paths so an identical relocation is safe.
    content = content_identity(root)
    content_contract = {
        "game_data_sha256": content["game_data"]["sha256"],
        "assets_sha256": content["assets"]["sha256"],
        "assets_file_count": content["assets"]["file_count"],
    }
    encoded_content = json.dumps(
        content_contract, sort_keys=True, separators=(",", ":")).encode("ascii")
    content_label = b"runtime-content-v1"
    digest.update(len(content_label).to_bytes(4, "big"))
    digest.update(content_label)
    digest.update(len(encoded_content).to_bytes(8, "big"))
    digest.update(encoded_content)

    # BC generators call this helper too.  Binding only source/native/content
    # would let a policy trained under a different NumPy/Torch/SB3 numerical
    # stack present the same implementation identity later.
    encoded_versions = json.dumps(
        runtime_versions_identity(), sort_keys=True,
        separators=(",", ":")).encode("utf-8")
    versions_label = b"runtime-versions-v1"
    digest.update(len(versions_label).to_bytes(4, "big"))
    digest.update(versions_label)
    digest.update(len(encoded_versions).to_bytes(8, "big"))
    digest.update(encoded_versions)
    return digest.hexdigest()


def _validate_runtime_versions() -> None:
    from importlib.metadata import version

    actual = {name: version(name) for name in _RUNTIME_VERSIONS}
    mismatches = {name: (actual[name], expected)
                  for name, expected in _RUNTIME_VERSIONS.items()
                  if actual[name] != expected}
    _require(not mismatches,
             f"训练运行时版本漂移（升级须重做数值回归）: {mismatches}")


def _validate_model_recipe(model) -> None:
    """Reject a foreign/resumed PPO whose hidden defaults changed."""
    clip_samples = tuple(float(model.clip_range(progress))
                         for progress in _SCHEDULE_PROBES)
    clip_vf_samples = (None if model.clip_range_vf is None else
                       tuple(float(model.clip_range_vf(progress))
                             for progress in _SCHEDULE_PROBES))
    actual = {
        "gae_lambda": float(model.gae_lambda),
        "n_epochs": int(model.n_epochs),
        # A schedule can equal the registered constant at progress=1 while
        # silently annealing later.  Sample the beginning, midpoint and end.
        "clip_range": clip_samples,
        "clip_range_vf": clip_vf_samples,
        "vf_coef": float(model.vf_coef),
        "max_grad_norm": float(model.max_grad_norm),
        "normalize_advantage": bool(model.normalize_advantage),
        "target_kl": model.target_kl,
        "use_sde": bool(getattr(model, "use_sde", False)),
        "sde_sample_freq": int(getattr(model, "sde_sample_freq", -1)),
    }
    expected_recipe = dict(_ALGORITHM_RECIPE)
    expected_recipe["clip_range"] = (
        _ALGORITHM_RECIPE["clip_range"],) * len(_SCHEDULE_PROBES)
    if _ALGORITHM_RECIPE["clip_range_vf"] is not None:
        expected_recipe["clip_range_vf"] = (
            _ALGORITHM_RECIPE["clip_range_vf"],) * len(_SCHEDULE_PROBES)
    differences = {
        key: (actual[key], expected)
        for key, expected in expected_recipe.items()
        if (actual[key] != expected if not isinstance(expected, float)
            else not math.isclose(actual[key], expected, rel_tol=0, abs_tol=1e-12))
    }
    _require(not differences,
             f"PPO 隐含算法配方漂移（foreign/resume checkpoint）: {differences}")


def _check_finite_tree(value, label: str) -> None:
    import torch

    if isinstance(value, torch.Tensor):
        _require(torch.isfinite(value).all().item(), f"{label} 含 NaN/Inf")
    elif isinstance(value, dict):
        for key, child in value.items():
            _check_finite_tree(child, f"{label}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _check_finite_tree(child, f"{label}[{index}]")
    elif isinstance(value, float):
        _require(math.isfinite(value), f"{label} 含非有限标量")


def _validate_checkpoint_bytes(payload: bytes, label: str,
                               require_leashed: bool = False) -> dict:
    """Validate one immutable checkpoint byte string."""
    import torch

    try:
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            bad_member = archive.testzip()
            _require(bad_member is None, f"checkpoint CRC 失败: {bad_member}")
            member_names = archive.namelist()
            _require(len(member_names) == len(set(member_names)),
                     f"checkpoint 含重复 ZIP 成员: {label}")
            names = set(member_names)
            _require({"data", "policy.pth", "policy.optimizer.pth"} <= names,
                     f"checkpoint 缺关键成员: {label}")
            data = json.loads(archive.read("data"))
            saved_sb3 = archive.read("_stable_baselines3_version").decode().strip()
            _require(saved_sb3 == _RUNTIME_VERSIONS["stable-baselines3"],
                     f"checkpoint SB3 版本 {saved_sb3} 与运行时配方不符")
            for name in sorted(n for n in names if n.endswith(".pth")):
                state = torch.load(io.BytesIO(archive.read(name)), map_location="cpu",
                                   weights_only=True)
                _check_finite_tree(state, name)
    except (OSError, KeyError, ValueError, RuntimeError, zipfile.BadZipFile,
            json.JSONDecodeError) as exc:
        if isinstance(exc, ValueError) and str(exc).startswith("checkpoint"):
            raise
        raise ValueError(f"checkpoint 不可读/不安全: {label}: {exc}") from exc
    _require(isinstance(data, dict), f"checkpoint data 不是对象: {label}")
    try:
        steps = data["num_timesteps"]
    except KeyError as exc:
        raise ValueError("checkpoint num_timesteps 缺失/非法") from exc
    _require(_is_plain_int(steps) and steps >= 0,
             "checkpoint num_timesteps 必须是非负普通整数")
    if require_leashed:
        _require("distill_beta" in data,
                 "resume 检查点不是 LeashedMaskablePPO（缺 distill_beta 标记）")
    return data


def _validate_checkpoint_file(path: str | pathlib.Path,
                              require_leashed: bool = False) -> dict:
    """CRC + metadata + finite policy/Adam, read from the path exactly once."""
    p = _checkpoint_path(path)
    try:
        payload = p.read_bytes()
    except OSError as exc:
        raise ValueError(f"checkpoint 不可读/不安全: {p}: {exc}") from exc
    return _validate_checkpoint_bytes(payload, str(p), require_leashed)


def _validate_leashed_metadata(data: dict) -> dict:
    _require("distill_beta" in data,
             "resume 检查点不是 LeashedMaskablePPO（缺 distill_beta 标记）")
    try:
        beta = float(data["distill_beta"])
    except (TypeError, ValueError) as exc:
        raise ValueError("resume 检查点 distill_beta 标记非法") from exc
    _require(math.isfinite(beta) and beta >= 0,
             "resume 检查点 distill_beta 必须是有限非负数")
    return data


def _validate_leashed_checkpoint(path: str | pathlib.Path) -> dict:
    """用保存元数据区分 Leashed 检查点，禁止普通 MaskablePPO 冒充续训源。"""
    return _validate_leashed_metadata(
        _validate_checkpoint_file(path, require_leashed=True))


def _capture_leashed_checkpoint(path: str | pathlib.Path) -> tuple[bytes, dict, str]:
    """Capture, validate and hash the exact bytes later passed to SB3.load()."""
    p = _checkpoint_path(path)
    try:
        payload = p.read_bytes()
    except OSError as exc:
        raise ValueError(f"resume 检查点不可读: {p}: {exc}") from exc
    data = _validate_leashed_metadata(
        _validate_checkpoint_bytes(payload, str(p), require_leashed=True))
    return payload, data, hashlib.sha256(payload).hexdigest()


def _select_batch_size(n_steps: int, n_envs: int, cap: int = 256) -> int:
    """保持 256 配方；仅在尾 minibatch 恰为 1 时下调，避免 std=NaN。"""
    rollout_size = n_steps * n_envs
    _require(rollout_size > 1, "n_steps * num_envs 必须大于 1")
    for size in range(cap, 1, -1):
        if rollout_size <= size or rollout_size % size != 1:
            return size
    return rollout_size


def _training_contract(args, model, batch_size: int,
                       manager_npz_sha256: str | None = None,
                       worker_npz_sha256: str | None = None,
                       demos_sha256: str | None = None,
                       implementation_sha256: str | None = None,
                       bc_aux_demos_sha256: str | None = None) -> dict:
    mode = "worker" if args.worker else "options" if args.options else \
        "flat_clock" if args.flat_clock else "flat"
    action_count = getattr(model.action_space, "n", None)
    return {
        "schema_version": 2,
        "contract_revision": _CONTRACT_REVISION,   # v32:+drink_sovereignty(④丙 环境语义入契约)
        "implementation_sha256": implementation_sha256,
        "mode": mode,
        "arch": args.arch,
        "max_steps": args.max_steps,
        "num_envs": args.num_envs,
        "n_steps": args.n_steps,
        "batch_size": batch_size,
        "gamma": args.gamma,
        "learning_rate": args.lr,
        "ent_coef": args.ent_coef,
        "device": str(model.device),
        "skip_dry": bool(args.skip_dry),
        "drink_sovereignty": not args.no_drink_sovereignty,
        # E4 rev5 双键(圈 7,三腿统一):disabled 或实况载荷;skip_dry 键
        # 保持 CLI 旗字面值不受此二键影响(rev3 勘正,契约与回执同构)。
        "dry_curriculum": _contract_dry_curriculum(args),
        "bc_aux": _contract_bc_aux(args, bc_aux_demos_sha256),
        "manager_npz_sha256": manager_npz_sha256,
        "worker_npz_sha256": worker_npz_sha256,
        "demos_sha256": demos_sha256,
        # Gymnasium exposes Discrete.n (and on some versions shape entries) as
        # NumPy integer scalars.  Normalize before this contract is embedded in
        # status.json/SB3 data; stdlib json intentionally cannot encode np.int64.
        "observation_shape": [int(value) for value in model.observation_space.shape],
        "action_n": None if action_count is None else int(action_count),
        "runtime_versions": dict(_RUNTIME_VERSIONS),
        "algorithm_recipe": dict(_ALGORITHM_RECIPE),
    }


def _validate_resume_contract(saved: dict | None, current: dict,
                              allow_manager_change: bool = False,
                              allow_legacy_resume: bool = False) -> None:
    if saved is None:
        _require(allow_legacy_resume,
                 "resume checkpoint 无 training_contract，无法证明原训练环境/资源；"
                 "如确需一次性迁移，请显式传 --allow-legacy-resume")
        print("   [legacy migration] 已显式允许无契约 checkpoint；"
              f"本腿将写入 contract_revision {_CONTRACT_REVISION} 契约")
        return
    _require(isinstance(saved, dict), "checkpoint training_contract 不是对象")
    allowed = {"manager_npz_sha256"} if allow_manager_change else set()
    differences = {key: (saved.get(key), current.get(key))
                   for key in sorted(set(saved) | set(current))
                   if key not in allowed and saved.get(key) != current.get(key)}
    _require(not differences, f"resume 训练/环境契约漂移: {differences}")


# ---- E1 ⑤A 干窗课程(PREREG-内容案 E1,rev3 核认勘正) ----

# 腿相对锚定常量:腿起点恒 = 王 zip 终点 3,497,984(P3 复点火恒自王 zip);
# 全局步锚定禁用——p 表序号 = (num_timesteps − 3,497,984) / 2048。
_DRY_CURRICULUM_LEG_START = 3_497_984
# 主表(批文即定,圈 2 附裁):前 147×2048=301,056 步线性 1.0→0.5,
# 后 97×2048=198,656 步持 0.5;147+97=244 量子恰等腿长 499,712。
_DRY_CURRICULUM_MAIN_TABLE = "linear:1.0:0.5:147,hold:0.5:97"


def _dry_window_mechanism_active(args) -> bool:
    """E1 波及面谓词(rev3 勘正,四门统一):干窗机制在位 = skip_dry ∨ schedule。"""
    return bool(args.skip_dry) or bool(args.dry_curriculum_schedule)


def _mount_dry_anchor_sentinel(args) -> bool:
    """E1 四门之 dry_cb 挂载门(原 :2101-2104 谓词):worker ∧ 干窗机制在位。"""
    return bool(args.worker) and _dry_window_mechanism_active(args)


def _precheck_dry_window_demos(args) -> None:
    """E1 四门之 demos/BC 预检门(原 :499-505):谓词改写为机制在位,断言原封。"""
    if not _dry_window_mechanism_active(args):
        return
    demos = pathlib.Path(__file__).resolve().parent / "runs" / "bc-worker" / "demos.npz"
    _require(demos.is_file(),
             f"干窗机制(--skip-dry/--dry-curriculum-schedule)所需示范集不存在: {demos}")
    # E6 探针集钉死:canonical BC-v1 demos 字节 ≡ 冻结常量(加载处断言)。
    _assert_bc_v1_demos_frozen(demos)
    policy = demos.with_name("policy_sd.pt")
    _require(policy.is_file(),
             f"干窗机制(--skip-dry/--dry-curriculum-schedule)所需 BC 权重不存在: {policy}")
    report = _validate_bc_report(policy, "data_gate")
    _load_dry_anchor_demos(demos, report.get("demos_sha256"))


def _capture_dry_window_demos_sha256(args) -> str | None:
    """E1 四门之 demos_sha256 捕获门(原 :1766-1771):谓词改写,路径与断言原封。"""
    if not _dry_window_mechanism_active(args):
        return None
    demos = pathlib.Path(__file__).resolve().parent / "runs" / "bc-worker" / "demos.npz"
    # E6 探针集钉死:捕获而得之 sha 必为冻结常量(下游 DryAnchorSentinel 之
    # expected_sha256 由此值供给,探针集经此传递性钉死)。
    _assert_bc_v1_demos_frozen(demos)
    report = _validate_bc_report(demos.with_name("policy_sd.pt"), "data_gate")
    _, _, demos_sha256 = _load_dry_anchor_demos(demos, report.get("demos_sha256"))
    return demos_sha256


def _parse_dry_curriculum_schedule(spec: str) -> tuple[float, ...]:
    """解析 --dry-curriculum-schedule 为逐 rollout"序号→p"全表。

    语法(逗号分隔段,段内冒号分隔):
      linear:<p0>:<p1>:<n> —— n(≥2)个 rollout 端点含线性 p0→p1,
                              第 k 项 = p0 + (p1−p0)·k/(n−1),k=0..n−1;
      hold:<p>:<n>         —— n(≥1)个 rollout 恒 p。
    全部 p 须为 [0, 1] 内有限数。主表 = linear:1.0:0.5:147,hold:0.5:97。
    """
    _require(isinstance(spec, str) and bool(spec.strip()),
             "--dry-curriculum-schedule 不能为空")
    table: list[float] = []
    for raw_segment in spec.split(","):
        segment = raw_segment.strip()
        fields = segment.split(":")
        if fields[0] == "linear":
            _require(len(fields) == 4,
                     f"--dry-curriculum-schedule 段格式应为 linear:<p0>:<p1>:<n>: {segment!r}")
            try:
                p0, p1, n = float(fields[1]), float(fields[2]), int(fields[3])
            except ValueError as exc:
                raise ValueError(
                    f"--dry-curriculum-schedule 段数值不可解析: {segment!r}") from exc
            _require(n >= 2, f"linear 段须 n≥2(单点请用 hold): {segment!r}")
            values = [p0 + (p1 - p0) * k / (n - 1) for k in range(n)]
        elif fields[0] == "hold":
            _require(len(fields) == 3,
                     f"--dry-curriculum-schedule 段格式应为 hold:<p>:<n>: {segment!r}")
            try:
                p, n = float(fields[1]), int(fields[2])
            except ValueError as exc:
                raise ValueError(
                    f"--dry-curriculum-schedule 段数值不可解析: {segment!r}") from exc
            _require(n >= 1, f"hold 段须 n≥1: {segment!r}")
            values = [p] * n
        else:
            raise ValueError(
                f"--dry-curriculum-schedule 未知段类型(只允许 linear/hold): {segment!r}")
        _require(all(math.isfinite(v) and 0.0 <= v <= 1.0 for v in values),
                 f"--dry-curriculum-schedule p 值必须在 [0, 1] 内: {segment!r}")
        table.extend(values)
    return tuple(table)


# ---- E3 ④乙 辅助示范通路(PREREG-内容案 E3;两旗互不强制,零侵入条款) ----


def _bc_aux_active(args) -> bool:
    """E3 在位谓词:辅助通路在位 = λ_bc>0 ∧ --bc-aux-demos 给定。

    图纸字面:λ_bc=0 或未给 --bc-aux-demos 时零侵入——不加载、不采样、
    不进损失图(G0-2a 张量级恒等先决);两旗互不强制,单独给任一旗不报错。
    """
    return args.bc_aux_lambda > 0 and bool(args.bc_aux_demos)


def _load_bc_aux_demos_v2(path: str | pathlib.Path):
    """E3 ④乙:bc-worker-v2 示范集专用验证器(镜像断言按世代分别成文)。

    v1 面(_BC_REPORT_SCHEMA_VERSION=1/_validate_bc_report/_load_dry_anchor_demos/
    canonical bc-worker 路径)原封零触碰;本验证器单列,v2 demos schema 承图纸
    E2 共同真源 = v1 键(X/Y/episode_id)+ 逐样本 masks 数组(采集时
    env.action_masks() 现场捕获系唯一 on-manifold 真源,obs 反推口径禁用)。
    世代条件化禁采镜像:v2 禁 11 允 12。v1 之榨干态断言((x[:,297]==1).any())
    系 dry-anchor 探针专属,不随镜(施工注记)。返回 (X, Y, masks, sha256)。
    """
    import numpy as np

    p = pathlib.Path(path)
    try:
        payload = p.read_bytes()
    except OSError as exc:
        raise ValueError(f"④乙 v2 示范集不可读: {p}: {exc}") from exc
    sha256 = hashlib.sha256(payload).hexdigest()
    try:
        with np.load(io.BytesIO(payload), allow_pickle=False) as data:
            _require(all(key in data for key in ("X", "Y", "episode_id", "masks")),
                     "④乙 v2 demos.npz 缺少 X/Y/episode_id/masks(v2 schema 承 E2)")
            x, y = data["X"].copy(), data["Y"].copy()
            episode_id, masks = data["episode_id"].copy(), data["masks"].copy()
    except (OSError, ValueError) as exc:
        raise ValueError(f"④乙 v2 示范集不可读: {p}: {exc}") from exc
    _require(x.ndim == 2 and x.shape[1] == 298
             and y.ndim == 1 and len(x) == len(y),
             f"④乙 v2 数组形状异常:X={x.shape},Y={y.shape}")
    _require(x.dtype == np.float32 and np.issubdtype(y.dtype, np.integer),
             f"④乙 v2 dtype 异常:X={x.dtype},Y={y.dtype}")
    _require(episode_id.ndim == 1 and len(episode_id) == len(x)
             and np.issubdtype(episode_id.dtype, np.integer)
             and len(np.unique(episode_id)) >= 2,
             "④乙 v2 episode_id 形状/类型/独立局数异常")
    _require(masks.ndim == 2 and masks.shape == (len(x), 15)
             and masks.dtype == np.bool_,
             f"④乙 v2 masks 形状/dtype 异常:{getattr(masks, 'shape', None)},"
             f"{getattr(masks, 'dtype', None)}")
    _require(bool(((y >= 0) & (y < 15)).all()), "④乙 v2 标签越界")
    _require(not np.isin(y, _WORKER_BC_V2_FORBIDDEN_ACTIONS).any(),
             "④乙 v2 示范集含世代禁采动作 11(v2 禁 11 允 12,守卫面不弱化)")
    # 裁量强化注记:逐样本标签须为自身掩码合法位(on-manifold 真实执行拍之必然,
    # 掩位标签将使辅助 CE 取 -1e8 位);其对 12 类对蕴含图纸 m[12]=True 断言。
    _require(bool(masks[np.arange(len(y)), y].all()),
             "④乙 v2 存在标签被自身掩码禁止的示范对(on-manifold 破缺,fail-loud)")
    return x, y, masks, sha256


def _filter_bc_aux_demo_pairs(x, y, masks):
    """E3 主案:限 12 类示范对(正样本注入之字面实现,不与 KING 锚在其余动作上
    对拉);全类 OC 系图纸备选未接线(启用须另案批准,非本施工面)。"""
    import numpy as np

    keep = y == 12
    _require(bool(keep.any()),
             "--bc-aux-demos 中没有 12 类示范对(主案限 12 类,fail-loud)")
    kept_masks = masks[keep]
    # 图纸数据面断言(E3④/G0-2b/E7 逐字):全部 12 类示范对 m[12]=True。
    _require(bool(kept_masks[:, 12].all()),
             "12 类示范对存在 m[12]=False:采集面 on-manifold 破缺")
    return x[keep], y[keep], kept_masks


# ---- E4 契约 rev5 双键 + E6 探针集钉死(PREREG-内容案 E4/E6) ----


def _contract_dry_curriculum(args):
    """E4 rev5 键:课⑤不在位 = "disabled";在位 = {"schedule": CLI 字面表述}。

    载荷取 --dry-curriculum-schedule 之 CLI 字面串(与 D3 逐腿附加项列同源,
    L-cur/L-full 携主表;L-base 无 schedule → disabled)。schedule 字面即
    全表语义真源(_parse_dry_curriculum_schedule 确定性展开),resume 对账
    由字符串相等承载。
    """
    return ({"schedule": str(args.dry_curriculum_schedule)}
            if args.dry_curriculum_schedule else "disabled")


def _contract_bc_aux(args, bc_aux_demos_sha256):
    """E4 rev5 键:④乙不在位 = "disabled";在位 = {"lambda": λ, "demos_sha256"}。

    在位谓词 = _bc_aux_active(λ_bc>0 ∧ demos 给定,E3 单一真源);在位而缺
    v2 件 sha 系装配错误,fail-loud 拒写契约。
    """
    if not _bc_aux_active(args):
        return "disabled"
    _require(_is_sha256(bc_aux_demos_sha256),
             "④乙在位但缺 bc-worker-v2 示范集 sha256,拒绝写入 rev5 契约")
    return {"lambda": float(args.bc_aux_lambda),
            "demos_sha256": bc_aux_demos_sha256}


def _assert_bc_v1_demos_frozen(path: str | pathlib.Path) -> str:
    """E6 探针集钉死:BC-v1 demos【字节】≡ 冻结常量,失配即炸(加载处断言)。

    担保物 = dry-anchor 双参考 0.7515/0.6305 跨腿跨案可比之 demos 数组同一性
    (承工程 B2/minor-5);BC 案内重生成仅刷新回执字段、demos 字节不变
    (np.savez_compressed 字节确定,实测在案)。返回实测 sha256。
    """
    actual = _capture_file_sha256(path, "BC-v1 demos(E6 探针集)")
    _require(actual == _BC_V1_DEMOS_SHA256,
             f"E6 探针集钉死失败:BC-v1 demos 字节漂移 {actual} != "
             f"{_BC_V1_DEMOS_SHA256}(三腿仪表探针一律钉 v1 字节,禁换集)")
    return actual


def _validate_args(args) -> None:
    _require(args.total_steps > 0, "--total-steps 必须 > 0")
    _require(args.num_envs > 0, "--num-envs 必须 > 0")
    _require(args.n_steps > 0, "--n-steps 必须 > 0")
    rollout_quantum = args.n_steps * args.num_envs
    _require(args.total_steps % rollout_quantum == 0,
             "--total-steps 必须能被 n_steps * num_envs 整除，"
             f"否则 SB3 会静默向上多采样（当前量子 {rollout_quantum}）")
    _require(args.max_steps > 0, "--max-steps 必须 > 0")
    _require(math.isfinite(args.lr) and args.lr > 0, "--lr 必须是有限正数")
    _require(math.isfinite(args.gamma) and 0 <= args.gamma <= 1,
             "--gamma 必须在 [0, 1] 内")
    _require(math.isfinite(args.ent_coef) and args.ent_coef >= 0,
             "--ent-coef 必须是有限非负数")
    _require(math.isfinite(args.distill_beta) and args.distill_beta >= 0,
             "--distill-beta 必须是有限非负数")
    _require(args.freeze_policy_steps >= 0, "--freeze-policy-steps 不能为负")
    _require(args.ckpt_every_steps > 0, "--ckpt-every-steps 必须 > 0")
    _require(args.sentinel_every > 0, "--sentinel-every 必须 > 0")
    _require(args.dry_anchor_every > 0, "--dry-anchor-every 必须 > 0")
    # E5 仪表旋钮(只记不裁;0 = 不在位 = 零侵入,G0-2a 先决)
    _require(args.distill_ce_probe_every >= 0, "--distill-ce-probe-every 不能为负")
    _require(args.drywin_metrics_every >= 0, "--drywin-metrics-every 不能为负")
    _require(args.distill_ce_probe_every == 0
             or (args.worker and args.algo == "mppo"),
             "--distill-ce-probe-every 只适用于 --worker --algo mppo"
             "(探针需 Leashed 教师)")
    _require(args.drywin_metrics_every == 0 or args.worker,
             "--drywin-metrics-every 只适用于 --worker")
    if args.run_name is not None:
        _require(bool(args.run_name) and pathlib.Path(args.run_name).name == args.run_name
                 and args.run_name not in (".", ".."),
                 "--run-name 必须是单个目录名，不能含路径分隔符")
    if args.seed is not None:
        _require(0 <= args.seed < 2**32, "--seed 必须在 [0, 2**32) 内")
        _require(args.seed + args.num_envs - 1 < 2**32,
                 "--seed + num_envs - 1 必须小于 2**32")

    modes = int(args.worker) + int(args.options) + int(args.flat_clock)
    _require(modes <= 1, "--worker/--options/--flat-clock 互斥")
    # E1 两旗互斥断言(承工程 B1)+ 四门之互斥/模式门:谓词 = skip_dry ∨ schedule。
    _require(not (args.skip_dry and args.dry_curriculum_schedule),
             "--skip-dry 与 --dry-curriculum-schedule 互斥")
    _require(not _dry_window_mechanism_active(args) or args.worker,
             "--skip-dry/--dry-curriculum-schedule 只能与 --worker 同用")
    if args.dry_curriculum_schedule:
        curriculum_table = _parse_dry_curriculum_schedule(args.dry_curriculum_schedule)
        _require(len(curriculum_table) * rollout_quantum >= args.total_steps,
                 f"--dry-curriculum-schedule p 表 {len(curriculum_table)} 项"
                 f"不足以覆盖本腿 {args.total_steps // rollout_quantum} 个 rollout"
                 "(腿相对锚定禁越界钳位)")
    _require(not args.worker_npz or args.options, "--worker-npz 只能与 --options 同用")
    _require(not args.teacher_override or (args.resume_from and args.worker),
             "--teacher-override 只能与 worker 侧 --resume-from 同用")
    _require(not args.allow_manager_change or (args.resume_from and args.worker),
             "--allow-manager-change 只允许 worker resume 显式换经理")
    _require(not args.allow_legacy_resume or args.resume_from,
             "--allow-legacy-resume 只能与 --resume-from 同用")
    _require(args.freeze_policy_steps == 0 or args.bc_init,
             "--freeze-policy-steps > 0 时必须提供 --bc-init")
    _require(args.bc_init or args.init_source == "bc",
             "--init-source checkpoint 必须与 --bc-init 同用")
    _require(args.distill_beta == 0 or (args.worker and args.algo == "mppo"),
             "--distill-beta > 0 只适用于 --worker --algo mppo")
    _require(not (args.calib_probes or args.calib_record_only)
             or (args.worker and args.algo == "mppo"),
             "G-CAL 参数只适用于 --worker --algo mppo")
    # E3 ④乙:两旗互不强制(单独给任一旗不报错);在位 = λ_bc>0 ∧ demos 给定。
    _require(math.isfinite(args.bc_aux_lambda) and args.bc_aux_lambda >= 0,
             "--bc-aux-lambda 必须是有限非负数")
    _require(not _bc_aux_active(args) or (args.worker and args.algo == "mppo"),
             "④乙辅助通路(--bc-aux-lambda>0 且 --bc-aux-demos)"
             "只适用于 --worker --algo mppo")   # 承 --distill-beta 同型门(裁量注记)
    if _bc_aux_active(args):
        _require(pathlib.Path(args.bc_aux_demos).is_file(),
                 f"④乙 v2 示范集(bc-worker-v2)不存在: {args.bc_aux_demos}")
        # v2 专用验证器 + 12 类主案过滤 fail-loud(镜像 _precheck 先例,
        # 加载即弃;不在位时零侵入——连文件存在性都不查)
        _filter_bc_aux_demo_pairs(*_load_bc_aux_demos_v2(args.bc_aux_demos)[:3])
    _require(args.arch != "attn" or not (args.worker or args.options or args.flat_clock),
             "EntityAttention 只支持 295 维平面观测")

    if args.resume_from:
        _require((args.worker or args.options) and args.algo == "mppo",
                 "--resume-from 只支持 worker/options 的 mppo 检查点")
        _require(not args.bc_init and args.freeze_policy_steps == 0,
                 "--resume-from 禁与 --bc-init/--freeze-policy-steps 同用")
        _require(_checkpoint_path(args.resume_from).is_file(),
                 f"resume 检查点不存在: {args.resume_from}")
        if args.worker:
            _validate_leashed_checkpoint(args.resume_from)
        else:
            # v31 经理续训口:通用检查点闸(CRC/关键成员/步数/权重有限性),
            # distill_beta 系工人 Leashed 专属标记,经理检查点不作此断言。
            _validate_checkpoint_file(args.resume_from)
    if args.bc_init:
        _require(pathlib.Path(args.bc_init).is_file(), f"BC 权重不存在: {args.bc_init}")
        if args.init_source == "bc":
            gate = ("data_gate" if args.worker else "hypothesis" if args.options
                    else "memoryless_hypothesis")
            _validate_bc_report(pathlib.Path(args.bc_init), gate)
        else:
            _validate_export_manifest(pathlib.Path(args.bc_init))
    if args.teacher_override:
        _require(pathlib.Path(args.teacher_override).is_file(),
                 f"教师覆写文件不存在: {args.teacher_override}")
        _validate_export_manifest(pathlib.Path(args.teacher_override))
    if args.worker:
        _require(args.algo == "mppo" and args.gamma == 1.0 and args.max_steps == 3000,
                 "PREREG-v23:--worker 须配 --algo mppo --gamma 1.0 --max-steps 3000")
        _require(pathlib.Path(args.manager_npz).is_file(),
                 f"经理 npz 不存在: {args.manager_npz}")
        if args.distill_beta > 0 and not args.resume_from:
            _require(pathlib.Path(args.teacher_sd).is_file(),
                     f"教师 state_dict 不存在: {args.teacher_sd}")
            _validate_bc_report(pathlib.Path(args.teacher_sd), "data_gate")
        # E1 四门之 demos/BC 预检门:skip_dry ∨ schedule(谓词在助手内,断言原封)
        _precheck_dry_window_demos(args)
    if args.options:
        _require(args.algo == "mppo" and args.gamma == 1.0 and args.max_steps == 3000,
                 "PREREG-v25:--options 须配 --algo mppo --gamma 1.0 --max-steps 3000")
        if args.worker_npz:
            _require(args.n_steps == 64 and args.seed is not None,
                     "PREREG-v25 D2:换届选举须 --n-steps 64 且显式 --seed")
            _require(pathlib.Path(args.worker_npz).is_file(),
                     f"工人 npz 不存在: {args.worker_npz}")
    if args.seed is not None:
        bad = set(range(7000, 7032)) | set(range(9000, 9032))
        _require(not any(args.seed + rank in bad for rank in range(args.num_envs)),
                 "种子纪律:--seed + 实际 env rank 撞探针/金种子段")

    try:
        probes = [int(x) for x in args.calib_probes.split(",") if x.strip()]
    except ValueError as exc:
        raise ValueError("--calib-probes 必须是逗号分隔的整数") from exc
    _require(all(p >= 0 for p in probes), "--calib-probes 不能包含负数")
    _select_batch_size(args.n_steps, args.num_envs)


def _prepare_run_dir(run_dir: pathlib.Path, resume_from: str | None,
                     protected_inputs=()) -> None:
    """同名重跑时保全旧产物，同时避免 progress/tb/ckpt 混入新尝试。"""
    run_dir.mkdir(parents=True, exist_ok=True)
    run_root = run_dir.resolve()
    protected = [pathlib.Path(p).resolve() for p in protected_inputs if p]
    in_output_tree = [p for p in protected
                      if p == run_root or p.is_relative_to(run_root)]
    _require(not in_output_tree,
             "训练输入不能位于本次 run 输出目录内；请先复制到 train/models "
             f"或独立 inputs 目录: {in_output_tree}")
    existing = [run_dir / name for name in _RUN_ARTIFACTS if (run_dir / name).exists()]
    if not existing:
        return
    if resume_from:
        source = _checkpoint_path(resume_from).resolve()
        _require(all(p.resolve() != source
                     and not (p.is_dir() and source.is_relative_to(p.resolve()))
                     for p in existing),
                 "不能从同一 run_dir 的 model_final 原地 resume；请换 run-name")
    archive = run_dir / "_attempts" / f"pre-{time.strftime('%Y%m%d-%H%M%S')}-{time.time_ns()}"
    archive.mkdir(parents=True)
    for path in existing:
        shutil.move(str(path), str(archive / path.name))
    print(f"   同名 run 旧产物已归档: {archive}")


class _RunLock:
    """进程级独占锁；内核会在崩溃/SIGKILL 时自动释放 flock。"""

    def __init__(self, run_dir: pathlib.Path):
        run_dir.mkdir(parents=True, exist_ok=True)
        self.path = run_dir / ".run.lock"
        self._file = open(self.path, "a+", encoding="utf-8")
        try:
            fcntl.flock(self._file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self._file.seek(0)
            owner = self._file.read().strip() or "unknown"
            self._file.close()
            raise RuntimeError(f"run_dir 正被另一训练进程占用: {run_dir} ({owner})") from exc
        self._file.seek(0)
        self._file.truncate()
        self._file.write(json.dumps({"pid": os.getpid(), "started_at": time.time()}))
        self._file.flush()
        os.fsync(self._file.fileno())

    def close(self) -> None:
        if getattr(self, "_file", None) is None or self._file.closed:
            return
        fcntl.flock(self._file.fileno(), fcntl.LOCK_UN)
        self._file.close()

    def __del__(self):
        self.close()


class _TrainingResources:
    """Own the run lock and VecEnv from acquisition through every failure path."""

    def __init__(self):
        self.run_lock = None
        self.vec_env = None

    @staticmethod
    def _close_vec_env(vec_env) -> None:
        import signal
        import threading

        # SIGALRM is process-global and can only be installed from the main
        # thread.  Keep embedding/tests safe by falling back to an ordinary
        # close outside it, and restore any host handler/timer afterwards.
        armed = threading.current_thread() is threading.main_thread()
        previous_handler = previous_alarm = None

        def _close_timeout(*_):
            raise TimeoutError("vec_env.close() 超时(疑似 worker 已死)")

        if armed:
            previous_handler = signal.getsignal(signal.SIGALRM)
            previous_alarm = signal.alarm(0)
            try:
                signal.signal(signal.SIGALRM, _close_timeout)
                signal.alarm(20)
            except Exception:
                # Do not leave the caller's timer cancelled if signal setup is
                # unavailable in an unusual embedding environment.
                signal.signal(signal.SIGALRM, previous_handler)
                if previous_alarm:
                    signal.alarm(previous_alarm)
                armed = False
        try:
            vec_env.close()
        except Exception as exc:
            print(f"vec_env.close 异常(忽略,不影响已保存的模型): {exc}")
        finally:
            if armed:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, previous_handler)
                if previous_alarm:
                    signal.alarm(previous_alarm)

    def close(self) -> None:
        # Clear ownership before calling user/library cleanup so a recursive
        # or repeated close remains idempotent even when close itself raises.
        vec_env, self.vec_env = self.vec_env, None
        run_lock, self.run_lock = self.run_lock, None
        try:
            if vec_env is not None:
                self._close_vec_env(vec_env)
        finally:
            if run_lock is not None:
                run_lock.close()


def _atomic_save_model(model, destination: str | pathlib.Path) -> pathlib.Path:
    """先写唯一临时 zip、完整验 CRC/finite，再原子发布 canonical。"""
    final = pathlib.Path(destination)
    if final.suffix.lower() != ".zip":
        final = pathlib.Path(f"{final}.zip")
    final.parent.mkdir(parents=True, exist_ok=True)
    tmp = final.with_name(
        f".{final.stem}.{os.getpid()}.{time.time_ns()}.tmp.zip")
    try:
        model.save(str(tmp))
        # SB3 close() 只把字节交给内核；在校验/原子替换前强制落盘，
        # 避免断电后留下一个名字已发布但数据未持久化的 checkpoint。
        with open(tmp, "rb") as stream:
            os.fsync(stream.fileno())
        _validate_checkpoint_file(tmp, require_leashed=hasattr(model, "distill_beta"))
        os.replace(tmp, final)
        try:
            directory_fd = os.open(final.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except OSError:
            # 部分非 POSIX/网络文件系统不支持目录 fsync；文件本身
            # 仍已持久化，且 replace 的原子性不受影响。
            pass
    finally:
        tmp.unlink(missing_ok=True)
    return final


def _validate_worker_bc_evidence(rec: dict, demos_payload: bytes,
                                 policy_payload: bytes) -> None:
    """Recompute the worker holdout gate from its immutable evidence bytes."""
    import numpy as np
    import torch

    try:
        with np.load(io.BytesIO(demos_payload), allow_pickle=False) as archive:
            _require(set(archive.files) == {"X", "Y", "episode_id"},
                     "BC worker demos.npz 字段必须精确为 X/Y/episode_id")
            x = archive["X"]
            y = archive["Y"]
            episode_id = archive["episode_id"]
    except Exception as exc:
        if isinstance(exc, ValueError) and str(exc).startswith("BC worker"):
            raise
        raise ValueError("BC worker demos.npz 不可解析") from exc

    pairs = rec["pairs"]
    _require(x.ndim == 2 and x.shape == (pairs, 298)
             and x.dtype == np.float32,
             f"BC worker X 形状/dtype 与报告不一致: {x.shape}/{x.dtype}")
    _require(y.shape == (pairs,) and y.dtype == np.int64,
             f"BC worker Y 形状/dtype 与报告不一致: {y.shape}/{y.dtype}")
    _require(episode_id.shape == (pairs,) and episode_id.dtype == np.int64,
             "BC worker episode_id 形状/dtype 与报告不一致")
    _require(np.isfinite(x).all(), "BC worker demos X 含 NaN/Inf")
    _require(bool(((0 <= y) & (y < 15)).all()),
             "BC worker demos Y 含越界动作")
    _require(not np.isin(y, _WORKER_BC_FORBIDDEN_ACTIONS).any(),
             "BC worker demos Y 含禁采动作 11/12(11 恒掩;12 系教师排水后不采)")
    _require(bool((episode_id >= 0).all()),
             "BC worker demos episode_id 不能为负")

    episodes = np.unique(episode_id)
    expected_demo_seeds = np.asarray(_WORKER_BC_DEMO_SEEDS, dtype=np.int64)
    _require(np.array_equal(episodes, expected_demo_seeds),
             "BC worker demos 必须精确覆盖固定示范种子 100..227 各至少一对")
    order = np.random.default_rng(23).permutation(episodes)
    n_holdout = max(1, int(round(len(order) * 0.1)))
    expected_episodes = np.sort(order[:n_holdout])
    reported_episodes = np.asarray(rec["held_out_episodes"], dtype=np.int64)
    _require(np.array_equal(reported_episodes, expected_episodes),
             "BC worker held_out_episodes 与确定性整局切分不一致")
    holdout_indices = np.flatnonzero(np.isin(episode_id, expected_episodes))
    _require(len(holdout_indices) == rec["held_out_pairs"],
             "BC worker held_out_pairs 与 demos episode_id 重算不一致")
    _require(0 < len(holdout_indices) < pairs,
             "BC worker 确定性整局切分产生空训练集或空 held-out")

    try:
        state = torch.load(io.BytesIO(policy_payload), map_location="cpu",
                           weights_only=True)
    except Exception as exc:
        raise ValueError("BC worker policy state_dict 不可解析") from exc
    required = _POLICY_HEAD_KEYS
    _require(isinstance(state, dict) and set(state) == set(required),
             "BC worker policy state_dict 字段必须精确匹配策略头")
    tensors = [state[key] for key in required]
    _require(all(isinstance(value, torch.Tensor) for value in tensors),
             "BC worker policy state_dict 含非 Tensor 值")
    w0, b0, w1, b1, wa, ba = tensors
    _require(w0.shape == (64, 298) and b0.shape == (64,)
             and w1.shape == (64, 64) and b1.shape == (64,)
             and wa.shape == (15, 64) and ba.shape == (15,),
             "BC worker policy 必须是 298→64→64→15")
    _require(all(value.dtype == torch.float32 for value in tensors),
             "BC worker policy 策略头 dtype 必须是 float32")
    _require(all(torch.isfinite(value).all().item() for value in tensors),
             "BC worker policy 策略头含 NaN/Inf")

    # Mirror bc_worker.train_bc exactly: one CPU batch over X[holdout].  Using
    # different chunk sizes can select a different GEMM kernel and flip an
    # argmax at a near-tie even though the policy bytes are identical.
    with torch.no_grad():
        obs = torch.from_numpy(np.ascontiguousarray(x[holdout_indices]))
        hidden = torch.tanh(torch.nn.functional.linear(obs, w0, b0))
        hidden = torch.tanh(torch.nn.functional.linear(hidden, w1, b1))
        logits = torch.nn.functional.linear(hidden, wa, ba)
        pred = logits.argmax(1).cpu().numpy()
    heldout_y = y[holdout_indices]
    top1 = round(float((pred == heldout_y).mean()), 4)
    _require(rec["held_out_top1"] == top1,
             "BC worker held_out_top1 与 demos/policy 重算不一致: "
             f"{rec['held_out_top1']} != {top1}")

    full_counts = np.bincount(y, minlength=15)
    gated_classes = np.flatnonzero(full_counts >= 300)
    reported_recalls = {int(key): value
                        for key, value in rec["class_recalls"].items()}
    _require(set(reported_recalls) == set(map(int, gated_classes)),
             "BC worker class_recalls 类集合与 demos 全集计数不一致")
    for class_id in gated_classes:
        mask = heldout_y == class_id
        recall = (round(float((pred[mask] == class_id).mean()), 3)
                  if mask.any() else 0.0)
        _require(reported_recalls[int(class_id)] == recall,
                 "BC worker class_recalls 与 demos/policy 重算不一致: "
                 f"class={class_id}, {reported_recalls[int(class_id)]} != {recall}")


def _recompute_replay_bc_evidence(required_gate: str,
                                  policy_payload: bytes) -> dict:
    """Execute the deterministic BC demo/replay pools from frozen weights.

    Aggregate JSON is not evidence: a random policy plus edited means and SHA
    fields used to pass.  These gates are infrequent, pre-training operations,
    so correctness wins over the roughly minute-scale deterministic replay.
    """
    import numpy as np
    import torch

    dimensions = {
        "hypothesis": (303, 3),
        "memoryless_hypothesis": (296, 15),
    }
    _require(required_gate in dimensions, f"未知 replay BC gate: {required_gate}")
    obs_dim, action_dim = dimensions[required_gate]
    try:
        state = torch.load(io.BytesIO(policy_payload), map_location="cpu",
                           weights_only=True)
    except Exception as exc:
        raise ValueError("BC replay policy state_dict 不可解析") from exc
    _require(isinstance(state, dict) and set(state) == set(_POLICY_HEAD_KEYS),
             "BC replay policy state_dict 字段必须精确匹配策略头")
    tensors = [state[key] for key in _POLICY_HEAD_KEYS]
    _require(all(isinstance(value, torch.Tensor) for value in tensors),
             "BC replay policy state_dict 含非 Tensor 值")
    w0, b0, w1, b1, wa, ba = tensors
    _require(w0.shape == (64, obs_dim) and b0.shape == (64,)
             and w1.shape == (64, 64) and b1.shape == (64,)
             and wa.shape == (action_dim, 64) and ba.shape == (action_dim,),
             f"BC replay policy 必须是 {obs_dim}→64→64→{action_dim}")
    _require(all(value.dtype == torch.float32 for value in tensors),
             "BC replay policy 策略头 dtype 必须是 float32")
    _require(all(torch.isfinite(value).all().item() for value in tensors),
             "BC replay policy 策略头含 NaN/Inf")

    def policy_action(obs, mask) -> int:
        vector = np.asarray(obs, dtype=np.float32)
        _require(vector.shape == (obs_dim,),
                 f"BC replay 观测维度异常: {vector.shape} != {(obs_dim,)}")
        valid = np.asarray(mask, dtype=bool)
        _require(valid.shape == (action_dim,) and bool(valid.any()),
                 "BC replay 动作掩码维度异常或全假")
        with torch.no_grad():
            x = torch.from_numpy(vector)
            hidden = torch.tanh(torch.nn.functional.linear(x, w0, b0))
            hidden = torch.tanh(torch.nn.functional.linear(hidden, w1, b1))
            logits = torch.nn.functional.linear(hidden, wa, ba)
            logits = logits.masked_fill(~torch.from_numpy(valid), -torch.inf)
            return int(logits.argmax().item())

    if required_gate == "hypothesis":
        from diablogym import OptionsEnv
        from diablogym.options_env import DIVE, FARM

        env = OptionsEnv(max_steps=3000)

        def teacher_action(manager_env, _obs, _mask) -> int:
            raw = manager_env.env._raw
            return (DIVE if (manager_env.exhausted
                             or raw["char_level"] >= raw["dungeon_level"] + 2)
                    else FARM)

        def rollout(chooser, seed: int) -> tuple[float, int]:
            obs, _ = env.reset(seed=seed)
            done = trunc = False
            total = 0.0
            pairs = 0
            while not (done or trunc):
                mask = env.action_masks()
                action = int(chooser(env, obs, mask))
                if not mask[action]:
                    action = FARM
                obs, reward, done, trunc, _ = env.step(action)
                total += float(reward)
                pairs += 1
            return total, pairs

        try:
            demo = [rollout(teacher_action, seed)
                    for seed in _WORKER_BC_DEMO_SEEDS]
            replay = [rollout(
                lambda _env, obs, mask: policy_action(obs, mask), seed)[0]
                for seed in _BC_REPLAY_SEEDS]
            teacher_replay = [rollout(teacher_action, seed)[0]
                              for seed in _BC_REPLAY_SEEDS]
        finally:
            env.close()
        teacher_demo_mean = sum(value for value, _ in demo) / len(demo)
        bc_mean = sum(replay) / len(replay)
        teacher_mean = sum(teacher_replay) / len(teacher_replay)
        _require(teacher_mean > 0, "BC manager 重算 teacher replay 非正")
        return {
            "pairs": sum(count for _, count in demo),
            "teacher_demo_mean": teacher_demo_mean,
            "bc_replay_7000": bc_mean,
            "teacher_7000": teacher_mean,
            "ratio": bc_mean / teacher_mean,
        }

    from diablogym import DiabloGymEnv, StagnationClockWrapper
    from diablogym.options_env import KILL_PATIENCE, dispatch

    env = StagnationClockWrapper(DiabloGymEnv(
        ticks_per_step=4, max_steps=3000, start_in_dungeon=True,
        include_raw=False, descend_ladder=True, death_ladder=True))

    def teacher_action(flat_env, _obs) -> int:
        raw = flat_env.env._raw
        if flat_env._clock >= KILL_PATIENCE:
            return 11
        mode = ("dive" if raw["char_level"] >= raw["dungeon_level"] + 2
                else "farm")
        return dispatch(mode, raw, bool(flat_env.env.action_masks()[14]))

    def rollout(chooser, seed: int) -> tuple[float, int]:
        obs, _ = env.reset(seed=seed)
        done = trunc = False
        total = 0.0
        pairs = 0
        while not (done or trunc):
            action = int(chooser(env, obs))
            obs, reward, done, trunc, _ = env.step(action)
            total += float(reward)
            pairs += 1
        return total, pairs

    try:
        demo = [rollout(teacher_action, seed)
                for seed in _WORKER_BC_DEMO_SEEDS]
        replay = [rollout(
            lambda flat_env, obs: policy_action(
                obs, flat_env.env.action_masks()), seed)[0]
            for seed in _BC_REPLAY_SEEDS]
        teacher_replay = [rollout(teacher_action, seed)[0]
                          for seed in _BC_REPLAY_SEEDS]
    finally:
        env.close()
    teacher_demo_mean = sum(value for value, _ in demo) / len(demo)
    bc_mean = sum(replay) / len(replay)
    teacher_mean = sum(teacher_replay) / len(teacher_replay)
    _require(teacher_mean > 0, "BC flat 重算 teacher replay 非正")
    return {
        "pairs": sum(count for _, count in demo),
        "teacher_mean_demo": teacher_demo_mean,
        "bc_replay_mean_7000s": bc_mean,
        "teacher_replay_mean_7000s": teacher_mean,
        "ratio": bc_mean / teacher_mean,
    }


def _validate_bc_report(p: pathlib.Path, required_gate: str,
                        expected_implementation_sha256: str | None = None,
                        *, policy_payload: bytes | None = None,
                        report_payload: bytes | None = None,
                        verify_replay: bool = True) -> dict:
    """验证 BC 闸门，并绑定权重、生成器及完整训练运行时。

    调用方若已为 TOCTOU 安全冻结了 policy/report 字节，必须通过 payload
    传入；验证器不会再读路径。训练加载与组装评测因此共享完全相同的
    schema/指标/来源闸门，而不是各维护一份逐渐漂移的子集。
    """
    from eval_contract import EvalContractError, PROTOCOL_VERSION, strict_json_loads

    _require(required_gate in {"data_gate", "hypothesis", "memoryless_hypothesis"},
             f"未知 BC gate: {required_gate}")
    report = p.with_name("bc_report.json")
    try:
        frozen_report = (report.read_bytes() if report_payload is None
                         else report_payload)
        rec = strict_json_loads(frozen_report)
    except (OSError, EvalContractError) as exc:
        raise ValueError(f"BC 闸门报告缺失/不可读: {report}") from exc
    _require(isinstance(rec, dict), f"BC 闸门报告必须是 JSON 对象: {report}")
    _require(set(rec) == _BC_PASS_KEYS[required_gate],
             f"BC 闸门报告字段/schema 不匹配: {report}")
    _require(_is_plain_int(rec.get("schema_version"))
             and rec["schema_version"] == _BC_REPORT_SCHEMA_VERSION,
             f"BC 闸门报告 schema 过期: {rec.get('schema_version')!r}")
    _require(rec[required_gate] == "PASS",
             f"拒绝加载未过 {required_gate} 闸的 BC 权重: {rec[required_gate]!r}")
    expected_sha = rec.get("policy_sha256")
    _require(_is_sha256(expected_sha),
             f"BC 闸门报告缺少 policy_sha256 绑定: {report}")
    try:
        frozen_policy = p.read_bytes() if policy_payload is None else policy_payload
    except OSError as exc:
        raise ValueError(f"BC 权重缺失/不可读: {p}") from exc
    actual_sha = hashlib.sha256(frozen_policy).hexdigest()
    _require(actual_sha == expected_sha,
             f"BC 权重与闸门报告 SHA 不匹配: {actual_sha} != {expected_sha}")

    # A policy hash proves which bytes were loaded, but not which world made
    # their demonstrations.  In particular, pre-v3 worker demos may contain
    # trajectories that returned to town.  Bind every PASS report to the
    # current environment/native/content bundle and the exact BC generator.
    _require(rec.get("protocol_version") == PROTOCOL_VERSION,
             f"BC 报告协议过期: {rec.get('protocol_version')!r} != {PROTOCOL_VERSION}")
    expected_impl = (expected_implementation_sha256
                     if expected_implementation_sha256 is not None
                     else _implementation_bundle_sha256())
    _require(rec.get("implementation_sha256") == expected_impl,
             "BC 报告的实现/引擎/游戏内容身份与当前运行时不一致")
    root = pathlib.Path(__file__).resolve().parents[1]
    generator_name = {
        "data_gate": "bc_worker.py",
        "hypothesis": "bc_manager.py",
        "memoryless_hypothesis": "bc_flat.py",
    }[required_gate]
    generator_sha = hashlib.sha256(
        (root / "train" / generator_name).read_bytes()).hexdigest()
    _require(rec.get("generator_sha256") == generator_sha,
             f"BC 报告生成器已漂移: train/{generator_name}")
    if required_gate == "data_gate":
        pairs = rec["pairs"]
        held_out_pairs = rec["held_out_pairs"]
        top1 = _finite_number(rec["held_out_top1"], "BC held_out_top1")
        _require(_is_plain_int(pairs) and pairs > 0
                 and _is_plain_int(held_out_pairs) and 0 < held_out_pairs < pairs,
                 "BC worker 报告样本计数非法")
        _require(top1 >= 0.95 and top1 <= 1.0,
                 "BC worker 报告 PASS 与 held-out top-1 不一致")
        episodes = rec["held_out_episodes"]
        _require(isinstance(episodes, list) and episodes
                 and all(_is_plain_int(value) and value >= 0 for value in episodes)
                 and episodes == sorted(set(episodes)),
                 "BC worker held_out_episodes 非规范")
        _require(isinstance(rec["class_weighted_retry"], bool),
                 "BC worker class_weighted_retry 必须是 bool")
        recalls = rec["class_recalls"]
        _require(isinstance(recalls, dict), "BC worker class_recalls 必须是对象")
        for raw_class, raw_recall in recalls.items():
            try:
                class_id = int(raw_class)
            except (TypeError, ValueError) as exc:
                raise ValueError("BC worker class_recalls 键必须是动作编号") from exc
            _require(str(class_id) == str(raw_class) and 0 <= class_id < 15,
                     f"BC worker class_recalls 键非法: {raw_class!r}")
            recall = _finite_number(raw_recall,
                                    f"BC worker class_recalls[{raw_class!r}]")
            _require(0.85 <= recall <= 1.0,
                     "BC worker 报告 PASS 与逐类召回不一致")
        _require(_is_sha256(rec.get("demos_sha256")),
                 "BC worker 报告缺少 demos_sha256")
        demos_path = p.with_name("demos.npz")
        try:
            demos_payload = demos_path.read_bytes()
        except OSError as exc:
            raise ValueError(
                f"BC worker 示范集缺失/不可读: {demos_path}") from exc
        actual_demos_sha = hashlib.sha256(demos_payload).hexdigest()
        _require(actual_demos_sha == rec["demos_sha256"],
                 "BC worker demos.npz 与闸门报告 SHA 不匹配: "
                 f"{actual_demos_sha} != {rec['demos_sha256']}")
        _validate_worker_bc_evidence(rec, demos_payload, frozen_policy)
        manager = root / "train" / "models" / "v22-h-manager" / "policy.npz"
        manager_sha = hashlib.sha256(manager.read_bytes()).hexdigest()
        _require(rec.get("manager_npz_sha256") == manager_sha,
                 "BC worker 报告未绑定当前冻结 manager NPZ")
    else:
        pairs = rec["pairs"]
        ratio = _finite_number(rec["ratio"], "BC replay ratio")
        _require(_is_plain_int(pairs) and pairs > 0, "BC 报告样本计数非法")
        if required_gate == "hypothesis":
            demo_key, bc_key, teacher_key = (
                "teacher_demo_mean", "bc_replay_7000", "teacher_7000")
        else:
            demo_key, bc_key, teacher_key = (
                "teacher_mean_demo", "bc_replay_mean_7000s",
                "teacher_replay_mean_7000s")
        _finite_number(rec[demo_key], f"BC {demo_key}")
        bc_replay = _finite_number(rec[bc_key], f"BC {bc_key}")
        teacher_replay = _finite_number(rec[teacher_key], f"BC {teacher_key}")
        _require(teacher_replay > 0,
                 f"BC {teacher_key} 必须为正，replay ratio 才有定义")
        recomputed_ratio = bc_replay / teacher_replay
        _require(math.isclose(ratio, recomputed_ratio,
                              rel_tol=1e-12, abs_tol=1e-12),
                 "BC replay ratio 与报告中的 BC/teacher 指标不一致: "
                 f"{ratio} != {recomputed_ratio}")
        _require(ratio >= 0.85, "BC 报告 PASS 与 replay ratio 不一致")
        if verify_replay:
            cache_key = (required_gate, actual_sha, expected_impl, generator_sha)
            evidence = _BC_REPLAY_CACHE.get(cache_key)
            cache_miss = evidence is None
            if evidence is None:
                print(f"   BC {required_gate}: 重放固定 demo/replay 种子复核报告证据")
                evidence = _recompute_replay_bc_evidence(
                    required_gate, frozen_policy)
            for key, actual_value in evidence.items():
                reported_value = rec[key]
                if key == "pairs":
                    matches = (_is_plain_int(reported_value)
                               and reported_value == actual_value)
                else:
                    matches = (isinstance(reported_value, (int, float))
                               and not isinstance(reported_value, bool)
                               and math.isclose(float(reported_value),
                                                float(actual_value),
                                                rel_tol=1e-12,
                                                abs_tol=1e-9))
                _require(matches,
                         "BC replay 报告与冻结 policy/当前 runtime 重算不一致: "
                         f"{key}={reported_value!r} != {actual_value!r}")
            if cache_miss:
                _BC_REPLAY_CACHE[cache_key] = dict(evidence)
    return rec


def _load_dry_anchor_demos(path: str | pathlib.Path,
                           expected_sha256: str | None) -> tuple[object, object, str]:
    """Hash and parse the same demos.npz bytes, then enforce the BC binding."""
    import numpy as np

    p = pathlib.Path(path)
    _require(isinstance(expected_sha256, str) and len(expected_sha256) == 64,
             f"BC 闸门报告缺少 demos_sha256 绑定: {p.with_name('bc_report.json')}")
    try:
        payload = p.read_bytes()
    except OSError as exc:
        raise ValueError(f"dry-anchor 示范集不可读: {p}: {exc}") from exc
    actual_sha256 = hashlib.sha256(payload).hexdigest()
    _require(actual_sha256 == expected_sha256,
             f"dry-anchor demos SHA 不匹配: {actual_sha256} != {expected_sha256}")
    try:
        with np.load(io.BytesIO(payload), allow_pickle=False) as data:
            _require(all(key in data for key in ("X", "Y", "episode_id")),
                     "dry-anchor demos.npz 缺少 X/Y/episode_id")
            x, y = data["X"].copy(), data["Y"].copy()
            episode_id = data["episode_id"].copy()
    except (OSError, ValueError) as exc:
        raise ValueError(f"dry-anchor 示范集不可读: {p}: {exc}") from exc
    _require(x.ndim == 2 and x.shape[1] == 298
             and y.ndim == 1 and len(x) == len(y),
             f"dry-anchor 数组形状异常:X={x.shape},Y={y.shape}")
    _require(x.dtype == np.float32 and np.issubdtype(y.dtype, np.integer),
             f"dry-anchor dtype 异常:X={x.dtype},Y={y.dtype}")
    _require(episode_id.ndim == 1 and len(episode_id) == len(x)
             and np.issubdtype(episode_id.dtype, np.integer)
             and len(np.unique(episode_id)) >= 2,
             "dry-anchor episode_id 形状/类型/独立局数异常")
    _require(bool((x[:, 297] == 1.0).any()), "dry-anchor 示范集没有榨干态")
    return x, y, actual_sha256


def _export_manifest_path(p: pathlib.Path) -> pathlib.Path:
    return p.with_name(f"{p.name}.manifest.json")


def _validate_export_manifest(p: pathlib.Path) -> dict:
    import torch

    from eval_contract import EvalContractError, strict_json_loads

    manifest_path = _export_manifest_path(p)
    try:
        rec = strict_json_loads(manifest_path.read_bytes())
    except (OSError, EvalContractError) as exc:
        raise ValueError(f"checkpoint 导出清单缺失/不可读: {manifest_path}") from exc
    _require(isinstance(rec, dict), f"checkpoint 导出清单必须是 JSON 对象: {manifest_path}")
    _require(set(rec) == {
        "schema_version", "artifact_type", "artifact_sha256",
        "source_checkpoint", "source_checkpoint_sha256", "tensor_count"},
        f"checkpoint 导出清单字段异常: {manifest_path}")
    _require(_is_plain_int(rec["schema_version"])
             and rec["schema_version"] == _EXPORT_MANIFEST_SCHEMA_VERSION,
             f"checkpoint 导出清单 schema 非法: {rec['schema_version']!r}")
    _require(rec.get("artifact_type") == "checkpoint_policy_state",
             f"导出清单 artifact_type 异常: {rec.get('artifact_type')!r}")
    expected = rec.get("artifact_sha256")
    try:
        artifact_payload = p.read_bytes()
    except OSError as exc:
        raise ValueError(f"checkpoint 导出件不可读: {p}: {exc}") from exc
    actual = hashlib.sha256(artifact_payload).hexdigest()
    _require(_is_sha256(expected) and expected == actual,
             f"checkpoint 导出件与清单 SHA 不匹配: {actual} != {expected!r}")
    source_sha = rec.get("source_checkpoint_sha256")
    _require(_is_sha256(source_sha),
             "checkpoint 导出清单缺少 source_checkpoint_sha256")
    source_checkpoint = rec.get("source_checkpoint")
    _require(isinstance(source_checkpoint, str) and source_checkpoint
             and pathlib.Path(source_checkpoint).is_absolute(),
             "checkpoint 导出清单 source_checkpoint 必须是绝对路径")
    source_path = pathlib.Path(source_checkpoint)
    _require(str(source_path.resolve()) == source_checkpoint,
             "checkpoint 导出清单 source_checkpoint 必须是规范绝对路径")
    _require(source_path.resolve() != p.resolve(),
             "checkpoint 导出件不能把自身声明为源 checkpoint")
    try:
        source_payload = source_path.read_bytes()
    except OSError as exc:
        raise ValueError(
            f"checkpoint 导出清单声明的源 checkpoint 不可读: {source_path}") from exc
    actual_source_sha = hashlib.sha256(source_payload).hexdigest()
    _require(actual_source_sha == source_sha,
             "checkpoint 导出清单的源 checkpoint SHA 不匹配: "
             f"{actual_source_sha} != {source_sha}")
    _validate_checkpoint_bytes(source_payload, str(source_path))

    tensor_count = rec.get("tensor_count")
    _require(_is_plain_int(tensor_count) and tensor_count > 0,
             "checkpoint 导出清单 tensor_count 必须是正整数")
    try:
        artifact_state = torch.load(
            io.BytesIO(artifact_payload), map_location="cpu", weights_only=True)
        with zipfile.ZipFile(io.BytesIO(source_payload)) as source_archive:
            source_state = torch.load(
                io.BytesIO(source_archive.read("policy.pth")),
                map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ValueError("checkpoint 导出件或源 policy state_dict 不可解析") from exc
    _require(isinstance(artifact_state, dict) and isinstance(source_state, dict),
             "checkpoint 导出件与源 policy 必须都是 state_dict")
    _require(tensor_count == len(artifact_state) == len(source_state),
             "checkpoint 导出清单 tensor_count 与导出件/源 policy 不一致")
    _require(set(artifact_state) == set(source_state),
             "checkpoint 导出件字段与源 checkpoint policy 不一致")
    for key in source_state:
        source_value = source_state[key]
        artifact_value = artifact_state[key]
        _require(isinstance(source_value, torch.Tensor)
                 and isinstance(artifact_value, torch.Tensor),
                 f"checkpoint policy 字段不是 Tensor: {key}")
        _require(torch.isfinite(artifact_value).all().item(),
                 f"checkpoint 导出件含 NaN/Inf: {key}")
        _require(artifact_value.shape == source_value.shape
                 and artifact_value.dtype == source_value.dtype
                 and torch.equal(artifact_value, source_value),
                 f"checkpoint 导出件张量与源 checkpoint policy 不一致: {key}")
    return rec


def _load_bc_state_dict(path: str, policy, required_gate: str,
                        source_kind: str = "bc") -> dict:
    """校验 BC 闸门与关键张量，禁止“0 键命中但 strict=False”静默起跑。"""
    import torch

    p = pathlib.Path(path)
    manifest = None
    if source_kind == "bc":
        expected_sha256 = _validate_bc_report(p, required_gate)["policy_sha256"]
    elif source_kind == "checkpoint":
        manifest = _validate_export_manifest(p)
        expected_sha256 = manifest["artifact_sha256"]
    else:
        raise ValueError(f"未知 init source kind: {source_kind}")

    try:
        payload = p.read_bytes()
    except OSError as exc:
        raise ValueError(f"--bc-init 权重不可读: {p}: {exc}") from exc
    actual_sha256 = hashlib.sha256(payload).hexdigest()
    _require(actual_sha256 == expected_sha256,
             f"BC/init 权重在闸门校验后发生漂移: "
             f"{actual_sha256} != {expected_sha256}")
    # Integrity check and torch deserialization consume the same bytes.
    sd = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=True)
    _require(isinstance(sd, dict), "--bc-init 必须是 policy state_dict")
    bad_types = [k for k, value in sd.items() if not isinstance(value, torch.Tensor)]
    _require(not bad_types, f"BC state_dict 含非 Tensor 值: {bad_types}")
    nonfinite = [k for k, value in sd.items()
                 if not torch.isfinite(value).all().item()]
    _require(not nonfinite, f"BC state_dict 含 NaN/Inf: {nonfinite}")
    target = policy.state_dict()
    if source_kind == "checkpoint":
        missing_all = sorted(set(target) - set(sd))
        unexpected_all = sorted(set(sd) - set(target))
        mismatched_all = sorted(
            key for key in set(target) & set(sd)
            if target[key].shape != sd[key].shape)
        dtype_mismatched = sorted(
            key for key in set(target) & set(sd)
            if target[key].dtype != sd[key].dtype)
        _require(not missing_all and not unexpected_all and not mismatched_all
                 and not dtype_mismatched,
                 "checkpoint 全量 policy state_dict 与目标不精确一致: "
                 f"missing={missing_all}, unexpected={unexpected_all}, "
                 f"shape={mismatched_all}, dtype={dtype_mismatched}")
        _require(manifest is not None and manifest["tensor_count"] == len(sd),
                 "checkpoint 导出清单 tensor_count 与 state_dict 不一致")
    missing = [k for k in _POLICY_HEAD_KEYS if k not in sd]
    mismatched = [k for k in _POLICY_HEAD_KEYS if k in sd and
                  (k not in target or target[k].shape != sd[k].shape)]
    _require(not missing, f"BC state_dict 缺少策略头键: {missing}")
    _require(not mismatched, f"BC state_dict 策略头形状不匹配: {mismatched}")
    return sd


def make_env(max_steps: int = 1500, deep: bool = False, death_ladder: bool = False,
             options: bool = False, flat_clock: bool = False,
             worker: bool = False, manager_npz: str | None = None,
             worker_npz: str | None = None, skip_dry: float | bool = False,
             drink_sovereignty: bool = True,
             manager_npz_sha256: str | None = None,
             worker_npz_sha256: str | None = None,
             implementation_sha256: str | None = None):
    if implementation_sha256 is not None:
        actual_implementation = _implementation_bundle_sha256()
        _require(actual_implementation == implementation_sha256,
                 "训练实现 bundle 在 VecEnv 创建前发生漂移: "
                 f"{actual_implementation} != {implementation_sha256}")
    from diablogym import DiabloGymEnv

    def with_seed_discipline(env):
        """平面/策略脑通用的逐局种子包装；worker 自身有跨窗口局状态，不套。"""
        import gymnasium as gym
        import numpy as np
        from diablogym.worker_env import sample_train_seed

        class _SeedDiscipline(gym.Wrapper):
            def __init__(self, wrapped):
                super().__init__(wrapped)
                self._seed_rng = np.random.default_rng()

            def reset(self, *, seed=None, options=None):
                if seed is not None:
                    if 7000 <= seed < 7032 or 9000 <= seed < 9032:
                        raise ValueError(f"训练 reset 拒绝保留种子 {seed}")
                    self._seed_rng = np.random.default_rng(seed)
                else:
                    seed = sample_train_seed(self._seed_rng)
                obs, info = self.env.reset(seed=seed, options=options)
                info["episode_seed"] = seed
                return obs, info

            def action_masks(self):
                return self.env.action_masks()

        return Monitor(_SeedDiscipline(env))

    if worker:
        # v23:FARM 操作脑在位训练——episode = 冻结 H 经理选中的一个 FARM 窗口
        # (rng_seed=None → 各子进程独立熵源,种子采样器拒采 7000/9000 段)
        # v26:skip_dry=True 时干层复访窗由脚本代跑,不进学习分布(绿洲处方)
        from diablogym import WorkerWindowEnv
        return Monitor(WorkerWindowEnv(manager_npz=manager_npz, max_steps=max_steps,
                                       skip_dry=skip_dry,
                                       drink_sovereignty=drink_sovereignty,
                                       manager_sha256=manager_npz_sha256))
    if options:
        # v22:策略脑/操作脑——OptionsEnv 自带 deep+death_ladder 默认
        # v25:worker_npz 非空时挂 npz 工人(NumpyManager 在本函数体内构造——
        # spawn 子进程免 torch,PREREG-v25 D1 条款),并套种子纪律薄包装
        from diablogym import NumpyManager, OptionsEnv

        if worker_npz:
            # 条款要点:工人以 npz+numpy 前向进子进程(不 pickle 网络、不 load SB3
            # 模型、不逐拍 torch 前向)。torch 模块本身随 train_ppo 顶层 import 进入
            # 子进程(v23 先例同),"无 torch"断言不可实现,预注册已如实修正。
            net = NumpyManager(worker_npz, expected_sha256=worker_npz_sha256)
            net.require_io_shape(298, 15, "Options worker")
            env = OptionsEnv(max_steps=max_steps,
                             drink_sovereignty=drink_sovereignty,
                             workers={0: lambda obs, mask: net.choose(obs, mask)})
        else:
            env = OptionsEnv(max_steps=max_steps,
                             drink_sovereignty=drink_sovereignty)

        # 无论是否挂 npz 工人，经理训练都必须遵守同一种子纪律。
        return with_seed_discipline(env)
    if flat_clock:
        # v22 恶魔臂 F:296 维平面(停滞钟与策略脑同一块表)
        from diablogym import StagnationClockWrapper
        return with_seed_discipline(StagnationClockWrapper(DiabloGymEnv(
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
    return with_seed_discipline(env)


def _is_publishable_rollout_boundary(model) -> bool:
    """A full buffer is insufficient when G-CAL rejected its update."""
    rollout_full = bool(getattr(getattr(model, "rollout_buffer", None),
                                "full", False))
    return rollout_full and not bool(getattr(model, "_calib_tripped", False))


class AtomicRolloutCheckpointCallback(BaseCallback):
    """只在上一 rollout 已完成更新的边界保存，杜绝“步数已记、梯度未吃”。"""

    def __init__(self, run_dir: pathlib.Path, every_steps: int = 250_000,
                 implementation_sha256: str | None = None):
        super().__init__()
        self.run_dir = run_dir
        self.every_steps = every_steps
        self.implementation_sha256 = implementation_sha256
        self.period = None
        self.next_at = None
        self.last_saved = None

    def _on_training_start(self) -> None:
        quantum = int(self.model.n_steps * self.model.get_env().num_envs)
        self.period = max(quantum, (self.every_steps // quantum) * quantum)
        self.next_at = int(self.num_timesteps + self.period)

    def _save_due(self) -> None:
        if self.next_at is None or self.num_timesteps < self.next_at:
            return
        # G-CAL trips on the first minibatch and intentionally skips its
        # optimizer step.  num_timesteps has nevertheless advanced by the full
        # rollout.  A checkpoint here would revive the exact "steps recorded,
        # gradients not consumed" state this callback exists to prevent.
        if bool(getattr(getattr(self, "model", None),
                        "_calib_tripped", False)):
            print("   [G-CAL] 当前 rollout 未完成更新，拒绝发布 checkpoint")
            return
        step = int(self.num_timesteps)
        if self.last_saved != step:
            if self.implementation_sha256 is not None:
                actual = _implementation_bundle_sha256()
                _require(actual == self.implementation_sha256,
                         "训练期间实现/引擎/游戏内容发生漂移，拒绝发布 checkpoint: "
                         f"{actual} != {self.implementation_sha256}")
            path = self.run_dir / "ckpt" / f"model_{step}_steps.zip"
            _atomic_save_model(self.model, path)
            self.last_saved = step
            print(f"   rollout-boundary checkpoint: {path}")
        while self.next_at <= step:
            self.next_at += self.period

    def _on_rollout_start(self) -> None:
        # 首次调用 num_timesteps=起点；后续调用发生在上一 rollout train() 之后。
        self._save_due()

    def _on_step(self) -> bool:
        return True

    def _on_training_end(self) -> None:
        # 正常收官不会进入下一次 _on_rollout_start；只有 buffer.full 才表示
        # 最后一批确已 train()。回调中途终止的半 rollout 不得冒充 checkpoint。
        if _is_publishable_rollout_boundary(self.model):
            self._save_due()


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
        self._last_emit_step = None

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
            while self.num_timesteps >= self.next_at:
                self.next_at += self.every
            self._emit(final=False)
        return True

    def _emit(self, final: bool) -> None:
        import numpy as np

        step = int(self.num_timesteps)
        if self._last_emit_step == step:
            return
        per_env = self.model.get_env().get_attr("stats")   # 经 Monitor.__getattr__ 透传
        agg = {"windows": 0, "dry": 0, "fresh": 0, "ff_windows": 0,
               "ff_dry": 0, "episodes": 0, "reseeds": 0}    # ff_dry: v26 绿洲口径
        reasons = {}
        for stats in per_env:
            for key in agg:
                agg[key] += stats.get(key, 0)
            for key, value in stats.get("reasons", {}).items():
                reasons[key] = reasons.get(key, 0) + value
        top1 = int(self.action_counts.argmax()) if self.action_counts is not None else -1
        share = (float(self.action_counts[top1] / max(1, self.action_counts.sum()))
                 if top1 >= 0 else 0.0)
        line = {"sentinel": "v23", "step": step, **agg,
                "dry_share": round(agg["dry"] / max(1, agg["dry"] + agg["fresh"]), 4),
                "reasons": reasons, "top1_action": top1, "top1_share": round(share, 4),
                # v24 皮筋读数(与 gate_ledger 双簿对账)
                "beta": getattr(self.model, "distill_beta", None),
                "distill_ce": getattr(self.model, "_last_distill_ce", None),
                "teacher_diverge": getattr(self.model, "_last_diverge", None)}
        if final:
            line["final"] = True
        with open(self.run_dir / "sentinel.jsonl", "a") as f:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
        self._last_emit_step = step
        print(f"   [哨兵] {line}")

    def _on_training_end(self) -> None:
        # 腿长常取 499,712(<500k)；若只按间隔写，完整腿反而零哨兵记录。
        if self.num_timesteps > 0 and self._last_emit_step != int(self.num_timesteps):
            self._emit(final=True)


class DryAnchorSentinel(BaseCallback):
    """v26 干层锚哨兵(只记不裁,PREREG-v26 R26.6):demos.npz 中榨干旗=1 的教师态
    固定抽 2000,每 500k 步测学生 argmax 对教师标签的失配率——skip_dry 下干层行为
    无锚裸奔,这只表是它唯一的观察者。"""

    def __init__(self, run_dir: pathlib.Path, demos_npz: str,
                 expected_sha256: str, every: int = 500_000):
        super().__init__()
        import numpy as np
        self.run_dir = run_dir
        self.every = every
        self.next_at = every
        self._last_emit_step = None
        X, Y, self.demos_sha256 = _load_dry_anchor_demos(
            demos_npz, expected_sha256)
        m = X[:, 297] == 1.0      # 观测第 298 维 = 榨干旗(工人观测契约)
        idx = np.random.default_rng(26).choice(
            np.flatnonzero(m), size=min(2000, int(m.sum())), replace=False)
        self.X, self.Y = X[idx], Y[idx]
        if (not np.isfinite(self.X).all() or np.any(self.Y < 0)
                or np.any(self.Y >= 15)):
            raise ValueError("dry-anchor 样本含非有限观测或越界标签")

    def _on_training_start(self) -> None:
        self.next_at = ((self.num_timesteps // self.every) + 1) * self.every

    def _on_step(self) -> bool:
        if self.num_timesteps >= self.next_at:
            while self.num_timesteps >= self.next_at:
                self.next_at += self.every
            self._emit(final=False)
        return True

    def _emit(self, final: bool) -> None:
        import numpy as np
        import torch as th

        step = int(self.num_timesteps)
        if self._last_emit_step == step:
            return
        with th.no_grad():
            obs = th.as_tensor(self.X, device=self.model.device)
            # 干层锚系只记不裁遥测:掩码保留 v28-v30 旧口径(11/12 恒掩)
            # 以维持跨腿失配曲线同尺——v32 主权后部署掩码含 12,但教师
            # 标签(demos)无 12,主权行为由评测 a12 仪表另量(PREREG-v32
            # 口径注);改此口径会断代历史遥测,故如实登记不改。
            masks = th.ones((len(self.X), 15), dtype=th.bool,
                            device=self.model.device)
            masks[:, 11] = masks[:, 12] = False
            masks[:, 14] = obs[:, _GEAR_PRESENT_INDEX] > 0.5
            dist = self.model.policy.get_distribution(obs, action_masks=masks)
            pred = dist.distribution.logits.argmax(-1).cpu().numpy()
        mis = float((pred != self.Y).mean())
        line = {"sentinel": "dry-anchor", "step": step,
                "mismatch": round(mis, 4), "n": int(len(self.Y))}
        if final:
            line["final"] = True
        with open(self.run_dir / "sentinel.jsonl", "a") as f:
            f.write(json.dumps(line) + "\n")
        self._last_emit_step = step
        print(f"   [干层锚] {line}")

    def _on_training_end(self) -> None:
        if self.num_timesteps > 0 and self._last_emit_step != int(self.num_timesteps):
            self._emit(final=True)


class DryCurriculumCallback(BaseCallback):
    """E1 ⑤A 课程回调:_on_rollout_start 于采集开始前推送本 rollout 之 p_skip。

    - p 表锚定腿相对 rollout 序号 (num_timesteps − 3,497,984)/2048,
      全局步锚定禁用;腿前/失准/越界一律抛(禁钳位——对抗席"越界钳至表尾
      恒 0.5"构造由此关死);
    - schedule_table 属性暴露"序号→p"全表(供驱动器落 DRY_CURRICULUM_TABLE);
    - 逐 rollout 落账实际推送 p(pushed 账 + run_dir/dry_curriculum.jsonl),
      并回调内断言 env 读回值 ≡ 注册表对应项,失配即抛(rev3 勘正卷 E1② 逐字);
    - 推送经 VecEnv.env_method("set_skip_dry_p", p) 过 Monitor __getattr__
      透传至 WorkerWindowEnv;回调序钉死于回调列首(见 _main cbs 组装)。
    """

    def __init__(self, schedule_table, run_dir: pathlib.Path | None = None,
                 leg_start: int = _DRY_CURRICULUM_LEG_START):
        super().__init__()
        table = tuple(float(p) for p in schedule_table)
        _require(len(table) > 0, "dry-curriculum p 表不能为空")
        _require(all(math.isfinite(p) and 0.0 <= p <= 1.0 for p in table),
                 "dry-curriculum p 表必须全部在 [0, 1] 内")
        self.schedule_table = table
        self.leg_start = int(leg_start)
        self.run_dir = pathlib.Path(run_dir) if run_dir is not None else None
        self.pushed: list[dict] = []   # 逐 rollout 实际推送账(序号→p)
        self.quantum = None

    def _on_training_start(self) -> None:
        self.quantum = int(self.model.n_steps * self.model.get_env().num_envs)

    def _rollout_index(self) -> int:
        offset = int(self.num_timesteps) - self.leg_start
        _require(offset >= 0,
                 f"dry-curriculum 腿相对锚定失义: num_timesteps={self.num_timesteps} "
                 f"在腿起点 {self.leg_start} 之前(全局步锚定禁用,腿恒自王 zip 复点火)")
        _require(offset % self.quantum == 0,
                 f"dry-curriculum rollout 边界失准: 腿内偏移 {offset} "
                 f"不是量子 {self.quantum} 的整数倍")
        index = offset // self.quantum
        _require(index < len(self.schedule_table),
                 f"dry-curriculum 腿相对 rollout 序号 {index} 越界"
                 f"(p 表长 {len(self.schedule_table)},禁钳位)")
        return index

    def _on_rollout_start(self) -> None:
        index = self._rollout_index()
        p = self.schedule_table[index]
        env = self.model.get_env()
        env.env_method("set_skip_dry_p", p)   # 经 Monitor __getattr__ 透传
        # rev3 E1② 恒等断言:实际在位 p(逐 env 读回)≡ 注册表对应项,失配即抛。
        for rank, actual in enumerate(env.get_attr("skip_dry")):
            _require(float(actual) == p,
                     f"dry-curriculum 恒等断言失配: env[{rank}] 在位 p={actual} "
                     f"!= 注册表[{index}]={p}")
        entry = {"rollout_index": int(index), "p": float(p),
                 "num_timesteps": int(self.num_timesteps)}
        self.pushed.append(entry)
        if self.run_dir is not None:
            with open(self.run_dir / "dry_curriculum.jsonl", "a") as f:
                f.write(json.dumps(entry) + "\n")

    def _on_step(self) -> bool:
        return True


# ---- E5 新增仪表(PREREG-内容案 E5;全数只记不裁,默认零侵入,纳入 W-G0
# 证明范围;探针示范集一律钉 BC-v1 demos 字节,E6) ----


def _probe_legacy_masks(obs):
    """E5 探针共用旧口径掩码(承 DryAnchorSentinel v28-v30 口径逐字:11/12 恒掩,
    14 依 gear 位)——示范集系 v1 世代无逐样本掩码,完整部署掩码不可自 obs 全量
    重构(E2 注记,反推口径系第二真源禁用);取跨腿同尺之旧口径并在输出行注记
    mask_mode,只记不裁(施工裁量,交接单单列)。"""
    import torch as th

    masks = th.ones((len(obs), 15), dtype=th.bool, device=obs.device)
    masks[:, 11] = masks[:, 12] = False
    masks[:, 14] = obs[:, _GEAR_PRESENT_INDEX] > 0.5
    return masks


class DistillCeProbe(BaseCallback):
    """E5① 干/鲜 distill_ce 分列离线探针(只记不裁;DryAnchorSentinel 之孪生件)。

    固定示范态集(BC-v1 demos,E6 字节钉死)上按 x[:,297] 干旗分干/鲜两组,
    算教师-学生 distill CE(公式镜像 leashed_ppo train() 皮筋段:
    ce = −Σ t_probs·logp_all 之均值;教师/学生同喂旧口径掩码);专用 rng
    (承 dry-anchor rng(26) 先例),零触训练路径(纯读+IO,不碰训练 RNG/梯度/
    env 流)。训练内 buffer 分列为守 G0-2a 零侵入证明面而废止(工程 M2),
    本件即其注册替代形制;课②定标数据供给义务(圈 12 改写)同由此满足。
    输出 run_dir/distill_ce_probe.jsonl。
    """

    def __init__(self, run_dir: pathlib.Path, demos_npz: str, every: int):
        super().__init__()
        import numpy as np

        self.run_dir = run_dir
        self.every = int(every)
        _require(self.every > 0, "distill-ce 探针间隔必须 > 0")
        self.next_at = self.every
        self._last_emit_step = None
        # E6:加载处断言 = 冻结常量即 expected_sha256(伪字节必炸)。
        X, _, self.demos_sha256 = _load_dry_anchor_demos(
            demos_npz, _BC_V1_DEMOS_SHA256)
        dry_rows = np.flatnonzero(X[:, 297] == 1.0)
        fresh_rows = np.flatnonzero(X[:, 297] == 0.0)
        _require(len(dry_rows) > 0 and len(fresh_rows) > 0,
                 "distill-ce 探针需干/鲜两组示范态均非空(fail-loud)")
        rng = np.random.default_rng(_E5_PROBE_RNG_SEED)
        dry_idx = rng.choice(dry_rows,
                             size=min(_E5_PROBE_GROUP_CAP, len(dry_rows)),
                             replace=False)
        fresh_idx = rng.choice(fresh_rows,
                               size=min(_E5_PROBE_GROUP_CAP, len(fresh_rows)),
                               replace=False)
        self.X_dry, self.X_fresh = X[dry_idx], X[fresh_idx]
        if not (np.isfinite(self.X_dry).all() and np.isfinite(self.X_fresh).all()):
            raise ValueError("distill-ce 探针样本含非有限观测")

    def _on_training_start(self) -> None:
        self.next_at = ((self.num_timesteps // self.every) + 1) * self.every

    def _on_step(self) -> bool:
        if self.num_timesteps >= self.next_at:
            while self.num_timesteps >= self.next_at:
                self.next_at += self.every
            self._emit(final=False)
        return True

    def _group_ce(self, x) -> float:
        import torch as th

        from leashed_ppo import HUGE_NEG

        with th.no_grad():
            obs = th.as_tensor(x, device=self.model.device)
            masks = _probe_legacy_masks(obs)
            t_logits = self.model.teacher(obs)
            t_logits = th.where(masks, t_logits,
                                th.full_like(t_logits, HUGE_NEG))
            t_probs = th.softmax(t_logits, dim=-1)
            dist = self.model.policy.get_distribution(obs, action_masks=masks)
            logp_all = dist.distribution.logits   # 归一化 log-probs(掩位≈-1e8)
            return float(-(t_probs * logp_all).sum(dim=-1).mean())

    def _emit(self, final: bool) -> None:
        step = int(self.num_timesteps)
        if self._last_emit_step == step:
            return
        _require(getattr(self.model, "teacher", None) is not None,
                 "distill-ce 探针需教师在位(Leashed teacher;fail-loud)")
        line = {"probe": "distill-ce", "step": step,
                "dry_ce": round(self._group_ce(self.X_dry), 6),
                "dry_n": int(len(self.X_dry)),
                "fresh_ce": round(self._group_ce(self.X_fresh), 6),
                "fresh_n": int(len(self.X_fresh)),
                "beta": getattr(self.model, "distill_beta", None),
                "mask_mode": "dry-anchor-legacy",
                "demos_sha16": self.demos_sha256[:16]}
        if final:
            line["final"] = True
        with open(self.run_dir / "distill_ce_probe.jsonl", "a") as f:
            f.write(json.dumps(line) + "\n")
        self._last_emit_step = step
        print(f"   [干/鲜蒸馏探针] {line}")

    def _on_training_end(self) -> None:
        if self.num_timesteps > 0 and self._last_emit_step != int(self.num_timesteps):
            self._emit(final=True)


class DryWindowMetricsCallback(BaseCallback):
    """E5② 干窗行为仪表(只记不裁;审计缺口 i 之闭合起点,基线自本案首建)。

    两读数面,均挂现有采样面、零新增训练侧接触:
    ① 干态动作分布——固定干态示范集(BC-v1 demos 干旗态,E6 字节钉死,
       dry-anchor 同款采样面)上学生策略之分布熵与 argmax 直方图(旧口径掩码,
       mask_mode 注记随行);
    ② 窗口经济——SB3 rollout infos 流之窗末 option_extra(学习窗,快进窗
       不经此流)按干/鲜分组聚合工资 W 与宽度(τ̄/depth=dlvl_end);逐 emit
       区间清零(区间局部均值);n=0 组记 n:0、均值 null 不消失(fail-closed)。
    输出 run_dir/drywin_metrics.jsonl(台账词 DRYWIN_METRICS 之进程侧原料)。
    """

    _WINDOW_KEYS = ("n", "wage_sum", "tau_sum", "depth_sum")

    def __init__(self, run_dir: pathlib.Path, demos_npz: str, every: int):
        super().__init__()
        import numpy as np

        self.run_dir = run_dir
        self.every = int(every)
        _require(self.every > 0, "drywin 仪表间隔必须 > 0")
        self.next_at = self.every
        self._last_emit_step = None
        # E6:加载处断言 = 冻结常量即 expected_sha256(伪字节必炸)。
        X, _, self.demos_sha256 = _load_dry_anchor_demos(
            demos_npz, _BC_V1_DEMOS_SHA256)
        dry_rows = np.flatnonzero(X[:, 297] == 1.0)
        _require(len(dry_rows) > 0, "drywin 仪表需干态示范集非空(fail-loud)")
        rng = np.random.default_rng(_E5_PROBE_RNG_SEED)
        idx = rng.choice(dry_rows,
                         size=min(_E5_PROBE_GROUP_CAP, len(dry_rows)),
                         replace=False)
        self.X_dry = X[idx]
        if not np.isfinite(self.X_dry).all():
            raise ValueError("drywin 仪表干态样本含非有限观测")
        self._acc = self._fresh_acc()

    @classmethod
    def _fresh_acc(cls) -> dict:
        return {group: dict.fromkeys(cls._WINDOW_KEYS, 0)
                for group in ("dry", "fresh")}

    def _on_training_start(self) -> None:
        self.next_at = ((self.num_timesteps // self.every) + 1) * self.every

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", ()):
            extra = info.get("option_extra") if isinstance(info, dict) else None
            if extra is None:
                continue
            acc = self._acc["dry" if extra.get("dry") else "fresh"]
            acc["n"] += 1
            acc["wage_sum"] += float(extra["W"])
            acc["tau_sum"] += float(extra["tau"])
            acc["depth_sum"] += float(extra["dlvl_end"])
        if self.num_timesteps >= self.next_at:
            while self.num_timesteps >= self.next_at:
                self.next_at += self.every
            self._emit(final=False)
        return True

    def _dry_state_readout(self) -> tuple[float, list[int]]:
        import numpy as np
        import torch as th

        with th.no_grad():
            obs = th.as_tensor(self.X_dry, device=self.model.device)
            masks = _probe_legacy_masks(obs)
            dist = self.model.policy.get_distribution(obs, action_masks=masks)
            entropy = float(dist.distribution.entropy().mean())
            pred = dist.distribution.logits.argmax(-1).cpu().numpy()
        hist = np.bincount(pred, minlength=15)
        return entropy, [int(count) for count in hist]

    @staticmethod
    def _window_summary(acc: dict) -> dict:
        n = acc["n"]
        mean = (lambda total: round(total / n, 4) if n else None)
        return {"n": int(n), "wage_mean": mean(acc["wage_sum"]),
                "tau_mean": mean(acc["tau_sum"]),
                "depth_mean": mean(acc["depth_sum"])}

    def _emit(self, final: bool) -> None:
        step = int(self.num_timesteps)
        if self._last_emit_step == step:
            return
        entropy, hist = self._dry_state_readout()
        line = {"metrics": "drywin", "step": step,
                "dry_state_entropy": round(entropy, 6),
                "dry_state_n": int(len(self.X_dry)),
                "dry_state_argmax_hist": hist,
                "windows": {group: self._window_summary(self._acc[group])
                            for group in ("dry", "fresh")},
                "mask_mode": "dry-anchor-legacy",
                "demos_sha16": self.demos_sha256[:16]}
        if final:
            line["final"] = True
        with open(self.run_dir / "drywin_metrics.jsonl", "a") as f:
            f.write(json.dumps(line) + "\n")
        self._acc = self._fresh_acc()
        self._last_emit_step = step
        print(f"   [干窗行为] {line}")

    def _on_training_end(self) -> None:
        if self.num_timesteps > 0 and self._last_emit_step != int(self.num_timesteps):
            self._emit(final=True)


# E5③ 金丝雀 a12/局 中期仪表(检查点离线序列用;可独立调用的统计函数+记录器,
# 不挂训练回调——训练路径零接触,RC.11 逐点如实登记,只记不裁)。
_A12_CANARY_SCHEMA_VERSION = "a12-canary/1"
_A12_CANARY_STATS_KEYS = frozenset({
    "episodes", "a12_total", "a12_per_episode", "episodes_with_a12", "a12_max"})


def a12_canary_stats(a12_counts) -> dict:
    """E5③ 统计件:逐局 a12 实饮计数序列 → a12/局 读数(驱动器自检查点评测
    档案逐局提取后调用)。空序列/负数/非整数 fail-loud(零局之 a12/局 无定义,
    禁静默记 0 冒充实测)。"""
    counts = list(a12_counts)
    _require(len(counts) > 0, "a12 金丝雀统计需 ≥1 局(空序列 fail-loud)")
    _require(all(_is_plain_int(count) for count in counts),
             "a12 逐局计数必须全为整数")
    _require(all(count >= 0 for count in counts), "a12 逐局计数不能为负")
    total = sum(counts)
    return {"episodes": len(counts),
            "a12_total": int(total),
            "a12_per_episode": round(total / len(counts), 6),
            "episodes_with_a12": sum(1 for count in counts if count > 0),
            "a12_max": int(max(counts))}


def record_a12_canary(out_path: str | pathlib.Path, *, checkpoint_step: int,
                      manager: str, stats: dict, tag: str | None = None) -> dict:
    """E5③ 记录器:a12 金丝雀读数落 jsonl 一行(台账词 A12_CANARY 之进程侧
    原料;schema 封闭,键集合精确等断言,fail-loud)。返回落笔行。"""
    _require(_is_plain_int(checkpoint_step) and checkpoint_step >= 0,
             "a12 金丝雀 checkpoint_step 必须是非负整数")
    _require(isinstance(manager, str) and bool(manager),
             "a12 金丝雀 manager 必须是非空字符串")
    _require(isinstance(stats, dict) and set(stats) == set(_A12_CANARY_STATS_KEYS),
             f"a12 金丝雀 stats 键集合必须精确等于 {sorted(_A12_CANARY_STATS_KEYS)}")
    line = {"canary": "a12", "schema_version": _A12_CANARY_SCHEMA_VERSION,
            "checkpoint_step": int(checkpoint_step), "manager": manager}
    if tag is not None:
        _require(isinstance(tag, str) and bool(tag),
                 "a12 金丝雀 tag 给定时必须是非空字符串")
        line["tag"] = tag
    line.update(stats)
    with open(out_path, "a") as f:
        f.write(json.dumps(line) + "\n")
    return line


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

    def _write_status(self, now: float, training_ended: bool = False) -> None:
        elapsed = now - self.t0
        target_steps = self.config.get("target_global_steps",
                                       self.config["total_steps"])
        rollout_full = bool(getattr(
            getattr(getattr(self, "model", None), "rollout_buffer", None),
            "full", False))
        status = {
            "run": self.run_dir.name,
            "total_steps": int(self.num_timesteps),
            "target_steps": target_steps,
            "start_steps": self.config.get("start_steps", 0),
            "leg_steps": int(self.num_timesteps
                             - self.config.get("start_steps", 0)),
            "leg_target_steps": self.config["total_steps"],
            "episodes": self.ep_count,
            "sps": round((self.num_timesteps - self._steps0) / max(1e-9, elapsed)),
            "elapsed_sec": round(elapsed),
            "updated_at": now,
            "training_ended": training_ended,
            "rollout_full": rollout_full,
            "target_reached": int(self.num_timesteps) >= int(target_steps),
            "config": self.config,
        }
        # dashboard 轮询不应读到半截 JSON。
        tmp = self.run_dir / "status.tmp.json"
        tmp.write_text(json.dumps(status, ensure_ascii=False))
        tmp.replace(self.run_dir / "status.json")

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
            self._write_status(now)
        return True

    def _on_training_end(self) -> None:
        # 短训练或早停可在 1s 刷新窗内结束；若不强制落盘，
        # 驱动会把完整腿误判为少训了数步。
        self._write_status(time.time(), training_ended=True)
        self.close()

    def close(self) -> None:
        """异常路径也能幂等关闭逐局日志文件。"""
        if not self._progress.closed:
            self._progress.close()


def _main(resources: _TrainingResources):
    ap = argparse.ArgumentParser()
    ap.add_argument("--total-steps", type=int, default=1_998_848,
                    help="新增样本数；必须整除 n_steps×num_envs，禁止 SB3 静默超采")
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
    ap.add_argument("--dry-curriculum-schedule", default=None,
                    help="E1 ⑤A 干窗课程退火表:逗号分隔段,"
                         "'linear:<p0>:<p1>:<n>'(n≥2,端点含线性)或 'hold:<p>:<n>';"
                         "p∈[0,1] 为干层复访窗的脚本代跑概率,按腿相对 rollout 序号"
                         " (num_timesteps−3497984)/2048 取表,逐 rollout 于采集前推送;"
                         "与 --skip-dry 互斥,仅 --worker。主表 = "
                         "linear:1.0:0.5:147,hold:0.5:97")
    ap.add_argument("--no-drink-sovereignty", action="store_true",
                    help="v32 对照腿:关闭工人喝药主权(m[12] 恢复恒掩);"
                         "默认开 = ④丙 新协议常态,0.5 反射恒兜底")
    ap.add_argument("--ent-coef", type=float, default=0.02,
                    help="熵系数(v22 恶魔臂微调用 0.005 防 BC 漂移)")
    ap.add_argument("--bc-init", default=None,
                    help="行为克隆热启动:载入策略头 state_dict 路径")
    ap.add_argument("--init-source", choices=["bc", "checkpoint"], default="bc",
                    help="--bc-init 的来源类型；checkpoint 必须带 export_manager_sd 清单")
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
    ap.add_argument("--bc-aux-lambda", type=float, default=0.0,
                    help="E3 ④乙:辅助示范 CE 系数 λ_bc(正样本注入,主案限 12 类"
                         "示范对,不与 KING 锚对拉;主案冻结常量 0.015625,D7)。"
                         "须与 --bc-aux-demos 同在方在位;两旗互不强制,"
                         "任一不在 → 零侵入(不加载不采样不进损失图)")
    ap.add_argument("--bc-aux-demos", default=None,
                    help="E3 ④乙:bc-worker-v2 示范集 demos.npz 路径(v2 schema="
                         "X/Y/episode_id+逐样本 masks,专用验证器;v1 canonical"
                         " 路径 runs/bc-worker 分毫不动)")
    ap.add_argument("--resume-from", default=None,
                    help="v24 分腿续训:上一腿 model_final.zip 路径(禁与 --bc-init/--freeze 同用)")
    ap.add_argument("--calib-probes", default="",
                    help="v24 G-CAL 探针全局步(逗号分隔,只在腿 1 传 300000,600000)")
    ap.add_argument("--calib-record-only", action="store_true",
                    help="v28:G-CAL 只记不裁——tripped 位照写 calib.jsonl,旗不武装"
                         "(续航起点分歧 41.5%%,20%% 阈值对定居点失义;面板修正)")
    ap.add_argument("--teacher-override", default=None,
                    help="v30 锚随王走:resume 时以此 sd 覆写 zip 驮带的 teacher_path"
                         "(经 load kwargs 注入,_setup_model 一次建对;仅 resume 分支有效)")
    ap.add_argument("--allow-manager-change", action="store_true",
                    help="显式允许 worker resume 更换 manager_npz；默认契约禁止")
    ap.add_argument("--allow-legacy-resume", action="store_true",
                    help="一次性迁移无 training_contract 的旧 checkpoint；默认拒绝")
    # B1-E0 仪表旋钮(封闭枚举三枚,PREREG-B1;皆纯读+IO,不触 RNG/梯度/env 流/
    # 掩码/契约字段;默认值逐字承继原写死常量,缺省行为零漂移,W-G0 实弹钉死)
    ap.add_argument("--ckpt-every-steps", type=int, default=250_000,
                    help="B1-E0:暴露 AtomicRolloutCheckpointCallback.every_steps"
                         "(全局步;量子对齐与拒发半更新 ckpt 由回调原逻辑保证)")
    ap.add_argument("--sentinel-every", type=int, default=500_000,
                    help="B1-E0:WorkerSentinelCallback 汇总间隔(全局步,纯读+IO)")
    ap.add_argument("--dry-anchor-every", type=int, default=500_000,
                    help="B1-E0:DryAnchorSentinel 间隔(全局步;自有 rng(26),"
                         "不碰训练 RNG)")
    # E5 仪表旋钮(PREREG-内容案 E5,封闭枚举两枚;皆纯读+IO,只记不裁,
    # 默认 0 = 不在位 = 代码路径与 HEAD 等价,G0-2a 先决;探针示范集钉
    # BC-v1 demos 字节,E6)
    ap.add_argument("--distill-ce-probe-every", type=int, default=0,
                    help="E5①:干/鲜 distill_ce 分列离线探针间隔(全局步;"
                         "0=不在位;固定示范态集按 x[:,297] 干旗分组,专用 "
                         "rng,零触训练路径;输出 distill_ce_probe.jsonl)")
    ap.add_argument("--drywin-metrics-every", type=int, default=0,
                    help="E5②:干窗行为仪表间隔(全局步;0=不在位;干态动作"
                         "分布熵/a 分布 + 干/鲜窗工资与宽度 τ̄/depth,只记 "
                         "drywin_metrics.jsonl)")
    args = ap.parse_args()

    try:
        _validate_runtime_versions()
        _validate_args(args)
    except ValueError as exc:
        ap.error(str(exc))

    run_name = args.run_name or (
        time.strftime("ppo-l1-%m%d-%H%M%S")
        + f"-{os.getpid()}-{time.time_ns() % 1_000_000_000:09d}")
    run_dir = pathlib.Path(__file__).resolve().parent / "runs" / run_name
    try:
        run_lock = _RunLock(run_dir)
    except RuntimeError as exc:
        ap.error(str(exc))
    resources.run_lock = run_lock
    protected_inputs = [args.bc_init, args.teacher_override]
    if args.worker:
        protected_inputs.append(args.manager_npz)
        if args.distill_beta > 0 and not args.resume_from:
            protected_inputs.append(args.teacher_sd)
    if args.options and args.worker_npz:
        protected_inputs.append(args.worker_npz)
    if _bc_aux_active(args):
        protected_inputs.append(args.bc_aux_demos)   # E3:v2 示范集同受保护
    _prepare_run_dir(run_dir, args.resume_from, protected_inputs)

    # Capture all externally supplied brains before any VecEnv/subprocess can
    # load them.  Children receive these exact expectations and parse from a
    # single read, so an atomic path replacement is fail-loud rather than a
    # silent mixed-policy run.
    manager_npz_sha256 = (_capture_file_sha256(args.manager_npz, "manager_npz")
                          if args.worker else None)
    worker_npz_sha256 = (_capture_file_sha256(args.worker_npz, "worker_npz")
                         if args.worker_npz else None)
    implementation_sha256 = _implementation_bundle_sha256()

    fresh_teacher_sha256 = None
    if args.worker and args.distill_beta > 0 and not args.resume_from:
        fresh_teacher_sha256 = _validate_bc_report(
            pathlib.Path(args.teacher_sd), "data_gate")["policy_sha256"]

    teacher_override_sha256 = None
    if args.teacher_override:
        teacher_override_sha256 = _validate_export_manifest(
            pathlib.Path(args.teacher_override))["artifact_sha256"]

    # E1 四门之 demos_sha256 捕获门:skip_dry ∨ schedule(谓词在助手内,断言原封)
    demos_sha256 = _capture_dry_window_demos_sha256(args)

    # E1 ⑤A:课程表在此解析一次,供 env 初值与课程回调共用(_validate_args 已验)。
    dry_curriculum_table = (
        _parse_dry_curriculum_schedule(args.dry_curriculum_schedule)
        if args.dry_curriculum_schedule else None)

    # E3 ④乙:在位方加载 v2 示范集并过 12 类主案过滤(_validate_args 已 fail-loud
    # 预验);不在位 → 零侵入,不加载(图纸字面)。
    bc_aux_bank = None
    bc_aux_demos_sha256 = None
    if _bc_aux_active(args):
        _aux_x, _aux_y, _aux_masks, bc_aux_demos_sha256 = (
            _load_bc_aux_demos_v2(args.bc_aux_demos))
        bc_aux_bank = _filter_bc_aux_demo_pairs(_aux_x, _aux_y, _aux_masks)
    elif args.bc_aux_lambda > 0:
        # 图纸字面:未给 --bc-aux-demos 即零侵入(两旗互不强制);
        # 如实打印防误配静默(施工裁量注记)。
        print("   [④乙] --bc-aux-lambda>0 但未给 --bc-aux-demos:"
              "辅助通路按 E3 零侵入条款不在位")

    resume_checkpoint_bytes = None
    resume_data = None
    resume_checkpoint_sha256 = None
    if args.resume_from and args.worker:
        (resume_checkpoint_bytes, resume_data,
         resume_checkpoint_sha256) = _capture_leashed_checkpoint(args.resume_from)
    elif args.resume_from:
        # v31 经理续训口:通用捕获(字节冻结 + 通用闸 + sha),类保真交由加载段
        _resume_path = _checkpoint_path(args.resume_from)
        try:
            resume_checkpoint_bytes = _resume_path.read_bytes()
        except OSError as exc:
            raise ValueError(f"resume 检查点不可读: {_resume_path}: {exc}") from exc
        resume_data = _validate_checkpoint_bytes(
            resume_checkpoint_bytes, str(_resume_path), False)
        resume_checkpoint_sha256 = hashlib.sha256(
            resume_checkpoint_bytes).hexdigest()

    batch_size = _select_batch_size(args.n_steps, args.num_envs)
    hierarchical = args.worker or args.options or args.flat_clock
    effective_deep = True if hierarchical else args.deep
    effective_death_ladder = True if hierarchical else args.death_ladder
    config = {
        "total_steps": args.total_steps,
        "num_envs": args.num_envs,
        "device": args.device,
        "lr": args.lr,
        "n_steps": args.n_steps,
        "batch_size": batch_size,
        "max_steps": args.max_steps,
        "seed": args.seed,
        "algo": ({"rppo": "RecurrentPPO/MlpLstmPolicy",
                  "mppo": "MaskablePPO/MlpPolicy(gear-key mask)"}.get(args.algo, "PPO/MlpPolicy")
                 + ("+EntityAttention" if args.arch == "attn" else "")),
        "goal": ("深水区:层数递进奖金,活着往下潜(L3/L4)" if effective_deep
                 else "地牢 1 层:杀怪拿 XP,找楼梯下 2 层"),
        "deep": effective_deep,
        "death_ladder": effective_death_ladder,
        "gamma": args.gamma,
        "options": args.options,      # v22:True 时 Monitor ep_len 口径=策略脑决策数
        "flat_clock": args.flat_clock,
        "worker": args.worker,        # v23:True 时 ep 口径=FARM 窗口,reward=工资 w
        "skip_dry": args.skip_dry,
        "drink_sovereignty": not args.no_drink_sovereignty,   # v32 ④丙
        # E4 rev5 双键(契约与 config 回执同构增键;skip_dry 键仍 CLI 旗
        # 字面值,机制在位状态由此二键承载,rev3 勘正)
        "dry_curriculum": _contract_dry_curriculum(args),
        "bc_aux": _contract_bc_aux(args, bc_aux_demos_sha256),

        "bc_init": args.bc_init,
        "init_source": args.init_source,
        "ent_coef": args.ent_coef,
        "freeze_policy_steps": args.freeze_policy_steps,
        "distill_beta": args.distill_beta,    # v24 皮筋
        "resume_from": args.resume_from,
        "resume_checkpoint_sha256": resume_checkpoint_sha256,
        "worker_npz": args.worker_npz,        # v25 换届:经理训练挂 npz 工人
        # v30 接力:自证据链——本腿在谁治下、拴谁的锚,进程侧留回执(面板 minor)
        "manager_npz": args.manager_npz,
        "manager_npz_sha16": (manager_npz_sha256[:16]
                               if manager_npz_sha256 else None),
        "teacher_override": args.teacher_override,
        "teacher_override_sha16": (teacher_override_sha256[:16]
                                    if teacher_override_sha256 else None),
        "demos_sha16": demos_sha256[:16] if demos_sha256 else None,
        "implementation_sha16": implementation_sha256[:16],
        "allow_manager_change": args.allow_manager_change,
        "allow_legacy_resume": args.allow_legacy_resume,
        # B1-E0 仪表旋钮回执(只读遥测,不入 training_contract,契约零触碰)
        "ckpt_every_steps": args.ckpt_every_steps,
        "sentinel_every": args.sentinel_every,
        "dry_anchor_every": args.dry_anchor_every,
        # E5 仪表旋钮回执(同上:只读遥测,不入 training_contract)
        "distill_ce_probe_every": args.distill_ce_probe_every,
        "drywin_metrics_every": args.drywin_metrics_every,
    }
    print(f"== DiabloGym PPO 训练 == run={run_name}")
    print(f"   {config}")

    env_fn = functools.partial(
        make_env,
        max_steps=args.max_steps,
        deep=args.deep,
        death_ladder=args.death_ladder,
        options=args.options,
        flat_clock=args.flat_clock,
        worker=args.worker,
        manager_npz=args.manager_npz,
        worker_npz=args.worker_npz,
        # E1:课程腿 env p 初值 = 表首项——SB3 _setup_learn 之 env.reset() 先于
        # 首个 _on_rollout_start 推送发生,初值不对齐将使腿首窗口选择偏离
        # p≡1.0 端点恒等(G0-1/G0-2a);无课程时保持 CLI 旗原语义(bool→1.0/0.0)。
        skip_dry=(dry_curriculum_table[0] if dry_curriculum_table
                  else args.skip_dry),
        drink_sovereignty=not args.no_drink_sovereignty,
        manager_npz_sha256=manager_npz_sha256,
        worker_npz_sha256=worker_npz_sha256,
        implementation_sha256=implementation_sha256,
    )
    if args.num_envs == 1:
        vec_env = DummyVecEnv([env_fn])
    else:
        vec_env = SubprocVecEnv([env_fn] * args.num_envs, start_method="spawn")
    resources.vec_env = vec_env

    common = dict(
        learning_rate=args.lr,
        gamma=args.gamma,
        ent_coef=args.ent_coef,  # 默认 0.02(首训 0.01 曾面壁塌缩);v22 恶魔臂 0.005
        gae_lambda=_ALGORITHM_RECIPE["gae_lambda"],
        n_epochs=_ALGORITHM_RECIPE["n_epochs"],
        clip_range=_ALGORITHM_RECIPE["clip_range"],
        clip_range_vf=_ALGORITHM_RECIPE["clip_range_vf"],
        vf_coef=_ALGORITHM_RECIPE["vf_coef"],
        max_grad_norm=_ALGORITHM_RECIPE["max_grad_norm"],
        normalize_advantage=_ALGORITHM_RECIPE["normalize_advantage"],
        target_kl=_ALGORITHM_RECIPE["target_kl"],
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
            n_steps=args.n_steps, batch_size=batch_size,
            policy_kwargs=dict(lstm_hidden_size=128, n_lstm_layers=1, **policy_kwargs),
            use_sde=_ALGORITHM_RECIPE["use_sde"],
            sde_sample_freq=_ALGORITHM_RECIPE["sde_sample_freq"],
            **common,
        )
    elif args.algo == "mppo":
        # v16:掩码采样与掩码更新都由 MaskablePPO 处理;掩码本身来自
        # env.action_masks()(经 VecEnv.env_method 收集)。注意这是算法实现的
        # 整体更换,开牌异常时首要嫌疑人(诚实账本已记)。
        # v24:worker 路一律走 LeashedMaskablePPO(β=0 时 G-KL-B 证与原版逐位等价)
        calib = [int(x) for x in args.calib_probes.split(",") if x.strip()]
        if args.resume_from and args.options:
            # v31 经理续训口:类保真(存什么类续什么类,M29 系平 MaskablePPO;
            # 不涉教师/β,无 G-KL-B 义务);封条断言照 v24 原封。
            from sb3_contrib import MaskablePPO
            _require(resume_checkpoint_bytes is not None and resume_data is not None,
                     "resume checkpoint 捕获状态缺失")
            _load_kw = {"seed": args.seed} if args.seed is not None else {}
            model = MaskablePPO.load(io.BytesIO(resume_checkpoint_bytes), env=vec_env,
                                     device=args.device, **_load_kw)
            model.tensorboard_log = str(run_dir / "tb")
            saved_lr = model.learning_rate
            _require(not callable(saved_lr) and math.isclose(float(saved_lr), args.lr,
                                                              rel_tol=0, abs_tol=1e-12),
                     f"resume 学习率不符: checkpoint={saved_lr}, CLI={args.lr}")
            _require(math.isclose(float(model.ent_coef), args.ent_coef,
                                  rel_tol=0, abs_tol=1e-12)
                     and math.isclose(float(model.gamma), args.gamma,
                                      rel_tol=0, abs_tol=1e-12)
                     and model.n_steps == args.n_steps
                     and model.batch_size == batch_size,
                     "PREREG-v24 封-5:resume 腿超参与冻结配方不符")
            _require(model.target_kl is None, "PREREG-v24 D4:target_kl 必须为 None")
            if args.seed is not None:
                model.set_random_seed(args.seed)
                model.seed = args.seed  # set_random_seed 不更新持久化属性,防 zip 续写旧 seed
            print(f"   [v31] options resume @ {model.num_timesteps} 步(经理续训口)")
        elif args.resume_from:
            from leashed_ppo import LeashedMaskablePPO
            _require(resume_checkpoint_bytes is not None and resume_data is not None,
                     "resume checkpoint 捕获状态缺失")
            _load_kw = {"seed": args.seed} if args.seed is not None else {}
            if args.distill_beta == 0:
                # β=0 不消费教师；旧 zip 中的绝对路径不应让可用检查点因搬家而失效。
                _load_kw.update(teacher_path=None, teacher_sha256=None)
            elif args.teacher_override:
                # v30 锚随王走:kwargs 在 zip data 之后、_setup_model 之前生效,
                # 教师一次建对(post-load 重建系次优解,面板 major 裁定弃用)
                _load_kw["teacher_path"] = args.teacher_override
                _load_kw["teacher_sha256"] = teacher_override_sha256
            else:
                saved_teacher = resume_data.get("teacher_path")
                _require(isinstance(saved_teacher, str) and saved_teacher,
                         "β>0 resume 检查点没有 teacher_path；请显式 --teacher-override")
                teacher_report = _validate_bc_report(
                    pathlib.Path(saved_teacher), "data_gate")
                current_teacher_sha = teacher_report["policy_sha256"]
                saved_teacher_sha = resume_data.get("teacher_sha256")
                if saved_teacher_sha is not None:
                    _require(saved_teacher_sha == current_teacher_sha,
                             "检查点教师 SHA 与当前 BC 报告不一致；"
                             "换锚必须显式 --teacher-override")
                    _load_kw["teacher_sha256"] = saved_teacher_sha
                else:
                    # 旧检查点没有 SHA 字段：只允许以当前 PASS+绑定报告做一次 TOFU 迁移。
                    _load_kw["teacher_sha256"] = current_teacher_sha
            model = LeashedMaskablePPO.load(io.BytesIO(resume_checkpoint_bytes), env=vec_env,
                                            device=args.device, **_load_kw)
            if getattr(model, "teacher_path", None) and not args.teacher_override:
                _validate_bc_report(pathlib.Path(model.teacher_path), "data_gate")
            # PREREG-v24 D4:β 显式覆盖(load 直写 __dict__ 无校验,不许静默续命);
            # tb 路径同理(否则腿 2-8 曲线全写进腿 1 目录);旋钮封条断言。
            _require(hasattr(model, "distill_beta"),
                     "LeashedMaskablePPO.load 后缺少 distill_beta 内部属性")
            model.distill_beta = args.distill_beta
            model.calib_probes, model.calib_out = calib, (
                str(run_dir / "calib.jsonl") if calib else None)
            model.calib_record_only = args.calib_record_only
            model.tensorboard_log = str(run_dir / "tb")
            saved_lr = model.learning_rate
            _require(not callable(saved_lr) and math.isclose(float(saved_lr), args.lr,
                                                              rel_tol=0, abs_tol=1e-12),
                     f"resume 学习率不符: checkpoint={saved_lr}, CLI={args.lr}")
            _require(math.isclose(float(model.ent_coef), args.ent_coef,
                                  rel_tol=0, abs_tol=1e-12)
                     and math.isclose(float(model.gamma), args.gamma,
                                      rel_tol=0, abs_tol=1e-12)
                     and model.n_steps == args.n_steps
                     and model.batch_size == batch_size,
                     "PREREG-v24 封-5:resume 腿超参与冻结配方不符")
            _require(model.target_kl is None, "PREREG-v24 D4:target_kl 必须为 None")
            if args.teacher_override:
                # v30 身份链断言(面板 blocker:闸过的文件与训练吃进的文件必须同一)
                _require(model.teacher_path == args.teacher_override, "教师覆写未生效")
                _require(model.teacher[0].in_features == 298
                         and model.teacher[-1].out_features == 15,
                         "自锚教师形状异常(须 298→15 工人网)")
            if args.distill_beta > 0:
                _require(model.teacher is not None, "β>0 但教师未随 teacher_path 重建")
            if args.seed is not None:
                model.set_random_seed(args.seed)
                model.seed = args.seed  # set_random_seed 不会更新持久化属性，防 zip 继续写腿1 seed
            print(f"   [v24] resume @ {model.num_timesteps} 步,β={model.distill_beta}")
        elif args.worker:
            from leashed_ppo import LeashedMaskablePPO
            model = LeashedMaskablePPO(
                "MlpPolicy", vec_env, n_steps=args.n_steps, batch_size=batch_size,
                policy_kwargs=policy_kwargs or None,
                distill_beta=args.distill_beta,
                teacher_path=args.teacher_sd if args.distill_beta > 0 else None,
                teacher_sha256=fresh_teacher_sha256,
                calib_probes=calib,
                calib_out=str(run_dir / "calib.jsonl") if calib else None,
                **common)
            model.calib_record_only = args.calib_record_only
        else:
            from sb3_contrib import MaskablePPO
            model = MaskablePPO("MlpPolicy", vec_env, n_steps=args.n_steps,
                                batch_size=batch_size,
                                policy_kwargs=policy_kwargs or None, **common)
    else:
        model = PPO("MlpPolicy", vec_env, n_steps=args.n_steps, batch_size=batch_size,
                    policy_kwargs=policy_kwargs or None,
                    use_sde=_ALGORITHM_RECIPE["use_sde"],
                    sde_sample_freq=_ALGORITHM_RECIPE["sde_sample_freq"],
                    **common)

    # E3 ④乙:辅助通路挂载与 λ 显式覆盖(resume/fresh 两路同一插点;承 β 覆盖
    # 先例:load 直写 __dict__ 无校验,不许 zip 驮值静默续命。bank 不在位 →
    # λ 显式归零,train() 辅助段整段不进图(零侵入))。
    if bc_aux_bank is not None:
        from leashed_ppo import derive_bc_aux_rng
        _require(hasattr(model, "bc_aux_lambda"),
                 "④乙辅助通路要求 LeashedMaskablePPO(--worker --algo mppo)")
        model.mount_bc_aux_demos(*bc_aux_bank, rng=derive_bc_aux_rng(args.seed))
        model.bc_aux_lambda = args.bc_aux_lambda
        print(f"   [④乙] 辅助示范通路在位: λ_bc={model.bc_aux_lambda},"
              f" 12类示范对 n={len(bc_aux_bank[1])},"
              f" demos_sha16={bc_aux_demos_sha256[:16]}")
    elif hasattr(model, "bc_aux_lambda"):
        model.bc_aux_lambda = 0.0

    _validate_model_recipe(model)
    current_contract = _training_contract(
        args, model, batch_size,
        manager_npz_sha256=manager_npz_sha256,
        worker_npz_sha256=worker_npz_sha256,
        demos_sha256=demos_sha256,
        implementation_sha256=implementation_sha256,
        bc_aux_demos_sha256=bc_aux_demos_sha256,   # E4 rev5:④乙在位载荷
    )
    if args.resume_from:
        _validate_resume_contract(
            getattr(model, "diablogym_contract", None), current_contract,
            allow_manager_change=args.allow_manager_change,
            allow_legacy_resume=args.allow_legacy_resume)
    model.diablogym_contract = current_contract
    config["training_contract"] = current_contract
    config["teacher_sha256"] = getattr(model, "teacher_sha256", None)

    # status 的 total_steps 是 SB3 全局步；target_steps 也必须同口径。
    config["start_steps"] = int(model.num_timesteps)
    config["target_global_steps"] = int(model.num_timesteps + args.total_steps)

    if args.bc_init:
        # v22 恶魔臂:BC 热启动策略头;冻结期只训价值头(经典雷:新价值头的
        # 首次 PPO 更新会摧毁 BC 策略,先冻结抗住)
        gate = ("data_gate" if args.worker else "hypothesis" if args.options
                else "memoryless_hypothesis")
        sd = _load_bc_state_dict(args.bc_init, model.policy, gate, args.init_source)
        missing, unexpected = model.policy.load_state_dict(
            sd, strict=args.init_source == "checkpoint")
        _require(not unexpected, f"BC state_dict 含未知键: {unexpected}")
        _require(all(k not in missing for k in _POLICY_HEAD_KEYS),
                 f"BC 策略头未完整加载: {missing}")
        print(f"   BC 热启动:loaded(missing={len(missing)}, unexpected={len(unexpected)})")
        if args.freeze_policy_steps > 0:
            from stable_baselines3.common.callbacks import BaseCallback

            pi_params = (list(model.policy.mlp_extractor.policy_net.parameters())
                         + list(model.policy.action_net.parameters()))
            if getattr(model.policy, "share_features_extractor", False):
                pi_params += list(model.policy.features_extractor.parameters())
            # 共享特征提取器若仍被 value loss 更新，即使头部 requires_grad=False，
            # BC 策略输出也会在“冻结”期漂移。去重后一并冻结。
            pi_params = list({id(p): p for p in pi_params}.values())
            for p in pi_params:
                p.requires_grad = False

            class _Unfreeze(BaseCallback):
                def __init__(self, when):
                    super().__init__()
                    self.when, self.done_ = when, False

                def _on_rollout_start(self):
                    # PPO 在 rollout 收完后才统一更新。若在跨过阈值的
                    # _on_step 中解冻，该整批(包含阈值前样本)都会更新
                    # 策略。只在下一个 rollout 起点解冻，硬保证前
                    # freeze_policy_steps 个样本只训价值头。
                    if not self.done_ and self.num_timesteps >= self.when:
                        for p in pi_params:
                            p.requires_grad = True
                        self.done_ = True
                        print(f"   策略头解冻 @ {self.num_timesteps}")

                def _on_step(self):
                    return True

            unfreeze_cb = _Unfreeze(args.freeze_policy_steps)
        else:
            unfreeze_cb = None
    else:
        unfreeze_cb = None

    # 每 ~25 万个已完成更新的样本存一次原子检查点；499,712 步腿至少有中点保护。
    # B1-E0:三处间隔改由 CLI 旋钮供值(默认逐字承旧常量,缺省行为零漂移)。
    ckpt = AtomicRolloutCheckpointCallback(
        run_dir, every_steps=args.ckpt_every_steps,
        implementation_sha256=implementation_sha256)
    sentinel_cb = (WorkerSentinelCallback(run_dir, every=args.sentinel_every)
                   if args.worker else None)
    # E1 四门之 dry_cb 挂载门:worker ∧ (skip_dry ∨ schedule)(谓词在助手内)
    dry_cb = (DryAnchorSentinel(run_dir, str(pathlib.Path(__file__).resolve().parent
                                             / "runs" / "bc-worker" / "demos.npz"),
                                  demos_sha256, every=args.dry_anchor_every)
              if _mount_dry_anchor_sentinel(args) else None)
    # E1 ⑤A 课程回调(schedule 仅 --worker,_validate_args 已断言)
    curriculum_cb = (DryCurriculumCallback(dry_curriculum_table, run_dir=run_dir)
                     if (args.worker and dry_curriculum_table) else None)
    # E5 仪表挂载(只记不裁;旋钮 0 = 不挂载 = 回调列与 HEAD 等价,G0-2a
    # 先决;探针示范集一律钉 canonical BC-v1 demos 字节,E6 构造内断言)
    _probe_demos = str(pathlib.Path(__file__).resolve().parent
                       / "runs" / "bc-worker" / "demos.npz")
    distill_ce_cb = (DistillCeProbe(run_dir, _probe_demos,
                                    every=args.distill_ce_probe_every)
                     if (args.worker and args.distill_ce_probe_every > 0)
                     else None)
    if distill_ce_cb is not None:
        _require(getattr(model, "teacher", None) is not None,
                 "--distill-ce-probe-every>0 需 Leashed 教师在位"
                 "(β>0 或 teacher_path;fail-loud 于点火前)")
    drywin_cb = (DryWindowMetricsCallback(run_dir, _probe_demos,
                                          every=args.drywin_metrics_every)
                 if (args.worker and args.drywin_metrics_every > 0) else None)
    # 让唯一持有文件句柄的 callback 最后构造；其后的 setup 不再有可失败 I/O。
    callback = EpisodeJsonlCallback(run_dir, config)
    learn_completed = False
    try:
        # E1 回调序钉死:课程回调居列首——p 于 _on_rollout_start 采集开始前、
        # 先于一切其余 rollout-start 副作用推送在位(CallbackList 按列序分发)。
        cbs = (([curriculum_cb] if curriculum_cb else [])
               + [callback, ckpt] + ([unfreeze_cb] if unfreeze_cb else [])
               + ([sentinel_cb] if sentinel_cb else [])
               + ([dry_cb] if dry_cb else [])
               # E5 仪表居列尾(纯读+IO;不在位时本两项为空,列与 HEAD 等价)
               + ([distill_ce_cb] if distill_ce_cb else [])
               + ([drywin_cb] if drywin_cb else []))
        # v24:resume 腿 reset_num_timesteps=False(False 语义 = 再训 N 步,全局步连续
        # → ckpt 文件名全局唯一、β 日程与预算记账不断;审计 BLOCKER 2)
        model.learn(total_timesteps=args.total_steps, callback=cbs,
                    reset_num_timesteps=not args.resume_from)
        # collect_rollouts 被 callback 中途终止时 learn() 也会正常返回；此外
        # G-CAL 可在 full buffer 的首个 minibatch 拒绝整个更新。两者都不能
        # 把 num_timesteps 已记、梯度未吃的权重发布成正式终点。
        learn_completed = _is_publishable_rollout_boundary(model)
    finally:
        active_error = sys.exc_info()[0] is not None
        callback.close()
        save_error = None
        model_published = False
        if not active_error and learn_completed:
            try:
                final_implementation = _implementation_bundle_sha256()
                _require(final_implementation == implementation_sha256,
                         "训练期间实现/引擎/游戏内容发生漂移，拒绝发布 model_final: "
                         f"{final_implementation} != {implementation_sha256}")
                _atomic_save_model(model, run_dir / "model_final.zip")
                model_published = True
            except Exception as exc:
                # close 必须执行；原实现在 save 失败时会直接跳过子进程清理。
                save_error = exc
                print(f"模型保存失败: {exc}")
        else:
            # 异常或半 rollout 早停时，num_timesteps 已可能包含尚未更新的样本；
            # 这类权重不能冒充正式终点。最近的 rollout-boundary ckpt 仍可恢复。
            print("训练未停在完整更新边界，拒绝发布 model_final.zip")
        # 工作进程崩溃时 close 可能在断管上阻塞；资源所有者
        # 统一做超时清理，并恢复宿主的 SIGALRM handler/timer。
        resources.close()
        if model_published:
            print(f"模型已保存: {run_dir}/model_final.zip")
        elif save_error is not None and not active_error:
            raise save_error


def main():
    resources = _TrainingResources()
    try:
        return _main(resources)
    finally:
        # Covers every pre-learn failure too: model/load, contract, BC init,
        # and callback construction.  The normal learn-finally path is
        # idempotent and clears these handles before returning here.
        resources.close()


if __name__ == "__main__":
    main()

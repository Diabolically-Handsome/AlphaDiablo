"""把冻结 plain Worker(SB3 zip)导出成带部署合约的严格 NPZ。

新 checkpoint 从 training contract 推导 observation/action12 语义；未持久化
这些字段的历史 checkpoint 必须显式给 ``--observation-view`` 与
``--action12-mode``。输出含六张矩阵和唯一 ``worker_contract_json`` 成员，
并做 1000 个已选 policy-space 观测的 SB3↔NumPy argmax parity。
"""
import argparse
import hashlib
import io
import json
import os
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

DEF_ZIP = ROOT / "train" / "models" / "v24-worker-leg7" / "model.zip"


def _canonical_contract_sha256(contract) -> str | None:
    if contract is None:
        return None
    if not isinstance(contract, dict):
        raise ValueError("checkpoint 的 diablogym_contract 必须是 dict 或 null")
    try:
        payload = json.dumps(
            contract,
            # Must match train_ppo._canonical_json_sha256 exactly; contracts
            # contain human-readable non-ASCII fields such as the training goal.
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("checkpoint 的 diablogym_contract 不是严格 JSON") from exc
    return hashlib.sha256(payload).hexdigest()


def _resolve_declared_value(
        *,
        explicit: str | None,
        derived: str | None,
        label: str) -> str:
    if explicit is not None and derived is not None and explicit != derived:
        raise ValueError(
            f"{label} 与 checkpoint training contract 冲突:"
            f"{explicit!r} != {derived!r}")
    selected = explicit if explicit is not None else derived
    if selected is None:
        raise ValueError(
            f"checkpoint 未携可推导的 {label}；旧 checkpoint 必须显式传 --{label.replace('_', '-')}")
    return selected


def _worker_contract_from_checkpoint(
        model,
        *,
        observation_view: str | None,
        action12_mode: str | None,
        source_checkpoint_sha256: str) -> dict:
    from diablogym.worker_env import (
        WORKER_ACTION12_ENVIRONMENT_MASK,
        WORKER_ACTION12_PERMANENTLY_MASKED,
        WORKER_OBSERVATION_VIEW_LEGACY_V3,
        WORKER_OBSERVATION_VIEW_RAW_V4,
        make_worker_npz_contract,
    )

    training_contract = getattr(model, "diablogym_contract", None)
    training_contract_sha256 = _canonical_contract_sha256(training_contract)
    derived_view = None
    derived_action12 = None
    if training_contract is not None:
        if "legacy_policy_observation_view" in training_contract:
            legacy_view = training_contract["legacy_policy_observation_view"]
            if not isinstance(legacy_view, bool):
                raise ValueError(
                    "training contract.legacy_policy_observation_view 必须是 bool")
            derived_view = (
                WORKER_OBSERVATION_VIEW_LEGACY_V3
                if legacy_view
                else WORKER_OBSERVATION_VIEW_RAW_V4
            )
        if "drink_sovereignty" in training_contract:
            sovereignty = training_contract["drink_sovereignty"]
            if not isinstance(sovereignty, bool):
                raise ValueError(
                    "training contract.drink_sovereignty 必须是 bool")
            derived_action12 = (
                WORKER_ACTION12_ENVIRONMENT_MASK
                if sovereignty
                else WORKER_ACTION12_PERMANENTLY_MASKED
            )
    return make_worker_npz_contract(
        observation_view=_resolve_declared_value(
            explicit=observation_view,
            derived=derived_view,
            label="observation_view",
        ),
        action12_mode=_resolve_declared_value(
            explicit=action12_mode,
            derived=derived_action12,
            label="action12_mode",
        ),
        source_checkpoint_sha256=source_checkpoint_sha256,
        source_training_contract_sha256=training_contract_sha256,
    )


def _plain_policy_arrays(model) -> dict[str, np.ndarray]:
    """Extract only the architecture that NumpyManager actually implements."""
    import torch as th
    from stable_baselines3.common.torch_layers import FlattenExtractor

    policy = model.policy
    policy_name = type(policy).__name__
    if (
        policy_name == "A12MixtureMaskableActorCriticPolicy"
        or policy_name == "AsymmetricWorkerMaskableActorCriticPolicy"
        or getattr(policy, "_bc_aux_circuit_spec", None) is not None
        or getattr(
            getattr(policy, "mlp_extractor", None),
            "context_adapter",
            None,
        ) is not None
        or hasattr(policy, "a12_gate")
    ):
        raise ValueError(
            "A12/BC-aux/asymmetric policy 不能导出为 plain six-matrix Worker NPZ")
    if tuple(getattr(model.observation_space, "shape", ())) != (298,):
        raise ValueError(
            "plain six-matrix Worker NPZ 只支持 298 维策略观测")
    if not isinstance(getattr(policy, "features_extractor", None), FlattenExtractor):
        raise ValueError("Worker NPZ 只支持 FlattenExtractor")
    layers = list(getattr(policy.mlp_extractor, "policy_net", ()))
    if (
        len(layers) != 4
        or not isinstance(layers[0], th.nn.Linear)
        or not isinstance(layers[1], th.nn.Tanh)
        or not isinstance(layers[2], th.nn.Linear)
        or not isinstance(layers[3], th.nn.Tanh)
        or not isinstance(getattr(policy, "action_net", None), th.nn.Linear)
    ):
        raise ValueError(
            "Worker NPZ 只支持 Linear-Tanh-Linear-Tanh + Linear action head")
    sd = policy.state_dict()
    keys = (
        "mlp_extractor.policy_net.0.weight",
        "mlp_extractor.policy_net.0.bias",
        "mlp_extractor.policy_net.2.weight",
        "mlp_extractor.policy_net.2.bias",
        "action_net.weight",
        "action_net.bias",
    )
    missing = [key for key in keys if key not in sd]
    if missing:
        raise ValueError(f"plain Worker policy 缺少权重:{missing}")
    arrays = {
        "w0": sd[keys[0]].detach().cpu().numpy(),
        "b0": sd[keys[1]].detach().cpu().numpy(),
        "w1": sd[keys[2]].detach().cpu().numpy(),
        "b1": sd[keys[3]].detach().cpu().numpy(),
        "wa": sd[keys[4]].detach().cpu().numpy(),
        "ba": sd[keys[5]].detach().cpu().numpy(),
    }
    if not all(np.isfinite(array).all() for array in arrays.values()):
        raise ValueError("策略权重含 NaN/Inf，拒绝导出")
    return arrays


def main():
    from sb3_contrib import MaskablePPO

    from diablogym.worker_env import (
        NumpyManager,
        WORKER_ACTION12_ENVIRONMENT_MASK,
        WORKER_ACTION12_PERMANENTLY_MASKED,
        WORKER_NPZ_CONTRACT_MEMBER,
        WORKER_OBSERVATION_VIEW_LEGACY_V3,
        WORKER_OBSERVATION_VIEW_RAW_V4,
        canonical_worker_npz_contract_json,
    )

    ap = argparse.ArgumentParser()
    ap.add_argument("zip_path", nargs="?", type=pathlib.Path, default=DEF_ZIP)
    ap.add_argument("output", nargs="?", type=pathlib.Path)
    ap.add_argument(
        "--observation-view",
        choices=(
            WORKER_OBSERVATION_VIEW_LEGACY_V3,
            WORKER_OBSERVATION_VIEW_RAW_V4,
        ),
        help=(
            "源 checkpoint 未持久化视图时必须显式声明；"
            "若 contract 已持久化则本值只能与之相同"),
    )
    ap.add_argument(
        "--action12-mode",
        choices=(
            WORKER_ACTION12_PERMANENTLY_MASKED,
            WORKER_ACTION12_ENVIRONMENT_MASK,
        ),
        help=(
            "源 checkpoint 未持久化喝药主权时必须显式声明；"
            "若 contract 已持久化则本值只能与之相同"),
    )
    args = ap.parse_args()
    zip_p = args.zip_path
    out = args.output or zip_p.parent / "policy.npz"
    source_file = zip_p if zip_p.suffix.lower() == ".zip" else pathlib.Path(f"{zip_p}.zip")
    if not source_file.is_file():
        ap.error(f"checkpoint 不存在: {source_file}")
    if out.suffix.lower() != ".npz":
        ap.error("输出路径必须以 .npz 结尾")
    if out.resolve() == source_file.resolve():
        ap.error("输出路径不能覆盖源 checkpoint")
    # Capture once so the loaded model and recorded source identity cannot be
    # split by a concurrent path replacement.
    source_payload = source_file.read_bytes()
    source_sha256 = hashlib.sha256(source_payload).hexdigest()
    model = MaskablePPO.load(io.BytesIO(source_payload), device="cpu")
    try:
        worker_contract = _worker_contract_from_checkpoint(
            model,
            observation_view=args.observation_view,
            action12_mode=args.action12_mode,
            source_checkpoint_sha256=source_sha256,
        )
        arrays = _plain_policy_arrays(model)
    except ValueError as exc:
        ap.error(str(exc))
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(f".{out.name}.{os.getpid()}.tmp.npz")
    arrays[WORKER_NPZ_CONTRACT_MEMBER] = np.asarray(
        canonical_worker_npz_contract_json(worker_contract))
    try:
        np.savez(tmp, **arrays)
        net = NumpyManager(str(tmp))
        net.require_worker_contract()
        rng = np.random.default_rng(0)
        mism = 0
        for _ in range(1000):
            # This is already-selected policy space.  Complete legacy-v3 rows
            # are generated at the environment boundary, never decoded here
            # from a filtered raw-v4 vector.
            policy_obs = rng.standard_normal(298).astype(np.float32)
            raw_mask = rng.random(15) > 0.2
            raw_mask[0] = True
            policy_mask = net.worker_mask(raw_mask)
            a_np = net.choose_worker(
                policy_obs,
                raw_mask,
                observation_view=net.worker_observation_view,
            )
            a_sb, _ = model.predict(
                policy_obs,
                action_masks=policy_mask,
                deterministic=True,
            )
            mism += int(a_np != int(a_sb))
        if mism:
            raise SystemExit("PARITY FAIL")
        os.replace(tmp, out)
    finally:
        tmp.unlink(missing_ok=True)
    print(
        f"npz 已存 {out};parity 失配 {mism}/1000;"
        f"view={worker_contract['observation_view']};"
        f"a12={worker_contract['action12_mode']};"
        f"源 checkpoint sha256:{source_sha256[:16]}")


if __name__ == "__main__":
    main()

"""32 种子标准评估（协议版本取自 eval_contract.PROTOCOL_VERSION）。

协议(全部条件都是结果的一部分,缺一不可比):
  - 种子集固定 9000-9031,只用于最终评估,永不参与训练/调参;
  - argmax 确定性策略,max_steps=1500,ticks_per_step=4;
  - 引擎源码钉死在 bootstrap.sh 的 ENGINE_REF(换引擎版本必须重建整张排行榜);
  - 空载机器上运行:引擎的回合推进读真实墙钟(nthread_has_500ms_passed),
    高负载下个别 tick 会少推一个逻辑回合导致轨迹漂移——2026-07-05 实测:
    空载下跨进程 4 次评估逐种子位级一致;训练同机并行时中位数曾漂过 0.5。

教训:8 种子的运气波动曾把 run6/run8 分别高估 77%/57%(15.6→8.8,13.2→8.4)。

用法(仓库根目录):
  .venv/bin/python train/evaluate.py train/runs/<run>/model_final
  自动识别 RecurrentPPO/MaskablePPO 与自定义特征提取器；结果写入当前
  版本 leaderboard-v<PROTOCOL_VERSION>.md，旧榜保留为只读历史。
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import os
import pathlib
import re
import statistics as s
import sys
import time
import zipfile
from collections.abc import Mapping

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from eval_contract import (PROTOCOL_VERSION, bridge_binary_path, exclusive_lock,
                           loaded_engine_binary_path, resolve_checkpoint_file,
                           runtime_identity, sha256_file, source_bundle_sha256,
                           strict_json_loads)

SEEDS = list(range(9000, 9032))
LEADERBOARD = (pathlib.Path(__file__).resolve().parent
               / f"leaderboard-v{PROTOCOL_VERSION}.md")
LEADERBOARD_LOCK = ROOT / "train" / "runs" / "eval-locks" / "leaderboard-main.lock"

MAIN_SOURCE_FILES = ("train/evaluate.py", "train/models.py")
MAIN_PROTOCOL = {
    "name": "diablogym.standalone.main",
    "seeds": SEEDS,
    "max_steps": 1500,
    "ticks_per_step": 4,
    "start_in_dungeon": True,
    "hero_class": 0,
    "include_raw": False,
    "descend_ladder": False,
    "death_ladder": False,
    "disable_level_backtracking": True,
    "action_selection": "deterministic_argmax; action masks for MaskablePPO",
}

LEADERBOARD_HEADER = (
    f"# Leaderboard protocol v{PROTOCOL_VERSION} — deterministic evaluation, "
    "32 fixed seeds\n\n"
    "Protocol: argmax policy, seeds 9000-9031 (never used for training or\n"
    "hyper-parameter selection), 1500 steps/episode, idle machine, engine\n"
    "pinned to `ENGINE_REF` in bootstrap.sh. See train/evaluate.py.\n\n"
    "| run | mean kills | median | max | zero-kill | reached L2 |\n"
    "|---|---|---|---|---|---|\n"
)

_CONTRACT_MARKER = "diablogym-standalone-contract-v1:"
_ROW_MARKER = "diablogym-standalone-row-v1:"
_ROW_MARKER_RE = re.compile(
    r"\s+<!-- diablogym-standalone-row-v1:([A-Za-z0-9_=\-]+) -->$")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


def _canonical_json(value) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False).encode("utf-8")


def _encode_marker(value) -> str:
    return base64.urlsafe_b64encode(_canonical_json(value)).decode("ascii")


def _decode_marker(payload: str):
    try:
        raw = base64.b64decode(payload.encode("ascii"), altchars=b"-_", validate=True)
    except (ValueError, UnicodeEncodeError) as exc:
        raise ValueError("排行榜 provenance marker 不是合法 base64url") from exc
    return strict_json_loads(raw)


def contract_sha256(contract: Mapping) -> str:
    return hashlib.sha256(_canonical_json(contract)).hexdigest()


def _standalone_sources(root: pathlib.Path,
                        source_files: tuple[str, ...]) -> dict:
    if not source_files or len(source_files) != len(set(source_files)):
        raise ValueError("standalone source_files 契约异常")
    files = {}
    for relative in source_files:
        path = pathlib.PurePosixPath(relative)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"standalone 源码路径非法:{relative!r}")
        source = root / relative
        if not source.is_file():
            raise ValueError(f"standalone 协议源码缺失:{source}")
        files[relative] = sha256_file(source)
    return {"sha256": source_bundle_sha256(files), "files": files}


def freeze_standalone_contract(*, evaluator: str, protocol: Mapping,
                               source_files: tuple[str, ...],
                               root: pathlib.Path = ROOT) -> dict:
    """冻结 standalone evaluator 的二进制、内容、源码与协议身份。"""
    root = root.resolve()
    if not evaluator:
        raise ValueError("standalone evaluator 名称不能为空")
    normalized_protocol = strict_json_loads(_canonical_json(protocol))
    return {
        "schema_version": 1,
        "protocol_version": PROTOCOL_VERSION,
        "evaluator": evaluator,
        "protocol": normalized_protocol,
        "runtime": runtime_identity(root, bridge_binary_path(root)),
        "sources": _standalone_sources(root, source_files),
    }


def require_fresh_native_runtime(evaluator: str) -> None:
    """A disk hash cannot identify bridge pages mapped before the freeze point."""
    if "_diablogym" in sys.modules:
        raise RuntimeError(
            f"{evaluator} 必须在未预载 diablogym bridge 的新进程中运行")


def verify_loaded_native_runtime(contract: Mapping,
                                 root: pathlib.Path = ROOT) -> None:
    """Check the mapped extension path and every frozen runtime input."""
    native = sys.modules.get("_diablogym")
    if native is None:
        raise RuntimeError("评测环境导入后仍未找到已映射的 _diablogym bridge")
    try:
        expected = contract["runtime"]
        expected_path = pathlib.Path(expected["bridge"]["path"]).resolve()
        actual_path = pathlib.Path(native.__file__).resolve()
        content = expected["content"]
        data_dir = pathlib.Path(content["game_data"]["path"]).parent
        assets_dir = pathlib.Path(content["assets"]["path"])
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        raise ValueError("standalone contract 的 native runtime 结构异常") from exc
    if actual_path != expected_path:
        raise RuntimeError(
            f"实际加载 bridge 路径与冻结身份不一致:{actual_path} != {expected_path}")
    loaded_engine_binary_path(expected["engine"]["path"])
    current = runtime_identity(
        root.resolve(), actual_path, data_dir=data_dir, assets_dir=assets_dir)
    if current != expected:
        raise RuntimeError(
            "native import 期间 bridge/engine/内容/依赖版本/协议源码发生变化")


def verify_standalone_contract(contract: Mapping,
                               root: pathlib.Path = ROOT) -> None:
    """运行后重哈希所有 runtime/content/source，漂移即拒绝发布。"""
    try:
        runtime = contract["runtime"]
        content = runtime["content"]
        data_dir = pathlib.Path(content["game_data"]["path"]).parent
        assets_dir = pathlib.Path(content["assets"]["path"])
        source_names = tuple(contract["sources"]["files"])
        expected_protocol = contract["protocol"]
        evaluator = contract["evaluator"]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("standalone contract 结构异常") from exc
    root = root.resolve()
    current = {
        "schema_version": 1,
        "protocol_version": PROTOCOL_VERSION,
        "evaluator": evaluator,
        "protocol": strict_json_loads(_canonical_json(expected_protocol)),
        "runtime": runtime_identity(
            root, bridge_binary_path(root),
            data_dir=data_dir, assets_dir=assets_dir),
        "sources": _standalone_sources(root, source_names),
    }
    if current != contract:
        raise RuntimeError(
            "standalone 评估期间 protocol/source/bridge/engine/MPQ/Resources 漂移")


def verify_checkpoint_identity(path: str | pathlib.Path, expected_sha256: str) -> None:
    checkpoint = pathlib.Path(path)
    try:
        actual = sha256_file(checkpoint)
    except OSError as exc:
        raise RuntimeError(f"评估后 checkpoint 不可读:{checkpoint}") from exc
    if actual != expected_sha256:
        raise RuntimeError(
            f"评估期间 checkpoint 发生变化:{actual} != {expected_sha256}")


def main_contract() -> dict:
    return freeze_standalone_contract(
        evaluator=f"standalone-main-v{PROTOCOL_VERSION}", protocol=MAIN_PROTOCOL,
        source_files=MAIN_SOURCE_FILES)


def checkpoint_snapshot(model_path: str | pathlib.Path
                        ) -> tuple[pathlib.Path, bytes, str]:
    """只读一次 checkpoint；hash、类型识别和 SB3.load 共用同一字节快照。"""
    path = resolve_checkpoint_file(model_path)
    payload = path.read_bytes()
    if not payload:
        raise ValueError(f"checkpoint 为空:{path}")
    return path, payload, hashlib.sha256(payload).hexdigest()


def atomic_write_text(path: str | pathlib.Path, payload: str) -> None:
    """同目录 fsync 临时文件后原子替换，异常时不破坏旧正式文件。"""
    target = pathlib.Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f".{target.name}.{os.getpid()}.{time.time_ns()}.tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, target)
    finally:
        tmp.unlink(missing_ok=True)


def _markdown_row_key(line: str) -> str | None:
    if not line.startswith("|"):
        return None
    cells = line.rstrip("\r\n").split("|")
    return cells[1].strip() if len(cells) >= 3 else None


def _contract_marker(contract: Mapping) -> str:
    return f"<!-- {_CONTRACT_MARKER}{_encode_marker(contract)} -->"


def _validate_row_marker(line: str, contract: Mapping) -> dict:
    match = _ROW_MARKER_RE.search(line.rstrip("\r\n"))
    if match is None:
        raise ValueError(
            f"protocol-v{PROTOCOL_VERSION} 排行榜数据行缺少 provenance marker")
    visible = line[:match.start()].rstrip()
    provenance = _decode_marker(match.group(1))
    if not isinstance(provenance, dict):
        raise ValueError("排行榜行 provenance 必须是对象")
    common = {
        "schema_version", "contract_sha256", "protocol_version", "kind",
        "row_key", "row_sha256",
    }
    kind = provenance.get("kind")
    expected_fields = (common | {"model_path", "model_sha256", "mode"}
                       if kind == "model" else
                       common | {"policy", "oracle_path", "oracle_sha256",
                                 "result_sha256"}
                       if kind == "scripted_ref" else
                       common | {"archive_path", "archive_sha256",
                                 "worker_sha256", "manager_sha256"}
                       if kind == "assembled" else set())
    if not expected_fields or set(provenance) != expected_fields:
        raise ValueError("排行榜行 provenance 字段/类型异常")
    if (provenance["schema_version"] != 1
            or provenance["protocol_version"] != PROTOCOL_VERSION
            or provenance["contract_sha256"] != contract_sha256(contract)
            or provenance["row_sha256"]
            != hashlib.sha256(visible.encode("utf-8")).hexdigest()
            or provenance["row_key"] != _markdown_row_key(visible)):
        raise ValueError("排行榜行 provenance 与可见行/全局合同不一致")
    if kind == "model":
        if (not isinstance(provenance["model_path"], str)
                or not pathlib.Path(provenance["model_path"]).is_absolute()
                or _SHA256_RE.fullmatch(provenance["model_sha256"]) is None
                or not isinstance(provenance["mode"], str)
                or not provenance["mode"]):
            raise ValueError("模型排行榜行身份异常")
    elif kind == "scripted_ref":
        if (not isinstance(provenance["policy"], str) or not provenance["policy"]
                or not isinstance(provenance["oracle_path"], str)
                or not pathlib.Path(provenance["oracle_path"]).is_absolute()
                or _SHA256_RE.fullmatch(provenance["oracle_sha256"]) is None
                or _SHA256_RE.fullmatch(provenance["result_sha256"]) is None):
            raise ValueError("脚本参考行身份异常")
    else:
        if (not isinstance(provenance["archive_path"], str)
                or not pathlib.Path(provenance["archive_path"]).is_absolute()
                or any(not isinstance(provenance[key], str)
                       or _SHA256_RE.fullmatch(provenance[key]) is None
                       for key in ("archive_sha256", "worker_sha256",
                                   "manager_sha256"))):
            raise ValueError("组装体排行榜行身份异常")
    return provenance


def _is_data_row(line: str) -> bool:
    key = _markdown_row_key(line)
    return (key is not None and key != "run"
            and not set(key) <= {"-", ":"})


def _validate_board_text(text: str, contract: Mapping,
                         initial_text: str) -> None:
    if not text.endswith("\n"):
        raise ValueError(
            f"protocol-v{PROTOCOL_VERSION} 排行榜必须以完整换行结尾")
    lines = text.splitlines()
    nonempty = [line for line in lines if line.strip()]
    if not nonempty or nonempty[0] != _contract_marker(contract):
        raise ValueError(
            "排行榜缺少匹配的 "
            f"protocol-v{PROTOCOL_VERSION} 全局合同；旧榜只读，必须另建/"
            f"rebuild v{PROTOCOL_VERSION} 榜")
    if sum(line.startswith(f"<!-- {_CONTRACT_MARKER}") for line in lines) != 1:
        raise ValueError("排行榜全局合同 marker 数量异常")
    seen_keys = set()
    table_headers = 0
    for line in lines:
        key = _markdown_row_key(line)
        if key is None or set(key) <= {"-", ":"}:
            continue
        if key == "run":
            table_headers += 1
            if _ROW_MARKER in line:
                raise ValueError("排行榜保留键 run 不可作为数据行")
            continue
        if key in seen_keys:
            raise ValueError(f"排行榜含重复 run key:{key!r}")
        seen_keys.add(key)
        _validate_row_marker(line, contract)
    if table_headers != 1:
        raise ValueError("排行榜必须且只能包含一个 run 表头")

    # 全局合同只描述评估制度，表头则定义可见列的语义。移除所有已验证数据行
    # 后必须逐字节还原建榜模板，避免手工改列名/顺序后继续混排新结果。
    skeleton = "".join(
        line for line in text.splitlines(keepends=True)
        if not _is_data_row(line))
    expected = _contract_marker(contract) + "\n\n" + initial_text
    if skeleton != expected:
        raise ValueError(
            f"protocol-v{PROTOCOL_VERSION} 排行榜表头/列协议发生变化，"
            "必须另建/rebuild")


def ensure_leaderboard_compatible(path: str | pathlib.Path,
                                  contract: Mapping, *,
                                  initial_text: str) -> None:
    target = pathlib.Path(path)
    if target.exists():
        _validate_board_text(
            target.read_text(encoding="utf-8"), contract, initial_text)


def versioned_row_key(label: str, identity_sha256: str) -> str:
    """以内容身份派生稳定短键；完整 SHA 仍保存在行 provenance 中。"""
    if (not isinstance(label, str) or not label.strip()
            or any(ch in label for ch in "|\r\n")):
        raise ValueError(f"非法排行榜标签:{label!r}")
    if _SHA256_RE.fullmatch(identity_sha256) is None:
        raise ValueError("排行榜版本键需要完整的小写 SHA-256")
    return f"{label.strip()}@{identity_sha256[:16]}"


def model_leaderboard_row(visible: str, *, row_key: str, contract: Mapping,
                          model_path: str, model_sha256: str, mode: str) -> str:
    provenance = {
        "schema_version": 1,
        "contract_sha256": contract_sha256(contract),
        "protocol_version": PROTOCOL_VERSION,
        "kind": "model",
        "row_key": row_key,
        "row_sha256": hashlib.sha256(visible.encode("utf-8")).hexdigest(),
        "model_path": str(pathlib.Path(model_path).resolve()),
        "model_sha256": model_sha256,
        "mode": mode,
    }
    row = f"{visible} <!-- {_ROW_MARKER}{_encode_marker(provenance)} -->"
    _validate_row_marker(row, contract)
    return row


def scripted_leaderboard_row(visible: str, *, row_key: str, contract: Mapping,
                             policy: str, oracle_path: str, oracle_sha256: str,
                             result_sha256: str) -> str:
    provenance = {
        "schema_version": 1,
        "contract_sha256": contract_sha256(contract),
        "protocol_version": PROTOCOL_VERSION,
        "kind": "scripted_ref",
        "row_key": row_key,
        "row_sha256": hashlib.sha256(visible.encode("utf-8")).hexdigest(),
        "policy": policy,
        "oracle_path": str(pathlib.Path(oracle_path).resolve()),
        "oracle_sha256": oracle_sha256,
        "result_sha256": result_sha256,
    }
    row = f"{visible} <!-- {_ROW_MARKER}{_encode_marker(provenance)} -->"
    _validate_row_marker(row, contract)
    return row


def assembled_leaderboard_row(visible: str, *, row_key: str, contract: Mapping,
                              archive_path: str, archive_sha256: str,
                              worker_sha256: str, manager_sha256: str) -> str:
    provenance = {
        "schema_version": 1,
        "contract_sha256": contract_sha256(contract),
        "protocol_version": PROTOCOL_VERSION,
        "kind": "assembled",
        "row_key": row_key,
        "row_sha256": hashlib.sha256(visible.encode("utf-8")).hexdigest(),
        "archive_path": str(pathlib.Path(archive_path).resolve()),
        "archive_sha256": archive_sha256,
        "worker_sha256": worker_sha256,
        "manager_sha256": manager_sha256,
    }
    row = f"{visible} <!-- {_ROW_MARKER}{_encode_marker(provenance)} -->"
    _validate_row_marker(row, contract)
    return row


def upsert_leaderboard_rows(path: str | pathlib.Path,
                            rows: Mapping[str, str], *,
                            contract: Mapping,
                            initial_text: str,
                            lock_path: str | pathlib.Path | None = None) -> None:
    """只在同一 v3 全局合同内去重、读改写并原子提交榜单。"""
    target = pathlib.Path(path)
    for key, row in rows.items():
        if (not key or key == "run" or set(key) <= {"-", ":"}
                or any(ch in key for ch in "|\r\n")):
            raise ValueError(f"非法排行榜行键:{key!r}")
        if "\n" in row.rstrip("\n") or _markdown_row_key(row) != key:
            raise ValueError(f"排行榜行与键不一致:{key!r}")
        _validate_row_marker(row, contract)
    lock = (pathlib.Path(lock_path) if lock_path is not None
            else target.with_name(f".{target.name}.lock"))
    with exclusive_lock(lock, f"{target.name} 排行榜"):
        if target.exists():
            text = target.read_text(encoding="utf-8")
            _validate_board_text(text, contract, initial_text)
        else:
            text = _contract_marker(contract) + "\n\n" + initial_text
        lines = text.splitlines(keepends=True)
        existing = {
            _markdown_row_key(line): line.rstrip("\r\n")
            for line in lines if _is_data_row(line)
        }
        pending = []
        for key, row in rows.items():
            old = existing.get(key)
            new = row.rstrip("\r\n")
            if old is not None and old != new:
                raise ValueError(
                    f"排行榜键 {key!r} 已绑定不同结果；拒绝静默覆盖旧行")
            if old is None:
                pending.append(row)
        try:
            last_row = max(i for i, line in enumerate(lines) if line.startswith("|"))
        except ValueError as exc:
            raise ValueError(f"排行榜缺少 Markdown 表格:{target}") from exc
        for row in pending:
            lines.insert(last_row + 1, row if row.endswith("\n") else row + "\n")
            last_row += 1
        # 持榜单锁做最后一道紧贴 commit 的重哈希，封住 evaluate() 返回后
        # 到 os.replace 前的替换窗口。只复验本次请求行；其余旧行是历史证据，
        # 不依赖原模型仍留在原路径。
        verify_standalone_contract(contract)
        for row in rows.values():
            provenance = _validate_row_marker(row, contract)
            if provenance["kind"] == "model":
                verify_checkpoint_identity(
                    provenance["model_path"], provenance["model_sha256"])
            elif provenance["kind"] == "scripted_ref":
                try:
                    oracle_sha = sha256_file(provenance["oracle_path"])
                except OSError as exc:
                    raise RuntimeError("发布前 oracle 不可读") from exc
                if oracle_sha != provenance["oracle_sha256"]:
                    raise RuntimeError("probe oracle 在发布前发生变化")
            else:
                verify_checkpoint_identity(
                    provenance["archive_path"], provenance["archive_sha256"])
        payload = "".join(lines)
        _validate_board_text(payload, contract, initial_text)
        if pending:
            atomic_write_text(target, payload)


def _model_kind_from_payload(payload: bytes, model_path: str | pathlib.Path) -> str:
    try:
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            data = json.loads(archive.read("data"))
        module = data.get("policy_class", {}).get("__module__", "").lower()
        if "recurrent" in module:
            return "recurrent"
        if "maskable" in module:
            return "masked"
    except (AttributeError, KeyError, TypeError, ValueError, json.JSONDecodeError,
            zipfile.BadZipFile):
        pass
    name = str(model_path).lower()
    if "lstm" in name:
        return "recurrent"
    if "mask" in name:
        return "masked"
    return "ppo"


def validated_episode_extra(info, seed: int) -> dict:
    extra = info.get("episode_extra") if isinstance(info, dict) else None
    if not isinstance(info, dict) or info.get("episode_seed") != seed:
        raise RuntimeError(f"seed {seed} 的终局 info 身份异常")
    required = {"kills", "depth", "died"}
    if not isinstance(extra, dict) or not required <= set(extra):
        raise RuntimeError(f"seed {seed} 缺少完整 episode_extra")
    kills, depth, died = extra["kills"], extra["depth"], extra["died"]
    if (not isinstance(kills, int) or isinstance(kills, bool) or kills < 0
            or not isinstance(depth, int) or isinstance(depth, bool) or depth < 1
            or not isinstance(died, bool)):
        raise RuntimeError(f"seed {seed} 的 episode_extra 类型/范围异常")
    return extra


def model_kind(model_path: str) -> str:
    """从 SB3 存档元数据识别算法，避免依赖目录名里恰好含 mask/lstm。"""
    _path, payload, _digest = checkpoint_snapshot(model_path)
    return _model_kind_from_payload(payload, model_path)


def evaluate(model_path: str, recurrent: bool | None = None,
             masked: bool | None = None, *, contract: Mapping | None = None):
    require_fresh_native_runtime("evaluate.py")
    if contract is None:
        contract = main_contract()

    from diablogym import DiabloGymEnv
    import models  # noqa: F401  (注册自定义提取器,load 时需要可导入)
    verify_loaded_native_runtime(contract)

    checkpoint, payload, model_sha256 = checkpoint_snapshot(model_path)
    kind = _model_kind_from_payload(payload, model_path)
    recurrent = (kind == "recurrent") if recurrent is None else recurrent
    masked = (kind == "masked") if masked is None else masked
    if recurrent and masked:
        raise ValueError("检查点不能同时按 RecurrentPPO 与 MaskablePPO 评估")
    if recurrent:
        from sb3_contrib import RecurrentPPO
        model = RecurrentPPO.load(io.BytesIO(payload), device="cpu")
    elif masked:
        from sb3_contrib import MaskablePPO
        model = MaskablePPO.load(io.BytesIO(payload), device="cpu")
    else:
        from stable_baselines3 import PPO
        model = PPO.load(io.BytesIO(payload), device="cpu")

    env = DiabloGymEnv(ticks_per_step=4, max_steps=1500,
                       start_in_dungeon=True, include_raw=False)
    kills, zeros, depth2 = [], 0, 0
    t0 = time.time()
    try:
        for seed in SEEDS:
            obs, _ = env.reset(seed=seed)
            st = None
            ep_start = np.ones((1,), dtype=bool)
            done = trunc = False
            info = {}
            while not (done or trunc):
                if recurrent:
                    a, st = model.predict(
                        obs, state=st, episode_start=ep_start, deterministic=True)
                    ep_start = np.zeros((1,), dtype=bool)
                elif masked:
                    # 掩码是策略分布的一部分:评估不带掩码 = 换了一个策略
                    a, _ = model.predict(obs, action_masks=env.action_masks(),
                                         deterministic=True)
                else:
                    a, _ = model.predict(obs, deterministic=True)
                obs, _reward, done, trunc, info = env.step(int(a))
            ex = validated_episode_extra(info, seed)
            k = ex["kills"]
            kills.append(k)
            zeros += (k == 0)
            depth2 += (ex["depth"] >= 2)
    finally:
        env.close()

    verify_standalone_contract(contract)
    verify_checkpoint_identity(checkpoint, model_sha256)

    result = {
        "model": str(checkpoint),
        "model_sha256": model_sha256,
        "mean": round(s.mean(kills), 1),
        "median": s.median(kills),
        "max": max(kills),
        "zero": f"{zeros}/{len(SEEDS)}",
        "depth2": depth2,
        "secs": round(time.time() - t0, 1),
        "mode": "recurrent" if recurrent else "masked" if masked else "ppo",
        "contract": contract,
        "contract_sha256": contract_sha256(contract),
    }
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_path")
    kind = ap.add_mutually_exclusive_group()
    kind.add_argument("--recurrent", action="store_true", help="强制按 RecurrentPPO 加载")
    kind.add_argument("--masked", action="store_true", help="强制按 MaskablePPO 加载")
    kind.add_argument("--ppo", action="store_true", help="强制按普通 PPO 加载")
    args = ap.parse_args()
    forced = "recurrent" if args.recurrent else "masked" if args.masked else "ppo" if args.ppo else None
    model_path = args.model_path
    contract = main_contract()
    ensure_leaderboard_compatible(
        LEADERBOARD, contract, initial_text=LEADERBOARD_HEADER)
    r = evaluate(model_path,
                 recurrent=(forced == "recurrent") if forced else None,
                 masked=(forced == "masked") if forced else None,
                 contract=contract)
    if (r.get("contract") != contract
            or r.get("contract_sha256") != contract_sha256(contract)):
        raise RuntimeError("评估结果未绑定发车前 standalone contract")
    name = pathlib.Path(model_path).parent.name or pathlib.Path(model_path).stem
    row_key = versioned_row_key(name, r["model_sha256"])
    visible = (f"| {row_key} | {r['mean']} | {r['median']} | "
               f"{r['max']} | {r['zero']} | {r['depth2']} |")
    print(f"均击杀 {r['mean']} | 中位 {r['median']} | 最高 {r['max']} | "
          f"零杀 {r['zero']} | 到2层 {r['depth2']}  [{r['secs']}s]")
    line = model_leaderboard_row(
        visible, row_key=row_key, contract=contract, model_path=r["model"],
        model_sha256=r["model_sha256"], mode=r["mode"])
    upsert_leaderboard_rows(
        LEADERBOARD, {row_key: line}, contract=contract,
        initial_text=LEADERBOARD_HEADER,
        lock_path=LEADERBOARD_LOCK)
    print(f"已写入 {LEADERBOARD.name}")


if __name__ == "__main__":
    main()

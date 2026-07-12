"""评测档案 schema v2 / protocol v3、身份绑定与并发写入契约。

新档案必须能回答三件事：谁被评、使用了哪些二进制/游戏内容、逐种子明细
是否真的推出所声明的汇总。旧档案没有这些证据，默认拒绝；法证代码只有
在显式给出完整的历史文件 SHA-256 时才可读取。
"""

from __future__ import annotations

import contextlib
import ctypes
import datetime as dt
import fcntl
import hashlib
import importlib.metadata
import io
import json
import math
import os
import pathlib
import platform
import re
import stat as stat_module
import statistics
import sys
import sysconfig
import zipfile
from typing import Any, Iterable, Mapping


SCHEMA_VERSION = 2
PROTOCOL_NAME = "diablogym.eval_assembled"
PROTOCOL_VERSION = 3
PROTOCOL_MAX_STEPS = 3000
UINT32_MAX = 2**32 - 1
DEFAULT_MANAGER_RELATIVE = "train/models/v22-h-manager/policy.npz"
DEFAULT_MANAGER_SHA256 = "0f2264860b0960e7951efd424836b90c09c002cebca7bf8109fd669b13be63d7"

RUNTIME_PACKAGE_VERSIONS = {
    "numpy": "2.5.0",
    "gymnasium": "1.3.0",
    "torch": "2.12.1",
    "stable-baselines3": "2.9.0",
    "sb3-contrib": "2.9.0",
    "tensorboard": "2.21.0",
}

PROTOCOL_SOURCE_FILES = (
    "train/eval_assembled.py",
    "train/eval_contract.py",
    "python/diablogym/__init__.py",
    "python/diablogym/env.py",
    "python/diablogym/nav.py",
    "python/diablogym/options_env.py",
    "python/diablogym/worker_env.py",
)

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_TAG_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
_BASE_AGG_KEYS = {
    "n", "ret_mean", "ret_median", "died", "depth_median", "l3",
    "kills_mean", "farm_tau_mean", "farm_descend_rate", "override_rate",
    "cap_rate",
}
_ENGAGEMENT_KEYS = {
    "worker_calls", "worker_action_hist", "worker_divergences",
    "script_divergence_rate",
}
_ROW_KEYS = {
    "seed", "ret", "depth", "died", "kills", "farm_n", "farm_tau_mean",
    "farm_tau_sum", "farm_descend", "windows", "beats", "overrides", "cap",
    "mode_seq",
}


class EvalContractError(ValueError):
    """评测档案不满足身份、schema 或数值契约。"""


class OutputReservationError(FileExistsError):
    """目标档案已存在，或同 tag 正由另一个进程生成。"""


class OperationalFailure(RuntimeError):
    """驱动已识别的训练、导出或评测基础设施失败。"""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise EvalContractError(message)


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _finite_number(value: Any, label: str) -> float:
    _require(isinstance(value, (int, float)) and not isinstance(value, bool),
             f"{label} 必须是数值")
    result = float(value)
    _require(math.isfinite(result), f"{label} 必须有限")
    return result


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: str | pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _validate_sha256(value: Any, label: str) -> str:
    _require(isinstance(value, str) and _SHA256_RE.fullmatch(value) is not None,
             f"{label} 必须是完整的小写 SHA-256")
    return value


def source_bundle_sha256(files: Mapping[str, str]) -> str:
    """按相对路径与各文件完整 SHA 生成稳定的协议源码 bundle SHA。"""
    h = hashlib.sha256()
    for name, digest in sorted(files.items()):
        _validate_sha256(digest, f"runtime.python_protocol.files[{name!r}]")
        h.update(name.encode("utf-8"))
        h.update(b"\0")
        h.update(digest.encode("ascii"))
        h.update(b"\n")
    return h.hexdigest()


def runtime_versions_identity() -> dict[str, Any]:
    """Bind numerical dependencies without importing their heavy modules."""
    try:
        packages = {
            name: importlib.metadata.version(name)
            for name in RUNTIME_PACKAGE_VERSIONS
        }
    except importlib.metadata.PackageNotFoundError as exc:
        raise EvalContractError(f"评测运行时依赖缺失: {exc.name}") from exc
    mismatches = {
        name: (packages[name], expected)
        for name, expected in RUNTIME_PACKAGE_VERSIONS.items()
        if packages[name] != expected
    }
    _require(not mismatches,
             f"评测运行时版本漂移（升级须重做数值回归）: {mismatches}")
    return {
        "python": {
            "implementation": sys.implementation.name,
            "version": platform.python_version(),
            "cache_tag": sys.implementation.cache_tag,
        },
        "packages": packages,
    }


def default_game_data_dir() -> pathlib.Path:
    """与 DiabloGymEnv 的默认 data_dir 保持同一平台契约。"""
    return (pathlib.Path.home()
            / "Library" / "Application Support" / "diasurgical" / "devilution")


def default_assets_dir(root: pathlib.Path) -> pathlib.Path:
    return (root / "build" / "engine" / "devilutionx.app"
            / "Contents" / "Resources")


def _absolute_path(path: str | pathlib.Path) -> pathlib.Path:
    """取得绝对词法路径但不消解 symlink，便于复验路径本身的选择。"""
    return pathlib.Path(os.path.abspath(os.fspath(pathlib.Path(path).expanduser())))


def _stat_signature(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (value.st_dev, value.st_ino, value.st_size,
            value.st_mtime_ns, value.st_ctime_ns)


def _stable_file_sha256(path: pathlib.Path, label: str) -> str:
    """哈希一个始终由同一路径指向、且读取期间未变化的普通文件。"""
    _require(not path.is_symlink(), f"{label} 不允许符号链接: {path}")
    try:
        with open(path, "rb") as stream:
            before = os.fstat(stream.fileno())
            _require(stat_module.S_ISREG(before.st_mode),
                     f"{label} 不是普通文件: {path}")
            digest = hashlib.sha256()
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
            after = os.fstat(stream.fileno())
        current = path.stat()
    except OSError as exc:
        raise EvalContractError(f"{label} 不可稳定读取: {path}: {exc}") from exc
    signature = _stat_signature(before)
    _require(signature == _stat_signature(after)
             and signature == _stat_signature(current),
             f"{label} 在哈希期间发生变化: {path}")
    return digest.hexdigest()


def game_data_identity(data_dir: str | pathlib.Path | None = None
                       ) -> dict[str, Any]:
    """按 LoadGameArchives 的优先级绑定默认 data_dir 中实际主 MPQ。"""
    directory = _absolute_path(data_dir or default_game_data_dir())
    _require(directory.is_dir(), f"游戏 data_dir 不存在: {directory}")
    for name in ("DIABDAT.MPQ", "diabdat.mpq", "spawn.mpq"):
        candidate = directory / name
        if candidate.is_file():
            return {
                "path": str(candidate),
                "sha256": _stable_file_sha256(candidate, "游戏主 MPQ"),
            }
    raise EvalContractError(
        f"游戏 data_dir 缺少 DIABDAT.MPQ/diabdat.mpq/spawn.mpq: {directory}")


def _resource_files(directory: pathlib.Path) -> list[pathlib.Path]:
    try:
        entries = list(directory.rglob("*"))
    except OSError as exc:
        raise EvalContractError(f"资源树不可遍历: {directory}: {exc}") from exc
    files: list[pathlib.Path] = []
    for entry in entries:
        _require(not entry.is_symlink(), f"资源树不允许符号链接: {entry}")
        if entry.is_file():
            files.append(entry)
        else:
            _require(entry.is_dir(), f"资源树含特殊文件: {entry}")
    return sorted(files, key=lambda path: path.relative_to(directory).as_posix())


def assets_tree_identity(assets_dir: str | pathlib.Path) -> dict[str, Any]:
    """以相对路径和每个文件的原始内容计算确定性 Resources 树哈希。"""
    directory = _absolute_path(assets_dir)
    _require(directory.is_dir(), f"引擎 Resources 目录不存在: {directory}")
    _require(not directory.is_symlink(),
             f"引擎 Resources 目录不允许符号链接: {directory}")
    files = _resource_files(directory)
    _require(bool(files), f"引擎 Resources 目录为空: {directory}")
    digest = hashlib.sha256(b"diablogym-resources-tree-v1\0")
    signatures: dict[str, tuple[int, int, int, int, int]] = {}
    try:
        for path in files:
            relative = path.relative_to(directory).as_posix()
            relative_bytes = relative.encode("utf-8")
            with open(path, "rb") as stream:
                before = os.fstat(stream.fileno())
                _require(stat_module.S_ISREG(before.st_mode),
                         f"资源树条目不是普通文件: {path}")
                _require(0 <= before.st_size < 2**64,
                         f"资源树文件大小无法编码: {path}")
                digest.update(len(relative_bytes).to_bytes(8, "big"))
                digest.update(relative_bytes)
                digest.update(before.st_size.to_bytes(8, "big"))
                read_size = 0
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
                    read_size += len(chunk)
                after = os.fstat(stream.fileno())
            signature = _stat_signature(before)
            _require(read_size == before.st_size
                     and signature == _stat_signature(after),
                     f"资源文件在树哈希期间发生变化: {path}")
            signatures[relative] = signature
    except OSError as exc:
        raise EvalContractError(f"资源树不可稳定读取: {exc}") from exc

    final_files = _resource_files(directory)
    final_relatives = [path.relative_to(directory).as_posix()
                       for path in final_files]
    _require(final_relatives == list(signatures),
             "资源树文件集合在哈希期间发生变化")
    try:
        stable = all(_stat_signature(path.stat()) == signatures[relative]
                     for path, relative in zip(final_files, final_relatives))
    except OSError as exc:
        raise EvalContractError(f"资源树无法完成读后复验: {exc}") from exc
    _require(stable, "资源树文件在哈希完成前发生变化")
    return {
        "path": str(directory),
        "sha256": digest.hexdigest(),
        "file_count": len(files),
    }


def content_identity(root: pathlib.Path, *,
                     data_dir: str | pathlib.Path | None = None,
                     assets_dir: str | pathlib.Path | None = None
                     ) -> dict[str, Any]:
    root = root.resolve()
    return {
        "game_data": game_data_identity(data_dir),
        "assets": assets_tree_identity(assets_dir or default_assets_dir(root)),
    }


def engine_binary_path(root: pathlib.Path) -> pathlib.Path:
    """定位 bridge 实际链接的 DevilutionX 共享引擎，歧义时 fail closed。"""
    engine_root = root.resolve() / "build" / "engine"
    if sys.platform == "darwin":
        names = ("liblibdevilutionx_so.dylib",)
    elif sys.platform == "win32":
        names = ("libdevilutionx_so.dll", "liblibdevilutionx_so.dll")
    else:
        names = ("liblibdevilutionx_so.so",)
    directories = (
        engine_root,
        *(engine_root / config for config in
          ("Release", "RelWithDebInfo", "Debug", "MinSizeRel")),
    )
    matches = {
        candidate.resolve()
        for directory in directories
        for name in names
        if (candidate := directory / name).is_file()
    }
    _require(len(matches) == 1,
             "无法唯一定位 DevilutionX engine 共享库: "
             f"root={engine_root}, matches={sorted(map(str, matches))}")
    return next(iter(matches))


def loaded_engine_binary_path(expected_path: str | pathlib.Path) -> pathlib.Path:
    """Resolve the engine image actually mapped in this process, fail closed."""
    expected = pathlib.Path(expected_path).resolve()
    candidates: set[pathlib.Path] = set()
    if sys.platform == "darwin":
        process = ctypes.CDLL(None)
        image_count = process._dyld_image_count
        image_count.argtypes = []
        image_count.restype = ctypes.c_uint32
        image_name = process._dyld_get_image_name
        image_name.argtypes = [ctypes.c_uint32]
        image_name.restype = ctypes.c_char_p
        for index in range(image_count()):
            raw = image_name(index)
            if raw:
                path = pathlib.Path(os.fsdecode(raw))
                if path.name == expected.name:
                    candidates.add(path.resolve())
    elif sys.platform == "win32":
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        get_handle = kernel32.GetModuleHandleW
        get_handle.argtypes = [ctypes.c_wchar_p]
        get_handle.restype = ctypes.c_void_p
        get_filename = kernel32.GetModuleFileNameW
        get_filename.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p,
                                 ctypes.c_uint32]
        get_filename.restype = ctypes.c_uint32
        handle = get_handle(expected.name)
        if handle:
            buffer = ctypes.create_unicode_buffer(32768)
            if get_filename(handle, buffer, len(buffer)):
                candidates.add(pathlib.Path(buffer.value).resolve())
    else:
        maps = pathlib.Path("/proc/self/maps")
        if maps.is_file():
            for line in maps.read_text(encoding="utf-8", errors="strict").splitlines():
                fields = line.split(maxsplit=5)
                if len(fields) == 6 and not fields[5].endswith(" (deleted)"):
                    path = pathlib.Path(fields[5])
                    if path.name == expected.name:
                        candidates.add(path.resolve())
    _require(len(candidates) == 1,
             "无法唯一定位当前进程实际映射的 DevilutionX engine: "
             f"expected={expected}, mapped={sorted(map(str, candidates))}")
    actual = next(iter(candidates))
    _require(actual == expected,
             f"实际映射 engine 路径与冻结身份不一致: {actual} != {expected}")
    return actual


def runtime_identity(root: pathlib.Path, bridge_path: pathlib.Path,
                     engine_path: pathlib.Path | None = None, *,
                     data_dir: str | pathlib.Path | None = None,
                     assets_dir: str | pathlib.Path | None = None) -> dict[str, Any]:
    root = root.resolve()
    bridge_path = bridge_path.resolve()
    engine_path = (engine_path or engine_binary_path(root)).resolve()
    files: dict[str, str] = {}
    for relative in PROTOCOL_SOURCE_FILES:
        path = root / relative
        if not path.is_file():
            raise EvalContractError(f"协议源码缺失: {path}")
        files[relative] = sha256_file(path)
    return {
        "bridge": {
            "path": str(bridge_path),
            "sha256": sha256_file(bridge_path),
        },
        "engine": {
            "path": str(engine_path),
            "sha256": sha256_file(engine_path),
        },
        "content": content_identity(
            root, data_dir=data_dir, assets_dir=assets_dir),
        "versions": runtime_versions_identity(),
        "python_protocol": {
            "sha256": source_bundle_sha256(files),
            "files": files,
        },
    }


def bridge_binary_path(root: pathlib.Path) -> pathlib.Path:
    suffix = sysconfig.get_config_var("EXT_SUFFIX")
    _require(isinstance(suffix, str) and bool(suffix), "无法确定当前 Python 扩展 ABI")
    path = root.resolve() / "build" / f"_diablogym{suffix}"
    _require(path.is_file(), f"当前 ABI 的 bridge 二进制不存在: {path}")
    return path


def resolve_checkpoint_file(path: str | pathlib.Path) -> pathlib.Path:
    candidate = pathlib.Path(path)
    if candidate.is_file():
        return candidate.resolve()
    if candidate.suffix.lower() != ".zip":
        zipped = pathlib.Path(f"{candidate}.zip")
        if zipped.is_file():
            return zipped.resolve()
    raise EvalContractError(f"checkpoint 不存在: {candidate}")


def checkpoint_num_timesteps_bytes(payload: bytes, label: str = "checkpoint") -> int:
    """Read SB3 progress from the same immutable bytes used by the loader."""
    try:
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            data = json.loads(archive.read("data"))
        value = data["num_timesteps"]
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError,
            zipfile.BadZipFile) as exc:
        raise EvalContractError(f"checkpoint num_timesteps 不可读: {label}") from exc
    _require(_is_int(value) and value >= 0,
             f"checkpoint num_timesteps 非法: {value!r}")
    return value


def checkpoint_num_timesteps(path: str | pathlib.Path) -> int:
    checkpoint = resolve_checkpoint_file(path)
    try:
        payload = checkpoint.read_bytes()
    except OSError as exc:
        raise EvalContractError(
            f"checkpoint num_timesteps 不可读: {checkpoint}") from exc
    return checkpoint_num_timesteps_bytes(payload, str(checkpoint))


def file_identity(kind: str, path: str | pathlib.Path,
                  *, num_timesteps: int | None = None,
                  gate_report_path: str | pathlib.Path | None = None) -> dict[str, Any]:
    resolved = pathlib.Path(path).resolve()
    _require(resolved.is_file(), f"策略文件不存在: {resolved}")
    report_sha = None
    if gate_report_path is not None:
        report = pathlib.Path(gate_report_path).resolve()
        _require(report.is_file(), f"策略闸门报告不存在: {report}")
        report_sha = sha256_file(report)
    return {
        "kind": kind,
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "num_timesteps": num_timesteps,
        "gate_report_sha256": report_sha,
    }


def script_worker_identity(protocol_bundle_sha256: str) -> dict[str, Any]:
    _validate_sha256(protocol_bundle_sha256, "protocol bundle SHA")
    return {
        "kind": "script",
        "path": None,
        "sha256": _sha256_text(
            f"diablogym-script-farm-v1\0{protocol_bundle_sha256}"),
        "num_timesteps": None,
        "gate_report_sha256": None,
    }


def freeze_eval_identity(root: pathlib.Path, worker_spec: str | pathlib.Path,
                         manager_path: str | pathlib.Path | None = None
                         ) -> dict[str, Any]:
    """在启动评测子进程前冻结全部输入与当前运行时身份。"""
    root = root.resolve()
    runtime = runtime_identity(root, bridge_binary_path(root))
    spec = str(worker_spec)
    if spec == "script":
        worker = script_worker_identity(runtime["python_protocol"]["sha256"])
    elif spec == "bc":
        path = root / "train" / "runs" / "bc-worker" / "policy_sd.pt"
        worker = file_identity(
            "bc_state_dict", path, gate_report_path=path.with_name("bc_report.json"))
    elif pathlib.Path(spec).suffix.lower() == ".npz":
        worker = file_identity("numpy_policy", pathlib.Path(spec))
    else:
        checkpoint = resolve_checkpoint_file(spec)
        try:
            checkpoint_payload = checkpoint.read_bytes()
        except OSError as exc:
            raise EvalContractError(f"checkpoint 不可读: {checkpoint}") from exc
        worker = {
            "kind": "sb3_checkpoint", "path": str(checkpoint),
            "sha256": hashlib.sha256(checkpoint_payload).hexdigest(),
            "num_timesteps": checkpoint_num_timesteps_bytes(
                checkpoint_payload, str(checkpoint)),
            "gate_report_sha256": None,
        }
    manager_file = (pathlib.Path(manager_path) if manager_path is not None else
                    root / DEFAULT_MANAGER_RELATIVE)
    manager = file_identity("numpy_policy", manager_file)
    if manager_path is None:
        _require(manager["sha256"] == DEFAULT_MANAGER_SHA256,
                 "默认 v22-H manager SHA 漂移: "
                 f"{manager['sha256']} != {DEFAULT_MANAGER_SHA256}")
    return {"worker": worker, "manager": manager, "runtime": runtime}


def expected_eval_identity(snapshot: Mapping[str, Any], *, tag: str,
                           seeds: Iterable[int]) -> dict[str, Any]:
    worker, manager, runtime = (
        snapshot["worker"], snapshot["manager"], snapshot["runtime"])
    game_data = runtime["content"]["game_data"]
    assets = runtime["content"]["assets"]
    return {
        "expected_tag": tag,
        "expected_seeds": list(seeds),
        "expected_worker_kind": worker["kind"],
        "expected_worker_sha256": worker["sha256"],
        "expected_worker_gate_report_sha256": worker["gate_report_sha256"],
        "expected_manager_kind": manager["kind"],
        "expected_manager_sha256": manager["sha256"],
        "expected_worker_num_timesteps": worker["num_timesteps"],
        "expected_bridge_sha256": runtime["bridge"]["sha256"],
        "expected_engine_sha256": runtime["engine"]["sha256"],
        "expected_game_data_path": game_data["path"],
        "expected_game_data_sha256": game_data["sha256"],
        "expected_assets_path": assets["path"],
        "expected_assets_sha256": assets["sha256"],
        "expected_assets_file_count": assets["file_count"],
        "expected_runtime_versions": runtime["versions"],
        "expected_protocol_bundle_sha256": runtime["python_protocol"]["sha256"],
    }


def verify_eval_identity(snapshot: Mapping[str, Any], root: pathlib.Path) -> None:
    """子进程结束后重哈希；任何评测期间输入替换都使档案作废。"""
    verify_file_identity(snapshot["worker"])
    verify_file_identity(snapshot["manager"])
    try:
        content = snapshot["runtime"]["content"]
        data_dir = pathlib.Path(content["game_data"]["path"]).parent
        assets_dir = pathlib.Path(content["assets"]["path"])
    except (KeyError, TypeError, ValueError) as exc:
        raise EvalContractError("冻结的 content identity 结构异常") from exc
    current = runtime_identity(
        root.resolve(), bridge_binary_path(root.resolve()),
        data_dir=data_dir, assets_dir=assets_dir)
    _require(current == snapshot["runtime"],
             "评测期间 bridge、engine、游戏数据、资源树或 Python 协议源码发生变化")


def verify_file_identity(identity: Mapping[str, Any]) -> None:
    if identity.get("kind") == "script":
        return
    path = identity.get("path")
    _require(isinstance(path, str) and pathlib.Path(path).is_file(),
             f"策略身份路径失效: {path!r}")
    actual = sha256_file(path)
    _require(actual == identity.get("sha256"),
             f"评测期间策略文件发生变化: {actual} != {identity.get('sha256')!r}")
    expected_report = identity.get("gate_report_sha256")
    if expected_report is not None:
        report_path = pathlib.Path(path).with_name("bc_report.json")
        _require(report_path.is_file(), f"评测期间 BC gate report 消失: {report_path}")
        actual_report = sha256_file(report_path)
        _require(actual_report == expected_report,
                 "评测期间 BC gate report 发生变化: "
                 f"{actual_report} != {expected_report!r}")


def make_protocol(seeds: Iterable[int]) -> dict[str, Any]:
    return {
        "name": PROTOCOL_NAME,
        "version": PROTOCOL_VERSION,
        "environment": "OptionsEnv",
        "max_steps": PROTOCOL_MAX_STEPS,
        "action_selection": "argmax_with_action_masks",
        "manager_forward": "numpy_tanh_mlp",
        "reward": "undiscounted_manager_ledger",
        "deterministic": True,
        "seeds": list(seeds),
    }


def make_meta(*, tag: str, seeds: Iterable[int], worker: Mapping[str, Any],
              manager: Mapping[str, Any], runtime: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "tag": tag,
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "protocol": make_protocol(seeds),
        "worker": dict(worker),
        "manager": dict(manager),
        "runtime": dict(runtime),
    }


def recompute_agg(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    _require(n > 0, "评测 rows 不能为空")
    rets = sorted(row["ret"] for row in rows)
    farm_n = sum(row["farm_n"] for row in rows)
    return {
        "n": n,
        "ret_mean": round(sum(rets) / n, 1),
        "ret_median": round(statistics.median(rets), 2),
        "died": sum(row["died"] for row in rows),
        "depth_median": statistics.median(row["depth"] for row in rows),
        "l3": sum(row["depth"] >= 3 for row in rows),
        "kills_mean": round(sum(row["kills"] for row in rows) / n, 1),
        "farm_tau_mean": round(sum(row["farm_tau_sum"] for row in rows)
                               / max(1, farm_n), 1),
        "farm_descend_rate": round(sum(row["farm_descend"] for row in rows)
                                    / max(1, farm_n), 4),
        "override_rate": round(sum(row["overrides"] for row in rows)
                               / max(1, sum(row["beats"] for row in rows)), 4),
        "cap_rate": round(sum(row["cap"] for row in rows)
                          / max(1, sum(row["windows"] for row in rows)), 4),
    }


def _validate_identity(identity: Any, label: str,
                       runtime_bundle_sha256: str) -> None:
    _require(isinstance(identity, dict), f"meta.{label} 必须是对象")
    _require(set(identity) == {
        "kind", "path", "sha256", "num_timesteps", "gate_report_sha256"},
        f"meta.{label} 字段异常")
    kind = identity["kind"]
    allowed = ({"script", "bc_state_dict", "numpy_policy", "sb3_checkpoint"}
               if label == "worker" else {"numpy_policy"})
    _require(kind in allowed, f"meta.{label}.kind 非法: {kind!r}")
    _validate_sha256(identity["sha256"], f"meta.{label}.sha256")
    if kind == "script":
        _require(identity["path"] is None and identity["num_timesteps"] is None
                 and identity["gate_report_sha256"] is None,
                 "script worker 不应声明文件、步数或闸门报告")
        expected = script_worker_identity(runtime_bundle_sha256)["sha256"]
        _require(identity["sha256"] == expected,
                 "script worker SHA 未绑定当前协议源码 bundle")
    else:
        _require(isinstance(identity["path"], str) and bool(identity["path"]),
                 f"meta.{label}.path 缺失")
        if kind == "sb3_checkpoint":
            _require(_is_int(identity["num_timesteps"])
                     and identity["num_timesteps"] >= 0,
                     "SB3 worker 必须声明非负 num_timesteps")
        else:
            _require(identity["num_timesteps"] is None,
                     f"{kind} 不应声明 num_timesteps")
        report_sha = identity["gate_report_sha256"]
        if kind == "bc_state_dict":
            _validate_sha256(report_sha, "meta.worker.gate_report_sha256")
        else:
            _require(report_sha is None,
                     f"{kind} 不应声明 gate_report_sha256")


def _validate_rows(rows: Any, seeds: list[int]) -> list[Mapping[str, Any]]:
    _require(isinstance(rows, list) and rows, "rows 必须是非空数组")
    _require(len(rows) == len(seeds), "rows 数量与协议 seeds 不一致")
    seen: list[int] = []
    for index, row in enumerate(rows):
        label = f"rows[{index}]"
        _require(isinstance(row, dict), f"{label} 必须是对象")
        _require(set(row) == _ROW_KEYS, f"{label} 字段异常")
        for key in ("seed", "depth", "kills", "farm_n", "farm_tau_sum",
                    "farm_descend",
                    "windows", "beats", "overrides", "cap"):
            _require(_is_int(row[key]), f"{label}.{key} 必须是整数")
        _require(0 <= row["seed"] <= UINT32_MAX,
                 f"{label}.seed 必须在 uint32 范围")
        _require(1 <= row["depth"] <= 16 and row["kills"] >= 0,
                 f"{label} 深度必须在 [1,16]，击杀不能为负")
        _require(row["farm_n"] >= 0 and 0 < row["windows"] <= row["beats"]
                 <= PROTOCOL_MAX_STEPS,
                 f"{label} 窗口/微步计数越界")
        _require(row["kills"] <= PROTOCOL_MAX_STEPS,
                 f"{label}.kills 超出单局微步预算")
        _require(0 <= row["farm_descend"] <= row["farm_n"] <= row["windows"],
                 f"{label} FARM 计数关系异常")
        _require(0 <= row["overrides"] <= row["beats"],
                 f"{label}.overrides 超出 beats")
        _require(0 <= row["cap"] <= row["windows"],
                 f"{label}.cap 超出 windows")
        _require(isinstance(row["died"], bool), f"{label}.died 必须是 bool")
        _finite_number(row["ret"], f"{label}.ret")
        tau_mean = _finite_number(row["farm_tau_mean"], f"{label}.farm_tau_mean")
        tau_sum = _finite_number(row["farm_tau_sum"], f"{label}.farm_tau_sum")
        _require(tau_mean >= 0 and tau_sum >= 0, f"{label} FARM tau 不能为负")
        _require(tau_sum <= PROTOCOL_MAX_STEPS,
                 f"{label}.farm_tau_sum 超出单局微步预算")
        expected_tau = round(tau_sum / max(1, row["farm_n"]), 1)
        _require(row["farm_tau_mean"] == expected_tau,
                 f"{label}.farm_tau_mean 与 farm_tau_sum 不一致")
        sequence = row["mode_seq"]
        _require(isinstance(sequence, str), f"{label}.mode_seq 必须是字符串")
        death_marker = sequence.endswith("†")
        core_sequence = sequence[:-1] if death_marker else sequence
        _require("†" not in core_sequence and set(core_sequence) <= {"F", "D", "R"},
                 f"{label}.mode_seq 含非法/错位标记")
        _require(death_marker == row["died"],
                 f"{label}.mode_seq 死亡标记与 died 不一致")
        _require(len(core_sequence) == row["windows"],
                 f"{label}.mode_seq 长度与 windows 不一致")
        _require(core_sequence.count("F") == row["farm_n"],
                 f"{label}.mode_seq FARM 数与 farm_n 不一致")
        seen.append(row["seed"])
    _require(seen == seeds, "rows seed 顺序/集合与协议 seeds 不精确一致")
    _require(len(seen) == len(set(seen)), "rows 含重复 seed")
    return rows


def _validate_agg(agg: Any, rows: list[Mapping[str, Any]], worker_kind: str) -> None:
    _require(isinstance(agg, dict), "agg 必须是对象")
    expected_keys = _BASE_AGG_KEYS | (_ENGAGEMENT_KEYS if worker_kind != "script" else set())
    _require(set(agg) == expected_keys, "agg 字段与 worker 类型/schema 不一致")
    for key in ("n", "died", "l3"):
        _require(_is_int(agg[key]), f"agg.{key} 必须是整数")
    for key in _BASE_AGG_KEYS - {"n", "died", "l3"}:
        _finite_number(agg[key], f"agg.{key}")
    n = len(rows)
    _require(0 <= agg["died"] <= n and 0 <= agg["l3"] <= n,
             "agg 死亡/L3 计数超界")
    for key in ("farm_descend_rate", "override_rate", "cap_rate"):
        _require(0 <= float(agg[key]) <= 1, f"agg.{key} 必须在 [0,1]")
    recomputed = recompute_agg(rows)
    for key, expected in recomputed.items():
        _require(agg[key] == expected,
                 f"agg.{key} 与 rows 重算不一致: {agg[key]!r} != {expected!r}")
    if worker_kind != "script":
        calls = agg["worker_calls"]
        _require(_is_int(calls) and calls >= 0, "agg.worker_calls 必须是非负整数")
        _require(calls <= sum(row["beats"] for row in rows),
                 "agg.worker_calls 超出全部底层拍数")
        histogram = agg["worker_action_hist"]
        _require(isinstance(histogram, dict), "agg.worker_action_hist 必须是对象")
        normalized: dict[int, int] = {}
        for raw_key, value in histogram.items():
            try:
                key = int(raw_key)
            except (TypeError, ValueError) as exc:
                raise EvalContractError("worker_action_hist 动作键必须是整数") from exc
            _require(str(key) == str(raw_key) or raw_key == key,
                     f"worker_action_hist 动作键非规范: {raw_key!r}")
            _require(0 <= key < 15 and key not in normalized,
                     f"worker_action_hist 动作键非法/重复: {raw_key!r}")
            _require(_is_int(value) and value >= 0,
                     f"worker_action_hist[{raw_key!r}] 必须是非负整数")
            normalized[key] = value
        _require(sum(normalized.values()) == calls,
                 "worker_action_hist 总数与 worker_calls 不一致")
        divergences = agg["worker_divergences"]
        _require(_is_int(divergences) and 0 <= divergences <= calls,
                 "agg.worker_divergences 必须在 [0, worker_calls]")
        divergence = _finite_number(
            agg["script_divergence_rate"], "agg.script_divergence_rate")
        _require(0 <= divergence <= 1,
                 "agg.script_divergence_rate 必须在 [0,1]")
        _require(agg["script_divergence_rate"]
                 == round(divergences / max(1, calls), 4),
                 "script_divergence_rate 与原始分歧计数不一致")


def validate_eval_archive(document: Any, *, expected_tag: str | None = None,
                          expected_seeds: Iterable[int] | None = None,
                          expected_worker_kind: str | None = None,
                          expected_worker_sha256: str | None = None,
                          expected_worker_gate_report_sha256: str | None = None,
                          expected_manager_kind: str | None = None,
                          expected_manager_sha256: str | None = None,
                          expected_worker_num_timesteps: int | None = None,
                          expected_bridge_sha256: str | None = None,
                          expected_engine_sha256: str | None = None,
                          expected_game_data_path: str | None = None,
                          expected_game_data_sha256: str | None = None,
                          expected_assets_path: str | None = None,
                          expected_assets_sha256: str | None = None,
                          expected_assets_file_count: int | None = None,
                          expected_runtime_versions: Mapping[str, Any] | None = None,
                          expected_protocol_bundle_sha256: str | None = None
                          ) -> dict[str, Any]:
    """严格校验一个 schema v2 档案；legacy 永远不会从这里静默放行。"""
    _require(isinstance(document, dict), "评测档案必须是 JSON 对象")
    _require(set(document) == {"schema_version", "meta", "agg", "rows"},
             "评测档案顶层字段异常或属于未授权 legacy schema")
    _require(document["schema_version"] == SCHEMA_VERSION,
             f"评测 schema 必须为 v{SCHEMA_VERSION}")
    meta = document["meta"]
    _require(isinstance(meta, dict), "meta 必须是对象")
    _require(set(meta) == {
        "tag", "created_at_utc", "protocol", "worker", "manager", "runtime"},
        "meta 字段异常")
    tag = meta["tag"]
    _require(isinstance(tag, str) and _TAG_RE.fullmatch(tag) is not None,
             "meta.tag 非法")
    _require(isinstance(meta["created_at_utc"], str) and bool(meta["created_at_utc"]),
             "meta.created_at_utc 缺失")
    try:
        created_at = dt.datetime.fromisoformat(meta["created_at_utc"])
    except ValueError as exc:
        raise EvalContractError("meta.created_at_utc 不是 ISO-8601 时间") from exc
    _require(created_at.tzinfo is not None
             and created_at.utcoffset() == dt.timedelta(0),
             "meta.created_at_utc 必须是 UTC 时间")
    if expected_tag is not None:
        _require(tag == expected_tag, f"评测 tag 错标: {tag!r} != {expected_tag!r}")

    protocol = meta["protocol"]
    _require(isinstance(protocol, dict), "meta.protocol 必须是对象")
    _require(set(protocol) == {
        "name", "version", "environment", "max_steps", "action_selection",
        "manager_forward", "reward", "deterministic", "seeds"},
        "meta.protocol 字段异常")
    _require(protocol["name"] == PROTOCOL_NAME
             and protocol["version"] == PROTOCOL_VERSION,
             "评测协议名称/版本不匹配")
    _require(protocol["environment"] == "OptionsEnv"
             and protocol["max_steps"] == PROTOCOL_MAX_STEPS
             and protocol["action_selection"] == "argmax_with_action_masks"
             and protocol["manager_forward"] == "numpy_tanh_mlp"
             and protocol["reward"] == "undiscounted_manager_ledger"
             and protocol["deterministic"] is True,
             "评测协议旋钮漂移")
    seeds = protocol["seeds"]
    _require(isinstance(seeds, list) and seeds
             and all(_is_int(seed) and 0 <= seed <= UINT32_MAX for seed in seeds),
             "meta.protocol.seeds 必须是非空 uint32 整数数组")
    _require(len(seeds) == len(set(seeds)), "meta.protocol.seeds 含重复值")
    if expected_seeds is not None:
        _require(seeds == list(expected_seeds), "评测 seed 列表与调用契约不一致")

    runtime = meta["runtime"]
    _require(isinstance(runtime, dict)
             and set(runtime) == {
                 "bridge", "engine", "content", "versions", "python_protocol"},
             "meta.runtime 字段异常")
    bridge = runtime["bridge"]
    _require(isinstance(bridge, dict) and set(bridge) == {"path", "sha256"}
             and isinstance(bridge["path"], str) and bool(bridge["path"]),
             "meta.runtime.bridge 字段异常")
    bridge_sha = _validate_sha256(bridge["sha256"], "meta.runtime.bridge.sha256")
    engine = runtime["engine"]
    _require(isinstance(engine, dict) and set(engine) == {"path", "sha256"}
             and isinstance(engine["path"], str) and bool(engine["path"]),
             "meta.runtime.engine 字段异常")
    engine_sha = _validate_sha256(engine["sha256"], "meta.runtime.engine.sha256")
    content = runtime["content"]
    _require(isinstance(content, dict)
             and set(content) == {"game_data", "assets"},
             "meta.runtime.content 字段异常")
    game_data = content["game_data"]
    _require(isinstance(game_data, dict)
             and set(game_data) == {"path", "sha256"}
             and isinstance(game_data["path"], str)
             and pathlib.Path(game_data["path"]).is_absolute()
             and pathlib.Path(game_data["path"]).name
             in {"DIABDAT.MPQ", "diabdat.mpq", "spawn.mpq"},
             "meta.runtime.content.game_data 字段异常")
    game_data_sha = _validate_sha256(
        game_data["sha256"], "meta.runtime.content.game_data.sha256")
    assets = content["assets"]
    _require(isinstance(assets, dict)
             and set(assets) == {"path", "sha256", "file_count"}
             and isinstance(assets["path"], str)
             and pathlib.Path(assets["path"]).is_absolute()
             and _is_int(assets["file_count"])
             and assets["file_count"] > 0,
             "meta.runtime.content.assets 字段异常")
    assets_sha = _validate_sha256(
        assets["sha256"], "meta.runtime.content.assets.sha256")
    versions = runtime["versions"]
    _require(isinstance(versions, dict)
             and set(versions) == {"python", "packages"},
             "meta.runtime.versions 字段异常")
    python_runtime = versions["python"]
    _require(isinstance(python_runtime, dict)
             and set(python_runtime) == {"implementation", "version", "cache_tag"}
             and all(isinstance(value, str) and bool(value)
                     for value in python_runtime.values()),
             "meta.runtime.versions.python 字段异常")
    packages = versions["packages"]
    _require(isinstance(packages, dict)
             and packages == RUNTIME_PACKAGE_VERSIONS,
             "meta.runtime.versions.packages 与冻结数值栈不一致")
    py_protocol = runtime["python_protocol"]
    _require(isinstance(py_protocol, dict)
             and set(py_protocol) == {"sha256", "files"}
             and isinstance(py_protocol["files"], dict) and py_protocol["files"],
             "meta.runtime.python_protocol 字段异常")
    _require(set(py_protocol["files"]) == set(PROTOCOL_SOURCE_FILES),
             "协议源码 bundle 文件集合不完整/含未知文件")
    for name in py_protocol["files"]:
        path = pathlib.PurePosixPath(name)
        _require(not path.is_absolute() and ".." not in path.parts,
                 f"协议源码路径非法: {name!r}")
    bundle_sha = _validate_sha256(
        py_protocol["sha256"], "meta.runtime.python_protocol.sha256")
    _require(source_bundle_sha256(py_protocol["files"]) == bundle_sha,
             "协议源码 bundle SHA 与逐文件 SHA 不一致")
    if expected_bridge_sha256 is not None:
        _require(bridge_sha == expected_bridge_sha256, "bridge SHA 与调用契约不一致")
    if expected_engine_sha256 is not None:
        _require(engine_sha == expected_engine_sha256,
                 "engine SHA 与调用契约不一致")
    if expected_game_data_path is not None:
        _require(game_data["path"] == expected_game_data_path,
                 "游戏主 MPQ 路径与调用契约不一致")
    if expected_game_data_sha256 is not None:
        _require(game_data_sha == expected_game_data_sha256,
                 "游戏主 MPQ SHA 与调用契约不一致")
    if expected_assets_path is not None:
        _require(assets["path"] == expected_assets_path,
                 "Resources 路径与调用契约不一致")
    if expected_assets_sha256 is not None:
        _require(assets_sha == expected_assets_sha256,
                 "Resources 树 SHA 与调用契约不一致")
    if expected_assets_file_count is not None:
        _require(assets["file_count"] == expected_assets_file_count,
                 "Resources 文件数与调用契约不一致")
    if expected_runtime_versions is not None:
        _require(versions == expected_runtime_versions,
                 "Python/数值依赖版本与调用契约不一致")
    if expected_protocol_bundle_sha256 is not None:
        _require(bundle_sha == expected_protocol_bundle_sha256,
                 "协议源码 bundle SHA 与调用契约不一致")

    _validate_identity(meta["worker"], "worker", bundle_sha)
    _validate_identity(meta["manager"], "manager", bundle_sha)
    if expected_worker_kind is not None:
        _require(meta["worker"]["kind"] == expected_worker_kind,
                 "worker 类型与调用输入不一致")
    if expected_worker_sha256 is not None:
        _require(meta["worker"]["sha256"] == expected_worker_sha256,
                 "worker SHA 与调用输入不一致（疑似复制/错标档案）")
    if expected_worker_gate_report_sha256 is not None:
        _require(meta["worker"]["gate_report_sha256"]
                 == expected_worker_gate_report_sha256,
                 "worker gate report SHA 与调用输入不一致")
    if expected_manager_kind is not None:
        _require(meta["manager"]["kind"] == expected_manager_kind,
                 "manager 类型与调用输入不一致")
    if expected_manager_sha256 is not None:
        _require(meta["manager"]["sha256"] == expected_manager_sha256,
                 "manager SHA 与调用输入不一致（疑似复制/错标档案）")
    if expected_worker_num_timesteps is not None:
        _require(meta["worker"]["num_timesteps"] == expected_worker_num_timesteps,
                 "worker num_timesteps 与调用输入不一致")

    rows = _validate_rows(document["rows"], seeds)
    _validate_agg(document["agg"], rows, meta["worker"]["kind"])
    return document


def strict_json_loads(payload: str | bytes) -> Any:
    def reject_constant(value: str) -> None:
        raise EvalContractError(f"JSON 含非标准/非有限常量: {value}")

    def reject_duplicate_keys(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise EvalContractError(f"JSON 对象含重复键: {key!r}")
            result[key] = value
        return result

    try:
        return json.loads(payload, parse_constant=reject_constant,
                          object_pairs_hook=reject_duplicate_keys)
    except json.JSONDecodeError as exc:
        raise EvalContractError(f"评测 JSON 不可解析: {exc}") from exc


def read_eval_archive(path: str | pathlib.Path, *,
                      trusted_legacy_sha256: str | None = None,
                      **expected: Any) -> dict[str, Any]:
    """读取并校验档案；legacy 只按调用者给出的完整文件 SHA 显式放行。"""
    archive_path = pathlib.Path(path)
    payload = archive_path.read_bytes()
    document = strict_json_loads(payload)
    if isinstance(document, dict) and document.get("schema_version") == SCHEMA_VERSION:
        return validate_eval_archive(document, **expected)
    _validate_sha256(trusted_legacy_sha256, "trusted_legacy_sha256")
    actual = hashlib.sha256(payload).hexdigest()
    _require(actual == trusted_legacy_sha256,
             f"legacy 档案 SHA 不匹配: {actual} != {trusted_legacy_sha256}")
    _require(isinstance(document, dict) and set(document) == {"agg", "rows"},
             "legacy 档案基本结构异常")
    return document


@contextlib.contextmanager
def exclusive_lock(lock_path: str | pathlib.Path, purpose: str = "资源"):
    """获取一个持久 lock 文件的跨进程、非阻塞独占锁。"""
    lock_path = pathlib.Path(lock_path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise OutputReservationError(
                f"{purpose}正由另一评测进程占用: {lock_path}") from exc
        os.ftruncate(descriptor, 0)
        os.write(descriptor, f"pid={os.getpid()}\n".encode("ascii"))
        os.fsync(descriptor)
        yield lock_path
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


@contextlib.contextmanager
def reserve_output(path: str | pathlib.Path):
    """用持久 lock 文件对同一目标做跨进程、非阻塞独占预约。"""
    output = pathlib.Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    lock_path = output.with_name(f".{output.name}.lock")
    with exclusive_lock(lock_path, "同 tag"):
        if output.exists():
            raise OutputReservationError(f"评测档案已存在，拒绝覆写: {output}")
        yield output

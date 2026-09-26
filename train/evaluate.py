"""32-seed standard evaluation (the protocol version comes from eval_contract.PROTOCOL_VERSION).

Protocol (every condition is part of the result; drop any one and results are not comparable):
  - the seed set is fixed at 9000-9031, used only for final evaluation, never for training or tuning;
  - deterministic argmax policy, max_steps=1500, ticks_per_step=4;
  - the engine source is pinned to ENGINE_REF in bootstrap.sh (a new engine version means rebuilding the whole leaderboard);
  - run on an idle machine: the engine advances game turns off the real wall clock (nthread_has_500ms_passed),
    so under heavy load an occasional tick advances one logic turn too few and the trajectory drifts. Measured 2026-07-05:
    idle, 4 evaluations across processes were bit-identical per seed; with training running on the same machine the median once drifted by 0.5.

Lesson: with 8 seeds, luck once overstated run6/run8 by 77%/57% (15.6->8.8, 13.2->8.4).

Usage (from the repository root):
  .venv/bin/python train/evaluate.py train/runs/<run>/model_final
  RecurrentPPO/MaskablePPO and custom feature extractors are detected automatically; results go to the current
  version's leaderboard-v<PROTOCOL_VERSION>.md, and old boards are kept as read-only history.
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
        raise ValueError("leaderboard provenance marker is not valid base64url") from exc
    return strict_json_loads(raw)


def contract_sha256(contract: Mapping) -> str:
    return hashlib.sha256(_canonical_json(contract)).hexdigest()


def _standalone_sources(root: pathlib.Path,
                        source_files: tuple[str, ...]) -> dict:
    if not source_files or len(source_files) != len(set(source_files)):
        raise ValueError("standalone source_files contract is malformed")
    files = {}
    for relative in source_files:
        path = pathlib.PurePosixPath(relative)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"illegal standalone source path: {relative!r}")
        source = root / relative
        if not source.is_file():
            raise ValueError(f"standalone protocol source missing: {source}")
        files[relative] = sha256_file(source)
    return {"sha256": source_bundle_sha256(files), "files": files}


def freeze_standalone_contract(*, evaluator: str, protocol: Mapping,
                               source_files: tuple[str, ...],
                               root: pathlib.Path = ROOT) -> dict:
    """Freeze the standalone evaluator's binary, content, source and protocol identity."""
    root = root.resolve()
    if not evaluator:
        raise ValueError("standalone evaluator name must not be empty")
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
            f"{evaluator} must run in a fresh process that has not preloaded the diablogym bridge")


def verify_loaded_native_runtime(contract: Mapping,
                                 root: pathlib.Path = ROOT) -> None:
    """Check the mapped extension path and every frozen runtime input."""
    native = sys.modules.get("_diablogym")
    if native is None:
        raise RuntimeError("no mapped _diablogym bridge found after importing the evaluation environment")
    try:
        expected = contract["runtime"]
        expected_path = pathlib.Path(expected["bridge"]["path"]).resolve()
        actual_path = pathlib.Path(native.__file__).resolve()
        content = expected["content"]
        data_dir = pathlib.Path(content["game_data"]["path"]).parent
        assets_dir = pathlib.Path(content["assets"]["path"])
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        raise ValueError("standalone contract native runtime structure is malformed") from exc
    if actual_path != expected_path:
        raise RuntimeError(
            f"actually loaded bridge path differs from the frozen identity: {actual_path} != {expected_path}")
    loaded_engine_binary_path(expected["engine"]["path"])
    current = runtime_identity(
        root.resolve(), actual_path, data_dir=data_dir, assets_dir=assets_dir)
    if current != expected:
        raise RuntimeError(
            "bridge/engine/content/dependency versions/protocol source changed during native import")


def verify_standalone_contract(contract: Mapping,
                               root: pathlib.Path = ROOT) -> None:
    """Re-hash all runtime/content/source after the run; refuse to publish on any drift."""
    try:
        runtime = contract["runtime"]
        content = runtime["content"]
        data_dir = pathlib.Path(content["game_data"]["path"]).parent
        assets_dir = pathlib.Path(content["assets"]["path"])
        source_names = tuple(contract["sources"]["files"])
        expected_protocol = contract["protocol"]
        evaluator = contract["evaluator"]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("standalone contract structure is malformed") from exc
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
            "protocol/source/bridge/engine/MPQ/Resources drifted during standalone evaluation")


def verify_checkpoint_identity(path: str | pathlib.Path, expected_sha256: str) -> None:
    checkpoint = pathlib.Path(path)
    try:
        actual = sha256_file(checkpoint)
    except OSError as exc:
        raise RuntimeError(f"checkpoint unreadable after evaluation: {checkpoint}") from exc
    if actual != expected_sha256:
        raise RuntimeError(
            f"checkpoint changed during evaluation: {actual} != {expected_sha256}")


def main_contract() -> dict:
    return freeze_standalone_contract(
        evaluator=f"standalone-main-v{PROTOCOL_VERSION}", protocol=MAIN_PROTOCOL,
        source_files=MAIN_SOURCE_FILES)


def checkpoint_snapshot(model_path: str | pathlib.Path
                        ) -> tuple[pathlib.Path, bytes, str]:
    """Read the checkpoint once; the hash, type detection and SB3.load share one byte snapshot."""
    path = resolve_checkpoint_file(model_path)
    payload = path.read_bytes()
    if not payload:
        raise ValueError(f"checkpoint is empty: {path}")
    return path, payload, hashlib.sha256(payload).hexdigest()


def atomic_write_text(path: str | pathlib.Path, payload: str) -> None:
    """fsync a temp file in the same directory, then replace atomically; on error the old file stays intact."""
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
            f"protocol-v{PROTOCOL_VERSION} leaderboard data row lacks a provenance marker")
    visible = line[:match.start()].rstrip()
    provenance = _decode_marker(match.group(1))
    if not isinstance(provenance, dict):
        raise ValueError("leaderboard row provenance must be an object")
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
        raise ValueError("leaderboard row provenance fields/types are malformed")
    if (provenance["schema_version"] != 1
            or provenance["protocol_version"] != PROTOCOL_VERSION
            or provenance["contract_sha256"] != contract_sha256(contract)
            or provenance["row_sha256"]
            != hashlib.sha256(visible.encode("utf-8")).hexdigest()
            or provenance["row_key"] != _markdown_row_key(visible)):
        raise ValueError("leaderboard row provenance disagrees with the visible row/global contract")
    if kind == "model":
        if (not isinstance(provenance["model_path"], str)
                or not pathlib.Path(provenance["model_path"]).is_absolute()
                or _SHA256_RE.fullmatch(provenance["model_sha256"]) is None
                or not isinstance(provenance["mode"], str)
                or not provenance["mode"]):
            raise ValueError("model leaderboard row identity is malformed")
    elif kind == "scripted_ref":
        if (not isinstance(provenance["policy"], str) or not provenance["policy"]
                or not isinstance(provenance["oracle_path"], str)
                or not pathlib.Path(provenance["oracle_path"]).is_absolute()
                or _SHA256_RE.fullmatch(provenance["oracle_sha256"]) is None
                or _SHA256_RE.fullmatch(provenance["result_sha256"]) is None):
            raise ValueError("scripted reference row identity is malformed")
    else:
        if (not isinstance(provenance["archive_path"], str)
                or not pathlib.Path(provenance["archive_path"]).is_absolute()
                or any(not isinstance(provenance[key], str)
                       or _SHA256_RE.fullmatch(provenance[key]) is None
                       for key in ("archive_sha256", "worker_sha256",
                                   "manager_sha256"))):
            raise ValueError("assembled-agent leaderboard row identity is malformed")
    return provenance


def _is_data_row(line: str) -> bool:
    key = _markdown_row_key(line)
    return (key is not None and key != "run"
            and not set(key) <= {"-", ":"})


def _validate_board_text(text: str, contract: Mapping,
                         initial_text: str) -> None:
    if not text.endswith("\n"):
        raise ValueError(
            f"protocol-v{PROTOCOL_VERSION} leaderboard must end with a complete newline")
    lines = text.splitlines()
    nonempty = [line for line in lines if line.strip()]
    if not nonempty or nonempty[0] != _contract_marker(contract):
        raise ValueError(
            "leaderboard lacks a matching "
            f"protocol-v{PROTOCOL_VERSION} global contract; old boards are read-only, create/"
            f"rebuild a v{PROTOCOL_VERSION} board instead")
    if sum(line.startswith(f"<!-- {_CONTRACT_MARKER}") for line in lines) != 1:
        raise ValueError("leaderboard global contract marker count is wrong")
    seen_keys = set()
    table_headers = 0
    for line in lines:
        key = _markdown_row_key(line)
        if key is None or set(key) <= {"-", ":"}:
            continue
        if key == "run":
            table_headers += 1
            if _ROW_MARKER in line:
                raise ValueError("reserved leaderboard key run cannot be a data row")
            continue
        if key in seen_keys:
            raise ValueError(f"leaderboard has a duplicate run key: {key!r}")
        seen_keys.add(key)
        _validate_row_marker(line, contract)
    if table_headers != 1:
        raise ValueError("leaderboard must contain exactly one run header")

    # The global contract describes only the evaluation regime; the header defines the meaning of the visible columns. After removing all
    # verified data rows the board must restore the template byte for byte, so hand-edited column names/order cannot keep mixing in new results.
    skeleton = "".join(
        line for line in text.splitlines(keepends=True)
        if not _is_data_row(line))
    expected = _contract_marker(contract) + "\n\n" + initial_text
    if skeleton != expected:
        raise ValueError(
            f"protocol-v{PROTOCOL_VERSION} leaderboard header/column protocol changed; "
            "create/rebuild a new board")


def ensure_leaderboard_compatible(path: str | pathlib.Path,
                                  contract: Mapping, *,
                                  initial_text: str) -> None:
    target = pathlib.Path(path)
    if target.exists():
        _validate_board_text(
            target.read_text(encoding="utf-8"), contract, initial_text)


def versioned_row_key(label: str, identity_sha256: str) -> str:
    """Derive a stable short key from the content identity; the full SHA stays in the row provenance."""
    if (not isinstance(label, str) or not label.strip()
            or any(ch in label for ch in "|\r\n")):
        raise ValueError(f"illegal leaderboard label: {label!r}")
    if _SHA256_RE.fullmatch(identity_sha256) is None:
        raise ValueError("a leaderboard version key needs a full lowercase SHA-256")
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
    """Deduplicate, read-modify-write and atomically commit the board, only within the same v3 global contract."""
    target = pathlib.Path(path)
    for key, row in rows.items():
        if (not key or key == "run" or set(key) <= {"-", ":"}
                or any(ch in key for ch in "|\r\n")):
            raise ValueError(f"illegal leaderboard row key: {key!r}")
        if "\n" in row.rstrip("\n") or _markdown_row_key(row) != key:
            raise ValueError(f"leaderboard row does not match its key: {key!r}")
        _validate_row_marker(row, contract)
    lock = (pathlib.Path(lock_path) if lock_path is not None
            else target.with_name(f".{target.name}.lock"))
    with exclusive_lock(lock, f"{target.name} leaderboard"):
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
                    f"leaderboard key {key!r} is already bound to a different result; refusing to silently overwrite the old row")
            if old is None:
                pending.append(row)
        try:
            last_row = max(i for i, line in enumerate(lines) if line.startswith("|"))
        except ValueError as exc:
            raise ValueError(f"leaderboard lacks a Markdown table: {target}") from exc
        for row in pending:
            lines.insert(last_row + 1, row if row.endswith("\n") else row + "\n")
            last_row += 1
        # While holding the board lock, do one last re-hash right before the commit, closing the swap window between evaluate() returning
        # and os.replace. Re-verify only the rows of this request; the other old rows are historical evidence
        # and do not depend on the original model still being at its original path.
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
                    raise RuntimeError("oracle unreadable before publishing") from exc
                if oracle_sha != provenance["oracle_sha256"]:
                    raise RuntimeError("probe oracle changed before publishing")
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
        raise RuntimeError(f"seed {seed} final info identity is malformed")
    required = {"kills", "depth", "died"}
    if not isinstance(extra, dict) or not required <= set(extra):
        raise RuntimeError(f"seed {seed} lacks a complete episode_extra")
    kills, depth, died = extra["kills"], extra["depth"], extra["died"]
    if (not isinstance(kills, int) or isinstance(kills, bool) or kills < 0
            or not isinstance(depth, int) or isinstance(depth, bool) or depth < 1
            or not isinstance(died, bool)):
        raise RuntimeError(f"seed {seed} episode_extra type/range is malformed")
    return extra


def model_kind(model_path: str) -> str:
    """Detect the algorithm from SB3 archive metadata instead of relying on a directory name that happens to contain mask/lstm."""
    _path, payload, _digest = checkpoint_snapshot(model_path)
    return _model_kind_from_payload(payload, model_path)


def evaluate(model_path: str, recurrent: bool | None = None,
             masked: bool | None = None, *, contract: Mapping | None = None):
    require_fresh_native_runtime("evaluate.py")
    if contract is None:
        contract = main_contract()

    from diablogym import DiabloGymEnv
    import models  # noqa: F401  (registers the custom extractor; must be importable at load time)
    verify_loaded_native_runtime(contract)

    checkpoint, payload, model_sha256 = checkpoint_snapshot(model_path)
    kind = _model_kind_from_payload(payload, model_path)
    recurrent = (kind == "recurrent") if recurrent is None else recurrent
    masked = (kind == "masked") if masked is None else masked
    if recurrent and masked:
        raise ValueError("a checkpoint cannot be evaluated as both RecurrentPPO and MaskablePPO")
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
                    # The mask is part of the policy distribution: evaluating without it = a different policy
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
    kind.add_argument("--recurrent", action="store_true", help="force loading as RecurrentPPO")
    kind.add_argument("--masked", action="store_true", help="force loading as MaskablePPO")
    kind.add_argument("--ppo", action="store_true", help="force loading as plain PPO")
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
        raise RuntimeError("evaluation result is not bound to the pre-launch standalone contract")
    name = pathlib.Path(model_path).parent.name or pathlib.Path(model_path).stem
    row_key = versioned_row_key(name, r["model_sha256"])
    visible = (f"| {row_key} | {r['mean']} | {r['median']} | "
               f"{r['max']} | {r['zero']} | {r['depth2']} |")
    print(f"mean kills {r['mean']} | median {r['median']} | max {r['max']} | "
          f"zero kills {r['zero']} | reached level 2 {r['depth2']}  [{r['secs']}s]")
    line = model_leaderboard_row(
        visible, row_key=row_key, contract=contract, model_path=r["model"],
        model_sha256=r["model_sha256"], mode=r["mode"])
    upsert_leaderboard_rows(
        LEADERBOARD, {row_key: line}, contract=contract,
        initial_text=LEADERBOARD_HEADER,
        lock_path=LEADERBOARD_LOCK)
    print(f"wrote {LEADERBOARD.name}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Fail closed when a Mach-O advertises an OS older than its dependencies.

Only non-system dylibs are audited: macOS system libraries are selected by
dyld from the running OS and their SDK-side load commands are not deployment
constraints of the application.  All Homebrew/project dylibs are traversed.
"""

from __future__ import annotations

import argparse
import pathlib
import re
import subprocess
from collections import deque


_VERSION_RE = re.compile(r"^[0-9]+(?:\.[0-9]+){1,2}$")
_DEPENDENCY_RE = re.compile(r"^\s+(\S.*?)\s+\(compatibility version ")
_RPATH_RE = re.compile(r"^path (.*?) \(offset [0-9]+\)$")


def version_tuple(value: str) -> tuple[int, int, int]:
    if not _VERSION_RE.fullmatch(value):
        raise ValueError(f"非法 macOS 版本:{value!r}")
    parts = [int(part) for part in value.split(".")]
    return tuple((parts + [0, 0])[:3])


def parse_minos(load_commands: str) -> tuple[str, ...]:
    """Extract every architecture's LC_BUILD_VERSION/legacy minimum."""
    values: list[str] = []
    command: str | None = None
    for raw in load_commands.splitlines():
        line = raw.strip()
        if line.startswith("cmd "):
            command = line[4:]
            continue
        if command == "LC_BUILD_VERSION" and line.startswith("minos "):
            values.append(line.split(None, 1)[1])
            command = None
        elif command == "LC_VERSION_MIN_MACOSX" and line.startswith("version "):
            values.append(line.split(None, 1)[1])
            command = None
    if not values:
        raise ValueError("Mach-O 没有 macOS minimum-version load command")
    for value in values:
        version_tuple(value)
    return tuple(values)


def parse_rpaths(load_commands: str) -> tuple[str, ...]:
    values: list[str] = []
    in_rpath = False
    for raw in load_commands.splitlines():
        line = raw.strip()
        if line == "cmd LC_RPATH":
            in_rpath = True
        elif in_rpath and line.startswith("path "):
            match = _RPATH_RE.fullmatch(line)
            if not match:
                raise ValueError(f"无法解析 LC_RPATH:{line!r}")
            values.append(match.group(1))
            in_rpath = False
        elif line.startswith("cmd "):
            in_rpath = False
    return tuple(values)


def parse_dependencies(linked_libraries: str) -> tuple[str, ...]:
    values: list[str] = []
    for line in linked_libraries.splitlines()[1:]:
        match = _DEPENDENCY_RE.match(line)
        if match:
            values.append(match.group(1))
    return tuple(values)


def _otool(flag: str, path: pathlib.Path) -> str:
    result = subprocess.run(
        ["/usr/bin/otool", flag, str(path)], text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"otool {flag} {path} 失败:{result.stderr.strip()}")
    return result.stdout


def _is_system_library(name: str) -> bool:
    return name.startswith("/System/Library/") or name.startswith("/usr/lib/")


def _expand_loader_token(value: str, loader: pathlib.Path) -> pathlib.Path | None:
    if value.startswith("@loader_path/"):
        return loader.parent / value.removeprefix("@loader_path/")
    if value.startswith("@executable_path/"):
        # A Python extension has no stable executable directory.  Known/search
        # candidates below can still resolve the install-name by basename.
        return None
    return pathlib.Path(value) if value.startswith("/") else None


def _resolve_dependency(
    name: str,
    loader: pathlib.Path,
    rpaths: tuple[str, ...],
    known: dict[str, set[pathlib.Path]],
) -> pathlib.Path:
    direct = _expand_loader_token(name, loader)
    if direct is not None and direct.is_file():
        return direct.resolve()
    if name.startswith("@rpath/"):
        suffix = name.removeprefix("@rpath/")
        for rpath in rpaths:
            base = _expand_loader_token(rpath, loader)
            if base is not None:
                candidate = base / suffix
                if candidate.is_file():
                    return candidate.resolve()
    matches = sorted(known.get(pathlib.Path(name).name, set()))
    if len(matches) == 1:
        return matches[0].resolve()
    if len(matches) > 1:
        raise RuntimeError(f"依赖 {name} 有多个候选:{matches}")
    raise RuntimeError(f"无法解析 {loader} 的非系统依赖:{name}")


def audit(
    binaries: list[pathlib.Path],
    deployment_target: str,
    search_roots: list[pathlib.Path],
) -> list[tuple[pathlib.Path, tuple[str, ...]]]:
    requested = version_tuple(deployment_target)
    known: dict[str, set[pathlib.Path]] = {}
    for path in binaries:
        known.setdefault(path.name, set()).add(path.resolve())
    for root in search_roots:
        for pattern in ("*.dylib", "*.so"):
            for path in root.rglob(pattern):
                known.setdefault(path.name, set()).add(path.resolve())

    queue = deque((path.resolve(), True) for path in binaries)
    visited: set[pathlib.Path] = set()
    audited: list[tuple[pathlib.Path, tuple[str, ...]]] = []
    while queue:
        path, is_root = queue.popleft()
        if path in visited:
            continue
        visited.add(path)
        if not path.is_file():
            raise RuntimeError(f"Mach-O 不存在:{path}")
        load_commands = _otool("-l", path)
        minimums = parse_minos(load_commands)
        if is_root and any(version_tuple(value) != requested for value in minimums):
            raise RuntimeError(
                f"{path} minos={minimums}，不等于请求的 {deployment_target}")
        if any(version_tuple(value) > requested for value in minimums):
            raise RuntimeError(
                f"{path} minos={minimums} 高于产物目标 {deployment_target}")
        audited.append((path, minimums))

        rpaths = parse_rpaths(load_commands)
        for dependency in parse_dependencies(_otool("-L", path)):
            if _is_system_library(dependency):
                continue
            resolved = _resolve_dependency(dependency, path, rpaths, known)
            if resolved != path:
                queue.append((resolved, False))
    return audited


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--deployment-target", required=True)
    parser.add_argument("--search-root", action="append", default=[])
    parser.add_argument("binary", nargs="+")
    args = parser.parse_args()
    try:
        audited = audit(
            [pathlib.Path(value) for value in args.binary],
            args.deployment_target,
            [pathlib.Path(value) for value in args.search_root],
        )
    except (OSError, RuntimeError, ValueError) as exc:
        parser.exit(1, f"macOS minos 审计失败: {exc}\n")
    summary = ", ".join(
        f"{path.name}:{'/'.join(minimums)}" for path, minimums in audited)
    print(f"macOS minos 审计通过({args.deployment_target}): {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

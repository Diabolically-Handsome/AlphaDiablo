#!/usr/bin/env python3
"""Build/ABI contract tests that do not require compiling the engine."""

from __future__ import annotations

import importlib.util
import pathlib
import tempfile
import unittest
from unittest import mock


ROOT = pathlib.Path(__file__).resolve().parents[1]
AUDIT_PATH = ROOT / "cmake" / "audit_macos_minos.py"
SPEC = importlib.util.spec_from_file_location("audit_macos_minos", AUDIT_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"无法载入 {AUDIT_PATH}")
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


class BuildContractTests(unittest.TestCase):
    def test_minos_parses_modern_and_legacy_multiarch_commands(self):
        text = """
Load command 1
      cmd LC_BUILD_VERSION
  cmdsize 32
 platform 1
    minos 15.5
Load command 2
      cmd LC_VERSION_MIN_MACOSX
  cmdsize 16
  version 12.3
      sdk 13.1
"""
        self.assertEqual(AUDIT.parse_minos(text), ("15.5", "12.3"))
        self.assertEqual(AUDIT.version_tuple("15.5"), (15, 5, 0))
        with self.assertRaises(ValueError):
            AUDIT.version_tuple("15")

    def test_rpaths_and_dependencies_are_parsed_without_install_id_loss(self):
        load = """
Load command 4
          cmd LC_RPATH
      cmdsize 88
         path @loader_path/My Engine (offset 12)
"""
        linked = """bundle.so:
\t@rpath/libengine.dylib (compatibility version 0.0.0, current version 0.0.0)
\t/usr/lib/libSystem.B.dylib (compatibility version 1.0.0, current version 1.0.0)
"""
        self.assertEqual(AUDIT.parse_rpaths(load), ("@loader_path/My Engine",))
        self.assertEqual(
            AUDIT.parse_dependencies(linked),
            ("@rpath/libengine.dylib", "/usr/lib/libSystem.B.dylib"),
        )

    def test_build_files_pin_modern_findpython_and_post_link_audit(self):
        cmake = (ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
        build = (ROOT / "build.sh").read_text(encoding="utf-8")
        self.assertIn("set(PYBIND11_FINDPYTHON ON)", cmake)
        self.assertIn(
            "find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)",
            cmake,
        )
        self.assertIn("ALPHADIABLO_EXPECTED_PYTHON_EXECUTABLE", cmake)
        self.assertIn("ALPHADIABLO_EXPECTED_PYTHON_INCLUDE_DIR", cmake)
        self.assertIn("ALPHADIABLO_EXPECTED_PYTHON_EXT_SUFFIX", build)
        self.assertIn("audit_macos_minos.py", build)
        self.assertIn("BOOTSTRAP_CLEAN=1 ./bootstrap.sh", build)
        self.assertIn("libdevilutionx_so devilutionx", cmake)
        self.assertNotIn('set(CMAKE_OSX_DEPLOYMENT_TARGET "11.0"', cmake)

    def test_audit_rejects_a_newer_non_system_dependency(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp) / "bridge.so"
            dependency = pathlib.Path(tmp) / "libengine.dylib"
            root.touch()
            dependency.touch()
            loads = {
                ("-l", root.resolve()): "cmd LC_BUILD_VERSION\nminos 15.0\n",
                ("-L", root.resolve()): (
                    f"{root}:\n\t{dependency} (compatibility version 0.0.0, "
                    "current version 0.0.0)\n"),
                ("-l", dependency.resolve()):
                    "cmd LC_BUILD_VERSION\nminos 16.0\n",
                ("-L", dependency.resolve()): f"{dependency}:\n",
            }

            def fake_otool(flag, path):
                return loads[(flag, pathlib.Path(path).resolve())]

            with mock.patch.object(AUDIT, "_otool", side_effect=fake_otool):
                with self.assertRaisesRegex(RuntimeError, "高于产物目标"):
                    AUDIT.audit([root], "15.0", [pathlib.Path(tmp)])


if __name__ == "__main__":
    unittest.main()

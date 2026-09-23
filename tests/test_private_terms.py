"""Private-term guard: the repository stays clean and the scanner still bites.

The real terms are not stored in the repository, not even as hashes.  The
guard reads their SHA-256 from $PRIVATE_TERM_SHA256 (a CI secret) or from
the file named by $PRIVATE_TERMS_FILE; without either, the repository scan
below is skipped (reported as a skip, not a pass).  All other tests use
random stand-in terms generated at run time.
"""
from __future__ import annotations

import base64
import bz2
import gzip
import importlib.util
import io
import lzma
import os
import pathlib
import secrets
import shutil
import string
import struct
import subprocess
import sys
import tempfile
import unittest
import zipfile
import zlib
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "check_private_terms", ROOT / "tools" / "check_private_terms.py")
guard = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = guard
SPEC.loader.exec_module(guard)

ENV_KEYS = (guard.ENV_HASHES, guard.ENV_FILE)


def _word(n: int = 9) -> str:
    return "".join(secrets.choice(string.ascii_lowercase) for _ in range(n))


def _digest() -> str:
    return secrets.token_hex(32)


def _env(**values):
    """os.environ without the guard's variables, plus `values`."""
    env = {k: v for k, v in os.environ.items() if k not in ENV_KEYS}
    env.update(values)
    return mock.patch.dict(os.environ, env, clear=True)


class RepositoryIsCleanTest(unittest.TestCase):
    def setUp(self):
        self.hashes = guard.load_hashes()
        if not self.hashes:
            self.skipTest("PRIVATE_TERM_SHA256 not configured")

    def test_tracked_files_contain_no_private_term(self):
        hits = guard.scan_paths(guard.tracked_files(), self.hashes)
        self.assertEqual([str(h) for h in hits], [])

    def test_configured_hashes_are_well_formed(self):
        for digest in self.hashes:
            self.assertRegex(digest, r"^[0-9a-f]{64}$")


class LoadHashesTest(unittest.TestCase):
    def test_nothing_configured(self):
        self.assertEqual(guard.load_hashes({}), frozenset())
        self.assertEqual(guard.load_hashes({guard.ENV_HASHES: "  \n"}), frozenset())

    def test_environment_variable(self):
        a, b, c = _digest(), _digest(), _digest()
        got = guard.load_hashes({guard.ENV_HASHES: f"{a}, {b.upper()}\n{c}\n"})
        self.assertEqual(got, frozenset({a, b, c}))

    def test_file_with_comments(self):
        a, b = _digest(), _digest()
        with tempfile.TemporaryDirectory() as folder:
            path = pathlib.Path(folder) / "terms.sha256"
            path.write_text(f"# private terms\n{a}  # first\n\n{b}\n", encoding="utf-8")
            got = guard.load_hashes({guard.ENV_FILE: str(path),
                                     guard.ENV_HASHES: a})
        self.assertEqual(got, frozenset({a, b}))

    def test_bad_entries_are_rejected_without_echo(self):
        term = _word()
        with self.assertRaises(ValueError) as caught:
            guard.load_hashes({guard.ENV_HASHES: f"{_digest()} {term}"})
        self.assertNotIn(term, str(caught.exception))
        with self.assertRaises(ValueError):
            guard.load_hashes({guard.ENV_FILE: str(ROOT / "no-such-file.sha256")})


class ScannerTest(unittest.TestCase):
    def setUp(self):
        self.first, self.last = _word(), _word(7)
        self.hashes = frozenset({
            guard.term_digest(f"{self.first} {self.last}"),
            guard.term_digest(self.first + self.last),
        })

    def hits(self, data: bytes, path: str = "sample.bin"):
        return guard.scan_bytes(data, path, self.hashes)

    def assertCaught(self, data: bytes, path: str = "sample.bin"):
        self.assertTrue(self.hits(data, path), "stand-in term was not caught")

    def test_term_digest_normalises_case_and_space(self):
        self.assertEqual(guard.term_digest(f"  {self.first.upper()} {self.last} "),
                         guard.term_digest(f"{self.first} {self.last}"))

    def test_plain_text_forms(self):
        f, l = self.first, self.last
        for text in (f"Copyright (c) 2026 {f.title()} {l.title()}\n",
                     f"authors = [{{ name = \"{f} {l}\" }}]\n",
                     f'"path": "/Users/{f}{l}/Desktop/run/tb"\n',
                     f"C:\\Users\\{f.title()}{l.title()}\\AppData\n",
                     f"owner={f.title()}{l.title()}Id\n",
                     f"{f.title()} Q. {l.title()}\n",
                     f"{f}_{l}\n"):
            with self.subTest(text=text):
                self.assertCaught(text.encode())

    def test_text_hits_carry_line_numbers(self):
        data = f"one\ntwo\n/home/{self.first}{self.last}/x\n".encode()
        self.assertEqual({h.line for h in self.hits(data, "a.txt")}, {3})

    def test_escaped_forms(self):
        term = f"{self.first} {self.last}"
        for label, text in (
                ("percent", "".join(f"%{ord(c):02X}" for c in term)),
                ("percent space", f"q={self.first}%20{self.last}&x=1"),
                ("json \\u", "".join(f"\\u{ord(c):04x}" for c in term)),
                ("python \\x", "".join(f"\\x{ord(c):02x}" for c in term)),
                ("html decimal", "".join(f"&#{ord(c)};" for c in term)),
                ("html hex", "".join(f"&#x{ord(c):x};" for c in term)),
                ("html named", f"<b>{self.first}&nbsp;{self.last}</b>")):
            with self.subTest(label=label):
                self.assertCaught(text.encode())

    def test_encoded_forms(self):
        raw = f"/Users/{self.first}{self.last}/Desktop/AlphaDiablo".encode()
        std = base64.b64encode(b'{"path": "' + raw + b'"}')
        url = base64.urlsafe_b64encode(b'{"path": "' + raw + b'"}??>>')
        for label, data in (
                ("base64", b"<!-- marker:" + std + b" -->"),
                ("base64url", b"<!-- marker:" + url + b" -->"),
                ("base32", b"id=" + base64.b32encode(raw)),
                ("base32 lower", b"id=" + base64.b32encode(raw).lower().rstrip(b"=")),
                ("hex", raw.hex().encode()),
                ("utf-16le", b"\x00\x01" + raw.decode().encode("utf-16-le")),
                ("utf-16be", b"\x01" + raw.decode().encode("utf-16-be")),
                ("gzip", gzip.compress(raw)),
                ("zlib", zlib.compress(raw)),
                ("bzip2", bz2.compress(raw)),
                ("xz", lzma.compress(raw))):
            with self.subTest(label=label):
                self.assertCaught(data)

    def test_deep_nesting(self):
        data = f"{self.first} {self.last}".encode()
        for layer in range(guard.MAX_DEPTH - 1):
            data = (base64.b64encode(data) if layer % 2 == 0
                    else zlib.compress(data))
        self.assertCaught(data)

    def test_png_text_chunks(self):
        def chunk(kind: bytes, body: bytes) -> bytes:
            crc = zlib.crc32(kind + body) & 0xFFFFFFFF
            return struct.pack(">I", len(body)) + kind + body + struct.pack(">I", crc)
        text = f"Author\0\0{self.first} {self.last}".encode()
        png = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", bytes(13))
               + chunk(b"zTXt", b"Comment\0\0" + zlib.compress(text))
               + chunk(b"IEND", b""))
        self.assertCaught(png)

    def test_zip_members_and_names(self):
        raw = f"tensorboard_log: /Users/{self.first}{self.last}/tb".encode()
        for compression in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED):
            with self.subTest(compression=compression):
                buf = io.BytesIO()
                with zipfile.ZipFile(buf, "w", compression) as zf:
                    zf.writestr("data", raw)
                self.assertCaught(buf.getvalue(), "model.zip")
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr(f"{self.first}-{self.last}/notes.txt", b"nothing here")
        self.assertCaught(buf.getvalue(), "bundle.zip")

    def test_nested_zip_inside_base64(self):
        inner = io.BytesIO()
        with zipfile.ZipFile(inner, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("x.txt", f"{self.first} {self.last}")
        self.assertCaught(b"blob=" + base64.b64encode(inner.getvalue()))

    def test_initial_and_surname(self):
        initial = self.first[0]
        hashes = frozenset({guard.term_digest(f"{initial} {self.last}")})
        for text in (f"{initial.upper()}. {self.last.title()}",
                     f"by {initial.upper()} {self.last.title()}, 2026"):
            with self.subTest(text=text):
                self.assertTrue(guard.scan_bytes(text.encode(), "a.txt", hashes))

    def test_paths_are_checked_and_masked(self):
        for path in (f"docs/{self.first}{self.last}/readme.md",
                     f"docs/{self.first}%20{self.last}/readme.md"):
            with self.subTest(path=path):
                hits = self.hits(b"clean content", path)
                self.assertTrue(hits)
                for hit in hits:
                    self.assertNotIn(self.first, str(hit).lower())
                    self.assertNotIn(self.last, str(hit).lower())

    def test_zip_member_names_are_masked(self):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr(f"{self.first}{self.last}.txt", b"x")
        hits = self.hits(buf.getvalue(), "a.zip")
        self.assertTrue(hits)
        for hit in hits:
            self.assertNotIn(self.first + self.last, str(hit).lower())

    def test_reports_never_print_the_digest(self):
        hits = self.hits(f"{self.first} {self.last}\n".encode(), "a.txt")
        self.assertTrue(hits)
        for hit in hits:
            for digest in self.hashes:
                self.assertNotIn(digest[:8], str(hit))
            self.assertIn("private term #", str(hit))

    def test_near_misses_are_not_hits(self):
        f, l = self.first, self.last
        for text in (f"{f[:-1]} {l}", f"{f}x{l}", f"{l} {f}", f"{f}\n\n",
                     f"{f}ab {l}", f"{f[1:]} {l}"):
            with self.subTest(text=text):
                self.assertEqual(self.hits(text.encode()), [])


@unittest.skipUnless(shutil.which("git"), "git not available")
class CommitScanTest(unittest.TestCase):
    def setUp(self):
        self.term = f"{_word()} {_word(7)}"
        self.hashes = frozenset({guard.term_digest(self.term)})
        self.tmp = tempfile.TemporaryDirectory()
        self.repo = pathlib.Path(self.tmp.name)
        self.git("init", "-q")

    def tearDown(self):
        self.tmp.cleanup()

    def git(self, *args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(self.repo), "-c", "user.name=t",
             "-c", "user.email=t@example.invalid", "-c", "commit.gpgsign=false",
             *args], check=True, capture_output=True, text=True).stdout.strip()

    def commit(self, name: str, content: str, message: str = "add") -> str:
        (self.repo / name).write_text(content, encoding="utf-8")
        self.git("add", name)
        self.git("commit", "-q", "-m", message)
        return self.git("rev-parse", "HEAD")

    def test_tree_and_message_are_scanned(self):
        clean = self.commit("a.txt", "nothing here\n")
        dirty = self.commit("b.txt", f"by {self.term}\n")
        noted = self.commit("c.txt", "fine\n", f"thanks {self.term}")
        self.assertEqual(guard.scan_rev(clean, self.hashes, self.repo)[1], [])
        self.assertTrue(guard.scan_rev(dirty, self.hashes, self.repo)[1])
        self.git("rm", "-q", "b.txt")
        self.git("commit", "-q", "-m", "remove")
        head = self.git("rev-parse", "HEAD")
        self.assertEqual(guard.scan_rev(head, self.hashes, self.repo)[1], [])
        hits = guard.scan_rev(noted, self.hashes, self.repo)[1]
        self.assertTrue(any("<commit" in str(h) for h in hits))


class MainTest(unittest.TestCase):
    def test_exit_status(self):
        term = f"{_word()} {_word(7)}"
        with tempfile.TemporaryDirectory() as folder:
            dirty = pathlib.Path(folder) / "dirty.txt"
            dirty.write_text(f"signed: {term}\n", encoding="utf-8")
            clean = ROOT / "LICENSE"
            quiet = mock.patch("sys.stdout", new_callable=io.StringIO)
            with quiet, mock.patch("sys.stderr", new_callable=io.StringIO):
                with _env(**{guard.ENV_HASHES: guard.term_digest(term)}):
                    self.assertEqual(guard.main([str(clean)]), 0)
                    self.assertEqual(guard.main([str(dirty)]), 1)
                    self.assertEqual(guard.main(["--require", str(clean)]), 0)
                with _env():
                    self.assertEqual(guard.main([str(dirty)]), 0)
                    self.assertEqual(guard.main(["--require", str(dirty)]), 2)
                with _env(**{guard.ENV_HASHES: "not-a-digest"}):
                    self.assertEqual(guard.main([str(clean)]), 2)

    def test_skip_warns_in_github_actions(self):
        with _env(GITHUB_ACTIONS="true"), \
                mock.patch("sys.stdout", new_callable=io.StringIO) as out:
            self.assertEqual(guard.main([str(ROOT / "LICENSE")]), 0)
        self.assertIn("SKIP - no hashes configured", out.getvalue())
        self.assertIn("::warning", out.getvalue())


if __name__ == "__main__":
    unittest.main()

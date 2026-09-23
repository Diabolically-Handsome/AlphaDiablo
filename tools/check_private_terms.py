#!/usr/bin/env python3
"""Private-term guard: fail when a tracked file contains a private term.

The terms are not stored in this repository, not even as hashes: the hash
of a short name can be reversed with a list of names.  The guard reads the
SHA-256 of each term's lowercase form from outside the repository:

* PRIVATE_TERM_SHA256 - an environment variable (in CI, a repository
  secret) holding hex digests separated by spaces, commas or newlines;
* PRIVATE_TERMS_FILE  - the path of a local file with one digest per line
  ("#" starts a comment).

With neither set, the guard prints "SKIP - no hashes configured" (plus a
GitHub Actions warning in CI) and exits 0; with --require it exits 2.

Tokens are collected from:

* the raw bytes of every tracked file, text or binary, plus UTF-16LE and
  UTF-16BE strings;
* the file path itself;
* URL (%xx), \\uXXXX / \\xXX and HTML entity escapes, decoded;
* base64, base32 and hex runs, decoded and scanned again;
* zip members and member names (.zip, .npz, .pt, .docx, ...), gzip,
  bzip2, xz and zlib streams, and PNG text chunks,
  recursively up to MAX_DEPTH layers.

Each ASCII word run is lowercased and also split at camelCase and
letter/digit boundaries.  Neighbouring tokens are joined with and without
a space, so a two-word term is caught as "first last" and as "firstlast".
A single letter between two words (a middle initial) is also skipped for
the join, and a single letter joins the next word ("f last").

Usage (from anywhere inside the repository):

    python tools/check_private_terms.py              # scan all tracked files
    python tools/check_private_terms.py PATH [...]   # scan files or folders
    python tools/check_private_terms.py --rev REV    # scan a commit: tree,
                                                     # message and author
    python tools/check_private_terms.py --hash -     # read a term from stdin,
                                                     # print its digest

Exit status: 0 = clean (or no hashes configured), 1 = private term found,
2 = usage or configuration error, or no hashes configured with --require.
Hits are reported by file, line and the term's position in the sorted
hash list (never the digest).  Matching words in the reported path or zip
member name are masked, so the term is never printed.
"""
from __future__ import annotations

import argparse
import base64
import binascii
import bz2
import hashlib
import html
import io
import lzma
import os
import pathlib
import re
import subprocess
import sys
import urllib.parse
import zipfile
import zlib
from typing import NamedTuple

ROOT = pathlib.Path(__file__).resolve().parents[1]

ENV_HASHES = "PRIVATE_TERM_SHA256"
ENV_FILE = "PRIVATE_TERMS_FILE"

MAX_DEPTH = 6            # nesting limit for encoded / compressed / zip layers
MAX_MEMBER_BYTES = 256 * 1024 * 1024
SKIP_DIRS = {".git", "build", ".venv", "__pycache__", ".pytest_cache"}

_DIGEST = re.compile(r"[0-9a-f]{64}")
_ASCII_RUN = re.compile(rb"[A-Za-z0-9]+")
_TEXT_RUN = re.compile(r"[A-Za-z0-9]+")
_UTF16LE_RUN = re.compile(rb"(?:[\x20-\x7e]\x00){4,}")
_UTF16BE_RUN = re.compile(rb"(?:\x00[\x20-\x7e]){4,}")
_B64_RUN = re.compile(rb"[A-Za-z0-9+/_-]{16,}={0,2}")
_B32_RUN = re.compile(rb"(?:[A-Z2-7]{16,}|[a-z2-7]{16,})={0,6}")
_HEX_RUN = re.compile(rb"(?:[0-9a-fA-F]{2}){4,}")
_PERCENT = re.compile(rb"%[0-9A-Fa-f]{2}")
_BACKSLASH_ESC = re.compile(
    rb"\\u([0-9A-Fa-f]{4})|\\U([0-9A-Fa-f]{8})|\\x([0-9A-Fa-f]{2})")
_ENTITY = re.compile(rb"&(?:#[0-9]{1,7}|#[xX][0-9A-Fa-f]{1,6}|[A-Za-z][A-Za-z0-9]{1,31});")
_CAMEL_PART = re.compile(r"[A-Z]?[a-z]+|[A-Z]+(?![a-z])|[0-9]+")
_MIN_PART, _MIN_TOKEN, _MAX_TOKEN = 2, 4, 64
_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


class Hit(NamedTuple):
    path: str          # already masked
    line: int | None
    context: str       # already masked
    digest: str
    term: int          # 1-based position of the digest in the sorted hash list

    def __str__(self) -> str:
        # Report the term's position, not its digest: a digest (or a prefix
        # of it) printed in a public CI log could be reversed.
        where = self.path if self.line is None else f"{self.path}:{self.line}"
        ctx = f" [{self.context}]" if self.context else ""
        return f"{where}{ctx}: private term #{self.term}"


def term_digest(term: str) -> str:
    return hashlib.sha256(term.strip().lower().encode("utf-8")).hexdigest()


def _parse_digests(text: str, source: str) -> set[str]:
    found: set[str] = set()
    number = 0
    for line in text.splitlines():
        for item in re.split(r"[\s,;]+", line.split("#", 1)[0].strip()):
            if not item:
                continue
            number += 1
            item = item.lower()
            if not _DIGEST.fullmatch(item):
                # Never echo the entry itself: it may be a digest or a term.
                raise ValueError(f"{source}: entry {number} is not a SHA-256 hex digest")
            found.add(item)
    return found


def load_hashes(environ=None) -> frozenset[str]:
    """Digests from $PRIVATE_TERM_SHA256 and the file named by $PRIVATE_TERMS_FILE."""
    env = os.environ if environ is None else environ
    hashes: set[str] = set()
    raw = env.get(ENV_HASHES, "")
    if raw.strip():
        hashes |= _parse_digests(raw, ENV_HASHES)
    name = env.get(ENV_FILE, "")
    if name.strip():
        try:
            text = pathlib.Path(name).expanduser().read_text(encoding="utf-8")
        except OSError as exc:
            raise ValueError(f"{ENV_FILE}: cannot read the file ({exc.strerror})") from None
        hashes |= _parse_digests(text, ENV_FILE)
    return frozenset(hashes)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("ascii", "ignore")).hexdigest()


def _candidates(runs: list[str]) -> set[str]:
    """Lowercase word candidates, their camel parts and neighbour joins."""
    out: set[str] = set()
    previous: str | None = None
    before: str | None = None      # the word before a single-letter `previous`
    for run in runs:
        low = run.lower()
        if len(run) >= _MIN_PART:
            out.add(low)
            parts = [p.lower() for p in _CAMEL_PART.findall(run)]
            if len(parts) > 1:
                out.update(parts)
                for a, b in zip(parts, parts[1:]):
                    out.add(a + b)
                    out.add(f"{a} {b}")
        if previous is not None:
            out.add(previous + low)
            out.add(f"{previous} {low}")
            if before is not None and len(previous) == 1:
                out.add(before + low)
                out.add(f"{before} {low}")
        before = previous
        previous = low
    return {c for c in out if _MIN_TOKEN <= len(c) <= _MAX_TOKEN}


def _utf16_runs(data: bytes) -> list[str]:
    runs: list[str] = []
    for pattern, codec in ((_UTF16LE_RUN, "utf-16-le"), (_UTF16BE_RUN, "utf-16-be")):
        for m in pattern.finditer(data):
            runs.extend(_TEXT_RUN.findall(m.group().decode(codec, errors="ignore")))
    return runs


def _hashed_hits(data: bytes, hashes: frozenset[str]) -> set[str]:
    found: set[str] = set()
    ascii_runs = [m.decode("ascii") for m in _ASCII_RUN.findall(data)]
    for group in (ascii_runs, _utf16_runs(data)):
        for cand in _candidates(group):
            digest = _sha(cand)
            if digest in hashes:
                found.add(digest)
    return found


def _hashed_candidates(hashes: frozenset[str], runs: list[str]) -> set[str]:
    return {c for c in _candidates(runs) if _sha(c) in hashes}


def _mask(text: str, hashes: frozenset[str]) -> str:
    """Replace every word of `text` that forms a private term with ***."""
    runs = list(_TEXT_RUN.finditer(text))
    bad: set[int] = set()
    for i, m in enumerate(runs):
        if _hashed_candidates(hashes, [m.group()]):
            bad.add(i)
        if i and _hashed_candidates(hashes, [runs[i - 1].group(), m.group()]):
            bad.update({i - 1, i})
        if i > 1 and _hashed_candidates(
                hashes, [runs[i - 2].group(), runs[i - 1].group(), m.group()]):
            bad.update({i - 2, i - 1, i})
    if not bad:
        return text
    pieces, last = [], 0
    for i, m in enumerate(runs):
        if i in bad:
            pieces.append(text[last:m.start()] + "***")
            last = m.end()
    pieces.append(text[last:])
    return "".join(pieces)


def _decode_base64(raw: bytes) -> bytes | None:
    text = raw.rstrip(b"=")
    if len(text) < 16:
        return None
    padded = text + b"=" * (-len(text) % 4)
    try:
        if b"-" in padded or b"_" in padded:
            if b"+" in padded or b"/" in padded:
                return None
            return base64.urlsafe_b64decode(padded)
        return base64.b64decode(padded, validate=True)
    except (binascii.Error, ValueError):
        return None


def _decode_base32(raw: bytes) -> bytes | None:
    text = raw.rstrip(b"=").upper()
    if len(text) < 16 or len(text) % 8 in (1, 3, 6):
        return None
    try:
        return base64.b32decode(text + b"=" * (-len(text) % 8))
    except (binascii.Error, ValueError):
        return None


def _backslash_sub(m: re.Match) -> bytes:
    digits = m.group(1) or m.group(2) or m.group(3)
    try:
        return chr(int(digits, 16)).encode("utf-8", "ignore")
    except (ValueError, OverflowError):
        return m.group()


def _unescaped(data: bytes) -> bytes | None:
    """Decode %xx, \\uXXXX / \\xXX and HTML entities; None when nothing changes."""
    out = data
    if _PERCENT.search(out):
        out = urllib.parse.unquote_to_bytes(out)
    if _BACKSLASH_ESC.search(out):
        out = _BACKSLASH_ESC.sub(_backslash_sub, out)
    if _ENTITY.search(out):
        text = html.unescape(out.decode("utf-8", "surrogateescape"))
        try:
            out = text.encode("utf-8", "surrogateescape")
        except UnicodeEncodeError:      # an entity named a lone surrogate
            out = text.encode("utf-8", "ignore")
    return out if out != data else None


def _inflate(data: bytes, wbits: int) -> bytes:
    return zlib.decompressobj(wbits).decompress(data, MAX_MEMBER_BYTES)


def _decompressed(data: bytes) -> tuple[str, bytes] | None:
    """Decompress a gzip, bzip2, xz or zlib stream that starts at byte 0."""
    try:
        if data[:2] == b"\x1f\x8b":
            return "gzip", _inflate(data, 31)
        if data[:3] == b"BZh":
            return "bzip2", bz2.BZ2Decompressor().decompress(data, MAX_MEMBER_BYTES)
        if data[:6] == b"\xfd7zXZ\x00":
            return "xz", lzma.LZMADecompressor().decompress(data, MAX_MEMBER_BYTES)
        if (len(data) > 2 and data[0] & 0x0F == 8 and data[0] >> 4 <= 7
                and (data[0] << 8 | data[1]) % 31 == 0):
            return "zlib", _inflate(data, 15)
    except (OSError, EOFError, ValueError, zlib.error, lzma.LZMAError):
        pass
    return None


def _png_texts(data: bytes):
    """Keywords and (decompressed) text of PNG tEXt / zTXt / iTXt chunks."""
    pos = len(_PNG_MAGIC)
    while pos + 8 <= len(data):
        length = int.from_bytes(data[pos:pos + 4], "big")
        kind, body = data[pos + 4:pos + 8], data[pos + 8:pos + 8 + length]
        pos += 12 + length
        try:
            if kind == b"zTXt":
                key, _, rest = body.partition(b"\0")
                yield key + b"\n" + _inflate(rest[1:], 15)
            elif kind == b"iTXt":
                key, _, rest = body.partition(b"\0")
                compressed, rest = rest[:1] == b"\1", rest[2:]
                lang, _, rest = rest.partition(b"\0")
                label, _, text = rest.partition(b"\0")
                yield b"\n".join((key, lang, label,
                                  _inflate(text, 15) if compressed else text))
        except zlib.error:
            continue
        if kind == b"IEND":
            break


class _Scanner:
    def __init__(self, path: str, hashes: frozenset[str]):
        self.path = _mask(path, hashes)
        self.hashes = hashes
        self.order = {d: i for i, d in enumerate(sorted(hashes), start=1)}
        self.hits: list[Hit] = []

    def add(self, digests, line: int | None, context: str) -> None:
        for digest in sorted(digests):
            self.hits.append(Hit(self.path, line, _mask(context, self.hashes),
                                 digest, self.order[digest]))

    def blob(self, data: bytes, context: str, depth: int,
             line: int | None = None) -> None:
        self.add(_hashed_hits(data, self.hashes), line, context)
        if depth >= MAX_DEPTH or not data:
            return
        layer = depth + 1
        if data[:4] == b"PK\x03\x04":
            self._zip(data, context, depth)
        elif data[:8] == _PNG_MAGIC:
            for text in _png_texts(data):
                self.blob(text, f"{context}(png text)", layer)
        else:
            unpacked = _decompressed(data)
            if unpacked:
                self.blob(unpacked[1], f"{context}({unpacked[0]})", layer)
        unescaped = _unescaped(data)
        if unescaped is not None:
            self.blob(unescaped, f"{context} unescaped".strip(), layer, line)
        for m in _B64_RUN.finditer(data):
            decoded = _decode_base64(m.group())
            if decoded:
                self.blob(decoded, f"{context} base64@{m.start()}".strip(), layer, line)
        for m in _B32_RUN.finditer(data):
            decoded = _decode_base32(m.group())
            if decoded:
                self.blob(decoded, f"{context} base32@{m.start()}".strip(), layer, line)
        for m in _HEX_RUN.finditer(data):
            self.blob(binascii.unhexlify(m.group()),
                      f"{context} hex@{m.start()}".strip(), layer, line)

    def _zip(self, data: bytes, context: str, depth: int) -> None:
        try:
            with zipfile.ZipFile(io.BytesIO(data)) as archive:
                for info in archive.infolist():
                    member = f"{context}!{info.filename}" if context else info.filename
                    self.add(_hashed_hits(info.filename.encode("utf-8", "ignore"),
                                          self.hashes), None, f"{member} (name)")
                    if info.is_dir() or info.file_size > MAX_MEMBER_BYTES:
                        continue
                    self.blob(archive.read(info), member, depth + 1)
                if archive.comment:
                    self.blob(archive.comment, f"{context}(zip comment)", depth + 1)
        except (zipfile.BadZipFile, OSError, RuntimeError, ValueError, EOFError,
                NotImplementedError, zlib.error):
            pass


def scan_bytes(data: bytes, path: str, hashes: frozenset[str]) -> list[Hit]:
    """Scan one file's content (and its path) and return every hit."""
    raw_path = path.encode("utf-8", "ignore")
    unescaped_path = _unescaped(raw_path)
    path_digests = _hashed_hits(raw_path, hashes)
    if unescaped_path is not None:
        escaped_digests = _hashed_hits(unescaped_path, hashes)
        if escaped_digests:
            # Show (and mask) the decoded path, so the term is never printed
            # in its escaped form either.
            path = unescaped_path.decode("utf-8", "replace")
            path_digests |= escaped_digests
    head = _Scanner(path, hashes)
    head.add(path_digests, None, "path")
    whole = _Scanner(path, hashes)
    whole.blob(data, "", 0)
    if not whole.hits:
        return head.hits
    # Re-scan text line by line so that hits carry a line number.
    if b"\x00" not in data[:8192] and data[:4] != b"PK\x03\x04":
        lined = _Scanner(path, hashes)
        for number, line in enumerate(data.splitlines(), start=1):
            lined.blob(line, "", 0, number)
        if {h.digest for h in lined.hits} >= {h.digest for h in whole.hits}:
            return head.hits + lined.hits
    return head.hits + whole.hits


def tracked_files(root: pathlib.Path = ROOT) -> list[pathlib.Path]:
    try:
        out = subprocess.run(["git", "-C", str(root), "ls-files", "-z"],
                             check=True, capture_output=True).stdout
        names = [n for n in out.decode("utf-8").split("\0") if n]
        if names:
            return [root / n for n in names]
    except (OSError, subprocess.CalledProcessError):
        pass
    return sorted(_walk(root))


def _walk(root: pathlib.Path):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for name in filenames:
            yield pathlib.Path(dirpath) / name


def scan_paths(paths, hashes: frozenset[str],
               root: pathlib.Path = ROOT) -> list[Hit]:
    hits: list[Hit] = []
    for path in map(pathlib.Path, paths):
        if path.is_dir():
            hits.extend(scan_paths(sorted(_walk(path)), hashes, root))
            continue
        if not path.is_file():
            continue
        try:
            rel = path.resolve().relative_to(root.resolve()).as_posix()
        except ValueError:
            rel = path.as_posix()
        hits.extend(scan_bytes(path.read_bytes(), rel, hashes))
    return hits


def scan_rev(rev: str, hashes: frozenset[str], root: pathlib.Path = ROOT,
             seen: set | None = None) -> tuple[int, list[Hit]]:
    """Scan every blob in commit REV's tree, plus its message and author lines.

    `seen` collects (blob id, path) pairs; pairs already in it are skipped,
    so scanning a series of commits reads each unchanged file only once.
    """
    seen = set() if seen is None else seen
    def git(*args: str) -> bytes:
        return subprocess.run(["git", "-C", str(root), *args],
                              check=True, capture_output=True).stdout

    commit = git("rev-parse", "--verify", f"{rev}^{{commit}}").decode().strip()
    entries = []
    for record in git("ls-tree", "-r", "-z", "--full-tree", commit).split(b"\0"):
        if not record:
            continue
        meta, name = record.split(b"\t", 1)
        _mode, kind, oid = meta.split()
        if kind == b"blob" and (oid, name) not in seen:
            seen.add((oid, name))
            entries.append((oid, name.decode("utf-8", "replace")))
    hits = scan_bytes(git("cat-file", "commit", commit), f"<commit {commit[:12]}>", hashes)
    with subprocess.Popen(["git", "-C", str(root), "cat-file", "--batch"],
                          stdin=subprocess.PIPE, stdout=subprocess.PIPE) as proc:
        for oid, name in entries:
            proc.stdin.write(oid + b"\n")
            proc.stdin.flush()
            size = int(proc.stdout.readline().split()[2])
            data = proc.stdout.read(size)
            proc.stdout.read(1)
            hits.extend(scan_bytes(data, f"{commit[:12]}:{name}", hashes))
        proc.stdin.close()
    return len(entries), hits


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("paths", nargs="*", type=pathlib.Path,
                        help="files or folders to scan (default: tracked files)")
    parser.add_argument("--rev", action="append", default=[], metavar="REV",
                        help="scan commit REV (tree, message, author) instead; repeatable")
    parser.add_argument("--require", action="store_true",
                        help="exit 2 instead of 0 when no hashes are configured")
    parser.add_argument("--hash", metavar="TERM",
                        help="print the SHA-256 entry for TERM ('-' reads it from stdin) and exit")
    args = parser.parse_args(argv)
    if args.hash is not None:
        term = sys.stdin.readline() if args.hash == "-" else args.hash
        if not term.strip():
            parser.error("--hash needs a non-empty term")
        print(term_digest(term))
        return 0
    if args.rev and args.paths:
        parser.error("give either PATHs or --rev, not both")
    try:
        hashes = load_hashes()
    except ValueError as exc:
        print(f"private-term guard: ERROR - {exc}", file=sys.stderr)
        return 2
    if not hashes:
        print(f"private-term guard: SKIP - no hashes configured "
              f"(set {ENV_HASHES} or {ENV_FILE})")
        if os.environ.get("GITHUB_ACTIONS") == "true":
            print(f"::warning title=private-term guard::no hashes configured; "
                  f"add the {ENV_HASHES} repository secret to enable this check")
        return 2 if args.require else 0
    if args.rev:
        hits, count, seen = [], 0, set()
        for rev in args.rev:
            try:
                scanned, found = scan_rev(rev, hashes, seen=seen)
            except (OSError, subprocess.CalledProcessError, ValueError, IndexError):
                print(f"private-term guard: ERROR - cannot read commit {rev}", file=sys.stderr)
                return 2
            count += scanned
            hits.extend(found)
        what = f"{len(args.rev)} commit(s), {count} files"
    else:
        files = args.paths or tracked_files()
        hits = scan_paths(files, hashes)
        what = f"{len(files)} path(s)" if args.paths else f"{len(files)} tracked files"
    for hit in hits:
        print(hit)
    if hits:
        print(f"private-term guard: FAIL - {len(hits)} hit(s) in {what}",
              file=sys.stderr)
        return 1
    print(f"private-term guard: OK - {what}, {len(hashes)} hashes, 0 hits")
    return 0


if __name__ == "__main__":
    sys.exit(main())

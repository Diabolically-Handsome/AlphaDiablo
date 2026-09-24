"""Seed audit of the round-9 exam (PREREG-ROUND9.md decision 6): have the exam seeds ever been played, trained on or
labelled? By seed fields and directory names, not by grep (the 13:25 grep matched decimals of timestamps).
Read-only; writes only --out. CPU, about a minute. make_exam9_freeze.sh prints the command; run it at the freeze.

usage (WSL): python3 -B seed_audit9.py [--out F.json] [--include-exam-batches]
Looked at:
  directories  <root>/live/*/s<seed>, <root>/r9-work/**/s<seed> (depth <= 3)          -> a game was played
  row files    <root>/relabel*/**/*.jsonl, <root>/dagger/**/*.jsonl, <root>/sft/*/*.jsonl, <exp>/teacher-batches/**,
               <exp>/bench/** (*.json, *.jsonl): a line is parsed only if it contains an audited seed as a whole
               number; then a hit is a key 'seed' (at any depth) equal to it, or a string like '<batch>/s<seed>#..' or
               '.../s<seed>/...'; any other occurrence is listed as a 'mention' (not a hit).
Audited: the 16 King + 16 Butcher exam seeds (27 distinct) and the replacement seeds (listed apart).
Not looked at (declared exception): <root>/seed-audit/ (the tick-0 Butcher-quest probe, bench/butcher_probe.py).
Batches named exam9-* / gate9-* / smoke9-* are the exam's own games: excluded unless --include-exam-batches.
Exit 1 when an exam seed has a hit.
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

import exam9_lib as L

OWN = ('exam9-', 'gate9-', 'smoke9-')


def walk_seed(v, want, depth=0):
    if depth > 6:
        return set()
    out = set()
    if isinstance(v, dict):
        for k, x in v.items():
            if k == 'seed' and isinstance(x, int) and x in want:
                out.add(x)
            else:
                out |= walk_seed(x, want, depth + 1)
    elif isinstance(v, list):
        for x in v:
            out |= walk_seed(x, want, depth + 1)
    elif isinstance(v, str):
        for m in re.finditer(r'(?:^|/)s(\d{7})(?:#|/|$)', v):
            if int(m.group(1)) in want:
                out.add(int(m.group(1)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=None)
    ap.add_argument('--include-exam-batches', action='store_true')
    a = ap.parse_args()
    cfg = L.load_config()
    root, exp = Path(cfg['root']), Path(cfg['exp'])
    exam = set(cfg['king_seeds']) | set(cfg['butcher_seeds'])
    repl = set(cfg['replacements']['butcher']) | set(cfg['replacements']['skeleton_king'])
    want = exam | repl
    pat = re.compile(r'(?<!\d)(' + '|'.join(str(s) for s in sorted(want)) + r')(?!\d)')
    t0 = time.time()
    hits, mentions = [], []
    n_dirs = n_files = n_rows = n_parsed = 0
    # directories
    for base, depth in ((root / 'live', 1), (root / 'r9-work', 3)):
        for d in base.rglob('s*') if base.exists() else []:
            rel = d.relative_to(base)
            if len(rel.parts) - 1 > depth or not d.is_dir():
                continue
            n_dirs += 1
            m = re.fullmatch(r's(\d{7})', d.name)
            if m and int(m.group(1)) in want:
                if rel.parts[0].startswith(OWN) and not a.include_exam_batches:
                    continue
                hits.append(dict(seed=int(m.group(1)), where=str(d), how='game directory'))
    # row files
    files = []
    for g in ('relabel*/**/*.jsonl', 'dagger/**/*.jsonl', 'sft/*/*.jsonl'):
        files += sorted(root.glob(g))
    for g in ('teacher-batches/**/*.json', 'teacher-batches/**/*.jsonl', 'bench/**/*.json', 'bench/**/*.jsonl'):
        files += sorted(exp.glob(g))
    for f in files:
        if not f.is_file():
            continue
        n_files += 1
        try:
            with open(f, encoding='utf-8', errors='replace') as fh:
                if f.suffix == '.json':
                    text = fh.read()
                    n_rows += 1
                    if pat.search(text):
                        n_parsed += 1
                        try:
                            found = walk_seed(json.loads(text), want)
                        except ValueError:
                            found = set()
                        for s in found:
                            hits.append(dict(seed=s, where=str(f), how='seed field / id in json'))
                        rest = {int(x) for x in pat.findall(text)} - found
                        for s in rest:
                            mentions.append(dict(seed=s, where=str(f)))
                    continue
                for i, line in enumerate(fh):
                    n_rows += 1
                    if not pat.search(line):
                        continue
                    n_parsed += 1
                    try:
                        found = walk_seed(json.loads(line), want)
                    except ValueError:
                        found = set()
                    for s in found:
                        hits.append(dict(seed=s, where=f'{f}:{i + 1}', how='seed field / id in row'))
                    for s in {int(x) for x in pat.findall(line)} - found:
                        mentions.append(dict(seed=s, where=f'{f}:{i + 1}'))
        except OSError as exc:
            mentions.append(dict(seed=None, where=str(f), error=repr(exc)))
    exam_hits = [h for h in hits if h['seed'] in exam]
    out = dict(time=time.time(), seconds=round(time.time() - t0, 1), exam_seeds=sorted(exam), replacement_seeds=sorted(repl),
               directories_scanned=n_dirs, files_scanned=n_files, rows_scanned=n_rows, rows_parsed=n_parsed,
               exam_hits=exam_hits, replacement_hits=[h for h in hits if h['seed'] in repl and h['seed'] not in exam],
               mentions=mentions[:200], mentions_total=len(mentions),
               excluded=['<root>/seed-audit (tick-0 probe, declared exception)'] + ([] if a.include_exam_batches else [f'live/{p}* (exam games)' for p in OWN]))
    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=1), encoding='utf-8')
    print(f"seed audit: {n_dirs} game dirs, {n_files} files, {n_rows} rows ({n_parsed} parsed) in {out['seconds']} s; "
          f"exam-seed hits {len(exam_hits)}, replacement-seed hits {len(out['replacement_hits'])}, mentions {len(mentions)}")
    for h in exam_hits[:20]:
        print('  HIT', h)
    for mnt in mentions[:10]:
        print('  mention', mnt)
    sys.exit(1 if exam_hits else 0)


if __name__ == '__main__':
    main()

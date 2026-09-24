#!/usr/bin/env python3
"""Make a runnable copy of option_brain/ with your own paths.

The published files are the frozen round-9 files with five literal placeholders
in place of machine-specific values (see PROVENANCE.md):

    $AD_WORKSPACE  the folder whose experiments/ subfolder holds the code folders
                   (this option_brain/ folder plays the role of experiments/)
    $AD_ROOT       the run root (live/, sft/, relabel/, exam9-* files)
    $AD_HOME       the home folder that holds the other run folders
                   (engine/bridge builds, base model weights, executor weights)
    $AD_GPU_UUID   the CUDA device used for the brain (nvidia-smi -L)
    $AD_GPU2_UUID  a second CUDA device (only named in an old constant)

This script copies the tree to OUT and substitutes the placeholders; it never
edits the files in place.  Substituting the original values reproduces the
original bytes of every file except the two whose edits are listed in
PROVENANCE.md.

    python localize.py OUT --workspace /path/to/AlphaDiablo --root /path/to/run-root \
        --home /path/to/home --gpu GPU-xxxxxxxx-... [--gpu2 GPU-...]
"""
import argparse
import pathlib
import shutil

HERE = pathlib.Path(__file__).resolve().parent
SKIP = {'localize.py', 'README.md', 'PROVENANCE.md', 'provenance.json'}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('out', type=pathlib.Path)
    ap.add_argument('--workspace', required=True)
    ap.add_argument('--root', required=True)
    ap.add_argument('--home', required=True)
    ap.add_argument('--gpu', required=True)
    ap.add_argument('--gpu2', default='')
    a = ap.parse_args()
    if a.out.exists():
        ap.error(f'{a.out} exists; give a new folder')
    subs = [('$AD_WORKSPACE', a.workspace), ('$AD_ROOT', a.root), ('$AD_HOME', a.home),
            ('$AD_GPU2_UUID', a.gpu2), ('$AD_GPU_UUID', a.gpu)]
    for src in sorted(p for p in HERE.rglob('*') if p.is_file()):
        rel = src.relative_to(HERE)
        if rel.as_posix() in SKIP or '__pycache__' in rel.parts:
            continue
        dst = a.out / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        data = src.read_bytes()
        for ph, value in subs:
            data = data.replace(ph.encode(), value.encode())
        dst.write_bytes(data)
        shutil.copymode(src, dst)
    print(f'wrote {a.out}')


if __name__ == '__main__':
    main()

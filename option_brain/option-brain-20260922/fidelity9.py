"""Re-score decisions with a given adapter through the unchanged broker (PREREG-ROUND9.md sections 5 and 8).

  section 5 (smoke, before the gate):
    old-arm fidelity: the frozen hf_broker.py + option_sft.py + sft/d9facts3 re-score ~500 prompts drawn uniformly from
    the four round-8 held-out batches; every argmax letter must equal the logged choice.
    determinism: those 500 scored twice with d9facts3, and 200 test rows of fidelity.new_rel scored twice with the new arm's
    adapter (round 9b: relabel-d11f5 / d11facts5; relabel-d10f4 / d10facts4 before);
    the argmax letters must be identical (probability differences only reported).
  section 8 (after the exam): ~2000 decisions per arm, stratified by game, re-scored with that arm's adapter; every
    argmax must equal the logged choice.
The random seed is exam9_config.json fidelity.sample_seed (frozen with the config). This script only writes requests
and compares answers (CPU). The broker (GPU) is started by launch_smoke9.sh, or by hand with the exam's own command:
<py_model> -B hf_broker.py --adapter <adapter> --bus <bus> (cwd the experiment directory); touch <bus>/STOP after the
last response.

usage (WSL):
  python3 -B fidelity9.py make --bus DIR --n N (--batches B1 [B2 ...] [--stratify-games] | --relabel REL F1 [F2 ...])
                          [--repeat R] [--sample-seed S]
      writes DIR/requests/r<rep>-<k>.json (the logged / row prompt, the system prompt live_runner sends, the option
      letters; R identical copies) and DIR/manifest.json. Refuses an existing DIR.
  python3 -B fidelity9.py compare --bus DIR [--out F.json]
      argmax of copy 1 vs the logged choice (live batches), argmax identical across the copies; exit 1 on any
      mismatch or missing answer. Reports the largest probability difference to the log and across copies.
"""
import argparse
import collections
import json
import random
import sys
from pathlib import Path

import exam9_lib as L

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def pool_from_batches(cfg, batches):
    live = Path(cfg['root']) / 'live'
    pool = []
    for b in batches:
        for d in sorted((live / b).glob('s*')):
            if not d.is_dir():
                continue
            variant = (L.read_json(d / 'started.json') or {}).get('variant', 'v2')
            for r in L.iter_rows(d):
                if r.get('prompt') and r.get('options') and r.get('choice'):
                    pool.append(dict(game=f'{b}/{d.name}', variant=variant, n=r.get('n'), prompt=r['prompt'], options=r['options'],
                                     choice=r['choice'], confidence=r.get('confidence'), probs=r.get('probs')))
    return pool


def pool_from_relabel(cfg, rel, files):
    pool = []
    for f in files:
        p = Path(cfg['root']) / rel / f'{f}.jsonl'
        for i, line in enumerate(open(p, encoding='utf-8')):
            r = json.loads(line)
            pool.append(dict(game=f'{rel}/{f}', variant='v2', n=i + 1, prompt=r['prompt'], options=r['options'], choice=None,
                             label=r.get('label'), acceptable=r.get('acceptable')))
    return pool


def make(a, cfg):
    from render import SYSTEMS  # the system prompts live_runner sends (SYSTEMS[variant]); training uses SYSTEM_V2
    bus = Path(a.bus)
    if bus.exists():
        sys.exit(f'refused: {bus} exists')
    pool = pool_from_batches(cfg, a.batches) if a.batches else pool_from_relabel(cfg, a.relabel[0], a.relabel[1:])
    rng = random.Random(a.sample_seed)
    if a.stratify_games:
        by = collections.defaultdict(list)
        for i, it in enumerate(pool):
            by[it['game']].append(i)
        games = sorted(by)
        base, extra = divmod(min(a.n, len(pool)), len(games))
        pick = []
        for j, g in enumerate(games):
            k = min(len(by[g]), base + (1 if j < extra else 0))
            pick += rng.sample(by[g], k)
        pick = sorted(pick)
    else:
        pick = sorted(rng.sample(range(len(pool)), min(a.n, len(pool))))
    (bus / 'requests').mkdir(parents=True)
    (bus / 'responses').mkdir()
    man = dict(source=a.batches or a.relabel, n_pool=len(pool), n=len(pick), repeat=a.repeat, sample_seed=a.sample_seed,
               stratify_games=a.stratify_games, items=[])
    for k, i in enumerate(pick, 1):
        it = pool[i]
        labels = [o['label'] for o in it['options']]
        req = dict(system=SYSTEMS[it['variant']], user=it['prompt'], labels=labels, kinds=[o.get('kind') for o in it['options']])
        for rep in range(1, a.repeat + 1):
            rid = f'r{rep}-{k:05d}'
            (bus / 'requests' / f'{rid}.json').write_text(json.dumps(dict(req, id=rid), ensure_ascii=False), encoding='utf-8')
        man['items'].append({kk: v for kk, v in it.items() if kk not in ('prompt', 'options', 'variant')} | dict(k=k))
    (bus / 'manifest.json').write_text(json.dumps(man, indent=1), encoding='utf-8')
    print(f'wrote {len(pick)} x {a.repeat} requests from a pool of {len(pool)} ({man["source"]}) to {bus}')


def compare(a):
    bus = Path(a.bus)
    man = json.loads((bus / 'manifest.json').read_text(encoding='utf-8'))
    rep = man.get('repeat', 1)
    logged_same = logged_diff = rep_diff = missing = 0
    worst_log = worst_rep = 0.0
    bad = []
    for it in man['items']:
        res = [L.read_json(bus / 'responses' / f"r{r}-{it['k']:05d}.json") for r in range(1, rep + 1)]
        if any(x is None for x in res):
            missing += 1
            continue
        if it.get('choice'):
            if res[0].get('choice') == it['choice']:
                logged_same += 1
            else:
                logged_diff += 1
                bad.append(dict(kind='logged', game=it['game'], n=it['n'], logged=it['choice'], now=res[0].get('choice'),
                                logged_conf=it.get('confidence'), now_conf=res[0].get('confidence')))
            lp, np_ = it.get('probs') or {}, res[0].get('probs') or {}
            for key in set(lp) & set(np_):
                worst_log = max(worst_log, abs(lp[key] - np_[key]))
        for x in res[1:]:
            if x.get('choice') != res[0].get('choice'):
                rep_diff += 1
                bad.append(dict(kind='repeat', game=it['game'], n=it['n'], first=res[0].get('choice'), again=x.get('choice')))
            for key in set(x.get('probs') or {}) & set(res[0].get('probs') or {}):
                worst_rep = max(worst_rep, abs(x['probs'][key] - res[0]['probs'][key]))
    n = len(man['items'])
    out = dict(n=n, repeat=rep, logged_same=logged_same, logged_different=logged_diff, repeat_different=rep_diff, missing=missing,
               max_abs_prob_diff_to_log=worst_log, max_abs_prob_diff_between_copies=worst_rep, mismatches=bad[:50])
    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=1), encoding='utf-8')
    print(f"fidelity: {n} items x {rep}: vs log {logged_same} same / {logged_diff} different; between copies {rep_diff} different; "
          f"{missing} missing; max |dp| to log {worst_log:.4g}, between copies {worst_rep:.4g}")
    for b in bad[:10]:
        print('  MISMATCH', b)
    sys.exit(0 if logged_diff == 0 and rep_diff == 0 and missing == 0 else 1)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    m = sub.add_parser('make')
    m.add_argument('--bus', required=True)
    m.add_argument('--n', type=int, required=True)
    src = m.add_mutually_exclusive_group(required=True)
    src.add_argument('--batches', nargs='+')
    src.add_argument('--relabel', nargs='+', metavar=('REL', 'FILE'))
    m.add_argument('--stratify-games', action='store_true')
    m.add_argument('--repeat', type=int, default=1)
    m.add_argument('--sample-seed', type=int, default=None)
    c = sub.add_parser('compare')
    c.add_argument('--bus', required=True)
    c.add_argument('--out', default=None)
    a = ap.parse_args()
    cfg = L.load_config()
    if a.cmd == 'make':
        if a.sample_seed is None:
            a.sample_seed = cfg['fidelity']['sample_seed']
        if a.relabel and len(a.relabel) < 2:
            sys.exit('--relabel needs REL and at least one FILE')
        make(a, cfg)
    else:
        compare(a)


if __name__ == '__main__':
    main()

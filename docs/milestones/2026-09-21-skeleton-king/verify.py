"""Asset-free consistency check of the published Skeleton King evidence ZIP.

Usage: python verify.py skeleton-king-evidence.zip [directory-with-video-assets]
This checks records, NOT an independent engine replay or policy evaluation.
"""
import collections
import hashlib
import json
from pathlib import Path
import sys
import zipfile


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def verify(data):
    manifest = json.loads(data['manifest.json'])
    require(set(data) == set(manifest['files']) | {'manifest.json'}, 'unexpected/missing archive entry')
    for name, entry in manifest['files'].items():
        require(len(data[name]) == entry['bytes'], 'length mismatch: ' + name)
        require(hashlib.sha256(data[name]).hexdigest() == entry['sha256'], 'hash mismatch: ' + name)
    doc = lambda name: json.loads(data[name])
    lines = lambda name: [json.loads(line) for line in data[name].decode('utf-8').splitlines()]
    audit, final = doc('verified-success.json'), doc('final-state.json')
    pause, closed = doc('native/pause.json'), doc('native/closed.json')
    journal = lines('native/journal.jsonl')
    chain = '0' * 64
    for i, row in enumerate(journal):
        require(row['seq'] == i and row['previous'] == chain, 'journal ordering')
        require(digest({k: v for k, v in row.items() if k != 'chain'}) == row['chain'], 'journal chain')
        chain = row['chain']
    require(journal[0]['kind'] == 'reset' and journal[0]['args'] == {'seed': 20260927}, 'reset seed')
    require(len(journal) == audit['journal_rows'] == 63153, 'journal count')
    require([r['tick'] for r in journal if r['kind'] == 'tick'] == list(range(1, 56388)), 'tick continuity')
    require(pause['seq'] == len(journal) and pause['chain'] == chain == audit['journal_chain'], 'checkpoint chain')
    require(pause['state_sha256'] == journal[-1]['state_sha256'], 'final native checkpoint identity')
    require(closed['pause_sha256'] == hashlib.sha256(data['native/pause.json']).hexdigest(), 'closed checkpoint identity')
    require(closed['reason'] == 'skeleton_king_success' and not closed['dead'], 'closure reason')
    require(pause['tick'] == final['tick'] == 56387, 'final tick')
    require(pause['state']['normal_difficulty'] is True and final['hero']['warrior'] is True, 'difficulty/class')
    for state in (final, pause['state']):
        require(state['king_kills'] == 1 and state['king_quest_done'] is True, 'native King outcome')
        require(not state['hero']['dead'] and state['hero']['hp_fixed'] > 0, 'hero survival')
    require(final['hero'] == pause['state']['hero'], 'final hero consistency')
    hero = final['hero']
    require((hero['level'], hero['armor'], hero['gold'], hero['hp_fixed'], hero['max_hp_fixed'])
            == (7, 32, 736, 6144, 6144), 'reported hero totals')
    count_heals = lambda group: sum(not item['empty'] and item.get('heal_kind', 0) > 0 for item in group)
    require(count_heals(final['belt']) == 1 and count_heals(final['inventory']) == 12, 'remaining potions')
    for name in data:
        if name.startswith('ancestry/'):
            prior = lines(name)
            require(journal[:len(prior)] == prior, 'history prefix: ' + name)
    decisions = [row for tag in ('rl-initial', 'rl-r1', 'rl-r2', 'rl-r4')
                 for row in lines(f'decisions/{tag}.jsonl')]
    require([r['decision'] for r in decisions] == list(range(1, 8851)), 'execution decision sequence')
    require(dict(collections.Counter(r['component'] for r in decisions)) == audit['decision_components'], 'control attribution')
    drinks = [r for r in decisions if r['component'] == 'strategist-directed-service'
              and r['command']['kind'] == 'drink' and 54726 <= r['tick'] <= 56387]
    require([r['tick'] for r in drinks] == audit['last_encounter_drink_ticks'] and len(drinks) == 7, 'drink evidence')
    goals = lines('teacher-goals.jsonl')
    require(len(goals) == 448, 'goal count')
    for row in goals:
        before = row['public_observation_before']
        require(row['goal']['state_id'] == before.get('state_id', digest(before)), 'goal before-state binding')
        require(row['label_status'] == 'unreviewed_candidate_demonstration_not_ground_truth', 'teacher qualification')
    require(goals[-1]['actual_result']['state']['king_kills'] == 1, 'last goal outcome')
    media = doc('media/final-fight-verification.json')
    require(media['passed'] and media['all_ticks_contiguous'] and media['native_replay_verified'], 'clip verification')
    require(media['first_tick'] == 54682 and media['last_tick'] == 56387
            and media['frames'] == 5118 and media['seconds'] == 85.3, 'clip exact coverage')
    frames = lines('media/final-fight-frames.jsonl')
    require([(r['tick'], r['subframe']) for r in frames] == [(t, s) for t in range(54682, 56388) for s in range(3)],
            'clip subframe index')
    require([r['index'] for r in frames] == list(range(5118)), 'clip frame order')
    require(media['source_journal_sha256'] == hashlib.sha256(data['native/journal.jsonl']).hexdigest(), 'clip to source binding')
    return dict(passed=True, journal_rows=len(journal), native_ticks=56387, teacher_goals=len(goals),
                execution_decisions=len(decisions), clip_frames=len(frames),
                scope='record consistency, not independent native-engine replay')


def main():
    if len(sys.argv) not in (2, 3):
        raise SystemExit(__doc__)
    with zipfile.ZipFile(sys.argv[1]) as archive:
        names = archive.namelist()
        require(len(names) == len(set(names)), 'duplicate ZIP entry')
        require(all(not n.startswith(('/', '\\')) and '..' not in Path(n).parts for n in names), 'unsafe archive path')
        data = {name: archive.read(name) for name in names}
    result = verify(data)
    if len(sys.argv) == 3:
        for name, expected in json.loads(data['media/assets.json']).items():
            path = Path(sys.argv[2]) / name
            with path.open('rb') as f:
                actual = hashlib.file_digest(f, 'sha256').hexdigest()
            require(actual == expected['sha256'] and path.stat().st_size == expected['bytes'], 'media hash: '+name)
        result['video_asset_hashes_match'] = True
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()

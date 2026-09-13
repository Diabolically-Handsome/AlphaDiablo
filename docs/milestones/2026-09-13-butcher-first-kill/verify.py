"""Verify the public evidence package; this does NOT execute Diablo or estimate a win rate."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def read(relative):
    return json.loads((ROOT / relative).read_text(encoding='utf8'))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    manifest = read('manifest.json')
    for relative, expected in manifest['files'].items():
        path = (ROOT / relative).resolve()
        require(path.is_relative_to(ROOT.resolve()), 'Path escapes evidence root')
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        require(actual == expected, f'File changed: {relative}')
    actual_paths = {str(path.relative_to(ROOT)).replace('\\', '/')
                    for path in ROOT.rglob('*') if path.is_file()
                    and path.name != 'manifest.json' and '__pycache__' not in path.parts}
    require(actual_paths == set(manifest['files']), 'Missing or unlisted evidence file')
    winner = read('evidence/winner/result.json')
    completion = read('evidence/completion-audit.json')
    summary = read('evidence/summary.json')
    require(completion['status'] == 'VERIFIED_FIRST_REPRODUCIBLE_DEVELOPMENT_WIN', 'Unverified original milestone')
    require(all(completion['requirements'].values()), 'Original completion audit has a failed requirement')
    require(winner['row']['seed'] == 2133013 and not winner['row']['died'], 'Wrong winner')
    require(winner['row']['char_level'] == 6 and winner['row']['kills'] == 274, 'Wrong endpoint')
    require(winner['task']['final_quest']['done'] and winner['task']['first_done_player_alive'], 'No living native kill')
    require(not winner['task']['initial_quest']['done'], 'Quest already complete at reset')
    initial = read('evidence/winner/gear-history.json')['changes'][0]['after']
    require((initial['char_level'], initial['hp'], initial['gold'], initial['belt_heals']) == (1, 70, 100, 2), 'Wrong initial state')
    final = read('evidence/winner/spending-audit.json')['final_player']
    require((final['hp'], final['max_hp'], final['belt_heals']) == (83, 110, 1), 'Wrong final health/supplies')
    for purchase in completion['prices']:
        require(purchase['gold_before'] - purchase['gold_after'] == purchase['price'], 'Purchase did not pay recorded price')
    require(len(completion['prices']) == 12, 'Unexpected purchase count')
    for relative in ('evidence/original-replay-result.json', 'evidence/video-replay-result.json'):
        replay = read(relative)
        require(replay['row'] == winner['row'] and replay['task'] == winner['task'], 'Replay mismatch: ' + relative)
        require(replay['rows_sha_v3'] == winner['rows_sha_v3'], 'Result digest mismatch')
    recording = read('evidence/video-replay-verification.json')
    require(recording['status'] == 'VERIFIED_NATIVE_VIDEO_REPLAY', 'Unverified recording')
    require(len(recording['physical_audits_exact']) == 9 and all(recording['physical_audits_exact'].values()), 'Recording changed a physical audit')
    require(recording['new_training_steps'] == 0 and not recording['independent_validation'], 'Recording has wrong evidence scope')
    frames = read('evidence/video-frames.json')
    require(frames[-1]['quest_done'] and frames[-1]['hp'] == 83, 'Video does not include winning endpoint')
    encounter = [f for f in frames if f['step'] >= 26173]
    require([f['step'] for f in encounter] == list(range(26173, 26506)), 'Video is missing an encounter step')
    require(summary['independent_validation_games'] == 0, 'Reused seeds mislabeled independent')
    candidates = [row for row in summary['original_games'] if row['recipe'] == 'candidate']
    require(len(candidates) == 2 and sum(row['butcher_done'] for row in candidates) == 1, 'Failed candidate omitted')
    for mode in ('baseline', 'candidate'):
        require(read(f'evidence/native-tests-{mode}.json')['passed'], 'Native engineering checks failed')
    receipt = read('evidence/combat-bc-training-receipt.json')
    require(receipt['model_sha256'] == summary['models']['w1'], 'Wrong BC checkpoint identity')
    print(f'PASS: {len(manifest["files"])} file hashes; native kill; paid purchases; two exact replays; all encounter frames.')
    print('Scope: one winning reused development seed, not independent generalization or full-game completion.')


if __name__ == '__main__':
    main()

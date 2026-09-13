"""Check the smooth-video evidence; this neither executes Diablo nor proves a win rate."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(relative):
    return json.loads((ROOT/relative).read_text(encoding='utf8'))


def main():
    manifest = read('manifest.json')
    for relative, expected in manifest['files'].items():
        path = (ROOT/relative).resolve()
        require(path.is_relative_to(ROOT), 'Unsafe manifest path')
        require(hashlib.sha256(path.read_bytes()).hexdigest() == expected, 'Changed file: '+relative)
    verification = read('capture-verification.json')
    require(verification['status'] == 'VERIFIED_SMOOTH_NATIVE_REPLAY', 'Unverified capture')
    require(verification['row_exact'] and verification['task_exact'], 'Gameplay mismatch')
    require(len(verification['physical_audits_exact']) == 9
            and all(verification['physical_audits_exact'].values()), 'Physical audit mismatch')
    require(verification['new_training_steps'] == 0
            and not verification['independent_validation'], 'Wrong research scope')
    frames = read('frame-index.json')
    steps = list(range(51, 91)) + list(range(12001, 12031)) + list(range(26151, 26506))
    expected = [(step, tick, sub) for step in steps for tick in range(1, 5) for sub in range(3)]
    require([(f['step'], f['tick'], f['subframe']) for f in frames] == expected, 'Missing/duplicate frame')
    require(len(frames) == 5100 and frames[-1]['quest_done'] and frames[-1]['hp'] == 83,
            'Wrong frame count or winning endpoint')
    media = read('media.json')
    require(media['fps'] == 60 and media['encoded_frames'] == 5400
            and media['duration_seconds'] == 90, 'Wrong presentation cadence')
    require(media['engine_sub_tick_fractions'] == [0, 42, 85]
            and media['denominator'] == 128, 'Wrong native interpolation')
    if len(sys.argv) > 1:
        video = Path(sys.argv[1])
        require(video.stat().st_size == media['bytes'], 'Video size mismatch')
        with video.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        require(digest == media['sha256'], 'Video checksum mismatch')
    print('PASS: 1700 native ticks, 5100 rendered frames, 90 seconds at 60 fps; exact replay, not a new trial.')


if __name__ == '__main__':
    main()

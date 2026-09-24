"""Engine numbers of fact-line version 4 (PLAN-annotations-v4.md sections 1 and 5), 2026-09-23 round 9 data step.

The calculation kernel itself lives in facts.py (a documented deviation of the round-9 code step: facts.py already
held the parsers, and one copy keeps training and live identical). This module is its read-only facade for the
gate and rule scripts (gates_round9.py, tick_sim.py, dagger_round9.py) plus test T3 of the plan: every engine
constant the kernel uses equals the value read again from the game's TSV tables, and the TSV files are the ones
recorded here (sha256). Standard library only; importing it changes nothing in facts.py.

usage (WSL): python3 -B engine_numbers.py            prints the T3 check (exit 1 on a mismatch)
"""
import csv
import hashlib
import sys
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from facts import (V4_BASES, V4_BOSSES, V4_FPS, V4_INTERRUPT, V4_SWING, damage_dist_v4, fight_v4, hit_pct,  # noqa: E402,F401
                   loadout_v4, potion_heal_v4, swap_v4, weapon_options_v4)

TSV = Path('$AD_HOME/alphadiablo-dev/devilutionX/assets/txtdata')
# sha256 of the tables the constants were checked against (PLAN section 0: identical to the live assets), recorded
# 2026-09-23 by this module's first run; check_t3 reports a table that changed since
TSV_SHA256 = {
    'monsters/monstdat.tsv': '3ffc0d9aad41164ec4bfff385ae6ac3e24ab53759fa3c025747dc5580935f371',
    'monsters/unique_monstdat.tsv': '988cedfd23ab4076c60ced2951c814da762c3e9854b51502394bbd32f7e4f641',
    'items/itemdat.tsv': '56daa0e75806f2a9cb83eac4ea9549e432d0da30f2b9e31d6a5d0a73a4567e38',
    'classes/warrior/animations.tsv': '99e026c45b78d2bd45c5042592e4e78dd762efc9f8629571d25bcb0fbf545c7b',
    'classes/warrior/attributes.tsv': '31bbcaa49f184e2f757e39fdc1a6f341cda44dab352b0a854004077bb9da29e9',
}
TSV_FILES = tuple(TSV_SHA256)
BOSS_ROWS = {'king': 'MT_SKING', 'butcher': 'MT_CLEAVER'}
CLASS = {'Undead': 'undead', 'Demon': 'demon', 'Animal': 'animal'}
# itemType of itemdat -> the kernel's weapon type (V4_SWING keys) or 'Shield' / 'Bow'
ITEM_TYPE = {'Sword': 'Sword', 'Axe': 'Axe', 'Mace': 'Mace', 'Bow': 'Bow', 'Staff': 'Staff', 'Shield': 'Shield'}
# animations.tsv action frame per kernel weapon type (the chained-attack period, F7: player.cpp:1332-1344 starts the
# next attack once currentFrame >= _pAFNum)
ACTION_FRAME = {'Sword': 'swordActionFrame', 'Mace': 'maceActionFrame', 'Axe': 'axeActionFrame', 'Staff': 'staffActionFrame'}


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _rows(rel):
    with open(TSV / rel, encoding='utf-8') as fh:
        return list(csv.DictReader(fh, delimiter='\t'))


def _kv(rel):
    return {r['Variable']: r['Value'] for r in _rows(rel)}


def boss_from_tsv(key):
    """The kernel's boss row rebuilt from monstdat + unique_monstdat (unique level 0 -> type level + 5, monster.h:395-405;
    unique HP halved in single player, monster.cpp:210-214; custom to-hit / armour 0 -> the type's values)."""
    m = next(r for r in _rows('monsters/monstdat.tsv') if r['_monster_id'] == BOSS_ROWS[key])
    u = next(r for r in _rows('monsters/unique_monstdat.tsv') if r['type'] == BOSS_ROWS[key])
    lvl = int(u['level']) or int(m['level']) + 5
    return dict(mlvl=lvl, hp=int(u['maxHp']) // 2, armour=int(u['customArmorClass']) or int(m['armorClass']),
                to_hit=int(u['customToHit']) or int(m['toHit']), lo=int(u['minDamage']), hi=int(u['maxDamage']),
                cls=CLASS[m['monsterClass']], attack_frames=int(m['frames[6]'].split(',')[2]),
                recovery_frames=int(m['frames[6]'].split(',')[3]), hit_frame=int(m['animFrameNum']))


def check_t3():
    """[(what, kernel value, table value)] of every mismatch (empty = T3 passes)."""
    bad = [(f'sha256 {f}', want, sha256(TSV / f)) for f, want in TSV_SHA256.items() if sha256(TSV / f) != want]
    for key in BOSS_ROWS:
        t = boss_from_tsv(key)
        for k in ('mlvl', 'hp', 'armour', 'to_hit', 'lo', 'hi', 'cls'):
            if V4_BOSSES[key][k] != t[k]:
                bad.append((f'{key}.{k}', V4_BOSSES[key][k], t[k]))
    items = {r['name']: r for r in _rows('items/itemdat.tsv')}
    for name, (kind, two) in sorted(V4_BASES.items()):
        r = items.get(name)
        if r is None:
            if name != "Griswold's Edge":          # a unique name of its own (unique_itemdat), base Broad Sword type
                bad.append((f'itemdat {name}', (kind, two), None))
            continue
        want = (ITEM_TYPE.get(r['itemType']), r['equipType'] == 'Two-handed')
        if (kind, two) != want:
            bad.append((f'itemdat {name}', (kind, two), want))
    anim = _kv('classes/warrior/animations.tsv')
    for kind, var in ACTION_FRAME.items():
        if V4_SWING[kind] != int(anim[var]):
            bad.append((f'swing {kind}', V4_SWING[kind], int(anim[var])))
    if int(anim['recoveryFrames']) != 6 or int(anim['blockingFrames']) != 2:
        bad.append(('recovery/blocking frames', (6, 2), (anim['recoveryFrames'], anim['blockingFrames'])))
    return bad


def main():
    bad = check_t3()
    shas = {f: sha256(TSV / f) for f in TSV_FILES}
    print({'t3_mismatches': bad, 'tsv_sha256': shas, 'bosses_from_tsv': {k: boss_from_tsv(k) for k in BOSS_ROWS},
           'interrupt_frames': str(V4_INTERRUPT), 'fps': V4_FPS, 'regen_per_second':
           {k: str(Fraction(b['mlvl'] // 2 * V4_FPS, 64)) for k, b in V4_BOSSES.items()}})
    sys.exit(1 if bad else 0)


if __name__ == '__main__':
    main()

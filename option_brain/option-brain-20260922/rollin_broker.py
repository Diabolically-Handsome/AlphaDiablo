"""DAgger roll-in broker: a hand rule with some randomness, used only to REACH varied states.

Its choices are never training labels. The teacher labels the visited states afterwards.
Town: buy potions if a shop is open, take the cathedral entrance once seen, otherwise explore
north-west (where the cathedral is). Dungeon: drink when low, fight visible enemies, loot
nearby, explore, and sometimes take the stairs down; on level 3 sometimes enter the tomb.
With probability EPS any legal option is chosen uniformly, so states off the rule's path
also get visited. Randomness is seeded from the request id, so runs are reproducible.
"""
import json
import os
import random
import re
import sys
import time
from pathlib import Path

EPS = float(os.environ.get('ROLLIN_EPS', '0.15'))
OPT = re.compile(r'^([A-Za-z])\. (.*)$')


def parse(user):
    lines = user.splitlines()
    opts = {}
    if 'Options:' in lines:
        for ln in lines[lines.index('Options:') + 1:]:
            m = OPT.match(ln)
            if m:
                opts[m.group(1)] = m.group(2)
    loc = next((ln for ln in lines if ln.startswith('Location:')), '')
    hero = next((ln for ln in lines if ln.startswith('Hero:')), '')
    m = re.search(r'healing potions (\d+) in belt and (\d+) in pack', hero)
    pots = int(m.group(1)) + int(m.group(2)) if m else 0
    m = re.search(r'Hero: level (\d+)', hero)
    clvl = int(m.group(1)) if m else 1
    m = re.search(r'Dungeon level (\d+)', loc)
    dlvl = 0 if 'Town' in loc else (int(m.group(1)) if m else -1)
    return opts, dlvl, pots, clvl, 'Quest area' in loc


def rule(req, rng):
    labels, kinds = req['labels'], req['kinds']
    opts, dlvl, pots, clvl, in_quest = parse(req['user'])
    by = {}
    for l, k in zip(labels, kinds):
        by.setdefault(k, []).append(l)
    text = lambda l: opts.get(l, '')
    first = lambda k: by.get(k, [None])[0]
    hpf = req.get('hpfrac') or 1.0
    enemies = req.get('enemies') or 0
    if rng.random() < EPS:
        return rng.choice(labels), 'random'
    if hpf < 0.5 and first('drink'):
        return first('drink'), 'drink'
    reach = [l for l in by.get('fight', []) if 'no known path' not in text(l)]
    if enemies and reach:
        if hpf < 0.3 and first('fallback'):
            return first('fallback'), 'fallback'
        if len(reach) >= 2 and first('fight_all'):
            return first('fight_all'), 'fight_all'
        return reach[0], 'fight'
    for k in ('dismiss', 'stat', 'resume'):
        if first(k):
            return first(k), k
    if dlvl == 0:
        buys = [l for l in by.get('buy', []) if 'Healing' in text(l) or 'healing' in text(l)]
        if buys and pots < 5:
            return buys[0], 'buy'
        downs = [l for l in by.get('exit', []) if 'DOWN' in text(l)]
        if downs:
            return downs[0], 'exit_down'
        for d in ('north-west', 'north', 'west', 'south-west', 'north-east'):
            ex = [l for l in by.get('explore', []) if f'toward the {d} ' in text(l)]
            if ex:
                return ex[0], 'explore_' + d
        return rng.choice(labels), 'town_other'
    # Dungeon.
    ups = [l for l in by.get('exit', []) if 'UP' in text(l) or 'back to the main' in text(l)]
    if pots == 0 and hpf < 0.5 and ups:
        return ups[0], 'retreat_up'
    near = lambda l: int(m.group(1)) if (m := re.search(r'\((\d+) steps away\)', text(l))) else 99
    for k, lim in (('pickup', 10), ('object', 7)):
        c = [l for l in by.get(k, []) if near(l) <= lim]
        if c:
            return c[0], k
    if dlvl == 3 and first('quest') and rng.random() < 0.4:
        return first('quest'), 'quest'
    downs = [l for l in by.get('exit', []) if 'DOWN' in text(l)]
    if downs and dlvl < 3 and (not by.get('explore') or rng.random() < 0.12):
        return downs[0], 'exit_down'
    if by.get('explore'):
        return rng.choice(by['explore'][:3]), 'explore'
    for k in ('pickup', 'object', 'quest', 'exit', 'npc'):
        if first(k):
            return first(k), k
    return labels[0], 'first'


MODE = os.environ.get('ROLLIN_MODE', 'basic')
MARGIN = int(os.environ.get('ROLLIN_DESCEND_MARGIN', '1'))  # deep: need clvl >= 2*dlvl + MARGIN to go down
RESTOCK = int(os.environ.get('ROLLIN_RESTOCK_POTS', '1'))  # deep: go back to town at <= this many potions


def rule_deep(req, rng):
    """Deeper roll-in (round 2): grow before descending, restock at Pepin, enter the tomb when strong."""
    labels, kinds = req['labels'], req['kinds']
    opts, dlvl, pots, clvl, in_quest = parse(req['user'])
    m = re.search(r', gold (\d+),', req['user'])
    gold = int(m.group(1)) if m else 0
    by = {}
    for l, k in zip(labels, kinds):
        by.setdefault(k, []).append(l)
    text = lambda l: opts.get(l, '')
    first = lambda k: by.get(k, [None])[0]
    hpf = req.get('hpfrac') or 1.0
    enemies = req.get('enemies') or 0
    ups = [l for l in by.get('exit', []) if 'UP' in text(l) or 'back to the main' in text(l)]
    downs = [l for l in by.get('exit', []) if 'DOWN' in text(l)]
    if rng.random() < EPS:
        return rng.choice(labels), 'random'
    if hpf < 0.5 and first('drink'):
        return first('drink'), 'drink'
    reach = [l for l in by.get('fight', []) if 'no known path' not in text(l)]
    if dlvl != 0 and enemies and pots == 0 and hpf < 0.35 and ups:
        return ups[0], 'flee_up'
    if enemies and reach:
        if hpf < 0.3 and first('fallback'):
            return first('fallback'), 'fallback'
        if len(reach) >= 2 and first('fight_all'):
            return first('fight_all'), 'fight_all'
        return reach[0], 'fight'
    for k in ('stat', 'resume'):
        if first(k):
            return first(k), k
    if dlvl == 0:
        heals = [l for l in by.get('buy', []) if 'Healing' in text(l)]
        if heals and pots < 6:
            return heals[0], 'buy'
        if first('dismiss'):
            return first('dismiss'), 'dismiss'
        pepin = [l for l in by.get('npc', []) if 'Pepin' in text(l)]
        need_pepin = hpf < 0.9 or (pots < 4 and gold >= 60)
        if pepin and (hpf < 0.9 or (pots < 4 and gold >= 60 and rng.random() < 0.5)):
            return pepin[0], 'talk_pepin'  # free healing when hurt; potions when affordable
        if need_pepin and not pepin:
            # Pepin not seen yet: he lives south-east of the cathedral entrance.
            for d in ('south-east', 'east', 'south', 'north-east', 'south-west'):
                ex = [l for l in by.get('explore', []) if f'toward the {d} ' in text(l)]
                if ex:
                    return ex[0], 'find_pepin_' + d
        if downs:
            return downs[0], 'exit_down'
        for d in ('north-west', 'north', 'west', 'south-west', 'north-east'):
            ex = [l for l in by.get('explore', []) if f'toward the {d} ' in text(l)]
            if ex:
                return ex[0], 'explore_' + d
        return rng.choice(labels), 'town_other'
    if first('dismiss'):
        return first('dismiss'), 'dismiss'
    if pots <= RESTOCK and hpf < 0.7 and ups:
        return ups[0], 'restock_up'
    near = lambda l: int(mm.group(1)) if (mm := re.search(r'\((\d+) steps away\)', text(l))) else 99
    for k, lim in (('pickup', 10), ('object', 7)):
        c = [l for l in by.get(k, []) if near(l) <= lim]
        if c:
            return c[0], k
    if dlvl == 3 and not in_quest and first('quest') and (clvl >= 5 and rng.random() < 0.5 or rng.random() < 0.05):
        return first('quest'), 'quest'
    strong = clvl >= 2 * dlvl + MARGIN
    if downs and dlvl < 3 and not in_quest and (not by.get('explore') or (strong and rng.random() < 0.35)):
        return downs[0], 'exit_down'
    if by.get('explore'):
        return rng.choice(by['explore'][:3]), 'explore'
    for k in ('pickup', 'object', 'quest', 'exit', 'npc'):
        if first(k):
            return first(k), k
    return labels[0], 'first'


def rule_deep3(req, rng):
    """Roll-in v3 (2026-09-22 night): rule_deep plus (a) refill the belt from the pack when it has no healing
    potion, and (b) on dungeon level 3 enter the tomb only at character level >= 6 with >= 4 potions (before
    that, never). Used to reach the tomb and the Skeleton King for teacher labels; choices are never labels."""
    labels, kinds = req['labels'], req['kinds']
    opts, dlvl, pots, clvl, in_quest = parse(req['user'])
    text = lambda l: opts.get(l, '')
    m = re.search(r'healing potions (\d+) in belt and (\d+) in pack', req['user'])
    belt_pots = int(m.group(1)) if m else 0
    by = {}
    for l, k in zip(labels, kinds):
        by.setdefault(k, []).append(l)
    refill = [l for l in by.get('belt', []) if 'Healing' in text(l)]
    if rng.random() < EPS:
        return rng.choice(labels), 'random'
    if belt_pots == 0 and refill:
        return refill[0], 'refill_belt'
    if dlvl == 3 and not in_quest and by.get('quest') and clvl >= 6 and pots >= 4 and not req.get('enemies'):
        return by['quest'][0], 'enter_tomb'
    choice, why = rule_deep(req, rng)
    if kinds[labels.index(choice)] == 'quest' and not (clvl >= 6 and pots >= 4):
        alt = by.get('explore') or by.get('fight') or by.get('pickup') or [l for l in labels if kinds[labels.index(l)] != 'quest']
        return alt[0], 'not_ready_for_tomb'
    return choice, why


def main():
    bus = Path(sys.argv[1])
    (bus / 'requests').mkdir(parents=True, exist_ok=True)
    (bus / 'responses').mkdir(parents=True, exist_ok=True)
    (bus / 'broker-ready.json').write_text(json.dumps(dict(model='rollin-rule', mode=MODE, margin=MARGIN, restock=RESTOCK, eps=EPS, pid=os.getpid(), time=time.time())))
    served = 0
    while not (bus / 'STOP').exists():
        for p in sorted((bus / 'requests').glob('*.json')):
            try:
                req = json.loads(p.read_text())
            except Exception:
                continue
            rng = random.Random(req['id'])
            choice, why = ({'deep': rule_deep, 'deep3': rule_deep3}.get(MODE, rule))(req, rng)
            res = dict(id=req['id'], choice=choice, confidence=1.0, model='rollin-rule', why=why)
            tmp = bus / 'responses' / (p.stem + '.tmp')
            tmp.write_text(json.dumps(res))
            os.replace(tmp, bus / 'responses' / (p.stem + '.json'))
            p.unlink()
            served += 1
        time.sleep(0.02)
    (bus / 'broker-stopped.json').write_text(json.dumps(dict(served=served, time=time.time())))


if __name__ == '__main__':
    main()

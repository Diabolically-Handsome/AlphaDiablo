"""Option layer: public state -> enumerated, executable, grounded options.

The strategist no longer writes coordinates. Code lists every option that the
frozen executor can actually run from the current public state; the model only
returns a label. Each option compiles to the executor goal format already used
by the Astra runs (move / fight / service command lists), so no new actuator
exists. Only public observation and the remembered known map are used.
"""
from collections import deque
import copy
import math

LABELS = [chr(c) for c in range(ord('A'), ord('Z') + 1)] + [chr(c) for c in range(ord('a'), ord('z') + 1)]
MAX_OPTIONS = len(LABELS)  # 52

NPC_NAMES = {
    'smith': 'Griswold the blacksmith (buys/sells/repairs gear)',
    'healer': 'Pepin the healer (free healing, sells potions)',
    'witch': 'Adria the witch (sells potions/scrolls, buys magic items)',
    'cain': 'Deckard Cain (identifies items)',
    'other': 'townsperson',
}
EXIT_TEXT = {0: 'stairs DOWN to the next level', 1: 'stairs UP to the previous level/town', 2: 'exit back to the main dungeon'}
SECTORS = ['east', 'south-east', 'south', 'south-west', 'west', 'north-west', 'north', 'north-east']
# Equipment list index -> body slot (DevilutionX inv_body_loc order).
SLOT = {0: 'head', 1: 'left ring', 2: 'right ring', 3: 'amulet', 4: 'weapon hand', 5: 'shield hand', 6: 'body'}
# Item location code -> item type and the body slots it can occupy.
LOC_NAME = {1: 'one-handed', 2: 'two-handed', 3: 'body armor', 4: 'helm', 5: 'ring', 6: 'amulet'}
LOC_SLOTS = {1: (4, 5), 2: (4, 5), 3: (6,), 4: (0,), 5: (1, 2), 6: (3,)}
# 2026-09-23 equip-swap fix (gate9-new-butcher s4150011 and s4150029 stopped with no_progress_cycle_40 after 37 and 36
# equips in a row). The bridge's equip is a shift-click on the pack item (bridge-r3 manual_control.hpp:305-318), i.e.
# SRC inv.cpp CheckInvCut (:829-844) then AutoEquip (:1353-1366), which puts the item into the first free body slot
# in index order, the left hand (index 4) before the right hand (index 5). So a shield equipped while a two-hander is
# worn (the two-hander comes off first) lands in index 4, and the next one-handed weapon in index 5: in all 451 logged
# games every one of the 105 entries into 'weapon hand: Buckler' came that way. The engine treats the two hands alike
# (SRC items.cpp:2833-2840 adds up both hands' damage and armour; block and weapon type come from either hand,
# player.h:896-902 isHoldingItem, items.cpp:2675-2705; durability loss follows the item, player.cpp:465-525), and a
# later equip replaces the item of the same class in whichever hand holds it (inv.cpp:829-844). The prompt therefore
# names a hand by what it holds: a shield is shown in the shield hand (5), any other hand item in the weapon hand (4).
# With the usual hands (a weapon or two-hander in 4, a shield in 5) that is the item's own index, so those prompts
# stay byte for byte the same.


def is_shield(it):
    """The shield rule of the equip option: a one-handed item with armor and no damage."""
    return it.get('location') == 1 and bool(it.get('armor')) and not it.get('max_damage')


def shown_slot(it, equipment):
    """SLOT index worn item it is shown in: its own index, except that a hand item is shown by what it is (a shield in
    the shield hand, anything else in the weapon hand), unless the other hand's item is shown in that hand already."""
    i = it.get('index')
    if i not in (4, 5):
        return i
    role = 5 if is_shield(it) else 4
    if role != i and any(not e['empty'] and e.get('index') == role and (5 if is_shield(e) else 4) == role for e in equipment):
        return i
    return role


def shown_equipment(state):
    """[(shown slot, item)] of the worn items, in list order; sorted by shown slot only when a hand item is shown in
    the other hand (the usual hands keep their order byte for byte)."""
    eq = [e for e in state['equipment'] if not e['empty']]
    out = [(shown_slot(e, eq), e) for e in eq]
    if any(s != e.get('index') for s, e in out):
        out.sort(key=lambda p: p[0])
    return out


# 2026-09-23 round 9 (live_runner passes max_main_depth=MAX_MAIN_DEPTH): the bridge never lets the hero below main
# dungeon level 3. Every live game runs it in manual mode (skeleton-king-dual-brain-20260921-r16/session.py:113
# manual_configure -> bridge-r3/bridge-src/diablogym.cpp:3163-3165 gManualControl), whose episode reset
# (diablogym.cpp:2154 ResetResourceEpisode -> resource_protocol.hpp:334-335) installs ManualTransitionGuard
# (manual_control.hpp:14-24: 'return target >= 0 && target <= 3' for every main-level transition). Standing on the
# stairs down calls StartNewLvl(WM_DIABNEXTLVL, currlevel + 1) (SRC levels/trigs.cpp:879-885), which returns at once
# when the guard refuses (SRC player.cpp:2912-2915). So the stairs down of level 3 lead nowhere: in all 425 logged
# games no scene below level 3 was reached, and every 'Take the stairs DOWN' on level 3 that completed (355) left the
# hero on level 3 (heldout-butcher-v3/s4150042: 35 in a row, 0 ticks each, until the cycle stop). The limit belongs to
# that bridge build (bridge_sha256 05fc3009..., in all 984 native/started.json under $AD_ROOT
# on 2026-09-23): live_runner.py BRIDGE_MAX_MAIN_DEPTH refuses '--menu r9' with any other build, so a rebuilt bridge
# that allows deeper levels cannot keep the stairs hidden.
# Exit message codes are the engine's interface_mode (SRC interfac.h:25-27; manual_control.hpp:157): 0 = next level.
MAX_MAIN_DEPTH = 3
EXIT_NEXT_LEVEL = 0
# 2026-09-24 (r9-work/butcher-l2-down; live_runner passes butcher_level=BUTCHER_LEVEL with '--menu r9'): the Butcher is
# only ever on main dungeon level 2. DevilutionX (alphadiablo-dev 34c4cfc2) assets/txtdata/quests/questdat.tsv:8 gives
# The Butcher qdlvl 2 and qdmultlvl 2, which become Quest::_qlevel (quests.cpp:223-226); Quest::IsAvailable is false
# on a set level or when currlevel != _qlevel (quests.cpp:915-921), and the Butcher is placed only when it is true
# (monster.cpp:535-537 PlaceQuestMonsters, 3459-3460 MT_CLEAVER). On the Butcher mission the stairs down of main level
# 2 therefore only lead out of the mission: any decision on level 3 or deeper is a boundary violation (PREREG-ROUND9.md
# 4.1 and 4.4 item 3a, milestones d3). gate9-new-r2-butcher (d11facts5, facts v5): s4150012 took 'Take the stairs DOWN'
# on level 2 under 'Pace: ... Fighting the Butcher (on this level): ready', reached level 3, went back to town and
# looped until tick_budget (decisions ~700-776); s4150018 and s4150019 also reached level 3 before killing him.
# With butcher_level, 'Take / Walk next to the stairs DOWN' is not offered on that main level on the Butcher mission;
# the Skeleton King mission, the other levels and quest areas keep their menu.
BUTCHER_LEVEL = 2


def scene_key(state):
    return ','.join(map(str, state['scene']))


def future(state):
    h = state['hero']
    return (h['future_x'], h['future_y'])


def position(state):
    h = state['hero']
    return (h['x'], h['y'])


def cheb(a, b):
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


def allowed(state, known):
    """Identical to KnownMap.allowed: known walkable tiles minus visible occupants."""
    data = known[scene_key(state)]
    result = {tuple(map(int, k.split(','))) for k, v in data['tiles'].items() if v}
    result -= {(e['x'], e['y']) for e in state['enemies'] + state['npcs']}
    result.add(future(state))
    return result


def distances(state, known):
    """4-neighbour BFS distances, same connectivity as KnownMap.path."""
    ok = allowed(state, known)
    start = future(state)
    dist = {start: 0}
    q = deque([start])
    while q:
        p = q.popleft()
        for dx, dy in ((0, -1), (1, 0), (0, 1), (-1, 0)):
            n = (p[0] + dx, p[1] + dy)
            if n in ok and n not in dist:
                dist[n] = dist[p] + 1
                q.append(n)
    return dist


def approach(dist, x, y, on=False):
    """Nearest reachable tile adjacent (Chebyshev <= 1) to (x, y); the tile itself if on."""
    best = None
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            if (dx, dy) == (0, 0) and not on:
                continue
            p = (x + dx, y + dy)
            if p in dist and (best is None or (dist[p], p) < best):
                best = (dist[p], p)
    return best  # (distance, tile) or None


def sector(origin, target):
    dx, dy = target[0] - origin[0], target[1] - origin[1]
    if dx == 0 and dy == 0:
        return None
    ang = math.degrees(math.atan2(dy, dx)) % 360  # screen y grows south in map coordinates
    return SECTORS[int(((ang + 22.5) % 360) // 45)]


def hp(state):
    h = state['hero']
    return h['hp_fixed'] / 64, h['max_hp_fixed'] / 64


def ui_open(state):
    return bool(state.get('dialog')) or state.get('vendor') not in (None, '', 'none')


def item_actions(state, item, group):
    """Same legality rules as the frozen v1 protocol.item_actions."""
    h = state['hero']
    gold = h['gold']
    acts = []
    ui = ui_open(state)
    if group == 'belt':
        if item.get('heal_kind', 0) > 0:
            acts.append('drink')
        if not ui:
            acts.append('unbelt_exact')
    if group == 'floor' and position(state) == (item.get('x'), item.get('y')):
        acts.append('pickup')
    if group == 'inventory' and not ui:
        if item.get('location') in range(1, 7) and item.get('can_use', False):
            acts.append('equip')
        if item.get('heal_kind', 0) > 0 and any(p['empty'] for p in state['belt']):
            acts.append('belt')
    if group == 'stock' and item.get('fits') and item.get('price', gold + 1) <= gold and not state.get('dialog'):
        acts.append('buy')
    for action, key in (('sell', 'sale_price'), ('repair', 'repair_price'), ('identify', 'identify_price')):
        if key in item and item[key] > 0 and (action == 'sell' or item[key] <= gold) and not state.get('dialog'):
            acts.append(action)
    return acts


def item_name(item):
    if item.get('gold', 0) > 0 and item.get('name') == 'Gold':
        return f"{item['gold']} gold"
    parts = [item.get('name', '?')]
    stats = []
    if item.get('armor'):
        stats.append(f"armor {item['armor']}")
    if item.get('max_damage'):
        stats.append(f"damage {item.get('min_damage', 0)}-{item['max_damage']}")
    if item.get('max_durability'):
        stats.append(f"durability {item.get('durability', 0)}/{item['max_durability']}")
    if item.get('identified') is False:
        stats.append('unidentified')
    bonus = [f"{k[6:]} {v:+}" for k, v in sorted(item.items()) if k.startswith('bonus_') and v and item.get('identified')]
    stats += bonus[:3]
    if not item.get('can_use', True):
        stats.append('cannot use')
    if stats:
        parts.append('(' + ', '.join(stats) + ')')
    return ' '.join(parts)


def identity(item):
    return tuple(item['identity'])


# 2026-09-23 round 9 (calm_stats): "enemy adjacent" as the training labels define it (TEACHER.md:48, round-9 section:
# an enemy marked 'adjacent/in attack range' or '0 steps' on the Visible enemies line, or an 'Attack ...' option 'in
# attack range now' or '0 steps away'; dagger_round9.py enemy_adjacent, 14:35 version; ROUND9-DATA-PLAN.md section 3
# still says 0-1 steps, TEACHER.md and dagger_round9.py say 0). It is read from the numbers the prompt shows, so that a
# logged prompt tells whether the rule applied: the Visible enemies line lists the first SHOWN_ENEMIES living enemies
# with '(adjacent/in attack range)' (attack_legal) or '(N steps)' (render.py:68-70), and the fight options the
# FIGHT_OPTIONS nearest, King first, with 'in attack range now' or 'N steps away' (below). N is the BFS distance to the
# nearest reachable tile next to the enemy (approach; 0 = the hero's tile is next to it). tests_round9_menu.py checks
# that enemy_adjacent(state) equals dagger_round9.enemy_adjacent(prompt, options) on synthetic and Astra snapshot states.
CLOSE_STEPS = 0
SHOWN_ENEMIES = 10
FIGHT_OPTIONS = 8


def enemy_steps(dist, e):
    a = approach(dist, e['x'], e['y'])
    return a[0] if a else 999


def enemy_adjacent(enemies, dist):
    """An enemy listed in the prompt that is in attack range or at most CLOSE_STEPS steps from its attack tile."""
    ranked = sorted(enemies, key=lambda e: (not e.get('king'), enemy_steps(dist, e), e['id']))
    return any(e.get('attack_legal') or enemy_steps(dist, e) <= CLOSE_STEPS
               for e in enemies[:SHOWN_ENEMIES] + ranked[:FIGHT_OPTIONS])


class Option(dict):
    """kind, text, goal (executor goal without id/state_id/reason), key, dest."""


def _service(commands, **extra):
    goal = dict(mode='service', commands=commands, allow_visible_enemies=True, allow_unspent=True)
    goal.update(extra)
    return goal


def _item_cmd(kind, group, item, state):
    cmd = dict(kind=kind, group=group, args=dict(index=item['index'], identity=list(item['identity'])))
    if kind == 'buy':
        cmd['args']['vendor'] = item['vendor']
    if kind in ('sell', 'repair', 'identify'):
        price = item[{'sell': 'sale_price', 'repair': 'repair_price', 'identify': 'identify_price'}[kind]]
        cmd['args'].update(equipped=group == 'equipment', price=price)
        if kind == 'repair':
            cmd['args']['durability'] = item['durability']
        if kind == 'sell':
            cmd['args']['vendor'] = state['vendor']
    return cmd


def enumerate_options(state, known, memory=None, operated=(), mission='skeleton_king', variant='v1',
                      max_main_depth=None, calm_stats=False, hidden=None, butcher_level=None):
    """Return (options, dropped) with at most MAX_OPTIONS options, deterministic order.

    Round-9 menu rules (2026-09-23; live_runner turns them on, the defaults give the round-8 menu byte for byte,
    which relabel.py's replays of old Astra snapshots rely on):
      memory['bought']: identities of items bought during the current town visit; no 'Sell' option for them
          (round 8: 103 buy-then-sell, 18,072 gold lost, FAILURE-CENSUS-20260923.md section 5).
      max_main_depth: no 'Take / Walk next to the stairs DOWN' on a main level whose next level the bridge refuses
          (MAX_MAIN_DEPTH above).
      calm_stats: no 'Spend 1 attribute point' while an enemy is adjacent (enemy_adjacent); 70% of round-8 points
          were spent with enemies in view (FAILURE-CENSUS-20260923.md section 7); a case: d1-test #193, 5 unspent
          points with two enemies adjacent.
      butcher_level: on the Butcher mission, no 'Take / Walk next to the stairs DOWN' on this main level (BUTCHER_LEVEL
          above: he lives there, deeper is outside the mission).
    hidden: a list; each option a round-9 rule leaves out is appended as dict(rule, kind, text) (rule 'bought',
    'stairs_down_refused', 'stat_enemy_adjacent' or 'butcher_level_stairs_down'), for live_runner's menu_hidden log
    field."""
    memory = memory or {}
    dist = distances(state, known)
    here = future(state)
    cur, mx = hp(state)
    data = known[scene_key(state)]
    ui = ui_open(state)
    out = []

    def add(kind, text, goal, key, dest=None, priority=50):
        out.append(Option(kind=kind, text=text, goal=goal, key=key, dest=list(dest) if dest else None, priority=priority))

    def hide(rule, kind, text):
        if hidden is not None:
            hidden.append(dict(rule=rule, kind=kind, text=text))

    enemies = [e for e in state['enemies'] if e.get('hp', 1) > 0]

    # Resume an interrupted move/fight goal.
    current = memory.get('current')
    if current and current.get('mode') in ('move', 'fight'):
        add('resume', 'Resume the interrupted goal: ' + memory.get('current_text', current['mode']), copy.deepcopy(current), ('resume',), priority=1)

    # Drinking (belt healing potions, only when not at full health).
    if cur < mx:
        seen = set()
        for it in state['belt']:
            if it['empty'] or it.get('heal_kind', 0) <= 0 or it['heal_kind'] in seen:
                continue
            seen.add(it['heal_kind'])
            label = {1: 'healing potion', 2: 'full healing potion'}.get(it['heal_kind'], 'rejuvenation potion')
            add('drink', f"Drink a {label} now (HP {cur:.0f}/{mx:.0f})", _service([_item_cmd('drink', 'belt', it, state)]),
                ('drink',), priority=2)

    # Fighting.
    def edist(e):
        a = approach(dist, e['x'], e['y'])
        return a[0] if a else 999
    ranked = sorted(enemies, key=lambda e: (not e.get('king'), edist(e), e['id']))
    for e in ranked[:FIGHT_OPTIONS]:
        tag = ' [SKELETON KING]' if e.get('king') else ''
        reach = 'in attack range now' if e.get('attack_legal') else (f'{edist(e)} steps away' if edist(e) < 999 else 'no known path')
        add('fight', f"Attack {e['name']}{tag} #{e['id']} (hp {e['hp']}/{e['max_hp']}, {reach})",
            dict(mode='fight', target_ids=[e['id']], max_ticks=400, allow_unspent=True), ('fight', e['id']), priority=5)
    if len(enemies) >= 2:
        ids = [e['id'] for e in sorted(enemies, key=lambda e: (edist(e), e['id']))]
        add('fight_all', f"Fight all {len(enemies)} visible enemies, nearest first",
            dict(mode='fight', target_ids=ids, max_ticks=400, allow_unspent=True), ('fight_all',), priority=6)
    if enemies and position(state) == future(state):
        # The executor refuses to hold while the hero is still stepping (point != future tile);
        # offering it then fails with 0 ticks and the game never advances.
        ids = [e['id'] for e in sorted(enemies, key=lambda e: (edist(e), e['id']))]
        add('hold', "Hold this tile and only strike enemies that come adjacent (use in doorways/corridors)",
            dict(mode='fight', target_ids=ids, max_ticks=400, allow_unspent=True, hold_position=list(position(state))), ('hold',), priority=7)

    # Tactical fallback: back off from enemies, or retreat to a narrow tile so they come one at a time.
    if enemies:
        ok = allowed(state, known)
        epos = [(e['x'], e['y']) for e in enemies]

        def emin(p):
            return min(cheb(p, q) for q in epos)

        def narrow(p):
            n = [(p[0] + dx, p[1] + dy) in ok for dx, dy in ((0, 1), (0, -1), (1, 0), (-1, 0))]
            return (n[0] and n[1] and not n[2] and not n[3]) or (n[2] and n[3] and not n[0] and not n[1])

        now = emin(here)
        back = [(emin(p), -d, p) for p, d in dist.items() if 0 < d <= 8]
        if back:
            e_best, nd, p = max(back)
            if e_best >= now + 3:
                add('fallback', (f"Back off away from the enemies ({-nd} steps, ends {e_best} tiles from the nearest one)" if variant == 'v1'
                                 else f"Walk away from the enemies ({-nd} steps)"),
                    dict(mode='move', targets=[list(p)], max_ticks=400, allow_unspent=True), ('fallback', 'away'), dest=p, priority=4)
        chokes = sorted((d, p) for p, d in dist.items() if 0 < d <= 15 and narrow(p) and emin(p) > now and emin(p) >= 3)
        taken = []
        for d, p in chokes:
            if any(cheb(p, q) <= 3 for q in taken):
                continue
            taken.append(p)
            add('fallback', (f"Fall back to a narrow doorway/corridor tile {d} steps away (enemies must come one at a time; then hold it)" if variant == 'v1'
                             else f"Walk back to a narrow doorway/corridor tile {d} steps away"),
                dict(mode='move', targets=[list(p)], max_ticks=400, allow_unspent=True), ('fallback', 'choke', len(taken)), dest=p, priority=4)
            if len(taken) == 2:
                break

    # Earned attribute points (calm_stats: only with no enemy adjacent; the points stay, they are offered again as
    # soon as no listed enemy is in attack range or CLOSE_STEPS steps away).
    if state['hero'].get('unspent_stats', 0) > 0:
        calm = not (calm_stats and enemy_adjacent(enemies, dist))
        for attr in ('strength', 'dexterity', 'vitality', 'magic'):
            if calm:
                add('stat', f"Spend 1 attribute point on {attr}", _service([dict(kind='stat', args=dict(attribute=attr))]), ('stat', attr), priority=8)
            else:
                hide('stat_enemy_adjacent', 'stat', f"Spend 1 attribute point on {attr}")

    # Shop / dialog services (only while a vendor window is open).
    if ui:
        add('dismiss', 'Close the shop/dialog window', _service([dict(kind='dismiss', args={})]), ('dismiss',), priority=9)
        groups = {}
        for it in state['stock']:
            if it['empty'] or 'buy' not in item_actions(state, it, 'stock'):
                continue
            sig = (it.get('name'), it.get('price'), it.get('armor'), it.get('max_damage'), it.get('heal_kind'))
            groups.setdefault(sig, []).append(it)
        bought = memory.get('bought') or frozenset()  # items bought during this town visit (live_runner)
        buys = sorted(groups.values(), key=lambda g: (g[0].get('heal_kind', 0) <= 0, not g[0].get('can_use', True), -g[0].get('price', 0)))
        for g in buys[:14]:
            it = g[0]
            add('buy', f"Buy {item_name(it)} for {it['price']} gold" + (f" ({len(g)} in stock)" if len(g) > 1 else ''),
                _service([_item_cmd('buy', 'stock', it, state)]), ('buy', frozenset(identity(x) for x in g)), priority=10)
        for group in ('inventory', 'equipment'):
            for it in state[group]:
                if it['empty']:
                    continue
                acts = item_actions(state, it, group)
                where = 'equipped ' if group == 'equipment' else ''
                if 'sell' in acts and it.get('name') != 'Gold':
                    text = f"Sell {where}{item_name(it)} for {it['sale_price']} gold"
                    if identity(it) in bought:
                        hide('bought', 'sell', text)
                    else:
                        add('sell', text, _service([_item_cmd('sell', group, it, state)]), ('sell', identity(it)), priority=11)
                if 'repair' in acts:
                    add('repair', f"Repair {where}{item_name(it)} for {it['repair_price']} gold", _service([_item_cmd('repair', group, it, state)]),
                        ('repair', identity(it)), priority=11)
                if 'identify' in acts:
                    add('identify', f"Identify {where}{item_name(it)} for {it['identify_price']} gold", _service([_item_cmd('identify', group, it, state)]),
                        ('identify', identity(it)), priority=12)
    else:
        # Inventory management outside shops.
        for it in state['inventory']:
            if it['empty']:
                continue
            acts = item_actions(state, it, 'inventory')
            if 'equip' in acts:
                slots = LOC_SLOTS.get(it.get('location'), ())
                if it.get('location') == 1:
                    slots = (5,) if it.get('armor') and not it.get('max_damage') else (4,)
                worn = [e for s, e in shown_equipment(state) if s in slots]  # hands by what they hold (shown_slot)
                now = '; '.join(item_name(w) for w in worn) if worn else 'nothing'
                slot = ' / '.join(SLOT[i] for i in slots) or 'slot'
                add('equip', f"Equip {item_name(it)} in {slot} (currently {now})", _service([_item_cmd('equip', 'inventory', it, state)]),
                    ('equip', identity(it)), priority=13)
            if 'belt' in acts:
                add('belt', f"Move {item_name(it)} from pack to belt", _service([_item_cmd('belt', 'inventory', it, state)]),
                    ('belt', identity(it)), priority=14)
        for it in state['belt']:
            if it['empty'] or it.get('heal_kind', 0) > 0:
                continue
            if 'unbelt_exact' in item_actions(state, it, 'belt'):
                add('unbelt', f"Move {item_name(it)} from belt to pack", _service([_item_cmd('unbelt_exact', 'belt', it, state)]),
                    ('unbelt_exact', identity(it)), priority=15)

    # Items on the floor (visible and remembered in this scene).
    floor = {}
    for it in state['floor']:
        if not it['empty']:
            floor[identity(it)] = it
    for tok, it in data.get('floor_memory', {}).items():
        floor.setdefault(identity(it), it)
    picks = []
    for ident, it in floor.items():
        p = (it['x'], it['y'])
        if p not in dist:
            continue
        picks.append((dist[p], ident, it))
    for d, ident, it in sorted(picks, key=lambda v: (v[0], v[1]))[:8]:
        cmds = [] if position(state) == (it['x'], it['y']) else [dict(kind='move', target=[it['x'], it['y']], max_ticks=1000)]
        cmd = dict(kind='pickup', group='floor', args=dict(index=it['index'], identity=list(it['identity'])))
        cmds.append(cmd)
        add('pickup', f"Pick up {item_name(it)} ({d} steps away)", _service(cmds), ('pickup', ident), dest=(it['x'], it['y']), priority=20)

    # NPCs (town).
    npcs = {n['id']: n for n in data.get('npcs_memory', {}).values()}
    for n in state['npcs']:
        npcs[n['id']] = n
    for nid, n in sorted(npcs.items()):
        a = approach(dist, n['x'], n['y'])
        if not a:
            continue
        name = NPC_NAMES.get(n.get('kind'), 'townsperson')
        cmds = [] if cheb(position(state), (n['x'], n['y'])) <= 1 and ui is False else [dict(kind='move', target=list(a[1]), max_ticks=1000)]
        cmds.append(dict(kind='talk', args=dict(id=nid)))
        verb = 'Talk to' if not cmds[:-1] else 'Walk to and talk to'
        if n.get('kind') not in ('smith', 'healer', 'witch', 'cain'):
            # Non-vendor townsfolk open a store-type menu that the state reports as neither dialog nor
            # vendor, so no close option would appear while the bridge rejects every walk. Close it at once.
            cmds.append(dict(kind='dismiss', args={}))
        add('npc', f"{verb} {name} ({a[0]} steps away)", _service(cmds), ('npc', nid), dest=(n['x'], n['y']), priority=25)

    # Objects: doors, chests, barrels, sarcophagi (visible and remembered, not yet operated here).
    objs = {o['id']: o for o in data.get('objects_memory', {}).values()}
    for o in state['objects']:
        objs[o['id']] = o
    cands = []
    for oid, o in objs.items():
        if oid in operated:
            continue
        a = approach(dist, o['x'], o['y'])
        if not a:
            continue
        cands.append((a[0], oid, o, a[1]))
    for d, oid, o, tile in sorted(cands)[:10]:
        kind = {'door': 'door', 'barrel': 'barrel'}.get(o.get('kind'), 'chest/sarcophagus/object')
        cmds = [] if cheb(position(state), (o['x'], o['y'])) <= 1 else [dict(kind='move', target=list(tile), max_ticks=1000)]
        cmds.append(dict(kind='operate', args=dict(id=oid)))
        add('object', f"Open/operate the {kind} #{oid} ({d} steps away)", _service(cmds), ('object', oid), dest=(o['x'], o['y']), priority=30)

    # Exits and quest entrances.
    depth, set_id = state['scene'][0], state['scene'][1]  # manual_control.hpp:106: (main depth, set level or 0)
    for e in data.get('exits', {}).values():
        # Stairs down to a level the bridge refuses (MAX_MAIN_DEPTH) cannot change the level: not offered.
        refused = (max_main_depth is not None and e.get('message') == EXIT_NEXT_LEVEL and not set_id
                   and depth + 1 > max_main_depth)
        # Butcher mission: no stairs down from his main level (BUTCHER_LEVEL): deeper is outside the mission.
        beyond = (butcher_level is not None and mission == 'butcher' and e.get('message') == EXIT_NEXT_LEVEL
                  and not set_id and depth == butcher_level)
        rule = 'stairs_down_refused' if refused else 'butcher_level_stairs_down' if beyond else None
        p = (e['x'], e['y'])
        if p in dist:
            text = f"Take the {EXIT_TEXT.get(e.get('message'), 'exit')} ({dist[p]} steps away)"
            if rule:
                hide(rule, 'exit', text)
            else:
                add('exit', text, dict(mode='move', targets=[list(p)], max_ticks=800, allow_unspent=True), ('exit', e.get('message'), p),
                    dest=p, priority=35)
        else:
            a = approach(dist, *p)
            # 2026-09-23 08:50: not when already next to it (0 steps): that move does nothing (king3-v3/s4150019
            # chose it 35 times in a row, 0 ticks each, until the cycle detector stopped the game).
            if a and a[0] > 0:
                text = f"Walk next to the {EXIT_TEXT.get(e.get('message'), 'exit')} ({a[0]} steps away)"
                if rule:
                    hide(rule, 'exit', text)
                else:
                    add('exit', text, dict(mode='move', targets=[list(a[1])], max_ticks=800, allow_unspent=True),
                        ('exit', e.get('message'), p), dest=p, priority=35)
    for e in data.get('quest_entrances', {}).values():
        p = (e['x'], e['y'])
        if p in dist:
            add('quest', f"Enter the quest area (Skeleton King's tomb) ({dist[p]} steps away)",
                dict(mode='move', targets=[list(p)], max_ticks=800, allow_unspent=True), ('quest', p), dest=p, priority=36)

    # Exploration: nearest frontier in each of eight directions.
    tiles = data['tiles']
    best = {}
    for p, d in dist.items():
        unknown = sum(f'{p[0]+dx},{p[1]+dy}' not in tiles for dx, dy in ((0, 1), (0, -1), (1, 0), (-1, 0)))
        if not unknown or d == 0:
            continue
        s = sector(here, p)
        if s is None:
            continue
        cand = (d, -unknown, p)
        if s not in best or cand < best[s]:
            best[s] = cand
    for s in SECTORS:
        if s in best:
            d, nu, p = best[s]
            add('explore', f"Explore toward the {s} (unexplored edge {d} steps away)",
                dict(mode='move', targets=[list(p)], max_ticks=800, allow_unspent=True), ('explore', s), dest=p, priority=40)

    add('stop', 'Stop and save the game', dict(mode='stop'), ('stop',), priority=99)

    if variant == 'v2':
        # Fight options before fallback; fallback after hold.
        for o in out:
            if o['kind'] == 'fallback':
                o['priority'] = 7.5
    out.sort(key=lambda o: o['priority'])
    kept, dropped = out[:MAX_OPTIONS], out[MAX_OPTIONS:]
    if not any(o['kind'] == 'stop' for o in kept):
        kept[-1] = [o for o in out if o['kind'] == 'stop'][0]
    for label, o in zip(LABELS, kept):
        o['label'] = label
    return kept, dropped


def compile_goal(option, goal_id, state_id, reason):
    goal = copy.deepcopy(option['goal'])
    goal.update(id=goal_id, state_id=state_id, reason=reason)
    return goal

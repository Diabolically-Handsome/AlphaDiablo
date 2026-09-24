"""Deterministic English prompt for an option-choosing strategy brain."""
from options import hp, item_name, scene_key, future, approach, distances, SLOT, shown_equipment

SYSTEM = (
    "You are the strategy brain of a Diablo I warrior in a normal-difficulty game. "
    "A separate executor carries out whatever option you pick: it walks known paths, attacks the targets you choose, "
    "and performs shop/item actions exactly. You decide routes, fights, retreats, shopping, equipment, attribute points and drinking potions; "
    "nobody else drinks for you. Game time is paused while you think. Control returns to you when something new happens "
    "(new enemy, damage taken, level change, goal finished or blocked). "
    "Play like an experienced human: keep health up, retreat or hold a doorway when outnumbered, buy healing potions and repair gear in town, "
    "collect gold and useful items, explore to find the stairs, and only fight bosses when strong enough. "
    "Reply with exactly one option letter and nothing else."
)

SYSTEM_V2 = (
    "You are the strategy brain of a Diablo I warrior in a normal-difficulty game. "
    "A separate executor carries out whatever option you pick: it walks known paths, attacks the targets you choose, "
    "and performs shop/item actions exactly. You decide routes, fights, retreats, shopping, equipment, attribute points and drinking potions; "
    "nobody else drinks for you. Game time is paused while you think. Control returns to you when something new happens "
    "(new enemy, damage taken, level change, goal finished or blocked). "
    "Normal monsters on early levels are usually worth killing for experience and loot; drink a healing potion when health is low; "
    "back off only when health is low or you are clearly losing; hold a narrow tile against a strong boss with escorts. "
    "Keep enough healing potions, repair and upgrade gear in town, collect gold and items, explore to find the stairs down. "
    "Reply with exactly one option letter and nothing else."
)
SYSTEMS = {'v1': SYSTEM, 'v2': SYSTEM_V2}

MISSION = {
    'skeleton_king': "Kill the Skeleton King (his tomb entrance is on dungeon level 3) and stay alive.",
    'butcher': "Kill the Butcher (he is on dungeon level 2, in a room full of blood and corpses) and stay alive.",
}


def location_text(state):
    floor, set_id = state['scene']
    if floor == 0:
        return 'Town (Tristram). The cathedral entrance leads to dungeon level 1.'
    if set_id:
        return f"Quest area below dungeon level {floor} (the Skeleton King's tomb)."
    return f'Dungeon level {floor}.'


def hero_text(state):
    h = state['hero']
    cur, mx = hp(state)
    belt = sum(1 for it in state['belt'] if not it['empty'] and it.get('heal_kind', 0) > 0)
    pack = sum(1 for it in state['inventory'] if not it['empty'] and it.get('heal_kind', 0) > 0)
    return (f"Hero: level {h['level']}, HP {cur:.0f}/{mx:.0f} ({100*cur/max(mx,1):.0f}%), gold {h['gold']}, armor {h['armor']}, "
            f"damage {h['min_damage']}-{h['max_damage']}, healing potions {belt} in belt and {pack} in pack, "
            f"unspent attribute points {h.get('unspent_stats', 0)}. Strength {h['strength']}, dexterity {h['dexterity']}, vitality {h['vitality']}.")


def gear_text(state):
    # A hand is named by what it holds (options.shown_slot: the engine may put a shield in the left-hand slot).
    parts = [f"{SLOT.get(s, 'slot')}: {item_name(it)}" for s, it in shown_equipment(state)]
    return 'Equipped: ' + ('; '.join(parts) if parts else 'nothing') + '.'


def enemies_text(state, known):
    enemies = [e for e in state['enemies'] if e.get('hp', 1) > 0]
    if not enemies:
        return 'Visible enemies: none.'
    dist = distances(state, known)
    rows = []
    for e in enemies[:10]:
        a = approach(dist, e['x'], e['y'])
        where = 'adjacent/in attack range' if e.get('attack_legal') else (f'{a[0]} steps' if a else 'unreachable')
        king = ' SKELETON KING' if e.get('king') else ''
        rows.append(f"{e['name']}{king} #{e['id']} hp {e['hp']}/{e['max_hp']} ({where})")
    more = f' (+{len(enemies)-10} more)' if len(enemies) > 10 else ''
    return f'Visible enemies ({len(enemies)}): ' + '; '.join(rows) + more + '.'


def render(state, known, options, mission, interrupted=None, recent=()):
    lines = [f"Mission: {MISSION[mission]}", f"Location: {location_text(state)}", hero_text(state), gear_text(state),
             enemies_text(state, known)]
    if state.get('vendor') not in (None, '', 'none'):
        lines.append(f"A shop window is open ({state['vendor']}).")
    if interrupted:
        lines.append('Interrupted goal: ' + interrupted)
    if recent:
        lines.append('Recent results (oldest first): ' + ' | '.join(recent))
    lines.append('Options:')
    for o in options:
        lines.append(f"{o['label']}. {o['text']}")
    lines.append('Answer with one letter.')
    return '\n'.join(lines)

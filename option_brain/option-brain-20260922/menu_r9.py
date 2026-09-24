"""Round-9 menu rules for rows made without them (2026-09-23; review of the round-9 runner, finding 2).

Rows written by 'live_runner.py --menu r9' already follow the round-9 menu and carry what it hid (menu_hidden) and
what was bought during the town visit (menu_bought). Older rows (round 8 and before, and the training rows made from
them: relabel-d*, extract_round9.py, dagger_round9.py) were made with the round-8 menu. For such a row this module
finds the options the round-9 menu (options.enumerate_options with max_main_depth, calm_stats, memory['bought'])
would not have offered, from what the row shows:
  stairs_down_refused  'Take / Walk next to the stairs DOWN' on main dungeon level >= MAX_MAIN_DEPTH, from the row's
                       scene or else the prompt's 'Location: Dungeon level N.' line (render.py location_text). Exact.
  stat_enemy_adjacent  'Spend 1 attribute point' with an enemy adjacent: dagger_round9.enemy_adjacent on the prompt and
                       options, which the live rule options.enemy_adjacent equals (tests_round9_menu.py). Exact.
  bought               'Sell X' during the town visit in which X was bought (a completed 'Buy X' row earlier in the same
                       visit of the same game). Only with the game's rows (game_hidden). By item name, as the log keeps
                       no identities, so an identical item owned before the visit is hidden as well (marked approx).
  butcher_level_stairs_down  (2026-09-24, options.py BUTCHER_LEVEL; enumerate_options butcher_level) 'Take / Walk next
                       to the stairs DOWN' on main dungeon level BUTCHER_LEVEL on the Butcher mission (the prompt's
                       'Mission:' line, render.py MISSION). Exact. Rows logged with '--menu r9' before this rule existed
                       get it too (game_hidden); rows logged with it no longer list those options, so nothing changes.
without_hidden() then removes them and re-letters the options, the prompt's option block, label / acceptable / choice /
probs; when the label or the logged choice is one of them it reports a conflict instead, and the caller decides.
The stat rule of the training labels (dexterity to 45, then vitality) is dagger_round9.py's, not this module's.
usage: rows, conflict = without_hidden(row, row_hidden(row))            one training or log row
       for row, hidden in game_hidden(rows): ...                         the rows of one decisions.jsonl, in order
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from options import LABELS, MAX_MAIN_DEPTH, EXIT_TEXT, EXIT_NEXT_LEVEL, BUTCHER_LEVEL  # noqa: E402
from render import MISSION  # noqa: E402

_LOCATION = re.compile(r'^Location: Dungeon level (\d+)\.$', re.M)      # render.py location_text, main levels only
_MISSION = re.compile(r'^Mission: (.*)$', re.M)                          # render.py render, first line
_MISSION_KEY = {text: key for key, text in MISSION.items()}
_BUY = re.compile(r'^Buy (.+) for \d+ gold(?: \(\d+ in stock\))?$')      # options.py buy option
_SELL = re.compile(r'^Sell (?:equipped )?(.+) for \d+ gold$')            # options.py sell option
STAIRS_DOWN = tuple(f'{verb} the {EXIT_TEXT[EXIT_NEXT_LEVEL]} (' for verb in ('Take', 'Walk next to'))
OPTIONS_HEAD, OPTIONS_TAIL = '\nOptions:\n', '\nAnswer with one letter.'  # render.py render


def main_depth(row):
    """Main dungeon level of the row's state; None in town and in quest areas."""
    sc = row.get('scene')
    if sc is not None:
        return sc[0] if sc[0] and not sc[1] else None
    m = _LOCATION.search(row['prompt'].split(OPTIONS_HEAD)[0])
    return int(m.group(1)) if m else None


def mission_of(row):
    """'butcher' / 'skeleton_king' from the prompt's 'Mission:' line (render.py MISSION); None if it is not one of them
    (or the row has no prompt)."""
    m = _MISSION.search(row.get('prompt', '').split(OPTIONS_HEAD)[0])
    return _MISSION_KEY.get(m.group(1)) if m else None


def butcher_level_hidden(row, butcher_level=BUTCHER_LEVEL):
    """[dict(rule, label, kind, text)] of the row's stairs-down options that the Butcher-mission rule leaves out."""
    if butcher_level is None or main_depth(row) != butcher_level or mission_of(row) != 'butcher':
        return []
    return [dict(label=o['label'], kind=o['kind'], text=o['text'], rule='butcher_level_stairs_down') for o in row['options']
            if o['kind'] == 'exit' and o['text'].startswith(STAIRS_DOWN)]


def row_hidden(row, max_main_depth=MAX_MAIN_DEPTH, bought_names=frozenset(), butcher_level=BUTCHER_LEVEL):
    """[dict(rule, label, kind, text[, approx])] of the row's options that the round-9 menu leaves out."""
    from dagger_round9 import enemy_adjacent  # the training labels' definition (imported here: it loads facts.py)
    depth = main_depth(row)
    beyond = {h['label'] for h in butcher_level_hidden(row, butcher_level)}
    adjacent = None
    out = []
    for o in row['options']:
        h = dict(label=o['label'], kind=o['kind'], text=o['text'])
        if o['kind'] == 'exit' and depth is not None and depth + 1 > max_main_depth and o['text'].startswith(STAIRS_DOWN):
            out.append(dict(h, rule='stairs_down_refused'))
        elif o['label'] in beyond:
            out.append(dict(h, rule='butcher_level_stairs_down'))
        elif o['kind'] == 'stat':
            if adjacent is None:
                adjacent = enemy_adjacent(row['prompt'], row['options'])
            if adjacent:
                out.append(dict(h, rule='stat_enemy_adjacent'))
        elif o['kind'] == 'sell' and bought_names:
            m = _SELL.match(o['text'])
            if m and m.group(1) in bought_names:
                out.append(dict(h, rule='bought', approx=True))
    return out


def game_hidden(rows, max_main_depth=MAX_MAIN_DEPTH, butcher_level=BUTCHER_LEVEL):
    """[(row, hidden)] for the rows of one game's decisions.jsonl in order. Rows written with '--menu r9' (menu_hidden
    present) already follow the rules of 2026-09-23: hidden = butcher_level_hidden only ([] when they were logged
    with that rule too)."""
    bought = set()
    out = []
    for r in rows:
        if tuple(r['scene']) != (0, 0):
            bought = set()  # the visit is over once the hero is out of town (live_runner.py town_bought)
        out.append((r, butcher_level_hidden(r, butcher_level) if 'menu_hidden' in r
                    else row_hidden(r, max_main_depth, frozenset(bought), butcher_level)))
        m = _BUY.match(r.get('text', '')) if r.get('kind') == 'buy' else None
        if m and r.get('result') == 'completed' and (r.get('receipt') or {}).get('gold_change', 0) < 0:
            bought.add(m.group(1))
    return out


def without_hidden(row, hidden):
    """(new_row, None) with the hidden options removed and everything re-lettered; (row, None) if nothing is hidden;
    (None, conflict) if the row's label or logged choice is hidden. A hidden player_choice (training rows: what the
    student did) becomes None."""
    if not hidden:
        return row, None
    gone = {h['label'] for h in hidden}
    hit = [k for k in ('label', 'choice') if row.get(k) in gone]
    if hit:
        return None, dict(fields=hit, hidden=[h for h in hidden if h['label'] in {row[k] for k in hit}])
    head, sep, rest = row['prompt'].partition(OPTIONS_HEAD)
    block, sep2, end = rest.partition(OPTIONS_TAIL)
    if not sep or not sep2 or block.split('\n') != [f"{o['label']}. {o['text']}" for o in row['options']]:
        raise ValueError(f"prompt option block does not match the options: {row.get('id', row.get('n'))}")
    kept = [o for o in row['options'] if o['label'] not in gone]
    new_label = {o['label']: LABELS[i] for i, o in enumerate(kept)}
    new = dict(row, options=[dict(o, label=new_label[o['label']]) for o in kept])
    new['prompt'] = head + sep + '\n'.join(f"{o['label']}. {o['text']}" for o in new['options']) + sep2 + end
    for k in ('label', 'choice', 'player_choice'):
        if row.get(k) is not None:
            new[k] = new_label.get(row[k])
    if row.get('acceptable') is not None:
        new['acceptable'] = [new_label[x] for x in row['acceptable'] if x in new_label]
    if isinstance(row.get('probs'), dict):
        new['probs'] = {new_label[k]: v for k, v in row['probs'].items() if k in new_label}
    new['menu_r9_hidden'] = hidden
    return new, None

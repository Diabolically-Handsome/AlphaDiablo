"""Menu-level 'no pointless undo' guard (live_runner.py --undo-guard; off by default).

Hides, for a while, options that would simply undo the previous action while nothing has changed:
  1. talk: talking again to a townsperson/trader whose window was just closed, while the material state
     (scene, gold, HP, items, kills) is unchanged since the close;
  2. stairs: taking the stairs (or quest-area exit/entrance) straight back the way the hero just came,
     while nothing changed since arriving, no enemy is visible, HP >= 50% and fewer than 3 decisions
     have been made since arriving; an enemy whose attack option says 'no known path' (shown as
     '(unreachable)') does not count as visible, except the Skeleton King and the Butcher (stairs-loop fix);
  3b. equip (added 2026-09-23 08:05): an equip that takes something off and would put on a worn set already worn
     since the item set last changed (catches 3-step cycles such as sword+shield -> two-hander -> shield alone ->
     sword+shield: the cycle is cut at the two-hander step); an equip into empty slots is never hidden;
     2026-09-23 equip-swap fix: equip_effect reads the hands as the prompt names them (options.shown_slot). The engine
     can hold the shield in the left-hand slot (index 4) and the weapon in the right one (5); an equipped weapon then
     replaces the weapon in slot 5, but the old slot rule predicted that the shield in slot 4 would come off, so
     neither rule 3 nor 3b caught the 2- and 3-step weapon cycles in that configuration (gate9-new-butcher
     s4150011: 37 equips in a row, s4150029: 36);
  3. equip: an equip that would exactly undo the last completed equip (the last swap): put back the one
     item that swap took off and take off what it put on, so the worn set is again the one from before
     that swap; only while the item set and the worn set are unchanged since that swap. Only the last swap
     counts, an equip into a slot that holds nothing is never hidden, and nothing is hidden after a swap
     that took off two items (a two-hander replacing weapon and shield: no single equip restores both).
It never picks an action; it only hides options that change nothing but the clock, like the runner's
existing no-effect equip/pickup block. The runner logs every hidden option text (row 'undo_hidden').

Runner contract:
  hidden(state, options, n) -> set of labels to hide. n = number of the decision about to be made
      (1-based). Called on the option list that would otherwise be shown. Never hides every option.
  observe(before, opt, goal, result, after, n) after decision n was executed. before = the state the
      decision was made in (before the runner's auto-close of an open window), after = the state after
      the goal, result = the runner's final result string.
A block lapses for good at the first change of the relevant signature (it does not come back if the
signature later returns to the old value).

Review fixes against the first draft (backup/undo_guard.before-fix-0974343a.py):
  - Stairs direction came from the option text and a floor comparison that, on leaving a quest area
    (e.g. tomb 3,1 -> 3,0), hid 'stairs DOWN' (a new level, not the way back) and never hid the 'quest'
    option (the actual way back). Direction now comes from the exit message code in the option key
    (options.py add(..., key=('exit', message, p)); 0 down, 1 up, 2 back to the main dungeon) and the
    quest option ('quest', p) is covered.
  - A trader window closed by the runner's auto-close (the brain chose something elsewhere) or replaced
    by a new talk did not count as a close, so talk/auto-close/talk loops were not caught.
  - A block that lapsed could silently return when the signature came back to the old value.
  - Hiding could leave an empty menu (the broker then has no legal label); now nothing is hidden then.
Second review fix (backup/undo_guard.before-swapfix-2fc65f4d.py):
  - The equip rule added a block for every item taken off by any swap since the item set last changed, so
    it also hid equips that are not undos of the last swap. Example: wearing Short Sword + Buckler, equip a
    two-handed Short Bow (both go to the pack), then equip a Club: 'Equip Buckler in shield hand (currently
    nothing)' and 'Equip Short Sword (currently Club)' were hidden, though only 'Equip Short Bow' undoes the
    last swap. Now each completed equip replaces the record (self.swap), and an equip is hidden only if, by
    the slot rule the equip option's '(currently ...)' text uses, it would restore exactly the worn set
    from before the last swap; the record also lapses when the worn set changes.
Stairs-loop fix (r9-work/stairs-loop patch 01, applied with the equip-swap fix on 2026-09-23 for the round-9 re-freeze):
  - Any living enemy in state['enemies'] switched the stairs rule off, also one the hero has no known path to.
    Seed 4150019 shows 2-4 Skeleton Captains '(unreachable)' (no known path to them, hp 1-3) whenever the hero arrives
    on dungeon level 2, so 'Take the stairs UP (3 steps away)' was never hidden there and the brain took it at
    once: gate9-new-king 6 times, gate9-new-butcher 10 times, round-8 butcher1-v3 14 and king3-v3 3 times, the
    games ending in no_progress_cycle_40 (town -> 1 -> 2 -> 1 -> town ...). Now an enemy counts only if it has
    a known path or no attack option in this menu (options.py lists the FIGHT_OPTIONS nearest; the others are
    counted, as before); the Skeleton King and the Butcher always count. Same test as facts.surrounded_v4
    ('Unreachable enemies do not count'). The exposure is bounded: an enemy that walks up to the hero is
    reachable (the hero's tile is next to it), and its first hit changes HP, which lifts the block for good.
"""
from collections import Counter

from options import LOC_SLOTS, ui_open, shown_slot

EXIT_DOWN, EXIT_UP, EXIT_MAIN = 0, 1, 2  # options.EXIT_TEXT message codes
RULES = ('talk', 'stairs', 'equip', 'hold')


def item_sig(state):
    return tuple(sorted(tuple(x.get('identity', ())) for g in ('equipment', 'inventory', 'belt')
                        for x in state[g] if not x['empty']))


def material(state):
    h = state['hero']
    return (tuple(state['scene']), h['gold'], h['hp_fixed'], item_sig(state), state.get('kills', 0))


def enemy_ids(state):
    return frozenset(e['id'] for e in state['enemies'] if e.get('hp', 1) > 0)


def unreachable_ids(options):
    """Ids of the enemies whose attack option in this menu says 'no known path' (options.py: no known tile next to
    the enemy; render.py shows it as '(unreachable)'). An enemy without an attack option is not in the set."""
    out = set()
    for o in options:
        key = o.get('key') or ()
        if o['kind'] == 'fight' and len(key) >= 2 and o['text'].endswith(', no known path)'):
            out.add(key[1])
    return frozenset(out)


def is_boss(e):
    return bool(e.get('king')) or e.get('name') == 'The Butcher'


def stairs_threat(state, unreach=frozenset()):
    """A living enemy that keeps the stairs rule off: every visible one except those in unreach (no known path);
    the Skeleton King and the Butcher always count."""
    return any(e.get('hp', 1) > 0 and (is_boss(e) or e.get('id') not in unreach) for e in state['enemies'])


def is_hold(option):
    """A hold order: the 'hold' option, or 'resume' of an interrupted hold (added 2026-09-23 06:20: in
    king2-d8/s4150021 the guard hid 'Hold this tile' and the brain chose 'Resume the interrupted goal: Hold this
    tile' instead, 29 times in a row, 400 idle ticks each, against enemies behind a closed door)."""
    return option['kind'] == 'hold' or (option['kind'] == 'resume' and 'Hold this tile' in option.get('text', ''))


def worn(state):
    return {tuple(x.get('identity', ())) for x in state['equipment'] if not x['empty']}


def target_slots(it):
    """Body slots (equipment 'index') an equip of pack item it goes to. Same rule as the equip option's
    '(currently ...)' text in options.enumerate_options: a one-handed item with armor and no damage is a
    shield (shield hand), any other one-handed item a weapon (weapon hand); a two-hander takes both hands."""
    if it.get('location') == 1:
        return (5,) if it.get('armor') and not it.get('max_damage') else (4,)
    return LOC_SLOTS.get(it.get('location'), ())


def equip_effect(state, ident):
    """(worn items that come off, worn set afterwards) for equipping pack item ident; None if it is not in the pack."""
    it = next((x for x in state['inventory'] if not x['empty'] and tuple(x.get('identity', ())) == ident), None)
    if it is None:
        return None
    slots = target_slots(it)
    eq = [e for e in state['equipment'] if not e['empty']]
    off = {tuple(e.get('identity', ())) for e in eq if shown_slot(e, eq) in slots}  # hands by what they hold
    if set(slots) & {4, 5}:  # a worn two-hander comes off whenever either hand is equipped
        off |= {tuple(e.get('identity', ())) for e in state['equipment'] if not e['empty'] and e.get('location') == 2}
    return off, (worn(state) - off) | {ident}


def exit_code(option):
    """Message code of an 'exit' option: from its key ('exit', message, p); text only as a fallback."""
    key = option.get('key') or ()
    if len(key) >= 2 and key[1] in (EXIT_DOWN, EXIT_UP, EXIT_MAIN):
        return key[1]
    text = option['text']
    if 'stairs DOWN' in text:
        return EXIT_DOWN
    if 'stairs UP' in text:
        return EXIT_UP
    if 'back to the main' in text:
        return EXIT_MAIN
    return None


def way_back(came_from, scene):
    """Options that lead straight back from scene to came_from: {('exit', code)} and/or {('quest',)}."""
    came_from, scene = tuple(came_from), tuple(scene)
    if came_from[0] < scene[0]:                 # went down (town -> DL1, DLk -> DLk+1)
        back = {('exit', EXIT_UP)}
        if scene[1]:
            back.add(('exit', EXIT_MAIN))
        return back
    if came_from[0] > scene[0]:                 # went up (DLk -> DLk-1 or town)
        return {('exit', EXIT_DOWN)}
    if not came_from[1] and scene[1]:           # entered a quest area on the same floor
        return {('exit', EXIT_UP), ('exit', EXIT_MAIN)}
    if came_from[1] and not scene[1]:           # left a quest area back to the main dungeon
        return {('quest',)}
    return set()


class UndoGuard:
    def __init__(self):
        self.talk_block = {}      # npc id -> material signature when its window was closed
        self.open_npc = None      # npc id whose window a completed talk left open
        self.arrival = None       # dict(scene, came_from, n, sig, back, changed)
        self.swap = None          # the last completed equip: dict(before=worn set, after=worn set, isig=item set after it)
        self.hold_block = None    # (material signature, visible enemy ids) after a hold that changed nothing
        self.worn_seen = set()    # rule 3b: worn sets seen while the item set stayed self.worn_isig
        self.worn_isig = None
        self.counts = Counter()   # hidden option instances per rule
        self.decisions_with_hidden = 0
        self.all_hidden_kept = 0  # decisions where every option qualified and so nothing was hidden
        self.last = {}            # label -> rule, for the last hidden() call

    @property
    def swap_block(self):
        """Identities an equip may not put back now (for logging, e.g. bench/hooks_trace_runner.py): the one
        item the last swap took off, if it took off exactly one; else nothing."""
        s = self.swap
        off = s['before'] - s['after'] if s else frozenset()
        return tuple(off) if len(off) == 1 else ()

    def _undoes_swap(self, state, ident, isig):
        """True if equipping pack item ident would exactly undo the last swap: the item set and the worn set are
        still what that swap left, the swap took off ident and nothing else, and ident's slots now hold exactly
        what the swap put on, so the equip would restore the worn set from before the swap. An equip whose slots
        hold nothing is never an undo."""
        s = self.swap
        if not s or s['isig'] != isig or worn(state) != s['after'] or s['before'] - s['after'] != {ident}:
            return False
        eff = equip_effect(state, ident)
        return eff is not None and bool(eff[0]) and eff[1] == s['before']

    def _rule(self, o, state, sig, isig, n, unreach=frozenset()):
        key = o.get('key') or ()
        kind = o['kind']
        if kind == 'npc' and len(key) >= 2 and key[1] in self.talk_block and self.talk_block[key[1]] == sig:
            return 'talk'
        if kind == 'equip' and len(key) >= 2 and self._undoes_swap(state, tuple(key[1]), isig):
            return 'equip'
        if kind == 'equip' and len(key) >= 2 and self.worn_isig == isig and self.worn_seen:
            eff = equip_effect(state, tuple(key[1]))
            # An equip into empty slots (nothing comes off) is never hidden: a hand must never be left empty.
            if eff is not None and eff[0] and eff[1] != worn(state) and frozenset(eff[1]) in self.worn_seen:
                return 'equip'
        if is_hold(o) and self.hold_block is not None and self.hold_block == (sig, enemy_ids(state)):
            return 'hold'
        a = self.arrival
        if kind in ('exit', 'quest') and a and not a['changed'] and tuple(state['scene']) == a['scene']:
            which = ('exit', exit_code(o)) if kind == 'exit' else ('quest',)
            h = state['hero']
            enemies = stairs_threat(state, unreach)
            if (which in a['back'] and not enemies and 2 * h['hp_fixed'] >= max(h['max_hp_fixed'], 1)
                    and n - a['n'] < 3 and sig == a['sig']):
                return 'stairs'
        return None

    def hidden(self, state, options, n):
        """Return the set of option labels to hide at decision n (the decision about to be made)."""
        sig = material(state)
        isig = item_sig(state)
        unreach = unreachable_ids(options)
        found = {}
        for o in options:
            rule = self._rule(o, state, sig, isig, n, unreach)
            if rule:
                found[o['label']] = rule
        if found and len(found) == len(options):
            self.all_hidden_kept += 1  # never leave the brain without a legal option
            found = {}
        self.last = found
        if found:
            self.decisions_with_hidden += 1
            self.counts.update(found.values())
        return set(found)

    def observe(self, before, opt, goal, result, after, n):
        """Update the guard after decision n was executed."""
        # Rule 3b bookkeeping (before the scene-change return: the item set, not the scene, bounds it).
        ib, ia = item_sig(before), item_sig(after)
        if ib == ia and self.worn_isig == ia:
            self.worn_seen |= {frozenset(worn(before)), frozenset(worn(after))}
        elif ib == ia:
            self.worn_isig, self.worn_seen = ia, {frozenset(worn(before)), frozenset(worn(after))}
        else:
            self.worn_isig, self.worn_seen = ia, {frozenset(worn(after))}
        if tuple(after['scene']) != tuple(before['scene']):
            self.talk_block.clear()
            self.open_npc = None
            self.swap = None
            self.hold_block = None
            self.arrival = dict(scene=tuple(after['scene']), came_from=tuple(before['scene']), n=n + 1,
                                sig=material(after), back=way_back(before['scene'], after['scene']), changed=False)
            return
        msig = material(after)
        isig = item_sig(after)
        cmds = goal.get('commands', []) if goal.get('mode') == 'service' else []
        kinds = [c.get('kind') for c in cmds]
        talked = opt['kind'] == 'npc' and result == 'completed' and 'talk' in kinds
        # Rule 1. The window a completed talk left open is closed now (the brain's 'dismiss', the runner's
        # auto-close before an action elsewhere, or a new talk that replaced it): block that npc.
        if self.open_npc is not None and (talked or not ui_open(after)):
            self.talk_block[self.open_npc] = msig
            self.open_npc = None
        if talked:
            nid = next(c['args']['id'] for c in cmds if c['kind'] == 'talk')
            if ui_open(after) and 'dismiss' not in kinds:
                self.open_npc = nid
            else:
                self.talk_block[nid] = msig  # closed at once (townsfolk: talk + dismiss) or nothing opened
        # Rule 3. Each completed equip replaces the record of the last swap (only the last swap can be undone).
        if opt['kind'] == 'equip' and result == 'completed':
            self.swap = dict(before=frozenset(worn(before)), after=frozenset(worn(after)), isig=isig)
        # Rule 4 (added 2026-09-23 03:40). A hold that ran its full time and changed nothing (no HP change, no kill,
        # same items, same gold; nobody came adjacent) is not offered again while the state and the visible
        # enemies stay the same. Seen in king1-facts/s4150019: 34 holds of 400 ticks in a row against enemies
        # that never came.
        if is_hold(opt):
            self.hold_block = (msig, enemy_ids(after)) if msig == material(before) else None
        elif self.hold_block is not None and self.hold_block[0] != msig:
            self.hold_block = None
        # Blocks lapse for good at the first change.
        self.talk_block = {k: v for k, v in self.talk_block.items() if v == msig}
        if self.swap and (self.swap['isig'] != isig or worn(after) != self.swap['after']):
            self.swap = None
        if self.arrival and msig != self.arrival['sig']:
            self.arrival['changed'] = True

    def summary(self):
        return dict(hidden_by_rule={r: self.counts.get(r, 0) for r in RULES}, decisions_with_hidden=self.decisions_with_hidden,
                    all_hidden_kept=self.all_hidden_kept)

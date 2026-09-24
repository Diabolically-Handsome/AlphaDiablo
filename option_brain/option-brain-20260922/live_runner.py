"""Play one normal-start game with an option-choosing local strategy brain.

WSL side. The frozen executor (goal head + frozen model158 for melee + exact
service commands) is reused unchanged via the strategy-brain-sft Controller. The
brain only returns an option letter; code compiles it to the executor goal. No
automatic drinking, retreat, shopping or stat allocation; no resource injection.
"""
import argparse
import collections
import copy
import hashlib
import json
import os
import signal
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(EXP / 'strategy-brain-sft-20260922'))
from options import (enumerate_options, compile_goal, hp, LABELS, ui_open, _item_cmd, item_name, MAX_MAIN_DEPTH,  # noqa: E402
                     CLOSE_STEPS, BUTCHER_LEVEL)
from render import render, SYSTEMS  # noqa: E402

INTERRUPTED = {'new_enemy', 'damage_received', 'bounded_yield', 'earned_stat_review', 'emergency_review', 'no_motion'}
CLEARS = {'completed', 'targets_no_longer_visible', 'king_dead', 'arrived', 'scene_changed'}
# --r3-chase (game_executor_r3.py diff D, chase_last_seen) removes the visibility-boundary flicker loop. No survival
# benefit was shown when games are scored by deaths per tick of exposure (a loop stop is not a survival; review and
# fixer batch 2026-09-23, rule brain, training seeds: 40 pairs, 0.400 vs 0.347 deaths per 10k ticks, rate ratio 1.15,
# 95% CI 0.70-1.94; flicker loops 0 vs 10).
# 2026-09-23 owner decision (ROADMAP-20260923.md section 5, item 5): --r3-chase and --auto-belt are part of the fixed
# hands. Round-9 games and the final exam keep them, as round 8 did (also on its 30 held-out games), so that old and
# new brains are paired on the same hands; a choice of hands, not a claim of benefit. The test seeds of
# bench/PREREG.md stay refused; held-out seeds (>= CHASE_REFUSED_FROM) need the explicit --allow-heldout.
CHASE_REFUSED_SEEDS = frozenset({4150002, 4150005, 4150006, 4150009, 4150027, 4150031})
CHASE_REFUSED_FROM = 4150032  # every seed >= this one is held out
# 2026-09-23 round 9 (ROADMAP-20260923.md goal 5): an operator stop gets its own status. SIGTERM, SIGINT or SIGHUP asks
# the game to stop: the decision in progress is finished (its goal runs to its normal end and is logged), then the game
# is closed with status 'operator_stop' instead of 'engineering_failure' (as the 08:39 stops of king3-v3/butcher1-v3
# were recorded). While the runner waits for the brain it waits at most STOP_GRACE more seconds for the reply (nothing
# has been done yet, so the decision is dropped without an action). A second signal drops at once what is in progress
# (DROPPABLE: waiting for the brain, the executor running the goal, or an --auto-belt step); result.json 'stop' then
# says what was dropped and what had already changed (ticks, gold, HP, scene, position, items; from the last state
# the runner observed). Python runs a signal handler between two bridge calls, so a drop inside the executor can
# leave the native journal one command behind the engine (skeleton-king-dual-brain-20260921-r16/session.py:138-143:
# b.step, then its journal row); a replay then refuses the recording (session.py:207-209), as it should. A second
# signal while the menu is built, a row is logged or the game is closed only notes itself. After a drop, 'decisions'
# and 'choice_kinds' in result.json count the logged rows only (the dropped one is in 'stop'). A signal the parent set to be ignored stays ignored, as before round 9 (nohup: SIGHUP;
# background jobs of a non-interactive shell, e.g. the '( ... ) &' games of run_mixed_v3.sh: SIGINT, so a Ctrl-C at
# the launching terminal does not reach them). The operator may write why to <out>/stop-reason.txt before signalling.
# How to stop a batch (run_live_exec.sh, run_rollin_exec.sh and run_mixed_v3*.sh start the next seed when a game ends
# and stop the broker only after the last one): 1. touch $AD_ROOT/STOP (no new game starts;
# the script then touches bus/STOP after the running games, which stops the broker and frees the GPU); 2. kill -TERM
# each runner (pgrep -f '^[^ ]*python -B live_runner.py'; a pattern without the '^' also matches any shell whose
# command line contains it); 3. remove the STOP file afterwards (run_sft_chain.sh and run_dagger_chain.sh obey it too).
STOP_GRACE = 30.0
STOP_SIGNALS = tuple(getattr(signal, s) for s in ('SIGTERM', 'SIGINT', 'SIGHUP') if hasattr(signal, s))  # no SIGHUP on Windows
DROPPABLE = ('belt', 'ask', 'execute')
# 2026-09-23 round 9: code versions (sha256 of the files on disk at start). The brain process may have been started
# earlier than the game from another version of its file: bus/broker-ready.json, which the broker writes when it is
# ready (model, adapter, pid, time), is copied as well. With '--menu r9' both go into started.json ('code', 'broker');
# with '--menu r8' into code.json, so that started.json keeps the round-8 keys (bench/hooks_compare.py:88-90).
CODE_FILES = ('options.py', 'render.py', 'live_runner.py', 'facts.py', 'undo_guard.py', 'game_executor_r3.py', 'hf_broker.py',
              'option_sft.py', 'rollin_broker.py')
OPTIONAL_CODE_FILES = ('engine_numbers.py',)  # facts v4 (PLAN-annotations-v4.md section 5), recorded once it exists
FACTS_VERSIONS = ('1', '2', '3', '4', '5')  # the versions facts.py implements (4: PLAN-annotations-v4.md; 5: round 9b,
# r9-work/r9b/FACTS-V5-SPEC.md, the Butcher at the stairs); started.json records it as facts.version
# 2026-09-23 round 9: '--menu r9' hides the stairs down of main level MAX_MAIN_DEPTH because this bridge build refuses
# every deeper main level (options.py MAX_MAIN_DEPTH comment: manual_control.hpp:14-24). The limit is a property of
# the build, so the game's bridge (native/started.json bridge.bridge_sha256) must be one listed here, or the game is
# refused before its first decision (a rebuilt bridge must be checked and added with its own limit).
BRIDGE_MAX_MAIN_DEPTH = {'05fc300940ac491d9e05fdd37dacecf7e9e139811de102d0c886caee8f0952b1': MAX_MAIN_DEPTH}


class OperatorStop(BaseException):
    """Drops the current decision after an operator stop. A BaseException, so that no 'except Exception' in the hands
    swallows it."""


def code_shas():
    shas = {}
    for f in CODE_FILES + OPTIONAL_CODE_FILES:
        p = HERE / f
        if p.exists():
            shas[f] = hashlib.sha256(p.read_bytes()).hexdigest()
        elif f in CODE_FILES:
            shas[f] = None
    return shas


def broker_info(bus):
    p = bus / 'broker-ready.json'
    try:
        return json.loads(p.read_text(encoding='utf-8')) if p.exists() else None
    except (OSError, ValueError) as exc:
        return dict(unreadable=repr(exc))


def state_changes(before, now):
    """What changed between two observed states (operator stop: what a dropped goal or belt step had already done)."""
    ch = dict(ticks=now['tick'] - before['tick'], gold_change=now['hero']['gold'] - before['hero']['gold'],
              hp_change=round((now['hero']['hp_fixed'] - before['hero']['hp_fixed']) / 64, 2),
              scene_before=list(before['scene']), scene_after=list(now['scene']),
              moved=[before['hero']['x'], before['hero']['y']] != [now['hero']['x'], now['hero']['y']],
              items_changed=item_sig(before) != item_sig(now))
    ch['partially_executed'] = bool(ch['ticks'] or ch['gold_change'] or ch['hp_change'] or ch['moved'] or ch['items_changed']
                                    or ch['scene_before'] != ch['scene_after'])
    return ch


def canonical(v):
    return json.dumps(v, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False)


def digest(v):
    return hashlib.sha256(canonical(v).encode()).hexdigest()


def summary(s, butcher):
    h = s['hero']
    return dict(tick=s['tick'], scene=list(s['scene']), level=h['level'], xp=h.get('xp'), hp=round(h['hp_fixed'] / 64, 2),
                max_hp=round(h['max_hp_fixed'] / 64, 2), gold=h['gold'], armor=h['armor'], dead=h['dead'], kills=s.get('kills', 0),
                king_kills=s.get('king_kills', 0), king_quest_done=s.get('king_quest_done', False), butcher=butcher,
                belt_heals=sum(1 for x in s['belt'] if not x['empty'] and x.get('heal_kind', 0) > 0))


def item_sig(state):
    return tuple(sorted(tuple(x.get('identity', ())) for g in ('equipment', 'inventory', 'belt') for x in state[g] if not x['empty']))


# 2026-09-24 delivery-failed menu rule (--menu r9; r9-work/delivery-fail). The executor's walk_to returns
# 'delivery_failed_twice' when it sent the same next step three times and the hero stayed idle on its tile
# (strategy-brain-sft-20260922/game_executor_r2.py:99-102). In town every such step logged by 2026-09-24 was onto
# a cow's extra tile: DevilutionX InitCows marks the cow's tile and the tiles NW, NE and N of it as occupied, but the
# public state lists only the cow's own tile, so KnownMap.allowed / options.allowed keep the other three walkable and
# the same route through them is planned again from the same tile. gate9-new-r2-butcher/s4150029 chose 'Explore toward
# the south-west' 37 times in a row, 3 ticks each, until no_progress_cycle_40 (logs to 2026-09-24 05:35: in town 269 of
# 271 immediate retries of the failed option failed again). So when a goal ends so with no living enemy in view, the same
# option (same key and same goal) is not offered again while stuck_sig is unchanged. With enemies in view (all 714
# dungeon cases) the refused tile is most often a walking monster's and a retry often goes on (of 482 immediate retries
# 184 moved, 115 failed again): no block.
DELIVERY_FAILED = 'delivery_failed_twice'


def stuck_sig(state, known):
    """Scene, hero tile, gold, HP, items, visible enemies and townsfolk (id, position; enemies also HP), known tiles."""
    h = state['hero']
    return (tuple(state['scene']), h['x'], h['y'], h['gold'], h['hp_fixed'], item_sig(state),
            tuple(sorted((e['id'], e['x'], e['y'], e.get('hp')) for e in state['enemies'])),
            tuple(sorted((n['id'], n['x'], n['y']) for n in state['npcs'])),
            len(known.get(','.join(map(str, state['scene'])), {}).get('tiles', {})))


def owned_ids(state):
    """Identities of the items the hero has (worn, pack, belt), with multiplicity; gold piles excluded."""
    return collections.Counter(tuple(x.get('identity', ())) for g in ('equipment', 'inventory', 'belt') for x in state[g]
                               if not x['empty'] and x.get('name') != 'Gold')


def bought_items(before, goal, after):
    """Identities that a buy decision added (round 9, shop rule). The shop hands over a bought item under a fresh
    identity (strategy-brain-sft-20260922/purchase_audit.py:1), so the delivered items are found as the identities
    that are new after the decision, not from the stock quote."""
    if goal.get('mode') != 'service' or not any(c.get('kind') == 'buy' for c in goal.get('commands', [])):
        return set()
    return set(owned_ids(after) - owned_ids(before))


def ask(bus, rid, user, labels, timeout, kinds=None, hpfrac=None, enemies=0, system=None, stop=None):
    req = dict(id=rid, system=system, user=user, labels=labels, kinds=kinds, hpfrac=hpfrac, enemies=enemies)
    tmp = bus / 'requests' / f'{rid}.tmp'
    tmp.write_text(json.dumps(req, ensure_ascii=False), encoding='utf-8')
    tmp.replace(bus / 'requests' / f'{rid}.json')
    resp = bus / 'responses' / f'{rid}.json'
    t0 = time.time()
    while time.time() - t0 < timeout:
        if resp.exists():
            try:
                return json.loads(resp.read_text(encoding='utf-8'))
            except json.JSONDecodeError:
                pass
        if stop and time.time() - stop['time'] > STOP_GRACE:
            # The request stays on the bus (also when a second signal lands here): every broker unlinks the request
            # after answering it, without missing_ok (hf_broker.py:36, rollin_broker.py:224, rule_broker.py:26,
            # broker.py:73), so removing it while the shared broker computes it would crash the broker of all games.
            raise OperatorStop(f"no brain reply within {STOP_GRACE:g} s after {stop['signal']}")
        time.sleep(0.02)
    raise TimeoutError('broker did not answer')


def auto_belt(c, s, state, out, rid, memo=None):
    """--auto-belt: one tidy-up step before decision rid. Returns False to stop belt keeping for this game.
    memo (a dict kept by the caller): after a step that moved nothing, the item set is remembered and no step is
    tried again until it changes (e.g. the pack is full)."""
    if ui_open(state) or state['hero']['dead'] or any(e.get('hp', 1) > 0 for e in state['enemies']):
        return True
    if memo is not None and memo.get('stuck') == item_sig(state):
        return True
    pack_heal = [x for x in state['inventory'] if not x['empty'] and x.get('heal_kind', 0) > 0]
    if not pack_heal:
        return True
    free = sum(1 for x in state['belt'] if x['empty'])
    junk = [x for x in state['belt'] if not x['empty'] and x.get('heal_kind', 0) <= 0]
    cmds = []
    for it in junk[:max(0, len(pack_heal) - free)]:
        cmds.append(('unbelt', _item_cmd('unbelt_exact', 'belt', it, state), item_name(it)))
    moves = []
    for k, (what, cmd, name) in enumerate(cmds):
        try:
            res = c.e.run(dict(mode='service', commands=[cmd], allow_visible_enemies=False, allow_unspent=True,
                               id=f'{rid}-unbelt{k}', state_id=digest(s.state), reason='auto belt: make room for healing potions'))
        except Exception as exc:  # noqa: BLE001 - logged, belt keeping stops, the game goes on
            (out / 'auto-belt.jsonl').open('a').write(json.dumps(dict(rid=rid, moves=moves, error=f'{what} {name}: {exc}')) + chr(10))
            return False
        moves.append([what, name, res])
        if res != 'completed':
            break
    now = s.state
    free = [x for x in now['belt'] if x['empty']]
    pack_heal = [x for x in now['inventory'] if not x['empty'] and x.get('heal_kind', 0) > 0]
    for k, it in enumerate(pack_heal[:len(free)]):
        if ui_open(s.state) or any(e.get('hp', 1) > 0 for e in s.state['enemies']):
            break
        cur = next((x for x in s.state['inventory'] if not x['empty'] and x.get('identity') == it.get('identity')), None)
        if cur is None:
            break
        try:
            res = c.e.run(dict(mode='service', commands=[_item_cmd('belt', 'inventory', cur, s.state)], allow_visible_enemies=False,
                               allow_unspent=True, id=f'{rid}-belt{k}', state_id=digest(s.state), reason='auto belt: healing potion to belt'))
        except Exception as exc:  # noqa: BLE001
            (out / 'auto-belt.jsonl').open('a').write(json.dumps(dict(rid=rid, moves=moves, error=f'belt {item_name(cur)}: {exc}')) + chr(10))
            return False
        moves.append(['belt', item_name(cur), res])
        if res != 'completed':
            break
    if moves:
        (out / 'auto-belt.jsonl').open('a').write(json.dumps(dict(rid=rid, tick=s.state['tick'], moves=moves)) + chr(10))
    if memo is not None and cmds + pack_heal[:len(free)] and not any(m[2] == 'completed' for m in moves):
        memo['stuck'] = item_sig(s.state)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--mission', choices=('skeleton_king', 'butcher'), default='skeleton_king')
    ap.add_argument('--ticks', type=int, default=30000)
    ap.add_argument('--decisions', type=int, default=1500)
    ap.add_argument('--minutes', type=float, default=60)
    ap.add_argument('--bus', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--tag', default='')
    ap.add_argument('--variant', default='v2', choices=('v1', 'v2'))
    ap.add_argument('--allow-stop', action='store_true', help='offer the voluntary stop option (off by default: budgets end games)')
    # Opt-in hands selection (game_executor_r3.py). Without --executor nothing below changes: r2 is used,
    # game_executor_r3 is not imported and started.json/result.json are written exactly as before.
    ap.add_argument('--executor', choices=('r2', 'r3'), default=None, help='hands: r2 (default, unchanged) or r3')
    ap.add_argument('--r3-chain', action='store_true', help='r3: chain swings (queue Shift-attack during swing/block/got-hit)')
    ap.add_argument('--r3-target', action='store_true', help='r3: strike the lowest-HP adjacent listed target first')
    ap.add_argument('--r3-hold-floor', type=float, default=None, metavar='FLOAT',
                    help='r3: holds return damage_received only at HP <= FLOAT*max or a one-step loss >= 20%% of max')
    ap.add_argument('--r3-chain-guard', choices=('min_damage', 'post_hit'), default=None,
                    help='r3 chaining guard (needs --executor r3 --r3-chain): min_damage (default, appendix spec) or '
                         'post_hit (proposed spec change); same ExecutorR3 code path as bench/run_moment.py --chain-guard')
    ap.add_argument('--r3-chase', action='store_true',
                    help='r3: chase_last_seen (game_executor_r3.py diff D): when the fight targets leave sight, walk '
                         '(bounded) toward the last public position of the chosen one instead of returning at once. Part '
                         'of the round-8/9 hands and the final exam (owner decision 2026-09-23, see the comment); no '
                         'survival benefit shown. Refused on the test seeds of bench/PREREG.md; held-out seeds need '
                         '--allow-heldout')
    # Opt-in menu guard (undo_guard.py). Without --undo-guard it is not imported and nothing changes.
    ap.add_argument('--undo-guard', action='store_true',
                    help='hide options that would only undo the previous action (undo_guard.py); logged per decision as undo_hidden')
    # Opt-in digested fact lines (facts.py FactsTracker), identical to rerender.py's training prompts.
    ap.add_argument('--facts', action='store_true', help='insert digested fact lines before Options (facts.py); off by default')
    # Opt-in belt keeping (2026-09-23 06:40). In king1/king2 the belt on level 3 and in the tomb held 5-7 mana
    # potions and scrolls and only 1-3 healing potions while up to 14 healing potions sat in the pack (only belt
    # potions can be drunk in a fight); the brain never tidied it. Like the auto-close of a shop window, this is
    # plain inventory handling by the hands: with no enemy visible and no window open, move non-healing belt items
    # to the pack when healing potions in the pack need the room, then move healing potions from the pack into free
    # belt slots. No resources are created, nothing is dropped or sold. Refused on test/held-out seeds.
    ap.add_argument('--auto-belt', action='store_true', help='hands keep healing potions in the belt (see comment); off by default')
    # 2026-09-23: the final held-out check plays seeds >= 4150032 with the configuration validated on training seeds.
    # The exam/test seeds of bench/PREREG.md (4150002, 4150005, 4150006, 4150009, 4150027, 4150031) stay refused.
    ap.add_argument('--allow-heldout', action='store_true',
                    help='allow --r3-chase/--auto-belt on held-out seeds >= 4150032 (final held-out check); test seeds stay refused')
    # 2026-09-23 round 9 menu (options.enumerate_options docstring): no 'Sell' of an item bought during the current
    # town visit, no stairs down that the bridge refuses (level 3, BRIDGE_MAX_MAIN_DEPTH), no attribute point while an
    # enemy is adjacent, on the Butcher mission no stairs down from his level 2 (BUTCHER_LEVEL, 2026-09-24). Each row
    # then logs what the rules hid (menu_hidden) and what was bought (menu_bought).
    # On by default for every new game; '--menu r8' gives the round-8 menu and round-8 rows and started.json keys (the
    # code versions go to code.json instead; the operator-stop handling and the FACTS_VERSION check stay on).
    ap.add_argument('--menu', choices=('r9', 'r8'), default='r9',
                    help='option menu rules: r9 (default; round-9 shop/stairs/stat rules, menu_hidden/menu_bought per row) '
                         'or r8 (round-8 menu, rows and started.json keys)')
    a = ap.parse_args()
    facts_env = os.environ.get('FACTS_VERSION')
    if a.facts and (facts_env is None or not facts_env.strip()):
        # 2026-09-23 round 9: facts.py reads FACTS_VERSION and silently uses version 2 when it is missing; a brain
        # trained on version 3 (d9facts3) or 4 then sees lines it was never trained on. Refuse instead.
        ap.error('--facts needs the FACTS_VERSION environment variable set explicitly to the facts version the brain was '
                 'trained on (e.g. export FACTS_VERSION=3 for d9facts3); facts.py would otherwise fall back to version 2')
    if a.facts and facts_env.strip() not in FACTS_VERSIONS:
        # facts.py tests 'FACTS_VERSION >= 4' / '>= 5', so e.g. 6 would run version 5 while started.json said 6.
        ap.error(f'FACTS_VERSION must be one of {", ".join(FACTS_VERSIONS)} (the versions facts.py implements), got {facts_env!r}')
    if a.facts and int(facts_env) >= 4 and not (a.executor == 'r3' and a.r3_chain and a.r3_chain_guard == 'post_hit'):
        # PLAN-annotations-v4.md section 5 (live_runner.py): the v4 swing and kill-time numbers assume the r3 hands
        # with chained swings and the post_hit guard (formula sheet F7, gate G1).
        ap.error('--facts with FACTS_VERSION >= 4 needs --executor r3 --r3-chain --r3-chain-guard post_hit')
    if a.executor != 'r3' and (a.r3_chain or a.r3_target or a.r3_hold_floor is not None):
        ap.error('--r3-chain/--r3-target/--r3-hold-floor need --executor r3')
    if a.r3_chain_guard is not None and not (a.executor == 'r3' and a.r3_chain):
        ap.error('--r3-chain-guard needs --executor r3 --r3-chain')
    if a.r3_chase and a.executor != 'r3':
        ap.error('--r3-chase needs --executor r3')
    if a.r3_chase and (a.seed in CHASE_REFUSED_SEEDS or (a.seed >= CHASE_REFUSED_FROM and not a.allow_heldout)):
        ap.error(f'--r3-chase is refused on test/held-out seed {a.seed} (bench/PREREG.md test seeds: 4150002, 4150005, '
                 '4150006, 4150009, 4150027, 4150031; held-out seeds >= 4150032 need --allow-heldout)')
    if a.auto_belt and (a.seed in CHASE_REFUSED_SEEDS or (a.seed >= CHASE_REFUSED_FROM and not a.allow_heldout)):
        ap.error(f'--auto-belt is refused on test/held-out seed {a.seed} (not validated there)')
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    bus = Path(a.bus)
    deadline = time.time() + a.minutes * 60
    from game_support_r3 import Controller  # frozen executor chain
    executor = None
    if a.executor is not None:
        # Must run before Controller() is constructed: it sets the Executor class Controller builds.
        from game_executor_r3 import install
        executor = install(a.executor, chain=a.r3_chain, target_rule=a.r3_target, hold_floor=a.r3_hold_floor,
                           chain_guard=a.r3_chain_guard or 'min_damage',
                           **(dict(chase_last_seen=True) if a.r3_chase else {}))  # passed only when on
    started = dict(seed=a.seed, mission=a.mission, tick_limit=a.ticks, decision_limit=a.decisions, minutes=a.minutes,
                   tag=a.tag, variant=a.variant, allow_stop=a.allow_stop, time=time.time(), normal_start=True, no_resource_injection=True, no_assistant_rescue=True)
    if executor is not None:
        started['executor'] = executor
    guard = None
    if a.undo_guard:
        import undo_guard
        guard = undo_guard.UndoGuard()
        started['undo_guard'] = dict(on=True, sha256=hashlib.sha256(Path(undo_guard.__file__).read_bytes()).hexdigest())
    tracker = None
    if a.facts:
        import facts as facts_mod
        if facts_mod.FACTS_VERSION != int(facts_env):
            raise RuntimeError(f'facts.py runs version {facts_mod.FACTS_VERSION}, FACTS_VERSION says {facts_env}')
        tracker = facts_mod.FactsTracker()
        started['facts'] = dict(on=True, sha256=hashlib.sha256(Path(facts_mod.__file__).read_bytes()).hexdigest(),
                                version=facts_mod.FACTS_VERSION)
        if facts_mod.FACTS_VERSION >= 4:
            # PLAN-annotations-v4.md section 5: the engine-number table and the hands the v4 numbers assume.
            eng = HERE / 'engine_numbers.py'
            started['facts'].update(engine_numbers_sha256=hashlib.sha256(eng.read_bytes()).hexdigest() if eng.exists() else None,
                                    hands=dict(executor=a.executor, chain=a.r3_chain, chain_guard=a.r3_chain_guard, target=a.r3_target,
                                               hold_floor=a.r3_hold_floor, chase=a.r3_chase, auto_belt=a.auto_belt))
    if a.auto_belt:
        started['auto_belt'] = True
    if a.allow_heldout:
        started['allow_heldout'] = True
    code = dict(code=code_shas(), broker=broker_info(bus))
    if a.menu == 'r9':
        started['menu'] = a.menu
        started['menu_rules'] = dict(no_sell_bought_this_visit=True, stat_close_steps=CLOSE_STEPS,
                                     max_main_depth_by_bridge=dict(BRIDGE_MAX_MAIN_DEPTH))
        started.update(code)
    else:
        (out / 'code.json').write_text(json.dumps(dict(menu=a.menu, **code)), encoding='utf-8')
    (out / 'started.json').write_text(json.dumps(started), encoding='utf-8')
    log = (out / 'decisions.jsonl').open('x', encoding='utf-8')
    c = None
    status = 'engineering_failure'
    memory = {}
    operated = set()
    equip_blocked = {}  # option text -> item signature at the time the equip had no effect
    move_blocked = {}   # option key -> (scene, x, y) where a move with this key did nothing (0 ticks, no step)
    delivery_blocked = {}  # --menu r9: option key -> (stuck_sig, option goal) after that goal ended DELIVERY_FAILED
    belt_ok = True      # --auto-belt: switched off for the rest of the game after any refusal
    belt_memo = {}      # --auto-belt: item set at which the last step moved nothing
    # max_main_depth: from the bridge, once it is known; butcher_level: the Butcher mission's stairs-down rule
    # (options.py BUTCHER_LEVEL, 2026-09-24; enumerate_options applies it only when mission is 'butcher')
    menu_kw = dict(calm_stats=True, butcher_level=BUTCHER_LEVEL) if a.menu == 'r9' else {}
    town_bought = set()  # --menu r9: identities bought during the current town visit (cleared once the hero is out of town)
    recent = collections.deque(maxlen=4)
    counts = collections.Counter()
    stalls = 0
    n = 0
    loop_sigs = collections.deque(maxlen=40)
    initial = final = None
    stop = {}  # operator stop request (STOP_GRACE comment): signal, time
    phase = ['setup']  # setup, decide, belt, ask, execute, record, closing; a second signal drops only DROPPABLE work
    work = {}  # the work in progress: what ('decision' / 'auto_belt'), n (the decision), before (state when it started)

    def on_signal(signum, frame):
        name = signal.Signals(signum).name
        if not stop:
            stop.update(signal=name, time=time.time())
            return
        stop.setdefault('second_signal', name)
        if phase[0] in DROPPABLE:
            raise OperatorStop(f'second signal {name}')

    handled = [sig for sig in STOP_SIGNALS if signal.getsignal(sig) != signal.SIG_IGN]  # inherited ignores stay
    for sig in handled:
        signal.signal(sig, on_signal)
    try:
        c = Controller(out / 'native', a.seed, a.ticks, deadline)
        if a.menu == 'r9':
            bridge = json.loads((out / 'native' / 'started.json').read_text(encoding='utf-8')).get('bridge', {}).get('bridge_sha256')
            if bridge not in BRIDGE_MAX_MAIN_DEPTH:
                raise RuntimeError(f'--menu r9: bridge {bridge} is not in BRIDGE_MAX_MAIN_DEPTH; check the main levels its '
                                   'ManualTransitionGuard allows (manual_control.hpp) and add it, or run with --menu r8')
            menu_kw['max_main_depth'] = BRIDGE_MAX_MAIN_DEPTH[bridge]
        s = c.s
        state = s.state
        butcher = s.b.butcher_events()
        initial = summary(state, butcher)
        if tuple(state['scene']) != (0, 0) or state['hero']['level'] != 1:
            raise RuntimeError('not a normal town start')
        milestones = {}
        while True:
            phase[0] = 'decide'
            state = s.state
            butcher = s.b.butcher_events()
            final = summary(state, butcher)
            if state['hero']['dead']:
                status = 'dead'
                break
            if a.mission == 'skeleton_king' and state.get('king_kills', 0) > 0 and state.get('king_quest_done'):
                status = 'success'
                break
            if a.mission == 'butcher' and butcher.get('kills', 0) > 0:
                status = 'success'
                break
            if n >= a.decisions:
                status = 'decision_budget'
                break
            if time.time() >= deadline:
                status = 'wall_budget'
                break
            if stop:
                status = 'operator_stop'
                break
            if a.auto_belt and belt_ok:
                work.update(what='auto_belt', n=n + 1, before=state, choice=None)
                phase[0] = 'belt'
                belt_ok = auto_belt(c, s, state, out, f'{a.tag or "g"}-{a.seed}-{n + 1:05d}', belt_memo)
                phase[0] = 'decide'
                state = s.state
                if stop:
                    status = 'operator_stop'
                    break
            scene = tuple(state['scene'])
            key = f'd{scene[0]}'
            milestones.setdefault(key, state['tick'])
            known = s.map.scenes
            ops = {oid for (sc, oid) in operated if sc == scene}
            if scene != (0, 0):
                town_bought.clear()  # the visit is over once the hero is out of town
            menu_memory = dict(memory, bought=frozenset(town_bought)) if town_bought else memory
            # --menu r9: every option the code leaves out for the brain is logged (menu_hidden: the round-9 rules of
            # enumerate_options, and the runner's no-effect / 0-tick / delivery-failed repeats below; the undo guard logs
            # undo_hidden), so each level of ROADMAP-20260923.md section 4 can count the brain's own choices apart from code's.
            menu_hidden = [] if a.menu == 'r9' else None
            options, dropped = enumerate_options(state, known, menu_memory, ops, a.mission, a.variant, hidden=menu_hidden, **menu_kw)
            if not a.allow_stop:
                options = [o for o in options if o['kind'] != 'stop']  # stop is always last, letters unchanged
            isig = item_sig(state)
            bkey = lambda o: repr(o.get('key', o['text']))  # item identity, not the text (distance in text changes)
            if any(o['kind'] in ('equip', 'pickup') and equip_blocked.get(bkey(o)) == isig for o in options):
                # An equip/pickup that just had no effect (items unchanged since) is not offered again; re-letter.
                if menu_hidden is not None:
                    menu_hidden += [dict(rule='no_effect', kind=o['kind'], text=o['text']) for o in options
                                    if o['kind'] in ('equip', 'pickup') and equip_blocked.get(bkey(o)) == isig]
                options = [o for o in options if not (o['kind'] in ('equip', 'pickup') and equip_blocked.get(bkey(o)) == isig)]
                for i, o in enumerate(options):
                    o['label'] = LABELS[i]
            # 2026-09-23 10:10: a move (exit / quest / explore) that just completed in 0 ticks without a step or a scene
            # change is not offered again while the hero stands on the same tile (king3-v3/s4150019: 'Walk next to the
            # stairs UP (0 steps away)' x35; heldout-butcher-v3/s4150042: 'Take the stairs DOWN (0 steps away)' x35).
            here = (scene, state['hero']['x'], state['hero']['y'])
            if any(o['kind'] in ('exit', 'quest', 'explore') and move_blocked.get(bkey(o)) == here for o in options):
                kept = [o for o in options if not (o['kind'] in ('exit', 'quest', 'explore') and move_blocked.get(bkey(o)) == here)]
                if kept:
                    if menu_hidden is not None:
                        menu_hidden += [dict(rule='noop_move', kind=o['kind'], text=o['text']) for o in options
                                        if o['kind'] in ('exit', 'quest', 'explore') and move_blocked.get(bkey(o)) == here]
                    options = kept
                    for i, o in enumerate(options):
                        o['label'] = LABELS[i]
            # 2026-09-24 (DELIVERY_FAILED comment): an option whose goal ended 'delivery_failed_twice' with no enemy in
            # view is not offered again (same key, same goal) while stuck_sig is unchanged.
            if delivery_blocked:
                dsig = stuck_sig(state, known)
                dfail = lambda o: delivery_blocked.get(bkey(o)) == (dsig, canonical(o['goal']))  # noqa: E731
                if any(dfail(o) for o in options):
                    kept = [o for o in options if not dfail(o)]
                    if kept:
                        menu_hidden += [dict(rule='delivery_failed', kind=o['kind'], text=o['text']) for o in options if dfail(o)]
                        options = kept
                        for i, o in enumerate(options):
                            o['label'] = LABELS[i]
            undo_hidden = None
            if guard is not None:
                # Options that would only undo the previous action while nothing changed are not offered; re-letter.
                hide = guard.hidden(state, options, n + 1)
                undo_hidden = [o['text'] for o in options if o['label'] in hide]
                if hide:
                    options = [o for o in options if o['label'] not in hide]
                    for i, o in enumerate(options):
                        o['label'] = LABELS[i]
            interrupted = memory.get('interrupted_text')
            user = render(state, known, options, a.mission, interrupted, list(recent))
            if tracker is not None:
                user = tracker.render(user)
            labels = [o['label'] for o in options]
            if stop:  # a stop that came while the menu was built: nothing asked yet
                status = 'operator_stop'
                break
            n += 1
            work.update(what='decision', n=n, before=None, choice=None)
            phase[0] = 'ask'
            rid = f'{a.tag or "g"}-{a.seed}-{n:05d}'
            checkpoint = digest(s.b.manual_checkpoint())
            cur_hp, max_hp = hp(state)
            reply = ask(bus, rid, user, labels, timeout=180, kinds=[o['kind'] for o in options], hpfrac=cur_hp / max(max_hp, 1),
                        enemies=sum(1 for e in state['enemies'] if e.get('hp', 1) > 0), system=SYSTEMS[a.variant], stop=stop)
            if digest(s.b.manual_checkpoint()) != checkpoint:
                raise RuntimeError('native state advanced while thinking')
            if reply.get('choice') not in labels:
                raise RuntimeError('broker returned no legal label: ' + str(reply.get('error')))
            phase[0] = 'record'  # the answer is in; from here the decision is executed or logged, not dropped half-way
            opt = next(o for o in options if o['label'] == reply['choice'])
            counts[opt['kind']] += 1
            work['choice'] = dict(label=opt['label'], kind=opt['kind'], text=opt['text'])
            row = dict(n=n, tick=state['tick'], scene=list(scene), hp=list(hp(state)), options=[dict(label=o['label'], kind=o['kind'], text=o['text']) for o in options],
                       choice=opt['label'], kind=opt['kind'], text=opt['text'], confidence=reply.get('confidence'), probs=reply.get('probs'),
                       think_seconds=reply.get('seconds'), prompt=user)
            if guard is not None:
                row['undo_hidden'] = undo_hidden
            if menu_hidden is not None:
                row['menu_hidden'] = menu_hidden
                row['menu_bought'] = sorted(list(i) for i in town_bought)
            if opt['kind'] == 'stop':
                log.write(canonical(dict(row, result='strategist_stop')) + '\n')
                status = 'strategist_stop'
                break
            goal = compile_goal(opt, rid, digest(state), f'option {opt["label"]}: {opt["text"]}'[:290])
            skip = None
            if goal['mode'] == 'move' and s.map.path(state, goal['targets'][0]) is None:
                # Menu and executor path finders disagreed (rare; same code in principle). Do nothing this turn,
                # report it like any failed move, and keep the evidence for later diagnosis.
                tgt = tuple(goal['targets'][0])
                (out / 'path-mismatch.jsonl').open('a').write(json.dumps(dict(
                    n=n, text=opt['text'], target=list(tgt), future=[state['hero']['future_x'], state['hero']['future_y']],
                    position=[state['hero']['x'], state['hero']['y']], target_in_allowed=tgt in s.map.allowed(state),
                    scene=list(state['scene']), enemies=[[e['x'], e['y']] for e in state['enemies']])) + chr(10))
                skip = 'no_known_path'
            before = copy.deepcopy(state)
            (out / 'pending.json').write_text(json.dumps(dict(n=n, choice=opt['label'], text=opt['text'], goal=goal,
                                                              equipment=[e for e in state['equipment'] if not e['empty']],
                                                              inventory=[e for e in state['inventory'] if not e['empty']]), default=str), encoding='utf-8')
            work['before'] = before
            phase[0] = 'execute'
            auto_closed = None
            if skip is None and ui_open(state) and opt['kind'] not in ('buy', 'sell', 'repair', 'identify', 'dismiss', 'drink', 'stat'):
                # A shop/dialog is open and the brain chose something elsewhere: close the window first,
                # as a human's click elsewhere would (the bridge rejects walking while a store is open).
                auto_closed = c.e.run(dict(mode='service', commands=[dict(kind='dismiss', args={})], allow_visible_enemies=True,
                                           allow_unspent=True, id=rid + '-close', state_id=digest(state),
                                           reason='close the open window before acting elsewhere'))
            try:
                result = skip if skip is not None else c.e.run(goal)
            except RuntimeError as exc:
                phase[0] = 'record'
                # Frozen executor refuses when the game accepted an equip but the item did not land
                # (e.g. the swapped-out two-hander has no room in a full pack). Continue only if the
                # game state is provably unchanged; otherwise it stays an engineering failure.
                if 'unbelt changed equipment or resources' in str(exc):
                    # Frozen check also trips when a monster hits during the move. Continue only if the
                    # item reached the pack, equipment and gold are identical and HP did not go up.
                    now = s.state
                    ids = lambda st, g: sorted(tuple(x.get('identity', ())) for x in st[g] if not x['empty'])
                    moved = [tuple(c['args']['identity']) for c in goal['commands'] if c['kind'] == 'unbelt_exact']
                    ok = (ids(now, 'equipment') == ids(before, 'equipment') and now['hero']['gold'] == before['hero']['gold']
                          and now['hero']['hp_fixed'] <= before['hero']['hp_fixed'] and all(m in ids(now, 'inventory') for m in moved))
                    (out / 'unbelt-hit.jsonl').open('a').write(json.dumps(dict(n=n, text=opt['text'], ok=ok,
                                                                               hp_before=before['hero']['hp_fixed'], hp_after=now['hero']['hp_fixed'])) + chr(10))
                    if not ok:
                        raise
                    result = 'damage_received'
                elif 'actual destination differs' in str(exc):
                    now = s.state
                    ids = lambda st, g: sorted(tuple(x.get('identity', ())) for x in st[g] if not x['empty'])
                    same = all(ids(now, g) == ids(before, g) for g in ('equipment', 'inventory', 'belt'))
                    (out / 'equip-noop.jsonl').open('a').write(json.dumps(dict(n=n, text=opt['text'], unchanged=same,
                                                                                  keys=sorted(k for k in now if 'cursor' in k or 'hand' in k))) + chr(10))
                    if not same:
                        raise
                    equip_blocked[repr(opt.get('key', opt['text']))] = item_sig(before)
                    result = 'rejected:equip_had_no_effect'
                else:
                    raise
            except TimeoutError as exc:
                phase[0] = 'record'
                if str(exc) == 'native_tick_budget':
                    status = 'tick_budget'
                    log.write(canonical(dict(row, result='native_tick_budget')) + '\n')
                    break
                raise
            phase[0] = 'record'
            after = s.state
            if (goal['mode'] == 'service' and result == 'completed' and any(c['kind'] == 'pickup' for c in goal['commands'])
                    and item_sig(after) == item_sig(before) and after['hero']['gold'] == before['hero']['gold']):
                # Pickup "completed" but nothing reached the pack or purse (e.g. pack full): say so and stop offering it.
                result = 'rejected:pickup_had_no_effect'
                equip_blocked[repr(opt.get('key', opt['text']))] = item_sig(before)
            if (goal['mode'] == 'move' and result == 'completed' and after['tick'] == before['tick']
                    and tuple(after['scene']) == tuple(before['scene'])
                    and (after['hero']['x'], after['hero']['y']) == (before['hero']['x'], before['hero']['y'])):
                move_blocked[repr(opt.get('key', opt['text']))] = (tuple(before['scene']), before['hero']['x'], before['hero']['y'])
            if a.menu == 'r9' and result == DELIVERY_FAILED and not any(e.get('hp', 1) > 0 for e in after['enemies']):
                delivery_blocked[repr(opt.get('key', opt['text']))] = (stuck_sig(after, s.map.scenes), canonical(opt['goal']))
            receipt = dict(reason=result, ticks=after['tick'] - before['tick'],
                           hp_change=round((after['hero']['hp_fixed'] - before['hero']['hp_fixed']) / 64, 2),
                           gold_change=after['hero']['gold'] - before['hero']['gold'], kills_change=after['kills'] - before['kills'],
                           scene_before=list(before['scene']), scene_after=list(after['scene']),
                           moved=[before['hero']['x'], before['hero']['y']] != [after['hero']['x'], after['hero']['y']])
            logged = dict(row, goal=goal, result=result, receipt=receipt, auto_closed_window=auto_closed,
                          after=summary(after, s.b.butcher_events()))
            log.write(canonical(logged) + '\n')
            log.flush()
            if a.menu == 'r9':
                town_bought |= bought_items(before, goal, after)
            if tracker is not None:
                tracker.observe(logged)
            if guard is not None:
                guard.observe(before, opt, goal, result, after, n)
            # Memory for the next decision (public receipts only).
            same_scene = tuple(after['scene']) == scene
            if goal['mode'] in ('move', 'fight') and result in INTERRUPTED and same_scene:
                cur = {k: v for k, v in goal.items() if k not in ('id', 'state_id', 'reason')}
                if goal['mode'] == 'fight':
                    alive = {e['id'] for e in after['enemies']}
                    cur['target_ids'] = [i for i in cur['target_ids'] if i in alive]
                if goal['mode'] == 'move' or cur['target_ids']:
                    memory = dict(current=cur, current_text=opt['text'] if opt['kind'] != 'resume' else memory.get('current_text', 'goal'),
                                  interrupted_text=f"{opt['text'] if opt['kind'] != 'resume' else memory.get('current_text', 'goal')} (stopped because: {result})")
                else:
                    memory = {}
            elif goal['mode'] in ('move', 'fight') or not same_scene:
                memory = {}
            if goal['mode'] == 'service' and result == 'completed':
                for cmd in goal['commands']:
                    if cmd['kind'] == 'operate':
                        operated.add((scene, cmd['args']['id']))
            txt = opt['text'][:70]
            recent.append(f"{txt} -> {result}, {receipt['ticks']} ticks" + (f", HP {receipt['hp_change']:+.0f}" if receipt['hp_change'] else '')
                          + (f", kills +{receipt['kills_change']}" if receipt['kills_change'] else '')
                          + (f", gold {receipt['gold_change']:+d}" if receipt['gold_change'] else ''))
            if not same_scene:
                recent.clear()
            progress = receipt['ticks'] or receipt['moved'] or receipt['hp_change'] or receipt['gold_change'] or receipt['kills_change'] or not same_scene
            stalls = 0 if progress or result == 'completed' else stalls + 1
            if stalls >= 4:
                status = 'no_progress_4'
                break
            # Same material state for 40 consecutive decisions = circling (applies equally to every arm).
            sig = (tuple(after['scene']), after['hero']['level'], after['kills'], after['hero']['gold'], after['hero']['hp_fixed'],
                   sum(1 for x in after['inventory'] if not x['empty']), sum(1 for x in after['belt'] if not x['empty']),
                   len(s.map.scenes.get(','.join(map(str, after['scene'])), {}).get('tiles', {})))
            loop_sigs.append(sig)
            if len(loop_sigs) == loop_sigs.maxlen and len(set(loop_sigs)) == 1:
                status = 'no_progress_loop_40'
                break
            # Cycling among at most 3 material states for 40 decisions (e.g. stairs up/down, swapping two
            # equal armours) is also no progress; stop the game rather than burn the budget.
            if len(loop_sigs) == loop_sigs.maxlen and len(set(loop_sigs)) <= 3:
                status = 'no_progress_cycle_40'
                break
        if status in ('engineering_failure',):
            status = 'ended'
    except OperatorStop as exc:
        dropped_in, phase[0] = phase[0], 'closing'
        status = 'operator_stop'
        stop.update(dropped=str(exc), dropped_what=work.get('what'), dropped_in=dropped_in)
        if work.get('before') is not None and c is not None:
            # What the dropped goal / belt step had done by then (last observed state; STOP_GRACE comment).
            try:
                stop['changes'] = state_changes(work['before'], c.s.state)
            except Exception as err:  # noqa: BLE001 - reported; closing goes on
                stop['changes_unknown'] = repr(err)
        if work.get('what') == 'decision':
            n -= 1  # 'decisions' counts logged rows; the dropped one is stop.decision_dropped
            if work.get('choice'):
                stop['dropped_choice'] = work['choice']
                counts[work['choice']['kind']] -= 1
                counts = +counts
    except Exception as exc:
        phase[0] = 'closing'
        if isinstance(exc, TimeoutError) and str(exc) in ('native_tick_budget', 'wall_clock_budget'):
            status = 'tick_budget' if 'tick' in str(exc) else 'wall_budget'
        else:
            status = 'engineering_failure'
            (out / 'failure.json').write_text(json.dumps(dict(error=repr(exc), traceback=traceback.format_exc())), encoding='utf-8')
    finally:
        # Further signals only note themselves: closing the game and writing result.json must not be cut short.
        phase[0] = 'closing'

        def on_signal_closing(signum, frame):
            stop.setdefault('signal', signal.Signals(signum).name)
            stop.setdefault('time', time.time())
            stop['signal_while_closing'] = True

        for sig in handled:
            signal.signal(sig, on_signal_closing)
        log.close()
        if c is not None:
            try:
                st = c.s.state
                final = summary(st, c.s.b.butcher_events())
                c.close(status)
            except Exception as exc:
                (out / 'close-failure.json').write_text(json.dumps(dict(error=repr(exc), traceback=traceback.format_exc())), encoding='utf-8')
        res = dict(seed=a.seed, mission=a.mission, tag=a.tag, status=status, decisions=n, choice_kinds=dict(counts),
                   initial=initial, final=final, milestones=locals().get('milestones'), seconds=time.time() - started['time'])
        if executor is not None:
            res['executor'] = executor
        if guard is not None:
            res['undo_guard'] = guard.summary()
        if stop:
            note = out / 'stop-reason.txt'
            given = note.read_text(encoding='utf-8', errors='replace').strip()[:500] if note.exists() else ''
            what = stop.get('dropped_what') if 'dropped' in stop else None
            lost = work['n'] if what == 'decision' else None  # asked, possibly begun, not logged
            ch = stop.get('changes')
            done = ('' if ch is None else
                    f", partially executed (ticks {ch['ticks']:+d}, gold {ch['gold_change']:+d}, HP {ch['hp_change']:+g}"
                    f"{', scene changed' if ch['scene_before'] != ch['scene_after'] else ''}{', moved' if ch['moved'] else ''}"
                    f"{', items changed' if ch['items_changed'] else ''})" if ch['partially_executed'] else ', nothing had changed yet')
            if what == 'decision':
                how = 'before any action' if stop['dropped_in'] == 'ask' else 'while executing' + done
                tail = f"; decision {lost} dropped {how}: {stop['dropped']}"
            elif what == 'auto_belt':
                tail = f"; the auto-belt step before decision {work['n']} dropped{done}: {stop['dropped']}"
            else:
                tail = f"; dropped ({stop['dropped']})" if 'dropped' in stop else '; no decision dropped'
            res['stop'] = dict(stop, time=round(stop['time'], 3), decision_dropped=lost,
                               reason=f"operator stop ({stop['signal']})" + (f': {given}' if given else '') + tail)
        (out / 'result.json').write_text(json.dumps(res), encoding='utf-8')
        print(json.dumps(res))


if __name__ == '__main__':
    main()

"""Shared read-only helpers of the round-9 exam tools (check_exam9_wave.py, score_exam9.py, fidelity9.py,
seed_audit9.py, run_exam9_chain.sh). Written 2026-09-23 for PREREG-ROUND9.md; part of the exam freeze list. Nothing
here writes a file.

Definitions (PREREG-ROUND9.md 4.1, 4.5 and the appendix):
  outcome    Y = 1 iff result.json status == 'success' (live_runner ends the game at the mission boss kill: King
             mission king_kills > 0 and king_quest_done; Butcher mission a Butcher kill). Death: status 'dead'.
  classes    ok  success, dead, tick_budget, decision_budget.
             A   external infrastructure, re-run once in the make-up wave: wall_budget; no result.json (WSL or host
                 crash, OOM killer); engineering_failure whose failure.json 'error' (repr of the exception) begins
                 with 'OSError(', 'IOError(' or 'MemoryError(', or contains one of A_SUBSTRINGS (the messages and
                 '[Errno N]' forms of errno 28 disk full, 12 out of memory, 5 I/O error); engineering_failure
                 'broker did not answer' (always: the arm's code can cause a broker timeout only by crashing the
                 broker deterministically, which the make-up reproduces, and a second failure counts 0; the broker
                 batch's death records are reported, not required); operator_stop without a non-empty
                 stop-reason.txt when every game of the wave still running then stopped the same way within 60 s or
                 has no result.json (refine_wave: WSL or host shut down).
             B   the arm's own code or behaviour, Y = 0, no re-run: every other engineering_failure (facts /
                 options / undo_guard / executor exceptions, OSError subclasses whose repr begins with their own
                 name such as FileNotFoundError, PermissionError or BrokenPipeError and carries none of
                 A_SUBSTRINGS, 'broker returned no legal label', 'native state advanced while thinking'), 'ended',
                 'strategist_stop', every no_progress_*.
             op_stop             operator_stop with a non-empty stop-reason.txt: a human stop (forbidden), Y = 0,
                 sensitivity analysis S2.
             op_stop_unexplained operator_stop without a non-empty stop-reason.txt that is not a whole-wave stop:
                 Y = 0 and treated like an operator stop (S2). There is no manual re-classification.
  boundary   Butcher mission: result.json milestones has a key d3 or deeper (a decision on main level 3+ or in the
             tomb), even if the Butcher is killed later. The decision rows (scene[0] >= 3) are a cross-check, and
             the definition for a game without result.json.
  retreat    denominator: dungeon decisions (scene[0] != 0) whose prompt Hero line says 'healing potions a in belt and
             b in pack' with a + b <= 1, whose prompt says 'Visible enemies: none.', and whose menu has an option
             containing 'stairs UP' or 'exit back to the main dungeon'; numerator: those that chose such an option.
  low potion an event starts at a dungeon decision whose potions are <= 1 after a decision with >= 2; it ends at the
             first of: town, potions back to >= 2 in the dungeon (refill), death, end of the game.
  buy-sell   exact (code integrity): a 'sell' whose item identity is in the row's menu_bought (bought in this town
             visit; live_runner.bought_items). By base name (descriptive): same visit / a later visit (with loss).
"""
import collections
import hashlib
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
CONFIG_PATH = HERE / 'exam9_config.json'
MISSION_KEY = {'skeleton_king': 'king', 'butcher': 'butcher'}
MISSIONS = ('skeleton_king', 'butcher')
# class A engineering failures (PREREG 4.5; live_runner writes failure.json error=repr(exc))
A_PREFIXES = ('OSError(', 'IOError(', 'MemoryError(')
A_SUBSTRINGS = ('No space left on device', 'Cannot allocate memory', 'Input/output error', '[Errno 28]', '[Errno 12]', '[Errno 5]')
CHECK_OK_EXITS = (0, 3)  # check_exam9_wave.py exits that leave a wave counted (3: class-A stop, PREREG 3.3)
VALID = ('success', 'dead', 'tick_budget', 'decision_budget')
POT_RE = re.compile(r'healing potions (\d+) in belt and (\d+) in pack')
LEVEL_RE = re.compile(r'Hero: level (\d+)')
DEX_RE = re.compile(r'dexterity (\d+)')
BOSS_TEXT = {'skeleton_king': ' SKELETON KING #', 'butcher': 'The Butcher'}


def load_config(path=None):
    return json.loads(Path(path or CONFIG_PATH).read_text(encoding='utf-8'))


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def read_json(p):
    try:
        return json.loads(Path(p).read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return None


def parse_freeze(path):
    """sha256sum output -> {path: sha}."""
    out = {}
    for line in Path(path).read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        sha, name = line.split(None, 1)
        out[name.lstrip('*')] = sha
    return out


def iter_rows(game_dir):
    """decisions.jsonl rows; a torn last line (a runner killed mid-write) is skipped."""
    p = Path(game_dir) / 'decisions.jsonl'
    if not p.exists():
        return
    with open(p, encoding='utf-8') as f:
        for line in f:
            if not line.endswith('\n'):
                break
            try:
                yield json.loads(line)
            except ValueError:
                continue


def potions(prompt):
    m = POT_RE.search(prompt or '')
    return int(m.group(1)) + int(m.group(2)) if m else None


def no_enemy(prompt):
    return any(x.startswith('Visible enemies: none') for x in (prompt or '').split('\n'))


def in_town(row):
    return (row.get('scene') or [0, 0])[0] == 0


# ------------------------------------------------------------------ batches and waves
def broker_batch_of(out_batch_dir):
    """The broker batch of an out-batch: its registration.json 'broker_batch', else the batch itself."""
    reg = read_json(Path(out_batch_dir) / 'registration.json') or {}
    return reg.get('broker_batch') or Path(out_batch_dir).name


def wave_game_dirs(live, broker_batch):
    """Every game directory served by one broker batch (both missions' out-batches)."""
    live = Path(live)
    outs = [live / broker_batch] + [p.parent for p in live.glob('*/registration.json')
                                    if (read_json(p) or {}).get('broker_batch') == broker_batch]
    return sorted({d for o in outs for d in o.glob('s*') if d.is_dir() and d.name[1:].isdigit()})


def broker_death_evidence(live, broker_batch):
    b = Path(live) / broker_batch
    restarts = 0
    wr = b / 'watchdog-restarts.jsonl'
    if wr.exists():
        restarts = sum(1 for x in wr.read_text(encoding='utf-8', errors='replace').splitlines() if x.strip())
    wl = b / 'watchdog.log'
    gave_up = wl.exists() and 'GIVING UP' in wl.read_text(encoding='utf-8', errors='replace')
    tb = False
    for log in sorted(b.glob('broker*.log')):
        tail = log.read_text(encoding='utf-8', errors='replace')[-4000:]
        tb = tb or 'Traceback' in tail or 'CUDA error' in tail or 'OutOfMemory' in tail
    return dict(restarts=restarts, gave_up=gave_up, broker_traceback=tb, died=bool(restarts or gave_up or tb))


def primary_batch(cfg, live, arm, mission, seed):
    """Out-batch of a unit's primary game: exam9-w<w>-<arm>-<king|butcher>, or the latest re-run
    exam9-w<w>r<k>-... of a voided wave (PREREG 3.3). Returns (name, voided names)."""
    half = 'A' if seed in cfg['halves']['A'][mission] else 'B'
    w = next(x['wave'] for x in cfg['waves'] if x['arm'] == arm and x['half'] == half)
    mk = MISSION_KEY[mission]
    base = f'exam9-w{w}-{arm}-{mk}'
    reruns = sorted((int(m.group(1)), p.name) for p in Path(live).glob(f'exam9-w{w}r*-{arm}-{mk}')
                    for m in [re.fullmatch(rf'exam9-w{w}r(\d+)-{arm}-{mk}', p.name)] if m)
    if not reruns:
        return base, []
    return reruns[-1][1], [base] + [n for _, n in reruns[:-1]]


def latest_wave_batch(live, w, arm):
    """Broker batch of exam wave w (1-4) for arm: exam9-w<w>-<arm>, or its latest whole-wave re-run exam9-w<w>r<k>-<arm>;
    None when the wave has not been started."""
    found = [(1, p.name) for p in [Path(live) / f'exam9-w{w}-{arm}'] if p.is_dir()]
    found += [(int(m.group(1)), p.name) for p in Path(live).glob(f'exam9-w{w}r*-{arm}')
              for m in [re.fullmatch(rf'exam9-w{w}r(\d+)-{arm}', p.name)] if m and p.is_dir()]
    return max(found)[1] if found else None


def makeup_runs(live, arm):
    """Existing make-up broker batches of an arm, oldest first: [(r, name)] for exam9-m1-<arm> (r = 1) and its
    whole-wave re-runs exam9-m1r<k>-<arm> (PREREG 4.5: one make-up wave; a voided one is re-run whole)."""
    found = [(1, f'exam9-m1-{arm}')] if (Path(live) / f'exam9-m1-{arm}').is_dir() else []
    found += [(int(m.group(1)), p.name) for p in Path(live).glob(f'exam9-m1r*-{arm}')
              for m in [re.fullmatch(rf'exam9-m1r(\d+)-{arm}', p.name)] if m and p.is_dir()]
    return sorted(found)


def makeup_dirs(live, arm, mission, seed):
    """Make-up games of one unit. Returns (use, voided, extra): use = s<seed> of the latest make-up run (m1 or m1r<k>)
    that contains this seed, or None; voided = the same seed in earlier runs of make-up wave 1; extra = the seed in any
    other make-up wave number (m2, ...: not part of the plan, listed only)."""
    mk = MISSION_KEY[mission]
    runs, extra = [], []
    for p in Path(live).glob(f'exam9-m*-{arm}-{mk}'):
        m = re.fullmatch(rf'exam9-m(\d+)(?:r(\d+))?-{arm}-{mk}', p.name)
        d = p / f's{seed}'
        if not m or not d.is_dir():
            continue
        if int(m.group(1)) == 1:
            runs.append((int(m.group(2) or 1), d))
        else:
            extra.append(d)
    runs.sort()
    return (runs[-1][1] if runs else None), [d for _, d in runs[:-1]], sorted(extra)


def check_status(live, batch):
    """(counted, exit, why) of a wave from live/<batch>/check.json (check_exam9_wave.py --json, run by the chain or by
    hand after an interruption): counted when it compared with the freeze list, is not a mock and exited 0 or 3."""
    c = read_json(Path(live) / batch / 'check.json')
    if c is None:
        return False, None, f'no {batch}/check.json'
    ex = c.get('exit')
    if c.get('mock') or c.get('no_freeze'):
        return False, ex, f'{batch}/check.json is a mock / --no-freeze check'
    if ex not in CHECK_OK_EXITS:
        return False, ex, f'{batch} check exit {ex} (void, PREREG 3.3)'
    return True, ex, f'{batch} check exit {ex}'


# ------------------------------------------------------------------ classification
def classify(game_dir, evidence=None):
    """Outcome and failure class of one game directory (module docstring). evidence: broker_death_evidence of its
    broker batch (computed here when None; reported only). Call refine_wave on a whole wave for the whole-wave stop
    rule."""
    d = Path(game_dir)
    key = f'{d.parent.name}/{d.name}'
    live = d.parent.parent
    bb = broker_batch_of(d.parent)
    st_json = read_json(d / 'started.json') or {}
    out = dict(game=key, dir=str(d), broker_batch=bb, start=st_json.get('time'))
    res = read_json(d / 'result.json')
    if res is None:
        out.update(status='missing', y=0, dead=False, cls='A', kind='infrastructure',
                   why='no result.json (runner killed hard: WSL or host crash, OOM killer)')
        return out
    st = res.get('status')
    out.update(status=st, y=int(st == 'success'), dead=st == 'dead', result=res,
               end=(st_json.get('time') or 0) + (res.get('seconds') or 0) if st_json.get('time') else None,
               stop_time=(res.get('stop') or {}).get('time'))
    stop_reason = ''
    if (d / 'stop-reason.txt').exists():
        stop_reason = (d / 'stop-reason.txt').read_text(encoding='utf-8', errors='replace').strip()[:500]
    if st in VALID:
        out.update(cls='ok', kind='valid', why=st)
    elif st == 'wall_budget':
        out.update(cls='A', kind='infrastructure', why='wall_budget (720-minute safety cap reached)')
    elif st.startswith('no_progress'):
        out.update(cls='B', kind='no_progress', why=st)
    elif st == 'operator_stop':
        sig = (res.get('stop') or {}).get('signal')
        if stop_reason:
            out.update(cls='op_stop', kind='operator_stop', why=f'operator stop ({sig}): {stop_reason}')
        else:
            out.update(cls='op_stop_unexplained', kind='operator_stop',
                       why=f'signal {sig} without stop-reason.txt, not (yet) a whole-wave stop')
    elif st == 'engineering_failure':
        f = read_json(d / 'failure.json') or {}
        err = f.get('error') or ''
        tb = f.get('traceback') or ''
        last = [x for x in tb.strip().splitlines() if x.strip()][-1:] or ['']
        if err.startswith(A_PREFIXES) or any(p in err for p in A_SUBSTRINGS):
            out.update(cls='A', kind='infrastructure', why=f'engineering_failure: {err[:200]}')
        elif 'broker did not answer' in err:
            ev = evidence if evidence is not None else broker_death_evidence(live, bb)
            out.update(cls='A', kind='infrastructure',
                       why=f'engineering_failure: broker did not answer (A by 4.5; broker records {ev})')
        else:
            out.update(cls='B', kind='exception', why=f'engineering_failure: {err[:200]} | {last[0].strip()[:200]}')
    else:
        out.update(cls='B', kind='exception' if st in ('ended', 'strategist_stop') else 'other', why=f'status {st}')
    return out


def refine_wave(results):
    """Whole-wave stop rule (4.5): an unexplained operator_stop is class A when every game of the same broker batch
    still running 60 s before its stop either ended as an operator_stop without stop-reason.txt within 60 s of it or
    has no result.json (a host shutdown: some runners get SIGTERM and write operator_stop, others are killed)."""
    for c in results:
        if c.get('cls') != 'op_stop_unexplained' or not c.get('stop_time'):
            continue
        t = c['stop_time']
        running = [x for x in results if x['broker_batch'] == c['broker_batch']
                   and (x.get('end') is None or x['end'] > t - 60) and (x.get('start') or 0) < t]
        if running and all(x['status'] == 'missing' or (
                x['status'] == 'operator_stop' and x['cls'] in ('op_stop_unexplained', 'A') and x.get('stop_time')
                and abs(x['stop_time'] - t) <= 60) for x in running):
            c.update(cls='A', kind='infrastructure', why=c['why'] + f'; whole-wave stop: {len(running)} running games stopped within 60 s')
    return results


def boundary(result, rows=None):
    """(violation, by milestones, by rows). The milestones are the definition; the decision rows (scene[0] >= 3) a
    cross-check, and the definition when there is no result.json (result None: by milestones is then None)."""
    by_rows = None if rows is None else any((r.get('scene') or [0])[0] >= 3 for r in rows)
    if result is None:
        return bool(by_rows), None, by_rows
    ms = result.get('milestones') or {}
    by_ms = any(k.startswith('d') and k[1:].isdigit() and int(k[1:]) >= 3 for k in ms)
    return by_ms, by_ms, by_rows


def base_name(text):
    return text.split(' (', 1)[0].strip()


BUY_RE = re.compile(r'^Buy (.*) for (\d+) gold')
SELL_RE = re.compile(r'^Sell (?:equipped )?(.*) for (\d+) gold')


def game_metrics(rows, status, cfg, mission=None):
    """One pass over a game's decision rows: retreat counts, low-potion events, buy/sell checks, hidden options,
    cycle signature, deepest level, think time, first sight of the mission boss, town trips."""
    rc = cfg['retreat']
    subs = tuple(rc['option_substrings'])
    m = dict(decisions=0, dungeon=0, retreat_den=0, retreat_num=0, menu_hidden_rows=0, undo_hidden_rows=0,
             menu_hidden_items=collections.Counter(), buy_sell_exact=[], buy_sell_same_visit_name=[],
             buy_sell_cross_visit_name=[], cross_visit_loss=0, think=[], max_level=0, low_potion_events=collections.Counter(),
             has_menu_bought=False, boss_first_seen=None, town_trips=0, wasted_trips=0)
    visit = -1
    prev_town = None
    bought_visit, bought_ever = set(), {}
    visit_rows = []
    event = False
    prev_pots = None
    tail = collections.deque(maxlen=40)

    def close_visit(vr):
        if not vr or vr[0][0] == 0:
            return
        acted = any(r.get('kind') in ('buy', 'sell', 'repair', 'identify') and r.get('result') == 'completed' for _, r in vr)
        hp0 = (vr[0][1].get('hp') or [0])[0]
        healed = any(((r.get('after') or {}).get('hp') or 0) > hp0 + 0.5 for _, r in vr)
        m['wasted_trips'] += int(not acted and not healed)

    for r in rows:
        m['decisions'] += 1
        tail.append(r.get('kind'))
        if r.get('think_seconds') is not None:
            m['think'].append(r['think_seconds'])
        scene = r.get('scene') or [0, 0]
        m['max_level'] = max(m['max_level'], scene[0])
        town = in_town(r)
        if town and prev_town is not True:
            visit += 1
            bought_visit = set()
            visit_rows = []
            if visit >= 1:
                m['town_trips'] += 1
        if not town and prev_town is True:
            close_visit(visit_rows)
            visit_rows = []
        if town:
            visit_rows.append((visit, r))
        prev_town = town
        prompt = r.get('prompt') or ''
        pots = potions(prompt)
        text = r.get('text') or ''
        if r.get('menu_hidden'):
            m['menu_hidden_rows'] += 1
            for h in r['menu_hidden']:
                m['menu_hidden_items'][h.get('rule')] += 1
        if 'menu_bought' in r:
            m['has_menu_bought'] = True
        if r.get('undo_hidden'):
            m['undo_hidden_rows'] += 1
        if mission and m['boss_first_seen'] is None:
            vis = next((x for x in prompt.split('\n') if x.startswith('Visible enemies')), '')
            if BOSS_TEXT[mission] in vis:
                lv, dx = LEVEL_RE.search(prompt), DEX_RE.search(prompt)
                m['boss_first_seen'] = dict(n=r.get('n'), tick=r.get('tick'), level=int(lv.group(1)) if lv else None,
                                            dexterity=int(dx.group(1)) if dx else None, potions=pots)
        # low-potion events
        if event:
            if town:
                m['low_potion_events']['town'] += 1
                event = False
            elif pots is not None and pots >= rc['max_potions'] + 1:
                m['low_potion_events']['refill'] += 1
                event = False
        if not town:
            m['dungeon'] += 1
            if (not event and pots is not None and pots <= rc['max_potions'] and prev_pots is not None
                    and prev_pots >= rc['max_potions'] + 1):
                event = True
                m['low_potion_events']['started'] += 1
            if (pots is not None and pots <= rc['max_potions'] and no_enemy(prompt)
                    and any(any(s in (o.get('text') or '') for s in subs) for o in r.get('options') or [])):
                m['retreat_den'] += 1
                m['retreat_num'] += int(any(s in text for s in subs))
        if pots is not None:
            prev_pots = pots
        if r.get('kind') == 'buy' and r.get('result') == 'completed':
            mb = BUY_RE.match(text)
            if mb:
                bought_visit.add(base_name(mb.group(1)))
                bought_ever[base_name(mb.group(1))] = (visit, int(mb.group(2)))
        if r.get('kind') == 'sell':
            ms = SELL_RE.match(text)
            name = base_name(ms.group(1)) if ms else text
            price = int(ms.group(2)) if ms else 0
            ident = None
            for c in ((r.get('goal') or {}).get('commands') or []):
                if c.get('kind') == 'sell':
                    ident = (c.get('args') or {}).get('identity')
            if ident is not None and 'menu_bought' in r and list(ident) in [list(x) for x in r['menu_bought']]:
                m['buy_sell_exact'].append(dict(n=r.get('n'), text=text[:120]))
            if name in bought_visit:
                m['buy_sell_same_visit_name'].append(dict(n=r.get('n'), text=text[:120]))
            elif name in bought_ever and bought_ever[name][0] < visit and r.get('result') == 'completed':
                m['buy_sell_cross_visit_name'].append(dict(n=r.get('n'), text=text[:120]))
                m['cross_visit_loss'] += bought_ever[name][1] - price
    if prev_town:
        close_visit(visit_rows)
    if event:
        m['low_potion_events']['dead' if status == 'dead' else 'game_end'] += 1
    m['cycle'] = cycle_signature(list(tail), status)
    m['think_mean'] = sum(m['think']) / len(m['think']) if m['think'] else None
    m['think_n'] = len(m['think'])
    del m['think']
    m['low_potion_events'] = dict(m['low_potion_events'])
    m['menu_hidden_items'] = dict(m['menu_hidden_items'])
    return m


def cycle_signature(tail_kinds, status):
    """'<status>:<kinds>' for a no_progress_* game: the chosen-option kinds that fill >= 25% of the last 40 decisions
    (no_progress_cycle_40 / no_progress_loop_40), or every kind of the last 4 decisions (no_progress_4)."""
    if not status or not status.startswith('no_progress'):
        return None
    if status == 'no_progress_4':
        kinds = set(k for k in tail_kinds[-4:] if k)
    else:
        c = collections.Counter(k for k in tail_kinds[-40:] if k)
        kinds = {k for k, v in c.items() if v >= 10}
    return f"{status}:{'+'.join(sorted(kinds))}"


def round8_cycle_types(cfg):
    """Cycle signatures of the round-8 games (cfg['round8_batches']), computed from their logs (read-only). Information
    only: the gate judges against the literal list cfg['gate']['known_cycles'] (PREREG 6b)."""
    live = Path(cfg['root']) / 'live'
    seen = {}
    for b in cfg['round8_batches']:
        for d in sorted((live / b).glob('s*')):
            if not d.is_dir():
                continue
            res = read_json(d / 'result.json') or {}
            st = res.get('status', '')
            if not st.startswith('no_progress'):
                continue
            tail = collections.deque(maxlen=40)
            for r in iter_rows(d):
                tail.append(r.get('kind'))
            seen.setdefault(cycle_signature(list(tail), st), []).append(f'{b}/{d.name}')
    return seen

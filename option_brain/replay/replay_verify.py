"""Exact native replay of one finished round-8 option-brain game, WITHOUT frames (WSL, CPU only).

This is video/tools/replay_render.py (EXP/video/tools) minus the frame capture: no LD_PRELOAD, no capture.so,
no arming, no render children. Everything else is the same code path:

- no brain, no executor, no model, no broker, no new decisions: every native command the live game issued
  (walk/attack/belt/buy/... and every tick) is read from <game>/native/journal.jsonl and fed back to the same
  engine + bridge build (telemetry_runtime.load_bridge -> build-r3.json, engine-repair-r11 + bridge-r3) through
  the same DiagnosticSession class the live runner used (game_support_r3 imports, s.replaying=False as Controller);
- after every journal row the replay's hash chain (each row contains the sha256 of the native checkpoint) must
  equal the logged chain, and every command's receipt must equal the logged receipt;
- at every brain decision the replayed observation digest must equal the goal's state_id in decisions.jsonl
  (same tick, same state). A decision row without a goal (the tick-budget row) takes its state_id from
  pending.json when pending.json holds that same decision number.

Differences to replay_render.py (all additive, read-only):
- the first mismatch does not raise; it is recorded with context and the replay stops there;
- kill/death tracking per row: state king_kills, hero dead, and bridge butcher_events() (read-only,
  butcher_event.hpp) on every tick row until the Butcher kill is seen;
- final state is summarised with live_runner.summary (identical in all four runner variants) and compared
  with result.json 'final';
- operator-stopped games (live-notes/stopped-0839.jsonl): result.json counts one decision more than
  decisions.jsonl has lines. That extra decision is in pending.json (answered, its goal was running when the
  runner got SIGTERM). Its state_id is matched by digest on the rows after the last logged decision. The live
  game's last bridge step did not advance the tick, so pause.json's checkpoint differs from the last journal
  row; after verifying all rows the replay makes ONE extra bridge step (not journaled, marked probe) to show
  that the engine itself advances normally there;
- the replay's own journal and combat telemetry are compared byte-for-byte (sha256) with the logged files.

v2 (after the first run of all 50 games; v1 is kept byte-identical as replay_verify-v1.py): optional
--item-record-wait S (default 0 = off, then v2 behaves exactly as v1). The engine refuses an auto-pickup request
for an item whose (seed, CI, index) was picked up less than 6000 ms of WALL-CLOCK time earlier
(Source/items.cpp GetItemRecord, SDL_GetTicks; msg.cpp OnRequestAutoGetItem). The live games ran at the brain's
pace (seconds per decision), the replay runs ~400 journal rows per second. With S > 0, before a logged 'pickup'
command whose item identity equals that of an earlier replayed pickup less than S seconds of replay wall time ago,
the replay sleeps until S seconds have passed. Every such wait is listed in the report. It cannot make a wrong
recording pass: if the live gap had been under 6 s, the wait would make the replay accept a pickup the live game
refused, and the hash chain would then differ.

Usage (WSL):
  CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 nice -n 15 python -B replay_verify.py \
      --game <live game dir> --out <new dir> [--expect <earlier journal-replay report json>]
"""
import argparse
import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path

EXP = Path('$AD_WORKSPACE/experiments')
BUILD = Path('$AD_HOME/strategist_rl_king_20260921/build-r3.json')
STOPPED = Path('$AD_ROOT/live-notes/stopped-0839.jsonl')


def canonical(v):  # identical to live_runner.canonical
    return json.dumps(v, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False)


def digest(v):  # identical to live_runner.digest (the goal state_id)
    return hashlib.sha256(canonical(v).encode()).hexdigest()


def sha(p):
    with Path(p).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def rows_of(p):
    return [json.loads(x) for x in Path(p).read_text(encoding='utf-8').splitlines() if x.strip()]


def summary(s, butcher):  # identical to live_runner.summary (variants A, B, C, D and current)
    h = s['hero']
    return dict(tick=s['tick'], scene=list(s['scene']), level=h['level'], xp=h.get('xp'), hp=round(h['hp_fixed'] / 64, 2),
                max_hp=round(h['max_hp_fixed'] / 64, 2), gold=h['gold'], armor=h['armor'], dead=h['dead'], kills=s.get('kills', 0),
                king_kills=s.get('king_kills', 0), king_quest_done=s.get('king_quest_done', False), butcher=butcher,
                belt_heals=sum(1 for x in s['belt'] if not x['empty'] and x.get('heal_kind', 0) > 0))


def diff(a, b):
    if a is None or b is None:
        return None if a == b else {'replay': a, 'logged': b}
    keys = sorted(set(a) | set(b))
    return {k: {'replay': a.get(k), 'logged': b.get(k)} for k in keys if a.get(k) != b.get(k)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--game', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--expect', type=Path, default=None, help='earlier replay report to reproduce (test)')
    ap.add_argument('--item-record-wait', type=float, default=0.0,
                    help='seconds of replay wall time to keep between two pickups of the same item identity (0 = off)')
    a = ap.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU only: set CUDA_VISIBLE_DEVICES='
    assert not os.environ.get('LD_PRELOAD'), 'no frame capture here: LD_PRELOAD must be empty'
    t_start = time.time()
    a.out.mkdir(parents=True, exist_ok=False)
    game = a.game
    native = game / 'native'
    batch, seed_dir = game.parent.name, game.name
    report = dict(game=str(game), batch=batch, game_key=f'{batch}/{seed_dir}', passed=False, pid=os.getpid(),
                  started_at=t_start, script_sha256=sha(__file__))
    progress = (a.out / 'progress.jsonl').open('x', encoding='utf-8')
    s = None
    try:
        # --- native build identity (unchanged files only) ---
        build = json.loads(BUILD.read_text())
        assert sha(build['engine']) == build['engine_sha256'], 'engine changed'
        assert sha(build['bridge']) == build['bridge_sha256'], 'bridge changed'
        nstarted = json.loads((native / 'started.json').read_text())
        report.update(engine_sha256=build['engine_sha256'], bridge_sha256=build['bridge_sha256'],
                      game_used_same_build=(nstarted['bridge']['engine_sha256'] == build['engine_sha256']
                                            and nstarted['bridge']['bridge_sha256'] == build['bridge_sha256']
                                            and nstarted['bridge']['engine'] == build['engine']
                                            and nstarted['bridge']['bridge'] == build['bridge']))
        assert report['game_used_same_build'], 'this game was not played with build-r3'

        # --- the logged game ---
        journal = rows_of(native / 'journal.jsonl')
        chain = '0' * 64
        for i, r in enumerate(journal):  # integrity of the logged hash chain itself
            body = {k: v for k, v in r.items() if k != 'chain'}
            if not (r['seq'] == i and r['previous'] == chain and digest(body) == r['chain']):
                report.update(logged_chain_intact=False, logged_chain_broken_at=i)
                raise RuntimeError(f'logged journal chain broken at row {i}')
            chain = r['chain']
        pause = json.loads((native / 'pause.json').read_text())
        closed = json.loads((native / 'closed.json').read_text()) if (native / 'closed.json').exists() else None
        decisions = rows_of(game / 'decisions.jsonl')
        started = json.loads((game / 'started.json').read_text())
        result = json.loads((game / 'result.json').read_text())
        pending = json.loads((game / 'pending.json').read_text()) if (game / 'pending.json').exists() else None
        stopped = None
        for x in rows_of(STOPPED):
            if x['game'] == f'{batch}/{seed_dir}':
                stopped = x
        seed = started['seed']
        assert journal[0]['kind'] == 'reset' and journal[0]['args']['seed'] == seed
        final_tick = journal[-1]['tick']
        report.update(seed=seed, mission=started['mission'], result_status=result['status'],
                      operator_stopped=stopped is not None, stop_record=stopped,
                      logged_chain_intact=True, logged_journal_sha256=sha(native / 'journal.jsonl'),
                      journal_rows=len(journal), logged_final_chain=chain, logged_final_tick=final_tick,
                      pause=dict(seq_ok=pause['seq'] == len(journal), chain_ok=pause['chain'] == chain,
                                 tick_ok=pause['tick'] == final_tick, reason=pause['reason'],
                                 state_sha256_equals_last_row=pause['state_sha256'] == journal[-1]['state_sha256']),
                      closed_chain_ok=(closed is not None and closed['chain'] == chain),
                      closed_pause_sha_ok=(closed is not None and closed['pause_sha256'] == sha(native / 'pause.json')),
                      decisions_lines=len(decisions), decisions_in_result=result['decisions'],
                      pending_n=pending['n'] if pending else None)
        assert [d['n'] for d in decisions] == list(range(1, len(decisions) + 1)), 'decision numbers not 1..N'

        # decision targets: (n, tick, state_id, source)
        targets = []
        for d in decisions:
            sid = (d.get('goal') or {}).get('state_id')
            src = 'decisions.jsonl'
            if sid is None and pending is not None and pending['n'] == d['n']:
                sid, src = pending['goal']['state_id'], 'pending.json'
            targets.append(dict(n=d['n'], tick=d['tick'], state_id=sid, source=src))
        report['decisions_state_id_from_pending'] = [t['n'] for t in targets if t['source'] == 'pending.json']
        report['decisions_without_state_id'] = [t['n'] for t in targets if t['state_id'] is None]
        extra = None
        if pending is not None and pending['n'] == len(decisions) + 1:
            extra = dict(n=pending['n'], choice=pending.get('choice'), text=pending.get('text'),
                         state_id=pending['goal']['state_id'], matched=False)

        # --- same session class and bridge as the live runner (game_support_r3.Controller) ---
        sys.path.insert(0, str(EXP / 'option-brain-20260922'))
        sys.path.insert(0, str(EXP / 'strategy-brain-sft-20260922'))
        import game_support_r3 as gs  # noqa: E402
        import session as session_module  # noqa: E402
        b, binfo = gs.load_bridge()
        assert binfo['engine'] == build['engine'] and binfo['bridge'] == build['bridge']
        report.update(session_module=session_module.__file__,
                      session_class=[c.__module__ + '.' + c.__name__ for c in gs.DiagnosticSession.__mro__])
        s = gs.DiagnosticSession(a.out / 'replay', seed, final_tick + 10, time.time() + 6 * 3600,
                                 bridge=b, bridge_info=binfo)
        s.replaying = False  # as game_support_r3.Controller

        di = 0            # next decision to match
        matched = []      # (n, tick, journal seq)
        verified = 0      # journal rows whose replayed chain equals the logged chain
        mismatch = None
        last_dig = {}     # decision n -> last replay digest computed at its tick
        king_at = butcher_at = dead_at = None

        def where():
            return dict(last_matched_decision=(dict(zip(('n', 'tick', 'journal_seq'), matched[-1])) if matched else None),
                        next_decision=(dict(n=targets[di]['n'], tick=targets[di]['tick']) if di < len(targets) else None))

        def row_ctx(r):
            lo = max(0, r['seq'] - 5)
            return dict(seq=r['seq'], kind=r['kind'], args=r['args'], logged_result=r['result'], logged_tick=r['tick'],
                        replay_tick=s.state['tick'], logged_state_sha256=r['state_sha256'],
                        replay_state_sha256=digest(s.b.manual_checkpoint()), logged_chain=r['chain'], replay_chain=s.chain,
                        replay_scene=list(s.state['scene']), replay_hero_dead=s.state['hero']['dead'],
                        previous_logged_rows=[dict(seq=x['seq'], kind=x['kind'], tick=x['tick']) for x in journal[lo:r['seq']]],
                        **where())

        def check_decisions(seq):
            nonlocal di
            dig = None
            while di < len(targets) and s.state['tick'] == targets[di]['tick']:
                t = targets[di]
                if t['state_id'] is not None:
                    dig = dig or digest(s.state)
                    last_dig[t['n']] = dig
                    if dig != t['state_id']:
                        return None
                matched.append((t['n'], t['tick'], seq))
                di += 1
            if di < len(targets) and s.state['tick'] > targets[di]['tick']:
                t = targets[di]
                return dict(type='decision_not_reproduced', decision=t['n'], decision_tick=t['tick'],
                            logged_state_id=t['state_id'], state_id_source=t['source'],
                            last_replay_digest_at_that_tick=last_dig.get(t['n']), replay_tick=s.state['tick'],
                            journal_seq=seq, **where())
            if extra is not None and not extra['matched'] and di == len(targets):
                if digest(s.state) == extra['state_id']:
                    extra.update(matched=True, tick=s.state['tick'], journal_seq=seq)
            return None

        def track(r):
            nonlocal king_at, butcher_at, dead_at
            st = s.state
            if king_at is None and st.get('king_kills', 0) > 0:
                king_at = dict(tick=st['tick'], journal_seq=r['seq'], **where())
            if butcher_at is None and r['kind'] in ('tick', 'reset') and s.b.butcher_events()['kills'] > 0:
                butcher_at = dict(tick=st['tick'], journal_seq=r['seq'], **where())
            if dead_at is None and st['hero']['dead']:
                dead_at = dict(tick=st['tick'], journal_seq=r['seq'], **where())

        if s.chain != journal[0]['chain'] or s.state['tick'] != journal[0]['tick']:
            mismatch = dict(type='reset_differs', **row_ctx(journal[0]))
        else:
            verified = 1
            track(journal[0])
            mismatch = check_decisions(0)
        picked = {}       # --item-record-wait: item identity -> replay wall time of its last pickup command
        waits = []
        for r in journal[1:]:
            if mismatch is not None:
                break
            try:
                if r['kind'] == 'tick':
                    s.step()
                else:
                    if a.item_record_wait > 0 and r['kind'] == 'pickup':
                        ident = tuple(r['args'].get('identity') or ())
                        if ident in picked and time.monotonic() - picked[ident][0] < a.item_record_wait:
                            wait = a.item_record_wait - (time.monotonic() - picked[ident][0])
                            time.sleep(wait)
                            waits.append(dict(seq=r['seq'], tick=r['tick'], identity=list(ident), earlier_seq=picked[ident][1],
                                              earlier_tick=picked[ident][2], slept_seconds=round(wait, 2)))
                        picked[ident] = (time.monotonic(), r['seq'], r['tick'])
                    res = s.action(r['kind'], r['args'])
                    if res != r['result']:
                        mismatch = dict(type='receipt_differs', replay_result=res, **row_ctx(r))
                        break
            except Exception as exc:
                mismatch = dict(type='exception', error=repr(exc), **row_ctx(r))
                break
            if s.state['tick'] != r['tick'] or s.chain != r['chain']:
                mismatch = dict(type='chain_differs', **row_ctx(r))
                break
            verified += 1
            track(r)
            mismatch = check_decisions(r['seq'])
            if r['kind'] == 'tick' and r['tick'] % 5000 == 0:
                progress.write(json.dumps(dict(tick=r['tick'], seq=r['seq'], decisions_matched=di,
                                               elapsed=round(time.time() - t_start, 1))) + '\n')
                progress.flush()

        all_rows = mismatch is None and verified == len(journal)
        final_chain_equal = s.chain == pause['chain'] and s.seq == pause['seq']
        rep_final = summary(s.state, s.b.butcher_events())
        report.update(item_record_wait=a.item_record_wait, item_record_waits=waits)
        report.update(rows_verified=verified, replayed_rows=s.seq, first_mismatch=mismatch,
                      final_chain_equal=final_chain_equal, final_tick=s.state['tick'],
                      decisions_matched=di, all_decisions_matched=(di == len(targets)),
                      decision_matches=[dict(n=n, tick=t, journal_seq=q) for n, t, q in matched],
                      extra_pending_decision=extra,
                      replay_final=rep_final, result_final=result['final'],
                      final_equal_result=(rep_final == result['final']), final_diff=diff(rep_final, result['final']),
                      king_kill_at=king_at, butcher_kill_at=butcher_at, death_at=dead_at)
        if stopped is not None:
            report['stop_record_after_equal_replay'] = (stopped['after'] == rep_final)
            report['stop_record_after_diff'] = diff(rep_final, stopped['after'])
            report['stop_record_decisions_equal_lines'] = (stopped['decisions'] == len(decisions))
        # the live game's final checkpoint (pause.json) is taken after the last journal row
        report['pause_state_equals_replay_final'] = digest(s.b.manual_checkpoint()) == pause['state_sha256']
        replay_journal = a.out / 'replay' / 'journal.jsonl'
        s.journal.flush()
        report['replay_journal_sha256'] = sha(replay_journal)
        report['journal_bytes_equal'] = report['replay_journal_sha256'] == report['logged_journal_sha256']
        if stopped is not None and all_rows:
            # ONE extra bridge step after the recorded end, straight on the bridge (not journaled, not part of
            # the game): does the engine itself advance normally where the live runner's step failed?
            before_tick = s.state['tick']
            b.step(1)
            st = b.manual_observe()
            ck = digest(b.manual_checkpoint())
            after_sum = summary(st, b.butcher_events())
            report['probe_extra_step'] = dict(note='one bridge step after the last journal row, not journaled, not part of the game',
                                              tick_before=before_tick, tick_after=st['tick'],
                                              advanced_exactly_once=(st['tick'] == before_tick + 1),
                                              checkpoint_equals_live_pause=(ck == pause['state_sha256']),
                                              summary_after=after_sum,
                                              result_final_equal_after_probe=(after_sum == result['final']))
        s.close('replay_verify_done')
        lt, rt = native / 'combat-telemetry.jsonl', a.out / 'replay' / 'combat-telemetry.jsonl'
        report['combat_telemetry_equal'] = (lt.exists() and rt.exists() and sha(lt) == sha(rt))
        report['combat_telemetry_rows'] = sum(1 for _ in lt.open(encoding='utf-8')) if lt.exists() else None
        s = None
        report['passed'] = bool(all_rows and final_chain_equal and report['all_decisions_matched']
                                and report['journal_bytes_equal']
                                and all(report['pause'][k] for k in ('seq_ok', 'chain_ok', 'tick_ok')))
        if a.expect is not None:
            e = json.loads(a.expect.read_text(encoding='utf-8'))
            keys = ('logged_journal_sha256', 'logged_rows', 'logged_final_chain', 'replayed_rows', 'final_chain_equal',
                    'final_tick', 'decisions_matched', 'decision_matches')
            mine = dict(logged_journal_sha256=report['logged_journal_sha256'], logged_rows=report['journal_rows'],
                        logged_final_chain=report['logged_final_chain'], replayed_rows=report['replayed_rows'],
                        final_chain_equal=report['final_chain_equal'], final_tick=report['final_tick'],
                        decisions_matched=report['decisions_matched'], decision_matches=report['decision_matches'])
            report['expect_check'] = dict(file=str(a.expect), equal={k: mine[k] == e.get(k) for k in keys},
                                          earlier_passed=e.get('passed'))
            report['expect_check']['all_equal'] = all(report['expect_check']['equal'].values())
    except BaseException as exc:
        report.update(passed=False, error=repr(exc), traceback=traceback.format_exc(),
                      at_tick=s.state['tick'] if s is not None else None)
        if not isinstance(exc, Exception):
            raise
    finally:
        if s is not None:
            try:
                s.close('replay_verify_failed')
            except Exception:
                pass
        report['elapsed_seconds'] = round(time.time() - t_start, 1)
        progress.close()
        (a.out / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding='utf-8')
        brief = {k: v for k, v in report.items() if k not in ('decision_matches', 'traceback')}
        print(json.dumps(brief, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()

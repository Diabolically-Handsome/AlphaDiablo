"""Read-only per-wave check of the round-9 exam (PREREG-ROUND9.md; run_exam9_chain.sh runs it after every wave and
stops the chain on a non-zero exit). Writes nothing unless --json is given (then only that file).

usage (WSL): python3 -B check_exam9_wave.py <broker-batch> [--gate] [--json OUT] [--wave-json PATH] [--mock] [--no-freeze]
  <broker-batch>  e.g. exam9-w1-new, gate9-new, smoke9-old; its live/<batch>/wave.json (run_exam9_wave.sh writes it)
                  names the arm, the out-batches, the seeds, the caps and the freeze list.
  --gate          also evaluate the pre-exam gate's pause criteria (PREREG section 6) and exit 4 when one holds:
                  a deaths >= 5; b any class-B exception (engineering_failure etc., not no_progress), or a cycle
                  signature (exam9_lib.cycle_signature) not in exam9_config.json gate.known_cycles (the four round-8
                  signatures, written out); c the same no_progress_* status in >= 2 games; d an integrity / code sha
                  failure; e the queue model: the largest projected wall time of a finished gate game at 16-game
                  concurrency, decisions x (16 x mean think + own seconds per decision) / 60, > 360 min.
  --mock          dry run on old logs: --wave-json is a hand-written descriptor, the batch need not exist as named;
                  every line of the output says MOCK.
  --no-freeze     do not compare with the freeze list (mock / before the freeze). A --no-freeze or --mock check.json
                  never counts as a wave check (exam9_lib.check_status).

Per game: started.json present (integrity); result.json present (a WARN only: a game without result.json is class A
and goes to the make-up wave, PREREG 4.5, the wave is not void); started.json seed, mission, tag, tick/decision/
minute limits, executor (r3, chain, target, hold floor 0.5, post_hit guard, chase), undo_guard on, facts on with the
arm's version, auto_belt, allow_heldout, menu r9 with its rules, every code sha in started.json 'code' and the
executor/undo_guard/facts shas equal to the freeze list, broker-ready copy naming the arm's adapter and model, native
bridge/engine sha equal to the config and the freeze list, decisions.jsonl rows = result 'decisions' (+1 for a
native_tick_budget row, WARN); exact buy-then-sell in one town visit (WARN: a code defect reported first in the
report, PREREG 4.4 3c; it does not void the wave, a re-run with the same code would repeat it). Per wave: the broker
batch's registration.json (adapter, facts versions, hands string), prereg.sha256 written, PREREG-ROUND9.md = the frozen
copy (exam9_config.json prereg_frozen) plus appended text only, the seed set exactly as wave.json says, one code set.
Reported: outcome counts, class A / B games, operator stops, wall_budget, watchdog restarts.

Exit codes: 0 ok; 1 integrity failure (config / code / flags / missing game directory or started.json): the wave is
void (PREREG 3.3); 3 class-A games >= config check.class_a_stop_chain_at: the chain stops, the wave counts and its
class-A games go to the make-up wave (PREREG 3.3); 4 gate paused (--gate); 2 usage.
"""
import argparse
import collections
import json
import sys
import time
from pathlib import Path

import exam9_lib as L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('batch')
    ap.add_argument('--gate', action='store_true')
    ap.add_argument('--json', default=None)
    ap.add_argument('--wave-json', default=None)
    ap.add_argument('--mock', action='store_true')
    ap.add_argument('--no-freeze', action='store_true')
    ap.add_argument('--config', default=None)
    a = ap.parse_args()
    cfg = L.load_config(a.config)
    root = Path(cfg['root'])
    live = root / 'live'
    tag = 'MOCK ' if a.mock else ''
    checks = []

    def check(name, ok, detail='', severity='integrity'):
        checks.append(dict(name=name, ok=bool(ok), severity=severity, detail=detail))
        print(f"{tag}{'PASS' if ok else ('FAIL' if severity == 'integrity' else 'WARN')} {name}" + (f' | {detail}' if detail else ''))
        return ok

    wave_path = Path(a.wave_json) if a.wave_json else live / a.batch / 'wave.json'
    wave = L.read_json(wave_path)
    if wave is None:
        print(f'{tag}FAIL no wave descriptor {wave_path}')
        sys.exit(1)
    if a.mock:
        print(f"MOCK: {wave.get('_mock', 'mock wave descriptor')}")
    arm = wave['arm']
    arm_cfg = cfg['arms'][arm] if not a.mock else dict(adapter=wave['adapter'], facts_version=wave['facts_version'])
    check('wave.json arm/adapter/facts version match exam9_config.json',
          wave.get('adapter') == arm_cfg['adapter'] and wave.get('facts_version') == arm_cfg['facts_version'],
          f"arm {arm} adapter {wave.get('adapter')} facts {wave.get('facts_version')}")
    fv = arm_cfg['facts_version']
    adapter = arm_cfg['adapter']
    exp = Path(cfg['exp'])

    # --- freeze list
    freeze = {}
    if not a.no_freeze:
        fl = Path(wave.get('freeze_list') or cfg['freeze_list'])
        if check('freeze list exists', fl.exists(), str(fl)):
            freeze = L.parse_freeze(fl)
            if wave.get('freeze_sha256'):
                check('freeze list unchanged since the wave started', L.sha256_file(fl) == wave['freeze_sha256'])

    def frozen(name):
        return freeze.get(str(exp / name))

    # --- broker batch registration and prereg sha
    reg = L.read_json(live / wave['broker_batch'] / 'registration.json')
    if check('broker batch registration.json exists', reg is not None, str(live / wave['broker_batch'])):
        check('registration adapter', reg.get('adapter') == adapter, reg.get('adapter'))
        check('registration facts_version and adapter_facts_version',
              reg.get('facts_version') == fv and reg.get('adapter_facts_version', fv) == fv,
              f"{reg.get('facts_version')} / {reg.get('adapter_facts_version')}")
        check('registration hands string', reg.get('hands') == cfg['hands'], repr(reg.get('hands')))
    ps = live / wave['broker_batch'] / 'prereg.sha256'
    if not a.mock:
        pre = exp / cfg['prereg']
        frozen_copy = Path(cfg['prereg_frozen'])
        if check('prereg.sha256 written at wave start', ps.exists(), str(ps)):
            rec = L.parse_freeze(ps)
            same = rec.get(str(pre)) == L.sha256_file(pre)
            check('PREREG unchanged since the wave started', same,
                  '' if same else 'text appended during the wave (allowed if append-only; see the next line)', severity='warn')
        if not a.no_freeze and check('frozen PREREG copy exists', frozen_copy.exists(), str(frozen_copy)):
            check('PREREG = frozen copy + appended text only', pre.read_bytes().startswith(frozen_copy.read_bytes()),
                  f'{pre} vs {frozen_copy}')

    # --- games
    minutes = wave.get('minutes', cfg['minutes'])  # a number, or {mission: minutes} (mock descriptors of old batches)
    games, code_sets, broker_pids = [], set(), set()
    status_counts = collections.Counter()
    # classify every game of the wave first: the whole-wave stop rule (exam9_lib.refine_wave) needs all of them
    evidence = L.broker_death_evidence(live, wave['broker_batch'])
    classes = {}
    for mission in L.MISSIONS:
        for seed in wave['seeds'].get(mission) or []:
            if wave['out_batches'].get(mission):
                classes[(mission, seed)] = L.classify(live / wave['out_batches'][mission] / f's{seed}', evidence=evidence)
    L.refine_wave(list(classes.values()))
    for mission in L.MISSIONS:
        ob = wave['out_batches'].get(mission)
        seeds = wave['seeds'].get(mission) or []
        if not ob:
            continue
        on_disk = sorted(int(p.name[1:]) for p in (live / ob).glob('s*') if p.is_dir() and p.name[1:].isdigit())
        check(f'{ob}: game directories = wave seeds', on_disk == sorted(seeds),
              f'on disk {on_disk} expected {sorted(seeds)}' if on_disk != sorted(seeds) else f'{len(seeds)} games')
        for seed in seeds:
            d = live / ob / f's{seed}'
            c = classes[(mission, seed)]
            g = dict(game=c['game'], seed=seed, mission=mission, status=c['status'], cls=c['cls'], kind=c['kind'],
                     why=c['why'], y=c['y'], dead=c['dead'])
            games.append(g)
            status_counts[c['status']] += 1
            pre = f'{ob}/s{seed}'
            if c['status'] == 'missing':
                check(f'{pre} started.json present', (d / 'started.json').exists(), 'runner never started')
                check(f'{pre} result.json', False, c['why'] + '; class A -> make-up wave (4.5)', severity='warn')
                if mission == 'butcher':
                    g['boundary'], _, _ = L.boundary(None, list(L.iter_rows(d)))
                continue
            res = c['result']
            st = L.read_json(d / 'started.json') or {}
            wall_cap = minutes[mission] if isinstance(minutes, dict) else minutes
            ok = (st.get('seed') == seed and st.get('mission') == mission and st.get('tag') == ob
                  and st.get('tick_limit') == wave['ticks'][mission] and st.get('decision_limit') == cfg['decision_limit']
                  and float(st.get('minutes', -1)) == float(wall_cap))
            check(f'{pre} started seed/mission/tag/limits', ok,
                  '' if ok else f"{st.get('seed')} {st.get('mission')} {st.get('tag')} {st.get('tick_limit')} {st.get('decision_limit')} {st.get('minutes')}")
            ex = st.get('executor') or {}
            ok = (ex.get('kind') == 'r3' and ex.get('chain') is True and ex.get('target_rule') is True and ex.get('hold_floor') == 0.5
                  and ex.get('chain_guard') == 'post_hit' and ex.get('chase_last_seen') is True
                  and (st.get('undo_guard') or {}).get('on') is True and st.get('auto_belt') is True and st.get('allow_heldout') is True)
            check(f'{pre} hands (r3 chain target hold0.5 post_hit chase, undo guard, auto belt, allow_heldout)', ok,
                  '' if ok else json.dumps(dict(executor=ex, undo=st.get('undo_guard'), belt=st.get('auto_belt'), heldout=st.get('allow_heldout')))[:300])
            fa = st.get('facts') or {}
            check(f'{pre} facts on, version {fv}', fa.get('on') is True and fa.get('version') == fv, f"{fa.get('version')}")
            check(f'{pre} menu r9', st.get('menu') == 'r9' and (st.get('menu_rules') or {}).get('no_sell_bought_this_visit') is True,
                  f"menu {st.get('menu')}")
            code = st.get('code')
            if check(f'{pre} code shas recorded', isinstance(code, dict) and code, ''):
                code_sets.add(json.dumps(code, sort_keys=True))
                if freeze:
                    bad = [f for f, s in code.items() if frozen(f) != s]
                    check(f'{pre} code shas = freeze list', not bad, f'differ or not frozen: {bad}' if bad else '')
            if freeze:
                bad = []
                if ex.get('r3_sha256') != frozen('game_executor_r3.py'):
                    bad.append('game_executor_r3.py')
                if ex.get('r2_sha256') != freeze.get(str(exp.parent / 'strategy-brain-sft-20260922' / 'game_executor_r2.py')):
                    bad.append('game_executor_r2.py')
                if (st.get('undo_guard') or {}).get('sha256') != frozen('undo_guard.py'):
                    bad.append('undo_guard.py')
                if fa.get('sha256') != frozen('facts.py'):
                    bad.append('facts.py')
                if fv >= 4 and fa.get('engine_numbers_sha256') != frozen('engine_numbers.py'):
                    bad.append('engine_numbers.py')
                check(f'{pre} executor/undo_guard/facts shas = freeze list', not bad, str(bad) if bad else '')
            br = st.get('broker') or {}
            check(f'{pre} broker adapter/model', br.get('adapter') == adapter and br.get('model') == cfg['broker_model'],
                  f"{br.get('adapter')} {br.get('model')}")
            if br.get('pid') is not None:
                broker_pids.add(br.get('pid'))
            nat = (L.read_json(d / 'native' / 'started.json') or {}).get('bridge') or {}
            ok = nat.get('bridge_sha256') == cfg['bridge_sha256'] and nat.get('engine_sha256') == cfg['engine_sha256']
            if freeze:
                ok = ok and freeze.get(cfg['bridge_path']) == nat.get('bridge_sha256') and freeze.get(cfg['engine_path']) == nat.get('engine_sha256')
            check(f'{pre} native bridge/engine sha', ok, f"{str(nat.get('bridge_sha256'))[:12]} {str(nat.get('engine_sha256'))[:12]}")
            rows = list(L.iter_rows(d))
            nrow = len(rows)
            ok = nrow == res.get('decisions') or (nrow == res.get('decisions', 0) + 1 and rows and rows[-1].get('result') == 'native_tick_budget')
            check(f'{pre} decisions.jsonl rows = result decisions', ok, f'{nrow} rows, result {res.get("decisions")}', severity='warn')
            m = L.game_metrics(rows, c['status'], cfg, mission)
            check(f'{pre} no buy-then-sell in one town visit (exact, menu_bought)', not m['buy_sell_exact'],
                  (('CODE DEFECT (4.4 3c) ' + str(m['buy_sell_exact'][:3])) if m['buy_sell_exact'] else
                   ('' if m['has_menu_bought'] else 'rows have no menu_bought (not a round-9 menu log)')), severity='warn')
            g.update(decisions=res.get('decisions'), seconds=res.get('seconds'), start=st.get('time'), cycle=m['cycle'],
                     think_mean=m['think_mean'], think_n=m['think_n'], ticks=(res.get('final') or {}).get('tick'))
            if mission == 'butcher':
                v, by_ms, by_rows = L.boundary(res, rows)
                g['boundary'] = v
                if by_rows != by_ms:
                    check(f'{pre} boundary: milestones and decision rows agree', False, f'milestones {by_ms}, rows {by_rows}', severity='warn')
    if code_sets:
        check('one code set across the wave', len(code_sets) == 1, f'{len(code_sets)} distinct')
    # --- watchdog restarts, broker log
    wr = live / wave['broker_batch'] / 'watchdog-restarts.jsonl'
    restarts = [json.loads(x) for x in wr.read_text().splitlines() if x.strip()] if wr.exists() else []
    check('broker watchdog restarts', True, f'{len(restarts)} restart(s)' + (f': {restarts}' if restarts else ''), severity='warn')
    if len(broker_pids) > 1:
        check('games saw one broker pid at start', False, f'{sorted(broker_pids)} (a restart before some games started?)', severity='warn')
    # --- summary
    cls = collections.Counter(g['cls'] for g in games)
    print(f"{tag}SUMMARY {a.batch}: {len(games)} games; status {dict(status_counts)}; classes {dict(cls)}")
    for g in games:
        if g['cls'] != 'ok':
            print(f"{tag}  {g['cls']:<20} {g['game']}: {g['why']}")
    for g in games:
        if g.get('boundary'):
            print(f"{tag}  boundary (Butcher mission on level 3+/tomb): {g['game']} status {g['status']}")
    out = dict(mock=a.mock, no_freeze=a.no_freeze, batch=a.batch, wave=wave, time=time.time(), checks=checks, games=games,
               status_counts=dict(status_counts), classes=dict(cls), watchdog_restarts=restarts)
    rc = 0
    if any(not c['ok'] and c['severity'] == 'integrity' for c in checks):
        rc = 1
    elif cls.get('A', 0) >= cfg['check']['class_a_stop_chain_at']:
        print(f"{tag}STOP-CHAIN: {cls['A']} class-A failures (>= {cfg['check']['class_a_stop_chain_at']}): infrastructure suspect")
        rc = 3
    if a.gate:
        out['gate'] = gate(cfg, games, code_mismatch=rc == 1, tag=tag)
        if out['gate']['pause'] and rc == 0:
            rc = 4
    out['exit'] = rc
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1, default=str), encoding='utf-8')
    print(f'{tag}EXIT {rc}')
    sys.exit(rc)


def gate(cfg, games, code_mismatch, tag=''):
    """Decision 7: pause criteria of the training-seed gate (new arm, one wave)."""
    gc = cfg['gate']
    reasons = []
    deaths = sum(g['dead'] for g in games)
    if deaths >= gc['deaths_pause_at']:
        reasons.append(f'deaths {deaths} >= {gc["deaths_pause_at"]}')
    exc = [g['game'] for g in games if g['cls'] == 'B' and g['kind'] != 'no_progress']
    if exc:
        reasons.append(f'class-B exception(s): {exc}')
    known = set(gc['known_cycles'])  # the literal round-8 list (PREREG 6b); the recomputation below is information only
    r8 = L.round8_cycle_types(cfg)
    if set(r8) != known:
        print(f"{tag}GATE note: round-8 logs now give cycle signatures {sorted(r8)}, config known_cycles {sorted(known)} (the config list is used)")
    new_cycles = sorted({g['cycle'] for g in games if g.get('cycle') and g['cycle'] not in known})
    if new_cycles:
        reasons.append(f'cycle type(s) not in known_cycles (round 8): {new_cycles}')
    npc = collections.Counter(g['status'] for g in games if str(g['status']).startswith('no_progress'))
    rep = {k: v for k, v in npc.items() if v >= gc['same_no_progress_pause_at']}
    if rep:
        reasons.append(f'same no_progress type in >= {gc["same_no_progress_pause_at"]} games: {rep}')
    if code_mismatch:
        reasons.append('integrity / code sha mismatch (see FAIL lines)')
    # e (PREREG 6e): queue model. Per-decision latency at 16 games = concurrency x mean think + the game's own time;
    # own time per game = seconds/decisions minus (mean alive games during its run) x mean think, floored at 0;
    # projected minutes = decisions x (concurrency x mean think + own) / 60; pause when the largest > wall_fraction x cap.
    done = [g for g in games if g.get('decisions') and g.get('seconds') and g.get('start')]
    limit = gc['wall_fraction'] * cfg['minutes']
    think = [(g['think_mean'], g['think_n']) for g in done if g.get('think_mean') is not None]
    proj = None
    if done and think:
        tbar = sum(t * n for t, n in think) / sum(n for _, n in think)
        spans = [(g['start'], g['start'] + g['seconds']) for g in done]
        rows = []
        for g in done:
            s0, s1 = g['start'], g['start'] + g['seconds']
            grid = [s0 + (s1 - s0) * (i + 0.5) / 200 for i in range(200)]
            alive = sum(sum(1 for a, b in spans if a <= t < b) for t in grid) / len(grid)
            own = max(0.0, g['seconds'] / g['decisions'] - alive * tbar)
            rows.append(dict(game=g['game'], decisions=g['decisions'], observed_min=round(g['seconds'] / 60, 1), mean_alive=round(alive, 2),
                             own_s_per_decision=round(own, 3), projected_min=round(g['decisions'] * (gc['concurrency'] * tbar + own) / 60, 1)))
        worst = max(rows, key=lambda r: r['projected_min'])
        proj = dict(mean_think_s=round(tbar, 4), concurrency=gc['concurrency'], longest=worst, limit_min=limit,
                    observed_longest_min=round(max(g['seconds'] for g in done) / 60, 1),
                    cap_6000_decisions_min=round(6000 * (gc['concurrency'] * tbar + sorted(r['own_s_per_decision'] for r in rows)[len(rows) // 2]) / 60, 1),
                    per_game=rows)
        if worst['projected_min'] > limit:
            reasons.append(f"wall rule e: {worst['game']} projects {worst['projected_min']:.0f} min > {limit:.0f} min at "
                           f"{gc['concurrency']} games ({worst['decisions']} decisions, mean think {tbar:.3f} s, own {worst['own_s_per_decision']} s)")
    else:
        reasons.append('wall rule e not computable (no finished game with think times)')
    # the draft's formula (median s/decision x 16/19 x largest decisions), printed for comparison only
    old = None
    if done:
        per = sorted(g['seconds'] / g['decisions'] for g in done)
        med = per[len(per) // 2] if len(per) % 2 else (per[len(per) // 2 - 1] + per[len(per) // 2]) / 2
        old = round(med * gc['concurrency'] / len(games) * max(g['decisions'] for g in done) / 60, 1)
    print(f"{tag}GATE deaths {deaths}; class-B exceptions {len(exc)}; cycles {sorted({g['cycle'] for g in games if g.get('cycle')})}; "
          f"known cycles (config) {sorted(known)}")
    print(f"{tag}GATE wall rule e (queue model): longest {None if proj is None else proj['longest']}, limit {limit:.0f} min; "
          f"6000-decision game {None if proj is None else proj['cap_6000_decisions_min']} min; draft formula (info) {old} min")
    print(f"{tag}GATE {'PAUSE: ' + '; '.join(reasons) if reasons else 'PASS'}")
    return dict(pause=bool(reasons), reasons=reasons, deaths=deaths, known_cycles=sorted(known), round8_cycles_recomputed=r8,
                queue_model=proj, draft_formula_min=old)


if __name__ == '__main__':
    main()

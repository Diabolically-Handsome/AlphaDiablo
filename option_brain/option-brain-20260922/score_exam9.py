"""Score the round-9 paired final exam (PREREG-ROUND9.md sections 4 and 8; CPU, reads the logs, writes only --out-json /
--out-md). Written 2026-09-23; in the freeze list; not changed until the exam is over.

usage (WSL):
  python3 -B score_exam9.py --exam [--final] --out-json F.json --out-md F.md
      the exam: units from exam9_config.json (32 King + 32 Butcher games): primary game
      live/exam9-w<wave>-<arm>-<king|butcher>/s<seed> (wave by the half the seed is in, ABBA; the latest re-run
      exam9-w<wave>r<k>-... when a wave was voided under 3.3), make-up game live/exam9-m1-<arm>-<king|butcher>/s<seed>
      (or its latest whole re-run exam9-m1r<k>-...) for a class-A primary. Classes are decided per wave
      (exam9_lib.classify + refine_wave). Every broker batch whose games are scored must have a counted wave check
      (live/<batch>/check.json with exit 0 or 3, exam9_lib.check_status); otherwise the report is marked INCOMPLETE at
      its top, whatever --final says.
  python3 -B score_exam9.py --manifest M.json --out-json F.json --out-md F.md
      any other pairing (the mock dry run on round-8 logs): M = {"label", "mock": true, "units": [{"arm", "mission",
      "seed", "primary": "<out-batch>/s<seed>", "makeups": [...]}], "extra": [{"arm", "mission", "game"}],
      "king_seeds_with_butcher_quest": [...]}. 'extra' games enter only the per-arm descriptive totals.

Outcome (ITT, decision 3): Y = 1 iff status 'success'; everything else 0. A class-A primary (exam9_lib.classify) is
replaced by its make-up game; a class-A make-up (second failure) counts 0 and its pair is dropped in sensitivity
analysis S1; a missing make-up leaves the unit 'pending' (counted 0 and listed; the report is marked INCOMPLETE
unless --final is given). Operator stops (with or without stop-reason.txt, unless the whole-wave rule made them class
A) count 0; S2 drops their pairs; S3 drops both and pending pairs. There is no manual re-classification file.
Confirmatory test (decision 5): exact sign-flip test over the distinct seeds, D_seed = sum over its missions of
(Y_new - Y_old), one-sided (new better), alpha 0.05. Descriptive: per-mission 2x2, McNemar exact two-sided p,
Newcombe (method 10) and Tango score 95% CIs of the paired difference, Wilson CIs per arm.
Adoption rule (decision 4): (1) K_new >= K_old combined and K_new - K_old >= -1 per mission; (2) D_new <= D_old
combined; (3a) Butcher-mission boundary violations new <= 1 and (new < old or new == 0); (3b) retreat metric >= 80% (fewer than 20
qualifying new-arm decisions: not evaluable, does not block); (3c) exact buy-then-sell in one town visit = code
defect (integrity flag, not a model metric). Significance changes nothing.
"""
import argparse
import collections
import json
import math
import statistics
import sys
import time
from pathlib import Path

import exam9_lib as L

Z = 1.959963984540054


# ---------------------------------------------------------------- statistics
def wilson(k, n, z=Z):
    if n == 0:
        return None
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, c - h), min(1.0, c + h)


def binom_cdf_half(k, n):
    return sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * binom_cdf_half(min(b, c), n))


def newcombe_paired(a, b, c, d, z=Z):
    """Newcombe (1998) method 10 (hybrid score, Wilson margins); phi with AD-BC replaced by max(AD-BC-N/2, 0) when
    AD > BC, phi = 0 when a margin is 0. Difference p_new - p_old with b = new-only, c = old-only."""
    n = a + b + c + d
    if n == 0:
        return None
    p1, p2 = (a + b) / n, (a + c) / n
    l1, u1 = wilson(a + b, n, z)
    l2, u2 = wilson(a + c, n, z)
    num = a * d - b * c
    den = math.sqrt((a + b) * (c + d) * (a + c) * (b + d))
    if den == 0:
        phi = 0.0
    else:
        if num > 0:
            num = max(num - n / 2, 0)
        phi = num / den
    th = p1 - p2
    dl = math.sqrt(max(0.0, (p1 - l1) ** 2 - 2 * phi * (p1 - l1) * (u2 - p2) + (u2 - p2) ** 2))
    du = math.sqrt(max(0.0, (u1 - p1) ** 2 - 2 * phi * (u1 - p1) * (p2 - l2) + (p2 - l2) ** 2))
    return max(-1.0, th - dl), min(1.0, th + du)


def tango(b, c, n, z=Z):
    """Tango (1998) score CI for p_new - p_old in a paired design (b = new-only, c = old-only, n pairs)."""
    if n == 0:
        return None

    def score(D):
        W = -b - c + (2 * n - b + c) * D
        q = (-W + math.sqrt(max(0.0, W * W + 8 * n * c * D * (1 - D)))) / (4 * n)
        var = n * (2 * q + D * (1 - D))
        num = b - c - n * D
        if var <= 1e-15:
            return 0.0 if abs(num) < 1e-12 else math.copysign(math.inf, num)
        return num / math.sqrt(var)

    th = (b - c) / n
    eps = 1e-9

    def solve(lo, hi, target):  # score is decreasing in D
        if (score(lo) - target) * (score(hi) - target) > 0:
            return None
        for _ in range(200):
            mid = (lo + hi) / 2
            if (score(mid) - target) > 0:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2

    lo = -1.0 if score(-1 + eps) <= z else solve(-1 + eps, th, z)
    hi = 1.0 if score(1 - eps) >= -z else solve(th, 1 - eps, -z)
    return lo, hi


def signflip(ds):
    """Exact sign-flip distribution of sum(D) with each nonzero D flipped with probability 1/2."""
    nz = [abs(d) for d in ds if d != 0]
    dist = {0: 1}
    for v in nz:
        nd = collections.Counter()
        for s, k in dist.items():
            nd[s + v] += k
            nd[s - v] += k
        dist = nd
    tot = 2 ** len(nz)
    t = sum(ds)
    return dict(T=t, nonzero=len(nz), p_one_sided=sum(k for s, k in dist.items() if s >= t) / tot,
                p_two_sided=sum(k for s, k in dist.items() if abs(s) >= abs(t)) / tot)


def fmt_ci(ci, pct=True):
    if ci is None:
        return '—'
    f = (lambda x: f'{100 * x:.0f}%') if pct else (lambda x: f'{x:+.2f}')
    return f'{f(ci[0])} 至 {f(ci[1])}'


def pp(ci):
    return '—' if ci is None else f'{100 * ci[0]:+.0f} 至 {100 * ci[1]:+.0f} 个百分点'


# ---------------------------------------------------------------- units
def exam_units(cfg):
    live = Path(cfg['root']) / 'live'
    units, voided = [], set()
    for mission in L.MISSIONS:
        seeds = cfg['king_seeds'] if mission == 'skeleton_king' else cfg['butcher_seeds']
        for seed in seeds:
            half = 'A' if seed in cfg['halves']['A'][mission] else 'B'
            assert seed in cfg['halves'][half][mission]
            for arm in ('new', 'old'):
                ob, vo = L.primary_batch(cfg, live, arm, mission, seed)
                voided.update(vo)
                use, mvoid, mextra = L.makeup_dirs(live, arm, mission, seed)
                rel = lambda p: f'{p.parent.name}/{p.name}'
                voided.update(p.parent.name for p in mvoid)
                units.append(dict(arm=arm, mission=mission, seed=seed, primary=f'{ob}/s{seed}',
                                  makeups=[rel(use)] if use else [], makeups_extra=[rel(p) for p in mextra]))
    return dict(label='第 9 轮终考', mock=False, units=units, extra=[], voided_waves=sorted(voided),
                king_seeds_with_butcher_quest=cfg['king_seeds_with_butcher_quest'],
                butcher_seeds_with_king_quest=cfg['butcher_seeds_with_king_quest'])


class Classifier:
    """exam9_lib.classify per game, decided per wave: every game of a broker batch is classified with that batch's
    broker-death evidence, then refine_wave applies the whole-wave stop rule (PREREG 4.5)."""

    def __init__(self, live):
        self.live, self.cache, self.done, self.evidence = Path(live), {}, set(), {}

    def __call__(self, d):
        d = Path(d)
        bb = L.broker_batch_of(d.parent)
        if bb not in self.done:
            ev = self.evidence[bb] = L.broker_death_evidence(self.live, bb)
            res = [L.classify(x, ev) for x in L.wave_game_dirs(self.live, bb)]
            L.refine_wave(res)
            self.cache.update((r['dir'], r) for r in res)
            self.done.add(bb)
        if str(d) not in self.cache:
            self.cache[str(d)] = L.classify(d, self.evidence.get(bb))
        return self.cache[str(d)]


def prefix_match(d1, d2):
    """How many of the original (class-A) game's logged decisions the make-up game reproduces, row by row
    (n, tick, scene, chosen letter, option text)."""
    keys = ('n', 'tick', 'scene', 'choice', 'text')
    rows1 = [[r.get(x) for x in keys] for r in L.iter_rows(d1)]
    k = 0
    for r1, r2 in zip(rows1, L.iter_rows(d2)):
        if r1 != [r2.get(x) for x in keys]:
            break
        k += 1
    return dict(matched=k, original_rows=len(rows1), reproduced=k == len(rows1))


def resolve(u, cfg, live, classify):
    prim = classify(live / u['primary'])
    used, flags = prim, dict(a_rerun=False, a2=False, pending=False, op_stop=False,
                             extra_makeups=len(u.get('makeups') or []) > 1 or bool(u.get('makeups_extra')))
    makeup_info = None
    if prim['cls'] == 'A':
        if u.get('makeups'):
            mk = classify(live / u['makeups'][0])
            flags['a_rerun'] = True
            makeup_info = dict(game=mk['game'], status=mk['status'], cls=mk['cls'], why=mk['why'],
                               prefix=prefix_match(live / u['primary'], live / u['makeups'][0]))
            used = mk
            if mk['cls'] == 'A':
                flags['a2'] = True
        else:
            flags['pending'] = True
    if used['cls'] in ('op_stop', 'op_stop_unexplained'):
        flags['op_stop'] = True
    rows = list(L.iter_rows(Path(used['dir'])))
    res = used.get('result') or {}
    m = L.game_metrics(rows, used['status'], cfg, u['mission'])
    bnd = bnd_rows = None
    if u['mission'] == 'butcher':
        # without result.json (pending or second failure) the decision rows define the violation (appendix)
        bnd, _, bnd_rows = L.boundary(used.get('result'), rows)
    fin = res.get('final') or {}
    y = 0 if (flags['pending'] or flags['a2']) else used['y']
    return dict(arm=u['arm'], mission=u['mission'], seed=u['seed'], primary=prim['game'], primary_status=prim['status'],
                primary_cls=prim['cls'], primary_why=prim['why'], used=used['game'], status=used['status'], cls=used['cls'],
                why=used['why'], y=y, dead=bool(used['dead']) and not flags['pending'], flags=flags, makeup=makeup_info,
                boundary=bnd, boundary_rows=bnd_rows, king_kills=fin.get('king_kills'), butcher_kills=(fin.get('butcher') or {}).get('kills'),
                kill_tick=fin.get('tick') if used['status'] == 'success' else None, end_tick=fin.get('tick'),
                broker_batch=used.get('broker_batch'), out_batch=Path(used['dir']).parent.name, seconds=res.get('seconds'),
                decisions=res.get('decisions'),
                minutes=round(res['seconds'] / 60, 1) if res.get('seconds') else None, level=fin.get('level'),
                deepest=m['max_level'], metrics=m)


# ---------------------------------------------------------------- analysis
def analyse(units, cfg, king_bq, butcher_kq=frozenset(), exclude=None):
    """Paired analysis over the units; exclude(pair) -> True drops a (mission, seed) pair."""
    by = {(u['mission'], u['seed'], u['arm']): u for u in units}
    pairs = []
    for (mission, seed, arm), u in by.items():
        if arm != 'new':
            continue
        o = by.get((mission, seed, 'old'))
        if o is None:
            continue
        if exclude and exclude(u, o):
            continue
        pairs.append((mission, seed, u, o))
    out = dict(missions={}, n_pairs=len(pairs))
    for mission in L.MISSIONS:
        ps = [p for p in pairs if p[0] == mission]
        a = sum(1 for p in ps if p[2]['y'] and p[3]['y'])
        b = sum(1 for p in ps if p[2]['y'] and not p[3]['y'])
        c = sum(1 for p in ps if not p[2]['y'] and p[3]['y'])
        d = sum(1 for p in ps if not p[2]['y'] and not p[3]['y'])
        n = len(ps)
        da = sum(1 for p in ps if p[2]['dead'] and p[3]['dead'])
        db = sum(1 for p in ps if p[2]['dead'] and not p[3]['dead'])
        dc = sum(1 for p in ps if not p[2]['dead'] and p[3]['dead'])
        dd = n - da - db - dc
        out['missions'][mission] = dict(
            n=n, both=a, new_only=b, old_only=c, neither=d, K_new=a + b, K_old=a + c,
            D_new=sum(p[2]['dead'] for p in ps), D_old=sum(p[3]['dead'] for p in ps),
            mcnemar_p_two_sided=mcnemar_exact(b, c), newcombe_ci=newcombe_paired(a, b, c, d), tango_ci=tango(b, c, n),
            wilson_new=wilson(a + b, n), wilson_old=wilson(a + c, n), diff=(b - c) / n if n else None,
            deaths=dict(both=da, new_only=db, old_only=dc, neither=dd, mcnemar_p_two_sided=mcnemar_exact(db, dc),
                        newcombe_ci=newcombe_paired(da, db, dc, dd), tango_ci=tango(db, dc, n),
                        wilson_new=wilson(da + db, n), wilson_old=wilson(da + dc, n)))
    ms = out['missions']
    out['K_new'] = sum(v['K_new'] for v in ms.values())
    out['K_old'] = sum(v['K_old'] for v in ms.values())
    out['D_new'] = sum(v['D_new'] for v in ms.values())
    out['D_old'] = sum(v['D_old'] for v in ms.values())
    seeds = collections.defaultdict(int)
    for mission, seed, u, o in pairs:
        seeds[seed] += u['y'] - o['y']
    ds = [seeds[s] for s in sorted(seeds)]
    out['seed_level'] = dict(signflip(ds), n_seeds=len(ds), seeds_new_better=sum(1 for x in ds if x > 0),
                             seeds_old_better=sum(1 for x in ds if x < 0), seeds_tied=sum(1 for x in ds if x == 0),
                             D_by_seed={str(s): seeds[s] for s in sorted(seeds)})
    # boundary (Butcher mission) over the pairs kept
    bu = [p for p in pairs if p[0] == 'butcher']
    out['V_new'] = sum(bool(p[2]['boundary']) for p in bu)
    out['V_old'] = sum(bool(p[3]['boundary']) for p in bu)
    # strata (PREREG section 2): King seeds with / without the Butcher quest; Butcher seeds with / without the King quest
    st = {}
    for name, mission, cond in (('骷髅王组·同时有屠夫任务', 'skeleton_king', lambda s: s in king_bq),
                                ('骷髅王组·没有屠夫任务', 'skeleton_king', lambda s: s not in king_bq),
                                ('屠夫组·同时有骷髅王任务', 'butcher', lambda s: s in butcher_kq),
                                ('屠夫组·没有骷髅王任务', 'butcher', lambda s: s not in butcher_kq)):
        ps = [p for p in pairs if p[0] == mission and cond(p[1])]
        n = len(ps)
        a_ = sum(1 for p in ps if p[2]['y'] and p[3]['y'])
        b = sum(1 for p in ps if p[2]['y'] and not p[3]['y'])
        c = sum(1 for p in ps if not p[2]['y'] and p[3]['y'])
        st[name] = dict(n=n, K_new=sum(p[2]['y'] for p in ps), K_old=sum(p[3]['y'] for p in ps), new_only=b, old_only=c,
                        mcnemar_p_two_sided=mcnemar_exact(b, c), tango_ci=tango(b, c, n),
                        newcombe_ci=newcombe_paired(a_, b, c, n - a_ - b - c), wilson_new=wilson(a_ + b, n), wilson_old=wilson(a_ + c, n),
                        D_new=sum(p[2]['dead'] for p in ps), D_old=sum(p[3]['dead'] for p in ps),
                        V_new=sum(bool(p[2]['boundary']) for p in ps) if mission == 'butcher' else None,
                        V_old=sum(bool(p[3]['boundary']) for p in ps) if mission == 'butcher' else None)
    out['strata'] = st
    return out


def retreat(units, arm, cfg):
    us = [u for u in units if u['arm'] == arm]
    den = sum(u['metrics']['retreat_den'] for u in us)
    num = sum(u['metrics']['retreat_num'] for u in us)
    per = [u['metrics']['retreat_num'] / u['metrics']['retreat_den'] for u in us if u['metrics']['retreat_den']]
    return dict(den=den, num=num, pooled=num / den if den else None, per_game_mean=sum(per) / len(per) if per else None,
                games_with_qualifying=len(per), evaluable=den >= cfg['retreat']['min_decisions'])


def adoption(an, rt_new, rt_old, cfg):
    ms = an['missions']
    c1_parts = {m: ms[m]['K_new'] - ms[m]['K_old'] for m in L.MISSIONS}
    c1 = an['K_new'] >= an['K_old'] and all(v >= -1 for v in c1_parts.values())
    c2 = an['D_new'] <= an['D_old']
    c3a = an['V_new'] <= 1 and (an['V_new'] < an['V_old'] or an['V_new'] == 0)
    if rt_new['evaluable']:
        c3b = rt_new['pooled'] >= cfg['retreat']['threshold']
    else:
        c3b = None
    c3 = c3a and (c3b is not False)
    if c1 and c2 and c3:
        verdict = '采纳'
    elif c1 and c2:
        verdict = '不采纳：成绩持平或更好但行为指标未达标，查原因'
    else:
        verdict = '不采纳'
    return dict(c1=c1, c1_per_mission_diff=c1_parts, c2=c2, c3a=c3a, c3b=c3b, c3=c3, verdict=verdict,
                c3a_note='两臂越界都为 0，按 18:10 批准的写法算满足' if an['V_old'] == 0 and an['V_new'] == 0 else None)


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument('--exam', action='store_true')
    g.add_argument('--manifest')
    ap.add_argument('--final', action='store_true', help='count pending (make-up not run) units as 0 and do not mark INCOMPLETE')
    ap.add_argument('--out-json', required=True)
    ap.add_argument('--out-md', required=True)
    ap.add_argument('--config', default=None)
    a = ap.parse_args()
    cfg = L.load_config(a.config)
    live = Path(cfg['root']) / 'live'
    man = exam_units(cfg) if a.exam else json.loads(Path(a.manifest).read_text(encoding='utf-8'))
    t0 = time.time()
    classify = Classifier(live)
    units = [resolve(u, cfg, live, classify) for u in man['units']]
    # every scored game's broker batch needs a counted wave check (exam only; PREREG 3.3)
    wave_checks = {}
    if a.exam:
        for u in units:
            for g in [u['used']] + ([u['primary']] if u['primary'] != u['used'] else []):
                ob = g.split('/')[0]
                if not (live / ob).is_dir():
                    continue
                bb = L.broker_batch_of(live / ob)
                if bb not in wave_checks:
                    ok, ex, why = L.check_status(live, bb)
                    wave_checks[bb] = dict(counted=ok, exit=ex, why=why)
    unchecked = sorted(bb for bb, v in wave_checks.items() if not v['counted'])
    extra = []
    for e in man.get('extra') or []:
        c = classify(live / e['game'])
        extra.append(dict(arm=e['arm'], mission=e['mission'], game=c['game'], status=c['status'], y=c['y'], dead=c['dead']))
    king_bq = set(man.get('king_seeds_with_butcher_quest') or [])
    butcher_kq = set(man.get('butcher_seeds_with_king_quest') or [])
    itt = analyse(units, cfg, king_bq, butcher_kq)
    rt = {arm: retreat(units, arm, cfg) for arm in ('new', 'old')}
    adopt = adoption(itt, rt['new'], rt['old'], cfg)
    sens = {}
    for name, fn in (('S1 剔除 A 类二次失败的对', lambda u, o: u['flags']['a2'] or o['flags']['a2']),
                     ('S2 剔除有人工停局的对', lambda u, o: u['flags']['op_stop'] or o['flags']['op_stop']),
                     ('S3 剔除两者及待补跑的对', lambda u, o: any(x['flags'][k] for x in (u, o) for k in ('a2', 'op_stop', 'pending')))):
        an = analyse(units, cfg, king_bq, butcher_kq, exclude=fn)
        ad = adoption(an, rt['new'], rt['old'], cfg)
        sens[name] = dict(n_pairs=an['n_pairs'], K_new=an['K_new'], K_old=an['K_old'], D_new=an['D_new'], D_old=an['D_old'],
                          V_new=an['V_new'], V_old=an['V_old'], p_one_sided=an['seed_level']['p_one_sided'], verdict=ad['verdict'],
                          differs=ad['verdict'] != adopt['verdict'] or ((an['seed_level']['p_one_sided'] <= 0.05) != (itt['seed_level']['p_one_sided'] <= 0.05)))
    integrity = dict(
        buy_sell_exact=[dict(game=u['used'], rows=u['metrics']['buy_sell_exact']) for u in units if u['metrics']['buy_sell_exact']],
        menu_bought_logged=all(u['metrics']['has_menu_bought'] or not u['metrics']['decisions'] for u in units),
        pending=[f"{u['arm']} {u['mission']} {u['seed']}" for u in units if u['flags']['pending']],
        extra_makeups=[f"{u['arm']} {u['mission']} {u['seed']}" for u in units if u['flags']['extra_makeups']],
        boundary_rows_disagree=[u['used'] for u in units if u['mission'] == 'butcher' and u['boundary_rows'] is not None
                                and bool(u['boundary_rows']) != bool(u['boundary'])],
        voided_waves=man.get('voided_waves') or [],
        wave_checks=wave_checks, waves_unchecked=unchecked,
        not_run=sorted({u['primary'].split('/')[0] for u in units if not (live / u['primary'].split('/')[0]).is_dir()}) if a.exam else [],
        broker_restarts={bb: ev for bb, ev in classify.evidence.items() if ev['restarts'] or ev['gave_up'] or ev['broker_traceback']},
        units=len(units), expected_units=64 if a.exam else len(man['units']))
    # descriptive per arm
    desc = {}
    for arm in ('new', 'old'):
        us = [u for u in units if u['arm'] == arm]
        ex = [e for e in extra if e['arm'] == arm]
        dm = {}
        for mission in L.MISSIONS:
            um = [u for u in us if u['mission'] == mission]
            em = [e for e in ex if e['mission'] == mission]
            dm[mission] = dict(games=len(um) + len(em), kills=sum(u['y'] for u in um) + sum(e['y'] for e in em),
                               deaths=sum(u['dead'] for u in um) + sum(e['dead'] for e in em),
                               kill_ticks=sorted(u['kill_tick'] for u in um if u['kill_tick'] is not None))
        mh = sum(u['metrics']['menu_hidden_rows'] for u in us)
        uh = sum(u['metrics']['undo_hidden_rows'] for u in us)
        nd = sum(u['metrics']['decisions'] for u in us)
        lp = collections.Counter()
        for u in us:
            lp.update(u['metrics']['low_potion_events'])
        for mission in L.MISSIONS:
            seen = [u['metrics']['boss_first_seen'] for u in us if u['mission'] == mission and u['metrics']['boss_first_seen']]
            med = lambda k: statistics.median([s[k] for s in seen if s.get(k) is not None]) if any(s.get(k) is not None for s in seen) else None
            dm[mission]['boss_first_seen'] = dict(games=len(seen), median_level=med('level'), median_dexterity=med('dexterity'),
                                                  median_potions=med('potions'), median_tick=med('tick'))
            # cumulative kills and deaths by tick (deaths a competing event; games without a kill censored at their end)
            cap = cfg['ticks'][mission]
            um = [u for u in us if u['mission'] == mission]
            dm[mission]['cumulative'] = [dict(tick=t, kills=sum(1 for u in um if u['kill_tick'] is not None and u['kill_tick'] <= t),
                                              deaths=sum(1 for u in um if u['dead'] and (u['end_tick'] or 0) <= t))
                                         for t in [cap * k // 5 for k in range(1, 6)]]
        desc[arm] = dict(missions=dm, statuses=dict(collections.Counter(u['status'] for u in us)),
                         classes=dict(collections.Counter(u['cls'] for u in us)), decisions=nd,
                         menu_hidden_rows=mh, undo_hidden_rows=uh, low_potion_events=dict(lp),
                         town_trips=sum(u['metrics']['town_trips'] for u in us), wasted_trips=sum(u['metrics']['wasted_trips'] for u in us),
                         cross_visit_sell_loss_gold=sum(u['metrics']['cross_visit_loss'] for u in us),
                         wall_minutes_by_batch={b: round(max(u['seconds'] or 0 for u in us if u['out_batch'] == b) / 60, 1)
                                                for b in sorted({u['out_batch'] for u in us})},
                         buy_sell_same_visit_by_name=sum(len(u['metrics']['buy_sell_same_visit_name']) for u in us),
                         buy_sell_cross_visit_by_name=sum(len(u['metrics']['buy_sell_cross_visit_name']) for u in us),
                         cycles=dict(collections.Counter(u['metrics']['cycle'] for u in us if u['metrics']['cycle'])),
                         butcher_killed_in_king_mission=[u['seed'] for u in us if u['mission'] == 'skeleton_king' and (u['butcher_kills'] or 0) > 0],
                         king_killed_in_butcher_mission=[u['seed'] for u in us if u['mission'] == 'butcher' and (u['king_kills'] or 0) > 0])
    both = {}
    for mission in L.MISSIONS:
        diffs = []
        for u in units:
            if u['arm'] == 'new' and u['mission'] == mission and u['kill_tick'] is not None:
                o = next((x for x in units if x['arm'] == 'old' and x['mission'] == mission and x['seed'] == u['seed']), None)
                if o and o['kill_tick'] is not None:
                    diffs.append(u['kill_tick'] - o['kill_tick'])
        both[mission] = dict(n=len(diffs), median_new_minus_old_ticks=statistics.median(diffs) if diffs else None, diffs=diffs)
    out = dict(label=man.get('label'), mock=bool(man.get('mock')), final=a.final, time=time.time(), seconds=round(time.time() - t0, 1),
               itt=itt, retreat=rt, adoption=adopt, sensitivity=sens, integrity=integrity, descriptive=desc,
               kill_time_paired=both, units=[{k: v for k, v in u.items() if k != 'metrics'} | dict(
                   retreat_den=u['metrics']['retreat_den'], retreat_num=u['metrics']['retreat_num'], cycle=u['metrics']['cycle'],
                   menu_hidden_rows=u['metrics']['menu_hidden_rows']) for u in units], extra=extra)
    Path(a.out_json).write_text(json.dumps(out, indent=1, ensure_ascii=False, default=str), encoding='utf-8')
    Path(a.out_md).write_text(markdown(out, cfg), encoding='utf-8')
    print(f"{'MOCK ' if out['mock'] else ''}verdict: {adopt['verdict']} | K {itt['K_new']}/{itt['K_old']} D {itt['D_new']}/{itt['D_old']} "
          f"V {itt['V_new']}/{itt['V_old']} retreat new {rt['new']['num']}/{rt['new']['den']} | sign-flip p1 {itt['seed_level']['p_one_sided']:.4f} "
          f"| pending {len(integrity['pending'])} | waves unchecked {len(integrity['waves_unchecked'])}, not run {len(integrity['not_run'])} "
          f"| wrote {a.out_json} {a.out_md}")


def markdown(o, cfg):
    it, ad, rt = o['itt'], o['adoption'], o['retreat']
    mname = {'skeleton_king': '骷髅王', 'butcher': '屠夫'}
    yes = lambda b: '满足' if b else ('无法评估（不挡采纳）' if b is None else '不满足')
    L_ = []
    w = L_.append
    if o['mock']:
        w('> **【模拟试跑】** 本报告用第 8 轮留出日志冒充「新」「旧」两臂，两臂其实是同一个模型（d9facts3）在不同种子上的对局，'
          '按排序硬配成对。只为检验打分脚本能跑通、数字对得上，**不代表任何比较，不能引用**。\n')
    w(f"# {o['label']}：打分报告\n")
    inc = o['integrity']['pending'] and not o['final']
    ig = o['integrity']
    if ig.get('waves_unchecked') or ig.get('not_run'):
        w('> **【未完成】** 以下批次还没有通过的逐波核对（check.json 退出码须为 0 或 3），或者还没跑，其成绩不能当正式数据：'
          + '；'.join([f"{bb}（{ig['wave_checks'][bb]['why']}）" for bb in ig.get('waves_unchecked') or []]
                     + [f'{b}（没跑）' for b in ig.get('not_run') or []]) + '\n')
    w('## 一、结论\n')
    w(f"- **判定：{ad['verdict']}**" + ('（**未完成**：有 A 类故障局还没补跑，见第七节）' if inc else ''))
    diffs = [k for k, v in o['sensitivity'].items() if v['differs']]
    if diffs:
        w(f"- **注意：敏感性分析与意向处理结论不同**（{'、'.join(diffs)}），以意向处理为准，见第六节。")
    if o['integrity']['buy_sell_exact']:
        w(f"- **代码缺陷：** 出现同一趟进城内买了又卖（{len(o['integrity']['buy_sell_exact'])} 局），第 9 轮菜单本应藏掉这类选项；结论需复核。")
    w(f"- 击杀（两任务合计，意向处理）：新 {it['K_new']}，旧 {it['K_old']}；死亡：新 {it['D_new']}，旧 {it['D_old']}；"
      f"屠夫任务越界：新 {it['V_new']}，旧 {it['V_old']}。")
    sl = it['seed_level']
    w(f"- 确认性检验（种子层精确符号翻转，单侧，新优于旧）：{sl['n_seeds']} 个种子，新好 {sl['seeds_new_better']}、旧好 {sl['seeds_old_better']}、"
      f"持平 {sl['seeds_tied']}；T={sl['T']}，p={sl['p_one_sided']:.4f}（α=0.05）。显著与否不改变判定。")
    w('- 本考试只能检出约 30 个百分点以上的击杀率差；不显著不能解读为「持平」。')
    an, ao = cfg['arms']['new'], cfg['arms']['old']   # round 9b: the arms come from exam9_config.json (new = d11facts5, v5)
    w(f"- 比较的是整套：新模型 {an['tag']} 加第 {an['facts_version']} 版结论行，对旧模型 {ao['tag']} 加第 {ao['facts_version']} 版结论行，"
      '两臂都用第 9 轮菜单和同一双手。'
      '旧臂的成绩不能和第 8 轮留出的 17/30 相比，这次考试也说明不了第 9 轮菜单本身的作用。\n')
    w('## 二、各任务配对结果\n')
    w('| 任务 | 对数 | 都杀 | 只有新杀 | 只有旧杀 | 都没杀 | 新击杀率（Wilson 95%） | 旧击杀率（Wilson 95%） | 差（新−旧） | Newcombe 95% | Tango 95% | McNemar 精确双侧 p | 死亡 新/旧 |')
    w('|---|---|---|---|---|---|---|---|---|---|---|---|---|')
    for m in L.MISSIONS:
        v = it['missions'][m]
        n = v['n']
        dtxt = '—' if v['diff'] is None else '%+.0f 个百分点' % (100 * v['diff'])
        w(f"| {mname[m]} | {n} | {v['both']} | {v['new_only']} | {v['old_only']} | {v['neither']} | "
          f"{v['K_new']}/{n}（{fmt_ci(v['wilson_new'])}） | {v['K_old']}/{n}（{fmt_ci(v['wilson_old'])}） | "
          f"{dtxt} | {pp(v['newcombe_ci'])} | {pp(v['tango_ci'])} | "
          f"{v['mcnemar_p_two_sided']:.3f} | {v['D_new']}/{v['D_old']} |")
    w('\n死亡的同样表格：\n')
    w('| 任务 | 都死 | 只有新死 | 只有旧死 | 都没死 | 新死亡率（Wilson 95%） | 旧死亡率（Wilson 95%） | Newcombe 95%（新−旧） | Tango 95% | McNemar 精确双侧 p |')
    w('|---|---|---|---|---|---|---|---|---|---|')
    for m in L.MISSIONS:
        v, n = it['missions'][m]['deaths'], it['missions'][m]['n']
        w(f"| {mname[m]} | {v['both']} | {v['new_only']} | {v['old_only']} | {v['neither']} | {v['both'] + v['new_only']}/{n}（{fmt_ci(v['wilson_new'])}） | "
          f"{v['both'] + v['old_only']}/{n}（{fmt_ci(v['wilson_old'])}） | {pp(v['newcombe_ci'])} | {pp(v['tango_ci'])} | {v['mcnemar_p_two_sided']:.3f} |")
    w('\n以上 p 值和区间只作描述（4.2）；唯一的确认性检验在第三节。\n')
    w('## 三、种子层确认性检验\n')
    w(f"- 每个种子 D = 它各任务的（Y新 − Y旧）之和；{sl['nonzero']} 个种子 D≠0。统计量 T = ΣD = {sl['T']}。")
    w(f"- 精确符号翻转：单侧 p = {sl['p_one_sided']:.4f}，双侧 p = {sl['p_two_sided']:.4f}（双侧只作参考）。")
    w(f"- 逐种子 D：{', '.join(f'{k[-3:]}:{v:+d}' for k, v in sl['D_by_seed'].items())}\n")
    w('## 四、采纳规则逐条\n')
    ms = it['missions']
    w(f"1. 击杀：合计 新 {it['K_new']} ≥ 旧 {it['K_old']}？每任务 新−旧 ≥ −1？"
      f"（骷髅王 {ad['c1_per_mission_diff']['skeleton_king']:+d}，屠夫 {ad['c1_per_mission_diff']['butcher']:+d}）→ **{yes(ad['c1'])}**")
    w(f"2. 死亡：合计 新 {it['D_new']} ≤ 旧 {it['D_old']}？→ **{yes(ad['c2'])}**")
    w(f"3a. 屠夫任务越界（在 3 层或墓里做过决定）：新 {it['V_new']} ≤ 1 且（少于旧 {it['V_old']} 或为 0）？→ **{yes(ad['c3a'])}**"
      + (f"（注：{ad['c3a_note']}）" if ad['c3a_note'] else ''))
    rn, ro = rt['new'], rt['old']
    f_ = lambda x: '—' if x is None else f'{100 * x:.1f}%'
    w(f"3b. 撤退比例（地牢里、治疗药 ≤ 1 瓶、视野内没有活怪、菜单上有上楼或出墓选项的决定里，选了上楼或出墓的比例）：新 {rn['num']}/{rn['den']} = {f_(rn['pooled'])}"
      f"（按局平均 {f_(rn['per_game_mean'])}，{rn['games_with_qualifying']} 局有合格决定）；旧 {ro['num']}/{ro['den']} = {f_(ro['pooled'])}"
      f"（按局平均 {f_(ro['per_game_mean'])}）；门槛 80%，合格决定少于 {cfg['retreat']['min_decisions']} 个记无法评估 → **{yes(ad['c3b'])}**")
    w(f"3c. 代码完整性（不是模型成绩）：同一趟进城内买了又卖（按 menu_bought 精确核对）{len(o['integrity']['buy_sell_exact'])} 局"
      + ('' if o['integrity']['menu_bought_logged'] else '；**日志里没有 menu_bought 字段，精确核对不可用**') + '。')
    w(f"\n判定表：1、2、3 都满足 → 采纳；1、2 满足而 3 不满足 → 不采纳，查原因；1 或 2 不满足 → 不采纳。显著性不改变任何一格；这批考试种子无论如何都算用掉了。\n")
    w('## 五、分层\n')
    w('| 层 | 对数 | 新杀（Wilson 95%） | 旧杀（Wilson 95%） | 只有新杀 | 只有旧杀 | McNemar 双侧 p | Newcombe 95% | Tango 95% | 死亡 新/旧 | 越界 新/旧 |')
    w('|---|---|---|---|---|---|---|---|---|---|---|')
    for k, v in it['strata'].items():
        w(f"| {k} | {v['n']} | {v['K_new']}（{fmt_ci(v['wilson_new'])}） | {v['K_old']}（{fmt_ci(v['wilson_old'])}） | {v['new_only']} | {v['old_only']} | "
          f"{v['mcnemar_p_two_sided']:.3f} | {pp(v['newcombe_ci'])} | {pp(v['tango_ci'])} | "
          f"{v['D_new']}/{v['D_old']} | {'—' if v['V_new'] is None else str(v['V_new']) + '/' + str(v['V_old'])} |")
    w('')
    w('## 六、敏感性分析\n')
    w('| 分析 | 对数 | 击杀 新/旧 | 死亡 新/旧 | 越界 新/旧 | 单侧 p | 判定 | 与意向处理不同？ |')
    w('|---|---|---|---|---|---|---|---|')
    w(f"| 意向处理（主分析） | {it['n_pairs']} | {it['K_new']}/{it['K_old']} | {it['D_new']}/{it['D_old']} | {it['V_new']}/{it['V_old']} | {sl['p_one_sided']:.4f} | {ad['verdict']} | — |")
    for k, v in o['sensitivity'].items():
        w(f"| {k} | {v['n_pairs']} | {v['K_new']}/{v['K_old']} | {v['D_new']}/{v['D_old']} | {v['V_new']}/{v['V_old']} | {v['p_one_sided']:.4f} | {v['verdict']} | {'是' if v['differs'] else '否'} |")
    w('\n（撤退比例在敏感性分析里沿用全体数字。）\n')
    w('## 七、故障与补跑\n')
    fails = [u for u in o['units'] if u['primary_cls'] != 'ok' or u['cls'] != 'ok']
    if not fails:
        w('- 没有 A 类或 B 类故障，也没有人工停局。')
    for u in fails:
        mk = u.get('makeup')
        s = f"- {u['arm']} {mname[u['mission']]} {u['seed']}：原局 {u['primary']} {u['primary_status']}（{u['primary_cls']}：{u['primary_why']}）"
        if mk:
            pf = mk['prefix']
            s += (f"；补跑 {mk['game']} {mk['status']}（{mk['cls']}）；复现原局前 {pf['matched']}/{pf['original_rows']} 个决定"
                  + ('，完全复现' if pf['reproduced'] else '，**未完全复现**'))
        if u['flags']['pending']:
            s += '；**待补跑**'
        if u['flags']['a2']:
            s += '；**二次失败，记 0**'
        if u['flags']['op_stop']:
            s += '；人工停局（禁止），记 0'
        w(s)
    if o['integrity']['extra_makeups']:
        w(f"- **同一单元出现计划外的补跑批次**（m1 以外的补跑波；只用 m1 或其最新整波重跑）：{o['integrity']['extra_makeups']}")
    if o['integrity']['voided_waves']:
        w(f"- 按 3.3 作废、已整波重跑的批次（不计分）：{o['integrity']['voided_waves']}")
    if o['integrity']['broker_restarts']:
        w(f"- 推理进程死亡或重启记录：{o['integrity']['broker_restarts']}")
    if o['integrity']['boundary_rows_disagree']:
        w(f"- 越界判定（milestones）与决定行（scene≥3）不一致的局：{o['integrity']['boundary_rows_disagree']}")
    w('')
    w('## 八、次终点（只报数，不作判据）\n')
    for arm in ('new', 'old'):
        d = o['descriptive'][arm]
        w(f"- **{'新' if arm == 'new' else '旧'}臂**：结局 {d['statuses']}；故障类别 {d['classes']}；决定 {d['decisions']} 个，"
          f"菜单规则藏掉选项的决定 {d['menu_hidden_rows']} 个，不许撤销规则藏掉选项的决定 {d['undo_hidden_rows']} 个；"
          f"低药事件结局 {d['low_potion_events']}；按名字的买了又卖：同趟 {d['buy_sell_same_visit_by_name']}、跨趟 {d['buy_sell_cross_visit_by_name']}；"
          f"打转类型 {d['cycles']}；骷髅王任务杀了屠夫的种子 {d['butcher_killed_in_king_mission']}；屠夫任务杀了骷髅王的种子 {d['king_killed_in_butcher_mission']}；"
          f"开局后回城 {d['town_trips']} 趟，其中白跑（没买卖修鉴、没回血）{d['wasted_trips']} 趟；跨趟买了又卖亏 {d['cross_visit_sell_loss_gold']} 金；"
          f"各输出批次最长一局墙钟（分钟）{d['wall_minutes_by_batch']}。")
        for m in L.MISSIONS:
            v = d['missions'][m]
            b = v['boss_first_seen']
            w(f"  - {mname[m]}：{v['games']} 局（含只作描述的额外局），杀 {v['kills']}，死 {v['deaths']}；击杀帧数（升序）{v['kill_ticks']}；"
              f"第一次见到首领 {b['games']} 局，中位等级 {b['median_level']}、敏捷 {b['median_dexterity']}、药 {b['median_potions']}、帧 {b['median_tick']}；"
              f"累计（帧：杀/死）{'，'.join(str(c['tick']) + '：' + str(c['kills']) + '/' + str(c['deaths']) for c in v['cumulative'])}")
    for m in L.MISSIONS:
        v = o['kill_time_paired'][m]
        w(f"- {mname[m]}：两臂都杀的 {v['n']} 个种子，击杀帧数差（新−旧）中位 {v['median_new_minus_old_ticks']}。")
    w('')
    w('## 九、检出力（预注册 4.3 的精确计算）\n')
    w('ρ 是两臂共享种子运气的比例，取 0–0.6；每格前一个数是单侧，后一个是双侧。')
    w('| 设定 | +10pp | +20pp | +25pp | +30pp | +35pp |')
    w('|---|---|---|---|---|---|')
    w('| 每个任务 16 对，旧臂击杀率 0.50 | .06–.07 / .02–.04 | .17–.21 / .11 | .26–.33 / .17–.20 | .38–.47 / .27–.32 | .51–.62 / .39–.46 |')
    w('| 每个任务 16 对，旧臂击杀率 0.65 | .05–.07 / .02–.03 | .20–.21 / .10–.11 | .31–.35 / .20 | .48–.52 / .32–.33 | .71 / .51 |')
    w('| 两个任务合并 32 对，旧臂 0.55 | .14–.17 / .08–.11 | .42–.55 / .29–.43 | .60–.74 / .46–.63 | — | — |')
    w('\n击杀率差约 30 个百分点时，单个任务的检出力也只有一半左右；不显著不等于持平，显著与否不进入采纳规则。\n')
    w('## 十、逐局表\n')
    w('| 臂 | 任务 | 种子 | 用的局 | 结局 | 类别 | Y | 死 | 越界 | 决定 | 分钟 | 最深 | 撤退 分子/分母 |')
    w('|---|---|---|---|---|---|---|---|---|---|---|---|---|')
    for u in sorted(o['units'], key=lambda u: (u['mission'], u['seed'], u['arm'])):
        w(f"| {u['arm']} | {mname[u['mission']]} | {u['seed']} | {u['used']} | {u['status']} | {u['cls']} | {u['y']} | {int(u['dead'])} | "
          f"{'' if u['boundary'] is None else int(u['boundary'])} | {u['decisions']} | {u['minutes']} | {u['deepest']} | {u['retreat_num']}/{u['retreat_den']} |")
    if o['extra']:
        w('\n只进描述合计、不进配对的额外局：' + '；'.join(f"{e['arm']} {e['game']} {e['status']}" for e in o['extra']))
    return '\n'.join(L_) + '\n'


if __name__ == '__main__':
    main()

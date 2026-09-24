"""Executor r3 ("hands" r3): r2's frozen executor plus four opt-in fixes.

Spec: EXECUTOR-DIAGNOSIS.md, English technical appendix (step 1, diffs A/B/C); diff D is specified below.

  A  target_rule  Among the brain's listed targets that are adjacent now, strike the one with the
                  lowest current HP first (ties: the brain's list order). Never an unlisted id;
                  non-adjacent order unchanged; no adjacent listed target -> r2's pick.
  B  chain        Attack chaining: while the hero is in a swing / block / got-hit (modes 4/6/7) on
                  the same standing-legal target, re-queue the native Shift-attack (attack_stand ->
                  CMD_SATTACKXY) so the engine starts the next swing on the frame after the hit
                  instead of waiting for the whole 16-frame animation. Only 'attack_stand' is ever
                  chained (never CMD_ATTACKID, never a chase). A rejected chained command is counted,
                  not raised. Two guards (chain_guard):
                    'min_damage' (default; the appendix spec): chain in any of modes 4/6/7 when
                        target hp > hero min_damage. The command is queued BEFORE the swing's hit,
                        so when that hit kills the target the engine starts the queued swing at the
                        corpse's now-empty tile one tick later (DoAttack hits whatever stands at
                        tile + _pdir). The guard cannot predict a kill: real damage is
                        rnd(min..max) * (1 + bonus%) + bonus mod + _pDamageMod, x2 on a crit
                        (clvl%), +-50% vs animals/undead with a sword (player.cpp PlrHitMonst).
                    'post_hit' (proposed spec change, needs the project owner's approval): in PM_ATTACK the
                        chain is queued only once the current swing's hit frame has been processed
                        (read-only telemetry: frame > attack_frame) and the target is still alive;
                        in block/got-hit (no hit pending) it is queued as in the spec. No HP guard.
                        So that the queue lands on the first tick the engine accepts it
                        (CheckNewPath: currentFrame >= _pAFNum), the 4-tick beat is cut short on
                        the tick whose hit was just processed. The next swing then starts one tick
                        after the hit - the same tick as a pre-queued chain - but only on a target
                        that survived; if the hit killed it, r2's retarget sends the next target's
                        attack on that same beat instead of a swing into an empty cell.
  C  hold_floor   In hold goals only: return 'damage_received' on a damaging step that leaves
                  HP <= hold_floor * max HP, or on a single-step loss >= 20% of max HP; otherwise
                  keep holding. (Non-hold goals keep r2's exit on any HP loss.)
  D  chase_last_seen  (visibility-boundary flicker fix, added 2026-09-23.) Non-hold fight goals only.
                  When no authorised target is listed any more, r2 returns 'targets_no_longer_visible'
                  at once; at a lit-radius boundary the approach walk can hide the target and the
                  next goal's walk shows it again, forever. With this flag the goal instead walks
                  (goal head, walk_to in 8-tick chunks, like r2's approach) toward a tile next to the
                  last public position of the last chosen target, as listed in s.state['enemies']
                  earlier in the SAME goal, for at most CHASE_TICKS ticks per disappearance. Every
                  chunk goes back through all of fight()'s exits (dead, scene, damage, stats,
                  new_enemy against the goal's start set, ...). The fight resumes as soon as an
                  authorised target is listed again; the walk's own 'new_enemy' stop is therefore
                  ignored when the only newly listed ids are authorised targets (the target coming
                  back) and returned as before for any other id. It returns
                  'targets_no_longer_visible' when: the target was never listed in this goal; the
                  public kill count rose since it was last listed (it most likely died - the engine
                  counts the kill on the tick the monster leaves the list); the hero stands next to
                  the last-seen tile and it is still not listed; no tile next to it has a known
                  path; the chunk walk reports no_known_path; the chase budget runs out; or the
                  goal's own tick budget runs out while it is still not listed.

Every flag defaults OFF. With all flags off fight() is literally r2's fight() (super call), so the
game is bit-identical to r2. With chain_guard='min_damage' (default) the r3 code path is exactly the
one validated before the guard option existed (same commands, same beats, same log records).
With chase_last_seen off (default) nothing of D runs: no extra record, count, config key or install()
info key, and the goal returns at exactly the same point as before D existed.
game_executor_r2.py / game_support_r3.py are not modified: install() swaps the Executor name that
game_support_r3.Controller looks up at construction time.
Nothing here uses a GPU (model158 and the goal head run on CPU exactly as in r2).
"""
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_SFT = _HERE.parent / 'strategy-brain-sft-20260922'
if str(_SFT) not in sys.path:
    sys.path.insert(0, str(_SFT))

import game_executor_r2 as _r2  # noqa: E402
from game_executor_r2 import (Executor, pick_held_target, legal_edges, idle, point, future, scene,  # noqa: E402
                              digest, np, torch, DIRS)

R2_EXECUTOR = Executor          # the frozen r2 class, kept for install('r2')
CHAIN_MODES = (4, 6, 7)         # PM_ATTACK, PM_BLOCK, PM_GOTHIT
PM_ATTACK = 4
BIG_HIT_FRACTION = 0.20         # diff C: a single-step loss of >= 20% of max HP ends a hold
CHAIN_GUARDS = ('min_damage', 'post_hit')
CHASE_TICKS = 48                # diff D: walking budget per disappearance of the chosen target
CHASE_CHUNK = 8                 # diff D: ticks per walk_to chunk (r2's approach refresh interval)


def _sha(path):
    with open(path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()


def _check_guard(chain, chain_guard):
    if chain_guard not in CHAIN_GUARDS:
        raise ValueError(f'chain_guard must be one of {CHAIN_GUARDS}')
    if chain_guard != 'min_damage' and not chain:
        raise ValueError('chain_guard other than min_damage requires chain=True')


class _LogWithCounts:
    """Transparent wrapper of r2's native decisions log: close() also writes native/r3-counts.json.

    Controller.close() calls self.e.log.close(); this is the only hook needed, so r2 and
    game_support_r3 stay untouched. Writing the counts can never break the close."""

    def __init__(self, f, on_close):
        self._f = f
        self._on_close = on_close

    def write(self, text):
        return self._f.write(text)

    def flush(self):
        return self._f.flush()

    def close(self):
        try:
            self._on_close()
        except Exception:  # counts are diagnostics only; never block the native close
            pass
        finally:
            self._f.close()

    def __getattr__(self, name):
        return getattr(self._f, name)


class ExecutorR3(Executor):
    chain_guard = 'min_damage'  # class default (the spec guard); __init__ sets the instance value
    chase_last_seen = False     # class default (diff D off); __init__ sets the instance value
    chase_ticks = CHASE_TICKS

    def __init__(self, s, chain=False, target_rule=False, hold_floor=None, chain_guard='min_damage',
                 chase_last_seen=False):
        _check_guard(bool(chain), chain_guard)  # before anything is built
        super().__init__(s)
        self.chase_last_seen = bool(chase_last_seen)
        self.chain = bool(chain)
        self.target_rule = bool(target_rule)
        if hold_floor is not None:
            hold_floor = float(hold_floor)
            if not 0.0 < hold_floor < 1.0:
                raise ValueError('hold_floor must be a fraction of max HP in (0, 1)')
        self.hold_floor = hold_floor
        self.chain_guard = chain_guard
        self.r3_counts = Counter()
        self.log = _LogWithCounts(self.log, self.write_counts)

    @property
    def active(self):
        return self.chain or self.target_rule or self.hold_floor is not None or self.chase_last_seen

    def config(self):
        c = dict(kind='r3', chain=self.chain, target_rule=self.target_rule, hold_floor=self.hold_floor,
                 chain_guard=self.chain_guard)
        if self.chase_last_seen:  # diff D: key present only when on (default output unchanged)
            c.update(chase_last_seen=True, chase_ticks=self.chase_ticks)
        return c

    def write_counts(self):
        path = Path(self.s.out) / 'r3-counts.json'
        tmp = path.with_suffix('.json.writing')
        tmp.write_text(json.dumps(dict(config=self.config(), counts=dict(sorted(self.r3_counts.items()))),
                                  indent=1, sort_keys=True), encoding='utf-8')
        os.replace(tmp, path)

    # ---- post_hit guard helpers (read-only; used only when chain_guard == 'post_hit') ---------------
    def _swing_frames(self):
        """(frame, attack_frame) of the hero's current animation from the read-only combat telemetry.
        DiagnosticSession.snapshot_combat() also checks that the engine checkpoint is unchanged."""
        snap = getattr(self.s, 'snapshot_combat', None)
        tel = snap() if snap is not None else self.s.b.manual_combat_telemetry()
        return tel['hero']['frame'], tel['hero']['attack_frame']

    def _hit_pending(self):
        """True while the hero is in PM_ATTACK and the swing's hit frame has not been processed yet.
        The engine hits on the tick that starts at currentFrame == _pAFNum - 1 (== attack_frame) and
        then advances the frame, so after that tick the telemetry shows frame > attack_frame."""
        if self.s.state['hero']['mode'] != PM_ATTACK:
            return False
        frame, attack_frame = self._swing_frames()
        return frame <= attack_frame

    def _beat_post_hit(self):
        """The 4-tick beat, cut short right after the tick on which a pending hit was processed."""
        s = self.s
        pending = self._hit_pending()
        for i in range(4):
            s.step(1)
            if s.state['hero']['dead']:
                return
            now = self._hit_pending()
            if pending and not now and s.state['hero']['mode'] == PM_ATTACK:
                if i < 3:
                    self.r3_counts['beat_cut_after_hit'] += 1
                return
            pending = now

    # ---- diff D helper (used only when chase_last_seen is on and the goal is not a hold) -------------
    def _chase_step(self, g, start, authorized, ch):
        """No authorised target is listed: start or continue the bounded walk toward the chosen
        target's last public position. Returns fight()'s exit reason, or None to go back to the top of
        fight()'s loop (which re-checks every exit and resumes the fight if a target is listed again).
        ch: dict(last_seen={id: (x, y)}, chosen=id or None, kills=int or None, active=None or dict)."""
        s = self.s
        if ch['active'] is None:
            tid = ch['chosen']
            if tid is None or tid not in ch['last_seen']:
                self.r3_counts['chase_skipped_never_listed'] += 1
                return 'targets_no_longer_visible'
            if s.state['kills'] > ch['kills']:
                # The engine counts the kill on the tick the monster leaves the list: most likely dead.
                self.r3_counts['chase_skipped_kill_counted'] += 1
                return 'targets_no_longer_visible'
            ch['active'] = dict(target_id=tid, last_seen=list(ch['last_seen'][tid]), start=s.state['tick'])
            self.r3_counts['chase_started'] += 1
        c = ch['active']
        tx, ty = c['last_seen']

        def give_up(why):
            self.r3_counts['chase_gave_up_' + why] += 1
            self.record('r3-chase-last-seen', dict(event='gave_up_' + why, target_id=c['target_id'], last_seen=c['last_seen'],
                                                   position=list(point(s.state)), ticks=s.state['tick'] - c['start']))
            return 'targets_no_longer_visible'

        x, y = point(s.state)
        if max(abs(x - tx), abs(y - ty)) <= 1:
            return give_up('arrived')           # next to the last-seen tile and still not listed
        left = self.chase_ticks - (s.state['tick'] - c['start'])
        if left <= 0:
            return give_up('budget')
        candidates = []
        for dx, dy in DIRS[1:]:
            q = (tx + dx, ty + dy)
            path = s.map.path(s.state, q)
            if path is not None:
                candidates.append((len(path), q))
        if not candidates:
            return give_up('unreachable')
        q = min(candidates)[1]
        listed = {e['id'] for e in s.state['enemies']}   # == walk_to's own 'seen' set
        t0 = s.state['tick']
        result = self.walk_to(q, limit=min(CHASE_CHUNK, left, g.get('max_ticks', 2000) - (t0 - start)))
        self.r3_counts['chase_ticks'] += s.state['tick'] - t0
        new = {e['id'] for e in s.state['enemies']} - listed
        self.record('r3-chase-last-seen', dict(event='walk', target_id=c['target_id'], last_seen=c['last_seen'], approach=list(q),
                                               walk=result, position=list(point(s.state)), ticks=s.state['tick'] - c['start'],
                                               listed_again=sorted(new & set(authorized))))
        if result == 'new_enemy':
            if new - set(authorized):
                self.r3_counts['chase_new_enemy'] += 1
                return 'new_enemy'                  # a non-target id: the same stop as r2's approach walk
            return None                             # only authorised target(s) came back: resume the fight
        if result == 'no_known_path':
            return give_up('no_known_path')
        if result not in ('arrived', 'bounded_yield', 'damage_received'):
            return result                           # dead / scene_changed / no_motion / ...: as r2's approach
        return None

    def fight(self, g):
        if not self.active:
            return super().fight(g)  # exact r2
        # From here: verbatim copy of r2 fight() (game_executor_r2.py:114-192) with diffs A, B, C, D
        # and the r3 log field. Each change is marked 'r3'.
        s=self.s;start=s.state['tick'];oldscene=scene(s.state)
        authorized=list(g['target_ids']);seen={e['id'] for e in s.state['enemies']}
        no_progress=0;previous=None
        hp_at_start=s.state['hero']['hp_fixed']
        stats_at_start=s.state['hero']['unspent_stats']
        last_hp=hp_at_start;past_r2_exit=False  # r3 diff C
        post_hit=self.chain and self.chain_guard=='post_hit'  # r3 diff B guard option
        chase=self.chase_last_seen and not g.get('hold_position')  # r3 diff D (never in a hold)
        ch=dict(last_seen={},chosen=None,kills=None,active=None) if chase else None
        self.r3_counts['fight_goals']+=1
        while s.state['tick']-start<g.get('max_ticks',2000):
            if s.state['hero']['dead']:return 'dead'
            if scene(s.state)!=oldscene:return 'scene_changed'
            if s.state['king_kills']>0:return 'king_dead'
            h=s.state['hero']
            hold=g.get('hold_position')
            if hold and (point(s.state)!=tuple(hold) or future(s.state)!=tuple(hold)):return 'hold_position_lost'
            if hold and self.hold_floor is not None:
                # r3 diff C (replaces r2:127 inside holds only).
                hp_now=h['hp_fixed']
                if hp_now<last_hp:
                    if hp_now<=self.hold_floor*h['max_hp_fixed']:
                        self.r3_counts['hold_exit_floor']+=1;return 'damage_received'
                    if last_hp-hp_now>=BIG_HIT_FRACTION*h['max_hp_fixed']:
                        self.r3_counts['hold_exit_big_hit']+=1;return 'damage_received'
                    self.r3_counts['hold_hits_absorbed']+=1
                    if hp_now<hp_at_start and not past_r2_exit:
                        past_r2_exit=True;self.r3_counts['hold_goals_kept_past_r2_exit']+=1
                last_hp=hp_now
            elif h['hp_fixed']<hp_at_start:return 'damage_received'
            if h['unspent_stats']>stats_at_start:return 'earned_stat_review'
            if {e['id'] for e in s.state['enemies']}-seen:return 'new_enemy'
            enemies={e['id']:e for e in s.state['enemies']}
            remaining=[i for i in authorized if i in enemies]
            if chase:
                # r3 diff D: remember public positions; walk toward the last one while none is listed.
                if remaining:
                    for i in remaining:ch['last_seen'][i]=(enemies[i]['x'],enemies[i]['y'])
                    ch['kills']=s.state['kills']
                    if ch['active'] is not None:
                        self.r3_counts['chase_target_listed_again']+=1
                        self.record('r3-chase-last-seen',dict(event='target_listed_again',target_id=ch['active']['target_id'],
                                    last_seen=ch['active']['last_seen'],listed=remaining,position=list(point(s.state)),
                                    ticks=s.state['tick']-ch['active']['start']))
                        ch['active']=None
                else:
                    stop=self._chase_step(g,start,authorized,ch)
                    if stop is not None:return stop
                    continue
            if not remaining:return 'targets_no_longer_visible'
            # r3 diff A (replaces r2:133-136).
            standing_legal={i for i in remaining if s.b.manual_can_standing_attack(i)} if (hold or self.target_rule) else set()
            r2_pick=pick_held_target(remaining,standing_legal) if hold else remaining[0]
            chosen=r2_pick
            if self.target_rule:
                adj=[i for i in remaining if i in standing_legal or (not hold and enemies[i]['attack_legal'])]
                if adj:chosen=min(adj,key=lambda i:(enemies[i]['hp'],remaining.index(i)))
            if chase:ch['chosen']=chosen  # r3 diff D
            target=enemies[chosen]
            can_attack=chosen in standing_legal if hold else target['attack_legal']
            r3=dict(chained=False,r2_pick=r2_pick,target_rule_changed=chosen!=r2_pick)
            if not can_attack and hold:
                # Explicit strategist wait: no movement, no secret attack.
                s.step()
                self.record('strategist-directed-hold',dict(position=list(hold),target_priority=authorized,reason='wait_for_native_standing_range',r3=r3))
                continue
            if not can_attack:
                # The brain's explicit target supplies the waypoint, not a
                # macro-selected replacement enemy. Approach is goal-head work.
                candidates=[]
                for dx,dy in DIRS[1:]:
                    q=(target['x']+dx,target['y']+dy);path=s.map.path(s.state,q)
                    if path is not None:candidates.append((len(path),q))
                if not candidates:return 'target_unreachable'
                q=min(candidates)[1]
                if point(s.state)==q:
                    s.step(4);no_progress+=1
                    if no_progress>=2:return 'adjacent_but_native_attack_unavailable'
                    continue
                # Refresh the explicit enemy's public position after at most
                # eight ticks; never keep chasing its obsolete approach tile.
                result=self.walk_to(q,limit=8)
                if result not in ('arrived','bounded_yield','damage_received','no_known_path'):return result
                if result=='no_known_path':
                    no_progress+=1
                    if no_progress>=2:return 'approach_replan_failed_twice'
                else:no_progress=0
                continue
            self.r3_counts['model_beats']+=1
            if r3['target_rule_changed']:self.r3_counts['target_rule_changed']+=1
            legal=legal_edges(s);mask=[False]*15;mask[0]=True
            for i in range(1,9):mask[i]=legal[i] and not bool(hold)
            mask[9]=True;mask[12]=False  # Drinking belongs exclusively to the strategy brain.
            self.adapter.sync(mask);obs,actual=self.adapter.observation()
            with torch.no_grad():
                logits=self.actor(torch.tensor(obs)[None])[0];p=torch.softmax(logits.masked_fill(~torch.tensor(mask),-1e9),0)
                a=int(p.argmax())
            np.savez_compressed(self.obsdir/f'{self.decisions+1:06d}.npz',obs=obs,mask=actual)
            before=dict(tick=s.state['tick'],hp=h['hp_fixed'],target_hp=target['hp'],position=list(point(s.state)),mode=h['mode'])
            r=None;kind='advance_native_animation';chained=False
            if a==9:
                # r3 diff B (replaces r2:174-180): r2 re-issues only when idle or on a new target;
                # r3 additionally re-queues the native Shift-attack during swing/block/got-hit.
                stand=s.b.manual_can_standing_attack(target['id'])
                if post_hit:
                    # Proposed guard: never queue ahead of a pending hit (it may kill the target);
                    # queue once that hit is resolved and the target is still alive (it is in 'enemies').
                    same=(self.attack_command_target==target['id'] and h['mode'] in CHAIN_MODES and bool(stand))
                    hit_resolved=h['mode']!=PM_ATTACK or not self._hit_pending()
                    chained=same and hit_resolved
                    r3['hit_resolved']=hit_resolved
                    if same and not hit_resolved:self.r3_counts['chain_held_for_pending_hit']+=1
                else:
                    chained=(self.chain and self.attack_command_target==target['id'] and h['mode'] in CHAIN_MODES
                             and bool(stand) and target['hp']>h['min_damage'])
                if self.attack_command_target!=target['id'] or idle(s.state) or chained:
                    kind='attack_stand' if hold or stand else 'attack'
                    r=s.action(kind,{'id':target['id']},digest(s.state))
                    if r['accepted']:self.attack_command_target=target['id']
                    self.r3_counts['attack_commands']+=1
                    if chained:
                        self.r3_counts['chain_sent']+=1
                        if r['accepted']:self.r3_counts['chain_accepted']+=1
            elif a==12:
                raise RuntimeError('worker drinking is forbidden in strategy-brain experiment')
            elif 1<=a<=8:
                x,y=future(s.state);dx,dy=DIRS[a];kind='walk'
                r=s.action(kind,dict(x=x+dx,y=y+dy),digest(s.state));self.attack_command_target=None
            if r is not None and not r['accepted']:
                # r3 diff B (replaces r2:186): a rejected chained command is data, not a crash.
                if chained:self.r3_counts['chain_rejected']+=1
                else:raise RuntimeError('RL legal action rejected:'+r['reason'])
            if post_hit:self._beat_post_hit()
            else:s.step(4)
            r3['chained']=chained
            self.record('frozen-model158',dict(action=a,probabilities=p.tolist(),logits=logits.tolist(),target_id=target['id'],native_kind=kind,receipt=r,before=before,r3=r3))
            current=(s.state['hero']['hp_fixed'],s.state['kills'],point(s.state),tuple((e['id'],e['hp']) for e in s.state['enemies']))
            no_progress=no_progress+1 if current==previous else 0;previous=current
            if no_progress>=12:return 'two_no_progress_windows'
        if chase and ch['active'] is not None and not any(e['id'] in authorized for e in s.state['enemies']):
            # r3 diff D: the goal's own tick budget ran out during a chase, target still not listed.
            self.r3_counts['chase_gave_up_goal_budget']+=1
            self.record('r3-chase-last-seen',dict(event='gave_up_goal_budget',target_id=ch['active']['target_id'],
                        last_seen=ch['active']['last_seen'],position=list(point(s.state)),ticks=s.state['tick']-ch['active']['start']))
            return 'targets_no_longer_visible'
        return 'bounded_yield'


def install(kind='r2', chain=False, target_rule=False, hold_floor=None, chain_guard='min_damage', chase_last_seen=False):
    """Select the hands that game_support_r3.Controller will construct. Call BEFORE Controller().

    kind='r2' leaves (restores) the frozen r2 Executor; kind='r3' makes Controller build ExecutorR3
    with the given flags. Returns the executor config (for started.json / result.json).
    chase_last_seen (diff D) appears in the returned info only when it is on."""
    import game_support_r3 as gs
    info = dict(kind=kind, chain=bool(chain), target_rule=bool(target_rule),
                hold_floor=None if hold_floor is None else float(hold_floor), chain_guard=chain_guard,
                r2_sha256=_sha(_r2.__file__), r3_sha256=_sha(__file__))
    if chase_last_seen:
        info.update(chase_last_seen=True, chase_ticks=CHASE_TICKS)
    if kind == 'r2':
        if chain or target_rule or hold_floor is not None or chain_guard != 'min_damage' or chase_last_seen:
            raise ValueError('r3 flags require kind="r3"')
        gs.Executor = R2_EXECUTOR
        return info
    if kind != 'r3':
        raise ValueError('unknown executor kind: ' + repr(kind))
    _check_guard(bool(chain), chain_guard)
    flags = dict(chain=bool(chain), target_rule=bool(target_rule), hold_floor=hold_floor, chain_guard=chain_guard)
    if chase_last_seen:
        flags['chase_last_seen'] = True
    if hold_floor is not None and not 0.0 < float(hold_floor) < 1.0:
        raise ValueError('hold_floor must be a fraction of max HP in (0, 1)')

    class ConfiguredExecutorR3(ExecutorR3):
        def __init__(self, s):
            super().__init__(s, **flags)

    gs.Executor = ConfiguredExecutorR3
    return info

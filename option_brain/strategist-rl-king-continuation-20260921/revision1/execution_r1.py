import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
"""Public-observation RL adapter and continuous manual actuator. No old macros."""
import importlib.util, sys, time
import numpy as np
import torch
from config import *
sys.path.extend([str(PRIOR),str(R16)])
from session import Session, idle, ready_edge, point, future, scene
from runtime import ScopedBridge, Adapter, install_package
from model import load_actor
from navigation import GoalHead, features

def load_bridge():
    build=read(BUILD)
    if sha(build['engine'])!=build['engine_sha256'] or sha(build['bridge'])!=build['bridge_sha256']:raise RuntimeError('native identity changed')
    spec=importlib.util.spec_from_file_location('_diablogym',build['bridge'])
    b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)
    return b,build

def pick_held_target(remaining, standing_legal):
    # The strategist explicitly orders this priority, never an automatic
    # selection of a hidden or unauthorized enemy.
    return next((i for i in remaining if i in standing_legal),remaining[0])

def heal_allowed(state,threshold=.65):
    h=state['hero']
    return h['hp_fixed']<=threshold*h['max_hp_fixed'] and any(not p['empty'] and p.get('heal_kind',0)>0 for p in state['belt'])

def legal_edges(s):
    x,y=future(s.state);allowed=s.map.allowed(s.state)
    visible={(a,b) for a,b,_ in s.state['tiles']}
    return [True]+[(x+dx,y+dy) in allowed and (x+dx,y+dy) in visible for dx,dy in DIRS[1:]]

def edge_delivery_due(state,last):
    return last is None or (ready_edge(state) and future(state)==last) or (idle(state) and future(state)!=last)

class ContinuousAdapter(Adapter):
    def __init__(self,s,modules):
        self.manual_mask=np.array([True]+[False]*14,bool)
        super().__init__(s,modules)
    def mask(self):return self.manual_mask.copy()
    def sync(self,mask):
        self.manual_mask=np.asarray(mask,bool)
        e=self.env.env;e._raw=self.bridge.observe();e._steps=self.s.state['tick']//4
        e._ep_kills=int(self.s.state['kills'])
        e._controller_snapshot=e._capture_controller_snapshot(e._raw)
        e._visited.add(point(self.s.state))
        self.env._last_base_obs=e._vectorize(e._raw)
    def execute(self,action):raise RuntimeError('legacy macro is forbidden in this candidate')

class Executor:
    def __init__(self,s):
        self.s=s;self.proxy=ScopedBridge(s);modules=install_package(self.proxy)
        self.adapter=ContinuousAdapter(s,modules);self.actor,self.actor_info=load_actor()
        self.head=GoalHead();d=torch.load(HEAD,map_location='cpu',weights_only=True)
        self.head.load_state_dict(d['state_dict']);self.head.eval();self.head.requires_grad_(False)
        self.decisions=0;self.goal=None;self.attack_command_target=None
        self.log=(s.out/'decisions.jsonl').open('x',encoding='utf-8')
        self.obsdir=s.out/'observations';self.obsdir.mkdir()
    def record(self,kind,body):
        self.decisions+=1
        if self.decisions>MAX_DECISIONS:raise TimeoutError('policy_decision_budget')
        row=dict(decision=self.decisions,goal=self.goal['id'],tick=self.s.state['tick'],component=kind,**body)
        self.log.write(canonical(row)+'\n');self.log.flush()
        return row
    def set_goal(self,g):
        if g.get('mode') not in ('move','fight','service','stop'):raise ValueError('unknown goal mode')
        if not isinstance(g.get('reason'),str) or not g['reason']:raise ValueError('explicit strategist reason required')
        self.goal=g;self.adapter.goal=g;self.proxy.goal=g
        self.adapter.env._win_begin(0)
    def navigation_edge(self,target):
        s=self.s;route=s.map.path(s.state,target)
        if route is None:return 'no_known_path'
        if not route:return 'at_destination'
        x,y=future(s.state);q=route[0];edge=(q[0]-x,q[1]-y);legal=legal_edges(s)
        f=features((target[0]-x,target[1]-y),edge,legal)
        with torch.no_grad():
            z=self.head(torch.tensor(f)[None])[0];a=int(z.masked_fill(~torch.tensor(legal),-1e9).argmax())
        if DIRS[a]!=edge:raise RuntimeError('navigation head chose wrong edge; no silent teacher fallback')
        r=s.action('walk',dict(x=x+DIRS[a][0],y=y+DIRS[a][1]),digest(s.state))
        self.record('trained-goal-head',dict(input=f.tolist(),logits=z.tolist(),action=a,receipt=r,destination=list(q)))
        if not r['accepted']:return 'rejected:'+r['reason']
        self.attack_command_target=None
        return 'submitted'
    def walk_to(self,target,limit=800,stop_new=True):
        s=self.s;start=s.state['tick'];oldscene=scene(s.state);seen={e['id'] for e in s.state['enemies']}
        hp=s.state['hero']['hp_fixed'];last=None;last_motion=start;last_point=point(s.state);redeliveries=0
        while s.state['tick']-start<limit:
            if s.state['hero']['dead']:return 'dead'
            if scene(s.state)!=oldscene:return 'scene_changed'
            if stop_new and {e['id'] for e in s.state['enemies']}-seen:return 'new_enemy'
            if s.state['hero']['hp_fixed']<hp:return 'damage_received'
            if s.state['hero']['unspent_stats'] and not self.goal.get('allow_unspent',False):return 'earned_stat_review'
            if point(s.state)==tuple(target) and idle(s.state):return 'arrived'
            if edge_delivery_due(s.state,last):
                if last is not None and idle(s.state) and future(s.state)!=last:
                    redeliveries+=1
                    if redeliveries>2:return 'delivery_failed_twice'
                route=s.map.path(s.state,target)
                if route is None:return 'no_known_path'
                if route:
                    result=self.navigation_edge(target)
                    if result!='submitted':return result
                    last=route[0]
            s.step()
            if point(s.state)!=last_point:
                last_motion=s.state['tick'];last_point=point(s.state);redeliveries=0
            if s.state['tick']-last_motion>=60:return 'no_motion'
        return 'bounded_yield'
    def fight(self,g):
        s=self.s;start=s.state['tick'];oldscene=scene(s.state)
        authorized=list(g['target_ids']);seen={e['id'] for e in s.state['enemies']}
        no_progress=0;previous=None
        while s.state['tick']-start<g.get('max_ticks',2000):
            if s.state['hero']['dead']:return 'dead'
            if scene(s.state)!=oldscene:return 'scene_changed'
            if s.state['king_kills']>0:return 'king_dead'
            h=s.state['hero']
            hold=g.get('hold_position')
            if hold and (point(s.state)!=tuple(hold) or future(s.state)!=tuple(hold)):return 'hold_position_lost'
            if h['hp_fixed']<=.30*h['max_hp_fixed']:return 'emergency_review'
            if {e['id'] for e in s.state['enemies']}-seen:return 'new_enemy'
            enemies={e['id']:e for e in s.state['enemies']}
            remaining=[i for i in authorized if i in enemies]
            if not remaining:return 'targets_no_longer_visible'
            standing_legal={i for i in remaining if s.b.manual_can_standing_attack(i)} if hold else set()
            chosen=pick_held_target(remaining,standing_legal) if hold else remaining[0]
            target=enemies[chosen]
            can_attack=chosen in standing_legal if hold else target['attack_legal']
            if not can_attack and hold:
                # Explicit strategist wait: no movement, no secret attack.
                s.step()
                self.record('strategist-directed-hold',dict(position=list(hold),target_priority=authorized,reason='wait_for_native_standing_range'))
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
            legal=legal_edges(s);mask=[False]*15;mask[0]=True
            for i in range(1,9):mask[i]=legal[i] and not bool(hold)
            mask[9]=True;mask[12]=heal_allowed(s.state,g.get('heal_threshold',.65))
            self.adapter.sync(mask);obs,actual=self.adapter.observation()
            with torch.no_grad():
                logits=self.actor(torch.tensor(obs)[None])[0];p=torch.softmax(logits.masked_fill(~torch.tensor(mask),-1e9),0)
                a=int(p.argmax())
            np.savez_compressed(self.obsdir/f'{self.decisions+1:06d}.npz',obs=obs,mask=actual)
            before=dict(tick=s.state['tick'],hp=h['hp_fixed'],target_hp=target['hp'],position=list(point(s.state)),mode=h['mode'])
            r=None;kind='advance_native_animation'
            if a==9:
                # Keep an already issued native swing/chase; do not repeatedly
                # reset it at decision boundaries. Re-issue only when idle.
                if self.attack_command_target!=target['id'] or idle(s.state):
                    kind='attack_stand' if hold or s.b.manual_can_standing_attack(target['id']) else 'attack'
                    r=s.action(kind,{'id':target['id']},digest(s.state))
                    if r['accepted']:self.attack_command_target=target['id']
            elif a==12:
                potion=next(p for p in s.state['belt'] if not p['empty'] and p.get('heal_kind',0)>0)
                kind='drink';r=s.action(kind,dict(index=potion['index'],identity=potion['identity']),digest(s.state))
                if r['accepted']:self.adapter.env._win['voluntary_drinks']+=1
            elif 1<=a<=8:
                x,y=future(s.state);dx,dy=DIRS[a];kind='walk'
                r=s.action(kind,dict(x=x+dx,y=y+dy),digest(s.state));self.attack_command_target=None
            if r is not None and not r['accepted']:raise RuntimeError('RL legal action rejected:'+r['reason'])
            s.step(4)
            self.record('frozen-model158',dict(action=a,probabilities=p.tolist(),logits=logits.tolist(),target_id=target['id'],native_kind=kind,receipt=r,before=before))
            current=(s.state['hero']['hp_fixed'],s.state['kills'],point(s.state),tuple((e['id'],e['hp']) for e in s.state['enemies']))
            no_progress=no_progress+1 if current==previous else 0;previous=current
            if no_progress>=12:return 'two_no_progress_windows'
        return 'bounded_yield'
    def service(self,g):
        # Finite exact brain-authored actions. No automatic stock selection,
        # scoring, shopping, loot sweep, stat allocation or retreat policy.
        for index,command in enumerate(g['commands']):
            self.service_index=index
            if command['kind']=='unbelt':raise ValueError('unsafe pixel unbelt forbidden; use verified unbelt_exact')
            if command['kind']=='move':
                reason=self.walk_to(command['target'],limit=command.get('max_ticks',1000),stop_new=command.get('stop_new',True))
                if reason!='arrived':return reason
                continue
            for _ in range(60):
                if idle(self.s.state) or command['kind'] in ('drink','dismiss'):break
                if self.s.state['hero']['dead']:return 'dead'
                self.s.step()
            if not idle(self.s.state) and command['kind'] not in ('drink','dismiss'):return 'native_not_idle'
            kind=command['kind'];args=dict(command.get('args',{}))
            if kind in ('pickup','equip','belt','unbelt_exact','buy','sell','repair','identify'):
                # Resolve the same public identity again after index shifts;
                # identity, not an untrusted stale slot, is the authority.
                group=command.get('group',{'pickup':'floor','buy':'stock','unbelt_exact':'belt'}.get(kind,'inventory'))
                items=[x for x in self.s.state[group] if not x['empty'] and list(x.get('identity',()))==list(args['identity'])]
                if len(items)!=1:return 'specified_item_unavailable'
                args['index']=items[0]['index']
            before=dict(hero=self.s.state['hero'],inventory=self.s.state['inventory'],belt=self.s.state['belt'],equipment=self.s.state['equipment'])
            r=self.s.action(kind,args,digest(self.s.state))
            if r['accepted']:self.s.step(4 if kind in ('pickup','operate','talk') else 1)
            self.record('strategist-directed-service',dict(command=command,receipt=r,before=before,hero_after=self.s.state['hero']))
            if r['accepted'] and kind in ('unbelt_exact','equip'):
                destination='inventory' if kind=='unbelt_exact' else 'equipment'
                if not any(not it['empty'] and list(it.get('identity',()))==list(args['identity']) for it in self.s.state[destination]):
                    raise RuntimeError('item action accepted but actual destination differs: '+kind)
                if kind=='unbelt_exact':
                    if before['equipment']!=self.s.state['equipment'] or any(before['hero'][k]!=self.s.state['hero'][k] for k in ('gold','hp_fixed')):
                        raise RuntimeError('unbelt changed equipment or resources')
            if not r['accepted'] and not(kind=='dismiss' and r['reason']=='no_dialog'):return 'rejected:'+r['reason']
            if self.s.state['hero']['dead']:return 'dead'
            if self.s.state['enemies'] and not g.get('allow_visible_enemies',False):return 'enemy_during_service'
        self.service_index=len(g['commands']);return 'completed'
    def run(self,g):
        self.set_goal(g);self.service_index=None
        if g['mode']=='fight':return self.fight(g)
        if g['mode']=='move':
            for q in g['targets']:
                reason=self.walk_to(q,limit=g.get('max_ticks',1000))
                if reason!='arrived':return reason
            return 'completed'
        if g['mode']=='service':return self.service(g)
        return 'strategist_stop'

def public_summary(s):
    state=s.state;key=','.join(map(str,state['scene']));known=s.map.scenes[key]
    walkable=s.map.allowed(state);tiles=known['tiles'];frontiers=[]
    # Read-only navigation diagnostics, not automatic choice of objective.
    for q in sorted(walkable):
        missing=sum(f'{q[0]+dx},{q[1]+dy}' not in tiles for dx,dy in DIRS[1:])
        if missing:
            path=s.map.path(state,q)
            if path is not None:frontiers.append(dict(x=q[0],y=q[1],distance=len(path),unknown_neighbors=missing))
    frontiers.sort(key=lambda q:(q['distance'],-q['unknown_neighbors']))
    return dict(state_id=digest(state),tick=state['tick'],scene=state['scene'],hero=state['hero'],kills=state['kills'],
        king_kills=state['king_kills'],king_quest_done=state['king_quest_done'],butcher=s.b.butcher_events(),enemies=state['enemies'],floor=state['floor'],objects=state['objects'],npcs=state['npcs'],
        exits=list(known['exits'].values()),quest_entrances=state['quest_entrances'],inventory=state['inventory'],belt=state['belt'],equipment=state['equipment'],stock=state['stock'],
        remembered_floor=s.map.remembered(state,'floor'),remembered_objects=s.map.remembered(state,'objects'),remembered_npcs=s.map.remembered(state,'npcs'),
        frontier_count=len(frontiers),frontiers=frontiers[:40],known_tiles=len(tiles))

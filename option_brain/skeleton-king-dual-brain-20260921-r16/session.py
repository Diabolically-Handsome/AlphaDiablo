"""Single native game, immutable journal, observed-map pathfinding. No policy."""
import hashlib
import copy
import importlib.util
import json
import os
import time
from collections import deque
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=Path('$AD_HOME/skeleton_king_dual_brain_20260920')
ASSETS='$AD_HOME/butcher_hit_score_20260913/source/build/engine/devilutionx.app/Contents/Resources'
DATA='$AD_HOME/Library/Application Support/diasurgical/devilution'

def canonical(value): return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False)
def digest(value): return hashlib.sha256(canonical(value).encode()).hexdigest()
def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.writing')
    with tmp.open('w',encoding='utf-8') as f:
        json.dump(value,f,ensure_ascii=False,indent=2);f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)
def load_bridge():
    build=read(HERE/'build.json'); manifest=read(build['manifest'])
    assert sha(build['engine'])==build['engine_sha256']
    assert sha(build['bridge'])==manifest['bridge_sha256']
    spec=importlib.util.spec_from_file_location('_diablogym',build['bridge'])
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module

def scene(state):return tuple(state['scene'])
def point(state):return state['hero']['x'],state['hero']['y']
def future(state):return state['hero']['future_x'],state['hero']['future_y']
def idle(state):return state['hero']['mode']==0 and point(state)==future(state) and state['hero']['walkpath0']==-1
def ready_edge(state):return idle(state) or (state['hero']['mode'] in (1,2,3) and state['hero']['walkpath0']==-1)

def visible_contacts(state):
    """Public proximity only; does not inspect hidden enemy future positions."""
    x,y=future(state)
    return {e['id'] for e in state['enemies']
            if e.get('attack_legal') is True or max(abs(e['x']-x),abs(e['y']-y))<=1}

class KnownMap:
    def __init__(self):self.scenes={}
    def observe(self,state):
        key=','.join(map(str,scene(state)))
        data=self.scenes.setdefault(key,{'tiles':{},'exits':{},'quest_entrances':{}})
        for x,y,walkable in state['tiles']:data['tiles'][f'{x},{y}']=bool(walkable)
        for e in state['exits']:data['exits'][f"{e['x']},{e['y']}"]=e
        for e in state['quest_entrances']['entries']:data['quest_entrances'][f"{e['x']},{e['y']}"]=e
        # Every native tick is observed, not only local-model decision endpoints.
        # Retain only public objects/items/NPCs actually seen. A remembered door
        # is a historical landmark, NEVER permission to operate an unseen ID.
        visible={(x,y) for x,y,_ in state['tiles']}
        walkable={(x,y):bool(w) for x,y,w in state['tiles']}
        for group in ('objects','floor','npcs'):
            memory=data.setdefault(group+'_memory',{})
            current={}
            for entity in state.get(group,[]):
                token=digest(entity['identity']) if group=='floor' else str(entity['id'])
                current[token]=copy.deepcopy(entity)
            for token,old in list(memory.items()):
                if (old['x'],old['y']) in visible and token not in current:del memory[token]
            for token,entity in current.items():
                entity['last_seen_tick']=state['tick']
                if group=='objects':entity['last_seen_walkable']=walkable.get((entity['x'],entity['y']))
                memory[token]=entity

    def remembered(self,state,group):
        if group not in ('objects','floor','npcs'):raise ValueError('unsupported memory group')
        data=self.scenes[','.join(map(str,scene(state)))]
        visible={(e.get('id') if group!='floor' else digest(e['identity']),e['x'],e['y']) for e in state.get(group,[])}
        result=[]
        for token,old in data.get(group+'_memory',{}).items():
            e=copy.deepcopy(old)
            key=token if group=='floor' else int(token)
            e['currently_observed']=(key,e['x'],e['y']) in visible
            result.append(e)
        return sorted(result,key=lambda e:(e['y'],e['x']))
    def allowed(self,state):
        data=self.scenes[','.join(map(str,scene(state)))]
        result={tuple(map(int,k.split(','))) for k,v in data['tiles'].items() if v}
        # Visible occupants only; no private native monster grid.
        result-={(e['x'],e['y']) for e in state['enemies']+state['npcs']}
        result.add(future(state));return result
    def path(self,state,target,bounds=None):
        target=tuple(target);start=future(state);allowed=self.allowed(state)
        if bounds is not None:
            x0,y0,x1,y1=bounds
            allowed={q for q in allowed if x0<=q[0]<=x1 and y0<=q[1]<=y1}
            if start not in allowed:return None
        if target not in allowed:return None
        parent={start:None};queue=deque([start])
        while queue:
            p=queue.popleft()
            if p==target:
                path=[]
                while p!=start:path.append(p);p=parent[p]
                return path[::-1]
            for dx,dy in ((0,-1),(1,0),(0,1),(-1,0)):
                q=(p[0]+dx,p[1]+dy)
                if q in allowed and q not in parent:parent[q]=p;queue.append(q)
        return None

class Session:
    def __init__(self,output,seed,tick_limit,deadline,bridge=None,bridge_info=None):
        self.out=Path(output);self.out.mkdir(parents=True,exist_ok=False)
        self.seed=seed;self.tick_limit=tick_limit;self.deadline=deadline
        self.b=bridge or load_bridge();self.b.manual_configure()
        self.journal=(self.out/'journal.jsonl').open('x',encoding='utf-8')
        self.seq=0;self.chain='0'*64;self.map=KnownMap()
        self.b.init(ASSETS,str(self.out/'save'),DATA,0,False)
        self.b.reset(seed)
        self.state=self.b.manual_observe();self.map.observe(self.state)
        self.record('reset',{'seed':seed},None)
        write(self.out/'started.json',dict(seed=seed,tick_limit=tick_limit,deadline=deadline,started_at=time.time(),bridge=bridge_info or read(HERE/'build.json')))
    def guard(self):
        if time.time()>=self.deadline:raise TimeoutError('wall_clock_budget')
        if self.state['hero']['dead']:raise RuntimeError('player_dead_terminal')
        if self.state['tick']>=self.tick_limit:raise TimeoutError('native_tick_budget')
    def record(self,kind,args,result):
        checkpoint=self.b.manual_checkpoint()
        row=dict(seq=self.seq,kind=kind,args=args,result=result,tick=self.state['tick'],state_sha256=digest(checkpoint),previous=self.chain)
        self.chain=digest(row);row['chain']=self.chain
        self.journal.write(canonical(row)+'\n');self.journal.flush();self.seq+=1
        return row
    def action(self,kind,args,state_id=None):
        self.guard()
        if state_id is not None and state_id!=digest(self.state):raise ValueError('stale observation rejected before execution')
        result=self.b.manual_action(kind,args)
        self.state=self.b.manual_observe();self.map.observe(self.state)
        self.record(kind,args,result)
        return result
    def step(self,n=1):
        for _ in range(n):
            self.guard();old=self.state['tick'];self.b.step(1)
            self.state=self.b.manual_observe();self.map.observe(self.state)
            if self.state['tick']!=old+1:raise RuntimeError('native tick did not advance exactly once')
            self.record('tick',{},None)
            if self.state['hero']['dead']:break
        return self.state
    def move(self,target,max_ticks=400,stop_new_enemy=True,bounds=None,known_enemies=None):
        original_scene=scene(self.state);original_enemies=set(known_enemies or ())|{e['id'] for e in self.state['enemies']}
        initial_hp=self.state['hero']['hp_fixed'];initial_contacts=visible_contacts(self.state)
        start=self.state['tick'];last=None;unchanged=0;old=point(self.state)
        self.last_move_evidence=dict(target=list(target),tick_before=start,
            position_before=list(point(self.state)),native_walk_commands=[])
        def finish(reason):
            self.last_move_evidence.update(reason=reason,tick_after=self.state['tick'],
                position_after=list(point(self.state)),future_after=list(future(self.state)),
                native_walk_submitted=any(c['accepted'] for c in self.last_move_evidence['native_walk_commands']))
            return reason
        while self.state['tick']-start<max_ticks:
            if scene(self.state)!=original_scene:return finish('scene_changed')
            if self.state['hero']['dead']:return finish('dead')
            if stop_new_enemy and {e['id'] for e in self.state['enemies']}-original_enemies:return finish('new_enemy')
            if self.state['tick']>start:
                # Return to the hands for another decision; never secretly
                # attack, drink, retreat, cancel a path, or stop native motion.
                if self.state['hero']['hp_fixed']<initial_hp:return finish('damage_received')
                if initial_contacts or visible_contacts(self.state):return finish('combat_contact')
                # A submitted edge can become occupied while we are walking.
                # Do not wait sixty stationary ticks for an obsolete waypoint.
                if last is not None and last!=future(self.state) and last not in self.map.allowed(self.state):return finish('blocked')
            if point(self.state)==tuple(target) and idle(self.state):return finish('arrived')
            # A NEW explicit movement choice must submit its first native
            # edge even during an attack, hit/block animation, or an older
            # chase path. OnWalk only retargets path/destAction; it does not
            # force the current animation to finish or suppress interruption.
            # Subsequent edges keep the previously validated continuous chain.
            if last is None or (ready_edge(self.state) and future(self.state)==last):
                route=self.map.path(self.state,target,bounds=bounds)
                if route is None:return finish('blocked')
                if route:
                    last=route[0]
                    result=self.action('walk',dict(x=last[0],y=last[1]))
                    self.last_move_evidence['native_walk_commands'].append(
                        dict(tick=self.state['tick'],destination=list(last),accepted=result['accepted'],reason=result['reason']))
                    if not result['accepted']:return finish('blocked:'+result['reason'])
            self.step()
            unchanged=unchanged+1 if point(self.state)==old else 0;old=point(self.state)
            if unchanged>=60:return finish('no_motion')
        return finish('bounded_yield')
    def checkpoint(self,reason):
        write(self.out/'pause.json',dict(reason=reason,seed=self.seed,tick=self.state['tick'],seq=self.seq,chain=self.chain,
                                        state_sha256=digest(self.b.manual_checkpoint()),state=self.state,map=self.map.scenes,at=time.time()))
        os.fsync(self.journal.fileno())
    def close(self,reason):
        self.checkpoint(reason)
        write(self.out/'closed.json',dict(reason=reason,tick=self.state['tick'],chain=self.chain,
            pause_sha256=sha(self.out/'pause.json'),dead=self.state['hero']['dead'],at=time.time()))
        self.journal.close();self.b.end_game()

def verified_rows(source):
    source=Path(source)
    rows=[json.loads(line) for line in (source/'journal.jsonl').read_text().splitlines()]
    if not rows or rows[0]['kind']!='reset':raise RuntimeError('missing reset')
    chain='0'*64
    for i,row in enumerate(rows):
        body={k:v for k,v in row.items() if k!='chain'}
        if row['seq']!=i or row['previous']!=chain or digest(body)!=row['chain']:raise RuntimeError('journal altered')
        chain=row['chain']
    pause=read(source/'pause.json')
    if pause['seq']!=len(rows) or pause['chain']!=chain or pause['tick']!=rows[-1]['tick'] or pause['state_sha256']!=rows[-1]['state_sha256']:
        raise RuntimeError('checkpoint does not cover every recorded command; rollback forbidden')
    if pause['state']['hero']['dead']:raise RuntimeError('death is terminal; reconstruction for continuation forbidden')
    return rows

def replay(source,out,tick_limit,deadline):
    rows=verified_rows(source)
    session=Session(out,rows[0]['args']['seed'],tick_limit,deadline)
    try:
        for row in rows:
            if row['kind']=='tick':session.step()
            elif row['kind']!='reset':
                result=session.action(row['kind'],row['args'])
                if result!=row['result']:raise RuntimeError(('receipt mismatch',row['seq']))
            actual=digest(session.b.manual_checkpoint())
            if actual!=row['state_sha256']:raise RuntimeError(('replay mismatch',row['seq'],row['kind'],actual,row['state_sha256']))
        session.checkpoint('replay_verified')
        return session
    except BaseException:
        session.close('replay_failed');raise

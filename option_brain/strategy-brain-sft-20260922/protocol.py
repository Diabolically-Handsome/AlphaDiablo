"""Shared, read-only public view and grounded strategist command contract."""
import copy
import hashlib
import json
from collections import deque

VERSION='strategy-public/1'
SYSTEM='''You are the sole strategy brain of a normal-start Diablo I warrior. Goal: defeat the Skeleton King and remain alive; stop on death or success. A frozen RL worker handles authorized combat and known-map movement. Only you choose destinations, enemies, shopping, equipment, earned attributes and drinking. No free items, healing, identifying or hidden knowledge. Health/mana are real points, not fixed-point units. Unknown map and unidentified bonuses are unavailable.
Return ONLY one JSON object with game_id,state_id,goal_id copied exactly, reason (brief), and action. Action forms:
{"mode":"move","target":[x,y]} on reachable known floor (frontiers reveal more).
{"mode":"fight","targets":[visible_enemy_ids],"hold":false} or hold:true to stay at current tile; target order is yours.
{"mode":"service","kind":"drink|pickup|equip|belt|unbelt_exact|buy|sell|repair|identify","ref":"group:index"} selecting a listed item and one of its allowed actions.
{"mode":"service","kind":"talk|operate","id":visible_id}; {"mode":"service","kind":"stat","attribute":"strength|dexterity|vitality|magic"}; {"mode":"service","kind":"dismiss"}.
{"mode":"resume"} continues a saved move/fight goal after a drink or interruption, with a fresh state binding. {"mode":"stop"} saves and ends the game.
Each service is one real action, then inspect the receipt. Purchase is not equip; command accepted is not completion. Drink never clears the current combat goal. Native hit/block interruptions remain. Damage, new enemies, transitions and completed/blocked goals return control to you; thinking pauses game time. Use the last four receipts, do not repeat a failed instruction blindly. Two consecutive invalid/no-progress attempts end the game. Save resources, review gear/durability, explore and prepare; there is no mandatory readiness threshold. Never claim to have performed an action yourself.'''

def canonical(x):return json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False)
def digest(x):return hashlib.sha256(canonical(x).encode()).hexdigest()
def position(s):return s['hero']['x'],s['hero']['y']
def near(a,b):return max(abs(a[0]-b[0]),abs(a[1]-b[1]))<=1

def normalized(x):
    if isinstance(x,list):return [normalized(v) for v in x]
    if not isinstance(x,dict):return x
    out={}
    for k,v in x.items():
        if k.endswith('_fixed'):
            if k not in ('hp_fixed','max_hp_fixed','mana_fixed','max_mana_fixed','bonus_hp_fixed'):
                raise ValueError('unknown fixed-point field '+k)
            out[k[:-6]]=v/64
        else:out[k]=normalized(v)
    return out

def item_actions(s,item,group):
    h=s['hero']; gold=h['gold']; acts=[];ui=bool(s.get('dialog')) or s.get('vendor') not in (None,'','none')
    if group=='belt':
        if item.get('heal_kind',0)>0:acts.append('drink')
        if not ui:acts.append('unbelt_exact')
    if group=='floor' and position(s)==(item.get('x'),item.get('y')):acts.append('pickup')
    if group=='inventory' and not ui:
        if item.get('location') in range(1,7) and item.get('can_use',False):acts.append('equip')
        if item.get('heal_kind',0)>0 and any(p['empty'] for p in s['belt']):acts.append('belt')
    if group=='stock' and item.get('fits') and item.get('price',gold+1)<=gold and not s.get('dialog'):
        acts.append('buy')
    for action,key in (('sell','sale_price'),('repair','repair_price'),('identify','identify_price')):
        if key in item and item[key]>0 and (action=='sell' or item[key]<=gold) and not s.get('dialog'):acts.append(action)
    return acts

def item_view(s,item,group):
    keep={'name','identified','quality','location','min_damage','max_damage','armor','durability',
          'max_durability','can_use','min_strength','min_dexterity','min_magic','heal_kind','gold',
          'portal_scroll','price','sale_price','repair_price','identify_price','vendor','fits','x','y'}
    if item.get('identified'):keep|={k for k in item if k.startswith('bonus_')}
    data={k:v for k,v in item.items() if k in keep and (v not in (0,False) or k in ('identified','can_use','fits'))}
    return dict(ref=f"{group}:{item['index']}",**normalized(data),actions=item_actions(s,item,group))

def map_summary(s,known):
    key=','.join(map(str,s['scene']));m=known[key];tiles=m['tiles']
    if 'tiles' not in s:
        # Historical public summaries did not retain a tile grid. Never invent one.
        return dict(known_tiles=s.get('known_tiles'),reachable_tiles=None,local_walkable_rows=[],
            frontiers_xy_distance_unknown=[[f['x'],f['y'],f['distance'],f['unknown_neighbors']] for f in s.get('frontiers',[])[:32]],
            frontier_count=s.get('frontier_count'),exits=s['exits'],
            quest_entrances=s['quest_entrances']['entries'],historical_grid_unrecorded=True)
    allowed={tuple(map(int,k.split(','))) for k,v in tiles.items() if v}
    allowed-={(e['x'],e['y']) for e in s['enemies']+s['npcs']}
    start=(s['hero']['future_x'],s['hero']['future_y']);allowed.add(start)
    dist={start:0};q=deque([start])
    while q:
        x,y=q.popleft()
        for dx,dy in ((0,1),(0,-1),(1,0),(-1,0)):
            p=(x+dx,y+dy)
            if p in allowed and p not in dist:dist[p]=dist[(x,y)]+1;q.append(p)
    front=[]
    for p,d in dist.items():
        n=sum(f'{p[0]+dx},{p[1]+dy}' not in tiles for dx,dy in ((0,1),(0,-1),(1,0),(-1,0)))
        if n:front.append([*p,d,n])
    front.sort(key=lambda v:(v[2],v[0],v[1]))
    selected=front if len(front)<=32 else front[:24]+front[-8:]
    # No hidden tiles, and no map state from later demonstrations.
    rows={}
    for x,y in sorted(allowed):
        if abs(x-start[0])<=10 and abs(y-start[1])<=10:rows.setdefault(y,[]).append(x)
    compact=[]
    for y,xs in sorted(rows.items()):
        ranges=[];a=b=xs[0]
        for x in xs[1:]:
            if x==b+1:b=x
            else:ranges.append([a,b]);a=b=x
        ranges.append([a,b]);compact.append([y,ranges])
    return dict(known_tiles=len(tiles),reachable_tiles=len(dist),local_walkable_rows=compact,
                frontiers_xy_distance_unknown=selected,frontier_count=len(front),
                exits=list(m['exits'].values()),quest_entrances=list(m['quest_entrances'].values()))

def make_input(s,known,game_id,goal_id,current=None,returns=(),retreat_reason=None):
    hero=normalized(s['hero'])
    obs={k:copy.deepcopy(s.get(k)) for k in ('scene','tick','kills','king_kills','king_quest_done','dialog','vendor')}
    if 'dialog' not in s:obs['historical_ui_flags_unrecorded']=True
    obs['hero']=hero
    for group in ('inventory','belt','equipment','stock','floor'):
        obs[group]=[item_view(s,it,group) for it in s[group] if not it['empty']]
    obs['enemies']=copy.deepcopy(s['enemies'])
    obs['objects']=copy.deepcopy(s['objects']);obs['npcs']=copy.deepcopy(s['npcs'])
    obs['legal_operate_ids']=[e['id'] for e in s['objects'] if near(position(s),(e['x'],e['y']))]
    obs['legal_talk_ids']=[e['id'] for e in s['npcs'] if near(position(s),(e['x'],e['y']))]
    return dict(protocol=VERSION,game_id=game_id,state_id=digest(s)[:16],goal_id=goal_id,
                observation=obs,known_map=map_summary(s,known),current_goal=current,
                retreat_reason=retreat_reason,last_four_returns=list(returns)[-4:])

def resolve_response(reply,packet,s,known,current=None):
    if set(reply)!={'game_id','state_id','goal_id','reason','action'}:raise ValueError('response keys')
    for k in ('game_id','state_id','goal_id'):
        if reply[k]!=packet[k]:raise ValueError('stale/wrong '+k)
    if digest(s)[:16]!=packet['state_id']:raise ValueError('live state changed')
    if not isinstance(reply['reason'],str) or not 1<=len(reply['reason'])<=300:raise ValueError('reason')
    a=reply['action'];mode=a.get('mode');g=dict(id=packet['goal_id'],state_id=digest(s),reason=reply['reason'])
    if mode=='resume':
        if not current:raise ValueError('nothing to resume')
        a=copy.deepcopy(current);mode=a['mode']
    if mode=='move':
        if set(a)!={'mode','target'}:raise ValueError('move keys')
        p=a['target']
        if not isinstance(p,list) or len(p)!=2 or any(type(v)!=int for v in p):raise ValueError('coordinates')
        m=known[','.join(map(str,s['scene']))]['tiles']
        if not m.get(f'{p[0]},{p[1]}'):raise ValueError('unknown/nonwalkable destination')
        g.update(mode='move',targets=[p],max_ticks=800,allow_unspent=True)
    elif mode=='fight':
        if set(a)!={'mode','targets','hold'} or type(a['hold'])!=bool:raise ValueError('fight keys')
        visible={e['id'] for e in s['enemies']}
        ids=a['targets']
        if not isinstance(ids,list) or not ids or len(ids)!=len(set(ids)) or any(type(i)!=int or i not in visible for i in ids):raise ValueError('unseen/repeated enemy')
        g.update(mode='fight',target_ids=ids,max_ticks=400,allow_unspent=True)
        if a['hold']:g['hold_position']=list(position(s))
    elif mode=='service':
        kind=a.get('kind');cmd=dict(kind=kind,args={})
        if 'ref' in a:
            if set(a)!={'mode','kind','ref'}:raise ValueError('item service keys')
            group,idx=a['ref'].split(':');idx=int(idx)
            if group not in ('inventory','equipment','belt','floor','stock'):raise ValueError('item group')
            items=[i for i in s[group] if i['index']==idx and not i['empty']]
            if len(items)!=1 or kind not in item_actions(s,items[0],group):raise ValueError('illegal item action')
            it=items[0];cmd.update(group=group,args=dict(index=idx,identity=it['identity']))
            if kind=='buy':cmd['args']['vendor']=it['vendor']
            if kind in ('sell','repair','identify'):
                cmd['args'].update(equipped=group=='equipment',price=it[{'sell':'sale_price','repair':'repair_price','identify':'identify_price'}[kind]])
                if kind=='repair':cmd['args']['durability']=it['durability']
                if kind=='sell':cmd['args']['vendor']=s['vendor']
        elif kind in ('talk','operate'):
            if set(a)!={'mode','kind','id'} or a['id'] not in packet['observation']['legal_'+kind+'_ids']:raise ValueError('illegal nearby target')
            cmd['args']['id']=a['id']
        elif kind=='stat':
            if set(a)!={'mode','kind','attribute'} or a['attribute'] not in ('strength','dexterity','vitality','magic') or s['hero']['unspent_stats']<=0:raise ValueError('unearned/invalid stat')
            cmd['args']['attribute']=a['attribute']
        elif kind=='dismiss':
            if set(a)!={'mode','kind'} or not(s.get('dialog') or s.get('vendor') not in (None,'','none')):raise ValueError('no observed UI to dismiss')
        else:raise ValueError('unsupported service')
        g.update(mode='service',commands=[cmd],allow_visible_enemies=True,allow_unspent=True)
    elif mode=='stop':
        if set(a)!={'mode'}:raise ValueError('stop keys')
        g['mode']='stop'
    else:raise ValueError('unknown mode')
    return g,a

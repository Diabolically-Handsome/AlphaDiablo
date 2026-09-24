"""Explicit manager scope, not a learned language-to-goal representation."""
import copy
import numpy as np
from config import DIRS, digest

def validate(goal,state,state_id):
    if state_id!=digest(state):raise ValueError('stale manager observation')
    required={'id','mode','instruction','bounds','max_decisions','min_hp','allowed_actions'}
    if not required.issubset(goal):raise ValueError('incomplete goal')
    if goal['mode'] not in ('fight','retreat'):raise ValueError('unsupported manager mode')
    if not isinstance(goal['id'],str) or not goal['id']:raise ValueError('goal id')
    if type(goal['max_decisions']) is not int or not 1<=goal['max_decisions']<=64:raise ValueError('goal count')
    b=goal['bounds']
    if len(b)!=4 or any(type(v) is not int for v in b) or b[0]>b[2] or b[1]>b[3]:raise ValueError('bounds')
    if any(type(a) is not int or a not in (*range(10),12,13) for a in goal['allowed_actions']):raise ValueError('unsupported action')
    if goal['mode']=='fight' and goal.get('target_id') not in {e['id'] for e in state['enemies']}:raise ValueError('unseen target')
    if goal['mode']=='retreat' and (not isinstance(goal.get('target'),list) or len(goal['target'])!=2):raise ValueError('retreat target')
    if not 0<=goal['min_hp']<=state['hero']['max_hp_fixed']/64:raise ValueError('invalid health review threshold')
    return copy.deepcopy(goal)

def scope_mask(base,goal,state,known,canonical_target=None,route=None):
    result=np.array(base,dtype=bool,copy=True)
    allow=set(goal['allowed_actions']);result&=np.array([i in allow for i in range(15)])
    result[[10,11,14]]=False
    h=state['hero'];x,y=h['x'],h['y'];x0,y0,x1,y1=goal['bounds']
    occupied={(e['x'],e['y']) for e in state['enemies']}
    for a,(dx,dy) in enumerate(DIRS,1):
        tx,ty=x+dx,y+dy
        result[a]&=bool(x0<=tx<=x1 and y0<=ty<=y1 and known['tiles'].get(f'{tx},{ty}',False) and (tx,ty) not in occupied)
        if goal['mode']=='retreat':
            result[a]&=bool(route and (tx,ty)==tuple(route[0]))
    result[9]&=goal['mode']=='fight' and canonical_target==goal.get('target_id')
    # Potion consumption remains a neural choice, with honest native legality.
    result[12]&=h['belt_heals']>0 and h['hp_fixed']<h['max_hp_fixed']
    return result

def progress_key(state):
    h=state['hero']
    return (tuple(state['scene']),h['x'],h['y'],h['hp_fixed'],h['belt_heals'],state['kills'],
            tuple((e['id'],e['hp'],e['x'],e['y']) for e in state['enemies']))

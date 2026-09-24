"""Frozen observation/macro implementation with an audited manual owner."""
import importlib, sys, types
import numpy as np
from config import SOURCE, R16, digest
from public_view import restrict_raw, restrict_map
from goal import scope_mask

class ScopedBridge:
    COMMANDS={'act_wait','act_controller_attack_monster','act_explore_walk',
              'act_drink','act_controller_operate','act_pickup_at'}
    def __init__(self,session):
        self.s=session;self.goal=None;self.mirror_reset_available=True
        self.command_log=[]
    def known(self):return self.s.map.scenes[','.join(map(str,self.s.state['scene']))]
    def observe(self):return restrict_raw(self.s.b.observe(),self.s.state,self.known())
    def reset(self,seed):
        # Attach Python accounting to an already exactly-replayed state; this
        # is NOT another native reset and may happen only once, before effects.
        if not self.mirror_reset_available or seed!=self.s.seed:raise RuntimeError('native reset forbidden')
        self.mirror_reset_available=False
        return self.observe()
    def step(self,ticks=1):
        self.s.step(int(ticks));return self.observe()
    def local_map(self,radius):
        return restrict_map(self.s.b.local_map(radius),self.s.state,self.known(),int(radius))
    def probe_tile(self,x,y):
        # No live fogged occupancy/object queries through the old pathfinder.
        visible={(a,b) for a,b,_ in self.s.state['tiles']}
        if (x,y) in visible:
            result=dict(self.s.b.probe_tile(x,y))
            if 'monster' in result:result['monster']=any((e['x'],e['y'])==(x,y) for e in self.s.state['enemies'])
            return result
        return dict(walkable=bool(self.known()['tiles'].get(f'{x},{y}',False)),monster=False,
                    door=False,closed_door=False,hazard=False,explosive_softwall=False)
    def __getattr__(self,name):
        if name in ('init','end_game','manual_action','manual_configure') or name.startswith('configure_'):
            raise RuntimeError('native lifecycle/configuration is owned by Session: '+name)
        native=getattr(self.s.b,name)
        if name.startswith('act_'):
            if name not in self.COMMANDS:raise RuntimeError('unregistered native command:'+name)
            def command(*args,**kwargs):
                self.s.guard()
                if self.goal is None:raise RuntimeError('no strategic owner')
                if name=='act_controller_attack_monster':
                    if self.goal['mode']!='fight' or int(args[0])!=self.goal['target_id']:
                        raise RuntimeError('legacy macro attempted a different target')
                if name=='act_explore_walk':
                    x,y=map(int,args[:2]);x0,y0,x1,y1=self.goal['bounds']
                    if not(x0<=x<=x1 and y0<=y<=y1 and self.known()['tiles'].get(f'{x},{y}',False)):
                        raise RuntimeError('macro attempted unknown or out-of-scope walk')
                result=native(*args,**kwargs)
                self.s.state=self.s.b.manual_observe();self.s.map.observe(self.s.state)
                self.s.record('legacy:'+name,dict(args=list(args),kwargs=kwargs),result)
                self.command_log.append(dict(name=name,args=list(args),accepted=result,tick=self.s.state['tick']))
                return result
            return command
        if name.startswith(('debug','test_','force','inject','set_')):
            raise RuntimeError('developer/native mutation forbidden:'+name)
        return native

def install_package(bridge):
    if 'diablogym' in sys.modules:raise RuntimeError('ambiguous native package already imported')
    package=types.ModuleType('diablogym');package.__path__=[str(SOURCE/'python/diablogym')]
    package.bridge=bridge;sys.modules['diablogym']=package
    for relative in ('train','train/runs/r10-staging'):sys.path.insert(0,str(SOURCE/relative))
    env=importlib.import_module('diablogym.env');package.DiabloGymEnv=env.DiabloGymEnv
    options=importlib.import_module('diablogym.options_env');package.OptionsEnv=options.OptionsEnv
    return env,options

class Adapter:
    def __init__(self,session,package_modules):
        envmod,optmod=package_modules;self.s=session;self.bridge=envmod.bridge
        self.goal=None;self.owner_changes=[]
        self.env=optmod.OptionsEnv(max_steps=30000,workers={},drink_sovereignty=True,
            worker_observation_view='dual-v4-asymmetric-v3',start_in_dungeon=False,
            resource_protocol='off',resource_retreat='off',resource_portal='off',
            resource_sweep='off',resource_gold_grab='off',resource_emergency_stop='off',
            resource_identify='off',resource_weapon_upgrade='off',
            assets_dir='$AD_HOME/butcher_hit_score_20260913/source/build/engine/devilutionx.app/Contents/Resources',
            save_dir=str(session.out/'save'),data_dir='$AD_HOME/Library/Application Support/diasurgical/devilution')
        before=digest(session.b.manual_checkpoint());tick=session.state['tick'];seq=session.seq
        # Frozen reset normally ends the native game and configures its own
        # resource protocol. This adapter attaches to the existing manual game;
        # it resets ONLY Python wrapper bookkeeping, never native lifecycle.
        self.env.env._configure_native_resource_protocol=self.assert_native_manual_scope
        self.env.reset(seed=session.seed)
        assert digest(session.b.manual_checkpoint())==before and session.state['tick']==tick and session.seq==seq
        self.env.action_masks=lambda:np.array([True,False,False],bool)
        self.env._worker_masks=self.mask
        # Disallow any script election or reflex path, including accidental
        # future use of OptionsEnv.step rather than the explicit worker beat.
        def forbidden(*args,**kwargs):raise RuntimeError('old automatic manager/reflex is disabled')
        self.env.step=forbidden;self.env._drain=forbidden
        self.env._win_begin(0)
        self.assert_owners()
    def assert_native_manual_scope(self):
        raw=self.s.b.observe()
        resource=raw.get('resource_state',{})
        forbidden=('enabled','loot_economy','weapon_purchase_enabled','retreat_enabled','portal_enabled',
                   'identify_enabled','sweep_enabled')
        if any(resource.get(k,False) for k in forbidden):raise RuntimeError('native resource bypass remains enabled')
    def assert_owners(self):
        for name in ('resource_service','retreat_service','portal_service','sweep_service','gold_grab_service'):
            service=getattr(self.env,name,None)
            if service is not None and getattr(service,'active',False):raise RuntimeError('script took ownership:'+name)
    def set_goal(self,goal):
        if self.goal is not None:
            # Bookkeeping only. No call to legacy option dispatch/termination.
            self.env._decisions+=1
            self.env._last_tau=self.env.env._steps-self.env._win['t0']
            self.env._last_opt=0
        self.goal=goal;self.bridge.goal=goal;self.env._win_begin(0)
        self.owner_changes.append(dict(id=goal['id'],tick=self.s.state['tick']))
    def mask(self):
        if self.goal is None:return np.array([True]+[False]*14,bool)
        base,_=self.env.env.controller_action_context()
        snap=self.env.env._controller_snapshot_for(self.env.env._raw)
        target=self.env.env._canonical_engage_candidate(snap)
        target_id=target.monster_id if target is not None else None
        route=None
        if self.goal['mode']=='retreat':route=self.s.map.path(self.s.state,self.goal['target'],bounds=self.goal['bounds'])
        return scope_mask(base,self.goal,self.s.state,self.bridge.known(),target_id,route)
    def observation(self):
        self.assert_owners();before=digest(self.s.b.manual_checkpoint());seq=self.s.seq
        obs=self.env._worker_policy_observation('dual-v4-asymmetric-v3')
        mask=self.mask()
        if obs.shape!=(13012,) or not np.isfinite(obs).all():raise RuntimeError('malformed full policy view')
        from diablogym.controller_wire import DUAL_WORKER_LAYOUT
        segment=next(s for s in DUAL_WORKER_LAYOUT.segments if s.name=='action_mask')
        if not np.array_equal(obs[segment.start:segment.stop],mask.astype(np.float32)):raise RuntimeError('embedded/action mask mismatch')
        if self.s.seq!=seq or digest(self.s.b.manual_checkpoint())!=before:raise RuntimeError('observation advanced game')
        return obs,mask
    def execute(self,action):
        if not self.mask()[action]:raise RuntimeError('masked action rejected before execution')
        before=len(self.bridge.command_log)
        r,done,trunc,info,audit,_,_=self.env._beat(int(action),worker_authority=True)
        self.assert_owners()
        consumed=bool(info.get('action12_audit',{}).get('consumed'))
        if consumed:self.env._win['voluntary_drinks']+=1
        commands=self.bridge.command_log[before:]
        if any(c['name']=='act_drink' for c in commands) and action!=12:raise RuntimeError('unrequested drink')
        return dict(requested=action,executed=audit.executed_action,fuse=bool(audit.fuse_tripped),
            done=bool(done),truncated=bool(trunc),effect=info.get('action_effect_audit'),
            drink=info.get('action12_audit'),commands=commands)

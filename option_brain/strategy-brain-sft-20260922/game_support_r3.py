"""Independent native session and grounded execution. No automatic strategy."""
import sys
import time
from pathlib import Path
from common import CODE,RUN,sha,write
OLD=CODE.parent/'strategist-rl-king-continuation-20260921'
sys.path.insert(0,str(OLD/'revision3'))
from telemetry_runtime import load_bridge,DiagnosticSession
from game_executor_r2 import Executor
from protocol import digest,resolve_response
from purchase_audit import purchased_item

class Controller:
    def __init__(self,out,seed,tick_limit,deadline):
        b,build=load_bridge()
        self.s=DiagnosticSession(out,seed,tick_limit,deadline,bridge=b,bridge_info=build)
        self.s.replaying=False
        self.e=Executor(self.s);self.current=None;self.retreat_reason=None;self.last=[]
    def execute(self,reply,packet):
        before=self.s.state
        g,a=resolve_response(reply,packet,before,self.s.map.scenes,self.current)
        if a['mode']=='move' and self.s.map.path(before,a['target']) is None:raise ValueError('no known reachable path')
        if a['mode'] in ('move','fight'):self.current=a
        result=self.e.run(g);after=self.s.state
        receipt=dict(reason=result,action=a,tick_before=before['tick'],tick_after=after['tick'],
            ticks=after['tick']-before['tick'],hp_change=(after['hero']['hp_fixed']-before['hero']['hp_fixed'])/64,
            gold_change=after['hero']['gold']-before['hero']['gold'],kills_change=after['kills']-before['kills'],
            scene_before=before['scene'],scene_after=after['scene'],position_before=[before['hero']['x'],before['hero']['y']],
            position_after=[after['hero']['x'],after['hero']['y']],accepted_is_not_completed=True)
        if a['mode']=='service' and result=='completed':
            cmd=g['commands'][0];kind=cmd['kind'];args=cmd.get('args',{})
            if kind in ('buy','pickup','equip','belt','unbelt_exact'):
                dest={'buy':('inventory','belt'),'pickup':('inventory','belt'),'equip':('equipment',),
                      'belt':('belt',),'unbelt_exact':('inventory',)}[kind]
                item_id=args['identity']
                moved=any(not x['empty'] and x.get('identity')==item_id for group in dest for x in after[group])
                if kind=='buy':
                    quote=next(x for x in before['stock'] if x.get('identity')==item_id)
                    delivered=purchased_item(before,after,quote)
                    moved=delivered is not None
                    receipt['delivered_identity']=delivered['identity'] if delivered else None
                    receipt['stock_identity_may_refresh']=True
                if kind=='pickup':
                    old=next(x for x in before['floor'] if x.get('identity')==item_id)
                    if old.get('gold',0)>0:moved=receipt['gold_change']>0
                receipt['actual_item_destination_verified']=moved
                if not moved:receipt['reason']='item_not_delivered'
            if kind=='buy':
                old=next(x for x in before['stock'] if x.get('identity')==args['identity'])
                if receipt['gold_change']!=-old['price']:raise RuntimeError('native purchase cost invariant')
                receipt['actual_paid']=old['price']
            if kind=='drink':
                receipt['potion_consumed']=not any(not x['empty'] and x.get('identity')==args['identity'] for x in after['belt'])
                receipt['combat_goal_preserved']=self.current is not None and self.current['mode']=='fight'
            if kind=='stat' and after['hero']['unspent_stats']!=before['hero']['unspent_stats']-1:raise RuntimeError('earned attribute receipt mismatch')
        if before['scene']!=after['scene']:
            self.current=None  # Targets and enemy IDs are scene-local; explicitly report this boundary.
        elif result in ('completed','targets_no_longer_visible','king_dead') and a['mode'] in ('move','fight'):
            self.current=None
        if a['mode']=='move' and before['enemies']:
            self.retreat_reason=reply['reason']
        self.last.append(receipt)
        return receipt
    def close(self,reason):
        self.e.log.close();self.s.close(reason)

def stalled(receipt):
    if receipt['reason'] in ('no_known_path','target_unreachable','no_motion','two_no_progress_windows',
          'adjacent_but_native_attack_unavailable','delivery_failed_twice','item_not_delivered','specified_item_unavailable'):return True
    if receipt['reason'].startswith('rejected:'):return True
    return receipt['ticks']==0 and receipt['position_after']==receipt['position_before'] and not any(receipt[k] for k in ('hp_change','gold_change','kills_change'))





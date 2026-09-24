import sys,importlib.util
from collections import Counter
from pathlib import Path
REV=Path(__file__).resolve().parent
sys.path.insert(0,str(REV.parent))
sys.path.insert(0,str(REV.parent/'revision1'))
from execution_r1 import *

def item_counts(items):
    return Counter(tuple(i['identity']) for i in items if not i['empty'] and i.get('name')!='Gold')

def audit_sale(before,after,args,accepted):
    """Check actual instantaneous native transaction, not claimed success."""
    assert before['tick']==after['tick'],'sale advanced game clock'
    assert before['equipment']==after['equipment'],'sale changed equipment'
    assert before['belt']==after['belt'],'sale changed belt'
    assert {k:v for k,v in before['hero'].items() if k!='gold'}=={k:v for k,v in after['hero'].items() if k!='gold'},'sale changed non-gold hero state'
    old=item_counts(before['inventory']);new=item_counts(after['inventory']);target=tuple(args['identity'])
    expected=old.copy()
    delta=after['hero']['gold']-before['hero']['gold']
    if accepted:
        assert old[target]==1,'sale target must exist exactly once'
        expected[target]-=1
        assert new==+expected,'sale changed wrong items'
        assert delta==args['price'] and delta>0,'sale gold differs from quote'
    else:
        assert old==new and delta==0,'rejected sale mutated resources'

class TradeSession(Session):
    def action(self,kind,args,state_id=None):
        before=self.state
        result=super().action(kind,args,state_id)
        if kind=='sell' and args.get('vendor')=='witch':
            try:audit_sale(before,self.state,args,result['accepted'])
            except BaseException:
                write(self.out/'trade-invariant-failure.json',dict(before=before,after=self.state,args=args,receipt=result))
                raise
            with (self.out/'trade-audit.jsonl').open('a') as f:
                f.write(canonical(dict(tick=self.state['tick'],args=args,receipt=result,gold_before=before['hero']['gold'],gold_after=self.state['hero']['gold'],actual_items_verified=True))+'\n')
        return result

def load_bridge():
    build=read(ROOT/'build-r2.json')
    if sha(build['engine'])!=build['engine_sha256'] or sha(build['bridge'])!=build['bridge_sha256']:raise RuntimeError('native identity changed')
    spec=importlib.util.spec_from_file_location('_diablogym',build['bridge'])
    b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)
    return b,build

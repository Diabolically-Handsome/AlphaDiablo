import sys, importlib.util
from pathlib import Path
REV=Path(__file__).resolve().parent
sys.path.insert(0,str(REV.parent/'revision2'))
from trade_runtime import *

class DiagnosticSession(TradeSession):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.replaying=True
        self.telemetry=(self.out/'combat-telemetry.jsonl').open('x',encoding='utf-8')
    def snapshot_combat(self):
        before=digest(self.b.manual_checkpoint())
        data=self.b.manual_combat_telemetry()
        if digest(self.b.manual_checkpoint())!=before:raise RuntimeError('read-only telemetry changed game')
        return data
    def step(self,n=1):
        for _ in range(n):
            capture=scene(self.state)==(3,1)
            before=self.snapshot_combat() if capture else None
            super().step(1)
            if capture:
                after=self.snapshot_combat()
                self.telemetry.write(canonical(dict(reconstruction=self.replaying,journal_seq=self.seq-1,before=before,after=after))+'\n')
                self.telemetry.flush()
            if self.state['hero']['dead']:break
        return self.state
    def close(self,reason):
        self.telemetry.close()
        return super().close(reason)

def load_bridge():
    build=read(ROOT/'build-r3.json')
    if sha(build['engine'])!=build['engine_sha256'] or sha(build['bridge'])!=build['bridge_sha256']:raise RuntimeError('native identity changed')
    spec=importlib.util.spec_from_file_location('_diablogym',build['bridge'])
    b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)
    return b,build

from pathlib import Path
import hashlib, json, os

HERE=Path(__file__).resolve().parent
PRIOR=HERE.parent/'strategist-rl-hands-20260921'
R16=HERE.parent/'skeleton-king-dual-brain-20260921-r16'
SOURCE=Path('$AD_HOME/butcher_gear_wear_v2_20260913/source')
NETWORK_SOURCE=Path('$AD_HOME/farm_imitation_20260912/source')
MODEL=Path('$AD_HOME/butcher_preparation_v13r2_20260918/rounds/r0007/update/model.pt')
MODEL_SHA='bd85ce98491f8f1c0056d3924f4c0eb3bdb6f9cc4f04cf5b734f78815f3dc1db'
ROOT=Path('$AD_HOME/strategist_rl_king_20260921')
GPU='$AD_GPU_UUID'
DIRS=((0,0),(0,-1),(1,-1),(1,0),(1,1),(0,1),(-1,1),(-1,0),(-1,-1))
SEED=20260927
MAX_TICKS=120000
MAX_DECISIONS=30000
SECONDS=14400
PARENT_BUILD=Path('$AD_HOME/skeleton_king_dual_brain_20260920/bridge-standing-r16-1790009088006077210/build.json')

def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def canonical(v):return json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False)
def digest(v):return hashlib.sha256(canonical(v).encode()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.writing')
    with tmp.open('w',encoding='utf-8') as f:
        json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)
def identity():
    tail=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()
    return dict(pid=os.getpid(),boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),start_ticks=int(tail[19]))

PARENT_GAME=Path('$AD_HOME/skeleton_king_dual_brain_20260920/formal/segment-3')
PARENT_PAUSE_SHA='975b278043b3b290171165f2694354dd1845bbd3c065e8d5485161844d227d26'
PARENT_JOURNAL_SHA='aa2815c7e8836134be65bf2446d9e1249bb3243a184a4682b19a7b96b6d0541b'
BUILD=Path('$AD_HOME/strategist_rl_butcher_20260921/build-r3.json')
HEAD=Path('$AD_HOME/strategist_rl_butcher_20260921/goal-head.pt')
HEAD_SHA='a2cb7d3b909f93b61148eea2a4878d2eef9eef922e16fca4676e2f974fc5251c'
HISTORICAL_MINISTRAL_REQUESTS=1624

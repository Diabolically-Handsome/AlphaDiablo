"""File receipts and identity-scoped Linux deadline supervision."""
import hashlib
import json
import os
import signal
import time
from datetime import datetime
from pathlib import Path

CODE = Path(__file__).resolve().parent
RUN = Path('$AD_HOME/strategy_brain_sft_20260922')
OPT_END = datetime.fromisoformat('2026-09-22T13:00:00+00:00').timestamp()
NEW_GAME_END = datetime.fromisoformat('2026-09-22T16:30:00+00:00').timestamp()
SAVE_AT = datetime.fromisoformat('2026-09-22T16:45:00+00:00').timestamp()
HARD_END = datetime.fromisoformat('2026-09-22T17:00:00+00:00').timestamp()
GPU5090='$AD_GPU_UUID'
GPU5080='$AD_GPU2_UUID'


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def write(path, obj, exclusive=False):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if exclusive:
        with path.open('x',encoding='utf-8') as f:json.dump(obj,f,ensure_ascii=False,indent=2)
    else:
        temp=path.with_name(path.name+f'.{os.getpid()}.tmp')
        temp.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')
        os.replace(temp,path)


def identity(pid=None):
    pid=pid or os.getpid()
    fields=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()
    return {'pid':pid,'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
            'start_ticks':int(fields[19]),'process_group':int(fields[2])}


def matches(record):
    try:return all(identity(record['pid'])[k]==record[k] for k in ('pid','boot_id','start_ticks','process_group'))
    except (FileNotFoundError,ProcessLookupError):return False


def register(kind, pid=None):
    record=dict(identity(pid),kind=kind,time=time.time(),code=str(CODE))
    RUN.mkdir(exist_ok=True)
    with (RUN/'owned-processes.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(record)+'\n')
    return record


def check_time(training=False):
    if time.time() >= (OPT_END if training else HARD_END):raise TimeoutError('absolute deadline reached')
    if (RUN/'STOP').exists():raise RuntimeError('STOP requested')


def terminate_owned(record):
    if not matches(record):return False
    # Only a session leader created by this experiment may own a process group.
    if record['process_group']==record['pid']:
        os.killpg(record['pid'],signal.SIGTERM)
    else:os.kill(record['pid'],signal.SIGTERM)
    return True

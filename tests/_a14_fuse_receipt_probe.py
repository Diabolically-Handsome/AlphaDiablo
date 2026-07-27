"""a14 保险丝回执探针(回归 + 零池成本预检,2026-07-27)。

复现条件:BC-v1 教师采集流中,a14 提案在(①拍级保险丝 ②窗口级 fuse 包装)
两条拒绝路径上都必须携带显式拒绝回执——2_102 池死于①缺失、2_106 池死于
②丢传播。本探针在【已烧】池的全部种子上重放教师采集环(注册表进程内旁路;
不触 bc_worker.main、不创建任何一次性标记、零写盘),断言全程零崩,并统计
两类拒绝拍的回执在位率。

用法: python a14_fuse_receipt_probe.py [start end)...  默认两个已烧池全量。
"""
import sys, os, collections
import pathlib
_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / 'python'))
sys.path.insert(0, str(_ROOT / 'train'))

import diablogym.worker_env as we

ranges = []
args = sys.argv[1:]
if args:
    for i in range(0, len(args), 2):
        ranges.append((int(args[i]), int(args[i + 1])))
else:
    ranges = [(2_104_000, 2_104_128), (2_106_000, 2_106_128)]

lo = min(r[0] for r in ranges)
hi = max(r[1] for r in ranges)
we._CURRENT_BC_V1_RANGE = (lo, hi)   # 诊断旁路:只放宽本进程的登记检查

import bc_worker as bw
from diablogym.worker_env import WorkerWindowEnv

env = WorkerWindowEnv(str(bw.NPZ), max_steps=3000, rng_seed=0, seed_scope="bc-v1",
                      legacy_policy_observation_view=True)
stats = collections.Counter()
for lo_r, hi_r in ranges:
    for seed in range(lo_r, hi_r):
        obs, _ = env.reset(seed=seed)
        while obs is not None:
            a = bw.teacher_action(env)
            obs2, w, term, trunc, info = env.step(a)   # 任何回执洞在此抛异常
            stats["beats"] += 1
            if a == 14:
                stats["a14_requests"] += 1
                if info.get("executed_action") is None:
                    stats["a14_rejected_or_noeffect"] += 1
            obs = env.next_window() if (term or trunc) else obs2
        stats["episodes"] += 1
        if stats["episodes"] % 16 == 0:
            print(f"  {stats['episodes']} 局: {dict(stats)}", flush=True)

print(f"PASS: {dict(stats)} —— 全程零回执异常", flush=True)
os._exit(0)

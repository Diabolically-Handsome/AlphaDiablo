"""零翻转探针(PREREG-v32 案由·先行实证之落档件;锚桥证①)。

对给定工人 npz,在 R1 重生成之 BC demos 观测集(166,383 obs,脚本工人
示范窗分布——**覆盖面登记:非工人自身部署轨迹,亦非金池轨迹**)上比较
解掩 12 前后的 argmax:翻转计数、a12 logit 名次统计。报告落
train/runs/probe-zeroflip/report.json(含运行时五 sha),全文 sha 系
v32 驱动器冻结常量。
用法:.venv/bin/python train/probe_zeroflip.py
"""
import hashlib
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "train"))

from diablogym.worker_env import NumpyManager           # noqa: E402
from eval_contract import (bridge_binary_path,           # noqa: E402
                           runtime_identity)

WORKERS = {
    "v28-leg1(现任)": ROOT / "train" / "models" / "v28-worker-leg1" / "policy.npz",
    "v24-leg7(王座)": ROOT / "train" / "models" / "v24-worker-leg7" / "policy.npz",
}
DEMOS = ROOT / "train" / "runs" / "bc-worker" / "demos.npz"
OUT = ROOT / "train" / "runs" / "probe-zeroflip" / "report.json"
OUT.parent.mkdir(parents=True, exist_ok=True)


def main():
    X = np.load(DEMOS)["X"]
    rt = runtime_identity(ROOT, bridge_binary_path(ROOT))
    report = {
        "obs_count": int(len(X)),
        "demos_sha256": hashlib.sha256(DEMOS.read_bytes()).hexdigest(),
        "obs_distribution_note": "BC demos(脚本工人示范窗分布);非部署轨迹、非金池",
        "runtime_five": {
            "bridge": rt["bridge"]["sha256"], "engine": rt["engine"]["sha256"],
            "game_data": rt["content"]["game_data"]["sha256"],
            "assets": rt["content"]["assets"]["sha256"],
            "protocol": rt["python_protocol"]["sha256"]},
        "workers": {},
    }
    for name, path in WORKERS.items():
        net = NumpyManager(str(path))
        logits = np.stack([net.logits(x) for x in X])
        cur = logits.copy(); cur[:, [11, 12]] = -1e9
        new = logits.copy(); new[:, 11] = -1e9
        flips = int((cur.argmax(1) != new.argmax(1)).sum())
        rank = (logits[:, 12][:, None]
                >= np.delete(logits, [11, 12], axis=1)).sum(1)
        report["workers"][name] = {
            "npz_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "argmax_flips": flips,
            "a12_wins_all": int((logits[:, 12] >= new.max(1)).sum()),
            "a12_rank_mean": round(float(rank.mean()), 2),
            "a12_rank_max": int(rank.max()),
        }
        print(f"  {name}: flips={flips} a12全胜={report['workers'][name]['a12_wins_all']}")
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=1))
    print(f"报告落档 {OUT}")
    any_flip = any(w["argmax_flips"] for w in report["workers"].values())
    return 1 if any_flip else 0


if __name__ == "__main__":
    sys.exit(main())

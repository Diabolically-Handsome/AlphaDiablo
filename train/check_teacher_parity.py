"""v30 G-KL-W:自锚教师保真闸——教师 sd 前向对工人 npz 前向 1000 obs argmax 逐位。

用法:.venv/bin/python train/check_teacher_parity.py <teacher_sd.pt> <worker_policy.npz>
预注册口径:obs = np.random.default_rng(0).standard_normal((1000, 298)).astype(float32)。
退出码 0 = 0/1000 失配;非零 = 失配即闸死(PREREG-v30 D1)。
"""
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "train"))


def main():
    import torch as th

    from leashed_ppo import build_teacher
    from diablogym.worker_env import NumpyManager

    sd_path, npz_path = sys.argv[1], sys.argv[2]
    teacher = build_teacher(sd_path)
    net = NumpyManager(npz_path)
    obs = np.random.default_rng(0).standard_normal((1000, 298)).astype(np.float32)
    with th.no_grad():
        t = teacher(th.as_tensor(obs)).numpy()
    mism = sum(int(t[i].argmax()) != int(net.logits(obs[i]).argmax())
               for i in range(1000))
    print(f"G-KL-W parity 失配 {mism}/1000")
    sys.exit(0 if mism == 0 else 1)


main()

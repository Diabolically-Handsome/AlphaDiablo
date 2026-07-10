"""v25:把冻结工人(SB3 zip)的策略侧导出 npz(经理训练子进程免 torch)。
用法:.venv/bin/python train/export_worker_npz.py [zip路径] [npz输出]
默认:train/models/v24-worker-leg7/model.zip → 同目录 policy.npz + parity 自检。
"""
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

DEF_ZIP = ROOT / "train" / "models" / "v24-worker-leg7" / "model.zip"


def main():
    from sb3_contrib import MaskablePPO

    from diablogym.worker_env import NumpyManager

    zip_p = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else DEF_ZIP
    out = pathlib.Path(sys.argv[2]) if len(sys.argv) > 2 else zip_p.parent / "policy.npz"
    model = MaskablePPO.load(str(zip_p).replace(".zip", ""), device="cpu")
    sd = model.policy.state_dict()
    np.savez(out,
             w0=sd["mlp_extractor.policy_net.0.weight"].numpy(),
             b0=sd["mlp_extractor.policy_net.0.bias"].numpy(),
             w1=sd["mlp_extractor.policy_net.2.weight"].numpy(),
             b1=sd["mlp_extractor.policy_net.2.bias"].numpy(),
             wa=sd["action_net.weight"].numpy(),
             ba=sd["action_net.bias"].numpy())
    net = NumpyManager(str(out))          # 通用 MLP 前向,298→15 同构可用
    rng = np.random.default_rng(0)
    mask = np.ones(15, dtype=bool)
    mask[11] = mask[12] = False
    mism = 0
    for _ in range(1000):
        obs = rng.standard_normal(298).astype(np.float32)
        a_np = net.choose(obs, mask)
        a_sb, _ = model.predict(obs, action_masks=mask, deterministic=True)
        mism += int(a_np != int(a_sb))
    print(f"npz 已存 {out};parity 失配 {mism}/1000")
    if mism:
        raise SystemExit("PARITY FAIL")


if __name__ == "__main__":
    main()

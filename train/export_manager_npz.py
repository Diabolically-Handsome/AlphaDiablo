"""v23:把冻结 v22-H 经理的策略侧权重导出为 npz(numpy 前向用,子进程免 torch)。
用法:.venv/bin/python train/export_manager_npz.py
产出:train/models/v22-h-manager/policy.npz + 1000 obs 位级 parity 自检。
"""
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

MODEL_DIR = ROOT / "train" / "models" / "v22-h-manager"


def main():
    from sb3_contrib import MaskablePPO

    from diablogym.worker_env import NumpyManager

    model = MaskablePPO.load(str(MODEL_DIR / "model_final"), device="cpu")
    sd = model.policy.state_dict()
    out = MODEL_DIR / "policy.npz"
    np.savez(
        out,
        w0=sd["mlp_extractor.policy_net.0.weight"].numpy(),
        b0=sd["mlp_extractor.policy_net.0.bias"].numpy(),
        w1=sd["mlp_extractor.policy_net.2.weight"].numpy(),
        b1=sd["mlp_extractor.policy_net.2.bias"].numpy(),
        wa=sd["action_net.weight"].numpy(),
        ba=sd["action_net.bias"].numpy(),
    )
    # parity 自检:1000 个随机 303 维观测,numpy argmax ≡ SB3 predict(全掩码开)
    mgr = NumpyManager(str(out))
    rng = np.random.default_rng(0)
    mask = np.ones(3, dtype=bool)
    mismatch = 0
    for _ in range(1000):
        obs = rng.standard_normal(303).astype(np.float32)
        a_np = mgr.choose(obs, mask)
        a_sb3, _ = model.predict(obs, action_masks=mask, deterministic=True)
        mismatch += int(a_np != int(a_sb3))
    print(f"npz 已存 {out};parity 失配 {mismatch}/1000")
    if mismatch:
        raise SystemExit("PARITY FAIL —— 预注册回退:训练侧改用 SB3 predict")


if __name__ == "__main__":
    main()

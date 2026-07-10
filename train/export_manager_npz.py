"""v23/v25:把经理的策略侧权重导出为 npz(numpy 前向用,子进程免 torch)。
用法:.venv/bin/python train/export_manager_npz.py [zip路径] [npz输出]
默认:train/models/v22-h-manager/model_final.zip → 同目录 policy.npz;
自带 1000 obs 位级 parity 自检(G-KL-C 判据,失配即退出非零)。
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

    zip_p = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else MODEL_DIR / "model_final.zip"
    out = pathlib.Path(sys.argv[2]) if len(sys.argv) > 2 else zip_p.parent / "policy.npz"
    model = MaskablePPO.load(str(zip_p).replace(".zip", ""), device="cpu")
    sd = model.policy.state_dict()
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
    obs_dim = sd["mlp_extractor.policy_net.0.weight"].shape[1]
    n_act = sd["action_net.weight"].shape[0]
    rng = np.random.default_rng(0)
    mask = np.ones(n_act, dtype=bool)
    mismatch = 0
    for _ in range(1000):
        obs = rng.standard_normal(obs_dim).astype(np.float32)
        a_np = mgr.choose(obs, mask)
        a_sb3, _ = model.predict(obs, action_masks=mask, deterministic=True)
        mismatch += int(a_np != int(a_sb3))
    print(f"npz 已存 {out};parity 失配 {mismatch}/1000")
    if mismatch:
        raise SystemExit("PARITY FAIL —— 预注册回退:训练侧改用 SB3 predict")


if __name__ == "__main__":
    main()

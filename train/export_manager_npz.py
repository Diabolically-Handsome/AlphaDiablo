"""v23/v25:把经理的策略侧权重导出为 npz(numpy 前向用,子进程免 torch)。
用法:.venv/bin/python train/export_manager_npz.py [zip路径] [npz输出]
默认:train/models/v22-h-manager/model_final.zip → 同目录 policy.npz;
自带 1000 obs 位级 parity 自检(G-KL-C 判据,失配即退出非零)。
"""
import argparse
import hashlib
import io
import os
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

MODEL_DIR = ROOT / "train" / "models" / "v22-h-manager"


def main():
    from sb3_contrib import MaskablePPO

    from diablogym.worker_env import NumpyManager

    ap = argparse.ArgumentParser()
    ap.add_argument("zip_path", nargs="?", type=pathlib.Path,
                    default=MODEL_DIR / "model_final.zip")
    ap.add_argument("output", nargs="?", type=pathlib.Path)
    args = ap.parse_args()
    zip_p = args.zip_path
    out = args.output or zip_p.parent / "policy.npz"
    source_file = zip_p if zip_p.suffix.lower() == ".zip" else pathlib.Path(f"{zip_p}.zip")
    if not source_file.is_file():
        ap.error(f"checkpoint 不存在: {source_file}")
    if out.suffix.lower() != ".npz":
        ap.error("输出路径必须以 .npz 结尾")
    if out.resolve() == source_file.resolve():
        ap.error("输出路径不能覆盖源 checkpoint")
    # Capture once: loading and provenance must describe the same immutable
    # checkpoint bytes even if the path is replaced while this export runs.
    source_payload = source_file.read_bytes()
    source_sha256 = hashlib.sha256(source_payload).hexdigest()
    model = MaskablePPO.load(io.BytesIO(source_payload), device="cpu")
    sd = model.policy.state_dict()
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(f".{out.name}.{os.getpid()}.tmp.npz")
    arrays = {
        "w0": sd["mlp_extractor.policy_net.0.weight"].detach().cpu().numpy(),
        "b0": sd["mlp_extractor.policy_net.0.bias"].detach().cpu().numpy(),
        "w1": sd["mlp_extractor.policy_net.2.weight"].detach().cpu().numpy(),
        "b1": sd["mlp_extractor.policy_net.2.bias"].detach().cpu().numpy(),
        "wa": sd["action_net.weight"].detach().cpu().numpy(),
        "ba": sd["action_net.bias"].detach().cpu().numpy(),
    }
    if not all(np.isfinite(a).all() for a in arrays.values()):
        raise ValueError("策略权重含 NaN/Inf，拒绝导出")
    # parity 自检:1000 个随机观测与随机有效掩码,numpy argmax ≡ SB3 predict。
    try:
        np.savez(tmp, **arrays)
        mgr = NumpyManager(str(tmp))
        obs_dim = sd["mlp_extractor.policy_net.0.weight"].shape[1]
        n_act = sd["action_net.weight"].shape[0]
        rng = np.random.default_rng(0)
        mismatch = 0
        for _ in range(1000):
            obs = rng.standard_normal(obs_dim).astype(np.float32)
            mask = rng.random(n_act) > 0.2
            mask[int(rng.integers(0, n_act))] = True
            a_np = mgr.choose(obs, mask)
            a_sb3, _ = model.predict(obs, action_masks=mask, deterministic=True)
            mismatch += int(a_np != int(a_sb3))
        if mismatch:
            raise SystemExit("PARITY FAIL —— 预注册回退:训练侧改用 SB3 predict")
        os.replace(tmp, out)
    finally:
        tmp.unlink(missing_ok=True)
    print(f"npz 已存 {out};parity 失配 {mismatch}/1000;"
          f"源 checkpoint sha256:{source_sha256[:16]}")


if __name__ == "__main__":
    main()

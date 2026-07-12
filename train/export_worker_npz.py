"""v25:把冻结工人(SB3 zip)的策略侧导出 npz(经理训练子进程免 torch)。
用法:.venv/bin/python train/export_worker_npz.py [zip路径] [npz输出]
默认:train/models/v24-worker-leg7/model.zip → 同目录 policy.npz + parity 自检。
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

DEF_ZIP = ROOT / "train" / "models" / "v24-worker-leg7" / "model.zip"


def main():
    from sb3_contrib import MaskablePPO

    from diablogym.worker_env import NumpyManager

    ap = argparse.ArgumentParser()
    ap.add_argument("zip_path", nargs="?", type=pathlib.Path, default=DEF_ZIP)
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
    # Capture once so the loaded model and recorded source identity cannot be
    # split by a concurrent path replacement.
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
    try:
        np.savez(tmp, **arrays)
        net = NumpyManager(str(tmp))       # 通用 MLP 前向,obs/action 维度由权重推断
        rng = np.random.default_rng(0)
        obs_dim = sd["mlp_extractor.policy_net.0.weight"].shape[1]
        n_act = sd["action_net.weight"].shape[0]
        mism = 0
        for _ in range(1000):
            obs = rng.standard_normal(obs_dim).astype(np.float32)
            mask = rng.random(n_act) > 0.2
            mask[int(rng.integers(0, n_act))] = True
            a_np = net.choose(obs, mask)
            a_sb, _ = model.predict(obs, action_masks=mask, deterministic=True)
            mism += int(a_np != int(a_sb))
        if mism:
            raise SystemExit("PARITY FAIL")
        os.replace(tmp, out)
    finally:
        tmp.unlink(missing_ok=True)
    print(f"npz 已存 {out};parity 失配 {mism}/1000;"
          f"源 checkpoint sha256:{source_sha256[:16]}")


if __name__ == "__main__":
    main()

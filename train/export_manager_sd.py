"""v25:从经理 zip 抽取全量 policy state_dict(含价值头)→ .pt(M-warm 注入用)。
用法:.venv/bin/python train/export_manager_sd.py [zip] [out.pt]
默认:train/models/v22-h-manager/model_final.zip → 同目录 policy_full_sd.pt
"""
import hashlib
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

DEF = ROOT / "train" / "models" / "v22-h-manager"


def main():
    import torch
    from sb3_contrib import MaskablePPO

    zip_p = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else DEF / "model_final.zip"
    out = pathlib.Path(sys.argv[2]) if len(sys.argv) > 2 else zip_p.parent / "policy_full_sd.pt"
    model = MaskablePPO.load(str(zip_p).replace(".zip", ""), device="cpu")
    sd = {k: v.detach().clone() for k, v in model.policy.state_dict().items()}
    torch.save(sd, out)
    h = hashlib.sha256(out.read_bytes()).hexdigest()[:16]
    print(f"全量 policy sd({len(sd)} 张量,含价值头)已存 {out} sha256:{h}")


if __name__ == "__main__":
    main()

"""v25:从经理 zip 抽取全量 policy state_dict(含价值头)→ .pt(M-warm 注入用)。
用法:.venv/bin/python train/export_manager_sd.py [zip] [out.pt]
默认:train/models/v22-h-manager/model_final.zip → 同目录 policy_full_sd.pt
"""
import argparse
import hashlib
import io
import json
import os
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

DEF = ROOT / "train" / "models" / "v22-h-manager"


def main():
    import torch
    from sb3_contrib import MaskablePPO

    ap = argparse.ArgumentParser()
    ap.add_argument("zip_path", nargs="?", type=pathlib.Path,
                    default=DEF / "model_final.zip")
    ap.add_argument("output", nargs="?", type=pathlib.Path)
    args = ap.parse_args()
    zip_p = args.zip_path
    out = args.output or zip_p.parent / "policy_full_sd.pt"
    source_file = zip_p if zip_p.suffix.lower() == ".zip" else pathlib.Path(f"{zip_p}.zip")
    if not source_file.is_file():
        ap.error(f"checkpoint 不存在: {source_file}")
    if out.suffix.lower() != ".pt":
        ap.error("输出路径必须以 .pt 结尾")
    if out.resolve() == source_file.resolve():
        ap.error("输出路径不能覆盖源 checkpoint")
    # Read exactly once.  Loading from the captured bytes and hashing those
    # same bytes prevents a concurrent replacement from forging provenance.
    source_payload = source_file.read_bytes()
    source_sha256 = hashlib.sha256(source_payload).hexdigest()
    model = MaskablePPO.load(io.BytesIO(source_payload), device="cpu")
    sd = {k: v.detach().cpu().clone() for k, v in model.policy.state_dict().items()}
    if not all(torch.isfinite(v).all().item() for v in sd.values()):
        raise ValueError("policy state_dict 含 NaN/Inf，拒绝导出")
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(f".{out.name}.{os.getpid()}.tmp")
    try:
        torch.save(sd, tmp)
        os.replace(tmp, out)
    finally:
        tmp.unlink(missing_ok=True)
    artifact_sha = hashlib.sha256(out.read_bytes()).hexdigest()
    manifest = {
        "schema_version": 1,
        "artifact_type": "checkpoint_policy_state",
        "artifact_sha256": artifact_sha,
        "source_checkpoint": str(source_file.resolve()),
        "source_checkpoint_sha256": source_sha256,
        "tensor_count": len(sd),
    }
    manifest_path = out.with_name(f"{out.name}.manifest.json")
    manifest_tmp = manifest_path.with_name(f".{manifest_path.name}.{os.getpid()}.tmp")
    try:
        manifest_tmp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
        os.replace(manifest_tmp, manifest_path)
    finally:
        manifest_tmp.unlink(missing_ok=True)
    print(f"全量 policy sd({len(sd)} 张量,含价值头)已存 {out} "
          f"sha256:{artifact_sha[:16]};清单 {manifest_path}")


if __name__ == "__main__":
    main()

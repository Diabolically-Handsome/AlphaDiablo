"""v25: extract the full policy state_dict (including the value head) from a manager zip -> .pt (for M-warm injection).
Usage: .venv/bin/python train/export_manager_sd.py [zip] [out.pt]
Default: train/models/v22-h-manager/model_final.zip -> policy_full_sd.pt in the same directory
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
        ap.error(f"checkpoint does not exist: {source_file}")
    if out.suffix.lower() != ".pt":
        ap.error("output path must end with .pt")
    if out.resolve() == source_file.resolve():
        ap.error("output path must not overwrite the source checkpoint")
    # Read exactly once.  Loading from the captured bytes and hashing those
    # same bytes prevents a concurrent replacement from forging provenance.
    source_payload = source_file.read_bytes()
    source_sha256 = hashlib.sha256(source_payload).hexdigest()
    model = MaskablePPO.load(io.BytesIO(source_payload), device="cpu")
    sd = {k: v.detach().cpu().clone() for k, v in model.policy.state_dict().items()}
    if not all(torch.isfinite(v).all().item() for v in sd.values()):
        raise ValueError("policy state_dict contains NaN/Inf; refusing to export")
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
    print(f"full policy sd ({len(sd)} tensors, including the value head) saved to {out} "
          f"sha256: {artifact_sha[:16]}; manifest {manifest_path}")


if __name__ == "__main__":
    main()

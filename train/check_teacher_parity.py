"""v30 G-KL-W: self-anchored teacher fidelity gate -- logits, deployment-mask probabilities and argmax, all criteria.

Usage: .venv/bin/python train/check_teacher_parity.py <teacher_sd.pt> <worker_policy.npz>
Pre-registered criterion: obs = np.random.default_rng(0).standard_normal((1000, 298)).astype(float32).
Exit code 0 = logits/probabilities nearly bit-identical and 0/1000 mismatches; a scaled temperature must also be caught.
"""
import argparse
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "train"))


def parity_metrics(
        teacher,
        net,
        obs: np.ndarray,
        *,
        observation_view: str = "legacy-v3") -> dict:
    import torch as th

    with th.no_grad():
        teacher_logits = teacher(th.as_tensor(obs)).cpu().numpy()
    numpy_logits = np.stack([
        net.worker_logits(row, observation_view=observation_view)
        for row in obs
    ])
    raw_abs = np.abs(teacher_logits - numpy_logits)

    masks = np.ones_like(teacher_logits, dtype=bool)
    masks[:, 11] = masks[:, 12] = False
    masks[:, 14] = np.random.default_rng(1).random(len(obs)) > 0.5

    def masked_probs(logits):
        masked = np.where(masks, logits, -1e8)
        shifted = masked - masked.max(axis=1, keepdims=True)
        exp = np.exp(shifted)
        return exp / exp.sum(axis=1, keepdims=True)

    teacher_probs = masked_probs(teacher_logits)
    numpy_probs = masked_probs(numpy_logits)
    return {
        "raw_argmax_mismatch": int(
            np.count_nonzero(teacher_logits.argmax(1) != numpy_logits.argmax(1))),
        "masked_argmax_mismatch": int(
            np.count_nonzero(teacher_probs.argmax(1) != numpy_probs.argmax(1))),
        "logits_max_abs": float(raw_abs.max()),
        "logits_mean_abs": float(raw_abs.mean()),
        "masked_prob_max_abs": float(np.abs(teacher_probs - numpy_probs).max()),
    }


def parity_passes(metrics: dict) -> bool:
    """Keep the gate thresholds in one place so the CLI and regression tests cannot drift apart."""
    return (metrics["raw_argmax_mismatch"] == 0
            and metrics["masked_argmax_mismatch"] == 0
            and metrics["logits_max_abs"] < 1e-4
            and metrics["masked_prob_max_abs"] < 1e-6)


def main():
    import torch as th

    from leashed_ppo import build_teacher
    from diablogym.worker_env import (
        NumpyManager,
        WORKER_ACTION12_PERMANENTLY_MASKED,
        WORKER_OBSERVATION_VIEW_LEGACY_V3,
    )

    ap = argparse.ArgumentParser(description="check argmax agreement between a teacher state_dict and a worker NPZ")
    ap.add_argument("teacher_sd", type=pathlib.Path)
    ap.add_argument("worker_npz", type=pathlib.Path)
    args = ap.parse_args()
    for path in (args.teacher_sd, args.worker_npz):
        if not path.is_file():
            ap.error(f"file does not exist: {path}")

    sd_path, npz_path = str(args.teacher_sd), str(args.worker_npz)
    teacher = build_teacher(sd_path)
    net = NumpyManager(npz_path)
    contract = net.require_worker_contract()
    if contract["observation_view"] != WORKER_OBSERVATION_VIEW_LEGACY_V3:
        ap.error(
            "KING teacher parity is only defined for the full legacy-v3 policy observation")
    if contract["action12_mode"] != WORKER_ACTION12_PERMANENTLY_MASKED:
        ap.error(
            "KING teacher parity requires a permanently-masked action12 contract")
    obs = np.random.default_rng(0).standard_normal((1000, 298)).astype(np.float32)
    metrics = parity_metrics(
        teacher,
        net,
        obs,
        observation_view=net.worker_observation_view,
    )
    ok = parity_passes(metrics)
    print(f"G-KL-W parity: {metrics}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

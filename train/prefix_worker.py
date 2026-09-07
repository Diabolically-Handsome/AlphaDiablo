"""Fixed R16 prefix inference with persistent private CPU RNG streams.

This module is loaded only for earned-dive-suffix-v1. No prefix action or model
is attached to the learner or its optimizer. Calls are synchronous; the lock
serializes multiple prefix workers sharing a DummyVecEnv process.
"""
from contextlib import contextmanager
import hashlib
import io
from pathlib import Path
import random
import threading

import numpy as np
import torch

WORKER_PREFIX_R16_SHA256 = "7e31dc5402caed733443abb9fef383c877d93b3623b199ba4b5c6293092592f0"

_RNG_LOCK = threading.RLock()


def _capture_rng():
    state = np.random.get_state()
    return (random.getstate(), (state[0], state[1].copy(), *state[2:]),
            torch.random.get_rng_state().clone())


def _restore_rng(state):
    random.setstate(state[0])
    np.random.set_state(state[1])
    torch.random.set_rng_state(state[2])


class PrivatePrefixRNG:
    """Borrow process RNGs for one call, then restore every learner RNG exactly."""

    def __init__(self):
        with _RNG_LOCK:
            self.state = _capture_rng()

    @contextmanager
    def activate(self):
        with _RNG_LOCK:
            caller = _capture_rng()
            try:
                _restore_rng(self.state)
                yield
            finally:
                self.state = _capture_rng()
                _restore_rng(caller)


def _load_model(payload):
    from leashed_ppo import LeashedMaskablePPO
    return LeashedMaskablePPO.load(io.BytesIO(payload), env=None, device="cpu",
                                   teacher_path=None, teacher_sha256=None)


def load_prefix_worker(path, *, expected_sha256):
    """Load exactly the registered R16; preserve caller RNG even on load failure."""
    if expected_sha256 != WORKER_PREFIX_R16_SHA256:
        raise ValueError("Prefix worker must be the exact registered R16 SHA256")
    payload = Path(path).read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise ValueError("Prefix worker bytes differ from the registered R16")
    private_rng = PrivatePrefixRNG()
    with private_rng.activate():
        model = _load_model(payload)
        if tuple(model.observation_space.shape) != (13012,) or model.action_space.n != 15:
            raise ValueError("Prefix worker must preserve the 13012 observation / 15 action contract")
        model.policy.set_training_mode(False)
        model.policy.requires_grad_(False)

    def callback(obs, mask):
        if callback.on_beat is not None:
            raise RuntimeError("Fixed prefix worker cannot carry a learner/teacher on_beat hook")
        with private_rng.activate():
            action, _ = model.predict(obs, action_masks=mask, deterministic=False)
            return int(action)

    def episode_reseed(seed):
        if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or not 0 <= int(seed) < 2**32:
            raise ValueError("Prefix episode seed must be a uint32 integer")
        with private_rng.activate():
            model.set_random_seed(int(seed))

    callback.diablogym_worker_observation_view = "dual-v4-asymmetric-v3"
    callback.diablogym_worker_action12_mode = "environment-mask"
    callback.source_sha256 = expected_sha256
    callback.on_beat = None
    callback.episode_reseed = episode_reseed
    return callback

"""Custom policy architecture.

EntityAttentionExtractor (v9): AlphaStar-style perception
  - 12 scalars -> MLP
  - 8 monster tokens (dx, dy, hp%, presence bit) -> embedding + 2-layer self-attention, CLS pooling
    (padding tokens with presence bit 0 are masked; CLS is always present, so "no monster in view" is handled naturally)
  - 11x11x2 map -> small CNN
  concatenated -> 256-dim features. About 600k parameters (~13x the MLP champion, still a small model).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

_N_SCALAR = 12
_N_TOKENS = 8
_TOKEN_DIM = 4
_MAP_SIDE = 11
_MAP_CH = 2
_N_EXTRA = 9  # v13 potions 4 dims + v14 equipment 4 dims + v19 strength gauge 1 dim (tail of the vector, merged into the scalar branch)


class EntityAttentionExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        expected = _N_SCALAR + _N_TOKENS * _TOKEN_DIM + _MAP_CH * _MAP_SIDE * _MAP_SIDE + _N_EXTRA
        if observation_space.shape != (expected,):
            raise ValueError(f"observation layout mismatch: {observation_space.shape} != ({expected},)")
        if features_dim <= 0:
            raise ValueError(f"features_dim must be > 0, got {features_dim}")

        self.scalar_net = nn.Sequential(nn.Linear(_N_SCALAR + _N_EXTRA, 64), nn.ReLU())

        self.token_embed = nn.Linear(_TOKEN_DIM, 64)
        self.cls = nn.Parameter(torch.zeros(1, 1, 64))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=64, nhead=4, dim_feedforward=128,
            dropout=0.0, batch_first=True, norm_first=True,
        )
        # norm_first=True is incompatible with PyTorch's nested-tensor fast path; the current
        # version silently falls back to the regular path and warns. Pin the choice explicitly so that
        # heuristic changes in future versions cannot quietly change the training execution path.
        self.attn = nn.TransformerEncoder(
            encoder_layer, num_layers=2, enable_nested_tensor=False)

        self.map_net = nn.Sequential(
            nn.Conv2d(_MAP_CH, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * _MAP_SIDE * _MAP_SIDE, 128), nn.ReLU(),
        )

        self.out = nn.Sequential(nn.Linear(64 + 64 + 128, features_dim), nn.ReLU())

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        b = obs.shape[0]
        scalars = obs[:, :_N_SCALAR]
        tokens = obs[:, _N_SCALAR:_N_SCALAR + _N_TOKENS * _TOKEN_DIM].reshape(b, _N_TOKENS, _TOKEN_DIM)
        map_lo = _N_SCALAR + _N_TOKENS * _TOKEN_DIM
        map_hi = map_lo + _MAP_CH * _MAP_SIDE * _MAP_SIDE
        maps = obs[:, map_lo:map_hi].reshape(b, _MAP_CH, _MAP_SIDE, _MAP_SIDE)
        extras = obs[:, -_N_EXTRA:]  # v13 potions 4 + v14 equipment 4 + v19 strength gauge 1

        s = self.scalar_net(torch.cat([scalars, extras], dim=1))

        t = self.token_embed(tokens)
        cls = self.cls.expand(b, -1, -1)
        seq = torch.cat([cls, t], dim=1)  # (b, 1+8, 64)
        pad = tokens[:, :, 3] == 0        # presence bit 0 -> padding
        mask = torch.cat([torch.zeros(b, 1, dtype=torch.bool, device=obs.device), pad], dim=1)
        ent = self.attn(seq, src_key_padding_mask=mask)[:, 0]  # CLS pooling

        m = self.map_net(maps)
        return self.out(torch.cat([s, ent, m], dim=1))

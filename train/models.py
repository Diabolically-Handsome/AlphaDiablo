"""自定义策略架构。

EntityAttentionExtractor(v9):AlphaStar 式感知
  - 12 标量 → MLP
  - 8 个怪物 token(dx,dy,hp%,存在位)→ 嵌入 + 2 层 self-attention,CLS 池化
    (存在位=0 的 padding token 被 mask;CLS 恒在,天然处理"视野无怪")
  - 11×11×2 地图 → 小 CNN
  拼接 → 256 维特征。约 60 万参数(MLP 冠军的 ~13 倍,仍是小模型)。
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
_N_EXTRA = 8  # v13 药 4 维 + v14 装备 4 维(向量尾部,并入标量分支)


class EntityAttentionExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        expected = _N_SCALAR + _N_TOKENS * _TOKEN_DIM + _MAP_CH * _MAP_SIDE * _MAP_SIDE + _N_EXTRA
        assert observation_space.shape[0] == expected, \
            f"观测布局不匹配: {observation_space.shape[0]} != {expected}"

        self.scalar_net = nn.Sequential(nn.Linear(_N_SCALAR + _N_EXTRA, 64), nn.ReLU())

        self.token_embed = nn.Linear(_TOKEN_DIM, 64)
        self.cls = nn.Parameter(torch.zeros(1, 1, 64))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=64, nhead=4, dim_feedforward=128,
            dropout=0.0, batch_first=True, norm_first=True,
        )
        self.attn = nn.TransformerEncoder(encoder_layer, num_layers=2)

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
        extras = obs[:, -_N_EXTRA:]  # v13 药 4 维 + v14 装备 4 维(向量尾部)

        s = self.scalar_net(torch.cat([scalars, extras], dim=1))

        t = self.token_embed(tokens)
        cls = self.cls.expand(b, -1, -1)
        seq = torch.cat([cls, t], dim=1)  # (b, 1+8, 64)
        pad = tokens[:, :, 3] == 0        # 存在位=0 → padding
        mask = torch.cat([torch.zeros(b, 1, dtype=torch.bool, device=obs.device), pad], dim=1)
        ent = self.attn(seq, src_key_padding_mask=mask)[:, 0]  # CLS 池化

        m = self.map_net(maps)
        return self.out(torch.cat([s, ent, m], dim=1))

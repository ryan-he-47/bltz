"""Conditional byte decoder head (docs/01-design.md §5.4, D1).

(h, delta) -> 256-way byte logits. delta is ALWAYS a relative offset (1..k_max),
never an absolute position (simple_point_cloud v6 static-fan lesson: with
absolute positions the decoder learns to ignore the query). Given one h, all
delta queries are independent -> one batched forward covers the whole horizon.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ConditionalByteHead(nn.Module):
    def __init__(
        self,
        d_in: int = 768,
        k_max: int = 48,
        d_delta: int = 64,
        hidden: int = 1536,
        depth: int = 4,
    ):
        super().__init__()
        self.delta_emb = nn.Embedding(k_max, d_delta)
        layers: list[nn.Module] = [nn.Linear(d_in + d_delta, hidden), nn.SiLU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.SiLU()]
        self.mlp = nn.Sequential(*layers)
        self.out = nn.Linear(hidden, 256)

    def forward(self, h: torch.Tensor, delta_idx: torch.Tensor) -> torch.Tensor:
        """h (N, d_in), delta_idx (N,) long in [0, k_max) (i.e. delta - 1) -> (N, 256)."""
        x = torch.cat([h, self.delta_emb(delta_idx)], dim=-1)
        return self.out(self.mlp(x))

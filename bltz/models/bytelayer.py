"""Learnable byte-layer input encoder (2026-09-27 用户设计):

Joint-trained input path replacing frozen-lambda + adapter. Design spec
(user): a NORMAL transformer layer at backbone width — NOT a Set Transformer —
with the byte-level plumbing unchanged:
  * single-patch receptive field: self-attention runs within one unit's bytes
  * local position indices: byte position inside the patch (0..l_max-1)
  * attention pooling (PMA) unchanged: one seed vector pools the patch bytes

byte_emb + pos_emb (both d_model) -> one TransformerEncoderLayer (pre-norm,
GELU, no dropout, 4x FFN) -> PMA(k=1) -> (B, S, d_model). Trainable end-to-end
from the MDN NLL; the frozen AE keeps generating prediction TARGETS only.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .encoder import PMA


class ByteLayerEncoder(nn.Module):
    def __init__(self, l_max: int = 32, d_model: int = 768, nhead: int = 12):
        super().__init__()
        self.l_max = l_max
        self.byte_emb = nn.Embedding(256, d_model)
        self.pos_emb = nn.Embedding(l_max, d_model)
        self.layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward=4 * d_model,
            dropout=0.0, activation="gelu", batch_first=True, norm_first=True,
        )
        self.pool = PMA(d_model, nhead, k=1)

    def forward(self, byte_ids: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        """(B, S, L) byte ids + pad mask -> (B, S, d_model) patch tokens."""
        B, S, L = byte_ids.shape
        pos = torch.arange(L, device=byte_ids.device)
        x = self.byte_emb(byte_ids.clamp(0, 255)) + self.pos_emb(pos)  # (B,S,L,d)
        pm = pad_mask.reshape(B * S, L)
        x = self.layer(x.reshape(B * S, L, -1), src_key_padding_mask=pm)
        h = self.pool(x, key_padding_mask=pm)  # (B*S, 1, d)
        return h.reshape(B, S, -1)

"""Learnable byte-layer input encoder (2026-09-27 用户设计, 09-28 稳定性修复):

Joint-trained input path replacing frozen-lambda + adapter. Design spec
(user): a NORMAL transformer layer at backbone width — NOT a Set Transformer —
with the byte-level plumbing unchanged:
  * single-patch receptive field: self-attention runs within one unit's bytes
  * local position indices: byte position inside the patch (0..l_max-1)
  * attention pooling (learnable-seed) unchanged in mechanism

Numerics fix (589611 事故复盘): the original PMA/MAB pooling fed RAW q/kv to
attention (AE-inference design — no backward there). Under trained fp16 the
seed grew 75x and the residual stream is unbounded -> fp16 backward activation
grads overflowed the 65504 cliff (sporadic inf from step 168, permanent NaN at
12850; GradScaler x65536 amplifies). AttnPool keeps the SAME mechanism but
pre-normalizes q and kv so logits and their gradients stay bounded.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttnPool(nn.Module):
    """Learnable-seed attention pooling, PRE-NORM q/kv (fp16-safe)."""

    def __init__(self, d_model: int, nhead: int):
        super().__init__()
        self.seed = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.ln_q = nn.LayerNorm(d_model)
        self.ln_kv = nn.LayerNorm(d_model)
        self.q = nn.Linear(d_model, d_model, bias=False)
        self.kv = nn.Linear(d_model, 2 * d_model, bias=False)
        self.o = nn.Linear(d_model, d_model, bias=False)
        self.ln_o = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor) -> torch.Tensor:
        """x (N, L, d), mask (N, L) True=pad -> (N, 1, d)."""
        N = x.shape[0]
        q = self.q(self.ln_q(self.seed.expand(N, -1, -1)))
        k, v = self.kv(self.ln_kv(x)).chunk(2, dim=-1)
        att = F.scaled_dot_product_attention(
            q, k, v, attn_mask=~key_padding_mask.unsqueeze(1)
        )
        return self.ln_o(self.o(att) + self.seed.expand(N, -1, -1))


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
        self.pool = AttnPool(d_model, nhead)

    def forward(self, byte_ids: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        """(B, S, L) byte ids + pad mask -> (B, S, d_model) patch tokens."""
        B, S, L = byte_ids.shape
        pos = torch.arange(L, device=byte_ids.device)
        x = self.byte_emb(byte_ids.clamp(0, 255)) + self.pos_emb(pos)  # (B,S,L,d)
        pm = pad_mask.reshape(B * S, L)
        x = self.layer(x.reshape(B * S, L, -1), src_key_padding_mask=pm)
        h = self.pool(x, pm)  # (B*S, 1, d)
        return h.reshape(B, S, -1)

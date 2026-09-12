"""Set Transformer patch encoder (docs/01-design.md §5.2, D2/D8).

Per patch: byte embedding (256+PAD) concat learned within-patch position
embedding -> Linear -> ISAB x L -> PMA(k=1) -> Linear to backbone d_model.
Sees ONLY the current patch's bytes; cross-patch context is the backbone's job.
ISAB/PMA adapted from simple_point_cloud with key_padding_mask threading.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ..data import PAD_ID


class MAB(nn.Module):
    """Multihead Attention Block: H = LN(Q + MHA(Q, KV)); out = LN(H + FFN(H))."""

    def __init__(self, d_model: int, nhead: int, ff_mult: int = 4):
        super().__init__()
        self.mha = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_mult * d_model),
            nn.GELU(),
            nn.Linear(ff_mult * d_model, d_model),
        )

    def forward(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h = self.ln1(
            q
            + self.mha(q, kv, kv, key_padding_mask=key_padding_mask, need_weights=False)[0]
        )
        return self.ln2(h + self.ff(h))


class ISAB(nn.Module):
    """Induced Set Attention Block: X -> MAB(X, MAB(I, X)) with m inducing points."""

    def __init__(self, d_model: int, nhead: int, inducing: int):
        super().__init__()
        self.induce = nn.Parameter(torch.randn(1, inducing, d_model) * 0.02)
        self.mab1 = MAB(d_model, nhead)
        self.mab2 = MAB(d_model, nhead)

    def forward(
        self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        h = self.mab1(
            self.induce.expand(x.shape[0], -1, -1), x, key_padding_mask=key_padding_mask
        )
        return self.mab2(x, h)


class PMA(nn.Module):
    """Pooling by Multihead Attention with k learned seed vectors."""

    def __init__(self, d_model: int, nhead: int, k: int = 1):
        super().__init__()
        self.seed = nn.Parameter(torch.randn(1, k, d_model) * 0.02)
        self.mab = MAB(d_model, nhead)

    def forward(
        self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        return self.mab(
            self.seed.expand(x.shape[0], -1, -1), x, key_padding_mask=key_padding_mask
        )


class SetTransformerEncoder(nn.Module):
    """byte_ids (B, S, L) + pad_mask -> patch latents (B, S, d_out)."""

    def __init__(
        self,
        l_max: int = 16,
        d_byte: int = 384,
        d_pos: int = 64,
        d_model: int = 512,
        nhead: int = 8,
        layers: int = 2,
        inducing: int = 16,
        d_out: int = 768,
    ):
        super().__init__()
        self.l_max = l_max
        self.byte_emb = nn.Embedding(PAD_ID + 1, d_byte, padding_idx=PAD_ID)
        self.pos_emb = nn.Embedding(l_max, d_pos)
        self.in_proj = nn.Linear(d_byte + d_pos, d_model)
        self.blocks = nn.ModuleList(
            [ISAB(d_model, nhead, inducing) for _ in range(layers)]
        )
        self.pool = PMA(d_model, nhead, k=1)
        self.out_proj = nn.Linear(d_model, d_out)

    def forward(self, byte_ids: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        B, S, L = byte_ids.shape
        pos = torch.arange(L, device=byte_ids.device)
        x = torch.cat(
            [self.byte_emb(byte_ids), self.pos_emb(pos).expand(B, S, L, -1)], dim=-1
        )
        x = self.in_proj(x).reshape(B * S, L, -1)
        kpm = pad_mask.reshape(B * S, L)
        for blk in self.blocks:
            x = blk(x, key_padding_mask=kpm)
        pooled = self.pool(x, key_padding_mask=kpm).squeeze(1)
        return self.out_proj(pooled).reshape(B, S, -1)

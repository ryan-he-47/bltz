"""Token baseline for the Stage-2 twin control (docs/02-construction-plan.md §3).

GPT-2 tokenizer (50,257) + the SAME Backbone class as ByteFieldLM + untied
lm_head. Sharing the backbone isolates the comparison to I/O (byte-patch
encoder/conditional head vs token embedding/lm_head) at matched trunk.
Loss: standard next-token cross-entropy on shifted labels. BPB comparability
follows docs/01 §9 D-5 (neither side sees inside the current unit).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import Backbone


class TokenLlama(nn.Module):
    def __init__(
        self,
        d_model: int = 768,
        nhead: int = 12,
        layers: int = 12,
        ffn_mult: int = 4,
        max_len: int = 1024,
        vocab: int = 50257,
    ):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab, d_model)
        self.backbone = Backbone(
            d_model=d_model, nhead=nhead, layers=layers, ffn_mult=ffn_mult, max_len=max_len
        )
        self.lm_head = nn.Linear(d_model, vocab, bias=False)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """idx (B, T) long -> logits (B, T, vocab)."""
        return self.lm_head(self.backbone(self.tok_emb(idx)))


def next_token_loss(model: TokenLlama, idx: torch.Tensor) -> torch.Tensor:
    """Standard shifted next-token CE (fp32 logits for stability)."""
    logits = model(idx[:, :-1])
    return F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]).float(), idx[:, 1:].reshape(-1)
    )

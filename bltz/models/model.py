"""BltzLM: encoder -> backbone -> conditional head wiring."""
from __future__ import annotations

import torch
import torch.nn as nn

from .backbone import Backbone
from .encoder import SetTransformerEncoder
from .head import ConditionalByteHead, FiLMByteHead

_HEADS = {"concat": ConditionalByteHead, "film": FiLMByteHead}


class BltzLM(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        m = cfg.model
        self.encoder = SetTransformerEncoder(
            l_max=cfg.segment.l_max,
            d_byte=m.d_byte,
            d_pos=m.d_pos,
            d_model=m.d_enc,
            nhead=m.enc_heads,
            layers=m.enc_layers,
            inducing=m.inducing,
            d_out=m.d_model,
        )
        self.backbone = Backbone(
            d_model=m.d_model,
            nhead=m.bb_heads,
            layers=m.bb_layers,
            ffn_mult=m.ffn_mult,
            max_len=cfg.data.n_patches + 8,
            grad_ckpt=bool(m.get("grad_ckpt", False)),
        )
        head_cls = _HEADS[str(m.get("head_type", "concat"))]
        self.head = head_cls(
            d_in=m.d_model,
            k_max=m.k_max,
            d_delta=m.head_delta_dim,
            hidden=m.head_hidden,
            depth=m.head_depth,
        )

    def forward(self, byte_ids: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        """(B, S, L) -> h (B, S, d_model): backbone output at every patch position."""
        latents = self.encoder(byte_ids, pad_mask)
        return self.backbone(latents)

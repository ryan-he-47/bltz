"""BltzLMv2 (docs/31): frozen AE -> wide nonlinear adapter -> backbone -> MDN.

Gradient hard-cut: the AE receives ONLY reconstruction gradients (trained by
scripts/train_ae.py, then frozen here); the backbone receives ONLY the MDN NLL,
with lambda inputs detached. The adapter is deliberately nonlinear-wide
(48 -> 2048 -> GELU -> 768): a bare linear lift would pin the first layer's
input rank at <=48 and force layer-1's FFN to double as the adapter
(2026-09-17 拍板, no control arm). BOS: learned embedding in lambda space.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .autoencoder import ByteStringAE
from .backbone import Backbone
from .mdn import MDNHead


class BltzLMv2(nn.Module):
    def __init__(self, cfg, ae: ByteStringAE):
        super().__init__()
        m = cfg.model
        self.ae = ae.eval()
        self.ae.requires_grad_(False)  # hard gradient cut (CoSE Sec.3.3, hardened)
        self.l_max = ae.l_max
        d_emb = int(m.d_emb)
        self.adapter = nn.Sequential(
            nn.LayerNorm(d_emb),
            nn.Linear(d_emb, int(m.adapter_width)),
            nn.GELU(),
            nn.Linear(int(m.adapter_width), int(m.d_model)),
        )
        self.bos = nn.Parameter(torch.randn(1, 1, d_emb) * 0.02)
        self.backbone = Backbone(
            d_model=int(m.d_model), nhead=int(m.bb_heads), layers=int(m.bb_layers),
            ffn_mult=int(m.ffn_mult), max_len=int(cfg.data.n_patches) + 8,
            grad_ckpt=bool(m.get("grad_ckpt", False)),
        )
        self.head = MDNHead(
            d_in=int(m.d_model), hidden=int(m.mdn_hidden),
            n_comp=int(m.n_comp), d_emb=d_emb, sigma_floor=float(m.sigma_floor),
        )

    def encode_units(self, byte_ids: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        """(B, S, L) -> λ (B, S, d_emb), detached (never trains the AE here)."""
        B, S, L = byte_ids.shape
        with torch.no_grad():
            lam = self.ae.encoder(byte_ids.reshape(B * S, L), pad_mask.reshape(B * S, L))
        return lam.view(B, S, -1).detach()

    def forward(self, byte_ids: torch.Tensor, pad_mask: torch.Tensor):
        """Returns (h, lam): backbone states h (B, S, d_model) predicting the
        next embedding, and detached targets lam (B, S, d_emb)."""
        lam = self.encode_units(byte_ids, pad_mask)
        B, S, D = lam.shape
        x = torch.cat([self.bos.expand(B, 1, D), lam[:, :-1]], dim=1)  # BOS, λ_1..λ_{S-1}
        return self.backbone(self.adapter(x)), lam

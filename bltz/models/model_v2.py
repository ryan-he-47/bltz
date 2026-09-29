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
from .bytelayer import ByteLayerEncoder
from .mdn import MDNHead


class BltzLMv2(nn.Module):
    def __init__(self, cfg, ae: ByteStringAE):
        super().__init__()
        m = cfg.model
        self.ae = ae.eval()
        self.ae.requires_grad_(False)  # hard gradient cut (CoSE Sec.3.3, hardened)
        self.l_max = ae.l_max
        d_emb = int(m.d_emb)
        ae_d = int(ae.encoder.out_proj.out_features)
        if ae_d != d_emb:
            raise ValueError(
                f"AE embedding dim mismatch: ckpt has d_emb={ae_d}, v2 config wants "
                f"{d_emb} — retrain the AE with the right d_emb (573890 事故: "
                f"ae.yaml 基础配置未从起跑组 32 改到 48)"
            )
        # input_mode (2026-09-27 用户设计): frozen = lambda + adapter (original);
        # learnable = ByteLayerEncoder joint-trained from the NLL — the frozen
        # AE then serves ONLY as prediction-target generator.
        self.learnable_input = str(m.get("input_mode", "frozen")) == "learnable"
        if self.learnable_input:
            # 2026-09-29 严谨对照规格: 与冻结 AE 编码器同构, 梯度联通,
            # 无 48 维瓶颈 (proj + LN 到 d_model). 目标仍由冻结 AE 提供.
            self.in_enc = ByteLayerEncoder(ae.encoder, int(m.d_model))
        elif bool(m.get("adapter_linear", False)):
            # adapter_linear (2026-09-25 control arm): LN + bare Linear lift. The
            # nonlinear-wide default exists because "a bare linear lift would pin
            # layer-1's input rank at <=48 and force layer-1's FFN to double as
            # the adapter" (2026-09-17 拍板) — this arm tests that claim.
            self.adapter = nn.Sequential(
                nn.LayerNorm(d_emb),
                nn.Linear(d_emb, int(m.d_model)),
            )
        else:
            self.adapter = nn.Sequential(
                nn.LayerNorm(d_emb),
                nn.Linear(d_emb, int(m.adapter_width)),
                nn.GELU(),
                nn.Linear(int(m.adapter_width), int(m.d_model)),
            )
        bos_dim = int(m.d_model) if self.learnable_input else d_emb
        self.bos = nn.Parameter(torch.randn(1, 1, bos_dim) * 0.02)
        self.backbone = Backbone(
            d_model=int(m.d_model), nhead=int(m.bb_heads), layers=int(m.bb_layers),
            ffn_mult=int(m.ffn_mult), max_len=int(cfg.data.n_patches) + 8,
            grad_ckpt=bool(m.get("grad_ckpt", False)),
        )
        self.head = MDNHead(
            d_in=int(m.d_model), hidden=int(m.mdn_hidden),
            n_comp=int(m.n_comp), d_emb=d_emb, sigma_floor=float(m.sigma_floor),
            swiglu=bool(m.get("mdn_swiglu", False)), depth=int(m.get("mdn_depth", 2)),
        )

    def encode_units(self, byte_ids: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        """(B, S, L) -> λ (B, S, d_emb), detached (never trains the AE here).
        Chunked: AE encodes each unit independently, so batching in 8192-row
        chunks is numerically identical — but avoids CUDA "invalid
        configuration argument" in MHA when B*S explodes at scale-run batch
        sizes (49k rows broke the kernel launch, 589472 事故)."""
        B, S, L = byte_ids.shape
        flat = byte_ids.reshape(B * S, L)
        pm = pad_mask.reshape(B * S, L)
        outs = []
        with torch.no_grad():
            for i in range(0, B * S, 8192):
                outs.append(self.ae.encoder(flat[i : i + 8192], pm[i : i + 8192]))
        lam = torch.cat(outs, dim=0)
        return lam.view(B, S, -1).detach()

    def backbone_input(self, byte_ids: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        """(B, S, L) -> backbone-ready (B, S, d_model): BOS + shifted input
        tokens, mode-dependent (frozen: lambda->adapter | learnable: byte layer)."""
        B, S, _ = byte_ids.shape
        if self.learnable_input:
            toks = self.in_enc(byte_ids, pad_mask)
            return torch.cat([self.bos.expand(B, 1, -1), toks[:, :-1]], dim=1)
        lam = self.encode_units(byte_ids, pad_mask)
        x = torch.cat([self.bos.expand(B, 1, -1), lam[:, :-1]], dim=1)
        return self.adapter(x)

    def forward(self, byte_ids: torch.Tensor, pad_mask: torch.Tensor):
        """Returns (h, lam): backbone states h (B, S, d_model) predicting the
        next embedding, and detached targets lam (B, S, d_emb)."""
        lam = self.encode_units(byte_ids, pad_mask)  # targets: always frozen AE
        B, S, D = lam.shape
        if self.learnable_input:
            toks = self.in_enc(byte_ids, pad_mask)
            x = torch.cat([self.bos.expand(B, 1, -1), toks[:, :-1]], dim=1)
        else:
            x = torch.cat([self.bos.expand(B, 1, D), lam[:, :-1]], dim=1)  # BOS, λ_1..λ_{S-1}
            x = self.adapter(x)
        return self.backbone(x), lam

"""Causal transformer backbone over patch latents (docs/01-design.md §5.3).

Llama-style: RMSNorm, RoPE, SwiGLU FFN, pre-norm residuals, SDPA with
is_causal=True (flash path on sm89). No token embedding table anywhere —
input is the encoder's patch latents, already at d_model.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.float() * scale).type_as(x) * self.weight


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Interleaved RoPE. x (B, nh, T, hd); cos/sin (T, hd/2) fp32."""
    x1, x2 = x[..., ::2], x[..., 1::2]
    cos = cos[None, None]
    sin = sin[None, None]
    out1 = x1 * cos - x2 * sin
    out2 = x1 * sin + x2 * cos
    return torch.stack([out1, out2], dim=-1).flatten(-2).type_as(x)


class Block(nn.Module):
    def __init__(self, d_model: int, nhead: int, ffn_mult: int):
        super().__init__()
        self.nhead = nhead
        self.hd = d_model // nhead
        self.ln1 = RMSNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.o = nn.Linear(d_model, d_model, bias=False)
        self.ln2 = RMSNorm(d_model)
        d_ff = ffn_mult * d_model
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        h = self.ln1(x)
        qkv = self.qkv(h).reshape(B, T, 3, self.nhead, self.hd).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = apply_rope(q, cos[:T], sin[:T])
        k = apply_rope(k, cos[:T], sin[:T])
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.o(a.transpose(1, 2).reshape(B, T, D))
        h2 = self.ln2(x)
        return x + self.w2(F.silu(self.w1(h2)) * self.w3(h2))


class Backbone(nn.Module):
    """Patch-latent causal transformer. Input (B, T, d_model) -> (B, T, d_model)."""

    def __init__(
        self,
        d_model: int = 768,
        nhead: int = 12,
        layers: int = 12,
        ffn_mult: int = 4,
        max_len: int = 520,
        rope_base: float = 500000.0,
    ):
        super().__init__()
        self.rope_base = rope_base
        self.hd = d_model // nhead
        self.blocks = nn.ModuleList(
            [Block(d_model, nhead, ffn_mult) for _ in range(layers)]
        )
        self.final_norm = RMSNorm(d_model)
        cos, sin = self._build_rope(max_len)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def _build_rope(self, n: int):
        inv_freq = 1.0 / (
            self.rope_base ** (torch.arange(0, self.hd, 2).float() / self.hd)
        )
        t = torch.arange(n).float()
        freqs = torch.outer(t, inv_freq)
        return freqs.cos(), freqs.sin()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.shape[1]
        if T > self.rope_cos.shape[0]:  # lazy extension (inference growth)
            cos, sin = self._build_rope(T)
            self.rope_cos = cos.to(x.device)
            self.rope_sin = sin.to(x.device)
        cos = self.rope_cos.to(x.device)
        sin = self.rope_sin.to(x.device)
        for blk in self.blocks:
            x = blk(x, cos, sin)
        return self.final_norm(x)

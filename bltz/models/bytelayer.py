"""Learnable byte-layer input encoder — 2026-09-29 用户严谨对照规格:

与冻结 AE 编码器**同构**(byte+pos 拼接 -> in_proj -> N× 自注意力 -> 池化),
唯一区别:
  (1) 梯度联通可训练,只服务骨干预测(预测目标仍由独立冻结 AE 提供);
  (2) 取消信息瓶颈:AE 的 out_norm+out_proj(width->48) 换成
      proj(width->d_model) + LayerNorm(d_model)(常规层间连接).

结构镜像自加载的 AE ckpt(inducing=0 -> 标准自注意力层,逐层同宽同头数).
池化用 AttnPool(pre-norm 种子注意力)而非 AE 的 PMA:机制等价(可学习种子
注意力池化),但 PMA 裸 q/kv 是推理部件,训练态 fp16 反向必溢出(589611
悬崖:seed 涨 75×,12850 步永久 NaN;GradScaler x65536 放大)。
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
    """Mirror of ByteStringEncoder (autoencoder.py) minus the d_emb bottleneck.

    `ae_enc` = the loaded frozen AE's encoder: its widths/heads/layers/l_max
    are mirrored 1:1 so the ONLY deltas vs the frozen-input control arm are
    gradient flow and the bottleneck replacement (user spec 2026-09-29).
    """

    def __init__(self, ae_enc: nn.Module, d_model: int):
        super().__init__()
        d_byte = ae_enc.byte_emb.embedding_dim
        d_pos = ae_enc.pos_emb.embedding_dim
        width = ae_enc.in_proj.out_features
        n_layers = len(ae_enc.layers)
        first = ae_enc.layers[0]
        if not isinstance(first, nn.TransformerEncoderLayer):
            raise ValueError(
                "learnable mirror only supports standard self-attention AEs "
                "(inducing=0); ISAB AEs are inference-only ancestors")
        heads = first.self_attn.num_heads
        self.l_max = ae_enc.pos_emb.num_embeddings
        self.byte_emb = nn.Embedding(256, d_byte)
        self.pos_emb = nn.Embedding(self.l_max, d_pos)
        self.in_proj = nn.Linear(d_byte + d_pos, width)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                width, heads, dim_feedforward=4 * width,
                dropout=0.0, activation="gelu", batch_first=True,
                norm_first=True,
            )
            for _ in range(n_layers)
        ])
        self.pool = AttnPool(width, heads)  # fp16-safe PMA twin
        self.out_norm = nn.LayerNorm(width)
        # bottleneck replacement: projection + adaptive LN (no 48-dim squeeze)
        self.proj = nn.Linear(width, d_model)
        self.proj_norm = nn.LayerNorm(d_model)

    def forward(self, byte_ids: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        """(B, S, L) byte ids + pad mask -> (B, S, d_model) patch tokens."""
        B, S, L = byte_ids.shape
        pos = torch.arange(L, device=byte_ids.device).expand(B * S, L)
        ids = byte_ids.reshape(B * S, L).clamp(0, 255)
        pm = pad_mask.reshape(B * S, L)
        x = torch.cat([self.byte_emb(ids), self.pos_emb(pos)], dim=-1)
        x = self.in_proj(x)
        for lyr in self.layers:
            x = lyr(x, src_key_padding_mask=pm)
        h = self.pool(x, pm)[:, 0]
        return self.proj_norm(self.proj(self.out_norm(h))).reshape(B, S, -1)

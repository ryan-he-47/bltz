"""v2 objective: MDN NLL of the next patch embedding (docs/31 §2.3).

Teacher forcing on detached encoder latents; mean NLL over non-boundary
positions. Document-boundary masking (2026-10-03 用户拍板, bltz/masking.py):
attention is blocked across documents inside a window (block-diagonal causal
bias), and the unlearnable cross-document target — a document's first unit
predicted from the previous document's context — is excluded from the mean.
`doc_start` (B, S) comes from the cache's unit_flag; when absent (legacy
diagnostic paths) the loss falls back to the plain unmasked mean.
"""
from __future__ import annotations

import torch

from .masking import boundary_weight, doc_attn_bias


def mdn_nll_loss(model, batch: dict[str, torch.Tensor], cfg=None) -> torch.Tensor:
    ds = batch.get("doc_start")
    bias = doc_attn_bias(ds) if ds is not None else None
    h, lam = model(batch["byte_ids"], batch["pad_mask"], attn_bias=bias)
    nll = model.head.nll(h, lam)  # (B, S) per-position
    if ds is None:
        return nll.mean()
    w = boundary_weight(ds)
    return (nll * w).sum() / w.sum().clamp_min(1.0)

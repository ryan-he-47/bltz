"""v2 objective: MDN NLL of the next patch embedding (docs/31 §2.3).

Teacher forcing on detached encoder latents; mean NLL over all S positions.
Positions are never masked (v2 cache has no padding inside a sequence —
ShardReader guarantees S full units).
"""
from __future__ import annotations

import torch


def mdn_nll_loss(model, batch: dict[str, torch.Tensor], cfg) -> torch.Tensor:
    h, lam = model(batch["byte_ids"], batch["pad_mask"])
    return model.head.nll(h, lam).mean()

"""MTP loss with patch-prior geometric decay (docs/01-design.md §6, D4).

For every patch position j (h_j sees patches <= j):
  K_j = len(p_{j+1}) + ... + len(p_{j+n})   (truncated at sequence end)
  query (h_j, delta) for delta = 1..K_j -> CE against the true byte at
  flat[ends[j] + delta - 1]
  weight w(delta) = lam^(k(delta)-1), k(delta) = which future patch (1..n)
  the target byte falls in; loss = sum(w * CE) / sum(w).

mtp_targets() is the single source of truth for query/target construction —
shared by the loss and by scripts/diag_delta.py (D-1 per-delta curves).
Query evaluation is chunked (loss.query_chunk) to bound memory — same trick
as MoB_Head's --loss-chunk.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def mtp_targets(
    batch: dict[str, torch.Tensor], n: int, lam: float, k_max: int
) -> dict[str, torch.Tensor]:
    """Query grid + targets + weights for the MTP objective.

    Query (j, delta=d) is valid iff d < K_j[j]; its target is the byte at
    flat[ends[j] + d]; its weight is lam^(k-1), k = target patch offset (1..n).
    Returns j_idx (S-1,), delta (k_max,), targets/valid/w (B, S-1, k_max),
    K_j (B, S-1)."""
    flat, ends, patch_pos = batch["flat"], batch["ends"], batch["patch_pos"]
    B, S = ends.shape
    dev = ends.device

    j_idx = torch.arange(S - 1, device=dev)
    jn = torch.clamp(j_idx + n, max=S - 1)  # (S-1,)
    K_j = ends[:, jn] - ends[:, : S - 1]  # (B, S-1)

    delta = torch.arange(k_max, device=dev)  # delta index; delta = idx + 1
    tpos = ends[:, : S - 1, None] + delta[None, None, :]  # (B, S-1, k_max)
    valid = delta[None, None, :] < K_j[:, :, None]
    tpos_c = torch.minimum(tpos, (batch["flat_len"] - 1).view(B, 1, 1))

    targets = torch.gather(flat, 1, tpos_c.reshape(B, -1)).reshape(B, S - 1, k_max)
    qpatch = torch.gather(patch_pos, 1, tpos_c.reshape(B, -1)).reshape(B, S - 1, k_max)
    k = qpatch - j_idx[None, :, None]  # target patch offset, 1..n where valid
    # r: byte position of the target WITHIN its own patch (0-based). D-1
    # stratification: k carries the cross-patch degradation trend, r carries
    # the within-patch entropy profile (word-start concentration).
    starts = torch.roll(ends, 1, dims=1)
    starts[:, 0] = 0
    qstarts = torch.gather(starts, 1, qpatch.reshape(B, -1)).reshape(B, S - 1, k_max)
    r = tpos_c - qstarts
    w = torch.where(valid, lam ** (k.float() - 1.0), torch.zeros(B, S - 1, k_max, device=dev))
    return {
        "j_idx": j_idx, "delta": delta, "targets": targets, "valid": valid,
        "w": w, "K_j": K_j, "k": k, "r": r,
    }


def mtp_loss(model, batch: dict[str, torch.Tensor], cfg) -> torch.Tensor:
    n = cfg.loss.n_patches_ahead
    lam = float(cfg.loss.lam)
    k_max = cfg.model.k_max

    B, S, _ = batch["byte_ids"].shape
    dev = batch["byte_ids"].device
    h = model(batch["byte_ids"], batch["pad_mask"])[:, : S - 1]  # (B, S-1, d)
    q = mtp_targets(batch, n, lam, k_max)
    targets, valid, w = q["targets"], q["valid"], q["w"]

    b_all = (
        torch.arange(B, device=dev).view(B, 1, 1).expand(B, S - 1, k_max).reshape(-1)
    )
    j_all = q["j_idx"].view(1, S - 1, 1).expand(B, S - 1, k_max).reshape(-1)
    d_all = q["delta"].view(1, 1, k_max).expand(B, S - 1, k_max).reshape(-1)
    t_all = targets.reshape(-1)
    w_all = w.reshape(-1)
    sel_all = valid.reshape(-1).nonzero(as_tuple=True)[0]

    chunk = int(cfg.loss.query_chunk)
    loss_sum = torch.zeros((), device=dev)
    w_sum = torch.zeros((), device=dev)
    for s in range(0, sel_all.numel(), chunk):
        sel = sel_all[s : s + chunk]
        h_q = h[b_all[sel], j_all[sel]]
        logits = model.head(h_q, d_all[sel])
        ce = F.cross_entropy(logits.float(), t_all[sel], reduction="none")
        loss_sum = loss_sum + (ce * w_all[sel]).sum()
        w_sum = w_sum + w_all[sel].sum()
    return loss_sum / w_sum.clamp_min(1e-8)

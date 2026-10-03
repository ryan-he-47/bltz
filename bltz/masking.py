"""Document-boundary masking (2026-10-03 用户拍板): block-diagonal causal
attention by document + boundary loss exclusion — no cache rebuild.

Signals come from the cache's `unit_flag` (ShardWriter wrote 1 on every
document's FIRST unit; read path returns it via sequence_units_flags /
sequence_flags). Per window:
  * attention: units from different documents never attend to each other
    (bias 0 iff j<=i and doc(i)==doc(j), else -inf) — Megatron-style
    reset-attention semantics;
  * loss: the unlearnable cross-document target (a document's first unit,
    predicted from the previous document's context) is dropped from the NLL
    mean (boundary_weight = 0 at doc-start positions).

dtype: the bias is built in fp16 (mem-efficient SDPA consumes it directly);
Backbone casts to the activation dtype when they differ (fp32 eval/bf16).
"""
from __future__ import annotations

import torch


def doc_attn_bias(doc_start: torch.Tensor, dtype: torch.dtype = torch.float16) -> torch.Tensor:
    """(B, S) bool doc-start flags -> additive attention bias (B, 1, S, S).

    0 where j <= i and unit j is in the same document as unit i; -inf otherwise.
    Rows before the first doc-start (window starting mid-document, id = -1)
    form one segment — the window's opening context, masked causally as usual.
    """
    B, S = doc_start.shape
    ids = doc_start.long().cumsum(1) - 1  # inclusive cumsum - 1: same id per doc
    same = ids.unsqueeze(2) == ids.unsqueeze(1)  # (B, S, S)
    same &= torch.ones(S, S, dtype=torch.bool, device=doc_start.device).tril()
    neg = torch.full((), float("-inf"), dtype=dtype, device=doc_start.device)
    zero = torch.zeros((), dtype=dtype, device=doc_start.device)
    return torch.where(same.unsqueeze(1), zero, neg)


def boundary_weight(doc_start: torch.Tensor) -> torch.Tensor:
    """(B, S) doc-start flags -> fp32 loss weights: 0 at doc-start targets
    (cross-document prediction is unlearnable), 1 elsewhere."""
    return (~doc_start).float()


def doc_seqlens_from_start(doc_start: torch.Tensor) -> list[list[int]]:
    """(B, S) bool doc-start flags -> per-window document segment lengths.

    Segment order is window-major, position-minor — i.e. exactly the flat
    order of the (B*S) attention layout used by the xformers backend."""
    ds = doc_start.detach().cpu()
    S = int(ds.shape[1])
    out: list[list[int]] = []
    for b in range(ds.shape[0]):
        starts = ds[b].nonzero(as_tuple=False).flatten().tolist()
        bounds = [0] + [s for s in starts if 0 < s < S] + [S]
        out.append([b2 - b1 for b1, b2 in zip(bounds[:-1], bounds[1:])])
    return out


def xf_bias_from_start(doc_start: torch.Tensor):
    """xformers BlockDiagonalCausalMask over the flattened sequence (2026-10-03
    效率优化: structured mask, no (B,1,S,S) matrix; kernel skips cross-block
    pairs. V100 bench: 64ms vs 267ms dense vs 127ms plain causal at
    B=24/S=4096/H=16/D=64 fwd+bwd)."""
    from xformers.ops.fmha.attn_bias import BlockDiagonalCausalMask

    flat = [L for win in doc_seqlens_from_start(doc_start) for L in win]
    return BlockDiagonalCausalMask.from_seqlens(flat)


def make_attn_bias(doc_start: torch.Tensor | None, mode: str = "auto"):
    """Forward-decision factory for the boundary attention bias.

    mode="xformers": structured mask (ImportError if unavailable — fail loud);
    "dense": (B,1,S,S) fp16 additive bias (portable fallback);
    "auto": xformers when importable, else dense. None doc_start -> None."""
    if doc_start is None:
        return None
    if mode in ("auto", "xformers"):
        try:
            return xf_bias_from_start(doc_start)
        except ImportError:
            if mode == "xformers":
                raise
    return doc_attn_bias(doc_start)

"""Attention-mask backend benchmark (2026-10-03, docs/39 #38 follow-up).

Decides the boundary-masking efficiency question: at production attention
shapes (B=24, S=4096, H=16, D=64, fp16, V100 sm70), compare
  1. plain causal SDPA (is_causal=True)          -- theoretical floor
  2. dense (B,1,S,S) fp16 bias SDPA               -- current implementation
  3. xformers BlockDiagonalCausalMask             -- structured (no matrix)
plus an output cross-check between 2 and 3.

  srun -p stingy --exclude=gpu-v100s-06 --gres=gpu:1 --cpus-per-task=8 \
       --mem=32G --time=0:20:00 python scripts/bench_attn_mask.py
"""
from __future__ import annotations

import random
import time

import torch
import torch.nn.functional as F

B, S, H, D = 24, 4096, 16, 64
DEV = "cuda"
DT = torch.float16
ITERS, WARMUP = 10, 3


def make_seqlens() -> list[list[int]]:
    rng = random.Random(0)
    out = []
    for _ in range(B):
        nseg = rng.choice([3, 4, 4, 5, 6])
        cuts = sorted(rng.sample(range(64, S - 64), nseg - 1))
        lens = [c2 - c1 for c1, c2 in zip([0] + cuts, cuts + [S])]
        out.append(lens)
    return out


def dense_bias_from_seqlens(seqs: list[list[int]]) -> torch.Tensor:
    doc_start = torch.zeros(B, S, dtype=torch.bool, device=DEV)
    for b, lens in enumerate(seqs):
        pos = 0
        for L in lens:
            doc_start[b, pos] = True
            pos += L
    ids = doc_start.long().cumsum(1) - 1
    same = ids.unsqueeze(2) == ids.unsqueeze(1)
    same &= torch.ones(S, S, dtype=torch.bool, device=DEV).tril()
    neg = torch.full((), float("-inf"), dtype=DT, device=DEV)
    zero = torch.zeros((), dtype=DT, device=DEV)
    return torch.where(same.unsqueeze(1), zero, neg)


def bench(fn, iters: int = ITERS) -> float:
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


def main() -> None:
    print(f"cap {torch.cuda.get_device_capability()}")
    torch.manual_seed(0)
    q = torch.randn(B, H, S, D, device=DEV, dtype=DT)
    k = torch.randn(B, H, S, D, device=DEV, dtype=DT)
    v = torch.randn(B, H, S, D, device=DEV, dtype=DT)
    seqs = make_seqlens()
    n_seg = sum(len(s) for s in seqs)
    print(f"segs/window {n_seg / B:.1f}, mean seg len {B * S / n_seg:.0f}")

    def f_plain() -> None:
        q1 = q.clone().requires_grad_(True)
        k1 = k.clone().requires_grad_(True)
        v1 = v.clone().requires_grad_(True)
        F.scaled_dot_product_attention(q1, k1, v1, is_causal=True).sum().backward()

    ms = bench(f_plain)
    torch.cuda.reset_peak_memory_stats()
    f_plain()
    torch.cuda.synchronize()
    print(f"[plain causal]         {ms:7.2f} ms  peak {torch.cuda.max_memory_allocated() / 2**30:5.2f} GiB")

    def f_dense() -> None:
        bias = dense_bias_from_seqlens(seqs)
        q1 = q.clone().requires_grad_(True)
        k1 = k.clone().requires_grad_(True)
        v1 = v.clone().requires_grad_(True)
        F.scaled_dot_product_attention(q1, k1, v1, attn_mask=bias).sum().backward()

    ms = bench(f_dense)
    torch.cuda.reset_peak_memory_stats()
    f_dense()
    torch.cuda.synchronize()
    print(f"[dense bias fp16]      {ms:7.2f} ms  peak {torch.cuda.max_memory_allocated() / 2**30:5.2f} GiB")

    from xformers.ops import memory_efficient_attention
    from xformers.ops.fmha.attn_bias import BlockDiagonalCausalMask

    flat = [L for s in seqs for L in s]

    def f_xf() -> None:
        bias = BlockDiagonalCausalMask.from_seqlens(flat)
        q1 = q.transpose(1, 2).reshape(1, B * S, H, D).contiguous().requires_grad_(True)
        k1 = k.transpose(1, 2).reshape(1, B * S, H, D).contiguous().requires_grad_(True)
        v1 = v.transpose(1, 2).reshape(1, B * S, H, D).contiguous().requires_grad_(True)
        memory_efficient_attention(q1, k1, v1, attn_bias=bias).sum().backward()

    try:
        ms = bench(f_xf)
        torch.cuda.reset_peak_memory_stats()
        f_xf()
        torch.cuda.synchronize()
        print(f"[xformers BGM]         {ms:7.2f} ms  peak {torch.cuda.max_memory_allocated() / 2**30:5.2f} GiB")
    except Exception as e:  # noqa: BLE001
        print(f"[xformers BGM] FAILED: {type(e).__name__}: {str(e)[:400]}")

    try:
        bias = dense_bias_from_seqlens(seqs)
        o_d = F.scaled_dot_product_attention(q, k, v, attn_mask=bias)
        o_x = memory_efficient_attention(
            q.transpose(1, 2).reshape(1, B * S, H, D).contiguous(),
            k.transpose(1, 2).reshape(1, B * S, H, D).contiguous(),
            v.transpose(1, 2).reshape(1, B * S, H, D).contiguous(),
            attn_bias=BlockDiagonalCausalMask.from_seqlens(flat),
        ).reshape(B, S, H, D).transpose(1, 2)
        diff = (o_d.float() - o_x.float()).abs().max().item()
        print(f"[cross-check] max|dense-xf| = {diff:.4f} (fp16 tolerance)")
    except Exception as e:  # noqa: BLE001
        print(f"[cross-check] FAILED: {type(e).__name__}: {str(e)[:300]}")


if __name__ == "__main__":
    main()

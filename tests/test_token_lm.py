"""Script-style test for the token baseline (Stage-2 twin control).
Run: python tests/test_token_lm.py"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from bytefield.models.token_lm import TokenLlama, next_token_loss


def check(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def main() -> None:
    torch.manual_seed(0)
    model = TokenLlama(d_model=128, nhead=4, layers=2, ffn_mult=4, max_len=64).cuda()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"token baseline params (tiny): {n_params/1e6:.2f}M")

    idx = torch.randint(0, 50257, (4, 32), device="cuda")
    with torch.no_grad():
        logits = model(idx)
    check(logits.shape == (4, 32, 50257), f"logits shape: {logits.shape}")
    check(torch.isfinite(logits).all(), "non-finite logits")

    loss = next_token_loss(model, idx)
    expected = torch.log(torch.tensor(50257.0))
    check(abs(loss.item() - expected.item()) < 0.2, f"init loss {loss.item():.3f} != ln V {expected:.3f}")
    print(f"init loss: {loss.item():.4f} (ln 50257 = {expected:.4f})")

    # causality: changing future tokens must not change earlier-position logits
    idx2 = idx.clone()
    idx2[:, 16:] = torch.randint(0, 50257, (4, 16), device="cuda")
    with torch.no_grad():
        l1 = model(idx)
        l2 = model(idx2)
    check(torch.allclose(l1[:, :16], l2[:, :16], atol=1e-5), "LEAK: early logits changed with future tokens")
    check(not torch.allclose(l1[:, 16:], l2[:, 16:]), "logits should change where tokens changed")
    print("causality (shared Backbone): OK")

    # single-batch overfit: must dive
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    batch = torch.randint(0, 50257, (4, 32), device="cuda")
    first = None
    for step in range(150):
        loss = next_token_loss(model, batch)
        if first is None:
            first = loss.item()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    last = loss.item()
    print(f"overfit: {first:.4f} -> {last:.4f}")
    check(last < 0.5, f"overfit failed: {last:.4f}")

    print("test_token_lm.py: ALL PASS")


if __name__ == "__main__":
    main()

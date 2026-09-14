"""Script-style test for the FiLM-conditioned byte head.
Checks: (1) identity-at-init (zero-init conditioner -> every delta gives the
same output at init); (2) delta path is trainable (mod receives nonzero grads);
(3) shapes/dtype through BltzLM with model.head_type=film; (4) fp16 stability
smoke (no inf/nan under autocast). Run: python tests/test_film_head.py"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from bltz.config import load
from bltz.models import BltzLM
from bltz.models.head import FiLMByteHead


def check(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def main() -> None:
    torch.manual_seed(0)
    d_in, k_max, hidden = 64, 8, 128
    head = FiLMByteHead(d_in=d_in, k_max=k_max, d_delta=16, hidden=hidden, depth=4, film_layers=2).cuda()
    h = torch.randn(32, d_in, device="cuda")

    # (1) identity at init: zero-init conditioner -> output independent of delta
    out_d0 = head(h, torch.zeros(32, dtype=torch.long, device="cuda"))
    out_d5 = head(h, torch.full((32,), 5, dtype=torch.long, device="cuda"))
    check(
        torch.allclose(out_d0, out_d5, atol=1e-6),
        f"FiLM not identity at init: max diff {(out_d0 - out_d5).abs().max().item()}",
    )
    check(out_d0.shape == (32, 256), f"bad output shape {out_d0.shape}")

    # (2) delta path trainable: backward gives nonzero grads to the conditioners
    loss = head(h, torch.randint(0, k_max, (32,), device="cuda")).square().mean()
    loss.backward()
    g = head.films[0].mod.weight.grad
    check(g is not None and g.abs().sum().item() > 0, "no grad into FiLM conditioner")
    head.zero_grad(set_to_none=True)

    # (3) BltzLM with head_type=film builds and forwards at smoke scale
    cfg = load("configs/smoke.yaml")
    cfg.model.head_type = "film"
    model = BltzLM(cfg).cuda()
    n_film = sum(p.numel() for p in model.head.parameters())
    B, S, L = 2, cfg.data.n_patches, cfg.segment.l_max
    byte_ids = torch.randint(0, 256, (B, S, L), device="cuda")
    pad_mask = torch.zeros(B, S, L, dtype=torch.bool, device="cuda")
    h_out = model(byte_ids, pad_mask)
    check(h_out.shape == (B, S, cfg.model.d_model), f"bad BltzLM(film) output {h_out.shape}")
    logits = model.head(h_out[:, :-1].reshape(-1, cfg.model.d_model), torch.randint(0, cfg.model.k_max, (B * (S - 1),), device="cuda"))
    check(logits.shape == (B * (S - 1), 256) and torch.isfinite(logits).all(), "bad film logits")

    # (4) fp16 autocast stability smoke
    with torch.autocast("cuda", dtype=torch.float16):
        h16 = model(byte_ids, pad_mask)
        lg = model.head(h16[:, :-1].reshape(-1, cfg.model.d_model), torch.zeros(B * (S - 1), dtype=torch.long, device="cuda"))
    check(torch.isfinite(lg.float()).all(), "non-finite fp16 film logits")

    # concat head untouched (default config)
    model_c = BltzLM(load("configs/smoke.yaml")).cuda()
    from bltz.models.head import ConditionalByteHead
    check(isinstance(model_c.head, ConditionalByteHead), "default head_type drifted")

    print(f"film head params (smoke cfg): {n_film}")
    print("test_film_head.py: ALL PASS")


if __name__ == "__main__":
    main()

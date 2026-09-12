"""Script-style model test: shapes, patch-level causality (no leak), delta
conditioning, loss finiteness + gradient flow. Run: python tests/test_model.py"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from bltz.config import load
from bltz.data import build_batch
from bltz.models import BltzLM
from bltz.objectives import mtp_loss
from bltz.segment import Segmenter


def check(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def main() -> None:
    cfg = load("configs/smoke.yaml")
    torch.manual_seed(0)
    seg = Segmenter(cfg.segment.l_max, cfg.segment.p_split, cfg.segment.p_merge)
    model = BltzLM(cfg).cuda()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"smoke model params: {n_params/1e6:.2f}M")

    texts = [
        "The quick brown fox jumps over the lazy dog. " * 20,
        "bltz encodes patches with a set transformer. " * 20,
    ]
    batch = build_batch(texts, seg, cfg.data.n_patches, cfg.segment.l_max)
    batch = {k: v.cuda() for k, v in batch.items()}

    with torch.no_grad():
        h = model(batch["byte_ids"], batch["pad_mask"])
    B = batch["byte_ids"].shape[0]
    check(h.shape == (B, cfg.data.n_patches, cfg.model.d_model), f"h shape: {h.shape}")
    check(torch.isfinite(h).all(), "h has non-finite values")

    # ---- no-leak: bytes of patch j+1 must NOT change h_j ----
    text_a = "Alpha beta gamma delta epsilon zeta eta theta. " * 20
    text_b = "Alpha beta gamma delta XXXXXXXX YYYYYYYY ZZZZZZZZ. " * 20
    ba = build_batch([text_a], seg, cfg.data.n_patches, cfg.segment.l_max)
    bb = build_batch([text_b], seg, cfg.data.n_patches, cfg.segment.l_max)
    ba = {k: v.cuda() for k, v in ba.items()}
    bb = {k: v.cuda() for k, v in bb.items()}
    # find first patch index whose bytes differ
    differ = (ba["byte_ids"] != bb["byte_ids"]).any(dim=-1)[0]
    j0 = int(differ.nonzero(as_tuple=True)[0][0])
    with torch.no_grad():
        ha = model(ba["byte_ids"], ba["pad_mask"])[0]
        hb = model(bb["byte_ids"], bb["pad_mask"])[0]
    same_before = torch.allclose(ha[:j0], hb[:j0], atol=1e-5)
    check(same_before, f"LEAK: h[<first-differ-patch={j0}] changed with future bytes")
    check(not torch.allclose(ha[j0], hb[j0], atol=1e-4), f"h[{j0}] should change when its own patch bytes change")
    print(f"no-leak OK (first differing patch index = {j0})")

    # ---- delta conditioning: same h, different delta -> different logits ----
    with torch.no_grad():
        h_last = ha[-1:]
        logits_d0 = model.head(h_last, torch.tensor([0], device="cuda"))
        logits_d5 = model.head(h_last, torch.tensor([5], device="cuda"))
    check(not torch.allclose(logits_d0, logits_d5), "head ignores delta (static-fan cheat signature)")
    print("delta conditioning OK")

    # ---- loss: finite scalar, gradients reach all three modules ----
    loss = mtp_loss(model, batch, cfg)
    check(loss.dim() == 0 and torch.isfinite(loss), f"bad loss: {loss}")
    model.zero_grad(set_to_none=True)
    loss.backward()
    for name, mod in [("encoder", model.encoder), ("backbone", model.backbone), ("head", model.head)]:
        gnorm = sum(p.grad.float().pow(2).sum().item() for p in mod.parameters() if p.grad is not None) ** 0.5
        check(gnorm > 0, f"{name} gets no gradient")
        print(f"grad norm {name}: {gnorm:.4f}")
    print(f"loss on random init batch: {loss.item():.4f} (ln 256 = {torch.log(torch.tensor(256.0)):.4f})")

    print("test_model.py: ALL PASS")


if __name__ == "__main__":
    main()

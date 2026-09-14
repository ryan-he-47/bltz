"""Script-style test for the T1' length-grouped encoder path.
Core acceptance: grouped forward must equal the padded-canvas forward
elementwise (padded slots were masked out of every computation, so removing
them is an exact transform). Also: determinism and degenerate fallback.
Run: python tests/test_enc_grouped.py"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from bltz.models.encoder import SetTransformerEncoder


def check(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def rand_batch(rng: torch.Generator, B: int, S: int, L: int, dev: str = "cuda"):
    """Random patches with variable real lengths 1..L (right-padded)."""
    lens = torch.randint(1, L + 1, (B, S), generator=rng, device=dev)
    byte_ids = torch.full((B, S, L), 256, dtype=torch.long, device=dev)
    pad_mask = torch.ones(B, S, L, dtype=torch.bool, device=dev)
    for b in range(B):
        for j in range(S):
            ln = int(lens[b, j])
            byte_ids[b, j, :ln] = torch.randint(0, 256, (ln,), generator=rng, device=dev)
            pad_mask[b, j, :ln] = False
    return byte_ids, pad_mask


def main() -> None:
    torch.manual_seed(0)
    g = torch.Generator(device="cuda").manual_seed(0)
    kw = dict(l_max=16, d_byte=64, d_pos=16, d_model=96, nhead=4, layers=2, inducing=8, d_out=128)
    enc_g = SetTransformerEncoder(grouped=True, **kw).cuda().eval()
    enc_c = SetTransformerEncoder(grouped=False, **kw).cuda().eval()
    enc_c.load_state_dict(enc_g.state_dict())

    # (1) equivalence on random variable-length batches (two shapes)
    for B, S in ((2, 24), (3, 7)):
        byte_ids, pad_mask = rand_batch(g, B, S, 16)
        with torch.no_grad():
            a = enc_g(byte_ids, pad_mask)
            b = enc_c(byte_ids, pad_mask)
        check(a.shape == (B, S, 128), f"bad shape {a.shape}")
        md = (a - b).abs().max().item()
        check(torch.allclose(a, b, atol=1e-4), f"grouped != canvas, max diff {md}")
        print(f"equivalence B={B} S={S}: max diff {md:.2e}")

    # (1b) extreme: all patches exactly l_max (no padding at all)
    lens_full = torch.ones(2, 8, dtype=torch.long, device="cuda") * 16
    byte_ids = torch.randint(0, 256, (2, 8, 16), generator=g, device="cuda")
    pad_mask = torch.zeros(2, 8, 16, dtype=torch.bool, device="cuda")
    with torch.no_grad():
        a = enc_g(byte_ids, pad_mask)
        b = enc_c(byte_ids, pad_mask)
    check(torch.allclose(a, b, atol=1e-4), "full-length batch mismatch")

    # (2) determinism
    byte_ids, pad_mask = rand_batch(g, 2, 16, 16)
    with torch.no_grad():
        a1 = enc_g(byte_ids, pad_mask)
        a2 = enc_g(byte_ids, pad_mask)
    check(torch.equal(a1, a2), "grouped path not deterministic")

    # (3) degenerate fallback: a zero-length patch row -> canvas path, no crash
    byte_ids, pad_mask = rand_batch(g, 2, 8, 16)
    pad_mask[0, 3] = True  # patch (0,3) becomes zero-length
    byte_ids[0, 3] = 256
    with torch.no_grad():
        out = enc_g(byte_ids, pad_mask)
    check(out.shape == (2, 8, 128), "fallback did not run")

    print("test_enc_grouped.py: ALL PASS")


if __name__ == "__main__":
    main()

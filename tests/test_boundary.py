"""Boundary-masking tests (2026-10-03 文档分块注意力掩码 + 边界 loss 掩码).

Covers: cache doc-flag readback, verify-resplit flag propagation, bias
structure, block-diagonal causal equivalence (masked full-window forward ==
per-document chunked forwards), cross-document invariance, loss weights.
Run:  python tests/test_boundary.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from bltz.masking import boundary_weight, doc_attn_bias  # noqa: E402
from bltz.models.backbone import Backbone  # noqa: E402
from bltz.shards import ShardReader, ShardWriter  # noqa: E402
from bltz.verify import VerifyResplitter  # noqa: E402


def test_shard_flags() -> None:
    # ignore_cleanup_errors: Windows keeps .npy mmap handles open until GC
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        w = ShardWriter(str(Path(tmp) / "shard-00000"), l_max=32)
        w.add_document_units([b"aa", b"bb", b"cc"])
        w.add_document_units([b"dd", b"ee"])
        w.add_document_units([b"ff", b"gg", b"hh", b"ii"])
        w.close({"type": "word"})
        r = ShardReader(tmp)
        units, flags = r.sequence_units_flags(0, 9)
        assert units == [b"aa", b"bb", b"cc", b"dd", b"ee", b"ff", b"gg", b"hh", b"ii"]
        assert flags == [True, False, False, True, False, True, False, False, False], flags
        assert r.sequence_units(0, 9) == units
        assert r.sequence_flags(0, 9) == flags
    print("ok  shard doc-flag readback")


def test_verify_propagation() -> None:
    v = VerifyResplitter.__new__(VerifyResplitter)
    for k in ("n_in", "n_flag", "n_piece", "n_floor", "n_floor_fail",
              "n_out", "n_verify_rows", "_calls"):
        setattr(v, k, 0)
    v.log_every = 0
    # b"AB" was resplit into two pieces; b"CD"/b"XY" pass through.
    v._cache = {b"AB": (b"A", b"B"), b"CD": (b"CD",), b"XY": (b"X", b"Y")}
    units, fl = v.process_batch(
        [[b"AB", b"CD", b"XY"]], [[True, False, False]])
    assert units == [[b"A", b"B", b"CD", b"X", b"Y"]], units
    assert fl == [[True, False, False, False, False]], fl
    # legacy call shape (no flags) keeps the old return type
    assert v.process_batch([[b"CD"]]) == [[b"CD"]]
    print("ok  verify-resplit flag propagation")


def _ref_allowed(doc_start: torch.Tensor) -> torch.Tensor:
    B, S = doc_start.shape
    ids = doc_start.long().cumsum(1) - 1
    a = torch.zeros(B, S, S, dtype=torch.bool)
    for b in range(B):
        for i in range(S):
            for j in range(S):
                a[b, i, j] = (j <= i) and (ids[b, i] == ids[b, j])
    return a


def test_bias_structure() -> None:
    torch.manual_seed(0)
    for ds_list in ([1, 0, 0, 1, 0, 1, 0],
                    [0, 0, 1, 0, 0],
                    [1, 0, 1, 0],
                    [1, 1, 0]):
        ds = torch.tensor([ds_list], dtype=torch.bool)
        bias = doc_attn_bias(ds, dtype=torch.float32)
        assert bias.shape == (1, 1, len(ds_list), len(ds_list))
        allowed = bias[0, 0] == 0
        ref = _ref_allowed(ds)[0]
        assert torch.equal(allowed, ref), (ds_list, allowed.int(), ref.int())
        blocked = bias[0, 0] == float("-inf")
        assert torch.equal(blocked, ~ref)
    print("ok  bias structure (block-diagonal causal, incl. mid-doc window start)")


def test_backbone_equivalence() -> None:
    torch.manual_seed(3)
    bb = Backbone(d_model=16, nhead=2, layers=2, ffn_mult=2, max_len=64).eval()
    x = torch.randn(1, 6, 16)

    # single document: masked bias must equal the plain causal path
    ds1 = torch.tensor([[1, 0, 0, 0, 0, 0]], dtype=torch.bool)
    out_plain = bb(x)
    out_mask = bb(x, attn_bias=doc_attn_bias(ds1, dtype=torch.float32))
    assert torch.allclose(out_plain, out_mask, atol=1e-6), \
        (out_plain - out_mask).abs().max()

    # two documents [0:3) [3:6): masked window == chunked per-document forwards
    ds2 = torch.tensor([[1, 0, 0, 1, 0, 0]], dtype=torch.bool)
    out_full = bb(x, attn_bias=doc_attn_bias(ds2, dtype=torch.float32))
    out_a = bb(x[:, :3])
    out_b = bb(x[:, 3:])
    assert torch.allclose(out_full[:, :3], out_a, atol=1e-6), \
        (out_full[:, :3] - out_a).abs().max()
    assert torch.allclose(out_full[:, 3:], out_b, atol=1e-6), \
        (out_full[:, 3:] - out_b).abs().max()

    # cross-document invariance: editing doc B must not change doc A states
    # (trivially held by causality; the load-bearing direction is the other
    # one — doc A leaking into doc B via past-context attention).
    x2 = x.clone()
    x2[0, 4] += 7.0
    out_full2 = bb(x2, attn_bias=doc_attn_bias(ds2, dtype=torch.float32))
    assert torch.allclose(out_full[:, :3], out_full2[:, :3], atol=1e-6)
    assert not torch.allclose(out_full[:, 3:], out_full2[:, 3:])
    # edit inside doc A: masked -> doc B UNCHANGED (no cross-doc past leak);
    # unmasked -> doc B CHANGES (sanity: the test can detect the leak at all).
    x3 = x.clone()
    x3[0, 1] += 7.0
    out_full3 = bb(x3, attn_bias=doc_attn_bias(ds2, dtype=torch.float32))
    assert torch.allclose(out_full[:, 3:], out_full3[:, 3:], atol=1e-6)
    assert not torch.allclose(bb(x3)[:, 3:], bb(x)[:, 3:])
    print("ok  block-diagonal equivalence + cross-doc invariance")


def test_boundary_weight() -> None:
    ds = torch.tensor([[1, 0, 1, 0, 0]], dtype=torch.bool)
    w = boundary_weight(ds)
    assert torch.equal(w, torch.tensor([[0.0, 1.0, 0.0, 1.0, 1.0]]))
    print("ok  boundary loss weight (doc-start targets excluded)")


def test_seqlens_from_start() -> None:
    from bltz.masking import doc_seqlens_from_start

    ds = torch.tensor([
        [1, 0, 0, 1, 0, 0],   # -> [3, 3]
        [0, 0, 1, 0, 0, 1],   # -> [2, 3, 1]
        [1, 0, 0, 0, 0, 0],   # -> [6]
    ], dtype=torch.bool)
    assert doc_seqlens_from_start(ds) == [[3, 3], [2, 3, 1], [6]]
    print("ok  doc_seqlens_from_start (window-major flat order)")


def test_xformers_equivalence() -> None:
    try:
        import xformers.ops  # noqa: F401
    except ImportError:
        print("skip xformers equivalence (not installed)")
        return
    torch.manual_seed(5)
    bb = Backbone(d_model=16, nhead=2, layers=2, ffn_mult=2, max_len=64).eval()
    x = torch.randn(2, 7, 16)
    ds = torch.tensor([[1, 0, 0, 1, 0, 0, 1],
                       [0, 1, 0, 0, 0, 1, 0]], dtype=torch.bool)
    from bltz.masking import doc_attn_bias, xf_bias_from_start

    o_dense = bb(x, attn_bias=doc_attn_bias(ds, dtype=torch.float32))
    o_xf = bb(x, attn_bias=xf_bias_from_start(ds))
    d = (o_dense - o_xf).abs().max().item()
    assert d < 2e-2, d
    print(f"ok  xformers block-diagonal == dense bias forward (max d {d:.4f})")


if __name__ == "__main__":
    test_shard_flags()
    test_verify_propagation()
    test_bias_structure()
    test_backbone_equivalence()
    test_boundary_weight()
    test_seqlens_from_start()
    test_xformers_equivalence()
    print("ALL OK")

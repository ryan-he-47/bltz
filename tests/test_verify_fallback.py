"""Online verify+resplit mechanism tests (bltz/verify.py).

A RANDOM tiny AE fails EM on almost everything -> exercises the full cascade
(BPE resplit -> re-verify -> char-aware <=3B floor) deterministically. Also
checks the real word AE passes common words untouched (skipped if ckpt absent).

Run: python tests/test_verify_fallback.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from bltz.config import Cfg
from bltz.models.autoencoder import ByteStringAE
from bltz.verify import VerifyResplitter, _char_chunks, fit_window

TOK = r"data\qwen35_tokenizer"
REAL_AE = r"checkpoints\ae_48_256_word_full\best.pt"


def _tiny_ae(l_max: int = 8) -> ByteStringAE:
    cfg = Cfg({"model": {"l_max": l_max, "d_byte": 16, "d_pos": 8,
                         "enc_width": 32, "enc_layers": 1, "enc_heads": 2,
                         "inducing": 0, "d_emb": 8, "d_delta": 8,
                         "dec_hidden": 32, "dec_depth": 2, "film_layers": 1}})
    torch.manual_seed(0)
    return ByteStringAE(cfg).eval()


def test_char_chunks():
    # char-aware: never splits a UTF-8 char; every piece <= cap bytes
    s = "aé中🙂b".encode("utf-8")  # 1+2+3+4+1 = 11 bytes
    out = _char_chunks(s, 3)
    assert b"".join(out) == s, "byte conservation"
    # cap respected except a single char longer than cap (never split a char)
    assert all(len(p) <= 3 or len(p.decode("utf-8")) == 1 for p in out), out
    assert all(p.decode("utf-8") for p in out), "every piece valid UTF-8"
    assert out == [b"a\xc3\xa9", "中".encode(), "🙂".encode(), b"b"], out
    # dirty bytes round-trip
    dirty = b"\x80\xfeabc"
    assert b"".join(_char_chunks(dirty, 2)) == dirty
    print("[ok] char_chunks: conservation, cap, char-safety, dirty bytes")


def test_cascade_mechanics():
    ae = _tiny_ae()
    v = VerifyResplitter(ae, TOK, ent_thresh=0.2, floor_bytes=3,
                         device="cpu", log_every=0)
    units = [b"hello ", b"world ", b"antidisestablishment", b"a ", b"9"]
    out = v.process(units)
    assert b"".join(out) == b"".join(units), "byte conservation"
    assert len(out) >= len(units), "count only grows"
    assert all(1 <= len(u) <= ae.l_max for u in out), "l_max respected"
    # determinism + memoization: second call identical, zero new verify rows
    rows = v.n_verify_rows
    out2 = v.process(units)
    assert out2 == out, "deterministic"
    assert v.n_verify_rows == rows, "cache hit: no re-verify"
    # floor pieces are char-safe and <= 3 bytes
    assert all(len(u) <= 3 or u in units for u in out), "floor cap"
    print(f"[ok] cascade: {units} -> {out}")
    print(f"     stats: flag {v.n_flag}, pieces {v.n_piece}, floor {v.n_floor}")


def test_fit_window():
    units = [b"a", b"b", b"c", b"d"]
    assert fit_window(units, 3) == [b"a", b"b", b"c"]
    assert fit_window(units, 8) == units
    print("[ok] fit_window trim")


def test_real_ae_passthrough():
    if not os.path.exists(REAL_AE):
        print("[skip] real AE ckpt not present")
        return
    st = torch.load(REAL_AE, map_location="cpu", weights_only=False)
    ae = ByteStringAE(Cfg(st["cfg"]))
    ae.load_state_dict(st["model"])
    ae = ae.eval()
    v = VerifyResplitter(ae, TOK, ent_thresh=0.2, floor_bytes=3,
                         device="cpu", log_every=0)
    easy = [b"the ", b"and ", b"of ", b"to ", b"in "]
    out = v.process(easy)
    assert out == easy, f"common words must pass untouched, got {out}"
    print("[ok] real AE: common words pass untouched")


if __name__ == "__main__":
    test_char_chunks()
    test_cascade_mechanics()
    test_fit_window()
    test_real_ae_passthrough()
    print("test_verify_fallback: ALL OK")

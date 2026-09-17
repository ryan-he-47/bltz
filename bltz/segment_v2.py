"""v2 segmentation: enhanced BPE (docs/31 §2.1, 2026-09-17 拍板).

Two boundary sets per sample:
  1. rule pre-tokenizer: byte-class runs — space-like / word (ASCII alnum +
     bytes >= 0x80, i.e. utf8 continuations stay inside words) / symbol
     (ASCII punctuation);
  2. Qwen3.5 BPE tokenizer boundaries (tokenizer.json, local data dir).

Combine at every candidate gap (union of both sets): both agree -> always
split; only one -> split with probability p (0.5, baked once per sample —
single-pass training makes online re-rolling equivalent, 2026-09-17 拍板).
Units hard-capped at l_max bytes (long symbol runs are chopped).
"""
from __future__ import annotations

import os
import random
from functools import lru_cache

_SPACE = frozenset((9, 10, 11, 12, 13, 32))


def _byte_class(b: int) -> int:
    if b in _SPACE:
        return 0
    if b >= 0x80 or (0x30 <= b <= 0x39) or (0x41 <= b <= 0x5A) or (0x61 <= b <= 0x7A):
        return 1  # word (alnum + utf8 continuation)
    return 2  # symbol


def pre_boundaries(raw: bytes) -> set[int]:
    """Run edges of the byte-class pre-tokenizer (0 and len excluded)."""
    out: set[int] = set()
    prev = None
    for i, b in enumerate(raw):
        c = _BYTE_CLASS[b]
        if prev is not None and c != prev:
            out.add(i)
        prev = c
    return out


_BYTE_CLASS = [_byte_class(b) for b in range(256)]

@lru_cache(maxsize=1)
def _tokenizer(tok_dir: str):
    from tokenizers import Tokenizer

    return Tokenizer.from_file(os.path.join(tok_dir, "tokenizer.json"))


def bpe_boundaries(raw: bytes, tok_dir: str) -> set[int]:
    """BPE token start offsets, mapped to BYTE offsets. Dirty-UTF8 fallback: {}."""
    try:
        s = raw.decode("utf-8", errors="surrogateescape")
        enc = _tokenizer(tok_dir).encode(s, add_special_tokens=False)
    except Exception:
        return set()
    # char idx -> byte idx; surrogateescape round-trips exactly
    cum = [0]
    acc = 0
    for ch in s:
        acc += len(ch.encode("utf-8", errors="surrogateescape"))
        cum.append(acc)
    out: set[int] = set()
    for start, _end in enc.offsets:
        b = cum[start]
        if 0 < b < len(raw):
            out.add(b)
    return out


def segment_v2(
    raw: bytes,
    rng: random.Random,
    tok_dir: str,
    p: float = 0.5,
    l_max: int = 32,
) -> list[bytes]:
    """Enhanced-BPE segment one sample into byte-string units."""
    pre = pre_boundaries(raw)
    bpe = bpe_boundaries(raw, tok_dir)
    cuts: list[int] = []
    for g in sorted(pre | bpe):
        if g in pre and g in bpe:
            cuts.append(g)
        elif rng.random() < p:
            cuts.append(g)
    units: list[bytes] = []
    pos = 0
    for g in cuts + [len(raw)]:
        seg = raw[pos:g]
        pos = g
        for i in range(0, len(seg), l_max):  # hard cap
            if seg[i : i + l_max]:
                units.append(seg[i : i + l_max])
    return units

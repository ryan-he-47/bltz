"""Script-style test for the fixed semantic segmenter. Run: python tests/test_segment.py"""
from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bytefield.segment import Segmenter

SAMPLES = [
    "Hello, world! This is ByteField.",
    "The quick brown fox jumps over the lazy dog. 123 + 4.56 = 128.56",
    "don't can't it's we're they've I'd",
    "你好,世界。这是一段混合文本 with English words 混合在一起。",
    "emoji test 🚀🔥 done",
    "def f(x):\n    return x**2  # comment\n",
    "   leading spaces and trailing punctuation...",
    "a",
    "!!!",
    "supercalifragilisticexpialidocious " * 3,  # long word -> subdivision
    "UTF-8 边界:äöü αβγ δεζ ηθικ",
]


def check(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def main() -> None:
    seg = Segmenter(l_max=16, p_split=0.5, p_merge=0.5)

    # deterministic base: roundtrip + validity + length bound
    for s in SAMPLES:
        chunks = seg.segment_str(s)
        check("".join(chunks) == s, f"roundtrip failed for {s[:40]!r}")
        for c in chunks:
            b = c.encode("utf-8")
            check(len(b) <= 16, f"chunk > l_max bytes: {c!r} ({len(b)})")
            b.decode("utf-8")  # must not raise
        chunks2 = seg.segment_str(s)
        check(chunks == chunks2, f"non-deterministic base segmentation: {s[:40]!r}")

    # empty input
    check(seg.segment_str("") == [], "empty string must give no chunks")

    # bytes API roundtrip
    data = "混合 bytes API test 123!".encode("utf-8")
    bchunks = seg.segment_bytes(data)
    check(b"".join(bchunks) == data, "bytes roundtrip failed")

    # augmentation: still roundtrip + valid, and actually perturbs with p=0.5
    rng = random.Random(42)
    changed = 0
    for s in SAMPLES:
        base = seg.segment_str(s)
        aug = seg.segment_str(s, rng=rng, augment=True)
        check("".join(aug) == s, f"augment roundtrip failed: {s[:40]!r}")
        for c in aug:
            check(len(c.encode("utf-8")) <= 16, f"augment chunk > l_max: {c!r}")
            c.encode("utf-8").decode("utf-8")
        if aug != base:
            changed += 1
    check(changed >= len(SAMPLES) // 2, f"augmentation barely fires ({changed}/{len(SAMPLES)})")

    # unit-level spot checks (base semantics)
    seg0 = Segmenter(l_max=16)
    en = seg0.segment_str("Hello, world!")
    check(en == ["Hello, ", "world!"], f"english: {en}")
    cjk = seg0.segment_str("你好,世界")
    check(cjk == ["你", "好,", "世", "界"], f"cjk: {cjk}")
    contra = seg0.segment_str("don't stop")
    check(contra == ["don't ", "stop"], f"contraction: {contra}")

    # stats on a realistic paragraph
    para = (
        "The Byte Latent Transformer encodes bytes into dynamically sized patches. "
        "Patches are segmented based on the entropy of the next byte, allocating "
        "more compute where data complexity demands it. 这是一个混合段落,用来观察 "
        "CJK 与英文混合时的 patch 长度分布。" * 4
    )
    chunks = seg0.segment_str(para)
    lens = sorted(len(c.encode("utf-8")) for c in chunks)
    n = len(lens)
    print(f"chunks={n}  avg={sum(lens)/n:.2f}B  p50={lens[n//2]}  p95={lens[int(n*0.95)]}  max={lens[-1]}")
    print("sample chunks:", [c for c in seg0.segment_str('Patches are segmented, 没问题')][:10])

    print("test_segment.py: ALL PASS")


if __name__ == "__main__":
    main()

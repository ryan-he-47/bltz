"""Script-style test for the disk cache layer (ShardWriter/ShardReader) and its
equivalence with the in-memory build_batch path. Run: python tests/test_cache.py"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import random

from bltz.data import build_batch
from bltz.segment import Segmenter
from bltz.shards import ShardReader, ShardWriter

TEXTS = [
    "The Byte Latent Transformer encodes bytes into dynamically sized patches. " * 12,
    "Patches are segmented based on the entropy of the next byte. " * 12,
    "你好世界,这是一段混合文本 with English words 混合在一起。" * 12,
]
TMP = Path("data/test_cache_tmp")
S = 32  # patches per sequence
L_MAX = 16


def check(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def main() -> None:
    shutil.rmtree(TMP, ignore_errors=True)
    seg = Segmenter(L_MAX, 0.0, 0.0)

    # ---- write a mini cache ----
    writer = ShardWriter(str(TMP / "shard-00000"), L_MAX)
    for t in TEXTS:
        writer.add_document_units(seg.segment_bytes(t.encode("utf-8")))
    meta = writer.close({"l_max": L_MAX, "p_split": 0.0, "p_merge": 0.0})
    total_units = meta["n_units"]
    print(f"mini cache: {total_units} units, {meta['n_bytes']} bytes, {meta['n_docs']} docs")

    reader = ShardReader(str(TMP))
    n_seq = reader.n_sequences(S)
    check(n_seq == total_units // S, f"n_sequences: {n_seq} vs {total_units // S}")
    check(n_seq > 0, "no sequences in mini cache")

    # ---- equivalence with the in-memory path (no augmentation) ----
    ref = build_batch(TEXTS, seg, S, L_MAX)
    got = reader.make_batch(list(range(n_seq)), S)
    for key in ["byte_ids", "ends", "flat", "patch_pos", "flat_len"]:
        same = (ref[key] == got[key]).all()
        check(same, f"cache vs build_batch mismatch on {key}")
    print("cache == build_batch (no augmentation): OK")

    # ---- augmentation: roundtrip + validity + length bound ----
    rng = random.Random(7)
    for gi in [0, n_seq // 2, n_seq - 1]:
        raw_units = reader.sequence_units(gi, S + 64) if reader.n_sequences(S + 64) > gi else None
        # raw stream starting at this sequence's start, generously long
        si_u0 = reader._seq_location(gi, S)
        flat_all, lens_all, _ = reader.get_units(si_u0[0], si_u0[1], min(S + 64, reader.shards[si_u0[0]]["meta"]["n_units"] - si_u0[1]))
        raw_all = bytes(flat_all)
        units = reader.sequence_units(gi, S)
        from bltz.segment import augment_unit_bytes
        aug = augment_unit_bytes(units, L_MAX, 0.8, 0.8, rng)
        aug_join = b"".join(aug)
        check(raw_all.startswith(aug_join) or aug_join.startswith(raw_all), f"augment roundtrip broken at seq {gi}")
        for ub in aug:
            check(0 < len(ub) <= L_MAX, f"augmented unit out of bounds: {ub!r}")
            ub.decode("utf-8")
    print("augmentation roundtrip + UTF-8 + length bound: OK")

    # ---- augmented make_batch: uniform shapes across a batch ----
    batch = reader.make_batch(list(range(min(8, n_seq))), S, augment=True, p_split=0.5, p_merge=0.5, rng=rng)
    check(batch["byte_ids"].shape == (min(8, n_seq), S, L_MAX), f"augmented batch shape: {batch['byte_ids'].shape}")
    print("augmented make_batch shapes: OK")

    shutil.rmtree(TMP, ignore_errors=True)
    print("test_cache.py: ALL PASS")


if __name__ == "__main__":
    main()

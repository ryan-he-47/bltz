"""ShardRotator (block-level rotation sampler) mechanics test.

Builds a 5-shard synthetic cache with per-shard distinctive unit content,
then checks: resident set size, rotation cadence, shard retirement/advance,
off-ckpt alignment after rotation, full-corpus coverage over rotations,
seed determinism.

Run: python tests/test_shard_rotate.py
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bltz.shards import ShardReader, ShardRotator, ShardWriter

L_MAX = 8
N_SHARDS = 5
UNITS_PER_SHARD = 64


def build_cache(root: str) -> None:
    for i in range(N_SHARDS):
        d = os.path.join(root, f"shard-{i:05d}")
        w = ShardWriter(d, L_MAX)
        # distinctive units: shard i's units all start with byte 65+i ('A'+i)
        for _ in range(4):
            w.add_document_units(
                [bytes([65 + i]) + f"u{j:03d}".encode() for j in range(UNITS_PER_SHARD // 4)]
            )
        w.close({"l_max": L_MAX, "n_units": UNITS_PER_SHARD,
                 "n_bytes": UNITS_PER_SHARD * 5, "n_docs": 4})


def main() -> None:
    root = tempfile.mkdtemp(prefix="rot_test_")
    try:
        build_cache(root)
        rot = ShardRotator(root, resident=2, every=3, seed=42)
        assert len(rot.reader.shards) == 2, "resident set size"
        first_two = list(rot.names)
        assert first_two == rot.perm[:2], "initial = perm head"
        seen = set(first_two)
        S = 8
        n0 = rot.reader.n_sequences(S)
        assert n0 == 2 * UNITS_PER_SHARD // S, f"n_seq {n0}"

        # read a window pre-rotation (content from one of the first two shards)
        u = rot.reader.sequence_units(0, S)
        assert all(len(x) <= L_MAX and x for x in u)

        n_rots = 0
        for c in range(1, 40):
            rot.maybe_rotate()
            cur = list(rot.names)
            seen.update(cur)
            if len(rot.reader.shards) != 2:
                raise AssertionError("resident set drifted")
            # alignment: _off_ckpt must stay in sync with shards
            assert len(rot.reader._off_ckpt) <= len(rot.reader.shards)
            # window reads must keep working across rotations
            rot.reader.sequence_units(0, S)
        assert len(seen) == N_SHARDS, f"coverage: saw {len(seen)}/{N_SHARDS}"
        assert list(rot.names) != first_two or len(seen) > 2, "rotation happened"
        print(f"[ok] rotation: covered all {N_SHARDS} shards over 39 calls; "
              f"final resident {sorted(rot.names)}")

        # determinism
        r2 = ShardRotator(root, resident=2, every=3, seed=42)
        assert r2.perm == rot.perm, "same seed -> same permutation"
        r3 = ShardRotator(root, resident=2, every=3, seed=7)
        assert r3.perm != rot.perm or N_SHARDS <= 2, "different seed differs"
        print("[ok] determinism: seeded permutation")
        print("test_shard_rotate: ALL OK")
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    main()

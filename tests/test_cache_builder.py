"""Script-style test for the parallel cache builder on synthetic parquet files
(no network). Run: python tests/test_cache_builder.py"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pyarrow as pa
import pyarrow.parquet as pq

from bltz.segment import Segmenter
from bltz.shards import ShardReader
from scripts.build_cache import _process_file

TEXTS = [
    "The Byte Latent Transformer encodes bytes into dynamically sized patches. " * 15,
    "Entropy patching allocates compute where complexity demands it. " * 15,
]
TMP = Path("data/test_cache_builder_tmp")
L_MAX = 16


def check(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def _write_parquet(path: Path, texts: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.table({"text": texts})
    pq.write_table(table, str(path))


def main() -> None:
    shutil.rmtree(TMP, ignore_errors=True)
    pdir = TMP / "parquet"
    docs_per_file = 200
    for i in range(2):
        rows = [TEXTS[(i + k) % len(TEXTS)] for k in range(docs_per_file)]
        _write_parquet(pdir / f"{i:03d}_00000.parquet", rows)

    cache = TMP / "cache"
    seg_cfg = {"l_max": L_MAX, "p_split": 0.0, "p_merge": 0.0}
    for i in range(2):
        meta = _process_file(
            (i, str(pdir / f"{i:03d}_00000.parquet"), str(cache / f"shard-{i:05d}"),
             L_MAX, 0, seg_cfg)
        )
        check(meta["n_docs"] == docs_per_file, f"shard {i} docs: {meta['n_docs']}")
        print(f"shard-{i:05d}: {meta['n_units']} units, {meta['n_bytes']} bytes")

    reader = ShardReader(str(cache))
    check(len(reader.shards) == 2, f"reader sees {len(reader.shards)} shards")
    n_seq = reader.n_sequences(32)
    check(n_seq > 0, "no sequences")

    # content must equal sequential segmentation of the same texts
    seg = Segmenter(L_MAX, 0.0, 0.0)
    file0_texts = [TEXTS[k % len(TEXTS)] for k in range(docs_per_file)]
    expected_units = [u for t in file0_texts for u in seg.segment_bytes(t.encode("utf-8"))]
    got_units = []
    u0 = 0
    while len(got_units) < 50:
        got_units.extend(reader.sequence_units(u0, 32)[: 50 - len(got_units)])
        u0 += 1
    check(got_units[:20] == expected_units[:20], f"content mismatch: {got_units[:3]} vs {expected_units[:3]}")

    shutil.rmtree(TMP, ignore_errors=True)
    print("test_cache_builder.py: ALL PASS")


if __name__ == "__main__":
    main()

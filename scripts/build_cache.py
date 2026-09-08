"""Build the on-disk segmented cache from FineWeb-Edu (Stage 0 cache step).

Usage: python scripts/build_cache.py [configs/default.yaml] [--set a.b=value]
  e.g. --set data.cache_dir=data/fineweb10b --set build.max_docs=2000 --set build.shard_units=40000000

Streams docs, segments with the DETERMINISTIC BASE segmenter (augmentation is
applied at READ time), writes shard-NNNNN/ directories of ~shard_units units.
Resumable at shard granularity: progress.json is only written at shard close;
a crash mid-shard loses that shard's docs (they are re-streamed on resume).
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datasets import load_dataset

from bytefield.config import parse_cli
from bytefield.segment import Segmenter
from bytefield.shards import ShardWriter


def main() -> None:
    cfg = parse_cli("configs/default.yaml")
    b = cfg.get("build", None)
    max_docs = int(getattr(b, "max_docs", 0)) if b else 0
    shard_units = int(getattr(b, "shard_units", 40_000_000)) if b else 40_000_000
    cache_dir = cfg.data.cache_dir
    os.makedirs(cache_dir, exist_ok=True)

    # drop incomplete shards (dir exists but has no meta.json)
    done_shards = 0
    for name in sorted(os.listdir(cache_dir)):
        d = os.path.join(cache_dir, name)
        if not (os.path.isdir(d) and name.startswith("shard-")):
            continue
        if os.path.exists(os.path.join(d, "meta.json")):
            done_shards += 1
        else:
            shutil.rmtree(d)
            print(f"dropped incomplete shard dir: {d}")

    progress_path = os.path.join(cache_dir, "progress.json")
    done_docs, shard_idx = 0, done_shards
    if os.path.exists(progress_path):
        with open(progress_path, encoding="utf-8") as f:
            prog = json.load(f)
        done_docs = int(prog.get("done_docs", 0))
        shard_idx = int(prog.get("shard_idx", done_shards))
        if prog.get("done"):
            print(f"cache already complete ({done_docs} docs, {shard_idx} shards); nothing to do")
            return
    print(f"resume state: done_docs={done_docs}, next shard index={shard_idx}")

    seg = Segmenter(cfg.segment.l_max, 0.0, 0.0)  # deterministic BASE only
    ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
    if done_docs:
        ds = ds.skip(done_docs)

    def write_progress(done: bool = False) -> None:
        with open(progress_path, "w", encoding="utf-8") as f:
            json.dump({"done_docs": done_docs, "shard_idx": shard_idx, "done": done}, f)

    writer: ShardWriter | None = None
    units_in_shard = 0
    t0 = time.time()
    n_bytes = 0
    for i, row in enumerate(ds):
        if max_docs and i >= max_docs:
            break
        if writer is None:
            writer = ShardWriter(os.path.join(cache_dir, f"shard-{shard_idx:05d}"), cfg.segment.l_max)
            units_in_shard = 0
        units = seg.segment_bytes(row["text"].encode("utf-8"))
        writer.add_document_units(units)
        units_in_shard += len(units)
        n_bytes += len(row["text"].encode("utf-8"))
        done_docs += 1
        if units_in_shard >= shard_units:
            meta = writer.close(cfg.segment.to_dict())
            print(f"shard-{shard_idx:05d} closed: {meta['n_units']} units, {meta['n_bytes']/2**20:.0f} MiB")
            shard_idx += 1
            writer = None
            write_progress()
        if done_docs % 2000 == 0:
            rate = done_docs / (time.time() - t0)
            print(f"docs={done_docs} bytes={n_bytes/2**30:.2f} GiB rate={rate:.0f} docs/s", flush=True)

    if writer is not None and writer.n_units:
        meta = writer.close(cfg.segment.to_dict())
        print(f"shard-{shard_idx:05d} closed (final): {meta['n_units']} units, {meta['n_bytes']/2**20:.0f} MiB")
        shard_idx += 1
    write_progress(done=(not max_docs))
    print(f"DONE: {done_docs} docs, {n_bytes/2**30:.2f} GiB text, {shard_idx} shards, {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()

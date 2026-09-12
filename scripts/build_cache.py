"""Build the on-disk segmented cache from FineWeb-Edu (Stage 0 cache step).

Usage: python scripts/build_cache.py [configs/default.yaml] [--set a.b=value]
  full build:  python -u scripts/build_cache.py configs/default.yaml
  test build:  --set build.max_docs_per_file=300 --set build.workers=2
  local files: --set build.parquet_dir=E:/path/with/parquet/files

Two phases:
  1. download: fetch the 14 sample/10BT parquet files via hf_hub_download
     (resumable, checksummed, ~30GB total). Skipped for files already present
     in build.parquet_dir (or after a previous download).
  2. segment: multiprocessing over parquet files — each worker reads one
     parquet, segments docs with the DETERMINISTIC BASE segmenter
     (augmentation is applied at READ time), writes shard-{idx:05d}/ in the
     standard layout (bytes.npy/unit_len.npy/unit_flag.npy/meta.json).

One output shard per input parquet. Resume granularity = parquet file:
a shard with meta.json is complete and skipped; a shard dir without meta.json
is dropped and rebuilt. Single-threaded fallback: --set build.workers=1.
"""
from __future__ import annotations

import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bltz.config import parse_cli

N_FILES = 14  # sample/10BT/000_00000.parquet ... 013_00000.parquet


def _process_file(args: tuple[int, str, str, int, int, dict[str, Any]]) -> dict[str, Any]:
    """Worker: segment one parquet file into one shard directory."""
    file_idx, parquet_path, out_dir, l_max, max_docs, seg_cfg = args
    import pyarrow.parquet as pq

    from bltz.segment import Segmenter
    from bltz.shards import ShardWriter

    seg = Segmenter(l_max, 0.0, 0.0)  # deterministic BASE only
    writer = ShardWriter(out_dir, l_max)
    pf = pq.ParquetFile(parquet_path)
    n_docs = 0
    t0 = time.time()
    for rb in pf.iter_batches(batch_size=2000, columns=["text"]):
        for text in rb.column("text").to_pylist():
            if max_docs and n_docs >= max_docs:
                break
            units = [u.encode("utf-8") for u in seg.segment_str(text)]
            writer.add_document_units(units)
            n_docs += 1
        if max_docs and n_docs >= max_docs:
            break
    meta = writer.close(seg_cfg)
    meta["file_idx"] = file_idx
    meta["sec"] = round(time.time() - t0, 1)
    return meta


def _shard_dir(cache_dir: str, idx: int) -> str:
    return os.path.join(cache_dir, f"shard-{idx:05d}")


def main() -> None:
    cfg = parse_cli("configs/default.yaml")
    b = cfg.get("build", None)
    workers = int(getattr(b, "workers", 8)) if b else 8
    max_docs = int(getattr(b, "max_docs_per_file", 0)) if b else 0
    parquet_dir = getattr(b, "parquet_dir", "") if b else ""
    cache_dir = cfg.data.cache_dir
    os.makedirs(cache_dir, exist_ok=True)
    seg_cfg = cfg.segment.to_dict()

    # ---- figure out which input files still need processing ----
    todo: list[int] = []
    for i in range(N_FILES):
        d = _shard_dir(cache_dir, i)
        if os.path.exists(os.path.join(d, "meta.json")):
            continue  # complete
        if os.path.isdir(d):
            shutil.rmtree(d)  # incomplete shard from a crashed run
        todo.append(i)
    if not todo:
        print("cache already complete (all 14 shards have meta.json); nothing to do")
        return
    print(f"shards to build: {todo}")

    # ---- phase 1: make sure the parquet files are local ----
    paths: dict[int, str] = {}
    if parquet_dir:
        for i in todo:
            p = os.path.join(parquet_dir, f"{i:03d}_00000.parquet")
            if not os.path.exists(p):
                raise FileNotFoundError(f"missing local parquet: {p}")
            paths[i] = p
        print(f"phase 1: using local parquet dir {parquet_dir}")
    else:
        from huggingface_hub import hf_hub_download

        for i in todo:
            fn = f"sample/10BT/{i:03d}_00000.parquet"
            print(f"phase 1: downloading {fn} ...", flush=True)
            paths[i] = hf_hub_download(
                repo_id="HuggingFaceFW/fineweb-edu", filename=fn, repo_type="dataset"
            )
    print("phase 1 done")

    # ---- phase 2: segment in parallel (one worker per parquet file) ----
    jobs = [
        (i, paths[i], _shard_dir(cache_dir, i), cfg.segment.l_max, max_docs, seg_cfg)
        for i in todo
    ]
    t0 = time.time()
    if workers <= 1 or len(jobs) == 1:
        for j in jobs:
            meta = _process_file(j)
            print(f"shard-{meta['file_idx']:05d}: {meta['n_units']} units, "
                  f"{meta['n_bytes']/2**20:.0f} MiB, {meta['n_docs']} docs, {meta['sec']}s", flush=True)
    else:
        import multiprocessing as mp

        with mp.Pool(min(workers, len(jobs))) as pool:
            for meta in pool.imap_unordered(_process_file, jobs):
                print(f"shard-{meta['file_idx']:05d}: {meta['n_units']} units, "
                      f"{meta['n_bytes']/2**20:.0f} MiB, {meta['n_docs']} docs, {meta['sec']}s", flush=True)
    print(f"phase 2 done in {time.time()-t0:.0f}s; cache at {cache_dir}")


if __name__ == "__main__":
    main()

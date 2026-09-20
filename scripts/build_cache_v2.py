"""Build the v2 enhanced-BPE cache from FineWeb-Edu (docs/32 S1).

Same skeleton as build_cache.py (one shard per parquet, resume per file,
heartbeats); segmentation swapped for segment_v2 (rule pre-tokenizer x
Qwen3.5 BPE, agree -> always split, disagree -> p_disagree, BAKED once per
document: rng seeded by global doc index — single-pass training makes
online re-rolling equivalent, 2026-09-17 拍板). l_max=32 (v2 units can be
longer than v1's 16).

  full:  python -u scripts/build_cache_v2.py configs/cache_v2.yaml \
           --set data.cache_dir=... --set seg.tok_dir=<Qwen3.5-2B dir>
  smoke: --set build.max_docs_per_file=300
"""
from __future__ import annotations

import os
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bltz.config import parse_cli

N_FILES = 14  # sample/10BT/000_00000.parquet ... 013_00000.parquet


def _process_file(args: tuple[int, str, str, int, dict[str, Any]]) -> dict[str, Any]:
    file_idx, parquet_path, out_dir, max_docs, seg_cfg = args
    import pyarrow.parquet as pq

    from bltz.segment_v2 import segment_v2_batch
    from bltz.shards import ShardWriter

    l_max = int(seg_cfg["l_max"])
    p = float(seg_cfg["p_disagree"])
    tok_dir = str(seg_cfg["tok_dir"])
    writer = ShardWriter(out_dir, l_max)
    pf = pq.ParquetFile(parquet_path)
    n_docs = 0
    t0 = time.time()
    for rb in pf.iter_batches(batch_size=2000, columns=["text"]):
        texts = rb.column("text").to_pylist()
        if max_docs and n_docs >= max_docs:
            break
        take = len(texts) if not max_docs else min(len(texts), max_docs - n_docs)
        raws = [texts[i].encode("utf-8") for i in range(take)]
        rngs = [random.Random(20260918 + file_idx * 100_000_000 + n_docs + i)
                for i in range(take)]
        for units in segment_v2_batch(raws, rngs, tok_dir, p=p, l_max=l_max):
            if units:
                writer.add_document_units(units)
            n_docs += 1
        if n_docs % 5000 < 2000 or (max_docs and n_docs == max_docs):
            rate = n_docs / max(time.time() - t0, 1e-9)
            eta = f", ETA {(max_docs - n_docs) / rate / 60:.0f} min" if max_docs else ""
            print(f"  [shard-{file_idx:05d}] {n_docs} docs ({rate:.0f} docs/s{eta}), "
                  f"{writer.n_units/1e6:.1f}M units, {time.time()-t0:.0f}s", flush=True)
    meta = writer.close({
        "type": "enhanced_bpe", "p_disagree": p, "tokenizer": "Qwen3.5-2B",
        "l_max": l_max, "baked_seed": 20260918,
    })
    meta["file_idx"] = file_idx
    meta["sec"] = round(time.time() - t0, 1)
    return meta


def _shard_dir(cache_dir: str, idx: int) -> str:
    return os.path.join(cache_dir, f"shard-{idx:05d}")


def main() -> None:
    cfg = parse_cli("configs/cache_v2.yaml")
    b = cfg.get("build", None)
    workers = int(getattr(b, "workers", 4)) if b else 4
    max_docs = int(getattr(b, "max_docs_per_file", 0)) if b else 0
    parquet_dir = getattr(b, "parquet_dir", "") if b else ""
    only_files = getattr(b, "only_files", "") if b else ""  # e.g. "0" or "0,3"
    cache_dir = cfg.data.cache_dir
    os.makedirs(cache_dir, exist_ok=True)

    if str(only_files).strip():  # "0" alone is a valid selection (int 0 is falsy!)
        want = [int(x) for x in str(only_files).split(",") if x.strip()]
        todo = []
        for i in want:
            d = _shard_dir(cache_dir, i)
            if os.path.exists(os.path.join(d, "meta.json")):
                print(f"shard-{i:05d} already complete, skip")
                continue
            if os.path.isdir(d):
                shutil.rmtree(d)
            todo.append(i)
    else:
        todo = []
        for i in range(N_FILES):
            d = _shard_dir(cache_dir, i)
            if os.path.exists(os.path.join(d, "meta.json")):
                continue
            if os.path.isdir(d):
                shutil.rmtree(d)
            todo.append(i)
    if not todo:
        print("cache already complete (all 14 shards have meta.json); nothing to do")
        return
    print(f"shards to build: {todo}", flush=True)

    paths: dict[int, str] = {}
    if parquet_dir:
        for i in todo:
            p = os.path.join(parquet_dir, f"{i:03d}_00000.parquet")
            if not os.path.exists(p):
                raise FileNotFoundError(f"missing local parquet: {p}")
            paths[i] = p
        print(f"phase 1: using local parquet dir {parquet_dir}", flush=True)
    else:
        from huggingface_hub import hf_hub_download

        for i in todo:
            fn = f"sample/10BT/{i:03d}_00000.parquet"
            print(f"phase 1: downloading {fn} ...", flush=True)
            paths[i] = hf_hub_download(
                repo_id="HuggingFaceFW/fineweb-edu", filename=fn, repo_type="dataset"
            )
    print("phase 1 done", flush=True)

    jobs = [
        (i, paths[i], _shard_dir(cache_dir, i), max_docs, cfg.seg.to_dict())
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
    print(f"phase 2 done in {time.time()-t0:.0f}s; cache at {cache_dir}", flush=True)


if __name__ == "__main__":
    main()

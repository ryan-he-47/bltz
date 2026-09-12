"""Tokenize FineWeb-Edu sample-10BT into GPT-2 token-id shards (parallel).

Usage mirrors build_cache.py:
  full build:  python -u scripts/build_token_cache.py configs/baseline.yaml
  test build:  --set build.max_docs_per_file=300 --set build.workers=2
  local files: --set build.parquet_dir=E:/path/with/parquet/files

phase 1 reuses the SAME parquet downloads as build_cache.py (hf_hub_download
caches them); phase 2 batch-encodes with the Rust `tokenizers` lib (fast)
and writes shard-{idx:05d}/ in the bytecache-analogous layout. One output
shard per input parquet; resume granularity = file (meta.json present = skip).
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

N_FILES = 14


def _process_file(args: tuple[int, str, str, int]) -> dict[str, Any]:
    """Worker: tokenize one parquet file into one token shard directory."""
    file_idx, parquet_path, out_dir, max_docs = args
    import pyarrow.parquet as pq
    from tokenizers import Tokenizer

    from bltz.token_cache import EOS_ID, TokenShardWriter

    tok = Tokenizer.from_pretrained("gpt2")
    writer = TokenShardWriter(out_dir)
    pf = pq.ParquetFile(parquet_path)
    n_docs = 0
    t0 = time.time()
    for rb in pf.iter_batches(batch_size=1000, columns=["text"]):
        texts = rb.column("text").to_pylist()
        if max_docs:
            texts = texts[: max(0, max_docs - n_docs)]
        encs = tok.encode_batch(texts, add_special_tokens=False)
        for e in encs:
            writer.add_document_ids(e.ids + [EOS_ID])
        n_docs += len(texts)
        if max_docs and n_docs >= max_docs:
            break
    meta = writer.close(50257, "gpt2")
    meta["file_idx"] = file_idx
    meta["sec"] = round(time.time() - t0, 1)
    return meta


def _shard_dir(cache_dir: str, idx: int) -> str:
    return os.path.join(cache_dir, f"shard-{idx:05d}")


def main() -> None:
    cfg = parse_cli("configs/baseline.yaml")
    b = cfg.get("build", None)
    workers = int(getattr(b, "workers", 8)) if b else 8
    max_docs = int(getattr(b, "max_docs_per_file", 0)) if b else 0
    parquet_dir = getattr(b, "parquet_dir", "") if b else ""
    cache_dir = cfg.data.cache_dir
    os.makedirs(cache_dir, exist_ok=True)

    todo: list[int] = []
    for i in range(N_FILES):
        d = _shard_dir(cache_dir, i)
        if os.path.exists(os.path.join(d, "meta.json")):
            continue
        if os.path.isdir(d):
            shutil.rmtree(d)
        todo.append(i)
    if not todo:
        print("token cache already complete; nothing to do")
        return
    print(f"token shards to build: {todo}")

    paths: dict[int, str] = {}
    if parquet_dir:
        for i in todo:
            p = os.path.join(parquet_dir, f"{i:03d}_00000.parquet")
            if not os.path.exists(p):
                raise FileNotFoundError(f"missing local parquet: {p}")
            paths[i] = p
    else:
        from huggingface_hub import hf_hub_download

        for i in todo:
            fn = f"sample/10BT/{i:03d}_00000.parquet"
            print(f"phase 1: downloading {fn} ...", flush=True)
            paths[i] = hf_hub_download(
                repo_id="HuggingFaceFW/fineweb-edu", filename=fn, repo_type="dataset"
            )
    print("phase 1 done")

    jobs = [(i, paths[i], _shard_dir(cache_dir, i), max_docs) for i in todo]
    t0 = time.time()
    if workers <= 1 or len(jobs) == 1:
        for j in jobs:
            meta = _process_file(j)
            print(f"shard-{meta['file_idx']:05d}: {meta['n_tokens']} tokens, "
                  f"{meta['n_docs']} docs, {meta['sec']}s", flush=True)
    else:
        import multiprocessing as mp

        with mp.Pool(min(workers, len(jobs))) as pool:
            for meta in pool.imap_unordered(_process_file, jobs):
                print(f"shard-{meta['file_idx']:05d}: {meta['n_tokens']} tokens, "
                      f"{meta['n_docs']} docs, {meta['sec']}s", flush=True)
    print(f"phase 2 done in {time.time()-t0:.0f}s; token cache at {cache_dir}")


if __name__ == "__main__":
    main()

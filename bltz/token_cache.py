"""Token-id cache for the GPT-2 token baseline (Stage-2 twin control).

Layout per shard directory (shard-NNNNN/):
  tokens.npy    uint16, concatenated token ids of all docs in the shard;
                documents are separated by EOS_ID (<|endoftext|> = 50256)
  doc_flag.npy  uint8, 1 on the FIRST token of each document
  meta.json     {n_tokens, n_docs, vocab, tokenizer}

Sequences are sliced at read time: (B, T) idx batches. v1 simplification
(same as the byte cache): plain causal mask, sequences may cross documents;
doc-boundary masking is a v2 item (flags are stored for it).
"""
from __future__ import annotations

import json
import os
from typing import Any

import numpy as np
import torch

EOS_ID = 50256  # <|endoftext|>


class TokenShardWriter:
    """Accumulates token ids and writes one shard directory on close()."""

    def __init__(self, out_dir: str):
        os.makedirs(out_dir, exist_ok=True)
        self.out_dir = out_dir
        self._tokens: list[int] = []
        self._flags: list[int] = []
        self.n_docs = 0

    @property
    def n_tokens(self) -> int:
        return len(self._tokens)

    def add_document_ids(self, ids: list[int]) -> None:
        """ids already include the trailing EOS separator."""
        if not ids:
            return
        for i, t in enumerate(ids):
            self._tokens.append(t)
            self._flags.append(1 if i == 0 else 0)
        self.n_docs += 1

    def close(self, vocab: int, tokenizer_name: str) -> dict[str, Any]:
        np.save(
            os.path.join(self.out_dir, "tokens.npy"),
            np.asarray(self._tokens, dtype=np.uint16),
        )
        np.save(
            os.path.join(self.out_dir, "doc_flag.npy"),
            np.asarray(self._flags, dtype=np.uint8),
        )
        meta: dict[str, Any] = {
            "n_tokens": len(self._tokens),
            "n_docs": self.n_docs,
            "vocab": vocab,
            "tokenizer": tokenizer_name,
        }
        with open(os.path.join(self.out_dir, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=1)
        return meta


class TokenShardReader:
    """Read-time sequence slicer + batch builder over token shards."""

    def __init__(self, cache_dir: str):
        self.cache_dir = cache_dir
        self.shards: list[dict[str, Any]] = []
        for name in sorted(os.listdir(cache_dir)):
            d = os.path.join(cache_dir, name)
            if not (os.path.isdir(d) and name.startswith("shard-")):
                continue
            with open(os.path.join(d, "meta.json"), encoding="utf-8") as f:
                meta = json.load(f)
            self.shards.append({
                "dir": d,
                "tokens": np.load(os.path.join(d, "tokens.npy"), mmap_mode="r"),
                "doc_flag": np.load(os.path.join(d, "doc_flag.npy"), mmap_mode="r"),
                "meta": meta,
            })
        if not self.shards:
            raise FileNotFoundError(f"no token shards found under {cache_dir}")

    def n_sequences(self, seq_len: int) -> int:
        return sum(s["meta"]["n_tokens"] // seq_len for s in self.shards)

    def _locate(self, global_idx: int, seq_len: int) -> tuple[int, int]:
        for si, s in enumerate(self.shards):
            n_seq = s["meta"]["n_tokens"] // seq_len
            if global_idx < n_seq:
                return si, global_idx * seq_len
            global_idx -= n_seq
        raise IndexError(global_idx)

    def make_batch(self, indices: list[int], seq_len: int) -> dict[str, torch.Tensor]:
        seqs = []
        for gi in indices:
            si, t0 = self._locate(gi, seq_len)
            ids = np.asarray(self.shards[si]["tokens"][t0 : t0 + seq_len])
            seqs.append(torch.from_numpy(ids.copy()).long())
        return {"idx": torch.stack(seqs)}

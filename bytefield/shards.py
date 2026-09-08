"""On-disk cache of segmented unit streams (Stage 0 cache layer).

Layout per shard directory (shard-00000/, shard-00001/, ...):
  bytes.npy     uint8, concatenated unit bytes
  unit_len.npy  uint8, per-unit byte length (<= l_max)
  unit_flag.npy uint8, per-unit flags (bit0: starts a new document)
  meta.json     {n_units, n_bytes, n_docs, l_max}

Sequences are sliced at READ time (any n_patches works with the same cache),
never across shard boundaries. Units are stored from the DETERMINISTIC BASE
segmentation; p_split/p_merge augmentation is applied at read time
(augment_unit_bytes) so every epoch resamples it (design doc §5.1).
"""
from __future__ import annotations

import json
import os
import random
from typing import Any

import numpy as np
import torch

from .data import collate_sequences, tensorize_units
from .segment import augment_unit_bytes


class ShardWriter:
    """Accumulates segmented units and writes one shard directory on close()."""

    def __init__(self, out_dir: str, l_max: int):
        os.makedirs(out_dir, exist_ok=True)
        self.out_dir = out_dir
        self.l_max = l_max
        self._bytes = bytearray()
        self._lens: list[int] = []
        self._flags: list[int] = []
        self.n_docs = 0

    @property
    def n_units(self) -> int:
        return len(self._lens)

    def add_document_units(self, units: list[bytes]) -> None:
        for i, ub in enumerate(units):
            if not (0 < len(ub) <= self.l_max):
                raise ValueError(f"bad unit length {len(ub)} (l_max={self.l_max})")
            self._bytes.extend(ub)
            self._lens.append(len(ub))
            self._flags.append(1 if i == 0 else 0)
        self.n_docs += 1

    def close(self, segmenter_cfg: dict[str, Any]) -> dict[str, Any]:
        np.save(
            os.path.join(self.out_dir, "bytes.npy"),
            np.frombuffer(bytes(self._bytes), dtype=np.uint8),
        )
        np.save(os.path.join(self.out_dir, "unit_len.npy"), np.asarray(self._lens, dtype=np.uint8))
        np.save(os.path.join(self.out_dir, "unit_flag.npy"), np.asarray(self._flags, dtype=np.uint8))
        meta = {
            "n_units": len(self._lens),
            "n_bytes": len(self._bytes),
            "n_docs": self.n_docs,
            "l_max": self.l_max,
            "segmenter": segmenter_cfg,
        }
        with open(os.path.join(self.out_dir, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=1)
        return meta


class ShardReader:
    """Read-time sequence slicer + batch builder over one or more shards."""

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
                "bytes": np.load(os.path.join(d, "bytes.npy"), mmap_mode="r"),
                "unit_len": np.load(os.path.join(d, "unit_len.npy"), mmap_mode="r"),
                "unit_flag": np.load(os.path.join(d, "unit_flag.npy"), mmap_mode="r"),
                "meta": meta,
            })
        if not self.shards:
            raise FileNotFoundError(f"no shards found under {cache_dir}")
        self.l_max = int(self.shards[0]["meta"]["l_max"])
        # per-shard unit byte offsets (exclusive prefix sums), built lazily
        self._unit_off: list[np.ndarray] = []

    def n_sequences(self, n_patches: int) -> int:
        return sum(s["meta"]["n_units"] // n_patches for s in self.shards)

    def _offsets(self, si: int) -> np.ndarray:
        while len(self._unit_off) <= si:
            k = len(self._unit_off)
            self._unit_off.append(np.cumsum(self.shards[k]["unit_len"], dtype=np.int64))
        return self._unit_off[si]

    def get_units(self, si: int, u0: int, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Units [u0, u0+n) of shard si -> (flat bytes, lens, flags)."""
        s = self.shards[si]
        off = self._offsets(si)
        b0 = 0 if u0 == 0 else int(off[u0 - 1])
        b1 = int(off[u0 + n - 1])
        return (
            np.asarray(s["bytes"][b0:b1]),
            np.asarray(s["unit_len"][u0 : u0 + n]),
            np.asarray(s["unit_flag"][u0 : u0 + n]),
        )

    def _seq_location(self, global_idx: int, n_patches: int) -> tuple[int, int]:
        """Global sequence index -> (shard_idx, unit_start)."""
        for si, s in enumerate(self.shards):
            n_seq = s["meta"]["n_units"] // n_patches
            if global_idx < n_seq:
                return si, global_idx * n_patches
            global_idx -= n_seq
        raise IndexError(global_idx)

    def sequence_tensors(self, global_idx: int, n_patches: int) -> dict[str, torch.Tensor]:
        si, u0 = self._seq_location(global_idx, n_patches)
        flat, lens, _ = self.get_units(si, u0, n_patches)
        return tensorize_units(
            torch.from_numpy(flat.copy()).long(),
            torch.from_numpy(lens.copy()).long(),
            self.l_max,
        )

    def sequence_units(self, global_idx: int, n_patches: int) -> list[bytes]:
        """Raw unit byte strings (needed for the augmentation path)."""
        si, u0 = self._seq_location(global_idx, n_patches)
        flat, lens, _ = self.get_units(si, u0, n_patches)
        units: list[bytes] = []
        pos = 0
        for Ln in lens:
            n_bytes = int(Ln)  # uint8 scalar would overflow on pos arithmetic (NumPy 2)
            units.append(bytes(flat[pos : pos + n_bytes]))
            pos += n_bytes
        return units

    def make_batch(
        self,
        indices: list[int],
        n_patches: int,
        augment: bool = False,
        p_split: float = 0.0,
        p_merge: float = 0.0,
        rng: random.Random | None = None,
    ) -> dict[str, torch.Tensor]:
        rng_eff: random.Random = rng if rng is not None else random.Random()
        seqs = []
        for gi in indices:
            if augment:
                units = self.sequence_units(gi, n_patches)
                units = augment_unit_bytes(units, self.l_max, p_split, p_merge, rng_eff)
                # augmentation changes the unit COUNT; refit to exactly n_patches
                units = self._refit(units, gi, n_patches)
                if len(units) != n_patches:
                    # shard-tail edge: not enough raw units to pad with ->
                    # fall back to the un-augmented sequence (safe, shape-stable)
                    seqs.append(self.sequence_tensors(gi, n_patches))
                    continue
                flat = torch.from_numpy(np.frombuffer(b"".join(units), dtype=np.uint8).copy()).long()
                lens = torch.tensor([len(u) for u in units], dtype=torch.long)
                seqs.append(tensorize_units(flat, lens, self.l_max))
            else:
                seqs.append(self.sequence_tensors(gi, n_patches))
        return collate_sequences(seqs)

    def _refit(self, units: list[bytes], gi: int, n_patches: int) -> list[bytes]:
        """Pad with following raw units of the same shard or trim the tail, so
        the augmented sequence has exactly n_patches units. May still return a
        short list at the shard tail; the caller falls back in that case."""
        if len(units) > n_patches:
            return units[:n_patches]
        si, u0 = self._seq_location(gi, n_patches)
        s = self.shards[si]
        extra_u0 = u0 + n_patches
        need = n_patches - len(units)
        if extra_u0 + need > s["meta"]["n_units"]:
            return units  # short; caller falls back
        off = self._offsets(si)
        pos = 0 if extra_u0 == 0 else int(off[extra_u0 - 1])
        for Ln in np.asarray(s["unit_len"][extra_u0 : extra_u0 + need]):
            n_bytes = int(Ln)  # uint8 scalar would overflow on pos arithmetic (NumPy 2)
            units.append(bytes(np.asarray(s["bytes"][pos : pos + n_bytes])))
            pos += n_bytes
        return units[:n_patches]

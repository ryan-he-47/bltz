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
from array import array
from collections import deque
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
        # compact backing stores (cluster OOM lesson 2026-09-12: a Python list
        # of ints costs ~8B/entry in pointers -> 144M units/shard was
        # ~2.3GB/worker, 8 workers pegged MaxRSS 45.6G vs the 48G limit and
        # mp.Pool hung on the OOM-killed workers). array('b') = 1B/entry.
        self._lens = array("b")
        self._flags = array("b")
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
            np.frombuffer(self._bytes, dtype=np.uint8),  # zero-copy view
        )
        np.save(os.path.join(self.out_dir, "unit_len.npy"), np.frombuffer(self._lens, dtype=np.uint8))
        np.save(os.path.join(self.out_dir, "unit_flag.npy"), np.frombuffer(self._flags, dtype=np.uint8))
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


def _load_shard(d: str, meta: dict, mm: str | None) -> dict[str, Any]:
    return {
        "dir": d,
        "bytes": np.load(os.path.join(d, "bytes.npy"), mmap_mode=mm),
        "unit_len": np.load(os.path.join(d, "unit_len.npy"), mmap_mode=mm),
        "unit_flag": np.load(os.path.join(d, "unit_flag.npy"), mmap_mode=mm),
        "meta": meta,
    }


class ShardRotator:
    """Block-level rotation sampler (2026-09-30 用户拍板: 服务器积德).

    Keeps `resident` shards in RAM (each ~3.5GB for the word cache) instead of
    preloading the whole corpus; windows are drawn uniformly from the resident
    set, and every `every` batch calls the oldest shard is retired and the
    next one (random permutation, seeded) swapped in. The next shard is loaded
    in a BACKGROUND THREAD right after each rotation, so the swap costs no
    training time; IO per rotation = one sequential GPFS read.

    Data-order note (拍板口径): coarse block-random instead of iid — a run
    sees ~2 resident blocks at a time and cycles the full corpus across
    rotations. reader interface = ShardReader over the resident subset.
    """

    def __init__(self, cache_dir: str, resident: int = 4, every: int = 2000,
                 seed: int = 0):
        import threading

        self.cache_dir = cache_dir
        self.every = int(every)
        self.calls = 0
        names = sorted(
            n for n in os.listdir(cache_dir)
            if os.path.isdir(os.path.join(cache_dir, n)) and n.startswith("shard-")
        )
        if len(names) < resident:
            raise ValueError(f"cache has {len(names)} shards < resident {resident}")
        self.perm = random.Random(seed).sample(names, len(names))
        first = self.perm[:resident]
        self.reader = ShardReader(cache_dir, preload=True, only=first)
        self.names: deque[str] = deque(first)
        self.pos = resident  # next position in perm to bring in
        self._threading = threading
        self._bg: tuple[str, Any, list] | None = None  # (name, thread, holder)
        self._start_next()

    def _start_next(self) -> None:
        name = self.perm[self.pos % len(self.perm)]
        holder: list = []

        def work() -> None:
            import traceback as _tb
            try:
                d = os.path.join(self.cache_dir, name)
                with open(os.path.join(d, "meta.json"), encoding="utf-8") as f:
                    meta = json.load(f)
                holder.append(_load_shard(d, meta, None))
            except Exception:
                holder.append(_tb.format_exc())  # surfaced by maybe_rotate

        th = self._threading.Thread(target=work, daemon=True)
        th.start()
        self._bg = (name, th, holder)

    def maybe_rotate(self) -> None:
        """Call once per training batch; rotates on every `every`-th call."""
        self.calls += 1
        if not self.every or self.calls % self.every != 0:
            return
        name, th, holder = self._bg
        th.join()  # almost always finished (had `every` steps to load)
        if not holder:
            raise RuntimeError(f"bg shard load of {name} produced nothing")
        if isinstance(holder[0], str):  # captured bg exception
            raise RuntimeError(f"bg shard load of {name} failed:\n{holder[0]}")
        new = holder[0]
        if new["meta"]["l_max"] != self.reader.l_max:
            raise ValueError(f"shard {name} l_max mismatch")
        # retire oldest; keep shards/_off_ckpt aligned (indexed by position)
        self.reader.shards.pop(0)
        if self.reader._off_ckpt:
            self.reader._off_ckpt.pop(0)
        self.reader.shards.append(new)
        self.names.popleft()
        self.names.append(name)
        self.pos += 1
        self._start_next()
        print(f"[rotate] retired, resident now: {sorted(self.names)}", flush=True)


class ShardReader:
    """Read-time sequence slicer + batch builder over one or more shards."""
    def __init__(self, cache_dir: str, preload: bool = False,
                 only: list[str] | None = None):
        self.cache_dir = cache_dir
        self.shards: list[dict[str, Any]] = []
        for name in sorted(os.listdir(cache_dir)):
            d = os.path.join(cache_dir, name)
            if not (os.path.isdir(d) and name.startswith("shard-")):
                continue
            if only is not None and name not in only:
                continue
            with open(os.path.join(d, "meta.json"), encoding="utf-8") as f:
                meta = json.load(f)
            # preload (2026-09-29 prof finding): mmap over GPFS costs ~10ms per
            # page fault and dominates the batch-build path (~1.6s/step at
            # batch 36). Loading the arrays into RAM once (sequential read)
            # kills per-step IO entirely — request enough --mem.
            mm = None if preload else "r"
            self.shards.append(_load_shard(d, meta, mm))
        if not self.shards:
            raise FileNotFoundError(f"no shards found under {cache_dir}")
        self.l_max = int(self.shards[0]["meta"]["l_max"])
        # sparse unit->byte offset checkpoints (one int64 per OFF_BLOCK units),
        # built lazily per shard. 2026-09-13 cluster lesson: a full int64 cumsum
        # per shard is 573M x 8B = 4.6GB *anonymous* — a batch touching ~8
        # shards allocated ~37GB and got OOM-killed under the 48G cgroup
        # (bench bltz arm; the token arm slices by arithmetic and was immune).
        # Sparse ckpt: ~1.1MB per shard, probe = sum of <=OFF_BLOCK uint8s.
        self._off_ckpt: list[np.ndarray] = []

    _OFF_BLOCK = 4096

    def n_sequences(self, n_patches: int) -> int:
        return sum(s["meta"]["n_units"] // n_patches for s in self.shards)

    def _ckpt(self, si: int) -> np.ndarray:
        """Sparse exclusive prefix sums: ck[j] = byte offset of unit j*_OFF_BLOCK."""
        while len(self._off_ckpt) <= si:
            k = len(self._off_ckpt)
            lens = self.shards[k]["unit_len"]  # memmap uint8
            n = int(self.shards[k]["meta"]["n_units"])
            ck = np.empty(n // self._OFF_BLOCK + 1, dtype=np.int64)
            ck[0] = 0
            acc = 0
            B = self._OFF_BLOCK
            for j in range(1, len(ck)):
                acc += int(np.asarray(lens[(j - 1) * B : j * B], dtype=np.int64).sum())
                ck[j] = acc
            self._off_ckpt.append(ck)
        return self._off_ckpt[si]

    def _byte_off(self, si: int, u: int) -> int:
        """Byte offset of unit u (bytes preceding it) via sparse ckpt + local scan."""
        ck = self._ckpt(si)
        j, r = divmod(u, self._OFF_BLOCK)
        if r == 0:
            return int(ck[j])
        lens = self.shards[si]["unit_len"]
        return int(ck[j] + np.asarray(lens[j * self._OFF_BLOCK : u], dtype=np.int64).sum())

    def get_units(self, si: int, u0: int, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Units [u0, u0+n) of shard si -> (flat bytes, lens, flags)."""
        s = self.shards[si]
        b0 = 0 if u0 == 0 else self._byte_off(si, u0)
        b1 = self._byte_off(si, u0 + n)
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

    def sequence_units_flags(self, global_idx: int, n_patches: int) -> tuple[list[bytes], list[bool]]:
        """Raw unit byte strings + doc-start flags (True = first unit of a
        document; from the cache's unit_flag — 2026-10-03 边界掩码信号)."""
        si, u0 = self._seq_location(global_idx, n_patches)
        flat, lens, flags = self.get_units(si, u0, n_patches)
        units: list[bytes] = []
        pos = 0
        for Ln in lens:
            n_bytes = int(Ln)  # uint8 scalar would overflow on pos arithmetic (NumPy 2)
            units.append(bytes(flat[pos : pos + n_bytes]))
            pos += n_bytes
        return units, [bool(f) for f in np.asarray(flags).tolist()]

    def sequence_units(self, global_idx: int, n_patches: int) -> list[bytes]:
        """Raw unit byte strings (needed for the augmentation path)."""
        return self.sequence_units_flags(global_idx, n_patches)[0]

    def sequence_flags(self, global_idx: int, n_patches: int) -> list[bool]:
        """Doc-start flags only (cheap path — no unit slicing)."""
        si, u0 = self._seq_location(global_idx, n_patches)
        return [bool(f) for f in np.asarray(
            self.shards[si]["unit_flag"][u0 : u0 + n_patches]).tolist()]

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
        pos = self._byte_off(si, extra_u0)
        for Ln in np.asarray(s["unit_len"][extra_u0 : extra_u0 + need]):
            n_bytes = int(Ln)  # uint8 scalar would overflow on pos arithmetic (NumPy 2)
            units.append(bytes(np.asarray(s["bytes"][pos : pos + n_bytes])))
            pos += n_bytes
        return units[:n_patches]

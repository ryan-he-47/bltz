"""T3.1 packing waste audit (docs/24): real patch-length distribution, padding
ratio of the fixed (S, l_max) byte canvas, per-row byte totals. READ-ONLY.
Run:  python -u scripts/diag_packing.py [--set data.cache_dir=data/cache_dryrun]
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from bltz.config import parse_cli
from bltz.shards import ShardReader


def audit(reader, S: int, L: int, n_rows: int, augment: bool, p_split: float, p_merge: float, rng) -> dict:
    n_seq = reader.n_sequences(S)
    idx = [rng.randrange(n_seq) for _ in range(n_rows)]
    batch = reader.make_batch(idx, S, augment=augment, p_split=p_split, p_merge=p_merge, rng=rng)
    ends = batch["ends"].numpy()  # (B, S) inclusive cumulative byte ends
    flat_len = batch["flat_len"].numpy()
    B = ends.shape[0]
    lens = np.diff(ends, axis=1, prepend=0).reshape(-1)
    for b in range(B):
        assert lens.reshape(B, -1)[b].sum() == flat_len[b], f"len/flat mismatch row {b}"
    total = flat_len.astype(np.float64)
    return {
        "lens": lens,
        "row_bytes": total,
        "pad_ratio": 1.0 - total.sum() / (B * S * L),
    }


def main() -> None:
    cfg = parse_cli("configs/default.yaml")
    reader = ShardReader(cfg.data.cache_dir)
    S = cfg.data.n_patches
    L = cfg.segment.l_max
    n_rows = int(cfg.get("diag", {}).get("n_rows", 256)) if cfg.get("diag") else 256
    rng = random.Random(0)

    for augment in (False, True):
        r = audit(reader, S, L, n_rows, augment, cfg.segment.p_split, cfg.segment.p_merge, rng)
        lens = r["lens"].astype(np.int64)
        hist = np.bincount(lens, minlength=L + 1)[1 : L + 1]
        print(f"\n=== augment={augment} (n_rows={n_rows}, S={S}) ===")
        print(f"patch bytes: mean {lens.mean():.2f} p50 {np.percentile(lens,50):.0f} "
              f"p90 {np.percentile(lens,90):.0f} p99 {np.percentile(lens,99):.0f} max {lens.max()}")
        print(f"hist 1..{L}: {hist.tolist()}")
        print(f"padding ratio of (S,{L}) canvas: {r['pad_ratio']*100:.1f}%")
        rb = r["row_bytes"]
        print(f"row total bytes: mean {rb.mean():.0f} p10 {np.percentile(rb,10):.0f} "
              f"p90 {np.percentile(rb,90):.0f} min {rb.min():.0f} max {rb.max():.0f}")
        print(f"canvas bytes {S*L} -> real {rb.mean():.0f}; encoder waste factor "
              f"{S*L/rb.mean():.2f}x")
    print("\ndiag_packing.py: DONE")


if __name__ == "__main__":
    main()

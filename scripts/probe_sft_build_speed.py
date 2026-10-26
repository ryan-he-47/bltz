"""Throughput probe for the SFT cache build (2026-10-26, refine docs/48 budget).

Measures on the cluster node:
  * parquet row iteration speed (rows/s, bytes/s);
  * `segment_word` speed on rendered conversations (convs/s, units/s);
  * extrapolation to the full 484,570-conversation dataset for the
    segmentation half of the cache build (the AE-verify half runs on GPU and
    was measured separately by the training-side online verify).

CPU-only, a few seconds:
  python scripts/probe_sft_build_speed.py --data-dir DIR --tok DIR --n 300
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pyarrow.parquet as pq

from bltz.segment_v2 import segment_word

MARK = {"system": b"<|system|>", "user": b"<|user|>", "assistant": b"<|assistant|>",
        "end": b"<|end|>"}


def render(row) -> bytes:
    parts = []
    for m in row["messages"]:
        role = str(m.get("role", "")).lower()
        if role in MARK:
            parts.append(MARK[role])
        parts.append(str(m.get("content", "")).encode("utf-8", errors="replace"))
        if role == "assistant":
            parts.append(MARK["end"])
    return b"".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--tok", required=True)
    ap.add_argument("--n", type=int, default=300)
    args = ap.parse_args()

    files = sorted(Path(args.data_dir).glob("*.parquet"))
    # ---- parquet read speed ----
    t0 = time.time()
    rows = []
    for f in files:
        for b in pq.ParquetFile(f).iter_batches(batch_size=512):
            rows.extend(b.to_pylist())
            if len(rows) >= args.n:
                break
        if len(rows) >= args.n:
            break
    t_read = time.time() - t0
    rows = rows[: args.n]
    print(f"[speed] read {len(rows)} rows in {t_read:.2f}s "
          f"({len(rows)/t_read:.0f} rows/s)")

    # ---- render + segment speed ----
    t0 = time.time()
    total_units = total_bytes = 0
    texts = []
    for r in rows:
        texts.append(render(r))
    t_render = time.time() - t0

    t0 = time.time()
    units = []
    for tx in texts:
        u = segment_word(tx, args.tok, l_max=32)
        units.append(len(u))
        total_units += len(u)
        total_bytes += len(tx)
    t_seg = time.time() - t0
    n = len(texts)
    convs_s = n / max(t_seg, 1e-9)
    units_s = total_units / max(t_seg, 1e-9)
    print(f"[speed] render {n} convs in {t_render:.2f}s")
    print(f"[speed] segment {n} convs ({total_bytes/1e6:.2f}MB, "
          f"{total_units} units = {total_units/n:.0f}/conv) in {t_seg:.2f}s")
    print(f"[speed] -> {convs_s:.0f} convs/s, {units_s/1e6:.3f}M units/s "
          f"(single process)")

    TOTAL_CONVS = 484_570
    TOTAL_UNITS = 396_000_000
    print(f"\n[speed] extrapolation to {TOTAL_CONVS} convs / ~396M units:")
    print(f"  segmentation single-process: {TOTAL_CONVS/convs_s/60:.1f} min")
    print(f"  (with 4 workers: {TOTAL_CONVS/convs_s/60/4:.1f} min)")
    print(f"  parquet read (4.7GB): {(TOTAL_CONVS/max(len(rows)/t_read,1e-9))/60:.1f} min")
    print("  NOTE: AE verify half runs on GPU; measured next in the build job.")


if __name__ == "__main__":
    main()

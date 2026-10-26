"""AE-verify throughput on GPU for the SFT cache build (docs/48 budget).

Loads the frozen word AE on CUDA, renders+segments a small sample of real
smol-smoltalk conversations, then runs the same batched VerifyResplitter
forward used by the training-side online verify, and extrapolates to the full
~396M-unit dataset (the GPU half of the cache build; segmentation half was
measured as ~10-40 min on CPU).

  python scripts/probe_sft_verify_speed.py --data-dir DIR --tok DIR --ae CKPT
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pyarrow.parquet as pq
import torch

from bltz.config import Cfg
from bltz.models.autoencoder import ByteStringAE
from bltz.segment_v2 import segment_word
from bltz.verify import VerifyResplitter

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
    ap.add_argument("--ae", required=True)
    ap.add_argument("--n", type=int, default=60)
    args = ap.parse_args()

    print(f"[verify-speed] cuda={torch.cuda.is_available()} "
          f"({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'})")
    st = torch.load(args.ae, map_location="cpu", weights_only=False)
    ae = ByteStringAE(Cfg(st["cfg"]))
    ae.load_state_dict(st["model"])
    ae = ae.to("cuda").eval()
    ver = VerifyResplitter(ae, args.tok, ent_thresh=0.2, floor_bytes=3,
                           device="cuda", log_every=0)

    files = sorted(Path(args.data_dir).glob("*.parquet"))
    rows = []
    for f in files:
        for b in pq.ParquetFile(f).iter_batches(batch_size=256):
            rows.extend(b.to_pylist())
            if len(rows) >= args.n:
                break
        if len(rows) >= args.n:
            break
    rows = rows[: args.n]

    t0 = time.time()
    units = []
    for r in rows:
        units.extend(segment_word(render(r), args.tok, l_max=32))
    t_seg = time.time() - t0
    print(f"[verify-speed] {len(rows)} convs -> {len(units)} units "
          f"(segment {t_seg:.2f}s, CPU)")

    # warmup then timed verify (exactly the path used at cache build / training)
    ver._verify(units[:2048])
    torch.cuda.synchronize()
    t0 = time.time()
    em, ent = ver._verify(units)
    torch.cuda.synchronize()
    dt = time.time() - t0
    flag = float(((~em).sum() + (ent > ver.ent_thresh).sum()) / max(len(units), 1))
    print(f"[verify-speed] verify {len(units)} units in {dt:.3f}s "
          f"({len(units)/dt/1e6:.3f}M units/s, flag≈{flag:.2%})")

    TOTAL = 396_000_000
    print(f"[verify-speed] extrapolation: {TOTAL/1e6:.0f}M units / "
          f"{len(units)/dt/1e6:.2f}M/s = {TOTAL/max(len(units)/dt,1)/60:.1f} min GPU")
    print("[verify-speed] NOTE: flagged units additionally go through BPE-resplit "
          "+ a second verify pass (~0.6% of units) -> add ~10-20%.")


if __name__ == "__main__":
    main()

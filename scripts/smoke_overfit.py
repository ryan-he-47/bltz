"""Smoke 1: single-batch overfit. A correct MTP implementation must drive the
loss on a FIXED batch close to 0. Run: python scripts/smoke_overfit.py"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from bltz.config import parse_cli
from bltz.data import build_batch
from bltz.models import BltzLM
from bltz.segment import Segmenter
from bltz.trainer import train

TEXT = (
    "The Byte Latent Transformer encodes bytes into dynamically sized patches, "
    "which serve as the primary units of computation. Patches are segmented "
    "based on the entropy of the next byte, allocating more compute and model "
    "capacity where increased data complexity demands it. We present the first "
    "flop controlled scaling study of byte-level models up to 8B parameters. "
) * 4


def main() -> None:
    cfg = parse_cli("configs/smoke.yaml")
    torch.manual_seed(cfg.train.seed)
    seg = Segmenter(cfg.segment.l_max, 0.0, 0.0)  # no augmentation for overfit
    batch = build_batch([TEXT], seg, cfg.data.n_patches, cfg.segment.l_max)
    n_seq = batch["byte_ids"].shape[0]
    keep = min(4, n_seq)
    batch = {k: v[:keep] for k, v in batch.items()}  # ONE fixed batch
    print(f"overfit batch: B={keep}, S={cfg.data.n_patches}, flat_len={batch['flat_len'][0].item()}")

    model = BltzLM(cfg).cuda()
    history = train(model, lambda: batch, cfg)

    losses = [r["loss"] for r in history if "loss" in r]
    first, last = losses[0], losses[-1]
    print(f"loss: {first:.4f} -> {last:.4f}")
    if last > 0.5:
        raise AssertionError(f"overfit failed: final loss {last:.4f} > 0.5")
    print("smoke_overfit.py: PASS")


if __name__ == "__main__":
    main()

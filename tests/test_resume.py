"""Script-style test for breakpoint resume: train -> kill -> resume must continue
steps and loss trajectory (not restart from scratch). Run: python tests/test_resume.py"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from bltz.config import load
from bltz.data import build_batch
from bltz.models import BltzLM
from bltz.segment import Segmenter
from bltz.trainer import train

TEXT = (
    "The Byte Latent Transformer encodes bytes into dynamically sized patches, "
    "allocating more compute where data complexity demands it. " * 8
)
TMP = Path("checkpoints/test_resume_tmp")


def check(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def main() -> None:
    cfg = load("configs/smoke.yaml")
    seg = Segmenter(cfg.segment.l_max, 0.0, 0.0)
    batch = build_batch([TEXT], seg, cfg.data.n_patches, cfg.segment.l_max)
    batch = {k: v[:2] for k, v in batch.items()}
    shutil.rmtree(TMP, ignore_errors=True)

    torch.manual_seed(cfg.train.seed)
    model = BltzLM(cfg).cuda()
    cfg.train.steps = 15
    cfg.train.ckpt_dir = str(TMP)
    cfg.train.log_every = 5
    cfg.train.ckpt_every = 5
    h1 = train(model, lambda: batch, cfg)
    losses1 = [r["loss"] for r in h1 if "loss" in r]
    check((TMP / "ckpt_full.pt").exists(), "ckpt_full.pt not written")
    last1 = losses1[-1]

    torch.manual_seed(999)  # different init: proves the state actually loads
    model2 = BltzLM(cfg).cuda()
    cfg.train.steps = 25
    cfg.train.resume = str(TMP / "ckpt_full.pt")
    h2 = train(model2, lambda: batch, cfg)
    steps2 = [r["step"] for r in h2 if "loss" in r]
    losses2 = [r["loss"] for r in h2 if "loss" in r]
    check(steps2[0] >= 15, f"resume did not continue steps: {steps2}")
    check(losses2[0] < last1 + 1.0, f"resume loss jumped: {losses2[0]} vs {last1}")
    check(losses2[-1] <= losses2[0] or losses2[-1] < last1, f"no progress after resume: {losses2}")
    print(f"phase1 last {last1:.4f}@14; resumed {losses2[0]:.4f}@{steps2[0]} -> {losses2[-1]:.4f}@{steps2[-1]}")

    shutil.rmtree(TMP, ignore_errors=True)
    print("test_resume.py: ALL PASS")


if __name__ == "__main__":
    main()

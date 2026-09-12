"""Script-style integration test: the exact training path of the twin run —
ShardReader.make_batch(augment=True) -> trainer. Run: python tests/test_shard_train.py"""
from __future__ import annotations

import random
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from bltz.config import load
from bltz.models import BltzLM
from bltz.segment import Segmenter
from bltz.shards import ShardReader, ShardWriter
from bltz.trainer import train

TEXTS = [
    "The Byte Latent Transformer encodes bytes into dynamically sized patches. " * 30,
    "Entropy patching allocates compute where data complexity demands it. " * 30,
    "混合文本 stress test:你好世界 with English words and 数字 123。" * 30,
]
TMP = Path("data/test_shard_train_tmp")


def check(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def main() -> None:
    cfg = load("configs/smoke.yaml")
    l_max = cfg.segment.l_max
    S = cfg.data.n_patches

    shutil.rmtree(TMP, ignore_errors=True)
    seg = Segmenter(l_max, 0.0, 0.0)
    writer = ShardWriter(str(TMP / "shard-00000"), l_max)
    for t in TEXTS:
        writer.add_document_units(seg.segment_bytes(t.encode("utf-8")))
    meta = writer.close(cfg.segment.to_dict())
    print(f"mini cache: {meta['n_units']} units")

    reader = ShardReader(str(TMP))
    n_seq = reader.n_sequences(S)
    check(n_seq >= 8, f"not enough sequences: {n_seq}")
    rng = random.Random(0)

    torch.manual_seed(cfg.train.seed)
    model = BltzLM(cfg).cuda()

    def batch_fn() -> dict[str, torch.Tensor]:
        idx = [rng.randrange(n_seq) for _ in range(4)]
        return reader.make_batch(
            idx, S, augment=True,
            p_split=cfg.segment.p_split, p_merge=cfg.segment.p_merge, rng=rng,
        )

    cfg.train.steps = 40
    cfg.train.log_every = 10
    cfg.train.ckpt_dir = str(TMP / "ckpt")
    history = train(model, batch_fn, cfg)
    losses = [r["loss"] for r in history if "loss" in r]
    print(f"loss: {losses[0]:.4f} -> {losses[-1]:.4f}")
    check(losses[-1] < losses[0], "loss did not decrease on shard-reader path")

    shutil.rmtree(TMP, ignore_errors=True)
    print("test_shard_train.py: ALL PASS")


if __name__ == "__main__":
    main()

"""Train the bltz twin arm (Stage 1/2).
Run: python -u scripts/train_bltz.py [configs/default.yaml] [--set a.b=value]
  resume: --set train.resume=checkpoints/twin/ckpt_full.pt"""
from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from bltz.config import parse_cli
from bltz.models import BltzLM
from bltz.shards import ShardReader
from bltz.trainer import train


def main() -> None:
    cfg = parse_cli("configs/default.yaml")
    reader = ShardReader(cfg.data.cache_dir)
    S = cfg.data.n_patches
    n_seq = reader.n_sequences(S)
    print(f"byte cache: {n_seq} sequences at S={S}")

    torch.manual_seed(cfg.train.seed)
    model = BltzLM(cfg).cuda()
    print(f"BltzLM params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

    rng = random.Random(cfg.train.seed)

    def batch_fn() -> dict[str, torch.Tensor]:
        idx = [rng.randrange(n_seq) for _ in range(cfg.train.batch)]
        return reader.make_batch(
            idx, S, augment=True,
            p_split=cfg.segment.p_split, p_merge=cfg.segment.p_merge, rng=rng,
        )

    train(model, batch_fn, cfg)


if __name__ == "__main__":
    main()

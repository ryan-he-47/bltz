"""Kernel-level profile of ONE backbone forward — where do the 3.7s actually go?
Run: python scripts/prof_backbone.py [--set train.batch=8]"""
from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from bytefield.config import parse_cli
from bytefield.models import ByteFieldLM
from bytefield.shards import ShardReader


def main() -> None:
    cfg = parse_cli("configs/default.yaml")
    reader = ShardReader(cfg.data.cache_dir)
    S = cfg.data.n_patches
    rng = random.Random(0)
    idx = [rng.randrange(reader.n_sequences(S)) for _ in range(cfg.train.batch)]
    batch = reader.make_batch(idx, S)
    batch = {k: v.cuda() for k, v in batch.items()}

    torch.manual_seed(0)
    model = ByteFieldLM(cfg).cuda()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        lat = model.encoder(batch["byte_ids"], batch["pad_mask"])
        for _ in range(3):  # warmup
            model.backbone(lat)
    torch.cuda.synchronize()

    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            model.backbone(lat)
        torch.cuda.synchronize()
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=22))
    print("prof_backbone.py: DONE")


if __name__ == "__main__":
    main()

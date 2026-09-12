"""Profile ONE full training step (mtp_loss fwd + backward) — find the backward
pathology. Run: python scripts/prof_step.py [--set train.batch=8]"""
from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from bltz.config import parse_cli
from bltz.models import BltzLM
from bltz.objectives import mtp_loss
from bltz.shards import ShardReader


def main() -> None:
    cfg = parse_cli("configs/default.yaml")
    reader = ShardReader(cfg.data.cache_dir)
    S = cfg.data.n_patches
    rng = random.Random(0)
    idx = [rng.randrange(reader.n_sequences(S)) for _ in range(cfg.train.batch)]
    batch = reader.make_batch(idx, S)
    batch = {k: v.cuda() for k, v in batch.items()}

    torch.manual_seed(0)
    model = BltzLM(cfg).cuda()

    def step():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = mtp_loss(model, batch, cfg)
        model.zero_grad(set_to_none=True)
        loss.backward()

    for _ in range(2):  # warmup: autotune, allocator, autograd setup
        step()
    torch.cuda.synchronize()

    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        step()
        torch.cuda.synchronize()
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
    print("prof_step.py: DONE")


if __name__ == "__main__":
    main()

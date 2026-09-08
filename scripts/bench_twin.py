"""Twin-scale probe: N steps on configs/default.yaml reading the dryrun cache.
Reports params, ms/step, peak VRAM — real numbers for the Stage-1 budget
decision. NOT the full run; ~20 steps, seconds. Run:
  python scripts/bench_twin.py [--set train.steps=30] [--set data.cache_dir=data/cache_dryrun]"""
from __future__ import annotations

import random
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from bytefield.config import parse_cli
from bytefield.models import ByteFieldLM
from bytefield.shards import ShardReader
from bytefield.trainer import train


def main() -> None:
    cfg = parse_cli("configs/default.yaml")
    cfg.train.steps = int(cfg.get("bench", {}).get("steps", 20)) if cfg.get("bench") else 20
    cfg.train.ckpt_dir = "checkpoints/bench_tmp"
    cfg.train.log_every = 5

    reader = ShardReader(cfg.data.cache_dir)
    S = cfg.data.n_patches
    n_seq = reader.n_sequences(S)
    print(f"cache: {cfg.data.cache_dir} -> {n_seq} sequences at S={S}")
    if n_seq < 8:
        raise RuntimeError("not enough sequences in cache for the probe")

    torch.manual_seed(cfg.train.seed)
    model = ByteFieldLM(cfg).cuda()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model params: {n_params/1e6:.1f}M (twin target ~124M)")

    rng = random.Random(0)

    def batch_fn() -> dict[str, torch.Tensor]:
        idx = [rng.randrange(n_seq) for _ in range(cfg.train.batch)]
        return reader.make_batch(
            idx, S, augment=True,
            p_split=cfg.segment.p_split, p_merge=cfg.segment.p_merge, rng=rng,
        )

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t0 = time.time()
    history = train(model, batch_fn, cfg)
    torch.cuda.synchronize()
    dt = time.time() - t0

    losses = [r["loss"] for r in history if "loss" in r]
    peak = torch.cuda.max_memory_allocated() / 2**20
    ms = dt / cfg.train.steps * 1000
    print(f"\nsteps={cfg.train.steps} batch={cfg.train.batch} S={S}")
    print(f"loss: {losses[0]:.4f} -> {losses[-1]:.4f}")
    print(f"speed: {ms:.0f} ms/step -> est. 20k steps = {ms*20000/3600000:.1f} h")
    print(f"peak VRAM: {peak:.0f} MiB (budget 8192)")

    shutil.rmtree("checkpoints/bench_tmp", ignore_errors=True)
    print("bench_twin.py: DONE")


if __name__ == "__main__":
    main()

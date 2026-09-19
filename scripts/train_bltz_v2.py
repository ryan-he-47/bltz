"""Train the v2 backbone POC (docs/32 S3, frozen-AE arm).

  python -u scripts/train_bltz_v2.py configs/v2.yaml \
      --set train.ae_ckpt=<ae best.pt> [--set train.resume=...]

Segmentation augmentation is BAKED into the v2 cache (2026-09-17 拍板) ->
augment=False at read time. Everything else reuses the v1 trainer (WSD,
spike_skip, graceful interrupt, milestones).
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from bltz.config import Cfg, parse_cli
from bltz.models.autoencoder import ByteStringAE
from bltz.models.model_v2 import BltzLMv2
from bltz.objectives_v2 import mdn_nll_loss
from bltz.shards import ShardReader
from bltz.trainer import train


def main() -> None:
    cfg = parse_cli("configs/v2.yaml")
    reader = ShardReader(cfg.data.cache_dir)
    S = int(cfg.data.n_patches)
    n_seq = reader.n_sequences(S)
    print(f"v2 cache: {n_seq} sequences at S={S}", flush=True)

    ae_state = torch.load(cfg.train.ae_ckpt, map_location="cpu", weights_only=False)
    ae = ByteStringAE(Cfg(ae_state["cfg"]))
    ae.load_state_dict(ae_state["model"])
    print(f"[v2] frozen AE from {cfg.train.ae_ckpt} (step {ae_state['step']})", flush=True)

    torch.manual_seed(int(cfg.train.seed))
    model = BltzLMv2(cfg, ae).cuda()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"BltzLMv2 params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M "
          f"(trainable {trainable/1e6:.1f}M)", flush=True)

    rng = random.Random(int(cfg.train.seed))

    def batch_fn() -> dict[str, torch.Tensor]:
        idx = [rng.randrange(n_seq) for _ in range(int(cfg.train.batch))]
        return reader.make_batch(idx, S, augment=False)

    train(model, batch_fn, cfg, loss_fn=mdn_nll_loss)


if __name__ == "__main__":
    main()

"""Train the v2 backbone (docs/32 S3, frozen-AE arm; default: linear MDN head).

  python -u scripts/train_bltz_v2.py configs/v2.yaml \
      --set train.ae_ckpt=<ae best.pt> [--set train.resume=...]

  torchrun --nproc_per_node=2 scripts/train_bltz_v2.py ...   (DDP 2026-09-27)
  Each rank draws disjoint random batches (seed + 7919*rank); gradient
  allreduce makes gn/found_inf rank-consensus by construction (inf spreads
  through the average), so the spike-skip guard needs no extra sync. ckpt
  format is unchanged (DDP/wrapper layers stripped by trainer._state_model).

Segmentation augmentation is BAKED into the v2 cache (2026-09-17 拍板) ->
augment=False at read time. Everything else reuses the v1 trainer (WSD,
spike_skip, graceful interrupt, milestones).

Online AE verify+resplit (2026-09-29 拍板, train.verify.*): each window's
units pass EM/entropy gating against the frozen AE; failures are BPE-resplit,
re-verified, and degraded to <=floor_bytes char chunks as the floor. Online
(not baked into the cache) so encoder/threshold iterations need no rebuild.
"""
from __future__ import annotations

import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn

from bltz.config import Cfg, parse_cli
from bltz.data import collate_sequences, tensorize_units
from bltz.models.autoencoder import ByteStringAE
from bltz.models.model_v2 import BltzLMv2
from bltz.objectives_v2 import mdn_nll_loss
from bltz.shards import ShardReader
from bltz.trainer import train


def build_batch(reader: ShardReader, verifier, idx: list[int], S: int,
                l_max: int) -> dict[str, torch.Tensor]:
    """One training window batch. verifier=None -> raw cache path (unchanged);
    else online verify+resplit per window (bltz/verify.py, 2026-09-29 拍板:
    在线做不碰缓存), trimmed to exactly S units."""
    if verifier is None:
        return reader.make_batch(idx, S, augment=False)
    seqs = []
    for gi in idx:
        units = reader.sequence_units(gi, S)
        units = verifier.process(units)
        if len(units) < S:  # cannot happen (splits only grow) — belt & braces
            units = reader.sequence_units(gi, S)
        units = units[:S]
        flat = torch.from_numpy(
            np.frombuffer(b"".join(units), dtype=np.uint8).copy()).long()
        lens = torch.tensor([len(u) for u in units], dtype=torch.long)
        seqs.append(tensorize_units(flat, lens, l_max))
    return collate_sequences(seqs)


class _LossModule(nn.Module):
    """DDP vehicle: forward computes the loss so EVERY trainable param (incl.
    the MDN head, which lives outside BltzLMv2.forward) participates in the
    DDP forward pass -> no find_unused_parameters needed. `core` is the state
    carrier; trainer._state_model() strips it for ckpt I/O."""

    def __init__(self, lm: BltzLMv2):
        super().__init__()
        self.core = lm

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        h, lam = self.core(batch["byte_ids"], batch["pad_mask"])
        return self.core.head.nll(h, lam).mean()


def main() -> None:
    cfg = parse_cli("configs/v2.yaml")
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world > 1
    if distributed:
        torch.cuda.set_device(local_rank)
        dist.init_process_group("nccl")

    # attention backend (2026-09-27 scale prep): V100 has no flash (sm80+);
    # mem-efficient (xformers cutlass lineage) is the memory-safe path for
    # S=4096. math stays available as fallback unless disabled.
    mcfg = cfg.model
    torch.backends.cuda.enable_flash_sdp(bool(mcfg.get("attn_flash", False)))
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(bool(mcfg.get("attn_math", True)))
    if rank == 0:
        cap = torch.cuda.get_device_capability()
        print(f"[attn] flash={torch.backends.cuda.flash_sdp_enabled()} "
              f"mem_efficient={torch.backends.cuda.mem_efficient_sdp_enabled()} "
              f"math={torch.backends.cuda.math_sdp_enabled()} "
              f"cap={cap} (flash needs sm80+)", flush=True)

    reader = ShardReader(cfg.data.cache_dir)
    S = int(cfg.data.n_patches)
    n_seq = reader.n_sequences(S)
    if rank == 0:
        print(f"v2 cache: {n_seq} sequences at S={S} (world={world})", flush=True)

    ae_state = torch.load(cfg.train.ae_ckpt, map_location="cpu", weights_only=False)
    ae = ByteStringAE(Cfg(ae_state["cfg"]))
    ae.load_state_dict(ae_state["model"])
    if rank == 0:
        print(f"[v2] frozen AE from {cfg.train.ae_ckpt} (step {ae_state['step']})", flush=True)

    torch.manual_seed(int(cfg.train.seed))  # same init on every rank (DDP bcasts)
    model = BltzLMv2(cfg, ae).cuda()
    if rank == 0:
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"BltzLMv2 params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M "
              f"(trainable {trainable/1e6:.1f}M)", flush=True)

    train_mod: nn.Module = model
    loss_fn = mdn_nll_loss
    if distributed:
        train_mod = torch.nn.parallel.DistributedDataParallel(
            _LossModule(model), device_ids=[local_rank]
        )
        loss_fn = lambda m, b, c: m(b)  # DDP forward -> loss

    rng = random.Random(int(cfg.train.seed) + 7919 * rank)

    # online verify+resplit (train.verify.*): the frozen AE judges its own
    # units; failures are replaced by verified pieces. Deterministic given
    # (AE ckpt, tokenizer, thresholds) — pin all three per run.
    verifier = None
    vcfg = cfg.train.get("verify", None)
    if vcfg and bool(vcfg.get("enabled", False)):
        from bltz.verify import VerifyResplitter

        verifier = VerifyResplitter(
            model.ae, str(vcfg.tok_dir),
            ent_thresh=float(vcfg.get("ent_thresh", 0.2)),
            floor_bytes=int(vcfg.get("floor_bytes", 3)),
        )
        if rank == 0:
            print(f"[verify] online resplit ON: TH={verifier.ent_thresh} bits, "
                  f"floor<={verifier.floor_bytes}B", flush=True)

    def batch_fn() -> dict[str, torch.Tensor]:
        idx = [rng.randrange(n_seq) for _ in range(int(cfg.train.batch))]
        return build_batch(reader, verifier, idx, S, reader.l_max)

    train(train_mod, batch_fn, cfg, loss_fn=loss_fn, is_rank0=(rank == 0))
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

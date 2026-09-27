"""Scale smoke (2026-09-27): 0.5B config at S=4096 on 2x V100.

Builds the model once, then for each per-GPU batch size runs a few real
train steps (fwd/bwd/optim) and reports peak VRAM + step time + attention
backend flags. Decides the batch/LR strategy for the full scale run.

  torchrun --nproc_per_node=2 scripts/smoke_scale.py configs/v2.yaml \
    --set model.d_model=1024 --set model.bb_layers=28 --set model.bb_heads=16 \
    --set data.n_patches=4096 [--set data.cache_dir=... --set train.ae_ckpt=...]
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.distributed as dist

from bltz.config import Cfg, parse_cli
from bltz.models.autoencoder import ByteStringAE
from bltz.models.model_v2 import BltzLMv2
from bltz.shards import ShardReader
from train_bltz_v2 import _LossModule


def main() -> None:
    cfg = parse_cli("configs/v2.yaml")
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    torch.cuda.set_device(local_rank)
    if world > 1:
        dist.init_process_group("nccl")

    mcfg = cfg.model
    torch.backends.cuda.enable_flash_sdp(bool(mcfg.get("attn_flash", False)))
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(bool(mcfg.get("attn_math", True)))
    cap = torch.cuda.get_device_capability()
    if rank == 0:
        print(f"[attn] flash={torch.backends.cuda.flash_sdp_enabled()} "
              f"mem_efficient={torch.backends.cuda.mem_efficient_sdp_enabled()} "
              f"math={torch.backends.cuda.math_sdp_enabled()} cap={cap}", flush=True)

    S = int(cfg.data.n_patches)
    reader = ShardReader(cfg.data.cache_dir)
    n_seq = reader.n_sequences(S)
    ae_state = torch.load(cfg.train.ae_ckpt, map_location="cpu", weights_only=False)
    ae = ByteStringAE(Cfg(ae_state["cfg"]))
    ae.load_state_dict(ae_state["model"])
    model = BltzLMv2(cfg, ae).cuda()
    module = _LossModule(model)
    if world > 1:
        module = torch.nn.parallel.DistributedDataParallel(module, device_ids=[local_rank])
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if rank == 0:
        print(f"model: {n_params/1e6:.1f}M trainable, S={S}, n_seq={n_seq}, world={world}", flush=True)

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    batch_sizes = [int(x) for x in (os.environ.get("SMOKE_BATCHES", "1,2,4,8")).split(",")]
    for bs in batch_sizes:
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            t_steps = []
            for i in range(8):
                idx = [(rank * 99991 + i * 7919 + j) % n_seq for j in range(bs)]
                batch = {k: v.cuda() for k, v in reader.make_batch(idx, S, augment=False).items()}
                torch.cuda.synchronize()
                t0 = time.time()
                opt.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.float16):
                    loss = module(batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                torch.cuda.synchronize()
                if i >= 2:  # skip warmup
                    t_steps.append(time.time() - t0)
            peak = torch.cuda.max_memory_allocated() / 2**30
            msg = (f"[smoke] batch={bs} ok | peak {peak:.2f} GiB | "
                   f"{sum(t_steps)/len(t_steps):.2f} s/step | loss {float(loss.detach()):.2f}")
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            msg = f"[smoke] batch={bs} OOM"
        if rank == 0:
            print(msg, flush=True)
        if world > 1:
            dist.barrier()
    if rank == 0:
        print("[smoke] DONE", flush=True)
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

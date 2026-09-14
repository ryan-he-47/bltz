"""Per-module timing probe at twin scale: where does a training step's time go?
Times encoder / backbone / head-queries / full fwd+bwd separately.
Run: python scripts/bench_parts.py [--set train.batch=8]"""
from __future__ import annotations

import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from bltz.config import parse_cli
from bltz.models import BltzLM
from bltz.objectives import mtp_targets
from bltz.shards import ShardReader

REPS = 5


def _time(fn, reps=REPS) -> float:
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t0) / reps * 1000


def main() -> None:
    cfg = parse_cli("configs/default.yaml")
    reader = ShardReader(cfg.data.cache_dir)
    S = cfg.data.n_patches
    n_seq = reader.n_sequences(S)
    rng = random.Random(0)
    idx = [rng.randrange(n_seq) for _ in range(cfg.train.batch)]
    batch = reader.make_batch(idx, S, augment=True, p_split=0.1, p_merge=0.1, rng=rng)
    batch = {k: v.cuda() for k, v in batch.items()}

    torch.manual_seed(0)
    model = BltzLM(cfg).cuda()
    print(f"params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M, "
          f"batch={cfg.train.batch}, S={S}")

    byte_ids, pad_mask = batch["byte_ids"], batch["pad_mask"]

    def _peak() -> float:
        torch.cuda.synchronize()
        return torch.cuda.max_memory_allocated() / 2**20

    torch.cuda.reset_peak_memory_stats()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        t_enc = _time(lambda: model.encoder(byte_ids, pad_mask))
        lat = model.encoder(byte_ids, pad_mask)
        p_enc = _peak()
        t_bb = _time(lambda: model.backbone(lat))
        h = model.backbone(lat)[:, : S - 1]
        p_bb = _peak()

        q = mtp_targets(batch, cfg.loss.n_patches_ahead, 0.5, cfg.model.k_max)
        n_valid = int(q["valid"].sum())
        sel = q["valid"].reshape(-1).nonzero(as_tuple=True)[0]
        B = byte_ids.shape[0]
        b_all = torch.arange(B, device="cuda").view(B, 1, 1).expand(B, S - 1, cfg.model.k_max).reshape(-1)
        j_all = q["j_idx"].view(1, S - 1, 1).expand(B, S - 1, cfg.model.k_max).reshape(-1)
        d_all = q["delta"].view(1, 1, cfg.model.k_max).expand(B, S - 1, cfg.model.k_max).reshape(-1)

        def head_fwd():
            h_q = h[b_all[sel], j_all[sel]]
            return model.head(h_q, d_all[sel])

        t_head = _time(head_fwd)
        p_head = _peak()

        def full_step():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                from bltz.objectives import mtp_loss
                loss = mtp_loss(model, batch, cfg)
            model.zero_grad(set_to_none=True)
            loss.backward()

        t_step = _time(full_step, reps=3)
        p_step = _peak()

    n_patches_step = cfg.train.batch * S
    n_slots = cfg.train.batch * S * cfg.segment.l_max
    print(f"\nvalid queries/step: {n_valid}")
    print(f"encoder fwd : {t_enc:8.0f} ms   ({n_patches_step} patches x 16 bytes)")
    print(f"backbone fwd: {t_bb:8.0f} ms   ({n_patches_step} patch tokens)")
    print(f"head fwd    : {t_head:8.0f} ms   ({n_valid} queries)")
    print(f"full fwd+bwd: {t_step:8.0f} ms")
    print(f"peaks MiB   : enc {p_enc:.0f} / bb {p_bb:.0f} / head {p_head:.0f} / step {p_step:.0f}")
    print(f"byte slots  : real {int((~pad_mask).sum())} / canvas {n_slots} "
          f"({int((~pad_mask).sum())/n_slots*100:.1f}% effective)")
    print(f"\n=> measured step time {t_step/1000:.1f}s; 20k steps = {t_step*20000/3600000:.0f} h")
    print("bench_parts.py: DONE")


if __name__ == "__main__":
    main()

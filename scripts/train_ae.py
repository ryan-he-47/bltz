"""v2 S0: train a standalone byte-string autoencoder (docs/32 S0).

Pool: re-segment the local dryrun cache with the enhanced-BPE segmenter
(baked once per sequence, p_disagree). Train/val split over DISTINCT strings
(95/5). Loss: mean CE over valid Δ queries (bytes + EOS). Reports loss,
byte-acc, exact-match (train & val) every log_every; ckpt best/last.

  python scripts/train_ae.py configs/ae.yaml [--set a.b=value]
"""
from __future__ import annotations

import json
import os
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn.functional as F

from bltz.config import parse_cli
from bltz.models.autoencoder import EOS_ID, ByteStringAE
from bltz.segment_v2 import segment_v2
from bltz.shards import ShardReader
from bltz.trainer import WSD


def build_pool(cfg) -> tuple[list[bytes], list[bytes]]:
    reader = ShardReader(cfg.data.cache_dir)
    n_seq = reader.n_sequences(512)
    n_use = min(int(cfg.data.pool_sequences), n_seq)
    seen: dict[bytes, None] = {}
    for gi in range(n_use):
        raw = b"".join(reader.sequence_units(gi, 512))
        rng = random.Random(1000003 + gi)  # baked per sequence
        for u in segment_v2(raw, rng, cfg.seg.tok_dir, float(cfg.seg.p_disagree), int(cfg.seg.l_max)):
            if len(u) >= 1:
                seen[u] = None
    pool = list(seen.keys())
    rng = random.Random(int(cfg.train.seed))
    rng.shuffle(pool)
    n_val = max(1, int(len(pool) * float(cfg.data.val_frac)))
    print(f"[ae] pool: {len(pool)} distinct strings from {n_use} seqs; val {n_val}", flush=True)
    return pool[n_val:], pool[:n_val]


def tensorize(strings: list[bytes], l_max: int, dev: str):
    B = len(strings)
    byte_ids = torch.zeros(B, l_max, dtype=torch.long)
    lens = torch.zeros(B, dtype=torch.long)
    for i, s in enumerate(strings):
        b = list(s[:l_max])
        byte_ids[i, : len(b)] = torch.tensor(b, dtype=torch.long)
        lens[i] = len(b)
    pad_mask = torch.arange(l_max).expand(B, l_max) >= lens[:, None]
    return byte_ids.to(dev), lens.to(dev), pad_mask.to(dev)


def batch_loss(model, byte_ids, lens, pad_mask):
    """Mean CE over valid Δ queries; targets: bytes then EOS at Δ=len+1."""
    B, L = byte_ids.shape
    K = model.l_max + 1
    logits = model(byte_ids, pad_mask)  # (B, K, 257)
    d = torch.arange(1, K + 1, device=byte_ids.device)  # Δ = 1..l_max+1
    # target byte at Δ: position Δ-1 (0-based) if Δ<=len else EOS if Δ==len+1
    pos = (d - 1).view(1, K).expand(B, K)
    tgt = torch.where(
        d.view(1, K) <= lens[:, None],
        byte_ids.gather(1, pos.clamp(max=L - 1)).long(),
        torch.where(
            d.view(1, K) == lens[:, None] + 1,
            torch.full_like(byte_ids[:, :1], EOS_ID).expand(B, K),
            torch.full_like(byte_ids[:, :1], -100).expand(B, K),
        ),
    )
    ce = F.cross_entropy(
        logits.float().reshape(-1, 257), tgt.reshape(-1), ignore_index=-100
    )
    with torch.no_grad():
        pred = logits.argmax(-1)
        valid = d.view(1, K) <= lens[:, None] + 1
        byte_mask = d.view(1, K) <= lens[:, None]
        bacc = ((pred == tgt) & byte_mask).sum().item() / max(byte_mask.sum().item(), 1)
        em = (((pred == tgt) | ~valid).all(dim=1)).float().mean().item()
    return ce, bacc, em


def evaluate(model, strings, cfg, dev, batches: int = 4) -> tuple[float, float, float]:
    model.eval()
    rng = random.Random(777)
    tot = [0.0, 0.0, 0.0]
    n = 0
    with torch.no_grad():
        for _ in range(batches):
            ss = rng.sample(strings, min(int(cfg.train.batch), len(strings)))
            byte_ids, lens, pad_mask = tensorize(ss, model.l_max, dev)
            ce, bacc, em = batch_loss(model, byte_ids, lens, pad_mask)
            tot[0] += float(ce); tot[1] += bacc; tot[2] += em
            n += 1
    model.train()
    return tot[0] / n, tot[1] / n, tot[2] / n


def main() -> None:
    cfg = parse_cli()
    dev = "cuda"
    torch.manual_seed(int(cfg.train.seed))
    stream = bool(cfg.data.get("stream", False))
    if stream:
        # full-cache mode: batch = units of one random sequence (no in-memory
        # pool — the v2 cache has ~1e9 units). Val = 4 fixed random sequences.
        reader = ShardReader(cfg.data.cache_dir)
        n_seq = reader.n_sequences(512)
        val_rng = random.Random(777)
        val_pool = [u for gi in val_rng.sample(range(n_seq), 4)
                    for u in reader.sequence_units(gi, 512)]
        train_pool = None
        print(f"[ae] stream mode: {n_seq} seqs, val {len(val_pool)} strings", flush=True)
    else:
        train_pool, val_pool = build_pool(cfg)
    model = ByteStringAE(cfg).to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[ae] params {n_params / 1e6:.2f}M", flush=True)

    opt = torch.optim.AdamW(
        model.parameters(), lr=float(cfg.train.lr),
        betas=tuple(cfg.train.betas), weight_decay=float(cfg.train.wd),
    )
    sched = WSD(float(cfg.train.lr), int(cfg.train.warmup), int(cfg.train.steps),
                float(cfg.train.decay_frac), float(cfg.train.final_frac))
    os.makedirs(cfg.train.ckpt_dir, exist_ok=True)
    log_path = os.path.join(cfg.train.ckpt_dir, "train.log")
    rng = random.Random(int(cfg.train.seed) + 1)
    t0 = time.time()
    best = float("inf")
    model.train()
    for step in range(int(cfg.train.steps)):
        if stream:
            units = reader.sequence_units(rng.randrange(n_seq), 512)
            ss = rng.sample(units, min(int(cfg.train.batch), len(units)))
        else:
            ss = rng.sample(train_pool, int(cfg.train.batch))
        byte_ids, lens, pad_mask = tensorize(ss, model.l_max, dev)
        for g in opt.param_groups:
            g["lr"] = sched(step)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bool(cfg.train.bf16)):
            ce, bacc, em = batch_loss(model, byte_ids, lens, pad_mask)
        opt.zero_grad(set_to_none=True)
        ce.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.train.clip))
        opt.step()
        if step % int(cfg.train.log_every) == 0 or step == int(cfg.train.steps) - 1:
            rec = {"step": step, "loss": round(float(ce), 4),
                   "bacc": round(bacc, 4), "em": round(em, 4),
                   "gn": round(float(gn), 2), "lr": f"{sched(step):.2e}",
                   "sec": round(time.time() - t0, 1)}
            if step % (int(cfg.train.log_every) * 10) == 0:
                vce, vbacc, vem = evaluate(model, val_pool, cfg, dev)
                rec.update({"val_loss": round(vce, 4), "val_bacc": round(vbacc, 4),
                            "val_em": round(vem, 4)})
                if vce < best:
                    best = vce
                    torch.save({"model": model.state_dict(), "cfg": cfg.to_dict(),
                                "step": step, "val_loss": vce},
                               os.path.join(cfg.train.ckpt_dir, "best.pt"))
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            print(rec, flush=True)
    torch.save({"model": model.state_dict(), "cfg": cfg.to_dict(),
                "step": int(cfg.train.steps)},
               os.path.join(cfg.train.ckpt_dir, "last.pt"))
    vce, vbacc, vem = evaluate(model, val_pool, cfg, dev, batches=16)
    print(f"[done] val: loss {vce:.4f} bacc {vbacc:.4f} em {vem:.4f}", flush=True)


if __name__ == "__main__":
    main()

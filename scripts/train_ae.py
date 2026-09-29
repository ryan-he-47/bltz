"""v2 S0: train a standalone byte-string autoencoder (docs/32 S0).

Pool: re-segment the local dryrun cache with the enhanced-BPE segmenter
(baked once per sequence, p_disagree). Train/val split over DISTINCT strings
(95/5). Loss: mean CE over valid Δ queries (bytes + EOS). Reports loss,
byte-acc, exact-match (train & val) every log_every; ckpt best/last.

  python scripts/train_ae.py configs/ae.yaml [--set a.b=value]
"""
from __future__ import annotations

import json
import math
import os
import random
import sys
import time
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn.functional as F

from bltz.config import parse_cli
from bltz.models.autoencoder import EOS_ID, ByteStringAE
from bltz.segment_v2 import segment_v2
from bltz.shards import ShardReader
from bltz.trainer import WSD, CosSched, grad_norm


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
    # typo-cluster augmentation (docs/32; p=0 -> off). Online, at batch time;
    # val stays clean. Freq throttle: top-N strings from a quick pre-pass get
    # p x freq_scale (common short strings should rarely be corrupted).
    tcfg = cfg.get("typo", None)
    typo_p = float(getattr(tcfg, "p", 0.0)) if tcfg else 0.0
    freq_set = None
    if typo_p > 0 and bool(getattr(tcfg, "freq_throttle", True)):
        from collections import Counter

        reader0 = ShardReader(cfg.data.cache_dir)
        n0 = reader0.n_sequences(512)
        cnt: Counter[bytes] = Counter()
        for gi in random.Random(31337).sample(range(n0), min(int(getattr(tcfg, "freq_seqs", 256)), n0)):
            cnt.update(reader0.sequence_units(gi, 512))
        topn = int(getattr(tcfg, "freq_topk", 30000))
        freq_set = frozenset(u for u, _ in cnt.most_common(topn))
        print(f"[typo] freq throttle: top {len(freq_set)} strings get "
              f"p*{float(getattr(tcfg, 'freq_scale', 0.25))}", flush=True)
    typo_rng = random.Random(91517)
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
    # sched: wsd (default, house) | cosine (2026-09-29 5B 配方: warmup ->
    # cosine peak..train.final)
    if str(cfg.train.get("sched", "wsd")) == "cosine":
        sched = CosSched(float(cfg.train.lr), int(cfg.train.warmup),
                         int(cfg.train.steps), float(cfg.train.get("final", 1e-6)))
    else:
        sched = WSD(float(cfg.train.lr), int(cfg.train.warmup), int(cfg.train.steps),
                    float(cfg.train.decay_frac), float(cfg.train.final_frac))
    # fp16 + GradScaler + gn spike guard — aligned with the LM main-run
    # trainer (2026-09-29: 数值稳定手法对齐). bf16 path keeps plain backward.
    use_bf16 = bool(cfg.train.bf16)
    scaler = torch.amp.GradScaler("cuda", enabled=not use_bf16)
    med_hist: deque[float] = deque(maxlen=1000)
    skips = 0
    os.makedirs(cfg.train.ckpt_dir, exist_ok=True)
    log_path = os.path.join(cfg.train.ckpt_dir, "train.log")
    rng = random.Random(int(cfg.train.seed) + 1)
    t0 = time.time()
    best = float("inf")
    model.train()
    # auto-resume + periodic last.pt (2026-09-29: 3-day 5B runs need recovery)
    last_path = os.path.join(cfg.train.ckpt_dir, "last.pt")
    start_step = 0
    if os.path.exists(last_path):
        st_ = torch.load(last_path, map_location="cpu", weights_only=False)
        model.load_state_dict(st_["model"])
        if "opt" in st_:
            opt.load_state_dict(st_["opt"])
            scaler.load_state_dict(st_["scaler"])
        start_step = int(st_["step"]) + 1
        print(f"[ae] resume from last.pt at step {start_step}", flush=True)
    ckpt_every = int(cfg.train.get("ckpt_every", 10000))
    for step in range(start_step, int(cfg.train.steps)):
        if stream:
            units = reader.sequence_units(rng.randrange(n_seq), 512)
            ss = rng.sample(units, min(int(cfg.train.batch), len(units)))
        else:
            ss = rng.sample(train_pool, int(cfg.train.batch))
        if typo_p > 0:
            from bltz.typo import corrupt, p_eff

            ss = [
                (c if (c := corrupt(s, typo_rng,
                                    tuple(getattr(tcfg, "ops", (0.4, 0.2, 0.15, 0.15, 0.1))),
                                    float(getattr(tcfg, "two_edit_p", 0.15)),
                                    int(cfg.seg.l_max))) is not None else s)
                if typo_rng.random() < p_eff(s, typo_p, freq_set,
                                             float(getattr(tcfg, "freq_scale", 0.25)))
                else s
                for s in ss
            ]
        byte_ids, lens, pad_mask = tensorize(ss, model.l_max, dev)
        for g in opt.param_groups:
            g["lr"] = sched(step)
        with torch.autocast("cuda", dtype=torch.bfloat16 if use_bf16 else torch.float16):
            ce, bacc, em = batch_loss(model, byte_ids, lens, pad_mask)
        opt.zero_grad(set_to_none=True)
        if use_bf16:
            ce.backward()
        else:
            scaler.scale(ce).backward()
            scaler.unscale_(opt)
        gn = grad_norm(model)
        median = sorted(med_hist)[len(med_hist) // 2] if len(med_hist) >= 50 else None
        thresh = max(float(cfg.train.get("spike_skip", 10.0)) * median, 2000.0) \
            if median is not None else float("inf")
        if not math.isfinite(gn) or gn > thresh:
            skips += 1
            if not use_bf16:
                scaler.update()
            continue
        med_hist.append(gn)
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.train.clip))
        if use_bf16:
            opt.step()
        else:
            scaler.step(opt)
            scaler.update()
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
        if ckpt_every and (step + 1) % ckpt_every == 0:
            torch.save({"model": model.state_dict(), "cfg": cfg.to_dict(),
                        "opt": opt.state_dict(), "scaler": scaler.state_dict(),
                        "step": step}, last_path)
    torch.save({"model": model.state_dict(), "cfg": cfg.to_dict(),
                "opt": opt.state_dict(), "scaler": scaler.state_dict(),
                "step": int(cfg.train.steps) - 1},
               os.path.join(cfg.train.ckpt_dir, "last.pt"))
    vce, vbacc, vem = evaluate(model, val_pool, cfg, dev, batches=16)
    print(f"[done] val: loss {vce:.4f} bacc {vbacc:.4f} em {vem:.4f}", flush=True)


if __name__ == "__main__":
    main()

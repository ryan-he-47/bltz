"""Smoke 2: tiny real-data run (stream a few FineWeb-Edu docs) + inference-loop
smoke with a theta_stop sweep. Mechanics only — no quality bar.
Run: python scripts/smoke_tiny_train.py [--set a.b=value]"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from bytefield.config import parse_cli
from bytefield.data import build_batch, stream_fineweb_texts
from bytefield.infer import generate
from bytefield.models import ByteFieldLM
from bytefield.segment import Segmenter
from bytefield.trainer import train

N_DOCS = 1500


def main() -> None:
    cfg = parse_cli("configs/smoke.yaml")
    torch.manual_seed(cfg.train.seed)
    seg = Segmenter(cfg.segment.l_max, cfg.segment.p_split, cfg.segment.p_merge)

    print(f"streaming {N_DOCS} docs from FineWeb-Edu sample-10BT ...", flush=True)
    t0 = time.time()
    texts = list(stream_fineweb_texts(N_DOCS))
    print(f"  got {len(texts)} docs, {sum(len(t) for t in texts)/1e6:.2f} MB text in {time.time()-t0:.0f}s")

    batch = build_batch(texts, seg, cfg.data.n_patches, cfg.segment.l_max)
    n_seq = batch["byte_ids"].shape[0]
    print(f"sequences: {n_seq} (S={cfg.data.n_patches} patches each)")

    model = ByteFieldLM(cfg).cuda()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model params: {n_params/1e6:.2f}M")

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    history = train(model, lambda: _sample_seq(batch), cfg)
    dt = time.time() - t0
    peak = torch.cuda.max_memory_allocated() / 2**20
    losses = [r["loss"] for r in history if "loss" in r]
    print(f"train: {dt:.0f}s total, {dt/cfg.train.steps*1000:.0f} ms/step, peak VRAM {peak:.0f} MiB")
    print(f"loss: {losses[0]:.4f} -> {losses[-1]:.4f}")

    # ---- inference-loop smoke: theta_stop sweep ----
    prompt = b"The Byte Latent Transformer encodes bytes into"
    print(f"\nprompt: {prompt.decode()!r}")
    for theta in [0.5, 1.0, 2.0, float("inf")]:
        r = generate(
            model, prompt, seg, cfg,
            max_commits=12, theta_stop=theta, device="cuda",
        )
        tail = r["text"][len(prompt):].decode("utf-8", errors="replace")
        print(f"\ntheta_stop={theta}: spans={r['spans']}")
        print(f"  +{tail!r}")
        print(f"  H(delta) per step: {r['entropies'][:3]}")

    print("\nsmoke_tiny_train.py: DONE")


def _sample_seq(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    n = batch["byte_ids"].shape[0]
    idx = torch.randint(0, n, (8,))
    return {k: v[idx] for k, v in batch.items()}


if __name__ == "__main__":
    main()

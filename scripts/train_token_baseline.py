"""Train the Stage-2 twin control: GPT-2 token baseline with the SAME Backbone
recipe and the SAME trainer loop (loss_fn injection), at matched bytes/step
(configs/baseline.yaml header has the matching math; docs/02 §3, docs/06).
Run: python -u scripts/train_token_baseline.py [configs/baseline.yaml] [--set a.b=value]"""
from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from bltz.config import parse_cli
from bltz.models.token_lm import TokenLlama, next_token_loss
from bltz.token_cache import TokenShardReader
from bltz.trainer import train


def main() -> None:
    cfg = parse_cli("configs/baseline.yaml")
    reader = TokenShardReader(cfg.data.cache_dir)
    T = cfg.data.seq_len
    n_seq = reader.n_sequences(T)
    print(f"token cache: {n_seq} sequences at T={T}")

    torch.manual_seed(cfg.train.seed)
    m = cfg.model
    model = TokenLlama(
        d_model=m.d_model, nhead=m.bb_heads, layers=m.bb_layers,
        ffn_mult=m.ffn_mult, max_len=T + 8, vocab=m.vocab,
    ).cuda()
    print(f"token baseline params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

    rng = random.Random(cfg.train.seed)

    def batch_fn() -> dict[str, torch.Tensor]:
        idx = [rng.randrange(n_seq) for _ in range(cfg.train.batch)]
        return reader.make_batch(idx, T)

    train(model, batch_fn, cfg, loss_fn=lambda m_, b, c: next_token_loss(m_, b["idx"]))


if __name__ == "__main__":
    main()

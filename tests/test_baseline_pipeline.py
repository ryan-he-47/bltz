"""Script-style test for the token baseline pipeline: token cache builder ->
reader -> TokenLlama loss sanity. Run: python tests/test_baseline_pipeline.py"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pyarrow as pa
import pyarrow.parquet as pq
import torch

from bltz.models.token_lm import TokenLlama, next_token_loss
from bltz.token_cache import EOS_ID, TokenShardReader
from scripts.build_token_cache import _process_file

TEXTS = [
    "The Byte Latent Transformer encodes bytes into dynamically sized patches. " * 8,
    "Entropy patching allocates compute where data complexity demands it. " * 8,
]
TMP = Path("data/test_baseline_tmp")


def check(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def main() -> None:
    shutil.rmtree(TMP, ignore_errors=True)
    pdir = TMP / "parquet"
    pdir.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table({"text": TEXTS * 100}), str(pdir / "000_00000.parquet"))

    meta = _process_file((0, str(pdir / "000_00000.parquet"), str(TMP / "cache" / "shard-00000"), 0))
    check(meta["n_docs"] == len(TEXTS) * 100, f"docs: {meta['n_docs']}")
    print(f"shard: {meta['n_tokens']} tokens, {meta['n_docs']} docs")

    reader = TokenShardReader(str(TMP / "cache"))
    T = 64
    n_seq = reader.n_sequences(T)
    check(n_seq > 0, "no sequences")
    batch = reader.make_batch([0, 1, 2], T)
    check(batch["idx"].shape == (3, T), f"batch shape: {batch['idx'].shape}")
    check(bool((batch["idx"] >= 0).all() and (batch["idx"] < 50257).all()), "token id range")
    check(bool((batch["idx"] == EOS_ID).any()), "EOS separator missing")

    from tokenizers import Tokenizer

    tok = Tokenizer.from_pretrained("gpt2")
    expected = tok.encode(TEXTS[0], add_special_tokens=False).ids
    got = batch["idx"][0, :T].tolist()  # cache slice is T tokens long
    check(got == expected[: len(got)], f"first tokens mismatch:\n{got[:12]}\n{expected[:12]}")
    print("cache content == tokenizer output: OK")

    torch.manual_seed(0)
    model = TokenLlama(d_model=128, nhead=4, layers=2, ffn_mult=4, max_len=T + 8).cuda()
    loss = next_token_loss(model, batch["idx"].cuda())
    ln_v = torch.log(torch.tensor(50257.0)).item()
    check(abs(loss.item() - ln_v) < 0.3, f"init loss {loss.item():.3f} != ln V {ln_v:.3f}")
    print(f"init loss: {loss.item():.4f} (ln 50257 = {ln_v:.4f})")

    shutil.rmtree(TMP, ignore_errors=True)
    print("test_baseline_pipeline.py: ALL PASS")


if __name__ == "__main__":
    main()

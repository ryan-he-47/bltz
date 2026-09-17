"""AE smoke test: shapes, EOS targets, decode round-trip, tiny overfit.
Run: python tests/test_ae.py
"""
from __future__ import annotations

import torch

from bltz.config import load
from bltz.models.autoencoder import EOS_ID, ByteStringAE

cfg = load("configs/ae.yaml")
cfg.model.d_emb = 16
cfg.model.enc_width = 64
cfg.model.dec_hidden = 64
cfg.model.enc_layers = 1
model = ByteStringAE(cfg)
dev = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(dev)

strings = [b"hello", b"world", b"a", b"the quick brown fox ", b"x" * 32]
B = len(strings)
L = cfg.model.l_max
byte_ids = torch.zeros(B, L, dtype=torch.long)
lens = torch.zeros(B, dtype=torch.long)
for i, s in enumerate(strings):
    b = list(s[:L])
    byte_ids[i, : len(b)] = torch.tensor(b)
    lens[i] = len(b)
pad_mask = torch.arange(L).expand(B, L) >= lens[:, None]
byte_ids, lens, pad_mask = byte_ids.to(dev), lens.to(dev), pad_mask.to(dev)

logits = model(byte_ids, pad_mask)
assert logits.shape == (B, L + 1, 257), logits.shape

# loss + targets sanity
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from train_ae import batch_loss

ce, bacc, em = batch_loss(model, byte_ids, lens, pad_mask)
assert ce.isfinite(), ce
print(f"init: ce {float(ce.detach()):.3f} bacc {bacc:.3f} em {em:.3f}")

# decode shape
lam = model.encoder(byte_ids, pad_mask)
out = model.decode(lam)
assert len(out) == B and all(len(o) <= L for o in out)

# tiny overfit: 400 steps on these 5 strings -> should nail them
opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
model.train()
for step in range(400):
    ce, bacc, em = batch_loss(model, byte_ids, lens, pad_mask)
    opt.zero_grad()
    ce.backward()
    opt.step()
print(f"overfit400: ce {float(ce.detach()):.4f} bacc {bacc:.4f} em {em:.4f}")
assert float(ce.detach()) < 0.05 and em == 1.0, "overfit failed"
dec = model.decode(model.encoder(byte_ids, pad_mask))
for s, o in zip(strings, dec):
    assert bytes(o) == s[:L].rstrip(b"\x00") or bytes(o) == bytes(list(s[:L])), (s, o)
print("test_ae: OK")

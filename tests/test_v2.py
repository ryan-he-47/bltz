"""v2 smoke: BltzLMv2 shapes, MDN NLL finite, tiny overfit, sampling chain.
Run: python tests/test_v2.py
"""
from __future__ import annotations

import torch

from bltz.config import Cfg, load
from bltz.models.autoencoder import ByteStringAE
from bltz.models.model_v2 import BltzLMv2
from bltz.objectives_v2 import mdn_nll_loss

dev = "cuda" if torch.cuda.is_available() else "cpu"

# tiny AE + tiny v2 model
ae_cfg = load("configs/ae.yaml")
ae_cfg.model.d_emb = 16
ae_cfg.model.enc_width = 64
ae_cfg.model.dec_hidden = 64
ae_cfg.model.enc_layers = 1
ae = ByteStringAE(ae_cfg)

v2_cfg = Cfg({
    "model": {"d_emb": 16, "adapter_width": 128, "d_model": 96, "bb_layers": 2,
              "bb_heads": 4, "ffn_mult": 4, "grad_ckpt": False,
              "mdn_hidden": 128, "n_comp": 8, "sigma_floor": 1e-3},
    "data": {"n_patches": 16},
    "train": {},
})
model = BltzLMv2(v2_cfg, ae).to(dev)

B, S, L = 3, 16, 12
byte_ids = torch.randint(32, 126, (B, S, L), device=dev)
lens = torch.randint(2, L + 1, (B, S), device=dev)
pad_mask = torch.arange(L, device=dev).expand(B, S, L) >= lens[..., None]

h, lam = model(byte_ids, pad_mask)
assert h.shape == (B, S, 96) and lam.shape == (B, S, 16)
assert not lam.requires_grad, "targets must be detached"

loss = mdn_nll_loss(model, {"byte_ids": byte_ids, "pad_mask": pad_mask}, v2_cfg)
assert loss.isfinite(), loss
# init should be near N(0,I) NLL: 0.5*d*(1+ln 2pi) for unit-variance init
print(f"init nll {float(loss):.3f} (d=16, unit-Gaussian ref ~{0.5 * 16 * (1 + 1.8379):.1f})")

# frozen AE really frozen
loss.backward()
assert all(p.grad is None for p in model.ae.parameters()), "AE got gradients!"

# overfit one fixed batch -> NLL must dive
opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=3e-3)
l0 = float(loss.detach())
for _ in range(300):
    opt.zero_grad()
    loss = mdn_nll_loss(model, {"byte_ids": byte_ids, "pad_mask": pad_mask}, v2_cfg)
    loss.backward()
    opt.step()
print(f"overfit300: nll {l0:.2f} -> {float(loss):.2f}")
assert float(loss) < l0 - 2.0, "no overfit"

# sampling chain: BOS -> sample -> re-encode -> repeat (mechanics only)
with torch.no_grad():
    cur = model.bos.expand(1, 1, -1)
    seq = []
    for _ in range(4):
        h_step = model.backbone(model.adapter(cur))[:, -1]
        nxt = model.head.sample(h_step, tau=1.0)
        seq.append(nxt)
        cur = torch.cat([cur, nxt.unsqueeze(1)], dim=1)
    out = model.ae.decode(torch.cat(seq, dim=0))
    print("sampled strings:", [bytes(o) for o in out])
print("test_v2: OK")

"""MDN head variants (2026-09-24 ablation基建): wide buffer / SwiGLU blocks /
backward-compat with pre-swiglu cfgs. Run: python tests/test_mdn.py
"""
from __future__ import annotations

import torch

from bltz.config import Cfg, load
from bltz.models.autoencoder import ByteStringAE
from bltz.models.mdn import MDNHead, _SwiGLU
from bltz.models.model_v2 import BltzLMv2

dev = "cuda" if torch.cuda.is_available() else "cpu"
K, D, H = 8, 16, 6208  # output width K*(1+2D) = 8*33 = 264 here; H tests width plumbing

for swiglu in (False, True):
    head = MDNHead(d_in=96, hidden=H, n_comp=K, d_emb=D, sigma_floor=0.05,
                   swiglu=swiglu).to(dev)
    h = torch.randn(2, 7, 96, device=dev)
    lp, mu, sig = head.params(h)
    assert lp.shape == (2, 7, K) and mu.shape == (2, 7, K, D) and sig.shape == (2, 7, K, D)
    assert (sig > 0).all(), "sigma must be positive"
    tgt = torch.randn(2, 7, D, device=dev)
    nll = head.nll(h, tgt)
    assert nll.shape == (2, 7) and nll.isfinite().all()
    nll.sum().backward()
    g = [p.grad is not None and p.grad.isfinite().all() for p in head.parameters()]
    assert all(g), "some head param got no/NaN grad"
    s = head.sample(h.detach(), tau=0.8)
    assert s.shape == (2, 7, D)
    print(f"swiglu={swiglu}: shapes/backward/sample OK, "
          f"params {sum(p.numel() for p in head.parameters()) / 1e6:.2f}M")

# depth ladder: 0 = linear, 1 = single buffer, 2 = current default
for depth in (0, 1):
    head = MDNHead(d_in=96, hidden=H, n_comp=K, d_emb=D, sigma_floor=0.05,
                   depth=depth).to(dev)
    assert not hasattr(head, "fc1") or depth >= 1
    h = torch.randn(2, 7, 96, device=dev)
    lp, mu, sig = head.params(h)
    assert lp.shape == (2, 7, K) and sig.shape == (2, 7, K, D)
    nll = head.nll(h, torch.randn(2, 7, D, device=dev))
    assert nll.isfinite().all()
    nll.sum().backward()
    g = [p.grad is not None and p.grad.isfinite().all() for p in head.parameters()]
    assert all(g), f"depth={depth}: some head param got no/NaN grad"
    print(f"depth={depth}: shapes/backward OK, "
          f"params {sum(p.numel() for p in head.parameters()) / 1e6:.2f}M")

# swiglu block structure
blk = _SwiGLU(96, H)
assert isinstance(blk.w1, torch.nn.Linear) and blk.w1.out_features == H

# BltzLMv2 wiring: default cfg (no mdn_swiglu key) -> plain MLP (old ckpt compat)
ae_cfg = load("configs/ae.yaml")
ae_cfg.model.d_emb = 16
ae_cfg.model.enc_width = 64
ae_cfg.model.dec_hidden = 64
ae_cfg.model.enc_layers = 1
ae = ByteStringAE(ae_cfg)
base = Cfg({"model": {"d_emb": 16, "adapter_width": 128, "d_model": 96, "bb_layers": 2,
                      "bb_heads": 4, "ffn_mult": 4, "grad_ckpt": False,
                      "mdn_hidden": 128, "n_comp": 8, "sigma_floor": 1e-3},
            "data": {"n_patches": 16}, "train": {}})
m1 = BltzLMv2(base, ae)
assert isinstance(m1.head.fc1, torch.nn.Linear), "default must stay plain MLP"
sw = Cfg({"model": {**base.model.to_dict(), "mdn_swiglu": True},
          "data": {"n_patches": 16}, "train": {}})
m2 = BltzLMv2(sw, ae)
assert isinstance(m2.head.fc1, _SwiGLU), "mdn_swiglu=true must build SwiGLU blocks"

# old-format state dict (fc1.weight/fc1.bias) still loads into a plain head
sd = m1.state_dict()
m3 = BltzLMv2(base, ae)
m3.load_state_dict(sd)
print("test_mdn: OK")

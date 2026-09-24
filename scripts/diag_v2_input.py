"""v2 input-side geometry diag (2026-09-24, user direction):

After the frozen AE (48d lambda) the learnable adapter (LN -> 48->2048->GELU
->768) forms the backbone's INPUT space. If that space is anisotropic /
collapsed (narrow cone, low effective rank, dead dims), the backbone spends
capacity undoing it -> learning-efficiency loss vs token embeddings.

Measures, on both lambda (48d) and adapter output (768d):
  * anisotropy: mean/std of pairwise cosine over random position pairs
  * effective rank (participation ratio of singular spectrum, centered)
  * per-dim variance: dead-dim count (var < 1e-4 of max)
  * norm distribution
Plus random-Gaussian reference anisotropy (~0 by construction) for scale.

  python scripts/diag_v2_input.py <v2_ckpt> <ae_ckpt> --units-pkl <pkl>
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch

from bltz.config import Cfg
from bltz.models.autoencoder import ByteStringAE
from bltz.models.model_v2 import BltzLMv2
from train_ae import tensorize

DEV = "cuda"
N_SEQS = 16
N_PAIRS = 20000


def stats(name: str, X: np.ndarray, rng: np.random.Generator) -> None:
    """X (N, d) geometry stats."""
    n, d = X.shape
    i = rng.integers(0, n, N_PAIRS)
    j = rng.integers(0, n, N_PAIRS)
    a = X[i] / np.linalg.norm(X[i], axis=1, keepdims=True).clip(1e-9)
    b = X[j] / np.linalg.norm(X[j], axis=1, keepdims=True).clip(1e-9)
    cos = (a * b).sum(1)
    Xc = X - X.mean(0, keepdims=True)
    sv = np.linalg.svd(Xc, compute_uv=False)
    pr = float((sv ** 2).sum() ** 2 / (sv ** 4).sum())
    dv = Xc.var(0)
    dead = int((dv < 1e-4 * dv.max()).sum())
    norms = np.linalg.norm(X, axis=1)
    print(f"[{name}] d={d} N={n}")
    print(f"  anisotropy: mean cos {cos.mean():+.4f} |cos| {np.abs(cos).mean():.4f} "
          f"std {cos.std():.4f}   (random-ref ~0, collapse -> >0.5)")
    print(f"  spectrum: eff-rank {pr:.1f}/{d} | top1 sv share {sv[0] ** 2 / (sv ** 2).sum():.4f} "
          f"| top10 {(sv[:10] ** 2).sum() / (sv ** 2).sum():.4f}")
    print(f"  dims: dead {dead}/{d} | var ratio p50/max {np.median(dv) / dv.max():.4f}")
    print(f"  norms: mean {norms.mean():.2f} std {norms.std():.2f} "
          f"min {norms.min():.2f} max {norms.max():.2f}", flush=True)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    v2_path, ae_path = sys.argv[1], sys.argv[2]
    pkl = sys.argv[sys.argv.index("--units-pkl") + 1]
    src = pickle.load(open(pkl, "rb"))
    ae_state = torch.load(ae_path, map_location="cpu", weights_only=False)
    ae = ByteStringAE(Cfg(ae_state["cfg"]))
    ae.load_state_dict(ae_state["model"])
    v2_state = torch.load(v2_path, map_location="cpu", weights_only=False)
    model = BltzLMv2(Cfg(v2_state["cfg"]), ae).to(DEV).eval()
    model.load_state_dict(v2_state["model"])

    lams, xs = [], []
    with torch.no_grad():
        for units in src["distinct_seq_units"][:N_SEQS]:
            bids, lens, pm = tensorize(units, model.l_max, DEV)
            lam = model.encode_units(bids.unsqueeze(0), pm.unsqueeze(0))
            x = model.adapter(lam)  # post-adapter backbone input (B, S, 768)
            lams.append(lam[0].float().cpu().numpy())
            xs.append(x[0].float().cpu().numpy())
    lam = np.concatenate(lams)
    x = np.concatenate(xs)
    rng = np.random.default_rng(0)
    print("== position-weighted (frequency confound incl.) ==", flush=True)
    stats("lambda 48d (frozen AE out)", lam, rng)
    stats("adapter out 768d (backbone input)", x, rng)

    # distinct-unit view: dedupe strings -> uniform over WORD TYPES, removes the
    # "the 'the' is everywhere" frequency confound; this is the space the GMM
    # actually has to represent.
    uniq = list(set(src["freq_units"]))
    rng_np = np.random.default_rng(1)
    sel = rng_np.choice(len(uniq), size=min(20000, len(uniq)), replace=False)
    uniq = [uniq[i] for i in sel]
    ul, ux = [], []
    with torch.no_grad():
        for i in range(0, len(uniq), 2048):
            b, l, mk = tensorize(uniq[i : i + 2048], model.l_max, DEV)
            lam_u = model.encode_units(b.unsqueeze(0), mk.unsqueeze(0))[0]
            ux.append(model.adapter(lam_u.unsqueeze(0))[0].float().cpu().numpy())
            ul.append(lam_u.float().cpu().numpy())
    print("== distinct-unit (uniform over word types) ==", flush=True)
    stats("lambda 48d distinct", np.concatenate(ul), rng)
    stats("adapter out 768d distinct", np.concatenate(ux), rng)
    ref = rng.standard_normal((x.shape[0], x.shape[1])).astype(np.float32)
    stats("random gaussian ref 768d", ref, rng)
    print("[input] DONE", flush=True)


if __name__ == "__main__":
    main()

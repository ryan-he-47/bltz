"""Typo-cluster AE arm comparison — DESIGN-INTENT metrics (2026-09-21, after
the anti-noise-EM lesson: typos are not noise to be robust against, they are
data points to be FAITHFUL to).

Per arm:
  A. typo 等效噪声 σ_typo: per-dim std of (λ_variant − λ_source) — what "one
     typo" means in embedding units (calibrates tier D's noise levels).
  B. typo 保留率 (fidelity): encode(typo) -> decode -> EM to the TYPO target;
     plus the attraction failure (decoded == clean source = cloud collapsed).
     Per-op breakdown via force_op.
  C. 点云/真词分离: mean ‖λ_variant − λ_source‖ (cloud radius) and the
     nearest-neighbor distance distribution among REAL words (typo clouds
     should widen real-word separation at the low percentiles).
  D. failure 行为模式 at σ_typo-calibrated tiers (0.5x/1x/2x): decode
     (λ_probe + N(0, tier·σ_typo)) and classify: exact / typo-like (edit<=2)
     / far-real (in freq vocab, edit>2) / garbage (else). Typo arms should
     shift mass garbage -> typo-like.

  python scripts/diag_ae_typo.py ckptA ckptB [...] --cache data/fineweb_v2_local
"""
from __future__ import annotations

import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
import torch.nn.functional as F

from bltz.config import Cfg
from bltz.models.autoencoder import ByteStringAE
from bltz.shards import ShardReader
from bltz.typo import corrupt
from diag_v2 import lev
from train_ae import tensorize

DEV = "cuda"
OPS = ("sub", "del", "dup", "trans", "case")


def encode_all(model, strings):
    lats = []
    with torch.no_grad():
        for i in range(0, len(strings), 512):
            b, l, m = tensorize(strings[i : i + 512], model.l_max, DEV)
            lats.append(model.encoder(b, m).float().cpu())
    return torch.cat(lats)


def load_model(path):
    state = torch.load(path, map_location="cpu", weights_only=False)
    model = ByteStringAE(Cfg(state["cfg"]))
    model.load_state_dict(state["model"])
    return model.to(DEV).eval(), state


def decode_strs(model, lam):
    with torch.no_grad():
        return [bytes(o) for o in model.decode(lam.to(DEV))]


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ckpts = []
    skip_next = False
    for a in sys.argv[1:]:
        if skip_next:
            skip_next = False
            continue
        if a.startswith("--"):
            skip_next = True
            continue
        ckpts.append(a)
    cache = sys.argv[sys.argv.index("--cache") + 1] if "--cache" in sys.argv else "data/fineweb_v2_local"
    reader = ShardReader(cache)
    n_seq = reader.n_sequences(512)
    val_units = [u for gi in random.Random(777).sample(range(n_seq), 4)
                 for u in reader.sequence_units(gi, 512)]
    probe_pool = [u for gi in random.Random(4242).sample(range(n_seq), 24)
                  for u in reader.sequence_units(gi, 512)]
    cnt: Counter[bytes] = Counter()
    for gi in random.Random(31337).sample(range(n_seq), min(256, n_seq)):
        cnt.update(reader.sequence_units(gi, 512))
    vocab = frozenset(u for u, _ in cnt.most_common(30000))  # "real word" proxy
    rng_w = random.Random(555)
    src_words = rng_w.sample([u for u in val_units if 5 <= len(u) <= 20], 300)
    probe_src = random.Random(4243).sample(probe_pool, 1500)
    print(f"[typo-diag2] {len(ckpts)} arms; src_words {len(src_words)}, "
          f"probe {len(probe_src)}, vocab {len(vocab)}", flush=True)

    for path in ckpts:
        model, state = load_model(path)
        name = Path(path).parent.name
        # per-op variants + free variants over the same source words
        per_op: dict[str, tuple[list[bytes], list[bytes]]] = {op: ([], []) for op in OPS}
        free_v, free_s = [], []
        rng_v = random.Random(7777)
        for w in src_words:
            for op in OPS:
                v = corrupt(w, rng_v, force_op=op)
                if v is not None:
                    per_op[op][0].append(v); per_op[op][1].append(w)
            v = corrupt(w, rng_v)
            if v is not None:
                free_v.append(v); free_s.append(w)
        lam_s = encode_all(model, free_s)
        lam_v = encode_all(model, free_v)
        delta = lam_v - lam_s
        sigma_typo = float(delta.std())           # per-dim typo-equivalent sigma
        radius = float(delta.norm(dim=-1).mean())
        # A/B: fidelity + attraction
        dec_v = decode_strs(model, lam_v)
        fid = np.mean([a == b for a, b in zip(dec_v, free_v)])
        snap = np.mean([a == s for a, s in zip(dec_v, free_s)])
        op_lines = []
        for op in OPS:
            vs, ss = per_op[op]
            if not vs:
                continue
            dl = decode_strs(model, encode_all(model, vs))
            f = np.mean([a == b for a, b in zip(dl, vs)])
            sn = np.mean([a == s for a, s in zip(dl, ss)])
            op_lines.append(f"{op}:fid {f:.3f}/snap {sn:.3f}(n={len(vs)})")
        # C: real-word NN separation among val real strings
        real = [u for u in val_units if u in vocab][:1000]
        lam_r = F.normalize(encode_all(model, real), dim=-1)
        sim = lam_r @ lam_r.T
        sim.fill_diagonal_(-2)
        nn_dist = (1 - sim.max(-1).values).numpy()
        # D: calibrated noise failure modes
        lam_p = encode_all(model, probe_src)
        tiers = {}
        for mult in (0.5, 1.0, 2.0):
            s = sigma_typo * mult
            torch.manual_seed(0)
            with torch.no_grad():
                dec = decode_strs(model, lam_p + torch.randn_like(lam_p) * s)
            cls = {"exact": 0, "typo": 0, "far_real": 0, "garbage": 0}
            for a, b in zip(dec, probe_src):
                e = lev(a, b)
                if e == 0:
                    cls["exact"] += 1
                elif e <= 2:
                    cls["typo"] += 1
                elif a in vocab:
                    cls["far_real"] += 1
                else:
                    cls["garbage"] += 1
            tiers[mult] = {k: v / len(probe_src) for k, v in cls.items()}
        print(f"\n### {name} (step {state['step']})", flush=True)
        print(f"  σ_typo(per-dim) {sigma_typo:.4f} | cloud radius ‖Δλ‖ {radius:.3f}", flush=True)
        print(f"  typo 保留率(fid) {fid:.4f} | 吸引失败(snap 回源词) {snap:.4f}", flush=True)
        print(f"  per-op: {' | '.join(op_lines)}", flush=True)
        print(f"  real-word NN dist: p1 {np.percentile(nn_dist, 1):.4f} "
              f"p5 {np.percentile(nn_dist, 5):.4f} med {np.median(nn_dist):.4f}", flush=True)
        for mult, row in tiers.items():
            print(f"  noise {mult:.1f}σ_t: exact {row['exact']:.3f} typo {row['typo']:.3f} "
                  f"far_real {row['far_real']:.3f} garbage {row['garbage']:.3f}", flush=True)
    print("\n[typo-diag2] DONE", flush=True)


if __name__ == "__main__":
    main()

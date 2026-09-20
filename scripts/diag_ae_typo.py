"""Typo-cluster AE arm comparison (docs/32 工具箱 #2 验收对照).

Per arm (each an AE ckpt trained with a different typo.p on the SAME local
cache), report side by side:
  1. clean val EM/loss (floor: within 2pp of the plain arm);
  2. noise-robustness curve: decode(lambda + N(0, s)) EM for s in
     [0.02..0.30], plus edit-distance stats of the WRONG decodes (typo arm
     should cluster its errors at 1-2 edits instead of scattering);
  3. typo-variant neighborhood: variants of real strings should sit next to
     the source in lambda space (cos rank percentile vs background);
  4. geometry quick: PR / anisotropy / kNN-CC.

  python scripts/diag_ae_typo.py ckptA ckptB [...] --cache data/fineweb_v2_local
"""
from __future__ import annotations

import random
import sys
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
from train_ae import batch_loss, tensorize

DEV = "cuda"
NOISE_GRID = (0.02, 0.05, 0.10, 0.15, 0.20, 0.30)


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


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ckpts = [a for a in sys.argv[1:] if not a.startswith("--")]
    cache = sys.argv[sys.argv.index("--cache") + 1] if "--cache" in sys.argv else "data/fineweb_v2_local"
    reader = ShardReader(cache)
    n_seq = reader.n_sequences(512)
    val_units = [u for gi in random.Random(777).sample(range(n_seq), 4)
                 for u in reader.sequence_units(gi, 512)]
    probe_pool = [u for gi in random.Random(4242).sample(range(n_seq), 24)
                  for u in reader.sequence_units(gi, 512)]
    probe = random.Random(4243).sample(probe_pool, 2000)
    rng_t = random.Random(555)
    words = [u for u in val_units if 5 <= len(u) <= 20]
    src_words = rng_t.sample(words, min(300, len(words)))

    print(f"[typo-diag] {len(ckpts)} arms; val {len(val_units)}, probe {len(probe)}, "
          f"src words {len(src_words)}", flush=True)
    for path in ckpts:
        model, state = load_model(path)
        name = Path(path).parent.name
        # 1. clean val
        ce, bacc, em = batch_loss(model, *tensorize(val_units[:1024], model.l_max, DEV))
        # wait: batch_loss returns train-mode metrics incl. em; fine (eval tensors)
        line = f"\n### {name} (step {state['step']})\n  clean val(1024): CE {float(ce):.4f} EM {em:.4f}"
        # 2. noise curve
        lam = encode_all(model, probe)
        for s in NOISE_GRID:
            torch.manual_seed(0)
            with torch.no_grad():
                dec = [bytes(o) for o in model.decode(lam + torch.randn_like(lam).to(DEV) * s)]
            ed = [lev(a, b[: model.l_max]) for a, b in zip(dec, probe)]
            ed = np.array(ed)
            wrong = ed[ed > 0]
            line += (f"\n  noise {s:.2f}: EM {(ed == 0).mean():.4f} | "
                     f"wrong: medEd {np.median(wrong) if len(wrong) else 0:.1f}, "
                     f"edit<=2 {np.mean(wrong <= 2) if len(wrong) else 1:.3f}")
        # 3. typo-variant neighborhood
        variants, srcs = [], []
        rng_v = random.Random(7777)
        for w in src_words:
            for _ in range(3):
                v = corrupt(w, rng_v)
                if v is not None:
                    variants.append(v); srcs.append(w)
        lv = F.normalize(encode_all(model, variants), dim=-1)
        ls_ = F.normalize(encode_all(model, srcs), dim=-1)
        lp = F.normalize(encode_all(model, probe[:500]), dim=-1)
        cos_sv = (lv * ls_).sum(-1).numpy()
        cos_bg = (lv @ lp.T).max(-1).numpy()
        line += (f"\n  typo-variant cos to SOURCE: mean {cos_sv.mean():.3f} | "
                 f"max-cos to probe background: {cos_bg.mean():.3f} | "
                 f"source-closer rate {(cos_sv > cos_bg).mean():.3f}")
        # 4. geometry quick
        lat = encode_all(model, probe)
        lat_n = F.normalize(lat, dim=-1)
        sim = (lat_n @ lat_n.T).numpy()
        n = len(lat)
        aniso = float(sim[~np.eye(n, dtype=bool)].mean())
        X = (lat - lat.mean(0)).numpy()
        sv = np.linalg.svd(X, compute_uv=False)
        pr = float(sv.sum() ** 2 / (sv**2).sum())
        line += f"\n  geometry: aniso {aniso:+.3f} | PR {pr:.1f}/{lat.shape[1]}"
        print(line, flush=True)
    print("\n[typo-diag] DONE", flush=True)


if __name__ == "__main__":
    main()

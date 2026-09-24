"""Variant-cloud spacing vs sigma floor (2026-09-24, user pushback):

Global NN gap (1.05) is the WRONG scale for the sigma question. The task
scale is INTRA-CLOUD: leading-space / case / suffix / 1-2-byte lookalike
variants form a point cloud around each word; the GMM's sigma must resolve
WITHIN that cloud (failures are dominated by edit<=2 confusions, 53%).

Measure on the snap table (Qwen vocab + cache-top, the GMM's actual support):
  * for sampled anchor words: top-15 lambda-NNs within the table, byte edit
    distance of each -> lambda-distance distribution bucketed by edit{1,2,3,4+}
  * per-anchor distance to the nearest edit<=2 variant (the confusion set)
  * verdict line: p10/p50 of edit<=2 variant spacing vs sigma_floor 0.05
    and vs the pi-weighted sigma the model actually uses (0.05 pinned).

  python scripts/diag_v2_variants.py <ae_ckpt> --units-pkl <pkl> [--topn 100000]
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
from diag_v2 import lev
from snap_decode import build_table
from train_ae import tensorize

DEV = "cuda"
N_ANCHOR = 3000
KNN = 15


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ae_path = sys.argv[1]
    pkl = sys.argv[sys.argv.index("--units-pkl") + 1]
    topn = int(sys.argv[sys.argv.index("--topn") + 1]) if "--topn" in sys.argv else 100_000
    src = pickle.load(open(pkl, "rb"))
    ae_state = torch.load(ae_path, map_location="cpu", weights_only=False)
    ae = ByteStringAE(Cfg(ae_state["cfg"]))
    ae.load_state_dict(ae_state["model"])
    ae = ae.to(DEV).eval()
    print(f"[load] ae step {ae_state.get('step')}", flush=True)

    table, V, Vn, V2 = build_table(ae, "data/qwen35_tokenizer", src["freq_units"], topn)

    rng = np.random.default_rng(0)
    ai = rng.choice(len(table), size=min(N_ANCHOR, len(table)), replace=False)
    A = V[ai]                                  # (N, 48)
    sim = (A @ Vn.T)                           # (N, |V|) cosine
    d2 = (A.pow(2).sum(1, keepdim=True) + V2.sum(1).unsqueeze(0) - 2 * (A @ V.T)).clamp_min(0)
    for r in range(N_ANCHOR):
        d2[r, ai[r]] = float("inf")
    top = d2.topk(KNN, largest=False).indices.cpu().numpy()
    dist = d2.topk(KNN, largest=False).values.sqrt().cpu().numpy()

    buckets: dict[int, list[float]] = {1: [], 2: [], 3: [], 4: []}
    nearest_le2 = []
    t0 = 0
    import time
    t0 = time.time()
    for r in range(N_ANCHOR):
        a = table[ai[r]]
        first_le2 = None
        for j in range(KNN):
            b = table[int(top[r, j])]
            e = lev(a, b)
            buckets[e if e <= 3 else 4].append(float(dist[r, j]))
            if e <= 2 and first_le2 is None:
                first_le2 = float(dist[r, j])
        if first_le2 is not None:
            nearest_le2.append(first_le2)
        if (r + 1) % 500 == 0:
            print(f"  [heartbeat] anchor {r + 1}/{N_ANCHOR} "
                  f"({(r + 1) / max(time.time() - t0, 1e-9):.0f}/s)", flush=True)

    for e in (1, 2, 3, 4):
        v = np.array(buckets[e])
        lab = f"edit={e}" if e <= 3 else "edit>=4"
        print(f"[{lab}] n={len(v)} lambda-dist p10 {np.percentile(v, 10):.3f} "
              f"p25 {np.percentile(v, 25):.3f} p50 {np.percentile(v, 50):.3f} "
              f"p75 {np.percentile(v, 75):.3f}", flush=True)
    v = np.array(nearest_le2)
    print(f"[nearest edit<=2 variant] n={len(v)}/{N_ANCHOR} | "
          f"p10 {np.percentile(v, 10):.3f} p25 {np.percentile(v, 25):.3f} "
          f"p50 {np.percentile(v, 50):.3f} p75 {np.percentile(v, 75):.3f} "
          f"p90 {np.percentile(v, 90):.3f}", flush=True)
    print(f"[verdict] vs sigma_floor 0.05: ratio of edit<=2 variant spacing to sigma: "
          f"p10 {np.percentile(v, 10) / 0.05:.1f}x p50 {np.percentile(v, 50) / 0.05:.1f}x",
          flush=True)
    print("[variants] DONE", flush=True)


if __name__ == "__main__":
    main()

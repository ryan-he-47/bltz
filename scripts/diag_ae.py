"""AE embedding-geometry diagnostics (docs/32 S0): on the trained ckpt, encode
held-out strings and report the geometry suite — PR / anisotropy / kNN-CC /
Silhouette Coefficient (CoSE: SC ~ prediction quality, Pearson 0.92).

  python scripts/diag_ae.py <ae_ckpt.pt> [--cache data/cache_dryrun]
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn.functional as F

from bltz.config import Cfg
from bltz.models.autoencoder import ByteStringAE

DEV = "cuda"


def knn_cc(sim: np.ndarray, k: int = 5) -> float:
    n = sim.shape[0]
    s = sim.copy()
    np.fill_diagonal(s, -2)
    knn = np.argpartition(-s, k, axis=1)[:, :k]
    adj = np.zeros((n, n), dtype=bool)
    for i in range(n):
        adj[i, knn[i]] = True
    adj |= adj.T
    cc = []
    for i in range(n):
        nb = np.nonzero(adj[i])[0]
        if len(nb) < 2:
            continue
        links = adj[np.ix_(nb, nb)].sum() / 2
        cc.append(links / (len(nb) * (len(nb) - 1) / 2))
    return float(np.mean(cc))


def main() -> None:
    path = sys.argv[1]
    state = torch.load(path, map_location="cpu", weights_only=False)
    cfg = Cfg(state["cfg"])
    model = ByteStringAE(cfg)
    model.load_state_dict(state["model"])
    model = model.to(DEV).eval()

    # rebuild the same pool split as training (same seed / order)
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from train_ae import build_pool, tensorize

    train_pool, val_pool = build_pool(cfg)
    rng = random.Random(4242)
    probe = rng.sample(val_pool, min(2000, len(val_pool)))

    lats = []
    with torch.no_grad():
        for i in range(0, len(probe), 512):
            chunk = probe[i : i + 512]
            byte_ids, lens, pad_mask = tensorize(chunk, model.l_max, DEV)
            lats.append(model.encoder(byte_ids, pad_mask).float().cpu())
    lat = torch.cat(lats)
    # recon sanity on the probe set
    rec = model.decode(model.encoder(byte_ids, pad_mask).to(DEV))  # last chunk
    em = np.mean([bytes(r) == c[: model.l_max] for r, c in zip(rec, chunk)])

    lat_n = F.normalize(lat, dim=-1)
    sim = (lat_n @ lat_n.T).numpy()
    n = len(lat)
    aniso = float(sim[~np.eye(n, dtype=bool)].mean())
    X = (lat - lat.mean(0)).numpy()
    sv = np.linalg.svd(X, compute_uv=False)
    pr = float(sv.sum() ** 2 / (sv**2).sum())
    print(f"ckpt step {state['step']}, probe n={n}, d_emb={lat.shape[1]}")
    print(f"last-chunk exact-match: {em:.3f}")
    print(f"anisotropy (mean pairwise cos): {aniso:+.3f}")
    print(f"participation ratio: {pr:.1f} / {lat.shape[1]}")
    print(f"kNN(k=5) clustering coeff: {knn_cc(sim):.3f}")

    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    scs = []
    Xn = lat_n.numpy()
    for k in (5, 10, 15, 20, 25):
        lab = KMeans(n_clusters=k, n_init=4, random_state=0).fit_predict(Xn)
        scs.append(silhouette_score(Xn, lab, metric="cosine"))
    print(f"silhouette (cosine, k=5..25 mean): {np.mean(scs):.3f}  {[round(s, 3) for s in scs]}")

    # nearest neighbors of a few probe strings (interpretability)
    for q in ["the", "ing", "19", " = ", "\\n"]:
        qb = q.encode()
        with torch.no_grad():
            qids, qlen, qmask = tensorize([qb], model.l_max, DEV)
            ql = F.normalize(model.encoder(qids, qmask).float().cpu(), dim=-1)
        nn = (ql @ lat_n.T).squeeze(0).numpy().argsort()[::-1][:8]
        print(f"NN of {qb!r}: {[probe[i] for i in nn]}")


if __name__ == "__main__":
    main()

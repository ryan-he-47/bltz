"""AE full acceptance trio (2026-09-19 拍板: diag BEFORE the POC job):

  1. geometry suite on real cache units (PR / anisotropy / kNN-CC / SC / NN)
  2. frequency-stratified val — val strings bucketed by their occurrence count
     in a training-draw sample; the ZERO-freq bucket is the true string-level
     generalization number (stream-mode val alone is same-distribution, head-
     dominated)
  3. distinct-string growth curve + Heaps' law fit (V = K·n^β) extrapolated to
     the full cache — the AE's EFFECTIVE sample size (segmentation-diversity-
     bounded, not text-volume-bounded)

Two data modes (identical rng draws either way):
  cluster-direct:  python scripts/diag_ae_v2.py <ckpt.pt> --cache <v2 cache dir>
  local no-quota:  python scripts/diag_ae_v2.py <ckpt.pt> --units-pkl <pkl>
                   (pkl produced by the login-node sampler — keys: freq_units,
                   val_units, probe_pool, distinct_seq_units, n_seq, total_units)
"""
from __future__ import annotations

import pickle
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
from train_ae import batch_loss, tensorize

DEV = "cuda"

FREQ_SEQS = 256        # training-draw frequency sample (~131k units)
PROBE_SEQS = 24        # geometry probe pool source
DISTINCT_LADDER = (10, 50, 100, 300, 1000)  # sequences


def load_sources(cache: str) -> dict:
    from bltz.shards import ShardReader

    reader = ShardReader(cache)
    n_seq = reader.n_sequences(512)
    total_units = sum(s["meta"]["n_units"] for s in reader.shards)
    freq_units = [u for gi in random.Random(31337).sample(range(n_seq), min(FREQ_SEQS, n_seq))
                  for u in reader.sequence_units(gi, 512)]
    val_units = [u for gi in random.Random(777).sample(range(n_seq), min(4, n_seq))
                 for u in reader.sequence_units(gi, 512)]
    probe_pool = [u for gi in random.Random(4242).sample(range(n_seq), min(PROBE_SEQS, n_seq))
                  for u in reader.sequence_units(gi, 512)]
    ladder_max = min(max(DISTINCT_LADDER), n_seq)
    distinct_seq_units = [reader.sequence_units(gi, 512)
                          for gi in random.Random(999).sample(range(n_seq), ladder_max)]
    return {
        "freq_units": freq_units, "val_units": val_units, "probe_pool": probe_pool,
        "distinct_seq_units": distinct_seq_units, "n_seq": n_seq, "total_units": total_units,
    }


def encode_all(model, strings, dev) -> torch.Tensor:
    lats = []
    with torch.no_grad():
        for i in range(0, len(strings), 512):
            byte_ids, lens, pad_mask = tensorize(strings[i : i + 512], model.l_max, dev)
            lats.append(model.encoder(byte_ids, pad_mask).float().cpu())
    return torch.cat(lats)


def recon_metrics(model, strings, dev):
    """Teacher-forced CE / byte-acc / EM over the bucket (any size)."""
    if not strings:
        return float("nan"), float("nan"), float("nan")
    ce_t, bacc_t, em_t, nb = 0.0, 0.0, 0.0, 0
    with torch.no_grad():
        for i in range(0, len(strings), 512):
            chunk = strings[i : i + 512]
            byte_ids, lens, pad_mask = tensorize(chunk, model.l_max, dev)
            ce, bacc, em = batch_loss(model, byte_ids, lens, pad_mask)
            ce_t += float(ce); bacc_t += bacc; em_t += em; nb += 1
    return ce_t / nb, bacc_t / nb, em_t / nb


def main() -> None:
    path = sys.argv[1]
    if "--units-pkl" in sys.argv:
        with open(sys.argv[sys.argv.index("--units-pkl") + 1], "rb") as f:
            src = pickle.load(f)
    else:
        cache = sys.argv[sys.argv.index("--cache") + 1] if "--cache" in sys.argv else "data/cache_dryrun"
        src = load_sources(cache)
    state = torch.load(path, map_location="cpu", weights_only=False)
    model = ByteStringAE(Cfg(state["cfg"]))
    model.load_state_dict(state["model"])
    model = model.to(DEV).eval()
    n_seq, total_units = src["n_seq"], src["total_units"]
    print(f"[diag] ckpt {path} step {state['step']}; n_seq={n_seq}, "
          f"total_units={total_units/1e9:.2f}B", flush=True)

    # ---- part 2: frequency-stratified val ----------------------------------
    freq: Counter[bytes] = Counter(src["freq_units"])
    print(f"[freq] {FREQ_SEQS} seqs -> {len(src['freq_units'])} units, "
          f"{len(freq)} distinct strings", flush=True)
    buckets: dict[str, list[bytes]] = {"zero": [], "low(1-3)": [], "mid(4-63)": [], "high(>63)": []}
    for u in src["val_units"]:
        c = freq.get(u, 0)
        b = "zero" if c == 0 else "low(1-3)" if c <= 3 else "mid(4-63)" if c <= 63 else "high(>63)"
        buckets[b].append(u)
    print("\n=== frequency-stratified val ===", flush=True)
    for name, ss in buckets.items():
        ce, bacc, em = recon_metrics(model, ss, DEV)
        print(f"  {name:10s} n={len(ss):5d}  CE {ce:.4f}  bacc {bacc:.4f}  EM {em:.4f}", flush=True)

    # ---- part 1: geometry on cache units -----------------------------------
    rng_g = random.Random(4242)
    rng_g.sample(range(n_seq), min(PROBE_SEQS, n_seq))  # rng parity with sampler
    probe = rng_g.sample(src["probe_pool"], min(2000, len(src["probe_pool"])))
    lat = encode_all(model, probe, DEV)
    lat_n = F.normalize(lat, dim=-1)
    sim = (lat_n @ lat_n.T).numpy()
    n = len(lat)
    aniso = float(sim[~np.eye(n, dtype=bool)].mean())
    X = (lat - lat.mean(0)).numpy()
    sv = np.linalg.svd(X, compute_uv=False)
    pr = float(sv.sum() ** 2 / (sv**2).sum())
    s2 = sim.copy(); np.fill_diagonal(s2, -2)
    knn = np.argpartition(-s2, 5, axis=1)[:, :5]
    adj = np.zeros((n, n), dtype=bool)
    for i in range(n):
        adj[i, knn[i]] = True
    adj |= adj.T
    ccs = []
    for i in range(n):
        nb = np.nonzero(adj[i])[0]
        if len(nb) >= 2:
            ccs.append(adj[np.ix_(nb, nb)].sum() / 2 / (len(nb) * (len(nb) - 1) / 2))
    print("\n=== geometry ===", flush=True)
    print(f"  probe n={n}, d_emb={lat.shape[1]}")
    print(f"  anisotropy {aniso:+.3f} | PR {pr:.1f}/{lat.shape[1]} | kNN-CC {np.mean(ccs):.3f}", flush=True)
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    Xn = lat_n.numpy()
    scs = [silhouette_score(Xn, KMeans(k, n_init=4, random_state=0).fit_predict(Xn), metric="cosine")
           for k in (5, 10, 15, 20, 25)]
    print(f"  SC(cos,k=5..25) mean {np.mean(scs):.3f} {[round(s, 3) for s in scs]}", flush=True)
    for q in ["the", "ing", "19", " = ", "\n", "antidiscrimination"]:
        qb = q.encode()
        with torch.no_grad():
            qids, _qlen, qmask = tensorize([qb], model.l_max, DEV)
            ql = F.normalize(model.encoder(qids, qmask).float().cpu(), dim=-1)
        nn_idx = (ql @ lat_n.T).squeeze(0).numpy().argsort()[::-1][:8]
        print(f"  NN of {qb!r}: {[probe[i] for i in nn_idx]}", flush=True)

    # ---- part 3: distinct-string growth + Heaps -----------------------------
    ladder = [t for t in DISTINCT_LADDER if t <= len(src["distinct_seq_units"])] \
        or [len(src["distinct_seq_units"])]
    seen: set[bytes] = set()
    xs, ys = [], []
    ptr = 0
    for target in ladder:
        while ptr < target:
            seen.update(src["distinct_seq_units"][ptr])
            ptr += 1
        xs.append(target * 512)
        ys.append(len(seen))
    logx = np.log(np.array(xs, float))
    logy = np.log(np.array(ys, float))
    beta, logK = np.polyfit(logx, logy, 1)
    est = float(np.exp(logK) * total_units ** beta)
    print("\n=== distinct-string growth ===", flush=True)
    print("  ladder:", list(zip(xs, ys)), flush=True)
    print(f"  Heaps fit: V = e^{logK:.2f} · n^{beta:.3f};  at {total_units/1e9:.1f}B units "
          f"-> est distinct ~{est/1e6:.0f}M", flush=True)
    print("[diag] DONE", flush=True)


if __name__ == "__main__":
    main()

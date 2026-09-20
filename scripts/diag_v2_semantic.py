"""v2 semantic-structure diagnostics (2026-09-20, user direction):

The lambda space is ORTHOGRAPHY-shaped (reconstruction pressure), not semantic.
So look for semantics where it can actually live — in h (768-dim backbone
state, shaped by prediction pressure):

  part 1 — failure-case dump: short context + top-5 candidate table for
           high-pi wrong predictions with large angular miss; eyeball whether
           they are gibberish or semantically-adjacent-but-spelled-differently.
           -> checkpoints/semantic_cases.txt
  part 2 — h clustering vs lambda clustering: k-means on both spaces over the
           same positions; NMI against string coarse types; intra-cluster mean
           pairwise edit distance (semantic space should be orthographically
           LOOSER but type-CLEANER); per-cluster member printout; t-SNE png
           (viz/h_tsne.png) colored by string type.

  python scripts/diag_v2_semantic.py <v2_ckpt.pt> <ae_ckpt.pt> --units-pkl <pkl>
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
import torch.nn.functional as F

from bltz.config import Cfg
from bltz.models.autoencoder import ByteStringAE
from bltz.models.model_v2 import BltzLMv2
from diag_v2 import conf_scores, lev
from train_ae import tensorize

DEV = "cuda"
EVAL_SEQS = 32
TOPK = 5
N_CASES = 120
N_CLUSTERS = 60


def str_type(s: bytes) -> str:
    if not s.strip(b" \t\n\r"):
        return "space"
    if any(48 <= b <= 57 for b in s):
        return "digitful"
    letters = [b for b in s if 65 <= b <= 90 or 97 <= b <= 122]
    if not letters:
        return "symbol"
    core = s.decode("utf-8", errors="ignore").strip()
    if core and core[0].isupper():
        return "Capitalized"
    if all(97 <= b <= 122 or b < 65 or b > 122 for b in s):
        return "lower"
    return "mixed"


def nmi(labels_true: np.ndarray, labels_pred: np.ndarray) -> float:
    from sklearn.metrics import normalized_mutual_info_score

    return float(normalized_mutual_info_score(labels_true, labels_pred))


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    v2_path, ae_path = sys.argv[1], sys.argv[2]
    pkl = sys.argv[sys.argv.index("--units-pkl") + 1]
    src = pickle.load(open(pkl, "rb"))

    ae_state = torch.load(ae_path, map_location="cpu", weights_only=False)
    ae = ByteStringAE(Cfg(ae_state["cfg"]))
    ae.load_state_dict(ae_state["model"])
    v2_state = torch.load(v2_path, map_location="cpu", weights_only=False)
    model = BltzLMv2(Cfg(v2_state["cfg"]), ae)
    model.load_state_dict(v2_state["model"])
    model = model.to(DEV).eval()

    # ---------- collect per-position records ---------------------------------
    seqs = src["distinct_seq_units"][:EVAL_SEQS]
    hs, lams, trues, ctxs = [], [], [], []
    top1_pi, rec_cos = [], []
    with torch.no_grad():
        for units in seqs:
            byte_ids, lens, pad_mask = tensorize(units, model.l_max, DEV)
            lam = model.encode_units(byte_ids.unsqueeze(0), pad_mask.unsqueeze(0))
            S, D = lam.shape[1], lam.shape[2]
            x = torch.cat([model.bos.expand(1, 1, -1), lam[:, :-1]], dim=1)
            h = model.backbone(model.adapter(x))
            logit_pi, mu, _ = (t.float() for t in model.head.params(h))
            cos = F.cosine_similarity(mu[0, :, :, :].float(),
                                      lam[0].float().unsqueeze(1).expand(-1, 64, -1), dim=-1)
            for i in range(2, S - 1):
                hs.append(h[0, i].float().cpu())
                lams.append(lam[0, i].float().cpu())
                trues.append(units[i][: model.l_max])
                ctxs.append(b"".join(units[max(0, i - 6) : i]))
                k = int(logit_pi[0, i].argmax())
                top1_pi.append(float(F.softmax(logit_pi[0, i], -1)[k]))
                rec_cos.append(float(cos[i, k]))
    H = torch.stack(hs)
    LAM = torch.stack(lams)
    print(f"[sem] collected {len(trues)} positions", flush=True)

    # ---------- part 1: failure-case dump -------------------------------------
    records = []
    with torch.no_grad():
        ptr = 0
        for units in seqs:
            byte_ids, lens, pad_mask = tensorize(units, model.l_max, DEV)
            lam = model.encode_units(byte_ids.unsqueeze(0), pad_mask.unsqueeze(0))
            S, D = lam.shape[1], lam.shape[2]
            x = torch.cat([model.bos.expand(1, 1, -1), lam[:, :-1]], dim=1)
            h = model.backbone(model.adapter(x))
            logit_pi, mu, _ = (t.float() for t in model.head.params(h))
            topk = logit_pi[0].topk(TOPK, -1).indices
            for i in range(2, S - 1):
                mu_c = mu[0, i, topk[i]].float()
                dl = model.ae.decode(mu_c)
                cf = conf_scores(model.ae, mu_c, dl)
                records.append((ptr, [bytes(d) for d in dl], cf,
                                F.softmax(logit_pi[0, i], -1)[topk[i]].cpu().numpy()))
                ptr += 1
    assert len(records) == len(trues)
    cases = []
    for i, (_p, cands, cft, pis) in enumerate(records):
        em0 = cands[0] == trues[i]
        if em0 or top1_pi[i] < 0.40:
            continue
        cases.append((rec_cos[i], i, cands, cft, pis))
    cases.sort(key=lambda t: t[0])  # most angularly-deviated first
    with open("checkpoints/semantic_cases.txt", "w", encoding="utf-8") as f:
        for cosv, i, cands, cft, pis in cases[:N_CASES]:
            f.write(f"\n=== ctx: ...{ctxs[i][-72:]!r}\n"
                    f"    TRUE: {trues[i]!r}  (type={str_type(trues[i])})\n")
            for j in range(TOPK):
                f.write(f"    cand{j}: {cands[j]!r:24s} pi={pis[j]:.3f} "
                        f"conf={cft[j]:+.2f} edit={lev(cands[j], trues[i])}\n")
    print(f"[sem] dumped {min(N_CASES, len(cases))} high-conf wrong cases "
          f"(pool {len(cases)}) -> checkpoints/semantic_cases.txt", flush=True)

    # ---------- part 2: h clustering vs lambda clustering ---------------------
    from sklearn.cluster import KMeans

    types = np.array([str_type(t) for t in trues])
    Hn = F.normalize(H, dim=-1).numpy()
    Ln = F.normalize(LAM, dim=-1).numpy()
    km_h = KMeans(N_CLUSTERS, n_init=3, random_state=0).fit_predict(Hn)
    km_l = KMeans(N_CLUSTERS, n_init=3, random_state=0).fit_predict(Ln)
    print(f"\n[cluster] NMI(h-clusters, types) = {nmi(types, km_h):.3f} | "
          f"NMI(lambda-clusters, types) = {nmi(types, km_l):.3f}", flush=True)

    def intra_edit(km, tag):
        import random as _r

        rng = _r.Random(0)
        dists = []
        for c in range(N_CLUSTERS):
            members = [t for t, k in zip(trues, km) if k == c]
            if len(members) >= 4:
                for a, b in zip(rng.sample(members, 2), rng.sample(members, 2)):
                    dists.append(lev(a, b) / max(max(len(a), len(b)), 1))
        print(f"[cluster] intra-cluster normalized edit distance ({tag}): "
              f"{np.mean(dists):.3f}", flush=True)

    intra_edit(km_h, "h space")
    intra_edit(km_l, "lambda space")

    # per-cluster member printout (h space): type purity + samples
    print("\n[cluster] h-space clusters (top-25 by size):", flush=True)
    order = np.argsort(np.bincount(km_h))[::-1][:25]
    for c in order:
        members = [t for t, k in zip(trues, km_h) if k == c]
        mtypes = [str_type(t) for t in members]
        pur = max(set(mtypes), key=mtypes.count)
        pur_r = mtypes.count(pur) / len(mtypes)
        uniq = list(dict.fromkeys(members))[:8]
        print(f"  c{c:02d} n={len(members):4d} purity {pur}:{pur_r:.2f} "
              f"samples: {[u.decode('utf-8', errors='replace') for u in uniq]}", flush=True)

    # t-SNE of h colored by coarse type
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE

    sub = np.random.RandomState(0).choice(len(trues), min(6000, len(trues)), replace=False)
    emb = TSNE(n_components=2, init="pca", random_state=0, perplexity=40).fit_transform(Hn[sub])
    labs = types[sub]
    palette = {"space": "#888888", "digitful": "#d62728", "symbol": "#9467bd",
               "Capitalized": "#2ca02c", "lower": "#1f77b4", "mixed": "#ff7f0e"}
    fig, ax = plt.subplots(figsize=(8, 7))
    for t, c in palette.items():
        m = labs == t
        ax.scatter(emb[m, 0], emb[m, 1], s=3, c=c, label=f"{t} ({m.sum()})", alpha=0.55)
    ax.legend(markerscale=3, fontsize=9)
    ax.set_title("backbone h t-SNE, colored by true-next-string type")
    fig.tight_layout()
    fig.savefig("viz/h_tsne.png", dpi=140)
    print("\n[sem] saved viz/h_tsne.png", flush=True)
    print("[sem] DONE", flush=True)


if __name__ == "__main__":
    main()

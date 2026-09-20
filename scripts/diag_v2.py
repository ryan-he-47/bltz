"""v2 POC capability eval (docs/32 S4): the byte-level answer to "does it work".

On eval sequences (v2 cache sample, --units-pkl mode from diag_ae_v2):
  1. next-unit prediction under THREE readouts over the top-5 pi components:
     argmax-pi / min-sigma / decoder-confidence rerank (decode each candidate's
     mu, then score the decoded string's own likelihood under that component —
     the decoder doubles as an on-manifold verifier);
  2. byte-level match (Levenshtein) between predicted and true strings —
     separates "complete gibberish" from "typo-level" failures;
  3. held-out-ish NLL, pi stats (collapse watch);
  4. generation with component sampling + confidence-rerank reprojection.

  python scripts/diag_v2.py <v2_ckpt.pt> <ae_ckpt.pt> --units-pkl <pkl>
"""
from __future__ import annotations

import pickle
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
import torch.nn.functional as F

from bltz.config import Cfg
from bltz.models.autoencoder import EOS_ID, ByteStringAE
from bltz.models.model_v2 import BltzLMv2
from train_ae import tensorize

DEV = "cuda"
EVAL_SEQS = 32
GEN_PREFIX = 16
GEN_STEPS = 64
TOPK = 5


def lev(a: bytes, b: bytes) -> int:
    """Byte-level Levenshtein edit distance."""
    la, lb = len(a), len(b)
    if la < lb:
        a, b, la, lb = b, a, lb, la
    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        cur = [i] + [0] * lb
        ca = a[i - 1]
        for j in range(1, lb + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != b[j - 1]))
        prev = cur
    return prev[lb]


@torch.no_grad()
def conf_scores(ae: ByteStringAE, mu: torch.Tensor, dec_lists: list[list[int]]) -> np.ndarray:
    """Self-consistency: mean logprob of the component's OWN decoded string
    under its decoder (incl. EOS). Higher = more on-manifold / committed."""
    reps = [len(d) + 1 for d in dec_lists]
    reps_t = torch.tensor(reps, device=DEV)
    mu_q = mu.repeat_interleave(reps_t, dim=0)
    d_idx = torch.cat([torch.arange(r, device=DEV) for r in reps])
    tgt = torch.cat([torch.tensor(d + [EOS_ID], device=DEV) for d in dec_lists])
    out = torch.zeros(len(dec_lists), device=DEV)
    idx_map = torch.repeat_interleave(torch.arange(len(dec_lists), device=DEV), reps_t)
    step = 200_000
    for s in range(0, d_idx.shape[0], step):
        lp = F.log_softmax(ae.decoder(mu_q[s : s + step], d_idx[s : s + step]).float(), dim=-1)
        out.scatter_add_(0, idx_map[s : s + step],
                         lp.gather(-1, tgt[s : s + step, None]).squeeze(-1))
    return (out / reps_t.float()).cpu().numpy()


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # GBK console safe
    v2_path, ae_path = sys.argv[1], sys.argv[2]
    pkl = sys.argv[sys.argv.index("--units-pkl") + 1]
    src = pickle.load(open(pkl, "rb"))

    ae_state = torch.load(ae_path, map_location="cpu", weights_only=False)
    ae = ByteStringAE(Cfg(ae_state["cfg"]))
    ae.load_state_dict(ae_state["model"])
    v2_state = torch.load(v2_path, map_location="cpu", weights_only=False)
    cfg = Cfg(v2_state["cfg"])
    model = BltzLMv2(cfg, ae)
    model.load_state_dict(v2_state["model"])
    model = model.to(DEV).eval()
    print(f"[diag_v2] v2 step {v2_state['step']}, d_emb={model.ae.encoder.out_proj.out_features}",
          flush=True)

    seqs = src["distinct_seq_units"][:EVAL_SEQS]

    nll_all, ent_all, eff_all, top1_all = [], [], [], []
    stats = {name: {"em": 0, "edits": [], "ned": [], "prefix": []}
             for name in ("argmax_pi", "min_sigma", "conf_rerank")}
    cov_top5_dec = 0  # true string among decoded top-5 candidates
    tot = 0

    with torch.no_grad():
        for units in seqs:
            byte_ids, lens, pad_mask = tensorize(units, model.l_max, DEV)
            lam = model.encode_units(byte_ids.unsqueeze(0), pad_mask.unsqueeze(0))
            S, D = lam.shape[1], lam.shape[2]
            x = torch.cat([model.bos.expand(1, 1, -1), lam[:, :-1]], dim=1)
            h = model.backbone(model.adapter(x))
            logit_pi, mu, sigma = (t.float() for t in model.head.params(h))
            # NLL + pi stats (skip BOS position 0)
            z = (lam.unsqueeze(-2).float() - mu) / sigma
            log_n = -0.5 * z.pow(2).sum(-1) - sigma.log().sum(-1) - 0.5 * D * np.log(2 * np.pi)
            nll = -torch.logsumexp(F.log_softmax(logit_pi, -1) + log_n, -1)
            nll_all.append(nll[0, 1:].mean().item())
            pi = F.softmax(logit_pi[0, 1:], -1)
            ent_all.append((-(pi * pi.clamp_min(1e-12).log()).sum(-1)).mean().item())
            eff_all.append((1.0 / pi.pow(2).sum(-1)).mean().item())
            top1_all.append(pi.max(-1).values.mean().item())

            # candidates: top-K pi components at positions 1..S-2
            lp = logit_pi[0]                       # (S, K)
            mu_s = mu[0].float()                   # (S, K, D)
            sg_s = sigma[0].float().mean(-1)       # (S, K)
            topk = lp.topk(TOPK, dim=-1).indices   # (S, K)
            pos_ids, cand_mu, cand_sig = [], [], []
            for i in range(1, S - 1):
                for j in range(TOPK):
                    pos_ids.append(i)
                    cand_mu.append(mu_s[i, topk[i, j]])
                    cand_sig.append(sg_s[i, topk[i, j]].item())
            cand_mu_t = torch.stack(cand_mu)
            dec_lists = model.ae.decode(cand_mu_t)               # raw decode
            confs = conf_scores(model.ae, cand_mu_t, dec_lists)  # self-consistency
            w = 0
            for i in range(1, S - 1):
                cands = [bytes(d) for d in dec_lists[w : w + TOPK]]
                conf_k = confs[w : w + TOPK]
                sig_k = cand_sig[w : w + TOPK]
                w += TOPK
                true_b = units[i][: model.l_max]
                picks = {
                    "argmax_pi": cands[0],
                    "min_sigma": cands[int(np.argmin(sig_k))],
                    "conf_rerank": cands[int(np.argmax(conf_k))],
                }
                cov_top5_dec += int(true_b in cands)
                tot += 1
                for name, pred in picks.items():
                    st = stats[name]
                    e = lev(pred, true_b)
                    st["em"] += int(e == 0)
                    st["edits"].append(e)
                    st["ned"].append(e / max(len(true_b), 1))
                    p = 0
                    for a, b in zip(pred, true_b):
                        if a != b:
                            break
                        p += 1
                    st["prefix"].append(p / max(len(true_b), 1))

    print("\n=== capability (eval seqs, next-unit) ===", flush=True)
    print(f"  held-out NLL {np.mean(nll_all):.3f}")
    print(f"  true string among decoded top-5 candidates: {cov_top5_dec / tot:.4f}", flush=True)
    print(f"\n  {'readout':12s} {'EM':>7s} {'medEdits':>9s} {'meanNED':>8s} {'prefix%':>8s} "
          f"{'edit<=1':>8s} {'edit<=2':>8s} {'edit>3':>7s}", flush=True)
    for name, st in stats.items():
        ed = np.array(st["edits"])
        print(f"  {name:12s} {st['em'] / tot:7.4f} {float(np.median(ed)):9.1f} "
              f"{float(np.mean(st['ned'])):8.3f} {np.mean(st['prefix']):8.3f} "
              f"{(ed <= 1).mean():8.3f} {(ed <= 2).mean():8.3f} {(ed > 3).mean():7.3f}",
              flush=True)
    print("\n=== pi stats (collapse watch) ===", flush=True)
    print(f"  entropy {np.mean(ent_all):.2f} nats | effective comps "
          f"{np.mean(eff_all):.1f}/64 | top1 share {np.mean(top1_all):.3f}", flush=True)

    # ---- generation with confidence-rerank reprojection ----------------------
    print("\n=== generation (sample component, conf-rerank top-3, reproject) ===", flush=True)
    prompt_units = seqs[0][:GEN_PREFIX]
    byte_ids, lens, pad_mask = tensorize(prompt_units, model.l_max, DEV)
    lam0 = model.encode_units(byte_ids.unsqueeze(0), pad_mask.unsqueeze(0))
    for tau in (0.7, 1.0):
        torch.manual_seed(0)
        cur = torch.cat([model.bos.expand(1, 1, -1), lam0], dim=1)
        outs: list[str] = []
        confs_out: list[float] = []
        with torch.no_grad():
            for _ in range(GEN_STEPS):
                h = model.backbone(model.adapter(cur))[:, -1:]
                logit_pi, mu, _ = model.head.params(h)
                lp = logit_pi[0, 0].float()
                k3 = lp.topk(3).indices
                if tau <= 0:
                    kk = k3[:1]
                else:
                    probs = F.softmax(lp[k3] / tau, dim=-1)
                    kk = k3[torch.multinomial(probs, 1)]
                cmu = mu[0, 0, kk].float()
                dl = model.ae.decode(cmu)
                cs = conf_scores(model.ae, cmu, dl)
                best = int(np.argmax(cs))
                bs = bytes(dl[best])
                confs_out.append(float(cs[best]))
                outs.append(bs.decode("utf-8", errors="replace"))
                bids, _, pmask = tensorize([bs], model.l_max, DEV)
                lam_r = model.encode_units(bids.unsqueeze(0), pmask.unsqueeze(0))
                cur = torch.cat([cur, lam_r], dim=1)
        joined = "".join(outs)
        print(f"\n--- tau={tau} top3-conf-rerank (distinct {len(set(outs))}/{GEN_STEPS}, "
              f"mean conf {np.mean(confs_out):.3f}) ---\n{joined[:600]}", flush=True)
    print("\n[diag_v2] DONE", flush=True)


if __name__ == "__main__":
    main()

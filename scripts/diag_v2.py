"""v2 POC capability eval (docs/32 S4): the byte-level answer to "does it work".

On eval sequences (v2 cache sample, --units-pkl mode from diag_ae_v2):
  1. next-unit prediction: argmax-pi component mean -> decode -> exact-match +
     teacher-forced byte CE of the TRUE next string under the predicted
     embedding; persistence baseline (repeat previous unit) as sanity floor;
  2. held-out-ish NLL vs train NLL (generalization gap);
  3. pi stats: entropy / effective component count / top1 share (collapse
     early-warning);
  4. generation: prefix -> GMM sampling at several tau (word-level temperature,
     the v2 payoff) -> decode each component mean -> re-encode the DECODED
     string for the next step (reprojection, docs/31 §2.4 default-on).

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

    # ---- per-sequence forward: params, NLL, pi stats, next-unit eval --------
    nll_all, ent_all, eff_all, top1_all = [], [], [], []
    em_hit, em_tot, byte_ce_sum, byte_n = 0, 0, 0.0, 0
    pers_hit, pers_tot = 0, 0
    with torch.no_grad():
        for units in seqs:
            byte_ids, lens, pad_mask = tensorize(units, model.l_max, DEV)
            lam = model.encode_units(byte_ids.unsqueeze(0), pad_mask.unsqueeze(0))
            B, S, D = 1, lam.shape[1], lam.shape[2]
            x = torch.cat([model.bos.expand(B, 1, D), lam[:, :-1]], dim=1)
            h = model.backbone(model.adapter(x))
            logit_pi, mu, sigma = (t.float() for t in model.head.params(h))
            # NLL (fp32, same form as training loss)
            z = (lam.unsqueeze(-2).float() - mu) / sigma
            log_n = -0.5 * z.pow(2).sum(-1) - sigma.log().sum(-1) \
                - 0.5 * D * np.log(2 * np.pi)
            nll = -torch.logsumexp(F.log_softmax(logit_pi, -1) + log_n, -1)
            nll_all.append(nll[0, 1:].mean().item())  # skip BOS position
            pi = F.softmax(logit_pi[0, 1:], -1)
            ent_all.append((-(pi * pi.clamp_min(1e-12).log()).sum(-1)).mean().item())
            eff_all.append((1.0 / pi.pow(2).sum(-1)).mean().item())
            top1_all.append(pi.max(-1).values.mean().item())
            # next-unit eval: argmax-pi component mean -> decode
            k = logit_pi[0, 1:].argmax(-1)  # (S-1,)
            mu_hat = mu[0, 1:, :, :].gather(1, k[:, None, None].expand(-1, 1, D)).squeeze(1)
            dec = model.ae.decode(mu_hat)  # list of byte lists, position i predicts units[i+1]
            K = model.l_max + 1
            for i in range(S - 2):  # leave one for the persistence pair
                true_b = list(units[i + 1])
                em_tot += 1
                em_hit += int(dec[i] == true_b)
                pers_tot += 1
                pers_hit += int(list(units[i]) == true_b)
                # teacher-forced byte CE of the true string under mu_hat
                tgt = true_b + [EOS_ID]
                didx = torch.arange(len(tgt), device=DEV)
                lg = model.ae.decoder(mu_hat[i : i + 1].expand(len(tgt), -1), didx)
                byte_ce_sum += F.cross_entropy(
                    lg.float(), torch.tensor(tgt, device=DEV)).item() * len(tgt)
                byte_n += len(tgt)
    print("\n=== capability (eval seqs, next-unit) ===", flush=True)
    print(f"  held-out NLL {np.mean(nll_all):.3f} (train tail ~-32.8)")
    print(f"  next-unit exact-match {em_hit / em_tot:.4f} (n={em_tot}); "
          f"persistence baseline {pers_hit / pers_tot:.4f}")
    print(f"  teacher-forced byte CE of true string under predicted emb: "
          f"{byte_ce_sum / byte_n:.4f} nats/byte")
    print("\n=== pi stats (collapse watch) ===", flush=True)
    print(f"  entropy {np.mean(ent_all):.2f} nats | effective comps "
          f"{np.mean(eff_all):.1f}/64 | top1 share {np.mean(top1_all):.3f}", flush=True)

    # ---- generation ---------------------------------------------------------
    print("\n=== generation (reprojection on) ===", flush=True)
    prompt_units = seqs[0][:GEN_PREFIX]
    byte_ids, lens, pad_mask = tensorize(prompt_units, model.l_max, DEV)
    lam0 = model.encode_units(byte_ids.unsqueeze(0), pad_mask.unsqueeze(0))
    for tau in (0.0, 0.7, 1.0, 1.3):
        torch.manual_seed(0)
        cur = torch.cat([model.bos.expand(1, 1, -1), lam0], dim=1)
        outs: list[str] = []
        with torch.no_grad():
            for _ in range(GEN_STEPS):
                h = model.backbone(model.adapter(cur))[:, -1:]
                nxt = model.head.sample(h, tau=tau)  # (1,1,D) component mean
                bs = bytes(model.ae.decode(nxt.reshape(1, -1))[0])
                outs.append(bs.decode("utf-8", errors="replace"))
                # reprojection: next input = encoder(decoded string)
                bids, _, pmask = tensorize([bs], model.l_max, DEV)
                lam_r = model.encode_units(bids.unsqueeze(0), pmask.unsqueeze(0))
                cur = torch.cat([cur, lam_r], dim=1)
        joined = "".join(outs)
        distinct = len(set(outs))
        print(f"\n--- tau={tau} (distinct {distinct}/{GEN_STEPS}) ---\n{joined[:600]}", flush=True)
    print("\n[diag_v2] DONE", flush=True)


if __name__ == "__main__":
    main()

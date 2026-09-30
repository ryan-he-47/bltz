"""Pushforward bpb (用户设计 2026-09-30): the model as a BYTE language model.

P_model(s) = ∫ GMM(λ) · P_dec(s|λ) dλ   (decoder pushforward)

No table / no OOV / no temperature calibration. Stratified Monte-Carlo per
component (m samples each, π_k-weighted); μ-plug printed only as an estimator
diagnostic (密度泛函硬规则: 不是读出). Off-manifold density mass contributes
~0 to P(s_true) — honestly penalized, unlike snap renormalization which
drops it.

Per-arm training-aligned streams (docs/38 铁律 1, 用户指示 #2):
  word arms: pkl units -> VerifyResplitter (same AE+thresholds as training)
  bpe arm:   same raw texts re-segmented pure_bpe (its native stream, no
             verify existed then)
  + a noverify word arm row to measure the alignment's effect.

Run: python pushforward_bpb.py
"""
from __future__ import annotations

import pickle
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, r"E:\Trash_things\char_lm")
sys.path.insert(0, r"E:\Trash_things\char_lm\scripts")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from bltz.config import Cfg
from bltz.models.autoencoder import EOS_ID, ByteStringAE
from bltz.models.model_v2 import BltzLMv2
from bltz.segment_v2 import segment_v2
from bltz.verify import VerifyResplitter
from train_ae import tensorize

DEV = "cuda"
M_PER_COMP = 8          # stratified samples per GMM component
PKL = r"data\fineweb_word_local\eval_sample.pkl"
TOK = r"data\qwen35_tokenizer"
NSEQ = 64

ARMS = [
    ("word-frozen", r"C:\Users\he\AppData\Local\Temp\opencode\v2_word_1b_last.pt",
     r"checkpoints\ae_48_256_word_full\best.pt", "word"),
    ("word-learnin", r"C:\Users\he\AppData\Local\Temp\opencode\v2_word_1b_learnin_last.pt",
     r"checkpoints\ae_48_256_word_full\best.pt", "word"),
    ("word-frozen-noverify", r"C:\Users\he\AppData\Local\Temp\opencode\v2_word_1b_last.pt",
     r"checkpoints\ae_48_256_word_full\best.pt", "word-noverify"),
    ("bpe-60k", r"C:\Users\he\AppData\Local\Temp\opencode\v2_bpe_60k_last.pt",
     r"C:\Users\he\AppData\Local\Temp\opencode\ae48_typo_full_best.pt", "bpe"),
]


def load_arm(v2_path, ae_path):
    ae_state = torch.load(ae_path, map_location="cpu", weights_only=False)
    ae = ByteStringAE(Cfg(ae_state["cfg"]))
    ae.load_state_dict(ae_state["model"])
    st = torch.load(v2_path, map_location="cpu", weights_only=False)
    model = BltzLMv2(Cfg(st["cfg"]), ae).to(DEV).eval()
    model.load_state_dict(st["model"])
    return model, st["step"]


@torch.no_grad()
def dec_eval(ae, lam_pr: torch.Tensor, tgt: torch.Tensor, valid: torch.Tensor,
             l_max: int, comp_w: torch.Tensor, m_per_comp: int):
    """Two reads of the same decoder queries, lam_pr (P,R,D) rows grouped as
    (m samples × K comps) per position; comp_w (P,K) = softmax(π):
      joint:    log P_dec(true|λ) per (P,R) -> exact-string pushforward
      marginal: P_byte(b|ctx,Δ) = Σ_k π_k mean_m P_dec(b|λ,Δ) -> byte-marginal
                pushforward (2026-09-30 v2: the joint read is degenerate for
                parallel FiLM decoders — exact-string joints multiply marginal
                leaks and floor out at ~e-40 for both arms; the byte marginal
                is the honest byte-LM price: variant leakage shows up as
                byte-level entropy instead of multiplicative collapse).
    Returns (joint_acc (P,R) f64, marg_bytes (P,) f64, marg_eos (P,) f64,
             marg_by_d list f64).
    v3 (2026-09-30 用户): drop EOS from the metric entirely — pure per-byte CE
    over the real sample; EOS pricing was a segmentation-identity tax (short-
    unit arms pay it more often per byte). Kept as a separate accumulator for
    diagnosis."""
    P, R, D = lam_pr.shape
    lam_flat = lam_pr.reshape(P * R, D).float()
    joint = torch.zeros(P, R, dtype=torch.float64, device=lam_pr.device)
    marg_b = torch.zeros(P, dtype=torch.float64, device=lam_pr.device)
    marg_e = torch.zeros(P, dtype=torch.float64, device=lam_pr.device)
    by_d = torch.zeros(l_max + 1, dtype=torch.float64, device=lam_pr.device)
    by_d_n = torch.zeros(l_max + 1, dtype=torch.float64, device=lam_pr.device)
    ar = torch.arange(P * R, device=lam_pr.device)
    K = comp_w.shape[1]
    for d in range(l_max + 1):
        vmask = valid[:, d]
        if not bool(vmask.any()):
            continue
        d_idx = torch.full((P * R,), d, dtype=torch.long, device=lam_pr.device)
        logits = ae.decoder(lam_flat, d_idx)
        logp = F.log_softmax(logits.float(), -1)
        tg = tgt[:, d].clamp(min=0).repeat_interleave(R)
        lp_d = logp[ar, tg].view(P, R).double()
        joint += torch.where(vmask.unsqueeze(1), lp_d, torch.zeros_like(lp_d))
        # byte marginal: softmax probs, mean over m, π-weighted over K
        pb = logp.exp().view(P, m_per_comp, K, 257).mean(1)          # (P,K,257)
        pm = (comp_w.unsqueeze(-1).double() * pb.double()).sum(1)    # (P,257)
        nll_d = -torch.log(pm[torch.arange(P, device=lam_pr.device),
                                tgt[:, d].clamp(min=0)].clamp(min=1e-45))
        is_eos = (tgt[:, d] == EOS_ID)
        is_byte = vmask & ~is_eos
        marg_b += torch.where(is_byte, nll_d, torch.zeros_like(nll_d))
        marg_e += torch.where(is_eos, nll_d, torch.zeros_like(nll_d))
        by_d[d] += torch.where(is_byte, nll_d, torch.zeros_like(nll_d)).sum()
        by_d_n[d] += is_byte.sum()
    return joint, marg_b, marg_e, by_d, by_d_n


@torch.no_grad()
def eval_arm(model, seqs, tag):
    rng = torch.Generator(device=DEV).manual_seed(20260930)
    tot_nll = 0.0        # nats (exact-joint pushforward)
    marg_nll = 0.0       # nats, BYTES ONLY (v3 headline — no EOS)
    marg_eos_nll = 0.0   # nats, EOS positions (diagnostic, not in headline)
    by_d_sum = None
    by_d_cnt = None
    tot_bytes = 0
    n_pos = 0
    n_floor = 0          # P_hat underflow (< 1e-40) positions
    mc_gap = []          # half-sample |Δnll| per position (MC noise gauge)
    plug_nll = 0.0
    t0 = time.time()
    Lmax = model.l_max
    for si, units in enumerate(seqs):
        bids, lens, pm = tensorize(units, Lmax, DEV)
        h = model.backbone(model.backbone_input(bids.unsqueeze(0), pm.unsqueeze(0)))
        lp, mu, sig = (t.float() for t in model.head.params(h))
        K = lp.shape[-1]
        D = mu.shape[-1]
        positions = list(range(1, len(units) - 1))
        true_units = [units[i][:Lmax] for i in positions]
        true_units = [u for u in true_units if u]
        positions = [i for i, u in zip(positions, [units[i][:Lmax] for i in positions]) if u]
        P = len(positions)
        # targets (P, Lmax+1): bytes at d<L, EOS at d=L
        tgt = torch.full((P, Lmax + 1), -1, dtype=torch.long, device=DEV)
        for j, tb in enumerate(true_units):
            L = len(tb)
            tgt[j, :L] = torch.tensor(list(tb), device=DEV)
            tgt[j, L] = EOS_ID
        valid = tgt >= 0
        # stratified λ samples: eps (P, m, K, D)
        eps = torch.randn(P, M_PER_COMP, K, D, generator=rng, device=DEV)
        lam_s = mu[0, positions].unsqueeze(1) + \
            sig[0, positions].unsqueeze(1) * eps          # (P, m, K, D)
        mK = M_PER_COMP * K
        probs = torch.softmax(lp[0, positions].double(), -1)  # (P, K)
        joint, marg_b, marg_e, by_d, by_d_n = dec_eval(
            model.ae, lam_s.reshape(P, mK, D), tgt, valid, Lmax, probs, M_PER_COMP)
        if by_d_sum is None:
            by_d_sum = torch.zeros(Lmax + 1, dtype=torch.float64)
            by_d_cnt = torch.zeros(Lmax + 1, dtype=torch.float64)
        by_d_sum += by_d.cpu()
        by_d_cnt += by_d_n.cpu()
        lp_s = joint.view(P, M_PER_COMP, K)
        lme = torch.logsumexp(lp_s, dim=1) - np.log(M_PER_COMP)   # (P, K)
        p_hat = (probs * lme.exp()).sum(-1)                        # (P,)  exact-joint
        # half-sample MC gap (joint read)
        lme_a = torch.logsumexp(lp_s[:, : M_PER_COMP // 2], 1) - np.log(M_PER_COMP // 2)
        lme_b = torch.logsumexp(lp_s[:, M_PER_COMP // 2:], 1) - np.log(M_PER_COMP // 2)
        pa = (probs * lme_a.exp()).sum(-1)
        pb = (probs * lme_b.exp()).sum(-1)
        okm = (pa > 0) & (pb > 0)
        mc_gap.extend((torch.log(pa[okm]) - torch.log(pb[okm])).abs().cpu().tolist())
        nll = -torch.log(p_hat.clamp(min=1e-45))
        n_floor += int((p_hat <= 1e-40).sum())
        tot_nll += float(nll.sum())
        marg_nll += float(marg_b.sum())
        marg_eos_nll += float(marg_e.sum())
        tot_bytes += sum(len(u) for u in true_units)
        n_pos += P
        print(f"  [{tag}] seq {si+1}/{len(seqs)} "
              f"({n_pos / max(time.time()-t0, 1e-9):.0f} pos/s)", flush=True)
    log2e = float(np.log2(np.e))
    bpb = tot_nll * log2e / max(tot_bytes, 1)
    bpb_m = marg_nll * log2e / max(tot_bytes, 1)
    print(f"[{tag}] byteCE bpb {bpb_m:.4f} (no EOS) | byteCE+EOS "
          f"{(marg_nll + marg_eos_nll) * log2e / max(tot_bytes, 1):.4f} | "
          f"joint bpb {bpb:.4f} | nats/unit {tot_nll / n_pos:.3f} | pos {n_pos} "
          f"bytes {tot_bytes} | floored {n_floor} ({n_floor / n_pos:.3%}) | "
          f"MC med|Δnll| {float(np.median(mc_gap)):.4f}", flush=True)
    if by_d_sum is not None:
        prof = (by_d_sum / by_d_cnt.clamp(min=1))[:12].tolist()
        print(f"[{tag}] per-Δ byte nats: "
              + " ".join(f"{v:.2f}" for v in prof), flush=True)
    return bpb_m


def main() -> None:
    src = pickle.load(open(PKL, "rb"))
    raw_texts = [b"".join(u for u in seq) for seq in src["distinct_seq_units"][:NSEQ]]
    word_seqs_raw = src["distinct_seq_units"][:NSEQ]

    for tag, v2p, aep, mode in ARMS:
        model, step = load_arm(v2p, aep)
        print(f"\n===== {tag} (v2 step {step}) =====", flush=True)
        if mode == "bpe":
            rng = random.Random(7)
            seqs = [segment_v2(t, rng, TOK, l_max=32, pure_bpe=True)
                    for t in raw_texts]
        else:
            seqs = [list(s) for s in word_seqs_raw]
            if mode == "word":
                ver = VerifyResplitter(model.ae, TOK, ent_thresh=0.2,
                                       floor_bytes=3, device=DEV, log_every=0)
                seqs = ver.process_batch(seqs)  # training-aligned stream
                print(f"  [{tag}] verify+resplit: flag {ver.n_flag}/"
                      f"{ver.n_in} ({ver.n_flag / max(ver.n_in, 1):.3%}), "
                      f"in->out x{ver.n_out / max(ver.n_in, 1):.4f}", flush=True)
        eval_arm(model, seqs, tag)
        del model
        torch.cuda.empty_cache()
    print("[pushfwd] DONE", flush=True)


if __name__ == "__main__":
    main()

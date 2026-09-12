"""D-1 diagnostic (v2): per-delta accuracy & entropy curves on held-out
sequences, stratified by patch offset k and within-patch position r
(docs/01-design.md §9 — hard acceptance item).

Why stratified (v2, found 2026-09-08 on the smoke ckpt): our word+delimiter
segmentation makes H(delta) SAWTOOTH — delta=1 is always a patch's first byte
(usually a new word's first letter, high entropy; this is exactly BLT's
word-start entropy concentration), while mid-word deltas are low entropy.
Raw per-delta step monotonicity therefore false-alarms. The real signals:
  [2] per-k cross-patch trend — predictive power must degrade as the target
      patch gets further (THE cheat detector: flat/increasing = head is not
      forecasting = static-fan analog);
  [1] per-delta H range — if ~0, the head ignores delta entirely (static fan);
  [3] per-r within-patch profile — H should peak at r=0 and drop (word-start
      concentration = direct measurement of hypothesis H1).

Usage: python scripts/diag_delta.py [ckpt.pt] [configs/xxx.yaml] [--set a.b=value]
Data source: the on-disk cache if present (last --set diag.n_seq sequences,
held out from training), otherwise a fresh mini stream of FineWeb-Edu docs.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn.functional as F

from bltz.config import Cfg, parse_cli
from bltz.data import build_batch, stream_fineweb_texts
from bltz.models import BltzLM
from bltz.objectives import mtp_targets
from bltz.segment import Segmenter
from bltz.shards import ShardReader


def main() -> None:
    ckpt = None
    argv = []
    for a in sys.argv[1:]:
        if a.endswith(".pt"):
            ckpt = a
        else:
            argv.append(a)
    sys.argv = [sys.argv[0]] + argv
    cfg = parse_cli("configs/default.yaml")
    ckpt = ckpt or os.path.join(cfg.train.ckpt_dir, "best.pt")
    d = cfg.get("diag", None)
    n_seq = int(getattr(d, "n_seq", 64)) if d else 64
    n_batch = int(getattr(d, "n_batch", 8)) if d else 8
    k_show = int(getattr(d, "k_show", 16)) if d else 16

    print(f"ckpt: {ckpt}")
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    mcfg = Cfg(state["cfg"])  # build the model from the ckpt's own config
    model = BltzLM(mcfg).cuda()
    model.load_state_dict(state["model"])
    model.eval()
    print(f"model loaded (step {state.get('step')}, loss {state.get('loss')})")

    batches = _held_out_batches(cfg, mcfg, n_seq, n_batch)

    n = mcfg.loss.n_patches_ahead
    lam = float(mcfg.loss.lam)
    k_max = mcfg.model.k_max
    k_show = min(k_show, k_max)

    l_max = mcfg.segment.l_max
    acc_d = torch.zeros(k_show)
    ent_d = torch.zeros(k_show)
    cnt_d = torch.zeros(k_show)
    acc_k = torch.zeros(n)
    ent_k = torch.zeros(n)
    cnt_k = torch.zeros(n)
    acc_r = torch.zeros(l_max)
    ent_r = torch.zeros(l_max)
    cnt_r = torch.zeros(l_max)

    with torch.no_grad():
        for batch in batches:
            batch = {k: v.cuda() for k, v in batch.items()}
            B, S, _ = batch["byte_ids"].shape
            h = model(batch["byte_ids"], batch["pad_mask"])[:, : S - 1]
            q = mtp_targets(batch, n, lam, k_max)
            for di in range(k_show):
                valid = q["valid"][:, :, di]
                if not valid.any():
                    continue
                h_q = h[:, : S - 1][valid]
                logits = model.head(h_q, torch.full((h_q.shape[0],), di, device="cuda"))
                logp = F.log_softmax(logits.float(), dim=-1)
                p = logp.exp()
                correct = (logp.argmax(-1) == q["targets"][:, :, di][valid]).float()
                H = -(p * logp).sum(-1)
                acc_d[di] += correct.sum().cpu()
                ent_d[di] += H.sum().cpu()
                cnt_d[di] += valid.sum().cpu()
                kk = (q["k"][:, :, di][valid] - 1).cpu()
                rr = q["r"][:, :, di][valid].cpu()
                acc_k += torch.bincount(kk, weights=correct.cpu(), minlength=n)
                ent_k += torch.bincount(kk, weights=H.cpu(), minlength=n)
                cnt_k += torch.bincount(kk, minlength=n)
                acc_r += torch.bincount(rr, weights=correct.cpu(), minlength=l_max)
                ent_r += torch.bincount(rr, weights=H.cpu(), minlength=l_max)
                cnt_r += torch.bincount(rr, minlength=l_max)

    a_d = acc_d / cnt_d.clamp_min(1)
    e_d = ent_d / cnt_d.clamp_min(1)
    a_k = acc_k / cnt_k.clamp_min(1)
    e_k = ent_k / cnt_k.clamp_min(1)
    a_r = acc_r / cnt_r.clamp_min(1)
    e_r = ent_r / cnt_r.clamp_min(1)

    print(f"\n[1] raw per-delta curves ({int(cnt_d.sum())} queries, n={n}, lam={lam}):")
    print(f"{'delta':>5} {'acc':>8} {'H(nats)':>8} {'count':>8}")
    for di in range(k_show):
        print(f"{di+1:>5} {a_d[di]:>8.4f} {e_d[di]:>8.3f} {int(cnt_d[di]):>8}")

    print("\n[2] per-k cross-patch trend (k = target in the k-th future patch):")
    print(f"{'k':>3} {'acc':>8} {'H(nats)':>8} {'count':>8}")
    for ki in range(n):
        print(f"{ki+1:>3} {a_k[ki]:>8.4f} {e_k[ki]:>8.3f} {int(cnt_k[ki]):>8}")

    print("\n[3] per-r within-patch profile (r = byte position inside target patch):")
    print(f"{'r':>3} {'acc':>8} {'H(nats)':>8} {'count':>8}")
    for ri in range(l_max):
        if cnt_r[ri] > 0:
            print(f"{ri:>3} {a_r[ri]:>8.4f} {e_r[ri]:>8.3f} {int(cnt_r[ri]):>8}")

    tol = 0.005
    k_acc_viol = int(((a_k[1:] > a_k[:-1] + tol) & (cnt_k[1:] > 0)).sum())
    k_ent_viol = int(((e_k[1:] < e_k[:-1] - tol) & (cnt_k[1:] > 0)).sum())
    h_range = float(e_d[cnt_d > 0].max() - e_d[cnt_d > 0].min())
    r0_vs_rest = float(e_r[0] - e_r[1:][cnt_r[1:] > 0].mean()) if (cnt_r[1:] > 0).any() else 0.0
    print("\nD-1 v2 verdict components:")
    print(f"  cross-patch trend (k): acc rises {k_acc_viol}/{n-1} transitions, H drops {k_ent_viol}/{n-1} (tol={tol})")
    print(f"  delta-usage: per-delta H range = {h_range:.3f} nats (< 0.05 = head ignores delta = static fan)")
    print(f"  within-patch (r): H(r=0) {e_r[0]:.3f}, mean H(r>0) {float(e_r[1:][cnt_r[1:] > 0].mean()) if (cnt_r[1:] > 0).any() else 0.0:.3f}, excess {r0_vs_rest:+.3f} (word-start concentration = H1 evidence)")
    if h_range < 0.05:
        print("VERDICT: FAIL — head entropy flat across delta (static-fan cheat signature)")
    elif k_acc_viol > 0:
        print("VERDICT: WARN — accuracy does not degrade across future patches "
              "(acceptable only at smoke scale; must be 0 at Stage-1 scale)")
    else:
        print("VERDICT: OK — cross-patch degradation + delta-usage confirmed (D-1)")


def _held_out_batches(cfg, mcfg, n_seq: int, n_batch: int):
    S = mcfg.data.n_patches
    try:
        reader = ShardReader(cfg.data.cache_dir)
        total = reader.n_sequences(S)
        idx = list(range(total - n_seq, total))  # tail = held out
        print(f"data: cache {cfg.data.cache_dir} ({total} sequences, using last {n_seq})")
        return [reader.make_batch(idx[i : i + n_batch], S) for i in range(0, n_seq, n_batch)]
    except (FileNotFoundError, NotADirectoryError):
        seg = Segmenter(mcfg.segment.l_max, 0.0, 0.0)
        texts = list(stream_fineweb_texts(200))
        print(f"data: no cache found; streamed {len(texts)} docs from FineWeb-Edu")
        batch = build_batch(texts, seg, S, mcfg.segment.l_max)
        n = batch["byte_ids"].shape[0]
        idx = torch.arange(max(0, n - n_seq), n)
        batch = {k: v[idx] for k, v in batch.items()}
        return [{k: v[i : i + n_batch] for k, v in batch.items()} for i in range(0, n_seq, n_batch)]


if __name__ == "__main__":
    main()

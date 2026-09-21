"""Vocab-snap decode for v2 (2026-09-22, user idea, validated):

Table V = Qwen3.5 vocab (as byte strings) U top-N frequent cache units,
encoded once by the frozen AE -> legal-word embedding matrix.

Readouts on eval sequences:
  dec  : argmax-pi component mean -> AE decoder (baseline)
  cos  : argmax_n cos(mu_hat, V_n)
  gmm  : argmax_n sum_k pi_k * N(V_n; mu_k, sig_k)   (density-quantized —
         the principled one: a proper WORD-LEVEL categorical distribution,
         sigma voting included; temperature/top-k/top-p all well-defined)

Usage:
  python scripts/snap_decode.py <v2_ckpt> <ae_ckpt> [--units-pkl pkl]
         [--cache data/fineweb_v2_local] [--topn 100000] [--gen]
"""
from __future__ import annotations

import pickle
import random
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
import torch.nn.functional as F

from bltz.config import Cfg
from bltz.models.autoencoder import ByteStringAE
from bltz.models.model_v2 import BltzLMv2
from bltz.segment_v2 import _tokenizer
from diag_v2 import lev
from train_ae import tensorize

DEV = "cuda"


def build_table(ae: ByteStringAE, tok_dir: str, cache_units: list[bytes], topn: int) -> tuple[list[bytes], torch.Tensor, torch.Tensor, torch.Tensor]:
    tok = _tokenizer(tok_dir)
    qwen_strs = []
    nv = tok.get_vocab_size()
    t0 = time.time()
    for i in range(nv):
        s = tok.decode([i])
        if s and 0 < len(s.encode("utf-8")) <= ae.l_max:
            qwen_strs.append(s.encode("utf-8"))
        if i % 40000 == 0 and i:
            print(f"  [heartbeat] vocab {i}/{nv}", flush=True)
    freq = Counter(cache_units)
    cache_top = [u for u, _ in freq.most_common(topn)]
    table = list(dict.fromkeys(qwen_strs + cache_top))
    print(f"[snap] table: {len(qwen_strs)} qwen + {len(cache_top)} cache-top -> {len(table)}", flush=True)
    lats = []
    with torch.no_grad():
        for i in range(0, len(table), 4096):
            b, l, mk = tensorize(table[i : i + 4096], ae.l_max, DEV)
            lats.append(ae.encoder(b, mk).float())
        if len(table) > 40960:
            print("  [heartbeat] table encoded", flush=True)
    V = torch.cat(lats)
    return table, V, F.normalize(V, dim=-1), V.pow(2)


def gmm_scores(logit_pi, mu, sig, V, V2):
    """word-level log-weights: log sum_k pi_k N(V_n; mu_k, sig_k), (N,) fp32."""
    D = mu.shape[-1]
    inv = 1.0 / sig.pow(2)
    log_n = -0.5 * (V2 @ inv.T - 2.0 * (V @ (mu * inv).T)
                    + (mu.pow(2) * inv).sum(-1, keepdim=True).T) \
        - sig.log().sum(-1, keepdim=True).T - 0.5 * D * np.log(2 * np.pi)
    return torch.logsumexp(F.log_softmax(logit_pi, -1).unsqueeze(0) + log_n, dim=1)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    v2_path, ae_path = sys.argv[1], sys.argv[2]
    pkl = sys.argv[sys.argv.index("--units-pkl") + 1] if "--units-pkl" in sys.argv else None
    topn = int(sys.argv[sys.argv.index("--topn") + 1]) if "--topn" in sys.argv else 100_000
    do_gen = "--gen" in sys.argv

    ae_state = torch.load(ae_path, map_location="cpu", weights_only=False)
    ae = ByteStringAE(Cfg(ae_state["cfg"]))
    ae.load_state_dict(ae_state["model"])
    v2_state = torch.load(v2_path, map_location="cpu", weights_only=False)
    model = BltzLMv2(Cfg(v2_state["cfg"]), ae).to(DEV).eval()
    model.load_state_dict(v2_state["model"])
    src = pickle.load(open(pkl, "rb")) if pkl else None
    cache_units = src["freq_units"] if src else []
    table, V, Vn, V2 = build_table(model.ae, "data/qwen35_tokenizer", cache_units, topn)
    table_set = set(table)

    seqs = src["distinct_seq_units"][:16]
    em = {"dec": 0, "cos": 0, "gmm": 0}
    ed2 = {"dec": 0, "cos": 0, "gmm": 0}
    cov = 0
    tot = 0
    t0 = time.time()
    with torch.no_grad():
        for si, units in enumerate(seqs):
            bids, lens, pm = tensorize(units, model.l_max, DEV)
            lam = model.encode_units(bids.unsqueeze(0), pm.unsqueeze(0))
            S, D = lam.shape[1], lam.shape[2]
            x = torch.cat([model.bos.expand(1, 1, D), lam[:, :-1]], dim=1)
            h = model.backbone(model.adapter(x))
            logit_pi, mu, sig = (t.float() for t in model.head.params(h))
            for i in range(1, S - 1):
                true_b = units[i][: model.l_max]
                tot += 1
                cov += int(true_b in table_set)
                k = int(logit_pi[0, i].argmax())
                dec = bytes(model.ae.decode(mu[0, i, k].reshape(1, -1))[0])
                cos_s = table[int((F.normalize(mu[0, i, k].reshape(1, -1), dim=-1) @ Vn.T).argmax())]
                gmm_s = table[int(gmm_scores(logit_pi[0, i], mu[0, i], sig[0, i], V, V2).argmax())]
                for name, pred in (("dec", dec), ("cos", cos_s), ("gmm", gmm_s)):
                    e = lev(pred, true_b)
                    em[name] += int(e == 0)
                    ed2[name] += int(e <= 2)
            print(f"  [heartbeat] eval seq {si + 1}/{len(seqs)} "
                  f"({tot / max(time.time() - t0, 1e-9):.0f} pos/s)", flush=True)
    print(f"\ncoverage {cov / tot:.4f} | EM: dec {em['dec'] / tot:.4f} cos {em['cos'] / tot:.4f} "
          f"gmm {em['gmm'] / tot:.4f} | edit<=2: dec {ed2['dec'] / tot:.4f} cos {ed2['cos'] / tot:.4f} "
          f"gmm {ed2['gmm'] / tot:.4f}", flush=True)

    if do_gen:
        n_prompts = 4
        taus = (0.0, 0.6, 0.8, 1.0, 1.2)
        for units in seqs[:n_prompts]:
            bids, lens, pm = tensorize(units[:16], model.l_max, DEV)
            base = torch.cat([model.bos.expand(1, 1, -1),
                              model.encode_units(bids.unsqueeze(0), pm.unsqueeze(0))], dim=1)
            print(f"\n===== prompt: {b''.join(units[:16])[:80]!r} =====", flush=True)
            for tau in taus:
                torch.manual_seed(0)
                outs = []
                cur_t = base.clone()
                with torch.no_grad():
                    for _ in range(48):
                        h = model.backbone(model.adapter(cur_t))[:, -1:]
                        logit_pi, mu, sig = (t.float() for t in model.head.params(h))
                        w = gmm_scores(logit_pi[0, 0], mu[0, 0], sig[0, 0], V, V2)
                        j = int(w.argmax()) if tau <= 0 else int(torch.multinomial(F.softmax(w / tau, -1), 1))
                        bs = table[j]
                        outs.append(bs)
                        nb, _, npm = tensorize([bs], model.l_max, DEV)
                        cur_t = torch.cat([cur_t, model.encode_units(nb.unsqueeze(0), npm.unsqueeze(0))], dim=1)
                print(f"--- tau={tau} ---\n"
                      + b"".join(outs).decode("utf-8", errors="replace")[:400], flush=True)
    print("\n[snap] DONE", flush=True)


if __name__ == "__main__":
    main()

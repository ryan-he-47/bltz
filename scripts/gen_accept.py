"""Extended generation battery for v2 acceptance (docs/38 §5; 8 prompts x 6 readouts).

Reconstruction (2026-10-26) of the ad-hoc gen_long script used for the p1/p2
acceptance logs (§5, never committed). Output format kept identical so the CPT
S-tier acceptance log is comparable with the 0.5B milestone logs.

Readouts (all density-functional — AGENTS.md hard rule; NEVER component means):
  mode            : global MAP of the GMM (head.mode -> AE decode)
  modes-peak T    : enumerate local modes (head.modes), peak-altitude weighting
                    log_dens/T (docs/39 #36) + nucleus 0.95 sampling
  snap argmax     : argmax over the protocol table (top-500k cache units) of
                    log sum_k pi_k N(V_n; mu_k, sig_k)
  snap t=0.8/1.0  : temperature sampling over the same table distribution

Protocol table = top-500k frequent units scanned from the OLD word cache
(data/fineweb_word_local, 188.6M units) — the exact table of the p2 acceptance
(word arm: no qwen; Qwen BPE vocab is OOD for the word AE).

  python scripts/gen_accept.py <v2_ckpt> <ae_ckpt> [--cache DIR]
         [--steps 48] [--topn 500000]
"""
from __future__ import annotations

import pickle
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
from bltz.segment_v2 import segment_word
from bltz.shards import ShardReader
from bltz.verify import VerifyResplitter
from snap_decode import build_table, gmm_scores
from train_ae import tensorize

DEV = "cuda"
TAUS_PEAK = (0.6, 1.0)
TAUS_SNAP = (0.0, 0.8, 1.0)
TOK = "data/qwen35_tokenizer"
CLASSIC: list[bytes] = [
    b"The history of artificial intelligence began",
    b"Scientists at the university have discovered a new",
    b"Once upon a time, in a small village by the sea,",
    b"The stock market closed higher on Monday after",
]


def load_model(v2_path: str, ae_path: str):
    ae_state = torch.load(ae_path, map_location="cpu", weights_only=False)
    ae = ByteStringAE(Cfg(ae_state["cfg"]))
    ae.load_state_dict(ae_state["model"])
    v2_state = torch.load(v2_path, map_location="cpu", weights_only=False)
    model = BltzLMv2(Cfg(v2_state["cfg"]), ae).to(DEV).eval()
    model.load_state_dict(v2_state["model"])
    print(f"[gen_accept] ckpt={v2_path} step={v2_state.get('step')} "
          f"l_max={model.l_max}", flush=True)
    return model, ae


def build_protocol_table(model, cache_dir: str, topn: int, table_cache: str | None = None):
    """Top-`topn` cache units by frequency -> density table (no qwen)."""
    if table_cache and Path(table_cache).exists():
        obj = torch.load(table_cache, map_location=DEV, weights_only=False)
        print(f"[gen_accept] table loaded from {table_cache} "
              f"({len(obj['table'])} entries)", flush=True)
        return obj["table"], obj["V"], obj["V2"]
    t0 = time.time()
    reader = ShardReader(cache_dir)
    sh = reader.shards[0]
    n = int(sh["meta"]["n_units"])
    ul = np.asarray(sh["unit_len"], dtype=np.int64)
    fl = np.asarray(sh["bytes"], dtype=np.uint8)
    freq: Counter = Counter()
    pos = 0
    for s in range(0, n, 4_000_000):
        e = min(s + 4_000_000, n)
        lens = ul[s:e]
        buf = fl[pos:pos + int(lens.sum())].tobytes()
        p = 0
        for L in lens:
            freq[buf[p:p + L]] += 1
            p += L
        pos += int(lens.sum())
        print(f"  [gen] table scan {e}/{n} "
              f"({e / max(time.time() - t0, 1e-9) / 1e6:.2f}M/s)", flush=True)
    cache_units = [u for u, _ in freq.most_common(topn)]
    table, V, _Vn, V2 = build_table(model.ae, TOK, cache_units, topn, use_qwen=False)
    if table_cache:
        torch.save({"table": table, "V": V, "V2": V2}, table_cache)
        print(f"[gen_accept] table cached to {table_cache}", flush=True)
    return table, V, V2


def segment_prompt(model, ae, pb: bytes) -> list[bytes]:
    """Training-aligned segmentation of a raw prompt.

    MUST be segment_word (word arm: TRAILING-space units, e.g. b'them ') —
    the legacy segment_v2 BPE style (LEADING space, b' them') is OOD for the
    word AE (docs/38 §0 OOD 铁律; AE decodes b' village' as b'Ovillage').
    """
    units = segment_word(pb, TOK, l_max=model.l_max)
    ver = VerifyResplitter(ae, TOK, ent_thresh=0.2, floor_bytes=3,
                           device=DEV, log_every=0)
    return ver.process(units)


@torch.no_grad()
def gen_readout(model, table, V, V2, prompt_units: list[bytes], kind: str,
                tau: float, steps: int) -> str:
    cur = list(prompt_units)
    outs: list[bytes] = []
    for gstep in range(steps):
        bids, _, pm = tensorize(cur, model.l_max, DEV)
        lam_u = model.encode_units(bids.unsqueeze(0), pm.unsqueeze(0))
        x = torch.cat([model.bos.expand(1, 1, -1), lam_u], dim=1)
        h = model.backbone(model.adapter(x))[:, -1:]
        if not torch.isfinite(h).all():  # docs/38 数值守卫: 发散即停该轨迹
            print(f"  [gen] non-finite h at step {gstep}; stop trajectory", flush=True)
            break
        if kind == "mode":
            mv = model.head.mode(h[:, 0:1])[0, 0]
            if not torch.isfinite(mv).all():
                print(f"  [gen] non-finite mode at step {gstep}; stop", flush=True)
                break
            dec = bytes(model.ae.decode(mv.reshape(1, -1))[0])
        elif kind == "modes-peak":
            modes, _log_mass, log_dens = model.head.modes(h[0, 0:])
            if not torch.isfinite(log_dens).all():
                print(f"  [gen] non-finite log_dens at step {gstep}; stop", flush=True)
                break
            p = F.softmax(log_dens / max(tau, 1e-6), dim=-1)
            ps, idx = p.sort(descending=True)
            keep = int(torch.searchsorted(torch.cumsum(ps, 0),
                                          torch.tensor(0.95, device=DEV),
                                          right=True).clamp(min=1))
            top_p = ps[:keep] / ps[:keep].sum()
            j = int(idx[torch.multinomial(top_p, 1)])
            dec = bytes(model.ae.decode(modes[j: j + 1])[0])
        else:  # snap
            logit_pi, mu, sig = (t.float() for t in model.head.params(h))
            if not torch.isfinite(logit_pi).all():
                print(f"  [gen] non-finite logit_pi at step {gstep}; stop", flush=True)
                break
            w = gmm_scores(logit_pi[0, 0], mu[0, 0], sig[0, 0], V, V2)
            j = int(w.argmax()) if tau <= 0 else int(
                torch.multinomial(F.softmax(w / tau, -1), 1))
            dec = table[j]
        if not dec:
            break
        outs.append(dec)
        cur.append(dec)
    return b"".join(outs).decode("utf-8", errors="replace")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    v2_path, ae_path = sys.argv[1], sys.argv[2]
    cache_dir = (sys.argv[sys.argv.index("--cache") + 1]
                 if "--cache" in sys.argv else "data/fineweb_word_local")
    steps = int(sys.argv[sys.argv.index("--steps") + 1]) if "--steps" in sys.argv else 48
    topn = int(sys.argv[sys.argv.index("--topn") + 1]) if "--topn" in sys.argv else 500_000
    pkl = (sys.argv[sys.argv.index("--units-pkl") + 1]
           if "--units-pkl" in sys.argv else "data/fineweb_word_local/eval_sample.pkl")
    table_cache = (sys.argv[sys.argv.index("--table-cache") + 1]
                   if "--table-cache" in sys.argv else None)

    model, ae = load_model(v2_path, ae_path)
    table, V, V2 = build_protocol_table(model, cache_dir, topn, table_cache)

    # prompts: first 4 = eval-sequence snippets (i.e. the p1/p2 §5 prompts 0-3),
    # prompts 4-7 = the classic openers of the same battery.
    src = pickle.load(open(pkl, "rb"))
    prompts: list[bytes] = [b"".join(seq[:16]) for seq in src["distinct_seq_units"][:4]]
    prompts += CLASSIC
    prompt_units = [list(seq[:16]) for seq in src["distinct_seq_units"][:4]]
    prompt_units += [segment_prompt(model, ae, pb) for pb in CLASSIC]

    kinds = [("mode", 0.0), ("modes-peak", TAUS_PEAK[0]), ("modes-peak", TAUS_PEAK[1]),
             ("snap", TAUS_SNAP[0]), ("snap", TAUS_SNAP[1]), ("snap", TAUS_SNAP[2])]
    for i, pb in enumerate(prompts):
        print(f"\n==================== prompt {i}: {pb[:80]!r} ==========", flush=True)
        for kind, tau in kinds:
            torch.manual_seed(0)
            txt = gen_readout(model, table, V, V2, prompt_units[i], kind, tau, steps)
            label = ("mode" if kind == "mode" else
                     f"{kind} T={tau}" if kind == "modes-peak" else
                     "snap argmax" if tau == 0.0 else f"snap t={tau}")
            print(f"--- {label} ---\n{txt[:1200]}", flush=True)
    print("\n[gen_accept] DONE", flush=True)


if __name__ == "__main__":
    main()

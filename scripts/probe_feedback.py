"""Autoregressive feedback probe (2026-09-28, user question):

The generator always worked at 34% one-step EM — what kills long generations
is the FEEDBACK channel: decoded typos re-encoded feed garbage embeddings back
(emb->decode->emb sandwich). Now that the input side can take embeddings
directly, test 5 selection schemes x 2 feedback modes on free-run generation:

  schemes (picks a 48-d lambda-space vector at each step):
    sample    : k ~ Categorical(pi), input mu_k
    argmax_pi : input mu_{argmax pi}                 (greedy)
    min_sigma : input mu_{argmin mean sigma}         (sigma rerank)
    mode      : input GMM MAP point (mean-shift)
    snap      : input table embedding of gmm-argmax word
  feedback:
    embed     : feed the 48-d vector directly
    reencode  : AE-decode -> bytes -> frozen re-encode (current pipeline)

Metrics vs the true continuation: EM / NED at steps {1,4,16,64}, survival
(fraction of steps decoding to a non-empty string).

  python scripts/probe_feedback.py <v2_ckpt> <ae_ckpt> --units-pkl <pkl>
"""
from __future__ import annotations

import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
import torch.nn.functional as F

from bltz.config import Cfg
from bltz.models.autoencoder import ByteStringAE
from bltz.models.model_v2 import BltzLMv2
from diag_v2 import lev
from snap_decode import build_table, gmm_scores
from train_ae import tensorize

DEV = "cuda"
N_PROMPTS = 32
GEN_STEPS = 64
PROMPT = 16


@torch.no_grad()
def pick(scheme, pi_lp, mu, sig, V, V2, table):
    if scheme == "sample":
        k = int(torch.multinomial(F.softmax(pi_lp, -1), 1))
        return mu[k]
    if scheme == "argmax_pi":
        return mu[int(pi_lp.argmax())]
    if scheme == "min_sigma":
        return mu[int(sig.mean(-1).argmin())]
    if scheme == "mode":
        raise RuntimeError("mode needs head")
    if scheme == "snap":
        j = int(gmm_scores(pi_lp, mu, sig, V, V2).argmax())
        return V[j]
    raise ValueError(scheme)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    v2_path, ae_path = sys.argv[1], sys.argv[2]
    pkl = sys.argv[sys.argv.index("--units-pkl") + 1]
    src = pickle.load(open(pkl, "rb"))
    ae_state = torch.load(ae_path, map_location="cpu", weights_only=False)
    ae = ByteStringAE(Cfg(ae_state["cfg"]))
    ae.load_state_dict(ae_state["model"])
    v2_state = torch.load(v2_path, map_location="cpu", weights_only=False)
    model = BltzLMv2(Cfg(v2_state["cfg"]), ae).to(DEV).eval()
    model.load_state_dict(v2_state["model"])
    print(f"[load] v2 step {v2_state.get('step')}, ||head.out|| "
          f"{model.head.out.weight.norm():.2f}", flush=True)

    table, V, _Vn, V2 = build_table(model.ae, "data/qwen35_tokenizer",
                                    src["freq_units"], 100_000)
    seqs = src["distinct_seq_units"][:N_PROMPTS]

    configs = [(s, f) for s in ("sample", "argmax_pi", "min_sigma", "mode", "snap")
               for f in (("embed", "reencode") if s != "snap" else ("embed",))]
    print(f"{'config':<22} {'EM@1':>6} {'EM@4':>6} {'EM@16':>6} {'EM@64':>6} "
          f"{'NED@64':>7} {'surv':>6}", flush=True)

    for scheme, fb in configs:
        em = np.zeros(GEN_STEPS)
        ned = np.zeros(GEN_STEPS)
        surv = np.zeros(GEN_STEPS)
        t0 = time.time()
        for units in seqs:
            truth = units[PROMPT : PROMPT + GEN_STEPS]
            bids, _, pm = tensorize(units[:PROMPT], model.l_max, DEV)
            lam = model.encode_units(bids.unsqueeze(0), pm.unsqueeze(0))  # (1,P,48)
            past = [lam[0, i] for i in range(PROMPT)]
            for t in range(GEN_STEPS):
                x = torch.cat([model.bos.expand(1, 1, -1),
                               torch.stack(past).unsqueeze(0)], dim=1)
                h = model.backbone(model.adapter(x))[:, -1:]
                pi_lp, mu, sig = (q[0, 0].float() for q in model.head.params(h))
                if scheme == "mode":
                    vec = model.head.mode(h)[0, 0]
                    bs = bytes(model.ae.decode(vec.reshape(1, -1))[0])
                else:
                    vec = pick(scheme, pi_lp, mu, sig, V, V2, table)
                    if scheme == "snap":
                        j = int(gmm_scores(pi_lp, mu, sig, V, V2).argmax())
                        bs = table[j]
                    else:
                        bs = bytes(model.ae.decode(vec.reshape(1, -1))[0])
                e = lev(bs, truth[t])
                em[t] += int(e == 0)
                ned[t] += e / max(len(truth[t]), 1)
                surv[t] += int(len(bs) > 0)
                if fb == "reencode":
                    nb, _, npm = tensorize([bs], model.l_max, DEV)
                    past.append(model.encode_units(nb.unsqueeze(0), npm.unsqueeze(0))[0, 0])
                else:
                    past.append(vec)
        n = len(seqs)
        print(f"{scheme + '/' + fb:<22} {em[0]/n:6.3f} {em[3]/n:6.3f} "
              f"{em[15]/n:6.3f} {em[-1]/n:6.3f} {ned[-1]/n:7.3f} {surv[-1]/n:6.3f}"
              f"   ({time.time()-t0:.0f}s)", flush=True)
    print("[probe] DONE", flush=True)


if __name__ == "__main__":
    main()

"""v2 output-side geometry + sigma diag (2026-09-24, user direction):

1. Backbone OUTPUT h (768d) effective rank / anisotropy — does the shell /
   low-rank signature persist at the output end (position-weighted view; h is
   contextual so no distinct-unit dedup)?
2. Empirical sigma distribution from the MDN head: does sigma hug the floor
   (0.05)? If the floor binds, expressiveness is limited (user hypothesis);
   if sigma floats well above, the floor is dead weight. Report raw histogram
   AND pi-responsibility-weighted percentiles (the sigma that actually carries
   probability mass), plus the quantization margin: typical sigma vs typical
   nearest-neighbor distance in lambda space.

  python scripts/diag_v2_output.py <v2_ckpt> <ae_ckpt> --units-pkl <pkl>
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
from diag_v2_input import stats
from train_ae import tensorize

DEV = "cuda"
N_SEQS = 16
SIGMA_FLOOR = 0.05


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

    hs, lams, sigs, pis = [], [], [], []
    with torch.no_grad():
        for units in src["distinct_seq_units"][:N_SEQS]:
            bids, lens, pm = tensorize(units, model.l_max, DEV)
            lam = model.encode_units(bids.unsqueeze(0), pm.unsqueeze(0))
            S, D = lam.shape[1], lam.shape[2]
            x = torch.cat([model.bos.expand(1, 1, D), lam[:, :-1]], dim=1)
            h = model.backbone(model.adapter(x))
            logit_pi, mu, sig = (t.float() for t in model.head.params(h))
            hs.append(h[0].float().cpu().numpy())
            lams.append(lam[0].float().cpu().numpy())
            sigs.append(sig[0].cpu().numpy())                    # (S, K, 48)
            pis.append(F.softmax(logit_pi[0], -1).cpu().numpy())  # (S, K)

    h = np.concatenate(hs)
    lam = np.concatenate(lams)
    sig = np.concatenate(sigs)          # (N, K, 48)
    pi = np.concatenate(pis)            # (N, K)
    N, K, D = sig.shape
    print(f"positions {N}, K {K}, d {D}", flush=True)

    rng = np.random.default_rng(0)
    stats("backbone output h 768d (position-weighted)", h, rng)

    # sigma distribution: raw over all (pos, k, dim)
    sf = sig.reshape(-1)
    at_floor = float((sf <= SIGMA_FLOOR + 1e-3).mean())
    print(f"[sigma] raw: floor-occupancy {at_floor:.4f} | "
          f"p5 {np.percentile(sf, 5):.4f} p25 {np.percentile(sf, 25):.4f} "
          f"p50 {np.percentile(sf, 50):.4f} p75 {np.percentile(sf, 75):.4f} "
          f"p95 {np.percentile(sf, 95):.4f} max {sf.max():.3f}")
    # responsibility-weighted: the sigma of components that carry mass
    w = (pi[..., None] * np.ones_like(sig)).reshape(-1)
    w = w / w.sum()
    order = np.argsort(sf)
    cw = np.cumsum(w[order])
    q = lambda p: float(sf[order][np.searchsorted(cw, p)])
    print(f"[sigma] pi-weighted: p5 {q(0.05):.4f} p25 {q(0.25):.4f} "
          f"p50 {q(0.50):.4f} p75 {q(0.75):.4f} p95 {q(0.95):.4f}")
    # per-position winning-component sigma (what argmax-pi would use)
    k_star = pi.argmax(-1)
    sig_star = sig[np.arange(N), k_star]          # (N, 48)
    print(f"[sigma] argmax-pi component: mean {sig_star.mean():.4f} "
          f"median {np.median(sig_star):.4f} floor-occ "
          f"{float((sig_star <= SIGMA_FLOOR + 1e-3).mean()):.4f}")

    # quantization margin: typical NN distance in lambda space vs typical sigma
    uniq = list(set(src["freq_units"]))[:20000]
    with torch.no_grad():
        from diag_v2_input import stats as _  # noqa
        lats = []
        for i in range(0, len(uniq), 2048):
            b, l, mk = tensorize(uniq[i : i + 2048], model.l_max, DEV)
            lats.append(model.encode_units(b.unsqueeze(0), mk.unsqueeze(0))[0].float().cpu())
    V = torch.cat(lats).numpy()
    Vn = V / np.linalg.norm(V, axis=1, keepdims=True).clip(1e-9)
    sim = Vn @ Vn.T
    np.fill_diagonal(sim, -2)
    nn_cos = sim.max(1)
    nn_l2 = np.sqrt((2 - 2 * nn_cos).clip(0)) * np.linalg.norm(V, axis=1).mean()
    print(f"[margin] distinct-unit NN: cos {nn_cos.mean():.3f} (p10 {np.percentile(nn_cos, 10):.3f}) "
          f"| approx L2 gap {nn_l2.mean():.3f} vs median pi-weighted sigma {q(0.50):.3f}")
    print("[output] DONE", flush=True)


if __name__ == "__main__":
    main()

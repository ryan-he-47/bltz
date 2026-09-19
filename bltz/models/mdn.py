"""MDN head for v2 (docs/31 §2.3): two-layer nonlinear buffer -> diagonal GMM.

Output target is a STRUCTURED GMM parameter space, not a flat vocab — the
buffer cushions the backbone from it (2026-09-18 拍板, mirrors the input-side
adapter; no control arm). Diagonal covariance only: each component encircles
ONE byte string's local region; cross-dim correlation is absorbed by component
multiplicity (Graves MDN / CoSE precedent). Full covariance at d=32 would
already cost ~27.6M in the output layer alone — rejected.

Shapes: h (..., d_in) -> logit_pi (..., K), mu (..., K, d), sigma (..., K, d).
Small-init final layer + zero bias => initial GMM ~ single wide Gaussian
(pi uniform, mu=0, sigma=1) — adaLN-Zero-flavored stable start.
Inference takes the component MEAN only (μ_k; 2026-09-17 拍板): temperature
acts on pi alone.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MDNHead(nn.Module):
    def __init__(
        self,
        d_in: int = 768,
        hidden: int = 1024,
        n_comp: int = 64,
        d_emb: int = 48,
        sigma_floor: float = 1e-3,
    ):
        super().__init__()
        self.n_comp = n_comp
        self.d_emb = d_emb
        self.sigma_floor = sigma_floor
        self.fc1 = nn.Linear(d_in, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.out = nn.Linear(hidden, n_comp * (1 + 2 * d_emb))
        nn.init.normal_(self.out.weight, std=1e-3)
        nn.init.zeros_(self.out.bias)

    def params(self, h: torch.Tensor):
        """h (..., d_in) -> logit_pi (..., K), mu (..., K, d), sigma (..., K, d)."""
        K, D = self.n_comp, self.d_emb
        x = F.silu(self.fc2(F.silu(self.fc1(h))))
        o = self.out(x)
        logit_pi = o[..., :K]
        mu = o[..., K : K + K * D].view(*o.shape[:-1], K, D)
        sigma = F.elu(o[..., K + K * D :].view(*o.shape[:-1], K, D)) + 1.0 + self.sigma_floor
        return logit_pi, mu, sigma

    def nll(self, h: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Per-position NLL of target (..., d) under the GMM, fp32. -> (...,)."""
        logit_pi, mu, sigma = (t.float() for t in self.params(h))
        target = target.float()
        D = target.shape[-1]
        z = (target.unsqueeze(-2) - mu) / sigma  # (..., K, D)
        log_n = -0.5 * z.pow(2).sum(-1) - sigma.log().sum(-1) - 0.5 * D * math.log(2 * math.pi)
        return -torch.logsumexp(F.log_softmax(logit_pi, dim=-1) + log_n, dim=-1)

    @torch.no_grad()
    def sample(self, h: torch.Tensor, tau: float = 1.0) -> torch.Tensor:
        """h (..., d_in) -> mu of one component, (..., d). tau<=0 -> argmax pi."""
        logit_pi, mu, _sigma = self.params(h)
        if tau <= 0:
            k = logit_pi.argmax(dim=-1)
        else:
            k = torch.multinomial(F.softmax(logit_pi / tau, dim=-1).reshape(-1, self.n_comp), 1)
            k = k.view(h.shape[:-1])
        return mu.gather(-2, k.unsqueeze(-1).unsqueeze(-1).expand(*k.shape, 1, self.d_emb)).squeeze(-2)

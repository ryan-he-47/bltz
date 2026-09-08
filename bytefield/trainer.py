"""Lean training loop: AdamW + WSD + bf16 autocast + spike guard + checkpoints.

House rules (docs/AGENTS.md): betas (0.9, 0.95), eps 1e-8, wd 0.1, clip 1.0,
spike guard = skip step when grad norm > max(10x running median, 2000)
(min 50 samples before the guard arms). WSD schedule by default.
"""
from __future__ import annotations

import json
import os
import time
from collections import deque
from collections.abc import Callable
from typing import Any

import torch

from .objectives import mtp_loss


class WSD:
    """warmup -> stable -> linear decay to final_frac * peak over last decay_frac."""

    def __init__(self, peak, warmup, steps, decay_frac=0.4, final_frac=0.1):
        self.peak, self.warmup, self.steps = peak, warmup, steps
        self.decay_frac, self.final_frac = decay_frac, final_frac

    def __call__(self, step: int) -> float:
        if step < self.warmup:
            return self.peak * (step + 1) / self.warmup
        decay_start = self.steps * (1.0 - self.decay_frac)
        if step < decay_start:
            return self.peak
        t = min((step - decay_start) / max(self.steps - decay_start, 1), 1.0)
        floor = self.peak * self.final_frac
        return self.peak + (floor - self.peak) * t


def grad_norm(model: torch.nn.Module) -> float:
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += p.grad.detach().float().pow(2).sum().item()
    return total**0.5


def train(
    model: torch.nn.Module,
    batch_fn: Callable[[], dict[str, torch.Tensor]],
    cfg,
    device: str = "cuda",
) -> list[dict[str, Any]]:
    """batch_fn() -> batch dict (CPU tensors). Returns the log history."""
    tcfg = cfg.train
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=tcfg.lr,
        betas=tuple(tcfg.betas),
        eps=tcfg.eps,
        weight_decay=tcfg.wd,
    )
    sched = WSD(tcfg.lr, tcfg.warmup, tcfg.steps, tcfg.decay_frac, tcfg.final_frac)
    os.makedirs(tcfg.ckpt_dir, exist_ok=True)
    log_path = os.path.join(tcfg.ckpt_dir, "train.log")

    history: list[dict[str, Any]] = []
    med_hist: deque[float] = deque(maxlen=1000)
    best = float("inf")
    last_loss = float("nan")
    skips = 0
    t0 = time.time()

    def log(rec: dict[str, Any]) -> None:
        history.append(rec)
        line = json.dumps(rec, ensure_ascii=False)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
        print(line, flush=True)

    for step in range(tcfg.steps):
        batch = {k: v.to(device) for k, v in batch_fn().items()}
        for g in opt.param_groups:
            g["lr"] = sched(step)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=tcfg.bf16):
            loss = mtp_loss(model, batch, cfg)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = grad_norm(model)

        median = sorted(med_hist)[len(med_hist) // 2] if len(med_hist) >= 50 else None
        thresh = max(tcfg.spike_skip * median, 2000.0) if median is not None else float("inf")
        if gn > thresh:
            skips += 1
            log({"step": step, "event": "spike_skip", "gn": round(gn, 1), "thresh": round(thresh, 1)})
            continue
        med_hist.append(gn)
        torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.clip)
        opt.step()

        lval = float(loss.item())
        last_loss = lval
        if lval < best:
            best = lval
            torch.save(
                {"model": model.state_dict(), "cfg": cfg.to_dict(), "step": step, "loss": lval},
                os.path.join(tcfg.ckpt_dir, "best.pt"),
            )
        if step % tcfg.log_every == 0 or step == tcfg.steps - 1:
            log({
                "step": step,
                "loss": round(lval, 4),
                "gn": round(gn, 1),
                "lr": f"{sched(step):.2e}",
                "skips": skips,
                "sec": round(time.time() - t0, 1),
            })

    torch.save(
        {"model": model.state_dict(), "cfg": cfg.to_dict(), "step": tcfg.steps, "loss": last_loss},
        os.path.join(tcfg.ckpt_dir, "last.pt"),
    )
    return history

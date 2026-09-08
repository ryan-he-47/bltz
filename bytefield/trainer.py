"""Lean training loop: AdamW + WSD + bf16 autocast + spike guard + checkpoints.

House rules (docs/AGENTS.md): betas (0.9, 0.95), eps 1e-8, wd 0.1, clip 1.0,
spike guard = skip step when grad norm > max(10x running median, 2000)
(min 50 samples before the guard arms). WSD schedule by default.
"""
from __future__ import annotations

import json
import math
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

    # precision: bf16 (sm89+, no scaler) | fp16 + GradScaler (V100 path, MoB
    # house rule — sm70 has no bf16). train.bf16=false selects fp16.
    use_bf16 = bool(tcfg.bf16)
    scaler = torch.cuda.amp.GradScaler(enabled=not use_bf16)

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
        with torch.autocast(
            "cuda", dtype=torch.bfloat16 if use_bf16 else torch.float16
        ):
            loss = mtp_loss(model, batch, cfg)
        opt.zero_grad(set_to_none=True)
        if use_bf16:
            loss.backward()
        else:
            scaler.scale(loss).backward()
            scaler.unscale_(opt)  # guard/clip must see UNSCALED grad norms
        gn = grad_norm(model)

        median = sorted(med_hist)[len(med_hist) // 2] if len(med_hist) >= 50 else None
        thresh = max(tcfg.spike_skip * median, 2000.0) if median is not None else float("inf")
        if not math.isfinite(gn) or gn > thresh:
            skips += 1
            log({"step": step, "event": "spike_skip", "gn": round(gn, 1), "thresh": round(thresh, 1)})
            if not use_bf16:
                scaler.update()  # keep the scale fresh on skipped steps
            continue
        med_hist.append(gn)
        torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.clip)
        if use_bf16:
            opt.step()
        else:
            scaler.step(opt)
            scaler.update()

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

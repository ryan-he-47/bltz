"""Lean training loop: AdamW + WSD + bf16 autocast + spike guard + checkpoints
+ graceful interrupt (STOP file / SIGINT-SIGTERM flag, second Ctrl+C = hard bail).

House rules (docs/AGENTS.md): betas (0.9, 0.95), eps 1e-8, wd 0.1, clip 1.0,
spike guard = skip step when grad norm > max(10x running median, 2000)
(min 50 samples before the guard arms). WSD schedule by default.
"""
from __future__ import annotations

import json
import math
import os
import signal
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
    loss_fn: Callable[[torch.nn.Module, dict[str, torch.Tensor], Any], torch.Tensor] | None = None,
) -> list[dict[str, Any]]:
    """batch_fn() -> batch dict (CPU tensors). Returns the log history.

    loss_fn(model, batch, cfg) -> scalar loss; defaults to mtp_loss (the
    ByteField objective). The token baseline passes a wrapper around
    next_token_loss so both arms share this exact loop (optimizer, schedule,
    spike guard, precision paths)."""
    if loss_fn is None:
        loss_fn = mtp_loss
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
    scaler = torch.amp.GradScaler("cuda", enabled=not use_bf16)

    history: list[dict[str, Any]] = []
    med_hist: deque[float] = deque(maxlen=1000)
    best = float("inf")
    last_loss = float("nan")
    skips = 0
    t0 = time.time()

    # ---- breakpoint resume (--set train.resume=<ckpt_full.pt>) ----
    # Full state: model+optimizer+schedule-relevant counters+RNG. Data sampling
    # order is NOT captured (ShardReader sampling is random with replacement —
    # approximation, documented in docs/07).
    start_step = 0
    resume = str(tcfg.get("resume", "") or "")
    if resume:
        # load to CPU: RNG state must be a CPU ByteTensor for set_rng_state;
        # model/optimizer states are placed on the model's device by load_state_dict.
        state = torch.load(resume, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["opt"])
        start_step = int(state["step"]) + 1
        med_hist = deque(state.get("med_hist", []), maxlen=1000)
        best = float(state.get("best", "inf"))
        skips = int(state.get("skips", 0))
        torch.set_rng_state(state["rng_cpu"])
        torch.cuda.set_rng_state(state["rng_cuda"])
        if not use_bf16 and "scaler" in state:
            scaler.load_state_dict(state["scaler"])
        print(f"resumed from {resume} at step {start_step}", flush=True)
    ckpt_every = int(tcfg.get("ckpt_every", 1000))
    ckpt_keep = int(tcfg.get("ckpt_keep", 2))

    # ---- graceful interrupt (local runs: free the GPU on demand) ----
    # Two channels, one save-and-stop path; the check sits at loop top so CUDA
    # is never mid-kernel when state is captured:
    #   1. SIGINT/SIGTERM (Ctrl+C in a console run) — the handler only sets a
    #      flag (saving inside a signal handler with CUDA in flight is unsafe);
    #      a second Ctrl+C raises KeyboardInterrupt for a harder bail-out,
    #      still caught below for a best-effort save.
    #   2. STOP file at <ckpt_dir>/STOP — the ONLY channel that reaches a
    #      detached/hidden process (Start-Process):
    #      `New-Item <ckpt_dir>\STOP -ItemType File`. Consumed on use so a
    #      later resume does not instantly stop again.
    stop_path = os.path.join(tcfg.ckpt_dir, "STOP")
    interrupt: dict[str, Any] = {"sig": None}

    def _on_signal(signum, frame) -> None:
        if interrupt["sig"] is not None:
            raise KeyboardInterrupt  # second signal: hard bail, saved below
        interrupt["sig"] = int(signum)

    prev_handlers: dict[Any, Any] = {}
    for sig_ in (signal.SIGINT, signal.SIGTERM):
        try:
            prev_handlers[sig_] = signal.signal(sig_, _on_signal)
        except (ValueError, OSError, RuntimeError):
            pass  # not the main thread / unsupported signal — STOP file still works

    def _restore_handlers() -> None:
        for sig_, h_ in prev_handlers.items():
            try:
                signal.signal(sig_, h_)
            except (ValueError, OSError, RuntimeError):
                pass

    def stop_requested() -> str | None:
        if interrupt["sig"] is not None:
            return f"signal {interrupt['sig']}"
        if os.path.exists(stop_path):
            return "STOP file"
        return None

    def save_full(step: int) -> None:
        # rotate: ckpt_full.pt -> ckpt_full.pt.1 (one previous full state kept,
        # so a post-collapse checkpoint is not the only recovery point)
        path = os.path.join(tcfg.ckpt_dir, "ckpt_full.pt")
        if ckpt_keep > 1 and os.path.exists(path):
            bak = path + ".1"
            if os.path.exists(bak):
                os.remove(bak)
            os.replace(path, bak)
        torch.save(
            {
                "model": model.state_dict(),
                "opt": opt.state_dict(),
                "step": step,
                "med_hist": list(med_hist),
                "best": best,
                "skips": skips,
                "rng_cpu": torch.get_rng_state(),
                "rng_cuda": torch.cuda.get_rng_state(),
                "scaler": scaler.state_dict(),
                "cfg": cfg.to_dict(),
            },
            path,
        )

    def log(rec: dict[str, Any]) -> None:
        history.append(rec)
        line = json.dumps(rec, ensure_ascii=False)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
        print(line, flush=True)

    step = start_step - 1  # stays bound for the interrupt paths even pre-loop
    stopped: str | None = None
    try:
        for step in range(start_step, tcfg.steps):
            why = stop_requested()
            if why is not None:
                stopped = why
                break
            batch = {k: v.to(device) for k, v in batch_fn().items()}
            for g in opt.param_groups:
                g["lr"] = sched(step)
            with torch.autocast(
                "cuda", dtype=torch.bfloat16 if use_bf16 else torch.float16
            ):
                loss = loss_fn(model, batch, cfg)
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
            if ckpt_every and (step + 1) % ckpt_every == 0:
                save_full(step)
    except KeyboardInterrupt:
        stopped = "KeyboardInterrupt(hard)"
    finally:
        _restore_handlers()

    if stopped is not None:
        # model/opt are always a consistent pair here: mid-step state is either
        # both pre-step or both post-step, and we label with the last step whose
        # update is guaranteed fully applied -> resume at most duplicates 1 step.
        if step > start_step:
            save_full(step - 1)
        if os.path.exists(stop_path):
            try:
                os.remove(stop_path)  # consume: a resume must not stop instantly
            except OSError:
                pass
        log({
            "step": max(step, start_step),
            "event": "interrupt_stop",
            "reason": stopped,
            "saved_step": step - 1 if step > start_step else None,
        })
        return history

    save_full(tcfg.steps - 1)
    torch.save(
        {"model": model.state_dict(), "cfg": cfg.to_dict(), "step": tcfg.steps, "loss": last_loss},
        os.path.join(tcfg.ckpt_dir, "last.pt"),
    )
    return history

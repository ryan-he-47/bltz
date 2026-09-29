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


class CosSched:
    """warmup linear ramp -> cosine decay peak..floor (2026-09-29, AE 5B 配方)."""

    def __init__(self, peak: float, warmup: int, steps: int, floor: float):
        self.peak, self.warmup, self.steps, self.floor = peak, warmup, steps, floor

    def __call__(self, step: int) -> float:
        if step < self.warmup:
            return self.peak * (step + 1) / self.warmup
        t = min(1.0, (step - self.warmup) / max(1, self.steps - self.warmup))
        return self.floor + 0.5 * (self.peak - self.floor) * (1.0 + math.cos(math.pi * t))


def grad_norm(model: torch.nn.Module) -> float:
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    if not grads:
        return 0.0
    # foreach single-sync (2026-09-29 prof finding: ~200 .item() roundtrips/step
    # on slow server cores cost real wall time)
    return float(torch.norm(torch.stack(torch._foreach_norm(grads))))


def _state_model(m: torch.nn.Module) -> torch.nn.Module:
    """Strip DDP / loss-wrapper layers -> the module whose state_dict matches
    the on-disk ckpt format (single-GPU runs: identity)."""
    m = getattr(m, "module", m)
    return getattr(m, "core", m)


def train(
    model: torch.nn.Module,
    batch_fn: Callable[[], dict[str, torch.Tensor]],
    cfg,
    device: str = "cuda",
    loss_fn: Callable[[torch.nn.Module, dict[str, torch.Tensor], Any], torch.Tensor] | None = None,
    is_rank0: bool = True,
) -> list[dict[str, Any]]:
    """batch_fn() -> batch dict (CPU tensors). Returns the log history.

    loss_fn(model, batch, cfg) -> scalar loss; defaults to mtp_loss (the
    bltz objective). The token baseline passes a wrapper around
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
    # fp16_init_scale (2026-09-30 可学习臂事故): 默认 65536 在锐化的密度模型
    # 上反向必溢出(gn=Infinity 级联);稳定水位实测 ~1024,正式 run 显式压低。
    use_bf16 = bool(tcfg.bf16)
    scaler = torch.amp.GradScaler(
        "cuda", enabled=not use_bf16,
        init_scale=float(tcfg.get("fp16_init_scale", 65536.0)),
        growth_interval=int(tcfg.get("fp16_growth_interval", 2000)),
    )

    history: list[dict[str, Any]] = []
    med_hist: deque[float] = deque(maxlen=1000)
    best = float("inf")
    pending_best = False
    last_loss = float("nan")
    skips = 0
    consec_skips = 0
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
        _state_model(model).load_state_dict(state["model"])
        start_step = int(state["step"]) + 1
        if "opt" in state:
            opt.load_state_dict(state["opt"])
            med_hist = deque(state.get("med_hist", []), maxlen=1000)
            best = float(state.get("best", "inf"))
            skips = int(state.get("skips", 0))
            torch.set_rng_state(state["rng_cpu"])
            torch.cuda.set_rng_state(state["rng_cuda"])
            if not use_bf16 and "scaler" in state:
                scaler.load_state_dict(state["scaler"])
        else:
            # weight-only milestone: fresh optimizer/scaler/RNG (house rule:
            # branch restart peak LR must not exceed the parent run's end LR)
            print("note: weight-only resume (milestone), optimizer reset", flush=True)
        print(f"resumed from {resume} at step {start_step}", flush=True)
    ckpt_every = int(tcfg.get("ckpt_every", 1000))
    ckpt_keep = int(tcfg.get("ckpt_keep", 2))
    # WSD decay-on-demand 里程碑:每 milestone_every 步永久保留一份全状态
    # (0=关)。rotation 只管崩溃恢复;里程碑是实验分支点(2026-09-13 用户
    # 指正:只留尾段两个 ckpt 等于废了 WSD 的稳定段分支能力)。
    milestone_every = int(tcfg.get("milestone_every", 0))

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

    def save_full(
        step: int, name: str = "ckpt_full.pt", rotate: bool = True, state_only: bool = False
    ) -> None:
        if not is_rank0:  # DDP: only rank 0 writes (concurrent torch.save corrupts)
            return
        # rotate: shift ckpt_full.pt -> .1 -> .2 ... keeping ckpt_keep copies
        # (rolling window, 2026-09-27 用户拍板: scale run 只留最近 3 份,
        # disk quota 撑不住永久里程碑堆叠).
        path = os.path.join(tcfg.ckpt_dir, name)
        if rotate and ckpt_keep > 1 and os.path.exists(path):
            oldest = f"{path}.{ckpt_keep - 1}"
            if os.path.exists(oldest):
                os.remove(oldest)
            for i in range(ckpt_keep - 2, -1, -1):
                src = path if i == 0 else f"{path}.{i}"
                if os.path.exists(src):
                    os.replace(src, f"{path}.{i + 1}")
        payload: dict[str, Any] = {"model": _state_model(model).state_dict(), "cfg": cfg.to_dict(), "step": step}
        if not state_only:
            payload.update(
                {
                    "opt": opt.state_dict(),
                    "med_hist": list(med_hist),
                    "best": best,
                    "skips": skips,
                    "rng_cpu": torch.get_rng_state(),
                    "rng_cuda": torch.cuda.get_rng_state(),
                    "scaler": scaler.state_dict(),
                }
            )
        torch.save(payload, path)

    def log(rec: dict[str, Any]) -> None:
        if not is_rank0:
            return
        history.append(rec)
        line = json.dumps(rec, ensure_ascii=False)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
        print(line, flush=True)

    step = start_step - 1  # stays bound for the interrupt paths even pre-loop
    stopped: str | None = None
    # resume re-warmup (train.rewarmup, 2026-09-30 事故修复): weight-only
    # resume resets Adam; fresh moments + full LR = destructive aggression.
    # Ramp LR linearly over the first rewarmup steps from the resume point
    # (peak still = sched(step) -> house rule peak<=parent end LR holds).
    rewarm = int(tcfg.get("rewarmup", 0))
    # phase profiling (BLTZ_PROF=1, default off — zero behavior change):
    # per-step wall time split batch/fwd/bwd/gn/opt/io, printed every log_every.
    prof = os.environ.get("BLTZ_PROF", "") == "1"
    pt = {"batch": 0.0, "fwd": 0.0, "bwd": 0.0, "gn": 0.0, "opt": 0.0, "io": 0.0}
    pn = 0
    try:
        for step in range(start_step, tcfg.steps):
            why = stop_requested()
            if why is not None:
                stopped = why
                break
            _t = time.time() if prof else 0.0
            batch = {k: v.to(device) for k, v in batch_fn().items()}
            if prof:
                pt["batch"] += time.time() - _t; _t = time.time()
            for g in opt.param_groups:
                g["lr"] = sched(step)
                if rewarm and step - start_step < rewarm:
                    g["lr"] = min(g["lr"], float(tcfg.lr)
                                        * (step - start_step + 1) / rewarm)
            with torch.autocast(
                "cuda", dtype=torch.bfloat16 if use_bf16 else torch.float16
            ):
                loss = loss_fn(model, batch, cfg)
            if prof:
                pt["fwd"] += time.time() - _t; _t = time.time()
            opt.zero_grad(set_to_none=True)
            if use_bf16:
                loss.backward()
            else:
                scaler.scale(loss).backward()
                scaler.unscale_(opt)  # guard/clip must see UNSCALED grad norms
            if prof:
                torch.cuda.synchronize()
                pt["bwd"] += time.time() - _t; _t = time.time()
            gn = grad_norm(model)
            if prof:
                pt["gn"] += time.time() - _t; _t = time.time()

            median = sorted(med_hist)[len(med_hist) // 2] if len(med_hist) >= 50 else None
            thresh = max(tcfg.spike_skip * median, 2000.0) if median is not None else float("inf")
            if not math.isfinite(gn) or gn > thresh:
                skips += 1
                consec_skips += 1
                log({"step": step, "event": "spike_skip", "gn": round(gn, 1), "thresh": round(thresh, 1)})
                if not use_bf16:
                    scaler.update()  # keep the scale fresh on skipped steps
                if consec_skips >= 500:
                    # stall watchdog (589611 事故: 全 NaN 步烧了 8800 步才被人工
                    # 停下 — 500 连跳即视为停滞,保存后退出等人工分析)
                    log({"step": step, "event": "stall_watchdog", "consec": consec_skips})
                    stopped = f"stall_watchdog({consec_skips} consecutive skips)"
                    break
                continue
            consec_skips = 0
            med_hist.append(gn)
            torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.clip)
            if use_bf16:
                opt.step()
            else:
                scaler.step(opt)
                scaler.update()
            if prof:
                torch.cuda.synchronize()
                pt["opt"] += time.time() - _t; _t = time.time()

            lval = float(loss.item())
            last_loss = lval
            if lval < best:
                best = lval
                pending_best = True  # flushed at log boundary (see below)
            if prof:
                pt["io"] += time.time() - _t
                pn += 1
            if step % tcfg.log_every == 0 or step == tcfg.steps - 1:
                # best.pt flush throttled to log boundaries (2026-09-29 prof
                # finding: saving 0.5GB to GPFS on EVERY improving step cost
                # 0.34-0.56s/step in the early phase — best is an artifact,
                # 50-step granularity loses nothing)
                if pending_best and is_rank0:
                    torch.save(
                        {"model": _state_model(model).state_dict(), "cfg": cfg.to_dict(), "step": step, "loss": best},
                        os.path.join(tcfg.ckpt_dir, "best.pt"),
                    )
                    pending_best = False
                if prof and pn:
                    tot = sum(pt.values())
                    print(f"[prof] step {step}: batch {pt['batch']/pn:.3f} "
                          f"fwd {pt['fwd']/pn:.3f} bwd {pt['bwd']/pn:.3f} "
                          f"gn {pt['gn']/pn:.3f} opt {pt['opt']/pn:.3f} "
                          f"io {pt['io']/pn:.3f} | sum {tot/pn:.3f}s/step",
                          flush=True)
                    for k in pt:
                        pt[k] = 0.0
                    pn = 0
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
            if milestone_every and (step + 1) % milestone_every == 0:
                save_full(step, name=f"ckpt_s{step + 1:07d}.pt", rotate=False, state_only=True)
    except KeyboardInterrupt:
        stopped = "KeyboardInterrupt(hard)"
    finally:
        _restore_handlers()

    if stopped is not None:
        # model/opt are always a consistent pair here: mid-step state is either
        # both pre-step or both post-step, and we label with the last step whose
        # update is guaranteed fully applied -> resume at most duplicates 1 step.
        if pending_best and is_rank0:
            torch.save(
                {"model": _state_model(model).state_dict(), "cfg": cfg.to_dict(), "step": max(step, start_step), "loss": best},
                os.path.join(tcfg.ckpt_dir, "best.pt"),
            )
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

    if pending_best and is_rank0:
        torch.save(
            {"model": _state_model(model).state_dict(), "cfg": cfg.to_dict(), "step": tcfg.steps - 1, "loss": best},
            os.path.join(tcfg.ckpt_dir, "best.pt"),
        )
    save_full(tcfg.steps - 1)
    torch.save(
        {"model": model.state_dict(), "cfg": cfg.to_dict(), "step": tcfg.steps, "loss": last_loss},
        os.path.join(tcfg.ckpt_dir, "last.pt"),
    )
    return history

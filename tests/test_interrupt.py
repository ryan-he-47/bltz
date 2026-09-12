"""Script-style test for graceful interrupt: STOP file and SIGINT both save a
full checkpoint and stop cleanly (no progress loss beyond the current step);
a hard KeyboardInterrupt still gets a best-effort save; resume from the
interrupt checkpoint continues from the right step.
Run: python tests/test_interrupt.py"""
from __future__ import annotations

import shutil
import signal
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from bltz.config import load
from bltz.data import build_batch
from bltz.models import BltzLM
from bltz.segment import Segmenter
from bltz.trainer import train

TEXT = (
    "The Byte Latent Transformer encodes bytes into dynamically sized patches, "
    "allocating more compute where data complexity demands it. " * 8
)
TMP = Path("checkpoints/test_interrupt_tmp")


def check(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def fresh():
    """New smoke run dir + model; ckpt_every=0 so only interrupt saves fire."""
    shutil.rmtree(TMP, ignore_errors=True)
    cfg = load("configs/smoke.yaml")
    seg = Segmenter(cfg.segment.l_max, 0.0, 0.0)
    batch = build_batch([TEXT], seg, cfg.data.n_patches, cfg.segment.l_max)
    batch = {k: v[:2] for k, v in batch.items()}
    torch.manual_seed(cfg.train.seed)
    model = BltzLM(cfg).cuda()
    cfg.train.steps = 100
    cfg.train.ckpt_dir = str(TMP)
    cfg.train.log_every = 1000  # only step 0 / final step log
    cfg.train.ckpt_every = 0    # isolate interrupt saves from periodic saves
    return cfg, batch, model


def saved_step() -> int:
    state = torch.load(TMP / "ckpt_full.pt", map_location="cpu", weights_only=False)
    return int(state["step"])


def main() -> None:
    # ---- Phase A: STOP file (touched from batch_fn, simulating the async user
    # action against a detached process) -> detected at next loop top ----------
    cfg, batch, model = fresh()
    calls = {"n": 0}

    def batch_fn_stop():
        calls["n"] += 1
        if calls["n"] == 4:  # 4th fetch = step 3's batch; STOP seen at step 4 top
            (TMP / "STOP").touch()
        return batch

    h = train(model, batch_fn_stop, cfg)
    events = [r for r in h if r.get("event") == "interrupt_stop"]
    check(len(events) == 1, f"no interrupt_stop event: {h}")
    check(events[0]["reason"] == "STOP file", events[0])
    check((TMP / "ckpt_full.pt").exists(), "ckpt_full.pt not written on STOP")
    check(saved_step() == 3, f"saved step {saved_step()} != 3")
    check(events[0]["saved_step"] == 3, events[0])
    check(not (TMP / "STOP").exists(), "STOP file not consumed")
    check(not (TMP / "last.pt").exists(), "last.pt written despite interrupt")

    # ---- Phase B: resume from the interrupt checkpoint continues from step 4 --
    torch.manual_seed(999)  # different init: proves the state actually loads
    model2 = BltzLM(cfg).cuda()
    cfg.train.steps = 8
    cfg.train.resume = str(TMP / "ckpt_full.pt")
    h2 = train(model2, lambda: batch, cfg)
    steps2 = [r["step"] for r in h2 if "loss" in r]
    check(steps2 and steps2[0] >= 4, f"resume after STOP restarted: {steps2}")
    check((TMP / "last.pt").exists(), "last.pt missing after resumed completion")

    # ---- Phase C: SIGINT via raise_signal -> flag path -> graceful save -------
    cfg, batch, model = fresh()
    calls = {"n": 0}

    def batch_fn_sig():
        calls["n"] += 1
        if calls["n"] == 3:  # step 2's fetch; flag seen at step 3 top
            signal.raise_signal(signal.SIGINT)
        return batch

    h3 = train(model, batch_fn_sig, cfg)
    ev3 = [r for r in h3 if r.get("event") == "interrupt_stop"]
    check(len(ev3) == 1 and "signal" in ev3[0]["reason"], h3)
    check(saved_step() == 2, f"SIGINT saved step {saved_step()} != 2")
    check(
        signal.getsignal(signal.SIGINT) is signal.default_int_handler,
        "SIGINT handler leaked past train()",
    )

    # ---- Phase D: hard KeyboardInterrupt (second Ctrl+C semantics) -> save ----
    cfg, batch, model = fresh()
    calls = {"n": 0}

    def batch_fn_hard():
        calls["n"] += 1
        if calls["n"] == 3:  # during step 2's fetch, before any step-2 compute
            raise KeyboardInterrupt
        return batch

    h4 = train(model, batch_fn_hard, cfg)
    ev4 = [r for r in h4 if r.get("event") == "interrupt_stop"]
    check(len(ev4) == 1 and "KeyboardInterrupt" in ev4[0]["reason"], h4)
    check(saved_step() == 1, f"hard-interrupt saved step {saved_step()} != 1")

    shutil.rmtree(TMP, ignore_errors=True)
    print("test_interrupt.py: ALL PASS")


if __name__ == "__main__":
    main()

"""Plot the main-run training log for eyeball inspection (POC: trend reading).
Warmup steps cut so the random-init loss doesn't squash the y-axis.
Run: python viz/plot_train.py [logpath] [out.png]"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

LOG = sys.argv[1] if len(sys.argv) > 1 else r"C:\Users\he\AppData\Local\Temp\opencode\train_main_final.log"
OUT = sys.argv[2] if len(sys.argv) > 2 else "viz/main_544172.png"
WARMUP = 1000
ROLL = 10  # records (50-step cadence -> 500-step window)

steps, loss, gn, lr = [], [], [], []
with open(LOG, encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        if "loss" not in r:
            continue
        steps.append(r["step"])
        loss.append(r["loss"])
        gn.append(r["gn"])
        lr.append(float(r["lr"]))
steps = np.array(steps)
loss = np.array(loss)
gn = np.array(gn)
lr = np.array(lr)

m = steps >= WARMUP
steps, loss, gn, lr = steps[m], loss[m], gn[m], lr[m]

def roll(x: np.ndarray) -> np.ndarray:
    return np.convolve(x, np.ones(ROLL) / ROLL, mode="valid")

rsteps = steps[ROLL - 1 :]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

ax1.plot(steps, loss, color="tab:blue", alpha=0.25, lw=0.8, label="loss (raw, 50-step)")
ax1.plot(rsteps, roll(loss), color="tab:blue", lw=2, label=f"loss (rolling {ROLL*50} steps)")
ax1.axvline(12000, color="gray", ls=":", lw=1)
ax1.text(12100, ax1.get_ylim()[0], "decay start", color="gray", fontsize=8, rotation=90, va="bottom")
ax1.set_ylabel("MTP mixture loss")
ax1.legend(loc="upper right")
ax1.grid(alpha=0.3)

ax1b = ax1.twinx()
ax1b.plot(steps, lr, color="tab:red", ls="--", lw=1, alpha=0.7)
ax1b.set_ylabel("LR", color="tab:red")
ax1b.tick_params(axis="y", labelcolor="tab:red")

ax2.plot(steps, gn, color="tab:orange", alpha=0.3, lw=0.8, label="grad norm (raw)")
ax2.plot(rsteps, roll(gn), color="tab:orange", lw=2, label=f"grad norm (rolling {ROLL*50})")
ax2.set_xlabel("step")
ax2.set_ylabel("grad norm")
ax2.set_yscale("log")
ax2.legend(loc="upper right")
ax2.grid(alpha=0.3)

fig.suptitle("bltz main run 544172 (batch16, fp16, LR 4e-4, WSD 20k) — warmup cut")
fig.tight_layout()
Path(OUT).parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=150)
print(f"saved {OUT}")
print(f"loss range shown: {loss.min():.4f} .. {loss.max():.4f}")

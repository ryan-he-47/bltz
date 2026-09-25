"""Six-arm TinyStories head-ablation curves (one figure, rolling-smoothed).
Run: python scripts/plot_ts_arms.py"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

TMP = Path(r"C:\Users\he\AppData\Local\Temp\opencode")
ARMS = ["ts_base", "ts_wide", "ts_swiglu", "ts_shallow", "ts_linear", "ts_k16"]
LABEL = {
    "ts_base": "base (1024x2)",
    "ts_wide": "wide (6208x2)",
    "ts_swiglu": "swiglu (6208x2)",
    "ts_shallow": "shallow (1024x1)",
    "ts_linear": "linear (0 layer)",
    "ts_k16": "k16 (K=16)",
}
ROLL = 20  # 50-step cadence -> 1000-step window

fig, ax = plt.subplots(figsize=(11, 6), dpi=140)
for arm in ARMS:
    steps, losses = [], []
    with open(TMP / f"{arm}_train.log", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if "loss" in r:
                steps.append(r["step"])
                losses.append(r["loss"])
    s = steps[ROLL // 2 :]
    sm = [sum(losses[i - ROLL : i]) / ROLL for i in range(ROLL, len(losses) + 1)]
    ax.plot(steps[: len(sm)], sm, label=f"{LABEL[arm]}  (final {losses[-1]:.2f})",
            linewidth=1.6)
ax.set_xlabel("step")
ax.set_ylabel("MDN NLL (same AE, cross-arm comparable)")
ax.set_title("TinyStories MDN-head ablation — six arms, 40k steps")
ax.grid(alpha=0.3)
ax.legend(fontsize=9)
out = Path("viz/ts_arms.png")
fig.tight_layout()
fig.savefig(out)
print("saved", out)

"""CPT S 档 vs p1/p2 loss 曲线叠加(2026-10-23 用户要求: 当前段和前段画在一起)。

绝对步坐标: p1 [0,25431] -> p2 [20004,50863](同数据分支) -> CPT [45429, ...]
(新数据 DCLM+Edu)。p2 在 CPT 分支点之后的旧尾巴(45429->50863)用灰色虚线
保留,便于对比"如果继续旧数据会怎样"。

  python -u scripts/plot_cpt_vs_p12.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "viz" / "word_cpt_vs_p12_curve.png"
ROLL = 10  # 50-step cadence -> 500-step rolling window


def load(path: Path) -> tuple[np.ndarray, np.ndarray]:
    steps, loss = [], []
    for line in path.open(encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if "loss" in r:
            steps.append(r["step"])
            loss.append(r["loss"])
    return np.array(steps), np.array(loss)


def roll(x: np.ndarray) -> np.ndarray:
    return np.convolve(x, np.ones(ROLL) / ROLL, mode="valid")


def plot_run(ax, steps, loss, color, label, raw_alpha=0.18):
    ax.plot(steps, loss, color=color, alpha=raw_alpha, lw=0.6)
    ax.plot(steps[ROLL - 1:], roll(loss), color=color, lw=1.8, label=label)


p1s, p1l = load(ROOT / "checkpoints" / "word_05b_p1_train.log")
p2s, p2l = load(ROOT / "checkpoints" / "word_05b_p2_train.log")
cs, cl = load(ROOT / "checkpoints" / "word_cpt_s_stable_train.log")

fig, ax = plt.subplots(figsize=(13, 6.2))
plot_run(ax, p1s, p1l, "tab:cyan", "p1 (FineWeb 5B units, 0-25.4k)")
plot_run(ax, p2s, p2l, "tab:blue", "p2 (FineWeb to 10B units, 20k-50.9k)")
# p2 旧尾巴(CPT 分支点之后)灰色虚线参考
m = p2s >= 45429
ax.plot(p2s[m][ROLL - 1:], roll(p2l[m]), color="gray", ls="--", lw=1.2,
        label="p2 tail on OLD data (45429-50863, superseded)")
ax.plot(p2s[m], p2l[m], color="gray", alpha=0.12, lw=0.6)
plot_run(ax, cs, cl, "tab:red", "CPT S (NEW data: DCLM 60% + Edu 40%, 45429-)")

ax.axvline(20004, color="gray", ls=":", lw=1)
ax.text(20100, -22.6, "p1→p2  branch (pre_decay)", color="gray", fontsize=8, rotation=90, va="bottom")
ax.axvline(45429, color="tab:orange", ls=":", lw=1.4)
ax.text(45550, -22.6, "CPT data switch (new corpus, zero replay)", color="tab:orange", fontsize=8, rotation=90, va="bottom")

ax.set_xlim(0, cs[-1] + 600)
ws = 2000
yl = np.concatenate([p1l[p1s >= ws], p2l[p2s >= ws], cl[cs >= ws]])
ax.set_ylim(yl.min() - 0.8, yl.max() + 1.2)
ax.set_xlabel("absolute step (p2/CPT resume continue the global counter)")
ax.set_ylabel("MTP mixture loss (rolling 500 steps)")
ax.grid(alpha=0.3)
ax.legend(loc="upper right", fontsize=8)
n_new = int(cs[-1] - 45429)
fig.suptitle(f"bltz word-arm 0.5B — p1/p2 vs CPT S-tier; CPT at step {cs[-1]} "
             f"(+{n_new} new-data steps, {n_new*196608/1e9:.1f}B units)")
fig.tight_layout()
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=150)
print(f"saved {OUT}")
print(f"p1 last {p1l[-1]:.3f} | p2 tail old-data end {p2l[-1]:.3f} | CPT now {cl[-1]:.3f} at step {cs[-1]}")
print(f"CPT loss min {cl.min():.2f} max {cl.max():.2f}")
